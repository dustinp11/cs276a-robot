"""Waypoint tracking and AprilTag navigation with modular move_forward and rotate.

Refactored to provide two core behaviors:
1. move_forward(to_coord, from_coord, target_tag):
   - Moves forward halfway to to_coord via dead reckoning.
   - Halts and settles 1 second to recalibrate heading and localized distance.
   - Moves forward the remaining distance to to_coord.
   - Halts and adjusts position based on localized distance from target_tag
     (moving forward if short, backward if overshoot) until within tolerance.

2. rotate(target_tag):
   - Rotates CCW in 30-degree increments until target AprilTag is in vision.
   - Performs fine microadjustments (with 1s settling between turns) until directly facing it.
   - If rotating to AprilTag 3, prints "facing apriltag3" and pauses 3 seconds.

The run() function executes the complete multi-waypoint sequence:
- Leg 1: move_forward to (1, 0) from (0, 0) towards Tag 0
- Turn: rotate towards Tag 1
- Leg 2: move_forward to (1, 2) from (1, 0) towards Tag 1
- Turn: rotate towards Tag 3
- Turn: rotate towards Tag 5
- Leg 3: move_forward to (0, 0) from (1, 2) towards Tag 5
- Turn: rotate towards Tag 0
- Concludes by printing "end".

Robot coordinates are logged to coordinate.txt every second.
"""

import json
import math
from pathlib import Path
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError:
    AprilTagDetectionArray = None

try:
    from hw2_interfaces.srv import GetRobotPose
except ImportError:
    GetRobotPose = None


def load_landmarks():
    """Load surveyed coordinates for landmarks from landmarks.json."""
    landmarks = {
        0: {'x': 1.75, 'y': 0.0},
        1: {'x': 1.0, 'y': 2.2},
        3: {'x': -0.4, 'y': 2.0},
        5: {'x': -0.14, 'y': -0.29},
    }
    possible_paths = [
        Path(__file__).resolve().parent.parent / 'config' / 'landmarks.json',
        Path('config/landmarks.json'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        possible_paths.insert(
            0,
            Path(get_package_share_directory('robot_testing')) / 'config' / 'landmarks.json'
        )
    except Exception:
        pass

    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for lm in data.get('landmarks', []):
                        lid = lm.get('id')
                        if lid in landmarks:
                            landmarks[lid]['x'] = float(lm.get('x_m', landmarks[lid]['x']))
                            landmarks[lid]['y'] = float(lm.get('y_m', landmarks[lid]['y']))
                    return landmarks
            except Exception:
                pass
    return landmarks


class ControlNode(Node):
    """Controls multi-step navigation using modular move_forward and rotate behaviors."""

    def __init__(self):
        super().__init__('hw2_control')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.on_pose, 1
        )

        if AprilTagDetectionArray is not None:
            self.detection_sub = self.create_subscription(
                AprilTagDetectionArray, '/detections', self.on_detections, 10
            )
        else:
            self.detection_sub = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_client = None
        if GetRobotPose is not None:
            self.pose_client = self.create_client(GetRobotPose, '/get_robot_pose')
        else:
            self.get_logger().info('hw2_interfaces not found; relying on TF and /robot_pose.')

        # Surveyed landmarks
        self.landmarks = load_landmarks()
        self.tag0_x = self.landmarks[0]['x']
        self.tag1_y = self.landmarks[1]['y']
        self.tag3_x = self.landmarks[3]['x']
        self.tag3_y = self.landmarks[3]['y']
        self.tag5_x = self.landmarks[5]['x']
        self.tag5_y = self.landmarks[5]['y']

        # Calibrated parameters
        self.tolerance = 0.035  # 3.5 cm tolerance
        self.settle_time_s = 1.0  # 1s delay for camera stream to stabilize
        self.turn_time_360_s = 3.75  # 3.75s per 360 deg turn on current surface
        self.angular_speed_ccw = 2.0 * math.pi / self.turn_time_360_s

        # Alignment tolerance: 10 deg (~86.5 px)
        self.tag1_align_tolerance_deg = 10.0
        self.tag1_align_tolerance_rad = math.radians(self.tag1_align_tolerance_deg)
        self.tag1_align_pixel_tolerance = 491.0 * math.tan(self.tag1_align_tolerance_rad)

        # Tracking state
        self.latest_tag_detections = {}
        self.latest_tag1_detection = None
        self.latest_robot_pose = None
        self.latest_robot_pose_time = 0.0
        self.pose_future = None
        self.request_started = 0.0

        # Position estimation
        self.current_x = 0.0
        self.current_y = 0.0

        # Coordinates logging: reset file on every run, record position every 1 second
        self.coord_paths = [
            Path('coordinate.txt').resolve(),
            Path('coordinates.txt').resolve(),
            (Path(__file__).resolve().parent.parent / 'coordinate.txt').resolve(),
            (Path(__file__).resolve().parent.parent / 'coordinates.txt').resolve(),
        ]
        self.coord_paths = list(dict.fromkeys(self.coord_paths))

        for path in self.coord_paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    pass
            except Exception as e:
                self.get_logger().warn(f'Failed to initialize coordinate file {path}: {e}')

        self.coord_timer = self.create_timer(1.0, self.record_coordinates_tick)

        self.get_logger().info(
            f'hw2_control initialized: Calibrated omega={self.angular_speed_ccw:.3f} rad/s. '
            f'Logging coordinates every 1s.'
        )

        # Launch mission runner in background thread
        self.mission_thread = threading.Thread(target=self.run, daemon=True)
        self.mission_thread.start()

    def stop_robot(self):
        """Halt all robot movement immediately."""
        self.cmd_pub.publish(Twist())

    def on_detections(self, msg):
        """Track detections of all AprilTags."""
        for det in msg.detections:
            info = {
                'cx': float(det.centre.x),
                'cy': float(det.centre.y),
                'time': time.monotonic(),
            }
            self.latest_tag_detections[det.id] = info
            if det.id == 1:
                self.latest_tag1_detection = info

    def on_pose(self, msg):
        """Cache pose from /robot_pose topic."""
        self.latest_robot_pose = msg
        self.latest_robot_pose_time = time.monotonic()

    def get_tag_transform(self, tag_id):
        """Lookup AprilTag transform in TF buffer for a given tag ID."""
        candidate_cam_frames = ('camera_frame', 'camera_link', 'camera_optical_frame')
        candidate_tag_frames = (
            f'tag_{tag_id}',
            f'tag36h11:{tag_id}',
            f'tag_36h11_{tag_id}',
        )
        for cam in candidate_cam_frames:
            for tag in candidate_tag_frames:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        cam, tag, rclpy.time.Time()
                    )
                    return trans
                except Exception:
                    continue
        return None

    def get_tag_z_distance(self, tag_id):
        """Lookup AprilTag z-distance (depth in front of camera) via TF or pose."""
        # 1. Direct TF lookup: camera frame -> tag frame
        tf_trans = self.get_tag_transform(tag_id)
        if tf_trans is not None:
            z = float(tf_trans.transform.translation.z)
            if math.isfinite(z) and 0.05 < z < 10.0:
                return z

        # 2. Fallback: check /robot_pose topic if recent
        if self.latest_robot_pose is not None:
            if time.monotonic() - self.latest_robot_pose_time < 3.0:
                px = self.latest_robot_pose.pose.position.x
                py = self.latest_robot_pose.pose.position.y
                if math.isfinite(px) and math.isfinite(py):
                    tag_info = self.landmarks.get(tag_id)
                    if tag_info is not None:
                        z = math.hypot(tag_info['x'] - px, tag_info['y'] - py)
                        if 0.05 < z < 10.0:
                            return z

        return None

    def get_tag_bearing_error(self, tag_id):
        """
        Calculate bearing error to specified AprilTag in radians and pixel offset.
        Positive -> tag is to the left (robot needs to turn CCW).
        Negative -> tag is to the right (robot needs to turn CW).
        Returns (bearing_rad, pixel_error) or (None, None).
        """
        now = time.monotonic()
        det = self.latest_tag_detections.get(tag_id)
        if det is not None and now - det['time'] < 2.0:
            cx = det['cx']
            bearing_rad = math.atan2(640.0 - cx, 491.0)
            pixel_error = cx - 640.0
            return bearing_rad, pixel_error

        # Fallback: check TF transform for tag
        tf_trans = self.get_tag_transform(tag_id)
        if tf_trans is not None:
            lx = tf_trans.transform.translation.x
            lz = tf_trans.transform.translation.z
            if math.isfinite(lx) and math.isfinite(lz) and lz > 0.05:
                bearing_rad = math.atan2(-lx, lz)
                pixel_error = lx * 491.0 / lz
                return bearing_rad, pixel_error

        return None, None

    def request_localization_service(self):
        """Request fresh pose from /get_robot_pose service if available."""
        if self.pose_client is None or not self.pose_client.service_is_ready():
            return
        if self.pose_future is not None:
            if time.monotonic() - self.request_started > 3.0:
                self.pose_future.cancel()
                self.pose_future = None
            return

        self.request_started = time.monotonic()
        self.pose_future = self.pose_client.call_async(GetRobotPose.Request())
        self.pose_future.add_done_callback(self.on_pose_service_response)

    def on_pose_service_response(self, future):
        """Handle response from /get_robot_pose service."""
        if future is not self.pose_future:
            return
        self.pose_future = None
        if future.cancelled():
            return
        try:
            response = future.result()
            if response.success:
                self.latest_robot_pose = response.pose
                self.latest_robot_pose_time = time.monotonic()
        except Exception:
            pass

    def get_estimated_position(self):
        """Return (x, y) estimated position coordinates in meters."""
        if self.latest_robot_pose is not None:
            if time.monotonic() - self.latest_robot_pose_time < 3.0:
                px = self.latest_robot_pose.pose.position.x
                py = self.latest_robot_pose.pose.position.y
                if math.isfinite(px) and math.isfinite(py):
                    return float(px), float(py)
        return float(self.current_x), float(self.current_y)

    def record_coordinates_tick(self):
        """Print estimated position coordinate to coordinate.txt every 1 second."""
        x, y = self.get_estimated_position()
        coord_line = f'({x:.3f}, {y:.3f})\n'
        for path in self.coord_paths:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(coord_line)
                    f.flush()
            except Exception as e:
                self.get_logger().warn(f'Failed to write coordinate to {path}: {e}')

    def drive_segment(self, distance, ux, uy, speed=0.18):
        """Drive straight along unit vector (ux, uy) for distance meters."""
        if distance <= 0.005 or not rclpy.ok():
            return
        dt = 0.05
        traveled = 0.0
        target_dist = abs(distance)
        direction = 1.0 if speed >= 0 else -1.0
        nominal_speed = abs(speed)

        while traveled < target_dist - 0.005 and rclpy.ok():
            dist_remaining = target_dist - traveled
            if target_dist > 0.3:
                curr_speed = 0.35 if dist_remaining > 0.15 else 0.18
            else:
                curr_speed = min(nominal_speed, 0.15)

            twist = Twist()
            twist.linear.x = float(direction * curr_speed)
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

            step = curr_speed * dt
            traveled += step
            self.current_x += ux * direction * step
            self.current_y += uy * direction * step
            time.sleep(dt)

        self.stop_robot()

    def rotate_angle(self, angle_rad, direction=1.0):
        """Rotate in place by angle_rad (radians) in direction (+1.0 CCW, -1.0 CW)."""
        target_angle = abs(angle_rad)
        if target_angle <= 0.005 or not rclpy.ok():
            return
        dt = 0.05
        traveled = 0.0
        v_ang = float(direction * self.angular_speed_ccw)

        while traveled < target_angle and rclpy.ok():
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = v_ang
            self.cmd_pub.publish(twist)

            step = abs(v_ang) * dt
            traveled += step
            time.sleep(dt)

        self.stop_robot()

    def rotate(self, target_tag):
        """
        Rotate CCW in 30-degree increments until target AprilTag is in vision,
        then do fine microadjustments until directly facing it.
        """
        self.get_logger().info(
            f'rotate: Searching CCW for AprilTag {target_tag} in 30-degree steps...'
        )

        # 1. Search loop: rotate 30 degrees CCW until tag is in vision
        step_angle = 30.0 * math.pi / 180.0
        while rclpy.ok():
            bearing_rad, _ = self.get_tag_bearing_error(target_tag)
            if bearing_rad is not None:
                self.get_logger().info(
                    f'rotate: AprilTag {target_tag} acquired! Starting fine alignment...'
                )
                break

            self.get_logger().info(
                f'rotate: Turning 30 degrees CCW to find AprilTag {target_tag}...'
            )
            self.rotate_angle(step_angle, direction=1.0)
            time.sleep(self.settle_time_s)

        # 2. Microadjustment loop: turn exact angle error until within tolerance
        while rclpy.ok():
            self.request_localization_service()
            time.sleep(self.settle_time_s)

            bearing_rad, pixel_err = self.get_tag_bearing_error(target_tag)
            if bearing_rad is None:
                time.sleep(0.5)
                bearing_rad, pixel_err = self.get_tag_bearing_error(target_tag)
                if bearing_rad is None:
                    self.get_logger().warn(
                        f'rotate: Lost sight of AprilTag {target_tag}.'
                    )
                    break

            deg_off = math.degrees(abs(bearing_rad))
            if (abs(pixel_err) <= self.tag1_align_pixel_tolerance or
                    abs(bearing_rad) <= self.tag1_align_tolerance_rad):
                self.get_logger().info(
                    f'rotate: Directly facing AprilTag {target_tag} within tolerance '
                    f'({pixel_err:.1f} px, {deg_off:.1f} deg).'
                )
                if target_tag == 3:
                    print('facing apriltag3')
                    self.get_logger().info('facing apriltag3')
                    time.sleep(3.0)
                break

            direction = 1.0 if bearing_rad > 0 else -1.0
            direction_str = 'CCW (left)' if direction > 0 else 'CW (right)'
            self.get_logger().info(
                f'rotate: Fine-adjusting {deg_off:.1f} deg {direction_str} '
                f'towards AprilTag {target_tag}...'
            )
            self.rotate_angle(abs(bearing_rad), direction=direction)

    def move_forward(self, to_coord, from_coord, target_tag):
        """
        Move forward towards an AprilTag and to_coord, stopping halfway to recalibrate,
        and adjusting at to_coord based on localized distance from the AprilTag.
        """
        x0, y0 = float(from_coord[0]), float(from_coord[1])
        x1, y1 = float(to_coord[0]), float(to_coord[1])
        self.current_x, self.current_y = x0, y0

        dx = x1 - x0
        dy = y1 - y0
        total_dist = math.hypot(dx, dy)
        ux = dx / total_dist if total_dist > 1e-6 else 0.0
        uy = dy / total_dist if total_dist > 1e-6 else 0.0

        half_dist = total_dist / 2.0

        tag_info = self.landmarks.get(target_tag, {'x': 1.75, 'y': 0.0})
        tag_x, tag_y = tag_info['x'], tag_info['y']
        target_z = math.hypot(tag_x - x1, tag_y - y1)

        self.get_logger().info(
            f'move_forward: From ({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f}) '
            f'towards AprilTag {target_tag} (dist: {total_dist:.3f} m, half: {half_dist:.3f} m).'
        )

        # Stage 1: Move halfway via dead reckoning
        self.drive_segment(half_dist, ux, uy, speed=0.35)
        self.stop_robot()
        time.sleep(self.settle_time_s)

        # Stage 2: Recalibrate halfway
        bearing_rad, pixel_err = self.get_tag_bearing_error(target_tag)
        if bearing_rad is not None:
            deg_off = math.degrees(abs(bearing_rad))
            if (abs(pixel_err) > self.tag1_align_pixel_tolerance and
                    abs(bearing_rad) > self.tag1_align_tolerance_rad):
                direction = 1.0 if bearing_rad > 0 else -1.0
                self.get_logger().info(
                    f'move_forward (halfway): Re-centering {deg_off:.1f} deg on Tag {target_tag}...'
                )
                self.rotate_angle(abs(bearing_rad), direction=direction)
                time.sleep(self.settle_time_s)

        # Measure localized distance to target
        self.request_localization_service()
        time.sleep(0.5)
        z = self.get_tag_z_distance(target_tag)
        if z is not None:
            remaining_dist = max(0.0, z - target_z)
            self.get_logger().info(
                f'move_forward (halfway): Tag {target_tag} z={z:.3f} m '
                f'(target z={target_z:.3f} m). Remaining distance: {remaining_dist:.3f} m.'
            )
            self.current_x = x1 - ux * remaining_dist
            self.current_y = y1 - uy * remaining_dist
        else:
            remaining_dist = total_dist - half_dist
            self.get_logger().warn(
                f'move_forward (halfway): Tag {target_tag} depth not found; '
                f'using nominal remaining: {remaining_dist:.3f} m.'
            )

        # Stage 3: Drive remaining distance to goal
        self.drive_segment(remaining_dist, ux, uy, speed=0.35)
        self.stop_robot()
        time.sleep(self.settle_time_s)

        # Stage 4: Verify and adjust at to_coord based on localized distance
        max_adjustments = 5
        for _ in range(max_adjustments):
            self.request_localization_service()
            time.sleep(0.5)
            z = self.get_tag_z_distance(target_tag)
            if z is None:
                self.get_logger().warn(
                    f'move_forward: Tag {target_tag} not visible for final adjustment.'
                )
                break

            error = z - target_z
            self.get_logger().info(
                f'move_forward: Tag {target_tag} z={z:.3f} m (error: {error:+.3f} m).'
            )

            if abs(error) <= self.tolerance:
                self.get_logger().info(
                    f'move_forward: Goal ({x1:.2f}, {y1:.2f}) reached within tolerance!'
                )
                break

            adjust_speed = 0.15 if error > 0 else -0.15
            adjust_dist = abs(error)
            dir_str = 'forward' if error > 0 else 'backward'
            self.get_logger().info(
                f'move_forward: Adjusting {adjust_dist:.3f} m {dir_str} to reach goal...'
            )
            self.drive_segment(adjust_dist, ux, uy, speed=adjust_speed)
            self.stop_robot()
            time.sleep(self.settle_time_s)

        self.current_x, self.current_y = x1, y1

    def run(self):
        """Execute the multi-waypoint mission using move_forward and rotate."""
        self.get_logger().info('Starting mission run...')
        time.sleep(self.settle_time_s)

        # Step 1: Move from (0, 0) to (1, 0) towards AprilTag 0
        self.move_forward((1.0, 0.0), (0.0, 0.0), 0)

        # Step 2: Rotate towards AprilTag 1
        self.rotate(1)

        # Step 3: Move from (1, 0) to (1, 2) towards AprilTag 1
        self.move_forward((1.0, 2.0), (1.0, 0.0), 1)

        # Step 4: Rotate towards AprilTag 3
        self.rotate(3)

        # Step 5: Rotate towards AprilTag 5
        self.rotate(5)

        # Step 6: Move from (1, 2) to (0, 0) towards AprilTag 5
        self.move_forward((0.0, 0.0), (1.0, 2.0), 5)

        # Step 7: Rotate towards AprilTag 0
        self.rotate(0)

        self.stop_robot()
        print('end')
        self.get_logger().info('Mission complete: printed "end".')


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
