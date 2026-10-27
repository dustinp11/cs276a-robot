"""Waypoint tracking to (1, 0), align to Tag 1, traverse to (1, 2), and loop to (0, 0).

Controls the robot through seven sequential steps:
1. Step 1: Forward-only navigation to (1.0, 0.0):
   - Initial stationary check: waits 1 second for camera to stabilize, reads
     AprilTag 0 z-distance (depth in front of camera).
   - Tag 0 is at x = 1.75 m; target depth at (1, 0) is z = 0.75 m.
   - Drives forward ONLY (angular.z strictly 0.0, no rotation).
   - Once dead reckoning reaches target distance, halts and settles 1 second.
   - Measures AprilTag 0 z-distance directly.
   - If short of (1, 0) (z > 0.75 m + tolerance), moves forward by the difference.
   - Repeats until tag z is within tolerance of 0.75 m.

2. Step 2: Turn CCW until directly facing AprilTag 1 (ID 1 ONLY):
   - Rotates CCW in 30-degree increments (calibrated angular speed: 3.75s per 360 deg).
   - Stops and waits 1 second every 30 degrees to check for AprilTag 1.
   - Once AprilTag 1 is seen, calculates the exact bearing angle it is off.
   - Turns ONLY the number of degrees it is off.
   - In between each fine align rotation, halts and waits 1 second for the
     camera stream to settle before measuring again.
   - Concludes alignment when directly facing AprilTag 1 (|error| <= 10 deg / ~86 px).

3. Step 3: Navigation to (1.0, 2.0) towards AprilTag 1:
   - Moves forward 1.0 m towards AprilTag 1 via dead reckoning.
   - Halts and settles 1 second to re-center towards AprilTag 1 (with 1 second delay
     between each turn).
   - Once centered, determines localization distance to (1, 2) from AprilTag 1 z-distance.
   - Drives forward by the remaining distance to (1, 2).
   - Settles 1 second at (1, 2) and checks AprilTag 1 z-distance:
     - Adjusts forward if short of (1, 2).
     - Adjusts backward if it overshoots (1, 2).
     - Repeats until within tolerance.
   - If AprilTag 1 is no longer visible: prints that it can no longer see the
     AprilTag, and ends navigation.

4. Step 4: Turn 90 degrees CCW towards AprilTag 3 (ID 3):
   - Turns 90 degrees CCW directly towards AprilTag 3, settles 1s.
   - Looks for AprilTag 3 and fine-adjusts (1s delay between turns) until directly facing it.
   - Prints "facing apriltag3" in terminal and pauses for 3 seconds.

5. Step 5: Turn 63 degrees CCW towards AprilTag 5 (ID 5):
   - Rotates 63 degrees CCW, settles 1s.
   - Fine-adjusts with 1s delay in between turns until directly facing AprilTag 5.

6. Step 6: Navigation 2.23 m back towards (0, 0):
   - Moves forward 1.0 m towards AprilTag 5 via dead reckoning.
   - Halts and settles 1 second to re-center towards AprilTag 5 (1s delay between turns).
   - Determines localization distance to (0, 0) from AprilTag 5 z-distance.
   - Drives forward remaining distance to (0, 0).
   - Settles 1 second at (0, 0) and adjusts forward or backward until within tolerance.

7. Step 7: Turn 120 degrees CCW towards AprilTag 0 (ID 0):
   - Rotates 120 degrees CCW, settles 1s.
   - Fine-adjusts with 1s delay in between turns until AprilTag 0 is center.
   - Ends navigation.
"""

import json
import math
from pathlib import Path
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
    tag0_x = 1.75
    tag1_y = 2.2
    tag3_x = -0.4
    tag3_y = 2.0
    tag5_x = -0.14
    tag5_y = -0.29
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
                        if lid == 0:
                            tag0_x = float(lm.get('x_m', 1.75))
                        elif lid == 1:
                            tag1_y = float(lm.get('y_m', 2.2))
                        elif lid == 3:
                            tag3_x = float(lm.get('x_m', -0.4))
                            tag3_y = float(lm.get('y_m', 2.0))
                        elif lid == 5:
                            tag5_x = float(lm.get('x_m', -0.14))
                            tag5_y = float(lm.get('y_m', -0.29))
                    return tag0_x, tag1_y, tag3_x, tag3_y, tag5_x, tag5_y
            except Exception:
                pass
    return tag0_x, tag1_y, tag3_x, tag3_y, tag5_x, tag5_y


class ControlNode(Node):
    """Controls multi-step navigation: (1, 0) -> align Tag 1 -> (1, 2)."""

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

        # Target definitions for landmarks and waypoints
        (
            self.tag0_x,
            self.tag1_y,
            self.tag3_x,
            self.tag3_y,
            self.tag5_x,
            self.tag5_y,
        ) = load_landmarks()
        self.target_x_wp1 = 1.0
        self.target_z_wp1 = self.tag0_x - self.target_x_wp1
        self.target_y_wp2 = 2.0
        self.target_z_wp2 = self.tag1_y - self.target_y_wp2
        self.target_z_origin = math.hypot(self.tag5_x, self.tag5_y)

        self.tolerance = 0.035  # 3.5 cm tolerance
        self.settle_time_s = 0.7  # 1s delay for camera stream to stabilize

        # Calibrated parameters: on current surface, full 360-deg turn takes 3.75s
        self.turn_time_360_s = 3.75
        self.angular_speed_cw = 2.0 * math.pi / self.turn_time_360_s
        self.angular_speed_ccw = 2.0 * math.pi / self.turn_time_360_s
        self.step_angle_rad = 30.0 * math.pi / 180.0  # 30 degrees per search step
        self.turn_step_traveled = 0.0

        # AprilTag tracking
        self.latest_tag_detections = {}
        self.latest_tag1_detection = None
        self.align_target_angle = 0.0
        self.align_traveled_angle = 0.0
        self.align_direction = 1.0
        # Increased tolerance to 10 deg (~86 px) so wheels don't get stuck on micro-angles
        self.tag1_align_tolerance_deg = 7.0
        self.tag1_align_tolerance_rad = math.radians(self.tag1_align_tolerance_deg)
        self.tag1_align_pixel_tolerance = 491.0 * math.tan(self.tag1_align_tolerance_rad)

        # Direction for final position adjustments (+1.0 forward, -1.0 backward)
        self.adjust_direction = 1.0

        # Control and motion variables
        self.state = 'initial_settling'
        self.target_drive_dist = 1.0
        self.distance_traveled = 0.0
        self.stop_time = self.get_clock().now()
        self.last_tick = self.get_clock().now()

        # Pose cache from service or topic
        self.latest_robot_pose = None
        self.latest_robot_pose_time = 0.0
        self.pose_future = None
        self.request_started = 0.0

        # Coordinates logging: reset file on every run, record position every 1 second
        self.coord_paths = [Path('coordinates.txt').resolve()]
        repo_coord_path = (Path(__file__).resolve().parent.parent / 'coordinates.txt').resolve()
        if repo_coord_path not in self.coord_paths:
            self.coord_paths.append(repo_coord_path)

        for path in self.coord_paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    pass
            except Exception as e:
                self.get_logger().warn(f'Failed to initialize coordinate file {path}: {e}')

        self.coord_timer = self.create_timer(1.0, self.record_coordinates_tick)

        self.timer = self.create_timer(0.1, self.control_tick)
        self.get_logger().info(
            f'hw2_control initialized: Leg 1 to (1, 0), align Tag 1, Leg 2 to (1, 2). '
            f'Tag 0 x={self.tag0_x:.3f} m, Tag 1 y={self.tag1_y:.3f} m, '
            f'Settle delay={self.settle_time_s:.1f}s. Logging coordinates every 1s.'
        )

    def get_estimated_position(self):
        """Return (x, y) estimated position coordinates in meters."""
        # 1. Prefer latest localization pose from hw2_localization if recent
        if self.latest_robot_pose is not None:
            if time.monotonic() - self.latest_robot_pose_time < 3.0:
                px = self.latest_robot_pose.pose.position.x
                py = self.latest_robot_pose.pose.position.y
                if math.isfinite(px) and math.isfinite(py):
                    return float(px), float(py)

        # 2. State-based estimation using dead reckoning and tag depth measurements
        hypot_1_2 = math.hypot(1.0, 2.0)
        ux = -1.0 / hypot_1_2
        uy = -2.0 / hypot_1_2

        state = self.state

        if state == 'initial_settling':
            z = self.get_tag0_z_distance()
            if z is not None:
                return float(self.tag0_x - z), 0.0
            return 0.0, 0.0

        if state == 'driving_forward':
            return float(self.distance_traveled), 0.0

        if state == 'settling_and_checking':
            z = self.get_tag0_z_distance()
            if z is not None:
                return float(self.tag0_x - z), 0.0
            return 1.0, 0.0

        if state in (
            'turn_step_ccw_30',
            'tag1_search_stopped_query',
            'tag1_settling_and_measuring',
            'tag1_rotating_to_align',
        ):
            return 1.0, 0.0

        if state == 'driving_to_wp2_half':
            return 1.0, float(self.distance_traveled)

        if state in ('tag1_mid_settling_and_measuring', 'tag1_mid_rotating_to_align'):
            z = self.get_tag1_z_distance()
            if z is not None:
                return 1.0, float(self.tag1_y - z)
            return 1.0, 1.0

        if state == 'driving_to_wp2_remaining':
            return 1.0, float(1.0 + self.distance_traveled)

        if state == 'settling_and_checking_wp2':
            z = self.get_tag1_z_distance()
            if z is not None:
                return 1.0, float(self.tag1_y - z)
            return 1.0, 2.0

        if state == 'adjusting_wp2':
            return 1.0, float(2.0 + self.adjust_direction * self.distance_traveled)

        if state in (
            'tag3_turn_step_ccw_30',
            'tag3_search_stopped_query',
            'tag3_settling_and_measuring',
            'tag3_rotating_to_align',
            'tag3_pause_before_step5',
            'turn_63_ccw_to_tag5',
            'tag5_settling_and_measuring',
            'tag5_rotating_to_align',
        ):
            return 1.0, 2.0

        if state == 'driving_to_origin_half':
            return (
                float(1.0 + ux * self.distance_traveled),
                float(2.0 + uy * self.distance_traveled),
            )

        if state in ('tag5_mid_settling_and_measuring', 'tag5_mid_rotating_to_align'):
            z = self.get_tag5_z_distance()
            if z is not None:
                dist_to_origin = z - self.target_z_origin
                return float(-ux * dist_to_origin), float(-uy * dist_to_origin)
            return float(1.0 + ux * 1.0), float(2.0 + uy * 1.0)

        if state == 'driving_to_origin_remaining':
            traveled = 1.0 + self.distance_traveled
            return float(1.0 + ux * traveled), float(2.0 + uy * traveled)

        if state == 'settling_and_checking_origin':
            z = self.get_tag5_z_distance()
            if z is not None:
                dist_to_origin = z - self.target_z_origin
                return float(-ux * dist_to_origin), float(-uy * dist_to_origin)
            return 0.0, 0.0

        if state == 'adjusting_origin':
            offset = self.adjust_direction * self.distance_traveled
            return float(-ux * offset), float(-uy * offset)

        if state in (
            'turn_120_ccw_to_tag0',
            'tag0_final_settling_and_measuring',
            'tag0_final_rotating_to_align',
            'done',
        ):
            z = self.get_tag0_z_distance()
            if z is not None:
                return float(self.tag0_x - z), 0.0
            return 0.0, 0.0

        return 0.0, 0.0

    def record_coordinates_tick(self):
        """Print estimated position coordinate to coordinates.txt every 1 second."""
        x, y = self.get_estimated_position()
        coord_line = f'({x:.3f}, {y:.3f})\n'
        for path in self.coord_paths:
            try:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(coord_line)
                    f.flush()
            except Exception as e:
                self.get_logger().warn(f'Failed to write coordinate to {path}: {e}')

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
                if tag_id == 0 and math.isfinite(px):
                    z = self.tag0_x - px
                    if 0.1 < z < 10.0:
                        return z
                elif tag_id == 1 and math.isfinite(py):
                    z = self.tag1_y - py
                    if 0.05 < z < 10.0:
                        return z
                elif tag_id == 5 and math.isfinite(px) and math.isfinite(py):
                    z = math.hypot(self.tag5_x - px, self.tag5_y - py)
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

        # Fallback: check TF transform
        tf_trans = self.get_tag_transform(tag_id)
        if tf_trans is not None:
            lx = tf_trans.transform.translation.x
            lz = tf_trans.transform.translation.z
            if math.isfinite(lx) and math.isfinite(lz) and lz > 0.05:
                bearing_rad = math.atan2(-lx, lz)
                pixel_error = lx * 491.0 / lz
                return bearing_rad, pixel_error

        return None, None

    def get_tag0_z_distance(self):
        """Lookup AprilTag 0 z-distance."""
        return self.get_tag_z_distance(0)

    def get_tag1_transform(self):
        """Lookup AprilTag 1 transform in TF buffer."""
        return self.get_tag_transform(1)

    def get_tag1_z_distance(self):
        """Lookup AprilTag 1 z-distance."""
        return self.get_tag_z_distance(1)

    def get_tag5_z_distance(self):
        """Lookup AprilTag 5 z-distance."""
        return self.get_tag_z_distance(5)

    def get_tag0_bearing_error(self):
        """Bearing error to AprilTag 0."""
        return self.get_tag_bearing_error(0)

    def get_tag1_bearing_error(self):
        """Bearing error to AprilTag 1."""
        return self.get_tag_bearing_error(1)

    def get_tag3_bearing_error(self):
        """Bearing error to AprilTag 3."""
        return self.get_tag_bearing_error(3)

    def get_tag5_bearing_error(self):
        """Bearing error to AprilTag 5."""
        return self.get_tag_bearing_error(5)

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

    def control_tick(self):
        """Main periodic control loop executing multi-step navigation."""
        now = self.get_clock().now()
        dt = (now - self.last_tick).nanoseconds / 1e9
        self.last_tick = now
        if dt <= 0.0 or dt > 0.5:
            dt = 0.1

        if self.state == 'done':
            self.stop_robot()
            return

        # Phase 1: Initial stationary settling and distance determination for Leg 1
        if self.state == 'initial_settling':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                self.request_localization_service()
                z = self.get_tag0_z_distance()
                if z is not None:
                    drive_dist = z - self.target_z_wp1
                    self.get_logger().info(
                        f'Initial AprilTag 0 z distance: {z:.3f} m. '
                        f'Target z: {self.target_z_wp1:.3f} m. '
                        f'Planned forward drive: {drive_dist:.3f} m.'
                    )
                    self.target_drive_dist = max(0.1, drive_dist)
                    self.distance_traveled = 0.0
                    self.state = 'driving_forward'
                    return

            if time_stopped > (self.settle_time_s + 4.0):
                self.get_logger().warn(
                    'Initial tag detection timed out; assuming start at (0, 0) '
                    'and driving 1.0 m.'
                )
                self.target_drive_dist = 1.0
                self.distance_traveled = 0.0
                self.state = 'driving_forward'
            return

        # Phase 2: Forward motion to (1, 0) (strictly angular.z = 0.0, no rotation)
        if self.state == 'driving_forward':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed forward drive of {self.distance_traveled:.3f} m. '
                    f'Settling {self.settle_time_s:.1f}s before checking tag 0...'
                )
                self.state = 'settling_and_checking'
                self.stop_time = self.get_clock().now()
                return

            if self.target_drive_dist > 0.3:
                speed = 0.35 if dist_remaining > 0.15 else 0.18
            else:
                speed = 0.15

            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = 0.0  # Forward ONLY - never spin or rotate
            self.cmd_pub.publish(twist)
            self.distance_traveled += speed * dt
            return

        # Phase 3: Stationary verification and fine adjustment using AprilTag 0 z-distance
        if self.state == 'settling_and_checking':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            z = self.get_tag0_z_distance()
            if z is not None:
                error = z - self.target_z_wp1
                self.get_logger().info(
                    f'Stationary AprilTag 0 z distance: {z:.3f} m '
                    f'(target: {self.target_z_wp1:.3f} m, error: {error:+.3f} m).'
                )

                if abs(error) <= self.tolerance:
                    self.get_logger().info(
                        f'Waypoint (1, 0) reached within tolerance! '
                        f'Final tag z = {z:.3f} m (error: {error:+.3f} m). '
                        f'Step 2: Searching for AprilTag 1 in 30-degree CCW steps...'
                    )
                    self.stop_robot()
                    self.latest_tag1_detection = None
                    self.turn_step_traveled = 0.0
                    self.state = 'turn_step_ccw_30'
                elif error > self.tolerance:
                    self.get_logger().info(
                        f'Robot is {error:.3f} m short of (1, 0). '
                        f'Moving forward {error:.3f} m to reach position...'
                    )
                    self.target_drive_dist = error
                    self.distance_traveled = 0.0
                    self.state = 'driving_forward'
                else:
                    self.get_logger().info(
                        f'Robot reached (1, 0) (tag z={z:.3f} m <= {self.target_z_wp1:.3f} m). '
                        f'Step 2: Searching for AprilTag 1 in 30-degree CCW steps...'
                    )
                    self.stop_robot()
                    self.latest_tag1_detection = None
                    self.turn_step_traveled = 0.0
                    self.state = 'turn_step_ccw_30'
                return

            if time_stopped > (self.settle_time_s + 5.0):
                self.get_logger().warn(
                    'AprilTag 0 not detected during settling check. Continuing to poll...'
                )
            return

        # Phase 4: Step turn 30 degrees CCW to search for AprilTag 1
        if self.state == 'turn_step_ccw_30':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = v_ang * dt
            self.turn_step_traveled += step_dt
            self.cmd_pub.publish(twist)

            if self.turn_step_traveled >= self.step_angle_rad:
                self.stop_robot()
                self.get_logger().info(
                    f'Turned 30 degrees CCW. Stopping for {self.settle_time_s:.1f}s '
                    f'delay to check for AprilTag 1...'
                )
                self.state = 'tag1_search_stopped_query'
                self.stop_time = self.get_clock().now()
            return

        # Phase 5: Stationary query after 30-degree spin
        if self.state == 'tag1_search_stopped_query':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            bearing_rad, pixel_err = self.get_tag1_bearing_error()
            if bearing_rad is not None:
                self.get_logger().info(
                    f'AprilTag 1 acquired! Pixel offset: {pixel_err:.1f} px, '
                    f'bearing: {math.degrees(bearing_rad):.1f} deg. '
                    f'Settling {self.settle_time_s:.1f}s before starting fine alignment...'
                )
                self.stop_robot()
                self.state = 'tag1_settling_and_measuring'
                self.stop_time = self.get_clock().now()
                return

            if time_stopped > (self.settle_time_s + 2.0):
                self.get_logger().info(
                    f'AprilTag 1 not visible at this angle after {self.settle_time_s:.1f}s delay. '
                    f'Turning CCW another 30 degrees...'
                )
                self.turn_step_traveled = 0.0
                self.state = 'turn_step_ccw_30'
            return

        # Phase 6: Settle 1s and measure exact angle off from AprilTag 1 (ID 1 ONLY)
        if self.state == 'tag1_settling_and_measuring':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            bearing_rad, pixel_err = self.get_tag1_bearing_error()
            if bearing_rad is not None:
                deg_off = math.degrees(abs(bearing_rad))
                self.get_logger().info(
                    f'Stationary AprilTag 1 measurement: pixel offset={pixel_err:.1f} px, '
                    f'angle off={deg_off:.1f} deg.'
                )

                # Check if directly facing AprilTag 1 within increased tolerance (10 deg / ~86 px)
                if (abs(pixel_err) <= self.tag1_align_pixel_tolerance or
                        abs(bearing_rad) <= self.tag1_align_tolerance_rad):
                    self.stop_robot()
                    self.get_logger().info(
                        f'Directly facing AprilTag 1 within tolerance! '
                        f'(offset: {pixel_err:.1f} px, {deg_off:.1f} deg <= '
                        f'{self.tag1_align_tolerance_deg:.1f} deg).'
                    )
                    self.get_logger().info(
                        'Step 3: Moving forward 1.0 m towards AprilTag 1 via dead reckoning...'
                    )
                    self.target_drive_dist = 1.0
                    self.distance_traveled = 0.0
                    self.state = 'driving_to_wp2_half'
                    return

                # Calculate rotation to turn ONLY the number of degrees it is off
                self.align_target_angle = abs(bearing_rad)
                self.align_direction = 1.0 if bearing_rad > 0 else -1.0
                self.align_traveled_angle = 0.0
                direction_str = 'CCW (left)' if self.align_direction > 0 else 'CW (right)'
                self.get_logger().info(
                    f'Turning {deg_off:.1f} degrees {direction_str} towards AprilTag 1...'
                )
                self.state = 'tag1_rotating_to_align'
                return

            if time_stopped > (self.settle_time_s + 3.0):
                self.get_logger().warn(
                    'Lost AprilTag 1 during settling. Resuming 30-degree CCW search...'
                )
                self.turn_step_traveled = 0.0
                self.state = 'turn_step_ccw_30'
            return

        # Phase 7: Rotate ONLY the target angle, then stop and wait 1s
        if self.state == 'tag1_rotating_to_align':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.align_direction * self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = abs(v_ang) * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                traveled_deg = math.degrees(self.align_traveled_angle)
                self.get_logger().info(
                    f'Completed fine rotation of {traveled_deg:.1f} deg. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay before checking tag...'
                )
                self.state = 'tag1_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 8: Drive forward 1.0 m towards AprilTag 1 via dead reckoning
        if self.state == 'driving_to_wp2_half':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed 1.0 m intermediate drive '
                    f'(traveled {self.distance_traveled:.3f} m). '
                    f'Stopping for {self.settle_time_s:.1f}s delay before re-centering on Tag 1...'
                )
                self.state = 'tag1_mid_settling_and_measuring'
                self.stop_time = self.get_clock().now()
                return

            if self.target_drive_dist > 0.3:
                speed = 0.35 if dist_remaining > 0.15 else 0.18
            else:
                speed = 0.15

            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = 0.0  # Forward ONLY towards AprilTag 1
            self.cmd_pub.publish(twist)
            self.distance_traveled += speed * dt
            return

        # Phase 9: Settle 1s at 1m mark, check center on Tag 1, or determine distance to (1, 2)
        if self.state == 'tag1_mid_settling_and_measuring':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            bearing_rad, pixel_err = self.get_tag1_bearing_error()

            if bearing_rad is not None:
                deg_off = math.degrees(abs(bearing_rad))
                self.get_logger().info(
                    f'Intermediate AprilTag 1 measurement: offset={pixel_err:.1f} px, '
                    f'angle off={deg_off:.1f} deg.'
                )

                # If centered towards AprilTag 1 within tolerance:
                if (abs(pixel_err) <= self.tag1_align_pixel_tolerance or
                        abs(bearing_rad) <= self.tag1_align_tolerance_rad):
                    self.stop_robot()
                    z = self.get_tag1_z_distance()
                    if z is not None:
                        remaining_dist = z - self.target_z_wp2
                        self.get_logger().info(
                            f'Centered towards AprilTag 1! Tag 1 z={z:.3f} m, '
                            f'target z={self.target_z_wp2:.3f} m. '
                            f'Localization distance to (1, 2): {remaining_dist:.3f} m.'
                        )
                        self.target_drive_dist = max(0.0, remaining_dist)
                        self.distance_traveled = 0.0
                        self.state = 'driving_to_wp2_remaining'
                        return
                    elif time_stopped > (self.settle_time_s + 2.0):
                        self.get_logger().warn(
                            'Tag 1 depth not resolved; using nominal 1.0 m remaining distance.'
                        )
                        self.target_drive_dist = 1.0
                        self.distance_traveled = 0.0
                        self.state = 'driving_to_wp2_remaining'
                        return
                    return

                # Not yet centered: rotate ONLY the angle off towards AprilTag 1
                self.align_target_angle = abs(bearing_rad)
                self.align_direction = 1.0 if bearing_rad > 0 else -1.0
                self.align_traveled_angle = 0.0
                direction_str = 'CCW (left)' if self.align_direction > 0 else 'CW (right)'
                self.get_logger().info(
                    f'Re-centering at 1m mark: turning {deg_off:.1f} deg {direction_str} '
                    f'towards AprilTag 1...'
                )
                self.state = 'tag1_mid_rotating_to_align'
                return

            if time_stopped > (self.settle_time_s + 2.0):
                print('Can no longer see AprilTag 1.')
                self.get_logger().info(
                    'Can no longer see AprilTag 1 at 1m mark. Ending navigation.'
                )
                self.stop_robot()
                self.state = 'done'
            return

        # Phase 10: Intermediate rotate to align with AprilTag 1, then settle 1s
        if self.state == 'tag1_mid_rotating_to_align':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.align_direction * self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = abs(v_ang) * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                traveled_deg = math.degrees(self.align_traveled_angle)
                self.get_logger().info(
                    f'Completed intermediate turn of {traveled_deg:.1f} deg. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay before checking tag...'
                )
                self.state = 'tag1_mid_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 11: Drive remaining distance to (1, 2)
        if self.state == 'driving_to_wp2_remaining':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed remaining drive of {self.distance_traveled:.3f} m to (1, 2). '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay to verify position...'
                )
                self.state = 'settling_and_checking_wp2'
                self.stop_time = self.get_clock().now()
                return

            if self.target_drive_dist > 0.3:
                speed = 0.35 if dist_remaining > 0.15 else 0.18
            else:
                speed = 0.15

            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = 0.0  # Forward ONLY towards AprilTag 1
            self.cmd_pub.publish(twist)
            self.distance_traveled += speed * dt
            return

        # Phase 12: Settle 1s at (1, 2); verify position with Tag 1 z-distance
        if self.state == 'settling_and_checking_wp2':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            z = self.get_tag1_z_distance()
            now_mono = time.monotonic()
            tag1_seen = (z is not None) or (
                self.latest_tag1_detection is not None and
                now_mono - self.latest_tag1_detection['time'] < 1.5
            )

            if tag1_seen and z is not None:
                error = z - self.target_z_wp2
                self.get_logger().info(
                    f'Stationary AprilTag 1 z distance: {z:.3f} m '
                    f'(target: {self.target_z_wp2:.3f} m, error: {error:+.3f} m).'
                )

                if abs(error) <= self.tolerance:
                    self.get_logger().info(
                        f'Waypoint (1, 2) reached within tolerance! '
                        f'Final tag z = {z:.3f} m (error: {error:+.3f} m). '
                        f'Step 4: Turning 90 degrees CCW directly towards AprilTag 3...'
                    )
                    self.stop_robot()
                    self.align_target_angle = math.pi / 2.0  # 90 degrees
                    self.align_direction = 1.0  # CCW
                    self.align_traveled_angle = 0.0
                    self.state = 'turn_90_ccw_to_tag3'
                    return
                elif error > self.tolerance:
                    self.get_logger().info(
                        f'Robot is {error:.3f} m short of (1, 2). '
                        f'Moving forward {error:.3f} m to reach position...'
                    )
                    self.target_drive_dist = error
                    self.adjust_direction = 1.0
                    self.distance_traveled = 0.0
                    self.state = 'adjusting_wp2'
                else:
                    overshoot = abs(error)
                    self.get_logger().info(
                        f'Robot overshot (1, 2) by {overshoot:.3f} m. '
                        f'Moving backward {overshoot:.3f} m to reach position...'
                    )
                    self.target_drive_dist = overshoot
                    self.adjust_direction = -1.0
                    self.distance_traveled = 0.0
                    self.state = 'adjusting_wp2'
                return

            if time_stopped > (self.settle_time_s + 1.5):
                print('Can no longer see AprilTag 1.')
                self.get_logger().info('Can no longer see AprilTag 1. Ending navigation.')
                self.stop_robot()
                self.state = 'done'
            return

        # Phase 13: Fine adjustment forward or backward to reach (1, 2)
        if self.state == 'adjusting_wp2':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                direction_label = 'forward' if self.adjust_direction > 0 else 'backward'
                self.get_logger().info(
                    f'Completed {direction_label} adjustment of '
                    f'{self.distance_traveled:.3f} m. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay...'
                )
                self.state = 'settling_and_checking_wp2'
                self.stop_time = self.get_clock().now()
                return

            twist = Twist()
            twist.linear.x = float(0.15 * self.adjust_direction)
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.distance_traveled += 0.15 * dt
            return

        # Phase 14: Direct 90 degrees CCW turn towards AprilTag 3
        if self.state == 'turn_90_ccw_to_tag3':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = v_ang * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                self.get_logger().info(
                    f'Turned 90 degrees CCW. Stopping for {self.settle_time_s:.1f}s '
                    f'delay to look for AprilTag 3 and adjust...'
                )
                self.state = 'tag3_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 15: Settle 1s, look for AprilTag 3, and measure angle off
        if self.state == 'tag3_settling_and_measuring':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            bearing_rad, pixel_err = self.get_tag3_bearing_error()
            if bearing_rad is not None:
                deg_off = math.degrees(abs(bearing_rad))
                self.get_logger().info(
                    f'Stationary AprilTag 3 measurement: pixel offset={pixel_err:.1f} px, '
                    f'angle off={deg_off:.1f} deg.'
                )

                if (abs(pixel_err) <= self.tag1_align_pixel_tolerance or
                        abs(bearing_rad) <= self.tag1_align_tolerance_rad):
                    self.stop_robot()
                    print('facing apriltag3')
                    self.get_logger().info('facing apriltag3')
                    self.get_logger().info(
                        'Directly facing AprilTag 3. Pausing 3 seconds before Step 5...'
                    )
                    self.state = 'tag3_pause_before_step5'
                    self.stop_time = self.get_clock().now()
                    return

                self.align_target_angle = abs(bearing_rad)
                self.align_direction = 1.0 if bearing_rad > 0 else -1.0
                self.align_traveled_angle = 0.0
                direction_str = 'CCW (left)' if self.align_direction > 0 else 'CW (right)'
                self.get_logger().info(
                    f'Turning {deg_off:.1f} degrees {direction_str} towards AprilTag 3...'
                )
                self.state = 'tag3_rotating_to_align'
                return

            if time_stopped > (self.settle_time_s + 4.0):
                self.get_logger().warn(
                    'AprilTag 3 not detected after 90-degree turn. '
                    'Proceeding with 3-second pause before Step 5...'
                )
                print('facing apriltag3')
                self.state = 'tag3_pause_before_step5'
                self.stop_time = self.get_clock().now()
            return

        # Phase 16: Rotate ONLY target angle towards AprilTag 3, then stop and wait 1s
        if self.state == 'tag3_rotating_to_align':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.align_direction * self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = abs(v_ang) * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                traveled_deg = math.degrees(self.align_traveled_angle)
                self.get_logger().info(
                    f'Completed fine rotation of {traveled_deg:.1f} deg towards Tag 3. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay before checking tag...'
                )
                self.state = 'tag3_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 17: Pause 3 seconds after facing AprilTag 3 before moving to Step 5
        if self.state == 'tag3_pause_before_step5':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < 2:
                return

            self.get_logger().info(
                'Step 5: Turning 63 degrees CCW towards AprilTag 5...'
            )
            self.align_target_angle = math.radians(63.0)
            self.align_direction = 1.0  # CCW
            self.align_traveled_angle = 0.0
            self.state = 'turn_63_ccw_to_tag5'
            return

        # Phase 18: Turn 63 degrees CCW towards AprilTag 5
        if self.state == 'turn_63_ccw_to_tag5':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = v_ang * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                self.get_logger().info(
                    f'Turned 63 degrees CCW. Stopping for {self.settle_time_s:.1f}s '
                    f'delay before fine-adjusting to AprilTag 5...'
                )
                self.state = 'tag5_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 19: Settle 1s and fine-adjust towards AprilTag 5
        if self.state == 'tag5_settling_and_measuring':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            bearing_rad, pixel_err = self.get_tag5_bearing_error()
            if bearing_rad is not None:
                deg_off = math.degrees(abs(bearing_rad))
                self.get_logger().info(
                    f'Stationary AprilTag 5 measurement: pixel offset={pixel_err:.1f} px, '
                    f'angle off={deg_off:.1f} deg.'
                )

                if (abs(pixel_err) <= self.tag1_align_pixel_tolerance or
                        abs(bearing_rad) <= self.tag1_align_tolerance_rad):
                    self.stop_robot()
                    self.get_logger().info(
                        f'Directly facing AprilTag 5 within tolerance! '
                        f'(offset: {pixel_err:.1f} px, {deg_off:.1f} deg <= '
                        f'{self.tag1_align_tolerance_deg:.1f} deg).'
                    )
                    self.get_logger().info(
                        'Step 6: Moving forward 1.0 m towards (0, 0) via dead reckoning...'
                    )
                    self.target_drive_dist = 1.0
                    self.distance_traveled = 0.0
                    self.state = 'driving_to_origin_half'
                    return

                self.align_target_angle = abs(bearing_rad)
                self.align_direction = 1.0 if bearing_rad > 0 else -1.0
                self.align_traveled_angle = 0.0
                direction_str = 'CCW (left)' if self.align_direction > 0 else 'CW (right)'
                self.get_logger().info(
                    f'Fine-adjusting: turning {deg_off:.1f} degrees {direction_str} '
                    f'towards AprilTag 5...'
                )
                self.state = 'tag5_rotating_to_align'
                return

            if time_stopped > (self.settle_time_s + 3.0):
                self.get_logger().warn(
                    'AprilTag 5 not detected after 63-deg turn. '
                    'Proceeding with 1.0 m forward drive towards (0, 0)...'
                )
                self.target_drive_dist = 1.0
                self.distance_traveled = 0.0
                self.state = 'driving_to_origin_half'
            return

        # Phase 20: Rotate ONLY target angle towards AprilTag 5, then stop and wait 1s
        if self.state == 'tag5_rotating_to_align':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.align_direction * self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = abs(v_ang) * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                traveled_deg = math.degrees(self.align_traveled_angle)
                self.get_logger().info(
                    f'Completed fine rotation of {traveled_deg:.1f} deg towards Tag 5. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay before checking tag...'
                )
                self.state = 'tag5_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 21: Drive forward 1.0 m towards AprilTag 5 via dead reckoning
        if self.state == 'driving_to_origin_half':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed 1.0 m intermediate drive towards (0, 0) '
                    f'(traveled {self.distance_traveled:.3f} m). '
                    f'Stopping for {self.settle_time_s:.1f}s delay before re-centering on Tag 5...'
                )
                self.state = 'tag5_mid_settling_and_measuring'
                self.stop_time = self.get_clock().now()
                return

            if self.target_drive_dist > 0.3:
                speed = 0.35 if dist_remaining > 0.15 else 0.18
            else:
                speed = 0.15

            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = 0.0  # Forward ONLY towards AprilTag 5
            self.cmd_pub.publish(twist)
            self.distance_traveled += speed * dt
            return

        # Phase 22: Settle 1s at 1m mark, check center on Tag 5, or determine distance to (0, 0)
        if self.state == 'tag5_mid_settling_and_measuring':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            bearing_rad, pixel_err = self.get_tag5_bearing_error()

            if bearing_rad is not None:
                deg_off = math.degrees(abs(bearing_rad))
                self.get_logger().info(
                    f'Intermediate AprilTag 5 measurement: offset={pixel_err:.1f} px, '
                    f'angle off={deg_off:.1f} deg.'
                )

                if (abs(pixel_err) <= self.tag1_align_pixel_tolerance or
                        abs(bearing_rad) <= self.tag1_align_tolerance_rad):
                    self.stop_robot()
                    z = self.get_tag5_z_distance()
                    if z is not None:
                        remaining_dist = z - self.target_z_origin
                        self.get_logger().info(
                            f'Centered towards AprilTag 5! Tag 5 z={z:.3f} m, '
                            f'target z={self.target_z_origin:.3f} m. '
                            f'Localization distance to (0, 0): {remaining_dist:.3f} m.'
                        )
                        self.target_drive_dist = max(0.0, remaining_dist)
                        self.distance_traveled = 0.0
                        self.state = 'driving_to_origin_remaining'
                        return
                    elif time_stopped > (self.settle_time_s + 2.0):
                        self.get_logger().warn(
                            'Tag 5 depth not resolved; using nominal 1.23 m remaining distance.'
                        )
                        self.target_drive_dist = 1.23
                        self.distance_traveled = 0.0
                        self.state = 'driving_to_origin_remaining'
                        return
                    return

                self.align_target_angle = abs(bearing_rad)
                self.align_direction = 1.0 if bearing_rad > 0 else -1.0
                self.align_traveled_angle = 0.0
                direction_str = 'CCW (left)' if self.align_direction > 0 else 'CW (right)'
                self.get_logger().info(
                    f'Re-centering at 1m mark: turning {deg_off:.1f} deg {direction_str} '
                    f'towards AprilTag 5...'
                )
                self.state = 'tag5_mid_rotating_to_align'
                return

            if time_stopped > (self.settle_time_s + 2.0):
                self.get_logger().warn(
                    'AprilTag 5 not seen at 1m mark; driving remaining 1.23 m towards (0, 0)...'
                )
                self.target_drive_dist = 1.23
                self.distance_traveled = 0.0
                self.state = 'driving_to_origin_remaining'
            return

        # Phase 23: Intermediate rotate to align with AprilTag 5, then settle 1s
        if self.state == 'tag5_mid_rotating_to_align':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.align_direction * self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = abs(v_ang) * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                traveled_deg = math.degrees(self.align_traveled_angle)
                self.get_logger().info(
                    f'Completed intermediate turn of {traveled_deg:.1f} deg towards Tag 5. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay before checking tag...'
                )
                self.state = 'tag5_mid_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 24: Drive remaining distance to (0, 0)
        if self.state == 'driving_to_origin_remaining':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed remaining drive of {self.distance_traveled:.3f} m to (0, 0). '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay to verify position...'
                )
                self.state = 'settling_and_checking_origin'
                self.stop_time = self.get_clock().now()
                return

            if self.target_drive_dist > 0.3:
                speed = 0.35 if dist_remaining > 0.15 else 0.18
            else:
                speed = 0.15

            twist = Twist()
            twist.linear.x = float(speed)
            twist.angular.z = 0.0  # Forward ONLY towards AprilTag 5
            self.cmd_pub.publish(twist)
            self.distance_traveled += speed * dt
            return

        # Phase 25: Settle 1s at (0, 0); verify position with Tag 5 z-distance
        if self.state == 'settling_and_checking_origin':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            z = self.get_tag5_z_distance()

            if z is not None:
                error = z - self.target_z_origin
                self.get_logger().info(
                    f'Stationary AprilTag 5 z distance: {z:.3f} m '
                    f'(target: {self.target_z_origin:.3f} m, error: {error:+.3f} m).'
                )

                if abs(error) <= self.tolerance:
                    self.get_logger().info(
                        f'Waypoint (0, 0) reached within tolerance! '
                        f'Final tag z = {z:.3f} m (error: {error:+.3f} m). '
                        f'Step 7: Turning 120 degrees CCW towards AprilTag 0...'
                    )
                    self.stop_robot()
                    self.align_target_angle = math.radians(120.0)
                    self.align_direction = 1.0  # CCW
                    self.align_traveled_angle = 0.0
                    self.state = 'turn_120_ccw_to_tag0'
                    return
                elif error > self.tolerance:
                    self.get_logger().info(
                        f'Robot is {error:.3f} m short of (0, 0). '
                        f'Moving forward {error:.3f} m to reach position...'
                    )
                    self.target_drive_dist = error
                    self.adjust_direction = 1.0
                    self.distance_traveled = 0.0
                    self.state = 'adjusting_origin'
                    return
                else:
                    overshoot = abs(error)
                    self.get_logger().info(
                        f'Robot overshot (0, 0) by {overshoot:.3f} m. '
                        f'Moving backward {overshoot:.3f} m to reach position...'
                    )
                    self.target_drive_dist = overshoot
                    self.adjust_direction = -1.0
                    self.distance_traveled = 0.0
                    self.state = 'adjusting_origin'
                    return

            if time_stopped > (self.settle_time_s + 2.0):
                self.get_logger().warn(
                    'AprilTag 5 not detected during origin check. '
                    'Proceeding to turn 120 degrees CCW towards AprilTag 0...'
                )
                self.stop_robot()
                self.align_target_angle = math.radians(100.0)
                self.align_direction = 1.0  # CCW
                self.align_traveled_angle = 0.0
                self.state = 'turn_120_ccw_to_tag0'
            return

        # Phase 26: Fine adjustment forward or backward to reach (0, 0)
        if self.state == 'adjusting_origin':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                direction_label = 'forward' if self.adjust_direction > 0 else 'backward'
                self.get_logger().info(
                    f'Completed {direction_label} adjustment of '
                    f'{self.distance_traveled:.3f} m. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay...'
                )
                self.state = 'settling_and_checking_origin'
                self.stop_time = self.get_clock().now()
                return

            twist = Twist()
            twist.linear.x = float(0.15 * self.adjust_direction)
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.distance_traveled += 0.15 * dt
            return

        # Phase 27: Turn 120 degrees CCW towards AprilTag 0
        if self.state == 'turn_120_ccw_to_tag0':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = v_ang * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                self.get_logger().info(
                    f'Turned 120 degrees CCW. Stopping for {self.settle_time_s:.1f}s '
                    f'delay before fine-adjusting to AprilTag 0...'
                )
                self.state = 'tag0_final_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return

        # Phase 28: Settle 1s and fine-adjust towards AprilTag 0
        if self.state == 'tag0_final_settling_and_measuring':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped < self.settle_time_s:
                return

            self.request_localization_service()
            bearing_rad, pixel_err = self.get_tag0_bearing_error()
            if bearing_rad is not None:
                deg_off = math.degrees(abs(bearing_rad))
                self.get_logger().info(
                    f'Stationary AprilTag 0 measurement: pixel offset={pixel_err:.1f} px, '
                    f'angle off={deg_off:.1f} deg.'
                )

                if (abs(pixel_err) <= self.tag1_align_pixel_tolerance or
                        abs(bearing_rad) <= self.tag1_align_tolerance_rad):
                    self.stop_robot()
                    self.get_logger().info(
                        f'AprilTag 0 is centered! Final offset: {pixel_err:.1f} px, '
                        f'{deg_off:.1f} deg <= {self.tag1_align_tolerance_deg:.1f} deg. '
                        f'Navigation mission successfully complete!'
                    )
                    self.state = 'done'
                    return

                self.align_target_angle = abs(bearing_rad)
                self.align_direction = 1.0 if bearing_rad > 0 else -1.0
                self.align_traveled_angle = 0.0
                direction_str = 'CCW (left)' if self.align_direction > 0 else 'CW (right)'
                self.get_logger().info(
                    f'Fine-adjusting: turning {deg_off:.1f} degrees {direction_str} '
                    f'towards AprilTag 0...'
                )
                self.state = 'tag0_final_rotating_to_align'
                return

            if time_stopped > (self.settle_time_s + 3.0):
                self.get_logger().info(
                    'AprilTag 0 not detected after 120-deg turn. Navigation complete.'
                )
                self.stop_robot()
                self.state = 'done'
            return

        # Phase 29: Rotate ONLY target angle towards AprilTag 0, then stop and wait 1s
        if self.state == 'tag0_final_rotating_to_align':
            twist = Twist()
            twist.linear.x = 0.0
            v_ang = float(self.align_direction * self.angular_speed_ccw)
            twist.angular.z = v_ang
            step_dt = abs(v_ang) * dt
            self.align_traveled_angle += step_dt
            self.cmd_pub.publish(twist)

            if self.align_traveled_angle >= self.align_target_angle:
                self.stop_robot()
                traveled_deg = math.degrees(self.align_traveled_angle)
                self.get_logger().info(
                    f'Completed fine rotation of {traveled_deg:.1f} deg towards Tag 0. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay before checking tag...'
                )
                self.state = 'tag0_final_settling_and_measuring'
                self.stop_time = self.get_clock().now()
            return


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
