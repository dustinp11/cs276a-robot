"""Waypoint tracking to (1, 0), align to AprilTag 1, and traverse to (1, 2).

Controls the robot through three sequential steps:
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

3. Step 3: Forward navigation to (1.0, 2.0) towards AprilTag 1:
   - Moves forward 2.0 m towards AprilTag 1 via dead reckoning.
   - Halts and settles 1 second to check for AprilTag 1.
   - If AprilTag 1 is still visible: adjusts forward via AprilTag 1 z-distance
     until reaching the coordinate point (1, 2).
   - If AprilTag 1 is no longer visible: prints that it can no longer see the
     AprilTag, and ends navigation.
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
    """Load surveyed coordinates for Tag 0 (x) and Tag 1 (y) from landmarks.json."""
    tag0_x = 1.75
    tag1_y = 2.2
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
                        if lm.get('id') == 0:
                            tag0_x = float(lm.get('x_m', 1.75))
                        elif lm.get('id') == 1:
                            tag1_y = float(lm.get('y_m', 2.2))
                    return tag0_x, tag1_y
            except Exception:
                pass
    return tag0_x, tag1_y


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

        # Target definitions for waypoint 1 (1, 0) and waypoint 2 (1, 2)
        self.tag0_x, self.tag1_y = load_landmarks()
        self.target_x_wp1 = 1.0
        self.target_z_wp1 = self.tag0_x - self.target_x_wp1
        self.target_y_wp2 = 2.0
        self.target_z_wp2 = self.tag1_y - self.target_y_wp2

        self.tolerance = 0.035  # 3.5 cm tolerance
        self.settle_time_s = 1.0  # 1s delay for camera stream to stabilize

        # Calibrated parameters: on current surface, full 360-deg turn takes 3.75s
        self.turn_time_360_s = 3.75
        self.angular_speed_cw = 2.0 * math.pi / self.turn_time_360_s
        self.angular_speed_ccw = 2.0 * math.pi / self.turn_time_360_s
        self.step_angle_rad = 30.0 * math.pi / 180.0  # 30 degrees per search step
        self.turn_step_traveled = 0.0

        # AprilTag 1 fine-alignment tracking (ID 1 ONLY)
        self.latest_tag1_detection = None
        self.align_target_angle = 0.0
        self.align_traveled_angle = 0.0
        self.align_direction = 1.0
        # Increased tolerance to 10 deg (~86 px) so wheels don't get stuck on micro-angles
        self.tag1_align_tolerance_deg = 10.0
        self.tag1_align_tolerance_rad = math.radians(self.tag1_align_tolerance_deg)
        self.tag1_align_pixel_tolerance = 491.0 * math.tan(self.tag1_align_tolerance_rad)

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

        self.timer = self.create_timer(0.1, self.control_tick)
        self.get_logger().info(
            f'hw2_control initialized: Leg 1 to (1, 0), align Tag 1, Leg 2 to (1, 2). '
            f'Tag 0 x={self.tag0_x:.3f} m, Tag 1 y={self.tag1_y:.3f} m, '
            f'Settle delay={self.settle_time_s:.1f}s.'
        )

    def stop_robot(self):
        """Halt all robot movement immediately."""
        self.cmd_pub.publish(Twist())

    def on_detections(self, msg):
        """Track detections of AprilTag 1 ONLY; ignore all other tags."""
        for det in msg.detections:
            if det.id == 1:
                self.latest_tag1_detection = {
                    'cx': float(det.centre.x),
                    'cy': float(det.centre.y),
                    'time': time.monotonic(),
                }

    def on_pose(self, msg):
        """Cache pose from /robot_pose topic."""
        self.latest_robot_pose = msg
        self.latest_robot_pose_time = time.monotonic()

    def get_tag0_z_distance(self):
        """Lookup AprilTag 0 z-distance (depth in front of camera) via TF or pose."""
        # 1. Direct TF lookup: camera frame -> tag_0 frame
        candidate_cam_frames = ('camera_frame', 'camera_link', 'camera_optical_frame')
        candidate_tag_frames = ('tag_0', 'tag36h11:0', 'tag_36h11_0')
        for cam in candidate_cam_frames:
            for tag in candidate_tag_frames:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        cam, tag, rclpy.time.Time()
                    )
                    z = float(trans.transform.translation.z)
                    if math.isfinite(z) and 0.1 < z < 10.0:
                        return z
                except Exception:
                    continue

        # 2. Fallback: check /robot_pose topic if recent
        if self.latest_robot_pose is not None:
            if time.monotonic() - self.latest_robot_pose_time < 3.0:
                px = self.latest_robot_pose.pose.position.x
                if math.isfinite(px):
                    z = self.tag0_x - px
                    if 0.1 < z < 10.0:
                        return z

        return None

    def get_tag1_transform(self):
        """Lookup AprilTag 1 transform in TF buffer (ID 1 ONLY)."""
        candidate_cam_frames = ('camera_frame', 'camera_link', 'camera_optical_frame')
        candidate_tag_frames = ('tag_1', 'tag36h11:1', 'tag_36h11_1')
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

    def get_tag1_z_distance(self):
        """Lookup AprilTag 1 z-distance (depth in front of camera) via TF or pose."""
        # 1. Direct TF lookup: camera frame -> tag_1 frame
        tf_trans = self.get_tag1_transform()
        if tf_trans is not None:
            z = float(tf_trans.transform.translation.z)
            if math.isfinite(z) and 0.05 < z < 10.0:
                return z

        # 2. Fallback: check /robot_pose topic if recent
        if self.latest_robot_pose is not None:
            if time.monotonic() - self.latest_robot_pose_time < 2.0:
                py = self.latest_robot_pose.pose.position.y
                if math.isfinite(py):
                    z = self.tag1_y - py
                    if 0.05 < z < 10.0:
                        return z

        return None

    def get_tag1_bearing_error(self):
        """
        Calculate bearing error to AprilTag 1 in radians and pixel offset.
        Positive -> tag is to the left (robot needs to turn CCW).
        Negative -> tag is to the right (robot needs to turn CW).
        Returns (bearing_rad, pixel_error) or (None, None).
        """
        # 1. Check pixel detection on /detections (AprilTag 1 ONLY)
        now = time.monotonic()
        if (self.latest_tag1_detection is not None and
                now - self.latest_tag1_detection['time'] < 2.0):
            cx = self.latest_tag1_detection['cx']
            bearing_rad = math.atan2(640.0 - cx, 491.0)
            pixel_error = cx - 640.0
            return bearing_rad, pixel_error

        # 2. Check TF transform for tag_1 ONLY
        tf_trans = self.get_tag1_transform()
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
                        'Step 3: Moving forward 2.0 m towards (1, 2) via dead reckoning...'
                    )
                    self.target_drive_dist = 2.0  # From (1, 0) to (1, 2) is 2.0 m along heading
                    self.distance_traveled = 0.0
                    self.state = 'driving_to_wp2'
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

        # Phase 8: Drive forward 2.0 m to (1, 2) towards AprilTag 1 via dead reckoning
        if self.state == 'driving_to_wp2':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed dead reckoning drive of {self.distance_traveled:.3f} m to (1, 2). '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay to check AprilTag 1...'
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

        # Phase 9: Settle 1s and check if AprilTag 1 is still visible; adjust if visible, else end
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
                # Still sees AprilTag 1! Adjust the same way as waypoint 1
                error = z - self.target_z_wp2
                self.get_logger().info(
                    f'Stationary AprilTag 1 z distance: {z:.3f} m '
                    f'(target: {self.target_z_wp2:.3f} m, error: {error:+.3f} m).'
                )

                if abs(error) <= self.tolerance:
                    self.get_logger().info(
                        f'Waypoint (1, 2) reached within tolerance! '
                        f'Final tag z = {z:.3f} m (error: {error:+.3f} m). Navigation complete.'
                    )
                    self.stop_robot()
                    self.state = 'done'
                elif error > self.tolerance:
                    self.get_logger().info(
                        f'Robot is {error:.3f} m short of (1, 2). '
                        f'Moving forward {error:.3f} m to reach position...'
                    )
                    self.target_drive_dist = error
                    self.distance_traveled = 0.0
                    self.state = 'adjusting_forward_wp2'
                else:
                    self.get_logger().info(
                        f'Robot reached (1, 2) (tag z={z:.3f} m <= {self.target_z_wp2:.3f} m). '
                        f'Stopping at goal. Navigation complete.'
                    )
                    self.stop_robot()
                    self.state = 'done'
                return

            if time_stopped > (self.settle_time_s + 1.5):
                print('Can no longer see AprilTag 1.')
                self.get_logger().info('Can no longer see AprilTag 1. Ending navigation.')
                self.stop_robot()
                self.state = 'done'
            return

        # Phase 10: Fine forward adjustment to (1, 2)
        if self.state == 'adjusting_forward_wp2':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed adjustment of {self.distance_traveled:.3f} m. '
                    f'Stopping for {self.settle_time_s:.1f}s settling delay...'
                )
                self.state = 'settling_and_checking_wp2'
                self.stop_time = self.get_clock().now()
                return

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            self.distance_traveled += 0.15 * dt
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
