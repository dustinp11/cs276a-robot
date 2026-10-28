"""Waypoint tracking to (1, 0) followed by AprilTag 1 search and fine alignment.

Controls the robot in two sequential steps:
1. Step 1: Forward-only navigation to (1.0, 0.0):
   - Initial stationary check: waits 3 seconds for camera to stabilize, reads
     AprilTag 0 z-distance (depth in front of camera).
   - Tag 0 is at x = 1.75 m; target depth at (1, 0) is z = 0.75 m.
   - Drives forward ONLY (angular.z strictly 0.0, no rotation).
   - Once dead reckoning reaches target distance, halts and settles 3 seconds.
   - Measures AprilTag 0 z-distance directly.
   - If short of (1, 0) (z > 0.75 m + tolerance), moves forward by the difference.
   - Repeats until tag z is within tolerance of 0.75 m.

2. Step 2: Turn CCW until directly facing AprilTag 1 (ID 1 ONLY):
   - Rotates CCW in 30-degree increments (calibrated angular speed: 3.75s per 360 deg).
   - Stops and waits 3 seconds every 30 degrees to check for AprilTag 1.
   - Once AprilTag 1 is seen, calculates the exact bearing angle it is off.
   - Turns ONLY the number of degrees it is off.
   - In between each fine align rotation, halts and waits 3 seconds for the
     camera stream to settle before measuring again.
   - Once directly facing AprilTag 1 (|error| <= 30 px / ~3.5 deg), halts and concludes.
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


def load_tag0_x():
    """Load AprilTag 0 surveyed x position from config/landmarks.json."""
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
                            return float(lm.get('x_m', 1.75))
            except Exception:
                pass
    return 1.75


class ControlNode(Node):
    """Controls forward translation to (1, 0) and discrete alignment to AprilTag 1."""

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

        # Target definitions for waypoint 1
        self.tag0_x = load_tag0_x()
        self.target_x = 1.0
        # When robot is at target_x, depth to Tag 0 is (tag0_x - target_x)
        self.target_z = self.tag0_x - self.target_x
        self.tolerance = 0.035  # 3.5 cm tolerance
        self.settle_time_s = 3.0  # 3s delay for camera stream to stabilize

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
            f'hw2_control initialized: Forward to (1, 0), then 30-deg search & align to Tag 1. '
            f'Angular speed: {self.angular_speed_ccw:.3f} rad/s (3.75s per 360 deg).'
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
            # Camera optical center cx = 640.0, fx = 491.0
            # If cx < 640, tag is to the left in image -> turn CCW (positive)
            bearing_rad = math.atan2(640.0 - cx, 491.0)
            pixel_error = cx - 640.0
            return bearing_rad, pixel_error

        # 2. Check TF transform for tag_1 ONLY
        tf_trans = self.get_tag1_transform()
        if tf_trans is not None:
            lx = tf_trans.transform.translation.x
            lz = tf_trans.transform.translation.z
            if math.isfinite(lx) and math.isfinite(lz) and lz > 0.05:
                # In camera optical frame, +X is right, +Z is forward
                # If lx < 0 (left), turn CCW (positive)
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
        """Main periodic control loop executing forward drive and AprilTag 1 alignment."""
        now = self.get_clock().now()
        dt = (now - self.last_tick).nanoseconds / 1e9
        self.last_tick = now
        if dt <= 0.0 or dt > 0.5:
            dt = 0.1

        if self.state == 'done':
            self.stop_robot()
            return

        # Phase 1: Initial stationary settling and distance determination
        if self.state == 'initial_settling':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                self.request_localization_service()
                z = self.get_tag0_z_distance()
                if z is not None:
                    drive_dist = z - self.target_z
                    self.get_logger().info(
                        f'Initial AprilTag 0 z distance: {z:.3f} m. '
                        f'Target z: {self.target_z:.3f} m. '
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

        # Phase 2: Forward motion only (strictly angular.z = 0.0, no rotation)
        if self.state == 'driving_forward':
            dist_remaining = self.target_drive_dist - self.distance_traveled
            if dist_remaining <= 0.005:
                self.stop_robot()
                self.get_logger().info(
                    f'Completed forward drive of {self.distance_traveled:.3f} m. '
                    f'Stopping for 3s settling delay before checking tag 0...'
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
                error = z - self.target_z  # Positive means robot is short (too far from tag)
                self.get_logger().info(
                    f'Stationary AprilTag 0 z distance: {z:.3f} m '
                    f'(target: {self.target_z:.3f} m, error: {error:+.3f} m).'
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
                        f'Robot reached (1, 0) (tag z={z:.3f} m <= {self.target_z:.3f} m). '
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
                    'Turned 30 degrees CCW. Stopping for 3s delay to check for AprilTag 1...'
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
                    f'Settling 3s before starting fine alignment...'
                )
                self.stop_robot()
                self.state = 'tag1_settling_and_measuring'
                self.stop_time = self.get_clock().now()
                return

            if time_stopped > (self.settle_time_s + 2.0):
                self.get_logger().info(
                    'AprilTag 1 not visible at this angle after 3s delay. '
                    'Turning CCW another 30 degrees...'
                )
                self.turn_step_traveled = 0.0
                self.state = 'turn_step_ccw_30'
            return

        # Phase 6: Settle 3s and measure exact angle off from AprilTag 1 (ID 1 ONLY)
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

                # Check if directly facing AprilTag 1 (within 30 px or 3.5 deg)
                if abs(pixel_err) <= 30.0 or abs(bearing_rad) <= math.radians(3.5):
                    self.stop_robot()
                    self.get_logger().info(
                        f'Directly facing AprilTag 1! (offset: {pixel_err:.1f} px, '
                        f'{deg_off:.1f} deg). Navigation complete.'
                    )
                    self.state = 'done'
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

            # If tag not seen yet during stationary settling, wait a moment longer
            if time_stopped > (self.settle_time_s + 3.0):
                self.get_logger().warn(
                    'Lost AprilTag 1 during settling. Resuming 30-degree CCW search...'
                )
                self.turn_step_traveled = 0.0
                self.state = 'turn_step_ccw_30'
            return

        # Phase 7: Rotate ONLY the target angle, then stop and wait 3s
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
                    f'Stopping for 3s settling delay before checking AprilTag 1 again...'
                )
                self.state = 'tag1_settling_and_measuring'
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
