"""Waypoint tracking to (1, 0) followed by 90-degree CCW turn from hw1_solution.

Controls the robot in two sequential steps:
1. Step 1: Forward-only navigation to (1.0, 0.0):
   - Initial stationary check: waits 5 seconds for camera to stabilize, reads
     AprilTag 0 z-distance (depth in front of camera).
   - Tag 0 is at x = 1.75 m; target depth at (1, 0) is z = 0.75 m.
   - Drives forward ONLY (angular.z strictly 0.0, no rotation).
   - Once dead reckoning reaches target distance, halts and settles 5 seconds.
   - Measures AprilTag 0 z-distance directly.
   - If short of (1, 0) (z > 0.75 m + tolerance), moves forward by the difference.
   - Repeats until tag z is within tolerance of 0.75 m.

2. Step 2: Turn 90 degrees CCW:
   - Rotates in-place 90 degrees CCW using the calibrated dead reckoning rotation
     control from hw1_solution (angular_speed_ccw = 2 * pi / 5.2 rad/s).
   - Halts and ends navigation when the 90-degree turn is complete.
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
    """Controls forward-only translation to (1, 0) using AprilTag 0 z-distance."""

    def __init__(self):
        super().__init__('hw2_control')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.on_pose, 1
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_client = None
        if GetRobotPose is not None:
            self.pose_client = self.create_client(GetRobotPose, '/get_robot_pose')
        else:
            self.get_logger().info('hw2_interfaces not found; relying on TF and /robot_pose.')

        # Target definitions
        self.tag0_x = load_tag0_x()
        self.target_x = 1.0
        # When robot is at target_x, depth to Tag 0 is (tag0_x - target_x)
        self.target_z = self.tag0_x - self.target_x
        self.tolerance = 0.035  # 3.5 cm tolerance
        self.settle_time_s = 5.0  # 5s delay before polling localization for camera catchup

        # Calibrated parameters from robot_test.py / hw1_solution
        self.angular_speed_cw = 2.0 * math.pi / 4.5
        self.angular_speed_ccw = 2.0 * math.pi / 5.2
        self.angle_tolerance = 0.1
        self.current_yaw = 0.0
        self.target_yaw = math.pi / 2.0

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
            f'hw2_control initialized: Forward-only to (1, 0). '
            f'Tag 0 x={self.tag0_x:.3f} m, Target Tag z={self.target_z:.3f} m, '
            f'Tolerance={self.tolerance:.3f} m.'
        )

    def stop_robot(self):
        """Halt all robot movement immediately."""
        self.cmd_pub.publish(Twist())

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

    def get_rotation_direction(self, heading_error):
        """Calibrated rotational velocity from hw1_solution."""
        if heading_error > 0:
            return self.angular_speed_ccw
        else:
            return -self.angular_speed_cw

    def update_dead_reckoning(self, linear_vel, angular_vel, dt):
        """Propagate orientation estimate using dead reckoning from hw1_solution."""
        self.current_yaw += angular_vel * dt
        self.current_yaw = (self.current_yaw + math.pi) % (2.0 * math.pi) - math.pi

    def control_tick(self):
        """Main periodic control loop executing forward navigation and 90-deg CCW turn."""
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
                    f'Stopping for 5s settling delay before checking tag...'
                )
                self.state = 'settling_and_checking'
                self.stop_time = self.get_clock().now()
                return

            # Choose appropriate speed: moderate for long run, gentle for fine adjustments
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

        # Phase 3: Stationary verification and fine adjustment using AprilTag z-distance
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
                        f'Step 2: Turning 90 degrees CCW via hw1_solution rotation...'
                    )
                    self.stop_robot()
                    self.current_yaw = 0.0
                    self.target_yaw = math.pi / 2.0
                    self.state = 'turning_90_ccw'
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
                        f'Step 2: Turning 90 degrees CCW via hw1_solution rotation...'
                    )
                    self.stop_robot()
                    self.current_yaw = 0.0
                    self.target_yaw = math.pi / 2.0
                    self.state = 'turning_90_ccw'
                return

            if time_stopped > (self.settle_time_s + 5.0):
                self.get_logger().warn(
                    'AprilTag 0 not detected during settling check. Continuing to poll...'
                )
            return

        # Phase 4: Turn 90 degrees CCW based on hw1_solution dead reckoning rotation
        if self.state == 'turning_90_ccw':
            heading_error = self.target_yaw - self.current_yaw
            heading_error = (heading_error + math.pi) % (2.0 * math.pi) - math.pi

            if abs(heading_error) < self.angle_tolerance:
                self.stop_robot()
                self.get_logger().info(
                    f'Turned 90 degrees CCW (yaw={self.current_yaw:.3f} rad). '
                    f'Navigation complete.'
                )
                self.state = 'done'
                return

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = float(self.get_rotation_direction(heading_error))
            self.update_dead_reckoning(0.0, twist.angular.z, dt)
            self.cmd_pub.publish(twist)
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
