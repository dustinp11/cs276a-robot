"""ROS 2 Navigation Node for Closed-Loop Square Trajectory.

Consumes estimated robot pose from /ekf_slam/pose and commands /cmd_vel
using closed-loop feedback through all 4 waypoints of a 7 ft x 7 ft square.
"""

import csv
import math
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from hw3.square_navigation import NavigationState, SquareNavigator


class SquareNavigationNode(Node):
    """Executes closed-loop waypoint square traversal using EKF-SLAM pose feedback."""

    def __init__(self):
        super().__init__('square_navigation_node')

        # Declare parameters
        self.declare_parameter('square_size_ft', 7.0)
        self.declare_parameter('max_linear_velocity', 0.20)
        self.declare_parameter('min_linear_velocity', 0.05)
        self.declare_parameter('angular_velocity_ccw', 1.77)
        self.declare_parameter('angular_velocity_cw', 1.75)
        self.declare_parameter('max_angular_velocity', 1.77)
        self.declare_parameter('min_angular_velocity', 0.20)
        self.declare_parameter('linear_kp', 0.60)
        self.declare_parameter('angular_kp', 1.20)
        self.declare_parameter('waypoint_tolerance_m', 0.15)
        self.declare_parameter('heading_tolerance_rad', 0.12)
        self.declare_parameter('heading_drive_threshold_rad', 0.45)
        self.declare_parameter('turn_step_deg', 30.0)
        self.declare_parameter('turn_settle_time_s', 1.0)
        self.declare_parameter('pose_timeout_s', 0.60)
        self.declare_parameter('max_position_uncertainty_m', 0.40)
        self.declare_parameter('max_heading_uncertainty_rad', 0.40)
        self.declare_parameter('control_frequency_hz', 10.0)
        self.declare_parameter('auto_start', False)
        self.declare_parameter('output_dir', 'hw3_results')
        self.declare_parameter('log_csv', True)

        # Initialize SquareNavigator controller
        self.navigator = SquareNavigator(
            square_size_ft=self.get_parameter('square_size_ft').value,
            max_linear_v_mps=self.get_parameter('max_linear_velocity').value,
            min_linear_v_mps=self.get_parameter('min_linear_velocity').value,
            angular_velocity_ccw_radps=self.get_parameter('angular_velocity_ccw').value,
            angular_velocity_cw_radps=self.get_parameter('angular_velocity_cw').value,
            max_angular_omega_radps=self.get_parameter('max_angular_velocity').value,
            min_angular_omega_radps=self.get_parameter('min_angular_velocity').value,
            linear_kp=self.get_parameter('linear_kp').value,
            angular_kp=self.get_parameter('angular_kp').value,
            waypoint_tolerance_m=self.get_parameter('waypoint_tolerance_m').value,
            heading_tolerance_rad=self.get_parameter('heading_tolerance_rad').value,
            heading_drive_threshold_rad=self.get_parameter('heading_drive_threshold_rad').value,
            turn_step_deg=self.get_parameter('turn_step_deg').value,
            turn_settle_time_s=self.get_parameter('turn_settle_time_s').value,
            pose_timeout_s=self.get_parameter('pose_timeout_s').value,
            max_pos_uncertainty_m=self.get_parameter('max_position_uncertainty_m').value,
            max_heading_uncertainty_rad=self.get_parameter('max_heading_uncertainty_rad').value,
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/navigation/target_waypoint', 10)

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/ekf_slam/pose',
            self.on_pose_received,
            10
        )
        self.start_sub = self.create_subscription(
            Empty,
            '/start_navigation',
            self.on_start_topic,
            10
        )

        # Services
        self.start_srv = self.create_service(
            Trigger,
            '/start_navigation',
            self.on_start_service
        )
        self.stop_srv = self.create_service(
            Trigger,
            '/stop_navigation',
            self.on_stop_service
        )

        # Logging setup
        self.log_csv = self.get_parameter('log_csv').value
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.log_file_path = self.output_dir / 'navigation_log.csv'
        self.csv_initialized = False

        if self.log_csv:
            self._init_csv_log()

        # Control loop timer
        freq = max(1.0, float(self.get_parameter('control_frequency_hz').value))
        self.control_timer = self.create_timer(1.0 / freq, self.control_loop_tick)

        # Auto-start if requested
        if self.get_parameter('auto_start').value:
            self.get_logger().info("auto_start is True: starting square navigation immediately.")
            self.navigator.start()
        else:
            self.get_logger().info(
                "Square Navigation Node initialized in IDLE mode. "
                "Waiting for start command (run: 'ros2 service call /start_navigation std_srvs/srv/Trigger' "
                "or 'ros2 topic pub --once /start_navigation std_msgs/msg/Empty \"{}\"')."
            )

    def _init_csv_log(self):
        """Initialize navigation telemetry CSV file."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp_s',
                    'waypoint_index',
                    'target_x_m',
                    'target_y_m',
                    'robot_x_m',
                    'robot_y_m',
                    'robot_theta_rad',
                    'distance_error_m',
                    'heading_error_rad',
                    'cmd_v_mps',
                    'cmd_omega_radps',
                    'state',
                    'status_message',
                ])
            self.csv_initialized = True
        except Exception as e:
            self.get_logger().warn(f"Failed to initialize navigation CSV log: {e}")

    def on_pose_received(self, msg: PoseWithCovarianceStamped):
        """Handle latest robot pose estimate from EKF-SLAM."""
        stamp = msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9

        # Extract planar position
        x_slam = msg.pose.pose.position.x
        y_slam = msg.pose.pose.position.y

        # Extract yaw from quaternion (x, y, z, w)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta_slam = math.atan2(siny_cosp, cosy_cosp)

        # Extract 1-sigma uncertainty from 6x6 covariance
        cov = msg.pose.covariance
        var_x = max(0.0, cov[0])
        var_y = max(0.0, cov[7])
        var_theta = max(0.0, cov[35])
        sigma_pos = math.sqrt(var_x + var_y)
        sigma_theta = math.sqrt(var_theta)

        # Forward to navigator
        self.navigator.update_ekf_pose(
            x_slam=x_slam,
            y_slam=y_slam,
            theta_slam=theta_slam,
            sigma_pos_m=sigma_pos,
            sigma_theta_rad=sigma_theta,
            stamp_s=stamp_s,
        )

    def on_start_topic(self, _msg: Empty):
        """Handle start trigger via topic publish."""
        success, message = self.navigator.start()
        if success:
            self.get_logger().info(f"Start topic received: {message}")
        else:
            self.get_logger().warn(f"Start topic rejected: {message}")

    def on_start_service(self, _request, response):
        """Handle start trigger via ROS 2 service call."""
        success, message = self.navigator.start()
        response.success = success
        response.message = message
        if success:
            self.get_logger().info(f"Start service called: {message}")
        else:
            self.get_logger().warn(f"Start service rejected: {message}")
        return response

    def on_stop_service(self, _request, response):
        """Handle stop trigger via ROS 2 service call."""
        success, message = self.navigator.stop("User requested stop via service")
        self._publish_stop()
        response.success = success
        response.message = message
        self.get_logger().info(f"Stop service called: {message}")
        return response

    def _publish_stop(self):
        """Publish zero velocity to halt motors immediately."""
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def control_loop_tick(self):
        """Execute one control cycle and publish motor commands."""
        now_s = self.get_clock().now().nanoseconds * 1e-9
        output = self.navigator.step(now_s)

        # 1. Publish /cmd_vel
        twist = Twist()
        twist.linear.x = float(output.linear_v_mps)
        twist.angular.z = float(output.angular_omega_radps)
        self.cmd_vel_pub.publish(twist)

        # 2. Publish status string
        status_msg = String()
        status_msg.data = f"[{output.state.value}] {output.status_message}"
        self.status_pub.publish(status_msg)

        # 3. Publish target waypoint
        wp_msg = PointStamped()
        wp_msg.header.stamp = self.get_clock().now().to_msg()
        wp_msg.header.frame_id = "map"
        wp_msg.point.x = output.target_x_m
        wp_msg.point.y = output.target_y_m
        wp_msg.point.z = 0.0
        self.waypoint_pub.publish(wp_msg)

        # 4. Log to CSV
        if self.csv_initialized:
            try:
                with open(self.log_file_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"{now_s:.3f}",
                        output.waypoint_index,
                        f"{output.target_x_m:.4f}",
                        f"{output.target_y_m:.4f}",
                        f"{output.robot_x_m:.4f}",
                        f"{output.robot_y_m:.4f}",
                        f"{output.robot_theta_rad:.4f}",
                        f"{output.distance_error_m:.4f}",
                        f"{output.heading_error_rad:.4f}",
                        f"{output.linear_v_mps:.4f}",
                        f"{output.angular_omega_radps:.4f}",
                        output.state.value,
                        output.status_message,
                    ])
            except Exception:
                pass

        # 5. Log periodically to terminal if active
        if output.state in (NavigationState.DRIVING, NavigationState.TURNING, NavigationState.SETTLING):
            self.get_logger().info(
                f"[{output.state.value} WP{output.waypoint_index}] dist={output.distance_error_m:.2f}m, "
                f"head_err={math.degrees(output.heading_error_rad):.1f}°, "
                f"cmd=(v={output.linear_v_mps:.2f}m/s, w={output.angular_omega_radps:.2f}rad/s)",
                throttle_duration_sec=1.0
            )
        elif output.state == NavigationState.COMPLETED:
            self.get_logger().info(
                "Square navigation completed! Robot safely halted.",
                throttle_duration_sec=5.0
            )

    def destroy_node(self):
        """Ensure robot is stopped when node is destroyed."""
        self._publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SquareNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Navigation interrupted by keyboard.")
    finally:
        node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
