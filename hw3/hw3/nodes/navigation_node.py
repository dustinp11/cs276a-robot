"""ROS 2 Navigation Node for Closed-Loop Square Trajectory.

Consumes estimated robot pose from /ekf_slam/pose and commands /cmd_vel
using fixed-heading configured-length legs, midpoint pauses, and direct quarter-turns.
"""

import csv
import json
import math
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Twist
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from hw3.square_navigation import NavigationState, SquareNavigator
from hw3.turn_calibration import read_forward_calibration, read_turn_calibration
from hw3.vision_navigation import NavigationPolicy
from dataclasses import fields


class SquareNavigationNode(Node):
    """Executes closed-loop waypoint square traversal using EKF-SLAM pose feedback."""

    def __init__(self):
        """Initialize calibrated square control and ROS mission interfaces.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        super().__init__('square_navigation_node')
        self.calibration = read_turn_calibration(self)
        self.forward_calibration = read_forward_calibration(self)
        self._declare_parameters()
        self._create_navigator()
        self._create_publishers()
        self._create_subscriptions()
        self._create_services()
        self._configure_logging()
        self._create_control_timer()
        self._configure_autostart()

    def _declare_parameters(self):
        """Declare navigation parameters under the navigation namespace.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        for field in fields(NavigationPolicy):
            self.declare_parameter('navigation.' + field.name, field.default)
        # declare parameters
        self.declare_parameter('navigation.square_size_ft', 6.0)
        self.declare_parameter('navigation.side_count', 4)
        self.declare_parameter('navigation.side_length_m', 0.0)
        self.declare_parameter('navigation.waypoint_tolerance_m', 0.01)
        self.declare_parameter('navigation.heading_tolerance_rad', 0.12)
        self.declare_parameter('navigation.pose_timeout_s', 0.60)
        self.declare_parameter('navigation.max_position_uncertainty_m', 0.40)
        self.declare_parameter('navigation.max_heading_uncertainty_rad', 0.40)
        self.declare_parameter('navigation.control_frequency_hz', 20.0)
        self.declare_parameter('navigation.auto_start', False)
        self.declare_parameter('navigation.output_dir', 'hw3_results')
        self.declare_parameter('navigation.log_csv', True)

        self.declare_parameter('navigation.turn_pulse_rad', math.pi / 12)
        self.declare_parameter('navigation.turn_settle_s', 3.0)

    def _create_navigator(self):
        """Build the pure square controller from shared calibration and parameters.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # initialize squarenavigator controller
        policy = NavigationPolicy(**{field.name: self.get_parameter('navigation.' + field.name).value
                                     for field in fields(NavigationPolicy)})
        self.navigator = SquareNavigator(
            policy=policy,
            side_count=self.get_parameter('navigation.side_count').value,
            side_length_m=self.get_parameter('navigation.side_length_m').value,
            turn_calibration=self.calibration,
            turn_pulse_rad=self.get_parameter('navigation.turn_pulse_rad').value,
            turn_settle_s=self.get_parameter('navigation.turn_settle_s').value,
            square_size_ft=self.get_parameter('navigation.square_size_ft').value,
            forward_calibration=self.forward_calibration,
            waypoint_tolerance_m=self.get_parameter('navigation.waypoint_tolerance_m').value,
            heading_tolerance_rad=self.get_parameter('navigation.heading_tolerance_rad').value,
            pose_timeout_s=self.get_parameter('navigation.pose_timeout_s').value,
            max_pos_uncertainty_m=self.get_parameter('navigation.max_position_uncertainty_m').value,
            max_heading_uncertainty_rad=self.get_parameter('navigation.max_heading_uncertainty_rad').value,
        )

    def _create_publishers(self):
        """Create velocity, status, and target waypoint publishers."""
        # publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)
        self.waypoint_pub = self.create_publisher(PointStamped, '/navigation/target_waypoint', 10)

    def _create_subscriptions(self):
        """Subscribe to estimator poses and mission start requests.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        self.vision_sub = self.create_subscription(
            String, '/ekf_slam/vision_feedback', self.on_vision_feedback, 10)
        # subscribers
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

    def _create_services(self):
        """Expose explicit mission start and stop services.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # services
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

    def _configure_logging(self):
        """Configure optional navigation telemetry recording."""
        # logging setup
        self.log_csv = self.get_parameter('navigation.log_csv').value
        self.output_dir = Path(self.get_parameter('navigation.output_dir').value)
        self.log_file_path = self.output_dir / 'navigation_log.csv'
        self.csv_initialized = False

        if self.log_csv:
            self._init_csv_log()

    def _create_control_timer(self):
        """Schedule control ticks at the configured frequency."""
        # control loop timer
        freq = max(1.0, float(self.get_parameter('navigation.control_frequency_hz').value))
        self.control_timer = self.create_timer(1.0 / freq, self.control_loop_tick)

    def _configure_autostart(self):
        """Optionally start navigation or explain how to start it.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        # auto-start if requested
        if self.get_parameter('navigation.auto_start').value:
            self.get_logger().info("auto_start is True: starting square navigation immediately.")
            self.navigator.start()
        else:
            self.get_logger().info(
                "Square Navigation Node initialized in IDLE mode. "
                "Waiting for start command (run: 'ros2 service call /start_navigation std_srvs/srv/Trigger' "
                "or 'ros2 topic pub --once /start_navigation std_msgs/msg/Empty \"{}\"')."
            )

    def _init_csv_log(self):
        """Initialize navigation telemetry CSV file.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
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
                    'vision_age_s', 'visual_progress_m', 'ekf_progress_m',
                    'forward_clearance_m', 'corner_nominal_rad', 'corner_correction_rad', 'corner_feedback_source', 'corner_residual_rad', 'navigation_heading_rad',
                    'midpoint_progress_m', 'midpoint_remaining_m', 'midpoint_source',
                    'midpoint_applied_correction_m',
                    'startup_scan_angle_rad', 'startup_scan_nominal_progress_rad', 'startup_correction_rad',
                    'startup_corrections', 'startup_feedback_source', 'startup_heading_residual_rad',
                    'startup_view_search_rad', 'startup_view_matched', 'startup_view_tag_ids',
                    'startup_reference_positions_m', 'startup_current_positions_m',
                    'midpoint_heading_source', 'midpoint_heading_residual_rad',
                    'midpoint_corrections', 'midpoint_correction_rad',
                    'late_look_angle_rad', 'late_scan_angle_rad',
                    'late_alignment_corrections', 'late_alignment_correction_rad',
                    'return_travel_m', 'estimated_start_distance_m',
                    'start_x_slam_m', 'start_y_slam_m',
                ])
            self.csv_initialized = True
        except Exception as e:
            self.get_logger().warn(f"Failed to initialize navigation CSV log: {e}")

    def on_pose_received(self, msg: PoseWithCovarianceStamped):
        """Handle latest robot pose estimate from EKF-SLAM.

        Args:
            msg: Incoming ROS message for this callback, with its original timestamp and frame.

        Returns:
            None. State changes and publications are performed in place.
        """
        stamp = msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9

        # extract planar position
        x_slam = msg.pose.pose.position.x
        y_slam = msg.pose.pose.position.y

        # extract yaw from quaternion (x, y, z, w)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta_slam = math.atan2(siny_cosp, cosy_cosp)

        # extract 1-sigma uncertainty from 6x6 covariance
        cov = msg.pose.covariance
        var_x = max(0.0, cov[0])
        var_y = max(0.0, cov[7])
        var_theta = max(0.0, cov[35])
        sigma_pos = math.sqrt(var_x + var_y)
        sigma_theta = math.sqrt(var_theta)

        # forward to navigator
        self.navigator.update_ekf_pose(
            x_slam=x_slam,
            y_slam=y_slam,
            theta_slam=theta_slam,
            sigma_pos_m=sigma_pos,
            sigma_theta_rad=sigma_theta,
            stamp_s=stamp_s,
        )

    def on_vision_feedback(self, message):
        """Validate stamped base-frame geometry before handing it to navigation.

        Args:
            message: JSON String from the EKF's accepted image callback.

        Returns:
            None. Malformed feedback is ignored and cannot refresh vision age.
        """
        try:
            data = json.loads(message.data)
            stamp = float(data['stamp_s'])
            tags = [[int(tag), float(x), float(y)] for tag, x, y in data['tags']]
            known_ids = [int(tag) for tag in data['known_ids']]
            tag_yaws = [[int(tag), float(yaw)] for tag, yaw in data.get('tag_yaws', [])]
            landmark_sigmas = [[int(tag), float(sigma)] for tag, sigma in data.get('landmark_sigmas', [])]
            ekf_heading = data.get('ekf_heading')
            ekf_heading = [float(v) for v in ekf_heading] if ekf_heading is not None else None
            if ekf_heading is not None and len(ekf_heading) != 2:
                return
            clearance = data['forward_clearance_m']
            values = [stamp] + [float(v) for tag in tags + tag_yaws + landmark_sigmas for v in tag] + (ekf_heading or [])
            if clearance is not None:
                values.append(float(clearance))
            if not all(math.isfinite(v) for v in values):
                return
            if any(tag < 0 for tag in known_ids) or any(tag[0] < 0 for tag in tags):
                return
            self.navigator.update_vision(stamp, tags, known_ids, clearance, tag_yaws,
                                         landmark_sigmas, ekf_heading)
        except (ValueError, TypeError, KeyError, OverflowError):
            self.get_logger().warn('Ignoring malformed vision feedback')

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
        """Execute one control cycle and publish motor commands.

        Args:
            None.

        Returns:
            None. State changes and publications are performed in place.
        """
        now_s = self.get_clock().now().nanoseconds * 1e-9
        output = self.navigator.step(now_s)

        self._publish_command(output)
        self._publish_status(output)
        self._publish_target(output)
        self._record_output(output, now_s)
        self._report_progress(output)

    def _publish_command(self, output):
        """Publish the controller’s planar velocity command.

        Args:
            output: NavigationOutput with positions in meters and angles in radians.

        Returns:
            None.
        """
        # 1. publish /cmd_vel
        twist = Twist()
        twist.linear.x = float(output.linear_v_mps)
        twist.angular.z = float(output.angular_omega_radps)
        self.cmd_vel_pub.publish(twist)


    def _publish_status(self, output):
        """Publish the mission state and reason.

        Args:
            output: NavigationOutput with positions in meters and angles in radians.

        Returns:
            None.
        """
        # 2. publish status string
        status_msg = String()
        status_msg.data = f"[{output.state.value}] {output.status_message}"
        self.status_pub.publish(status_msg)


    def _publish_target(self, output):
        """Publish the target in the SLAM map frame.

        Args:
            output: NavigationOutput with positions in meters and angles in radians.

        Returns:
            None.
        """
        # 3. publish target waypoint
        wp_msg = PointStamped()
        wp_msg.header.stamp = self.get_clock().now().to_msg()
        wp_msg.header.frame_id = "map"
        wp_msg.point.x, wp_msg.point.y = self.navigator.target
        wp_msg.point.z = 0.0
        self.waypoint_pub.publish(wp_msg)


    def _record_output(self, output, now_s):
        """Append one navigation telemetry row when recording is enabled.

        Args:
            output: NavigationOutput with positions in meters and angles in radians.
            now_s: Current ROS time in seconds.

        Returns:
            None.
        """
        # 4. log to csv
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
                        now_s - self.navigator.vision.valid_stamp,
                        self.navigator.vision.progress,
                        self.navigator._ekf_progress() if self.navigator.latest_slam_pose else '',
                        self.navigator.vision.clearance,
                        self.navigator.corner_used,
                        self.navigator.correction_used,
                        self.navigator.corner_source,
                        self.navigator.corner_residual,
                        self.navigator.navigation_yaw,
                        self.navigator.midpoint_progress_m,
                        self.navigator.midpoint_remaining_m,
                        self.navigator.midpoint_source,
                        self.navigator.progress_correction_m,
                        self.navigator.startup_scan_used,
                        self.navigator.startup_scan_nominal_progress,
                        self.navigator.startup_correction_used,
                        self.navigator.startup_correction_count,
                        self.navigator.startup_source,
                        self.navigator.startup_heading_residual,
                        self.navigator.startup_search_used,
                        self.navigator.startup_view_matched,
                        json.dumps(self.navigator.startup_view_ids),
                        json.dumps(self.navigator.startup_view_reference),
                        json.dumps(self.navigator.startup_view_current),
                        self.navigator.midpoint_heading_source,
                        self.navigator.midpoint_heading_residual,
                        self.navigator.midpoint_correction_count,
                        self.navigator.midpoint_correction_used,
                        self.navigator.late.scan_approach,
                        self.navigator.late.scan_used,
                        self.navigator.late.extra_count,
                        self.navigator.late.extra_used,
                        self.navigator.late.return_travel_m,
                        self.navigator.late._home_distance(),
                        self.navigator.x0_slam,
                        self.navigator.y0_slam,
                    ])
            except OSError as error:
                self.get_logger().warn(f'Navigation CSV write failed: {error}', throttle_duration_sec=5.0)


    def _report_progress(self, output):
        """Report active control errors or completion with throttling.

        Args:
            output: NavigationOutput with positions in meters and angles in radians.

        Returns:
            None.
        """
        # 5. log periodically to terminal if active
        if output.state in (NavigationState.DRIVING, NavigationState.TURNING, NavigationState.CORRECTING,
                            NavigationState.RELOCALIZING_TURN, NavigationState.RETURN_DRIVING, NavigationState.REORIENTING):
            self.get_logger().info(
                f"[WP{output.waypoint_index}] dist={output.distance_error_m:.2f}m, "
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
    """Run this module through its command-line entry point.

    Args:
        args: Optional command-line arguments; None uses the process arguments.

    Returns:
        None. State changes and publications are performed in place.
    """
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
