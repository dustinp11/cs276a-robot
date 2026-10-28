"""Map-frame waypoint tracking with stop-and-measure AprilTag localization.

This node controls the robot to move from (0, 0) to (1, 0):
1. First, while stationary, query hw2_localization (via service /get_robot_pose or
   topic /robot_pose) to determine the current robot pose in the map frame.
2. If already at (1, 0), stop and end.
3. Otherwise, drive towards (1, 0) using the 3-stage control loop (rotate_to_goal ->
   drive -> rotate_to_orient) and dead reckoning from hw1_solution.
4. During motion, do not run hw2_localization since the camera has trouble
   detecting AprilTags while moving.
5. Once the HW1 motion loop finishes, stop the robot completely and query
   hw2_localization again while stationary.
6. If not yet at (1, 0), repeat motion using the newly localized position as the
   starting state until hw2_localization confirms the robot is at (1, 0).
"""

from math import atan2, cos, hypot, isfinite, pi, sin
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

try:
    from hw2_interfaces.srv import GetRobotPose
except ImportError:
    GetRobotPose = None


TARGET_WAYPOINT = (1.0, 0.0, 0.0)


def yaw_from_pose(pose):
    """Read planar yaw in radians from a PoseStamped unit quaternion."""
    q = pose.pose.orientation
    return atan2(2.0 * (q.w * q.z + q.x * q.y),
                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PIDcontroller:
    """PID controller for differential drive robot from hw1_solution."""

    def __init__(self, Kp, Ki, Kd, maximum_value=0.567333):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = [0.0, 0.0, 0.0]
        self.I = [0.0, 0.0]
        self.lastError = [0.0, 0.0]
        self.timestep = 0.1
        self.maximumValue = maximum_value

    def setTarget(self, state):
        """Set target state and reset integral/derivative accumulators."""
        self.I = [0.0, 0.0]
        self.lastError = [0.0, 0.0]
        self.target = list(state)

    def getError(self, currentState, targetState, drive_backwards):
        """Return distance and heading error between current and target state."""
        delta_x = targetState[0] - currentState[0]
        delta_y = targetState[1] - currentState[1]
        distance = hypot(delta_x, delta_y)
        angle_to_target = atan2(delta_y, delta_x)

        if drive_backwards:
            desired_heading = angle_to_target + pi
            desired_heading = (desired_heading + pi) % (2.0 * pi) - pi
            distance = -distance
        else:
            desired_heading = angle_to_target

        heading_error = desired_heading - currentState[2]
        heading_error = (heading_error + pi) % (2.0 * pi) - pi

        if abs(distance) < 0.05:
            heading_error = targetState[2] - currentState[2]
            heading_error = (heading_error + pi) % (2.0 * pi) - pi
            distance = 0.0

        return [distance, heading_error]

    def setMaximumUpdate(self, mv):
        """Set maximum linear velocity magnitude."""
        self.maximumValue = mv

    def update(self, currentState, drive_backwards=False):
        """Calculate linear and angular control updates."""
        e = self.getError(currentState, self.target, drive_backwards)

        P = [self.Kp * e[0], self.Kp * e[1]]
        self.I[0] += self.Ki * e[0] * self.timestep
        self.I[1] += self.Ki * e[1] * self.timestep
        D = [
            self.Kd * (e[0] - self.lastError[0]),
            self.Kd * (e[1] - self.lastError[1]),
        ]
        result = [P[0] + self.I[0] + D[0], P[1] + self.I[1] + D[1]]

        self.lastError = [e[0], e[1]]

        if abs(result[0]) > self.maximumValue:
            result[0] = (1.0 if result[0] > 0 else -1.0) * self.maximumValue

        max_angular = 1.5  # rad/s
        if abs(result[1]) > max_angular:
            result[1] = (1.0 if result[1] > 0 else -1.0) * max_angular

        if abs(e[0]) < 0.05:
            result[0] = 0.0
        return result


class ControlNode(Node):
    """Coordinates stationary visual localization with dead-reckoning HW1 motion."""

    def __init__(self):
        super().__init__('hw2_control')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot_pose', self.on_pose, 1
        )

        # Connect to /get_robot_pose service if interface is available
        self.pose_client = None
        if GetRobotPose is not None:
            self.pose_client = self.create_client(GetRobotPose, '/get_robot_pose')
        else:
            self.get_logger().info('hw2_interfaces not found; relying on /robot_pose topic.')

        self.declare_parameter('max_pose_age_s', 0.5)
        self.max_pose_age_s = float(self.get_parameter('max_pose_age_s').value)
        self.declare_parameter('pose_request_timeout_s', 1.5)
        self.request_timeout_s = float(
            self.get_parameter('pose_request_timeout_s').value
        )

        # Robot parameters from hw1_solution
        self.forward_speed = 0.567333
        self.angular_speed_cw = 2.0 * pi / 4.5
        self.angular_speed_ccw = 2.0 * pi / 5.2
        self.tolerance = 0.1
        self.angle_tolerance = 0.1
        self.settle_time_s = 0.5  # Pause to ensure vehicle is stationary before camera capture

        self.target_point = list(TARGET_WAYPOINT)
        self.current_state = [0.0, 0.0, 0.0]  # Assumed start at (0, 0, 0)
        self.drive_backwards = False  # Keep front camera pointed toward AprilTag 0

        self.pid = PIDcontroller(
            1.0, 0.01, 0.005, maximum_value=self.forward_speed
        )
        self.pid.setTarget(self.target_point)

        # High-level state: 'localizing' -> 'driving' -> 'done'
        self.state = 'localizing'
        # HW1 motion sub-stage: 'rotate_to_goal' -> 'drive' -> 'rotate_to_orient'
        self.hw1_stage = 'rotate_to_goal'

        self.pose_future = None
        self.request_started = 0.0
        self.stop_time = self.get_clock().now()
        self.last_tick = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.control_tick)

        self.get_logger().info(
            'hw2_control initialized: Target is (1, 0). '
            'Stationary localization enabled.'
        )

    def stop_robot(self):
        """Command zero velocity to halt all robot movement."""
        self.cmd_pub.publish(Twist())

    def on_pose(self, pose):
        """Receive pose from /robot_pose topic (used only when stationary)."""
        if self.state == 'localizing':
            now = self.get_clock().now()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                self.process_localized_pose(pose)

    def request_localization_service(self):
        """Request fresh pose from /get_robot_pose service while stationary."""
        if self.pose_client is None:
            return

        if self.pose_future is not None:
            if time.monotonic() - self.request_started > self.request_timeout_s:
                self.get_logger().warn('Localization request timed out; retrying...')
                future = self.pose_future
                self.pose_future = None
                try:
                    self.pose_client.remove_pending_request(future)
                except Exception:
                    pass
                future.cancel()
            return

        if not self.pose_client.service_is_ready():
            return

        self.request_started = time.monotonic()
        self.pose_future = self.pose_client.call_async(GetRobotPose.Request())
        self.pose_future.add_done_callback(self.on_pose_service_response)

    def on_pose_service_response(self, future):
        """Process response from /get_robot_pose service."""
        if future is not self.pose_future:
            return
        self.pose_future = None
        if future.cancelled():
            return
        try:
            response = future.result()
        except Exception as exception:
            self.get_logger().error(f'Pose request failed: {exception}')
            return

        if response.success:
            if self.state == 'localizing':
                self.process_localized_pose(response.pose)
        else:
            self.get_logger().debug(f'Localization waiting: {response.message}')

    def process_localized_pose(self, pose):
        """Check position reported by hw2_localization against goal (1, 0)."""
        p, q = pose.pose.position, pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(isfinite(v) for v in values):
            return

        x = float(p.x)
        y = float(p.y)
        yaw = float(yaw_from_pose(pose))

        dist_to_goal = hypot(self.target_point[0] - x, self.target_point[1] - y)
        self.get_logger().info(
            f'Measured pose via hw2_localization: pos=({x:.3f}, {y:.3f}), '
            f'yaw={yaw:.3f} rad. Distance to (1, 0): {dist_to_goal:.3f} m.'
        )

        if dist_to_goal <= self.tolerance:
            self.get_logger().info(
                f'Robot reached target (1.0, 0.0) within tolerance: '
                f'{dist_to_goal:.3f} m <= {self.tolerance:.3f} m. Complete!'
            )
            self.state = 'done'
            self.stop_robot()
        else:
            self.get_logger().info(
                f'Robot not yet at (1, 0) ({dist_to_goal:.3f} m > {self.tolerance:.3f} m). '
                f'Starting HW1 control loop to drive to (1, 0)...'
            )
            self.current_state = [x, y, yaw]
            self.pid.setTarget(self.target_point)
            self.hw1_stage = 'rotate_to_goal'
            self.state = 'driving'

    def get_desired_heading_to_goal(self, target_wp, drive_backwards):
        """Calculate desired planar heading angle to target position."""
        delta_x = target_wp[0] - self.current_state[0]
        delta_y = target_wp[1] - self.current_state[1]
        angle_to_target = atan2(delta_y, delta_x)

        if drive_backwards:
            desired_heading = angle_to_target + pi
            desired_heading = (desired_heading + pi) % (2.0 * pi) - pi
        else:
            desired_heading = angle_to_target

        return desired_heading

    def get_rotation_direction(self, heading_error):
        """Calibrated rotational velocity based on sign of heading error."""
        if heading_error > 0:
            return self.angular_speed_ccw
        else:
            return -self.angular_speed_cw

    def update_dead_reckoning(self, linear_vel, angular_vel, dt):
        """Propagate state estimate between visual localization updates."""
        self.current_state[0] += linear_vel * cos(self.current_state[2]) * dt
        self.current_state[1] += linear_vel * sin(self.current_state[2]) * dt
        self.current_state[2] += angular_vel * dt
        self.current_state[2] = (self.current_state[2] + pi) % (2.0 * pi) - pi

    def control_tick(self):
        """Main periodic control loop."""
        now = self.get_clock().now()
        dt = (now - self.last_tick).nanoseconds / 1e9
        self.last_tick = now
        if dt <= 0.0 or dt > 0.5:
            dt = 0.1

        if self.state == 'done':
            self.stop_robot()
            return

        # 1. State: Stationary localization check
        if self.state == 'localizing':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            # Ensure robot has stopped moving before querying camera detection
            if time_stopped >= self.settle_time_s:
                self.request_localization_service()
            return

        # 2. State: Driving via HW1 control loop (NO localization while in motion)
        if self.state == 'driving':
            self.execute_hw1_step(dt)

    def execute_hw1_step(self, dt):
        """Step the HW1 3-stage control loop and dead reckoning."""
        target_wp = self.target_point
        delta_x = target_wp[0] - self.current_state[0]
        delta_y = target_wp[1] - self.current_state[1]
        position_error = hypot(delta_x, delta_y)
        twist_msg = Twist()

        # Stage 1: Rotate to face the goal
        if self.hw1_stage == 'rotate_to_goal':
            if position_error < self.tolerance:
                self.hw1_stage = 'rotate_to_orient'
                twist_msg.angular.z = 0.0
            else:
                desired_heading = self.get_desired_heading_to_goal(
                    target_wp, self.drive_backwards
                )
                heading_error = desired_heading - self.current_state[2]
                heading_error = (heading_error + pi) % (2.0 * pi) - pi

                if abs(heading_error) < 0.05:
                    self.hw1_stage = 'drive'
                    twist_msg.angular.z = 0.0
                else:
                    twist_msg.angular.z = float(self.get_rotation_direction(heading_error))

        # Stage 2: Drive towards the goal
        elif self.hw1_stage == 'drive':
            position_error = hypot(delta_x, delta_y)
            if position_error < self.tolerance:
                self.hw1_stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            else:
                update_value = self.pid.update(self.current_state, self.drive_backwards)
                twist_msg.linear.x = float(update_value[0])

        # Stage 3: Rotate to target orientation
        elif self.hw1_stage == 'rotate_to_orient':
            heading_error = target_wp[2] - self.current_state[2]
            heading_error = (heading_error + pi) % (2.0 * pi) - pi

            if abs(heading_error) < self.angle_tolerance:
                # HW1 loop has completed for this run!
                # Stop robot and switch back to stationary localization check.
                self.get_logger().info(
                    'HW1 loop reached goal via dead reckoning. '
                    'Stopping robot to verify position via hw2_localization...'
                )
                self.stop_robot()
                self.state = 'localizing'
                self.stop_time = self.get_clock().now()
                return
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))

        self.update_dead_reckoning(twist_msg.linear.x, twist_msg.angular.z, dt)
        self.cmd_pub.publish(twist_msg)


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
