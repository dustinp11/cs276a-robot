"""Map-frame waypoint tracking with stop-and-measure AprilTag localization.

This node controls the robot through two sequential waypoints:
1. Move from (0, 0) to (1, 0):
   - Check initial position via hw2_localization while stationary.
   - Drive to (1, 0) using the HW1 3-stage controller (rotate_to_goal -> drive
     -> rotate_to_orient) and dead reckoning.
   - No camera localization while in motion.
   - Stop and verify position via hw2_localization, adjusting until (1, 0) is reached.

2. At (1, 0), find AprilTag 1:
   - When AprilTag 1 is not in view, spin CCW by 15 degrees, stop completely,
     and query hw2_localization for AprilTag 1.
   - Repeat spinning 15 degrees CCW, stopping, and querying until AprilTag 1 is seen.
   - Once seen, center AprilTag 1 in the camera view using calibrated rotation.

3. Traverse to (1, 2):
   - Drive to (1, 2) using HW1 controller and dead reckoning.
   - Stop completely before consulting the camera for position.
   - If AprilTag 1 cannot be seen because it is too close:
     - Back up in steps, stopping each time to query hw2_localization, until
       AprilTag 1 is visible.
     - Compute the difference between the calculated position and (1, 2).
     - Drive forward that remaining distance based on odometer readings.
   - Conclude at (1, 2).
"""

from math import atan2, cos, hypot, isfinite, pi, sin
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

try:
    from apriltag_msgs.msg import AprilTagDetectionArray
except ImportError:
    AprilTagDetectionArray = None

try:
    from hw2_interfaces.srv import GetRobotPose
except ImportError:
    GetRobotPose = None


WAYPOINT_1 = (1.0, 0.0, 0.0)
WAYPOINT_2 = (1.0, 2.0, pi / 2.0)
SPIN_STEP_RAD = pi / 12.0  # 15 degrees CCW


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

        if abs(distance) < 0.02:
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

        if abs(e[0]) < 0.02:
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

        if AprilTagDetectionArray is not None:
            self.detection_sub = self.create_subscription(
                AprilTagDetectionArray, '/detections', self.on_detections, 10
            )
        else:
            self.detection_sub = None

        self.pose_client = None
        if GetRobotPose is not None:
            self.pose_client = self.create_client(GetRobotPose, '/get_robot_pose')
        else:
            self.get_logger().info('hw2_interfaces not found; relying on /robot_pose topic.')

        self.declare_parameter('max_pose_age_s', 1.0)
        self.max_pose_age_s = float(self.get_parameter('max_pose_age_s').value)
        self.declare_parameter('pose_request_timeout_s', 3.0)
        self.request_timeout_s = float(
            self.get_parameter('pose_request_timeout_s').value
        )

        # Calibrated parameters from robot_test.py / hw1_solution
        self.forward_speed = 0.567333
        self.angular_speed_cw = 2.0 * pi / 4.5
        self.angular_speed_ccw = 2.0 * pi / 5.2
        self.tolerance = 0.02  # Lowered tolerance (2 cm) to avoid stopping short
        self.angle_tolerance = 0.1
        self.settle_time_s = 1.0  # Increased settling time to allow laggy camera to update

        # Sequential waypoints: WP1=(1, 0, 0), WP2=(1, 2, pi/2)
        self.wp1 = list(WAYPOINT_1)
        self.wp2 = list(WAYPOINT_2)
        self.current_state = [0.0, 0.0, 0.0]  # Assumed start at (0, 0, 0)
        self.drive_backwards = False

        self.pid = PIDcontroller(
            1.0, 0.01, 0.005, maximum_value=self.forward_speed
        )
        self.pid.setTarget(self.wp1)

        # State machine:
        # 'localizing_wp1' -> 'driving_wp1' -> 'search_spin_ccw_15' ->
        # 'search_stopped_query' -> 'centering_tag1' -> 'driving_wp2' ->
        # 'localizing_wp2' -> 'backing_up' -> 'forward_odom_to_goal' -> 'done'
        self.state = 'localizing_wp1'
        self.hw1_stage = 'rotate_to_goal'

        # AprilTag 1 visual search and centering variables
        self.latest_tag1_detection = None
        self.tag1_centered_count = 0
        self.turn_traveled = 0.0
        self.query_start_time = 0.0
        self.tag1_found_in_query = False

        # Backing up / odometer variables for WP2
        self.loc_wp2_start_time = 0.0
        self.backup_distance_step = 0.2  # Back up in 20 cm steps if tag is too close
        self.backup_traveled = 0.0
        self.odom_target_distance = 0.0
        self.odom_traveled = 0.0

        self.pose_future = None
        self.request_started = 0.0
        self.localization_request_count = 0
        self.localization_service_available = None
        self.stop_time = self.get_clock().now()
        self.last_tick = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.control_tick)

        self.get_logger().info(
            'hw2_control initialized: Leg 1 to (1, 0), Leg 2 to (1, 2).'
        )

    def stop_robot(self):
        """Command zero velocity to halt all robot movement."""
        self.cmd_pub.publish(Twist())

    def on_detections(self, msg):
        """Track detections of AprilTag 1 for centering after WP1."""
        for det in msg.detections:
            if det.id == 1:
                self.latest_tag1_detection = {
                    'cx': float(det.centre.x),
                    'cy': float(det.centre.y),
                    'time': time.monotonic(),
                }

    def on_pose(self, pose):
        """Receive pose from /robot_pose topic (used only when stationary)."""
        stationary_states = (
            'localizing_wp1', 'localizing_wp2',
            'search_stopped_query', 'backing_up_settling',
        )
        if self.state in stationary_states:
            now = self.get_clock().now()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                if self.state == 'search_stopped_query':
                    if (self.latest_tag1_detection and
                            time.monotonic() - self.latest_tag1_detection['time'] < 1.0):
                        self.tag1_found_in_query = True
                        self.process_localized_pose(pose)
                else:
                    self.process_localized_pose(pose)

    def request_localization_service(self):
        """Request fresh pose from /get_robot_pose service while stationary."""
        if self.pose_client is None:
            return

        if self.pose_future is not None:
            if time.monotonic() - self.request_started > self.request_timeout_s:
                future = self.pose_future
                self.pose_future = None
                try:
                    self.pose_client.remove_pending_request(future)
                except Exception:
                    pass
                future.cancel()
            return

        service_ready = self.pose_client.service_is_ready()
        if service_ready != self.localization_service_available:
            self.localization_service_available = service_ready
            if service_ready:
                self.get_logger().info('Localization service /get_robot_pose available.')
            else:
                self.get_logger().warn('Localization service /get_robot_pose waiting.')
        if not service_ready:
            return

        self.localization_request_count += 1
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

        stationary_states = (
            'localizing_wp1', 'localizing_wp2',
            'search_stopped_query', 'backing_up_settling',
        )
        if response.success:
            if self.state in stationary_states:
                if self.state == 'search_stopped_query':
                    tag_match = (
                        1 in response.tag_ids or
                        (self.latest_tag1_detection and
                         time.monotonic() - self.latest_tag1_detection['time'] < 1.0)
                    )
                    if tag_match:
                        self.tag1_found_in_query = True
                        self.process_localized_pose(response.pose)
                    else:
                        self.get_logger().info(
                            'Localization response received, but AprilTag 1 was not among tags.'
                        )
                else:
                    self.process_localized_pose(response.pose)
        else:
            self.get_logger().debug(f'Localization waiting: {response.message}')

    def process_localized_pose(self, pose):
        """Check position reported by hw2_localization against active goal."""
        p = pose.pose.position
        values = (p.x, p.y, p.z)
        if not all(isfinite(v) for v in values):
            return

        yaw = yaw_from_pose(pose)
        if not isfinite(yaw):
            return

        x = float(p.x)
        y = float(p.y)
        yaw = float(yaw)

        self.get_logger().info(
            f'Measured pose via hw2_localization: pos=({x:.3f}, {y:.3f}), '
            f'yaw={yaw:.3f} rad, state={self.state}.'
        )

        # Handle WP1 verification
        if self.state == 'localizing_wp1':
            dist_to_wp1 = hypot(self.wp1[0] - x, self.wp1[1] - y)
            self.get_logger().info(f'Distance to WP1 (1, 0): {dist_to_wp1:.3f} m.')
            if dist_to_wp1 <= self.tolerance:
                self.get_logger().info(
                    f'WP1 (1.0, 0.0) reached within tolerance: {dist_to_wp1:.3f} m <= '
                    f'{self.tolerance:.3f} m. '
                    f'Searching for AprilTag 1: will spin CCW 15 degrees, stop, and query...'
                )
                self.current_state = [x, y, yaw]
                self.stop_robot()
                self.turn_traveled = 0.0
                self.state = 'search_spin_ccw_15'
            else:
                self.get_logger().info(
                    f'Not yet at (1, 0) ({dist_to_wp1:.3f} m > {self.tolerance:.3f} m). '
                    f'Adjusting drive to (1, 0)...'
                )
                self.current_state = [x, y, yaw]
                self.pid.setTarget(self.wp1)
                self.hw1_stage = 'rotate_to_goal'
                self.state = 'driving_wp1'

        # Handle AprilTag 1 discovered during 15-degree search query
        elif self.state == 'search_stopped_query':
            self.get_logger().info(
                f'AprilTag 1 acquired by hw2_localization! Pos: ({x:.3f}, {y:.3f}), '
                f'yaw={yaw:.3f}. Centering AprilTag 1 in camera...'
            )
            self.current_state = [x, y, yaw]
            self.tag1_centered_count = 0
            self.state = 'centering_tag1'

        # Handle WP2 verification
        elif self.state == 'localizing_wp2':
            dist_to_wp2 = hypot(self.wp2[0] - x, self.wp2[1] - y)
            self.get_logger().info(f'Distance to WP2 (1, 2): {dist_to_wp2:.3f} m.')
            if dist_to_wp2 <= self.tolerance:
                self.get_logger().info(
                    f'WP2 (1.0, 2.0) reached within tolerance: {dist_to_wp2:.3f} m <= '
                    f'{self.tolerance:.3f} m. Complete!'
                )
                self.state = 'done'
                self.stop_robot()
            else:
                self.get_logger().info(
                    f'Not yet at (1, 2) ({dist_to_wp2:.3f} m > {self.tolerance:.3f} m). '
                    f'Adjusting drive to (1, 2)...'
                )
                self.current_state = [x, y, yaw]
                self.pid.setTarget(self.wp2)
                self.hw1_stage = 'drive'
                self.state = 'driving_wp2'

        # Handle localization acquired after backing up
        elif self.state == 'backing_up_settling':
            dist_to_wp2 = hypot(self.wp2[0] - x, self.wp2[1] - y)
            self.get_logger().info(
                f'AprilTag 1 acquired after backing up! Position: ({x:.3f}, {y:.3f}). '
                f'Difference to (1, 2): {dist_to_wp2:.3f} m.'
            )
            self.current_state = [x, y, yaw]
            self.odom_target_distance = dist_to_wp2
            self.odom_traveled = 0.0
            self.state = 'forward_odom_to_goal'
            self.get_logger().info(
                f'Moving forward {dist_to_wp2:.3f} m based on odometer readings...'
            )

    def handle_centering_tag1(self, dt):
        """Fine-adjust rotation so AprilTag 1 is centered in the camera view."""
        twist = Twist()
        tag1_visible = False
        if self.latest_tag1_detection is not None:
            if time.monotonic() - self.latest_tag1_detection['time'] < 1.0:
                tag1_visible = True

        if tag1_visible:
            cx = self.latest_tag1_detection['cx']
            pixel_error = cx - 640.0
            if abs(pixel_error) <= 30.0:
                self.tag1_centered_count += 1
                twist.angular.z = 0.0
                if self.tag1_centered_count >= 2:
                    self.get_logger().info(
                        f'AprilTag 1 centered (pixel error: {pixel_error:.1f} px). '
                        f'Turn complete! Starting traversal to (1, 2)...'
                    )
                    self.stop_robot()
                    self.current_state[2] = pi / 2.0
                    self.pid.setTarget(self.wp2)
                    self.hw1_stage = 'drive'
                    self.state = 'driving_wp2'
                    return
            else:
                self.tag1_centered_count = 0
                if pixel_error < 0:
                    twist.angular.z = float(self.angular_speed_ccw * 0.35)
                else:
                    twist.angular.z = float(-self.angular_speed_cw * 0.35)
        else:
            self.get_logger().info(
                'AprilTag 1 orientation set. Traversing to (1, 2)...'
            )
            self.stop_robot()
            self.current_state[2] = pi / 2.0
            self.pid.setTarget(self.wp2)
            self.hw1_stage = 'drive'
            self.state = 'driving_wp2'
            return

        self.update_dead_reckoning(0.0, twist.angular.z, dt)
        self.cmd_pub.publish(twist)

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

    def execute_hw1_step(self, dt, target_wp):
        """Step the HW1 3-stage control loop and dead reckoning towards target_wp."""
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
                twist_msg.angular.z = 0.0
                return True
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))

        self.update_dead_reckoning(twist_msg.linear.x, twist_msg.angular.z, dt)
        self.cmd_pub.publish(twist_msg)
        return False

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

        # 1. Stationary localization check for WP1 (1, 0)
        if self.state == 'localizing_wp1':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                self.request_localization_service()
            return

        # 2. Drive to WP1 (1, 0) via HW1 control loop
        if self.state == 'driving_wp1':
            reached = self.execute_hw1_step(dt, self.wp1)
            if reached:
                self.get_logger().info(
                    'HW1 loop reached WP1 by dead reckoning. Stopping to verify position...'
                )
                self.stop_robot()
                self.state = 'localizing_wp1'
                self.stop_time = self.get_clock().now()
            return

        # 3. Spin CCW by 15 degrees to search for AprilTag 1
        if self.state == 'search_spin_ccw_15':
            v_ang = float(self.angular_speed_ccw)
            twist = Twist()
            twist.angular.z = v_ang
            step_angle = v_ang * dt
            self.turn_traveled += step_angle
            self.update_dead_reckoning(0.0, v_ang, dt)
            self.cmd_pub.publish(twist)
            if self.turn_traveled >= SPIN_STEP_RAD:
                self.get_logger().info(
                    'Turned 15 degrees CCW. Stopping to query hw2_localization for AprilTag 1...'
                )
                self.stop_robot()
                self.state = 'search_stopped_query'
                self.stop_time = self.get_clock().now()
                self.query_start_time = time.monotonic()
                self.tag1_found_in_query = False
            return

        # 4. Stationary query for AprilTag 1 after 15-degree spin
        if self.state == 'search_stopped_query':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                self.request_localization_service()

            if self.state != 'search_stopped_query':
                return

            # Allow 3.5s query window to accommodate laggy camera frames
            if time.monotonic() - self.query_start_time > 3.5:
                self.get_logger().info(
                    'AprilTag 1 not detected at this angle. '
                    'Spinning CCW by another 15 degrees...'
                )
                self.turn_traveled = 0.0
                self.state = 'search_spin_ccw_15'
            return

        # 5. Center AprilTag 1 in camera
        if self.state == 'centering_tag1':
            self.handle_centering_tag1(dt)
            return

        # 6. Drive to WP2 (1, 2) via HW1 control loop
        if self.state == 'driving_wp2':
            reached = self.execute_hw1_step(dt, self.wp2)
            if reached:
                self.get_logger().info(
                    'HW1 loop reached WP2 by dead reckoning. Stopping to verify position...'
                )
                self.stop_robot()
                self.state = 'localizing_wp2'
                self.stop_time = self.get_clock().now()
                self.loc_wp2_start_time = time.monotonic()
            return

        # 7. Stationary localization check for WP2 (1, 2)
        if self.state == 'localizing_wp2':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                self.request_localization_service()
            if time.monotonic() - self.loc_wp2_start_time > 3.5:
                self.get_logger().warn(
                    'Cannot see AprilTag 1 near (1, 2) (tag may be too close). '
                    'Backing up until AprilTag 1 is visible...'
                )
                self.state = 'backing_up'
                self.backup_traveled = 0.0
            return

        # 8. Back up until AprilTag 1 is visible
        if self.state == 'backing_up':
            v = -0.15
            twist = Twist()
            twist.linear.x = v
            step_dist = abs(v) * dt
            self.backup_traveled += step_dist
            self.update_dead_reckoning(v, 0.0, dt)
            self.cmd_pub.publish(twist)
            if self.backup_traveled >= self.backup_distance_step:
                self.get_logger().info(
                    f'Backed up {self.backup_traveled:.2f} m. Stopping to check camera...'
                )
                self.stop_robot()
                self.state = 'backing_up_settling'
                self.stop_time = self.get_clock().now()
                self.loc_wp2_start_time = time.monotonic()
            return

        # 9. Stationary settling check after backing up
        if self.state == 'backing_up_settling':
            self.stop_robot()
            time_stopped = (now - self.stop_time).nanoseconds / 1e9
            if time_stopped >= self.settle_time_s:
                self.request_localization_service()
            if time.monotonic() - self.loc_wp2_start_time > 3.5:
                self.get_logger().info('AprilTag 1 still not visible. Backing up further...')
                self.state = 'backing_up'
                self.backup_traveled = 0.0
            return

        # 10. Move forward remaining distance based on odometer readings
        if self.state == 'forward_odom_to_goal':
            dist_left = self.odom_target_distance - self.odom_traveled
            v = 0.2 if dist_left > 0.1 else 0.1
            twist = Twist()
            twist.linear.x = v
            step_dist = v * dt
            self.odom_traveled += step_dist
            self.update_dead_reckoning(v, 0.0, dt)
            self.cmd_pub.publish(twist)
            if self.odom_traveled >= self.odom_target_distance:
                self.get_logger().info(
                    f'Traversed {self.odom_traveled:.3f} m based on odometer readings. '
                    f'Reached (1.0, 2.0)! Complete.'
                )
                self.stop_robot()
                self.state = 'done'
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
