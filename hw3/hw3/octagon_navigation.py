"""Octagon navigation controller and ROS 2 node.

Implements the dedicated 8-sided octagon traversal:
- Starts with an initial 45-degree CCW turn
- 8 sides (0.75m each) with 1-second midpoint and post-forward camera pauses
- Measured half-second 45-degree left turns at [-0.35, +0.35]
- Optional SLAM waypoint tracking (mission:=slam_octagon)
- Completes cleanly after 8 sides (single lap)
"""

from collections import deque
import math

import rclpy
from rclpy.node import Node

from .motor_control import ForwardCalibration, TurnCalibration
from .slam_vision import (
    BaseNavigationNode,
    NavigationOutput,
    NavigationPolicy,
    NavigationState,
    VisualFeedback,
    Waypoint,
    wrap_angle,
)


def octagon_vertices(origin: tuple[float, float, float], side_length_m: float = 0.75) -> tuple:
    """Return fixed SLAM-frame endpoints for an 8-sided regular octagon."""
    x, y, yaw = origin
    vertices = []
    for index in range(1, 9):
        heading = yaw + index * (math.pi / 4.0)
        x += side_length_m * math.cos(heading)
        y += side_length_m * math.sin(heading)
        vertices.append((x, y))
    vertices[-1] = origin[:2]
    return tuple(vertices)


class OctagonNavigator:
    """Pure controller for 8-sided octagon navigation."""

    def __init__(
        self,
        policy: NavigationPolicy = None,
        side_count: int = 8,
        side_length_m: float = 0.75,
        turn_calibration: TurnCalibration = None,
        forward_calibration: ForwardCalibration = None,
        waypoint_tolerance_m: float = 0.05,
        heading_tolerance_rad: float = 0.12,
        pose_timeout_s: float = 0.60,
        max_pos_uncertainty_m: float = 0.40,
        max_heading_uncertainty_rad: float = 0.40,
        **_ignored,
    ):
        """Initialize octagon navigation controller parameters, timing, and state."""
        self.policy = policy or NavigationPolicy(manual_motion_enabled=True)
        self.side_count = side_count
        self.side_length_m = side_length_m if side_length_m > 0 else 0.75
        self.turn_calibration = turn_calibration
        self.forward = forward_calibration or ForwardCalibration()
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.heading_tolerance_rad = heading_tolerance_rad
        self.pose_timeout_s = pose_timeout_s
        self.max_pos_uncertainty_m = max_pos_uncertainty_m
        self.max_heading_uncertainty_rad = max_heading_uncertainty_rad

        self.corner_angle_rad = 2.0 * math.pi / self.side_count  # pi/4 (45 deg)
        self.state = NavigationState.IDLE
        self.is_active = False
        self.stop_reason = 'Not started'

        self.side_index = 0
        self.current_wp_idx = 0
        self.midpoint_done = False
        self.drive_used = 0.0
        self.corner_used = 0.0

        self.origin_latched = False
        self.x0_slam = self.y0_slam = self.theta0_slam = 0.0
        self.latest_slam_pose = None
        self.latest_pose_stamp_s = -math.inf
        self.latest_uncertainty = (math.inf, math.inf)

        self.navigation_x = self.navigation_y = self.navigation_yaw = 0.0
        self.heading_time = -math.inf
        self.course_heading = 0.0
        self.target = (0.0, 0.0)
        self.last_command = (0.0, 0.0)
        self.last_tick = None
        self.phase_started = None

        self.vision = VisualFeedback(self.policy)

    def start(self) -> tuple[bool, str]:
        """Activate octagon navigation mission and transition to initial wait state."""
        if self.turn_calibration is None:
            return False, 'Turn calibration is required.'
        if self.state != NavigationState.IDLE:
            return False, 'Mission already started; reset/relaunch for a new run.'
        self.is_active = True
        self.state = NavigationState.WAITING_FOR_POSE
        return True, 'Starting manual octagon navigation'

    def stop(self, reason: str = 'Manual stop requested') -> tuple[bool, str]:
        """Halt navigation mission with a given reason."""
        self.stop_reason = reason
        self.is_active = False
        self.state = NavigationState.STOPPED
        return True, reason

    def update_ekf_pose(self, x_slam: float, y_slam: float, theta_slam: float, sigma_pos_m: float, sigma_theta_rad: float, stamp_s: float):
        """Update controller with latest EKF-SLAM pose and covariance estimates."""
        self.latest_slam_pose = (x_slam, y_slam, wrap_angle(theta_slam))
        self.latest_uncertainty = (sigma_pos_m, sigma_theta_rad)
        self.latest_pose_stamp_s = stamp_s
        if self.is_active and not self.origin_latched:
            self.x0_slam, self.y0_slam, self.theta0_slam = self.latest_slam_pose
            self.origin_latched = True
            self.navigation_yaw = self.theta0_slam
            self.heading_time = stamp_s
            self.navigation_x, self.navigation_y = x_slam, y_slam

    def update_vision(self, stamp: float, tags: list, known_ids: list, clearance, tag_yaws, sigmas, ekf_heading):
        """Update visual feedback state with latest AprilTag detection results."""
        self.vision.update(stamp, tags, known_ids, clearance, self.latest_slam_pose, self.drive_used, tag_yaws, sigmas, ekf_heading)

    def _best_pose(self) -> tuple[float, float, float]:
        """Return latest valid SLAM pose or dead-reckoned odometry if timed out."""
        now = self.last_tick
        if self.latest_slam_pose is not None and now is not None:
            if -0.05 <= now - self.latest_pose_stamp_s <= self.pose_timeout_s:
                return self.latest_slam_pose
        return self.navigation_x, self.navigation_y, self.navigation_yaw

    def _transition(self, state: NavigationState, now: float):
        """Switch navigation state and record the phase transition timestamp."""
        self.state = state
        self.phase_started = now

    def _output(self, message: str, linear: float = 0.0, angular: float = 0.0) -> NavigationOutput:
        """Construct a navigation telemetry output container for the current tick."""
        self.last_command = (linear, angular)
        rx, ry, rtheta = self._best_pose()
        error = wrap_angle(self.course_heading - rtheta)
        rem = max(0.0, self.side_length_m - self.drive_used)
        return NavigationOutput(
            linear_v_mps=linear,
            angular_omega_radps=angular,
            state=self.state,
            status_message=message,
            waypoint_index=self.current_wp_idx,
            target_x_m=self.target[0],
            target_y_m=self.target[1],
            robot_x_m=rx,
            robot_y_m=ry,
            remaining_distance_m=rem,
            heading_error_rad=error,
            uncertainty_pos_m=self.latest_uncertainty[0],
            uncertainty_heading_rad=self.latest_uncertainty[1],
        )

    def step(self, current_time_s: float) -> NavigationOutput:
        """Advance navigation state machine by one discrete time step."""
        now = float(current_time_s)
        dt = 0.0 if self.last_tick is None else max(0.0, min(now - self.last_tick, 0.15))
        self.navigation_x += self.last_command[0] * dt * math.cos(self.navigation_yaw)
        self.navigation_y += self.last_command[0] * dt * math.sin(self.navigation_yaw)
        self.navigation_yaw = wrap_angle(self.navigation_yaw + self.last_command[1] * dt)
        self.drive_used += abs(self.last_command[0]) * dt
        if self.state == NavigationState.TURNING:
            self.corner_used += abs(self.last_command[1]) * dt
        self.last_tick = now

        if not self.is_active:
            return self._output(self.stop_reason)
        if self.phase_started is None:
            self.phase_started = now

        handlers = {
            NavigationState.WAITING_FOR_POSE: self._wait_for_start,
            NavigationState.TURNING: self._manual_turn,
            NavigationState.DRIVING: self._manual_drive,
            NavigationState.MIDPOINT_PAUSE: self._manual_midpoint_pause,
            NavigationState.STOPPING: self._manual_pause,
        }
        handler = handlers.get(self.state, lambda t: self._output(f'Unhandled state: {self.state}'))
        return handler(now)

    def _wait_for_start(self, now: float) -> NavigationOutput:
        """Latch origin pose and initiate first 45-degree corner turn."""
        if not self.origin_latched:
            self.x0_slam, self.y0_slam, self.theta0_slam = self._best_pose()
            self.navigation_yaw = self.theta0_slam
            self.origin_latched = True
        self.side_index = 0
        return self._manual_begin_turn(now)

    def _manual_begin_turn(self, now: float) -> NavigationOutput:
        """Initialize corner turn angle and transition to turning state."""
        self.current_wp_idx = self.side_index + 1
        self.course_heading = self.theta0_slam + (self.side_index + 1) * self.corner_angle_rad
        self.corner_used = 0.0
        self._transition(NavigationState.TURNING, now)
        return self._output(f'[Octagon] Leg {self.side_index + 1} Turn 45° CCW', angular=self.turn_calibration.rate(1))

    def _manual_turn(self, now: float) -> NavigationOutput:
        """Command open-loop calibrated corner turn until timed duration elapses."""
        turn_duration = getattr(self.policy, 'corner_turn_duration_s', 0.5)
        if now - self.phase_started + 1e-9 >= turn_duration:
            self._begin_leg(now)
            return self._manual_drive(now)
        return self._output(f'[Octagon] Leg {self.side_index + 1} Turning 45° CCW', angular=self.turn_calibration.rate(1))

    def _begin_leg(self, now: float):
        """Compute leg target point and transition to forward driving state."""
        self.drive_used = 0.0
        self.midpoint_done = False
        rx, ry, _ = self._best_pose()
        self.target = (
            rx + self.side_length_m * math.cos(self.course_heading),
            ry + self.side_length_m * math.sin(self.course_heading),
        )
        self._transition(NavigationState.DRIVING, now)

    def _manual_drive(self, now: float) -> NavigationOutput:
        """Execute constant forward velocity until reaching midpoint or corner."""
        half_distance = self.side_length_m / 2.0
        goal_distance = self.side_length_m if self.midpoint_done else half_distance
        if self.drive_used + 1e-9 >= goal_distance:
            return self._manual_checkpoint(now)
        return self._output(f'[Octagon] Driving Leg {self.side_index + 1}', linear=self.forward.speed_mps)

    def _manual_checkpoint(self, now: float) -> NavigationOutput:
        """Transition to midpoint pause or post-leg corner pause."""
        if not self.midpoint_done:
            self._transition(NavigationState.MIDPOINT_PAUSE, now)
            return self._output(f'[Octagon] Leg {self.side_index + 1} Midpoint Pause')
        self.side_index += 1
        self._transition(NavigationState.STOPPING, now)
        return self._output(f'[Octagon] Leg {self.side_index} Complete; Pause before turn')

    def _manual_midpoint_pause(self, now: float) -> NavigationOutput:
        """Pause robot at leg midpoint before resuming forward drive."""
        if now - self.phase_started + 1e-9 < self.policy.manual_midpoint_pause_s:
            return self._output(f'[Octagon] Leg {self.side_index + 1} Midpoint Pause')
        self.midpoint_done = True
        self._transition(NavigationState.DRIVING, now)
        return self._manual_drive(now)

    def _manual_pause(self, now: float) -> NavigationOutput:
        """Pause robot after driving a leg before initiating next corner turn."""
        if now - self.phase_started + 1e-9 < self.policy.manual_forward_pause_s:
            return self._output(f'[Octagon] Leg {self.side_index} Post-Forward Pause')
        if self.side_index >= self.side_count:
            self.state = NavigationState.COMPLETED
            self.is_active = False
            return self._output(f'[Octagon] Traversal complete: {self.side_count} sides and turns.')
        return self._manual_begin_turn(now)


class OctagonNavigationNode(BaseNavigationNode):
    """ROS 2 Node dedicated to running Octagon missions."""

    def __init__(self):
        """Initialize ROS 2 octagon navigation node, parameters, and controller."""
        super().__init__('octagon_navigation_node')
        policy = NavigationPolicy(
            manual_motion_enabled=True,
            corner_turn_duration_s=self.get_parameter('navigation.corner_turn_duration_s').value if self.has_parameter('navigation.corner_turn_duration_s') else 0.5,
            manual_forward_pause_s=self.get_parameter('navigation.manual_forward_pause_s').value if self.has_parameter('navigation.manual_forward_pause_s') else 1.0,
            manual_midpoint_pause_s=self.get_parameter('navigation.manual_midpoint_pause_s').value if self.has_parameter('navigation.manual_midpoint_pause_s') else 1.0,
        )
        self.navigator = OctagonNavigator(
            policy=policy,
            side_count=8,
            side_length_m=float(self.get_parameter('navigation.side_length_m').value or 0.75),
            turn_calibration=self.calibration,
            forward_calibration=self.forward_calibration,
            waypoint_tolerance_m=float(self.get_parameter('navigation.waypoint_tolerance_m').value),
            heading_tolerance_rad=float(self.get_parameter('navigation.heading_tolerance_rad').value),
        )
        if self.get_parameter('navigation.auto_start').value:
            self.get_logger().info('Octagon auto-start enabled: starting traversal.')
            self.navigator.start()
        else:
            self.get_logger().info('Octagon Navigation Node initialized in IDLE mode. Waiting for /start_navigation.')


def octagon_navigation_main(args=None):
    """Run OctagonNavigationNode until ROS shutdown."""
    rclpy.init(args=args)
    node = OctagonNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
