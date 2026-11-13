"""Closed-loop square waypoint navigation controller using EKF-SLAM pose feedback.

Directs the robot counterclockwise through a 6 ft x 6 ft square (or configurable size)
using straight travel and calibrated fixed-rate pivot pulses:
    WP0: (0, 0) ft     [0.0000, 0.0000] m - Start / Origin
    WP1: (7, 0) ft     [2.1336, 0.0000] m - Bottom-right (+X)
    WP2: (7, 7) ft     [2.1336, 2.1336] m - Top-right (+X, +Y)
    WP3: (0, 7) ft     [0.0000, 2.1336] m - Top-left (+Y)
    WP4: (0, 0) ft     [0.0000, 0.0000] m - Return to start (Origin)

Coordinates are maintained in the local navigation frame anchored at the initial EKF
pose upon start:
    x_nav =  (x_slam - x_0) * cos(theta_0) + (y_slam - y_0) * sin(theta_0)
    y_nav = -(x_slam - x_0) * sin(theta_0) + (y_slam - y_0) * cos(theta_0)
    theta_nav = wrap_angle(theta_slam - theta_0)
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple

from .coordinates import wrap_angle
from hw3.turn_calibration import TurnCalibration


class NavigationState(Enum):
    """Navigation controller state machine states."""
    IDLE = "IDLE"
    WAITING_FOR_POSE = "WAITING_FOR_POSE"
    TURNING = "TURNING"
    DRIVING = "DRIVING"
    PAUSED_STALE_POSE = "PAUSED_STALE_POSE"
    PAUSED_HIGH_UNCERTAINTY = "PAUSED_HIGH_UNCERTAINTY"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class Waypoint:
    """Target waypoint in meters."""
    index: int
    x_m: float
    y_m: float
    description: str


@dataclass
class NavigationOutput:
    """Command output and telemetry from one controller cycle."""
    linear_v_mps: float
    angular_omega_radps: float
    state: NavigationState
    status_message: str
    waypoint_index: int
    target_x_m: float
    target_y_m: float
    robot_x_m: float
    robot_y_m: float
    robot_theta_rad: float
    distance_error_m: float
    heading_error_rad: float
    sigma_pos_m: float
    sigma_theta_rad: float


class SquareNavigator:
    """Manages closed-loop waypoint following for the square trajectory."""

    FEET_TO_METERS = 0.3048

    def __init__(
        self,
        square_size_ft: float = 6.0,
        max_linear_v_mps: float = 0.20,
        min_linear_v_mps: float = 0.05,
        linear_kp: float = 0.60,
        waypoint_tolerance_m: float = 0.10,
        heading_tolerance_rad: float = 0.12,  # ~7 degrees
        pose_timeout_s: float = 0.60,
        max_pos_uncertainty_m: float = 0.40,
        max_heading_uncertainty_rad: float = 0.40,
        turn_calibration: Optional[TurnCalibration] = None,
        turn_pulse_rad: float = math.pi / 6,
        turn_settle_s: float = 1.0,
    ):
        """Initialize local square waypoints, calibrated turning, and mission limits.

        Args:
            square_size_ft: Square side length in feet, converted internally to meters.
            max_linear_v_mps: Maximum requested straight speed in meters per second.
            min_linear_v_mps: Minimum nonzero straight speed in meters per second.
            linear_kp: Proportional distance gain for straight travel.
            waypoint_tolerance_m: Distance from a target waypoint accepted as arrival, in meters.
            heading_tolerance_rad: Maximum absolute heading error accepted as aligned, in radians.
            pose_timeout_s: Maximum allowable pose age in seconds.
            max_pos_uncertainty_m: Maximum allowable position standard deviation in meters.
            max_heading_uncertainty_rad: Maximum allowable yaw standard deviation in radians.
            turn_calibration: Measured fixed motor input and direction-specific yaw rates.
            turn_pulse_rad: Maximum nominal angle of each timed turn pulse, in radians.
            turn_settle_s: Stopped settling interval after each turn pulse, in seconds.

        Returns:
            None. State changes and publications are performed in place.
        """
        self.turn_calibration = turn_calibration
        self.turn_pulse_rad = float(turn_pulse_rad)
        self.turn_settle_s = float(turn_settle_s)
        self.turn = None
        self.stop_reason = "Navigation inactive"
        if not 0 < self.turn_pulse_rad <= math.pi / 2 or not math.isfinite(self.turn_settle_s) or self.turn_settle_s < 0:
            raise ValueError("Use a positive turn pulse <= pi/2 and nonnegative settle time.")
        self.square_size_ft = float(square_size_ft)
        self.square_size_m = self.square_size_ft * self.FEET_TO_METERS

        self.max_linear_v = float(max_linear_v_mps)
        self.min_linear_v = float(min_linear_v_mps)

        self.linear_kp = float(linear_kp)

        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.heading_tolerance_rad = float(heading_tolerance_rad)

        self.pose_timeout_s = float(pose_timeout_s)
        self.max_pos_uncertainty_m = float(max_pos_uncertainty_m)
        self.max_heading_uncertainty_rad = float(max_heading_uncertainty_rad)

        # Build waypoints: counterclockwise square starting at (0, 0)
        s = self.square_size_m
        self.waypoints: List[Waypoint] = [
            Waypoint(0, 0.0, 0.0, "WP0: Bottom-left (Start)"),
            Waypoint(1, s, 0.0, "WP1: Bottom-right (+X)"),
            Waypoint(2, s, s, "WP2: Top-right (+X, +Y)"),
            Waypoint(3, 0.0, s, "WP3: Top-left (+Y)"),
            Waypoint(4, 0.0, 0.0, "WP4: Bottom-left (Return to origin)"),
        ]

        self.current_wp_idx = 1  # First destination is WP1
        self.state = NavigationState.IDLE
        self.is_active = False

        # Anchor origin (latched upon start)
        self.origin_latched = False
        self.x0_slam = 0.0
        self.y0_slam = 0.0
        self.theta0_slam = 0.0

        # Latest received EKF pose
        self.latest_slam_pose: Optional[Tuple[float, float, float]] = None
        self.latest_uncertainty: Tuple[float, float] = (0.0, 0.0)  # (sigma_pos, sigma_theta)
        self.latest_pose_stamp_s: float = 0.0
        self._validate_motion()

    def _validate_motion(self):
        """Reject nonfinite or nonpositive motion settings before any command."""
        values = (self.square_size_m, self.max_linear_v, self.min_linear_v,
                  self.pose_timeout_s, self.heading_tolerance_rad, self.linear_kp,
                  self.max_pos_uncertainty_m, self.max_heading_uncertainty_rad)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Navigation distances, speeds, tolerances, and limits must be positive.')
        if self.min_linear_v > self.max_linear_v:
            raise ValueError('Minimum linear velocity must not exceed maximum velocity.')

    def start(self) -> Tuple[bool, str]:
        """Start or resume the square navigation mission.

        Args:
            None.

        Returns:
            (success, message), indicating whether navigation may begin or resume.
        """
        if self.turn_calibration is None:
            return False, "Set measured CCW and CW turn calibration before starting."
        if self.state == NavigationState.COMPLETED:
            return False, "Mission already completed. Reset navigator to run again."

        if self.state == NavigationState.STOPPED and self.turn is not None:
            return False, "Turn stopped; reset before restarting the mission."
        self.is_active = True
        if not self.origin_latched:
            self.state = NavigationState.WAITING_FOR_POSE
            return True, "Navigation started. Waiting for initial EKF pose to anchor origin."
        else:
            self.state = NavigationState.TURNING
            return True, f"Navigation resumed towards Waypoint {self.current_wp_idx}."

    def stop(self, reason: str = "Manual stop requested") -> Tuple[bool, str]:
        """Stop the robot immediately and halt navigation."""
        self.stop_reason = reason
        self.is_active = False
        self.state = NavigationState.STOPPED
        return True, f"Navigation stopped: {reason}"

    def reset(self):
        """Reset navigator state to initial unstarted condition."""
        self.is_active = False
        self.state = NavigationState.IDLE
        self.origin_latched = False
        self.current_wp_idx = 1
        self.latest_slam_pose = None
        self.latest_pose_stamp_s = 0.0
        self.turn = None
        self.stop_reason = "Navigation inactive"

    def update_ekf_pose(
        self,
        x_slam: float,
        y_slam: float,
        theta_slam: float,
        sigma_pos_m: float,
        sigma_theta_rad: float,
        stamp_s: float,
    ):
        """Update latest robot pose and uncertainty estimate from EKF-SLAM.

        Args:
            x_slam: Estimated robot x coordinate in the SLAM frame, in meters.
            y_slam: Estimated robot y coordinate in the SLAM frame, in meters.
            theta_slam: Estimated robot yaw in the SLAM frame, in radians.
            sigma_pos_m: Robot planar position standard deviation, in meters.
            sigma_theta_rad: Robot yaw standard deviation, in radians.
            stamp_s: Observation or pose timestamp in seconds in the ROS clock domain.

        Returns:
            None. State changes and publications are performed in place.
        """
        values = (x_slam, y_slam, theta_slam, sigma_pos_m, sigma_theta_rad, stamp_s)
        if not all(math.isfinite(v) for v in values) or min(sigma_pos_m, sigma_theta_rad) < 0:
            self.latest_pose_stamp_s = -math.inf
            return
        self.latest_slam_pose = (float(x_slam), float(y_slam), wrap_angle(float(theta_slam)))
        self.latest_uncertainty = (float(sigma_pos_m), float(sigma_theta_rad))
        self.latest_pose_stamp_s = float(stamp_s)

        # Latch initial pose as origin if waiting for it
        if self.is_active and not self.origin_latched:
            if (sigma_pos_m <= self.max_pos_uncertainty_m and
                    sigma_theta_rad <= self.max_heading_uncertainty_rad):
                self.x0_slam = float(x_slam)
                self.y0_slam = float(y_slam)
                self.theta0_slam = wrap_angle(float(theta_slam))
                self.origin_latched = True
                self.current_wp_idx = 1
                self.state = NavigationState.TURNING

    def slam_to_nav_frame(
        self,
        x_slam: float,
        y_slam: float,
        theta_slam: float,
    ) -> Tuple[float, float, float]:
        """Transform coordinates from global EKF SLAM frame to local navigation frame.

        Args:
            x_slam: Estimated robot x coordinate in the SLAM frame, in meters.
            y_slam: Estimated robot y coordinate in the SLAM frame, in meters.
            theta_slam: Estimated robot yaw in the SLAM frame, in radians.

        Returns:
            (x_m, y_m, yaw_rad) relative to the latched navigation origin.
        """
        if not self.origin_latched:
            return 0.0, 0.0, 0.0

        dx = x_slam - self.x0_slam
        dy = y_slam - self.y0_slam
        c0 = math.cos(self.theta0_slam)
        s0 = math.sin(self.theta0_slam)

        x_nav = dx * c0 + dy * s0
        y_nav = -dx * s0 + dy * c0
        theta_nav = wrap_angle(theta_slam - self.theta0_slam)
        return x_nav, y_nav, theta_nav

    def _output(self, message, linear=0.0, angular=0.0):
        """Build telemetry in the local frame for a moving or stopped command."""
        target = self.waypoints[self.current_wp_idx]
        pose = self.slam_to_nav_frame(*self.latest_slam_pose) if self.latest_slam_pose else (0, 0, 0)
        dx, dy = target.x_m - pose[0], target.y_m - pose[1]
        error = wrap_angle(math.atan2(dy, dx) - pose[2])
        return NavigationOutput(linear, angular, self.state, message, self.current_wp_idx,
                                target.x_m, target.y_m, *pose, math.hypot(dx, dy), error,
                                *self.latest_uncertainty)

    def _pose_problem(self, now):
        """Return a pause reason when feedback is missing, invalid, stale, or uncertain.

        Args:
            now: Current controller time in seconds in the same clock domain as pose stamps.

        Returns:
            (pause_state, reason), or None when pose feedback is usable.
        """
        if not self.origin_latched or self.latest_slam_pose is None:
            return NavigationState.WAITING_FOR_POSE, 'Waiting for initial EKF pose'
        values = (*self.latest_slam_pose, *self.latest_uncertainty, self.latest_pose_stamp_s)
        if not all(math.isfinite(v) for v in values):
            return NavigationState.PAUSED_STALE_POSE, 'Invalid pose feedback'
        age = now - self.latest_pose_stamp_s
        if age < -0.05 or age > self.pose_timeout_s:
            return NavigationState.PAUSED_STALE_POSE, 'Pose feedback is stale'
        if (self.latest_uncertainty[0] > self.max_pos_uncertainty_m or
                self.latest_uncertainty[1] > self.max_heading_uncertainty_rad):
            return NavigationState.PAUSED_HIGH_UNCERTAINTY, 'Pose uncertainty is too high'
        return None

    def _target_error(self):
        """Return distance and signed heading error to the current waypoint."""
        x, y, yaw = self.slam_to_nav_frame(*self.latest_slam_pose)
        target = self.waypoints[self.current_wp_idx]
        dx, dy = target.x_m - x, target.y_m - y
        return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - yaw)

    def _advance_waypoint(self):
        """Advance to another corner or finish the square, clearing turn state."""
        self.turn = None
        if self.current_wp_idx == len(self.waypoints) - 1:
            self.state, self.is_active = NavigationState.COMPLETED, False
        else:
            self.current_wp_idx += 1
            self.state = NavigationState.TURNING

    def _turn_command(self, now, error):
        """Use timed fixed-rate pulses with settle intervals and a hard angle budget.

        Args:
            now: Current controller time in seconds in the same clock domain as pose stamps.
            error: Signed heading error to the target, in radians.

        Returns:
            NavigationOutput containing a fixed-rate pulse or a stopped command.
        """
        if self.turn is None:
            self.turn = FixedTurnPulse(self.turn_calibration, abs(error),
                                       self.turn_pulse_rad, self.turn_settle_s)
        omega, message = self.turn.step(now, error, self.heading_tolerance_rad)
        if self.turn.exhausted:
            self.stop('Turn budget exhausted; check calibration and heading feedback')
            return self._output(self.stop_reason)
        if self.turn.done:
            self.turn = None
            self.state = NavigationState.DRIVING
        return self._output(message, angular=omega)

    def step(self, current_time_s: float) -> NavigationOutput:
        """Compute one nonblocking straight/turn command, with stops during settling.

        Args:
            current_time_s: Current controller time in seconds in the pose timestamp clock domain.

        Returns:
            NavigationOutput or (yaw_rate_radps, status) for the controller being stepped.
        """
        now = float(current_time_s)
        if not self.is_active:
            return self._output(self.stop_reason)
        problem = self._pose_problem(now)
        if problem:
            self.state, message = problem
            if self.turn:
                self.turn.pause(now)
            return self._output(message)
        return self._step_target(now)

    def _step_target(self, now):
        """Handle waypoint arrival and select straight travel or pivoting.

        Args:
            now: Current controller time in seconds in the same clock domain as pose stamps.

        Returns:
            NavigationOutput for arrival, a pivot pulse, or straight travel.
        """
        distance, error = self._target_error()
        if distance <= self.waypoint_tolerance_m:
            self._advance_waypoint()
            return self._output('Waypoint reached; stopped before next phase')
        if self.turn is not None or abs(error) > self.heading_tolerance_rad:
            self.state = NavigationState.TURNING
            return self._turn_command(now, error)
        self.state = NavigationState.DRIVING
        speed = max(self.min_linear_v, min(self.max_linear_v, self.linear_kp * distance))
        return self._output('Driving straight', linear=speed)


class FixedTurnPulse:
    """Fixed-rate pivot pulses bounded by elapsed time and total commanded angle.

    The angle budget is an open-loop bound, not a measured rotation. Feedback
    may finish a turn, but cannot authorize unlimited spinning. Units: rad, s.
    """

    def __init__(self, calibration, target_angle, pulse_rad, settle_s):
        self.calibration = calibration
        self.pulse_rad, self.settle_s = pulse_rad, settle_s
        self.budget = min(math.pi, target_angle + math.radians(30))
        self.used = self.omega = 0.0
        self.deadline = self.ready_at = 0.0
        self.last_time = None
        self.done = self.exhausted = False

    def _account(self, now):
        """Account once for the last held command, bounded by the motor watchdog."""
        if self.last_time is not None:
            dt = max(0.0, min(now - self.last_time, 0.15))
            self.used += abs(self.omega) * dt
        self.last_time = now

    def pause(self, now):
        """Stop a pulse and settle while preserving the consumed turn budget."""
        self._account(now)
        self.omega = 0.0
        self.ready_at = now + self.settle_s

    def step(self, now, error, tolerance):
        """Return a fixed rate or zero; never restart before the stopped settle ends."""
        self._account(now)
        if self.omega and (now >= self.deadline or abs(error) <= tolerance):
            self.pause(now)
        if self.omega:
            return self.omega, 'Fixed-rate turn pulse'
        if now < self.ready_at:
            return 0.0, 'Stopped; waiting for heading to settle'
        return self._next_pulse(now, error, tolerance)

    def _next_pulse(self, now, error, tolerance):
        """Accept alignment or issue a pulse whose duration fits the remaining budget.

        Args:
            now: Current controller time in seconds in the same clock domain as pose stamps.
            error: Signed heading error to the target, in radians.
            tolerance: Absolute heading tolerance, in radians.

        Returns:
            (yaw_rate_radps, status); zero rate when aligned or the budget is exhausted.
        """
        if abs(error) <= tolerance:
            self.done = True
            return 0.0, 'Turn finished; stopped'
        remaining = self.budget - self.used
        if remaining <= tolerance:
            self.exhausted = True
            return 0.0, 'Turn angle budget exhausted'
        self.omega = self.calibration.rate(error)
        angle = min(abs(error), self.pulse_rad, remaining)
        self.deadline = now + angle / abs(self.omega)
        return self.omega, 'Starting fixed-rate turn pulse'
