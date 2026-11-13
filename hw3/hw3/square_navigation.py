"""Four fixed-heading legs with corner scans and fresh visual verification."""

from dataclasses import dataclass
from enum import Enum
import math

from .coordinates import wrap_angle
from .turn_calibration import TurnCalibration
from .vision_navigation import NavigationPolicy, VisualFeedback


class NavigationState(Enum):
    """Motion phases and terminal outcomes recorded in navigation_log.csv."""
    IDLE = 'IDLE'
    WAITING_FOR_POSE = 'WAITING_FOR_POSE'
    DRIVING = 'DRIVING'
    STOPPING = 'STOPPING'
    SCANNING = 'SCANNING'
    LOCALIZING = 'LOCALIZING'
    ALIGNING = 'ALIGNING'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'


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
    """Execute six-foot legs using tag displacement and fixed initial-frame headings."""

    FEET_TO_METERS = 0.3048

    def __init__(self, square_size_ft=6.0, max_linear_v_mps=0.12,
                 min_linear_v_mps=0.05, linear_kp=0.60, waypoint_tolerance_m=0.10,
                 heading_tolerance_rad=0.12, pose_timeout_s=0.60,
                 max_pos_uncertainty_m=0.40, max_heading_uncertainty_rad=0.40,
                 turn_calibration=None, turn_pulse_rad=math.pi / 12,
                 turn_settle_s=1.0, policy=None):
        """Configure a stopped four-leg mission without supplying a surveyed map.

        Args:
            square_size_ft: Desired forward travel per side, in feet.
            max_linear_v_mps: Maximum forward speed in meters per second.
            min_linear_v_mps: Minimum forward speed in meters per second.
            linear_kp: Gain converting remaining meters to requested speed.
            waypoint_tolerance_m: Allowed shortfall per leg in meters.
            heading_tolerance_rad: Allowed absolute side-heading error in radians.
            pose_timeout_s: Maximum EKF pose age in seconds.
            max_pos_uncertainty_m: Maximum position standard deviation in meters.
            max_heading_uncertainty_rad: Maximum yaw standard deviation in radians.
            turn_calibration: Measured direction-specific motor inputs and rad/s rates.
            turn_pulse_rad: Maximum alignment correction pulse angle, in radians.
            turn_settle_s: Stopped settling time after motion, in seconds.
            policy: NavigationPolicy with vision, stability, clearance, and time limits.

        Returns:
            None. Motion starts only after start() and fresh visual localization.
        """
        self.square_size_m = float(square_size_ft) * self.FEET_TO_METERS
        self.max_linear_v, self.min_linear_v = max_linear_v_mps, min_linear_v_mps
        self.linear_kp, self.waypoint_tolerance_m = linear_kp, waypoint_tolerance_m
        self.heading_tolerance_rad, self.pose_timeout_s = heading_tolerance_rad, pose_timeout_s
        self.max_pos_uncertainty_m = max_pos_uncertainty_m
        self.max_heading_uncertainty_rad = max_heading_uncertainty_rad
        self.turn_calibration = turn_calibration
        self.turn_pulse_rad, self.turn_settle_s = turn_pulse_rad, turn_settle_s
        self.policy = policy or NavigationPolicy()
        self._validate_motion()
        self.reset()

    def _validate_motion(self):
        """Reject invalid motion settings before the first command.

        Args:
            None.

        Returns:
            None. Raises ValueError for invalid limits.
        """
        limits = (self.square_size_m, self.max_linear_v, self.min_linear_v, self.linear_kp,
                  self.waypoint_tolerance_m, self.heading_tolerance_rad, self.pose_timeout_s,
                  self.max_pos_uncertainty_m, self.max_heading_uncertainty_rad,
                  self.turn_pulse_rad, self.turn_settle_s)
        if not all(math.isfinite(v) and v > 0 for v in limits):
            raise ValueError('Navigation limits must be finite and positive.')
        if self.min_linear_v > self.max_linear_v or self.turn_pulse_rad > math.pi / 2:
            raise ValueError('Invalid speed range or turn pulse larger than 90 degrees.')

    def reset(self):
        """Clear the mission, including visual references and timed-motion budgets.

        Args:
            None.

        Returns:
            None. Leaves the controller idle with no motion request.
        """
        self.state, self.is_active = NavigationState.IDLE, False
        self.stop_reason = 'Navigation inactive'
        self.current_wp_idx, self.side_index = 1, 0
        self.latest_slam_pose, self.latest_pose_stamp_s = None, -math.inf
        self.latest_uncertainty = (math.inf, math.inf)
        self.origin_latched = False
        self.x0_slam = self.y0_slam = self.theta0_slam = 0.0
        self.leg_origin, self.target = (0.0, 0.0), (self.square_size_m, 0.0)
        self.progress = self.drive_used = self.scan_used = 0.0
        self.last_command, self.last_tick = (0.0, 0.0), None
        self.phase_started = None
        self.turn = None
        self.vision = VisualFeedback(self.policy)

    def start(self):
        """Start only an idle mission; a stopped run requires reset/relaunch."""
        if self.turn_calibration is None:
            return False, 'Set measured CCW and CW calibration first.'
        if self.state != NavigationState.IDLE:
            return False, 'Mission already started; reset/relaunch for a new run.'
        self.is_active, self.state = True, NavigationState.WAITING_FOR_POSE
        return True, 'Waiting for a pose and stable known-tag observations.'

    def stop(self, reason='Manual stop requested'):
        """Latch a terminal stop; ordinary start requests cannot resume motion."""
        self.stop_reason, self.is_active = reason, False
        self.state = NavigationState.STOPPED
        return True, reason

    def update_ekf_pose(self, x_slam, y_slam, theta_slam, sigma_pos_m, sigma_theta_rad, stamp_s):
        """Cache finite SLAM pose and latch its origin after an explicit start.

        Args:
            x_slam: Robot x coordinate in the SLAM frame, in meters.
            y_slam: Robot y coordinate in the SLAM frame, in meters.
            theta_slam: SLAM yaw in radians, positive counterclockwise.
            sigma_pos_m: Planar position standard deviation in meters.
            sigma_theta_rad: Heading standard deviation in radians.
            stamp_s: ROS pose time in seconds.

        Returns:
            None. Invalid feedback invalidates the cached timestamp.
        """
        values = (x_slam, y_slam, theta_slam, sigma_pos_m, sigma_theta_rad, stamp_s)
        if not all(math.isfinite(v) for v in values) or min(sigma_pos_m, sigma_theta_rad) < 0:
            self.latest_pose_stamp_s = -math.inf
            return
        self.latest_slam_pose = (x_slam, y_slam, wrap_angle(theta_slam))
        self.latest_uncertainty, self.latest_pose_stamp_s = (sigma_pos_m, sigma_theta_rad), stamp_s
        if self.is_active and not self.origin_latched:
            self.x0_slam, self.y0_slam, self.theta0_slam = self.latest_slam_pose
            self.origin_latched = True

    def slam_to_nav_frame(self, x_slam, y_slam, theta_slam):
        """Return (x_m, y_m, yaw_rad) in the initial robot-centered frame."""
        dx, dy = x_slam - self.x0_slam, y_slam - self.y0_slam
        c, s = math.cos(self.theta0_slam), math.sin(self.theta0_slam)
        return c * dx + s * dy, -s * dx + c * dy, wrap_angle(theta_slam - self.theta0_slam)

    def update_vision(self, stamp, tags, known_ids, clearance):
        """Pass timestamped base-frame observations to the independent vision tracker."""
        if self.latest_slam_pose is not None:
            self.vision.update(stamp, tags, known_ids, clearance,
                               self.latest_slam_pose, self._ekf_progress())

    def _heading(self):
        """Return the absolute SLAM heading anchored to the initial robot direction."""
        return self.theta0_slam + self.side_index * math.pi / 2

    def _heading_error(self):
        """Return wrapped error to the fixed side heading, never to a moving waypoint."""
        return wrap_angle(self._heading() - self.latest_slam_pose[2])

    def _ekf_progress(self):
        """Project the current EKF position onto this leg's fixed forward axis."""
        x, y, _ = self.latest_slam_pose
        return ((x - self.leg_origin[0]) * math.cos(self._heading())
                + (y - self.leg_origin[1]) * math.sin(self._heading()))

    def _transition(self, state, now):
        """Enter a phase and restart its elapsed-time deadline."""
        self.state, self.phase_started = state, now

    def _pose_problem(self, now):
        """Return the reason feedback cannot authorize motion, or None."""
        if not self.origin_latched or self.latest_slam_pose is None:
            return 'Waiting for initial EKF pose'
        if not -0.05 <= now - self.latest_pose_stamp_s <= self.pose_timeout_s:
            return 'Pose feedback is stale or invalid'
        if (self.latest_uncertainty[0] > self.max_pos_uncertainty_m
                or self.latest_uncertainty[1] > self.max_heading_uncertainty_rad):
            return 'Pose uncertainty exceeds navigation limits'
        return None

    def _account_motion(self, now):
        """Bound command accounting by the motor watchdog; never assume a fixed tick."""
        dt = 0.0 if self.last_tick is None else max(0.0, min(now - self.last_tick, 0.15))
        self.drive_used += abs(self.last_command[0]) * dt
        if self.state == NavigationState.SCANNING:
            self.scan_used += abs(self.last_command[1]) * dt
        self.last_tick = now

    def _output(self, message, linear=0.0, angular=0.0):
        """Build telemetry and remember the issued command for budget accounting."""
        pose = self.slam_to_nav_frame(*self.latest_slam_pose) if self.latest_slam_pose else (0, 0, 0)
        target = self.slam_to_nav_frame(*self.target, self._heading())[:2]
        error = self._heading_error() if self.latest_slam_pose else 0.0
        self.last_command = linear, angular
        return NavigationOutput(linear, angular, self.state, message, self.current_wp_idx,
                                *target, *pose, max(0.0, self.square_size_m - self.progress),
                                error, *self.latest_uncertainty)

    def _fail(self, reason):
        """Latch failure and return an immediate zero command."""
        self.stop(reason)
        return self._output(reason)

    def step(self, current_time_s):
        """Advance one nonblocking phase with stale-data and bounded-motion guards.

        Args:
            current_time_s: ROS controller time in seconds.

        Returns:
            NavigationOutput with stopped commands on idle, waits, completion, or failure.
        """
        now = float(current_time_s)
        self._account_motion(now)
        if not self.is_active:
            return self._output(self.stop_reason)
        if self.phase_started is None:
            self.phase_started = now
        problem = self._pose_problem(now)
        if problem and self.state != NavigationState.WAITING_FOR_POSE:
            return self._fail(problem)
        handlers = {NavigationState.WAITING_FOR_POSE: self._wait_for_start,
                    NavigationState.DRIVING: self._drive, NavigationState.STOPPING: self._settle,
                    NavigationState.SCANNING: self._scan, NavigationState.LOCALIZING: self._localize,
                    NavigationState.ALIGNING: self._align}
        return handlers[self.state](now)

    def _wait_for_start(self, now):
        """Require stable observed landmarks before authorizing the first straight leg."""
        if now - self.phase_started > self.policy.localization_timeout_s:
            return self._fail('Initial visual localization timed out')
        if self._pose_problem(now) or not self.vision.stable(now, self.phase_started):
            return self._output('Waiting for stable known-tag localization')
        self._begin_leg(now)
        return self._output('Visual start verified; ready to drive')

    def _begin_leg(self, now):
        """Anchor this side at the current pose without changing its absolute heading."""
        self.leg_origin = self.latest_slam_pose[:2]
        c, s = math.cos(self._heading()), math.sin(self._heading())
        self.target = (self.leg_origin[0] + c * self.square_size_m,
                       self.leg_origin[1] + s * self.square_size_m)
        self.progress = self.drive_used = 0.0
        self.vision.begin_leg(self._heading())
        self._transition(NavigationState.DRIVING, now)

    def _drive(self, now):
        """Prefer same-tag travel, stop at clearance limits, and never pivot mid-side.

        Args:
            now: ROS controller time in seconds.

        Returns:
            NavigationOutput for forward motion, arrival, or terminal feedback failure.
        """
        if not self.vision.fresh(now):
            return self._fail('Lost fresh tag observations while driving')
        if self.vision.clearance is not None and self.vision.clearance <= self.policy.wall_stop_distance_m:
            return self._fail('Visible wall inside forward clearance limit')
        if self.drive_used > self.policy.drive_budget_factor * self.square_size_m:
            return self._fail('Straight command budget exhausted before visual arrival')
        self.progress = self._ekf_progress()
        if now - self.vision.progress_stamp <= self.policy.vision_timeout_s:
            self.progress = self.vision.extrapolated_progress(self.progress)
        remaining = self.square_size_m - self.progress
        if remaining <= self.waypoint_tolerance_m:
            self._transition(NavigationState.STOPPING, now)
            return self._output('Six-foot leg reached; stopping before scan')
        if abs(self._heading_error()) > self.policy.drive_heading_limit_rad:
            return self._fail('Heading drift exceeded limit during straight travel')
        speed = max(self.min_linear_v, min(self.max_linear_v, self.linear_kp * remaining))
        return self._output('Driving straight with visual progress feedback', linear=speed)

    def _settle(self, now):
        """Stop for settling, then begin one bounded nominal full-turn scan."""
        if now - self.phase_started >= self.turn_settle_s:
            self.scan_used = 0.0
            self._transition(NavigationState.SCANNING, now)
        return self._output('Stopped before full-turn scan')

    def _scan(self, now):
        """Scan at the measured CCW input; timed completion is not visual verification."""
        if now - self.phase_started > self.policy.scan_timeout_s:
            return self._fail('Full-turn scan exceeded its time limit')
        if self.scan_used >= 2 * math.pi:
            self._transition(NavigationState.LOCALIZING, now)
            return self._output('Nominal 360 scan complete; waiting for visual localization')
        return self._output('Nominal 360 scan; collecting tags', angular=self.turn_calibration.rate(1))

    def _localize(self, now):
        """Require fresh stable mapped-tag observations after the scan, then align."""
        if now - self.phase_started > self.policy.localization_timeout_s:
            return self._fail('Post-scan visual localization timed out')
        if not self.vision.stable(now, self.phase_started + self.turn_settle_s):
            return self._output('Stopped; gathering stable post-scan known-tag updates')
        self.side_index += 1
        self.turn = None
        self._transition(NavigationState.ALIGNING, now)
        return self._output('Localized; aligning to initial heading plus 90-degree side increments')

    def _align(self, now):
        """Apply bounded fixed-rate pulses and verify alignment with stopped vision.

        Args:
            now: ROS controller time in seconds.

        Returns:
            NavigationOutput; no leg begins on timed prediction alone.
        """
        if now - self.phase_started > self.policy.localization_timeout_s + 10:
            return self._fail('Alignment or visual verification timed out')
        if self.turn is None:
            self.turn = FixedTurnPulse(self.turn_calibration, abs(self._heading_error()),
                                       self.turn_pulse_rad, self.turn_settle_s)
        omega, message = self.turn.step(now, self._heading_error(), self.heading_tolerance_rad)
        if self.turn.exhausted:
            return self._fail('Alignment angle budget exhausted')
        if omega and not self.vision.fresh(now):
            self.turn.pause(now)
            return self._output('Stopped alignment; waiting for fresh tags')
        verified = self.turn.done and omega == 0 and abs(self._heading_error()) <= self.heading_tolerance_rad
        if verified and self.vision.stable(now, max(self.phase_started, self.turn.ready_at)):
            return self._finish_alignment(now)
        return self._output(message, angular=omega)

    def _finish_alignment(self, now):
        """Finish four sides at the initial heading or begin the next fixed-heading leg."""
        self.turn = None
        if self.side_index == 4:
            self.state, self.is_active = NavigationState.COMPLETED, False
            return self._output('Four legs and scans complete; final heading visually verified')
        self.current_wp_idx = self.side_index + 1
        self._begin_leg(now)
        return self._output('Next side visually aligned; ready to drive')


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
