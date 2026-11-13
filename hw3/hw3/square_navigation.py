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
            max_pos_uncertainty_m: Confidence limit for optional EKF corrections, in meters.
            max_heading_uncertainty_rad: Confidence limit for optional EKF yaw corrections, in radians.
            turn_calibration: Measured direction-specific motor inputs and rad/s rates.
            turn_pulse_rad: Maximum alignment correction pulse angle, in radians.
            turn_settle_s: Stopped settling time after motion, in seconds.
            policy: NavigationPolicy with vision, stability, clearance, and time limits.

        Returns:
            None. The first leg needs a fresh estimator heartbeat, but no visible tags.
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
        self.scan_stage_index, self.scan_ready_at = 0, 0.0
        self.last_command, self.last_tick = (0.0, 0.0), None
        self.phase_started = None
        self.navigation_yaw, self.heading_time = 0.0, -math.inf
        self.turn = None
        self.vision = VisualFeedback(self.policy)

    def start(self):
        """Start only an idle mission; a stopped run requires reset/relaunch."""
        if self.turn_calibration is None:
            return False, 'Set measured CCW and CW calibration first.'
        if self.state != NavigationState.IDLE:
            return False, 'Mission already started; reset/relaunch for a new run.'
        self.is_active, self.state = True, NavigationState.WAITING_FOR_POSE
        return True, 'Starting first six-foot leg; no prior tags required.'

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
            self.navigation_yaw, self.heading_time = theta_slam, stamp_s
        elif self.vision.fresh(stamp_s) and self._uncertainty_ok():
            self.navigation_yaw, self.heading_time = theta_slam, stamp_s

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
        return wrap_angle(self._heading() - self.navigation_yaw)

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
        return None

    def _uncertainty_ok(self):
        """Select trustworthy optional EKF heading corrections without blocking motion."""
        return (self.latest_uncertainty[0] <= self.max_pos_uncertainty_m
                and self.latest_uncertainty[1] <= self.max_heading_uncertainty_rad)

    def _account_motion(self, now):
        """Integrate travel and fallback heading without requiring visual observations.

        Args:
            now: Current ROS controller time in seconds.

        Returns:
            None. Integrates each held-command interval once, limited by the motor
            watchdog; an EKF heading correction already covers time up to its stamp.
        """
        dt = 0.0 if self.last_tick is None else max(0.0, min(now - self.last_tick, 0.15))
        heading_start = max(self.last_tick if self.last_tick is not None else now, self.heading_time)
        heading_dt = max(0.0, min(now - heading_start, 0.15))
        self.navigation_yaw = wrap_angle(self.navigation_yaw + self.last_command[1] * heading_dt)
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
        """Begin the first leg on a fresh pose heartbeat without any landmark prerequisite."""
        if now - self.phase_started > self.policy.pose_wait_timeout_s:
            return self._fail('Initial estimator pose timed out')
        if self._pose_problem(now):
            return self._output('Waiting for initial estimator pose; tags are optional')
        self._begin_leg(now)
        return self._drive(now)

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
        if self.vision.wall_too_close(now):
            return self._fail('Visible wall inside forward clearance limit')
        self.progress = self.drive_used
        if now - self.vision.progress_stamp <= self.policy.vision_timeout_s:
            self.progress = self.vision.extrapolated_progress(self._ekf_progress())
        remaining = self.square_size_m - self.progress
        budget_reached = self.drive_used >= self.policy.drive_budget_factor * self.square_size_m
        if remaining <= self.waypoint_tolerance_m or budget_reached:
            self._transition(NavigationState.STOPPING, now)
            return self._output('Leg travel target or nominal budget reached; stopping before scan')
        speed = max(self.min_linear_v, min(self.max_linear_v, self.linear_kp * remaining))
        source = 'calibrated command prediction; vision optional'
        if now - self.vision.progress_stamp <= self.policy.vision_timeout_s:
            source = 'visual progress feedback'
        return self._output('Driving straight with ' + source, linear=speed)

    def _settle(self, now):
        """Stop for settling, then begin one bounded nominal full-turn scan."""
        if now - self.phase_started >= self.turn_settle_s:
            self.scan_used = 0.0
            self.scan_stage_index, self.scan_ready_at = 0, now
            self._transition(NavigationState.SCANNING, now)
        return self._output('Stopped before full-turn scan')

    def _scan(self, now):
        """Alternate bounded nominal turn stages with stopped camera-acquisition pauses.

        Args:
            now: Current ROS controller time in seconds.

        Returns:
            NavigationOutput for a stage, camera pause, timeout, or stopped completion.
            Stage angles are command predictions, not measured physical rotation.
        """
        if now - self.phase_started > self.policy.scan_timeout_s:
            return self._fail('Full-turn scan exceeded its time limit')
        if now < self.scan_ready_at:
            return self._output('Scan paused; allowing camera frames to arrive')
        if self.scan_stage_index >= math.ceil(2 * math.pi / self.policy.scan_step_rad):
            self._transition(NavigationState.LOCALIZING, now)
            return self._output('Staged 360 scan complete; settling with optional vision updates')
        goal = min(2 * math.pi, (self.scan_stage_index + 1) * self.policy.scan_step_rad)
        if self.scan_used >= goal:
            return self._pause_scan_stage(now)
        return self._output('Scan stage: fixed-rate turn', angular=self.turn_calibration.rate(1))

    def _pause_scan_stage(self, now):
        """Stop after each stage, including the last, before accepting another phase."""
        self.scan_stage_index += 1
        self.scan_ready_at = now + self.policy.scan_pause_s
        return self._output(f'Scan stage {self.scan_stage_index} complete; stopped for camera')

    def _localize(self, now):
        """Settle for a fixed duration, accepting optional vision without requiring tags."""
        if now - self.phase_started < self.turn_settle_s:
            return self._output('Post-scan settle; tag observations are optional')
        self.side_index += 1
        self.turn = None
        self._transition(NavigationState.ALIGNING, now)
        return self._output('Aligning to next absolute side heading using the best available estimate')

    def _align(self, now):
        """Align with calibrated prediction and optional visual heading corrections.

        Args:
            now: ROS controller time in seconds.

        Returns:
            NavigationOutput. Absent tags never block a leg; exhausted correction
            budgets continue best effort without claiming visual verification.
        """
        if now - self.phase_started > self.policy.alignment_timeout_s:
            return self._finish_alignment(now, 'Alignment timeout; continuing best effort')
        if self.turn is None:
            self.turn = FixedTurnPulse(self.turn_calibration, abs(self._heading_error()),
                                       self.turn_pulse_rad, self.turn_settle_s)
        omega, message = self.turn.step(now, self._heading_error(), self.heading_tolerance_rad)
        if self.turn.exhausted:
            return self._finish_alignment(now, 'Alignment correction budget exhausted; continuing best effort')
        if self.turn.done and omega == 0 and abs(self._heading_error()) <= self.heading_tolerance_rad:
            return self._finish_alignment(now, 'Aligned by best available heading; vision optional')
        return self._output(message, angular=omega)

    def _finish_alignment(self, now, message):
        """Finish four sides or begin the next leg without requiring a visible tag."""
        self.turn = None
        if self.side_index == 4:
            self.state, self.is_active = NavigationState.COMPLETED, False
            return self._output('Four legs and scans complete. ' + message)
        self.current_wp_idx = self.side_index + 1
        self._begin_leg(now)
        return self._output(message)


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
