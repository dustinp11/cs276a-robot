"""Regular polygon legs with calibrated corner turns and optional visual corrections."""

from dataclasses import dataclass
from enum import Enum
import math

from .coordinates import wrap_angle
from .late_navigation import LateNavigation
from .turn_calibration import ForwardCalibration
from .vision_navigation import NavigationPolicy, VisualFeedback, relative_tag_turn, startup_view_match


class NavigationState(Enum):
    """Motion phases and terminal outcomes recorded in navigation_log.csv."""
    IDLE = 'IDLE'
    WAITING_FOR_POSE = 'WAITING_FOR_POSE'
    STARTUP_SCAN_TURN = 'STARTUP_SCAN_TURN'
    STARTUP_SCAN_PAUSE = 'STARTUP_SCAN_PAUSE'
    STARTUP_SCAN_CORRECTING = 'STARTUP_SCAN_CORRECTING'
    STARTUP_VIEW_SEARCH = 'STARTUP_VIEW_SEARCH'
    DRIVING = 'DRIVING'
    MIDPOINT_PAUSE = 'MIDPOINT_PAUSE'
    MIDPOINT_CORRECTING = 'MIDPOINT_CORRECTING'
    STOPPING = 'STOPPING'
    TURNING = 'TURNING'
    TURN_STAGE_PAUSE = 'TURN_STAGE_PAUSE'
    CORNER_PAUSE = 'CORNER_PAUSE'
    CORRECTING = 'CORRECTING'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'
    RELOCALIZING_TURN = 'RELOCALIZING_TURN'
    RELOCALIZING_PAUSE = 'RELOCALIZING_PAUSE'
    REORIENTING = 'REORIENTING'
    REORIENT_PAUSE = 'REORIENT_PAUSE'
    RETURN_DRIVING = 'RETURN_DRIVING'
    RETURN_PAUSE = 'RETURN_PAUSE'


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
    """Execute configured-length legs using tag displacement and fixed initial-frame headings."""

    FEET_TO_METERS = 0.3048

    def __init__(self, square_size_ft=6.0, waypoint_tolerance_m=0.01,
                 heading_tolerance_rad=0.12, pose_timeout_s=0.60,
                 max_pos_uncertainty_m=0.40, max_heading_uncertainty_rad=0.40,
                 turn_calibration=None, turn_pulse_rad=math.pi / 12,
                 turn_settle_s=3.0, policy=None, forward_calibration=None,
                 side_count=4, side_length_m=0.0):
        """Configure a stopped regular-polygon mission without supplying a surveyed map.

        Args:
            square_size_ft: Legacy side length in feet when side_length_m is zero.
            side_count: Number of sides; exterior left turns are 2*pi/side_count.
            side_length_m: Side length in meters, or zero to use square_size_ft.
            waypoint_tolerance_m: Allowed shortfall per leg in meters.
            heading_tolerance_rad: Allowed absolute side-heading error in radians.
            pose_timeout_s: Maximum EKF pose age in seconds.
            max_pos_uncertainty_m: Confidence limit for optional EKF corrections, in meters.
            max_heading_uncertainty_rad: Confidence limit for optional EKF corrections, in radians.
            turn_calibration: Measured direction-specific motor inputs and rad/s rates.
            turn_pulse_rad: Maximum alignment correction pulse angle, in radians.
            turn_settle_s: Stopped settling time after motion, in seconds.
            policy: NavigationPolicy with optional vision freshness and phase timing settings.
            forward_calibration: Fixed straight motor pair and measured speed in m/s.

        Returns:
            None. Mission start requires neither EKF messages nor visible tags.
        """
        if not math.isfinite(side_count) or int(side_count) != side_count or side_count < 3:
            raise ValueError('Polygon side count must be an integer of at least three.')
        if not math.isfinite(side_length_m) or side_length_m < 0:
            raise ValueError('Side length must be finite and nonnegative.')
        self.side_count = int(side_count)
        self.corner_angle_rad = 2 * math.pi / self.side_count
        self.square_size_m = side_length_m or float(square_size_ft) * self.FEET_TO_METERS
        self.forward = forward_calibration or ForwardCalibration()
        self.waypoint_tolerance_m = waypoint_tolerance_m
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
        limits = (self.square_size_m, self.forward.speed_mps,
                  self.waypoint_tolerance_m, self.heading_tolerance_rad, self.pose_timeout_s,
                  self.max_pos_uncertainty_m, self.max_heading_uncertainty_rad,
                  self.turn_pulse_rad, self.turn_settle_s)
        if not all(math.isfinite(v) and v > 0 for v in limits):
            raise ValueError('Navigation limits must be finite and positive.')
        if self.turn_pulse_rad > math.pi / 2:
            raise ValueError('Turn pulses must not exceed 90 nominal degrees.')

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
        self.progress = self.drive_used = 0.0
        self.corner_used = self.correction_used = self.correction_signed = 0.0
        self.corner_reference, self.corner_source = {}, 'prediction'
        self.corner_residual = None
        self.corner_start_heading = self.correction_goal = 0.0
        self.corner_stage_goal, self.corner_stage_done = 0.0, True
        self.correction_count = 0
        self.last_command, self.last_tick = (0.0, 0.0), None
        self.phase_started = None
        self.navigation_yaw, self.heading_time = 0.0, -math.inf
        self.navigation_x = self.navigation_y = 0.0
        self.course_heading = None
        self._reset_midpoint()
        self.vision = VisualFeedback(self.policy)
        self.late = LateNavigation(self, NavigationState)
        self._reset_startup_scan()

    def _reset_startup_scan(self):
        """Clear startup scan progress without moving the recorded start frame.

        Args:
            None.

        Returns:
            None. Clears initial view snapshots, search budgets, and correction counters.
        """
        self.startup_scan_used = self.startup_scan_goal = 0.0
        self.startup_scan_nominal_progress = self.startup_scan_step_goal = 0.0
        self.startup_reference = {}
        self.startup_view_reference, self.startup_view_current, self.startup_view_ids = {}, {}, []
        self.startup_view_matched = False
        self.startup_search_used = self.startup_search_signed = self.startup_search_goal = 0.0
        self.startup_search_omega = 0.0
        self.startup_correction_count = 0
        self.startup_correction_used = self.startup_correction_signed = 0.0
        self.startup_correction_goal = self.startup_correction_omega = 0.0
        self.startup_source, self.startup_heading_residual = 'not started', None

    def start(self):
        """Start only an idle mission; a stopped run requires reset/relaunch."""
        if self.turn_calibration is None:
            return False, 'Set measured CCW and CW calibration first.'
        if self.state != NavigationState.IDLE:
            return False, 'Mission already started; reset/relaunch for a new run.'
        self.is_active, self.state = True, NavigationState.WAITING_FOR_POSE
        message = 'Starting timed turn-and-drive mission' if self.policy.manual_motion_enabled else 'Starting first leg; no prior tags required.'
        return True, message

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
            self.navigation_x, self.navigation_y = x_slam, y_slam
        elif self.vision.fresh(stamp_s) and self._uncertainty_ok():
            self.navigation_yaw, self.heading_time = theta_slam, stamp_s
            self.navigation_x, self.navigation_y = x_slam, y_slam

    def slam_to_nav_frame(self, x_slam, y_slam, theta_slam):
        """Return (x_m, y_m, yaw_rad) in the initial robot-centered frame."""
        dx, dy = x_slam - self.x0_slam, y_slam - self.y0_slam
        c, s = math.cos(self.theta0_slam), math.sin(self.theta0_slam)
        return c * dx + s * dy, -s * dx + c * dy, wrap_angle(theta_slam - self.theta0_slam)

    def update_vision(self, stamp, tags, known_ids, clearance, tag_yaws=None,
                      landmark_sigmas=None, ekf_heading=None):
        """Pass timestamped base-frame observations to the independent vision tracker."""
        self.vision.update(stamp, tags, known_ids, clearance,
                           self._best_pose(), self._ekf_progress(), tag_yaws, landmark_sigmas, ekf_heading)

    def _heading(self):
        """Return the absolute SLAM heading anchored to the initial robot direction."""
        return (self.theta0_slam + self.side_index * self.corner_angle_rad
                if self.course_heading is None else self.course_heading)

    def _heading_error(self):
        """Return wrapped error to the fixed side heading, never to a moving waypoint."""
        return wrap_angle(self._heading() - self.navigation_yaw)

    def _ekf_progress(self):
        """Project the current EKF position onto this leg's fixed forward axis."""
        x, y, _ = self._best_pose()
        return ((x - self.leg_origin[0]) * math.cos(self._heading())
                + (y - self.leg_origin[1]) * math.sin(self._heading()))

    def _transition(self, state, now):
        """Enter a phase and restart its elapsed-time deadline."""
        self.state, self.phase_started = state, now

    def _best_pose(self):
        """Use a fresh EKF pose if available, otherwise use calibrated command prediction."""
        now = self.last_tick
        if self.latest_slam_pose is not None and now is not None:
            if -0.05 <= now - self.latest_pose_stamp_s <= self.pose_timeout_s:
                return self.latest_slam_pose
        return self.navigation_x, self.navigation_y, self.navigation_yaw

    def _uncertainty_ok(self):
        """Select trustworthy optional EKF corrections without blocking motion."""
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
        self.navigation_x += self.last_command[0] * heading_dt * math.cos(self.navigation_yaw)
        self.navigation_y += self.last_command[0] * heading_dt * math.sin(self.navigation_yaw)
        self.navigation_yaw = wrap_angle(self.navigation_yaw + self.last_command[1] * heading_dt)
        self.drive_used += abs(self.last_command[0]) * dt
        if self.state == NavigationState.STARTUP_SCAN_TURN:
            self.startup_scan_used += abs(self.last_command[1]) * dt
        elif self.state == NavigationState.STARTUP_VIEW_SEARCH:
            self.startup_search_used += abs(self.last_command[1]) * dt
            self.startup_search_signed += self.last_command[1] * dt
        elif self.state == NavigationState.STARTUP_SCAN_CORRECTING:
            self.startup_correction_used += abs(self.last_command[1]) * dt
            self.startup_correction_signed += self.last_command[1] * dt
        elif self.state == NavigationState.TURNING:
            self.corner_used += abs(self.last_command[1]) * dt
        elif self.state == NavigationState.MIDPOINT_CORRECTING:
            self.midpoint_correction_used += abs(self.last_command[1]) * dt
            self.midpoint_correction_signed += self.last_command[1] * dt
        elif self.state == NavigationState.CORRECTING:
            self.correction_used += abs(self.last_command[1]) * dt
            self.correction_signed += self.last_command[1] * dt
        self.late.account_motion(*self.last_command, dt)
        self.last_tick = now

    def _output(self, message, linear=0.0, angular=0.0):
        """Build telemetry and remember the issued command for budget accounting."""
        pose = self.slam_to_nav_frame(*self._best_pose())
        target = self.slam_to_nav_frame(*self.target, self._heading())[:2]
        error = self._heading_error()
        self.last_command = linear, angular
        return NavigationOutput(linear, angular, self.state, message, self.current_wp_idx,
                                *target, *pose, self._remaining_distance(),
                                error, *self.latest_uncertainty)

    def _remaining_distance(self):
        """Report start/vertex distance during late navigation, otherwise the side remainder."""
        if self.late.active or (self.state == NavigationState.COMPLETED
                                and self.policy.late_scan_start_corner <= self.side_count):
            return math.dist((self.navigation_x, self.navigation_y), self.target)
        return max(0.0, self.square_size_m - self.progress)

    def step(self, current_time_s):
        """Advance one phase, using model prediction whenever observations are unavailable.

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
        if self.policy.manual_motion_enabled:
            return self._manual_step(now)
        if self.late.active:
            return self.late.step(now)
        handlers = {NavigationState.WAITING_FOR_POSE: self._wait_for_start,
                    NavigationState.STARTUP_SCAN_TURN: self._startup_scan_turn,
                    NavigationState.STARTUP_SCAN_PAUSE: self._startup_scan_pause,
                    NavigationState.STARTUP_SCAN_CORRECTING: self._startup_scan_correct,
                    NavigationState.STARTUP_VIEW_SEARCH: self._startup_view_search,
                    NavigationState.DRIVING: self._drive,
                    NavigationState.MIDPOINT_PAUSE: self._midpoint_pause,
                    NavigationState.MIDPOINT_CORRECTING: self._midpoint_correct,
                    NavigationState.STOPPING: self._settle,
                    NavigationState.TURNING: self._turn_corner,
                    NavigationState.TURN_STAGE_PAUSE: self._turn_stage_pause,
                    NavigationState.CORNER_PAUSE: self._corner_pause,
                    NavigationState.CORRECTING: self._correct_corner}
        return handlers[self.state](now)

    def _wait_for_start(self, now):
        """Latch the local start frame, then optionally scan before the first forward leg.

        Args:
            now: Current ROS time in seconds.

        Returns:
            Stopped scan output when enabled, otherwise the initial drive command.
            Neither branch waits for an EKF message or requires a visible tag.
        """
        if not self.origin_latched:
            self.x0_slam, self.y0_slam, self.theta0_slam = self._best_pose()
            self.navigation_yaw, self.heading_time = self.theta0_slam, now
            self.origin_latched = True
        if self.policy.manual_motion_enabled:
            return self._manual_begin_turn(now)
        if self.policy.startup_scan_enabled:
            self._transition(NavigationState.STARTUP_SCAN_PAUSE, now)
            return self._output('Startup scan: initial stopped acquisition before one full revolution')
        self._begin_leg(now)
        return self._drive(now)

    def _manual_step(self, now):
        """Run timed turns and forward halves separated by fixed manual pauses."""
        handlers = {NavigationState.WAITING_FOR_POSE: self._wait_for_start,
                    NavigationState.TURNING: self._manual_turn,
                    NavigationState.DRIVING: self._manual_drive,
                    NavigationState.MIDPOINT_PAUSE: self._manual_midpoint_pause,
                    NavigationState.STOPPING: self._manual_pause}
        return handlers[self.state](now)

    def _manual_begin_turn(self, now):
        """Start one fixed left turn; feedback remains available for logging only."""
        self.current_wp_idx = self.side_index + 1
        self.course_heading = self.theta0_slam + (self.side_index + 1) * self.corner_angle_rad
        self.corner_used = 0.0
        self.corner_source = 'Timed manual turn; no feedback corrections'
        self._transition(NavigationState.TURNING, now)
        return self._output('Manual left turn', angular=self.turn_calibration.rate(1))

    def _manual_turn(self, now):
        """Finish the timed turn and immediately begin the next straight segment."""
        if now - self.phase_started + 1e-9 >= self.policy.corner_turn_duration_s:
            self._begin_leg(now, heading=self.course_heading)
            return self._manual_drive(now)
        return self._output('Manual left turn', angular=self.turn_calibration.rate(1))

    def _manual_drive(self, now):
        """Count commanded travel only while moving; stop at the half- and full-side marks."""
        self.progress = self.drive_used
        distance = self.square_size_m if self.midpoint_done else self.square_size_m / 2
        if self.drive_used + 1e-9 >= distance:
            return self._manual_checkpoint(now)
        return self._output('Manual forward segment', linear=self.forward.speed_mps)

    def _manual_checkpoint(self, now):
        """Stop at the midpoint or count a completed side before the next turn."""
        if not self.midpoint_done:
            self._transition(NavigationState.MIDPOINT_PAUSE, now)
            return self._output('Manual midpoint pause')
        self.side_index += 1
        self._transition(NavigationState.STOPPING, now)
        return self._output('Manual side finished; pause before next turn')

    def _manual_midpoint_pause(self, now):
        """Wait one fixed interval without using vision to alter distance or heading.

        Args:
            now: Current ROS time in seconds.

        Returns:
            Stopped output during the pause, then the next forward command. Paused
            time never contributes to the nominal travel distance.
        """
        if now - self.phase_started + 1e-9 < self.policy.manual_midpoint_pause_s:
            return self._output('Manual midpoint pause')
        self.midpoint_done = True
        self.midpoint_progress_m, self.midpoint_remaining_m = self.drive_used, self.square_size_m - self.drive_used
        self.midpoint_source = 'Manual command prediction; no feedback adjustments'
        self._transition(NavigationState.DRIVING, now)
        return self._manual_drive(now)

    def _manual_pause(self, now):
        """Pause after the full side, immediately before the next turn, or final completion."""
        if now - self.phase_started + 1e-9 < self.policy.manual_forward_pause_s:
            return self._output('Manual post-forward pause')
        if self.side_index >= self.side_count:
            self.state, self.is_active = NavigationState.COMPLETED, False
            return self._output(f'Manual mission complete: {self.side_count} timed turns and sides')
        return self._manual_begin_turn(now)

    def _startup_scan_turn(self, now):
        """Rotate one calibrated scan step, then stop for stationary camera observations."""
        if (self.startup_scan_used + 1e-9 >= self.startup_scan_goal
                or now - self.phase_started >= self.policy.corner_timeout_s):
            self.startup_scan_nominal_progress = self.startup_scan_step_goal
            self._transition(NavigationState.STARTUP_SCAN_PAUSE, now)
            return self._output('Startup scan step finished; stopped for camera')
        return self._output('Startup scan: calibrated left rotation', angular=self.turn_calibration.rate(1))

    def _startup_view_match(self, now):
        """Compare the current stopped view with the initial relative tag positions."""
        self.startup_view_current = self.vision.stable_tag_positions(now, self.phase_started)
        return startup_view_match(self.startup_view_reference, self.startup_view_current, self.policy)

    def _startup_adjustment_budget(self):
        """Share one finite angular budget across startup search and visible-tag realignment."""
        return max(0.0, self.policy.startup_view_search_max_rad
                   - self.startup_search_used - self.startup_correction_used)

    def _select_startup_view_search(self, now):
        """Seek the original view after four turns without trusting a global SLAM heading.

        Args:
            now: Current ROS time after a stationary acquisition interval, in seconds.

        Returns:
            True when a bounded reacquisition pivot is selected; False otherwise.
            Missing original tags initially require no search. Once visible, the
            signed bearing error permits either direction for coarse alignment.
        """
        budget = self._startup_adjustment_budget()
        if (not self.startup_view_reference or budget < self.policy.corner_min_correction_rad
                or self.startup_correction_count >= self.policy.corner_max_corrections):
            return False
        current = self.vision.stable_tag_positions(now, self.phase_started)
        match = self._startup_view_match(now)
        if match is None:
            if self.startup_view_reference.keys() & current.keys():
                return False
            error = self.policy.startup_view_search_step_rad
        else:
            error, _, complete = match
            fine_budget = min(budget, max(0, self.policy.corner_max_corrections
                              - self.startup_correction_count) * self.policy.startup_view_search_step_rad)
            if complete or abs(error) <= max(0.0, fine_budget) + self.policy.startup_view_bearing_tolerance_rad:
                return False
        angle = min(abs(error), self.policy.startup_view_search_step_rad, budget)
        self.startup_search_goal = self.startup_search_used + angle
        self.startup_search_omega = self.turn_calibration.rate(error)
        self.startup_correction_count += 1
        self._transition(NavigationState.STARTUP_VIEW_SEARCH, now)
        return True

    def _startup_view_search(self, now):
        """Turn one small reacquisition step, then stop to compare original tag bearings."""
        if (self.startup_search_used + 1e-9 >= self.startup_search_goal
                or now - self.phase_started >= self.policy.corner_timeout_s):
            self._transition(NavigationState.STARTUP_SCAN_PAUSE, now)
            return self._output('Initial-view search step finished; stopped for camera')
        return self._output('Seeking original startup tag view', angular=self.startup_search_omega)

    def _startup_heading_error(self, now):
        """Check closure against the initial direction without redefining the start frame.

        Args:
            now: Current ROS time after the final stopped scan acquisition, in seconds.

        Returns:
            Signed correction in radians, or None without useful visual feedback.
            Same-tag orientation takes priority over a confidence-gated EKF heading.
        """
        self.startup_view_matched, self.startup_view_ids = False, []
        match = self._startup_view_match(now)
        if match is not None:
            error, self.startup_view_ids, self.startup_view_matched = match
            self.navigation_yaw, self.heading_time = wrap_angle(self.theta0_slam - error), now
            state = 'matched' if self.startup_view_matched else 'partially reacquired'
            self.startup_source = f'Original startup view {state}; tags {self.startup_view_ids}'
            return 0.0 if self.startup_view_matched else error
        yaws = self.vision.stable_tag_yaws(now, self.phase_started)
        measured = relative_tag_turn(self.startup_reference, yaws,
                                     self.startup_scan_nominal_progress + self.startup_search_signed + self.startup_correction_signed)
        if measured is not None:
            self.navigation_yaw = wrap_angle(self.theta0_slam + measured)
            self.heading_time, self.startup_source = now, 'Initial-tag orientation closure; original view not verified'
            return wrap_angle(-measured)
        yaw = self.vision.confident_ekf_yaw(now, self.phase_started)
        if yaw is not None and self.vision.known_ids.intersection(yaws) and self.policy.corner_ekf_weight > 0:
            self.navigation_yaw, self.heading_time = yaw, now
            self.startup_source = 'Confident EKF startup-heading closure; original view not verified'
            return self.policy.corner_ekf_weight * wrap_angle(self.theta0_slam - yaw)
        self.startup_source = 'Calibrated scan closure; original view not verified'
        return None

    def _select_startup_correction(self, error, now):
        """Select a bounded signed closure pulse after the full scan.

        Args:
            error: Signed observed heading correction in radians, or None.
            now: Current ROS time in seconds.

        Returns:
            True when a pulse is selected; False when the first forward leg should start.
        """
        budget = self._startup_adjustment_budget()
        angle = min(abs(error), self.policy.startup_view_search_step_rad, budget) if error is not None else 0.0
        if angle + 1e-9 < self.policy.corner_min_correction_rad or self.startup_correction_count >= self.policy.corner_max_corrections:
            return False
        self.startup_correction_goal = self.startup_correction_used + angle
        self.startup_correction_omega = self.turn_calibration.rate(error)
        self.startup_correction_count += 1
        self._transition(NavigationState.STARTUP_SCAN_CORRECTING, now)
        return True

    def _startup_scan_pause(self, now):
        """Acquire observations between scan steps, then proceed without requiring tags.

        Args:
            now: Current ROS time in seconds.

        Returns:
            Stopped NavigationOutput selecting another scan step, a bounded closure
            correction, or the first forward leg after one complete nominal revolution.
        """
        if now - self.phase_started < self.policy.startup_scan_pause_s:
            return self._output('Startup scan pause; allowing camera and EKF updates')
        if self.startup_scan_nominal_progress == 0:
            self.startup_reference = self.vision.stable_tag_yaws(now, self.phase_started)
            self.startup_view_reference = self.vision.stable_tag_positions(now, self.phase_started)
            self.startup_source = f'Initial view captured: tags {sorted(self.startup_view_reference)}'
        if self.startup_scan_nominal_progress + 1e-9 < 2 * math.pi:
            remaining = min(self.policy.startup_scan_step_rad, 2 * math.pi - self.startup_scan_nominal_progress)
            duration = self.policy.startup_scan_step_duration_s * remaining / self.policy.startup_scan_step_rad
            self.startup_scan_step_goal = self.startup_scan_nominal_progress + remaining
            self.startup_scan_goal = self.startup_scan_used + self.turn_calibration.ccw_radps * duration
            self._transition(NavigationState.STARTUP_SCAN_TURN, now)
            return self._output('Startup scan: next stopped-to-stopped rotation step ready')
        if self._select_startup_view_search(now):
            return self._output('Startup scan: bounded search for original tag IDs and bearings')
        self.startup_heading_residual = self._startup_heading_error(now)
        if self._select_startup_correction(self.startup_heading_residual, now):
            return self._output('Startup scan heading correction selected; ' + self.startup_source)
        self._begin_leg(now)
        return self._output('Startup scan complete; first forward leg ready; ' + self.startup_source)

    def _startup_scan_correct(self, now):
        """Apply a closure pivot, then acquire another stopped observation before proceeding."""
        if (self.startup_correction_used + 1e-9 >= self.startup_correction_goal
                or now - self.phase_started >= self.policy.corner_timeout_s):
            self._transition(NavigationState.STARTUP_SCAN_PAUSE, now)
            return self._output('Startup closure pulse finished; stopped to recheck heading')
        return self._output('Startup scan closure correction', angular=self.startup_correction_omega)

    def _begin_leg(self, now, heading=None, target=None):
        """Start a nominal side, optionally aimed toward a fixed late-mission vertex.

        Args:
            now: Current ROS time in seconds.
            heading: Optional fixed SLAM heading in radians for the upcoming leg.
            target: Optional fixed SLAM (x_m, y_m) vertex for telemetry.

        Returns:
            None. Starts ordinary calibrated forward travel and its existing midpoint pause.
        """
        self.course_heading = heading
        self.leg_origin = self._best_pose()[:2]
        c, s = math.cos(self._heading()), math.sin(self._heading())
        self.target = target if target is not None else (
            self.leg_origin[0] + c * self.square_size_m, self.leg_origin[1] + s * self.square_size_m)
        self.progress = self.drive_used = 0.0
        self._reset_midpoint()
        self.leg_reference = self.vision.stable_tag_yaws(now, self.phase_started)
        self.leg_start_heading = self.navigation_yaw
        self.vision.begin_leg(self._heading())
        self._transition(NavigationState.DRIVING, now)

    def _planned_vertex(self, completed_sides):
        """Return an ideal polygon vertex in the fixed SLAM start frame, in meters."""
        if completed_sides == self.side_count:
            return self.x0_slam, self.y0_slam
        angles = [self.theta0_slam + index * self.corner_angle_rad for index in range(completed_sides)]
        return (self.x0_slam + self.square_size_m * sum(math.cos(v) for v in angles),
                self.y0_slam + self.square_size_m * sum(math.sin(v) for v in angles))

    def _planned_center(self):
        """Return the ideal polygon center in the recorded start's SLAM frame, in meters."""
        vertices = [self._planned_vertex(index) for index in range(self.side_count)]
        return tuple(sum(values) / self.side_count for values in zip(*vertices))

    def _reset_midpoint(self):
        """Clear midpoint distance/heading budgets while preserving the fixed side course.

        Args:
            None.

        Returns:
            None. Resets per-leg estimates, reference normals, and correction counters.
        """
        self.midpoint_done = False
        self.midpoint_progress_m = self.midpoint_remaining_m = None
        self.midpoint_source = 'not reached'
        self.progress_correction_m = 0.0
        self.midpoint_correction_count = 0
        self.midpoint_correction_used = self.midpoint_correction_signed = 0.0
        self.midpoint_correction_goal = self.midpoint_correction_omega = 0.0
        self.midpoint_heading_source, self.midpoint_heading_residual = 'not reached', None
        self.leg_reference, self.leg_start_heading = {}, self.navigation_yaw

    def _first_half(self, now):
        """Drive half the nominal side using command prediction, then stop once."""
        self.progress = self.drive_used
        remaining = self.square_size_m / 2 - self.drive_used
        if remaining <= 0:
            self._transition(NavigationState.MIDPOINT_PAUSE, now)
            return self._output('Half-side reached; stopped for camera processing')
        return self._output('Driving nominal first half; no steering corrections',
                            linear=self.forward.speed_mps)

    def _midpoint_estimate(self, now):
        """Select a fresh post-stop EKF displacement or retain nominal half-side travel."""
        age = now - self.latest_pose_stamp_s
        post_stop = self.latest_pose_stamp_s >= self.phase_started
        if post_stop and -0.05 <= age <= self.pose_timeout_s and self._uncertainty_ok():
            return self._ekf_progress(), 'fresh EKF'
        return self.drive_used, 'command prediction fallback'

    def _apply_midpoint_correction(self, estimate):
        """Clamp the EKF distance adjustment around nominal command-integrated travel."""
        limit = min(self.policy.midpoint_max_correction_m, self.square_size_m / 4)
        delta = estimate - self.drive_used
        self.progress_correction_m = max(-limit, min(limit, delta))
        self.progress = self.drive_used + self.progress_correction_m
        self.midpoint_remaining_m = max(0.0, self.square_size_m - self.progress)

    def _update_midpoint_distance(self, now):
        """Update the bounded distance plan, retaining prior feedback if the new pose is stale."""
        estimate, source = self._midpoint_estimate(now)
        if source == 'fresh EKF' or self.midpoint_progress_m is None:
            self.midpoint_progress_m, self.midpoint_source = estimate, source
            self._apply_midpoint_correction(estimate)
        elif self.midpoint_source != 'command prediction fallback':
            self.midpoint_source = 'retained prior bounded midpoint estimate'

    def _midpoint_heading_error(self, now):
        """Prefer a confident fixed-course heading, otherwise measure same-tag drift.

        Args:
            now: Current ROS time after a stopped acquisition interval, in seconds.

        Returns:
            Signed correction in radians, or None without usable post-stop feedback.
            Relative normals maintain the leg's initial direction without a global map.
        """
        yaws = self.vision.stable_tag_yaws(now, self.phase_started)
        yaw = self.vision.confident_ekf_yaw(now, self.phase_started)
        if yaw is not None and self.vision.known_ids.intersection(yaws) and self.policy.corner_ekf_weight > 0:
            self.navigation_yaw, self.heading_time = yaw, now
            self.midpoint_heading_source = f'Confident EKF fixed-course heading tags {sorted(self.vision.known_ids)}'
            return self.policy.corner_ekf_weight * wrap_angle(self._heading() - yaw)
        measured = relative_tag_turn(self.leg_reference, yaws, self.midpoint_correction_signed)
        if measured is not None:
            self.navigation_yaw = wrap_angle(self.leg_start_heading + measured)
            self.heading_time, self.midpoint_heading_source = now, 'Stable same-tag straight-heading drift'
            return wrap_angle(-measured)
        reason = self.vision.ekf_rejection_reason or 'insufficient stable mapped-tag observations'
        self.midpoint_heading_source = 'No stable heading feedback; ' + reason
        return None

    def _select_midpoint_turn(self, error, now):
        """Select a useful heading pulse within the shared angle/count policy limits.

        Args:
            error: Signed observed heading correction in radians, or None.
            now: Current ROS time in seconds.

        Returns:
            True when a midpoint pivot is selected; False when forward travel should resume.
        """
        remaining = max(0.0, self.policy.corner_correction_limit_rad - self.midpoint_correction_used)
        angle = min(abs(error), self.turn_pulse_rad, remaining) if error is not None else 0.0
        if angle + 1e-9 < self.policy.corner_min_correction_rad or self.midpoint_correction_count >= self.policy.corner_max_corrections:
            return False
        self.midpoint_correction_goal = self.midpoint_correction_used + angle
        self.midpoint_correction_omega = self.turn_calibration.rate(error)
        self.midpoint_correction_count += 1
        self._transition(NavigationState.MIDPOINT_CORRECTING, now)
        return True

    def _midpoint_pause(self, now):
        """Recheck distance and heading after a fixed stationary acquisition interval.

        Args:
            now: Current ROS controller time in seconds.

        Returns:
            Stopped NavigationOutput selecting a heading correction or forward continuation.
            Every pulse gets another acquisition pause. Missing observations never extend
            the fixed wait; distance adjustments remain bounded around commanded travel.
        """
        if now - self.phase_started < self.policy.midpoint_pause_s:
            return self._output('Midpoint pause; checking remaining distance and optional heading feedback')
        self._update_midpoint_distance(now)
        self.midpoint_heading_residual = self._midpoint_heading_error(now)
        if self._select_midpoint_turn(self.midpoint_heading_residual, now):
            return self._output('Midpoint heading correction selected; ' + self.midpoint_heading_source)
        self.midpoint_done = True
        self._transition(NavigationState.DRIVING, now)
        return self._output(f'Midpoint {self.midpoint_source}; '
                            f'applied correction {self.progress_correction_m:+.3f} m; '
                            f'{self.midpoint_remaining_m:.3f} m remaining; ' + self.midpoint_heading_source)

    def _midpoint_correct(self, now):
        """Apply a fixed-rate pivot, then stop to recheck both distance and heading."""
        if (self.midpoint_correction_used + 1e-9 >= self.midpoint_correction_goal
                or now - self.phase_started >= self.policy.corner_timeout_s):
            self._transition(NavigationState.MIDPOINT_PAUSE, now)
            return self._output('Midpoint pivot finished; stopped to recheck distance and heading')
        return self._output('Midpoint heading correction from ' + self.midpoint_heading_source,
                            angular=self.midpoint_correction_omega)

    def _drive(self, now):
        """Drive the remainder after stopped distance and heading corrections; no live overrides.

        Args:
            now: ROS controller time in seconds.

        Returns:
            NavigationOutput for straight travel or a stopped midpoint/corner phase.
        """
        if not self.midpoint_done:
            return self._first_half(now)
        self.progress = self.drive_used + self.progress_correction_m
        remaining = self.square_size_m - self.progress
        budget_reached = self.drive_used >= self.policy.drive_budget_factor * self.square_size_m
        if remaining <= self.waypoint_tolerance_m or budget_reached:
            self._transition(NavigationState.STOPPING, now)
            return self._output('Leg travel target or nominal budget reached; stopping before corner turn')
        return self._output('Driving remaining distance on unchanged side heading',
                            linear=self.forward.speed_mps)

    def _settle(self, now):
        """Stop before capturing optional tag references and beginning a direct left turn."""
        if now - self.phase_started >= self.turn_settle_s:
            self._begin_corner(now)
        return self._output('Stopped before calibrated left corner')

    def _begin_corner(self, now):
        """Anchor one direct corner turn without requiring a visible landmark.

        Args:
            now: ROS controller time in seconds.

        Returns:
            None. Captures optional stopped tag normals and resets correction budgets.
        """
        self.course_heading = None
        self.corner_start_heading = self.navigation_yaw
        self.corner_reference = self.vision.stable_tag_yaws(now, self.phase_started)
        nominal = self.turn_calibration.ccw_radps * self.policy.corner_turn_duration_s
        self.corner_stage_goal = min(nominal, self.policy.corner_stage_angle_rad)
        self.corner_stage_done = self.corner_stage_goal >= nominal
        self.corner_used = self.correction_used = self.correction_signed = 0.0
        self.correction_count = 0
        self.corner_source = 'calibrated turn prediction'
        self.corner_residual = None
        self.side_index += 1
        if self.side_index >= self.policy.late_scan_start_corner:
            self.late.start(now, self.side_index)
        else:
            self._transition(NavigationState.TURNING, now)

    def _turn_corner(self, now):
        """Turn to the current stage budget, then stop for a bounded camera interval."""
        if (self.corner_used + 1e-9 >= self.corner_stage_goal
                or now - self.phase_started >= self.policy.corner_timeout_s):
            state = NavigationState.CORNER_PAUSE if self.corner_stage_done else NavigationState.TURN_STAGE_PAUSE
            self._transition(state, now)
            return self._output('Turn stage finished; stopped for camera processing')
        return self._output('Calibrated left corner stage', angular=self.turn_calibration.rate(1))

    def _relative_corner_measurement(self, now):
        """Measure turn rotation and transfer a visually anchored reference to overlapping tags.

        Args:
            now: Current ROS time after the stationary acquisition interval, in seconds.

        Returns:
            Measured total left rotation in radians, or None without stable shared tags.
            Newly visible tags become references only after a shared tag anchors rotation.
        """
        yaws = self.vision.stable_tag_yaws(now, self.phase_started)
        measured = relative_tag_turn(self.corner_reference, yaws, self.corner_used + self.correction_signed)
        if measured is None:
            return None
        for tag, yaw in yaws.items():
            self.corner_reference.setdefault(tag, wrap_angle(yaw + measured))
        self.navigation_yaw = wrap_angle(self.corner_start_heading + measured)
        self.heading_time, self.corner_source = now, 'stable same-tag face orientation'
        return measured

    def _plan_corner_remainder(self, measured):
        """Finish the calibrated remainder, allowing a bounded visually measured adjustment.

        Args:
            measured: Total relative rotation in radians, or None without usable vision.

        Returns:
            None. Updates the remaining turn budget. An actionable visual adjustment
            consumes one of the three attempts. Only extra left rotation consumes
            angle budget; omitting planned rotation leaves room for a right correction.
        """
        nominal = self.turn_calibration.ccw_radps * self.policy.corner_turn_duration_s
        remaining = max(0.0, nominal - self.corner_used)
        if measured is not None and self.correction_count < self.policy.corner_max_corrections:
            delta = max(-remaining, self.corner_angle_rad - measured - remaining)
            limit = max(0.0, self.policy.corner_correction_limit_rad - self.correction_used)
            delta = max(-limit, min(limit, delta))
            if abs(delta) + 1e-9 >= self.policy.corner_min_correction_rad:
                remaining += delta
                self.correction_used += max(0.0, delta)
                self.correction_count += 1
        self.corner_stage_goal = self.corner_used + remaining
        self.corner_stage_done = True

    def _turn_stage_pause(self, now):
        """Pause between stages, then finish using optional vision or the timed remainder."""
        if now - self.phase_started < self.turn_settle_s:
            return self._output('Intermediate corner pause; gathering optional stable tag observations')
        self._plan_corner_remainder(self._relative_corner_measurement(now))
        self._transition(NavigationState.TURNING, now)
        return self._output('Corner remainder ready; source: ' + self.corner_source)

    def _corner_error(self, now):
        """Prefer relative tag orientation, then optional fresh EKF heading corrections.

        Args:
            now: ROS controller time in seconds after a stopped acquisition interval.

        Returns:
            Signed residual in radians, or None without useful vision. A same-tag
            comparison corrects relative rotation to the configured exterior angle. EKF
            fallback uses an image-associated heading only with confident mapped tags,
            scaled by corner_ekf_weight.
            Missing or stale observations never authorize extra corrective rotation.
        """
        measured = self._relative_corner_measurement(now)
        if measured is not None:
            return wrap_angle(self.corner_angle_rad - measured)
        yaw = self.vision.confident_ekf_yaw(now, self.phase_started)
        if self.policy.corner_ekf_weight > 0 and yaw is not None:
            self.navigation_yaw = yaw
            self.heading_time = now
            self.corner_source = f'EKF with confident post-turn landmarks {sorted(self.vision.known_ids)}'
            return self.policy.corner_ekf_weight * self._heading_error()
        reason = self.vision.ekf_rejection_reason if self.policy.corner_ekf_weight else 'disabled'
        self.corner_source = 'Timed fallback; no stable shared tag; EKF ' + reason
        return None

    def _corner_pause(self, now):
        """Wait a fixed camera interval, then correct briefly or continue without tags.

        Args:
            now: Current ROS controller time in seconds.

        Returns:
            Stopped NavigationOutput. Corrections share a small total nominal angle
            budget and a maximum count. Subminimum angles are skipped, including a
            remaining budget too small to move the robot. Tags never block a leg.
        """
        if now - self.phase_started < self.turn_settle_s:
            return self._output('Corner pause; allowing optional camera observations to arrive')
        error = self._corner_error(now)
        self.corner_residual = error
        remaining = self.policy.corner_correction_limit_rad - self.correction_used
        minimum = self.policy.corner_min_correction_rad
        angle = min(abs(error), self.turn_pulse_rad, remaining) if error is not None else 0.0
        if angle + 1e-9 < minimum or self.correction_count >= self.policy.corner_max_corrections:
            return self._finish_corner(now)
        self.correction_goal = self.correction_used + angle
        self.correction_omega = self.turn_calibration.rate(error)
        self.correction_count += 1
        self._transition(NavigationState.CORRECTING, now)
        return self._output('Optional bounded corner correction selected')

    def _correct_corner(self, now):
        """Apply one fixed-rate correction, then stop for another optional observation."""
        if self.correction_used >= self.correction_goal or now - self.phase_started >= self.policy.corner_timeout_s:
            self._transition(NavigationState.CORNER_PAUSE, now)
            return self._output('Corner correction finished; stopped for camera processing')
        return self._output('Bounded corner correction from ' + self.corner_source,
                            angular=self.correction_omega)

    def _finish_corner(self, now):
        """Finish the lap or advance to the next side; visual feedback is always optional."""
        if self.side_index == self.side_count:
            self.state, self.is_active = NavigationState.COMPLETED, False
            return self._output(f'{self.side_count} sides complete; final corner used ' + self.corner_source)
        self.current_wp_idx = self.side_index + 1
        self._begin_leg(now)
        return self._output('Next side ready; corner used ' + self.corner_source)
