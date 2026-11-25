"""Waypoint following from a known start with optional stopped visual corrections."""

import math
from hw3.coordinates import wrap_angle
from hw4.localization import Feedback, stable_feedback
from hw4.checkpoint import checkpoint_pose, goal_confirmed


class Navigator:
    """Follow one path; command integration never establishes measured arrival."""

    def __init__(self, grid, path, forward, turn, policy, now, initial_yaw=0.0):
        """Initialize a stopped run with world-meter waypoints and shared calibration."""
        # initialize stopped tracking and finite budgets
        self.grid, self.path, self.policy = grid, path, policy
        self.forward, self.turn = forward, turn
        self.index = 1 if len(path) > 1 else 0
        self.state, self.reason = 'waiting', 'waiting for initial measured pose'
        self.started, self.after, self.last_tick = now, now, now
        self.command, self.expected, self.measured = (0.0, 0.0), None, None
        self.deadline = now + policy['feedback_wait_s']
        self.corrections = 0
        self.turn_progress, self.turn_target_yaw = 0.0, None
        self.initial_yaw = initial_yaw
        self.commanded_distance, self.segment_budget = 0.0, None
        self.feedback_required = policy['feedback_required']
        self.motion_clearance = grid.margin
        self.checkpoint_attempts = 0
        self.pose_source, self.feedback_note = 'waiting for vision', ''
        if not self.feedback_required:
            self.expected = (*path[0], initial_yaw)
            self.pose_source = 'configured start'
            self.reason = 'ready to move from configured start pose'

    @property
    def finished(self):
        """Report terminal states independently of whether the robot arrived."""
        return self.state in ('complete', 'unverified', 'failed', 'stopped')

    def finish(self, state, reason):
        """Latch a terminal state and immediately request a motor stop."""
        if self.finished:
            return 0.0, 0.0
        self.state, self.reason, self.command = state, reason, (0.0, 0.0)
        return self.command

    def tick(self, now, samples):
        """Return forward m/s and signed rad/s for the next 20 Hz control tick."""
        # enforce mission duration and watchdog timing
        if self.finished:
            return 0.0, 0.0
        if now - self.started > self.policy['mission_timeout_s']:
            return self.finish('failed', 'mission time budget exhausted')
        if self.state in ('moving', 'turning') and now - self.last_tick > 0.15:
            return self.finish('failed', 'control tick exceeded motor watchdog interval')

        # dispatch bounded motion or stopped acquisition
        self.last_tick = now
        if self.state == 'moving':
            return self.motion_tick(now, samples)
        if self.state == 'turning':
            return self.turn_tick(now, samples)
        if self.state == 'verifying':
            return self.verify_tick(now, samples)
        if now < self.after:
            return 0.0, 0.0
        pose = stable_feedback(samples, self.after, now, self.policy)
        if pose is not None:
            error = self.accept_pose(pose)
            if not error or self.feedback_required:
                self.pose_source, self.feedback_note = 'measured landmarks', ''
                return self.decide(now, pose)
            self.feedback_note = f'visual correction rejected: {error}'
        if not self.feedback_required:
            self.pose_source = 'command prediction' if self.measured or self.expected[:2] != self.path[0] else self.pose_source
            if pose is None:
                self.feedback_note = 'no stable visual correction; continuing predicted route'
            return self.decide(now, Feedback(*self.expected, now), measured=False)
        if now > self.deadline:
            return self.finish('failed', 'fresh consistent post-stop localization unavailable')
        return 0.0, 0.0

    def motion_tick(self, now, samples):
        """Stop on stale vision or segment completion; never advance a waypoint here."""
        # require fresh vision throughout bounded motion
        if self.feedback_required and (not samples or not 0 <= now - samples[-1].stamp <= self.policy['pose_timeout_s']):
            return self.finish('failed', 'localization became stale during motion')
        if now < self.deadline:
            return self.command

        # predict only the executed command interval
        duration = now - self.segment_start
        x, y, yaw = self.segment_pose.x, self.segment_pose.y, self.segment_pose.yaw
        v, omega = self.command
        self.expected = (x + v * duration * math.cos(yaw), y + v * duration * math.sin(yaw),
                         wrap_angle(yaw + omega * duration))
        return self.wait_for_feedback(now)

    def wait_for_feedback(self, now):
        """Stop and require fresh stable post-settle images before the next action."""
        # separate measured arrival from motor completion
        self.state, self.command = 'waiting', (0.0, 0.0)
        self.after = now + self.policy['settle_s']
        self.deadline = self.after + self.policy['feedback_wait_s']
        self.pose_source = 'command prediction'
        self.reason = ('stopped; acquiring post-settle measurements' if self.feedback_required
                       else 'brief checkpoint; visual correction optional')
        return self.command

    def accept_pose(self, pose):
        """Reject unsafe poses and large disagreements with command prediction."""
        # check physical clearance and prediction consistency
        if not self.segment_is_clear(pose.xy, pose.xy):
            return 'measured pose violates configured robot clearance'
        reference = self.expected or (*self.path[0], self.initial_yaw)
        if math.dist(pose.xy, reference[:2]) > self.policy['max_position_jump_m']:
            return 'measured position disagrees with previous prediction or start placement'
        if abs(wrap_angle(pose.yaw - reference[2])) > self.policy['max_heading_jump_rad']:
            return 'measured heading disagrees with previous command prediction'
        return None

    def decide(self, now, pose, measured=True):
        """Choose the next action without confusing predicted and measured arrival."""
        # retain the source of waypoint progress
        error = self.accept_pose(pose)
        if error:
            return self.finish('failed', error)
        if measured:
            self.measured = pose
        tolerance = self.target_tolerance()
        if math.dist(pose.xy, self.path[self.index]) <= tolerance:
            return self.begin_verification(now)

        # select the next fixed waypoint
        target = self.path[self.index]
        heading = math.atan2(target[1] - pose.y, target[0] - pose.x)
        residual = wrap_angle(heading - pose.yaw)
        if abs(residual) > self.policy['heading_tolerance_rad']:
            return self.begin_turn(now, pose, residual)
        self.corrections = 0
        return self.drive_segment(now, pose, target)

    def target_tolerance(self):
        """Use tighter accuracy at the active goal than at intermediate waypoints."""
        return (self.policy['goal_tolerance_m'] if self.index >= len(self.path) - 1
                else self.policy['position_tolerance_m'])

    def begin_verification(self, now):
        """Hold the current target until fresh stopped tag observations verify it."""
        # prediction cannot advance or complete waypoints
        self.wait_for_feedback(now)
        self.state = 'verifying'
        self.reason = f'verifying waypoint {self.index} with fresh AprilTag observations'
        return self.command

    def verify_tick(self, now, samples):
        """Correct from measured pose; stop explicitly if verification is unavailable."""
        # collect independent frames after the stop
        if now < self.after:
            return 0.0, 0.0
        final = self.index == len(self.path) - 1
        pose = checkpoint_pose(samples, self.after, now, self.policy, final)
        if pose is None:
            if now >= self.deadline:
                return self.finish('failed', 'waypoint verification timed out; target not advanced')
            return 0.0, 0.0

        # accept bounded measured correction over dead reckoning
        error = self.apply_checkpoint_pose(pose)
        if error:
            return self.finish('failed', error)
        distance = math.dist(pose.xy, self.path[self.index])
        if distance > self.target_tolerance():
            self.checkpoint_attempts += 1
            if self.checkpoint_attempts > self.policy['max_checkpoint_corrections']:
                return self.finish('failed', 'waypoint correction budget exhausted without measured arrival')
            self.state = 'waiting'
            return self.decide(now, pose)

        # verify every goal frame before finishing
        if final and not goal_confirmed(samples, self.after, now, self.path[self.index], self.policy):
            if now >= self.deadline:
                return self.finish('failed', 'goal observations did not all confirm arrival tolerance')
            self.feedback_note = 'waiting for all goal samples to confirm tolerance'
            return 0.0, 0.0
        self.index += 1
        self.commanded_distance, self.segment_budget, self.corrections = 0.0, None, 0
        self.checkpoint_attempts = 0
        if self.index == len(self.path):
            return self.finish('complete', 'goal verified by fresh stable AprilTag observations')
        self.state = 'waiting'
        return self.decide(now, pose)

    def apply_checkpoint_pose(self, pose):
        """Replace prediction with a stable measured pose within bounded correction limits."""
        # permit correction without trusting arbitrary pose jumps
        clearance = self.grid.clearance(pose.xy)
        if clearance < self.policy['recovery_min_clearance_m']:
            return 'checkpoint pose too close to collision for automatic correction'
        if self.expected and math.dist(pose.xy, self.expected[:2]) > self.policy['checkpoint_max_correction_m']:
            return 'checkpoint localization correction exceeds position limit'
        if self.expected and abs(wrap_angle(pose.yaw - self.expected[2])) > self.policy['checkpoint_max_heading_rad']:
            return 'checkpoint localization correction exceeds heading limit'
        self.expected, self.measured = (pose.x, pose.y, pose.yaw), pose
        self.motion_clearance = min(self.grid.margin, clearance)
        self.pose_source = 'measured AprilTag checkpoint'
        self.feedback_note = f'checkpoint correction accepted; target distance {math.dist(pose.xy, self.path[self.index]):.3f} m'
        return None

    def segment_is_clear(self, start, end):
        """Allow only outward recovery when a measured pose is inside the safety margin."""
        # never move closer while recovering clearance
        gap = self.grid.clearance(start, end)
        if gap < self.motion_clearance - 1e-8:
            return False
        if start != end and self.motion_clearance < self.grid.margin:
            return self.grid.clearance(end) > self.grid.clearance(start) + 1e-4
        return True

    def begin_turn(self, now, pose, residual):
        """Rotate continuously until facing the next planned waypoint."""
        # target the full planned travel heading
        if self.corrections >= self.policy['max_turns']:
            return self.finish('failed', 'heading correction budget exhausted')
        self.corrections += 1
        self.turn_angle = abs(residual)
        self.turn_direction = 1.0 if residual > 0 else -1.0
        self.turn_origin, self.turn_start = pose, now
        self.last_turn_pose = pose
        self.turn_progress = 0.0
        self.turn_target_yaw = wrap_angle(pose.yaw + self.turn_direction * self.turn_angle)

        # refresh motors until measured heading arrives
        self.state, self.reason = 'turning', 'turning continuously to planned waypoint heading'
        self.command = 0.0, self.turn.rate(residual)
        duration = self.policy['turn_timeout_s'] if self.feedback_required else self.turn_angle / abs(self.command[1])
        self.deadline = now + duration
        if not self.feedback_required:
            self.reason = 'executing full calibrated turn from estimated heading'
        return self.command

    def turn_tick(self, now, samples):
        """Stop on measured turn completion, feedback loss, or an exhausted timeout."""
        if not self.feedback_required:
            return self.calibrated_turn_tick(now)

        # reject stale feedback while turning continuously
        if not samples or not 0 <= now - samples[-1].stamp <= self.policy['turn_feedback_timeout_s']:
            return self.finish('failed', 'fresh heading feedback unavailable during continuous turn')
        pose = samples[-1]
        if pose.stamp > self.turn_start:
            error = self.check_turn_pose(pose)
            if error:
                return self.finish('failed', error)
            self.turn_progress = self.turn_direction * wrap_angle(pose.yaw - self.turn_origin.yaw)
            residual = wrap_angle(self.turn_target_yaw - pose.yaw)
            if abs(residual) <= self.policy['heading_tolerance_rad']:
                self.measured = pose
                self.expected = pose.x, pose.y, pose.yaw
                self.corrections = 0
                return self.drive_segment(now, pose, self.path[self.index])
            self.command = 0.0, self.turn.rate(residual)

        if now >= self.deadline:
            return self.finish('failed', 'continuous turn timed out without measured heading arrival')
        return self.command

    def calibrated_turn_tick(self, now):
        """Execute the full timed angle; its resulting heading is a prediction."""
        # finish the turn without waiting for vision
        if now < self.deadline:
            return self.command
        yaw = wrap_angle(self.turn_origin.yaw + self.command[1] * (now - self.turn_start))
        pose = Feedback(self.turn_origin.x, self.turn_origin.y, yaw, now)
        self.expected = pose.x, pose.y, pose.yaw
        self.pose_source = 'command prediction'
        self.feedback_note = 'turn completion is predicted from calibration'
        return self.drive_segment(now, pose, self.path[self.index])

    def check_turn_pose(self, pose):
        """Reject unsafe displacement and implausible frame-to-frame heading jumps."""
        # preserve footprint clearance throughout measured rotation
        if not self.segment_is_clear(pose.xy, pose.xy):
            return 'measured turn position violates robot clearance'
        if math.dist(pose.xy, self.turn_origin.xy) > self.policy['max_position_jump_m']:
            return 'excessive measured translation during in-place turn'
        change = abs(wrap_angle(pose.yaw - self.last_turn_pose.yaw))
        elapsed = max(0.0, pose.stamp - self.last_turn_pose.stamp)
        limit = self.policy['max_heading_jump_rad'] + abs(self.command[1]) * elapsed
        if change > limit:
            return 'measured heading jumped during continuous turn'
        self.last_turn_pose = pose
        return None

    def drive_segment(self, now, pose, target):
        """Drive for up to one second, shortening near a waypoint or clearance limit."""
        # verify arrival before issuing tiny corrections
        if math.dist(pose.xy, target) <= self.target_tolerance():
            return self.begin_verification(now)
        if not self.segment_is_clear(pose.xy, target):
            return self.finish('failed', 'measured pose no longer has a clear route to waypoint')
        maximum = self.forward.speed_mps * self.policy['drive_duration_s']
        requested = min(math.dist(pose.xy, target), maximum)
        distance = self.clear_drive_distance(pose, requested)
        if distance < min(requested, 0.05) - 1e-6:
            return self.finish('failed', 'predicted forward sweep violates robot clearance')

        # bound commands without claiming measured arrival
        if self.segment_budget is None:
            self.segment_budget = 1.5 * math.dist(pose.xy, target) + maximum
        if self.commanded_distance + distance > self.segment_budget:
            return self.finish('failed', 'forward command budget exhausted without measured waypoint arrival')
        self.commanded_distance += distance
        return self.begin_segment(now, pose, self.forward.speed_mps, 0.0,
                                  distance / self.forward.speed_mps, 'driving toward waypoint for up to one second')

    def forward_sweep_free(self, pose, distance):
        """Include one motor-watchdog interval of possible stopping overrun."""
        guarded = distance + self.forward.speed_mps * 0.15
        end = pose.x + guarded * math.cos(pose.yaw), pose.y + guarded * math.sin(pose.yaw)
        return self.segment_is_clear(pose.xy, end)

    def clear_drive_distance(self, pose, requested):
        """Shorten a segment only when its full projected sweep is blocked."""
        # retain long travel wherever clearance allows
        if self.forward_sweep_free(pose, requested):
            return requested
        lower, upper = 0.0, requested
        for _ in range(12):
            middle = (lower + upper) / 2
            if self.forward_sweep_free(pose, middle):
                lower = middle
            else:
                upper = middle

        return lower * 0.95

    def begin_segment(self, now, pose, linear, angular, duration, reason):
        """Freeze the calibrated command and monotonic deadline for straight travel."""
        # refresh motors throughout the requested duration
        self.state, self.reason = 'moving', reason
        self.segment_start, self.segment_pose = now, pose
        self.deadline = now + duration
        self.command = linear, angular
        return self.command
