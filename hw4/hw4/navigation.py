"""Waypoint following from a known start with optional stopped visual corrections."""

import math
from hw3.coordinates import wrap_angle
from hw4.localization import Feedback, stable_feedback
from hw4.checkpoint import checkpoint_pose, goal_confirmed
from hw4.tag_search import TagSearch


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
        self.visual_feedback_mode = policy['visual_feedback']
        self.goal_feedback_active, self.using_goal_reference = False, False
        self.motion_clearance = grid.margin
        self.checkpoint_attempts = 0
        self.search, self.search_attempted = None, False
        self.unverified_waypoints = []
        self.pose_source, self.feedback_note = 'waiting for vision', ''
        if not self.feedback_required:
            self.expected = (*path[0], initial_yaw)
            self.pose_source = 'configured start'
            self.reason = 'ready to move from configured start pose'

    def visual_feedback_enabled(self):
        """Latch the goal approach inside its predicted radius, in world meters."""
        # keep approach feedback active after any correction
        if self.visual_feedback_mode != 'off' and self.expected is not None:
            final = self.index >= len(self.path) - 1
            distance = math.dist(self.expected[:2], self.path[-1])
            radius = (self.policy['goal_tolerance_m'] if self.visual_feedback_mode == 'at_goal'
                      else self.policy['visual_enable_distance_m'])
            if final and distance <= radius + 1e-6:
                self.goal_feedback_active = True
        return (self.visual_feedback_mode == 'throughout'
                or self.visual_feedback_mode in ('at_goal', 'near_goal') and self.goal_feedback_active)

    def continue_prediction(self, now):
        """Follow the frozen path while deliberately ignoring all observed poses."""
        self.state, self.pose_source = 'waiting', 'command prediction'
        self.feedback_note = 'visual feedback intentionally disabled during planned traversal'
        return self.decide(now, Feedback(*self.expected, now), measured=False)

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

    def tick(self, now, samples, goal_samples=None):
        """Return commands, using independent reference-tag samples during the goal approach."""
        self.using_goal_reference = self.goal_feedback_active and goal_samples is not None
        if self.using_goal_reference:
            samples = goal_samples

        # enforce mission duration and watchdog timing
        if self.finished:
            return 0.0, 0.0
        if now - self.started > self.policy['mission_timeout_s']:
            return self.finish('failed', 'mission time budget exhausted')
        if self.state in ('moving', 'turning', 'searching') and any(self.command) and now - self.last_tick > 0.15:
            return self.finish('failed', 'control tick exceeded motor watchdog interval')

        # dispatch bounded motion or stopped acquisition
        self.last_tick = now
        if self.state == 'moving':
            return self.motion_tick(now, samples)
        if self.state == 'turning':
            return self.turn_tick(now, samples)
        if self.state == 'verifying':
            return self.verify_tick(now, samples)
        if self.state == 'searching':
            return self.search_tick(now, samples)
        if now < self.after:
            return 0.0, 0.0
        if not self.visual_feedback_enabled():
            return self.continue_prediction(now)
        pose = stable_feedback(samples, self.after, now, self.policy)
        if pose is not None:
            error = self.accept_pose(pose)
            if not error or self.feedback_required:
                self.pose_source, self.feedback_note = 'measured landmarks', ''
                return self.decide(now, pose)
            self.feedback_note = f'visual correction rejected: {error}'
        if not self.feedback_required:
            if pose is None and self.commanded_distance > 0 and not self.search_attempted:
                return self.begin_search(now)
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
        if not self.visual_feedback_enabled():
            return self.continue_prediction(now)
        if self.goal_feedback_active and self.index == len(self.path) - 1:
            return self.begin_verification(now)
        return self.wait_for_feedback(now)

    def wait_for_feedback(self, now):
        """Stop to allow fresh post-settle images before the next action."""
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
        # advance blind traversal without a visual stop
        if not self.visual_feedback_enabled():
            return self.advance_waypoint(now, Feedback(*self.expected, now), verified=False)

        # prefer visual confirmation before predicted fallback
        self.wait_for_feedback(now)
        self.state = 'verifying'
        if not self.feedback_required:
            wait = (self.policy['goal_feedback_wait_s'] if self.goal_feedback_active
                    else self.policy['search_observation_wait_s'])
            self.deadline = self.after + wait
        self.reason = ('correcting goal approach with fresh AprilTag observations'
                       if self.goal_feedback_active else
                       f'verifying waypoint {self.index} with fresh AprilTag observations')
        return self.command

    def verify_tick(self, now, samples):
        """Prefer measured arrival; search or use explicit prediction fallback in fast mode."""
        # collect independent frames after the stop
        if now < self.after:
            return 0.0, 0.0
        final = self.index == len(self.path) - 1
        pose = checkpoint_pose(samples, self.after, now, self.policy, final)
        if pose is None:
            if now >= self.deadline:
                if self.using_goal_reference:
                    tag = f"{self.policy['goal_reference_family']}:{self.policy['goal_reference_id']}"
                    return self.finish('unverified', f'fresh stable goal reference {tag} unavailable; no search performed')
                if not self.feedback_required and not self.search_attempted:
                    return self.begin_search(now)
                if not self.feedback_required:
                    return self.resume_prediction(now)
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
                if not self.feedback_required:
                    return self.resume_prediction(now)
                return self.finish('failed', 'goal observations did not all confirm arrival tolerance')
            self.feedback_note = 'waiting for all goal samples to confirm tolerance'
            return 0.0, 0.0
        return self.advance_waypoint(now, pose, verified=True)

    def advance_waypoint(self, now, pose, verified):
        """Record how a target was reached and preserve unverified progress explicitly."""
        # distinguish estimated progress from measured arrival
        if not verified:
            self.unverified_waypoints.append(self.index)
        self.index += 1
        self.commanded_distance, self.segment_budget, self.corrections = 0.0, None, 0
        self.checkpoint_attempts, self.search_attempted = 0, False
        if self.index == len(self.path):
            if verified:
                return self.finish('complete', 'goal verified by fresh stable AprilTag observations')
            return self.finish('unverified', 'predicted route end reached; goal not visually confirmed')
        self.state = 'waiting'
        return self.decide(now, pose, measured=verified)

    def begin_search(self, now):
        """Search once per waypoint while retaining its target and predicted position."""
        if not self.policy['tag_search_enabled']:
            return self.resume_prediction(now)
        if not self.segment_is_clear(self.expected[:2], self.expected[:2]):
            return self.finish('failed', 'predicted search position violates robot clearance')

        # search without resetting waypoint or distance budgets
        self.search = TagSearch(now, self.expected, self.turn, self.policy)
        self.search_attempted, self.state = True, 'searching'
        self.turn_target_yaw, self.turn_progress = None, 0.0
        self.reason = 'turning to reacquire agreeing tags; current waypoint retained'
        self.command = self.search.command
        return self.command

    def search_tick(self, now, samples):
        """Use stopped reacquired feedback, or resume prediction after the bounded search."""
        # preserve estimated rotation even without visual observations
        self.command = self.search.tick(now, samples)
        self.expected = self.search.expected
        self.turn_progress = self.search.rotated
        self.pose_source = 'command prediction during tag search'
        self.reason = f'tag search: {self.search.phase}; current waypoint retained'
        if self.search.phase == 'found':
            error = self.apply_checkpoint_pose(self.search.pose)
            if error:
                self.feedback_note = f'search correction rejected: {error}'
                self.command = self.search.resume(now)
            else:
                return self.decide(now, self.search.pose)
        if self.search.phase == 'done':
            self.feedback_note = 'bounded tag search exhausted; resuming predicted route'
            return self.resume_prediction(now)
        return self.command

    def resume_prediction(self, now):
        """Continue the same target, advancing only if predicted position is within tolerance."""
        # never label fallback progress as measured success
        self.state, self.pose_source = 'waiting', 'command prediction'
        self.feedback_note = 'visual confirmation unavailable; using predicted waypoint progress'
        pose = Feedback(*self.expected, now)
        if math.dist(pose.xy, self.path[self.index]) <= self.target_tolerance():
            return self.advance_waypoint(now, pose, verified=False)
        return self.decide(now, pose, measured=False)

    def apply_checkpoint_pose(self, pose):
        """Replace prediction with a stable measured pose within bounded correction limits."""
        # permit correction without trusting arbitrary pose jumps
        clearance = self.grid.clearance(pose.xy)
        if clearance < self.policy['recovery_min_clearance_m']:
            return 'checkpoint pose too close to collision for automatic correction'
        limit = self.policy['checkpoint_max_correction_m']
        refining = (self.index == len(self.path) - 1 and self.measured is not None
                    and math.dist(self.measured.xy, self.path[-1]) <= self.policy['goal_refine_radius_m'])
        if refining:
            limit = min(limit, self.policy['goal_max_correction_m'])
        if self.expected and math.dist(pose.xy, self.expected[:2]) > limit:
            return 'checkpoint localization correction exceeds position limit'
        if self.expected and abs(wrap_angle(pose.yaw - self.expected[2])) > self.policy['checkpoint_max_heading_rad']:
            return 'checkpoint localization correction exceeds heading limit'
        self.expected, self.measured = (pose.x, pose.y, pose.yaw), pose
        self.motion_clearance = min(self.grid.margin, clearance)
        self.pose_source = ('measured goal reference '
                            f"{self.policy['goal_reference_family']}:{self.policy['goal_reference_id']}"
                            if self.using_goal_reference else 'measured AprilTag checkpoint')
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
        maximum = self.approach_step_limit(pose, target)
        requested = min(math.dist(pose.xy, target), maximum)
        distance = self.clear_drive_distance(pose, requested)
        if distance < min(requested, 0.05) - 1e-6:
            return self.finish('failed', 'predicted forward sweep violates robot clearance')

        # bound commands without claiming measured arrival
        if self.segment_budget is None:
            self.segment_budget = (1.5 * math.dist(pose.xy, target)
                                   + self.forward.speed_mps * self.policy['drive_duration_s'])
        if self.commanded_distance + distance > self.segment_budget:
            return self.finish('failed', 'forward command budget exhausted without measured waypoint arrival')
        self.commanded_distance += distance
        return self.begin_segment(now, pose, self.forward.speed_mps, 0.0,
                                  distance / self.forward.speed_mps, 'driving toward waypoint for up to one second')

    def approach_step_limit(self, pose, target):
        """Shorten final moves without changing fixed-power motor calibration."""
        # stop at the visual approach boundary
        maximum = self.forward.speed_mps * self.policy['drive_duration_s']
        if self.index < len(self.path) - 1:
            return maximum
        distance = math.dist(pose.xy, target)
        radius = self.policy['visual_enable_distance_m']
        if self.visual_feedback_mode in ('near_goal', 'throughout') and not self.goal_feedback_active and distance > radius + 1e-6:
            return min(maximum, distance - radius)

        # shrink correction distance as the goal approaches
        if self.goal_feedback_active or distance <= radius:
            return min(maximum, self.policy['goal_max_step_m'], distance * self.policy['goal_step_fraction'])
        return maximum

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
