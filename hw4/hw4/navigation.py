"""Waypoint following with measured continuous turns and bounded straight travel."""

import math
from hw3.coordinates import wrap_angle
from hw4.localization import Feedback, stable_feedback


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

    @property
    def finished(self):
        """Report terminal states independently of whether the robot arrived."""
        return self.state in ('complete', 'failed', 'stopped')

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
        pose = stable_feedback(samples, self.after, now, self.policy)
        if pose is not None:
            return self.decide(now, pose)
        if now > self.deadline:
            return self.finish('failed', 'fresh consistent post-stop localization unavailable')
        return 0.0, 0.0

    def motion_tick(self, now, samples):
        """Stop on stale vision or pulse completion; never advance a waypoint here."""
        # require fresh vision throughout bounded motion
        if not samples or not 0 <= now - samples[-1].stamp <= self.policy['pose_timeout_s']:
            return self.finish('failed', 'localization became stale during motion')
        if now < self.deadline:
            return self.command

        # predict only the executed command interval
        duration = now - self.pulse_start
        x, y, yaw = self.pulse_pose.x, self.pulse_pose.y, self.pulse_pose.yaw
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
        self.reason = 'stopped; acquiring post-settle measurements'
        return self.command

    def accept_pose(self, pose):
        """Reject unsafe poses and large disagreements with command prediction."""
        # check physical clearance and prediction consistency
        if not self.grid.segment_free(pose.xy, pose.xy):
            return 'measured pose violates configured robot clearance'
        reference = self.expected or (*self.path[0], self.initial_yaw)
        if math.dist(pose.xy, reference[:2]) > self.policy['max_position_jump_m']:
            return 'measured position disagrees with previous prediction or start placement'
        if abs(wrap_angle(pose.yaw - reference[2])) > self.policy['max_heading_jump_rad']:
            return 'measured heading disagrees with previous command prediction'
        return None

    def decide(self, now, pose):
        """Use stopped measured feedback for arrival, heading correction, or travel."""
        # advance only using stopped measured feedback
        error = self.accept_pose(pose)
        if error:
            return self.finish('failed', error)
        self.measured = pose
        while self.index < len(self.path) and math.dist(pose.xy, self.path[self.index]) <= self.policy['position_tolerance_m']:
            self.index += 1
            self.corrections = 0
            self.commanded_distance, self.segment_budget = 0.0, None
        if self.index == len(self.path):
            return self.finish('complete', 'goal reached using fresh stable landmark pose')

        # select the next fixed waypoint
        target = self.path[self.index]
        heading = math.atan2(target[1] - pose.y, target[0] - pose.x)
        residual = wrap_angle(heading - pose.yaw)
        if abs(residual) > self.policy['heading_tolerance_rad']:
            return self.begin_turn(now, pose, residual)
        self.corrections = 0
        return self.drive_pulse(now, pose, target)

    def begin_turn(self, now, pose, residual):
        """Rotate continuously toward a measured heading at most 30 degrees away."""
        # limit whole turns without timed pulses
        if self.corrections >= self.policy['max_turns']:
            return self.finish('failed', 'heading correction budget exhausted')
        self.corrections += 1
        self.turn_angle = min(abs(residual), self.policy['turn_step_rad'])
        self.turn_direction = 1.0 if residual > 0 else -1.0
        self.turn_origin, self.turn_start = pose, now
        self.turn_progress = 0.0
        self.turn_target_yaw = wrap_angle(pose.yaw + self.turn_direction * self.turn_angle)

        # refresh motors until measured heading arrives
        self.state, self.reason = 'turning', 'continuous measured turn toward waypoint'
        self.deadline = now + self.policy['turn_timeout_s']
        self.command = 0.0, self.turn.rate(residual)
        return self.command

    def turn_tick(self, now, samples):
        """Stop on measured turn completion, feedback loss, or an exhausted timeout."""
        # reject stale feedback while turning continuously
        if not samples or not 0 <= now - samples[-1].stamp <= self.policy['turn_feedback_timeout_s']:
            return self.finish('failed', 'fresh heading feedback unavailable during continuous turn')
        pose = samples[-1]
        if pose.stamp > self.turn_start:
            error = self.check_turn_pose(pose)
            if error:
                return self.finish('failed', error)
            self.turn_progress = self.turn_direction * wrap_angle(pose.yaw - self.turn_origin.yaw)
            if self.turn_progress >= self.turn_angle - self.policy['turn_stop_tolerance_rad']:
                self.expected = pose.x, pose.y, pose.yaw
                return self.wait_for_feedback(now)

        if now >= self.deadline:
            return self.finish('failed', 'continuous turn timed out without measured heading arrival')
        return self.command

    def check_turn_pose(self, pose):
        """Reject unsafe displacement, heading flips, or rotation in the wrong direction."""
        # preserve footprint clearance throughout measured rotation
        if not self.grid.segment_free(pose.xy, pose.xy):
            return 'measured turn position violates robot clearance'
        if math.dist(pose.xy, self.turn_origin.xy) > self.policy['max_position_jump_m']:
            return 'excessive measured translation during in-place turn'
        progress = self.turn_direction * wrap_angle(pose.yaw - self.turn_origin.yaw)
        if progress < -self.policy['heading_tolerance_rad']:
            return 'measured rotation is opposite to requested turn'
        if progress > self.turn_angle + self.policy['max_heading_jump_rad']:
            return 'measured heading jumped beyond continuous turn target'
        return None

    def drive_pulse(self, now, pose, target):
        """Check the actual forward sweep, including one watchdog interval of overrun."""
        # validate measured route before committing forward
        if not self.grid.segment_free(pose.xy, target):
            return self.finish('failed', 'measured pose no longer has a clear route to waypoint')
        distance = min(math.dist(pose.xy, target), self.policy['drive_chunk_m'])
        if self.segment_budget is None:
            self.segment_budget = 1.5 * math.dist(pose.xy, target) + self.policy['drive_chunk_m']
        if self.commanded_distance + distance > self.segment_budget:
            return self.finish('failed', 'forward command budget exhausted without measured waypoint arrival')
        self.commanded_distance += distance

        # reserve clearance for delayed motor stopping
        guarded = distance + self.forward.speed_mps * 0.15
        end = pose.x + guarded * math.cos(pose.yaw), pose.y + guarded * math.sin(pose.yaw)
        if not self.grid.segment_free(pose.xy, end):
            return self.finish('failed', 'predicted forward sweep violates robot clearance')

        return self.begin_pulse(now, pose, self.forward.speed_mps, 0.0,
                                distance / self.forward.speed_mps, 'driving short straight segment')

    def begin_pulse(self, now, pose, linear, angular, duration, reason):
        """Freeze the calibrated command and monotonic deadline for one pulse."""
        # freeze pulse timing before publishing movement
        self.state, self.reason = 'moving', reason
        self.pulse_start, self.pulse_pose = now, pose
        self.deadline = now + duration
        self.command = linear, angular
        return self.command
