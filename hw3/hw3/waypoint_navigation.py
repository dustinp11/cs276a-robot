"""Fixed polygon waypoints with stopped, bounded SLAM-assisted motion updates."""

from collections import deque
import math

from .coordinates import wrap_angle


def polygon_vertices(origin, side_length_m, side_count):
    """Return fixed SLAM-frame endpoints, turning left before each polygon side.

    Args:
        origin: Initial (x_m, y_m, yaw_rad) in the SLAM frame.
        side_length_m: Positive side length in meters.
        side_count: Number of polygon sides, at least three.

    Returns:
        Tuple of xy endpoints in meters; the last endpoint is exactly the origin.
    """
    x, y, yaw = origin
    vertices = []
    for index in range(1, side_count + 1):
        heading = yaw + index * 2 * math.pi / side_count
        x, y = x + side_length_m * math.cos(heading), y + side_length_m * math.sin(heading)
        vertices.append((x, y))
    vertices[-1] = origin[:2]
    return tuple(vertices)


class WaypointNavigation:
    """Reuse calibrated motor commands while checking SLAM only when stopped."""

    def __init__(self, navigator, states):
        """Keep mission state separate from the baseline timed controller."""
        self.nav, self.states = navigator, states
        self.pose, self.vertices, self.last_tick = None, (), None
        self.samples = deque(maxlen=3)
        self.image_stamp = -math.inf
        self.phase, self.source = 'initial', 'command prediction'
        self.turn_end, self.turn_rate = 0.0, 0.0
        self.drive_goal, self.leg_distance = 0.0, 0.0
        self.attempts, self.correction_used = 0, 0.0

    @property
    def policy(self):
        """Return the shared validated navigation settings."""
        return self.nav.policy

    @property
    def lap_index(self):
        """Return the one-based lap number, retaining the final lap after completion."""
        return min(self.nav.side_index // self.nav.side_count + 1, self.policy.waypoint_laps)

    def _emit(self, message, linear=0.0, angular=0.0):
        """Publish the accepted/predicted pose and feedback reason through existing telemetry."""
        self.nav.navigation_x, self.nav.navigation_y, self.nav.navigation_yaw = self.pose
        self.nav.corner_source = self.source
        return self.nav._output(message, linear=linear, angular=angular)

    def _predict(self, now):
        """Integrate the previous motor command, bounded by the existing 150 ms watchdog."""
        dt = 0.0 if self.last_tick is None else max(0.0, min(now - self.last_tick, 0.15))
        v, omega = self.nav.last_command
        x, y, yaw = self.pose
        self.pose = (x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw),
                     wrap_angle(yaw + omega * dt))
        self.last_tick = now

    def _initialize(self, now):
        """Freeze the initial SLAM frame and all vertices without waiting for tag visibility.

        Args:
            now: Current controller time in seconds.

        Returns:
            First stopped acquisition command; no estimator reset is performed.
        """
        nav = self.nav
        self.pose = nav._best_pose()
        nav.x0_slam, nav.y0_slam, nav.theta0_slam = self.pose
        nav.origin_latched = True
        self.vertices = polygon_vertices(self.pose, nav.square_size_m, nav.side_count)
        self.last_tick = now
        self._select_target()
        return self._pause(now, 'initial')

    def _select_target(self):
        """Select a precomputed vertex, never a new point relative to the current pose."""
        vertex_index = self.nav.side_index % self.nav.side_count
        self.nav.target = self.vertices[vertex_index]
        self.nav.current_wp_idx = self.nav.side_index + 1
        self.nominal_heading = self.nav.theta0_slam + (vertex_index + 1) * self.nav.corner_angle_rad
        self.nav.course_heading = self.nominal_heading

    def _pause(self, now, phase):
        """Stop for a fixed acquisition interval and clear per-checkpoint correction budgets."""
        self.phase = phase
        self.samples.clear()
        self.attempts, self.correction_used = 0, 0.0
        self.nav._transition(self.states.MIDPOINT_PAUSE if phase == 'midpoint' else self.states.STOPPING, now)
        return self._emit('Waypoint pause: ' + phase)

    def _sample_feedback(self, now):
        """Collect distinct post-stop image-associated poses, without requiring multiple tags.

        Args:
            now: Current controller time in seconds.

        Returns:
            None. Unavailable or uncertain observations leave calibrated prediction intact.
        """
        nav, vision = self.nav, self.nav.vision
        fresh = (nav.phase_started <= vision.stamp <= now
                 and now - vision.stamp <= self.policy.vision_timeout_s
                 and -0.05 <= now - nav.latest_pose_stamp_s <= nav.pose_timeout_s)
        if not fresh or not vision.known_ids or not nav._uncertainty_ok():
            self.source = 'prediction: no fresh confident mapped-tag pose'
            return
        self.source = 'prediction: collecting stable stopped SLAM poses'
        if vision.stamp > self.image_stamp and nav.latest_pose_stamp_s >= vision.stamp - 0.05:
            self.image_stamp = vision.stamp
            self.samples.append(nav.latest_slam_pose)

    def _accept_feedback(self, now):
        """Accept stable stopped poses only within bounded innovation from motion prediction.

        Args:
            now: Current controller time in seconds.

        Returns:
            Boolean indicating whether a fresh three-image SLAM pose was accepted.
        """
        if len(self.samples) < 3 or now - self.image_stamp > self.policy.vision_timeout_s:
            return False
        candidate = self.samples[-1]
        stable = all(math.dist(p[:2], candidate[:2]) <= 0.05
                     and abs(wrap_angle(p[2] - candidate[2])) <= math.radians(5)
                     for p in self.samples)
        bounded = (math.dist(self.pose[:2], candidate[:2]) <= self.policy.waypoint_max_position_jump_m
                   and abs(wrap_angle(candidate[2] - self.pose[2])) <= self.policy.waypoint_max_heading_jump_rad)
        if stable and bounded:
            self.pose, self.source = candidate, 'accepted stopped SLAM pose'
            return True
        self.source = 'prediction: unstable or excessive SLAM innovation'
        return False

    def _bearing(self):
        """Return heading to the fixed vertex, using the side heading inside arrival tolerance."""
        dx, dy = self.nav.target[0] - self.pose[0], self.nav.target[1] - self.pose[1]
        return self.nominal_heading if math.hypot(dx, dy) <= self.policy.waypoint_position_tolerance_m else math.atan2(dy, dx)

    def _arrived(self):
        """Avoid driving away from a reached vertex or chasing backward after passing its plane."""
        dx, dy = self.nav.target[0] - self.pose[0], self.nav.target[1] - self.pose[1]
        along = dx * math.cos(self.nominal_heading) + dy * math.sin(self.nominal_heading)
        return (math.hypot(dx, dy) <= self.policy.waypoint_position_tolerance_m
                or (self.phase == 'midpoint' and along <= 0))

    def _turn(self, now, angle, correction=False):
        """Start a calibrated pivot; main turns and optional corrections have separate bounds.

        Args:
            now: Current controller time in seconds.
            angle: Signed requested heading change in radians.
            correction: Whether this pulse spends the checkpoint correction budget.

        Returns:
            Pivot command with zero forward velocity.
        """
        limit = min(self.nav.turn_pulse_rad, self.policy.waypoint_correction_budget_rad - self.correction_used) if correction else math.pi / 2
        angle = max(-limit, min(limit, angle))
        self.turn_rate = self.nav.turn_calibration.rate(angle)
        self.turn_end = now + abs(angle / self.turn_rate) if angle else now
        if correction:
            self.attempts += 1
            self.correction_used += abs(angle)
        self.nav._transition(self.states.CORRECTING if correction else self.states.TURNING, now)
        return self._emit('Waypoint heading pivot', angular=self.turn_rate)

    def _start_side(self, now):
        """Aim toward the next fixed point, with nominal distance as the no-feedback baseline."""
        self.nav.drive_used = 0.0
        self.leg_distance = self.nav.square_size_m
        self.nav.course_heading = self._bearing()
        return self._turn(now, wrap_angle(self.nav.course_heading - self.pose[2]))

    def _drive(self, now):
        """Move at fixed input until the checkpoint or hard per-side command budget is reached."""
        self.nav.progress = self.nav.drive_used
        if self.nav.drive_used + 1e-9 >= self.drive_goal:
            return self._pause(now, 'midpoint' if self.phase == 'first_half' else 'end')
        self.nav._transition(self.states.DRIVING, now)
        return self._emit('Waypoint forward segment', linear=self.nav.forward.speed_mps)

    def _resume_drive(self, now, feedback):
        """Bound distance updates and resume either the first half or remaining side.

        Args:
            now: Current controller time in seconds.
            feedback: Whether this pause accepted a stable SLAM pose.

        Returns:
            Forward or stopped checkpoint command; each side has a finite motor-on budget.
        """
        nav, delta = self.nav, self.policy.waypoint_max_distance_adjustment_m
        remaining = max(0.0, self.leg_distance - nav.drive_used)
        if feedback:
            measured = math.dist(self.pose[:2], nav.target)
            remaining = max(0.0, min(remaining + delta, max(remaining - delta, measured)))
            if self._arrived():
                remaining = 0.0
        self.drive_goal = min(nav.square_size_m + delta, nav.drive_used + remaining)
        if self.phase == 'after_turn':
            self.leg_distance = self.drive_goal
            self.drive_goal = min(self.drive_goal, nav.square_size_m / 2)
            self.phase = 'first_half'
        else:
            self.phase = 'second_half'
        return self._drive(now)

    def _finish_pause(self, now):
        """Advance checkpoints without waiting indefinitely for observations.

        Args:
            now: Current controller time in seconds.

        Returns:
            Next pivot, forward command, or terminal completion output.
        """
        feedback = self._accept_feedback(now)
        if self.phase in ('initial', 'end'):
            return self._next_side(now)
        if feedback and self._arrived():
            return self._resume_drive(now, feedback)
        error = wrap_angle(self._bearing() - self.pose[2])
        self.nav.course_heading = self._bearing()
        budget = self.policy.waypoint_correction_budget_rad - self.correction_used
        if (feedback and abs(error) >= self.policy.corner_min_correction_rad
                and self.attempts < self.policy.corner_max_corrections
                and budget >= self.policy.corner_min_correction_rad):
            return self._turn(now, error, correction=True)
        return self._resume_drive(now, feedback)

    def _next_side(self, now):
        """Complete exactly the configured number of fixed targets, including return to start."""
        if self.phase == 'end':
            self.nav.side_index += 1
        if self.nav.side_index == self.nav.side_count * self.policy.waypoint_laps:
            self.nav.state, self.nav.is_active = self.states.COMPLETED, False
            return self._emit(f'Waypoint mission complete: {self.policy.waypoint_laps} laps; final target is the starting point')
        self._select_target()
        return self._start_side(now)

    def step(self, now):
        """Advance fixed-waypoint control using only stopped optional SLAM corrections.

        Args:
            now: Current ROS controller time in seconds.

        Returns:
            Existing NavigationOutput consumed by the unchanged ROS adapter and logger.
        """
        if self.pose is None:
            return self._initialize(now)
        self._predict(now)
        if self.nav.state in (self.states.TURNING, self.states.CORRECTING):
            return self._step_turn(now)
        if self.nav.state == self.states.DRIVING:
            return self._drive(now)
        self._sample_feedback(now)
        if now - self.nav.phase_started + 1e-9 < self.policy.waypoint_pause_s:
            return self._emit('Waypoint settling: ' + self.source)
        return self._finish_pause(now)

    def _step_turn(self, now):
        """Stop at the pulse deadline, then reacquire feedback before making another decision."""
        if now + 1e-9 < self.turn_end:
            return self._emit('Waypoint heading pivot', angular=self.turn_rate)
        if self.nav.state == self.states.TURNING:
            return self._pause(now, 'after_turn')
        self.samples.clear()
        self.nav._transition(self.states.MIDPOINT_PAUSE if self.phase == 'midpoint' else self.states.STOPPING, now)
        return self._emit('Waypoint correction finished; reacquiring stopped feedback')
