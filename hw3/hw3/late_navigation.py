"""Single inward looks at late corners and segmented return to the recorded start."""

import math

from .coordinates import wrap_angle


class LateNavigation:
    """Reuse calibrated motion and optional vision for late polygon legs."""

    def __init__(self, navigator, states):
        """Keep mission state separate from ordinary side and corner control.

        Args:
            navigator: Owning calibrated polygon controller.
            states: Shared NavigationState enum, supplied to avoid circular imports.

        Returns:
            None. Initializes an inactive, stopped late-mission helper.
        """
        self.nav, self.states = navigator, states
        self.active = False
        self.return_travel_m = self.segment_used_m = 0.0
        self.return_result = 'return not attempted'
        self.pivot_used = self.pivot_goal = self.scan_used = 0.0
        self.pivot_rate = 0.0
        self.corner = self.extra_count = 0
        self.extra_used = 0.0
        self.scan_approach = 0.0

    def start(self, now, corner):
        """Begin one bounded inward look before choosing the outgoing heading.

        Args:
            now: Current ROS time in seconds.
            corner: Completed side count, starting at one.

        Returns:
            None. Sets a stopped acquisition or calibrated scan phase.
        """
        self.active, self.corner, self.scan_used = True, corner, 0.0
        self.extra_count, self.extra_used = 0, 0.0
        self.target = self.nav._planned_vertex(min(corner + 1, self.nav.side_count))
        self.nav.target = self.target
        self.action = 'complete' if corner == self.nav.side_count else 'drive'
        if corner == self.nav.side_count - 1:
            self.action, self.return_travel_m = 'return', 0.0
        self.scan_approach = self._look_direction()
        if abs(self.scan_approach) < self.nav.policy.corner_min_correction_rad:
            self._enter('RELOCALIZING_PAUSE', now)
        else:
            self._pivot(now, self.scan_approach, 'RELOCALIZING_TURN')

    def _enter(self, name, now):
        """Enter a shared navigation phase with a fresh deadline."""
        self.nav._transition(self.states[name], now)

    def _look_direction(self):
        """Choose a signed pivot toward the estimated center, capped at 90 degrees."""
        x, y = self.nav._planned_center()
        dx, dy = x - self.nav.navigation_x, y - self.nav.navigation_y
        if math.hypot(dx, dy) < 1e-6:
            return 0.0
        angle = wrap_angle(math.atan2(dy, dx) - self.nav.navigation_yaw)
        limit = self.nav.policy.late_scan_max_angle_rad
        return max(-limit, min(limit, angle))

    def account_motion(self, linear, angular, dt):
        """Count applied pivot and return travel once per controller interval.

        Args:
            linear: Last held forward velocity in meters per second.
            angular: Last held signed angular velocity in radians per second.
            dt: Applied interval in seconds, bounded by the existing motor watchdog.

        Returns:
            None. Updates search and return budgets independently of EKF pose corrections.
        """
        if not self.active:
            return
        if self.nav.state in (self.states.RELOCALIZING_TURN, self.states.REORIENTING):
            self.pivot_used += abs(angular) * dt
            if self.nav.state == self.states.RELOCALIZING_TURN:
                self.scan_used += abs(angular) * dt
        if self.nav.state == self.states.RETURN_DRIVING:
            self.return_travel_m += abs(linear) * dt
            self.segment_used_m += abs(linear) * dt

    def step(self, now):
        """Dispatch one interruptible scan, alignment, or return phase."""
        handlers = {'RELOCALIZING_TURN': self._scan_turn,
                    'RELOCALIZING_PAUSE': self._scan_pause,
                    'REORIENTING': self._align_turn,
                    'REORIENT_PAUSE': self._align_pause,
                    'RETURN_DRIVING': self._return_drive,
                    'RETURN_PAUSE': self._return_pause}
        return handlers[self.nav.state.name](now)

    def _pivot(self, now, angle, phase):
        """Set an intentional pivot; small optional residuals are handled separately."""
        self.pivot_goal, self.pivot_used = abs(angle), 0.0
        self.pivot_rate = self.nav.turn_calibration.rate(angle)
        self._enter(phase, now)

    def _pivot_finished(self, now):
        """Terminate a pivot by integrated angle or its existing elapsed-time deadline."""
        return (self.pivot_used + 1e-9 >= self.pivot_goal
                or now - self.nav.phase_started >= self.nav.policy.corner_timeout_s)

    def _scan_turn(self, now):
        """Finish the single inward look, then stop for camera observations."""
        if self._pivot_finished(now):
            self._enter('RELOCALIZING_PAUSE', now)
            return self.nav._output('Center look finished; stopped for camera')
        return self.nav._output(f'Late corner {self.corner}: looking toward estimated center',
                                angular=self.pivot_rate)

    def _refresh_pose(self, now):
        """Accept a current pose only after a confident, stable post-stop tag observation.

        Args:
            now: Current ROS time in seconds.

        Returns:
            True after applying a finite, fresh, sufficiently confident pose; False otherwise.
            Scan orientation alone never counts as a landmark observation.
        """
        nav, vision = self.nav, self.nav.vision
        yaw = vision.confident_ekf_yaw(now, nav.phase_started)
        stable = vision.stable_tag_yaws(now, nav.phase_started)
        age = now - nav.latest_pose_stamp_s
        if yaw is None or not vision.known_ids.intersection(stable):
            return False
        if (nav.latest_slam_pose is None or not 0 <= age <= nav.pose_timeout_s
                or nav.latest_pose_stamp_s < vision.stamp or not nav._uncertainty_ok()):
            return False
        nav.navigation_x, nav.navigation_y = nav.latest_slam_pose[:2]
        nav.navigation_yaw, nav.heading_time = yaw, now
        nav.corner_source = f'Late relocalization with tags {sorted(vision.known_ids.intersection(stable))}'
        return True

    def _scan_pause(self, now):
        """Pause once after looking inward, then leave regardless of tag visibility.

        Args:
            now: Current ROS time in seconds.

        Returns:
            Stopped NavigationOutput selecting the outgoing heading. Never repeats the
            look or escalates to a full revolution when feedback is missing.
        """
        if now - self.nav.phase_started < self.nav.turn_settle_s:
            return self.nav._output('Center-look pause; gathering optional confident tag observations')
        found = self._refresh_pose(now)
        if not found:
            self.nav.corner_source = 'Center look finished; calibrated pose fallback'
        if self.action == 'complete' and self._needs_more_return():
            self.action = 'return'
        self._begin_alignment(now)
        return self.nav._output('Center look acquired useful feedback' if found else 'Center look finished without useful feedback; continuing')

    def _begin_alignment(self, now):
        """Choose the next fixed vertex, the start position, or the final start heading.

        Args:
            now: Current ROS time in seconds.

        Returns:
            None. Plans a calibrated shortest-direction pivot, then a fixed acquisition pause.
        """
        nav = self.nav
        self.heading = self._target_heading()
        nav.course_heading = self.heading
        angle = wrap_angle(self.heading - nav.navigation_yaw)
        if abs(angle) < nav.policy.corner_min_correction_rad:
            self._enter('REORIENT_PAUSE', now)
        else:
            self._pivot(now, angle, 'REORIENTING')

    def _target_heading(self):
        """Aim at the fixed destination, or restore initial yaw after returning."""
        if self.action == 'complete':
            return self.nav.theta0_slam
        return math.atan2(self.target[1] - self.nav.navigation_y,
                          self.target[0] - self.nav.navigation_x)

    def _align_turn(self, now):
        """Reorient with the existing measured direction-specific motor pairs."""
        if self._pivot_finished(now):
            self._enter('REORIENT_PAUSE', now)
            return self.nav._output('Outgoing heading reached by prediction; stopped for optional feedback')
        return self.nav._output('Reorienting after late scan', angular=self.pivot_rate)

    def _align_pause(self, now):
        """Use up to three bounded corrections after alignment, then advance regardless.

        Args:
            now: Current ROS time in seconds.

        Returns:
            Stopped navigation output, or the next drive/completion transition.
            Correction attempts require new post-stop observations and share a 15-degree budget.
        """
        nav = self.nav
        if now - nav.phase_started < nav.turn_settle_s:
            return nav._output('Outgoing-heading pause; waiting a fixed camera interval')
        fresh = self._refresh_pose(now)
        if fresh:
            self.heading = self._target_heading()
            nav.course_heading = self.heading
        error = nav.policy.corner_ekf_weight * wrap_angle(self.heading - nav.navigation_yaw)
        angle = min(abs(error), nav.turn_pulse_rad, max(0.0, nav.policy.corner_correction_limit_rad - self.extra_used))
        if (fresh and angle + 1e-9 >= nav.policy.corner_min_correction_rad
                and self.extra_count < nav.policy.corner_max_corrections):
            self.extra_count += 1
            self.extra_used += angle
            self._pivot(now, math.copysign(angle, error), 'REORIENTING')
            return nav._output('Confident post-scan heading correction selected')
        return self._advance(now)

    def _advance(self, now):
        """Start the next leg, the return segment, or terminal completion.

        Args:
            now: Current ROS time in seconds.

        Returns:
            Stopped NavigationOutput. A final localization correction can resume the
            bounded return instead of falsely claiming arrival at the start.
        """
        nav = self.nav
        if self.action == 'complete':
            if self._needs_more_return():
                self.action = 'return'
                self._begin_alignment(now)
                return nav._output('Final localization revised start distance; resuming bounded return')
            self.return_result = self._return_outcome()
            self.active, nav.is_active, nav.state = False, False, self.states.COMPLETED
            return nav._output('Mission complete; ' + self.return_result)
        if self.action == 'return':
            return self._start_return_segment(now)
        self.active = False
        nav.current_wp_idx = nav.side_index + 1
        nav._begin_leg(now, heading=self.heading, target=self.target)
        return nav._output('Late scan finished; next side targets the planned vertex')

    def _home_distance(self):
        """Return estimated distance in meters to the fixed recorded start."""
        return math.hypot(self.nav.navigation_x - self.nav.x0_slam,
                          self.nav.navigation_y - self.nav.y0_slam)

    def _needs_more_return(self):
        """Allow another return segment only while outside tolerance with travel budget left."""
        return (self._home_distance() > self.nav.policy.return_tolerance_m
                and self.return_travel_m < self.nav.policy.return_max_distance_m - self.nav.waypoint_tolerance_m)

    def _return_outcome(self):
        """Describe estimated arrival separately from a budget-limited, unverified return."""
        distance = self._home_distance()
        outcome = 'estimated start reached' if distance <= self.nav.policy.return_tolerance_m else 'return budget exhausted; arrival unverified'
        return f'{outcome}; estimated residual {distance:.3f} m'

    def _finish_return(self, now):
        """Perform corner eight once, then restore the starting heading after any resumed return."""
        self.return_result = self._return_outcome()
        if self.nav.side_index < self.nav.side_count:
            self.nav._begin_corner(now)
        else:
            self.action = 'complete'
            self._begin_alignment(now)
        return self.nav._output(self.return_result + '; final heading acquisition')

    def _start_return_segment(self, now):
        """Drive toward the start in bounded segments, or finish the best-effort return.

        Args:
            now: Current ROS time in seconds.

        Returns:
            Stopped navigation output. A return budget exhaustion is logged as unverified,
            rather than claiming the physical robot arrived at the start.
        """
        nav = self.nav
        distance = self._home_distance()
        budget = max(0.0, nav.policy.return_max_distance_m - self.return_travel_m)
        if distance <= nav.policy.return_tolerance_m or budget <= nav.waypoint_tolerance_m:
            return self._finish_return(now)
        self.segment_goal_m = min(distance, nav.policy.return_segment_m, budget)
        self.segment_used_m = 0.0
        nav.current_wp_idx, nav.target = nav.side_count, (nav.x0_slam, nav.y0_slam)
        self._enter('RETURN_DRIVING', now)
        return nav._output(f'Return segment planned: {self.segment_goal_m:.3f} m; start distance {distance:.3f} m')

    def _return_drive(self, now):
        """Use fixed forward inputs until the current return segment is complete."""
        if self.segment_used_m >= self.segment_goal_m:
            self._enter('RETURN_PAUSE', now)
            return self.nav._output('Return segment finished; stopped to update distance and bearing')
        return self.nav._output('Returning toward recorded start', linear=self.nav.forward.speed_mps)

    def _return_pause(self, now):
        """Refresh localization between return segments, then choose the next bearing."""
        if now - self.nav.phase_started < self.nav.turn_settle_s:
            return self.nav._output('Return pause; optional localization update')
        if not self._refresh_pose(now):
            self.nav.corner_source = 'Return checkpoint; calibrated pose fallback'
        self._begin_alignment(now)
        return self.nav._output('Updated bearing toward recorded start; timed fallback if tags missing')
