"""Fixed targets, optional stopped feedback, and bounded waypoint mission behavior."""

import math
import unittest

from hw3.square_navigation import NavigationState, SquareNavigator
from hw3.turn_calibration import TurnCalibration
from hw3.vision_navigation import NavigationPolicy
from hw3.waypoint_navigation import polygon_vertices


class TestWaypointNavigation(unittest.TestCase):
    def navigator(self, laps=1):
        """Return a started eight-side mission with fixed-rate motor calibration."""
        calibration = TurnCalibration(-0.35, 0.35, 0.37, -0.37, math.pi / 2, math.pi / 2)
        nav = SquareNavigator(side_count=8, side_length_m=0.75,
                              turn_calibration=calibration,
                              policy=NavigationPolicy(waypoint_motion_enabled=True,
                                                      waypoint_laps=laps,
                                                      waypoint_pause_s=1.0),
                              turn_pulse_rad=math.radians(10))
        nav.start()
        return nav

    def feedback(self, nav, pose, stamp):
        """Deliver one confident pose and matching known-tag observation through the cache."""
        nav.update_ekf_pose(*pose, 0.05, 0.05, stamp)
        nav.vision.stamp, nav.vision.known_ids = stamp, {7}

    def test_vertices_are_closed_and_rigidly_anchored(self):
        """All eight fixed edges are 75 cm in an arbitrary initial SLAM frame."""
        origin = (2.0, -1.0, 0.37)
        vertices = polygon_vertices(origin, 0.75, 8)
        self.assertEqual(vertices[-1], origin[:2])
        for a, b in zip((origin[:2],) + vertices[:-1], vertices):
            self.assertAlmostEqual(math.dist(a, b), 0.75)
        self.assertAlmostEqual(math.atan2(vertices[0][1] + 1, vertices[0][0] - 2),
                               origin[2] + math.pi / 4)

    def test_no_tags_completes_without_scans_or_homing(self):
        """Missing feedback never stalls progress or changes the fixed planned endpoints.

        Args:
            None.

        Returns:
            None. Checks bounded duration, finite motor commands, and exact target closure.
        """
        nav = self.navigator()
        targets, forward_time = {}, 0.0
        for tick in range(2000):
            output = nav.step(tick * 0.05)
            targets.setdefault(output.waypoint_index, (output.target_x_m, output.target_y_m))
            self.assertEqual(targets[output.waypoint_index], (output.target_x_m, output.target_y_m))
            self.assertFalse(output.linear_v_mps and output.angular_omega_radps)
            forward_time += 0.05 if output.linear_v_mps else 0
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertEqual(nav.side_index, 8)
        self.assertLess(tick * 0.05, 80)
        self.assertLess(math.dist(nav.waypoint.pose[:2], (0, 0)), 0.06)
        self.assertLess(forward_time * nav.forward.speed_mps, 8 * 0.76)

    def test_stopped_feedback_corrects_in_both_directions(self):
        """Fresh single-tag observations can correct an underturn or an overshoot.

        Args:
            None.

        Returns:
            None. Each correction uses the corresponding full calibrated pivot rate.
        """
        for offset, direction in ((0.20, -1), (-0.20, 1)):
            nav = self.navigator()
            candidate = None
            for tick in range(200):
                now = tick * 0.05
                if nav.waypoint.phase == 'after_turn' and nav.state == NavigationState.STOPPING:
                    candidate = candidate or (*nav.waypoint.pose[:2], nav.waypoint.pose[2] + offset)
                    self.feedback(nav, candidate, now)
                output = nav.step(now)
                if output.state == NavigationState.CORRECTING:
                    self.assertEqual(output.angular_omega_radps, direction * math.pi / 2)
                    self.assertEqual(nav.waypoint.source, 'accepted stopped SLAM pose')
                    break
            else:
                self.fail('Expected an optional stopped heading correction.')

    def test_jump_and_stale_feedback_do_not_move_vertices(self):
        """Reject implausible or stale feedback without changing the original plan.

        Args:
            None.

        Returns:
            None. Verifies prediction and the fixed vertex tuple remain unchanged.
        """
        nav = self.navigator()
        nav.step(0)
        vertices = nav.waypoint.vertices
        nav.waypoint.samples.extend([(10.0, 10.0, 0.0)] * 3)
        nav.waypoint.image_stamp = 0.8
        self.assertFalse(nav.waypoint._accept_feedback(1.0))
        self.assertEqual(nav.waypoint.pose, (0.0, 0.0, 0.0))
        nav.waypoint.samples.clear()
        nav.waypoint.samples.extend([(0.1, 0, 0)] * 3)
        self.assertFalse(nav.waypoint._accept_feedback(3.0))
        self.assertEqual(vertices, nav.waypoint.vertices)

    def test_unstable_feedback_is_rejected(self):
        """Three distinct observations must agree before they can influence motion."""
        nav = self.navigator()
        nav.step(0)
        nav.waypoint.samples.extend([(0, 0, 0), (0.2, 0, 0), (0.1, 0, 0)])
        nav.waypoint.image_stamp = 0.9
        self.assertFalse(nav.waypoint._accept_feedback(1.0))

    def test_distance_adjustment_has_a_total_side_budget(self):
        """Repeated estimates cannot accumulate unbounded extra forward travel.

        Args:
            None.

        Returns:
            None. Verifies the per-side 90 cm motor-on distance cap.
        """
        nav = self.navigator()
        nav.step(0)
        nav.waypoint.phase = 'after_turn'
        nav.waypoint.leg_distance = 0.75
        nav.waypoint.pose = (-0.3, 0, 0)
        nav.waypoint._resume_drive(1, True)
        nav.drive_used = 0.375
        nav.waypoint.phase = 'midpoint'
        nav.waypoint._resume_drive(2, True)
        self.assertLessEqual(nav.waypoint.drive_goal, 0.90)

    def test_repeated_inaccurate_feedback_cannot_spin_forever(self):
        """Even adversarial accepted heading feedback spends only three bounded pulses.

        Args:
            None.

        Returns:
            None. The controller eventually drives rather than continually correcting.
        """
        nav = self.navigator()
        correcting_ticks = 0
        for tick in range(400):
            now = tick * 0.05
            if nav.waypoint.phase == 'after_turn':
                self.feedback(nav, (0, 0, math.pi / 4 + 0.20), now)
            output = nav.step(now)
            correcting_ticks += output.state == NavigationState.CORRECTING
            if nav.waypoint.phase == 'first_half':
                break
        self.assertEqual(nav.waypoint.attempts, 3)
        self.assertLessEqual(correcting_ticks, 9)
        self.assertEqual(output.state, NavigationState.DRIVING)

    def test_stop_interrupts_any_phase(self):
        """The unchanged stop service can terminate driving, pauses and pivots immediately."""
        for ticks in (1, 22, 35, 60, 90):
            nav = self.navigator()
            for tick in range(ticks):
                nav.step(tick * 0.05)
            nav.stop()
            output = nav.step(ticks * 0.05)
            self.assertEqual((output.linear_v_mps, output.angular_omega_radps), (0, 0))
            self.assertEqual(output.state, NavigationState.STOPPED)

    def test_manual_and_waypoint_modes_are_mutually_exclusive(self):
        """Fail clearly if a configuration accidentally enables both controllers."""
        with self.assertRaises(ValueError):
            NavigationPolicy(manual_motion_enabled=True, waypoint_motion_enabled=True)

    def test_default_waypoint_pause_is_three_seconds(self):
        """Use three-second acquisition by default without changing timed baseline pauses."""
        policy = NavigationPolicy(waypoint_motion_enabled=True)
        self.assertEqual(policy.waypoint_pause_s, 3.0)
        self.assertEqual(policy.manual_midpoint_pause_s, 1.0)
        self.assertEqual(policy.manual_forward_pause_s, 1.0)

    def test_two_laps_reuse_targets_and_estimator_feedback(self):
        """Traverse sixteen sides without resetting the frame or feedback tracker between laps.

        Args:
            None.

        Returns:
            None. Checks repeated fixed targets, lap numbering, and final completion.
        """
        nav, targets = self.navigator(laps=2), {}
        feedback_tracker = nav.vision
        for tick in range(4000):
            output = nav.step(tick * 0.05)
            targets[output.waypoint_index] = nav.target
            self.assertIs(nav.vision, feedback_tracker)
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertEqual(nav.side_index, 16)
        self.assertEqual(nav.waypoint.lap_index, 2)
        for index in range(1, 9):
            self.assertEqual(targets[index], targets[index + 8])
        self.assertEqual(nav.target, (nav.x0_slam, nav.y0_slam))

    def test_invalid_lap_counts_are_rejected(self):
        """Reject zero, fractional and boolean lap counts before starting motion."""
        for laps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                NavigationPolicy(waypoint_laps=laps)

    def simulated_underturn(self, observations):
        """Run a planar plant whose pivots achieve only 85 percent of the nominal angle.

        Args:
            observations: Whether perfect stopped tag-based poses reach the controller.

        Returns:
            Mean physical distance to the current target at each side's final pause.
        """
        nav, physical, previous, errors = self.navigator(), (0, 0, 0), (0, 0), []
        for tick in range(3000):
            x, y, yaw = physical
            v, omega = previous
            physical = (x + v * 0.05 * math.cos(yaw), y + v * 0.05 * math.sin(yaw),
                        yaw + omega * 0.05 * 0.85)
            if observations and tick % 2 == 0:
                self.feedback(nav, physical, tick * 0.05)
            old_phase = nav.waypoint.phase
            output = nav.step(tick * 0.05)
            previous = output.linear_v_mps, output.angular_omega_radps
            if old_phase != 'end' and nav.waypoint.phase == 'end':
                errors.append(math.dist(physical[:2], nav.target))
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertEqual(len(errors), 8)
        return sum(errors) / len(errors)

    def test_stopped_slam_feedback_improves_systematic_underturn(self):
        """A repeatable pivot bias produces smaller waypoint errors with stopped feedback."""
        baseline = self.simulated_underturn(False)
        corrected = self.simulated_underturn(True)
        self.assertLess(corrected, baseline * 0.75)

    def test_passed_vertex_does_not_request_a_reverse_chase(self):
        """A confident midpoint pose beyond the target ends the leg instead of turning back.

        Args:
            None.

        Returns:
            None. Verifies a stopped endpoint command rather than a reverse chase.
        """
        nav = self.navigator()
        nav.step(0)
        nav.waypoint.phase = 'midpoint'
        x, y = nav.target
        nav.waypoint.pose = (x + 0.1, y + 0.1, math.pi / 4)
        nav.drive_used, nav.waypoint.leg_distance = 0.375, 0.75
        output = nav.waypoint._resume_drive(1, True)
        self.assertEqual((output.linear_v_mps, output.angular_omega_radps), (0, 0))
        self.assertEqual(nav.waypoint.phase, 'end')


if __name__ == '__main__':
    unittest.main()
