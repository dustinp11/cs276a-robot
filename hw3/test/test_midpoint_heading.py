"""Stopped midpoint heading corrections and bounded remaining-distance updates."""

import math
import unittest

from hw3.square_navigation import NavigationState, SquareNavigator
from hw3.turn_calibration import TurnCalibration
from hw3.vision_navigation import NavigationPolicy


class TestMidpointHeading(unittest.TestCase):
    def navigator(self, **policy_overrides):
        """Create a half-traveled leg stopped for a one-second acquisition fixture.

        Args:
            policy_overrides: Additional NavigationPolicy settings for the check.

        Returns:
            Pure navigator in MIDPOINT_PAUSE with its fixed forward target preserved.
        """
        values = dict(midpoint_pause_s=1.0)
        values.update(policy_overrides)
        calibration = TurnCalibration(-0.35, 0.35, 0.37, -0.37, math.pi / 2, math.pi / 2)
        nav = SquareNavigator(side_length_m=0.75, turn_calibration=calibration,
                              policy=NavigationPolicy(**values))
        nav.start()
        nav.step(0)
        nav.drive_used = nav.square_size_m / 2
        nav.last_command = (0, 0)
        nav.step(1)
        return nav

    def observe(self, nav, stamps, x, yaw, known=True):
        """Supply matching confident poses and stable normal observations for one stop."""
        for stamp in stamps:
            ids = [7] if known else []
            sigmas = [[7, 0.02]] if known else []
            heading = [yaw, 0.02] if known else None
            nav.update_vision(stamp, [[7, 2, 0]], ids, None, [[7, 0.4 - yaw]], sigmas, heading)
            nav.update_ekf_pose(x, 0, yaw, 0.02, 0.02, stamp)

    def test_midpoint_selects_heading_correction_in_both_directions(self):
        """A stopped confident heading observation can request either measured pivot pair.

        Args:
            None.

        Returns:
            None. Checks half-weighted 10-degree pivots, stopped selection, and fixed course.
        """
        for yaw, direction in ((math.radians(20), -1), (math.radians(-20), 1)):
            nav = self.navigator()
            target, heading = nav.target, nav._heading()
            self.observe(nav, (1.1, 1.5, 1.9), 0.4, yaw)
            cmd = nav.step(2)
            self.assertEqual(cmd.state, NavigationState.MIDPOINT_CORRECTING)
            self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))
            self.assertEqual((nav.target, nav._heading()), (target, heading))
            self.assertAlmostEqual(nav.midpoint_correction_goal, math.radians(10))
            cmd = nav.step(2.05)
            self.assertEqual(cmd.linear_v_mps, 0)
            self.assertGreater(cmd.angular_omega_radps * direction, 0)

    def test_pivot_is_followed_by_pause_and_updated_distance_plan(self):
        """Recheck position after rotating rather than keeping the earlier midpoint estimate.

        Args:
            None.

        Returns:
            None. Checks a second stopped acquisition updates remaining forward travel.
        """
        nav = self.navigator()
        self.observe(nav, (1.1, 1.5, 1.9), 0.425, math.radians(20))
        nav.step(2)
        for tick in range(1, 20):
            cmd = nav.step(2 + tick * 0.05)
            if cmd.state == NavigationState.MIDPOINT_PAUSE:
                break
        self.assertEqual(nav.state, NavigationState.MIDPOINT_PAUSE)
        stopped = nav.phase_started
        self.assertEqual(nav.step(stopped + 0.8).linear_v_mps, 0)
        self.observe(nav, (stopped + 0.1, stopped + 0.5, stopped + 0.9), 0.295, 0)
        nav.step(stopped + 1)
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertAlmostEqual(nav.progress_correction_m, -0.08)
        self.assertAlmostEqual(nav.midpoint_remaining_m, 0.455)
        self.assertEqual(nav.midpoint_correction_count, 1)
        self.assertGreater(nav.step(stopped + 1.05).linear_v_mps, 0)

    def test_same_tag_drift_can_correct_without_confident_global_map(self):
        nav = self.navigator()
        nav.leg_reference = {7: 0.4}
        self.observe(nav, (1.1, 1.5, 1.9), 0.375, math.radians(12), known=False)
        nav.step(2)
        self.assertEqual(nav.state, NavigationState.MIDPOINT_CORRECTING)
        self.assertAlmostEqual(nav.midpoint_correction_goal, math.radians(12))
        self.assertLess(nav.midpoint_correction_omega, 0)
        self.assertIn('same-tag', nav.midpoint_heading_source)

    def test_stale_heading_feedback_cannot_select_a_midpoint_turn(self):
        nav = self.navigator()
        self.observe(nav, (1.1, 1.5, 1.9), 0.375, math.radians(30))
        nav.step(4)
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertEqual(nav.midpoint_correction_count, 0)

    def test_fourth_midpoint_correction_is_not_selected(self):
        """Count limit applies even when a deliberately larger angular budget remains.

        Args:
            None.

        Returns:
            None. Persistent fresh heading error cannot prevent forward continuation.
        """
        nav = self.navigator(corner_correction_limit_rad=math.pi / 6)
        nav.turn_pulse_rad = math.radians(5)
        for attempt in range(4):
            stopped = 1 + attempt * 2
            nav._transition(NavigationState.MIDPOINT_PAUSE, stopped)
            self.observe(nav, (stopped + 0.1, stopped + 0.5, stopped + 0.9), 0.375, math.radians(20))
            nav.step(stopped + 1)
            self.assertEqual(nav.midpoint_correction_count, min(attempt + 1, 3))
            if attempt < 3:
                self.assertEqual(nav.state, NavigationState.MIDPOINT_CORRECTING)
                nav.midpoint_correction_used = nav.midpoint_correction_goal
        self.assertEqual(nav.state, NavigationState.DRIVING)

    def test_missing_new_pose_retains_previous_distance_adjustment(self):
        """A camera gap after a pivot must not erase a previously accepted distance update.

        Args:
            None.

        Returns:
            None. Confirms the old bounded correction survives while forward travel resumes.
        """
        nav = self.navigator()
        self.observe(nav, (1.1, 1.5, 1.9), 0.425, math.radians(20))
        nav.step(2)
        nav.midpoint_correction_used = nav.midpoint_correction_goal
        nav._transition(NavigationState.MIDPOINT_PAUSE, 3)
        nav.step(4)
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertAlmostEqual(nav.progress_correction_m, 0.05)
        self.assertAlmostEqual(nav.midpoint_remaining_m, 0.325)
        self.assertIn('retained', nav.midpoint_source)

    def test_midpoint_total_angle_budget_limits_corrections(self):
        """Limit all midpoint heading pulses to the configured total nominal angle.

        Args:
            None.

        Returns:
            None. Persistent observed error cannot authorize rotation beyond the budget.
        """
        nav = self.navigator()
        self.observe(nav, (1.1, 1.5, 1.9), 0.375, math.radians(40))
        nav.step(2)
        self.assertAlmostEqual(nav.midpoint_correction_goal, math.radians(15))
        nav.midpoint_correction_used = nav.midpoint_correction_goal
        nav._transition(NavigationState.MIDPOINT_PAUSE, 3)
        self.observe(nav, (3.1, 3.5, 3.9), 0.375, math.radians(40))
        nav.step(4)
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertEqual(nav.midpoint_correction_count, 1)

    def test_manual_stop_interrupts_midpoint_pivot(self):
        nav = self.navigator()
        self.observe(nav, (1.1, 1.5, 1.9), 0.375, math.radians(20))
        nav.step(2)
        nav.stop()
        cmd = nav.step(2.1)
        self.assertEqual(cmd.state, NavigationState.STOPPED)
        self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))


if __name__ == '__main__':
    unittest.main()
