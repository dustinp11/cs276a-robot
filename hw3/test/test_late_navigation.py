"""Checks for bounded late-corner scans and a last leg aimed at the recorded start."""

import math
import unittest

from hw3.square_navigation import NavigationState, SquareNavigator
from hw3.turn_calibration import TurnCalibration
from hw3.vision_navigation import NavigationPolicy


class TestLateNavigation(unittest.TestCase):
    def navigator(self, **overrides):
        """Start an octagon with its measured pivots and enabled late-corner behavior.

        Args:
            overrides: Optional motion-policy overrides.

        Returns:
            Started pure navigator for local state-machine checks.
        """
        values = dict(corner_turn_duration_s=0.5, corner_stage_angle_rad=math.pi / 6,
                      late_scan_start_corner=6)
        values.update(overrides)
        calibration = TurnCalibration(-0.35, 0.35, 0.37, -0.37, math.pi / 2, math.pi / 2)
        nav = SquareNavigator(side_count=8, side_length_m=0.75, turn_calibration=calibration,
                              policy=NavigationPolicy(**values))
        nav.start()
        nav.step(0)
        return nav

    def observe(self, nav, stamps, pose=(0.2, 0.3, 0.4)):
        """Feed consistent stamped tag frames and their matching low-uncertainty poses."""
        for stamp in stamps:
            nav.update_vision(stamp, [[42, 2, 0]], [42], None, [[42, 0.6]],
                              [[42, 0.02]], [pose[2], 0.02], [[42, 1, 1, 0.02]])
            nav.update_ekf_pose(*pose, 0.02, 0.02, stamp)

    def test_full_octagon_and_return_complete_without_tags(self):
        """No observation is required for the complete bounded late-mission sequence.

        Args:
            None.

        Returns:
            None. Checks scans at corners 6–8, segmented return, closure, and termination.
        """
        nav = self.navigator()
        scanned, returning, previous = set(), 0, None
        for tick in range(1, 12000):
            cmd = nav.step(tick * 0.05)
            if cmd.state == NavigationState.RELOCALIZING_TURN:
                scanned.add(nav.side_index)
            returning += int(cmd.state == NavigationState.RETURN_DRIVING and previous != cmd.state)
            self.assertFalse(cmd.linear_v_mps and cmd.angular_omega_radps)
            previous = cmd.state
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertEqual(scanned, {6, 7, 8})
        self.assertGreaterEqual(returning, 2)
        self.assertEqual(nav.side_index, 8)
        self.assertLessEqual(math.hypot(nav.navigation_x, nav.navigation_y), 0.10)
        self.assertLessEqual(nav.late.return_travel_m, nav.policy.return_max_distance_m + 0.02)
        self.assertIn('estimated start reached', cmd.status_message)

    def test_scan_faces_lowest_covariance_tag_without_hard_coded_ids(self):
        nav = self.navigator()
        nav.navigation_yaw = 0
        nav.vision.mapped_landmarks = [[0, 2, 0, 0.05], [42, 0, 2, 0.01]]
        tag, angle = nav.late._scan_direction()
        self.assertEqual(tag, 42)
        self.assertAlmostEqual(angle, math.pi / 2)

    def test_confident_post_stop_frames_end_scan_early(self):
        """Verify confident post stop frames end scan early.

        Args:
            None.

        Returns:
            None. Checks commands and transitions without ROS or hardware.
        """
        nav = self.navigator()
        nav.side_index = 6
        nav.late.start(1, 6)
        nav.late._enter('RELOCALIZING_PAUSE', 3)
        self.observe(nav, (4, 5, 5.8))
        nav.step(6)
        self.assertIn(nav.state, (NavigationState.REORIENTING, NavigationState.REORIENT_PAUSE))
        self.assertAlmostEqual(nav.navigation_x, 0.2)
        self.assertAlmostEqual(nav.navigation_y, 0.3)
        self.assertAlmostEqual(nav.late.scan_used, 0)
        self.assertIn('42', nav.corner_source)

    def test_scan_budget_exhaustion_continues_without_tags(self):
        nav = self.navigator()
        nav.side_index = 6
        nav.late.start(1, 6)
        nav.late.scan_used = nav.policy.late_scan_max_angle_rad
        nav.late._enter('RELOCALIZING_PAUSE', 3)
        nav.step(6)
        self.assertIn(nav.state, (NavigationState.REORIENTING, NavigationState.REORIENT_PAUSE))
        self.assertTrue(nav.is_active)

    def test_pre_stop_observations_do_not_end_scan(self):
        nav = self.navigator()
        nav.side_index = 6
        nav.late.start(1, 6)
        self.observe(nav, (1.1, 1.5, 2))
        nav.late._enter('RELOCALIZING_PAUSE', 3)
        nav.step(6)
        self.assertEqual(nav.state, NavigationState.RELOCALIZING_TURN)

    def test_last_leg_targets_fixed_start_after_pose_updates(self):
        """Verify last leg targets fixed start after pose updates.

        Args:
            None.

        Returns:
            None. Checks commands and transitions without ROS or hardware.
        """
        nav = self.navigator()
        nav.x0_slam, nav.y0_slam = 1.2, -0.7
        nav.navigation_x, nav.navigation_y = 2, 0.5
        nav.side_index = 7
        nav.late.start(1, 7)
        self.assertEqual(nav.late.target, (1.2, -0.7))
        self.observe(nav, (2, 3, 3.8), pose=(2.1, 0.6, 0.4))
        self.assertEqual((nav.x0_slam, nav.y0_slam), (1.2, -0.7))
        nav.late._begin_alignment(4)
        self.assertAlmostEqual(nav.late.heading, math.atan2(-1.3, -0.9))

    def test_return_segment_uses_actual_remaining_distance(self):
        """Verify return segment uses actual remaining distance.

        Args:
            None.

        Returns:
            None. Checks commands and transitions without ROS or hardware.
        """
        nav = self.navigator()
        nav.side_index = 7
        nav.late.start(1, 7)
        nav.navigation_x, nav.navigation_y = 0.42, 0
        nav.late._start_return_segment(2)
        self.assertEqual(nav.state, NavigationState.RETURN_DRIVING)
        self.assertAlmostEqual(nav.late.segment_goal_m, 0.25)
        nav.navigation_x = 0.15
        nav.late._start_return_segment(3)
        self.assertAlmostEqual(nav.late.segment_goal_m, 0.15)
        self.assertEqual(nav.target, (0, 0))

    def test_final_localization_can_resume_return_without_a_ninth_corner(self):
        """A corner-eight pose correction can reveal closure error and resume travel.

        Args:
            None.

        Returns:
            None. Verifies final reacquisition does not falsely complete or add a ninth corner.
        """
        nav = self.navigator()
        nav.side_index = 8
        nav.late.start(1, 8)
        nav.late._enter('RELOCALIZING_PAUSE', 3)
        self.observe(nav, (4, 5, 5.8), pose=(0.5, 0, 0))
        nav.step(6)
        self.assertEqual(nav.late.action, 'return')
        self.assertTrue(nav.is_active)
        nav.navigation_x = 0.05
        nav.late._start_return_segment(7)
        self.assertEqual(nav.side_index, 8)
        self.assertEqual(nav.late.action, 'complete')
        self.assertIn(nav.state, (NavigationState.REORIENTING, NavigationState.REORIENT_PAUSE))

    def test_return_budget_exhaustion_is_not_reported_as_arrival(self):
        nav = self.navigator()
        nav.side_index = 7
        nav.late.start(1, 7)
        nav.navigation_x, nav.navigation_y = 2, 0
        nav.late.return_travel_m = nav.policy.return_max_distance_m
        cmd = nav.late._start_return_segment(2)
        self.assertEqual(nav.side_index, 8)
        self.assertIn('arrival unverified', cmd.status_message)
        self.assertTrue(nav.is_active)

    def test_return_checkpoints_do_not_reset_corner_correction_budget(self):
        nav = self.navigator()
        nav.side_index = 7
        nav.late.start(1, 7)
        nav.late.extra_count = 3
        nav.late.extra_used = nav.policy.corner_correction_limit_rad
        nav.late._begin_alignment(2)
        self.assertEqual(nav.late.extra_count, 3)
        self.assertEqual(nav.late.extra_used, nav.policy.corner_correction_limit_rad)

    def test_manual_stop_interrupts_scan_and_return(self):
        for state in (NavigationState.RELOCALIZING_TURN, NavigationState.RETURN_DRIVING):
            nav = self.navigator()
            nav.side_index = 7
            nav.late.start(1, 7)
            nav._transition(state, 1)
            nav.stop()
            cmd = nav.step(2)
            self.assertEqual(cmd.state, NavigationState.STOPPED)
            self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))

    def test_late_navigation_limits_are_validated(self):
        for values in ({'late_scan_start_corner': 6.5}, {'late_scan_max_angle_rad': 7},
                       {'return_segment_m': 4}, {'return_tolerance_m': -0.1}):
            with self.assertRaises(ValueError):
                NavigationPolicy(**values)


if __name__ == '__main__':
    unittest.main()
