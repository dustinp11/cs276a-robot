"""One staged startup revolution, bounded heading closure, and relaxed confidence gate."""

import math
import unittest

from hw3.square_navigation import NavigationState, SquareNavigator
from hw3.turn_calibration import TurnCalibration
from hw3.vision_navigation import NavigationPolicy, VisualFeedback


class TestStartupScan(unittest.TestCase):
    def navigator(self):
        """Start the octagon profile with a scan and five-degree correction pulses."""
        calibration = TurnCalibration(-0.35, 0.35, 0.37, -0.37, math.pi / 2, math.pi / 2)
        nav = SquareNavigator(side_count=8, side_length_m=0.75, turn_calibration=calibration,
                              turn_pulse_rad=math.pi / 36,
                              policy=NavigationPolicy(startup_scan_enabled=True,
                                                      corner_turn_duration_s=0.5,
                                                      corner_stage_angle_rad=math.pi / 6))
        nav.start()
        return nav

    def observe(self, nav, stamps, tag_yaw):
        """Feed distinct stationary observations without requiring a globally mapped tag."""
        for stamp in stamps:
            nav.update_vision(stamp, [[7, 2, 0]], [], None, [[7, tag_yaw]])

    def test_full_scan_precedes_forward_motion_without_tags(self):
        """Scan exactly one nominal circle in eight stopped-to-stopped steps.

        Args:
            None.

        Returns:
            None. Verifies no required detections, no translation, and an unchanged start frame.
        """
        nav = self.navigator()
        previous, turns, pauses = None, 0, 0
        for tick in range(1200):
            cmd = nav.step(tick * 0.05)
            turns += int(cmd.state == NavigationState.STARTUP_SCAN_TURN and previous != cmd.state)
            pauses += int(cmd.state == NavigationState.STARTUP_SCAN_PAUSE and previous != cmd.state)
            if cmd.linear_v_mps:
                break
            self.assertEqual(nav.drive_used, 0)
            self.assertEqual((nav.navigation_x, nav.navigation_y), (0, 0))
            previous = cmd.state
        self.assertGreater(cmd.linear_v_mps, 0)
        self.assertEqual((turns, pauses), (8, 9))
        self.assertGreaterEqual(nav.startup_scan_used + 1e-9, 2 * math.pi)
        self.assertLessEqual(nav.startup_scan_used, 2 * math.pi + math.pi / 40 + 1e-6)
        self.assertEqual(nav.side_index, 0)
        self.assertEqual((nav.x0_slam, nav.y0_slam, nav.theta0_slam), (0, 0, 0))
        self.assertEqual(nav.startup_correction_count, 0)

    def test_complete_octagon_with_startup_scan_and_no_sensor_messages(self):
        """The startup scan and full octagon must finish with no vision or EKF feedback.

        Args:
            None.

        Returns:
            None. Checks no late maneuvers were re-enabled and all eight corners execute.
        """
        nav = self.navigator()
        corners = set()
        for tick in range(6000):
            cmd = nav.step(tick * 0.05)
            self.assertFalse(cmd.linear_v_mps and cmd.angular_omega_radps)
            if cmd.state == NavigationState.TURNING:
                corners.add(nav.side_index)
            self.assertNotIn(cmd.state, (NavigationState.RELOCALIZING_TURN,
                                         NavigationState.RELOCALIZING_PAUSE,
                                         NavigationState.REORIENTING,
                                         NavigationState.REORIENT_PAUSE,
                                         NavigationState.RETURN_DRIVING,
                                         NavigationState.RETURN_PAUSE))
            if not nav.is_active:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertEqual(corners, set(range(1, 9)))

    def test_scan_closure_correction_can_turn_either_direction(self):
        """Use the same initial tag to detect residual yaw after the nominal full revolution.

        Args:
            None.

        Returns:
            None. Under/overshoot select opposite five-degree pulses without relatching origin.
        """
        for residual in (-10, 10):
            nav = self.navigator()
            nav.step(0)
            self.observe(nav, (1, 2, 2.8), 0.4)
            nav.step(3)
            nav.startup_scan_used = 2 * math.pi
            nav._transition(NavigationState.STARTUP_SCAN_PAUSE, 10)
            self.observe(nav, (11, 12, 12.8), 0.4 - math.radians(residual))
            cmd = nav.step(13)
            self.assertEqual(cmd.state, NavigationState.STARTUP_SCAN_CORRECTING)
            self.assertAlmostEqual(nav.startup_correction_goal, math.radians(5))
            self.assertEqual(nav.startup_correction_count, 1)
            cmd = nav.step(13.05)
            self.assertLess(cmd.angular_omega_radps * residual, 0)
            self.assertEqual(cmd.linear_v_mps, 0)
            self.assertEqual((nav.x0_slam, nav.y0_slam, nav.theta0_slam), (0, 0, 0))

    def test_stale_startup_observations_do_not_request_a_correction(self):
        nav = self.navigator()
        nav.step(0)
        nav.startup_reference = {7: 0.4}
        nav.startup_scan_used = 2 * math.pi
        self.observe(nav, (1, 2, 2.8), 0.2)
        nav._transition(NavigationState.STARTUP_SCAN_PAUSE, 10)
        nav.step(13)
        self.assertEqual(nav.state, NavigationState.DRIVING)
        self.assertEqual(nav.startup_correction_count, 0)

    def test_manual_stop_during_startup_turn(self):
        nav = self.navigator()
        nav.step(0)
        nav.step(3)
        self.assertGreater(nav.step(3.05).angular_omega_radps, 0)
        nav.stop()
        cmd = nav.step(3.1)
        self.assertEqual(cmd.state, NavigationState.STOPPED)
        self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))

    def test_fifteen_degree_heading_uncertainty_gate(self):
        for sigma, accepted in ((11.6, True), (15, True), (15.1, False)):
            feedback = VisualFeedback(NavigationPolicy())
            feedback.update(3.2, [[7, 2, 0]], [7], None, (0, 0, 0), 0,
                            landmark_sigmas=[[7, 0.02]], ekf_heading=[0.1, math.radians(sigma)])
            yaw = feedback.confident_ekf_yaw(3.2, 3)
            self.assertEqual(yaw is not None, accepted)

    def test_startup_scan_settings_are_validated(self):
        for values in ({'startup_scan_enabled': 1}, {'startup_scan_step_rad': math.pi},
                       {'startup_scan_pause_s': -1}):
            with self.assertRaises(ValueError):
                NavigationPolicy(**values)


if __name__ == '__main__':
    unittest.main()
