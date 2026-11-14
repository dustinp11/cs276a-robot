"""Fixed timed octagon traversal with short midpoint and pre-turn pauses."""

import math
import unittest

from hw3.square_navigation import NavigationState, SquareNavigator
from hw3.turn_calibration import TurnCalibration
from hw3.vision_navigation import NavigationPolicy


class TestManualNavigation(unittest.TestCase):
    def navigator(self):
        """Start simple motion while deliberately enabling features it must bypass.

        Args:
            None.

        Returns:
            Started navigator with one-second midpoint and pre-turn pauses.
        """
        calibration = TurnCalibration(-0.35, 0.35, 0.37, -0.37, math.pi / 2, math.pi / 2)
        policy = NavigationPolicy(manual_motion_enabled=True, manual_forward_pause_s=1.0,
                                  manual_midpoint_pause_s=1.0,
                                  startup_scan_enabled=True, late_scan_start_corner=6,
                                  corner_turn_duration_s=0.5)
        nav = SquareNavigator(side_count=8, side_length_m=0.75,
                              turn_calibration=calibration, policy=policy)
        nav.start()
        return nav

    def test_eight_cycles_have_one_second_midpoint_and_pre_turn_pauses(self):
        """Measure turns, forward halves, and both one-second pauses at 20 Hz.

        Args:
            None.

        Returns:
            None. Confirms 0.5-second pivots, nominal 75 cm legs, and no extra phases.
        """
        nav = self.navigator()
        runs, previous = [], None
        for tick in range(1000):
            cmd = nav.step(tick * 0.05)
            self.assertIn(cmd.state, (NavigationState.TURNING, NavigationState.DRIVING,
                                     NavigationState.MIDPOINT_PAUSE, NavigationState.STOPPING, NavigationState.COMPLETED))
            if cmd.state != previous:
                runs.append([cmd.state, 0])
            runs[-1][1] += 1
            if cmd.state == NavigationState.TURNING:
                self.assertEqual(cmd.linear_v_mps, 0)
                self.assertEqual(nav.turn_calibration.motors(cmd.angular_omega_radps), (-0.35, 0.35))
            elif cmd.state == NavigationState.DRIVING:
                self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0.203, 0))
            else:
                self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))
            previous = cmd.state
            if not nav.is_active:
                break
        cycle = [(NavigationState.TURNING, 10), (NavigationState.DRIVING, 37),
                 (NavigationState.MIDPOINT_PAUSE, 20), (NavigationState.DRIVING, 37),
                 (NavigationState.STOPPING, 20)]
        self.assertEqual([tuple(run) for run in runs[:-1]], cycle * 8)
        self.assertEqual(nav.side_index, 8)
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertAlmostEqual(tick * 0.05, 49.6)
        self.assertEqual(nav.startup_scan_used, 0)
        self.assertEqual(nav.midpoint_correction_count, 0)
        self.assertFalse(nav.late.active)

    def test_pose_and_tag_feedback_cannot_change_timed_commands(self):
        """Arbitrary conflicting pose estimates cannot steer or extend the simple mission.

        Args:
            None.

        Returns:
            None. Compares every command and phase against an identical sensor-free run.
        """
        baseline, noisy = self.navigator(), self.navigator()
        for tick in range(1000):
            now = tick * 0.05
            yaw = math.sin(now) * 3
            noisy.update_vision(now, [[7, 2, 0]], [7], None, [[7, yaw]],
                                [[7, 0.01]], [yaw, 0.01])
            noisy.update_ekf_pose(5 * math.cos(now), 5 * math.sin(now), yaw, 0.01, 0.01, now)
            expected, actual = baseline.step(now), noisy.step(now)
            self.assertEqual((actual.linear_v_mps, actual.angular_omega_radps, actual.state),
                             (expected.linear_v_mps, expected.angular_omega_radps, expected.state))
            if not baseline.is_active:
                break
        self.assertEqual(noisy.state, NavigationState.COMPLETED)

    def test_manual_stop_interrupts_turn_drive_and_both_pauses(self):
        """The manual stop interrupts every phase of the simplified motion sequence.

        Args:
            None.

        Returns:
            None. Checks terminal zero commands during turns, travel, and both pauses.
        """
        for phase_time in (0, 0.5, 2.35, 5.2):
            nav = self.navigator()
            for tick in range(round(phase_time / 0.05) + 1):
                nav.step(tick * 0.05)
            nav.stop()
            cmd = nav.step(phase_time + 0.05)
            self.assertEqual(cmd.state, NavigationState.STOPPED)
            self.assertEqual((cmd.linear_v_mps, cmd.angular_omega_radps), (0, 0))

    def test_manual_settings_are_validated(self):
        for values in ({'manual_motion_enabled': 1}, {'manual_forward_pause_s': -0.5},
                       {'manual_midpoint_pause_s': -1}):
            with self.assertRaises(ValueError):
                NavigationPolicy(**values)


if __name__ == '__main__':
    unittest.main()
