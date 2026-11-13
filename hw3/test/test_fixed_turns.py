"""Regression checks for calibrated mapping, bounded pivots, and command timing."""

import ast
import math
import json
import threading
from pathlib import Path
from types import SimpleNamespace as Record
import unittest

import numpy as np

from hw3.turn_calibration import TurnCalibration
from hw3.velocity_mapping import VelocityMapper
from hw3.square_navigation import FixedTurnPulse, NavigationState, SquareNavigator


ROOT = Path(__file__).resolve().parents[2]


def load_method(path, class_name, method_name, namespace=None):
    """Load an actual ROS callback without importing unavailable ROS libraries.

    Args:
        path: Path to the CSV file being read.
        class_name: Name of the ROS adapter class containing the method under test.
        method_name: Name of the method to load without importing ROS.
        namespace: Optional globals supplied to the isolated method.

    Returns:
        The requested method as a callable for isolated regression checks.
    """
    tree = ast.parse((ROOT / path).read_text())
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == class_name)
    method = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == method_name)
    for arg in method.args.args:
        arg.annotation = None
    method.returns = None
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope[method_name]


class TestFixedTurns(unittest.TestCase):
    def setUp(self):
        self.calibration = TurnCalibration(0.3, 0.3, 0.6, 0.5)

    def test_direction_specific_fixed_rates_and_inputs(self):
        for request in (0.001, 0.2, 2.0):
            self.assertEqual(self.calibration.rate(request), 0.6)
            self.assertEqual(self.calibration.motors(request), (-0.3, 0.3))
            self.assertEqual(self.calibration.rate(-request), -0.5)
            self.assertEqual(self.calibration.motors(-request), (0.3, -0.3))
        self.assertEqual(self.calibration.motors(0), (0, 0))

    def test_measured_direction_specific_motor_inputs(self):
        calibration = TurnCalibration(0.43, 0.40, 2 * math.pi / 3.55, 2 * math.pi / 3.59)
        mapper = VelocityMapper(calibration)
        ccw, cw = mapper.map_command(0, 1), mapper.map_command(0, -1)
        self.assertEqual((ccw.left, ccw.right), (-0.43, 0.43))
        self.assertEqual((cw.left, cw.right), (0.40, -0.40))
        self.assertAlmostEqual(ccw.angular_radps * 3.55, 2 * math.pi)
        self.assertAlmostEqual(cw.angular_radps * 3.59, -2 * math.pi)

    def test_unmeasured_or_invalid_calibration_is_rejected(self):
        for values in [(0.3, 0.3, 0, 0), (0.6, 0.3, 0.5, 0.5),
                       (0.3, 0.6, 0.5, 0.5), (0.3, 0.3, -1, 1), (0.3, 0.3, math.nan, 1)]:
            with self.assertRaises(ValueError):
                TurnCalibration(*values)
        nav = SquareNavigator()
        self.assertFalse(nav.start()[0])

    def test_pulse_stops_and_settles_before_restarting(self):
        pulse = FixedTurnPulse(self.calibration, math.pi / 2, 0.3, 1.0)
        self.assertEqual(pulse.step(1.0, math.pi / 2, 0.1)[0], 0.6)
        self.assertEqual(pulse.step(1.5, math.pi / 2, 0.1)[0], 0)
        self.assertEqual(pulse.step(2.4, math.pi / 2, 0.1)[0], 0)
        self.assertEqual(pulse.step(2.5, math.pi / 2, 0.1)[0], 0.6)

    def test_frozen_heading_cannot_command_a_full_revolution(self):
        pulse = FixedTurnPulse(self.calibration, math.pi / 2, math.pi / 6, 0.1)
        for tick in range(300):
            pulse.step(1 + tick * 0.05, math.pi / 2, 0.1)
            if pulse.exhausted:
                break
        self.assertTrue(pulse.exhausted)
        self.assertLess(pulse.used, math.radians(125))

    def test_feedback_stops_before_pulse_deadline(self):
        pulse = FixedTurnPulse(self.calibration, 1.0, 0.5, 0.2)
        pulse.step(1.0, 1.0, 0.1)
        self.assertEqual(pulse.step(1.1, 0.01, 0.1)[0], 0)
        self.assertEqual(pulse.step(1.4, 0.01, 0.1)[0], 0)
        self.assertTrue(pulse.done)

    def test_square_completes_with_fixed_turns_and_straight_segments(self):
        """Verify square completes with fixed turns and straight segments.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        x = y = yaw = 0.0
        for tick in range(5000):
            now = 1 + tick * 0.05
            nav.update_ekf_pose(x, y, yaw, 0.01, 0.01, now)
            cmd = nav.step(now)
            self.assertFalse(cmd.linear_v_mps and cmd.angular_omega_radps)
            x += cmd.linear_v_mps * math.cos(yaw) * 0.05
            y += cmd.linear_v_mps * math.sin(yaw) * 0.05
            yaw += cmd.angular_omega_radps * 0.05
            if nav.state == NavigationState.COMPLETED:
                break
        self.assertEqual(nav.state, NavigationState.COMPLETED)
        self.assertLess(math.hypot(x, y), nav.waypoint_tolerance_m)

    def test_stale_pose_stops_a_running_pulse(self):
        nav = SquareNavigator(turn_calibration=self.calibration)
        nav.start()
        nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, 1.0)
        side = nav.square_size_m
        nav.update_ekf_pose(side, 0, 0, 0.01, 0.01, 1.1)
        nav.step(1.1)
        self.assertGreater(nav.step(1.15).angular_omega_radps, 0)
        self.assertEqual(nav.step(2.0).angular_omega_radps, 0)
        self.assertEqual(nav.state, NavigationState.PAUSED_STALE_POSE)

    def test_fresh_but_frozen_heading_stops_mission_and_cannot_auto_resume(self):
        """Verify fresh but frozen heading stops mission and cannot auto resume.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        nav = SquareNavigator(turn_calibration=self.calibration, turn_settle_s=0.1)
        nav.start()
        nav.update_ekf_pose(0, 0, 0, 0.01, 0.01, 1.0)
        for tick in range(400):
            now = 1.1 + tick * 0.05
            nav.update_ekf_pose(nav.square_size_m, 0, 0, 0.01, 0.01, now)
            result = nav.step(now)
            if nav.state == NavigationState.STOPPED:
                break
        self.assertEqual(nav.state, NavigationState.STOPPED)
        self.assertEqual(result.angular_omega_radps, 0)
        self.assertFalse(nav.start()[0])
        self.assertEqual(nav.step(now + 1).angular_omega_radps, 0)


class TestAppliedCommandTiming(unittest.TestCase):
    def test_stop_and_repeated_callbacks_integrate_each_interval_once(self):
        """Verify stop and repeated callbacks integrate each interval once.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        path = 'hw3/hw3/nodes/ekf_slam_node.py'
        method = load_method(path, 'EkfSlamNode', '_predict_command_until')
        received = []
        node = Record(last_prediction_time=1.0, last_cmd_time=1.0,
                      command_hold_s=0.15, command_prediction=True, last_cmd=(0, 0.6),
                      ekf=Record(predict_commanded_velocity=lambda *v: received.append(v)))
        method(node, 1.05)
        method(node, 1.05)
        method(node, 1.1)
        self.assertAlmostEqual(sum(v[2] for v in received), 0.1)
        node.last_cmd, node.last_cmd_time = (0, 0), 1.1
        method(node, 1.15)
        self.assertAlmostEqual(sum(v[1] * v[2] for v in received), 0.06)

    def test_silent_commands_stop_prediction_at_motor_watchdog(self):
        """Verify silent commands stop prediction at motor watchdog.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        method = load_method('hw3/hw3/nodes/ekf_slam_node.py', 'EkfSlamNode',
                             '_predict_command_until')
        received = []
        node = Record(last_prediction_time=1.0, last_cmd_time=1.0,
                      command_hold_s=0.15, command_prediction=True, last_cmd=(0, 0.6),
                      ekf=Record(predict_commanded_velocity=lambda *v: received.append(v)))
        method(node, 10.0)
        self.assertAlmostEqual(received[0][2], 0.15)
        method(node, 11.0)
        self.assertEqual(len(received), 1)


class TestFixedMapper(unittest.TestCase):
    def test_mapper_publishes_model_rate_and_exact_motor_inputs(self):
        """Verify mapper publishes model rate and exact motor inputs.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        motors, applied = [], []
        twist = lambda: Record(linear=Record(x=0.0), angular=Record(z=0.0))
        method = load_method('hw3/hw3/nodes/velocity_mapping.py', 'VelocityToMotorNode',
                             'on_command', {'Twist': twist, 'Float32MultiArray': Record})
        node = Record(mapper=VelocityMapper(TurnCalibration(0.3, 0.3, 0.6, 0.5)),
                      motor_pub=Record(publish=motors.append),
                      applied_pub=Record(publish=applied.append))
        for request in (0.01, -0.01, 0):
            message = twist()
            message.angular.z = request
            method(node, message)
        self.assertEqual([m.data for m in motors], [[-0.3, 0.3], [0.3, -0.3], [0, 0]])
        self.assertEqual([m.angular.z for m in applied], [0.6, -0.5, 0])

    def test_straight_motion_uses_existing_linear_calibration(self):
        mapper = VelocityMapper(TurnCalibration(0.3, 0.3, 0.6, 0.5))
        command = mapper.map_command(0.2, 0.7)
        self.assertAlmostEqual(command.left, 0.08 + 0.2 / 1.772915625)
        self.assertEqual(command.left, command.right)
        self.assertEqual(command.angular_radps, 0)
        self.assertAlmostEqual(command.linear_mps, 0.2)

    def test_saturation_and_nonfinite_commands(self):
        mapper = VelocityMapper(TurnCalibration(0.3, 0.3, 0.6, 0.5))
        command = mapper.map_command(10, 0)
        self.assertEqual(command.left, 0.5)
        self.assertAlmostEqual(command.linear_mps, (0.5 - 0.08) * 1.772915625)
        self.assertEqual(mapper.map_command(math.nan, 1).left, 0)
        self.assertEqual(mapper.map_command(1, math.inf).right, 0)


class TestHw3SerialDriver(unittest.TestCase):
    def test_motor_watchdog_expires_without_a_new_command(self):
        method = load_method('hw3/hw3/nodes/motor_control.py', 'MotorControllerNode',
                             '_current_command')
        node = Record(command_lock=threading.Lock(), timeout_duration=0.15,
                      last_command_time=1.0, motor_command={'T': 1, 'L': -0.3, 'R': 0.3})
        self.assertEqual(method(node, 1.1), node.motor_command)
        self.assertEqual(method(node, 1.2), {'T': 1, 'L': 0, 'R': 0})

    def test_serial_packet_retains_existing_protocol(self):
        method = load_method('hw3/hw3/nodes/motor_control.py', 'MotorControllerNode',
                             '_write', {'json': json})
        sent = []
        method(Record(robot=Record(write=sent.append)), {'T': 1, 'L': -0.3, 'R': 0.3})
        self.assertTrue(sent[0].endswith(b'\n'))
        self.assertEqual(json.loads(sent[0]), {'T': 1, 'L': -0.3, 'R': 0.3})
