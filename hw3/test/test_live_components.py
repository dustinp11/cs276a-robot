"""Numerical checks of live geometry, timestamp handling, and recorded-map export."""

import json
import math
from pathlib import Path
import tempfile
from urllib.error import HTTPError
from urllib.request import urlopen
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from numpy.testing import assert_allclose

from hw3_code.ekf_slam import EkfSlam
from hw3_code.motion import MotionHistory
from hw3_code.motor_model import MOTOR_DEFAULTS, MotorModel
from hw3_code.observations import ObservationAdapter, tag_corners
from hw3_code.plot_run import compare_landmarks, export
from hw3_code.recording import RunRecorder
from hw3_code.settings import DEFAULTS, validate_parameters
from hw3_code.web_view import BrowserMap
from hw3_code.waypoint_control import ControlSettings, FixedSquareController, WaypointController, square_waypoints


class LiveComponentTests(unittest.TestCase):
    """Check required live interfaces without launching an isolated route simulator."""

    def test_raw_pnp_and_robot_axes(self):
        """Recover metric range/bearing from distorted pixel corners and camera extrinsics."""
        matrix = np.array([[620., 0., 640.], [0., 620., 360.], [0., 0., 1.]])
        distortion = np.array([-0.3, 0.1, 0., 0., 0.])
        points, _ = cv2.projectPoints(tag_corners(.172), np.array([0.2, -0.1, 0.]),
                                     np.array([0.3, 0.1, 2.]), matrix, distortion)
        adapter = ObservationAdapter(.172, [0.05, 0., .2], [0., 0., 0.], .03, .02, .025, 3.)
        adapter.set_camera(matrix, distortion, 'camera_frame')
        distance, bearing, covariance, error = adapter.convert(points.reshape(4, 2))
        self.assertAlmostEqual(distance, math.hypot(2.05, -.3), places=5)
        self.assertAlmostEqual(bearing, math.atan2(-.3, 2.05), places=5)
        self.assertLess(error, 1e-5)
        self.assertGreater(np.linalg.eigvalsh(covariance)[0], 0.)

    def test_motion_history_expiry_and_boundaries(self):
        """Integrate actual command intervals and stop extrapolating after driver expiry."""
        history = MotionHistory(.15, .2, .002, .005, timeout=.15)
        history.add(1., 1., 0.)
        history.add(1.1, 2., 0.)
        slam = EkfSlam()
        history.propagate(slam, 1., 1.4)
        assert_allclose(slam.robot_pose, [.4, 0., 0.], atol=1e-10)
        self.assertFalse(history.add(1.05, 1., 0.))
        history.prune(1.2)
        self.assertEqual(len(history.samples), 1)
        self.assertGreater(slam.covariance[0, 0], 0.)

    def test_predicting_current_pose_preserves_posterior(self):
        """Control-time lookahead must not advance or mutate the delayed posterior."""
        import copy
        history = MotionHistory(.15, .2, .002, .005)
        history.add(1., 1., 0.)
        posterior = EkfSlam()
        predicted = copy.deepcopy(posterior)
        history.propagate(predicted, 1., 1.1)
        assert_allclose(posterior.robot_pose, [0., 0., 0.])
        assert_allclose(predicted.robot_pose, [.1, 0., 0.])

    def test_square_feet_conversion_and_closed_route(self):
        """Seven-foot vertices close in the start frame and repeat without a known map."""
        goals = square_waypoints(7., 2)
        self.assertEqual(len(goals), 8)
        assert_allclose(goals[0], [2.1336, 0., math.pi/2.])
        assert_allclose(goals[1][:2], [2.1336, 2.1336])
        assert_allclose(goals[3], [0., 0., 0.])
        assert_allclose(goals[:4], goals[4:])

    def test_controller_stops_without_feedback(self):
        """A live caller with missing/stale feedback receives a zero command."""
        controller = WaypointController(square_waypoints())
        self.assertEqual(controller.step(None), (0., 0.))
        self.assertEqual(controller.step([0., 0., 0.], pose_age=1.), (0., 0.))

    def test_recording_and_svg_export(self):
        """Recorded full covariance produces a valid map, trajectory CSV, and world errors."""
        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder(directory, {'tag_size_m': .172})
            slam = EkfSlam()
            slam.observe(('41h12', 9), 2., 0., np.diag([.01, .001]))
            recorder.snapshot(slam, 1., lap=0, event='periodic')
            recorder.snapshot(slam, 2., lap=1, event='lap_complete')
            recorder.close()
            truth = {'start_pose': [1., 2., math.pi/2.],
                     'landmarks': [{'id': 9, 'x_m': 1., 'y_m': 4.}]}
            output = Path(directory)/'map.svg'
            export(recorder.directory, output, truth, lap=1)
            self.assertEqual(ET.parse(output).getroot().tag, '{http://www.w3.org/2000/svg}svg')
            errors = json.loads(output.with_suffix('.errors.json').read_text())
            self.assertLess(errors['mean_landmark_error_m'], 1e-10)
            self.assertTrue(output.with_suffix('.csv').exists())

    def test_unobserved_landmarks_not_hidden_in_metric(self):
        """Missing and spurious IDs are reported separately from matched landmark error."""
        result = compare_landmarks([(('41h12', 1), np.array([0., 0.]), np.eye(2))],
                                   {'landmarks': [{'id': 2, 'x_m': 0., 'y_m': 0.}]})
        self.assertEqual(result['matched_count'], 0)
        self.assertIsNone(result['mean_landmark_error_m'])
        self.assertEqual(result['missing_ids'], [['41h12', 2]])
        self.assertEqual(result['spurious_ids'], [['41h12', 1]])

    def test_our_mapping_and_adjustable_turn_input(self):
        """Preserve our mapping with the confirmed 0.5 turning input."""
        model = MotorModel(MOTOR_DEFAULTS)
        left, right, linear, angular = model.map_command(.15, 0.)
        self.assertAlmostEqual(left, .08+.15/1.772915625)
        self.assertEqual(left, right)
        self.assertAlmostEqual(linear, .15)
        self.assertEqual(angular, 0.)
        self.assertEqual(ControlSettings().max_speed, .567333)
        self.assertAlmostEqual(model.map_command(.35, 0.)[2], .35)
        faster = model.map_command(ControlSettings().max_speed, 0.)
        self.assertAlmostEqual(faster[0], .4)
        self.assertAlmostEqual(faster[1], .4)
        self.assertAlmostEqual(faster[2], .567333)
        for requested in (.1, .8):
            left, right, linear, angular = model.map_command(0., requested)
            self.assertAlmostEqual(left, -(.5+.127/2.*requested/1e6))
            self.assertAlmostEqual(right, .5+.127/2.*requested/1e6)
            self.assertEqual(linear, 0.)
            self.assertEqual(angular, MOTOR_DEFAULTS['turn_rate_ccw'])
        self.assertLess(model.map_command(0., -.8)[3], 0.)
        self.assertEqual(model.map_command(0., 0.), (0., 0., 0., 0.))
        self.assertEqual(model.map_command(float('nan'), 0.), (0., 0., 0., 0.))

    def test_fixed_square_drives_without_landmark_input(self):
        """Complete four legs only after each required quarter-turn has been commanded."""
        controller = FixedSquareController()
        progress = np.zeros(2)
        for number in range(4):
            self.assertEqual(controller.step(progress), (controller.settings.max_speed, 0.))
            progress[0] += 2.1336
            self.assertEqual(controller.step(progress), (0., 0.))
            self.assertEqual(controller.step(progress), (0., controller.settings.rotation_speed))
            progress[1] += math.pi/2.
            self.assertEqual(controller.step(progress), (0., 0.))
            self.assertEqual(controller.index, number+1)
        self.assertTrue(controller.done)
        self.assertEqual(controller.step(progress), (0., 0.))

    def test_turn_cannot_be_skipped_after_first_leg(self):
        """The first pause must lead to sustained left commands, never immediate forward."""
        controller = FixedSquareController()
        controller.step([0., 0.])
        self.assertEqual(controller.step([2.1336, 0.]), (0., 0.))
        for turn in (0., .05, .4, 1.0, math.pi/2.-.001):
            self.assertEqual(controller.step([2.1336, turn]), (0., controller.settings.rotation_speed))
            self.assertEqual(controller.index, 0)
        self.assertEqual(controller.step([2.1336, math.pi/2.]), (0., 0.))
        self.assertEqual(controller.step([2.1336, math.pi/2.]), (controller.settings.max_speed, 0.))

    def test_route_motion_tracks_turn_commands_and_expiry(self):
        """Only the duration of commanded angular motion counts toward turn completion."""
        history = MotionHistory(.15, .2, .002, .005, timeout=.15)
        rate = 2*math.pi/3.75
        history.add(1., 0., rate)
        history.add(1.1, 0., rate)
        history.add(1.2, 0., 0.)
        distance, angle = history.totals(1., 2.)
        self.assertAlmostEqual(distance, 0.)
        self.assertAlmostEqual(angle, rate*.2)
        self.assertLess(angle, math.pi/2.)

    def test_browser_snapshot_without_network(self):
        """The viewer exposes estimator poses/covariances and does not invent landmarks."""
        import threading
        viewer = BrowserMap.__new__(BrowserMap)
        viewer.lock = threading.Lock()
        viewer.data = {'trajectory': []}
        slam = EkfSlam()
        viewer.update(slam, 1., 'drive', 0, None, 0)
        self.assertEqual(viewer.data['landmarks'], [])
        self.assertEqual(viewer.data['pose'], [0., 0., 0.])
        slam.observe(('41h12', 9), 2., 0., np.diag([.01, .001]))
        viewer.update(slam, 2., 'turn_left', 0, .1, 0)
        self.assertEqual(viewer.data['landmarks'][0]['id'], '41h12:9')
        self.assertEqual(len(viewer.data['trajectory']), 2)
        json.dumps(viewer.data, allow_nan=False)

    def test_browser_map_http_and_state(self):
        """Serve only the map page and real estimator snapshot, with no file access."""
        try:
            viewer = BrowserMap('127.0.0.1', 0)
        except PermissionError:
            self.skipTest('This environment does not allow local socket listeners')
        address = f'http://127.0.0.1:{viewer.server.server_port}'
        try:
            slam = EkfSlam()
            viewer.update(slam, 1., 'drive', 0, None, 0)
            with urlopen(address+'/state.json', timeout=2.) as response:
                data = json.load(response)
            self.assertEqual(data['landmarks'], [])
            self.assertEqual(data['pose'], [0., 0., 0.])
            self.assertEqual(data['status'], 'drive')
            with urlopen(address, timeout=2.) as response:
                self.assertIn('HW3 live robot map', response.read().decode())
            with self.assertRaises(HTTPError) as error:
                urlopen(address+'/etc/passwd', timeout=2.)
            self.assertEqual(error.exception.code, 404)
        finally:
            viewer.close()

    def test_parameter_validation(self):
        """Reject invalid route/noise settings and permit zero corrected-bit tolerance."""
        values = dict(DEFAULTS)
        values['max_hamming'] = 0
        validate_parameters(values)
        values['square_side_ft'] = -1.
        with self.assertRaises(ValueError):
            validate_parameters(values)


if __name__ == '__main__':
    unittest.main()
