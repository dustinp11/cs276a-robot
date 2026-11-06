"""Numerical checks of live geometry, timestamp handling, and recorded-map export."""

import json
import math
from pathlib import Path
import tempfile
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
from hw3_code.waypoint_control import WaypointController, square_waypoints


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

    def test_motor_mapping_and_nominal_turn_rate(self):
        """Match requested mean turn rate arithmetically without claiming physical response."""
        model = MotorModel(MOTOR_DEFAULTS)
        left, right, linear, angular = model.map_command(.15, 0.)
        self.assertEqual(left, right)
        self.assertAlmostEqual(linear, .15)
        self.assertEqual(angular, 0.)
        turns = [model.map_command(0., .4) for _ in range(1000)]
        mean_rate = sum(item[3] for item in turns)/len(turns)
        self.assertLess(abs(mean_rate-.4), MOTOR_DEFAULTS['turn_rate_ccw']/1000.)
        self.assertTrue(all(abs(value) <= .5 for item in turns for value in item[:2]))
        self.assertEqual(model.map_command(0., 0.), (0., 0., 0., 0.))
        self.assertEqual(model.map_command(float('nan'), 0.), (0., 0., 0., 0.))
        self.assertEqual(model.map_command(.1, .1), (0., 0., 0., 0.))

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
