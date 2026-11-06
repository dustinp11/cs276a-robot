"""Hardware-free checks of EKF geometry, uncertainty, and landmark discovery."""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from hw3_code.ekf_slam import (
    EkfSlam, _inverse_observation, _motion_model, _observation_model, wrap_angle,
)


def numerical_jacobian(function, point):
    """Differentiate a smooth test function with centered finite differences."""
    columns = []
    for index in range(len(point)):
        offset = np.zeros(len(point))
        offset[index] = 1e-6
        columns.append((function(point + offset) - function(point - offset)) / 2e-6)
    return np.column_stack(columns)


class EkfSlamTests(unittest.TestCase):
    """Check estimator behavior without ROS, camera, or motor dependencies."""

    def setUp(self):
        """Use explicit synthetic covariances, not claimed hardware measurements."""
        self.noise = np.diag([0.03**2, math.radians(1.)**2])
        self.motion_noise = np.diag([0.01**2, math.radians(0.5)**2])

    def assert_valid_covariance(self, slam):
        """Check symmetry, finite entries, and positive semidefiniteness numerically."""
        covariance = slam.covariance
        self.assertTrue(np.isfinite(covariance).all())
        assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance)[0], -1e-10)

    def test_empty_start_and_unrestricted_discovery(self):
        """Discover more than eight IDs, without preloading a map or world start."""
        slam = EkfSlam()
        assert_allclose(slam.robot_pose, [0., 0., 0.])
        self.assertEqual(slam.landmark_ids, ())
        for index in range(12):
            self.assertEqual(slam.observe(('36h11', index), 2., index/10., self.noise), 'initialized')
        self.assertEqual(len(slam.state), 27)
        self.assertEqual(len(slam.landmark_ids), 12)
        self.assert_valid_covariance(slam)

    def test_straight_turn_and_reverse_prediction(self):
        """Predict simple motions and wrap orientation across the angle boundary."""
        slam = EkfSlam()
        slam.predict(1., 0., self.motion_noise)
        slam.predict(0., math.pi/2., self.motion_noise)
        slam.predict(-0.5, 0., self.motion_noise)
        assert_allclose(slam.robot_pose, [1., -0.5, math.pi/2.], atol=1e-12)
        slam.predict(0., math.pi, self.motion_noise)
        self.assertAlmostEqual(slam.robot_pose[2], -math.pi/2.)
        self.assert_valid_covariance(slam)

    def test_motion_jacobians(self):
        """Compare both motion derivatives with independently differenced poses."""
        pose, control = np.array([0.3, -0.1, 0.7]), np.array([0.2, 0.15])
        _, state_jacobian, input_jacobian = _motion_model(pose, *control)
        state_numeric = numerical_jacobian(lambda x: _motion_model(x, *control)[0], pose)
        input_numeric = numerical_jacobian(lambda u: _motion_model(pose, *u)[0], control)
        assert_allclose(state_jacobian, state_numeric, atol=1e-8)
        assert_allclose(input_jacobian, input_numeric, atol=1e-8)

    def test_observation_and_initialization_jacobians(self):
        """Check forward and inverse geometry derivatives and round-trip coordinates."""
        pose, measurement = np.array([0.3, -0.1, 0.7]), np.array([2., 0.4])
        position, robot_jacobian, measurement_jacobian = _inverse_observation(pose, measurement)
        state = np.concatenate([pose, position])
        predicted, jacobian = _observation_model(state, 3)
        assert_allclose(predicted, measurement, atol=1e-12)
        assert_allclose(jacobian, numerical_jacobian(lambda x: _observation_model(x, 3)[0], state), atol=1e-8)
        assert_allclose(robot_jacobian, numerical_jacobian(lambda x: _inverse_observation(x, measurement)[0], pose), atol=1e-8)
        assert_allclose(measurement_jacobian, numerical_jacobian(lambda z: _inverse_observation(pose, z)[0], measurement), atol=1e-8)

    def test_initialization_cross_covariances(self):
        """New landmarks share pose uncertainty and are not independent map entries."""
        initial = np.diag([0.04, 0.09, 0.01])
        slam = EkfSlam(initial)
        slam.observe(10, 2., 0., self.noise)
        assert_allclose(slam.landmark(10)[0], [2., 0.])
        assert_allclose(slam.covariance[:3, :3], initial)
        assert_allclose(slam.covariance[3:, :3], [[0.04, 0., 0.], [0., 0.09, 0.02]])
        slam.observe(23, 3., math.pi/2., self.noise)
        self.assertGreater(np.linalg.norm(slam.covariance[3:5, 5:7]), 0.)
        self.assert_valid_covariance(slam)

    def test_missing_and_reappearing_landmark(self):
        """Keep an unseen landmark fixed and reduce robot uncertainty on reobservation."""
        slam = EkfSlam()
        slam.observe(10, 2., 0., self.noise)
        position = slam.landmark(10)[0]
        slam.predict(0.5, 0., np.diag([0.04, 0.01]))
        prior_variance = slam.covariance[0, 0]
        assert_allclose(slam.landmark(10)[0], position)
        self.assertEqual(slam.observe(10, 1.5, 0., self.noise), 'updated')
        self.assertEqual(slam.landmark_ids, (10,))
        self.assertLess(slam.covariance[0, 0], prior_variance)
        self.assert_valid_covariance(slam)

    def test_gate_rejects_outlier_without_mutation(self):
        """An inconsistent repeat observation cannot overwrite the estimate."""
        slam = EkfSlam()
        slam.observe(10, 2., 0., self.noise)
        state, covariance = slam.state, slam.covariance
        self.assertEqual(slam.observe(10, 20., 1., self.noise), 'rejected')
        assert_allclose(slam.state, state)
        assert_allclose(slam.covariance, covariance)

    def test_bearing_wrap_at_pi(self):
        """Nearby bearings across the angle boundary remain a small innovation."""
        slam = EkfSlam()
        slam.observe(10, 2., math.pi - 0.001, self.noise)
        self.assertEqual(slam.observe(10, 2., -math.pi + 0.001, self.noise), 'updated')
        self.assertEqual(slam.landmark_ids, (10,))
        self.assert_valid_covariance(slam)

    def test_correlated_map_updates(self):
        """Seeing one landmark can correct another through joint correlations."""
        slam = EkfSlam()
        slam.observe(10, 2., 0., self.noise)
        slam.predict(0.3, 0., np.diag([0.01, 0.001]))
        slam.observe(20, 2., 0.5, self.noise)
        before = slam.landmark(20)[0]
        self.assertEqual(slam.observe(10, 1.67, 0., self.noise), 'updated')
        self.assertGreater(np.linalg.norm(slam.landmark(20)[0] - before), 1e-8)
        self.assert_valid_covariance(slam)

    def test_validation_and_snapshot_isolation(self):
        """Reject invalid uncertainty/input and protect state from external mutation."""
        slam = EkfSlam()
        with self.assertRaises(ValueError):
            slam.observe(1, -1., 0., self.noise)
        with self.assertRaises(ValueError):
            slam.observe(1, 1., 0., np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            slam.predict(float('nan'), 0., self.motion_noise)
        with self.assertRaises(ValueError):
            slam.predict(1., 0., np.diag([-1., 1.]))
        with self.assertRaises(ValueError):
            EkfSlam(gate_threshold=0.)
        state = slam.state
        state[:] = 100.
        assert_allclose(slam.robot_pose, [0., 0., 0.])
        self.assertEqual(slam.landmark_ids, ())


if __name__ == '__main__':
    unittest.main()
