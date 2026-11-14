"""Comprehensive Synthetic Unit Tests for EKF-SLAM (Step 4.3).

Explicitly covers the 10 required synthetic scenarios:
1. Robot motion with a single stationary landmark.
2. Robot motion with multiple stationary landmarks.
3. Initializing a new landmark.
4. Observing an existing landmark multiple times.
5. A landmark disappearing and reappearing.
6. Multiple landmarks being discovered dynamically.
7. Angle wrapping near -pi and pi.
8. No landmarks being visible.
9. Motion estimation with noisy observations.
10. Covariance symmetry and positive semidefiniteness.
"""

import math
import unittest
import numpy as np

from hw3.coordinates import wrap_angle
from hw3.ekf_slam import EkfSlam


class TestEkfSlamSynthetic(unittest.TestCase):
    def setUp(self):
        self.ekf = EkfSlam(
            initial_robot_pose=(0.0, 0.0, 0.0),
            initial_robot_sigma=(1e-4, 1e-4, 1e-4),
            vo_process_noise=(0.02, 0.02, 0.02),
            cmd_process_noise=(0.1, 0.05, 0.1),
            base_measurement_noise=(0.03, 0.03),
        )

    def test_01_robot_motion_single_stationary_landmark(self):
        """1. Robot motion with a single stationary landmark.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Stationary landmark in world at (2.0, 0.0)
        # Robot starts at (0.0, 0.0, 0.0)
        # First observation: landmark at relative x=2.0, y=0.0
        self.ekf.update_landmark(tag_id=0, x_rel=2.0, y_rel=0.0)

        # Move robot forward by 0.5m in 5 steps of 0.1m
        true_x = 0.0
        for _ in range(5):
            dx_step = 0.1
            true_x += dx_step
            # Prediction step
            self.ekf.predict_visual_odometry(dx=dx_step, dy=0.0, dtheta=0.0)
            # Measurement in robot frame: relative x = 2.0 - true_x, y = 0
            rel_x = 2.0 - true_x
            self.ekf.update_landmark(tag_id=0, x_rel=rel_x, y_rel=0.0)

        pose = self.ekf.get_robot_pose()
        lm = self.ekf.get_landmark_pose(tag_id=0)

        self.assertAlmostEqual(pose.x, 0.5, delta=0.02)
        self.assertAlmostEqual(pose.y, 0.0, delta=0.02)
        self.assertAlmostEqual(pose.theta, 0.0, delta=0.02)
        self.assertAlmostEqual(lm.x, 2.0, delta=0.02)
        self.assertAlmostEqual(lm.y, 0.0, delta=0.02)

    def test_02_robot_motion_multiple_stationary_landmarks(self):
        """2. Robot motion with multiple stationary landmarks.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Two stationary landmarks in world:
        # Tag 0 at (2.0, 0.5), Tag 1 at (2.0, -0.5)
        self.ekf.update_landmark(tag_id=0, x_rel=2.0, y_rel=0.5)
        self.ekf.update_landmark(tag_id=1, x_rel=2.0, y_rel=-0.5)

        # Move robot forward 1.0m and turn left 45 deg
        num_steps = 10
        dx_step = 0.1
        dtheta_step = math.radians(4.5)

        true_x, true_y, true_theta = 0.0, 0.0, 0.0
        for _ in range(num_steps):
            true_x += dx_step * math.cos(true_theta)
            true_y += dx_step * math.sin(true_theta)
            true_theta = wrap_angle(true_theta + dtheta_step)

            self.ekf.predict_visual_odometry(dx=dx_step, dy=0.0, dtheta=dtheta_step)

            # Observations in new robot frame
            for tag_id, (gx, gy) in [(0, (2.0, 0.5)), (1, (2.0, -0.5))]:
                dx = gx - true_x
                dy = gy - true_y
                # Rotate into robot frame
                c, s = math.cos(true_theta), math.sin(true_theta)
                rx = dx * c + dy * s
                ry = -dx * s + dy * c
                self.ekf.update_landmark(tag_id=tag_id, x_rel=rx, y_rel=ry)

        pose = self.ekf.get_robot_pose()
        self.assertAlmostEqual(pose.x, true_x, delta=0.05)
        self.assertAlmostEqual(pose.y, true_y, delta=0.05)
        self.assertAlmostEqual(pose.theta, true_theta, delta=0.05)

    def test_03_initializing_new_landmark(self):
        """3. Initializing a new landmark correctly augments state and covariance.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        self.assertEqual(self.ekf.num_landmarks, 0)
        self.assertEqual(self.ekf.state_dim, 3)

        # Robot at origin; observe new tag 4 at relative x=1.5, y=0.5
        success = self.ekf.update_landmark(tag_id=4, x_rel=1.5, y_rel=0.5)
        self.assertTrue(success)
        self.assertEqual(self.ekf.num_landmarks, 1)
        self.assertEqual(self.ekf.state_dim, 5)

        lm = self.ekf.get_landmark_pose(tag_id=4)
        self.assertIsNotNone(lm)
        self.assertAlmostEqual(lm.x, 1.5, places=5)
        self.assertAlmostEqual(lm.y, 0.5, places=5)

        # Landmark covariance must be non-zero and positive-definite
        self.assertGreater(lm.covariance[0, 0], 0.0)
        self.assertGreater(lm.covariance[1, 1], 0.0)
        # Check cross-covariance exists in P
        self.assertFalse(np.allclose(self.ekf.P[:3, 3:5], 0.0))

    def test_04_observing_existing_landmark_multiple_times(self):
        """4. Observing an existing landmark multiple times reduces landmark covariance.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Robot stationary at origin
        self.ekf.update_landmark(tag_id=2, x_rel=2.0, y_rel=0.0)
        init_cov = self.ekf.get_landmark_pose(tag_id=2).covariance.copy()

        # Perform 10 consecutive measurements of the same landmark without motion
        for _ in range(10):
            self.ekf.update_landmark(tag_id=2, x_rel=2.0, y_rel=0.0)

        final_cov = self.ekf.get_landmark_pose(tag_id=2).covariance
        # Trace of covariance should strictly decrease due to Kalman updates
        self.assertLess(np.trace(final_cov), np.trace(init_cov))

    def test_05_landmark_disappearing_and_reappearing(self):
        """5. A landmark disappearing and reappearing retains position without duplicates.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Discover Tag 7
        self.ekf.update_landmark(tag_id=7, x_rel=1.0, y_rel=1.0)
        self.assertEqual(self.ekf.num_landmarks, 1)
        lm_init = self.ekf.get_landmark_pose(tag_id=7)

        # Robot moves for 5 steps during which Tag 7 is NOT visible
        for _ in range(5):
            self.ekf.predict_visual_odometry(dx=0.1, dy=0.0, dtheta=0.0)

        # Tag 7 position in map must remain intact
        lm_mid = self.ekf.get_landmark_pose(tag_id=7)
        self.assertAlmostEqual(lm_mid.x, lm_init.x, places=5)
        self.assertAlmostEqual(lm_mid.y, lm_init.y, places=5)

        # Tag 7 reappears at relative position from robot (robot is at x=0.5)
        # Global landmark is at (1.0, 1.0), so relative to robot is (0.5, 1.0)
        self.ekf.update_landmark(tag_id=7, x_rel=0.5, y_rel=1.0)

        # Still only 1 landmark in filter (no duplicate)
        self.assertEqual(self.ekf.num_landmarks, 1)
        self.assertIn(7, self.ekf.idx_to_tag_id)

    def test_06_multiple_landmarks_discovered_dynamically(self):
        """6. Multiple landmarks being discovered dynamically expands the state vector.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        self.assertEqual(self.ekf.num_landmarks, 0)

        # Discover Tag 1
        self.ekf.update_landmark(tag_id=1, x_rel=1.0, y_rel=0.0)
        self.assertEqual(self.ekf.num_landmarks, 1)
        self.assertEqual(self.ekf.state_dim, 5)

        # Move forward
        self.ekf.predict_visual_odometry(dx=1.0, dy=0.0, dtheta=0.0)

        # Discover Tag 2
        self.ekf.update_landmark(tag_id=2, x_rel=1.0, y_rel=0.5)
        self.assertEqual(self.ekf.num_landmarks, 2)
        self.assertEqual(self.ekf.state_dim, 7)

        # Turn and discover Tag 3
        self.ekf.predict_visual_odometry(dx=0.0, dy=0.0, dtheta=math.pi / 2.0)
        self.ekf.update_landmark(tag_id=3, x_rel=1.0, y_rel=0.0)
        self.assertEqual(self.ekf.num_landmarks, 3)
        self.assertEqual(self.ekf.state_dim, 9)

        # All landmarks accessible
        lms = self.ekf.get_all_landmarks()
        self.assertEqual(set(lms.keys()), {1, 2, 3})

    def test_07_angle_wrapping_near_boundary(self):
        """7. Angle wrapping near -pi and pi handled correctly without singularity.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Initialize robot near +pi
        ekf_wrap = EkfSlam(
            initial_robot_pose=(0.0, 0.0, math.pi - 0.05),
            initial_robot_sigma=(1e-3, 1e-3, 1e-3),
        )

        # Rotate by +0.1 rad, which crosses pi into negative domain (-pi + 0.05)
        ekf_wrap.predict_visual_odometry(dx=0.0, dy=0.0, dtheta=0.10)
        pose = ekf_wrap.get_robot_pose()

        expected_theta = wrap_angle(math.pi - 0.05 + 0.10)
        self.assertAlmostEqual(pose.theta, expected_theta, places=5)
        self.assertTrue(-math.pi <= pose.theta < math.pi)

        # Now test measurement update crossing pi boundary
        # Landmark at (-2.0, 0.0), global angle is pi
        ekf_wrap.update_landmark(tag_id=9, x_rel=2.0, y_rel=0.0)
        self.assertTrue(-math.pi <= ekf_wrap.get_robot_pose().theta < math.pi)

    def test_08_no_landmarks_visible(self):
        """8. No landmarks visible propagates uncertainty without error.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Multiple steps of prediction without any measurement updates
        init_cov = self.ekf.get_robot_pose().covariance.copy()

        for _ in range(10):
            self.ekf.predict_visual_odometry(dx=0.1, dy=0.0, dtheta=0.05)

        final_cov = self.ekf.get_robot_pose().covariance
        # Pose uncertainty must grow monotonically in absence of measurements
        self.assertGreater(final_cov[0, 0], init_cov[0, 0])
        self.assertGreater(final_cov[1, 1], init_cov[1, 1])
        self.assertGreater(final_cov[2, 2], init_cov[2, 2])

    def test_09_motion_estimation_with_noisy_observations(self):
        """9. Motion estimation with noisy observations converges close to true state.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        np.random.seed(42)
        # Known landmark at (2.0, 0.0)
        self.ekf.update_landmark(tag_id=0, x_rel=2.0, y_rel=0.0)

        true_x = 0.0
        for _ in range(20):
            dx_true = 0.05
            true_x += dx_true
            # Noisy motion input (std = 0.01)
            noisy_dx = dx_true + float(np.random.normal(0, 0.005))
            self.ekf.predict_visual_odometry(dx=noisy_dx, dy=0.0, dtheta=0.0)

            # Noisy measurement (std = 0.015)
            true_rel_x = 2.0 - true_x
            noisy_rel_x = true_rel_x + float(np.random.normal(0, 0.015))
            noisy_rel_y = float(np.random.normal(0, 0.015))
            self.ekf.update_landmark(tag_id=0, x_rel=noisy_rel_x, y_rel=noisy_rel_y)

        pose = self.ekf.get_robot_pose()
        # Filter should smooth noise and track ground truth within 3-sigma
        self.assertAlmostEqual(pose.x, true_x, delta=0.06)
        self.assertAlmostEqual(pose.y, 0.0, delta=0.06)

    def test_10_covariance_symmetry_and_positive_definiteness(self):
        """10. Covariance symmetry and positive semi-definiteness maintained throughout.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Initialize several landmarks and execute mixed predictions and updates
        self.ekf.update_landmark(tag_id=1, x_rel=1.5, y_rel=0.5)
        self.ekf.update_landmark(tag_id=2, x_rel=2.0, y_rel=-0.5)

        for i in range(15):
            self.ekf.predict_visual_odometry(dx=0.08, dy=0.01, dtheta=0.03)
            self.ekf.update_landmark(tag_id=1, x_rel=1.5 - 0.08 * i, y_rel=0.5)
            if i % 3 == 0:
                # Add another landmark
                self.ekf.update_landmark(tag_id=10 + i, x_rel=1.0, y_rel=1.0)

            # Check symmetry: P == P^T
            np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-10)

            # Check positive semi-definiteness: all eigenvalues >= 0
            eigenvalues = np.linalg.eigvals(self.ekf.P)
            self.assertTrue(np.all(eigenvalues.real >= -1e-9), f"Negative eigenvalues found: {eigenvalues}")

    def test_11_robot_completes_square_trajectory(self):
        """11. Robot manually executes a 1m x 1m square around 4 tags and returns to origin.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # 4 landmarks placed outside the square:
        # Tag 0 at (1.5, 0.5) [visible along Leg 1: (0,0) -> (1,0)]
        # Tag 1 at (0.5, 1.5) [visible along Leg 2: (1,0) -> (1,1)]
        # Tag 2 at (-0.5, 0.5) [visible along Leg 3: (1,1) -> (0,1)]
        # Tag 3 at (0.5, -0.5) [visible along Leg 4: (0,1) -> (0,0)]
        landmarks = {
            0: (1.5, 0.5),
            1: (0.5, 1.5),
            2: (-0.5, 0.5),
            3: (0.5, -0.5),
        }

        # Simulated square path: 4 legs of 1m forward, each followed by 90 deg CCW turn
        # 10 steps per leg (0.1m per step)
        true_x, true_y, true_theta = 0.0, 0.0, 0.0

        for leg in range(4):
            # Move forward 1 meter
            for _ in range(10):
                dx = 0.1
                true_x += dx * math.cos(true_theta)
                true_y += dx * math.sin(true_theta)
                self.ekf.predict_visual_odometry(dx=dx, dy=0.0, dtheta=0.0)

                # Observe landmarks within 2.5m
                for tag_id, (gx, gy) in landmarks.items():
                    dist = math.hypot(gx - true_x, gy - true_y)
                    if dist <= 2.2:
                        c, s = math.cos(true_theta), math.sin(true_theta)
                        rel_x = (gx - true_x) * c + (gy - true_y) * s
                        rel_y = -(gx - true_x) * s + (gy - true_y) * c
                        if rel_x > 0.1:  # In front of camera
                            self.ekf.update_landmark(tag_id=tag_id, x_rel=rel_x, y_rel=rel_y)

            # Turn 90 deg CCW
            dtheta = math.pi / 2.0
            true_theta = wrap_angle(true_theta + dtheta)
            self.ekf.predict_visual_odometry(dx=0.0, dy=0.0, dtheta=dtheta)

            for tag_id, (gx, gy) in landmarks.items():
                dist = math.hypot(gx - true_x, gy - true_y)
                if dist <= 2.2:
                    c, s = math.cos(true_theta), math.sin(true_theta)
                    rel_x = (gx - true_x) * c + (gy - true_y) * s
                    rel_y = -(gx - true_x) * s + (gy - true_y) * c
                    if rel_x > 0.1:
                        self.ekf.update_landmark(tag_id=tag_id, x_rel=rel_x, y_rel=rel_y)

        # After full loop, robot is back at origin (0, 0, 0)
        pose = self.ekf.get_robot_pose()
        self.assertAlmostEqual(pose.x, 0.0, delta=0.08)
        self.assertAlmostEqual(pose.y, 0.0, delta=0.08)
        self.assertAlmostEqual(pose.theta, 0.0, delta=0.08)
        # All 4 landmarks should be mapped accurately
        for tag_id, (gx, gy) in landmarks.items():
            lm = self.ekf.get_landmark_pose(tag_id)
            self.assertIsNotNone(lm)
            self.assertAlmostEqual(lm.x, gx, delta=0.08)
            self.assertAlmostEqual(lm.y, gy, delta=0.08)

    def test_12_untracked_drift_heading_recovery(self):
        """12. Re-observation of a known landmark recovers heading after untracked drift.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Initialize landmark 1 at (2.0, 0.0) from origin facing 0
        self.ekf.update_landmark(tag_id=1, x_rel=2.0, y_rel=0.0)
        lm = self.ekf.get_landmark_pose(1)
        self.assertAlmostEqual(lm.x, 2.0, delta=0.05)
        self.assertAlmostEqual(lm.y, 0.0, delta=0.05)

        # Robot physically rotates 90 deg CCW without visual tracking (blind gap of 5s)
        for _ in range(25):
            self.ekf.predict_untracked_drift(dt=0.2)

        # Heading uncertainty should have grown
        self.assertGreater(math.degrees(math.sqrt(self.ekf.P[2, 2])), 15.0)

        # Re-observe landmark 1: in physical frame (+90 deg), landmark is at bearing -90 deg
        # (x_rel = 0, y_rel = -2.0)
        success = self.ekf.update_landmark(tag_id=1, x_rel=0.0, y_rel=-2.0)
        self.assertTrue(success)

        # Robot heading should recover towards +90 deg (+pi/2)
        pose = self.ekf.get_robot_pose()
        self.assertAlmostEqual(pose.theta, math.pi / 2.0, delta=0.20)
        # Heading uncertainty should collapse back down
        self.assertLess(math.degrees(math.sqrt(self.ekf.P[2, 2])), 15.0)


if __name__ == '__main__':
    unittest.main()
