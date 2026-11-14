"""Unit tests for coordinate frame transforms, math utilities, and camera extrinsics."""

import math
import unittest
import numpy as np

from hw3.coordinates import (
    CameraExtrinsics,
    euler_to_rotation_matrix,
    invert_transform,
    make_transform,
    quaternion_to_rotation_matrix,
    wrap_angle,
)


class TestCoordinates(unittest.TestCase):
    def test_wrap_angle(self):
        """Test wrapping angles to [-pi, pi)."""
        self.assertAlmostEqual(wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle(math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(-math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(2.0 * math.pi + 0.5), 0.5)
        self.assertAlmostEqual(wrap_angle(-2.0 * math.pi - 0.5), -0.5)

    def test_transform_inversion(self):
        """Test inverting rigid 4x4 transform."""
        r = euler_to_rotation_matrix(0.1, 0.2, 0.3)
        t = make_transform(r, [1.0, 2.0, 3.0])
        inv_t = invert_transform(t)
        eye = t @ inv_t
        np.testing.assert_allclose(eye, np.eye(4), atol=1e-12)

    def test_quaternion_conversion(self):
        """Test converting quaternion to rotation matrix.

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # 90 deg rotation around Z
        qz = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
        r = quaternion_to_rotation_matrix(*qz)
        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        np.testing.assert_allclose(r, expected, atol=1e-7)

    def test_camera_extrinsics_optical_to_base(self):
        """Test camera optical frame to robot base frame mapping.

        In camera frame:
        +Z is forward (robot +X)
        +X is right (robot -Y)
        +Y is down (robot -Z)

        Args:
            None.

        Returns:
            None. Assertions verify the documented behavior.
        """
        # Mount with zero roll/pitch/yaw and mount height z=0.2m
        cam = CameraExtrinsics(x_m=0.0, y_m=0.0, z_m=0.2, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)

        # Point 2 meters straight ahead of camera: [0, 0, 2] in camera frame
        t_cam_tag = make_transform(np.eye(3), [0.0, 0.0, 2.0])
        x_rel, y_rel, yaw_rel = cam.landmark_in_base_2d(t_cam_tag)

        self.assertAlmostEqual(x_rel, 2.0, places=4)
        self.assertAlmostEqual(y_rel, 0.0, places=4)

        # Point 1 meter to the right of camera: [1, 0, 2] in camera frame
        # In robot base frame: right is -Y
        t_cam_tag_right = make_transform(np.eye(3), [1.0, 0.0, 2.0])
        x_rel, y_rel, _ = cam.landmark_in_base_2d(t_cam_tag_right)
        self.assertAlmostEqual(x_rel, 2.0, places=4)
        self.assertAlmostEqual(y_rel, -1.0, places=4)


if __name__ == '__main__':
    unittest.main()
