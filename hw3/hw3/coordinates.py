"""Coordinate frame transformations and geometric utilities for HW3.

Standard conventions used:
- Robot base frame (base_link):
    +X forward, +Y left, +Z up. Yaw is positive counter-clockwise around +Z.
- Camera optical frame (camera_frame):
    +X right, +Y down, +Z forward along optical axis.
- Landmark frame:
    AprilTag coordinate frame as estimated by PnP.
"""

import math
from typing import Tuple, Union
import numpy as np


def wrap_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi).

    Args:
        angle: Input angle in radians.

    Returns:
        Normalized angle in [-pi, pi).
    """
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from Z-Y-X Euler angles (radians)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3, dtype=float)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def make_transform(rotation_matrix: np.ndarray, translation: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix."""
    t = np.eye(4, dtype=float)
    t[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    t[:3, 3] = np.asarray(translation, dtype=float).flatten()[:3]
    return t


def invert_transform(t: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transformation matrix."""
    inv = np.eye(4, dtype=float)
    r = t[:3, :3]
    p = t[:3, 3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ p
    return inv


class CameraExtrinsics:
    """Represents the rigid mount transform of the camera on the robot base.

    Converts camera optical coordinates (+Z forward, +X right, +Y down)
    into robot base coordinates (+X forward, +Y left, +Z up).
    """

    def __init__(
        self,
        x_m: float = 0.0,
        y_m: float = 0.0,
        z_m: float = 0.2,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ):
        self.xyz = np.array([x_m, y_m, z_m], dtype=float)
        self.rpy_rad = np.radians([roll_deg, pitch_deg, yaw_deg])

        # Rotation mapping optical coordinates to body coordinates:
        # X_body = Z_opt (forward)
        # Y_body = -X_opt (left)
        # Z_body = -Y_opt (up)
        self.optical_to_body = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)

        r_mount = euler_to_rotation_matrix(*self.rpy_rad)
        self.r_base_cam = r_mount @ self.optical_to_body
        self.t_base_cam = make_transform(self.r_base_cam, self.xyz)
        self.t_cam_base = invert_transform(self.t_base_cam)

    def camera_to_base_transform(self, t_cam_landmark: np.ndarray) -> np.ndarray:
        """Transform a landmark pose from camera optical frame to robot base frame.

        T_base_landmark = T_base_camera @ T_camera_landmark
        """
        return self.t_base_cam @ t_cam_landmark

    def landmark_in_base_2d(self, t_cam_landmark: np.ndarray) -> Tuple[float, float, float]:
        """Extract 2D (x, y, yaw) of landmark in robot base frame.

        Returns:
            (x_rel, y_rel, yaw_rel) in robot frame (meters, meters, radians).
        """
        t_base_l = self.camera_to_base_transform(t_cam_landmark)
        x_rel = float(t_base_l[0, 3])
        y_rel = float(t_base_l[1, 3])
        yaw_rel = float(math.atan2(t_base_l[1, 0], t_base_l[0, 0]))
        return x_rel, y_rel, yaw_rel

    def landmark_range_bearing(self, t_cam_landmark: np.ndarray) -> Tuple[float, float]:
        """Compute planar range and bearing in robot base frame.

        Returns:
            (range_m, bearing_rad)
        """
        x_rel, y_rel, _ = self.landmark_in_base_2d(t_cam_landmark)
        rng = float(math.hypot(x_rel, y_rel))
        bearing = float(math.atan2(y_rel, x_rel))
        return rng, bearing
