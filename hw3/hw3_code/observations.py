"""Convert raw-image AprilTag corners to metric planar robot observations."""

import math

import cv2
import numpy as np


def camera_mount(xyz, rpy_deg):
    """Return base-from-optical rotation and translation; mount RPY uses body axes."""
    roll, pitch, yaw = np.radians(rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rx = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    return rz @ ry @ rx @ optical, np.asarray(xyz, dtype=float)


def tag_corners(size):
    """Return corners in the exact ordering used by the existing apriltag_ros PnP code."""
    half = size / 2.
    return np.array([[-half, -half, 0.], [half, -half, 0.],
                     [half, half, 0.], [-half, half, 0.]], dtype=float)


class ObservationAdapter:
    """Use raw K/D for PnP, then transform tag centers into the planar robot frame."""

    def __init__(self, size, xyz, rpy, range_std, range_scale, bearing_std, max_error):
        """Store measured geometry and provisional range/bearing uncertainty settings."""
        if not math.isfinite(size) or size <= 0:
            raise ValueError('tag_size_m must be positive')
        self.objects = tag_corners(size)
        self.rotation, self.translation = camera_mount(xyz, rpy)
        self.range_std, self.range_scale = range_std, range_scale
        self.bearing_std, self.max_error = bearing_std, max_error
        self.matrix, self.distortion, self.frame = None, None, None

    def set_camera(self, matrix, distortion, frame):
        """Copy the intrinsics of the raw image; reject invalid or missing calibration."""
        matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
        distortion = np.asarray(distortion, dtype=float)
        if not np.isfinite(matrix).all() or matrix[0, 0] <= 1 or matrix[1, 1] <= 1:
            raise ValueError('Invalid raw camera intrinsics')
        if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError('Expected calibrated OpenCV pinhole distortion coefficients')
        self.matrix, self.distortion, self.frame = matrix.copy(), distortion.copy(), frame

    def convert(self, corners):
        """Return robot range, bearing, 2x2 covariance, and pixel reprojection RMS.

        Args:
            corners: Four raw-image pixel corners in detector order.
        """
        if self.matrix is None:
            raise ValueError('CameraInfo has not arrived')
        points = np.asarray(corners, dtype=float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError('Expected four finite image corners')
        try:
            ok, rvec, tvec = cv2.solvePnP(self.objects, points, self.matrix, self.distortion)
        except cv2.error as error:
            raise ValueError(f'PnP failed: {error}') from error
        if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
            raise ValueError('Invalid or behind-camera PnP pose')
        return self._measurement(points, rvec, tvec)

    def _measurement(self, points, rvec, tvec):
        """Reject poor projections and express the tag center relative to the robot."""
        projected, _ = cv2.projectPoints(self.objects, rvec, tvec, self.matrix, self.distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - points)**2, axis=1))))
        if not math.isfinite(error) or error > self.max_error:
            raise ValueError('Tag reprojection error exceeds configured pixel threshold')
        position = self.rotation @ tvec.ravel() + self.translation
        distance = math.hypot(position[0], position[1])
        if distance <= 0.05:
            raise ValueError('Tag is too close to the robot reference point')
        variance = (self.range_std + self.range_scale * distance**2)**2
        return distance, math.atan2(position[1], position[0]), np.diag([variance, self.bearing_std**2]), error
