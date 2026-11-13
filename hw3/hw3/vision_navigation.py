"""Fresh tag feedback and wall-plane geometry for navigation without a prior map."""

from dataclasses import dataclass, fields
import math
from statistics import median

import numpy as np

from .coordinates import wrap_angle


@dataclass(frozen=True)
class NavigationPolicy:
    """Limits in meters, radians, and seconds; clearances are from robot center."""

    vision_timeout_s: float = 1.0
    pose_wait_timeout_s: float = 8.0
    alignment_timeout_s: float = 18.0
    wall_stop_distance_m: float = 0.45
    drive_budget_factor: float = 1.25
    scan_timeout_s: float = 20.0
    scan_step_rad: float = math.pi / 4
    scan_pause_s: float = 0.5

    def __post_init__(self):
        """Reject invalid limits rather than silently disabling motion guards."""
        if not all(math.isfinite(getattr(self, f.name)) and getattr(self, f.name) > 0
                   for f in fields(self)):
            raise ValueError('Navigation vision limits must be finite and positive.')
        if self.drive_budget_factor < 1:
            raise ValueError('Drive budget factor must be at least one.')
        if self.scan_step_rad > math.pi / 2:
            raise ValueError('Scan stages must be at most 90 nominal degrees.')


class VisualFeedback:
    """Track optional accepted tag feedback and relative travel anchors."""

    def __init__(self, policy):
        """Initialize empty observations and no active leg anchors."""
        self.policy = policy
        self.stamp = self.valid_stamp = -math.inf
        self.clearance, self.clearance_stamp = None, -math.inf
        self.baselines = {}
        self.progress = None
        self.progress_stamp, self.progress_prediction = -math.inf, 0.0
        self.heading = None

    def begin_leg(self, heading):
        """Reset relative tag travel anchors for a fixed SLAM-frame heading."""
        self.heading = heading
        self.baselines.clear()
        self.progress, self.progress_stamp = None, -math.inf
        self.progress_prediction = 0.0

    def update(self, stamp, tags, known_ids, clearance, pose, predicted_progress):
        """Cache one distinct accepted frame and estimate travel from stationary tags.

        Args:
            stamp: Original image time in ROS seconds; duplicates are ignored.
            tags: Accepted base-frame [tag_id, x_m, y_m] relative positions.
            known_ids: IDs successfully updated that were mapped before this frame.
            clearance: Forward wall-plane intersection in meters, or None.
            pose: Current SLAM-frame (x_m, y_m, yaw_rad) estimate.
            predicted_progress: EKF forward displacement since this leg began, in meters.

        Returns:
            None. Updates freshness, stopped-pose history, and relative travel.
        """
        if stamp <= self.stamp:
            return
        self.stamp = stamp
        if tags:
            self.clearance, self.clearance_stamp = clearance, stamp
        if tags and known_ids:
            self.valid_stamp = stamp
        if self.heading is not None:
            self._update_progress(tags, known_ids, pose[2], predicted_progress)

    def _update_progress(self, tags, known_ids, yaw, predicted_progress):
        """Project same-tag displacement along the fixed leg heading.

        Args:
            tags: Accepted [tag_id, base_x_m, base_y_m] records.
            known_ids: Successfully reobserved map IDs in this frame.
            yaw: EKF robot heading in the SLAM frame, in radians.
            predicted_progress: EKF leg displacement used only to anchor a newly seen ID.

        Returns:
            None. Existing tag anchors provide the median relative travel in meters.
        """
        c, s = math.cos(yaw - self.heading), math.sin(yaw - self.heading)
        projected = {int(tag): c * x - s * y for tag, x, y in tags if tag in known_ids}
        estimates = [self.baselines[tag] - value for tag, value in projected.items()
                     if tag in self.baselines]
        if estimates:
            self.progress, self.progress_stamp = median(estimates), self.stamp
            self.progress_prediction = predicted_progress
        anchor = self.extrapolated_progress(predicted_progress)
        for tag, value in projected.items():
            self.baselines.setdefault(tag, anchor + value)

    def extrapolated_progress(self, predicted_progress):
        """Bridge the short inter-image interval using only the latest EKF displacement."""
        if self.progress is None:
            return predicted_progress
        return self.progress + predicted_progress - self.progress_prediction

    def fresh(self, now):
        """Require a real image timestamp within the allowed age, not a pose timer."""
        return -0.05 <= now - self.valid_stamp <= self.policy.vision_timeout_s

    def wall_too_close(self, now):
        """Use fresh wall geometry even from a tag first discovered during this leg."""
        return (self.clearance is not None
                and -0.05 <= now - self.clearance_stamp <= self.policy.vision_timeout_s
                and self.clearance <= self.policy.wall_stop_distance_m)



def forward_wall_clearance(base_tag):
    """Intersect the robot's forward ray with a vertical tag's supporting plane.

    Args:
        base_tag: 4x4 tag-to-base transform. Tag local +z is its face normal;
            base +x is forward, +y left, +z up. Translation is in meters.

    Returns:
        Positive forward intersection distance from robot center in meters,
        or None for nonfinite, tilted, parallel, or behind-robot planes. This
        assumes wall-mounted tags and does not detect untagged obstacles.
    """
    matrix = np.asarray(base_tag, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    normal, point = matrix[:3, 2], matrix[:3, 3]
    if abs(normal[2]) > 0.3 or abs(normal[0]) < 0.5:
        return None
    distance = float(normal @ point / normal[0])
    return distance if distance > 0 else None
