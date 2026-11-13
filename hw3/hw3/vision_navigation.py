"""Fresh tag feedback and wall-plane geometry for navigation without a prior map."""

from collections import deque
from dataclasses import dataclass, fields
import math
from statistics import median

import numpy as np

from .coordinates import wrap_angle


@dataclass(frozen=True)
class NavigationPolicy:
    """Limits in meters, radians, and seconds; clearances are from robot center."""

    vision_timeout_s: float = 1.0
    localization_timeout_s: float = 8.0
    stable_window_s: float = 0.6
    stable_position_m: float = 0.05
    stable_heading_rad: float = 0.08
    min_known_tags: int = 2
    min_frames: int = 3
    wall_stop_distance_m: float = 0.45
    drive_heading_limit_rad: float = 0.35
    drive_budget_factor: float = 1.25
    scan_timeout_s: float = 20.0

    def __post_init__(self):
        """Reject invalid limits rather than silently disabling motion guards."""
        if not all(math.isfinite(getattr(self, f.name)) and getattr(self, f.name) > 0
                   for f in fields(self)):
            raise ValueError('Navigation vision limits must be finite and positive.')
        if self.min_frames < 3 or self.min_known_tags < 2 or self.drive_budget_factor < 1:
            raise ValueError('Require >=3 frames, >=2 known tags, and drive budget >=1.')


class VisualFeedback:
    """Track fresh known-tag updates, stable stopped poses, and relative progress."""

    def __init__(self, policy):
        """Initialize empty observations and no active leg anchors."""
        self.policy = policy
        self.stamp = self.valid_stamp = -math.inf
        self.clearance = None
        self.samples = deque(maxlen=100)
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
        if tags and known_ids:
            self.valid_stamp, self.clearance = stamp, clearance
        self.samples.append((stamp, set(known_ids), pose))
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

    def stable(self, now, since):
        """Require multiple known tags and stable poses from distinct stopped frames.

        Args:
            now: Current ROS time in seconds.
            since: Earliest allowed frame time, after stopping/settling.

        Returns:
            True only with sufficient fresh frames spanning the stability window.
        """
        cutoff = max(since, now - self.policy.stable_window_s - 0.3)
        samples = [s for s in self.samples if s[0] >= cutoff and s[1]]
        if not self.fresh(now) or len(samples) < self.policy.min_frames:
            return False
        if samples[-1][0] - samples[0][0] < self.policy.stable_window_s:
            return False
        ids = set().union(*(s[1] for s in samples))
        origin = samples[0][2]
        return len(ids) >= self.policy.min_known_tags and all(
            math.hypot(p[0] - origin[0], p[1] - origin[1]) <= self.policy.stable_position_m
            and abs(wrap_angle(p[2] - origin[2])) <= self.policy.stable_heading_rad
            for _, _, p in samples)


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
