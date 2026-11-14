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
    corner_timeout_s: float = 18.0
    corner_turn_duration_s: float = 0.95
    corner_stage_angle_rad: float = math.pi
    corner_min_observations: int = 3
    corner_yaw_spread_rad: float = math.pi / 60
    corner_min_correction_rad: float = math.pi / 36
    corner_correction_limit_rad: float = math.pi / 12
    corner_max_corrections: int = 3
    corner_ekf_weight: float = 0.5
    corner_landmark_sigma_m: float = 0.06
    corner_heading_sigma_rad: float = math.pi / 18
    late_scan_start_corner: int = 99
    late_scan_max_angle_rad: float = math.pi / 2
    return_segment_m: float = 0.25
    return_tolerance_m: float = 0.10
    return_max_distance_m: float = 3.0
    drive_budget_factor: float = 1.25
    midpoint_pause_s: float = 3.0
    midpoint_max_correction_m: float = 0.10

    def __post_init__(self):
        """Validate nominal motion limits and bounded optional corrections.

        Args:
            None.

        Returns:
            None. Raises ValueError for invalid phase timing, angles, or counts.
        """
        if not all(math.isfinite(getattr(self, f.name)) and getattr(self, f.name) > 0
                   for f in fields(self) if f.name != 'corner_ekf_weight'):
            raise ValueError('Navigation vision limits must be finite and positive.')
        if not math.isfinite(self.corner_ekf_weight) or not 0 <= self.corner_ekf_weight <= 1:
            raise ValueError('Corner EKF weight must be finite and between zero and one.')
        if int(self.late_scan_start_corner) != self.late_scan_start_corner:
            raise ValueError('Late scan start corner must be an integer.')
        if self.late_scan_max_angle_rad > math.pi / 2:
            raise ValueError('Late center look must be at most 90 degrees.')
        if self.return_segment_m > self.return_max_distance_m:
            raise ValueError('Return segment cannot exceed the total return travel budget.')
        if self.drive_budget_factor < 1:
            raise ValueError('Drive budget factor must be at least one.')
        if self.corner_min_correction_rad > self.corner_correction_limit_rad:
            raise ValueError('Minimum useful correction cannot exceed the total correction budget.')
        if self.corner_correction_limit_rad > math.pi / 4:
            raise ValueError('Optional corner corrections must total at most 45 nominal degrees.')
        if int(self.corner_min_observations) != self.corner_min_observations:
            raise ValueError('Corner observation count must be an integer.')
        if int(self.corner_max_corrections) != self.corner_max_corrections:
            raise ValueError('Corner correction count must be an integer.')


class VisualFeedback:
    """Track optional accepted tag feedback and relative travel anchors."""

    def __init__(self, policy):
        """Initialize empty observations and no active leg anchors.

        Args:
            policy: NavigationPolicy with freshness and uncertainty thresholds.

        Returns:
            None. Starts with no trusted observations.
        """
        self.policy = policy
        self.stamp = self.valid_stamp = -math.inf
        self.clearance, self.clearance_stamp = None, -math.inf
        self.baselines = {}
        self.progress = None
        self.progress_stamp, self.progress_prediction = -math.inf, 0.0
        self.heading = None
        self.tag_yaws, self.yaw_stamp = {}, -math.inf
        self.yaw_history = deque(maxlen=128)
        self.known_ids, self.landmark_sigmas, self.ekf_heading = set(), {}, None
        self.ekf_rejection_reason = 'no image-associated heading'

    def begin_leg(self, heading):
        """Reset relative tag travel anchors for a fixed SLAM-frame heading."""
        self.heading = heading
        self.baselines.clear()
        self.progress, self.progress_stamp = None, -math.inf
        self.progress_prediction = 0.0

    def update(self, stamp, tags, known_ids, clearance, pose, predicted_progress, tag_yaws=None,
               landmark_sigmas=None, ekf_heading=None):
        """Cache one distinct accepted frame and estimate travel from stationary tags.

        Args:
            stamp: Original image time in ROS seconds; duplicates are ignored.
            tags: Accepted base-frame [tag_id, x_m, y_m] relative positions.
            known_ids: IDs successfully updated that were mapped before this frame.
            clearance: Forward wall-plane intersection in meters, or None.
            pose: Current SLAM-frame (x_m, y_m, yaw_rad) estimate.
            predicted_progress: EKF forward displacement since this leg began, in meters.
            tag_yaws: Optional [tag_id, face_normal_yaw_rad] pairs in the robot base frame.
            landmark_sigmas: Pre-update [tag_id, relative major-axis standard deviation in meters].
            ekf_heading: Post-image [SLAM yaw, yaw standard deviation], in radians.

        Returns:
            None. Updates optional geometry, orientation snapshots, freshness, and travel.
        """
        if stamp <= self.stamp:
            return
        self.stamp = stamp
        self.known_ids = set(known_ids)
        self.landmark_sigmas = dict(landmark_sigmas or [])
        self.ekf_heading = ekf_heading
        self.tag_yaws = dict(tag_yaws or [])
        self.yaw_history.append((stamp, self.tag_yaws))
        self.yaw_stamp = stamp if self.tag_yaws else -math.inf
        if tags:
            self.clearance, self.clearance_stamp = clearance, stamp
        if tags and known_ids:
            self.valid_stamp = stamp
        if self.heading is not None:
            self._update_progress(tags, known_ids, pose[2], predicted_progress)

    def stable_tag_yaws(self, now, stopped_at):
        """Collect consistent face-normal observations captured during this stop.

        Args:
            now: Current ROS time in seconds.
            stopped_at: Beginning of this stationary acquisition interval, in seconds.

        Returns:
            Tag-ID-indexed median yaws in radians, excluding stale or inconsistent tags.
            Samples may span the stop, but each tag's newest sample must still be fresh.
        """
        samples = {}
        for stamp, yaws in self.yaw_history:
            if stopped_at <= stamp <= now:
                for tag, yaw in yaws.items():
                    samples.setdefault(tag, []).append((stamp, yaw))
        result = {}
        for tag, observations in samples.items():
            if now - observations[-1][0] <= self.policy.vision_timeout_s:
                yaw = self._stable_yaw([v for _, v in observations])
                if yaw is not None:
                    result[tag] = yaw
        return result

    def _stable_yaw(self, samples):
        """Return a circular median only after enough mutually consistent samples."""
        if len(samples) < self.policy.corner_min_observations:
            return None
        samples = samples[-self.policy.corner_min_observations:]
        values = [samples[0] + wrap_angle(v - samples[0]) for v in samples]
        if max(values) - min(values) > self.policy.corner_yaw_spread_rad:
            return None
        return wrap_angle(median(values))

    def confident_ekf_yaw(self, now, stopped_at):
        """Select an image-associated heading only when every updated known tag is confident.

        Args:
            now: Current ROS time in seconds.
            stopped_at: Start of the current stopped acquisition interval, in seconds.

        Returns:
            SLAM yaw in radians, or None for stale, missing, or uncertain feedback.
            Landmark confidence uses pre-update robot-relative covariance, including
            robot-landmark cross correlations.
        """
        if self.stamp < stopped_at or not 0 <= now - self.stamp <= self.policy.vision_timeout_s:
            self.ekf_rejection_reason = 'no fresh post-stop image'
            return None
        if not self.known_ids or self.ekf_heading is None:
            self.ekf_rejection_reason = 'no accepted mapped tag or image heading'
            return None
        sigmas = [self.landmark_sigmas.get(tag, math.inf) for tag in self.known_ids]
        if not all(math.isfinite(v) and 0 <= v <= self.policy.corner_landmark_sigma_m for v in sigmas):
            self.ekf_rejection_reason = 'missing or uncertain robot-relative landmark covariance'
            return None
        yaw, sigma = self.ekf_heading
        if math.isfinite(yaw) and math.isfinite(sigma) and 0 <= sigma <= self.policy.corner_heading_sigma_rad:
            self.ekf_rejection_reason = ''
            return yaw
        self.ekf_rejection_reason = 'uncertain image heading'
        return None

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


def tag_normal_yaw(base_tag):
    """Return a vertical tag face normal's base-frame yaw in radians, or None."""
    matrix = np.asarray(base_tag, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    normal = matrix[:3, 2]
    if abs(normal[2]) > 0.3 or math.hypot(normal[0], normal[1]) < 0.9:
        return None
    return math.atan2(normal[1], normal[0])


def relative_tag_turn(reference, current, expected_turn):
    """Estimate rotation from the same tag's before/after face orientation.

    Args:
        reference: Tag-ID-indexed base-frame normal yaws before turning, in radians.
        current: Tag-ID-indexed normal yaws after turning, in radians.
        expected_turn: Signed nominal turn including correction commands, in radians.

    Returns:
        Signed median yaw change, positive CCW, or None without consistent matches.
        Reject candidates more than 45 degrees from prediction (including PnP
        face flips), and matched-tag disagreement above 15 degrees. No prior map
        positions or tag orientations are used.
    """
    changes = [wrap_angle(reference[tag] - current[tag]) for tag in reference.keys() & current.keys()]
    changes = [v for v in changes if abs(wrap_angle(v - expected_turn)) <= math.pi / 4]
    if not changes:
        return None
    center = changes[0]
    measured = wrap_angle(median([center + wrap_angle(v - center) for v in changes]))
    if any(abs(wrap_angle(v - measured)) > math.pi / 12 for v in changes):
        return None
    return measured
