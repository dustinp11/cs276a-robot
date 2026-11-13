"""Visual Motion Estimation using AprilTag Observations (Step 2).

Estimates relative planar robot motion (Delta x, Delta y, Delta theta) and uncertainty
between consecutive camera frames using stationary AprilTags.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from .coordinates import (
    CameraExtrinsics,
    invert_transform,
    wrap_angle,
)


@dataclass(frozen=True)
class TagObservation:
    """Observation of a single AprilTag in a camera frame."""
    tag_id: int
    family: str
    t_cam_tag: np.ndarray      # 4x4 transform from tag to camera
    t_base_tag: np.ndarray     # 4x4 transform from tag to robot base
    decision_margin: float
    hamming: int
    distance_m: float
    quality_weight: float


@dataclass(frozen=True)
class FrameObservation:
    """Collection of AprilTag observations at a specific timestamp."""
    stamp_ns: int
    stamp_s: float
    tags: Dict[int, TagObservation]


@dataclass(frozen=True)
class VisualMotionEstimate:
    """Planar relative motion estimate between two consecutive frames."""
    dx: float                    # Relative displacement along robot X-axis at t1 (meters)
    dy: float                    # Relative displacement along robot Y-axis at t1 (meters)
    dtheta: float                # Relative rotation from t1 to t2 (radians)
    covariance: np.ndarray       # 3x3 covariance matrix [dx, dy, dtheta]
    t1_s: float
    t2_s: float
    dt_s: float
    matched_tag_ids: Tuple[int, ...]
    valid: bool
    status_message: str


class VisualMotionEstimator:
    """Estimates relative robot motion between consecutive camera frames."""

    def __init__(
        self,
        extrinsics: Optional[CameraExtrinsics] = None,
        min_common_tags: int = 1,
        max_dt_s: float = 1.5,
        min_decision_margin: float = 20.0,
        max_hamming: int = 1,
        max_tag_distance_m: float = 5.0,
        min_tag_distance_m: float = 0.1,
        max_trans_disagreement_m: float = 0.35,
        max_rot_disagreement_deg: float = 25.0,
        base_trans_noise_m: float = 0.02,
        base_rot_noise_rad: float = 0.03,
    ):
        self.extrinsics = extrinsics if extrinsics is not None else CameraExtrinsics()
        self.min_common_tags = max(1, int(min_common_tags))
        self.max_dt_s = float(max_dt_s)
        self.min_decision_margin = float(min_decision_margin)
        self.max_hamming = int(max_hamming)
        self.max_tag_distance_m = float(max_tag_distance_m)
        self.min_tag_distance_m = float(min_tag_distance_m)
        self.max_trans_disagreement_m = float(max_trans_disagreement_m)
        self.max_rot_disagreement_rad = math.radians(float(max_rot_disagreement_deg))
        self.base_trans_noise_m = float(base_trans_noise_m)
        self.base_rot_noise_rad = float(base_rot_noise_rad)

        self._previous_frame: Optional[FrameObservation] = None

    def reset(self):
        """Clear observation history."""
        self._previous_frame = None

    def process_frame(
        self,
        stamp_s: float,
        detections: List[dict],
        stamp_ns: Optional[int] = None,
    ) -> VisualMotionEstimate:
        """Process a frame of AprilTag detections and estimate relative motion.

        Args:
            stamp_s: Timestamp in seconds.
            detections: List of detection dictionaries, each containing:
                - 'id': int
                - 'family': str (optional, default '36h11')
                - 't_cam_tag': 4x4 np.ndarray (tag pose in camera optical frame)
                - 'margin': float (decision margin)
                - 'hamming': int
            stamp_ns: Optional timestamp in nanoseconds.

        Returns:
            VisualMotionEstimate describing relative motion and uncertainty.
        """
        if stamp_ns is None:
            stamp_ns = int(stamp_s * 1e9)

        current_tags: Dict[int, TagObservation] = {}
        for det in detections:
            tag_id = int(det['id'])
            margin = float(det.get('margin', 30.0))
            hamming = int(det.get('hamming', 0))
            t_cam_tag = np.asarray(det['t_cam_tag'], dtype=float)

            if hamming > self.max_hamming or margin < self.min_decision_margin:
                continue

            # In camera optical frame, distance along optical axis +Z
            dist = float(np.linalg.norm(t_cam_tag[:3, 3]))
            if dist < self.min_tag_distance_m or dist > self.max_tag_distance_m:
                continue

            # Tag pose in robot base frame
            t_base_tag = self.extrinsics.camera_to_base_transform(t_cam_tag)

            # Quality weight: high margin, smaller distance
            weight = max(1.0, min(margin, 100.0)) / (1.0 + dist * dist)

            current_tags[tag_id] = TagObservation(
                tag_id=tag_id,
                family=det.get('family', '36h11'),
                t_cam_tag=t_cam_tag,
                t_base_tag=t_base_tag,
                decision_margin=margin,
                hamming=hamming,
                distance_m=dist,
                quality_weight=weight,
            )

        current_frame = FrameObservation(
            stamp_ns=stamp_ns,
            stamp_s=stamp_s,
            tags=current_tags,
        )

        if self._previous_frame is None:
            self._previous_frame = current_frame
            return VisualMotionEstimate(
                dx=0.0,
                dy=0.0,
                dtheta=0.0,
                covariance=np.diag([1e6, 1e6, 1e6]),
                t1_s=stamp_s,
                t2_s=stamp_s,
                dt_s=0.0,
                matched_tag_ids=(),
                valid=False,
                status_message="First observation frame received; baseline established",
            )

        prev_frame = self._previous_frame
        dt = stamp_s - prev_frame.stamp_s

        if dt <= 0.0 or dt > self.max_dt_s:
            self._previous_frame = current_frame
            return VisualMotionEstimate(
                dx=0.0,
                dy=0.0,
                dtheta=0.0,
                covariance=np.diag([1e6, 1e6, 1e6]),
                t1_s=prev_frame.stamp_s,
                t2_s=stamp_s,
                dt_s=dt,
                matched_tag_ids=(),
                valid=False,
                status_message=f"Time step invalid or exceeded threshold: {dt:.3f}s",
            )

        common_ids = sorted(set(prev_frame.tags.keys()).intersection(current_tags.keys()))

        if len(common_ids) < self.min_common_tags:
            self._previous_frame = current_frame
            return VisualMotionEstimate(
                dx=0.0,
                dy=0.0,
                dtheta=0.0,
                covariance=np.diag([1e6, 1e6, 1e6]),
                t1_s=prev_frame.stamp_s,
                t2_s=stamp_s,
                dt_s=dt,
                matched_tag_ids=(),
                valid=False,
                status_message="No common landmarks visible between consecutive frames",
            )

        # Estimate relative motion for each common tag
        # Stationary landmark: T_R1_R2 = T_R1_L @ inv(T_R2_L)
        candidates = []
        for tag_id in common_ids:
            obs1 = prev_frame.tags[tag_id]
            obs2 = current_tags[tag_id]

            t_r1_l = obs1.t_base_tag
            t_r2_l = obs2.t_base_tag

            t_r1_r2 = t_r1_l @ invert_transform(t_r2_l)

            cand_dx = float(t_r1_r2[0, 3])
            cand_dy = float(t_r1_r2[1, 3])
            cand_dtheta = float(math.atan2(t_r1_r2[1, 0], t_r1_r2[0, 0]))

            cand_weight = math.sqrt(obs1.quality_weight * obs2.quality_weight)
            mean_dist = 0.5 * (obs1.distance_m + obs2.distance_m)

            candidates.append({
                'id': tag_id,
                'dx': cand_dx,
                'dy': cand_dy,
                'dtheta': cand_dtheta,
                'weight': cand_weight,
                'distance': mean_dist,
            })

        # Outlier rejection when 3 or more tags are visible
        inliers = self._filter_outliers(candidates)

        if not inliers:
            self._previous_frame = current_frame
            return VisualMotionEstimate(
                dx=0.0,
                dy=0.0,
                dtheta=0.0,
                covariance=np.diag([1e6, 1e6, 1e6]),
                t1_s=prev_frame.stamp_s,
                t2_s=stamp_s,
                dt_s=dt,
                matched_tag_ids=(),
                valid=False,
                status_message="All landmark motion estimates rejected due to mutual disagreement",
            )

        # Combine inlier candidates
        weights = np.array([c['weight'] for c in inliers], dtype=float)
        w_sum = float(np.sum(weights))
        if w_sum < 1e-9:
            weights = np.ones(len(inliers), dtype=float)
            w_sum = float(len(inliers))
        weights /= w_sum

        dx_comb = float(np.sum([c['dx'] * w for c, w in zip(inliers, weights)]))
        dy_comb = float(np.sum([c['dy'] * w for c, w in zip(inliers, weights)]))

        sin_sum = float(np.sum([math.sin(c['dtheta']) * w for c, w in zip(inliers, weights)]))
        cos_sum = float(np.sum([math.cos(c['dtheta']) * w for c, w in zip(inliers, weights)]))
        dtheta_comb = float(math.atan2(sin_sum, cos_sum))

        # Covariance estimation
        cov = self._compute_covariance(inliers, dx_comb, dy_comb, dtheta_comb, weights)

        # Advance frame history
        self._previous_frame = current_frame

        matched_ids = tuple(c['id'] for c in inliers)
        return VisualMotionEstimate(
            dx=dx_comb,
            dy=dy_comb,
            dtheta=dtheta_comb,
            covariance=cov,
            t1_s=prev_frame.stamp_s,
            t2_s=stamp_s,
            dt_s=dt,
            matched_tag_ids=matched_ids,
            valid=True,
            status_message=f"VO successful using {len(matched_ids)} tags: {list(matched_ids)}",
        )

    def _filter_outliers(self, candidates: List[dict]) -> List[dict]:
        """Reject candidates that disagree significantly with the majority."""
        if len(candidates) < 3:
            return candidates

        # Check mutual agreement
        inliers = []
        for i, c1 in enumerate(candidates):
            agreements = 0
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                trans_err = math.hypot(c1['dx'] - c2['dx'], c1['dy'] - c2['dy'])
                rot_err = abs(wrap_angle(c1['dtheta'] - c2['dtheta']))
                if (trans_err <= self.max_trans_disagreement_m and
                        rot_err <= self.max_rot_disagreement_rad):
                    agreements += 1

            # Keep candidate if it agrees with at least half the other candidates
            if agreements >= (len(candidates) - 1) // 2:
                inliers.append(c1)

        return inliers if inliers else candidates

    def _compute_covariance(
        self,
        inliers: List[dict],
        dx: float,
        dy: float,
        dtheta: float,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Compute the 3x3 covariance matrix for the combined relative motion."""
        num_tags = len(inliers)
        mean_distance = float(np.average([c['distance'] for c in inliers], weights=weights))

        # Base noise scales with squared distance (depth uncertainty of monocular tag)
        dist_scale = max(1.0, (mean_distance / 1.5) ** 2)

        # Uncertainty reduces with number of independent tags
        tag_scale = 1.0 / math.sqrt(num_tags)

        var_x = (self.base_trans_noise_m * dist_scale * tag_scale) ** 2
        var_y = (self.base_trans_noise_m * dist_scale * tag_scale) ** 2
        var_theta = (self.base_rot_noise_rad * tag_scale) ** 2

        # If multiple tags, add empirical sample variance to reflect agreement dispersion
        if num_tags >= 2:
            sample_var_x = float(np.sum([w * (c['dx'] - dx) ** 2 for c, w in zip(inliers, weights)]))
            sample_var_y = float(np.sum([w * (c['dy'] - dy) ** 2 for c, w in zip(inliers, weights)]))
            sample_var_theta = float(np.sum([
                w * (wrap_angle(c['dtheta'] - dtheta)) ** 2 for c, w in zip(inliers, weights)
            ]))

            var_x += sample_var_x
            var_y += sample_var_y
            var_theta += sample_var_theta

        cov = np.diag([var_x, var_y, var_theta])
        return cov
