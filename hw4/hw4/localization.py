"""Known-map AprilTag pose estimates using HW3's camera geometry conventions."""

from dataclasses import dataclass
import math
import numpy as np

from hw3.coordinates import CameraExtrinsics, make_transform, wrap_angle
from hw4.tags import tag_key


@dataclass(frozen=True)
class Feedback:
    """Measured footprint-center pose; capture time uses monotonic seconds."""

    x: float
    y: float
    yaw: float
    stamp: float
    tags: tuple = ()

    @property
    def xy(self):
        """Return world position in meters."""
        return self.x, self.y


def tag_transform(item):
    """Map PnP printed-right/up/outward-normal axes into the measured world frame."""
    angle = math.radians(item['face_yaw_deg'])
    sine, cosine = math.sin(angle), math.cos(angle)
    axes = np.array([[-sine, 0., cosine], [cosine, 0., sine], [0., 1., 0.]])
    return make_transform(axes, [item['x_m'], item['y_m'], item['z_m']])


class Localizer:
    """Reject impossible poses and disagreeing mapped-tag measurements."""

    def __init__(self, environment, mount, policy):
        """Use camera offsets relative to the footprint center, not an assumed base origin."""
        # cache measured geometry and camera mounting
        self.camera_base = CameraExtrinsics(**mount).t_cam_base
        self.landmarks = {tag_key(item): tag_transform(item) for item in environment['landmarks']}
        self.policy = policy
        self.width = environment['workspace']['width_m']
        self.height = environment['workspace']['height_m']

    def candidate(self, tag_id, camera_tag):
        """Recover world-from-base pose and reject excessive height, tilt, or range."""
        # reject implausible geometry before estimating pose
        distance = np.linalg.norm(camera_tag[:3, 3])
        if not np.isfinite(camera_tag).all() or not 0.1 <= distance <= self.policy['max_tag_distance_m']:
            raise ValueError('invalid tag range or transform')
        if camera_tag[2, 3] <= 0:
            raise ValueError('tag lies behind camera')

        # compose measured transforms without prediction
        base = self.landmarks[tag_id] @ np.linalg.inv(camera_tag) @ self.camera_base
        tilt = math.acos(float(np.clip(base[2, 2], -1.0, 1.0)))
        if abs(base[2, 3]) > self.policy['max_height_error_m'] or tilt > math.radians(self.policy['max_tilt_deg']):
            raise ValueError('implausible base height or tilt')
        x, y = float(base[0, 3]), float(base[1, 3])
        if not 0 <= x <= self.width or not 0 <= y <= self.height:
            raise ValueError('estimated robot lies outside workspace')
        return x, y, math.atan2(base[1, 0], base[0, 0])

    def agree(self, first, second):
        """Check independent tag solutions before averaging their poses."""
        return (math.dist(first[:2], second[:2]) <= self.policy['max_disagreement_m']
                and abs(wrap_angle(first[2] - second[2])) <= self.policy['max_disagreement_rad'])

    def estimate(self, measurements, stamp):
        """Return an agreeing pose from ((family, ID), camera_tag) pairs."""
        # recover independent poses from mapped tags
        candidates = []
        for tag_id, transform in measurements:
            try:
                candidates.append((tag_id, self.candidate(tag_id, transform)))
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not candidates:
            raise ValueError('no usable mapped tag transform')

        # require an internally consistent majority
        groups = [[other for other in candidates if self.agree(anchor[1], other[1])] for anchor in candidates]
        group = max(groups, key=len)
        if len(group) <= len(candidates) // 2 or any(not self.agree(a[1], b[1]) for a in group for b in group):
            raise ValueError('mapped tags disagree about robot pose')
        values = np.array([pose for _, pose in group])
        x, y = values[:, :2].mean(axis=0)
        yaw = math.atan2(np.sin(values[:, 2]).mean(), np.cos(values[:, 2]).mean())
        return Feedback(float(x), float(y), yaw, stamp, tuple(tag for tag, _ in group))


def stable_feedback(samples, after, now, policy):
    """Average distinct, recent, post-settle frames only when their poses agree."""
    # require independent frames captured after settling
    fresh = {sample.stamp: sample for sample in samples
             if sample.stamp >= after and 0 <= now - sample.stamp <= policy['pose_timeout_s']}
    recent = sorted(fresh.values(), key=lambda sample: sample.stamp)[-policy['stable_samples']:]
    if len(recent) < policy['stable_samples']:
        return None
    anchor = recent[-1]
    if any(math.dist(sample.xy, anchor.xy) > policy['stable_position_spread_m']
           or abs(wrap_angle(sample.yaw - anchor.yaw)) > policy['stable_heading_spread_rad'] for sample in recent):
        return None

    # average stopped observations with circular yaw
    x = sum(sample.x for sample in recent) / len(recent)
    y = sum(sample.y for sample in recent) / len(recent)
    yaw = math.atan2(sum(math.sin(sample.yaw) for sample in recent), sum(math.cos(sample.yaw) for sample in recent))
    return Feedback(x, y, yaw, anchor.stamp, tuple(sorted({tag for sample in recent for tag in sample.tags})))
