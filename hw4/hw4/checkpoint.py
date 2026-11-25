"""Fresh stopped AprilTag checks for waypoint progress and measured goal arrival."""

import math
from hw4.localization import stable_feedback


def checkpoint_pose(samples, after, now, policy, final=False):
    """Return a consistent stopped pose, or None until enough distinct frames arrive."""
    # allow slower observations while remaining stopped
    settings = dict(policy, pose_timeout_s=policy['checkpoint_window_s'])
    if final:
        settings['stable_position_spread_m'] = min(policy['stable_position_spread_m'], policy['goal_tolerance_m'] / 2)
    pose = stable_feedback(samples, after, now, settings)
    if pose is None or now - pose.stamp > policy['pose_timeout_s']:
        return None
    return pose


def goal_confirmed(samples, after, now, target, policy):
    """Require every confirming goal observation to lie within the goal tolerance."""
    # do not let averaging hide overshoot
    fresh = {pose.stamp: pose for pose in samples if pose.stamp >= after
             and 0 <= now - pose.stamp <= policy['checkpoint_window_s']}
    poses = sorted(fresh.values(), key=lambda pose: pose.stamp)[-policy['stable_samples']:]
    return (len(poses) == policy['stable_samples']
            and now - poses[-1].stamp <= policy['pose_timeout_s']
            and all(math.dist(pose.xy, target) <= policy['goal_tolerance_m'] for pose in poses))
