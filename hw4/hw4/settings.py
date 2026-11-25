"""Shared settings reader for ROS and offline planning."""

import math
from pathlib import Path
import yaml

from hw4.planner import get_mode
from hw4.tags import canonical_family


def load_settings(path):
    """Read ROS-format tuning and validate limits before motion is permitted."""
    # reject unusable tuning before allowing motion
    data = yaml.safe_load(Path(path).read_text())['/**']['ros__parameters']
    nav = data['navigation']
    get_mode(nav['mode'])
    for name, value in nav.items():
        if name in ('mode', 'feedback_required', 'visual_feedback', 'tag_search_enabled',
                    'goal_reference_family', 'goal_reference_id'):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'navigation.{name} must be positive and finite.')

    for name in ('feedback_required', 'tag_search_enabled'):
        if type(nav[name]) is not bool:
            raise ValueError(f'navigation.{name} must be true or false.')

    if nav['visual_feedback'] not in ('at_goal', 'near_goal', 'throughout', 'off'):
        raise ValueError('navigation.visual_feedback must be at_goal, near_goal, throughout, or off.')
    if nav['feedback_required'] and nav['visual_feedback'] != 'throughout':
        raise ValueError('Required feedback needs visual_feedback: throughout.')
    if nav['goal_step_fraction'] > 1:
        raise ValueError('navigation.goal_step_fraction must be at most one.')

    canonical_family(nav['goal_reference_family'])
    if type(nav['goal_reference_id']) is not int or nav['goal_reference_id'] < 0:
        raise ValueError('navigation.goal_reference_id must be a nonnegative integer.')

    # constrain sampling and footprint integration limits
    for name in ('stable_samples', 'max_turns', 'max_checkpoint_corrections'):
        if type(nav[name]) is not int or nav[name] < 1:
            raise ValueError(f'navigation.{name} must be a positive integer.')
    if nav['stable_samples'] < 3 or nav['stable_samples'] > 20:
        raise ValueError('Use 3 through 20 distinct stable samples.')
    if not 0.02 <= nav['grid_resolution_m'] <= 0.20:
        raise ValueError('Use grid resolution between 0.02 and 0.20 meters.')
    if nav['drive_duration_s'] > 1.0:
        raise ValueError('Drive segments must not exceed one second between stopped checks.')

    if nav['goal_tolerance_m'] > nav['position_tolerance_m']:
        raise ValueError('Goal tolerance must not exceed waypoint tolerance.')

    # reject unusable quality and mount settings
    for name, value in data['localization'].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'localization.{name} must be finite and nonnegative.')
    if type(data['localization']['min_inlier_tags']) is not int or data['localization']['min_inlier_tags'] < 2:
        raise ValueError('localization.min_inlier_tags must be an integer of at least two.')

    for name, value in data['camera_mount'].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'camera_mount.{name} must be finite.')
    return data
