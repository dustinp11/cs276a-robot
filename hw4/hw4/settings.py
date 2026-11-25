"""Shared settings reader for ROS and offline planning."""

import math
from pathlib import Path
import yaml

from hw4.planner import get_mode


def load_settings(path):
    """Read ROS-format tuning and validate limits before motion is permitted."""
    # reject unusable tuning before allowing motion
    data = yaml.safe_load(Path(path).read_text())['/**']['ros__parameters']
    nav = data['navigation']
    get_mode(nav['mode'])
    for name, value in nav.items():
        if name == 'mode':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'navigation.{name} must be positive and finite.')

    # constrain sampling and footprint integration limits
    for name in ('stable_samples', 'max_turns'):
        if type(nav[name]) is not int or nav[name] < 1:
            raise ValueError(f'navigation.{name} must be a positive integer.')
    if nav['stable_samples'] < 3 or nav['stable_samples'] > 20:
        raise ValueError('Use 3 through 20 distinct stable samples.')
    if not 0.02 <= nav['grid_resolution_m'] <= 0.20:
        raise ValueError('Use grid resolution between 0.02 and 0.20 meters.')
    if nav['drive_duration_s'] > 1.0:
        raise ValueError('Drive segments must not exceed one second between stopped checks.')

    # reject unusable quality and mount settings
    for name, value in data['localization'].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'localization.{name} must be finite and nonnegative.')
    for name, value in data['camera_mount'].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'camera_mount.{name} must be finite.')
    return data
