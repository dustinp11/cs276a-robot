"""Read measured environment data without requiring ROS."""

import argparse
import json
from pathlib import Path

from hw4.tags import tag_key


def load_environment(path):
    """Load the JSON measurements; this does not validate geometry or units."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def missing_measurements(value, path=""):
    """Yield paths to null entries in nested measurement data."""
    # retain paths to every unknown measurement
    if isinstance(value, dict):
        for key, child in value.items():
            yield from missing_measurements(child, f"{path}.{key}".strip("."))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from missing_measurements(child, f"{path}[{index}]")
    elif value is None:
        yield path


def main():
    """Read a config and report unfilled measurements without starting hardware."""
    # report missing values before geometry validation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="path to environment.json")
    args = parser.parse_args()
    missing = list(missing_measurements(load_environment(args.config)))
    if missing:
        print("Measurements to fill:\n" + "\n".join(missing))
        return 1
    validate_environment(load_environment(args.config))
    print("Measured fields validated; collision geometry is checked when planning.")
    return 0


def number(value, name, minimum=None):
    """Require a finite numeric measurement, optionally bounded below."""
    # reject booleans and nonfinite physical measurements
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number.')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}.')
    return float(value)


def validate_environment(data):
    """Validate measured rectangles, robot dimensions, endpoints, and upright tags."""
    # validate measured inputs before hardware initialization
    missing = list(missing_measurements(data))
    if missing:
        raise ValueError('Fill measurements: ' + ', '.join(missing))
    for section, fields in [('workspace', ('width_m', 'height_m')),
                            ('robot', ('length_m', 'width_m'))]:
        for field in fields:
            number(data[section][field], f'{section}.{field}', 0.001)

    # check positions and safety margin
    number(data['robot']['safety_margin_m'], 'robot.safety_margin_m', 0.0)
    for section in ('start', 'goal'):
        for axis in ('x', 'y'):
            number(data[section][f'{axis}_m'], f'{section}.{axis}_m')
    number(data['start']['yaw_deg'], 'start.yaw_deg')
    validate_obstacles(data)
    validate_landmarks(data['landmarks'])
    return data


def validate_obstacles(data):
    """Require measured physical rectangles wholly inside the workspace."""
    # keep physical rectangles inside measured boundaries
    if not data['obstacles']:
        raise ValueError('HW4 requires at least one obstacle.')
    for item in data['obstacles']:
        for axis, extent in [('x', 'width_m'), ('y', 'height_m')]:
            lower = number(item[f'{axis}_min_m'], f'obstacle.{axis}_min_m', 0.0)
            upper = number(item[f'{axis}_max_m'], f'obstacle.{axis}_max_m', 0.0)
            if not lower < upper <= data['workspace'][extent]:
                raise ValueError(f'Obstacle {axis} bounds must increase inside the workspace.')


def validate_landmarks(landmarks):
    """Require unique family/ID pairs, measured centers, normals, and sizes."""
    # allow overlapping ids across different families
    keys = [tag_key(item) for item in landmarks]
    if not keys or any(type(tag_id) is not int or not 0 <= tag_id < 2**31 for _, tag_id in keys):
        raise ValueError('Use surveyed landmarks with nonnegative int32 IDs.')
    if len(set(keys)) != len(keys):
        raise ValueError('Landmark family/ID pairs must be unique.')
    if any(family == '36h11' and tag_id >= 587 for family, tag_id in keys):
        raise ValueError('36h11 IDs must be from 0 through 586.')

    # validate upright tag pose measurements
    for item in landmarks:
        for field in ('x_m', 'y_m', 'face_yaw_deg'):
            number(item[field], f'tag {tag_key(item)}.{field}')
        number(item['z_m'], 'tag.z_m', 0.0)
        number(item['tag_size_m'], 'tag.tag_size_m', 0.001)


def load_validated_environment(path):
    """Load measurements with readable errors before initializing hardware."""
    try:
        return validate_environment(load_environment(path))
    except (KeyError, TypeError) as error:
        raise ValueError(f'Invalid environment structure: {error}') from error


if __name__ == "__main__":
    raise SystemExit(main())
