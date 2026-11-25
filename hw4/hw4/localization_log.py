"""Record independent tag estimates against a user-measured fixed robot center."""

import json
import math

from hw4.results import RunLog


class LocalizationLog(RunLog):
    """Reuse observation recording and add a flat per-tag error CSV."""

    def __init__(self, directory, paths, reference):
        """Archive the fixed world-frame reference xy in meters, not an estimated pose."""
        # preserve original observations alongside independent estimates
        super().__init__(directory, paths)
        self.capture_count, self.usable_count = 0, 0
        self.reference = reference
        self.raw_observations = self.observations
        self.observations = self
        self.per_tag = self.open_csv('per_tag.csv', [
            'time_s', 'image_stamp_ns', 'tag', 'reason', 'x_m', 'y_m', 'yaw_rad',
            'camera_range_m', 'dx_m', 'dy_m', 'position_error_m', 'fusion_status'])
        (self.directory / 'localization_reference.json').write_text(json.dumps({
            'xy_m': reference, 'source': 'user physical placement',
            'heading_ground_truth': None, 'mode': 'manual rotation; motors not launched'}, indent=2) + '\n')

    def writerow(self, row):
        """Keep every raw observation and flatten its individual candidate estimates."""
        # retain failures without inventing position measurements
        self.raw_observations.writerow(row)
        self.capture_count += 1
        self.usable_count += int(row[2] is True)
        for tag, details in json.loads(row[-1]).items():
            pose = details.get('pose_xy_yaw')
            if pose is None:
                self.per_tag.writerow([*row[:2], tag, details['rejected'], *([''] * 7), 'rejected'])
                continue
            dx, dy = pose[0] - self.reference[0], pose[1] - self.reference[1]
            self.per_tag.writerow([*row[:2], tag, 'usable', *pose, details['camera_range_m'],
                                   dx, dy, math.hypot(dx, dy), details.get('fusion_status', 'not fused')])
