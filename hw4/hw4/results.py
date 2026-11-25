"""Small JSON, CSV, and SVG artifacts for comparing planned and observed motion."""

import csv
import json
import math
from pathlib import Path
import shutil

from hw4.plan_plot import map_svg, plan_description, plan_segments


def path_metrics(grid, path):
    """Report planned length and body clearance in meters, not hardware results."""
    segments = list(zip(path, path[1:])) or [(path[0], path[0])]
    return {'length_m': sum(math.dist(a, b) for a, b in segments),
            'min_body_clearance_m': min(grid.clearance(a, b) for a, b in segments)}


def save_plans(directory, grid, environment, plans):
    """Save the selected mode path and an overhead plot including the planning grid."""
    # archive the selected mode and geometry
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {name: {'waypoints_xy_m': path, 'segments': plan_segments(environment, path),
                      **path_metrics(grid, path)} for name, path in plans.items()}
    (directory / 'planned_paths.json').write_text(json.dumps(payload, indent=2) + '\n')
    (directory / 'planned_paths.svg').write_text(map_svg(grid, environment, plans))
    (directory / 'planned_path.txt').write_text(plan_description(environment, plans))



class RunLog:
    """Archive effective inputs and persist feedback, commands, and terminal state."""

    def __init__(self, directory, paths):
        """Create one run session; refuse to overwrite an earlier mission."""
        # reserve one immutable session before recording
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            raise ValueError('output_dir must be absolute.')
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / 'session.lock').open('x') as lock:
            lock.write('run1\n')
        for name, source in paths.items():
            shutil.copy2(source, self.directory / name)

        # open separate observation and command streams
        self.handles = []
        self.observations = self.open_csv('observations.csv', [
            'time_s', 'image_stamp_ns', 'accepted', 'reason', 'x_m', 'y_m', 'yaw_rad', 'tag_ids'])
        self.commands = self.open_csv('commands.csv', [
            'time_s', 'state', 'waypoint', 'reason', 'linear_mps', 'angular_radps',
            'pose_age_s', 'measured_x_m', 'measured_y_m', 'measured_yaw_rad',
            'predicted_x_m', 'predicted_y_m', 'predicted_yaw_rad',
            'turn_steps', 'commanded_segment_distance_m', 'segment_budget_m', 'measured_body_clearance_m', 'turn_target_yaw_rad', 'turn_progress_rad',
            'pose_source', 'feedback_note'])

    def open_csv(self, name, header):
        """Open a line-buffered CSV stream with explicit units in its header."""
        # flush rows promptly for interrupted runs
        handle = (self.directory / name).open('w', buffering=1, newline='')
        self.handles.append(handle)
        writer = csv.writer(handle)
        writer.writerow(header)
        return writer

    def status(self, state, reason, **details):
        """Write a current result and append the same event to its history."""
        # preserve terminal result alongside transition history
        payload = {'iteration': 'run1', 'state': state, 'reason': reason,
                   'goal_reached': state == 'complete', **details}
        text = json.dumps(payload)
        (self.directory / 'status.json').write_text(text + '\n')
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')

    def close(self):
        """Flush local run artifacts when the ROS node exits."""
        for handle in self.handles:
            handle.close()
