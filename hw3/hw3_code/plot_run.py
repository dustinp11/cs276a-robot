"""Export recorded estimated trajectories and landmark uncertainty as a standalone SVG."""

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np


def read_estimates(directory, lap=None):
    """Read complete JSONL snapshots; optionally stop at a recorded lap-complete snapshot."""
    snapshots = []
    with (Path(directory) / 'run.jsonl').open() as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('kind') != 'estimate':
                continue
            snapshots.append(record)
            if lap is not None and record.get('event') == 'lap_complete' and record.get('lap') == lap:
                return snapshots
    if not snapshots or lap is not None:
        raise ValueError('No estimate snapshots or requested completed lap found')
    return snapshots


def alignment(ground_truth):
    """Build the measured-start world alignment; no best-fit map alignment is performed."""
    x, y, yaw = ground_truth.get('start_pose', [0., 0., 0.])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return rotation, np.array([x, y])


def map_data(snapshots, ground_truth):
    """Transform estimated trajectory, landmarks, and positional covariance into one frame."""
    rotation, translation = alignment(ground_truth)
    trajectory = np.array([rotation @ item['state'][:2] + translation for item in snapshots])
    final = snapshots[-1]
    state, covariance = np.array(final['state']), np.array(final['covariance'])
    landmarks = []
    for number, key in enumerate(final['landmark_ids']):
        index = 3 + 2*number
        position = rotation @ state[index:index+2] + translation
        uncertainty = rotation @ covariance[index:index+2, index:index+2] @ rotation.T
        landmarks.append((tuple(key), position, uncertainty))
    return trajectory, landmarks


def ellipse_points(position, covariance):
    """Return points on the landmark's 95% positional covariance ellipse."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    angles = np.linspace(0., 2.*math.pi, 65)
    circle = np.array([np.cos(angles), np.sin(angles)])
    return (eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 0.)*5.991464547)[:, None]*circle)).T + position


def compare_landmarks(landmarks, truth):
    """Report Euclidean matched errors and missing/spurious IDs without assuming a count."""
    actual = {(item.get('family', '41h12'), item['id']): np.array([item['x_m'], item['y_m']])
              for item in truth.get('landmarks', [])}
    estimated = {key: position for key, position, _ in landmarks}
    matched = sorted(actual.keys() & estimated.keys())
    errors = [{'id': list(key), 'error_m': float(np.linalg.norm(estimated[key]-actual[key]))} for key in matched]
    return {'matched_count': len(matched), 'mean_landmark_error_m': float(np.mean([item['error_m'] for item in errors])) if errors else None,
            'per_landmark': errors, 'missing_ids': [list(key) for key in sorted(actual.keys()-estimated.keys())],
            'spurious_ids': [list(key) for key in sorted(estimated.keys()-actual.keys())]}


def make_svg(trajectory, landmarks, truth):
    """Draw equal-scale metric axes, estimated path, tag labels, and covariance ellipses."""
    ellipses = [ellipse_points(position, covariance) for _, position, covariance in landmarks]
    actual = [np.array([item['x_m'], item['y_m']]) for item in truth.get('landmarks', [])]
    points = np.vstack([trajectory] + ellipses + ([np.array(actual)] if actual else []))
    lower, upper = points.min(axis=0)-0.3, points.max(axis=0)+0.3
    scale = 680. / max(float(np.max(upper-lower)), 1.)
    def screen(point):
        return (80.+(point[0]-lower[0])*scale, 790.-(point[1]-lower[1])*scale)
    def polyline(points, color, width):
        text = ' '.join(f'{x:.2f},{y:.2f}' for x, y in map(screen, points))
        return f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="{width}"/>'
    parts = _svg_header(bool(truth)) + _svg_grid(lower, upper, screen)
    parts.append(polyline(trajectory, '#1976d2', 2.5))
    for (key, position, _), ellipse in zip(landmarks, ellipses):
        x, y = screen(position)
        label = html.escape(f'{key[0]}:{key[1]}')
        parts += [polyline(ellipse, '#238636', 1.5), f'<circle cx="{x}" cy="{y}" r="4" fill="#238636"/>',
                  f'<text x="{x+7}" y="{y-7}">{label}</text>']
    for point in actual:
        x, y = screen(point)
        parts.append(f'<path d="M{x-5},{y-5} L{x+5},{y+5} M{x-5},{y+5} L{x+5},{y-5}" stroke="#d32f2f" stroke-width="2"/>')
    parts.append('</svg>')
    return '\n'.join(parts)


def _svg_header(world):
    """Label map provenance, uncertainty confidence, and physical units."""
    frame = 'world frame aligned by measured start' if world else 'local starting frame'
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="880" viewBox="0 0 960 880">',
            '<rect width="960" height="880" fill="white"/>', '<g font-family="sans-serif" font-size="12"></g>',
            '<text x="40" y="30" font-size="20">HW3 estimated trajectory and landmark map</text>',
            f'<text x="40" y="55">{frame}; coordinates in meters; equal axis scale</text>',
            '<text x="40" y="77">Blue: estimated path. Green: landmarks and 95% covariance. Red crosses: measured truth.</text>',
            '<text x="450" y="850">x (m)</text>', '<text x="15" y="450">y (m)</text>']


def _svg_grid(lower, upper, screen):
    """Draw bounded metric grid lines and coordinate labels."""
    parts = []
    spacing = max(0.5, 10**math.floor(math.log10(max(float(np.max(upper-lower))/8., 0.01))))
    for axis in (0, 1):
        for value in np.arange(math.ceil(lower[axis]/spacing)*spacing, upper[axis], spacing):
            left, right = lower.copy(), upper.copy()
            left[axis] = right[axis] = value
            x1, y1 = screen(left)
            x2, y2 = screen(right)
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ddd"/>')
            x, y = (x1, 812.) if axis == 0 else (35., y1)
            parts.append(f'<text x="{x}" y="{y}" font-size="11">{value:g}</text>')
    return parts


def export(directory, output, truth=None, lap=None):
    """Write a new SVG, timestamped trajectory CSV, and optional ground-truth error JSON."""
    snapshots = read_estimates(directory, lap)
    truth = {} if truth is None else truth
    trajectory, landmarks = map_data(snapshots, truth)
    output = Path(output)
    if output.suffix.lower() != '.svg':
        raise ValueError('Output must have an .svg extension')
    targets = [output, output.with_suffix('.csv')]
    if truth:
        targets.append(output.with_suffix('.errors.json'))
    if any(path.exists() for path in targets):
        raise FileExistsError('Choose a new output name; export files already exist')
    with output.open('x') as target:
        target.write(make_svg(trajectory, landmarks, truth))
    with output.with_suffix('.csv').open('x', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['stamp_s', 'x_m', 'y_m', 'yaw_rad'])
        yaw_offset = truth.get('start_pose', [0., 0., 0.])[2]
        for record, position in zip(snapshots, trajectory):
            yaw = (record['state'][2]+yaw_offset+math.pi) % (2*math.pi)-math.pi
            writer.writerow([record['stamp'], *position, yaw])
    if truth:
        with output.with_suffix('.errors.json').open('x') as target:
            json.dump(compare_landmarks(landmarks, truth), target, indent=2)


def main(args=None):
    """Export a recorded run without requiring ROS, a camera, or plotting libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--output', required=True, help='New .svg filename')
    parser.add_argument('--ground-truth', help='JSON with measured start_pose and landmarks')
    parser.add_argument('--lap', type=int)
    options = parser.parse_args(args)
    truth = json.loads(Path(options.ground_truth).read_text()) if options.ground_truth else None
    if truth is not None and ('start_pose' not in truth or 'landmarks' not in truth):
        raise ValueError('Ground truth must include measured start_pose and landmarks')
    export(options.run_directory, options.output, truth, options.lap)
    print(f'Exported {options.output} and timestamped trajectory CSV')


if __name__ == '__main__':
    main()
