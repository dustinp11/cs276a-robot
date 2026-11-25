"""Summarize a fixed-position, manually rotated localization recording without ROS."""

import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

from hw4.grid import build_grid
from hw4.plan_plot import axes, background, landmarks, screen, text

COLORS = ('#1769cc', '#d33b32', '#16834c', '#8438a5', '#b46c00', '#009aab',
          '#d33884', '#627000', '#4a41aa', '#885037', '#00704f', '#555555')
FIELDS = ('tag', 'samples', 'mean_x_cm', 'mean_y_cm', 'bias_cm', 'rmse_cm', 'max_error_cm')


def tag_summary(tag, rows):
    """Return position-error statistics; repeated frames are not independent accuracy trials."""
    # compare independent tag solutions with reference
    xs = [float(row['x_m']) for row in rows]
    ys = [float(row['y_m']) for row in rows]
    errors = [float(row['position_error_m']) for row in rows]
    dx = sum(float(row['dx_m']) for row in rows) / len(rows)
    dy = sum(float(row['dy_m']) for row in rows) / len(rows)
    values = (tag, len(rows), sum(xs) / len(xs) * 100, sum(ys) / len(ys) * 100,
              math.hypot(dx, dy) * 100, math.sqrt(sum(e*e for e in errors) / len(errors)) * 100,
              max(errors) * 100)
    return dict(zip(FIELDS, values))


def scatter_plot(environment, reference, groups, summaries):
    """Plot per-tag robot positions on the measured map; axes use centimeters."""
    # retain map geometry and family labels
    grid = build_grid(environment)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="1000" font-family="Arial,sans-serif">',
             '<rect width="1250" height="1000" fill="white"/>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
             text(40, 35, 'Fixed-position localization: individual tag estimates', 23),
             text(40, 65, 'Cross: user-supplied robot center. Colored dots: estimated robot positions.', 15),
             text(40, 90, 'Heading is estimated, not independently measured. Plot is not a robot trajectory.', 15)]
    parts += background(grid) + axes(grid) + landmarks(grid, environment)

    # distinguish each tag without connecting samples
    for index, (tag, rows) in enumerate(sorted(groups.items())):
        color = COLORS[index % len(COLORS)]
        for row in rows:
            x, y = screen((float(row['x_m']), float(row['y_m'])))
            parts.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{color}" opacity="0.5"/>')
        stats = summaries[tag]
        parts.append(text(850, 200 + index*35, f'{tag}: n={len(rows)}, RMSE={stats["rmse_cm"]:.1f} cm', 14, color=color))

    # mark the physically measured reference center
    x, y = screen(reference)
    parts += [f'<path d="M {x-10} {y} H {x+10} M {x} {y-10} V {y+10}" stroke="black" stroke-width="3"/>',
              text(40, 935, f'Reference center: ({reference[0]*100:.1f}, {reference[1]*100:.1f}) cm', 16), '</svg>']
    return '\n'.join(parts)


def write_report(directory):
    """Write per-tag summary CSV and SVG from saved records, including interrupted runs."""
    # recover archived geometry and recorded reference
    directory = Path(directory)
    reference = json.loads((directory / 'localization_reference.json').read_text())['xy_m']
    environment = json.loads((directory / 'environment.json').read_text())
    groups = defaultdict(list)
    with (directory / 'per_tag.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['x_m']:
                groups[row['tag']].append(row)

    # summarize only usable independently estimated positions
    summaries = {tag: tag_summary(tag, rows) for tag, rows in sorted(groups.items())}
    with (directory / 'tag_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summaries.values())
    (directory / 'tag_positions.svg').write_text(scatter_plot(environment, reference, groups, summaries))
    return summaries


def main():
    """Regenerate a report locally after copying a robot result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    for tag, stats in write_report(args.directory).items():
        print(f'{tag}: {stats["samples"]} samples, RMSE {stats["rmse_cm"]:.1f} cm')


if __name__ == '__main__':
    main()
