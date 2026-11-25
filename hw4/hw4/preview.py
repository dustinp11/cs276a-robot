"""Plan and visualize the selected mode offline without ROS or motor access."""

import argparse
from hw4.config import load_validated_environment
from hw4.grid import build_grid
from hw4.planner import get_mode
from hw4.results import save_plans
from hw4.plan_plot import plan_description
from hw4.settings import load_settings


def prepare_plans(environment, policy):
    """Build a grid and plan the configured mode from the measured start."""
    # select behavior before constructing any paths
    mode = get_mode(policy['mode'])
    grid = build_grid(environment, policy['grid_resolution_m'], policy['planning_buffer_m'])
    start = tuple(environment['start'][f'{axis}_m'] for axis in ('x', 'y'))
    goal = tuple(environment['goal'][f'{axis}_m'] for axis in ('x', 'y'))
    plans = {policy['mode']: mode.plan_path(grid, start, goal)}
    return grid, plans


def main():
    """Create planned_paths.json and SVG for a measured environment."""
    # keep offline planning separate from hardware
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--settings', default='config/navigation.yaml')
    parser.add_argument('--output', default='results/preview')
    args = parser.parse_args()
    environment = load_validated_environment(args.config)
    grid, plans = prepare_plans(environment, load_settings(args.settings)['navigation'])
    save_plans(args.output, grid, environment, plans)
    print(plan_description(environment, plans))
    print(f'Open {args.output}/planned_paths.svg; no hardware accessed.')


if __name__ == '__main__':
    main()
