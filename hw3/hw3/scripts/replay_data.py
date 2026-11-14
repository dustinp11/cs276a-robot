#!/usr/bin/env python3
"""Replay recorded AprilTag observations through Visual Odometry and EKF-SLAM."""

import argparse
from pathlib import Path
import sys

from hw3.coordinates import CameraExtrinsics
from hw3.data_recorder import DataReplayer, export_landmarks_csv, export_trajectory_csv
from hw3.ekf_slam import EkfSlam
from hw3.visual_odometry import VisualMotionEstimator


def main(args=None):
    """Run this module through its command-line entry point.

    Args:
        args: Optional command-line arguments; None uses the process arguments.

    Returns:
        None. State changes and publications are performed in place.
    """
    parser = argparse.ArgumentParser(description="Replay recorded data through VO and EKF-SLAM.")
    parser.add_argument('input', help="Path to input recorded JSON file")
    parser.add_argument('--output-dir', '-o', default='hw3_results', help="Directory to save exported CSVs")
    parser.add_argument('--use-vo', action='store_true', default=True, help="Use visual odometry for prediction")
    parser.add_argument('--use-cmd', action='store_true', default=True, help="Use cmd_vel as fallback prediction")
    parsed = parser.parse_args(args=args)

    input_path = Path(parsed.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(parsed.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading recordings from: {input_path}")
    replayer = DataReplayer(input_path)
    print(f"Loaded {len(replayer.records)} recorded frames. Processing...")

    results = replayer.run(use_vo=parsed.use_vo, use_cmd_fallback=parsed.use_cmd)

    traj = results['trajectory']
    lms = results['landmarks']
    vo_history = results['vo_history']

    num_vo_valid = sum(1 for v in vo_history if v.valid)
    print("\nReplay Complete:")
    print(f"  Processed frames: {len(traj)}")
    print(f"  Valid VO steps:   {num_vo_valid}/{len(vo_history)}")
    print(f"  Discovered tags:  {list(lms.keys())}")

    if traj:
        final_pose = traj[-1]
        print(f"  Final Robot Pose: x={final_pose['x']:.3f}m, y={final_pose['y']:.3f}m, "
              f"theta={final_pose['theta']:.3f}rad (+/- {final_pose['sigma_theta']:.3f})")

    print("\nEstimated Landmark Positions:")
    for tag_id, lm in sorted(lms.items()):
        print(f"  Tag {tag_id:2d}: ({lm['x']:6.3f}, {lm['y']:6.3f}) m | "
              f"sigma_x={lm['sigma_x']:.3f}m, sigma_y={lm['sigma_y']:.3f}m")

    traj_csv = output_dir / "replay_trajectory.csv"
    map_csv = output_dir / "replay_landmarks.csv"
    export_trajectory_csv(traj, traj_csv)
    export_landmarks_csv(lms, map_csv)
    print(f"\nExported trajectory to: {traj_csv}")
    print(f"Exported landmark map to: {map_csv}")


if __name__ == '__main__':
    main()
