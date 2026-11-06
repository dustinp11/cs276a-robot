# HW3 recorded-map evaluation

The live node saves `parameters.json` and `run.jsonl` in a new timestamped directory
under `~/hw3_runs`. It logs estimated poses, dynamically discovered landmark order,
full joint covariance, accepted/rejected range/bearing observations, applied-model
motion, commands, lap snapshots, and camera calibration. This is the robot's estimate,
not a hand-drawn or planned trajectory.

After a run, with the HW3 overlay sourced:

```bash
ros2 run hw3_code plot_run ~/hw3_runs/RUN_DIRECTORY --output square.svg
ros2 run hw3_code plot_run ~/hw3_runs/RUN_DIRECTORY --lap 1 --output lap1.svg
```

Each export creates a standalone SVG map and a timestamped trajectory CSV. Use new
output names. The SVG includes estimated landmark positions and 95% covariance
ellipses; open it in a browser. No ROS runtime is required for the export module:

```bash
PYTHONPATH=hw3 python3 -m hw3_code.plot_run RUN_DIRECTORY --output square.svg
```

For measured-world alignment and average landmark error, supply a JSON file:

```json
{
  "start_pose": [0.0, 0.0, 0.0],
  "landmarks": [
    {"family": "41h12", "id": 12, "x_m": 1.0, "y_m": 2.0}
  ]
}
```

Those numbers are schema examples only. Replace them with measured world-frame
start x/y in meters, yaw in radians, and actual landmark centers. Use the same robot
reference point and marker center definition as the filter. Keep this file out of
estimator configuration.

```bash
ros2 run hw3_code plot_run RUN_DIRECTORY --ground-truth measured.json --output world.svg
```

This transforms estimates using the measured start pose, rotates the landmark
covariances, overlays ground truth, and writes `world.errors.json` with matched-ID
average Euclidean error, individual errors, and missing/spurious IDs. It does not
fit an alignment to the final map. Start-pose measurement uncertainty is not added
to the plotted covariance; the alignment is treated as fixed.
