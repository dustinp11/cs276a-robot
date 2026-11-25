# HW4 development contract

- Work locally under `robot_local/hw4/`. The user handles robot access,
  pull/build/run, and result transfers. Do not SSH to the robot.
- Preserve `run_commands.txt` entry points throughout development:
  `launch/hw4.launch.py`, `hw4_navigation` in `hw4/mission_node.py`,
  `/hw4/start`, and `/hw4/stop`. Extend their internals as features are added.
- Keep physical measurements in `config/environment.json`; use
  `config/motion.yaml` as the shared source for mapper/controller calibration.
- Use forward `[0.4, 0.4]`, explicitly requested by the user on November 24,
  2026. The user observed that `[0.185, 0.15]` does not move the robot on the
  current surface. Do not restore that pair from HW3 defaults.
- Forward speed 0.567333 m/s is a historical estimate from the older mapper,
  not a new measurement on this surface. Keep mapper and pulse timing aligned;
  update this speed after physical calibration. Reverse remains uncalibrated.
- Preserve HW3 turns: CCW `[-0.35, 0.35]`, CW `[0.37, -0.37]`, nominal
  pi/2 rad/s magnitudes, unless the user supplies new turn calibration.
- Reuse HW3's calibration readers, mapper, and motor driver. Do not silently
  fall back to HW2 values or the overridden HW3 default CCW pair.
- HW3 mapping selects fixed motor pairs, not variable-speed motion. Keep
  forward and pivot commands separate, and refresh at 20 Hz for the 0.15 s
  watchdog. Nominal applied velocity is not measured odometry.
- Save configs, logs, trajectories, and explicit success/failure results under
  `hw4/results/<timestamp>/`. Do not commit result data. Keep hardware claims
  separate from local syntax and synthetic checks.

## Run 1

- `config/navigation.yaml` selects fast/safe and contains controller,
  localization, and reused HW3 camera-mount settings. The launcher loads it
  from the installed package; rebuild after editing.
- `preview.py` and the ROS mission use the same selected mode. Preserve
  continuous footprint collision checks during waypoint reduction.
- `fast.py` is implemented; `safe.py` contains function signatures/docstrings
  only. Do not implement safe mode until the user requests it. Selecting safe
  must report unimplemented before hardware starts, without falling back to fast.
- Navigation turns continuously to the full next-waypoint heading, then
  transitions directly to forward travel without intermediate turn stages or
  a stopped acquisition between turn and drive. Do not restore turn pulses.
  Drive for up to one second before stopped pose checks; shorten only near
  waypoints or collision limits. Preserve fresh feedback and failure timeouts. Do not substitute command prediction for measured
  arrival, or silently continue without feedback.
- Preserve successful/failed terminal outcomes when stop or shutdown follows.
- Label this iteration `run1` in archived status. The synthetic test map is
  never a replacement for `config/environment.json` measurements.
- For blocks of five or more consecutive code lines, begin with a short
  lowercase intent comment and separate the next logical block with a blank line.
- Wall tags use 36h11; obstacle tags use 41h12, called `Standard41h12` by the
  detector. Match by `(family, ID)` and preserve unique family-aware TF frames.
  Combine detector packets per camera capture before counting stable samples.
