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
- Use the November 24 user turn calibration: CCW `[-0.43, 0.43]` takes
  3.55 s per full rotation; CW `[0.4, -0.4]` takes 3.59 s. The positive rate
  magnitudes are `2*pi/3.55` and `2*pi/3.59` rad/s. Do not restore HW3 turns.
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
- The user explicitly requested movement without mandatory pose acquisition.
  Default fast mode has `feedback_required: false`: start from the configured
  pose, execute full calibrated timed turns, drive up to one second, then check
  optional stopped visual corrections. Do not block motion on missing tags.
- Log command prediction versus accepted visual correction and rejection reason.
  Predicted route completion must end as `unverified`, never measured arrival.
  Keep strict feedback mode available for comparison, not enabled by default.
- Preserve successful/failed terminal outcomes when stop or shutdown follows.
- Label this iteration `run1` in archived status. The synthetic test map is
  never a replacement for `config/environment.json` measurements.
- For blocks of five or more consecutive code lines, begin with a short
  lowercase intent comment and separate the next logical block with a blank line.
- Wall tags use 36h11; obstacle tags use 41h12, called `Standard41h12` by the
  detector. Match by `(family, ID)` and preserve unique family-aware TF frames.
  Combine detector packets per camera capture before counting stable samples.

- Generate and print the frozen plan during launch initialization, before
  `/hw4/start` can move the robot. Preview and execution must use the same path.
  Keep a motor-free local preview command and label planned versus measured data.

- The current physical test starts at 180 degrees and stops at intermediate
  goal (0.75, 1.54) m, left of the obstacle. The original final goal is saved
  in `final_goal` and must not be executed automatically during this test.
