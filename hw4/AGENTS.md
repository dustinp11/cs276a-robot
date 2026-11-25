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
- Start fast mode from the configured pose without mandatory initial visual
  acquisition. Execute full calibrated timed turns and up to one-second drives.
  The latest user request is **NO 360 / NO TAG SEARCH**, correction only at the
  END. Default to `visual_feedback: at_goal`, `tag_search_enabled: false`, and
  `feedback_required: false`. Follow the entire frozen path using prediction,
  then stop for visual correction only within the predicted final goal tolerance.
  Keep correction active once enabled, even after a large accepted position update.
  Cap correction drives to 0.10 m and half the remaining distance, checking fresh
  stopped poses after each. Preserve the calibrated motor inputs.
  Never start a search when the search flag is false. Missing final feedback must
  produce an unverified predicted finish; ordinary turns toward the goal remain.
  Other feedback modes and search are optional baselines, not this run's defaults.
- At waypoint checks, accept stable bounded corrections even when command
  prediction drifted by more than the usual optional correction gate. Keep the
  same target and correct when feedback is usable. If feedback stays missing,
  resume predicted travel; mark any predicted waypoint advancement unverified.
  Correction, collision, watchdog, and mission limits still fail explicitly.
- Goal success requires all confirming fresh observations within 0.05 m of the
  active goal. The 0.03 m threshold caused oscillations around a 3–5 cm error in
  the physical run; do not restore it without improved calibration. This is measured localization tolerance, not exact ground truth.
- If measured drift placed the robot inside the 0.10 m safety margin, allow only
  outward recovery preserving its measured starting clearance, never below the
  configured recovery floor. Log that temporary clearance and the correction.
- Log observed per-tag ranges, target/goal distances, and measured versus
  predicted pose. A predicted-only route end must report `unverified`, never
  measured goal success. A visually confirmed goal may report `complete` while
  preserving the list of earlier unverified waypoint advances.
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

- The robot starts facing 180 degrees and must navigate to the assignment goal
  (0.65, 2.295) m. The earlier (0.75, 1.54) m intermediate test target is retired.
  Keep a single active `goal`; do not substitute a test endpoint for this goal.

- For near-goal refinement, reject corrections over 0.12 m after a measured
  checkpoint already placed the robot within 0.20 m. The physical log switched
  between stationary pose clusters about 0.24 m apart; do not chase those jumps.
  The latest user request replaces unconditional averaging with outlier rejection.
  Use a unique mutually agreeing strict majority, at least two tags, with 0.20 m
  position and 20 degree heading gates. Reject lone or ambiguous estimates; do
  not guess which of two conflicting tags is correct. Average retained positions
  equally and use circular heading. Log each candidate and its fusion status.
- Confirmed obstacle tag detection-square size remains 0.185 m; the user briefly
  said 0.18 m then corrected it back to 0.185 m. Do not apply a whole-pattern ratio.


- `hw4.launch.py localization_only:=true` records manual fixed-center rotation
  without launching motor drivers, velocity mapping, or navigation. Preserve the
  same PoseSource and Localizer used by navigation. Log the supplied physical
  reference separately from estimates; do not claim measured heading or full
  rotation coverage from estimated yaw. Archive per-tag errors and a position plot.

- The camera rectification fix lives in the shared `rubikpi_ros2/robot_vision`
  driver: published K/P must match the remap matrix, with zero D and identity R
  for the already rectified output. Rebuild `robot_vision_camera` in the SDK
  workspace before sourcing it; building only hw3/hw4 leaves the old camera code.
- The latest manual rotation reference is the original final goal (0.65, 2.295) m.
  This diagnostic reference does not change the active navigation goal.
