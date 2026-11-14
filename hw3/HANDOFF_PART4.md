# HW3 handoff: current robot code and Part 4 experiments

Updated 2026-11-13. Verified baseline: branch `hw3_dustin`, commit `8622f70`
(`add pauses 1 second`). The repository was clean and matched
`origin/hw3_dustin` before this document was added. This handoff does not implement
new navigation behavior, commit changes, or update the robot.

## Read first: scope and assignment requirements

Read the workspace `../AGENTS.md` (relative to the repository root), the actual
assignment `../assignment_writeups/hw3-2026.pdf`, and the current source files.
Workspace `../context.md` describes historical HW2 code and calibration;
`hw3_agents.md` is an early, now superseded HW3 plan. Neither describes the active
octagon. The beginning of `hw3/README.md` describes the active manual octagon;
later sections describe optional feedback behavior that it bypasses.

The next agent will work on additional trajectories, potentially using estimated
AprilTag positions to guide motion. A square is a possible choice, not a decision
already made. Keep this proposed extension distinct from the assignment itself:

- Part 3 (page 1): build maps from an unknown starting pose using a square and an
  eight-point octagon; do not give SLAM landmark positions or even the number of
  landmarks beforehand. Discover landmarks dynamically.
- Part 4 (page 2): compare those maps; measure the starting pose and transform the
  estimated map into the world frame for comparison with the surveyed map;
  measure average landmark error; investigate whether repeated laps change results.
- Parts 6–7: explain initialization, models, noise and landmark lifecycle; show
  estimated trajectory, landmark positions and landmark covariance; discuss
  repeated-lap improvement.

**Part 4 does not explicitly require navigation with a known map.** If using
“assumed tag positions,” distinguish a map estimated during an earlier lap from
surveyed ground truth. Prefer reusing the learned map for subsequent navigation
without injecting surveyed positions into the unknown-map SLAM experiment.
Record which map/frame is used and whether landmarks remain estimated or fixed.
Keep surveyed data available for evaluation separately. Do not silently turn the
experiment into known-map localization or claim fitted alignment measures the
physical starting pose.

## Workspace, deployment, and coding expectations

- Mac workspace: `/Users/dustinpham/ucsd/fall2026/cs276a`.
- `robot_local/` is the robot Git repository root. Its `robot_testing/` directory
  is the historical inner Python package, not another deployment root.
- Robot checkout observed throughout testing: `/home/ubuntu/robot_testing`
  (`~/robot_testing` as user `ubuntu`). This is the same repository as local
  `robot_local/`, with a different directory name. Other workspace directories
  are not automatically present on the robot.
- Hardware: headless RubikPi 3, ROS 2 Jazzy. The Mac does development and plotting;
  an Ubuntu ARM64 UTM VM can run offline RViz. No live RViz is required on the robot.
- Edit robot code in `robot_local/`; reuse existing helpers and keep new files
  minimal. Keep functions short. For functions longer than 10 lines, use complete
  docstrings with purpose, `Args:` and `Returns:`; state units and frames.
- Do not commit/push without user authorization. Local edits do not reach the
  robot until committed, pushed, pulled, built and relaunched by the user.
- Do not claim hardware validation from local tests. The user physically tests
  and supplies CSVs, observations and navigation logs.

## Active behavior: simple timed octagon

The current octagon deliberately ignores SLAM/vision for motor decisions:

**45° left → forward 37.5 cm → stop 1 s → forward 37.5 cm → stop 1 s → repeat eight times → finish.**

It turns before each side, including the first side. There is no deliberate
pause between the turn and forward motion. After the eighth side and final
pause, it completes without a ninth turn. Forward distance is integrated from
nominal commanded speed, excluding stopped time; it is not measured wheel travel.
At 20 Hz, each half takes approximately 1.85 seconds of motor-on time.

There is **no startup 360, no late center-facing scan, no homing to the start,
no heading correction and no midpoint distance correction** in this profile.
Camera, AprilTag detection, SLAM and CSV recording remain active. The midpoint
pause gives the camera time to process but does not change motor commands.
Starting/continuing does not require two tags or any tags to be visible.

The generic controller retains old feedback and scan code for other profiles.
Do not assume a function's presence means the manual octagon calls it. Likewise,
do not enable all those experiments at once when adding a new mission.

### Calibration and where to edit it

Motor arrays are `[left, right]`; these are driver inputs, not m/s.

| Motion | Motor inputs | Nominal model / duration |
| --- | --- | --- |
| Forward | `[+0.185, +0.15]` | `0.203 m/s` |
| Active octagon left pivot | `[-0.35, +0.35]` | `pi/2 rad/s`, `0.5 s` for a nominal 45° |
| Clockwise correction, if enabled elsewhere | `[+0.37, -0.37]` | `pi/2 rad/s` |
| Generic square left pivot | `[-0.40, +0.37]` | `pi/2 rad/s`; separate square timing |

The robot tended to drift left because the right wheels were stronger. Forward
speed was measured as 203 cm in 10 seconds at `[0.18, 0.15]`; left input was then
raised to `0.185`. Speed has not been remeasured at the new trim. Treat `0.203`
as a provisional model. Physical turns vary with slip, friction and pulse length.
Small inputs have deadzones; short turn pulses may not produce useful motion.

Shared calibration lives in `hw3/config/hw3_params.yaml`, but the octagon
dictionary in `hw3/launch/hw3_navigation.launch.py` overrides its turn settings:

```python
'navigation.manual_motion_enabled': True,
'navigation.manual_forward_pause_s': 1.0,
'navigation.manual_midpoint_pause_s': 1.0,
'navigation.startup_scan_enabled': False,
'navigation.side_count': 8,
'navigation.side_length_m': 0.75,
'navigation.late_scan_start_corner': 99,
'navigation.corner_turn_duration_s': 0.5,
'navigation.corner_stage_angle_rad': 0.7853981633974483,
'navigation.corner_ekf_weight': 0.0,
```

Change **that launch override** to tune the active turn duration, then rebuild
and relaunch. Editing only the YAML square duration will not affect this profile.
Both mapper and navigator receive matching octagon turn calibration overrides.
Do not change the physical motor pair without also revisiting its nominal rate.

## Code map and estimator behavior

Paths below are relative to the robot repository root.

| File | Role |
| --- | --- |
| `hw3/launch/hw3_octagon.launch.py` | Wrapper selecting `mission:=octagon` in the main launch. |
| `hw3/launch/hw3_navigation.launch.py` | Includes camera/detector; launches estimation, mapping, driver and navigation; sets profile overrides. Default mission without the wrapper is **square**. |
| `hw3/hw3/nodes/navigation_node.py` | ROS adapter, parameters, 20 Hz timer, services and navigation CSV. Executable `navigation_node`, ROS name `square_navigation_node` even for octagons. |
| `hw3/hw3/square_navigation.py` | Shared polygon controller, class `SquareNavigator`. `_manual_step`, `_manual_begin_turn`, `_manual_turn`, `_manual_drive`, `_manual_checkpoint`, `_manual_midpoint_pause`, `_manual_pause` implement the active sequence. |
| `hw3/hw3/vision_navigation.py` | Policy, visual histories, confidence/stability gates and geometry for optional feedback. |
| `hw3/hw3/late_navigation.py` | Old late relocalization/homing helpers; inactive in manual octagon. |
| `hw3/hw3/ekf_slam.py` | Pure EKF, dynamically augmented landmark state and full joint covariance. Relative landmark covariance includes robot–landmark correlations. |
| `hw3/hw3/nodes/ekf_slam_node.py` | Detection filtering, timestamped transforms, EKF updates, pose/TF/markers/feedback and CSV exports. |
| `hw3/hw3/visual_odometry.py`, `nodes/visual_odometry_node.py` | Relative visual motion estimation; the navigation launch selects command prediction instead. |
| `hw3/hw3/turn_calibration.py` | Shared fixed forward/pivot calibration objects. |
| `hw3/hw3/velocity_mapping.py`, `nodes/velocity_mapping.py` | Pure mapping and ROS adapter: `/cmd_vel` → fixed motor pairs and `/cmd_vel_applied`. |
| `hw3/hw3/nodes/motor_control.py` | Serial motor driver, shutdown stop and command watchdog. |
| `hw3/hw3/data_recorder.py` | CSV export helpers. |
| `hw3/hw3/nodes/publish_map_node.py` | Offline saved-CSV map/path visualization, not a navigation map loader. |
| `hw3/setup.py` | Installed configs, launch files and console entry points. |

There is no separate octagon navigation class/file: the polygon parameters select
eight sides in the existing square-named controller.

Navigation launch overrides EKF prediction to `cmd_vel` on `/cmd_vel_applied`.
That topic contains the mapper's **nominal applied velocity**, not encoder
odometry or measured ground truth. VO and command motion are not both integrated
for the same interval. Tag observations still update robot and landmark states.
Forward mapping holds its fixed pair and suppresses angular steering while
moving. Reverse requests stop because reverse is not calibrated.

SLAM begins in an arbitrary local frame, conventionally robot `(0,0,0)`, with
no surveyed tag positions. Small initial pose covariance anchors that local gauge;
it does not mean the physical starting world pose is known. Landmarks are added
as detected. Coordinates may be negative legitimately. Camera optical axes are
right/down/forward; robot axes are forward/left/up. Use `coordinates.py` helpers
instead of swapping axes based on an RViz screenshot.

Known-map HW2 localization in `robot_testing/hw2_localization.py` may be useful
as a reference, but its old map/calibration should not be copied as HW3 truth.
The current Part 4 map-reuse/navigation path has **not** been implemented.

## Run on the robot: exact terminals

First stop previous launch sessions with Ctrl+C. Use one complete stack per run.
After authorized changes have been committed and pushed on the Mac, update on
the robot (not from Mac `~/robot_runs`, which is not a Git checkout):

```bash
cd ~/robot_testing
git switch hw3_dustin
git pull --ff-only origin hw3_dustin
```

If checkout/pull reports local changes, inspect `git status` first. Preserve
results/edits with `git stash push -u -m "robot changes before update"` if needed.
Do not discard run files unless the user says they are unwanted. Generated CSVs
have been tracked on some branches and can block switching. Avoid accidental
merge commits and broad resets when the Mac branch is the intended source.

**Robot terminal 1: build and launch the complete current octagon.**

```bash
cd ~/robot_testing
source /opt/ros/jazzy/setup.bash
source rubikpi_ros2/install/setup.bash
colcon build --base-paths hw3 --packages-select hw3 &&
source install/setup.bash &&
ros2 launch hw3 hw3_octagon.launch.py
```

`--base-paths hw3` matters: the root is also a package, and plain package
selection previously failed to discover nested HW3. The `rubikpi_ros2` setup
provides the existing robot camera and AprilTag dependencies. The main launch
starts the camera and **36h11** detector; do not separately launch duplicate
camera, detector, motor driver or controller nodes. Motion waits for the service.

**Robot terminal 2: inspect, then start.**

```bash
source /opt/ros/jazzy/setup.bash
source ~/robot_testing/install/setup.bash
ros2 topic info /ekf_slam/pose --verbose
ros2 service call /start_navigation std_srvs/srv/Trigger '{}'
```

Expect exactly **one** pose publisher. Stop explicitly with:

```bash
ros2 service call /stop_navigation std_srvs/srv/Trigger '{}'
```

Ctrl+C terminal 1 stops the launch and triggers final estimator export. Keep the
driver watchdog and stop service. The user supervises physical motion and does
not want automatic wall-distance stops or tag-visibility requirements added to
these experiments. Do not remove normal command-loss/shutdown motor stopping.

For another independent attempt, end the stack, archive results, then relaunch
fresh. A future repeated-lap experiment should intentionally keep the **same
SLAM state** across laps; implement that explicitly rather than mixing separate
unrelated runs or restarting the estimator and calling it repeated-map learning.

### Duplicate publisher incident

Two `ekf_slam_node` publishers with different GIDs were actually observed on
`/ekf_slam/pose`. Alternating estimates caused impossible jumps while stopped.
Stop all old ROS launch sessions and launch exactly once. Check `/cmd_vel`,
`/cmd_vel_applied`, and `/motor_commands` publisher counts too if behavior is odd.
`Unknown topic` usually means the stack is not running or the shell environment
does not match. After fixing duplicates the robot still slipped: duplicates were
a real problem, not an explanation for every remaining turn error.

## Results: avoid the two-directory trap

Output paths default to `hw3_results` **relative to launch working directory**.
Using terminal 1 above writes `~/robot_testing/hw3_results/`. Launching from the
inner `hw3/` directory instead writes `~/robot_testing/hw3/hw3_results/`.
Both have existed. Check modification times and timestamps before choosing data.

```bash
ls -lh ~/robot_testing/hw3_results/
head -n 2 ~/robot_testing/hw3_results/ekf_slam_trajectory.csv
tail -n 2 ~/robot_testing/hw3_results/ekf_slam_trajectory.csv
```

- `ekf_slam_trajectory.csv`: `timestamp_s,x_m,y_m,theta_rad,sigma_x_m,sigma_y_m,sigma_theta_rad`.
- `ekf_slam_landmarks.csv`: `tag_id,x_m,y_m,sigma_x_m,sigma_y_m,cov_xy`.
- `navigation_log.csv`: state, target/estimated pose, commands and feedback/error
  diagnostics. Use it to distinguish a skipped correction from an executed one.

Positions/std deviations are meters; angles are radians; `cov_xy` is m².
EKF exports every 10 seconds and on shutdown. Trajectory records cover the
estimator's lifetime, including waiting between missions. The landmark CSV holds
latest estimates; previously logged trajectory samples are not retrospectively
smoothed after a later map correction. Archive the three files together after
shutdown, before a new launch can overwrite them.

## Plotting and ground truth

Plotting lives outside the robot repository, in workspace `vis_code/`.
Put the latest trajectory and landmark CSVs directly into `vis_code/data/`.
The surveyed map is `vis_code/data/walls.txt`. From `vis_code/`:

```bash
python plot.py
```

This reads the two CSVs and walls, saves `octagon_positions.png`, and prints
only mean landmark error and mean path error, both in centimeters. `plot.py`
uses shared `plot_walls.py` logic; `python plot_walls.py` instead draws the
reference square. The existing octagon reference vertices, supplied in cm, are:

```text
(106,62), (54,115), (54,190), (106,242),
(181,242), (234,190), (234,115), (181,62)
```

The actual arena is **3 × 3 meters**, not the earlier assumed 8 × 8 feet.
Surveyed `walls.txt` is `tag_id,x_m,y_m,facing_rad`:

```text
0, 3, 1,  3.1416
7, 3, 2,  3.1416
1, 2, 3, -1.5708
3, 1, 3, -1.5708
4, 0, 2,  0
8, 0, 1,  0
9, 1, 0,  1.5708
5, 2, 0,  1.5708
```

Current evaluation fits one rigid 2D rotation + translation from matched final
tag IDs to surveyed positions, without scale/reflection. It applies the same
transform to all trajectory positions and rotates covariance with `R C Rᵀ`.
Reference path is black/dotted for the octagon; trajectory is red; estimated tags
include covariance ellipses. Archives are under `data/square_runs/` and
`data/octo_runs/`; do not assume the highest-numbered archive is the user's newest
data when they have just pasted replacements directly into `data/`.

Important limitations for Part 4:

- Fitted landmark residuals reuse the evaluation tags to choose alignment;
  they are not independent validation. The writeup instead asks for a measured
  starting pose. Support/report that measured transform when available, and
  clearly label any fitted alternative.
- Mean path error is nearest-reference-edge distance over trajectory samples,
  including stopped samples. It is not time-synchronized physical tracking error,
  and cannot prove correct order, direction or completion.
- Plot outlier filtering is a display choice; current mean path error includes
  all trajectory samples. Do not hide bad measurements to improve reported metrics.
- Covariance is model confidence, not ground truth. A good final landmark map
  can coexist with a poor historical trajectory or poor physical driving.

## Lessons to carry into the next implementation

- Odometry is poor; the command model also cannot measure slip. Vision is often
  better but has latency, low frame rate, occasional bad detections and EKF jumps.
- Timed turns sometimes under/overshoot. Strong global EKF heading correction
  sometimes made a bad turn worse. Do not hard-code corner 6 compensation from
  one run or use a delayed observation as instantaneous heading truth.
- Global landmark covariance can be high while robot-relative uncertainty is
  small. `get_relative_landmark_covariance()` includes cross-covariance and is a
  useful existing helper. Confidence alone does not remove systematic map error.
- Previous confidence gates often skipped corrections because observations were
  stale or heading uncertainty was too high. No visibility assumption should
  indefinitely prevent mission progress. Small corrections should permit either
  direction, use calibrated motor magnitudes and bounded attempts.
- Startup full rotations, staged rotations and matching the original view were
  tried repeatedly and explicitly scrapped. Late 90° center-facing look-arounds
  and homing at corners 6–8 also performed badly and were disabled. They remain
  dormant code, not an accepted design to reactivate silently.
- Keep the simple octagon as a baseline. For a map-guided or repeated-lap mission,
  make the new behavior explicit and preserve a way to run this baseline. Choose
  a consistent map frame and deal with SLAM map corrections before steering toward
  a point calculated from an earlier pose/map snapshot.
- No Part 4 repeated-lap controller or learned-map initialization has been added
  in this handoff. Decide how to retain estimator state, segment laps in logs,
  handle missing observations and compare per-lap map uncertainty/error. Reuse
  existing geometry, calibration, logging and ROS adapters wherever practical.

## Local verification

From the robot repository root on the Mac:

```bash
../.venv/bin/python hw3/run_tests.py
```

Verified for this handoff: **132 tests passed**. These cover synthetic estimator,
geometry, mapping and controller behavior, including the manual octagon. They do
not test actual ROS discovery, camera latency, serial hardware, friction or
physical distances/angles. Run appropriate local checks after implementation;
the user builds and tests hardware using the robot terminals above.
