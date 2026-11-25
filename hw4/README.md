# HW4 — run 1

Run 1 plans around measured obstacles and starts from the configured pose.
It executes calibrated motion with optional known-map AprilTag corrections. It reuses HW3's motor driver, fixed motor mapper, motion
calibration, and camera calibration. The ROS interfaces in `run_commands.txt`
remain the entry points for subsequent iterations.

## Before running

1. Check `config/environment.json` against the physical setup. Supplied wall-tag
   coordinates, headings, start/goal, obstacle points, and 16 cm robot dimensions
   are entered in meters. The wall coordinates imply a 3.000 × 2.925 m workspace.
   The start heading is 180°, wall-tag center heights and printed sizes are both
   0.172 m, and the safety margin is 0.10 m.
2. Check the reused camera mount in `config/navigation.yaml`. Its offsets must
   describe the camera relative to the **robot footprint center** used in the
   map, which may differ from the old `base_link` origin. The initial values
   match HW3: `(0, 0, 0.2)` m and zero mounting rotation.
3. Inspect the plan using the preview-only block in `run_commands.txt`. It
   opens `results/preview/planned_paths.svg` locally without ROS or motors.
4. Set `navigation.mode` to `fast` (`safe` is an unimplemented skeleton) in that same file.
5. Place the robot at the measured start pose. Stop other motor/navigation
   stacks before starting HW4. Run the terminal commands in `run_commands.txt`.

The launcher validates measurements before starting hardware. Terminal 1 launches
camera, detection, motors, HW4, and recording together. Terminal 2 starts one
mission with `/hw4/start`; `/hw4/stop` cancels it. End terminal 1 with Ctrl+C before
copying results back. A second mission requires a new launch and timestamped
result directory. No robot access, commit, or push happens automatically.

At launch, the node saves the plot and prints the waypoint/turn table while
remaining idle. `/hw4/start` executes that same frozen route.

The start service reports whether the mission was **accepted**, not whether the
robot reached the goal. Follow `/hw4/status`, console messages, or `status.json`
for its eventual result. `complete` requires fresh, stable measured goal proximity.
`unverified` means the predicted route ended without visual confirmation.

The current test starts facing 180° and stops at the intermediate goal
**(75, 154) cm**, left of the obstacle. The original final goal is retained in
`final_goal` but is not executed. The first planned turn is about **14.9° right**.

## Measuring the map

Use meters for lengths and degrees for configured angles. Convert feet by
multiplying by 0.3048 and inches by 0.0254. Unknown values remain `null`.
JSON does not allow comments or trailing commas.

The four obstacle positions are **41h12 tag centers**, not physical corners.
They are now part of `landmarks`, alongside the 36h11 wall tags. IDs 1–4 have
confirmed face headings 180°, 90°, 0°, and 270°, respectively, center height
0.187 m, and printed size 0.185 m. The confirmed physical obstacle rectangle is
x = 1.14–1.76 m and y = 1.23–1.85 m; it is not a diamond.

One detector handles each family on `/detections/tag36h11` and `/detections/tag41h12`.
The repository calls the second detector family `Standard41h12`; `tags.py`
normalizes that spelling to `41h12`. Matching always uses `(family, ID)` and
unique TF names such as `tag_36h11_1` and `tag_41h12_1`.
`observations.py` combines both detector packets for the same camera capture,
so two families do not count as two independent stability samples. It waits at
most 0.15 s for the other detector before using available observations.

Choose a fixed overhead world frame: origin at the workspace's lower-left
corner, `+x` right, `+y` up in that view, `+z` above the floor. Positive yaw is
counterclockwise from `+x`: right = 0°, up = 90°, left = 180°.

| Config entry | Meaning |
| --- | --- |
| `workspace` | Interior width along x and height along y, with lower bounds `(0, 0)`. |
| `robot.length_m`, `robot.width_m` | Full footprint including protrusions; its rectangle center is the planning/localization reference point. |
| `robot.safety_margin_m` | Required extra robot-body clearance for localization and control errors. Choose a positive practical margin; this is a tuning choice, not a measured dimension. |
| `start` | Initial footprint-center position and forward yaw in the fixed world frame. |
| `goal` | Active test target: intermediate point (75, 154) cm, on the left side of the obstacle. Final heading is unrestricted. |
| `final_goal` | Original assignment goal (65, 229.5) cm, displayed as inactive and not executed during this test. |
| `obstacles` | Physical axis-aligned rectangle bounds **before** robot-size inflation. Duplicate entries for additional obstacles. |
| `landmarks` | Family (`36h11` or `41h12`), numeric ID, printed-tag center `(x, y, z)`, outward printed-face normal yaw, and tag edge size. Numeric IDs may repeat across families. |

Tags must be upright and vertical. `face_yaw_deg` describes the direction the
printed face points: toward `-y` is -90°. Measure the black square size used by
the detector, excluding the white paper border. Each detector automatically uses
its configured IDs and individual sizes; it does not assume a fixed tag count.

## Planning and movement

`grid.py` encloses the footprint in a circle of radius
`hypot(length_m, width_m) / 2`, covering all in-place rotations. Free grid
positions must preserve that radius, the physical safety margin, and a separate
0.05 m planning buffer. Continuous segment checks prevent corner cutting.

`config/navigation.yaml` selects the behavior with `navigation.mode: fast`.
`planner.py` dispatches the mode; `fast.py` contains the implemented distance A*
planner and constructs the existing calibrated waypoint follower. It connects
exact endpoints to the grid and reduces waypoints using collision-checked
line-of-sight shortcuts. This is an approximate shortest-distance route, not a
minimum-execution-time controller; stopped localization checks remain in place.

`safe.py` contains only documented `plan_path` and `create_navigator` signatures.
Safe planning and execution are intentionally unimplemented. Selecting
`navigation.mode: safe` raises a clear error before the launcher starts hardware;
it never falls back to fast mode. Preview and the mission compute only the
selected mode's route.

`localization.py` combines measured tag poses with timestamp-matched camera/tag
transforms. It rejects excessive range, implausible height/tilt, and disagreeing
tag solutions. It can use a single usable tag; it does not estimate a calibrated
pose covariance. `ros_localization.py` preserves capture timestamps and never
substitutes an older latest TF for the requested image timestamp.

`navigation.py` follows fixed waypoints. The current fast configuration sets
`navigation.feedback_required: false`, as requested for the physical test:

1. Begin from the configured start position and **180° heading** immediately
   after `/hw4/start`; do not wait to acquire a visual pose.
2. Execute the full turn toward the next waypoint using the calibrated motor
   pair and nominal angular rate. Motor commands remain active for the calculated
   duration; turn completion is **predicted**, not measured, in this mode.
3. Switch directly to `[0.4, 0.4]` forward travel for up to **one second**.
   Shorten travel near waypoints or a projected collision limit, including a
   watchdog interval of possible stopping overrun.
4. Stop for the configured settling interval, accept a fresh stable visual
   correction if available and consistent, then continue. Missing or rejected
   corrections do not block the next movement; their reason is logged.
5. Stop at the end of the predicted path with state `unverified` unless fresh
   stable landmark feedback confirms arrival within 0.08 m, yielding `complete`.

Without visual correction, wheel slip and imperfect turn/speed calibration cause
position and heading error to accumulate. Collision checks then protect the
**predicted** footprint; they cannot establish physical clearance. Place the robot
at the configured start pose for every run. This code does not detect unmapped
obstacles.

`feedback_required: true` retains the previous strict behavior for comparison:
initial/post-drive stable-pose acquisition, continuous visually measured turns,
and failure on lost feedback. This is not the default fast experiment. Both
settings retain command-refresh watchdog checks, collision checks on their pose
estimate, bounded motion budgets, explicit stop, and a mission timeout.

## Calibration baseline

`config/motion.yaml` contains the user-selected forward inputs for the current
surface and the new user-measured full-rotation turn values. The mapper and HW4
controller use the same file and HW3 calibration readers.

| Motion | Left input | Right input | Nominal model velocity |
| --- | --- | --- | --- |
| Straight | 0.4 | 0.4 | 0.567333 m/s (historical estimate) |
| Counterclockwise | -0.43 | 0.43 | +1.769911 rad/s (360° in 3.55 s) |
| Clockwise | 0.4 | -0.4 | -1.750191 rad/s (360° in 3.59 s) |

The new measured turn inputs and direction-specific rates replace the earlier
HW3 turn calibration. The user reported on November 24, 2026 that `[0.185, 0.15]` does not move
the robot on the current surface and selected `[0.4, 0.4]`. Do not revert to the
smaller pair. The 0.567333 m/s model comes from the older `robot_testing` mapper;
it needs physical remeasurement on the current surface. Reverse is uncalibrated. Smaller requested speeds do not slow the fixed-pair mapper. Commands are
refreshed at 20 Hz; the reused driver's watchdog is 0.15 s.

Camera intrinsics use the same installed HW3 calibration, respecting the existing
`CAMERA_CALIBRATION` environment override. The actual file is archived per run.
The camera launch enables GStreamer diagnostics (`GST_DEBUG=3` unless already
set) to expose startup failures before the driver reads its error bus. These
messages appear in the terminal and launch logs. This is diagnostic logging,
not a fix for a camera pipeline that cannot start.

## Data to bring back

Bring back the entire `hw4/results/<timestamp>/` directory, plus your description
and videos of the physical run. Results are excluded from Git.

- `planned_paths.json`, `planned_paths.svg`, and `planned_path.txt`: the selected
  route, numbered waypoints, segment lengths, absolute headings, signed turns,
  and minimum planned body clearance. The overhead plot labels every landmark
  by family/ID, draws its face normal, marks the start heading and robot radius,
  and distinguishes physical obstacles from excluded robot-center positions.
  Axes and the text table use centimeters to match the survey. These are
  planned coordinates, not a measured robot trajectory.
- `observations.csv`: capture timestamps, accepted/rejected feedback, rejection
  reasons, measured world poses, and tag IDs.
- `commands.csv`: state/waypoint, nominal commands, feedback age, measured pose,
  command-predicted pose, correction budgets, measured body clearance, turn
  target/progress angles, pose source, and optional-correction rejection reason.
- `status.json` and `events.jsonl`: iteration label `run1`, current/terminal
  state, explicit arrival status, elapsed time, and transition history.
- `environment.json`, `motion.yaml`, `navigation.yaml`, `camera_parameter.yaml`:
  effective config snapshots; `commit.txt` and `git_status.txt`: revision context.
- `telemetry/` and `ros_logs/`: rosbag and ROS logs. No raw video is recorded.

Command-predicted pose and `/cmd_vel_applied` are models, not measured odometry.
Landmark pose estimates and computed clearance are not physical ground truth.
The assignment's physical videos must be recorded separately.

## Local checks

From `robot_local/hw4/`, with NumPy and PyYAML installed:

```bash
python3 -m hw4.config config/environment.json
python3 -m hw4.preview config/environment.json --settings config/navigation.yaml --output results/preview
python3 -m unittest discover -s test -v
```

The preview never connects to hardware. `test/sample_environment.json` is explicitly
synthetic and is not the robot measurement config. Tests cover geometry, fast-mode planning, safe-mode rejection,
transform recovery, feedback rejection, failure handling, and ideal
closed-loop runs. They do not validate ROS wiring, camera latency, PnP accuracy,
motor calibration, slip, or physical clearance.

## Source files and rebuilds

`mission_node.py` handles ROS services, the timer, and log coordination. Pure
planning, geometry, localization, tag identity, and control remain in their own small modules.
`preview.py` uses the same planner offline. Hardware drivers remain reused HW3
dependencies rather than copied implementations.

Use `run_commands.txt` for the separate pull step followed by build/launch/start/stop. It rebuilds HW3 and
HW4 each time. Source measurements and motion calibration are passed explicitly;
`navigation.yaml` is installed by the build, so rebuild after changing it.
Restart the launch after changing any configuration. Keep the same branch on
both checkouts; only the user operates the robot and transfers result files.
