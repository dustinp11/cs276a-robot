# HW4 — run 1

Run 1 plans around measured obstacles and follows one route using known-map
AprilTag localization. It reuses HW3's motor driver, fixed motor mapper, motion
calibration, and camera calibration. The ROS interfaces in `run_commands.txt`
remain the entry points for subsequent iterations.

## Before running

1. Check `config/environment.json` against the physical setup. Supplied wall-tag
   coordinates, headings, start/goal, obstacle points, and 16 cm robot dimensions
   are entered in meters. The wall coordinates imply a 3.000 × 2.925 m workspace.
   The start heading is 90°, wall-tag center heights and printed sizes are both
   0.172 m, and the safety margin is 0.10 m.
2. Check the reused camera mount in `config/navigation.yaml`. Its offsets must
   describe the camera relative to the **robot footprint center** used in the
   map, which may differ from the old `base_link` origin. The initial values
   match HW3: `(0, 0, 0.2)` m and zero mounting rotation.
3. Set `navigation.mode` to `fast` (`safe` is an unimplemented skeleton) in that same file.
4. Place the robot at the measured start pose. Stop other motor/navigation
   stacks before starting HW4. Run the terminal commands in `run_commands.txt`.

The launcher validates measurements before starting hardware. Terminal 1 launches
camera, detection, motors, HW4, and recording together. Terminal 2 starts one
mission with `/hw4/start`; `/hw4/stop` cancels it. End terminal 1 with Ctrl+C before
copying results back. A second mission requires a new launch and timestamped
result directory. No robot access, commit, or push happens automatically.

The start service reports whether the mission was **accepted**, not whether the
robot reached the goal. Follow `/hw4/status`, console messages, or `status.json`
for its eventual result. `complete` requires fresh, stable measured goal proximity.

## Measuring the map

Use meters for lengths and degrees for configured angles. Convert feet by
multiplying by 0.3048 and inches by 0.0254. Unknown values remain `null`.
JSON does not allow comments or trailing commas.

The four obstacle positions are **41h12 tag centers**, not physical corners.
They are now part of `landmarks`, alongside the 36h11 wall tags. IDs 1–4 have
confirmed face headings 180°, 90°, 0°, and 270°, respectively, center height
0.187 m, and printed size 0.172 m. The confirmed physical obstacle rectangle is
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
| `goal` | Target footprint-center position; final heading is unrestricted. |
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

`navigation.py` follows fixed waypoints:

1. Require three distinct, fresh, consistent pose observations.
2. Turn toward the waypoint in fixed-rate pulses of at most 15°.
3. Stop, settle for one second, and obtain new consistent observations.
4. Drive at the HW3 fixed speed for at most 0.20 m, checking the actual forward
   sweep plus a watchdog interval of stopping overrun.
5. Repeat until a fresh stable pose is within 0.08 m of the goal.

Straight motion and pivoting remain separate. Feedback loss, implausible pose
jumps, unsafe sweep, control stalls, and exhausted budgets stop the mission.
There is no blind timed continuation or automatic searching when tags disappear.
The pose can be unavailable near the obstacle or during turns; those failures
are useful run-1 evidence for the next iteration. The code does not detect an
unmapped obstacle; it uses the measured obstacle map.

## Calibration baseline

`config/motion.yaml` contains the **effective** HW3
`hw3_navigation.launch.py mission:=slam_square` values. The mapper and HW4
controller use the same file and HW3 calibration readers.

| Motion | Left input | Right input | Nominal model velocity |
| --- | --- | --- | --- |
| Straight | 0.185 | 0.15 | 0.203 m/s |
| Counterclockwise | -0.35 | 0.35 | +pi/2 rad/s |
| Clockwise | 0.37 | -0.37 | -pi/2 rad/s |

The HW3 default CCW pair `[-0.40, 0.37]` was overridden by `slam_square`; it is
not used here. HW3 measured 0.203 m/s at `[0.18, 0.15]` before raising the left
trim to 0.185 pending remeasurement. Preserve that distinction when evaluating
run 1. Smaller requested speeds do not slow the fixed-pair mapper. Commands are
refreshed at 20 Hz; the reused driver's watchdog is 0.15 s.

Camera intrinsics use the same installed HW3 calibration, respecting the existing
`CAMERA_CALIBRATION` environment override. The actual file is archived per run.

## Data to bring back

Bring back the entire `hw4/results/<timestamp>/` directory, plus your description
and videos of the physical run. Results are excluded from Git.

- `planned_paths.json` and `planned_paths.svg`: the selected route, its length,
  and minimum planned body clearances; the SVG shows the grid and expanded obstacles.
- `observations.csv`: capture timestamps, accepted/rejected feedback, rejection
  reasons, measured world poses, and tag IDs.
- `commands.csv`: state/waypoint, nominal commands, feedback age, measured pose,
  command-predicted pose, correction budgets, and measured body clearance.
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
