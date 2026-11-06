# HW3: live square EKF-SLAM (`hw3_code`)

This package runs a **7 ft × 7 ft** counterclockwise square (2.1336 m per side),
detects **Standard41h12** tags with measured edge size **0.172 m**, and estimates
both the robot trajectory and the landmark map. The map starts empty, with no
known landmark coordinates, ID list, count, or measured world start pose.

The standalone controller simulator and its simulation tests have been removed.

## What runs

```text
raw camera + CameraInfo → AprilTag corners → distortion-aware PnP
                                             ↓ robot-frame range/bearing
                                     delayed-vision EKF-SLAM
                                       ↓ predicted current pose
                                     square waypoint control
                                             ↓ /hw3/cmd_vel
                                     HW3 velocity mapping
                                       ↓               ↓
                                motor commands   applied-command model
                                       ↓               ↓
                                  motor driver      EKF prediction

EKF posterior → estimated trajectory + landmark/covariance visualization + run log
```

The default driver commands four **7 ft straight segments and left 90° turns**.
It uses a separate command-odometry estimate for route progress, so tag detections
and SLAM corrections never block or redirect the square. EKF-SLAM continues in
parallel to estimate the trajectory and landmarks. Forward speed defaults to
0.15 m/s. Physical dimensions still depend on the surface-specific speed/turn
calibration; commanding 7 ft does not establish a measured 7 ft traveled distance.

## Pull, build, and run on the robot

Use Bash from the `robot_local` repository checkout on the robot. Stop the previous
HW2 camera, detector, controllers, and motor driver first; this launch starts its
own camera/detector and uses the same physical motor serial port.

```bash
source /opt/ros/jazzy/setup.bash
source rubikpi_ros2/install/setup.bash
bash hw3/scripts/build.sh
source hw3/install/setup.bash
ros2 launch hw3_code hw3.launch.py
```

Use the ROS setup path actually installed on the robot if it differs. The existing
vision workspace must already supply `robot_vision_camera`, `apriltag_ros`, and
`apriltag_msgs`. This package additionally requires NumPy, OpenCV, pyserial, the
standard ROS geometry/sensor/navigation/visualization messages, and `std_srvs`.
Build only this nested package with `--paths .` (the build script does so); do not
rely on parent-package discovery.

After the first build, the helper uses the existing repository's ROS/vision setup:

```bash
bash hw3/scripts/run.sh
bash hw3/scripts/run.sh laps:=3
```

The default launch **arms autonomous movement**. It waits only for the motor-motion
heartbeat and a short startup delay before driving. **No tag detections are needed.** To map/view while stationary first:

```bash
bash hw3/scripts/run.sh auto_start:=false
ros2 service call /hw3/start std_srvs/srv/Trigger '{}'
```

Pause/resume without resetting the map:

```bash
ros2 service call /hw3/stop std_srvs/srv/Trigger '{}'
ros2 service call /hw3/start std_srvs/srv/Trigger '{}'
```

Ctrl-C shuts down the launch. Faults and clock/timing failures stop motion and
require a new run. A completed square remains stopped while logging/visualizing
until shutdown. Missing/absent vision does not pause the route. Completion is based
on command odometry; it is not independent proof of physical waypoint accuracy.

### Route placement and physical settings

Place the robot at the intended first corner with its forward direction along the
first side. The local route is:

```text
(0, 0) → (2.1336, 0) → (2.1336, 2.1336) → (0, 2.1336) → (0, 0)
```

It turns left at each corner and returns to the initial heading. The operator must
place this route inside the measured arena with room for the robot footprint and
turns. There is no obstacle-avoidance planner. Record the world start pose separately
for evaluation; do not enter it into the filter.

Useful overrides:

```bash
bash hw3/scripts/run.sh side_ft:=7.0 laps:=3 tag_size_m:=0.172 frame_rate:=10
```

Before interpreting metric results, verify these settings against the robot:

- `config/camera_parameter.yaml` is copied from our existing calibration and expects
  1280×720. An alternate actual calibration can be passed with
  `camera_parameter_path:=/absolute/path/camera_parameter.yaml`.
- `config/hw3.yaml` contains camera mount `[x, y, z] = [0, 0, 0.2]` m and zero mount
  RPY copied from our HW2 configuration. Confirm the camera origin/angles. RPY is
  the body-aligned camera mount rotation; the optical-axis conversion is applied
  separately (optical z forward, x right, y down).
- Motor commands use **our** `robot_testing/velocity_mapping.py` equations: wheel
  base 0.127 m, linear deadzone 0.08, linear slope 1.772915625, and angular slope
  1e6. The custom pulse-density turn behavior has been removed. For this surface,
  HW3's turn input and motor-driver limit default to **0.6** instead of 0.5.
- Pure turns send continuous opposing wheel inputs. Because our angular slope is
  1e6, changing the requested angular velocity barely changes motor power. Use
  `turn_input` to change turning effort and `motor_limit` to permit it through the
  driver. Keep `turn_input` at or below `motor_limit`. Straight-drive mapping is
  unchanged; the controller still limits forward speed to 0.15 m/s.
- The nominal turn-rate estimate remains 2π/3.75 rad/s from the earlier HW2 setting.
  It is **not calibrated for the new surface or the stronger 0.6 input**. Remeasure
  a complete rotation in each direction and set `turn_rate_ccw = 2π / CCW_seconds`
  and `turn_rate_cw = 2π / CW_seconds` in the mapper section of `config/hw3.yaml`.
  Incorrect rates bias command-based prediction even when the wheels turn properly.

  ```bash
  bash hw3/scripts/run.sh turn_input:=0.6 motor_limit:=0.6
  ```

- Noise settings are provisional tuning values, not newly measured covariance.
  Fit/check them using actual stationary observations and movement logs.

## View the robot from your Mac (no graphical desktop required)

The launch serves a read-only browser map on the Rubik Pi at port **8080**. Leave
the robot launch running, and find the Pi's network address in its SSH terminal:

```bash
hostname -I
```

On your Mac, open **`http://ROBOT_IP:8080/`** in Safari or Chrome using the Pi address
reachable from the Mac. The page shows the estimated robot trajectory/heading,
landmark IDs/positions, 95% covariance ellipses, and connection/vision status.
No ROS or RViz installation is needed on the Mac. Both devices must be on a
network where the Mac can reach the Pi. If direct access is unavailable, forward
it through your existing SSH access from a Mac terminal:

```bash
ssh -N -L 8080:127.0.0.1:8080 ubuntu@ROBOT_IP
```

Then open **`http://localhost:8080/`** on the Mac. The page's Pause display button
only pauses drawing; use `/hw3/stop` or Ctrl-C in the robot launch to stop motion.
If port 8080 is occupied, launch with `web_port:=8081` and use that port instead.
The viewer serves only its HTML page and estimator state, not arbitrary robot files.

An empty map means no landmarks have been accepted yet. The displayed blue path
then comes from command dead reckoning; the viewer never invents landmarks or
labels a planned route as a measured trajectory.

## Optional RViz visualization

The node publishes these estimates at 2 Hz in fixed frame **`hw3_map`**:

| Topic | Type | Content |
|---|---|---|
| `/hw3/trajectory` | `nav_msgs/Path` | Timestamped algorithm-estimated trajectory |
| `/hw3/landmarks` | `visualization_msgs/MarkerArray` | Estimated centers, family/ID labels, green 95% position-covariance ellipses |
| `/hw3/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Estimated planar robot pose and covariance |
| `/hw3/status` | `std_msgs/String` | Control/wait state, landmark count, completed laps, skipped frames |
| `/hw3/detections` | `apriltag_msgs/AprilTagDetectionArray` | Raw detector identities, corners, and decode quality |
| `/hw3/image_raw` | `sensor_msgs/Image` | Unrectified camera images |
| `/hw3/camera_info` | `sensor_msgs/CameraInfo` | Matching raw K and distortion coefficients |
| `/hw3/applied_motion` | `geometry_msgs/TwistStamped` | Nominal velocity corresponding to dispatched motor commands |

On a machine with RViz2, ROS discovery to the robot, and the HW3 overlay sourced:

```bash
rviz2 -d "$(ros2 pkg prefix --share hw3_code)/config/hw3.rviz"
```

RViz can run directly on the Rubik Pi if `ros-jazzy-rviz2` is installed and an
Ubuntu graphical session is visible on an attached monitor or through remote
desktop. Plain SSH from a Mac does not display its windows. On a robot with that
graphical session, `rviz:=true` starts the same view as part of launch. For manual setup, choose fixed frame `hw3_map`, add Path
`/hw3/trajectory`, and add MarkerArray `/hw3/landmarks`. All markers/path points
already use the fixed frame; no world-frame transform is required. Publishers are
transient-local so a newly opened viewer can receive the latest map and path.

Monitor status without a GUI:

```bash
ros2 topic echo /hw3/status
ros2 topic hz /hw3/detections
```

## Recorded outputs and browser-viewable map

Each run writes a unique directory under `~/hw3_runs/`, printed at startup:

- `parameters.json`: estimator/controller settings used.
- `run.jsonl`: timestamped estimated states, **full joint covariance**, landmark-ID
  order, observation/gate decisions, commands, applied-model motion, calibration,
  lap-completion snapshots, and faults.

At each completed lap the full map/covariance is saved. Periodic estimates continue
at 2 Hz. The path preserves the estimate at each timestamp; this EKF does not
retroactively smooth the entire past trajectory after a later observation.

Export any recorded run to a standalone map you can open in a browser:

```bash
ros2 run hw3_code plot_run ~/hw3_runs/RUN_DIRECTORY --output square.svg
ros2 run hw3_code plot_run ~/hw3_runs/RUN_DIRECTORY --lap 1 --output lap1.svg
```

Each creates an SVG showing the estimated trajectory, landmark labels, and 95%
covariance ellipses, plus a timestamped `.csv` trajectory. Use new output names.
You can copy run logs to the computer and export without ROS:

```bash
PYTHONPATH=hw3 python3 -m hw3_code.plot_run RUN_DIRECTORY --output square.svg
```

The export uses only Python and NumPy; no plotting dependency is needed. See
[`evaluation/README.md`](evaluation/README.md) for measured-start world alignment,
ground-truth overlay, missing/spurious landmark reporting, and average landmark error.

## Estimation and timing details

- `observations.py` uses detector pixel corners and **raw** CameraInfo K/D for PnP.
  Images remain unrectified. This avoids the existing driver's rectified-image/P
  mismatch. Detector pose estimation is disabled; no tag TF or known tag map is used.
- Detector family is `Standard41h12`; no `tag.ids`, `tag.frames`, or `tag.sizes`
  whitelist is loaded. Internal landmark keys are `(41h12, ID)` and grow dynamically.
- Each tag yields planar robot-frame range/bearing. Margin/hamming, positive depth,
  reprojection error, and range checks reject unsuitable observations. A new
  landmark augments the state/cross-covariance; returning IDs update existing entries.
- Range standard deviation starts as `range_std_m + range_scale * range²`; bearing
  standard deviation is separately configured. These are starting models to validate.
- Motion prediction uses **command-based dead reckoning**, not measured wheel odometry.
  The mapper reports the nominal velocity corresponding to each dispatched motor
  command, using separate configurable clockwise/counterclockwise turn-rate estimates. Encoder feedback is unavailable in the supplied interface;
  IMU gyro units are not established, so no unverified IMU data is fused.
- Prediction input variances are `distance_noise²*abs(distance) + idle_distance_noise²*dt`
  and the analogous turn expression. The reported command history includes the fixed-input
  turn-rate estimate and command expiry, rather than pretending requested angular
  velocity is the measured physical turn rate.
- The persistent estimator runs 0.35 s behind ROS time, applying accepted observations
  at capture time and then propagating through saved commands. A second estimate integrates commands without visual correction for fixed-route
  progress; its temporary predicted copy supplies the current drive pose. Duplicate/too-late frames are discarded
  and counted; increase `vision_delay_s` if actual detector latency exceeds the buffer.
- Initial pose `(0,0,0)` with zero covariance chooses the local reference frame.
  No measured world pose is supplied. Maps and covariances are approximate Gaussian
  EKF estimates; false IDs, poor geometry, or bad calibration can still corrupt them.

## Verification and remaining assignment work

Twenty-one local numerical/component checks passed (one additional localhost HTTP
listener check was skipped because the development environment denies socket
listeners): EKF derivatives/covariance,
landmark discovery/reobservation, distorted-camera geometry, timestamped command
expiry, square dimensions, and log/map exports. Run them from the repository root:

```bash
PYTHONPATH=hw3 python3 -m unittest discover -s hw3/test -p 'test_*.py' -v
```

ROS is unavailable on the development Mac, so a real ROS build/launch, camera,
serial connection, browser HTTP access, surface-specific motor response, and
physical trajectory have **not** been tested on the robot.
Run logs/status provide the evidence for the next iteration. HW2 source is unchanged.

This implementation covers the requested square experiment. The assignment's
octagon experiment, physical ground-truth measurements, comparative results,
report, video, and final code ZIP still need to be completed.
