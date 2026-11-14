## Octagon trial

Build the same HW3 package and launch the octagon profile:

```bash
cd ~/robot_testing
source /opt/ros/jazzy/setup.bash
source rubikpi_ros2/install/setup.bash
colcon build --base-paths hw3 --packages-select hw3
source install/setup.bash
ros2 launch hw3 hw3_octagon.launch.py
```

In another robot terminal, start the mission:

```bash
source /opt/ros/jazzy/setup.bash
ros2 service call /start_navigation std_srvs/srv/Trigger '{}'
```

Before the first forward leg, the octagon profile now performs one staged
startup revolution: initial 3-second acquisition pause → nominal 45-degree left
step → 3-second pause, repeated until the commanded angle totals 360 degrees.
Use the existing `[-0.35, +0.35]` pivot calibration. Callbacks and SLAM stay active
throughout; no tag is required to complete any step. A final stopped comparison
with the original tag orientation, or a confident EKF heading, may select bounded
left/right closure corrections before forward motion. Missing feedback continues
from calibrated prediction. The original start position and heading are latched
before scanning and are never silently replaced by the post-scan pose.

`startup_scan_enabled`, `startup_scan_step_rad`, and `startup_scan_pause_s`
configure this behavior. It is enabled by the octagon launch and disabled in the
base square profile. Startup progress and corrections are recorded in navigation
CSV columns prefixed with `startup_`. Initial mapping can benefit from the extra
observations; the scan does not certify physical closure or guarantee every tag
will be visible.

All **eight legs** use nominal 75 cm forward travel. All **eight corners** use
30-degree first stage → 3-second pause → finish toward 45 degrees → pause →
optional bounded corrections. The last two legs follow the same sequence as the
first six. Center-looking maneuvers, position-based late-waypoint reorientation,
and return-to-origin motion are disabled (`late_scan_start_corner: 99`), including
an explicit override in the octagon launch. No special maneuver begins at corner 6.

The measured left pivot uses `[-0.35, +0.35]`, with a nominal rate of pi/2 rad/s.
Small corrections can turn either direction: CW uses `[+0.37, -0.37]`. Midpoint
pauses recheck remaining distance and heading, with another pause after each
optional pivot. The octagon launch sets `turn_pulse_rad` to 5 degrees, keeping
each requested correction small. Preserve the 5-degree minimum, three-attempt
limit, 15-degree nominal angular budget, and bounded distance adjustments. Missing observations
never block completion. Straight inputs remain `[+0.185, +0.15]` at nominal
0.203 m/s. No surveyed coordinates are used for navigation.

The profile lives in `launch/hw3_navigation.launch.py`. The experimental late
navigation helper remains available but is not entered by either default profile.
Existing CSV formats, start/stop services, and build/launch commands are retained.
The full octagon approximately returns to the start by geometry; it no longer
actively drives toward an estimated start position.

If pose coordinates or heading alternate between incompatible estimates, check
`ros2 topic info /ekf_slam/pose --verbose` before tuning motion. Multiple SLAM
publishers can interleave unrelated map frames even when they use the same frame
name. Stop older launch sessions before running the current stack; confidence
thresholds and extra corrections cannot reconcile conflicting pose publishers.

# HW3: Visual Odometry, EKF-SLAM, and Testing Infrastructure

## Fixed-rate turning: calibrate, then run the square

The navigation launch uses **straight travel and separate fixed-rate pivots**.
For a positive (CCW) pivot, the mapper sends exactly `[L, R] = [-0.40, +0.37]`;
for CW correction it sends `[+0.37, -0.37]`. Nonzero angular command magnitude does not change
the motor input. The HW2 package and its mapping remain unchanged. All square-control code lives in HW3.

Measured motion settings are shared by navigation and mapping in
`hw3/config/hw3_params.yaml`:

```yaml
turn_calibration:
  enabled: false  # the navigation launch enables it for the relevant nodes
  ccw_left_input: -0.40
  ccw_right_input: 0.37
  cw_left_input: 0.37
  cw_right_input: -0.37
  ccw_radps: 1.5707963267948966  # approximately 90 degrees in 1 second
  cw_radps: 1.5707963267948966   # approximately 90 degrees CW in 1 second
```

Forward travel uses **left +0.185, right +0.15** to compensate for left drift.
The speed prediction provisionally retains the baseline **2.03 m in 10 seconds
= 0.203 m/s**, measured with the previous `[0.18, 0.15]` pair. Remeasure distance
over time at the new trim before treating that speed as calibrated. Both motors
hold their fixed values during forward travel; the old speed-dependent mapping and end-of-leg taper have been removed.
The mapper publishes the same measured speed on `/cmd_vel_applied` for the EKF.

```yaml
forward_calibration:
  left_input: 0.185
  right_input: 0.15
  speed_mps: 0.203
```

Nominal travel times are **4.504 seconds for 3 feet** and **9.009 seconds for
6 feet**, excluding pauses and estimator-based remaining-distance corrections.
The controller samples commands at 20 Hz, so stopping is quantized to a control
interval. Distance acceptance is now 0.01 m in the prediction, rather than the
previous 0.10 m shortfall allowance; this is not a physical accuracy guarantee.
Reverse requests stop because this calibration only measures forward motion.

The latest CCW calibration uses approximately 90 degrees in 1 second at
`[-0.40, +0.37]`. CW correction uses approximately 90 degrees in 1 second at
`[+0.37, -0.37]`. Both nominal rate magnitudes are `pi/2` rad/s. Main square
corners always turn left; clockwise motion is only an optional overshoot correction.
For each direction, calculate `rate_radps = total_angle_degrees * pi / 180 / seconds`.
For a complete turn this is `2*pi / turn_seconds`. If using several rotations,
include all rotations in the total angle. Measure both directions, including
short pulses: acceleration and stopping can make a short pulse differ from the
average rate over several full rotations. Zero rates intentionally prevent
navigation from starting; do not substitute guessed calibration values.

Build **HW3** after pulling these changes, from the actual repository
root on the robot (its directory name does not matter):

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths hw3 --packages-select hw3
source install/setup.bash
```

Stop previously launched camera, detector, motor, and controller nodes.
From the repository root, launch the complete experiment in one terminal:

```bash
source /opt/ros/jazzy/setup.bash
source rubikpi_ros2/install/setup.bash
source install/setup.bash
ros2 launch hw3 hw3_navigation.launch.py \
  params_file:="$(pwd)/hw3/config/hw3_params.yaml" square_size_ft:=6.0
```

The launch includes the camera and tag36h11 detector. It installs the existing
`config/camera_parameter.yaml` calibration with HW3; override it with
`camera_parameter_path:=/absolute/path/to/calibration.yaml` when needed.
Use `launch_sensors:=false` only when your camera and detector are already running.
A shorter square changes the target distance; it does not correct linear calibration
or braking error that caused overshoot.

Using the source YAML path means later calibration edits require a **launch
restart**, not another build. Once the nodes are ready, start the square from
another sourced terminal:

```bash
ros2 service call /start_navigation std_srvs/srv/Trigger '{}'
```

Stop with `ros2 service call /stop_navigation std_srvs/srv/Trigger '{}'` or Ctrl-C
in the launch terminal. Do not run another controller or teleop concurrently.

The mapper publishes its nominal calibrated velocity on `/cmd_vel_applied`.
Navigation mode makes the EKF integrate that topic, including the last command
interval before a stop; it does not also integrate VO motion. Tag observations
still correct the robot/landmark state. These are model-based velocities, not
encoder measurements. Manual visual-only mode retains its VO prediction source.

The square now uses the following phases on each of four sides:

`DRIVE first 3 ft → MIDPOINT_PAUSE → optional MIDPOINT_CORRECTING → MIDPOINT_PAUSE → DRIVE remainder → STOPPING → TURNING 90° → CORNER_PAUSE → optional CORRECTING → next side`

- Every leg starts and continues without requiring any visible or previously
  mapped tag. Missing/stale EKF messages use calibrated command prediction;
  high covariance does not block the mission.
- Drive a nominal first 3 feet (0.9144 m, half the configured side length),
  then stop for `navigation.midpoint_pause_s` (default 3.0 seconds). Callbacks
  remain active during the pause so delayed images and EKF updates can arrive.
- At pause completion, project a fresh, sufficiently confident post-stop EKF
  position onto the original side axis. Compare it with nominal traveled distance
  and clamp the adjustment to `navigation.midpoint_max_correction_m` (default
  +/-0.10 m, capped at one quarter of a side for smaller squares). The remainder
  stays approximately 3 feet +/-10 cm; a bad estimate cannot reduce it to 5 cm.
- During the same midpoint pause, check heading before resuming forward travel.
  Prefer fresh image-associated EKF heading with confident robot-relative tag
  covariance and three stable stopped observations, applying the existing 50%
  heading-error weight. Aim toward the fixed course chosen at leg start; do not
  redirect toward a moving waypoint. If no confident EKF heading exists, compare
  stable same-tag normals with the stopped reference captured at leg start and
  correct yaw drift relative to that starting direction, without relying on the
  global map. Missing either reference simply skips that feedback source.
- Midpoint pivots use fixed calibrated motor inputs, with a 5-degree minimum,
  at most three attempts, and a 15-degree total nominal budget per midpoint.
  These use the existing `corner_min_correction_rad`, `corner_max_corrections`,
  `corner_correction_limit_rad`, and `turn_pulse_rad` policy settings. Midpoint
  counters are separate from corner counters. Never combine forward travel and
  turning in the same command.
- After each midpoint pivot, pause again for `midpoint_pause_s` (3 seconds), then
  recheck both heading and remaining distance. Require a new post-stop image
  for another pivot. Fresh position feedback can revise the bounded distance
  adjustment; stale position feedback retains the previous bounded adjustment
  instead of erasing it. Once corrections finish or feedback is unavailable,
  freeze the distance plan for the second half and integrate the calibrated
  0.203 m/s command. Continuous moving-camera/EKF updates cannot override it.
- Missing, stale, or uncertain initial EKF data use zero distance adjustment.
  No tag observation is required to resume forward travel. The fixed heading
  target and straight motor pair remain unchanged; only stopped pivots correct
  direction. Return-to-start segments already recheck both distance and bearing
  at their acquisition pauses.
- Stop for three seconds, then apply one initial left-turn pulse for
  `navigation.corner_turn_duration_s` (default 0.95 seconds) using the measured
  fixed CCW inputs. This shorter pulse aims to reduce physical overshoot. The
  stored rate still predicts 85.5° for 0.95 seconds; the EKF and logs do not pretend
  it measured 90°. There is no 360° scan at a corner; the octagon startup scan happens only once.
  Duration has control-tick quantization.
- Stop for another `navigation.turn_settle_s` (default 3 seconds), allowing delayed
  camera observations to arrive. A missing observation never extends this pause.
- Prefer the same tag's face-normal orientation before and after turning to
  measure actual relative yaw change and correct that rotation toward 90°,
  independently of any bias in the prior global heading. Tag orientations are transformed from
  camera optical coordinates into robot-base coordinates. This comparison uses
  no estimated global tag position. Reject likely PnP face flips and inconsistent
  matched-tag changes. Use `navigation.corner_min_observations` (default 3)
  distinct stopped frames per tag, with normal-yaw spread no greater than
  `navigation.corner_yaw_spread_rad` (default 3 degrees). Only images captured
  after stopping can correct the turn. These sample-consistency settings also
  apply to references captured before the turn.
- If no usable same-tag comparison exists, EKF turn feedback is permitted only
  from a fresh image captured after the current stop. Every accepted known tag
  updated by that image must have a pre-update largest robot-relative positional
  standard deviation of at most `navigation.corner_landmark_sigma_m` (default 0.06 m).
  Propagate the full joint robot/landmark covariance through the robot-frame
  position Jacobian, including yaw and robot-landmark cross correlations, then
  use the largest eigenvalue of the resulting 2x2 covariance. Shared global
  uncertainty can cancel; no global landmark covariance is artificially reduced. One confident tag is sufficient;
  there are no hard-coded tag IDs. Missing covariance or mixed-confidence updates
  cause the EKF fallback to be skipped.
- The image-associated EKF yaw must also have standard deviation at most
  `navigation.corner_heading_sigma_rad` (default 15 degrees). Apply half the
  resulting absolute heading error (`navigation.corner_ekf_weight: 0.5`), then
  enforce the pulse and total-angle limits. Set weight to zero to disable this
  fallback. A new post-stop image is required after each pulse; timer-published
  predictions alone cannot authorize another correction. Source tag IDs are
  recorded in `corner_feedback_source`; skipped feedback records whether the
  reason was freshness, missing map observations, relative covariance, or heading
  uncertainty. Covariance is model confidence, not
  independent proof of map or heading accuracy.
- Optional corrections use the measured fixed CW/CCW inputs. They never reduce
  motor magnitude to request a small turn. Residual errors below
  `navigation.corner_min_correction_rad` (default 5 degrees) are accepted without
  motion because smaller pulses are not requested. Five-degree pulses are a physical
  trial; the existing 20 Hz controller quantizes duration to about 4.5 degrees
  at the nominal 90 deg/s rate, and motor deadzones may still prevent movement. A remaining
  correction budget below that minimum is skipped rather than rounded upward.
  Omitting a planned left-turn remainder consumes a correction decision but no
  extra-angle budget, so a measured overshoot can still receive a clockwise
  correction. Extending that remainder consumes the additional angle budget.
  Corrections total at most
  `navigation.corner_correction_limit_rad` (default 15 nominal degrees), in at most
  `navigation.corner_max_corrections` pulses (default 3). Pause after each pulse
  before considering another observation. Timer quantization may add up to one
  control interval of motion. Noisy map estimates cannot authorize repeated spins.
- Side targets remain anchored to initial heading + 90°, 180°, 270°, and 360°.
  Four quarter-turns finish the lap facing approximately the initial direction.

The complete four-side sequence can finish with zero detections. Missing tags,
high covariance, and lack of visual stability never by themselves stop it.
There is no autonomous wall-distance stop. Clearance is recorded in the CSV
only; a supervising human uses `/stop_navigation` or Ctrl-C to stop motion.
Missing/stale EKF messages also do not abort the mission. A corner deadline ends
the current turn and proceeds to the camera pause rather than aborting the square.
The motor driver's command watchdog and shutdown stop remain unchanged.

Wall-plane intersection from visible tags remains a diagnostic measurement in
meters from robot center. It does not impose a travel limit or require known
wall coordinates. No surveyed map is loaded by navigation.

The EKF uses higher-uncertainty command prediction plus gated tag observations.
Command covariance grows with elapsed time rather than shrinking with callback
frequency. Vision updates require fresh images, exact-time TF, 36h11 family, and
configured margin/hamming/range checks; stale latest-TF fallback is disabled.
The relative-travel check still depends on EKF yaw, and all turn/correction motor
rates remain subject to acceleration, battery, and surface error. Test a short
leg and turn/correction before another full lap; hardware validation is pending.

For recording/debugging, include the applied commands as well as outputs:

```bash
ros2 bag record /cmd_vel /cmd_vel_applied /motor_commands \
  /ekf_slam/pose /ekf_slam/trajectory /ekf_slam/landmarks \
  /ekf_slam/vision_feedback /detections /tf /tf_static -o hw3_results/direct_turn_square
```

Local synthetic checks (no ROS or hardware): `python3 hw3/run_tests.py`.

This package implements Steps 2, 3, and 4 of Homework 3 for the differential-drive robot without wheel encoders or IMU:

1. **Visual Odometry (VO)**: Planar relative motion estimation $(\Delta x, \Delta y, \Delta \theta)$ between consecutive camera frames using stationary AprilTags.
2. **EKF-SLAM**: 2D Extended Kalman Filter simultaneously estimating the robot's trajectory $(x_R, y_R, \theta_R)$ and mapping dynamically discovered landmarks $(x_i, y_i)$ with full covariance propagation.
3. **Testing Infrastructure**: Standalone synthetic unit tests, data recorder/replayer tools, manual movement test mode, RViz markers/ellipses, and CSV exporters.

---

## 1. Package Structure

```
hw3/
├── package.xml                   # ROS 2 package manifest (ament_python)
├── setup.py                      # Build configuration & console script entry points
├── setup.cfg                     # ROS 2 Python install prefix configuration
├── README.md                     # Comprehensive documentation and math reference
├── config/
│   └── hw3_params.yaml           # Tunable parameters for VO, EKF, quality, & visualization
├── launch/
│   ├── hw3.launch.py             # Full HW3 launch (VO + EKF-SLAM + cmd_vel fallback)
│   └── hw3_manual.launch.py      # Manual pick-and-walk testing mode (pure VO prediction)
├── hw3/
│   ├── __init__.py
│   ├── coordinates.py            # Rigid transformations, camera-to-base extrinsics, angle wrapping
│   ├── visual_odometry.py        # Core visual motion estimator from stationary AprilTags
│   ├── ekf_slam.py               # Core EKF-SLAM state filter, Jacobians, and dynamic landmarks
│   ├── square_navigation.py       # Pure waypoint and bounded turn-pulse controller
│   ├── vision_navigation.py       # Fresh-tag verification, relative travel, wall-plane geometry
│   ├── turn_calibration.py        # Shared fixed input and measured CW/CCW rates
│   ├── velocity_mapping.py        # Pure command-to-motor mapping
│   ├── data_recorder.py          # Observation/command recorder, replayer, and CSV exporters
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── visual_odometry_node.py # ROS 2 node publishing /visual_odometry/* topics
│   │   ├── ekf_slam_node.py        # ROS 2 node running EKF-SLAM & publishing markers/TF
│   │   ├── navigation_node.py      # ROS mission interfaces and telemetry
│   │   ├── velocity_mapping.py     # ROS adapter for pure mapping
│   │   └── motor_control.py        # Serial transport and motor watchdog
│   └── scripts/
│       ├── __init__.py
│       ├── record_data.py        # CLI recording tool
│       ├── replay_data.py        # CLI replay tool
│       └── export_map.py         # CLI CSV export utility
├── scripts/                      # Thin compatibility wrappers for the packaged CLI tools
│   ├── record_data.py
│   ├── replay_data.py
│   └── export_map.py
├── test/
│   ├── __init__.py
│   ├── test_coordinates.py       # Transform & angle wrapping unit tests
│   ├── test_visual_odometry.py   # Translation, rotation, fusion, & outlier unit tests
│   ├── test_ekf_slam.py          # 11 comprehensive synthetic tests (covering all 10 prompt specs + square)
│   ├── test_recording.py         # Data recorder/replayer & CSV export unit tests
│   └── sample_square_run.json    # Pre-recorded 60-frame synthetic square trajectory for verification
└── run_tests.py                  # Standalone test runner (no ROS or hardware required)
```

---

HW3 has no runtime dependency on `robot_testing`. Pure estimation/control modules
contain the algorithms; `nodes/` contains ROS and serial adapters. The camera
and AprilTag detector remain external sensor services. Command flow is:

`square_navigation → /cmd_vel → velocity_mapping → /motor_commands → motor_control`

The mapper also publishes `/cmd_vel_applied` for EKF command prediction.
The outer `scripts/` wrappers reuse `hw3/scripts/`; there is one implementation
of each CLI tool.

## 2. Mathematical Models and Derivations

### 2.1 Coordinate Conventions

- **Robot Base Frame (`base_link`)**:
  - $+X$: Forward
  - $+Y$: Left
  - $+Z$: Up
  - Heading $\theta$: Yaw angle counter-clockwise around $+Z$.
- **Camera Optical Frame (`camera_frame`)**:
  - $+X$: Right in image plane
  - $+Y$: Down in image plane
  - $+Z$: Forward along optical axis into the scene
- **Extrinsic Mapping**:
  $$\begin{bmatrix} X_{base} \\ Y_{base} \\ Z_{base} \end{bmatrix} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix} \begin{bmatrix} X_{opt} \\ Y_{opt} \\ Z_{opt} \end{bmatrix} + \mathbf{t}_{mount}$$
  With camera mount translation $\mathbf{t}_{mount} = [0, 0, 0.2]^T$ m and zero mount tilt.
  Given landmark transform in camera frame $T_{cam}^L$, its pose in base frame is:
  $$T_{base}^L = T_{base}^{cam} T_{cam}^L$$

### 2.2 Visual Motion Estimation (Step 2)

For stationary landmark $L_i$ observed at timestamps $t_1$ and $t_2$:
$$T_{R_1}^{R_2(i)} = T_{R_1}^{L_i} (T_{R_2}^{L_i})^{-1}$$
The planar motion candidate in the robot's base frame at $t_1$ is:
$$\Delta x_i = T_{R_1}^{R_2(i)}[0, 3], \quad \Delta y_i = T_{R_1}^{R_2(i)}[1, 3]$$
$$\Delta \theta_i = \text{atan2}(T_{R_1}^{R_2(i)}[1, 0], T_{R_1}^{R_2(i)}[0, 0])$$

**Multiple Landmark Fusion**:
Each landmark is weighted by detection margin and distance:
$$w_i = \frac{\max(1, \min(\text{margin}_i, 100))}{1 + d_i^2}$$
When $\ge 3$ tags are visible, candidates with pairwise disagreement $> \text{threshold}$ are rejected as outliers.
The combined estimate is:
$$\Delta x = \frac{\sum w_i \Delta x_i}{\sum w_i}, \quad \Delta y = \frac{\sum w_i \Delta y_i}{\sum w_i}$$
$$\Delta \theta = \text{atan2}\left(\sum w_i \sin\Delta \theta_i, \sum w_i \cos\Delta \theta_i\right)$$
Uncertainty $\Sigma_{vo} \in \mathbb{R}^{3\times 3}$ is scaled by squared distance, reduced by $\sqrt{N_{tags}}$, and augmented with sample variance of the inliers.
When no common tags are visible, visual odometry outputs `valid = False` without fabricating movement.

### 2.3 EKF-SLAM State Representation (Step 3)

The state vector dynamically expands as landmarks are discovered:
$$\mathbf{x} = \begin{bmatrix} x_R \\ y_R \\ \theta_R \\ m_{1, x} \\ m_{1, y} \\ \vdots \\ m_{N, x} \\ m_{N, y} \end{bmatrix} \in \mathbb{R}^{3 + 2N}, \quad \mathbf{P} \in \mathbb{R}^{(3+2N) \times (3+2N)}$$
The robot begins at an arbitrary initial origin $\mathbf{x}_0 = [0, 0, 0]^T$ with small initial covariance $\mathbf{P}_0 = \text{diag}(\sigma_{x,0}^2, \sigma_{y,0}^2, \sigma_{\theta,0}^2)$.

### 2.4 Prediction Step

**Visual Odometry Input (Preferred)**:
Relative planar displacement $(\Delta x, \Delta y, \Delta \theta)$ in robot base frame:
$$\begin{bmatrix} x_{k+1} \\ y_{k+1} \end{bmatrix} = \begin{bmatrix} x_k \\ y_k \end{bmatrix} + \begin{bmatrix} \cos\theta_k & -\sin\theta_k \\ \sin\theta_k & \cos\theta_k \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix}$$
$$\theta_{k+1} = \text{wrap}(\theta_k + \Delta \theta)$$
Robot motion Jacobian:
$$F_R = \frac{\partial f_R}{\partial x_R} = \begin{bmatrix} 1 & 0 & -\Delta x \sin\theta_k - \Delta y \cos\theta_k \\ 0 & 1 & \Delta x \cos\theta_k - \Delta y \sin\theta_k \\ 0 & 0 & 1 \end{bmatrix}$$
Control input Jacobian:
$$G_R = \frac{\partial f_R}{\partial u} = \begin{bmatrix} \cos\theta_k & -\sin\theta_k & 0 \\ \sin\theta_k & \cos\theta_k & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
Covariance propagation:
$$P_{RR} \leftarrow F_R P_{RR} F_R^T + G_R Q_{vo} G_R^T$$
$$P_{RM} \leftarrow F_R P_{RM}, \quad P_{MR} \leftarrow P_{RM}^T, \quad P_{MM} \leftarrow P_{MM}$$

**Commanded Velocity Fallback**:
If VO is unavailable and linear velocity $v$ or angular velocity $\omega$ is commanded over $\Delta t$:
$$\Delta x = v \Delta t, \quad \Delta y = 0, \quad \Delta \theta = \omega \Delta t$$
Process noise $Q_{cmd} = \text{diag}(\sigma_v^2, \sigma_{lat}^2, \sigma_\omega^2) \Delta t$ during commanded motion with higher uncertainty reflecting motor inconsistency.

### 2.5 Measurement Model & Update

For landmark $i$ at $(m_{i, x}, m_{i, y})$, the 2D range and bearing observation model is:
$$dx = m_{i, x} - x_R, \quad dy = m_{i, y} - y_R$$
$$\hat{r} = \sqrt{dx^2 + dy^2}, \quad \hat{\beta} = \text{wrap}(\text{atan2}(dy, dx) - \theta_R)$$
Measurement Jacobian $H_i \in \mathbb{R}^{2 \times (3+2N)}$:
$$H_i[:, 0:3] = \begin{bmatrix} -\frac{dx}{\hat{r}} & -\frac{dy}{\hat{r}} & 0 \\ \frac{dy}{q} & -\frac{dx}{q} & -1 \end{bmatrix}, \quad H_i[:, 3+2i : 3+2i+2] = \begin{bmatrix} \frac{dx}{\hat{r}} & \frac{dy}{\hat{r}} \\ -\frac{dy}{q} & \frac{dx}{q} \end{bmatrix}$$
where $q = dx^2 + dy^2$.
Innovation and covariance:
$$\boldsymbol{\nu} = \begin{bmatrix} r_{meas} - \hat{r} \\ \text{wrap}(\beta_{meas} - \hat{\beta}) \end{bmatrix}, \quad S = H_i P H_i^T + R_i$$
Outlier rejection via Mahalanobis distance ($D^2 = \boldsymbol{\nu}^T S^{-1} \boldsymbol{\nu} \le 9.21$).
Kalman gain & state update:
$$K = P H_i^T S^{-1}, \quad \mathbf{x} \leftarrow \mathbf{x} + K \boldsymbol{\nu}$$
Covariance update via Joseph form for numerical stability:
$$P \leftarrow (I - K H_i) P (I - K H_i)^T + K R_i K^T, \quad P \leftarrow \frac{1}{2}(P + P^T)$$

### 2.6 Dynamic Landmark Initialization

When an AprilTag is detected for the first time:
$$\alpha = \theta_R + \beta_{meas}$$
$$m_{new, x} = x_R + r_{meas} \cos\alpha, \quad m_{new, y} = y_R + r_{meas} \sin\alpha$$
Jacobians:
$$J_{x_R} = \begin{bmatrix} 1 & 0 & -r_{meas} \sin\alpha \\ 0 & 1 & r_{meas} \cos\alpha \end{bmatrix}, \quad J_z = \begin{bmatrix} \cos\alpha & -r_{meas} \sin\alpha \\ \sin\alpha & r_{meas} \cos\alpha \end{bmatrix}$$
Augmented covariance:
$$P_{aug} = \begin{bmatrix} P & P[:, 0:3] J_{x_R}^T \\ J_{x_R} P[0:3, :] & J_{x_R} P_{RR} J_{x_R}^T + J_z R J_z^T \end{bmatrix}$$

---

## 3. How to Build and Run

### 3.1 Build the Package

From the repository root:
```bash
# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Build hw3
colcon build --base-paths hw3 --packages-select hw3

# Source overlay
source install/setup.bash
```

### 3.2 Run Synthetic Unit Tests (No Robot Hardware Needed)

Run the standalone Python unit test suite:
```bash
# Run synthetic estimator, navigation, mapping, and protocol checks
PYTHONPATH=hw3 python3 hw3/run_tests.py
```
Or via pytest (if installed):
```bash
PYTHONPATH=hw3 pytest hw3/test
```

### 3.3 Start the Camera and AprilTag Detector

From the repository root in separate terminals:
```bash
# Terminal 1: Camera stream with rectification
./scripts/launch_camera.sh

# Terminal 2: AprilTag detector
./scripts/launch_apriltag.sh
```

### 3.4 Launch HW3 Nodes

**Option A: State Estimation Only (Default HW3 Steps 2 & 3)**:
```bash
ros2 launch hw3 hw3.launch.py
```
*(Estimates VO and SLAM. Robot does not move on its own since this is an estimator, not a controller).*

**Option B: Manual Pick-and-Walk Testing (Pure VO, no motor control)**:
```bash
ros2 launch hw3 hw3_manual.launch.py
```

**Option C: HW3 SLAM + Hardware Motors Enabled**:
After entering measured turn calibration, enable the HW3 motor adapters:
```bash
ros2 launch hw3 hw3.launch.py launch_motors:=true
```
Commands use straight travel or fixed-rate pivots; combined steering is suppressed.
In another terminal, send velocity commands:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### 3.5 Record Live AprilTag Observations

```bash
ros2 run hw3 record_data --output hw3_results/my_manual_run.json
# Or run script directly:
./hw3/scripts/record_data.py -o hw3_results/my_manual_run.json
```

### 3.6 Replay Recorded Observations

```bash
ros2 run hw3 replay_data hw3_results/my_manual_run.json
# To test immediately with the included sample square run:
./hw3/scripts/replay_data.py hw3/test/sample_square_run.json
```

### 3.7 View Trajectory and Map in RViz

Start RViz 2:
```bash
rviz2
```
In RViz:
1. Set **Fixed Frame** to `map`.
2. Add **Path** display subscribing to topic `/ekf_slam/trajectory`.
3. Add **PoseWithCovariance** display subscribing to `/ekf_slam/pose`.
4. Add **MarkerArray** display subscribing to `/ekf_slam/landmarks` (displays green landmark spheres, white tag ID labels, blue 2-sigma landmark covariance ellipses, and red robot pose covariance ellipse).
5. Add **TF** display to observe `map -> base_link -> camera_frame`.

### 3.8 Export Map and Trajectory to CSV

The `ekf_slam_node` automatically exports updated CSV files every 10 seconds to `hw3_results/`.
You can also manually export from any recorded run:
```bash
./hw3/scripts/export_map.py --input-json hw3_results/my_manual_run.json --output-dir hw3_results
```
Generated CSVs:
- `ekf_slam_trajectory.csv`: `timestamp_s, x_m, y_m, theta_rad, sigma_x_m, sigma_y_m, sigma_theta_rad`
- `ekf_slam_landmarks.csv`: `tag_id, x_m, y_m, sigma_x_m, sigma_y_m, cov_xy`

### 3.9 Viewing in RViz (Offline Map Publisher Node)

If you have completed a run, the EKF node automatically saves the final landmarks and trajectory to `hw3_results/ekf_slam_landmarks.csv` and `hw3_results/ekf_slam_trajectory.csv`. You can republish them directly to RViz with latched QoS (no rosbags required):
```bash
ros2 run hw3 publish_map --landmarks hw3_results/ekf_slam_landmarks.csv --trajectory hw3_results/ekf_slam_trajectory.csv
```
This continually broadcasts the static `map` frame and publishes all landmarks, tag IDs, covariance ellipses, and trajectory points so you can open RViz anytime (`rviz2`) and inspect the map!

### 3.10 Closed-Loop Square Navigation (6 ft x 6 ft)

The `square_navigation_node` drives four 6 ft legs with initial-frame headings,
direct quarter-turns and optional visual corrections as described above. Ideal corners are shown
below; actual leg origins come from the current post-turn pose, so the controller
does not fabricate a corrected corner or guarantee physical closure:
- Waypoints:
  - WP0: `(0, 0)` ft [0.000, 0.000] m — Origin / Start
  - WP1: `(6, 0)` ft [1.8288, 0.0000] m — Bottom-right (+X)
  - WP2: `(6, 6)` ft [1.8288, 1.8288] m — Top-right (+X, +Y)
  - WP3: `(0, 6)` ft [0.0000, 1.8288] m — Top-left (+Y)
  - WP4: `(0, 0)` ft [0.0000, 0.0000] m — Return to start

Launch all nodes (EKF-SLAM + motor hardware + navigation controller in IDLE mode):
```bash
ros2 launch hw3 hw3_navigation.launch.py
```

Start the square trajectory explicitly:
```bash
ros2 service call /start_navigation std_srvs/srv/Trigger
# Or via topic:
ros2 topic pub --once /start_navigation std_msgs/msg/Empty "{}"
```

Stop navigation at any time:
```bash
ros2 service call /stop_navigation std_srvs/srv/Trigger
```

---

## 4. Verification Checklist & Expected Behavior

| Scenario | Expected Behavior | Failure Case to Watch For |
| :--- | :--- | :--- |
| **Move toward a stationary tag** | Estimated $\Delta x > 0$, $\Delta y \approx 0$, $\Delta \theta \approx 0$. Robot position advances along $+X$. Tag stays at fixed global coordinates. | Sign inversion (robot retreating instead of advancing) caused by inverted relative transform. |
| **Rotate in front of tag** | Estimated $\Delta \theta$ matches turn direction (CCW $> 0$). $\Delta x, \Delta y \approx 0$. Landmark global coordinates remain stationary. | False translational jump caused by ignoring camera offset or incorrect yaw sign. |
| **Multiple tags visible** | Tags are weighted and fused. Trajectory uncertainty shrinks. Landmark estimates remain consistent. | One tag pulling estimate away due to missing outlier rejection. |
| **Tag disappears & reappears** | Map retains landmark position and covariance while invisible. When it reappears, it is recognized by ID without creating a duplicate. Uncertainty decreases. | Duplicate landmark added, or filter divergence upon tag reappearance. |
| **New tag discovered** | Filter state vector and covariance matrix expand dynamically by $+2$. Cross-correlations with robot pose and existing map are correctly established. | Covariance initialized to zero, causing singular matrix inversion ($S^{-1}$). |
| **No tags visible** | Visual odometry outputs `valid = False`. Filter reports VO unavailable. Uncertainty grows monotonically. No motion is fabricated. | Filter hallucinating movement or resetting to origin. |
| **Manual square traversal** | Robot returns to origin $(0, 0, 0)$. Discovered landmarks form consistent quadrilateral layout matching physical room measurements. | Drifting off indefinitely without loop closure. |

---

## 5. Known Limitations and Assumptions

1. **Stationary Environment**: Assumes AprilTags are rigidly affixed to static walls/objects. Moving a tag while in view will corrupt motion estimation.
2. **Double-Counting Correlation**: When VO motion prediction and landmark measurement updates are both derived from the same tag in the same frame, the inter-frame measurement is partially reused. Process and measurement noise parameters ($Q$ and $R$) are tuned to prevent filter overconfidence.
3. **Planar Assumption**: Operates in 2D $(x, y, \theta)$. Extreme pitch or roll (>35 deg, e.g. picking up the robot at a steep tilt) will violate the planar motion model.
4. **Lighting & Motion Blur**: Fast manual movements or low lighting may cause AprilTag detection drops, temporarily pausing VO.

Navigation CSV adds `vision_age_s`, `visual_progress_m`, `ekf_progress_m`,
`forward_clearance_m`, `corner_nominal_rad`, `corner_correction_rad`, `corner_feedback_source`, `corner_residual_rad`, `navigation_heading_rad`,
`midpoint_progress_m`, `midpoint_remaining_m`, `midpoint_source`, and
`midpoint_applied_correction_m`, `midpoint_heading_source`,
`midpoint_heading_residual_rad`, `midpoint_corrections`, and
`midpoint_correction_rad` to the existing columns. These
record visual freshness, competing travel estimates, wall clearance, and timed
corner prediction, optional correction, navigation heading, and midpoint correction separately. Navigation heading can be
model-predicted while the EKF pose remains uncertain. EKF landmark and trajectory CSV formats are unchanged.
