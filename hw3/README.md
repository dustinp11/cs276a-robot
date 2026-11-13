# HW3: Visual Odometry, EKF-SLAM, and Testing Infrastructure

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
│   ├── data_recorder.py          # Observation/command recorder, replayer, and CSV exporters
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── visual_odometry_node.py # ROS 2 node publishing /visual_odometry/* topics
│   │   └── ekf_slam_node.py        # ROS 2 node running EKF-SLAM & publishing markers/TF
│   └── scripts/
│       ├── __init__.py
│       ├── record_data.py        # CLI recording tool
│       ├── replay_data.py        # CLI replay tool
│       └── export_map.py         # CLI CSV export utility
├── scripts/                      # Symlinked / executable CLI entry points
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
Process noise $Q_{cmd} = \text{diag}((\sigma_v \Delta t)^2, (\sigma_{lat} \Delta t)^2, (\sigma_\omega \Delta t)^2)$ with higher uncertainty reflecting motor inconsistency.

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
colcon build --packages-select hw3

# Source overlay
source install/setup.bash
```

### 3.2 Run Synthetic Unit Tests (No Robot Hardware Needed)

Run the standalone Python unit test suite:
```bash
# Run all 21 unit tests (covers all 10 synthetic requirements + square trajectory)
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
If you want to drive the robot with `/cmd_vel` or teleop while running SLAM:
```bash
ros2 launch hw3 hw3.launch.py launch_motors:=true
```
In another terminal, send velocity commands:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Or launch SLAM + motors + keyboard teleop together:
```bash
ros2 launch hw3 hw3_teleop.launch.py
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

### 3.9 Recording Rosbag and Viewing in RViz After Manual Mode

There are two easy ways to record and view the landmarks and trajectory after running manual mode:

#### Method 1: Live Rosbag Recording (Recommended)
You can automatically record a ROS 2 bag during manual mode using the launch argument:
```bash
ros2 launch hw3 hw3_manual.launch.py record_bag:=true bag_name:=hw3_results/manual_square_bag
```
Or record in a separate terminal:
```bash
ros2 bag record /ekf_slam/landmarks /ekf_slam/trajectory /ekf_slam/pose /tf /tf_static -o hw3_results/manual_square_bag
```
After completing the manual run (Ctrl+C), replay the rosbag in a loop so it stays visible in RViz:
```bash
ros2 bag play hw3_results/manual_square_bag -l
```
Then start RViz (`rviz2`), set the Fixed Frame to `map`, and add `/ekf_slam/landmarks` (MarkerArray) and `/ekf_slam/trajectory` (Path).

#### Method 2: Offline Map Publisher Node (No Rosbag Needed)
If you already finished a run, the EKF node automatically saved the final landmarks and trajectory to `hw3_results/ekf_slam_landmarks.csv`. You can republish them directly to RViz with latched QoS:
```bash
ros2 run hw3 publish_map --landmarks hw3_results/ekf_slam_landmarks.csv --trajectory hw3_results/ekf_slam_trajectory.csv
```
This continually broadcasts the static `map` frame and publishes all landmarks, tag IDs, covariance ellipses, and trajectory points so you can open RViz anytime and inspect the map!

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
