# Robot workflows

Run these Bash scripts on the robot, from any working directory. Paths to the
repository, ROS overlays, calibration file, and detection log are resolved from
the scripts' location. The examples below start in the repository root.

## Build the vision workspace

```bash
source /opt/ros/jazzy/setup.bash
cd rubikpi_ros2
colcon build --packages-select robot_vision_camera apriltag_ros
```

Rebuild after changing camera code or AprilTag configuration. Each script sources
ROS and the appropriate workspace overlay automatically. To use another ROS
installation, set `ROS_SETUP` to its absolute `setup.bash` path.

## Capture calibration images

From the repository root, run these in separate terminals:

```bash
./scripts/run_camera.sh
./scripts/capture_calibration.sh --interval 5 --count 25
```

`run_camera.sh` starts the raw camera stream. Capture saves a new timestamped
directory under `calibration/images/`. An explicit relative `--output` is relative
to your current directory. See [calibration](../calibration/README.md) for offline
calibration commands; those Python tools stay in `calibration/`.

## Live detection

From the repository root, run one command per terminal:

```bash
./scripts/launch_camera.sh
./scripts/launch_apriltag.sh
./scripts/monitor_detections.sh
```

`launch_camera.sh` enables rectification using
`config/camera_parameter.yaml`. Override it
with `CAMERA_CALIBRATION=/absolute/path/to/camera_parameter.yaml`.
The detector uses `rubikpi_ros2/apriltag_ros/cfg/tags_36h11.yaml`; check tag IDs and
measured sizes there, then rebuild. All three commands forward additional
arguments to `ros2`, for example `./scripts/launch_camera.sh frame_rate:=5`.

Alternatively, `./scripts/start_live.sh 36h11 0.165` starts the calibrated camera,
detector, and logger together with an explicit family and measured tag size in
meters. It appends non-empty detections to `detections.txt` at the repository root
and stops its children on Ctrl-C. Use one camera workflow at a time.
Generated detection and coordinate logs are ignored by Git. Keep experiment
artifacts in the ignored `hw2_results/` directory for local analysis.

## Stationary 36h11 rosbag test

This starts only the existing camera executable, the AprilTag detector, and
rosbag. It does not start motors, navigation, RViz, or the camera launch file's
Foxglove bridge. Stop other camera and navigation sessions before running it.
Keep the robot stationary throughout the test.

On the robot, switch to `dustin-hw3` in the actual checkout containing this
`scripts/` directory. Its parent directories and checkout name do not matter.
If the vision workspace has not been built, run from that checkout's root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths rubikpi_ros2 --build-base rubikpi_ros2/build \
  --install-base rubikpi_ros2/install --packages-select robot_vision_camera apriltag_ros
```

If rosbag is missing, install `sudo apt install ros-jazzy-rosbag2`.
The script sources the ROS installation and `rubikpi_ros2/install/setup.bash`.
No build of the outer `robot_testing` package is needed for this test.

Start recording from the checkout root:

```bash
bash scripts/record_tags.sh
```

The default printed tag edge size is **0.172 m**, as used in HW2. For a different
measured size or a named output directory, use:

```bash
bash scripts/record_tags.sh 0.165 /absolute/path/to/new_bag_directory
```

Measure the tag's outer black square edge, excluding the surrounding white
paper. All tags in this simple test must share the supplied size. There is no
tag ID whitelist and no surveyed landmark map. Hold a 36h11 tag still in front
of the camera for approximately 10 seconds, then press **Ctrl-C once**. Wait
until the launch command finishes so rosbag can write its metadata. If any
camera, detector, or recorder process exits, the remaining processes shut down.

The terminal prints the absolute bag directory. By default it is
`runs/tags_<timestamp>` inside the checkout, ignored by Git. A successful camera
stream can still produce a bag with no detected tags; verify actual detections.
The recorded topics are:

| Topic | Contents |
| --- | --- |
| `/camera/image_raw/compressed` | Rectified JPEG frames at 5 fps |
| `/camera/camera_info` | Timestamped intrinsics and camera calibration |
| `/detections` | Detected families, IDs, pixel corners, and quality |
| `/tf` | Detected tag positions and orientations relative to `camera_frame` |
| `/tf_static` | Any available static transforms; none are added by this test |
| `/rosout`, `/parameter_events` | Available node logs and parameter events |

`/detections` does not itself contain 3D positions; those are in `/tf`.
Tag TF names default to `tag36h11:<id>`. Camera optical coordinates are
`+x` right, `+y` down, `+z` forward, in meters. This records relative tag poses,
not a robot/world map or SLAM output. Raw uncompressed images are not recorded.

In another robot terminal, check an actual detection while recording:

```bash
source /opt/ros/jazzy/setup.bash
source rubikpi_ros2/install/setup.bash
ros2 topic echo /detections --once --filter 'len(m.detections) > 0'
```

After stopping, inspect the printed bag directory:

```bash
ros2 bag info /absolute/path/to/the/bag
```

Check for camera frames and `/tf` messages, not just empty detection arrays.
Copy the **entire bag directory**, including `metadata.yaml`, into the Ubuntu
VM. From a VM terminal with network access to the robot:

```bash
mkdir -p ~/robot_runs
scp -r ROBOT_USER@ROBOT_IP:/absolute/path/to/the/bag ~/robot_runs/
```

For a basic RViz check in the VM, start `rviz2 --ros-args -p use_sim_time:=true`,
set **Fixed Frame** to `camera_frame`, and add a **TF** display. In a second VM
terminal, source ROS and replay the copied bag with
`ros2 bag play /absolute/path/to/copied/bag --clock`. Visible tag frames show
their recorded relative locations. No motor nodes should be running for replay.
The custom `/detections` message requires `apriltag_msgs` in the VM to inspect
it; the standard TF and camera messages can be viewed independently.

## HW1 waypoint navigation

```bash
./scripts/run.sh
```

This builds `robot_testing` from the repository root, sources its overlay, and
launches the motor controller, velocity mapping, and HW1 navigation nodes.
Additional arguments are forwarded to the launch command.
