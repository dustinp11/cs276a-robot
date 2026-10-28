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
`calibration/results/images_2026-10-26_classic/camera_parameter.yaml`. Override it
with `CAMERA_CALIBRATION=/absolute/path/to/camera_parameter.yaml`.
The detector uses `rubikpi_ros2/apriltag_ros/cfg/tags_36h11.yaml`; check tag IDs and
measured sizes there, then rebuild. All three commands forward additional
arguments to `ros2`, for example `./scripts/launch_camera.sh frame_rate:=5`.

Alternatively, `./scripts/start_live.sh 36h11 0.165` starts the calibrated camera,
detector, and logger together with an explicit family and measured tag size in
meters. It appends non-empty detections to `detections.txt` at the repository root
and stops its children on Ctrl-C. Use one camera workflow at a time. Consult the
[hardware handoff](../APRILTAG_LIVE_DETECTION_HANDOFF.md) for known camera issues
and the recommended three-terminal workflow.

## HW1 waypoint navigation

```bash
./scripts/run.sh
```

This builds `robot_testing` from the repository root, sources its overlay, and
launches the motor controller, velocity mapping, and HW1 navigation nodes.
Additional arguments are forwarded to the launch command.
