# Live AprilTag Detection: Project Handoff

## Purpose and current state

This note is intended to give a future engineer or AI agent enough context to
continue the live AprilTag detection test without relying on this chat. The
project is a ROS 2 workspace for a Rubik Pi camera. The goal is to detect
AprilTags for a future SLAM workflow, first confirming detection at a maximum
distance of about 7 ft (2.13 m).

**The live detection pipeline has not yet been verified on the robot.** The
camera became hot during testing and was left to cool. Do not start it again
until it is safe to do so.

The pipeline should be:

```text
Rubik Pi camera
  -> /camera/image_raw and /camera/camera_info
  -> apriltag_ros
  -> /detections and /tf
```

The AprilTag detector, camera, and Foxglove bridge have appeared in the ROS
graph. However, the detector has not been observed publishing detection
messages, and no detection records have been written.

## Repository and relevant files

Repository root on the development machine:

```text
/home/ubuntu/robot_testing
```

The robot commands below assume the repository is available at
`~/robot_testing` and its ROS workspace is `~/robot_testing/rubikpi_ros2`.

Relevant files:

- `scripts/run_camera.sh`: starts only the raw camera. The recommended workflow below uses
  `scripts/launch_camera.sh` for rectification; do not run these at the same time as
  another camera launch.
- `rubikpi_ros2/robot_vision/launch/robot_vision_camera.launch.py`: camera
  launch file; defaults to 1280x720 at 10 fps and also launches
  `foxglove_bridge`.
- `rubikpi_ros2/robot_vision/src/robot_vision_camera.cpp`: camera publishers,
  calibration loading, rectification, and message timestamps.
- `rubikpi_ros2/apriltag_ros/launch/apriltag_launch.py`: launch file for the
  AprilTag detector; its default config is `tags_36h11.yaml`.
- `rubikpi_ros2/apriltag_ros/cfg/tags_36h11.yaml`: example 36h11 config that
  currently filters to tag IDs 0 and 1.
- `calibration/results/images_2026-10-26_classic/camera_parameter.yaml`: latest
  camera calibration used in the live test.
- `calibration/results/images_2026-10-26_classic/calibration.json`: latest calibration
  report, including intrinsics and per-view checkerboard poses.
- `larger_april.png` and `smaller_april.png`: example distance images of the
  team's printed tags. Their small, blurred depictions have not established
  the tag family.

## Camera calibration and how it is used

The latest calibration is in
`calibration/results/images_2026-10-26_classic/`. It is for **1280x720**
images, reports an OpenCV `plumb_bob` model, uses a measured checkerboard
square size of **0.029 m**, and has an RMS reprojection error of about
**0.220 px**. Its camera matrix and distortion coefficients are in
`camera_parameter.yaml`.

The live camera node uses those values as follows:

1. It loads the YAML passed in `camera_parameter_path`.
2. With `image_rectify:=true`, it computes undistortion maps and rectifies the
   outgoing image.
3. It publishes `CameraInfo` alongside each image. The source code stamps the
   image and matching camera-info message using the same ROS timestamp.
4. `apriltag_ros` uses the rectified image and `CameraInfo` projection matrix
   to estimate each detected tag's pose relative to the camera.

The camera launch uses this calibration when its path is passed explicitly as
shown below. If no path is passed, it uses the different calibration bundled
with `robot_vision`.

The calibration JSON also contains `rvec_board_to_camera` and
`tvec_board_to_camera` for each checkerboard image. These describe the
checkerboard's pose in each calibration view, **not** the fixed camera-to-robot
mounting transform. The latest calibration used `square_size_m: 0.029`, so its
board-to-camera translations are expressed in meters.

For SLAM, the camera intrinsics alone are not all the required transforms. A
robot-frame estimate additionally needs the camera-to-robot/base mounting
transform, and the SLAM system needs consistent tag/map frames. AprilTag pose
estimation also needs each tag's correct physical edge length.

## Tag family, IDs, and physical size

The config name `tags_36h11.yaml` is an example configuration; it does not
prove that the team's printed tags are family 36h11. The selected config
contents seen during the discussion were:

- Family: `36h11`
- IDs: `0` and `1`
- Edge size: `0.165` m (16.5 cm)

The printed tag family and ID must be determined from the source/generator
that created the tag, or from a clear close-up and a reliable decode. Tests on
the small, blurry distance images returned inconsistent family candidates, so
they did **not** identify the family. Do not assume 36h11 just because the
repository has a 36h11 config.

Measure tag size along the square tag edge at the boundary between the black
tag and its white border. Do not measure the whole sheet or label. Convert
centimeters to meters by dividing by 100 (for example, 16.5 cm = 0.165 m).
Correct size is needed for metric pose; it does not select the tag family.

Important config behavior:

- `apriltag_launch.py` loads the `tags_36h11.yaml` from the active ROS
  workspace overlay by default.
- That YAML has `tag.ids: [0, 1]`; in this implementation, a nonempty ID list
  filters detections to those IDs. A tag with another ID can be ignored even
  if its family is correct. Its `tag.sizes` entries override the global
  `size`.
- The standalone detector publishes a detection array for each synchronized
  camera callback, including an empty array when it processes a frame but
  accepts no tag. Therefore no `/detections` messages at all is different
  from messages with `detections: []`.

## Starting the system

Run on the Rubik Pi, after the camera has cooled and only one person/process is
using it. The scripts source the ROS environment in each terminal. See
[scripts/README.md](scripts/README.md) for all entry points and path overrides.

### Three-terminal workflow

Use only this three-terminal workflow. Do not run `scripts/start_live.sh`,
`scripts/run_camera.sh`, or any other camera launch at the same time. Let the camera
cool before testing if it has become hot.

Before the first run, ensure the workspace copy of `apriltag_ros` is built.
This matters because the system package under `/opt/ros/jazzy` may not contain
this repository's `apriltag_launch.py`:

```bash
cd ~/robot_testing/rubikpi_ros2
source /opt/ros/jazzy/setup.bash
colcon build --packages-select apriltag_ros
source install/setup.bash
ros2 pkg prefix apriltag_ros
```

The package prefix should resolve inside
`~/robot_testing/rubikpi_ros2/install`, not `/opt/ros/jazzy`.

**Terminal 1 — camera with the latest calibration and rectification:**

```bash
cd ~/robot_testing
./scripts/launch_camera.sh
```

**Terminal 2 — detector:**

```bash
cd ~/robot_testing
./scripts/launch_apriltag.sh
```

This uses the workspace's 36h11 config with IDs 0 and 1. Before relying on it,
edit `apriltag_ros/cfg/tags_36h11.yaml` so family, IDs, and edge sizes match
the actual printed tag. Then rebuild and source the workspace:

```bash
cd ~/robot_testing/rubikpi_ros2
source /opt/ros/jazzy/setup.bash
colcon build --packages-select apriltag_ros
source install/setup.bash
```

Restart the detector after changing its configuration.

**Terminal 3 — inspect detector messages:**

```bash
cd ~/robot_testing
./scripts/monitor_detections.sh
```

An array containing a tag entry indicates a detection. An empty `detections`
list indicates a processed frame with no accepted tag. In Foxglove, use a Raw
Messages panel for `/detections`; it is data, not an annotated camera video.
An Image panel should use `/camera/image_raw/compressed`.

## Checks and commands

Start with process inventory so duplicate camera instances do not compete for
the hardware:

```bash
pgrep -af 'robot_vision_camera|apriltag_node|ros2 launch'
```

Stop an owned launch with Ctrl-C in its terminal when possible. A ROS launch
can leave a camera-node child behind; inspect exact PIDs before signaling:

```bash
ps -eo user,pid,ppid,stat,etime,args | grep -E '[r]obot_vision_camera|[a]priltag_node|[r]os2 launch'
```

Avoid broad `pkill`/`killall`; coordinate before stopping another person's
process. Use exact current PIDs only.

Useful graph/rate checks, with the system running:

```bash
ros2 node list
ros2 node info /apriltag
ros2 topic list
ros2 topic info /camera/image_raw --verbose
ros2 topic info /camera/camera_info --verbose
ros2 topic info /detections --verbose
ros2 topic hz /camera/image_raw
ros2 topic hz /camera/camera_info
ros2 topic hz /detections
ros2 topic echo --once /detections
```

`ros2 topic hz` waits for messages; leave it running for several seconds and
press Ctrl-C to see its measurements. A topic can appear in the ROS graph
without data being delivered.

## Observed troubleshooting history

These are observations reported during the session, not a confirmed root
cause:

- At one point two `/robot_vision_camera` nodes were running concurrently.
  Their launch/node PIDs were manually stopped; later `pgrep` showed none
  before a new `scripts/run_camera.sh` launch.
- ROS graph checks showed `/apriltag`, camera, and Foxglove bridge nodes.
  `ros2 node info /apriltag` showed subscriptions to
  `/camera/image_raw` and `/camera/camera_info`, and a `/detections` publisher.
- `ros2 topic info /detections` showed one publisher (`/apriltag`) and one
  subscriber during one check. Camera topics showed one publisher and two subscribers
  during one check.
- With the live detector setup, `/camera/image_raw` was measured at roughly
  0.6–2 Hz with large gaps, while `/camera/camera_info` was around 10 Hz.
  `/detections` had no observed rate, and `ros2 topic echo --once /detections`
  remained blank.
- With the camera started alone after process cleanup, `/camera/image_raw` was
  also measured around 1.5–2 Hz. This means the slow image delivery was
  observed without the AprilTag detector; it is not established that AprilTag
  caused it.
- Foxglove logs confirmed subscriptions to raw/compressed images and camera
  info, but did not prove image data was arriving. Foxglove previously logged
  `outbox ... full`, which indicates client streaming overload can also affect
  visualization.
- The camera became hot; testing was paused so it could cool. Thermal state
  and its relationship to low frame rate are not yet established.

## Recommended next diagnostic sequence

1. Let the camera cool. Start exactly one camera and AprilTag detector; verify
   process count first. Do not start a second camera using `scripts/run_camera.sh`.
2. Inspect the camera terminal for pipeline, calibration-loading, or
   rectification errors. Confirm the node reports the intended 1280x720 size
   and latest calibration path.
3. Measure image and camera-info rates again. Camera source code intends to
   stamp image and camera-info messages with the same timestamp per frame, but
   the large observed rate mismatch needs investigation. Confirm actual message
   timestamps and whether `apriltag_ros` receives synchronized pairs.
4. Check `ros2 node info /apriltag` and detailed topic endpoint QoS. Confirm
   the detector is subscribed to the same image and camera-info topics with
   compatible QoS. Do not infer successful delivery only from graph presence.
5. If image callbacks reach AprilTag but `/detections` still emits no arrays,
   inspect detector terminal logs and image/camera-info synchronization. If
   arrays arrive but contain no tags, then check family, ID filters, physical
   tag size, tag pixel footprint, focus, lighting, and detector tuning.
6. During diagnosis, temporarily disconnect Foxglove or close image panels to
   reduce network/CPU load. This is a diagnostic step, not a proven fix.
7. Test a known-family tag close to the camera first. Only after successful
   close-range detection should detection be assessed at 7 ft.

Do not claim the 7-ft detection test, camera calibration on the live stream,
or AprilTag-to-robot extrinsics are validated until tested on the hardware.
