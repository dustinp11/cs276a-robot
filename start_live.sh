#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
camera_calibration="$script_dir/calibration/results/images_2026-10-26_classic/camera_parameter.yaml"

usage() {
  cat <<EOF
Usage: $0 FAMILY TAG_SIZE_METERS

Start the calibrated camera and live AprilTag detection. Non-empty detections
are appended to detections.txt. FAMILY must match the printed tag.
TAG_SIZE_METERS is the edge length
from the black/white border boundary, required for correct pose estimates.

Supported families:
  16h5 25h9 36h11 Circle21h7 Circle49h12 Custom48h12
  Standard41h12 Standard52h13

Example:
  $0 36h11 0.165
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

family="$1"
tag_size="$2"
case "$family" in
  16h5|25h9|36h11|Circle21h7|Circle49h12|Custom48h12|Standard41h12|Standard52h13)
    ;;
  *)
    printf 'Unsupported AprilTag family: %s\n' "$family" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "$tag_size" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
   ! awk -v size="$tag_size" 'BEGIN { exit !(size > 0) }'; then
  printf 'TAG_SIZE_METERS must be a positive number.\n' >&2
  exit 2
fi

if [[ ! -f "$camera_calibration" ]]; then
  printf 'Camera calibration file not found: %s\n' "$camera_calibration" >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
source "$script_dir/rubikpi_ros2/install/setup.bash"
set -u

params_file="$(mktemp)"
cat >"$params_file" <<EOF
apriltag:
  ros__parameters:
    image_transport: raw
    family: "$family"
    size: $tag_size
    max_hamming: 1
    detector:
      threads: 4
      decimate: 1.0
      blur: 0.0
      refine: true
      sharpening: 0.25
      debug: false
    pose_estimation_method: "pnp"
EOF

camera_pid=
detector_pid=
logger_pid=
detections_file="$script_dir/detections.txt"
cleanup() {
  trap - EXIT INT TERM
  for pid in "$logger_pid" "$detector_pid" "$camera_pid"; do
    if [[ -n "$pid" ]]; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$logger_pid" "$detector_pid" "$camera_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -f "$params_file"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting calibrated camera and AprilTag detection.\n'
printf 'Family: %s; tag edge size: %s m; rectification: enabled.\n' "$family" "$tag_size"
printf 'Non-empty detections will be appended to %s\n' "$detections_file"
printf 'Press Ctrl-C to stop all processes.\n'

ros2 launch robot_vision_camera robot_vision_camera.launch.py \
  image_rectify:=true \
  camera_parameter_path:="$camera_calibration" &
camera_pid=$!

ros2 run apriltag_ros apriltag_node --ros-args \
  -r image_rect:=/camera/image_raw \
  -r camera_info:=/camera/camera_info \
  --params-file "$params_file" &
detector_pid=$!

python3 - "$detections_file" <<'PY' &
import json
import sys

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.node import Node


class DetectionLogger(Node):
    def __init__(self, path):
        super().__init__("apriltag_detection_logger")
        self.output = open(path, "a", encoding="utf-8", buffering=1)
        self.subscription = self.create_subscription(
            AprilTagDetectionArray, "/detections", self.on_detections, 10
        )

    def on_detections(self, message):
        stamp = message.header.stamp
        for detection in message.detections:
            record = {
                "stamp_sec": stamp.sec,
                "stamp_nanosec": stamp.nanosec,
                "family": detection.family,
                "id": detection.id,
                "hamming": detection.hamming,
                "decision_margin": detection.decision_margin,
                "center": [detection.centre.x, detection.centre.y],
                "corners": [[point.x, point.y] for point in detection.corners],
            }
            self.output.write(json.dumps(record) + "\n")

    def destroy_node(self):
        self.output.close()
        super().destroy_node()


rclpy.init()
node = DetectionLogger(sys.argv[1])
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
PY
logger_pid=$!

wait "$logger_pid"
