#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.bash"

source_vision
exec ros2 launch apriltag_ros apriltag_launch.py \
  image_topic:=/camera/image_raw \
  camera_info_topic:=/camera/camera_info "$@"
