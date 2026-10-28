#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.bash"

# Raw camera stream for calibration capture (no image rectification).
source_vision
exec ros2 launch robot_vision_camera robot_vision_camera.launch.py image_rectify:=false "$@"
