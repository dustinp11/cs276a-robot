#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.bash"

usage() {
  cat <<EOF
Usage: $0 [TAG_SIZE_METERS [OUTPUT_DIRECTORY]]

Record a stationary camera and all visible 36h11 AprilTags. No motor nodes.
Default tag edge: 0.172 m; replace with your measured printed tag size.
Default output: <checkout>/runs/tags_<timestamp> (must not already exist).
Relative output paths are relative to your current working directory.
Press Ctrl-C once and wait for rosbag to finalize before copying the folder.
EOF
}

if [[ $# -eq 1 && ( "$1" == -h || "$1" == --help ) ]]; then
  usage
  exit 0
fi
if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

tag_size="${1:-0.172}"
if [[ ! "$tag_size" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
   ! awk -v size="$tag_size" 'BEGIN { exit !(size > 0) }'; then
  printf 'TAG_SIZE_METERS must be a positive number.\n' >&2
  exit 2
fi

require_calibration
source_vision
if ! ros2 bag record --help >/dev/null 2>&1; then
  printf 'rosbag2 is missing. Install: sudo apt install ros-jazzy-rosbag2\n' >&2
  exit 1
fi
for package in robot_vision_camera apriltag_ros; do
  ros2 pkg prefix "$package" >/dev/null
done

output="${2:-$repo_root/runs/tags_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$output" ]]; then
  printf 'Output already exists; choose a new directory: %s\n' "$output" >&2
  exit 2
fi
mkdir -p -- "$(dirname -- "$output")"
output="$(cd -- "$(dirname -- "$output")" && pwd)/$(basename -- "$output")"

printf 'Stationary 36h11 recording; tag size: %s m; camera: 5 fps.\n' "$tag_size"
printf 'Bag directory: %s\n' "$output"
printf 'No motors are launched. Stop other camera/navigation workflows first.\n'
printf 'Point a tag at the camera. Press Ctrl-C once to stop and finalize.\n'
exec ros2 launch "$repo_root/scripts/record_tags.launch.py" \
  tag_size_m:="$tag_size" calibration:="$camera_calibration" output:="$output"
