# Shared paths and ROS setup for the Bash entry points in scripts/.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
vision_workspace="$repo_root/rubikpi_ros2"
camera_calibration="${CAMERA_CALIBRATION:-$repo_root/calibration/results/images_2026-10-26_classic/camera_parameter.yaml}"

source_setup() {
  if [[ ! -f "$1" ]]; then
    printf 'Setup file not found: %s\nInstall ROS or build the workspace first (see scripts/README.md).\n' "$1" >&2
    exit 1
  fi
  # Generated ROS setup scripts may reference unset variables.
  set +u
  source "$1"
  set -u
}

source_ros() {
  source_setup "${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
}

source_vision() {
  source_ros
  source_setup "$vision_workspace/install/setup.bash"
}

require_calibration() {
  if [[ ! -f "$camera_calibration" ]]; then
    printf 'Camera calibration file not found: %s\n' "$camera_calibration" >&2
    exit 1
  fi
}
