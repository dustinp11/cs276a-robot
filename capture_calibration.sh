#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
source "$script_dir/rubikpi_ros2/install/setup.bash"

exec python3 "$script_dir/calibration/capture_ros.py" "$@"
