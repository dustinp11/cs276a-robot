#!/usr/bin/env bash
set -euo pipefail

# source the same ROS and vision overlay used by our existing camera scripts
hw3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$hw3_root/../scripts/lib/common.bash"
source_vision
source_setup "$hw3_root/install/setup.bash"
exec ros2 launch hw3_code hw3.launch.py "$@"
