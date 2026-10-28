#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.bash"

source_ros
cd "$repo_root"
colcon build --packages-select robot_testing
source_setup "$repo_root/install/setup.bash"
exec ros2 launch robot_testing hw1_solution.launch.py "$@"
