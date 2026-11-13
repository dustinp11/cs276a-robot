#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
colcon build --packages-select robot_testing
source install/setup.bash
exec ros2 run robot_testing robot_test
