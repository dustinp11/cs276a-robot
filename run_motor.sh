#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")/rubikpi_ros2"

colcon build --packages-select robot_control
source install/setup.bash

echo "Starting motor_control. Press Ctrl+C to stop."
exec ros2 run robot_control motor_control
