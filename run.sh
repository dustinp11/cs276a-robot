cd ~/robot_testing
colcon build --packages-select robot_testing
source install/setup.bash
ros2 run robot_testing robot_test