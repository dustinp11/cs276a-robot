# re_calibrate

This standalone ROS 2 Python package publishes `[-0.5, 0.5]` to the
`motor_commands` topic for 1.27 seconds, then publishes `[0.0, 0.0]` to stop.
The `robot_testing` motor driver must be running; the command waits up to ten
seconds for its topic subscription and exits with an error if no driver appears.

Because this repository root is itself a ROS package, build this nested package
by explicitly selecting its directory:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths re_calibrate --packages-select re_calibrate
source install/setup.bash
# In a separate terminal, start the robot's serial motor driver:
ros2 run robot_testing motor_control
# Then run:
ros2 run re_calibrate turn_robot
```
