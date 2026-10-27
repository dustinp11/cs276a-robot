cd ~/robot_testing/rubikpi_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch robot_vision_camera robot_vision_camera.launch.py \
  image_rectify:=true \
  camera_parameter_path:=/home/ubuntu/robot_testing/calibration/results/images_2026-10-26_classic/camera_parameter.yaml