cd ~/robot_testing/rubikpi_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch apriltag_ros apriltag_launch.py \
  image_topic:=/camera/image_raw \
  camera_info_topic:=/camera/camera_info