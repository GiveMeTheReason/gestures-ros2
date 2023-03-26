#!/bin/bash
set -e

# setup ros2 environment
echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
echo "source /root/ros_ws/install/setup.bash" >> ~/.bashrc

exec "$@"
