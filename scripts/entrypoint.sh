#!/bin/bash

source /opt/ros/kilted/setup.bash
source /home/ubuntu/catkin_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
exec bash -c "$*"