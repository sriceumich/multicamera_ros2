#!/bin/bash

source /opt/ros/kilted/setup.bash
source /home/ubuntu/catkin_ws/install/setup.bash
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/catkin_ws/install/lib/python3.12/site-packages

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
exec bash -c "$*"