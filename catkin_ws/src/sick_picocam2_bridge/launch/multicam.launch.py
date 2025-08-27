from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import ThisLaunchFileDir
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis
import os
import yaml


def generate_launch_description():
    Aravis.update_device_list()
    sick_camera_IDs = []
    for i in range(Aravis.get_n_devices()):
        sick_camera_IDs.append(f"aravis:{Aravis.get_device_id(i)}")

    pkg_share = get_package_share_directory('sick_picocam2_bridge')
    camera_config = os.path.join(pkg_share, 'config', 'cameras.yaml')
    
    #Update camera config automatically 
    with open(camera_config, 'r') as file:
        data = yaml.safe_load(file)
    data["sick_picocam2_multicam"]["ros__parameters"]["streams"] = sick_camera_IDs

    with open(camera_config, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=camera_config,
                              description='Path to param YAML'),
        Node(
            package='sick_picocam2_bridge',
            executable='multicam_node',
            name='sick_picocam2_multicam',
            parameters=[LaunchConfiguration('config')],
            output='screen'
        )
    ])
