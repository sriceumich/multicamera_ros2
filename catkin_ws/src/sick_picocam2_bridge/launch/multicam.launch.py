from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import ThisLaunchFileDir
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('sick_picocam2_bridge')
    default_cfg = os.path.join(pkg_share, 'config', 'cameras.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_cfg,
                              description='Path to param YAML'),
        Node(
            package='sick_picocam2_bridge',
            executable='multicam_node',
            name='sick_picocam2_multicam',
            parameters=[LaunchConfiguration('config')],
            output='screen'
        )
    ])
