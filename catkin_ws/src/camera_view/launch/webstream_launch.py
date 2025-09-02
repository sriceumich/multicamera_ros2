from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_view',
            executable='webstream_node',
            name='wildlife_webstream',
            output='screen'
        )
    ])
