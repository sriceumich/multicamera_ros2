from setuptools import setup
package_name = 'sick_picocam2_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/multicam.launch.py']),
        ('share/' + package_name + '/config', ['config/cameras.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='You',
    maintainer_email='you@example.com',
    description='Publish image_raw + camera_info per SICK PICOcam2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'multicam_node = sick_picocam2_bridge.multicam_node:main',
            'gstreamercameranode = sick_picocam2_bridge.GStreamerCameraNode:main'
        ],
    },
)
