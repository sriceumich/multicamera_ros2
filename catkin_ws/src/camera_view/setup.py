from setuptools import setup, find_packages

package_name = 'camera_view'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(
        exclude=['camera_view.static', 'camera_view.static.*']
    ),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/webstream_launch.py']),
    ],
    install_requires=[
        'setuptools',
        'multi_rtsp_cam_pkg',
        'fastapi',
        'uvicorn',
        'opencv-python',
        'numpy',
        'pyyaml'
    ],
    include_package_data=True,
    package_data={
        'camera_view': [
            'static/**/*',   # all static assets recursively
            'ssl/*',         # cert.pem + key.pem
        ],
    },
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='WebSocket/HTTP viewer node for wildlife detector streams',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'webstream_node = camera_view.webstream_node:main'
        ],
    },
)
