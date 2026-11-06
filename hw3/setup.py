"""Package the independent HW3 ROS nodes and configuration."""

from glob import glob
from setuptools import find_packages, setup

setup(
    name='hw3_code',
    version='0.1.0',
    packages=find_packages(include=['hw3_code', 'hw3_code.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/hw3_code']),
        ('share/hw3_code', ['package.xml', 'README.md']),
        ('share/hw3_code/launch', glob('launch/*.launch.py')),
        ('share/hw3_code/config', glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Live square-route EKF-SLAM with Standard41h12 detection and map visualization.',
    license='LicenseRef-Proprietary',
    entry_points={'console_scripts': [
        'motor_control = hw3_code.motor_control:main',
        'velocity_mapping = hw3_code.velocity_mapping:main',
        'slam = hw3_code.slam_node:main',
        'plot_run = hw3_code.plot_run:main',
    ]},
)
