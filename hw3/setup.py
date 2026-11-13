from setuptools import find_packages, setup

package_name = 'hw3'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/hw3.launch.py',
            'launch/hw3_manual.launch.py',
            'launch/hw3_navigation.launch.py',
            'launch/hw3_octagon.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/hw3_params.yaml',
            '../config/camera_parameter.yaml',
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'pyserial'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='HW3 Visual Odometry, EKF-SLAM, and Testing Infrastructure',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'visual_odometry_node = hw3.nodes.visual_odometry_node:main',
            'ekf_slam_node = hw3.nodes.ekf_slam_node:main',
            'motor_control = hw3.nodes.motor_control:main',
            'velocity_mapping = hw3.nodes.velocity_mapping:main',
            'navigation_node = hw3.nodes.navigation_node:main',
            'record_data = hw3.scripts.record_data:main',
            'replay_data = hw3.scripts.replay_data:main',
            'export_map = hw3.scripts.export_map:main',
            'publish_map = hw3.nodes.publish_map_node:main',
        ],
    },
)
