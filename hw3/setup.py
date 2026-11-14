from setuptools import find_packages, setup

package_name = 'hw3'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
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
    description='HW3 Visual Odometry, EKF-SLAM, and Robot Navigation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'visual_odometry_node = hw3.slam_vision:visual_odometry_main',
            'ekf_slam_node = hw3.slam_vision:ekf_slam_main',
            'motor_control = hw3.motor_control:motor_control_main',
            'velocity_mapping = hw3.motor_control:velocity_mapping_main',
            'navigation_node = hw3.square_navigation:navigation_node_main',
            'square_navigation_node = hw3.square_navigation:navigation_node_main',
            'octagon_navigation_node = hw3.octagon_navigation:octagon_navigation_main',
            'record_data = hw3.slam_vision:record_data_main',
            'replay_data = hw3.slam_vision:replay_data_main',
            'export_map = hw3.slam_vision:export_map_main',
            'publish_map = hw3.slam_vision:publish_map_main',
        ],
    },
)
