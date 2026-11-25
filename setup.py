from setuptools import find_packages, setup

package_name = 'robot_testing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/hw1_solution.launch.py',
            'launch/hw2.launch.py',
            'launch/hw4_safe.launch.py',
            'launch/hw4_safe_navigation.launch.py',
            'launch/hw4_fast.launch.py',
            'launch/hw4_fast_navigation.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/landmarks.json',
            'config/hw4_landmarks.json',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_test = robot_testing.robot_test:main',
            'motor_control = robot_testing.motor_control:main',
            'velocity_mapping = robot_testing.velocity_mapping:main',
            'hw1_solution = robot_testing.hw1_solution:main',
            'hw2_control = robot_testing.hw2_control:main',
            'hw2_localization = robot_testing.hw2_localization:main',
            'voronoi_planner = robot_testing.voronoi_planner:main',
            'astar_planner = robot_testing.astar_planner:main',
            'hw4_localization = robot_testing.hw4_localization:main',
            'hw4_safe = robot_testing.hw4_safe:main',
            'hw4_control = robot_testing.hw4_safe:main',
            'hw4_fast = robot_testing.hw4_fast:main',
            'hw4_visualization = robot_testing.hw4_visualization:main',
            'hw4_fast_visualization = robot_testing.hw4_fast_visualization:main',
        ],
    },
)
