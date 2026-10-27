from setuptools import find_packages, setup

package_name = 're_calibrate'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Publish a fixed motor turn command for 1.27 seconds.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'turn_robot = re_calibrate.turn_robot:main',
        ],
    },
)
