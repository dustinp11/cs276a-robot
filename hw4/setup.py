"""Install the standalone HW4 package and measurement template."""

from setuptools import find_packages, setup


setup(
    name="hw4",
    version="0.1.0",
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/hw4"]),
        ("share/hw4", ["package.xml"]),
        ("share/hw4/config", ["config/environment.json", "config/motion.yaml", "config/navigation.yaml"]),
        ("share/hw4/launch", ["launch/hw4.launch.py"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="ubuntu@todo.todo",
    description="HW4 run 1: grid planning and known-map waypoint navigation",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "hw4_config = hw4.config:main",
        "hw4_navigation = hw4.mission_node:main",
    ]},
)
