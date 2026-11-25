"""Safe Voronoi-Based Path Planning and Closed-Loop Navigation Node (hw4_safe).

This node implements the safe path planner for Homework 4 using a Voronoi-based
approach maximizing obstacle clearance, with closed-loop visual waypoint following
via known AprilTag landmark coordinates.
"""

from robot_testing.hw4_control import (
    HW4ControlNode,
    HW4Navigator,
    NavigationState,
    main as control_main,
    wrap_angle,
)

# Export HW4SafeNode as an alias of the safe Voronoi controller node
HW4SafeNode = HW4ControlNode


def main(args=None):
    """Entry point for the renamed hw4_safe ROS 2 executable."""
    control_main(args=args)


if __name__ == '__main__':
    main()
