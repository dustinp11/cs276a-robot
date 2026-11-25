"""Unit tests for HW4 Fast A* path planning algorithm."""

import math
import unittest

from robot_testing.astar_planner import AStarPlanner, OccupancyGrid
from robot_testing.voronoi_planner import Environment, SafePathPlanner, VoronoiGraph


class TestAStarPlanner(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.planner = AStarPlanner(self.env, resolution_cm=1.0)
        self.start = (234.5, 63.0)
        self.goal = (65.0, 229.5)

    def test_occupancy_grid_cspace_dilation(self):
        """Verify obstacle and boundary inflation by robot conservative radius R=17.89 cm."""
        grid = self.planner.grid
        r = self.env.robot.conservative_radius_cm

        # Center of obstacle is occupied
        cx, cy = self.env.obstacle.center
        gx, gy = grid.world_to_grid(cx, cy)
        self.assertFalse(grid.is_free(gx, gy))

        # Right outside physical obstacle, within R_robot, must be occupied (C-space buffer)
        # Obstacle x_min is 114, y_min is 123.
        # Point at (110, 123) is 4 cm away from obstacle (< 17.89 cm) -> must be occupied
        gx, gy = grid.world_to_grid(110.0, 123.0)
        self.assertFalse(grid.is_free(gx, gy))

        # Far away free point (e.g. start position (234.5, 63.0)) must be free
        gx, gy = grid.world_to_grid(self.start[0], self.start[1])
        self.assertTrue(grid.is_free(gx, gy))

        # Goal position (65.0, 229.5) must be free
        gx, gy = grid.world_to_grid(self.goal[0], self.goal[1])
        self.assertTrue(grid.is_free(gx, gy))

        # Outside workspace bounds must not be free
        gx, gy = grid.world_to_grid(5.0, 5.0)  # less than 17.89 cm from wall
        self.assertFalse(grid.is_free(gx, gy))

    def test_astar_path_validity_and_safety(self):
        """Verify A* finds a collision-free path between start and goal."""
        path = self.planner.plan(self.start, self.goal)

        self.assertIsNotNone(path)
        self.assertGreater(len(path.raw_waypoints), 10)
        self.assertGreaterEqual(path.waypoint_count, 2)

        # Starts at start, ends at goal
        self.assertAlmostEqual(path.smoothed_waypoints[0][0], self.start[0], places=2)
        self.assertAlmostEqual(path.smoothed_waypoints[0][1], self.start[1], places=2)
        self.assertAlmostEqual(path.smoothed_waypoints[-1][0], self.goal[0], places=2)
        self.assertAlmostEqual(path.smoothed_waypoints[-1][1], self.goal[1], places=2)

        # Every point on the smoothed segments must be collision-free
        robot = self.env.robot
        r_robot = robot.conservative_radius_cm
        self.assertGreaterEqual(path.min_clearance_cm, r_robot - 0.05)

        for i in range(len(path.smoothed_waypoints) - 1):
            p1 = path.smoothed_waypoints[i]
            p2 = path.smoothed_waypoints[i + 1]
            heading = path.headings_rad[i]
            # Exact polygon footprint sweep
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            samples = int(dist * 2) + 2
            for s in range(samples + 1):
                t = s / samples
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                collided = robot.is_collision(x, y, heading, self.env.workspace, self.env.obstacle)
                self.assertFalse(collided, f"Collision detected at ({x:.2f}, {y:.2f}) on segment {i}")

    def test_astar_is_shorter_than_voronoi(self):
        """Verify Fast A* produces a shorter route than the safe Voronoi planner."""
        fast_path = self.planner.plan(self.start, self.goal)

        vg = VoronoiGraph(self.env, boundary_sample_step_cm=2.0)
        safe_planner = SafePathPlanner(self.env, vg)
        safe_path = safe_planner.plan(self.start, self.goal)

        print(f"\n[Comparison] Fast A* length: {fast_path.total_length_cm:.2f} cm | "
              f"Safe Voronoi length: {safe_path.total_length_cm:.2f} cm")
        print(f"[Comparison] Fast A* min clearance: {fast_path.min_clearance_cm:.2f} cm | "
              f"Safe Voronoi min clearance: {safe_path.min_clearance_cm:.2f} cm")

        # Fast A* must be shorter by at least 30 cm
        self.assertLess(fast_path.total_length_cm, safe_path.total_length_cm - 30.0)

        # Safe Voronoi must have significantly higher clearance (bottleneck safety)
        self.assertGreater(safe_path.min_clearance_cm, fast_path.min_clearance_cm + 25.0)

    def test_direct_collision_detection(self):
        """Confirm that direct straight line between start and goal is indeed in collision."""
        # Straight line between start and goal intersects obstacle
        direct_c = self.env.segment_clearance(self.start, self.goal, samples=200)
        self.assertLess(direct_c, 0.0)  # Penetrates obstacle boundary


if __name__ == '__main__':
    unittest.main()
