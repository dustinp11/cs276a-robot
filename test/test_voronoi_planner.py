"""Unit tests for Voronoi safe path planning (HW4)."""

import math
import unittest
import numpy as np

from robot_testing.voronoi_planner import (
    Environment,
    Obstacle,
    RobotFootprint,
    SafePathPlanner,
    VoronoiGraph,
    Workspace,
)


class TestVoronoiPlanner(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace(x_min=0.0, x_max=300.0, y_min=0.0, y_max=292.5)
        self.obstacle = Obstacle(x_min=114.0, x_max=176.0, y_min=123.0, y_max=185.0)
        self.robot = RobotFootprint(width_cm=16.0, length_cm=16.0)
        self.env = Environment(self.workspace, self.obstacle, self.robot)
        self.vg = VoronoiGraph(self.env, boundary_sample_step_cm=2.5)
        self.planner = SafePathPlanner(self.env, self.vg)

        self.start = (234.5, 63.0)
        self.goal = (65.0, 229.5)

    def test_start_and_goal_validity(self):
        """1. Verify Start and Goal are valid, collision-free positions."""
        c_start = self.env.point_clearance(*self.start)
        c_goal = self.env.point_clearance(*self.goal)
        req = self.robot.conservative_radius_cm

        self.assertGreater(c_start, req, "Start must have clearance > robot radius")
        self.assertGreater(c_goal, req, "Goal must have clearance > robot radius")
        self.assertTrue(self.workspace.contains_point(*self.start, margin=req))
        self.assertTrue(self.workspace.contains_point(*self.goal, margin=req))
        self.assertFalse(self.obstacle.contains_point(*self.start, margin=req))
        self.assertFalse(self.obstacle.contains_point(*self.goal, margin=req))

    def test_robot_footprint_geometry_and_camera_offset(self):
        """Verify camera reference offset and conservative radius."""
        r = self.robot.conservative_radius_cm
        # sqrt(16^2 + 8^2) = sqrt(320) ≈ 17.8885
        self.assertAlmostEqual(r, math.hypot(16.0, 8.0), places=3)

        # Check local corners with camera at (0, 0)
        corners = self.robot.local_corners()
        self.assertEqual(len(corners), 4)
        self.assertEqual(corners[0], (0.0, 8.0))      # Front-Left
        self.assertEqual(corners[1], (0.0, -8.0))     # Front-Right
        self.assertEqual(corners[2], (-16.0, -8.0))   # Rear-Right
        self.assertEqual(corners[3], (-16.0, 8.0))    # Rear-Left

        # Rotate robot and verify world corners
        w_corners_0 = self.robot.world_corners(100.0, 100.0, 0.0)
        self.assertEqual(w_corners_0[0], (100.0, 108.0))
        self.assertEqual(w_corners_0[2], (84.0, 92.0))

        # Check exact collision checking with obstacle
        # A pose where camera is at (115, 154) facing east (0 rad): rear body is at x=99..115 (free)
        # But front edge touches obstacle at x=115 -> collision
        self.assertTrue(self.robot.is_collision(115.0, 154.0, 0.0, self.workspace, self.obstacle))
        # Clear pose far from obstacle
        self.assertFalse(self.robot.is_collision(50.0, 50.0, 0.0, self.workspace, self.obstacle))

    def test_voronoi_graph_connectivity(self):
        """2 & 4. Voronoi graph is generated and connects free space."""
        self.assertGreater(len(self.vg.valid_vertices), 50)
        # Graph should have non-empty adjacency
        total_edges = sum(len(edges) for edges in self.vg.adj.values())
        self.assertGreater(total_edges, 100)

    def test_planned_path_endpoints(self):
        """7. Path starts and ends exactly at specified coordinates."""
        path = self.planner.plan(self.start, self.goal)
        self.assertAlmostEqual(path.smoothed_waypoints[0][0], self.start[0], places=2)
        self.assertAlmostEqual(path.smoothed_waypoints[0][1], self.start[1], places=2)
        self.assertAlmostEqual(path.smoothed_waypoints[-1][0], self.goal[0], places=2)
        self.assertAlmostEqual(path.smoothed_waypoints[-1][1], self.goal[1], places=2)

    def test_footprint_remains_inside_workspace_and_avoids_obstacle(self):
        """2 & 3. Robot full footprint remains in workspace and avoids obstacle along entire path."""
        path = self.planner.plan(self.start, self.goal)
        smoothed = path.smoothed_waypoints

        for i in range(len(smoothed) - 1):
            p1, p2 = smoothed[i], smoothed[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            heading = math.atan2(dy, dx)
            for t in np.linspace(0.0, 1.0, 20):
                x = (1.0 - t) * p1[0] + t * p2[0]
                y = (1.0 - t) * p1[1] + t * p2[1]
                # Check collision with exact rotated polygonal footprint
                in_collision = self.robot.is_collision(x, y, heading, self.workspace, self.obstacle)
                self.assertFalse(in_collision, f"Collision detected at ({x:.2f}, {y:.2f})")
                # Footprint clearance must be strictly positive
                footprint_c = self.robot.min_clearance(x, y, heading, self.workspace, self.obstacle)
                self.assertGreater(footprint_c, 10.0, "Footprint must maintain substantial clearance")

    def test_path_clearance_and_bottleneck(self):
        """5. Path maximizes minimum clearance."""
        path = self.planner.plan(self.start, self.goal)
        # Corridor bottleneck is ~57 cm
        self.assertGreater(path.bottleneck_clearance_cm, 50.0)
        self.assertGreater(path.min_clearance_cm, 50.0)
        self.assertGreater(path.average_clearance_cm, 55.0)

    def test_smoothing_does_not_introduce_collisions(self):
        """6. Smoothing preserves clearance and introduces no collisions."""
        path = self.planner.plan(self.start, self.goal, smoothing_tolerance_cm=2.0)
        self.assertLess(len(path.smoothed_waypoints), len(path.raw_waypoints))
        self.assertGreater(path.min_clearance_cm, path.bottleneck_clearance_cm - 2.5)

    def test_direct_line_fails(self):
        """Compare with direct start-to-goal line to demonstrate necessity of safe planner."""
        direct_c = self.env.segment_clearance(self.start, self.goal, samples=50)
        self.assertLess(direct_c, 0.0, "Direct path must intersect obstacle (negative clearance)")


if __name__ == '__main__':
    unittest.main()
