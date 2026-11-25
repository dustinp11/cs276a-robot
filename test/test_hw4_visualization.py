"""Unit tests for trajectory loading, metric computation, and visualization comparison."""

import tempfile
from pathlib import Path
import unittest

from robot_testing.voronoi_planner import Environment, SafePathPlanner, VoronoiGraph
from robot_testing.hw4_visualization import (
    compute_trajectory_metrics,
    load_actual_trajectory,
    load_hw4_landmarks,
    plot_environment_and_path,
)


class TestHW4Visualization(unittest.TestCase):
    def setUp(self):
        self.env = Environment()
        self.vg = VoronoiGraph(self.env, boundary_sample_step_cm=2.0)
        self.planner = SafePathPlanner(self.env, self.vg)
        self.path = self.planner.plan((234.5, 63.0), (65.0, 229.5))
        self.landmarks = load_hw4_landmarks()

    def test_load_parenthesized_trajectory(self):
        """Verify parsing of standard (x, y) coordinates.txt format."""
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            f.write("(234.50, 63.00)\n")
            f.write("(200.12, 64.50)\n")
            f.write("(150.00, 66.80)\n")
            f.write("(69.50, 69.50)\n")
            f.write("(57.00, 126.00)\n")
            f.write("(57.10, 189.00)\n")
            f.write("(65.00, 229.50)\n")
            temp_path = f.name

        try:
            pts = load_actual_trajectory(temp_path)
            self.assertEqual(len(pts), 7)
            self.assertAlmostEqual(pts[0][0], 234.50, places=2)
            self.assertAlmostEqual(pts[0][1], 63.00, places=2)
            self.assertAlmostEqual(pts[-1][0], 65.00, places=2)
            self.assertAlmostEqual(pts[-1][1], 229.50, places=2)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_csv_trajectory(self):
        """Verify parsing of structured CSV trajectory format."""
        with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
            f.write("timestamp_s,x_cm,y_cm,yaw_deg,state,fresh\n")
            f.write("1795544658.10,234.50,63.00,0.00,DRIVING,1\n")
            f.write("1795544659.10,210.00,63.50,5.00,DRIVING,1\n")
            f.write("1795544660.10,180.00,65.00,5.00,DRIVING,1\n")
            f.write("1795544661.10,65.00,229.50,80.00,GOAL_REACHED,1\n")
            temp_path = f.name

        try:
            pts = load_actual_trajectory(temp_path)
            self.assertEqual(len(pts), 4)
            self.assertAlmostEqual(pts[0][0], 234.50, places=2)
            self.assertAlmostEqual(pts[-1][0], 65.00, places=2)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_trajectory_metrics_and_plot(self):
        """Verify metric calculation and figure generation."""
        sample_pts = [
            (234.5, 63.0),
            (190.0, 64.0),
            (140.0, 66.0),
            (70.0, 70.0),
            (57.0, 126.0),
            (57.1, 189.0),
            (65.0, 229.5),
        ]
        metrics = compute_trajectory_metrics(sample_pts, self.path, self.env)
        self.assertEqual(metrics['count'], 7)
        self.assertAlmostEqual(metrics['start_error_cm'], 0.0, places=2)
        self.assertAlmostEqual(metrics['goal_error_cm'], 0.0, places=2)
        self.assertGreater(metrics['actual_length_cm'], 200.0)
        self.assertLess(metrics['mean_cte_cm'], 5.0)  # Close to planned path

        with tempfile.NamedTemporaryFile('w', suffix='.png', delete=False) as f:
            out_img = f.name

        try:
            plot_environment_and_path(self.path, self.env, self.landmarks, sample_pts, out_img)
            self.assertTrue(Path(out_img).exists())
            self.assertGreater(Path(out_img).stat().st_size, 10000)
        finally:
            Path(out_img).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
