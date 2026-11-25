"""Visualization and Trajectory Comparison for HW4 Fast A* Planner.

Visualizes:
1. Workspace boundaries ([0, 300] x [0, 292.5] cm).
2. Actual obstacle ([114, 176] x [123, 185] cm).
3. Configuration-space inflated obstacle and boundary clearance zones (R = 17.89 cm).
4. Discrete Occupancy Grid (free vs impassable cells).
5. All 12 AprilTag landmarks (36h11 wall tags and 41h12 obstacle tags) with IDs and facing vectors.
6. Start (234.5, 63) and Goal (65, 229.5) coordinates.
7. Raw A* grid search path and final smoothed shortest-distance path.
8. Physical robot footprints (16 x 16 cm) and camera reference points along waypoints.
9. Actual robot trajectory overlay (when hw4_fast_trajectory.txt exists) with comparison metrics.
"""

import argparse
import math
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from robot_testing.astar_planner import AStarPlanner, FastPlannedPath, OccupancyGrid
from robot_testing.hw4_visualization import (
    compute_trajectory_metrics,
    load_hw4_landmarks,
)
from robot_testing.voronoi_planner import Environment, SafePathPlanner, VoronoiGraph, _point_to_segment_distance


def load_fast_trajectory(file_path: Optional[str] = None) -> List[Tuple[float, float]]:
    """Load recorded robot coordinates from fast trajectory files."""
    candidates = []
    if file_path:
        candidates.append(Path(file_path))
    pkg_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        Path('hw4_fast_trajectory.txt'),
        pkg_root / 'hw4_fast_trajectory.txt',
        Path('hw4_fast_actual_trajectory.csv'),
        pkg_root / 'hw4_fast_actual_trajectory.csv',
        Path('coordinates.txt'),
        pkg_root / 'coordinates.txt',
    ])

    found_path = None
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            found_path = p
            break

    if found_path is None:
        return []

    pts = []
    float_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

    with open(found_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('timestamp'):
                continue

            numbers = [float(n) for n in float_pattern.findall(line)]
            if len(numbers) >= 2:
                if line.count(',') >= 2 and len(numbers) >= 3 and numbers[0] > 1000:
                    x, y = numbers[1], numbers[2]
                else:
                    x, y = numbers[0], numbers[1]

                if abs(x) < 10.0 and abs(y) < 10.0:
                    x *= 100.0
                    y *= 100.0

                if -50.0 <= x <= 350.0 and -50.0 <= y <= 350.0:
                    pts.append((x, y))

    print(f"Loaded {len(pts)} recorded trajectory points from: {found_path}")
    return pts


def plot_fast_environment_and_path(
    fast_path: FastPlannedPath,
    env: Environment,
    landmarks_data: dict,
    actual_trajectory: Optional[List[Tuple[float, float]]] = None,
    safe_path: Optional[object] = None,
    output_filename: str = 'hw4_fast_path_visualization.png',
):
    """Generate high-resolution visualization of HW4 Fast A* path planning."""
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    ax.set_aspect('equal')
    ax.set_xlim(-20, 320)
    ax.set_ylim(-20, 315)

    ws = env.workspace
    obs = env.obstacle
    robot = env.robot
    r_robot = robot.conservative_radius_cm

    grid_obj = OccupancyGrid(env, resolution_cm=fast_path.grid_resolution_cm)
    has_actual = bool(actual_trajectory and len(actual_trajectory) > 1)

    # 1. Occupancy Grid background
    # Transpose grid for imshow (x horizontal, y vertical)
    grid_img = np.zeros((grid_obj.ny, grid_obj.nx, 4), dtype=float)
    # Impassable / C-space inflated cells: subtle light red
    grid_img[grid_obj.grid.T, :] = [0.99, 0.90, 0.90, 0.85]
    # Free cells: subtle light slate
    grid_img[~grid_obj.grid.T, :] = [0.98, 0.99, 1.00, 1.00]

    ax.imshow(
        grid_img,
        origin='lower',
        extent=[ws.x_min, ws.x_max, ws.y_min, ws.y_max],
        zorder=0,
        interpolation='nearest'
    )

    # 2. Workspace boundary
    ax.add_patch(patches.Rectangle(
        (ws.x_min, ws.y_min), ws.width, ws.height,
        linewidth=2.5, edgecolor='#1e293b', facecolor='none', zorder=1,
        label=f'Workspace (10 ft: {ws.width:.1f} x {ws.height:.1f} cm)'
    ))

    # Boundary clearance margin
    ax.add_patch(patches.Rectangle(
        (ws.x_min + r_robot, ws.y_min + r_robot),
        ws.width - 2 * r_robot, ws.height - 2 * r_robot,
        linewidth=1.2, edgecolor='#94a3b8', linestyle='--', facecolor='none', zorder=2,
        label=f'Boundary Clearance Margin (R = {r_robot:.1f} cm)'
    ))

    # 3. Inflated Obstacle (C-Space)
    inf_x = obs.x_min - r_robot
    inf_y = obs.y_min - r_robot
    inf_w = obs.width + 2 * r_robot
    inf_h = obs.height + 2 * r_robot
    inflated_box = patches.FancyBboxPatch(
        (inf_x, inf_y), inf_w, inf_h,
        boxstyle=f"round,pad=0,rounding_size={r_robot:.1f}",
        linewidth=1.8, edgecolor='#dc2626', linestyle='--', facecolor='#fecaca', alpha=0.5,
        zorder=3, label=f'C-Space Inflated Obstacle (dilated by {r_robot:.1f} cm)'
    )
    ax.add_patch(inflated_box)

    # 4. Actual Physical Obstacle
    ax.add_patch(patches.Rectangle(
        (obs.x_min, obs.y_min), obs.width, obs.height,
        linewidth=2.0, edgecolor='#991b1b', facecolor='#ef4444', alpha=0.9,
        zorder=4, label=f'Actual Obstacle ({obs.width:.0f} x {obs.height:.0f} cm box)'
    ))
    ax.text(obs.center[0], obs.center[1], 'OBSTACLE\n62 x 62 cm',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=5)

    # 5. Direct start-to-goal path (showing collision)
    start = fast_path.smoothed_waypoints[0]
    goal = fast_path.smoothed_waypoints[-1]
    ax.plot([start[0], goal[0]], [start[1], goal[1]],
            color='#b91c1c', linestyle=':', linewidth=2.0, zorder=6,
            label='Direct Path (Intersects Obstacle - Unsafe!)')
    mid_x = (start[0] + goal[0]) / 2.0
    mid_y = (start[1] + goal[1]) / 2.0
    ax.scatter(mid_x, mid_y, marker='x', s=160, color='#991b1b', linewidths=3.0, zorder=7)

    # 6. Raw A* Grid Path
    raw_x = [p[0] for p in fast_path.raw_waypoints]
    raw_y = [p[1] for p in fast_path.raw_waypoints]
    ax.plot(raw_x, raw_y, color='#0284c7', linewidth=1.5, linestyle=':', alpha=0.6,
            zorder=8, label=f'Raw A* Grid Path ({len(fast_path.raw_waypoints)} cells)')

    # Optional: Safe Voronoi Path reference if provided for comparison
    if safe_path is not None:
        safe_x = [p[0] for p in safe_path.smoothed_waypoints]
        safe_y = [p[1] for p in safe_path.smoothed_waypoints]
        ax.plot(safe_x, safe_y, color='#059669', linewidth=2.2, linestyle='-.', alpha=0.7,
                zorder=9, label=f'Safe Voronoi Route ({safe_path.total_length_cm:.1f} cm, clr={safe_path.min_clearance_cm:.1f} cm)')

    # 7. Final Smoothed Fast A* Path
    smooth_x = [p[0] for p in fast_path.smoothed_waypoints]
    smooth_y = [p[1] for p in fast_path.smoothed_waypoints]
    fast_style = '--' if has_actual else '-'
    ax.plot(smooth_x, smooth_y, color='#7c3aed', linewidth=3.5, linestyle=fast_style,
            solid_capstyle='round', zorder=10,
            label=f'Fast A* Shortest Path ({fast_path.waypoint_count} WPs, {fast_path.total_length_cm:.1f} cm)')

    # 8. Robot Footprints at Planned Waypoints
    for idx, (wp_x, wp_y) in enumerate(fast_path.smoothed_waypoints):
        heading = fast_path.headings_rad[idx]
        corners = robot.world_corners(wp_x, wp_y, heading)
        poly = patches.Polygon(corners, closed=True, linewidth=1.5,
                               edgecolor='#6d28d9', facecolor='#ddd6fe', alpha=0.55, zorder=11)
        ax.add_patch(poly)

        # Camera reference point
        ax.scatter(wp_x, wp_y, color='#5b21b6', s=45, zorder=12)

        # Forward orientation vector
        arrow_len = 10.0
        ax.arrow(wp_x, wp_y, arrow_len * math.cos(heading), arrow_len * math.sin(heading),
                 head_width=3.5, head_length=3.5, fc='#5b21b6', ec='#5b21b6', zorder=12)

        # Waypoint label
        lbl_offset_x = 8.0 if wp_x < 150 else -8.0
        lbl_offset_y = 8.0 if wp_y < 150 else -8.0
        c_val = env.point_clearance(wp_x, wp_y)
        ax.annotate(
            f"WP {idx}\n({wp_x:.1f}, {wp_y:.1f})\nclr={c_val:.1f}cm",
            (wp_x, wp_y),
            xytext=(wp_x + lbl_offset_x, wp_y + lbl_offset_y),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#7c3aed", alpha=0.9),
            fontsize=8, fontweight='semibold', zorder=15,
            ha='left' if lbl_offset_x > 0 else 'right'
        )

    # 9. Actual Trajectory (if available)
    metrics = {}
    if has_actual:
        metrics = compute_trajectory_metrics(actual_trajectory, fast_path, env)
        act_x = [p[0] for p in actual_trajectory]
        act_y = [p[1] for p in actual_trajectory]

        ax.plot(
            act_x, act_y,
            color='#2563eb',
            linewidth=2.8,
            linestyle='-',
            marker='o',
            markersize=4.0,
            markerfacecolor='#1d4ed8',
            markeredgecolor='white',
            markeredgewidth=0.6,
            alpha=0.95,
            zorder=16,
            label=f'Actual Robot Trajectory ({metrics["duration_s"]}s, {metrics["actual_length_cm"]:.1f} cm)'
        )

        # Travel progression arrows
        step_stride = max(2, len(actual_trajectory) // 10)
        for i in range(0, len(actual_trajectory) - 1, step_stride):
            p_curr = actual_trajectory[i]
            p_next = actual_trajectory[min(i + step_stride, len(actual_trajectory) - 1)]
            dx = p_next[0] - p_curr[0]
            dy = p_next[1] - p_curr[1]
            dist = math.hypot(dx, dy)
            if dist > 3.0:
                ax.annotate(
                    '',
                    xy=(p_curr[0] + dx * 0.6, p_curr[1] + dy * 0.6),
                    xytext=(p_curr[0], p_curr[1]),
                    arrowprops=dict(arrowstyle="->", color='#1e40af', lw=2.0),
                    zorder=17
                )

        ax.scatter(act_x[0], act_y[0], marker='o', s=150, color='#3b82f6',
                   edgecolors='#1e3a8a', linewidths=2.5, zorder=18,
                   label=f'Actual Start ({act_x[0]:.1f}, {act_y[0]:.1f}) cm')
        ax.scatter(act_x[-1], act_y[-1], marker='D', s=160, color='#9333ea',
                   edgecolors='black', linewidths=2.0, zorder=18,
                   label=f'Actual Finish ({act_x[-1]:.1f}, {act_y[-1]:.1f}) cm')

    # 10. AprilTags (Wall and Obstacle)
    wall_plotted = False
    obs_tag_plotted = False

    for lm in landmarks_data.get('landmarks', []):
        tid = lm['id']
        family = lm['family']
        x_cm = lm['x_m'] * 100.0
        y_cm = lm['y_m'] * 100.0
        facing_deg = lm['facing_deg']
        facing_rad = math.radians(facing_deg)

        is_wall = (family == '36h11')
        color = '#2563eb' if is_wall else '#d97706'
        marker = 's' if is_wall else 'D'
        tag_type = 'Wall Tag (36h11, 17.2cm)' if is_wall else 'Obstacle Tag (41h12, 18.7cm)'

        lbl = None
        if is_wall and not wall_plotted:
            lbl = tag_type
            wall_plotted = True
        elif not is_wall and not obs_tag_plotted:
            lbl = tag_type
            obs_tag_plotted = True

        ax.scatter(x_cm, y_cm, marker=marker, s=90, color=color, edgecolors='black',
                   linewidths=1.2, zorder=14, label=lbl)

        normal_len = 8.0
        ax.arrow(x_cm, y_cm, normal_len * math.cos(facing_rad), normal_len * math.sin(facing_rad),
                 head_width=2.5, head_length=2.5, fc=color, ec=color, zorder=14)

        tx = x_cm + 10.0 * math.cos(facing_rad)
        ty = y_cm + 10.0 * math.sin(facing_rad)
        tag_str = f"Tag {tid}\n({family})"
        ax.text(tx, ty, tag_str, fontsize=7.5, fontweight='bold', color=color,
                ha='center', va='center', zorder=15,
                bbox=dict(boxstyle="square,pad=0.15", fc="#ffffff", ec=color, alpha=0.8))

    # 11. Start and Goal Highlights
    ax.scatter(start[0], start[1], marker='o', s=180, color='#16a34a', edgecolors='black',
               linewidths=2.0, zorder=19, label=f'Planned Start ({start[0]}, {start[1]})')
    ax.scatter(goal[0], goal[1], marker='*', s=260, color='#eab308', edgecolors='black',
               linewidths=2.0, zorder=19, label=f'Planned Goal ({goal[0]}, {goal[1]})')

    # Title & Formatting
    if has_actual:
        title_str = (
            "Homework 4: Fast A* Shortest Path Planning vs Actual Trajectory\n"
            f"Planned Length: {fast_path.total_length_cm:.1f} cm | Actual Length: {metrics.get('actual_length_cm', 0.0):.1f} cm | "
            f"Mean Cross-Track Error: {metrics.get('mean_cte_cm', 0.0):.1f} cm | Goal Error: {metrics.get('goal_error_cm', 0.0):.1f} cm"
        )
    else:
        title_str = (
            "Homework 4: Fast A* Shortest-Distance Path Planning\n"
            f"Path Length: {fast_path.total_length_cm:.1f} cm | Min Clearance: {fast_path.min_clearance_cm:.1f} cm | "
            f"Planning Time: {fast_path.planning_time_s * 1000.0:.1f} ms"
        )

    ax.set_title(title_str, fontsize=12.5, fontweight='bold', pad=15)
    ax.set_xlabel("X (cm)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Y (cm)", fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Info summary box
    if has_actual:
        summary_text = (
            "FAST A* TRAJECTORY METRICS\n"
            "-------------------------\n"
            f"Run Duration:         {metrics['duration_s']} s ({metrics['count']} records)\n"
            f"Planned Path Length:  {metrics['planned_length_cm']:.1f} cm\n"
            f"Actual Path Length:   {metrics['actual_length_cm']:.1f} cm\n"
            f"Length Difference:    {metrics['length_diff_cm']:+.1f} cm\n"
            f"Mean Cross-Track Err: {metrics['mean_cte_cm']:.2f} cm\n"
            f"Max Cross-Track Err:  {metrics['max_cte_cm']:.2f} cm\n"
            f"Start Position Error: {metrics['start_error_cm']:.2f} cm\n"
            f"Goal Arrival Error:   {metrics['goal_error_cm']:.2f} cm\n"
            f"Min Actual Clearance: {metrics['min_actual_clearance_cm']:.2f} cm\n"
            f"Safety Check:         {'SAFE' if metrics['min_actual_clearance_cm'] >= r_robot else 'UNSAFE'}"
        )
    else:
        summary_text = (
            "FAST A* PLANNING METRICS\n"
            "------------------------\n"
            f"Start: ({start[0]:.1f}, {start[1]:.1f}) cm\n"
            f"Goal:  ({goal[0]:.1f}, {goal[1]:.1f}) cm\n"
            f"Planned Length: {fast_path.total_length_cm:.2f} cm\n"
            f"Direct Line: {math.hypot(goal[0]-start[0], goal[1]-start[1]):.2f} cm (BLOCKED)\n"
            f"Min Obstacle Clearance: {fast_path.min_clearance_cm:.2f} cm\n"
            f"Robot Footprint: 16 x 16 cm\n"
            f"Conservative Radius: {r_robot:.2f} cm\n"
            f"Net Robot Body Buffer: {fast_path.min_clearance_cm - r_robot:.2f} cm\n"
            f"Grid Resolution: {fast_path.grid_resolution_cm:.1f} cm\n"
            f"Planning Time: {fast_path.planning_time_s * 1000.0:.1f} ms"
        )

    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=8.5,
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", fc="#f1f5f9", ec="#64748b", alpha=0.95),
        zorder=20
    )

    ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.18), ncol=3, fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Visualization plot saved to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Plot HW4 Fast A* Path vs Actual Trajectory.")
    parser.add_argument('--trajectory', '-t', type=str, default=None,
                        help='Path to recorded coordinates file (hw4_fast_trajectory.txt or CSV)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output image filename (default: hw4_fast_path_visualization.png)')
    parser.add_argument('--compare', action='store_true',
                        help='Include safe Voronoi path in visualization for direct comparison')
    args = parser.parse_args()

    env = Environment()
    planner = AStarPlanner(env, resolution_cm=1.0)

    start = (234.5, 63.0)
    goal = (65.0, 229.5)
    fast_path = planner.plan(start, goal)
    landmarks = load_hw4_landmarks()

    safe_path = None
    if args.compare:
        vg = VoronoiGraph(env, boundary_sample_step_cm=2.0)
        safe_planner = SafePathPlanner(env, vg)
        safe_path = safe_planner.plan(start, goal)

    actual_pts = load_fast_trajectory(args.trajectory)

    if actual_pts:
        output_file = args.output or 'hw4_fast_trajectory_comparison.png'
        print(f"Plotting Fast A* planned path vs actual trajectory ({len(actual_pts)} points)...")
        plot_fast_environment_and_path(fast_path, env, landmarks, actual_pts, safe_path, output_file)
        plot_fast_environment_and_path(fast_path, env, landmarks, actual_pts, safe_path, 'hw4_fast_path_visualization.png')
    else:
        output_file = args.output or 'hw4_fast_path_visualization.png'
        print("Plotting Fast A* shortest-distance planned path...")
        plot_fast_environment_and_path(fast_path, env, landmarks, None, safe_path, output_file)


if __name__ == '__main__':
    main()
