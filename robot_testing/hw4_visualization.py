"""Comprehensive environment, Voronoi graph, and safe path visualization (HW4).

Visualizes:
1. Workspace boundaries ([0, 300] x [0, 292.5] cm).
2. Actual obstacle ([114, 176] x [123, 185] cm).
3. Inflated obstacle / C-space boundary (dilated by robot conservative radius R=17.89 cm).
4. Workspace boundary clearance zone.
5. All 8 Wall AprilTags (36h11, 17.2 cm) with IDs, coordinates, and facing vectors.
6. All 4 Obstacle AprilTags (41h12, 18.7 cm) with IDs, coordinates, and facing vectors.
7. Free-space Voronoi graph (medial axis).
8. Start (234.5, 63) and Goal (65, 229.5) coordinates.
9. Final safe planned path with waypoint sequence and clearances.
10. Direct start-to-goal line demonstrating collision with obstacle.
11. Physical robot footprint (16 x 16 cm rectangle) and camera reference point at waypoints.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple

# Ensure matplotlib uses a writable temp cache directory
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from robot_testing.voronoi_planner import (
    Environment,
    Obstacle,
    PlannedPath,
    RobotFootprint,
    SafePathPlanner,
    VoronoiGraph,
    Workspace,
    _point_to_segment_distance,
)


def load_hw4_landmarks(config_path: Optional[str] = None):
    """Load surveyed landmarks from config/hw4_landmarks.json."""
    if config_path is None:
        possible = [
            Path(__file__).resolve().parent.parent / 'config/hw4_landmarks.json',
            Path('config/hw4_landmarks.json'),
        ]
        for p in possible:
            if p.exists():
                config_path = str(p)
                break
    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(f"hw4_landmarks.json not found at {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_actual_trajectory(file_path: Optional[str] = None) -> List[Tuple[float, float]]:
    """
    Load recorded robot coordinates from trajectory file.
    Supports:
    1. Parenthesized tuples: '(x, y)' per line (coordinates.txt / hw4_trajectory.txt)
    2. Comma-separated CSV lines (hw4_actual_trajectory.csv)
    Automatically normalizes meters to centimeters if needed.
    """
    candidates = []
    if file_path:
        candidates.append(Path(file_path))
    pkg_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        Path('hw4_trajectory.txt'),
        Path('coordinates.txt'),
        Path('coordinate.txt'),
        pkg_root / 'hw4_trajectory.txt',
        pkg_root / 'coordinates.txt',
        Path('hw4_actual_trajectory.csv'),
        pkg_root / 'hw4_actual_trajectory.csv',
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
                # If CSV has timestamp first (large unix / monotonic time)
                if line.count(',') >= 2 and len(numbers) >= 3 and numbers[0] > 1000:
                    x, y = numbers[1], numbers[2]
                else:
                    x, y = numbers[0], numbers[1]

                # Convert meters to centimeters if coordinates are small (< 10)
                if abs(x) < 10.0 and abs(y) < 10.0:
                    x *= 100.0
                    y *= 100.0

                # Validate reasonable coordinates within workspace margin
                if -50.0 <= x <= 350.0 and -50.0 <= y <= 350.0:
                    pts.append((x, y))

    print(f"Loaded {len(pts)} recorded trajectory coordinates from: {found_path}")
    return pts


def compute_trajectory_metrics(
    actual_pts: List[Tuple[float, float]],
    planned_path: PlannedPath,
    env: Environment,
) -> dict:
    """Compute comparison metrics between actual trajectory and planned path."""
    if not actual_pts:
        return {}

    wps = planned_path.smoothed_waypoints
    start_pt = wps[0]
    goal_pt = wps[-1]

    # 1. Total actual trajectory length
    actual_length_cm = 0.0
    for i in range(len(actual_pts) - 1):
        actual_length_cm += math.hypot(
            actual_pts[i + 1][0] - actual_pts[i][0],
            actual_pts[i + 1][1] - actual_pts[i][1],
        )

    # 2. Start and Goal errors
    start_err_cm = math.hypot(actual_pts[0][0] - start_pt[0], actual_pts[0][1] - start_pt[1])
    goal_err_cm = math.hypot(actual_pts[-1][0] - goal_pt[0], actual_pts[-1][1] - goal_pt[1])

    # 3. Cross-track errors to closest planned segment
    cross_track_errors = []
    min_actual_clearance_cm = float('inf')

    for px, py in actual_pts:
        # Distance to closest segment
        min_seg_dist = float('inf')
        for j in range(len(wps) - 1):
            d = _point_to_segment_distance(px, py, wps[j], wps[j + 1])
            if d < min_seg_dist:
                min_seg_dist = d
        cross_track_errors.append(min_seg_dist)

        # Clearance to walls/obstacle at this point
        c = env.point_clearance(px, py)
        if c < min_actual_clearance_cm:
            min_actual_clearance_cm = c

    mean_cte_cm = float(np.mean(cross_track_errors)) if cross_track_errors else 0.0
    max_cte_cm = float(np.max(cross_track_errors)) if cross_track_errors else 0.0

    return {
        'count': len(actual_pts),
        'duration_s': len(actual_pts),  # 1 record per second
        'actual_length_cm': actual_length_cm,
        'planned_length_cm': planned_path.total_length_cm,
        'length_diff_cm': actual_length_cm - planned_path.total_length_cm,
        'start_error_cm': start_err_cm,
        'goal_error_cm': goal_err_cm,
        'mean_cte_cm': mean_cte_cm,
        'max_cte_cm': max_cte_cm,
        'min_actual_clearance_cm': min_actual_clearance_cm,
        'bottleneck_clearance_cm': planned_path.bottleneck_clearance_cm,
    }


def plot_environment_and_path(
    path: PlannedPath,
    env: Environment,
    landmarks_data: dict,
    actual_trajectory: Optional[List[Tuple[float, float]]] = None,
    output_filename: str = 'hw4_safe_path_visualization.png',
):
    """Generate high-resolution diagram of HW4 workspace, Voronoi graph, and actual vs planned path."""
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    ax.set_aspect('equal')
    ax.set_xlim(-20, 320)
    ax.set_ylim(-20, 315)

    ws = env.workspace
    obs = env.obstacle
    robot = env.robot
    r_robot = robot.conservative_radius_cm
    has_actual = bool(actual_trajectory and len(actual_trajectory) > 1)
    metrics = compute_trajectory_metrics(actual_trajectory, path, env) if has_actual else {}

    # 1. Workspace background and boundary
    ax.add_patch(patches.Rectangle(
        (ws.x_min, ws.y_min), ws.width, ws.height,
        linewidth=2.5, edgecolor='#1e293b', facecolor='#f8fafc', zorder=1,
        label=f'Workspace (10 ft nominal: {ws.width:.1f} x {ws.height:.1f} cm)'
    ))

    # Boundary clearance zone
    ax.add_patch(patches.Rectangle(
        (ws.x_min + r_robot, ws.y_min + r_robot),
        ws.width - 2 * r_robot, ws.height - 2 * r_robot,
        linewidth=1.2, edgecolor='#94a3b8', linestyle='--', facecolor='none', zorder=2,
        label=f'Boundary Clearance Margin (R = {r_robot:.1f} cm)'
    ))

    # 2. Free-space Voronoi Graph
    vg = VoronoiGraph(env, boundary_sample_step_cm=2.0)
    vor_plotted = False
    for u, edges in vg.adj.items():
        if u not in vg.valid_vertices:
            continue
        p1 = vg.vertices[u]
        for v, dist, edge_c in edges:
            if v > u and v in vg.valid_vertices:
                p2 = vg.vertices[v]
                lbl = 'Free-Space Voronoi Graph (Medial Axis)' if not vor_plotted else None
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#cbd5e1', linewidth=1.0,
                        linestyle='-', zorder=3, label=lbl)
                vor_plotted = True

    # 3. Inflated Obstacle (Configuration Space)
    inf_x = obs.x_min - r_robot
    inf_y = obs.y_min - r_robot
    inf_w = obs.width + 2 * r_robot
    inf_h = obs.height + 2 * r_robot
    inflated_box = patches.FancyBboxPatch(
        (inf_x, inf_y), inf_w, inf_h,
        boxstyle=f"round,pad=0,rounding_size={r_robot:.1f}",
        linewidth=1.8, edgecolor='#dc2626', linestyle='--', facecolor='#fee2e2', alpha=0.6,
        zorder=4, label=f'C-Space Inflated Obstacle (dilated by {r_robot:.1f} cm)'
    )
    ax.add_patch(inflated_box)

    # 4. Actual Physical Obstacle
    ax.add_patch(patches.Rectangle(
        (obs.x_min, obs.y_min), obs.width, obs.height,
        linewidth=2.0, edgecolor='#991b1b', facecolor='#ef4444', alpha=0.9,
        zorder=5, label=f'Actual Obstacle ({obs.width:.0f} x {obs.height:.0f} cm box)'
    ))
    ax.text(obs.center[0], obs.center[1], 'OBSTACLE\n62 x 62 cm',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=6)

    # 5. Direct start-to-goal path (showing collision)
    start = path.smoothed_waypoints[0]
    goal = path.smoothed_waypoints[-1]
    ax.plot([start[0], goal[0]], [start[1], goal[1]],
            color='#b91c1c', linestyle=':', linewidth=2.0, zorder=7,
            label='Direct Path (Intersects Obstacle - Unsafe!)')
    mid_x = (start[0] + goal[0]) / 2.0
    mid_y = (start[1] + goal[1]) / 2.0
    ax.scatter(mid_x, mid_y, marker='x', s=160, color='#991b1b', linewidths=3.0, zorder=8)
    ax.text(mid_x + 5, mid_y - 8, 'Direct Collision!', fontsize=9, fontweight='bold', color='#991b1b', zorder=8)

    # 6. Final Smoothed Planned Safe Path
    smooth_x = [p[0] for p in path.smoothed_waypoints]
    smooth_y = [p[1] for p in path.smoothed_waypoints]
    planned_style = '--' if has_actual else '-'
    ax.plot(smooth_x, smooth_y, color='#047857', linewidth=3.5, linestyle=planned_style,
            solid_capstyle='round', zorder=9,
            label=f'Planned Safe Path ({path.waypoint_count} WPs, {path.total_length_cm:.1f} cm)')

    # 7. Robot Footprint at Planned Waypoints
    for idx, (wp_x, wp_y) in enumerate(path.smoothed_waypoints):
        heading = path.headings_rad[idx]
        corners = robot.world_corners(wp_x, wp_y, heading)
        poly = patches.Polygon(corners, closed=True, linewidth=1.5,
                               edgecolor='#0f766e', facecolor='#a7f3d0', alpha=0.5, zorder=10)
        ax.add_patch(poly)

        # Camera reference point (middle of front edge)
        ax.scatter(wp_x, wp_y, color='#065f46', s=45, zorder=12)

        # Forward orientation vector
        arrow_len = 10.0
        ax.arrow(wp_x, wp_y, arrow_len * math.cos(heading), arrow_len * math.sin(heading),
                 head_width=3.5, head_length=3.5, fc='#065f46', ec='#065f46', zorder=12)

        # Waypoint label
        lbl_offset_x = 8.0 if wp_x < 150 else -8.0
        lbl_offset_y = 8.0 if wp_y < 150 else -8.0
        c_val = env.point_clearance(wp_x, wp_y)
        ax.annotate(
            f"WP {idx}\n({wp_x:.1f}, {wp_y:.1f})\nclr={c_val:.1f}cm",
            (wp_x, wp_y),
            xytext=(wp_x + lbl_offset_x, wp_y + lbl_offset_y),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#047857", alpha=0.9),
            fontsize=8, fontweight='semibold', zorder=15,
            ha='left' if lbl_offset_x > 0 else 'right'
        )

    # 8. Actual Robot Trajectory (if available)
    if has_actual:
        act_x = [p[0] for p in actual_trajectory]
        act_y = [p[1] for p in actual_trajectory]

        # Draw actual trajectory line and 1s recorded points
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
            zorder=17,
            label=f'Actual Robot Trajectory ({metrics["duration_s"]}s, {metrics["actual_length_cm"]:.1f} cm)'
        )

        # Draw directional travel arrows every few steps
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
                    zorder=18
                )

        # Highlight actual start and finish points
        ax.scatter(act_x[0], act_y[0], marker='o', s=150, color='#3b82f6',
                   edgecolors='#1e3a8a', linewidths=2.5, zorder=19,
                   label=f'Actual Start ({act_x[0]:.1f}, {act_y[0]:.1f}) cm')
        ax.scatter(act_x[-1], act_y[-1], marker='D', s=160, color='#9333ea',
                   edgecolors='black', linewidths=2.0, zorder=19,
                   label=f'Actual Finish ({act_x[-1]:.1f}, {act_y[-1]:.1f}) cm')

    # 9. AprilTags (Wall and Obstacle)
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

        # Facing normal arrow
        normal_len = 8.0
        ax.arrow(x_cm, y_cm, normal_len * math.cos(facing_rad), normal_len * math.sin(facing_rad),
                 head_width=2.5, head_length=2.5, fc=color, ec=color, zorder=14)

        # Tag text label
        tx = x_cm + 10.0 * math.cos(facing_rad)
        ty = y_cm + 10.0 * math.sin(facing_rad)
        tag_str = f"Tag {tid}\n({family})"
        ax.text(tx, ty, tag_str, fontsize=7.5, fontweight='bold', color=color,
                ha='center', va='center', zorder=15,
                bbox=dict(boxstyle="square,pad=0.15", fc="#ffffff", ec=color, alpha=0.8))

    # 10. Start and Goal Highlights
    ax.scatter(start[0], start[1], marker='o', s=180, color='#16a34a', edgecolors='black',
               linewidths=2.0, zorder=16, label=f'Planned Start ({start[0]}, {start[1]})')
    ax.scatter(goal[0], goal[1], marker='*', s=260, color='#eab308', edgecolors='black',
               linewidths=2.0, zorder=16, label=f'Planned Goal ({goal[0]}, {goal[1]})')

    # Formatting
    title_text = (
        "Homework 4: Safe Voronoi Path Planning vs Actual Robot Trajectory\n"
        if has_actual else
        "Homework 4: Maximum-Safety Voronoi Path Planning\n"
    )
    subtitle = (
        f"Planned Length: {path.total_length_cm:.1f} cm | Actual Length: {metrics.get('actual_length_cm', 0.0):.1f} cm | "
        f"Mean Cross-Track Error: {metrics.get('mean_cte_cm', 0.0):.1f} cm | Goal Error: {metrics.get('goal_error_cm', 0.0):.1f} cm"
        if has_actual else
        f"Path Length: {path.total_length_cm:.1f} cm | Min Clearance: {path.min_clearance_cm:.1f} cm | "
        f"Robot Net Margin: {path.min_clearance_cm - r_robot:.1f} cm"
    )
    ax.set_title(title_text + subtitle, fontsize=12.5, fontweight='bold', pad=15)
    ax.set_xlabel("X (cm)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Y (cm)", fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Info summary box
    if has_actual:
        summary_text = (
            "TRAJECTORY COMPARISON METRICS\n"
            "----------------------------\n"
            f"Run Duration:         {metrics['duration_s']} s ({metrics['count']} records)\n"
            f"Planned Path Length:  {metrics['planned_length_cm']:.1f} cm\n"
            f"Actual Path Length:   {metrics['actual_length_cm']:.1f} cm\n"
            f"Length Difference:    {metrics['length_diff_cm']:+.1f} cm\n"
            f"Mean Cross-Track Err: {metrics['mean_cte_cm']:.2f} cm\n"
            f"Max Cross-Track Err:  {metrics['max_cte_cm']:.2f} cm\n"
            f"Start Position Error: {metrics['start_error_cm']:.2f} cm\n"
            f"Goal Arrival Error:   {metrics['goal_error_cm']:.2f} cm\n"
            f"Min Actual Clearance: {metrics['min_actual_clearance_cm']:.2f} cm\n"
            f"Planned Bottleneck:   {metrics['bottleneck_clearance_cm']:.2f} cm\n"
            f"Collision-Free:       {'YES (SAFE)' if metrics['min_actual_clearance_cm'] >= r_robot else 'NO'}"
        )
    else:
        summary_text = (
            "PATH & SAFETY METRICS\n"
            "--------------------\n"
            f"Start: ({start[0]:.1f}, {start[1]:.1f}) cm\n"
            f"Goal:  ({goal[0]:.1f}, {goal[1]:.1f}) cm\n"
            f"Max-Min Bottleneck: {path.bottleneck_clearance_cm:.2f} cm\n"
            f"Smoothed Min Clearance: {path.min_clearance_cm:.2f} cm\n"
            f"Average Clearance: {path.average_clearance_cm:.2f} cm\n"
            f"Planned Length: {path.total_length_cm:.2f} cm\n"
            f"Direct Line: {math.hypot(goal[0]-start[0], goal[1]-start[1]):.2f} cm (BLOCKED)\n"
            f"Robot Size: 16 x 16 cm\n"
            f"Camera Offset: middle top edge\n"
            f"Bounding Radius: {r_robot:.2f} cm\n"
            f"Physical Robot Margin: {path.min_clearance_cm - r_robot:.2f} cm"
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
    parser = argparse.ArgumentParser(description="Plot HW4 Safe Voronoi Path vs Actual Trajectory.")
    parser.add_argument('--trajectory', '-t', type=str, default=None,
                        help='Path to recorded coordinates file (coordinates.txt, hw4_trajectory.txt, or CSV)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output image filename (default: hw4_trajectory_comparison.png if trajectory found)')
    args = parser.parse_args()

    env = Environment()
    vg = VoronoiGraph(env, boundary_sample_step_cm=2.0)
    planner = SafePathPlanner(env, vg)

    start = (234.5, 63.0)
    goal = (65.0, 229.5)
    path = planner.plan(start, goal)
    landmarks = load_hw4_landmarks()

    actual_pts = load_actual_trajectory(args.trajectory)

    if actual_pts:
        output_file = args.output or 'hw4_trajectory_comparison.png'
        print(f"Plotting planned path vs actual trajectory ({len(actual_pts)} points)...")
        plot_environment_and_path(path, env, landmarks, actual_pts, output_file)
        # Also update hw4_safe_path_visualization.png
        plot_environment_and_path(path, env, landmarks, actual_pts, 'hw4_safe_path_visualization.png')
    else:
        output_file = args.output or 'hw4_safe_path_visualization.png'
        print("No recorded trajectory file found yet. Plotting planned safe Voronoi path...")
        plot_environment_and_path(path, env, landmarks, None, output_file)
        print("Tip: Run navigation first to generate coordinates.txt / hw4_trajectory.txt, then rerun this script.")
