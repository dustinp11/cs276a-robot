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

import json
import math
import os
from pathlib import Path
from typing import Optional

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


def plot_environment_and_path(
    path: PlannedPath,
    env: Environment,
    landmarks_data: dict,
    output_filename: str = 'hw4_safe_path_visualization.png',
):
    """Generate high-resolution diagram of the HW4 workspace, Voronoi graph, and path."""
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    ax.set_aspect('equal')
    ax.set_xlim(-20, 320)
    ax.set_ylim(-20, 315)

    ws = env.workspace
    obs = env.obstacle
    robot = env.robot
    r_robot = robot.conservative_radius_cm

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
    # Inflated rectangle with rounded corners
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
    # Collision midpoint marker
    mid_x = (start[0] + goal[0]) / 2.0
    mid_y = (start[1] + goal[1]) / 2.0
    ax.scatter(mid_x, mid_y, marker='x', s=160, color='#991b1b', linewidths=3.0, zorder=8)
    ax.text(mid_x + 5, mid_y - 8, 'Direct Collision!', fontsize=9, fontweight='bold', color='#991b1b', zorder=8)

    # 6. Raw Voronoi Route (pre-smoothing)
    raw_x = [p[0] for p in path.raw_waypoints]
    raw_y = [p[1] for p in path.raw_waypoints]
    ax.plot(raw_x, raw_y, color='#0284c7', linewidth=1.2, linestyle='-.', alpha=0.5,
            zorder=8, label=f'Raw Voronoi Path ({len(path.raw_waypoints)} nodes)')

    # 7. Final Smoothed Safe Path
    smooth_x = [p[0] for p in path.smoothed_waypoints]
    smooth_y = [p[1] for p in path.smoothed_waypoints]
    ax.plot(smooth_x, smooth_y, color='#047857', linewidth=3.5, solid_capstyle='round',
            zorder=9, label=f'Final Safe Voronoi Path ({path.waypoint_count} WPs, {path.total_length_cm:.1f} cm)')

    # 8. Robot Footprint at Waypoints
    for idx, (wp_x, wp_y) in enumerate(path.smoothed_waypoints):
        heading = path.headings_rad[idx]
        corners = robot.world_corners(wp_x, wp_y, heading)
        poly = patches.Polygon(corners, closed=True, linewidth=1.5,
                               edgecolor='#0f766e', facecolor='#a7f3d0', alpha=0.7, zorder=10)
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
               linewidths=2.0, zorder=16, label=f'Start Camera Reference ({start[0]}, {start[1]})')
    ax.scatter(goal[0], goal[1], marker='*', s=260, color='#eab308', edgecolors='black',
               linewidths=2.0, zorder=16, label=f'Goal Camera Reference ({goal[0]}, {goal[1]})')

    # Formatting
    ax.set_title(
        "Homework 4: Maximum-Safety Voronoi Path Planning\n"
        f"Path Length: {path.total_length_cm:.1f} cm | Min Clearance: {path.min_clearance_cm:.1f} cm | "
        f"Robot Net Margin: {path.min_clearance_cm - r_robot:.1f} cm",
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xlabel("X (cm)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Y (cm)", fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Info summary box
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

    ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.16), ncol=3, fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Visualization plot saved to: {output_filename}")


def main():
    env = Environment()
    vg = VoronoiGraph(env, boundary_sample_step_cm=2.0)
    planner = SafePathPlanner(env, vg)

    start = (234.5, 63.0)
    goal = (65.0, 229.5)
    path = planner.plan(start, goal)

    landmarks = load_hw4_landmarks()
    plot_environment_and_path(path, env, landmarks, 'hw4_safe_path_visualization.png')


if __name__ == '__main__':
    main()
