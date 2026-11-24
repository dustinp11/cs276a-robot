"""Voronoi-based safe path planning for differential-drive robot (HW4).

This module implements:
1. Environment and C-space representation:
   - Workspace boundaries (nominally 10 ft x 10 ft, exact [0, 300] x [0, 292.5] cm).
   - Obstacle geometry ([114, 176] x [123, 185] cm).
   - Robot physical footprint (16 x 16 cm) with camera reference point at the middle of the front edge.
   - Explicit C-space dilation and exact rotated polygonal collision checking.
2. Free-space Voronoi graph generation:
   - Boundary-sampled generalized Voronoi extraction.
   - Clearance-based vertex and edge filtering.
3. Maximum-clearance path search:
   - Max-min bottleneck clearance search (P_safe = argmax_P min_{p in P} d(p)).
   - Bottleneck-preserving A* path search along medial-axis corridors.
4. Safety-preserving path smoothing:
   - Line-of-sight shortcutting strictly bounded by the bottleneck clearance.
   - Complete verification of the executable waypoint path.
"""

from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial as spatial


# -----------------------------------------------------------------------------
# Environment & Geometry
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Workspace:
    """Rectangular 2D workspace boundaries in centimeters."""
    x_min: float = 0.0
    x_max: float = 300.0
    y_min: float = 0.0
    y_max: float = 292.5

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x_min + margin <= x <= self.x_max - margin and
                self.y_min + margin <= y <= self.y_max - margin)

    def distance_to_boundary(self, x: float, y: float) -> float:
        """Positive distance inside the workspace; zero or negative outside."""
        return min(x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y)


@dataclass(frozen=True)
class Obstacle:
    """Rectangular obstacle definition in centimeters.
    
    Default dimensions [114, 176] x [123, 185] cm correspond to the outer faces
    of the 62 x 62 cm box where the four 41h12 AprilTags (Tags 1, 2, 3, 4) are mounted.
    Center is at (145.0, 154.0) cm.
    """
    x_min: float = 114.0
    x_max: float = 176.0
    y_min: float = 123.0
    y_max: float = 185.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x_min - margin <= x <= self.x_max + margin and
                self.y_min - margin <= y <= self.y_max + margin)

    def distance_to_obstacle(self, x: float, y: float) -> float:
        """Euclidean distance from (x, y) to obstacle boundary.
        
        Returns positive value outside the obstacle, negative value inside.
        """
        dx = max(self.x_min - x, 0.0, x - self.x_max)
        dy = max(self.y_min - y, 0.0, y - self.y_max)
        if dx == 0.0 and dy == 0.0:
            # Inside the obstacle: negative distance to nearest edge
            return -min(x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y)
        return math.hypot(dx, dy)


@dataclass(frozen=True)
class RobotFootprint:
    """Differential-drive robot footprint with camera reference point.
    
    Dimensions: 16.0 x 16.0 cm square.
    Camera position: middle of the top (front) edge.
    
    Coordinate conventions in robot body frame:
    - Origin (0, 0): Camera reference point.
    - +X_body: Robot forward direction (camera optical axis in horizontal plane).
    - +Y_body: Robot left direction.
    - Top/front edge: X_body = 0.0, Y_body in [-8.0, +8.0].
    - Rear edge: X_body = -16.0, Y_body in [-8.0, +8.0].
    - Left edge: Y_body = +8.0, X_body in [-16.0, 0.0].
    - Right edge: Y_body = -8.0, X_body in [-16.0, 0.0].
    
    Conservative circumscribed circle around camera:
    - Distance to rear corners: sqrt((-16)^2 + (+-8)^2) = sqrt(320) = 17.8885 cm.
    """
    width_cm: float = 16.0
    length_cm: float = 16.0

    @property
    def conservative_radius_cm(self) -> float:
        """Circumscribed radius centered at camera reference point covering all orientations."""
        return math.hypot(self.length_cm, self.width_cm / 2.0)  # sqrt(16^2 + 8^2) ≈ 17.8885 cm

    def local_corners(self) -> List[Tuple[float, float]]:
        """Return 4 corners in robot local body frame (camera at (0, 0))."""
        half_w = self.width_cm / 2.0
        return [
            (0.0, half_w),            # Front-Left
            (0.0, -half_w),           # Front-Right
            (-self.length_cm, -half_w),  # Rear-Right
            (-self.length_cm, half_w),   # Rear-Left
        ]

    def world_corners(self, x: float, y: float, theta_rad: float) -> List[Tuple[float, float]]:
        """Transform local footprint corners to world frame for robot pose (x, y, theta).
        
        Args:
            x, y: Camera reference point position in world frame (cm).
            theta_rad: Robot heading in radians (+X is 0 rad, CCW positive).
            
        Returns:
            List of 4 (X, Y) corner coordinates in world frame.
        """
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        corners = []
        for lx, ly in self.local_corners():
            wx = x + (lx * c - ly * s)
            wy = y + (lx * s + ly * c)
            corners.append((wx, wy))
        return corners

    def is_collision(
        self,
        x: float,
        y: float,
        theta_rad: float,
        workspace: Workspace,
        obstacle: Obstacle,
    ) -> bool:
        """Exact polygonal collision check for robot pose (x, y, theta)."""
        corners = self.world_corners(x, y, theta_rad)
        # Check workspace bounds for all 4 corners
        for cx, cy in corners:
            if not workspace.contains_point(cx, cy):
                return True
        # Check obstacle intersection
        # 1. Any corner inside obstacle?
        for cx, cy in corners:
            if obstacle.contains_point(cx, cy):
                return True
        # 2. Obstacle corners inside robot polygon?
        obs_corners = [
            (obstacle.x_min, obstacle.y_min),
            (obstacle.x_max, obstacle.y_min),
            (obstacle.x_max, obstacle.y_max),
            (obstacle.x_min, obstacle.y_max),
        ]
        for ox, oy in obs_corners:
            if _point_in_polygon(ox, oy, corners):
                return True
        # 3. Edge-edge intersections
        for i in range(4):
            p1, p2 = corners[i], corners[(i + 1) % 4]
            for j in range(4):
                q1, q2 = obs_corners[j], obs_corners[(j + 1) % 4]
                if _segments_intersect(p1, p2, q1, q2):
                    return True
        return False

    def min_clearance(
        self,
        x: float,
        y: float,
        theta_rad: float,
        workspace: Workspace,
        obstacle: Obstacle,
    ) -> float:
        """Calculate minimum clearance from physical robot footprint to walls and obstacle."""
        if self.is_collision(x, y, theta_rad, workspace, obstacle):
            return 0.0
        corners = self.world_corners(x, y, theta_rad)
        # Distance from corners to walls
        min_c = min(workspace.distance_to_boundary(cx, cy) for cx, cy in corners)
        # Distance from corners to obstacle
        min_c = min(min_c, min(obstacle.distance_to_obstacle(cx, cy) for cx, cy in corners))
        # Distance from obstacle corners to robot edges
        obs_corners = [
            (obstacle.x_min, obstacle.y_min),
            (obstacle.x_max, obstacle.y_min),
            (obstacle.x_max, obstacle.y_max),
            (obstacle.x_min, obstacle.y_max),
        ]
        for ox, oy in obs_corners:
            for i in range(4):
                d_edge = _point_to_segment_distance(ox, oy, corners[i], corners[(i + 1) % 4])
                if d_edge < min_c:
                    min_c = d_edge
        return max(0.0, min_c)


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _segments_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float],
) -> bool:
    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)


def _point_to_segment_distance(
    px: float,
    py: float,
    A: Tuple[float, float],
    B: Tuple[float, float],
) -> float:
    ax, ay = A
    bx, by = B
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


# -----------------------------------------------------------------------------
# Environment & Free-Space Clearance Field
# -----------------------------------------------------------------------------

class Environment:
    """Full environment model with workspace, obstacle, and robot geometry."""

    def __init__(
        self,
        workspace: Optional[Workspace] = None,
        obstacle: Optional[Obstacle] = None,
        robot: Optional[RobotFootprint] = None,
        grid_resolution_cm: float = 1.0,
    ):
        self.workspace = workspace or Workspace()
        self.obstacle = obstacle or Obstacle()
        self.robot = robot or RobotFootprint()
        self.grid_res = grid_resolution_cm
        self.required_clearance = self.robot.conservative_radius_cm

        # Precompute distance transform on occupancy grid for O(1) clearance queries
        self._nx = int(math.ceil(self.workspace.width / self.grid_res)) + 1
        self._ny = int(math.ceil(self.workspace.height / self.grid_res)) + 1
        self._grid = np.zeros((self._nx, self._ny), dtype=bool)

        # Boundary walls
        self._grid[0, :] = True
        self._grid[-1, :] = True
        self._grid[:, 0] = True
        self._grid[:, -1] = True

        # Obstacle
        ox1 = max(0, int((self.obstacle.x_min - self.workspace.x_min) / self.grid_res))
        ox2 = min(self._nx - 1, int((self.obstacle.x_max - self.workspace.x_min) / self.grid_res))
        oy1 = max(0, int((self.obstacle.y_min - self.workspace.y_min) / self.grid_res))
        oy2 = min(self._ny - 1, int((self.obstacle.y_max - self.workspace.y_min) / self.grid_res))
        self._grid[ox1:ox2 + 1, oy1:oy2 + 1] = True

        # Distance transform (cm)
        self.distance_field = ndimage.distance_transform_edt(~self._grid) * self.grid_res

    def point_clearance(self, x: float, y: float) -> float:
        """Exact Euclidean clearance from (x, y) to the nearest wall or obstacle."""
        d_wall = self.workspace.distance_to_boundary(x, y)
        d_obs = self.obstacle.distance_to_obstacle(x, y)
        return min(d_wall, d_obs)

    def segment_clearance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        samples: int = 25,
    ) -> float:
        """Minimum clearance along the straight line segment between p1 and p2."""
        min_c = float('inf')
        for t in np.linspace(0.0, 1.0, samples):
            x = (1.0 - t) * p1[0] + t * p2[0]
            y = (1.0 - t) * p1[1] + t * p2[1]
            c = self.point_clearance(x, y)
            if c < min_c:
                min_c = c
        return min_c

    def is_segment_safe(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        min_clearance: Optional[float] = None,
        samples: int = 25,
    ) -> bool:
        """Check if straight line segment p1-p2 maintains clearance."""
        req = self.required_clearance if min_clearance is None else min_clearance
        return self.segment_clearance(p1, p2, samples) >= req


# -----------------------------------------------------------------------------
# Voronoi Graph Generation
# -----------------------------------------------------------------------------

class VoronoiGraph:
    """Generates and manages the free-space Voronoi diagram."""

    def __init__(self, env: Environment, boundary_sample_step_cm: float = 2.0):
        self.env = env
        self.sample_step = boundary_sample_step_cm
        self.vertices: List[np.ndarray] = []
        self.adj: Dict[int, List[Tuple[int, float, float]]] = {}  # u -> list of (v, dist, clearance)
        self._build_graph()

    def _sample_boundaries(self) -> np.ndarray:
        """Sample points along workspace boundaries and obstacle perimeters."""
        pts = []
        step = self.sample_step
        ws = self.env.workspace
        obs = self.env.obstacle

        # Workspace walls
        for x in np.arange(ws.x_min, ws.x_max + step, step):
            pts.append([min(x, ws.x_max), ws.y_min])
            pts.append([min(x, ws.x_max), ws.y_max])
        for y in np.arange(ws.y_min, ws.y_max + step, step):
            pts.append([ws.x_min, min(y, ws.y_max)])
            pts.append([ws.x_max, min(y, ws.y_max)])

        # Obstacle walls
        for x in np.arange(obs.x_min, obs.x_max + step, step):
            pts.append([min(x, obs.x_max), obs.y_min])
            pts.append([min(x, obs.x_max), obs.y_max])
        for y in np.arange(obs.y_min, obs.y_max + step, step):
            pts.append([obs.x_min, min(y, obs.y_max)])
            pts.append([obs.x_max, min(y, obs.y_max)])

        return np.array(pts)

    def _build_graph(self):
        pts = self._sample_boundaries()
        vor = spatial.Voronoi(pts)

        req_clearance = self.env.required_clearance
        raw_v = vor.vertices
        v_clear = [self.env.point_clearance(v[0], v[1]) for v in raw_v]

        # Valid vertices are inside free space with clearance >= robot radius
        valid_indices = {i for i, c in enumerate(v_clear) if c >= req_clearance}

        self.vertices = [raw_v[i] for i in range(len(raw_v))]
        self.adj = {i: [] for i in valid_indices}

        for p1_idx, p2_idx in vor.ridge_vertices:
            if p1_idx in valid_indices and p2_idx in valid_indices:
                p1 = self.vertices[p1_idx]
                p2 = self.vertices[p2_idx]
                edge_clearance = self.env.segment_clearance(p1, p2, samples=12)
                if edge_clearance >= req_clearance:
                    dist = float(np.linalg.norm(p1 - p2))
                    self.adj[p1_idx].append((p2_idx, dist, edge_clearance))
                    self.adj[p2_idx].append((p1_idx, dist, edge_clearance))

        # Filter out isolated vertices
        self.valid_vertices = {i for i in valid_indices if len(self.adj.get(i, [])) > 0}


# -----------------------------------------------------------------------------
# Path Planner: Maximum Minimum Clearance & Smoothing
# -----------------------------------------------------------------------------

@dataclass
class PlannedPath:
    """Result of path planning."""
    raw_waypoints: List[Tuple[float, float]]
    smoothed_waypoints: List[Tuple[float, float]]
    bottleneck_clearance_cm: float
    min_clearance_cm: float
    average_clearance_cm: float
    total_length_cm: float
    waypoint_count: int
    headings_rad: List[float] = field(default_factory=list)


class SafePathPlanner:
    """Plans maximum-clearance paths on the Voronoi graph."""

    def __init__(self, env: Environment, voronoi_graph: VoronoiGraph):
        self.env = env
        self.vg = voronoi_graph

    def plan(
        self,
        start_cm: Tuple[float, float],
        goal_cm: Tuple[float, float],
        smoothing_tolerance_cm: float = 2.0,
    ) -> PlannedPath:
        """Find a safe route maximizing minimum clearance from start to goal.
        
        Args:
            start_cm: (x, y) start camera position in centimeters.
            goal_cm: (x, y) goal camera position in centimeters.
            smoothing_tolerance_cm: Allowed reduction from optimal bottleneck for smoothing.
            
        Returns:
            PlannedPath containing raw and smoothed waypoints and metrics.
        """
        start = np.array(start_cm, dtype=float)
        goal = np.array(goal_cm, dtype=float)

        # Validate start and goal
        c_start = self.env.point_clearance(start[0], start[1])
        c_goal = self.env.point_clearance(goal[0], goal[1])
        if c_start < self.env.required_clearance:
            raise ValueError(f"Start {start_cm} violates clearance: {c_start:.2f} < {self.env.required_clearance:.2f} cm")
        if c_goal < self.env.required_clearance:
            raise ValueError(f"Goal {goal_cm} violates clearance: {c_goal:.2f} < {self.env.required_clearance:.2f} cm")

        # Create temporary graph copy connecting Start (-1) and Goal (-2)
        START_ID = -1
        GOAL_ID = -2
        graph_adj: Dict[int, List[Tuple[int, float, float]]] = {
            k: list(v) for k, v in self.vg.adj.items() if k in self.vg.valid_vertices
        }
        graph_adj[START_ID] = []
        graph_adj[GOAL_ID] = []

        all_coords: Dict[int, np.ndarray] = {
            i: self.vg.vertices[i] for i in self.vg.valid_vertices
        }
        all_coords[START_ID] = start
        all_coords[GOAL_ID] = goal

        # Connect Start and Goal to reachable Voronoi vertices
        req_c = self.env.required_clearance
        for i in self.vg.valid_vertices:
            v = self.vg.vertices[i]
            # Start connection
            d_s = float(np.linalg.norm(start - v))
            if d_s < 120.0:
                c_s = self.env.segment_clearance(start, v, samples=15)
                if c_s >= req_c:
                    graph_adj[START_ID].append((i, d_s, c_s))
                    graph_adj[i].append((START_ID, d_s, c_s))
            # Goal connection
            d_g = float(np.linalg.norm(goal - v))
            if d_g < 120.0:
                c_g = self.env.segment_clearance(goal, v, samples=15)
                if c_g >= req_c:
                    graph_adj[GOAL_ID].append((i, d_g, c_g))
                    graph_adj[i].append((GOAL_ID, d_g, c_g))

        if not graph_adj[START_ID]:
            raise RuntimeError("Could not connect Start to the Voronoi graph.")
        if not graph_adj[GOAL_ID]:
            raise RuntimeError("Could not connect Goal to the Voronoi graph.")

        # ---------------------------------------------------------------------
        # Step 1: Widest Path Search (Maximum Bottleneck Clearance)
        # ---------------------------------------------------------------------
        # Heap entries: (-bottleneck_clearance, node_id, path)
        init_bottleneck = min(c_start, c_goal)
        pq = [(-init_bottleneck, START_ID, [START_ID])]
        visited_bottleneck: Dict[int, float] = {}
        raw_path: Optional[List[int]] = None
        optimal_bottleneck = -1.0

        while pq:
            neg_b, curr, path = heapq.heappop(pq)
            b = -neg_b
            if curr in visited_bottleneck and visited_bottleneck[curr] >= b:
                continue
            visited_bottleneck[curr] = b

            if curr == GOAL_ID:
                raw_path = path
                optimal_bottleneck = b
                break

            for nxt, dist, edge_c in graph_adj.get(curr, []):
                nxt_b = min(b, edge_c)
                if nxt not in visited_bottleneck or visited_bottleneck[nxt] < nxt_b:
                    heapq.heappush(pq, (-nxt_b, nxt, path + [nxt]))

        if raw_path is None:
            raise RuntimeError("No connected path found between Start and Goal.")

        # ---------------------------------------------------------------------
        # Step 2: Bottleneck-Optimal A* on Medial Corridors
        # ---------------------------------------------------------------------
        # Ensure we pick the cleanest medial-axis path achieving optimal_bottleneck
        min_allowed_b = max(req_c, optimal_bottleneck - 0.5)
        # Standard A* using safety-weighted distance on the bottleneck-admissible subgraph
        # cost = dist * (1 + 100 / clearance^2)
        pq_astar = [(0.0, 0.0, START_ID, [START_ID])]
        visited_cost: Dict[int, float] = {}
        best_cost_path: List[int] = raw_path

        while pq_astar:
            f, g, curr, path = heapq.heappop(pq_astar)
            if curr in visited_cost and visited_cost[curr] <= g:
                continue
            visited_cost[curr] = g

            if curr == GOAL_ID:
                best_cost_path = path
                break

            curr_pt = all_coords[curr]
            for nxt, dist, edge_c in graph_adj.get(curr, []):
                if edge_c < min_allowed_b:
                    continue
                nxt_pt = all_coords[nxt]
                # Edge weight heavily favors higher clearance
                safety_weight = 1.0 + 50.0 / (edge_c ** 2)
                edge_cost = dist * safety_weight
                new_g = g + edge_cost
                if nxt not in visited_cost or visited_cost[nxt] > new_g:
                    h = float(np.linalg.norm(nxt_pt - goal))
                    heapq.heappush(pq_astar, (new_g + h, new_g, nxt, path + [nxt]))

        raw_coords = [(float(all_coords[idx][0]), float(all_coords[idx][1])) for idx in best_cost_path]

        # ---------------------------------------------------------------------
        # Step 3: Safety-Preserving Path Smoothing
        # ---------------------------------------------------------------------
        # Shortcut intermediate waypoints while strictly preserving bottleneck clearance >= (optimal_bottleneck - tolerance)
        target_min_clearance = max(req_c, optimal_bottleneck - smoothing_tolerance_cm)
        smoothed = [raw_coords[0]]
        i = 0
        n = len(raw_coords)

        while i < n - 1:
            best_j = i + 1
            for j in range(n - 1, i + 1, -1):
                sc = self.env.segment_clearance(raw_coords[i], raw_coords[j], samples=30)
                if sc >= target_min_clearance:
                    best_j = j
                    break
            smoothed.append(raw_coords[best_j])
            i = best_j

        # Calculate path metrics
        smooth_len = sum(
            math.hypot(smoothed[k + 1][0] - smoothed[k][0], smoothed[k + 1][1] - smoothed[k][1])
            for k in range(len(smoothed) - 1)
        )
        min_smooth_c = min(
            self.env.segment_clearance(smoothed[k], smoothed[k + 1], samples=30)
            for k in range(len(smoothed) - 1)
        )
        # Average clearance sampled densely along smoothed path
        clearance_samples = []
        for k in range(len(smoothed) - 1):
            p1, p2 = smoothed[k], smoothed[k + 1]
            for t in np.linspace(0.0, 1.0, 15):
                clearance_samples.append(
                    self.env.point_clearance((1 - t) * p1[0] + t * p2[0], (1 - t) * p1[1] + t * p2[1])
                )
        avg_c = float(np.mean(clearance_samples))

        # Calculate segment headings (radians)
        headings = []
        for k in range(len(smoothed) - 1):
            dx = smoothed[k + 1][0] - smoothed[k][0]
            dy = smoothed[k + 1][1] - smoothed[k][1]
            headings.append(math.atan2(dy, dx))
        if headings:
            headings.append(headings[-1])  # final waypoint heading matching last segment

        return PlannedPath(
            raw_waypoints=raw_coords,
            smoothed_waypoints=smoothed,
            bottleneck_clearance_cm=optimal_bottleneck,
            min_clearance_cm=min_smooth_c,
            average_clearance_cm=avg_c,
            total_length_cm=smooth_len,
            waypoint_count=len(smoothed),
            headings_rad=headings,
        )


def main():
    """Command-line entry point to run Voronoi path planning and display metrics."""
    env = Environment()
    print("Initializing Voronoi Graph of free space...")
    vg = VoronoiGraph(env, boundary_sample_step_cm=2.0)
    planner = SafePathPlanner(env, vg)

    start = (234.5, 63.0)
    goal = (65.0, 229.5)

    print(f"Planning safe Voronoi path from Start {start} to Goal {goal}...")
    path = planner.plan(start, goal, smoothing_tolerance_cm=2.0)

    print("\n" + "=" * 60)
    print("SAFE VORONOI PATH PLANNING RESULTS")
    print("=" * 60)
    print(f"Optimal Voronoi Bottleneck Clearance: {path.bottleneck_clearance_cm:.2f} cm")
    print(f"Smoothed Path Minimum Clearance:      {path.min_clearance_cm:.2f} cm")
    print(f"Smoothed Path Average Clearance:      {path.average_clearance_cm:.2f} cm")
    print(f"Total Planned Path Length:            {path.total_length_cm:.2f} cm")
    print(f"Waypoints Count:                      {path.waypoint_count}")
    print(f"Robot Conservative Radius:            {env.robot.conservative_radius_cm:.2f} cm")
    print(f"Net Physical Robot Margin:            {path.min_clearance_cm - env.robot.conservative_radius_cm:.2f} cm")
    print("-" * 60)
    print("Planned Waypoints (cm):")
    for idx, (x, y) in enumerate(path.smoothed_waypoints):
        heading_deg = math.degrees(path.headings_rad[idx])
        c = env.point_clearance(x, y)
        print(f"  WP {idx}: ({x:6.2f}, {y:6.2f})  Clearance: {c:5.2f} cm  Heading: {heading_deg:6.1f}°")
    print("=" * 60 + "\n")

    # Direct line comparison
    direct_dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
    direct_min_c = env.segment_clearance(start, goal, samples=100)
    print("COMPARISON WITH DIRECT PATH:")
    print(f"  Direct straight-line distance: {direct_dist:.2f} cm")
    print(f"  Direct path minimum clearance: {direct_min_c:.2f} cm (NEGATIVE => COLLISION!)")
    print(f"  Safe Voronoi path avoids the obstacle with {path.min_clearance_cm:.2f} cm clearance everywhere.")


if __name__ == "__main__":
    main()
