"""Fast A* Path Planning Module for HW4.

Implements the shortest-distance collision-free path planner for Homework 4:
1. Occupancy grid representation with configuration-space obstacle inflation
   using the robot's physical dimensions (16 x 16 cm) and camera offset
   (R_conservative = sqrt(16^2 + 8^2) ≈ 17.89 cm).
2. 8-connected A* search with Euclidean heuristic and strict corner-cutting prevention.
3. Path length minimization cost function (no clearance penalty).
4. Collision-preserving shortcut smoothing to eliminate grid zig-zags.
"""

from dataclasses import dataclass
import heapq
import math
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from robot_testing.voronoi_planner import (
    Environment,
    Obstacle,
    RobotFootprint,
    Workspace,
    _point_to_segment_distance,
)


@dataclass
class FastPlannedPath:
    raw_waypoints: List[Tuple[float, float]]
    smoothed_waypoints: List[Tuple[float, float]]
    headings_rad: List[float]
    total_length_cm: float
    min_clearance_cm: float
    waypoint_count: int
    planning_time_s: float
    grid_resolution_cm: float

    @property
    def bottleneck_clearance_cm(self) -> float:
        """Compatibility property for trajectory metric evaluation."""
        return self.min_clearance_cm


class OccupancyGrid:
    """2D Configuration-Space Occupancy Grid representation of the workspace."""

    def __init__(
        self,
        env: Environment,
        resolution_cm: float = 1.0,
    ):
        self.env = env
        self.res = resolution_cm
        self.ws = env.workspace
        self.obs = env.obstacle
        self.robot = env.robot
        self.r_robot = self.robot.conservative_radius_cm  # ~17.8885 cm

        self.nx = int(math.ceil(self.ws.width / self.res))
        self.ny = int(math.ceil(self.ws.height / self.res))

        # False = Free, True = Occupied/Impassable
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)
        self._build_cspace_grid()

    def _build_cspace_grid(self):
        """Mark cells whose distance to boundaries or obstacle is less than R_robot."""
        # Cell center coordinates
        xs = (np.arange(self.nx) + 0.5) * self.res + self.ws.x_min
        ys = (np.arange(self.ny) + 0.5) * self.res + self.ws.y_min

        # 1. Workspace boundary inflation (margin = r_robot)
        x_min_valid = self.ws.x_min + self.r_robot
        x_max_valid = self.ws.x_max - self.r_robot
        y_min_valid = self.ws.y_min + self.r_robot
        y_max_valid = self.ws.y_max - self.r_robot

        for ix, x in enumerate(xs):
            if x < x_min_valid or x > x_max_valid:
                self.grid[ix, :] = True

        for iy, y in enumerate(ys):
            if y < y_min_valid or y > y_max_valid:
                self.grid[:, iy] = True

        # 2. Obstacle C-space inflation
        # Bounding box of inflated obstacle
        obs_inf_x_min = self.obs.x_min - self.r_robot
        obs_inf_x_max = self.obs.x_max + self.r_robot
        obs_inf_y_min = self.obs.y_min - self.r_robot
        obs_inf_y_max = self.obs.y_max + self.r_robot

        ix_start = max(0, int(math.floor((obs_inf_x_min - self.ws.x_min) / self.res)))
        ix_end = min(self.nx, int(math.ceil((obs_inf_x_max - self.ws.x_min) / self.res)))
        iy_start = max(0, int(math.floor((obs_inf_y_min - self.ws.y_min) / self.res)))
        iy_end = min(self.ny, int(math.ceil((obs_inf_y_max - self.ws.y_min) / self.res)))

        for ix in range(ix_start, ix_end):
            x = xs[ix]
            for iy in range(iy_start, iy_end):
                y = ys[iy]
                if self.obs.distance_to_obstacle(x, y) < self.r_robot:
                    self.grid[ix, iy] = True

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(math.floor((x - self.ws.x_min) / self.res))
        gy = int(math.floor((y - self.ws.y_min) / self.res))
        gx = max(0, min(self.nx - 1, gx))
        gy = max(0, min(self.ny - 1, gy))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = (gx + 0.5) * self.res + self.ws.x_min
        y = (gy + 0.5) * self.res + self.ws.y_min
        return x, y

    def is_free(self, gx: int, gy: int) -> bool:
        if 0 <= gx < self.nx and 0 <= gy < self.ny:
            return not self.grid[gx, gy]
        return False


class AStarPlanner:
    """Shortest-path A* planner with Euclidean heuristic and corner-cutting prevention."""

    def __init__(
        self,
        env: Optional[Environment] = None,
        resolution_cm: float = 1.0,
    ):
        self.env = env if env is not None else Environment()
        self.res = resolution_cm
        self.grid = OccupancyGrid(self.env, resolution_cm=self.res)

    def plan(
        self,
        start_cm: Tuple[float, float] = (234.5, 63.0),
        goal_cm: Tuple[float, float] = (65.0, 229.5),
    ) -> FastPlannedPath:
        """Find the shortest collision-free path from start to goal."""
        t_start = time.time()

        # Validate start and goal are collision free
        r_robot = self.env.robot.conservative_radius_cm
        start_clr = self.env.point_clearance(start_cm[0], start_cm[1])
        goal_clr = self.env.point_clearance(goal_cm[0], goal_cm[1])

        if start_clr < r_robot:
            raise ValueError(
                f"Start position {start_cm} violates robot clearance ({start_clr:.2f} cm < {r_robot:.2f} cm)"
            )
        if goal_clr < r_robot:
            raise ValueError(
                f"Goal position {goal_cm} violates robot clearance ({goal_clr:.2f} cm < {r_robot:.2f} cm)"
            )

        start_cell = self.grid.world_to_grid(start_cm[0], start_cm[1])
        goal_cell = self.grid.world_to_grid(goal_cm[0], goal_cm[1])

        # Priority queue entries: (f_score, g_score, (gx, gy))
        h_start = math.hypot(start_cm[0] - goal_cm[0], start_cm[1] - goal_cm[1])
        pq = [(h_start, 0.0, start_cell)]

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}

        # 8-connected neighborhood movements: (dx, dy, step_length_multiplier)
        directions = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        ]

        found_goal = False

        while pq:
            f, g, u = heapq.heappop(pq)
            if u == goal_cell:
                found_goal = True
                break

            if g > g_score.get(u, float('inf')):
                continue

            ux, uy = u

            for dx, dy, step_mult in directions:
                vx, vy = ux + dx, uy + dy

                # Check if target cell is free
                if not self.grid.is_free(vx, vy):
                    continue

                # Corner-cutting check: prevent diagonal moves if either adjacent orthogonal cell is occupied
                if dx != 0 and dy != 0:
                    if not self.grid.is_free(ux + dx, uy) or not self.grid.is_free(ux, uy + dy):
                        continue

                tentative_g = g + step_mult * self.res
                v = (vx, vy)

                if tentative_g < g_score.get(v, float('inf')):
                    g_score[v] = tentative_g
                    came_from[v] = u

                    # Euclidean distance heuristic to goal
                    wx, wy = self.grid.grid_to_world(vx, vy)
                    h = math.hypot(wx - goal_cm[0], wy - goal_cm[1])
                    f_val = tentative_g + h
                    heapq.heappush(pq, (f_val, tentative_g, v))

        if not found_goal:
            raise RuntimeError(
                f"A* planner failed to find a valid collision-free path from {start_cm} to {goal_cm}."
            )

        # Reconstruct path from goal back to start
        curr = goal_cell
        grid_path = []
        while curr in came_from:
            grid_path.append(self.grid.grid_to_world(curr[0], curr[1]))
            curr = came_from[curr]

        grid_path.append(start_cm)
        grid_path.reverse()
        grid_path[-1] = goal_cm  # Ensure exact goal coordinate

        # Smooth / shortcut the path
        smoothed = self._smooth_path(grid_path)

        # Calculate headings
        headings = []
        for i in range(len(smoothed)):
            if i < len(smoothed) - 1:
                dx = smoothed[i + 1][0] - smoothed[i][0]
                dy = smoothed[i + 1][1] - smoothed[i][1]
                headings.append(math.atan2(dy, dx))
            else:
                headings.append(headings[-1] if headings else 0.0)

        # Path metrics
        total_len = 0.0
        min_c = float('inf')
        for i in range(len(smoothed) - 1):
            p1 = smoothed[i]
            p2 = smoothed[i + 1]
            total_len += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            c = self.env.segment_clearance(p1, p2, samples=int(math.hypot(p2[0]-p1[0], p2[1]-p1[1])) + 2)
            if c < min_c:
                min_c = c

        t_elapsed = time.time() - t_start

        return FastPlannedPath(
            raw_waypoints=grid_path,
            smoothed_waypoints=smoothed,
            headings_rad=headings,
            total_length_cm=total_len,
            min_clearance_cm=min_c,
            waypoint_count=len(smoothed),
            planning_time_s=t_elapsed,
            grid_resolution_cm=self.res,
        )

    def _smooth_path(self, raw_path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Greedy raycast shortcutting: bypass intermediate grid steps with straight line
        segments as long as the entire robot footprint remains collision-free.
        """
        if len(raw_path) <= 2:
            return raw_path

        r_robot = self.env.robot.conservative_radius_cm
        smoothed = [raw_path[0]]
        curr_idx = 0

        while curr_idx < len(raw_path) - 1:
            next_idx = len(raw_path) - 1
            while next_idx > curr_idx + 1:
                p1 = smoothed[-1]
                p2 = raw_path[next_idx]
                dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                samples = max(5, int(dist * 1.5))
                # Segment clearance must strictly respect the conservative radius
                if self.env.segment_clearance(p1, p2, samples=samples) >= r_robot:
                    # Also verify exact polygon footprint sweep
                    heading = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    collided = False
                    for s in range(samples + 1):
                        t = s / samples
                        x = p1[0] + t * (p2[0] - p1[0])
                        y = p1[1] + t * (p2[1] - p1[1])
                        if self.env.robot.is_collision(x, y, heading, self.env.workspace, self.env.obstacle):
                            collided = True
                            break
                    if not collided:
                        break
                next_idx -= 1

            smoothed.append(raw_path[next_idx])
            curr_idx = next_idx

        return smoothed


def main():
    """Standalone CLI verification of the A* planner."""
    planner = AStarPlanner(resolution_cm=1.0)
    start = (234.5, 63.0)
    goal = (65.0, 229.5)

    print("=" * 60)
    print("HW4 FAST A* PATH PLANNER INITIALIZING")
    print(f"Start: {start} cm | Goal: {goal} cm")
    path = planner.plan(start, goal)

    print(f"Planning complete in {path.planning_time_s * 1000.0:.1f} ms")
    print(f"Raw A* points: {len(path.raw_waypoints)} | Smoothed waypoints: {path.waypoint_count}")
    print(f"Fast A* Path Length: {path.total_length_cm:.2f} cm")
    print(f"Minimum Obstacle Clearance: {path.min_clearance_cm:.2f} cm")
    print("Smoothed Waypoints:")
    for i, (wx, wy) in enumerate(path.smoothed_waypoints):
        heading_deg = math.degrees(path.headings_rad[i])
        print(f"  WP {i}: ({wx:.1f}, {wy:.1f}) cm | Heading: {heading_deg:.1f}°")
    print("=" * 60)


if __name__ == '__main__':
    main()
