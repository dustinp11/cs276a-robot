"""Fast mode: distance-based A* with the existing calibrated waypoint follower."""

import heapq
import math


def build_graph(grid):
    """Cache geometric edge length and robot-body clearance for every grid edge."""
    # cache continuous clearance before searching routes
    graph = {point: {} for point in grid.cells.values()}
    for cell, point in grid.cells.items():
        for neighbor in grid.neighbors(cell):
            other = grid.cells[neighbor]
            if other not in graph[point]:
                edge = math.dist(point, other), grid.clearance(point, other)
                graph[point][other] = graph[other][point] = edge

    return graph


def connect_anchor(graph, grid, point):
    """Connect an exact endpoint to up to eight nearest visible grid positions."""
    # preserve exact endpoints when connecting grid
    if not grid.segment_free(point, point):
        raise ValueError(f'Endpoint {point} violates robot clearance.')
    candidates = sorted(graph, key=lambda other: math.dist(point, other))
    graph.setdefault(point, {})
    connected = 0

    # connect actual positions without snapping
    for other in candidates:
        if other == point or not grid.segment_free(point, other):
            continue
        edge = math.dist(point, other), grid.clearance(point, other)
        graph[point][other] = graph[other][point] = edge
        connected += 1
        if connected == 8:
            break


def shortest_path(graph, start, goal, clearance):
    """Run Euclidean-heuristic A* subject to a minimum edge clearance."""
    # prioritize traveled cost plus remaining distance
    costs, previous = {start: 0.0}, {}
    queue = [(math.dist(start, goal), 0.0, start)]
    while queue:
        _, cost, point = heapq.heappop(queue)
        if cost > costs[point]:
            continue
        if point == goal:
            return reconstruct(previous, goal)

        # expand only sufficiently clear edges
        for other, (length, gap) in graph[point].items():
            candidate = cost + length
            if gap + 1e-9 >= clearance and candidate < costs.get(other, math.inf):
                costs[other], previous[other] = candidate, point
                heapq.heappush(queue, (candidate + math.dist(other, goal), candidate, other))

    raise ValueError('No collision-free route connects start and goal.')


def reconstruct(previous, point):
    """Trace predecessor links into start-to-goal world coordinates."""
    # recover ordered waypoints from predecessor links
    path = [point]
    while point in previous:
        point = previous[point]
        path.append(point)
    return list(reversed(path))


def simplify_path(grid, path, clearance):
    """Remove waypoints only when the shortcut preserves the chosen clearance."""
    # retain clearance when removing intermediate waypoints
    result, index = [path[0]], 0
    while index < len(path) - 1:
        next_index = len(path) - 1
        while next_index > index + 1 and not grid.segment_free(path[index], path[next_index], clearance - 1e-8):
            next_index -= 1
        result.append(path[next_index])
        index = next_index

    return result


def plan_path(grid, start_xy, goal_xy, graph=None) -> list[tuple[float, float]]:
    """Return collision-free world-meter waypoints minimizing grid travel distance."""
    # preserve the existing distance planning baseline
    start, goal = tuple(start_xy), tuple(goal_xy)
    graph = build_graph(grid) if graph is None else {p: dict(edges) for p, edges in graph.items()}
    connect_anchor(graph, grid, start)
    connect_anchor(graph, grid, goal)

    # add a valid direct connection
    if start != goal and grid.segment_free(start, goal):
        edge = math.dist(start, goal), grid.clearance(start, goal)
        graph[start][goal] = graph[goal][start] = edge

    clearance = grid.margin
    path = shortest_path(graph, start, goal, clearance)
    return simplify_path(grid, path, clearance)


def create_navigator(grid, path, forward, turn, policy, now, initial_yaw=0.0):
    """Return the existing Navigator using meter waypoints and a radian start yaw.

    Fast currently means distance planning; motion retains stopped pose checks.
    """
    from hw4.navigation import Navigator
    return Navigator(grid, path, forward, turn, policy, now, initial_yaw)
