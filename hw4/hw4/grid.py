"""Collision geometry and a grid of robot-center positions, in world meters."""

from dataclasses import dataclass
import math


def point_segment_distance(point, start, end):
    """Return Euclidean distance to a closed line segment, in meters."""
    # project onto the finite movement segment
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if not length2 else ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2
    t = max(0.0, min(1.0, t))
    return math.dist(point, (start[0] + t * dx, start[1] + t * dy))


def intersects_rectangle(start, end, rectangle):
    """Check segment intersection using two closed axis-aligned slabs."""
    # clip against both obstacle coordinate slabs
    lower, upper = 0.0, 1.0
    for axis, bounds in enumerate((rectangle[:2], rectangle[2:])):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if not bounds[0] <= start[axis] <= bounds[1]:
                return False
            continue

        # clip movement against each slab
        first, last = sorted((bound - start[axis]) / delta for bound in bounds)
        lower, upper = max(lower, first), min(upper, last)
        if lower > upper:
            return False

    return True


def rectangle_distance(start, end, rectangle):
    """Return exact segment-to-rectangle distance, zero for any intersection."""
    # check interior crossing before measuring separation
    if intersects_rectangle(start, end, rectangle):
        return 0.0
    xmin, xmax, ymin, ymax = rectangle
    corners = [(x, y) for x in (xmin, xmax) for y in (ymin, ymax)]
    distances = [point_segment_distance(corner, start, end) for corner in corners]
    for x, y in (start, end):
        distances.append(math.hypot(max(xmin - x, 0, x - xmax), max(ymin - y, 0, y - ymax)))

    return min(distances)


@dataclass
class Grid:
    """Grid cells [column_x, row_y] and continuous circular-footprint collision checks."""

    width: float
    height: float
    radius: float
    margin: float
    resolution: float
    rectangles: list
    cells: dict

    def clearance(self, start, end=None):
        """Return minimum robot-body clearance along a segment, before safety margin."""
        # treat workspace boundaries as collision surfaces
        end = start if end is None else end
        wall = min(x for point in (start, end) for x in
                   (point[0], self.width - point[0], point[1], self.height - point[1]))
        distances = [wall] + [rectangle_distance(start, end, rect) for rect in self.rectangles]
        return min(distances) - self.radius

    def segment_free(self, start, end, clearance=None):
        """Require the entire segment to preserve the configured body clearance."""
        threshold = self.margin if clearance is None else max(self.margin, clearance)
        return self.clearance(start, end) >= threshold + 1e-9

    def neighbors(self, cell):
        """Yield eight-connected neighbors without diagonal obstacle cutting."""
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            other = cell[0] + dx, cell[1] + dy
            if other in self.cells and self.segment_free(self.cells[cell], self.cells[other]):
                yield other


def build_grid(environment, resolution_m=0.05, extra_margin_m=0.0):
    """Enclose the full robot footprint in a circle and discretize free space."""
    # include footprint and error allowance together
    workspace, robot = environment['workspace'], environment['robot']
    radius = math.hypot(robot['length_m'], robot['width_m']) / 2
    rectangles = [(item['x_min_m'], item['x_max_m'], item['y_min_m'], item['y_max_m'])
                  for item in environment['obstacles']]
    grid = Grid(workspace['width_m'], workspace['height_m'], radius,
                robot['safety_margin_m'] + extra_margin_m, resolution_m, rectangles, {})

    # keep only collision-free grid positions
    for ix in range(math.floor(grid.width / resolution_m) + 1):
        for iy in range(math.floor(grid.height / resolution_m) + 1):
            point = ix * resolution_m, iy * resolution_m
            if grid.segment_free(point, point):
                grid.cells[ix, iy] = point

    return grid
