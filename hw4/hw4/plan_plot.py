"""Dependency-free overhead SVG and turn table for inspecting a plan before motion."""

from html import escape
import math

SCALE, LEFT, BOTTOM = 220.0, 140.0, 810.0


def screen(point):
    """Convert world-meter xy to an SVG point with world +y upward."""
    return LEFT + point[0] * SCALE, BOTTOM - point[1] * SCALE


def text(x, y, label, size=14, anchor='start', color='#17233b'):
    """Draw a readable label with an opaque white outline."""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" stroke="white" stroke-width="3" paint-order="stroke">{escape(str(label))}</text>')


def arrow(point, yaw_deg, color, length=0.14):
    """Draw an orientation arrow in world coordinates."""
    yaw = math.radians(yaw_deg)
    end = point[0] + length * math.cos(yaw), point[1] + length * math.sin(yaw)
    x1, y1 = screen(point)
    x2, y2 = screen(end)
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            'stroke-width="2.5" marker-end="url(#arrow)"/>')


def plan_segments(environment, path):
    """Return geometric headings and signed turns; positive turns are left/CCW."""
    # use the same angle wrapping as navigation
    previous = environment['start']['yaw_deg']
    segments = []
    for index, (start, end) in enumerate(zip(path, path[1:]), 1):
        heading = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360
        turn = (heading - previous + 180) % 360 - 180
        segments.append({'waypoint': index, 'from_xy_m': start, 'to_xy_m': end,
                         'distance_m': math.dist(start, end), 'heading_deg': heading, 'turn_deg': turn})
        previous = heading

    return segments


def plan_description(environment, plans):
    """Return a centimeter waypoint and turn table suitable for console output."""
    # identify geometric intent separately from actual motion
    start = environment['start']
    lines = ['PLANNED PATH ONLY; no measured trajectory',
             f'Start: ({start["x_m"]*100:.1f}, {start["y_m"]*100:.1f}) cm, heading {start["yaw_deg"]:.1f} deg',
             'Headings: 0=+x/right, 90=+y/up; +turn=left, -turn=right']
    for name, path in plans.items():
        lines += [f'Mode: {name}', 'Target      x_cm    y_cm   distance_cm   heading_deg   turn_deg']
        for segment in plan_segments(environment, path):
            x, y = segment['to_xy_m']
            lines.append(f'W{segment["waypoint"]:<7} {x*100:7.1f} {y*100:7.1f} '
                         f'{segment["distance_m"]*100:13.1f} {segment["heading_deg"]:13.1f} {segment["turn_deg"]:+10.1f}')

    if 'final_goal' in environment:
        final = environment['final_goal']
        lines.append(f'Final goal ({final["x_m"]*100:.1f}, {final["y_m"]*100:.1f}) cm is saved but not executed.')
    return '\n'.join(lines) + '\n'


def background(grid):
    """Draw physical walls, excluded center positions, and the 5 cm planning grid."""
    # distinguish physical geometry from clearance exclusions
    top = BOTTOM - grid.height * SCALE
    padding = (grid.radius + grid.margin) * SCALE
    width, height = grid.width * SCALE, grid.height * SCALE
    parts = [f'<rect x="{LEFT}" y="{top}" width="{width}" height="{height}" fill="#f7dcdc" stroke="#17233b" stroke-width="3"/>',
             f'<rect x="{LEFT+padding}" y="{top+padding}" width="{width-2*padding}" height="{height-2*padding}" fill="#f6f8fb" stroke="#b26b6b" stroke-dasharray="5 4"/>']
    for point in grid.cells.values():
        x, y = screen(point)
        parts.append(f'<circle cx="{x}" cy="{y}" r="0.8" fill="#c3ccd8"/>')

    # expand each obstacle by the center exclusion
    for xmin, xmax, ymin, ymax in grid.rectangles:
        x, y = screen((xmin, ymax))
        w, h = (xmax - xmin) * SCALE, (ymax - ymin) * SCALE
        parts.append(f'<rect x="{x-padding}" y="{y-padding}" width="{w+2*padding}" height="{h+2*padding}" rx="{padding}" fill="#f7dcdc" stroke="#b26b6b" stroke-dasharray="5 4"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#414b5a"/>')
        parts.append(text(x + w/2, y + h/2, 'OBSTACLE', 13, 'middle', '#111'))

    return parts


def axes(grid):
    """Label centimeter axes without changing world-frame orientation."""
    parts = []
    for value in range(0, math.floor(grid.width * 100) + 1, 50):
        x, _ = screen((value / 100, 0))
        parts.append(text(x, BOTTOM + 25, value, 13, 'middle'))
    for value in range(0, math.floor(grid.height * 100) + 1, 50):
        _, y = screen((0, value / 100))
        parts.append(text(LEFT - 100, y + 4, value, 13, 'end'))
    parts += [text(LEFT + grid.width*SCALE/2, BOTTOM + 65, 'x (cm) →', 17, 'middle'),
              text(LEFT - 100, BOTTOM - grid.height*SCALE - 24, 'y (cm) ↑', 17)]
    return parts


def landmarks(grid, environment):
    """Show every tag's family/ID and the outward printed-face normal."""
    # keep wall labels outside the workspace
    parts = []
    for tag in environment['landmarks']:
        point = tag['x_m'], tag['y_m']
        x, y = screen(point)
        color = '#a05c00' if tag.get('family', '36h11') == '36h11' else '#8438a5'
        parts += [arrow(point, tag['face_yaw_deg'], color),
                  f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>']
        dx, dy, anchor = 12.0, -10.0, 'start'
        if point[0] == 0 or tag['face_yaw_deg'] == 180:
            dx, dy, anchor = -12.0, -8.0, 'end'
        if point[1] == 0:
            dx, dy, anchor = 0.0, 45.0, 'middle'
        elif point[1] == grid.height or tag['face_yaw_deg'] == 90:
            dx, dy, anchor = 0.0, -15.0, 'middle'
        if point[0] == grid.width:
            dx, dy, anchor = 12.0, -8.0, 'start'
        label = f'{tag.get("family", "36h11")}:{tag["id"]}'
        parts.append(text(x + dx, y + dy, label, 13, anchor, color))

    return parts


def route(grid, environment, plans):
    """Number waypoints, show travel direction, and mark the starting robot heading."""
    # distinguish the requested route from physical motion
    parts = []
    for name, path in plans.items():
        for start, end in zip(path, path[1:]):
            x1, y1 = screen(start)
            x2, y2 = screen(end)
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#1769cc" stroke-width="3.5" marker-end="url(#arrow)"/>')
        for index, point in enumerate(path):
            x, y = screen(point)
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="white" stroke="#1769cc" stroke-width="2"/>')
            parts.append(text(x + 12, y - 10, f'W{index}', 15, color='#1769cc'))

    # show actual robot size at placement
    start, goal = environment['start'], environment['goal']
    point = start['x_m'], start['y_m']
    x, y = screen(point)
    parts += [f'<circle cx="{x}" cy="{y}" r="{grid.radius*SCALE}" fill="none" stroke="#17834c" stroke-width="2"/>',
              arrow(point, start['yaw_deg'], '#17834c', 0.23),
              text(x + 10, y + 35, f'START ({point[0]*100:.1f}, {point[1]*100:.1f}) cm; {start["yaw_deg"]:g}°', 14, color='#17834c')]
    x, y = screen((goal['x_m'], goal['y_m']))
    left_of_obstacle = grid.rectangles and goal['x_m'] < min(rect[0] for rect in grid.rectangles)
    label_x, anchor = (x - 12, 'end') if left_of_obstacle else (x + 12, 'start')
    parts.append(text(label_x, y + 18, f'{goal.get("label", "goal").upper()} ({goal["x_m"]*100:.1f}, {goal["y_m"]*100:.1f}) cm', 14, anchor))
    if 'final_goal' in environment:
        final = environment['final_goal']
        x, y = screen((final['x_m'], final['y_m']))
        parts.append(f'<circle cx="{x}" cy="{y}" r="7" fill="none" stroke="#68758b" stroke-width="2" stroke-dasharray="3 2"/>')
        parts.append(text(x + 12, y - 10, 'FINAL GOAL (inactive)', 13, color='#68758b'))
    return parts


def map_svg(grid, environment, plans):
    """Create a self-contained engineering map and numbered turn table."""
    # reserve room for map and geometric instructions
    table = plan_description(environment, plans).splitlines()
    height = max(1120, 930 + len(table) * 23)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{height}" height="{height}" viewBox="0 0 {height} {height}" font-family="Arial,sans-serif">',
             f'<rect width="{height}" height="{height}" fill="white"/>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
             text(45, 35, 'HW4 fast: planned path before motion', 25),
             text(45, 63, 'Blue: route   Dark: physical obstacle   Pink: excluded robot-center positions', 15),
             text(45, 88, 'Orange: 36h11 wall tags   Purple: 41h12 obstacle tags   Tag arrows: face normals', 15),
             text(45, 113, f'Robot radius {grid.radius*100:.1f} cm + planning clearance {grid.margin*100:.1f} cm; all axes in cm', 15)]
    parts += background(grid) + axes(grid) + route(grid, environment, plans) + landmarks(grid, environment)
    for index, line in enumerate(table):
        parts.append(f'<text x="45" y="{925+index*23}" font-family="monospace" font-size="13">{escape(line)}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)
