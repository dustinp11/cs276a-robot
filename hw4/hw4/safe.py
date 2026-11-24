"""Future safe mode interfaces; no planning or execution is implemented."""


def plan_path(grid, start_xy, goal_xy) -> list[tuple[float, float]]:
    """Plan a future high-clearance route through the robot-center grid.

    Returns:
        Ordered world-frame (x_m, y_m) waypoints, including start and goal.
        An unreachable goal should raise ValueError rather than return a route.
    """
    ...


def create_navigator(grid, path, forward, turn, policy, now, initial_yaw=0.0) -> "Navigator":
    """Construct a future safe-mode controller without starting motion.

    Returns:
        A Navigator-compatible controller with tick(now, samples), finish(state,
        reason), and the telemetry fields used by MissionNode. Commands use m/s
        and rad/s; time uses monotonic seconds. Arrival must use measured pose.
    """
    ...
