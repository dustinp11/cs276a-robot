# HW3 plan: first traverse a 7 ft square

## Scope and sources

Read `../AGENTS.md`, `../context.md`, `../assignment_writeups/hw3-2026.pdf`,
and the HW2 controller, localization, motor mapping, motor driver, launch, and
package setup. The checked-out code is authoritative where the context differs.
The context's description of the old HW3 work is historical; this branch starts
from HW2 commit `e6c386e`.

This document contains a plan and proposed interfaces only. No executable code,
Python stubs, launch changes, or package changes are part of this step. The Python
blocks below show signatures and docstrings; their ellipses mark unimplemented
bodies. Wait for implementation instructions before changing robot behavior.

Current objective: one counterclockwise lap of a **7 ft x 7 ft path** using the
calibrated HW2 motion behavior. Feet are an assumption based on the assignment;
7 ft = 2.1336 m. This is the path size, not the arena size. The assignment calls
for a 10 ft square arena, with 8 ft as the minimum reduced arena side. Robot size,
turning sweep, and placement must fit within the available clearance.

This milestone includes vision-assisted quarter-turns, but not full SLAM, an
octagon, a surveyed map, plots, a web interface, or a new ROS package. A completed
lap is not evidence that localization or mapping works.

## Revised turning objective

Timed turns alone are insufficient. At a corner, observe a stationary AprilTag,
save it as a relative orientation reference, turn toward a 90-degree change,
stop, reobserve, and correct the measured residual when vision permits. No tag's
world position or world orientation needs to be known beforehand.

The original tag will usually leave a forward camera's field of view before a
90-degree turn finishes. Its disappearance is not a 90-degree measurement. Seeing
a different, previously unobserved tag after the turn is also insufficient: its
relationship to the first tag is unknown. Do not center that new tag and call the
result a right angle.

Use the same tag's full relative orientation before/after small stopped turns.
Its unknown fixed world orientation cancels when computing robot heading change.
Compare the horizontal direction of a consistent tag axis (the printed face
normal) after applying the optical-to-base rotation. For planar motion,
`robot_yaw_change = wrap(tag_normal_yaw_before - tag_normal_yaw_after)`.
Reject ambiguous/flipped PnP orientations and tags with unreliable projected
normals. Translation and camera offset can change pixel bearing even during a
turn; bearing subtraction alone is only an approximation and is not the chosen
heading measurement.

If two tags are visible together, record their relative orientations and hand
the reference to the second tag before losing the first. This small, run-local
orientation dictionary can extend heading feedback through the turn; it contains
no surveyed positions and is not an EKF map. Only accept a handoff while heading
is visually constrained. Propagate confidence through handoffs rather than
treating every new anchor as exact.

The turning sequence is: stop/settle -> capture reference -> bounded coarse turn
while retaining useful visibility -> stop/settle -> measure -> bounded corrective
pulse -> repeat. Reduce pulse size near the target, with a maximum correction
count and time budget. HW2's minimum effective pulse and approximately 10-degree
alignment tolerance are starting hardware constraints, not a promise of precise
quarter-turns. Log requested, visually measured, and timed-only rotation separately.

If visibility cannot span the quarter-turn, a timed remainder is a prediction,
not verification. Default to stopping with an explicit unverified-turn result
instead of advancing to the next side. Test actual tag overlap first; continuous
coverage, another independently validated heading sensor, or a later estimator
would be needed to close a blind interval reliably. Do not silently substitute
the commanded angle as a visual measurement.

## Motion and coordinate contract

- Use a local start frame: initial robot center `(0, 0)`, initial forward direction
  `+x`, left `+y`, positive yaw counterclockwise. This chooses a coordinate gauge;
  it does not supply the robot's unknown pose in the surveyed world frame.
- Planned corners in meters: `(0, 0) -> (2.1336, 0) -> (2.1336, 2.1336) ->
  (0, 2.1336) -> (0, 0)`.
- Drive each side straight, stop, settle, turn left by `pi/2` with relative-tag
  feedback, stop, and settle. Advance only after a verified turn.
  Include the fourth turn to finish at the initial commanded heading. Stop after
  the lap; do not automatically repeat.
- Start with the HW2 nominal straight speed of 0.18 m/s and 1.0 s settle time.
  Retain the existing 0.18 m/s end-of-long-segment taper. Expose speed, side length
  in meters, settle time, and full-turn calibration as ROS parameters later.
- Initially retain 3.75 s per full turn: a nominal quarter-turn is 0.9375 s.
  This is a surface-dependent starting calibration, not a measured guarantee.
- Keep angular velocity zero during straight travel and linear velocity zero
  during turns. The existing mapper saturates in-place turns near motor commands
  `[-0.5, +0.5]`; reducing `Twist.angular.z` does not provide a slower turn.
- Use monotonic elapsed time and interruptible waits for motion. Refresh commands
  about every 0.05 s, below the motor driver's 0.15 s timeout. Bound the last wait
  by the remaining duration rather than rounding a quarter-turn to whole ticks.
- Planned corners are targets. Command-integrated coordinates are dead reckoning,
  not measured positions. Never assign the next target to the estimated position
  just because a leg ended. Measure physical closure error separately on the robot.

## Reuse and file budget

| Existing component | Intended use |
| --- | --- |
| `robot_testing/motor_control.py` | Reuse unchanged: serial protocol, command clipping, command timeout, shutdown stop. |
| `robot_testing/velocity_mapping.py` | Reuse unchanged: tuned straight mapping and fixed-speed in-place turn mapping. |
| `hw2_control.ControlNode.drive_segment`, `rotate_angle`, `stop_robot` | Move into a small shared `MotionNode` in the same existing module; both controllers inherit them. |
| `hw2_control.ControlNode` | Retain the HW2 mission, detections, known-map service, and coordinate logger. Change its initialization only as needed for the shared base. |
| Camera and AprilTag scripts/calibration | Reuse now for relative-tag turning observations. |
| `hw2_localization.quaternion_transform` and camera mount geometry | Reuse for relative observations; avoid the surveyed-map pose estimator for HW3. |

Proposed eventual additions: **two executable/configuration files**,
`robot_testing/hw3_control.py` and `launch/hw3.launch.py`, inside the existing
`robot_testing` package. This document is the only new file at the planning stage.
Edit existing `setup.py` later to install the launch file and add the entry point.
Use launch/ROS parameters rather than adding a YAML file or another helper module.
Add focused checks in an existing test file when implementation begins.

The shared base's constructor must only initialize motion resources. It must not
load surveyed landmarks, subscribe to vision, clear coordinate logs, or start a
mission thread. Do not instantiate or subclass the current `ControlNode` directly
for HW3: its constructor starts the HW2 mission and creates those dependencies.
`ControlNode` will instead inherit the extracted base and retain its own startup.

Two intentional improvements to the shared primitives need HW2 regression checks:

1. Honor the requested straight speed. Currently `drive_segment(..., speed=0.18)`
   silently selects 0.35 m/s for distances greater than 0.3 m. Keep HW2's explicit
   0.35 m/s calls and end taper, while allowing HW3 to select 0.18 m/s.
2. Use measured elapsed time rather than adding an assumed 0.05 s per loop.
   Preserve the calibrated turn rate and directions. Hardware tuning must follow
   because changed timing can change the physical result.

Do not call HW2 `move_forward`, `rotate(target_tag)`, `get_tag_z_distance`, or
`get_estimated_position` from HW3. They depend on surveyed tags, target-tag search,
or known-map pose updates. Reusing them would carry HW2 assumptions into SLAM.

## Proposed function contracts

### Shared motion in existing `robot_testing/hw2_control.py`

`Node` denotes `rclpy.node.Node`. The signatures retain the existing movement
arguments so HW2 call sites can stay intact. `current_x/current_y` remain a simple
command-integrated fallback; they are not a future EKF state.

```python
class MotionNode(Node):
    """Share calibrated timed motion between the HW2 and HW3 missions."""

    def __init__(self, node_name: str) -> None:
        """Initialize /cmd_vel, calibration, cancellation, and local position.

        Create no mission thread, vision subscription, pose service, or log file.
        Store a stop event for interrupting motion and settle waits. Let each
        mission validate its parameters before starting movement.
        """
        ...

    def stop_robot(self) -> None:
        """Publish a zero Twist without permanently cancelling the mission."""
        ...

    def request_stop(self) -> None:
        """Cancel the mission, interrupt waits, and publish a zero Twist."""
        ...

    def settle(self, duration_s: float) -> None:
        """Stop and wait for duration_s seconds, returning early on cancellation.

        Reject negative or nonfinite durations. Preserve the stopped state.
        Callers must not begin another motion after cancellation or ROS shutdown.
        """
        ...

    def drive_segment(
        self, distance: float, ux: float, uy: float, speed: float = 0.18
    ) -> None:
        """Drive a timed straight segment and stop on completion or interruption.

        Args:
            distance: Nonnegative travel distance in meters.
            ux: Local-frame x component of the unit forward heading.
            uy: Local-frame y component of the unit forward heading.
            speed: Nonzero signed speed in m/s; negative means reverse.

        Honor the requested speed, retaining HW2's end taper without increasing
        speed above its requested magnitude. Use monotonic elapsed time and
        integrate only the commanded travel into current_x/current_y. The unit
        vector records heading for that estimate; it does not steer the robot.
        Check cancellation and ROS state before every command, and stop in a
        finally block. Reject invalid inputs before commanding movement.
        """
        ...

    def rotate_angle(self, angle_rad: float, direction: float = 1.0) -> None:
        """Turn through a timed angle using the calibrated full-turn duration.

        Args:
            angle_rad: Nonnegative angle in radians.
            direction: +1.0 for counterclockwise or -1.0 for clockwise.

        Use elapsed time and the existing fixed physical turn-speed calibration.
        Publish regularly, bound the final wait, and stop in a finally block.
        Reject nonfinite angles and directions other than +1.0 or -1.0.
        This commands a turn; it does not measure the achieved heading.
        """
        ...
```

Zero-distance and zero-angle requests should leave the robot stopped. No motion
method should report successful physical arrival merely because its timer expired.

### New `robot_testing/hw3_control.py`

Keep the square runner small. A separate path planner, generic state-machine
framework, or waypoint file is unnecessary for four equal sides. A cancelled or
failed mission must stop and exit, never advance to another side.

```python
class SquareControlNode(MotionNode):
    """Drive a square using timed straight travel and visually checked turns."""

    def __init__(self) -> None:
        """Initialize and validate square parameters without starting movement.

        Defaults: side_length_m=2.1336, linear_speed_mps=0.18,
        settle_time_s=1.0, turn_time_360_s=3.75. Require positive finite side,
        speed, and turn duration, and a finite nonnegative settle duration.
        Subscribe to detections and timestamped TF for relative heading feedback.
        Initialize a run-local tag orientation dictionary, with no surveyed map
        or SLAM dependency. Validate visual age, tolerance, and correction limits.
        """
        ...

    def run(self) -> None:
        """Drive four sides and four left quarter-turns, then remain stopped.

        Begin with a stopped settle interval. For each nominal heading
        0, pi/2, pi, 3*pi/2, call the shared straight primitive with its unit
        direction, settle, call turn_with_landmarks(pi/2), and settle. Stop the
        mission if that turn cannot be visually verified. Log side and phase.
        Check cancellation between phases and always stop in a finally block.
        Do not replace integrated position with a planned corner coordinate.
        """
        ...

    def capture_heading_reference(self) -> dict[tuple[str, int], float]:
        """Record visible tag face-normal yaw at the start of a stopped turn.

        Return (family, ID) -> base-frame normal yaw in radians from one fresh
        detection frame and matching TF. Apply camera mount rotation and quality
        gates; return an empty dictionary if no reference is usable. No world
        landmark coordinates or orientations are required.
        """
        ...

    def measure_turn(
        self, reference: dict[tuple[str, int], float]
    ) -> float | None:
        """Estimate signed robot yaw change from fresh matched tag orientations.

        Compare the same stationary tag before/after rotation, with positive
        change counterclockwise. Reject stale, inconsistent, or flipped poses;
        return None when no trustworthy reference remains. Maintain overlapping
        tag handoffs only while visual heading is constrained. A new tag alone
        cannot establish the completed angle. Never use elapsed time as a visual
        measurement; expose uncertainty for the turn's acceptance decision.
        """
        ...

    def turn_with_landmarks(self, angle_rad: float) -> bool:
        """Turn by a signed relative angle using stopped visual measurements.

        Capture a reference, perform bounded timed pulses via rotate_angle, and
        settle before measuring the residual. Use fresh, consistent observations
        and a hardware-feasible tolerance; bound correction count and duration.
        Return True only for a visually verified turn within tolerance and with
        acceptable uncertainty. Stop and return False on reference loss, exhausted
        corrections, cancellation, or unavailable initial vision. Always stop on
        exit. A blind timed remainder cannot make this return True.
        """
        ...


def main(args=None) -> None:
    """Spin ROS while one worker executes the square, then cleanly shut down.

    Construct the node before starting the worker so invalid parameters cannot
    start motion. Keep ROS callbacks responsive while timed motion runs. After
    success, failure, or interruption, request cancellation, join the worker
    before destroying its publisher, stop, destroy the node, and shut down ROS.
    Report worker failures to the main thread and exit without another lap.
    """
    ...
```

### New `launch/hw3.launch.py`

```python
def generate_launch_description() -> LaunchDescription:
    """Launch the existing motor stack and the HW3 square controller.

    Forward side length, straight speed, settle time, and full-turn calibration
    as ROS parameters. Start motor_control, velocity_mapping, and hw3_control
    only. Do not launch hw2_control or hw2_localization. Terminate the launch
    when the square controller exits; retain the motor driver's shutdown stop.
    """
    ...
```

### Relative vision interfaces for square turning

These signatures describe the relative observation handling in `hw3_control.py`.
Reuse existing `Detection`/`Observation` records
and `quaternion_transform` where useful. A detector observation is relative
geometry, not a known world pose.

```python
def relative_landmark_measurement(
    camera_tag: np.ndarray,
    base_camera: np.ndarray,
) -> tuple[float, float] | None:
    """Convert a tag transform to planar range and bearing at the robot base.

    Args:
        camera_tag: 4x4 transform mapping tag coordinates into camera optical
            coordinates, obtained at the detection timestamp.
        base_camera: 4x4 calibrated transform mapping camera optical coordinates
            into base coordinates (+x forward, +y left, +z up).

    Returns:
        (horizontal range_m, bearing_rad), positive bearing to the left, or None
        for nonfinite or unusable geometry. Apply the camera mount translation
        and optical-axis rotation before projecting onto the base xy plane.
        Use no surveyed landmark position or hardcoded pixel-bearing constants.
    """
    ...


def on_detections(self, message: AprilTagDetectionArray) -> None:
    """Cache fresh relative tag poses for turning and later SLAM observations.

    This will be a node method when vision is introduced. Validate detection
    timestamp, family, ID, hamming, and margin; look up camera-to-tag TF at that
    same timestamp. Reject missing or stale transforms and repeated frames.
    Retain (family, ID), timestamp, and full relative pose for turn feedback;
    range/bearing alone discards the orientation needed here. Pass consistent
    snapshots to the motion thread under a lock. Missing vision must not trigger
    unbounded tag searching or make a known-map pose update.
    """
    ...
```

Later SLAM must grow its landmark state when a new ID appears, with covariance
and robot-landmark cross-covariance; it must not preallocate eight known tags or
use `config/landmarks.json` as prior map data. Reuse camera intrinsics, measured
tag size, mount geometry, and detector quality gates independently of surveyed
world coordinates. The current detector TF configuration names only tags
0, 1, 3, and 5; verify its configuration and naming behavior for the actual arena
IDs so newly detected tags can provide transforms. The turn controller must not
depend on a preselected first ID or an assumed number of tags.

## Implementation order when requested

1. Extract the shared motion base in the existing HW2 module; retain HW2's
   high-level mission and its existing public primitive arguments.
2. Correct speed selection and elapsed-time handling in those shared primitives.
   Add cancellation and stopped cleanup. Keep hardware calibration values explicit.
3. Add relative tag observations and the stopped turn-feedback loop in the same
   new controller file. Verify usable tag overlap and pose orientation stability.
4. Add the square runner, launch file, and existing-package entry point. Use the
   existing camera/detector scripts in separate terminals; no known-map service.
5. Run local checks, then test a visually verified quarter-turn and a measured
   straight segment on the robot before attempting a complete lap.

## Validation when implementation begins

Locally, use mocked time/publishers in the existing motion test file to check
requested speed on a 2.1336 m segment, elapsed-time turn termination, four equal
legs/four quarter-turns, stopping on cancellation/error, and no commands after
cancellation. With synthetic relative tag transforms, check positive/negative
heading change, invariance to unknown fixed tag orientation and translation,
overlapping handoffs, rejection of stale/ambiguous poses, and stopping when a
new tag cannot verify the old reference. Check that construction alone starts
no motion and that HW2 keeps its explicit speeds and mission sequence. Confirm
the square requires no known-landmark service. Run syntax/package checks and the
relevant existing checks that the local environment supports. These do not
validate ROS integration or physical traversal.

On the robot, first measure a single straight segment and a quarter-turn on the
actual surface, then one complete lap. Measure side lengths, corner angles,
endpoint displacement, final heading, and stop behavior. A provisional tuning
target is side error <= 0.10 m, quarter-turn error <= 5 degrees, endpoint error
<= 0.25 m, and final heading error <= 10 degrees; these are proposed engineering
targets, not assignment requirements or current guarantees. Measure the minimum
effective correction pulse, visual orientation noise, and tag visibility through
a quarter-turn before choosing the final heading tolerance. Unverified turns
must remain stopped rather than completing a lap using assumed headings.

Future build/run commands, available only after implementation:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_testing
source install/setup.bash
# start the existing calibrated camera and AprilTag detector in separate terminals
ros2 launch robot_testing hw3.launch.py side_length_m:=2.1336
```

No robot code has been run or hardware behavior validated by writing this plan.
