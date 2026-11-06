# HW3 component checks

The isolated route simulator and its simulator-dependent tests have been removed.
Remaining tests verify filter mathematics and live-pipeline components, without
launching ROS nodes or commanding hardware. NumPy and OpenCV are required.

From the Git repository root:

```bash
PYTHONPATH=hw3 python3 -m unittest discover -s hw3/test -p 'test_*.py' -v
```

Checks cover EKF Jacobians, covariance/cross-correlations, landmark discovery and
reobservations, raw distorted-image PnP, optical-to-robot axes, command timestamp
boundaries/expiry, feet-to-meters route geometry, and recorded SVG/CSV map exports.
Synthetic projected corners are geometry fixtures, not hardware calibration.

ROS topic/QoS integration, camera timing, actual motor response, and physical
SLAM accuracy require on-robot validation. No standalone route simulation remains.
