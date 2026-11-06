"""Planar EKF-SLAM with an empty initial map and ID-associated landmarks.

State order is [robot_x, robot_y, yaw, landmark_1_x, landmark_1_y, ...].
Coordinates are meters in a local starting frame; angles are radians. Inputs
are traveled-distance/yaw increments and robot-frame range/bearing observations.
This module has no ROS dependencies and does not consume known map coordinates.
"""

import math

import numpy as np


def wrap_angle(angle):
    """Normalize an angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _vector(value, size, name):
    """Validate and copy a finite vector with the requested number of entries."""
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector of length {size}')
    return result.copy()


def _covariance(value, size, name, positive=False):
    """Validate a symmetric covariance; require positive definiteness if requested."""
    result = np.asarray(value, dtype=float)
    if result.shape != (size, size) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite {size} by {size} matrix')
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f'{name} must be symmetric')
    result = (result + result.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(result)
    if eigenvalues[0] < 0 or (positive and eigenvalues[0] <= 0):
        raise ValueError(f'{name} must be positive {"definite" if positive else "semidefinite"}')
    return result.copy()


def _motion_model(pose, distance, turn):
    """Return midpoint-motion pose and state/input Jacobians for one short increment."""
    heading = pose[2] + turn / 2.0
    c, s = math.cos(heading), math.sin(heading)
    predicted = pose + [distance * c, distance * s, turn]
    predicted[2] = wrap_angle(predicted[2])
    state_jacobian = np.array([[1., 0., -distance*s], [0., 1., distance*c], [0., 0., 1.]])
    input_jacobian = np.array([[c, -distance*s/2.], [s, distance*c/2.], [0., 1.]])
    return predicted, state_jacobian, input_jacobian


def _observation_model(state, index):
    """Return predicted range/bearing and its Jacobian over the joint state."""
    dx, dy = state[index:index+2] - state[:2]
    squared_range = dx*dx + dy*dy
    if squared_range <= 1e-12:
        raise ValueError('Range/bearing is undefined at the robot position')
    distance = math.sqrt(squared_range)
    predicted = np.array([distance, wrap_angle(math.atan2(dy, dx) - state[2])])
    jacobian = np.zeros((2, len(state)))
    jacobian[:, :3] = [[-dx/distance, -dy/distance, 0.], [dy/squared_range, -dx/squared_range, -1.]]
    jacobian[:, index:index+2] = [[dx/distance, dy/distance], [-dy/squared_range, dx/squared_range]]
    return predicted, jacobian


def _inverse_observation(pose, measurement):
    """Return new landmark coordinates and robot/measurement initialization Jacobians."""
    distance, bearing = measurement
    heading = pose[2] + bearing
    c, s = math.cos(heading), math.sin(heading)
    position = pose[:2] + [distance*c, distance*s]
    robot_jacobian = np.array([[1., 0., -distance*s], [0., 1., distance*c]])
    measurement_jacobian = np.array([[c, -distance*s], [s, distance*c]])
    return position, robot_jacobian, measurement_jacobian


class EkfSlam:
    """Estimate a robot pose and a dynamically growing planar landmark map.

    The initial robot pose is the origin of the local SLAM frame. A zero initial
    covariance fixes that coordinate reference; it does not supply a measured
    world pose. Optional initial covariance represents local initial uncertainty.
    Landmark keys can be tag IDs or (family, ID) tuples; their count is unrestricted.
    Callers must supply observations once each, in time order, at the predicted
    robot pose and transform camera measurements into the robot frame beforehand.
    """

    def __init__(self, initial_covariance=None, gate_threshold=9.210340371976184):
        """Start with an empty map and a squared Mahalanobis observation gate.

        Args:
            initial_covariance: Optional 3x3 covariance in m/rad coordinates.
            gate_threshold: Positive squared-distance limit (default: 99% for
                two Gaussian residual dimensions). This does not gate first sightings.
        """
        self._state = np.zeros(3)
        covariance = np.zeros((3, 3)) if initial_covariance is None else initial_covariance
        self._covariance = _covariance(covariance, 3, 'initial_covariance')
        self._indices = {}
        self.gate_threshold = float(gate_threshold)
        if not math.isfinite(self.gate_threshold) or self.gate_threshold <= 0:
            raise ValueError('gate_threshold must be finite and positive')

    @property
    def state(self):
        """Return a copy of the full state, ordered robot first then discovered landmarks."""
        return self._state.copy()

    @property
    def covariance(self):
        """Return a copy of the full joint covariance, including cross-correlations."""
        return self._covariance.copy()

    @property
    def robot_pose(self):
        """Return local robot [x (m), y (m), yaw (rad)]."""
        return self._state[:3].copy()

    @property
    def landmark_ids(self):
        """Return landmark keys in discovery/state order."""
        return tuple(self._indices)

    def landmark(self, landmark_id):
        """Return copies of a landmark's [x, y] position and 2x2 covariance."""
        index = self._indices[landmark_id]
        return (self._state[index:index+2].copy(),
                self._covariance[index:index+2, index:index+2].copy())

    def predict(self, distance_m, turn_rad, increment_covariance):
        """Propagate one short motion increment with a midpoint unicycle model.

        Args:
            distance_m: Signed traveled distance in meters, not a velocity.
            turn_rad: Signed yaw increment in radians, positive counterclockwise.
            increment_covariance: 2x2 covariance of [distance_m, turn_rad], including
                any correlation; units are m², m*rad, and rad². This is input noise,
                mapped into the joint state by the motion input Jacobian.

        Split large curved movements into short chronological increments. Landmark
        world coordinates stay fixed; robot/map cross-covariances still propagate.
        """
        distance, turn = _vector([distance_m, turn_rad], 2, 'motion increment')
        noise = _covariance(increment_covariance, 2, 'increment_covariance')
        pose, robot_jacobian, input_jacobian = _motion_model(self.robot_pose, distance, turn)
        state_jacobian = np.eye(len(self._state))
        state_jacobian[:3, :3] = robot_jacobian
        covariance = state_jacobian @ self._covariance @ state_jacobian.T
        covariance[:3, :3] += input_jacobian @ noise @ input_jacobian.T
        self._state[:3] = pose
        self._covariance = (covariance + covariance.T) / 2.0

    def observe(self, landmark_id, range_m, bearing_rad, measurement_covariance):
        """Initialize or correct a landmark from an ID-associated observation.

        Args:
            landmark_id: Hashable detector identity; no predeclared ID list is used.
            range_m: Positive planar robot-to-landmark range in meters.
            bearing_rad: Angle from robot forward, positive counterclockwise.
            measurement_covariance: Positive-definite 2x2 covariance of range and
                bearing, with units m², m*rad, and rad².

        Returns:
            str: 'initialized', 'updated', or 'rejected' (innovation outside gate).
                A first observation augments the state once, without a second update.

        No call is needed when a tag leaves view: its state/correlations are retained.
        Reappearance with the same ID updates the existing landmark. Invalid numeric
        input raises ValueError. Callers must validate new detections before insertion.
        """
        hash(landmark_id)
        measurement = _vector([range_m, bearing_rad], 2, 'observation')
        if measurement[0] <= 1e-6:
            raise ValueError('range_m must exceed 1e-6 meters')
        measurement[1] = wrap_angle(measurement[1])
        noise = _covariance(measurement_covariance, 2, 'measurement_covariance', positive=True)
        if landmark_id not in self._indices:
            self._initialize_landmark(landmark_id, measurement, noise)
            return 'initialized'
        return self._correct(self._indices[landmark_id], measurement, noise)

    def _initialize_landmark(self, landmark_id, measurement, noise):
        """Augment the joint state while retaining all initialization correlations."""
        position, robot_jacobian, measurement_jacobian = _inverse_observation(self.robot_pose, measurement)
        cross = robot_jacobian @ self._covariance[:3, :]
        landmark_covariance = robot_jacobian @ self._covariance[:3, :3] @ robot_jacobian.T
        landmark_covariance += measurement_jacobian @ noise @ measurement_jacobian.T
        covariance = np.block([[self._covariance, cross.T], [cross, landmark_covariance]])
        self._indices[landmark_id] = len(self._state)
        self._state = np.concatenate([self._state, position])
        self._covariance = (covariance + covariance.T) / 2.0

    def _correct(self, index, measurement, noise):
        """Gate an innovation, then correct all correlated robot/landmark variables."""
        predicted, jacobian = _observation_model(self._state, index)
        innovation = measurement - predicted
        innovation[1] = wrap_angle(innovation[1])
        cross = self._covariance @ jacobian.T
        residual_covariance = jacobian @ cross + noise
        distance_squared = float(innovation @ np.linalg.solve(residual_covariance, innovation))
        if distance_squared > self.gate_threshold:
            return 'rejected'
        gain = np.linalg.solve(residual_covariance, cross.T).T
        self._apply_correction(gain, jacobian, innovation, noise)
        return 'updated'

    def _apply_correction(self, gain, jacobian, innovation, noise):
        """Use Joseph covariance form to preserve symmetry and numerical stability."""
        self._state += gain @ innovation
        self._state[2] = wrap_angle(self._state[2])
        residual = np.eye(len(self._state)) - gain @ jacobian
        covariance = residual @ self._covariance @ residual.T + gain @ noise @ gain.T
        self._covariance = (covariance + covariance.T) / 2.0
