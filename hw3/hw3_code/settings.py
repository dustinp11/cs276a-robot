"""Default HW3 parameters; noise and motor values need on-robot characterization."""

DEFAULTS = {
    'square_side_ft': 7.0, 'laps': 1, 'tag_size_m': 0.172,
    'web_host': '0.0.0.0', 'web_port': 8080,
    'auto_start': True, 'frame_id': 'hw3_map', 'output_directory': '~/hw3_runs',
    'camera_xyz_m': [0.0, 0.0, 0.2], 'camera_rpy_deg': [0.0, 0.0, 0.0],
    'image_width': 1280, 'image_height': 720,
    'range_std_m': 0.03, 'range_scale': 0.02, 'bearing_std_rad': 0.025,
    'max_reprojection_px': 3.0, 'min_decision_margin': 20.0, 'max_hamming': 1,
    'max_range_m': 5.0, 'gate_threshold': 9.210340371976184,
    'distance_noise': 0.15, 'turn_noise': 0.20,
    'idle_distance_noise': 0.002, 'idle_turn_noise': 0.005,
    'vision_delay_s': 0.35,
    'motion_timeout_s': 0.20, 'command_hold_s': 0.15,
    'max_speed_m_s': 0.567333, 'max_turn_rad_s': 0.8,
    'startup_wait_s': 2.0, 'left_turn_time_s': 0.48, 'corner_settle_s': 1.0,
    'max_run_s': 900.0,
}


def validate_parameters(values):
    """Reject invalid geometry, timing, and noise settings before commanding motion."""
    import math
    for key, value in values.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool) and key != 'max_hamming':
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
    for key in ('laps', 'web_port', 'image_width', 'image_height'):
        if type(values[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if type(values['max_hamming']) is not int or not 0 <= values['max_hamming'] <= 3:
        raise ValueError('max_hamming must be an integer from 0 to 3')
    for key in ('camera_xyz_m', 'camera_rpy_deg'):
        if len(values[key]) != 3 or not all(math.isfinite(v) for v in values[key]):
            raise ValueError(f'{key} must contain three finite values')

    if values['web_port'] > 65535:
        raise ValueError('web_port must be at most 65535')
