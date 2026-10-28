# Camera calibration

Standalone HW2 calibration scripts. Full work log, setup, capture and calibration commands,
output definitions, and hardware checks are in [hw2_agents.md](../../hw2_agents.md).
Start there. No movement modules are imported or modified.

For an easy periodic ROS camera capture, leave the camera launch running in one terminal and
run `./capture_calibration.sh` from the repository root in another. It saves 25 PNG images,
at least 5 seconds apart, under `calibration/images/`. Move the calibration board between
captures. Adjust the interval or count with, for example,
`./capture_calibration.sh --interval 8 --count 20`.

Calibrate the captured 7x7-inner-corner board and compare pinhole and fisheye models with:

```sh
python3 calibration/calibrate.py \
  --images calibration/images --columns 7 --rows 7 \
  --min-views 10 --output calibration/results/board_7x7
```

The output includes `corners/` overlays, `detections.json`, per-view errors in
`calibration.json`, a deployment-format pinhole `camera_parameter.yaml`, and
`model_comparison.json`. Set `--square-size-m` to the measured square side in meters if
metric board-to-camera translations are needed; otherwise translations are in board-square
units. The intrinsics are unaffected by that scale.
