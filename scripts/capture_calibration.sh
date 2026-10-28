#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.bash"

source_vision
exec python3 "$repo_root/calibration/capture_ros.py" "$@"
