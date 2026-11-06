#!/usr/bin/env bash
set -euo pipefail

# build the nested package explicitly, independent of parent package discovery
hw3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$hw3_root"
colcon build --paths . --packages-select hw3_code "$@"
