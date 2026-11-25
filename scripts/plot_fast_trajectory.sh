#!/usr/bin/env bash
# Plot HW4 Fast A* actual robot trajectory vs planned shortest path
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "============================================================"
echo "HW4 FAST A* TRAJECTORY PLOTTER: PLANNED PATH VS ACTUAL"
echo "============================================================"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_EXEC" -m robot_testing.hw4_fast_visualization "$@"
echo "Plot complete. Image saved to hw4_fast_path_visualization.png / hw4_fast_trajectory_comparison.png"
