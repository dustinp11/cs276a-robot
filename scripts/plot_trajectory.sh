#!/usr/bin/env bash
# Plot HW4 actual robot trajectory vs planned safe Voronoi path
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
echo "HW4 TRAJECTORY PLOTTER: PLANNED PATH VS ACTUAL TRAJECTORY"
echo "============================================================"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" "$PYTHON_EXEC" -m robot_testing.hw4_visualization "$@"
echo "Plot complete. Image saved to hw4_trajectory_comparison.png"
