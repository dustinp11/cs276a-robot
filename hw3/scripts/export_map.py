#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in hw3.scripts.export_map."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw3.scripts.export_map import main


if __name__ == '__main__':
    main()
