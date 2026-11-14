#!/usr/bin/env python3
"""Run all HW3 synthetic unit tests."""

import sys
import unittest
from pathlib import Path

# load only the inner hw3 package; no robot_testing installation is required
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "hw3"))

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(str(repo_root / 'hw3/test'), pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
