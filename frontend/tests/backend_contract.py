"""Compatibility entry point for the current backend regression suite (no live AI)."""
from pathlib import Path
import sys
import unittest

if __name__ == '__main__':
    backend = Path(__file__).resolve().parents[2] / 'backend'
    sys.path.insert(0, str(backend))
    suite = unittest.defaultTestLoader.discover(str(backend / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
