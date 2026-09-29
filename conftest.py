"""Make the app package importable when pytest is launched at the repo root."""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).parent / "fnd drc project"
sys.path.insert(0, str(PROJECT_DIR))
