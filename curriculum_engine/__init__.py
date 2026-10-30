"""Curriculum Intelligence Engine (Group 9): overlap, redundancy and missing-prerequisite analysis of NMIMS syllabi."""
import os
from pathlib import Path

__version__ = "1.0.0"
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CE_DATA_DIR", ROOT / "data" / "curriculum"))   # vendored: lives under this project's data folder
RESULTS = Path(__file__).resolve().parent / "results"              # vendored: only the tuned thresholds and measured precisions
SEED = 42
