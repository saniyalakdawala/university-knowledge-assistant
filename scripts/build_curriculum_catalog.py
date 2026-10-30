"""Rebuild data/curriculum/srb_catalog.json from the SRB PDF:  python scripts/build_curriculum_catalog.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import DATA_DIR, SRB_PDF_PATH
from backend.curriculum.srb_catalog import save_catalog

(DATA_DIR / "curriculum").mkdir(parents=True, exist_ok=True)
p = save_catalog(SRB_PDF_PATH, str(DATA_DIR / "curriculum" / "srb_catalog.json"))
print(f"{len(p['programmes'])} programmes, {len(p['courses'])} course records, {len(p['departments'])} departments with outcomes")
