"""Compatibility CLI for family sheets; implementation lives in heat_eval.sheets.

Prefer python -m tools.heat_eval.sheets --receipt ... --renderer ... --out ...
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.heat_eval.sheets import draw_family_panels as render_panels, main

if __name__ == "__main__":
    raise SystemExit(main())
