"""Complete retained WRF-word replay for the active verification diffusion group."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    os.environ.get("GPUWM_WRF_EXACT") != "1"
    or os.environ.get("GPUWM_WRF_EXACT_DIFFUSION", "1") == "0",
    reason="requires a WRF-exact process with the diffusion control on",
)
def test_active_diffusion_matches_every_retained_compiled_wrf_word(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "tools/wrf_exact_diffusion_oracle.py"),
         "--fixtures", str(root / "tests/data/wrf471_diffusion"),
         "--oracle-tools", str(root / "tools/wrf_diffusion_oracle"),
         "--out", str(tmp_path)],
        cwd=root, check=True,
    )
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert len(receipt["deformation"]) == len(receipt["horizontal"]) == 14
    metrics = [m for fields in receipt["summary"].values() for m in fields.values()]
    assert sum(m["words"] for m in metrics) == 2_722_902
    assert all(m["different_words"] == m["max_ulp"] == 0 for m in metrics)
