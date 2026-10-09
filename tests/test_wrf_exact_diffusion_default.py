"""Production uses WRF diffusion order; strict retains its attribution opt-out."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = (
    "import importlib.util, json, sys\n"
    "spec = importlib.util.spec_from_file_location('wrf_exact_probe', sys.argv[1])\n"
    "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
    "print(json.dumps([m.ENABLED, m.DIFFUSION_ENABLED, list(m.STRICT_OPTIONS)]))\n"
)


def _selection(**env):
    clean = {k: v for k, v in __import__("os").environ.items() if not k.startswith("GPUWM_WRF_EXACT")}
    clean.update(env)
    out = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "gpuwm/wrf_exact.py")],
                         env=clean, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


@pytest.mark.parametrize("env,enabled,diffusion", [
    ({}, False, True),
    ({"GPUWM_WRF_EXACT": "0"}, False, True),
    ({"GPUWM_WRF_EXACT_DIFFUSION": "1"}, False, True),
    ({"GPUWM_WRF_EXACT": "1"}, True, True),
    ({"GPUWM_WRF_EXACT": "1", "GPUWM_WRF_EXACT_DIFFUSION": "1"}, True, True),
    ({"GPUWM_WRF_EXACT": "1", "GPUWM_WRF_EXACT_DIFFUSION": "0"}, True, False),
])
def test_strict_build_selects_wrf_diffusion_order_by_default(env, enabled, diffusion):
    got_enabled, got_diffusion, options = _selection(**env)
    assert (got_enabled, got_diffusion) == (enabled, diffusion)
    assert ("-DGPUWM_WRF_EXACT_C_DIFFUSION=1" in options) is diffusion
