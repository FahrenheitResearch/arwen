"""km_opt=3 (3-D Smagorinsky) against compiled WRF 4.6.1, word for word.

The column oracle (tools/wrf461_km3_oracle) compiles WRF 4.6.1's own
cal_deform_and_div, calculate_N2, smag_km, horizontal_diffusion_2 and
vertical_diffusion_2 (gfortran 13.3, -O0, no contraction, glibc 2.39 libm)
and records every output word for 26 states: the 14 retained real-state
crops plus 12 LES-scale regimes and edge cases, 4,128 columns, each run with
mix_isotropic 0 and 1 and with isfflx 0, 1 and 2.  This file holds the
committed seal of that capture: the synthetic input states and the SHA-256
of every WRF output array.

The GPU gate runs the production launchers under the strict WRF-arithmetic
diffusion stage (GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_DIFFUSION=1) and requires
every engine array to carry the WRF array's digest: deformation tensors,
BN2, the four exchange coefficients, the horizontal tendencies built from
the engine's own coefficients, and the vertical driver's tendencies and
surface outputs.

One declared difference is asserted, not hidden: WRF hands
vertical_diffusion_w_2 the horizontal coefficient xkmh, the engine hands it
xkmv (bbdc11150; dycore.launch_wrf_smag2d_vertical).  The vertical w words
must equal WRF's run with xkmv in that slot (``wcontrol``) and must differ
from stock WRF exactly where the capture recorded xkmh != xkmv.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/wrf461_km3"
TOOLS = ROOT / "tools/wrf461_km3_oracle"
RETAINED = ROOT / "tests/data/wrf471_diffusion"

#: digests.json as sealed from the W1 capture (tools/wrf461_km3_oracle/seal.py).
DIGESTS_SHA256 = "240c8a652e37530ed1c2d783af3dc39c80df5af3f37596e2cf1cb9d6ba3c2e9c"


@lru_cache(maxsize=None)
def _tool(name):
    for path in (TOOLS, ROOT / "tools/wrf_diffusion_oracle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(f"km3_oracle_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digests():
    body = (DATA / "digests.json").read_bytes()
    assert hashlib.sha256(body).hexdigest() == DIGESTS_SHA256, "digests.json changed without a re-seal"
    return json.loads(body)


@lru_cache(maxsize=1)
def _retained():
    return {n: a for n, a, _m in _tool("km3_cases").retained_cases(RETAINED)}


def _inputs(name, row):
    """The exact input state the capture used for one case."""
    seal = _tool("seal")
    if "inputs_file" in row:
        path = DATA / row["inputs_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["inputs_file_sha256"]
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
    else:
        arrays = _retained()[name]
    assert seal.inputs_sha256(arrays) == row["inputs_sha256"], name
    return arrays


def test_seal_covers_the_stated_columns_and_regimes():
    sealed = _digests()
    cases = sealed["cases"]
    assert len(cases) == 26 and sealed["columns"] == 4128
    assert sum(int(r["meta"]["nx"]) * int(r["meta"]["ny"]) for r in cases.values()) == 4128
    synthetic = {n for n, r in cases.items() if r["meta"]["family"] == "synthetic-les"}
    assert synthetic == {"les_cbl", "les_shallow_cumulus", "les_stable_night", "les_snow_ice",
                         "les_marine_sc", "les_high_terrain", "les_vortex_caps", "les_map_hilat",
                         "edge_zero_flow", "edge_tiny_flow", "edge_supersaturated",
                         "edge_qc_threshold"}
    assert sealed["build"]["wrf_release"] == "4.6.1"
    assert sealed["build"]["source_sha256"]["dyn_em/module_diffusion_em.F"] == (
        "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54")
    assert "-ffp-contract=off" in sealed["build"]["flags"] and not sealed["build"]["links_libmvec"]
    # 12 tensor/coefficient keys per iso pair, 7 horizontal, 2 x 3 x 9 vertical.
    for row in cases.values():
        assert len(row["wrf_sha256"]) == 7 + 2 * (5 + 7 + 2 * 3 * 9)


def test_every_sealed_input_state_is_reproduced():
    for name, row in _digests()["cases"].items():
        _inputs(name, row)


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("GPUWM_WRF_EXACT") != "1"
                    or os.environ.get("GPUWM_WRF_EXACT_DIFFUSION") != "1",
                    reason="the bitwise gate needs the strict WRF-arithmetic diffusion stage, "
                           "selected before import")
def test_km3_chain_matches_every_compiled_wrf461_word():
    import cupy  # noqa: F401  (device test: the conftest marker reads this import)
    compare = _tool("compare")
    sealed = _digests()
    moved, wdiv = [], []
    for name, row in sealed["cases"].items():
        arrays = _inputs(name, row)
        meta = row["meta"]
        wrf = row["wrf_sha256"]
        for iso in (0, 1):
            got = compare.engine_case(arrays, meta, iso)
            for key, value in got.items():
                arm, field = key.split("__")
                digest = hashlib.sha256(np.ascontiguousarray(value, np.float32).tobytes()).hexdigest()
                if arm == "coef":
                    expected = wrf[compare.reference_key(arm, field, iso)]
                elif arm == "h_chain":
                    expected = wrf[f"wrf__iso{iso}__h_{field}"]
                else:
                    flux = arm[-1]
                    expected = wrf[f"wrf__iso{iso}__v{flux}_wcontrol_{field}"]
                    stock = wrf[f"wrf__iso{iso}__v{flux}_stock_{field}"]
                    if field == "w":
                        differs = row["stock_w_differs"][f"iso{iso}_v{flux}"] > 0
                        if (digest != stock) != differs:
                            wdiv.append((name, iso, flux))
                    else:
                        assert stock == expected, (name, key)
                if digest != expected:
                    moved.append((name, iso, key))
    assert not moved, f"{len(moved)} engine arrays differ from compiled WRF 4.6.1: {moved[:20]}"
    assert not wdiv, f"vertical w departs from the declared xkmv choice: {wdiv}"
