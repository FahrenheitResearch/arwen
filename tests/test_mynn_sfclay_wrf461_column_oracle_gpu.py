"""WOOF's MYNN surface layer against WRF 4.6.1's SFCLAY1D_mynn, 0 ULP.

The fixture (tests/data/oracles/mynn/sfclay-columns-wrf461) is the
unmodified WRF v4.6.1 module_sf_mynn.F compiled by gfortran 13.3.0 at -O2 on
glibc 2.39 (x86-64 with FMA, so the ifunc libm words), driven by
tools/mynn_sfclay_wrf461_column_oracle over 110 columns -- convective, stable
night, snow and sea ice, open water from calm to hurricane force, high
terrain, thin and coarse first layers, and the zolrib non-convergent fallback
-- through eleven option sets (isftcflx 0-3, isfflx 0/1, three grid lengths,
seeded first steps and restarts, and three with WRF's stochastic roughness,
spp_pbl=1, under a fixed pattern) and 2-3 successive timesteps each.

Every one of the 35 outputs must be bitwise equal, in the arithmetic mode
the process runs in: the strict build (GPUWM_WRF_EXACT=1) and default
arithmetic were both measured at 0 ULP on the RTX PRO 6000 (sm_120, NVRTC
12.9) on 2026-10-07, 244,860 values.  Before the fix the strict build
missed 28,153 of the 181,720 values of the eight spp_pbl=0 option sets by up
to 423 ULP.  Replay enters each call with WRF's own
state; free-running carries WOOF's own state and runs WOOF's first-step
seeding, so a drift anywhere in the chain fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import requires_gpu

# The device work happens inside tools/.../compare.py, which this module does
# not import at top level, so conftest's cupy-import detector cannot see it:
# mark the module explicitly so `-m "not gpu"` never opens a device here.
pytestmark = pytest.mark.gpu

FIXTURE = (Path(__file__).resolve().parents[1] / "tests" / "data" / "oracles"
           / "mynn" / "sfclay-columns-wrf461")


@requires_gpu
def test_mynn_surface_kernel_matches_wrf461_bitwise():
    from tools.mynn_sfclay_wrf461_column_oracle.columns import CONFIGS
    from tools.mynn_sfclay_wrf461_column_oracle.compare import (
        coverage, run, summarize)

    receipt = run(FIXTURE / "oracle-o2.bin.gz", FIXTURE)
    summary = summarize(receipt)
    bad = {name: row for name, row in summary["per_field"].items()
           if row["mismatched"]}
    assert summary["compared_values"] > 180_000
    assert summary["mismatched_values"] == 0, bad
    assert len(receipt["seed"]) == sum(1 for c in CONFIGS if c[5])
    assert {c[7] for c in CONFIGS} == {0, 1}
    cover = coverage(FIXTURE / "oracle-o2.bin.gz", FIXTURE)
    assert all(cover["regime"].get(str(r), 0) > 0 for r in (1, 2, 3, 4))
    assert cover["zol_abs_gt_10"] > 0 and cover["hfx_floor"] > 0
