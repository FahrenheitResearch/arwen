"""WOOF's shipped legacy RRTMG path against WRF v4.6.1's Fortran, bit for bit.

tools/rrtmg_legacy_column_oracle runs the UNMODIFIED WRF v4.6.1
RRTMG_LWRAD / RRTMG_SWRAD (phys/module_ra_rrtmg_lw.F, _sw.F) the way
module_radiation_driver.F drives them on one radiation step -- phy_prep's
t8w, radconst + calc_coszen at the interval midpoint, cal_cldfra1, the CAM
ozone chain (oznini, ozn_time_int, ozn_p_int) or the wrapper's O3DATA, the
aer_opt = 3 optics, then RTHRATEN, GLW, OLR, GSW, SWDOWN, SWDDIR, SWDDIF --
compiled with gfortran 13.3.0 -O0 and no contraction against glibc 2.39, on
128 columns (convective, tropical and marine cloud, stable night, snow and
sea ice, Antarctic polar night, high terrain, grazing sun on both sides of
the terminator, 22 edge cases) under nine regimes: Thompson (mp 8, CAM
ozone, wrapper ozone, p_top 2 kPa), WSM6, Morrison (wrapper radii), P3
(no snow species), Kessler (warm rain), Thompson aerosol-aware with
aer_opt 0 and 3.  The fixture in tests/data/oracles/rrtmg_legacy/column
holds the inputs and WRF's outputs.

Here the forecast's own adapter, ``RRTMGLegacyRadiation.__call__``, runs on
the device on the same inputs (device t8w, cal_cldfra1, ozone, aerosol,
McICA, batched CUDA LW/SW engines, output maps) and every graded field must
carry WRF's float32 bit pattern.

aer_opt = 3 is graded against WRF v4.6.1's RRTMG_SWRAD fed the operational
HRRR fork's gt_aod + calc_aerosol_rrtmg_sw (NOAA-EMC/HRRR v4.1.21), the
generation gpuwm.core.rrtmg_aerosol_optics implements.  Stock WRF v4.6.1's
own aer_opt = 3 optics differ from that generation in two lines of gt_aod
(RH capped at 99.1 instead of 98; each layer's 550 nm AOD floored at
1e-3); that difference is a declared generation, recorded in the lane
report, not graded here.

WRF's published RTHRATENSW is RTHRATEN - RTHRATENLW (driver :2866-2872),
not RRTMG_SWRAD's own heating rate; WOOF publishes the latter and its
theta tendency RTHRATENLW + RTHRATENSW equals WRF's RTHRATEN bit for bit
(graded as ``rthraten``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import HAS_GPU

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "oracles" / "rrtmg_legacy" / "column"
TOOLS = REPO / "tools" / "rrtmg_legacy_column_oracle"

#: WRF field -> WOOF output (tools/rrtmg_legacy_column_oracle/compare.py).
GRADED = {"rthratenlw": "rthratenlw", "rthratensw_wrapper": "rthratensw",
          "rthraten": "rthraten", "glw": "glw", "olr": "olr", "gsw": "gsw",
          "swdown": "swdown", "coszen": "coszen", "cldfra": "cldfra",
          "swddir": "swddir", "swddif": "swddif", "o3rad": "o3rad"}


def _meta():
    return json.loads((FIXTURE / "regimes.json").read_text())


REGIMES = sorted(_meta()["regimes"])


def gpu_gate(fn):
    return pytest.mark.gpu(
        pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")(fn))


def _ordered(a):
    i = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def bit_report(wrf, woof):
    """(differing values, max ULP) on the float32 bit patterns."""
    wrf = np.ascontiguousarray(wrf, np.float32)
    woof = np.ascontiguousarray(woof, np.float32)
    assert wrf.shape == woof.shape, (wrf.shape, woof.shape)
    differ = int((wrf.view(np.uint32) != woof.view(np.uint32)).sum())
    return differ, int(np.abs(_ordered(wrf) - _ordered(woof)).max())


@pytest.fixture(scope="module")
def wrf():
    return dict(np.load(FIXTURE / "wrf461.npz"))


@pytest.fixture(scope="module")
def run_regime():
    sys.path.insert(0, str(TOOLS))
    from run_woof import run_regime as run
    return run


def test_fixture_covers_every_regime_and_field(wrf):
    """The fixture is what the docstring says: nine regimes, 128 columns,
    both day and night columns, every graded field present."""
    meta = _meta()
    assert len(meta["regimes"]) == 9 and meta["ncol"] == 128
    for name, regime in meta["regimes"].items():
        for field in GRADED:
            if field == "o3rad" and regime["o3input"] != 2:
                continue
            assert f"{name}/{field}" in wrf, (name, field)
        coszen = wrf[f"{name}/coszen"]
        assert (coszen > 0).sum() >= 60 and (coszen <= 0).sum() >= 30
    prov = json.loads((FIXTURE / "provenance.json").read_text())
    assert prov["arms"]["thompson28_aer3"] == "fork"
    assert all(arm == "stock" for name, arm in prov["arms"].items()
               if name != "thompson28_aer3")


def test_the_bit_comparison_can_fail(wrf):
    """A one-ULP change in one value is caught: the gate is not vacuous."""
    a = wrf["thompson8/rthratenlw"]
    b = a.copy()
    b.view(np.uint32)[3, 7] += 1
    assert bit_report(a, b) == (1, 1)
    assert bit_report(a, a) == (0, 0)


@gpu_gate
@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.skipif(not __import__("gpuwm.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="WRF word-for-word oracle of the strict (GPUWM_WRF_EXACT=1) path; a default process runs 2.9.1's arithmetic")
def test_adapter_matches_wrf461_bit_for_bit(regime, wrf, run_regime):
    import cupy  # noqa: F401  -- the adapter runs on the device
    meta = _meta()
    woof = run_regime(FIXTURE, meta, regime)
    misses = {}
    for wk, ok in GRADED.items():
        key = f"{regime}/{wk}"
        if wk == "o3rad" and meta["regimes"][regime]["o3input"] != 2:
            continue    # o3input = 0: the wrapper builds O3DATA itself
        assert key in wrf, key
        assert ok in woof, (regime, ok)
        differ, max_ulp = bit_report(wrf[key], woof[ok])
        if differ:
            misses[wk] = (differ, max_ulp)
    assert not misses, f"{regime}: (differing values, max ULP) {misses}"
