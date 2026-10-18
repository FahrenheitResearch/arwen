"""ra_lw/ra_sw_physics = 90: the COSZEN carrier is WRF 4.6.1's, bit for bit.

WRF 4.6.1 has no analytic clear-sky scheme, so of the adapter's outputs only
COSZEN has a WRF definition: radconst + calc_coszen on the radiation call,
at ``xtime + radt*0.5`` (module_radiation_driver.F:1188-1189, :1206-1208).
The fixture ``tests/data/oracles/analytic_clear_sky/wrf461_coszen.npz`` is
WRF's own routines, compiled unmodified at -O0 with gfortran 13.3.0 and fed
by WRF's own time library, over 82 columns and 14 calendar cases
(tools/analytic_clear_sky_oracle; make_columns.py, build.sh,
compare.py fixture).

The defect these tests hold fixed: the adapter took the cosine from a
float64 evaluation at ``valid_time`` plus half of the dt-ROUNDED radiation
cadence (and half a time step when radt was 0), so it missed WRF by up to
2.1e-3 (radt=0), 1.2e-3 (radt=15 min, dt=72 s) and by a few float32
spacings everywhere else; 191 of 1,148 words matched.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.wrf_radiation_clock import (
    radiation_call_scalars, wrf_calendar_scalars)

F = np.float32
FIXTURE = (Path(__file__).parent / "data" / "oracles" / "analytic_clear_sky"
           / "wrf461_coszen.npz")


@pytest.fixture(scope="module")
def oracle():
    return dict(np.load(FIXTURE))


def _cases(o):
    for k, label in enumerate(o["case_labels"]):
        yield (k, str(label), datetime(*[int(v) for v in o["starts"][k]]),
               int(o["elapsed"][k]), float(o["radt"][k]), float(o["dt"][k]))


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def test_fixture_spans_the_declared_regimes(oracle):
    assert oracle["lat"].size >= 64
    regimes = set(oracle["regimes"].tolist())
    assert {"convective", "stable", "snow-ice", "ocean", "high-terrain",
            "edge"} <= regimes
    cz = oracle["wrf_coszen"]
    assert (cz > 0).any() and (cz < 0).any()
    # both sides of radconst's JULIAN = 80 branch
    assert (oracle["wrf_julian"] < 80).any() and (oracle["wrf_julian"] >= 80).any()


def test_calendar_scalars_are_wrfs_frame_words(oracle):
    """julian, xtime and gmt equal what WRF's frame and time library made."""
    for k, label, start, elapsed, _radt, _dt in _cases(oracle):
        julian, xtime, gmt = wrf_calendar_scalars(start, elapsed)
        for name, got in (("julian", julian), ("xtime", xtime), ("gmt", gmt)):
            want = oracle[f"wrf_{name}"][k]
            assert _bits(got) == _bits(want), (label, name, got, want)


def test_julian_rounds_at_day_plus_one_as_wrf_does():
    """REAL(dayOfYear_r8) - 1.0 is not REAL(day) on the first day of the
    year: 00:21 on 1 January is 0.014583349 in WRF, not 0.014583333."""
    julian, _, _ = wrf_calendar_scalars(datetime(2024, 1, 1), 1260.0)
    assert julian == F(F(1260.0 / 86400.0 + 1.0) - F(1.0))
    assert julian != F(1260.0 / 86400.0)


def test_half_interval_uses_the_namelist_radt_raw():
    """WRF adds radt*0.5 to xtime whatever NINT(radt*60/dt) rounds the
    call cadence to, and adds nothing when radt is 0."""
    start = datetime(2026, 7, 4, 18)
    _, xtime, _ = radiation_call_scalars(start, 5400.0, 15.0)
    assert xtime == F(97.5)          # not 90 + 936 s / 2 / 60 (dt = 72 s)
    _, xtime, _ = radiation_call_scalars(start, 600.0, 0.0)
    assert xtime == F(10.0)          # not 10 + dt/2


def _adapter_coszen(o, k, start, elapsed, radt, dt, *, composed):
    import cupy as cp
    from gpuwm.core.analytic_radiation import AnalyticClearSkyRadiation
    from gpuwm.core.radiation_composition import ComposedRadiation

    ncol = o["lat"].size
    lat = cp.asarray(o["lat"].reshape(1, ncol))
    lon = cp.asarray(o["lon"].reshape(1, ncol))
    level = cp.full((2, 1, ncol), F(290.0), cp.float32)
    atmosphere = {"theta": level, "pressure": cp.full_like(level, F(95000.0)),
                  "qv": cp.full_like(level, F(0.008))}
    fields = {"albedo": cp.full((1, ncol), F(0.2), cp.float32),
              "glw": cp.zeros((1, ncol), cp.float32)}
    # The referee is WRF v4.6.1: its sun (2.9.1 defaults to wrf_480).
    cfg = SimpleNamespace(radt=radt, radt_minutes=radt, dt=dt, clock_dt=0.0,
                          use_adaptive_time_step=False, swint_opt=0,
                          solar_generation="wrf_461")
    adapter = AnalyticClearSkyRadiation(start, lat, lon)
    if composed:
        adapter = ComposedRadiation(start, lat, lon, longwave_adapter=adapter)
    result = adapter(atmosphere=atmosphere, fields=fields,
                     state=SimpleNamespace(elapsed_seconds=float(elapsed)),
                     cfg=cfg)
    assert result.coszen.dtype == cp.float32
    return cp.asnumpy(result.coszen).reshape(-1)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("composed", [False, True],
                         ids=["ra90-adapter", "ra-lw90-sw0-composition"])
@pytest.mark.skipif(not __import__("gpuwm.wrf_exact", fromlist=["ENABLED"]).ENABLED,
                    reason="WRF word-for-word oracle of the strict (GPUWM_WRF_EXACT=1) path; a default process runs 2.9.1's arithmetic")
def test_coszen_is_wrf461_word_for_word(oracle, composed):
    words = 0
    for k, label, start, elapsed, radt, dt in _cases(oracle):
        got = _adapter_coszen(oracle, k, start, elapsed, radt, dt,
                              composed=composed)
        want = oracle["wrf_coszen"][k]
        bad = np.nonzero(_bits(got) != _bits(want))[0]
        assert bad.size == 0, (label, bad[:8].tolist(),
                               got[bad[:4]].tolist(), want[bad[:4]].tolist())
        words += got.size
    assert words == oracle["wrf_coszen"].size
