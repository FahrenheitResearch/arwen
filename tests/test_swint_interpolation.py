"""WRF swint_opt = 1 (gpuwm.core.swint): twins, kernels and the fork oracle.

The oracle is the operational HRRR fork's own Fortran
(NOAA-EMC/HRRR v4.1.21, module_radiation_driver.F radconst, calc_coszen,
update_swinterp_parameters and interp_sw_radiation, extracted verbatim and
compiled with gfortran -O0 -ffp-contract=off against glibc by
tools/hrrr_radiation_driver_oracle), stored in
tests/data/swint_oracle/swint_oracle.npz.  Three layers:

1. the NumPy float32 twins of the fit and the evaluation against the fork,
   word for word (the twins call glibc's own logf/powf through
   gpuwm.core.noahmp_libm);
2. the three CUDA kernels against the fork, word for word (GPU): the fit,
   the evaluation and the per-step zenith cosine, which the kernel forms
   with glibc's sinf/cosf/asinf from the calendar scalars alone;
3. branch coverage the fixture was built to reach, asserted on the twins
   directly (fresh column, clamped exponents, night either side, the
   derived SWDDIF/SWDDNI/GSW), and the driver-facing carrier on a day/night
   strip.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import swint

F = np.float32
REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "swint_oracle" / "swint_oracle.npz"
FIXTURE_SHA256 = REPO / "tests" / "data" / "swint_oracle" / "SHA256"

gpu = pytest.mark.gpu

FLUXES = ("swdown", "swddir", "swddni", "swddif", "gsw")
COEFS = ("bb", "bx", "gg", "gx", "coszen_ref", "swdown_ref", "swddir_ref")


def bits_equal(name, got, want):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    differ = got.view(np.uint32) != want.view(np.uint32)
    assert not differ.any(), (
        f"{name}: {int(differ.sum())} of {got.size} words differ; first "
        f"got {got[differ][:4]} want {want[differ][:4]}")
    return got.size


@pytest.fixture(scope="module")
def oracle():
    assert FIXTURE.is_file(), (
        "tests/data/swint_oracle/swint_oracle.npz is cut by "
        "tools/hrrr_radiation_driver_oracle/run_swint_oracle.py from the "
        "fork's Fortran; the fixture is part of the tree")
    want = FIXTURE_SHA256.read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert got == want, f"swint oracle fixture sha256 {got} != pinned {want}"
    data = np.load(FIXTURE)
    receipt = json.loads(str(data["receipt"]))
    assert receipt["commit"] == "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"
    return {k: data[k] for k in data.files if k != "receipt"}


def _replay_twins(o):
    """The twins over the fixture's calls, in the fixture's layout."""
    cz = o["in/coszen"]
    ncall, ncol = cz.shape
    nloc = o["in/czloc"].shape[1]
    state = {name: np.zeros(ncol, np.float32) for name in
             ("bb", "bx", "gg", "gx", "coszen_ref", "swdown_ref",
              "swddir_ref")}
    out = {name: np.zeros((ncall, ncol), np.float32) for name in COEFS}
    flux = {name: np.zeros((ncall, 1 + nloc, ncol), np.float32)
            for name in FLUXES}
    for ic in range(ncall):
        swint.update_swinterp_parameters(
            cz[ic], o["in/czcall"][ic], o["in/swddir"][ic],
            o["in/swdown"][ic], state["swddir_ref"], state["bb"],
            state["bx"], state["swdown_ref"], state["gg"], state["gx"],
            state["coszen_ref"])
        for name in COEFS:
            out[name][ic] = state[name]
        suns = [o["in/czcall"][ic]] + [o["in/czloc"][ic, il]
                                       for il in range(nloc)]
        for il, czl in enumerate(suns):
            res = swint.interp_sw_radiation(
                state["coszen_ref"], czl, state["swddir_ref"], state["bb"],
                state["bx"], state["swdown_ref"], state["gg"], state["gx"],
                o["in/albedo"])
            for name, arr in zip(FLUXES, res):
                flux[name][ic, il] = arr
    return out, flux


def test_fixture_receipt_names_the_fork_and_the_flags(oracle):
    data = np.load(FIXTURE)
    receipt = json.loads(str(data["receipt"]))
    assert receipt["tag"] == "v4.1.21"
    assert receipt["sources"]["module_radiation_driver.F"] == (
        "7464639e53f6f40b4810943ee9fa21a40b0f53f7525c08cc13b84ae4c7d82801")
    assert "-ffp-contract=off" in receipt["flags"]
    assert "-O0" in receipt["flags"]


def test_fixture_reaches_every_branch(oracle):
    """The column set exercises what the routines branch on: clamped
    exponents at both ends, exponent zero (ratio one), night on either
    side of a call, fresh and stored references."""
    o = oracle
    bb, gg = o["out/bb"], o["out/gg"]
    assert (bb == F(2.5)).any() and (bb == F(-0.5)).any()
    assert (gg == F(2.5)).any() and (gg == F(-0.5)).any()
    interior = (bb > F(-0.5)) & (bb < F(2.5)) & (bb != F(0.0))
    assert interior.sum() > 1000
    night = ~((o["in/coszen"] > swint.COSZEN_MIN)
              & (o["in/czcall"] > swint.COSZEN_MIN))
    assert night.any() and (~night).any()
    assert (o["out/swdown"] == F(0.0)).any()


def test_twins_match_the_fork_word_for_word(oracle):
    o = oracle
    out, flux = _replay_twins(o)
    words = 0
    for name in COEFS:
        words += bits_equal(name, out[name], o[f"out/{name}"])
    for name in FLUXES:
        words += bits_equal(name, flux[name], o[f"out/{name}"])
    print(f"swint twins vs fork: {words} words, max ULP 0")


def test_coszen_twin_is_the_fork_shape(oracle):
    """The NumPy cosine twin differs from the fork only by NumPy's
    transcendentals (dossier section 9.1's seam): within a few float32
    spacings of the order-one terms it sums."""
    o = oracle
    sets = o["coszen_in/sets"]
    worst = 0.0
    for s, (julian, xtime, gmt) in enumerate(sets):
        got = swint.coszen_loc_host(o["coszen_in/xlat"], o["coszen_in/xlon"],
                                    julian, xtime, gmt)
        want = o["coszen_out/coszen"][s]
        worst = max(worst, float(np.max(np.abs(
            got.astype(np.float64) - want.astype(np.float64)))))
    assert worst <= 2.0e-6, worst


def test_calendar_scalars_are_the_adapter_statements():
    """julian/xtime/gmt are formed exactly as the legacy adapter forms
    them for its radiation call, at the step's own elapsed time."""
    start = datetime(2026, 7, 4, 18, 30)
    julian, xtime, gmt = swint.calendar_scalars(start, 5400.0)
    assert julian == F((185 - 1) + 20.0 / 24.0)
    assert xtime == F(90.0)
    assert gmt == F(18.5)
    assert all(isinstance(v, np.float32) for v in (julian, xtime, gmt))


def test_branches_fresh_column_clamp_night_and_derived_fluxes():
    n = 6
    cz = np.array([0.8, 0.8, 0.8, 0.8, 0.0, 0.8], F)
    czl = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.0], F)
    swddir = np.full(n, F(600.0))
    swdown = np.full(n, F(800.0))
    zeros = lambda: np.zeros(n, F)
    bb, bx, gg, gx = zeros(), zeros(), zeros(), zeros()
    cref, dref, bref = zeros(), zeros(), zeros()
    # Column 1 starts with a stored reference that makes the exponent clamp
    # high (tiny reference flux at nearly the same sun); column 2 one that
    # clamps low (huge reference flux); column 3 a stored reference with
    # ratio exactly 1 (exponent 0).
    bx[1] = gx[1] = F(1.0)
    bref[1] = dref[1] = F(1.0)
    cref[1] = F(0.79)
    bx[2] = gx[2] = F(1.0)
    bref[2] = dref[2] = F(5000.0)
    cref[2] = F(0.79)
    bx[3] = gx[3] = F(1.0)
    bref[3] = dref[3] = F(300.0)
    cref[3] = F(0.8)
    swint.update_swinterp_parameters(cz, czl, swddir, swdown, bref, bb, bx,
                                     dref, gg, gx, cref)
    # fresh column 0: linear first guess -> exponent one (within the
    # 1 - 1e-4 floor the log sees)
    assert abs(float(bb[0]) - 1.0) < 2e-3 and abs(float(gg[0]) - 1.0) < 2e-3
    assert bb[1] == F(2.5) and gg[1] == F(2.5)
    assert bb[2] == F(-0.5) and gg[2] == F(-0.5)
    assert bb[3] == F(0.0) and gg[3] == F(0.0) and bx[3] == swddir[3]
    # night on either side: coefficients zero, reference still stored
    assert bb[4] == bx[4] == gg[4] == gx[4] == F(0.0)
    assert bb[5] == bx[5] == gg[5] == gx[5] == F(0.0)
    assert np.all(cref == cz) and np.all(dref == swdown) and np.all(bref == swddir)
    albedo = np.full(n, F(0.25))
    now = np.full(n, F(0.5))
    sw, dir_, dni, dif, gsw = swint.interp_sw_radiation(
        cref, now, bref, bb, bx, dref, gg, gx, albedo)
    # clamped columns take the ratio rule
    assert dir_[1] == F(F(now[1] / cref[1]) * bref[1])
    assert sw[2] == F(F(now[2] / cref[2]) * dref[2])
    # exponent-zero column holds the reference flux at any sun
    assert dir_[3] == bref[3] and sw[3] == dref[3]
    # night reference (column 4): zero though the sun is up now
    assert sw[4] == dir_[4] == dni[4] == dif[4] == gsw[4] == F(0.0)
    day = np.array([0, 1, 2, 3, 5])
    bits_equal("swddif", dif[day], (sw[day] - dir_[day]).astype(F))
    bits_equal("swddni", dni[day], (dir_[day] / now[day]).astype(F))
    bits_equal("gsw", gsw[day], (sw[day] * (F(1.0) - albedo[day])).astype(F))


def test_night_now_zeroes_a_daytime_fit():
    n = 3
    cz = np.full(n, F(0.9))
    czl = np.full(n, F(0.85))
    swddir = np.full(n, F(700.0))
    swdown = np.full(n, F(900.0))
    bb, bx, gg, gx, cref, dref, bref = (np.zeros(n, F) for _ in range(7))
    swint.update_swinterp_parameters(cz, czl, swddir, swdown, bref, bb, bx,
                                     dref, gg, gx, cref)
    now = np.array([-0.2, 0.0, 1.0e-4], F)
    res = swint.interp_sw_radiation(cref, now, bref, bb, bx, dref, gg, gx,
                                    np.full(n, F(0.2)))
    for arr in res:
        assert np.all(arr == F(0.0))


# ---------------------------------------------------------------------------
# Kernels against the fork (GPU).
# ---------------------------------------------------------------------------

def _device_carrier(ncol, lat=None, lon=None):
    from gpuwm.core.swint import ShortwaveInterpolation
    if lat is None:
        lat = np.linspace(-60.0, 60.0, ncol).astype(F)
    if lon is None:
        lon = np.linspace(-170.0, 170.0, ncol).astype(F)
    return ShortwaveInterpolation(start_time=datetime(2026, 6, 15, 12),
                                  latitude_deg=lat.reshape(1, ncol),
                                  longitude_deg=lon.reshape(1, ncol),
                                  shape=(1, ncol)), lat, lon


@gpu
def test_kernels_match_the_fork_word_for_word(oracle):
    import cupy as cp
    o = oracle
    cz = o["in/coszen"]
    ncall, ncol = cz.shape
    nloc = o["in/czloc"].shape[1]
    carrier, _lat, _lon = _device_carrier(ncol)
    shape = (1, ncol)
    fields = {name: cp.zeros(shape, dtype=cp.float32)
              for name in (*swint.STATE_FIELDS, "swdown", "gsw")}
    fields["albedo"] = cp.asarray(o["in/albedo"].reshape(shape))
    key = {"bb": "swint_bb", "bx": "swint_bx", "gg": "swint_gg",
           "gx": "swint_gx", "coszen_ref": "coszen_ref",
           "swdown_ref": "swdown_ref", "swddir_ref": "swddir_ref"}
    words = 0
    for ic in range(ncall):
        fields["swdown"][...] = cp.asarray(o["in/swdown"][ic].reshape(shape))
        carrier._coszen_loc[...] = cp.asarray(o["in/czcall"][ic])
        f = {name: fields[name].reshape(-1) for name in swint.STATE_FIELDS}
        f["swint_albedo"][...] = fields["albedo"].reshape(-1)
        carrier._k_update(carrier._grid, carrier._block,
                          (np.int64(ncol), cp.asarray(cz[ic]),
                           carrier._coszen_loc,
                           cp.asarray(o["in/swddir"][ic]),
                           fields["swdown"].reshape(-1),
                           f["swddir_ref"], f["swint_bb"], f["swint_bx"],
                           f["swdown_ref"], f["swint_gg"], f["swint_gx"],
                           f["coszen_ref"]))
        for name in COEFS:
            words += bits_equal(f"{name} call {ic}",
                                cp.asnumpy(f[key[name]]),
                                o[f"out/{name}"][ic])
        suns = [o["in/czcall"][ic]] + [o["in/czloc"][ic, il]
                                       for il in range(nloc)]
        for il, czl in enumerate(suns):
            carrier._coszen_loc[...] = cp.asarray(czl)
            carrier._interpolate(fields)
            for name in FLUXES:
                words += bits_equal(
                    f"{name} call {ic} sun {il}",
                    cp.asnumpy(fields[name]).reshape(-1),
                    o[f"out/{name}"][ic, il])
    print(f"swint kernels vs fork: {words} words, max ULP 0")


@gpu
def test_coszen_loc_kernel_matches_the_fork_word_for_word(oracle):
    """radconst + calc_coszen in the kernel, from julian/xtime/gmt alone,
    against the fork's calc_coszen on every calendar set (both sides of
    radconst's JULIAN = 80 branch, day and night points)."""
    import cupy as cp
    o = oracle
    lat, lon = o["coszen_in/xlat"], o["coszen_in/xlon"]
    n = lat.size
    carrier, _lat, _lon = _device_carrier(n, lat, lon)
    words = 0
    for s, (julian, xtime, gmt) in enumerate(o["coszen_in/sets"]):
        got = cp.asnumpy(carrier.coszen_loc_at(julian, xtime, gmt))
        want = o["coszen_out/coszen"][s]
        assert (want > 0).any() and (want < 0).any()
        words += bits_equal(f"coszen_loc set {s}", got, want)
    print(f"coszen_loc kernel vs fork: {words} words, max ULP 0")


@gpu
def test_radiation_step_then_between_calls_through_the_carrier():
    """The driver-facing entry points on a day/night strip: the radiation
    step leaves the interpolation's fluxes (not the scheme's) in the
    fields, a later step follows the sun, night columns are zero, and the
    kernel results equal the twins fed the kernel's own cosine."""
    import cupy as cp
    ncol = 512
    carrier, lat, lon = _device_carrier(ncol)
    shape = (1, ncol)
    fields = {name: cp.zeros(shape, dtype=cp.float32)
              for name in (*swint.STATE_FIELDS, "swdown", "gsw")}
    fields["albedo"] = cp.full(shape, F(0.3), dtype=cp.float32)
    elapsed = 3600.0 * 6.0
    # the radiation call's own sun: half an interval ahead
    coszen_rad = cp.asnumpy(carrier.coszen_loc(elapsed + 450.0)).reshape(shape)
    swdown_rad = (F(1000.0) * np.maximum(coszen_rad, 0.0)).astype(F)
    swddir_rad = (F(0.8) * swdown_rad).astype(F)
    fields["swdown"][...] = cp.asarray(swdown_rad)
    carrier.radiation_step(fields, coszen=cp.asarray(coszen_rad),
                           swddir=cp.asarray(swddir_rad),
                           elapsed_seconds=elapsed)
    now = cp.asnumpy(carrier._coszen_loc).reshape(-1).copy()
    after_call = cp.asnumpy(fields["swdown"]).reshape(-1)
    bits_equal("swdown_ref is the scheme's SWDOWN",
               cp.asnumpy(fields["swdown_ref"]), swdown_rad)
    bits_equal("swint_albedo captured", cp.asnumpy(fields["swint_albedo"]),
               cp.asnumpy(fields["albedo"]))
    day = (coszen_rad.reshape(-1) > swint.COSZEN_MIN) & (now > swint.COSZEN_MIN)
    assert day.any() and (~day).any()
    assert np.all(after_call[~day] == F(0.0))
    assert not np.array_equal(after_call[day], swdown_rad.reshape(-1)[day])
    # the twins, fed the kernel's own cosine, give the same words
    z = lambda: np.zeros(ncol, F)
    bb, bx, gg, gx, cref, dref, bref = (z() for _ in range(7))
    swint.update_swinterp_parameters(coszen_rad, now, swddir_rad, swdown_rad,
                                     bref, bb, bx, dref, gg, gx, cref)
    twin = swint.interp_sw_radiation(cref, now, bref, bb, bx, dref, gg, gx,
                                     np.full(ncol, F(0.3)))
    for name, arr in zip(FLUXES, twin):
        bits_equal(f"carrier {name} vs twin",
                   cp.asnumpy(fields[name]).reshape(-1), arr)
    carrier.between_calls(fields, elapsed_seconds=elapsed + 300.0)
    later = cp.asnumpy(fields["swdown"]).reshape(-1)
    assert not np.array_equal(later[day], after_call[day])
    assert np.all(later[~day] == F(0.0))
    carrier.between_calls(fields, elapsed_seconds=elapsed + 300.0)
    bits_equal("deterministic", cp.asnumpy(fields["swdown"]).reshape(-1), later)
