"""WRF swint_opt = 1 (gpuwm.core.swint): twins, kernels and two oracles.

The oracle of record is WRF v4.6.1's own Fortran (radconst, calc_coszen,
update_swinterp_parameters and interp_sw_radiation cut unchanged from
module_radiation_driver.F and driven step by step in the radiation
driver's order, gfortran 13.3 -O0 -ffp-contract=off, by
tools/swint_wrf461_oracle), stored in
tests/data/swint_wrf461_oracle/swint_wrf461.npz: six scenarios (CONUS
day, polar day and night with live snow albedo, subsolar points, forced
fit branches, random globe, cosine clamp points).

The older fixture tests/data/swint_oracle/swint_oracle.npz is the
operational HRRR fork's Fortran (NOAA-EMC/HRRR v4.1.21, WRFV3.9 base).
The fork predates two upstream WRF bug fixes the engine follows (the
exponent-zero ratio rule, WRF cc02f989f, and the [-1, 1] cosine clamp,
WRF 650be0c7d); it is graded word for word everywhere else, and on the
words those fixes touch the expected value is the v4.6.1 statement
applied to the fork's own state.

Layers:

1. the NumPy float32 twins of the fit and the evaluation against both
   oracles, word for word (the twins use WOOF's own float32 logf/powf
   through gpuwm.core.noahmp_libm);
2. the CUDA kernels against both oracles, word for word (GPU): the fit,
   the evaluation and the per-step zenith cosine, which the kernel forms
   with WOOF's own float32 sinf/cosf/asinf from the calendar scalars
   alone; against v4.6.1 through the production entry points
   radiation_step / between_calls;
3. branch coverage the fixtures were built to reach, asserted directly
   (fresh column, clamped and zero exponents, night either side, the
   cosine clamp, the derived SWDDIF/SWDDNI/GSW, the live albedo), and the
   driver-facing carrier on a day/night strip.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_twins_match_the_fork_word_for_word -> test_twins_match_the_fork_with_the_upstream_fix.
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import swint

F = np.float32
REPO = Path(__file__).resolve().parents[1]
FIXTURE = Path(os.environ.get('GPUWM_SWINT_ORACLE',
    REPO / "tests" / "data" / "swint_oracle" / "swint_oracle.npz"))
FIXTURE_SHA256 = REPO / "tests" / "data" / "swint_oracle" / "SHA256"

gpu = pytest.mark.gpu

FLUXES = ("swdown", "swddir", "swddni", "swddif", "gsw")
COEFS = ("bb", "bx", "gg", "gx", "coszen_ref", "swdown_ref", "swddir_ref")

V461 = REPO / "tests" / "data" / "swint_wrf461_oracle" / "swint_wrf461.npz"
V461_SHA256 = REPO / "tests" / "data" / "swint_wrf461_oracle" / "SHA256"
#: run_swint.F90's per-step output rows
V461_ROWS = ("coszen_loc", "coszen", "bb", "bx", "gg", "gx", "coszen_ref",
             "swdown_ref", "swddir_ref", "swdown", "swddir", "swddni",
             "swddif", "gsw")
CARRIER_KEY = {"bb": "swint_bb", "bx": "swint_bx", "gg": "swint_gg",
               "gx": "swint_gx", "coszen_ref": "coszen_ref",
               "swdown_ref": "swdown_ref", "swddir_ref": "swddir_ref"}


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


def _fork_with_upstream_fixes(o):
    """The fork fixture's fluxes with WRF cc02f989f applied: wherever the
    fork's stored exponent is exactly zero on a day column, v4.6.1 takes
    the ratio rule where the fork took ``Bx * coszen**0``.  The replacement
    words are v4.6.1's statements on the fork's own state, in float32.
    Returns the expected fluxes and how many sun/column cells changed."""
    exp = {name: o[f"out/{name}"].copy() for name in FLUXES}
    ncall, nsun, _ncol = exp["swdown"].shape
    changed = 0
    for ic in range(ncall):
        czr = o["out/coszen_ref"][ic]
        bb, gg = o["out/bb"][ic], o["out/gg"][ic]
        for il in range(nsun):
            czl = (o["in/czcall"][ic] if il == 0
                   else o["in/czloc"][ic, il - 1])
            day = (czr > swint.COSZEN_MIN) & (czl > swint.COSZEN_MIN)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (czl / czr).astype(F)
            fb = day & (bb == F(0.0))
            fg = day & (gg == F(0.0))
            d = np.where(fb, (ratio * o["out/swddir_ref"][ic]).astype(F),
                         exp["swddir"][ic, il]).astype(F)
            sdn = np.where(fg, (ratio * o["out/swdown_ref"][ic]).astype(F),
                           exp["swdown"][ic, il]).astype(F)
            hit = fb | fg
            with np.errstate(divide="ignore", invalid="ignore"):
                exp["swddir"][ic, il] = d
                exp["swdown"][ic, il] = sdn
                exp["swddif"][ic, il] = np.where(
                    hit, (sdn - d).astype(F), exp["swddif"][ic, il])
                exp["swddni"][ic, il] = np.where(
                    fb, (d / czl).astype(F), exp["swddni"][ic, il])
                exp["gsw"][ic, il] = np.where(
                    fg, (sdn * (F(1.0) - o["in/albedo"])).astype(F),
                    exp["gsw"][ic, il])
            changed += int(hit.sum())
    return exp, changed


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


def test_twins_match_the_fork_with_the_upstream_fix(oracle):
    o = oracle
    out, flux = _replay_twins(o)
    expected, changed = _fork_with_upstream_fixes(o)
    # the fork fixture reaches the exponent-zero branch the fix changes
    assert changed > 100
    words = 0
    for name in COEFS:
        words += bits_equal(name, out[name], o[f"out/{name}"])
    for name in FLUXES:
        words += bits_equal(name, flux[name], expected[name])
    print(f"swint twins vs fork (+cc02f989f on {changed} cells): {words} "
          "words, max ULP 0")


def test_coszen_twin_is_the_fork_shape(oracle):
    """The NumPy cosine twin differs from the fork only by NumPy's
    transcendentals (dossier section 9.1's seam): within a few float32
    spacings of the order-one terms it sums."""
    o = oracle
    sets = o["coszen_in/sets"]
    worst = 0.0
    for s, (julian, xtime, gmt) in enumerate(sets):
        got = swint.coszen_loc_host(o["coszen_in/xlat"], o["coszen_in/xlon"],
                                    julian, xtime, gmt, generation='wrf_461')
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
    # exponent-zero column takes the ratio rule (WRF cc02f989f): the flux
    # follows the sun instead of holding the reference flux
    assert dir_[3] == F(F(now[3] / cref[3]) * bref[3])
    assert sw[3] == F(F(now[3] / cref[3]) * dref[3])
    assert dir_[3] != bref[3]
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
                                  shape=(1, ncol), generation='wrf_461'), lat, lon


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
    key = CARRIER_KEY
    expected, changed = _fork_with_upstream_fixes(o)
    assert changed > 100
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
                    expected[name][ic, il])
    print(f"swint kernels vs fork (+cc02f989f on {changed} cells): {words} "
          "words, max ULP 0")


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
        # the fork stores the unbounded sum; v4.6.1 bounds it (650be0c7d)
        want = np.minimum(np.maximum(o["coszen_out/coszen"][s], F(-1.0)),
                          F(1.0)).astype(F)
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
    now2 = cp.asnumpy(carrier._coszen_loc).reshape(-1).copy()
    assert not np.array_equal(later[day], after_call[day])
    cref = cp.asnumpy(fields["coszen_ref"]).reshape(-1)
    day2 = (cref > swint.COSZEN_MIN) & (now2 > swint.COSZEN_MIN)
    assert np.all(later[~day2] == F(0.0))
    # sunrise between the call and this step: the call stored no fit (its
    # step sun was down) but did store the reference, so v4.6.1's
    # exponent-zero rule scales the reference by the cosine ratio (WRF
    # cc02f989f); the fork left these columns dark until the next call
    sunrise = day2 & ~day
    if sunrise.any():
        bits_equal("sunrise columns take the ratio rule", later[sunrise],
                   (np.float32(now2[sunrise] / cref[sunrise])
                    * swdown_rad.reshape(-1)[sunrise]).astype(F))
    carrier.between_calls(fields, elapsed_seconds=elapsed + 300.0)
    bits_equal("deterministic", cp.asnumpy(fields["swdown"]).reshape(-1), later)


# ---------------------------------------------------------------------------
# WRF v4.6.1 (the oracle of record): tools/swint_wrf461_oracle.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v461():
    assert V461.is_file(), (
        "tests/data/swint_wrf461_oracle/swint_wrf461.npz is cut by "
        "tools/swint_wrf461_oracle/cut_fixture.py from an oracle build")
    want = V461_SHA256.read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(V461.read_bytes()).hexdigest()
    assert got == want, f"v4.6.1 swint fixture sha256 {got} != pinned {want}"
    data = np.load(V461)
    return {k: data[k] for k in data.files}


def _row(name):
    return V461_ROWS.index(name)


def test_v461_receipt_and_coverage(v461):
    """The fixture is unmodified v4.6.1 at -O0 and reaches every branch the
    two upstream fixes and the live albedo change: the cosine clamp, zero
    exponents on day columns (direct and global), both exponent clamps,
    night on either side, and albedo changing between radiation calls."""
    receipt = json.loads(str(v461["receipt"]))
    assert receipt["wrf_commit"] == "d66e442fccc04111067e29274c9f9eaccc3cef28"
    assert receipt["wrf_source_sha256"]["phys/module_radiation_driver.F"] == (
        "779875651512709c47d50ef6c1e1216ad94726f1996c5e45a0db8eb800733ffa")
    assert "-ffp-contract=off" in receipt["flags"]
    assert "-O0" in receipt["flags"]
    assert "GNU Fortran" in receipt["build_receipt"]
    names = [str(n) for n in v461["scenarios"]]
    assert set(names) == {"conus_day", "polar", "subsolar", "edge",
                          "stress", "clamp"}
    count = dict(clamp1=0, bb0=0, gg0=0, bbhi=0, bblo=0, gghi=0, gglo=0,
                 night=0, day=0, albedo_moves=0, columns=0)
    for name in names:
        w = v461[f"{name}/wrf"]
        count["columns"] += w.shape[2]
        cz = w[:, _row("coszen_loc")]
        count["clamp1"] += int((cz == F(1.0)).sum())
        day = (w[:, _row("coszen_ref")] > swint.COSZEN_MIN) & (
            cz > swint.COSZEN_MIN)
        count["day"] += int(day.sum())
        count["night"] += int((~day).sum())
        bb, gg = w[:, _row("bb")], w[:, _row("gg")]
        count["bb0"] += int((day & (bb == 0) & (w[:, _row("swddir_ref")] > 0)).sum())
        count["gg0"] += int((day & (gg == 0) & (w[:, _row("swdown_ref")] > 0)).sum())
        count["bbhi"] += int((bb == F(2.5)).sum())
        count["bblo"] += int((bb == F(-0.5)).sum())
        count["gghi"] += int((gg == F(2.5)).sum())
        count["gglo"] += int((gg == F(-0.5)).sum())
        alb = v461[f"{name}/albedo"]
        count["albedo_moves"] += int((alb[1:] != alb[:-1]).sum())
    assert count["columns"] >= 64
    for key, value in count.items():
        assert value > 0, (key, count)
    print("v4.6.1 swint fixture coverage:", count)


def _replay_v461_twins(v461, name):
    """The NumPy twins over one scenario, the cosines taken from WRF's
    own output (the twin cosine uses NumPy's transcendentals)."""
    w = v461[f"{name}/wrf"]
    nstep, _rows, ncol = w.shape
    radflag = v461[f"{name}/radflag"]
    state = {k: np.zeros(ncol, F) for k in COEFS}
    got = np.zeros_like(w)
    icall = 0
    for s in range(nstep):
        czl = w[s, _row("coszen_loc")]
        if radflag[s]:
            swint.update_swinterp_parameters(
                w[s, _row("coszen")], czl, v461[f"{name}/swddir"][icall],
                v461[f"{name}/swdown"][icall], state["swddir_ref"],
                state["bb"], state["bx"], state["swdown_ref"], state["gg"],
                state["gx"], state["coszen_ref"])
            icall += 1
        res = swint.interp_sw_radiation(
            state["coszen_ref"], czl, state["swddir_ref"], state["bb"],
            state["bx"], state["swdown_ref"], state["gg"], state["gx"],
            v461[f"{name}/albedo"][s])
        got[s, _row("coszen_loc")] = czl
        got[s, _row("coszen")] = w[s, _row("coszen")]
        for k in COEFS:
            got[s, _row(k)] = state[k]
        for k, arr in zip(FLUXES, res):
            got[s, _row(k)] = arr
    return got


@pytest.mark.parametrize("name", ["conus_day", "polar", "subsolar", "edge",
                                  "stress", "clamp"])
def test_twins_match_wrf461_word_for_word(v461, name):
    got = _replay_v461_twins(v461, name)
    w = v461[f"{name}/wrf"]
    words = 0
    for k in V461_ROWS:
        words += bits_equal(f"{name} {k}", got[:, _row(k)], w[:, _row(k)])
    print(f"swint twins vs WRF v4.6.1 ({name}): {words} words, max ULP 0")


def _run_v461_carrier(v461, name, cp):
    """WOOF's production entry points over one scenario: radiation_step on
    a radiation step (with WRF's radiation-step cosine as the scheme's
    COSZEN), between_calls otherwise, the live albedo of each step."""
    w = v461[f"{name}/wrf"]
    nstep, _rows, ncol = w.shape
    shape = (1, ncol)
    start = datetime.fromisoformat(str(v461[f"{name}/start"]))
    carrier = swint.ShortwaveInterpolation(
        start_time=start,
        latitude_deg=v461[f"{name}/xlat"].reshape(shape),
        longitude_deg=v461[f"{name}/xlon"].reshape(shape), shape=shape,
        # The referee is WRF v4.6.1: its sun (2.9.1 defaults to wrf_480).
        generation="wrf_461")
    fields = {k: cp.zeros(shape, dtype=cp.float32)
              for k in (*swint.STATE_FIELDS, "swdown", "gsw", "albedo")}
    radflag = v461[f"{name}/radflag"]
    elapsed = v461[f"{name}/elapsed"]
    got = np.zeros_like(w)
    icall = 0
    for s in range(nstep):
        j, x, g = swint.calendar_scalars(start, float(elapsed[s]))
        assert (j, x, g) == (v461[f"{name}/julian"][s],
                             v461[f"{name}/xtime"][s],
                             v461[f"{name}/gmt"][s])
        fields["albedo"][...] = cp.asarray(
            v461[f"{name}/albedo"][s].reshape(shape))
        if radflag[s]:
            fields["swdown"][...] = cp.asarray(
                v461[f"{name}/swdown"][icall].reshape(shape))
            carrier.radiation_step(
                fields, coszen=cp.asarray(w[s, _row("coszen")].reshape(shape)),
                swddir=cp.asarray(v461[f"{name}/swddir"][icall].reshape(shape)),
                elapsed_seconds=float(elapsed[s]))
            icall += 1
        else:
            carrier.between_calls(fields, elapsed_seconds=float(elapsed[s]))
        got[s, _row("coszen_loc")] = cp.asnumpy(carrier._coszen_loc).reshape(-1)
        got[s, _row("coszen")] = w[s, _row("coszen")]
        for k, key in CARRIER_KEY.items():
            got[s, _row(k)] = cp.asnumpy(fields[key]).reshape(-1)
        for k in FLUXES:
            got[s, _row(k)] = cp.asnumpy(fields[k]).reshape(-1)
        if radflag[s]:
            # the kernel's own radiation-step cosine (xtime + radt*0.5)
            radt = F(v461["radt_minutes"])
            k = cp.asnumpy(carrier.coszen_loc_at(
                j, F(x + F(radt * F(0.5))), g)).reshape(-1)
            bits_equal(f"{name} step {s} radiation-step coszen", k,
                       w[s, _row("coszen")])
    return got


@gpu
@pytest.mark.parametrize("name", ["conus_day", "polar", "subsolar", "edge",
                                  "stress", "clamp"])
def test_kernels_match_wrf461_word_for_word(v461, name):
    """Word for word under the strict build.  Default arithmetic compiles
    with flush-to-zero, so there the columns whose direct beam is a float32
    subnormal (the edge scenario's ``subnormal_direct``) may lose those
    words, by less than the smallest normal float; every other column is
    still held word for word."""
    from gpuwm import wrf_exact
    import cupy as cp
    got = _run_v461_carrier(v461, name, cp)
    w = v461[f"{name}/wrf"]
    keep = np.ones(w.shape[2], bool)
    if not wrf_exact.ENABLED:
        keep = v461[f"{name}/label"].astype(str) != "subnormal_direct"
        tiny = np.finfo(np.float32).tiny
        diff = np.abs(got[:, :, ~keep].astype(np.float64)
                      - w[:, :, ~keep].astype(np.float64))
        assert float(diff.max(initial=0.0)) < tiny
    words = 0
    for k in V461_ROWS:
        words += bits_equal(f"{name} {k}", got[:, _row(k)][:, keep],
                            w[:, _row(k)][:, keep])
    print(f"swint kernels vs WRF v4.6.1 ({name}): {words} words, max ULP 0")


@gpu
def test_gsw_takes_the_live_albedo_between_radiation_calls():
    """v4.6.1 passes grid%albedo to interp_sw_radiation on every step, so a
    surface whose albedo changes between radiation calls (snowfall, melt)
    changes GSW at once; the albedo the last radiation call used is not
    held."""
    import cupy as cp
    ncol = 64
    carrier, _lat, _lon = _device_carrier(
        ncol, np.full(ncol, 35.0, F), np.linspace(-100.0, -60.0, ncol).astype(F))
    shape = (1, ncol)
    fields = {k: cp.zeros(shape, dtype=cp.float32)
              for k in (*swint.STATE_FIELDS, "swdown", "gsw")}
    fields["albedo"] = cp.full(shape, F(0.2), dtype=cp.float32)
    elapsed = 3600.0 * 4.0
    cosine = carrier.coszen_loc(elapsed + 450.0).reshape(shape).copy()
    fields["swdown"][...] = cp.asarray(
        (F(1000.0) * np.maximum(cp.asnumpy(cosine), 0)).astype(F))
    carrier.radiation_step(fields, coszen=cosine,
                           swddir=fields["swdown"] * F(0.8),
                           elapsed_seconds=elapsed)
    fields["albedo"][...] = F(0.85)                  # fresh snow
    carrier.between_calls(fields, elapsed_seconds=elapsed + 300.0)
    sw = cp.asnumpy(fields["swdown"]).reshape(-1)
    assert (sw > 0).sum() > ncol // 2
    bits_equal("gsw with the live albedo",
               cp.asnumpy(fields["gsw"]).reshape(-1),
               (sw * (F(1.0) - F(0.85))).astype(F))
    # the captured albedo is still the radiation call's (restart state)
    assert bool((fields["swint_albedo"] == F(0.2)).all())


def test_interpolate_refuses_without_an_albedo():
    """GSW needs the step's albedo; a fields dict with neither ALBEDO nor
    ALBSOL is refused by name rather than evaluated on the captured copy."""
    carrier = swint.ShortwaveInterpolation.__new__(swint.ShortwaveInterpolation)
    carrier.shape = (1, 2)

    class _Arr:
        pass
    with pytest.raises(KeyError, match="live albedo"):
        # _flat is never reached for the state fields: the albedo lookup
        # happens first only if the state fields pass, so give it stubs
        carrier._flat = lambda a, n: a
        carrier._interpolate({name: _Arr() for name in swint.STATE_FIELDS})
