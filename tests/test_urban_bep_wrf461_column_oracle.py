"""BEP (sf_urban_physics = 2) against WRF v4.6.1: the column oracle.

``tools/urban_bep_wrf461_oracle/build.sh`` compiles WRF v4.6.1's
``module_sf_bep.F`` (byte-identical to v4.7.1's) and its dependencies
byte-unmodified at -O0 and drives ``SUBROUTINE BEP`` (4 steps) and the
zeroing + BEP + Noah / Noah-MP coupling block + surface-diagnostic override
(2 steps; the blocks are cut from the v4.6.1 files by line range and are
byte-identical to the v4.7.1 cuts) over 100 (URBPARM.TBL, 3 classes) and 180
(URBPARM_LCZ.TBL, 11 classes) columns: sixteen regimes (convective day,
stable night, low sun, high terrain day and night, snow-season cold, coastal
gale, hot desert, towers to 75 m, exactly zero wind, sun on the horizon,
overcast, strong inversion ...), eight morphology variants and four edge
columns (urban fraction 0, 1e-3, 0.02, 1).

Two WRF arms, both in ``tests/data/oracles/urban/bep461``:

* ``cr``   WRF linked with correctly rounded acosf/asinf/atanf/tanf
  (``build_crlibm_arm.sh``): what BEP computes with an exact libm.  WOOF must
  match every word, bitwise.
* ``host`` WRF linked with the build host's C library (Ubuntu 24.04, glibc
  2.39), whose acosf/asinf/atanf/tanf are not correctly rounded.  The two arms
  differ only on the regime-9 columns (cosine of the zenith angle 0.24, where
  that acosf is 1 ULP off), and so does WOOF; the distance is pinned.

One declared departure: WRF's coupling block divides the rural stress by the
level-1 wind speed, so a column with exactly zero wind gets -inf drag and a
NaN ust (regime 13).  WOOF's calm guard (``urban_bep_couple.cu``) leaves the
directionless rural stress out and takes ust's zero-wind limit; those words
are not graded bitwise and must be finite.
"""

from __future__ import annotations

import hashlib
import importlib.util
import tarfile
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tests" / "data" / "oracles" / "urban" / "bep461" \
    / "wrf461-bep-fixtures.tar.xz"
ARCHIVE_SHA256 = "60535187eb3a47de7fc1d5f0d895cd07d137141ee4dc843b5f26de06302c5afa"
TOOL = ROOT / "tools" / "urban_bep_wrf461_oracle" / "compare.py"

#: host arm (glibc 2.39 acosf) against WOOF, measured on W1 (RTX PRO 6000,
#: NVRTC 12.9), identical under GPUWM_WRF_EXACT=1 and default arithmetic:
#: words whose bits differ and the largest ULP distance, all on regime 9.
HOST_ARM_BITS_DIFFER = 285
HOST_ARM_MAX_ULP = 114


def _compare():
    spec = importlib.util.spec_from_file_location("urban_bep461_compare", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory) -> Path:
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256
    out = tmp_path_factory.mktemp("bep461")
    with tarfile.open(ARCHIVE, "r:xz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(out, filter="data")
        else:                      # Python without extraction filters
            tar.extractall(out)
    return out / "bep461"


def test_columns_span_the_regimes_and_edges(fixtures):
    census = _compare().column_census(fixtures / "cr")
    assert census["nlcd"]["urban_columns"] >= 64
    assert census["lcz"]["urban_columns"] >= 64
    for tset, nclass in (("nlcd", 3), ("lcz", 11)):
        assert census[tset]["regimes"] == list(range(1, 17))
        assert census[tset]["classes"] == list(range(1, nclass + 1))
    cmp = _compare()
    for tset in ("nlcd", "lcz"):
        fx = cmp.load(fixtures / "cr" / f"bep461_{tset}" / "steps")
        frc = fx["frc_urb2d"]
        assert (frc == 0).sum() == 1 and (frc == 1).any()
        assert (frc == np.float32(1e-3)).any() and (frc == np.float32(0.02)).any()
        assert (fx["swdown"] == 0).any() and (fx["cosz_urb2d"] < 0).any()
        assert (fx["p_phy"][0] < 75000).any()             # high terrain
        assert (fx["th_phy"][0] < 262).any()              # snow-season cold
        assert (fx["hi_urb2d"][14] > 0).any()             # a 75 m tower bin
        assert (fx["hi_urb2d"] < 0).any()                 # rejected histogram
        calm = (fx["u_phy"][0] == 0) & (fx["v_phy"][0] == 0)
        assert set(fx["regime"][calm].tolist()) == {13}


def test_wrf_calm_column_division_is_recorded(fixtures):
    """The WRF defect WOOF does not copy: -inf drag and NaN ust exactly on
    the zero-wind columns, finite everywhere else, both LSM blocks."""
    cmp = _compare()
    for tset in ("nlcd", "lcz"):
        fx = cmp.load(fixtures / "cr" / f"bep461_couple_{tset}" / "steps")
        calm = (fx["u_phy"][0] == 0) & (fx["v_phy"][0] == 0)
        assert calm.sum() >= 6
        for lsm in ("noah", "noahmp"):
            for step in (1, 2):
                tag = f"{lsm}_s{step}_out_"
                assert np.all(fx[tag + "a_u_bep"][0][calm] == -np.inf)
                assert np.all(fx[tag + "a_v_bep"][0][calm] == -np.inf)
                assert np.all(np.isnan(fx[tag + "ust"][calm]))
                assert np.all(np.isfinite(fx[tag + "a_u_bep"][0][~calm]))
                assert np.all(np.isfinite(fx[tag + "ust"][~calm]))


def test_host_and_correctly_rounded_wrf_differ_only_on_regime_9(fixtures):
    cmp = _compare()
    for name in ("bep461_nlcd", "bep461_lcz", "bep461_couple_nlcd",
                 "bep461_couple_lcz"):
        host = cmp.load(fixtures / "host" / name / "steps")
        cr = cmp.load(fixtures / "cr" / name / "steps")
        regime, ncol = host["regime"], host["regime"].size
        touched = np.zeros(ncol, dtype=bool)
        for key, a in host.items():
            b = cr[key]
            if a.dtype.kind != "f":
                assert np.array_equal(a, b), key
                continue
            differ = (np.ascontiguousarray(a).view(np.uint32)
                      != np.ascontiguousarray(b).view(np.uint32))
            if differ.any():
                assert differ.shape[-1] == ncol, key
                touched |= differ.reshape(-1, ncol).any(axis=0)
        assert touched.any(), name
        assert set(regime[touched].tolist()) == {9}, name


@requires_gpu
def test_bep_and_coupling_are_bitwise_wrf_with_an_exact_libm(fixtures):
    result = _compare().run(fixtures / "cr")
    assert result["words"] > 5_000_000
    assert result["bits_differ"] == 0, result["first_differences"]
    assert result["max_ulp"] == 0
    # the calm-column words WRF left non-finite: a_u_bep and a_v_bep at
    # level 1 and ust, both LSM blocks, both steps; WOOF's are finite
    calm = result["calm_wrf_nonfinite_words"]
    assert calm == 3 * 2 * 2 * (6 + 11)
    assert result["calm_woof_nonfinite_words"] == 0


@requires_gpu
def test_host_libm_arm_distance_is_pinned(fixtures):
    result = _compare().run(fixtures / "host")
    assert result["bits_differ"] == HOST_ARM_BITS_DIFFER
    assert result["max_ulp"] == HOST_ARM_MAX_ULP
    assert result["calm_woof_nonfinite_words"] == 0


@requires_gpu
def test_calm_guard_keeps_the_coupling_finite():
    """Two columns, one with exactly zero level-1 wind: the calm one gets no
    rural drag term and ust = ((urban stress) + (1-frc)*ust**2)**.5; the
    windy one is untouched by the guard."""
    import cupy as cp

    from gpuwm.core.urban_bep_couple import PBL_TERMS, launch_bep_couple

    nz, ncol = 3, 2

    def lev(values):
        return cp.asarray(np.ascontiguousarray(
            np.asarray(values, dtype=np.float32).reshape(nz, 1, ncol)))

    def sfc(values):
        return cp.asarray(np.asarray(values, dtype=np.float32).reshape(1, ncol))

    frc = np.float32(0.6)
    u = np.array([[0.0, 2.0]] * nz, dtype=np.float32)
    v = np.array([[0.0, 1.0]] * nz, dtype=np.float32)
    pbl = {n: lev(np.zeros((nz, ncol))) for n in PBL_TERMS}
    pbl["a_u_bep"] = lev(np.full((nz, ncol), -0.01))
    pbl["a_v_bep"] = lev(np.full((nz, ncol), -0.01))
    pbl["sf_bep"] = lev(np.ones((nz, ncol)))
    pbl["vl_bep"] = lev(np.full((nz, ncol), 0.9))
    us = np.float32(0.3)
    fields = {"ust": sfc([us, us]), "tsk": sfc([290.0, 290.0]),
              "hfx": sfc([50.0, 50.0]), "qfx": sfc([1e-5, 1e-5]),
              "lh": sfc([25.0, 25.0]), "grdflx": sfc([10.0, 10.0]),
              "albedo": sfc([0.2, 0.2]), "emiss": sfc([0.95, 0.95])}
    diag = {n: sfc([-1.0, -1.0]) for n in ("ts_urb2d", "sh_urb2d",
                                            "lh_urb2d", "g_urb2d", "rn_urb2d")}
    bep_out = {"rl_up_urb": sfc([400.0, 400.0]), "rs_abs_urb": sfc([0.0, 0.0]),
               "emiss_urb": sfc([0.9, 0.9]), "grdflx_urb": sfc([5.0, 5.0])}
    launch_bep_couple(lsm=2, frc_urb2d=sfc([frc, frc]),
                      dz8w=lev(np.full((nz, ncol), 10.0)),
                      rho=lev(np.full((nz, ncol), 1.2)), u_phy=lev(u),
                      v_phy=lev(v), glw=sfc([300.0, 300.0]),
                      swdown=sfc([0.0, 0.0]), bep_out=bep_out, pbl=pbl,
                      fields=fields, diag=diag)
    a_u = cp.asnumpy(pbl["a_u_bep"]).reshape(nz, ncol)
    a_v = cp.asnumpy(pbl["a_v_bep"]).reshape(nz, ncol)
    ust = cp.asnumpy(fields["ust"]).reshape(ncol)
    assert np.isfinite(a_u).all() and np.isfinite(a_v).all()
    assert np.isfinite(ust).all() and ust[0] > 0
    weighted = np.float32(-0.01) * frc
    # calm column: the frc-weighted urban coefficient only
    assert a_u[0, 0] == weighted and a_v[0, 0] == weighted
    # windy column: the rural drag is added, as WRF adds it
    assert a_u[0, 1] < weighted and a_v[0, 1] < weighted
    # zero wind leaves no urban stress (a_u*u = 0, b_u = 0), so ust is the
    # rural magnitude's square root
    expect = np.sqrt(np.float64((1 - frc) * us * us))
    assert abs(float(ust[0]) - expect) <= 2e-7 * expect
@requires_gpu
@pytest.mark.parametrize("lsm", [2, 4])
def test_full_urban_calm_guard_keeps_ust_positive(lsm):
    """Two columns, one with exactly zero level-1 wind: the calm one gets no
    rural drag term and ust = ((urban stress) + (1-frc)*ust**2)**.5; the
    windy one is untouched by the guard."""
    import cupy as cp

    from gpuwm.core.urban_bep_couple import PBL_TERMS, launch_bep_couple

    nz, ncol = 3, 2

    def lev(values):
        return cp.asarray(np.ascontiguousarray(
            np.asarray(values, dtype=np.float32).reshape(nz, 1, ncol)))

    def sfc(values):
        return cp.asarray(np.asarray(values, dtype=np.float32).reshape(1, ncol))

    frc = np.float32(1.0)
    u = np.array([[0.0, 2.0]] * nz, dtype=np.float32)
    v = np.array([[0.0, 1.0]] * nz, dtype=np.float32)
    pbl = {n: lev(np.zeros((nz, ncol))) for n in PBL_TERMS}
    pbl["a_u_bep"] = lev(np.full((nz, ncol), -0.01))
    pbl["a_v_bep"] = lev(np.full((nz, ncol), -0.01))
    pbl["sf_bep"] = lev(np.ones((nz, ncol)))
    pbl["vl_bep"] = lev(np.full((nz, ncol), 0.9))
    us = np.float32(0.3)
    fields = {"ust": sfc([us, us]), "tsk": sfc([290.0, 290.0]),
              "hfx": sfc([50.0, 50.0]), "qfx": sfc([1e-5, 1e-5]),
              "lh": sfc([25.0, 25.0]), "grdflx": sfc([10.0, 10.0]),
              "albedo": sfc([0.2, 0.2]), "emiss": sfc([0.95, 0.95])}
    diag = {n: sfc([-1.0, -1.0]) for n in ("ts_urb2d", "sh_urb2d",
                                            "lh_urb2d", "g_urb2d", "rn_urb2d")}
    bep_out = {"rl_up_urb": sfc([400.0, 400.0]), "rs_abs_urb": sfc([0.0, 0.0]),
               "emiss_urb": sfc([0.9, 0.9]), "grdflx_urb": sfc([5.0, 5.0])}
    launch_bep_couple(lsm=lsm, frc_urb2d=sfc([frc, frc]),
                      dz8w=lev(np.full((nz, ncol), 10.0)),
                      rho=lev(np.full((nz, ncol), 1.2)), u_phy=lev(u),
                      v_phy=lev(v), glw=sfc([300.0, 300.0]),
                      swdown=sfc([0.0, 0.0]), bep_out=bep_out, pbl=pbl,
                      fields=fields, diag=diag)
    a_u = cp.asnumpy(pbl["a_u_bep"]).reshape(nz, ncol)
    a_v = cp.asnumpy(pbl["a_v_bep"]).reshape(nz, ncol)
    ust = cp.asnumpy(fields["ust"]).reshape(ncol)
    assert np.isfinite(a_u).all() and np.isfinite(a_v).all()
    assert np.isfinite(ust).all() and ust[0] > 0
    weighted = np.float32(-0.01) * frc
    # calm column: the frc-weighted urban coefficient only
    assert a_u[0, 0] == weighted and a_v[0, 0] == weighted
    # full urban coverage has no rural drag weight
    assert a_u[0, 1] == weighted and a_v[0, 1] == weighted
    # zero stress at full coverage uses the declared surface-layer floor
    assert ust[0] >= np.float32(0.1)


    from _urban_calm_pbl import follow_calm_surface
    follow_calm_surface(float(ust[0]), float(cp.asnumpy(fields["hfx"])[0, 0]),
                        float(cp.asnumpy(fields["qfx"])[0, 0]))
