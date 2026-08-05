"""Grell-Freitas float32 reference against the byte-frozen WRF v4.6.1 oracle.

Stage 1 of the port: GFDRV's column preparation
(module_cu_gf_wrfdrv.F:383-492).  The oracle for it is
``gf-stage-levels.csv`` / ``gf-stage-surface.csv``, which record the prepared
column exactly as the driver built it, beside the raw 3-D inputs in
``gf-levels.csv`` / ``gf-surface.csv`` that it built them from.

Every assertion here is ``max_ulp == 0``.  The preparation is straight
arithmetic on measured constants -- no libm, no branches that a rounding can
flip -- so there is no honest reason for it to be anything else, and a
tolerance here would hide exactly the mixed-precision mistake this stage
exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.gf_oracle import (
    GF_NCASE,
    GF_NDX,
    GF_NZ,
    load_gf_oracle,
    measure_fields,
    stage_rows_to_distrust,
)
from gpuwm.verify.gf_ref import (
    DEEP_CP,
    DEEP_G,
    DEEP_RV,
    GFS_CP,
    GFS_G,
    GFS_RV,
    gf_driver_prep,
)

_LEVEL_FIELDS = [
    "zo",
    "po",
    "t2d",
    "q2d",
    "tn",
    "qo",
    "tshall",
    "qshall",
    "dhdt",
    "us",
    "vs",
    "rhoi",
]
_SURFACE_FIELDS = ["ter11", "psur", "xlandi", "hfxi", "qfxi", "ccn"]


@pytest.fixture(scope="module")
def fixture():
    return load_gf_oracle()


@pytest.fixture(scope="module")
def prepared(fixture):
    lv = fixture.levels
    sf = fixture.surface
    dt = float(np.unique(sf["dt"])[0])
    out = {}
    # dx is a fixture axis, and GFDRV takes it as a scalar for the whole
    # tile, so the reference is called once per dx and the results are
    # scattered back into the fixture's column order.
    for idx in range(1, GF_NDX + 1):
        sel = sf["idx"] == idx
        dx = float(np.unique(sf["dx"][sel])[0])
        got = gf_driver_prep(
            u=lv["u"][sel],
            v=lv["v"][sel],
            w=lv["w"][sel],
            t=lv["t"][sel],
            qv=lv["qv"][sel],
            p=lv["p"][sel],
            pi=lv["pi"][sel],
            rho=lv["rho"][sel],
            dz8w=lv["dz8w"][sel],
            p8w=lv["p8w"][sel],
            rthften=lv["rthften"][sel],
            rqvften=lv["rqvften"][sel],
            rthraten=lv["rthraten"][sel],
            rthblten=lv["rthblten"][sel],
            rqvblten=lv["rqvblten"][sel],
            ht=sf["ht"][sel],
            hfx=sf["hfx"][sel],
            qfx=sf["qfx"][sel],
            xland=sf["xland"][sel],
            kpbl=sf["kpbl"][sel],
            dt=dt,
            dx=dx,
        )
        for name, arr in got.items():
            if name not in out:
                shape = (fixture.ncol,) + arr.shape[1:]
                out[name] = np.zeros(shape, dtype=arr.dtype)
            out[name][sel] = arr
    return out


def test_fixture_shape(fixture):
    assert fixture.ncol == GF_NCASE * GF_NDX * 2
    assert fixture.levels["t"].shape == (fixture.ncol, GF_NZ)
    assert fixture.stage_levels["t2d"].shape == (fixture.ncol, GF_NZ)


def test_the_two_constant_sets_disagree():
    """Three of the four shared constant names differ between the driver and
    the cloud model, and the port must keep both.  This is a guard against a
    future tidy-up harmonising them."""
    assert np.float32(GFS_G) != DEEP_G
    assert np.float32(GFS_CP) != DEEP_CP
    assert np.float32(GFS_RV) != DEEP_RV


def test_gfs_constants_are_the_honest_doubles():
    """The trap: assigned to a real(8) and written out, ``con_g`` reads
    ``float64(float32(9.80665))``.  Its arithmetic does not behave that way --
    the fixture needs the honest double on all 8640 lanes of ``omeg``, and the
    float32-widened value misses 1488 of them by 1 ULP.  This test exists so a
    future reader of ``gf-pow-probe.txt``'s stored-word rows cannot quietly
    "correct" the constants back."""
    assert GFS_G.view(np.int64) == np.float64(9.80665).view(np.int64)
    assert GFS_G.view(np.int64) != np.int64(0x40239D0140000000)
    assert GFS_CP.view(np.int64) == np.float64(1004.6).view(np.int64)


@pytest.mark.parametrize("field", _LEVEL_FIELDS)
def test_driver_prep_levels_bitwise(fixture, prepared, field):
    (m,) = measure_fields(prepared, fixture.stage_levels, [field])
    assert m.n_mismatch == 0, f"{field}: {m.n_mismatch} non-comparable lanes"
    assert m.max_ulp == 0, (
        f"{field}: max {m.max_ulp} ULP at column {m.worst_col} k={m.worst_k}"
    )


def test_driver_prep_omeg_bitwise(fixture, prepared):
    """``omeg = -g*rho*w`` with the real(8) GFS g.  Broken out because getting
    its precision wrong is a silent 1-2 ULP error in every convecting column
    and nothing else in the preparation would catch it."""
    want = {"omeg": fixture.stage_levels["omeg_in"]}
    (m,) = measure_fields(prepared, want, ["omeg"])
    assert m.n_mismatch == 0
    assert m.max_ulp == 0, (
        f"omeg: max {m.max_ulp} ULP at column {m.worst_col} k={m.worst_k}"
    )


def test_driver_prep_mconv_bitwise(fixture, prepared):
    """``mconv`` accumulates through a real(8) divide and rounds to real(4)
    every iteration."""
    want = {"mconv": fixture.stage_surface["mconv_in"]}
    (m,) = measure_fields(prepared, want, ["mconv"])
    assert m.n_mismatch == 0
    assert m.max_ulp == 0, f"mconv: max {m.max_ulp} ULP at column {m.worst_col}"


@pytest.mark.parametrize("field", _SURFACE_FIELDS)
def test_driver_prep_surface_bitwise(fixture, prepared, field):
    (m,) = measure_fields(prepared, fixture.stage_surface, [field])
    assert m.n_mismatch == 0
    assert m.max_ulp == 0, f"{field}: max {m.max_ulp} ULP at column {m.worst_col}"


def test_kpbl_passes_through(fixture, prepared):
    assert np.array_equal(prepared["kpbli"], fixture.stage_surface["kpbli"])


def test_stage_fixture_records_its_own_fidelity(fixture):
    """The stage decomposition is only usable where it reproduces GFDRV.  208
    of 216 rows do; the 8 that do not are a measured consequence of the
    driver's mixed-precision constants and are marked, not hidden."""
    distrust = stage_rows_to_distrust(fixture)
    assert int(distrust.sum()) == 8
    assert int((~distrust).sum()) == 208
    # No branch flipped anywhere: the residual is a magnitude shift only.
    cons = fixture.consistency
    assert np.array_equal(cons["ktop_direct"], cons["ktop_driver"])


def test_column_independence_holds(fixture):
    """GF at the GFDRV boundary is bitwise column-independent on this fixture,
    which is what makes a one-column-per-thread GPU mapping admissible."""
    import csv as _csv
    from gpuwm.verify.gf_oracle import GF_ORACLE_DIR

    with (GF_ORACLE_DIR / "gf-isolation.csv").open(newline="") as handle:
        rows = list(_csv.DictReader(handle))
    assert len(rows) == GF_NCASE * GF_NDX * 2
    assert all(int(r["ndiff_level_words"]) == 0 for r in rows)
    assert all(int(r["ndiff_surface_words"]) == 0 for r in rows)
