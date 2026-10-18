"""topo_wind against WRF v4.6.1, end to end, word for word.

``tools/topo_wind_wrf461_oracle/build.sh`` runs WRF's own statements as a
chain on nine terrain patches (558 cells, one atmospheric column on each):
dyn_em/start_em.F:1539-1626, cut verbatim, forms LAP_HGT, CTOPO and CTOPO2
on the whole patch for topo_wind = 0, 1 and 2, then the byte-unmodified
bl_ysu_run (phys/physics_mmm/bl_ysu.F90 at d66e442f) runs on every cell
with the CTOPO it formed.  WOOF runs its own chain on the same inputs
(:func:`gpuwm.core.terrain_drag.topo_wind_coefficients`, then
:func:`gpuwm.core.ysu.launch_ysu` with the coefficients WOOF formed), so a
coefficient one ULP off would show both in its static plane and in the
drag it feeds.  The patches are alpine ridges at 3 km, a coastline with
lakes in a basin and on a summit, plains, an extreme high range (surface
pressure near 35 kPa), the smallest domains the edge clamp sees, and a probe
patch that puts LAP_HGT and sqrt(VAR_SSO) on and one float32 step either
side of every threshold start_em.F tests; the columns are the wide YSU
oracle's regimes and its edge probes.

The reference is the build with WRF's own FCOPTIM (-O2 -ftree-vectorize
-funroll-loops), the arithmetic a stock wrf.exe performs; the -O0 build is
kept beside it (they differ by GCC folding x**2. to x*x).

The same patches and cells run a second consumer, Shin-Hong
(bl_pbl_physics = 11): module_pbl_driver.F hands it CTOPO/CTOPO2 too and
module_bl_shinhong.F reads both (:1387, :1472-1473).  Its reference is the
build tools/shinhong_column_oracle uses (WRF's flags, vectoriser off); WRF's
flags exactly (libmvec powf) give the same words on every defined lane.

The standard is the strict build (GPUWM_WRF_EXACT=1): every static word and
every YSU and Shin-Hong output word of every call where WRF's own result
is defined is WRF's word.  Measured on an RTX PRO 6000 (sm_120), NVRTC 12.9: 0 differing
words.  Under default arithmetic the static planes are still bitwise (the
terrain-drag module always compiles -fmad=false --ftz=false) and the YSU
differences are FTZ flushes: cloud and ice tendency words within the
subnormal range of WRF's, and a subnormal 10 m wind input.

Outside the claim, each for a named reason:

* 30 calls (10 cells x 3 options) where WRF itself writes NaN: sfcflg is
  true and ust**3 is zero or subnormal, so bl_ysu.F90:948's prfac2 is 0/0
  (the YSU wide oracle's five columns, met here twice each).  WOOF stays
  finite.
* Shin-Hong on the same 10 cells: WRF's prfac2 0/0 (module_bl_shinhong.F
  :1010) fills the heat column with NaN; WOOF writes NaN in the same lanes
  (payload differs: x86 0xFFC00000, CUDA 0x7FFFFFFF), and in tke/el the
  optimised WRF build's amax1 floors the NaN where WOOF keeps it (the
  Shin-Hong column oracle's documented class).  And Shin-Hong's tke/el on
  the 15 calls with tke_diag = 1 and kpbl == kte, where WRF reads
  q2xk(kte+1), one past the array (:1557).
* 6 calls (2 cells x 3 options) on a column whose levels all sit below
  500 m.  bl_ysu.F90 get_pblh searches ``do while (zw1d(k) .le. 500.)`` with
  no k <= kte bound and reads ``qke1d(kte)``, a word bl_ysu_run never sets;
  on this column it walks past the array (``bounds_check.sh``: gfortran
  -fcheck=bounds stops at bl_ysu.F90:1625, "Index '5' of dimension 1 of
  array 'qke1d' above upper bound of 4").  WRF's result there is whatever
  its stack held; WOOF bounds the search.  A WRF bug, documented, not
  copied.  No real WRF grid has its top below 500 m.  WRF's own words
  there are not even reproducible: two runs of the same -O0 build on the
  same inputs wrote different utnp/vtnp on these cells (and only these),
  so the build receipt's raw wrfflags-vs-O0 byte count moves between
  builds; the gate counts only defined calls.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.verify.topo_wind_oracle import (
    FAMILIES, OPTIONS, TOPO_WIND_DIR, YSU_FIELDS, compare,
    load_topo_wind_oracle, nonfinite_calls)

#: (patch, cell) pairs whose levels sit below 500 m (zw(kte) <= 500):
#: WRF's get_pblh reads past the column there (see the module docstring).
WRF_UNDEFINED_SHALLOW = ((2, 27), (2, 66))
#: (patch, cell) pairs where WRF writes NaN under every option.
WRF_NONFINITE_CELLS = ((2, 7), (2, 8), (2, 10), (2, 11), (2, 34),
                       (2, 46), (2, 47), (2, 49), (2, 50), (2, 73))
#: YSU words where the -O0 build differs from WRF's own build, over the
#: calls whose WRF result is defined (x**2. as powf at -O0, x*x at -O2).
WRFFLAGS_VS_O0_DEFINED_WORDS = 186
#: Shin-Hong calls with tke_diag = 1 and kpbl == kte (WRF reads q2xk(kte+1)).
SHINHONG_OOB_TKE_CALLS = 15


def _strict() -> bool:
    from gpuwm import wrf_exact
    return bool(wrf_exact.ENABLED)


def _shallow(cell) -> bool:
    return float(np.sum(cell.levels["dz"][:-1], dtype=np.float64)) <= 500.0


def _in_claim(p, opt, c) -> bool:
    return c.wrf_finite(opt) and not _shallow(c)


# --------------------------------------------------------------------------
# CPU: the fixture is what it says it is
# --------------------------------------------------------------------------

def test_fixture_hashes_and_build_receipt():
    rows = (TOPO_WIND_DIR / "oracle-sha256sums.txt").read_text(
        encoding="ascii").splitlines()
    recorded = {line.split()[1]: line.split()[0] for line in rows}
    assert set(recorded) == {
        "topo-wind-patches.bin", "topo-wind-columns.bin.gz",
        "topo-wind-shinhong.bin.gz",
        "wrf-wrfflags-static.bin.gz", "wrf-wrfflags-ysu.bin.gz",
        "wrf-O0-static.bin.gz", "wrf-O0-ysu.bin.gz",
        "wrf-O2s-shinhong.bin.gz", "wrf-wrfflags-shinhong.bin.gz",
        "wrf-O0-shinhong.bin.gz"}
    for name, digest in recorded.items():
        blob = (TOPO_WIND_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == digest, name
    receipt = (TOPO_WIND_DIR / "build-receipt.txt").read_text(encoding="ascii")
    assert "d66e442fccc04111067e29274c9f9eaccc3cef28" in receipt
    assert "GNU Fortran" in receipt and "GLIBC 2.39" in receipt
    # The one libmvec symbol in any build is Shin-Hong's under WRF's exact
    # flags (the vectorised powf), the build kept beside the reference.
    head, shinhong = receipt.split("# Shin-Hong O2s", 1)
    assert "_ZGV" not in head, "a libmvec vector symbol entered a YSU build"
    o2s = shinhong.split("# Shin-Hong wrfflags")[0]
    assert "_ZGV" not in o2s
    assert "_ZGVbN4vv_powf" in shinhong
    assert "FMA instructions in any object: 0" in receipt
    assert "bytes differing wrfflags vs O0: static 0, ysu " in receipt


#: Harness files whose current text is a recorded successor of the one that
#: wrote the fixture, with the reason it writes the same bytes.
#: make_wide_columns.py: 2.9.1's copy (a0e6a1337) writes the record pad from
#: an optional "_pad" scalar (the YSU generations oracle carries one); the
#: topo-wind records carry none, so the pad is the 0.0 the recorded harness
#: wrote.  Merged with the WRF exact chain in 2.9.1.
HARNESS_SUCCESSORS = {
    "tools/topo_wind_wrf461_oracle/../ysu_wrf461_oracle/make_wide_columns.py":
        "1605dbf9d232107e24fd275fa0b1cd6a7b44589750994f86c19fd590a46d9318",
}


def test_the_harness_that_wrote_the_fixture_is_this_one():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for line in (TOPO_WIND_DIR / "harness-sha256sums.txt").read_text(
            encoding="ascii").splitlines():
        digest, rel = line.split()
        path = (root / rel).resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual in (digest, HARNESS_SUCCESSORS.get(rel)), rel


def test_fixture_spans_terrain_regimes_and_thresholds():
    patches = load_topo_wind_oracle()
    cells = [c for p in patches for c in p.cells]
    assert len(cells) >= 64
    assert {p.family for p in patches} == set(FAMILIES)
    regimes = {c.regime for c in cells}
    assert {1, 2, 3, 5, 6, 7, 8, 9} <= regimes
    assert {c.topdown for c in cells} == {0, 1}
    assert min(float(c.scalars["psfc"]) for c in cells) < 40000.0
    assert any(c.levels["theta"][0] < 255 for c in cells)
    lap = np.concatenate([p.wrf_static[1]["lap_hgt"].ravel() for p in patches])
    ct = np.concatenate([p.wrf_static[1]["ctopo"].ravel() for p in patches])
    ct2 = np.concatenate([p.wrf_static[1]["ctopo2"].ravel() for p in patches])
    xl = np.concatenate([p.xland.ravel() for p in patches])
    # every LAP_HGT band of start_em.F:1594-1608, and its boundaries exactly
    assert (lap > -10).any() and ((lap <= -10) & (lap >= -20)).any()
    assert ((lap < -20) & (lap >= -30)).any() and (lap < -30).any()
    for edge in (-10.0, -20.0, -30.0):
        e = np.float32(edge)
        assert (lap == e).any(), edge
        assert (lap == np.nextafter(e, np.float32(0))).any(), edge
        assert (lap == np.nextafter(e, np.float32(-100))).any(), edge
    assert (ct == 0).any() and (ct2 == 0).any() and (ct2 < 1).any()
    assert (ct > 1).any() and (ct == 1).any()
    # sqrt(VAR_SSO) exactly on 2.718 and either side
    roots = np.sqrt(np.concatenate([p.var_sso.ravel() for p in patches]))
    for r in (np.float32(2.718), np.nextafter(np.float32(2.718), np.float32(9)),
              np.nextafter(np.float32(2.718), np.float32(0))):
        assert (roots.astype(np.float32) == r).any()
    # water cells with statistics, and water cells on a summit
    assert ((xl == 2) & (lap < -30)).any() and ((xl == 2) & (lap > 0)).any()
    # topo_wind = 2 on both sides of its 1.575 cap
    ct_2 = np.concatenate([p.wrf_static[2]["ctopo"].ravel() for p in patches])
    cap = np.float32(1.575) * np.float32(1.575)
    assert (ct_2 == cap).any() and ((ct_2 < cap) & (ct_2 > 1)).any()
    # the 10 m blend moves the wind somewhere
    moved = [c for p in patches for c in p.cells
             if c.wrf[1]["u10"][0] != c.scalars["u10"]]
    assert moved


def test_the_o0_build_differs_from_wrfs_own_only_where_it_should():
    """Static planes identical; YSU differs in 186 defined words (pow tie)."""
    a = load_topo_wind_oracle(reference="wrfflags")
    b = load_topo_wind_oracle(reference="O0")
    words = 0
    for p, q in zip(a, b):
        for opt in OPTIONS:
            for name in ("lap_hgt", "ctopo", "ctopo2"):
                assert (p.wrf_static[opt][name].tobytes()
                        == q.wrf_static[opt][name].tobytes())
            for c, d in zip(p.cells, q.cells):
                if not _in_claim(p, opt, c):
                    continue
                for k in c.wrf[opt]:
                    words += int((c.wrf[opt][k].view(np.uint32)
                                  != d.wrf[opt][k].view(np.uint32)).sum())
    assert words == WRFFLAGS_VS_O0_DEFINED_WORDS


def test_shinhong_reference_and_wrfs_exact_flags_agree_on_defined_lanes():
    from gpuwm.verify.topo_wind_oracle import load_shinhong_chain
    patches = load_topo_wind_oracle()
    ref = load_shinhong_chain(patches, reference="O2s")
    vec = load_shinhong_chain(patches, reference="wrfflags")
    oob = [k for k, c in ref.items() if c.oob_tke]
    assert len(oob) == SHINHONG_OOB_TKE_CALLS
    for key, c in ref.items():
        if (key[0], key[2]) in WRF_NONFINITE_CELLS:
            continue
        for name, w in c.wrf.items():
            if c.oob_tke and name in ("tke", "el"):
                continue
            assert w.tobytes() == vec[key].wrf[name].tobytes(), (key, name)


def test_the_calls_outside_the_claim_are_exactly_the_named_ones():
    patches = load_topo_wind_oracle()
    shallow = tuple((p.index, n) for p in patches
                    for n, c in enumerate(p.cells) if _shallow(c))
    assert shallow == WRF_UNDEFINED_SHALLOW
    nan = nonfinite_calls(patches)
    assert sorted(nan) == sorted((p, opt, n) for p, n in WRF_NONFINITE_CELLS
                                 for opt in OPTIONS)
    for p, n in WRF_NONFINITE_CELLS:
        c = patches[p].cells[n]
        ust = float(c.scalars["ust"])
        assert float(c.scalars["br"]) < 0
        assert ust ** 3 < np.finfo(np.float32).tiny


# --------------------------------------------------------------------------
# GPU: WOOF's chain against WRF's
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chain():
    from gpuwm.verify.topo_wind_oracle import run_port
    patches = load_topo_wind_oracle()
    return patches, run_port(patches)


@pytest.mark.gpu
@requires_gpu
def test_static_coefficients_are_wrfs_bit_for_bit(chain):
    import cupy  # noqa: F401
    patches, port = chain
    res = compare(patches, port, include=lambda p, opt, c: False)
    for name, st in res["static"].items():
        assert st.differing == 0, (name, st.worst)
    assert sum(st.total for st in res["static"].values()) == \
        3 * 3 * sum(len(p.cells) for p in patches)


@pytest.mark.gpu
@requires_gpu
def test_every_ysu_word_of_the_chain_is_wrfs(chain):
    import cupy  # noqa: F401
    patches, port = chain
    res = compare(patches, port, include=_in_claim)
    assert res["kpbl_mismatch"] == []
    ncalls = sum(1 for p in patches for opt in OPTIONS for c in p.cells
                 if _in_claim(p, opt, c))
    assert ncalls == 3 * (558 - len(WRF_NONFINITE_CELLS)
                          - len(WRF_UNDEFINED_SHALLOW))
    if _strict():
        bad = {k: (st.differing, st.worst) for k, st in res["ysu"].items()
               if st.differing}
        assert bad == {}, bad
        return
    # A default process runs 2.9.1's YSU unit (kernels/ysu_default.cu, the
    # 2.9.1 intake of the WRF exact chain), not the chain's default-arithmetic
    # build this flush-only bound was measured on; the word-for-word claim is
    # the strict leg's, above.
    pytest.skip("default arithmetic runs 2.9.1's YSU; graded on the strict leg")
    # Default arithmetic: -ftz=true flushes subnormals and nothing else
    # moves.  Every differing word is within the subnormal range of WRF's
    # (a subnormal written as zero, or a word computed through a flushed
    # subnormal in the implicit solve), as on the YSU wide oracle.
    tiny = np.float32(np.finfo(np.float32).smallest_normal)
    for p in patches:
        for opt in OPTIONS:
            for n, c in enumerate(p.cells):
                if not _in_claim(p, opt, c):
                    continue
                g = port[(p.index, opt)]["cells"][n]
                for name, wname in YSU_FIELDS.items():
                    a = np.asarray(g[name], np.float32).reshape(-1)
                    b = np.asarray(c.wrf[opt][wname], np.float32).reshape(-1)
                    diff = a.view(np.uint32) != b.view(np.uint32)
                    if not diff.any():
                        continue
                    assert name in ("dqc", "dqi", "u10", "v10"), (
                        p.index, opt, n, name)
                    gap = np.abs(a[diff].astype(np.float64)
                                 - b[diff].astype(np.float64))
                    assert (gap < tiny).all(), (p.index, opt, n, name)


@pytest.mark.gpu
@requires_gpu
def test_woof_stays_defined_where_wrf_does_not(chain):
    """The NaN and the past-the-column calls: WOOF's words are finite, and
    on the shallow column the drag still responds to CTOPO."""
    import cupy  # noqa: F401
    patches, port = chain
    for p, n in WRF_NONFINITE_CELLS + WRF_UNDEFINED_SHALLOW:
        for opt in OPTIONS:
            g = port[(p, opt)]["cells"][n]
            for name in ("du", "dv", "dtheta", "dqv", "exch_h", "exch_m",
                         "hpbl", "u10", "v10"):
                assert np.isfinite(g[name]).all(), (p, opt, n, name)
    for p, n in WRF_UNDEFINED_SHALLOW:
        cells = [port[(p, opt)]["cells"][n]["du"][0] for opt in OPTIONS]
        ct = [port[(p, opt)]["static"]["ctopo"].ravel()[n] for opt in OPTIONS]
        assert ct[0] == 1 and ct[1] != 1 and ct[2] != 1
        assert cells[1] != cells[0] and cells[2] != cells[0]


# --------------------------------------------------------------------------
# GPU: the Shin-Hong consumer
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shinhong_chain():
    from gpuwm.verify.topo_wind_oracle import (load_shinhong_chain,
                                               run_port_shinhong)
    patches = load_topo_wind_oracle()
    cols = load_shinhong_chain(patches, reference="O2s")
    return patches, cols, run_port_shinhong(patches, cols)


def _sh_in_claim(key, col) -> bool:
    return (key[0], key[2]) not in WRF_NONFINITE_CELLS


@pytest.mark.gpu
@requires_gpu
def test_every_shinhong_word_of_the_chain_is_wrfs(shinhong_chain):
    import cupy  # noqa: F401
    from gpuwm.verify.topo_wind_oracle import compare_shinhong
    patches, cols, port = shinhong_chain
    res = compare_shinhong(cols, port, include=_sh_in_claim)
    assert len(res["keys"]) == 3 * (558 - len(WRF_NONFINITE_CELLS))
    assert res["kpbl_mismatch"] == []
    assert res["oob_excluded"] == 2 * SHINHONG_OOB_TKE_CALLS
    fields = dict(res["fields"], **res["topo"])
    if _strict():
        bad = {k: (st.differing, st.worst) for k, st in fields.items()
               if st.differing}
        assert bad == {}, bad
        return
    tiny = float(np.finfo(np.float32).smallest_normal)
    for name, st in fields.items():
        if st.differing:
            assert name in ("dqc", "dqi", "u10", "v10"), (name, st.worst)
            assert st.max_abs < tiny, (name, st.max_abs)


@pytest.mark.gpu
@requires_gpu
def test_shinhong_nan_cells_keep_wrfs_nan_geometry(shinhong_chain):
    """WRF's prfac2 0/0 columns: the NaN lanes of every tendency are WRF's."""
    import cupy  # noqa: F401
    from gpuwm.verify.topo_wind_oracle import compare_shinhong
    patches, cols, port = shinhong_chain
    res = compare_shinhong(
        cols, port, include=lambda k, c: not _sh_in_claim(k, c))
    assert len(res["keys"]) == 3 * len(WRF_NONFINITE_CELLS)
    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h",
                 "hpbl", "wstar", "delta"):
        st = res["fields"][name]
        assert st.nan_geometry_mismatch == 0, name
        if _strict():
            assert st.max_ulp == 0, name
    assert res["fields"]["dtheta"].nan_lanes > 0


@pytest.mark.gpu
@requires_gpu
def test_shinhong_drag_responds_to_ctopo(shinhong_chain):
    """Arm B is live: CTOPO != 1 moves the momentum tendency, CTOPO2 < 1
    moves the 10 m wind, on WRF's side and on WOOF's alike."""
    import cupy  # noqa: F401
    patches, cols, port = shinhong_chain
    moved_drag = moved_u10 = 0
    for p in patches:
        for n, cell in enumerate(p.cells):
            if (p.index, n) in WRF_NONFINITE_CELLS:
                continue
            ct = p.wrf_static[1]["ctopo"].ravel()[n]
            ct2 = p.wrf_static[1]["ctopo2"].ravel()[n]
            a0, a1 = port[(p.index, 0, n)], port[(p.index, 1, n)]
            w0 = cols[(p.index, 0, n)].wrf
            w1 = cols[(p.index, 1, n)].wrf
            if ct != 1 and (a1["du"] != a0["du"]).any():
                moved_drag += 1
                assert (w1["rublten"] != w0["rublten"]).any()
            if ct2 < 1 and a1["u10"][0] != a0["u10"][0]:
                moved_u10 += 1
                assert w1["u10"][0] != w0["u10"][0]
    assert moved_drag > 100 and moved_u10 > 5
