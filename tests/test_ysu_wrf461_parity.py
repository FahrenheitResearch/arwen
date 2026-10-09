"""YSU against the byte-unmodified WRF v4.6.1 module, for the first time.

``bl_pbl_physics=1`` is the PBL scheme in every registry template, and until
this file existed no WRF number had ever been produced for it: the CUDA kernel
was checked only against ``gpuwm.verify.npref.np_ysu_column``, a float64 mirror
of the same transcription, so a misread line in the transcription agreed with
itself.  ``tools/ysu_wrf461_oracle/`` now drives ``bl_ysu_run`` in the pinned
tree and dumps every input and output; this module measures the kernel against
that dump and pins what it measures.

**The numbers below are a measurement, not a target.**  They record what the
shipped kernel actually does, so that it cannot change without someone saying
so in a commit.  They are asserted for *equality*, not as an upper bound: a
kernel that got better silently is as much a drift as one that got worse, and
the whole point of this file is that the residual is documented.  Every
baseline here was produced twice -- on the local RTX 5090 (Windows driver) and
on a rented RTX 5090 (Linux, CUDA 12.9, cupy 14.1.1) -- and every maximum was
identical on both.  The *count* of differing lanes was not, so no count is
pinned.

Two fixture columns are held out of the ULP table and asserted separately, by
:func:`test_the_two_documented_behavioural_divergences_are_exactly_these`.
They do not measure arithmetic; each is a place where the port and WRF take
different branches, and folding a branch disagreement into a ULP maximum hides
it behind a big number:

* case 12 -- ``ust``, ``hfx`` and ``qfx`` are all exactly zero.  ``ysu.cu``
  returns zero tendencies; WRF has no such short circuit and computes a
  703 m PBL with nine levels in it.
* case 13 -- ``ust`` is a subnormal, so ``-ftz`` makes case 12's short circuit
  fire on a nonzero input.  WRF meanwhile divides 0 by 0 in ``prfac2`` and
  fills the column with NaN.

CLOSED in lane/parity-286, and now inside the ordinary table:

* case 7  -- ``br`` is the smallest positive subnormal.  WRF's ``br > 0``
  (bl_ysu.F90:613) is true and the column is stable; sm_120 DAZes the
  subnormal in every float32 compare, so the kernel's old ``br <= 0.0f`` was
  also true and the column went convective (wstar 0.2362 and delta 18.49 where
  WRF writes 0, hpbl 1525 ULP, momentum 4.5e7 ULP, theta 1.8e9 ULP).  The
  compare is now made in double through the bit-decoding ``ysu_f2d``
  (ysu_topo.cuh, the Shin-Hong ``sh_f2d`` fix), and case 7 measures inside
  the table below with kpbl equal to WRF's.
* the surface-drag arm -- WRF's driver passes ``ctopo = ctopo2 = 1`` on every
  default run (``module_bl_ysu.F:404``), so ``bl_ysu.F90:1308`` is the drag
  diagonal WRF computes; the kernel used the ctopo-absent ``:1315``
  (``1+fric``), which no WRF run reaches.  Every entry point now takes the
  ported ctopo arm, so ``du``/``dv`` are graded against ``utnp_ctopo``/
  ``vtnp_ctopo``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.ysu_oracle import (
    LEVEL_FIELD_MAP,
    NOCTOPO_FIELD_MAP,
    SURFACE_FIELD_MAP,
    YSU_ORACLE_DIR,
    load_ysu_oracle,
    ysu_port_outputs,
)

#: Columns whose disagreement with WRF is a branch, not a rounding.  Asserted
#: one at a time below instead of being averaged into a ULP maximum.
BRANCH_DIVERGENCE_CASES = (12, 13)

#: Worst ULP distance from ``kernels/ysu.cu`` to the word ``bl_ysu_run`` wrote,
#: over the 22 columns that take the same branches as WRF.  This is the kernel
#: as it ships, and since lane/parity-286 it is WRF's word on every lane except
#: the ones the card's flush-to-zero reaches (``SUBNORMAL_LANES``): the dqv,
#: dqc and dqi maxima below are those lanes, where WRF wrote a subnormal
#: tendency and the -ftz kernel writes exactly zero
#: (test_ysu_is_bitwise_wrf_outside_the_flushed_subnormal_lanes).
#:
#: Measured on an RTX PRO 6000 (sm_120) under NVRTC 13.4.92 (cupy-cuda13x)
#: and NVRTC 12.9.86 (cupy-cuda12x), identical.  The history of the residue,
#: each step isolated by changing one expression and re-measuring:
#:
#:   dtheta 884345697 -> 1    ``(f1 - thx + 300)``, WRF's association
#:                            (bl_ysu.F90:1103), not ``(rhs + 300 - theta)``.
#:   hpbl 112 -> 1, exch 283/48 -> 7, dv 46604 -> 23302
#:                            ``ep1 = RV/RD - 1.0f``, WRF's float32 EP_1.
#:   everything -> 0          lane/parity-286 (sweep row 10): glibc's powf and
#:                            expf (gfk_pow/gfk_exp from glibc_flt32.cuh) in
#:                            place of CUDA's, every ``x**r`` with a REAL
#:                            exponent spelled as the powf call gfortran makes
#:                            (ust**3., wscale**4., zfac**pfac, entfac's
#:                            **2., prnumfac's **2.), wstar/wscale as
#:                            ``**h1`` (WRF's powf, not a cube root), WRF's
#:                            association for chi, temps and prnumfac, and the
#:                            unit compiled with --fmad=false (WRF's reference
#:                            is gfortran -O0, no contraction).  One missed
#:                            ``wscale**4.`` was worth dv 91 ULP on cases 1
#:                            and 21 by itself.
#:   dtheta 1 -> 0            WRF's YSU reads thx = (th*pi)/pi, never the
#:                            model's theta (phy_prep's t_phy = th*pi, then
#:                            bl_ysu.F90:419), and hands the solver
#:                            (ttend*pi2d)/pi2d (module_bl_ysu.F:452); the
#:                            kernel now spells both round trips (ysu_thx).
BASELINE_MAX_ULP = {
    "du": 0,
    "dv": 0,
    "dtheta": 0,
    "dqv": 15070,
    "dqc": 207470,
    "dqi": 30808,
    "exch_h": 0,
    "exch_m": 0,
    "hpbl": 0,
    "wstar": 0,
    "delta": 0,
}

#: The same port momentum against WRF's ctopo-ABSENT call (``ad(1) = 1+fric``,
#: bl_ysu.F90:1315), the arm the kernel used to take and no WRF run reaches.
#: Kept so that the move onto WRF's default arm stays visible: the port is
#: bitwise WRF's ctopo arm and exactly WRF_CTOPO_GAP_MAX_ULP from this one
#: (test_ysu_takes_wrfs_default_ctopo_drag_arm).
NOCTOPO_BASELINE_MAX_ULP = {"du": 182, "dv": 182}

#: The distance between WRF's two arms, measured WRF against WRF with no port
#: involved: the same call with and without ``ctopo``.
WRF_CTOPO_GAP_MAX_ULP = {"utnp": 182, "vtnp": 182}

#: Every lane where ``bl_ysu_run`` wrote a subnormal.  CuPy appends
#: ``-ftz=true`` unconditionally, so the kernel writes exactly zero in all of
#: them.  Pinned as a count so that neither the flush nor the fixture's ability
#: to reach it can quietly disappear.
SUBNORMAL_LANES = {"dqv": 5, "dqc": 5, "dqi": 1}


def _fixture():
    return load_ysu_oracle()


def _arithmetic_mask(fixture) -> np.ndarray:
    return np.asarray([c not in BRANCH_DIVERGENCE_CASES for c in fixture.cases])


def _measure(fixture, port, mask) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, column in LEVEL_FIELD_MAP.items():
        got = np.ascontiguousarray(port[name][:, :, mask], np.float32)
        want = np.ascontiguousarray(fixture.level_reference[column][:, :, mask])
        out[name] = int(fp32_ulp_distance(got, want).max())
    for name, column in SURFACE_FIELD_MAP.items():
        got = np.ascontiguousarray(port[name][:, mask], np.float32)
        want = np.ascontiguousarray(fixture.surface_reference[column][:, mask])
        out[name] = int(fp32_ulp_distance(got, want).max())
    return out


# --------------------------------------------------------------------------
# CPU-only: the fixture is what it says it is.
# --------------------------------------------------------------------------

def test_fixture_is_the_pinned_wrf_module_and_its_own_receipts():
    """The CSVs must hash to what build.sh recorded next to them."""
    receipts = (YSU_ORACLE_DIR / "oracle-sha256sums.txt").read_text(
        encoding="ascii").splitlines()
    recorded = {}
    for line in receipts:
        digest, path = line.split(maxsplit=1)
        recorded[path.strip().rsplit("/", 1)[-1]] = digest
    for name in ("ysu-levels.csv", "ysu-surface.csv"):
        blob = (YSU_ORACLE_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == recorded[name], name
    # The two WRF sources the fixture came from are named in the same receipt,
    # so a fixture regenerated from a different tree cannot pass silently.
    assert "bl_ysu.F90" in " ".join(receipts)
    assert "ccpp_kind_types.F" in " ".join(receipts)


def test_libmvec_receipt_shows_the_reference_is_scalar_libm():
    """gfortran's vector libm must not be in the reference, and the grep for it
    must be capable of firing -- the report carries its own positive control."""
    report = (YSU_ORACLE_DIR / "libmvec-report.txt").read_text(encoding="ascii")
    reference, _, rest = report.partition("# -O2")
    assert "_ZGV" not in reference, report
    assert "_ZGVbN4v_expf" in rest, (
        "the positive control produced no vector symbol, so the absence of one"
        " in the -O0 reference proves nothing")
    assert "U expf" in reference and "U powf" in reference, report


def test_fixture_covers_the_branches_that_matter():
    fixture = _fixture()
    assert fixture.nz == 40 and fixture.ncase == 24
    br = fixture.inputs["br"].reshape(-1)
    smallest = np.float32(np.finfo(np.float32).smallest_subnormal)
    # both signed zeros, both signed smallest subnormals, the smallest normal
    assert (br == 0).sum() >= 2 and (np.signbit(br) & (br == 0)).any()
    assert (br == smallest).any() and (br == -smallest).any()
    assert (br == np.float32(np.finfo(np.float32).smallest_normal)).any()
    # a subnormal ust and a subnormal qfx, for the -ftz branch probes
    assert (fixture.inputs["ust"].reshape(-1) == smallest).any()
    assert (fixture.inputs["qfx"].reshape(-1) == smallest).any()
    # WRF's cloud test is `.gt. 0.01e-3`; the fixture straddles it exactly
    qc = fixture.inputs["qc"]
    assert (qc == np.float32(0.01e-3)).any()
    assert (qc == np.nextafter(np.float32(0.01e-3), np.float32(1))).any()
    # both values of ysu_topdown_pblmix, and a PBL that fills the column
    assert set(fixture.topdown) == {0, 1}
    kpbl = fixture.kpbl_reference.reshape(-1)
    assert kpbl.min() <= 2 and kpbl.max() == fixture.nz


def test_wrfs_default_ctopo_arm_differs_from_the_ctopo_absent_arm():
    """A WRF-against-WRF number: no GPU, no port, no tolerance to argue about.

    ``module_bl_ysu.F:404`` always passes ``ctopo``/``ctopo2``, and the
    Registry default fills both with 1.0, so ``bl_ysu.F90:1308`` -- not
    ``:1315`` -- is the surface-drag diagonal of every default WRF run.  The
    kernel takes that arm (the paj TKE block, ``get_pblh`` and the Beljaars
    ``vconv`` in ysu_topo.cuh) on every entry point.  This pins that the two
    arms really differ on this fixture, so the gate below is not vacuous.
    """
    fixture = _fixture()
    for plain, ctopo in (("utnp", "utnp_ctopo"), ("vtnp", "vtnp_ctopo")):
        distance = fp32_ulp_distance(fixture.level_reference[plain],
                                     fixture.level_reference[ctopo])
        assert int(distance.max()) == WRF_CTOPO_GAP_MAX_ULP[plain], (
            f"{plain}: ctopo gap moved to {int(distance.max())}")
        assert int(distance.max()) > 0, (
            "if the two arms agree the fixture stopped reaching vconvlim < 1"
            " and this test has become vacuous")


# --------------------------------------------------------------------------
# GPU: the measurement itself.
# --------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_ysu_cuda_column_holds_its_measured_distance_from_wrf():
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    measured = _measure(fixture, port, _arithmetic_mask(fixture))
    assert measured == BASELINE_MAX_ULP, (
        "YSU's distance from the unmodified WRF module changed.\n"
        f"  measured {measured}\n  recorded {BASELINE_MAX_ULP}\n"
        "If a field got worse, something regressed.  If it got better, say so:"
        " update the table in the same commit as the improvement, with the"
        " attribution, so the residual stays documented.")


@pytest.mark.gpu
@requires_gpu
def test_ysu_is_bitwise_wrf_outside_the_flushed_subnormal_lanes():
    """Every word the kernel writes on the 22 arithmetic columns is the word
    ``bl_ysu_run`` (and module_bl_ysu.F:452 for theta) wrote, except where
    WRF wrote a subnormal tendency, which the -ftz kernel writes as exactly
    zero (test_ftz_flushes_every_subnormal_tendency_wrf_wrote pins those).

    This replaced the dtheta round-trip test: the last dtheta ULP was WRF's
    (ttnp*pi2d)/pi2d, and the kernel now spells that round trip (ysu_thx).
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    mask = _arithmetic_mask(fixture)
    smallest_normal = np.float32(np.finfo(np.float32).smallest_normal)
    for name, column in LEVEL_FIELD_MAP.items():
        want = np.ascontiguousarray(fixture.level_reference[column][:, :, mask])
        got = np.ascontiguousarray(port[name][:, :, mask], np.float32)
        flushed = (np.abs(want) > 0) & (np.abs(want) < smallest_normal)
        np.testing.assert_array_equal(
            got.view(np.uint32)[~flushed], want.view(np.uint32)[~flushed],
            err_msg=name)
    for name, column in SURFACE_FIELD_MAP.items():
        want = np.ascontiguousarray(fixture.surface_reference[column][:, mask])
        got = np.ascontiguousarray(port[name][:, mask], np.float32)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32),
                                      err_msg=name)


@pytest.mark.gpu
@requires_gpu
def test_ysu_kpbl_matches_wrf_exactly_outside_the_short_circuit():
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    mask = _arithmetic_mask(fixture)
    got = np.asarray(port["kpbl"]).reshape(-1)[mask]
    want = fixture.kpbl_reference.reshape(-1)[mask]
    np.testing.assert_array_equal(got, want)


@pytest.mark.gpu
@requires_gpu
def test_ysu_momentum_is_this_far_from_wrfs_ctopo_absent_arm():
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    mask = _arithmetic_mask(fixture)
    for name, column in NOCTOPO_FIELD_MAP.items():
        got = np.ascontiguousarray(port[name][:, :, mask], np.float32)
        want = np.ascontiguousarray(fixture.level_reference[column][:, :, mask])
        assert int(fp32_ulp_distance(got, want).max()) == \
            NOCTOPO_BASELINE_MAX_ULP[name], name


@pytest.mark.gpu
@requires_gpu
def test_ysu_takes_wrfs_default_ctopo_drag_arm():
    """Per column, the port is never farther from WRF's ctopo arm than from the
    ctopo-absent one, and strictly closer somewhere.

    Measured at the change (box F RTX 5090, sm_120, NVRTC 13.4): case 9 dv
    22 -> 0 ULP and case 7 dv 22 -> 0 against the ctopo arm; case 18 reads
    du/dv 91 ULP against the ctopo arm and 273 against the absent one.  Before
    the change the kernel sat on the absent arm (case 9 dv 0 against it, 22
    against WRF's default).  With the row-10 arithmetic closed the port is 0
    ULP from the ctopo arm on every column and 182 from the absent one.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    mask = _arithmetic_mask(fixture)
    strictly = False
    for name in ("du", "dv"):
        got = np.ascontiguousarray(port[name], np.float32)
        on = fp32_ulp_distance(
            got, fixture.level_reference[LEVEL_FIELD_MAP[name]])
        off = fp32_ulp_distance(
            got, fixture.level_reference[NOCTOPO_FIELD_MAP[name]])
        for i in np.flatnonzero(mask):
            assert int(on[..., i].max()) <= int(off[..., i].max()), (
                name, fixture.cases[i], int(on[..., i].max()),
                int(off[..., i].max()))
            strictly |= int(on[..., i].max()) < int(off[..., i].max())
    assert strictly, "the two arms no longer separate the port on any column"


@pytest.mark.gpu
@requires_gpu
def test_ftz_flushes_every_subnormal_tendency_wrf_wrote():
    """``-ftz=true`` is appended by CuPy and cannot be turned off from here.

    Asserted as an equality in both directions: the fixture must still reach
    subnormal tendencies at all, and the kernel must still be writing exactly
    zero for all of them.  If either changes the flush has been fixed or the
    fixture has stopped exercising it, and both deserve a commit message.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    smallest_normal = np.float32(np.finfo(np.float32).smallest_normal)
    found = {}
    for name, column in LEVEL_FIELD_MAP.items():
        want = fixture.level_reference[column]
        subnormal = (np.abs(want) > 0) & (np.abs(want) < smallest_normal)
        if not subnormal.any():
            continue
        found[name] = int(subnormal.sum())
        got = np.ascontiguousarray(port[name], np.float32)
        assert np.all(got[subnormal] == 0), (
            f"{name}: a subnormal lane survived -ftz, which would be news")
    assert found == SUBNORMAL_LANES, found


@pytest.mark.gpu
@requires_gpu
def test_the_two_documented_behavioural_divergences_are_exactly_these():
    """The held-out columns, stated as what they are rather than as ULP.

    Each assertion is the divergence written down.  If one is ever closed this
    test fails, which is the intended way to find out.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    index = {case: i for i, case in enumerate(fixture.cases)}

    # cases 12 and 13: ysu.cu:250 short circuits where WRF runs the scheme.
    for case, wrf_kpbl in ((12, 9), (13, 2)):
        i = index[case]
        assert int(port["kpbl"].reshape(-1)[i]) == 1
        assert int(fixture.kpbl_reference.reshape(-1)[i]) == wrf_kpbl
        assert float(port["hpbl"].reshape(-1)[i]) == float(
            fixture.inputs["dz"][0, 0, i]), (
            "the short circuit reports the first layer depth as the PBL top")
        for name in ("du", "dv", "dtheta", "dqv", "exch_h", "exch_m"):
            assert np.all(port[name][:, 0, i] == 0), (case, name)
        assert not np.all(fixture.level_reference["utnp"][:, 0, i] == 0), (
            f"case {case}: WRF produced tendencies here, so the short circuit"
            " is a divergence and not an agreement")

    # case 13 also pins WRF's own 0/0: ust**3 underflows to zero while
    # wstar3 is zero and sfcflg is true, so prfac2 is 0/0 and the column
    # fills with NaN.  gpuwm implements the defined behaviour instead.
    i = index[13]
    wrf_nan = ~np.isfinite(fixture.level_reference["rthblten"][:, 0, i])
    assert wrf_nan.sum() == fixture.nz, (
        "WRF's NaN column disappeared; the 0/0 in prfac2 was the point")
    assert np.all(np.isfinite(port["dtheta"][:, 0, i]))


@pytest.mark.gpu
@requires_gpu
def test_a_positive_subnormal_br_takes_wrfs_stable_arm():
    """Case 7, closed: bl_ysu.F90:613 ``if(br(i).gt.0.0) sfcflg = .false.``.

    The smallest positive subnormal br must read as stable, as it does in
    WRF, although sm_120 DAZes it in every float32 compare.  Before the fix the
    kernel wrote wstar 0.2362 and delta 18.49 here where WRF writes 0.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    i = {case: n for n, case in enumerate(fixture.cases)}[7]
    assert fixture.inputs["br"].reshape(-1)[i] == np.float32(
        np.finfo(np.float32).smallest_subnormal)
    assert 7 not in BRANCH_DIVERGENCE_CASES
    for name in ("wstar", "delta"):
        assert float(fixture.surface_reference[name].reshape(-1)[i]) == 0.0
        assert float(port[name].reshape(-1)[i]) == 0.0, name
    assert int(port["kpbl"].reshape(-1)[i]) == int(
        fixture.kpbl_reference.reshape(-1)[i])
