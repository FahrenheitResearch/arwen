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

Three fixture columns are held out of the ULP table and asserted separately, by
:func:`test_the_three_documented_behavioural_divergences_are_exactly_these`.
They do not measure arithmetic; each is a place where the port and WRF take
different branches, and folding a branch disagreement into a ULP maximum hides
it behind a big number:

* case 7  -- ``br`` is the smallest positive subnormal.  WRF's ``br > 0`` is
  true and the column is stable; CuPy's unconditional ``-ftz=true`` flushes the
  subnormal so the kernel's ``br <= 0.0f`` is also true and the column is
  convective.  Two different regimes, not two roundings.
* case 12 -- ``ust``, ``hfx`` and ``qfx`` are all exactly zero.  ``ysu.cu``
  returns zero tendencies; WRF has no such short circuit and computes a
  703 m PBL with nine levels in it.
* case 13 -- ``ust`` is a subnormal, so ``-ftz`` makes case 12's short circuit
  fire on a nonzero input.  WRF meanwhile divides 0 by 0 in ``prfac2`` and
  fills the column with NaN.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.ysu_oracle import (
    CTOPO_FIELD_MAP,
    LEVEL_FIELD_MAP,
    SURFACE_FIELD_MAP,
    YSU_ORACLE_DIR,
    load_ysu_oracle,
    ysu_port_outputs,
)

#: Columns whose disagreement with WRF is a branch, not a rounding.  Asserted
#: one at a time below instead of being averaged into a ULP maximum.
BRANCH_DIVERGENCE_CASES = (7, 12, 13)

#: Worst ULP distance from ``kernels/ysu.cu`` to the word ``bl_ysu_run`` wrote,
#: over the 21 columns that take the same branches as WRF.  This is the kernel
#: as it ships.  Each cause below was isolated by patching exactly one
#: expression in ysu.cu and re-measuring on both GPUs, in
#: tools/ysu_wrf461_oracle:
#:
#:   dtheta 884345697 -> 1    CLOSED.  ``(rhs + 300 - theta)`` is now spelled
#:                            WRF's way, ``(f1 - thx + 300)``
#:                            (bl_ysu.F90:1103).  The tendency is a difference
#:                            of two ~300 K numbers, so the two associations
#:                            differed by one ULP of theta = 3.4e-07 K/s after
#:                            rdt -- which is ~8.8e8 ULP of a tendency WRF
#:                            rounds to exactly zero.  The 1 ULP that is left
#:                            is WRF's own ttnp*pi2d / pi2d round trip, proved
#:                            lane by lane in
#:                            test_the_last_dtheta_ulp_is_wrfs_own_pi2d_round_trip
#:                            -- so the kernel's theta tendency IS WRF's, and
#:                            the residue is WRF undoing its own multiply.
#:   hpbl   112 -> 1          CLOSED.  ``ep1`` is now ``RV/RD - 1.0f`` rather
#:   exch_h 283 -> 7          than ``RVOVRD - 1.0f``: CUDA_DEFINES["RVOVRD"] is
#:   exch_m  48 -> 7          RV/RD in Python *doubles*, rounded once to
#:   dv   46604 -> 23302      float32, and WRF's EP_1 is the float32 quotient.
#:                            They differ by 1 ULP at 1.608, so ep1 differed by
#:                            2 ULP at 0.608, and thv, the Richardson profile
#:                            and the whole PBL diagnosis inherited it.  The
#:                            right-hand column is what is pinned below.
#:   nothing                  ``ust**3.`` spelled as powf rather than us*us*us;
#:                            cbrtf rather than powf(x, 1/3f); WRF's exact
#:                            association for fric, xkzm, prnumfac and entfac.
#:                            All four measured; all four worth zero here.
#:
#: What no respelling reaches is CUDA's expf/powf against glibc's,
#: amplified by the implicit solve: exch_h/exch_m stay 7 ULP (2.4e-04 m2/s) and
#: the momentum and moisture tendencies stay 1457/23302 ULP -- but only
#: 4.2e-08 m/s2 and 3.1e-11 kg/kg/s in absolute terms, because those tendencies
#: are themselves near-total cancellations.
BASELINE_MAX_ULP = {
    "du": 1457,
    "dv": 23302,
    "dtheta": 1,
    "dqv": 23302,
    "dqc": 207470,
    "dqi": 30808,
    "exch_h": 7,
    "exch_m": 7,
    "hpbl": 1,
    "wstar": 1,
    "delta": 1,
}

#: Same measurement against the momentum tendencies from WRF's *own* driver
#: path -- ``ctopo = ctopo2 = 1``, which ``module_bl_ysu.F:404`` passes on every
#: column of every default run.  ``ysu.cu`` implements the other arm
#: (``ad(i,1) = 1+fric``) and has no ``ctopo`` argument at all.
CTOPO_BASELINE_MAX_ULP = {"du": 1457, "dv": 23302}

#: The cost of that missing arm, measured WRF against WRF with no port
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


def test_the_ctopo_arm_wrf_actually_uses_is_not_the_arm_the_port_implements():
    """A WRF-against-WRF number: no GPU, no port, no tolerance to argue about.

    ``module_bl_ysu.F:404`` always passes ``ctopo``/``ctopo2``, and the
    Registry default fills both with 1.0, so ``bl_ysu.F90:1308`` -- not
    ``:1315`` -- is the surface-drag diagonal of every default WRF run.  That
    arm needs the paj TKE block, ``get_pblh`` and the Beljaars ``vconv``, none
    of which ``ysu.cu`` has.  This is what it costs on this fixture.
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
def test_the_last_dtheta_ulp_is_wrfs_own_pi2d_round_trip():
    """``dtheta`` is bitwise WRF's tendency; the 1 ULP is WRF undoing itself.

    ``bl_ysu.F90:1103`` returns ``ttnp = (f1-thx+300)*rdt*pi2d`` and
    ``module_bl_ysu.F:452`` immediately hands the solver ``ttnp/pi2d``.  The
    kernel produces the un-multiplied quantity directly, so it cannot match the
    round-tripped one on every lane.  Push the kernel's answer back through the
    same multiply and it must land on ``ttnp`` exactly -- which says the
    remaining ULP is WRF's own multiply-then-divide and not a transcription
    difference.  Asserted for *every* differing lane, so one lane failing to
    be explained this way fails the test.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    mask = _arithmetic_mask(fixture)
    got = np.ascontiguousarray(port["dtheta"][:, :, mask], np.float32)
    want = np.ascontiguousarray(fixture.level_reference["rthblten"][:, :, mask])
    ttnp = np.ascontiguousarray(fixture.level_reference["ttnp"][:, :, mask])
    exner = np.ascontiguousarray(fixture.inputs["exner"][:, :, mask])
    differing = fp32_ulp_distance(got, want) > 0
    assert differing.any(), (
        "no lane differs, so this test has stopped demonstrating anything;"
        " either dtheta became bitwise or the fixture stopped reaching it")
    pushed = np.float32(got * exner)
    np.testing.assert_array_equal(pushed.view(np.uint32)[differing],
                                  ttnp.view(np.uint32)[differing])


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
def test_ysu_momentum_is_this_far_from_wrfs_own_ctopo_driver_path():
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    mask = _arithmetic_mask(fixture)
    for name, column in CTOPO_FIELD_MAP.items():
        got = np.ascontiguousarray(port[name][:, :, mask], np.float32)
        want = np.ascontiguousarray(fixture.level_reference[column][:, :, mask])
        assert int(fp32_ulp_distance(got, want).max()) == \
            CTOPO_BASELINE_MAX_ULP[name], name


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
def test_the_three_documented_behavioural_divergences_are_exactly_these():
    """The held-out columns, stated as what they are rather than as ULP.

    Each assertion is the divergence written down.  If one is ever closed this
    test fails, which is the intended way to find out.
    """
    import cupy  # noqa: F401

    fixture = _fixture()
    port = ysu_port_outputs(fixture)
    index = {case: i for i, case in enumerate(fixture.cases)}

    # case 7: -ftz turns a subnormal br into a regime change.
    i = index[7]
    assert fixture.inputs["br"].reshape(-1)[i] == np.float32(
        np.finfo(np.float32).smallest_subnormal)
    assert float(fixture.surface_reference["wstar"].reshape(-1)[i]) == 0.0, (
        "WRF read br > 0 and took the stable arm")
    assert float(port["wstar"].reshape(-1)[i]) > 0.0, (
        "the kernel read the flushed br as <= 0 and took the convective arm")
    assert float(fixture.surface_reference["delta"].reshape(-1)[i]) == 0.0
    assert float(port["delta"].reshape(-1)[i]) > 0.0

    # cases 12 and 13: ysu.cu:203 short circuits where WRF runs the scheme.
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
