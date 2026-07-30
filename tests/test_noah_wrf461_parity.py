"""Noah against the byte-unmodified WRF v4.6.1 driver, for the first time.

``sf_surface_physics=2`` is the land surface in every registry template and
every production config in this tree, and until this file existed no WRF
number had ever been produced for it.  ``kernels/noah.cu`` was checked only
against ``gpuwm.verify.npref.np_noah_column`` -- a float64 mirror of the same
transcription -- plus energy closure and physics plausibility, all of which a
transcription error present in both copies passes.  ``tools/noah_wrf461_oracle/``
now drives ``lsm`` in the pinned tree over 42 land columns and dumps every
driver input and every driver output; this module measures the kernel against
that dump and pins what it measures.

**The numbers below are a measurement, not a target.**  They record what the
shipped kernel actually does, so it cannot change without someone saying so in
a commit.  They are asserted for *equality*: a kernel that got better silently
is as much a drift as one that got worse.  Every baseline was produced twice
on the local RTX 5090 in separate processes and was bit-identical.

What the oracle found on its first run
--------------------------------------

1. ``chklowq`` was 0.0 against WRF's 1.0 on the open-water and sea-ice
   columns.  ``module_sf_noahdrv.F`` writes CHKLOWQ for every column at
   :809-814, *before* the XLAND branch; the kernel returned from its three
   skip paths without writing it.  Fixed in its own commit, and the mirror had
   it right all along -- which is precisely why 37 self-consistency tests
   could not see it.
2. Case 18 -- ``SNOW`` is the smallest positive subnormal.  Held out below.
3. The frozen-ground infiltration family (cases 6, 7, 42) and the fp64 energy
   residual, both described beside their baselines.

This gate has been observed to fail
----------------------------------

A gate that has never fired is not evidence, so it was made to fire twice
before it was committed, by patching ``kernels/noah.cu`` and re-running:

* deleting the one-line ``chklowq`` fix -- the defect this oracle found --
  moves ``chklowq`` from 0 to 1065353216 and the table assertion fails;
* respelling ``q2k = qv1/(1+qv1)`` as the algebraically identical
  ``1 - 1/(1+qv1)`` moves ``snopcx`` from 1125 to 1243 and it fails again.

One mutation that did *not* fire is worth recording too: perturbing ``val =
1 - expf(-kdt*dt1)`` by a relative 1e-7 changed nothing, because ``kdt*dt1``
is small enough that the perturbation falls below float32 resolution there.
The table is sensitive to the physics, not to every keystroke in the file.

Held-out columns
----------------

Three fixture columns are excluded from the ULP table and asserted
individually.  None of them measures arithmetic; folding a branch or a
not-ported routine into a ULP maximum hides it behind a big number.

* **case 18** -- ``SNOW`` is the smallest positive float32 subnormal.  WRF's
  ``IF(SNOW(I,J).GT.0.0)`` is TRUE, so the column takes the ice-saturation
  ``Q2SAT``/``DQSDT2`` blend; CuPy appends ``-ftz=true`` unconditionally, the
  subnormal flushes, and the kernel's ``snow_a[idx] > 0.0f`` is FALSE, so it
  takes the snow-free arm.  The port's answer for case 18 is *bit-identical to
  its own case 1*, which is the snow-free column; WRF's is not.  Cost: 0.083 K
  in TSK, 1.95 W/m2 in HFX, 2.87 W/m2 in LH.  This is the fourth instance of
  the ``-ftz`` class in this project and the first one an oracle fixture has
  ever caught.
* **case 25** -- ``CHS``/``CHS2``/``CQS2`` are all the smallest positive
  subnormal.  WRF itself fills the column with NaN, including SMOIS, SH2O and
  TSLB.  The kernel produces NaN in a *different* subset: TSK, HFX, LH,
  GRDFLX, QSFC and POTEVP are NaN in both, but the kernel keeps a finite
  0.464 soil moisture where WRF has NaN, and writes QFX = 0 where WRF writes
  147.7.  Neither answer is usable; the row is here so that "an unrunnable
  input is unrunnable in both" is a recorded fact rather than an assumption.
* **case 28** -- ``ivgtyp == isice``.  WRF calls ``SFLX_GLACIAL``
  (``module_sf_noahlsm_glacial_only.F``); gpuwm/core/noah.py's docstring
  records that routine as not ported, and ``noah_column`` returns without
  touching the column.  Measured size of that restriction on one 60 s step:
  TSK 255.0 against 273.15 K, HFX 0 against 388.9 W/m2, LH 0 against
  -319.1 W/m2, SH2O 0.28 against 1.0.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.noah_oracle import (
    NOAH_ORACLE_DIR,
    NOAH_ORACLE_FILES,
    OUTPUT_COLUMNS,
    SOIL_OUTPUT_COLUMNS,
    load_noah_oracle,
    noah_port_outputs,
)

#: Columns whose disagreement with WRF is a branch or a missing routine, not a
#: rounding.  Asserted one at a time below instead of being folded into a ULP
#: maximum.  See the module docstring for what each one is.
HELD_OUT_CASES = (18, 25, 28)

#: Worst ULP distance from ``kernels/noah.cu``'s ``noah_column`` to the word
#: ``lsm`` wrote, over the 39 columns that take the same branches as WRF and
#: over all four switch fixtures.  This is the kernel as it ships.
#:
#: The table has four populations, and they are not the same kind of thing:
#:
#:   sfcrunoff 60640600     FROZEN-GROUND INFILTRATION, cases 6, 7 and 42 --
#:   sh2o           6508    every column that is both frozen and melting snow.
#:   smcrel         4729    SRT (module_sf_noahlsm.F:3792-3806) reduces the
#:   smois          1627    infiltration limit by FCR, an expf of a powf series
#:   smstav          474    in ACRT = CVFRZ*FRZX/DICE, and then splits
#:   smstot           78    RUNOFF1 = PCPDRP - INFMAX.  The transcription is
#:                          faithful line for line; what differs is CUDA's
#:                          expf/powf against glibc's, amplified by a split
#:                          between two nearly equal quantities.  In absolute
#:                          terms the whole family is 4.85e-03 mm of water per
#:                          60 s step -- the SAME water, to three digits, that
#:                          appears as the 4.85e-05 volumetric SH2O gap, so it
#:                          is a redistribution and not a leak.  Closing it
#:                          needs a device expf/powf that matches glibc on
#:                          this argument range, i.e. the same work
#:                          gpuwm/core/noahmp_libm.py did for Noah-MP.
#:
#:   noahres       164754   THE ENERGY RESIDUAL IS COMPUTED IN A DIFFERENT
#:                          PRECISION.  noah.cu:1495 accumulates
#:                          (solnet+lwdn) - sheat + ssoil - eta - emissi*sigma*T^4
#:                          - flx1 - flx2 - flx3 in float64 and rounds once;
#:                          module_sf_noahdrv.F:1308 accumulates it in float32.
#:                          The sum is a near-total cancellation of terms of
#:                          order 500 W/m2 into a residual of order 0.01, so
#:                          WRF's answer carries only the low bits and is
#:                          quantised to ~3e-05 -- see the literal
#:                          -0.0313720703125 in the fixture.  gpuwm's number is
#:                          the more accurate one and is NOT WRF's.  This is
#:                          the "FP64-then-round-once is a third function"
#:                          trap, in a field that is diagnostic only: noahres
#:                          feeds no prognostic variable.
#:
#:   snopcx          1125   SNOWMELT BOOKKEEPING, cases 13 and 14.  Both are
#:   acsnom           836   SNOMLT*1000 accumulations, and both inherit the
#:                          same expf/powf gap through SNOPAC.
#:
#:   hfx              375   EVERYTHING ELSE: nvcc's FMA contraction plus
#:   grdflx           170   CUDA/glibc libm.  Proved for LAI, which is 1 ULP on
#:   ...              <=10  30 of 39 columns: WRF's
#:                          (1-f)*LAIMIN + f*LAIMAX at -O0 gives
#:                          2.2200002670288086, and fmaf(f, LAIMAX,
#:                          (1-f)*LAIMIN) gives 2.2200000286102295, which is
#:                          exactly what the kernel produces.  nvcc contracts
#:                          by default; gfortran at -O0 does not.
#:
#: Five fields are already bit-identical to WRF on every column of every
#: fixture: albbck, emiss, z0, snotime, acsnow, chklowq and tslb.
BASELINE_MAX_ULP = {
    "sfcrunoff": 60640600,
    "noahres": 164754,
    "sh2o": 6508,
    "smcrel": 4729,
    "smois": 1627,
    "snopcx": 1125,
    "acsnom": 836,
    "smstav": 474,
    "hfx": 375,
    "grdflx": 170,
    "smstot": 78,
    "snow": 10,
    "snowh": 9,
    "canwat": 8,
    "snowc": 8,
    "qfx": 5,
    "potevp": 5,
    "lh": 3,
    "qsfc": 3,
    "albedo": 3,
    "tsk": 2,
    "znt": 1,
    "lai": 1,
    "udrunoff": 1,
    "albbck": 0,
    "emiss": 0,
    "z0": 0,
    "snotime": 0,
    "acsnow": 0,
    "chklowq": 0,
    "tslb": 0,
}

#: The absolute size of the SFLX_GLACIAL restriction on case 28, in each
#: field's own units.  Not ULP: this is a routine gpuwm does not have, so a
#: rounding metric would be a category error.  Pinned so the registry warning
#: can quote a number and so the restriction cannot silently grow.
GLACIAL_ABSOLUTE_GAP = {
    "tsk": 18.14999389648438,
    "hfx": 388.85174560546875,
    "lh": 319.0890197753906,
    "grdflx": 42.21175003051758,
    "snopcx": 316.5550231933594,
}


def _fixtures():
    return [load_noah_oracle(name) for name in NOAH_ORACLE_FILES]


def _arithmetic_mask(fixture) -> np.ndarray:
    return np.asarray([c not in HELD_OUT_CASES for c in fixture.cases])


def _measure(fixture, port, mask) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in OUTPUT_COLUMNS:
        got = np.ascontiguousarray(port[name][:, mask], np.float32)
        want = np.ascontiguousarray(fixture.reference[name][:, mask])
        out[name] = int(fp32_ulp_distance(got, want).max())
    for name in SOIL_OUTPUT_COLUMNS:
        got = np.ascontiguousarray(port[name][:, :, mask], np.float32)
        want = np.ascontiguousarray(fixture.soil_reference[name][:, :, mask])
        out[name] = int(fp32_ulp_distance(got, want).max())
    return out


# --------------------------------------------------------------------------
# CPU-only: the fixture is what it says it is.
# --------------------------------------------------------------------------

def test_fixture_is_the_pinned_wrf_driver_and_its_own_receipts():
    """The CSVs must hash to what build.sh recorded next to them."""
    receipts = (NOAH_ORACLE_DIR / "oracle-sha256sums.txt").read_text(
        encoding="ascii").splitlines()
    recorded = {}
    for line in receipts:
        digest, path = line.split(maxsplit=1)
        recorded[path.strip().rsplit("/", 1)[-1]] = digest
    for name in NOAH_ORACLE_FILES:
        blob = (NOAH_ORACLE_DIR / name).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == recorded[name], name
    # The WRF sources the fixture came from are named in the same receipt, so a
    # fixture regenerated from a different tree cannot pass silently.  So are
    # the three parameter tables, which is what makes this fixture a check on
    # gpuwm's SOIL_VEG_GEN_PARM transcription and not just on the physics.
    joined = " ".join(receipts)
    for expected in ("module_sf_noahdrv.F", "module_sf_noahlsm.F",
                     "module_sf_noahlsm_glacial_only.F",
                     "module_model_constants.F", "module_wrf_error.F",
                     "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        assert expected in joined, expected


def test_libmvec_receipt_shows_the_reference_is_scalar_libm():
    """gfortran's vector libm must not be in the reference, and the grep for it
    must be capable of firing -- the report carries its own positive control."""
    report = (NOAH_ORACLE_DIR / "libmvec-report.txt").read_text(encoding="ascii")
    reference, _, rest = report.partition("# -O2")
    assert "_ZGV" not in reference, report
    assert "_ZGVbN4v_expf" in rest, (
        "the positive control produced no vector symbol, so the absence of one"
        " in the -O0 reference proves nothing")
    for symbol in ("U expf", "U powf", "U logf", "U log10f", "U atanf"):
        assert symbol in reference, report


def test_the_four_switch_fixtures_are_not_the_same_file():
    """A switch the fixture cannot discriminate is a switch nothing checks.

    ``noah-lsm-thcnd2.csv`` really was byte-identical to ``noah-lsm.csv``
    until soil types 3 and 4 were added: TDFCND's ``opt_thcnd == 2`` arm
    (``module_sf_noahlsm.F:4173``) is reachable for no other soil type.
    """
    blobs = {name: (NOAH_ORACLE_DIR / name).read_bytes()
             for name in NOAH_ORACLE_FILES}
    base = blobs["noah-lsm.csv"]
    for name, blob in blobs.items():
        if name == "noah-lsm.csv":
            continue
        assert blob != base, (
            f"{name} is byte-identical to noah-lsm.csv: no fixture column"
            " reaches what that switch changes")


def test_fixture_covers_the_branches_that_matter():
    fixture = load_noah_oracle()
    assert fixture.ncase == 42
    smallest = np.float32(np.finfo(np.float32).smallest_subnormal)
    inputs = {k: v.reshape(-1) for k, v in fixture.inputs.items()
              if v.ndim == 2}
    # signed zeros and subnormals on five different sign compares
    assert (inputs["rainbl"] == smallest).any()
    assert (np.signbit(inputs["rainbl"]) & (inputs["rainbl"] == 0)).any()
    assert (inputs["snow"] == smallest).any()
    assert (inputs["qv1"] == smallest).any()
    assert (np.signbit(inputs["qv1"]) & (inputs["qv1"] == 0)).any()
    assert (np.signbit(inputs["vegfra"]) & (inputs["vegfra"] == 0)).any()
    assert (inputs["canwat"] == smallest).any()
    assert (inputs["chs"] == smallest).any()
    # the three .GT. boundaries in the driver's snow block, straddled exactly
    assert (inputs["sfctmp"] == np.float32(273.15)).any()
    assert (inputs["sfctmp"]
            == np.nextafter(np.float32(273.15), np.float32(1e9))).any()
    assert (inputs["tsk"] == np.float32(273.14)).any()
    assert (inputs["tsk"] == np.float32(273.0)).any()
    assert (inputs["swdown"] == np.float32(10.0)).any()
    # the three columns the driver skips or hands to another routine
    assert (inputs["xland"] >= 1.5).any()
    assert (inputs["xice"] >= 0.5).any()
    assert fixture.glacial.any()
    # soil type 14 at a land point, which WRF rewrites to 7
    isltyp = fixture.isltyp.reshape(-1)
    xice = inputs["xice"]
    assert ((isltyp == 14) & (xice == 0.0)).any()
    # and the only two soil types opt_thcnd can change
    assert (isltyp == 3).any() and (isltyp == 4).any()


def test_the_baseline_table_names_every_measured_field():
    """A field silently dropped from the table would silently stop being
    gated.  The table and the oracle's field list must be the same set."""
    assert set(BASELINE_MAX_ULP) == set(OUTPUT_COLUMNS) | set(
        SOIL_OUTPUT_COLUMNS)


# --------------------------------------------------------------------------
# GPU: the measurement itself.
# --------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_noah_cuda_column_holds_its_measured_distance_from_wrf():
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    measured: dict[str, int] = {}
    for fixture in _fixtures():
        port = noah_port_outputs(fixture)
        one = _measure(fixture, port, _arithmetic_mask(fixture))
        for name, value in one.items():
            measured[name] = max(measured.get(name, 0), value)
    assert measured == BASELINE_MAX_ULP, (
        "Noah's distance from the unmodified WRF driver changed.\n"
        f"  measured {measured}\n  recorded {BASELINE_MAX_ULP}\n"
        "If a field got worse, something regressed.  If it got better, say so:"
        " update the table in the same commit as the improvement, with the"
        " expression that changed.")


@pytest.mark.gpu
@requires_gpu
def test_case_18_is_the_ftz_flush_and_nothing_else():
    """The subnormal-SNOW column proves the flush rather than asserting it.

    If ``-ftz`` is what is happening, the port's case 18 must be bit-identical
    to its own snow-free case 1 in every field the two share an input for --
    because after the flush the two columns ARE the same column.  WRF's two
    must differ, because WRF does not flush.  Both halves are asserted; the
    second is what stops this test passing on a fixture that lost its probe.
    """
    import cupy  # noqa: F401

    fixture = load_noah_oracle()
    port = noah_port_outputs(fixture)
    i1 = fixture.cases.index(1)
    i18 = fixture.cases.index(18)
    assert float(fixture.inputs["snow"].reshape(-1)[i18]) == float(
        np.finfo(np.float32).smallest_subnormal)

    # snowc is an input difference between the two columns, so it is excluded.
    shared = [f for f in OUTPUT_COLUMNS if f != "snowc"]
    for field in shared:
        got = np.ascontiguousarray(port[field], np.float32).reshape(-1)
        assert got[i1].tobytes() == got[i18].tobytes(), (
            f"{field}: the port's subnormal-snow column is no longer identical"
            " to its snow-free column, so -ftz is not the whole story any more")
    wrf_differs = [f for f in shared
                   if fixture.reference[f].reshape(-1)[i1].tobytes()
                   != fixture.reference[f].reshape(-1)[i18].tobytes()]
    assert wrf_differs, (
        "WRF's subnormal-snow column matches its snow-free column, so the"
        " fixture no longer discriminates the flush and this test is vacuous")
    tsk = fixture.reference["tsk"].reshape(-1)
    assert abs(float(tsk[i18]) - float(tsk[i1])) > 0.05


@pytest.mark.gpu
@requires_gpu
def test_case_25_is_unusable_in_wrf_too():
    """A subnormal exchange coefficient destroys the column in WRF as well.

    This is not a port defect and must not be recorded as one, but the two
    NaN patterns are not the same and that is worth pinning: WRF loses the
    soil state, the kernel keeps it.
    """
    import cupy  # noqa: F401

    fixture = load_noah_oracle()
    port = noah_port_outputs(fixture)
    i25 = fixture.cases.index(25)
    for field in ("tsk", "hfx", "lh", "grdflx", "qsfc", "potevp"):
        assert not np.isfinite(fixture.reference[field].reshape(-1)[i25]), field
        assert not np.isfinite(
            np.ascontiguousarray(port[field], np.float32).reshape(-1)[i25]), field
    # WRF's soil state is NaN; the port's is finite.  Recorded, not endorsed.
    assert not np.isfinite(fixture.soil_reference["smois"][0].reshape(-1)[i25])
    assert np.isfinite(
        np.ascontiguousarray(port["smois"], np.float32)[0].reshape(-1)[i25])


@pytest.mark.gpu
@requires_gpu
def test_the_glacial_restriction_is_this_big():
    """SFLX_GLACIAL is not ported.  This is what that costs, in W/m2 and K."""
    import cupy  # noqa: F401

    from gpuwm.verify.noah_oracle import glacial_divergence

    fixture = load_noah_oracle()
    port = noah_port_outputs(fixture)
    gaps = glacial_divergence(fixture, port)
    assert gaps, ("the fixture no longer carries a land-ice column, so the"
                  " documented restriction is no longer measured")
    for field, expected in GLACIAL_ABSOLUTE_GAP.items():
        assert gaps[field] == pytest.approx(expected, rel=1e-6), (
            f"{field}: SFLX_GLACIAL gap moved to {gaps[field]}")
