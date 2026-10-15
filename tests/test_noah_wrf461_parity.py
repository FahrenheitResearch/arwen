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
   residual, both described beside their baselines.  The first of those was
   filed as libm noise for a release and was not: ``SRT`` was being reached
   with ``REDPRM``'s ``FRZFACT`` where WRF passes its ``FRZX``, so ``ACRT``
   ran 1/FRZK = 6.67x large and the frozen-ground infiltration limiter did
   not limit.  Both the mirror and the kernel had made the same substitution,
   so only this fixture could see it -- and only once something graded the
   *mirror* against WRF too, which nothing did until
   :func:`test_the_mirror_reproduces_wrfs_frozen_ground_infiltration`.

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

The two CPU gates below were made to fire the same way, on the tree as it
stood before the FRZX fix: the mirror gate reports case 6 SFCRUNOFF
8.477e-05 against WRF's 4.935e-03, and the source gate reports ``noah_smflx``
called with ``frzfact``.  nvcc is a third witness -- it emitted ``variable
"frzx" was declared but never referenced`` for ``noah.cu`` until the fix.

Held-out columns
----------------

Two fixture columns are excluded from the ULP table and asserted
individually.  Neither measures arithmetic; folding a branch into a ULP
maximum hides it behind a big number.

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
Land ice (SFLX_GLACIAL), closed 2026-10-12
------------------------------------------

Case 28 is ``ivgtyp == isice``, where WRF's driver calls ``SFLX_GLACIAL``
(``module_sf_noahlsm_glacial_only.F``).  Until golden/ports-pbl the kernel
returned from it untouched -- after one 60 s step TSK 255.0 against WRF's
273.15 K, HFX 0 against 388.9 W/m2, LH 0 against -319.1 W/m2, SH2O 0.28
against 1.0, measured by this file and pinned as a restriction.  The routine
is ported now and case 28 sits inside the ordinary table: every field 0 ULP
except ZNT 1 (BURIAL = 7*Z0BRD - SNOWH, contracted to an FMA) and the
float64 NOAHRES.  The one land-ice column grades a gap; it does not grade a
port, so ``tools/noah_wrf461_oracle/run_lsm_glacial.F90`` adds 24 more, every
one land ice, reaching SFLX_GLACIAL's branches (snowfall at both density
laws and both 0.2 overrides, freezing rain, rain on a warm skin, melt, the
dew adjustment's three arms, sublimation, snow aging both sides of ALB,
SNOWZ0 both sides of its burial test, both SNOWPACK clamps, steep ice
temperature profiles), run with and without ``frpcpn``.  Those are pinned in
:data:`GLACIAL_BASELINE_MAX_ULP`.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.kernels import module_source
from gpuwm.core.noah import load_tables, pack_params
from gpuwm.verify.noah_oracle import (
    FIXTURE_DT,
    FIXTURE_DZS,
    FIXTURE_ISICE,
    FIXTURE_ISURBAN,
    FIXTURE_XICE_THRESHOLD,
    INPUT_COLUMNS,
    NOAH_GLACIAL_ORACLE_FILES,
    NOAH_ORACLE_DIR,
    NOAH_ORACLE_FILES,
    OUTPUT_COLUMNS,
    SOIL_OUTPUT_COLUMNS,
    load_noah_oracle,
    noah_port_outputs,
)
from gpuwm.verify.npref import np_noah_column

#: Columns whose disagreement with WRF is a branch, not a rounding.  Asserted
#: one at a time below instead of being folded into a ULP maximum.  See the
#: module docstring for what each one is.  Case 28 (land ice) left this tuple
#: when SFLX_GLACIAL was ported (golden/ports-pbl, 2026-10-12).
HELD_OUT_CASES = (18, 25)

#: SRT's own frozen-ground predicate, ``IF (DICE > 1.E-2)``
#: (``module_sf_noahlsm.F:3794``): below this the infiltration limiter is
#: inert and a column says nothing about FRZX.
FROZEN_DICE_M = 1.0e-2

#: How far the float64 mirror may sit from WRF's float32 SFCRUNOFF on the
#: columns where it IS live.  ``RUNOFF1 = PCPDRP - INFMAX``
#: (``module_sf_noahlsm.F:3830``) subtracts two nearly equal quantities, so
#: the precision gap is amplified; measured worst is 7.8e-06 over all four
#: fixtures, against 9.9e-01 when SRT is handed FRZFACT instead of FRZX.
FROZEN_RUNOFF_REL = 1.0e-4

#: Worst ULP distance from ``kernels/noah.cu``'s ``noah_column`` to the word
#: ``lsm`` wrote, over the 40 columns that take the same branches as WRF and
#: over all four switch fixtures (39 until case 28, land ice, joined with the
#: SFLX_GLACIAL port: it moved ZNT 0 -> 1 and nothing else -- (4) below).
#:
#: The frozen-ground family was reached with REDPRM's FRZFACT instead of its
#: FRZX until this branch -- see
#: :func:`test_the_mirror_reproduces_wrfs_frozen_ground_infiltration` for the
#: chain and the defect.  The whole table has been re-measured on the device
#: since the fix; what moved, and what was stale for other reasons, is set out
#: beside the dict below.
#:
#: The table has four populations, and they are not the same kind of thing:
#:
#:   sfcrunoff      2812    FROZEN-GROUND INFILTRATION, cases 6, 7 and 42 --
#:   smstav            2    every column that is both frozen and melting snow.
#:   smstot            1    SRT (module_sf_noahlsm.F:3792-3806) reduces the
#:   sh2o              0    infiltration limit by FCR, an expf of a powf series
#:   smcrel            0    in ACRT = CVFRZ*FRZX/DICE, and then splits
#:   smois             0    RUNOFF1 = PCPDRP - INFMAX.  These six numbers were
#:                          read as CUDA's expf/powf against glibc's, amplified
#:                          by a split between two nearly equal quantities.
#:                          That reading was wrong: the float64 mirror calls
#:                          glibc and was 98% away from WRF on the same three
#:                          columns, so the gap was ACRT, not libm.  With FRZX
#:                          restored, three of the six are exactly bitwise and
#:                          the remaining libm residue is now visible at its
#:                          true size -- sfcrunoff 2812 ULP, the mirror's own
#:                          7.8e-06 relative gap in the same place.
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
#:   snopcx          1141   SNOWMELT BOOKKEEPING, cases 13 and 14.  Both are
#:   acsnom           836   SNOMLT*1000 accumulations, and both inherit the
#:                          same expf/powf gap through SNOPAC.  Neither moved
#:                          with FRZX.
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
#: Ten fields are bit-identical to WRF on every column of every fixture:
#: albbck, emiss, z0, snotime, acsnow, chklowq, tslb, and -- newly, with FRZX
#: restored -- sh2o, smcrel and smois.
#: RE-MEASURED IN FULL for the FRZX fix, and again for 2.8 (item (2) below).
#:
#: (1) Improved by the FRZX repair -- six rows, every one of them a
#:     reduction, and three of them to exactly bitwise:
#:
#:       sfcrunoff  60641303 -> 2812      sh2o    6508 -> 0
#:       smcrel         4729 -> 0         smois   1627 -> 0
#:       smstav          474 -> 2         smstot    78 -> 1
#:
#:     The docstring above predicted snopcx and acsnom would move through
#:     SNOPAC as well.  They did not: both are byte-identical before and
#:     after, so the FRZX chain does not reach them on these four fixtures.
#:
#: (2) Eight rows the FRZX change also re-pinned, from a reading taken on its
#:     own branch, came back into this tree with values the kernel here does
#:     not produce: snow 0, snowh 0, tsk 1, znt 0, canwat 5, lh 4, qsfc 6 and
#:     snopcx 1141.  MEASURED 2026-09-28 on sm_120, where every release gate
#:     of this line runs: the RTX 5090 with NVRTC 12.9.86, 13.0.48 and
#:     13.4.92 and the RTX 5070 Ti with 13.0.88 and 13.3.33 all give, to the
#:     ULP, the values the table held before that merge --
#:
#:       snow 10     snowh 9     tsk 2     znt 1
#:       canwat 8    lh 3        qsfc 3    snopcx 1125
#:
#:     -- together with the six FRZX rows in (1) exactly as that change
#:     recorded them.  noah.cu at the branch head and at this tree's parent
#:     differ only by the three FRZX call sites, so the other reading was
#:     not of a different kernel source; no sm_120 compiler reproduces it,
#:     and the card it came from is not recorded.  The development desktop's
#:     RTX 3080 (sm_86, whose noah.cu frame is already known to differ:
#:     224 B against sm_120's 176 B) is the one release platform this table
#:     has not been read on, and it is read at the cut's Windows step; a
#:     different answer there makes this table per-architecture.
#:
#: (3) RE-MEASURED 2026-09-30 for A146 on the RTX 5070 Ti (sm_120, NVRTC
#:     13.4.92): NVRTC had compiled every float division by a compile-time
#:     constant as a multiply by the rounded reciprocal on Blackwell, and
#:     noah.cu now spells those divisions ``__fdiv_rn``, the IEEE quotient.
#:     Eight rows moved, and six of them to the sm_86 reading below, so the
#:     reciprocal multiply was most of what split the two cards:
#:
#:       snow 10 -> 2      snowh 9 -> 1      tsk 2 -> 1       znt 1 -> 0
#:       canwat 8 -> 5     lh 3 -> 4 (worse)  qsfc 3 -> 6 (worse)
#:       snopcx 1125 -> 1141 (worse)
#:
#:     Every other row, the six FRZX rows in (1) included, read as before.
#:
#: (4) golden/ports-pbl, 2026-10-12: SFLX_GLACIAL ported, case 28 joins the
#:     table.  Its every field is WRF's word except ZNT (1 ULP: SNOWZ0's
#:     BURIAL = 7*Z0BRD - SNOWH contracted to an FMA) and NOAHRES (the float64
#:     residual, 16286 ULP, inside the 164754 the land columns already set).
#:     ZNT 0 -> 1 is the only row that moved.  The 41 non-glacial columns are
#:     byte-identical to the kernel before the port on all four fixtures
#:     (measured on the G1 RTX PRO 6000, sm_120, NVRTC 12.9.86, base cd336e62d
#:     against the port), so no ordinary land column changed a word.
BASELINE_MAX_ULP = {
    "noahres": 164754,
    "sfcrunoff": 2812,
    "snopcx": 1141,
    "acsnom": 836,
    "hfx": 375,
    "grdflx": 170,
    "snow": 2,
    "snowh": 1,
    "snowc": 8,
    "canwat": 5,
    "qfx": 5,
    "potevp": 5,
    "qsfc": 6,
    "lh": 4,
    "albedo": 3,
    "smstav": 2,
    "tsk": 1,
    "znt": 1,
    "lai": 1,
    "udrunoff": 1,
    "smstot": 1,
    "sh2o": 0,
    "smcrel": 0,
    "smois": 0,
    "albbck": 0,
    "emiss": 0,
    "z0": 0,
    "snotime": 0,
    "acsnow": 0,
    "chklowq": 0,
    "tslb": 0,
}

#: sm_86 reads the eight rows (2) names differently, and exactly as the
#: unrecorded reading did: MEASURED 2026-09-28 on the RTX 3080 (sm_86,
#: NVRTC 13.4.92, Windows) at the 2.8.0 rehearsal (snow 0, snowh 0, tsk 1,
#: znt 0, canwat 5, lh 4, qsfc 6, snopcx 1141).  sm_89 reads them exactly as
#: sm_86 does: MEASURED 2026-09-30 on the node-1 RTX 4090 (sm_89, NVRTC
#: 13.4.92, Linux), the same dict at the 5f6007af9 noah.cu and at the urban
#: hand-over's noah.cu, so the row records the card and no change to the
#: Noah column.  Since A146 (3) the sm_120 table above reads six of those
#: eight the same, so only snow and snowh remain per-architecture (A146
#: moves neither card: NVRTC keeps the IEEE division below compute_100).
_SM86_ROWS = {**BASELINE_MAX_ULP, "snow": 0, "snowh": 0}
BASELINE_MAX_ULP_BY_ARCHITECTURE = {
    "86": _SM86_ROWS,
    "89": _SM86_ROWS,
}

#: Where an architecture reads differently under one NVRTC build, keyed
#: (compute capability as CuPy writes it, NVRTC build); it wins over the
#: architecture's row.  A167: the sm_120 table above is A146's reading under
#: 13.4.92.  Under 12.9.86, the compiler of the default [gpu] extra
#: (cupy-cuda12x), sm_120 reads snow 0 and snowh 0 and every other row as
#: that table: the sm_86/89 row exactly.  Before A146, 12.9.86, 13.0.48
#: and 13.4.92 read one table on the RTX 5090 ((2) above); with the
#: divisions spelled __fdiv_rn the two compilers read snow and snowh
#: differently on sm_120.  MEASURED 2026-10-01 on node-4's RTX 5070 Ti and
#: node-2's RTX 5090, cupy-cuda12x 14.2.0, two processes each, at the A167
#: lane tip on integrate/2.8 1a0e0090e; under 13.4.92 both cards read the
#: sm_120 table above.
BASELINE_MAX_ULP_BY_ARCH_AND_BUILD = {
    ("120", "12.9.86"): _SM86_ROWS,
}
#: Rows an (architecture, build) row may move from its architecture's row.
ARCH_AND_BUILD_SENSITIVE_FIELDS = ("snow", "snowh")

#: Worst ULP distance from the SFLX_GLACIAL port to the word ``lsm`` wrote,
#: over the 24 land-ice columns of both glacial fixtures
#: (``noah-lsm-glacial.csv`` and ``noah-lsm-glacial-frpcpn.csv``).
#: MEASURED 2026-10-12 on the G1 RTX PRO 6000 (sm_120) under NVRTC 12.9.86
#: and 13.2, identical.
#: TSK, HFX, the whole TSLB profile,
#: SNOW, ALBEDO, EMISS, SNOTIME and SNOWC are WRF's word on every column; the
#: residue is FMA contraction and CUDA's expf/powf (CSNOW, SNOW_NEW, ALCALC's
#: aging, SNOWPACK) carried into the latent-heat and melt bookkeeping, and the
#: float64 NOAHRES this unit computes for every column (OPEN 3 in the
#: registry).
GLACIAL_BASELINE_MAX_ULP = {
    "noahres": 165327,
    "snopcx": 5,
    "sfcrunoff": 4,
    "acsnom": 4,
    "potevp": 4,
    "qfx": 3,
    "lh": 2,
    "qsfc": 2,
    "snowh": 2,
    "grdflx": 1,
    "znt": 1,
    "tsk": 0,
    "hfx": 0,
    "canwat": 0,
    "snow": 0,
    "snowc": 0,
    "albedo": 0,
    "albbck": 0,
    "emiss": 0,
    "z0": 0,
    "snotime": 0,
    "lai": 0,
    "smstav": 0,
    "smstot": 0,
    "udrunoff": 0,
    "acsnow": 0,
    "chklowq": 0,
    "smois": 0,
    "tslb": 0,
    "sh2o": 0,
    "smcrel": 0,
}

#: How far the float64 mirror (``npref._noah_glacial``) may sit from WRF's
#: float32 words on the land-ice columns, relative, with a 1e-3 floor on the
#: denominator so a near-zero flux does not turn a 1e-8 W/m2 gap into a
#: relative blow-up.  Measured worst 4.6e-05 (HFX on a near-cancelling
#: -10.9 W/m2 column).
GLACIAL_MIRROR_REL = 1.0e-4


def _fixtures():
    return [load_noah_oracle(name) for name in NOAH_ORACLE_FILES]


def _glacial_fixtures():
    return [load_noah_oracle(name) for name in NOAH_GLACIAL_ORACLE_FILES]


def _arithmetic_mask(fixture) -> np.ndarray:
    return np.asarray([c not in HELD_OUT_CASES for c in fixture.cases])


def _mirror_column(fixture, i) -> dict:
    """One fixture column as the plain dict ``np_noah_column`` takes."""
    col = {field: float(fixture.inputs[field].reshape(-1)[i])
           for field in INPUT_COLUMNS}
    col["ivgtyp"] = int(fixture.ivgtyp.reshape(-1)[i])
    col["isltyp"] = int(fixture.isltyp.reshape(-1)[i])
    for field in ("smois", "tslb", "sh2o"):
        col[field] = np.asarray(
            fixture.inputs[field].reshape(4, -1)[:, i], np.float64)
    return col


def _srt_dice(fixture, i) -> float:
    """SRT's ``DICE`` for one column, in m: total column soil-ice depth.

    ``module_sf_noahlsm.F:3754`` seeds it with ``-ZSOIL(1)*SICE(1)`` and
    ``:3765`` accumulates ``(ZSOIL(KS-1)-ZSOIL(KS))*SICE(KS)``.
    """
    smois = fixture.inputs["smois"].reshape(4, -1)[:, i]
    sh2o = fixture.inputs["sh2o"].reshape(4, -1)[:, i]
    sice = np.asarray(smois - sh2o, np.float64)
    zsoil = -np.cumsum(np.asarray(FIXTURE_DZS, np.float64))
    dice = -zsoil[0] * sice[0]
    for k in range(1, len(sice)):
        dice += (zsoil[k - 1] - zsoil[k]) * sice[k]
    return float(dice)


def _call_argument_lists(source: str, callee: str) -> list[list[str]]:
    """Every parenthesised argument list ``callee`` appears with, in order.

    The first is the definition, whose parameter list names the positions;
    the rest are the call sites.
    """
    out: list[list[str]] = []
    for match in re.finditer(rf"\b{callee}\s*\(", source):
        depth, j = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(source[j], 0)
            j += 1
        out.append([a.strip() for a in source[match.end():j - 1].split(",")])
    return out


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
    for name in (*NOAH_ORACLE_FILES, *NOAH_GLACIAL_ORACLE_FILES):
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
                     "run_lsm.F90", "run_lsm_glacial.F90",
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


def test_a_compiler_row_moves_only_what_the_compiler_reaches():
    """A167: an (architecture, build) row may differ from its
    architecture's row only in ARCH_AND_BUILD_SENSITIVE_FIELDS, so a
    compiler row cannot quietly re-baseline a field no compiler moved."""
    for (capability, build), row in BASELINE_MAX_ULP_BY_ARCH_AND_BUILD.items():
        base = BASELINE_MAX_ULP_BY_ARCHITECTURE.get(capability,
                                                    BASELINE_MAX_ULP)
        assert set(row) == set(base), (capability, build)
        moved = {name for name in row if row[name] != base[name]}
        assert moved and moved <= set(ARCH_AND_BUILD_SENSITIVE_FIELDS), (
            capability, build, sorted(moved))


def test_the_mirror_reproduces_wrfs_frozen_ground_infiltration():
    """SRT must spend REDPRM's ``FRZX``, and only WRF can say whether it does.

    ``REDPRM`` builds ``FRZFACT = (SMCMAX/SMCREF)*(0.412/0.468)`` and
    ``FRZX = FRZK*FRZFACT`` (``module_sf_noahlsm.F:2477-2478``), and ``SFLX``
    passes *FRZX* down (``:769``, ``:784``).  The receiving dummy is merely
    *named* ``FRZFACT`` in NOPAC (``:1909``), SNOPAC (``:3015``) and SMFLX
    (``:2670``), and is renamed back to ``FRZX`` in SRT (``:3655``), which
    spends it as ``ACRT = CVFRZ*FRZX/DICE`` (``:3795``).  Handing that chain
    the REDPRM local ``FRZFACT`` instead scales ACRT by ``1/FRZK`` = 6.67,
    ``FCR`` saturates at 1 and the frozen-ground infiltration limiter stops
    limiting.  No self-consistency gate in this tree can see that, because
    ``npref`` and ``noah.cu`` made the same substitution -- which is why this
    is measured against WRF's own word and nothing else.
    """
    params = pack_params(load_tables())
    measured = []
    for name in NOAH_ORACLE_FILES:
        fixture = load_noah_oracle(name)
        for i, case in enumerate(fixture.cases):
            want = float(fixture.reference["sfcrunoff"].reshape(-1)[i])
            if case in HELD_OUT_CASES or want <= 0.0:
                continue
            if _srt_dice(fixture, i) <= FROZEN_DICE_M:
                continue
            out = np_noah_column(_mirror_column(fixture, i), params,
                                 FIXTURE_DT, FIXTURE_DZS,
                                 isurban=FIXTURE_ISURBAN,
                                 isice=FIXTURE_ISICE,
                                 xice_threshold=FIXTURE_XICE_THRESHOLD,
                                 **fixture.switches)
            measured.append((name, case, out["sfcrunoff"], want))
    assert len(measured) == 3 * len(NOAH_ORACLE_FILES), (
        "the fixture no longer carries three frozen columns that run off, so"
        " this test cannot see the frozen-ground limiter at all")
    for name, case, got, want in measured:
        assert got == pytest.approx(want, rel=FROZEN_RUNOFF_REL), (
            f"{name} case {case}: mirror SFCRUNOFF {got!r} against WRF's"
            f" {want!r}.  SRT's ACRT is not being formed from REDPRM's FRZX.")


def test_the_kernel_hands_smflx_the_same_word_the_mirror_does():
    """``noah.cu`` inlines NOPAC and SNOPAC, so its ``noah_smflx`` call sites
    ARE ``SFLX``'s ``:769``/``:784``, and the word they pass must be ``frzx``.

    The mirror is graded against WRF above; the kernel cannot be, without a
    device.  This is the part of that claim a source can carry: the argument
    standing in ``noah_smflx``'s ``frzfact`` slot -- named after the Fortran
    dummy, as in WRF -- is the ``frzx`` REDPRM built, at every call site.
    nvcc says the same thing more bluntly: before this was true it reported
    ``variable "frzx" was declared but never referenced``.
    """
    lists = _call_argument_lists(module_source("noah"), "noah_smflx")
    signature, calls = lists[0], lists[1:]
    slot = [p.split()[-1].lstrip("*&") for p in signature].index("frzfact")
    assert len(calls) == 3, (
        f"noah.cu calls noah_smflx {len(calls)} times, not 3 (WRF: NOPAC"
        " :2048 and :2111, SNOPAC :3367); read the new one before repinning")
    for args in calls:
        assert len(args) == len(signature)
        assert args[slot] == "frzx", (
            f"noah_smflx is called with {args[slot]!r} where WRF's SFLX"
            " passes FRZX (module_sf_noahlsm.F:769, :784)")


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
    from gpuwm.certify.compile_platform import nvrtc_build

    capability = str(cupy.cuda.Device().compute_capability)
    build = nvrtc_build()
    recorded = BASELINE_MAX_ULP_BY_ARCH_AND_BUILD.get(
        (capability, build),
        BASELINE_MAX_ULP_BY_ARCHITECTURE.get(capability, BASELINE_MAX_ULP))
    assert measured == recorded, (
        "Noah's distance from the unmodified WRF driver changed.\n"
        f"  measured {measured}\n  recorded {recorded}\n"
        f"  on sm_{capability} under NVRTC {build}\n"
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
    # WRF's soil state is NaN at this case and the port's is finite.  The two
    # asserts pin that difference as measured; neither column is claimed right.
    assert not np.isfinite(fixture.soil_reference["smois"][0].reshape(-1)[i25])
    assert np.isfinite(
        np.ascontiguousarray(port["smois"], np.float32)[0].reshape(-1)[i25])


def test_the_land_ice_and_urban_categories_follow_the_dataset():
    """ISURBAN and ISICE are the land-use dataset's, not MODIS's for everyone.

    WRF hands its Noah driver the input file's ISURBAN/ISICE
    (share/input_wrf.F:849-862).  The driver here used to pass 13 and 15 for
    every dataset, so under USGS -- where 13 is evergreen broadleaf forest,
    15 is mixed forest, 1 is urban and 24 is snow or ice -- the mixed forest
    was skipped as land ice and the real ice ran ordinary SFLX.  Measured on
    the float64 mirror with the USGS tables: category 15 runs SFLX (skip 0)
    and category 24 runs SFLX_GLACIAL (skip 3); with the MODIS tables the
    same two categories do the opposite.
    """
    from gpuwm.core.noah import NOAH_LANDUSE_CATEGORY_IDS, noah_category_ids

    assert noah_category_ids(pack_params(load_tables())) == (13, 15)
    usgs = pack_params(load_tables(mminlu="USGS"))
    assert noah_category_ids(usgs) == (1, 24)
    fixture = load_noah_oracle()
    base = _mirror_column(fixture, fixture.cases.index(1))
    expect = {"USGS": {15: 0, 24: 3}, "MODIFIED_IGBP_MODIS_NOAH": {15: 3}}
    for dataset, cases in expect.items():
        params = pack_params(load_tables(mminlu=dataset))
        for category, skip in cases.items():
            out = np_noah_column(dict(base, ivgtyp=category), params,
                                 FIXTURE_DT, FIXTURE_DZS)
            assert out["skip"] == skip, (dataset, category, out["skip"])
            assert out["tsk"] != base["tsk"], (
                dataset, category, "the column was left untouched")
    # every dataset VEGPARM.TBL carries has an identity, and an unknown one
    # is refused by name rather than run with somebody else's categories
    for dataset in ("USGS", "MODIFIED_IGBP_MODIS_NOAH", "NLCD40", "USGS-RUC",
                    "MODI-RUC"):
        assert dataset in NOAH_LANDUSE_CATEGORY_IDS
    # (the two -RUC blocks carry RUC's column layout and are not Noah's to
    # load; the three below are)
    for dataset in ("USGS", "MODIFIED_IGBP_MODIS_NOAH", "NLCD40"):
        noah_category_ids(pack_params(load_tables(mminlu=dataset)))
    import dataclasses
    stranger = dataclasses.replace(usgs, lutype="NOT-A-DATASET")
    with pytest.raises(ValueError, match="ISURBAN/ISICE"):
        noah_category_ids(stranger)


def test_the_glacial_fixtures_reach_the_branches_they_claim():
    """Every column is land ice, the two passes differ, and WRF's own words
    show the branches were taken: melt (SFCRUNOFF > 0), snowfall (SNOW up),
    deposition (LH < 0), sublimation (LH > 0), the entry reset (SNOW set to
    100 mm from less), snow aging (ALBEDO below the fresh-snow value), the
    SNOWZ0 burial arm (ZNT above Z0S) and a skin pinned at TFREEZ."""
    plain, frpcpn = _glacial_fixtures()
    assert (NOAH_ORACLE_DIR / "noah-lsm-glacial.csv").read_bytes() != (
        NOAH_ORACLE_DIR / "noah-lsm-glacial-frpcpn.csv").read_bytes()
    for fixture in (plain, frpcpn):
        assert fixture.ncase == 24 and fixture.glacial.all()
        ref = {k: v.reshape(-1) for k, v in fixture.reference.items()}
        snow_in = fixture.inputs["snow"].reshape(-1)
        assert (ref["sfcrunoff"] > 0).sum() >= 3
        assert (ref["snow"] > np.maximum(snow_in, 100.0)).sum() >= 3
        assert (ref["lh"] < 0).sum() >= 3 and (ref["lh"] > 0).sum() >= 10
        assert ((snow_in < 100.0) & (ref["snow"] == np.float32(100.0))).sum() >= 2
        assert (ref["albedo"] < ref["albedo"].max()).sum() >= 2
        assert (ref["znt"] > np.float32(0.001)).any()
        assert (ref["tsk"] == np.float32(273.15)).sum() >= 3
    # SR = 0.3 at case 6 and SR = 0.8 at case 23 send the frpcpn pass down
    # the other precipitation arm
    for case in (6, 23):
        i = plain.cases.index(case)
        assert plain.reference["snow"].reshape(-1)[i] != \
            frpcpn.reference["snow"].reshape(-1)[i], case


def test_the_mirror_reproduces_wrfs_land_ice_columns():
    """The float64 mirror's SFLX_GLACIAL against WRF's float32 words, on every
    land-ice column of every fixture -- the CPU half of the port's grade."""
    params = pack_params(load_tables())
    checked = 0
    for fixture in (*_fixtures(), *_glacial_fixtures()):
        for i in np.flatnonzero(fixture.glacial.reshape(-1)):
            out = np_noah_column(_mirror_column(fixture, i), params,
                                 FIXTURE_DT, FIXTURE_DZS,
                                 isurban=FIXTURE_ISURBAN,
                                 isice=FIXTURE_ISICE,
                                 xice_threshold=FIXTURE_XICE_THRESHOLD,
                                 **fixture.switches)
            assert out["skip"] == 3, "the mirror skipped a land-ice column"
            for field in OUTPUT_COLUMNS:
                if field == "noahres":
                    continue
                want = float(fixture.reference[field].reshape(-1)[i])
                got = float(out[field])
                assert abs(got - want) <= GLACIAL_MIRROR_REL * max(
                    abs(want), 1.0e-3), (fixture.name, fixture.cases[i],
                                         field, got, want)
            for field in SOIL_OUTPUT_COLUMNS:
                want = fixture.soil_reference[field].reshape(4, -1)[:, i]
                np.testing.assert_allclose(
                    np.asarray(out[field], np.float64), want,
                    rtol=GLACIAL_MIRROR_REL, atol=0.0,
                    err_msg=f"{fixture.name} case {fixture.cases[i]} {field}")
            checked += 1
    assert checked == 4 + 2 * 24


@pytest.mark.gpu
@requires_gpu
def test_noah_cuda_land_ice_columns_hold_their_measured_distance_from_wrf():
    """SFLX_GLACIAL on the card, over both glacial fixtures, asserted for
    equality as the land table is."""
    import cupy  # noqa: F401

    measured: dict[str, int] = {}
    for fixture in _glacial_fixtures():
        port = noah_port_outputs(fixture)
        one = _measure(fixture, port, np.ones(fixture.ncase, dtype=bool))
        for name, value in one.items():
            measured[name] = max(measured.get(name, 0), value)
    assert measured == GLACIAL_BASELINE_MAX_ULP, (
        "the land-ice port's distance from WRF changed.\n"
        f"  measured {measured}\n  recorded {GLACIAL_BASELINE_MAX_ULP}")


@pytest.mark.gpu
@requires_gpu
def test_a_land_ice_column_is_no_longer_left_untouched():
    """The defect the port closed, asserted in WRF's own units: on case 28
    the kernel used to leave TSK at 255.0 K and HFX at 0 where WRF writes
    273.15 K and 388.9 W/m2.  Both are now WRF's word."""
    import cupy  # noqa: F401

    fixture = load_noah_oracle()
    port = noah_port_outputs(fixture)
    i = fixture.cases.index(28)
    assert fixture.glacial.reshape(-1)[i]
    assert float(fixture.inputs["tsk"].reshape(-1)[i]) == 255.0
    for field in ("tsk", "hfx", "lh", "grdflx", "snopcx"):
        got = np.ascontiguousarray(port[field], np.float32).reshape(-1)[i]
        want = fixture.reference[field].reshape(-1)[i]
        assert got.tobytes() == want.tobytes(), (field, got, want)
    for field in ("smois", "sh2o", "smcrel"):
        assert np.all(np.asarray(port[field])[:, 0, i] == 1.0), field
