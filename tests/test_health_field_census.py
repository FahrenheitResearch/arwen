"""The health-descriptor ceiling, measured rather than projected.

``gpuwm.core.health.collect_state_fields`` auto-walks ``driver.fields``, so
every admitted scheme adds descriptors against a fixed ``MAX_HEALTH_FIELDS``.
The cap is enforced INSIDE ``collect_state_fields``, which a forecast first
calls on a synchronized boundary -- after allocation, after ingest, after the
run has started.  So a configuration over the cap is not a configuration that
fails to launch; it is a configuration that dies mid-forecast.

Nobody was measuring it.  What existed were two projections, each obtained by
multiplying a single-domain delta by four (~196, and ~707 of 1024 "about 69%").
Both measure a quantity that does not exist: the cap is applied per
``DomainState`` (``model.py`` builds one ``StateHealthValidator`` per domain,
passing no ``extra_tables``), and the ROOT and a CHILD are structurally
different inventories, so neither one times four is anything.  The tests below
instantiate the real four-domain configuration and count.

WHY THIS FILE EXISTS AT ALL
---------------------------
A ceiling nobody measures is one a user discovers mid-forecast.  The exhaustive
sweep here is the measurement, and it re-runs on every suite invocation, so a
scheme admitted next month is counted whether or not its author thinks about
this file.  It is deliberately NOT marked ``slow``: a gate that is routinely
deselected is the situation this task was created to end.  It costs about 65 s.

NO DEVICE IS OPENED
-------------------
The sweep runs in a fresh subprocess, which is what lets the census bind
``cupy`` to NumPy before any gpuwm module can capture a device handle -- and
which incidentally exercises the shipped CLI rather than a re-implementation of
it.  This module imports no cupy, directly or in a subprocess.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import health_field_census as census

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "health_field_census.py"
#: The reference four-domain case: d01 250x200, d02 500x400, d03 501x501,
#: d04 600x600, nz=49.  A case-named config is legitimate here -- this is a
#: fixture reference, not generic code.
FOUR_DOMAIN_CONFIG = REPO_ROOT / "configs" / "real74_4dom.toml"

# ---------------------------------------------------------------------------
# Measured on the four-domain reference case.  Every number below came out of
# `python tools/health_field_census.py configs/real74_4dom.toml`; none is
# projected, and none may be edited to make a failure go away.  A scheme that
# moves them is a scheme whose author must re-run the census and record what it
# now says -- which is the entire point of this file.
# ---------------------------------------------------------------------------

#: Peak per-domain descriptor count over the whole selectable cross-product.
#: Re-measured 2026-07-26, the first measurement since b682ef3 made the census
#: unrunnable: ``gpuwm/core/mynn_pbl_gpu.py`` built three
#: ``cp.ReductionKernel`` objects at module import, and the census's
#: fail-closed host backend has no such symbol, so every run of this file
#: errored out and 601 was a number nobody could reproduce.  The kernels
#: are now built on first use and the
#: sweep runs again.  601 -> 570 was a DROP, and it was not an improvement in
#: the inventory: Noah-MP left the SELECTABLE set.  ``gpuwm/physics_compat.py``
#: refused ``sf_surface_physics=4`` above the host-era 352-column ceiling, and
#: d01 is 50,000 columns, so every one of the 240 ``lsm4`` points in the
#: cross-product carried that refusal and no ``lsm4`` row was measured at
#: all.  The four ``lsm4`` rows that had held the peak were gone.
#:
#: Re-derived independently 2026-07-27 before this number was written down,
#: because a predicted measurement is not a measurement.  With the 352-column
#: ceiling the sweep returned 230 measured rows and 730 refusals against a
#: 960-point cross-product (5 x 4 x 3 x 4 x 2 x 2), all 240 ``lsm4``
#: refusals naming ``Noah-MP column budget (sf_surface_physics=4,
#: columns=50000)``, and a control run with
#: ``GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET=360000`` returned 290 rows, 670
#: refusals, and 601 at root 229 on exactly the old four-way tie -- proof
#: that the 570 peak was the old peak made unreachable, not shrunk.
#:
#: 2026-07-27, LATER THE SAME DAY: the slab orchestration's timing runs
#: moved ``NOAHMP_MEASURED_COLUMN_CEILING`` to its measured 360,000 (see
#: gpuwm/physics_compat.py), so Noah-MP is selectable on every domain of
#: this case again and the control's prediction became the measurement:
#: the re-run sweep returns 290 measured rows, 670 refusals, **601 at root
#: 229 on exactly the old four-way tie**, and NO refusal anywhere in the
#: report names the column budget (the 180 remaining ``lsm4`` refusals are
#: 60 MYNN/Noah-MP pair, 60 MYNN half-suite, 40 LSM-without-surface-layer
#: and 20 km_opt=4/pbl0, every one predating the budget).
#
#: Re-measured 2026-07-30 after the WRF-owned MYNN/RUC and MYNN/Noah-MP
#: pairings became selectable: 396 measured rows, 756 refusals, and 632 at
#: root 260.  The 31-descriptor increase is RUC's retained MYNN fractional
#: sea-ice surface-layer result, which is required for WRF's post-LSM blend.
#:
#: Re-measured again 2026-07-30 after the complete WRF v4.6.1 pairing
#: table replaced the reciprocal half-suite and PBL-off diffusion gates:
#: 792 measured rows, 360 refusals, and the same 632/root-260 peak.  The
#: remaining refusals are 288 WRF-fatal pairing rows plus 72 active-LSM
#: rows whose surface exchange fields have no ArWen writer.
#: Headroom remains 392 descriptors under the unchanged 1024 ceiling.
WORST_MEASURED_COUNT = 632
#: The worst combination(s).  MYNN/RUC with Kain-Fritsch is widest; km_opt
#: does not change the inventory -- every member here peaks at the SAME 632.
#:
#: km3 joined this set when the LES lane made km_opt=3 selectable in
#: validate_run_config, and LEFT it again on the 1.7 line when the
#: validator was taught the constraint the registry had always declared:
#: components/turbulence/options/smagorinsky-3d requires
#: bl_pbl_physics=0, because km_opt=3's vertical exchange pair is applied
#: by the PBL-off-gated vertical_diffusion_2.  Every member of this set
#: is pbl5 (MYNN), so km_opt=3 is not selectable here -- for exactly the
#: reason km_opt=2 never was.  The measured peak did NOT move: still 632,
#: headroom still 392 of 1024.  Two members now instead of three, and
#: "km_opt does not change the inventory" still holds across both.
WORST_SELECTIONS = frozenset({
    "mp18-lsm3-pbl5-sfclay5-cu1-km1",
    "mp18-lsm3-pbl5-sfclay5-cu1-km4",
})
#: The worst combination, per domain.  The root is about 41% of a child:
#: a child carries 196 ``nest.scratch`` rolling/SINT descriptors plus 178
#: ``lbc`` descriptors for the rolling boundary its FORCE attaches, and the
#: root carries 2 packed LBC descriptors instead -- 260 - 2 + 196 + 178 = 632.
WORST_ROOT_COUNT = 260
WORST_CHILD_COUNT = 632

#: An early-warning band, not the cap.  632 of 1024 leaves 392.  The current
#: peak includes the 31 retained fractional-sea-ice descriptors used only by
#: the MYNN/RUC WRF ownership sequence.  The band is NOT recalibrated to the
#: new peak: it exists to fire while there is still room to decide.
#: Tightening only -- raising this number is not a way to make a failure pass.
EARLY_WARNING_COUNT = 768

#: ``health.py``'s ceiling comment cites "a four-domain NSSL-2 step currently
#: reaches 527 descriptors".  This is the combination that produces exactly
#: that, which is what makes the census checkable against something written
#: before it existed rather than only against itself.
CALIBRATION_SELECTION = "mp18-lsm2-pbl0-sfclay1-cu0-km1"
CALIBRATION_CHILD_COUNT = 527

#: What ``MAX_HEALTH_FIELDS`` costs: seven fixed ``integration_health_*``
#: device slots per DomainState.  48 KiB per domain at 1024.
CEILING_METADATA_BYTES_PER_DOMAIN = 49152


@pytest.fixture(scope="module")
def four_domain_census(tmp_path_factory):
    """The full selectable sweep, from a fresh interpreter, run from a cwd
    that is NOT the repository.

    The foreign cwd is load-bearing.  ``gpuwm`` is installed editable, and its
    finder resolves to whichever checkout ran ``pip install -e`` -- not to the
    tree under test.  Run as a script, ``sys.path[0]`` is ``tools/``, which
    holds no ``gpuwm``, so the editable finder wins and the census silently
    measures a different worktree.  Running from a temporary directory means
    only the tool's own ``sys.path`` claim can save it.
    """
    foreign_cwd = tmp_path_factory.mktemp("not-the-repository")
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--json", str(FOUR_DOMAIN_CONFIG)],
        cwd=str(foreign_cwd), capture_output=True, text=True, timeout=1800,
        env={**os.environ, "GPUWM_NO_LOCAL_GPU": "1"})
    assert completed.returncode == 0, (
        f"the census failed:\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout)


def _child_rows(report):
    """Rows by selection, each per-domain map keyed by int grid id."""
    return {row["selection"]: {int(k): v for k, v in row["per_domain"].items()}
            for row in report["rows"]}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_no_selectable_combination_exceeds_the_ceiling(four_domain_census):
    """The headline: nothing a user can pick dies at the health gate."""
    report = four_domain_census
    assert report["over_ceiling"] == [], (
        "these selectable combinations exceed MAX_HEALTH_FIELDS and would "
        f"die mid-forecast: {report['over_ceiling']}")
    assert report["worst_count"] <= report["max_health_fields"]
    assert report["rows"], "the census measured nothing at all"


def test_worst_selectable_count_is_the_recorded_measurement(
        four_domain_census):
    """The peak is pinned, so growth is visible instead of merely tolerated."""
    report = four_domain_census
    assert report["worst_count"] == WORST_MEASURED_COUNT, (
        f"the worst selectable descriptor count moved from "
        f"{WORST_MEASURED_COUNT} to {report['worst_count']} "
        f"({report['worst_selection']}).  This is not a test to edit: re-run "
        f"`python tools/health_field_census.py {FOUR_DOMAIN_CONFIG.name}`, "
        "record the new number here, and state the remaining headroom out of "
        f"{report['max_health_fields']} in the commit message.")
    peaks = {row["selection"] for row in report["rows"]
             if row["peak"] == report["worst_count"]}
    assert peaks == WORST_SELECTIONS, (
        f"the worst-case combination set changed to {sorted(peaks)}")


def test_noahmp_slice_matches_the_current_wrf_authority(
        four_domain_census):
    """Why 570 became 601 again, as a gate rather than as a comment.

    From 573939c to the slab orchestration, ``validate_run_config`` handed
    the readiness authority ``columns=nx*ny`` and the authority refused
    ``sf_surface_physics=4`` above the host-era 352-column ceiling, so
    every ``lsm4`` combination was out of the selectable set at d01's
    50,000 columns and the peak read 570.  On 2026-07-27 the ceiling moved
    to the slab path's measured 360,000 (gpuwm/physics_compat.py), every
    domain of this case fits under it, and the peak went back up to 601 --
    the outcome the previous revision of this test named as correct, and
    the number its ``GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET=360000`` control
    had already measured.

    Pinned three ways, because each one on its own could be true for the
    wrong reason: the ``lsm4`` slice is measured again and at its recorded
    size; NO refusal anywhere in the report names the column budget (this
    case has no domain wider than the measured ceiling, so a budget refusal
    here would mean the ceiling and the measurement drifted apart); and the
    ``lsm4`` refusals that remain are exactly the pre-budget reasons, so
    the rise is Noah-MP returning and not some validator quietly loosening.
    If a wider case ever enters this census, the budget rail is expected to
    speak again -- for the width nothing has measured, not for this one.

    Re-pinned 2026-07-30: 60 -> 72.  The v1.1 any-combo registry expansion
    made mp_physics=18 selectable in the lsm4 slice; the re-run census
    measures a uniform 12 rows per MP across {0, 1, 6, 8, 10, 18} where
    the 2026-07-27 census had five MP values, and the worst-count peak is
    unchanged at 601 (pinned by the test above).

    Re-pinned later on 2026-07-30: 72 -> 96.  The WRF-owned MYNN/Noah-MP
    pairing adds four rows per MP (cu 0/1 x km_opt 1/4) and retires exactly
    the corresponding 24 pairing refusals.  The remaining 192 refusals are
    the unchanged surface-layer-off, MYNN half-suite, and PBL-off diffusion
    gates.

    Re-pinned for Lane C: 96 -> 192 measured and 192 -> 96 refused.
    WRF's complete 12-cell PBL/surface-layer table admits MYNN PBL with
    revised/classic MM5 surface layers and PBL-off with every represented
    surface layer, while vertical_diffusion_2 retires the km_opt=4/PBL-off
    rail.  The 96 remaining lsm4 refusals are exactly 72 WRF-fatal pairing
    rows and 24 ArWen structural surface-layer-off rows.

    Re-pinned for the LES lane, 2026-08-02: every count below is EXACTLY
    1.5x its previous value -- rows 792 -> 1188, rejected 360 -> 540, the
    lsm4 slice 192 -> 288, lsm4 refusals 96 -> 144 -- and nothing else
    moved.  ``selectable_scalar_values`` PROBES validate_run_config rather
    than transcribing it, precisely so it tracks the validator; the
    validator now admits km_opt=3, so the turbulence axis went from
    {1, 4} to {1, 3, 4} and every slice grew by exactly one third of its
    new size.  The lsm4 slice measures a uniform 96 rows per km_opt value.

    km_opt=2 is deliberately absent: the probe replaces one scalar on the
    REFERENCE config, which runs bl_pbl_physics=1 (YSU), and km_opt=2 is
    admitted only with the PBL off.  So the census sees three turbulence
    options here, not four, and would see km_opt=2 only on a PBL-off
    reference.

    Re-pinned on the 1.5 integration line: three independent movements
    land at once, and every count is their exact product.  The LES lane's
    x1.5 (turbulence axis {1, 4} -> {1, 3, 4}) and the mp=28 route's x7/6
    (microphysics axis grew from 6 to 7 selectable schemes -- the mp28
    lane never re-measured this census, so its movement surfaces here)
    give rows 792 -> 1386 and WRF-side rejections 360 -> 630; the SASE
    closure adds 448 rejections of its own (32 per (km_opt, mp) cell x 2
    x 7), all km_opt refusals, and zero measured rows.  lsm4: 224 rows
    (112 per km_opt value), 112 WRF-scheme refusals (Lane C's 96 carried
    across the axis growth), 112 SASE refusals.

    Re-pinned 2026-08-03 for the gray-zone lane: Shin-Hong
    (``bl_pbl_physics=11``) became routable and this census was never
    re-run, so the pins were stale rather than wrong -- 1386 is IDENTICAL
    at ``cf159eb2`` (the shipped 1.5.1), ``cfcfa9a9`` (the pre-merge lane
    tip that added the route) and ``61488333`` (the merge).  Scheme 11
    adds ONE pbl value to the sweep and every count below is that value's
    own slice, measured and not projected:

        rows        1386 -> 1722   (+336)
        rejected    1302 -> 1638   (+336)
        lsm4 rows    336 ->  420   ( +84)
        lsm4 WRF-scheme refusals
                     168 ->  252   ( +84)

    224 is exactly sfclay {1, 91} x lsm {0, 2, 3, 4} x km_opt {1, 4} x
    7 microphysics x cu {0, 1}, and the 224 refusals are the same product
    over the sfclay values WRF v4.6.1's table refuses with this scheme,
    {0, 5}.  Shin-Hong takes YSU's surface-layer pairing exactly, so its
    slice is the SAME SIZE as YSU's at every cut: the measured census
    reads 224 rows and 224 refusals for pbl=1 as well, and 56 of each in
    the lsm4 slice.  The whole sweep is now the 7 x 4 x 5 x 4 x 2 x 2 =
    2240 product, and 1148 + 1092 = 2240 accounts for all of it.

    MYJ RE-PIN, 2026-08-09 (lane/port-myj).  This port routes TWO new
    selectors -- sf_sfclay_physics=2 (Eta similarity) and bl_pbl_physics=2
    (MYJ) -- so it widens BOTH swept axes at once, the first change here to
    do that.  The cross-product goes 5 sfclay x 6 pbl x 4 lsm x 2 km_opt x
    7 mp x 2 cu = 3360, and 1260 + 2100 = 3360 accounts for all of it.
    Every number below was measured by
    `python tools/health_field_census.py --json configs/real74_4dom.toml`
    on this tree, and each delta is accounted for exactly:

        rows        1148 -> 1260  (+112)
        rejected    1092 -> 2100  (+1008)
        lsm4 rows    280 ->  308   (+28)
        lsm4 WRF-scheme refusals
                     168 ->  392  (+224)
        SASE refusals
                     448 ->  560  (+112), of which 112 are the MYJ
                                  pairing law rather than km_opt

    +112 rows is exactly MYJ's own slice: sfclay {2} x lsm {0,2,3,4} x
    km_opt {1,4} x 7 mp x cu {0,1}.  MYJ is the FIRST scheme here whose
    slice is smaller than YSU's, and the reason is the pairing: WRF admits
    YSU with two surface layers and MYJ with exactly one
    (module_physics_init.F:3770-3772), so MYJ measures 112 rows where YSU
    measures 224.  The +1008 refusals are 448 (MYJ with a surface layer it
    refuses) plus 560 (the Eta layer with a PBL it refuses) -- the pairing
    law, counted from both sides.  SASE's extra 112 are the 
    sfclay=2 x pbl=900 cells, which now hit validate_myj_pairing before
    they ever reach the km_opt check, so the "every SASE refusal names
    km_opt" assertion below is scoped to the cells where that is still
    true instead of being deleted.

    What did NOT move, again, is the point: the worst selectable
    descriptor count is unchanged at 632 of 1024.  MYJ adds nine 2-D
    surface fields and two 3-D columns under its own selectors, and the
    widest configuration is still someone else's.

    THE KM_OPT AXIS IS TWO VALUES, NOT THREE, ON THIS LINE.  The census
    probes validate_run_config for the selectable set rather than
    transcribing one, and the 1.7 validator refuses km_opt=3 unless
    bl_pbl_physics=0.  The reference config this sweep is built from runs
    a PBL, so km_opt=3 is genuinely not selectable here and the probe
    reports it honestly.  Every count below is therefore the 1.5.2 line's
    two-thirds where km_opt multiplies -- and the two numbers that do NOT
    multiply by it, the 632 peak and the 608 widest pbl11 row, are
    unchanged, which is the claim this file exists to hold.

    SASE is untouched in kind and scales in count with the axis: 448
    rejections, 112 of them in the lsm4 slice, zero measured rows.  The one thing worth checking beyond arithmetic is the
    ceiling, and scheme 11 does not approach it -- the widest pbl=11 row
    measures 608 descriptors against the unchanged 632 peak, so the peak,
    the peak set and the 392 of 1024 headroom all stay exactly where they
    were.

    What did NOT move is the point: the worst selectable descriptor count
    is unchanged at 632 of 1024.  A mixing option joining or leaving the
    selectable set ties the existing peak instead of moving it, which is
    what "km_opt does not change the inventory" has always claimed and is
    now measured across two values again.

    Re-pinned 2026-08-09 for the Milbrandt-Yau lane: mp_physics=9 became
    routable, so the microphysics axis is 8 values instead of 7 and the
    whole sweep is now 8 x 4 x 5 x 4 x 2 x 2 = 2560, with 1312 + 1248
    accounting for all of it.  Scheme 9 adds exactly ONE microphysics
    value and every count below is that value's own slice, measured:

        rows        1148 -> 1312   (+164)
        rejected    1092 -> 1248   (+156)
        sase rej     448 ->  512    (+64)
        lsm4 rows    280 ->  320    (+40)
        lsm4 WRF-scheme refusals
                     168 ->  192    (+24)
        lsm4 SASE refusals
                     112 ->  128    (+16)

    Its slice is the SAME SIZE as every other scheme's at every cut,
    which is the point: 164 rows like mp 0/1/6/8/10/18/28, not a
    partial one.  It was partial for one measurement -- 82 rows -- while
    gpuwm/core/kf.py refused Kain-Fritsch with mp=9, and that refusal was
    a real gap rather than a WRF fact (Registry.EM_COMMON:3025 declares
    qi and qs, so F_QI/F_QS are both true and KF_eta_CPS has its branch).
    Fixing the contract rather than pinning the halved number is why the
    slice is whole here.  The ceiling is untouched: the widest mp=9 row
    measures 420 descriptors against the unchanged 632 peak.

    1.9 INTEGRATION RE-PIN.  The two blocks above are each a lane's own
    measurement against 1.8.7, and neither is the number once both land:
    MYJ widens the PBL and surface-layer axes while Milbrandt-Yau widens
    the microphysics axis, so the movements MULTIPLY and adding the two
    deltas would under-count by 428 rows.  That intermediate sweep was
    5 sfclay x 6 pbl x 4 lsm x 2 km_opt x 8 mp x 2 cu = 3840, with
    1440 + 2400 accounting for all of it.

    P3 then adds the NINTH microphysics value, and WDM6 the TENTH.
    Neither touches a PBL, surface-layer, land-surface, cumulus or
    km_opt axis, so both movements are one more value on one axis.  The
    sweep is
    5 sfclay x 6 pbl x 4 lsm x 2 km_opt x 10 mp x 2 cu = 4800, and
    1800 + 3000 = 4800 accounts for all of it.  Every number below was
    re-measured by
    `python tools/health_field_census.py --json configs/real74_4dom.toml`
    on the merged tree, and each one is the per-microphysics-value rate
    carried across two more values:

        rows        1260 -> 1440 -> 1620 -> 1800  (180 per mp value)
        rejected    2100 -> 2400 -> 2700 -> 3000  (300 per mp value)
        lsm4 rows    308 ->  352 ->  396 ->  440  ( 44 per mp value)
        lsm4 WRF-scheme refusals
                     392 ->  448 ->  504 ->  560  ( 56 per mp value)
        SASE refusals
                     560 ->  640 ->  720 ->  800  ( 80 per mp value), of
                                   which 160 are the MYJ pairing law
                                   rather than km_opt

    The per-value rates are asserted, not assumed: the measured census
    reports exactly 180 rows and 300 refusals for EVERY one of the ten
    microphysics values, mp=50 and mp=16 included, so each new scheme's
    slice is a full one.

    The ceiling is untouched by any of the four lanes: the peak is
    still 632 of 1024 at mp18-lsm3-pbl5-sfclay5-cu1-km1.  Neither MYJ's
    surface fields, nor Milbrandt-Yau's twelve transported moments, nor
    P3's rime pair and previous-step carriers, nor WDM6's three
    transported numbers produce the widest configuration.
    """
    report = four_domain_census
    assert len(report["rows"]) == 1800
    # Re-pinned when the SASE closure joined the dispatch table.  The
    # census derives its sweep from PHYSICS_SLOT_DISPATCH on purpose --
    # "an admitted scheme joins the census the moment it is routed" --
    # so a new routed selector necessarily moves this number.  The rows
    # added are accounted for EXACTLY, and the accounting is asserted
    # below rather than asserted away.  (Union re-pin on the 1.5
    # integration line: the LES lane widened the km_opt sweep, the SASE
    # lane added the routed pbl900 selector; both movements land here.
    # Gray-zone lane, 2026-08-03: the routed Shin-Hong selector adds its
    # own 336 measured rows and 336 refusals, the second scheme to prove
    # this sentence by moving these numbers.)
    assert len(report["rejected"]) == 3000
    sase = [row for row in report["rejected"] if "pbl900" in row["selection"]]
    assert len(sase) == 800, len(sase)  # 80 per mp value x 10 mp values
    # Every one of them is the same refusal, and it is a real one: the
    # closure supplies the mixing km_opt would apply, so it is admitted
    # only at km_opt=0 and this census never sweeps km_opt=0.
    # SPLIT, not relaxed.  The 160 cells that pair SASE with the Eta
    # surface layer are refused by validate_myj_pairing before the km_opt
    # check is reached, so they carry the MYJ message; every OTHER SASE
    # refusal still names km_opt, and that is asserted separately rather
    # than absorbed into a weaker predicate.  16 per microphysics value,
    # so the two 1.9 microphysics ports carry it 144 to 160.
    myj_paired = [row for row in sase if "sfclay2" in row["selection"]]
    assert len(myj_paired) == 160, len(myj_paired)
    assert all("Eta similarity" in row["reason"] for row in myj_paired)
    assert all("km_opt" in row["reason"]
               for row in sase if row not in myj_paired)
    # THE GAP THIS RECORDS, deliberately, rather than hiding: because the
    # sweep never tries km_opt=0, the closure contributes ZERO measured
    # rows.  This census does not cover SASE at all.  Widening the sweep
    # to pair each PBL scheme with the km_opt values that scheme actually
    # admits is the fix, and it will move the rows count as well as this
    # number.
    assert not [row for row in report["rows"] if "pbl900" in row["selection"]]
    lsm4_rows = [row for row in report["rows"]
                 if row["sf_surface_physics"] == 4]
    assert len(lsm4_rows) == 440, (
        f"the measured lsm4 slice is {len(lsm4_rows)} rows, not the 440 the "
        "1.9 census recorded (44 per microphysics value across 10 values: "
        "the MYJ line's 44 per value, unchanged, because Milbrandt-Yau, P3 "
        "and WDM6 each add a microphysics value and none adds a PBL or "
        "surface-layer one); re-run and re-pin")
    budget_refusals = [entry["selection"] for entry in report["rejected"]
                       if "Noah-MP column budget" in entry["reason"]]
    assert not budget_refusals, (
        "the column budget refused combinations on a case with no domain "
        "wider than the measured 360,000-column ceiling: "
        f"{budget_refusals[:5]}")
    refused = [entry for entry in report["rejected"]
               if "-lsm4-" in entry["selection"]]
    # The Lane C claim is about the WRF authority, so it is asserted on
    # the WRF schemes and NOT widened to absorb a scheme WRF does not
    # have.  SASE's own lsm4 refusals are counted beside it: widening
    # the WRF number would have quietly retired a measured WRF-authority
    # number to accommodate an unrelated addition, which is the failure
    # mode this whole census exists to prevent.
    wrf_refused = [entry for entry in refused
                   if "pbl900" not in entry["selection"]]
    assert len(wrf_refused) == 560, (
        f"{len(wrf_refused)} WRF-scheme lsm4 refusals against the 560 "
        "recorded on the 1.9 line -- Lane C's 96 (72 WRF-fatal "
        "PBL/surface-layer rows and 24 active-LSM rows without an ArWen "
        "surface-exchange writer) carried across the two km_opt values, "
        "giving the 1.5 line's 112, plus Shin-Hong's own 56, plus MYJ's "
        "224 -- lsm4 cells that ask for the MYJ PBL with a surface layer it "
        "refuses and the Eta surface layer with a PBL it refuses, the same "
        "law counted from both sides -- all of it per microphysics value at "
        "56 per value, so Milbrandt-Yau, P3 and WDM6 each multiply it up by "
        "one more value, 7 to 8 to 9 to 10")
    sase_refused = [entry for entry in refused
                    if "pbl900" in entry["selection"]]
    assert len(sase_refused) == 200, len(sase_refused)
    myj_paired_lsm4 = [entry for entry in sase_refused
                       if "sfclay2" in entry["selection"]]
    assert len(myj_paired_lsm4) == 40, len(myj_paired_lsm4)
    assert all("km_opt" in entry["reason"] for entry in sase_refused
               if entry not in myj_paired_lsm4)


def test_worst_selectable_count_stays_inside_the_early_warning_band(
        four_domain_census):
    """Fire while there is still room to decide, not at the cap."""
    report = four_domain_census
    assert report["worst_count"] <= EARLY_WARNING_COUNT, (
        f"the worst selectable count {report['worst_count']} crossed the "
        f"{EARLY_WARNING_COUNT} early-warning band on the way to "
        f"{report['max_health_fields']}.  Raising this constant is not the "
        "fix; decide whether the ceiling should move (it costs "
        f"{CEILING_METADATA_BYTES_PER_DOMAIN} B per domain, so cost is not "
        "the obstacle) or whether the inventory should stop growing.")


def test_the_ceiling_gate_can_actually_fail(monkeypatch):
    """Show the failing form.

    A gate never observed to fail is not evidence.  Drop the ceiling below the
    inventory and confirm the census reports a breach -- and, critically, that
    it reports it as a BREACH and not as "the production validators refuse
    this combination".  The breach is raised as a bare ``ValueError`` from
    inside ``collect_state_fields``, so a census that files every
    ``ValueError`` as a refusal reports the one thing it exists to find as
    proof that the thing cannot happen.  That is exactly what the pre-crash
    version did.

    The breach object is produced by the real ``collect_state_fields``, not
    hand-made, so the classifier is tested against production's own exception.
    """
    from gpuwm.core import health

    state = type("MinimalState", (), {})()
    state.u = np.zeros((2, 3, 4), dtype=np.float32)
    state.p = np.full((2, 3, 4), 1.0e4, dtype=np.float32)

    assert len(health.collect_state_fields(state, backend="gpu")) == 2
    monkeypatch.setattr(health, "MAX_HEALTH_FIELDS", 1)
    with pytest.raises(ValueError) as caught:
        health.collect_state_fields(state, backend="gpu")
    breach = caught.value
    assert "MAX_HEALTH_FIELDS" in str(breach)
    assert census._is_ceiling_breach(breach), (
        "a real ceiling breach was not recognised as one, so the census "
        "would file it under 'the validators refuse this'")
    assert census._measured_count_from_breach(breach) == 2

    def raise_breach(*args, **kwargs):
        raise breach

    monkeypatch.setattr(census, "domain_descriptor_names", raise_breach)
    report = census.experiment_census(
        FOUR_DOMAIN_CONFIG,
        selections=[census.Selection(18, 4, 1, 1, 1, 1)])
    assert report["rejected"] == [], (
        "a ceiling breach was filed as a production refusal")
    assert len(report["over_ceiling"]) == 1
    assert report["over_ceiling"][0]["count"] == 2
    assert report["rows"] == []


def test_the_census_cli_exits_nonzero_on_a_breach(monkeypatch, capsys):
    """And the shipped entry point reports it, so a sweep cannot pass quietly.

    Verified against ``main`` -- the thing that actually runs -- rather than
    against ``experiment_census`` alone.
    """
    monkeypatch.setattr(
        census, "experiment_census",
        lambda *a, **k: {
            "config": "x", "max_health_fields": 1024,
            "ceiling_metadata_bytes_per_domain": 49152,
            "selectable_axes": {}, "domains": [], "rows": [], "rejected": [],
            "over_ceiling": [{"selection": "s", "grid_id": 4, "count": 1200,
                              "reason": "over"}],
            "worst_selection": "s", "worst_count": 1200, "headroom": -176})
    monkeypatch.setattr(census, "install_host_array_backend", lambda: None)
    assert census.main([str(FOUR_DOMAIN_CONFIG)]) == 1
    assert "OVER CEILING" in capsys.readouterr().out


def test_a_measurement_failure_is_never_filed_as_a_refusal(monkeypatch):
    """``MemoryError`` is not evidence that a configuration is unselectable.

    The pre-crash census caught every exception and recorded it as "the
    production validators refuse (not user-selectable)".  Its sweep exhausted
    host memory partway through, so 58 of 140 configuration-valid combinations
    went unmeasured while the report asserted they could not be run -- and an
    unmeasured combination is precisely the one that might be over the
    ceiling.
    """
    def boom(*args, **kwargs):
        raise MemoryError("Unable to allocate 9.35 MiB")

    monkeypatch.setattr(census, "domain_descriptor_names", boom)
    with pytest.raises(MemoryError):
        census.experiment_census(
            FOUR_DOMAIN_CONFIG,
            selections=[census.Selection(18, 4, 1, 1, 1, 1)])


# ---------------------------------------------------------------------------
# The structure of the number -- i.e. why the projections were wrong
# ---------------------------------------------------------------------------

def test_per_domain_counts_are_not_one_domain_times_four(four_domain_census):
    """A root and a child are different inventories; neither scales.

    This is the assumption both earlier estimates rested on.  It is false in
    both directions: the peak is not the sum (the cap is per DomainState), and
    the per-domain counts are not equal (the root has no nest tables).
    """
    report = four_domain_census
    per_domain = _child_rows(report)[report["worst_selection"]]
    assert per_domain[1] == WORST_ROOT_COUNT
    assert per_domain[2] == per_domain[3] == per_domain[4] == WORST_CHILD_COUNT
    assert per_domain[1] != per_domain[4], (
        "root and child counts are equal, so the per-domain families changed")
    assert report["worst_count"] == max(per_domain.values())
    assert report["worst_count"] < sum(per_domain.values()), (
        "the reported exposure looks like a sum across the nest; the cap is "
        "per DomainState")


def test_child_domains_carry_the_rolling_boundary_descriptors(
        four_domain_census):
    """A child's inventory includes what its first FORCE attaches.

    ``NestCoupler.force`` calls ``attach_nest_boundaries``, which sets
    ``state._lateral_boundary_device``; a child has no ``lbc_forcing_tables``,
    so ``collect_state_fields`` takes the unpacked branch and walks that
    object -- 178 descriptors on the worst combination.  A census that stopped
    at scratch allocation reported 423 for a child that in fact runs at 570,
    and understating the count is the dangerous direction.  If this assertion
    ever fails because the census stopped attaching the boundary, the peak it
    reports is fiction.
    """
    report = four_domain_census
    worst = next(row for row in report["rows"]
                 if row["selection"] == report["worst_selection"])
    root = {k: v for k, v in worst["breakdown"]["1"].items()}
    child = {k: v for k, v in worst["breakdown"]["4"].items()}
    assert child["lbc"] == 178, f"child lbc family is {child}"
    assert child["nest.scratch"] == 196
    assert root["lbc"] == 2, (
        "the root should take the packed one-descriptor LBC path")
    assert "nest.scratch" not in root


def test_grid_dimensions_do_not_change_the_descriptor_count(
        four_domain_census):
    """The count is a property of the configuration, not of its size.

    Evidence from the same sweep: d02 is 500x400 with parent_grid_ratio 4,
    d03 is 501x501 with ratio 3, d04 is 600x600 with ratio 3, and all three
    return identical counts on every measured combination.  That is why the
    number is quotable for the flagship without re-running it per resolution.
    """
    report = four_domain_census
    dims = {d["grid_id"]: (d["nx"], d["ny"]) for d in report["domains"]}
    assert len({dims[2], dims[3], dims[4]}) == 3, (
        "the three children now share dimensions, so this proves nothing")
    for selection, per_domain in _child_rows(report).items():
        assert per_domain[2] == per_domain[3] == per_domain[4], (
            f"{selection} counts differ across children of different sizes: "
            f"{per_domain}")


def test_the_census_reproduces_the_recorded_527_descriptor_step(
        four_domain_census):
    """Calibration against the only figure written down before this existed.

    ``health.py``'s ceiling justification says a four-domain NSSL-2 step
    reaches 527.  It does -- exactly -- at ``mp18-lsm2-pbl0-sfclay1-cu0``.
    Reproducing a number recorded independently of this tool is the strongest
    check available that it is counting the same thing the ceiling was set
    against, and it settles what the older "~707 of 1024" projection was: a
    per-domain baseline added to itself four times.
    """
    rows = _child_rows(four_domain_census)
    assert CALIBRATION_SELECTION in rows, (
        f"{CALIBRATION_SELECTION} is no longer selectable, so the 527 "
        "calibration cannot be checked; find the combination that now "
        "reproduces health.py's cited figure, or amend that comment")
    assert rows[CALIBRATION_SELECTION][4] == CALIBRATION_CHILD_COUNT


# ---------------------------------------------------------------------------
# Completeness of the sweep, and what the ceiling costs
# ---------------------------------------------------------------------------

def test_the_sweep_accounts_for_every_axis_combination(four_domain_census):
    """Measured + refused == the whole cross-product: nothing was dropped.

    The axes themselves are not pinned to a literal -- admitting a scheme
    legitimately widens them, and the sweep then measures the wider set.  What
    must hold is that every point in the cross-product is either measured or
    explicitly refused by a production validator, so no combination can go
    quietly uncounted.
    """
    report = four_domain_census
    axes = report["selectable_axes"]
    expected = 1
    for values in axes.values():
        assert values, f"an axis came back empty: {axes}"
        expected *= len(values)
    assert len(report["rows"]) + len(report["rejected"]) == expected, (
        f"{expected} combinations in the cross-product but "
        f"{len(report['rows'])} measured + {len(report['rejected'])} refused")
    assert report["rejected"], (
        "no combination was refused, which means the refusal path is "
        "untested here")


def test_the_census_measured_this_checkout(four_domain_census):
    """The subprocess measured THIS tree, not the pip-installed one.

    Compared against the production table imported in-process, so a census
    that resolved ``gpuwm`` to another worktree disagrees here instead of
    quietly reporting someone else's inventory.
    """
    from gpuwm.core.physics import PHYSICS_SLOT_DISPATCH

    axes = four_domain_census["selectable_axes"]
    for selector in ("sf_surface_physics", "bl_pbl_physics",
                     "sf_sfclay_physics"):
        assert axes[selector] == sorted(PHYSICS_SLOT_DISPATCH[selector]), (
            f"the census reported {selector}={axes[selector]} but this tree "
            f"routes {sorted(PHYSICS_SLOT_DISPATCH[selector])}; the census "
            "measured a different checkout")


def test_the_ceiling_costs_fixed_metadata_not_field_storage(
        four_domain_census):
    """Record what raising ``MAX_HEALTH_FIELDS`` would actually cost.

    Seven ``integration_health_*`` slots are sized by the ceiling rather than
    by the measured count, and they are per ``DomainState``: 48 KiB per domain
    at 1024, 192 KiB across a four-domain nest.  Doubling the ceiling adds
    that much again.  VRAM is a correctness bar on this hardware, so the cost
    had to be established rather than assumed -- and having established it,
    nobody should argue the ceiling cannot move for VRAM reasons.  It is a
    tripwire against unnoticed growth.
    """
    report = four_domain_census
    assert (report["ceiling_metadata_bytes_per_domain"]
            == CEILING_METADATA_BYTES_PER_DOMAIN)
    assert report["max_health_fields"] == 1024
    per_descriptor = (report["ceiling_metadata_bytes_per_domain"]
                      / report["max_health_fields"])
    assert per_descriptor == 48.0, (
        "the per-descriptor metadata footprint changed; re-derive the cost of "
        "the ceiling before quoting it")
