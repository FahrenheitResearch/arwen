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
#: Headroom remains 392 descriptors under the unchanged 1024 ceiling.
WORST_MEASURED_COUNT = 632
#: The worst combination(s).  MYNN/RUC with Kain-Fritsch is widest; km_opt
#: does not change the inventory.
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


def test_noahmp_slice_includes_the_wrf_owned_mynn_pairing(
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
    """
    report = four_domain_census
    lsm4_rows = [row for row in report["rows"]
                 if row["sf_surface_physics"] == 4]
    assert len(lsm4_rows) == 96, (
        f"the measured lsm4 slice is {len(lsm4_rows)} rows, not the 96 the "
        "2026-07-30 MYNN-pairing census recorded; re-run and re-pin")
    budget_refusals = [entry["selection"] for entry in report["rejected"]
                       if "Noah-MP column budget" in entry["reason"]]
    assert not budget_refusals, (
        "the column budget refused combinations on a case with no domain "
        "wider than the measured 360,000-column ceiling: "
        f"{budget_refusals[:5]}")
    refused = [entry for entry in report["rejected"]
               if "-lsm4-" in entry["selection"]]
    assert len(refused) == 192, (
        f"{len(refused)} lsm4 refusals against the 192 recorded after "
        "retiring the 24 MYNN/Noah-MP pairing refusals (120 MYNN "
        "half-suite, 48 LSM-without-surface-layer, 24 km_opt=4/pbl0)")


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
