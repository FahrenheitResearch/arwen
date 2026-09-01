"""The supervisor's state machine, driven entirely by injected callables.

No GPU, no model, no NetCDF.  Everything the supervisor decides is a
decision about records, and every one of those decisions is a burn this
program has already paid for:

  * a child one child-step behind its parent is drift, and drift that
    nothing reports is how a tree loses an hour;
  * an analysis with 41,882 nonzero cells whose resumed state hashes
    identical to the background was DROPPED, and "dropped" and "worked"
    are otherwise indistinguishable;
  * an exact-zero increment that nonetheless MOVED the state means
    rehydration perturbed it, which is the same instrument failing in
    the other direction.
"""

from datetime import datetime, timezone

import pytest

from gpuwm.cycle.clock import CycleClock
from gpuwm.cycle.contracts import CycleRefusal
from gpuwm.cycle.ledger import CycleLedger
from gpuwm.cycle.supervisor import CycleSupervisor

ANCHOR = datetime(2026, 8, 14, 18, 0, tzinfo=timezone.utc)


def _clock(n_cycles=2, cycle_seconds=960.0):
    return CycleClock.build(epoch_anchor=ANCHOR, parent_dt_seconds=120.0,
                            cycle_seconds=cycle_seconds, n_cycles=n_cycles)


def _parent(clock):
    def advance_parent(cycle_index, anchor_in):
        return {"parent_ticks": clock.boundary_ticks(cycle_index),
                "anchor_ticks": clock.boundary_ticks(cycle_index),
                "kind": "replay"}
    return advance_parent


def _ingestion(*, nonzero, background="bg-sha", analysis="an-sha"):
    return {"background_sha256": background,
            "increment_sha256": "inc-sha",
            "analysis_sha256": analysis,
            "increment_nonzero_cells": nonzero,
            "increment_l2": {"u": 1.83, "v": 2.04, "qr": 3.1e-4},
            "increment_fields": ["u", "v", "qr"]}


def _supervisor(tmp_path, clock=None, **kwargs):
    clock = clock or _clock()
    kwargs.setdefault("advance_parent", _parent(clock))
    return CycleSupervisor(clock=clock, ledger=CycleLedger(tmp_path),
                           root=tmp_path, **kwargs)


# -- the arming triple ---------------------------------------------------

def test_arming_detects_child_drift(tmp_path):
    """A child one child-step behind the parent must halt the cycle."""
    clock = _clock()
    ratio = clock.child_ratio(30.0)          # 4 child steps per parent step

    def advance_children(cycle_index, anchor, placements):
        behind = clock.boundary_ticks(cycle_index) // ratio.child_ticks - 1
        return [{"grid_id": 2, "state": "LIVE",
                 "child_ticks": behind,
                 "child_step_ticks": ratio.child_ticks}]

    def plan_children(cycle_index, anchor):
        return [{"grid_id": 2, "state": "PLANNED", "lat": 35.0, "lon": -97.0}]

    supervisor = _supervisor(tmp_path, clock=clock,
                             plan_children=plan_children,
                             advance_children=advance_children)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run()
    observed = excinfo.value.observed
    assert observed["reason"] == "CLOCK_UNARMED"
    assert observed["grid_id"] == 2
    assert observed["expected_ticks"] == clock.boundary_ticks(1)
    assert observed["observed_child_ticks_scaled"] == (
        clock.boundary_ticks(1) - ratio.child_ticks)
    assert CycleLedger(tmp_path).state()["halt_reason"] == "CLOCK_UNARMED"


def test_arming_detects_a_parent_off_its_boundary(tmp_path):
    clock = _clock()

    def advance_parent(cycle_index, anchor_in):
        ticks = clock.boundary_ticks(cycle_index) - clock.parent_step_ticks
        return {"parent_ticks": ticks, "anchor_ticks": ticks}

    supervisor = _supervisor(tmp_path, clock=clock,
                             advance_parent=advance_parent)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run()
    assert excinfo.value.observed["reason"] == "CLOCK_UNARMED"
    assert excinfo.value.observed["observed_parent_ticks"] == (
        clock.boundary_ticks(1) - clock.parent_step_ticks)


def test_armed_cycle_records_the_triple_in_the_receipt(tmp_path):
    clock = _clock(n_cycles=1)
    ratio = clock.child_ratio(30.0)

    def advance_children(cycle_index, anchor, placements):
        return [{"grid_id": 2, "state": "LIVE",
                 "child_ticks": (clock.boundary_ticks(cycle_index)
                                 // ratio.child_ticks),
                 "child_step_ticks": ratio.child_ticks}]

    supervisor = _supervisor(tmp_path, clock=clock,
                             advance_children=advance_children)
    result = supervisor.run()
    assert result["cycles_completed"] == [1]
    receipt = result["receipts"][0]
    assert receipt["arming"]["armed"] is True
    assert receipt["arming"]["parent_ticks"] == clock.boundary_ticks(1)
    assert receipt["arming"]["anchor_ticks"] == clock.boundary_ticks(1)
    assert receipt["arming"]["children"][0]["scaled_ticks"] == (
        clock.boundary_ticks(1))


# -- the ingestion gate, both arms ---------------------------------------

def test_dropped_analysis_is_caught(tmp_path):
    def analyse(cycle_index, anchor):
        return {"state": "APPLIED",
                "ingestion": _ingestion(nonzero=1000, background="same-sha",
                                        analysis="same-sha")}

    supervisor = _supervisor(tmp_path, analyse=analyse)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run()
    assert excinfo.value.observed["reason"] == "ANALYSIS_NOT_INGESTED"
    assert excinfo.value.observed["increment_nonzero_cells"] == 1000
    assert "1000" in str(excinfo.value)
    assert "same-sha" in str(excinfo.value)


def test_null_arm_requires_bit_stability(tmp_path):
    """Both directions: an unmoved null arm passes, a moved one halts."""
    def moved(cycle_index, anchor):
        return {"state": "NULL_ARM",
                "ingestion": _ingestion(nonzero=0, background="bg",
                                        analysis="drifted")}

    supervisor = _supervisor(tmp_path, analyse=moved)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run()
    assert excinfo.value.observed["reason"] == "ANALYSIS_NOT_INGESTED"
    assert excinfo.value.observed["arm"] == "NULL_ARM"

    def unmoved(cycle_index, anchor):
        return {"state": "NULL_ARM",
                "ingestion": _ingestion(nonzero=0, background="bg",
                                        analysis="bg")}

    other = tmp_path / "null-arm-stable"
    supervisor = _supervisor(other, analyse=unmoved)
    result = supervisor.run()
    assert result["cycles_completed"] == [1, 2]
    assert result["halted"] is False


def test_rejected_analysis_halts(tmp_path):
    def analyse(cycle_index, anchor):
        return {"state": "REJECTED", "reason": "LETKF rejected the volume",
                "observation_count": 0}

    supervisor = _supervisor(tmp_path, analyse=analyse)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run()
    assert excinfo.value.observed["reason"] == "ANALYSIS_REJECTED"


# -- failure policy ------------------------------------------------------

def test_missing_obs_runs_forecast_only_then_halts_at_budget(tmp_path):
    clock = _clock(n_cycles=6)
    seen = []

    def analyse(cycle_index, anchor):
        seen.append(cycle_index)
        return None

    supervisor = _supervisor(tmp_path, clock=clock, analyse=analyse,
                             max_forecast_only_cycles=3)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run()
    assert excinfo.value.observed["reason"] == "STALE_ANALYSIS_BUDGET_EXHAUSTED"
    assert excinfo.value.observed["forecast_only_streak"] == 4
    assert seen == [1, 2, 3, 4]
    ledger = CycleLedger(tmp_path)
    states = [r.get("state") for r in ledger.settle()
              if r.get("event") == "analysis"]
    assert states == ["SKIPPED_NO_OBS"] * 4
    # three cycles completed forecast-only; the fourth halted
    assert ledger.state()["last_completed_cycle"] == 3


def test_refused_placement_does_not_kill_the_cycle(tmp_path):
    clock = _clock(n_cycles=1)

    def plan_children(cycle_index, anchor):
        return [{"grid_id": 2, "state": "REFUSED",
                 "reason": "child leaves the parent domain",
                 "lat": 35.0, "lon": -97.0}]

    supervisor = _supervisor(tmp_path, clock=clock,
                             plan_children=plan_children)
    result = supervisor.run()
    assert result["cycles_completed"] == [1]
    assert result["halted"] is False
    receipt = result["receipts"][0]
    assert receipt["placements"][0]["state"] == "REFUSED"
    assert receipt["children"] == []
    assert receipt["refusals"][0]["grid_id"] == 2
    assert CycleLedger(tmp_path).state()["live_children"] == {"2": "REFUSED"}


def test_clamped_placement_requires_the_flag(tmp_path):
    clock = _clock(n_cycles=1)

    def plan_children(cycle_index, anchor):
        return [{"grid_id": 2, "state": "PLANNED", "clamped": True,
                 "lat": 35.0, "lon": -97.0}]

    supervisor = _supervisor(tmp_path, clock=clock,
                             plan_children=plan_children)
    result = supervisor.run()
    # Default-off: a clamp is REFUSED, the slot goes unfilled, the cycle
    # survives.  Clamping silently is how a nest ends up somewhere else.
    assert result["receipts"][0]["placements"][0]["state"] == "REFUSED"
    assert result["halted"] is False

    allowed = _supervisor(tmp_path / "clamp-allowed", clock=clock,
                          plan_children=plan_children,
                          advance_children=lambda c, a, p: [
                              {"grid_id": 2, "state": "LIVE",
                               "child_ticks": clock.boundary_ticks(c) // 30000,
                               "child_step_ticks": 30000}],
                          allow_placement_clamp=True)
    receipt = allowed.run()["receipts"][0]
    assert receipt["placements"][0]["state"] == "PLANNED"
    assert receipt["placements"][0]["clamped"] is True


def test_diverged_child_is_retired_and_the_cycle_continues(tmp_path):
    clock = _clock(n_cycles=2)

    def plan_children(cycle_index, anchor):
        return [{"grid_id": 2, "state": "PLANNED", "lat": 35.0, "lon": -97.0}]

    def advance_children(cycle_index, anchor, placements):
        if cycle_index == 1:
            return [{"grid_id": 2, "state": "DIVERGED", "field": "w",
                     "cell": [12, 44, 3], "value": 1.4e12,
                     "child_ticks": 0, "child_step_ticks": 30000}]
        return []

    supervisor = _supervisor(tmp_path, clock=clock,
                             plan_children=plan_children,
                             advance_children=advance_children)
    result = supervisor.run()
    assert result["cycles_completed"] == [1, 2]
    assert result["halted"] is False
    receipt = result["receipts"][0]
    assert receipt["children"][0]["state"] == "RETIRED"
    assert receipt["children"][0]["retired_because"] == "DIVERGED"
    assert receipt["children"][0]["field"] == "w"
    assert CycleLedger(tmp_path).state()["live_children"] == {"2": "RETIRED"}


def test_parent_divergence_halts(tmp_path):
    def advance_parent(cycle_index, anchor_in):
        raise RuntimeError("vertical velocity refusal at k=31, w=612 m/s")

    supervisor = _supervisor(tmp_path, advance_parent=advance_parent)
    with pytest.raises(CycleRefusal) as excinfo:
        supervisor.run()
    assert excinfo.value.observed["reason"] == "PARENT_DIVERGED"
    assert "612" in str(excinfo.value.observed["error"])


def test_a_diverged_child_never_costs_the_parent(tmp_path):
    """The deliberate asymmetry, asserted rather than described."""
    clock = _clock(n_cycles=3)

    def plan_children(cycle_index, anchor):
        return [{"grid_id": g, "state": "PLANNED"} for g in (2, 3)]

    def advance_children(cycle_index, anchor, placements):
        return [{"grid_id": 2, "state": "DIVERGED", "field": "w",
                 "child_ticks": 0, "child_step_ticks": 30000},
                {"grid_id": 3, "state": "LIVE",
                 "child_ticks": clock.boundary_ticks(cycle_index) // 30000,
                 "child_step_ticks": 30000}]

    supervisor = _supervisor(tmp_path, clock=clock,
                             plan_children=plan_children,
                             advance_children=advance_children)
    result = supervisor.run()
    assert result["cycles_completed"] == [1, 2, 3]


# -- resume --------------------------------------------------------------

def test_resume_skips_completed_cycles(tmp_path):
    clock = _clock(n_cycles=4)
    supervisor = _supervisor(tmp_path, clock=CycleClock.build(
        epoch_anchor=ANCHOR, parent_dt_seconds=120.0, cycle_seconds=960.0,
        n_cycles=2))
    assert supervisor.run()["cycles_completed"] == [1, 2]

    seen = []

    def advance_parent(cycle_index, anchor_in):
        seen.append(cycle_index)
        return {"parent_ticks": clock.boundary_ticks(cycle_index),
                "anchor_ticks": clock.boundary_ticks(cycle_index)}

    resumed = CycleSupervisor(clock=clock, ledger=CycleLedger(tmp_path),
                              root=tmp_path, advance_parent=advance_parent)
    result = resumed.run(resume=True)
    assert seen == [3, 4]
    assert result["cycles_completed"] == [3, 4]
    assert result["resumed_from_cycle"] == 3


def test_a_halted_ledger_refuses_to_run_again(tmp_path):
    def analyse(cycle_index, anchor):
        return {"state": "REJECTED", "detail": "no volume passed QC"}

    supervisor = _supervisor(tmp_path, analyse=analyse)
    with pytest.raises(CycleRefusal):
        supervisor.run()
    again = _supervisor(tmp_path, analyse=analyse)
    with pytest.raises(CycleRefusal) as excinfo:
        again.run()
    assert excinfo.value.observed["halt_reason"] == "ANALYSIS_REJECTED"
