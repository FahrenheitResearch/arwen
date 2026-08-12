"""CPU contracts for the relocation runner: schedule, receipts, restart.

The runner is exercised against the same CPU fake tree the staging tests
prove the primitive on, so a scheduled move here executes the REAL
relocate_child -- plan, admissibility, donor alignment, transplant --
with only the device replaced by numpy.  What is asserted:

- config governance for the leg-2 keys (cadence_seconds,
  [[relocation.move]], their interaction with [relocation.follow]);
- cadence gating and manual-itinerary consumption at cycle boundaries,
  through the storm tracker's exact provider signature;
- receipts: time, offsets (requested AND executed), overlap fraction,
  fields transplanted, fill counts and provenance, staging block;
- the restart posture: every executed move chains the live fingerprint,
  and a checkpoint header carrying the relocation block refuses with the
  promises-nothing posture BY NAME.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_relocation import (RESTART_ACROSS_MOVE_POSTURE,
                                        RelocationRefusal,
                                        mark_fingerprint_across_move)
from gpuwm.core.relocation_runner import (ManualMoveProvider,
                                          RelocationRunner)
from gpuwm.experiment import (RelocationConfig, ScheduledRelocationMove,
                              _build_relocation)

from test_nest_relocation_staging import (_cpu_tree, _footprint_statics,
                                          _initializer, _ramp_state)


# ---------------------------------------------------------------------------
# Config governance
# ---------------------------------------------------------------------------

def _domains(history_interval_s=900.0):
    """Stand-ins carrying exactly what _build_relocation reads.

    parent_id and history_interval_s joined the list for the refl-stash
    cadence contract (issue #111): the tracker watches grid_id 2's
    PARENT, so the root's history interval is the one that gates the
    follow cadence.
    """
    return [
        SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                        time_step_fract_num=0, time_step_fract_den=1,
                        history_interval_s=history_interval_s),
        SimpleNamespace(grid_id=2, parent_id=1, time_step=None,
                        history_interval_s=history_interval_s),
    ]


def _build(raw, run_seconds=3600.0):
    return _build_relocation(raw, "test.toml", _domains(), run_seconds)


def _raw(**over):
    table = {"enabled": True, "grid_id": 2, "max_move_parent_cells": 4}
    table.update(over)
    return {"relocation": table}


_MOVE = {"at_seconds": 600.0, "di_parent_cells": 2}
_FOLLOW = dict(field="uh", threshold=25.0, fallback_threshold=40.0,
               search_margin_cells=10, min_shift_cells=2,
               max_shift_cells=4, cooldown_seconds=900.0)


def test_manual_itinerary_loads_and_echoes():
    cfg = _build(_raw(cadence_seconds=600.0, move=[
        _MOVE, {"at_seconds": 1200.0, "dj_parent_cells": -1}]))
    assert cfg.cadence_seconds == 600.0
    assert [m.to_json() for m in cfg.moves] == [
        {"at_seconds": 600.0, "di_parent_cells": 2, "dj_parent_cells": 0},
        {"at_seconds": 1200.0, "di_parent_cells": 0, "dj_parent_cells": -1},
    ]
    echo = cfg.receipt()
    assert echo["cadence_seconds"] == 600.0
    assert len(echo["moves"]) == 2 and echo["follow"] is None


def test_move_row_unknown_key_refuses_by_name():
    with pytest.raises(ValueError, match="di_parent_cell"):
        _build(_raw(move=[{"at_seconds": 600.0, "di_parent_cell": 2}]))


def test_move_row_requires_at_seconds():
    with pytest.raises(ValueError, match="at_seconds"):
        _build(_raw(move=[{"di_parent_cells": 2}]))


def test_move_rows_must_increase_strictly():
    with pytest.raises(ValueError, match="strictly"):
        _build(_raw(move=[_MOVE, dict(_MOVE)]))


def test_move_must_land_on_a_root_step():
    with pytest.raises(ValueError, match="whole number of root steps"):
        _build(_raw(move=[{"at_seconds": 90.0, "di_parent_cells": 1}]))


def test_move_past_the_run_end_refuses():
    with pytest.raises(ValueError, match="past the end"):
        _build(_raw(move=[_MOVE]), run_seconds=600.0)


def test_move_off_the_cadence_refuses():
    with pytest.raises(ValueError, match="cadence"):
        _build(_raw(cadence_seconds=480.0, move=[_MOVE]))


def test_cadence_must_land_on_a_root_step():
    with pytest.raises(ValueError, match="whole number of root steps"):
        _build(_raw(cadence_seconds=90.0, move=[
            {"at_seconds": 180.0, "di_parent_cells": 1}]))


def test_cadence_without_a_follow_source_refuses():
    with pytest.raises(ValueError, match="no follow source"):
        _build(_raw(cadence_seconds=600.0))


def test_two_follow_sources_refuse():
    with pytest.raises(ValueError, match="two"):
        _build(_raw(move=[_MOVE], follow=dict(_FOLLOW)))


def test_moves_on_a_disabled_relocation_refuse():
    with pytest.raises(ValueError, match="enabled"):
        _build({"relocation": {"enabled": False, "move": [_MOVE]}})


def test_fractional_cells_refuse():
    with pytest.raises(ValueError, match="whole number of parent cells"):
        ScheduledRelocationMove(at_seconds=600.0, di_parent_cells=1.5)


def test_follow_table_with_cadence_loads():
    cfg = _build(_raw(cadence_seconds=900.0, follow=dict(_FOLLOW)))
    assert cfg.cadence_seconds == 900.0
    assert cfg.follow is not None and not cfg.moves


# --- The refl stash has to be able to serve the cadence (issue #111) -----
#
# refl_10cm is stashed by the microphysics on the HISTORY cadence, so a
# tracker consulted off that cadence asks for a plane that does not
# exist -- and finds out at the first evaluation where UH is under
# threshold and the echo fallback is consulted, which is hours into a
# run and is precisely when a storm-following nest should be working.
# The contract was already written in prose above cadence_seconds in
# configs/moving_nest_20110427_follow_2km.toml; nothing enforced it.


def _follow_build(cadence, history_interval_s=900.0, run_seconds=43200.0):
    return _build_relocation(
        _raw(cadence_seconds=cadence, follow=dict(_FOLLOW)), "test.toml",
        _domains(history_interval_s), run_seconds)


@pytest.mark.parametrize("cadence", [900.0, 1800.0, 2700.0, 3600.0])
def test_a_cadence_the_stash_can_serve_loads(cadence):
    """Whole multiples of the watched domain's history interval pass."""
    cfg = _follow_build(cadence)
    assert cfg.cadence_seconds == cadence
    assert cfg.follow is not None


@pytest.mark.parametrize("cadence", [300.0, 600.0, 1200.0, 2400.0])
def test_a_cadence_the_stash_cannot_serve_refuses_at_load(cadence):
    """And it refuses AT LOAD, naming both knobs and the fix."""
    with pytest.raises(ValueError) as caught:
        _follow_build(cadence)
    message = str(caught.value)
    assert "cadence_seconds" in message           # knob one
    assert "history_interval_s" in message        # knob two
    assert "whole multiple" in message            # the fix
    assert "refl_10cm" in message                 # why


def test_a_follow_block_with_no_cadence_refuses():
    """Absent means EVERY cycle boundary, which the stash cannot serve."""
    with pytest.raises(ValueError, match="no cadence_seconds"):
        _follow_build(None)


def test_the_refusal_names_the_watched_parent_not_the_moving_child():
    """grid_id names the child that MOVES; the tracker reads its parent.

    Naming the wrong domain would send a reader to edit a
    history_interval_s that has nothing to do with the failure.
    """
    with pytest.raises(ValueError) as caught:
        _follow_build(600.0)
    message = str(caught.value)
    assert "grid_id = 1" in message and "parent of the relocating" in message


@pytest.mark.parametrize("domains", [
    # A child whose parent_id names no domain, and a domain list with no
    # history_interval_s at all.  Both are OTHER validators' refusals;
    # this check must not preempt them with a cadence message that sends
    # the reader to the wrong knob.
    [SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                     time_step_fract_num=0, time_step_fract_den=1,
                     history_interval_s=900.0),
     SimpleNamespace(grid_id=2, parent_id=7, time_step=None,
                     history_interval_s=900.0)],
    [SimpleNamespace(grid_id=1, parent_id=None, time_step=60,
                     time_step_fract_num=0, time_step_fract_den=1),
     SimpleNamespace(grid_id=2, parent_id=1, time_step=None)],
])
def test_an_unresolvable_tree_is_left_to_its_own_validator(domains):
    cfg = _build_relocation(
        _raw(cadence_seconds=600.0, follow=dict(_FOLLOW)), "test.toml",
        domains, 43200.0)
    assert cfg.follow is not None


def test_an_itinerary_without_a_tracker_is_not_bound_by_the_stash():
    """[[relocation.move]] reads no fields, so no plane has to exist."""
    cfg = _build(_raw(cadence_seconds=600.0, move=[_MOVE]))
    assert cfg.cadence_seconds == 600.0 and cfg.follow is None


def test_the_shipped_follow_config_satisfies_its_own_contract():
    """The config whose comment states this rule must obey it.

    It is the only committed [relocation.follow] worked example, so a
    contract it fails is a contract that ships broken.
    """
    import tomllib

    root = Path(__file__).parents[1]
    for name in ("moving_nest_20110427_follow_2km.toml",
                 "moving_nest_20110427_spawn_2km.toml"):
        path = root / "configs" / name
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        if "follow" not in raw.get("relocation", {}):
            continue
        cadence = raw["relocation"]["cadence_seconds"]
        child = next(d for d in raw["domain"]
                     if d["grid_id"] == raw["relocation"]["grid_id"])
        parent = next(d for d in raw["domain"]
                      if d["grid_id"] == child["parent_id"])
        multiples = cadence / parent["history_interval_s"]
        assert multiples == int(multiples) and multiples >= 1, name


# ---------------------------------------------------------------------------
# The manual provider speaks the tracker's exact contract
# ---------------------------------------------------------------------------

def test_manual_provider_fires_each_row_exactly_once():
    provider = ManualMoveProvider((
        ScheduledRelocationMove(60.0, 1, 0),
        ScheduledRelocationMove(120.0, 0, -1)))
    # (parent_state, nest_footprint, t) -- the storm tracker's signature.
    assert provider(None, None, 30.0) is None
    assert provider(None, None, 60.0) == (1, 0)
    assert provider(None, None, 60.0) is None
    assert provider(None, None, 120.0) == (0, -1)
    assert provider(None, None, 180.0) is None
    assert provider.unconsumed() == ()


def test_manual_provider_reports_unfired_rows():
    provider = ManualMoveProvider((ScheduledRelocationMove(60.0, 1, 0),))
    assert provider(None, None, 120.0) is None
    assert len(provider.unconsumed()) == 1


# ---------------------------------------------------------------------------
# The runner on the CPU fake tree
# ---------------------------------------------------------------------------

def _clock(ticks):
    return SimpleNamespace(ticks=int(ticks), tick_den=1,
                           elapsed_seconds=float(ticks))


def _model(parent, child):
    nodes = {int(parent.cfg.grid_id): parent, int(child.cfg.grid_id): child}
    model = SimpleNamespace(
        root=parent,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)),
        experiment_fingerprint="f" * 64)
    model.node = lambda gid: nodes[gid]
    return model


def _runner(parent_plane, config, **kwargs):
    kwargs.setdefault("staging", "host")
    kwargs.setdefault("initializer", _initializer(parent_plane))
    kwargs.setdefault(
        "static_provenance",
        "footprint-parametric synthetic statics (test)")
    kwargs.setdefault("on_child_built", lambda *args: None)
    return RelocationRunner(
        config=config,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)),
        **kwargs)


def _advance(model, parent, child, ticks):
    parent.clock = _clock(ticks)
    child.clock = _clock(ticks)


def _manual_config(moves, cadence=None):
    return RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=4,
        min_overlap_fraction=0.25, cadence_seconds=cadence,
        moves=tuple(moves))


def test_scheduled_moves_execute_at_their_boundaries(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    start_i = int(child.cfg.i_parent_start)
    start_j = int(child.cfg.j_parent_start)
    model = _model(parent, child)
    receipts_path = tmp_path / "relocation.json"
    runner = _runner(parent_plane, _manual_config((
        ScheduledRelocationMove(60.0, 2, 0),
        ScheduledRelocationMove(120.0, 0, 1))),
        receipts_path=receipts_path)
    base_fingerprint = model.experiment_fingerprint

    outcomes = []
    for ticks in (0, 60, 120, 180):
        _advance(model, parent, child, ticks)
        outcomes.append(runner.on_period_begin(
            model, {1: parent.clock}, period=ticks // 60))
    summary = runner.close_receipt(model)

    assert outcomes[0] is None                      # t=0 is placement
    assert outcomes[1]["event"] == "relocated"
    assert outcomes[2]["event"] == "relocated"
    assert outcomes[3]["event"] == "held"           # itinerary exhausted
    assert child.cfg.i_parent_start == start_i + 2
    assert child.cfg.j_parent_start == start_j + 1
    assert runner.moves_executed == 2
    assert summary["moves_executed"] == 2
    assert summary["unconsumed_moves"] == []

    # The move receipt's required content: time, offsets, overlap
    # fraction, fields transplanted, fill counts, provenance, staging.
    move = outcomes[1]
    assert move["elapsed_seconds"] == 60.0
    assert move["requested_shift_parent_cells"] == [2, 0]
    assert move["executed_shift_parent_cells"] == [2, 0]
    assert move["clamped_by"] == []
    assert move["placement_from"]["i_parent_start"] == start_i
    assert move["placement_to"]["i_parent_start"] == start_i + 2
    assert 0.0 < move["overlap_fraction"] < 1.0
    fill = move["fill"]
    assert fill["fields_transplanted"] > 0
    assert fill["overlap_cells"] + fill["spin_up_cells"] == (
        int(child.cfg.run.nx) * int(child.cfg.run.ny))
    assert "parent" in fill["strip_fill_source"]
    assert "footprint-parametric" in fill["static_fields"]
    assert move["staging"]["mode"] == "host"
    assert move["donor_alignment_pass"] is True
    assert move["generation"] == 1
    assert outcomes[2]["generation"] == 2

    # Fingerprint marking: chained, deterministic, and reconstructible
    # from the receipts alone.
    assert model.experiment_fingerprint != base_fingerprint
    expected = base_fingerprint
    for outcome in outcomes[1:3]:
        assert outcome["experiment_fingerprint"]["before"] == expected
        expected = mark_fingerprint_across_move(
            expected, outcome["record_sha256"])
        assert outcome["experiment_fingerprint"]["after"] == expected
    assert model.experiment_fingerprint == expected

    # The run receipts: on the model, and durable at receipts_path.
    assert model._relocation_receipts is runner.receipts
    document = json.loads(receipts_path.read_text(encoding="utf-8"))
    assert document["config"]["enabled"] is True
    events = [row["event"] for row in document["receipts"]]
    assert events == ["relocated", "relocated", "held", "summary"]


def test_the_run_end_summary_appears_exactly_once(tmp_path):
    """One run, one summary row -- however many times the executor closes.

    ``execute_experiment`` closes the runner every time it returns, and on
    a leg-walking route (``gpuwm.runtime.walk_spawn_legs``) it returns once
    per leg.  A live 12-move run shipped ``relocation_receipts.json`` with
    two byte-identical summary rows in a 25-row list, so a consumer
    counting summaries double-counted.  The surviving row must also be the
    LAST state, not the first: freezing it would report a stale
    ``moves_executed``.
    """

    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    receipts_path = tmp_path / "relocation.json"
    runner = _runner(parent_plane, _manual_config((
        ScheduledRelocationMove(60.0, 1, 0),
        ScheduledRelocationMove(120.0, 1, 0))),
        receipts_path=receipts_path)

    # Leg one: one move, then the executor returns and closes.
    for ticks in (0, 60):
        _advance(model, parent, child, ticks)
        runner.on_period_begin(model, {1: parent.clock})
    first = runner.close_receipt(model)
    assert first["moves_executed"] == 1

    # Leg two: the second move, and a second close.
    _advance(model, parent, child, 120)
    runner.on_period_begin(model, {1: parent.clock})
    second = runner.close_receipt(model)

    summaries = [row for row in runner.receipts if row["event"] == "summary"]
    assert len(summaries) == 1
    assert summaries[0] is second
    assert summaries[0]["moves_executed"] == 2
    assert summaries[0]["unconsumed_moves"] == []

    document = json.loads(receipts_path.read_text(encoding="utf-8"))
    events = [row["event"] for row in document["receipts"]]
    assert events.count("summary") == 1
    # And it stays the run's final row rather than being stranded mid-list
    # by the moves that came after the first close.
    assert events == ["relocated", "relocated", "summary"]


def test_cadence_gates_the_opportunities():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(120.0, 1, 0),), cadence=120.0))
    assert runner.cadence_periods == 2
    consulted = []
    original = runner.provider

    def spy(parent_state, footprint, t):
        consulted.append(t)
        return original(parent_state, footprint, t)

    runner.provider = spy
    for ticks in (0, 60, 120, 180, 240):
        _advance(model, parent, child, ticks)
        runner.on_period_begin(model, {1: parent.clock})
    # Only the cadence boundaries were consulted -- never t=0, never the
    # off-cadence periods.
    assert consulted == [120.0, 240.0]
    assert runner.moves_executed == 1


def test_requested_shift_clamps_to_the_bounds_and_says_so():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 9, 0),)))  # over max_move=4
    _advance(model, parent, child, 60)
    outcome = runner.on_period_begin(model, {1: parent.clock})
    assert outcome["event"] == "relocated"
    assert outcome["requested_shift_parent_cells"] == [9, 0]
    assert outcome["executed_shift_parent_cells"] == [4, 0]
    assert "max_move_parent_cells" in outcome["clamped_by"]


def test_unstarted_child_holds_with_a_reason():
    parent_plane, parent, child = _cpu_tree()
    child._started = False
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    _advance(model, parent, child, 60)
    outcome = runner.on_period_begin(model, {1: parent.clock})
    assert outcome["event"] == "held"
    assert "not started" in outcome["reason"]


def test_executed_moves_notify_a_provider_that_asks():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 9, 0),)))
    notified = []
    runner.provider.notify_move_executed = (
        lambda t, shift: notified.append((t, shift)))
    _advance(model, parent, child, 60)
    runner.on_period_begin(model, {1: parent.clock})
    # Accepted-move semantics: the notification carries the EXECUTED
    # (clamped) shift, not the request.
    assert notified == [(60.0, (4, 0))]


def test_before_rebuild_receives_the_grid_id_before_the_release():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    _advance(model, parent, child, 60)
    dropped = []
    runner.on_period_begin(model, {1: parent.clock},
                           before_rebuild=lambda gid: dropped.append(gid))
    assert dropped == [2]


def test_runner_construction_refusals():
    schedule = SimpleNamespace(period_ticks=60,
                               clock=SimpleNamespace(tick_den=1))
    with pytest.raises(RelocationRefusal, match="enabled"):
        RelocationRunner(config=RelocationConfig(), schedule=schedule,
                         on_child_built=lambda *a: None)
    bounds_only = RelocationConfig(enabled=True, grid_id=2)
    with pytest.raises(RelocationRefusal, match="no follow source"):
        RelocationRunner(config=bounds_only, schedule=schedule,
                         on_child_built=lambda *a: None)
    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    with pytest.raises(RelocationRefusal, match="on_child_built"):
        RelocationRunner(config=manual, schedule=schedule,
                         on_child_built=None)


def test_executor_refuses_a_follow_source_with_no_runner():
    from gpuwm.core.model import ModelRuntimeStatus, execute_experiment

    exp = SimpleNamespace(relocation=_manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    model = SimpleNamespace(
        _runtime_status=ModelRuntimeStatus(),
        _activation_context={"experiment": exp})
    with pytest.raises(RuntimeError, match="relocation.move"):
        execute_experiment(model, validate_state=False)


def test_run_experiment_front_door_lift_and_residual_refusals(
        tmp_path, monkeypatch):
    """Leg-3 governance: the case-data route no longer refuses a follow
    source by name -- it wires the real-data runner instead -- while a
    follow source with no child to move still refuses at the door."""
    from gpuwm import runtime

    from gpuwm.core.streaming import OFF as _STREAMING_OFF

    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    # ``streaming`` is not decoration on this double: since preflight-ledger
    # and feat-route-wire, run_experiment refuses a [tiles] block at
    # ADMISSION -- before the ingest -- because none of its arms consult one.
    # A real Experiment always carries the attribute (build_experiment
    # defaults it to OFF); a double without it raises AttributeError from
    # inside the front door and never reaches the refusal under test.
    single = SimpleNamespace(relocation=manual, domains=(object(),),
                             feedback=0, tiles=_STREAMING_OFF)
    with pytest.raises(ValueError, match="no nest to move"):
        runtime.run_experiment(single, None, tmp_path / "out")

    # Multi-domain: the old front-door ValueError is GONE.  The config
    # passes the gate and reaches ordinary preparation (witnessed by a
    # sentinel planted on the first preparation step).
    class _ReachedPreparation(Exception):
        pass

    import gpuwm.io.wrfout as wrfout_module

    def _sentinel(_outdir):
        raise _ReachedPreparation

    monkeypatch.setattr(
        wrfout_module, "quarantine_orphan_wrfouts", _sentinel)
    tree = SimpleNamespace(relocation=manual,
                           domains=(object(), object()), feedback=0,
                           tiles=_STREAMING_OFF)
    with pytest.raises(_ReachedPreparation):
        runtime.run_experiment(tree, None, tmp_path / "out2")


def test_prepared_tree_route_still_refuses_a_corridor_less_follow_source():
    """A prepared tree runs without the case's GEOG source, so the
    refusal survives for CORRIDOR-LESS bundles -- and now names the
    remedy: re-prepare with --statics-corridor (the sealed
    child-resolution statics the runner crops per move)."""
    import inspect

    from gpuwm import prepared_domain_tree_forecast as route

    source = inspect.getsource(route)
    assert "cannot rebuild a relocated child's statics" in source
    assert "--statics-corridor" in source
    assert "gpuwm run" in source


def test_run_route_wires_initializer_preparer_and_provenance(tmp_path):
    """build_real_relocation_runner assembles the leg-3 machinery: the
    footprint-rebuilt-statics initializer (with its alignment frame,
    strip-fill provenance and post-transplant hook) and the land-surface
    preparer, bound into a RelocationRunner."""
    from gpuwm.runtime import (RealRelocationChildPreparer,
                               build_real_relocation_runner)
    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.ingest.relocation_init import (
        REAL_DATA_FOOTPRINT_REBUILT_STATICS, REAL_DATA_STRIP_FILL_SOURCE)
    from gpuwm.static.lambert import LambertGrid
    from test_nest_relocation_staging import _scaffold

    exp = _scaffold()
    child_dc = exp.domains[1]
    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    exp = SimpleNamespace(
        relocation=manual, domains=exp.domains, vertical=object(),
        start_time=None)
    grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    node = SimpleNamespace(cfg=child_dc, grid=grid)
    model = SimpleNamespace(
        node=lambda gid: node, _input_catalog=object(),
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)))
    runner = build_real_relocation_runner(exp, None, model, tmp_path)
    assert isinstance(runner, RelocationRunner)
    assert isinstance(runner.on_child_built, RealRelocationChildPreparer)
    assert runner.static_provenance == REAL_DATA_FOOTPRINT_REBUILT_STATICS
    initializer = runner.initializer
    assert initializer.strip_fill_source == REAL_DATA_STRIP_FILL_SOURCE
    assert initializer.donor_alignment_frame_width == (
        int(child_dc.run.spec_bdy_width)
        + int(getattr(child_dc, "blend_width", 5)))
    assert callable(initializer.post_transplant)
    # Bounds-only [relocation] stays the manual/API mechanism: no runner.
    bounds_only = SimpleNamespace(relocation=RelocationConfig(
        enabled=True, grid_id=2), domains=exp.domains)
    assert build_real_relocation_runner(
        bounds_only, None, model, tmp_path) is None


# ---------------------------------------------------------------------------
# Restart across a move: marked, and loud
# ---------------------------------------------------------------------------

def test_fingerprint_marking_is_deterministic_and_order_sensitive():
    marked_a = mark_fingerprint_across_move("base", "record1")
    assert marked_a == mark_fingerprint_across_move("base", "record1")
    assert marked_a != "base"
    forward = mark_fingerprint_across_move(marked_a, "record2")
    backward = mark_fingerprint_across_move(
        mark_fingerprint_across_move("base", "record2"), "record1")
    assert forward != backward


def test_checkpoint_relocation_block_refuses_with_the_posture():
    from gpuwm.io.restart import tree_fingerprint_mismatch_reason

    header = {
        "experiment_fingerprint": "marked",
        "relocation": {"moves": 2, "segment_id": "seg123",
                       "posture": RESTART_ACROSS_MOVE_POSTURE},
    }
    reason = tree_fingerprint_mismatch_reason(
        2, header, SimpleNamespace())
    assert "2 nest relocation(s)" in reason
    assert "seg123" in reason
    assert "promises nothing" in reason
    assert "TOLERATED_EXPERIMENT" in reason


def test_moved_receipts_supply_the_checkpoint_header_block():
    """write_tree_restart reads model._relocation_receipts; the rows the
    runner produces must carry exactly the fields it stamps."""
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _manual_config(
        (ScheduledRelocationMove(60.0, 1, 0),)))
    _advance(model, parent, child, 60)
    runner.on_period_begin(model, {1: parent.clock})
    moved = [row for row in model._relocation_receipts
             if row.get("event") == "relocated"]
    assert len(moved) == 1
    assert set(moved[0]) >= {"record_sha256", "segment_id", "grid_id"}


def test_prepared_tree_route_wires_the_corridor_runner(tmp_path):
    """The corridor lift: build_prepared_tree_relocation_runner assembles
    the SAME RelocationRunner shape the case-data route does -- the
    shared initializer with the corridor-crop statics seam, the
    prepared-route preparer, corridor provenance -- and keeps the
    bounds-only and corridor-less postures."""

    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.runtime import (PreparedTreeRelocationChildPreparer,
                               build_prepared_tree_relocation_runner)
    from gpuwm.static.corridor import (CORRIDOR_REBUILT_STATICS,
                                       CORRIDOR_STRIP_FILL_SOURCE,
                                       ChildStaticsCorridor)
    from gpuwm.static.lambert import LambertGrid
    from test_nest_relocation_staging import _scaffold

    scaffold = _scaffold()
    child_dc = scaffold.domains[1]
    manual = _manual_config((ScheduledRelocationMove(60.0, 1, 0),))
    exp = SimpleNamespace(
        relocation=manual, domains=scaffold.domains, vertical=object(),
        start_time=None)
    grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    node = SimpleNamespace(cfg=child_dc, grid=grid)
    model = SimpleNamespace(
        node=lambda gid: node,
        schedule=SimpleNamespace(
            period_ticks=60, clock=SimpleNamespace(tick_den=1)))
    corridor = ChildStaticsCorridor(
        geometry={"grid_id": 2, "parent_grid_ratio": 3, "child_nx": 12,
                  "child_ny": 12, "corridor_nx": 36, "corridor_ny": 36},
        fields={}, cache_sha256="f" * 64)
    workspace = object()

    runner = build_prepared_tree_relocation_runner(
        exp, statics_corridor=corridor, model=model, outdir=tmp_path,
        radiation_workspace=workspace)
    assert isinstance(runner, RelocationRunner)
    assert isinstance(runner.on_child_built,
                      PreparedTreeRelocationChildPreparer)
    assert runner.on_child_built.data is None
    assert runner.on_child_built._radiation_workspace is workspace
    assert runner.static_provenance == CORRIDOR_REBUILT_STATICS
    initializer = runner.initializer
    assert initializer.static_provenance == CORRIDOR_REBUILT_STATICS
    assert initializer.strip_fill_source == CORRIDOR_STRIP_FILL_SOURCE
    assert callable(initializer.post_transplant)
    assert initializer.donor_alignment_frame_width == (
        int(child_dc.run.spec_bdy_width)
        + int(getattr(child_dc, "blend_width", 5)))

    # Bounds-only [relocation] stays runnerless, exactly as elsewhere.
    bounds_only = SimpleNamespace(relocation=RelocationConfig(
        enabled=True, grid_id=2), domains=scaffold.domains)
    assert build_prepared_tree_relocation_runner(
        bounds_only, statics_corridor=corridor, model=model,
        outdir=tmp_path) is None

    # A follow source with no corridor may not sneak past the preflight
    # refusal through this constructor.
    with pytest.raises(ValueError, match="verified statics corridor"):
        build_prepared_tree_relocation_runner(
            exp, statics_corridor=None, model=model, outdir=tmp_path)
