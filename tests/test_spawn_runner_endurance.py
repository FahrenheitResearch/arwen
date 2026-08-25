"""What the spawn runner costs over a 384-hour forecast.

Both defects here are invisible on a two-hour test and fatal on a long
one, which is why they need tests that COUNT rather than tests that
assert an outcome:

* the receipts file was re-serialised WHOLE on every leg boundary, so
  the bytes a run writes to it grow as the square of its length -- and
  the in-memory ledger it was serialising from grew without bound;
* the leg walk asked for a boundary every history interval for the whole
  run whenever any retired slot could still re-arm, so a 384 h run with
  a re-armable slot rebuilt its entire schedule ~4,600 times to discover
  ~4,600 times that a cooldown had not elapsed.

The instrument rule applies: each test measures the cost directly (bytes
written, entries retained, boundaries requested) rather than asserting
that a run "feels" cheaper, and each has a control that shows the meter
moves when the work is genuinely there.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.nest_lifecycle import RearmConfig, RetireConfig
from gpuwm.core.nest_spawn import SpawnConfig
from gpuwm.core.spawn_runner import (SPAWN_RUNNER_CONTRACT, HELD_RECEIPTS_KEPT,
                                     SpawnRunner)
from test_nest_spawn_init import _experiment, _live_parent
from test_spawn_runner import LEG, _leg, _walk


@pytest.fixture()
def case():
    exp = _experiment()
    parent, grids = _live_parent(exp)
    dormant = replace(exp.domains[1],
                      spawn=SpawnConfig(trigger="time", at_s=LEG))
    return {"exp": replace(exp, domains=(exp.domains[0], dormant)),
            "parent": parent, "grids": grids}


def _held_runner(case, monkeypatch, tmp_path, name="spawn_receipts.jsonl"):
    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *_a, **_k: None)
    return SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np,
        receipts_path=tmp_path / name)


def _held_model(runner, case):
    """One model, walked once, consulted at many instants.

    The runner reads three surfaces off a model and none of them change
    while every watch holds, so rebuilding it per boundary would only
    make this file slow.
    """
    model, _ = _walk(_leg(runner.active, 60.0), case["parent"].state,
                     grids=(case["grids"][0],))
    return model


# ---------------------------------------------------------------------------
# The receipts file: appended, never rewritten
# ---------------------------------------------------------------------------

def test_every_boundary_appends_and_no_boundary_rewrites(case, monkeypatch,
                                                         tmp_path):
    """THE STRUCTURAL PROOF, not a timing: after each boundary the file's
    earlier bytes are unchanged and only new bytes have arrived.  A
    rewrite-and-replace cannot satisfy that -- it re-serialises the whole
    ledger, so the prefix moves the moment any earlier record's rendering
    would differ, and the byte count grows as the square of the run."""
    runner = _held_runner(case, monkeypatch, tmp_path)
    model = _held_model(runner, case)
    path = runner.receipts_path
    sizes, previous = [], b""
    for step in range(1, 6):
        runner.on_leg_boundary(model, t=10.0 * step)
        blob = path.read_bytes()
        assert blob.startswith(previous), (
            "an earlier boundary's bytes were rewritten")
        sizes.append(len(blob) - len(previous))
        previous = blob
    # Every boundary costs about the same: the per-record size, not the
    # ledger-so-far size.  A whole-file rewrite makes these grow 1, 2,
    # 3, 4, 5.
    assert max(sizes) < 2 * min(sizes)


def test_the_receipts_file_is_one_complete_json_object_per_line(
        case, monkeypatch, tmp_path):
    """The CONTENT contract is unchanged: every decision is recorded, and
    every line stands alone -- so a killed process loses nothing earlier
    and a reader never has to parse a truncated array."""
    runner = _held_runner(case, monkeypatch, tmp_path)
    runner.on_leg_boundary(_held_model(runner, case), t=60.0)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    runner.on_leg_boundary(model)
    runner.close_receipt()
    lines = runner.receipts_path.read_text(
        encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["event"] for row in rows] == ["held", "spawned", "closed"]
    assert {row["contract"] for row in rows} == {SPAWN_RUNNER_CONTRACT}
    # The watch's own per-evaluation receipts still ride along.
    assert rows[0]["watch_receipts"]


def test_closing_twice_appends_rather_than_truncating_the_ledger(
        case, monkeypatch, tmp_path):
    """close_receipt closes the handle, so a record after it reopens the
    file -- and a reopen in write mode would erase the whole run."""
    runner = _held_runner(case, monkeypatch, tmp_path)
    runner.on_leg_boundary(_held_model(runner, case), t=60.0)
    runner.close_receipt()
    runner.close_receipt()
    events = [json.loads(line)["event"] for line
              in runner.receipts_path.read_text(
                  encoding="utf-8").splitlines()]
    assert events == ["held", "closed", "closed"]


def test_a_run_with_no_receipts_path_still_records_in_memory(
        case, monkeypatch):
    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *_a, **_k: None)
    runner = SpawnRunner.from_experiment(
        case["exp"], on_child_built=lambda *_a: None, array_module=np)
    runner.on_leg_boundary(_held_model(runner, case), t=60.0)
    assert runner.receipts[-1]["event"] == "held"


# ---------------------------------------------------------------------------
# The in-memory ledger: bounded in run length, complete in decisions
# ---------------------------------------------------------------------------

def test_the_in_memory_ledger_is_bounded_by_the_run_not_its_length(
        case, monkeypatch, tmp_path):
    """A held boundary says nothing changed.  There is one per history
    interval for the whole run -- ~4,600 of them at 384 h -- and the file
    keeps every one.  Memory keeps a window of them, so the ledger a
    caller embeds in its own receipt does not grow with the forecast."""
    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *_a, **_k: None)
    # A watch that never fires inside the loop, so every boundary is a
    # HELD one and the window is the only thing bounding the ledger.
    exp = _experiment()
    late = replace(exp.domains[1],
                   spawn=SpawnConfig(trigger="time", at_s=1.0e6))
    runner = SpawnRunner.from_experiment(
        replace(exp, domains=(exp.domains[0], late)),
        on_child_built=lambda *_a: None, array_module=np,
        receipts_path=tmp_path / "spawn_receipts.jsonl")
    model = _held_model(runner, case)
    for step in range(1, HELD_RECEIPTS_KEPT + 21):
        runner.on_leg_boundary(model, t=10.0 * step)
    assert len(runner.receipts) == HELD_RECEIPTS_KEPT
    # The window is the RECENT end, so `receipts[-1]` is still the last
    # boundary -- which is what every caller reads it for.
    assert runner.receipts[-1]["elapsed_seconds"] == pytest.approx(
        10.0 * (HELD_RECEIPTS_KEPT + 20))
    # ...and the file kept all of them.
    kept = runner.receipts_path.read_text(encoding="utf-8").splitlines()
    assert len(kept) == HELD_RECEIPTS_KEPT + 20


def test_a_decision_is_never_evicted_from_the_in_memory_ledger(
        case, monkeypatch, tmp_path):
    """Held boundaries are the unbounded stream; decisions are bounded by
    the config (slots x max_firings), so they are all retained.  A caller
    that scans the ledger for what the run DID must not have to read the
    file to find out."""
    runner = _held_runner(case, monkeypatch, tmp_path)
    model, _ = _walk(_leg(runner.active), case["parent"].state,
                     grids=(case["grids"][0],))
    runner.on_leg_boundary(model)
    for step in range(1, HELD_RECEIPTS_KEPT + 21):
        runner.on_leg_boundary(model, t=LEG + 10.0 * step)
    events = [row["event"] for row in runner.receipts]
    assert events.count("spawned") == 1
    assert len(runner.receipts) == HELD_RECEIPTS_KEPT + 1


# ---------------------------------------------------------------------------
# The leg walk: a boundary per DECISION POINT, not per history interval
# ---------------------------------------------------------------------------

def _rearm_case():
    """A slot that has fired, retired, and is waiting out its cooldown."""
    exp = _experiment()
    dormant = replace(
        exp.domains[1],
        spawn=SpawnConfig(trigger="time", at_s=LEG),
        retire=RetireConfig(trigger="time", at_s=LEG, min_lifetime_s=0.0,
                            sustained_s=0.0),
        rearm=RearmConfig(max_firings=3, cooldown_s=3600.0))
    return replace(exp, domains=(exp.domains[0], dormant))


def _runner_in_cooldown(retired_at=600.0):
    exp = _rearm_case()
    runner = SpawnRunner.from_experiment(
        exp, on_child_built=lambda *_a: None, array_module=np)
    runner.retired = {2}
    runner.retired_times = {2: retired_at}
    runner.episodes = {2: 1}
    runner.controller.watches[2].fired = True
    return runner


def test_a_slot_in_cooldown_asks_for_no_boundary_until_it_expires():
    """needs_boundaries stayed true for the whole run whenever any slot
    could still re-arm, and walk_spawn_legs rebuilds the entire schedule
    at every boundary it is given.  A cooldown is a KNOWN instant: there
    is nothing to decide before it."""
    runner = _runner_in_cooldown(retired_at=600.0)
    assert runner.needs_boundaries
    # Nothing can happen between the retirement and the cooldown's end.
    assert runner.next_decision_time(600.0) == pytest.approx(4200.0)
    assert runner.next_decision_time(4200.0) == pytest.approx(4200.0)


def _armed(**over):
    base = dict(trigger="pressure", threshold=25.0, level_hpa=850.0,
                earliest_s=7200.0, latest_s=36000.0)
    base.update(over)
    exp = _experiment()
    dormant = replace(exp.domains[1], spawn=SpawnConfig(**base))
    return SpawnRunner.from_experiment(
        replace(exp, domains=(exp.domains[0], dormant)),
        on_child_built=lambda *_a: None, array_module=np)


def test_an_armed_watch_asks_for_no_boundary_before_its_window_opens():
    runner = _armed()
    assert runner.needs_boundaries
    assert runner.next_decision_time(0.0) == pytest.approx(7200.0)
    # Inside the window every boundary is a real decision point again.
    assert runner.next_decision_time(7200.0) is None
    assert runner.next_decision_time(9000.0) is None


def test_a_stash_backed_watch_keeps_every_boundary_so_its_window_resets():
    """THE ONE PLACE SKIPPING WOULD CHANGE AN ANSWER.  A uh or
    reflectivity window is zeroed by the runner at every boundary it
    takes, and what a watch reads is "the strongest since I last
    looked".  Skip six hours of them and the next look sees six hours of
    accumulation -- a slot coming off its cooldown would fire on
    rotation that happened while it was spent.  Pressure carries no
    window, which is why it is exempt."""
    stash = _armed(trigger="uh", threshold=60.0, level_hpa=None)
    assert stash.next_decision_time(0.0) is None
    echo = _armed(trigger="reflectivity", threshold=45.0, level_hpa=None)
    assert echo.next_decision_time(0.0) is None
    # ...and the same slot on pressure, which reads no stash, skips.
    assert _armed().next_decision_time(0.0) == pytest.approx(7200.0)


def test_a_stash_backed_retire_forfeits_it_too():
    """The retirement watch reads the SAME window, so a live episode
    decaying on uh keeps the boundaries even when its own decision
    instant would otherwise be knowable."""
    runner = _live_field_retire(min_lifetime_s=7200.0)
    assert runner.next_decision_time(0.0) is None


def _live_field_retire(min_lifetime_s=0.0, trigger="uh", threshold=60.0,
                       level_hpa=None):
    """A live episode whose retirement reads the LIVE FIELD."""
    exp = _experiment()
    dormant = replace(
        exp.domains[1],
        spawn=SpawnConfig(trigger="time", at_s=LEG),
        retire=RetireConfig(trigger=trigger, threshold=threshold,
                            level_hpa=level_hpa,
                            min_lifetime_s=min_lifetime_s,
                            sustained_s=0.0),
        rearm=RearmConfig(max_firings=3, cooldown_s=3600.0))
    runner = SpawnRunner.from_experiment(
        replace(exp, domains=(exp.domains[0], dormant)),
        on_child_built=lambda *_a: None, array_module=np)
    runner.spawned = {2: (1, 1)}
    runner.birth_times = {2: 0.0}
    runner.episodes = {2: 1}
    runner.controller.watches[2].fired = True
    return runner


def test_a_live_slot_that_can_retire_keeps_every_boundary():
    """The control.  A field retirement reads the plane, so its instant
    belongs to the weather and the walk must keep asking."""
    runner = _live_field_retire()
    assert runner.needs_boundaries
    assert runner.next_decision_time(600.0) is None


def test_a_minimum_lifetime_is_a_known_instant_even_for_a_field_retire():
    """Before min_lifetime_s every evaluation is a guaranteed hold, so
    those boundaries buy nothing; after it the field decides."""
    runner = _live_field_retire(min_lifetime_s=7200.0, trigger="pressure",
                                threshold=10.0, level_hpa=850.0)
    assert runner.next_decision_time(0.0) == pytest.approx(7200.0)
    assert runner.next_decision_time(7200.0) is None


def test_a_live_slot_on_a_time_retire_knows_its_own_instant():
    runner = SpawnRunner.from_experiment(
        _rearm_case(), on_child_built=lambda *_a: None, array_module=np)
    runner.spawned = {2: (1, 1)}
    runner.birth_times = {2: 600.0}
    runner.episodes = {2: 1}
    runner.controller.watches[2].fired = True
    # retire at_s is EPISODE AGE, measured from born_t.
    assert runner.next_decision_time(600.0) == pytest.approx(600.0 + LEG)


def test_the_leg_walk_stops_at_the_decision_point_not_every_interval():
    """The count the defect is measured in.  Over a 100-leg span with a
    slot in cooldown for 60 of them, the walk must ask ~1 time, not 60."""
    from gpuwm.runtime import spawn_leg_boundary

    runner = _runner_in_cooldown(retired_at=0.0)
    leg, total = 300.0, 30000.0
    asked, elapsed = 0, 0.0
    while elapsed + 1e-9 < total:
        boundary = spawn_leg_boundary(runner, elapsed, leg=leg, total=total)
        assert boundary > elapsed
        asked += 1
        elapsed = boundary
    # 30,000 s of run at a 300 s leg is 100 intervals.  The slot's
    # cooldown ends at 3,600 s; before it there is exactly one decision
    # point, and after it every interval is one again.
    assert asked == 1 + int((total - 3600.0) / leg)


def test_a_run_with_nothing_pending_still_runs_to_the_end_in_one_leg():
    """The pre-existing shape, unchanged: once nothing can happen the
    walk takes the whole rest of the forecast as one leg."""
    from gpuwm.runtime import spawn_leg_boundary

    exp = _experiment()
    dormant = replace(exp.domains[1],
                      spawn=SpawnConfig(trigger="time", at_s=LEG))
    runner = SpawnRunner.from_experiment(
        replace(exp, domains=(exp.domains[0], dormant)),
        on_child_built=lambda *_a: None, array_module=np)
    runner.controller.watches[2].fired = True
    runner.spawned = {2: (1, 1)}
    assert not runner.needs_boundaries
    assert spawn_leg_boundary(runner, 0.0, leg=300.0,
                              total=30000.0) == pytest.approx(30000.0)


def test_a_pending_time_trigger_still_stops_on_its_own_instant():
    """A manual trigger's instant is knowable, so the walk goes straight
    to it -- and must not overshoot it."""
    from gpuwm.runtime import spawn_leg_boundary

    exp = _experiment()
    dormant = replace(exp.domains[1],
                      spawn=SpawnConfig(trigger="time", at_s=3000.0))
    runner = SpawnRunner.from_experiment(
        replace(exp, domains=(exp.domains[0], dormant)),
        on_child_built=lambda *_a: None, array_module=np)
    assert runner.next_decision_time(0.0) == pytest.approx(3000.0)
    assert spawn_leg_boundary(runner, 0.0, leg=300.0,
                              total=30000.0) == pytest.approx(3000.0)


def test_a_follower_pins_the_cadence_even_while_a_slot_sleeps():
    """A LIVE follower is relocated at leg boundaries, so a walk that
    skipped to a cooldown's end would stop moving the nest.  The
    relocation cadence is the floor."""
    from gpuwm.runtime import spawn_leg_boundary

    runner = _runner_in_cooldown(retired_at=0.0)
    runner.spawned = {1: (1, 1)}
    assert spawn_leg_boundary(runner, 0.0, leg=300.0, total=30000.0,
                              relocation_cadence_s=300.0) == pytest.approx(
                                  300.0)
