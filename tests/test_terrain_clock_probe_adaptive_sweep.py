"""The adaptive-row walk of tools/terrain_clock_probe.py (adaptive-sweep).

The walk itself, with the GPU run replaced by a rule: per wind, weakest
first, the longest max_time_step from the ladder that held at every CFL
target pair, each wind walked from the step the weaker wind held, and the
stronger winds not run past a wind that held none.  Runs already made are
reused, and no run starts after the deadline.
"""
from __future__ import annotations

import json

import pytest

import tools.terrain_clock_probe as probe

LADDER = (15.0, 12.0, 9.0, 6.0)


def _fake(holds, calls):
    """run_adaptive_cell stand-in: held where max_step (s/km) is at or
    under holds[wind] (None: nothing holds), at both target pairs alike
    unless the pair is (1.4, 0.98) and holds carries a second value."""

    def run(ridge, *, wind, max_step, start_step, seconds, target_cfl,
            target_hcfl, increase_pct, etac, criterion, settings):
        per_km = max_step / (ridge.dx / 1000.0)
        calls.append((wind, round(per_km, 6), target_cfl))
        limit = holds[wind]
        if isinstance(limit, tuple):
            limit = limit[0] if target_cfl == 1.2 else limit[1]
        held = limit is not None and per_km <= limit + 1e-9
        return {"held": held, "peak_w": 1.0, "bound": 2.0,
                "mean_step_s": max_step, "wall_s": 0.0,
                "stopped_at_s": None if held else 60.0}
    return run


def _row(monkeypatch, holds, **kw):
    calls = []
    monkeypatch.setattr(probe, "run_adaptive_cell", _fake(holds, calls))
    row = probe.adaptive_sweep_row(
        probe.Ridge(3000.0, 4500.0, 0.1), sorted(holds), LADDER,
        seconds=43200.0, criterion="blowup", log=lambda text: None, **kw)
    return row, calls


def test_each_wind_walks_from_the_step_the_weaker_wind_held(monkeypatch):
    row, calls = _row(monkeypatch, {40.0: 15.0, 50.0: 9.0, 60.0: 6.0})
    assert row["complete"]
    assert row["adaptive_s_per_km"] == [15.0, 9.0, 6.0]
    assert row["adaptive_top_s_per_km"] == [15.0, 15.0, 9.0]
    # 60 m/s starts at 9 s/km, where 50 m/s held: 15 and 12 are never run.
    assert [c for c in calls if c[0] == 60.0][0][1] == 9.0


def test_a_step_holds_only_where_every_target_pair_held(monkeypatch):
    row, _ = _row(monkeypatch, {40.0: (15.0, 12.0)})
    assert row["adaptive_s_per_km"] == [12.0]


def test_past_a_wind_that_held_none_the_stronger_winds_are_not_run(
        monkeypatch):
    row, calls = _row(monkeypatch, {40.0: 12.0, 50.0: None, 60.0: 15.0})
    assert row["adaptive_s_per_km"] == [12.0, None, None]
    assert row["adaptive_top_s_per_km"] == [15.0, 12.0, None]
    assert not [c for c in calls if c[0] == 60.0]


def test_runs_already_made_are_reused_and_the_row_is_the_same(monkeypatch):
    holds = {40.0: 12.0, 50.0: 9.0}
    first, calls = _row(monkeypatch, holds)
    done = {run["key"]: run for run in first["runs"]}
    again, calls_again = _row(monkeypatch, holds, done=done)
    assert calls and not calls_again
    assert again["adaptive_s_per_km"] == first["adaptive_s_per_km"]
    assert again["adaptive_top_s_per_km"] == first["adaptive_top_s_per_km"]


def test_no_run_starts_after_the_deadline(monkeypatch):
    row, calls = _row(monkeypatch, {40.0: 12.0}, deadline=0.0)
    assert not calls
    assert not row["complete"]
    assert row["adaptive_s_per_km"] == [None]


def test_the_worker_writes_a_row_only_when_its_walk_is_complete(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(probe, "run_adaptive_cell",
                        _fake({40.0: 9.0}, calls))
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps({"dx": 3000.0, "crest": 4500.0,
                                "ridge_slope": 0.1,
                                "settings": "generated"}) + "\n",
                    encoding="utf-8")
    out = tmp_path / "rows"
    assert probe.adaptive_sweep_worker(
        jobs, out, [40.0], LADDER, deadline=float("inf"), seconds=43200.0,
        criterion="blowup", provenance={"box": "test"}) == 1
    row = json.loads((out / "adaptive-generated-dx3000-c4500-r0.1-blowup"
                      ".json").read_text(encoding="utf-8"))
    assert row["adaptive_s_per_km"] == [9.0]
    assert row["provenance"]["box"] == "test"
    partial = out / "adaptive-generated-dx3000-c4500-r0.1-blowup.partial.jsonl"
    assert len(partial.read_text(encoding="utf-8").splitlines()) == len(calls)


def test_the_default_ladder_is_the_shipped_one_then_down_to_3_s_per_km():
    from gpuwm.terrain_clock import MAP_PATH

    shipped = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    assert list(probe.SHIPPED_ADAPTIVE_LADDER) == pytest.approx(
        shipped["adaptive"]["ladder_s_per_km"])
    assert probe.ADAPTIVE_LADDER_BELOW[-1] == 3.0


# ---------------------------------------------------------------------------
# The 100 m/s peak rule (lead decision 3 after step 5, 2026-10-07).
# ---------------------------------------------------------------------------


def test_blowup_stops_at_100_m_s_whatever_the_bound():
    """A run holds only where it stayed finite and its peak |w| stayed
    under 100 m/s; ten times the old bound is never under 200 m/s, so the
    100 m/s limit is the one that acts on every blow-up run."""
    assert probe.BLOWUP_PEAK_W == 100.0
    # LA's deciding cell at 70 m/s: bound 4 x 70 x 0.3571 + 20 = 120 m/s,
    # held at 119.3 m/s under ten times it; it stops now.
    bound = 4.0 * 70.0 * 0.3571 + 20.0
    assert 119.3 < probe.BLOWUP_FACTOR * bound
    for criterion in ("blowup", "steady"):
        assert probe.stops(100.0, bound, criterion)
        assert probe.stops(119.3, bound, criterion)
        assert not probe.stops(99.99, bound, criterion)
        assert probe.stops(float("nan"), bound, criterion)
        assert probe.stops(float("inf"), bound, criterion)
    # The map's own bound is untouched.
    assert probe.stops(bound + 0.1, bound, "map")
    assert not probe.stops(bound - 0.1, bound, "map")
    assert not probe.stops(110.0, bound, "map")


def test_a_recorded_run_held_past_100_m_s_is_read_as_stopped():
    """Records made before the rule are re-read under it: held past
    100 m/s is a stop, exactly what the probe now records for that run."""
    assert probe.held_under_peak_rule({"held": True, "peak_w": 99.9})
    assert not probe.held_under_peak_rule({"held": True, "peak_w": 119.3})
    assert not probe.held_under_peak_rule({"held": True, "peak_w": 100.0})
    assert not probe.held_under_peak_rule({"held": False, "peak_w": 30.0})
    assert not probe.held_under_peak_rule({"held": True, "peak_w": None})
