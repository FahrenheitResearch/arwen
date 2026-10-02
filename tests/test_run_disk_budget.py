"""A run keeps only the checkpoints it needs, and run-plan refuses a run its disk cannot hold.

Breakage these prevent: a 12 hour 1 km run filled a 58 GB disk at hour nine
because every hourly checkpoint was kept and nothing compared the run's
output with the free disk before the download started.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import disk_budget
from gpuwm.resume import (KEEP_CHECKPOINTS_ENV, checkpoint_retention,
                          discover_checkpoint_sets,
                          retire_superseded_checkpoints)


def _write_set(directory: Path, when: datetime, set_id: str, ids=(1, 2, 3), mtime=0,
               declared=(1, 2, 3)):
    """Members ``ids`` of a set whose header declares ``declared``, as the tree writer writes them."""
    stamp = when.strftime("%Y-%m-%d_%H_%M_%S")
    for gid in ids:
        path = directory / f"gpuwmrst_d{gid:02d}_{stamp}__{set_id}.npz"
        header = {"domain_ids": sorted(declared), "grid_id": gid,
                  "checkpoint_set_id": set_id}
        with path.open("wb") as stream:
            np.savez(stream, __gpuwm_restart_header__=np.frombuffer(
                json.dumps(header).encode("utf-8"), dtype=np.uint8))
        os.utime(path, ns=(mtime, mtime))


def test_retention_keeps_newest_complete_sets(tmp_path):
    t0 = datetime(2020, 1, 1)
    for hour in range(5):
        _write_set(tmp_path, t0 + timedelta(hours=hour), f"s{hour}", mtime=hour * 10**9)
    removed = retire_superseded_checkpoints(tmp_path, keep=1)
    assert len(removed) == 12
    left = discover_checkpoint_sets(tmp_path)
    assert [s.set_id for s in left] == ["s4"]


def test_retention_does_not_count_a_torn_newer_set(tmp_path):
    t0 = datetime(2020, 1, 1)
    _write_set(tmp_path, t0, "a", mtime=1)
    _write_set(tmp_path, t0 + timedelta(hours=1), "b", mtime=2)
    _write_set(tmp_path, t0 + timedelta(hours=2), "torn", ids=(2, 3), mtime=3)
    retire_superseded_checkpoints(tmp_path, keep=1)
    assert sorted(s.set_id for s in discover_checkpoint_sets(tmp_path)) == ["b", "torn"]


def test_retention_unset_keeps_everything(tmp_path, monkeypatch):
    monkeypatch.delenv(KEEP_CHECKPOINTS_ENV, raising=False)
    t0 = datetime(2020, 1, 1)
    for hour in range(3):
        _write_set(tmp_path, t0 + timedelta(hours=hour), f"s{hour}")
    assert checkpoint_retention() is None
    assert retire_superseded_checkpoints(tmp_path) == []
    assert len(discover_checkpoint_sets(tmp_path)) == 3
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "0")
    assert checkpoint_retention() is None
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "2")
    assert checkpoint_retention() == 2
    monkeypatch.setenv(KEEP_CHECKPOINTS_ENV, "-1")
    with pytest.raises(ValueError):
        checkpoint_retention()


def _exp(hours=12, restart=3600.0):
    from test_history_disk_layout import priced_run

    def dom(gid, nx, interval):
        return SimpleNamespace(grid_id=gid, history_interval_s=interval,
                               run=priced_run(nx, nx, 49))
    return SimpleNamespace(run_seconds=hours * 3600.0, restart_interval_s=restart,
                           domains=(dom(1, 100, 3600.0), dom(2, 200, 900.0)))


def test_projection_counts_frames_and_held_checkpoints():
    p = disk_budget.projected_run_bytes(_exp(), keep_checkpoints=1, fetch=None, chain=None, render=True)
    d1, d2 = p["domains"]
    assert (d1["history_frames"], d2["history_frames"]) == (13, 49)
    assert p["checkpoint_sets_held"] == 2
    root, nest = 100 * 100 * 49, 200 * 200 * 49
    assert p["checkpoint_bytes"] == int(root * disk_budget.ROOT_CHECKPOINT_BYTES_PER_CELL * 2
                                        + nest * disk_budget.CHECKPOINT_BYTES_PER_CELL * 2)
    assert p["picture_bytes"] == sum(disk_budget.projected_picture_bytes(
        d.run.nx, d.run.ny, 12 * 3600, d.history_interval_s) for d in _exp().domains)
    # Nothing downloaded and nothing prepared: the whole is the grid's own output.
    assert (p["download_bytes"], p["preparation_bytes"]) == (0, 0)
    assert p["total_bytes"] == p["history_bytes"] + p["checkpoint_bytes"] + p["picture_bytes"]
    keep_all = disk_budget.projected_run_bytes(_exp(), keep_checkpoints=None, fetch=None, chain=None, render=True)
    assert keep_all["checkpoint_sets_held"] == 12



@pytest.mark.parametrize("delay,interval", [(0, 60), (120, 60), (120, 180), (540, 60)])
def test_a_late_nest_is_charged_the_frames_its_clock_writes(delay, interval):
    """A nest that starts late writes history from its own start.

    Admission charged it every frame of the whole run: a nest starting
    540 s into a 600 s run at one frame a minute was priced at 11 frames
    and pictures where the scheduler writes 2, and a run that fit its
    disk was refused.  The frames are counted here by the real clock and
    schedule executor, not by a second formula.
    """
    from dataclasses import replace

    from gpuwm.core.clock import build_schedule, execute_schedule, resolve_clock
    from test_clock import _chain_experiment, _with_delayed_start

    exp = _with_delayed_start(
        _chain_experiment((1, 3), run_seconds=600, history_s=interval), 2, delay)
    exp = replace(exp, restart_interval_s=60)
    written = execute_schedule(build_schedule(exp, resolve_clock(exp, lbc_interval_s=60)))
    p = disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None, chain=None, render=True)
    assert {row["grid_id"]: row["history_frames"] for row in p["domains"]} == written.histories
    cells = 30 * 30 * 4
    frames = sum(written.histories.values())
    # A190: volume-only scaling ignored the selected writer inventory,
    # including horizontal faces, surface fields and container headers.
    assert p["history_bytes"] == sum(
        disk_budget.history_frame_bytes(domain.run) * written.histories[domain.grid_id]
        for domain in exp.domains)
    assert p["picture_bytes"] == sum(disk_budget.projected_picture_bytes(
        30, 30, max(0, 600 - (delay if gid == 2 else 0)), interval)
        for gid in written.histories)
    # A checkpoint set carries a nest that has not started yet too, so the
    # checkpoint charge does not move with the start.
    assert p["checkpoint_sets_held"] == 2
    assert p["checkpoint_bytes"] == int(2 * cells * (
        disk_budget.ROOT_CHECKPOINT_BYTES_PER_CELL + disk_budget.CHECKPOINT_BYTES_PER_CELL))
    # The refusal is drawn at exactly what the run writes.
    assert disk_budget.disk_refusal(p, p["total_bytes"]) is None
    assert disk_budget.disk_refusal(p, p["total_bytes"] - 1) is not None
    if delay == 540:
        assert written.histories == {1: 11, 2: 2}
    # A run that draws no pictures is charged the same late-start history
    # and nothing for pictures.
    dark = disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None, chain=None, render=False)
    assert [row["history_frames"] for row in dark["domains"]] == [
        row["history_frames"] for row in p["domains"]]
    assert dark["history_bytes"] == p["history_bytes"]
    assert dark["picture_bytes"] == 0
    assert dark["total_bytes"] == p["total_bytes"] - p["picture_bytes"]


def test_a_nest_that_starts_after_the_run_ends_is_charged_no_frames():
    from test_clock import _chain_experiment, _with_delayed_start

    exp = _with_delayed_start(
        _chain_experiment((1, 3), run_seconds=600, history_s=60), 2, 900)
    p = disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None, chain=None, render=True)
    assert [row["history_frames"] for row in p["domains"]] == [11, 0]
    assert p["domains"][1]["picture_bytes"] == 0


def test_projection_reproduces_the_measured_run():
    """The run the constants were read from: its projection is within 3% of what it wrote, and not under it."""
    import json
    record = json.loads((Path(__file__).resolve().parents[1] / "tools" / "wiki_seed" / "runs"
                         / "bytes-per-cell" / "sizes.json").read_text(encoding="utf-8"))
    intervals = {"d01": 3600.0, "d02": 900.0, "d03": 900.0}
    from gpuwm.config import RunConfig

    # A190 retires the full-history constant. Resolve the physics recorded
    # for this saved measurement, rather than pricing a dimensions-only grid.
    exp = SimpleNamespace(run_seconds=2 * 3600.0, restart_interval_s=3600.0, domains=tuple(
        SimpleNamespace(grid_id=i + 1, history_interval_s=intervals[name],
                        run=RunConfig(nx=g["columns"][0], ny=g["columns"][1], nz=g["levels"],
                                      dx=g["dx_km"] * 1000, dy=g["dx_km"] * 1000,
                                      ztop=20000, dt=15, run_seconds=7200,
                                      moist=True, mp_physics=10, sf_sfclay_physics=91,
                                      sf_surface_physics=2, bl_pbl_physics=1,
                                      ra_lw_physics=4, ra_sw_physics=4, nwp_diagnostics=1))
        for i, (name, g) in enumerate(sorted(record["grids"].items()))))
    # This older sample also drew raw variables, now an explicit request.
    p = disk_budget.projected_run_bytes(exp, keep_checkpoints=None, fetch=None, chain=None,
                                        render=True, render_products="all,variables")
    wrote = {"history": 0, "checkpoint": 0}
    for name, size in record["files"].items():
        wrote["history" if name.startswith("wrfout") else "checkpoint"] += size
    for key in ("history", "checkpoint"):
        assert wrote[key] <= p[f"{key}_bytes"] <= 1.03 * wrote[key], key
    pictures = sum(g["picture_bytes"] for g in record["grids"].values())
    assert pictures <= p["picture_bytes"]


def test_refusal_names_both_numbers():
    p = disk_budget.projected_run_bytes(_exp(), keep_checkpoints=1, fetch=None, chain=None, render=True)
    assert disk_budget.disk_refusal(p, p["total_bytes"] + 1) is None
    words = disk_budget.disk_refusal(p, 2 * disk_budget.GIB)
    assert f"{p['total_bytes'] / disk_budget.GIB:.1f} GiB" in words
    assert "2.0 GiB free" in words


@pytest.mark.parametrize("begin,end,resume", [(1, 420, None), (61, 481, 180),
                                              (120, 300, 300), (181, None, 120)])
def test_history_windows_and_resume_match_the_executed_clock(begin, end, resume):
    from dataclasses import replace
    from gpuwm.core.clock import build_schedule, execute_schedule, resolve_clock
    from test_clock import _chain_experiment, _with_delayed_start

    exp = _with_delayed_start(
        _chain_experiment((1, 3), run_seconds=600, history_s=60), 2, 120)
    exp = replace(exp, domains=tuple(replace(domain, history_begin_s=begin,
                                            history_end_s=end)
                                     for domain in exp.domains))
    clock = resolve_clock(exp, lbc_interval_s=60)
    schedule = build_schedule(exp, clock)
    options = {}
    if resume is not None:
        clocks = clock.clocks()
        for item in clocks.values():
            item.ticks = resume * clock.tick_den
        options = dict(clocks=clocks, start_period=resume * clock.tick_den // schedule.period_ticks,
                       started_grid_ids={gid for gid, item in clocks.items()
                                         if item.spec.start_ticks <= item.ticks},
                       committed_initial_history_grid_ids={gid for gid, item in clocks.items()
                                                           if item.history_due()})
    written = execute_schedule(schedule, **options)
    projection = disk_budget.projected_run_bytes(
        exp, keep_checkpoints=1, fetch=None, chain=None, render=False,
        resume_seconds=resume)
    assert {row["grid_id"]: row["history_frames"] for row in projection["domains"]} == {
        domain.grid_id: written.histories.get(domain.grid_id, 0) for domain in exp.domains}


def test_domain_selection_overrides_tree_without_trimming_checkpoints():
    from dataclasses import replace
    from gpuwm.io.history_selection import HistorySelection
    from test_clock import _chain_experiment

    exp = _chain_experiment((1, 3), run_seconds=600, history_s=60)
    full = disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None,
                                           chain=None, render=False)
    trimmed = HistorySelection.from_mapping({"history_vars": ["MU"]})
    exp = replace(exp, output=trimmed, domains=(exp.domains[0],
                  replace(exp.domains[1], output=HistorySelection())))
    projected = disk_budget.projected_run_bytes(exp, keep_checkpoints=1, fetch=None,
                                                chain=None, render=False)
    assert projected["domains"][0]["history_bytes"] < full["domains"][0]["history_bytes"]
    assert projected["domains"][1]["history_bytes"] == full["domains"][1]["history_bytes"]
    assert projected["checkpoint_bytes"] == full["checkpoint_bytes"]
    free = projected["total_bytes"]
    assert disk_budget.disk_refusal(projected, free) is None
    assert "partway when the disk fills" in disk_budget.disk_refusal(full, free)


def test_run_option_keep_checkpoints_is_a_whole_number(tmp_path):
    from gpuwm.runplan import PlanError, _run_option
    assert _run_option("keep_checkpoints", 3, tmp_path) == 3
    for bad in (-1, True, 1.5, "2"):
        with pytest.raises(PlanError):
            _run_option("keep_checkpoints", bad, tmp_path)


def test_surface_flux_option_lands_in_shared():
    from gpuwm.domain_wizard import with_surface_flux_option
    text = '[experiment]\nname = "x"\n\n[shared]\nnz = 49\nisftcflx = 0\n\n[[domain]]\ngrid_id = 1\n'
    out = with_surface_flux_option(text, 1)
    shared = out.split("[shared]")[1].split("[[domain]]")[0]
    assert [l for l in shared.splitlines() if l.startswith("isftcflx")] == ["isftcflx = 1"]
    assert with_surface_flux_option(text, None) == text
    quoted = with_surface_flux_option(text.replace("[shared]", '["shared"]'), 2)
    assert "isftcflx = 2" in quoted
    with pytest.raises(ValueError):
        with_surface_flux_option(text, 3)
