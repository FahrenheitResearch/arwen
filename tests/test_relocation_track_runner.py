"""The track writer riding the relocation runner.

Everything here is about the seam rather than the formats (those are
pinned in ``test_storm_track_writer.py``): does the runner wake up on
the right boundaries, does it reuse the mover's own fix when the two
share one, does it locate independently when they do not, and -- the
property the whole feature has to have -- does a run WITHOUT a track
block behave exactly as it did before the feature existed?

The last one is the important one.  A track file is a diagnostic.  If
configuring one changed a single decision the runner made, the feature
would be steering the model it claims to observe.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.relocation_runner import RelocationRunner
from gpuwm.core.storm_track_writer import TrackConfig, TrackWriter
from gpuwm.experiment import RelocationConfig

from test_nest_relocation_staging import (_cpu_tree, _footprint_statics,
                                          _initializer, _ramp_state)

import datetime as dt

INIT = dt.datetime(2025, 10, 24, 12)


class _Grid:
    # e_we/e_sn are WRF's STAGGERED counts; a position-only row reads
    # them to find the middle of the domain (grid_center_latlon).
    def __init__(self, lat0=14.0, lon0=-74.0, step=0.02, e_we=7, e_sn=7):
        self.lat0, self.lon0, self.step = lat0, lon0, step
        self.e_we, self.e_sn = e_we, e_sn

    def ij_to_latlon(self, x, y):
        return (self.lat0 + (y - 1.0) * self.step,
                self.lon0 + (x - 1.0) * self.step)


def _clock(ticks):
    # ``spec`` carries the domain's activation epoch: the state publisher
    # derives ``domain_start_offset`` from ``spec.start_ticks``
    # (gpuwm/core/state.py), so a fake clock without it cannot cross a
    # period boundary.  Both domains here start with the run.
    return SimpleNamespace(ticks=int(ticks), tick_den=1,
                           elapsed_seconds=float(ticks),
                           spec=SimpleNamespace(start_ticks=0,
                                                step_ticks=60))


def _model(parent, child):
    nodes = {int(parent.cfg.grid_id): parent, int(child.cfg.grid_id): child}
    model = SimpleNamespace(
        root=parent,
        schedule=SimpleNamespace(period_ticks=60,
                                 clock=SimpleNamespace(tick_den=1)),
        experiment_fingerprint="f" * 64)
    model.node = lambda gid: nodes[gid]
    return model


def _dress(parent, child):
    """Give the fake tree the two things a track row is read from."""
    for node in (parent, child):
        node.grid = _Grid()
        node.state.physics = SimpleNamespace(fields={
            "u10": np.full((6, 6), 30.0, dtype=np.float32),
            "v10": np.full((6, 6), 40.0, dtype=np.float32)})
    return parent, child


class _Provider:
    """A tracker stand-in implementing BOTH halves of the real seam.

    ``desired_shift`` for the runner and ``locate`` for the writer, with
    ``last_fix`` cached exactly as :class:`StormTracker` caches it -- so
    this test exercises the reuse path rather than papering over it.
    """

    def __init__(self, shifts=()):
        self.shifts = dict(shifts)
        self.last_fix = None
        self.locates = []
        self.consultations = []

    def locate(self, parent_state, footprint, t, refinement=None):
        self.locates.append(float(t))
        fix = SimpleNamespace(
            found={"ci": 10.0 + t / 600.0, "cj": 8.0, "cells": 5,
                   "max_value": -968.0},
            evidence={"t": float(t), "extremum_units": "hPa"},
            field_used="pressure", search_box=(slice(0, 6), slice(0, 6)),
            center_parent_ij=(10.0 + t / 600.0, 8.0), extremum=968.0,
            refined_on=None, refined_cell_ij=None,
            footprint=SimpleNamespace(grid_id=2), plane_shape=(6, 6),
            threshold_used=1004.0, raw_shift=(0.0, 0.0))
        self.last_fix = fix
        return fix

    def __call__(self, parent_state, footprint, t, refinement=None):
        self.consultations.append(float(t))
        self.locate(parent_state, footprint, t, refinement=refinement)
        return self.shifts.get(float(t))

    desired_shift = __call__


def _config(cadence=None, track=None):
    return RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=4,
        min_overlap_fraction=0.25, cadence_seconds=cadence,
        follow=SimpleNamespace(refine_grid_id=None, field="pressure"),
        track=track)


def _runner(parent_plane, config, provider, **kwargs):
    kwargs.setdefault("staging", "host")
    kwargs.setdefault("initializer", _initializer(parent_plane))
    kwargs.setdefault("static_provenance", "synthetic (test)")
    kwargs.setdefault("on_child_built", lambda *args: None)
    return RelocationRunner(
        config=config, provider=provider,
        schedule=SimpleNamespace(period_ticks=60,
                                 clock=SimpleNamespace(tick_den=1)),
        **kwargs)


def _seconds(rows):
    """The clock column of each row, back to model seconds since INIT.

    The file carries a VALID TIME, so a test that wants "which
    boundaries fired" has to convert -- and doing it here rather than
    inline keeps every assertion below reading as a list of boundaries.
    """
    import datetime as _dt
    from gpuwm.core.storm_track_writer import TIME_FORMAT
    return [(_dt.datetime.strptime(r.split(",")[0], TIME_FORMAT)
             - INIT).total_seconds() for r in rows]


def _writer(tmp_path, config):
    return TrackWriter(config, initial_time=INIT, outdir=tmp_path)


def _run(runner, model, parent, child, ticks):
    for tick in ticks:
        parent.clock = _clock(tick)
        child.clock = _clock(tick)
        runner.on_period_begin(model, {1: parent.clock, 2: child.clock},
                               period=tick // 60)


# ---------------------------------------------------------------------------
# Absent means byte-identical
# ---------------------------------------------------------------------------

def test_no_track_config_writes_no_file_and_changes_no_decision(tmp_path):
    """The property the feature must have: configuring nothing behaves
    exactly as the runner did before the writer existed."""
    def run(with_track):
        parent_plane, parent, child = _cpu_tree()
        _dress(parent, child)
        model = _model(parent, child)
        provider = _Provider({120.0: (1, 0), 300.0: (0, 1)})
        writer = (_writer(tmp_path / ("on" if with_track else "off"),
                          TrackConfig("t.csv"))
                  if with_track else None)
        runner = _runner(parent_plane, _config(cadence=60.0), provider,
                         track_writer=writer)
        _run(runner, model, parent, child, range(60, 421, 60))
        runner.close_receipt(model)
        return (provider.consultations,
                [(int(child.cfg.i_parent_start),
                  int(child.cfg.j_parent_start))],
                [r["event"] for r in runner.receipts])

    assert run(False) == run(True)
    assert not (tmp_path / "off").exists()
    assert (tmp_path / "on" / "t.csv").is_file()


def test_a_runner_without_a_writer_has_no_track_cadence(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    runner = _runner(parent_plane, _config(cadence=60.0), _Provider())
    assert runner.track_cadence_periods is None
    assert runner.track_writer is None


# ---------------------------------------------------------------------------
# The free case: one consultation serves both
# ---------------------------------------------------------------------------

def test_a_track_at_the_mover_cadence_reuses_the_mover_s_own_fix(tmp_path):
    """The default interval costs a format call and nothing else: the
    provider is consulted once per boundary, not twice."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    provider = _Provider()
    writer = _writer(tmp_path, TrackConfig("t.csv"))
    runner = _runner(parent_plane, _config(cadence=120.0), provider,
                     track_writer=writer)
    _run(runner, model, parent, child, range(60, 601, 60))
    runner.close_receipt(model)
    # Five cadence boundaries (120, 240, 360, 480, 600), one locate each.
    assert provider.consultations == [120.0, 240.0, 360.0, 480.0, 600.0]
    assert provider.locates == provider.consultations
    rows = (tmp_path / "t.csv").read_text(
        encoding="utf-8").splitlines()[1:]
    assert _seconds(rows) == provider.consultations


def test_a_held_cadence_still_writes_a_track_row(tmp_path):
    """The storm was there whether or not the nest was allowed to follow
    it, so a hold is a row -- otherwise the track would be a record of
    MOVES rather than of the vortex."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    provider = _Provider()                      # never proposes anything
    writer = _writer(tmp_path, TrackConfig("t.csv"))
    runner = _runner(parent_plane, _config(cadence=60.0), provider,
                     track_writer=writer)
    _run(runner, model, parent, child, range(60, 301, 60))
    runner.close_receipt(model)
    assert all(r["event"] == "held" for r in runner.receipts
               if r["event"] != "summary")
    rows = (tmp_path / "t.csv").read_text(
        encoding="utf-8").splitlines()[1:]
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# A finer interval gets its own boundaries
# ---------------------------------------------------------------------------

def test_a_finer_interval_locates_on_its_own_boundaries(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    provider = _Provider()
    writer = _writer(tmp_path, TrackConfig("t.csv", interval_seconds=60.0))
    runner = _runner(parent_plane, _config(cadence=180.0), provider,
                     track_writer=writer)
    assert runner.track_cadence_periods == 1        # every root step
    _run(runner, model, parent, child, range(60, 541, 60))
    runner.close_receipt(model)
    # The mover is consulted on ITS cadence only ...
    assert provider.consultations == [180.0, 360.0, 540.0]
    # ... while the writer got a fix at every boundary.
    assert provider.locates == [60.0 * n for n in range(1, 10)]
    rows = (tmp_path / "t.csv").read_text(
        encoding="utf-8").splitlines()[1:]
    assert len(rows) == 9


def test_a_track_only_boundary_leaves_the_mover_untouched(tmp_path):
    """locate is stateless and silent, so however often the writer looks
    the nest goes to exactly the same places."""
    def placements(interval):
        parent_plane, parent, child = _cpu_tree()
        _dress(parent, child)
        model = _model(parent, child)
        provider = _Provider({180.0: (1, 0), 360.0: (0, 1)})
        writer = (None if interval is None else
                  _writer(tmp_path / f"i{interval:g}",
                          TrackConfig("t.csv", interval_seconds=interval)))
        runner = _runner(parent_plane, _config(cadence=180.0), provider,
                         track_writer=writer)
        _run(runner, model, parent, child, range(60, 541, 60))
        runner.close_receipt(model)
        return (int(child.cfg.i_parent_start), int(child.cfg.j_parent_start),
                provider.consultations)

    assert placements(None) == placements(60.0)


# ---------------------------------------------------------------------------
# It cannot fail the forecast
# ---------------------------------------------------------------------------

def test_a_writer_that_explodes_is_a_receipt_row_not_a_dead_run(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)

    class Exploding:
        from types import SimpleNamespace as _NS
        stream = _NS(config=_NS(interval_seconds=None))

        def emit(self, *a, **k):
            raise RuntimeError("disk on fire")

        def close(self):
            return {"path": "t.csv", "records": 0, "skipped": 0}

    runner = _runner(parent_plane, _config(cadence=60.0), _Provider(),
                     track_writer=Exploding())
    runner.track_cadence_periods = 1
    _run(runner, model, parent, child, [60, 120])
    runner.close_receipt(model)
    faults = [r for r in runner.receipts if r["event"] == "track_faulted"]
    assert faults and "disk on fire" in faults[0]["reason"]
    # and the run kept going
    assert any(r["event"] == "held" for r in runner.receipts)


def test_a_provider_with_no_locate_is_simply_not_asked(tmp_path):
    """ManualMoveProvider has no locate; a track block with one is
    refused at load, but the runner must not crash if handed one."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    writer = _writer(tmp_path, TrackConfig("t.csv", interval_seconds=60.0))
    runner = _runner(parent_plane, _config(cadence=180.0),
                     lambda *a, **k: None, track_writer=writer)
    _run(runner, model, parent, child, [60, 120, 180])
    runner.close_receipt(model)
    assert (tmp_path / "t.csv").read_text(encoding="utf-8").splitlines() == [
        __import__("gpuwm.core.storm_track_writer",
                   fromlist=["CSV_HEADER"]).CSV_HEADER]


# ---------------------------------------------------------------------------
# The run-end summary
# ---------------------------------------------------------------------------

def test_the_summary_carries_a_per_stream_tally(tmp_path):
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    writer = _writer(tmp_path, TrackConfig("t.csv"))
    runner = _runner(parent_plane, _config(cadence=60.0), _Provider(),
                     track_writer=writer)
    _run(runner, model, parent, child, range(60, 3661, 60))
    summary = runner.close_receipt(model)
    tally = summary["track"]
    assert tally["records"] == 61
    assert tally["skipped"] == 0
    assert tally["path"].endswith("t.csv")
    assert summary["track_records"] == 61


# ---------------------------------------------------------------------------
# t = 0, and the empty deck it prevents
# ---------------------------------------------------------------------------

def test_the_track_emits_at_t_zero_but_the_mover_does_not(tmp_path):
    """An a-deck's first record is TAU 0 -- the analysis time.

    on_period_begin fires at the START of each period, so t =
    run_seconds is never an opportunity.  Without a t = 0 emission a run
    of exactly one hour has no whole-hour instant at all and writes an
    EMPTY deck; that was measured on the real tree (3600 s, 0 records)
    before this existed.  The mover keeps ticks != 0, because relocating
    a nest before it has integrated a single step is not a move.
    """
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    provider = _Provider({120.0: (1, 0)})
    writer = _writer(tmp_path, TrackConfig("t.csv"))
    runner = _runner(parent_plane, _config(cadence=60.0), provider,
                     track_writer=writer)
    _run(runner, model, parent, child, [0, 60, 120])
    runner.close_receipt(model)

    rows = (tmp_path / "t.csv").read_text(
        encoding="utf-8").splitlines()[1:]
    assert _seconds(rows) == [0.0, 60.0, 120.0]
    # The first row IS the initial position, at the run's own start.
    assert rows[0].split(",")[0] == INIT.strftime("%Y-%m-%d_%H:%M:%S")
    # ... and the mover was NOT consulted at t = 0.
    assert provider.consultations == [60.0, 120.0]
    assert 0.0 in provider.locates


def test_an_hour_long_run_records_its_initial_position(tmp_path):
    """The regression, stated as the shape of the run that exposed it:
    60 root periods, boundaries at 0..3540, and t=3600 never reached.
    Without the t = 0 opportunity the first row was one cadence in, and
    the run's own initial position -- the one instant a track is always
    expected to carry -- was missing."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    writer = _writer(tmp_path, TrackConfig("t.csv"))
    runner = _runner(parent_plane, _config(cadence=360.0), _Provider(),
                     track_writer=writer)
    _run(runner, model, parent, child, range(0, 3600, 60))
    runner.close_receipt(model)
    rows = (tmp_path / "t.csv").read_text(
        encoding="utf-8").splitlines()[1:]
    assert _seconds(rows)[0] == 0.0, (
        "a run's track must start at its own initial time")
    assert _seconds(rows) == [float(t) for t in range(0, 3600, 360)]


def test_a_run_without_a_track_is_still_untouched_at_t_zero(tmp_path):
    """The t = 0 opportunity belongs to the track and nothing else: a
    runner with no writer must not acquire an extra boundary."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    provider = _Provider()
    runner = _runner(parent_plane, _config(cadence=60.0), provider)
    _run(runner, model, parent, child, [0, 60, 120])
    assert provider.consultations == [60.0, 120.0]
    assert provider.locates == [60.0, 120.0]
    assert runner.receipts == [] or all(
        r["elapsed_seconds"] != 0.0 for r in runner.receipts)


# ---------------------------------------------------------------------------
# Position-only: the row is the MOVER, and it costs nothing to write
# ---------------------------------------------------------------------------

def _uh_config(cadence=None, track=None):
    import dataclasses
    return dataclasses.replace(
        _config(cadence=cadence, track=track),
        follow=SimpleNamespace(refine_grid_id=None, field="uh"))


def _uh_writer(tmp_path, config):
    return TrackWriter(config, initial_time=INIT, outdir=tmp_path,
                       tracked_field="uh")


def test_a_position_only_track_row_is_the_child_not_the_parent(tmp_path):
    """The mover is d02 and the tracker searches d01, so the two grids
    are deliberately given different origins here: a row taken from the
    wrong one is off by a whole degree and cannot pass by accident."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    parent.grid = _Grid(lat0=14.0, lon0=-74.0)
    child.grid = _Grid(lat0=20.0, lon0=-60.0)
    model = _model(parent, child)
    writer = _uh_writer(tmp_path, TrackConfig("t.csv"))
    runner = _runner(parent_plane, _uh_config(cadence=180.0), _Provider(),
                     track_writer=writer)
    _run(runner, model, parent, child, range(60, 541, 60))
    runner.close_receipt(model)
    rows = (tmp_path / "t.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "valid_time,lat_deg,lon_deg"
    lat, lon = (float(v) for v in rows[1].split(",")[1:])
    # The child's own centre, in the child's own index space.
    from gpuwm.core.storm_track_writer import grid_center_latlon
    assert (lat, lon) == pytest.approx(grid_center_latlon(child.grid))
    assert abs(lat - 14.0) > 1.0 and abs(lon + 74.0) > 1.0


def test_a_position_only_track_only_boundary_does_not_locate(tmp_path):
    """The row does not read the fix, so a finer track interval must not
    buy a whole-plane reduction per row.

    The control is right above in
    ``test_a_finer_interval_locates_on_its_own_boundaries``: the SAME
    cadences under a pressure tracker locate nine times.
    """
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    provider = _Provider()
    writer = _uh_writer(tmp_path, TrackConfig("t.csv", interval_seconds=60.0))
    runner = _runner(parent_plane, _uh_config(cadence=180.0), provider,
                     track_writer=writer)
    assert runner.track_cadence_periods == 1
    _run(runner, model, parent, child, range(60, 541, 60))
    runner.close_receipt(model)
    # Located only where the MOVER was consulted -- three times, not nine.
    assert provider.locates == [180.0, 360.0, 540.0]
    assert provider.consultations == [180.0, 360.0, 540.0]
    # ...and every boundary still wrote a row.
    rows = (tmp_path / "t.csv").read_text(
        encoding="utf-8").splitlines()[1:]
    assert len(rows) == 9
    assert _seconds(rows) == [60.0 * n for n in range(1, 10)]
    assert all(len(r.split(",")) == 3 for r in rows)


def test_a_position_only_track_steers_nothing_either(tmp_path):
    """The property the whole feature has to have, for the new shape:
    configuring a track file changes no decision the runner makes."""
    def placements(interval):
        parent_plane, parent, child = _cpu_tree()
        _dress(parent, child)
        model = _model(parent, child)
        provider = _Provider({180.0: (1, 0), 360.0: (0, 1)})
        writer = (None if interval is None else
                  _uh_writer(tmp_path / f"u{interval:g}",
                             TrackConfig("t.csv", interval_seconds=interval)))
        runner = _runner(parent_plane, _uh_config(cadence=180.0), provider,
                         track_writer=writer)
        _run(runner, model, parent, child, range(60, 541, 60))
        runner.close_receipt(model)
        return (int(child.cfg.i_parent_start), int(child.cfg.j_parent_start),
                provider.consultations)

    assert placements(None) == placements(60.0)


# ---------------------------------------------------------------------------
# The fix on the per-step stream
# ---------------------------------------------------------------------------
#
# The CSV is the archive; the event is the same fix, live, emitted at the
# instant the row is written.  A map tailing progress.jsonl can then draw
# the track without also tailing (and parsing the declared header of) a
# second file.


def _logged(model, tmp_path):
    from datetime import datetime

    from gpuwm.progress_log import StepLog, publish_step_log

    log = StepLog(start_time=datetime(2026, 8, 15, 0, 0, 0),
                  run_seconds=3600.0,
                  jsonl_path=tmp_path / "progress.jsonl")
    publish_step_log(model, log)
    return log


def _fixes(tmp_path, log):
    from gpuwm.progress_log import read_step_log

    log.close(status="SUCCESS")
    return [r for r in read_step_log(tmp_path / "progress.jsonl")
            if r["event"] == "track_fix"]


def test_every_written_track_row_is_also_an_event(tmp_path):
    """One row, one event, same position, same instant."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    log = _logged(model, tmp_path)
    writer = _writer(tmp_path / "t", TrackConfig("t.csv"))
    runner = _runner(parent_plane, _config(cadence=60.0), _Provider(),
                     track_writer=writer)
    _run(runner, model, parent, child, range(0, 241, 60))
    runner.close_receipt(model)

    rows = (tmp_path / "t" / "t.csv").read_text(
        encoding="utf-8").splitlines()[1:]
    fixes = _fixes(tmp_path, log)
    assert len(fixes) == len(rows) > 0
    assert [f["model_seconds"] for f in fixes] == _seconds(rows)
    assert all(f["domain"] == 2 for f in fixes), "the domain being STEERED"
    for fix, row in zip(fixes, rows):
        lat, lon = row.split(",")[1:3]
        assert fix["lat"] == pytest.approx(float(lat), abs=5e-7)
        assert fix["lon"] == pytest.approx(float(lon), abs=5e-7)
        assert fix["found"] is True


def test_a_run_without_a_track_block_emits_no_fixes(tmp_path):
    """No track file, no track events: the inertness rule, per feature."""
    parent_plane, parent, child = _cpu_tree()
    _dress(parent, child)
    model = _model(parent, child)
    log = _logged(model, tmp_path)
    runner = _runner(parent_plane, _config(cadence=60.0), _Provider())
    _run(runner, model, parent, child, range(0, 241, 60))
    assert _fixes(tmp_path, log) == []
