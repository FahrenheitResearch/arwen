"""A head-bound single-domain forecast keeps the clock its seal chooses.

The long step is derived from the crest-level winds of the start state
AND of the boundary data (``terrain_clock.clock_for_prepared_cache``).  A
forecast bound to a chained head reads only the start state, so each
boundary interval is folded into the reading as it loads, and one that
moves the clock ends the attempt; the runner then runs again on the seal,
says so on the supervisor heartbeat as a new attempt, and keeps the
attempt's outputs in their own folder.  The same repair the tree runner
has (A136 L6), on the single-domain runner (A136 L3).

CPU only; no device, no source data.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import requires_cupy

from gpuwm import supervisor
from gpuwm.ingest.boundary_stream import StreamedClockChanged


def _clock(dt="15", division=1, sound=4):
    return {"domains": [{"grid_id": 1, "dt_s": dt, "step_division": division,
                         "time_step_sound": sound}]}


def test_a_set_aside_attempt_leaves_the_heartbeat_where_the_watchdog_reads_it(
        tmp_path):
    from gpuwm import prepared_domain_tree_forecast as tree

    outdir = tmp_path / "run"
    outdir.mkdir()
    for number in (1, 2):
        (outdir / "wrfout").mkdir()
        (outdir / "wrfout" / "frame").write_text(str(number))
        (outdir / "progress.json").write_text(str(number))
        (outdir / supervisor.HEARTBEAT_NAME).write_text(f"beat {number}")
        tree._set_aside_streamed_attempt(outdir)
    assert sorted(entry.name for entry in outdir.iterdir()) == [
        supervisor.HEARTBEAT_NAME, "streamed-attempt", "streamed-attempt-2"]
    assert (outdir / supervisor.HEARTBEAT_NAME).read_text() == "beat 2"
    for name, number in (("streamed-attempt", "1"),
                         ("streamed-attempt-2", "2")):
        assert (outdir / name / "wrfout" / "frame").read_text() == number
        assert sorted(entry.name for entry in (outdir / name).iterdir()) \
            == ["progress.json", "wrfout"]


def test_a_declared_new_attempt_is_not_a_regression(tmp_path):
    """The restart record lets steps and model time start again."""

    path = tmp_path / supervisor.HEARTBEAT_NAME
    beat = supervisor.RuntimeHeartbeat(
        path, run_id="run", config_sha256="a" * 64,
        started_at_utc="2026-09-30T12:00:00Z")
    beat(model_elapsed_seconds=7200.0, outer_step=240,
         last_durable_wrfout=None, last_checkpoint=None)
    before = supervisor.read_heartbeat(path)
    supervisor.restart_attempt(beat, "the terrain clock moved")
    after = supervisor.read_heartbeat(path)
    assert after.status == supervisor.RESTART_STATUS
    assert after.restart == {"attempt": 2,
                             "reason": "the terrain clock moved"}
    assert after.outer_step == 0 and after.model_elapsed_seconds == 0.0
    assert supervisor._heartbeat_regression(before, after) is None
    beat(model_elapsed_seconds=60.0, outer_step=2,
         last_durable_wrfout=None, last_checkpoint=None)
    stepping = supervisor.read_heartbeat(path)
    assert stepping.restart == after.restart
    assert supervisor._heartbeat_regression(after, stepping) is None
    # Without the record the same step count going back is still read as
    # a regression: only a declared attempt starts again.
    undeclared = supervisor.Heartbeat(**{**stepping.as_dict(),
                                         "restart": None})
    assert "outer_step moved backward" in supervisor._heartbeat_regression(
        before, undeclared)


class _Observer:
    def __init__(self):
        self.restarts = []

    def restarting(self, reason):
        self.restarts.append(reason)


@requires_cupy
@pytest.mark.parametrize("resumed", [False, True])
def test_a_moved_clock_runs_the_single_domain_forecast_again_on_the_seal(
        tmp_path, monkeypatch, capsys, resumed):
    from gpuwm import prepared_single_domain_forecast as psdf

    head_inputs = SimpleNamespace(
        physics_receipt={"terrain_clock": _clock()},
        stream_head={"head_sha256": "d" * 64})
    sealed_inputs = SimpleNamespace(
        physics_receipt={"terrain_clock": _clock(dt="10", division=2)},
        stream_head=None)
    runs, seals = [], []

    def preflight(**binding):
        return head_inputs

    def sealed_binding(inputs, *, outdir, observer):
        # The heartbeat is still in place while the rerun waits for the
        # seal, and the attempt is already set aside.
        seals.append((inputs, sorted(entry.name for entry in outdir.iterdir())))
        return sealed_inputs

    def run_prepared_forecast(bound, *, output_directory, restart, **_):
        runs.append((bound, restart))
        wrfout = output_directory / "wrfout"
        wrfout.mkdir()
        (wrfout / "wrfout_d01").write_text(
            "attempt" if bound is head_inputs else "sealed")
        if bound is head_inputs:
            raise StreamedClockChanged(
                "boundary interval 3600 s to 7200 s carries a crest-level "
                "wind that moves the terrain-derived clock of d01")
        return {"schema": "s", "status": "PASS", "source": "gfs",
                "run_seconds": 7200.0, "history_interval_seconds": 3600.0,
                "gridded_output": {"frame_count": 3},
                "input": {"prepared_content_sha256": "c" * 64}}

    monkeypatch.setattr(psdf, "preflight_prepared_forecast", preflight)
    monkeypatch.setattr(psdf, "_sealed_binding", sealed_binding)
    monkeypatch.setattr(psdf, "run_prepared_forecast", run_prepared_forecast)
    observer = _Observer()
    outdir = tmp_path / "out"
    checkpoint = tmp_path / "earlier" / "gpuwmrst_d01"
    checkpoint.parent.mkdir()
    code = psdf.main([
        "--source", "gfs", "--prepared-root", str(tmp_path / "bundle"),
        "--prepared-head-sha256", "d" * 64,
        "--source-manifest-sha256", "0" * 64,
        "--experiment-config", str(tmp_path / "e.toml"),
        "--wps-namelist", str(tmp_path / "n.wps"),
        "--run-seconds", "7200", "--history-interval-seconds", "3600",
        "--io-mode", "history", "--outdir", str(outdir),
        *(["--restart", str(checkpoint)] if resumed else []),
    ], observer=observer)
    err = capsys.readouterr().err
    assert code == 0, err
    assert [bound for bound, _ in runs] == [head_inputs, sealed_inputs]
    assert runs[0][1] == (checkpoint if resumed else None)
    # A checkpoint stepped on the head's clock is not resumed on the seal.
    assert runs[1][1] is None
    assert ("the sealed forecast runs from its start time" in err) == resumed
    assert observer.restarts and "terrain-derived clock" in observer.restarts[0]
    assert seals and seals[0][0] is head_inputs
    assert "streamed-attempt" in seals[0][1]
    attempt = outdir / "streamed-attempt"
    assert (attempt / "wrfout" / "wrfout_d01").read_text() == "attempt"
    assert (outdir / "wrfout" / "wrfout_d01").read_text() == "sealed"
    assert f"kept in {attempt}" in err
