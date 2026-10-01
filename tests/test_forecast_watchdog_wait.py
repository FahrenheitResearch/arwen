"""A forecast waiting at a seam is timed by its wait, not by the step bound.

The breakage (A136 L4): a chained forecast that catches up with its
preparation waits at a seam for the next boundary interval.  The wait
refreshed only ``progress.json``, and the watchdog `gpuwm go` runs beside
the forecast reads ``run-progress.json``, so a wait longer than
max(3 x p99 step, 120 s) was stopped as "forecast stalled" while the
source lead it waited for was still on schedule.  Every rolling source
posts slower than the model runs, so such waits are the normal state of a
run that starts as its source posts.

Now the wait publishes ``waiting:source`` or ``waiting:preparation`` with
its record on the heartbeat, and the watchdog bounds a source wait by the
lead's ``late_at`` + 120 s and a preparation wait by the producer silence
limit.  The cause comes from the producer heartbeat, and a lead later than
its budget ends the run by name with exit 75.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from gpuwm import supervisor
from gpuwm.forecast_supervisor import ForecastHeartbeat, ForecastWatchdog
from gpuwm.ingest import boundary_stream as bs


def _iso(instant: datetime) -> str:
    return instant.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _watchdog(tmp_path, *, prepared_root=None):
    outdir = tmp_path / "run"
    outdir.mkdir(parents=True, exist_ok=True)
    command = ["python", "-m", "gpuwm.prepared_single_domain_forecast",
               "--outdir", str(outdir)]
    if prepared_root is not None:
        command[3:3] = ["--prepared-root", str(prepared_root)]
    clock = _Clock()
    watchdog = ForecastWatchdog(command, clock=clock)
    heartbeat = ForecastHeartbeat(
        outdir / supervisor.HEARTBEAT_NAME, run_id=watchdog.run_id,
        config_sha256=watchdog.digest, started_at_utc=watchdog.started_at)
    return watchdog, heartbeat, clock, outdir


def _step_to(heartbeat, watchdog, clock, steps: int, *, wall: float = 1.0):
    """Integrate ``steps`` outer steps of ``wall`` seconds each."""

    for step in range(steps + 1):
        heartbeat(model_elapsed_seconds=10.0 * step, outer_step=step,
                  last_durable_wrfout=None, last_checkpoint=None)
        assert watchdog.check(os.getpid()) is None
        clock.now += wall


def _write_progress(outdir: Path, waited: float) -> None:
    (outdir / "progress.json").write_text(json.dumps({
        "status": "INTEGRATING", "model_elapsed_seconds": 50.0,
        "waiting": {"reason": "boundary interval 1 (3600 s to 7200 s) is "
                              "not prepared yet",
                    "waited_seconds": waited}}), encoding="utf-8")


def test_a_seam_wait_said_only_in_progress_json_is_stopped_as_a_stall(tmp_path):
    """The reproduction: what a seam wait did before the fix.

    The worker integrates, then sits at a seam for 150 s refreshing only
    progress.json.  The watchdog reads run-progress.json, sees an
    integrating record that stopped moving, and stops the worker once the
    step bound (120 s floor) passes, while the wait was legitimate.
    """

    watchdog, heartbeat, clock, outdir = _watchdog(tmp_path)
    _step_to(heartbeat, watchdog, clock, 5)
    verdict = None
    for waited in range(0, 151, 5):
        _write_progress(outdir, float(waited))
        verdict = watchdog.check(os.getpid())
        if verdict is not None:
            break
        clock.now += 5.0
    assert verdict is not None and "forecast stalled in integrating" in verdict
    assert waited < 150


def _seam_waits(heartbeat, outdir, *, clock, emitted=None, producer=None):
    def publish(block):
        payload = {"status": "INTEGRATING" if block else "RUNNING",
                   "model_elapsed_seconds": 50.0}
        if block is not None:
            payload["waiting"] = block
        (outdir / "progress.json").write_text(json.dumps(payload),
                                              encoding="utf-8")

    def emit(event, **fields):
        if emitted is not None:
            emitted.append({"event": event, **fields})

    return bs.SeamWaits(
        emit=emit, observer=heartbeat, publish=publish,
        model_time=lambda index: {
            "phase": "seam", "interval": index,
            "model_elapsed_seconds": 50.0,
            "model_valid_time": "2026-09-30T12:00:50Z"},
        log_path=outdir / bs.WAIT_LOG_NAME, producer_path=producer,
        clock=clock)


def _source_cause(*, lead=3, expected=None, late=None):
    now = datetime.now(timezone.utc)
    return {"on": "source", "source": "gefs", "cycle": "2026-09-30T12",
            "lead": lead, "valid_time": "2026-09-30T15:00:00Z",
            "expected_at": _iso(expected or now - timedelta(minutes=1)),
            "late_at": _iso(late or now + timedelta(minutes=59)),
            "since_utc": _iso(now)}


def test_the_same_seam_wait_published_on_the_heartbeat_is_not_stopped(tmp_path):
    """After the fix: the wait says itself on the heartbeat and outlives 150 s."""

    watchdog, heartbeat, clock, outdir = _watchdog(tmp_path)
    _step_to(heartbeat, watchdog, clock, 5)
    waits = _seam_waits(heartbeat, outdir, clock=clock)
    for waited in range(0, 181, 5):
        waits({"reason": "gefs f003 is not posted yet", "interval": 1,
               "waited_seconds": float(waited), "cause": _source_cause()})
        assert watchdog.check(os.getpid()) is None, waited
        clock.now += 5.0
    record = supervisor.read_heartbeat(outdir / supervisor.HEARTBEAT_NAME)
    assert record.status == "waiting:source"
    assert record.wait["on"] == "source" and record.wait["lead"] == 3
    assert set(record.wait) == set(supervisor.WAIT_RECORD_FIELDS)
    # The wait ends: the record goes back to integrating and the run steps on.
    waits(None)
    record = supervisor.read_heartbeat(outdir / supervisor.HEARTBEAT_NAME)
    assert record.status == "integrating" and record.wait is None
    assert watchdog.check(os.getpid()) is None


def test_a_source_wait_past_its_late_time_is_still_stopped(tmp_path):
    """The backstop: a source wait past late_at + 120 s ends the run."""

    watchdog, heartbeat, clock, outdir = _watchdog(tmp_path)
    _step_to(heartbeat, watchdog, clock, 3)
    now = datetime.now(timezone.utc)
    waits = _seam_waits(heartbeat, outdir, clock=clock)
    cause = _source_cause(late=now - timedelta(seconds=90))
    waits({"reason": "late", "interval": 1, "waited_seconds": 0.0,
           "cause": cause})
    # 90 s past late_at: inside the 120 s grace.
    assert watchdog.check(os.getpid()) is None
    watchdog._utc_clock = lambda: now + timedelta(seconds=31)
    clock.now += 5.0
    waits({"reason": "late", "interval": 1, "waited_seconds": 5.0,
           "cause": cause})
    verdict = watchdog.check(os.getpid())
    assert verdict is not None and "past its late time" in verdict
    assert "lead 3" in verdict


def test_a_waiting_record_that_stops_refreshing_is_a_hung_worker(tmp_path):
    """Refreshing is liveness: silence past the producer limit stops it."""

    stream = tmp_path / "prepared" / bs.STREAM_DIRNAME
    stream.mkdir(parents=True)
    (stream / bs.PRODUCER_NAME).write_text(json.dumps(
        {"slowest_build_seconds": 50.0}), encoding="utf-8")
    watchdog, heartbeat, clock, outdir = _watchdog(
        tmp_path, prepared_root=tmp_path / "prepared")
    _step_to(heartbeat, watchdog, clock, 3)
    waits = _seam_waits(heartbeat, outdir, clock=clock)
    waits({"reason": "boundary interval 1 is not prepared yet",
           "interval": 1, "waited_seconds": 0.0, "cause": None})
    assert watchdog.check(os.getpid()) is None
    # The limit is max(120 s, 3 x the slowest build, 50 s) = 150 s.
    clock.now += 149.0
    assert watchdog.check(os.getpid()) is None
    clock.now += 2.0
    verdict = watchdog.check(os.getpid())
    assert verdict is not None and "waiting:preparation" in verdict
    assert "bound 150.0 s" in verdict


def test_waits_never_enter_the_step_wall_history(tmp_path):
    watchdog, heartbeat, clock, outdir = _watchdog(tmp_path)
    _step_to(heartbeat, watchdog, clock, 4, wall=2.0)
    before = list(watchdog.history._values)
    waits = _seam_waits(heartbeat, outdir, clock=clock)
    for _ in range(40):
        waits({"reason": "r", "interval": 1, "waited_seconds": 0.0,
               "cause": None})
        watchdog.check(os.getpid())
        clock.now += 5.0
    waits(None)
    watchdog.check(os.getpid())
    heartbeat(model_elapsed_seconds=60.0, outer_step=6,
              last_durable_wrfout=None, last_checkpoint=None)
    watchdog.check(os.getpid())
    assert all(value <= 5.0 for value in watchdog.history._values)
    assert len(watchdog.history._values) >= len(before)


def test_the_wait_record_belongs_to_waiting_records_only():
    base = dict(schema=supervisor.HEARTBEAT_SCHEMA, run_id="r",
                config_digest="d", pid=1, started_at_utc="s",
                updated_at_utc="2026-09-30T00:00:00+00:00",
                model_elapsed_seconds=0.0, outer_step=0,
                last_durable_wrfout=None, last_checkpoint=None)
    wait = {"on": "source", "lead": 3, "expected_at": None, "late_at": None,
            "since_utc": "x"}
    record = supervisor.Heartbeat(status="waiting:source", wait=wait, **base)
    assert supervisor.Heartbeat.from_mapping(record.as_dict()) == record
    assert "wait" not in supervisor.Heartbeat(status="integrating",
                                              **base).as_dict()
    with pytest.raises(ValueError, match="waiting record"):
        supervisor.Heartbeat(status="integrating", wait=wait, **base)
    with pytest.raises(ValueError, match="does not match"):
        supervisor.Heartbeat(status="waiting:preparation", wait=wait, **base)
    with pytest.raises(ValueError, match="exactly"):
        supervisor.Heartbeat(status="waiting:source",
                             wait={"on": "source"}, **base)
    with pytest.raises(ValueError, match="invalid heartbeat status"):
        supervisor.Heartbeat(status="waiting:card", **base)


def test_finalization_goes_back_only_to_the_seal_wait():
    """After the last step the one wait left is the seal's.

    A head-bound forecast that has stepped through its last interval binds
    the seal after its ``finalizing:bind-prepared-seal`` record, and while
    the preparation is still sealing that wait publishes
    ``waiting:preparation``.  THE BREAKAGE this pin moved for: it was read
    as a regression, so `gpuwm go`'s watchdog stopped (124) a forecast
    whose steps were all done whenever it caught up with the seal.  A
    source wait after finalization is still one: no source lead is left
    to wait on once every interval has been integrated.
    """

    base = dict(schema=supervisor.HEARTBEAT_SCHEMA, run_id="r",
                config_digest="d", pid=1, started_at_utc="s",
                model_elapsed_seconds=0.0, outer_step=0,
                last_durable_wrfout=None, last_checkpoint=None)
    finishing = supervisor.Heartbeat(
        status="finalizing:bind-prepared-seal",
        updated_at_utc="2026-09-30T00:00:00+00:00", **base)
    waiting = supervisor.Heartbeat(
        status="waiting:preparation",
        updated_at_utc="2026-09-30T00:00:01+00:00",
        wait={"on": "preparation", "lead": None, "expected_at": None,
              "late_at": None, "since_utc": "x"}, **base)
    assert supervisor._heartbeat_regression(finishing, waiting) is None
    source = supervisor.Heartbeat(
        status="waiting:source",
        updated_at_utc="2026-09-30T00:00:01+00:00",
        wait={"on": "source", "lead": 3, "expected_at": None,
              "late_at": None, "since_utc": "x"}, **base)
    assert "moved backward" in supervisor._heartbeat_regression(
        finishing, source)
    for status in ("integrating", "preparing:restore-checkpoint"):
        assert "moved backward" in supervisor._heartbeat_regression(
            finishing, supervisor.Heartbeat(
                status=status, updated_at_utc="2026-09-30T00:00:01+00:00",
                **base))


def test_a_watchdog_poll_that_missed_the_seal_announcement_accepts_the_wait(
        tmp_path):
    """Why the seal wait is accepted after any finalizing record.

    The watchdog reads ``run-progress.json`` on its own poll.  The end-of-run
    seal wait publishes ``waiting:preparation`` as soon as it finds no seal,
    so ``finalizing:bind-prepared-seal`` stands for well under a poll and
    the record a poll last read is the one before it.  Drives the real
    heartbeat through the tree runner's finalization order, polls only
    where a poll can land, waits past the 120 s step bound, and finishes.
    A relaxation held to the announcing record alone fails here with
    "status moved backward from finalizing:trajectory-digest-d01".
    """

    from gpuwm.runtime import _finalizing_progress

    watchdog, heartbeat, clock, outdir = _watchdog(tmp_path)
    _step_to(heartbeat, watchdog, clock, 3)
    for phase in ("synchronize-device", "microphysics-transition-receipt",
                  "final-health-d01"):
        _finalizing_progress(heartbeat, phase)
        assert watchdog.check(os.getpid()) is None
        clock.now += 1.0
    _finalizing_progress(heartbeat, "trajectory-digest-d01",
                         work_bytes=1 << 20)
    assert watchdog.check(os.getpid()) is None
    clock.now += 1.0
    # No poll lands between the announcement and the wait's first record.
    _finalizing_progress(heartbeat, "bind-prepared-seal")
    waits = _seam_waits(heartbeat, outdir, clock=clock)
    for waited in range(0, 201, 5):
        waits({"reason": "the preparation is not sealed yet",
               "interval": None, "waited_seconds": float(waited),
               "cause": None})
        assert supervisor.read_heartbeat(
            outdir / supervisor.HEARTBEAT_NAME).status == "waiting:preparation"
        assert watchdog.check(os.getpid()) is None, waited
        clock.now += 5.0
    waits(None)
    assert supervisor.read_heartbeat(
        outdir / supervisor.HEARTBEAT_NAME).status == (
            "finalizing:bind-prepared-seal")
    assert watchdog.check(os.getpid()) is None
    for phase in ("verify-inputs", "write-receipts"):
        _finalizing_progress(heartbeat, phase)
        assert watchdog.check(os.getpid()) is None
    heartbeat.complete(40.0)
    assert watchdog.check(os.getpid()) is None


# -- the cause, from the producer heartbeat --------------------------------


def test_the_cause_follows_the_producer_heartbeat(tmp_path):
    """A source wait closes and a preparation wait opens as the lead posts."""

    clock = _Clock()
    emitted = []
    heartbeat = supervisor.RuntimeHeartbeat(
        tmp_path / supervisor.HEARTBEAT_NAME, run_id="r",
        config_sha256="d", started_at_utc="s")
    heartbeat(model_elapsed_seconds=50.0, outer_step=5,
              last_durable_wrfout=None, last_checkpoint=None)
    producer = tmp_path / bs.PRODUCER_NAME
    producer.write_text(json.dumps({"arrived": {
        "source": "gefs", "cycle": "2026-09-30T12", "lead": 3,
        "first_seen_at": "2026-09-30T15:43:31Z"}}), encoding="utf-8")
    waits = _seam_waits(heartbeat, tmp_path, clock=clock, emitted=emitted,
                        producer=producer)
    cause = _source_cause()
    for _ in range(14):     # 65 s on the source
        waits({"reason": "gefs f003 is not posted yet", "interval": 1,
               "waited_seconds": 0.0, "cause": cause})
        clock.now += 5.0
    progress = json.loads((tmp_path / "progress.json").read_text())
    block = progress["waiting"]
    assert block["on"] == "source" and block["lead"] == 3
    assert set(block) == {"reason", "waited_seconds", "on", "phase",
                          "since_utc", "source", "cycle", "lead",
                          "valid_time", "expected_at", "late_at", "interval"}
    # Then the lead posts and the build runs: one record closes, the other
    # opens.
    waits({"reason": "boundary interval 1 is not prepared yet",
           "interval": 1, "waited_seconds": 70.0, "cause": None})
    assert supervisor.read_heartbeat(
        tmp_path / supervisor.HEARTBEAT_NAME).status == "waiting:preparation"
    clock.now += 3.0
    waits(None)
    tags = [record["event"] for record in emitted]
    assert tags == ["source_wait_started", "source_wait_progress",
                    "source_wait_finished", "boundary_wait_started",
                    "boundary_wait_finished"]
    finished = emitted[2]
    assert finished["first_seen_at"] == "2026-09-30T15:43:31Z"
    assert finished["waited_seconds"] == pytest.approx(70.0)
    assert emitted[4]["cause"] == "preparation"
    assert emitted[4]["seconds"] == pytest.approx(3.0)
    assert "waiting" not in json.loads(
        (tmp_path / "progress.json").read_text())
    assert supervisor.read_heartbeat(
        tmp_path / supervisor.HEARTBEAT_NAME).status == "integrating"
    # Every event carries its declared fields, and the wait log has them all.
    from gpuwm.runplan import EVENT_TAGS, POSTING_EVENT_FIELDS

    for record in emitted:
        assert record["event"] in EVENT_TAGS
        assert set(POSTING_EVENT_FIELDS[record["event"]]) <= set(record)
    logged = [json.loads(line) for line in
              (tmp_path / bs.WAIT_LOG_NAME).read_text().splitlines()]
    assert [record["event"] for record in logged] == tags


def test_wait_cause_names_only_a_lead_the_interval_needs():
    beat = {"state": "waiting_for_source", "waiting_for": {
        "source": "gfs", "cycle": "2026-09-30T12", "lead": 6,
        "valid_time": "2026-09-30T18:00:00Z", "expected_at": "e",
        "late_at": "l", "since_utc": "s"}}
    needs_18z = datetime(2026, 9, 30, 18, tzinfo=timezone.utc)
    needs_15z = datetime(2026, 9, 30, 15, tzinfo=timezone.utc)
    assert bs.wait_cause(beat, needed_valid_time=needs_18z)["lead"] == 6
    assert bs.wait_cause(beat, needed_valid_time=needs_15z) is None
    assert bs.wait_cause({"state": "producing"}) is None


def _head(root: Path) -> dict:
    stream = bs.stream_dir(root)
    (stream / bs.SEGMENTS_DIRNAME).mkdir(parents=True)
    head = {"schema": bs.HEAD_SCHEMA, "basis": {"cache": {
        "directory": "prepared-cache",
        "lbc": {"schedule": [[0, 3600], [3600, 7200]], "fields": ["u"],
                "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4}}},
        "created_utc": _iso(datetime.now(timezone.utc)), "decision": {}}
    head["head_sha256"] = bs.head_sha256(head)
    (stream / bs.HEAD_NAME).write_text(json.dumps(head), encoding="utf-8")
    return head


def test_the_consumer_reports_the_producers_source_wait(tmp_path):
    root = tmp_path / "prepared"
    head = _head(root)
    writer_beat = {"pid": -1, "host": "elsewhere", "updated_epoch": 0,
                   "state": "waiting_for_source", "waiting_for": {
                       "source": "gfs", "cycle": "2026-09-30T12", "lead": 2,
                       "valid_time": "2026-09-30T14:00:00Z",
                       "expected_at": "2026-09-30T15:40:00Z",
                       "late_at": "2026-09-30T16:40:00Z",
                       "since_utc": "2026-09-30T15:40:00Z"}}
    (bs.stream_dir(root) / bs.PRODUCER_NAME).write_text(
        json.dumps(writer_beat), encoding="utf-8")
    intervals = bs.StreamedIntervals(
        root, head=head, start_time=datetime(2026, 9, 30, 12))
    report = intervals._wait_report(1, 12.0)
    assert report["cause"]["lead"] == 2
    assert "gfs f002 is not posted yet" in report["reason"]
    assert "16:40Z" in report["reason"]
    # Interval 0 ends at 13Z: a producer waiting on the 14Z lead is not
    # why interval 0 waits.
    assert intervals._wait_report(0, 1.0)["cause"] is None


def test_a_late_lead_reaches_the_forecast_as_source_behind(tmp_path):
    """The producer's failure names the lead; the consumer raises exit 75."""

    root = tmp_path / "prepared"
    head = _head(root)
    details = {"source": "gefs", "cycle": "2026-09-30T12", "lead": 30,
               "valid_time": "2026-10-01T18:00:00Z",
               "expected_at": "2026-09-30T15:53:00Z",
               "late_at": "2026-09-30T16:53:00Z", "late_after_minutes": 60,
               "last_answer": "not_posted"}
    writer = bs.PreparedTreeWriter.__new__(bs.PreparedTreeWriter)
    writer.published = True
    writer.output_root = root
    writer.staging = root
    writer._heartbeat = None
    writer._record_failure(bs.SourceBehind(details))
    failed = json.loads((bs.stream_dir(root) / bs.FAILED_NAME).read_text())
    assert failed["code"] == "source_behind"
    intervals = bs.StreamedIntervals(root, head=head, poll_seconds=0.01)
    with pytest.raises(bs.SourceBehind) as caught:
        intervals.require(1)
    assert caught.value.exit_code == 75
    assert caught.value.details["lead"] == 30
    text = str(caught.value)
    assert text.startswith("gefs f030 of the 2026-09-30T12 cycle has not "
                           "posted by 16:53Z, 60 min after its scheduled "
                           "time (15:53Z")
    stopped = caught.value.at(model_elapsed_seconds=97200.0,
                              model_valid_time="2026-10-01T15:00:00Z")
    assert "The forecast stopped at 27:00 (valid 2026-10-01T15Z)" in str(
        stopped)
    unheard = bs.SourceBehind({**details, "last_answer": "not_heard"})
    assert "could not be heard from" in str(unheard)
    assert "has not posted" not in str(unheard)


def test_the_producer_heartbeat_names_the_lead_it_waits_on(tmp_path):
    writer = bs.PreparedTreeWriter.__new__(bs.PreparedTreeWriter)
    writer._state, writer._waiting_for, writer._arrived = "producing", None, None
    writer._build_seconds, writer._segments_written = [], 0
    writer._heartbeat = None
    beat = bs._Heartbeat(tmp_path, writer)
    writer._heartbeat = beat
    writer.waiting_for_source({"source": "gfs", "cycle": "2026-09-30T12",
                               "lead": 4, "valid_time": "v",
                               "expected_at": "e", "late_at": "l"})
    record = json.loads((tmp_path / bs.PRODUCER_NAME).read_text())
    assert record["state"] == "waiting_for_source"
    assert record["waiting_for"]["lead"] == 4
    assert record["waiting_for"]["since_utc"]
    writer.source_arrived(first_seen_at="2026-09-30T15:43:31Z")
    record = json.loads((tmp_path / bs.PRODUCER_NAME).read_text())
    assert record["state"] == "producing" and "waiting_for" not in record
    assert record["arrived"] == {"source": "gfs", "cycle": "2026-09-30T12",
                                 "lead": 4,
                                 "first_seen_at": "2026-09-30T15:43:31Z"}


# -- the runner: the wait between two steps, and the stop at the seam -------


_DETAILS = {"source": "gefs", "cycle": "2026-09-30T12", "lead": 30,
            "valid_time": "2026-10-01T18:00:00Z",
            "expected_at": "2026-09-30T15:53:00Z",
            "late_at": "2026-09-30T16:53:00Z", "late_after_minutes": 60,
            "last_answer": "not_posted"}


class _LateIntervals:
    bounds = ((0.0, 3600.0), (3600.0, 7200.0))

    def __init__(self):
        self.asked = []

    def __getitem__(self, index):
        self.asked.append(index)
        if index == 1:
            raise bs.SourceBehind(_DETAILS)
        return object()


class _Writers:
    drained = 0

    def drain(self):
        self.drained += 1


def test_the_runner_waits_between_steps_and_stops_at_the_seam():
    from types import SimpleNamespace

    from gpuwm.prepared_single_domain_forecast import _require_next_interval

    intervals, writers = _LateIntervals(), _Writers()
    exp = SimpleNamespace(start_time=datetime(2026, 9, 30, 12))
    # Mid-interval: the interval the next step reads is already there.
    _require_next_interval(intervals, 1800.0, writers=writers, exp=exp)
    assert intervals.asked == [0] and writers.drained == 0
    # At the seam, the next step needs interval 1 and its lead is late.
    with pytest.raises(bs.SourceBehind) as caught:
        _require_next_interval(intervals, 3600.0, writers=writers, exp=exp)
    assert intervals.asked == [0, 1]
    assert writers.drained == 1          # the frames through the seam land
    assert caught.value.at_seam is True
    assert caught.value.details["model_elapsed_seconds"] == 3600.0
    assert caught.value.details["model_valid_time"] == "2026-09-30T13:00:00Z"


def _seam_stop(error, *, ticks, checkpoint_ticks=None, checkpoints=None):
    from types import SimpleNamespace

    from gpuwm.prepared_single_domain_forecast import _stop_at_seam

    written = []
    checkpoints = [] if checkpoints is None else checkpoints

    def restart_handler(tree, at):
        written.append(at)
        checkpoints.append(f"/run/restart-{at}")

    node = SimpleNamespace(clock=SimpleNamespace(ticks=ticks))
    schedule = SimpleNamespace(clock=SimpleNamespace(tick_den=1))
    abandoned = []
    waits = SimpleNamespace(abandon=lambda: abandoned.append(True))
    stopped = _stop_at_seam(
        error, model=object(), node=node,
        exp=SimpleNamespace(start_time=datetime(2026, 9, 30, 12)),
        schedule=schedule, restart_handler=restart_handler,
        checkpoint_ticks=checkpoint_ticks or {}, checkpoints=checkpoints,
        seam_waits=waits)
    assert abandoned == [True]
    return stopped, written


def test_a_stop_at_a_clean_seam_writes_one_checkpoint_there():
    error = bs.SourceBehind(_DETAILS).at(model_elapsed_seconds=3600.0,
                                         model_valid_time="t")
    error.at_seam = True
    stopped, written = _seam_stop(error, ticks=3600)
    assert written == [3600]
    assert stopped.details["checkpoint"] == "/run/restart-3600"
    # An instant the hourly checkpoint already wrote is not written twice.
    stopped, written = _seam_stop(error, ticks=3600,
                                  checkpoint_ticks={3600: "/run/hourly"},
                                  checkpoints=["/run/hourly"])
    assert written == [] and stopped.details["checkpoint"] == "/run/hourly"


def test_a_stop_caught_mid_step_keeps_the_hourly_checkpoint_and_writes_none():
    stopped, written = _seam_stop(bs.SourceBehind(_DETAILS), ticks=5400,
                                  checkpoints=["/run/hourly"])
    assert written == []
    assert stopped.details["checkpoint"] == "/run/hourly"
    assert stopped.details["model_elapsed_seconds"] == 5400.0
    # Before any step there is no model time to report.
    stopped, written = _seam_stop(bs.SourceBehind(_DETAILS), ticks=0)
    assert written == [] and stopped.details.get("model_elapsed_seconds") is None
