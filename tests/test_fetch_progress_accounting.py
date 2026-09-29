"""What a parallel fetch says about its own progress, and when it stops.

Three defects from one user's 44-file HRRR fetch, pinned here:

* the "N of M files done" line counted only the files that had STARTED,
  so its denominator grew as the fetch ran (0 of 6, 6 of 12, 16 of 22,
  35 of 38), and a failed file was counted as done;
* every in-flight ``fetch_progress`` event said 0 bytes, because the
  Rust backbone holds an object in memory until it is whole and nothing
  on disk grows while it moves;
* a failure on the last-submitted file surfaced only after every
  earlier file had landed (190 s and about 3 GB later), because the
  pool read results in submission order and let in-flight transfers
  run on.

A fourth came with the fix for the third: a Python transport between
attempts slept out its backoff, the server's Retry-After or the NOMADS
cooldown before it looked at the stop signal, so the refusal waited for
it.

Pure CPU: the backbone is a stand-in script, and nothing touches the
network.
"""
from __future__ import annotations

from datetime import datetime
import io
import sys
import threading
import time
from pathlib import Path

import pytest

from gpuwm import fetch_endpoints, fetch_pool, progress, rustwx_fetch


class _Stream(io.StringIO):
    """A stderr stand-in that is not a terminal and takes whole writes."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def isatty(self) -> bool:
        return False

    def write(self, text):
        with self._lock:
            return super().write(text)


def _monitor(**kwargs):
    stream = _Stream()
    events: list[tuple[str, dict]] = []
    monitor = progress.TransferMonitor(
        "fetch hrrr", stream=stream,
        events=lambda event, **fields: events.append((event, fields)),
        ticker=False, **kwargs)
    return monitor, stream, events


def _lines(stream, needle="files done"):
    return [line for line in stream.getvalue().splitlines() if needle in line]


def _wait_until(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _whole(events):
    return [fields["acquisition"] for event, fields in events
            if event == "fetch_progress" and "acquisition" in fields]


# ---------------------------------------------------------------------------
# The denominator is the whole request, from the first line
# ---------------------------------------------------------------------------

def test_the_first_line_counts_every_queued_file_not_only_the_started_ones():
    """38 queued, 6 in flight: the first line says 0 of 38, not 0 of 6."""

    monitor, stream, events = _monitor(interval=0.0)
    release = threading.Event()
    started: list[int] = []
    lock = threading.Lock()

    def action(index):
        def run():
            with lock:
                started.append(index)
            if index == 0:
                # Speak once every worker has a file in flight: the
                # moment the old line read "0 of 6".
                assert _wait_until(lambda: len(started) >= 6)
                monitor.tick(force=True)
                release.set()
            elif index < 6:
                release.wait(timeout=20.0)
            return {"name": f"f{index:02d}", "bytes": 1}
        return run

    jobs = [fetch_pool.TransferJob(name=f"f{index:02d}.grib2", url=None,
                                   action=action(index))
            for index in range(38)]
    entries, _receipt = fetch_pool.run_transfers(jobs, workers=6,
                                                 monitor=monitor)
    monitor.close()
    assert len(entries) == 38
    lines = _lines(stream)
    assert lines[0].startswith("fetch hrrr: 0 of 38 files done, "), lines[0]
    assert all(" of 38 files done" in line for line in lines)
    whole = _whole(events)
    assert whole[0]["files_total"] == 38
    assert whole[-1] == {"schema": "arwen.acquisition-progress.v1",
                         "files_total": 38, "files_completed": 38,
                         "transferred_bytes": 38}


def test_sizes_known_up_front_make_the_expected_total_from_the_start():
    monitor, stream, _events = _monitor(interval=0.0)
    monitor.begin(files=[("a.grib2", 100), ("b.grib2", 300)])
    monitor.tick(force=True)
    assert _lines(stream)[-1].startswith(
        "fetch hrrr: 0 of 2 files done, 0 B of 400 B, ")


def test_an_unknown_size_leaves_the_expected_total_unstated():
    monitor, stream, _events = _monitor(interval=0.0)
    monitor.begin(files=[("a.grib2", 100), ("b.grib2", None)])
    monitor.start("a.grib2")
    monitor.tick(force=True)
    tail = _lines(stream)[-1].split("files done, ", 1)[1]
    assert " of " not in tail


# ---------------------------------------------------------------------------
# A failed file is failed, not done
# ---------------------------------------------------------------------------

def test_a_failed_file_is_not_counted_as_done():
    monitor, stream, _events = _monitor(interval=0.0)
    for name in ("a", "b", "c"):
        monitor.start(name, host="aws")
    monitor.finish("a", size=10, seconds=1.0)
    monitor.finish("b", seconds=1.0, failed=True)
    monitor.tick(force=True)
    line = _lines(stream)[-1]
    assert line.startswith("fetch hrrr: 1 of 3 files done, 1 failed, "), line


def test_the_whole_request_count_leaves_a_failed_file_out_of_completed():
    monitor, _stream, events = _monitor(interval=0.0)
    monitor.begin(files=[(name, None) for name in ("a", "b", "c")])
    for name in ("a", "b", "c"):
        monitor.start(name, host="aws")
    monitor.finish("a", size=10, seconds=1.0)
    monitor.finish("b", seconds=1.0, failed=True)
    monitor.tick(force=True)
    assert _whole(events)[-1]["files_completed"] == 1
    assert _whole(events)[-1]["files_total"] == 3


def test_a_file_stopped_because_another_failed_is_neither_done_nor_failed():
    monitor, stream, events = _monitor(interval=0.0)
    monitor.begin(files=[("a", None), ("b", None)])
    monitor.start("a")
    monitor.start("b")
    monitor.finish("a", seconds=1.0, failed=True)
    monitor.finish("b", seconds=1.0, failed=True, cancelled=True)
    monitor.tick(force=True)
    assert _lines(stream)[-1].startswith(
        "fetch hrrr: 0 of 2 files done, 1 failed, ")
    completed = [fields for event, fields in events
                 if event == "fetch_completed"]
    assert completed[0]["failed"] is True and "cancelled" not in completed[0]
    assert completed[1]["failed"] is True and completed[1]["cancelled"] is True


def test_a_monitor_driven_without_a_declared_request_keeps_its_old_stream():
    """ERA5 drives the monitor by hand and publishes its own count."""

    monitor, _stream, events = _monitor(interval=0.0)
    monitor.start("a.grib2")
    monitor.tick(force=True)
    monitor.finish("a.grib2", size=4, seconds=1.0)
    monitor.close()
    assert [event for event, _fields in events] == [
        "fetch_started", "fetch_progress", "fetch_completed"]
    assert _whole(events) == []


# ---------------------------------------------------------------------------
# Bytes from the backbone's own progress output
# ---------------------------------------------------------------------------

def _backbone(tmp_path: Path, name: str, *, stderr: str, hold_s: float,
              marker: Path | None = None, until: Path | None = None,
              record: dict | None = None, code: int = 0) -> list[str]:
    """A stand-in rw_fetch: draws ``stderr``, holds, then answers.

    It holds for ``hold_s`` seconds or until ``until`` exists, then
    writes ``marker`` (proof it was NOT stopped) and prints ``record``.
    """

    script = tmp_path / f"{name}.py"
    script.write_text(
        "import json, pathlib, sys, time\n"
        f"sys.stderr.write({stderr!r})\n"
        "sys.stderr.flush()\n"
        f"deadline = time.monotonic() + {hold_s!r}\n"
        f"until = {str(until) if until else None!r}\n"
        "while time.monotonic() < deadline:\n"
        "    if until and pathlib.Path(until).exists():\n"
        "        break\n"
        "    time.sleep(0.02)\n"
        f"marker = {str(marker) if marker else None!r}\n"
        "if marker:\n"
        "    pathlib.Path(marker).write_text('ran to the end')\n"
        f"print(json.dumps({record or {}!r}))\n"
        f"sys.exit({code})\n", encoding="utf-8")
    return [sys.executable, str(script)]


def test_an_object_held_in_the_backbones_memory_reports_its_bytes(tmp_path):
    """All 207 in-flight events of the user's fetch said 0 bytes."""

    monitor, stream, events = _monitor(interval=0.0)
    name = "hrrr.t06z.wrfnatf01.grib2"
    go = tmp_path / "go"
    command = _backbone(
        tmp_path, "rw_fetch_stub",
        stderr="\nrw_fetch-progress f001 16777216 50331648\n",
        hold_s=30.0, until=go, record={"schema": "x"})

    def action():
        record = rustwx_fetch._run(command, what="fetch",
                                   on_progress=monitor.relay(name))
        return {**record, "bytes": 50331648}

    def moving() -> bool:
        record = monitor._files.get(name)
        return record is not None and record.moved() > 0

    result: dict = {}

    def pool():
        result["entries"] = fetch_pool.run_transfers(
            [fetch_pool.TransferJob(name=name, url=None, action=action,
                                    path=tmp_path / name)],
            workers=1, monitor=monitor)[0]

    runner = threading.Thread(target=pool)
    runner.start()
    try:
        assert _wait_until(moving), (
            "the backbone's byte count never reached the monitor")
        monitor.tick(force=True)
    finally:
        go.write_text("go")
        runner.join(timeout=30.0)
    in_flight = [fields for event, fields in events
                 if event == "fetch_progress" and "file" in fields]
    assert in_flight[-1]["bytes"] == 16777216
    assert in_flight[-1]["expected_bytes"] == 50331648
    assert "16.0 MiB of 48.0 MiB" in _lines(stream)[-1]
    assert _whole(events)[-1]["transferred_bytes"] == 16777216
    assert result["entries"][0]["schema"] == "x"


# ---------------------------------------------------------------------------
# The first failure ends the request at once
# ---------------------------------------------------------------------------

def test_a_late_failure_stops_the_backbones_in_flight_instead_of_waiting(
        tmp_path):
    """Job 9 fails at once while jobs 0 to 8 would each take 10 s.

    The pool used to wait out all nine before saying so.
    """

    admitted: list[int] = []
    markers = [tmp_path / f"ran-{index}" for index in range(9)]
    failed_at: list[float] = []

    def slow(index):
        command = _backbone(
            tmp_path, f"slow_{index}",
            stderr="\nrw_fetch-progress f000 16777216 503316480\n",
            hold_s=10.0, marker=markers[index], record={"schema": "x"})
        return lambda: rustwx_fetch._run(command, what="fetch")

    def fails_at_once():
        # Once the nine are under way: the shape of the user's fetch,
        # where the last-submitted file failed with the others in flight.
        _wait_until(lambda: stream.getvalue().count(" starting") >= 10,
                    timeout=10.0)
        time.sleep(0.2)
        failed_at.append(time.monotonic())
        raise ValueError("downloaded hrrr.t06z.soilf18.grib2 carries 0 "
                         "GRIB2 messages")

    monitor, stream, events = _monitor()
    jobs = [fetch_pool.TransferJob(name=f"f{index:02d}.grib2", url=None,
                                   action=slow(index))
            for index in range(9)]
    jobs.append(fetch_pool.TransferJob(name="hrrr.t06z.soilf18.grib2",
                                       url=None, action=fails_at_once))
    with pytest.raises(ValueError, match="soilf18.grib2 carries 0"):
        fetch_pool.run_transfers(
            jobs, workers=10, monitor=monitor,
            on_admitted=lambda index, entry: admitted.append(index))
    reported = time.monotonic() - failed_at[0]
    assert reported < 2.0, f"the pool took {reported:.1f} s to report"
    assert not any(marker.exists() for marker in markers), (
        "a backbone the pool should have stopped ran to the end")
    assert admitted == []
    assert ("fetch hrrr: hrrr.t06z.soilf18.grib2 failed, so the request "
            "cannot complete; stopping the 9 files in flight"
            ) in stream.getvalue()
    completed = {fields["file"]: fields for event, fields in events
                 if event == "fetch_completed"}
    assert completed["hrrr.t06z.soilf18.grib2"].get("cancelled") is None
    assert all(completed[f"f{index:02d}.grib2"]["cancelled"] is True
               for index in range(9))


def test_a_transport_that_checks_the_stop_signal_gives_up_at_its_next_chunk():
    ran_to_the_end: list[int] = []

    def cooperative(index):
        def run():
            for _chunk in range(200):          # 10 s of 50 ms chunks
                fetch_pool.raise_if_stopped()
                time.sleep(0.05)
            ran_to_the_end.append(index)
            return {"name": index, "bytes": 1}
        return run

    def fails_at_once():
        time.sleep(0.1)
        raise RuntimeError("NOMADS did not serve f018")

    jobs = [fetch_pool.TransferJob(name=f"j{index}", url=None,
                                   action=cooperative(index))
            for index in range(9)]
    jobs.append(fetch_pool.TransferJob(name="j9", url=None,
                                       action=fails_at_once))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not serve f018"):
        fetch_pool.run_transfers(jobs, workers=10)
    assert time.monotonic() - started < 5.0
    assert ran_to_the_end == []


def test_queued_files_never_start_once_the_request_has_failed():
    ran: list[str] = []
    gate = threading.Event()

    def ok(name):
        def run():
            ran.append(name)
            return {"name": name, "bytes": 1}
        return run

    def fails():
        gate.wait(timeout=10.0)
        raise RuntimeError("f003 refused")

    def holds():
        gate.set()
        for _chunk in range(200):
            fetch_pool.raise_if_stopped()
            time.sleep(0.05)
        return {"name": "held", "bytes": 1}

    monitor, stream, _events = _monitor()
    jobs = [fetch_pool.TransferJob(name="f003", url=None, action=fails),
            fetch_pool.TransferJob(name="held", url=None, action=holds)]
    jobs += [fetch_pool.TransferJob(name=f"q{index}", url=None,
                                    action=ok(f"q{index}"))
             for index in range(5)]
    with pytest.raises(RuntimeError, match="f003 refused"):
        fetch_pool.run_transfers(jobs, workers=2, monitor=monitor)
    assert ran == []
    said = stream.getvalue()
    assert ("f003 failed, so the request cannot complete; stopping the 1 "
            "file in flight and not starting the 5 files still queued"
            ) in said
    # A file that never started was never announced either.
    assert "q0 starting" not in said


def test_the_verified_prefix_is_still_admitted_before_the_refusal():
    admitted: list[str] = []

    def ok(name):
        return lambda: {"name": name, "bytes": 1}

    def cooperative():
        for _chunk in range(200):
            fetch_pool.raise_if_stopped()
            time.sleep(0.05)
        return {"name": "slow", "bytes": 1}

    def fails():
        time.sleep(0.2)
        raise ValueError("f009 carries 3 GRIB2 messages")

    jobs = [fetch_pool.TransferJob(name="f000", url=None, action=ok("f000")),
            fetch_pool.TransferJob(name="f003", url=None, action=ok("f003")),
            fetch_pool.TransferJob(name="f006", url=None, action=cooperative),
            fetch_pool.TransferJob(name="f009", url=None, action=fails)]
    with pytest.raises(ValueError, match="f009 carries 3"):
        fetch_pool.run_transfers(
            jobs, workers=4,
            on_admitted=lambda index, entry: admitted.append(entry["name"]))
    assert admitted == ["f000", "f003"]


def test_the_failure_is_raised_even_when_a_stopped_file_settles_first():
    """A file stopped BY the failure must never be what the caller hears.

    The failing worker fires the stop before its own result is settled,
    so a sibling that notices the stop can finish first.
    """

    class SlowToFinish:
        def begin(self, files=()):
            pass

        def start(self, name, **_kwargs):
            pass

        def stopped(self, name, **_kwargs):
            pass

        def finish(self, name, **kwargs):
            if name == "fails" and not kwargs.get("cancelled"):
                time.sleep(0.5)

    def sibling():
        for _chunk in range(400):
            fetch_pool.raise_if_stopped()
            time.sleep(0.01)
        return {"name": "sibling", "bytes": 1}

    def fails():
        time.sleep(0.1)
        raise ValueError("f012 does not verify")

    jobs = [fetch_pool.TransferJob(name="sibling", url=None, action=sibling),
            fetch_pool.TransferJob(name="fails", url=None, action=fails)]
    with pytest.raises(ValueError, match="f012 does not verify"):
        fetch_pool.run_transfers(jobs, workers=2, monitor=SlowToFinish())


def test_a_stopped_backbone_is_not_retried_or_walked_past(tmp_path,
                                                          monkeypatch):
    """The stop is not a network fault: no retry, no next host."""

    from gpuwm import fetch

    stopped = fetch_pool.TransferCancelled("f00: stopped")
    assert fetch_endpoints.fault_reason(stopped) is None
    calls: list[int] = []

    def cancelled(binary, **kwargs):
        calls.append(1)
        raise stopped

    monkeypatch.setattr(rustwx_fetch, "run_fetch", cancelled)
    with pytest.raises(fetch_pool.TransferCancelled):
        fetch._rw_fetch_hrrr(
            binary=Path("rw_fetch"), cycle=datetime(2026, 9, 25, 6), hour=0,
            kind="atmosphere", host="s3", mode="full-file", out=tmp_path,
            cache_dir=None, progress=lambda _line: None, retries=5)
    assert len(calls) == 1


def test_the_backbone_is_not_launched_once_the_request_has_failed(tmp_path):
    marker = tmp_path / "launched"
    command = _backbone(tmp_path, "never", stderr="", hold_s=0.0,
                        marker=marker, record={"schema": "x"})
    stop = fetch_pool._Stop()
    stop.fire()
    fetch_pool._CURRENT.job = fetch_pool.TransferContext("f00", stop)
    try:
        with pytest.raises(fetch_pool.TransferCancelled):
            rustwx_fetch._run(command, what="fetch")
    finally:
        fetch_pool._CURRENT.job = None
    assert not marker.exists()


def test_a_serial_fetch_says_what_it_did_not_start():
    monitor, stream, _events = _monitor()

    def fails():
        raise RuntimeError("f000 refused")

    jobs = [fetch_pool.TransferJob(name="f000", url=None, action=fails)]
    jobs += [fetch_pool.TransferJob(
        name=f"f00{index}", url=None,
        action=lambda: {"name": "x", "bytes": 1}) for index in range(1, 4)]
    with pytest.raises(RuntimeError, match="f000 refused"):
        fetch_pool.run_transfers(jobs, workers=1, monitor=monitor)
    assert ("f000 failed, so the request cannot complete; not starting the "
            "3 files still queued") in stream.getvalue()


def test_the_gfs_full_file_route_stops_its_progress_ticker(tmp_path,
                                                           monkeypatch):
    """The route never closed its monitor, so the ticker thread kept
    repeating the last files line for the rest of the process."""

    from gpuwm import fetch
    from tools import download_gfs_native_subset as gfs_transport

    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")

    def download(url, destination, **kwargs):
        destination.write_bytes(one * 696)

    monkeypatch.setattr(gfs_transport, "_download", download)
    monkeypatch.setattr(fetch, "gfs_live_index", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "_gfs_index_record_count",
                        lambda url, **kwargs: 696)
    made: list[progress.TransferMonitor] = []

    class Recorded(progress.TransferMonitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(fetch.progress_mod, "TransferMonitor", Recorded)
    fetch.fetch_gfs_fullfile(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3), area=None,
        out=tmp_path / "gfs-full", progress=lambda line: None)
    assert made, "the route made no monitor"
    assert all(monitor._stop.is_set() and monitor._thread is None
               for monitor in made)


def test_an_interrupt_inside_a_transfer_stops_the_rest_and_propagates():
    """Ctrl-C in one file is still an interrupt, and it stops the others."""

    admitted: list[str] = []
    ran_to_the_end: list[str] = []

    def ok():
        return {"name": "f000", "bytes": 1}

    def cooperative():
        for _chunk in range(200):
            fetch_pool.raise_if_stopped()
            time.sleep(0.05)
        ran_to_the_end.append("f006")
        return {"name": "f006", "bytes": 1}

    def interrupted():
        time.sleep(0.2)
        raise KeyboardInterrupt

    jobs = [fetch_pool.TransferJob(name="f000", url=None, action=ok),
            fetch_pool.TransferJob(name="f003", url=None, action=interrupted),
            fetch_pool.TransferJob(name="f006", url=None, action=cooperative)]
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        fetch_pool.run_transfers(
            jobs, workers=3,
            on_admitted=lambda index, entry: admitted.append(entry["name"]))
    assert time.monotonic() - started < 5.0
    assert admitted == ["f000"]
    assert ran_to_the_end == []


# -- A stopped file does not wait out its backoff ---------------------------
#
# The stop used to be looked at only once a wait was over: a file between
# attempts slept out its backoff, the server's Retry-After or the NOMADS
# cooldown first, and the refusal waited for it.


def _stopped_in(seconds: float) -> fetch_pool.TransferContext:
    """A pooled file whose request another file fails ``seconds`` from now."""

    stop = fetch_pool._Stop()
    timer = threading.Timer(seconds, stop.fire)
    timer.daemon = True
    timer.start()
    return fetch_pool.TransferContext("f018", stop)


def test_a_pooled_ladder_retry_is_not_waited_out_once_another_file_failed(
        tmp_path):
    from dataclasses import replace
    import re
    from urllib.error import HTTPError

    from gpuwm import fetch_routes

    plan = fetch_routes.resolve_request(
        "ifs", cycle=datetime(2026, 9, 12), hours=12)
    endpoints = tuple(replace(endpoint, name=f"endpoint-{i}",
                              base=f"https://endpoint-{i}.invalid")
                      for i, endpoint in enumerate(plan.route.hosts[:2]))
    objects = plan.objects[:2]
    plan = replace(plan, host=endpoints[0], ladder=endpoints, objects=objects,
                   primary_files=tuple(obj.relpath for obj in objects))
    busy: list[str] = []

    def download(url, dest, *, magic, opener=None):
        if dest.name == objects[0].name:
            # Every endpoint asks for 30 s: the round's wait before its
            # next attempt.
            busy.append(url)
            raise HTTPError(url, 503, "busy", {"Retry-After": "30"}, None)
        time.sleep(0.3)
        raise HTTPError(url, 404, "not available", {}, None)

    started = time.monotonic()
    with pytest.raises(ValueError, match=re.escape(objects[1].relpath)):
        fetch_routes.run_plan(plan, out=tmp_path, downloader=download,
                              file_workers=2, probe=lambda _: False,
                              progress=lambda _: None)
    assert time.monotonic() - started < 10.0, "the 30 s wait was served"
    assert len(busy) == 2, "a second round was asked for"


def test_a_pooled_gfs_retry_is_not_waited_out_once_another_file_failed(
        tmp_path, monkeypatch):
    from urllib.error import HTTPError

    from tools import download_gfs_native_subset as transport

    asked: list[str] = []

    def busy(request, **kwargs):
        asked.append(request.full_url)
        raise HTTPError(request.full_url, 503, "busy",
                        {"Retry-After": "20"}, None)

    monkeypatch.setattr(transport, "paced_urlopen", busy)
    started = time.monotonic()
    with fetch_pool.working_for(_stopped_in(0.2)), \
            pytest.raises(fetch_pool.TransferCancelled):
        transport._download("https://nomads.ncep.noaa.gov/cgi-bin/x",
                            tmp_path / "gfs.t00z.pgrb2.0p25.f018")
    assert time.monotonic() - started < 5.0, "the 20 s Retry-After was served"
    assert len(asked) == 1


def test_a_pooled_hrrr_range_retry_is_not_waited_out_once_another_file_failed(
        tmp_path, monkeypatch):
    from urllib.error import URLError

    from tools import download_hrrr_native_subset as transport

    asked: list[str] = []

    def reset(request, **kwargs):
        asked.append(request.full_url)
        raise URLError("connection reset by peer")

    monkeypatch.setattr(transport, "paced_urlopen", reset)
    started = time.monotonic()
    with fetch_pool.working_for(_stopped_in(0.2)), \
            pytest.raises(fetch_pool.TransferCancelled):
        transport._download_range(
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/x",
            transport.ByteRange(0, 9, "1:0", "1:0"),
            tmp_path / "range.part", 3)
    assert time.monotonic() - started < 1.5, "the 2 s backoff was served"
    assert len(asked) == 1


def test_a_pooled_request_does_not_serve_out_the_nomads_cooldown(
        tmp_path, monkeypatch):
    from gpuwm import nomads_governor as governor

    state = tmp_path / "rustwx_nomads_rate_limit.state"
    monkeypatch.setenv(governor.STATE_PATH_ENV, str(state))
    monkeypatch.delenv(governor.REQUEST_LOG_ENV, raising=False)
    now = governor._now_ms()
    cooldown_until = now + 15 * 60 * 1000
    assert governor.write_state(state, now, cooldown_until)
    sent: list[object] = []
    started = time.monotonic()
    with fetch_pool.working_for(_stopped_in(0.2)), \
            pytest.raises(fetch_pool.TransferCancelled):
        governor.paced_urlopen(
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/x",
            opener=lambda *args, **kwargs: sent.append(args))
    assert time.monotonic() - started < 5.0, "the cooldown was served"
    # Nothing was sent, and the node-wide record is untouched: giving up
    # is never looser than waiting.
    assert sent == []
    assert governor.read_state(state) == (now, cooldown_until)


def test_a_pooled_file_waiting_for_publication_is_not_waited_out_once_another_file_failed():
    """The live-cycle wait polled every 30 s and never looked at the stop,
    so a failed download waited for its latest hour to be published."""

    from gpuwm import fetch

    probed: list[str] = []

    def unpublished(url):
        probed.append(url)
        return False

    started = time.monotonic()
    with fetch_pool.working_for(_stopped_in(0.2)), \
            pytest.raises(fetch_pool.TransferCancelled):
        fetch._wait_for_hrrr_product(
            cycle=datetime(2026, 9, 27, 0), hour=18, product="wrfprs",
            candidates=("nomads",), probe=unpublished, clock=time.monotonic,
            deadline=time.monotonic() + 45.0, sleeper=time.sleep,
            progress=lambda _line: None, label="f18 soil")
    assert time.monotonic() - started < 5.0, "the 30 s poll was served"
    assert len(probed) == 1


def test_a_callers_own_clock_is_kept_and_the_real_one_waits_on_the_stop():
    naps: list[float] = []
    # Outside the pool, and on the serial transport: the caller's clock.
    fetch_pool.sleep_unless_stopped(3.0, sleep=naps.append)
    serial = fetch_pool.TransferContext("f000", fetch_pool._Stop(),
                                        shared=False)
    fetch_pool.sleep_unless_stopped(4.0, sleep=naps.append, job=serial)
    # A test's clock inside the pool is kept too, with the stop checked
    # on both sides of it.
    stop = fetch_pool._Stop()
    pooled = fetch_pool.TransferContext("f003", stop)
    fetch_pool.sleep_unless_stopped(5.0, sleep=naps.append, job=pooled)
    stop.fire()
    with pytest.raises(fetch_pool.TransferCancelled):
        fetch_pool.sleep_unless_stopped(6.0, sleep=naps.append, job=pooled)
    assert naps == [3.0, 4.0, 5.0]
    # The real clock inside the pool is the wait on the stop signal: all
    # of it while the request is healthy.
    healthy = fetch_pool.TransferContext("f006", fetch_pool._Stop())
    started = time.monotonic()
    fetch_pool.sleep_unless_stopped(0.2, sleep=time.sleep, job=healthy)
    assert time.monotonic() - started >= 0.15
