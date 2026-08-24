"""Per-file visibility while a parallel fetch is in flight.

The regression this pins, reported from a live drive of the Studio front
end: once every fetch went through :mod:`gpuwm.fetch_pool`, files moved
concurrently and each one printed exactly one line -- at COMPLETION.  A
user watched minutes of silence and then a burst of finished lines.  The
serial loop it replaced had said something per file, as it went.

Three surfaces are pinned here, because the fix has to hold at all three:

* a START line per file, on stderr, as its transfer begins;
* a consolidated in-flight line at a steady cadence, rewriting itself on
  a terminal and appended sparsely to a log;
* per-file records on the machine event stream, which is what Studio
  renders.

And one thing that must NOT move: the completion line's text.  A reader
with a parser for it keeps working.
"""
from __future__ import annotations

import io
import threading
import time

import pytest

from gpuwm import fetch_pool, progress


class _Stream(io.StringIO):
    """A stderr stand-in whose tty-ness the test decides."""

    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty
        self._lock = threading.Lock()

    def isatty(self) -> bool:
        return self._tty

    def write(self, text):                       # whole writes, one lock
        with self._lock:
            return super().write(text)


def _monitor(tty=False, **kwargs):
    stream = _Stream(tty)
    events: list[tuple[str, dict]] = []
    monitor = progress.TransferMonitor(
        "fetch hrrr", stream=stream,
        events=lambda event, **fields: events.append((event, fields)),
        ticker=False, **kwargs)
    return monitor, stream, events


# ---------------------------------------------------------------------------
# The start line
# ---------------------------------------------------------------------------

def test_a_start_line_names_the_file_its_host_and_the_expected_size():
    monitor, stream, _events = _monitor()
    monitor.start("hrrr.t02z.wrfnatf01.grib2", token="f01 atmosphere",
                  host="aws", expected_bytes=761_265_000)
    assert stream.getvalue() == (
        "fetch hrrr: f01 atmosphere: hrrr.t02z.wrfnatf01.grib2 starting "
        "(aws, 726.0 MiB expected)\n")


def test_the_start_line_omits_a_size_nobody_measured():
    monitor, stream, _events = _monitor()
    monitor.start("gfs.t00z.pgrb2.0p25.f000", host="nomads")
    assert stream.getvalue() == (
        "fetch hrrr: gfs.t00z.pgrb2.0p25.f000 starting (nomads)\n")


def test_the_start_line_omits_the_parenthetical_entirely_when_it_is_empty():
    monitor, stream, _events = _monitor()
    monitor.start("some.grib2")
    assert stream.getvalue() == "fetch hrrr: some.grib2 starting\n"


def test_a_start_line_is_one_whole_write_so_threads_cannot_interleave():
    monitor, stream, _events = _monitor()
    names = [f"f{index:03d}.grib2" for index in range(24)]
    threads = [threading.Thread(target=monitor.start, args=(name,),
                                kwargs={"host": "aws"}) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = stream.getvalue().splitlines()
    assert len(lines) == len(names)
    assert sorted(lines) == sorted(
        f"fetch hrrr: {name} starting (aws)" for name in names)


# ---------------------------------------------------------------------------
# The consolidated in-flight line
# ---------------------------------------------------------------------------

def test_the_progress_line_counts_files_bytes_and_an_aggregate_rate():
    monitor, stream, _events = _monitor(interval=0.0, clock=_clock([0.0, 4.0]))
    monitor.start("a.grib2", host="aws", expected_bytes=100)
    monitor.start("b.grib2", host="aws", expected_bytes=300)
    monitor.finish("a.grib2", size=100, seconds=1.0, host="aws")
    monitor.observe("b.grib2", 200)
    monitor.tick()
    line = stream.getvalue().splitlines()[-1]
    assert line.startswith("fetch hrrr: 1 of 2 files done, ")
    assert "of 400 B" in line
    assert "aggregate" in line


def test_the_progress_line_states_only_what_moved_when_no_size_is_known():
    monitor, stream, _events = _monitor(interval=0.0, clock=_clock([0.0, 2.0]))
    monitor.start("a.grib2", host="aws")
    monitor.observe("a.grib2", 4096)
    monitor.tick()
    line = stream.getvalue().splitlines()[-1]
    assert line.startswith("fetch hrrr: 0 of 1 files done, 4.0 KiB, ")
    assert " of " not in line.split("files done, ", 1)[1]


def test_a_log_rewrites_nothing_and_speaks_less_often_than_a_terminal():
    assert (progress.TRANSFER_LOG_INTERVAL_S
            > progress.TRANSFER_TTY_INTERVAL_S)
    monitor, stream, _events = _monitor(tty=False)
    assert monitor.interval() == progress.TRANSFER_LOG_INTERVAL_S
    monitor.start("a.grib2", host="aws")
    monitor.observe("a.grib2", 10)
    monitor.tick()
    monitor.observe("a.grib2", 10)
    monitor.tick()          # inside the interval: says nothing
    text = stream.getvalue()
    assert "\r" not in text
    assert text.count("files done") == 1


def test_a_terminal_rewrites_one_line_at_the_faster_cadence():
    monitor, stream, _events = _monitor(tty=True)
    assert monitor.interval() == progress.TRANSFER_TTY_INTERVAL_S
    monitor.start("a.grib2", host="aws")
    monitor.observe("a.grib2", 10)
    monitor.tick()
    text = stream.getvalue()
    assert "\rfetch hrrr: 0 of 1 files done" in text
    assert not text.endswith("\n")


# ---------------------------------------------------------------------------
# Bytes the monitor did not see: the growing file on disk
# ---------------------------------------------------------------------------

def test_bytes_are_read_off_disk_when_the_action_owns_the_copy(tmp_path):
    monitor, stream, _events = _monitor(interval=0.0,
                                        clock=_clock([0.0, 1.0, 2.0]))
    part = tmp_path / "a.grib2.part"
    monitor.start("a.grib2", host="aws", path=tmp_path / "a.grib2")
    part.write_bytes(b"x" * 2048)
    monitor.tick()
    assert "2.0 KiB" in stream.getvalue()
    part.rename(tmp_path / "a.grib2")
    (tmp_path / "a.grib2").write_bytes(b"x" * 5120)
    monitor.tick()
    assert "5.0 KiB" in stream.getvalue()


def test_bytes_are_seen_when_the_temp_name_is_not_derived_from_the_final(
        tmp_path):
    """The staged name owes nothing to the destination name.

    MEASURED, against the real ``rw_fetch.exe``: the backbone stages a
    whole object under a name of its own choosing and moves it into
    place at the end, so neither ``<final>`` nor ``<final>.part`` exists
    while the bytes are moving.  Guessing a second decoration would only
    move the guess; what the destination directory GAINED is a fact.
    """

    monitor, stream, _events = _monitor(interval=0.0,
                                        clock=_clock([0.0, 1.0, 2.0]))
    monitor.start("a.grib2", host="aws", path=tmp_path / "a.grib2")
    staged = tmp_path / "9f1c0b7e-2d44-4a01-bd54-6f0e2a3c8b11.tmp"
    staged.write_bytes(b"x" * 3072)
    monitor.tick()
    assert "3.0 KiB" in stream.getvalue()

    staged.rename(tmp_path / "a.grib2")
    monitor.finish("a.grib2", size=3072, host="aws")
    monitor.tick(force=True)
    assert "1 of 1 files done, 3.0 KiB" in stream.getvalue()


def test_the_directory_gain_is_counted_once_across_concurrent_files(tmp_path):
    """Six transfers into one directory are not six copies of its gain."""

    monitor, stream, _events = _monitor(interval=0.0,
                                        clock=_clock([0.0, 1.0]))
    for name in ("a.grib2", "b.grib2"):
        monitor.start(name, host="aws", path=tmp_path / name)
    (tmp_path / "staging-0.tmp").write_bytes(b"x" * 1024)
    (tmp_path / "staging-1.tmp").write_bytes(b"x" * 1024)
    monitor.tick()
    assert "2.0 KiB" in stream.getvalue()


def test_files_present_before_the_transfer_are_not_counted_as_moved(tmp_path):
    """A reused object already on disk did not move over the network."""

    (tmp_path / "already-here.grib2").write_bytes(b"x" * 8192)
    monitor, stream, _events = _monitor(interval=0.0,
                                        clock=_clock([0.0, 1.0]))
    monitor.start("a.grib2", host="aws", path=tmp_path / "a.grib2")
    monitor.tick()
    assert "0 of 1 files done, 0 B" in stream.getvalue()


# ---------------------------------------------------------------------------
# The machine stream Studio renders
# ---------------------------------------------------------------------------

def test_every_transfer_emits_started_progress_and_completed():
    monitor, _stream, events = _monitor(interval=0.0,
                                        clock=_clock([0.0, 1.0, 2.0]))
    monitor.start("a.grib2", token="f01 atmosphere", host="aws",
                  expected_bytes=400)
    monitor.observe("a.grib2", 200)
    monitor.tick()
    monitor.finish("a.grib2", size=400, seconds=3.5, host="aws")
    assert [event for event, _fields in events] == [
        "fetch_started", "fetch_progress", "fetch_completed"]

    started = events[0][1]
    assert started["file"] == "a.grib2"
    assert started["host"] == "aws"
    assert started["expected_bytes"] == 400
    assert started["token"] == "f01 atmosphere"

    moving = events[1][1]
    assert moving["file"] == "a.grib2"
    assert moving["bytes"] == 200
    assert moving["expected_bytes"] == 400

    done = events[2][1]
    assert done["file"] == "a.grib2"
    assert done["bytes"] == 400
    assert done["seconds"] == 3.5
    assert done["host"] == "aws"


def test_a_progress_record_is_emitted_for_each_file_still_in_flight():
    monitor, _stream, events = _monitor(interval=0.0,
                                        clock=_clock([0.0, 1.0]))
    monitor.start("a.grib2", host="aws")
    monitor.start("b.grib2", host="aws")
    monitor.finish("a.grib2", size=10, seconds=0.5, host="aws")
    monitor.observe("b.grib2", 20)
    monitor.tick()
    moving = [fields for event, fields in events if event == "fetch_progress"]
    assert [fields["file"] for fields in moving] == ["b.grib2"]


def test_the_event_names_are_published_for_a_consumer_to_switch_on():
    assert progress.TRANSFER_EVENTS == (
        "fetch_started", "fetch_progress", "fetch_completed")


def test_the_ambient_sink_carries_events_to_a_run_event_stream():
    seen: list[tuple[str, dict]] = []
    stream = _Stream(False)
    with progress.event_sink(lambda event, **fields: seen.append(
            (event, fields))):
        monitor = progress.TransferMonitor("fetch hrrr", stream=stream,
                                           ticker=False)
        monitor.start("a.grib2", host="aws")
        monitor.finish("a.grib2", size=10, seconds=0.5, host="aws")
    assert [event for event, _fields in seen] == [
        "fetch_started", "fetch_completed"]
    # And nothing leaks past the block.
    progress.TransferMonitor("fetch hrrr", stream=stream,
                             ticker=False).start("b.grib2")
    assert len(seen) == 2


def test_a_sink_that_raises_never_fails_the_transfer():
    stream = _Stream(False)

    def angry(_event, **_fields):
        raise RuntimeError("the consumer went away")

    monitor = progress.TransferMonitor("fetch hrrr", stream=stream,
                                       events=angry, ticker=False)
    monitor.start("a.grib2", host="aws")
    monitor.finish("a.grib2", size=10, seconds=0.5, host="aws")
    assert "starting (aws)" in stream.getvalue()


# ---------------------------------------------------------------------------
# The pool drives it, per job, default-on
# ---------------------------------------------------------------------------

def _entry(name, size=10):
    return {"name": name, "bytes": size}


def test_the_pool_starts_and_finishes_every_job_on_the_monitor():
    monitor, stream, events = _monitor()
    jobs = [fetch_pool.TransferJob(
        name=f"f{index:03d}.grib2",
        url="https://noaa-hrrr-bdp-pds.s3.amazonaws.com/x",
        token=f"f{index:02d} atmosphere", expected_bytes=100 + index,
        action=lambda index=index: _entry(f"f{index:03d}.grib2"))
        for index in range(3)]
    entries, _receipt = fetch_pool.run_transfers(jobs, workers=3,
                                                 monitor=monitor)
    assert len(entries) == 3
    tags = [event for event, _fields in events]
    assert tags.count("fetch_started") == 3
    assert tags.count("fetch_completed") == 3
    lines = stream.getvalue().splitlines()
    assert sum(1 for line in lines if "starting" in line) == 3
    # The host reaches the line from the job's url, with no route help.
    assert all("noaa-hrrr-bdp-pds.s3.amazonaws.com" in line
               for line in lines if "starting" in line)


def test_the_serial_transport_is_visible_too():
    monitor, stream, _events = _monitor()
    fetch_pool.run_transfers(
        [fetch_pool.TransferJob(name="only.grib2", url=None,
                                action=lambda: _entry("only.grib2"))],
        workers=1, monitor=monitor)
    assert "only.grib2 starting" in stream.getvalue()


def test_a_failed_job_is_reported_finished_so_the_line_stops_moving():
    monitor, _stream, events = _monitor()

    def bad():
        raise ValueError("downloaded f006 carries 3 GRIB2 messages")

    with pytest.raises(ValueError, match="f006 carries 3"):
        fetch_pool.run_transfers(
            [fetch_pool.TransferJob(name="f006.grib2", url=None, action=bad)],
            workers=1, monitor=monitor)
    tags = [event for event, _fields in events]
    assert tags == ["fetch_started", "fetch_completed"]
    assert events[-1][1]["failed"] is True


def test_a_pool_without_a_monitor_behaves_exactly_as_before():
    entries, receipt = fetch_pool.run_transfers(
        [fetch_pool.TransferJob(name="a", url=None, action=lambda: _entry("a"))],
        workers=1)
    assert [entry["name"] for entry in entries] == ["a"]
    assert receipt["files"] == 1


# ---------------------------------------------------------------------------
# The regression pin: the completion line's text does not move
# ---------------------------------------------------------------------------

def test_the_completion_line_is_byte_for_byte_what_it_always_was():
    assert progress.format_transfer_done_line(
        label="fetch hrrr", index=0, total=8, name="hrrr.t02z.wrfnatf01.grib2",
        note="725.8 MiB") == (
            "fetch hrrr: [1/8] hrrr.t02z.wrfnatf01.grib2 725.8 MiB")
    assert progress.format_transfer_done_line(
        label="fetch gfs", index=7, total=8, name="gfs.t00z.f021",
        note="already present") == (
            "fetch gfs: [8/8] gfs.t00z.f021 already present")


def test_the_route_still_prints_that_exact_completion_line():
    """The pin above is only worth anything if the route uses it."""

    from gpuwm import fetch_routes

    said: list[str] = []
    landed = fetch_routes._admission_reporter(
        "fetch hrrr", total=8, progress=said.append)
    landed(0, {"relpath": "hrrr.t02z.wrfnatf01.grib2",
               "bytes": 761_265_000, "reused": False})
    landed(1, {"relpath": "gfs.t00z.f021", "reused": True})
    assert said == [
        "fetch hrrr: [1/8] hrrr.t02z.wrfnatf01.grib2 726.0 MiB",
        "fetch hrrr: [2/8] gfs.t00z.f021 already present",
    ]


# ---------------------------------------------------------------------------
# The ticker, which is what makes the line appear without a caller
# ---------------------------------------------------------------------------

def test_the_ticker_speaks_while_a_transfer_is_in_flight_and_stops_after():
    stream = _Stream(False)
    monitor = progress.TransferMonitor("fetch hrrr", stream=stream,
                                       interval=0.01)
    monitor.start("a.grib2", host="aws")
    monitor.observe("a.grib2", 1024)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if "files done" in stream.getvalue():
            break
        time.sleep(0.01)
    monitor.finish("a.grib2", size=1024, seconds=0.5, host="aws")
    monitor.close()
    assert "files done" in stream.getvalue()
    settled = stream.getvalue()
    time.sleep(0.05)
    assert stream.getvalue() == settled     # the thread is gone, not sleeping


def _clock(readings):
    """A monotonic stand-in that walks ``readings`` and holds the last."""

    values = list(readings)

    def read() -> float:
        return values.pop(0) if len(values) > 1 else values[0]

    return read
