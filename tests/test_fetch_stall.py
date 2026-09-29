"""The Rust fetch backbone's live byte count and its chunk-stream budget.

A 36 h HRRR fetch through the backbone is 74 whole files, 44 GB, six in
flight.  Each file used to open one chunk stream per CPU thread (144 on
a 24-thread machine) and was held in memory until whole, so the progress
line and the run's ``fetch_progress`` events read 0 B for every file in
flight until it landed.  These tests pin the Python half of the remedy:
``rw_fetch``'s ``rw_fetch-progress`` lines are read while the backbone
runs and reach the transfer monitor, and every invocation is told its
share of one fetch-wide stream budget.  The stall limit, the resume and
the bounded pool themselves are pinned by the Rust tests in
``tools/rustwx/vendor/wx-core/src/download/client.rs``.
"""

from __future__ import annotations

from datetime import datetime
import functools
import io
import json
from pathlib import Path
import sys

import pytest

from gpuwm import fetch, fetch_pool, progress as progress_mod, rustwx_fetch
from gpuwm.fetch_bars import CERTIFIED_RECORD_BARS
from tools import download_hrrr_native_subset as hrrr_transport


def _stand_in(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "stand_in.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_progress_reaches_the_caller_while_the_backbone_is_still_running(
        tmp_path):
    """Streamed, not read at exit.

    The stand-in prints one progress line and then waits for the file
    the caller's callback creates.  A caller that only read stderr once
    the process had exited could never create it, and the stand-in
    would say so in its record.
    """

    sentinel = tmp_path / "heard"
    command = _stand_in(tmp_path, f"""
import json, os, sys, time
sys.stderr.write("\\r  Downloading chunks 1/3...\\nrw_fetch-progress f004 1048576 3000000\\n")
sys.stderr.flush()
deadline = time.monotonic() + 20
while not os.path.exists({str(sentinel)!r}) and time.monotonic() < deadline:
    time.sleep(0.05)
sys.stderr.write("\\nrw_fetch-progress f004 3000000 3000000\\n")
print(json.dumps({{"heard": os.path.exists({str(sentinel)!r})}}))
""")
    heard: list[tuple[int, int | None]] = []

    def on_progress(received, total):
        heard.append((received, total))
        sentinel.touch()

    record = rustwx_fetch._run(command, what="fetch", on_progress=on_progress)
    assert record == {"heard": True}
    assert heard == [(1048576, 3000000), (3000000, 3000000)]


def test_a_progress_line_is_never_the_failure_reason(tmp_path):
    command = _stand_in(tmp_path, """
import sys
sys.stderr.write("\\nrw_fetch-progress f000 5 -\\n"
                 "\\nrw_fetch: HTTP error: failed to read x"
                 " (gave up after 4 attempts)\\n")
sys.exit(3)
""")
    seen: list[tuple[int, int | None]] = []
    with pytest.raises(rustwx_fetch.RwFetchError) as caught:
        rustwx_fetch._run(command, what="fetch",
                          on_progress=lambda *pair: seen.append(pair))
    assert caught.value.transient
    assert caught.value.reason == (
        "HTTP error: failed to read x (gave up after 4 attempts)")
    assert seen == [(5, None)]


def test_progress_lines_parse_wherever_they_sit():
    parse = rustwx_fetch.parse_progress
    assert parse("rw_fetch-progress f012 10 20") == (10, 20)
    assert parse("  Downloading chunks 3/9...rw_fetch-progress f012 10 -") == (
        10, None)
    assert parse("  Downloading chunks 3/9...") is None
    assert parse("rw_fetch-progress f012 ten 20") is None
    assert rustwx_fetch._REASON_MARK not in "rw_fetch-progress f012 10 20"


def test_a_callback_that_raises_does_not_fail_the_fetch(tmp_path):
    command = _stand_in(tmp_path, """
import json, sys
sys.stderr.write("rw_fetch-progress f000 1 2\\n")
print(json.dumps({"ok": 1}))
""")

    def broken(*_):
        raise RuntimeError("a front end that went away")

    assert rustwx_fetch._run(command, what="fetch",
                             on_progress=broken) == {"ok": 1}


def test_run_fetch_hands_the_backbone_its_stream_share(tmp_path, monkeypatch):
    """``streams`` travels as the environment the backbone reads."""

    command = _stand_in(tmp_path, """
import json, os
print(json.dumps({"schema": "gpuwm-rw-fetch-record-v1",
                  "streams": os.environ.get("RUSTWX_DOWNLOAD_STREAMS")}))
""")
    monkeypatch.delenv(rustwx_fetch.STREAMS_ENV, raising=False)
    run = functools.partial(
        rustwx_fetch.run_fetch, Path(command[1]), model="hrrr",
        date="20260926", cycle=12, hours=(0,), product="wrfnat",
        out=tmp_path)
    monkeypatch.setattr(rustwx_fetch, "_run",
                        functools.partial(_run_through_python,
                                          rustwx_fetch._run))
    assert run(streams=3)["streams"] == "3"
    assert run()["streams"] is None
    for bad in (0, -2, True):
        with pytest.raises(ValueError):
            run(streams=bad)


def _run_through_python(real, command, **kwargs):
    # The stand-in is a script: run it with this interpreter, keeping the
    # arguments run_fetch built after the binary.
    return real([sys.executable, command[0], *command[1:]], **kwargs)


@pytest.mark.parametrize("workers,files,host,share", [
    (None, 74, None, fetch_pool.FETCH_STREAM_BUDGET
     // fetch_pool.DEFAULT_FILE_WORKERS),
    (1, 74, None, fetch_pool.FETCH_STREAM_BUDGET),
    (6, 1, None, fetch_pool.FETCH_STREAM_BUDGET),
    (64, 74, None, 1),
    (6, 74, "nomads.ncep.noaa.gov",
     fetch_pool.FETCH_STREAM_BUDGET // min(
         6, fetch_pool.host_worker_cap("nomads.ncep.noaa.gov", 6))),
])
def test_the_stream_budget_is_split_over_the_files_in_flight(
        workers, files, host, share):
    assert fetch_pool.chunk_streams_per_file(
        workers, files=files, host=host) == share


def test_six_files_in_flight_stay_within_the_fetch_budget():
    per_file = fetch_pool.chunk_streams_per_file(None, files=74)
    assert per_file * fetch_pool.DEFAULT_FILE_WORKERS <= (
        fetch_pool.FETCH_STREAM_BUDGET)
    assert per_file >= 1


def test_a_relay_turns_a_running_count_into_bytes_moved():
    stream = io.StringIO()
    monitor = progress_mod.TransferMonitor(
        "fetch hrrr", stream=stream, ticker=False)
    monitor.start("hrrr.t12z.wrfnatf00.grib2")
    first = monitor.relay("hrrr.t12z.wrfnatf00.grib2")
    first(1000, None)
    first(4000, 750_000_000)
    first(4000, 750_000_000)          # a repeat moves nothing
    record = monitor._files["hrrr.t12z.wrfnatf00.grib2"]
    assert (record.moved(), record.expected) == (4000, 750_000_000)
    # The backbone asked again starts from zero; what it moves the
    # second time is moved all the same.
    second = monitor.relay("hrrr.t12z.wrfnatf00.grib2")
    second(2500, 750_000_000)
    assert record.moved() == 6500


def test_the_rust_route_passes_its_share_and_relays_the_bytes(
        tmp_path, monkeypatch):
    calls: list[dict] = []

    def backbone(binary, **kwargs):
        calls.append(kwargs)
        kwargs["on_progress"](16_777_216, 442_060_360)
        kwargs["on_progress"](442_060_360, 442_060_360)
        return {"schema": rustwx_fetch.FETCH_RECORD_SCHEMA, "files": [{
            "name": "hrrr.t12z.wrfprsf10.grib2", "bytes": 442_060_360,
            "wall_seconds": 9.0, "source": "aws", "mode": "full-file",
            "mode_reason": "requested",
            "grib_url": "https://example/hrrr.t12z.wrfprsf10.grib2"}]}

    monkeypatch.setattr(rustwx_fetch, "run_fetch", backbone)
    monitor = progress_mod.TransferMonitor(
        "fetch hrrr", stream=io.StringIO(), ticker=False)
    monitor.start("hrrr.t12z.soilf10.grib2")
    fetch._rw_fetch_hrrr(
        binary=Path("rw_fetch"), cycle=datetime(2026, 9, 26, 12), hour=10,
        kind="soil", host="s3", mode="full-file", out=tmp_path,
        cache_dir=None, progress=lambda _: None,
        shown_name="hrrr.t12z.soilf10.grib2", streams=2,
        byte_relay=functools.partial(monitor.relay,
                                     "hrrr.t12z.soilf10.grib2"))
    assert calls[0]["streams"] == 2
    record = monitor._files["hrrr.t12z.soilf10.grib2"]
    assert (record.moved(), record.expected) == (442_060_360, 442_060_360)


def test_a_pooled_hrrr_fetch_gives_every_file_its_share_and_a_relay(
        tmp_path, monkeypatch):
    """What `gpuwm go` runs: the pool, the budget split, live bytes."""

    seen: list[dict] = []
    events: list[tuple[str, dict]] = []

    def fake(*, binary, cycle, hour, kind, host, mode, out, cache_dir,
             progress, retries=0, shown_name=None, streams=None,
             byte_relay=None):
        seen.append({"streams": streams, "relay": byte_relay})
        records = (CERTIFIED_RECORD_BARS["hrrr-atmosphere"]
                   if kind == "atmosphere"
                   else CERTIFIED_RECORD_BARS["hrrr-soil"])
        name = (f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
                if kind == "atmosphere"
                else f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2")
        payload = _grib2(records)
        if byte_relay is not None:
            byte_relay()(len(payload) // 2, len(payload))
        (out / name).write_bytes(payload)
        return {"name": name, "idx_name": None, "mode": "idx-subset",
                "mode_reason": "fixture", "bytes": len(payload),
                "wall_seconds": 0.1, "source": "aws",
                "grib_url": f"https://example.invalid/{name}",
                "selected_record_count": records, "probe": {}}

    monkeypatch.setattr(fetch, "_rw_fetch_hrrr", fake)
    with progress_mod.event_sink(
            lambda event, **fields: events.append((event, fields))):
        fetch.fetch_hrrr(
            cycle=datetime(2026, 9, 26, 12), hours=(0, 1, 2), area=None,
            out=tmp_path / "hrrr", transport="s3", engine="rust",
            engine_bin=Path("rw_fetch"), progress=lambda _: None,
            file_workers=6)
    assert len(seen) == 6
    assert {item["streams"] for item in seen} == {
        fetch_pool.chunk_streams_per_file(6, files=6)}
    assert all(callable(item["relay"]) for item in seen)
    done = [fields for event, fields in events
            if event == "fetch_completed"]
    assert len(done) == 6 and not any(fields["failed"] for fields in done)


def _grib2(records: int) -> bytes:
    """``records`` minimal GRIB2 envelopes: GRIB, edition 2, 7777."""

    one = (b"GRIB" + bytes([0, 0, 0, 2])
           + (20).to_bytes(8, "big") + b"7777")
    return one * records
