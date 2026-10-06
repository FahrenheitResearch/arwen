"""A high-resolution geography download survives a flaky host.

The breakage these pin: one TLS connection reset from the terrain tile
host ended whole storm-layout preparations as a raw ``URLError``
traceback at exit 1, because ``fetch_file`` asked each tile exactly once.
A body cut short by a clean close was worse: ``HTTPResponse.read(n)``
reports it as a quiet end of data, so the short tile was renamed into the
cache and hashed as if it were whole.  And two preparations sharing one
cache staged the same tile into one ``.partial``.  The lock that fixed
that then gave the waiting preparation a flat 600 s, however fast the
other one's download was still arriving: on the 2.28 GB default
land-cover file, any link slower than about 3.8 MB/s ended parallel
preparations on a fresh cache in a refusal.

The stand-in server below is a real HTTP server on the loopback, driven
through the production opener, so the retry, the resume and the refusal
are measured on the sockets a preparation uses.
"""
from __future__ import annotations

import contextlib
import http.server
import io
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error

import pytest

from gpuwm import fetch_guard
from gpuwm.ingest.source_coverage import (PreparationRefusal,
                                          owns_source_coverage_refusal)
from gpuwm.static import highres_fetch
from gpuwm.static.highres_fetch import SourceAbsent, fetch_file

PAYLOAD = bytes(range(256)) * 1200          # 307,200 bytes


@pytest.fixture
def waits(monkeypatch):
    """Record the backoff instead of sleeping through it."""
    seen: list[float] = []
    monkeypatch.setattr(highres_fetch, "_sleep", seen.append, raising=False)
    return seen


@pytest.fixture
def no_proxy(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                 "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")


def _reset() -> ConnectionResetError:
    return ConnectionResetError(104, "Connection reset by peer")


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}


# ---------------------------------------------------------------------------
# The production exception, through the opener seam
# ---------------------------------------------------------------------------

def test_a_reset_during_the_handshake_is_asked_again(tmp_path, waits):
    calls: list[int] = []

    def urlopen(url, offset):
        calls.append(offset)
        if len(calls) == 1:
            # Exactly what the sweep's preparations died of.
            raise urllib.error.URLError(_reset())
        return _Response(PAYLOAD)

    fetched = fetch_file("https://tiles.example/t.tif", tmp_path / "t.tif",
                         urlopen=urlopen)

    assert (tmp_path / "t.tif").read_bytes() == PAYLOAD
    assert fetched.bytes == len(PAYLOAD) and not fetched.cache_hit
    assert calls == [0, 0]
    assert waits == [2.0]


def test_a_host_that_keeps_resetting_is_a_named_refusal(tmp_path, waits):
    calls: list[int] = []

    def urlopen(url, offset):
        calls.append(offset)
        raise urllib.error.URLError(_reset())

    target = tmp_path / "cache" / "Copernicus_DSM_COG_10_N33_00_W118_00_DEM.tif"
    with pytest.raises(PreparationRefusal) as caught:
        fetch_file("https://copernicus-dem-30m.s3.amazonaws.com/x.tif",
                   target, urlopen=urlopen)

    message = str(caught.value)
    assert target.name in message
    assert "copernicus-dem-30m.s3.amazonaws.com" in message
    assert "Connection reset by peer" in message
    assert f"the last of {highres_fetch.FETCH_ATTEMPTS} attempts" in message
    assert "[static.highres] disabled" in caught.value.remedy
    assert len(calls) == highres_fetch.FETCH_ATTEMPTS == 5
    assert waits == [2.0, 4.0, 8.0, 16.0]
    assert not target.exists()


def test_the_refusal_reaches_a_preparation_door_as_two_lines(tmp_path,
                                                             waits, capsys):
    def urlopen(url, offset):
        raise urllib.error.URLError(_reset())

    def main() -> int:
        fetch_file("https://tiles.example/t.tif", tmp_path / "t.tif",
                   urlopen=urlopen)
        return 0

    status = owns_source_coverage_refusal(main)()

    err = capsys.readouterr().err
    assert status == 78
    assert "prep: REFUSED: [static.highres] could not download t.tif" in err
    assert "remedy: check this computer's connection to tiles.example" in err
    assert "Traceback" not in err


def test_an_absent_tile_is_not_asked_again(tmp_path, waits):
    calls: list[int] = []

    def urlopen(url, offset):
        calls.append(offset)
        raise SourceAbsent(f"{url} -> HTTP 404")

    with pytest.raises(SourceAbsent):
        fetch_file("https://tiles.example/t.tif", tmp_path / "t.tif",
                   urlopen=urlopen)
    assert calls == [0]
    assert waits == []


def test_a_failure_on_this_computer_is_not_a_network_refusal(tmp_path, waits):
    def urlopen(url, offset):
        raise PermissionError(13, "Permission denied", str(tmp_path))

    with pytest.raises(PermissionError):
        fetch_file("https://tiles.example/t.tif", tmp_path / "t.tif",
                   urlopen=urlopen)
    assert waits == []


# ---------------------------------------------------------------------------
# A flaky stand-in server, through the production opener
# ---------------------------------------------------------------------------

class _FlakyServer:
    """Answers each GET with the next scripted action.

    ``reset``: abort the connection (RST) without answering.
    ``truncate``: declare the whole remaining body, send half, close.
    ``503``: answer 503.
    ``serve``: answer the payload, honouring ``Range``.
    ``slow``: as ``serve``, but pause after the first half until another
    request arrives or a second passes.
    """

    def __init__(self, actions, payload: bytes = PAYLOAD) -> None:
        self.actions = list(actions)
        self.payload = payload
        self.ranges: list[str | None] = []
        self.lock = threading.Lock()
        self.second_request = threading.Event()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                with owner.lock:
                    owner.ranges.append(self.headers.get("Range"))
                    if len(owner.ranges) > 1:
                        owner.second_request.set()
                    action = (owner.actions.pop(0) if owner.actions
                              else "serve")
                if action == "reset":
                    self.connection.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack("ii", 1, 0))
                    self.connection.close()
                    self.close_connection = True
                    return
                if action == "503":
                    body = b"slow down"
                    self.send_response(503)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                start = 0
                requested = self.headers.get("Range")
                if requested:
                    start = int(requested.split("=")[1].split("-")[0])
                body = owner.payload[start:]
                self.send_response(206 if requested else 200)
                if requested:
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{len(owner.payload) - 1}/"
                        f"{len(owner.payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                half = len(body) // 2
                if action == "truncate":
                    self.wfile.write(body[:half])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if action == "slow":
                    self.wfile.write(body[:half])
                    self.wfile.flush()
                    owner.second_request.wait(1.0)
                    self.wfile.write(body[half:])
                    return
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      Handler)
        self.server.daemon_threads = True
        self.url = (f"http://127.0.0.1:{self.server.server_address[1]}"
                    "/tile.tif")
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self) -> "_FlakyServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_a_reset_then_a_cut_body_resumes_to_the_whole_tile(
        tmp_path, waits, no_proxy, monkeypatch):
    monkeypatch.setattr(highres_fetch, "_CHUNK", 16 * 1024)
    target = tmp_path / "tile.tif"
    with _FlakyServer(["reset", "truncate", "serve"]) as server:
        fetched = fetch_file(server.url, target)

    assert target.read_bytes() == PAYLOAD
    assert fetched.bytes == len(PAYLOAD)
    half = len(PAYLOAD) // 2
    # The third request resumed the bytes the cut body delivered.
    assert server.ranges == [None, None, f"bytes={half}-"]
    assert waits == [2.0, 4.0]
    assert not list(target.parent.glob(target.name + ".partial-*"))


def test_a_body_cut_short_is_never_cached_as_whole(tmp_path, waits,
                                                   no_proxy):
    target = tmp_path / "tile.tif"
    with _FlakyServer(["truncate", "serve"]) as server:
        fetched = fetch_file(server.url, target)

    assert target.read_bytes() == PAYLOAD
    assert fetched.bytes == len(PAYLOAD)
    assert server.ranges == [None, f"bytes={len(PAYLOAD) // 2}-"]


def test_a_throttling_host_is_waited_out(tmp_path, waits, no_proxy):
    target = tmp_path / "tile.tif"
    with _FlakyServer(["503", "503", "serve"]) as server:
        fetch_file(server.url, target)

    assert target.read_bytes() == PAYLOAD
    assert waits == [2.0, 4.0]


def test_a_host_that_never_recovers_is_refused_after_the_last_attempt(
        tmp_path, waits, no_proxy):
    target = tmp_path / "tile.tif"
    with _FlakyServer(["reset"] * 10) as server:
        with pytest.raises(PreparationRefusal) as caught:
            fetch_file(server.url, target)

    assert len(server.ranges) == highres_fetch.FETCH_ATTEMPTS
    assert "could not download tile.tif from 127.0.0.1" in str(caught.value)
    assert not target.exists()


def test_two_preparations_on_one_cache_download_a_tile_once(
        tmp_path, waits, no_proxy):
    target = tmp_path / "tile.tif"
    results: list[object] = [None, None]

    def fetch(slot: int) -> None:
        try:
            results[slot] = fetch_file(server.url, target)
        except BaseException as error:  # noqa: BLE001 - reported below
            results[slot] = error

    with _FlakyServer(["slow"]) as server:
        first = threading.Thread(target=fetch, args=(0,))
        first.start()
        # The second arrives while the first is mid-body on the tile.
        deadline = time.monotonic() + 10.0
        while (not highres_fetch._partial_progress(target)[0]
               and time.monotonic() < deadline):
            time.sleep(0.005)
        second = threading.Thread(target=fetch, args=(1,))
        second.start()
        first.join(30)
        second.join(30)

    assert all(isinstance(result, highres_fetch.FetchedFile)
               for result in results), results
    assert target.read_bytes() == PAYLOAD
    assert server.ranges == [None]
    assert [result.cache_hit for result in results] == [False, True]


# ---------------------------------------------------------------------------
# A slow but live download is waited for; a stalled one is refused
# ---------------------------------------------------------------------------

#: Larger than a buffered writer's buffer, so every block the stand-in
#: link delivers reaches the ``.partial`` on disk as it arrives.
_BLOCK = 64 * 1024

_LANDCOVER_URL = "https://landcover.example/CGLC_MODIS_LCZ_global.tif"


class _TrickleResponse:
    """A body that arrives one block at a time, like a slow link.

    After ``stall_after`` blocks it delivers nothing more until
    ``release`` exists, like a connection that has gone quiet.
    """

    status = 200

    def __init__(self, blocks: int, pause_s: float, stall_after: int | None,
                 release: Path) -> None:
        self.blocks = blocks
        self.pause_s = pause_s
        self.stall_after = stall_after
        self.release = release
        self.sent = 0
        self.headers = {"Content-Length": str(blocks * _BLOCK)}

    def __enter__(self) -> "_TrickleResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self, amount: int = -1) -> bytes:
        if self.sent >= self.blocks:
            return b""
        if self.stall_after is not None and self.sent == self.stall_after:
            while not self.release.exists():
                time.sleep(0.02)
        time.sleep(self.pause_s)
        self.sent += 1
        return bytes([self.sent % 256]) * _BLOCK


def _trickle_payload(blocks: int) -> bytes:
    return b"".join(bytes([(index + 1) % 256]) * _BLOCK
                    for index in range(blocks))


def _slow_holder(target: str, blocks: int, pause_s: float,
                 stall_after: int | None, release: str):
    """The preparation that got to the file first, on a slow link."""
    return fetch_file(
        _LANDCOVER_URL, Path(target),
        urlopen=lambda url, offset: _TrickleResponse(
            blocks, pause_s, stall_after, Path(release)))


_HOLDER_PROCESS = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("slow_holder", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
stall = None if sys.argv[5] == "-" else int(sys.argv[5])
module._slow_holder(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
                    stall, sys.argv[6])
print("fetched")
"""


@contextlib.contextmanager
def _holder_downloading(kind: str, target: Path, release: Path, *,
                        blocks: int, pause_s: float,
                        stall_after: int | None = None):
    """Start the first preparation's download and yield once it has bytes
    staged; on leaving, let it finish and check that it did."""
    outcome: list[object] = []
    if kind == "process":
        child = subprocess.Popen(
            [sys.executable, "-c", _HOLDER_PROCESS, __file__, str(target),
             str(blocks), str(pause_s),
             "-" if stall_after is None else str(stall_after), str(release)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=dict(os.environ), cwd=str(Path(__file__).resolve().parents[1]))

        def alive() -> bool:
            return child.poll() is None

        def finish() -> None:
            out, err = child.communicate(timeout=120)
            assert child.returncode == 0 and "fetched" in out, err
    else:
        def run() -> None:
            try:
                outcome.append(_slow_holder(str(target), blocks, pause_s,
                                            stall_after, str(release)))
            except BaseException as error:  # noqa: BLE001 - reported below
                outcome.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def alive() -> bool:
            return thread.is_alive()

        def finish() -> None:
            thread.join(120)
            assert len(outcome) == 1 and isinstance(
                outcome[0], highres_fetch.FetchedFile), outcome
    try:
        deadline = time.monotonic() + 120
        while (sum(size for _, size in
                   highres_fetch._partial_progress(target)[0]) == 0
               and time.monotonic() < deadline):
            assert alive() or target.exists(), "the holder stopped early"
            time.sleep(0.01)
        assert sum(size for _, size in
                   highres_fetch._partial_progress(target)[0]) > 0, \
            "the holder never staged a byte"
        yield
    finally:
        release.write_text("go", encoding="utf-8")
        finish()


def _never_downloads(url, offset):
    raise AssertionError("the waiting preparation opened a download of "
                         "its own while the other one held the file")


@pytest.mark.parametrize("kind", ["process", "thread"])
def test_a_slow_but_live_download_is_waited_for_past_the_lock_budget(
        tmp_path, monkeypatch, kind):
    """The second preparation reads the file from the cache once the first
    has downloaded it, however long that takes, while bytes keep arriving."""
    budget_s = 1.0
    monkeypatch.setenv(fetch_guard.LOCK_TIMEOUT_ENV, str(budget_s))
    target = tmp_path / "cache" / "CGLC_MODIS_LCZ_global.tif"
    target.parent.mkdir()
    blocks = 30
    with _holder_downloading(kind, target, tmp_path / "release.flag",
                             blocks=blocks, pause_s=0.1):
        started = time.monotonic()
        waited = fetch_file(_LANDCOVER_URL, target,
                            urlopen=_never_downloads)
        elapsed = time.monotonic() - started

    assert waited.cache_hit
    assert waited.bytes == blocks * _BLOCK
    assert target.read_bytes() == _trickle_payload(blocks)
    # The download outlasted the wait budget twice over and was still
    # waited for: only a holder that stops making progress is refused.
    assert elapsed >= 2 * budget_s, elapsed


@pytest.mark.parametrize("kind", ["process", "thread"])
def test_a_download_that_stops_growing_is_refused_after_the_budget(
        tmp_path, monkeypatch, kind):
    budget_s = 1.0
    monkeypatch.setenv(fetch_guard.LOCK_TIMEOUT_ENV, str(budget_s))
    target = tmp_path / "cache" / "CGLC_MODIS_LCZ_global.tif"
    target.parent.mkdir()
    release = tmp_path / "release.flag"
    with _holder_downloading(kind, target, release, blocks=30, pause_s=0.05,
                             stall_after=3):
        started = time.monotonic()
        with pytest.raises(PreparationRefusal) as caught:
            fetch_file(_LANDCOVER_URL, target, urlopen=_never_downloads)
        elapsed = time.monotonic() - started
        staged = sum(size for _, size in
                     highres_fetch._partial_progress(target)[0])

    message = str(caught.value)
    assert isinstance(caught.value, highres_fetch.HighresFetchRefusal)
    assert target.name in message and str(target.parent) in message
    assert "has not grown for 1 s" in message, message
    assert f"{staged} bytes" in message, message
    assert fetch_guard.LOCK_TIMEOUT_ENV in caught.value.remedy
    assert budget_s <= elapsed < 60, elapsed
    # The refusal left the other preparation's download alone: once it
    # finished, the file is in the cache whole.
    assert target.read_bytes() == _trickle_payload(30)
    assert fetch_file(_LANDCOVER_URL, target,
                      urlopen=_never_downloads).cache_hit
