"""Process concurrency and corruption recovery for the shared static cache."""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import multiprocessing
import os
from pathlib import Path
import threading
import time

import pytest

from gpuwm import fetch_guard
from gpuwm.static import highres_fetch


def _fetch_module():
    """A source snapshot can run the same regression without editing the tree."""
    source = os.environ.get("GPUWM_HIGHRES_FETCH_TEST_SOURCE")
    if source is None:
        return highres_fetch
    spec = importlib.util.spec_from_file_location(
        "gpuwm.static.highres_fetch_snapshot", source)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TileServer:
    def __init__(self, cache: Path):
        self.cache = cache
        self.payloads = {
            "/a.tif": bytes(range(256)) * 2048,
            "/b.tif": bytes(reversed(range(256))) * 2048,
        }
        self.requests = []
        self.staged_names = []
        self.truncate_next = False
        self.lock = threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                payload = owner.payloads[self.path]
                requested = self.headers.get("Range")
                start = int(requested.split("=")[1].split("-")[0]) \
                    if requested else 0
                with owner.lock:
                    owner.requests.append((self.path, requested))
                    owner.staged_names.extend(
                        path.name for path in owner.cache.glob("*.partial*"))
                    truncate = owner.truncate_next
                    owner.truncate_next = False
                body = payload[start:]
                self.send_response(206 if requested else 200)
                if requested:
                    self.send_header("Content-Range",
                                     f"bytes {start}-{len(payload) - 1}/"
                                     f"{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                half = len(body) // 2
                self.wfile.write(body[:half])
                self.wfile.flush()
                if truncate:
                    self.close_connection = True
                    return
                time.sleep(0.3)
                with owner.lock:
                    owner.staged_names.extend(
                        path.name for path in owner.cache.glob("*.partial*"))
                self.wfile.write(body[half:])

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                                      Handler)
        self.server.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture(autouse=True)
def _isolated_cache_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(tmp_path / "locks"))
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                 "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")


def _fetch_worker(cache, base, ready, start, results):
    module = _fetch_module()
    ready.put(os.getpid())
    start.wait(30)
    try:
        fetched = [module.fetch_file(base + "/" + name, Path(cache) / name)
                   for name in ("a.tif", "b.tif")]
        results.put({"pid": os.getpid(), "hits": [f.cache_hit for f in fetched],
                     "hashes": [f.sha256 for f in fetched]})
    except BaseException as error:
        results.put({"error": repr(error)})


def _run_workers(context, worker, args, count=4):
    ready, results = context.Queue(), context.Queue()
    start = context.Event()
    workers = [context.Process(target=worker,
                               args=(*args, ready, start, results))
               for _ in range(count)]
    try:
        for worker_process in workers:
            worker_process.start()
        pids = [ready.get(timeout=30) for _ in workers]
        assert len(set(pids)) == count
        start.set()
        outcomes = [results.get(timeout=60) for _ in workers]
        for worker_process in workers:
            worker_process.join(30)
            assert worker_process.exitcode == 0
        assert all("error" not in result for result in outcomes), outcomes
        return outcomes
    finally:
        start.set()
        for worker_process in workers:
            if worker_process.is_alive():
                worker_process.terminate()
            worker_process.join(5)
        ready.close()
        results.close()


def test_real_processes_download_each_tile_once_without_shared_partials(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    with _TileServer(cache) as server:
        outcomes = _run_workers(multiprocessing.get_context("spawn"),
                                _fetch_worker, (str(cache), server.base))
    assert sorted(server.requests) == [("/a.tif", None), ("/b.tif", None)]
    assert sum(not hit for result in outcomes for hit in result["hits"]) == 2
    expected = [hashlib.sha256(server.payloads["/" + name]).hexdigest()
                for name in ("a.tif", "b.tif")]
    assert all(result["hashes"] == expected for result in outcomes)
    assert server.staged_names
    assert all(".partial-" in name for name in server.staged_names)
    assert all(any(f".partial-{result['pid']}-" in name
                   for result in outcomes) for name in server.staged_names)
    for name in ("a.tif", "b.tif"):
        assert (cache / name).read_bytes() == server.payloads["/" + name]
    assert not list(cache.glob("*.partial*"))
    assert not list(cache.glob("*.resume.json"))


@pytest.mark.parametrize("damage", ["same-size", "short", "invalid-sidecar"])
def test_a_corrupt_cached_tile_is_rejected_and_refetched(tmp_path, damage):
    module = _fetch_module()
    target = tmp_path / "a.tif"
    with _TileServer(tmp_path) as server:
        first = module.fetch_file(server.base + "/a.tif", target)
        if damage == "same-size":
            target.write_bytes(b"\0" * first.bytes)
        elif damage == "short":
            target.write_bytes(b"short")
        else:
            target.with_name(target.name + ".sha256.json").write_text("[]\n")
        recovered = module.fetch_file(server.base + "/a.tif", target)
    assert not recovered.cache_hit
    assert recovered.sha256 == first.sha256
    assert target.read_bytes() == server.payloads["/a.tif"]
    assert server.requests == [("/a.tif", None), ("/a.tif", None)]
    assert not list(tmp_path.glob("*.partial*"))


def test_a_failed_download_resumes_in_a_new_owned_partial(tmp_path, monkeypatch):
    module = _fetch_module()
    monkeypatch.setattr(module, "FETCH_ATTEMPTS", 1)
    target = tmp_path / "a.tif"
    with _TileServer(tmp_path) as server:
        server.truncate_next = True
        with pytest.raises(module.HighresFetchRefusal):
            module.fetch_file(server.base + "/a.tif", target)
        old_partial, = tmp_path.glob("a.tif.partial-*")
        assert not target.exists()
        record = module.fetch_file(server.base + "/a.tif", target)
    assert target.read_bytes() == server.payloads["/a.tif"]
    assert record.bytes == len(server.payloads["/a.tif"])
    assert server.requests == [("/a.tif", None),
                               ("/a.tif", f"bytes={record.bytes // 2}-")]
    assert len(set(server.staged_names)) == 2
    assert not old_partial.exists()
    assert not list(tmp_path.glob("*.partial*"))
    assert not list(tmp_path.glob("*.resume.json"))


def test_pinned_integrity_is_checked_before_publication(tmp_path):
    module = _fetch_module()
    target = tmp_path / "a.tif"
    with _TileServer(tmp_path) as server:
        with pytest.raises(ValueError, match="SHA-256"):
            module.fetch_file(server.base + "/a.tif", target,
                              expected_sha256="0" * 64)
    assert not target.exists()
    assert not target.with_name(target.name + ".sha256.json").exists()
    assert not list(tmp_path.glob("*.partial*"))


class _DerivedBridge:
    """Slow opaque backend isolates cache publication from raster decoding."""
    def __init__(self, cache):
        self.cache = Path(cache)

    def highres_derive_window(self, request):
        with (self.cache / "derive-calls.log").open("a", encoding="utf-8") as log:
            log.write(f"{os.getpid()} {request['out_path']}\n")
        with Path(request["out_path"]).open("wb") as stream:
            stream.write(b"derived-first-half")
            stream.flush()
            time.sleep(0.4)
            stream.write(b"derived-second-half")
        return {"output_shape": [1, 1], "total_pixels": 1,
                "hole_pixels": 0, "resampling": "nearest"}


def _derive_worker(cache, kind, ready, start, results):
    module = _fetch_module()
    bridge = _DerivedBridge(cache)
    module._static_rust = lambda operation: bridge
    source = Path(cache) / "source.tif"
    fetched = module.FetchedFile(path=source, url="local:source",
                                 sha256=hashlib.sha256(b"source").hexdigest(),
                                 bytes=6, fetched_utc="", cache_hit=True)
    bbox = module.FootprintBBox(39.1, 39.2, -105.2, -105.1)
    ready.put(os.getpid())
    start.wait(30)
    try:
        if kind == "global":
            artifact, audit = module.derive_global_terrain_window(
                [fetched], bbox, Path(cache))
        elif kind == "terrain":
            artifact = module.derive_terrain_window([fetched], bbox, Path(cache))
        else:
            artifact = module.derive_landcover_window(fetched, bbox, Path(cache))
        results.put({"pid": os.getpid(), "hit": artifact.cache_hit,
                     "path": str(artifact.path), "hash": artifact.sha256})
    except BaseException as error:
        results.put({"error": repr(error)})


@pytest.mark.parametrize("kind", ["global", "terrain", "landcover"])
def test_real_processes_derive_one_complete_cached_window(tmp_path, kind):
    (tmp_path / "source.tif").write_bytes(b"source")
    outcomes = _run_workers(multiprocessing.get_context("spawn"),
                            _derive_worker, (str(tmp_path), kind))
    assert sum(not result["hit"] for result in outcomes) == 1
    assert len(set(result["hash"] for result in outcomes)) == 1
    call, = (tmp_path / "derive-calls.log").read_text().splitlines()
    assert ".partial-" in call
    assert Path(outcomes[0]["path"]).read_bytes() == \
        b"derived-first-halfderived-second-half"
    assert not list((tmp_path / "derived").glob("*.partial*"))


def _fork_lock_worker(target, results):
    try:
        with fetch_guard.hold("highres-fetch", target, timeout_s=0):
            results.put("acquired")
    except fetch_guard.FetchLockBusy:
        results.put("busy")


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="fork is a POSIX process-start path")
def test_forked_child_cannot_reenter_a_parent_tile_lock(tmp_path):
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    target = tmp_path / "tile.tif"
    with fetch_guard.hold("highres-fetch", target):
        child = context.Process(target=_fork_lock_worker,
                                args=(str(target), results))
        child.start()
        try:
            assert results.get(timeout=10) == "busy"
            child.join(10)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.terminate()
            child.join(5)
    results.close()
