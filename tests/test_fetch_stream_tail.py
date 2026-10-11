"""Stream identity, mutation refusal and completion versus ordered admission."""
import hashlib
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from gpuwm import fetch, fetch_pool, fetch_routes


def grib(length=113):
    data = bytearray(length)
    data[:4] = b"GRIB"
    data[7] = 2
    data[8:16] = length.to_bytes(8, "big")
    data[-4:] = b"7777"
    return bytes(data)


def publication(path):
    # The identity the binding compares, which rw_fetch's receipt must name.
    # Not path.stat(): on Windows its st_ctime is the creation time and its
    # st_dev changes width between Python 3.11 and 3.12.
    return fetch._publication(fetch._stat_identity(path))


def change_time(path):
    return fetch._stat_identity(path)[5]


def test_identity_names_the_file_its_size_and_a_change_time_that_moves(tmp_path):
    one, two = tmp_path / "one.grib2", tmp_path / "two.grib2"
    one.write_bytes(grib())
    two.write_bytes(grib(200))
    first = publication(one)
    status = one.stat()
    assert (first["bytes"], first["mtime_ns"]) == (status.st_size, status.st_mtime_ns)
    other = publication(two)
    assert other["dev"] == first["dev"] and other["ino"] != first["ino"]
    moved = tmp_path / "moved.grib2"
    one.rename(moved)
    renamed = publication(moved)
    kept = ("dev", "ino", "bytes", "mtime_ns")
    assert {key: renamed[key] for key in kept} == {key: first[key] for key in kept}
    # A same-size rewrite whose modification time is put back must still
    # move the change time, or a stale digest binds to the new bytes. On
    # Windows os.stat's st_ctime (the creation time) never moves here.
    status = moved.stat()
    changed = bytearray(grib()); changed[20] ^= 1
    for _ in range(2000):
        moved.write_bytes(changed)
        os.utime(moved, ns=(status.st_atime_ns, status.st_mtime_ns))
        rewritten = publication(moved)
        if rewritten["ctime_ns"] != renamed["ctime_ns"]:
            break
        time.sleep(0.001)
    assert rewritten["mtime_ns"] == renamed["mtime_ns"]
    assert rewritten["ctime_ns"] != renamed["ctime_ns"]


def test_fresh_receipt_avoids_payload_read_and_append_invalidates_it(tmp_path, monkeypatch):
    path = tmp_path / "fresh.grib2"
    payload = grib() * 3
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    fetch._remember_stream_facts(path, {"bytes": len(payload), "sha256": digest,
        "grib2_messages": 3, "publication":publication(path)})
    original_digest = fetch.sha256_file
    original_records = fetch.count_grib2_messages
    def forbidden(path):
        raise AssertionError("fresh stream payload was re-read")
    monkeypatch.setattr(fetch, "sha256_file", forbidden)
    monkeypatch.setattr(fetch, "count_grib2_messages", forbidden)
    assert fetch._fetched_digest(path) == digest
    assert fetch._fetched_records(path) == 3
    assert fetch.existing_file_digest(path) == digest
    path.write_bytes(payload + grib())
    monkeypatch.setattr(fetch, "sha256_file", original_digest)
    monkeypatch.setattr(fetch, "count_grib2_messages", original_records)
    assert fetch._fetched_digest(path) == hashlib.sha256(payload + grib()).hexdigest()
    assert fetch._fetched_records(path) == 4


def test_fresh_same_size_edit_with_preserved_mtime_cannot_reuse_stream_digest(tmp_path):
    path = tmp_path / "mutable.grib2"
    payload = grib()
    path.write_bytes(payload)
    fetch._remember_stream_facts(path, {"bytes":len(payload),
        "sha256":hashlib.sha256(payload).hexdigest(), "grib2_messages":1,
        "publication":publication(path)})
    status = path.stat()
    changed = bytearray(payload); changed[20] ^= 1
    path.write_bytes(changed)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert fetch.existing_file_digest(path) == hashlib.sha256(changed).hexdigest()


def test_change_between_publication_and_binding_is_refused_by_name(tmp_path):
    path = tmp_path / "late-edit.grib2"
    payload = grib()
    path.write_bytes(payload)
    receipt = {"bytes":len(payload), "sha256":hashlib.sha256(payload).hexdigest(),
        "grib2_messages":1, "publication":publication(path)}
    status = path.stat()
    published = change_time(path)
    changed = bytearray(payload); changed[20] ^= 1
    # The refusal reads the change time. On a filesystem with a coarse
    # timestamp tick (overlayfs on the sprint boxes, NTFS) a rewrite inside
    # the same tick keeps it, and the test then failed by order alone; the
    # same-tick case is the coarse-clock test below. Rewrite until the
    # change time moves (one tick, a few milliseconds).
    for _ in range(2000):
        path.write_bytes(changed)
        if change_time(path) != published:
            break
        time.sleep(0.001)
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    with pytest.raises(ValueError, match="late-edit.grib2.*changed after Rust publication"):
        fetch._remember_stream_facts(path, receipt)


def test_mutation_invalidates_digest_even_if_stat_clock_does_not_move(tmp_path, monkeypatch):
    path = tmp_path / "coarse-clock.grib2"
    payload = grib()
    path.write_bytes(payload)
    receipt = {"bytes":len(payload), "sha256":hashlib.sha256(payload).hexdigest(),
               "grib2_messages":1, "publication":publication(path)}
    identity = fetch._stat_identity(path)
    fetch._remember_stream_facts(path, receipt)
    monkeypatch.setattr(fetch, "_stat_identity", lambda _: identity)
    changed = bytearray(payload); changed[20] ^= 1
    path.write_bytes(changed)
    assert fetch.existing_file_digest(path) == hashlib.sha256(changed).hexdigest()


def test_another_process_writing_invalidates_digest_even_if_stat_clock_does_not_move(tmp_path, monkeypatch):
    path = tmp_path / "shared.grib2"
    payload = grib()
    path.write_bytes(payload)
    receipt = {"bytes":len(payload), "sha256":hashlib.sha256(payload).hexdigest(),
               "grib2_messages":1, "publication":publication(path)}
    identity = fetch._stat_identity(path)
    fetch._remember_stream_facts(path, receipt)
    monkeypatch.setattr(fetch, "_stat_identity", lambda _: identity)
    # Opened for update, not replaced: the same file, its size kept.
    subprocess.run([sys.executable, "-c", "import sys\n"
        "with open(sys.argv[1], 'r+b') as f:\n f.seek(20); f.write(b'\\x01')",
        str(path)], check=True, timeout=60)
    assert fetch.existing_file_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes() != payload


def test_reading_a_fresh_payload_keeps_its_receipt(tmp_path, monkeypatch):
    path = tmp_path / "read.grib2"
    payload = grib() * 2
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    fetch._remember_stream_facts(path, {"bytes":len(payload), "sha256":digest,
        "grib2_messages":2, "publication":publication(path)})
    # Decoders open, map and stat the payload; none of that is a change.
    assert path.read_bytes() == payload
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        assert view[:4] == b"GRIB"
    path.stat()
    def forbidden(path):
        raise AssertionError("a read invalidated the fresh receipt")
    monkeypatch.setattr(fetch, "sha256_file", forbidden)
    assert fetch._fetched_digest(path) == digest


def test_fetch_out_end_frees_the_folder_and_file_it_published(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    path = out / "held.grib2"
    payload = grib()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with fetch._fetch_out(out):
        fetch._remember_stream_facts(path, {"bytes":len(payload), "sha256":digest,
            "grib2_messages":1, "publication":publication(path)})
        with monkeypatch.context() as patch:
            def forbidden(path):
                raise AssertionError("fresh stream payload was re-read inside its fetch")
            patch.setattr(fetch, "sha256_file", forbidden)
            assert fetch._fetched_digest(path) == digest
    # A Windows notice holds the file open, which refuses both of these
    # (WinError 5) for as long as it is held.
    replacement = tmp_path / "replacement.grib2"
    replacement.write_bytes(grib(200))
    os.replace(replacement, path)
    out.rename(tmp_path / "renamed")
    moved = tmp_path / "renamed" / "held.grib2"
    assert fetch._fetched_digest(moved) == hashlib.sha256(grib(200)).hexdigest()


def test_soil_rename_keeps_verified_digest_without_reopening_bytes(tmp_path, monkeypatch):
    landed = tmp_path / "source.grib2"
    dest = tmp_path / "soil.grib2"
    payload = grib()
    landed.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    entry = {"bytes":len(payload), "sha256":digest,
        "grib2_messages":1, "publication":publication(landed)}
    fetch._rename_stream_payload(landed, dest, entry)
    def forbidden(path):
        raise AssertionError("renamed stream payload was re-read")
    monkeypatch.setattr(fetch, "sha256_file", forbidden)
    assert fetch._fetched_digest(dest) == digest
    assert not landed.exists()


def test_mismatched_stream_size_is_refused_by_name(tmp_path):
    path = tmp_path / "changed.grib2"
    path.write_bytes(grib())
    with pytest.raises(ValueError, match="changed.grib2.*byte count"):
        fetch._remember_stream_facts(path, {"bytes": 1, "sha256": "0" * 64,
            "grib2_messages": 1})


def test_completion_can_bind_while_prefix_waits():
    ready = threading.Event()
    completed, admitted = [], []
    caller = threading.get_ident()
    def early():
        assert ready.wait(3), "supplement was left behind the slow prefix"
        return {"name": "first", "bytes": 1}
    def complete(index, entry):
        assert threading.get_ident() == caller
        completed.append(index)
        ready.set()
    entries, receipt = fetch_pool.run_transfers([
        fetch_pool.TransferJob("first", None, early),
        fetch_pool.TransferJob("supplement", None, lambda: {"name": "supplement", "bytes": 2}),
    ], workers=2, on_completed=complete,
        on_admitted=lambda index, entry: admitted.append(index))
    assert completed == [1, 0]
    assert admitted == [0, 1]
    assert [entry["name"] for entry in entries] == ["first", "supplement"]
    assert receipt["worker_row"] == "balanced"


def test_scoped_completion_hook_restores_outer_hook():
    calls = []
    jobs = [fetch_pool.TransferJob("payload", None,
        lambda: {"name": "payload", "bytes": 1})]
    with fetch_pool.completion_hook(lambda index, entry: calls.append("outer")):
        with fetch_pool.completion_hook(lambda index, entry: calls.append("inner")):
            fetch_pool.run_transfers(jobs, workers=1)
        fetch_pool.run_transfers(jobs, workers=1)
    fetch_pool.run_transfers(jobs, workers=1)
    assert calls == ["inner", "outer"]


@pytest.mark.parametrize("row,width,budget", [("constrained",3,12), ("balanced",6,48), ("wide",12,48)])
def test_packaged_width_rows(row, width, budget, monkeypatch):
    monkeypatch.setenv("GPUWM_FETCH_ROW", row)
    assert fetch_pool.resolve_file_workers(None) == width
    assert fetch_pool.chunk_streams_per_file(None, files=49) == budget // width
    assert fetch_pool.resolve_file_workers(1) == 1


def test_link_hint_and_provider_are_table_selectors(monkeypatch):
    monkeypatch.delenv("GPUWM_FETCH_ROW", raising=False)
    monkeypatch.setenv("GPUWM_FETCH_PROVIDER", "runpod")
    monkeypatch.delenv("GPUWM_FETCH_BANDWIDTH_MBPS", raising=False)
    assert fetch_pool.worker_profile()["id"] == "runpod"
    monkeypatch.setenv("GPUWM_FETCH_BANDWIDTH_MBPS", "125")
    assert fetch_pool.worker_profile()["id"] == "wide"
    monkeypatch.setenv("GPUWM_FETCH_BANDWIDTH_MBPS", "nan")
    with pytest.raises(ValueError, match="finite"):
        fetch_pool.worker_profile()


def test_measured_box_row_is_selected_only_with_matching_link_budget(monkeypatch):
    monkeypatch.delenv("GPUWM_FETCH_ROW", raising=False)
    monkeypatch.setenv("GPUWM_FETCH_PROVIDER", "aws")
    monkeypatch.setenv("GPUWM_FETCH_BOX", "c7a.48xlarge")
    monkeypatch.setenv("GPUWM_FETCH_BANDWIDTH_MBPS", "125")
    assert fetch_pool.worker_profile()["id"] == "aws-c7a-high-bandwidth"
    assert fetch_pool.resolve_file_workers(None) == 12
    monkeypatch.setenv("GPUWM_FETCH_BANDWIDTH_MBPS", "10")
    assert fetch_pool.worker_profile()["id"] == "constrained"


@pytest.fixture
def range_origin():
    payload = grib(5 * 1024 * 1024 + 17) + grib(29)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            source = payload[:-1] if self.path == "/truncated" else payload
            raw = self.headers.get("Range")
            if raw:
                first, last = raw.removeprefix("bytes=").split("-", 1)
                start, end = int(first), int(last) if last else len(source) - 1
                end = min(end, len(source) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(source)}")
            else:
                start, end = 0, len(source) - 1
                self.send_response(200)
            data = source[start:end+1]
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", payload
    server.shutdown()
    worker.join()
    server.server_close()


def test_real_rust_object_stream_and_truncation(tmp_path, range_origin):
    binary = os.environ.get("GPUWM_RW_FETCH")
    if not binary:
        pytest.skip("S3 check sets GPUWM_RW_FETCH to the newly built backbone")
    base, payload = range_origin
    path = tmp_path / "verified.grib2"
    command = [binary, "object", "--url", base + "/complete", "--output", str(path)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes() == payload
    assert receipt["grib2_messages"] == 2
    assert receipt["publication"] == publication(path)
    assert time.time() - receipt["last_byte_unix"] < 3
    path = tmp_path / "truncated.grib2"
    result = subprocess.run([binary, "object", "--url", base + "/truncated",
        "--output", str(path)], text=True, capture_output=True, timeout=30)
    assert result.returncode == 1
    assert "truncated.grib2" in result.stderr and "truncated GRIB2" in result.stderr
    assert not path.exists() and not path.with_name(path.name + ".part").exists()


def test_pool_rust_transport_relays_bytes_before_admission(tmp_path, range_origin):
    if not os.environ.get("GPUWM_RW_FETCH"):
        pytest.skip("S3 check sets the newly built backbone")
    plan = fetch_routes.resolve_request("hrrr-prs", cycle=datetime(2026,10,3,12),
        hours=1, out=tmp_path)
    download = fetch._route_stream_downloader(plan, SimpleNamespace(fetch_workers=1))
    seen = []
    class Monitor:
        def begin(self, **kwargs): pass
        def start(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
        def relay(self, name):
            return lambda received,total: seen.append((name,received,total))
    origin,payload = range_origin
    entries,_ = fetch_pool.run_transfers([fetch_pool.TransferJob("visible",origin,
        lambda:download(origin+"/complete",tmp_path/"visible.grib2",magic="GRIB"))],
        workers=1,monitor=Monitor())
    assert seen and seen[-1] == ("visible",len(payload),len(payload))
    assert entries[0]["bytes"] == len(payload)


def test_real_route_download_quarantines_changed_destination(tmp_path, range_origin):
    if not os.environ.get("GPUWM_RW_FETCH"):
        pytest.skip("S3 check sets the newly built backbone")
    plan = fetch_routes.resolve_request("hrrr-prs", cycle=datetime(2026, 10, 3, 12),
        hours=1, out=tmp_path)
    download = fetch._route_stream_downloader(plan, SimpleNamespace(fetch_workers=2))
    assert download is not None
    dest = tmp_path / "changed.grib2"
    dest.write_bytes(b"changed input")
    origin, payload = range_origin
    entry = download(origin + "/complete", dest, magic="GRIB")
    assert dest.read_bytes() == payload
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert set(entry) == {"name", "bytes", "sha256", "url"}
    assert fetch._fetched_records(dest) == 2
    assert [p.read_bytes() for p in tmp_path.glob("changed.grib2.rejected-*")] == [b"changed input"]


def test_substituted_route_transport_stays_in_control(tmp_path, monkeypatch):
    plan = fetch_routes.resolve_request("hrrr-prs", cycle=datetime(2026, 10, 3, 12),
        hours=1, out=tmp_path)
    monkeypatch.setattr(fetch_routes, "_download_object", lambda *args, **kwargs: {})
    assert fetch._route_stream_downloader(plan, SimpleNamespace(fetch_workers=None)) is None
