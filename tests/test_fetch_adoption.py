"""Unmanaged files enter the normal route recovery path only after verification."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import json
import threading
from urllib.error import URLError

import pytest

from gpuwm import fetch, fetch_adoption, fetch_routes
from gpuwm.chain_events import GoChainEvents


def _plan(source="hrrr-prs"):
    route = fetch_routes.route_for(source)
    plan = fetch_routes.resolve_request(
        source, cycle=datetime(2026, 10, 18, route.cycle_hours[0]), hours=0)
    endpoint = replace(plan.ladder[0], name="archive",
                       base="https://test.s3.amazonaws.com")
    obj = replace(plan.objects[0], member_index=None,
                  index_key=plan.objects[0].key + ".idx")
    return replace(plan, host=endpoint, ladder=(endpoint,), objects=(obj,),
                   compose=(), donors=(), primary_files=(obj.relpath,))


def _body(magic="GRIB", value=b"original"):
    if magic != "GRIB":
        return magic.encode("ascii") + value
    size = 16 + len(value) + 4
    return b"GRIB\0\0\0\2" + size.to_bytes(8, "big") + value + b"7777"


def _host(monkeypatch, body, *, etag=None, index=b"1:0:message:\n", size=None):
    requests = []

    def answer(url, *, method="GET", opener=None):
        requests.append((method, url))
        assert method == "HEAD" or url.endswith(".idx")
        response = io.BytesIO(index)
        response.headers = {"Content-Length": str(len(body) if size is None else size),
                            "ETag": '"' + (etag or hashlib.md5(body).hexdigest()) + '"'}
        return response

    monkeypatch.setattr(fetch_adoption, "_request", answer)
    return requests


def _run(plan, out, downloader=None, **kwargs):
    return fetch_routes.run_plan(plan, out=out, downloader=downloader,
                                 probe=lambda _: False, progress=lambda _: None, **kwargs)


@pytest.mark.parametrize("source", fetch_routes.route_ids())
def test_adopts_matching_unrecorded_file_for_each_route(tmp_path, monkeypatch, source):
    plan = _plan(source)
    obj = plan.objects[0]
    body = _body(fetch_routes._magic_for(plan, obj.role))
    dest = tmp_path / obj.relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    requests = _host(monkeypatch, body)

    def no_download(*args, **kwargs):
        pytest.fail("a matching local file was downloaded")

    admitted = []
    payload = _run(plan, tmp_path, no_download,
                   on_lead=lambda *args: admitted.append(args))
    assert len(admitted) == 1
    entry = payload["files"][0]
    assert entry["adopted"] and not entry["reused"] and not entry["downloaded"]
    assert entry["sha256"] == hashlib.sha256(body).hexdigest()
    record = json.loads(fetch_routes._recovery_path(tmp_path, obj).read_text())
    assert record["file"]["adoption"]["method"] == "size-md5"
    assert requests == [("HEAD", entry["url"])]
    receipt = json.loads((tmp_path / "fetch-pool-receipt.json").read_text())
    assert [receipt[key] for key in ("adopted_files", "downloaded_files", "reused_files")] == [1, 0, 0]
    observer = GoChainEvents()
    observer._data_dir = tmp_path
    assert observer._fetch_fields()["adopted_files"] == 1
    authority = (tmp_path / fetch_routes.MANIFEST_NAME).read_bytes()
    second = _run(plan, tmp_path, no_download, on_lead=lambda *args: None)
    assert second["files"][0]["reused"] and not second["files"][0]["adopted"]
    assert len(requests) == 1
    assert (tmp_path / fetch_routes.MANIFEST_NAME).read_bytes() == authority
    assert fetch.fetch_throughput(tmp_path)["reused_files"] == 1
    assert fetch.fetch_throughput(tmp_path)["adopted_files"] == 0


@pytest.mark.parametrize("mismatch", ["md5", "size", "envelope"])
def test_rejects_and_downloads_mismatched_unrecorded_file(tmp_path, monkeypatch, mismatch):
    plan = _plan()
    obj = plan.objects[0]
    body = _body()
    damaged = body.replace(b"original", b"modified")
    if mismatch == "size":
        damaged += b"7777"
    elif mismatch == "envelope":
        damaged = damaged[:-4] + b"0000"
    dest = tmp_path / obj.relpath
    dest.write_bytes(damaged)
    _host(monkeypatch, body)
    downloads = []

    def download(url, target, **kwargs):
        downloads.append(url)
        target.write_bytes(body)
        return {"name": target.name, "url": url, "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest()}

    payload = _run(plan, tmp_path, download)
    assert len(downloads) == 1
    assert payload["files"][0]["downloaded"] and not payload["files"][0]["adopted"]
    assert dest.read_bytes() == body
    rejected = list(tmp_path.glob(dest.name + ".rejected-*"))
    assert len(rejected) == 1 and rejected[0].read_bytes() == damaged
    receipt = fetch.fetch_throughput(tmp_path)
    assert receipt["downloaded_files"] == 1 and receipt["adopted_files"] == 0


@pytest.mark.parametrize("index,adopted", [
    (b"1:0:first:\n2:28:last:\n", True),
    (b"1:0:first:\n2:29:last:\n", False),
    (b"1:0:first:\n", False),
])
def test_multipart_etag_requires_host_index_count_and_offsets(tmp_path, monkeypatch, index, adopted):
    plan = _plan()
    body = _body() * 2
    assert len(body) == 56
    dest = tmp_path / plan.objects[0].relpath
    dest.write_bytes(body)
    # An untrusted local index must not substitute for the host's index.
    dest.with_suffix(dest.suffix + ".idx").write_bytes(b"1:0:untrusted:\n")
    requests = _host(monkeypatch, body, etag="0" * 32 + "-2", index=index)
    result = fetch_adoption.adopt(dest, plan.objects[0], ladder=plan.ladder,
        magic="GRIB", verify_payload=fetch_routes._verify_payload, progress=lambda _: None)
    assert (result is not None) == adopted
    assert [method for method, _ in requests] == ["HEAD", "GET"]
    if result:
        assert result["adoption"]["messages"] == 2
        assert result["adoption"]["last_byte_offset"] == 28


def test_host_failure_keeps_existing_download_path(tmp_path, monkeypatch):
    plan = _plan()
    body = _body()
    dest = tmp_path / plan.objects[0].relpath
    dest.write_bytes(body)

    def unavailable(*args, **kwargs):
        raise URLError("offline test")

    monkeypatch.setattr(fetch_adoption, "_request", unavailable)
    assert fetch_adoption.adopt(dest, plan.objects[0], ladder=plan.ladder,
        magic="GRIB", verify_payload=fetch_routes._verify_payload, progress=lambda _: None) is None
    assert dest.read_bytes() == body


def test_checksum_mismatch_does_not_try_a_weaker_mirror(tmp_path, monkeypatch):
    plan = _plan()
    body = _body()
    dest = tmp_path / plan.objects[0].relpath
    dest.write_bytes(body.replace(b"original", b"modified"))
    requests = _host(monkeypatch, body)
    mirror = replace(plan.ladder[0], name="mirror", base="https://mirror.invalid")
    assert fetch_adoption.adopt(dest, plan.objects[0], ladder=(*plan.ladder, mirror),
        magic="GRIB", verify_payload=fetch_routes._verify_payload, progress=lambda _: None) is None
    assert len(requests) == 1


def test_local_verification_runs_in_parallel(tmp_path, monkeypatch):
    plan = _plan()
    first = plan.objects[0]
    second = replace(first, name="second.grib2", relpath="second.grib2", key="second.grib2")
    plan = replace(plan, objects=(first, second), primary_files=(first.relpath, second.relpath))
    body = _body()
    for obj in plan.objects:
        (tmp_path / obj.relpath).write_bytes(body)
    barrier = threading.Barrier(2)
    _host(monkeypatch, body)
    request = fetch_adoption._request

    def parallel(*args, **kwargs):
        barrier.wait(timeout=5)
        return request(*args, **kwargs)

    monkeypatch.setattr(fetch_adoption, "_request", parallel)
    monkeypatch.setattr(fetch_routes.os, "cpu_count", lambda: 2)
    assert _run(plan, tmp_path)["concurrency"]["adopted_files"] == 2


def test_parallel_verification_caps_open_responses_at_sixteen(tmp_path, monkeypatch):
    plan = _plan()
    body = _body()
    objects = tuple(replace(plan.objects[0], name=f"{i}.grib2", relpath=f"{i}.grib2",
                            key=f"{i}.grib2") for i in range(24))
    plan = replace(plan, objects=objects, primary_files=tuple(obj.relpath for obj in objects))
    for obj in objects:
        (tmp_path / obj.relpath).write_bytes(body)
    reached = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = maximum = 0

    class Response(io.BytesIO):
        def __enter__(self):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active == 16:
                    reached.set()
            assert release.wait(5)
            return self

        def __exit__(self, *args):
            nonlocal active
            with lock:
                active -= 1
            return super().__exit__(*args)

    def answer(*args, **kwargs):
        response = Response()
        response.headers = {"Content-Length": str(len(body)),
                            "ETag": '"' + hashlib.md5(body).hexdigest() + '"'}
        return response

    monkeypatch.setattr(fetch_adoption, "_request", answer)
    monkeypatch.setattr(fetch_routes.os, "cpu_count", lambda: 24)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run, plan, tmp_path)
        try:
            assert reached.wait(5)
            assert maximum == 16
        finally:
            release.set()
        receipt = future.result()["concurrency"]
    assert receipt["adopted_files"] == 24
    assert receipt["verification_workers"] == 24
    assert maximum == 16 and active == 0
