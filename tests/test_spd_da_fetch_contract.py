"""Fetch safety contracts required by the Rust transport migration.

These exercise the current consumer guards. They are not bench timings or
proof that a replacement Rust object interface exists.
"""
import json

import pytest

from gpuwm.ingest.boundary_stream import read_replaced_json
from tools import da_recent_observations as obs
from test_da_recent_observations import contribution, obj, volume


@pytest.mark.parametrize("payload,etag,reason", [
    (b"abc", '"replaced"', "ETag differs"),
    (b"ab", '"e"', "size differs"),
    (b"abcd", '"e"', "size differs|grew beyond"),
])
def test_replaced_or_wrong_size_object_is_never_published(
        tmp_path, monkeypatch, payload, etag, reason):
    from da_fetch_origin import origin
    target = tmp_path / "object"
    with origin(monkeypatch, payload, etag):
        with pytest.raises(ValueError, match=reason):
            obs._download(obj("noaa-mrms-pds", "a/data"), target)
    assert not target.exists()


def test_large_object_uses_verified_ranges_and_hashes_written_bytes(tmp_path, monkeypatch):
    from da_fetch_origin import origin
    payload = b"abcd" * (5 * 1024 * 1024)
    row = obj("noaa-gfs-bdp-pds", "a/object", len(payload))
    with origin(monkeypatch, payload) as server:
        result = obs._download(row, tmp_path / "object")
    assert result["sha256"] == obs.sha(tmp_path / "object")
    assert [span for _, span, _ in server.gets] == ["bytes=0-16777215", "bytes=16777216-20971519"]
    assert all(etag == '"e"' for _, _, etag in server.gets)


def test_compressed_object_is_preserved_as_published_bytes(tmp_path, monkeypatch):
    import gzip
    import hashlib
    from da_fetch_origin import origin
    payload = gzip.compress(b"opaque public object fixture", mtime=0)
    with origin(monkeypatch, payload) as server:
        server.encoding = "gzip"
        result = obs._download(obj("noaa-mrms-pds", "a/object.gz", len(payload)), tmp_path / "object.gz")
    assert (tmp_path / "object.gz").read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("fault", ["wrong_range", "ignore_range"])
def test_large_object_refuses_a_foreign_or_ignored_range(tmp_path, monkeypatch, fault):
    from da_fetch_origin import origin
    payload = b"x" * (16 * 1024 * 1024 + 1)
    with origin(monkeypatch, payload) as server:
        setattr(server, fault, True)
        with pytest.raises(ValueError, match="Content-Range|verified transfer"):
            obs._download(obj("noaa-gfs-bdp-pds", "a/object", len(payload)), tmp_path / "object")
    assert not (tmp_path / "object").exists()


def test_newest_first_decode_stops_at_first_complete_causal_volume(tmp_path):
    analysis = obs.utc("2026-10-03T22:00:00Z")
    keys = [f"2026/10/03/KAAA/KAAA20261003_{clock}_V06"
            for clock in ("215900", "215800", "215700")]
    rows = {}
    for key in keys:
        row = obj("unidata-nexrad-level2", key)
        row.update(path=str(tmp_path / key.rsplit("/", 1)[1]),
                   sha256="a" * 64, last_modified="2026-10-03T21:59:30Z")
        rows[(row["bucket"], key)] = row
    decoded = []

    def decode(binary, *, volume, out, **kwargs):
        decoded.append(volume.name)
        out.write_bytes(b"pack fixture")

    def read_pack(path):
        # The newest filename starts before cutoff but its measured end is
        # in the future. The next complete volume is causal.
        return volume(end_time=("2026-10-03T22:00:01" if "215900" in path.name
                                else "2026-10-03T21:59:00"))

    result, source, refusals = obs._site_volume(
        "KAAA", keys, rows, {}, analysis, tmp_path, "native", None, None,
        decode, lambda *args, **kwargs: {"status": "PASS"}, read_pack,
        lambda *args, **kwargs: contribution())
    assert result is not None and source["key"] == keys[1]
    assert decoded == [key.rsplit("/", 1)[1] for key in keys[:2]]
    assert refusals == [{"site": "KAAA", "key": keys[0],
                         "reason": "radar volume finishes after analysis cutoff"}]


def test_pending_truth_cannot_use_existing_complete_receipt_contract(tmp_path):
    radar = obj("unidata-nexrad-level2", "a/radar")
    truth = obj("noaa-mrms-pds", "a/truth")
    inv = {"objects": [radar, truth]}
    path = tmp_path / "radar"
    path.write_bytes(b"abc")
    fetched = {"schema": "da-rerun.recent-fetch.v1", "status": "complete",
               "inventory_sha256": obs.document_sha(inv),
               "objects": [{**radar, "path": str(path), "sha256": obs.sha(path)}]}
    with pytest.raises(ValueError, match="object roster differs"):
        obs.validate_fetched(inv, fetched)


def test_inventory_consumes_rust_metadata_as_it_arrives(tmp_path, monkeypatch):
    from da_fetch_origin import origin
    from test_da_recent_observations import case
    received = []
    with origin(monkeypatch) as server:
        result = obs.inventory(case(), first_slot_only=True, on_object=received.append)
    assert len(server.listings) == 2
    assert sorted(received, key=lambda row: row["key"]) == result["objects"]
    assert len(result["slots"]) == 1


def deferred_inventory():
    from test_da_recent_observations import case
    doc = case()
    key = "2026/10/03/KAAA/KAAA20261003_215500_V06"
    radar = obj("unidata-nexrad-level2", key)
    radar["last_modified"] = "2026-10-03T21:59:00Z"
    truth = obj("noaa-mrms-pds", "a/truth")
    return {"case": doc, "slots": [{"radar_candidates": {"KAAA": [key]}}],
            "objects": [radar, truth]}


def test_background_truth_failure_does_not_delay_or_invalidate_forecast(tmp_path, monkeypatch):
    import time
    from da_fetch_origin import origin
    from tools.da_object_fetch import truth_receipt
    inv = deferred_inventory()
    receipt = tmp_path / "fetch.json"
    with origin(monkeypatch) as server:
        server.gate.clear()
        server.fail = True
        try:
            obs.fetch(inv, tmp_path / "raw", receipt, 1)
            forecast = json.loads(receipt.read_text())
            assert forecast["schema"] == "da-rerun.recent-fetch.v2"
            obs.validate_fetched(inv, forecast)
            truth = json.loads(open(forecast["truth"]["receipt"]).read())
            assert truth["status"] == "pending"
            with pytest.raises(ValueError, match="scoring cannot read"):
                truth_receipt(inv, forecast)
        finally:
            server.gate.set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            truth = json.loads(open(forecast["truth"]["receipt"]).read())
            if truth["status"] != "pending":
                break
            time.sleep(.05)
        assert truth["status"] == "failed"
        assert server.gets and all(path == "/a/truth" for path, _, _ in server.gets)
        obs.validate_fetched(inv, forecast)


def test_truth_worker_launch_failure_never_blocks_forecast(tmp_path, monkeypatch):
    from tools import da_object_fetch
    def cannot_launch(*args, **kwargs):
        raise OSError("worker launch unavailable")
    monkeypatch.setattr(da_object_fetch.subprocess, "Popen", cannot_launch)
    inv = deferred_inventory()
    receipt = tmp_path / "fetch.json"
    obs.fetch(inv, tmp_path / "raw", receipt, 1)
    forecast = json.loads(receipt.read_text())
    obs.validate_fetched(inv, forecast)
    assert json.loads(open(forecast["truth"]["receipt"]).read())["status"] == "failed"


def test_lazy_radar_fetches_newest_causal_volume_only(tmp_path, monkeypatch):
    from da_fetch_origin import origin
    analysis = obs.utc("2026-10-03T22:00:00Z")
    keys = [f"2026/10/03/KAAA/KAAA20261003_{clock}_V06"
            for clock in ("215900", "215800", "215700")]
    rows = {}
    for key in keys:
        row = obj("unidata-nexrad-level2", key)
        row.update(destination=str(tmp_path / key.rsplit("/", 1)[1]),
                   last_modified="2026-10-03T21:59:30Z")
        rows[(row["bucket"], key)] = row
    def decode(binary, *, out, **kwargs):
        out.write_bytes(b"pack fixture")
    def read_pack(path):
        return volume(end_time=("2026-10-03T22:00:01" if "215900" in path.name
                                else "2026-10-03T21:59:00"))
    with origin(monkeypatch) as server:
        _, source, _ = obs._site_volume("KAAA", keys[::-1], rows, {}, analysis,
            tmp_path, "native", None, None, decode,
            lambda *args, **kwargs: {"status": "PASS"}, read_pack,
            lambda *args, **kwargs: contribution())
    assert source["key"] == keys[1]
    assert [path for path, _, _ in server.gets] == ['/'+key for key in keys[:2]]
    assert not (tmp_path / keys[2].rsplit('/', 1)[1]).exists()


def test_truth_binding_survives_verified_lazy_completion():
    from tools.da_object_fetch import fetch_binding
    fetched = {"schema": "da-rerun.recent-fetch.v2", "scope": "forecast",
               "inventory_sha256": "a"*64, "raw_root": "/owned/raw",
               "truth": {"receipt": "/owned/truth.json"}, "objects": [],
               "deferred_objects": [{"key": "radar"}]}
    before = fetch_binding(fetched)
    fetched["objects"] = [{"key": "radar", "sha256": "b"*64}]
    fetched["deferred_objects"] = []
    assert fetch_binding(fetched) == before
    fetched["truth"]["pid"] = 12345
    fetched["truth"]["log"] = "/owned/new-worker.log"
    assert fetch_binding(fetched) == before
    fetched["inventory_sha256"] = "c"*64
    assert fetch_binding(fetched) != before


@pytest.mark.parametrize("claim", ["incomplete", "stale"])
def test_optional_truth_claim_refusal_does_not_wait_or_fail_forecast(tmp_path, claim):
    import time
    from tools.da_object_fetch import launch_truth_for_forecast
    target = tmp_path / "truth"
    target.mkdir()
    path = target / "truth-job.json"
    path.write_text("" if claim == "incomplete" else json.dumps({"other_inputs": True}))
    started = time.monotonic()
    result = launch_truth_for_forecast({}, {}, target, 0, 0)
    assert time.monotonic() - started < 1
    assert result["status"] == "refused"
    assert "stale truth" in result["prevents"]
    assert not (target / "background-fetch.json").exists()


def test_real_fetch_emits_paired_stage_events(tmp_path, monkeypatch):
    from da_fetch_origin import origin
    ledger = tmp_path / "timeline.jsonl"
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(ledger))
    with origin(monkeypatch):
        obs.fetch({"objects": [obj("noaa-gfs-bdp-pds", "a/object")]},
                  tmp_path / "raw", tmp_path / "fetch.json", 1)
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [event["phase"] for event in events] == ["begin", "end"]
    assert all(event["stage"] == "fetch.inputs" for event in events)
    assert events[0]["attrs"]["span_id"] == events[1]["attrs"]["span_id"]


def test_truth_only_receipt_cannot_bypass_forecast_roster():
    with pytest.raises(ValueError, match="required radar/model sources are absent"):
        obs.validate_fetched({}, {"scope": "truth"})


def test_deferred_cache_io_failure_releases_forecast_lock(tmp_path, monkeypatch):
    inv = deferred_inventory()
    inv["objects"] = inv["objects"][:1]
    row = inv["objects"][0]
    root = tmp_path / "raw"
    path = obs.raw_path(root, row)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"abc")
    ancestor = {"schema": "da-rerun.recent-fetch.v1", "status": "complete",
                "objects": [{**row, "path": str(path), "sha256": obs.sha(path),
                             "fetched_at_utc": "2026-10-03T21:59:00Z"}]}
    def unreadable(*args):
        raise OSError("cache read failed")
    monkeypatch.setattr(obs, "sha", unreadable)
    with pytest.raises(OSError, match="cache read failed"):
        obs.fetch(inv, root, tmp_path/"fetch.json", 1, reuse_receipt=ancestor,
                  reuse_receipt_sha256="a"*64)
    assert not (root / ".recent-observation-fetch.lock").exists()


def test_background_receipt_updates_are_atomic_for_readers(tmp_path):
    import threading
    path = tmp_path / "truth-status.json"
    obs.save(path, {"status": "pending", "rows": []})
    finished = threading.Event()
    errors = []
    def update():
        try:
            for count in range(12):
                obs.save(path, {"status": "complete", "rows": list(range(10000)), "update": count})
        except BaseException as error:        # noqa: BLE001 - the thread's verdict is asserted below
            errors.append(error)
        finally:
            finished.set()
    writer = threading.Thread(target=update)
    writer.start()
    try:
        while not finished.is_set():
            # The reader join_truth uses: on Windows an open that meets the
            # replace is refused for a moment and retried; it never reads a
            # torn revision.
            assert read_replaced_json(path)["status"] in ("pending", "complete")
    finally:
        writer.join()
    # The writer must outlast the reader's opens as well: on the hosted
    # Windows runner (2.9.2 publish) it gave up after 0.50 s of backoff with
    # PermissionError, which pytest only reported as a thread warning.
    assert not errors, errors
