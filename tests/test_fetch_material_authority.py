"""Acquisition observations must not change the preparation's input authority."""
from datetime import datetime
import hashlib
import json

from gpuwm import fetch_routes
from test_fetch_routes import _fake_downloader


def test_pool_width_and_cached_timing_do_not_change_input_authority(tmp_path):
    plan = fetch_routes.resolve_request("rap", cycle=datetime(2026, 8, 16), hours=1)
    first = fetch_routes.run_plan(plan, out=tmp_path, file_workers=1,
                                 downloader=_fake_downloader([]), progress=lambda *_: None)
    authority = tmp_path / fetch_routes.MANIFEST_NAME
    before = authority.read_bytes()
    before_receipt = json.loads((tmp_path / "fetch-pool-receipt.json").read_text())
    second = fetch_routes.run_plan(plan, out=tmp_path, file_workers=2,
                                  downloader=_fake_downloader([]), progress=lambda *_: None)
    after_receipt = json.loads((tmp_path / "fetch-pool-receipt.json").read_text())
    assert first["concurrency"]["workers_requested"] == before_receipt["workers_requested"] == 1
    assert second["concurrency"]["workers_requested"] == after_receipt["workers_requested"] == 2
    assert before == authority.read_bytes()
    assert "concurrency" not in json.loads(before)
    assert all("reused" not in entry for entry in json.loads(before)["files"])
    assert all(entry["reused"] is False for entry in first["files"])
    assert all(entry["reused"] is True for entry in second["files"])


def test_changed_material_digest_changes_authority_and_partial_stays_partial(tmp_path):
    plan = fetch_routes.resolve_request("rap", cycle=datetime(2026, 8, 16), hours=0)
    payload = fetch_routes.run_plan(plan, out=tmp_path, downloader=_fake_downloader([]),
                                   progress=lambda *_: None, on_lead=lambda *_: None)
    authority = tmp_path / fetch_routes.MANIFEST_NAME
    before = hashlib.sha256(authority.read_bytes()).hexdigest()
    changed = dict(payload, files=[dict(payload["files"][0], sha256="f" * 64)])
    fetch_routes._write_route_manifest(tmp_path, changed)
    assert hashlib.sha256(authority.read_bytes()).hexdigest() != before
    fetch_routes._write_route_manifest(tmp_path, dict(changed, complete=False, concurrency=None))
    assert json.loads(authority.read_text())["complete"] is False
