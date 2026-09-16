"""Size selects the transfer mechanism; it never ends the request.

CPU-only: bytes are protocol fixtures, nothing is fetched and no run starts.
"""
import base64
import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm import remote_plan as rp, remote_worker as rw


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def saved(tmp_path):
    from gpuwm import domain_wizard as dw
    config = tmp_path / "saved map.toml"
    config.write_text(dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
        hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6, hours=24,
                         out=str(tmp_path / "forcing"), cadence=3), case_data=None), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": str(config)},
        "output_root": str(tmp_path / "local-output"), "run_options": {"render_products": "none"}}),
        encoding="utf-8")
    return config, plan


def build(saved):
    config, plan = saved
    return rp.build_bundle(plan, workspace="/node/work", outdir="/node/work/new-output",
        geog_root="/node/geography", expected_plan_sha256=sha(plan.read_bytes()),
        expected_config_sha256=sha(config.read_bytes()))


def test_an_oversize_companion_takes_the_verified_object_route(saved, tmp_path):
    config, _plan = saved
    companion = config.with_suffix(".d01-target.json")
    raw = b"x" * (200 * 1024)
    companion.write_bytes(raw)
    document = build(saved)
    assert document["schema"] == rp.BLOB_BUNDLE_SCHEMA
    entry = next(item for item in document["blobs"] if item["name"] == "case.d01-target.json")
    assert entry["placement"] == "inputs"
    assert entry["size"] == len(raw) and entry["sha256"] == sha(raw)
    # Nothing inline carries its payload.
    assert all(item["name"] != "case.d01-target.json" for item in document["files"])
    rewrite = next(item for item in document["rewrites"] if item["after"].endswith("case.d01-target.json"))
    assert "verified SHA-256 object transfer" in rewrite["basis"]


def test_the_object_route_lands_where_an_inline_companion_would_have(saved, tmp_path):
    from gpuwm import remote_input_transfer as transfer
    config, _plan = saved
    companion = config.with_suffix(".d01-target.json")
    companion.write_bytes(b"y" * (200 * 1024))
    document = build(saved)
    document["workspace"] = str(tmp_path.resolve())
    document["sha256"] = rp._sha(rp._encoded({k: v for k, v in document.items() if k != "sha256"}))
    for item in document["blobs"]:
        transfer.receive({"schema": "gpuwm.remote.request.v1", "action": "put-input",
                          "workspace": str(tmp_path), "size": item["size"], "sha256": item["sha256"]},
                         tmp_path, __import__("io").BytesIO(Path(item["source_path"]).read_bytes()))
    rp.stage(document, tmp_path)
    _bundle, directory = rp.read_bundle(tmp_path, document["id"], document["sha256"])
    assert (directory / "case.d01-target.json").read_bytes() == companion.read_bytes()


def test_a_rewritten_input_cannot_stream_and_the_refusal_says_why(saved, tmp_path):
    config, _plan = saved
    wps = config.with_suffix(".namelist.wps")
    body = "&share\n max_dom=1,\n/\n&geogrid\n dx=12000.,\n dy=12000.,\n"
    wps.write_text(body + "".join(f" comment_{index} = 1,\n" for index in range(12000)) + "/\n")
    assert wps.stat().st_size > rp.MAX_SINGLE_BYTES
    with pytest.raises(ValueError) as failure:
        build(saved)
    message = str(failure.value)
    assert wps.name in message
    assert "rewritten during staging" in message
    assert "cannot travel as a verified copy of it" in message
    assert f"{rp.MAX_SINGLE_BYTES:,} bytes" in message


def test_the_manifest_bound_comes_from_the_nodes_own_request_read():
    assert rp.MAX_MANIFEST_BYTES < rw.MAX_BYTES


def test_the_manifest_refusal_names_the_measured_bytes_and_the_maximum(saved, monkeypatch):
    monkeypatch.setattr(rp, "MAX_MANIFEST_BYTES", 16)
    with pytest.raises(ValueError) as failure:
        build(saved)
    message = str(failure.value)
    assert "bytes" in message and "16 bytes" in message
    assert "node reads at most" in message


def test_the_inline_manifest_refusal_names_its_measured_bytes(saved, monkeypatch):
    monkeypatch.setattr(rp, "MAX_INPUT_BYTES", 8)
    with pytest.raises(ValueError) as failure:
        build(saved)
    message = str(failure.value)
    assert "8 bytes" in message and "documents" in message
    assert "existing remote-input route" in message


def test_the_review_states_the_measured_manifest_size(saved, tmp_path, monkeypatch):
    monkeypatch.setattr("gpuwm.remote_plan.memory_review", lambda *_a, **_k:
        {"measured": False, "free_bytes": None, "refuse": False, "warn": True, "verdict": "CPU-only"})
    config, plan = saved
    node_output = tmp_path / "node-output"
    node_output.mkdir()
    document = rp.build_bundle(plan, workspace=str(tmp_path.resolve()),
        outdir=str(node_output / "new-run"), geog_root=str(tmp_path),
        expected_plan_sha256=sha(plan.read_bytes()), expected_config_sha256=sha(config.read_bytes()))
    rp.stage(document, tmp_path)
    value, _bundle, _directory = rp.review(
        {"bundle_id": document["id"], "expected_bundle_sha256": document["sha256"]}, tmp_path)
    assert value["manifest_bytes"] == len(rp._encoded(document))
    assert value["manifest_maximum_bytes"] == rp.MAX_MANIFEST_BYTES
