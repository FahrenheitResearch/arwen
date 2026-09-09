"""Native plot scheduling and transfer protocol; fixtures are not weather."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import remote_native_plots as plots, remote_processed_v2 as viewer, remote_preparation_v2 as preparation
from gpuwm import remote_artifacts as ra, remote_cli
from test_remote_artifacts import case, encoded
from test_remote_processed_v2 import native, query


@pytest.fixture
def prepared(native, monkeypatch):
    c = native.case
    viewer.catalog(query(c, products=viewer.DEFAULT_PRODUCTS), c.tmp_path)
    viewer._work_job(c.tmp_path, c.record["id"])
    monkeypatch.setattr(plots, "ensure", lambda *_: None)
    monkeypatch.setattr(preparation, "ensure", lambda *_: None)
    monkeypatch.setattr(plots, "_spacing", lambda *_: 3000.)
    renders = []
    def render(command, **kwargs):
        assert "--process-request" not in command
        request = json.loads(Path(command[command.index("--render-store-request") + 1]).read_text())
        renders.append(request)
        native_result = json.loads(Path(request["process_result"]).read_text())
        root = Path(request["out_dir"])
        root.mkdir()
        panels = []
        for slug in request["products"]:
            path = root / (slug + ".png")
            path.write_bytes(b"PNG transfer protocol fixture; no meteorology")
            panels.append({"slug": slug, "path": str(path), "sha256": ra._file_sha(path), "bytes": path.stat().st_size})
        result = {"schema": "arwen.native-store-render-result.v1", "frame_id": native_result["frame"]["id"],
            "identity": native_result["frame"]["identity"], "source_store_sha256": native_result["frame"]["rws_sha256"],
            "wrf_imported": False, "volume_store_created": False, "panels": panels}
        Path(command[-1]).write_bytes(encoded(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(plots.subprocess, "run", render)
    return SimpleNamespace(case=c, renders=renders, native=native)


def test_incremental_plots_reuse_compact_store_and_do_not_retry_published_frames(prepared):
    c = prepared.case
    assert plots.work_once(c.tmp_path, c.record["id"])["pending"] == 1
    assert len(prepared.renders) == 1 and len(prepared.native.calls) == 1
    assert prepared.renders[0]["products"] == sorted(viewer.DEFAULT_PRODUCTS)
    assert prepared.renders[0]["width"] == 1200 and prepared.renders[0]["height"] == 900
    c.status["state"] = "completed"
    state = plots.work_once(c.tmp_path, c.record["id"])
    assert state["done"] and state["ready"] == 1 and state["panels"] == 20
    assert len(prepared.renders) == 1 and c.frame.read_bytes() == c.raw


def test_selected_gallery_roundtrip_reuses_bytes_and_rejects_tampered_commit(prepared, monkeypatch):
    c = prepared.case
    plots.work_once(c.tmp_path, c.record["id"])
    monkeypatch.setattr(remote_cli, "_transport", lambda _, request, **kw: {"ok": True, "native_plots": plots.catalog(request, c.tmp_path)})
    transfers = []
    def download(_, request, path, frame):
        output = io.BytesIO()
        plots.stream(request, c.tmp_path, output)
        assert len(output.getvalue()) == frame["size_bytes"]
        path.write_bytes(output.getvalue())
        transfers.append(request)
    monkeypatch.setattr(ra, "_download", download)
    args = SimpleNamespace(workspace=str(c.tmp_path), job=c.record["id"], domain=1, sequence=1, cache_root=c.tmp_path / "local-plots")
    first = plots.sync(args, [], [])
    assert len(transfers) == 20 and first["transferred_bytes"] > 0
    gallery = Path(first["native_plots"]["gallery_path"])
    assert ra._file_sha(gallery) == first["native_plots"]["gallery_sha256"]
    assert gallery.read_text(encoding="utf-8").count("<figure>") == 20
    assert plots.sync(args, [], [])["transferred_bytes"] == 0 and len(transfers) == 20
    changed = dict(transfers[0], expected_commit_sha256="f" * 64)
    with pytest.raises(ValueError, match="authority changed"):
        plots.stream(changed, c.tmp_path, io.BytesIO())
    changed = dict(transfers[0], product="../../private")
    with pytest.raises(ValueError, match="product or checksum"):
        plots.stream(changed, c.tmp_path, io.BytesIO())


def test_source_mutation_is_refused_and_reported_once(prepared):
    c = prepared.case
    c.frame.write_bytes(c.raw + b"changed")
    plots.work_once(c.tmp_path, c.record["id"])
    c.status["state"] = "completed"
    state = plots.work_once(c.tmp_path, c.record["id"])
    assert state["done"] and state["failed"] == 1 and not prepared.renders


def test_native_plot_cli_requires_exact_sequence_and_cache():
    from gpuwm.cli import build_parser
    args = build_parser().parse_args(["remote", "sync-native-plots", "--host", "node-1", "--python", "/python",
        "--workspace", "/work", "--job", "job-1", "--domain", "2", "--sequence", "3", "--cache-root", "cache"])
    assert args.remote_action == "sync-native-plots" and args.sequence == 3 and args.domain == 2
