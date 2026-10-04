"""Original native live radar beside aggregate-only member forecasts."""
from dataclasses import replace
import gc
import json
from pathlib import Path

import pytest

from conftest import requires_gpu
from test_ensemble_runtime_forecast_gpu import runtime_case, _variant, _RuntimeWords, _RuntimeProducts
from test_ensemble_production_forecast_gpu import _sha

pytestmark = [pytest.mark.gpu, requires_gpu]


def _products(collector):
    collector.finish_run()
    receipt = collector.require_complete()
    assert not receipt["pending_rosters"] and not receipt["unavailable_products"]
    assert len(collector.spools) == 1
    records = {}
    for spool in collector.spools.values():
        assert len(spool.frames) == 4
        for valid, frame in spool.frames.items():
            assert frame["status"] == "complete" and not frame["unavailable_fields"]
            assert len(frame["products"]) == len(frame["maps"]) == 1
            for kind in ("products", "maps"):
                for path in frame[kind]:
                    records[f"{spool.domain}/{valid}/{kind}/{Path(path).name}"] = _sha(spool.root / path)
    return records


def _radar_artifacts(root):
    manifest = json.loads((root / "radar/manifest.json").read_bytes())
    assert manifest["schema"] == "simulated-radar.manifest/v1" and manifest["simulated"]
    assert manifest["volumes"]
    assert any(len(volume["source_times"]) == 2 for volume in manifest["volumes"])
    artifacts = {}
    for volume in manifest["volumes"]:
        assert {row["format"] for row in volume["files"]} == {"level2", "cfradial1"}
        assert volume["images"]
        key = (volume["domain"], volume["site"]["id"], volume["valid_time"])
        artifacts[key] = {row["path"]: _sha(root / row["path"])
            for row in (*volume["files"], *volume["images"])}
    return artifacts


def test_actual_live_radar_words_survive_aggregate_capture_and_history_retirement(runtime_case, tmp_path, monkeypatch):
    from gpuwm import runtime, rustwx
    from gpuwm.ensemble import runtime_context
    from gpuwm.ensemble.production import PreparedEnsembleSession
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, ensemble_scope, member_output_scope
    from gpuwm.ensemble.runtime_preparation import RuntimeMemberInputs
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    from gpuwm.ensemble.batch_products import FieldProducts
    from gpuwm.simulated_radar_config import SimulatedRadarOptions
    from test_ensemble_batch_product_spool_gpu import _renderer
    # A missing current artifact is a failure of this explicitly requested gate.
    assert rustwx.canonical_radar_binary().is_file()
    exp = _variant(runtime_case[0], "fixed")
    data = runtime_case[1]
    grid = runtime.experiment_grid(exp, data)
    latitude, longitude = grid.ref_lat, grid.ref_lon
    options = SimulatedRadarOptions.from_mapping({"enabled": True,
        "sites": [{"id": "SIM1", "lat": latitude,
                   "lon": longitude, "height_m": 1000.}],
        "scan_strategy": "custom", "elevations_deg": [.5],
        "formats": ["level2", "cfradial1"], "fields": ["reflectivity", "velocity"],
        "timing": "scan", "range_km": 30., "gate_spacing_m": 750.,
        "azimuth_step_deg": 10., "volume_duration_s": 12.})
    exp = replace(exp, simulated_radar=options)
    inputs = RuntimeMemberInputs(exp, data)
    renderer = _renderer()
    reference = _RuntimeWords(tmp_path / "ordinary-words")
    active = [reference]
    original_single = runtime_context.bind_current_member_state
    original_tree = runtime_context.bind_current_member_model
    def single(**owner):
        original_single(**owner)
        active[0].bind_state(**owner)
    def tree(model):
        original_tree(model)
        active[0].bind_model(model)
    monkeypatch.setattr(runtime_context, "bind_current_member_state", single)
    monkeypatch.setattr(runtime_context, "bind_current_member_model", tree)
    def collector(path):
        return HeadlineDiagnosticCollector(path, members=2, renderer=renderer,
            start_time=exp.start_time, keep_member_files=False,
            requests=(FieldProducts("temperature2", "K", (293.15,)),),
            tile_rows=16, render_products=("prob",))
    ordinary_collector = collector(tmp_path / "ordinary-products")
    ordinary_capture = _RuntimeProducts(ordinary_collector, reference)
    expected_radar = {}
    for member in range(2):
        reference.pure_member = member
        root = tmp_path / "ordinary" / f"member-{member:04d}"
        with member_output_scope(MemberOutputCapture(ordinary_capture.submit, member, True)):
            report = runtime.run_experiment(exp, data, root, progress_callback=reference.progress)
        assert report.nan_free and report.completed_seconds == 48.
        assert report.wrfout_paths and all(path.is_file() for path in report.wrfout_paths)
        expected_radar[member] = _radar_artifacts(root)
        gc.collect()
    expected_products = _products(ordinary_collector)
    words = _RuntimeWords(tmp_path / "ensemble-words", reference=reference)
    active[0] = words
    ensemble_collector = collector(tmp_path / "ensemble")
    session = PreparedEnsembleSession({"members": 2}, output_directory=ensemble_collector.root,
        collector=_RuntimeProducts(ensemble_collector, words),
        identical_members="identity gate: every member must reproduce the ordinary forecast word for word")
    with ensemble_scope(session):
        report = runtime.run_experiment(exp, data, ensemble_collector.root,
                                        progress_callback=words.progress)
    assert report.nan_free and report.completed_seconds == 48.
    manifest = session.last_manifest
    assert manifest["status"] == "PASS" and manifest["members_completed"] == [0, 1]
    assert manifest["member_history_files"] == []
    assert not list(ensemble_collector.root.glob("members/**/wrfout_*"))
    assert len(manifest["member_radar_products"]) == 2
    for row in manifest["member_radar_products"]:
        assert row["deleted_history_files"]
        assert _radar_artifacts(ensemble_collector.root / row["directory"]) == expected_radar[row["member_id"]]
    assert not words.owners and words.records == reference.records
    assert _products(ensemble_collector) == expected_products
    session.completed_products()
    (tmp_path / "ensemble-radar-identity.json").write_text(json.dumps({
        "schema": "gpuwm-ensemble-live-radar-identity.v1", "status": "PASS",
        "member_radar_products": manifest["member_radar_products"],
        "ordinary_words": reference.flush(), "ensemble_words": words.flush(),
        "products": expected_products}, indent=2) + "\n", encoding="utf-8")
