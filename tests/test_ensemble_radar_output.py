"""Requested native radar survives aggregate capture with verified cleanup."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.radar_output import finish_member_radar, verified_radar_products
from gpuwm.ensemble.runtime_context import current_capture
from test_ensemble_production_execution import Collector, inputs
from gpuwm.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from gpuwm.ensemble.packing import CardBudget
from gpuwm.ensemble.production import PreparedEnsembleSession


def _committed(root):
    radar = root / "radar"
    radar.mkdir()
    frames = [root / "wrfout_d01_initial", root / "wrfout_d01_final"]
    for frame in frames:
        frame.write_bytes(b"original volume")
    artifact = radar / "reflectivity.png"
    artifact.write_bytes(b"native image")
    row = {"path": "radar/reflectivity.png", "format": "png", "bytes": artifact.stat().st_size,
           "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    document = {"schema": "simulated-radar.manifest/v1", "simulated": True,
                "volumes": [{"files": [row], "images": []}], "loops": []}
    manifest = radar / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return frames, artifact, manifest, document


def _radar_inputs():
    selected = inputs()
    selected.experiment.simulated_radar = SimpleNamespace(enabled=True)
    return selected


@pytest.mark.parametrize("keep", [False, True])
def test_original_products_land_before_history_retirement_and_opt_in_retains_frames(tmp_path, keep):
    frames, artifact, manifest, _ = _committed(tmp_path)
    receipt = finish_member_radar(_radar_inputs(), tmp_path, {"wrfout_paths": frames}, keep_member_files=keep)
    assert artifact.is_file() and manifest.is_file()
    assert all(frame.exists() == keep for frame in frames)
    assert len(receipt["deleted_history_files"]) == (0 if keep else 2)
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("damage", ["missing", "words", "escape", "schema", "not_simulated"])
def test_failed_products_never_delete_original_histories(tmp_path, damage):
    frames, artifact, manifest, document = _committed(tmp_path)
    if damage == "missing":
        artifact.unlink()
    elif damage == "words":
        artifact.write_bytes(b"native imago")
    elif damage == "escape":
        document["volumes"][0]["files"][0]["path"] = "../outside.png"
    elif damage == "schema":
        document["schema"] = "another"
    else:
        document["simulated"] = False
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        finish_member_radar(_radar_inputs(), tmp_path, {"wrfout_paths": frames}, keep_member_files=False)
    assert all(frame.exists() for frame in frames)


def test_cleanup_rejects_any_foreign_history_before_deleting_first_own_frame(tmp_path):
    root = tmp_path / "member"
    root.mkdir()
    frames, *_ = _committed(root)
    foreign = tmp_path / "wrfout_foreign"
    foreign.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="inside its owned"):
        finish_member_radar(_radar_inputs(), root, {"wrfout_paths": [*frames, foreign]}, keep_member_files=False)
    assert all(frame.exists() for frame in frames) and foreign.exists()


def test_session_keeps_requested_radar_on_original_runner_without_retained_member_wrfouts(tmp_path):
    prepared, collector = _radar_inputs(), Collector()
    seen = []
    def original(member_inputs, *, output_directory, **kwargs):
        capture = current_capture()
        assert capture.keep_member_files
        assert member_inputs is prepared
        seen.append(capture.member_id)
        frames, *_ = _committed(output_directory)
        return {"status": "PASS", "wrfout_paths": frames}
    def native(**kwargs):
        pytest.fail("requested radar must retain its original volume consumer")
    # Two copies of one prepared input on purpose: the test is about which
    # runner a radar request keeps, not about a forecast ensemble.
    session = PreparedEnsembleSession({"members": 2}, output_directory=tmp_path,
        identical_members="engine mechanics test: every member runs the one prepared input on purpose",
        collector=collector, native_executor=native, cards=(CardBudget(0, 1000),),
        memory_model=EnsembleMemoryModel((MemoryComponent("ordinary", "ordinary", fixed_bytes=100),)),
        device_scope=lambda _: nullcontext())
    receipt = session.run_prepared(original, prepared)
    assert seen == [0, 1] and receipt["status"] == "PASS"
    assert receipt["member_history_files"] == []
    assert [row["member_id"] for row in receipt["member_radar_products"]] == seen
    assert all(len(row["deleted_history_files"]) == 2 for row in receipt["member_radar_products"])
    session.completed_products()
    radar_manifest = tmp_path / receipt["member_radar_products"][0]["directory"] / "radar/manifest.json"
    document = json.loads(radar_manifest.read_bytes())
    document["warnings"] = ["changed"]
    radar_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after publication"):
        session.completed_products()


def test_off_input_has_no_manifest_or_history_dependency(tmp_path):
    assert finish_member_radar(inputs(), tmp_path, object(), keep_member_files=False) is None


def test_failing_member_keeps_histories_even_if_a_partial_radar_manifest_exists(tmp_path):
    frames, *_ = _committed(tmp_path)
    with pytest.raises(RuntimeError, match="retains its original histories"):
        finish_member_radar(_radar_inputs(), tmp_path, {"status": "FAIL", "wrfout_paths": frames},
                            keep_member_files=False)
    assert all(frame.is_file() for frame in frames)
