"""Completed-run retirement cannot delete unscored or foreign payloads."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ensemble_campaign_retention import plan_retention, retire, pin


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "run"
    forecast = root / "forecast"
    checkpoint = forecast / "members/member-0007/gpuwmrst_d01_2024-01-01_01_00_00__abcdef.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"analytical checkpoint retention payload")
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return pin(path)
    members = [{"id": "m7", "member_id": 7, "seed": 2**63 + 17}]
    fields = [("temperature_2m", "temperature2", "K"),
              ("wind_speed_10m", "wind10", "m s-1"),
              ("precipitation_accumulation", "rain_total", "mm")]
    diagnostic = forecast / "member-diagnostics/frame.nc"
    diagnostic.parent.mkdir()
    diagnostic.write_bytes(b"analytical diagnostic retention payload")
    artifact = pin(diagnostic)
    archive = {"schema": "gpuwm-ensemble-member-diagnostics.v1", "member_order": [7],
        "member_metadata": members, "coverage": {"status": "complete"}, "unavailable": [],
        "files": [{**artifact, "path": "member-diagnostics/frame.nc"}]}
    archive_ref = write(diagnostic.parent / "manifest.json", archive)
    aggregate = forecast / "d01/products/ensemble.nc"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"analytical aggregate field payload")
    picture = forecast / "maps/d01/product/ensemble.png"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"analytical aggregate map payload")
    product = {"schema": "gpuwm-ensemble-output.v2", "member_order": [7], "member_metadata": members,
        "members_requested": 1, "probability_denominator": 1,
        "products": [{"field": field, "units": units, "comparison": "ge", "thresholds": [1.0]}
                     for _, field, units in fields],
        "frames": [{"valid_time": "2024-01-01T01:00:00Z", "status": "complete",
            "members_received": [7], "members_expected": 1, "products": ["d01/products/ensemble.nc"],
            "maps": ["maps/d01/product/ensemble.png"],
            "available_fields": [row[1] for row in fields]}]}
    product_ref = write(forecast / "d01/ensemble-manifest.json", product)
    ensemble = {"status": "PASS", "member_order": [7], "members_completed": [7],
        "completed_seconds": 3600,
        "products": {"domain_manifests": [{"manifest": "d01/ensemble-manifest.json"}], "pending_rosters": []},
        "member_results": [{"result": {"status": "PASS", "restart_contract": {
            "checkpoints_written": [str(checkpoint)]}}}]}
    ensemble_ref = write(forecast / "ensemble-run.json", ensemble)
    run_ref = write(root / "campaign-run-receipt.json", {
        "schema": "gpuwm-ensemble-campaign-run.v1", "status": "PASS",
        "execution_window": {"purpose": "calibration", "calibration_complete": True,
            "execution_run_seconds": 3600, "source_window_seconds": 3600},
        "ensemble_manifest": ensemble_ref})
    case = {"case_id": "analytical", "start_time": "2024-01-01T00:00:00Z", "runs": [{
        "recipe": "control", "members": members, "archive": {"manifest": archive_ref},
        "production_manifest": product_ref}]}
    request_ref = write(tmp_path / "scores/request.json", {"cases": [case]})
    match_ref = write(tmp_path / "scores/match.json", {"scope": "analytical metadata fixture"})
    products = []
    for quantity, field, units in fields:
        scores = {"missing_members": 0, "members": 1, "samples": 2}
        receipt = write(tmp_path / f"scores/{field}.json", {
            "schema": "gpuwm-ensemble-calibration.receipt.v1", "scores": scores,
            "provenance": {"case_id": "analytical", "recipe": "control", "quantity": quantity, "units": units,
                "diagnostic_manifest": archive_ref, "production_manifest": product_ref,
                "members": members, "lead_hours": [1], "match_receipt": [match_ref]}})
        products.append({"case_id": "analytical", "recipe": "control", "quantity": quantity,
                         "units": units, "members": 1, "scores": scores, "receipt": receipt})
    summary_ref = write(tmp_path / "scores/summary.json", {
        "schema": "gpuwm-ensemble-campaign-scores.v1", "status": "complete", "lead_hours": [1],
        "request": request_ref, "products": products})
    return dict(run_receipt=run_ref["path"], score_summary=summary_ref["path"], owned_root=tmp_path,
                case_id="analytical", recipe="control"), checkpoint, diagnostic


def test_retirement_keeps_receipts_and_scores_and_records_every_deleted_payload(completed, tmp_path):
    args, checkpoint, diagnostic = completed
    plan = plan_retention(**args)
    expected = {row['path']: row for row in plan['files']}
    preview = retire(plan, audit_directory=tmp_path / "preview")
    assert preview["status"] == "planned" and checkpoint.exists()
    result = retire(plan, audit_directory=tmp_path / "applied", apply=True)
    assert result["deleted_files"] == 4
    assert result["deleted_logical_bytes"] == sum(row['bytes'] for row in expected.values())
    assert not diagnostic.exists() and not checkpoint.exists()
    ledger = [json.loads(line) for line in (tmp_path / "applied/deleted-files.jsonl").read_text().splitlines()]
    assert {row['path']: row['sha256'] for row in ledger} == {path: row['sha256'] for path,row in expected.items()}
    assert Path(args['run_receipt']).is_file() and Path(args['score_summary']).is_file()
    assert (diagnostic.parent/'manifest.json').is_file()


def test_explicit_field_review_can_retain_scored_fields(completed, tmp_path):
    args, checkpoint, diagnostic = completed
    plan = plan_retention(**args, keep_scored_fields=True)
    result = retire(plan, audit_directory=tmp_path/'review-retention', apply=True)
    assert result['deleted_files'] == 1
    assert not checkpoint.exists() and diagnostic.exists()


@pytest.mark.parametrize("which", ["checkpoint", "diagnostic", "score"])
def test_changed_payload_causes_no_deletion(completed, tmp_path, which):
    args, checkpoint, diagnostic = completed
    plan = plan_retention(**args)
    target = {"checkpoint": checkpoint, "diagnostic": diagnostic, "score": Path(args["score_summary"])}[which]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        retire(plan, audit_directory=tmp_path / "refused", apply=True)
    assert checkpoint.exists() and diagnostic.exists()


def test_incomplete_scoring_preserves_checkpoint(completed):
    args, checkpoint, _ = completed
    path = Path(args["score_summary"])
    document = json.loads(path.read_text())
    document["products"].pop()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="every requested scored product"):
        plan_retention(**args)
    assert checkpoint.exists()


def test_foreign_checkpoint_is_refused_even_if_report_was_rebound(completed, tmp_path):
    args, checkpoint, _ = completed
    foreign = tmp_path / "gpuwmrst_d01_2024-01-01_01_00_00.npz"
    foreign.write_bytes(b"another task's retained payload")
    run_path = Path(args["run_receipt"])
    run = json.loads(run_path.read_text())
    ensemble_path = Path(run["ensemble_manifest"]["path"])
    ensemble = json.loads(ensemble_path.read_text())
    ensemble["member_results"][0]["result"]["restart_contract"]["checkpoints_written"] = [str(foreign)]
    ensemble_path.write_text(json.dumps(ensemble))
    run["ensemble_manifest"] = pin(ensemble_path)
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="owned run directory"):
        plan_retention(**args)
    assert foreign.exists() and checkpoint.exists()
