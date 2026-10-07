import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import da_recent_score as score


def case():
    return {"case_id": "synthetic-recent-only", "analysis_times_utc": ["2026-10-04T01:00:00Z", "2026-10-04T02:00:00Z", "2026-10-04T03:00:00Z"],
            "forecast_fork_utc": "2026-10-04T03:00:00Z", "center_lat": 31, "center_lon": -89}


def manifest(tmp_path):
    return {"out": str(tmp_path/"runs"), "slots": [{"analysis_time": stamp} for stamp in case()["analysis_times_utc"]]}


def test_full_member_pair_plan_and_exact_frame_patterns(tmp_path):
    jobs = score.plan(case(), manifest(tmp_path), tmp_path/"truth.json", tmp_path/"score", [1, 2])
    assert len(jobs) == 128
    assert [(job["member"], job["arm"]) for job in jobs[:4]] == [(0, "no-da"), (0, "da"), (1, "no-da"), (1, "da")]
    da = jobs[1]
    assert da["builder"][da["builder"].index("--pattern")+1] == "wrfout_*_0.nc"
    assert da["builder"][da["builder"].index("--frames")+1].endswith("seed-1/da/002/composites") or da["builder"][da["builder"].index("--frames")+1].endswith("seed-1\\da\\002\\composites")
    assert "--baseline" in da["argv"]
    assert "--baseline" not in jobs[0]["argv"]


def test_domain_suffix_does_not_match_other_members(tmp_path):
    job = score.plan(case(), manifest(tmp_path), tmp_path/"truth.json", tmp_path/"score", [1], 32, 2)[21]
    assert job["member"] == 10
    assert job["builder"][job["builder"].index("--pattern")+1] == "wrfout_*_10_d02.nc"


def test_new_recent_controller_uses_one_da_directory_for_all_hourly_legs(tmp_path):
    run = dict(manifest(tmp_path), schema="gpuwm-da.recent-run.v1")
    job = score.plan(case(), run, tmp_path/"truth.json", tmp_path/"score", [1], 1)[1]
    path = Path(job["builder"][job["builder"].index("--frames")+1])
    assert path == Path(run["out"]).resolve()/"seed-1/da/composites"


def test_contradictory_fork_or_seed_roster_refused(tmp_path):
    run = manifest(tmp_path); run["slots"][-1]["analysis_time"] = "2026-10-04T04:00:00Z"
    with pytest.raises(ValueError, match="fork"):
        score.plan(case(), run, tmp_path/"truth.json", tmp_path/"score", [1])
    with pytest.raises(ValueError, match="distinct"):
        score.plan(case(), manifest(tmp_path), tmp_path/"truth.json", tmp_path/"score", [1, 1])


def test_separate_clean_start_root_requires_same_case_identity(tmp_path):
    da = manifest(tmp_path); control = manifest(tmp_path)
    control["out"] = str(tmp_path/"separate-control")
    jobs = score.plan(case(), da, tmp_path/"truth.json", tmp_path/"score", [1], 1, control_run=control)
    assert "separate-control" in jobs[0]["builder"][jobs[0]["builder"].index("--frames")+1]
    control["prepared_content_sha256"] = "different"
    with pytest.raises(ValueError, match="identity"):
        score.plan(case(), da, tmp_path/"truth.json", tmp_path/"score", [1], 1, control_run=control)


def test_missing_products_emit_all_36_pending_rows_for_every_member(tmp_path):
    jobs = score.plan(case(), manifest(tmp_path), tmp_path/"truth.json", tmp_path/"score", [1], 2)
    def refuse(argv, log, timeout):
        return {"exit_code": 2, "status": "refused", "argv": argv, "log": str(log), "wall_seconds": .01}
    result = score.execute(jobs, tmp_path/"score", seeds=[1], members=2, runner=refuse)
    assert result["rows"] == 144 and result["complete_measurements"] == 0
    assert all(row["value"] is None for row in result["member_scalar_summaries"])
    rows = [json.loads(line) for line in (tmp_path/"score/all-members.jsonl").read_text().splitlines()]
    assert set(row["lead_hours"] for row in rows) == {1, 3, 6}
    assert all(row["schema"] == "regional-rain/v1" and row["measurement_status"] == "pending" for row in rows)


def test_interruption_stops_new_subprocesses_and_records_remaining_pending(tmp_path):
    jobs = score.plan(case(), manifest(tmp_path), tmp_path/"truth.json", tmp_path/"score", [1], 2)
    calls = []
    def interrupt(argv, log, timeout):
        calls.append(argv)
        return {"exit_code": 124, "status": "interrupted", "argv": argv, "log": str(log), "wall_seconds": .01}
    result = score.execute(jobs, tmp_path/"score", seeds=[1], members=2, runner=interrupt)
    assert len(calls) == 1
    assert result["rows"] == 144 and result["complete_measurements"] == 0


def test_scalar_mean_is_named_as_member_score_and_partial_roster_has_no_value(tmp_path):
    item = {"seed": 1, "member": 0, "arm": "da", "product": "member_000"}
    rows = score.pending_rows(item, "analytic", "2026-10-04T03:00:00Z", "fixture")
    one = dict(rows[0], member=0, measurement_status="complete", value=.8, paired_gain=.1)
    two = dict(rows[0], member=1, measurement_status="complete", value=1.2, paired_gain=.2)
    mean = score.scalar_summary([one, two], [1], 2)[0]
    assert mean["metric"] == "mean_member_footprint_rain_ratio"
    assert mean["value"] == pytest.approx(1.0)
    assert mean["mean_member_paired_gain"] == pytest.approx(.15)
    assert "not a score of ensemble-mean" in mean["definition"]
    assert score.scalar_summary([one], [1], 2)[0]["value"] is None


def test_existing_score_output_refused_instead_of_mixing_generations(tmp_path):
    jobs = score.plan(case(), manifest(tmp_path), tmp_path/"truth.json", tmp_path/"score", [1], 1)
    path = Path(jobs[0]["scores"]); path.parent.mkdir(parents=True); path.write_text("old")
    with pytest.raises(ValueError, match="fresh score"):
        score.execute(jobs, tmp_path/"score", seeds=[1], members=1)
    assert path.read_text() == "old"


def test_saved_synthetic_fields_cli_runs_native_scoring_and_campaign_join(tmp_path):
    times = np.arange(0, 21601, 120, dtype=float)
    rate = np.zeros((len(times), 41, 41), dtype=float); rate[:, 19:22, 19:22] = 12
    echo = np.zeros_like(rate); echo[:, 19:22, 19:22] = 40
    saved = tmp_path/"analytic.npz"
    np.savez_compressed(saved, times=times, rate=rate, accum=rate*times[:, None, None]/3600,
                        echo=echo, mask=np.ones_like(rate, dtype=np.uint8), resets=np.zeros(len(times), dtype=np.int64))
    field = lambda name: {"path": str(saved), "variable": name}
    common = {"schema": "regional-rain/input.v1", "is_stub": True,
        "grid": {"x_edges_m": (np.arange(42)*3000).tolist(), "y_edges_m": (np.arange(42)*3000).tolist(), "projection_id": "analytic-equal-area"},
        "times_seconds": field("times"), "rain_valid": field("mask"), "echo_dbz": field("echo"), "echo_valid": field("mask")}
    forecast = tmp_path/"forecast.json"; forecast.write_text(json.dumps(dict(common, rain_accum_mm=field("accum"), reset_ids=field("resets"))))
    truth = tmp_path/"truth.json"; truth.write_text(json.dumps(dict(common, rate_mm_h=field("rate"))))
    case_path = tmp_path/"case.json"; case_path.write_text(json.dumps(case()))
    run_path = tmp_path/"run.json"; run_path.write_text(json.dumps(manifest(tmp_path)))
    inputs_path = tmp_path/"inputs.json"; inputs_path.write_text(json.dumps({"schema": "da-rerun.saved-score-inputs.v1",
        "members": [{"seed": 1, "member": 0, "da": str(forecast), "no_da": str(forecast)}]}))
    out = tmp_path/"scores"
    process = subprocess.run([sys.executable, str(Path(score.__file__)), "--case", str(case_path), "--run-manifest", str(run_path),
        "--truth", str(truth), "--saved-inputs", str(inputs_path), "--seeds", "1", "--members", "1", "--out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stderr
    result = json.loads((out/"campaign-summary.json").read_text())
    assert result["rows"] == result["complete_measurements"] == 72
    assert all(row["value"] == pytest.approx(1) for row in result["member_scalar_summaries"])
    assert result["scientific_gate"] == "pending"
    assert result["ensemble_mean_field_status"].startswith("pending")
