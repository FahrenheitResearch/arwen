"""Run the shipped scorecard executable with recorded, digest-bound inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run([sys.executable, "-m", "gpuwm.cli", "obs", "scorecard", *map(str, args)],
                          cwd=ROOT, text=True, capture_output=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


def test_product_door_runs_recorded_demo(tmp_path):
    demo = ROOT / "tests/fixtures/obs_scorecard/demo.json"
    result = invoke(demo, "--out", tmp_path)
    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads((tmp_path / "scorecard.json").read_text())
    assert receipt["coverage_status"] == "partial"
    assert len(receipt["case_scores"]) == 20
    # The two retained cycles belong to one event. They cannot create two
    # independent case blocks by virtue of having different run ids.
    assert all(row["improvement"]["status"] in ("not_measured", "diagnostic")
               for row in receipt["comparisons"])


def test_required_missing_data_returns_three(tmp_path):
    demo = json.loads((ROOT / "tests/fixtures/obs_scorecard/demo.json").read_text())
    base = ROOT / "tests/fixtures/obs_scorecard"
    for case in demo["cases"]:
        case["pairs"]["path"] = str(base / case["pairs"]["path"])
    demo["metrics"].append({"id": "missing-gust", "source": "asos", "quantity": "wind_gust_10m",
                            "units": "m s-1", "required": True})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(demo))
    out = tmp_path / "out"
    result = invoke(path, "--out", out)
    assert result.returncode == 3, result.stdout + result.stderr
    receipt = json.loads((out / "scorecard.json").read_text())
    assert len(receipt["missing_required"]) == 2
    assert receipt["coverage_status"] == "incomplete"


def test_tampered_pair_export_is_refused(tmp_path):
    demo = json.loads((ROOT / "tests/fixtures/obs_scorecard/demo.json").read_text())
    base = ROOT / "tests/fixtures/obs_scorecard"
    for case in demo["cases"]:
        case["pairs"]["path"] = str(base / case["pairs"]["path"])
    demo["cases"][0]["pairs"]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(demo))
    result = invoke(path, "--out", tmp_path / "out")
    assert result.returncode == 1
    assert "digest differs" in result.stderr


def test_real_station_subset_matches_hand_calculation(tmp_path):
    demo = json.loads((ROOT / "tests/fixtures/obs_scorecard/demo.json").read_text())
    base = ROOT / "tests/fixtures/obs_scorecard"
    demo["cases"] = demo["cases"][:1]
    case = demo["cases"][0]
    case["pairs"]["path"] = str(base / case["pairs"]["path"])
    case["pairs"]["filters"]["stationId"] = "APC"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(demo))
    result = invoke(path, "--out", tmp_path / "out")
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "out/scorecard.json").read_text())
    row = next(r for r in receipt["case_scores"] if r["metric"] == "asos-temperature-rmse")
    # Exact values from the retained station export. One sample: RMSE = |f-o|.
    assert row["count"] == 1
    assert abs(row["candidate"] - abs(300.24637316417375 - 302.0388888888889)) < 1e-12
    assert abs(row["reference"] - abs(301.3427090173701 - 302.0388888888889)) < 1e-12


def test_active_output_lock_prevents_mixed_receipts(tmp_path):
    from gpuwm.verification_visuals import _run_lock
    out = tmp_path / "out"
    with _run_lock(out, name=".scorecard.lock"):
        result = invoke(ROOT / "tests/fixtures/obs_scorecard/demo.json", "--out", out)
    assert result.returncode == 2
    assert "lock" in result.stdout or "mix receipts" in result.stdout
    assert not (out / "scorecard.json").exists()
    assert invoke(ROOT / "tests/fixtures/obs_scorecard/demo.json", "--out", out).returncode == 0
