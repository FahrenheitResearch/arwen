"""Native scoring contract, with analytical values rather than model claims."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    compiler = shutil.which("rustc")
    if compiler is None:
        pytest.skip("native scoring check requires rustc")
    target = tmp_path_factory.mktemp("ensemble-score") / "score"
    subprocess.run([compiler, "--edition", "2021", "-O",
                    str(ROOT / "tools/ensemble_calibration_score.rs"),
                    "-o", str(target)], check=True)
    return target


def invoke(scorer, tmp_path, text, thresholds="-"):
    matched = tmp_path / "matched.tsv"
    matched.write_text(text, encoding="utf-8")
    return subprocess.run([str(scorer), str(matched), thresholds],
                          capture_output=True, text=True)


def test_analytical_crps_spread_rank_and_brier(scorer, tmp_path):
    run = invoke(scorer, tmp_path,
                 "sample_id\tweight\tobserved\ta\tb\n"
                 "s1\t1\t1\t0\t2\n"
                 "s2\t1\t3\t0\t2\n", "2")
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["crps"] == pytest.approx(1.0)
    assert result["fair_crps"] == pytest.approx(0.5)
    assert result["ensemble_mean_rmse"] == pytest.approx(2 ** 0.5)
    assert result["ensemble_mean_bias"] == pytest.approx(-1.0)
    assert result["rms_member_sample_spread"] == pytest.approx(2 ** 0.5)
    assert result["spread_skill_ratio"] == pytest.approx(1.0)
    assert result["finite_ensemble_corrected_spread_skill_ratio"] == pytest.approx(1.5 ** 0.5)
    assert result["rank_weight"] == [0, 1, 1]
    threshold = result["threshold_scores"][0]
    assert threshold["brier_score"] == pytest.approx(0.25)
    assert threshold["reliability"][1]["observed_frequency"] == 0.5
    assert threshold["reliability"][0]["observed_frequency"] is None


def test_ties_have_no_low_rank_bias(scorer, tmp_path):
    run = invoke(scorer, tmp_path,
                 "sample_id\tweight\tobserved\ta\tb\n"
                 "s1\t3\t0\t0\t0\n", "0")
    result = json.loads(run.stdout)
    assert result["rank_weight"] == [1, 1, 1]
    assert result["crps"] == 0
    assert result["spread_skill_ratio"] is None
    assert result["threshold_scores"][0]["brier_score"] == 0


def test_twenty_member_empirical_and_fair_crps(scorer, tmp_path):
    labels = "\t".join(f"member_{i:02d}" for i in range(20))
    values = "\t".join(str(i) for i in range(1, 21))
    run = invoke(scorer, tmp_path,
                 f"sample_id\tweight\tobserved\t{labels}\n"
                 f"s1\t1\t10.5\t{values}\n", "10.5")
    result = json.loads(run.stdout)
    assert result["members"] == 20
    assert result["crps"] == pytest.approx(1.675)
    assert result["fair_crps"] == pytest.approx(1.5)
    assert result["rms_member_sample_spread"] == pytest.approx(35 ** 0.5)
    assert result["rank_weight"][10] == 1
    assert sum(result["rank_weight"]) == 1
    assert result["threshold_scores"][0]["brier_score"] == pytest.approx(0.25)


def test_missing_data_never_become_zero(scorer, tmp_path):
    run = invoke(scorer, tmp_path,
                 "sample_id\tweight\tobserved\ta\tb\n"
                 "s1\t1\tNaN\t0\t0\n"
                 "s2\t1\t2\tNaN\t2\n"
                 "s3\t1\t2\t0\t2\n")
    result = json.loads(run.stdout)
    assert result["samples"] == 1
    assert result["missing_observations"] == 1
    assert result["missing_members"] == 1
    assert result["crps"] == pytest.approx(0.5)


def test_singleton_is_absolute_error_not_defined_spread(scorer, tmp_path):
    run = invoke(scorer, tmp_path,
                 "sample_id\tweight\tobserved\ta\n"
                 "s1\t1\t3\t1\n", "2")
    result = json.loads(run.stdout)
    assert result["crps"] == 2
    assert result["fair_crps"] is None
    assert result["rms_member_sample_spread"] is None
    assert result["threshold_scores"][0]["brier_score"] == 1


@pytest.mark.parametrize("body,thresholds,error", [
    ("sample_id\tweight\tobserved\ta\ta\ns\t1\t1\t0\t2\n", "-", "unique"),
    ("sample_id\tweight\tobserved\ta\ns\t1\t1\t0\ns\t1\t1\t0\n", "-", "repeated"),
    ("sample_id\tweight\tobserved\ta\ns\t0\t1\t0\n", "-", "positive"),
    ("sample_id\tweight\tobserved\ta\ns\t1\tNaN\t0\n", "-", "no complete"),
    ("sample_id\tweight\tobserved\ta\ns\t1\t1\t0\n", "NaN", "finite"),
    ("sample_id\tweight\tobserved\ta\ns\t1\t1e308\t0\n", "-", "overflowed"),
])
def test_invalid_inputs_refuse_named_breakage(scorer, tmp_path, body, thresholds, error):
    result = invoke(scorer, tmp_path, body, thresholds)
    assert result.returncode == 2
    assert error in result.stderr


def test_weighted_cases_and_member_permutation(scorer, tmp_path):
    rows = "sample_id\tweight\tobserved\ta\tb\tc\n"
    first = invoke(scorer, tmp_path, rows + "s1\t2\t1\t0\t1\t2\ns2\t1\t3\t0\t1\t2\n", "1,2")
    second = invoke(scorer, tmp_path, rows + "s1\t2\t1\t2\t0\t1\ns2\t1\t3\t1\t2\t0\n", "1,2")
    assert json.loads(first.stdout) == json.loads(second.stdout)
    result = json.loads(first.stdout)
    assert result["ensemble_mean_bias"] == pytest.approx(-2 / 3)
    assert result["ensemble_mean_rmse"] == pytest.approx((4 / 3) ** 0.5)
    assert sum(result["rank_weight"]) == pytest.approx(3)


def test_receipt_pins_native_input_and_source(scorer, tmp_path):
    spec = importlib.util.spec_from_file_location("ensemble_score", ROOT / "tools/ensemble_calibration_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matched = tmp_path / "matched.tsv"
    matched.write_text("sample_id\tweight\tobserved\ta\tb\ns\t1\t1\t0\t2\n")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(dict(case_id="fixture", recipe="fixture", quantity="scalar",
        units="1", member_ids=["a", "b"], source_receipts=["fixture-source"],
        observation_receipts=["fixture-observation"], match_receipt="fixture-match",
        valid_start="2000-01-01T00:00:00Z", valid_end="2000-01-01T01:00:00Z", spinup_seconds=0)))
    result = module.score(binary=scorer, matched=matched, thresholds="-", provenance=provenance)
    assert result["status"] == "matched-pair-scores-only"
    assert result["input_sha256"] == module.sha256(matched)
    assert result["scorer_sha256"] == module.sha256(scorer)
    assert result["scores"]["members"] == 2
    metadata = json.loads(provenance.read_text())
    metadata["member_ids"] = ["b", "a"]
    provenance.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="member order"):
        module.score(binary=scorer, matched=matched, thresholds="-", provenance=provenance)
