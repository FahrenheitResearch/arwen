"""Campaign bookkeeping refuses biased denominators and changed product events."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("campaign_score", TOOLS / "ensemble_campaign_score.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def contract():
    members = [{"id": "m19", "member_id": 19, "seed": 2**64 - 1}]
    archive = {"member_order": [19], "member_metadata": [{"member_id": 19, "seed": 2**64 - 1}]}
    manifest = {"schema": "gpuwm-ensemble-output.v2", "member_order": [19],
                "member_metadata": archive["member_metadata"], "members_requested": 1,
                "probability_denominator": 1, "products": [
                    {"field": "temperature2", "units": "K", "comparison": "ge", "thresholds": [293.1499938964844]},
                    {"field": "wind10", "units": "m s-1", "comparison": "ge", "thresholds": [10.]},
                    {"field": "rain_total", "units": "mm", "comparison": "ge", "thresholds": [25.]}],
                "frames": [{"valid_time": "2000-01-01 03:00:00", "status": "complete", "members_received": [19],
                    "members_expected": 1, "products": ["product.nc"], "available_fields": ["temperature2", "wind10", "rain_total"]}]}
    return manifest, archive, members, [campaign.instant("2000-01-01T03:00:00Z")]


def test_actual_f32_threshold_and_uint64_seed_are_retained():
    result = campaign.product_contract(*contract())
    assert result["temperature_2m"] == [293.1499938964844]


@pytest.mark.parametrize("change,match", [
    ("nominal_threshold", "float32"), ("different_seed", "seeds"),
    ("missing_member", "incomplete"), ("pending", "incomplete"), ("comparison", ">= event"),
])
def test_production_contract_refuses_changed_events_or_roster(change, match):
    manifest, archive, members, clocks = contract()
    if change == "nominal_threshold": manifest["products"][0]["thresholds"] = [293.15]
    if change == "different_seed": members[0]["seed"] -= 1
    if change == "missing_member": manifest["frames"][0]["members_received"] = []
    if change == "pending": manifest["frames"][0]["status"] = "pending"
    if change == "comparison": manifest["products"][0]["comparison"] = "gt"
    with pytest.raises(ValueError, match=match):
        campaign.product_contract(manifest, archive, members, clocks)


def test_common_observation_columns_ignore_member_values_and_keep_missing(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    a.write_text("sample_id\tweight\tobserved\tm19\nx\t1\tNaN\t1\ny\t1\t3\t2\n")
    b.write_text("sample_id\tweight\tobserved\tm19\nx\t1\tNaN\t2\ny\t1\t3\t4\n")
    left = campaign.join_matches([a], tmp_path / "left.tsv", ["m19"])
    right = campaign.join_matches([b], tmp_path / "right.tsv", ["m19"])
    assert left == right and left["rows"] == 2 and left["observed_rows"] == 1
    b.write_text(b.read_text().replace("y\t1\t3", "y\t1\t4"))
    assert campaign.join_matches([b], tmp_path / "changed.tsv", ["m19"])["sha256"] != left["sha256"]


@pytest.mark.parametrize("tail", ["x\t1\t2\tNaN\n", "x\t1\t2\tinf\n", "x\t1\t2\t1\nx\t1\t2\t2\n"])
def test_member_failure_or_duplicate_sample_cannot_silently_change_denominator(tmp_path, tail):
    source = tmp_path / "rows.tsv"
    source.write_text("sample_id\tweight\tobserved\tm19\n" + tail)
    with pytest.raises(ValueError):
        campaign.join_matches([source], tmp_path / "joined.tsv", ["m19"])


def test_manifest_pin_detects_mutation(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"members": 20}))
    pin = {"path": str(path), "sha256": campaign.sha256(path)}
    assert campaign.load_pin(pin)["members"] == 20
    path.write_text(json.dumps({"members": 19}))
    with pytest.raises(ValueError, match="pinned input changed"):
        campaign.load_pin(pin)
