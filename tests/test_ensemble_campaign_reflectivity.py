"""Family campaigns retain native event definitions and complete rosters."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ensemble_campaign_score as campaign
import ensemble_campaign_plot as charts


def test_quantity_table_registers_dbz_without_changing_archived_contract():
    assert campaign.QUANTITIES["composite_reflectivity"] == ("refl", "dBZ")
    assert len(campaign.ARCHIVED_QUANTITIES) == 3
    assert "composite_reflectivity" in charts.QUANTITY_TITLES


def fixture_frame():
    return {"quantity": "composite_reflectivity", "valid_time": "2024-01-08T18:10:00Z",
            "required_families": {"physics": ["p1", "p2"], "ai": ["a1"]}}


def native_receipt(frame):
    rows = [{"threshold": value, "samples": 10, "observed_events": 4, "forecast_events": 3,
             "hits": 2, "misses": 2, "false_alarms": 1, "correct_negatives": 5,
             "fss": .8, "csi": .4, "pod": .5, "far": 1/3, "frequency_bias": .75,
             "brier_score": .2} for value in (20, 35, 45)]
    return {"schema": "gpuwm-ensemble-grid-match.v1", "status": "complete",
            "quantity": frame["quantity"], "valid_time": frame["valid_time"],
            "grid": {"geometry_sha256": "same"},
            "families": {key: {"threshold_scores": rows, "members": len(ids)}
                         for key, ids in frame["required_families"].items()},
            "blend": {"threshold_scores": rows, "weights": {"physics": .5, "ai": .5}}}


def test_native_family_campaign_tables_and_analysis_chart(tmp_path, monkeypatch):
    binary = tmp_path / "matcher"
    binary.write_bytes(b"unit-fixture-only")
    calls = []
    original_run = campaign.subprocess.run

    def run(command, **kwargs):
        if command[0] != str(binary):
            return original_run(command, **kwargs)
        calls.append(command)
        frame = json.loads(Path(command[2]).read_text())
        return SimpleNamespace(stdout=json.dumps(native_receipt(frame)))

    monkeypatch.setattr(campaign.subprocess, "run", run)
    out = tmp_path / "campaign"
    result = campaign.run_campaign({"schema": "gpuwm-ensemble-grid-campaign.v1", "frames": [fixture_frame()]},
                                   matcher=binary, scorer=None, out=out)
    assert calls[0][1] == "match-grid"
    table = (out / "score-table.tsv").read_text()
    assert "\tai\tcomposite_reflectivity\tdBZ\t35" in table
    assert "\tphysics\tcomposite_reflectivity\tdBZ\t35" in table
    assert "\texperimental_blend\t" in table
    chart = charts.plot_scores(result, tmp_path / "charts")
    assert chart["files"] and all(Path(row["path"]).is_file() for row in chart["files"])


def test_grid_campaign_refuses_a_changed_required_roster(tmp_path, monkeypatch):
    binary = tmp_path / "matcher"
    binary.write_bytes(b"fixture")
    first = fixture_frame()
    second = {**first, "valid_time": "2024-01-08T18:20:00Z", "required_families": {"physics": ["p1"]}}
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps(native_receipt(first))))
    with pytest.raises(ValueError, match="changed the required family roster"):
        campaign.run_grid_campaign({"frames": [first, second]}, matcher=binary, out=tmp_path / "bad")
    assert json.loads((tmp_path / "bad" / "campaign-scores.json").read_text())["status"] == "failed"
