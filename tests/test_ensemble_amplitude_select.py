"""Frozen amplitude fitting uses all training arms and never held-out scores."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("amplitude_select", TOOLS / "ensemble_amplitude_select.py")
select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select)


def pin(path, record):
    path.write_text(json.dumps(record))
    return {"path": str(path), "sha256": select.sha256(path)}


def setup(tmp_path):
    plan = {"schema": "ensemble-calibration.amplitude-selection.v1", "training_cases": ["train-a", "train-b"],
            "held_out_cases": ["held-out"], "scored_lead_hours": [3, 4], "recenter_amplitudes": [0.5, 1., 1.5],
            "stochastic_amplitudes": [0.5, 1.], "products": ["temperature_2m", "wind_speed_10m", "forecast_total_precipitation"],
            "objective": "Equal-weight mean normalized empirical CRPS"}
    campaign = {"training_cases": plan["training_cases"], "held_out_cases": plan["held_out_cases"],
                "scored_leads": plan["scored_lead_hours"], "members": 20, "time_lag_members": 12,
                "recipes_conus": ["recentered", "time-lagged", "multi-model"]}
    request = {"stage": "recenter", "selection_plan": pin(tmp_path / "plan.json", plan),
               "campaign_plan": pin(tmp_path / "campaign.json", campaign), "arms": [
                   {"score_recipe": "control", "kind": "ordinary-control"}, *[
                       {"score_recipe": f"a{a}", "kind": "recentered", "recenter_amplitude": a, "stochastic_amplitude": 0.}
                       for a in plan["recenter_amplitudes"]]]}
    return request, plan, campaign


def summary(request, plan, values=None):
    products = []
    for case in plan["training_cases"]:
        for quantity in plan["products"]:
            for arm in request["arms"]:
                name = arm["score_recipe"]
                n = 1 if name == "control" else 12 if arm["kind"] == "time-lagged" else 20
                products.append({"case_id": case, "recipe": name, "held_out": False,
                    "quantity": select.normalize_quantity(quantity), "members": n,
                    "common_observation_mask": {"sha256": case + quantity, "rows": 8, "observed_rows": 8},
                    "scores": {"samples": 8, "members": n, "missing_members": 0,
                        "crps": (values or {}).get(name, 2. if name == "control" else 1.)}})
    return {"schema": "gpuwm-ensemble-campaign-scores.v1", "status": "complete", "lead_hours": plan["scored_lead_hours"], "products": products}


def test_alias_equal_weight_and_exact_lower_amplitude_tie(tmp_path):
    request, plan, _ = setup(tmp_path)
    request["score_summaries"] = [pin(tmp_path / "scores.json", summary(request, plan))]
    result = select.selection(request)
    assert result["selected_recenter_amplitude"] == 0.5
    assert result["rankings"][0]["objective"] == 0.5
    assert len(result["rankings"][0]["cells"]) == 6
    assert result["products"][-1] == "precipitation_accumulation"


@pytest.mark.parametrize("defect, message", [("heldout", "held-out"), ("dropped_row", "incomplete declared"),
    ("dropped_arm", "dropped a declared"), ("mask", "observation rows"), ("member_count", "member count")])
def test_no_heldout_training_or_dropped_arm(tmp_path, defect, message):
    request, plan, _ = setup(tmp_path)
    if defect == "dropped_arm": request["arms"].pop()
    scores = summary(request, plan)
    if defect == "heldout": scores["products"][0]["case_id"] = "held-out"
    if defect == "dropped_row": scores["products"].pop()
    if defect == "mask": scores["products"][1]["common_observation_mask"] = {"sha256": "changed"}
    if defect == "member_count": scores["products"][1]["members"] = 19
    request["score_summaries"] = [pin(tmp_path / "scores.json", scores)]
    with pytest.raises(ValueError, match=message): select.selection(request)


def test_zero_control_mae_is_explicit_and_common_to_every_candidate(tmp_path):
    request, plan, _ = setup(tmp_path)
    scores = summary(request, plan)
    scores["products"][0]["scores"]["crps"] = 0.
    request["score_summaries"] = [pin(tmp_path / "scores.json", scores)]
    result = select.selection(request)
    assert len(result["excluded_zero_denominators"]) == 1
    assert all(len(row["cells"]) == 5 for row in result["rankings"])


def test_global_stochastic_and_default_use_complete_grid(tmp_path):
    request, plan, campaign = setup(tmp_path)
    request["score_summaries"] = [pin(tmp_path / "first-scores.json", summary(request, plan))]
    first = select.selection(request)
    request["previous_selection"] = pin(tmp_path / "first-selection.json", first)
    request["stage"] = "stochastic-and-default"
    request["arms"] = [{"score_recipe": "control", "kind": "ordinary-control"}]
    for recipe in campaign["recipes_conus"]:
        for amplitude in [0., 0.5, 1.]:
            request["arms"].append({"score_recipe": f"{recipe}-{amplitude}", "kind": recipe,
                "stochastic_amplitude": amplitude, "recenter_amplitude": 0.5 if recipe == "recentered" else None})
    values = {arm["score_recipe"]: 0.6 if arm.get("stochastic_amplitude") == 1. else 1.
              for arm in request["arms"] if arm["kind"] != "ordinary-control"}
    request["score_summaries"] = [pin(tmp_path / "second-scores.json", summary(request, plan, values))]
    result = select.selection(request)
    assert result["selected_stochastic_amplitude"] == 1.
    assert result["selected_default"]["recipe"] == "recentered"
    assert len(result["rankings"][0]["cells"]) == 18
    bad = copy.deepcopy(request)
    bad["arms"].pop()
    bad["score_summaries"] = [pin(tmp_path / "dropped-scores.json", summary(bad, plan, values))]
    with pytest.raises(ValueError, match="dropped a declared recipe"): select.selection(bad)
