"""Apply the predeclared two-stage training objective without held-out scores."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Inside the wheel these are tools.<name>; the bare form is what a
# run by path resolves (`python tools/<this file>`), where tools/ is
# sys.path[0].
try:
    from tools.ensemble_campaign_score import load_pin, save
    from tools.ensemble_calibration_score import sha256
except ImportError:
    from ensemble_campaign_score import load_pin, save
    from ensemble_calibration_score import sha256

ALIASES = {"forecast_total_precipitation": "precipitation_accumulation"}
PREFERENCE = {"input-ensemble": 0, "recentered": 1, "time-lagged": 2, "multi-model": 3}


def normalize_quantity(name):
    return ALIASES.get(name, name)


def selection(request):
    plan = load_pin(request["selection_plan"])
    campaign = load_pin(request["campaign_plan"])
    if plan.get("schema") != "ensemble-calibration.amplitude-selection.v1":
        raise ValueError("unsupported frozen amplitude selection plan")
    training, held_out = set(plan["training_cases"]), set(plan["held_out_cases"])
    if (not training or training & held_out or training != set(campaign["training_cases"])
            or held_out != set(campaign["held_out_cases"])
            or plan["scored_lead_hours"] != campaign["scored_leads"]):
        raise ValueError("selection and campaign plans disagree on their frozen split or leads")
    quantities = [normalize_quantity(value) for value in plan["products"]]
    if len(set(quantities)) != len(quantities):
        raise ValueError("selection products repeat a quantity after alias normalization")
    arms = request["arms"]
    labels = [arm["score_recipe"] for arm in arms]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("selection arm labels must be distinct")
    by_label = {arm["score_recipe"]: arm for arm in arms}
    controls = [arm for arm in arms if arm["kind"] == "ordinary-control"]
    if len(controls) != 1:
        raise ValueError("one explicit ordinary control arm is required for every training case")
    control = controls[0]["score_recipe"]
    rows = {}
    masks = {}
    for reference in request["score_summaries"]:
        summary = load_pin(reference)
        if (summary.get("schema") != "gpuwm-ensemble-campaign-scores.v1"
                or summary.get("status") != "complete"
                or summary["lead_hours"] != plan["scored_lead_hours"]):
            raise ValueError("training summary is incomplete or uses different lead hours")
        for row in summary["products"]:
            case, arm = row["case_id"], row["recipe"]
            quantity = normalize_quantity(row["quantity"])
            if row.get("held_out") is not False or case not in training or case in held_out:
                raise ValueError("held-out or unknown case scores cannot enter amplitude fitting")
            if arm not in by_label or quantity not in quantities:
                raise ValueError("training summary contains an undeclared arm or product")
            key = (case, quantity, arm)
            if key in rows:
                raise ValueError("training summaries repeat an arm/case/product")
            scores = row["scores"]
            value = scores["crps"]
            if (scores["samples"] <= 0 or scores["missing_members"] != 0
                    or value is None or not math.isfinite(value) or value < 0):
                raise ValueError("every training arm needs complete member rows and finite empirical CRPS")
            if arm == control and (row["members"] != 1 or scores["members"] != 1):
                raise ValueError("ordinary-control empirical CRPS is MAE only for N=1")
            expected_members = (1 if arm == control else campaign["time_lag_members"]
                                if by_label[arm]["kind"] == "time-lagged" else campaign["members"])
            if row["members"] != expected_members or scores["members"] != expected_members:
                raise ValueError("a training arm changed the frozen scientific member count")
            mask = row["common_observation_mask"]
            mask_key = (case, quantity)
            if mask_key in masks and masks[mask_key] != mask:
                raise ValueError("training arms changed their frozen observation rows")
            masks[mask_key] = mask
            rows[key] = row
    expected = {(case, quantity, arm) for case in training for quantity in quantities for arm in labels}
    missing = sorted(expected - rows.keys())
    if missing:
        raise ValueError(f"incomplete declared training campaign; missing {missing}")
    excluded = []
    denominators = {}
    for case in sorted(training):
        for quantity in quantities:
            value = rows[(case, quantity, control)]["scores"]["crps"]
            denominators[(case, quantity)] = value
            if value == 0:
                excluded.append({"case_id": case, "quantity": quantity, "reason": "ordinary-control MAE is zero"})
    if len(excluded) == len(denominators):
        raise ValueError("every ordinary-control MAE is zero; normalized selection is undefined")

    def objective(selected):
        cells = []
        for arm in selected:
            for (case, quantity), denominator in sorted(denominators.items()):
                if denominator == 0:
                    continue
                raw = rows[(case, quantity, arm)]["scores"]["crps"]
                cells.append({"arm": arm, "case_id": case, "quantity": quantity,
                              "crps": raw, "control_mae": denominator, "normalized_crps": raw / denominator})
        return {"objective": math.fsum(cell["normalized_crps"] for cell in cells) / len(cells), "cells": cells}

    candidates = [arm for arm in arms if arm["kind"] != "ordinary-control"]
    stage = request["stage"]
    result = {"schema": "gpuwm-ensemble-amplitude-selection-result.v1", "stage": stage,
              "status": "training-selected-not-held-out-validated", "selection_plan": request["selection_plan"],
              "campaign_plan": request["campaign_plan"], "score_summaries": request["score_summaries"],
              "training_cases": sorted(training), "held_out_cases": sorted(held_out),
              "products": quantities, "declared_arms": arms, "excluded_zero_denominators": excluded,
              "objective_definition": plan["objective"], "rankings": [],
              "raw_training_scores": [rows[key] for key in sorted(rows)],
              "selector_sha256": sha256(Path(__file__))}
    if stage == "recenter":
        by_amplitude = {}
        for arm in candidates:
            amplitude = arm.get("recenter_amplitude")
            if (arm["kind"] != "recentered" or arm.get("stochastic_amplitude") != 0
                    or amplitude not in plan["recenter_amplitudes"] or amplitude in by_amplitude):
                raise ValueError("recenter stage requires exactly each declared amplitude with stochastic physics off")
            by_amplitude[amplitude] = arm["score_recipe"]
        if set(by_amplitude) != set(plan["recenter_amplitudes"]):
            raise ValueError("recenter stage dropped a declared amplitude arm")
        ranking = [{"recenter_amplitude": amplitude, "arm": arm, **objective([arm])}
                   for amplitude, arm in sorted(by_amplitude.items())]
        ranking.sort(key=lambda row: (row["objective"], row["recenter_amplitude"]))
        result["rankings"] = ranking
        result["selected_recenter_amplitude"] = ranking[0]["recenter_amplitude"]
    elif stage == "stochastic-and-default":
        prior = load_pin(request["previous_selection"])
        if (prior.get("schema") != result["schema"] or prior.get("stage") != "recenter"
                or prior.get("selection_plan") != request["selection_plan"]
                or prior.get("campaign_plan") != request["campaign_plan"]):
            raise ValueError("stochastic fitting needs the pinned first-stage result from the same frozen plans")
        recenter = prior["selected_recenter_amplitude"]
        if recenter not in plan["recenter_amplitudes"]:
            raise ValueError("first-stage selection is outside the frozen recenter amplitude grid")
        recipes = campaign["recipes_conus"]
        amplitudes = [0.0, *plan["stochastic_amplitudes"]]
        by_setting = {}
        for arm in candidates:
            key = (arm["kind"], arm.get("stochastic_amplitude"))
            if (key[0] not in recipes or key[1] not in amplitudes or key in by_setting
                    or (arm["kind"] == "recentered" and arm.get("recenter_amplitude") != recenter)):
                raise ValueError("stochastic stage changed its recipe/amplitude grid or selected recenter amplitude")
            by_setting[key] = arm["score_recipe"]
        if set(by_setting) != {(recipe, amplitude) for recipe in recipes for amplitude in amplitudes}:
            raise ValueError("stochastic stage dropped a declared recipe or off/on amplitude arm")
        ranking = [{"stochastic_amplitude": amplitude,
                    **objective([by_setting[(recipe, amplitude)] for recipe in recipes])}
                   for amplitude in plan["stochastic_amplitudes"]]
        ranking.sort(key=lambda row: (row["objective"], row["stochastic_amplitude"]))
        selected = ranking[0]["stochastic_amplitude"]
        defaults = [{"recipe": recipe, "stochastic_amplitude": amplitude,
                     "arm": by_setting[(recipe, amplitude)], **objective([by_setting[(recipe, amplitude)]])}
                    for recipe in recipes for amplitude in (0.0, selected)]
        defaults.sort(key=lambda row: (row["objective"], PREFERENCE[row["recipe"]], row["stochastic_amplitude"]))
        result.update(previous_selection=request["previous_selection"], selected_recenter_amplitude=recenter,
                      selected_stochastic_amplitude=selected, rankings=ranking, default_ranking=defaults,
                      selected_default={key: defaults[0][key] for key in ("recipe", "stochastic_amplitude", "arm")})
    else:
        raise ValueError("selection stage must be recenter or stochastic-and-default")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("a frozen selection receipt already exists; use a new output path")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    result = selection(request)
    result["request"] = {"path": str(args.request), "sha256": sha256(args.request)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, result)


if __name__ == "__main__":
    main()
