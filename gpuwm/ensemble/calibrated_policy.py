"""Validate frozen training selection and untouched held-out score evidence."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timedelta

SELECTION_SCHEMA = "gpuwm-ensemble-amplitude-selection-result.v1"
SCORES_SCHEMA = "gpuwm-ensemble-campaign-scores.v1"


def read_reference(reference, *, base=None):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("calibration evidence needs its exact path and SHA256")
    path = Path(reference["path"])
    if not path.is_absolute():
        if base is None:
            raise ValueError("relative calibration evidence needs its containing policy directory")
        path = Path(base) / path
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("calibration-evidence authority changed from its pinned bytes")
    return json.loads(data), path.resolve()


def reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validated_selection(training_reference, held_out_reference, *, base=None,
                        validation_cases=None):
    """Accept only a completed final training stage and frozen held-out rows.

    Held-out values are checked for completeness and finiteness. They are
    never used here to choose a different recipe, amplitude or ranking.
    """
    selected, selected_path = read_reference(training_reference, base=base)
    held_out, _ = read_reference(held_out_reference, base=base)
    if (selected.get("schema") != SELECTION_SCHEMA or
            selected.get("stage") != "stochastic-and-default" or
            selected.get("status") != "training-selected-not-held-out-validated"):
        raise ValueError("automatic default needs the final frozen training selection, not a partial fit")
    plan, _ = read_reference(selected["selection_plan"], base=selected_path.parent)
    campaign, _ = read_reference(selected["campaign_plan"], base=selected_path.parent)
    if (plan.get("schema") != "ensemble-calibration.amplitude-selection.v1" or
            campaign.get("schema") != "gpuwm-ensemble-calibration.campaign.v1"):
        raise ValueError("selected default lacks its frozen calibration plans")
    training = set(selected.get("training_cases", ()))
    reserved = set(selected.get("held_out_cases", ()))
    if (not training or not reserved or training & reserved or
            training != set(plan["training_cases"]) or training != set(campaign["training_cases"]) or
            reserved != set(plan["held_out_cases"]) or reserved != set(campaign["held_out_cases"])):
        raise ValueError("calibration selection changed its frozen training/held-out split")
    cases = reserved if validation_cases is None else set(validation_cases)
    if not cases or not cases <= reserved:
        raise ValueError("source policy validation cases must be an explicit nonempty held-out subset")
    winner = selected.get("selected_default")
    ranking = selected.get("default_ranking")
    if (not isinstance(winner, dict) or set(winner) != {"recipe", "stochastic_amplitude", "arm"} or
            not isinstance(ranking, list) or not ranking or
            {key: ranking[0].get(key) for key in winner} != winner):
        raise ValueError("automatic policy does not retain the frozen training winner")
    preference = {"input-ensemble": 0, "recentered": 1, "time-lagged": 2, "multi-model": 3}
    def rank_key(row):
        value = row.get("objective")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("training default ranking has a nonfinite objective")
        return value, preference[row["recipe"]], row["stochastic_amplitude"]
    if sorted(ranking, key=rank_key) != ranking:
        raise ValueError("training winner is not the predeclared objective/tie winner")
    prior, prior_path = read_reference(selected["previous_selection"], base=selected_path.parent)
    if (prior.get("schema") != SELECTION_SCHEMA or prior.get("stage") != "recenter" or
            prior.get("selection_plan") != selected["selection_plan"] or
            prior.get("campaign_plan") != selected["campaign_plan"] or
            prior.get("selected_recenter_amplitude") != selected.get("selected_recenter_amplitude") or
            selected["selected_recenter_amplitude"] not in plan["recenter_amplitudes"] or
            selected.get("selected_stochastic_amplitude") not in plan["stochastic_amplitudes"] or
            winner["stochastic_amplitude"] not in (0, selected["selected_stochastic_amplitude"])):
        raise ValueError("automatic policy changed the frozen two-stage amplitudes")
    if (held_out.get("schema") != SCORES_SCHEMA or held_out.get("status") != "complete" or
            held_out.get("lead_hours") != plan["scored_lead_hours"] or
            campaign["scored_leads"] != plan["scored_lead_hours"]):
        raise ValueError("automatic policy needs complete held-out scores at the frozen lead hours")
    declared = {row["score_recipe"]: row for row in selected["declared_arms"]}
    arm = declared.get(winner["arm"])
    if (arm is None or arm["kind"] != winner["recipe"] or
            arm.get("stochastic_amplitude") != winner["stochastic_amplitude"] or
            winner["recipe"] == "recentered" and
            arm.get("recenter_amplitude") != selected["selected_recenter_amplitude"]):
        raise ValueError("training winner does not match its declared scientific arm")
    quantities = set(selected["products"])
    expected_members = campaign["time_lag_members"] if winner["recipe"] == "time-lagged" else campaign["members"]
    observed = set()
    masks = {}
    for row in held_out.get("products", ()):
        if row.get("held_out") is not True or row.get("case_id") not in reserved or row["case_id"] in training:
            raise ValueError("held-out evidence contains a training, unknown or relabelled case")
        scores = row.get("scores", {})
        if (scores.get("samples", 0) <= 0 or scores.get("missing_members") != 0 or
                not isinstance(scores.get("crps"), (int, float)) or
                not math.isfinite(scores["crps"]) or scores["crps"] < 0):
            raise ValueError("held-out evidence has incomplete member rows or nonfinite scores")
        key = row["case_id"], row["quantity"]
        mask = row.get("common_observation_mask")
        if mask is None or key in masks and masks[key] != mask:
            raise ValueError("held-out comparison arms changed their frozen observation masks")
        masks[key] = mask
        if row.get("recipe") == winner["arm"] and row["case_id"] in cases:
            if row["quantity"] not in quantities or key in observed:
                raise ValueError("held-out winner repeats or changes a scored product")
            if row.get("members") != expected_members or scores.get("members") != expected_members:
                raise ValueError("held-out winner changed its frozen scientific member count")
            observed.add(key)
    if observed != {(case, quantity) for case in cases for quantity in quantities}:
        raise ValueError("held-out evidence does not completely evaluate the unchanged training winner")
    return {"selection": selected, "campaign": campaign, "plan": plan,
            "winner": winner, "validation_cases": sorted(cases),
            "calibrated_member_count": expected_members}


def validate_policy_row(row, *, base=None):
    evidence = validated_selection(row["calibration_evidence"], row["held_out_evidence"],
        base=base, validation_cases=row.get("validation_cases"))
    winner, selection, campaign = evidence["winner"], evidence["selection"], evidence["campaign"]
    if row.get("kind") != winner["recipe"]:
        raise ValueError("source policy substitutes another recipe for the measured winner")
    if row.get("stochastic_amplitude") != winner["stochastic_amplitude"]:
        raise ValueError("source policy retunes the selected stochastic amplitude")
    if row.get("calibrated_member_count") != evidence["calibrated_member_count"]:
        raise ValueError("source policy changed the calibrated member count")
    if row["kind"] == "recentered" and (row.get("amplitude") != selection["selected_recenter_amplitude"] or
            row.get("donor_source") != campaign["donor_source"]):
        raise ValueError("source policy changed the measured donor or recenter amplitude")
    if row["kind"] == "time-lagged" and row.get("max_lag_hours") != campaign["time_lag_max_age_hours"]:
        raise ValueError("source policy changed the calibrated time-lag cycle-age span")
    stochastic = row.get("stochastic")
    if winner["stochastic_amplitude"] == 0 and stochastic is not None:
        raise ValueError("source policy enables stochastic physics for a selected off arm")
    if winner["stochastic_amplitude"] != 0 and not stochastic:
        raise ValueError("source policy omitted the selected stochastic controls")
    if stochastic != frozen_stochastic_controls(winner["stochastic_amplitude"], campaign["stochastic_on"]):
        raise ValueError("source policy changed the selected campaign stochastic parameters or consumers")
    if row["kind"] == "multi-model":
        roster, _ = read_reference(row["roster_evidence"], base=base)
        if roster.get("schema") != "ensemble-calibration.rosters.v1":
            raise ValueError("multi-model default requires its original frozen source rosters")
        rule = row["trajectory_rule"]
        for case in (*selection["training_cases"], *evidence["validation_cases"]):
            original = roster["cases"][case]["multi-model"]
            trajectories = trajectories_for_rule(rule, datetime.fromisoformat(original["base"]["cycle"]))
            expected = tuple((item["trajectory"]["source"], item["trajectory"]["cycle"], item["trajectory"]["member"])
                             for item in original["members"])
            if tuple((item.source, item.cycle.isoformat(), item.member) for item in trajectories) != expected:
                raise ValueError("multi-model default changed the calibrated source/relative-cycle roster")
        order = row.get("selection_order")
        if (not isinstance(order, list) or any(type(index) is not int for index in order) or
                sorted(order) != list(range(len(rule))) or
                len({rule[index]["source"] for index in order[:2]}) < 2):
            raise ValueError("multi-model selection order must retain each original index and start with distinct models")
    return evidence


def frozen_stochastic_controls(amplitude, enabled):
    """The frozen campaign's stated WRF-reference scaling, materialized once."""
    from gpuwm.ensemble.stochastic import StochasticConfig
    if type(amplitude) not in (int, float) or not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("frozen stochastic scaling needs a finite nonnegative amplitude")
    if amplitude == 0:
        return None
    enabled = tuple(enabled)
    if len(set(enabled)) != len(enabled) or set(enabled) - {"sppt", "skebs", "spp_conv", "spp_pbl", "spp_lsm"}:
        raise ValueError("frozen stochastic plan names unknown or repeated consumers")
    result = {}
    if "sppt" in enabled:
        result["sppt"] = {"stddev": StochasticConfig.wrf_reference("sppt").stddev * amplitude}
    if "skebs" in enabled:
        result["skebs"] = {name: {"backscatter": StochasticConfig.wrf_reference("skebs_" + name).backscatter * amplitude * amplitude}
                           for name in ("psi", "theta")}
    spp = tuple(name for name in ("conv", "pbl", "lsm") if "spp_" + name in enabled)
    if spp:
        result["spp"] = {name: int(name in spp) for name in ("conv", "pbl", "lsm")}
        result["spp_configs"] = {name: {"stddev": StochasticConfig.wrf_reference("spp_" + name).stddev * amplitude}
                                 for name in spp}
    return result


def trajectories_for_rule(rule, base_cycle):
    from gpuwm.ensemble.recipes import SourceTrajectory
    from gpuwm.source_cycles import cycle_grid_for
    if not isinstance(rule, list) or not rule:
        raise ValueError("multi-model policy needs a nonempty frozen trajectory rule")
    result = []
    for row in rule:
        if (not isinstance(row, dict) or set(row) != {"source", "member", "cycle_anchor", "cycle_offset_hours"} or
                row["cycle_anchor"] not in {"base-cycle", "latest-native-cycle"} or
                type(row["cycle_offset_hours"]) is not int or row["cycle_offset_hours"] > 0):
            raise ValueError("multi-model rule needs explicit source/member and nonfuture relative-cycle semantics")
        anchor = base_cycle
        if row["cycle_anchor"] == "latest-native-cycle":
            grid = cycle_grid_for(row["source"])
            if grid is None:
                raise ValueError(f"{row['source']} has no native cycle grid for the frozen multi-model rule")
            anchor = next(base_cycle - timedelta(hours=age) for age in range(24)
                          if (base_cycle - timedelta(hours=age)).hour in grid.hours)
        result.append(SourceTrajectory(row["source"], anchor + timedelta(hours=row["cycle_offset_hours"]), row["member"]))
    return tuple(result)


def portable_policy(document, destination, *, evidence_base=None):
    """Publish only the small semantic evidence closure, with relative paths.

    Original source receipts are read and left unchanged. Every derivative
    document retains its original receipt digest without an internal path.
    Runtime inventories, raw input paths and scorer command lines are omitted.
    """
    from copy import deepcopy
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    output = deepcopy(document)
    folder = destination.with_name(destination.stem + "-evidence")
    written = {}
    def emit(role, original_ref, data):
        original_hash = original_ref["sha256"]
        payload = {**data, "original_receipt_sha256": original_hash,
                   "evidence_scope": "portable semantic derivative; original receipt unchanged"}
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        key = role, original_hash
        if key in written:
            return written[key]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (role + "-" + original_hash[:16] + ".json")
        with path.open("xb") as stream:
            stream.write(raw)
        value = {"path": path.relative_to(destination.parent).as_posix(),
                 "sha256": hashlib.sha256(raw).hexdigest()}
        written[key] = value
        return value
    def local(reference):
        # A selection lives one level below the policy. Its own references
        # are relative to that containing directory, not the policy root.
        return {**reference, "path": Path(reference["path"]).name}
    for row in output["source_defaults"].values():
        selected, selected_path = read_reference(row["calibration_evidence"], base=evidence_base)
        plan, _ = read_reference(selected["selection_plan"], base=selected_path.parent)
        campaign, _ = read_reference(selected["campaign_plan"], base=selected_path.parent)
        prior, _ = read_reference(selected["previous_selection"], base=selected_path.parent)
        held, _ = read_reference(row["held_out_evidence"], base=evidence_base)
        # Frozen plans are small declarative science documents. Their
        # values contain no runtime source paths and remain intact.
        plan_ref = emit("selection-plan", selected["selection_plan"], plan)
        campaign_ref = emit("campaign-plan", selected["campaign_plan"], campaign)
        prior_keys = ("schema", "stage", "status", "training_cases", "held_out_cases", "products",
                      "selected_recenter_amplitude", "rankings", "objective_definition", "excluded_zero_denominators")
        prior_data = {key: prior[key] for key in prior_keys if key in prior}
        prior_data.update(selection_plan=local(plan_ref), campaign_plan=local(campaign_ref))
        prior_ref = emit("recenter-selection", selected["previous_selection"], prior_data)
        keys = ("schema", "stage", "status", "training_cases", "held_out_cases", "products",
                "selected_recenter_amplitude", "selected_stochastic_amplitude", "selected_default",
                "default_ranking", "rankings", "objective_definition", "excluded_zero_denominators")
        selection_data = {key: selected[key] for key in keys if key in selected}
        selection_data.update(selection_plan=local(plan_ref), campaign_plan=local(campaign_ref), previous_selection=local(prior_ref),
            declared_arms=[{key: arm[key] for key in ("score_recipe", "kind", "recenter_amplitude", "stochastic_amplitude")
                            if key in arm} for arm in selected["declared_arms"]])
        held_data = {key: held[key] for key in ("schema", "status", "lead_hours", "spinup_seconds") if key in held}
        held_data["products"] = [{key: product[key] for key in
            ("case_id", "recipe", "held_out", "quantity", "units", "members", "scores", "common_observation_mask")
            if key in product} for product in held["products"]]
        row["calibration_evidence"] = emit("training-selection", row["calibration_evidence"], selection_data)
        row["held_out_evidence"] = emit("held-out-scores", row["held_out_evidence"], held_data)
        if row["kind"] == "multi-model":
            roster, _ = read_reference(row["roster_evidence"], base=evidence_base)
            cases = tuple(case for case, recipes in roster["cases"].items() if "multi-model" in recipes)
            roster_data = {"schema": roster["schema"], "cases": {case: {"multi-model": {
                "base": roster["cases"][case]["multi-model"]["base"],
                "members": [{"trajectory": member["trajectory"]}
                            for member in roster["cases"][case]["multi-model"]["members"]]}} for case in cases}}
            row["roster_evidence"] = emit("source-rosters", row["roster_evidence"], roster_data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return destination
