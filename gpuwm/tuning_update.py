"""Propose another opt-in search window from a sealed training-only selection."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path, PurePosixPath
import re
import sys

from .tuning import PRODUCTS, SCHEMA, digest, numerical, read_json, verify_packet, write_json

PROPOSAL_SCHEMA = "gpuwm.tuning-proposal/v1"


def _sealed_round(packet):
    packet = Path(packet).resolve()
    seal = read_json(packet / "packet-sha256.json")
    files = seal.get("files")
    if seal.get("schema") != "gpuwm.tuning-packet/v1" or not isinstance(files, dict) or "round.json" not in files:
        raise ValueError("proposal needs a sealed tuning packet containing round.json")
    for name, sha256 in files.items():
        if not isinstance(name, str) or "\\" in name or ":" in name:
            raise ValueError("sealed packet file names must be relative portable paths")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("sealed packet paths cannot escape the packet")
        destination = packet.joinpath(*relative.parts).resolve()
        if not destination.is_relative_to(packet) or not re.fullmatch(r"[0-9a-f]{64}", str(sha256)):
            raise ValueError("sealed packet paths and hashes are invalid")
    round_ = verify_packet(packet)
    if round_.get("schema") != SCHEMA or round_.get("product") not in PRODUCTS:
        raise ValueError("proposal applies only to a sealed city or research product round")
    train, heldout = round_["train_days"], round_["heldout_days"]
    if not train or len(set(train)) != len(train) or len(set(heldout)) != len(heldout) or not set(train).isdisjoint(heldout):
        raise ValueError("the packet must have disjoint unique training and held-out days")
    members = {member["id"]: member for member in round_["members"]}
    if len(members) != len(round_["members"]) or round_["baseline_member"] not in members:
        raise ValueError("the packet has duplicate members or lacks its baseline")
    table = round_["parameter_space"]
    names = [row["name"] for row in table]
    if not names or len(set(names)) != len(names):
        raise ValueError("the packet needs a unique parameter-space table")
    if any(set(member["parameters"]) != set(names) for member in members.values()):
        raise ValueError("member settings do not match the sealed parameter table")
    return packet, round_, members


def _selection(path, packet, round_, members):
    selection = read_json(path)
    if selection.get("round_sha256") != digest(packet / "round.json"):
        raise ValueError("selection belongs to different sealed round bytes")
    training = selection["training"]
    decision = training["decision"]
    if decision.get("heldout") is not None or decision.get("heldout_days") or decision.get("heldout_periods"):
        raise ValueError("proposal input must be the training-only selection, with no held-out comparison")
    if training.get("heldout_spread") is not None:
        raise ValueError("proposal input contains held-out spread")
    ranking = training["ranking"]
    identities = [row["member"] for row in ranking]
    if not identities or len(set(identities)) != len(identities) or set(identities) != set(members):
        raise ValueError("training ranking must identify each sealed member exactly once")
    if decision["winner"] != identities[0] or decision.get("selection_metric") != "mean_daily_rmse_training_only":
        raise ValueError("training selection has no consistent frozen training winner")
    for row in ranking:
        if row.get("heldout") is not None or row.get("heldout_score") is not None:
            raise ValueError("proposal input contains held-out ranking scores")
    training_days = set(round_["train_days"])
    for row in training.get("metrics", []) + training.get("masks", []):
        if row["day"] not in training_days:
            raise ValueError("training selection contains a score day outside the training window")
    features = training["train_features"]
    if features.get("method") != "training_only_station_day_period_mean":
        raise ValueError("proposal requires native training-only station/day/period features")
    if not features["labels"] or any(label["day"] not in training_days for label in features["labels"]):
        raise ValueError("proposal features include a held-out or unknown day")
    feature_members = features["members"]
    if len(set(feature_members)) != len(feature_members) or not set(feature_members).issubset(members):
        raise ValueError("feature member labels do not match the sealed packet")
    if len(features["predictions"]) != len(feature_members):
        raise ValueError("feature predictions do not match the member table")
    if len(features["observations"]) != len(features["labels"]) or any(
        len(row) != len(features["labels"]) for row in features["predictions"]
    ):
        raise ValueError("feature vectors do not match the native feature labels")
    candidates = [identity for identity in identities if identity.startswith("setting-")]
    if not candidates or not set(candidates).issubset(feature_members):
        raise ValueError("training selection has no complete candidate feature set")
    return training, features, candidates


def _active(member, row):
    name = row["name"]
    parameters = member["parameters"]
    schemes = row.get("schemes")
    applies = not schemes or parameters["land_model"] in schemes
    applies = applies and (not row.get("requires_urban") or member.get("physics", {}).get(
        "sf_urban_physics", parameters["urban_option"]
    ) == 1)
    if "active_parameters" in member and applies != (name in member["active_parameters"]):
        raise ValueError(f"sealed active parameter metadata disagrees for {member['id']} {name}")
    return applies


def _scalar(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numerical scalar")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def propose(packet, selection, numerics, output, *, method="ranked", keep=6, variance_k2=1.0, damping=0.5):
    if method not in {"ranked", "eki"}:
        raise ValueError("proposal method must be ranked or eki")
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("ranked proposal count must be a positive integer")
    variance_k2 = _scalar(variance_k2, "feature variance in K squared", positive=True)
    damping = _scalar(damping, "damping")
    if not 0 <= damping <= 1:
        raise ValueError("damping must be between zero and one")
    packet, round_, members = _sealed_round(packet)
    output = Path(output).resolve()
    if output.is_relative_to(packet) or output == Path(selection).resolve():
        raise ValueError("proposal output must be separate from the sealed packet and selection")
    if output.exists():
        raise ValueError("proposal output already exists; preserve it and choose a new output path")
    round_sha256 = digest(packet / "round.json")
    packet_sha256 = digest(packet / "packet-sha256.json")
    selection_sha256 = digest(selection)
    training, features, candidates = _selection(selection, packet, round_, members)
    numerics_sha256 = digest(numerics) if method == "eki" else None
    chosen = candidates[:keep] if method == "ranked" else candidates
    settings = {identity: copy.deepcopy(members[identity]["parameters"]) for identity in chosen}
    groups = []
    if method == "eki":
        discrete = [row["name"] for row in round_["parameter_space"] if row["kind"] in {"categorical", "integer"}]
        if not {"land_model", "bl_mynn_mixlength", "urban_option"}.issubset(discrete):
            raise ValueError("EKI groups must keep land model, mixing option and urban option categorical")
        grouped = {}
        for identity in candidates:
            group = tuple(members[identity]["parameters"][name] for name in discrete)
            grouped.setdefault(group, []).append(identity)
        feature_indices = {identity: index for index, identity in enumerate(features["members"])}
        for values, identities in grouped.items():
            columns = [row for row in round_["parameter_space"] if row["kind"] == "continuous"
                       and all(_active(members[identity], row) for identity in identities)]
            metadata = {"categorical_settings": dict(zip(discrete, values)), "source_members": identities,
                        "continuous_parameters": [row["name"] for row in columns]}
            if len(identities) < 2 or not columns:
                groups.append({**metadata, "status": "retained_ranked_no_update",
                               "reason": "fewer than two members" if len(identities) < 2 else "no shared active continuous parameters"})
                continue
            request = {
                "parameters": [[settings[identity][row["name"]] for row in columns] for identity in identities],
                "predictions": [features["predictions"][feature_indices[identity]] for identity in identities],
                "observations": features["observations"],
                "variances": [variance_k2 for _ in features["observations"]],
                "bounds": [[row["min"], row["max"]] for row in columns],
                "damping": damping,
            }
            updated = numerical(numerics, "eki", request)
            matrix = updated["parameters"]
            if len(matrix) != len(identities) or any(len(row) != len(columns) for row in matrix):
                raise ValueError("native EKI returned an incompatible parameter matrix")
            if updated.get("requires_new_forecasts") is not True or updated.get("solve_dimension") != len(identities):
                raise ValueError("native EKI did not return a member-sized forecast-required update")
            for identity, values in zip(identities, matrix):
                for row, value in zip(columns, values):
                    settings[identity][row["name"]] = _scalar(value, "native updated parameter")
            groups.append({**metadata, "status": "updated_needs_forecasts",
                           "numerical_receipt": {key: value for key, value in updated.items() if key != "parameters"}})
    proposals = [{"id": f"proposal-{index + 1:03d}", "source_member": identity,
                  "parameters": settings[identity]} for index, identity in enumerate(chosen)]
    result = {
        "schema": PROPOSAL_SCHEMA,
        "method": method,
        "product": round_["product"],
        "source_round_sha256": round_sha256,
        "source_packet_sha256": packet_sha256,
        "source_selection_sha256": selection_sha256,
        "source_numerics_sha256": numerics_sha256,
        "training_days_used": round_["train_days"],
        "frozen_training_winner": training["decision"]["winner"],
        "ranked_candidates": candidates,
        "members": proposals,
        "groups": groups,
        "feature_variance_k2": variance_k2 if method == "eki" else None,
        "damping": damping if method == "eki" else None,
        "feature_variance_assumption": "fixed per station/day/period mean; temporal and spatial correlation is not modeled" if method == "eki" else None,
        "requires_new_forecasts": True,
        "requires_new_unseen_holdout": True,
        "decision": "proposed settings only; no measured updated skill, default change or deployment",
    }
    if digest(packet / "packet-sha256.json") != packet_sha256:
        raise ValueError("packet seal changed during proposal construction")
    verify_packet(packet)
    if digest(packet / "round.json") != round_sha256 or digest(selection) != selection_sha256:
        raise ValueError("sealed round or training selection changed during proposal construction")
    if method == "eki" and digest(numerics) != numerics_sha256:
        raise ValueError("native numerical executable changed during proposal construction")
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("packet", "selection", "numerics", "out"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--method", choices=("ranked", "eki"), default="ranked")
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--variance-k2", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        result = propose(args.packet, args.selection, args.numerics, args.out, method=args.method,
                         keep=args.keep, variance_k2=args.variance_k2, damping=args.damping)
        print(f"{result['method']}: {len(result['members'])} proposed settings; fresh forecasts and unseen held-out days required")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"tuning update: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
