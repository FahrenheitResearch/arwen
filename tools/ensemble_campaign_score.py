"""Match archived members natively and score a frozen observation campaign.

JSON requests supply cases, time windows, pinned observation plans and final
archive/product manifests. This module only validates metadata, runs the
native matcher/scorer, and joins their tables without changing field values.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess

# Inside the wheel this is tools.ensemble_calibration_score; the bare form
# is what a run by path resolves, where tools/ is sys.path[0].
try:
    from tools.ensemble_calibration_score import score, sha256
except ImportError:
    from ensemble_calibration_score import score, sha256

QUANTITIES = {
    "temperature_2m": ("temperature2", "K"),
    "wind_speed_10m": ("wind10", "m s-1"),
    "precipitation_accumulation": ("rain_total", "mm"),
}


def instant(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def load_pin(reference):
    path = Path(reference["path"])
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != reference["sha256"]:
        raise ValueError(f"pinned input changed: {path}")
    return json.loads(body)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def label(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("case and recipe labels must be simple distinct filename labels")
    return value


def product_contract(manifest, archive, members, valid_times):
    order = [row["member_id"] for row in members]
    seeds = {row["member_id"]: row["seed"] for row in members}
    if (manifest.get("schema") != "gpuwm-ensemble-output.v2"
            or manifest.get("member_order") != order
            or archive.get("member_order") != order
            or manifest.get("members_requested") != len(order)
            or manifest.get("probability_denominator") != len(order)):
        raise ValueError("production and diagnostic manifests must retain the complete requested roster")
    for document in (manifest, archive):
        rows = document.get("member_metadata", [])
        if len(rows) != len(order) or {row["member_id"]: row["seed"] for row in rows} != seeds:
            raise ValueError("production or diagnostic manifest changed member seeds")
    fields = {}
    for row in manifest["products"]:
        name = row["field"]
        if name in fields:
            raise ValueError("production manifest has duplicate field contracts")
        fields[name] = row
    result = {}
    for quantity, (field, units) in QUANTITIES.items():
        row = fields.get(field)
        if row is None or row["units"] != units or row["comparison"] != "ge":
            raise ValueError(f"{field} requires the native scorer's declared units and >= event")
        thresholds = row["thresholds"]
        if not thresholds or len(set(thresholds)) != len(thresholds):
            raise ValueError(f"{field} has absent or duplicate production thresholds")
        for value in thresholds:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or struct.unpack("<f", struct.pack("<f", value))[0] != value):
                raise ValueError(f"{field} thresholds must already be the production float32 values")
        result[quantity] = thresholds
    frames = {}
    for frame in manifest["frames"]:
        clock = instant(frame["valid_time"])
        if clock in frames:
            raise ValueError("production manifest repeats a valid time")
        frames[clock] = frame
    for clock in valid_times:
        frame = frames.get(clock)
        if (frame is None or frame["status"] != "complete"
                or frame["members_received"] != sorted(order)
                or frame["members_expected"] != len(order)
                or not frame["products"]
                or any(field not in frame["available_fields"] for field, _ in QUANTITIES.values())):
            raise ValueError("a scored production frame is incomplete or lacks a scored field")
    return result


def join_matches(paths, destination, member_labels):
    """Concatenate native tables and bind their unmodified observed columns."""
    mask = hashlib.sha256()
    rows = 0
    observed = 0
    seen = set()
    with destination.open("w", encoding="utf-8", newline="\n") as target:
        target.write("sample_id\tweight\tobserved\t" + "\t".join(member_labels) + "\n")
        for path in paths:
            with path.open(encoding="utf-8") as source:
                header = source.readline().rstrip("\r\n").split("\t")
                if header != ["sample_id", "weight", "observed", *member_labels]:
                    raise ValueError("native matching changed the requested member order")
                for line in source:
                    columns = line.rstrip("\r\n").split("\t")
                    if len(columns) != len(header) or columns[0] in seen:
                        raise ValueError("native matching repeats a sample or changes its row width")
                    seen.add(columns[0])
                    if any(not math.isfinite(float(value)) for value in columns[3:]):
                        raise ValueError("a candidate has an unavailable member value; common-mask scoring cannot drop it silently")
                    observed += math.isfinite(float(columns[2]))
                    mask.update(("\t".join(columns[:3]) + "\n").encode())
                    target.write("\t".join(columns) + "\n")
                    rows += 1
    return {"sha256": mask.hexdigest(), "rows": rows, "observed_rows": observed,
            "definition": "ordered sample_id, weight and observed columns, including explicit missing observations"}


def run_campaign(request, *, matcher, scorer, out):
    out.mkdir(parents=True, exist_ok=False)
    save(out / "campaign-request.json", request)
    leads = request.get("lead_hours", list(range(3, 13)))
    if (not leads or any(type(lead) is not int or lead < 1 for lead in leads)
            or leads != sorted(set(leads))):
        raise ValueError("scored leads must be explicitly ordered distinct positive hours")
    if len({case["case_id"] for case in request["cases"]}) != len(request["cases"]):
        raise ValueError("campaign repeats a case")
    summary = {"schema": "gpuwm-ensemble-campaign-scores.v1", "status": "failed", "products": [],
               "lead_hours": leads, "spinup_seconds": request["spinup_seconds"],
               "matcher_sha256": sha256(matcher), "scorer_sha256": sha256(scorer),
               "scope": "matched observation scores; amplitude selection and scientific qualification are separate"}
    summary["request"] = {"path": str(out / "campaign-request.json"), "sha256": sha256(out / "campaign-request.json")}
    summary["orchestrator_sha256"] = sha256(Path(__file__))
    try:
        for case in request["cases"]:
            case_id = label(case["case_id"])
            case_root = out / case_id
            case_root.mkdir()
            start = instant(case["start_time"])
            valid_times = [start + timedelta(hours=lead) for lead in leads]
            plan = load_pin(case["plan"])
            precip = {row["lead_hour"]: row for row in case["precipitation_observations"]}
            if len(precip) != len(case["precipitation_observations"]) or any(lead not in precip for lead in leads):
                raise ValueError("campaign precipitation observations omit or repeat a scored lead")
            if not case["runs"] or len({run["recipe"] for run in case["runs"]}) != len(case["runs"]):
                raise ValueError("campaign needs distinct recipe runs")
            common_masks, common_thresholds = {}, {}
            for run in case["runs"]:
                recipe = label(run["recipe"])
                root = case_root / recipe
                root.mkdir()
                archive = load_pin(run["archive"]["manifest"])
                manifest = load_pin(run["production_manifest"])
                members = run["members"]
                if (not members or any(type(m["member_id"]) is not int or type(m["seed"]) is not int
                    or not 0 <= m["member_id"] < 2**64 or not 0 <= m["seed"] < 2**64 for m in members)):
                    raise ValueError("campaign members need exact uint64 identities and seeds")
                if instant(archive["precipitation_accumulation_start"]) != start:
                    raise ValueError("archive start differs from the frozen case window")
                thresholds = product_contract(manifest, archive, members, valid_times)
                for quantity, (_, units) in QUANTITIES.items():
                    if quantity in common_thresholds and thresholds[quantity] != common_thresholds[quantity]:
                        raise ValueError("recipe thresholds differ; Brier comparisons require the same production events")
                    common_thresholds[quantity] = thresholds[quantity]
                    matches, source_receipts, observation_receipts, match_receipts = [], [], [], []
                    for lead, clock in zip(leads, valid_times):
                        job = {"plan": case["plan"]["path"], "quantity": quantity, "valid_time": clock.isoformat(),
                               "archive": run["archive"], "members": members}
                        if quantity == "precipitation_accumulation":
                            path = Path(precip[lead]["path"])
                            if sha256(path) != precip[lead]["sha256"]:
                                raise ValueError("pinned precipitation observation changed")
                            job["precipitation_observation"] = str(path)
                        prefix = root / f"{quantity}-lead-{lead:02d}"
                        request_path, table = prefix.with_suffix(".request.json"), prefix.with_suffix(".tsv")
                        save(request_path, job)
                        result = subprocess.run([str(matcher), "match-diagnostics", str(request_path), str(table)],
                                                check=True, text=True, capture_output=True)
                        receipt = json.loads(result.stdout)
                        if receipt["plan"]["sha256"] != case["plan"]["sha256"]:
                            raise ValueError("native matcher used a different frozen plan")
                        receipt_path = prefix.with_suffix(".receipt.json")
                        save(receipt_path, receipt)
                        matches.append(table)
                        source_receipts.extend(receipt["source_receipts"])
                        observation_receipts.extend(receipt["observation_receipts"])
                        match_receipts.append({"path": str(receipt_path), "sha256": sha256(receipt_path)})
                    table = root / f"{quantity}-pooled.tsv"
                    mask = join_matches(matches, table, [m["id"] for m in members])
                    if quantity in common_masks and mask != common_masks[quantity]:
                        raise ValueError("recipe changed frozen observed samples, weights or values")
                    common_masks[quantity] = mask
                    provenance = {"case_id": case_id, "recipe": recipe, "held_out": case["held_out"],
                        "quantity": quantity, "units": units, "member_ids": [m["id"] for m in members], "members": members,
                        "source_receipts": source_receipts, "observation_receipts": observation_receipts,
                        "match_receipt": match_receipts, "valid_start": valid_times[0].isoformat(), "valid_end": valid_times[-1].isoformat(),
                        "spinup_seconds": request["spinup_seconds"], "lead_hours": leads, "common_observation_mask": mask,
                        "production_manifest": run["production_manifest"], "diagnostic_manifest": run["archive"]["manifest"],
                        "thresholds": thresholds[quantity], "threshold_interpretation": "actual production float32 values, unchanged",
                        "aggregation": "unit-weight pooled frozen observation samples for this case and quantity"}
                    provenance_path = root / f"{quantity}-provenance.json"
                    save(provenance_path, provenance)
                    receipt = score(binary=scorer, matched=table, thresholds=",".join(map(repr, thresholds[quantity])), provenance=provenance_path)
                    if receipt["scores"]["missing_members"] or receipt["scores"]["samples"] != mask["observed_rows"]:
                        raise ValueError("native scoring changed the frozen complete-member denominator")
                    score_path = root / f"{quantity}-scores.json"
                    save(score_path, receipt)
                    summary["products"].append({"case_id": case_id, "recipe": recipe, "held_out": case["held_out"],
                        "quantity": quantity, "units": units, "members": len(members), "scores": receipt["scores"],
                        "common_observation_mask": mask, "receipt": {"path": str(score_path), "sha256": sha256(score_path)}})
                load_pin(run["archive"]["manifest"])
                load_pin(run["production_manifest"])
            load_pin(case["plan"])
            save(case_root / "common-observation-masks.json", common_masks)
        summary["status"] = "complete"
        metrics = ("samples", "missing_observations", "crps", "fair_crps", "ensemble_mean_rmse",
                   "ensemble_mean_bias", "rms_member_sample_spread", "spread_skill_ratio")
        table = out / "score-table.tsv"
        with table.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("case_id\trecipe\theld_out\tquantity\tunits\tmembers\t" + "\t".join(metrics) + "\tthreshold_brier_scores\n")
            for product in summary["products"]:
                prefix = [str(product[key]) for key in ("case_id", "recipe", "held_out", "quantity", "units", "members")]
                values = ["null" if product["scores"][key] is None else str(product["scores"][key]) for key in metrics]
                brier = [{"threshold": row["threshold"], "brier_score": row["brier_score"]}
                         for row in product["scores"]["threshold_scores"]]
                stream.write("\t".join(prefix + values + [json.dumps(brier, separators=(",", ":"))]) + "\n")
        summary["score_table"] = {"path": str(table), "sha256": sha256(table)}
        return summary
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(out / "campaign-scores.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--matcher", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    run_campaign(request, matcher=args.matcher.resolve(), scorer=args.scorer.resolve(), out=args.out)


if __name__ == "__main__":
    main()
