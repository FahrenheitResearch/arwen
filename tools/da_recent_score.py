#!/usr/bin/env python3
"""Score every recent-case DA member and matched clean-start member.

All weather-field reads, remaps and metrics belong to the existing native scorer.
This controller composes commands, binds files, and summarizes scalar rows only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LEADS = (1, 3, 6)
THRESHOLDS = (1, 5, 10)
WIDTHS = (10, 25, 50)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def utc(value):
    stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() != dt.timedelta(0):
        raise ValueError("explicit UTC forecast fork required")
    return stamp


def metric_keys():
    for lead in LEADS:
        for metric in ("footprint_rain_ratio", "domain_rain_ratio", "echo_area_multiple"):
            yield lead, metric, None, None
        for threshold in THRESHOLDS:
            for width in WIDTHS:
                yield lead, "fss", threshold, width


def pending_rows(item, event, analysis, reason):
    return [{"schema": "regional-rain/v1", "event": event, "seed": item["seed"],
             "member": item["member"], "arm": item["arm"], "product": item["product"],
             "analysis_end": analysis, "lead_hours": lead, "window_seconds": [(lead-1)*3600, lead*3600],
             "metric": metric, "threshold_mm_h": threshold, "scale_km": width,
             "value": None, "status": "pending", "measurement_status": "pending",
             "screen_status": None, "reason": reason, "paired_gain": None}
            for lead, metric, threshold, width in metric_keys()]


def plan(case, run, truth, out, seeds, members=32, domain=1, saved_inputs=None, control_run=None):
    if type(members) is not int or not 1 <= members <= 64:
        raise ValueError("members must be an explicit integer in 1..64")
    if type(domain) is not int or domain < 1:
        raise ValueError("positive domain id required")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("distinct nonnegative integer seeds required")
    analyses = case["analysis_times_utc"]
    if not analyses:
        raise ValueError("case has no analysis times")
    fork = case.get("forecast_fork_utc", analyses[-1])
    utc(fork)
    if utc(fork) != utc(analyses[-1]):
        raise ValueError("case fork differs from final analysis")
    slots = run.get("slots", [])
    if not slots or utc(slots[-1]["analysis_time"]) != utc(fork):
        raise ValueError("run manifest fork differs from the case clock")
    if control_run is not None:
        if (not control_run.get("slots") or utc(control_run["slots"][-1]["analysis_time"]) != utc(fork)
                or any(control_run.get(key) != run.get(key) for key in ("engine_sha", "prepared_content_sha256", "physics_profile"))):
            raise ValueError("matched clean-start run identity or fork differs from DA")
    out = Path(out).resolve()
    run_root = Path(run["out"]).resolve()
    control_root = Path((control_run or run)["out"]).resolve()
    suffix = "" if domain == 1 else f"_d{domain:02d}"
    longitude = case.get("center_lon", case.get("longitude", -89.0))
    latitude = case.get("center_lat", case.get("latitude", 31.0))
    for coord in (longitude, latitude):
        if not math.isfinite(float(coord)):
            raise ValueError("finite equal-area center required")
    inputs = {}
    if saved_inputs is not None:
        if saved_inputs.get("schema") != "da-rerun.saved-score-inputs.v1":
            raise ValueError("wrong saved score input schema")
        for row in saved_inputs["members"]:
            key = (row["seed"], row["member"])
            if key in inputs:
                raise ValueError("duplicate saved member descriptor")
            inputs[key] = row
    jobs = []
    for seed in seeds:
        for member in range(members):
            directory = out/f"seed-{seed}"/f"m{member:03d}"
            supplied = inputs.get((seed, member), {})
            manifests = {arm: Path(supplied[key]).resolve() if key in supplied else directory/f"{arm}-input.json"
                         for arm, key in (("da", "da"), ("no-da", "no_da"))}
            builders = {}
            for arm in ("da", "no-da"):
                if (arm == "da" and "da" in supplied) or (arm == "no-da" and "no_da" in supplied):
                    builders[arm] = None
                    continue
                if arm == "da":
                    frames = run_root/f"seed-{seed}"/"da"
                    if run.get("schema") != "gpuwm-da.recent-run.v1":
                        frames /= f"{len(slots)-1:03d}"
                    frames /= "composites"
                else:
                    frames = control_root/f"seed-{seed}"/"no-da"/"composites"
                builders[arm] = [sys.executable, str(ROOT/"tools/regional_rain_manifest.py"), "model",
                    "--frames", str(frames), "--pattern", f"wrfout_*_{member}{suffix}.nc",
                    "--analysis-end", fork, "--center-lon", str(longitude), "--center-lat", str(latitude),
                    "--out", str(manifests[arm])]
            # Baseline is prepared once, then reused for the paired DA score.
            for arm in ("no-da", "da"):
                product = f"matched_clean_member_{member:03d}" if arm == "no-da" else f"member_{member:03d}"
                score_path = directory/f"{arm}.jsonl"
                score = [sys.executable, str(ROOT/"tools/regional_rain_score.py"),
                         "--forecast", str(manifests[arm]), "--truth", str(Path(truth).resolve()),
                         "--analysis-end", fork, "--event", case.get("case_id", "recent-gulf-se-development"),
                         "--seed", str(seed), "--product", product, "--out", str(score_path)]
                if arm == "da":
                    score += ["--baseline", str(manifests["no-da"])]
                jobs.append({"seed": seed, "member": member, "arm": arm, "product": product,
                             "builder": builders[arm], "argv": score, "manifest": str(manifests[arm]),
                             "baseline_manifest": str(manifests["no-da"]) if arm == "da" else None,
                             "scores": str(score_path), "analysis_end": fork})
    return jobs


def command(command, log, timeout):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GPUWM_NO_LOCAL_GPU="1", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "nt": process.kill()
                else: os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {"exit_code": 124, "wall_seconds": time.monotonic()-started,
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "timeout",
                    "argv": command, "log": str(log)}
    return {"exit_code": code, "wall_seconds": time.monotonic()-started,
            "status": "complete" if code == 0 else "refused", "argv": command, "log": str(log)}


def scalar_summary(rows, seeds, members):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["seed"], row["arm"], row["lead_hours"], row["metric"],
               row.get("threshold_mm_h"), row.get("scale_km"))
        grouped[key].append(row)
    result = []
    for (seed, arm, lead, metric, threshold, width), values in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        complete = [row for row in values if row.get("measurement_status") == "complete"
                    and isinstance(row.get("value"), (int, float)) and math.isfinite(row["value"])]
        roster = {row["member"] for row in values}
        valid = len(values) == members and roster == set(range(members)) and len(complete) == members
        gain = [row.get("paired_gain") for row in values]
        all_gains = valid and all(isinstance(value, (int, float)) and math.isfinite(value) for value in gain)
        result.append({"seed": seed, "arm": arm, "lead_hours": lead, "metric": "mean_member_"+metric,
                       "threshold_mm_h": threshold, "scale_km": width, "expected_members": members,
                       "complete_members": len(complete), "value": math.fsum(row["value"] for row in complete)/members if valid else None,
                       "mean_member_paired_gain": math.fsum(gain)/members if all_gains else None,
                       "measurement_status": "complete" if valid else "pending", "scientific_gate": "pending",
                       "definition": "arithmetic mean of member scalar scores; not a score of ensemble-mean weather fields"})
    return result


def execute(jobs, out, *, seeds, members, time_budget_seconds=1200, runner=command):
    if not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
        raise ValueError("positive finite shared scoring budget required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+time_budget_seconds
    rows, receipts, built = [], [], {}
    for index, item in enumerate(jobs):
        score_path = Path(item["scores"])
        score_path.parent.mkdir(parents=True, exist_ok=True)
        if score_path.exists():
            raise ValueError("fresh score output required: "+str(score_path))
        reason = None
        remaining = deadline-time.monotonic()
        manifest = Path(item["manifest"])
        if remaining <= 0:
            reason = "shared scoring time budget exhausted"
        elif item["builder"] is not None:
            record = runner(item["builder"], score_path.with_suffix(".manifest.log"), remaining)
            receipts.append(record)
            if record.get("status") == "interrupted":
                deadline = time.monotonic()
            built[str(manifest)] = record["exit_code"] == 0 and manifest.is_file()
            if not built[str(manifest)]:
                reason = "native model manifest unavailable or refused; see manifest log"
        elif not manifest.is_file():
            reason = "saved field manifest unavailable"
        if item["baseline_manifest"] and not Path(item["baseline_manifest"]).is_file():
            reason = "matched clean-start manifest unavailable"
        if reason is None:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                reason = "shared scoring time budget exhausted"
            else:
                record = runner(item["argv"], score_path.with_suffix(".score.log"), remaining)
                receipts.append(record)
                if record.get("status") == "interrupted":
                    deadline = time.monotonic()
                if record["exit_code"] != 0 or not score_path.is_file():
                    reason = "native rain scorer unavailable or refused; see score log"
        if reason is not None:
            event = item["argv"][item["argv"].index("--event")+1]
            values = pending_rows(item, event, item["analysis_end"], reason)
            score_path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in values))
        else:
            values = [json.loads(line) for line in score_path.read_text().splitlines() if line.strip()]
            keys = {(row["lead_hours"], row["metric"], row.get("threshold_mm_h"), row.get("scale_km")) for row in values}
            if len(values) != 36 or keys != set(metric_keys()):
                raise ValueError("native scorer row roster differs from the 1/3/6-hour contract")
            for row in values:
                row.update(member=item["member"], arm=item["arm"])
        rows.extend(values)
        save(out/"command-receipts.json", {"schema": "da-rerun.recent-score-commands.v1", "commands": receipts,
                                           "finished_products": index+1, "total_products": len(jobs)})
    allrows = out/"all-members.jsonl"
    allrows.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False)+"\n" for row in rows), encoding="utf-8")
    summary = {"schema": "da-rerun.recent-score-campaign.v1", "scientific_gate": "pending",
               "seeds": seeds, "members_per_seed": members, "rows": len(rows),
               "complete_measurements": sum(row.get("measurement_status") == "complete" for row in rows),
               "all_members_sha256": sha(allrows), "member_scalar_summaries": scalar_summary(rows, seeds, members),
               "ensemble_mean_field_status": "pending; ordinary WRF snapshots are not native diagnostic-spool mean inputs",
               "qualification_reason": "one development event; ensemble-mean fields, event-blocked bounds and ancillary gate losses are not qualified",
               "shared_scoring_budget_seconds": time_budget_seconds,
               "wall_seconds": max(0, time.monotonic()-(deadline-time_budget_seconds))}
    save(out/"campaign-summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--control-run-manifest", type=Path, help="separately scheduled matched clean-start output root")
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--fetch-receipt", type=Path)
    parser.add_argument("--truth-out", type=Path)
    parser.add_argument("--saved-inputs", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261003, 20261004])
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-budget-seconds", type=float, default=1200)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    case = json.loads(args.case.read_text(encoding="utf-8-sig"))
    run = json.loads(args.run_manifest.read_text(encoding="utf-8-sig"))
    control_run = json.loads(args.control_run_manifest.read_text(encoding="utf-8-sig")) if args.control_run_manifest else None
    from tools.regional_rain_score import source_revision
    revision = source_revision()
    if run.get("engine_sha") is not None and run["engine_sha"] != revision:
        raise ValueError("run engine SHA differs from the pinned scorer source")
    if not math.isfinite(args.time_budget_seconds) or args.time_budget_seconds <= 0:
        parser.error("positive finite shared scoring budget required")
    campaign_started = time.monotonic()
    truth_command_receipt = None
    if args.truth is None:
        if any(value is None for value in (args.inventory, args.fetch_receipt, args.truth_out)):
            parser.error("provide --truth or all of --inventory --fetch-receipt --truth-out")
        args.truth = args.truth_out/"truth.json"
        if not args.plan_only:
            if args.truth.exists():
                raise ValueError("fresh truth preparation output required")
            truth_command = [sys.executable, str(ROOT/"tools/da_recent_observations.py"), "truth",
                "--inventory", str(args.inventory), "--fetch-receipt", str(args.fetch_receipt),
                "--out", str(args.truth_out), "--center-lon", str(case.get("center_lon", case.get("longitude", -89))),
                "--center-lat", str(case.get("center_lat", case.get("latitude", 31)))]
            truth_command_receipt = command(truth_command, args.out/"truth-preparation.log", args.time_budget_seconds)
            save(args.out/"truth-command-receipt.json", truth_command_receipt)
    saved = json.loads(args.saved_inputs.read_text()) if args.saved_inputs else None
    jobs = plan(case, run, args.truth, args.out, args.seeds, args.members, args.domain, saved, control_run)
    save(args.out/"score-plan.json", {"case_sha256": sha(args.case), "run_manifest_sha256": sha(args.run_manifest),
                                     "truth": str(args.truth), "jobs": jobs, "scientific_gate": "pending"})
    if args.plan_only:
        print(json.dumps({"status": "score-plan-only", "products": len(jobs), "scientific_gate": "pending"}))
        return 0
    remaining = max(.000001, args.time_budget_seconds-(time.monotonic()-campaign_started))
    if truth_command_receipt and truth_command_receipt.get("status") == "interrupted":
        remaining = .000001
    result = execute(jobs, args.out, seeds=args.seeds, members=args.members, time_budget_seconds=remaining)
    result.update(shared_scoring_budget_seconds=args.time_budget_seconds,
                  truth_command=truth_command_receipt, wall_seconds=time.monotonic()-campaign_started,
                  case_sha256=sha(args.case), run_manifest_sha256=sha(args.run_manifest),
                  truth_manifest_sha256=sha(args.truth) if args.truth.is_file() else None,
                  scorer_commit=revision, run_engine_sha=run.get("engine_sha"),
                  source_authority_status="verified" if run.get("engine_sha") == revision and revision is not None else "unspecified synthetic/source-only authority",
                  control_run_manifest_sha256=sha(args.control_run_manifest) if args.control_run_manifest else None,
                  controller_sha256=sha(Path(__file__)))
    save(args.out/"campaign-summary.json", result)
    print(json.dumps({"summary": str(args.out/"campaign-summary.json"), "rows": result["rows"],
                      "complete_measurements": result["complete_measurements"], "scientific_gate": "pending"}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print("recent scoring refused: "+str(error), file=sys.stderr)
        raise SystemExit(2)
