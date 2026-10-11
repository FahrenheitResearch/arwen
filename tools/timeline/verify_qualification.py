"""C1 real-run receipts and full forecast artifact comparisons.

No arrays are decoded here. The supplied C4 gate hashes complete files and
uses its Rust readers when necessary. Gate views retain every history and
restart plus the engine's complete canonical/prepared/backend evidence.
Bookkeeping logs remain separately retained with their original reports.
"""

import argparse
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gpuwm.timeline.report import Timeline, union
from tools.bench.identity_gate import EVIDENCE_KEYS, run_gate, sha


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def science_evidence(document, prefix=""):
    """Keep every recognized scientific evidence field, preserving its tree."""
    if isinstance(document, dict):
        result = {}
        for key, value in document.items():
            if key in EVIDENCE_KEYS or (key == "content_sha256" and "prepared" in prefix.lower()):
                result[key] = value
            else:
                selected = science_evidence(value, prefix + "/" + key)
                if selected:
                    result[key] = selected
        return result
    if isinstance(document, list):
        selected = [science_evidence(value, prefix) for value in document]
        return selected if any(selected) else []
    return {}


def artifact_view(run, view):
    if view.exists():
        raise ValueError("artifact view already exists; never overwrite comparison evidence")
    forecast = run / "artifacts/chain/run"
    report = forecast / "report.json"
    document = json.loads(report.read_text())
    if not document.get("final_state_digest"):
        raise ValueError("real engine final-state digest is absent; history alone cannot prove carried state")
    view.mkdir(parents=True)
    records = []
    for path in sorted(forecast.rglob("*")):
        if path.is_symlink():
            raise ValueError("forecast artifact symlink would break evidence ownership")
        if path.is_file() and (path.name.startswith(("wrfout", "wrfrst", "gpuwmrst")) or path.suffix.lower() in (".nc", ".npz", ".npy", ".bin")):
            relative = path.relative_to(forecast)
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
            records.append({"source": str(path), "view": str(relative),
                            "bytes": path.stat().st_size, "sha256": sha(path)})
    if not records:
        raise ValueError("no real forecast history/restart files were retained")
    prepared = run / "artifacts/chain/prep/prepared-cache"
    if not (prepared / "header.json").is_file():
        raise ValueError("prepared-store header and array files are absent; content receipts alone do not bind forcing words")
    for path in sorted(prepared.rglob("*")):
        if path.is_symlink():
            raise ValueError("prepared artifact symlink would break evidence ownership")
        if path.is_file():
            relative = Path("prepared") / path.relative_to(prepared)
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
            records.append({"source": str(path), "view": str(relative),
                            "bytes": path.stat().st_size, "sha256": sha(path)})
    # Complete rendered products are part of the comparison too. Preserve the
    # relative consumer path, without interpreting weather fields in Python.
    for path in sorted((run / "artifacts").rglob("*.png")):
        if path.is_symlink():
            raise ValueError("rendered product symlink would break evidence ownership")
        relative = Path("products") / path.relative_to(run / "artifacts")
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        records.append({"source": str(path), "view": str(relative),
                        "bytes": path.stat().st_size, "sha256": sha(path)})
    selected = science_evidence(document)
    # The approved input contract also binds the complete physics/domain and
    # clock/restart metadata. These values must agree, without normalization.
    for key in ("physics", "domain", "run_seconds", "restart_contract"):
        selected[key] = document[key]
    contract = dict(selected["restart_contract"])
    logical = []
    for value in contract.get("checkpoints_written", []):
        checkpoint = Path(value).resolve()
        relative = checkpoint.relative_to(forecast.resolve()).as_posix()
        if not checkpoint.is_file():
            raise ValueError("restart receipt names a missing checkpoint")
        logical.append(relative)
    contract["checkpoints_written"] = logical
    selected["restart_contract"] = contract
    write(view / "science.json", selected)
    receipt = {"source_report": str(report), "source_report_sha256": sha(report),
               "method": "complete forecast artifact files plus engine-provided scientific evidence",
               "excluded": "orchestration logs, paths, timings, process ids and input staging outputs",
               "files": records, "evidence": selected}
    write(run / "artifact-inventory.json", receipt)
    return receipt


def analyze(run, output):
    timeline = Timeline()
    timeline.read(run / "ledger.jsonl")
    report = timeline.report(run_id=run.name, product="regional", cards=1)
    if report["integrate_union_seconds"] is None or report["proxy_ff"] is None:
        raise ValueError("complete measured integration/proxy boundaries are missing")
    start, end = report["origin"], report["endpoint"]
    spans = [(max(row["start"], start), min(row["end"], end)) for row in report["intervals"]
             if row["end"] > start and row["start"] < end]
    covered = union(spans)
    cursor = start
    gaps = []
    for a, b in covered:
        if a > cursor:
            gaps.append({"start": cursor, "end": a, "seconds": a - cursor})
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append({"start": cursor, "end": end, "seconds": end - cursor})
    report["unnamed_gaps"] = gaps
    report["max_unnamed_gap_seconds"] = max((gap["seconds"] for gap in gaps), default=0)
    report["gap_verdict"] = "PASS" if report["max_unnamed_gap_seconds"] <= 2 else "FAIL"
    write(output, report)
    print(json.dumps({"proxy_ff": report["proxy_ff"], "wall_seconds": report["proxy_wall_seconds"],
                      "integrate_seconds": report["integrate_union_seconds"],
                      "gap_verdict": report["gap_verdict"], "max_gap_seconds": report["max_unnamed_gap_seconds"]}))
    return 0 if report["gap_verdict"] == "PASS" else 1


def compare(args):
    baseline_view, candidate_view = args.output.parent / "baseline-view", args.output.parent / "candidate-view"
    base = artifact_view(args.baseline, baseline_view)
    candidate = artifact_view(args.candidate, candidate_view)
    gate = run_gate(baseline_view, candidate_view, scratch_dir=args.output.parent)
    write(args.output, {**gate, "baseline_inventory": base, "candidate_inventory": candidate})
    print(json.dumps({"verdict": gate["verdict"], "compared_files": len(gate["files"]),
                      "receipt": str(args.output)}))
    return 0 if gate["verdict"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--baseline", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return analyze(args.run, args.output) if args.command == "analyze" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
