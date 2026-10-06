#!/usr/bin/env python3
"""Score the regional rain gate's [L-1,L] hours for L=1,3,6 on saved fields."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gpuwm.verify.rain_gate import score_series, common_grid
from gpuwm.verify.rain_gate_readers import load_series
from gpuwm.verify.rain_gate_bridge import load


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True, help="regional-rain/input.v1 forecast manifest")
    parser.add_argument("--truth", type=Path, required=True, help="MRMS PrecipRate plus QC composite manifest")
    parser.add_argument("--baseline", type=Path, help="matched clean-start no-DA manifest, for paired FSS gain")
    parser.add_argument("--analysis-end", required=True, help="final analysis UTC, e.g. 2026-06-14T19:10:30Z")
    parser.add_argument("--event", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--product", required=True, help="member identity, ensemble_mean or analysed_deterministic")
    parser.add_argument("--observation-revision", default="MRMS archive-rich research")
    parser.add_argument("--out", type=Path, required=True, help="scores.jsonl; companion .receipt.json records config and hashes")
    args = parser.parse_args(argv)
    try:
        analysis = datetime.fromisoformat(args.analysis_end.replace("Z", "+00:00"))
        if analysis.tzinfo is None:
            raise ValueError("--analysis-end must include UTC time zone")
        if analysis.utcoffset().total_seconds() != 0:
            raise ValueError("--analysis-end must be UTC")
        load()
        forecast, fp = load_series(args.forecast, analysis_end=args.analysis_end)
        truth, tp = load_series(args.truth, analysis_end=args.analysis_end)
        baseline_provenance = None
        baseline = None
        if args.baseline:
            baseline, baseline_provenance = load_series(args.baseline, analysis_end=args.analysis_end)
        target = common_grid(forecast.grid, truth.grid, *([baseline.grid] if baseline is not None else []))
        rows, diagnostics = score_series(forecast, truth, target=target)
        if baseline is not None:
            baseline_rows, baseline_diagnostics = score_series(baseline, truth, target=target)
            # FSS comparisons need identical conservative grids and support.
            lookup = {(r["lead_hours"], r.get("threshold_mm_h"), r.get("scale_km"), r["metric"]): r for r in baseline_rows}
            for row in rows:
                b = lookup[(row["lead_hours"], row.get("threshold_mm_h"), row.get("scale_km"), row["metric"])]
                row["paired_baseline"] = {"value": b["value"], "support_hash": b["support_hash"], "manifest_sha256": _sha(args.baseline)}
                if row["metric"] == "fss":
                    same = row["support_hash"] == b["support_hash"] and row["projection_id"] == b["projection_id"]
                    row["paired_gain"] = row["value"]-b["value"] if same and row["value"] is not None and b["value"] is not None else None
                    if not same:
                        row["paired_reason"] = "baseline and DA have different complete neighborhood support"
            diagnostics["baseline"] = baseline_diagnostics
        try:
            head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            head = None
        for row in rows:
            # The spec's status refers to the campaign gate. A measured single
            # event is a screening row until event-blocked bounds exist.
            row["measurement_status"] = row["status"]
            row["status"] = "pending"
            row.update({"event": args.event, "seed": args.seed, "product": args.product,
                        "analysis_end": args.analysis_end, "observation_revision": args.observation_revision,
                        "scorer_commit": head,
                        "qualification_reason": "single-event screen; event-blocked simultaneous 95 percent bounds and remaining gate diagnostics are not measured"})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
        receipt = {"schema": "regional-rain/receipt.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "argv": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                   "exit_code": 0, "scorer_commit": head, "config": vars(args).copy(),
                   "forecast_sources": fp, "truth_sources": tp, "baseline_sources": baseline_provenance,
                   "diagnostics": diagnostics, "scores_sha256": _sha(args.out),
                   "limitations": ["legacy surviving archive reproduction has not been performed",
                                    "independent gauges and sensitivity QPE products require separate manifests",
                                    "no event-blocked campaign confidence bounds or ancillary DA gate losses"]}
        receipt["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in receipt["config"].items()}
        receipt_path = args.out.with_suffix(".receipt.json")
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({"scores": str(args.out), "receipt": str(receipt_path), "rows": len(rows),
                          "complete_measurements": sum(r["measurement_status"] == "complete" for r in rows),
                          "gate_status": "pending"}, sort_keys=True))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        print(f"regional rain scoring refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
