"""Join real stack receipts and refuse missing lane or arithmetic coverage."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def lane_errors(branch, summary, revision, source_diff_sha256):
    if summary is None:
        return [f"{branch}: no GPU receipt"]
    errors = []
    if summary["revision"] != revision or summary.get("source_diff_sha256") != source_diff_sha256:
        errors.append(f"{branch}: receipt describes different source bytes")
    if not summary["passed"]:
        errors.append(f"{branch}: GPU gate failed")
    modes = {row.get("mode"): row for row in summary["results"] if row.get("lane") == branch}
    for mode in ("strict", "default"):
        row = modes.get(mode)
        if row is None or row["exit_code"]:
            errors.append(f"{branch}: {mode} gate absent or failed")
            continue
        # base5 supplies scoring/health gates, not a scheme column oracle.
        if branch != "lane/verify-base5" and row.get("compared_words", 0) == 0:
            errors.append(f"{branch}: {mode} observed no numerical comparison words")
        if "verify-mynn-sfclay" in branch and row.get("differing_words", 0):
            errors.append(f"{branch}: {mode} MYNN words are not identical")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", "gpuwm", "tests", "tools"], cwd=ROOT)
    source_diff_sha256 = hashlib.sha256(diff).hexdigest()
    plan = json.loads((ROOT / "tools/verify_lanes_stack.json").read_text())
    errors, rows = [], []
    def read(path):
        return json.loads(path.read_text()) if path.exists() else None
    for name in ("qualified-ancestry", "qualified-cpu"):
        summary = read(args.receipts / name / "summary.json")
        if (summary is None or not summary["passed"] or summary["revision"] != revision
                or summary.get("source_diff_sha256") != source_diff_sha256):
            errors.append(f"{name}: absent, failed or for different source bytes")
    for lane in plan["lanes"]:
        if not lane["take"]:
            rows.append({"lane": lane["branch"], "tip": lane.get("tip", ""),
                         "mode": "skipped", "reason": lane["reason"]})
            continue
        branch = lane["branch"]
        summary = read(args.receipts / "gpu-gates" / branch.split("/")[-1] / "summary.json")
        errors.extend(lane_errors(branch, summary, revision, source_diff_sha256))
        modes = {} if summary is None else {row.get("mode"): row for row in summary["results"] if row.get("lane") == branch}
        for mode in ("strict", "default"):
            row = modes.get(mode, {})
            rows.append({"lane": branch, "tip": lane["tip"], "mode": mode,
                         "compared_words": row.get("compared_words"),
                         "differing_words": row.get("differing_words"),
                         "exit_code": row.get("exit_code"), "reason": lane["reason"]})
    fields = ("lane", "tip", "mode", "compared_words", "differing_words", "exit_code", "reason")
    with (args.out / "lanes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"revision": revision, "source_diff_sha256": source_diff_sha256,
              "passed": not errors, "errors": errors, "lanes": rows}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"STACK {'PASS' if not errors else 'FAIL'}: {len(errors)} missing or failed coverage checks")
    for error in errors:
        print(error)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
