"""Write a coverage-explicit sweep report from real replay verdicts."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def measurement(row):
    comparison = row.get("comparisons", {}).get(row.get("referee"), {})
    detail = next((d for d in comparison.get("details", [])
                   if d["field"] == "U" and d["step"] == 1), None)
    return detail.get("max_abs_diff") if detail else None


def report(previous, current):
    prev = {(r["combo"], r["case"]): r for r in previous}
    now = {(r["combo"], r["case"]): r for r in current}
    missing = set(prev) - set(now)
    lines = ["INCOMPLETE. Round 4 has not covered the full round-3 recording." if missing else
             "Round-4 coverage is complete. Numerical verdicts are below.", "",
             f"Replayed {len(set(prev) & set(now))} of {len(prev)} round-3 runs. Missing {len(missing)}.", "",
             "Missing runs have no round-4 verdict. Historical figures are not reused as new measurements.", "",
             "| Outcome | Round 3 | Round 4 observed |", "|---|---:|---:|"]
    old_count = Counter(r["outcome"] for r in previous)
    new_count = Counter(r["outcome"] for r in current)
    for outcome in sorted(set(old_count) | set(new_count)):
        lines.append(f"| {outcome} | {old_count[outcome]} | {new_count[outcome]} |")
    lines += ["", "| Round 3 to round 4 | Runs |", "|---|---:|"]
    moves = Counter((prev[key]["outcome"], row["outcome"]) for key, row in now.items() if key in prev)
    for (before, after), count in sorted(moves.items()):
        lines.append(f"| {before} to {after} | {count} |")
    lines += [f"| Missing round-4 verdict | {len(missing)} |", "",
              "| Family | Value | Recorded runs | Replayed | Step-1 U measurements | Median max abs U (m/s) |",
              "|---|---|---:|---:|---:|---:|"]
    for family in ("mp", "pbl", "sfclay", "lsm", "cu", "rad"):
        for value in sorted({r[family] for r in previous}):
            records = [r for r in previous if r[family] == value]
            selected = [r for r in current if str(r.get("factors", {}).get(family)) == value]
            sizes = [v for r in selected if (v := measurement(r)) is not None]
            median = f"{statistics.median(sizes):.9g}" if sizes else "unmeasured"
            lines.append(f"| {family} | {value} | {len(records)} | {len(selected)} | {len(sizes)} | {median} |")
    frame0, absent = Counter(), Counter()
    for row in current:
        comp = row.get("comparisons", {}).get(row.get("referee"), {})
        if comp.get("first_step") == 0:
            frame0.update(comp.get("first_fields", []))
        absent[len(comp.get("absent_unexcused", []))] += 1
    lines += ["", "Frame-0 first differing fields: " + json.dumps(dict(frame0), sort_keys=True),
              "", "Unexcused absent-field counts (count to runs): " + json.dumps(dict(absent), sort_keys=True),
              "", "U measurements require raw arrays. Digest misses alone do not establish a magnitude."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round3", type=Path, required=True)
    ap.add_argument("--results", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    previous = list(csv.DictReader(args.round3.open(encoding="utf-8-sig", newline="")))
    current = [json.loads(line) for path in args.results for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["combo"], row["case"]) for row in current]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate replay rows would inflate coverage")
    args.out.write_text(report(previous, current), encoding="utf-8")


if __name__ == "__main__":
    main()
