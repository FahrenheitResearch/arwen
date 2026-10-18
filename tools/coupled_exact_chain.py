"""Join replay verdicts to every recorded build without treating missing rows as passes."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def summarize(recorded, verdicts):
    indexed = {}
    for verdict in verdicts:
        key = (verdict["case"], verdict["combo"])
        if key in indexed:
            raise ValueError(f"duplicate verdict for {key}")
        indexed[key] = verdict
    rows = []
    for run in recorded:
        v = indexed.get((run["case"], run["combo"]))
        c = (v or {}).get("comparisons", {}).get(run["build"], {})
        state = c.get("first_by_class", {}).get("state", {})
        length = int(run["last_good_step"] or 0)
        steps = c.get("steps_compared", [])
        complete = bool(steps) and max(steps) == length and not c.get("missing_steps")
        exact = bool(c.get("matched")) and complete and run["status"] == "OK"
        rows.append(dict(build=run["build"], case=run["case"], combo=run["combo"],
                         recorded_status=run["status"], recorded_last_step=length,
                         outcome="MISSING" if v is None else v["outcome"],
                         exact_all_fields=exact,
                         first_step=c.get("first_step"), first_field=c.get("lead_field"),
                         first_state_step=state.get("step"),
                         first_state_field=next(iter(state.get("fields", [])), None),
                         absent=len(c.get("absent_unexcused", [])),
                         reshaped=len(c.get("reshaped", {})),
                         compared_frames=len(steps), factors=(v or {}).get("factors", {})))
    seams = Counter((r["first_state_step"], r["first_state_field"],
                     json.dumps(r["factors"], sort_keys=True)) for r in rows
                    if r["first_state_step"] is not None)
    totals = dict(recorded_rows=len(rows), case_combos=len({(r["case"], r["combo"]) for r in rows}),
                  measured_case_combos=len(indexed), missing_rows=sum(r["outcome"] == "MISSING" for r in rows),
                  all_field_exact_rows=sum(r["exact_all_fields"] for r in rows),
                  outcomes=dict(Counter(v["outcome"] for v in indexed.values())),
                  exact_lengths=dict(Counter(r["recorded_last_step"] for r in rows if r["exact_all_fields"])))
    return rows, totals, seams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    with (a.recordings / "runs.csv").open(newline="") as f:
        recorded = list(csv.DictReader(f))
    verdicts = [json.loads(line) for path in a.results for line in path.read_text().splitlines()
                if line.strip() and "combo" in json.loads(line)]
    rows, totals, seams = summarize(recorded, verdicts)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    with (a.out / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for row in rows:
            w.writerow(dict(row, factors=json.dumps(row["factors"], sort_keys=True)))
    with (a.out / "seams.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["first_state_step", "first_state_field", "factors", "recorded_build_rows"])
        for (step, field, factors), n in seams.most_common():
            w.writerow([step, field, factors, n])
    print(json.dumps(totals, sort_keys=True))
    return int(bool(totals["missing_rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
