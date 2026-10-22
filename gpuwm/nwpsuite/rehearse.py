"""Replay named past cycles without posting waits; verification stays a door."""
from __future__ import annotations

import json
from pathlib import Path
import time

from gpuwm.cycle.ledger import write_receipt_atomically
from .run import SuiteRunner, stamp, utc


def rehearse(plan, root, *, start, days, cycles_limit=None):
    first = utc(start).timestamp()
    end = first + days * 86400
    if first >= time.time() or end > time.time():
        raise ValueError("rehearsal_future: verifying observations must already exist")
    if days <= 0:
        raise ValueError("rehearsal_days: must be positive")
    if not any(r.get("kind") == "verify" for r in plan["rows"]):
        raise ValueError("rehearsal_unscored: suite must include the existing observation scorecard door")
    from .bindings import cadence
    step = int(min(cadence(r) for r in plan["rows"]))
    cycles = [stamp(t) for t in range(int(first), int(end), step)]
    if cycles_limit:
        cycles = cycles[:cycles_limit]
    write_receipt_atomically(Path(root) / "replay-window.json", {"cycles": cycles, "rehearsal": True})
    runner = SuiteRunner(plan, root, rehearsal=True)
    runner.run(cycles=cycles)
    baseline = {}
    jobs = runner.ledger.state()["jobs"]
    for definition in plan["rows"]:
        if definition.get("kind") != "verify":
            continue
        scores = []
        for cycle in cycles:
            from .bindings import is_due, row_at
            if not is_due(definition, cycle):
                continue
            row = row_at(definition, cycle)
            entry = jobs.get((row["id"], cycle), {})
            saved = entry.get("result", {}).get("score")
            if entry.get("phase") == "completed" and saved is not None:
                for score in saved if isinstance(saved, list) else [saved]:
                    scores.append(dict(cycle=cycle, score=score))
                continue
            context = runner.context(row, cycle, entry.get("attempt", 0))
            from .run import expand
            for score_file in expand(row.get("score_files", []), context):
                path = Path(score_file)
                if entry.get("phase") == "completed" and path.exists():
                    scores.append(dict(cycle=cycle, score=json.loads(path.read_text())))
        baseline[definition["id"]] = scores
    # A run without score coverage is not an acceptance baseline.
    from .bindings import is_due
    expected_scores = {r["id"]: {c for c in cycles if is_due(r, c)} for r in plan["rows"] if r.get("kind") == "verify"}
    definitions = {r["id"]: r for r in plan["rows"]}
    selected = [jobs.get((row["id"], cycle), dict(row=row["id"], cycle=cycle, phase="pending"))
                for row in plan["rows"] for cycle in cycles if is_due(row, cycle)]
    fallback = [entry for entry in selected if definitions[entry["row"]].get("kind") == "da" and
                entry.get("phase") == "skipped" and entry.get("reason") == "da_failed"]
    complete = all(x.get("phase") == "completed" or x in fallback for x in selected)
    success = bool(selected) and complete and bool(baseline) and all(
        {item["cycle"] for item in scores} == expected_scores[rid] and bool(scores) for rid, scores in baseline.items())
    result = dict(schema="gpuwm.suite.rehearsal.v1", suite_hash=runner.plan["suite_hash"], cycles=cycles,
                  accepted=success, degraded=bool(fallback),
                  fallbacks=[dict(row=x["row"], cycle=x["cycle"], reason=x["reason"], initialization="public_analysis") for x in fallback],
                  scores=baseline)
    write_receipt_atomically(Path(root) / "rehearsal.json", result)
    if success:
        write_receipt_atomically(Path(root) / "baseline.json", baseline)
        runner.ledger.append("baseline", suite_hash=runner.plan["suite_hash"], scores=baseline)
    return result
