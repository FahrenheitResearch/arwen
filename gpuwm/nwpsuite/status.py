"""One status document and one command hook on a health transition."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time

from gpuwm.cycle.ledger import write_receipt_atomically
from .ledger import SuiteLedger


def score_delta(score, baseline):
    """Numeric score leaves against the mean of the recorded rehearsal cases."""
    def numbers(value, prefix=""):
        if isinstance(value, dict):
            return {k: n for key, child in value.items() for k, n in numbers(child, prefix + "/" + key).items()}
        if isinstance(value, list):
            return {k: n for i, child in enumerate(value) for k, n in numbers(child, prefix + "/" + str(i)).items()}
        return {prefix: value} if isinstance(value, (int, float)) and not isinstance(value, bool) else {}
    if not score or not baseline:
        return None
    current = numbers(score[0] if isinstance(score, list) and len(score) == 1 else score)
    cases = [numbers(case["score"]) for case in baseline]
    return {key: value - sum(case[key] for case in cases if key in case) / sum(key in case for case in cases)
            for key, value in current.items() if any(key in case for case in cases)}


def status_document(plan, root, *, now=None):
    now = time.time() if now is None else now
    root = Path(root)
    state = SuiteLedger(root).state()
    baseline_path = root / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    health_path = root / "box-health.json"
    health = json.loads(health_path.read_text()) if health_path.exists() else {}
    definitions = {row["id"]: row for row in plan["rows"]}
    def ancestors(row_id):
        found, pending = set(), [row_id]
        while pending:
            current_id = pending.pop()
            if current_id in found:
                continue
            found.add(current_id)
            pending.extend(definitions[current_id].get("dependencies", []))
        return found
    verification = {r["id"]: ancestors(r["id"]) for r in plan["rows"] if r.get("kind") == "verify"}
    def read_score(definition, entry):
        score = entry.get("result", {}).get("score")
        if score is not None:
            return score if isinstance(score, list) else [score]
        if entry.get("phase") != "completed":
            return []
        from .run import expand
        from .bindings import row_at
        if entry.get("cycle"):
            definition = row_at(definition, entry["cycle"])
        context = dict(root=str(root.resolve()), row=definition["id"], cycle=entry["cycle"],
            out=str(Path(entry.get("work", "")) / "output"), work=entry.get("work", ""))
        from .bindings import resolve_context
        context = resolve_context(definition, dict(context, source=""))
        return [json.loads(Path(p).read_text()) for p in expand(definition.get("score_files", []), context) if Path(p).is_file()]
    rows = []
    behind_by_box = {}
    for row in plan["rows"]:
        entries = [x for (rid, _), x in state["jobs"].items() if rid == row["id"]]
        current = max(entries, key=lambda x: x["cycle"]) if entries else {}
        phase = current.get("phase", "pending")
        if phase == "waiting" and current.get("reason") == "card_admission":
            phase = "queued"
        if row["id"] in state["halted"] or "*" in state["halted"]:
            phase = "halted"
        if current.get("reason") == "source_behind":
            phase = "source_behind"
        deadline = current.get("deadline_unix")
        lag = max(0, (current.get("result", {}).get("finished_unix", now) - deadline)) if deadline else None
        score = read_score(row, current)
        score_rows = [rid for rid, parents in verification.items() if row["id"] in parents]
        if not score:
            score = [item for rid in score_rows for item in read_score(definitions[rid],
                state["jobs"].get((rid, current.get("cycle")), {}))]
        row_baseline = baseline.get(row["id"]) or [item for rid in score_rows for item in baseline.get(rid, [])]
        score = score or None
        from .bindings import row_at
        effective = row_at(row, current["cycle"]) if current.get("cycle") else row
        sources = effective.get("sources", [])
        source_index = int(current.get("source_index", 0))
        behind = (sources[:source_index + 1] if current.get("reason") == "source_behind" else
                  sources[source_index:source_index + 1] if phase == "waiting" else [])
        behind_by_box.setdefault(effective.get("box", "local"), set()).update(behind)
        rows.append(dict(id=row["id"], last_cycle=current.get("cycle"), state=phase,
            lag_seconds=lag, last_score=score,
            rehearsal_baseline=row_baseline or None, score_rows=score_rows, reason=current.get("reason"),
            score_delta=score_delta(score, row_baseline),
            remedy=current.get("result", {}).get("remedy"), sources_behind=behind))
    # Members belong to one suite row. CPU stages and the DA arm remain visible
    # separately, so a halted child does not report its parent as halted.
    groups = {}
    for item in rows:
        definition = definitions[item["id"]]
        key = definition.get("suite_row", item["id"]) if definition.get("kind") not in {"products", "verify", "da"} else item["id"]
        groups.setdefault(key, []).append(item)
    severity = {"halted": 9, "source_behind": 8, "skipped": 7, "waiting": 6, "queued": 5,
                "running": 4, "launch_intent": 4, "ready": 3, "planned": 2, "pending": 1, "completed": 0}
    rows = []
    for key, members in groups.items():
        if len(members) == 1 and key == members[0]["id"]:
            rows.append(members[0])
            continue
        worst = max(members, key=lambda r: severity.get(r["state"], 2))
        aggregate = dict(worst, id=key, members=members,
            sources_behind=sorted({source for member in members for source in member["sources_behind"]}),
            last_cycle=max((r["last_cycle"] for r in members if r["last_cycle"]), default=None),
            lag_seconds=max((r["lag_seconds"] for r in members if r["lag_seconds"] is not None), default=None))
        scored = next((r for r in members if r["last_score"] and r["last_cycle"] == aggregate["last_cycle"]), None)
        if scored:
            for name in ("last_score", "rehearsal_baseline", "score_rows", "score_delta"):
                aggregate[name] = scored[name]
        rows.append(aggregate)
    boxes = []
    ram = None
    try:
        ram = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
    except (OSError, StopIteration, ValueError):
        pass
    for box in plan.get("boxes", []):
        info = health.get(box["id"], {})
        local = box.get("host", "local") in {"local", "localhost", "127.0.0.1"}
        active = [x for x in state["jobs"].values() if x.get("box") == box["id"] and x.get("phase") in {"running", "launch_intent"}]
        occupancy = []
        for entry in active:
            assigned = None
            marker = Path(entry.get("work", "")) / "admitted.txt"
            try:
                assigned = json.loads(marker.read_text())["cards"]
            except (OSError, ValueError, KeyError):
                pass
            occupancy.append({"row": entry["row"], "cycle": entry["cycle"], "cards": entry.get("cards", 0),
                              "card_ids": assigned, "planned_card_ids": entry.get("card_ids"), "admitted": assigned is not None})
        boxes.append(dict(id=box["id"], cards=info.get("cards", box.get("cards")),
            card_occupancy=occupancy,
            host_ram_available_bytes=info.get("host_ram_available_bytes", ram if local else None),
            disk_free_bytes=info.get("disk_free_bytes", shutil.disk_usage(root).free if local else None),
            alive=info.get("alive"),
            sources_behind=sorted(behind_by_box.get(box["id"], set()))))
    phases = {r["state"] for r in rows}
    late = any((r["lag_seconds"] or 0) > 0 for r in rows)
    code = 2 if "halted" in phases else 1 if late or phases & {"source_behind", "skipped", "queued"} else 75 if "waiting" in phases else 0
    beat_path = root / "heartbeat.json"
    beat = json.loads(beat_path.read_text()) if beat_path.exists() else {}
    stale = now - beat.get("updated_unix", 0) > 120
    if stale and code == 0:
        code = 1
    return dict(schema="gpuwm.suite.status.v1", suite_hash=plan["suite_hash"], updated_unix=now,
                exit_code=code, state={0: "green", 1: "degraded", 2: "halted", 75: "waiting"}[code],
                heartbeat_stale=stale, rows=rows, boxes=boxes)


def publish_status(plan, root, *, now=None):
    root = Path(root)
    document = status_document(plan, root, now=now)
    path = root / "status.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    write_receipt_atomically(path, document)
    signature = lambda doc: (doc.get("state"), [(r["id"], r["state"]) for r in doc.get("rows", [])])
    if signature(old) != signature(document) and plan.get("on_state_change"):
        command = plan["on_state_change"]
        if isinstance(command, str):
            import shlex
            command = shlex.split(command)
        try:
            with (root / "alert.log").open("ab") as log:
                done = subprocess.run(command, input=json.dumps(document).encode(), stdout=log, stderr=log, timeout=30)
            SuiteLedger(root).append("alert", exit_code=done.returncode, state=document["state"])
        except (OSError, subprocess.TimeoutExpired) as error:
            SuiteLedger(root).append("alert_failed", message=str(error))
    return document
