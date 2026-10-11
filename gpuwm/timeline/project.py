"""Critical-path projections. Savings are removed work, not additive claims.

Reference nodes describe measured stage durations and real predecessor edges.
Independent deltas on a node remove the union of their reference-time spans.
Dependent deltas are marginal after their declared predecessors. Parallel
branches are scheduled again, so shortening a non-critical branch saves no wall.
"""
from __future__ import annotations

import json
from pathlib import Path

from .report import Timeline, seconds, union, union_seconds


def nonnegative(value, label):
    number = seconds(value)
    if number is None or number < 0:
        raise ValueError(f"{label} must be finite and nonnegative to preserve schedule accounting")
    return number


def topological(records, *, key="id"):
    indexed = {}
    for row in records:
        if not isinstance(row.get(key), str) or not row[key] or row[key] in indexed:
            raise ValueError(f"duplicate or missing {key} would apply work more than once")
        indexed[row[key]] = row
    ordered = []
    pending = dict(indexed)
    while pending:
        ready = [name for name, row in pending.items()
                 if all(dep in {r[key] for r in ordered} for dep in row.get("depends_on", []))]
        if not ready:
            raise ValueError("missing or cyclic depends_on would invent an invalid critical path")
        for name in sorted(ready):
            ordered.append(pending.pop(name))
    return ordered


def schedule(reference, durations):
    nodes = topological(reference["nodes"])
    placed = {}
    for row in nodes:
        previous = max((placed[d]["end"] for d in row.get("depends_on", [])), default=0.0)
        lag = nonnegative(row.get("lag_seconds", 0), f"{row['id']} lag_seconds")
        start = previous + lag
        placed[row["id"]] = {**row, "start": start, "end": start + durations[row["id"]]}
    endpoint = max((row["end"] for row in placed.values()), default=0)
    if endpoint <= 0:
        raise ValueError("empty or zero-length reference cannot establish projected FF")
    timeline = Timeline()
    mode = reference.get("mode")
    if mode not in {"rental", "proxy"}:
        raise ValueError("projection reference needs rental or proxy mode; denominators cannot be mixed")
    timeline.boundary("rental.requested" if mode == "rental" else "proxy.start", 0, "reference envelope")
    timeline.boundary("rental.terminated" if mode == "rental" else "proxy.end", endpoint, "scheduled last node")
    for row in placed.values():
        timeline.interval(row["stage"], row["start"], row["end"], row["evidence"],
                          card=row.get("card"), basis="projected_schedule")
    result = timeline.report(product=reference["product"], cards=reference.get("cards"))
    result["nodes"] = list(placed.values())
    # All ties on the critical path are retained.
    critical = set()
    def visit(name):
        if name in critical:
            return
        critical.add(name)
        row = placed[name]
        deps = row.get("depends_on", [])
        latest = max((placed[d]["end"] for d in deps), default=0)
        for dep in deps:
            if abs(placed[dep]["end"] - latest) < 1e-8:
                visit(dep)
    for name, row in placed.items():
        if abs(row["end"] - endpoint) < 1e-8:
            visit(name)
    result["critical_path_nodes"] = sorted(critical)
    return result


def _tail_uncovered(duration, removed, count):
    """Select a marginal tail from work left after this lane's dependencies."""
    remaining = []
    cursor = 0.0
    for start, end in union(removed):
        if start > cursor:
            remaining.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        remaining.append((cursor, duration))
    result = []
    for start, end in reversed(remaining):
        take = min(count, end - start)
        if take:
            result.append((end - take, end))
            count -= take
    if count > 1e-8:
        raise ValueError("marginal delta exceeds work left after depends_on")
    return result


def project(reference, deltas):
    if reference.get("schema") != "gpuwm.timeline.reference.v1":
        raise ValueError("reference needs gpuwm.timeline.reference.v1 nodes and measured dependency edges; waterfall overlap alone cannot prove the critical path")
    nodes = topological(reference["nodes"])
    durations = {}
    for row in nodes:
        if not row.get("stage") or not row.get("evidence"):
            raise ValueError("reference nodes need stage and evidence for measured durations")
        durations[row["id"]] = nonnegative(row["duration_seconds"], row["id"])
    baseline = schedule(reference, durations)
    field = "ff" if reference["mode"] == "rental" else "proxy_ff"
    measured = reference.get("measured_ff", baseline[field])
    if measured is not None:
        measured = nonnegative(measured, "measured_ff")
        if measured > 1 or baseline[field] is None or abs(measured - baseline[field]) > .01 * max(measured, .01):
            raise ValueError("reference schedule does not reproduce measured FF within 1 percent")
    expected_wall = reference.get("measured_wall_seconds")
    if expected_wall is not None:
        expected_wall = nonnegative(expected_wall, "measured_wall_seconds")
        if abs(baseline["wall_seconds"] - expected_wall) > .01 * max(expected_wall, .01):
            raise ValueError("reference schedule does not reproduce measured wall within 1 percent")
    applicable = [d for d in deltas if d.get("product") in (None, reference["product"])]
    ordered = topological(applicable, key="lane")
    removals = {}
    receipts = []
    def ancestors(row):
        result = set(row.get("depends_on", []))
        for dep in list(result):
            result |= ancestors(next(r for r in ordered if r["lane"] == dep))
        return result
    for delta in ordered:
        if delta.get("measured") is not True or not delta.get("evidence"):
            raise ValueError(f"{delta['lane']}: unmeasured delta cannot be used as delivered speed evidence")
        matches = [r for r in nodes if (r["id"] == delta.get("node") if delta.get("node") else r["stage"] == delta.get("stage"))]
        if len(matches) != 1:
            raise ValueError(f"{delta['lane']}: delta must identify one reference node; stage is absent or ambiguous")
        node = matches[0]["id"]
        duration = durations[node]
        dependencies = ancestors(delta)
        dependency_spans = [span for name, (target, spans) in removals.items()
                            if name in dependencies and target == node for span in spans]
        if "removed_spans" in delta:
            spans = []
            for start, end in delta["removed_spans"]:
                start = nonnegative(start, "removed span start")
                end = nonnegative(end, "removed span end")
                if start > end or end > duration:
                    raise ValueError("removed spans must stay inside their reference node")
                spans.append((start, end))
            spans = union(spans)
            # A marginal measurement cannot re-claim work removed by a
            # predecessor. Explicit spans use original reference coordinates.
            overlap = union_seconds(spans) + union_seconds(dependency_spans) - union_seconds(spans + dependency_spans)
            if overlap > 1e-8:
                raise ValueError("marginal removed_spans overlap depends_on work; this would double count predecessor savings")
        else:
            raw = delta.get("seconds_saved", delta.get("est_seconds_saved"))
            if raw is None:
                raise ValueError("delta needs measured seconds_saved or removed_spans")
            if "seconds_saved" in delta and "est_seconds_saved" in delta and delta["seconds_saved"] != delta["est_seconds_saved"]:
                raise ValueError("conflicting delta seconds would make projection ambiguous")
            count = nonnegative(raw, "seconds_saved")
            spans = _tail_uncovered(duration, dependency_spans, count)
        removals[delta["lane"]] = (node, spans)
        receipts.append({"lane": delta["lane"], "node": node, "depends_on": delta.get("depends_on", []),
                         "removed_reference_spans": spans, "evidence": delta["evidence"]})
    projected_durations = {}
    for node, duration in durations.items():
        removed = [span for target, spans in removals.values() if target == node for span in spans]
        projected_durations[node] = duration - union_seconds(removed)
    projected = schedule(reference, projected_durations)
    return {"schema": "gpuwm.timeline.projection.v1", "product": reference["product"],
            "mode": reference["mode"], "measured_ff": measured if field == "ff" else None,
            "measured_proxy_ff": measured if field == "proxy_ff" else None,
            "projected_ff": projected[field] if field == "ff" else None,
            "projected_proxy_ff": projected[field] if field == "proxy_ff" else None,
            "reference_wall_seconds": baseline["wall_seconds"], "projected_wall_seconds": projected["wall_seconds"],
            "seconds_saved": baseline["wall_seconds"] - projected["wall_seconds"],
            "stage_seconds_removed": {node: durations[node] - projected_durations[node] for node in durations},
            "lanes": receipts, "baseline": baseline, "projected": projected,
            "interpretation": "Independent removals are unioned on reference work. depends_on measurements are marginal. The DAG is rescheduled; stage savings and per-lane estimates are never summed as rental savings."}


def run(args):
    deltas = []
    for path in args.deltas:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        deltas.extend(value if isinstance(value, list) else value.get("deltas", [value]))
    results = []
    for path in args.reference:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        refs = value if isinstance(value, list) else value.get("products", [value])
        results.extend(project(ref, deltas) for ref in refs)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.png:
        from .report import waterfall
        if len(results) != 1:
            raise ValueError("one --png requires one product reference to avoid overwriting figures")
        waterfall(results[0]["projected"], args.png)
    out.write_text(json.dumps({"products": results}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0
