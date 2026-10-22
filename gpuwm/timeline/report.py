"""Forecast fraction accounting over C1 ledgers and explicit legacy receipts.

Only located first-step-start to last-step-end spans establish FF. Receipt
work totals and forecast pipeline timers are retained separately. This module
does not import CUDA, read model arrays, or change any emitter.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def timestamp(value):
    if (number := seconds(value)) is not None:
        return number
    if not isinstance(value, str):
        return None
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # An unzoned timestamp is ambiguous, not automatically local or UTC.
        return parsed.timestamp() if parsed.tzinfo else None
    except ValueError:
        return None


def union(spans):
    merged = []
    for start, end in sorted(spans, key=lambda span: (span[0], span[1])):
        if not all(math.isfinite(v) for v in (start, end)) or end < start:
            raise ValueError("non-finite or reversed interval would corrupt wall accounting")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def union_seconds(spans):
    return sum(end - start for start, end in union(spans))


class Timeline:
    """Read only explicitly supplied files. Preserve source paths and lines."""

    def __init__(self):
        self.events = []
        self.intervals = []
        self.unplaced = []
        self.boundaries = {}
        self.warnings = []
        self.inputs = []
        self._steps = defaultdict(list)

    def boundary(self, stage, at, evidence):
        at = timestamp(at)
        if at is None:
            raise ValueError(f"{evidence}: boundary needs a finite Unix time or zoned UTC time")
        self.boundaries.setdefault(stage, []).append({"at": at, "evidence": evidence})

    def interval(self, stage, start, end, evidence, **context):
        start, end = timestamp(start), timestamp(end)
        if start is None or end is None or end < start:
            raise ValueError(f"{evidence}: invalid interval would corrupt forecast fraction")
        self.intervals.append({"stage": stage, "start": start, "end": end,
                               "evidence": evidence, **context})

    def read(self, path):
        path = Path(path)
        name = path.name.lower()
        if (name.startswith(".env") or name.endswith(".env")
                or name in {"auth.json", ".cdsapirc", "andrew_accesskeys.csv"}
                or ".aws" in {part.lower() for part in path.parts}):
            raise ValueError("timeline inputs must be timing receipts, not credential files")
        self.inputs.append(str(path))
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig") as stream:
            # JSON receipts may be pretty printed; JSONL is streamed to avoid
            # keeping a full long-run progress stream in memory.
            if path.suffix == ".json" or name.endswith(".json.gz"):
                value = json.load(stream)
                self.receipt(value, str(path) + "#$")
            else:
                for line, text in enumerate(stream, 1):
                    if not text.strip():
                        continue
                    try:
                        value = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line}: malformed timing JSON: {exc.msg}") from exc
                    self.receipt(value, f"{path}:{line}")
        return self

    def receipt(self, item, evidence):
        if isinstance(item, list):
            for index, row in enumerate(item):
                self.receipt(row, f"{evidence}[{index}]")
            return
        if not isinstance(item, dict):
            return
        if "v" in item and "stage" in item and "phase" in item:
            if item["v"] != 1:
                raise ValueError(f"{evidence}: unsupported ledger version would misinterpret timing semantics")
            required = {"v", "t_utc", "t_mono", "host", "rental_id", "run_id",
                        "product", "stage", "phase", "card", "pid", "attrs"}
            if set(item) != required or not isinstance(item["attrs"], dict):
                raise ValueError(f"{evidence}: invalid C1 keys or attrs would lose timing context")
            if item["phase"] not in {"begin", "end", "mark"}:
                raise ValueError(f"{evidence}: invalid C1 phase prevents interval pairing")
            if timestamp(item["t_utc"]) is None or seconds(item["t_mono"]) is None:
                raise ValueError(f"{evidence}: C1 requires UTC and finite monotonic time")
            self.events.append({**item, "evidence": evidence})
            return
        # An explicit small sidecar can locate boundaries or intervals that
        # the old artifacts recorded outside the event stream.
        if item.get("schema") == "gpuwm.timeline.sources.v1":
            for stage, entry in item.get("boundaries", {}).items():
                self.boundary(stage, entry["at"], entry.get("evidence", evidence))
            for row in item.get("intervals", []):
                row = dict(row)
                ref = row.pop("evidence", evidence)
                self.interval(row.pop("stage"), row.pop("start"), row.pop("end"), ref, **row)
            self.unplaced.extend(item.get("unplaced", []))
            self.warnings.extend(item.get("warnings", []))
            return
        # Site cost snapshots: launch is NOT a rental request. Repeated
        # snapshots remain available for selecting the completed attempt.
        if isinstance(item.get("boxes"), list):
            for index, box in enumerate(item["boxes"]):
                rental = box.get("id", box.get("instanceId", f"{item.get('id')}:box:{index}"))
                for field, stage in (("requested", "rental.requested"),
                                     ("launched", "legacy.rental.launched"),
                                     ("ended", "rental.terminated")):
                    if box.get(field):
                        self.events.append(self._legacy_mark(stage, box[field],
                            f"{evidence}.boxes[{index}].{field}", rental_id=rental,
                            run_id=item.get("id")))
            for index, attempt in enumerate(item.get("attempts", [])):
                if seconds(attempt.get("forecastWallSeconds")) is not None:
                    self.unplaced.append({"stage": "legacy.forecast.timer",
                        "seconds": attempt["forecastWallSeconds"], "run_id": item.get("id"),
                        "attempt": index, "evidence": f"{evidence}.attempts[{index}].forecastWallSeconds"})
            return
        if item.get("schema") == "ordinary-nested-scheduler-v1":
            for name, row in item["members"].items():
                if row.get("started") and row.get("finished"):
                    self.interval("legacy.forecast.pipeline", row["started"], row["finished"],
                        f"{evidence}.members.{name}", run_id=name,
                        card=row.get("physical_gpu"), basis="preparation_plus_forecast_not_integration")
            return
        if item.get("event"):
            self.legacy_event(item, evidence)
            return
        # Unknown timing receipts keep duration-only measurements, never
        # assign them an invented wall placement or sum parallel members.
        for key, value in item.items():
            if isinstance(value, (dict, list)):
                self.receipt(value, f"{evidence}.{key}")
            elif key in {"forecast_plus_io_s", "forecast_plus_io_wall_seconds",
                         "integration_seconds", "loop_seconds_including_io", "total_wall_seconds"}:
                if seconds(value) is not None:
                    self.unplaced.append({"stage": "legacy." + key, "seconds": value,
                                          "evidence": f"{evidence}.{key}"})

    @staticmethod
    def _legacy_mark(stage, at, evidence, **context):
        return {"stage": stage, "phase": "mark", "t_utc": at,
                "t_mono": None, "host": "legacy", "attrs": {},
                "card": None, "pid": None, "run_id": None, "rental_id": None,
                "product": None, "evidence": evidence, **context}

    def legacy_event(self, row, evidence):
        at = timestamp(row.get("utc", row.get("timestamp", row.get("at"))))
        for field, scale in (("emitted_unix_ms", 1000), ("utc_unix_s", 1)):
            if seconds(row.get(field)) is not None:
                at = row[field] / scale
                break
        event = row["event"]
        context = {key: row.get(key) for key in ("run_id", "rental_id", "card", "pid", "product")}
        if event == "model_progress":
            # Polled progress is not a measured step; outer_step/domain sums
            # do not give its start. Retain the uncertainty.
            self.warnings.append(f"{evidence}: model_progress is a sample, not an integration boundary")
        if event == "step" and at is not None:
            duration = seconds(row.get("step_wall_seconds"))
            if duration is not None and duration >= 0:
                key = tuple(context[k] for k in ("rental_id", "run_id", "product", "card", "pid"))
                self._steps[key].append((at - duration, at, evidence))
        stage_map = {"boot": "boot.legacy", "fetch": "fetch.legacy", "prepare": "prep.legacy",
                     "forecast": "legacy.forecast.pipeline", "render": "post.render"}
        if event in {"stage_started", "stage_finished", "stage_start", "stage_end"}:
            stage = stage_map.get(row.get("stage"), row.get("stage"))
            if stage:
                start = seconds(row.get("started_unix_ms"))
                end = seconds(row.get("finished_unix_ms"))
                if start is not None and end is not None:
                    self.interval(stage, start / 1000, end / 1000, evidence, basis="legacy_stage", **context)
                elif at is not None:
                    phase = "begin" if event in {"stage_started", "stage_start"} else "end"
                    self.events.append({**self._legacy_mark(stage, at, evidence, **context), "phase": phase})
        marks = {"rental_requested": "rental.requested", "rental_terminated": "rental.terminated",
                 "run_claimed": "worker.claim", "proxy.start": "proxy.start", "proxy.end": "proxy.end"}
        if event in marks and at is not None:
            self.events.append(self._legacy_mark(marks[event], at, evidence, **context))
        if event == "output_written" and at is not None:
            duration = seconds(row.get("durable_after_seconds"))
            if duration is not None and duration >= 0:
                self.interval("forecast.output.write", at - duration, at, evidence,
                              basis="completion_minus_duration", **context)

    def report(self, *, rental_id=None, run_id=None, product=None, cards=None):
        events = self.events
        rentals = {e.get("rental_id") for e in events if e.get("rental_id") is not None}
        if rental_id is None and len(rentals) > 1:
            raise ValueError("multiple rentals would mix denominators; select --rental-id")
        rental_id = rental_id or next(iter(rentals), None)
        def selected(row):
            return ((rental_id is None or row.get("rental_id") in (None, rental_id))
                    and (run_id is None or row.get("run_id") in (None, run_id)
                         or row.get("stage", "").startswith("rental."))
                    and (product is None or row.get("product") in (None, product)
                         or row.get("stage", "").startswith("rental.")))
        events = [e for e in events if selected(e)]
        products = sorted({e["product"] for e in events if e.get("product") is not None})
        if product is None and len(products) == 1:
            product = products[0]
        warnings = list(self.warnings)
        boundaries = {stage: list(rows) for stage, rows in self.boundaries.items()}
        intervals = [dict(row) for row in self.intervals if selected(row)]
        pending = defaultdict(list)
        # Each host's monotonic clock is compared only to itself. Start UTC
        # locates the span on the cross-host axis; monotonic time supplies its
        # duration even if the host's wall clock adjusted during the stage.
        def order(e):
            return (str(e.get("host")), e.get("t_mono") if e.get("t_mono") is not None
                    else timestamp(e["t_utc"]), 0 if e["phase"] == "begin" else 1)
        for e in sorted(events, key=order):
            at = timestamp(e["t_utc"])
            stage = e["stage"]
            if at is None:
                raise ValueError(f"{e['evidence']}: event has no usable wall time")
            if e["phase"] == "mark":
                boundaries.setdefault(stage, []).append({"at": at, "evidence": e["evidence"]})
                attrs = e.get("attrs", {})
                if attrs.get("proxy") is True or "proxy" in str(attrs.get("basis", "")).lower():
                    alias = {"worker.claim": "proxy.start", "finish": "proxy.end"}.get(stage)
                    if alias:
                        boundaries.setdefault(alias, []).append({"at": at, "evidence": e["evidence"]})
                continue
            key = tuple(e.get(k) for k in ("host", "rental_id", "run_id", "card", "pid", "stage"))
            # Same-stage concurrent operations in one process need a stable
            # attrs.span_id. Nested spans without it pair in stack order.
            key += (e.get("attrs", {}).get("span_id"),)
            if e["phase"] == "begin":
                pending[key].append(e)
                continue
            if not pending[key] and not stage.startswith("forecast."):
                # The shell emitter launches a new Python helper for each
                # append, so its begin and end emitter pids differ. Pair only
                # an unambiguous context, never guess between concurrent jobs.
                candidates = [k for k, rows in pending.items() if rows
                              and k[:4] == key[:4] and k[5:] == key[5:]]
                if len(candidates) == 1:
                    key = candidates[0]
                    warnings.append(f"{e['evidence']}: paired unique stage context across emitter pids")
            if pending[key]:
                begin = pending[key].pop()
                start = timestamp(begin["t_utc"])
                end = at
                if begin.get("t_mono") is not None and e.get("t_mono") is not None:
                    end = start + e["t_mono"] - begin["t_mono"]
                    if abs(end - at) > 0.01:
                        warnings.append(f"{e['evidence']}: UTC adjusted; same-host monotonic duration used")
                if end < start:
                    raise ValueError(f"{e['evidence']}: reversed stage would corrupt wall accounting")
                intervals.append({"stage": stage, "start": start, "end": end,
                    "card": e.get("card"), "run_id": e.get("run_id"),
                    "basis": "C1" if begin.get("t_mono") is not None else "legacy_stage",
                    "attrs": {**begin.get("attrs", {}), **e.get("attrs", {})},
                    "evidence": [begin["evidence"], e["evidence"]]})
            else:
                warnings.append(f"{e['evidence']}: unmatched end for {stage}; no duration invented")
        for rows in pending.values():
            for e in rows:
                warnings.append(f"{e['evidence']}: unmatched begin for {e['stage']}; no duration invented")
        canonical = [r for r in intervals if r["stage"] == "forecast.integrate" and r.get("basis") == "C1"]
        for (rental, rid, prod, card, pid), steps in self._steps.items():
            if not selected({"rental_id": rental, "run_id": rid, "product": prod}):
                continue
            # Do not count a retrospective progress view again alongside C1.
            if any(r.get("run_id") in (None, rid) and r.get("card") == card for r in canonical):
                continue
            start, end = min(s[0] for s in steps), max(s[1] for s in steps)
            intervals.append({"stage": "forecast.integrate", "start": start, "end": end,
                "card": card, "run_id": rid, "pid": pid, "basis": "legacy_step_envelope",
                "evidence": [min(steps)[2], max(steps, key=lambda s: s[1])[2]]})
            warnings.append("Legacy step envelope uses emission time minus step duration; timestamps have millisecond resolution")
        def boundary(stage, choose=min):
            rows = boundaries.get(stage, [])
            return choose(r["at"] for r in rows) if rows else None
        requested, terminated = boundary("rental.requested"), boundary("rental.terminated", max)
        launched = boundary("legacy.rental.launched")
        proxy_start, proxy_end = boundary("proxy.start"), boundary("proxy.end", max)
        mode = "rental" if requested is not None and terminated is not None else "proxy"
        if mode == "rental":
            origin, endpoint = requested, terminated
        elif proxy_start is not None:
            origin = proxy_start
            # Last artifact means an observed output/finish mark, not latest
            # forecast-valid time, local file mtime, or parser invocation.
            artifact_marks = [boundary(s, max) for s in ("finish", "artifact.last", "proxy.end")]
            artifact_ends = [r["end"] for r in intervals if r["stage"].startswith(("post.", "upload.", "forecast.output."))]
            endpoints = [t for t in artifact_marks + artifact_ends if t is not None]
            endpoint = max(endpoints) if endpoints else None
        elif launched is not None and terminated is not None:
            mode, origin, endpoint = "legacy_rental", launched, terminated
            warnings.append("Rental request missing: launch-to-termination is a legacy denominator, not C2 rental FF")
        else:
            origin, endpoint = proxy_start, proxy_end
        wall = endpoint - origin if origin is not None and endpoint is not None else None
        if wall is not None and wall <= 0:
            raise ValueError("non-positive envelope would produce an invalid forecast fraction")
        def spans(rows):
            result = []
            for r in rows:
                start = max(r["start"], origin) if origin is not None else r["start"]
                end = min(r["end"], endpoint) if endpoint is not None else r["end"]
                if end > start:
                    result.append((start, end))
            return result
        integrates = [r for r in intervals if r["stage"] == "forecast.integrate"]
        integrated = union_seconds(spans(integrates)) if integrates else None
        first = min((r["start"] for r in integrates), default=boundary("forecast.first_step"))
        last = max((r["end"] for r in integrates), default=None)
        fraction = integrated / wall if integrated is not None and wall is not None else None
        if not integrates:
            warnings.append("No located integration interval: forecast fraction is unavailable")
        if wall is None:
            warnings.append("No complete measured envelope: wall and fraction are unavailable")
        if any(r["start"] < (origin if origin is not None else r["start"])
               or r["end"] > (endpoint if endpoint is not None else r["end"]) for r in integrates):
            warnings.append("Integration extends outside the envelope; fraction uses clipped intervals")
        unplaced = [r for r in self.unplaced if selected(r)]
        pipeline_seconds = union_seconds(spans([r for r in intervals if r["stage"] == "legacy.forecast.pipeline"]))
        observed_cards = {str(r["card"]) for r in integrates if r.get("card") is not None}
        if cards is None:
            counts = [e.get("attrs", {}).get("card_count") for e in events]
            cards = next((c for c in counts if isinstance(c, int) and not isinstance(c, bool) and c > 0), None)
        if cards is not None and (isinstance(cards, bool) or not isinstance(cards, int) or cards < 1 or len(observed_cards) > cards):
            raise ValueError("card count must cover allocated cards; otherwise occupancy is inflated")
        card_seconds = None
        if integrates and all(r.get("card") is not None for r in integrates):
            card_seconds = sum(union_seconds(spans([r for r in integrates if str(r["card"]) == card])) for card in observed_cards)
        occupancy = card_seconds / (cards * wall) if card_seconds is not None and cards is not None and wall else None
        # Integration card occupancy is a lower-bound activity proxy. It is
        # not kernel utilization. Explicit card_busy spans can establish a
        # broader measured busy fraction without charging host-only stages.
        busy = [r for r in intervals if r.get("attrs", {}).get("card_busy") is True]
        busy_fraction = None
        if busy and cards and wall and all(r.get("card") is not None for r in busy):
            busy_cards = {str(r["card"]) for r in busy}
            if len(busy_cards) > cards:
                raise ValueError("busy span cards exceed allocated cards")
            busy_fraction = sum(union_seconds(spans([r for r in busy if str(r["card"]) == c])) for c in busy_cards) / (cards * wall)
        if busy_fraction is None:
            warnings.append("Card-busy fraction is unmeasured; integration occupancy is reported separately")
        report = {"schema": "gpuwm.timeline.report.v1", "mode": mode, "product": product,
            "products": products,
            "rental_id": rental_id, "run_id": run_id, "ff": fraction if mode == "rental" else None,
            "proxy_ff": fraction if mode == "proxy" else None,
            "legacy_ff": fraction if mode == "legacy_rental" else None,
            "absolute_rental_wall_seconds": wall if mode == "rental" else None,
            "legacy_rental_wall_seconds": wall if mode == "legacy_rental" else None,
            "proxy_wall_seconds": wall if mode == "proxy" else None,
            "wall_seconds": wall, "origin": origin, "endpoint": endpoint,
            "integrate_union_seconds": integrated,
            "request_to_first_step_seconds": first - requested if first is not None and requested is not None else None,
            "origin_to_first_step_seconds": first - origin if first is not None and origin is not None else None,
            "tail_seconds": endpoint - last if endpoint is not None and last is not None else None,
            "card_busy_fraction": busy_fraction,
            "card_busy_basis": "explicit_attrs.card_busy_intervals" if busy_fraction is not None else "unmeasured",
            "integration_card_occupancy_fraction": occupancy, "allocated_cards": cards,
            "legacy_pipeline_union_seconds": pipeline_seconds or None,
            "legacy_pipeline_fraction_upper_bound": pipeline_seconds / wall if pipeline_seconds and wall else None,
            "intervals": sorted(intervals, key=lambda r: (r["start"], r["end"], r["stage"])),
            "boundaries": boundaries, "unplaced": unplaced,
            "inputs": self.inputs, "warnings": sorted(set(warnings)),
            "definition": "FF is the union of forecast.integrate across cards divided by rental.requested to rental.terminated. Proxy FF uses proxy.start to last artifact. Missing boundaries remain unknown."}
        if mode != "rental":
            report["request_to_first_step_seconds"] = None
        return report


def waterfall(report, path):
    """Analysis chart. Unknown work is not drawn as a located interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    intervals = report["intervals"]
    origin = report["origin"]
    if origin is None:
        origin = min((r["start"] for r in intervals), default=0)
    groups = defaultdict(list)
    for row in intervals:
        label = row["stage"]
        if row.get("card") is not None:
            label += f" / card {row['card']}"
        groups[label].append(row)
    labels = list(groups)
    height = min(20, max(4, len(labels) * 0.36 + 2))
    unplaced = [r for r in report["unplaced"] if seconds(r.get("seconds")) is not None]
    if unplaced and not intervals:
        fig, (ax, work_ax) = plt.subplots(2, 1, figsize=(12, height + 3), layout="constrained")
        work_ax.barh(range(len(unplaced)), [r["seconds"] for r in unplaced],
                     color="#a4a4a4", hatch="//")
        work_ax.set_yticks(range(len(unplaced)), [f"{r['stage']} [{i + 1}]" for i, r in enumerate(unplaced)])
        work_ax.invert_yaxis()
        work_ax.set_xlabel("Recorded work seconds, wall placement unknown; do not add rows")
        work_ax.set_title("Duration-only evidence, excluded from FF", fontsize=11)
    else:
        fig, ax = plt.subplots(figsize=(12, height), layout="constrained")
    for y, label in enumerate(labels):
        rows = groups[label]
        spans = union([(r["start"], r["end"]) for r in rows])
        color = "#218653" if label.startswith("forecast.integrate") else "#5c7caa"
        ax.broken_barh([(s - origin, e - s) for s, e in spans], (y - .35, .7), facecolors=color)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Wall seconds from measured origin")
    wall = report["wall_seconds"]
    if wall:
        ax.set_xlim(0, wall)
    mode = report["mode"]
    field = {"rental": "ff", "proxy": "proxy_ff", "legacy_rental": "legacy_ff"}[mode]
    ff = report[field]
    fraction = "unavailable" if ff is None else f"{ff:.4f}"
    title = {"rental": "RENTAL FF", "proxy": "PROXY FF", "legacy_rental": "LEGACY RENTAL FRACTION (launch basis)"}[mode]
    shown_wall = "unknown" if wall is None else f"{wall:.3f} s"
    ax.set_title(f"{report.get('product') or 'Forecast'}: {title} {fraction}\nMeasured envelope wall: {shown_wall}")
    note = "Green spans: integration union. Stage rows overlap and cannot be summed."
    if report["unplaced"]:
        note += f" {len(report['unplaced'])} duration-only measurements have no wall placement."
    if not intervals:
        ax.text(.5, .5, "No located stage spans were retained", transform=ax.transAxes, ha="center")
    fig.supxlabel(note, fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(report, output, png):
    output, png = Path(output), Path(png)
    output.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    # Render first so a missing optional plotting dependency cannot leave a
    # JSON claiming both artifacts were completed.
    waterfall(report, png)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    timeline = Timeline()
    for path in args.ledger:
        timeline.read(path)
    for stage, value in (("rental.requested", args.rental_requested),
                         ("rental.terminated", args.rental_terminated),
                         ("proxy.start", args.proxy_start), ("proxy.end", args.proxy_end)):
        if value is not None:
            timeline.boundary(stage, value, "explicit CLI --" + stage.replace(".", "-"))
    report = timeline.report(rental_id=args.rental_id, run_id=args.run_id,
                             product=args.product, cards=args.cards)
    output = Path(args.output)
    png = Path(args.png) if args.png else output.with_suffix(".png")
    write_report(report, output, png)
    return 0


def register_cli(sub):
    from pathlib import Path
    parser = sub.add_parser("timeline", help="calculate forecast fraction and waterfall, or project the critical path",
        description="Forecast fraction uses integration interval unions. gpuwm timeline LEDGER is shorthand for timeline report LEDGER. Proxy runs stay labelled proxy.")
    commands = parser.add_subparsers(dest="timeline_command", required=True)
    report = commands.add_parser("report", help="read C1 ledger, engine events, chain progress or site timing JSON")
    report.add_argument("ledger", type=Path, nargs="+", help="explicit timing JSON/JSONL paths; .jsonl.gz accepted")
    report.add_argument("--output", type=Path, default=Path("timeline.json"), help="JSON path; defaults to timeline.json")
    report.add_argument("--png", type=Path, help="waterfall path; defaults to the JSON path with .png suffix")
    report.add_argument("--rental-id", help="select one rental when a source has multiple rentals")
    report.add_argument("--run-id", help="select a run while retaining its rental boundaries")
    report.add_argument("--product", help="select product metadata; no product-specific accounting")
    report.add_argument("--cards", type=int, help="allocated card count, including idle cards")
    for flag in ("rental-requested", "rental-terminated", "proxy-start", "proxy-end"):
        report.add_argument("--" + flag, help="explicit measured boundary as zoned UTC or Unix seconds")
    report.set_defaults(func=run)
    project_parser = commands.add_parser("project", help="apply measured marginal deltas and reschedule a reference DAG",
        description="Union overlapping reference-work removals, apply depends_on marginal measurements, and report measured and projected FF side by side. References name measured durations and predecessor edges.")
    project_parser.add_argument("--reference", type=Path, action="append", required=True, help="gpuwm.timeline.reference.v1 JSON; repeat per product")
    project_parser.add_argument("--deltas", type=Path, action="append", default=[], help="measured lane delta JSON; repeatable, omitted for baseline")
    project_parser.add_argument("--output", type=Path, default=Path("timeline-project.json"))
    project_parser.add_argument("--png", type=Path, help="projected waterfall, for one product")
    from .project import run as project_run
    project_parser.set_defaults(func=project_run)
