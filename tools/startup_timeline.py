#!/usr/bin/env python3
"""Build an evidence-backed startup timeline from logs and receipts.

The tool reads only the paths supplied on the command line. Absolute wall
timestamps are required to calculate box-to-output latency. Untimed receipt
durations remain separate: worker durations and nested phases cannot be added
to an elapsed wall total. Source references include line numbers or JSON keys.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STAGES = (
    "install_bridges", "downloads", "statics_geography", "decode",
    "decode_compose", "horizontal_interpolation", "vertical_interpolation",
    "preparation", "seal", "bundle_verify", "bundle_restore", "health_scan", "model_build",
    "kernel_compile", "device_upload", "first_step", "forecast_to_first_hour",
)
ALIASES = {
    "system": "install_bridges", "engine": "install_bridges",
    "rust": "install_bridges", "venv312": "install_bridges",
    "venvft": "install_bridges", "tables": "install_bridges",
    "install": "install_bridges", "bridges": "install_bridges",
    "fetch": "downloads", "download": "downloads", "geog": "statics_geography",
    "static": "statics_geography", "statics": "statics_geography",
    "geography": "statics_geography", "decode_and_compose": "decode_compose",
    "horizontal": "horizontal_interpolation", "horizontal_seconds": "horizontal_interpolation",
    "initialize_all_times": "preparation", "initialization": "model_build",
    "initialize_physics": "model_build", "build_model": "model_build",
    "write_prepared_cache": "seal", "direct_wrf_export": "seal",
    "preflight_verify": "bundle_verify", "verify_prepared_cache": "bundle_verify",
    "preflight": "bundle_verify", "sealed_bundle_verify": "bundle_verify",
    "restore_prepared_cache": "bundle_restore", "restore_prepared_domain": "bundle_restore",
    "sealed_bundle_read": "bundle_restore",
    "store_full_state_health": "health_scan", "_store_full_state_health": "health_scan",
    "validate_store_fields": "health_scan", "validate_fields_cpu": "health_scan",
    "upload": "device_upload", "upload_to_device": "device_upload",
    "first_model_step": "first_step", "nvrtc": "kernel_compile",
}
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
CLOCK_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z?\]")


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value) and value >= 0:
            return value
    return None


def timestamp(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return number(value)


def model_timestamp(value: Any) -> float | None:
    if isinstance(value, str):
        return timestamp(value.replace("_", "T", 1))
    return None


def utc(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stage_name(name: str) -> str | None:
    name = name.lower().strip().replace("-", "_").replace(" ", "_")
    if name in STAGES:
        return name
    if name in ALIASES:
        return ALIASES[name]
    if "vertical_interpolation" in name:
        return "vertical_interpolation"
    if name.startswith("prep") or name == "initialize":
        return "preparation"
    if "compile" in name:
        return "kernel_compile"
    return None


def union_seconds(spans: list[tuple[float, float]]) -> float:
    """Elapsed covered time, with overlaps and duplicate spans counted once."""
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def evidence_ranges(rows: list[dict]) -> list[dict]:
    """Compact references retain raw source lines without repeating spans."""
    sources: dict[str, dict] = {}
    other = set()
    for row in rows:
        for reference in row.get("evidence", []):
            match = re.match(r"^(.*):(\d+)$", reference)
            if match:
                path, line = match[1], int(match[2])
                item = sources.setdefault(path, {"source": path, "first_line": line,
                                                  "last_line": line, "reference_count": 0})
                item["first_line"] = min(item["first_line"], line)
                item["last_line"] = max(item["last_line"], line)
                item["reference_count"] += 1
            else:
                other.add(reference)
    return list(sources.values()) + [{"reference": ref} for ref in sorted(other)]


@dataclass
class Timeline:
    origin: float | None = None
    intervals: list[dict] = field(default_factory=list)
    durations: list[dict] = field(default_factory=list)
    milestones: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    model_start: float | None = None
    forcing_rows: list[dict] = field(default_factory=list)
    _interval_index: dict = field(default_factory=dict, repr=False)

    def interval(self, stage: str, name: str, start: float, end: float, evidence: str,
                 *, basis: str = "explicit_wall_interval", host_call_only: bool = False) -> dict | None:
        if end < start:
            self.warnings.append(f"Rejected reversed interval: {evidence}")
            return
        key = (stage, name, start, end, host_call_only)
        item = self._interval_index.get(key)
        if item is not None:
            if evidence not in item["evidence"]:
                item["evidence"].append(evidence)
            return item
        item = {"stage": stage, "name": name, "start_unix_s": start,
                "end_unix_s": end, "duration_seconds": end - start,
                "basis": basis, "host_call_only": host_call_only,
                "timing_scope": "host_call" if host_call_only else "observed_wall_interval",
                "evidence": [evidence]}
        self.intervals.append(item)
        self._interval_index[key] = item
        return item

    def duration(self, stage: str, name: str, duration: float, evidence: str,
                 *, group: str | None = None, aggregate: bool = False) -> None:
        if not aggregate:
            for item in self.intervals:
                if item["stage"] == stage and item["name"] == name and abs(item["duration_seconds"] - duration) < 0.00001:
                    if evidence not in item["evidence"]:
                        item["evidence"].append(evidence)
                    return
        self.durations.append({"stage": stage, "name": name, "duration_seconds": duration,
                               "wall_placement": None, "group": group,
                               "aggregate": aggregate, "evidence": [evidence]})

    def milestone(self, name: str, at: float | None, evidence: str) -> None:
        if at is None:
            return
        previous = self.milestones.get(name)
        if previous is None or at < previous["unix_s"]:
            self.milestones[name] = {"unix_s": at, "utc": utc(at), "evidence": [evidence]}
        elif at == previous["unix_s"] and evidence not in previous["evidence"]:
            previous["evidence"].append(evidence)

    def event(self, item: dict, evidence: str, pending: dict) -> None:
        at = timestamp(item.get("utc", item.get("at", item.get("timestamp"))))
        emitted = number(item.get("emitted_unix_ms"))
        if emitted is not None:
            at = emitted / 1000.0
        if at is None:
            at = timestamp(item.get("unix_s", item.get("utc_unix_s")))
        event = str(item.get("event", ""))
        if event in ("box_available", "run_claimed", "origin"):
            self.milestone("origin", at, evidence)
        if event in ("run_start", "forecast_start"):
            self.milestone("forecast_start", at, evidence)
            model_start = model_timestamp(item.get("start_time"))
            if model_start is not None:
                self.model_start = model_start
        model_seconds = number(item.get("model_seconds", item.get("model_elapsed_seconds")))
        if model_seconds is None and self.model_start is not None:
            valid_time = model_timestamp(item.get("valid_time"))
            if valid_time is not None:
                model_seconds = valid_time - self.model_start
        domain = item.get("domain", item.get("grid_id", 1))
        if event == "step" and domain in (1, "1", "d01") and item.get("step") in (1, "1"):
            self.milestone("first_step", at, evidence)
            duration = number(item.get("step_wall_seconds"))
            if duration is not None and not any(row["stage"] == "first_step" for row in self.intervals + self.durations):
                if at is not None:
                    self.interval("first_step", "first_step", at - duration, at, evidence, basis="completion_minus_duration")
                else:
                    self.duration("first_step", "first_step", duration, evidence)
        if event in ("first_step", "first_forecast_hour_written"):
            self.milestone(event, at, evidence)
        if event == "output_written" and domain in (1, "1", "d01") and model_seconds is not None and abs(model_seconds - 3600) < 0.000001:
            self.milestone("first_forecast_hour_written", at, evidence)
        raw_stage = str(item.get("stage", item.get("name", item.get("module", ""))))
        stage = stage_name(raw_stage)
        name = str(item.get("name", item.get("filename", item.get("call", raw_stage))))
        if item.get("code") == "kernel_compile_progress":
            stage = "kernel_compile"
            name = str(item.get("module", "kernel_compile"))
        if stage is None:
            return
        pair_key = (name, item.get("pid"), item.get("worker"))
        if event == "stage_start":
            if at is not None:
                pending[pair_key] = (at, evidence)
            return
        if event == "stage_end":
            start_entry = pending.pop(pair_key, None)
            if at is not None and start_entry is not None:
                row = self.interval(stage, name, start_entry[0], at, start_entry[1])
                if row is not None and evidence not in row["evidence"]:
                    row["evidence"].append(evidence)
                return
        start = timestamp(item.get("start_unix_s", item.get("start_utc")))
        end = timestamp(item.get("end_unix_s", item.get("end_utc")))
        duration = number(item.get("duration_seconds", item.get("wall_seconds", item.get("seconds"))))
        if event == "phase" and item.get("measured_as"):
            # The native compile estimate is emitted at run end. Its emission
            # time is not its completion time and must not place it there.
            if duration is not None:
                self.duration(stage, name, duration, evidence, group=str(item["measured_as"]))
            return
        if start is not None and end is not None:
            row = self.interval(stage, name, start, end, evidence,
                                host_call_only=item.get("host_call_only") is True)
            if row is not None:
                for field_name in ("call", "bytes", "image_sha256", "image_bytes"):
                    if field_name in item:
                        row[field_name] = item[field_name]
        elif at is not None and duration is not None:
            self.interval(stage, name, at - duration, at, evidence, basis="completion_minus_duration")
        elif duration is not None:
            self.duration(stage, name, duration, evidence)

    def receipt(self, value: Any, evidence: str, pending: dict, key: str = "$") -> None:
        if isinstance(value, list):
            for i, child in enumerate(value):
                self.receipt(child, evidence, pending, f"{key}[{i}]")
            return
        if not isinstance(value, dict):
            return
        self.event(value, f"{evidence}#{key}", pending)
        if "compile_seconds" in value and "filename" in value and "status" in value:
            seconds = number(value["compile_seconds"])
            if seconds is not None:
                self.duration("kernel_compile", f"{value['filename']} compiler work", seconds,
                              f"{evidence}#{key}.compile_seconds", group=key)
        for name, child in value.items():
            child_key = f"{key}.{name}"
            if name == "forcing_stage_timings" and isinstance(child, list):
                for index, record in enumerate(child):
                    if not isinstance(record, dict):
                        continue
                    columns = record.get("column_stages_seconds", {})
                    self.forcing_rows.append({
                        "forcing_index": record.get("forcing_index"),
                        "valid_time": record.get("valid_time"),
                        "horizontal_seconds": number(record.get("horizontal_seconds")),
                        "total_seconds": number(record.get("total_seconds")),
                        "column_stages_seconds": {str(phase): seconds for phase, raw in columns.items()
                                                   if (seconds := number(raw)) is not None} if isinstance(columns, dict) else {},
                        "evidence": f"{evidence}#{child_key}[{index}]",
                    })
            if name == "timing_seconds" and isinstance(child, dict):
                for phase, seconds in child.items():
                    seconds = number(seconds)
                    stage = stage_name(phase)
                    if seconds is None:
                        continue
                    if phase == "total":
                        self.duration("total", "total", seconds,
                                      f"{evidence}#{child_key}.{phase}", group=child_key, aggregate=True)
                    elif stage is not None:
                        self.duration(stage, phase, seconds, f"{evidence}#{child_key}.{phase}", group=child_key)
            elif name == "horizontal_seconds":
                seconds = number(child)
                if seconds is not None:
                    self.duration("horizontal_interpolation", name, seconds, f"{evidence}#{child_key}", group=key)
            elif "vertical_interpolation" in name:
                seconds = number(child)
                if seconds is not None:
                    self.duration("vertical_interpolation", name, seconds, f"{evidence}#{child_key}", group=key)
            self.receipt(child, evidence, pending, child_key)

    def read(self, path: Path, *, date: str | None = None) -> None:
        self.inputs.append(str(path))
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace") as stream:
                text = stream.read()
        else:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        pending: dict[tuple, tuple] = {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if value is not None:
            self.receipt(value, str(path), pending)
            return
        day = datetime.fromisoformat(date).date() if date else None
        last_clock: float | None = None
        for line_number, line in enumerate(text.splitlines(), 1):
            evidence = f"{path}:{line_number}"
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    item = None
                if isinstance(item, dict):
                    self.event(item, evidence, pending)
                    continue
            at = None
            clock = CLOCK_RE.match(stripped)
            if clock and day is not None:
                parsed = datetime.fromisoformat(f"{day.isoformat()}T{clock[1]}+00:00")
                at = parsed.timestamp()
                if last_clock is not None and at < last_clock - 43200:
                    day += timedelta(days=1)
                    at += 86400
                last_clock = at
            elif clock:
                if "Clock-only log needs --date" not in self.warnings:
                    self.warnings.append("Clock-only log needs --date for absolute intervals")
            else:
                # A leading wall stamp is usable. A WRF filename or valid-time
                # embedded in the message is a model timestamp, not wall time.
                match = ISO_RE.match(stripped.lstrip("["))
                if match:
                    at = timestamp(match[0])
            body = CLOCK_RE.sub("", stripped).strip()
            start_match = re.fullmatch(r"start\s+([\w.-]+)", body)
            if start_match and at is not None:
                pending[(start_match[1], None, None)] = (at, evidence)
                continue
            done_match = re.fullmatch(r"done\s+([\w.-]+)\s+\(([\d.]+)\s*s\)", body)
            if done_match:
                name, seconds = done_match.groups()
                stage = stage_name(name)
                if stage is not None:
                    previous = pending.pop((name, None, None), None)
                    if at is not None:
                        start = previous[0] if previous else at - float(seconds)
                        row = self.interval(stage, name, start, at, evidence,
                                            basis="paired_log_markers" if previous else "completion_minus_duration")
                        if previous and row is not None and previous[1] not in row["evidence"]:
                            row["evidence"].append(previous[1])
                    else:
                        self.duration(stage, name, float(seconds), evidence)
                continue
            phase = re.search(r"gpuwm: phase ([\w.-]+):\s*([\d.]+)\s*elapsed seconds", body)
            if phase:
                self.event({"event": "phase", "name": phase[1], "wall_seconds": float(phase[2]), "unix_s": at}, evidence, pending)
            health = re.search(r"store-direct initialization.*full-state health gate.*?in\s+([\d.]+)s", body)
            if health:
                self.duration("health_scan", "store_full_state_health", float(health[1]), evidence)
        for name, _, _ in pending:
            self.warnings.append(f"No completion marker for {name} in {path}")

    def report(self, *, compact: bool = False) -> dict:
        origin = self.origin
        if origin is None and "origin" in self.milestones:
            origin = self.milestones["origin"]["unix_s"]
        endpoint = self.milestones.get("first_forecast_hour_written", {}).get("unix_s")
        first_step = self.milestones.get("first_step", {}).get("unix_s")
        if first_step is not None and endpoint is not None and endpoint >= first_step:
            self.interval("forecast_to_first_hour", "forecast_to_first_hour", first_step, endpoint,
                          "derived from first_step and first_forecast_hour_written milestones", basis="milestone_difference")
        def clipped_rows(start_bound, end_bound):
            rows = []
            for row in self.intervals:
                start, end = row["start_unix_s"], row["end_unix_s"]
                if start_bound is not None:
                    start = max(start, start_bound)
                if end_bound is not None:
                    end = min(end, end_bound)
                if end > start:
                    rows.append({**row, "start_unix_s": start, "end_unix_s": end,
                                 "duration_seconds": end - start})
            return rows

        def stage_summary(rows, *, include_unplaced):
            summary = []
            for stage in STAGES:
                spans = [(row["start_unix_s"], row["end_unix_s"]) for row in rows
                         if row["stage"] == stage and not row["host_call_only"]]
                host_spans = [(row["start_unix_s"], row["end_unix_s"]) for row in rows
                              if row["stage"] == stage and row["host_call_only"]]
                durations = ([row for row in self.durations if row["stage"] == stage and not row["aggregate"]]
                             if include_unplaced else [])
                summary.append({"stage": stage, "located_union_seconds": union_seconds(spans) if spans else None,
                                "host_call_union_seconds": union_seconds(host_spans) if host_spans else None,
                                "unplaced_recorded_work_seconds": sum(row["duration_seconds"] for row in durations) if durations else None,
                                "unplaced_count": len(durations),
                                "observed_interval_count": sum(row["stage"] == stage for row in rows),
                                "recorded_input_bytes": sum(int(number(row.get("bytes")) or 0) for row in rows if row["stage"] == stage),
                                "evidence_ranges": evidence_ranges([row for row in rows if row["stage"] == stage] + durations),
                                "status": "measured" if spans or durations else "host_calls_only" if host_spans else "unknown"})
            return summary

        def elapsed(start, end):
            return end - start if start is not None and end is not None and end >= start else None

        def window(start_bound, end_bound):
            # An absent endpoint cannot delimit a window. In particular,
            # later recurring calls are never silently treated as startup.
            rows = clipped_rows(start_bound, end_bound) if end_bound is not None else []
            total = elapsed(start_bound, end_bound)
            coverage = union_seconds([(row["start_unix_s"], row["end_unix_s"]) for row in rows])
            return {"origin_utc": utc(start_bound), "endpoint_utc": utc(end_bound),
                    "total_seconds": total,
                    "located_union_seconds": coverage if rows else None,
                    "unattributed_seconds": max(0.0, total - coverage) if total is not None else None,
                    "stage_summary": stage_summary(rows, include_unplaced=False),
                    **({"interval_count": len(rows)} if compact else {"intervals": rows}),
                    "boundary_status": "complete" if total is not None else "upper_bound_only" if end_bound is not None else "unknown",
                    "interpretation": "Only located intervals inside this window are ranked. Unplaced receipt work is excluded. The same callback stage after this endpoint is not assigned to this window."}

        clipped = clipped_rows(origin, endpoint)
        summary = stage_summary(clipped, include_unplaced=True)
        first_hour_window = window(origin, endpoint)
        startup_window = window(origin, first_step)
        after_first_step = (window(first_step, endpoint) if first_step is not None
                            else window(None, None))
        total = first_hour_window["total_seconds"]
        coverage = first_hour_window["located_union_seconds"]
        warnings = list(self.warnings)
        if origin is None:
            warnings.append("Box-available or run-claimed origin was not supplied or recorded")
        if endpoint is None:
            warnings.append("First forecast hour durable-write wall timestamp was not recorded")
        if first_step is None:
            warnings.append("First root-step completion wall timestamp was not recorded")
        forecast_start = self.milestones.get("forecast_start", {}).get("unix_s")
        forcing_summary: dict[str, dict] = {}
        for row in self.forcing_rows:
            values = {phase: seconds for phase, seconds in row["column_stages_seconds"].items()
                      if phase != "total_seconds"}
            if row["horizontal_seconds"] is not None:
                values["horizontal_interpolation"] = row["horizontal_seconds"]
            for phase, seconds in values.items():
                summary_row = forcing_summary.setdefault(phase, {
                    "name": phase, "recorded_work_seconds": 0.0,
                    "maximum_record_seconds": 0.0, "count": 0, "evidence": []})
                summary_row["recorded_work_seconds"] += seconds
                summary_row["maximum_record_seconds"] = max(summary_row["maximum_record_seconds"], seconds)
                summary_row["count"] += 1
                summary_row["evidence"].append(row["evidence"])
        duration_summary = []
        if compact:
            groups: dict[tuple, list[dict]] = {}
            for row in self.durations:
                groups.setdefault((row["stage"], row["name"], row["aggregate"]), []).append(row)
            for (stage, name, aggregate), rows in groups.items():
                duration_summary.append({"stage": stage, "name": name, "aggregate": aggregate,
                    "count": len(rows), "recorded_work_seconds": sum(row["duration_seconds"] for row in rows),
                    "evidence_ranges": evidence_ranges(rows)})
        return {"schema": "startup-timeline-v1", "origin_utc": utc(origin),
                "first_forecast_hour_written_utc": utc(endpoint), "total_seconds": total,
                "located_union_seconds": coverage,
                "unattributed_seconds": first_hour_window["unattributed_seconds"],
                "latencies_seconds": {"origin_to_first_step": elapsed(origin, first_step),
                                      "origin_to_first_hour_written": elapsed(origin, endpoint),
                                      "forecast_start_to_first_step": elapsed(forecast_start, first_step),
                                      "forecast_start_to_first_hour_written": elapsed(forecast_start, endpoint)},
                "startup_until_first_step": startup_window,
                "first_hour_window": first_hour_window,
                "after_first_step_until_first_hour": after_first_step,
                "stage_summary": summary,
                **({"interval_count_total": len(self.intervals), "unplaced_duration_count": len(self.durations),
                    "unplaced_duration_summary": duration_summary, "compact": True}
                   if compact else {"intervals": sorted(self.intervals, key=lambda row: row["start_unix_s"]),
                                    "duration_only": self.durations, "compact": False}),
                "forcing_work_breakdown": {"records": self.forcing_rows,
                                           "summary": sorted(forcing_summary.values(), key=lambda row: -row["recorded_work_seconds"]),
                                           "interpretation": "Native forcing labels are retained without recategorizing CPU copy phases as GPU upload. These unplaced work sums can nest or run in parallel, are part of preparation, and are not added to its elapsed wall total. Column and forcing totals are preserved separately from leaf work."},
                "milestones": self.milestones,
                "inputs": self.inputs, "warnings": list(dict.fromkeys(warnings)),
                "interpretation": "The top-level stage_summary retains the first-hour window plus unplaced work. Use startup_until_first_step for startup-only rankings and after_first_step_until_first_hour for recurring callbacks during stepping. Located intervals use wall-time unions. Host-call intervals measure API return, not synchronized device-transfer completion. Recorded input bytes belong to whole calls intersecting a window; calls may straddle cutoffs, so byte totals across windows must not be summed. Unplaced durations may nest or run in parallel and are recorded work, not elapsed wall totals. Stage rows can overlap and must not be summed."}


def markdown(report: dict) -> str:
    def shown(value: float | None) -> str:
        return "unknown" if value is None else f"{value:.3f}"
    lines = ["# Startup timeline", "", f"Recorded origin to first step completed: {shown(report['startup_until_first_step']['total_seconds'])} s.",
             f"Recorded origin to first forecast hour written: {shown(report['total_seconds'])} s.", "", report["interpretation"], "",
             "Startup through first-step completion:", "", "| Stage | Located wall union, s | Host-call union, s |", "|---|---:|---:|"]
    for row in report["startup_until_first_step"]["stage_summary"]:
        lines.append(f"| {row['stage']} | {shown(row['located_union_seconds'])} | {shown(row['host_call_union_seconds'])} |")
    lines.extend(["", "First-hour window plus unplaced receipt work:", "",
                  "| Stage | Located wall union, s | Host-call union, s | Unplaced recorded work, s |", "|---|---:|---:|---:|"])
    for row in report["stage_summary"]:
        lines.append(f"| {row['stage']} | {shown(row['located_union_seconds'])} | {shown(row['host_call_union_seconds'])} | {shown(row['unplaced_recorded_work_seconds'])} |")
    lines.extend(["", f"Forecast start to first step: {shown(report['latencies_seconds']['forecast_start_to_first_step'])} s.",
                  f"Forecast start to first forecast hour written: {shown(report['latencies_seconds']['forecast_start_to_first_hour_written'])} s."])
    lines.extend(["", "Evidence:", ""])
    for row in report.get("intervals", []) + report.get("duration_only", []):
        scope = " host-call" if row.get("host_call_only") else ""
        lines.append(f"- {row['name']}: {row['duration_seconds']:.3f} s{scope}; " + "; ".join(row["evidence"]))
    if report.get("compact"):
        for row in report["stage_summary"]:
            if row["observed_interval_count"] or row["unplaced_count"]:
                lines.append(f"- {row['stage']}: {row['observed_interval_count']} observed intervals, {row['recorded_input_bytes']} recorded input bytes; "
                             + json.dumps(row["evidence_ranges"], sort_keys=True))
    if report["warnings"]:
        lines.extend(["", "Unmeasured or incomplete:", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def csv_text(report: dict) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("window", "stage", "located_union_seconds", "host_call_union_seconds", "unplaced_recorded_work_seconds", "unplaced_count", "status", "observed_interval_count", "recorded_input_bytes"), extrasaction="ignore")
    writer.writeheader()
    writer.writerows({"window": "first_hour_plus_unplaced", **row} for row in report["stage_summary"])
    for name in ("startup_until_first_step", "first_hour_window", "after_first_step_until_first_hour"):
        writer.writerows({"window": name, **row} for row in report[name]["stage_summary"])
    return stream.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=Path, default=[], help="explicit text or JSONL log; repeatable")
    parser.add_argument("--receipt", action="append", type=Path, default=[], help="explicit JSON receipt; repeatable")
    parser.add_argument("--date", help="UTC date for bracketed clock-only logs, YYYY-MM-DD")
    parser.add_argument("--origin", help="box-available or run-claimed UTC wall timestamp")
    parser.add_argument("--output", type=Path, help="JSON report path; stdout when omitted")
    parser.add_argument("--csv", type=Path, help="stage summary CSV path")
    parser.add_argument("--markdown", type=Path, help="plain Markdown report path")
    parser.add_argument("--summary", action="store_true", help="omit repeated raw spans; keep unions, counts, input bytes and source ranges")
    args = parser.parse_args(argv)
    if not args.log and not args.receipt:
        parser.error("supply at least one explicit --log or --receipt path")
    origin = timestamp(args.origin) if args.origin else None
    if args.origin and origin is None:
        parser.error("--origin must be an ISO wall timestamp")
    if args.date:
        try:
            datetime.fromisoformat(args.date)
        except ValueError:
            parser.error("--date must be YYYY-MM-DD")
    timeline = Timeline(origin=origin)
    for path in args.log + args.receipt:
        timeline.read(path, date=args.date)
    report = timeline.report(compact=args.summary)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.suffix.lower() == ".gz":
            with gzip.open(args.output, "wt", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.csv:
        args.csv.write_text(csv_text(report), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
