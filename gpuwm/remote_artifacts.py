"""Bounded transfer of one existing, manifest-committed remote WRF frame.

This is a byte transport. No model, forcing, NetCDF rewrite or field extraction
is performed here. The consumer uses the existing native WRF reader.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time

SCHEMA = "gpuwm.remote-artifacts.v1"
INDEX_SCHEMA = "gpuwm.remote-artifact-index.v1"
MAX_INDEX_PAGE = 256
MAX_FRAME = 512 * 1024 * 1024
MAX_CACHE = 2 * 1024 * 1024 * 1024
MAX_EVENTS = 64 * 1024 * 1024
MAX_LINE = 128 * 1024
MAX_RECORDS = 100_000
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("remote artifact identity lacks a UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("_", "T", 1))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _domain(value):
    if type(value) is not int or not 1 <= value <= 999:
        raise ValueError("artifact domain must be an integer between 1 and 999")
    return value


def _inside(value, root, *, file=True):
    if not isinstance(value, str) or any(ord(c) < 32 for c in value):
        raise ValueError("artifact authority needs an absolute regular-file path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact authority needs an absolute path without traversal")
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError("artifact authority is outside this job's output tree") from None
    current = root
    if current.is_symlink():
        raise ValueError("artifact output tree must not be a symlink")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("artifact authority must not traverse a symlink")
    path = path.resolve(strict=True)
    if not path.is_relative_to(root) or (file and not path.is_file()):
        raise ValueError("artifact authority is not a regular file in this job")
    return path


def _raw(path, maximum):
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError(f"Artifact metadata '{path.name}' exceeds its {maximum}-byte limit")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Artifact metadata '{path.name}' needs a JSON object")
    return value, payload


def _stamp(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _authority(path, payload, *, sequence=None):
    value = {"remote_path": str(path), "sha256": _sha(payload), "utf8": payload.decode("utf-8")}
    if sequence is not None:
        value["sequence"] = sequence
    return value


def _resolved_authority(manifest, root, started):
    """Read the native config receipt near the start, without scanning outputs."""
    path = _inside(manifest.get("events_path"), root)
    scanned, previous = 0, 0
    with path.open("rb") as stream:
        for _ in range(64):
            line = stream.readline(MAX_LINE + 1)
            if not line or not line.endswith(b"\n"):
                return None
            scanned += len(line)
            if len(line) > MAX_LINE or scanned > 1024 * 1024:
                raise ValueError("Native producer configuration receipt exceeds its bounded scan")
            event = json.loads(line)
            if (not isinstance(event, dict) or event.get("schema_version") != "gpuwm.run-plan.event.v1"
                    or type(event.get("sequence")) is not int or event["sequence"] <= previous):
                raise ValueError("Native producer configuration event identity or sequence is invalid")
            previous = event["sequence"]
            if event.get("event") == "resolved_plan":
                if type(event.get("emitted_unix_ms")) is not int or event["emitted_unix_ms"] < started:
                    raise ValueError("Native producer configuration receipt predates this run")
                return event, _authority(path, line, sequence=event["sequence"])
    raise ValueError("Native producer configuration receipt was not published within its bounded event prefix")


def _hosted_producer(record, state, outer):
    """Follow only the native chain pointer, with exact saved-config linkage."""
    root, parent_path, parent, parent_bytes, parent_started = outer
    if parent.get("route") != "prepared":
        return (*outer, None)
    pointer = root / "chain" / "latest-run.txt"
    if not pointer.exists():
        return (*outer, None)
    pointer = _inside(str(pointer), root)
    with pointer.open("rb") as stream:
        pointer_bytes = stream.read(257)
    from gpuwm.run_stamp import is_run_folder
    name = pointer_bytes.decode("utf-8").strip()
    if (len(pointer_bytes) > 256 or not name or "/" in name or "\\" in name
            or not is_run_folder(name) or any(ord(c) < 32 for c in name)):
        raise ValueError("Native chain pointer does not name one owned stamped run")
    producer_root = _inside(str(pointer.parent / name), root, file=False)
    if not producer_root.is_dir():
        raise ValueError("Native chain pointer does not name a run directory")
    path = producer_root / "run-manifest.json"
    if not path.exists():
        return (*outer, None)  # The chain publishes its pointer before its manifest.
    path = _inside(str(path), producer_root)
    producer, payload = _raw(path, 48 * 1024)
    started = _timestamp(producer.get("started_at_utc"))
    config_path = _inside(record.get("snapshot_config"), Path(record["snapshot_plan"]).parent)
    if config_path.stat().st_size > 128 * 1024:
        raise ValueError("Native producer saved configuration exceeds its metadata byte limit")
    config_hash = _file_sha(config_path)
    if (config_hash != record.get("config_sha256") or config_hash != record.get("snapshot_sha256")
            or producer.get("schema") != "gpuwm.run-manifest.v1"
            or producer.get("run_dir") != str(producer_root) or producer.get("outputs_dir") != str(producer_root)
            or producer.get("pid") != parent["pid"]
            or producer.get("plan_source") != "gpuwm go " + str(config_path)
            or not HEX.fullmatch(str(producer.get("plan_sha256", "")))
            or not isinstance(producer.get("run_id"), str) or not producer["run_id"]
            or producer["run_id"] == parent["run_id"] or started < parent_started
            or (state.get("ended_at") and started > _timestamp(state["ended_at"]))):
        raise ValueError("Native chain producer does not match this job's saved configuration, process and time")
    parent_resolved = _resolved_authority(parent, root, parent_started)
    producer_resolved = _resolved_authority(producer, producer_root, started)
    if parent_resolved is None or producer_resolved is None:
        return (*outer, None)
    for resolved, _raw_receipt in (parent_resolved, producer_resolved):
        if resolved.get("config_source") != str(config_path) or resolved.get("config_sha256") != config_hash:
            raise ValueError("Native chain configuration receipts disagree with this job's exact saved config")
    if parent_resolved[0]["emitted_unix_ms"] > started:
        raise ValueError("Native chain producer started before the parent resolved its configuration")
    with pointer.open("rb") as stream:
        pointer_after = stream.read(257)
    if pointer_after != pointer_bytes:
        raise ValueError("Native chain pointer changed during producer attachment; refresh this job")
    binding = {"schema": "gpuwm.remote-producer-binding.v1",
               "parent_manifest": _authority(parent_path, parent_bytes),
               "chain_pointer": _authority(pointer, pointer_bytes),
               "parent_resolved": parent_resolved[1], "producer_resolved": producer_resolved[1]}
    return producer_root, path, producer, payload, started, binding


def bound_manifest(record, state):
    """Shared native run identity for bounded status and committed frame reads."""
    from gpuwm import remote_worker as rw
    if record.get("action") != "start-plan" or not record.get("snapshot_plan"):
        return None
    root = Path(record["outdir"])
    if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
        raise ValueError("recorded remote output tree changed or is not canonical")
    manifest_path = root / "run-manifest.json"
    if not manifest_path.exists():
        return None
    manifest_path = _inside(str(manifest_path), root)
    manifest, manifest_bytes = _raw(manifest_path, 48 * 1024)
    started = _timestamp(manifest.get("started_at_utc"))
    pid = manifest.get("pid")
    if (manifest.get("schema") != "gpuwm.run-manifest.v1"
            or manifest.get("run_dir") != str(root) or manifest.get("outputs_dir") != str(root)
            or manifest.get("plan_source") != record["snapshot_plan"]
            or manifest.get("plan_sha256") != record.get("plan_sha256")
            or started < _timestamp(record["created_at"])
            or type(pid) is not int or pid <= 0
            or not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]):
        raise ValueError("Remote run manifest does not match this job's saved plan, process and output identity")
    if state["state"] in {"running", "starting"} and not rw._has_token(pid, record["token"]):
        raise ValueError("Cannot prove the run manifest process belongs to this active remote job")
    if state["state"] in {"ownership_mismatch", "lost"}:
        raise ValueError("Resolve this remote job's process ownership before retrieving its frames")
    if state.get("ended_at") and started > _timestamp(state["ended_at"]):
        raise ValueError("Remote manifest was published after this job ended")
    return _hosted_producer(record, state, (root, manifest_path, manifest, manifest_bytes, started))


def _sequence(value, *, cursor=False):
    if type(value) is not int or not (0 if cursor else 1) <= value < 1 << 63:
        raise ValueError("Artifact sequence must be a positive integer" if not cursor else "Artifact index cursor must be a nonnegative integer")
    return value


def _bounded(value):
    if len(_encoded(value)) > 110 * 1024:
        raise ValueError("Selected committed frame metadata exceeds the bounded SSH response")
    return value


def catalog(request, workspace, *, sequence=None, metadata_only=False):
    """Read the selected durable job's existing commit stream; never glob data."""
    from gpuwm import remote_worker as rw
    fields = {"schema", "action", "workspace", "job", "domain", "after_sequence" if metadata_only else "sequence"}
    if not (sequence is not None and request.get("action") == "stream-artifact") and set(request) - fields:
        raise ValueError("unsupported remote artifact request fields")
    if sequence is None and "sequence" in request:
        sequence = _sequence(request["sequence"])
    elif sequence is not None:
        sequence = _sequence(sequence)
    after_sequence = _sequence(request.get("after_sequence", 0), cursor=True) if metadata_only else 0
    domain = _domain(request.get("domain", 1))
    directory = rw._directory(workspace, request.get("job"))
    record, state = rw._record(directory), rw._status(directory)
    if record.get("action") != "start-plan" or not record.get("snapshot_plan"):
        raise ValueError("Committed frame retrieval currently requires this job's saved run-plan manifest")
    result = {"schema": SCHEMA, "job_id": record["id"], "remote_output_root": record["outdir"],
              "available_domains": [], "frames": [], "waiting": True}
    if metadata_only:
        result.update(schema=INDEX_SCHEMA, domain=domain, entries=[], next_after_sequence=None, latest_sequence=None)
        result.pop("frames")
    bound = bound_manifest(record, state)
    if bound is None:
        return result
    root, manifest_path, manifest, manifest_bytes, started, producer_binding = bound
    pid = manifest["pid"]
    result.update({"run_id": manifest["run_id"], "remote_pid": pid,
                   "run_manifest": {"remote_path": str(manifest_path), "sha256": _sha(manifest_bytes),
                                    "utf8": manifest_bytes.decode("utf-8")}})
    if producer_binding is not None:
        result["producer_binding"] = producer_binding
    events_path = _inside(manifest.get("events_path"), root)
    if events_path.stat().st_size > MAX_EVENTS:
        raise ValueError("This run's event stream exceeds the 64 MiB preview scan limit")
    selected, selected_bytes, last_sequence, domains, scanned = None, None, 0, set(), 0
    indexed, more_entries, latest_sequence = [], False, None
    with events_path.open("rb") as stream:
        for index in range(MAX_RECORDS + 1):
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            scanned += len(line)
            if len(line) > MAX_LINE or scanned > MAX_EVENTS or index == MAX_RECORDS:
                raise ValueError("This run's commit stream exceeds the bounded preview record limit")
            if not line.endswith(b"\n"):
                if state["state"] in rw.TERMINAL:
                    raise ValueError("Terminal remote job has an incomplete final event line")
                break
            event = json.loads(line)
            if (not isinstance(event, dict) or event.get("schema_version") != "gpuwm.run-plan.event.v1"
                    or type(event.get("sequence")) is not int or event["sequence"] <= last_sequence):
                raise ValueError("Remote run event schema or increasing sequence is invalid")
            last_sequence = event["sequence"]
            if event.get("event") != "output_committed":
                continue
            emitted = event.get("emitted_unix_ms")
            if type(emitted) is not int:
                raise ValueError("Committed output has no exact publication timestamp")
            if emitted < started:
                continue
            event_domain = _domain(event.get("domain"))
            domains.add(event_domain)
            if event_domain == domain and metadata_only:
                _timestamp(event.get("valid_time"))
                latest_sequence = event["sequence"]
                if event["sequence"] > after_sequence:
                    if len(indexed) < MAX_INDEX_PAGE:
                        indexed.append({"sequence": event["sequence"], "domain": domain, "valid_time": event["valid_time"]})
                    else:
                        more_entries = True
            if event_domain == domain and (sequence is None or event["sequence"] == sequence):
                selected, selected_bytes = event, line
    result["available_domains"] = sorted(domains)
    if metadata_only:
        result.update(entries=indexed, latest_sequence=latest_sequence, waiting=latest_sequence is None,
                      next_after_sequence=indexed[-1]["sequence"] if more_entries else None)
        return _bounded(result)
    if selected is None:
        return _bounded(result)
    if len(selected_bytes) > 16 * 1024:
        raise ValueError("Selected frame commit exceeds the 16 KiB transport metadata limit")
    _timestamp(selected.get("valid_time"))
    path = _inside(selected.get("path"), root)
    before = _stamp(path)
    if not 0 < before[2] <= MAX_FRAME:
        raise ValueError(f"Committed frame '{path.name}' is {before[2]} bytes; preview transfer allows at most 512 MiB")
    if selected.get("size_bytes") is not None and selected["size_bytes"] != before[2]:
        raise ValueError("Committed remote WRF byte length changed after publication")
    digest = _file_sha(path)
    if _stamp(path) != before:
        raise ValueError("Committed remote WRF file changed while hashing; refresh its commit")
    frame = {"remote_path": str(path), "sha256": digest, "size_bytes": before[2],
             "domain": domain, "valid_time": selected["valid_time"],
             "commit": {"remote_path": str(events_path), "sequence": selected["sequence"],
                        "sha256": _sha(selected_bytes), "utf8": selected_bytes.decode("utf-8")}}
    frame["id"] = _sha(_encoded([record["id"], manifest["run_id"], frame]))
    result.update({"waiting": False, "frames": [frame]})
    return _bounded(result)


def _finite_seconds(value, *, positive=False):
    import math
    return (float(value) if type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0) else None)


def _progress_schedule(record):
    """Only the immutable, hash-bound TOML supplies the planned save cadence."""
    import tomllib
    if not record.get("snapshot_config"):
        return None
    path = _inside(record["snapshot_config"], Path(record["snapshot_plan"]).parent)
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if (len(raw) > 128 * 1024 or _sha(raw) != record.get("config_sha256")
            or _sha(raw) != record.get("snapshot_sha256")):
        raise ValueError("Progress schedule does not match the saved configuration")
    config = tomllib.loads(raw.decode("utf-8"))
    experiment = config.get("experiment", {})
    total = _finite_seconds(experiment.get("run_seconds"), positive=True)
    restart = _finite_seconds(experiment.get("restart_interval_s"))
    domains = config.get("domain")
    if total is None or restart is None or not isinstance(domains, list) or not 1 <= len(domains) <= 999:
        raise ValueError("Saved progress schedule has invalid timing or domains")
    schedule = []
    for domain in domains:
        grid_id = _domain(domain.get("grid_id"))
        interval = _finite_seconds(domain.get("history_interval_s"), positive=True)
        if interval is None or any(item[0] == grid_id for item in schedule):
            raise ValueError("Saved progress schedule has invalid output cadence or duplicate domains")
        schedule.append((grid_id, interval))
    start_time = experiment.get("start_time")
    if isinstance(start_time, datetime):
        start_time = start_time.isoformat()
    return {"start_ms": _timestamp(start_time), "run_seconds": total,
            "restart_interval_s": restart, "domains": schedule}


def _next_planned_save(model_seconds, interval, total):
    """The next configured boundary strictly after the reported model clock."""
    import math
    if model_seconds is None or not interval or model_seconds >= total:
        return None
    next_seconds = (math.floor(model_seconds / interval + 1e-9) + 1) * interval
    return next_seconds if next_seconds <= total + 1e-7 else None


def _forecast_progress(record, manifest, manifest_bytes, root, model, outputs, heartbeat, phase):
    schedule = _progress_schedule(record)
    if schedule is None:
        return None
    model = model or {}
    elapsed = _finite_seconds(model.get("model_seconds"))
    if elapsed is None:
        elapsed = _finite_seconds((heartbeat or {}).get("model_elapsed_seconds"))
    total = schedule["run_seconds"]
    if elapsed is not None and elapsed > total + 1e-6:
        raise ValueError("Native progress exceeds this saved forecast duration")
    def valid_time(seconds):
        if seconds is None:
            return None
        return datetime.fromtimestamp(schedule["start_ms"] / 1000 + seconds,
                                      timezone.utc).isoformat().replace("+00:00", "Z")
    def saved_seconds(event):
        seconds = (_timestamp(event["valid_time"]) - schedule["start_ms"]) / 1000
        return seconds if 0 <= seconds <= total else None
    clocks = {}
    for row in model.get("domains", []):
        seconds = _finite_seconds(row.get("model_seconds"))
        if seconds is not None and seconds <= total + 1e-6:
            clocks[_domain(row.get("domain"))] = seconds
    root_id = model.get("domain")
    if type(root_id) is int and elapsed is not None:
        clocks[root_id] = elapsed
    domains = []
    for grid_id, interval in schedule["domains"]:
        current = clocks.get(grid_id)
        # With one domain the heartbeat's clock is unambiguous; on a tree,
        # absent child clocks stay unavailable instead of borrowing d01's.
        if current is None and len(schedule["domains"]) == 1:
            current = elapsed
        next_seconds = _next_planned_save(current, interval, total)
        domains.append({"grid_id": grid_id, "model_seconds": current,
                        "history_interval_s": interval,
                        "last_save_model_seconds": saved_seconds(outputs[grid_id]) if grid_id in outputs else None,
                        "next_save_model_seconds": next_seconds,
                        "next_save_in_seconds": None if next_seconds is None else max(0., next_seconds - current)})
    checkpoint = {"interval_seconds": schedule["restart_interval_s"], "last_saved_model_seconds": None}
    # The supervisor validates a checkpoint before publishing this pointer.
    # Read only its owned path/name using the canonical discovery grammar.
    named = model.get("last_checkpoint") or (heartbeat or {}).get("last_checkpoint")
    if isinstance(named, str):
        from gpuwm.resume import _CHECKPOINT_NAME, _INSTANT_FORMAT
        try:
            path = _inside(named, root)
            match = _CHECKPOINT_NAME.fullmatch(path.name)
            if match and int(match.group("grid_id")) == min(row[0] for row in schedule["domains"]):
                when = datetime.strptime(match.group("instant"), _INSTANT_FORMAT).replace(tzinfo=timezone.utc)
                seconds = (int(when.timestamp() * 1000) - schedule["start_ms"]) / 1000
                if elapsed is not None and 0 <= seconds <= elapsed:
                    checkpoint["last_saved_model_seconds"] = seconds
        except (OSError, ValueError):
            pass
    next_checkpoint = _next_planned_save(elapsed, checkpoint["interval_seconds"], total)
    checkpoint.update(next_model_seconds=next_checkpoint,
                      in_seconds=None if next_checkpoint is None else max(0., next_checkpoint - elapsed))
    outer_step = model.get("outer_step", (heartbeat or {}).get("outer_step"))
    if type(outer_step) is not int or outer_step < 0:
        outer_step = None
    updated = model.get("emitted_unix_ms")
    if updated is None and heartbeat:
        updated = _timestamp(heartbeat["updated_at_utc"])
    return {"schema": "arwen.forecast-progress.v1", "outer_step": outer_step,
            "model_seconds": elapsed, "run_seconds": total,
            "wall_seconds": _finite_seconds(model.get("wall_seconds")),
            "speed_x": _finite_seconds(model.get("speed_x"), positive=True),
            "step_ms": _finite_seconds(model.get("step_ms"), positive=True),
            "valid_time": valid_time(elapsed), "updated_unix_ms": updated, "phase": phase,
            "domains": domains, "checkpoint": checkpoint,
            "source": {"run_id": manifest["run_id"], "manifest_sha256": _sha(manifest_bytes),
                       "snapshot_config_sha256": record["config_sha256"],
                       "event_sequence": model.get("sequence")}}


def _pipeline_progress(events, result, state, started):
    """Observed acquisition/preparation facts, separate from integration clocks."""
    stage = result.get("stage", "starting")
    if stage.startswith("preparing:"):
        stage = "prepare"
    phase = result.get("phase", stage)
    stage_started, updated, finished_wall = started, started, None
    acquisition, preparation, files = {}, None, {}
    for event in events:
        emitted = event["emitted_unix_ms"]
        updated = max(updated, emitted)
        tag = event.get("event")
        if tag == "stage_started":
            stage_started, finished_wall = emitted, None
        elif tag == "stage_finished":
            finished_wall = _finite_seconds(event.get("wall_seconds"))
        supplied = event.get("acquisition")
        if isinstance(supplied, dict) and supplied.get("schema") == "arwen.acquisition-progress.v1":
            for key in ("source", "provider", "phase", "dataset"):
                value = supplied.get(key)
                if isinstance(value, str) and len(value) <= 160:
                    acquisition[key] = value
            for key in ("requests_completed", "requests_total", "request_index", "files_completed", "files_total",
                        "forcing_hours", "forcing_times_total", "forcing_times_completed", "bytes_available", "transferred_bytes"):
                value = supplied.get(key)
                if type(value) is int and 0 <= value < 1 << 63:
                    acquisition[key] = value
            if type(supplied.get("reused")) is bool:
                acquisition["reused"] = supplied["reused"]
        name = event.get("file")
        if tag in ("fetch_started", "fetch_progress", "fetch_completed") and isinstance(name, str) and 0 < len(name) <= 512:
            item = files.setdefault(name, {})
            if tag == "fetch_started":
                item.clear()
            for key in ("bytes", "expected_bytes"):
                value = event.get(key)
                if type(value) is int and 0 <= value < 1 << 63:
                    item[key] = value
            if tag == "fetch_completed":
                item["completed"] = event.get("failed") is False
        if event.get("code") == "preparation_progress" and isinstance(event.get("preparation"), dict):
            supplied = event["preparation"]
            if supplied.get("schema") in ("gpuwm.prep-stage.v1", "gpuwm.prepare-progress/v1"):
                preparation = {key: supplied[key] for key in
                    ("schema", "label", "stage", "event", "backend", "index", "count", "phase", "phase_index", "phases_total", "elapsed_seconds", "outcome")
                    if key in supplied and isinstance(supplied[key], (str, int, float))
                    and len(str(supplied[key])) <= 512}
    if files:
        acquisition.setdefault("transferred_bytes", sum(item.get("bytes", 0) for item in files.values()))
        acquisition.setdefault("files_completed", sum(item.get("completed", False) for item in files.values()))
        expected = [item.get("expected_bytes") for item in files.values()]
        acquisition["expected_bytes"] = (sum(expected) if all(type(n) is int for n in expected)
                                           and len(files) == acquisition.get("files_total") else None)
    if stage == "fetch" and acquisition.get("phase"):
        phase = acquisition["phase"]
    elif stage in ("prepare", "initialize") and preparation is not None:
        phase = preparation.get("label", preparation.get("phase", phase))
    now = int(time.time() * 1000)
    if state.get("ended_at"):
        now = min(now, _timestamp(state["ended_at"]))
    value = {"schema": "arwen.pipeline-progress.v1", "stage": stage, "phase": phase,
             "state": state.get("state"), "started_unix_ms": stage_started, "updated_unix_ms": updated,
             "wall_seconds": (finished_wall if finished_wall is not None else max(0., (now - stage_started) / 1000.))}
    if acquisition:
        value["acquisition"] = acquisition
    if preparation is not None:
        value["preparation"] = preparation
    return value


def native_progress(record, state):
    """Small native progress/failure summary; never reads a meteorological file."""
    import math
    bound = bound_manifest(record, state)
    if bound is None:
        return {}
    root, _manifest_path, manifest, manifest_bytes, started, _producer_binding = bound
    result = {}
    heartbeat, latest_model, outputs, observed_events = None, None, {}, []
    progress_path = manifest.get("progress_path")
    if progress_path and Path(progress_path).exists():
        progress, _ = _raw(_inside(progress_path, root), 64 * 1024)
        from gpuwm.supervisor import HEARTBEAT_SCHEMA
        if (progress.get("schema") == HEARTBEAT_SCHEMA and progress.get("run_id") == manifest["run_id"]
                and progress.get("pid") == manifest["pid"] and progress.get("config_digest") == record.get("config_sha256")
                and _timestamp(progress.get("started_at_utc")) == started
                and _timestamp(progress.get("updated_at_utc")) >= started):
            heartbeat = progress
            elapsed = progress.get("model_elapsed_seconds")
            if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
                result["model_elapsed_seconds"] = elapsed
            phase = progress.get("status")
            if isinstance(phase, str) and len(phase) <= 160:
                result["stage"] = phase
                result["phase"] = phase
                result["phase_updated_unix_ms"] = _timestamp(progress["updated_at_utc"])
    events_path = _inside(manifest.get("events_path"), root)
    with events_path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - 512 * 1024)
        stream.seek(start)
        if start:
            stream.readline(MAX_LINE + 1)  # Discard only the truncated leading record.
        lines = stream.read(512 * 1024).splitlines(keepends=True)
    last_sequence = 0
    for line in lines:
        if not line.endswith(b"\n"):
            break
        if len(line) > MAX_LINE:
            raise ValueError("Native status event exceeds its bounded record length")
        event = json.loads(line)
        if (not isinstance(event, dict) or event.get("schema_version") != "gpuwm.run-plan.event.v1"
                or type(event.get("sequence")) is not int or event["sequence"] <= last_sequence):
            raise ValueError("Native status event identity or sequence is invalid")
        last_sequence = event["sequence"]
        emitted = event.get("emitted_unix_ms")
        if type(emitted) is not int or emitted < started:
            continue
        observed_events.append(event)
        tag = event.get("event")
        phase = event.get("phase")
        if (isinstance(phase, str) and phase and len(phase) <= 160
                and emitted >= result.get("phase_updated_unix_ms", started)):
            result["phase"] = phase
            result["phase_updated_unix_ms"] = emitted
        if tag in ("stage_started", "stage_finished", "failed") and isinstance(event.get("stage"), str):
            result["stage"] = event["stage"][:160]
        if tag == "output_committed" and isinstance(event.get("valid_time"), str):
            _timestamp(event["valid_time"])
            result["valid_time"] = event["valid_time"]
            if type(event.get("domain")) is int and 1 <= event["domain"] <= 999:
                outputs[event["domain"]] = event
        if tag == "model_progress":
            elapsed = event.get("model_seconds")
            if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
                if latest_model is not None and elapsed < latest_model["model_seconds"]:
                    raise ValueError("Native status model clock moved backward")
                latest_model = event
                result["model_elapsed_seconds"] = max(elapsed, result.get("model_elapsed_seconds", 0))
        if tag == "failed" and isinstance(event.get("message"), str):
            message = " ".join(event["message"].split())
            if message:
                result["error"] = message[:1600]
        render_summary = event.get("render_summary")
        if render_summary is None and isinstance(event.get("summary"), dict):
            render_summary = event["summary"].get("render_summary")
        if isinstance(render_summary, dict) and render_summary.get("schema") == "gpuwm.render-summary.v1":
            try:
                summary_size = len(_encoded(render_summary))
            except (ValueError, TypeError):
                summary_size = 64 * 1024 + 1
            if summary_size <= 64 * 1024:
                result["render_summary"] = render_summary
        if tag == "completed":
            result["stage"] = "completed"
    render_summary = result.get("render_summary")
    if isinstance(render_summary, dict) and isinstance(render_summary.get("summary_path"), str):
        try:
            from gpuwm.render_receipts import merge_recorded_summary
            summary_path = _inside(render_summary["summary_path"], root)
            merged = merge_recorded_summary(summary_path.parent, render_summary)
            if len(_encoded(merged)) <= 60 * 1024:
                result["render_summary"] = merged
        except (OSError, ValueError, KeyError, TypeError):
            # The native result remains valid if an optional legacy receipt is
            # absent or stale. Never discard current clocks for that reason.
            pass
    progress = _forecast_progress(record, manifest, manifest_bytes, root, latest_model,
                                  outputs, heartbeat, result.get("phase", result.get("stage")))
    if progress is not None:
        result["progress"] = progress
        result["pipeline_progress"] = _pipeline_progress(observed_events, result, state, started)
    return result


def stream(request, workspace, output):
    """Fixed binary operation; selectors name a known committed event, not a path."""
    fields = {"schema", "action", "workspace", "job", "domain", "sequence", "expected_frame_sha256",
              "expected_commit_sha256", "expected_manifest_sha256"}
    if (not isinstance(request, dict) or set(request) != fields
            or request.get("schema") != "gpuwm.remote.request.v1" or request.get("action") != "stream-artifact"
            or type(request.get("sequence")) is not int or request["sequence"] <= 0
            or any(not isinstance(request.get(key), str) or not HEX.fullmatch(request[key])
                   for key in ("expected_frame_sha256", "expected_commit_sha256", "expected_manifest_sha256"))):
        raise ValueError("Invalid committed artifact stream binding")
    value = catalog(request, workspace, sequence=request["sequence"])
    if value["waiting"] or len(value["frames"]) != 1:
        raise ValueError("The selected committed frame is no longer available")
    frame = value["frames"][0]
    if (frame["sha256"] != request["expected_frame_sha256"]
            or frame["commit"]["sha256"] != request["expected_commit_sha256"]
            or value["run_manifest"]["sha256"] != request["expected_manifest_sha256"]):
        raise ValueError("Committed artifact authority changed since discovery; refresh the frame")
    path, digest, copied = Path(frame["remote_path"]), hashlib.sha256(), 0
    before = _stamp(path)
    with path.open("rb") as source:
        while block := source.read(min(1024 * 1024, frame["size_bytes"] + 1 - copied)):
            copied += len(block)
            if copied > frame["size_bytes"]:
                raise ValueError("Committed frame grew during transfer")
            digest.update(block)
            output.write(block)
    output.flush()
    if copied != frame["size_bytes"] or digest.hexdigest() != frame["sha256"] or _stamp(path) != before:
        raise ValueError("Committed frame changed during transfer; incomplete local bytes must be discarded")


def stream_main():
    from gpuwm import remote_worker as rw
    try:
        if sys.platform != "linux":
            raise ValueError("remote artifact streams require Linux")
        payload = sys.stdin.buffer.read(rw.MAX_BYTES + 1)
        if len(payload) > rw.MAX_BYTES:
            raise ValueError("artifact stream request exceeds 128 KiB")
        request = json.loads(payload)
        if not isinstance(request, dict):
            raise ValueError("artifact stream request must be a JSON object")
        stream(request, rw._workspace(request), sys.stdout.buffer)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("remote artifact: " + str(error)[:4000], file=sys.stderr)
        return 2


def _download(command, request, path, frame, *, timeout=180):
    """One bounded SSH stream to a private temporary file; commit after SHA/exit."""
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".part")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    errors, failure, finished = bytearray(), [], threading.Event()

    def copy():
        try:
            digest, copied = hashlib.sha256(), 0
            with temporary.open("xb") as destination:
                while block := process.stdout.read(1024 * 1024):
                    copied += len(block)
                    if copied > frame["size_bytes"]:
                        raise ValueError("SSH artifact stream exceeds its committed byte length")
                    digest.update(block)
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
            if copied != frame["size_bytes"] or digest.hexdigest() != frame["sha256"]:
                raise ValueError("SSH artifact bytes do not match the committed frame SHA-256 and size")
        except Exception as error:
            failure.append(error)
        finally:
            process.stdout.close()
            finished.set()

    def diagnostics():
        while block := process.stderr.read(4096):
            errors.extend(block[:max(0, 16384 - len(errors))])
        process.stderr.close()

    readers = [threading.Thread(target=copy, daemon=True), threading.Thread(target=diagnostics, daemon=True)]
    for reader in readers:
        reader.start()
    try:
        process.stdin.write(_encoded(request) + b"\n")
        process.stdin.close()
        deadline = time.monotonic() + timeout
        while process.poll() is None or not finished.is_set():
            if failure:
                raise ValueError(str(failure[0]))
            if time.monotonic() >= deadline:
                raise ValueError("Committed frame transfer timed out; refresh to retry")
            time.sleep(.02)
        for reader in readers:
            reader.join(timeout=2)
        if failure:
            raise ValueError(str(failure[0]))
        if process.returncode != 0 or any(reader.is_alive() for reader in readers):
            raise ValueError("SSH artifact transfer failed: " + errors.decode("utf-8", errors="replace")[:2000])
        if path.exists():
            if path.is_symlink() or path.stat().st_size != frame["size_bytes"] or _file_sha(path) != frame["sha256"]:
                raise ValueError("Existing local artifact object has different bytes")
        else:
            os.replace(temporary, path)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for reader in readers:
            reader.join(timeout=2)
        temporary.unlink(missing_ok=True)


def sync(args, command, stream_command):
    from gpuwm.remote_cli import _transport
    domain = _domain(args.domain)
    request = {"schema": "gpuwm.remote.request.v1", "action": "artifacts", "workspace": args.workspace,
               "job": args.job, "domain": domain}
    if getattr(args, "sequence", None) is not None:
        request["sequence"] = _sequence(args.sequence)
    reply = _transport(command, request, timeout=120)
    if not reply["ok"]:
        raise ValueError(reply["error"]["message"])
    value = reply.get("artifacts")
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("job_id") != args.job:
        raise ValueError("Node returned artifact metadata for a different job")
    if value.get("waiting") is True:
        return {"artifacts": value, "transferred_bytes": 0}
    frames = value.get("frames")
    if not isinstance(frames, list) or len(frames) != 1:
        raise ValueError("Node artifact response must name one selected-domain frame")
    frame = frames[0]
    if (frame.get("domain") != domain or type(frame.get("size_bytes")) is not int
            or not 0 < frame["size_bytes"] <= MAX_FRAME or not HEX.fullmatch(str(frame.get("sha256", "")))):
        raise ValueError("Node artifact size, domain or SHA binding is invalid")
    from gpuwm.remote_artifact_cache import Cache, CacheRecovery
    stream_request = {**request, "action": "stream-artifact", "sequence": frame["commit"]["sequence"],
                      "expected_frame_sha256": frame["sha256"], "expected_commit_sha256": frame["commit"]["sha256"],
                      "expected_manifest_sha256": value["run_manifest"]["sha256"]}
    try:
        with Cache(args.cache_root, target_bytes=MAX_CACHE, max_frame_bytes=MAX_FRAME,
                   reader_leases=getattr(args, "reader_leases", False)) as cache:
            path, transferred, cache_status = cache.obtain(frame, lambda path: _download(stream_command, stream_request, path, frame))
    except CacheRecovery as recovery:
        value.update(waiting=True, frames=[])
        return {"artifacts": value, "transferred_bytes": 0,
                "cache_recovery": {"schema": "arwen.artifact-cache-recovery.v1", "sha256": recovery.digest,
                                   "reason": "corrupt_retained_object"}}
    frame["path"] = str(path)
    return {"artifacts": value, "transferred_bytes": transferred, "cache": cache_status}


def index(args, command):
    """Metadata-only timeline page from the same native commit authority."""
    from gpuwm.remote_cli import _transport
    domain = _domain(args.domain)
    request = {"schema": "gpuwm.remote.request.v1", "action": "artifact-index", "workspace": args.workspace,
               "job": args.job, "domain": domain, "after_sequence": _sequence(args.after_sequence, cursor=True)}
    reply = _transport(command, request, timeout=120)
    if not reply["ok"]:
        raise ValueError(reply["error"]["message"])
    value = reply.get("artifact_index")
    if (not isinstance(value, dict) or value.get("schema") != INDEX_SCHEMA or value.get("job_id") != args.job
            or value.get("domain") != domain or not isinstance(value.get("entries"), list)
            or len(value["entries"]) > MAX_INDEX_PAGE):
        raise ValueError("Node returned an invalid artifact timeline page for this job and domain")
    previous = args.after_sequence
    for entry in value["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"sequence", "domain", "valid_time"}:
            raise ValueError("Node timeline entry contains unsupported metadata")
        sequence = _sequence(entry["sequence"])
        if entry["domain"] != domain or sequence <= previous:
            raise ValueError("Node timeline entries do not have increasing selected-domain sequences")
        _timestamp(entry["valid_time"])
        previous = sequence
    cursor = value.get("next_after_sequence")
    if cursor is not None and (not value["entries"] or _sequence(cursor) != previous):
        raise ValueError("Node timeline page cursor does not match its last committed entry")
    latest = value.get("latest_sequence")
    if type(value.get("waiting")) is not bool:
        raise ValueError("Node timeline is missing its exact readiness state")
    if latest is None:
        if value["waiting"] is not True or value["entries"]:
            raise ValueError("Node timeline readiness disagrees with its committed entries")
    else:
        _sequence(latest)
        if (value["waiting"] is not False or (value["entries"] and latest < previous)
                or (cursor is not None and latest <= cursor)
                or (cursor is None and value["entries"] and latest != previous)
                or (not value["entries"] and latest > args.after_sequence)):
            raise ValueError("Node timeline pagination does not account for its latest committed time")
    return {"artifact_index": _bounded(value)}
