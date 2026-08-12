"""Chunked restart-extend orchestration for an uploading forecast cycle.

``gpuwm stream PLAN.toml`` selects the latest runtime cycle, waits for each
successive forcing hour, extends the immutable preparation, and runs one
sealed tree leg from the prior leg's checkpoint.  A bounded ``cycle_count``
can keep that process alive for exact hourly successor cycles.  It deliberately
does not mutate forcing in a live model process.  Production fetch,
preparation, hierarchy, forecast, and checkpoint writers remain the
authorities; this module publishes only an atomic controller chain which
hashes their artifacts.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request

from gpuwm import fetch_guard
from gpuwm.experiment import load_experiment
from gpuwm.fetch import (
    FETCH_MANIFEST_SCHEMA,
    HRRR_DEFAULT_MODE,
    HRRR_WAIT_POLL_SECONDS,
    Area,
    check_prior_request,
    fetch_hrrr,
    hrrr_object_url,
    parse_area,
    parse_cycle,
    resolve_fetch_engine,
    resolve_latest_cycle,
    sha256_file,
)
from gpuwm.hrrr_forecast import hrrr_cycle_horizon
from gpuwm.ingest.hrrr_target import load_hrrr_target_domain
from gpuwm.io.restart import read_restart_header
from gpuwm.namelist_import import parse_namelist
from gpuwm.nomads_governor import paced_urlopen


PLAN_SCHEMA = "gpuwm-stream-plan-v1"
CHAIN_SCHEMA = "gpuwm-stream-chain-v1"
LINK_SCHEMA = "gpuwm-stream-chain-link-v1"
ACTIVE_SCHEMA = "gpuwm-stream-active-cycle-v1"
PROGRAM_SCHEMA = "gpuwm-stream-multicycle-chain-v1"

_PLAN_TOP_KEYS = frozenset({"schema", "stream", "fetch", "prepare", "run"})
_STREAM_KEYS = frozenset({
    "work_root", "cycle", "cycle_count", "target_lead", "poll_seconds",
    "wait_timeout_seconds",
})
_FETCH_KEYS = frozenset({"area", "cache_dir"})
_PREPARE_KEYS = frozenset({
    "experiment_config", "domain_spec", "wps_namelist", "namelist_input",
    "stock_wrf_namelist_input", "geog_root", "physics_profile",
    "pipeline_workers", "prepare_workers", "child_workers",
    "preprocess_backend", "preprocess_workers", "cpu_preprocess_bridge",
    "acknowledgements",
})
_RUN_KEYS = frozenset({
    "io_mode", "health_debug", "gpu_uuid", "allow_shared_gpu",
})
_HEX = frozenset("0123456789abcdef")
_USER_AGENT = "gpuwm-stream/1"
_STREAM_HRRR_TRANSPORT = "s3"
_PINNED_PHYSICS_REGISTRY_ENV = "GPUWM_PINNED_PHYSICS_REGISTRY"
_PINNED_PHYSICS_REGISTRY_SHA256_ENV = \
    "GPUWM_PINNED_PHYSICS_REGISTRY_SHA256"
# Capacity is reserved before future leads exist, so their byte sizes cannot
# honestly be extrapolated from f000/f001.  This is an explicit enforced
# safety envelope for the sum of HRRR wrfnat + wrfprs full objects per hour.
# A later observation above it refuses before fetch and requires a new
# software contract rather than silently invalidating the initial reserve.
_SOURCE_HOUR_RESERVATION_BYTES = 8 * 1024 ** 3
_GENERATION_STORAGE_SAFETY_FACTOR = 2


@dataclasses.dataclass(frozen=True)
class StreamPlan:
    path: Path
    sha256: str
    identity_sha256: str
    work_root: Path
    cycle: str
    cycle_count: int
    target_lead: int
    poll_seconds: float
    wait_timeout_seconds: float
    area: Area | None
    cache_dir: Path | None
    experiment_config: Path
    domain_spec: Path
    domain_identity_sha256: str
    wps_namelist: Path
    namelist_input: Path
    stock_wrf_namelist_input: Path
    geog_root: Path
    physics_profile: str
    pipeline_workers: int
    prepare_workers: int | None
    child_workers: int
    preprocess_backend: str | None
    preprocess_workers: int | None
    cpu_preprocess_bridge: Path | None
    acknowledgements: tuple[str, ...]
    io_mode: str
    health_debug: bool
    gpu_uuid: str | None
    allow_shared_gpu: bool
    experiment: object
    authority_sources: tuple[tuple[str, Path, str], ...]
    geog_manifest_path: Path
    geog_manifest_sha256: str | None
    authority_manifest: Path | None = None


def _strict_table(raw, name: str, keys: frozenset[str], *, required=True):
    value = raw.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    unknown = sorted(set(value) - keys)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in [{name}]; known keys: "
            f"{sorted(keys)}")
    return value


def _finite_positive(value, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _worker(value, label: str, *, optional=False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 33):
        raise ValueError(f"{label} must be an integer from 1 through 32")
    return value


def _path(base: Path, value, label: str, *, directory=False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    result = Path(value).expanduser()
    if not result.is_absolute():
        result = base / result
    result = result.resolve()
    exists = result.is_dir() if directory else result.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} does not exist: {result}")
    return result


def _optional_path(base: Path, value, label: str, *, directory=False):
    if value is None:
        return None
    return _path(base, value, label, directory=directory)


def load_stream_plan(path: str | Path) -> StreamPlan:
    """Load one strict TOML plan and validate all production inputs."""
    if (os.environ.get(_PINNED_PHYSICS_REGISTRY_ENV) is not None
            or os.environ.get(
                _PINNED_PHYSICS_REGISTRY_SHA256_ENV) is not None):
        raise ValueError(
            "pinned physics-registry environment variables are reserved "
            "for stream child processes and must not be set at controller "
            "plan load")
    plan_path = Path(path).expanduser().resolve()
    payload = plan_path.read_bytes()
    raw = tomllib.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("stream plan must contain one TOML document")
    unknown = sorted(set(raw) - _PLAN_TOP_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level stream plan key(s): {unknown}")
    if raw.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"stream plan schema must be {PLAN_SCHEMA!r}")
    stream = _strict_table(raw, "stream", _STREAM_KEYS)
    fetch = _strict_table(raw, "fetch", _FETCH_KEYS, required=False)
    prepare = _strict_table(raw, "prepare", _PREPARE_KEYS)
    run = _strict_table(raw, "run", _RUN_KEYS, required=False)
    base = plan_path.parent

    required_stream = sorted({"work_root"} - set(stream))
    required_prepare = sorted({
        "experiment_config", "domain_spec", "wps_namelist",
        "namelist_input", "stock_wrf_namelist_input", "geog_root",
        "physics_profile",
    } - set(prepare))
    if required_stream or required_prepare:
        raise ValueError(
            "stream plan is missing required keys: "
            f"[stream] {required_stream}, [prepare] {required_prepare}")

    work_root = Path(stream["work_root"]).expanduser()
    if not work_root.is_absolute():
        work_root = base / work_root
    work_root = work_root.resolve()
    cycle = stream.get("cycle", "latest")
    if not isinstance(cycle, str):
        raise ValueError("[stream] cycle must be 'latest' or YYYY-MM-DDTHH")
    if cycle != "latest":
        parse_cycle(cycle, "hrrr")
    # One hour is deliberately conservative.  Every retained lead owns a
    # complete immutable root preparation, hierarchy, run, and checkpoint
    # generation; silently expanding an omitted target to f018/f048 would be
    # an operational disk/time bomb.
    target = stream.get("target_lead", 1)
    if (isinstance(target, bool)
            or not isinstance(target, int) or target < 1):
        raise ValueError("[stream] target_lead must be a positive integer")
    cycle_count = stream.get("cycle_count", 1)
    if (isinstance(cycle_count, bool) or not isinstance(cycle_count, int)
            or cycle_count < 1):
        raise ValueError("[stream] cycle_count must be a positive integer")
    poll = _finite_positive(
        stream.get("poll_seconds", HRRR_WAIT_POLL_SECONDS),
        "[stream] poll_seconds")
    timeout = _finite_positive(
        stream.get("wait_timeout_seconds", 2 * 3600),
        "[stream] wait_timeout_seconds")

    raw_area = fetch.get("area")
    if raw_area is None:
        area = None
    elif (isinstance(raw_area, list) and len(raw_area) == 4
          and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                  for item in raw_area)):
        area = parse_area(",".join(str(item) for item in raw_area))
    elif isinstance(raw_area, str):
        area = parse_area(raw_area)
    else:
        raise ValueError("[fetch] area must be four numeric corners or a string")

    experiment_config = _path(
        base, prepare["experiment_config"], "[prepare] experiment_config")
    experiment = load_experiment(experiment_config)
    if len(experiment.domains) < 2:
        raise ValueError(
            "gpuwm stream uses the checkpoint-writing prepared tree runner; "
            "the experiment must contain at least two domains")
    if float(experiment.restart_interval_s) != 3600.0:
        raise ValueError(
            "[experiment] restart_interval_s must be 3600 for hourly "
            "sealed stream legs")
    first_leg_end = experiment.start_time + timedelta(hours=1)
    late_starts = [
        domain.grid_id for domain in experiment.domains
        if experiment.domain_start_time(domain.grid_id) >= first_leg_end
    ]
    if late_starts:
        raise ValueError(
            "every domain must start before the first one-hour stream leg "
            "ends; checkpoint f001 cannot carry a never-started child "
            f"reliably (late grid ids: {late_starts})")

    acknowledgements = prepare.get("acknowledgements", [])
    if (not isinstance(acknowledgements, list)
            or any(not isinstance(item, str) or not item.strip()
                   for item in acknowledgements)):
        raise ValueError(
            "[prepare] acknowledgements must be an array of non-empty strings")
    backend_name = prepare.get("preprocess_backend")
    if backend_name not in (None, "cuda", "cpu", "auto"):
        raise ValueError(
            "[prepare] preprocess_backend must be cuda, cpu, or auto")
    io_mode = run.get("io_mode", "history")
    if io_mode not in ("history", "none"):
        raise ValueError("[run] io_mode must be 'history' or 'none'")
    health_debug = run.get("health_debug", False)
    if not isinstance(health_debug, bool):
        raise ValueError("[run] health_debug must be true or false")
    gpu_uuid = run.get("gpu_uuid")
    if (gpu_uuid is not None
            and (not isinstance(gpu_uuid, str)
                 or not gpu_uuid.startswith("GPU-") or not gpu_uuid.strip())):
        raise ValueError("[run] gpu_uuid must be an nvidia-smi GPU-* UUID")
    allow_shared_gpu = run.get("allow_shared_gpu", False)
    if not isinstance(allow_shared_gpu, bool):
        raise ValueError("[run] allow_shared_gpu must be true or false")
    profile = prepare["physics_profile"]
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("[prepare] physics_profile must be a non-empty id")

    wps = _path(base, prepare["wps_namelist"], "[prepare] wps_namelist")
    native = _path(base, prepare["namelist_input"], "[prepare] namelist_input")
    stock = _path(
        base, prepare["stock_wrf_namelist_input"],
        "[prepare] stock_wrf_namelist_input")
    max_dom = parse_namelist(native).get("domains", {}).get("max_dom", [])
    if not max_dom or int(max_dom[0]) != len(experiment.domains):
        raise ValueError(
            "[prepare] namelist_input max_dom must match the experiment "
            f"domain count {len(experiment.domains)}")
    stock_dom = parse_namelist(stock).get("domains", {}).get("max_dom", [])
    if not stock_dom or int(stock_dom[0]) != len(experiment.domains):
        raise ValueError(
            "[prepare] stock_wrf_namelist_input max_dom must match the "
            f"experiment domain count {len(experiment.domains)}")

    raw_sha256 = hashlib.sha256(payload).hexdigest()
    cache_dir = _optional_path(
        base, fetch.get("cache_dir"), "[fetch] cache_dir", directory=True)
    domain_spec = _path(
        base, prepare["domain_spec"], "[prepare] domain_spec")
    domain_identity_sha256 = load_hrrr_target_domain(
        domain_spec).identity_sha256()
    geog_root = _path(
        base, prepare["geog_root"], "[prepare] geog_root", directory=True)
    cpu_bridge = _optional_path(
        base, prepare.get("cpu_preprocess_bridge"),
        "[prepare] cpu_preprocess_bridge")
    # Raw bytes alone are insufficient when a copied plan retains relative
    # spellings but resolves them against a different directory.  Bind the
    # effective paths and immutable file bytes while keeping ``sha256`` as
    # the honest digest of PLAN.toml itself.
    physics_registry = Path(__file__).with_name("physics_registry_v2.json")
    geog_manifest = geog_root / "geog-fetch-manifest.json"
    geog_manifest_sha256 = (
        sha256_file(geog_manifest) if geog_manifest.is_file() else None)
    identity_inputs = [
        ("plan", plan_path),
        ("experiment_config", experiment_config),
        ("domain_spec", domain_spec),
        ("wps_namelist", wps),
        ("namelist_input", native),
        ("stock_wrf_namelist_input", stock),
        ("physics_registry", physics_registry),
    ]
    if cpu_bridge is not None:
        identity_inputs.append(("cpu_preprocess_bridge", cpu_bridge))
    identity_payload = {
        "schema": "gpuwm-stream-effective-plan-v1",
        "plan_sha256": raw_sha256,
        "work_root": str(work_root),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "geog_root": str(geog_root),
        "geog_fetch_manifest": {
            "path": str(geog_manifest),
            "sha256": geog_manifest_sha256,
        },
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in identity_inputs
        },
    }
    identity_sha256 = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()

    return StreamPlan(
        path=plan_path,
        sha256=raw_sha256,
        identity_sha256=identity_sha256,
        work_root=work_root,
        cycle=cycle,
        cycle_count=cycle_count,
        target_lead=target,
        poll_seconds=poll,
        wait_timeout_seconds=timeout,
        area=area,
        cache_dir=cache_dir,
        experiment_config=experiment_config,
        domain_spec=domain_spec,
        domain_identity_sha256=domain_identity_sha256,
        wps_namelist=wps,
        namelist_input=native,
        stock_wrf_namelist_input=stock,
        geog_root=geog_root,
        physics_profile=profile,
        pipeline_workers=_worker(
            prepare.get("pipeline_workers", 8),
            "[prepare] pipeline_workers"),
        prepare_workers=_worker(
            prepare.get("prepare_workers"),
            "[prepare] prepare_workers", optional=True),
        child_workers=_worker(
            prepare.get("child_workers", 8),
            "[prepare] child_workers"),
        preprocess_backend=backend_name,
        preprocess_workers=_worker(
            prepare.get("preprocess_workers"),
            "[prepare] preprocess_workers", optional=True),
        cpu_preprocess_bridge=cpu_bridge,
        acknowledgements=tuple(acknowledgements),
        io_mode=io_mode,
        health_debug=health_debug,
        gpu_uuid=gpu_uuid,
        allow_shared_gpu=allow_shared_gpu,
        experiment=experiment,
        authority_sources=tuple(
            (label, path, sha256_file(path))
            for label, path in identity_inputs),
        geog_manifest_path=geog_manifest,
        geog_manifest_sha256=geog_manifest_sha256,
    )


_ASSIGNMENT = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<equal>\s*=\s*)(?P<value>.*?)(?P<comment>\s*!.*)?$")


def _render_namelist_values(values) -> str:
    def one(value):
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        if isinstance(value, bool):
            return ".true." if value else ".false."
        return str(value)
    return ", ".join(one(value) for value in values) + ","


def _rewrite_namelist(text: str, updates: Mapping[str, Mapping[str, list]]) -> str:
    """Replace/insert simple WRF namelist assignments by section."""
    lines = text.splitlines()
    seen = {section: set() for section in updates}
    current = None
    out = []
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("&"):
            current = stripped[1:].strip().lower()
            out.append(raw)
            continue
        if stripped == "/":
            if current in updates:
                for key, values in updates[current].items():
                    if key not in seen[current]:
                        out.append(f" {key} = {_render_namelist_values(values)}")
            current = None
            out.append(raw)
            continue
        match = _ASSIGNMENT.match(raw)
        if match and current in updates:
            key = match.group("key").lower()
            if key in updates[current]:
                seen[current].add(key)
                raw = (f"{match.group('indent')}{match.group('key')}"
                       f"{match.group('equal')}"
                       f"{_render_namelist_values(updates[current][key])}"
                       f"{match.group('comment') or ''}")
        out.append(raw)
    missing_sections = sorted(section for section in updates if section not in seen)
    if missing_sections:
        raise ValueError(f"namelist lacks section(s) {missing_sections}")
    return "\n".join(out) + "\n"


def _materialize_input_namelist(
        template: Path, destination: Path, *, cycle: datetime, lead: int,
        domain_starts: list[datetime]) -> None:
    end = cycle + timedelta(hours=lead)
    max_dom = len(domain_starts)
    if not max_dom:
        raise ValueError("stream namelist materialization needs a domain")
    def repeated(value):
        return [value] * max_dom
    updates = {
        "time_control": {
            "run_days": [0], "run_hours": [lead],
            "run_minutes": [0], "run_seconds": [0],
            "start_year": [value.year for value in domain_starts],
            "start_month": [value.month for value in domain_starts],
            "start_day": [value.day for value in domain_starts],
            "start_hour": [value.hour for value in domain_starts],
            "start_minute": [value.minute for value in domain_starts],
            "start_second": [value.second for value in domain_starts],
            "end_year": repeated(end.year),
            "end_month": repeated(end.month),
            "end_day": repeated(end.day),
            "end_hour": repeated(end.hour),
            "end_minute": repeated(end.minute),
            "end_second": repeated(end.second),
        }
    }
    _publish_exact(
        destination,
        _rewrite_namelist(template.read_text(encoding="utf-8"), updates)
        .encode("utf-8"))


_TOML_ASSIGNMENT = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_-]*)"
    r"(?P<equal>\s*=\s*)(?P<value>.*?)(?P<comment>\s+#.*)?$")


def _toml_datetime(value: datetime) -> str:
    # ExperimentConfig intentionally owns offset-free UTC model instants.
    # A trailing ``Z`` makes TOML produce an aware datetime which the typed
    # loader correctly refuses, even though stream-cycle receipt timestamps
    # elsewhere are RFC 3339 UTC strings.
    return value.strftime("%Y-%m-%dT%H:%M:%S")


def _materialize_experiment(plan: StreamPlan, destination: Path, *,
                            cycle: datetime, lead: int) -> None:
    lines = plan.experiment_config.read_text(encoding="utf-8").splitlines()
    delta = cycle - plan.experiment.start_time
    domain_starts = [
        plan.experiment.domain_start_time(domain.grid_id) + delta
        for domain in plan.experiment.domains
    ]
    section = None
    domain_index = -1
    replaced = {"start_time": 0, "run_seconds": 0}
    out = []
    for raw in lines:
        stripped = raw.strip()
        if stripped == "[[domain]]":
            section = "domain"
            domain_index += 1
        elif stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]").strip()
        match = _TOML_ASSIGNMENT.match(raw)
        if match:
            key = match.group("key")
            replacement = None
            if section == "experiment" and key == "start_time":
                replacement = _toml_datetime(cycle)
                replaced["start_time"] += 1
            elif section == "experiment" and key == "run_seconds":
                replacement = str(lead * 3600)
                replaced["run_seconds"] += 1
            elif section == "domain" and key == "start_time":
                replacement = _toml_datetime(domain_starts[domain_index])
            if replacement is not None:
                raw = (f"{match.group('indent')}{match.group('key')}"
                       f"{match.group('equal')}{replacement}"
                       f"{match.group('comment') or ''}")
        out.append(raw)
    if replaced != {"start_time": 1, "run_seconds": 1}:
        raise ValueError(
            "experiment template must contain exactly one [experiment] "
            "start_time and run_seconds assignment")
    payload = ("\n".join(out) + "\n").encode("utf-8")
    _publish_exact(destination, payload)
    materialized = load_experiment(destination)
    if (materialized.start_time != cycle
            or float(materialized.run_seconds) != lead * 3600.0):
        raise RuntimeError("materialized experiment timing failed validation")


def _publish_exact(path: Path, payload: bytes) -> Path:
    """Create one immutable controller input, or adopt identical bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"refusing to replace changed stream artifact {path}")
        return path
    return fetch_guard.atomic_write_bytes(path, payload, tag="stream")


def _verify_authority_sources(plan: StreamPlan) -> None:
    for label, path, expected in plan.authority_sources:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(
                f"stream authority {label} changed after plan load: {path}")
    manifest = plan.geog_manifest_path
    observed = sha256_file(manifest) if manifest.is_file() else None
    if observed != plan.geog_manifest_sha256:
        raise ValueError(
            f"stream geog authority manifest changed after plan load: "
            f"{manifest}")


def _verify_pinned_authorities(plan: StreamPlan) -> None:
    if plan.authority_manifest is None:
        raise ValueError("stream plan authorities have not been pinned")
    payload = _json(plan.authority_manifest, "stream authority manifest")
    if (payload.get("schema") != "gpuwm-stream-authorities-v1"
            or payload.get("plan_identity_sha256") != plan.identity_sha256
            or not isinstance(payload.get("inputs"), list)):
        raise ValueError(
            f"stream authority manifest identity mismatch: "
            f"{plan.authority_manifest}")
    for item in payload["inputs"]:
        if not isinstance(item, dict):
            raise ValueError("stream authority manifest input is malformed")
        snapshot = Path(str(item.get("snapshot", "")))
        expected = item.get("sha256")
        if (not snapshot.is_file() or not isinstance(expected, str)
                or sha256_file(snapshot) != expected):
            raise ValueError(
                f"pinned stream authority changed or disappeared: "
                f"{item.get('role')} {snapshot}")
    geog = payload.get("geog")
    if not isinstance(geog, dict) or geog.get("root") != str(plan.geog_root):
        raise ValueError("stream authority geog identity is malformed")
    observed = (sha256_file(plan.geog_manifest_path)
                if plan.geog_manifest_path.is_file() else None)
    if observed != geog.get("manifest_sha256"):
        raise ValueError(
            f"stream geog authority manifest changed: "
            f"{plan.geog_manifest_path}")
def _pin_plan_authorities(plan: StreamPlan) -> StreamPlan:
    """Snapshot every small mutable plan input before controller use."""
    if plan.authority_manifest is not None:
        _verify_pinned_authorities(plan)
        return plan
    _verify_authority_sources(plan)
    root = plan.work_root / "plan-authorities" / plan.identity_sha256
    pinned = {}
    rows = []
    for label, source, expected in plan.authority_sources:
        suffix = "".join(source.suffixes) or ".bin"
        snapshot = _publish_exact(root / f"{label}{suffix}", source.read_bytes())
        if sha256_file(snapshot) != expected:
            raise ValueError(f"pinned stream authority hash mismatch: {snapshot}")
        pinned[label] = snapshot
        rows.append({
            "role": label,
            "original": str(source),
            "snapshot": str(snapshot.resolve()),
            "sha256": expected,
        })
    payload = {
        "schema": "gpuwm-stream-authorities-v1",
        "plan_identity_sha256": plan.identity_sha256,
        "raw_plan_sha256": plan.sha256,
        "inputs": rows,
        "geog": {
            "root": str(plan.geog_root),
            "manifest": str(plan.geog_manifest_path),
            "manifest_sha256": plan.geog_manifest_sha256,
        },
    }
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False) + "\n").encode("utf-8")
    authority_manifest = _publish_exact(root / "manifest.json", encoded)
    result = dataclasses.replace(
        plan,
        experiment_config=pinned["experiment_config"],
        domain_spec=pinned["domain_spec"],
        wps_namelist=pinned["wps_namelist"],
        namelist_input=pinned["namelist_input"],
        stock_wrf_namelist_input=pinned["stock_wrf_namelist_input"],
        cpu_preprocess_bridge=(
            None if plan.cpu_preprocess_bridge is None
            else pinned["cpu_preprocess_bridge"]),
        experiment=load_experiment(pinned["experiment_config"]),
        authority_manifest=authority_manifest,
    )
    _verify_pinned_authorities(result)
    return result


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def _verify_sha256_manifest(root: Path) -> str:
    """Verify a strict relative SHA256SUMS inventory and return its digest."""
    root = Path(root).resolve()
    manifest = root / "SHA256SUMS"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"unreadable SHA256SUMS manifest: {manifest}") from exc
    if not lines:
        raise ValueError(f"empty SHA256SUMS manifest: {manifest}")
    names = set()
    for line in lines:
        if "  " not in line:
            raise ValueError(f"malformed SHA256SUMS row in {manifest}")
        digest, name = line.split("  ", 1)
        relative = Path(name)
        if (len(digest) != 64 or any(char not in _HEX for char in digest)
                or not name or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or name in names):
            raise ValueError(f"unsafe or duplicate SHA256SUMS row in {manifest}")
        names.add(name)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"SHA256SUMS path escapes its root: {name}") from exc
        if not candidate.is_file() or sha256_file(candidate) != digest:
            raise ValueError(
                f"SHA256SUMS payload changed or disappeared: {candidate}")
    return sha256_file(manifest)


def _atomic_json(path: Path, payload) -> Path:
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    return fetch_guard.atomic_write_text(path, encoded, tag="stream")


def _publish_json_exact(path: Path, payload) -> Path:
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False) + "\n").encode("utf-8")
    return _publish_exact(path, encoded)


def _claim_work_root(plan: StreamPlan) -> Path:
    """Create/adopt only a work root explicitly owned by this exact plan."""
    root = plan.work_root
    marker = root / ".gpuwm-stream-owner.json"
    expected = {
        "schema": "gpuwm-stream-work-root-owner-v1",
        "status": "OWNED",
        "plan_identity_sha256": plan.identity_sha256,
        "raw_plan_sha256": plan.sha256,
        "work_root": str(root.resolve()),
    }
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        _publish_json_exact(marker, expected)
        return marker
    if not root.is_dir():
        raise ValueError(f"stream work_root is not a directory: {root}")
    if marker.is_file():
        if _json(marker, "stream work-root owner") != expected:
            raise ValueError(
                f"stream work_root belongs to another plan: {root}")
        return marker
    try:
        nonempty = next(root.iterdir(), None) is not None
    except OSError as exc:
        raise ValueError(f"stream work_root is unreadable: {root}") from exc
    if nonempty:
        raise ValueError(
            f"refusing preexisting unowned nonempty stream work_root: {root}")
    _publish_json_exact(marker, expected)
    return marker


def _stage_owner(path: Path, *, plan: StreamPlan, cycle: datetime,
                 lead: int, role: str) -> Path:
    """Reserve one create-only stage name before any child can populate it."""
    path = Path(path)
    marker = path.with_name(path.name + ".gpuwm-stream-owner.json")
    expected = {
        "schema": "gpuwm-stream-stage-owner-v1",
        "status": "RESERVED",
        "plan_identity_sha256": plan.identity_sha256,
        "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
        "lead": lead,
        "role": role,
        "path": str(path.resolve()),
    }
    if path.exists() and not marker.is_file():
        raise ValueError(
            f"refusing preexisting unowned stream stage: {path}")
    _publish_json_exact(marker, expected)
    return marker


def _digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact(path: Path, *, role: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_artifact(item: Mapping) -> None:
    if not isinstance(item, Mapping):
        raise ValueError("chain artifact entry is not an object")
    path = Path(str(item.get("path", "")))
    expected = item.get("sha256")
    if (not path.is_file() or not isinstance(expected, str)
            or sha256_file(path) != expected
            or path.stat().st_size != item.get("bytes")):
        raise ValueError(
            f"chain artifact changed or disappeared: {item.get('role')} {path}")


def _prepared_cache_payload_artifacts(
        cache: Path, *, role_prefix: str) -> list[dict]:
    """Hash raw NPY files; decoded-array hashes do not cover trailing bytes."""
    cache = Path(cache).resolve()
    header = _json(cache / "header.json", "prepared-cache header")
    arrays = header.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError(f"prepared-cache array inventory is absent: {cache}")
    result = []
    filenames = set()
    for index, (key, spec) in enumerate(sorted(arrays.items())):
        filename = spec.get("file") if isinstance(spec, dict) else None
        if (not isinstance(key, str) or not isinstance(filename, str)
                or Path(filename).name != filename
                or filename in filenames):
            raise ValueError(
                f"prepared-cache raw payload inventory is malformed: {cache}")
        filenames.add(filename)
        item = _artifact(
            cache / filename, role=f"{role_prefix}.payload.{index:05d}")
        item["cache_key"] = key
        result.append(item)
    return result


def _directory_file_artifacts(
        root: Path, *, role_prefix: str,
        exclude: tuple[Path, ...] = ()) -> list[dict]:
    """Bind every regular file under a create-only consumed artifact tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact tree does not exist: {root}")
    excluded = {Path(path).resolve() for path in exclude}
    files = sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix())
    result = []
    for index, path in enumerate(path for path in files
                                 if path not in excluded):
        item = _artifact(path, role=f"{role_prefix}.{index:05d}")
        item["relative_path"] = path.relative_to(root).as_posix()
        result.append(item)
    if not result:
        raise ValueError(f"artifact tree contains no bound files: {root}")
    return result


def _checkpoint_set(root: Path, *, role_prefix: str) -> list[dict]:
    """Hash every member of one tree checkpoint generation."""
    root = Path(root).resolve()
    header = read_restart_header(root)
    set_id = header.get("checkpoint_set_id")
    domain_ids = header.get("domain_ids")
    if (not isinstance(set_id, str) or not set_id
            or not isinstance(domain_ids, list) or not domain_ids
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in domain_ids)
            or domain_ids != sorted(set(domain_ids))):
        raise ValueError(f"checkpoint root lacks a complete tree header: {root}")
    found = {}
    for candidate in root.parent.glob(f"gpuwmrst_d*__{set_id}.npz"):
        member = read_restart_header(candidate)
        if member.get("checkpoint_set_id") == set_id:
            grid_id = member.get("grid_id")
            if (isinstance(grid_id, bool) or not isinstance(grid_id, int)
                    or grid_id in found):
                raise ValueError(
                    f"checkpoint set {set_id} has duplicate/malformed "
                    f"grid identity {grid_id!r}")
            found[grid_id] = candidate
    if set(found) != set(domain_ids):
        raise ValueError(
            f"checkpoint set {set_id} is partial: expected {domain_ids}, "
            f"found {sorted(found)}")
    return [
        _artifact(found[grid_id], role=f"{role_prefix}.d{grid_id:02d}")
        for grid_id in sorted(found)
    ]


def _healthy_run_receipt(path: Path) -> tuple[dict, Path]:
    payload = _json(path, "prepared-tree run receipt")
    if payload.get("status") != "PASS":
        raise ValueError(f"prepared-tree run did not PASS: {path}")
    health = payload.get("health")
    if not isinstance(health, dict):
        raise ValueError(f"prepared-tree run receipt lacks health: {path}")
    for phase in ("initial", "final"):
        rows = health.get(phase)
        if (not isinstance(rows, dict) or not rows
                or any(not isinstance(row, dict) or row.get("ok") is not True
                       for row in rows.values())):
            raise ValueError(
                f"prepared-tree {phase} health is not wholly PASS: {path}")
    output = payload.get("output")
    checkpoint = None if not isinstance(output, dict) else output.get(
        "last_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"prepared-tree run wrote no checkpoint: {path}")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = path.parents[1] / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"prepared-tree checkpoint named by {path} is absent: "
            f"{checkpoint_path}")
    return payload, checkpoint_path.resolve()


def _head_resource_metadata(url: str):
    """Anonymous HEAD probe returning evidence used by the watcher/gate."""
    request = Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        # Watcher shutdown joins every worker.  A short finite network bound
        # is therefore part of the cancellation contract: no detached HEAD
        # request can survive controller unwind and publish a late timeline.
        with paced_urlopen(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                return None
            raw_length = response.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else None
            except ValueError:
                length = None
            raw_modified = response.headers.get("Last-Modified")
            modified = None
            if raw_modified:
                try:
                    modified = _iso(parsedate_to_datetime(raw_modified))
                except (TypeError, ValueError, OverflowError):
                    modified = None
            return {
                "url": url,
                "content_length_bytes": length,
                "remote_last_modified": modified,
                "etag": response.headers.get("ETag"),
            }
    except (HTTPError, URLError, OSError):
        return None


class ProductionBackend:
    """Wall-clock, network, fetch, and subprocess boundary for tests."""

    def __init__(self, *, progress=print, probe=None,
                 sleeper=time.sleep, monotonic=time.monotonic,
                 now: Callable[[], datetime] = _utc_now):
        if probe is None:
            probe = _head_resource_metadata
        self.progress = progress
        self.probe = probe
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.now = now
        self._engine = None
        self._command_env = None
        self._gpu = None
        self._allow_shared_gpu = False
        self._pinned_physics_registry = None
        self._pinned_physics_registry_sha256 = None

    @contextmanager
    def gpu_allocation(self, plan: StreamPlan):
        """Use the supervisor's UUID lock namespace and explicit CUDA mask."""
        from gpuwm.supervisor import (
            GPUFileLock, default_lock_path, preflight_exclusive_gpu,
            select_gpu,
        )

        gpu = select_gpu(plan.gpu_uuid)
        lock_path = default_lock_path(gpu.uuid)
        run_id = f"stream-{plan.identity_sha256[:20]}"
        with GPUFileLock(gpu.uuid, path=lock_path, run_id=run_id):
            preflight_exclusive_gpu(
                gpu.uuid, approved_pids={os.getpid()},
                allow_shared_gpu=plan.allow_shared_gpu)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": gpu.uuid,
                "GPUWM_GPU_UUID": gpu.uuid,
                "GPUWM_GPU_DRIVER": gpu.driver_version,
                "GPUWM_GPU_NAME": gpu.name,
            })
            self._command_env = env
            self._gpu = gpu
            self._allow_shared_gpu = plan.allow_shared_gpu
            try:
                yield {
                    "schema": "gpuwm-stream-gpu-allocation-v1",
                    "status": "PASS",
                    "plan_identity_sha256": plan.identity_sha256,
                    "requested_uuid": plan.gpu_uuid,
                    "resolved_uuid": gpu.uuid,
                    "driver_version": gpu.driver_version,
                    "name": gpu.name,
                    "index": gpu.index,
                    "lock_path": str(lock_path.resolve()),
                    "cuda_visible_devices": gpu.uuid,
                    "allow_shared_gpu": plan.allow_shared_gpu,
                }
            finally:
                self._command_env = None
                self._gpu = None

    @staticmethod
    def _normalized_probe(value, url: str):
        if not value:
            return None
        if isinstance(value, Mapping):
            result = dict(value)
            result.setdefault("url", url)
            result.setdefault("content_length_bytes", None)
            result.setdefault("remote_last_modified", None)
            result.setdefault("etag", None)
            return result
        return {
            "url": url,
            "content_length_bytes": None,
            "remote_last_modified": None,
            "etag": None,
        }

    def probe_hour(self, cycle: datetime, lead: int,
                   *, stop_event: threading.Event | None = None):
        """Return a complete four-resource observation, or ``None``."""
        products = {}
        for product in ("wrfnat", "wrfprs"):
            if stop_event is not None and stop_event.is_set():
                return None
            transport = _STREAM_HRRR_TRANSPORT
            url = hrrr_object_url(
                cycle, lead, product, transport=transport)
            object_row = self._normalized_probe(self.probe(url), url)
            if stop_event is not None and stop_event.is_set():
                return None
            index_row = self._normalized_probe(
                self.probe(url + ".idx"), url + ".idx")
            if object_row is not None and index_row is not None:
                products[product] = {
                    "transport": transport,
                    "object_url": url,
                    "index_url": url + ".idx",
                    "object": object_row,
                    "index": index_row,
                }
        if len(products) != 2:
            return None
        observed = self.now()
        observed_iso = _iso(observed)
        resources = [
            row[kind]
            for row in products.values()
            for kind in ("object", "index")
        ]
        for row in resources:
            row["first_observed_at"] = observed_iso
        modified = [row["remote_last_modified"] for row in resources]
        remote_ready = None
        if all(isinstance(value, str) for value in modified):
            remote_ready = max(
                modified,
                key=lambda value: datetime.fromisoformat(
                    value.replace("Z", "+00:00")))
        object_lengths = [
            row["object"].get("content_length_bytes")
            for row in products.values()
        ]
        return {
            "first_observed_at": observed_iso,
            # This is the maximum remote HTTP Last-Modified over both
            # objects and both indexes.  It is metadata from the serving
            # endpoint, not a claim about producer completion/upload time.
            "remote_ready_last_modified_at": remote_ready,
            "object_content_length_bytes": (
                sum(object_lengths)
                if all(isinstance(value, int) and value >= 0
                       for value in object_lengths)
                else None),
            "products": products,
        }

    def resolve_latest(self) -> datetime:
        observed = self.now().astimezone(timezone.utc).replace(tzinfo=None)
        # f00 selects the newest cycle as it begins publishing.  Resolving
        # against the target lead would wait until that lead existed and
        # silently turn the streaming command into a retrospective fetch.
        return resolve_latest_cycle("hrrr", 0, now=observed, probe=self.probe)

    def wait_for_hour(self, cycle: datetime, lead: int, *, timeout_s: float,
                      poll_s: float,
                      stop_event: threading.Event | None = None) -> dict:
        deadline = self.monotonic() + timeout_s
        announced = False
        while True:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("stream availability watcher stopped")
            observed = self.probe_hour(
                cycle, lead, stop_event=stop_event)
            if observed is not None:
                self.progress(
                    f"stream f{lead:03d}: forcing object and index pairs "
                    f"first observed complete at "
                    f"{observed['first_observed_at']}")
                return observed
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"stream timed out after {timeout_s / 60.0:.0f} min "
                    f"waiting for both forcing products and indexes at "
                    f"f{lead:03d}; completed legs remain resumable")
            if not announced:
                self.progress(
                    f"stream f{lead:03d}: forcing is not complete yet; "
                    f"waiting up to {remaining / 60.0:.0f} min")
                announced = True
            delay = min(poll_s, remaining)
            if stop_event is None:
                self.sleeper(delay)
            else:
                stop_event.wait(delay)

    def observe_hour_after_fetch(self, cycle: datetime, lead: int) -> dict:
        """Re-HEAD the exact objects after transfer for drift detection."""
        observed = self.probe_hour(cycle, lead)
        if observed is None:
            raise RuntimeError(
                f"stream f{lead:03d}: forcing object/index disappeared "
                "during the full-file fetch")
        return observed

    def wait_for_next_cycle(self, after: datetime, *, timeout_s: float,
                            poll_s: float) -> datetime:
        expected = after + timedelta(hours=1)
        self.progress(
            f"stream: waiting unattended for next hourly cycle "
            f"{expected:%Y-%m-%dT%H}Z")
        self.wait_for_hour(
            expected, 0, timeout_s=timeout_s, poll_s=poll_s)
        self.progress(
            f"stream: observed next cycle {expected:%Y-%m-%dT%H}Z")
        return expected

    def _ensure_fetch_engine(self):
        if self._engine is None:
            self._engine = resolve_fetch_engine(
                "rust", progress=self.progress)
        return self._engine

    def preflight_fetch(self, _plan: StreamPlan) -> None:
        """Resolve and probe the required Rust fetcher before a long wait."""
        self._ensure_fetch_engine()

    def fetch_prefix(self, plan: StreamPlan, cycle: datetime, lead: int,
                     source_root: Path) -> Path:
        engine, engine_bin = self._ensure_fetch_engine()
        check_prior_request(
            source_root, source="hrrr", cycle=cycle, area=plan.area)
        return fetch_hrrr(
            cycle=cycle, hours=tuple(range(lead + 1)), area=plan.area,
            out=source_root, progress=self.progress,
            transport=_STREAM_HRRR_TRANSPORT,
            wait=True, wait_timeout_s=plan.wait_timeout_seconds,
            probe=self.probe, sleeper=self.sleeper, clock=self.monotonic,
            engine=engine, engine_bin=engine_bin, mode=HRRR_DEFAULT_MODE,
            cache_dir=plan.cache_dir)

    def bind_plan_authorities(self, plan: StreamPlan) -> None:
        """Route every child import through the pinned registry bytes."""
        if plan.authority_manifest is None:
            raise ValueError("cannot bind unpinned stream authorities")
        manifest = _json(
            plan.authority_manifest, "stream authority manifest")
        matches = [
            row for row in manifest.get("inputs", [])
            if isinstance(row, dict) and row.get("role") == "physics_registry"
        ]
        if len(matches) != 1:
            raise ValueError(
                "stream authority manifest lacks one physics registry")
        snapshot = Path(str(matches[0].get("snapshot", ""))).resolve()
        if (not snapshot.is_file()
                or sha256_file(snapshot) != matches[0].get("sha256")):
            raise ValueError("pinned stream physics registry changed")
        self._pinned_physics_registry = snapshot
        self._pinned_physics_registry_sha256 = matches[0]["sha256"]

    def run_command(self, argv: list[str], *, stage: str) -> None:
        if self._gpu is not None:
            from gpuwm.supervisor import preflight_exclusive_gpu
            preflight_exclusive_gpu(
                self._gpu.uuid, approved_pids={os.getpid()},
                allow_shared_gpu=self._allow_shared_gpu)
        self.progress(f"stream {stage}: " + subprocess.list2cmdline(argv))
        env = (os.environ.copy() if self._command_env is None
               else dict(self._command_env))
        if self._pinned_physics_registry is not None:
            env[_PINNED_PHYSICS_REGISTRY_ENV] = str(
                self._pinned_physics_registry)
            env[_PINNED_PHYSICS_REGISTRY_SHA256_ENV] = str(
                self._pinned_physics_registry_sha256)
        subprocess.run(argv, check=True, env=env)


def _availability_path(cycle_root: Path, lead: int) -> Path:
    if lead == 0:
        return cycle_root / "availability" / "f000.json"
    return cycle_root / "legs" / f"f{lead:03d}" / "availability.json"


def _validated_observation(path: Path, *, cycle: datetime, lead: int) -> dict:
    payload = _json(path, "forcing availability observation")
    if (payload.get("schema") != "gpuwm-stream-availability-v2"
            or payload.get("cycle")
            != cycle.strftime("%Y-%m-%dT%H:00:00Z")
            or payload.get("lead") != lead
            or not isinstance(payload.get("first_observed_at"), str)
            or not isinstance(payload.get("products"), dict)
            or set(payload["products"]) != {"wrfnat", "wrfprs"}):
        raise ValueError(f"availability observation identity mismatch: {path}")
    return payload


def _bind_fetch_transport(observation: dict, fetch_manifest: dict, *, lead: int,
                          source_root: Path | None = None,
                          post_fetch_observation: dict | None = None) -> dict:
    """Bind watcher timing to the exact objects the fetch actually used."""
    files = fetch_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("fetch manifest lacks transport-bound file inventory")
    by_role = {}
    for row in files:
        if (isinstance(row, dict) and row.get("forecast_hour") == lead
                and row.get("role") in {"atmosphere", "soil"}):
            role = row["role"]
            if role in by_role:
                raise ValueError("fetch manifest duplicates a forcing role")
            by_role[role] = row
    products = observation.get("products")
    post_products = (
        post_fetch_observation.get("products")
        if isinstance(post_fetch_observation, dict) else None)
    binding = {}
    for role, product in (("atmosphere", "wrfnat"), ("soil", "wrfprs")):
        row = by_role.get(role)
        observed = products.get(product) if isinstance(products, dict) else None
        if (not isinstance(row, dict) or not isinstance(observed, dict)
                or row.get("transport") != observed.get("transport")
                or row.get("url") != observed.get("object_url")
                or observed.get("index_url") != str(row.get("url")) + ".idx"):
            raise ValueError(
                f"watcher availability does not bind the effective {product} "
                "fetch object/transport")
        stable = None
        if post_fetch_observation is not None:
            after = (post_products.get(product)
                     if isinstance(post_products, dict) else None)
            before_object = observed.get("object")
            after_object = after.get("object") if isinstance(after, dict) else None
            before_index = observed.get("index")
            after_index = after.get("index") if isinstance(after, dict) else None
            for label, before, current in (
                    ("object", before_object, after_object),
                    ("index", before_index, after_index)):
                if not isinstance(before, dict) or not isinstance(current, dict):
                    raise ValueError(
                        f"{product} {label} lacks stable pre/post-fetch HEAD "
                        "metadata")
                length = before.get("content_length_bytes")
                etag = before.get("etag")
                if (before.get("url") != current.get("url")
                        or current.get("content_length_bytes") != length
                        or current.get("etag") != etag
                        or isinstance(length, bool)
                        or not isinstance(length, int) or length <= 0
                        or not isinstance(etag, str) or not etag):
                    raise ValueError(
                        f"{product} {label} URL/Content-Length/ETag changed "
                        "during fetch")
            object_bytes = before_object["content_length_bytes"]
            manifest_bytes = row.get("bytes")
            name = row.get("name")
            if (manifest_bytes != object_bytes
                    or not isinstance(name, str) or not name
                    or Path(name).name != name
                    or source_root is None):
                raise ValueError(
                    f"full-file {product} manifest size/path differs from "
                    "the observed S3 object")
            downloaded = Path(source_root) / name
            digest = row.get("sha256")
            if (not downloaded.is_file()
                    or downloaded.stat().st_size != manifest_bytes
                    or not isinstance(digest, str)
                    or sha256_file(downloaded) != digest):
                raise ValueError(
                    f"downloaded {product} bytes differ from the fetch "
                    "manifest or observed S3 object")
            stable = {
                "pre_fetch": {
                    "url": before_object["url"],
                    "content_length_bytes": object_bytes,
                    "etag": before_object["etag"],
                },
                "post_fetch": {
                    "url": after_object["url"],
                    "content_length_bytes": after_object[
                        "content_length_bytes"],
                    "etag": after_object["etag"],
                },
                "index_pre_fetch": {
                    "url": before_index["url"],
                    "content_length_bytes": before_index[
                        "content_length_bytes"],
                    "etag": before_index["etag"],
                },
                "index_post_fetch": {
                    "url": after_index["url"],
                    "content_length_bytes": after_index[
                        "content_length_bytes"],
                    "etag": after_index["etag"],
                },
            }
        binding[product] = {
            "transport": row["transport"],
            "object_url": row["url"],
            "index_url": observed["index_url"],
            "download_sha256": row.get("sha256"),
            "download_bytes": row.get("bytes"),
            "stable_remote_identity": stable,
        }
    result = dict(observation)
    result["effective_fetch"] = binding
    result["effective_fetch_sha256"] = _digest(binding)
    return result


class _AvailabilityWatcher:
    """Observe all target hours concurrently while earlier work executes."""

    def __init__(self, cycle_root: Path, *, backend, cycle: datetime,
                 target: int, plan: StreamPlan, progress=print):
        self.cycle_root = cycle_root
        self.backend = backend
        self.cycle = cycle
        self.target = target
        self.plan = plan
        self.progress = progress
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.errors: dict[int, BaseException] = {}
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        try:
            for lead in range(0, self.target + 1):
                path = _availability_path(self.cycle_root, lead)
                if path.is_file():
                    _validated_observation(path, cycle=self.cycle, lead=lead)
                    continue
                thread = threading.Thread(
                    target=self._worker, args=(lead,), daemon=True,
                    name=f"gpuwm-availability-f{lead:03d}")
                thread.start()
                self.threads.append(thread)
        except BaseException:
            # A malformed later pre-existing receipt can be discovered after
            # earlier lead workers have already started.  Starting is itself
            # transactional: no worker is allowed to outlive a failed start.
            self.close()
            raise

    def _worker(self, lead: int) -> None:
        try:
            try:
                observed = self.backend.wait_for_hour(
                    self.cycle, lead,
                    timeout_s=self.plan.wait_timeout_seconds,
                    poll_s=self.plan.poll_seconds,
                    stop_event=self.stop_event)
            except TypeError as error:
                # Small injected test backends predating the cancellable
                # boundary remain usable; production implements stop_event.
                if "stop_event" not in str(error):
                    raise
                observed = self.backend.wait_for_hour(
                    self.cycle, lead,
                    timeout_s=self.plan.wait_timeout_seconds,
                    poll_s=self.plan.poll_seconds)
            if self.stop_event.is_set():
                return
            payload = {
                "schema": "gpuwm-stream-availability-v2",
                "cycle": self.cycle.strftime("%Y-%m-%dT%H:00:00Z"),
                "lead": lead,
                **observed,
            }
            _publish_json_exact(
                _availability_path(self.cycle_root, lead), payload)
        except BaseException as error:
            if not self.stop_event.is_set():
                self.errors[lead] = error
        finally:
            with self.condition:
                self.condition.notify_all()

    def wait(self, lead: int) -> dict:
        path = _availability_path(self.cycle_root, lead)
        deadline = time.monotonic() + self.plan.wait_timeout_seconds
        while True:
            if path.is_file():
                return _validated_observation(
                    path, cycle=self.cycle, lead=lead)
            error = self.errors.get(lead)
            if error is not None:
                raise RuntimeError(
                    f"availability watcher failed for f{lead:03d}") from error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"availability watcher did not publish f{lead:03d} "
                    "before its controller timeout")
            with self.condition:
                self.condition.wait(min(remaining, 1.0))

    def close(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join()
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("availability watcher did not stop cleanly")


def _stage_output(path: Path, *, valid: Callable[[Path], object],
                  command: Callable[[], list[str]], backend,
                  stage: str, progress=print):
    """Adopt a valid create-only stage or quarantine a partial and rerun."""
    if path.exists():
        try:
            return valid(path)
        except (OSError, ValueError, KeyError, TypeError,
                json.JSONDecodeError) as error:
            aside = fetch_guard.quarantine(path, tag="interrupted")
            progress(
                f"stream {stage}: moved incomplete output aside to {aside} "
                f"({error}); rebuilding")
    backend.run_command(command(), stage=stage)
    return valid(path)


def _valid_root(path: Path, *, plan: StreamPlan, cycle: datetime, lead: int,
                source_sums: Path):
    wrapper = path / "public-wrapper-result.json"
    payload = _json(wrapper, "root-preparation result")
    expected_hours = list(range(lead + 1))
    if (payload.get("status") != "PASS"
            or payload.get("source_cycle") != cycle.isoformat()
            or payload.get("model_start_time") != cycle.isoformat()
            or payload.get("source_forecast_hours") != expected_hours
            or payload.get("model_forcing_hours") != expected_hours
            or payload.get("forcing_hours") != expected_hours
            or float(payload.get("history_interval_seconds", -1.0))
            != float(plan.experiment.root.history_interval_s)
            or payload.get("physics", {}).get("profile")
            != plan.physics_profile):
        raise ValueError(
            f"root preparation receipt identity/status mismatch: {wrapper}")
    cache_contract = payload.get("prepared_cache_contract")
    expected_operation = "initial" if lead == 1 else "extend-one-hour"
    if (not isinstance(cache_contract, dict)
            or cache_contract.get("mode") != "sealed-prefix-v1"
            or cache_contract.get("operation") != expected_operation):
        raise ValueError(
            f"root preparation is not the requested sealed operation: "
            f"{wrapper}")
    if lead > 1:
        expected_predecessor = (
            path.parent.parent / f"f{lead - 1:03d}" / "root-preparation")
        if Path(cache_contract.get("predecessor", "")).resolve() \
                != expected_predecessor.resolve():
            raise ValueError(
                f"root preparation names another predecessor: {wrapper}")
    bridge = path / "native" / "native-bridge"
    bridge_manifest_sha256 = _verify_sha256_manifest(bridge)
    cache = path / "native" / "prepared-cache"
    cache_header = _json(
        cache / "header.json", "root prepared-cache header")
    cache_identity = cache_header.get("identity")
    if not isinstance(cache_identity, dict):
        raise ValueError(f"root prepared-cache identity is missing: {cache}")
    from gpuwm.ingest.prepared_cache import PreparedCacheReader
    cache_verification = PreparedCacheReader(
        cache, expected_identity=cache_identity).verify_all()
    lbc = cache_header.get("metadata", {}).get("lbc")
    if (cache_identity.get("bridge_manifest_sha256")
            != bridge_manifest_sha256
            or cache_identity.get("source_manifest_sha256")
            != sha256_file(source_sums)
            or cache_identity.get("forcing_hours") != expected_hours
            or not isinstance(lbc, dict)
            or len(lbc.get("intervals", [])) != lead
            or cache_contract.get("content_sha256")
            != cache_verification["content_sha256"]):
        raise ValueError(
            f"root prepared-cache/bridge prefix identity mismatch: {cache}")
    report = path / "native" / "preparation-report" / "report.json"
    report_payload = _json(report, "root preparation report")
    if (report_payload.get("status") != "PASS"
            or report_payload.get("source_cycle") != cycle.isoformat()
            or report_payload.get("model_start_time") != cycle.isoformat()
            or report_payload.get("source_forecast_hours") != expected_hours
            or report_payload.get("model_forcing_hours") != expected_hours
            or report_payload.get("target_domain_sha256")
            != plan.domain_identity_sha256):
        raise ValueError(
            f"root preparation report identity/status mismatch: {report}")
    report_input = report_payload.get("input")
    report_cache = report_payload.get("prepared_cache")
    if (not isinstance(report_input, dict)
            or report_input.get("bridge") != str(bridge.resolve())
            or report_input.get("bridge_manifest_sha256")
            != bridge_manifest_sha256
            or report_input.get("source_manifest_sha256")
            != sha256_file(source_sums)
            or report_input.get("source_forecast_hours") != expected_hours
            or report_input.get("model_forcing_hours") != expected_hours
            or report_input.get("forcing_hours") != expected_hours
            or not isinstance(report_cache, dict)
            or report_cache.get("content_sha256")
            != cache_verification["content_sha256"]):
        raise ValueError(
            f"root preparation report input/cache identity mismatch: {report}")
    named_report = payload.get("preparation_report")
    if (not isinstance(named_report, str)
            or Path(named_report).resolve() != report.resolve()):
        raise ValueError(
            f"root preparation wrapper names another report: {wrapper}")
    if not source_sums.is_file():
        raise FileNotFoundError(
            f"root preparation source checksums disappeared: {source_sums}")
    return payload


def _valid_hierarchy(path: Path, *, plan: StreamPlan, cycle: datetime,
                     lead: int, source_sums: Path, root: Path,
                     wps: Path, native: Path, stock: Path):
    receipt = path / "receipt.json"
    payload = _json(receipt, "hierarchy receipt")
    provenance = payload.get("provenance")
    root_header = _json(
        root / "native" / "prepared-cache" / "header.json",
        "root prepared-cache header")
    if (payload.get("schema") != "gpuwm-native-hrrr-hierarchy-direct-v1"
            or payload.get("status") != "PASS"
            or payload.get("valid_time") != cycle.isoformat()
            or payload.get("domain_count") != len(plan.experiment.domains)
            or payload.get("forcing_hours") != list(range(lead + 1))
            or not isinstance(provenance, dict)
            or provenance.get("source_manifest_sha256")
            != sha256_file(source_sums)
            or provenance.get("wps_namelist_sha256") != sha256_file(wps)
            or provenance.get("native_namelist_input_sha256")
            != sha256_file(native)
            or provenance.get("stock_wrf_namelist_input_sha256")
            != sha256_file(stock)
            or provenance.get("root_static_receipt_sha256") != sha256_file(
                root / "native-static-receipt.json")
            or provenance.get("root_prepared_content_sha256")
            != root_header.get("content_sha256")):
        raise ValueError(
            f"hierarchy preparation receipt identity/status mismatch: "
            f"{receipt}")
    artifact = path / "hierarchy-artifacts" / "receipt.json"
    artifact_payload = _json(artifact, "hierarchy artifact receipt")
    expected_ids = [domain.grid_id for domain in plan.experiment.domains]
    if (artifact_payload.get("status") != "READY"
            or artifact_payload.get("domain_count") != len(expected_ids)
            or artifact_payload.get("grid_ids") != expected_ids):
        raise ValueError(
            f"hierarchy artifact receipt identity/status mismatch: {artifact}")
    return payload


def _valid_run(path: Path, *, plan: StreamPlan, experiment: Path,
               hierarchy: Path, expected_restart: Path | None,
               cycle: datetime, lead: int):
    payload, checkpoint = _healthy_run_receipt(
        path / "evidence" / "run-receipt.json")
    expected_ids = [domain.grid_id for domain in plan.experiment.domains]
    expected_health = {f"d{grid_id:02d}" for grid_id in expected_ids}
    experiment_row = payload.get("experiment")
    restart_row = payload.get("restart_contract")
    input_row = payload.get("input")
    authority = None if not isinstance(input_row, dict) else input_row.get(
        "authority_sha256")
    actual_restart = None if expected_restart is None else str(
        Path(expected_restart).resolve())
    if (payload.get("schema")
            != "gpuwm-prepared-domain-tree-forecast-v1"
            or not isinstance(experiment_row, dict)
            or experiment_row.get("start_time") != cycle.isoformat()
            or float(experiment_row.get("run_seconds", -1.0))
            != lead * 3600.0
            or not isinstance(restart_row, dict)
            or restart_row.get("mode") != "sealed-forcing-extension"
            or restart_row.get("restart_input") != actual_restart
            or not isinstance(authority, dict)
            or authority.get("experiment_config") != sha256_file(experiment)
            or authority.get("preparation_receipt")
            != sha256_file(hierarchy / "receipt.json")
            or input_row.get("prepared_root") != str(hierarchy.resolve())
            or input_row.get("forcing_hours") != list(range(lead + 1))
            or set(payload["health"]["initial"]) != expected_health
            or set(payload["health"]["final"]) != expected_health):
        raise ValueError(
            f"prepared-tree run receipt identity mismatch: "
            f"{path / 'evidence' / 'run-receipt.json'}")
    if expected_restart is not None and checkpoint == Path(
            expected_restart).resolve():
        raise ValueError(
            "sealed forecast output checkpoint is identical to its input")
    output_row = payload.get("output")
    output_files = (
        output_row.get("files") if isinstance(output_row, dict) else None)
    if (not isinstance(output_row, dict)
            or output_row.get("io_mode") != plan.io_mode
            or not isinstance(output_files, list)
            or output_row.get("frame_count") != len(output_files)
            or output_row.get("total_bytes") != sum(
                row.get("bytes", -1) if isinstance(row, dict) else -1
                for row in output_files)):
        raise ValueError("prepared-tree run output inventory is malformed")
    seen_outputs = set()
    for row in output_files:
        output_path = Path(str(row.get("path", ""))) \
            if isinstance(row, dict) else Path()
        try:
            output_path.resolve().relative_to(path.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError(
                "prepared-tree output inventory escapes its run root") from exc
        if (str(output_path.resolve()) in seen_outputs
                or not output_path.is_file()
                or isinstance(row.get("bytes"), bool)
                or not isinstance(row.get("bytes"), int)
                or output_path.stat().st_size != row["bytes"]
                or not isinstance(row.get("sha256"), str)
                or sha256_file(output_path) != row["sha256"]):
            raise ValueError(
                "prepared-tree output frame changed or disappeared")
        seen_outputs.add(str(output_path.resolve()))
    if plan.io_mode == "history" and not output_files:
        raise ValueError("prepared-tree history run wrote no output frames")
    header = read_restart_header(checkpoint)
    if (header.get("grid_id") != expected_ids[0]
            or header.get("domain_ids") != expected_ids
            or float(header.get("elapsed_seconds", -1.0))
            != lead * 3600.0
            or header.get("experiment_fingerprint")
            != experiment_row.get("fingerprint")
            or header.get("forcing_extension_mode") != "sealed-prefix-v1"):
        raise ValueError(
            f"prepared-tree checkpoint does not represent f{lead:03d}: "
            f"{checkpoint}")
    members = _checkpoint_set(checkpoint, role_prefix="validation")
    if len(members) != len(expected_ids):
        raise ValueError("prepared-tree checkpoint generation is incomplete")
    for item, grid_id in zip(members, expected_ids):
        member = read_restart_header(Path(item["path"]))
        if (member.get("grid_id") != grid_id
                or member.get("domain_ids") != expected_ids
                or float(member.get("elapsed_seconds", -1.0))
                != lead * 3600.0
                or member.get("experiment_fingerprint")
                != experiment_row.get("fingerprint")
                or member.get("forcing_extension_mode")
                != "sealed-prefix-v1"):
            raise ValueError(
                f"checkpoint member d{grid_id:02d} identity mismatch")
    return payload, checkpoint


def _append_option(argv: list[str], flag: str, value) -> None:
    if value is not None:
        argv.extend((flag, str(value)))


def _root_command(plan: StreamPlan, *, cycle: datetime, lead: int,
                  source_root: Path, source_sums: Path,
                  experiment: Path, native_namelist: Path,
                  output: Path,
                  predecessor: Path | None = None) -> list[str]:
    del experiment  # The typed experiment is consumed by the forecast leg.
    argv = [
        sys.executable, "-m", "gpuwm.source_cli",
        "--source", "hrrr",
        "--source-root", str(source_root),
        "--source-manifest", str(source_sums),
        "--source-manifest-sha256", sha256_file(source_sums),
        "--geog-root", str(plan.geog_root),
        "--domain-spec", str(plan.domain_spec),
        "--namelist-input", str(native_namelist),
        "--physics-profile", plan.physics_profile,
        "--valid-time", cycle.strftime("%Y-%m-%d_%H:%M:%S"),
        "--output-root", str(output),
        "--run-seconds", str(lead * 3600),
        "--history-interval-seconds",
        str(plan.experiment.root.history_interval_s),
        "--forecast-start-hour", "0",
        "--forecast-end-hour", str(lead),
        "--pipeline-workers", str(plan.pipeline_workers),
        "--sealed-prepared-cache",
    ]
    if lead == 1:
        if predecessor is not None:
            raise ValueError("initial stream leg cannot name a predecessor")
    else:
        if predecessor is None:
            raise ValueError("stream forcing extension requires its predecessor")
        argv.extend(("--extend-root-preparation", str(predecessor)))
    _append_option(argv, "--prepare-workers", plan.prepare_workers)
    _append_option(argv, "--preprocess-backend", plan.preprocess_backend)
    _append_option(argv, "--preprocess-workers", plan.preprocess_workers)
    _append_option(
        argv, "--cpu-preprocess-bridge", plan.cpu_preprocess_bridge)
    for acknowledgement in plan.acknowledgements:
        argv.extend(("--ack", acknowledgement))
    return argv


def _hierarchy_command(plan: StreamPlan, *, cycle: datetime,
                       source_sums: Path, root: Path, wps: Path,
                       native: Path, stock: Path,
                       output: Path) -> list[str]:
    argv = [
        sys.executable, "-m", "gpuwm.source_cli",
        "--source", "hrrr",
        "--root-preparation", str(root),
        "--domain-spec", str(plan.domain_spec),
        "--wps-namelist", str(wps),
        "--namelist-input", str(native),
        "--stock-wrf-namelist-input", str(stock),
        "--geog-root", str(plan.geog_root),
        "--source-manifest", str(source_sums),
        "--source-manifest-sha256", sha256_file(source_sums),
        "--valid-time", cycle.strftime("%Y-%m-%d_%H:%M:%S"),
        "--output-root", str(output),
        "--child-workers", str(plan.child_workers),
    ]
    _append_option(
        argv, "--cpu-preprocess-bridge", plan.cpu_preprocess_bridge)
    return argv


def _run_command(plan: StreamPlan, *, hierarchy: Path, experiment: Path,
                 restart: Path | None, output: Path) -> list[str]:
    receipt = hierarchy / "receipt.json"
    argv = [
        sys.executable, "-m", "gpuwm.prepared_domain_tree_forecast",
        "--prepared-root", str(hierarchy),
        "--preparation-receipt-sha256", sha256_file(receipt),
        "--experiment-config", str(experiment),
        "--experiment-config-sha256", sha256_file(experiment),
        "--io-mode", plan.io_mode,
        "--sealed-forcing-extension",
        "--outdir", str(output),
    ]
    if restart is not None:
        argv.extend(("--restart", str(restart)))
    if plan.health_debug:
        argv.append("--health-debug")
    return argv


def _load_state(path: Path, *, cycle: datetime, lead: int) -> dict:
    if not path.is_file():
        return {
            "schema": "gpuwm-stream-leg-state-v1",
            "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
            "lead": lead,
            "stages": {},
        }
    state = _json(path, "stream leg state")
    if (state.get("schema") != "gpuwm-stream-leg-state-v1"
            or state.get("cycle") != cycle.strftime("%Y-%m-%dT%H:00:00Z")
            or state.get("lead") != lead
            or not isinstance(state.get("stages"), dict)):
        raise ValueError(f"stream leg state identity mismatch: {path}")
    return state


def _state_mark(path: Path, state: dict, stage: str, key: str,
                value: str) -> None:
    row = state["stages"].setdefault(stage, {})
    if key == "started_at":
        row.pop("completed_at", None)
    row[key] = value
    _atomic_json(path, state)


def _observation(path: Path, *, backend, cycle: datetime, lead: int,
                 plan: StreamPlan,
                 watcher: _AvailabilityWatcher | None = None) -> dict:
    if watcher is not None:
        expected = _availability_path(watcher.cycle_root, lead)
        if path.resolve() != expected.resolve():
            raise ValueError("availability watcher path identity mismatch")
        return watcher.wait(lead)
    if path.is_file():
        return _validated_observation(path, cycle=cycle, lead=lead)
    observed = backend.wait_for_hour(
        cycle, lead, timeout_s=plan.wait_timeout_seconds,
        poll_s=plan.poll_seconds)
    payload = {
        "schema": "gpuwm-stream-availability-v2",
        "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
        "lead": lead,
        **observed,
    }
    _publish_json_exact(path, payload)
    return payload


def _estimated_generation_bytes(plan: StreamPlan, lead: int) -> int:
    """Conservative retained prep/checkpoint/history storage estimate."""
    total = 0
    seconds = lead * 3600.0
    for domain in plan.experiment.domains:
        cells_3d = (int(domain.run.nx) * int(domain.run.ny)
                    * int(domain.run.nz))
        cells_2d = int(domain.run.nx) * int(domain.run.ny)
        # Preparation caches, state/setup, physics scratch, one checkpoint,
        # and intermediate native/static artifacts.  Counts deliberately
        # exceed the current inventories so adding fields does not silently
        # erase the guard margin.
        preparation_and_checkpoint = (
            cells_3d * 4 * 192 + cells_2d * 4 * 96)
        # History mode writes t=0 as well as every scheduled cadence through
        # the endpoint.  Omitting that initial frame under-reserved every leg.
        frames = 1 + int(math.ceil(
            seconds / float(domain.history_interval_s)))
        history = frames * (
            cells_3d * 4 * 64 + cells_2d * 4 * 48)
        total += preparation_and_checkpoint + history
    return total * _GENERATION_STORAGE_SAFETY_FACTOR


def _existing_disk_anchor(path: Path) -> Path:
    """Nearest existing path whose device owns a possibly-new output path."""

    candidate = Path(path).resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(
                f"no existing filesystem anchor for stream path {path}")
        candidate = parent
    return candidate


def _disk_volume_identity(backend, path: Path) -> str:
    resolver = getattr(backend, "disk_volume_identity", None)
    if callable(resolver):
        value = resolver(Path(path))
    else:
        value = os.stat(_existing_disk_anchor(path)).st_dev
    if isinstance(value, bool) or not isinstance(value, (str, int)) \
            or str(value) == "":
        raise RuntimeError(f"invalid disk-volume identity for {path}: {value!r}")
    return str(value)


def _free_disk_bytes(backend, path: Path) -> int:
    getter = getattr(backend, "free_disk_bytes", None)
    value = (getter(Path(path)) if callable(getter)
             else shutil.disk_usage(_existing_disk_anchor(path)).free)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"invalid free-disk observation for {path}: {value!r}")
    return int(value)


def _disk_layout(plan: StreamPlan, backend) -> dict[str, object]:
    work_identity = _disk_volume_identity(backend, plan.work_root)
    if plan.cache_dir is None:
        return {
            "kind": "work-only",
            "work_identity": work_identity,
            "cache_identity": None,
            "same_volume": False,
        }
    cache_identity = _disk_volume_identity(backend, plan.cache_dir)
    return {
        "kind": ("shared-work-cache-volume"
                 if cache_identity == work_identity
                 else "split-work-cache-volumes"),
        "work_identity": work_identity,
        "cache_identity": cache_identity,
        "same_volume": cache_identity == work_identity,
    }


def _fetch_prefix_is_verified(source_root: Path, *, cycle: datetime,
                              lead: int) -> bool:
    """Return whether the exact requested source prefix is already sealed.

    This is deliberately an inventory check, not a state-marker check.  A
    crash after fetch publishes its manifest and checksums but before the
    controller advances state; on resume, those verified payloads require no
    source or cache write reservation.  Any unreadable or mismatched receipt
    falls back to pricing the whole fetch rather than weakening the gate.
    """

    manifest_path = Path(source_root) / "fetch-manifest.json"
    try:
        payload = _json(manifest_path, "fetch manifest")
        if (payload.get("schema") != FETCH_MANIFEST_SCHEMA
                or payload.get("source") != "hrrr"
                or payload.get("cycle")
                != cycle.strftime("%Y-%m-%dT%H:%M:%SZ")
                or payload.get("forecast_hours") != list(range(lead + 1))
                or not isinstance(payload.get("files"), list)):
            return False
        _verify_sha256_manifest(source_root)
        checksum_rows = {}
        for line in (Path(source_root) / "SHA256SUMS").read_text(
                encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            checksum_rows[name] = digest
        expected_pairs = {
            (hour, role)
            for hour in range(lead + 1)
            for role in ("atmosphere", "soil")
        }
        observed_pairs = {}
        checksum_receipts = []
        for row in payload["files"]:
            if not isinstance(row, dict):
                return False
            role = row.get("role")
            if role == "checksums":
                checksum_receipts.append(row)
                continue
            pair = (row.get("forecast_hour"), role)
            name = row.get("name")
            size = row.get("bytes")
            digest = row.get("sha256")
            if (pair not in expected_pairs or pair in observed_pairs
                    or not isinstance(name, str)
                    or name not in checksum_rows
                    or isinstance(size, bool) or not isinstance(size, int)
                    or size < 0 or not isinstance(digest, str)
                    or checksum_rows[name] != digest):
                return False
            candidate = Path(source_root) / name
            if (not candidate.is_file() or candidate.stat().st_size != size
                    or sha256_file(candidate) != digest):
                return False
            observed_pairs[pair] = name
        sums = Path(source_root) / "SHA256SUMS"
        if (set(observed_pairs) != expected_pairs
                or len(set(observed_pairs.values())) != len(expected_pairs)
                or set(checksum_rows) != set(observed_pairs.values())
                or len(checksum_receipts) != 1):
            return False
        checksum_receipt = checksum_receipts[0]
        if (checksum_receipt.get("name") != "SHA256SUMS"
                or checksum_receipt.get("forecast_hour") is not None
                or checksum_receipt.get("bytes") != sums.stat().st_size
                or checksum_receipt.get("sha256") != sha256_file(sums)):
            return False
    except (OSError, ValueError):
        return False
    return True


def _disk_capacity_gate(path: Path, *, plan: StreamPlan, backend,
                        cycle: datetime, lead: int,
                        observations: list[dict]) -> dict:
    """Reserve the complete program once, before its first source fetch."""
    layout = _disk_layout(plan, backend)
    if path.is_file():
        payload = _json(path, "stream disk-capacity receipt")
        if (payload.get("schema") != "gpuwm-stream-disk-capacity-v4"
                or payload.get("status") != "PASS"
                or payload.get("plan_identity_sha256")
                != plan.identity_sha256
                or payload.get("basis", {}).get("target_lead")
                != plan.target_lead
                or payload.get("basis", {}).get("cycle_count")
                != plan.cycle_count
                or payload.get("volume_layout") != layout):
            raise ValueError(f"disk-capacity receipt identity mismatch: {path}")
        return payload
    sizes = [row.get("object_content_length_bytes") for row in observations]
    if (not sizes or any(isinstance(value, bool) or not isinstance(value, int)
                         or value <= 0 for value in sizes)):
        raise RuntimeError(
            "stream disk gate requires HTTP Content-Length for both full "
            "forcing products before fetch")
    if any(value > _SOURCE_HOUR_RESERVATION_BYTES for value in sizes):
        raise RuntimeError(
            "stream disk gate observed a forcing hour above the enforced "
            "full-object reservation envelope")
    projected_source_hour = _SOURCE_HOUR_RESERVATION_BYTES
    projected_source_output = (
        projected_source_hour * (plan.target_lead + 1) * plan.cycle_count)
    projected_cache_copy = (
        projected_source_output if plan.cache_dir is not None else 0)
    projected_generations = sum(
        _estimated_generation_bytes(plan, hour)
        for hour in range(1, plan.target_lead + 1)) * plan.cycle_count
    fixed_margin = 2 * 1024 ** 3
    retained_total = (projected_source_output + projected_cache_copy
                      + projected_generations + fixed_margin)
    work_required = (
        projected_source_output + projected_generations + fixed_margin
        + (projected_cache_copy if layout["same_volume"] else 0))
    cache_required = (
        None if plan.cache_dir is None else
        (work_required if layout["same_volume"] else projected_cache_copy))
    work_free = _free_disk_bytes(backend, plan.work_root)
    cache_free = (
        None if plan.cache_dir is None else
        (work_free if layout["same_volume"] else
         _free_disk_bytes(backend, plan.cache_dir)))
    work_pass = work_free >= work_required
    cache_pass = (
        True if cache_required is None else cache_free >= cache_required)
    status = "PASS" if work_pass and cache_pass else "REFUSED"
    work_volume = {
        "path": str(plan.work_root),
        "identity": layout["work_identity"],
        "free_bytes_observed": work_free,
        "required_bytes": int(work_required),
        "remaining_bytes_after_requirement": int(work_free - work_required),
        "projected_source_output_bytes": int(projected_source_output),
        "projected_generation_bytes": int(projected_generations),
        "fixed_margin_bytes": fixed_margin,
        "projected_cache_copy_bytes": (
            int(projected_cache_copy) if layout["same_volume"] else 0),
    }
    cache_volume = None
    if plan.cache_dir is not None:
        cache_volume = {
            "path": str(plan.cache_dir),
            "identity": layout["cache_identity"],
            "same_as_work_volume": bool(layout["same_volume"]),
            "free_bytes_observed": int(cache_free),
            "required_bytes": int(cache_required),
            "remaining_bytes_after_requirement": int(
                cache_free - cache_required),
            "projected_cache_copy_bytes": int(projected_cache_copy),
        }
    payload = {
        "schema": "gpuwm-stream-disk-capacity-v4",
        "status": status,
        "plan_identity_sha256": plan.identity_sha256,
        "initial_cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
        "initial_lead": lead,
        "free_bytes_observed": int(work_free),
        "free_bytes_after_requirement": (
            int(work_free - retained_total) if layout["same_volume"]
            or plan.cache_dir is None else None),
        "conservative_retained_requirement_bytes": int(retained_total),
        "projected_source_bytes": int(projected_source_output),
        "projected_source_output_bytes": int(projected_source_output),
        "projected_cache_copy_bytes": int(projected_cache_copy),
        "projected_generation_bytes": int(projected_generations),
        "fixed_margin_bytes": fixed_margin,
        "volume_layout": layout,
        "work_volume": work_volume,
        "cache_volume": cache_volume,
        "basis": {
            "target_lead": plan.target_lead,
            "cycle_count": plan.cycle_count,
            "full_file_hour_bytes_observed": list(sizes),
            "full_file_hour_reservation_cap_bytes":
                _SOURCE_HOUR_RESERVATION_BYTES,
            "future_hour_size_policy": "enforced-fixed-upper-envelope-v1",
            "generation_storage_safety_factor":
                _GENERATION_STORAGE_SAFETY_FACTOR,
            "domain_count": len(plan.experiment.domains),
            "history_cadence_seconds": {
                f"d{domain.grid_id:02d}": float(domain.history_interval_s)
                for domain in plan.experiment.domains
            },
        },
    }
    if status == "REFUSED":
        refusal = path.parent / "disk-capacity-refusals" / (
            f"refused-{time.time_ns()}.json")
        _publish_json_exact(refusal, payload)
        raise RuntimeError(
            "stream disk-capacity refusal before fetch: conservative "
            "per-volume requirement exceeds free space "
            f"(work {work_required:,}/{work_free:,} bytes"
            + ("" if cache_required is None else
               f", cache {cache_required:,}/{cache_free:,} bytes")
            + f"); receipt {refusal}")
    _publish_json_exact(path, payload)
    return payload


def _disk_headroom_gate(path: Path, *, plan: StreamPlan, backend,
                        cycle: datetime, lead: int,
                        observations: list[dict],
                        fetch_prefix_verified: bool = False,
                        prior_fetch_prefix_verified: bool = False) -> Path:
    """Catch post-reservation/external consumption before each fetch."""
    layout = _disk_layout(plan, backend)
    previous = None
    if path.is_file():
        previous = _json(path, "stream disk-headroom receipt")
        if (previous.get("schema") != "gpuwm-stream-disk-headroom-v2"
                or previous.get("status") != "PASS"
                or previous.get("plan_identity_sha256")
                != plan.identity_sha256
                or previous.get("cycle")
                != cycle.strftime("%Y-%m-%dT%H:00:00Z")
                or previous.get("lead") != lead
                or previous.get("volume_layout") != layout):
            raise ValueError(f"disk-headroom receipt identity mismatch: {path}")
    source_hours = [
        observation.get("object_content_length_bytes")
        for observation in observations]
    if (not source_hours or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in source_hours)):
        raise RuntimeError(
            "stream disk headroom requires full-object Content-Length")
    if any(value > _SOURCE_HOUR_RESERVATION_BYTES for value in source_hours):
        raise RuntimeError(
            "stream disk headroom observed a forcing hour above the "
            "enforced full-object reservation envelope")
    if prior_fetch_prefix_verified and lead <= 1:
        raise ValueError(
            "a prior fetch prefix exists only for leads after f001")
    emergency_margin = 512 * 1024 ** 2
    generation = _estimated_generation_bytes(plan, lead)
    if fetch_prefix_verified:
        prefix_state = "requested-prefix-verified"
        pricing_policy = "verified-requested-prefix-zero-write"
        source_hours_priced = 0
        source_output = 0
    elif prior_fetch_prefix_verified:
        if len(source_hours) != 1:
            raise ValueError(
                "verified-prior-prefix headroom requires exactly the new "
                "terminal-hour observation")
        prefix_state = "prior-prefix-verified-terminal-write"
        pricing_policy = "observed-terminal-hour"
        source_hours_priced = 1
        source_output = int(source_hours[0])
    else:
        prefix_hours = lead + 1
        if len(source_hours) > prefix_hours:
            raise ValueError(
                "unverified-prefix headroom received more observations than "
                "the requested source prefix")
        missing_hours = prefix_hours - len(source_hours)
        prefix_state = "unverified-full-prefix-replacement"
        pricing_policy = (
            "observed-full-prefix" if missing_hours == 0 else
            "observed-plus-enforced-envelope-for-missing-hours")
        source_hours_priced = prefix_hours
        source_output = int(
            sum(source_hours)
            + missing_hours * _SOURCE_HOUR_RESERVATION_BYTES)
    cache_copy = source_output if plan.cache_dir is not None else 0
    work_required = int(
        source_output + generation + emergency_margin
        + (cache_copy if layout["same_volume"] else 0))
    cache_required = (
        None if plan.cache_dir is None else
        (work_required if layout["same_volume"] else cache_copy))
    work_free = _free_disk_bytes(backend, plan.work_root)
    cache_free = (
        None if plan.cache_dir is None else
        (work_free if layout["same_volume"] else
         _free_disk_bytes(backend, plan.cache_dir)))
    status = (
        "PASS" if work_free >= work_required
        and (cache_required is None or cache_free >= cache_required)
        else "REFUSED")
    payload = {
        "schema": "gpuwm-stream-disk-headroom-v2",
        "status": status,
        "plan_identity_sha256": plan.identity_sha256,
        "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
        "lead": lead,
        "free_bytes_observed": int(work_free),
        "next_generation_requirement_bytes": work_required,
        "projected_source_hour_bytes": source_output,
        "projected_source_output_bytes": source_output,
        "projected_cache_copy_bytes": cache_copy,
        "source_hours_observed": len(source_hours),
        "source_hours_priced": source_hours_priced,
        "source_hours_requiring_write": source_hours_priced,
        "source_hour_content_length_bytes": list(source_hours),
        "fetch_prefix_verified_before_gate": bool(fetch_prefix_verified),
        "prior_fetch_prefix_verified_before_gate": bool(
            prior_fetch_prefix_verified),
        "source_prefix_state": prefix_state,
        "source_hour_pricing_policy": pricing_policy,
        "projected_generation_bytes": int(generation),
        "emergency_margin_bytes": emergency_margin,
        "resume_recheck": previous is not None,
        "volume_layout": layout,
        "work_volume": {
            "path": str(plan.work_root),
            "identity": layout["work_identity"],
            "free_bytes_observed": work_free,
            "required_bytes": work_required,
            "remaining_bytes_after_requirement": work_free - work_required,
        },
        "cache_volume": (
            None if plan.cache_dir is None else {
                "path": str(plan.cache_dir),
                "identity": layout["cache_identity"],
                "same_as_work_volume": bool(layout["same_volume"]),
                "free_bytes_observed": int(cache_free),
                "required_bytes": int(cache_required),
                "remaining_bytes_after_requirement": int(
                    cache_free - cache_required),
            }),
    }
    if status == "REFUSED":
        refusal = path.parent / "disk-headroom-refusals" / (
            f"refused-{time.time_ns()}.json")
        _publish_json_exact(refusal, payload)
        raise RuntimeError(
            "stream disk-headroom refusal before fetch: next immutable "
            "generation exceeds per-volume free space "
            f"(work {work_required:,}/{work_free:,} bytes"
            + ("" if cache_required is None else
               f", cache {cache_required:,}/{cache_free:,} bytes")
            + f"); receipt {refusal}")
    if previous is None:
        _publish_json_exact(path, payload)
        return path
    attempt = path.parent / "disk-headroom-attempts" / (
        f"pass-{time.time_ns()}.json")
    _publish_json_exact(attempt, payload)
    return attempt


def _load_link(path: Path, *, plan: StreamPlan, cycle: datetime,
               lead: int, previous_link_hash: str | None,
               previous_checkpoint: Path | None = None) -> dict:
    payload = _json(path, "stream chain link")
    expected_cycle = cycle.strftime("%Y-%m-%dT%H:00:00Z")
    if (payload.get("schema") != LINK_SCHEMA
            or payload.get("plan_identity_sha256") != plan.identity_sha256
            or payload.get("cycle") != expected_cycle
            or payload.get("lead") != lead
            or payload.get("previous_link_sha256") != previous_link_hash
            or payload.get("status") != "PASS"):
        raise ValueError(f"stream chain link identity mismatch: {path}")
    claimed = payload.get("link_sha256")
    unsigned = dict(payload)
    unsigned.pop("link_sha256", None)
    if claimed != _digest(unsigned):
        raise ValueError(f"stream chain link digest mismatch: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"stream chain link has no artifacts: {path}")
    for item in artifacts:
        _validate_artifact(item)
    checkpoint = payload.get("checkpoint_root")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        raise ValueError(f"stream chain link checkpoint is absent: {path}")
    expected_restart = (
        None if previous_checkpoint is None
        else str(Path(previous_checkpoint).resolve()))
    if payload.get("restart_input") != expected_restart:
        raise ValueError(
            f"stream chain link restart predecessor mismatch: {path}")
    by_role = {}
    semantic_roles = {
        "experiment_config", "hierarchy_receipt", "run_receipt",
        "root_preparation_result", "root_preparation_report",
        "root_prepared_cache_header", "root_bridge_manifest",
        "fetch_checksums", "wps_namelist", "native_namelist",
        "stock_namelist",
    }
    for item in artifacts:
        role = item.get("role")
        if role in semantic_roles:
            if role in by_role:
                raise ValueError(f"stream chain link duplicates {role}: {path}")
            by_role[role] = Path(item["path"])
    if set(by_role) != semantic_roles:
        raise ValueError(f"stream chain link lacks semantic run artifacts: {path}")
    root = by_role["root_preparation_result"].parent
    if (by_role["root_preparation_report"]
            != root / "native" / "preparation-report" / "report.json"
            or by_role["root_prepared_cache_header"]
            != root / "native" / "prepared-cache" / "header.json"
            or by_role["root_bridge_manifest"]
            != root / "native" / "native-bridge" / "SHA256SUMS"):
        raise ValueError(f"stream chain link root paths disagree: {path}")
    _valid_root(
        root, plan=plan, cycle=cycle, lead=lead,
        source_sums=by_role["fetch_checksums"])
    run_root = by_role["run_receipt"].parents[1]
    hierarchy = by_role["hierarchy_receipt"].parent
    _valid_hierarchy(
        hierarchy, plan=plan, cycle=cycle, lead=lead,
        source_sums=by_role["fetch_checksums"], root=root,
        wps=by_role["wps_namelist"], native=by_role["native_namelist"],
        stock=by_role["stock_namelist"])
    _receipt, semantic_checkpoint = _valid_run(
        run_root, plan=plan, experiment=by_role["experiment_config"],
        hierarchy=hierarchy, expected_restart=previous_checkpoint,
        cycle=cycle, lead=lead)
    if semantic_checkpoint != Path(checkpoint).resolve():
        raise ValueError(
            f"stream chain link checkpoint disagrees with run receipt: {path}")
    return payload


def _summary(cycle_root: Path, *, plan: StreamPlan, cycle: datetime,
             target: int, links: list[dict], status: str) -> dict:
    rows = []
    previous_leg_end = None
    for link in links:
        stages = link["stages"]
        def stage_times(name: str) -> tuple[datetime, datetime, float]:
            started = datetime.fromisoformat(
                stages[name]["started_at"].replace("Z", "+00:00"))
            completed = datetime.fromisoformat(
                stages[name]["completed_at"].replace("Z", "+00:00"))
            return started, completed, max(
                0.0, (completed - started).total_seconds())

        observed = datetime.fromisoformat(
            link["availability"]["first_observed_at"].replace(
                "Z", "+00:00"))
        fetch_start, fetch_end, fetch_seconds = stage_times("fetch")
        root_start, root_end, root_seconds = stage_times("root_preparation")
        hierarchy_start, hierarchy_end, hierarchy_seconds = \
            stage_times("hierarchy")
        run_start, leg_end, run_seconds = stage_times("run")
        remote_ready_raw = link["availability"].get(
            "remote_ready_last_modified_at")
        remote_ready = (
            None if remote_ready_raw is None else datetime.fromisoformat(
                remote_ready_raw.replace("Z", "+00:00")))
        row = {
            "lead": link["lead"],
            "forcing_set_first_observed_at": _iso(observed),
            "remote_ready_last_modified_at": remote_ready_raw,
            "remote_ready_semantics": (
                "maximum HTTP Last-Modified over both forcing objects and "
                "both indexes; serving-endpoint metadata, not producer "
                "upload/completion time"),
            "fetch_started_at": _iso(fetch_start),
            "fetch_completed_at": _iso(fetch_end),
            "fetch_seconds": fetch_seconds,
            "root_preparation_started_at": _iso(root_start),
            "root_preparation_completed_at": _iso(root_end),
            "root_preparation_seconds": root_seconds,
            "hierarchy_preparation_started_at": _iso(hierarchy_start),
            "hierarchy_preparation_completed_at": _iso(hierarchy_end),
            "hierarchy_preparation_seconds": hierarchy_seconds,
            # Retained for readers of the initial streaming receipt draft.
            # It always means the hierarchy stage, never root preparation.
            "preparation_completed_at": _iso(hierarchy_end),
            "preparation_completed_at_semantics":
                "compatibility alias of hierarchy_preparation_completed_at",
            "forecast_started_at": _iso(run_start),
            "forecast_completed_at": _iso(leg_end),
            "forecast_seconds": run_seconds,
            "leg_completed_at": _iso(leg_end),
            "availability_to_fetch_complete_seconds": max(
                0.0, (fetch_end - observed).total_seconds()),
            "availability_to_leg_complete_seconds": max(
                0.0, (leg_end - observed).total_seconds()),
            "remote_ready_to_leg_complete_seconds": (
                None if remote_ready is None else max(
                    0.0, (leg_end - remote_ready).total_seconds())),
            "previous_leg_completed_before_this_lead_first_observed": (
                None if previous_leg_end is None
                else previous_leg_end <= observed),
            "previous_leg_completed_before_remote_ready_last_modified": (
                None if previous_leg_end is None or remote_ready is None
                else previous_leg_end <= remote_ready),
            "validity": "PASS",
            "checkpoint_root": link["checkpoint_root"],
            "link_sha256": link["link_sha256"],
        }
        rows.append(row)
        previous_leg_end = leg_end
    return {
        "schema": CHAIN_SCHEMA,
        "status": status,
        "plan": {
            "path": str(plan.path),
            "sha256": plan.sha256,
            "identity_sha256": plan.identity_sha256,
        },
        "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
        "target_lead": target,
        "completed_leads": [row["lead"] for row in rows],
        "latest_link_sha256": None if not links else links[-1]["link_sha256"],
        "timeline": rows,
        "updated_at": _iso(_utc_now()),
    }


def _select_cycle(plan: StreamPlan, backend, *,
                  after_cycle: datetime | None = None
                  ) -> tuple[datetime, int, Path]:
    plan.work_root.mkdir(parents=True, exist_ok=True)
    active_path = plan.work_root / "active-cycle.json"
    active = _json(active_path, "active stream cycle") \
        if active_path.is_file() else None
    if active is not None and (
            active.get("schema") != ACTIVE_SCHEMA
            or active.get("plan_identity_sha256") != plan.identity_sha256
            or active.get("status") not in ("ACTIVE", "COMPLETE")):
        raise ValueError(
            f"stream cycle marker belongs to a different plan or schema: "
            f"{active_path}")
    if active is not None and active.get("status") == "ACTIVE":
        cycle = parse_cycle(active["cycle"], "hrrr")
        target = active["target_lead"]
        if after_cycle is not None and cycle != after_cycle + timedelta(hours=1):
            raise ValueError(
                f"active stream cycle {cycle:%Y-%m-%dT%H}Z is not the "
                f"immediate hourly successor of completed cycle "
                f"{after_cycle:%Y-%m-%dT%H}Z")
    else:
        if after_cycle is not None:
            cycle = backend.wait_for_next_cycle(
                after_cycle, timeout_s=plan.wait_timeout_seconds,
                poll_s=plan.poll_seconds)
            expected = after_cycle + timedelta(hours=1)
            if cycle != expected:
                raise ValueError(
                    f"stream backend selected {cycle:%Y-%m-%dT%H}Z after "
                    f"{after_cycle:%Y-%m-%dT%H}Z; the next cycle must be "
                    f"exactly {expected:%Y-%m-%dT%H}Z (no leapfrogging)")
        else:
            cycle = (backend.resolve_latest() if plan.cycle == "latest"
                     else parse_cycle(plan.cycle, "hrrr"))
        horizon = hrrr_cycle_horizon(cycle)
        target = plan.target_lead
        if target > horizon:
            raise ValueError(
                f"target_lead f{target:03d} exceeds cycle "
                f"{cycle:%Y-%m-%dT%H}Z horizon f{horizon:03d}")
        _atomic_json(active_path, {
            "schema": ACTIVE_SCHEMA,
            "status": "ACTIVE",
            "plan_identity_sha256": plan.identity_sha256,
            "cycle": cycle.strftime("%Y-%m-%dT%H"),
            "target_lead": target,
            "selected_at": _iso(backend.now()),
        })
    if (isinstance(target, bool) or not isinstance(target, int)
            or target < 1 or target > hrrr_cycle_horizon(cycle)):
        raise ValueError(f"active stream target is invalid: {target!r}")
    cycle_root = plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
    return cycle, target, cycle_root


def _run_cycle_impl(plan: StreamPlan, *, backend, progress=print,
                    after_cycle: datetime | None = None) -> dict:
    """Run or resume every sealed hourly leg for one selected cycle."""
    plan = _pin_plan_authorities(plan)
    if hasattr(backend, "bind_plan_authorities"):
        backend.bind_plan_authorities(plan)
    cycle, target, cycle_root = _select_cycle(
        plan, backend, after_cycle=after_cycle)
    summary_path = cycle_root / "chain-summary.json"
    cycle_root.mkdir(parents=True, exist_ok=True)
    with fetch_guard.hold("stream-cycle", cycle_root, progress=progress):
        source_root = cycle_root / "source"
        _stage_owner(
            source_root, plan=plan, cycle=cycle, lead=0,
            role="source-prefix")
        watcher = _AvailabilityWatcher(
            cycle_root, backend=backend, cycle=cycle, target=target,
            plan=plan, progress=progress)
        # Attach before start so the outer unwind guard can close a partially
        # started watcher even if thread creation or receipt validation fails.
        setattr(backend, "_gpuwm_availability_watcher", watcher)
        watcher.start()
        links = []
        previous_hash = None
        prior_checkpoint = None
        prior_root = None
        for lead in range(1, target + 1):
            _verify_pinned_authorities(plan)
            leg = cycle_root / "legs" / f"f{lead:03d}"
            link_path = leg / "chain-link.json"
            if link_path.is_file():
                link = _load_link(
                    link_path, plan=plan, cycle=cycle, lead=lead,
                    previous_link_hash=previous_hash,
                    previous_checkpoint=prior_checkpoint)
                links.append(link)
                previous_hash = link["link_sha256"]
                prior_checkpoint = Path(link["checkpoint_root"])
                prior_root = (
                    cycle_root / "legs" / f"f{lead:03d}" /
                    "root-preparation")
                progress(f"stream f{lead:03d}: sealed link verified; skipped")
                continue

            leg.mkdir(parents=True, exist_ok=True)
            state_path = leg / "state.json"
            state = _load_state(state_path, cycle=cycle, lead=lead)
            availability = _observation(
                leg / "availability.json", backend=backend, cycle=cycle,
                lead=lead, plan=plan, watcher=watcher)
            zero_availability = watcher.wait(0)
            disk_receipt_path = plan.work_root / "disk-capacity.json"
            _disk_capacity_gate(
                disk_receipt_path, plan=plan, backend=backend,
                cycle=cycle, lead=lead,
                observations=[zero_availability, availability])
            disk_headroom_path = leg / "disk-headroom.json"
            fetch_prefix_verified = _fetch_prefix_is_verified(
                source_root, cycle=cycle, lead=lead)
            prior_fetch_prefix_verified = (
                lead > 1 and _fetch_prefix_is_verified(
                    source_root, cycle=cycle, lead=lead - 1))
            if fetch_prefix_verified:
                headroom_observations = (
                    [zero_availability, availability]
                    if lead == 1 else [availability])
            elif prior_fetch_prefix_verified:
                headroom_observations = [availability]
            else:
                # A corrupt/partial predecessor may be quarantined and every
                # requested object downloaded again.  Price the whole prefix
                # from its already-sealed HEAD observations, not merely the
                # newly uploading terminal hour.
                headroom_observations = [
                    watcher.wait(hour) for hour in range(lead + 1)]
            headroom_receipt_path = _disk_headroom_gate(
                disk_headroom_path, plan=plan, backend=backend,
                cycle=cycle, lead=lead,
                observations=headroom_observations,
                fetch_prefix_verified=fetch_prefix_verified,
                prior_fetch_prefix_verified=prior_fetch_prefix_verified)

            _state_mark(
                state_path, state, "fetch", "started_at", _iso(backend.now()))
            fetch_manifest = backend.fetch_prefix(
                plan, cycle, lead, source_root)
            if Path(fetch_manifest).resolve() != \
                    (source_root / "fetch-manifest.json").resolve():
                raise ValueError(
                    "production fetch returned an unexpected manifest path")
            source_sums = source_root / "SHA256SUMS"
            fetch_payload = _json(fetch_manifest, "fetch manifest")
            if fetch_payload.get("forecast_hours") != list(range(lead + 1)):
                raise ValueError(
                    f"fetch manifest does not seal f000..f{lead:03d}")
            post_fetch_observation = (
                backend.observe_hour_after_fetch(cycle, lead)
                if hasattr(backend, "observe_hour_after_fetch") else None)
            availability = _bind_fetch_transport(
                availability, fetch_payload, lead=lead,
                source_root=source_root,
                post_fetch_observation=post_fetch_observation)
            _state_mark(
                state_path, state, "fetch", "completed_at", _iso(backend.now()))

            authority = leg / "authorities"
            manifest_snapshot = _publish_exact(
                authority / "fetch-manifest.json", Path(fetch_manifest).read_bytes())
            sums_snapshot = _publish_exact(
                authority / "SHA256SUMS", source_sums.read_bytes())

            configs = leg / "configs"
            experiment = configs / "experiment.toml"
            native = configs / "namelist.input"
            stock = configs / "namelist.stock.input"
            wps = _publish_exact(
                configs / "namelist.wps", plan.wps_namelist.read_bytes())
            _materialize_experiment(
                plan, experiment, cycle=cycle, lead=lead)
            delta = cycle - plan.experiment.start_time
            domain_starts = [
                plan.experiment.domain_start_time(domain.grid_id) + delta
                for domain in plan.experiment.domains
            ]
            _materialize_input_namelist(
                plan.namelist_input, native, cycle=cycle, lead=lead,
                domain_starts=domain_starts)
            _materialize_input_namelist(
                plan.stock_wrf_namelist_input, stock, cycle=cycle, lead=lead,
                domain_starts=domain_starts)

            root = leg / "root-preparation"
            _stage_owner(
                root, plan=plan, cycle=cycle, lead=lead,
                role="root-preparation")
            _state_mark(
                state_path, state, "root_preparation", "started_at",
                _iso(backend.now()))
            with fetch_guard.hold(
                    "fetch-geog", plan.geog_root, progress=progress):
                _verify_pinned_authorities(plan)
                _stage_output(
                    root, valid=lambda path: _valid_root(
                        path, plan=plan, cycle=cycle, lead=lead,
                        source_sums=source_sums),
                    command=lambda: _root_command(
                        plan, cycle=cycle, lead=lead, source_root=source_root,
                        source_sums=source_sums, experiment=experiment,
                        native_namelist=native, output=root,
                        predecessor=prior_root),
                    backend=backend, stage=f"f{lead:03d} root preparation",
                    progress=progress)
                _verify_pinned_authorities(plan)
            _state_mark(
                state_path, state, "root_preparation", "completed_at",
                _iso(backend.now()))

            hierarchy = leg / "prepared-hierarchy"
            _stage_owner(
                hierarchy, plan=plan, cycle=cycle, lead=lead,
                role="prepared-hierarchy")
            _state_mark(
                state_path, state, "hierarchy", "started_at",
                _iso(backend.now()))
            with fetch_guard.hold(
                    "fetch-geog", plan.geog_root, progress=progress):
                _verify_pinned_authorities(plan)
                _stage_output(
                    hierarchy, valid=lambda path: _valid_hierarchy(
                        path, plan=plan, cycle=cycle, lead=lead,
                        source_sums=source_sums, root=root, wps=wps,
                        native=native, stock=stock),
                    command=lambda: _hierarchy_command(
                        plan, cycle=cycle, source_sums=source_sums, root=root,
                        wps=wps, native=native, stock=stock,
                        output=hierarchy),
                    backend=backend,
                    stage=f"f{lead:03d} hierarchy preparation",
                    progress=progress)
                _verify_pinned_authorities(plan)
            _state_mark(
                state_path, state, "hierarchy", "completed_at",
                _iso(backend.now()))

            run = leg / "run"
            _stage_owner(
                run, plan=plan, cycle=cycle, lead=lead,
                role="sealed-run")
            _state_mark(
                state_path, state, "run", "started_at", _iso(backend.now()))
            run_receipt, checkpoint = _stage_output(
                run, valid=lambda path: _valid_run(
                    path, plan=plan, experiment=experiment,
                    hierarchy=hierarchy, expected_restart=prior_checkpoint,
                    cycle=cycle, lead=lead),
                command=lambda: _run_command(
                    plan, hierarchy=hierarchy, experiment=experiment,
                    restart=prior_checkpoint, output=run),
                backend=backend, stage=f"f{lead:03d} sealed forecast",
                progress=progress)
            _state_mark(
                state_path, state, "run", "completed_at", _iso(backend.now()))

            prior_set = ([] if prior_checkpoint is None else
                         _checkpoint_set(
                             prior_checkpoint, role_prefix="restart_input"))
            output_set = _checkpoint_set(
                checkpoint, role_prefix="checkpoint_output")
            artifacts = [
                _artifact(
                    plan.work_root / ".gpuwm-stream-owner.json",
                    role="work_root_owner"),
                *([
                    _artifact(
                        plan.work_root / "gpu-allocation.json",
                        role="gpu_allocation")
                ] if (plan.work_root / "gpu-allocation.json").is_file()
                  else []),
                _artifact(
                    plan.authority_manifest, role="plan_authority_manifest"),
                _artifact(disk_receipt_path, role="disk_capacity_receipt"),
                _artifact(
                    headroom_receipt_path, role="disk_headroom_receipt"),
                _artifact(manifest_snapshot, role="fetch_manifest"),
                _artifact(sums_snapshot, role="fetch_checksums"),
                _artifact(experiment, role="experiment_config"),
                _artifact(native, role="native_namelist"),
                _artifact(stock, role="stock_namelist"),
                _artifact(wps, role="wps_namelist"),
                _artifact(plan.domain_spec, role="domain_spec"),
                _artifact(
                    root / "public-wrapper-result.json",
                    role="root_preparation_result"),
                _artifact(
                    root / "native" / "preparation-report" / "report.json",
                    role="root_preparation_report"),
                _artifact(
                    root / "native" / "prepared-cache" / "header.json",
                    role="root_prepared_cache_header"),
                *_prepared_cache_payload_artifacts(
                    root / "native" / "prepared-cache",
                    role_prefix="root_prepared_cache"),
                _artifact(
                    root / "native" / "native-bridge" / "SHA256SUMS",
                    role="root_bridge_manifest"),
                _artifact(hierarchy / "receipt.json", role="hierarchy_receipt"),
                _artifact(
                    hierarchy / "hierarchy-artifacts" / "receipt.json",
                    role="hierarchy_artifact_receipt"),
                *_directory_file_artifacts(
                    hierarchy / "hierarchy-artifacts",
                    role_prefix="hierarchy_payload",
                    exclude=(
                        hierarchy / "hierarchy-artifacts" / "receipt.json",
                    )),
                _artifact(
                    run / "evidence" / "run-receipt.json",
                    role="run_receipt"),
                *prior_set,
                *output_set,
            ]
            next(item for item in artifacts
                 if item["role"] == "fetch_manifest")["production_path"] = \
                str(Path(fetch_manifest).resolve())
            next(item for item in artifacts
                 if item["role"] == "fetch_checksums")["production_path"] = \
                str(source_sums.resolve())
            unsigned = {
                "schema": LINK_SCHEMA,
                "status": "PASS",
                "plan_identity_sha256": plan.identity_sha256,
                "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
                "lead": lead,
                "previous_link_sha256": previous_hash,
                "availability": availability,
                "stages": state["stages"],
                "checkpoint_root": str(checkpoint),
                "restart_input": (
                    None if prior_checkpoint is None
                    else str(prior_checkpoint.resolve())),
                "health": {
                    "initial": run_receipt["health"]["initial"],
                    "final": run_receipt["health"]["final"],
                },
                "artifacts": artifacts,
            }
            link = {**unsigned, "link_sha256": _digest(unsigned)}
            _publish_json_exact(link_path, link)
            links.append(link)
            previous_hash = link["link_sha256"]
            prior_checkpoint = checkpoint
            prior_root = root
            _atomic_json(
                summary_path,
                _summary(
                    cycle_root, plan=plan, cycle=cycle, target=target,
                    links=links, status="RUNNING"))
            progress(
                f"stream f{lead:03d}: PASS checkpoint {checkpoint.name}")

        watcher.close()
        if getattr(backend, "_gpuwm_availability_watcher", None) is watcher:
            delattr(backend, "_gpuwm_availability_watcher")
        summary = _summary(
            cycle_root, plan=plan, cycle=cycle, target=target,
            links=links, status="PASS")
        _atomic_json(summary_path, summary)
        _atomic_json(plan.work_root / "active-cycle.json", {
            "schema": ACTIVE_SCHEMA,
            "status": "COMPLETE",
            "plan_identity_sha256": plan.identity_sha256,
            "cycle": cycle.strftime("%Y-%m-%dT%H"),
            "target_lead": target,
            "completed_at": _iso(backend.now()),
            "chain_summary": str(summary_path.resolve()),
            "chain_summary_sha256": sha256_file(summary_path),
        })
        return summary


def _run_cycle(plan: StreamPlan, *, backend, progress=print,
               after_cycle: datetime | None = None) -> dict:
    """Close concurrent availability probes on every success/failure path."""
    try:
        return _run_cycle_impl(
            plan, backend=backend, progress=progress,
            after_cycle=after_cycle)
    finally:
        watcher = getattr(backend, "_gpuwm_availability_watcher", None)
        if watcher is not None:
            watcher.close()
            delattr(backend, "_gpuwm_availability_watcher")


def _cycle_record(summary: dict, summary_path: Path,
                  previous_cycle_link: str | None) -> dict:
    artifact = _artifact(summary_path, role="cycle_chain_summary")
    unsigned = {
        "cycle": summary["cycle"],
        "target_lead": summary["target_lead"],
        "completed_leads": summary["completed_leads"],
        "latest_leg_link_sha256": summary["latest_link_sha256"],
        "previous_cycle_link_sha256": previous_cycle_link,
        "chain_summary": artifact,
    }
    return {**unsigned, "cycle_link_sha256": _digest(unsigned)}


def _validate_cycle_record(record: Mapping,
                           previous_cycle_link: str | None,
                           *, plan: StreamPlan) -> None:
    if (not isinstance(record, Mapping)
            or record.get("previous_cycle_link_sha256")
            != previous_cycle_link):
        raise ValueError("multi-cycle chain predecessor mismatch")
    claimed = record.get("cycle_link_sha256")
    unsigned = dict(record)
    unsigned.pop("cycle_link_sha256", None)
    if claimed != _digest(unsigned):
        raise ValueError("multi-cycle chain link digest mismatch")
    _validate_artifact(record.get("chain_summary"))
    summary = _json(
        Path(record["chain_summary"]["path"]), "cycle chain summary")
    if (summary.get("schema") != CHAIN_SCHEMA
            or summary.get("status") != "PASS"
            or summary.get("cycle") != record.get("cycle")
            or summary.get("target_lead") != record.get("target_lead")
            or summary.get("latest_link_sha256")
            != record.get("latest_leg_link_sha256")):
        raise ValueError("multi-cycle record disagrees with its cycle summary")
    target = summary.get("target_lead")
    expected_leads = (list(range(1, target + 1))
                      if isinstance(target, int) and not isinstance(target, bool)
                      and target >= 1 else None)
    if (expected_leads is None
            or summary.get("completed_leads") != expected_leads
            or record.get("completed_leads") != expected_leads):
        raise ValueError("completed cycle does not contain every target lead")
    try:
        cycle = datetime.strptime(summary["cycle"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("completed cycle has an invalid cycle instant") from exc
    previous = None
    previous_checkpoint = None
    for lead in expected_leads:
        link = _load_link(
            Path(record["chain_summary"]["path"]).parent
            / "legs" / f"f{lead:03d}" / "chain-link.json",
            plan=plan, cycle=cycle, lead=lead,
            previous_link_hash=previous,
            previous_checkpoint=previous_checkpoint)
        previous = link["link_sha256"]
        previous_checkpoint = Path(link["checkpoint_root"])
    if previous != summary.get("latest_link_sha256"):
        raise ValueError("completed cycle's leg chain tail changed")


def _program_payload(plan: StreamPlan, records: list[dict], *,
                     status: str, started_at: str) -> dict:
    return {
        "schema": PROGRAM_SCHEMA,
        "status": status,
        "plan": {
            "path": str(plan.path),
            "sha256": plan.sha256,
            "identity_sha256": plan.identity_sha256,
        },
        "requested_cycle_count": plan.cycle_count,
        "completed_cycle_count": len(records),
        "cycles": records,
        "latest_cycle_link_sha256": (
            None if not records else records[-1]["cycle_link_sha256"]),
        "started_at": started_at,
        "updated_at": _iso(_utc_now()),
    }


def _validate_program(program: Mapping, *, plan: StreamPlan) -> list[dict]:
    """Deep-validate one outer multi-cycle program before adoption.

    Cycle records bind the heavy per-cycle evidence, but the outer summary's
    count, tail, status, and ordering are also operational claims.  Validate
    those derived claims in one place for both completed replay and crash
    recovery so neither path can return PASS from a forged wrapper.
    """
    expected_plan = {
        "path": str(plan.path),
        "sha256": plan.sha256,
        "identity_sha256": plan.identity_sha256,
    }
    records_value = program.get("cycles") if isinstance(program, Mapping) \
        else None
    if (not isinstance(program, Mapping)
            or program.get("schema") != PROGRAM_SCHEMA
            or program.get("plan") != expected_plan
            or program.get("requested_cycle_count") != plan.cycle_count
            or not isinstance(records_value, list)):
        raise ValueError(
            "multi-cycle stream summary belongs to another plan or schema")
    records = list(records_value)
    completed = program.get("completed_cycle_count")
    if (isinstance(completed, bool) or not isinstance(completed, int)
            or completed != len(records)):
        raise ValueError(
            "multi-cycle stream summary completed count disagrees with "
            "its cycle records")
    expected_tail = (
        None if not records else records[-1].get("cycle_link_sha256")
        if isinstance(records[-1], Mapping) else None)
    if program.get("latest_cycle_link_sha256") != expected_tail:
        raise ValueError(
            "multi-cycle stream summary tail disagrees with its cycle records")
    if len(records) > plan.cycle_count:
        raise ValueError(
            "multi-cycle stream summary exceeds the requested cycle count")
    status = program.get("status")
    if status == "PASS":
        if len(records) != plan.cycle_count:
            raise ValueError(
                "PASS multi-cycle summary has the wrong cycle count")
    elif status == "RUNNING":
        if len(records) >= plan.cycle_count:
            raise ValueError(
                "RUNNING multi-cycle summary already contains the requested "
                "cycle count")
    else:
        raise ValueError(f"unknown multi-cycle stream status {status!r}")
    if not isinstance(program.get("started_at"), str):
        raise ValueError("multi-cycle stream summary lacks started_at")

    previous_link = None
    previous_cycle = None
    for record in records:
        _validate_cycle_record(record, previous_link, plan=plan)
        try:
            cycle = datetime.strptime(
                record["cycle"], "%Y-%m-%dT%H:%M:%SZ")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "multi-cycle stream record has an invalid cycle instant") \
                from exc
        if previous_cycle is not None \
                and cycle != previous_cycle + timedelta(hours=1):
            raise ValueError(
                "multi-cycle stream records are not exact hourly successors")
        if previous_cycle is None and plan.cycle != "latest" \
                and cycle != parse_cycle(plan.cycle, "hrrr"):
            raise ValueError(
                "multi-cycle stream first record disagrees with the fixed "
                "plan cycle")
        previous_cycle = cycle
        previous_link = record["cycle_link_sha256"]
    return records


def _run_stream_impl(plan: StreamPlan, *, backend, progress=print) -> dict:
    """Run/resume a bounded unattended sequence of uploading cycles."""
    program_path = plan.work_root / "stream-summary.json"
    with fetch_guard.hold("stream-program", plan.work_root, progress=progress):
        plan = _pin_plan_authorities(plan)
        preexisting = program_path.is_file()
        if preexisting:
            program = _json(program_path, "multi-cycle stream summary")
            records = _validate_program(program, plan=plan)
            if program.get("status") == "PASS":
                return program
            started_at = program.get("started_at")
        else:
            records = []
            started_at = _iso(backend.now())
            program = _program_payload(
                plan, records, status="RUNNING", started_at=started_at)
            _validate_program(program, plan=plan)
            _atomic_json(program_path, program)

        # Crash window recovery: a cycle may have published its complete
        # chain and active marker immediately before the outer chain update.
        active_path = plan.work_root / "active-cycle.json"
        if preexisting and len(records) < plan.cycle_count \
                and active_path.is_file():
            active = _json(active_path, "active stream cycle")
            recorded_cycles = {record["cycle"] for record in records}
            active_cycle = active.get("cycle")
            try:
                active_cycle = parse_cycle(
                    active_cycle, "hrrr").strftime(
                        "%Y-%m-%dT%H:00:00Z")
            except (TypeError, ValueError):
                active_cycle = None
            if (active.get("schema") == ACTIVE_SCHEMA
                    and active.get("status") == "COMPLETE"
                    and active.get("plan_identity_sha256")
                    == plan.identity_sha256
                    and isinstance(active.get("chain_summary"), str)
                    and active_cycle not in recorded_cycles):
                recovered_path = Path(active["chain_summary"])
                if (recovered_path.is_file()
                        and sha256_file(recovered_path)
                        == active.get("chain_summary_sha256")):
                    recovered = _json(
                        recovered_path, "completed cycle chain summary")
                    if (recovered.get("cycle") != active_cycle
                            or recovered.get("target_lead")
                            != active.get("target_lead")):
                        raise ValueError(
                            "completed active marker disagrees with its cycle "
                            "chain summary")
                    previous = (None if not records else
                                records[-1]["cycle_link_sha256"])
                    records.append(_cycle_record(
                        recovered, recovered_path, previous))
                    recovered_status = (
                        "PASS" if len(records) == plan.cycle_count
                        else "RUNNING")
                    program = _program_payload(
                        plan, records, status=recovered_status,
                        started_at=started_at)
                    records = _validate_program(program, plan=plan)
                    _atomic_json(program_path, program)

        # Crash recovery above may have completed the final cycle and sealed
        # the outer PASS.  That is still verification/publishing work only;
        # do not ask for a GPU merely to enter a zero-iteration loop.
        if len(records) >= plan.cycle_count:
            return program
        fetch_preflight = getattr(backend, "preflight_fetch", None)
        if callable(fetch_preflight):
            fetch_preflight(plan)
        allocator = getattr(backend, "gpu_allocation", None)
        allocation_context = (
            allocator(plan) if callable(allocator) else nullcontext(None))
        with allocation_context as allocation:
            if allocation is not None:
                _publish_json_exact(
                    plan.work_root / "gpu-allocation.json", allocation)
            while len(records) < plan.cycle_count:
                after = (None if not records else datetime.strptime(
                    records[-1]["cycle"], "%Y-%m-%dT%H:%M:%SZ"))
                cycle_summary = _run_cycle(
                    plan, backend=backend, progress=progress,
                    after_cycle=after)
                cycle = datetime.strptime(
                    cycle_summary["cycle"], "%Y-%m-%dT%H:%M:%SZ")
                summary_path = (
                    plan.work_root / "cycles" / cycle.strftime("%Y%m%dT%H")
                    / "chain-summary.json")
                previous = (None if not records else
                            records[-1]["cycle_link_sha256"])
                records.append(_cycle_record(
                    cycle_summary, summary_path, previous))
                status = ("PASS" if len(records) == plan.cycle_count
                          else "RUNNING")
                program = _program_payload(
                    plan, records, status=status, started_at=started_at)
                records = _validate_program(program, plan=plan)
                _atomic_json(program_path, program)
        return program


def run_stream(plan: StreamPlan, *, backend=None, progress=print) -> dict:
    """Deep-verify a completed program, or lock/mask a GPU while advancing."""
    backend = ProductionBackend(progress=progress) if backend is None else backend
    _claim_work_root(plan)
    return _run_stream_impl(plan, backend=backend, progress=progress)


def stream_main(args) -> int:
    try:
        plan = load_stream_plan(args.plan)
    except OSError as exc:
        # Missing operator-declared inputs are a plan-load refusal (exit 2
        # through gpuwm.cli), not a Python traceback.  Keep this conversion
        # at the load boundary: an unexpected file disappearance during the
        # actual stream still retains its traceback and investigation value.
        raise ValueError(f"stream plan load refused: {exc}") from None
    summary = run_stream(plan)
    print(json.dumps({
        "status": summary["status"],
        "completed_cycle_count": summary["completed_cycle_count"],
        "cycles": [record["cycle"] for record in summary["cycles"]],
        "stream_summary": str(
            (plan.work_root / "stream-summary.json").resolve()),
    }, sort_keys=True))
    return 0


def register_cli(subparsers) -> None:
    # The disambiguation goes in BOTH strings on purpose.  ``help`` is the
    # only one argparse shows in ``gpuwm --help``'s command list, and
    # ``description`` is the only one it shows in ``gpuwm stream --help``,
    # which is where somebody who has already picked the wrong command
    # arrives.  One noun, two unrelated features, and the whole point of
    # the [tiles] naming ruling is that neither door stays silent about it.
    _NOT_THE_OTHER_ONE = (
        "NOT the out-of-core mode: to run a domain larger than the card, "
        "see the [tiles] table and docs/public/TILES.md")
    parser = subparsers.add_parser(
        "stream",
        help=("follow uploading forecast cycles with sealed hourly "
              f"restart-extend legs ({_NOT_THE_OTHER_ONE})"),
        description=("Follow an uploading forecast cycle with sealed hourly "
                     f"restart-extend legs.  {_NOT_THE_OTHER_ONE}."))
    parser.add_argument(
        "plan", type=Path, metavar="PLAN.toml",
        help=f"strict {PLAN_SCHEMA} orchestration plan")
    parser.set_defaults(func=stream_main)


__all__ = [
    "ACTIVE_SCHEMA", "CHAIN_SCHEMA", "LINK_SCHEMA", "PLAN_SCHEMA",
    "PROGRAM_SCHEMA",
    "ProductionBackend", "StreamPlan", "load_stream_plan", "register_cli",
    "run_stream", "stream_main",
]
