"""Default-on hydro door: decide at preparation whether a fabric covers the domain.

Drew, 2026-10-18: hydro is default on for every forecast and every ensemble
member whenever a hydrofabric covers the domain; ``--hydro off`` is the
opt-out; where no fabric exists hydro is off and the receipt names the gap.

This module is orchestration only. It reads the provider table
(``gpuwm/data/hydro/fabric-providers.v1.json``), the experiment's parent grid,
and runs the providers' own Rust tools (fabric builders, ``rw_hydrofabric
prepare``) as subprocesses. A new fabric source is a table row, never a new
code path here. No hydrologic number is computed in Python.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

SCHEMA = "gpuwm-hydro-door.v1"
TABLE_SCHEMA = "gpuwm-hydro-fabric-providers-v1"
TABLE = Path(__file__).resolve().parents[1] / "data/hydro/fabric-providers.v1.json"
ENV_TABLE = "GPUWM_HYDRO_FABRIC_PROVIDERS"      # extra rows, appended after the shipped table
ENV_CACHE = "GPUWM_HYDRO_FABRIC_CACHE"
ENV_DECISION = "GPUWM_HYDRO_DOOR_DECISION_JSON"  # launcher -> runner/member processes
PREPARED_KEYS = ("realization", "stages", "driver_factory")
LATTICE = 33  # parent-grid sample points per axis for coverage fractions


class HydroDoorRefusal(ValueError):
    """Explicit ``--hydro`` with no covering fabric: the request cannot be met."""


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


# -- provider table ---------------------------------------------------------

def load_providers(table=None):
    """Shipped rows, then any rows from ``$GPUWM_HYDRO_FABRIC_PROVIDERS``."""
    documents = [json.loads(Path(table or TABLE).read_text())]
    extra = os.environ.get(ENV_TABLE)
    if table is None and extra:
        documents.append(json.loads(Path(extra).read_text()))
    head = documents[0]
    rows = []
    for document in documents:
        if document.get("schema") != TABLE_SCHEMA:
            raise ValueError(f"hydro fabric provider table needs schema {TABLE_SCHEMA}")
        rows.extend(document["providers"])
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate hydro fabric provider ids make the door ambiguous")
    rows.sort(key=lambda row: (int(row["priority"]), row["id"]))
    return {"coupled_parent_lsm": head["coupled_parent_lsm"], "budget": head["budget"],
            "providers": rows, "table_sha256": _digest(documents)}


# -- domain geometry --------------------------------------------------------

def _wrap180(value):
    return (value + 180.0) % 360.0 - 180.0


def domain_geometry(config_path):
    """Parent-grid footprint from the experiment's own projection resolver."""
    import numpy as np
    from gpuwm.experiment import load_experiment
    from gpuwm.static.projection import grids_from_projection_config

    experiment = load_experiment(config_path)
    grid = grids_from_projection_config(experiment)[0]
    run = experiment.domains[0].run
    nx, ny = int(run.nx), int(run.ny)
    xs = np.linspace(1.0, float(nx), LATTICE)
    ys = np.linspace(1.0, float(ny), LATTICE)
    x, y = np.meshgrid(xs, ys)
    lat, lon = grid.ij_to_latlon(x.ravel(), y.ravel())
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    from .parent_geometry import bind
    from gpuwm.hydrofabric import parent_grid_contract
    contract = parent_grid_contract(bind(grid))
    return geometry_from_points(lat.tolist(), lon.tolist(), center_lon=float(grid.cen_lon),
                                parent_lsm=int(run.sf_surface_physics),
                                grid_key=_digest({"definition": grid.definition(), "nx": nx, "ny": ny}),
                                nx=nx, ny=ny, dx_m=float(run.dx),
                                parent_grid_contract=contract,
                                time_origin=experiment.start_time.isoformat() + "Z",
                                duration_seconds=float(experiment.run_seconds),
                                interval_seconds=float(run.bldt * 60.0 or run.dt))


def geometry_from_points(lat, lon, *, center_lon, parent_lsm, grid_key, **extra):
    relative = [_wrap180(value - center_lon) for value in lon]
    west = _wrap180(center_lon + min(relative))
    east = _wrap180(center_lon + max(relative))
    return {"south": min(lat), "north": max(lat), "west": west, "east": east,
            "crosses_dateline": west > east, "points": list(zip(lat, lon)),
            "parent_lsm": parent_lsm, "grid_key": grid_key, **extra}


def _inside(region, lat, lon):
    if region == "global":
        return True
    west, south, east, north = region
    if not south <= lat <= north:
        return False
    return west <= lon <= east if west <= east else (lon >= west or lon <= east)


def covered_fraction(coverage, points):
    regions = ["global"] if coverage == "global" else coverage
    inside = sum(1 for lat, lon in points if any(_inside(r, lat, lon) for r in regions))
    return inside / len(points) if points else 0.0


# -- availability probes (each runs the provider's own tool) -----------------

def _executable(spec):
    from gpuwm import bridges
    candidates = bridges.artifact_candidates(spec["env"], spec["filename"],
                                             crate_relative=spec.get("crate"))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def _expand(value):
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return None if "$" in expanded else Path(expanded)


def _builder_path(need):
    path = _expand(os.environ.get(need["env"]) or need.get("default"))
    return None if path is None else path.resolve()


def probe(row):
    """Every reason this row cannot run today; empty means available."""
    reasons = []
    if row["kind"] == "prepared":
        manifest = _expand(row.get("manifest"))
        if manifest is None or not manifest.is_file():
            reasons.append(f"prepared manifest not present: {row.get('manifest')}")
        return reasons
    builder = row["builder"]
    exe = _executable(builder["executable"])
    if exe is None:
        spec = builder["executable"]
        reasons.append(f"builder executable {spec['filename']} not found "
                       f"(set {spec['env']} or build {spec.get('crate')})")
    else:
        try:
            usage = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            text = usage.stdout + usage.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            text = ""
            reasons.append(f"builder executable {exe} did not answer: {error}")
        if text and not re.search(rf"(^|[\s|:]){re.escape(builder['subcommand'])}(\s|$|\|)", text):
            reasons.append(f"{exe.name} has no '{builder['subcommand']}' subcommand yet "
                           f"({builder.get('pending', 'builder not implemented')})")
        table = builder.get("source_table")
        if table:
            path = _repo_root() / table["path"]
            if not path.is_file():
                reasons.append(f"source table {table['path']} is not in this tree")
            else:
                admit = subprocess.run([str(exe), "admit", str(path), table["row"]],
                                       capture_output=True, text=True, timeout=30)
                if admit.returncode:
                    detail = (admit.stderr or admit.stdout).strip().splitlines()
                    reasons.append(f"source row {table['row']} not admitted: "
                                   f"{detail[-1] if detail else 'exit ' + str(admit.returncode)}")
    for need in builder.get("paths", ()):
        resolved = _builder_path(need)
        if resolved is None or not resolved.is_dir():
            reasons.append(f"{need['label']} not available (set {need['env']})")
    if row.get("preparation"):
        from .provider import recipe_path
        for name in ("parameters", "hydrofabric", "reservoirs"):
            spec = row["preparation"][name]
            if _executable(spec) is None:
                reasons.append(f"provider preparation needs {spec['filename']} ({spec['env']})")
        try:
            path = recipe_path(row)
            if not path.is_file():
                reasons.append(f"same-network parameter source recipe missing: {path}")
        except (OSError, ValueError) as error:
            reasons.append(f"parameter source recipe unavailable: {error}")
    return reasons


def _realization_reasons(row):
    name = row.get("realization")
    if not name:
        return [f"no admitted coupled realization: {row.get('realization_gap', 'none named')}"]
    from .config import resolve_configuration
    try:
        resolve_configuration(name)
    except ValueError as error:
        return [str(error)]
    return []


def evaluate(row, geometry, *, probe_row=probe):
    fraction = covered_fraction(row["coverage"], geometry["points"])
    reasons = []
    minimum = float(row.get("min_covered_fraction", 0.0))
    if fraction <= 0.0 or fraction < minimum:
        reasons.append(f"covers {fraction:.3f} of the parent domain; row needs >= "
                       f"{minimum:g} and > 0")
    if "parent_lsm" in row and geometry["parent_lsm"] not in row["parent_lsm"]:
        reasons.append("provider exchange requires its declared parent LSM; this row supports "
                       + str(row["parent_lsm"]))
    licence = row["data_licence"]
    if licence["status"] != "verified":
        reasons.append(f"data licence {licence['status']} ({licence['name']}): {licence['reason']}")
    reasons.extend(_realization_reasons(row))
    if fraction > 0.0:
        try:
            reasons.extend(probe_row(row))
        except Exception as error:  # noqa: BLE001 - a failing probe is a named gap
            reasons.append(f"availability probe failed: {type(error).__name__}: {error}")
    return {"id": row["id"], "owner": row["owner"], "kind": row["kind"],
            "covered_fraction": round(fraction, 6), "reasons": reasons,
            "verdict": "available" if not reasons else "rejected"}


# -- the decision -------------------------------------------------------------

def explicitly_prepared(settings):
    return bool(settings) and any(settings.get(key) for key in PREPARED_KEYS)


def _summary(geometry):
    if geometry is None:
        return None
    keys = ("south", "north", "west", "east", "crosses_dateline", "parent_lsm", "grid_key",
            "nx", "ny", "dx_m", "parent_grid_contract",
            "time_origin", "duration_seconds", "interval_seconds")
    return {key: geometry[key] for key in keys if key in geometry}


def decide(*, settings, requested, config_path=None, table=None, geometry=None,
           probe_row=probe, geometry_reader=None):
    """Return the door decision for one forecast; never touches a GPU.

    ``requested`` is the CLI value: None (default), "" (bare ``--hydro``),
    "off", or a settings path. ``settings`` is the [hydro] table already read
    by the activation policy (gpuwm.ensemble.optional_routing.activation_settings).
    """
    settings = dict(settings or {})
    base = {"schema": SCHEMA, "requested": requested}
    if requested == "off" or settings.get("activation_reason") == "cli_off":
        return {**base, "enabled": False, "reason": "cli_off", "gaps": []}
    if explicitly_prepared(settings):
        return {**base, "enabled": bool(settings.get("enabled")),
                "reason": settings.get("activation_reason", "configuration_enabled"),
                "fabric": "explicit settings", "gaps": []}
    if settings.get("activation_reason") == "configuration_off":
        return {**base, "enabled": False, "reason": "configuration_off", "gaps": []}
    providers = load_providers(table)
    base["table_sha256"] = providers["table_sha256"]
    if geometry is None:
        try:
            reader = geometry_reader or domain_geometry
            geometry = reader(config_path) if config_path is not None else None
        except Exception as error:  # noqa: BLE001 - the gap names the exact failure
            return {**base, "enabled": False, "reason": "no_domain_geometry",
                    "gaps": [{"owner": "hydro-world-door",
                              "gap": f"parent grid unreadable from {config_path}: "
                                     f"{type(error).__name__}: {error}"}]}
    if geometry is None:
        return {**base, "enabled": False, "reason": "no_domain_geometry",
                "gaps": [{"owner": "hydro-world-door",
                          "gap": "no experiment configuration, so no parent grid to cover"}]}
    base["footprint"] = _summary(geometry)
    candidates = [evaluate(row, geometry, probe_row=probe_row) for row in providers["providers"]]
    base["candidates"] = candidates
    allowed = providers["coupled_parent_lsm"]
    if geometry["parent_lsm"] not in allowed["values"]:
        return {**base, "enabled": False, "reason": "parent_lsm_unsupported",
                "gaps": [{"owner": allowed["owner"],
                          "gap": f"parent sf_surface_physics={geometry['parent_lsm']}; "
                                 f"{allowed['reason']}"}]}
    chosen = next((c for c in candidates if c["verdict"] == "available"), None)
    if chosen is None:
        gaps = [{"owner": c["owner"], "provider": c["id"], "gap": "; ".join(c["reasons"])}
                for c in candidates if c["covered_fraction"] > 0.0] or [
                {"owner": "hydro-world-fabric",
                 "gap": "no provider row covers this domain"}]
        return {**base, "enabled": False, "reason": "no_covering_fabric", "gaps": gaps}
    row = next(r for r in providers["providers"] if r["id"] == chosen["id"])
    return {**base, "enabled": True, "reason": "fabric_covered", "provider": row["id"],
            "provider_kind": row["kind"], "realization": row["realization"],
            "manifest_binding": row.get("manifest_binding", "fabric_manifest"),
            "attribution": row["data_licence"].get("attribution"), "gaps": [],
            "_row": row}


# -- materialisation at preparation -------------------------------------------

def cache_root(config_path):
    configured = os.environ.get(ENV_CACHE)
    if configured:
        return Path(configured).expanduser()
    return Path(config_path).expanduser().resolve().parent / ".gpuwm-hydro"


def _fill(template, values):
    return [str(part).format(**values) for part in template]


def materialize(decision, *, config_path, runner=subprocess.run, prepare=None):
    """Build (or reuse) the covering fabric and return the composed settings.

    The returned settings select the provider's realization with the prepared
    manifest bound; children read them through GPUWM_NEXTGEN_SETTINGS.
    """
    row = decision["_row"]
    footprint = decision["footprint"]
    started = time.perf_counter()
    build = {"provider": row["id"], "kind": row["kind"]}
    if row["kind"] == "prepared":
        manifest = _expand(row["manifest"])
        out = cache_root(config_path) / row["id"] / _digest(footprint)[:16]
        build["reused"] = True
    else:
        out = cache_root(config_path) / row["id"] / _digest(footprint)[:16]
        receipt = out / "door-prepared.json"
        manifest = None
        if receipt.is_file():
            manifest = Path(json.loads(receipt.read_text())["manifest"])
        paths = {}
        for need in row["builder"].get("paths", ()):
            if "argument" in need:
                resolved = _builder_path(need)
                if resolved is None or not resolved.is_dir():
                    raise RuntimeError(f"{need['label']} not available (set {need['env']})")
                paths[need["argument"]] = str(resolved)
        request = {"preparation": row.get("preparation"), "provider": row["id"], "footprint": footprint,
                   "builder": row["builder"], "data_licence": row["data_licence"],
                   "paths": paths}
        if row.get("preparation"):
            from .provider import preparation_identity
            request["source_recipe"] = preparation_identity(row)
        # A different recipe or initialization policy needs fresh preparation.
        # Keep the legacy footprint address only when its complete stamp matches;
        # rebuilding into it would overwrite an existing fabric and warm state.
        legacy_stamp = out / "door-request.json"
        if not (manifest is not None and manifest.is_file() and legacy_stamp.is_file()
                and json.loads(legacy_stamp.read_text()) == request):
            out = cache_root(config_path) / row["id"] / _digest(request)[:16]
            receipt = out / "door-prepared.json"
            manifest = (Path(json.loads(receipt.read_text())["manifest"])
                        if receipt.is_file() else None)
        stamp = out / "door-request.json"
        if (manifest is not None and manifest.is_file() and stamp.is_file()
                and json.loads(stamp.read_text()) == request):
            build["reused"] = True
        else:
            out.mkdir(parents=True, exist_ok=True)
            exe = _executable(row["builder"]["executable"])
            table = row["builder"].get("source_table") or {}
            values = {"exe": exe, "out": out, "row": table.get("row", ""),
                      "table": _repo_root() / table["path"] if table else "",
                      "west": footprint["west"], "south": footprint["south"],
                      "east": footprint["east"], "north": footprint["north"]}
            values.update(paths)
            if "parent_grid_contract" in footprint:
                (out / "parent-grid.json").write_text(json.dumps(footprint["parent_grid_contract"]["grid"], indent=2) + "\n")
            argv = _fill(row["builder"]["argv"], values)
            result = runner(argv, capture_output=True, text=True,
                            timeout=int(row["builder"].get("timeout_seconds", 1800)))
            if result.returncode:
                raise RuntimeError(f"hydro fabric builder {row['id']} failed: "
                                   f"{(result.stderr or result.stdout).strip()[-2000:]}")
            build["builder_argv"] = argv
            build["builder_seconds"] = time.perf_counter() - started
            config = {"geopackage": str(out / row["builder"].get("geopackage", "fabric.gpkg")),
                      "output": str(out / "prepared"),
                      "data_license": row["data_licence"]["name"]}
            if "parent_grid_contract" in footprint:
                config["grid"] = footprint["parent_grid_contract"]["grid"]
                config["parent_grid_contract"] = footprint["parent_grid_contract"]
            (out / "prepare.json").write_text(json.dumps(config, indent=2) + "\n")
            if row.get("preparation"):
                from .provider import prepare as prepare_provider
                prepared = prepare_provider(row=row, footprint=footprint, out=out,
                                            config=config, runner=runner)
            else:
                if prepare is None:
                    from gpuwm.hydrofabric import prepare
                prepared = prepare(out / "prepare.json")
            if isinstance(prepared, Mapping) and not prepared.get("manifest") and prepared.get("output"):
                prepared = dict(prepared, manifest=str(Path(prepared["output"]) / "manifest.json"))
            if not isinstance(prepared, Mapping) or not prepared.get("manifest"):
                raise RuntimeError("rw_hydrofabric prepare returned no manifest path")
            manifest = Path(prepared["manifest"]).resolve()
            if not manifest.is_relative_to(out.resolve()):
                raise RuntimeError("rw_hydrofabric prepare manifest escapes the requested output")
            if not manifest.is_file():
                raise RuntimeError(f"rw_hydrofabric prepare wrote no manifest at {manifest}")
            receipt.write_text(json.dumps(prepared, indent=2, sort_keys=True) + "\n")
            stamp.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
            build["reused"] = False
    build["manifest"] = str(manifest)
    build["seconds"] = time.perf_counter() - started
    composed = {"enabled": True, "realization": row["realization"],
                "settings_bindings": {decision["manifest_binding"]: str(manifest)},
                "activation_reason": "fabric_covered"}
    if row.get("preparation"):
        from .provider import composition
        composed.update(composition(manifest=manifest, footprint=footprint))
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out / "hydro-door.toml"
    settings_path.write_text(to_toml(composed))
    return composed, settings_path, build


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("hydro door settings must be finite")
        return repr(value)
    if isinstance(value, Mapping):
        return "{ " + ", ".join(json.dumps(str(k)) + " = " + _toml_value(v)
                                 for k, v in value.items()) + " }"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value))


def to_toml(settings):
    """Flat [hydro] table plus one level of subtables; enough for door output."""
    lines, tables = ["[hydro]"], []
    for key, value in settings.items():
        if isinstance(value, Mapping):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    for key, table in tables:
        lines.append(f"\n[hydro.{key}]")
        lines.extend(f"{name} = {_toml_value(item)}" for name, item in table.items())
    return "\n".join(lines) + "\n"


# -- receipts and process hand-off --------------------------------------------

def public(decision):
    """Receipt form: drop the internal row reference."""
    if decision is None:
        return None
    return {key: deepcopy(value) for key, value in decision.items() if not key.startswith("_")}


def publish_decision(decision):
    os.environ[ENV_DECISION] = json.dumps(public(decision), sort_keys=True)


def inherited_decision():
    text = os.environ.get(ENV_DECISION)
    return None if not text else json.loads(text)


def resolve_launch(settings, *, requested, config_path, table=None):
    """Launcher entry: the decision, the settings to run, and the settings file.

    Returns (settings, settings_path_or_None, decision). Explicit ``--hydro``
    without a covering fabric raises HydroDoorRefusal naming every gap.
    """
    decision = decide(settings=settings, requested=requested, config_path=config_path,
                      table=table)
    if decision["reason"] == "fabric_covered":
        composed, path, build = materialize(decision, config_path=config_path)
        decision["build"] = build
        merged = dict(settings or {})
        merged.update(composed)
        return merged, path, decision
    if requested == "" and not decision["enabled"] and decision["reason"] != "cli_off":
        raise HydroDoorRefusal(
            "--hydro was requested but no hydrofabric covers this domain: "
            + "; ".join(f"[{gap['owner']}] {gap['gap']}" for gap in decision["gaps"]))
    # The decision travels in GPUWM_HYDRO_DOOR_DECISION_JSON, never inside the
    # settings: settings feed the realization identity sealed in checkpoints.
    settings = dict(settings or {})
    if not decision["enabled"] and not explicitly_prepared(settings):
        settings["enabled"] = False
        settings["activation_reason"] = decision["reason"]
    return settings, None, decision
