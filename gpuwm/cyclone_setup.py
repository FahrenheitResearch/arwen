"""Author a GFS 12/3 km moving-nest setup from an explicitly selected center.

The map request reads only published f000 MSLP and wind fields. Configuration
creation reuses the ordinary domain author, physics suite and vortex tracker;
neither entry point starts a forecast or computes a new tracking algorithm.
"""
from __future__ import annotations

import contextlib
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

SCHEMA = "arwen.cyclone-setup.v1"
ROOT_DIMS = (200, 160)
CHILD_DIMS = (160, 160)
ROOT_DX_M = 12000.0
RATIO = 4


def _cycle(raw: str, *, latest: bool = False) -> datetime:
    from gpuwm import domain_wizard as dw
    text = raw.strip()
    if text.lower() == "latest" and not latest:
        raise ValueError("Select a center on a resolved GFS map first; creation requires that map's exact cycle")
    if len(text) == 10 and text.isdecimal():
        text = datetime.strptime(text, "%Y%m%d%H").strftime("%Y-%m-%dT%H")
    result = dw._resolve_cycle(text, source="gfs", hours=0)
    if result.hour not in (0, 6, 12, 18) or result.minute or result.second:
        raise ValueError("GFS initialization must name a published 00/06/12/18 UTC cycle")
    return result


def latest_map(cycle: str = "latest") -> dict:
    moment = _cycle(cycle, latest=True)
    return {
        "schema": SCHEMA, "kind": "map", "cycle": moment.strftime("%Y%m%d%H"),
        "map_request": {"source": "gfs", "date": moment.strftime("%Y-%m-%d"),
                        "hour": moment.hour, "forecast_hour": 0, "member": 0,
                        "product": "mslp_10m_winds", "bounds": [-85., -180., 85., 180.]},
        "forecast_started": False,
        "selection": "Click the circulation center on this exact GFS f000 pressure-and-wind map.",
    }


def configuration_text(*, cycle: str, point: tuple[float, float], hours: int = 6,
                       name: str = "GFS cyclone 12 km to 3 km", tiles: str = "auto",
                       source: str = "cyclone-setup.toml") -> tuple[str, object]:
    from gpuwm import domain_wizard as dw
    from gpuwm.companion_domains import VORTEX_PRESET, VORTEX_PRESET_SOURCE
    from gpuwm.starter_template import render_tables

    moment = _cycle(cycle)
    if type(hours) is not int or not 1 <= hours <= 384:
        raise ValueError("Cyclone duration must be an integer from 1 to 384 hours")
    if tiles not in ("off", "auto", "on"):
        raise ValueError("Tile mode must be off, auto or on")
    lat, lon = point
    if not all(math.isfinite(v) for v in point) or not -85 <= lat <= 85 or not -180 <= lon <= 180:
        raise ValueError("Select a finite center on the displayed GFS map")
    projection = dw._projection_entries(lat, lon, "auto")
    area = dw.fetch_area_hint(projection, *ROOT_DIMS, source="gfs", root_dx_m=ROOT_DX_M)
    profile = dw.resolved_physics_profile("gfs", None)
    text = dw.render_config(
        name=name, start_time=moment, hours=hours, projection=projection,
        dims=[ROOT_DIMS, CHILD_DIMS], ratios=(RATIO,), root_dx_m=ROOT_DX_M,
        profile=profile, cumulus_requested=False, tiles=tiles,
        fetch_hints={"source": "gfs", "cycle": moment.strftime("%Y-%m-%dT%H"),
                     "hours": max(3, math.ceil(hours / 3) * 3), "cadence": 3,
                     "area": area, "out": f"data/gfs-cyclone-{moment:%Y%m%d%H}"},
        case_data=None, history_interval_s=3600., nest_history_interval_s=900.)
    raw = tomllib.loads(text)
    child = next(row for row in raw["domain"] if row["grid_id"] == 2)
    child["follow"] = dict(VORTEX_PRESET)
    # This is one immediate following nest. Spawn/retire decisions are not part
    # of this quick-start; the chosen center is its initial registration.
    text = ("# GFS cyclone quick-start: 12 km parent and 3 km following nest.\n"
            "# Center and cycle were selected explicitly on a GFS f000 map.\n"
            f"# Existing vortex-lock preset: {VORTEX_PRESET_SOURCE}\n"
            "# Following uses the 850 hPa circulation; the selection map uses MSLP.\n"
            + render_tables(raw))
    experiment = dw.experiment_from_text(text, source=source)
    return text, experiment


def plan_cyclone(*, cycle: str, point: tuple[float, float], sizing, target_machine=None,
                 hours: int = 6, name: str = "GFS cyclone 12 km to 3 km",
                 tiles: str = "auto", source: str = "cyclone-setup.toml") -> dict:
    from gpuwm import domain_wizard as dw
    from gpuwm.companion_domains import VORTEX_PRESET, VORTEX_PRESET_SOURCE
    from gpuwm.configuration_recovery import MemoryAdmissionError

    text, experiment = configuration_text(cycle=cycle, point=point, hours=hours,
                                           name=name, tiles=tiles, source=source)
    phases = dw._sizing_phases(
        experiment, free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
        source="gfs", machine=target_machine, profile=sizing.device_profile,
        forcing_interval_seconds=10800.)
    budget = dw.sizing_budget_bytes(experiment, free_bytes=sizing.free_bytes,
        vram_gib=sizing.vram_gib, profile=sizing.device_profile,
        forcing_interval_seconds=10800.)
    if phases.peak_envelope_bytes > budget:
        raise MemoryAdmissionError(
            "The 12/3 km cyclone setup exceeds the selected computer's memory budget: "
            + phases.verdict(budget), peak_envelope_bytes=phases.peak_envelope_bytes,
            budget_bytes=budget, binding_phase=phases.binding_phase)
    return {
        "schema": SCHEMA, "kind": "configuration", "cycle": _cycle(cycle).strftime("%Y%m%d%H"),
        "source": "gfs", "hours": hours, "point": list(point), "tiles": tiles,
        "domains": [{"grid_id": d.grid_id, "parent_id": d.parent_id,
                     "nx": d.run.nx, "ny": d.run.ny, "nz": d.run.nz,
                     "dx_m": d.run.dx, "dy_m": d.run.dy,
                     "following": d.grid_id == 2} for d in experiment.domains],
        "follow": dict(VORTEX_PRESET), "follow_preset_source": VORTEX_PRESET_SOURCE,
        "profile": dw.resolved_physics_profile("gfs", None),
        "memory": {"peak_envelope_bytes": phases.peak_envelope_bytes, "budget_bytes": budget,
                   "binding_phase": phases.binding_phase, "free_bytes": sizing.free_bytes,
                   "sizing_basis": "measured-available" if sizing.measured else "declared-capacity"},
        "config_text": text, "forecast_started": False,
    }


def main(args) -> int:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.latest_map:
                result = latest_map(args.cycle)
            else:
                from gpuwm import domain_wizard as dw
                from gpuwm.companion_query import inspect_configuration
                from gpuwm.hrrr_prepared_bundle import render_wps_namelist
                from gpuwm.starter_template import _publish_new_files
                if args.point is None:
                    raise ValueError("Select the cyclone center on the resolved GFS map first")
                _cycle(args.cycle)
                sizing, machine, _ = dw._domain_target_hardware(args)
                out = args.out.expanduser().resolve() if args.out else None
                result = plan_cyclone(cycle=args.cycle, point=dw._parse_point(args.point),
                    hours=args.hours, name=args.name, tiles=args.tiles, sizing=sizing,
                    target_machine=machine, source=str(out or "cyclone-setup.toml"))
                if out is not None:
                    if out.suffix.lower() != ".toml":
                        raise ValueError("Save the cyclone configuration as a new .toml file")
                    text = result["config_text"]
                    experiment = dw.experiment_from_text(text, source=str(out))
                    wps = out.with_suffix(".namelist.wps")
                    receipt = out.with_suffix(".cyclone.json")
                    if any(path.exists() for path in (out, wps, receipt)):
                        raise ValueError("Choose a new output path; cyclone setup never overwrites an existing configuration")
                    wps_text = render_wps_namelist(experiment).replace(
                        " interval_seconds = 3600,", " interval_seconds = 10800,")
                    proof = {key: value for key, value in result.items() if key != "config_text"}
                    proof.update(output=str(out), output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                 wps_sha256=hashlib.sha256(wps_text.encode()).hexdigest())
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _publish_new_files(((wps, wps_text),
                        (receipt, json.dumps(proof, indent=2, allow_nan=False) + "\n"), (out, text)))
                    result["configuration"] = inspect_configuration(out)
                    result.update(config_path=str(out), receipt_path=str(receipt))
        print(json.dumps(result, allow_nan=False, default=str))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"schema": SCHEMA, "error": str(error), "forecast_started": False}))
        return 1


def register_cli(subparsers):
    from gpuwm.domain_wizard import CARD_VRAM_GIB
    parser = subparsers.add_parser("cyclone-setup", help="select a GFS f000 cyclone and author a 12/3 km following nest")
    parser.add_argument("--latest-map", action="store_true")
    parser.add_argument("--cycle", default="latest")
    parser.add_argument("--point")
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--name", default="GFS cyclone 12 km to 3 km")
    parser.add_argument("--tiles", choices=("off", "auto", "on"), default="auto")
    parser.add_argument("--hardware-json", type=Path)
    parser.add_argument("--target-host-memory-json", type=Path)
    parser.add_argument("--vram-gib", type=float)
    parser.add_argument("--card", choices=sorted(CARD_VRAM_GIB))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true", help="emit the JSON result (the default)")
    parser.set_defaults(func=main)
