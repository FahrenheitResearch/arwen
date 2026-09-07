"""Visual launchpad orchestration; the CLI and shared config remain authoritative.

This module performs no decoding, interpolation, fitting or forecast arithmetic.
Draft creation delegates to the existing domain wizard, and launches use the
durable CLI worker. The JSON protocol is local to the Rust launchpad.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys
import tomllib


def inspect_text(text, path=None):
    raw = tomllib.loads(text)
    result = {"document": raw, "name": raw.get("experiment", {}).get("name", "Imported forecast"),
              "domains": len(raw.get("domain", []))}
    if path is not None:
        from gpuwm.experiment import build_experiment_from_config_tables
        from gpuwm.static.projection import grids_from_projection_config
        import numpy as np
        source = Path(path)
        try:
            exp = build_experiment_from_config_tables(raw, source=str(source), base_dir=source.parent)
            result["timesteps"] = [str(exp.dt_exact(d.grid_id)) for d in exp.domains]
            result["grid_geometry"] = [
                {"grid_id": d.grid_id, "nx": d.run.nx, "ny": d.run.ny,
                 "dx": d.run.dx, "dy": d.run.dy} for d in exp.domains]
            result["footprints"] = []
            for grid in grids_from_projection_config(exp):
                # Actual projected cell-edge boundary, sampled for the display.
                x=np.linspace(.5,grid.e_we-.5,33); y=np.linspace(.5,grid.e_sn-.5,33)
                ii=np.concatenate([x,np.full(33,x[-1]),x[::-1],np.full(33,x[0])])
                jj=np.concatenate([np.full(33,y[0]),y,np.full(33,y[-1]),y[::-1]])
                lat,lon=grid.ij_to_latlon(ii,jj)
                result["footprints"].append(np.column_stack([lon,lat]).tolist())
        except (ValueError, TypeError, KeyError) as error:
            result["validation_error"] = str(error)
    return result


def catalog():
    from gpuwm.source_adapters import source_adapters
    from gpuwm.domain_wizard import DEFAULT_WIZARD_SOURCE, source_has_fetch_front_door
    from gpuwm.physics_menu import shipped_profiles, profile_facts, default_profile_for
    from gpuwm.physics_registry import physics_registry
    from gpuwm.experiment import _DOMAIN_KEYS
    sources = [{"id": s.source_id, "family": s.file_family,
                "fetch": source_has_fetch_front_door(s.source_id),
                "runnable": s.runnable, "horizon": s.max_forecast_hour,
                "default_profile": default_profile_for(s.source_id) if s.runnable else None}
               for s in source_adapters()]
    profiles = [profile_facts(p) for p in shipped_profiles()]
    components = physics_registry().get("components", {})
    return {"sources": sources, "profiles": profiles, "components": components, "domain_keys": sorted(_DOMAIN_KEYS),
            "default_source": DEFAULT_WIZARD_SOURCE,
            "cycle": datetime.now(timezone.utc).replace(hour=(datetime.now(timezone.utc).hour // 6)*6,
                         minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H"),
            "version": "2.7", "engine": "ArWen native WPS / Rust preprocessing"}


def number(data, name, *, positive=False, integral=False):
    value = data[name]
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if integral and not value.is_integer():
        raise ValueError(f"{name} must be a whole number; the domain wizard accepts integer {name}")
    return int(value) if integral else value


def create_draft(data):
    """Ask the existing polygon wizard to author an exact native config pair."""
    from gpuwm.domain_wizard import register_cli, domain_main
    folder = Path(data["directory"]).resolve(strict=True)
    lat, lon = number(data, "lat"), number(data, "lon")
    if not -90 < lat < 90 or not -180 <= lon <= 180:
        raise ValueError("The geographic rectangle needs latitude strictly between the poles and longitude in [-180, 180]. Import a config for other existing projection workflows.")
    width, height = number(data, "width", positive=True), number(data, "height", positive=True)
    hours=number(data,"hours",positive=True,integral=True)
    nz=number(data,"nz",positive=True,integral=True)
    dx=number(data,"dx",positive=True)
    history=number(data,"history",positive=True)
    # Author a requested geographic rectangle; the existing wizard projects,
    # sizes and verifies actual containment. This is not a new grid fitter.
    half_lat = height / (2 * 111.32)
    half_lon = width / (2 * 111.32 * math.cos(math.radians(lat)))
    if abs(lat) + half_lat >= 90 or half_lon >= 180:
        raise ValueError("This requested rectangle reaches a pole or spans the whole globe. Reduce its extent.")
    wrap = lambda x: (x + 180) % 360 - 180
    west, east = wrap(lon-half_lon), wrap(lon+half_lon)
    south, north = lat-half_lat, lat+half_lat
    polygon = {"type": "Polygon", "coordinates": [[[west,south],[east,south],[east,north],[west,north],[west,south]]]}
    path = folder / "region.geojson"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(polygon, stream)
    config = folder / "forecast.toml"
    argv = ["domain", "--polygon", str(path), "--source", str(data["source"]),
            "--cycle", str(data["cycle"]), "--hours", str(hours),
            "--root-dx", str(dx),
            "--nz", str(nz),
            "--name", str(data["name"]), "--out", str(config),
            "--data-dir", str(folder / "forcing"),
            "--history-interval", str(history),
            "--nest-history-interval", str(history),
            "--tiles", str(data.get("tiles", "off"))]
    if str(data.get("chain", "")).strip():
        argv += ["--chain", str(data["chain"])]
    if data.get("profile"):
        argv += ["--physics-profile", str(data["profile"])]
    if data.get("vram"):
        argv += ["--vram-gib", str(number(data,"vram",positive=True))]
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command", required=True))
    captured = io.StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        result = domain_main(parser.parse_args(argv))
    if result not in (0, None):
        raise ValueError(captured.getvalue())
    text = config.read_text(encoding="utf-8")
    return {"path": str(config), "text": text, "inspection": inspect_text(text, config),
            "creation_log": captured.getvalue(), "command": ["gpuwm", *argv],
            "requested_region": {"width_km": width, "height_km": height}}


def external_plan(data):
    directory = Path(data["directory"]).resolve(strict=True)
    kind = data["kind"]
    if kind == "met-em":
        from gpuwm.metem_door import resolve_metem_run
        resolved = resolve_metem_run(directory)
    elif kind == "wrfinput":
        from gpuwm.wrfinput_door import resolve_wrfinput_run
        resolved = resolve_wrfinput_run(directory)
    else:
        raise ValueError("Choose WPS met_em or WRF real.exe output")
    exp = resolved.experiment
    return {"name": exp.name, "domains": len(exp.domains), "start": str(exp.start_time),
            "seconds": exp.run_seconds, "message": "Existing input metadata and producing settings resolved. No preparation or forecast was started."}


def main():
    payload = json.load(sys.stdin)
    try:
        action = payload.pop("action")
        if action == "catalog": result = catalog()
        elif action == "inspect": result = inspect_text(payload["text"],payload.get("path"))
        elif action == "create": result = create_draft(payload)
        elif action == "external-plan": result = external_plan(payload)
        else: raise ValueError("Unknown launchpad operation")
        print(json.dumps({"ok": True, "value": result}, default=str, allow_nan=False))
    except (Exception, SystemExit) as error:
        print(json.dumps({"ok": False, "error": str(error)}, allow_nan=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
