"""Put a complete metadata recipe on a polygon through the ordinary domain door."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import tomllib

DATA_ROOT = Path(__file__).parent / "data"
RECIPE_TABLE = DATA_ROOT / "recipes" / "registry.v1.json"
REGION_TABLE = DATA_ROOT / "nwpsuite" / "regions.v1.json"


def _read_table(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _row(table, key, name):
    rows = table[key]
    if isinstance(rows, dict):
        if name in rows:
            return rows[name]
    else:
        for row in rows:
            if row.get("id") == name:
                return row
    raise ValueError(f"{key} table has no row {name!r}; missing metadata cannot define a grid or its forcing")


def _region_polygon(value, table_path):
    """Resolve suite polygon references and inline GeoJSON through metadata."""
    if isinstance(value, str):
        references = _read_table(Path(table_path).parent / "references.v1.json")
        value = _row(references["registries"], "polygons", value)
    if "bounds" in value:
        west, south, east, north = value["bounds"]
        return {"type": "Polygon", "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south]]]}
    return value


def select_aerosol(policy, source):
    """Resolve only declared availability, never invent zero or synthetic aerosol.

    Analysed candidates still require the complete pair on every actual frame at
    ingest. Climatology still requires its dataset at ingest. A table cannot
    certify files it has not read, or switch source halfway through a forecast.
    """
    rejected = []
    for row in policy:
        providers = row.get("providers")
        if providers is not None and source not in providers:
            rejected.append({"id": row["id"], "reason": "driving route does not declare the analysed pair"})
            continue
        return copy.deepcopy(row), rejected
    raise ValueError("aerosol_source_unavailable: no declared candidate supplies both QNWFA and QNIFA for initial and boundary frames; zero aerosol would change droplet activation")


def emit_recipe(*, recipe, region, spacing_km, source, cycle, hours,
                recipe_table=None, region_table=None, buffer_km=0.0,
                projection="auto", history_interval_s=None, name=None, cadence=None):
    from gpuwm import domain_wizard as wizard
    spacing_km = float(spacing_km)
    if not math.isfinite(spacing_km) or not wizard.MIN_ROOT_DX_KM <= spacing_km <= wizard.MAX_ROOT_DX_KM:
        raise ValueError("spacing_km must be finite and within the domain stencil spacing limits")
    if not math.isfinite(float(buffer_km)) or float(buffer_km) < 0:
        raise ValueError("polygon buffer must be finite and nonnegative; a negative buffer would exclude requested territory")
    recipe_table = Path(recipe_table or RECIPE_TABLE)
    registry = _read_table(recipe_table)
    local_recipe = Path(recipe)
    if local_recipe.is_file():
        recipe_path = local_recipe
        policy = registry.get("default_aerosol_sources", [])
        default_source = registry.get("default_source")
    else:
        row = _row(registry, "recipes", recipe)
        recipe_path = recipe_table.parent / row["config"]
        policy = row.get("aerosol_sources", [])
        default_source = row.get("default_source")
    payload = recipe_path.read_bytes()
    document = tomllib.loads(payload.decode("utf-8"))
    if len(document.get("domain", [])) != 1:
        raise ValueError("recipe_geometry_ambiguous: a polygon emission requires one root recipe domain; a nested layout needs declared refinement metadata")
    local_region = Path(region)
    if local_region.is_file():
        polygon = _read_table(local_region)
    else:
        table_path = Path(region_table or REGION_TABLE)
        region_row = _row(_read_table(table_path), "regions", region)
        polygon = _region_polygon(region_row["polygon"], table_path)
        declared_sources = region_row.get("sources", [])
        if isinstance(declared_sources, dict):
            declared_sources = declared_sources.get("global_ic", declared_sources.get("regional_ic", []))
        if declared_sources:
            default_source = declared_sources[0]
    source = wizard.resolve_source(source or default_source or document.get("fetch", {}).get("source"))
    # Reuse the ordinary polygon validator and projection operators. The only
    # temporary file is metadata, and it never contains input model data.
    with tempfile.TemporaryDirectory(prefix="gpuwm-region-") as scratch:
        path = Path(scratch) / "polygon.json"
        path.write_text(json.dumps(polygon), encoding="utf-8")
        footprint = wizard.load_polygon_footprint(path)
    projected = wizard._projection_entries(footprint.center_lat, footprint.center_lon, projection)
    dx = spacing_km * 1000.0
    dims = wizard.polygon_ladder_dims(footprint=footprint, projection=projected,
        ratios=(), buffers_km=(float(buffer_km),), root_dx_m=dx,
        minimum_axis=max(24, 2 * int(document["experiment"].get("spec_bdy_width", 5)) + 4))
    nx, ny = dims[0]
    wizard._pole_clearance_refusal(projected, nx, ny, dx, "--polygon")
    for refusal in (wizard.source_coverage_refusal, wizard.fetch_crop_refusal):
        reason = refusal(projected, nx, ny, source=source, root_dx_m=dx, target_option="--polygon")
        if reason:
            # Named so a table row that does not fit its source reads as the
            # row and the source, not as a bare grid.
            raise ValueError(f"region {region!r} with source {source}: {reason}")
    start = wizard._resolve_cycle(cycle, source=source, hours=hours)
    # A recipe's [static] source binds published native-grid fields to the
    # recipe's OWN grid (hrrr_v4_gsd41: the HRRR CONUS v4 statics).  This door
    # lays out a new grid on a new projection, where that source refuses
    # ("the case grid's projection differs from the source's"), so the
    # emitted configuration takes its geography from the WPS datasets like
    # any other new domain, and the receipt names the binding set aside.
    native_static = document.pop("static", None)
    document["projection"] = projected
    document["shared"]["map_proj"] = {"lambert": 1, "polar": 2, "mercator": 3}[projected["map_proj"]]
    document["experiment"].update(name=name or "region-recipe", start_time=start, run_seconds=float(hours) * 3600)
    document["domain"][0].update(nx=nx, ny=ny, dx=dx)
    if history_interval_s is not None:
        document["domain"][0]["history_interval_s"] = float(history_interval_s)
    selected = None
    rejected = []
    if int(document["shared"].get("mp_physics", 0)) == 28:
        selected, rejected = select_aerosol(policy, source)
        document["shared"].update(selected["settings"])
    # A full recipe's boundary cadence is part of its requested forcing.
    # The ordinary fetch planner refuses a cadence unsupported by the source.
    from gpuwm.fetch import fetch_accepts_area, fetch_accepts_cadence
    if cadence is None and fetch_accepts_cadence(source):
        cadence = document.get("fetch", {}).get("cadence")
    spacing, length = wizard.fetch_window(source, hours, cadence=cadence)
    document["fetch"] = {"source": source, "cycle": start.strftime("%Y-%m-%dT%H"),
        "hours": length}
    # Whole-object routes crop at preparation, so an acquisition crop would
    # make the emitted configuration refuse before it reaches that stage.
    if fetch_accepts_area(source):
        document["fetch"]["area"] = wizard.fetch_area_hint(
            projected, nx, ny, source=source, root_dx_m=dx, target_option="--polygon")
    if spacing is not None:
        document["fetch"]["cadence"] = spacing
    parts = []
    for key, value in document.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            parts.extend(wizard._render_table(key, row, array_of_tables=True) for row in value)
        else:
            parts.append(wizard._render_table(key, value))
    text = "\n".join(parts)
    exp = wizard.experiment_from_text(text, source=str(recipe_path))
    wizard.verify_polygon_containment(exp, footprint, (float(buffer_km),))
    receipt = {"schema": "gpuwm.domain-recipe.v1", "recipe": str(recipe),
        "recipe_sha256": hashlib.sha256(payload).hexdigest(),
        "polygon_sha256": hashlib.sha256(json.dumps(polygon, sort_keys=True).encode()).hexdigest(),
        "region": str(region), "shape": [nx, ny, document["shared"].get("nz")],
        "spacing_km": spacing_km, "projection": projected, "forcing_source": source,
        "aerosol_candidate": selected, "aerosol_rejected": rejected,
        "aerosol_actual_use": "recorded by the preparation and forecast aerosol_initialization receipt",
        "native_static_set_aside": native_static,
        "config_sha256": hashlib.sha256(text.encode()).hexdigest()}
    return text, receipt


def domain_recipe_main(args):
    from gpuwm import domain_wizard as wizard
    region = getattr(args, "region", None) or getattr(args, "polygon", None)
    if not region:
        raise ValueError("--recipe requires --region or --polygon; a point alone cannot define a country extent")
    if args.spacing_km is None and args.root_dx is None:
        raise ValueError("--recipe requires --spacing-km; the original recipe grid spacing must not silently become the requested grid")
    # Refuse competing settings rather than silently discarding a wizard request.
    for key in ("physics_profile", "physics_choices", "nz", "chain", "ladder", "forecast_start_hour", "member"):
        if getattr(args, key, None) is not None:
            raise ValueError(f"--recipe cannot also select --{key.replace('_', '-')}; competing metadata would change the complete recipe")
    buffers = wizard.parse_level_buffers(getattr(args, "buffer_km", None)) or (0.0,)
    if len(buffers) != 1:
        raise ValueError("a root recipe needs exactly one polygon buffer")
    text, receipt = emit_recipe(recipe=args.recipe, region=str(region),
        spacing_km=args.spacing_km or args.root_dx,
        source=args.source if getattr(args, "source_explicit", False) else None, cycle=args.cycle,
        hours=args.hours, recipe_table=args.recipe_table, region_table=args.region_table,
        buffer_km=buffers[0], projection=args.projection,
        history_interval_s=args.history_interval, name=args.name, cadence=args.cadence)
    wizard._write_atomic(args.out, text)
    receipt_path = args.out.with_suffix(".recipe.json")
    wizard._write_atomic(receipt_path, json.dumps(receipt, indent=2) + "\n")
    print(f"Wrote {args.out} and {receipt_path}; complete recipe, shape {receipt['shape']}, {receipt['spacing_km']:g} km")
    print(f"Next: gpuwm check {args.out}; gpuwm go {args.out}")
    return 0
