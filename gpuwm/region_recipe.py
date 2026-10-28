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
    """The row whose ``id`` or one of whose declared ``aliases`` is ``name``.

    A row's aliases are part of its metadata, so every door that takes a
    row name accepts them (``gpuwm domain --recipe hrrr`` refused the
    ``hrrr`` alias the recipe registry lists for ``hrrr_v4_gsd41``).
    """
    rows = table[key]
    if isinstance(rows, dict):
        if name in rows:
            return rows[name]
        rows = [{"id": ident, **row} for ident, row in rows.items() if isinstance(row, dict)]
    for row in rows:
        if row.get("id") == name:
            return row
    for row in rows:
        if name in row.get("aliases", ()):
            return row
    raise ValueError(f"{key} table has no row or alias {name!r}; missing metadata cannot define a grid or its forcing")


def recipe_row(name, recipe_table=None):
    """The packaged recipe registry row for an id or a declared alias."""
    return _row(_read_table(Path(recipe_table or RECIPE_TABLE)), "recipes", name)


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


def recipe_row_defaults(recipe, recipe_table=None):
    """The packaged row's declared default region and spacing, or None.

    A local TOML or a row without both declarations has no defaults.
    """
    if Path(recipe).is_file():
        return None
    row = recipe_row(recipe, recipe_table)
    region, spacing = row.get("default_region"), row.get("default_spacing_km")
    if region is None or spacing is None:
        return None
    return {"region": str(region), "spacing_km": float(spacing)}


def emit_recipe(*, recipe, region, spacing_km, source, cycle, hours,
                recipe_table=None, region_table=None, buffer_km=0.0,
                projection="auto", history_interval_s=None, name=None, cadence=None,
                smoke=False, forecast_start_hour=0):
    from datetime import timedelta

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
        recipe = row["id"]
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
    # --forecast-start-hour K: time zero is the cycle's fK lead and the
    # boundaries come from f(K+i), as the wizard door writes it.  --recipe
    # used to refuse the lead, so a recipe run could only start at a cycle.
    start_hour = int(forecast_start_hour or 0)
    lead_refusal = wizard.forecast_start_hour_refusal(source, start_hour, hours)
    if lead_refusal:
        raise ValueError(lead_refusal)
    cycle_time = wizard._resolve_cycle(cycle, source=source, hours=hours,
                                       start_hour=start_hour)
    if start_hour and source == "hrrr":
        from gpuwm.hrrr_forecast import hrrr_source_window
        hrrr_source_window(cycle=cycle_time, start_hour=start_hour,
                           run_seconds=float(hours) * 3600.0)
    start = cycle_time + timedelta(hours=start_hour)
    # A model-level source's native top moves with its upgrades (HRRR: 20 hPa
    # through v3, 15 hPa from v4), so a recipe written for one era can ask a
    # cycle from another for levels its files do not hold.  The emitted top
    # is the higher of the recipe's and the cycle era's (a table row on the
    # source, gpuwm.source_adapters.source_top_pa_for_cycle), decided here
    # before any download, and the receipt and output say so.  Breakage:
    # hrrr_v4_gsd41 (p_top 1500 Pa) on a pre-v4 HRRR cycle fetched 26 GB and
    # then refused at root_initialize ("target pressure lies above source
    # top"); the same day prepares at 2000 Pa.
    from gpuwm.source_adapters import source_top_pa_for_cycle
    recipe_top = document["shared"].get("p_top")
    recipe_top = None if recipe_top is None else float(recipe_top)
    era_top = source_top_pa_for_cycle(source, cycle_time)
    model_top = {"recipe_p_top_pa": recipe_top, "source_era_top_pa": era_top,
                 "emitted_p_top_pa": recipe_top, "raised": False}
    if era_top is not None and recipe_top is not None and recipe_top < era_top:
        document["shared"]["p_top"] = era_top
        model_top.update(emitted_p_top_pa=era_top, raised=True, reason=(
            f"{source} cycle {cycle_time:%Y-%m-%dT%H} reaches only "
            f"{era_top:g} Pa ({era_top / 100:g} hPa); the recipe's "
            f"{recipe_top:g} Pa top would be extrapolated past the analysis, "
            f"so p_top is {era_top:g} Pa"))
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
    if smoke:
        from gpuwm.smoke_execution import apply_smoke_settings
        apply_smoke_settings(document)
    # A full recipe's boundary cadence is part of its requested forcing.
    # The ordinary fetch planner refuses a cadence unsupported by the source.
    from gpuwm.fetch import fetch_accepts_area, fetch_accepts_cadence
    if cadence is None and fetch_accepts_cadence(source):
        cadence = document.get("fetch", {}).get("cadence")
    spacing, length = wizard.fetch_window(source, hours, start_hour, cadence=cadence)
    document["fetch"] = {"source": source, "cycle": cycle_time.strftime("%Y-%m-%dT%H"),
        "hours": length}
    if start_hour:
        document["fetch"]["forecast_start_hour"] = start_hour
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
        "forecast_start_hour": start_hour,
        "model_top": model_top,
        "aerosol_candidate": selected, "aerosol_rejected": rejected,
        "aerosol_actual_use": "recorded by the preparation and forecast aerosol_initialization receipt",
        "native_static_set_aside": native_static,
        "hrrr_smoke": ("on (--smoke)" if smoke else
                       document["shared"].get("smoke_off_reason") or None),
        "config_sha256": hashlib.sha256(text.encode()).hexdigest()}
    return text, receipt


def domain_recipe_main(args):
    from gpuwm import domain_wizard as wizard
    if getattr(args, "point", None) is not None:
        raise ValueError("--recipe requires --region or --polygon; a point alone cannot define a country extent")
    region = getattr(args, "region", None) or getattr(args, "polygon", None)
    spacing_km = args.spacing_km if args.spacing_km is not None else args.root_dx
    row_defaults = None
    if not region and spacing_km is None:
        # A bare --recipe means the recipe on its own declared layout: the
        # packaged row names that region and spacing as data, and the output
        # and receipt say they were used.  A requested region without a
        # spacing still refuses below.
        row_defaults = recipe_row_defaults(args.recipe, args.recipe_table)
        if row_defaults is None:
            raise ValueError("--recipe requires --region or --polygon and --spacing-km; this recipe declares no default region and spacing")
        region, spacing_km = row_defaults["region"], row_defaults["spacing_km"]
    if not region:
        raise ValueError("--recipe requires --region or --polygon; a point alone cannot define a country extent")
    if spacing_km is None:
        raise ValueError("--recipe requires --spacing-km; the original recipe grid spacing must not silently become the requested grid")
    # Refuse competing settings rather than silently discarding a wizard request.
    for key in ("physics_profile", "physics_choices", "nz", "chain", "ladder", "member"):
        if getattr(args, key, None) is not None:
            raise ValueError(f"--recipe cannot also select --{key.replace('_', '-')}; competing metadata would change the complete recipe")
    buffers = wizard.parse_level_buffers(getattr(args, "buffer_km", None)) or (0.0,)
    if len(buffers) != 1:
        raise ValueError("a root recipe needs exactly one polygon buffer")
    text, receipt = emit_recipe(recipe=args.recipe, region=str(region),
        spacing_km=spacing_km,
        source=args.source if getattr(args, "source_explicit", False) else None, cycle=args.cycle,
        hours=args.hours, recipe_table=args.recipe_table, region_table=args.region_table,
        buffer_km=buffers[0], projection=args.projection,
        history_interval_s=args.history_interval, name=args.name, cadence=args.cadence,
        smoke=bool(getattr(args, "smoke", False)),
        forecast_start_hour=getattr(args, "forecast_start_hour", None) or 0)
    if row_defaults is not None:
        receipt["defaults_from_recipe_row"] = row_defaults
        print(f"--recipe {args.recipe}: no --region/--spacing-km given; using the recipe row's "
              f"declared region {row_defaults['region']} at {row_defaults['spacing_km']:g} km")
    if receipt["model_top"].get("raised"):
        print(f"--recipe {args.recipe}: model top raised: {receipt['model_top']['reason']}")
    wizard._write_atomic(args.out, text)
    receipt_path = args.out.with_suffix(".recipe.json")
    wizard._write_atomic(receipt_path, json.dumps(receipt, indent=2) + "\n")
    print(f"Wrote {args.out} and {receipt_path}; complete recipe, shape {receipt['shape']}, {receipt['spacing_km']:g} km")
    print(f"Next: gpuwm check {args.out}; gpuwm go {args.out}")
    return 0
