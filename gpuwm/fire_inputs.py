"""Public fire input acquisition. Raster decoding and transforms stay in Rust."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

AUTHORITY = Path(__file__).with_name("authorities")

def canopy_fields():
    return json.loads((AUTHORITY / "sfire-input-fields.v1.json").read_text())["fields"]

def region_for_bounds(bounds, region_id=None):
    rows = json.loads((AUTHORITY / "sfire-regions.v1.json").read_text())["regions"]
    west, south, east, north = bounds
    for row in rows:
        if region_id is not None and row["id"] != region_id:
            continue
        w, s, e, n = row["bounds"]
        if w <= west < east <= e and s <= south < north <= n:
            return row
    raise ValueError(f"No fire input region covers {bounds} with region={region_id!r}")

def prepare_sources(config, bounds, base, cache, acquire, routes):
    """Resolve metadata rows, then fetch independent opaque rasters concurrently."""
    options = config.get("sources", {})
    region = region_for_bounds(bounds, options.get("region"))
    plan = {role: options.get(role, region[role]) for role in ("terrain", "fuel")}
    bound = {role: config[role] for role in ("terrain", "fuel") if role in config}
    supplied = config.get("canopy", {})
    # A captured legacy two-raster request remains two rasters. Public LANDFIRE
    # acquisition takes all four canopy layers by default, from the same year.
    canopy = dict(region["canopy"]) if "fuel" not in bound else {}
    canopy.update(options.get("canopy", {}))
    canopy.update({name: None for name in supplied})
    if canopy and set(canopy) != set(canopy_fields()):
        raise ValueError("Fire canopy inputs require cover, height, base height and bulk density together")
    for name, source in canopy.items():
        role = canopy_fields()[name]["role"]
        plan[name] = source
        if name in supplied:
            bound[name] = supplied[name]
        elif source is None or source not in routes or routes[source]["role"] != role:
            raise ValueError(f"Fire source {source!r} cannot supply {name}")
    accepted = options.get("accept_licenses", [])
    for role, source in plan.items():
        if role in bound:
            continue
        if source is None:
            raise ValueError(f"Fire region {region['id']} has no admitted {role} route; supply a provenance-bound public raster with path and sha256")
        if source not in routes or routes[source]["role"] != (role if role in ("fuel", "terrain") else canopy_fields()[role]["role"]):
            raise ValueError(f"Fire source {source!r} cannot supply {role}")
    # Validate complete coverage before reporting a licence gate, so the
    # first refusal cannot imply that accepting a DEM licence admits fuel.
    for role, source in plan.items():
        if role in bound:
            continue
        row = routes[source]
        if row.get("requires_license_acceptance") and row["license"] not in accepted:
            raise ValueError(f"Fire source {source} requires sources.accept_licenses to name {row['license']}: {row['license_url']}")
    def obtain(item):
        role, source = item
        started = time.monotonic()
        if role in bound:
            spec = dict(bound[role]); path = Path(spec["path"])
            spec["path"] = str((path if path.is_absolute() else base / path).resolve())
            metadata = dict(path=spec["path"], sha256=spec["sha256"])
        else:
            spec, metadata = acquire(source, int(options.get("fuel_year", 2024)), bounds, cache)
        metadata["wall_seconds"] = time.monotonic() - started
        return role, spec, metadata
    started = time.monotonic()
    workers = min(6, len(plan))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(obtain, plan.items()))
    inputs = {role: spec for role, spec, _ in results}
    metadata = {role: receipt for role, _, receipt in results}
    return inputs, metadata, dict(region=region["id"], acquisition_workers=workers,
                                  acquisition_wall_seconds=time.monotonic() - started)


def source_provenance(receipts):
    """Keep source identity and notices independent of fetch timing and cache state."""
    if isinstance(receipts, dict):
        return {key: source_provenance(value) for key, value in receipts.items()
                if key not in {"wall_seconds", "cache_hit", "fetched_utc"}}
    if isinstance(receipts, list):
        return [source_provenance(value) for value in receipts]
    return receipts
