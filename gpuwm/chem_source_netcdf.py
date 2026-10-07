"""Table-driven hourly extensive fields from regular NetCDF latitude/longitude.

Rust decodes, normalizes declared no-fire sentinels, masks and remaps. Python
resolves hours and manages the cache. Cell masses never divide by model area;
map factors select partition resolution only. Receipts retain excluded mass.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError

from gpuwm import netcdf_bridge, obs_regrid_bridge as remap


class EmissionSourceError(ValueError):
    """A source contract cannot supply a trustworthy hourly emission frame."""


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def parse_listing(text, description):
    """Match full basenames, keeping the newest creation of each observed hour."""
    parser = _Links()
    parser.feed(text)
    pattern = re.compile(description["pattern"])
    found = {}
    for href in parser.links:
        if urlparse(href).scheme or "/" in href or "\\" in href:
            continue
        match = pattern.fullmatch(href)
        if not match:
            continue
        hour = datetime.strptime(match["hour"], description["hour_format"]).replace(tzinfo=timezone.utc)
        creation = match["creation"]
        if hour not in found or creation > found[hour][0]:
            found[hour] = (creation, href)
    return {hour: value[1] for hour, value in sorted(found.items())}


def _hour(hour):
    if hour.tzinfo is None:
        hour = hour.replace(tzinfo=timezone.utc)
    hour = hour.astimezone(timezone.utc)
    if hour.minute or hour.second or hour.microsecond:
        raise EmissionSourceError("emission hour must start on an hour boundary; otherwise a whole-hour mass would be assigned to the wrong interval")
    return hour


def listing(row, hour, cache, *, refresh=False):
    """Cache publisher indexes for the table's TTL, using the engine transport."""
    from gpuwm.fetch_routes import _download_object
    description = row.grid["listing"]
    url = _hour(hour).strftime(description["url"])
    if urlparse(url).scheme != "https":
        raise EmissionSourceError("emission listing requires HTTPS; unverified transport could substitute source filenames")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (sha256(url.encode()).hexdigest() + ".html")
    if refresh or not path.exists() or time.time()-path.stat().st_mtime > description["cache_s"]:
        with tempfile.TemporaryDirectory(dir=cache) as scratch:
            staged = Path(scratch) / "index.html"
            try:
                _download_object(url, staged, magic="")
            except HTTPError as error:
                if error.code == 404:
                    return url, {}
                raise
            parse_listing(staged.read_text(encoding="utf-8"), description)
            staged.replace(path)
    return url, parse_listing(path.read_text(encoding="utf-8"), description)


def newest_posted_hour(row, cache, *, now=None):
    """Inspect the current and preceding month across a month boundary."""
    now = _hour((now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0))
    previous = now.replace(day=1) - timedelta(days=1)
    hours = []
    for month in (now, previous):
        _, entries = listing(row, month, cache, refresh=True)
        hours.extend(entries)
    return max(hours) if hours else None


def resolve_hour(row, hour, entries):
    hour = _hour(hour)
    if hour not in entries:
        newest = max(entries).isoformat() if entries else "none"
        raise EmissionSourceError(f"{row.name}: missing observed_hourly file for {hour.isoformat()}; newest posted hour in this index is {newest}; using another hour would invent emissions")
    return entries[hour]


def fetch_hour(row, hour, cache):
    """Resolve the unpredictable creation stamp, download and pin first bytes."""
    from gpuwm.fetch_routes import _download_object
    hour = _hour(hour)
    base, entries = listing(row, hour, cache)
    if hour not in entries:
        base, entries = listing(row, hour, cache, refresh=True)
    name = resolve_hour(row, hour, entries)
    path = Path(cache) / name
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["sha256"] != file_sha256(path):
            raise EmissionSourceError(f"{name}: cached bytes differ from the first-fetch sha256; changed emissions cannot reuse this frame identity")
        return path, receipt
    with tempfile.TemporaryDirectory(dir=cache) as scratch:
        staged = Path(scratch) / name
        receipt = _download_object(urljoin(base, name), staged, magic="")
        with netcdf_bridge.open_dataset(staged) as dataset:
            _check_hour(dataset, row, hour)
        staged.replace(path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return path, receipt


def _check_hour(dataset, row, hour):
    if _hour(dataset.variables[row.grid["time_coordinate"]].times()[0]) != hour:
        raise EmissionSourceError("NetCDF time differs from the requested emission hour; filename-only identity would apply mass in the wrong interval")


#: Remap plans by (row, source grid, window, destination grid, split, bound).
#: A run ingests one file per hour on one fixed source grid and one model
#: grid, so the plan is built once and every later hour only applies it;
#: rebuilding it cost 5 s (3 km) to 13 s (750 m) per hour on node-1.
_PLAN_CACHE: dict = {}
_PLAN_CACHE_LIMIT = 4


def _window(source_lat, source_lon, latitude, longitude, max_distance_m):
    """Row and column slices of the regular source grid that can reach the model.

    The source is an axis-aligned latitude/longitude grid, so the cells whose
    sub-points can land within ``max_distance_m`` of a model cell lie inside
    the model's latitude and longitude bounds widened by that distance plus
    two source spacings.  Everything outside is counted as unreachable mass
    in the receipt instead of being carried through the plan: a RAVE file is
    6240 x 2610 cells, a regional model sees a few thousand of them, and an
    unwindowed 750 m plan held 16 M x 25 destination indices (3.3 GB) for
    every hour.  Returns ``None`` (no windowing) when the grid is not
    axis-aligned or the model spans 180 degrees or more of the source's
    longitude convention.
    """
    import numpy as np

    lat = np.asarray(source_lat, dtype=np.float64)
    lon = np.asarray(source_lon, dtype=np.float64)
    if lat.ndim != 2 or lat.shape != lon.shape or min(lat.shape) < 2:
        return None
    column_lat = lat[:, 0]
    row_lon = lon[0, :]
    if not (np.all(lat == column_lat[:, None]) and np.all(lon == row_lon[None, :])):
        return None
    spacing = max(abs(float(column_lat[1] - column_lat[0])),
                  abs(float(row_lon[1] - row_lon[0])))
    margin_lat = float(np.degrees(max_distance_m / 6371229.0)) + 2.0 * spacing
    dest_lat = np.asarray(latitude, dtype=np.float64)
    dest_lon = np.asarray(longitude, dtype=np.float64)
    lat_lo = float(dest_lat.min()) - margin_lat
    lat_hi = float(dest_lat.max()) + margin_lat
    coslat = max(float(np.cos(np.radians(min(89.0, max(abs(lat_lo), abs(lat_hi)))))), 1e-3)
    margin_lon = margin_lat / coslat
    # Model longitudes in the source's own convention (RAVE: 0..360 east),
    # unwrapped around the middle of the source's longitude span so a model
    # edge just west of the first source column stays west of it.
    centre = 0.5 * (float(row_lon.min()) + float(row_lon.max()))
    shifted = (dest_lon - centre + 180.0) % 360.0 - 180.0 + centre
    lon_lo = float(shifted.min()) - margin_lon
    lon_hi = float(shifted.max()) + margin_lon
    if lon_hi - lon_lo >= 180.0:
        return None
    rows = np.nonzero((column_lat >= lat_lo) & (column_lat <= lat_hi))[0]
    cols = np.nonzero((row_lon >= lon_lo) & (row_lon <= lon_hi))[0]
    if rows.size == 0 or cols.size == 0:
        return (slice(0, 0), slice(0, 0))
    return (slice(int(rows.min()), int(rows.max()) + 1),
            slice(int(cols.min()), int(cols.max()) + 1))


def _plan(row, source_lat, source_lon, window, latitude, longitude,
          max_distance_m, n):
    import numpy as np

    digest = sha256()
    for array in (source_lat, source_lon, latitude, longitude):
        array = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    key = (getattr(row, "name", None), row.remap, digest.hexdigest(),
           None if window is None else (window[0].start, window[0].stop,
                                        window[1].start, window[1].stop),
           int(n), float(max_distance_m))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = remap.build_plan(
            method=row.remap, source_latitude=source_lat,
            source_longitude=source_lon, destination_latitude=latitude,
            destination_longitude=longitude, max_distance_m=max_distance_m,
            split_n=n)
        if len(_PLAN_CACHE) >= _PLAN_CACHE_LIMIT:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
        _PLAN_CACHE[key] = plan
    return plan


def ingest(row, path, fields, *, latitude, longitude, map_factors, dx_m,
           max_distance_m, valid_hour=None):
    """Return per-model-cell extensive fields and receipts, all in float64.

    Partition n = ceil(R * source_spacing_rad * max(map_factor) / dx).
    The angular spacing upper-bounds zonal spacing, including high latitude.
    Every split point keeps its original source's mask.  The finite source
    total includes masked quantities and the cells outside the model's reach
    window (:func:`_window`), which the receipt books as unreachable; NaNs
    have no known mass to count.  A variable whose row entry says
    ``"aggregation": "touch"`` (fire radiative power) gives every model cell a
    source cell reaches that cell's whole value instead of a 1/n^2 share
    (``gpuwm.obs_regrid_bridge.SUM_AGGREGATIONS``).
    """
    import numpy as np

    if row.kind != "emission" or row.format != "netcdf" or row.remap not in ("cell_sum", "cell_sum_split"):
        raise EmissionSourceError("this ingest requires extensive NetCDF emission rows with a cell-sum remap; averaging would change cell mass")
    if getattr(map_factors, "shape", None) != getattr(latitude, "shape", None):
        raise EmissionSourceError("model map factors must fill the mass-point grid; otherwise partition spacing would describe another domain")
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields):
        raise EmissionSourceError("requested emission fields must be nonempty and unique; duplicated fields would double-book mass")
    outputs, receipts = {}, {}
    digest = file_sha256(path)
    with netcdf_bridge.open_dataset(path) as dataset:
        if valid_hour is not None:
            _check_hour(dataset, row, _hour(valid_hour))
        full_lat = dataset.variables[row.grid["latitude"]][:]
        full_lon = dataset.variables[row.grid["longitude"]][:]
        n = remap.emission_split(full_lat, full_lon, dx_m, map_factors) if row.remap == "cell_sum_split" else 1
        window = _window(full_lat, full_lon, latitude, longitude, max_distance_m)

        def take(array):
            array = np.asarray(array)
            return np.ascontiguousarray(array if window is None else array[window])

        source_lat = take(full_lat)
        source_lon = take(full_lon)
        plan = None
        if source_lat.size:
            plan = _plan(row, source_lat, source_lon, window, latitude,
                         longitude, max_distance_m, n)
        for field in fields:
            spec = row.variables[field]
            variable = dataset.variables[spec["selector"]]
            if getattr(variable, "units", None) != spec["units"]:
                raise EmissionSourceError(f"{field}: file units differ from the row; the emitted quantity would have the wrong scale")
            aggregation = spec.get("aggregation", "partition")
            variable.set_auto_mask(False)
            values = variable.read_transformed(cache=False)
            if values.ndim == 3 and values.shape[0] == 1:
                values = values[0]
            if values.shape != full_lat.shape:
                raise EmissionSourceError(f"{field}: emission field does not fill the coordinate grid; mass would be assigned to wrong cells")
            values = remap.emission_values(values, spec.get("no_fire_sentinel"))
            full_total = float(values[np.isfinite(values)].sum(dtype=np.float64))
            values = take(values)
            window_total = float(values[np.isfinite(values)].sum(dtype=np.float64))
            if plan is None:
                out = np.zeros(latitude.shape, dtype=np.float64)
                out_valid = np.ones(latitude.shape, dtype=bool)
                totals = {"total_source_mass": 0.0, "total_remapped_mass": 0.0,
                          "unreachable_mass": 0.0, "masked_mass": 0.0}
                used = 0.0
            else:
                index, _reachable, used = plan
                rules = []
                for rule in spec.get("masks", ()):
                    mask = dataset.variables[rule["selector"]][:]
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]
                    rules.append((take(mask), rule.get("minimum", float("-inf")),
                                  rule.get("maximum", float("inf")),
                                  rule.get("nonzero_only", False)))
                valid = remap.mask_ranges(values, rules)
                out, out_valid, totals = remap.apply_sum_plan(
                    source_index=index, values=values, valid=valid,
                    destination_shape=latitude.shape, split_n=n,
                    aggregation=aggregation)
                totals = dict(totals)
            totals["total_source_mass"] = full_total
            totals["unreachable_mass"] += full_total - window_total
            outputs[field] = out
            receipts[field] = dict(
                totals, source_sha256=digest, units=spec["units"],
                cadence_s=row.time["cadence_s"], split_n=n,
                aggregation=aggregation,
                window=None if window is None else [
                    window[0].start, window[0].stop, window[1].start,
                    window[1].stop],
                max_used_distance_m=used, valid=out_valid)
    return outputs, receipts
