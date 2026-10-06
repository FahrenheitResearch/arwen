"""Local saved-field adapters for the regional rain scorer.

NetCDF payloads go through ``gpuwm.netcdf_bridge`` and observation packs
through the existing Rust-writer pack reader. NumPy files are an explicit
saved-array interchange, not an alternate GRIB or NetCDF decoder.

A ``regional-rain/input.v1`` manifest declares relative forecast clocks,
quality masks and grid bounds. Nothing guesses a reset, fills missing data
with dry values, downloads an object, or selects a nearest verification time.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "regional-rain/input.v1"
_DEPTH_UNITS = {"mm", "kg/m^2", "kg m-2", "kg m^-2", "kg m**-2"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC time needs a zone: {value!r}")
    return parsed.astimezone(timezone.utc)


class _Fields:
    def __init__(self, root: Path):
        self.root = root
        self.sources: dict[str, dict] = {}

    def _path(self, value, expected_hash=None):
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"saved scoring field does not exist: {path}")
        name = str(path)
        if name not in self.sources:
            self.sources[name] = {"path": name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        if expected_hash is not None and self.sources[name]["sha256"] != str(expected_hash).lower():
            raise ValueError(f"saved scoring source SHA-256 mismatch: {path}")
        return path

    def _one(self, entry, path_value):
        path = self._path(path_value, entry.get("sha256"))
        kind = entry.get("format") or ("obspack" if path.suffix == ".obspack" else
                                        "geopack" if path.suffix == ".geopack" else
                                        "npz" if path.suffix == ".npz" else
                                        "npy" if path.suffix == ".npy" else "netcdf")
        variable = str(entry.get("variable", entry.get("key", "values")))
        units = None
        if kind == "npz":
            with np.load(path, allow_pickle=False) as saved:
                if variable not in saved:
                    raise ValueError(f"{path}: no saved array {variable!r}")
                result = np.array(saved[variable], copy=True)
        elif kind == "npy":
            result = np.load(path, allow_pickle=False)
        elif kind in ("obspack", "geopack"):
            from gpuwm.obs.obspack import read_pack

            pack = read_pack(path)
            required_schema = ("gpuwm-obs.obs-grid.v1" if kind == "obspack" else "gpuwm-obs.obs-geo.v1")
            if pack.schema != required_schema:
                raise ValueError(f"{path}: schema {pack.schema!r}, expected {required_schema!r}")
            result = np.array(pack.array(variable), copy=True)
            units = pack.meta.get("units")
            self.sources[str(path)].update({"native_quantity": pack.meta.get("quantity"), "native_units": units,
                                           "native_is_stub": bool(pack.meta.get("provenance", {}).get("is_stub", False))})
        elif kind == "netcdf":
            from gpuwm import netcdf_bridge

            with netcdf_bridge.open_dataset(path) as dataset:
                if variable not in dataset.variables:
                    raise ValueError(f"{path}: no NetCDF variable {variable!r}")
                source = dataset.variables[variable]
                units = getattr(source, "units", None)
                result = np.ma.filled(source[:], np.nan)
        else:
            raise ValueError(f"saved field format {kind!r} is unsupported")
        if entry.get("units") is not None and units is not None and str(entry["units"]) != str(units):
            water_equivalent = str(entry["units"]) in _DEPTH_UNITS and str(units) in _DEPTH_UNITS
            if not water_equivalent:
                raise ValueError(f"{path}:{variable}: units {units!r}, manifest declares {entry['units']!r}")
        if "index" in entry:
            result = result[int(entry["index"])]
        reduction = entry.get("reduction")
        if reduction is not None:
            if reduction != "max_z":
                raise ValueError(f"unsupported saved-field reduction {reduction!r}")
            # The z dimension is explicitly requested. A time dimension is
            # retained when present; a one-frame 3-D field has no time axis.
            if result.ndim not in (3, 4):
                raise ValueError(f"max_z needs (z,y,x) or (time,z,y,x), got {result.shape}")
            from gpuwm.verify import rain_gate_bridge as native

            result = native.combine(np.moveaxis(result, -3, 0), mode="max")
        return np.asarray(result)

    def get(self, value, expected_units=None):
        if not isinstance(value, dict):
            return np.asarray(value)
        declared = value.get("units")
        if declared is not None and expected_units is not None and str(declared) != expected_units:
            if not (str(declared) in _DEPTH_UNITS and expected_units in _DEPTH_UNITS):
                raise ValueError(f"saved field units {declared!r} disagree with canonical units {expected_units!r}")
        if "constant" in value:
            number = float(value["constant"])
            shape = tuple(int(n) for n in value["shape"])
            if not np.isfinite(number) or any(n < 1 for n in shape):
                raise ValueError("constant saved-field descriptor needs a finite value and positive shape")
            return np.full(shape, number)
        if "and" in value:
            from gpuwm.verify import rain_gate_bridge as native

            masks = [_mask(self.get(part), "combined validity") for part in value["and"]]
            if not masks or any(mask.shape != masks[0].shape for mask in masks):
                raise ValueError("combined validity needs nonempty identical-shape explicit masks")
            return native.combine(np.stack(masks), mode="and")
        if "threshold" in value:
            from gpuwm.verify import rain_gate_bridge as native

            definition = value["threshold"]
            field = self.get(definition["field"])
            valid = _mask(self.get(definition["valid"]), "threshold validity")
            return native.quality_mask(field, valid, float(definition["minimum"]))
        if "sum" in value:
            parts = [self.get(part, expected_units=expected_units) for part in value["sum"]]
            if not parts or any(part.shape != parts[0].shape for part in parts):
                raise ValueError("saved-field sum needs nonempty identical-shape arrays")
            from gpuwm.verify import rain_gate_bridge as native

            return native.combine(np.stack(parts), mode="sum")
        if "path" in value:
            definition = dict(value)
            if expected_units is not None:
                definition.setdefault("units", expected_units)
            return self._one(definition, value["path"])
        if "paths" in value:
            paths = value["paths"]
            if not paths:
                raise ValueError("saved-field paths is empty")
            definition = dict(value)
            if expected_units is not None:
                definition.setdefault("units", expected_units)
            frames = [self._one(definition, path) for path in paths]
            if value.get("join", "frames") == "concat_time":
                if any(frame.ndim < 3 for frame in frames):
                    raise ValueError("concat_time needs files retaining a time dimension")
                return np.concatenate(frames, axis=0)
            if value.get("join", "frames") != "frames":
                raise ValueError("saved-field join must be frames or concat_time")
            frames = [frame[0] if frame.ndim >= 3 and frame.shape[0] == 1 else frame for frame in frames]
            return np.stack(frames)
        raise ValueError("field descriptor needs path, paths or sum")


def _mask(value, label):
    raw = np.asarray(value)
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{label} must carry only explicit zero/one validity")
    return raw.astype(bool)


def load_series(manifest_path, analysis_end=None):
    """Read a manifest into ``(rain_gate.Series, provenance)``.

    Field values can be inline arrays, ``{path, variable}``,
    ``{paths, variable}``, or ``{sum: [descriptor, ...]}``. Relative paths
    resolve beside the manifest. ``paths`` defaults to one frame per file;
    ``join: concat_time`` explicitly joins retained time axes. Units on a
    descriptor, when present in the native file, must match exactly.

    Projected bounds are supplied explicitly or derived from native
    latitude/longitude center arrays using the scorer's spherical equal-area
    projection. The latter geometry is labelled as center-derived and its
    edge inference is part of the receipt.
    """
    from gpuwm.verify import rain_gate as gate

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"input schema must be {SCHEMA!r}")
    fields = _Fields(manifest_path.parent)
    if "times_seconds" in manifest:
        times = np.asarray(fields.get(manifest["times_seconds"]), dtype=float)
        clock_source = "explicit_seconds_after_final_analysis"
    elif "times_utc" in manifest:
        reference = analysis_end or manifest.get("analysis_end")
        if reference is None:
            raise ValueError("absolute saved-field times need analysis_end")
        origin = _utc(reference)
        times = np.array([(_utc(stamp) - origin).total_seconds() for stamp in manifest["times_utc"]])
        clock_source = "explicit_utc_minus_final_analysis"
    else:
        raise ValueError("saved fields need times_seconds or times_utc")
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("saved-field clocks must be finite, strictly increasing and unique")
    definition = manifest["grid"]
    projection = str(definition.get("projection_id", ""))
    geometry_check = {"status": "explicit_bounds_no_native_center_check"}
    if "wrf_projection" in definition:
        from gpuwm import netcdf_bridge
        from gpuwm.static import rust_bridge

        source = definition["wrf_projection"]
        source = {"path": source} if isinstance(source, str) else source
        path = fields._path(source["path"], source.get("sha256"))
        with netcdf_bridge.open_dataset(path) as dataset:
            attributes = dataset.global_attributes
            map_code = int(attributes["MAP_PROJ"])
            kind = {1: "lambert", 2: "polar", 3: "mercator"}.get(map_code)
            if kind is None:
                raise ValueError(f"saved WRF projection MAP_PROJ={map_code} has no native corner adapter")
            nx = len(dataset.dimensions["west_east"])
            ny = len(dataset.dimensions["south_north"])
            spec = {
                "kind": kind, "ref_lat": float(attributes["CEN_LAT"]), "ref_lon": float(attributes["CEN_LON"]),
                "truelat1": float(attributes["TRUELAT1"]), "truelat2": float(attributes["TRUELAT2"]),
                "stand_lon": float(attributes["STAND_LON"]), "dx": float(attributes["DX"]), "dy": float(attributes["DY"]),
                "e_we": nx + 1, "e_sn": ny + 1, "known_x": (nx + 1) / 2, "known_y": (ny + 1) / 2,
                "moad_cen_lat": float(attributes.get("MOAD_CEN_LAT", attributes["CEN_LAT"])),
                "moad_cen_lon": float(attributes.get("MOAD_CEN_LON", attributes["CEN_LON"])),
            }
            native_coordinates = None
            if "XLAT" in dataset.variables and "XLONG" in dataset.variables:
                native_coordinates = (np.asarray(dataset.variables["XLAT"][:]), np.asarray(dataset.variables["XLONG"][:]))
        if "wps_grid_spec" in source:
            # A producer may retain the original binary64 projection config.
            # It is authoritative only when it matches the native array size.
            original = source["wps_grid_spec"]
            if int(original["e_we"]) != nx + 1 or int(original["e_sn"]) != ny + 1:
                raise ValueError("pinned WPS grid spec does not match saved WRF dimensions")
            spec = original
        handle = rust_bridge.grid_new(spec)
        try:
            latitude = rust_bridge.grid_array(handle, 3, 0, ny + 1, nx + 1)
            longitude = rust_bridge.grid_array(handle, 3, 1, ny + 1, nx + 1)
            if native_coordinates is not None:
                expected_lat = rust_bridge.grid_array(handle, 0, 0, ny, nx)
                expected_lon = rust_bridge.grid_array(handle, 0, 1, ny, nx)
                actual_lat, actual_lon = native_coordinates
                if actual_lat.shape != actual_lon.shape or actual_lat.shape[-2:] != (ny, nx):
                    raise ValueError("saved WRF XLAT/XLONG shape differs from native projection")
                if not np.all(np.isfinite(actual_lat)) or not np.all(np.isfinite(actual_lon)):
                    raise ValueError("saved WRF native mass coordinates are incomplete")
                lat_error = np.abs(actual_lat - expected_lat)
                lon_error = np.abs((actual_lon - expected_lon + 180.0) % 360.0 - 180.0)
                max_lat, max_lon = float(lat_error.max()), float(lon_error.max())
                if max_lat > 2e-5 or max_lon > 2e-5:
                    raise ValueError(f"saved WRF projection disagrees with native mass centers: latitude {max_lat:.8g}, longitude {max_lon:.8g} degrees")
                geometry_check = {"status": "passed_native_mass_coordinates", "tolerance_degrees": 2e-5,
                                  "max_latitude_error_degrees": max_lat, "max_longitude_error_degrees": max_lon}
            else:
                geometry_check = {"status": "not_measured_no_XLAT_XLONG"}
        finally:
            rust_bridge.grid_free(handle)
        center = definition.get("equal_area_center")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("native WRF projection needs frozen equal_area_center [longitude, latitude]")
        x_corner, y_corner = gate.project_laea(latitude, longitude, float(center[0]), float(center[1]))
        projection = f"spherical-laea-r6370000:{float(center[0]):.12g},{float(center[1]):.12g}"
        grid = gate.QuadGrid(x_corner, y_corner, projection)
        geometry_source = "native_wps_projection_corners_r6370000"
    elif "x_edges_m" in definition and "y_edges_m" in definition:
        if not projection:
            raise ValueError("rectangular meter bounds need an explicit equal-area projection_id")
        grid = gate.RectGrid(fields.get(definition["x_edges_m"]), fields.get(definition["y_edges_m"]), projection)
        geometry_source = "explicit_equal_area_rect_bounds"
    elif "x_corners_m" in definition and "y_corners_m" in definition:
        if not projection:
            raise ValueError("projected quadrilaterals need an explicit equal-area projection_id")
        grid = gate.QuadGrid(fields.get(definition["x_corners_m"]), fields.get(definition["y_corners_m"]), projection)
        geometry_source = "explicit_equal_area_quad_bounds"
    elif "latitude" in definition and "longitude" in definition:
        latitude = np.asarray(fields.get(definition["latitude"]), dtype=float)
        longitude = np.asarray(fields.get(definition["longitude"]), dtype=float)
        if latitude.ndim == 3 and latitude.shape[0] == 1:
            latitude, longitude = latitude[0], longitude[0]
        if latitude.shape != longitude.shape or latitude.ndim != 2:
            raise ValueError("native latitude/longitude must be same-shape 2-D centers")
        center = definition.get("equal_area_center")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("native geometry needs frozen equal_area_center [longitude, latitude]")
        if definition.get("lat_lon_regular", False):
            if (not np.allclose(latitude, latitude[:, :1], rtol=0, atol=1e-7)
                    or not np.allclose(longitude, longitude[:1, :], rtol=0, atol=1e-7)):
                raise ValueError("lat_lon_regular requires native regular geographic rows and columns")
            # Midpoints are exact angular bounds for a regular geographic
            # grid. Projection happens after corner recovery, not before it.
            lat_corner, lon_corner = gate.corners_from_centers(latitude, longitude)
            x_corner, y_corner = gate.project_laea(lat_corner, lon_corner, float(center[0]), float(center[1]))
            geometry_source = "native_regular_geographic_angular_bounds_r6370000"
        else:
            x, y = gate.project_laea(latitude, longitude, float(center[0]), float(center[1]))
            x_corner, y_corner = gate.corners_from_centers(x, y)
            geometry_source = "native_centers_laea_bilinear_edge_extrapolation"
        projection = f"spherical-laea-r6370000:{float(center[0]):.12g},{float(center[1]):.12g}"
        grid = gate.QuadGrid(x_corner, y_corner, projection)
    else:
        raise ValueError("grid needs equal-area edges, quadrilateral corners or native latitude/longitude")
    echo_times = times
    if "echo_times_seconds" in manifest:
        echo_times = np.asarray(fields.get(manifest["echo_times_seconds"]), dtype=float)
    elif "echo_times_utc" in manifest:
        reference = analysis_end or manifest.get("analysis_end")
        if reference is None:
            raise ValueError("absolute echo times need analysis_end")
        origin = _utc(reference)
        echo_times = np.array([(_utc(stamp) - origin).total_seconds() for stamp in manifest["echo_times_utc"]])
    if echo_times.ndim != 1 or not len(echo_times) or not np.all(np.isfinite(echo_times)) or np.any(np.diff(echo_times) <= 0):
        raise ValueError("echo clocks must be finite, strictly increasing and unique")
    names = ("rain_accum_mm", "rate_mm_h", "rain_valid", "reset_ids", "reset_carry_mm", "echo_dbz", "echo_valid")
    canonical_units = {"rain_accum_mm": "mm", "rate_mm_h": "mm/hr", "reset_carry_mm": "mm", "echo_dbz": "dBZ"}
    arrays = {name: fields.get(manifest[name], expected_units=canonical_units.get(name)) for name in names if name in manifest}
    has_rain = "rain_accum_mm" in arrays or "rate_mm_h" in arrays
    if not has_rain or ("rain_accum_mm" in arrays and "rate_mm_h" in arrays):
        raise ValueError("saved fields need exactly one rain_accum_mm or rate_mm_h")
    if "rain_valid" not in arrays:
        raise ValueError("saved rain needs explicit rain_valid quality/support mask")
    if "rain_accum_mm" in arrays and "reset_ids" not in arrays:
        raise ValueError("saved cumulative rain needs explicit reset_ids; resets cannot be inferred or clipped")
    if "echo_dbz" in arrays and "echo_valid" not in arrays:
        raise ValueError("saved reflectivity needs explicit echo_valid quality/support mask")
    shape = (len(times), *grid.shape)
    for name, array in arrays.items():
        if name == "reset_ids":
            if array.shape != (len(times),):
                raise ValueError(f"reset_ids shape {array.shape}, expected {(len(times),)}")
            if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
                raise ValueError("reset_ids must be exact finite integers")
        else:
            expected = (len(echo_times), *grid.shape) if name in ("echo_dbz", "echo_valid") else shape
            if array.shape != expected:
                raise ValueError(f"{name} shape {array.shape}, expected {expected}")
    arrays["rain_valid"] = _mask(arrays["rain_valid"], "rain_valid")
    if "echo_valid" in arrays:
        arrays["echo_valid"] = _mask(arrays["echo_valid"], "echo_valid")
    if "rain_accum_mm" in arrays:
        resets = arrays["reset_ids"]
        if np.any(np.diff(resets, axis=0) != 0) and "reset_carry_mm" not in arrays:
            raise ValueError("counter reset requires explicit reset_carry_mm before-reset water")
    footprints = None
    if "footprint_masks" in manifest:
        footprints = {}
        for lead, descriptor in manifest["footprint_masks"].items():
            mask = _mask(fields.get(descriptor), f"observed footprint lead {lead}")
            if mask.shape != grid.shape:
                raise ValueError(f"observed footprint lead {lead} shape differs from its native truth grid")
            footprints[str(lead)] = mask
    provenance = {
        "schema": SCHEMA, "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "sources": list(fields.sources.values()), "clock_source": clock_source,
        "geometry_source": geometry_source, "projection_id": projection,
        "geometry_check": geometry_check,
        "truth_revision": manifest.get("truth_revision"), "archive_grade": manifest.get("archive_grade", "archive-rich-research"),
        "is_stub": bool(manifest.get("is_stub", False)) or any(source.get("native_is_stub", False) for source in fields.sources.values()),
        "footprint_source": "explicit_native_truth_masks" if footprints is not None else "observed_35dbz_hour_union",
    }
    return gate.Series(grid=grid, times_seconds=times, echo_times_seconds=echo_times, footprint_masks=footprints, **arrays), provenance


__all__ = ["SCHEMA", "load_series"]
