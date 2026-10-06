"""Immutable prescribed smoke profiles bound to a radiation adapter's grid.

Raw profiles are Rust-decoded binary fields. Python reads metadata and
memory maps bytes; grid cropping, casts and temporal interpolation use CUDA.
This input provider does not implement smoke transport or emissions.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import math

import numpy as np

SCHEMA = "gpuwm-rrtmg-prescribed-smoke-v1"
GRID_TOLERANCE_DEG = 1.e-4
QUANTITY_UNITS = {
    "dry_mass_mixing_ratio": "ug/kg-dryair",
    "layer_aod": "1",
    "posted_mass_concentration": "kg/m3",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value, *, source, case_clock=False):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{source}: invalid UTC timestamp") from error
    else:
        raise ValueError(f"{source}: a UTC timestamp is required")
    if result.tzinfo is None and case_clock:
        result = result.replace(tzinfo=timezone.utc)
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError(f"{source}: timestamp must explicitly use UTC")
    return result.astimezone(timezone.utc)


def _shape(value, *, name, rank):
    if not isinstance(value, (list, tuple)) or len(value) != rank \
            or any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{name}: {rank} positive integer dimensions are required")
    return tuple(value)


def _stat(path):
    item = path.stat()
    return item.st_size, item.st_mtime_ns, item.st_dev, item.st_ino


@dataclass(frozen=True)
class _Member:
    name: str
    path: Path
    relative_path: str
    sha256: str
    dtype: str
    shape: tuple[int, ...]
    units: str
    stat: tuple[int, ...]

    def verify(self, *, deep=False):
        if _stat(self.path) != self.stat:
            raise ValueError(f"prescribed smoke member {self.name} changed after binding")
        if deep and _sha256(self.path) != self.sha256:
            raise ValueError(f"prescribed smoke member {self.name} SHA256 changed after binding")
        if deep and _stat(self.path) != self.stat:
            raise ValueError(f"prescribed smoke member {self.name} changed during identity verification")

    def identity(self):
        return {"path": self.relative_path, "sha256": self.sha256,
                "dtype": self.dtype, "shape": list(self.shape), "units": self.units}


@dataclass(frozen=True)
class _Manifest:
    path: Path
    sha256: str
    stat: tuple[int, ...]
    quantity: str
    units: str
    shape: tuple[int, int, int]
    start_time: datetime
    times: tuple[datetime, ...]
    geometry: dict[str, _Member]
    frames: tuple[dict[str, _Member], ...]
    provenance: dict

    def members(self):
        yield from self.geometry.values()
        for frame in self.frames:
            yield from frame.values()

    def verify(self, *, deep=False):
        if _stat(self.path) != self.stat:
            raise ValueError("prescribed smoke manifest changed after binding")
        if deep and _sha256(self.path) != self.sha256:
            raise ValueError("prescribed smoke manifest SHA256 changed after binding")
        for member in self.members():
            member.verify(deep=deep)

    def require_vertical(self, eta_levels, hybrid_opt, etac, p_top):
        vertical = self.provenance.get("vertical")
        required = {"eta_levels", "hybrid_opt", "etac", "p_top"}
        if not isinstance(vertical, dict) or set(vertical) != required:
            raise ValueError("prescribed smoke requires declared vertical provenance before use")
        if eta_levels is None or len(eta_levels) != self.shape[0] + 1:
            raise ValueError("prescribed smoke target eta levels must be explicit")
        expected = {"eta_levels": list(eta_levels), "hybrid_opt": hybrid_opt,
                    "etac": etac, "p_top": p_top}
        if vertical != expected:
            raise ValueError("prescribed smoke vertical provenance differs from the target configuration")

    def require_coverage(self, run_seconds):
        if isinstance(run_seconds, bool) or not isinstance(run_seconds, (int, float)) \
                or not math.isfinite(run_seconds) or run_seconds < 0:
            raise ValueError("smoke coverage requires a finite nonnegative run length")
        if self.start_time + timedelta(seconds=run_seconds) > self.times[-1]:
            raise ValueError("prescribed smoke does not cover the complete requested run; extrapolation is refused")

    def source_identity(self):
        self.verify(deep=True)
        return {"schema": "gpuwm-rrtmg-prescribed-smoke-source-identity-v1",
                "manifest_sha256": self.sha256, "quantity": self.quantity, "units": self.units,
                "native_shape": list(self.shape), "vertical_order": "bottom_to_top",
                "time_interpolation": "linear", "times": [value.isoformat() for value in self.times],
                "provenance": deepcopy(self.provenance),
                "geometry": {name: member.identity() for name, member in self.geometry.items()},
                "frames": [{name: member.identity() for name, member in frame.items()} for frame in self.frames]}


def _member(base_dir, item, *, name, shape, units):
    if not isinstance(item, dict) or set(item) != {"path", "sha256", "dtype", "shape", "units"}:
        raise ValueError(f"{name}: member needs path, sha256, dtype, shape and units")
    if item["units"] != units:
        raise ValueError(f"{name}: units must be {units!r}")
    if _shape(item["shape"], name=name, rank=len(shape)) != shape:
        raise ValueError(f"{name}: member shape differs from the declared native grid")
    if item["dtype"] not in ("<f4", "<f8"):
        raise ValueError(f"{name}: raw profiles must declare little-endian <f4 or <f8")
    token = item["path"]
    if not isinstance(token, str) or not token or Path(token).is_absolute():
        raise ValueError(f"{name}: member path must be relative to the manifest")
    path = (base_dir / token).resolve()
    if not path.is_relative_to(base_dir) or path.suffix.lower() not in (".bin", ".f32le", ".f64le", ".f4le", ".f8le"):
        raise ValueError(f"{name}: raw member must remain inside the manifest directory")
    expected_bytes = math.prod(shape) * np.dtype(item["dtype"]).itemsize
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"{name}: raw member is absent or has the wrong byte count {expected_bytes}")
    digest = item["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 \
            or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name}: lowercase SHA256 is required")
    before = _stat(path)
    if _sha256(path) != digest:
        raise ValueError(f"{name}: member SHA256 differs from the manifest")
    if _stat(path) != before:
        raise ValueError(f"{name}: member changed while its SHA256 was read")
    return _Member(name, path, Path(token).as_posix(), digest,
                   item["dtype"], shape, units, before)


def validate_smoke_manifest(path, start_time=None, nz=None):
    """Validate the complete metadata and member hashes without device access."""
    path = Path(path).resolve()
    before = _stat(path)
    payload = path.read_bytes()
    try:
        raw = json.loads(payload)
    except (ValueError, UnicodeError) as error:
        raise ValueError("prescribed smoke manifest is not valid JSON") from error
    required = {"schema", "quantity", "units", "shape", "vertical_order",
                "time_interpolation", "start_time", "geometry", "frames"}
    if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - required - {"provenance"}:
        raise ValueError("prescribed smoke manifest has missing or unknown metadata keys")
    if raw["schema"] != SCHEMA:
        raise ValueError(f"prescribed smoke manifest schema must be {SCHEMA}")
    quantity = raw["quantity"]
    if not isinstance(quantity, str) or quantity not in QUANTITY_UNITS or raw["units"] != QUANTITY_UNITS[quantity]:
        raise ValueError("prescribed smoke quantity/units is not a supported three-dimensional field")
    shape = _shape(raw["shape"], name="smoke native shape", rank=3)
    if nz is not None and (type(nz) is not int or shape[0] != nz):
        raise ValueError("prescribed smoke must carry every target vertical layer")
    if raw["vertical_order"] != "bottom_to_top":
        raise ValueError("prescribed smoke vertical order must be bottom_to_top")
    if raw["time_interpolation"] != "linear":
        raise ValueError("prescribed smoke must explicitly select bounded linear time interpolation")
    start = _utc(raw["start_time"], source="smoke start time")
    expected_start = start if start_time is None else _utc(start_time, source="case start time", case_clock=True)
    if start != expected_start:
        raise ValueError("prescribed smoke start time differs from the actual case start")
    geometry = raw["geometry"]
    if not isinstance(geometry, dict) or set(geometry) != {"latitude", "longitude"}:
        raise ValueError("prescribed smoke requires full native latitude and longitude metadata")
    base_dir = path.parent
    geometry_members = {
        name: _member(base_dir, geometry[name], name=name, shape=shape[1:], units=units)
        for name, units in (("latitude", "degrees_north"), ("longitude", "degrees_east"))}
    if not isinstance(raw["frames"], list) or not raw["frames"]:
        raise ValueError("prescribed smoke requires identified timestamped frames")
    times, frames = [], []
    frame_keys = {"valid_time", "value"}
    if quantity == "posted_mass_concentration":
        frame_keys |= {"donor_p", "donor_t"}
    for number, frame in enumerate(raw["frames"]):
        if not isinstance(frame, dict) or set(frame) != frame_keys:
            raise ValueError(f"smoke frame {number}: fields do not match its quantity")
        valid = _utc(frame["valid_time"], source=f"smoke frame {number}")
        if times and valid <= times[-1]:
            raise ValueError("prescribed smoke frame times must be strictly increasing")
        times.append(valid)
        members = {"value": _member(base_dir, frame["value"], name=f"frame {number} value",
                                     shape=shape, units=QUANTITY_UNITS[quantity])}
        if quantity == "posted_mass_concentration":
            for key, units in (("donor_p", "Pa"), ("donor_t", "K")):
                members[key] = _member(base_dir, frame[key], name=f"frame {number} {key}", shape=shape, units=units)
        frames.append(members)
    if times[0] != start:
        raise ValueError("the first prescribed smoke frame must be exactly the case start")
    if _stat(path) != before:
        raise ValueError("prescribed smoke manifest changed while it was read")
    provenance = raw.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("prescribed smoke provenance must be a metadata object")
    manifest = _Manifest(path, hashlib.sha256(payload).hexdigest(), before,
        quantity, raw["units"], shape, start, tuple(times), geometry_members, tuple(frames), provenance)
    manifest.verify()
    return manifest


def describe_smoke_source(path, *, start_time=None, nz=None):
    """CPU source binding for fingerprints and echo, without device access."""
    return validate_smoke_manifest(path, start_time=start_time, nz=nz).source_identity()


class BoundSmokeManifest:
    """A prescribed input source congruent with one local radiation grid."""

    def __init__(self, path, start_time, latitude_deg, longitude_deg, nz):
        self._manifest = validate_smoke_manifest(path, start_time, nz)
        shape = getattr(latitude_deg, "shape", None)
        if shape is None or len(shape) != 2 or tuple(shape) != tuple(getattr(longitude_deg, "shape", ())):
            raise ValueError("local radiation latitude/longitude must share a two-dimensional shape")
        self.local_shape = (nz, *_shape(tuple(shape), name="local radiation shape", rank=2))
        self._cache = {}
        self._vertical_verified = False
        self._coverage_run_seconds = None
        self._latitude_ref = latitude_deg
        self._longitude_ref = longitude_deg
        self._deferred = False
        self._geometry_bound = False
        self._bind_geometry(latitude_deg, longitude_deg)

    @staticmethod
    def _device_member(member):
        import cupy as cp
        member.verify()
        mapped = np.memmap(member.path, dtype=member.dtype, mode="r", shape=member.shape)
        result = cp.asarray(mapped)
        cp.cuda.get_current_stream().synchronize()
        del mapped
        member.verify()
        return result

    def _bind_geometry(self, latitude_deg, longitude_deg):
        import cupy as cp
        m = self._manifest
        lat = self._device_member(m.geometry["latitude"]).astype(cp.float64, copy=False)
        lon = self._device_member(m.geometry["longitude"]).astype(cp.float64, copy=False)
        local_lat = cp.asarray(latitude_deg, dtype=cp.float64)
        local_lon = cp.asarray(longitude_deg, dtype=cp.float64)
        if not bool(cp.all(cp.isfinite(lat) & cp.isfinite(lon)).item()) \
                or not bool(cp.all(cp.isfinite(local_lat) & cp.isfinite(local_lon)).item()):
            raise ValueError("smoke geometry contains missing or non-finite coordinates")
        if bool(cp.any(cp.abs(lat) > 90).item()) or bool(cp.any(cp.abs(local_lat) > 90).item()):
            raise ValueError("smoke geometry latitude is outside the physical degree range")
        residual = cp.maximum(cp.abs(lat - local_lat[0, 0]), cp.abs(lon - local_lon[0, 0]))
        first = int(cp.argmin(residual).item())
        j0, i0 = divmod(first, m.shape[2])
        ny, nx = self.local_shape[1:]
        if j0 + ny > m.shape[1] or i0 + nx > m.shape[2]:
            raise ValueError("local smoke window, including halos, falls outside the native grid")
        lat_error = cp.max(cp.abs(lat[j0:j0 + ny, i0:i0 + nx] - local_lat))
        lon_error = cp.max(cp.abs(lon[j0:j0 + ny, i0:i0 + nx] - local_lon))
        self.max_latitude_residual_deg = float(lat_error.item())
        self.max_longitude_residual_deg = float(lon_error.item())
        if max(self.max_latitude_residual_deg, self.max_longitude_residual_deg) > GRID_TOLERANCE_DEG:
            raise ValueError("smoke grid is not congruent with the local radiation grid; remapping is not performed")
        self.offset = (j0, i0)
        self._geometry_bound = True
        del lat, lon, local_lat, local_lon, residual

    def require_coverage(self, run_seconds):
        self._manifest.require_coverage(run_seconds)
        self._coverage_run_seconds = float(run_seconds)

    def require_vertical(self, eta_levels, hybrid_opt, etac, p_top):
        self._manifest.require_vertical(eta_levels, hybrid_opt, etac, p_top)
        self._vertical_verified = True

    def rebind(self, latitude_deg, longitude_deg):
        """Bind a rank/slab grid without carrying the parent's field cache."""
        if not self._vertical_verified or self._coverage_run_seconds is None:
            raise ValueError("smoke rebind requires verified target vertical and complete run coverage")
        self._manifest.verify(deep=True)
        shape = getattr(latitude_deg, "shape", None)
        if shape is None or len(shape) != 2 or tuple(shape) != tuple(getattr(longitude_deg, "shape", ())):
            raise ValueError("local radiation latitude/longitude must share a two-dimensional shape")
        other = object.__new__(type(self))
        other._manifest = self._manifest
        other.local_shape = (self.local_shape[0], *_shape(tuple(shape), name="local radiation shape", rank=2))
        other._cache = {}
        other._vertical_verified = True
        other._coverage_run_seconds = self._coverage_run_seconds
        other._latitude_ref = latitude_deg
        other._longitude_ref = longitude_deg
        other._deferred = False
        other._geometry_bound = False
        other._bind_geometry(latitude_deg, longitude_deg)
        return other

    def deferred_rebind(self, latitude_deg, longitude_deg):
        """A rank twin whose actual geography is gathered after construction.

        Array references are retained. Every at() validates their current
        coordinates before consuming data, so neutral placeholders cannot
        produce a profile and tile reuse cannot retain the old field cache.
        """
        if not self._vertical_verified or self._coverage_run_seconds is None:
            raise ValueError("smoke rebind requires verified target vertical and complete run coverage")
        shape = getattr(latitude_deg, "shape", None)
        if shape is None or len(shape) != 2 or tuple(shape) != tuple(getattr(longitude_deg, "shape", ())):
            raise ValueError("local radiation latitude/longitude must share a two-dimensional shape")
        other = object.__new__(type(self))
        other._manifest = self._manifest
        other.local_shape = (self.local_shape[0], *_shape(tuple(shape), name="local radiation shape", rank=2))
        other._cache = {}
        other._vertical_verified = True
        other._coverage_run_seconds = self._coverage_run_seconds
        other._latitude_ref, other._longitude_ref = latitude_deg, longitude_deg
        other._deferred = True
        other._geometry_bound = False
        return other

    def bind_current(self, latitude_deg, longitude_deg):
        """Update references if a tile replaces rather than edits its arrays."""
        self._latitude_ref, self._longitude_ref = latitude_deg, longitude_deg
        self._deferred = True

    def _ensure_geometry(self):
        if not self._deferred:
            return
        before = (self.local_shape, self.offset) if hasattr(self, "offset") else None
        shape = getattr(self._latitude_ref, "shape", None)
        if shape is None or len(shape) != 2 or tuple(shape) != tuple(getattr(self._longitude_ref, "shape", ())):
            raise ValueError("local radiation latitude/longitude must share a two-dimensional shape")
        self.local_shape = (self.local_shape[0], *_shape(tuple(shape), name="local radiation shape", rank=2))
        self._geometry_bound = False
        self._bind_geometry(self._latitude_ref, self._longitude_ref)
        if before != (self.local_shape, self.offset):
            self._cache.clear()

    def _load_frame(self, index):
        if index in self._cache:
            return self._cache[index]
        import cupy as cp
        j0, i0 = self.offset
        ny, nx = self.local_shape[1:]
        loaded = {}
        for name, member in self._manifest.frames[index].items():
            native = self._device_member(member)
            local = native[:, j0:j0 + ny, i0:i0 + nx].astype(cp.float32, copy=True)
            del native
            if not bool(cp.all(cp.isfinite(local)).item()):
                raise ValueError(f"prescribed smoke {name} contains missing or non-finite values")
            positive = name in ("donor_p", "donor_t")
            if bool(cp.any(local <= 0 if positive else local < 0).item()):
                raise ValueError(f"prescribed smoke {name} has physically invalid values")
            if name == "value" and self._manifest.quantity == "dry_mass_mixing_ratio" \
                    and bool(cp.any(local >= 11000).item()):
                raise ValueError("prescribed smoke dry mixing ratio exceeds the source tracer range")
            loaded[name] = cp.ascontiguousarray(local)
        self._cache[index] = loaded
        return loaded

    def at(self, elapsed_seconds, *, latitude_deg=None, longitude_deg=None):
        if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)) \
                or not math.isfinite(elapsed_seconds):
            raise ValueError("prescribed smoke requires finite elapsed model seconds")
        m = self._manifest
        moment = m.start_time + timedelta(seconds=elapsed_seconds)
        if moment < m.times[0] or moment > m.times[-1]:
            raise ValueError("prescribed smoke time is outside its identified frames; extrapolation is refused")
        if not self._vertical_verified:
            raise ValueError("prescribed smoke target vertical provenance must be verified before use")
        if self._coverage_run_seconds is None:
            raise ValueError("prescribed smoke complete run coverage must be verified before use")
        if latitude_deg is not None or longitude_deg is not None:
            if latitude_deg is None or longitude_deg is None:
                raise ValueError("current smoke grid requires both latitude and longitude")
            self.bind_current(latitude_deg, longitude_deg)
        self._ensure_geometry()
        m.verify()
        upper = bisect_left(m.times, moment)
        if m.times[upper] == moment:
            keep = {upper}
            self._cache = {index: value for index, value in self._cache.items() if index in keep}
            result = {"quantity": m.quantity, **self._load_frame(upper)}
        else:
            lower = upper - 1
            keep = {lower, upper}
            self._cache = {index: value for index, value in self._cache.items() if index in keep}
            fraction = np.float32((moment - m.times[lower]).total_seconds() /
                                  (m.times[upper] - m.times[lower]).total_seconds())
            left, right = self._load_frame(lower), self._load_frame(upper)
            from gpuwm.core.kernels import load_module
            import cupy as cp
            result = {"quantity": m.quantity}
            kernel = load_module("rrtmg_smoke_manifest").get_function("rrtmg_smoke_time_blend")
            for name in left:
                value = cp.empty(self.local_shape, cp.float32)
                kernel(((value.size + 127) // 128,), (128,),
                    (np.int64(value.size), fraction, left[name], right[name], value))
                result[name] = value
        self._cache = {index: value for index, value in self._cache.items() if index in keep}
        m.verify()
        return result

    def identity(self):
        """Reverify hashes before publishing a restart identity or receipt."""
        if not self._vertical_verified:
            raise ValueError("prescribed smoke target vertical provenance must be verified before identity publication")
        m = self._manifest
        m.verify(deep=True)
        return m.source_identity()

    def binding_receipt(self):
        """Runtime local grid checks, separate from immutable source identity."""
        if not self._geometry_bound:
            raise ValueError("prescribed smoke local geometry has not been validated yet")
        m = self._manifest
        m.verify()
        return {"schema": "gpuwm-rrtmg-prescribed-smoke-identity-v1",
                "manifest_sha256": m.sha256, "quantity": m.quantity, "units": m.units,
                "native_shape": list(m.shape), "local_shape": list(self.local_shape),
                "vertical_order": "bottom_to_top", "time_interpolation": "linear",
                "provenance": deepcopy(m.provenance),
                "coverage_run_seconds": self._coverage_run_seconds,
                "times": [value.isoformat() for value in m.times],
                "offset_ji": list(self.offset), "grid_tolerance_deg": GRID_TOLERANCE_DEG,
                "max_latitude_residual_deg": self.max_latitude_residual_deg,
                "max_longitude_residual_deg": self.max_longitude_residual_deg,
                "geometry": {name: member.identity() for name, member in m.geometry.items()},
                "frames": [{name: member.identity() for name, member in frame.items()} for frame in m.frames]}
