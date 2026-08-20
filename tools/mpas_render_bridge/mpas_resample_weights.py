"""Per-mesh nearest-cell resample weights for MPAS to WRF-shaped rendering.

SUPERSEDED -- REFERENCE GENERATOR, NOT A PRODUCT PATH.
The Rust crate ``tools/rustwx/crates/rw-mpas`` (binary ``rw_mpas_convert``)
owns this on the release line and reproduces this file bit for bit: identical
weights digests on both render windows, 404,772,334 float values compared, 127
of 127 rendered PNGs byte-identical.  Receipt:
``evidence/rw-mpas-converter-parity.json``.  This file stays because that
receipt is *about* it -- delete the reference and the parity claim becomes
unreproducible -- but it is kept as the generator of that reference, never as
a route a run takes.  A bare invocation refuses; ``--regenerate-reference``
is the acknowledged door for regenerating the goldens.

This is the one-time, per-mesh half of the MPAS-to-wrfout render bridge.  It
reads a rotated MPAS grid file (``latCell``/``lonCell`` in radians on the unit
sphere), builds a nearest-cell index map onto one or more structured target
windows, and caches the result as a compressed ``.npz`` keyed by the mesh
SHA-256 and by a canonical hash of the window specification.  The converter
(``mpas_history_to_wrfout.py``) consumes the cache; nothing here reads a
forecast field or touches a GPU.

Method: nearest cell centre, on the sphere, by 3-D chord distance through a
k-d tree over unit vectors.  Nearest-cell is chosen deliberately.  A resampled
value is exactly the value the model carried in that cell, so extremes survive
the transfer unchanged: a 70 dBZ core stays 70 dBZ, a 35 m/s gust stays
35 m/s.  Every smoothing scheme, inverse-distance included, pulls maxima down
and pushes minima up, which is the wrong trade for meteorological products
whose whole point is the tail of the distribution.

Later upgrade, deliberately not implemented here: barycentric interpolation on
the Delaunay dual of the cell centres.  That is the right choice for smooth
fields (geopotential height, mean-sea-level pressure) where nearest-cell
leaves visible hexagon facets at the coarse end of a variable-resolution mesh,
and it is second-order accurate where nearest-cell is zeroth-order.  It needs
a triangulation of the cell centres plus per-target barycentric coordinates,
so it is a strictly larger cache with the same interface.  When it lands it
should be selectable per field, not per file: nearest for anything judged on
extremes, barycentric for anything judged on gradients.

Neither scheme is conservative.  Nothing produced through this path is a
conservative remap and nothing here supports an accumulation-budget claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from netCDF4 import Dataset
import numpy as np
from scipy.spatial import cKDTree


WEIGHTS_SCHEMA = "mpas-port.nearest-cell-render-weights/v1"
RESAMPLE_METHOD = "nearest-cell (spherical k-d tree over unit vectors)"

# WRF's spherical earth.  The Lambert windows below emulate a WRF domain, so
# they must use WRF's radius, not the WGS84 mean radius, or the emitted XLAT
# and XLONG disagree with the projection attributes the renderer reads.
WRF_EARTH_RADIUS_M = 6_370_000.0
# The MPAS meshes carry a real Earth radius for physical distances.
MPAS_EARTH_RADIUS_M = 6_371_229.0

_RAD = np.pi / 180.0
_DEG = 180.0 / np.pi


class ResampleWeightsRefusal(RuntimeError):
    """A mesh, window-specification, or cache-binding refusal."""


@dataclass(frozen=True)
class LatLonWindow:
    """A regular latitude/longitude window; renders under WRF MAP_PROJ=6."""

    south: float
    north: float
    west: float
    east: float
    spacing_degrees: float
    description: str = ""

    kind: str = "latlon"

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "latlon",
            "south": float(self.south),
            "north": float(self.north),
            "west": float(self.west),
            "east": float(self.east),
            "spacing_degrees": float(self.spacing_degrees),
        }

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        latitude = _axis(self.south, self.north, self.spacing_degrees, "latitude")
        longitude = _axis(self.west, self.east, self.spacing_degrees, "longitude")
        if np.any(np.abs(latitude) > 90.0):
            raise ResampleWeightsRefusal("latitude window leaves [-90, 90]")
        lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
        return (
            np.ascontiguousarray(lat_grid, dtype=np.float64),
            np.ascontiguousarray(lon_grid, dtype=np.float64),
        )

    def projection_attributes(self) -> dict[str, Any]:
        latitude = _axis(self.south, self.north, self.spacing_degrees, "latitude")
        longitude = _axis(self.west, self.east, self.spacing_degrees, "longitude")
        centre_lat = 0.5 * (float(latitude[0]) + float(latitude[-1]))
        centre_lon = 0.5 * (float(longitude[0]) + float(longitude[-1]))
        # WRF writes DX/DY in metres even for a lat-lon grid; the nominal
        # spacing at the window centre is the honest number to publish.
        metres = self.spacing_degrees * _RAD * WRF_EARTH_RADIUS_M
        return {
            "MAP_PROJ": 6,
            "MAP_PROJ_CHAR": "Cylindrical Equidistant",
            "DX": float(metres),
            "DY": float(metres),
            "TRUELAT1": 0.0,
            "TRUELAT2": 0.0,
            "STAND_LON": centre_lon,
            "CEN_LAT": centre_lat,
            "CEN_LON": centre_lon,
            "MOAD_CEN_LAT": centre_lat,
            "POLE_LAT": 90.0,
            "POLE_LON": 0.0,
        }


@dataclass(frozen=True)
class LambertWindow:
    """A Lambert conformal window emulating a WRF domain (MAP_PROJ=1).

    The renderer's regional presentation, and the map furniture that comes
    with it, is what a real wrfout gets.  A Lambert window is how an MPAS
    frame asks for that treatment.
    """

    centre_lat: float
    centre_lon: float
    dx_metres: float
    nx: int
    ny: int
    truelat1: float = 30.0
    truelat2: float = 60.0
    stand_lon: float | None = None
    description: str = ""

    kind: str = "lambert"

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "lambert",
            "centre_lat": float(self.centre_lat),
            "centre_lon": float(self.centre_lon),
            "dx_metres": float(self.dx_metres),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "truelat1": float(self.truelat1),
            "truelat2": float(self.truelat2),
            "stand_lon": float(self._stand_lon()),
            "earth_radius_m": WRF_EARTH_RADIUS_M,
        }

    def _stand_lon(self) -> float:
        return float(self.centre_lon if self.stand_lon is None else self.stand_lon)

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        if self.nx < 2 or self.ny < 2:
            raise ResampleWeightsRefusal("a Lambert window needs at least 2x2 points")
        if not np.isfinite(self.dx_metres) or self.dx_metres <= 0.0:
            raise ResampleWeightsRefusal("Lambert dx must be positive and finite")
        projection = _lambert_projection(
            truelat1=self.truelat1,
            truelat2=self.truelat2,
            stand_lon=self._stand_lon(),
            dx_metres=self.dx_metres,
            known_lat=self.centre_lat,
            known_lon=self.centre_lon,
            known_i=0.5 * (self.nx + 1.0),
            known_j=0.5 * (self.ny + 1.0),
        )
        i_index = np.arange(1, self.nx + 1, dtype=np.float64)
        j_index = np.arange(1, self.ny + 1, dtype=np.float64)
        i_grid, j_grid = np.meshgrid(i_index, j_index, indexing="xy")
        latitude, longitude = _lambert_inverse(projection, i_grid, j_grid)
        return (
            np.ascontiguousarray(latitude, dtype=np.float64),
            np.ascontiguousarray(longitude, dtype=np.float64),
        )

    def projection_attributes(self) -> dict[str, Any]:
        latitude, longitude = self.coordinates()
        centre_j = self.ny // 2
        centre_i = self.nx // 2
        return {
            "MAP_PROJ": 1,
            "MAP_PROJ_CHAR": "Lambert Conformal",
            "DX": float(self.dx_metres),
            "DY": float(self.dx_metres),
            "TRUELAT1": float(self.truelat1),
            "TRUELAT2": float(self.truelat2),
            "STAND_LON": self._stand_lon(),
            "CEN_LAT": float(latitude[centre_j, centre_i]),
            "CEN_LON": float(longitude[centre_j, centre_i]),
            "MOAD_CEN_LAT": float(latitude[centre_j, centre_i]),
            "POLE_LAT": 90.0,
            "POLE_LON": 0.0,
        }


Window = LatLonWindow | LambertWindow


# The catalogue of render windows.  Two per mesh is the ruled fidelity
# contract: a coarse global overview plus the focus region at its own native
# resolution.
WINDOWS: dict[str, Window] = {
    "global": LatLonWindow(
        south=-90.0,
        north=90.0,
        west=-180.0,
        east=179.75,
        spacing_degrees=0.25,
        description="global 0.25 degree overview window",
    ),
    "focus": LambertWindow(
        centre_lat=37.0,
        centre_lon=-96.0,
        dx_metres=22_000.0,
        nx=240,
        ny=150,
        truelat1=30.0,
        truelat2=60.0,
        stand_lon=-96.0,
        description=(
            "CONUS focus window on a 22 km Lambert grid, matched to the "
            "x4.163842 refined-region spacing"
        ),
    ),
    # Kept for reference and for any consumer that wants a plain regular grid,
    # but NOT the product surface.  Measured 2026-08-12: a REGIONAL MAP_PROJ=6
    # frame makes the renderer's adaptive presentation collapse the map into a
    # shallow strip with most of the domain clipped away, and the only escape
    # is RUSTWX_PROJECTION_VARIANT=rectangular, which draws a flat
    # plate-carree CONUS.  The Lambert "focus" window above has no such
    # problem and gets the renderer's native regional treatment.  A global
    # MAP_PROJ=6 frame is fine; the defect is specific to a regional one.
    "focus-latlon": LatLonWindow(
        south=24.0,
        north=50.0,
        west=-125.0,
        east=-66.0,
        spacing_degrees=0.2,
        description=(
            "CONUS focus window on a 0.2 degree regular lat-lon grid; "
            "renders only under RUSTWX_PROJECTION_VARIANT=rectangular, "
            "prefer the Lambert focus window for products"
        ),
    ),
}


def _axis(first: float, last: float, step: float, role: str) -> np.ndarray:
    if not (np.isfinite(first) and np.isfinite(last) and np.isfinite(step)):
        raise ResampleWeightsRefusal(f"{role} window bounds must be finite")
    if last <= first or step <= 0.0:
        raise ResampleWeightsRefusal(f"{role} window must increase")
    count = int(round((last - first) / step)) + 1
    axis = first + np.arange(count, dtype=np.float64) * step
    if not np.isclose(axis[-1], last, rtol=0.0, atol=1.0e-9):
        raise ResampleWeightsRefusal(
            f"{role} spacing does not exactly partition its bounds"
        )
    axis[-1] = last
    return axis


@dataclass(frozen=True)
class _LambertProjection:
    cone: float
    hemi: float
    rebydx: float
    polei: float
    polej: float
    truelat1: float
    truelat2: float
    stand_lon: float


def _lambert_cone(truelat1: float, truelat2: float) -> float:
    if abs(truelat1 - truelat2) > 0.1:
        numerator = np.log10(np.cos(truelat1 * _RAD)) - np.log10(
            np.cos(truelat2 * _RAD)
        )
        denominator = np.log10(
            np.tan((45.0 - abs(truelat1) / 2.0) * _RAD)
        ) - np.log10(np.tan((45.0 - abs(truelat2) / 2.0) * _RAD))
        return float(numerator / denominator)
    return float(np.sin(abs(truelat1) * _RAD))


def _lambert_projection(
    *,
    truelat1: float,
    truelat2: float,
    stand_lon: float,
    dx_metres: float,
    known_lat: float,
    known_lon: float,
    known_i: float,
    known_j: float,
) -> _LambertProjection:
    """WRF ``module_llxy.F`` ``set_lc``, transcribed."""

    cone = _lambert_cone(truelat1, truelat2)
    hemi = 1.0 if truelat1 >= 0.0 else -1.0
    rebydx = WRF_EARTH_RADIUS_M / dx_metres
    delta_lon = known_lon - stand_lon
    if delta_lon > 180.0:
        delta_lon -= 360.0
    if delta_lon < -180.0:
        delta_lon += 360.0
    ctl1r = np.cos(truelat1 * _RAD)
    rsw = (
        rebydx
        * ctl1r
        / cone
        * (
            np.tan((90.0 * hemi - known_lat) * _RAD / 2.0)
            / np.tan((90.0 * hemi - truelat1) * _RAD / 2.0)
        )
        ** cone
    )
    arg = cone * (delta_lon * _RAD)
    polei = hemi * known_i - hemi * rsw * np.sin(arg)
    polej = hemi * known_j + rsw * np.cos(arg)
    return _LambertProjection(
        cone=cone,
        hemi=hemi,
        rebydx=rebydx,
        polei=float(polei),
        polej=float(polej),
        truelat1=float(truelat1),
        truelat2=float(truelat2),
        stand_lon=float(stand_lon),
    )


def _lambert_inverse(
    projection: _LambertProjection, i_grid: np.ndarray, j_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """WRF ``module_llxy.F`` ``ijll_lc``, transcribed."""

    hemi = projection.hemi
    chi1 = (90.0 - hemi * projection.truelat1) * _RAD
    chi2 = (90.0 - hemi * projection.truelat2) * _RAD
    inew = hemi * np.asarray(i_grid, dtype=np.float64)
    jnew = hemi * np.asarray(j_grid, dtype=np.float64)
    xx = inew - projection.polei
    yy = projection.polej - jnew
    radius_squared = xx * xx + yy * yy
    radius = np.sqrt(radius_squared) / projection.rebydx
    longitude = (
        projection.stand_lon + _DEG * np.arctan2(hemi * xx, yy) / projection.cone
    )
    longitude = np.mod(longitude + 360.0, 360.0)
    if abs(chi1 - chi2) < 1.0e-12:
        chi = 2.0 * np.arctan(
            (radius / np.tan(chi1)) ** (1.0 / projection.cone) * np.tan(chi1 * 0.5)
        )
    else:
        chi = 2.0 * np.arctan(
            (radius * projection.cone / np.sin(chi1)) ** (1.0 / projection.cone)
            * np.tan(chi1 * 0.5)
        )
    latitude = (90.0 - chi * _DEG) * hemi
    pole = radius_squared == 0.0
    if np.any(pole):
        latitude = np.where(pole, hemi * 90.0, latitude)
        longitude = np.where(pole, projection.stand_lon, longitude)
    longitude = np.where(longitude > 180.0, longitude - 360.0, longitude)
    longitude = np.where(longitude < -180.0, longitude + 360.0, longitude)
    return latitude, longitude


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MeshCoordinates:
    """Cell centres read from a grid, static, init or history file."""

    latitude_degrees: np.ndarray
    longitude_degrees: np.ndarray
    cell_ids: np.ndarray
    source_path: Path
    source_sha256: str

    @property
    def n_cells(self) -> int:
        return int(self.latitude_degrees.size)


def read_mesh_coordinates(path: str | Path, *, sha256: str | None = None) -> MeshCoordinates:
    """Read ``latCell``/``lonCell`` and bind the file by digest.

    Longitudes are normalised to ``[-180, 180)``.  A rotated mesh (the
    ``grid_rotate`` output the port pins) stores the post-rotation earth
    coordinates directly, so no de-rotation is applied or wanted here: the
    cell centre latitude and longitude in the file are where that cell is on
    the Earth.  The converter cross-checks this against the history file's own
    coordinates before it resamples anything.
    """

    target = Path(path).expanduser().resolve(strict=True)
    digest = sha256_file(target)
    if sha256 is not None and digest != sha256:
        raise ResampleWeightsRefusal(
            f"mesh {target} is sha256 {digest}, expected {sha256}"
        )
    with Dataset(target, "r") as dataset:
        dataset.set_auto_mask(False)
        for name in ("latCell", "lonCell"):
            if name not in dataset.variables:
                raise ResampleWeightsRefusal(f"mesh {target} has no {name}")
            units = str(getattr(dataset.variables[name], "units", "")).lower()
            if units not in {"rad", "radian", "radians"}:
                raise ResampleWeightsRefusal(
                    f"mesh {name} units {units!r} are not radians"
                )
        latitude = np.asarray(dataset.variables["latCell"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["lonCell"][:], dtype=np.float64)
        if "indexToCellID" in dataset.variables:
            cell_ids = np.ascontiguousarray(
                np.asarray(dataset.variables["indexToCellID"][:], dtype=np.int32)
            )
        else:
            cell_ids = np.arange(1, latitude.size + 1, dtype=np.int32)
    if latitude.shape != longitude.shape or latitude.ndim != 1:
        raise ResampleWeightsRefusal("mesh latCell/lonCell shapes disagree")
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ResampleWeightsRefusal("mesh coordinates are not finite")
    if np.any(np.abs(latitude) > np.pi / 2.0 + 1.0e-9):
        raise ResampleWeightsRefusal("mesh latitude leaves [-pi/2, pi/2]")
    latitude_degrees = latitude * _DEG
    longitude_degrees = np.mod(longitude * _DEG + 180.0, 360.0) - 180.0
    return MeshCoordinates(
        latitude_degrees=np.ascontiguousarray(latitude_degrees),
        longitude_degrees=np.ascontiguousarray(longitude_degrees),
        cell_ids=cell_ids,
        source_path=target,
        source_sha256=digest,
    )


def _unit_vectors(latitude_degrees: np.ndarray, longitude_degrees: np.ndarray) -> np.ndarray:
    lat = np.asarray(latitude_degrees, dtype=np.float64) * _RAD
    lon = np.asarray(longitude_degrees, dtype=np.float64) * _RAD
    cos_lat = np.cos(lat)
    return np.ascontiguousarray(
        np.stack(
            (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=-1
        )
    )


@dataclass(frozen=True)
class NearestCellWeights:
    """One cached nearest-cell index map for one mesh and one window."""

    window_name: str
    window: Window
    target_latitude: np.ndarray
    target_longitude: np.ndarray
    cell_index: np.ndarray
    distance_km: np.ndarray
    mesh_sha256: str
    mesh_path: str
    n_cells: int
    weights_sha256: str
    build_seconds: float
    schema: str = WEIGHTS_SCHEMA
    method: str = RESAMPLE_METHOD

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.cell_index.shape[0]), int(self.cell_index.shape[1]))

    def apply(self, values: np.ndarray, *, cell_axis: int = -1) -> np.ndarray:
        """Gather ``values`` (indexed by cell) onto the target window.

        The result is exactly the source array's values, reordered.  No
        arithmetic is performed, so the output dtype and every bit pattern in
        it come straight from the model field.
        """

        array = np.asarray(values)
        if array.shape[cell_axis] != self.n_cells:
            raise ResampleWeightsRefusal(
                f"field axis {cell_axis} is {array.shape[cell_axis]}; "
                f"mesh has {self.n_cells} cells"
            )
        moved = np.moveaxis(array, cell_axis, -1)
        gathered = moved[..., self.cell_index]
        return np.ascontiguousarray(gathered)

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "window": self.window_name,
            "window_spec": self.window.spec(),
            "method": self.method,
            "mesh_sha256": self.mesh_sha256,
            "mesh_path": self.mesh_path,
            "n_cells": self.n_cells,
            "shape": list(self.shape),
            "weights_sha256": self.weights_sha256,
            "max_nearest_distance_km": float(self.distance_km.max()),
            "mean_nearest_distance_km": float(self.distance_km.mean()),
            "conservative": False,
        }


def _weights_digest(window_name: str, spec: Mapping[str, Any], mesh_sha256: str,
                    cell_index: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(WEIGHTS_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(RESAMPLE_METHOD.encode("ascii"))
    digest.update(b"\0")
    digest.update(window_name.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_sha256(dict(spec)).encode("ascii"))
    digest.update(b"\0")
    digest.update(mesh_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(cell_index, dtype=np.int32).tobytes(order="C"))
    return digest.hexdigest()


def cache_path(cache_dir: str | Path, window_name: str, mesh_sha256: str,
               window: Window) -> Path:
    spec_digest = _canonical_sha256(window.spec())[:12]
    name = f"nearest-{window_name}-{mesh_sha256[:16]}-{spec_digest}.npz"
    return Path(cache_dir).expanduser().resolve() / name


def build_weights(
    mesh: MeshCoordinates,
    window_name: str,
    *,
    window: Window | None = None,
) -> NearestCellWeights:
    """Build one nearest-cell index map.  No cache is consulted or written."""

    chosen = window if window is not None else WINDOWS.get(window_name)
    if chosen is None:
        raise ResampleWeightsRefusal(f"unknown render window {window_name!r}")
    started = time.perf_counter()
    target_latitude, target_longitude = chosen.coordinates()
    if target_latitude.shape != target_longitude.shape or target_latitude.ndim != 2:
        raise ResampleWeightsRefusal("window coordinates must be a 2-D pair")
    tree = cKDTree(_unit_vectors(mesh.latitude_degrees, mesh.longitude_degrees))
    targets = _unit_vectors(target_latitude, target_longitude).reshape(-1, 3)
    chord, index = tree.query(targets, k=1, workers=-1)
    elapsed = time.perf_counter() - started
    cell_index = np.ascontiguousarray(
        np.asarray(index, dtype=np.int32).reshape(target_latitude.shape)
    )
    great_circle_km = (
        2.0
        * np.arcsin(np.clip(np.asarray(chord, dtype=np.float64) / 2.0, 0.0, 1.0))
        * MPAS_EARTH_RADIUS_M
        / 1000.0
    )
    distance_km = np.ascontiguousarray(
        great_circle_km.reshape(target_latitude.shape).astype(np.float32)
    )
    if np.any(cell_index < 0) or np.any(cell_index >= mesh.n_cells):
        raise ResampleWeightsRefusal("nearest-cell index left the mesh")
    return NearestCellWeights(
        window_name=window_name,
        window=chosen,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        cell_index=cell_index,
        distance_km=distance_km,
        mesh_sha256=mesh.source_sha256,
        mesh_path=str(mesh.source_path),
        n_cells=mesh.n_cells,
        weights_sha256=_weights_digest(
            window_name, chosen.spec(), mesh.source_sha256, cell_index
        ),
        build_seconds=float(elapsed),
    )


def save_weights(weights: NearestCellWeights, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = {
        "schema": WEIGHTS_SCHEMA,
        "method": RESAMPLE_METHOD,
        "window_name": weights.window_name,
        "window_spec": weights.window.spec(),
        "window_description": weights.window.description,
        "mesh_sha256": weights.mesh_sha256,
        "mesh_path": weights.mesh_path,
        "n_cells": weights.n_cells,
        "weights_sha256": weights.weights_sha256,
        "build_seconds": weights.build_seconds,
        "projection_attributes": weights.window.projection_attributes(),
    }
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(
                    json.dumps(payload, sort_keys=True, allow_nan=False)
                ),
                target_latitude=weights.target_latitude,
                target_longitude=weights.target_longitude,
                cell_index=weights.cell_index,
                distance_km=weights.distance_km,
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_weights(path: str | Path) -> NearestCellWeights:
    target = Path(path).expanduser().resolve(strict=True)
    with np.load(target, allow_pickle=False) as archive:
        payload = json.loads(str(archive["metadata"]))
        cell_index = np.ascontiguousarray(archive["cell_index"])
        target_latitude = np.ascontiguousarray(archive["target_latitude"])
        target_longitude = np.ascontiguousarray(archive["target_longitude"])
        distance_km = np.ascontiguousarray(archive["distance_km"])
    if payload.get("schema") != WEIGHTS_SCHEMA:
        raise ResampleWeightsRefusal(f"{target} is not {WEIGHTS_SCHEMA}")
    spec = payload["window_spec"]
    window = _window_from_spec(spec, payload.get("window_description", ""))
    expected = _weights_digest(
        payload["window_name"], spec, payload["mesh_sha256"], cell_index
    )
    if expected != payload.get("weights_sha256"):
        raise ResampleWeightsRefusal(f"{target} weights digest does not bind its index")
    return NearestCellWeights(
        window_name=payload["window_name"],
        window=window,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        cell_index=cell_index,
        distance_km=distance_km,
        mesh_sha256=payload["mesh_sha256"],
        mesh_path=payload["mesh_path"],
        n_cells=int(payload["n_cells"]),
        weights_sha256=payload["weights_sha256"],
        build_seconds=float(payload.get("build_seconds", 0.0)),
    )


def _window_from_spec(spec: Mapping[str, Any], description: str) -> Window:
    kind = spec.get("kind")
    if kind == "latlon":
        return LatLonWindow(
            south=spec["south"],
            north=spec["north"],
            west=spec["west"],
            east=spec["east"],
            spacing_degrees=spec["spacing_degrees"],
            description=description,
        )
    if kind == "lambert":
        return LambertWindow(
            centre_lat=spec["centre_lat"],
            centre_lon=spec["centre_lon"],
            dx_metres=spec["dx_metres"],
            nx=int(spec["nx"]),
            ny=int(spec["ny"]),
            truelat1=spec["truelat1"],
            truelat2=spec["truelat2"],
            stand_lon=spec["stand_lon"],
            description=description,
        )
    raise ResampleWeightsRefusal(f"unknown window kind {kind!r}")


def weights_for(
    mesh_path: str | Path,
    window_name: str,
    *,
    cache_dir: str | Path,
    mesh_sha256: str | None = None,
    rebuild: bool = False,
) -> tuple[NearestCellWeights, Path, bool]:
    """Cached accessor: ``(weights, cache_file, built_now)``."""

    window = WINDOWS.get(window_name)
    if window is None:
        raise ResampleWeightsRefusal(f"unknown render window {window_name!r}")
    digest = sha256_file(mesh_path)
    if mesh_sha256 is not None and digest != mesh_sha256:
        raise ResampleWeightsRefusal(
            f"mesh {mesh_path} is sha256 {digest}, expected {mesh_sha256}"
        )
    destination = cache_path(cache_dir, window_name, digest, window)
    if destination.is_file() and not rebuild:
        cached = load_weights(destination)
        if cached.mesh_sha256 != digest:
            raise ResampleWeightsRefusal("cached weights bind a different mesh")
        return cached, destination, False
    mesh = read_mesh_coordinates(mesh_path, sha256=digest)
    weights = build_weights(mesh, window_name, window=window)
    save_weights(weights, destination)
    return weights, destination, True


def verify_nearest_cell(
    weights: NearestCellWeights,
    mesh: MeshCoordinates,
    *,
    samples: int = 20,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Prove the gather is exact: resampled value == source cell value.

    The check runs on a synthetic field whose value is the cell index itself
    plus a float32 field with awkward bit patterns, so a bit-for-bit equality
    failure cannot hide behind a benign-looking number.
    """

    rng = np.random.default_rng(seed)
    ny, nx = weights.shape
    rows = rng.integers(0, ny, size=samples)
    columns = rng.integers(0, nx, size=samples)
    field = np.asarray(
        rng.standard_normal(mesh.n_cells) * np.float32(1.0e7), dtype=np.float32
    )
    resampled = weights.apply(field)
    records: list[dict[str, Any]] = []
    exact = True
    for row, column in zip(rows.tolist(), columns.tolist()):
        cell = int(weights.cell_index[row, column])
        source = field[cell]
        target = resampled[row, column]
        identical = source.tobytes() == target.tobytes()
        exact = exact and identical
        records.append(
            {
                "row": row,
                "column": column,
                "cell": cell,
                "cell_id": int(mesh.cell_ids[cell]),
                "target_lat": float(weights.target_latitude[row, column]),
                "target_lon": float(weights.target_longitude[row, column]),
                "source_lat": float(mesh.latitude_degrees[cell]),
                "source_lon": float(mesh.longitude_degrees[cell]),
                "great_circle_km": float(weights.distance_km[row, column]),
                "source_value": float(source),
                "resampled_value": float(target),
                "bitwise_identical": identical,
            }
        )
    whole_field = np.array_equal(
        resampled, field[weights.cell_index], equal_nan=False
    )
    return {
        "samples": samples,
        "all_bitwise_identical": bool(exact),
        "whole_window_bitwise_identical": bool(whole_field),
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path,
                        help="MPAS grid/static/init file carrying latCell and lonCell")
    parser.add_argument("--mesh-sha256", default=None,
                        help="refuse unless the mesh file has this digest")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--windows", default="global,focus",
                        help="comma separated window names, or 'all'")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="spot check 20 random target points bit-for-bit")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the build report to this path")
    return parser


DEMOTION_EXIT_CODE = 78
REFERENCE_ACKNOWLEDGEMENT = "--regenerate-reference"

_DEMOTION_REFUSAL = """\
refused: {tool} is not a product path.

This pair reads MPAS history, regrids it onto a structured window and writes
the NetCDF frame the renderer draws -- read, regrid and write are all Rust-only
under the 2.5 Python boundary, so running it puts data-path Python back on the
render route.

The Rust crate rw-mpas owns this route and was proven bit for bit against these
two files (identical weights digests on both windows, 404,772,334 float values
compared, 127 of 127 PNGs byte-identical -- evidence/rw-mpas-converter-parity.json).
Convert with the shipped binary instead:

    rw_mpas_convert --history HIST.nc [HIST2.nc ...] --mesh MESH.nc \\
        --out-dir OUT [--init INIT.nc] [--window focus|global] \\
        [--field-set full|surface] [--format cdf2|cdf5] [--json REPORT.json]

    (built from tools/rustwx: cargo build --release -p rw-mpas)

These files are kept as the reference generator that receipt is about, so if
you are REGENERATING that reference -- not producing a frame to render -- pass
{acknowledgement} and re-run. Frames it writes are fixtures, not products.\
"""


def refuse_unless_reference_regeneration(
    tool: str, argv: list[str], stream: Any = None
) -> int | None:
    """Return an exit code to stop on, or ``None`` to proceed.

    The breakage: data-path Python back on the render route, producing frames
    a user would take for products.  The remedy is printed in full -- the real
    ``rw_mpas_convert`` command -- alongside the one door that still opens.
    """

    import sys as _sys

    if REFERENCE_ACKNOWLEDGEMENT in argv:
        return None
    print(
        _DEMOTION_REFUSAL.format(
            tool=tool, acknowledgement=REFERENCE_ACKNOWLEDGEMENT
        ),
        file=_sys.stderr if stream is None else stream,
    )
    return DEMOTION_EXIT_CODE


def strip_reference_acknowledgement(argv: list[str]) -> list[str]:
    """The acknowledgement is a gate token, not a conversion option."""

    return [item for item in argv if item != REFERENCE_ACKNOWLEDGEMENT]


def _selected_windows(argument: str) -> Iterable[str]:
    if argument.strip() == "all":
        return tuple(WINDOWS)
    names = tuple(name.strip() for name in argument.split(",") if name.strip())
    unknown = [name for name in names if name not in WINDOWS]
    if unknown:
        raise ResampleWeightsRefusal(f"unknown render windows: {', '.join(unknown)}")
    return names


def main(argv: list[str] | None = None) -> int:
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    refusal = refuse_unless_reference_regeneration(
        "mpas_resample_weights.py", arguments
    )
    if refusal is not None:
        return refusal
    args = _parser().parse_args(strip_reference_acknowledgement(arguments))
    report: dict[str, Any] = {
        "schema": WEIGHTS_SCHEMA,
        "mesh": str(Path(args.mesh).expanduser().resolve(strict=True)),
        "windows": {},
    }
    mesh: MeshCoordinates | None = None
    for name in _selected_windows(args.windows):
        started = time.perf_counter()
        weights, path, built = weights_for(
            args.mesh,
            name,
            cache_dir=args.cache_dir,
            mesh_sha256=args.mesh_sha256,
            rebuild=args.rebuild,
        )
        wall = time.perf_counter() - started
        entry = {
            **weights.provenance(),
            "cache_file": str(path),
            "built_now": built,
            "build_seconds": weights.build_seconds,
            "wall_seconds": wall,
            "description": weights.window.description,
            "projection_attributes": weights.window.projection_attributes(),
        }
        if args.verify:
            if mesh is None:
                mesh = read_mesh_coordinates(args.mesh)
            entry["verification"] = verify_nearest_cell(weights, mesh)
        report["windows"][name] = entry
        print(
            f"{name}: {weights.shape[0]}x{weights.shape[1]} "
            f"built={built} build_s={weights.build_seconds:.2f} "
            f"max_nn_km={float(weights.distance_km.max()):.2f} -> {path}"
        )
        if args.verify:
            verification = report["windows"][name]["verification"]
            print(
                f"  verify: bitwise_identical="
                f"{verification['all_bitwise_identical']} "
                f"whole_window={verification['whole_window_bitwise_identical']}"
            )
    if args.json is not None:
        destination = Path(args.json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
            newline="\n",
        )
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
