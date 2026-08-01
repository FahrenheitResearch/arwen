"""The model grid an observation is superobbed onto, and its identity.

A gridded observation set is only meaningful beside the grid it was gridded
to, so the grid is a first-class, hashed object here rather than an implied
one.  :meth:`TargetGrid.identity_sha256` digests the projection descriptor
*and* the coordinate arrays, using the same canonicalization as
:meth:`gpuwm.ingest.hrrr_target.HrrrTargetDomain.identity_sha256`, so two
writers that agree on the grid produce the same digest and a consumer that
disagrees fails closed instead of silently assimilating into the wrong
columns.

:meth:`TargetGrid.from_wrfout` reconstructs the projection from the file's
global attributes and then **checks it against the file's own XLAT/XLONG**.
Reconstruction from attributes is the fast, exact path; trusting it without
checking is how a nest with a non-centred reference point quietly shifts
every observation by a cell.  The check is the artifact, not the model of
it, and a mismatch is a hard error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwm.static.projection import (EARTH_RADIUS_M, WPS_MAP_PROJ_NAMES,
                                     ProjectedGrid, projection_class)

#: Contract string for the serialized grid descriptor.
TARGET_GRID_SCHEMA = "gpuwm-obs.radar-target-grid.v1"

#: Largest lat/lon disagreement (degrees) tolerated between the projection
#: rebuilt from a wrfout's attributes and the XLAT/XLONG the file carries.
#: 1e-5 deg is about a metre — far below any grid spacing ArWen runs, and
#: far above float32 storage noise in the file.
GEOREF_TOLERANCE_DEG = 1.0e-5

_STANDARD_GRAVITY = 9.81


class GridMismatchError(ValueError):
    """The grid in hand is not the grid the caller demanded."""


@dataclass(frozen=True)
class TargetGrid:
    """A model grid with full 3-D georeference, hashed.

    ``lat``/``lon``/``terrain_m`` are ``(ny, nx)`` at mass points;
    ``z_w`` is ``(nz + 1, ny, nx)`` layer-interface heights above mean sea
    level, so ``z_w[0]`` is the terrain surface.  Levels are model levels,
    not height levels: an observation lands in the layer whose interfaces
    bracket it *in its own column*, which is the only placement that means
    anything on terrain-following coordinates.
    """

    name: str
    map_proj: str
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    ref_lat: float
    ref_lon: float
    truelat1: float
    truelat2: float
    stand_lon: float
    lat: np.ndarray
    lon: np.ndarray
    z_w: np.ndarray
    terrain_m: np.ndarray
    projection: ProjectedGrid
    source: str

    def __post_init__(self) -> None:
        if self.lat.shape != (self.ny, self.nx):
            raise ValueError(
                f"lat has shape {self.lat.shape}, expected {(self.ny, self.nx)}")
        if self.lon.shape != self.lat.shape:
            raise ValueError("lat and lon must share a shape")
        if self.z_w.shape != (self.nz + 1, self.ny, self.nx):
            raise ValueError(
                f"z_w has shape {self.z_w.shape}, expected "
                f"{(self.nz + 1, self.ny, self.nx)}")
        if self.terrain_m.shape != self.lat.shape:
            raise ValueError("terrain must share the mass-point shape")
        for name, array in (("lat", self.lat), ("lon", self.lon),
                            ("z_w", self.z_w), ("terrain_m", self.terrain_m)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} carries non-finite values")
        if np.any(np.diff(self.z_w, axis=0) <= 0.0):
            raise ValueError(
                "z_w must increase monotonically with level in every column")

    # -- identity ----------------------------------------------------------

    def descriptor(self) -> dict:
        """The projection half of the identity: small, readable, exact."""

        return {
            "schema": TARGET_GRID_SCHEMA,
            "name": self.name,
            "map_proj": self.map_proj,
            "nx": int(self.nx),
            "ny": int(self.ny),
            "nz": int(self.nz),
            "dx_m": float(self.dx_m),
            "dy_m": float(self.dy_m),
            "ref_lat": float(self.ref_lat),
            "ref_lon": float(self.ref_lon),
            "truelat1": float(self.truelat1),
            "truelat2": float(self.truelat2),
            "stand_lon": float(self.stand_lon),
            "earth_radius_m": float(EARTH_RADIUS_M),
        }

    def identity_sha256(self) -> str:
        """Digest of the descriptor and every coordinate array.

        The arrays are in because two grids can share every scalar and
        still differ — a regenerated terrain field, a different vertical
        stretching — and an observation set is bound to the columns it was
        placed in, not to the namelist that nominally produced them.
        """

        payload = dict(self.descriptor())
        payload["arrays"] = {
            name: _array_sha256(array)
            for name, array in (("lat", self.lat), ("lon", self.lon),
                                ("z_w", self.z_w),
                                ("terrain_m", self.terrain_m))
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def require_identity(self, expected_sha256: str) -> None:
        """Fail closed unless this grid is exactly the one named."""

        actual = self.identity_sha256()
        if actual != expected_sha256:
            raise GridMismatchError(
                f"target grid identity {actual} does not match the required "
                f"{expected_sha256}; refusing to grid observations onto a "
                "different grid than the one the product claims")

    # -- placement ---------------------------------------------------------

    def mass_index(self, lat, lon):
        """Fractional mass-point index ``(i, j)`` for latitudes/longitudes.

        ``i`` runs west-east and ``j`` south-north, both zero-based, so an
        exact grid point returns an exact integer.  WPS projection
        coordinates are one-based at mass points (``latlon_mass`` samples
        ``x = 1 .. e_we-1``), hence the offset.
        """

        x, y = self.projection.latlon_to_ij(lat, lon)
        return np.asarray(x) - 1.0, np.asarray(y) - 1.0

    def level_index(self, i_index, j_index, height_msl_m):
        """Model level containing each height, or -1 outside the column.

        ``i_index``/``j_index`` are integer mass-point indices already known
        to be inside the grid.  A gate below the terrain surface or above
        the model top returns -1: fail closed, and let the caller count it.
        """

        i_index = np.asarray(i_index, dtype=np.intp)
        j_index = np.asarray(j_index, dtype=np.intp)
        height = np.asarray(height_msl_m, dtype=np.float64)
        column = self.z_w[:, j_index, i_index]          # (nz+1, n)
        # searchsorted per column, vectorized: count interfaces below.
        level = (column[1:, :] <= height[None, :]).sum(axis=0)
        inside = (height >= column[0, :]) & (height < column[-1, :])
        return np.where(inside, level, -1)

    def inside(self, i_index, j_index):
        """Mask of index pairs that land on a mass point of this grid."""

        i_index = np.asarray(i_index)
        j_index = np.asarray(j_index)
        return ((i_index >= 0) & (i_index < self.nx)
                & (j_index >= 0) & (j_index < self.ny))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_projection(cls, projection: ProjectedGrid, *, z_w: np.ndarray,
                        terrain_m: np.ndarray | None = None,
                        name: str = "target", source: str = "projection"
                        ) -> "TargetGrid":
        """Build a target grid from a projection plus a vertical structure."""

        lat, lon = projection.latlon_mass()
        ny, nx = lat.shape
        z_w = np.asarray(z_w, dtype=np.float64)
        if z_w.ndim == 1:
            z_w = np.broadcast_to(z_w[:, None, None], (z_w.size, ny, nx))
        z_w = np.ascontiguousarray(z_w, dtype=np.float64)
        if terrain_m is None:
            terrain_m = z_w[0].copy()
        return cls(
            name=name,
            map_proj=projection.map_proj,
            nx=nx, ny=ny, nz=z_w.shape[0] - 1,
            dx_m=float(projection.dx), dy_m=float(projection.dy),
            ref_lat=float(projection.ref_lat),
            ref_lon=float(projection.ref_lon),
            truelat1=float(projection.truelat1),
            truelat2=float(projection.truelat2),
            stand_lon=float(projection.stand_lon),
            lat=np.ascontiguousarray(lat, dtype=np.float64),
            lon=np.ascontiguousarray(lon, dtype=np.float64),
            z_w=z_w,
            terrain_m=np.ascontiguousarray(terrain_m, dtype=np.float64),
            projection=projection,
            source=source)

    @classmethod
    def from_wrfout(cls, path: str | Path, *, frame: int = 0,
                    name: str | None = None) -> "TargetGrid":
        """Read a target grid out of a wrfout history file.

        Uses the file's own XLAT/XLONG to *verify* the projection rebuilt
        from its global attributes, and PH+PHB to build the layer-interface
        heights.  Both are what wrfout already carries; nothing is
        reconstructed from a namelist that may no longer exist.
        """

        import netCDF4                                   # noqa: PLC0415

        path = Path(path)
        with netCDF4.Dataset(path, "r") as dataset:
            attrs = {key: dataset.getncattr(key) for key in dataset.ncattrs()}
            missing = [key for key in ("MAP_PROJ", "TRUELAT1", "TRUELAT2",
                                       "STAND_LON", "CEN_LAT", "CEN_LON",
                                       "DX", "DY")
                       if key not in attrs]
            if missing:
                raise GridMismatchError(
                    f"{path.name} is missing the georeference attributes "
                    f"{missing}; it cannot anchor an observation grid")
            for variable in ("XLAT", "XLONG", "PH", "PHB", "HGT"):
                if variable not in dataset.variables:
                    raise GridMismatchError(
                        f"{path.name} has no {variable}; an observation grid "
                        "needs full 3-D georeference, not a subset history")
            file_lat = np.asarray(dataset.variables["XLAT"][frame],
                                  dtype=np.float64)
            file_lon = np.asarray(dataset.variables["XLONG"][frame],
                                  dtype=np.float64)
            geopotential = (np.asarray(dataset.variables["PH"][frame],
                                       dtype=np.float64)
                            + np.asarray(dataset.variables["PHB"][frame],
                                         dtype=np.float64))
            terrain = np.asarray(dataset.variables["HGT"][frame],
                                 dtype=np.float64)

        proj_code = int(attrs["MAP_PROJ"])
        if proj_code not in WPS_MAP_PROJ_NAMES:
            raise GridMismatchError(
                f"{path.name} declares MAP_PROJ {proj_code}, which this "
                f"stage cannot georeference (known: "
                f"{sorted(WPS_MAP_PROJ_NAMES)})")
        map_proj = WPS_MAP_PROJ_NAMES[proj_code]
        ny, nx = file_lat.shape
        projection = projection_class(map_proj)(
            ref_lat=float(attrs["CEN_LAT"]), ref_lon=float(attrs["CEN_LON"]),
            truelat1=float(attrs["TRUELAT1"]),
            truelat2=float(attrs["TRUELAT2"]),
            stand_lon=float(attrs["STAND_LON"]),
            dx=float(attrs["DX"]), dy=float(attrs["DY"]),
            e_we=nx + 1, e_sn=ny + 1)

        rebuilt_lat, rebuilt_lon = projection.latlon_mass()
        lat_error = float(np.max(np.abs(rebuilt_lat - file_lat)))
        lon_error = float(np.max(np.abs(
            (rebuilt_lon - file_lon + 180.0) % 360.0 - 180.0)))
        if max(lat_error, lon_error) > GEOREF_TOLERANCE_DEG:
            raise GridMismatchError(
                f"{path.name}: the projection rebuilt from its global "
                f"attributes disagrees with its own XLAT/XLONG by "
                f"{lat_error:.3e} deg lat / {lon_error:.3e} deg lon "
                f"(tolerance {GEOREF_TOLERANCE_DEG:.1e}); refusing to place "
                "observations on a georeference the file contradicts")

        z_w = geopotential / _STANDARD_GRAVITY
        return cls.from_projection(
            projection, z_w=z_w, terrain_m=terrain,
            name=name or path.stem, source=f"wrfout:{path.name}")


def _array_sha256(array: np.ndarray) -> str:
    """Digest an array with its dtype and shape, C-ordered.

    Same construction as :mod:`gpuwm.ingest.prepared_cache`: the bytes alone
    would let a reshape or a dtype change pass as identical.
    """

    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    hasher = hashlib.sha256()
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(b";")
    hasher.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    hasher.update(b";")
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()
