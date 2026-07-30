"""Validated target-domain geometry for native HRRR preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from gpuwm.static.lambert import LambertGrid


TARGET_DOMAIN_SCHEMA = "gpuwm-hrrr-target-domain-v1"
# LambertGrid uses WRF staggered dimensions e_we=1800/e_sn=1060 to represent
# the canonical HRRR GRIB mass grid and Rust bridge contract nx=1799/ny=1059.
HRRR_SOURCE_NX = 1799
HRRR_SOURCE_NY = 1059


@dataclass(frozen=True)
class HrrrSourceWindow:
    """Zero-based inclusive HRRR source window with interpolation halos."""

    i_start: int
    i_end: int
    j_start: int
    j_end: int
    parabolic_lower_halo_cells: int
    parabolic_upper_halo_cells: int
    surface_fallback_radius_cells: int
    target_source_i_range: tuple[float, float]
    target_source_j_range: tuple[float, float]

    @property
    def nx(self) -> int:
        return self.i_end - self.i_start + 1

    @property
    def ny(self) -> int:
        return self.j_end - self.j_start + 1

    def bridge_tuple(self) -> tuple[int, int, int, int]:
        return self.i_start, self.i_end, self.j_start, self.j_end

    def to_dict(self) -> dict[str, object]:
        return {
            "zero_based_inclusive": {
                "i": [self.i_start, self.i_end],
                "j": [self.j_start, self.j_end],
            },
            "shape": [self.ny, self.nx],
            "parabolic_halo_cells": {
                "below_floor": self.parabolic_lower_halo_cells,
                "above_floor": self.parabolic_upper_halo_cells,
            },
            "surface_fallback_radius_cells": (
                self.surface_fallback_radius_cells),
            "target_source_i_range": list(self.target_source_i_range),
            "target_source_j_range": list(self.target_source_j_range),
        }


@dataclass(frozen=True)
class HrrrTargetDomain:
    """One supported Lambert target for HRRR IC/LBC generation."""

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
    time_step_seconds: int
    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    surface_fallback_radius_cells: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("target-domain name must be a non-empty string")
        if self.map_proj.lower() != "lambert":
            raise ValueError(
                "native HRRR target map_proj must be 'lambert'; other "
                "projection families are not certified")
        for field_name in (
                "nx", "ny", "nz", "time_step_seconds", "spec_bdy_width",
                "spec_zone", "relax_zone", "surface_fallback_radius_cells"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"target-domain {field_name} must be an integer")
        if self.nz < 4:
            raise ValueError("native HRRR target nz must be at least 4")
        if self.spec_bdy_width != self.spec_zone + self.relax_zone:
            raise ValueError(
                "spec_bdy_width must equal spec_zone + relax_zone")
        minimum_axis = 2 * self.spec_bdy_width + 3
        if self.nx < minimum_axis or self.ny < minimum_axis:
            raise ValueError(
                f"target nx and ny must each be at least {minimum_axis}")
        scalars = (
            "dx_m", "dy_m", "ref_lat", "ref_lon", "truelat1",
            "truelat2", "stand_lon",
        )
        for field_name in scalars:
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(
                    f"target-domain {field_name} must be finite")
        if self.dx_m <= 0.0 or self.dy_m <= 0.0:
            raise ValueError("target-domain dx_m and dy_m must be positive")
        if not math.isclose(
                self.dx_m, self.dy_m, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("Lambert target requires dx_m == dy_m")
        if self.time_step_seconds <= 0:
            raise ValueError("target-domain time_step_seconds must be positive")
        if not 0 <= self.surface_fallback_radius_cells <= 64:
            raise ValueError(
                "target-domain surface_fallback_radius_cells must be in "
                "[0, 64]")
        if not -89.0 < self.ref_lat < 89.0:
            raise ValueError("target-domain ref_lat must be inside (-89, 89)")
        if not -180.0 <= self.ref_lon <= 180.0:
            raise ValueError("target-domain ref_lon must be inside [-180, 180]")
        if not -180.0 <= self.stand_lon <= 180.0:
            raise ValueError("target-domain stand_lon must be inside [-180, 180]")
        if not (0.0 < self.truelat1 < 89.0
                and 0.0 < self.truelat2 < 89.0):
            raise ValueError(
                "CONUS HRRR Lambert targets require positive true latitudes "
                "inside (0, 89)")
        # Constructing the grid is part of validation: it catches a singular
        # cone and any future projection validation added by LambertGrid.
        grid = self.grid()
        latitude, longitude = grid.latlon_mass()
        if (latitude.shape != (self.ny, self.nx)
                or not np.isfinite(latitude).all()
                or not np.isfinite(longitude).all()):
            raise ValueError("target Lambert grid is non-finite or mis-shaped")

    @classmethod
    def legacy_500x500(cls) -> "HrrrTargetDomain":
        return cls(
            name="hrrr_native_easy_500x500x49",
            map_proj="lambert",
            nx=500,
            ny=500,
            nz=49,
            dx_m=999.8071015811862,
            dy_m=999.8071015811862,
            ref_lat=35.5028506728143,
            ref_lon=-98.0021669285660,
            truelat1=38.5,
            truelat2=38.5,
            stand_lon=-97.5,
            time_step_seconds=5,
        )

    def grid(self) -> LambertGrid:
        return LambertGrid(
            self.ref_lat,
            self.ref_lon,
            self.truelat1,
            self.truelat2,
            self.stand_lon,
            self.dx_m,
            self.dy_m,
            self.nx + 1,
            self.ny + 1,
        )

    def contract_cfg(self) -> SimpleNamespace:
        """The domain configuration ``native_geometry_contract`` expects.

        Every producer and every verifier of an HRRR geometry document goes
        through this, so the two sides cannot describe the same target with
        different keys.
        """

        return SimpleNamespace(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            dx=self.dx_m,
            dy=self.dy_m,
        )

    def to_payload(self) -> dict[str, object]:
        return {"schema": TARGET_DOMAIN_SCHEMA, **asdict(self)}

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def load_hrrr_target_domain(path: Path | str | None) -> HrrrTargetDomain:
    """Load a strict domain specification, or the sealed legacy geometry."""

    if path is None:
        return HrrrTargetDomain.legacy_500x500()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("HRRR target-domain document must be a JSON object")
    if payload.get("schema") != TARGET_DOMAIN_SCHEMA:
        raise ValueError("unsupported HRRR target-domain schema")
    values = dict(payload)
    values.pop("schema")
    expected = set(HrrrTargetDomain.__dataclass_fields__)
    optional_v1_defaults = {"surface_fallback_radius_cells": 8}
    missing = expected - set(values)
    extra = set(values) - expected
    if extra or not missing <= set(optional_v1_defaults):
        raise ValueError(
            "HRRR target-domain fields differ from the exact schema: "
            f"missing={sorted(missing)}, extra={sorted(extra)}")
    for name in missing:
        values[name] = optional_v1_defaults[name]
    return HrrrTargetDomain(**values)


def required_hrrr_source_window(
    target: HrrrTargetDomain | LambertGrid,
    *,
    surface_fallback_radius: int | None = None,
) -> HrrrSourceWindow:
    """Return the exact source crop needed by all target C-grid points.

    Atmospheric and hydrometeor interpolation uses a four-point parabolic
    stencil spanning ``floor(index)-1`` through ``floor(index)+2``.  Surface
    land/water matching may search a bounded radius around the nearest source
    mass point.  The returned crop covers both requirements and refuses a
    target that would need any point outside the native 1799 x 1059 HRRR mass
    grid.
    """

    if isinstance(target, HrrrTargetDomain):
        grid = target.grid()
        declared_radius = target.surface_fallback_radius_cells
    elif isinstance(target, LambertGrid):
        grid = target
        declared_radius = 8
    else:
        raise TypeError("target must be HrrrTargetDomain or LambertGrid")
    radius = (declared_radius if surface_fallback_radius is None
              else int(surface_fallback_radius))
    if not 0 <= radius <= 64:
        raise ValueError("surface_fallback_radius must be in [0, 64]")
    if (isinstance(target, HrrrTargetDomain)
            and surface_fallback_radius is not None
            and radius != declared_radius):
        raise ValueError(
            "surface_fallback_radius override differs from the target-domain "
            "policy")

    # Local import avoids a circular import: hrrr.py also consumes target
    # geometry when it rotates winds.
    from gpuwm.ingest.hrrr import hrrr_source_grid

    source = hrrr_source_grid()
    coordinates = []
    mass_coordinates = None
    for latitude, longitude in (
            grid.latlon_mass(), grid.latlon_u(), grid.latlon_v()):
        source_x, source_y = source.latlon_to_ij(latitude, longitude)
        zero_x = np.asarray(source_x, dtype=np.float64) - 1.0
        zero_y = np.asarray(source_y, dtype=np.float64) - 1.0
        if not np.isfinite(zero_x).all() or not np.isfinite(zero_y).all():
            raise ValueError("target maps to non-finite HRRR source coordinates")
        coordinates.append((zero_x, zero_y))
        if mass_coordinates is None:
            mass_coordinates = (zero_x, zero_y)

    parabolic_i_min = min(
        int(np.floor(x).min()) - 1 for x, _ in coordinates)
    parabolic_i_max = max(
        int(np.floor(x).max()) + 2 for x, _ in coordinates)
    parabolic_j_min = min(
        int(np.floor(y).min()) - 1 for _, y in coordinates)
    parabolic_j_max = max(
        int(np.floor(y).max()) + 2 for _, y in coordinates)
    assert mass_coordinates is not None
    mass_x, mass_y = mass_coordinates
    # Candidate donors must have integer coordinates inside the radius-R
    # disk.  ceil(min(q)-R)..floor(max(q)+R) is the exact union bounding box;
    # round(q)+/-R is safe in the interior but can falsely reject a valid
    # near-edge target by requesting an ineligible extra cell.
    fallback_i_min = int(np.ceil(mass_x.min() - radius))
    fallback_i_max = int(np.floor(mass_x.max() + radius))
    fallback_j_min = int(np.ceil(mass_y.min() - radius))
    fallback_j_max = int(np.floor(mass_y.max() + radius))
    i_start = min(parabolic_i_min, fallback_i_min)
    i_end = max(parabolic_i_max, fallback_i_max)
    j_start = min(parabolic_j_min, fallback_j_min)
    j_end = max(parabolic_j_max, fallback_j_max)
    if (i_start < 0 or j_start < 0
            or i_end >= HRRR_SOURCE_NX or j_end >= HRRR_SOURCE_NY):
        raise ValueError(
            "target domain plus required interpolation halo leaves HRRR "
            "coverage: required zero-based inclusive window "
            f"i={i_start}..{i_end}, j={j_start}..{j_end}; native limits "
            f"are i=0..{HRRR_SOURCE_NX - 1}, j=0..{HRRR_SOURCE_NY - 1}")
    source_i_range = (
        min(float(value[0].min()) for value in coordinates),
        max(float(value[0].max()) for value in coordinates),
    )
    source_j_range = (
        min(float(value[1].min()) for value in coordinates),
        max(float(value[1].max()) for value in coordinates),
    )
    return HrrrSourceWindow(
        i_start=i_start,
        i_end=i_end,
        j_start=j_start,
        j_end=j_end,
        parabolic_lower_halo_cells=1,
        parabolic_upper_halo_cells=2,
        surface_fallback_radius_cells=radius,
        target_source_i_range=source_i_range,
        target_source_j_range=source_j_range,
    )


__all__ = [
    "HRRR_SOURCE_NX",
    "HRRR_SOURCE_NY",
    "HrrrSourceWindow",
    "HrrrTargetDomain",
    "TARGET_DOMAIN_SCHEMA",
    "load_hrrr_target_domain",
    "required_hrrr_source_window",
]
