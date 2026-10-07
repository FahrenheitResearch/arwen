"""Validated target-domain geometry for native HRRR preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from gpuwm.ingest.source_coverage import SourceCoverageRefusal
from gpuwm.static.lambert import LambertGrid
from gpuwm.grid_requirements import NATIVE_TARGET_INTERIOR_AXIS, boundary_axis


TARGET_DOMAIN_SCHEMA = "gpuwm-hrrr-target-domain-v1"
# LambertGrid uses WRF staggered dimensions e_we=1800/e_sn=1060 to represent
# the canonical HRRR GRIB mass grid and Rust bridge contract nx=1799/ny=1059.
HRRR_SOURCE_NX = 1799
HRRR_SOURCE_NY = 1059

#: The largest surface-donor search radius any target-domain policy may
#: declare.  Both the domain document and the window override enforce
#: it, and validated remediation advice (gpuwm/ingest/hrrr.py) must not
#: recommend a raise beyond it.
SURFACE_FALLBACK_RADIUS_MAX = 64


@lru_cache(maxsize=1)
def hrrr_coverage_envelope() -> tuple[float, float, float, float]:
    """``(south, west, north, east)``: the lat/lon envelope of the native
    HRRR mass grid, computed from the grid definition itself.

    THE one HRRR coverage definition.  ``gpuwm fetch``'s ``--area`` gate
    and ``gpuwm domain``'s suggested fetch box both consume this (via
    :func:`gpuwm.fetch.source_coverage_envelope`), so the two sides of
    that contract cannot disagree the way the retired hand-held box in
    gpuwm/fetch.py (lat 21.1..52.7, lon -134.2..-60.8) disagreed with a
    wizard box that clamped only at the pole: the real grid tops out at
    52.6157 N -- north of every mass point the 52.70 cap admitted, and
    south of the 54.39 a fitted 3 km CONUS root's margined box named.

    Computed over the boundary ring of the 1799 x 1059 mass grid, which
    is exact, not an approximation: the projected north pole lies
    outside the grid rectangle, latitude is monotone in projected
    distance from the pole point and longitude in angle around it, so
    every lat/lon extreme over the (convex) rectangle is attained on
    its boundary.  A full-grid scan agrees to the last bit
    (tests/test_fetch.py).
    """

    # Local import, matching required_hrrr_source_window: hrrr.py also
    # consumes target geometry, so a module-scope import would be a cycle.
    from gpuwm.ingest.hrrr import hrrr_source_grid

    grid = hrrr_source_grid()
    # One-based mass coordinates, as LambertGrid registers them.
    i = np.arange(1, HRRR_SOURCE_NX + 1, dtype=np.float64)
    j = np.arange(1, HRRR_SOURCE_NY + 1, dtype=np.float64)
    ring_x = np.concatenate((i, i, np.full(j.size, 1.0),
                             np.full(j.size, float(HRRR_SOURCE_NX))))
    ring_y = np.concatenate((np.full(i.size, 1.0),
                             np.full(i.size, float(HRRR_SOURCE_NY)), j, j))
    latitude, longitude = grid.ij_to_latlon(ring_x, ring_y)
    if not (np.isfinite(latitude).all() and np.isfinite(longitude).all()):
        raise ValueError("HRRR native grid produced a non-finite envelope")
    return (float(latitude.min()), float(longitude.min()),
            float(latitude.max()), float(longitude.max()))


#: How far, in degrees, a target's south-west mass point may sit from the
#: native grid's own and still be the native grid
#: (:func:`native_grid_identity`).  The native grid is described twice:
#: WPS's sphere (6,370 km, where the model integrates and where a target
#: spelled from HRRR's own namelist.wps lands) and the GRIB header's
#: (6,371.229 km, :data:`gpuwm.ingest.hrrr.HRRR_EARTH_RADIUS_M`, which
#: :func:`gpuwm.ingest.hrrr.hrrr_source_grid` reproduces).  The two
#: agree at the south-west corner by construction and drift apart by
#: the radius ratio (MEASURED: +0.173 / +0.102 cells in x / y at the
#: centre, +0.347 / +0.204 at the north-east corner).  This tolerance
#: covers the GRIB anchor's rounding to six decimal places, without
#: admitting a nearby shifted grid.
NATIVE_GRID_CORNER_TOLERANCE_DEG = 3.0e-6

#: Numerical tolerance against either of the two declared spacings:
#: the WPS spacing or its GRIB-sphere equivalent.  An arbitrary spacing
#: between those two is a different grid and must still interpolate.
NATIVE_GRID_SPACING_RTOL = 1.0e-9


def native_grid_identity(grid: LambertGrid) -> bool:
    """Whether ``grid`` IS the native HRRR mass grid, point for point.

    True when the target has the native grid's dimensions, cone and
    standard longitude, one of the two declared spacings within
    :data:`NATIVE_GRID_SPACING_RTOL`, and its south-west mass point within
    :data:`NATIVE_GRID_CORNER_TOLERANCE_DEG` of the native grid's.  Such a
    target takes the identity route: every mass point copies its own
    source cell, nothing is interpolated between cells, and no
    interpolation halo past the grid's edge is demanded.  The route that
    refused this grid demanded that halo for the outermost row and
    column, and the one-row trim it advised moved the relaxation zone
    one row inward of HRRR's own.
    """
    from gpuwm.ingest.hrrr import (
        HRRR_GRID_SPACING_M, HRRR_WPS_EQUIVALENT_DX_M, hrrr_source_grid)

    if not isinstance(grid, LambertGrid):
        return False
    if (int(grid.e_we) - 1, int(grid.e_sn) - 1) != (HRRR_SOURCE_NX,
                                                     HRRR_SOURCE_NY):
        return False
    source = hrrr_source_grid()
    if not (math.isclose(float(grid.truelat1), float(source.truelat1),
                         abs_tol=1.0e-9)
            and math.isclose(float(grid.truelat2), float(source.truelat2),
                             abs_tol=1.0e-9)
            and math.isclose(float(grid.stand_lon), float(source.stand_lon),
                             abs_tol=1.0e-9)):
        return False
    if not any(math.isclose(float(grid.dx), float(spacing),
                            rel_tol=NATIVE_GRID_SPACING_RTOL)
               for spacing in (HRRR_GRID_SPACING_M, HRRR_WPS_EQUIVALENT_DX_M)):
        return False
    corner_lat, corner_lon = grid.ij_to_latlon(1.0, 1.0)
    native_lat, native_lon = source.ij_to_latlon(1.0, 1.0)
    return (abs(float(corner_lat) - float(native_lat))
            <= NATIVE_GRID_CORNER_TOLERANCE_DEG
            and abs(float(corner_lon) - float(native_lon))
            <= NATIVE_GRID_CORNER_TOLERANCE_DEG)


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
    #: ``"interpolated"`` (every window before the identity route) or
    #: ``"identity"``: the target is the native grid itself
    #: (:func:`native_grid_identity`), the window is the whole grid and
    #: no halo past its edge is demanded.
    route: str = "interpolated"

    @property
    def nx(self) -> int:
        return self.i_end - self.i_start + 1

    @property
    def ny(self) -> int:
        return self.j_end - self.j_start + 1

    def bridge_tuple(self) -> tuple[int, int, int, int]:
        return self.i_start, self.i_end, self.j_start, self.j_end

    def to_dict(self) -> dict[str, object]:
        document = {
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
        if self.route != "interpolated":
            # Written only on the identity route, so every interpolated
            # window's record is byte for byte what it was.
            document["route"] = self.route
        return document

    def matches_record(self, recorded) -> bool:
        """Whether a recorded :meth:`to_dict` describes this window.

        The crop, halos and radius are integers and are compared exactly.
        The target's extent in source index units comes out of projection
        arithmetic, whose last digit depends on the machine that ran it, so
        it is compared within
        :data:`gpuwm.static.grid_identity.GRID_POSITION_TOLERANCE_CELLS` of
        a source cell.  The target itself is bound exactly beside this by
        its identity, so the allowance only ever admits rounding.
        """

        from gpuwm.static.grid_identity import GRID_POSITION_TOLERANCE_CELLS

        expected = self.to_dict()
        ranges = ("target_source_i_range", "target_source_j_range")
        if not isinstance(recorded, Mapping) or set(recorded) != set(expected):
            return False
        if any(recorded[key] != expected[key]
               for key in expected if key not in ranges):
            return False
        for key in ranges:
            values = recorded[key]
            if (not isinstance(values, (list, tuple)) or len(values) != 2
                    or any(isinstance(value, bool)
                           or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in values)):
                return False
            if any(abs(float(value) - float(bound))
                   > GRID_POSITION_TOLERANCE_CELLS
                   for value, bound in zip(values, expected[key])):
                return False
        return True


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
    #: The root clock, spelled the way WRF's own registry spells it
    #: (Registry.EM_COMMON: integer ``time_step`` plus the exact
    #: rational remainder).  A 1.5 km root runs a 7.5 s clock; an
    #: integer-only field refused that ladder outright, so the spec
    #: carries whole seconds here and the remainder in the two fract
    #: fields below -- exact, never a float.
    time_step_seconds: int
    time_step_fract_num: int = 0
    time_step_fract_den: int = 1
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
                "nx", "ny", "nz", "time_step_seconds",
                "time_step_fract_num", "time_step_fract_den",
                "spec_bdy_width", "spec_zone", "relax_zone",
                "surface_fallback_radius_cells"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"target-domain {field_name} must be an integer")
        if self.nz < 4:
            raise ValueError("native HRRR target nz must be at least 4")
        if self.spec_bdy_width != self.spec_zone + self.relax_zone:
            raise ValueError(
                "spec_bdy_width must equal spec_zone + relax_zone")
        minimum_axis = boundary_axis(
            self.spec_bdy_width, interior_points=NATIVE_TARGET_INTERIOR_AXIS)
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
        if self.time_step_seconds < 0:
            raise ValueError(
                "target-domain time_step_seconds must be non-negative "
                "whole seconds (the fractional remainder rides "
                "time_step_fract_num/den)")
        if self.time_step_fract_den < 1:
            raise ValueError(
                "target-domain time_step_fract_den must be at least 1")
        if not 0 <= self.time_step_fract_num < self.time_step_fract_den:
            raise ValueError(
                "target-domain time_step_fract_num must be a proper "
                "remainder: 0 <= num < den")
        if self.time_step_exact <= 0:
            raise ValueError("target-domain time step must be positive")
        if not (0 <= self.surface_fallback_radius_cells
                <= SURFACE_FALLBACK_RADIUS_MAX):
            raise ValueError(
                "target-domain surface_fallback_radius_cells must be in "
                f"[0, {SURFACE_FALLBACK_RADIUS_MAX}]")
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

    @property
    def time_step_exact(self) -> Fraction:
        """The root clock as the exact rational the three fields spell."""

        return (Fraction(self.time_step_seconds)
                + Fraction(self.time_step_fract_num,
                           self.time_step_fract_den))

    def to_payload(self) -> dict[str, object]:
        payload = {"schema": TARGET_DOMAIN_SCHEMA, **asdict(self)}
        # A whole-second clock's payload is byte-identical to what this
        # writer emitted before the fract fields existed, so every
        # stored identity_sha256 of an integer-clock target survives
        # the widening.  Only a genuinely fractional clock carries the
        # two extra keys.
        if self.time_step_fract_num == 0 and self.time_step_fract_den == 1:
            payload.pop("time_step_fract_num")
            payload.pop("time_step_fract_den")
        return payload

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
    optional_v1_defaults = {
        "surface_fallback_radius_cells": 8,
        # The rational-clock remainder (whole-second documents omit it;
        # a 1.5 km ladder's 7.5 s root clock carries 1/2 here).
        "time_step_fract_num": 0,
        "time_step_fract_den": 1,
    }
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
    mass point.  The returned crop covers both requirements.  A target whose
    parabolic stencil needs any point outside the native 1799 x 1059 HRRR
    mass grid is refused.  The surface search is clipped at the native
    grid's edge instead: no donor exists beyond it, and the donor search
    treats a crop edge that is HRRR's own edge as closed, so clipping there
    cannot hide a nearer donor.
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
    if not 0 <= radius <= SURFACE_FALLBACK_RADIUS_MAX:
        raise ValueError(
            "surface_fallback_radius must be in "
            f"[0, {SURFACE_FALLBACK_RADIUS_MAX}]")
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
    if native_grid_identity(grid):
        # The identity route: the target is the native grid.  Every mass
        # point copies its own source cell, the u and v faces between
        # cells read their two neighbours and the outermost faces, half
        # a cell past the grid's edge, take the edge cell (zero gradient;
        # they are the specified boundary row, which the lateral boundary
        # tables overwrite from the first step).  So the whole grid is
        # the window and no halo past it is demanded.  The soil donor
        # search keeps its radius, clipped at the edge as always.
        return HrrrSourceWindow(
            i_start=0, i_end=HRRR_SOURCE_NX - 1,
            j_start=0, j_end=HRRR_SOURCE_NY - 1,
            parabolic_lower_halo_cells=0, parabolic_upper_halo_cells=0,
            surface_fallback_radius_cells=radius,
            target_source_i_range=(0.0, float(HRRR_SOURCE_NX - 1)),
            target_source_j_range=(0.0, float(HRRR_SOURCE_NY - 1)),
            route="identity")
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
    # What HRRR must hold: every target's bilinear cell.  The parabolic
    # halo past HRRR's own edge is not demanded: there WPS metgrid's
    # sixteen-point stencil leaves the source and falls through to
    # four-point bilinear, which the projected operators do
    # (gpuwm.ingest.hrrr._parabolic_stencil_leaves_window).  Demanding
    # it refused a domain one row inside the HRRR grid
    # (configs/recipes/hrrr_v4_gsd41.toml) whose atmosphere HRRR covers.
    bilinear_i_min = min(int(np.floor(x).min()) for x, _ in coordinates)
    bilinear_i_max = max(int(np.floor(x).max()) + 1 for x, _ in coordinates)
    bilinear_j_min = min(int(np.floor(y).min()) for _, y in coordinates)
    bilinear_j_max = max(int(np.floor(y).max()) + 1 for _, y in coordinates)
    if (bilinear_i_min < 0 or bilinear_j_min < 0
            or bilinear_i_max >= HRRR_SOURCE_NX
            or bilinear_j_max >= HRRR_SOURCE_NY):
        # Same breakage, same class as the mapped route's window refusal:
        # a source grid that does not reach the domain.  A door that owns
        # one owns both, so the certified native route and a table-added
        # model answer a distant domain with the same two sentences.
        raise SourceCoverageRefusal(
            "target domain plus required interpolation halo leaves HRRR "
            "coverage: required zero-based inclusive window "
            f"i={bilinear_i_min}..{bilinear_i_max}, "
            f"j={bilinear_j_min}..{bilinear_j_max}; native limits "
            f"are i=0..{HRRR_SOURCE_NX - 1}, j=0..{HRRR_SOURCE_NY - 1}")
    # The donor search's box stops at HRRR's own edge.  Demanding source
    # cells past it refused domains whose atmosphere HRRR covers, only
    # because the search box around a coastal cell reached past the edge
    # where there is nothing to search.
    i_start = max(0, min(parabolic_i_min, fallback_i_min))
    i_end = min(HRRR_SOURCE_NX - 1, max(parabolic_i_max, fallback_i_max))
    j_start = max(0, min(parabolic_j_min, fallback_j_min))
    j_end = min(HRRR_SOURCE_NY - 1, max(parabolic_j_max, fallback_j_max))
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
    "SURFACE_FALLBACK_RADIUS_MAX",
    "HrrrSourceWindow",
    "HrrrTargetDomain",
    "TARGET_DOMAIN_SCHEMA",
    "hrrr_coverage_envelope",
    "load_hrrr_target_domain",
    "required_hrrr_source_window",
]
