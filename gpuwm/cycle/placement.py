"""Where a cycle child goes, asked geographically and answered in indices.

**A placement request is GEOGRAPHIC.  A parent index is a resolution
result, never a request.**  That one rule is what makes arbitrary spawn a
mechanism rather than a transcription of one domain's geometry: the same
request resolves against a parent of any projection, any resolution and
any origin, and it keeps meaning the same thing when the parent moves
under it between cycles.  A request expressed in parent indices silently
means a different place on every parent, and nothing reports it.

Resolution goes through :func:`gpuwm.downscale._nearest_parent_index` --
projection-agnostic, cos-lat weighted, and already proven on this line to
re-derive a reference run's real nest geometry from nothing but a point --
and then :func:`gpuwm.downscale._centered_placement` for the footprint.
The spine does not own a second copy of either.

**Refusal, never a silent clamp.**  A footprint that leaves fewer parent
rows than ``spec_bdy_width + blend_width`` on any side is REFUSED, with
the geographic request, the parent index it mapped to, the rows kept and
the rows required all named.  ``allow_clamp=True`` is available and is
always receipted, because a clamped nest keeps integrating while it has
stopped following its storm -- which looks exactly like a working nest.

No case names, no station names, no CONUS-shaped constants: every number
here is a parameter or is derived from the parent handed in (standing
owner rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gpuwm.cycle.contracts import PLACEMENT_SCHEMA, CycleRefusal

#: ``spec_bdy_width (5) + blend_width (5)``, the parent-row clearance
#: :func:`gpuwm.experiment.validate_spawn_placement` admits a spawn
#: against.  Kept as its two summands because the refusal has to say
#: which rule was violated, not just that a number was too small.
DEFAULT_SPEC_BDY_WIDTH = 5
DEFAULT_BLEND_WIDTH = 5
DEFAULT_KEEPOUT_PARENT_CELLS = DEFAULT_SPEC_BDY_WIDTH + DEFAULT_BLEND_WIDTH

#: Mean Earth radius, for the great-circle separation used to keep two
#: nests from being planted on the same storm.
EARTH_RADIUS_KM = 6371.0088

PLACEMENT_SOURCES = ("tracker", "schedule", "obs")


# ---------------------------------------------------------------------------
# the parent a request is resolved against
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParentGeometry:
    """Everything a geographic request needs from the parent, and no more.

    Deliberately not an ``ExperimentConfig`` or a live domain: the
    placement seam is imported by a receipt reader and by a planner in a
    process that will never open a GPU.
    """

    lat_field: Any
    lon_field: Any
    dx_m: float
    spec_bdy_width: int = DEFAULT_SPEC_BDY_WIDTH
    blend_width: int = DEFAULT_BLEND_WIDTH

    @property
    def ny(self) -> int:
        return int(np.shape(self.lat_field)[0])

    @property
    def nx(self) -> int:
        return int(np.shape(self.lat_field)[1])

    def latlon_at(self, j: int, i: int) -> tuple[float, float]:
        return (float(np.asarray(self.lat_field)[j, i]),
                float(np.asarray(self.lon_field)[j, i]))


def parent_geometry_from_fields(*, lat_field, lon_field, dx_m: float,
                                spec_bdy_width: int = DEFAULT_SPEC_BDY_WIDTH,
                                blend_width: int = DEFAULT_BLEND_WIDTH
                                ) -> ParentGeometry:
    """Build a :class:`ParentGeometry`, refusing mismatched fields by shape."""

    lat = np.asarray(lat_field, dtype=np.float64)
    lon = np.asarray(lon_field, dtype=np.float64)
    if lat.ndim != 2 or lat.shape != lon.shape:
        raise CycleRefusal(
            "parent geometry needs matching 2-D lat and lon fields",
            lat_shape=tuple(lat.shape), lon_shape=tuple(lon.shape))
    if not float(dx_m) > 0.0:
        raise CycleRefusal("parent spacing must be positive",
                           dx_m=float(dx_m))
    return ParentGeometry(lat_field=lat, lon_field=lon, dx_m=float(dx_m),
                          spec_bdy_width=int(spec_bdy_width),
                          blend_width=int(blend_width))


# ---------------------------------------------------------------------------
# the request and its resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PlacementRequest:
    """A child asked for by WHERE, not by which parent cell."""

    lat: float
    lon: float
    dx_m: float
    nx: int
    ny: int
    source: str          # "tracker" | "schedule" | "obs"
    strength: float      # the trigger value; used to rank when slots are scarce
    evidence: dict       # centroid, cells above threshold, max value, search box


@dataclass(frozen=True, slots=True)
class ResolvedPlacement:
    """What the parent's geometry made of a request."""

    grid_id: int
    i_parent_start: int
    j_parent_start: int
    parent_grid_ratio: int
    request: PlacementRequest
    state: str           # from CHILD_STATES
    clamped: bool
    refusal: str | None

    def to_payload(self) -> dict:
        """The receipt form: the geographic ask AND the resolved indices."""

        return {
            "schema": PLACEMENT_SCHEMA,
            "grid_id": int(self.grid_id),
            "state": str(self.state),
            "clamped": bool(self.clamped),
            "refusal": self.refusal,
            "i_parent_start": int(self.i_parent_start),
            "j_parent_start": int(self.j_parent_start),
            "parent_grid_ratio": int(self.parent_grid_ratio),
            "request": {
                "lat": float(self.request.lat),
                "lon": float(self.request.lon),
                "dx_m": float(self.request.dx_m),
                "nx": int(self.request.nx), "ny": int(self.request.ny),
                "source": str(self.request.source),
                "strength": float(self.request.strength),
                "evidence": dict(self.request.evidence),
            },
        }


def _format_lat(lat: float) -> str:
    return f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"


def _ratio_for(request: PlacementRequest, parent: ParentGeometry) -> int:
    """The refinement ratio a request implies, or a refusal naming it."""

    ratio = parent.dx_m / float(request.dx_m)
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1e-9:
        raise CycleRefusal(
            "child spacing does not divide the parent spacing exactly",
            parent_dx_m=parent.dx_m, child_dx_m=float(request.dx_m),
            ratio=float(ratio))
    if request.nx % rounded or request.ny % rounded:
        raise CycleRefusal(
            "child extent is not a whole number of parent cells",
            child_nx=int(request.nx), child_ny=int(request.ny),
            parent_grid_ratio=rounded,
            remainder_nx=int(request.nx) % rounded,
            remainder_ny=int(request.ny) % rounded)
    return rounded


def _centered_start(center: float, span: int) -> int:
    """1-based footprint start centred on a parent cell.

    Round half UP, not banker's, so a half-cell-ambiguous centre resolves
    deterministically toward the higher parent index -- the same rule
    :func:`gpuwm.downscale._centered_placement` states.
    """

    return int(math.floor(center + 1 - (span - 1) / 2.0 + 0.5))


def _confirm_with_primitive(parent: ParentGeometry, *, ratio: int,
                            request: PlacementRequest, i_start: int,
                            j_start: int, span_i: int, span_j: int) -> None:
    """Re-derive an accepted placement through the shared primitive."""

    from gpuwm.downscale import _centered_placement

    placement = _centered_placement(
        {"nx": parent.nx, "ny": parent.ny},
        j0=j_start - 1 + (span_j - 1) / 2.0,
        i0=i_start - 1 + (span_i - 1) / 2.0, ratio=ratio,
        child_nx=int(request.nx), child_ny=int(request.ny))
    if (int(placement.i_parent_start), int(placement.j_parent_start)) != (
            i_start, j_start):
        raise CycleRefusal(
            "the spine's centring disagrees with gpuwm.downscale's",
            spine_ij=(i_start, j_start),
            downscale_ij=(int(placement.i_parent_start),
                          int(placement.j_parent_start)),
            parent_grid_ratio=int(ratio))


def resolve_placement(request: PlacementRequest, *,
                      parent_geometry: ParentGeometry, grid_id: int,
                      keepout_parent_cells: int
                      = DEFAULT_KEEPOUT_PARENT_CELLS,
                      allow_clamp: bool = False) -> ResolvedPlacement:
    """Geographic request -> parent indices, or a named refusal.

    ``keepout_parent_cells`` defaults to ``spec_bdy_width + blend_width``,
    the same clearance :func:`gpuwm.experiment.validate_spawn_placement`
    admits a spawn against.  Matching it here means a placement this
    function accepts is one the experiment loader will also accept, so a
    plan does not die at materialization hours later.
    """

    from gpuwm.downscale import _nearest_parent_index

    parent = parent_geometry
    ratio = _ratio_for(request, parent)
    j0, i0 = _nearest_parent_index(np.asarray(parent.lat_field),
                                   np.asarray(parent.lon_field),
                                   float(request.lat), float(request.lon))
    span_i = int(request.nx) // ratio
    span_j = int(request.ny) // ratio
    # The centring arithmetic is _centered_placement's, transcribed rather
    # than called, and only here.  That function builds an
    # OfflineChildPlacement, whose __post_init__ runs the +-2 SINT stencil
    # admission and raises a bare ValueError for a footprint that hangs
    # off the parent -- which is precisely the case the spine has to
    # answer with its own named refusal.  The primitive is still the
    # authority: an ACCEPTED placement is handed back to it below, so a
    # divergence between these two lines and that function is a failure,
    # not a drift.
    i_start = _centered_start(i0, span_i)
    j_start = _centered_start(j0, span_j)
    need = int(keepout_parent_cells)

    # Clearance on all four sides, in parent rows, in 1-based namelist
    # semantics -- the same arithmetic validate_spawn_placement does.
    sides = (
        ("west edge", i_start - 1, "i", -1),
        ("east edge", parent.nx - (i_start + span_i - 1), "i", +1),
        ("south edge", j_start - 1, "j", -1),
        ("north edge", parent.ny - (j_start + span_j - 1), "j", +1),
    )
    violations = [(name, clear, axis, sign)
                  for name, clear, axis, sign in sides if clear < need]

    if violations and not allow_clamp:
        name, clear, _axis, _sign = min(violations, key=lambda row: row[1])
        refusal = (
            f"requested child at {_format_lat(request.lat)}/"
            f"{request.lon:.2f}W maps to parent index (j={j_start}, "
            f"i={i_start}); the {name} keeps {clear} parent rows of the "
            f"required {need} (spec_bdy_width {parent.spec_bdy_width} + "
            f"blend_width {parent.blend_width}); placement REFUSED")
        return ResolvedPlacement(
            grid_id=int(grid_id), i_parent_start=i_start,
            j_parent_start=j_start, parent_grid_ratio=ratio,
            request=request, state="REFUSED", clamped=False,
            refusal=refusal)

    clamped = False
    if violations:
        # Clamping is the caller's explicit choice and it is FLAGGED, not
        # absorbed: a clamped nest integrates happily while its storm
        # walks out of it, and nothing but this flag says so.
        i_start = min(max(i_start, need + 1), parent.nx - span_i + 1 - need)
        j_start = min(max(j_start, need + 1), parent.ny - span_j + 1 - need)
        clamped = True
        if i_start < need + 1 or j_start < need + 1:
            raise CycleRefusal(
                "parent is too small to hold this child even clamped",
                parent_nx=parent.nx, parent_ny=parent.ny,
                child_span_i=span_i, child_span_j=span_j,
                keepout_parent_cells=need)

    # Hand the accepted footprint back to the primitive.  It re-derives
    # the same start from the equivalent centre and runs WRF's own +-2
    # SINT stencil admission (nest_interp.register_nest), so a placement
    # this function returns is one the experiment loader and the nest
    # registration will both accept -- the plan does not die at
    # materialization hours later.
    _confirm_with_primitive(parent, ratio=ratio, request=request,
                            i_start=i_start, j_start=j_start,
                            span_i=span_i, span_j=span_j)

    return ResolvedPlacement(
        grid_id=int(grid_id), i_parent_start=i_start, j_parent_start=j_start,
        parent_grid_ratio=ratio, request=request, state="PLANNED",
        clamped=clamped, refusal=None)


# ---------------------------------------------------------------------------
# separation, so two nests are not planted on one storm
# ---------------------------------------------------------------------------
def separation_km(lat_a: float, lon_a: float,
                  lat_b: float, lon_b: float) -> float:
    """Great-circle distance in km.  Projection-agnostic, like the rest."""

    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi = phi_b - phi_a
    d_lam = math.radians(lon_b - lon_a)
    h = (math.sin(d_phi / 2.0) ** 2
         + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lam / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def rank_and_assign(requests: Sequence[PlacementRequest], *,
                    free_slots: Sequence[int], min_separation_km: float,
                    parent_geometry: ParentGeometry,
                    keepout_parent_cells: int = DEFAULT_KEEPOUT_PARENT_CELLS,
                    allow_clamp: bool = False) -> list[ResolvedPlacement]:
    """Rank by strength, thin by separation, hand out slots, refuse the rest.

    **A request never vanishes.**  Every input appears in the output --
    resolved, or REFUSED with the reason named.  A planner that silently
    dropped the fourth storm because there were three slots would produce
    a receipt indistinguishable from one where the fourth storm was never
    detected, and those two mean opposite things about the run.
    """

    ordered = sorted(requests, key=lambda item: -float(item.strength))
    slots = list(free_slots)
    out: list[ResolvedPlacement] = []
    taken: list[PlacementRequest] = []

    for item in ordered:
        crowd = next((other for other in taken
                      if separation_km(item.lat, item.lon,
                                       other.lat, other.lon)
                      < float(min_separation_km)), None)
        if crowd is not None:
            out.append(ResolvedPlacement(
                grid_id=-1, i_parent_start=-1, j_parent_start=-1,
                parent_grid_ratio=-1, request=item, state="REFUSED",
                clamped=False,
                refusal=(
                    f"within min_separation_km {float(min_separation_km):g} "
                    f"of an already-assigned child at "
                    f"{_format_lat(crowd.lat)}/{crowd.lon:.2f}W "
                    f"(separation "
                    f"{separation_km(item.lat, item.lon, crowd.lat, crowd.lon):.1f} km)")))
            continue
        if not slots:
            out.append(ResolvedPlacement(
                grid_id=-1, i_parent_start=-1, j_parent_start=-1,
                parent_grid_ratio=-1, request=item, state="REFUSED",
                clamped=False, refusal="slot pool exhausted"))
            continue
        resolved = resolve_placement(
            item, parent_geometry=parent_geometry, grid_id=slots[0],
            keepout_parent_cells=keepout_parent_cells,
            allow_clamp=allow_clamp)
        if resolved.state == "REFUSED":
            # A refused placement does not consume the slot: the next
            # storm down the ranking may well fit.
            out.append(resolved)
            continue
        slots.pop(0)
        taken.append(item)
        out.append(resolved)
    return out


# ---------------------------------------------------------------------------
# the three providers -- one signature
# ---------------------------------------------------------------------------
def _peaks(plane, threshold: float, *, parent: ParentGeometry,
           min_separation_km: float, max_children: int,
           window_cells: int) -> list[dict]:
    """Argmax, then a footprint window, then argmax again.

    The window is what stops two storms in one search box from producing
    one nest planted between them, on neither -- the failure mode
    :func:`gpuwm.core.storm_tracking.weighted_centroid` is called inside a
    box to avoid.  Here the boxes are discovered rather than declared: the
    strongest remaining cell defines the next box.
    """

    from gpuwm.core.storm_tracking import weighted_centroid

    work = np.array(plane, dtype=np.float64, copy=True)
    with np.errstate(invalid="ignore"):
        work[~np.isfinite(work)] = -np.inf
    found: list[dict] = []
    half = max(1, int(window_cells) // 2)

    while len(found) < int(max_children):
        peak = int(np.argmax(work))
        pj, pi = np.unravel_index(peak, work.shape)
        if not float(work[pj, pi]) >= float(threshold):
            break
        box = (slice(max(0, int(pj) - half),
                     min(parent.ny, int(pj) + half + 1)),
               slice(max(0, int(pi) - half),
                     min(parent.nx, int(pi) + half + 1)))
        centroid = weighted_centroid(work, float(threshold), box)
        # Consume the window whether or not it produced a centroid, so a
        # degenerate box cannot spin this loop.
        work[box] = -np.inf
        if centroid is None:
            continue
        cj = int(round(centroid["cj"]))
        ci = int(round(centroid["ci"]))
        lat, lon = parent.latlon_at(min(cj, parent.ny - 1),
                                    min(ci, parent.nx - 1))
        if any(separation_km(lat, lon, other["lat"], other["lon"])
               < float(min_separation_km) for other in found):
            continue
        found.append({"lat": lat, "lon": lon,
                      "strength": float(centroid["max_value"]),
                      "cells": int(centroid["cells"]),
                      "cj": float(centroid["cj"]),
                      "ci": float(centroid["ci"]),
                      "box": [int(box[0].start), int(box[0].stop),
                              int(box[1].start), int(box[1].stop)]})
    return found


@dataclass(frozen=True, slots=True)
class TrackerPlacementProvider:
    """Children on the parent's own storms: UH or composite reflectivity."""

    field: str
    threshold: float
    min_separation_km: float
    max_children: int
    child_nx: int
    child_ny: int
    child_dx_m: float

    def __call__(self, *, cycle_index: int, valid_time, parent_geometry,
                 signal) -> list[PlacementRequest]:
        if signal is None:
            return []
        plane = np.asarray(signal, dtype=np.float64)
        if plane.shape != (parent_geometry.ny, parent_geometry.nx):
            raise CycleRefusal(
                "tracker signal is not on the parent's mass grid",
                field=self.field, signal_shape=tuple(plane.shape),
                parent_shape=(parent_geometry.ny, parent_geometry.nx))
        window = max(1, int(self.child_nx
                            * self.child_dx_m / parent_geometry.dx_m))
        return [PlacementRequest(
            lat=peak["lat"], lon=peak["lon"], dx_m=float(self.child_dx_m),
            nx=int(self.child_nx), ny=int(self.child_ny), source="tracker",
            strength=peak["strength"],
            evidence={"field": self.field, "threshold": float(self.threshold),
                      "cells": peak["cells"], "max_value": peak["strength"],
                      "centroid_ji": [peak["cj"], peak["ci"]],
                      "search_box": peak["box"],
                      "cycle_index": int(cycle_index)})
            for peak in _peaks(
                plane, float(self.threshold), parent=parent_geometry,
                min_separation_km=float(self.min_separation_km),
                max_children=int(self.max_children), window_cells=window)]


@dataclass(frozen=True, slots=True)
class SchedulePlacementProvider:
    """Children from the run plan: a static list of times and points."""

    entries: Sequence[tuple]

    def __call__(self, *, cycle_index: int, valid_time, parent_geometry,
                 signal) -> list[PlacementRequest]:
        out = []
        for entry in self.entries:
            when, lat, lon, dx_m, nx, ny = entry
            if when != valid_time:
                continue
            out.append(PlacementRequest(
                lat=float(lat), lon=float(lon), dx_m=float(dx_m),
                nx=int(nx), ny=int(ny), source="schedule",
                # A scheduled child is not competing on storm strength; it
                # was asked for.  Rank it above anything a trigger found so
                # scarcity never silently drops a person's own request.
                strength=float("inf"),
                evidence={"valid_time": when,
                          "cycle_index": int(cycle_index)}))
        return out


def _document_plane(document, field: str, path: str):
    """The requested field's placement plane, from a radar-grid document.

    ``read_radar_grid`` returns the observation arrays under a
    ``"variables"`` mapping and keeps the top level for schema, status,
    provenance and grid identity.  Looking only at the top level refuses
    every real file while every injected-reader test passes, so the
    lookup checks ``variables`` first and falls back to the top level for
    a caller that hands in a bare ``{field: plane}`` mapping.
    """

    variables = document.get("variables") if hasattr(document, "get") else None
    if variables is not None and field in variables:
        plane = np.asarray(variables[field])
    elif field in document:
        plane = np.asarray(document[field])
    else:
        available = sorted(str(key) for key in (variables or document))
        raise CycleRefusal(
            "radar grid file does not carry the requested field",
            field=field, path=path, available=available)
    if plane.ndim == 4:
        # A per-radar variable (radar, nz, ny, nx) reduces over radars
        # first: placement asks where the storm is, not which radar saw it.
        with np.errstate(invalid="ignore"):
            plane = np.nanmax(plane, axis=0)
    if plane.ndim == 3:
        # A 3-D reflectivity volume collapses to the column maximum,
        # which is the plane a placement decision wants.
        with np.errstate(invalid="ignore"):
            plane = np.nanmax(plane, axis=0)
    return plane


@dataclass(frozen=True, slots=True)
class ObsPlacementProvider:
    """Children on the storm the RADAR sees, model field or no model field.

    This is the capability the existing spawn triggers cannot have: every
    one of them reads a model plane, so a storm the model has not yet
    produced can never trigger a nest, which is exactly the storm a
    forecaster most wants a nest on.  The observation file is already a
    NetCDF on a known target grid, so the whole cost is a read.
    """

    radar_grid_path: Any
    field: str
    threshold_dbz: float
    min_separation_km: float
    max_children: int
    child_nx: int
    child_ny: int
    child_dx_m: float
    #: Injected for tests and for a caller that already holds the plane.
    #: Defaults to the canonical reader, so the shipped path reads a real
    #: radar-grid file with its structural contract checked.
    reader: Callable[[Any, str], Any] | None = None

    def _plane(self):
        if self.reader is not None:
            return self.reader(self.radar_grid_path, self.field)
        from gpuwm.obs.radar_grid import read_radar_grid

        document = read_radar_grid(self.radar_grid_path)
        return _document_plane(document, self.field,
                               str(self.radar_grid_path))

    def __call__(self, *, cycle_index: int, valid_time, parent_geometry,
                 signal) -> list[PlacementRequest]:
        plane = np.asarray(self._plane(), dtype=np.float64)
        if plane.shape != (parent_geometry.ny, parent_geometry.nx):
            raise CycleRefusal(
                "observation plane is not on the parent's mass grid",
                field=self.field, obs_shape=tuple(plane.shape),
                parent_shape=(parent_geometry.ny, parent_geometry.nx),
                path=str(self.radar_grid_path))
        window = max(1, int(self.child_nx
                            * self.child_dx_m / parent_geometry.dx_m))
        return [PlacementRequest(
            lat=peak["lat"], lon=peak["lon"], dx_m=float(self.child_dx_m),
            nx=int(self.child_nx), ny=int(self.child_ny), source="obs",
            strength=peak["strength"],
            evidence={"field": self.field,
                      "threshold_dbz": float(self.threshold_dbz),
                      "cells": peak["cells"], "max_value": peak["strength"],
                      "centroid_ji": [peak["cj"], peak["ci"]],
                      "search_box": peak["box"],
                      "radar_grid_path": str(self.radar_grid_path),
                      "cycle_index": int(cycle_index)})
            for peak in _peaks(
                plane, float(self.threshold_dbz), parent=parent_geometry,
                min_separation_km=float(self.min_separation_km),
                max_children=int(self.max_children), window_cells=window)]


def build_placement_provider(**kwargs):
    """The factory ``gpuwm.cycle.cli`` imports by name.

    Lives in gpuwm.cycle.engine, which owns the join between the provider
    classes here and the slot pool in gpuwm.cycle.children.  Re-exported
    here because the CLI looks for it on this module and refuses by name
    if it is absent.
    """

    from gpuwm.cycle.engine import build_placement_provider as _build

    return _build(**kwargs)
