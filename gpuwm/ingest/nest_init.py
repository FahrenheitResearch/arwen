"""WRF-order initialization for real and parent-only child domains.

The real-data path is the scope-trimmed ``med_nest_initial`` transaction
ratified for Phase 5 (WRF v4.6.1 ``share/mediation_integrate.F:509-952``):

1. ERA5-direct initialization on the child's own grid and unblended fine
   terrain, including Noah soil/skin/landmask preprocessing;
2. SINT capture of only the parent ``ht``/``mub``/``phb`` blend operands;
3. ``blend_terrain`` on all three operands;
4. ``adjust_tempqv`` for the base-column-mass change; and
5. a ``start_domain`` analogue that reconstitutes the analytic base fields,
   reruns the equation-of-state diagnostics, and applies the nest-only
   ``press_adj`` column-mass correction.

The Noah result is deliberately produced before SINT/blending and is never
adjusted afterward.  That is WRF fidelity, not an omission: real.exe builds
the land state on fine terrain before wrf.exe blends the nest, and WRF has no
post-blend soil adjustment.

All SINT outputs and adjustment work arrays are init-only plain temporaries.
They are intentionally absent from the step-scratch registry and become
unreachable when initialization returns.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from gpuwm.case_data import (PerDomainSourceOrography, SourceOrography,
                             SourceOrographyDeclaration,
                             resolve_source_orography)
from gpuwm.core import constants as c
from gpuwm.core.diagnostics import update_diagnostics
from gpuwm.core.grid import BaseState, VerticalCoord, make_vertical_coord
from gpuwm.core.nest_interp import (adjust_tempqv, blend_terrain,
                                    register_nest, sint)
from gpuwm.core.microphysics_transition import (
    launch_microphysics_edge_parent_field,
    resolve_microphysics_transition,
    transition_handles_field,
    transition_parent_field_shape,
)
from gpuwm.core.state import DomainState
from gpuwm.experiment import DomainConfig, VerticalConfig
from gpuwm.ingest.horiz import (HorizontalSnapshot,
                                interpolate_era5_to_lambert,
                                interpolate_lake_skin_temperature)
from gpuwm.ingest.hrrr import (HrrrNativeSnapshot,
                               interpolate_hrrr_to_lambert)
from gpuwm.ingest.real import RealInitResult, initialize_real
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.ingest.soil import (NoahSoilState, reconciler_soil_temperature,
                               reconciler_sst, soil_source_orography)
from gpuwm.ingest.water_temperature import WaterTemperatureStatics
from gpuwm.static.build import (build_static_for_domain,
                                geog_selection_from_catalog)
from gpuwm.static.lambert import LambertGrid

#: The nested-child route's name in every water-temperature refusal and
#: receipt.  A child assembles under its parent's policy but with its
#: OWN grid, statics and land-use table.
_WATER_ROUTE = "the nested-child preparation route"

if TYPE_CHECKING:
    # Native setup consumes only ``cfg``, ``grid``, and ``state`` from an
    # already initialized parent.  Importing the forecast executor merely for
    # this annotation made every CPU-only RW-WPS route load
    # ``gpuwm.core.model``.  Keep the richer type for static analysis while
    # preserving a genuinely model-independent runtime import boundary.
    from gpuwm.core.model import DomainNode
else:
    DomainNode = Any


@dataclass(frozen=True)
class ChildInitResult:
    """A constructed child plus setup products needed by Task 14.

    ``static_fields`` always retain unblended ``HGT_M``.  ``soil`` is the
    pre-blend Noah state.  They are ``None`` only for the parent-only
    idealized branch, which has no external static or surface input.
    """

    state: DomainState
    grid: LambertGrid
    coord: VerticalCoord
    real: RealInitResult | None
    static_fields: Mapping[str, np.ndarray] | None
    horizontal: HorizontalSnapshot | None
    soil: NoahSoilState | None
    preprocess_receipt: Mapping[str, object] | None = None
    input_preparation_seconds: float | None = None
    domain: DomainConfig | None = None
    #: Per-field account of the parent-SINT positive-definite fix-up
    #: (:func:`clamp_parent_sint_undershoot`).  Empty when the
    #: interpolation landed clean, which is the ordinary case; ``None``
    #: for the branches that run no SINT at all.
    positive_definite_clamp: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PreparedChildInput:
    """Parent-independent static and meteorological child inputs.

    This object deliberately stops before ``initialize_real`` and every SINT /
    terrain-balance operation.  Building WPS_GEOG fields, mapping the source
    snapshot, and resolving lake/orography input are CPU/read-only work and may
    run concurrently across domains; model-state construction remains an
    explicit parent-before-child dependency phase.
    """

    domain: DomainConfig
    grid: LambertGrid
    static_fields: Mapping[str, np.ndarray]
    horizontal: HorizontalSnapshot
    declared_orography: np.ndarray | None
    lake_mask: np.ndarray
    lake_skin_temperature: np.ndarray
    preprocess_backend: object
    preprocess_receipt: Mapping[str, object]
    preparation_seconds: float
    #: Land-use attributes (ISWATER/ISLAKE/ISICE) resolved during
    #: preparation, where the static catalog is in scope.  Finalization
    #: needs them to reconcile ISLTYP before building the soil column, so
    #: the state is never built with a category the model does not use.
    landuse_attrs: Mapping[str, object] | None = None
    #: The water-temperature policy this child assembled under, carried
    #: for the same reason: finalization calls the soil router, the
    #: router refuses an unassembled raw SST/SKINTEMP pair, and the
    #: catalog that resolved the policy is no longer in scope there.
    water_temperature_policy: str | None = None
    #: The forcing mesh measured against this child's own grid
    #: (:class:`gpuwm.ingest.soil_downscale.SoilMeshPlan`), carried for the
    #: same reason as the policy above: finalization builds the child's
    #: soil column, the sub-source-cell reconstitution needs the SOURCE
    #: spacing, and the source snapshot is out of scope by then.  Without
    #: it a nest would inherit exactly the 0.25 degree soil quilt its
    #: parent was fixed for -- the child re-ingests the SAME source
    #: through the SAME interpolation, so the defect is not the parent's
    #: to hand down or withhold.  ``None`` only for a source with no
    #: regular lat/lon mesh to measure.
    soil_mesh: object | None = None

    def __post_init__(self) -> None:
        if (not np.isfinite(self.preparation_seconds)
                or self.preparation_seconds < 0.0):
            raise ValueError("preparation_seconds must be finite and non-negative")
        object.__setattr__(
            self, "preprocess_receipt",
            MappingProxyType(dict(self.preprocess_receipt)))


@dataclass(frozen=True)
class ParentInitView:
    """Minimal already-initialized parent needed by the child barrier."""

    cfg: DomainConfig
    grid: LambertGrid
    state: DomainState


@dataclass(frozen=True)
class NestedInputCatalog:
    """Source snapshots joined to an independently verified static catalog."""

    snapshots: tuple[object, ...]
    static_catalog: object
    inventory: tuple[str, ...] = ()
    files: tuple[object, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)
    #: The resolved ``[ingest] soil_texture_downscale`` declaration, carried
    #: for the same reason :class:`gpuwm.ingest.preflight.InputCatalog`
    #: carries it: a child sees the catalog and not the case data.  A nest
    #: that kept the forcing mesh in its soil while its parent did not --
    #: or DROPPED it while its parent kept it, which is the
    #: WRF-comparison direction -- would seam at the nest boundary in
    #: every field soil moisture drives.  Default True: silence means ON.
    soil_texture_downscale: bool = True

    def __post_init__(self) -> None:
        snapshots = tuple(self.snapshots)
        inventory = tuple(self.inventory)
        files = tuple(self.files)
        units = dict(self.units)
        provenance = dict(self.provenance)
        if not snapshots:
            raise ValueError("nested input catalog requires source snapshots")
        if self.static_catalog is None:
            raise ValueError("nested input catalog requires a static catalog")
        valid_times = tuple(
            getattr(snapshot, "valid_time", None) for snapshot in snapshots)
        if not all(isinstance(value, datetime) for value in valid_times):
            raise TypeError("every nested source snapshot needs a datetime valid_time")
        try:
            increasing = all(
                later > earlier
                for earlier, later in zip(valid_times, valid_times[1:]))
        except TypeError as exc:
            raise ValueError(
                "nested source valid times use incompatible timezone forms") from exc
        if len(valid_times) > 1 and not increasing:
            raise ValueError(
                "nested source snapshots must have unique increasing valid times")
        source_types = {type(snapshot) for snapshot in snapshots}
        if len(source_types) != 1:
            raise TypeError("nested source snapshots must use one adapter type")
        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "inventory", inventory)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "units", MappingProxyType(units))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    @classmethod
    def from_source_catalog(cls, source_catalog, static_catalog):
        """Preserve trajectory-relevant source metadata while splitting GEOG."""

        return cls(
            snapshots=tuple(getattr(source_catalog, "snapshots", ())),
            static_catalog=static_catalog,
            inventory=tuple(getattr(source_catalog, "inventory", ())),
            files=tuple(getattr(source_catalog, "files", ())),
            units=dict(getattr(source_catalog, "units", {})),
            provenance=dict(getattr(source_catalog, "provenance", {})),
        )

    @property
    def valid_times(self) -> tuple[datetime, ...]:
        return tuple(snapshot.valid_time for snapshot in self.snapshots)


class PendingChildInputs:
    """Bounded child-input work with deterministic parent-order collection."""

    def __init__(self, domains: Sequence[DomainConfig], grids, catalog,
                 source_orography, workers: int, *,
                 preprocess_backend: str | object = "cpu",
                 cpu_bridge: Path | str | None = None):
        self.domains = tuple(domains)
        if (isinstance(workers, bool) or not isinstance(workers, int)
                or workers < 1 or workers > 32):
            raise ValueError(
                f"child input workers must be an integer in [1, 32], got "
                f"{workers!r}")
        domain_ids = tuple(domain.grid_id for domain in self.domains)
        if len(set(domain_ids)) != len(domain_ids):
            raise ValueError("child input domains contain duplicate grid ids")
        missing_grids = tuple(
            grid_id for grid_id in domain_ids if grid_id not in grids)
        if missing_grids:
            labels = ", ".join(f"d{grid_id:02d}" for grid_id in missing_grids)
            raise ValueError(f"child input grids are missing {labels}")
        self.worker_budget = workers
        self.workers = min(workers, len(self.domains))
        if isinstance(preprocess_backend, str):
            normalized_backend = preprocess_backend.strip().lower()
        else:
            normalized_backend = None
        if (len(self.domains) > 1 and self.workers > 1
                and normalized_backend != "cpu"):
            raise ValueError(
                "parallel child input preparation requires the explicit CPU "
                "backend; CUDA/custom/auto backends must use workers=1")
        if normalized_backend == "cpu" and self.workers:
            quotient, remainder = divmod(workers, self.workers)
            per_slot = tuple(
                quotient + (1 if index < remainder else 0)
                for index in range(self.workers))
            self.preprocess_workers_by_domain = MappingProxyType({
                domain.grid_id: per_slot[index % self.workers]
                for index, domain in enumerate(self.domains)})
        else:
            self.preprocess_workers_by_domain = MappingProxyType({
                domain.grid_id: None for domain in self.domains})
        self.allocated_workers = sum(
            self.preprocess_workers_by_domain[domain.grid_id] or 1
            for domain in self.domains[:self.workers])
        self._executor = (ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="gpuwm-child-input")
            if self.workers else None)
        self._futures: dict[Future, int] = {}
        self._futures_by_id: dict[int, Future] = {}
        self._grids = grids
        self._catalog = catalog
        self._source_orography = source_orography
        self._preprocess_backend = preprocess_backend
        self._cpu_bridge = cpu_bridge
        self._next_submit_index = 0
        self._result: tuple[PreparedChildInput, ...] | None = None
        self._closed = False
        self._stream_started = False
        for _ in range(self.workers):
            self._submit_next()

    def _submit_next(self) -> None:
        if self._next_submit_index >= len(self.domains):
            return
        if self._executor is None:
            raise AssertionError("non-empty child hierarchy has no executor")
        domain = self.domains[self._next_submit_index]
        self._next_submit_index += 1
        future = self._executor.submit(
            _prepare_child_input_on_grid, domain,
            self._grids[domain.grid_id], self._catalog,
            self._source_orography, self._preprocess_backend,
            self.preprocess_workers_by_domain[domain.grid_id],
            self._cpu_bridge)
        self._futures[future] = domain.grid_id
        self._futures_by_id[domain.grid_id] = future

    def _cancel_pending(self) -> None:
        for future in self._futures:
            future.cancel()

    def iter_parent_order(self) -> Iterable[PreparedChildInput]:
        """Yield prepared domains in dependency order from a bounded window.

        No replacement job is submitted until the next parent-order result is
        available.  At most ``workers`` futures can therefore own prepared
        arrays, while finalization of one child overlaps preparation of the
        next independent child.
        """

        if self._result is not None:
            yield from self._result
            return
        if self._stream_started:
            raise RuntimeError("child input results are already being consumed")
        self._stream_started = True
        try:
            for domain in self.domains:
                grid_id = domain.grid_id
                future = self._futures_by_id[grid_id]
                try:
                    prepared = future.result()
                except Exception as exc:
                    self._cancel_pending()
                    raise RuntimeError(
                        f"d{grid_id:02d} independent input preparation "
                        "failed") from exc
                del self._futures_by_id[grid_id]
                del self._futures[future]
                self._submit_next()
                yield prepared
        finally:
            self.close()

    def result(self) -> tuple[PreparedChildInput, ...]:
        if self._result is not None:
            return self._result
        self._result = tuple(self.iter_parent_order())
        return self._result

    def close(self) -> None:
        if not self._closed:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _host(value, *, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=dtype)


def _child_grid_from_parent(
        child_dc: DomainConfig, parent_grid_id: int,
        parent_grid: LambertGrid) -> LambertGrid:
    if child_dc.parent_id == 0:
        raise ValueError("initialize_child requires a non-root DomainConfig")
    if child_dc.parent_id != parent_grid_id:
        raise ValueError(
            f"child grid_id={child_dc.grid_id} names parent_id="
            f"{child_dc.parent_id}, not supplied parent grid_id="
            f"{parent_grid_id}")
    cfg = child_dc.run
    grid = parent_grid.nest(
        child_dc.i_parent_start, child_dc.j_parent_start,
        child_dc.parent_grid_ratio, cfg.nx + 1, cfg.ny + 1,
        resolved_dx=cfg.dx, resolved_dy=cfg.dy)
    expected = (cfg.nx, cfg.ny, cfg.dx, cfg.dy)
    actual = (grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy)
    if actual != expected:
        raise ValueError(
            f"child grid layout resolves to {actual}, RunConfig declares "
            f"{expected}")
    return grid


def _child_grid(child_dc: DomainConfig, parent_node: DomainNode) -> LambertGrid:
    return _child_grid_from_parent(
        child_dc, parent_node.cfg.grid_id, parent_node.grid)


@lru_cache(maxsize=None)
def _shared_vertical_coord(vertical: VerticalConfig, nz: int) -> VerticalCoord:
    """Materialize the experiment's one shared vertical coordinate once."""
    if not vertical.eta_levels:
        raise ValueError(
            "real child initialization requires explicit shared eta_levels")
    eta = np.asarray(vertical.eta_levels, dtype=np.float64)
    if eta.shape != (int(nz) + 1,):
        raise ValueError(
            f"shared eta_levels has shape {eta.shape}, expected "
            f"({int(nz) + 1},) for every domain")
    return make_vertical_coord(
        int(nz), hybrid_opt=vertical.hybrid_opt, etac=vertical.etac,
        eta_levels=eta)


def _initial_snapshot(catalog, valid_time: datetime | None = None):
    snapshots = tuple(getattr(catalog, "snapshots", ()))
    valid_times = tuple(getattr(catalog, "valid_times", ()))
    if not snapshots or not valid_times:
        raise ValueError("input catalog has no decoded initial source snapshot")
    requested = valid_times[0] if valid_time is None else valid_time
    matches = tuple(snapshot for snapshot in snapshots
                    if snapshot.valid_time == requested)
    if len(matches) != 1:
        raise ValueError(
            "input catalog has no unique snapshot at domain start_time "
            f"{requested!s}; available valid times are {valid_times}")
    return matches[0]


def _static_catalog(catalog):
    """Select an optional source-independent WPS_GEOG catalog."""

    selected = getattr(catalog, "static_catalog", catalog)
    if selected is None:
        raise ValueError("input catalog does not bind a static WPS_GEOG catalog")
    return selected


def _read_source_orography(artifact: SourceOrography) -> np.ndarray:
    """Read one already domain-resolved declared artifact."""
    from gpuwm import netcdf_bridge

    with netcdf_bridge.open_dataset(artifact.path) as dataset:
        if artifact.variable not in dataset.variables:
            raise ValueError(
                f"source-orography variable {artifact.variable!r} is absent "
                f"from {artifact.path}")
        return np.asarray(
            dataset.variables[artifact.variable][0], dtype=np.float64)


def _catalog_source_declaration(catalog) -> SourceOrographyDeclaration | None:
    """Reconstitute domain-tagged declarations from catalog provenance."""
    entries = tuple(item for item in getattr(catalog, "files", ())
                    if getattr(item, "role", None) == "source_orography")
    if not entries:
        return None

    parsed: list[tuple[int | None, SourceOrography]] = []
    for entry in entries:
        provenance = str(getattr(
            entry, "provenance", getattr(entry, "detail", "")))
        fields: dict[str, str] = {}
        for token in provenance.split(";"):
            key, separator, value = token.partition("=")
            if not separator or not key or not value or key in fields:
                raise ValueError(
                    "catalog source_orography must record provenance "
                    "variable=<name>[;domain=dNN]")
            fields[key] = value
        if set(fields) not in ({"variable"}, {"variable", "domain"}):
            raise ValueError(
                "catalog source_orography must record provenance "
                "variable=<name>[;domain=dNN]")
        domain_id = None
        if "domain" in fields:
            label = fields["domain"]
            if (len(label) < 3 or label[0] != "d"
                    or not label[1:].isdigit() or int(label[1:]) < 1):
                raise ValueError(
                    "catalog source_orography domain provenance must have "
                    f"WRF form d01, d02, ..., got {label!r}")
            domain_id = int(label[1:])
        parsed.append((domain_id, SourceOrography(
            path=Path(entry.path), variable=fields["variable"])))

    tagged = tuple(item for item in parsed if item[0] is not None)
    if tagged:
        if len(tagged) != len(parsed):
            raise ValueError(
                "catalog source_orography entries must either all carry "
                "domain provenance or all use the legacy d01 declaration")
        domain_ids = tuple(domain_id for domain_id, _artifact in tagged)
        if len(set(domain_ids)) != len(domain_ids):
            raise ValueError(
                "catalog contains duplicate per-domain source_orography "
                "entries")
        return PerDomainSourceOrography(tuple(
            (int(domain_id), artifact)
            for domain_id, artifact in sorted(tagged)))

    if len(parsed) != 1:
        raise ValueError(
            "catalog must contain one legacy d01 source_orography or "
            "domain-tag every per-domain entry")
    return parsed[0][1]


def _declared_source_orography(
        catalog, domain_id: int,
        declaration: SourceOrographyDeclaration | None = None
        ) -> np.ndarray | None:
    """Read the artifact resolved for ``domain_id``, when declared."""
    if declaration is None:
        declaration = _catalog_source_declaration(catalog)
    artifact = resolve_source_orography(declaration, domain_id)
    return (_read_source_orography(artifact)
            if artifact is not None else None)


def _set_map_fields(state: DomainState, grid: LambertGrid) -> None:
    f, e = grid.coriolis_m()
    sina, cosa = grid.rotation_m()
    state.set_map_coriolis(
        grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(), f, e,
        sina=sina, cosa=cosa)


def _mass_registration(child_dc: DomainConfig, parent_node: DomainNode):
    return register_nest(
        nri=child_dc.parent_grid_ratio,
        nrj=child_dc.parent_grid_ratio,
        i_parent_start=child_dc.i_parent_start,
        j_parent_start=child_dc.j_parent_start,
        child_nx=child_dc.run.nx, child_ny=child_dc.run.ny,
        parent_nx=parent_node.cfg.run.nx,
        parent_ny=parent_node.cfg.run.ny,
        stagger="", wrapper="interp")


def _registration(child_dc: DomainConfig, parent_node: DomainNode,
                  stagger: str):
    return register_nest(
        nri=child_dc.parent_grid_ratio,
        nrj=child_dc.parent_grid_ratio,
        i_parent_start=child_dc.i_parent_start,
        j_parent_start=child_dc.j_parent_start,
        child_nx=child_dc.run.nx, child_ny=child_dc.run.ny,
        parent_nx=parent_node.cfg.run.nx,
        parent_ny=parent_node.cfg.run.ny,
        stagger=stagger, wrapper="interp")


def _capture_parent_blend_fields(child_dc: DomainConfig,
                                 parent_node: DomainNode):
    """SINT only the three fields that survive real-input overwrite."""
    reg = _mass_registration(child_dc, parent_node)
    parent = parent_node.state
    if parent.phb.ndim != 3:
        raise ValueError(
            "real child initialization requires a terrain parent with "
            "three-dimensional phb")
    return (sint(parent.ht, reg), sint(parent.mub2d, reg),
            sint(parent.phb, reg))


def _base_from_blended(state: DomainState, cfg, coord: VerticalCoord,
                       p_top: float) -> BaseState:
    """Reconstitute WRF's real multi-domain base fields after blending.

    ``start_domain_em.F:682-698`` recomputes ``pb``, ``t_init`` and ``alb``
    from the (already blended) MUB while retaining blended PHB when
    ``rebalance=0``.  gpuwm's real-base constants and operation ordering are
    reused here.  The adjusted total theta is rebased by the caller after
    this object is loaded.
    """
    terrain = np.array(_host(state.ht), copy=True)
    mub = np.array(_host(state.mub2d), copy=True)
    phb = np.array(_host(state.phb), copy=True)
    pb = (coord.c3h[:, None, None] * mub[None]
          + coord.c4h[:, None, None] + float(p_top))
    if np.any(pb <= 0.0) or not np.all(np.diff(pb, axis=0) < 0.0):
        raise ValueError("blended hybrid base pressure is not monotonic")
    lapse = 50.0
    temperature = np.maximum(
        200.0, cfg.base_temp + lapse * np.log(pb / c.P0))
    thb = temperature * (c.P0 / pb) ** c.RCP
    alb = c.RD * thb * (pb / c.P0) ** c.RCP / pb
    return BaseState(mub=mub, p_top=float(p_top), pb=pb, alb=alb,
                     thb=thb, phb=phb, terrain_z=terrain)


def _as_like(value, template):
    """Move setup input to the array backend/dtype of ``template``."""
    if isinstance(template, np.ndarray):
        return np.ascontiguousarray(
            _host(value, dtype=template.dtype), dtype=template.dtype)
    import cupy as cp

    return cp.asarray(value, dtype=template.dtype)


def _apply_press_adj_mu(state: DomainState, ht_fine) -> None:
    """Apply WRF's real-nest ``press_adj`` correction after EOS diagnosis.

    ``med_nest_initial`` sets ``press_adj`` immediately before
    ``start_domain`` (``mediation_integrate.F:795-797``).  On exactly that
    path, ``start_em.F:878-892`` updates ``MU_2`` from the newly diagnosed
    bottom-level ``al/alt/alb`` and then copies ``MU_2`` to ``MU_1``.  WRF
    deliberately does not rerun the EOS after this update.
    """
    if tuple(np.shape(ht_fine)) != tuple(state.ht.shape):
        raise ValueError(
            f"fine terrain shape {np.shape(ht_fine)} differs from blended "
            f"terrain shape {state.ht.shape}")
    fine = _as_like(ht_fine, state.ht)
    state.mup[...] += (
        state.al[0] / (state.alt[0] * state.alb[0])
        * np.float32(c.G) * (state.ht - fine))
    state.mup0[...] = state.mup


def _adjust_and_rederive(state: DomainState, cfg, coord: VerticalCoord,
                         save_mub, ht_fine, *,
                         column_mass_correction: bool = True) -> BaseState:
    """Run adjust_tempqv, base/EOS re-derivation, then nest press_adj.

    ``column_mass_correction=False`` drops the first and last of those and
    keeps only the re-derivation, WHICH IS WHAT A MOVE GETS IN WRF.

    Read the two mediation layers side by side.  Nest INITIALIZATION
    (``share/mediation_integrate.F``, ``med_nest_initial``) blends the
    terrain triple and then, at :763, calls ``adjust_tempqv`` on
    ``t_2``/``p``/``QVAPOR``, and at :809 sets ``press_adj = .TRUE.`` so
    ``start_domain`` applies the MU correction too.  A nest MOVE
    (``share/mediation_nest_move.F``) blends the SAME terrain triple --
    and then calls neither: ``adjust_tempqv`` appears nowhere in that
    file, and ``press_adj`` is explicitly set ``.FALSE.`` for both the
    parent (:242) and the nest (:261) before ``start_domain``.

    The asymmetry is not an oversight.  At t = 0 the child's columns came
    from the parent and have never been consistent with the child's own
    terrain, so temperature and vapour have to be corrected for the
    base-column-mass change.  A moving nest's columns are its OWN, already
    consistent -- the blend perturbs ``mub`` only in the frame, and
    correcting theta/qv against that perturbation injects a
    thermodynamic anomaly into a field that was already right.

    MEASURED (runs/ab-reloc-moisture, arms C vs D): with the t = 0
    lineage on every move, d02's rain fell 12% and cloud rose 10% within
    150 s of a move, edge-weighted (-83% in the outer frame, -5% deep
    inside) and recovering over about one rain fall-through time.  The
    descendant d03, whose driver is rebuilt identically but whose
    placement does not change, moved 0.5%.
    """
    # WRF's reference T is moist-theta perturbation from 300 K because
    # use_theta_m=1.  gpuwm stores dry theta; use_theta_m=0 is the
    # algebraically equivalent dry-frame branch.
    theta_300 = state.total_theta() - np.float32(300.0)
    if column_mass_correction:
        pb = state.pb if state.pb.ndim == 3 else state.pb[:, None, None]
        pressure_perturbation = state.p - pb
        adjust_tempqv(
            state.mub2d, save_mub, state.c3h, state.c4h,
            float(state.p_top), theta_300, pressure_perturbation, state.qv,
            use_theta_m=0)
    adjusted_theta = theta_300 + np.float32(300.0)
    base = _base_from_blended(state, cfg, coord, float(state.p_top))
    state.load_base(coord, base)
    state.thp[...] = adjusted_theta - state.thb
    update_diagnostics(state, cfg.hypsometric_opt)
    if column_mass_correction:
        _apply_press_adj_mu(state, ht_fine)
    return base


def _updated_real_result(original: RealInitResult,
                         base: BaseState) -> RealInitResult:
    state = original.state
    total_mu = _host(state.total_mu())
    c3h = _host(state.c3h)[:, None, None]
    c4h = _host(state.c4h)[:, None, None]
    p_top = float(state.p_top)
    return RealInitResult(
        state=state, coord=original.coord, base=base,
        surface_pressure=original.surface_pressure,
        surface_qv=original.surface_qv,
        dry_mass=total_mu,
        dry_pressure=c3h * total_mu[None] + c4h + p_top,
        total_pressure=_host(state.p),
        total_geopotential=_host(state.phb + state.php),
        total_specific_volume=_host(state.alt),
        integrated_moisture_pressure=original.integrated_moisture_pressure,
        hypsometric_opt=original.hypsometric_opt,
        hydrometeor_initialization=original.hydrometeor_initialization,
        surface_moisture_floor=original.surface_moisture_floor,
        initial_perturbation=original.initial_perturbation)


def _prepare_child_input_on_grid(
        child_dc: DomainConfig, grid: LambertGrid, catalog,
        source_orography: SourceOrographyDeclaration | None = None,
        preprocess_backend: str | object = "cuda",
        preprocess_workers: int | None = None,
        cpu_bridge: Path | str | None = None,
) -> PreparedChildInput:
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend

    started = time.perf_counter()
    preprocess = resolve_preprocess_backend(
        preprocess_backend, workers=preprocess_workers,
        cpu_bridge=cpu_bridge)
    cfg = child_dc.run
    if not cfg.moist:
        raise ValueError("ERA5-direct child initialization requires moist=True")
    if not cfg.terrain_opt:
        raise ValueError(
            "ERA5-direct child initialization requires terrain_opt != 0")

    static_catalog = _static_catalog(catalog)
    static_fields = build_static_for_domain(
        grid, static_catalog, child_dc.grid_id)
    source = _initial_snapshot(catalog, child_dc.start_time)
    has_invariant = "SOILGEO" in tuple(getattr(catalog, "inventory", ()))
    catalog_declaration = _catalog_source_declaration(catalog)
    declaration = (source_orography if source_orography is not None
                   else catalog_declaration)
    if has_invariant and declaration is not None:
        raise ValueError(
            "source-orography conflict: catalog contains both SOILGEO and "
            "a declared source_orography artifact")
    if isinstance(source, HrrrNativeSnapshot):
        if declaration is not None or has_invariant:
            raise ValueError(
                "source-orography conflict: HRRR already supplies SOILHGT")
        declared_orography = None
    else:
        declared_orography = _declared_source_orography(
            catalog, child_dc.grid_id, declaration)
    landuse_attrs = geog_selection_from_catalog(
        static_catalog, child_dc.grid_id).landuse_global_attrs()
    # Optional [static.highres] overlay: the validated config rides the
    # top-level input catalog (gpuwm.ingest.preflight.build_input_catalog).
    # Absent/disabled is the identity 30-arc-second path.
    _highres = getattr(catalog, "static_highres", None)
    if _highres is not None and getattr(_highres, "enabled", False):
        from gpuwm.static.highres_production import apply_highres_statics
        static_fields, _ = apply_highres_statics(
            static_fields, grid, config=_highres,
            domain_id=child_dc.grid_id,
            case_date=child_dc.start_time.date(),
            landuse_attrs=landuse_attrs)
    # The child assembles its water temperature under the SAME policy as
    # its parent; the catalog carries it because the child sees no case
    # data of its own.  Lakes come from the child's OWN land-use table
    # via ISLAKE, never a hard-coded category.
    water_temperature_policy = getattr(
        catalog, "water_temperature_policy", None)
    # LANDMASK is what the soil router decides land and water with, so
    # the assembly cannot name a water surface without it.  Absent, no
    # assembly is declared and the router refuses this child by route
    # name if its mapping carries a raw SST beside its SKINTEMP.
    _child_landmask_static = static_fields.get("LANDMASK")
    water_statics = (
        None if _child_landmask_static is None
        else WaterTemperatureStatics.for_route(
            route=_WATER_ROUTE, policy=water_temperature_policy,
            landmask=_child_landmask_static,
            lu_index=static_fields["LU_INDEX"],
            landuse_attrs=landuse_attrs))
    lake_mask = (
        (np.asarray(static_fields["LU_INDEX"]) ==
         int(landuse_attrs["ISLAKE"])) if water_statics is None
        else water_statics.lake)
    mapping_receipt: dict[str, object] = dict(preprocess.receipt())
    if isinstance(source, HrrrNativeSnapshot):
        soil_mapping_report: dict[str, object] = {}
        catalog_provenance = dict(getattr(catalog, "provenance", {}))
        raw_radius = catalog_provenance.get(
            "surface_fallback_radius_cells")
        if raw_radius is None:
            if catalog_provenance.get("adapter") == \
                    "native-HRRR-hierarchy-direct-v1":
                raise ValueError(
                    "native HRRR hierarchy catalog lacks its bound surface "
                    "fallback radius")
            surface_fallback_radius = 8
        else:
            if (isinstance(raw_radius, bool)
                    or not isinstance(raw_radius, int)
                    or not 0 <= raw_radius <= 64):
                raise ValueError(
                    "HRRR catalog surface_fallback_radius_cells must be an "
                    "integer in [0, 64]")
            surface_fallback_radius = raw_radius
        horizontal = interpolate_hrrr_to_lambert(
            source, grid, target_landmask=static_fields["LANDMASK"],
            soil_mapping_report=soil_mapping_report,
            surface_fallback_radius=surface_fallback_radius,
            backend=preprocess,
            target_name=f"domain {child_dc.grid_id}")
        skin = _host(horizontal.fields["SKINTEMP"])
        lake_skin_temperature = np.where(lake_mask, skin, np.nan)
        mapping_receipt.update({
            "source_adapter": "hrrr-native-state-v1",
            "surface_fallback_radius_cells": surface_fallback_radius,
            "soil_mapping": soil_mapping_report,
        })
    else:
        # No lake skin override on the regular-grid lane: metgrid's
        # masked=both SKINTEMP chain with static-landmask targets already
        # yields water-source skin at lakes, matching real.exe's no-TAVGSFC
        # behavior (module_initialize_real.F:2844-2866).
        lake_skin_temperature = None
        # (1) REAL INPUT FIRST, on the child's own grid and unblended HGT_M.
        # Masked-field target cells classify by the child's static LANDMASK,
        # matching metgrid's model-landmask processing.
        child_landmask = static_fields.get("LANDMASK")
        horizontal = interpolate_era5_to_lambert(
            source, grid,
            source_orography_catalog=(catalog if has_invariant else None),
            target_landmask=(None if child_landmask is None
                             else np.asarray(child_landmask) >= 0.5),
            water_temperature_statics=water_statics,
            backend=preprocess)
        mapping_receipt["source_adapter"] = "regular-grid-native-state-v1"
    return PreparedChildInput(
        domain=child_dc, grid=grid, static_fields=static_fields,
        horizontal=horizontal, declared_orography=declared_orography,
        landuse_attrs=dict(landuse_attrs),
        lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
        preprocess_backend=preprocess,
        preprocess_receipt=mapping_receipt,
        water_temperature_policy=water_temperature_policy,
        soil_mesh=_child_soil_mesh(source, grid, catalog),
        preparation_seconds=time.perf_counter() - started)


def _child_soil_mesh(source, grid, catalog):
    """The child's own :class:`SoilMeshPlan`, or ``None`` if unmeasurable.

    A nest re-ingests the SAME forcing through the SAME interpolation onto
    a grid that is FINER than its parent's, so the source mesh is more
    visible in a child than in the domain the defect was reported on.  The
    enable declaration rides the catalog for the same reason the water
    policy does: a child sees no case data of its own, and it must not
    diverge from its parent on a correctness remedy.

    A source with no regular lat/lon axes (native HRRR) has no spacing to
    measure here and returns ``None``.
    """
    from gpuwm.ingest.soil_downscale import soil_mesh_plan_from_case

    return soil_mesh_plan_from_case(source, grid, catalog)


def prepare_child_input(
        child_dc: DomainConfig, parent_node: DomainNode, catalog,
        source_orography: SourceOrographyDeclaration | None = None,
        *, preprocess_backend: str | object = "cuda",
        preprocess_workers: int | None = None,
        cpu_bridge: Path | str | None = None,
) -> PreparedChildInput:
    """Build the parent-independent portion of one child."""

    return _prepare_child_input_on_grid(
        child_dc, _child_grid(child_dc, parent_node), catalog,
        source_orography, preprocess_backend, preprocess_workers, cpu_bridge)


def start_child_input_preparations(
        exp, catalog,
        source_orography: SourceOrographyDeclaration | None = None,
        *, workers: int = 8, preprocess_backend: str | object = "cpu",
        cpu_bridge: Path | str | None = None,
) -> PendingChildInputs:
    """Launch bounded per-child static/source mapping in parallel.

    Domain geometry comes entirely from the validated experiment, so even a
    grandchild can be mapped before its parent's model state exists.  Callers
    must still finalize results parent-before-child.
    """

    from gpuwm.static.lambert import grids_from_projection_config

    grids = tuple(grids_from_projection_config(exp))
    if len(grids) != len(exp.domains):
        raise ValueError("experiment did not resolve one grid per domain")
    by_id = {
        domain.grid_id: grid
        for domain, grid in zip(exp.domains, grids)
    }
    return PendingChildInputs(
        exp.domains[1:], by_id, catalog, source_orography, workers,
        preprocess_backend=preprocess_backend, cpu_bridge=cpu_bridge)


def prepare_child_inputs_parallel(
        exp, catalog,
        source_orography: SourceOrographyDeclaration | None = None,
        *, workers: int = 8, preprocess_backend: str | object = "cpu",
        cpu_bridge: Path | str | None = None,
) -> tuple[PreparedChildInput, ...]:
    """Blocking convenience wrapper around child-input preparation."""

    with start_child_input_preparations(
            exp, catalog, source_orography, workers=workers,
            preprocess_backend=preprocess_backend,
            cpu_bridge=cpu_bridge) as pending:
        return pending.result()


def _assert_prepared_grid_matches_parent(
        prepared: PreparedChildInput, parent_node: DomainNode) -> None:
    expected = _child_grid(prepared.domain, parent_node)
    if prepared.grid is expected:
        return
    names = (
        "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
        "dx", "dy", "e_we", "e_sn", "known_x", "known_y",
    )
    drift = {
        name: {
            "prepared": getattr(prepared.grid, name),
            "parent_resolved": getattr(expected, name),
        }
        for name in names
        if getattr(prepared.grid, name) != getattr(expected, name)
    }
    if drift:
        raise ValueError(
            f"d{prepared.domain.grid_id:02d} prepared grid differs from "
            f"the live parent: {drift}")


def finalize_prepared_child(
        prepared: PreparedChildInput, parent_node: DomainNode,
        vertical: VerticalConfig, *, scratch_arena=None,
        dycore_state_workspace=None,
        state_backend: str = "cuda",
        sfcp_to_sfcp: bool = True,
        soil_layer_contract=None,
        initial_perturbation=None) -> ChildInitResult:
    """Run GPU/model-state and parent-dependent WRF child initialization.

    ``initial_perturbation`` is the experiment's validated
    :class:`gpuwm.experiment.PerturbationConfig` (or ``None``), passed
    only for children that initialize AT the experiment start time.  The
    bubbles are evaluated on THIS child's own grid inside its own
    ``initialize_real`` -- the child does not inherit the parent's
    perturbed theta, because real-data nest init re-ingests the source
    analysis per domain -- and they land BEFORE the terrain blend /
    adjust_tempqv / press_adj sequence, so the perturbed columns pass
    through WRF's child adjustments exactly like the analyzed ones.  A
    bubble centered outside this child is recorded, not refused; the
    coarse-domain containment refusal already ran.
    """

    if not isinstance(prepared, PreparedChildInput):
        raise TypeError("prepared must be a PreparedChildInput")
    if not isinstance(vertical, VerticalConfig):
        raise TypeError("vertical must be the shared VerticalConfig")
    child_dc = prepared.domain
    cfg = child_dc.run
    _assert_prepared_grid_matches_parent(prepared, parent_node)
    grid = prepared.grid
    static_fields = prepared.static_fields
    horizontal = prepared.horizontal
    coord = _shared_vertical_coord(vertical, cfg.nz)
    init_kwargs = dict(
        source_orography=prepared.declared_orography, p_top=vertical.p_top,
        sfcp_to_sfcp=sfcp_to_sfcp,
        preprocess_backend=prepared.preprocess_backend,
        state_backend=state_backend)
    if scratch_arena is not None:
        init_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        init_kwargs["dycore_state_workspace"] = dycore_state_workspace
    if initial_perturbation is not None:
        from gpuwm.ingest.init_perturbation import (
            build_initial_state_perturbation)
        init_kwargs["initial_perturbation"] = \
            build_initial_state_perturbation(
                initial_perturbation, grid, grid_id=int(child_dc.grid_id),
                require_containment=False)
    real = initialize_real(
        horizontal, cfg, coord, static_fields["HGT_M"], **init_kwargs)
    state = real.state
    _set_map_fields(state, grid)
    # initialize_real deliberately leaves EOS diagnostics to its caller.
    # adjust_tempqv needs the pre-blend perturbation pressure.
    update_diagnostics(state, cfg.hypsometric_opt)
    # The child's own horizontal fields carry the forcing's embedded
    # orography (nest_init passes the catalog to the interpolation above),
    # so a nested ERA5 case whose SOILGEO rides in the GRIB gets the same
    # elevation lapse the root does.  Resolving only prepared.declared_
    # orography skipped it on exactly the cases the root skipped it on --
    # and nests, being the finest grids, carry the sharpest terrain
    # disagreement with a ~31 km source and feel it most.
    child_orography = soil_source_orography(
        prepared.declared_orography, horizontal.fields)
    # ONE RULEBOOK (Drew's ruling, 2026-08-06): build the child's soil
    # column and its SH2O with the SAME reconciled ISLTYP the child's Noah
    # integrates, exactly as the root path does.  Nests are the finest
    # grids and so carry the most land/water-disagreeing shoreline cells.
    from gpuwm.core.landuse import reconciled_soil_category

    child_attrs = prepared.landuse_attrs
    child_soil_type = static_fields["SCT_DOM"]
    if child_attrs is not None:
        # The reconciler's two pieces of evidence come from ONE table of
        # per-source spellings (gpuwm/ingest/soil.py), never from a chain
        # written out here.  A chain written out here was short by the
        # native-HRRR SOILT spelling on 2026-08-06 and by the GFS
        # GFS_ST000010 spelling on 2026-08-08, and both times a child that
        # was holding the soil column right there in horizontal.fields
        # aborted with mismatch_landmask_ivgtyp instead of reading it.
        # Nests are the finest grids, so they carry the most disagreeing
        # shoreline and inland-water columns and feel this first.
        child_soil_type = reconciled_soil_category(
            static_fields["LU_INDEX"], soil_type=child_soil_type,
            xice=horizontal.fields.get("XICE", 0.0),
            iswater=int(child_attrs["ISWATER"]),
            islake=int(child_attrs["ISLAKE"]),
            isice=int(child_attrs["ISICE"]),
            soil_temperature=reconciler_soil_temperature(horizontal.fields),
            sst=reconciler_sst(horizontal.fields))
    soil = preprocess_land_surface_soil(
        horizontal.fields, sf_surface_physics=int(cfg.sf_surface_physics),
        soil_type=child_soil_type,
        deep_soil_temperature=static_fields["TMN"],
        lake_mask=(prepared.lake_mask
                   if prepared.lake_skin_temperature is not None else None),
        lake_skin_temperature=prepared.lake_skin_temperature,
        soil_layer_contract=soil_layer_contract,
        landmask=static_fields.get("LANDMASK"),
        terrain=(static_fields["HGT_M"]
                 if child_orography is not None else None),
        source_orography=child_orography,
        # getattr, exactly like the mapped route: a horizontal
        # snapshot without an assembly is not a silent fallback any
        # more, it is what the router refuses by route name when the
        # mapping carries a raw SST beside its SKINTEMP.
        water_temperature=getattr(horizontal, "water_temperature", None),
        water_temperature_policy=prepared.water_temperature_policy,
        # A nest inherits the defect unless it inherits the remedy: this
        # child's soil column is built from the same 0.25 degree source
        # its parent's was, onto a grid where one source cell is even
        # wider in model cells.
        soil_mesh=prepared.soil_mesh,
        route=_WATER_ROUTE)
    save_mub = state.mub2d.copy()

    # (2) Scope-trimmed SINT capture: ht/mub/phb ONLY.
    ht_int, mub_int, phb_int = _capture_parent_blend_fields(
        child_dc, parent_node)
    # A sealed root can have been prepared on CUDA while deterministic child
    # workers use the CPU backend (or vice versa).  SINT follows the parent
    # backend, so move its three retained operands to the new child's backend
    # before the in-place blend.
    ht_int = _as_like(ht_int, state.ht)
    mub_int = _as_like(mub_int, state.mub2d)
    phb_int = _as_like(phb_int, state.phb)

    # (3) WRF blends all three fields.  Never replace this with
    # blend-ht-then-derive: base-state construction is nonlinear.
    spec_width = int(cfg.spec_bdy_width)
    blend_width = int(getattr(child_dc, "blend_width", 5))
    blend_terrain(ht_int, state.ht, spec_bdy_width=spec_width,
                  blend_width=blend_width)
    blend_terrain(mub_int, state.mub2d, spec_bdy_width=spec_width,
                  blend_width=blend_width)
    blend_terrain(phb_int, state.phb, spec_bdy_width=spec_width,
                  blend_width=blend_width)

    # (4) theta/qv adjustment, then (5) start_domain base/EOS re-derivation
    # followed by its real-nest press_adj MU correction.  HGT_M remains the
    # saved pre-blend ht_fine operand.
    base = _adjust_and_rederive(
        state, cfg, coord, save_mub, static_fields["HGT_M"])
    updated = _updated_real_result(real, base)

    # The three SINT captures and adjustment work arrays are init-only.  Soil
    # remains exactly the pre-blend object constructed above.
    del ht_int, mub_int, phb_int, save_mub
    return ChildInitResult(
        state=state, grid=grid, coord=coord, real=updated,
        static_fields=static_fields, horizontal=horizontal, soil=soil,
        preprocess_receipt=prepared.preprocess_receipt,
        input_preparation_seconds=prepared.preparation_seconds,
        domain=child_dc)


def finalize_prepared_child_chain(
        prepared_inputs: Iterable[PreparedChildInput],
        root_node: DomainNode | ParentInitView,
        vertical: VerticalConfig, *, scratch_arena=None,
        dycore_state_workspace=None,
        state_backend: str = "cuda",
        sfcp_to_sfcp: bool = True,
        soil_layer_contract=None,
) -> tuple[ChildInitResult, ...]:
    """Finalize one prepared hierarchy at an explicit parent barrier.

    ``prepared_inputs`` must be parent-before-child.  Only the minimal
    initialized parent view is retained, which keeps this dependency phase
    usable by both the forecast tree and the native stock-WRF artifact path.
    """

    root_id = int(root_node.cfg.grid_id)
    parents: dict[int, DomainNode | ParentInitView] = {root_id: root_node}
    results = []
    for prepared in prepared_inputs:
        domain = prepared.domain
        grid_id = int(domain.grid_id)
        parent_id = int(domain.parent_id)
        if grid_id in parents:
            raise ValueError(
                f"prepared child hierarchy repeats d{grid_id:02d}")
        if parent_id not in parents:
            raise ValueError(
                f"prepared child d{grid_id:02d} reached the dependency "
                f"barrier before parent d{parent_id:02d}")
        result = finalize_prepared_child(
            prepared, parents[parent_id], vertical,
            scratch_arena=scratch_arena,
            dycore_state_workspace=dycore_state_workspace,
            state_backend=state_backend,
            sfcp_to_sfcp=sfcp_to_sfcp,
            soil_layer_contract=soil_layer_contract)
        results.append(result)
        parents[grid_id] = ParentInitView(
            cfg=domain, grid=result.grid, state=result.state)
    return tuple(results)


def initialize_child_chain_parallel(
        exp, root_node: DomainNode | ParentInitView, catalog,
        source_orography: SourceOrographyDeclaration | None = None,
        *, workers: int = 8, preprocess_backend: str | object = "cpu",
        cpu_bridge: Path | str | None = None, scratch_arena=None,
        dycore_state_workspace=None, state_backend: str = "cuda",
        sfcp_to_sfcp: bool = True,
        soil_layer_contract=None,
) -> tuple[ChildInitResult, ...]:
    """Run bounded independent preparation, then the parent dependency phase."""

    if int(root_node.cfg.grid_id) != int(exp.domains[0].grid_id):
        raise ValueError(
            "root node does not match the first experiment domain")
    with start_child_input_preparations(
            exp, catalog, source_orography, workers=workers,
            preprocess_backend=preprocess_backend,
            cpu_bridge=cpu_bridge) as pending:
        return finalize_prepared_child_chain(
            pending.iter_parent_order(), root_node, exp.vertical,
            scratch_arena=scratch_arena,
            dycore_state_workspace=dycore_state_workspace,
            state_backend=state_backend,
            sfcp_to_sfcp=sfcp_to_sfcp,
            soil_layer_contract=soil_layer_contract)


def initialize_child(
        child_dc: DomainConfig, parent_node: DomainNode, catalog,
        vertical: VerticalConfig,
        source_orography: SourceOrographyDeclaration | None = None,
        *, scratch_arena=None, dycore_state_workspace=None,
        state_backend: str = "cuda",
        sfcp_to_sfcp: bool = True,
        soil_layer_contract=None,
        preprocess_backend: str | object = "cuda",
        preprocess_workers: int | None = None,
        cpu_bridge: Path | str | None = None,
        initial_perturbation=None) -> ChildInitResult:
    """Initialize a real-data child in the binding five-step WRF order.

    The public sequential behavior is preserved.  Multi-domain startup may
    call :func:`start_child_input_preparations` early, then invoke
    :func:`finalize_prepared_child` parent-before-child at the dependency
    barrier.
    """

    if not isinstance(vertical, VerticalConfig):
        raise TypeError("vertical must be the shared VerticalConfig")
    prepared = prepare_child_input(
        child_dc, parent_node, catalog, source_orography,
        preprocess_backend=preprocess_backend,
        preprocess_workers=preprocess_workers,
        cpu_bridge=cpu_bridge)
    return finalize_prepared_child(
        prepared, parent_node, vertical,
        scratch_arena=scratch_arena,
        dycore_state_workspace=dycore_state_workspace,
        state_backend=state_backend,
        sfcp_to_sfcp=sfcp_to_sfcp,
        soil_layer_contract=soil_layer_contract,
        initial_perturbation=initial_perturbation)


def _coord_from_parent(state: DomainState, parent_cfg) -> VerticalCoord:
    """Copy the already-built shared coordinate for parent-only setup."""
    c1h = np.array(_host(state.c1h), copy=True)
    hybrid_opt = int(getattr(
        parent_cfg, "hybrid_opt",
        1 if np.array_equal(c1h, np.ones_like(c1h)) else 2))
    return VerticalCoord(
        znw=np.array(_host(state.znw), copy=True),
        znu=np.array(_host(state.znu), copy=True),
        dnw=np.array(_host(state.dnw), copy=True),
        rdnw=np.array(_host(state.rdnw), copy=True),
        dn=np.array(_host(state.dn), copy=True),
        rdn=np.array(_host(state.rdn), copy=True),
        fnp=np.array(_host(state.fnp), copy=True),
        fnm=np.array(_host(state.fnm), copy=True),
        hybrid_opt=hybrid_opt,
        etac=float(getattr(parent_cfg, "etac", 0.2)),
        p_top=(None if state.p_top is None else float(state.p_top)),
        c1f=np.array(_host(state.c1f), copy=True),
        c2f=np.array(_host(state.c2f), copy=True),
        c3f=np.array(_host(state.c3f), copy=True),
        c4f=np.array(_host(state.c4f), copy=True),
        c1h=c1h,
        c2h=np.array(_host(state.c2h), copy=True),
        c3h=np.array(_host(state.c3h), copy=True),
        c4h=np.array(_host(state.c4h), copy=True))


#: WRF ``start_domain``'s seeding pairs: the RK time-t copy each current
#: field is taken from.  Named rather than inlined because a child's state
#: is seeded on TWO occasions -- when it is cold-started from its parent,
#: and when a discrete relocation stamps an integrated state onto a child
#: rebuilt at a new placement -- and a pair that reached one path but not
#: the other would leave the first substep reading a stale copy.
RK_TIME_T_SEED_PAIRS = (
    ("u", "u0"), ("v", "v0"), ("w", "w0"),
    ("thp", "thp0"), ("php", "php0"), ("mup", "mup0"),
    ("qv", "qv0"), ("qc", "qc0"), ("qr", "qr0"),
    ("qi", "qi0"), ("qs", "qs0"), ("qg", "qg0"),
    ("nr", "nr0"), ("ni", "ni0"), ("ns", "ns0"),
    ("ng", "ng0"), ("qh", "qh0"),
    # mp=28.  ``nc0`` exists only for mp=28; a Morrison child has ``nc``
    # but no ``nc0`` and the None guard below skips it, so this row cannot
    # start seeding Morrison's untransported droplet number.
    ("nc", "nc0"), ("nwfa", "nwfa0"), ("nifa", "nifa0"),
    ("qndrop", "qndrop0"), ("qnr", "qnr0"),
    ("qni", "qni0"), ("qns", "qns0"), ("qng", "qng0"),
    ("qnh", "qnh0"), ("qnn", "qnn0"),
    ("qvolg", "qvolg0"), ("qvolh", "qvolh0"),
)


def seed_rk_time_t_copies(state) -> tuple[str, ...]:
    """Seed the RK time-t copies from the current fields (WRF start_domain).

    Returns the seeds actually written, so a caller can record which of the
    optional scheme-dependent copies its state carried rather than assuming
    the whole inventory was present.
    """
    written: list[str] = []
    for current, initial in RK_TIME_T_SEED_PAIRS:
        value = getattr(state, current, None)
        seed = getattr(state, initial, None)
        if value is not None and seed is not None:
            seed[...] = value
            written.append(initial)
    return tuple(written)


def _parent_only_base(parent, reg, terrain: bool) -> BaseState:
    if not terrain:
        return BaseState(
            mub=float(parent.mub), p_top=float(parent.p_top),
            pb=np.array(_host(parent.pb), copy=True),
            alb=np.array(_host(parent.alb), copy=True),
            thb=np.array(_host(parent.thb), copy=True),
            phb=np.array(_host(parent.phb), copy=True), terrain_z=None)
    return BaseState(
        mub=_host(sint(parent.mub2d, reg)), p_top=float(parent.p_top),
        pb=_host(sint(parent.pb, reg)),
        alb=_host(sint(parent.alb, reg)),
        thb=_host(sint(parent.thb, reg)),
        phb=_host(sint(parent.phb, reg)),
        terrain_z=_host(sint(parent.ht, reg)))


#: The parent-SINT fields this module fixes up after interpolation: the
#: microphysics NUMBER concentrations.  Every one of them is bounded at
#: zero from below by :func:`gpuwm.core.health.rule_for_field` (asserted
#: in the tests, so the clamp and the gate cannot drift apart about
#: membership), and every one carries magnitudes of 1e3..1e9 per
#: kilogram -- large enough that a float32 weighted sum can round across
#: zero.  Moisture mixing ratios share the same lower bound but are
#: O(1e-3), so their absolute rounding error is smaller by six decades
#: and has never been observed to cross; they are deliberately NOT
#: clamped here, because a fix-up with no evidence behind it is a
#: trajectory change nobody asked for.
POSITIVE_DEFINITE_MOMENTS = (
    "nc", "nr", "ni", "ns", "ng", "qndrop", "qnr", "qni", "qns",
    "qng", "qnh", "qnn", "nwfa", "nifa",
)

#: Rounding budget for one SINT, in units in the last place of the
#: field's own peak.  The interpolation is a weighted sum of float32
#: donors with negative lobes, so the error it can introduce tracks the
#: LARGEST magnitude it summed rather than the value it happens to land
#: on.  Eight ulps is generous for the stencil and still leaves any
#: physically meaningful concentration far outside the clamp.
_SINT_ROUNDING_ULPS = 8.0

#: Absolute floor, for a field whose peak carries no scale at all.  The
#: undershoot measured on the card was -2**-31 = -4.66e-10 per kilogram
#: in a field that was zero everywhere, where the relative term above is
#: itself zero.  1e-6 clears that by four decades while staying six
#: decades BELOW one particle per kilogram -- a quantity no microphysics
#: scheme can tell apart from none -- so the floor cannot erase anything
#: countable.
_SINT_ABSOLUTE_FLOOR = 1.0e-6


def positive_definite_clamp_tolerance(peak: float) -> float:
    """How far below zero a SINT of a field peaking at ``peak`` may land.

    Anything inside this is float32 arithmetic; anything beyond it is an
    interpolation that is actually wrong, and is left for the health gate
    to refuse.  A clamp with no ceiling would absorb exactly the failure
    the gate exists to catch.
    """
    eps = float(np.finfo(np.float32).eps)
    return max(_SINT_ABSOLUTE_FLOOR,
               _SINT_ROUNDING_ULPS * eps * abs(float(peak)))


def clamp_parent_sint_undershoot(child, *, names=POSITIVE_DEFINITE_MOMENTS):
    """Set rounding-scale negatives to zero on a freshly SINT-ed child.

    Returns a per-field account of what was changed -- cells, the
    tolerance applied, the field's peak and the most negative value seen
    -- so a birth certificate can carry the numbers rather than the
    assurance.  A field with nothing to clamp contributes NO entry, so a
    silent receipt means the interpolation landed clean.

    Values more negative than the field's tolerance are left exactly
    where they are.  This is the whole design: the fix-up removes a
    float32 artefact and refuses to hide a broken interpolation.
    """
    report: dict[str, dict[str, float | int]] = {}
    for name in names:
        target = getattr(child, name, None)
        if target is None or getattr(target, "size", 0) == 0:
            continue
        minimum = float(target.min())
        if minimum >= 0.0:
            continue
        peak = float(abs(target).max())
        tolerance = positive_definite_clamp_tolerance(peak)
        mask = (target < 0.0) & (target >= -tolerance)
        cells = int(mask.sum())
        if not cells:
            continue
        target[mask] = 0.0
        report[name] = {
            "cells": cells,
            "tolerance": tolerance,
            "peak": peak,
            "most_negative": minimum,
        }
    return report


def parent_only_init(child_dc: DomainConfig,
                     parent_node: DomainNode, *,
                     scratch_arena=None,
                     dycore_state_workspace=None,
                     array_module=None,
                     grid: LambertGrid | None = None) -> ChildInitResult:
    """Initialize an idealized ``input_from_file=F`` nest from its parent.

    Unlike the real-input scope trim, this branch retains the full-parent
    SINT fill: prognostics, active moisture/scalars, and horizontally varying
    base fields.  The vertical coordinate is copied from the parent; it is
    never derived per child.

    ``array_module`` is the :class:`DomainState` setup seam (NumPy for a
    CPU-testable child, default CuPy); it exists so the spawn path's CPU
    contracts exercise this exact initializer rather than a mock of it.

    ``grid`` optionally supplies an already-constructed child grid whose
    geometry must match the parent-resolved one (a relocation initializer
    passes its placement-translated grid so the map-factor/Coriolis fields
    are bitwise stable across placements on shared ground); omitted, the
    grid is resolved from the parent exactly as before.
    """
    cfg = child_dc.run
    parent = parent_node.state
    if cfg.nz != parent_node.cfg.run.nz:
        raise ValueError("parent-only initialization forbids vertical nesting")
    for name in ("hybrid_opt", "etac"):
        child_value = getattr(cfg, name, None)
        parent_value = getattr(parent_node.cfg.run, name, child_value)
        if child_value != parent_value:
            raise ValueError(
                f"parent-only initialization requires shared {name}: "
                f"child={child_value!r}, parent={parent_value!r}")
    if grid is None:
        grid = _child_grid(child_dc, parent_node)
    else:
        expected = (cfg.nx + 1, cfg.ny + 1, cfg.dx, cfg.dy)
        actual = (grid.e_we, grid.e_sn, grid.dx, grid.dy)
        if actual != expected:
            raise ValueError(
                f"supplied child grid geometry {actual} differs from the "
                f"config's (e_we, e_sn, dx, dy) = {expected}")
    mass_reg = _mass_registration(child_dc, parent_node)
    coord = _coord_from_parent(parent, parent_node.cfg.run)
    state_kwargs = {}
    if scratch_arena is not None:
        state_kwargs["scratch_arena"] = scratch_arena
    if dycore_state_workspace is not None:
        state_kwargs["dycore_state_workspace"] = dycore_state_workspace
    if array_module is not None:
        state_kwargs["array_module"] = array_module
    child = DomainState(cfg, **state_kwargs)
    child.load_base(
        coord, _parent_only_base(parent, mass_reg, bool(cfg.terrain_opt)))

    registrations = {
        "": mass_reg,
        "x": _registration(child_dc, parent_node, "x"),
        "y": _registration(child_dc, parent_node, "y"),
    }
    transition = resolve_microphysics_transition(parent_node.cfg.run, cfg)
    transition_backing = None
    if transition.mixed:
        parent_run = parent_node.cfg.run
        pnz, pny, pnx = (
            int(parent_run.nz), int(parent_run.ny), int(parent_run.nx))
        # Exact F16 full-field capacity without importing the forecast-only
        # preflight module into the standalone RW-WPS preparation wheel.
        slot_shape = (max(
            pnz * pny * (pnx + 1),
            pnz * (pny + 1) * pnx,
            (pnz + 1) * pny * pnx,
        ),)
        transition_backing = child.scratch(slot_shape, "nest_parent_field")
    for name, stagger in (("u", "x"), ("v", "y"), ("w", ""),
                          ("thp", ""), ("php", ""), ("mup", ""),
                          ("qv", ""), ("qc", ""), ("qr", ""),
                          ("qi", ""), ("qs", ""), ("qg", ""),
                          ("nc", ""), ("nr", ""), ("ni", ""),
                          ("ns", ""), ("ng", ""),
                          # mp=28 aerosol number tracers.  Mass-stagger,
                          # same generic SINT path as every other scalar;
                          # the loop is None-guarded so no other scheme
                          # sees them.
                          ("nwfa", ""), ("nifa", ""),
                          # mp=28 surface aerosol emission tendencies.
                          # 2-D mass-stagger, exactly like ``mup`` above.
                          # This matches WRF: Registry.EM_COMMON:492-493
                          # gives QNWFA2D/QNIFA2D the IO string
                          # ``i01{17}rhdu``, whose bare ``d`` is nest-down
                          # with the DEFAULT mass interpolator (compare the
                          # explicit ``d=(interp_mask_field:...)`` form at
                          # :871), so WRF interpolates them to a child too.
                          #
                          # It also has to happen here.  They are cross-step
                          # constants, and an aerosol-aware child whose
                          # parent already carries CCN takes thompson_init's
                          # has_CCN branch (module_mp_thompson.F:516-522),
                          # which does NOT derive nwfa2d -- :510 runs only on
                          # the fill branch.  Without this row the nest would
                          # run with zero surface aerosol emission under a
                          # parent that has it.
                          ("nwfa2d", ""), ("nifa2d", ""),
                          ("qh", ""), ("qndrop", ""), ("qnr", ""),
                          ("qni", ""), ("qns", ""), ("qng", ""),
                          ("qnh", ""), ("qnn", ""), ("qvolg", ""),
                          ("qvolh", ""),
                          ("h_diabatic", ""),
                          # The dycore's exported advective forcing pair
                          # (WRF RTHFTEN/RQVFTEN).  Mass-stagger, the same
                          # generic SINT path every scalar takes.  A child
                          # cold-starts with no previous dynamics step of
                          # its own, so without these rows its first
                          # cumulus call would see hard zeros in exactly
                          # the lane the parent has been feeding -- and
                          # unlike h_diabatic they are NOT reset on a
                          # microphysics-scheme boundary, because they are
                          # a dynamics product and carry nothing the
                          # departing scheme owned.
                          ("rthften", ""), ("rqvften", "")):
        source = getattr(parent, name, None)
        target = getattr(child, name, None)
        if target is None:
            continue
        if transition_handles_field(transition, name):
            parent_shape = transition_parent_field_shape(parent, name)
            count = int(np.prod(parent_shape))
            if transition_backing is None or count > transition_backing.size:
                raise RuntimeError(
                    "parent transition field exceeds F16 arena capacity")
            backing = transition_backing.reshape(-1)[:count].reshape(
                parent_shape)
            launch_microphysics_edge_parent_field(
                transition, parent, name, out=backing, coupled=False)
            sint(backing, registrations[stagger], out=target)
        elif source is not None and not (
                transition.mixed and name == "h_diabatic"):
            target[...] = sint(source, registrations[stagger])

    # A scheme boundary is a reconstructed child cold start.  NSSL-only
    # moments above are canonicalized from parent mass, while retained latent
    # heating must begin at zero rather than inheriting Thompson's closure.
    if transition.mixed and child.h_diabatic is not None:
        child.h_diabatic[...] = 0.0
    if transition.mixed:
        child._microphysics_transition_init_count = 1

    # Positive-definite fix-up BEFORE the RK time-level copies: the
    # seeded copies are what the first step reads, so a negative left
    # here would be duplicated into them and then refused by the
    # full-state gate at the newborn's first leg.
    clamped = clamp_parent_sint_undershoot(child)

    seed_rk_time_t_copies(child)
    _set_map_fields(child, grid)
    update_diagnostics(child, cfg.hypsometric_opt)
    return ChildInitResult(
        state=child, grid=grid, coord=coord, real=None,
        static_fields=None, horizontal=None, soil=None, domain=child_dc,
        positive_definite_clamp=clamped)


def blend_zone_mask(shape: tuple[int, int], *, spec_bdy_width: int = 5,
                    blend_width: int = 5) -> np.ndarray:
    """Boolean WRF specified-plus-blend frame for an unstaggered field."""
    ny, nx = map(int, shape)
    width = int(spec_bdy_width) + int(blend_width)
    if ny <= 0 or nx <= 0 or width <= 0:
        raise ValueError("shape and blend-zone width must be positive")
    j, i = np.indices((ny, nx))
    edge_distance = np.minimum.reduce((i, nx - 1 - i, j, ny - 1 - j))
    return edge_distance < width


@dataclass(frozen=True)
class N1StaticOracleResult:
    metrics: Mapping[str, float]
    fine_hgt_max_abs_diff_m: Mapping[str, float]
    passed: bool


def _produced_static(value) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray | None]:
    if isinstance(value, ChildInitResult):
        fine = (None if value.static_fields is None else
                np.asarray(value.static_fields["HGT_M"], dtype=np.float64))
        return (_host(value.state.ht), _host(value.state.mub2d), fine)
    if isinstance(value, (str, Path)):
        with np.load(value) as payload:
            fine = (np.asarray(payload["HGT_UNBLENDED"], dtype=np.float64)
                    if "HGT_UNBLENDED" in payload else None)
            return (np.asarray(payload["HGT"], dtype=np.float64),
                    np.asarray(payload["MUB"], dtype=np.float64), fine)
    if isinstance(value, Mapping):
        fine_value = value.get("HGT_UNBLENDED")
        return (np.asarray(value["HGT"], dtype=np.float64),
                np.asarray(value["MUB"], dtype=np.float64),
                (None if fine_value is None else
                 np.asarray(fine_value, dtype=np.float64)))
    raise TypeError("produced child must be ChildInitResult, mapping, or NPZ path")


def _reference_frame_path(directory: Path, label: str,
                          reference_frames: Mapping[int | str, object] | None,
                          domain_id: int) -> Path:
    """Resolve one child's reference wrfout frame.

    The frame name is case data -- it carries the campaign's valid time --
    so this generic module never spells one.  A caller that knows its case
    declares it through ``reference_frames`` (a bare name resolves inside
    the bundle's ``wrfout_reference`` directory, an absolute path is used
    as given).  Otherwise the frame is discovered in the bundle, which must
    hold exactly one frame per domain; anything else is a hard error naming
    the candidates rather than a silent pick.
    """
    declared = None
    if reference_frames:
        for key in (domain_id, label):
            if key in reference_frames:
                declared = reference_frames[key]
                break
    if declared is not None:
        path = Path(declared)
        return path if path.is_absolute() else directory / path
    matches = sorted(directory.glob(f"wrfout_{label}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"{label}: expected exactly one reference frame matching "
            f"wrfout_{label}_* under {directory}, found {len(matches)}"
            + (f" ({[p.name for p in matches]})" if matches else "")
            + "; pass reference_frames={" + f"{label!r}: <frame name>" + "} "
            "to name the frame this comparison is against.")
    return matches[0]


def compare_n1_static(produced: Mapping[int | str, object], bundle,
                      *, spec_bdy_width: int = 5, blend_width: int = 5,
                      reference_frames: Mapping[int | str, object] | None = None
                      ) -> N1StaticOracleResult:
    """Execute the pre-registered N1 child HGT/MUB static comparator.

    ``produced`` supplies d02/d03/d04 as live :class:`ChildInitResult`
    objects, mappings with ``HGT``/``MUB``/optional ``HGT_UNBLENDED``, or
    NPZ files with those keys.  The reference ``geo_em HGT_M`` is checked
    against the retained unblended input when provided; the blocking gate is
    the direct blended HGT and MUB comparison in the WRF blend zone.

    ``reference_frames`` optionally declares the reference wrfout frame per
    domain (keyed by id or ``dNN`` label); see :func:`_reference_frame_path`
    for the discovery fallback.
    """
    from gpuwm import netcdf_bridge

    # Ratified N1 static comparator thresholds.  Keep the production-side
    # optional comparator self-contained: standalone RW-WPS intentionally
    # excludes the developer verification package.
    hgt_blend_max_abs_diff_m = 1.5
    mub_blend_max_abs_diff_pa = 17.0

    bundle = Path(bundle)
    metrics: dict[str, float] = {}
    fine_metrics: dict[str, float] = {}
    passed = True
    for domain_id in (2, 3, 4):
        label = f"d{domain_id:02d}"
        value = produced.get(domain_id, produced.get(label))
        if value is None:
            raise KeyError(f"produced static fields omit {label}")
        hgt, mub, fine = _produced_static(value)
        wrf_path = _reference_frame_path(
            bundle / "wrfout_reference", label, reference_frames, domain_id)
        geo_path = bundle / "geo_em" / f"geo_em.{label}.nc"
        # HGT/MUB out of a WRF history tape and HGT_M out of a WPS
        # geo_em are meteorological/static fields being decoded, so both
        # go through the Rust bridge.
        with netcdf_bridge.open_dataset(wrf_path) as dataset:
            reference_hgt = np.asarray(dataset.variables["HGT"][0],
                                       dtype=np.float64)
            reference_mub = np.asarray(dataset.variables["MUB"][0],
                                       dtype=np.float64)
        with netcdf_bridge.open_dataset(geo_path) as dataset:
            reference_fine = np.asarray(dataset.variables["HGT_M"][0],
                                        dtype=np.float64)
        if hgt.shape != reference_hgt.shape or mub.shape != reference_mub.shape:
            raise ValueError(
                f"{label} produced/reference shapes differ: HGT "
                f"{hgt.shape}/{reference_hgt.shape}, MUB "
                f"{mub.shape}/{reference_mub.shape}")
        zone = blend_zone_mask(
            hgt.shape, spec_bdy_width=spec_bdy_width,
            blend_width=blend_width)
        hgt_metric = float(np.max(np.abs(hgt[zone] - reference_hgt[zone])))
        mub_metric = float(np.max(np.abs(mub[zone] - reference_mub[zone])))
        metrics[f"{label}_hgt_blend_max_abs_diff_m"] = hgt_metric
        metrics[f"{label}_mub_blend_max_abs_diff_pa"] = mub_metric
        passed &= (np.isfinite(hgt_metric)
                   and hgt_metric <= hgt_blend_max_abs_diff_m)
        passed &= (np.isfinite(mub_metric)
                   and mub_metric <= mub_blend_max_abs_diff_pa)
        if fine is not None:
            if fine.shape != reference_fine.shape:
                raise ValueError(
                    f"{label} unblended HGT shape {fine.shape} differs from "
                    f"geo_em HGT_M {reference_fine.shape}")
            fine_metrics[label] = float(
                np.max(np.abs(fine - reference_fine)))
    return N1StaticOracleResult(
        metrics=metrics, fine_hgt_max_abs_diff_m=fine_metrics,
        passed=bool(passed))


def write_n1_static_npz(result: ChildInitResult, path) -> Path:
    """Write the minimal controller-transfer artifact for N1 comparison."""
    if result.static_fields is None:
        raise ValueError("parent-only initialization has no N1 static fields")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, HGT=_host(result.state.ht), MUB=_host(result.state.mub2d),
        HGT_UNBLENDED=np.asarray(result.static_fields["HGT_M"],
                                 dtype=np.float64))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.ingest.nest_init",
        description="Run the pre-registered N1 child static oracle")
    parser.add_argument("--bundle", required=True, type=Path)
    for domain_id in (2, 3, 4):
        parser.add_argument(f"--d{domain_id:02d}", required=True, type=Path,
                            help="produced N1 NPZ for this child")
    parser.add_argument(
        "--reference-frame", action="append", default=[], metavar="dNN=FRAME",
        help="reference wrfout frame for one child, e.g. "
             "d02=wrfout_d02_<valid time>; repeatable.  Omit when the "
             "bundle holds exactly one frame per domain, which is then "
             "discovered.")
    args = parser.parse_args(argv)
    reference_frames: dict[str, str] = {}
    for pair in args.reference_frame:
        label, sep, frame = pair.partition("=")
        if not sep or not label or not frame:
            parser.error(
                f"--reference-frame {pair!r} must have the form dNN=FRAME")
        reference_frames[label] = frame
    result = compare_n1_static(
        {2: args.d02, 3: args.d03, 4: args.d04}, args.bundle,
        reference_frames=reference_frames or None)
    print(json.dumps({
        "metrics": dict(result.metrics),
        "fine_hgt_max_abs_diff_m": dict(result.fine_hgt_max_abs_diff_m),
        "passed": result.passed,
    }, indent=2, sort_keys=True))
    return 0 if result.passed else 1


__all__ = [
    "ChildInitResult", "N1StaticOracleResult", "NestedInputCatalog",
    "ParentInitView",
    "PendingChildInputs", "PreparedChildInput", "RK_TIME_T_SEED_PAIRS",
    "blend_zone_mask",
    "compare_n1_static", "finalize_prepared_child",
    "finalize_prepared_child_chain", "initialize_child",
    "initialize_child_chain_parallel", "parent_only_init",
    "prepare_child_input", "prepare_child_inputs_parallel",
    "seed_rk_time_t_copies",
    "start_child_input_preparations", "write_n1_static_npz",
]


if __name__ == "__main__":  # pragma: no cover - controller entry point
    raise SystemExit(main())
