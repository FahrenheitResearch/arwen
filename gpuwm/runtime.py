"""Generic real-case experiment runtime.

The prepare -> static -> ingest -> initialize -> integrate -> output
pipeline for the experiment path, single-domain in this task (the
multi-domain tree/schedule executor is Task 14).  Every operation here
is a pure extraction of the frozen reference-case machinery: identical
operations in identical order on identical values, with the formerly
implicit constants (bundle paths, the case start time, the eta-level
table, the forcing-coverage ceiling, ``sfcp_to_sfcp``, the trace-gas
override, the output identity) replaced by :class:`ExperimentConfig` /
:class:`CaseDataConfig` values.  Operation order and operand identity stay
stable so frozen verification profiles remain byte-inert.

Layering: this module never imports case profile modules; the frozen
reference profile imports *this* module and feeds it the frozen pinned
values.  Grid construction still consumes a declared WPS namelist
for GEOG resolution selection only; Lambert geometry comes directly from
the experiment's ``ProjectionConfig`` and registered nest layout.

Equivalence notes (the argument the A/B gate checks empirically):

- The experiment path runs each per-domain :class:`RunConfig` exactly as
  the loader built it, with ``clock_dt = 0.0`` (retired, architecture
  section C).  Every ``clock_dt`` consumer -- ``lateral_boundary_clock_dt``
  (ingest/lateral_bc.py:214-217), KF/physics ``_model_clock_dt``
  (core/kf.py:22-31, core/physics.py:86-96), the RRTMGP interval
  derivation (core/rrtmgp.py), ``_clock_scaled_diff6_factor``
  (core/dycore.py:841-854), and the npref mirror (:3903) -- resolves
  ``clock_dt <= 0`` to ``cfg.dt``, so it is bit-equivalent to the frozen
  profile's ``clock_dt = dt`` integration transform at
  ``DYNAMICS_SUBSTEPS = 1``.  The frozen profile keeps passing its
  explicit ``integration_cfg`` through this loop unchanged.
- The radiation trace-gas override travels through
  :class:`gpuwm.core.rrtmgp.RRTMGPRadiation`'s ``trace_gas_overrides``
  hook; a frozen profile's declared value is carried without conversion.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from gpuwm.case_data import CaseDataConfig
from gpuwm.config import (DEFAULT_COLUMN_CHUNK, RunConfig,
                          radiation_scheme_ids, soil_layer_count)
from gpuwm.core.grid import make_vertical_coord
from gpuwm.experiment import ExperimentConfig, VerticalConfig
from gpuwm.ingest.grib import cached_era5_forcing
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                     build_state_lateral_boundaries)
from gpuwm.ingest.real import initialize_real
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.static.build import (GeogSelection, build_static,
                                monthly_interp_to_date)
from gpuwm.static.lambert import grids_from_projection_config


MICROPHYSICS_TRANSITION_RECEIPT_NAME = "microphysics-transitions.json"
FEEDBACK_PROVENANCE_RECEIPT_NAME = "feedback-provenance.json"
FEEDBACK_EXPERIMENTAL_WARNING = (
    "WARNING: feedback = 1 is EXPERIMENTAL and is not certified against "
    "stock WRF yet; the certification reference is in progress."
)


# ---------------------------------------------------------------------------
# Prepared-case and run-summary containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedRealCase:
    """Setup-time inputs and live initial state for a real integration.

    ``final_analysis`` is the last forcing-time horizontal analysis
    (the verification profiles score against it);
    ``initial_snow_water_kgm2`` snapshots the t=0 Noah snow water for
    the snow-mask diagnostics.  Both are carried for the profile layer;
    the integration loop itself reads only ``cfg`` / ``grid`` /
    ``static_fields`` / ``initial_result``.
    """

    cfg: RunConfig
    grid: object
    static_fields: dict[str, np.ndarray]
    initial_result: object
    final_analysis: object
    initial_snow_water_kgm2: np.ndarray
    forcing_times: tuple[datetime, ...]
    geog_selection: GeogSelection | None = None


@dataclass(frozen=True)
class RealCaseRunSummary:
    """Results from the config-driven integration surface.

    Contains no oracle or gate results: those comparisons belong
    exclusively to the verification profiles (``gpuwm verify``).
    """

    wrfout_paths: tuple[Path, ...]
    nan_free: bool
    w_max_ms: float
    boundary_w_max_ms: float
    interior_w_max_ms: float
    w_max_boundary_row: int | None
    boundary_zone_blowup: bool
    dynamics_substeps: int
    ysu_nan_guard_fires: int
    surface_forcing_updates: int
    swdown_peak_wm2: float
    swdown_peak_time: datetime
    completed_seconds: float
    #: Final accumulated convective precipitation diagnostics; zero/None
    #: when cu_physics is off.
    rainc_max_mm: float = 0.0
    rainc_max_ji: tuple[int, int] | None = None
    rainc_max_lat: float | None = None
    rainc_max_lon: float | None = None


@dataclass(frozen=True)
class ExperimentRunSummary:
    """Tree-run handoff consumed by the CLI and supervisor."""

    wrfout_paths: tuple[Path, ...]
    completed_seconds: float
    nan_free: bool
    last_checkpoint: Path | None = None
    microphysics_transitions: tuple[Mapping[str, object], ...] = ()
    microphysics_transition_receipt: Path | None = None
    microphysics_transition_receipt_sha256: str | None = None
    feedback_provenance: Mapping[str, object] | None = None
    feedback_provenance_receipt: Path | None = None
    feedback_provenance_receipt_sha256: str | None = None


def feedback_provenance(exp: ExperimentConfig) -> Mapping[str, object] | None:
    """Machine-readable truth label for the experimental feedback tier."""
    if int(exp.feedback) != 1:
        return None
    vertical_mapping = (
        "shared-explicit-eta-ladder-horizontal-only"
        if exp.vertical.eta_levels
        else "shared-legacy-level-count-horizontal-only")
    return {
        "schema": "gpuwm-experimental-feedback-provenance-v1",
        "feedback": "experimental",
        "feedback_value": 1,
        "stock_wrf_certification": "reference-in-progress",
        "stock_wrf_certified": False,
        "restriction": "wrf-v4.6.1-copy_fcn-horizontal",
        "vertical_mapping": vertical_mapping,
    }


def _write_feedback_provenance_receipt(
        outdir: Path, exp: ExperimentConfig, *, resumed: bool
        ) -> tuple[Path | None, str | None, Mapping[str, object] | None]:
    """Atomically stamp feedback truth into the durable run provenance."""
    provenance = feedback_provenance(exp)
    if provenance is None:
        return None, None, None
    payload = dict(provenance)
    payload.update({
        "experiment": exp.name,
        "resumed": bool(resumed),
        "domain_ids": [int(domain.grid_id) for domain in exp.domains],
    })
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n").encode("utf-8")
    path = Path(outdir) / FEEDBACK_PROVENANCE_RECEIPT_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return path, hashlib.sha256(encoded).hexdigest(), payload


def _write_microphysics_transition_receipt(
        outdir: Path, model, exp: ExperimentConfig, *, resumed: bool
        ) -> tuple[Path, str, tuple[Mapping[str, object], ...]]:
    """Atomically publish the executable edge policies and force coverage."""

    transitions = tuple(
        dict(node.coupler.transition_receipt())
        for node in model.nodes_by_grid_id.values()
        if node.coupler is not None)
    for edge in transitions:
        count = int(edge["process_force_count"])
        interval = int(edge["parent_interval_ticks"])
        start = int(edge["process_start_parent_ticks"])
        final = int(edge["final_parent_ticks"])
        first = edge["first_parent_ticks"]
        last = edge["last_parent_ticks"]
        valid_observation = (
            edge.get("current_process_coverage_complete") is True
            and interval > 0 and final >= start
            and (final - start) % interval == 0
            and ((count == 0 and first is None and last is None)
                 or (count > 0
                     and int(first) == start + interval
                     and int(last) == final
                     and int(last) - int(first) == (count - 1) * interval)))
        if not valid_observation:
            raise RuntimeError(
                "microphysics transition force coverage is incomplete or "
                f"internally inconsistent: {edge}")
    payload = {
        "schema": "gpuwm.microphysics-transitions/v1",
        "status": "PASS",
        "experiment": exp.name,
        "experiment_fingerprint": model.experiment_fingerprint,
        "completed_seconds": model.root.clock.elapsed_seconds,
        "resumed_process": bool(resumed),
        "run_id": os.environ.get("GPUWM_RUN_ID"),
        "config_digest": os.environ.get("GPUWM_CONFIG_DIGEST"),
        "transitions": transitions,
    }
    encoded = (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8")
    path = outdir / MICROPHYSICS_TRANSITION_RECEIPT_NAME
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path, hashlib.sha256(encoded).hexdigest(), transitions


# ---------------------------------------------------------------------------
# Static/grid/vertical building blocks
# ---------------------------------------------------------------------------

_STATIC_BUILD_CACHE: dict[tuple, dict] = {}


def _cached_static_build(grid, geog_root, *,
                         selection: GeogSelection | None = None) -> dict:
    """Memoize :func:`build_static` for repeated case preparation.

    Several tests prepare a case independently; the WPS_GEOG tile build
    is pure and deterministic for a given grid, so it is shared.  Cached
    NumPy arrays are locked read-only — every consumer in this module
    derives new arrays rather than mutating the static fields, and the lock
    turns any future in-place write into an immediate ``ValueError`` instead
    of silent cross-test contamination.  Model state is still rebuilt from
    scratch on every call.
    """
    selection = (GeogSelection.fallback(geog_root) if selection is None
                 else selection)
    key = (str(geog_root), selection, grid.map_proj, grid.ref_lat,
           grid.ref_lon, grid.truelat1, grid.truelat2, grid.stand_lon,
           grid.dx, grid.e_we, grid.e_sn)
    hit = _STATIC_BUILD_CACHE.get(key)
    if hit is None:
        hit = build_static(grid, geog_root, selection=selection)
        for value in hit.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        _STATIC_BUILD_CACHE[key] = hit
    return hit


def load_source_orography(path, variable: str) -> np.ndarray:
    """Read the declared source-orography artifact (NetCDF, record 0)."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if variable not in ds.variables:
            raise ValueError(
                f"declared source-orography variable {variable!r} is not "
                f"in {path}; available: {sorted(ds.variables)}")
        return np.asarray(ds.variables[variable][0], dtype=np.float64)


def vertical_coord_for(vertical: VerticalConfig, nz: int):
    """The single-source vertical grid (F1 amendment, G4).

    ``eta_levels``/``p_top``/hybrid selectors come from the experiment's
    one :class:`VerticalConfig` -- never from per-domain RunConfig
    fields.  Two experiments differing in any eta level build two
    distinct grids from config alone (G4 completion).
    """
    if not vertical.eta_levels:
        raise ValueError(
            "the experiment runtime requires explicit eta_levels in "
            "[shared] (VerticalConfig.eta_levels); idealized nz/ztop "
            "scaffolds have no real-data vertical grid")
    eta = np.asarray(vertical.eta_levels, dtype=np.float64)
    return make_vertical_coord(nz, hybrid_opt=vertical.hybrid_opt,
                               etac=vertical.etac, eta_levels=eta)


def single_domain(exp: ExperimentConfig):
    """The one DomainConfig this task's runtime integrates (fail-loud)."""
    if len(exp.domains) != 1:
        raise NotImplementedError(
            f"experiment {exp.name!r} declares {len(exp.domains)} domains; "
            "the Task-2 runtime integrates a single domain -- the "
            "multi-domain tree/schedule executor lands in Task 14.")
    return exp.domains[0]


def experiment_grid(exp: ExperimentConfig, data: CaseDataConfig):
    """Build the case grid directly from projection/domain config."""
    dc = single_domain(exp)
    if dc.grid_id != data.output_domain:
        raise ValueError(
            f"[case_data] output_domain = {data.output_domain} does not "
            f"name the experiment's domain (grid_id = {dc.grid_id}).")
    grids = grids_from_projection_config(exp)
    grid = grids[exp.domains.index(dc)]
    cfg = dc.run
    if (cfg.nx, cfg.ny, cfg.dx, cfg.dy) != (
            grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy):
        raise ValueError(
            "experiment domain grid does not match its resolved config: "
            f"got {(cfg.nx, cfg.ny, cfg.dx, cfg.dy)}, "
            f"expected {(grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy)}")
    return grid


# ---------------------------------------------------------------------------
# Forcing discovery and the coverage-derived run ceiling
# ---------------------------------------------------------------------------

def forcing_snapshots(data: CaseDataConfig, input_catalog=None) -> dict:
    """Decode forcing under one input catalog's valid-time authority.

    The catalog is built here when a caller has not already built it.  Runtime
    decode then merges all declared products together and passes the catalog's
    exact selected/excluded times into the decoder; it never re-derives a
    schedule from records in the raw files.
    """
    if input_catalog is None:
        from gpuwm.ingest.preflight import build_input_catalog

        input_catalog = build_input_catalog(data)

    forcing_hashes = {
        Path(record.path).resolve(): record.sha256
        for record in getattr(input_catalog, "files", ())
        if record.role == "forcing"
    }
    content_sha256 = tuple(
        forcing_hashes.get(Path(path).resolve(), "")
        for path in data.forcing
    )
    if not all(content_sha256):
        content_sha256 = None

    decoded = cached_era5_forcing(
        data.forcing, data.vtable, content_sha256=content_sha256,
        valid_times=input_catalog.valid_times,
        excluded_valid_times=input_catalog.excluded_valid_times,
    )
    by_time: dict[datetime, object] = {}
    for snapshot in decoded.snapshots:
        if snapshot.valid_time in by_time:
            raise ValueError(
                f"duplicate forcing snapshot at {snapshot.valid_time}.")
        by_time[snapshot.valid_time] = snapshot
    actual = tuple(by_time)
    expected = tuple(input_catalog.valid_times)
    if actual != expected:
        raise ValueError(
            "runtime forcing decode did not consume the catalog's exact "
            f"ordered valid-time selection: expected {expected}, got {actual}; "
            "catalog exclusions: "
            f"{tuple(input_catalog.excluded_valid_times)}")
    return by_time


def forcing_schedule(exp: ExperimentConfig, data: CaseDataConfig,
                     available_times) -> tuple[datetime, ...]:
    """Validated forcing valid-time schedule from ``start_time`` onward.

    The run ceiling is DERIVED from validated forcing coverage: a
    ``run_seconds`` beyond the last usable forcing time is rejected here
    with the coverage stated (the frozen case's fixed-constant ceiling
    left the runtime path with this function).  A declared
    ``forcing_interval_s`` policy is enforced against the discovered
    schedule; without it, discovery accepts the file's own spacing.
    """
    times = sorted(available_times)
    if exp.start_time not in times:
        raise ValueError(
            f"forcing has no snapshot at the experiment start_time "
            f"{exp.start_time}; decoded valid times: {times}.")
    usable = tuple(t for t in times if t >= exp.start_time)
    if len(usable) < 2:
        raise ValueError(
            f"forcing declares only {len(usable)} snapshot(s) at/after "
            f"start_time {exp.start_time}; lateral-boundary forcing "
            "requires at least one interval (two valid times).")
    if data.forcing_interval_s is not None:
        for earlier, later in zip(usable, usable[1:]):
            delta = (later - earlier).total_seconds()
            if delta != data.forcing_interval_s:
                raise ValueError(
                    f"declared forcing_interval_s = "
                    f"{data.forcing_interval_s:g} but the decoded "
                    f"schedule steps {earlier} -> {later} "
                    f"({delta:g} s).")
    coverage = (usable[-1] - usable[0]).total_seconds()
    if exp.run_seconds > coverage:
        raise ValueError(
            f"run_seconds = {exp.run_seconds:g} exceeds the validated "
            f"forcing coverage of {coverage:g} s ({usable[0]} .. "
            f"{usable[-1]}); shorten the run or declare more forcing.")
    return usable


# ---------------------------------------------------------------------------
# Prepare: static -> ingest -> initialize (the extracted case machinery)
# ---------------------------------------------------------------------------

def prepare_real_case(cfg: RunConfig, *, grid, geog_root,
                      source_orography_path=None,
                      source_orography_variable=None,
                      vertical: VerticalConfig, sfcp_to_sfcp: bool,
                      snapshot_for, forcing_times, start_time: datetime,
                      trace_gas_overrides=None,
                      geog_selection: GeogSelection | None = None,
                      forcing_catalog=None,
                      scratch_arena=None,
                      dycore_state_workspace=None,
                      radiation_column_chunk=DEFAULT_COLUMN_CHUNK,
                      ) -> PreparedRealCase:
    """Run the real-case setup pipeline for one domain.

    Extraction of the frozen reference preparation: identical operations in
    identical order, parameterized by the formerly implicit values.
    ``snapshot_for(valid_time)`` supplies the decoded forcing snapshot
    for each entry of ``forcing_times`` (the first entry must be
    ``start_time`` -- it becomes the live initial state).
    ``trace_gas_overrides`` feeds the RRTMGP trace-gas policy hook;
    ``None`` keeps the scheme's frozen default composition.
    """
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.physics import initialize_physics

    times = tuple(forcing_times)
    if not times or times[0] != start_time:
        raise ValueError(
            f"forcing_times must begin at start_time {start_time}; got "
            f"{times[:1]}.")
    if (cfg.nx, cfg.ny) != (grid.e_we - 1, grid.e_sn - 1) or (
            cfg.dx, cfg.dy) != (grid.dx, grid.dy):
        raise ValueError(
            "RunConfig grid does not match the supplied Lambert grid: "
            f"got {(cfg.nx, cfg.ny, cfg.dx, cfg.dy)}, expected "
            f"{(grid.e_we - 1, grid.e_sn - 1, grid.dx, grid.dy)}")
    if (trace_gas_overrides is not None
            and radiation_scheme_ids(cfg) != (4, 4)):
        raise ValueError(
            "trace-gas overrides require ra_physics = 4 (effective "
            "LW/SW radiation = 4/4) "
            f"(RTE+RRTMGP), got {radiation_scheme_ids(cfg)}")

    if (source_orography_path is None) != (source_orography_variable is None):
        raise ValueError(
            "source_orography_path and source_orography_variable must be "
            "provided together")
    if (source_orography_path is not None and forcing_catalog is not None
            and "SOILGEO" in getattr(forcing_catalog, "inventory", ())):
        raise ValueError(
            "source-orography conflict: declared source_orography "
            f"{source_orography_path} variable={source_orography_variable} "
            "and forcing catalog SOILGEO via era5_z_invariant are both "
            "present; declare exactly one source")
    geog_selection = (GeogSelection.fallback(geog_root)
                      if geog_selection is None else geog_selection)
    static = _cached_static_build(
        grid, geog_root, selection=geog_selection)
    landuse_attrs = geog_selection.landuse_global_attrs()
    source_orography = None
    if source_orography_path is not None:
        source_orography = load_source_orography(
            source_orography_path, source_orography_variable)
    results = []
    horizontal = []
    for valid_time in times:
        source = snapshot_for(valid_time)
        # Metgrid classifies masked-field TARGET cells by the model
        # (geogrid) landmask, not by the nearest source LSM; the source-side
        # usable-point decision stays with the source LANDSEA inside the
        # masked operators.  Passing the static LANDMASK reproduces WPS and
        # keeps soil, skin, and physics on one land/water surface.
        met = interpolate_era5_to_lambert(
            source, grid, source_orography_catalog=forcing_catalog,
            target_landmask=np.asarray(static["LANDMASK"]) >= 0.5)
        coord = vertical_coord_for(vertical, cfg.nz)
        init_kwargs = dict(
            source_orography=source_orography, p_top=vertical.p_top,
            sfcp_to_sfcp=sfcp_to_sfcp)
        if scratch_arena is not None:
            init_kwargs["scratch_arena"] = scratch_arena
        if dycore_state_workspace is not None:
            init_kwargs["dycore_state_workspace"] = dycore_state_workspace
        result = initialize_real(
            met, cfg, coord, static["HGT_M"], **init_kwargs)
        f, e = grid.coriolis_m()
        # SINALPHA/COSALPHA (geo_em conventions): WRF's coriolis applies
        # the rotation terms unconditionally (module_em.F:761-769).
        sina, cosa = grid.rotation_m()
        result.state.set_map_coriolis(
            grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(), f, e,
            sina=sina, cosa=cosa)
        horizontal.append(met)
        results.append(result)
    boundaries = build_state_lateral_boundaries(
        [result.state for result in results], times,
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(results[0].state, boundaries)

    soil_fields = dict(horizontal[0].fields)
    # No lake skin override: with metgrid's masked=both SKINTEMP chain and
    # static-landmask target classification, lake cells already carry the
    # water-source skin value, and real.exe (no TAVGSFC) keeps exactly that
    # SKINTEMP wherever SST has no valid support
    # (module_initialize_real.F:2844-2866, :2898-2906).  The router forwards
    # this exact argument list to preprocess_noah_soil for Noah-geometry
    # schemes, so their soil state is unchanged by the LSM dispatch seam.
    soil = preprocess_land_surface_soil(
        soil_fields, sf_surface_physics=int(cfg.sf_surface_physics),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["TMN"],
        landmask=static["LANDMASK"],
        terrain=static["HGT_M"] if source_orography is not None else None,
        source_orography=source_orography)
    # WRF interpolates GREENFRAC/LAI to the run date
    # (module_initialize_real.F:1322-1335, mid-month anchors); shdmin/
    # shdmax stay the monthly extrema (:1348-1351).  With the supported
    # usemonalb=false path, landuse_init overwrites ALBEDO12M from the table.
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], start_time)
    lai = monthly_interp_to_date(static["LAI12M"], start_time)
    state = results[0].state
    # initialize_real loads prognostics but does not launch the EOS kernel.
    # Diagnose the time-zero atmosphere before the first RRTMGP call.
    update_diagnostics(state, cfg.hypsometric_opt)
    lat, lon = grid.latlon_mass()
    radiation = None
    if radiation_scheme_ids(cfg) == (4, 4):
        from gpuwm.physics_compat import (
            RRTMG_VARIANT_LEGACY, rrtmg_variant)
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            # Legacy RRTMG selected: construct the exact WRF v4.6.1 port
            # (root ozone routing; its constructor fails closed if the
            # assets/kernels are unavailable).  Never substitute
            # RTE+RRTMGP.  ghg_input=0 evaluates WRF's analytic
            # year-formula trace gases, so an explicit co2_vmr override
            # cannot be honored and must not be dropped silently.
            if trace_gas_overrides:
                raise ValueError(
                    "ra_rrtmg_variant='rrtmg_legacy' runs WRF's "
                    "ghg_input=0 analytic year-formula trace gases; the "
                    f"explicit trace-gas override {trace_gas_overrides!r} "
                    "(case co2_vmr) cannot be honored by the legacy port "
                    "-- remove it or select ra_rrtmg_variant="
                    "'rte-rrtmgp'")
            from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
            radiation = RRTMGLegacyRadiation(
                start_time, lat, lon, p_top=vertical.p_top)
        else:
            # Construct the experiment-configured adapter for both
            # trace-gas policies: an explicit override and the dated
            # default selected by None.  Otherwise initialize_physics
            # would construct its own adapter with the class-default
            # column chunk.
            from gpuwm.core.rrtmgp import RRTMGPRadiation
            radiation = RRTMGPRadiation(
                start_time, lat, lon,
                trace_gas_overrides=trace_gas_overrides,
                column_chunk=radiation_column_chunk)
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=start_time,
        cen_lat=float(getattr(grid, "cen_lat", np.mean(lat))),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]))
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        radiation=radiation,
        radiation_start_time=start_time, radiation_latitude=lat,
        radiation_longitude=lon)
    import cupy as cp
    driver.fields["snoalb"][...] = cp.asarray(
        static["SNOALB"] / 100.0, dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)

    # Seed time-zero surface diagnostics from the source analysis.  The
    # first model step replaces them through SFCLAY/Noah/YSU in WRF
    # ordering.
    met0 = horizontal[0].fields
    driver.fields["psfc"][...] = cp.asarray(
        results[0].surface_pressure, dtype=cp.float32)
    driver.fields["t2"][...] = met0["T2"]
    driver.fields["q2"][...] = cp.asarray(
        results[0].surface_qv, dtype=cp.float32)
    driver.fields["th2"][...] = (driver.fields["t2"]
                                  * (cp.float32(100000.0)
                                     / driver.fields["psfc"])
                                  ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (met0["U10"][:, :-1]
                                        + met0["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (met0["V10"][:-1]
                                        + met0["V10"][1:])
    return PreparedRealCase(
        cfg=cfg, grid=grid, static_fields=static,
        initial_result=results[0], final_analysis=horizontal[-1],
        initial_snow_water_kgm2=np.array(
            soil.snow_water, dtype=np.float64, copy=True),
        forcing_times=times, geog_selection=geog_selection)


def prepare_root_experiment_case(exp: ExperimentConfig,
                                 data: CaseDataConfig, *,
                                 input_catalog=None,
                                 forcing_by_time=None,
                                 scratch_arena=None,
                                 dycore_state_workspace=None
                                 ) -> PreparedRealCase:
    """Prepare the root domain of a single- or multi-domain experiment."""
    dc = exp.root
    cfg = dc.run
    if len(exp.domains) == 1:
        grid = experiment_grid(exp, data)
    else:
        from gpuwm.static.lambert import grids_from_projection_config
        grid = grids_from_projection_config(exp)[0]
    geog_selection = GeogSelection.from_case_data(
        data, domain_id=dc.grid_id)
    from gpuwm.ingest.preflight import build_input_catalog

    catalog = (build_input_catalog(data) if input_catalog is None
               else input_catalog)
    snapshots = (forcing_snapshots(data, catalog)
                 if forcing_by_time is None else forcing_by_time)
    decoded_times = tuple(snapshots)
    if decoded_times != tuple(catalog.valid_times):
        raise ValueError(
            "prepared forcing snapshots do not match the input catalog's "
            f"ordered valid times: expected {catalog.valid_times}, "
            f"got {decoded_times}; catalog exclusions: "
            f"{catalog.excluded_valid_times}")
    times = forcing_schedule(exp, data, snapshots)

    def snapshot_for(valid_time):
        try:
            return snapshots[valid_time]
        except KeyError as exc:
            raise ValueError(
                f"forcing has no snapshot at {valid_time!s}") from exc

    declared_orography = data.source_orography
    return prepare_real_case(
        cfg, grid=grid, geog_root=data.geog_root,
        source_orography_path=(declared_orography.path
                               if declared_orography is not None else None),
        source_orography_variable=(declared_orography.variable
                                   if declared_orography is not None else None),
        vertical=exp.vertical, sfcp_to_sfcp=data.sfcp_to_sfcp,
        snapshot_for=snapshot_for, forcing_times=times,
        start_time=exp.start_time,
        trace_gas_overrides=({"co2": data.co2_vmr}
                             if data.co2_vmr is not None else None),
        geog_selection=geog_selection, forcing_catalog=catalog,
        scratch_arena=scratch_arena,
        dycore_state_workspace=dycore_state_workspace,
        radiation_column_chunk=exp.column_chunk)


def prepare_experiment_case(exp: ExperimentConfig,
                            data: CaseDataConfig, *, input_catalog=None,
                            forcing_by_time=None) -> PreparedRealCase:
    """Assemble :func:`prepare_real_case` inputs from the config pair."""
    single_domain(exp)  # Preserve Task-2's fail-loud single-domain surface.
    return prepare_root_experiment_case(
        exp, data, input_catalog=input_catalog,
        forcing_by_time=forcing_by_time)


def prepare_child_case(initialized, child_dc, *, exp: ExperimentConfig,
                       data: CaseDataConfig, forcing_times,
                       radiation_workspace=None,
                       radiation_parent=None) -> PreparedRealCase:
    """Attach one child's per-domain physics driver after T12 init.

    T12 returns the WRF-order atmospheric/static/Noah setup products.  This
    helper performs the same surface/radiation initialization used by the
    root, including the unblended-terrain Noah state, date-interpolated GEOG
    climatologies, and time-zero diagnostics.  It intentionally constructs a
    distinct RRTMGP adapter for the child; Task 14 attaches the one allocated
    common chunk workspace after all drivers exist.

    ``radiation_parent`` is required when ``ra_rrtmg_variant =
    "rrtmg_legacy"`` is selected: the parent domain's
    ``RRTMGLegacyRadiation``, whose retained o33d field this child
    interpolates (WRF computes o3rad on the root only and hands nests the
    parent-interpolated field; a per-nest climatology evaluation would
    diverge from WRF).  Constructing a legacy child without it fails
    closed.
    """
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.rrtmgp import RRTMGPRadiation

    if (initialized.real is None or initialized.static_fields is None
            or initialized.horizontal is None or initialized.soil is None):
        raise ValueError("real experiment children require T12 real-data init")
    dc = exp.domain(child_dc.grid_id)
    if dc is not child_dc:
        raise ValueError("child_dc must be the experiment's DomainConfig")
    cfg = dc.run
    state = initialized.state
    static = initialized.static_fields
    soil = initialized.soil
    met0 = initialized.horizontal.fields
    real = initialized.real

    update_diagnostics(state, cfg.hypsometric_opt)
    domain_start_time = exp.domain_start_time(dc.grid_id)
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"],
                                             domain_start_time)
    lai = monthly_interp_to_date(static["LAI12M"], domain_start_time)
    lat, lon = initialized.grid.latlon_mass()
    radiation = None
    if radiation_scheme_ids(cfg) == (4, 4):
        from gpuwm.physics_compat import (
            RRTMG_VARIANT_LEGACY, rrtmg_variant)
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            # Legacy RRTMG child: the exact WRF v4.6.1 port with WRF's
            # root-compute + parent->child ozone routing.  The parent
            # adapter is mandatory -- a per-nest climatology evaluation
            # would diverge from WRF on d02+ (never fall back silently).
            if radiation_parent is None:
                raise ValueError(
                    "ra_rrtmg_variant='rrtmg_legacy' on child domain "
                    f"grid_id={dc.grid_id} requires radiation_parent= "
                    "(the parent domain's RRTMGLegacyRadiation): WRF "
                    "computes o3rad on the root domain only and hands "
                    "nests the parent-interpolated field")
            if data.co2_vmr is not None:
                raise ValueError(
                    "ra_rrtmg_variant='rrtmg_legacy' runs WRF's "
                    "ghg_input=0 analytic year-formula trace gases; the "
                    f"explicit case co2_vmr={data.co2_vmr!r} cannot be "
                    "honored by the legacy port -- remove it or select "
                    "ra_rrtmg_variant='rte-rrtmgp'")
            from gpuwm.core.nest_interp import register_nest
            from gpuwm.core.rrtmg_legacy import (ParentOzoneProvider,
                                                 RRTMGLegacyRadiation)
            parent_dc = exp.domain(dc.parent_id)
            registration = register_nest(
                nri=dc.parent_grid_ratio, nrj=dc.parent_grid_ratio,
                i_parent_start=dc.i_parent_start,
                j_parent_start=dc.j_parent_start,
                child_nx=cfg.nx, child_ny=cfg.ny,
                parent_nx=parent_dc.run.nx, parent_ny=parent_dc.run.ny,
                stagger="", wrapper="interp")
            radiation = RRTMGLegacyRadiation(
                exp.start_time, lat, lon,
                p_top=float(state.p_top),
                ozone_parent=ParentOzoneProvider(radiation_parent,
                                                 registration))
        else:
            overrides = ({"co2": data.co2_vmr}
                         if data.co2_vmr is not None else None)
            radiation = RRTMGPRadiation(
                exp.start_time, lat, lon, trace_gas_overrides=overrides,
                column_chunk=exp.column_chunk)
            if radiation_workspace is not None:
                radiation.column_chunk = radiation_workspace.column_chunk
                radiation.chunk_workspace = radiation_workspace
    geog_selection = GeogSelection.from_case_data(
        data, domain_id=dc.grid_id)
    landuse_attrs = geog_selection.landuse_global_attrs()
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=domain_start_time,
        cen_lat=float(getattr(initialized.grid, "cen_lat", np.mean(lat))),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]))
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        radiation=radiation, radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon)
    driver.fields["snoalb"][...] = cp.asarray(
        static["SNOALB"] / 100.0, dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    driver.fields["psfc"][...] = cp.asarray(
        real.surface_pressure, dtype=cp.float32)
    driver.fields["t2"][...] = met0["T2"]
    driver.fields["q2"][...] = cp.asarray(real.surface_qv, dtype=cp.float32)
    driver.fields["th2"][...] = (
        driver.fields["t2"]
        * (cp.float32(100000.0) / driver.fields["psfc"])
        ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (met0["U10"][:, :-1]
                                         + met0["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (met0["V10"][:-1]
                                         + met0["V10"][1:])
    return PreparedRealCase(
        cfg=cfg, grid=initialized.grid, static_fields=dict(static),
        initial_result=real, final_analysis=initialized.horizontal,
        initial_snow_water_kgm2=np.array(
            soil.snow_water, dtype=np.float64, copy=True),
        forcing_times=tuple(forcing_times),
        geog_selection=geog_selection)


# ---------------------------------------------------------------------------
# Run schedule (whole-step counts) and output calendars
# ---------------------------------------------------------------------------

def whole_step_count(duration: float, dt: float, name: str) -> int:
    """Return an exact whole-step count or reject an ambiguous schedule."""
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {duration}")
    steps = int(round(duration / dt))
    tolerance = max(1.0e-9, abs(duration) * 1.0e-12)
    if steps < 1 or not np.isclose(
            steps * dt, duration, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"{name}={duration} must be an integer multiple of dt={dt}")
    return steps


def configured_run_schedule(
        cfg: RunConfig, *, run_seconds: float | None = None,
        output_interval_s: float | None = None) -> tuple[int, int]:
    """Validate and return ``(outer_steps, output_outer_steps)``.

    The run-length ceiling is NOT here: it derives from validated
    forcing coverage (:func:`forcing_schedule`) on the experiment path,
    and the frozen case profile applies its own pinned ceiling before
    delegating.
    """
    if not np.isfinite(cfg.dt) or cfg.dt <= 0.0:
        raise ValueError(f"dt must be finite and > 0, got {cfg.dt}")
    run_seconds = cfg.run_seconds if run_seconds is None else run_seconds
    output_interval_s = (cfg.output_interval_s
                         if output_interval_s is None
                         else output_interval_s)
    return (
        whole_step_count(run_seconds, cfg.dt, "run_seconds"),
        whole_step_count(
            output_interval_s, cfg.dt, "output_interval_s"),
    )


def restart_outer_steps(
        cfg: RunConfig, *, restart_interval_s: float | None = None
        ) -> int | None:
    """Restart-write cadence in outer steps; ``None`` when disabled."""
    restart_interval_s = (cfg.restart_interval_s
                          if restart_interval_s is None
                          else restart_interval_s)
    if restart_interval_s <= 0.0:
        return None
    return whole_step_count(
        restart_interval_s, cfg.dt, "restart_interval_s")


def refl_10cm_due(outer_step: int, substep: int,
                  output_outer_steps: int,
                  dynamics_substeps: int) -> bool:
    """True only for the microphysics call immediately before an output.

    The final internal step owns the outer-step history frame.  Keeping this
    as a pure predicate makes the calendar wiring CPU-testable even if a
    future configuration restores more than one dynamics substep.
    """
    return ((outer_step + 1) % output_outer_steps == 0
            and substep + 1 == dynamics_substeps)


# ---------------------------------------------------------------------------
# Output identity
# ---------------------------------------------------------------------------

def _global_wrf_attrs(
        grid, start_time: datetime,
        geog_selection: GeogSelection | None = None, *, domain=None,
        coord=None, feedback=None) -> dict[str, object]:
    from gpuwm.io.wrfout import wrf_global_attrs

    landuse_attrs = (None if geog_selection is None
                     else geog_selection.landuse_global_attrs())
    identity = {}
    if domain is not None:
        run = getattr(domain, "run", domain)
        identity.update(
            grid_id=int(getattr(domain, "grid_id", run.grid_id)),
            parent_id=int(getattr(domain, "parent_id", 0)),
            i_parent_start=int(getattr(domain, "i_parent_start", 1)),
            j_parent_start=int(getattr(domain, "j_parent_start", 1)),
            parent_grid_ratio=int(
                getattr(domain, "parent_grid_ratio", 1)),
            dt=float(run.dt))
    if coord is not None:
        identity.update(hybrid_opt=int(coord.hybrid_opt),
                        etac=float(coord.etac))
    attrs = wrf_global_attrs(
        grid, start_time, landuse_attrs=landuse_attrs, **identity)
    if feedback is not None:
        attrs.update(
            GPUWM_FEEDBACK=str(feedback["feedback"]),
            GPUWM_FEEDBACK_VALUE=np.int32(feedback["feedback_value"]),
            GPUWM_FEEDBACK_STOCK_WRF_CERTIFIED=np.int32(0),
            GPUWM_FEEDBACK_CERTIFICATION=str(
                feedback["stock_wrf_certification"]))
    return attrs


def _metadata_frame(grid, static: dict) -> dict[str, np.ndarray]:
    lat, lon = grid.latlon_mass()
    lat_u, lon_u = grid.latlon_u()
    lat_v, lon_v = grid.latlon_v()
    f, e = grid.coriolis_m()
    sina, cosa = grid.rotation_m()
    return {
        "XLAT": lat, "XLONG": lon, "XLAT_U": lat_u, "XLONG_U": lon_u,
        "XLAT_V": lat_v, "XLONG_V": lon_v,
        "MAPFAC_M": grid.mapfac_m(), "MAPFAC_U": grid.mapfac_u(),
        "MAPFAC_V": grid.mapfac_v(), "F": f, "E": e,
        "SINALPHA": sina, "COSALPHA": cosa, "HGT": static["HGT_M"],
        "LANDMASK": static["LANDMASK"], "LU_INDEX": static["LU_INDEX"],
    }


def write_case_output(prepared, output_dir: Path, valid_time: datetime, *,
                      start_time: datetime, title: str, domain_id: int = 1,
                      expect_refl_10cm: bool = True,
                      feedback=None) -> Path:
    import cupy as cp
    from gpuwm.io.wrfout import (WrfoutWriter, state_frame,
                                 wrfout_filename)

    state = prepared.initial_result.state
    # initialize_real and every completed dycore step leave p/al/alt current.
    # Re-diagnosing here would make an observational output operation mutate
    # the next step's initial state, so case output is deliberately read-only.
    frame = state_frame(state, include_diagnostic_pressure=True)
    frame.update(_metadata_frame(prepared.grid, prepared.static_fields))
    rainnc = state.physics.microphysics.rainnc
    frame["RAINNC"] = cp.asnumpy(rainnc)
    if (expect_refl_10cm and prepared.cfg.mp_physics in (1, 6, 8, 10, 18)
            and state.qv is not None):
        # WRF do_radar_ref=1 equivalent: consume the field computed inside
        # the output-due microphysics call from its prepared p/post-call T.
        # Missing or double-consumed handoffs are cadence bugs and fail loud.
        from gpuwm.core.refl import consume_refl_10cm
        frame["REFL_10CM"] = cp.asnumpy(consume_refl_10cm(state))
    path = output_dir / wrfout_filename(valid_time, domain_id)
    with WrfoutWriter(
            path, nx=prepared.cfg.nx, ny=prepared.cfg.ny, nz=prepared.cfg.nz,
            dx=prepared.cfg.dx, dy=prepared.cfg.dy,
            title=title,
            global_attrs=_global_wrf_attrs(prepared.grid,
                                           start_time,
                                           getattr(prepared,
                                                   "geog_selection",
                                                   None),
                                           domain=prepared.cfg,
                                           coord=prepared.initial_result.coord,
                                           feedback=feedback),
            field_schema=frame,
            # The soil axis is the selected LSM's geometry.  Omitting this
            # took WrfoutWriter's old literal-4 default, so a nine-layer
            # scheme would have declared soil_layers_stag=4 here.
            soil_layers=soil_layer_count(prepared.cfg),
            ) as writer:
        writer.write_frame(valid_time.strftime("%Y-%m-%d_%H:%M:%S"), frame)
    return path


# ---------------------------------------------------------------------------
# Integrate: the extracted outer-step loop
# ---------------------------------------------------------------------------


def _preparation_progress(progress_callback, phase: str) -> None:
    reporter = getattr(progress_callback, "preparing", None)
    if reporter is not None:
        reporter(phase)


def apply_single_domain_pbl_cadence(physics, cfg) -> None:
    """Single-domain-loop PBL cadence: override ONLY at configured bldt=0.

    With ``cfg.bldt == 0`` every internal dynamics step runs the
    surface/PBL stack (WRF's bldt=0 semantics), and under the retired
    compatibility integrator the INTERNAL step is the authoritative
    interval, so the driver's setup values are overwritten with
    ``bldt_seconds = cfg.dt`` / ``stepbl = 1``.  A positive configured
    bldt keeps PhysicsDriver's WRF STEPBL calendar
    (``max(nint(bldt*60/dt), 1)``, gpuwm/core/physics.py) untouched --
    the previous unconditional override silently forced every-step PBL
    regardless of namelist bldt, a latent trap that was inert only
    because the campaign lineage runs bldt=0.
    """
    if cfg.bldt == 0.0:
        physics.bldt_seconds = cfg.dt
        physics.stepbl = 1


def integrate_prepared_case(
        output_dir, prepared, *, start_time: datetime, output_title: str,
        domain_id: int = 1, integration_cfg: RunConfig | None = None,
        restart_path=None, run_seconds: float | None = None,
        history_interval_s: float | None = None,
        restart_interval_s: float | None = None, progress_callback=None,
        health_debug: bool = False,
        feedback=None) -> RealCaseRunSummary:
    """Integrate a prepared real case and write its configured outputs.

    Extraction of the frozen reference integration loop: intentionally free
    of oracle comparisons and plotting so the normal run surface honors
    short forecasts and arbitrary configured output cadence.

    ``integration_cfg`` is the frozen profile's compatibility hook (its
    retired substep transform sets ``clock_dt = dt``); the experiment
    path leaves it ``None`` and integrates ``prepared.cfg`` as loaded
    (``clock_dt = 0.0``, which every consumer resolves to ``dt`` -- see
    the module docstring's equivalence notes).

    ``restart_path`` resumes a run: the deterministic preparation runs
    unchanged (rebuilding the setup and the resident LBC device tables),
    then :func:`gpuwm.io.restart.restore_restart` overwrites the full
    cross-step state and restores the clock, and the loop continues from
    the restored outer step through ``run_seconds`` (the TOTAL forecast
    length from ``start_time``, exactly as for an uninterrupted run).
    The initial start-time wrfout is not rewritten on resume.
    ``run_seconds``/``history_interval_s``/``restart_interval_s`` are the
    experiment/domain timing authority.  Legacy callers omit them and use
    the compatibility copies on ``cfg``.
    """
    import cupy as cp
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.dycore import stability_report, step
    from gpuwm.io.restart import (restart_filename, restore_restart,
                                  write_restart)
    from gpuwm.supervisor import validate_manifest_checkpoint

    cfg = prepared.cfg
    run_seconds = cfg.run_seconds if run_seconds is None else run_seconds
    history_interval_s = (cfg.output_interval_s
                          if history_interval_s is None
                          else history_interval_s)
    restart_interval_s = (cfg.restart_interval_s
                          if restart_interval_s is None
                          else restart_interval_s)
    outer_steps, output_outer_steps = configured_run_schedule(
        cfg, run_seconds=run_seconds,
        output_interval_s=history_interval_s)
    restart_write_steps = restart_outer_steps(
        cfg, restart_interval_s=restart_interval_s)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    integration_cfg = cfg if integration_cfg is None else integration_cfg
    state = prepared.initial_result.state
    _preparation_progress(progress_callback, "initialize-health-validator")
    health = StateHealthValidator(state)
    phase_hook_supported = "phase_observer" in inspect.signature(
        step).parameters
    # bldt=0 follows each internal dynamics step; radiation follows the
    # configured WRF STEPRA calendar on those internal steps.  A positive
    # configured bldt keeps the driver's WRF STEPBL calendar (see helper).
    apply_single_domain_pbl_cadence(state.physics, integration_cfg)
    outputs = []
    nan_free = True
    w_max = 0.0
    w_max_boundary_row = None
    boundary_w_max = 0.0
    interior_w_max = 0.0
    surface_forcing_updates = 0
    swdown_peak = -np.inf
    swdown_peak_time = start_time
    start_outer_step = 0
    last_checkpoint = None
    if restart_path is None:
        # No microphysics call precedes the cold-start frame, so there is no
        # WRF-arranged post-call reflectivity field to consume.
        _preparation_progress(progress_callback, "cold-start-wrfout")
        outputs.append(write_case_output(
            prepared, output_dir, start_time, start_time=start_time,
            title=output_title, domain_id=domain_id,
            expect_refl_10cm=False, feedback=feedback))
        # WRF resets the nwp_diagnostics running maxima each history
        # interval (module_diag_nwp.F:246-269); gpuwm's ratified placement
        # is immediately after the frame is durable.
        from gpuwm.core.uh_diag import reset_up_heli_max
        reset_up_heli_max(state)
    else:
        _preparation_progress(progress_callback, "validate-checkpoint")
        last_checkpoint = validate_manifest_checkpoint(restart_path)
        _preparation_progress(progress_callback, "restore-checkpoint")
        info = restore_restart(last_checkpoint, state, cfg)
        start_outer_step = whole_step_count(
            info.elapsed_seconds, cfg.dt, "restart elapsed_seconds")
        if start_outer_step >= outer_steps:
            raise ValueError(
                f"restart file is already at {info.elapsed_seconds} s; "
                f"nothing to integrate before run_seconds="
                f"{run_seconds}")
        trackers = info.run_trackers
        if trackers is not None:
            # Summary continuity: an interrupted-and-resumed run reports
            # the same run bookkeeping as an uninterrupted one.
            nan_free = bool(trackers["nan_free"])
            w_max = float(trackers["w_max_ms"])
            w_max_boundary_row = trackers["w_max_boundary_row"]
            boundary_w_max = float(trackers["boundary_w_max_ms"])
            interior_w_max = float(trackers["interior_w_max_ms"])
            swdown_peak = float(trackers["swdown_peak_wm2"])
            swdown_peak_time = datetime.fromisoformat(
                trackers["swdown_peak_time"])
        surface_forcing_updates = state.physics.call_counts["radiation"]
    _preparation_progress(progress_callback, "initial-health-gate")
    health.require_healthy(phase="initialized-or-restored")
    if progress_callback is not None:
        progress_callback(
            model_elapsed_seconds=float(state.elapsed_seconds),
            outer_step=start_outer_step,
            last_durable_wrfout=(outputs[-1] if outputs else None),
            last_checkpoint=last_checkpoint, phase="initialized-or-restored",
            step_wall_seconds=0.0)
    dynamics_substeps = int(round(cfg.dt / integration_cfg.dt))
    for outer_step in range(start_outer_step, outer_steps):
        outer_started = time.perf_counter()
        forcing_time = start_time + timedelta(seconds=outer_step * cfg.dt)
        for substep in range(dynamics_substeps):
            refl_due = (cfg.mp_physics in (1, 6, 8, 10, 18)
                        and refl_10cm_due(
                            outer_step, substep, output_outer_steps,
                            dynamics_substeps))
            phase = f"outer-{outer_step + 1}.substep-{substep + 1}"
            if health_debug and not phase_hook_supported:
                health.require_healthy(phase=phase + ".pre-step")
            step_kwargs = {"refl_10cm_due": refl_due}
            if health_debug and phase_hook_supported:
                step_kwargs["phase_observer"] = health.phase_observer
            step(state, integration_cfg, **step_kwargs)
            # Validator cadence (controller amendment, 2026-07-16): the
            # measured full-validation cost is 5.00% of step wall vs the
            # plan's <=2% gate, so the pre-registered remedy applies --
            # every 4th step (~1.25%) PLUS mandatory instants: the final
            # step, every output-due step, and every restart instant.
            # health_debug forces every step (attribution mode).
            step_index = outer_step * dynamics_substeps + substep
            mandatory = (
                outer_step == outer_steps - 1 and
                substep == dynamics_substeps - 1
            ) or refl_due or (
                restart_write_steps
                and (outer_step + 1) % restart_write_steps == 0
                and substep == dynamics_substeps - 1)
            if health_debug or mandatory or step_index % 4 == 0:
                health.require_healthy(phase=phase + ".post-step")
            width = cfg.spec_bdy_width
            report = stability_report(
                state, integration_cfg, boundary_width=width)
            nan_free = nan_free and not report["nan"]
            step_w_max = float(report["w_max"])
            if step_w_max > w_max:
                max_index = np.unravel_index(
                    report["w_argmax"], state.w.shape)
                _k, j, i = (int(index) for index in max_index)
                distance = min(j, cfg.ny - 1 - j, i, cfg.nx - 1 - i)
                # WRF/kernel boundary distance: d=0 is specified; d=1,2,3
                # are relaxation rows 1,2,3 for this case.
                w_max_boundary_row = (distance if distance < width else None)
                w_max = step_w_max
            boundary_w_max = max(
                boundary_w_max, report["boundary_w_max"])
            interior_w_max = max(
                interior_w_max, report["interior_w_max"])
            if not nan_free:
                raise RuntimeError(
                    "real-case integration produced a non-finite state at "
                    f"dynamics substep "
                    f"{dynamics_substeps * outer_step + substep + 1}")
        surface_forcing_updates = state.physics.call_counts["radiation"]
        step_swdown_peak = float(cp.max(state.physics.fields["swdown"]))
        if step_swdown_peak > swdown_peak:
            swdown_peak = step_swdown_peak
            swdown_peak_time = forcing_time
        if (outer_step + 1) % output_outer_steps == 0:
            valid = start_time + timedelta(seconds=(outer_step + 1) * cfg.dt)
            outputs.append(write_case_output(
                prepared, output_dir, valid, start_time=start_time,
                title=output_title, domain_id=domain_id,
                feedback=feedback))
            # History-interval reset of the UP_HELI_MAX window (the frame
            # above snapshotted the accumulator synchronously).
            from gpuwm.core.uh_diag import reset_up_heli_max
            reset_up_heli_max(state)
        if (restart_write_steps is not None
                and (outer_step + 1) % restart_write_steps == 0):
            valid = start_time + timedelta(seconds=(outer_step + 1) * cfg.dt)
            last_checkpoint = write_restart(
                output_dir / restart_filename(valid, f"d{domain_id:02d}"),
                state, cfg,
                run_trackers={
                    "nan_free": nan_free,
                    "w_max_ms": float(w_max),
                    "w_max_boundary_row": w_max_boundary_row,
                    "boundary_w_max_ms": float(boundary_w_max),
                    "interior_w_max_ms": float(interior_w_max),
                    "swdown_peak_wm2": float(swdown_peak),
                    "swdown_peak_time": swdown_peak_time.isoformat(),
                })
        # The state gate completed after the final internal step.  Publish
        # progress only after any due wrfout/checkpoint is durable, so a
        # heartbeat can never advertise unguarded or unpublished work.
        if progress_callback is not None:
            progress_callback(
                model_elapsed_seconds=float(state.elapsed_seconds),
                outer_step=outer_step + 1,
                last_durable_wrfout=(outputs[-1] if outputs else None),
                last_checkpoint=last_checkpoint, phase="post-d01-sync",
                step_wall_seconds=time.perf_counter() - outer_started)
    cp.cuda.runtime.deviceSynchronize()

    rainc_max = 0.0
    rainc_ji = None
    rainc_lat = None
    rainc_lon = None
    if state.physics.rainc is not None:
        rainc_host = cp.asnumpy(state.physics.rainc)
        j, i = np.unravel_index(int(np.argmax(rainc_host)),
                                rainc_host.shape)
        rainc_max = float(rainc_host[j, i])
        rainc_ji = (int(j), int(i))
        lat, lon = prepared.grid.latlon_mass()
        rainc_lat = float(lat[j, i])
        rainc_lon = float(lon[j, i])
    return RealCaseRunSummary(
        wrfout_paths=tuple(outputs), nan_free=nan_free,
        w_max_ms=w_max, boundary_w_max_ms=boundary_w_max,
        interior_w_max_ms=interior_w_max,
        w_max_boundary_row=w_max_boundary_row,
        boundary_zone_blowup=(not np.isfinite(boundary_w_max)
                              or boundary_w_max
                              > 5.0 * max(interior_w_max, 1.0)),
        dynamics_substeps=dynamics_substeps,
        ysu_nan_guard_fires=state.physics.ysu_nan_guard_fires,
        surface_forcing_updates=surface_forcing_updates,
        swdown_peak_wm2=swdown_peak, swdown_peak_time=swdown_peak_time,
        completed_seconds=float(state.elapsed_seconds),
        rainc_max_mm=rainc_max, rainc_max_ji=rainc_ji,
        rainc_max_lat=rainc_lat, rainc_max_lon=rainc_lon,
    )


# ---------------------------------------------------------------------------
# Resolved-config report (G2): every formerly implicit path/time/policy
# ---------------------------------------------------------------------------

def resolved_config_report(exp: ExperimentConfig, data: CaseDataConfig,
                           forcing_times=None, *, input_catalog=None) -> str:
    """Human- and test-readable enumeration of every resolved value.

    Every path, time, and policy that used to be implicit in the runtime
    path appears here by name.  ``input_catalog`` supplies the forcing
    selection authority and its exclusions; ``forcing_times`` may narrow that
    selection to the schedule actually consumed by preparation.
    """
    dc = single_domain(exp)
    cfg = dc.run
    lines = [f"resolved experiment configuration -- {exp.name}"]

    def add(key, value):
        lines.append(f"  {key} = {value}")

    for record in data.resolved_inputs():
        detail = f" ({record.detail})" if record.detail else ""
        add(f"input.{record.role}", f"{record.path}{detail}")
    if data.source_orography is None:
        add("input.source_orography",
            "era5_z_invariant from forcing SOILGEO")
    add("time.start_time", exp.start_time.isoformat())
    add("time.run_seconds", f"{exp.run_seconds:g}")
    add("time.dt", f"{cfg.dt:g}")
    add("time.history_interval_s", f"{dc.history_interval_s:g}")
    add("time.restart_interval_s", f"{exp.restart_interval_s:g}")
    add("radiation.column_chunk", f"{exp.column_chunk}")
    add("time.forcing_interval_s",
        "discover" if data.forcing_interval_s is None
        else f"{data.forcing_interval_s:g}")
    if input_catalog is not None and forcing_times is None:
        forcing_times = input_catalog.valid_times
    if forcing_times is not None:
        times = tuple(forcing_times)
        coverage = (times[-1] - times[0]).total_seconds()
        add("time.forcing_times",
            ", ".join(t.isoformat() for t in times))
        add("time.forcing_times_consumed",
            ", ".join(t.isoformat() for t in times))
        add("time.forcing_coverage_s", f"{coverage:g}")
    if input_catalog is not None:
        exclusions = tuple(input_catalog.excluded_valid_times)
        add("time.forcing_times_excluded_by_catalog",
            (", ".join(t.isoformat() for t in exclusions)
             if exclusions else "none"))
    add("vertical.nz", f"{cfg.nz}")
    add("vertical.eta_levels",
        f"{len(exp.vertical.eta_levels)} full levels "
        f"[{exp.vertical.eta_levels[0]:g} .. "
        f"{exp.vertical.eta_levels[-1]:g}]")
    add("vertical.p_top", f"{exp.vertical.p_top:g}")
    add("vertical.hybrid_opt", f"{exp.vertical.hybrid_opt}")
    add("vertical.etac", f"{exp.vertical.etac:g}")
    add("policy.sfcp_to_sfcp", str(data.sfcp_to_sfcp))
    add("policy.co2_vmr", (f"{data.co2_vmr:g}" if data.co2_vmr is not None
                           else "date-indexed NOAA annual policy"))
    add("policy.climatology_date",
        exp.start_time.date().isoformat()
        + " (monthly GEOG fields interpolated to the run date)")
    add("output.domain_id", f"{data.output_domain}")
    add("output.title", data.output_title)
    add("output.filename_pattern",
        f"wrfout_d{data.output_domain:02d}_<YYYY-MM-DD_HH_MM_SS>")
    add("grid.nx_ny_dx", f"{cfg.nx} x {cfg.ny} @ {cfg.dx:g} m")
    if exp.projection is not None:
        proj = exp.projection
        add("grid.projection",
            f"{proj.map_proj} ref=({proj.ref_lat:g}, {proj.ref_lon:g}) "
            f"truelat=({proj.truelat1:g}, {proj.truelat2:g}) "
            f"stand_lon={proj.stand_lon:g}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment-path entry points (CLI static / ingest / run)
# ---------------------------------------------------------------------------

def _write_npz(path, fields: dict[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    return path


def write_static(exp: ExperimentConfig, data: CaseDataConfig,
                 output) -> Path:
    """Build the experiment domain's GEOG fields into a portable NPZ."""
    dc = single_domain(exp)
    grid = experiment_grid(exp, data)
    selection = GeogSelection.from_case_data(data, domain_id=dc.grid_id)
    fields = build_static(
        grid, data.geog_root, selection=selection)
    return _write_npz(output, fields)


def write_ingest(exp: ExperimentConfig, data: CaseDataConfig,
                 output) -> Path:
    """Run real-data initialization and write its live FP32 state to NPZ.

    This is a stage artifact for inspection/reproducibility, not a restart
    file; ``gpuwm run`` rebuilds the deterministic setup from the same
    config.
    """
    prepared = prepare_experiment_case(exp, data)
    result = prepared.initial_result
    state = result.state
    import cupy as cp
    names = ("u", "v", "w", "thp", "php", "mup", "qv", "qc", "qr",
             "mub2d", "msft", "msfu", "msfv", "f", "e")
    fields = {name: cp.asnumpy(getattr(state, name)) for name in names}
    fields.update({
        "surface_pressure": result.surface_pressure,
        "surface_qv": result.surface_qv,
        "dry_mass": result.dry_mass,
        "dry_pressure": result.dry_pressure,
        "total_pressure": result.total_pressure,
        "total_geopotential": result.total_geopotential,
        "integrated_moisture_pressure": result.integrated_moisture_pressure,
        "case": np.asarray(exp.name),
    })
    return _write_npz(output, fields)


def run_experiment(exp: ExperimentConfig, data: CaseDataConfig, outdir, *,
                   restart=None, progress_callback=None,
                   health_debug: bool = False
                   ) -> RealCaseRunSummary | ExperimentRunSummary:
    """Prepare and integrate a single domain or a complete domain tree.

    Prints the resolved-config report (G2) before any device work, then
    runs the extracted prepare/integrate pipeline with every input and
    policy drawn from the config pair.
    """
    from gpuwm.io.wrfout import quarantine_orphan_wrfouts

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    experimental_feedback = feedback_provenance(exp)
    if experimental_feedback is not None:
        print(FEEDBACK_EXPERIMENTAL_WARNING)
    _preparation_progress(progress_callback, "quarantine-wrfout")
    quarantine_orphan_wrfouts(outdir)
    if len(exp.domains) == 1:
        # Frozen cardinal path: retain Task-2's exact preparation, loop,
        # output order, and single-file v3/v2 restart shims.
        _preparation_progress(progress_callback, "resolve-schedule")
        dc = single_domain(exp)
        from gpuwm.ingest.preflight import build_input_catalog

        catalog = build_input_catalog(data)
        snapshots = forcing_snapshots(data, catalog)
        times = forcing_schedule(exp, data, snapshots)
        print(resolved_config_report(
            exp, data, forcing_times=times, input_catalog=catalog))
        _preparation_progress(progress_callback, "prepare-case")
        prepared = prepare_experiment_case(
            exp, data, input_catalog=catalog, forcing_by_time=snapshots)
        summary = integrate_prepared_case(
            outdir, prepared, start_time=exp.start_time,
            output_title=data.output_title, domain_id=data.output_domain,
            run_seconds=exp.run_seconds,
            history_interval_s=dc.history_interval_s,
            restart_interval_s=exp.restart_interval_s,
            restart_path=restart, progress_callback=progress_callback,
            health_debug=health_debug, feedback=experimental_feedback)
        _write_feedback_provenance_receipt(
            outdir, exp, resumed=restart is not None)
        return summary

    _preparation_progress(progress_callback, "build-domain-tree")
    from gpuwm.core.model import build_experiment, execute_experiment
    from gpuwm.io.restart import (restore_tree_restart,
                                  write_tree_restart)
    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    from gpuwm.supervisor import validate_manifest_checkpoint

    model = build_experiment(exp, data)
    print(resolved_tree_config_report(exp, data, model._input_catalog))
    if restart is not None:
        _preparation_progress(progress_callback, "validate-checkpoint")
        restart = validate_manifest_checkpoint(restart)
        _preparation_progress(progress_callback, "restore-tree-checkpoint")
        restore_tree_restart(restart, model)

    _preparation_progress(progress_callback, "initialize-domain-writers")
    with PerDomainWrfoutWriters(
            model, outdir, start_time=exp.start_time,
            title=data.output_title) as writers:
        model._io_manager = writers

        def history_handler(tree, node, ticks):
            _submit_tree_history_frame(writers, node, ticks)

        def restart_handler(tree, ticks):
            valid = exp.start_time + timedelta(
                seconds=ticks / tree.schedule.clock.tick_den)
            tree._last_checkpoint = write_tree_restart(outdir, tree, valid)

        execute_experiment(
            model, history_handler=history_handler,
            restart_handler=restart_handler,
            progress_callback=progress_callback,
            health_debug=health_debug)
        writers.drain()
        paths = writers.paths
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    transition_path, transition_sha, transitions = \
        _write_microphysics_transition_receipt(
            outdir, model, exp, resumed=restart is not None)
    feedback_path, feedback_sha, feedback_receipt = \
        _write_feedback_provenance_receipt(
            outdir, exp, resumed=restart is not None)
    return ExperimentRunSummary(
        wrfout_paths=paths,
        completed_seconds=model.root.clock.elapsed_seconds,
        nan_free=True,
        last_checkpoint=getattr(model, "_last_checkpoint", None),
        microphysics_transitions=transitions,
        microphysics_transition_receipt=transition_path,
        microphysics_transition_receipt_sha256=transition_sha,
        feedback_provenance=feedback_receipt,
        feedback_provenance_receipt=feedback_path,
        feedback_provenance_receipt_sha256=feedback_sha)


def _submit_tree_history_frame(writers, node, ticks: int) -> None:
    """Production tree-history handoff, kept directly CPU-testable.

    REFL remains D2 driver-rebuilt state.  At a true period boundary the
    producing step has stashed the field, this function consumes it, and the
    restart callback drains the resulting D2H publication before writing the
    tree checkpoint.  A restored model suppresses this already-committed
    callback per due domain, so no missing stash is ever read.
    """
    refl_field = None
    if (ticks != 0 and node.state.qv is not None
            and node.state.physics.mp_physics in (1, 6, 8, 10, 18)):
        from gpuwm.core.refl import consume_refl_10cm
        refl_field = consume_refl_10cm(node.state)
    writers.submit(node, ticks, refl_field=refl_field)
    # History-interval reset of this domain's UP_HELI_MAX window.  Safe
    # ordering: submit's producer-stream wait_event fences the side-stream
    # D2H snapshot ahead of any later default-stream mutation, so zeroing
    # here can never race the staged copy.
    from gpuwm.core.uh_diag import reset_up_heli_max
    reset_up_heli_max(node.state)


def resolved_tree_config_report(exp: ExperimentConfig,
                                data: CaseDataConfig, catalog) -> str:
    """Compact resolved report for the multi-domain run surface."""
    lines = [f"resolved experiment configuration -- {exp.name}"]
    if exp.feedback == 1:
        lines.append("  feedback = experimental")
    for record in data.resolved_inputs():
        lines.append(f"  input.{record.role} = {record.path}")
    lines.extend((
        f"  time.start_time = {exp.start_time.isoformat()}",
        f"  time.run_seconds = {exp.run_seconds:g}",
        f"  time.restart_interval_s = {exp.restart_interval_s:g}",
        f"  radiation.column_chunk = {exp.column_chunk}",
        f"  input_catalog.sha256 = {catalog.fingerprint}",
        f"  domains = {len(exp.domains)}",
    ))
    for dc in exp.domains:
        lines.append(
            f"  domain.d{dc.grid_id:02d} = parent={dc.parent_id} "
            f"start={exp.domain_start_time(dc.grid_id).isoformat()} "
            f"{dc.run.nx}x{dc.run.ny} dx={dc.run.dx:g} "
            f"dt={dc.run.dt:g} history={dc.history_interval_s:g} "
            f"mp_physics={dc.run.mp_physics} "
            "nest_microphysics_transition="
            f"{dc.run.nest_microphysics_transition}")
    lines.append(
        "  output.filename_pattern = wrfout_d0X_<YYYY-MM-DD_HH_MM_SS>")
    return "\n".join(lines)


__all__ = [
    "ExperimentRunSummary", "FEEDBACK_EXPERIMENTAL_WARNING",
    "FEEDBACK_PROVENANCE_RECEIPT_NAME",
    "MICROPHYSICS_TRANSITION_RECEIPT_NAME",
    "PreparedRealCase", "RealCaseRunSummary",
    "configured_run_schedule",
    "experiment_grid", "feedback_provenance",
    "forcing_schedule", "forcing_snapshots",
    "integrate_prepared_case", "load_source_orography",
    "prepare_child_case", "prepare_experiment_case",
    "prepare_root_experiment_case", "prepare_real_case", "refl_10cm_due",
    "resolved_config_report", "resolved_tree_config_report",
    "restart_outer_steps", "run_experiment",
    "single_domain", "vertical_coord_for", "whole_step_count",
    "write_case_output", "write_ingest", "write_static",
]
