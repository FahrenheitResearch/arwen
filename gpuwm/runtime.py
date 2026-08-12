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
import math
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from gpuwm.case_data import CaseDataConfig
from gpuwm.ingest.water_temperature import (
    WaterTemperatureStatics, resolve_water_temperature_policy)
from gpuwm.certify.capsule import emit_run_capsule
from gpuwm.config import (DEFAULT_COLUMN_CHUNK, RunConfig,
                          radiation_scheme_ids, soil_layer_count)
from gpuwm.core.grid import make_vertical_coord
from gpuwm.core.noah import noah_initial_snow_albedo
from gpuwm.experiment import ExperimentConfig, VerticalConfig
from gpuwm.ingest.grib import cached_era5_forcing
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.lateral_bc import (StateBoundaryFrames,
                                     attach_lateral_boundaries)
from gpuwm.ingest.preprocess_backend import (
    CudaPreprocessBackend,
    release_backend_memory,
)
from gpuwm.ingest.real import initialize_real
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.static.build import (GeogSelection, build_static,
                                monthly_interp_to_date)
from gpuwm.static.lambert import grids_from_projection_config


#: ``mp_physics`` values whose microphysics call stages a scheme-native
#: REFL_10CM field for the history writer to consume.  Named, not inlined,
#: because this runtime carries THREE separate gates on it (case output,
#: the per-substep ``refl_due`` schedule, and the nested-tree history
#: handoff) and an inlined tuple that is updated at two of the three is a
#: silent no-radar-data output frame, not an error.
#:
#: 28 (Thompson aerosol-aware) belongs here on WRF's own structure: WRF
#: reaches ``calc_refl10cm`` from the single call site
#: ``mp_gt_driver:1458``, gated on ``diagflag .and. do_radar_ref == 1``
#: (module_mp_thompson.F:1449) and never on ``is_aerosol_aware``.  gpuwm
#: matches it -- ``gpuwm/core/microphysics_aerosol.py`` stages the field
#: through the same ``compute_and_stash_refl_10cm`` seam as mp=8 -- so
#: excluding 28 here does not disable a diagnostic, it strands a field
#: that the scheme has already computed and that ``refl.py``'s
#: consume-once contract then reports as an unconsumed stash.
REFL_10CM_MICROPHYSICS = (1, 6, 8, 9, 10, 16, 18, 28, 50)

MICROPHYSICS_TRANSITION_RECEIPT_NAME = "microphysics-transitions.json"
FEEDBACK_PROVENANCE_RECEIPT_NAME = "feedback-provenance.json"
INITIAL_PERTURBATION_RECEIPT_NAME = "initial-perturbation.json"
FEEDBACK_EXPERIMENTAL_WARNING = (
    "WARNING: feedback = 1 is EXPERIMENTAL and is not certified against "
    "stock WRF yet; the certification reference is in progress."
)
CONSERVATION_CLOSURE_RECEIPT_NAME = "conservation-closure.json"


#: This route's name in every water-temperature refusal and receipt.  Not a
#: case name and not a file name: the ROUTE, so a false or missing assembly
#: can be attributed without reading a traceback.
_WATER_ROUTE = "the ERA5 runtime route"


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
    #: Per-domain canonical trajectory digest, or ``None`` with the
    #: instrumentation disabled (see :func:`trajectory_digest_enabled`).
    trajectory_digest: dict | None = None


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
    #: Per-domain canonical trajectory digest, or ``None`` with the
    #: instrumentation disabled (see :func:`trajectory_digest_enabled`).
    trajectory_digest: dict | None = None


#: Environment switch that turns the trajectory-digest instrumentation off.
#: The A4 control pair runs the same short-window config with the digest on
#: and off; if the instrumentation participated in the trajectory the two runs
#: would differ, and the shipped comparator is what says whether they do.
TRAJECTORY_DIGEST_ENV = "GPUWM_TRAJECTORY_DIGEST"


def trajectory_digest_enabled() -> bool:
    """Whether the run-route trajectory digest is computed.  Default on."""
    import os

    raw = os.environ.get(TRAJECTORY_DIGEST_ENV)
    return True if raw is None else raw.strip().lower() not in {
        "0", "false", "no", "off"}


class _SingleDomainDigestClock:
    """The boundary-clock inputs the frozen single-domain loop owns.

    ``canonical_state_digest`` mixes WRF's REAL boundary accumulator
    ``dtbc`` into the digest.  The frozen single-domain loop maintains no
    such accumulator: ``dtbc`` is a domain-tree coupler property
    (``gpuwm/core/model.py`` root-boundary section) and lateral forcing on
    this route is applied from the configured interval.  Reporting the
    ``DomainClock`` initial value keeps the digest a complete, reproducible
    function of the state this route does own, and the run summary records
    that the route does not advance it, so nobody reads the digest as
    evidence about a boundary clock that never ran.
    """

    __slots__ = ()

    #: ``gpuwm.core.clock.DomainClock`` initialises ``dtbc_fp32`` to +0.0f.
    dtbc_fp32 = np.float32(0.0)

    #: Recorded beside the digest so the omission is legible in the capsule.
    provenance = (
        "the frozen single-domain loop maintains no WRF dtbc accumulator; "
        "the digest carries the DomainClock initial value")


def _frame_records(paths) -> list[dict[str, object]]:
    """Every emitted frame with the SHA-256 of the bytes now on disk."""
    records = []
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        records.append({"path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest()})
    return records


def _emit_front_door_capsule(outdir, *, emission_site: str, exp,
                             data: CaseDataConfig, wrfout_paths,
                             trajectory_digest, io_mode: str) -> Path:
    """Write the front door's certification capsule.

    Unconditional: it does not consult ``exp.feedback``, because a receipt
    that appears only on one physics tier is a receipt the other tier cannot
    be certified from.
    """
    run_context = {
        "runner_route_and_io_mode": {
            "route": emission_site, "io_mode": io_mode},
        "output_and_diagnostic_mode": {
            "io_mode": io_mode,
            "history_interval_seconds": float(exp.domains[0].history_interval_s)
            if getattr(exp.domains[0], "history_interval_s", None) is not None
            else None,
            "restart_interval_seconds": (
                None if exp.restart_interval_s is None
                else float(exp.restart_interval_s)),
        },
    }
    run_shape = {
        "route": emission_site,
        "domain_count": len(exp.domains),
        "run_seconds": float(exp.run_seconds),
        "start_time": exp.start_time.isoformat(),
        "output_title": data.output_title,
    }
    output = {
        "frames": _frame_records(wrfout_paths),
        "trajectory_digest": trajectory_digest,
    }
    return emit_run_capsule(
        outdir, emission_site=emission_site, run_context=run_context,
        run_shape=run_shape, output=output)


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
    # Unique temp and an explicit durability barrier, matching the
    # microphysics transition receipt below.  A fixed temp name is not
    # reentrant outside the supervisor's serialization, and a receipt whose
    # bytes never left the page cache is a receipt a power failure can
    # unwrite after the rename has already made it look durable.
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path, hashlib.sha256(encoded).hexdigest(), payload


def _write_initial_perturbation_receipt(
        outdir: Path, exp: ExperimentConfig, domain_receipts
        ) -> Path | None:
    """Atomically publish what the [perturbation] bubbles actually wrote.

    The treatment-proof receipt: the accepted config echoed value for
    value, plus each initialized domain's per-bubble application stats
    (cells touched, max theta delta, qv adjustment).  ``None`` -- and no
    file -- when the experiment carries no block, so an absent block
    leaves the run directory byte-identical.  Written as soon as the
    initial states exist, before integration, so even a run that dies
    mid-flight proves its arm.
    """
    if exp.perturbation is None:
        return None
    payload = {
        "schema": "gpuwm-initial-perturbation-receipt-v1",
        "experiment": exp.name,
        "config": exp.perturbation.receipt(),
        "domains": [dict(receipt) for receipt in domain_receipts],
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n").encode("utf-8")
    path = Path(outdir) / INITIAL_PERTURBATION_RECEIPT_NAME
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    for receipt in payload["domains"]:
        for row in receipt.get("bubbles", ()):
            if row.get("applied"):
                print(
                    f"initial perturbation: bubble {row['bubble']} on "
                    f"d{receipt['grid_id']:02d} touched "
                    f"{row['cells_touched']} cells, max theta delta "
                    f"{row['max_theta_added_k']:.3f} K"
                    + (", max qv delta "
                       f"{row['max_qv_delta_kg_kg']:.3e} kg/kg"
                       if "max_qv_delta_kg_kg" in row else ""))
            else:
                print(
                    f"initial perturbation: bubble {row['bubble']} on "
                    f"d{receipt['grid_id']:02d} not applied "
                    f"({row.get('reason', 'unstated')})")
    return path


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
    forcing_identities = (
        data.forcing_identity()
        if hasattr(data, "forcing_identity") else data.forcing)
    content_sha256 = tuple(
        forcing_hashes.get(Path(path).resolve(), "")
        for path in forcing_identities
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
    # Optional hi-res water-temperature overlay (task #71).  The decode
    # above serves RAW snapshots from the process cache, so the declared
    # overlay is applied here, mirroring build_input_catalog's
    # application to the catalog's own served snapshots (the nested-child
    # source).  Absent key: this branch never runs.
    water_overlay_path = getattr(data, "water_temperature_overlay", None)
    if water_overlay_path is not None:
        from gpuwm.ingest.water_overlay import (
            cached_water_temperature_overlay,
            overlay_snapshots_by_time,
        )

        overlay = cached_water_temperature_overlay(water_overlay_path)
        by_time, receipt = overlay_snapshots_by_time(by_time, overlay)
        print(
            "water-temperature overlay: replaced "
            f"{receipt['replaced_cells']} of {receipt['water_cells']} "
            f"water source cells per snapshot from {receipt['path']} "
            f"({receipt['fallback_cells']} kept ERA5 fallback)")
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

def declared_constant_glw(exp: ExperimentConfig) -> float | None:
    """The constant downward longwave this experiment DECLARED, or None.

    A real case reaches a preparer only after
    :func:`gpuwm.experiment.build_experiment` has run
    :func:`gpuwm.physics_compat.constant_longwave_refusal`, so a config
    that runs a land-surface scheme with ``ra_lw_physics = 0`` and got
    this far carries
    :data:`~gpuwm.physics_compat.CONSTANT_DOWNWARD_LONGWAVE_ACK`.  This
    turns that declaration into the number the preparer TYPES into
    :func:`~gpuwm.core.physics.initialize_physics`, because that function
    refuses to invent one.

    ``None`` for every other experiment, which is the normal answer: a
    run with a longwave scheme has its GLW written by that scheme, and
    passing a value would only pre-fill a buffer the scheme overwrites.
    """

    from gpuwm.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK

    if CONSTANT_DOWNWARD_LONGWAVE_ACK in tuple(exp.acknowledgements or ()):
        from gpuwm.core.physics import DECLARED_CONSTANT_GLW_WM2
        return DECLARED_CONSTANT_GLW_WM2
    return None


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
                      static_highres=None,
                      static_domain_id: int = 1,
                      initial_perturbation=None,
                      constant_glw_wm2: float | None = None,
                      water_temperature_policy=None,
                      ) -> PreparedRealCase:
    """Run the real-case setup pipeline for one domain.

    Extraction of the frozen reference preparation: identical operations in
    identical order, parameterized by the formerly implicit values.
    ``snapshot_for(valid_time)`` supplies the decoded forcing snapshot
    for each entry of ``forcing_times`` (the first entry must be
    ``start_time`` -- it becomes the live initial state).
    ``trace_gas_overrides`` feeds the RRTMGP trace-gas policy hook;
    ``None`` keeps the scheme's frozen default composition.
    ``initial_perturbation`` is the validated experiment
    :class:`gpuwm.experiment.PerturbationConfig` (or ``None``, the OFF
    contract): its bubbles are applied inside ``initialize_real`` for
    the START TIME ONLY -- later forcing times contribute unperturbed
    boundary frames, except the t=0 frame, which reads the perturbed
    state (a bubble inside the relax zone would enter it; interior
    bubbles leave every boundary strip byte-identical).  This is the
    coarse/single domain, so a bubble center outside the grid refuses.
    """
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.landuse import (initialize_landuse,
                                    reconciled_soil_category)
    from gpuwm.core.physics import initialize_physics
    from gpuwm.ingest.soil import (reconciler_soil_temperature,
                                   reconciler_sst)

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
    perturbation_applier = None
    if initial_perturbation is not None:
        # Built once, against this (coarse/single) domain's grid, so an
        # out-of-domain bubble center refuses HERE -- before any decode,
        # static build, or device work.
        from gpuwm.ingest.init_perturbation import (
            build_initial_state_perturbation)
        perturbation_applier = build_initial_state_perturbation(
            initial_perturbation, grid, grid_id=int(cfg.grid_id),
            require_containment=True)
    static = _cached_static_build(
        grid, geog_root, selection=geog_selection)
    landuse_attrs = geog_selection.landuse_global_attrs()
    if static_highres is not None and getattr(static_highres, "enabled",
                                              False):
        from gpuwm.static.highres_production import apply_highres_statics
        static, _ = apply_highres_statics(
            static, grid, config=static_highres,
            domain_id=static_domain_id, case_date=start_time.date(),
            landuse_attrs=landuse_attrs)
    source_orography = None
    if source_orography_path is not None:
        source_orography = load_source_orography(
            source_orography_path, source_orography_variable)
    # The surface the water-temperature assembly decides on: this domain's
    # own LANDMASK and the land-use table's own ISLAKE, so a lake and the
    # ocean stay in separate provider decisions even where a coarse
    # coastline connects them on the target.  Built once for the whole
    # forcing loop, because it is invariant across forcing times.
    water_statics = WaterTemperatureStatics.for_route(
        route=_WATER_ROUTE, policy=water_temperature_policy,
        landmask=static["LANDMASK"], lu_index=static["LU_INDEX"],
        landuse_attrs=landuse_attrs)
    # Only the first time's analysis/state and the last time's analysis
    # outlive this loop; every intermediate time contributes its perimeter
    # frames and is released before the next one is built.  Retaining all
    # N of them made setup, not the forecast, the memory-binding phase.
    initial_result = None
    initial_met = None
    forcing = StateBoundaryFrames(
        spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
        relax_zone=cfg.relax_zone)
    release_backend = CudaPreprocessBackend()
    met = result = None
    for index, valid_time in enumerate(times):
        if index:
            del met, result
            release_backend_memory(release_backend)
        source = snapshot_for(valid_time)
        # Metgrid classifies masked-field TARGET cells by the model
        # (geogrid) landmask, not by the nearest source LSM; the source-side
        # usable-point decision stays with the source LANDSEA inside the
        # masked operators.  Passing the static LANDMASK reproduces WPS and
        # keeps soil, skin, and physics on one land/water surface.
        met = interpolate_era5_to_lambert(
            source, grid, source_orography_catalog=forcing_catalog,
            target_landmask=np.asarray(static["LANDMASK"]) >= 0.5,
            water_temperature_statics=water_statics)
        coord = vertical_coord_for(vertical, cfg.nz)
        init_kwargs = dict(
            source_orography=source_orography, p_top=vertical.p_top,
            sfcp_to_sfcp=sfcp_to_sfcp)
        if scratch_arena is not None:
            init_kwargs["scratch_arena"] = scratch_arena
        if dycore_state_workspace is not None:
            init_kwargs["dycore_state_workspace"] = dycore_state_workspace
        if index == 0 and perturbation_applier is not None:
            # The bubbles perturb the LIVE INITIAL STATE only; the later
            # forcing times of this loop are boundary material and stay
            # the unperturbed analysis.
            init_kwargs["initial_perturbation"] = perturbation_applier
        result = initialize_real(
            met, cfg, coord, static["HGT_M"], **init_kwargs)
        f, e = grid.coriolis_m()
        # SINALPHA/COSALPHA (geo_em conventions): WRF's coriolis applies
        # the rotation terms unconditionally (module_em.F:761-769).
        sina, cosa = grid.rotation_m()
        result.state.set_map_coriolis(
            grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(), f, e,
            sina=sina, cosa=cosa)
        forcing.add_state(result.state)
        if index == 0:
            initial_met = met
            initial_result = result
    # ``met``/``result`` now name the LAST forcing time; the first are held
    # separately above.  Nothing between them is still resident.
    final_met = met
    boundaries = forcing.build(times)
    attach_lateral_boundaries(initial_result.state, boundaries)

    soil_fields = dict(initial_met.fields)
    # No lake skin override: with metgrid's masked=both SKINTEMP chain and
    # static-landmask target classification, lake cells already carry the
    # water-source skin value, and real.exe (no TAVGSFC) keeps exactly that
    # SKINTEMP wherever SST has no valid support
    # (module_initialize_real.F:2844-2866, :2898-2906).  The router forwards
    # this exact argument list to preprocess_noah_soil for Noah-geometry
    # schemes, so their soil state is unchanged by the LSM dispatch seam.
    # ONE RULEBOOK (Drew's ruling, 2026-08-06).  The soil column and the
    # liquid water derived from it must be built with the SAME category the
    # physics driver integrates, so ask for the reconciled ISLTYP here
    # rather than reading the raw geogrid SCT_DOM.  WRF gets this ordering
    # for free: real.exe reconciles at module_initialize_real.F:3608-3650
    # and LSMINIT (phys/module_sf_noahdrv.F) derives SH2O afterwards.  Ours
    # ran the other way round, because initialize_landuse below needs this
    # call's own outputs (snow, xice, TSLB) and therefore cannot precede it.
    # Evidence spellings come from the one per-source table in
    # gpuwm/ingest/soil.py, so this root call and the nested-child call in
    # gpuwm/ingest/nest_init.py cannot drift apart again: an inline chain
    # here knew only the mapped and classic per-layer names, which is the
    # same gap that aborted the native-HRRR and nested-GFS lanes.
    reconciled_soil_type = reconciled_soil_category(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        xice=soil_fields.get("XICE", 0.0),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        soil_temperature=reconciler_soil_temperature(soil_fields),
        sst=reconciler_sst(soil_fields))
    soil = preprocess_land_surface_soil(
        soil_fields, sf_surface_physics=int(cfg.sf_surface_physics),
        soil_type=reconciled_soil_type,
        deep_soil_temperature=static["TMN"],
        landmask=static["LANDMASK"],
        terrain=static["HGT_M"] if source_orography is not None else None,
        source_orography=source_orography,
        # The finished water temperature the ingest assembled: one provider
        # per connected body of water, never a per-cell choice between two
        # differently-mapped fields.  The policy and the route name travel
        # with it, because the router refuses a raw SST/SKINTEMP pair that
        # arrives with no decision attached.
        water_temperature=getattr(
            initial_met, "water_temperature", None),
        water_temperature_policy=water_temperature_policy,
        route=_WATER_ROUTE)
    # WRF interpolates GREENFRAC/LAI to the run date
    # (module_initialize_real.F:1322-1335, mid-month anchors); shdmin/
    # shdmax stay the monthly extrema (:1348-1351).  With the supported
    # usemonalb=false path, landuse_init overwrites ALBEDO12M from the table.
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], start_time)
    lai = monthly_interp_to_date(static["LAI12M"], start_time)
    state = initial_result.state
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
                start_time, lat, lon, p_top=vertical.p_top,
                o3input=cfg.o3input)
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
        static["LU_INDEX"], soil_type=reconciled_soil_type,
        landmask=static["LANDMASK"], snow=soil.snow_water, xice=soil.xice,
        valid_time=start_time,
        cen_lat=float(getattr(grid, "cen_lat", np.mean(lat))),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=soil.soil_temperature)
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        sst=soil_fields.get("SST", soil.tsk),
        glw=constant_glw_wm2,
        radiation=radiation,
        radiation_start_time=start_time, radiation_latitude=lat,
        radiation_longitude=lon)
    import cupy as cp
    driver.fields["snoalb"][...] = cp.asarray(
        noah_initial_snow_albedo(
            static["SNOALB"], static["LU_INDEX"], driver.noah_params,
            rdmaxalb=cfg.rdmaxalb),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)

    # Seed time-zero surface diagnostics from the source analysis.  The
    # first model step replaces them through SFCLAY/Noah/YSU in WRF
    # ordering.
    met0 = initial_met.fields
    driver.fields["psfc"][...] = cp.asarray(
        initial_result.surface_pressure, dtype=cp.float32)
    driver.fields["t2"][...] = met0["T2"]
    driver.fields["q2"][...] = cp.asarray(
        initial_result.surface_qv, dtype=cp.float32)
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
        initial_result=initial_result, final_analysis=final_met,
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
        water_temperature_policy=resolve_water_temperature_policy(data),
        scratch_arena=scratch_arena,
        dycore_state_workspace=dycore_state_workspace,
        radiation_column_chunk=exp.column_chunk,
        static_highres=getattr(data, "static_highres", None),
        static_domain_id=dc.grid_id,
        initial_perturbation=exp.perturbation,
        constant_glw_wm2=declared_constant_glw(exp))


def prepare_experiment_case(exp: ExperimentConfig,
                            data: CaseDataConfig, *, input_catalog=None,
                            forcing_by_time=None) -> PreparedRealCase:
    """Assemble :func:`prepare_real_case` inputs from the config pair."""
    single_domain(exp)  # Preserve Task-2's fail-loud single-domain surface.
    return prepare_root_experiment_case(
        exp, data, input_catalog=input_catalog,
        forcing_by_time=forcing_by_time)


def _child_radiation_adapter(exp: ExperimentConfig, data: CaseDataConfig,
                             dc, state, lat, lon, *,
                             radiation_workspace=None,
                             radiation_parent=None):
    """Construct one child domain's radiation adapter (shared by the
    t=0 preparer and the relocation preparer, so a relocated child's
    radiation is wired by exactly the code that wired it at start)."""
    cfg = dc.run
    radiation = None
    if radiation_scheme_ids(cfg) == (4, 4):
        from gpuwm.physics_compat import (
            RRTMG_VARIANT_LEGACY, rrtmg_variant)
        if rrtmg_variant(cfg) == RRTMG_VARIANT_LEGACY:
            # Legacy RRTMG child: the exact WRF v4.6.1 port with WRF's
            # root-compute + parent->child ozone routing.  The parent
            # adapter is mandatory -- a per-nest climatology evaluation
            # would diverge from WRF on d02+ (never fall back silently).
            if radiation_parent is None and cfg.o3input == 2:
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
                ozone_parent=(
                    ParentOzoneProvider(radiation_parent, registration)
                    if cfg.o3input == 2 else None),
                o3input=cfg.o3input)
        else:
            from gpuwm.core.rrtmgp import RRTMGPRadiation
            overrides = ({"co2": data.co2_vmr}
                         if data.co2_vmr is not None else None)
            radiation = RRTMGPRadiation(
                exp.start_time, lat, lon, trace_gas_overrides=overrides,
                column_chunk=exp.column_chunk)
            if radiation_workspace is not None:
                radiation.column_chunk = radiation_workspace.column_chunk
                radiation.chunk_workspace = radiation_workspace
    return radiation


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
    radiation = _child_radiation_adapter(
        exp, data, dc, state, lat, lon,
        radiation_workspace=radiation_workspace,
        radiation_parent=radiation_parent)
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
        isice=int(landuse_attrs["ISICE"]),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=soil.soil_temperature)
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=soil.tsk,
        soil_temperature=soil.soil_temperature,
        soil_moisture=soil.soil_moisture,
        liquid_moisture=soil.liquid_moisture,
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=soil.deep_soil_temperature,
        xice=soil.xice, snow=soil.snow_water, snow_depth=soil.snow_depth,
        glw=declared_constant_glw(exp),
        radiation=radiation, radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon)
    driver.fields["snoalb"][...] = cp.asarray(
        noah_initial_snow_albedo(
            static["SNOALB"], static["LU_INDEX"], driver.noah_params,
            rdmaxalb=cfg.rdmaxalb),
        dtype=cp.float32)
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
# Real-data relocation: the route-owned child preparer and runner wiring
# ---------------------------------------------------------------------------

def _relocation_host(value) -> np.ndarray:
    if hasattr(value, "__cuda_array_interface__"):
        return np.asarray(value.get())
    return np.array(value, copy=True)


def rebuild_child_driver_from_land_state(*, exp: ExperimentConfig,
                                         data: CaseDataConfig, model,
                                         initialized, child_dc, parent_node,
                                         land, landuse_attrs=None,
                                         radiation_factory=None) -> float:
    """Rebuild one child's physics driver over a supplied land state.

    The operation both mid-run child events need, and the reason they can
    share it: a relocation and a spawn differ ONLY in where the land state
    comes from (an index-space transplant plus donor fill for the first,
    WRF's masked parent interpolator for the second).  Once the fields are
    in hand the rebuild is identical -- ``initialize_landuse`` /
    ``initialize_physics`` against the footprint's OWN statics, then the
    continuation fields the constructor does not take by direct overwrite
    -- and it is the same wiring the t = 0 child preparer
    (:func:`prepare_child_case`) performs.

    ``land`` maps :data:`~gpuwm.ingest.relocation_init
    .LAND_SURFACE_CONTINUATION_FIELDS` names to host arrays; a name the
    caller does not supply falls back to the cold-start default, exactly
    as it did when this lived inside the relocation preparer.
    Accumulators are NOT here: they are re-initialised at the new
    footprint, and both callers' receipts say so.

    ``landuse_attrs`` and ``radiation_factory`` are the two route seams:
    ``None`` (every case-data caller) derives both from ``data`` exactly
    as before -- the GEOG selection's land-use identity and the shared
    child radiation adapter.  The prepared tree route, which has no
    ``CaseDataConfig``, passes its own native land-use identity and a
    factory reproducing its t=0 radiation wiring
    (``radiation_factory(child_dc, state, lat, lon) -> callable|None``;
    ``None`` lets ``initialize_physics`` build the scheme from the
    RunConfig, byte-for-byte the prepared t=0 path).

    Returns the wall seconds the rebuild took.
    """
    import time as _time

    import cupy as cp

    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.physics import initialize_physics

    started = _time.perf_counter()
    cfg = child_dc.run
    static = initialized.static_fields
    grid = initialized.grid
    state = initialized.state
    now = exp.start_time + timedelta(
        seconds=float(parent_node.clock.elapsed_seconds))
    # Climatology fields interpolate to the EVENT time: a child rebuilt
    # (or born) in May must not wear its January vegetation.
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], now)
    lai = monthly_interp_to_date(static["LAI12M"], now)
    lat, lon = grid.latlon_mass()
    if radiation_factory is None:
        parent_physics = getattr(parent_node.state, "physics", None)
        radiation = _child_radiation_adapter(
            exp, data, child_dc, state, lat, lon,
            radiation_workspace=(
                model._activation_context or {}).get("radiation_workspace"),
            radiation_parent=(None if parent_physics is None
                              else parent_physics.radiation_callable))
    else:
        radiation = radiation_factory(child_dc, state, lat, lon)
    if landuse_attrs is None:
        geog_selection = GeogSelection.from_case_data(
            data, domain_id=int(child_dc.grid_id))
        attrs = geog_selection.landuse_global_attrs()
    else:
        attrs = dict(landuse_attrs)
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"],
        snow=land.get("snow", 0.0), xice=land.get("xice", 0.0),
        valid_time=now,
        cen_lat=float(getattr(grid, "cen_lat", np.mean(lat))),
        mminlu=str(attrs["MMINLU"]),
        iswater=int(attrs["ISWATER"]),
        islake=int(attrs["ISLAKE"]),
        isice=int(attrs["ISICE"]),
        soil_temperature=land.get("tslb"))
    driver = initialize_physics(
        state, cfg, landuse=landuse,
        tsk=land.get("tsk", 300.0),
        soil_temperature=land.get("tslb", 285.0),
        soil_moisture=land.get("smois", 0.30),
        liquid_moisture=land.get("sh2o"),
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=static["TMN"],
        xice=land.get("xice", 0.0), snow=land.get("snow", 0.0),
        snow_depth=land.get("snowh", 0.0),
        sst=land.get("tsk"),
        glw=declared_constant_glw(exp),
        radiation=radiation, radiation_start_time=exp.start_time,
        radiation_latitude=lat, radiation_longitude=lon)
    driver.fields["snoalb"][...] = cp.asarray(
        noah_initial_snow_albedo(
            static["SNOALB"], static["LU_INDEX"], driver.noah_params,
            rdmaxalb=cfg.rdmaxalb),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    # Continuation fields the constructor does not take arrive by
    # direct overwrite -- the supplied arrays, on the whole child.
    for name in ("canwat", "snowc", "snotime", "qsfc", "ust",
                 "smcrel", "psfc", "t2", "q2", "th2", "u10", "v10"):
        target = driver.fields.get(name)
        value = land.get(name)
        if target is not None and value is not None:
            target[...] = cp.asarray(value, dtype=target.dtype)
    return _time.perf_counter() - started


class RealRelocationChildPreparer:
    """Physics rebuild + land-surface continuation for a relocated child.

    This is the ``on_child_built`` the real-data route hands the
    :class:`gpuwm.core.relocation_runner.RelocationRunner`, plus the two
    duck-typed seams the runner drives around it:

    * ``capture_outgoing(node)`` -- before the move, while the outgoing
      child is whole: snapshot the driver-held land-surface continuation
      state (:data:`~gpuwm.ingest.relocation_init.
      LAND_SURFACE_CONTINUATION_FIELDS`) and keep a reference to the
      outgoing footprint's statics.
    * ``__call__(initialized, new_dc, parent_node)`` -- the rebuild:
      FIRST assert the footprint-rebuilt statics equal the outgoing
      child's bitwise on shared ground (identical source + identical
      cells = identical bytes; any mismatch is a statics-build defect
      and refuses), then build the donor-fill plan, move the land state
      (overlap by index-space transplant, strip from nearest
      same-landmask-class donors), and rebuild the physics driver
      against the NEW statics through the same ``initialize_landuse`` /
      ``initialize_physics`` / radiation wiring the t=0 child preparer
      uses.  Accumulators are re-initialised, per leg 1's contract, and
      the receipt says so.
    * ``after_move(node)`` -- once the node carries the new placement:
      refresh the run's prepared-case bookkeeping and the domain's
      wrfout metadata/global attributes so later frames describe the
      footprint that produced them.
    """

    def __init__(self, *, exp: ExperimentConfig, data: CaseDataConfig,
                 model):
        self.exp = exp
        self.data = data
        self.model = model
        self.writers = None
        self.last_receipt = None
        self._captured = None
        self._pending_refresh = None

    def attach_writers(self, writers) -> None:
        self.writers = writers

    def capture_outgoing(self, node) -> None:
        from gpuwm.ingest.relocation_init import (
            LAND_SURFACE_CONTINUATION_FIELDS)

        driver = getattr(node.state, "physics", None)
        fields: dict[str, np.ndarray] = {}
        if driver is not None:
            for name in LAND_SURFACE_CONTINUATION_FIELDS:
                value = driver.fields.get(name)
                if value is not None:
                    fields[name] = _relocation_host(value)
        case = self.model._prepared_by_grid_id.get(int(node.cfg.grid_id))
        self._captured = {
            "grid_id": int(node.cfg.grid_id),
            "i_parent_start": int(node.cfg.i_parent_start),
            "j_parent_start": int(node.cfg.j_parent_start),
            "fields": fields,
            "static_fields": (None if case is None
                              else case.static_fields),
        }

    def __call__(self, initialized, new_dc, parent_node) -> None:
        import time as _time

        from gpuwm.core.nest_relocation import (Placement,
                                                RelocationRefusal,
                                                plan_relocation)
        from gpuwm.ingest.relocation_init import (
            LAND_SURFACE_CONTINUATION_FIELDS, donor_fill_plan,
            overlap_mask_for_plan, overlap_statics_mismatches)

        started = _time.perf_counter()
        captured = self._captured
        self._captured = None
        if captured is None or captured["grid_id"] != int(new_dc.grid_id):
            raise RelocationRefusal(
                "RealRelocationChildPreparer.__call__ without a matching "
                "capture_outgoing: the runner drives both seams, and a "
                "rebuild that never saw the outgoing child has no land "
                "state to move")
        cfg = new_dc.run
        static = initialized.static_fields
        if static is None:
            raise RelocationRefusal(
                "the relocation initializer produced no static fields; "
                "the real-data route requires footprint-rebuilt statics")
        plan = plan_relocation(
            placement_from=Placement(
                grid_id=captured["grid_id"],
                i_parent_start=captured["i_parent_start"],
                j_parent_start=captured["j_parent_start"]),
            placement_to=Placement(
                grid_id=int(new_dc.grid_id),
                i_parent_start=int(new_dc.i_parent_start),
                j_parent_start=int(new_dc.j_parent_start),
                generation=1),
            parent_grid_ratio=int(new_dc.parent_grid_ratio),
            child_nx=int(cfg.nx), child_ny=int(cfg.ny))

        # THE LOAD-BEARING ASSERTION (Drew's design ruling): statics
        # rebuilt from the same source over the same cells must equal the
        # outgoing child's bitwise on shared ground, or the bitwise
        # overlap transplant sits on ground that changed under it.
        if captured["static_fields"] is None:
            raise RelocationRefusal(
                "the outgoing child's statics are not on record, so the "
                "overlap-statics equality cannot be asserted; a move "
                "whose load-bearing claim cannot be checked is refused")
        statics_verdict = overlap_statics_mismatches(
            captured["static_fields"], static, plan)
        if not statics_verdict["pass"]:
            raise RelocationRefusal(
                "footprint-rebuilt statics differ from the outgoing "
                "child's on shared ground (identical source + identical "
                "cells must give identical bytes); this is a statics-"
                "build defect, not an input error: "
                f"{statics_verdict['mismatched_fields'] or statics_verdict}")

        overlap = overlap_mask_for_plan(plan, (cfg.ny, cfg.nx))
        fill = donor_fill_plan(overlap_mask=overlap,
                               landmask=np.asarray(static["LANDMASK"]))
        moved: dict[str, np.ndarray] = {}
        for name, old in captured["fields"].items():
            window = plan.window(old.shape)
            if window is None:
                continue
            (dst_j, src_j), (dst_i, src_i) = window
            staged = np.zeros_like(old)
            staged[..., dst_j, dst_i] = old[..., src_j, src_i]
            moved[name] = fill.apply(staged)

        driver_seconds = self._rebuild_driver(
            initialized, new_dc, parent_node, moved)
        self._pending_refresh = (int(new_dc.grid_id), initialized.grid,
                                 static)
        self.last_receipt = {
            "overlap_statics": {
                "compared_cells": statics_verdict["compared_cells"],
                "mismatched_fields": statics_verdict["mismatched_fields"],
                "pass": statics_verdict["pass"],
            },
            "donor_fill": dict(fill.counts),
            "fields_moved": sorted(moved),
            "fields_absent": sorted(
                set(LAND_SURFACE_CONTINUATION_FIELDS)
                - set(captured["fields"])),
            "accumulators_reinitialized": True,
            "driver_rebuild_seconds": driver_seconds,
            "preparer_seconds": _time.perf_counter() - started,
        }

    def _rebuild_driver(self, initialized, new_dc, parent_node,
                        moved) -> float:
        return rebuild_child_driver_from_land_state(
            exp=self.exp, data=self.data, model=self.model,
            initialized=initialized, child_dc=new_dc,
            parent_node=parent_node, land=moved)

    def after_move(self, node) -> None:
        import dataclasses

        pending = self._pending_refresh
        self._pending_refresh = None
        if pending is None:
            return
        grid_id, grid, static = pending
        case = self.model._prepared_by_grid_id.get(grid_id)
        if case is not None and dataclasses.is_dataclass(case):
            self.model._prepared_by_grid_id[grid_id] = dataclasses.replace(
                case, grid=grid, static_fields=dict(static))
        if self.writers is not None:
            self.writers.refresh_domain(grid_id, grid=grid,
                                        static_fields=static)


def build_real_relocation_runner(exp: ExperimentConfig,
                                 data: CaseDataConfig, model, outdir):
    """Wire the real-data route's RelocationRunner, or ``None``.

    ``None`` when the config names no follow source -- bounds-only
    ``[relocation]`` stays the manual/API mechanism it always was.  With
    a follow source, this is what makes the front-door refusal
    unnecessary on THIS route: the footprint-rebuilt statics initializer
    (:func:`gpuwm.ingest.relocation_init.real_relocation_initializer`)
    and the physics preparer (:class:`RealRelocationChildPreparer`) both
    exist here, because the route holds the input catalog with the
    static source.  Routes without a static source (the prepared domain
    tree) keep their refusal.
    """
    relocation = exp.relocation
    if not (relocation.enabled and (relocation.follow is not None
                                    or relocation.moves)):
        return None
    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.ingest.relocation_init import (
        REAL_DATA_FOOTPRINT_REBUILT_STATICS, real_relocation_initializer)

    grid_id = int(relocation.grid_id)
    nodes = getattr(model, "nodes_by_grid_id", None)
    if nodes is not None and grid_id not in nodes:
        # The follow target is a DORMANT nest: it has no node, no grid
        # and no birth footprint yet, so there is nothing to anchor the
        # placement-translated statics initializer on.  The leg walk
        # rebuilds this the moment the nest is born, which is also the
        # first instant it could legally move.
        return None
    node = model.node(grid_id)
    child_config = node.cfg
    initializer = real_relocation_initializer(
        catalog=model._input_catalog, vertical=exp.vertical,
        child_config=child_config, reference_grid=node.grid,
        reference_i_parent_start=child_config.i_parent_start,
        reference_j_parent_start=child_config.j_parent_start)
    preparer = RealRelocationChildPreparer(exp=exp, data=data, model=model)
    return RelocationRunner.from_experiment(
        exp, schedule=model.schedule, on_child_built=preparer,
        initializer=initializer,
        static_provenance=REAL_DATA_FOOTPRINT_REBUILT_STATICS,
        receipts_path=Path(outdir) / "relocation_receipts.json")


class PreparedTreeRelocationChildPreparer(RealRelocationChildPreparer):
    """The prepared tree route's relocation preparer.

    The capture / overlap-statics assertion / donor-fill machinery is
    the case-data preparer's, inherited unchanged -- the two routes
    differ only in the seams the case route derives from its
    ``CaseDataConfig``:

    * the driver rebuild uses the NATIVE land-use identity the prepared
      route already binds at t=0 and lets ``initialize_physics`` build
      radiation from the RunConfig (byte-for-byte the t=0
      ``initialize_prepared_physics`` wiring), then re-attaches the
      shared radiation workspace exactly as the tree runner does at
      start;
    * ``after_move`` refreshes the tree runner's SimpleNamespace
      prepared-case bookkeeping (the case route's is a dataclass), so
      the NEXT move's overlap-statics assertion holds the rebuilt
      footprint's statics, not the t=0 crop.
    """

    def __init__(self, *, exp: ExperimentConfig, model,
                 radiation_workspace=None):
        self.exp = exp
        self.data = None
        self.model = model
        self.writers = None
        self.last_receipt = None
        self._captured = None
        self._pending_refresh = None
        self._radiation_workspace = radiation_workspace

    def _rebuild_driver(self, initialized, new_dc, parent_node,
                        moved) -> float:
        from gpuwm.native_wrf_contract import NATIVE_LANDUSE_IDENTITY

        seconds = rebuild_child_driver_from_land_state(
            exp=self.exp, data=None, model=self.model,
            initialized=initialized, child_dc=new_dc,
            parent_node=parent_node, land=moved,
            landuse_attrs=dict(NATIVE_LANDUSE_IDENTITY),
            radiation_factory=lambda _dc, _state, _lat, _lon: None)
        driver = getattr(initialized.state, "physics", None)
        radiation = (None if driver is None
                     else driver.radiation_callable)
        if radiation is not None and self._radiation_workspace is not None:
            radiation.column_chunk = self._radiation_workspace.column_chunk
            radiation.chunk_workspace = self._radiation_workspace
        return seconds

    def after_move(self, node) -> None:
        pending = self._pending_refresh
        self._pending_refresh = None
        if pending is None:
            return
        grid_id, grid, static = pending
        case = self.model._prepared_by_grid_id.get(grid_id)
        if case is not None:
            # The tree runner's bookkeeping is a mutable SimpleNamespace;
            # refresh in place so the next capture_outgoing snapshots the
            # statics the child actually sits on.
            case.static_fields = dict(static)
            case.geog_selection = None
        if self.writers is not None:
            self.writers.refresh_domain(grid_id, grid=grid,
                                        static_fields=static)


def build_prepared_tree_relocation_runner(exp: ExperimentConfig, *,
                                          statics_corridor, model, outdir,
                                          radiation_workspace=None):
    """Wire the prepared tree route's RelocationRunner, or ``None``.

    The prepared-route counterpart of
    :func:`build_real_relocation_runner`: same runner, same initializer,
    same preparer seams -- only the statics source differs (the sealed
    corridor crop instead of a per-footprint GEOG build), which is what
    lifts this route's follow-source refusal WHEN a verified corridor is
    on hand.  ``None`` for bounds-only ``[relocation]`` exactly as on
    the case-data route.
    """
    relocation = exp.relocation
    if not (relocation.enabled and (relocation.follow is not None
                                    or relocation.moves)):
        return None
    if statics_corridor is None:
        raise ValueError(
            "build_prepared_tree_relocation_runner requires the verified "
            "statics corridor; the preflight refusal owns the "
            "corridor-less case and must not be bypassed here")
    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.ingest.relocation_init import real_relocation_initializer
    from gpuwm.static.corridor import (CORRIDOR_REBUILT_STATICS,
                                       corridor_footprint_statics_builder)

    node = model.node(int(relocation.grid_id))
    child_config = node.cfg
    initializer = real_relocation_initializer(
        vertical=exp.vertical, child_config=child_config,
        reference_grid=node.grid,
        reference_i_parent_start=child_config.i_parent_start,
        reference_j_parent_start=child_config.j_parent_start,
        statics_builder=corridor_footprint_statics_builder(
            statics_corridor))
    preparer = PreparedTreeRelocationChildPreparer(
        exp=exp, model=model, radiation_workspace=radiation_workspace)
    return RelocationRunner.from_experiment(
        exp, schedule=model.schedule, on_child_built=preparer,
        initializer=initializer,
        static_provenance=CORRIDOR_REBUILT_STATICS,
        receipts_path=Path(outdir) / "relocation_receipts.json")


class RealSpawnChildPreparer:
    """Physics/land attachment for a NEWBORN nest on the real-data route.

    The ``on_child_built`` the route hands
    :class:`gpuwm.core.spawn_runner.SpawnRunner`.  It is the same seam,
    and the same rule, as every leg boundary and every relocation: the
    initializer never invents driver state, the route re-initialises it
    here.

    WHERE THE LAND STATE COMES FROM.  A newborn has no prior self to
    continue from, so unlike the relocation preparer there is nothing to
    capture; and it has no ``real.exe`` product at a footprint nobody knew
    about until the trigger fired, so unlike the t = 0 child preparer
    there is no analysis-derived soil either.  What it does have is the
    live parent, and that is precisely the case WRF's own nest
    initialization is built for: ``med_nest_initial`` fills the whole
    fine grid from ``med_interp_domain(parent, nest)`` BEFORE any input
    file is consulted, and for a nest without one that interpolation is
    the initialization (Users' Guide chapter 5; share/mediation_integrate
    .F:670).  So the land state is
    :func:`~gpuwm.ingest.nest_spawn_init.spawn_land_state_from_parent` --
    the Registry's own masked surface interpolator, run against the
    newborn's OWN-GRID land-use categories -- and the driver rebuild is
    then the shared :func:`rebuild_child_driver_from_land_state`, byte
    for byte the sequence a relocation runs.

    ``last_receipt`` is the duck-typed seam the runner reads (the
    relocation runner's idiom), so the land accounting reaches the spawn
    receipt instead of dying here.
    """

    def __init__(self, *, exp: ExperimentConfig, data: CaseDataConfig,
                 model):
        self.exp = exp
        self.data = data
        self.model = model
        self.prepared_by_grid_id: dict[int, object] = {}
        self.last_receipt = None

    def __call__(self, initialized, child_dc, parent_node) -> None:
        import time as _time

        from gpuwm.ingest.nest_spawn_init import (SpawnInitRefusal,
                                                  spawn_land_state_from_parent)

        started = _time.perf_counter()
        grid_id = int(child_dc.grid_id)
        static = initialized.static_fields
        if static is None:
            raise SpawnInitRefusal(
                "the spawn initializer produced no static fields; the "
                "real-data route requires own-grid statics at the fired "
                "footprint, and the masked land interpolator has no "
                "destination land-use categories without them")
        parent_grid_id = int(parent_node.cfg.grid_id)
        parent_case = self.model._prepared_by_grid_id.get(parent_grid_id)
        parent_static = getattr(parent_case, "static_fields", None)
        if parent_static is None:
            raise SpawnInitRefusal(
                f"the parent d{parent_grid_id:02d} has no statics on "
                "record, so its land-use categories -- the SOURCE mask of "
                "WRF's masked surface interpolator -- are unavailable; a "
                "newborn's land state cannot be interpolated without them")
        geog_selection = GeogSelection.from_case_data(
            self.data, domain_id=grid_id)
        land = spawn_land_state_from_parent(
            child_dc, parent_node, static_fields=static,
            parent_static_fields=parent_static,
            landuse_attrs=geog_selection.landuse_global_attrs())
        driver_seconds = rebuild_child_driver_from_land_state(
            exp=self.exp, data=self.data, model=self.model,
            initialized=initialized, child_dc=child_dc,
            parent_node=parent_node, land=land["fields"])
        context = self.model._activation_context or {}
        snow = land["fields"].get("snow")
        # The newborn's own "initial result" IS the materialized child:
        # the wrfout writers read `initial_result.coord` off the prepared
        # case, and the parent-frame coordinate the SINT fill produced is
        # the one this domain will integrate on.  There is no analysis
        # product to point at, and inventing one would be a lie.
        prepared = PreparedRealCase(
            cfg=child_dc.run, grid=initialized.grid,
            static_fields=dict(static), initial_result=initialized,
            final_analysis=None,
            initial_snow_water_kgm2=(
                np.zeros(tuple(initialized.grid.latlon_mass()[0].shape),
                         dtype=np.float64) if snow is None
                else np.array(snow, dtype=np.float64, copy=True)),
            forcing_times=tuple(context.get("forcing_times", ())),
            geog_selection=geog_selection)
        self.prepared_by_grid_id[grid_id] = prepared
        self.last_receipt = {
            "land_surface": land["receipt"],
            "accumulators_reinitialized": True,
            "driver_rebuild_seconds": driver_seconds,
            "preparer_seconds": _time.perf_counter() - started,
        }


def build_real_spawn_runner(exp: ExperimentConfig, data: CaseDataConfig,
                            model, outdir):
    """Wire the real-data route's SpawnRunner, or ``None``.

    ``None`` when no ``[[domain]]`` declares ``spawn``.  This is what
    lifts the front-door refusal on THIS route and only here: the route
    holds the input catalog, so the newborn's own-grid statics can be
    built at the fired footprint, and it holds the case data and forcing
    calendar, so its physics driver can be attached.  Routes without
    them keep the refusal (gpuwm.experiment.refuse_unrouted_spawn).
    """
    from gpuwm.core.spawn_runner import SpawnRunner

    preparer = RealSpawnChildPreparer(exp=exp, data=data, model=model)

    def statics_provider(child_dc, parent_node):
        from gpuwm.ingest.nest_spawn_init import prepare_spawn_statics

        return prepare_spawn_statics(
            child_dc, parent_node, model._input_catalog,
            valid_date=exp.start_time)

    return SpawnRunner.from_experiment(
        exp, on_child_built=preparer, statics_provider=statics_provider,
        receipts_path=Path(outdir) / "spawn_receipts.json")


def _tree_forcing_cadence_seconds(catalog) -> float:
    """The tree builder's own LBC cadence, so a rebuilt leg clock matches.

    Imported rather than re-derived: ``resolve_clock`` must see exactly
    the interval ``build_experiment`` gave it, or a leg boundary would
    quietly re-phase the root's external-boundary calendar.
    """
    from gpuwm.core.model import _forcing_cadence_seconds

    return _forcing_cadence_seconds(catalog)


def _spawn_leg_seconds(exp: ExperimentConfig) -> float:
    """How often the walk stops to ask whether a nest should be born.

    Coarse on purpose.  Every boundary costs one schedule rebuild, and a
    boundary is only USEFUL where the trigger could newly fire, so this
    takes the relocation cadence when the config sets one (the same
    instant the tracker is already consulted, and already validated as a
    whole number of root steps), else the root's history interval, which
    is also where the reflectivity signal is stashed.
    """
    cadence = getattr(getattr(exp, "relocation", None),
                      "cadence_seconds", None)
    if cadence:
        return float(cadence)
    history = float(getattr(exp.root, "history_interval_s", 0.0) or 0.0)
    return history if history > 0.0 else float(exp.run_seconds)


def _retarget_tree_schedule(model, active_exp: ExperimentConfig,
                            end_seconds: float, lbc_interval_s) -> None:
    """Re-aim the live tree at ``end_seconds`` over ``active_exp``.

    The leg-boundary schedule surgery.  Clocks are minted fresh from the
    new domain set and then carried to the tick the tree is actually at,
    exactly as ``restore_tree_restart`` does across a checkpoint -- the
    executor derives its resume period from the root clock, so a tree
    whose clocks read the boundary resumes there rather than replaying.
    A node with no clock is a newborn: it joins at the boundary.
    """
    from dataclasses import replace as _replace

    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.state import refresh_model_time

    leg_exp = _replace(
        active_exp, run_seconds=float(end_seconds),
        domains=tuple(
            _replace(dc, run=_replace(dc.run, run_seconds=float(end_seconds)))
            for dc in active_exp.domains))
    tick_clock = resolve_clock(leg_exp, lbc_interval_s=lbc_interval_s)
    schedule = build_schedule(leg_exp, tick_clock)
    fresh = tick_clock.clocks()
    boundary_ticks = (0 if model.root.clock is None
                      else int(model.root.clock.ticks))
    for node in model.walk_parent_first():
        gid = int(node.cfg.grid_id)
        old = node.clock
        new = fresh[gid]
        ticks = boundary_ticks if old is None else int(old.ticks)  # noqa: E501
        new.ticks = ticks
        new.step_count = max(
            0, (ticks - new.spec.start_ticks) // new.spec.step_ticks)
        if old is not None and getattr(old, "dtbc_fp32", None) is not None:
            new.dtbc_fp32 = old.dtbc_fp32
        node.clock = new
        if old is None:
            refresh_model_time(node.state, new)
    model.schedule = schedule


def _attach_spawned_children(model, active_exp, record, writers,
                             preparer, lbc_interval_s,
                             coupler_factory=None) -> list[int]:
    """Give each newborn its node, clock, coupler, prepared case, writer.

    The clock is minted over the ACTIVATED domain set and carried to the
    birth tick.  ``_retarget_tree_schedule`` re-mints every clock on the
    next leg anyway, but a DomainNode is not constructible without one,
    and the newborn must read its own birth instant from the moment it
    exists rather than from the next boundary.
    """
    from types import MappingProxyType

    from gpuwm.core.clock import resolve_clock
    from gpuwm.core.model import DomainNode
    from gpuwm.core.state import refresh_model_time

    if coupler_factory is None:
        from gpuwm.core.nest import NestCoupler as coupler_factory

    boundary_ticks = int(model.root.clock.ticks)
    fresh = resolve_clock(active_exp, lbc_interval_s=lbc_interval_s).clocks()
    nodes = dict(model.nodes_by_grid_id)
    attached: list[int] = []
    for gid, child_result in sorted(record["child_results"].items()):
        gid = int(gid)
        child_dc = active_exp.domain(gid)
        parent = nodes[int(child_dc.parent_id)]
        clock = fresh[gid]
        clock.ticks = boundary_ticks
        clock.step_count = max(
            0, (boundary_ticks - clock.spec.start_ticks)
            // clock.spec.step_ticks)
        node = DomainNode(
            cfg=child_dc, grid=child_result.grid, state=child_result.state,
            clock=clock, parent=parent, children=[], coupler=None)
        node.coupler = coupler_factory(node, feedback=active_exp.feedback)
        node._started = True
        node.state._nest_restart_classification = "REBUILT"
        refresh_model_time(node.state, clock)
        parent.children.append(node)
        nodes[gid] = node
        # The writers and any prepared-case consumer read the tree, so it
        # has to carry the newborn before they are touched.
        model.nodes_by_grid_id = MappingProxyType(nodes)
        prepared = preparer.prepared_by_grid_id[gid]
        model._prepared_by_grid_id[gid] = prepared
        if writers is not None:
            writers.add_domain(gid, grid=node.grid,
                               static_fields=prepared.static_fields)
        attached.append(gid)
    model.nodes_by_grid_id = MappingProxyType(nodes)
    return attached


def _exchange_consumer_planes(model, steppers, direction: str,
                              names) -> list[int]:
    """Move ONE consumer's whole-domain planes between store and state.

    A whole-domain model consumer -- the spawn trigger
    (:class:`gpuwm.core.nest_spawn.SpawnWatch`) and the follow tracker
    (:class:`gpuwm.core.storm_tracking.StormTracker`) -- reads ONE plane off
    ``parent_state`` through ``storm_tracking.signal_plane``, which is
    ``state.existing_scratch(slot)``.  A streamed domain's arrays live in its
    store and its ``DomainState`` stops changing at attach, so that read
    returns the plane the state was allocated with: for the UH windows, zeros
    (state.py:720), for the whole run, with no error.

    Both directions are needed and they are not symmetric bookkeeping:
    ``publish`` is how the consumer sees the domain, ``adopt`` is how the
    domain sees that the consumer zeroed its window.  Cheap by construction
    -- one ``(ny, nx)`` plane per streamed domain per LEG boundary, not per
    step, and nothing at all when no domain streams.

    ``names`` is THIS consumer's slots and no one else's, on the same
    reasoning that gave the two consumers separate windows in the first place
    (Drew's ruling, 2026-08-07): the relocation runner resets the follow
    window on its own cadence, from inside ``execute_experiment``, and a
    spawn boundary that published or adopted that window as well could undo a
    reset the tracker had already made or hand it a window measured against a
    boundary it does not own.
    """
    if not steppers:
        return []
    from gpuwm.core import streaming as _streaming

    names = tuple(names)
    touched: list[int] = []
    for gid, stepper in sorted(steppers.items()):
        if not _streaming.is_streaming(stepper):
            continue
        node = model.nodes_by_grid_id.get(int(gid))
        if node is None:
            continue
        # Asked of the streaming module rather than looked up here: this
        # file is not a sanctioned scratch-API site and
        # tests/test_uh_lifecycle.py's roster is right to say so.  The
        # duck-typing (a reduced state, a test double, nwp_diagnostics = 0)
        # is inside allocated_planes.
        present = _streaming.allocated_planes(node.state, names)
        if not present:
            # nwp_diagnostics = 0 allocates no window at all, so there is no
            # plane for any consumer to read and nothing to move.
            continue
        getattr(stepper, direction)(present)
        touched.append(int(gid))
    return touched


def _spawn_consumer_planes() -> tuple[str, ...]:
    """The spawn trigger's own slot, as a streaming manifest key."""
    from gpuwm.core.uh_diag import UH_SPAWN_WINDOW_SLOT

    return (f"scratch/{UH_SPAWN_WINDOW_SLOT}",)


def _publish_consumer_planes(model, steppers, names) -> list[int]:
    return _exchange_consumer_planes(model, steppers, "publish", names)


def _adopt_consumer_planes(model, steppers, names) -> list[int]:
    return _exchange_consumer_planes(model, steppers, "adopt", names)


def _adjudicate_newborn_steppers(steppers, model, attached, factory):
    """Bind (or refuse) a stepper for every domain born this boundary.

    ``gpuwm.core.streaming.steppers_for_tree`` walks the tree ONCE, before
    the run, and returns ``{grid_id: stepper}``; the executor resolves a
    missing grid to ``dycore.step`` (model.py's "delayed-start child" note,
    written before streaming existed).  For a DELAYED-START child that is
    right: the domain was in the tree when the mapping was built and was
    adjudicated then, it simply had not started yet.  For a SPAWNED child it
    is not: that domain was not in the tree at all when the mapping was
    built, so nothing ever asked whether it should stream.

    The two failure modes of the silent fallback are opposite and both bad.
    A big newborn that should have streamed dies at the resident allocation
    the mode was turned on to avoid -- after hours of integration, at the
    one instant the run cannot be restarted from.  A small newborn that
    should not have streamed runs correctly, which is worse, because the
    run then certifies a spawn path that has never once been exercised
    under the mode its config says it is in.

    So: with streaming engaged (a non-empty mapping is the only thing that
    says so -- ``steppers_for_tree`` returns ``{}`` when ``[tiles]`` is
    absent AND when ``auto`` decides every domain fits), a newborn either
    gets an adjudicated stepper from the route's factory, or the run
    refuses HERE, naming the grid.  Never a silent fallthrough.
    """
    if not steppers or not attached:
        return steppers
    if factory is None:
        named = ", ".join(f"d{int(gid):02d}" for gid in sorted(attached))
        raise RuntimeError(
            f"[tiles] is engaged for this run ({len(steppers)} domain(s) "
            f"stream) and {named} was born at a spawn boundary, after the "
            "stepper mapping was built.  gpuwm.core.streaming"
            ".steppers_for_tree walks the tree once, before the run, so a "
            "domain that joins it later is not in the mapping and the "
            "executor would resolve it to gpuwm.core.dycore.step -- "
            "integrating a newborn RESIDENT inside a streamed run, with "
            "nothing in the log to say so.  This route must pass "
            "spawned_stepper_factory (grid_id, node) -> stepper | None so a "
            "newborn is adjudicated the same way its siblings were.")
    out = dict(steppers)
    for gid in sorted(int(g) for g in attached):
        stepper = factory(gid, model.node(gid))
        if stepper is not None:
            out[gid] = stepper
    return out


def walk_spawn_legs(model, exp: ExperimentConfig, data: CaseDataConfig, *,
                    spawn_runner, writers, lbc_interval_s,
                    relocation_runner=None, relocation_runner_factory=None,
                    coupler_factory=None, spawned_stepper_factory=None,
                    **execute_kwargs):
    """Integrate the run as LEGS so dormant nests can be born mid-run.

    The production consumer of
    :meth:`gpuwm.core.spawn_runner.SpawnRunner.on_leg_boundary`.  While
    a watch is still pending the walk stops every
    :func:`_spawn_leg_seconds`, asks, and either continues or activates;
    once nothing is pending it runs straight to the end in one leg, so
    the common shape is "a few cheap root-only legs, then the whole rest
    of the forecast".

    Restart across a spawn boundary inherits the moving-nest posture
    whole: it promises nothing (Drew, 2026-08-06).

    ``spawned_stepper_factory(grid_id, node) -> stepper | None`` adjudicates
    a NEWBORN's execution mode.  It is consulted only when this run is
    actually streaming something, and its absence in that case is a refusal
    rather than a fallthrough -- see :func:`_adjudicate_newborn_steppers`
    for the two ways the silent version goes wrong.  Every parent's stepper
    is left alone: a :class:`~gpuwm.core.streaming.StreamedDomain` owns the
    domain's arrays in its store and its tile buffers, and re-attaching one
    at a leg boundary would re-copy the ATTACH-TIME state over the store and
    silently discard every step the run has taken.
    """
    from gpuwm.core.model import execute_experiment

    leg = _spawn_leg_seconds(exp)
    total = float(exp.run_seconds)
    tol = 1.0e-9
    while True:
        elapsed = float(model.root.clock.elapsed_seconds)
        if spawn_runner.pending and elapsed + tol < total:
            boundary = min(total, (math.floor(elapsed / leg) + 1) * leg)
        else:
            boundary = total
        _retarget_tree_schedule(
            model, spawn_runner.active, boundary, lbc_interval_s)
        # The relocation runner watches ONE grid; while that grid is
        # still dormant it is not in the tree, and consulting it would
        # ask the model for a node that does not exist.
        runner = relocation_runner
        if runner is not None:
            target = getattr(runner.config, "grid_id", None)
            if target is not None and int(target) not in model.nodes_by_grid_id:
                runner = None
        execute_experiment(
            model, relocation_runner=runner, **execute_kwargs)
        elapsed = float(model.root.clock.elapsed_seconds)
        if elapsed + tol >= total:
            break
        # The boundary instant belongs to BOTH legs: the leg that just
        # ended emitted its history there, and the next leg's pre-loop
        # emit would publish the same instant again -- which for a
        # microphysics domain also means consuming a one-frame REFL
        # handoff that no longer exists.  This is the resume-boundary
        # ownership problem the checkpoint path already solves, so it is
        # solved the same way: mark exactly the domains whose frame is
        # already durable, and the next leg suppresses them one domain at
        # a time.
        model._resumed = True
        model._resume_committed_history_grid_ids = frozenset(
            gid for gid, node in model.nodes_by_grid_id.items()
            if node.clock.history_due())
        # The trigger reads the parent's WHOLE-DOMAIN plane off
        # ``node.state``; a streamed parent's arrays live in its store and
        # its state stopped changing at attach.  Publishing here -- once per
        # leg boundary, only the planes a consumer reads -- is what makes the
        # watch see the running domain instead of the attach-time zeros.
        planes = _spawn_consumer_planes()
        _publish_consumer_planes(model, execute_kwargs.get("steppers"), planes)
        record = spawn_runner.on_leg_boundary(model, t=elapsed)
        # The runner zeroed each parent's window on the STATE (its
        # "max since I last looked" reset).  The domain is the store, so the
        # zeroing has to reach it or the next fold accumulates on top of a
        # window the consumer already believes it spent.
        _adopt_consumer_planes(model, execute_kwargs.get("steppers"), planes)
        if record is not None:
            attached = _attach_spawned_children(
                model, record["experiment"], record,
                writers, spawn_runner.on_child_built, lbc_interval_s,
                coupler_factory)
            execute_kwargs["steppers"] = _adjudicate_newborn_steppers(
                execute_kwargs.get("steppers"), model, attached,
                spawned_stepper_factory)
            # A follow target that was dormant could not be wired at
            # build time; now that it exists, it can follow its storm.
            if relocation_runner is None and (
                    relocation_runner_factory is not None):
                relocation_runner = relocation_runner_factory()
                if relocation_runner is not None and writers is not None:
                    attach = getattr(relocation_runner.on_child_built,
                                     "attach_writers", None)
                    if callable(attach):
                        attach(writers)
    spawn_runner.close_receipt(model)


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
        coord=None, feedback=None, initial_condition=None,
        source: str | None = None) -> dict[str, object]:
    """Assemble one domain's wrfout global attributes.

    ``initial_condition`` is the preparation receipt's provenance block
    (see :func:`gpuwm.io.wrfout.initial_condition_global_attrs`).  It
    says WHAT the initial state was; ``start_time`` says only WHEN the
    model clock began, and at a nonzero forecast lead those are two
    different facts about two different times.  ``None`` writes no
    provenance attribute at all rather than asserting an analysis.
    """
    from gpuwm.io.wrfout import (
        initial_condition_global_attrs, wrf_global_attrs)

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
    # ``run`` carries the resolved physics selectors, which are what let a
    # reader tell "no shallow-cumulus scheme exists" from "one ran and
    # produced nothing".
    attrs = wrf_global_attrs(
        grid, start_time, landuse_attrs=landuse_attrs,
        run=(None if domain is None else getattr(domain, "run", domain)),
        **identity)
    if feedback is not None:
        attrs.update(
            GPUWM_FEEDBACK=str(feedback["feedback"]),
            GPUWM_FEEDBACK_VALUE=np.int32(feedback["feedback_value"]),
            GPUWM_FEEDBACK_STOCK_WRF_CERTIFIED=np.int32(0),
            GPUWM_FEEDBACK_CERTIFICATION=str(
                feedback["stock_wrf_certification"]))
    attrs.update(initial_condition_global_attrs(
        initial_condition, source=source))
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
    if getattr(state, "_streamed_domain", None) is not None:
        # REFUSED, not served.  This is the frozen reference integration
        # loop and its frame is a DIFFERENT frame from the async writer's --
        # `state_frame` order, the grid metadata block, an explicit RAINNC
        # row -- so serving it off the store would mean maintaining a second
        # store-side frame assembly whose only proof of correctness is that
        # somebody kept the two in step.  The production history path
        # (PerDomainWrfoutWriters.submit, which `gpuwm go` and the tree
        # runner both use) publishes streamed domains correctly; this one
        # says so instead of writing t = 0 into every frame, which is what
        # it did before this check existed.
        #
        # "[tiles] host store", not "streaming host store": the product has a
        # second door called `gpuwm stream`, and a refusal is the one place a
        # reader cannot afford to be told about the wrong feature.
        raise RuntimeError(
            "write_case_output was handed a domain whose arrays live in a "
            "[tiles] host store, not on this state.  Every frame after "
            "the cold-start one would be the initial condition with a later "
            "timestamp -- correct inventory, correct Times, no forecast.  "
            "Integrate this case through the experiment route "
            "(PerDomainWrfoutWriters), or publish the frame with "
            "gpuwm.core.streaming.StreamedDomain.history_fields().")
    # initialize_real and every completed dycore step leave p/al/alt current.
    # Re-diagnosing here would make an observational output operation mutate
    # the next step's initial state, so case output is deliberately read-only.
    frame = state_frame(state, include_diagnostic_pressure=True)
    frame.update(_metadata_frame(prepared.grid, prepared.static_fields))
    rainnc = state.physics.microphysics.rainnc
    frame["RAINNC"] = cp.asnumpy(rainnc)
    if (expect_refl_10cm
            and prepared.cfg.mp_physics in REFL_10CM_MICROPHYSICS
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


def _output_committed(progress_callback, *, domain_id: int,
                      valid_time: datetime, path: Path) -> None:
    """Tell an interested callback that one wrfout is durable.

    Same optional-hook convention as :func:`_preparation_progress`
    above: discovered by name, absent means nothing happens, so every
    existing ``progress_callback`` is unaffected.

    The existing ``last_durable_wrfout`` field on the per-step callback
    answers "which file was most recently published", which is what a
    heartbeat needs.  It cannot answer "a file just landed, here is its
    domain and valid time" -- a consumer would have to watch that field
    for changes and re-derive the rest from the filename.  This hook is
    raised at the exact call that published the file, with the three
    facts already in scope there.
    """

    reporter = getattr(progress_callback, "output_committed", None)
    if reporter is not None:
        reporter(domain=domain_id, valid_time=valid_time, path=path)


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


def _reset_streamed_up_heli_max(stepper) -> None:
    """The history-interval UP_HELI_MAX reset, applied to a streamed domain.

    ``reset_up_heli_max(state)`` zeroes the accumulator the frame just
    snapshotted.  On a streamed domain that accumulator is
    ``store["scratch/up_heli_max"]``; the loop's ``state`` holds the
    preparation copy and zeroing it changes nothing the model will read.
    ``up_heli_max`` is a SERIALIZED scratch slot
    (``restart.classify_scratch_slot`` says so), so it is one of the 229
    carriers and it goes into every checkpoint -- which means the omission
    was not merely a wrong diagnostic: it made a streamed checkpoint and a
    resident checkpoint of the same forecast differ, in exactly one member,
    at the first history interval.  A running maximum only ever grows, so
    the symptom is an UP_HELI_MAX window that never resets and a bit
    comparison that fails on one array out of 229 with everything else
    identical -- the most persuasive possible argument that the difference
    is "just a diagnostic" and can be ignored.

    ``None`` (the resident run, where ``stepper is dycore.step``) does
    nothing at all, so the unstreamed path is untouched.
    """
    if stepper is None:
        return
    buffer = stepper.store.get("scratch/up_heli_max")
    if buffer is not None:
        buffer[...] = 0.0


def integrate_prepared_case(
        output_dir, prepared, *, start_time: datetime, output_title: str,
        domain_id: int = 1, integration_cfg: RunConfig | None = None,
        restart_path=None, run_seconds: float | None = None,
        history_interval_s: float | None = None,
        restart_interval_s: float | None = None, progress_callback=None,
        health_debug: bool = False,
        feedback=None, stepper=None) -> RealCaseRunSummary:
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

    ``stepper`` is what one dynamics substep is taken with.  ``None`` binds
    ``gpuwm.core.dycore.step`` ITSELF -- not a wrapper around it -- so a run
    that configures no ``[tiles]`` executes the identical call it always
    did.  ``gpuwm.core.streaming.make_stepper`` returns either that same
    function or a :class:`~gpuwm.core.streaming.StreamedDomain`, which has
    the same signature and advances the same domain by the same substep out
    of a pinned host store, one tile at a time.  The loop around it --
    history cadence, restart cadence, the REFL_10CM handshake -- is not aware
    of the difference and does not need to be.

    THE OBSERVERS ARE ASKED OF THE STEPPER, NOT OF THE STATE.  That is not
    tidiness, it is a correctness fix.  Under ``[tiles] store = "host"``
    the domain lives in a pinned host store and
    ``gpuwm.core.streaming.attach`` fills it with ``gather.pinned_copy``,
    which COPIES: the prepared ``DomainState`` this function holds is a
    snapshot of t = 0 that no sweep ever writes again.  Reducing over it --
    which is what this loop did -- meant ``nan_free`` stayed true forever,
    ``w_max`` froze at its initial value, and a domain that went non-finite
    in the store completed and wrote a checkpoint recording that it had not.
    Keeping the state current instead is not available: the premise of the
    mode is that the domain does not fit on the card.  So the reduction is
    folded per tile inside the sweep and asked of the stepper here, exactly
    as ``dycore.stability_report`` is asked of it when the domain is resident
    -- ``streaming.stability_observer`` returns THAT function itself in that
    case, so a resident run executes the identical call it always did.

    ``StateHealthValidator`` is NOT yet folded.  See the comment at its
    cadence below: it is armed for a resident run and, under a host store, it
    still validates the t = 0 snapshot.
    """
    import cupy as cp
    from gpuwm.core import streaming as _streaming
    from gpuwm.core.health import StateHealthValidator
    # ``dycore.stability_report`` is deliberately NOT imported here: the
    # per-substep gate goes through ``stability_observer``, which returns that
    # function ITSELF for a resident domain and the sweep's per-tile fold of
    # the store for a streamed one.  Importing it anyway would leave the
    # obvious-looking wrong call one keystroke away.
    from gpuwm.core.dycore import step
    from gpuwm.core.streaming import (domain_call_counts, domain_field_max,
                                      is_streaming, stability_observer)
    from gpuwm.io.restart import (restart_filename, restore_restart,
                                  write_restart)
    from gpuwm.supervisor import validate_manifest_checkpoint

    stepper = step if stepper is None else stepper
    # ``gpuwm.core.dycore.stability_report`` ITSELF for a resident domain --
    # the same object, not a wrapper round it, so the resident path has no
    # "streaming disabled" branch that could be subtly different.
    stability_report = stability_observer(stepper)
    # Whether the full-state validator observes the live domain.  The
    # condition is store = "host" specifically, NOT "is streamed": with
    # store = "device" ``attach`` makes the store the DomainState's own
    # arrays, so the sweep writes the very memory the validator reads and it
    # is armed exactly as it always was.
    health_armed = not (is_streaming(stepper)
                        and getattr(stepper, "host_store", False))
    health_validations_unarmed = 0
    if health_debug and not health_armed:
        # "armed under [tiles]", not "under streaming", for the reason
        # write_case_output's refusal above carries: this sentence is read by
        # somebody who configured [tiles], and `gpuwm stream` is a different
        # feature with a prior claim on the other word.
        raise RuntimeError(
            "health_debug asks for a full-state validation every substep, but "
            "this domain is streamed to a host store: StateHealthValidator "
            "reads the resident DomainState and the sweep never writes it, so "
            "every one of those validations would pass regardless of what the "
            "forecast did.  Refused rather than run, because an attribution "
            "mode that cannot attribute is worse than no attribution mode.  "
            "The nan / w_max / CFL gate remains armed under [tiles] -- it "
            "folds the store per tile -- so a streamed run is still guarded, "
            "just not by this.")
    # WHERE THE DOMAIN IS.  Everything below that touches model state has to
    # ask, because a streamed domain's carriers are in the stepper's store
    # and the ``state`` this loop holds has been frozen at its preparation
    # values since ``attach`` copied them out.  The restart is the case
    # where getting it wrong is silent: ``write_restart(state, cfg)`` would
    # produce a complete, self-consistent, fully validating checkpoint of
    # the INITIAL CONDITION stamped with the current clock -- every shape
    # check passes and the file resumes into a forecast that threw away
    # every step taken.  ``is_streaming`` is ``False`` for the unstreamed
    # run, where ``stepper is dycore.step``, so the resident path below is
    # the identical code it always was.
    streamed = is_streaming(stepper)
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
    if is_streaming(stepper) and stepper.host_store:
        # SAID, not discovered.  The stability record is folded per tile and
        # is correct under streaming; this validator is not, and the failure
        # mode of a disarmed gate is that everything looks fine.  An operator
        # who is told loses nothing; an operator who is not gets a forecast
        # whose only whole-state gate has been passing on a snapshot of
        # t = 0.  See the comment at its cadence below for why it cannot
        # simply be pointed at a tile.
        import warnings

        warnings.warn(
            "[tiles] store = 'host': StateHealthValidator is bound to "
            "the prepared DomainState, which the sweep does not write, so "
            "the full-state health gate is NOT armed for this run.  Its "
            "validations are SKIPPED AND COUNTED rather than run, so a run "
            "summary cannot show validations that observed nothing.  The "
            "NaN/w_max/CFL/swdown record IS armed -- it is folded per tile "
            "out of the store (gpuwm.core.streaming.StreamedStability).",
            RuntimeWarning, stacklevel=2)
    # Asked of the STEPPER, not of the dycore: a streamed domain takes the
    # substep as a sweep of tiles and has no single phase to observe, so it
    # declines the hook and the loop falls back to validating between
    # substeps.  With no streaming configured this is the same signature it
    # has always been, because the stepper is the same function.
    phase_hook_supported = "phase_observer" in inspect.signature(
        stepper).parameters
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
        _output_committed(progress_callback, domain_id=domain_id,
                          valid_time=start_time, path=outputs[-1])
        # WRF resets the nwp_diagnostics running maxima each history
        # interval (module_diag_nwp.F:246-269); gpuwm's ratified placement
        # is immediately after the frame is durable.
        from gpuwm.core.uh_diag import reset_up_heli_max
        reset_up_heli_max(state)
        _reset_streamed_up_heli_max(stepper if streamed else None)
    else:
        _preparation_progress(progress_callback, "validate-checkpoint")
        last_checkpoint = validate_manifest_checkpoint(restart_path)
        _preparation_progress(progress_callback, "restore-checkpoint")
        # Into the STORE for a streamed domain, and into the state for a
        # resident one.  Not a preference: the resident reader's first act
        # is to allocate a host copy of every carrier and overwrite the
        # device state with it, and above the card's ceiling that state does
        # not exist.  Below it, the state exists but is not the domain --
        # restoring into it would leave the store holding the PREPARATION
        # values and the run would resume from t=0 with the checkpoint's
        # clock.  The streamed reader applies the same refusals in the same
        # order (config echo, setup fingerprint, physics setup fingerprint,
        # clock admissibility, member classification) plus one the resident
        # reader cannot make: a resuming resident state has every slot
        # allocated by preparation, so it cannot tell a restored carrier
        # from an unrestored one, and the streamed reader requires the
        # file's member set to BE the store's.
        info = (stepper.restore_restart(last_checkpoint, cfg) if streamed
                else restore_restart(last_checkpoint, state, cfg))
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
            refl_due = (cfg.mp_physics in REFL_10CM_MICROPHYSICS
                        and refl_10cm_due(
                            outer_step, substep, output_outer_steps,
                            dynamics_substeps))
            phase = f"outer-{outer_step + 1}.substep-{substep + 1}"
            if health_debug and not phase_hook_supported and health_armed:
                health.require_healthy(phase=phase + ".pre-step")
            step_kwargs = {"refl_10cm_due": refl_due}
            if health_debug and phase_hook_supported:
                step_kwargs["phase_observer"] = health.phase_observer
            stepper(state, integration_cfg, **step_kwargs)
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
                # NOT YET FOLDED, and loud about it above rather than
                # silent here.  ``StateHealthValidator`` is a descriptor
                # kernel over up to 1024 WHOLE fields with one block per
                # descriptor and no windowing, so it cannot be pointed at a
                # tile's interior the way the stability reduction can.  The
                # foldable form is a per-BUFFER validation issued after the
                # gather and before the step -- at that instant the buffer
                # holds exactly the store's bytes, halo included, so the
                # union over tiles covers the domain with no false positives
                # -- but it observes the state one step behind, needs its own
                # readback point inside the sweep, and multiplies the launch
                # by the tile count.  That is a bigger change than a
                # correctness fix should smuggle in, so it is stated, not
                # done.  Under a host store this call still validates the
                # t=0 snapshot -- so under a host store it is SKIPPED
                # AND COUNTED rather than run.  Counting the skips makes an
                # unarmed validator visible in the run summary instead of
                # indistinguishable from a passing one.  The nan / w_max /
                # CFL gate below is a different observer and IS armed: it
                # folds the store.
                if health_armed:
                    health.require_healthy(phase=phase + ".post-step")
                else:
                    health_validations_unarmed += 1
            width = cfg.spec_bdy_width
            # NOT stability_report(state, ...).  Under [tiles] with a host
            # store the domain's arrays are in the store and this state is
            # never written by the sweep, so reading it here reports the
            # condition the store was FILLED from -- healthy at t=0 and
            # healthy forever, which silently disarms the nan gate below and
            # freezes w_max and the CFL at their initial values.  step_health
            # asks the stepper: the dycore's own whole-domain reduction when
            # resident, the sweep's per-tile fold of the store when streamed.
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
        # Both of these are the DOMAIN's, and under a host store the domain
        # is not on ``state``: its call counts live on the sweep's carried
        # clock and its swdown maximum is in the store.
        surface_forcing_updates = domain_call_counts(
            stepper, state)["radiation"]
        # Once per OUTER step over one 2-D field, so the honest fix for the
        # streamed case is to read the store on the host rather than to add
        # another hook inside the sweep: no tile writes swdown's maximum, and
        # a 672x672 float32 plane is 1.8 MB.  Resident runs still take
        # ``cp.max`` over the state, which is what the argument names.
        step_swdown_peak = domain_field_max(
            stepper, state, "fields/swdown", state.physics.fields["swdown"])
        if step_swdown_peak > swdown_peak:
            swdown_peak = step_swdown_peak
            swdown_peak_time = forcing_time
        if (outer_step + 1) % output_outer_steps == 0:
            valid = start_time + timedelta(seconds=(outer_step + 1) * cfg.dt)
            outputs.append(write_case_output(
                prepared, output_dir, valid, start_time=start_time,
                title=output_title, domain_id=domain_id,
                feedback=feedback))
            _output_committed(progress_callback, domain_id=domain_id,
                              valid_time=valid, path=outputs[-1])
            # History-interval reset of the UP_HELI_MAX window (the frame
            # above snapshotted the accumulator synchronously).
            from gpuwm.core.uh_diag import reset_up_heli_max
            reset_up_heli_max(state)
            _reset_streamed_up_heli_max(stepper if streamed else None)
        if (restart_write_steps is not None
                and (outer_step + 1) % restart_write_steps == 0):
            valid = start_time + timedelta(seconds=(outer_step + 1) * cfg.dt)
            checkpoint_path = (
                output_dir / restart_filename(valid, f"d{domain_id:02d}"))
            trackers = {
                "nan_free": nan_free,
                "w_max_ms": float(w_max),
                "w_max_boundary_row": w_max_boundary_row,
                "boundary_w_max_ms": float(boundary_w_max),
                "interior_w_max_ms": float(interior_w_max),
                "swdown_peak_wm2": float(swdown_peak),
                "swdown_peak_time": swdown_peak_time.isoformat(),
            }
            if streamed:
                # From the pinned store, with ZERO device-to-host copies.
                # The resident writer would not fail here -- it would write
                # this loop's ``state``, which streaming froze at t=0 -- so
                # the branch is the difference between a checkpoint and a
                # forgery that passes every check in the reader.
                last_checkpoint = stepper.write_restart(
                    checkpoint_path, cfg, run_trackers=trackers).path
            else:
                last_checkpoint = write_restart(
                    checkpoint_path, state, cfg, run_trackers=trackers)
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

    # After the final device synchronization and after every history frame is
    # durable, so the digest observes the trajectory and cannot join it.
    trajectory_digest = None
    if trajectory_digest_enabled():
        from gpuwm.state_digest import canonical_state_digest

        trajectory_digest = {
            f"d{domain_id:02d}": canonical_state_digest(
                state, _SingleDomainDigestClock(), scope="trajectory"),
            "boundary_clock_provenance": _SingleDomainDigestClock.provenance,
        }

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
    if health_validations_unarmed:
        import warnings

        warnings.warn(
            f"{health_validations_unarmed} full-state health validations were "
            "skipped as unarmed over this streamed run; the per-substep "
            "nan / w_max / CFL gate ran on the store as normal.",
            RuntimeWarning, stacklevel=2)
    return RealCaseRunSummary(
        trajectory_digest=trajectory_digest,
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

def downward_longwave_source(exp: ExperimentConfig, cfg: RunConfig) -> str:
    """One line naming where a domain's downward longwave comes from.

    Written into every resolved-configuration report so that a run
    integrating a CONSTANT GLW says so on the receipt.  A published GLW
    row is indistinguishable from a measured one once it is in a wrfout
    file; this is the sentence that distinguishes them.

    The sentence is derived from
    :func:`gpuwm.physics_compat.downward_longwave_disposition` -- the
    same classification the config-load guard and ``initialize_physics``
    refuse on -- so the receipt can never describe a fate the engine
    does not enact.
    """

    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import (CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                      downward_longwave_disposition)

    lw, sw = radiation_scheme_ids(cfg)
    kind, consumer = downward_longwave_disposition(
        ra_lw_physics=lw, ra_sw_physics=sw,
        sf_surface_physics=int(cfg.sf_surface_physics))
    if kind == "scheme":
        return (f"computed every radiation call by ra_lw_physics={lw} "
                "(radt clock)")
    constant = declared_constant_glw(exp)
    if constant is None:
        if kind == "unused":
            return ("ra_lw_physics=0 and no constant declared -- nothing "
                    "reads or publishes GLW in this suite")
        # Unreachable through build_experiment, whose load guard refuses
        # exactly the consumed/published kinds without the token; stated
        # honestly anyway for a hand-assembled ExperimentConfig.
        return ("NO SOURCE: ra_lw_physics=0 with no constant declared, "
                f"yet GLW is {kind} -- this configuration is refused at "
                "config load and by initialize_physics")
    header = f"DECLARED CONSTANT {constant:g} W m-2, NOT a computed flux: "
    footer = f" (declared by {CONSTANT_DOWNWARD_LONGWAVE_ACK})"
    if kind == "consumed":
        return (header + "ra_lw_physics=0, so no scheme produces downward "
                f"longwave and the land surface ({consumer}) integrates "
                "this one number for the whole forecast" + footer)
    if kind == "published":
        return (header + "ra_lw_physics=0 and no land-surface scheme "
                "reads it, but shortwave keeps the radiation slot active, "
                "so this one number is published as the GLW row of every "
                "wrfout frame" + footer)
    return (header + "declared but UNUSED -- no land-surface scheme reads "
            "it and radiation is off, so it reaches no scheme and no "
            "wrfout row" + footer)


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
    add("radiation.downward_longwave",
        downward_longwave_source(exp, cfg))
    # THE CARRIER POLICY, always in the receipt, both values.  A reader
    # looking for "did this run integrate a sky nobody computed" gets an
    # answer whether or not the escape was taken, which is what makes the
    # answer trustworthy -- an absent line reads as "not applicable" and
    # this question is never not applicable to a run with a land surface.
    add("radiation.surface_radiation_policy",
        f"{cfg.surface_radiation_policy}"
        + ("" if cfg.surface_radiation_policy == "required" else
           " (EXPERIMENTAL FORCING: carriers with no producer are "
           "consumed at their allocation fill; not a valid configuration "
           "for a real case)"))
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
    highres = getattr(data, "static_highres", None)
    if highres is not None:
        for key, value in highres.echo().items():
            add(f"static.highres.{key}", value)
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
    highres = getattr(data, "static_highres", None)
    if highres is not None and getattr(highres, "enabled", False):
        from gpuwm.static.highres_production import apply_highres_statics
        fields, _ = apply_highres_statics(
            fields, grid, config=highres, domain_id=dc.grid_id,
            case_date=exp.start_time.date(),
            landuse_attrs=selection.landuse_global_attrs())
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
    from gpuwm.core.streaming import refuse_unrouted_streaming
    from gpuwm.io.wrfout import quarantine_orphan_wrfouts

    # BEFORE the ingest, on the same governance as the relocation and spawn
    # refusals below.  Neither this route's single-domain arm (which calls
    # integrate_prepared_case with stepper=None) nor either of its tree arms
    # (execute_experiment / walk_spawn_legs, both without steppers=) reads
    # exp.tiles at all -- so a [tiles] block here does not decide
    # anything, it is simply dropped, and the run integrates resident with
    # nothing in the log to say the mode never engaged.  That is the one
    # outcome gpuwm.core.streaming exists to prevent.
    refuse_unrouted_streaming(exp, "gpuwm run", consults_the_seam=False)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if exp.tiles.enabled:
        # [tiles] is SILENTLY IGNORED on this route, and silence is the
        # dangerous direction.  Neither branch below consults it: the
        # single-domain branch calls integrate_prepared_case, whose
        # `stepper` argument no caller in this checkout ever supplies, and
        # the tree branch calls execute_experiment with no `steppers`, so
        # both bind gpuwm.core.dycore.step unconditionally.  A user who
        # asked for out-of-core and got a resident run would find out at
        # the allocation the mode existed to avoid, with nothing in the log
        # to say the mode never engaged -- exactly the failure
        # streaming.make_stepper refuses rather than produces.
        #
        # Named at the front door, before minutes of ingest, on the same
        # precedent as the [relocation] follow-source refusal below.
        raise ValueError(
            "[tiles] is configured but this route cannot honour it: "
            "gpuwm.runtime.run_experiment binds gpuwm.core.dycore.step for "
            "every domain and never consults exp.tiles, so the run "
            "would be RESIDENT with no indication that the mode did not "
            "engage.  The routes that stream are "
            "gpuwm.prepared_single_domain_forecast and "
            "gpuwm.prepared_domain_tree_forecast, which wire "
            "streaming.builders_for_tree; delete the [tiles] table to "
            "run here.")
    follow_configured = exp.relocation.enabled and (
        exp.relocation.follow is not None or exp.relocation.moves)
    if follow_configured and len(exp.domains) == 1:
        # A follow source needs a child to move.  Named at the front
        # door, before minutes of ingest.
        raise ValueError(
            "[relocation] configures a follow source, but this "
            "experiment has a single domain and therefore no nest to "
            "move; remove [relocation.follow]/[[relocation.move]] or "
            "add the child the follow source is for.")
    # Spawn activation is leg-boundary schedule surgery -- build_schedule
    # bakes each domain's activation tick into the per-period op lists, so
    # a trigger-driven domain cannot join a schedule already expanded, and
    # a route that reserves the nest but walks ONE execute_experiment
    # would integrate the parent alone with the nest never born.  The
    # refusal that used to stand here is lifted for the TREE path below,
    # where walk_spawn_legs drives SpawnRunner and this route's catalog
    # supplies the newborn's statics and physics.  A dormant nest is by
    # definition a second domain, so the single-domain path can never
    # carry one; it refuses here rather than reaching a walk that is not
    # wired for it.
    from gpuwm.experiment import dormant_domain_ids
    if dormant_domain_ids(exp) and len(exp.domains) == 1:
        raise ValueError(
            "[[domain]] spawn = {...} on a single-domain experiment: a "
            "dormant nest needs a parent to be born from, and the "
            "single-domain path runs no tree walk.")
    # [tiles] is READ by the experiment loader, validated, echoed into
    # the resolved-config report -- and this route builds no stepper for it,
    # so before this refusal a configured `mode = "on"` integrated the domain
    # RESIDENT and said nothing.  Every other surface in this loader refuses
    # a knob it cannot honour ([relocation] on a single domain, a spawn on a
    # route with no walk) for the same reason: a configuration that silently
    # does nothing is discovered when the run dies at the allocation the
    # mode was turned on to avoid, or -- worse -- does not die and is quietly
    # a different experiment from the one that was asked for.  Streaming
    # needs a per-domain builder (store filled from the prepared state, tile
    # buffers on the domain's own physics selectors, geography inventoried,
    # boundary tables windowed per tile), which `gpuwm.core.streaming
    # .steppers_for_tree` takes as `builders=` and this route does not yet
    # supply.  Until it does, say so.
    if exp.tiles.enabled:
        raise ValueError(
            f"[tiles] mode = {exp.tiles.mode!r} is configured, but "
            "gpuwm.runtime.run_experiment wires no streamed-domain builder "
            "and would have integrated this experiment RESIDENT without "
            "saying so.  [tiles] is reachable today only "
            "through gpuwm.core.streaming.make_stepper with a route-owned "
            "build= callable (see tilestream/test_join.py); remove the "
            "[tiles] block to run resident.")
    # The land-state refusal that used to stand here is LIFTED (2026-08-07).
    # It named one gap -- "a newborn real-data nest has no defined
    # soil/land state, and how it should get one is an open physics
    # decision" -- and that decision is no longer open: it is WRF's, and
    # has been since nesting existed.  A nest with no input file of its
    # own is initialized by interpolating every field it needs from the
    # parent (Users' Guide chapter 5; med_nest_initial's unconditional
    # med_interp_domain, share/mediation_integrate.F:670), with the
    # surface/soil family going through the Registry's landmask-aware
    # interpolator (interp_mask_field:lu_index,iswater) rather than a
    # plain one -- the same operator, via the same call after each
    # shift_domain_em, that fills a moving nest's leading edge.
    # gpuwm.ingest.nest_spawn_init.spawn_land_state_from_parent is that
    # operator, RealSpawnChildPreparer is the attachment, and both live
    # on THIS route because it holds the input catalog and the case data.
    # Routes without them keep the refusal
    # (gpuwm.experiment.refuse_unrouted_spawn).
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
        _write_initial_perturbation_receipt(
            outdir, exp,
            ([prepared.initial_result.initial_perturbation]
             if exp.perturbation is not None else ()))
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
        _emit_front_door_capsule(
            outdir, emission_site="runtime.run_experiment:single-domain",
            exp=exp, data=data, wrfout_paths=summary.wrfout_paths,
            trajectory_digest=summary.trajectory_digest, io_mode="history")
        return summary

    _preparation_progress(progress_callback, "build-domain-tree")
    from gpuwm.core.model import build_experiment, execute_experiment
    from gpuwm.io.restart import (restore_tree_restart,
                                  write_tree_restart)
    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    from gpuwm.supervisor import validate_manifest_checkpoint

    model = build_experiment(exp, data)
    print(resolved_tree_config_report(exp, data, model._input_catalog))
    # The refusal this used to be is lifted HERE and only here: this
    # route holds the input catalog (and with it the static source), so
    # the real-data relocation initializer and physics preparer exist.
    # Routes without them still refuse inside execute_experiment.
    relocation_runner = build_real_relocation_runner(
        exp, data, model, outdir)
    # Same lift, same reason, one seam over: this route holds the input
    # catalog (own-grid statics at the fired footprint) and the case data
    # and forcing calendar (the newborn's physics driver), so it can
    # activate a dormant nest instead of refusing it.
    spawn_runner = build_real_spawn_runner(exp, data, model, outdir)
    _write_initial_perturbation_receipt(
        outdir, exp, getattr(model, "_initial_perturbation_receipts", ()))
    if restart is not None:
        _preparation_progress(progress_callback, "validate-checkpoint")
        restart = validate_manifest_checkpoint(restart)
        _preparation_progress(progress_callback, "restore-tree-checkpoint")
        restore_tree_restart(restart, model)

    _preparation_progress(progress_callback, "initialize-domain-writers")
    with PerDomainWrfoutWriters(
            model, outdir, start_time=exp.start_time,
            title=data.output_title,
            progress_callback=progress_callback) as writers:
        model._io_manager = writers
        if relocation_runner is not None:
            relocation_runner.on_child_built.attach_writers(writers)

        def history_handler(tree, node, ticks):
            _submit_tree_history_frame(writers, node, ticks)

        def restart_handler(tree, ticks):
            valid = exp.start_time + timedelta(
                seconds=ticks / tree.schedule.clock.tick_den)
            tree._last_checkpoint = write_tree_restart(outdir, tree, valid)

        # [tiles], on the SAME terms as the prepared domain-tree route
        # (gpuwm/prepared_domain_tree_forecast.py) and for the same reason.
        # Absent -- the default -- this is an empty mapping, no planner is
        # consulted, no tilestream module is imported and the executor binds
        # gpuwm.core.dycore.step for every grid exactly as it always did.
        # Configured, a domain the planner says will not fit resident gets a
        # streamed stepper or a loud refusal.  Before this call existed the
        # block was read, validated, echoed into the resolved-config report
        # and then IGNORED on this route: a user who wrote
        # [tiles] mode = "on" got a fully resident run with nothing
        # anywhere saying the mode never engaged.
        from gpuwm.core import streaming as _streaming

        steppers = _streaming.steppers_for_tree(model, exp.tiles)
        if spawn_runner is None:
            execute_experiment(
                model, history_handler=history_handler,
                restart_handler=restart_handler,
                progress_callback=progress_callback,
                health_debug=health_debug,
                relocation_runner=relocation_runner,
                steppers=steppers)
        else:
            # The leg walk: dormant nests are born mid-run and integrate
            # from their birth boundary onward.
            walk_spawn_legs(
                model, exp, data,
                spawn_runner=spawn_runner, writers=writers,
                lbc_interval_s=_tree_forcing_cadence_seconds(
                    model._input_catalog),
                relocation_runner=relocation_runner,
                relocation_runner_factory=(
                    lambda: build_real_relocation_runner(
                        exp, data, model, outdir)),
                history_handler=history_handler,
                restart_handler=restart_handler,
                progress_callback=progress_callback,
                health_debug=health_debug,
                steppers=steppers)
        writers.drain()
        paths = writers.paths
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    # After the writer drain above and after the final device synchronization,
    # so the digest observes the trajectory and cannot participate in it.
    trajectory_digest = None
    if trajectory_digest_enabled():
        from gpuwm.state_digest import canonical_state_digest

        trajectory_digest = {
            f"d{grid_id:02d}": canonical_state_digest(
                node.state, node.clock, scope="trajectory")
            for grid_id, node in sorted(model.nodes_by_grid_id.items())
        }
    transition_path, transition_sha, transitions = \
        _write_microphysics_transition_receipt(
            outdir, model, exp, resumed=restart is not None)
    feedback_path, feedback_sha, feedback_receipt = \
        _write_feedback_provenance_receipt(
            outdir, exp, resumed=restart is not None)
    _emit_front_door_capsule(
        outdir, emission_site="runtime.run_experiment:domain-tree",
        exp=exp, data=data, wrfout_paths=paths,
        trajectory_digest=trajectory_digest, io_mode="history")
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
        feedback_provenance_receipt_sha256=feedback_sha,
        trajectory_digest=trajectory_digest)


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
            and node.state.physics.mp_physics in REFL_10CM_MICROPHYSICS):
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
        # Per domain, not once for the root: child domains may resolve a
        # different ra_lw_physics, and this receipt's job is to name
        # where EACH domain's downward longwave comes from.
        lines.append(
            f"  domain.d{dc.grid_id:02d}.radiation.downward_longwave = "
            + downward_longwave_source(exp, dc.run))
    lines.append(
        "  output.filename_pattern = wrfout_d0X_<YYYY-MM-DD_HH_MM_SS>")
    return "\n".join(lines)


__all__ = [
    "CONSERVATION_CLOSURE_RECEIPT_NAME",
    "ExperimentRunSummary", "FEEDBACK_EXPERIMENTAL_WARNING",
    "FEEDBACK_PROVENANCE_RECEIPT_NAME",
    "MICROPHYSICS_TRANSITION_RECEIPT_NAME",
    "PreparedRealCase", "RealCaseRunSummary",
    "configured_run_schedule",
    "experiment_grid", "feedback_provenance",
    "forcing_schedule", "forcing_snapshots",
    "integrate_prepared_case", "load_source_orography",
    "prepare_child_case", "prepare_experiment_case",
    "declared_constant_glw", "downward_longwave_source",
    "prepare_root_experiment_case", "prepare_real_case", "refl_10cm_due",
    "resolved_config_report", "resolved_tree_config_report",
    "restart_outer_steps", "run_experiment",
    "single_domain", "vertical_coord_for", "whole_step_count",
    "write_case_output", "write_ingest", "write_static",
]
