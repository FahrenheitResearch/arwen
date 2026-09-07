"""Storage orchestration for a case initialized without a resident GPU state."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwm.ingest.memory_refusal import InitializationMemoryRefused


def _host(value):
    return np.array(value.get() if hasattr(value, 'get') else value, copy=True)


@dataclass(frozen=True)
class CaseStoreRequest:
    """An internal storage choice; the preprocessing engine stays explicit."""

    path: Path
    backend: str = 'cuda'
    resources: dict | None = None
    admissions: list = field(default_factory=list)


@dataclass(frozen=True)
class CasePhysicsContext:
    """The resolved case inputs that are not in a prepared surface bundle."""

    vertical: object
    reconciled_soil_type: np.ndarray
    sst: np.ndarray
    trace_gas_overrides: object
    radiation_column_chunk: int
    cam_ozone: object = None

    def __call__(self, result, cfg, met, surface, static, landuse_attrs,
                 grid, valid_time, *, row_start, domain_rows,
                 center_lat=None, constant_glw_wm2=None):
        from gpuwm.runtime import (
            _initialize_real_case_physics, apply_single_domain_pbl_cadence,
        )

        if self.reconciled_soil_type.shape != (domain_rows, cfg.nx):
            raise ValueError('resolved soil categories differ from the case grid')
        rows = slice(row_start, row_start + cfg.ny)
        fields = surface.fields
        soil = SimpleNamespace(**{
            name: fields[key] for name, key in {
                'tsk': 'TSK', 'soil_temperature': 'TSLB',
                'soil_moisture': 'SMOIS', 'liquid_moisture': 'SH2O',
                'deep_soil_temperature': 'TMN', 'xice': 'SEAICE',
                'snow_water': 'SNOW', 'snow_depth': 'SNOWH',
            }.items()})
        cam = self.cam_ozone
        if cam is not None:
            from dataclasses import replace
            lat, lon = grid.latlon_mass()
            cam = replace(cam, latitude_deg=lat, longitude_deg=lon)
        _initialize_real_case_physics(
            result, cfg, met, soil, {'SST': self.sst[rows]}, static,
            landuse_attrs, grid, valid_time, vertical=self.vertical,
            reconciled_soil_type=self.reconciled_soil_type[rows],
            trace_gas_overrides=self.trace_gas_overrides,
            radiation_column_chunk=self.radiation_column_chunk,
            center_lat=center_lat, constant_glw_wm2=constant_glw_wm2,
            **({"cam_ozone": cam} if cam is not None else {}))
        # The frozen ordinary loop makes this choice before its first step.
        # Apply it before the slab template is cloned into tile buffers.
        apply_single_domain_pbl_cadence(result.state.physics, cfg)


@dataclass(frozen=True)
class CaseStoreInput:
    path: Path
    identity: dict
    physics: CasePhysicsContext
    landuse_attrs: dict
    constant_glw_wm2: float | None
    receipt: dict
    admissions: tuple[dict, ...]


def write_case_store_input(request, *, cfg, vertical, times, initial_result,
                           met, soil, soil_fields, reconciled_soil_type,
                           boundaries, landuse_attrs, trace_gas_overrides,
                           radiation_column_chunk, constant_glw_wm2, cam_ozone=None):
    """Seal existing initialized arrays, then let the caller release them.

    This is an ephemeral, same-run cache. Its payload hashes are the existing
    prepared-cache contract; source/config provenance is still emitted by
    the ordinary runtime's run capsule.
    """
    from gpuwm.ingest.prepared_cache import write_prepared_cache
    from gpuwm.native_wrf_contract import canonical_noah_surface

    identity = {
        'schema': 'gpuwm-case-store-initialization-v1',
        'run': asdict(cfg), 'vertical': asdict(vertical),
        'forcing_times': [time.isoformat() for time in times],
        'preprocess_backend': request.backend,
    }
    receipt = write_prepared_cache(
        request.path, identity=identity, initial_result=initial_result,
        met=met, boundaries=boundaries, surface=canonical_noah_surface(soil),
        metadata={"initialization_memory": request.admissions})
    context = CasePhysicsContext(
        vertical=vertical, reconciled_soil_type=_host(reconciled_soil_type),
        sst=_host(soil_fields.get('SST', soil.tsk)),
        trace_gas_overrides=trace_gas_overrides,
        radiation_column_chunk=radiation_column_chunk, cam_ozone=cam_ozone)
    return CaseStoreInput(request.path, identity, context,
                          dict(landuse_attrs), constant_glw_wm2, receipt,
                          tuple(request.admissions))


def build_case_store(prepared, *, valid_time, decision, options, log=print):
    """Use the existing row-slab loader and its exact pinned-memory guard."""
    from dataclasses import replace

    from gpuwm.ingest.prepared_store import store_from_prepared_cache
    from gpuwm.native_wrf_contract import native_static_export_fields

    inputs = prepared.store_input
    if inputs is None:
        raise ValueError('case has no sealed host initialization')
    static = native_static_export_fields(prepared.static_fields, prepared.grid)
    bundle = store_from_prepared_cache(
        inputs.path, expected_identity=inputs.identity, cfg=prepared.cfg,
        static=static, landuse_attrs=inputs.landuse_attrs,
        grid=prepared.grid, valid_time=valid_time,
        rows_per_slab=min(64, int(decision.tile_ny), int(prepared.cfg.ny)),
        budget_bytes=options.host_budget_bytes,
        physics_initializer=inputs.physics,
        constant_glw_wm2=inputs.constant_glw_wm2, log=log)
    initial = SimpleNamespace(
        **vars(prepared.initial_result), state=bundle.template,
        coord=bundle.coord, base=bundle.base)
    return replace(prepared, initial_result=initial, store_input=None,
                   streamed_store=bundle, initialization_receipt={
                       'schema': 'gpuwm-case-store-initialization-v1',
                       'preparation': dict(inputs.receipt),
                       'memory': list(inputs.admissions),
                       'store': dict(bundle.receipt),
                       'forcing_clock': 'elapsed_seconds',
                   }), bundle


def initialization_resources(options):
    """Read available capacity before initialization allocations begin."""
    from gpuwm.core.preflight import (
        device_memory_probe_subprocess, host_available_bytes,
    )

    probe = device_memory_probe_subprocess()
    device_budget = None if probe is None else int(probe['free_bytes'])
    configured = options.vram_budget_bytes
    if configured is not None:
        device_budget = (int(configured) if device_budget is None
                         else min(device_budget, int(configured)))
    available_host = host_available_bytes()
    return {'device_budget_bytes': device_budget,
            'host_available_bytes': available_host, 'device_probe': probe}


def admit_case_initialization(request, cfg, met, times):
    """Check the remaining phase against its pre-allocation reading.

    Horizontal interpolation has already completed. Its live output is in
    the device estimate, compared with the earlier free-memory reading so
    those arrays are not charged twice. Horizontal workspace is not claimed
    by this check and an unknown price never becomes a feature refusal.
    """
    from gpuwm.core.preflight import (
        estimate_host_state_initialization, profile_from_device_probe,
    )

    shapes = {name: tuple(int(n) for n in value.shape)
              for name, value in met.fields.items()
              if len(value.shape) in (2, 3)}
    resources = request.resources or {}
    probe = resources.get('device_probe')
    total = None if probe is None else probe.get('total_bytes')
    estimate = estimate_host_state_initialization(
        cfg, analysis_shapes=shapes, forcing_times=len(times),
        vram_gib=None if total is None else int(total)/2**30,
        profile=profile_from_device_probe(probe))
    device_budget = resources.get('device_budget_bytes')
    host_budget = resources.get('host_available_bytes')
    record = {
        'preprocess_backend': request.backend, 'state_backend': 'cpu',
        'analysis_fields': shapes, 'forcing_times': len(times),
        'device_initialization_envelope_bytes': estimate.device.peak_envelope_bytes,
        'device_budget_bytes': device_budget,
        'host_initialization_floor_bytes': estimate.host_floor_bytes,
        'host_available_bytes': host_budget,
        'scope': 'remaining initialization after horizontal interpolation; host temporaries above the stated floor are unpriced',
    }
    request.admissions.append(record)
    if device_budget is not None and estimate.device.peak_envelope_bytes > device_budget:
        raise InitializationMemoryRefused(
            f"CUDA initialization estimates {estimate.device.peak_envelope_bytes/2**30:.2f} GiB "
            f"against {device_budget/2**30:.2f} GiB available. The streamed "
            "forecast does not remove initialization workspace. Free GPU "
            "memory or reduce the prepared grid.")
    if host_budget is not None and estimate.host_floor_bytes > host_budget:
        raise InitializationMemoryRefused(
            f"Host initialization requires at least {estimate.host_floor_bytes/2**30:.2f} GiB "
            f"against {host_budget/2**30:.2f} GiB available. Free host memory "
            "or reduce the prepared grid.")
    return record


def write_initialization_receipt(outdir, prepared, coverage):
    import json
    from gpuwm.fetch_guard import atomic_write_text

    record = dict(prepared.initialization_receipt or {})
    record['health_coverage'] = coverage
    atomic_write_text(Path(outdir)/'initialization.json',
                      json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
