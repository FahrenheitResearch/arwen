"""Native WRF debug arrays through the Rust text writer."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


def dump_fields(fields, step, *, directory="."):
    from gpuwm.static.sfire import _library, _json_buffer
    from gpuwm.static import rust_bridge
    library = _library()
    if not hasattr(library, "gpuwm_static_sfire_debug_array"):
        raise RuntimeError("fire_print_file needs the native SFIRE debug writer; rebuild static-fields")
    writer = library.gpuwm_static_sfire_debug_array
    writer.argtypes = (ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
                       ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    writer.restype = ctypes.c_int32
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, value in fields.items():
        if value.dtype != np.dtype(np.float32):
            continue
        host = value.get() if hasattr(value, "get") else value
        host = np.ascontiguousarray(host)
        if host.ndim > 3:
            raise ValueError("native SFIRE debug arrays have at most three axes")
        shape = (1,) * (3-host.ndim) + tuple(host.shape)
        nz, ny, nx = shape
        request = dict(name=name.lower(), path=str(directory), shape=list(shape),
                       bounds=[1, nx, 1, ny, 1, nz], step=int(step))
        document, length = _json_buffer(request)
        if writer(document, length, host.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), host.size):
            raise ValueError(rust_bridge.last_error(library))
        written.append(directory / f"{name.lower()}_{step:08d}.txt")
    return tuple(written)


def report_step(fire, cfg, *, fields=None):
    """Debug controls use the completed domain step, including tiled runs."""
    number = fire.grid.step_count
    if cfg.fire_print_msg > 0:
        print(f"SFIRE step {number}: time={fire.grid.time_seconds:.9g} s "
              f"CFL bound={fire.grid.last_cfl_bound:.9g} s "
              f"ignition nodes={fire.grid.last_ignited_counts}", flush=True)
    every = int(cfg.fire_print_file)
    if every > 0 and (number <= every or number % every == 0):
        values = fire.output_fields() if fields is None else fields
        if fields is None:
            from gpuwm.core.sfire import STAGE_FIELDS
            values.update({name.upper(): fire.grid.data[name][fire.grid.interior] for name in STAGE_FIELDS})
        dump_fields(values, number)


class FireDiagnosticComplete(RuntimeError):
    """Intentional stop after WRF's positive fire_test_steps diagnostic."""

    def __init__(self, passes, time_seconds):
        self.passes = int(passes)
        self.time_seconds = float(time_seconds)
        super().__init__(f"SFIRE uncoupled diagnostic completed {self.passes} fire passes "
                         f"at {self.time_seconds:.9g} s; fire_test_steps intentionally stops the run")


def run_test_steps(fire, state, cfg, atmosphere, time_seconds, dt, surface):
    """Run the compiled driver's inclusive 0..N loop before atmosphere feedback.

    Frozen atmospheric inputs are reused. The fire clock advances per pass,
    correcting WRF's reused time_start in this diagnostic loop.
    """
    passes = int(cfg.fire_test_steps) + 1
    for number in range(passes):
        start = time_seconds if number == 0 else fire.grid.time_seconds
        fire._advance_once(state, cfg, atmosphere, start, dt, surface, feedback=False, report=False)
        if fire.spotting is not None:
            fire.spotting.advance_atmosphere(state, cfg, fire, dt=dt)
        fire._report_step(cfg)
    raise FireDiagnosticComplete(passes, fire.grid.time_seconds)


def diagnostic_tile_step(state, cfg, **unused):
    """One fire-only tile pass; the caller owns the whole-domain stop."""
    from gpuwm.core.physics import _prepare_atmosphere
    driver = state.physics
    fire = driver.fire
    atmosphere = _prepare_atmosphere(state)
    surface = dict(driver.fields, psfc=atmosphere["p_interface"][0],
                   rainc=driver._zero_accumulator() if driver.rainc is None else driver.rainc,
                   rainnc=driver.microphysics.rainnc)
    fire._advance_once(state, cfg, atmosphere, fire.grid.time_seconds, float(cfg.dt), surface,
                       feedback=False)
    driver.call_counts["fire"] += 1


def report_streamed(streamed, cfg):
    """Write one complete fire domain after every tile's scatter is finished."""
    if not (cfg.fire_print_msg or cfg.fire_print_file):
        return
    from types import SimpleNamespace
    from gpuwm.io.sfire_schema import SFIRE_REGISTRY_FIELDS
    from gpuwm.core.sfire import STAGE_FIELDS
    scalars = streamed.scalars
    if not scalars or "fire_header" not in scalars:
        return
    grid = SimpleNamespace(**scalars["fire_header"]["fire"]["grid"])
    fire = SimpleNamespace(grid=grid)
    every = int(cfg.fire_print_file)
    fields = {}
    if every > 0 and (grid.step_count <= every or grid.step_count % every == 0):
        fields.update({name: value for name, value in streamed.history_fields().items()
                       if name in SFIRE_REGISTRY_FIELDS})
        canonical = streamed._run.canonical_store() if hasattr(streamed._run, "canonical_store") else streamed.store
        for name in STAGE_FIELDS:
            key = f"fire/grid.{name}"
            if key in canonical:
                fields[name.upper()] = canonical[key][1:-1, 1:-1]
    report_step(fire, cfg, fields=fields)


def run_streamed_test_steps(streamed, state, cfg):
    """Complete N+1 fire-only domain sweeps, then stop once globally."""
    passes = int(cfg.fire_test_steps) + 1
    for _ in range(passes):
        with streamed.allocation_scope():
            streamed._run.sweep(1, live_config=cfg, step_function=diagnostic_tile_step)
        streamed._run.drain()
        spotting = getattr(streamed, "_spotting_owner", None)
        if spotting is not None:
            spotting.advance(dt=float(cfg.dt))
            streamed._run.reseed_clock(streamed.scalars)
        report_streamed(streamed, cfg)
    clock = streamed.scalars["fire_clocks"]
    raise FireDiagnosticComplete(passes, clock["time_seconds"])
