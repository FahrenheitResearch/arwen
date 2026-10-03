"""Implicit vertical advection through the resident and decomposed dycore.

Run ``python -m tilestream.ieva_gate --devices 0,1,2,3 --output receipt.json``.
One physical card is also sufficient for the geometry gate.  The receipt
records the physical device assignment, so same-card results cannot be
mistaken for cross-card evidence.  This is an idealized dynamics and scalar
transport test, not an operational forecast qualification.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
import hashlib
import json
from pathlib import Path
import time
import tomllib

import numpy as np

from tilestream import harness


GRIDS = ((1, 1), (1, 2), (2, 1), (2, 2))


def gate_config(nx=160, ny=132, nz=50, *, variant="wrf_471"):
    """A forced hybrid-coordinate case with map factors and moist scalars."""
    return harness.make_config(
        nx, ny, nz, periodic=False, specified=True, nested=False,
        open_x=False, open_y=False, map_proj=1, terrain_opt=1,
        dx=3000.0, dy=3000.0, ztop=8000.0, hybrid_opt=2, etac=0.2,
        moist=True, moist_cq=True, mp_physics=0, zadvect_implicit=1,
        zadvect_implicit_variant=variant,
        w_damping=0, spec_bdy_width=10, spec_zone=1, relax_zone=9)


def _digest(fields):
    """Bounded-memory SHA-256 over shape, dtype and every persisted byte."""
    result = hashlib.sha256()
    for name, field in fields.items():
        value = np.ascontiguousarray(field)
        result.update(name.encode("utf-8"))
        result.update(value.dtype.str.encode("ascii"))
        result.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        result.update(memoryview(value).cast("B"))
    return result.hexdigest()


def _finite(fields):
    return all(bool(np.isfinite(value).all()) for value in fields.values())


@contextmanager
def observe_splits():
    """Count live implicit flux, including the separate scalar split.

    A low-Courant case can pass every geometry while never transporting an
    implicit flux.  Counting the actual final-stage split closes that hole.
    Calls are made on the stepping thread and synchronize only this gate.
    """
    import cupy as cp
    from gpuwm.core import ieva

    original_dynamics = ieva.prepare_dynamics
    original_scalar = ieva.split_scalar_omega
    counts = {"dynamics": [], "scalar": []}

    def dynamics(*args, **kwargs):
        result = original_dynamics(*args, **kwargs)
        counts["dynamics"].append(int(cp.count_nonzero(result.wwI)))
        return result

    def scalar(*args, **kwargs):
        result = original_scalar(*args, **kwargs)
        counts["scalar"].append(int(cp.count_nonzero(result[1])))
        return result

    ieva.prepare_dynamics = dynamics
    ieva.split_scalar_omega = scalar
    try:
        yield counts
    finally:
        ieva.prepare_dynamics = original_dynamics
        ieva.split_scalar_omega = original_scalar


def _seed_state(cfg, geo, seed):
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import set_w_surface

    state = harness.make_state(cfg, seed=seed, geography=geo)
    # A smooth, deliberately strong vertical current crosses WW_SPLIT's
    # explicit limit.  It vanishes at both lids and varies across both seams.
    z = np.sin(np.linspace(0.0, np.pi, cfg.nz + 1)).astype(np.float32)
    y = np.linspace(-1.0, 1.0, cfg.ny, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, cfg.nx, dtype=np.float32)[None, :]
    horizontal = np.float32(1.0) + np.float32(0.1) * x * y
    state.w[...] = cp.asarray(np.float32(100.0) * z[:, None, None]
                             * horizontal[None])
    state.u += cp.float32(15.0)
    set_w_surface(state, cfg)
    profile = np.linspace(0.004, 0.0002, cfg.nz, dtype=np.float32)
    state.qv[...] = cp.asarray(profile[:, None, None] * horizontal[None])
    state.qc[...] = cp.float32(1.0e-5)
    state.qr[...] = cp.float32(2.0e-6)
    update_diagnostics(state, cfg.hypsometric_opt)
    return state


def gate(nx=160, ny=132, nz=50, steps=4, *, devices=(0,), grids=GRIDS,
         controls=True, verbose=True, variant="wrf_471", experiment=None):
    """Compare every persisted byte and require active implicit transport."""
    import cupy as cp
    from gpuwm.ingest.lateral_bc import (
        attach_lateral_boundaries, build_state_lateral_boundaries)
    from tilestream import multigpu as mg

    cfg = gate_config(nx, ny, nz, variant=variant)
    projection, p_top, source_hash = None, None, None
    if experiment is not None:
        cfg, projection, p_top, source_hash = experiment_grid(experiment, variant)
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    devices = tuple(int(device) for device in devices)
    if not devices:
        raise ValueError("at least one physical device is required")
    if steps < 2:
        raise ValueError("at least two steps are required to exercise halo reuse")
    init_started = time.perf_counter()
    geo = harness.make_geography(cfg, terrain=True, periodic_faces=False,
                                 height=800.0, projection=projection)
    device = devices[0]
    memory = {}
    with cp.cuda.Device(device):
        if experiment is None:
            ref = _seed_state(cfg, geo, harness.DEFAULT_SEED)
            other = _seed_state(cfg, geo, harness.DEFAULT_SEED + 1)
            boundaries = build_state_lateral_boundaries(
                [ref, other], [0.0, 3600.0],
                spec_bdy_width=cfg.spec_bdy_width,
                spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
            del other
        else:
            memory = _resident_proof_memory(cfg, p_top)
            ref = _analytic_stream_state(cfg, geo, 0, ny, p_top=p_top)
            boundaries = _evolving_boundaries(ref, cfg)
        start = mg.download_state(ref)
        attach_lateral_boundaries(ref, boundaries)
        grid = _grid_identity(cfg, projection, ref)
        init_seconds = time.perf_counter() - init_started
        step_started = time.perf_counter()
        with observe_splits() as observed:
            harness.run_steps(ref, cfg, steps)
        step_seconds = time.perf_counter() - step_started
        truth = mg.download_state(ref)
        reference = {"hash": _digest(truth), "finite": _finite(truth),
                     "implicit_points": observed,
                     "init_seconds": init_seconds,
                     "step_seconds": step_seconds,
                     "mean_step_seconds": step_seconds / steps,
                     "memory_pool_used_bytes": cp.get_default_memory_pool().used_bytes(),
                     "memory_pool_held_bytes": cp.get_default_memory_pool().total_bytes()}
        del ref
        cp.get_default_memory_pool().free_all_blocks()

    results = {"shape": [nx, ny, nz], "steps": steps,
               "dt": cfg.dt, "zadvect_implicit": cfg.zadvect_implicit,
               "variant": variant,
               "grid": grid, "grid_sha256": _json_digest(grid),
               "experiment_sha256": source_hash,
               "memory_preflight": memory,
               "halo": mg.forced_halo(cfg),
               "scope": "idealized forced dynamics and moist scalar transport",
               "limitations": ["synthetic initial atmosphere and terrain",
                               "physical parameterizations disabled"],
               "controls_enabled": bool(controls),
               "reference": reference, "runs": {}, "controls": {}}
    active = all(any(count > 0 for count in values)
                 for values in reference["implicit_points"].values())
    ok = reference["finite"] and active

    def arm(grid, *, run_cfg=cfg, seam="zeros"):
        count = int(grid[0]) * int(grid[1])
        assigned = [devices[index % len(devices)] for index in range(count)]
        init_started = time.perf_counter()
        def state_factory(sub_cfg, *, spec):
            if experiment is None:
                return mg.forced_state_factory(geo)(sub_cfg, spec=spec)
            return _empty_stream_state(
                sub_cfg, mg.window_geography(geo, spec), p_top=p_top)
        with mg.MultiGPUDomain(
                run_cfg, grid=grid, devices=assigned, boundaries=boundaries,
                seam=seam, state_factory=state_factory) as dom:
            dom.load_from_host(start)
            dom.impose_clock(0.0)
            init_seconds = time.perf_counter() - init_started
            step_started = time.perf_counter()
            with observe_splits() as seen:
                dom.run(steps, step_mode="events", exchange_mode="events")
            dom.sync_all()
            step_seconds = time.perf_counter() - step_started
            host = dom.assemble_host()
        result = {"hash": _digest(host), "finite": _finite(host),
                  "devices": assigned, "implicit_points": seen,
                  "init_seconds": init_seconds, "step_seconds": step_seconds,
                  "mean_step_seconds": step_seconds / steps}
        result["match"] = result["hash"] == reference["hash"]
        if not result["match"]:
            result["diff"] = mg.compare_hosts(truth, host, nx)
        return result

    for grid in grids:
        name = f"{grid[0]}x{grid[1]}"
        row = arm(grid)
        row_active = all(any(value > 0 for value in values)
                         for values in row["implicit_points"].values())
        ok = ok and row["match"] and row["finite"] and row_active
        results["runs"][name] = row
        if verbose:
            print(f"IEVA {name} devices={row['devices']} "
                  f"match={row['match']} finite={row['finite']} "
                  f"implicit_active={row_active}", flush=True)

    if controls:
        # The poison seam must remain quarantined.  Disabling IEVA must
        # change the answer, otherwise this is not a treatment comparison.
        control_grid = next(grid for grid in GRIDS if grid[0] * grid[1] > 1)
        poison = arm(control_grid, seam="poison")
        explicit = arm(control_grid, run_cfg=replace(cfg, zadvect_implicit=0))
        results["controls"] = {
            "poison_seam_matches": poison["match"] and poison["finite"],
            "explicit_differs": not explicit["match"] and explicit["finite"],
            "explicit_hash": explicit["hash"]}
        ok = ok and all(results["controls"][name] for name in (
            "poison_seam_matches", "explicit_differs"))
    results["ok"] = bool(ok)
    return results


def _json_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _grid_identity(cfg, projection, state):
    import cupy as cp
    device_levels = cp.asnumpy(state.znw).tolist()
    return {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz,
            "dx": cfg.dx, "dy": cfg.dy, "ztop": cfg.ztop,
            "p_top": float(state.p_top),
            "eta_levels": (list(cfg.eta_levels) if cfg.eta_levels is not None
                           else device_levels),
            "eta_levels_device": device_levels,
            "hybrid_opt": cfg.hybrid_opt, "etac": cfg.etac,
            "projection": (harness.REAL74_PROJECTION if projection is None
                           else projection)}


def _resident_proof_memory(cfg, p_top):
    """Price the resident step and the retained host comparison arrays."""
    from types import SimpleNamespace
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.core.preflight import (
        FORECAST_POOL_HEADROOM, estimate_domain, forecast_pool_estimate_bytes)
    from gpuwm.core.resident_admission import admit
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS
    from tilestream.hoststore import check_allocatable

    estimate = estimate_domain(SimpleNamespace(grid_id=1, parent_id=0, run=cfg),
                               p_top=p_top, n_lbc_intervals=1)
    pool = forecast_pool_estimate_bytes(
        estimate.resident_bytes + estimate.transient_bytes,
        held_exact_bytes=estimate.held_exact_bytes,
        headroom=FORECAST_POOL_HEADROOM)
    admission = admit(
        "resident implicit-advection proof",
        {"forecast pool estimate": pool},
        stage="while integrating the whole reference domain", envelope=True)
    shapes = state_array_shapes(cfg)
    carrier_bytes = sum(4 * int(np.prod(shapes[name]))
                        for name in STATE_SERIALIZED_ATTRS if name in shapes)
    # Initial state, resident answer and current rank answer coexist on
    # host.  Eight FP64 column fields cover base-state construction work.
    host_bytes = 3 * carrier_bytes + 8 * 8 * cfg.nz * cfg.ny * cfg.nx
    check_allocatable(host_bytes)
    return {"device_admission": admission,
            "host_peak_plan_bytes": host_bytes,
            "host_carrier_bytes_per_copy": carrier_bytes}


def _evolving_boundaries(state, cfg):
    """Prescribe nonzero forcing without allocating a second full state."""
    from gpuwm.ingest.lateral_bc import (
        BoundaryInterval, FieldBoundary, LateralBoundaries, SideBoundary,
        build_state_lateral_boundaries)

    original = build_state_lateral_boundaries(
        [state, state], [0.0, 3600.0], spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    intervals = []
    for interval in original.intervals:
        fields = {}
        for name, field in interval.fields.items():
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                side = getattr(field, side_name)
                sides[side_name] = SideBoundary(
                    side.value, np.asarray(side.value) * 1.0e-7)
            fields[name] = FieldBoundary(**sides)
        intervals.append(BoundaryInterval(
            interval.start_seconds, interval.end_seconds, fields))
    return LateralBoundaries(tuple(intervals), cfg.spec_bdy_width,
                             cfg.spec_zone, cfg.relax_zone)


def experiment_grid(path, variant):
    """Read a root domain's grid without enabling its physical schemes."""
    payload = Path(path).read_bytes()
    raw = tomllib.loads(payload.decode("utf-8-sig"))
    domains = [domain for domain in raw["domain"]
               if int(domain.get("parent_id", 0)) == 0]
    if len(domains) != 1:
        raise ValueError("the proof needs exactly one root domain")
    values = dict(raw.get("shared", {}), **domains[0])
    cfg = gate_config(int(values["nx"]), int(values["ny"]),
                      int(values["nz"]), variant=variant)
    keys = ("ztop", "eta_levels", "hybrid_opt", "etac",
            "base_temp", "hypsometric_opt", "time_step_sound",
            "epssm", "emdiv", "smdiv", "h_sca_adv_order", "moist_adv_opt")
    updates = {key: values[key] for key in keys if key in values}
    if "eta_levels" in updates:
        updates["eta_levels"] = tuple(float(value) for value in updates["eta_levels"])
    updates.update(dx=float(values["dx"]),
                   dy=float(values.get("dy", values["dx"])),
                   dt=float(values.get("time_step", 3.0)))
    cfg = replace(cfg, **updates)
    projection = {key: float(value) for key, value in raw["projection"].items()
                  if key != "map_proj"}
    if raw["projection"].get("map_proj", "lambert") != "lambert":
        raise ValueError("this proof initializer supports Lambert projection")
    return cfg, projection, float(values["p_top"]), hashlib.sha256(payload).hexdigest()


def _empty_stream_state(cfg, geo, *, p_top=None):
    """Build the same base state and coordinate in slabs and tile buffers."""
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                etac=cfg.etac, eta_levels=cfg.eta_levels)
    if p_top is None:
        base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                               p_surf=cfg.p_surf, ztop=cfg.ztop,
                               terrain_z=geo.terrain)
    else:
        # This is the engine's existing real-data analytic base.  It honors
        # the pressure top directly, unlike the idealized sounding builder
        # whose top pressure is derived from its requested height.
        from gpuwm.ingest.real import _make_real_base_serial
        base = _make_real_base_serial(
            coord, geo.terrain, p_top, cfg.base_temp, cfg.hypsometric_opt)
    state = init_at_rest(cfg, coord, base)
    harness.install_geography(state, geo)
    return state


def _analytic_stream_state(cfg, geo, row, full_ny, *, p_top=None):
    """A slab of one global analytic state, independent of slab height.

    Eta mass flux comes from horizontal continuity, not physical w alone.
    Vertically sheared, horizontally varying winds make the implicit split
    active while keeping the horizontal Courant number modest.  Coordinates
    use global integer indices, including shared staggered faces.
    """
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics

    state = _empty_stream_state(cfg, geo, p_top=p_top)
    ix = np.arange(cfg.nx + 1, dtype=np.float64)
    jy = np.arange(row, row + cfg.ny + 1, dtype=np.float64)
    sx = np.sin(2.0 * np.pi * max(1, cfg.nx // 8) * ix / cfg.nx).astype(np.float32)
    sy = np.sin(2.0 * np.pi * max(1, full_ny // 8) * jy / full_ny).astype(np.float32)
    eta = cp.asnumpy(state.znu)
    shear = np.cos(np.pi * eta).astype(np.float32)[:, None, None]
    amplitude = np.float32(100.0 * 3.0 / cfg.dt)
    state.u[...] = cp.asarray(amplitude * shear * sx[None, None, :])
    state.v[...] = cp.asarray(np.float32(0.5) * amplitude * shear
                             * sy[None, :, None])
    pattern = sy[:-1, None] * sx[None, :-1]
    state.thp[...] = cp.asarray(np.float32(0.05) * pattern[None])
    profile = np.linspace(0.004, 0.0002, cfg.nz, dtype=np.float32)
    state.qv[...] = cp.asarray(profile[:, None, None]
                               * (np.float32(1.0) + np.float32(0.1) * pattern[None]))
    state.qc[...] = cp.float32(1.0e-5)
    state.qr[...] = cp.float32(2.0e-6)
    update_diagnostics(state, cfg.hypsometric_opt)
    return state


def _stream_start(cfg, *, slab_rows=64, device=0, projection=None, p_top=None):
    """Build the global host state a bounded GPU row slab at a time."""
    import cupy as cp
    from tilestream import bigdomain, driver, gather, hoststore

    geo = harness.make_geography(cfg, terrain=True, periodic_faces=True,
                                 height=800.0, projection=projection)
    start, geography = {}, {}
    setup = None
    with cp.cuda.Device(device):
        for row in range(0, cfg.ny, slab_rows):
            rows = min(slab_rows, cfg.ny - row)
            slab_cfg = replace(cfg, ny=rows)
            slab_geo = bigdomain.window_geography(geo, row, rows)
            state = _analytic_stream_state(slab_cfg, slab_geo, row, cfg.ny,
                                           p_top=p_top)
            if setup is None:
                setup = {"p_top": float(state.p_top),
                         "eta_levels": cp.asnumpy(state.znw).tolist()}
            inventories = (harness.state_arrays(state),
                           driver.geography_inventory(state))
            if not start:
                planned = 0
                for factor, inventory in zip((3, 1), inventories):
                    for value in inventory.values():
                        shape = value.shape[:-2] + (
                            cfg.ny + value.shape[-2] - rows, value.shape[-1])
                        planned += factor * int(np.prod(shape)) * value.dtype.itemsize
                hoststore.check_allocatable(planned)
            for target, inventory in zip((start, geography), inventories):
                for name, value in inventory.items():
                    if name not in target:
                        shape = value.shape[:-2] + (
                            cfg.ny + value.shape[-2] - rows, value.shape[-1])
                        target[name] = (gather.pinned_empty(shape, value.dtype)
                                        if target is geography else
                                        np.empty(shape, value.dtype))
                    target[name][..., row:row + value.shape[-2], :] = cp.asnumpy(value)
            del state, inventories, inventory, value
            cp.get_default_memory_pool().free_all_blocks()
    # Only the domain's final v face is an alias.  Slab construction has
    # overlapping v rows, and the later slab owns each interior overlap.
    for value in start.values():
        if value.shape[-1] == cfg.nx + 1:
            value[..., -1] = value[..., 0]
        if value.shape[-2] == cfg.ny + 1:
            value[..., -1, :] = value[..., 0, :]
    return start, geography, setup


def _stream_tile_state(cfg, *, p_top=None):
    # The neutral terrain makes 3-D base arrays exist before the real
    # geography is gathered.  The host store then replaces every carrier.
    return _empty_stream_state(cfg, harness.neutral_geography(cfg), p_top=p_top)


def stream_gate(nx=1797, ny=1057, nz=50, steps=2, *, devices=(0, 1, 2, 3),
                tile_nx=320, tile_ny=256, variant="wrf_legacy", experiment=None,
                resident_reference=False):
    """Fixed tiles over one, two and four cards without a resident domain.

    This full-size rung uses periodic lateral boundaries and no physical
    parameterizations.  The separate resident gate qualifies specified
    boundaries and monolithic equivalence.  Both run the real dycore and
    moist scalar transport, and require nonzero implicit flux.
    """
    import cupy as cp
    from tilestream import gather, mgstream

    devices = tuple(int(device) for device in devices)
    if not devices:
        raise ValueError("at least one physical device is required")
    if steps < 2:
        raise ValueError("at least two steps are required to exercise halo reuse")
    projection, source_hash, p_top = None, None, None
    if experiment is None:
        cfg = gate_config(nx, ny, nz, variant=variant)
    else:
        cfg, projection, p_top, source_hash = experiment_grid(experiment, variant)
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    cfg = replace(cfg, specified=False,
                  spec_zone=1, relax_zone=4, spec_bdy_width=5)
    start, geography, setup = _stream_start(
        cfg, device=devices[0], projection=projection, p_top=p_top)
    grid = {"nx": nx, "ny": ny, "nz": nz, "dx": cfg.dx, "dy": cfg.dy,
            "ztop": cfg.ztop, "p_top": setup["p_top"],
            "eta_levels": (list(cfg.eta_levels) if cfg.eta_levels is not None
                           else setup["eta_levels"]),
            "eta_levels_device": setup["eta_levels"],
            "hybrid_opt": cfg.hybrid_opt, "etac": cfg.etac,
            "projection": (harness.REAL74_PROJECTION if projection is None
                           else projection)}
    grid_hash = hashlib.sha256(json.dumps(
        grid, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    results = {"shape": [nx, ny, nz], "steps": steps, "variant": variant,
               "scope": "idealized periodic dynamics and moist scalar transport",
               "limitations": ["synthetic initial atmosphere and terrain",
                               "physical parameterizations disabled",
                               "periodic lateral boundaries"],
               "grid": grid, "grid_sha256": grid_hash,
               "experiment_sha256": source_hash, "dt": cfg.dt,
               "tile": [tile_nx, tile_ny], "halo": harness.halo_radius(cfg),
               "initial_hash": _digest(start), "runs": {}}
    reference = None
    ok = True
    if resident_reference:
        with cp.cuda.Device(devices[0]):
            geo = harness.make_geography(
                cfg, terrain=True, periodic_faces=True, height=800.0,
                projection=projection)
            state = _empty_stream_state(cfg, geo, p_top=p_top)
            for name, value in start.items():
                getattr(state, name)[...] = cp.asarray(value)
            with observe_splits() as observed:
                harness.run_steps(state, cfg, steps)
            host = {name: cp.asnumpy(value)
                    for name, value in harness.state_arrays(state).items()}
            reference = _digest(host)
            results["resident_reference"] = {
                "hash": reference, "finite": _finite(host),
                "implicit_points": observed}
            ok = results["resident_reference"]["finite"]
            del state, host
            cp.get_default_memory_pool().free_all_blocks()
    for count in (1, 2, 4):
        if count > len(devices):
            continue
        assigned = devices[:count]
        with cp.cuda.Device(assigned[0]):
            store = {name: gather.pinned_copy(value)
                     for name, value in start.items()}
        report = {}
        started = time.perf_counter()
        with observe_splits() as observed:
            mgstream.run_mgstream(
                store, cfg, tile_nx, tile_ny, halo=harness.halo_radius(cfg),
                nsteps=steps, devices=assigned, nbuffers=1,
                periodic=True, write_mode="shadow", partition="block",
                tile_state_factory=partial(_stream_tile_state, p_top=p_top), nz=nz,
                scalars={"elapsed_seconds": 0.0}, geography=geography,
                report=report)
        digest = _digest(store)
        if reference is None:
            reference = digest
        row = {"hash": digest, "devices": list(assigned),
               "match": digest == reference, "finite": _finite(store),
               "implicit_points": observed,
               "tiles": report.get("tiles"),
               "carriers": len(store),
               "wall_seconds_including_setup_and_digest": time.perf_counter() - started,
               "worker_loop_seconds": [worker["seconds"]
                                       for worker in report["per_worker"]],
               "mean_step_seconds": max(worker["seconds"]
                                        for worker in report["per_worker"]) / steps,
               "timing_scope": "instrumented loop including transfers and barriers"}
        active = all(any(value > 0 for value in values)
                     for values in observed.values())
        row["implicit_active"] = active
        ok = ok and row["match"] and row["finite"] and active
        results["runs"][str(count)] = row
        print(f"IEVA stream {nx}x{ny}x{nz} devices={assigned} "
              f"match={row['match']} finite={row['finite']} "
              f"implicit_active={active}", flush=True)
        del store
        for device in assigned:
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
    results["ok"] = bool(ok)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=132)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--variant", choices=("wrf_471", "wrf_legacy"),
                        default="wrf_471")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--tile-nx", type=int, default=320)
    parser.add_argument("--tile-ny", type=int, default=256)
    parser.add_argument("--experiment", type=Path,
                        help="bind grid and clock to this experiment TOML")
    parser.add_argument("--grids", default="1x1,1x2,2x1,2x2",
                        help="resident rank geometries as gy x gx")
    parser.add_argument("--no-controls", action="store_true",
                        help="omit extra poison-seam and explicit-advection arms")
    parser.add_argument("--resident-reference", action="store_true",
                        help="also compare streamed results to a whole resident state")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    kwargs = {"devices": tuple(int(value) for value in args.devices.split(",")),
              "variant": args.variant}
    if args.stream:
        kwargs.update(tile_nx=args.tile_nx, tile_ny=args.tile_ny,
                      experiment=args.experiment,
                      resident_reference=args.resident_reference)
    else:
        kwargs.update(experiment=args.experiment,
                      controls=not args.no_controls,
                      grids=tuple(tuple(int(part) for part in value.split("x"))
                                  for value in args.grids.split(",")))
    result = (stream_gate if args.stream else gate)(
        args.nx, args.ny, args.nz, args.steps, **kwargs)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
