#!/usr/bin/env python3
"""Full-RK3 WSM6 + scheduled Dudhia-SW smoke for a rented CUDA GPU.

This is deliberately a small synthetic integration gate.  The stock
``PhysicsDriver`` schedules the executable ``LW=0, SW=1`` Dudhia adapter at
every model step, its heating enters all RK stages, and stock WSM6 runs in
the post-RK microphysics slot on the same trajectory.  It is not a WRF
trajectory-parity or long-production-stability claim.

Stdout is exactly one strict JSON document.  Human diagnostics belong in
that document so a remote controller can retain the evidence verbatim.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


# Running ``python tools/...`` sets sys.path[0] to tools/.  Prefer this
# checkout over any unrelated editable gpuwm install on the controller.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _device_name(cp) -> str:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    raw = props["name"]
    return (raw.decode("utf-8", errors="replace")
            if isinstance(raw, bytes) else str(raw))


def _combined_config(hail_opt: int):
    """Return the exact combined-scheme configuration without using CUDA."""
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases import moist_bubble

    return validate_run_config(replace(
        moist_bubble.default_config(),
        nx=32, ny=8, nz=40, dt=1.5, run_seconds=15.0,
        time_step_sound=4,
        moist=True, mp_physics=6, wsm6_hail_opt=hail_opt,
        # Exercise the executable split adapter.  RRTM LW remains disabled;
        # selecting LW=1 must continue to fail closed at initialization.
        ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
        icloud=1, swrad_scat=1.0,
        # WRF radt=0 semantics: the driver must schedule radiation every step.
        radt=0.0, radt_minutes=0.0,
        km_opt=1, khdif=0.0, kvdif=0.0,
        diff_6th_opt=0, damp_opt=0,
        case="easy_physics_gpu_smoke",
    ))


def _build_state(hail_opt: int):
    import cupy as cp

    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import moist_bubble
    from gpuwm.verify.cases.wk82 import wk82_sounding

    cfg = _combined_config(hail_opt)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: wk82_sounding(z)[0],
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = moist_bubble.build(cfg, coord, base)

    # A smooth nonuniform mixed-phase cloud forces every WSM6 reservoir and
    # lets Dudhia see both liquid and ice optical paths before WSM6 converts
    # or sediments them.  Categories need not each survive the trajectory.
    z = state.height_half()[:, None, None]
    x = ((np.arange(cfg.nx) + 0.5) * cfg.dx
         - 0.5 * cfg.nx * cfg.dx)[None, None, :]
    y = ((np.arange(cfg.ny) + 0.5) * cfg.dy
         - 0.5 * cfg.ny * cfg.dy)[None, :, None]
    cloud = np.exp(
        -((x / 5000.0) ** 2 + (y / 2500.0) ** 2
          + ((z - 6000.0) / 1800.0) ** 2)).astype(np.float32)
    for name, amplitude in {
            "qc": 2.0e-4, "qi": 5.0e-5, "qr": 8.0e-5,
            "qs": 6.0e-5, "qg": 2.0e-5}.items():
        getattr(state, name)[:] += cp.asarray(np.float32(amplitude) * cloud)
    return cfg, state


def _failure_reasons(*, state_finite: bool, moisture_min: float,
                     condensate_max: float, moisture_change_max: float,
                     wsm6_heating_max: float, sw_heating_max: float,
                     sw_heating_min: float,
                     swdown_min: float, swdown_max: float,
                     lw_heating_max: float, glw_error: float,
                     precip_min: float, sr_min: float, sr_max: float,
                     elapsed_error: float, radiation_calls: int,
                     adapter_updates: int, microphysics_updates: int,
                     steps: int, scheme_ids: tuple[int, int]) -> list[str]:
    failures: list[str] = []
    finite_scalars = (
        moisture_min, condensate_max, moisture_change_max,
        wsm6_heating_max, sw_heating_max, sw_heating_min,
        swdown_min, swdown_max, lw_heating_max, glw_error,
        precip_min, sr_min, sr_max, elapsed_error)
    if not state_finite or not all(math.isfinite(v) for v in finite_scalars):
        failures.append("non-finite state or diagnostic")
    if moisture_min < -1.0e-10:
        failures.append("WSM6 moisture fell below -1e-10 kg kg-1")
    if condensate_max <= 0.0:
        failures.append("all retained condensate vanished")
    if moisture_change_max <= 1.0e-10:
        failures.append("WSM6 did not change any moisture field")
    if wsm6_heating_max <= 0.0:
        failures.append("WSM6 h_diabatic handoff is zero")
    if sw_heating_max <= 0.0:
        failures.append("scheduled Dudhia shortwave heating is zero")
    if sw_heating_min < -1.0e-12:
        failures.append("Dudhia shortwave heating is materially negative")
    if swdown_min <= 0.0 or swdown_max <= swdown_min:
        failures.append("Dudhia daylight SWDOWN is non-positive or uniform")
    if lw_heating_max != 0.0:
        failures.append("LW=0 produced a nonzero longwave tendency")
    if glw_error != 0.0:
        failures.append("SW-only radiation modified the held GLW field")
    if precip_min < -1.0e-10:
        failures.append("WSM6 precipitation diagnostics are negative")
    if sr_min < 0.0 or sr_max > 1.0:
        failures.append("WSM6 frozen precipitation fraction is outside [0,1]")
    if elapsed_error > 1.0e-6:
        failures.append("model clock does not equal steps * dt")
    if radiation_calls != steps or adapter_updates != steps:
        failures.append("Dudhia did not run through the every-step scheduler")
    if microphysics_updates != steps:
        failures.append("WSM6 diagnostics were not accepted every step")
    if scheme_ids != (0, 1):
        failures.append("driver did not resolve the requested LW/SW=0/1 pair")
    return failures


def _strict_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _strict_json(value):
    """Recursively remove NumPy scalars and non-finite JSON extensions."""
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _strict_float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run(steps: int, hail_opt: int, require_rtx_5090: bool) -> dict:
    import cupy as cp

    from gpuwm.core.dycore import run_steps, stability_report
    from gpuwm.core.physics import initialize_physics

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if hail_opt not in (0, 1):
        raise ValueError(f"hail_opt must be 0 or 1, got {hail_opt}")
    device = _device_name(cp)
    if require_rtx_5090 and "RTX 5090" not in device.upper():
        raise RuntimeError(
            f"--require-rtx-5090 requested, but CUDA device is {device!r}")

    free0, total = cp.cuda.runtime.memGetInfo()
    started = time.perf_counter()
    cfg, state = _build_state(hail_opt)
    shape2 = (cfg.ny, cfg.nx)
    latitude = cp.broadcast_to(
        cp.linspace(cp.float32(38.7), cp.float32(39.3), cfg.ny)[:, None],
        shape2).copy()
    longitude = cp.broadcast_to(
        cp.linspace(cp.float32(-87.4), cp.float32(-86.6), cfg.nx)[None, :],
        shape2).copy()
    driver = initialize_physics(
        state, cfg,
        radiation_start_time=datetime(1974, 4, 3, 18, 0),
        radiation_latitude=latitude,
        radiation_longitude=longitude,
        glw=311.0, swdown=0.0,
        # Spatial albedo contrast proves the flux field passed through the
        # actual adapter rather than a constant-forcing bypass.
        landmask=1.0,
    )
    driver.fields["albedo"][:] = cp.linspace(
        cp.float32(0.10), cp.float32(0.30), cfg.nx)[None, :]

    moisture_names = ("qv", "qc", "qi", "qr", "qs", "qg")
    initial = {name: getattr(state, name).copy() for name in moisture_names}
    initial_maxima = {
        name: float(cp.max(field).get()) for name, field in initial.items()}

    run_steps(state, cfg, n=steps)
    cp.cuda.Stream.null.synchronize()
    wall_seconds = time.perf_counter() - started

    state_arrays = (
        state.u, state.v, state.w, state.thp, state.p, state.php,
        *(getattr(state, name) for name in moisture_names),
        driver.rthratensw, driver.rthratenlw,
        driver.fields["swdown"], driver.fields["glw"])
    state_finite = all(bool(cp.all(cp.isfinite(a)).get()) for a in state_arrays)
    moisture_min = min(
        float(cp.min(getattr(state, name)).get()) for name in moisture_names)
    moisture_changes = {
        name: float(cp.max(cp.abs(getattr(state, name) - initial[name])).get())
        for name in moisture_names}
    moisture_maxima = {
        name: float(cp.max(getattr(state, name)).get())
        for name in moisture_names}
    condensate_max = float(cp.max(
        state.qc + state.qi + state.qr + state.qs + state.qg).get())
    moisture_change_max = max(moisture_changes.values())
    wsm6_heating_max = float(cp.max(cp.abs(state.h_diabatic)).get())
    sw_heating_max = float(cp.max(driver.rthratensw).get())
    sw_heating_min = float(cp.min(driver.rthratensw).get())
    lw_heating_max = float(cp.max(cp.abs(driver.rthratenlw)).get())
    swdown_min = float(cp.min(driver.fields["swdown"]).get())
    swdown_max = float(cp.max(driver.fields["swdown"]).get())
    glw_error = float(cp.max(
        cp.abs(driver.fields["glw"] - cp.float32(311.0))).get())
    precip_arrays = tuple(
        getattr(driver.microphysics, name) for name in (
            "rainnc", "rainncv", "snownc", "snowncv",
            "graupelnc", "graupelncv"))
    precip_min = min(float(cp.min(value).get()) for value in precip_arrays)
    sr_min = float(cp.min(driver.microphysics.sr).get())
    sr_max = float(cp.max(driver.microphysics.sr).get())
    elapsed_error = abs(float(state.elapsed_seconds) - steps * cfg.dt)
    radiation_calls = int(driver.call_counts["radiation"])
    adapter_updates = int(driver.radiation_callable.update_count)
    microphysics_updates = int(driver.microphysics_updates)
    scheme_ids = (int(driver.ra_lw_physics), int(driver.ra_sw_physics))
    failures = _failure_reasons(
        state_finite=state_finite,
        moisture_min=moisture_min,
        condensate_max=condensate_max,
        moisture_change_max=moisture_change_max,
        wsm6_heating_max=wsm6_heating_max,
        sw_heating_max=sw_heating_max,
        sw_heating_min=sw_heating_min,
        swdown_min=swdown_min,
        swdown_max=swdown_max,
        lw_heating_max=lw_heating_max,
        glw_error=glw_error,
        precip_min=precip_min,
        sr_min=sr_min,
        sr_max=sr_max,
        elapsed_error=elapsed_error,
        radiation_calls=radiation_calls,
        adapter_updates=adapter_updates,
        microphysics_updates=microphysics_updates,
        steps=steps,
        scheme_ids=scheme_ids)

    free1, _ = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return {
        "verdict": "PASS" if not failures else "FAIL",
        "scope": "synthetic full-RK3 WSM6 plus scheduled Dudhia SW",
        "not_a_claim": "WRF trajectory parity or production stability",
        "device": device,
        "versions": {
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        },
        "configuration": {
            "grid_nz_ny_nx": [cfg.nz, cfg.ny, cfg.nx],
            "dt_seconds": cfg.dt,
            "steps": steps,
            "simulated_seconds": _strict_float(state.elapsed_seconds),
            "mp_physics": cfg.mp_physics,
            "wsm6_hail_opt": cfg.wsm6_hail_opt,
            "ra_lw_physics": cfg.ra_lw_physics,
            "ra_sw_physics": cfg.ra_sw_physics,
            "radt_minutes": cfg.radt_minutes,
        },
        "scheduling": {
            "radiation_calls": radiation_calls,
            "dudhia_adapter_updates": adapter_updates,
            "microphysics_updates": microphysics_updates,
            "driver_scheme_ids": list(scheme_ids),
            "clock_error_seconds": _strict_float(elapsed_error),
        },
        "checks": {
            "state_finite": state_finite,
            "moisture_min_kg_kg-1": _strict_float(moisture_min),
            "retained_condensate_max_kg_kg-1": _strict_float(condensate_max),
            "wsm6_h_diabatic_max_K_s-1": _strict_float(wsm6_heating_max),
            "wsm6_precip_diagnostic_min_mm": _strict_float(precip_min),
            "wsm6_frozen_fraction_range": [
                _strict_float(sr_min), _strict_float(sr_max)],
            "dudhia_sw_heating_max_K_s-1": _strict_float(sw_heating_max),
            "dudhia_sw_heating_min_K_s-1": _strict_float(sw_heating_min),
            "lw_heating_max_abs_K_s-1": _strict_float(lw_heating_max),
            "swdown_range_W_m-2": [
                _strict_float(swdown_min), _strict_float(swdown_max)],
            "held_glw_max_error_W_m-2": _strict_float(glw_error),
        },
        "species_initial_max_kg_kg-1": {
            k: _strict_float(v) for k, v in initial_maxima.items()},
        "species_final_max_kg_kg-1": {
            k: _strict_float(v) for k, v in moisture_maxima.items()},
        "species_max_change_kg_kg-1": {
            k: _strict_float(v) for k, v in moisture_changes.items()},
        "stability": stability_report(state, cfg),
        "performance": {
            "wall_seconds_including_jit": _strict_float(wall_seconds),
            "device_memory_total_bytes": int(total),
            "device_memory_delta_bytes": int(free0 - free1),
            "cupy_pool_used_bytes": int(pool.used_bytes()),
            "cupy_pool_held_bytes": int(pool.total_bytes()),
        },
        "failures": failures,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hail-opt", type=int, choices=(0, 1), default=0)
    parser.add_argument("--require-rtx-5090", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = run(args.steps, args.hail_opt, args.require_rtx_5090)
        code = 0 if report["verdict"] == "PASS" else 1
    except Exception as exc:
        report = {
            "verdict": "FAIL",
            "scope": "synthetic full-RK3 WSM6 plus scheduled Dudhia SW",
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 1
    print(json.dumps(_strict_json(report), indent=2, sort_keys=True,
                     allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
