"""Short full-dycore WSM6 integration smoke for a rented CUDA GPU.

The case is deliberately small and synthetic.  A balanced WK82 moist bubble
is seeded with a smooth mixed-phase cloud, then advanced through complete
RK3/acoustic/scalar-advection/microphysics steps.  It is an integration and
gross-stability gate, not a trajectory-validation replacement.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time

import numpy as np


def _device_name(cp) -> str:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    return name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)


def _smoke_failures(*, report: dict, finite: bool, minimum: float,
                    initial_maxima: dict[str, float],
                    changes: dict[str, float],
                    total_condensate_max: float, h_diabatic_max: float,
                    elapsed_error: float) -> list[str]:
    """Evaluate the smoke without imposing nonphysical category survival.

    WSM6 categories are process reservoirs, not invariants.  In particular,
    direct WRF v4.6.1 consumes every qc level in the seeded mixed-phase
    oracle after 10 calls x 1.5 s while retaining other condensate.  The gate
    therefore proves that all seed categories were present, the coupled state
    remains valid, condensate remains, and microphysics actually changed it;
    it must not require each initial category to survive conversion.
    """
    failures: list[str] = []
    seeded = ("qc", "qi", "qr", "qs", "qg")
    if report["nan"] or not finite:
        failures.append("non-finite dycore or WSM6 state")
    if minimum < -1.0e-10:
        failures.append(
            f"hydrometeor minimum {minimum:.9g} is below -1e-10 kg kg-1")
    for q in seeded:
        value = initial_maxima[q]
        if not np.isfinite(value) or value <= 0.0:
            failures.append(f"mixed-phase smoke failed to seed {q}")
    if (not np.isfinite(total_condensate_max)
            or total_condensate_max <= 0.0):
        failures.append("all condensate vanished from the mixed-phase smoke")
    if max(changes.values()) <= 1.0e-10:
        failures.append("no moisture field changed across full model steps")
    if h_diabatic_max <= 0.0 or not np.isfinite(h_diabatic_max):
        failures.append("microphysics heating handoff is zero or non-finite")
    if elapsed_error > 1.0e-6:
        failures.append(
            f"model clock mismatch {elapsed_error:.9g} s")
    return failures


def _build_state(hail_opt: int):
    import cupy as cp

    from gpuwm.config import validate_run_config
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import moist_bubble
    from gpuwm.verify.cases.wk82 import wk82_sounding

    cfg = replace(
        moist_bubble.default_config(),
        nx=32, ny=8, nz=40, dt=1.5, run_seconds=15.0,
        time_step_sound=4, mp_physics=6, wsm6_hail_opt=hail_opt,
        km_opt=1, khdif=0.0, kvdif=0.0,
        diff_6th_opt=0, damp_opt=0,
        case="wsm6_gpu_smoke",
    )
    cfg = validate_run_config(cfg)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: wk82_sounding(z)[0],
        p_surf=cfg.p_surf, ztop=cfg.ztop,
    )
    state = moist_bubble.build(cfg, coord, base)

    # Smooth, nonuniform mixed-phase seed.  It is small enough to avoid a
    # shock but nonzero in all WSM6 categories, so the full interface and
    # process/sedimentation paths cannot silently disappear from this smoke.
    z = state.height_half()[:, None, None]
    x = ((np.arange(cfg.nx) + 0.5) * cfg.dx
         - 0.5 * cfg.nx * cfg.dx)[None, None, :]
    y = ((np.arange(cfg.ny) + 0.5) * cfg.dy
         - 0.5 * cfg.ny * cfg.dy)[None, :, None]
    cloud = np.exp(
        -((x / 5000.0) ** 2 + (y / 2500.0) ** 2
          + ((z - 6000.0) / 1800.0) ** 2)
    ).astype(np.float32)
    state.qc += cp.asarray(np.float32(2.0e-4) * cloud)
    state.qi += cp.asarray(np.float32(5.0e-5) * cloud)
    state.qr += cp.asarray(np.float32(8.0e-5) * cloud)
    state.qs += cp.asarray(np.float32(6.0e-5) * cloud)
    state.qg += cp.asarray(np.float32(2.0e-5) * cloud)
    return cfg, state


def run(steps: int, hail_opt: int, require_rtx_5090: bool) -> dict:
    import cupy as cp

    from gpuwm.core.dycore import run_steps, stability_report

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if hail_opt not in (0, 1):
        raise ValueError(f"hail_opt must be 0 or 1, got {hail_opt}")
    name = _device_name(cp)
    if require_rtx_5090 and "RTX 5090" not in name.upper():
        raise RuntimeError(
            f"--require-rtx-5090 requested, but CUDA device is {name!r}")

    free0, total = cp.cuda.runtime.memGetInfo()
    started = time.perf_counter()
    cfg, state = _build_state(hail_opt)
    initial = {
        species: getattr(state, species).copy()
        for species in ("qv", "qc", "qi", "qr", "qs", "qg")
    }
    initial_maxima = {
        species: float(field.max()) for species, field in initial.items()
    }
    run_steps(state, cfg, n=steps)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    report = stability_report(state, cfg)

    species = ("qv", "qc", "qi", "qr", "qs", "qg")
    finite = all(bool(cp.isfinite(getattr(state, q)).all()) for q in species)
    minimum = min(float(getattr(state, q).min()) for q in species)
    maxima = {q: float(getattr(state, q).max()) for q in species}
    changes = {
        q: float(cp.max(cp.abs(getattr(state, q) - initial[q])))
        for q in species
    }
    total_condensate_max = float(cp.max(
        state.qc + state.qi + state.qr + state.qs + state.qg))
    h_diabatic_max = float(cp.max(cp.abs(state.h_diabatic)))
    elapsed_error = abs(state.elapsed_seconds - steps * cfg.dt)
    failures = _smoke_failures(
        report=report, finite=finite, minimum=minimum,
        initial_maxima=initial_maxima, changes=changes,
        total_condensate_max=total_condensate_max,
        h_diabatic_max=h_diabatic_max, elapsed_error=elapsed_error)

    free1, _ = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return {
        "ok": not failures,
        "scope": "synthetic seeded mixed-phase full-RK3 integration smoke",
        "not_a_claim": "WRF trajectory parity or production stability",
        "device": name,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "grid": [cfg.nz, cfg.ny, cfg.nx],
        "dt_seconds": cfg.dt,
        "steps": steps,
        "simulated_seconds": state.elapsed_seconds,
        "elapsed_seconds_including_jit": elapsed,
        "device_memory_total_bytes": int(total),
        "device_memory_delta_bytes": int(free0 - free1),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
        "cupy_pool_held_bytes": int(pool.total_bytes()),
        "stability": report,
        "species_initial_max_kg_kg-1": initial_maxima,
        "species_max_kg_kg-1": maxima,
        "species_max_change_kg_kg-1": changes,
        "species_min_kg_kg-1": minimum,
        "total_condensate_max_kg_kg-1": total_condensate_max,
        "h_diabatic_max_K_s-1": h_diabatic_max,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--hail-opt", type=int, choices=(0, 1), default=0)
    parser.add_argument("--require-rtx-5090", action="store_true")
    args = parser.parse_args()
    result = run(args.steps, args.hail_opt, args.require_rtx_5090)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
