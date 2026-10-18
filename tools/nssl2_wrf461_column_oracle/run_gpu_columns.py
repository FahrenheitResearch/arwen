"""Run WOOF's production mp_physics=18 path over the oracle column set.

This drives the artifact, not a re-wiring of it: a real ``DomainState`` and
``PhysicsDriver`` (``initialize_physics``), the persistent production
binding, and ``gpuwm.core.microphysics.apply`` -- the same call the forecast
loop makes -- with the columns laid along x (``ny = 1``).

The microphysics-time inputs the scheme actually sees (full theta, dry
density, Exner, dz8w, pressure, staggered w) are formed by WOOF's own
``_prepare_fields`` from the DomainState.  They are read back and written as
the Fortran driver's input stream, so both engines start from the same FP32
bytes; preparation itself is outside the scheme and outside this oracle.

Outputs (scheme outputs only, before ``moist_physics_finish``):
  theta     post-scheme full theta (the ``mp_th`` slab WRF returns as th)
  16 Registry moments (#/kg, kg/kg, m3/kg), REFL_10CM (dBZ),
  re_cloud/re_ice/re_snow in metres (captured before WOOF's micron
  conversion, which WRF does not have), and the nine precipitation
  accumulators/rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUT3 = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
        "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh", "refl_10cm",
        "re_cloud", "re_ice", "re_snow")
OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv",
        "hailnc", "hailncv", "sr")
REGISTRY = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr", "qni",
            "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")
FORTRAN_INPUT = ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
                 "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
                 "qvolg", "qvolh", "pressure", "exner", "rho", "dz")


def production_step(src, dt_s: float, itimestep: int, stages: bool = False):
    """One WOOF mp_physics=18 step through ``microphysics.apply``.

    ``src`` maps column-major ``(ncol, nz)`` arrays: theta, pressure, rho,
    dz, the 16 Registry moments, ``w`` ``(ncol, nz + 1)`` and the four
    accumulators rainnc/snownc/grplnc/hailnc ``(ncol,)``.  Returns
    ``(out, fin, w_in, snaps)``: the scheme outputs, the prepared inputs WOOF
    actually used (the Fortran input), the staggered w, and (with
    ``stages``) the stage snapshots.
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core import microphysics, nssl2_runtime
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.refl import consume_refl_10cm
    from gpuwm.core.state import DomainState

    ncol, nz = src["theta"].shape
    g = np.float32(9.81)
    cfg = RunConfig(nx=ncol, ny=1, nz=nz, dx=3000.0, dy=3000.0,
                    ztop=float(src["dz"][0].astype(np.float64).sum()),
                    dt=float(dt_s), run_seconds=float(dt_s), moist=True,
                    mp_physics=18)
    state = DomainState(cfg)

    def col3(a):  # (ncol, nzx) -> (nzx, 1, ncol)
        return cp.asarray(np.ascontiguousarray(a.T[:, None, :]), dtype=cp.float32)

    state.thb[...] = cp.float32(0.0)
    state.thp[...] = col3(src["theta"])
    state.p[...] = col3(src["pressure"])
    state.alt[...] = col3((np.float32(1.0) / src["rho"]).astype(np.float32))
    state.phb[...] = cp.float32(0.0)
    zi = np.concatenate([np.zeros((ncol, 1)), np.cumsum(src["dz"].astype(np.float64), axis=1)], axis=1)
    state.php[...] = col3((zi * 9.81).astype(np.float32))
    state.w[...] = col3(src["w"])
    for name in REGISTRY:
        getattr(state, name)[...] = col3(src[name])

    driver = initialize_physics(state, cfg)
    driver.microphysics_updates = 0 if itimestep == 1 else itimestep - 1
    for comp, field in (("rainnc", "rainnc"), ("snownc", "snownc"),
                        ("graupelnc", "grplnc"), ("hailnc", "hailnc")):
        getattr(driver.microphysics, comp)[...] = cp.asarray(
            src[field][None, :], dtype=cp.float32)

    shape = state.p.shape
    prepared = nssl2_runtime._prepare_fields(state, shape)
    cp.cuda.Stream.null.synchronize()

    def host(a):  # (nzx, 1, ncol) -> (ncol, nzx)
        return np.ascontiguousarray(cp.asnumpy(a)[:, 0, :].T).astype(np.float32)

    fin = {name: src[name].astype(np.float32) for name in REGISTRY}
    fin.update(theta=host(prepared.theta), pressure=host(prepared.pressure),
               exner=host(prepared.pii), rho=host(prepared.rho),
               dz=host(prepared.dz))
    w_in = host(state.w)

    captured = {}
    original_convert = nssl2_runtime._convert_radii_to_microns

    def capture_radii(st, *, validate_values):
        cp.cuda.Stream.null.synchronize()
        captured["re_cloud"] = host(st.effc)
        captured["re_ice"] = host(st.effi)
        captured["re_snow"] = host(st.effs)
        captured["theta"] = host(st._scratch["mp_th"])
        return original_convert(st, validate_values=validate_values)

    # Optional stage capture, for attribution only: wrap the two module-level
    # launchers the production hooks call, copying the driver workspace
    # (concentration space) and the temperature around each.
    from gpuwm.core import nssl2_default_hooks, nssl2_fused_gs
    snaps = {}
    original_fused = nssl2_fused_gs.launch_fused_gs
    original_nucond = nssl2_default_hooks.launch_nucond

    def snap(stage, ws_state, theta, pii, t0=None):
        cp.cuda.Stream.null.synchronize()
        snaps[f"stage{stage}_state"] = cp.asnumpy(ws_state)[:, :, 0, :]
        snaps[f"stage{stage}_theta"] = host(theta)
        snaps[f"stage{stage}_t0"] = host(theta * pii if t0 is None else t0)

    def fused(workspace, full_theta, air_density, pressure_pa, exner, *a, **k):
        snap(1, workspace.state, full_theta, exner)
        original_fused(workspace, full_theta, air_density, pressure_pa, exner, *a, **k)
        # After GS, WRF's t0 is the GS scatter's temperature (the pre-melt
        # temg it carries), which WOOF keeps in the fused-GS temperature
        # buffer -- not theta*pii.  launch_fused_gs(ws, theta, rho, p, pii,
        # w, temperature_k, ...): temperature_k is a[1].
        snap(2, workspace.state, full_theta, exner, t0=a[1])

    def nucond(full_theta, air_density, pressure_pa, exner, *a, **k):
        original_nucond(full_theta, air_density, pressure_pa, exner, *a, **k)
        ws = state._scratch["nssl2_driver_state"]
        snap(3, ws, full_theta, exner)

    nssl2_runtime._convert_radii_to_microns = capture_radii
    if stages:
        nssl2_fused_gs.launch_fused_gs = fused
        nssl2_default_hooks.launch_nucond = nucond
    try:
        diags = microphysics.apply(state, cfg, float(dt_s), refl_10cm_due=True)
    finally:
        nssl2_runtime._convert_radii_to_microns = original_convert
        nssl2_fused_gs.launch_fused_gs = original_fused
        nssl2_default_hooks.launch_nucond = original_nucond
    cp.cuda.Stream.null.synchronize()

    out = {name: host(getattr(state, name)) for name in REGISTRY}
    out["theta"] = captured["theta"]
    out["refl_10cm"] = host(consume_refl_10cm(state))
    for name in ("re_cloud", "re_ice", "re_snow"):
        out[name] = captured[name]
    for name in OUT2:
        out[name] = cp.asnumpy(getattr(diags, name))[0].astype(np.float32)
    return out, fin, w_in, snaps


def run(columns: Path, dt_s: float, itimestep: int, out_prefix: Path,
        stages: bool = False) -> dict:
    import cupy as cp

    src = np.load(columns, allow_pickle=False)
    ncol, nz = src["theta"].shape
    out, fin, w_in, snaps = production_step(src, dt_s, itimestep, stages)
    if stages:
        np.savez(out_prefix.with_name(out_prefix.name + ".gpu_stages.npz"), **snaps)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_prefix.with_name(out_prefix.name + ".gpu.npz"), **out)
    with out_prefix.with_name(out_prefix.name + ".fin.bin").open("wb") as fh:
        fh.write(np.asarray([nz, ncol], dtype="<i4").tobytes())
        for name in FORTRAN_INPUT:
            fh.write(fin[name].astype("<f4").tobytes())
        fh.write(w_in.astype("<f4").tobytes())
        for name in ("rainnc", "snownc", "grplnc", "hailnc"):
            fh.write(src[name].astype("<f4").tobytes())
    np.savez(out_prefix.with_name(out_prefix.name + ".fin.npz"), **fin, w=w_in)

    from gpuwm import wrf_exact  # noqa: F401  (import-only; reads env)
    meta = {
        "columns": str(columns), "ncol": int(ncol), "nz": int(nz),
        "dt_s": float(np.float32(dt_s)), "itimestep": itimestep,
        "gpuwm_wrf_exact": os.environ.get("GPUWM_WRF_EXACT", "0"),
        "strict_compiler_hook_installed": bool(
            getattr(__import__("cupy.cuda.compiler", fromlist=["x"]),
                    "_gpuwm_wrf_exact", False)),
        "device": cp.cuda.runtime.getDeviceProperties(
            cp.cuda.runtime.getDevice())["name"].decode(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "fin_sha256": hashlib.sha256(out_prefix.with_name(
            out_prefix.name + ".fin.bin").read_bytes()).hexdigest(),
    }
    out_prefix.with_name(out_prefix.name + ".gpu.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("columns", type=Path)
    p.add_argument("out_prefix", type=Path)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--itimestep", type=int, default=2)
    p.add_argument("--stages", action="store_true",
                   help="also write per-stage driver snapshots for attribution")
    a = p.parse_args()
    run(a.columns, a.dt, a.itimestep, a.out_prefix, a.stages)


if __name__ == "__main__":
    main()
