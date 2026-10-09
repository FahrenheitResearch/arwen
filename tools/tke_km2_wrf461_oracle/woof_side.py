"""Run WOOF's production km_opt=2 launcher on every case and arm (GPU).

The launcher is gpuwm.core.dycore.launch_wrf_tke_km, the one the forecast
calls: deformation, calculate_N2, tke_km and tke_rhs on device.  The state
is presented through gpuwm.verify.diffusion_oracle.device_state, which hands
the case arrays to the unmodified launcher.  The arithmetic build is chosen
by the environment before import (GPUWM_WRF_EXACT=1 for the strict build).

Usage: python woof_side.py CASES_DIR OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arms import ARMS  # noqa: E402

STATE_KEYS = ("u", "v", "w", "php", "phb", "alt", "p", "thp", "thb", "qv",
              "qc", "qi", "tke", "mub2d", "mup", "mup0", "msft", "msfu",
              "msfv", "fnm", "fnp", "dn", "dnw", "c1h", "c2h")


def woof_chain(arrays, meta, arm):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import launch_wrf_tke_km
    from gpuwm.verify.diffusion_oracle import device_state
    name, iso, isfflx, ck, cd0, heat = arm
    nx, ny, nz = (int(meta[k]) for k in ("nx", "ny", "nz"))
    # A dry case carries no moisture species: the engine's dry state has
    # qv None, which selects the dry branches (WRF sees zeros in moist).
    keys = [k for k in STATE_KEYS
            if not (meta.get("dry") and k in ("qv", "qc", "qi"))]
    state = device_state({k: arrays[k] for k in keys},
                         cf=tuple(meta[k] for k in ("cf1", "cf2", "cf3")))
    state.physics = SimpleNamespace(fields={
        "ustm": cp.asarray(arrays["ust"]), "hfx": cp.asarray(arrays["hfx"])})
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=meta["dx"], dy=meta["dy"],
                    ztop=meta["ztop"], dt=meta["dt"], run_seconds=0.0,
                    open_x=bool(meta["bx"]), open_y=bool(meta["by"]),
                    km_opt=2, diff_opt=2, bl_pbl_physics=0,
                    sf_sfclay_physics=0 if isfflx == 0 else 1,
                    c_k=ck, mix_isotropic=iso,
                    mix_upper_bound=meta["mix_upper_bound"],
                    isfflx=isfflx, tke_drag_coefficient=cd0,
                    tke_heat_flux=heat)
    km = cp.zeros((nz, ny, nx), cp.float32)
    kh = cp.zeros_like(km)
    state.scratch((nz, ny, nx), "smag_rtke")[...] = 0
    d11, d22, d12 = launch_wrf_tke_km(state, cfg, km, kh, time_t=False)
    bn2 = state.scratch((nz, ny, nx + 1), "diff6_x").reshape(-1)[:nz * ny * nx]
    ii = np.minimum(np.arange(nx + 1), nx - 1) if meta["bx"] else np.arange(nx + 1) % nx
    jj = np.minimum(np.arange(ny + 1), ny - 1) if meta["by"] else np.arange(ny + 1) % ny
    out = {
        "kmh": cp.asnumpy(km), "khh": cp.asnumpy(kh),
        "kmv": cp.asnumpy(state.scratch((nz, ny, nx), "smag_kmv")),
        "khv": cp.asnumpy(state.scratch((nz, ny, nx), "smag_khv")),
        "bn2": cp.asnumpy(bn2.reshape((nz, ny, nx))),
        "rtke": cp.asnumpy(state.scratch((nz, ny, nx), "smag_rtke")),
        "d11": cp.asnumpy(d11), "d22": cp.asnumpy(d22),
        # D12 is staged at (k, j, i) corners for i < nx, j < ny; the WRF
        # array carries the redundant faces, indexed as production wrf_d does.
        "d12": np.ascontiguousarray(cp.asnumpy(d12)[:, jj][:, :, ii]),
    }
    # The inline tensors tke_rhs reads (defor13/23 at w levels, defor33) are
    # exposed by the existing diagnostic probe appended to the production
    # smag2d source: it calls the production device functions and holds no
    # tensor arithmetic of its own.
    from gpuwm.core import dycore
    from gpuwm.core.kernels import module_source
    probe_src = (module_source("smag2d").replace("#undef WRF_SMAG_GRID_ARGS", "")
                 .replace("#undef WRF_SMAG_MAKE_GRID", "")
                 + (HERE.parent / "wrf_diffusion_oracle" / "deformation_probe.cu")
                 .read_text())
    # The probe's device functions are production smag2d's, so it compiles
    # as production smag2d does (DIFFUSION_OPTIONS through the diffusion
    # compile site), as tools/smag2d_wrf461_oracle/oracle.py's probe does.
    from gpuwm.core.kernels import compile_diffusion_source, module_options
    probe = compile_diffusion_source(probe_src, "verify:km2-deformation-probe",
                                     module_options("smag2d"))
    extra = [cp.zeros_like(km), cp.zeros_like(km),
             cp.zeros((nz + 1, ny, nx + 1), cp.float32),
             cp.zeros((nz + 1, ny + 1, nx), cp.float32)]
    probe.get_function("oracle_expose_deformation")(
        ((nx + 128) // 128, ny + 1, nz + 1), (128, 1, 1),
        tuple(dycore._wrf_smag_grid_args(state, cfg, time_t=False) + [d11, d22]
              + extra + [np.int32(nz), np.int32(ny), np.int32(nx),
                         np.int32(state.phb.ndim == 3), np.int32(meta["bx"]),
                         np.int32(meta["by"])]))
    out.update(zip(("d33", "div", "d13", "d23"), [cp.asnumpy(a) for a in extra]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import cupy as cp
    from gpuwm import wrf_exact
    index = json.loads((args.cases / "cases.json").read_text())
    for case in index["cases"]:
        with np.load(args.cases / case["file"]) as data:
            meta = json.loads(str(data["meta_json"]))
            arrays = {k: data[k] for k in data.files if k != "meta_json"}
        for arm in ARMS:
            res = woof_chain(arrays, meta, arm)
            np.savez(args.out / f"woof-{case['name']}-{arm[0]}.npz", **res)
        print(case["name"], flush=True)
    receipt = {
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "GPUWM_WRF_EXACT_DIFFUSION": os.environ.get("GPUWM_WRF_EXACT_DIFFUSION", ""),
        "strict_options": list(wrf_exact.STRICT_OPTIONS) if wrf_exact.ENABLED else [],
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy": cp.__version__,
    }
    (args.out / "woof-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
