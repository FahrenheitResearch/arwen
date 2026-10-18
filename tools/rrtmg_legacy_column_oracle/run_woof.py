"""Run WOOF's shipped legacy RRTMG path on the oracle columns.

The adapter ``gpuwm.core.rrtmg_legacy.RRTMGLegacyRadiation`` is called once
per regime exactly as the forecast calls it (``__call__(atmosphere=,
fields=, state=, cfg=)``) on a (1, ncol) grid: device t8w, device
cal_cldfra1, the CAM ozone chain or the wrapper's O3DATA, the aer_opt = 3
device optics, device McICA, the batched CUDA LW and SW engines and the
device output mappings.  Nothing here computes physics.

The process arithmetic is whatever GPUWM_WRF_EXACT says at import (strict
build when 1, default arithmetic when unset); the output records which.
RRTMG_CLOUD_OPTICS_FORM (default wrf_461) selects the wrapper cloud form;
noaa_wrf39 is graded against build.sh's fork-cloud arm.

usage: python run_woof.py COLUMNS_DIR OUT_DIR [REGIME ...]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

F = np.float32
WOOF_OUTPUTS = ("rthratenlw", "rthratensw", "glw", "olr", "gsw", "swdown",
                "coszen", "cldfra", "swddir", "swddif")


def run_regime(cols_dir: Path, meta: dict, name: str) -> dict:
    import cupy as cp
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation

    r = meta["regimes"][name]
    g = dict(np.load(cols_dir / f"columns_{r['grid']}.npz"))
    ncol, nz = meta["ncol"], meta["nz"]
    ny, nx = 1, ncol

    def grid3(a):            # (ncol, nk) -> (nk, 1, ncol) device
        return cp.asarray(np.ascontiguousarray(np.asarray(a, F).T.reshape(-1, ny, nx)))

    def grid2(a):
        return cp.asarray(np.ascontiguousarray(np.asarray(a, F).reshape(ny, nx)))

    present = dict(zip(("qv", "qc", "qr", "qi", "qs", "qg"), r["species"]))
    state = SimpleNamespace(
        fnm=cp.asarray(g["fnm"]), fnp=cp.asarray(g["fnp"]),
        p_top=F(g["p_top"][0]), elapsed_seconds=float(meta["elapsed_seconds"]),
        physics=None)
    for q, on in present.items():
        setattr(state, q, grid3(g[q]) if on else None)
    hc, hi, hs = r["has_req"]
    state.effc = grid3(g["effc"]) if hc else None
    state.effi = grid3(g["effi"]) if hi else None
    state.effs = grid3(g["effs"]) if hs else None
    if r["aer_opt"] == 3:
        state.nwfa = grid3(g["nwfa"])
        state.nifa = grid3(g["nifa"])
    atmosphere = {"pressure": grid3(g["p3d"]), "p_interface": grid3(g["p8w"]),
                  "temperature": grid3(g["t3d"]), "exner": grid3(g["pi3d"]),
                  "dz": grid3(g["dz8w"]), "z_interface": grid3(g["z_at_w"])}
    fields = {k: grid2(g[k]) for k in ("tsk", "emiss", "albedo", "xland", "xice", "snow")}
    fields["swddir"] = cp.zeros((ny, nx), cp.float32)   # ask for SWDDIR/SWDDIF
    cfg = SimpleNamespace(mp_physics=r["mp"], radt=0.0,
                          radt_minutes=meta["radt_minutes"], icloud=1,
                          cldovrlp=2, idcor=0, ghg_input=0, aer_opt=r["aer_opt"],
                          o3input=r["o3input"], use_mp_re=1,
                          sf_surface_physics=2, bl_pbl_physics=0, icloud_bl=0,
                          rrtmg_cloud_optics_form=os.environ.get(
                              "RRTMG_CLOUD_OPTICS_FORM", "wrf_461"),
                          rrtmg_smoke_manifest="", dt=20.0,
                          # The referee is WRF v4.6.1: its sun.
                          solar_generation="wrf_461")
    start = datetime.fromisoformat(meta["start"])
    adapter = RRTMGLegacyRadiation(
        start, g["xlat"].reshape(ny, nx), g["xlong"].reshape(ny, nx),
        p_top=float(g["p_top"][0]), o3input=r["o3input"], aer_opt=r["aer_opt"])
    res = adapter(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    out = {}
    for key in WOOF_OUTPUTS:
        v = getattr(res, key)
        v = cp.asnumpy(v)
        out[key] = (v.reshape(nz, ncol).T.copy() if v.ndim == 3 else v.reshape(ncol))
    out["rthraten"] = cp.asnumpy(res.rthratenlw + res.rthratensw).reshape(nz, ncol).T.copy()
    o3 = adapter._o33d_grid
    if o3 is not None:
        out["o3rad"] = np.asarray(o3, F).reshape(nz, ncol).T.copy()
    return {k: np.asarray(v, F) for k, v in out.items()}


def main():
    cols_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((cols_dir / "regimes.json").read_text())
    names = sys.argv[3:] or list(meta["regimes"])
    import gpuwm
    from gpuwm import wrf_exact
    arm = "strict" if wrf_exact.ENABLED else "default"
    for name in names:
        out = run_regime(cols_dir, meta, name)
        np.savez_compressed(out_dir / f"woof_{arm}_{name}.npz", **out)
        print(f"{arm} {name}: ok", flush=True)
    receipt = wrf_exact.compile_receipt()
    (out_dir / f"woof_{arm}_receipt.json").write_text(json.dumps({
        "arm": arm, "gpuwm": getattr(gpuwm, "__version__", "?"),
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT"),
        "strict_options": receipt["strict_options"] if receipt["enabled"] else [],
        "compile_records": len(receipt["records"])}, indent=2) + "\n")


if __name__ == "__main__":
    main()
