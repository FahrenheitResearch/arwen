"""Grade the lane's CUDA WSM6 against WRF v4.6.1 on the refute column sets.

usage: python refute_run.py SET_DIR RUNS [--ny N] [--tier T] [--out JSON]
RUNS = "h,dt,n;h,dt,n;..."  (WRF outputs wrf-O0-h{h}-dt{dt}-n{n}.bin in SET_DIR)

Uses the lane's own grading helpers (tools.wsm6_wrf461_column_oracle.run_gpu)
for reading WRF output and per-word comparison, and the production
launchers.  --ny reshapes the columns to (nz, ny, ncol/ny) so the kernel's
2-D indexing is exercised.  Every word, every step, every field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.wsm6_wrf461_column_oracle import run_gpu as rg

TINY = np.float32(1.1754944e-38)


def run_gpu(columns, hail, dt, nsteps, ny):
    import cupy as cp
    from gpuwm.core.refl import launch_refl10cm_wsm6
    from gpuwm.core.wsm6 import launch_wsm6

    ncol, nz, level, surf = rg.read_columns(columns)
    assert ncol % ny == 0
    nx = ncol // ny
    f32 = np.float32
    dev = {n: cp.asarray(v.reshape(nz, ny, nx), dtype=cp.float32) for n, v in level.items()}
    s = {n: cp.zeros((ny, nx), dtype=cp.float32) for n in
         ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")}
    s["rainnc"][...] = cp.asarray(surf["rain"].reshape(ny, nx))
    s["snownc"][...] = cp.asarray(surf["snow"].reshape(ny, nx))
    s["graupelnc"][...] = cp.asarray(surf["graupel"].reshape(ny, nx))
    # garbage initial radii: WRF's effectRad must overwrite every level
    effc = cp.full((nz, ny, nx), f32(2.49), cp.float32)
    effi = cp.full((nz, ny, nx), f32(4.99), cp.float32)
    effs = cp.full((nz, ny, nx), f32(9.99), cp.float32)
    refl = cp.full((nz, ny, nx), f32(np.nan), cp.float32)
    t = cp.empty((nz, ny, nx), cp.float32)
    steps = []
    for _ in range(nsteps):
        launch_wsm6(dev["th"], dev["qv"], dev["qc"], dev["qr"], dev["qi"], dev["qs"], dev["qg"],
                    dev["den"], dev["pii"], dev["p"], dev["dz"], s["rainnc"], s["rainncv"],
                    s["snownc"], s["snowncv"], s["graupelnc"], s["graupelncv"], s["sr"],
                    dt, effc=effc, effi=effi, effs=effs, hail_opt=hail)
        cp.multiply(dev["th"], dev["pii"], out=t)
        launch_refl10cm_wsm6(dev["qv"], dev["qr"], dev["qs"], dev["qg"], t, dev["p"], refl,
                             hail_opt=hail, melting=True)
        cp.cuda.runtime.deviceSynchronize()
        g = lambda a: a.get().reshape(nz, ncol)
        lv = {"th": g(dev["th"]), "qv": g(dev["qv"]), "qc": g(dev["qc"]), "qi": g(dev["qi"]),
              "qr": g(dev["qr"]), "qs": g(dev["qs"]), "qg": g(dev["qg"]), "re_cloud": g(effc),
              "re_ice": g(effi), "re_snow": g(effs), "refl_10cm": g(refl), "t": g(t)}
        sf = {"rain": s["rainnc"], "rainncv": s["rainncv"], "snow": s["snownc"],
              "snowncv": s["snowncv"], "graupel": s["graupelnc"],
              "graupelncv": s["graupelncv"], "sr": s["sr"]}
        steps.append((lv, {k: v.get().reshape(ncol) for k, v in sf.items()}))
    return steps


def subnormal_columns(level, wrf):
    """Columns whose input or WRF trajectory holds a subnormal word anywhere."""
    def sub(a):
        a = np.abs(np.asarray(a, np.float32))
        return (a > 0) & (a < TINY)
    flag = np.zeros(next(iter(level.values())).shape[1], bool)
    for v in level.values():
        flag |= sub(v).any(axis=0)
    for lv, sf in wrf["steps"]:
        for v in lv.values():
            flag |= sub(v).any(axis=0)
        for v in sf.values():
            flag |= sub(v)
    return flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path)
    ap.add_argument("runs")
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--tier", type=int)
    ap.add_argument("--flavour", default="O0")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.tier:
        rg.force_tier(a.tier)
    regimes = json.loads((a.set_dir / "columns.json").read_text())["columns"]
    ncol, nz, level, _ = rg.read_columns(a.set_dir / "columns.bin")
    report = {**rg._header(), "set": a.set_dir.name, "ncol": ncol, "nz": nz, "ny": a.ny,
              "tier": a.tier, "flavour": a.flavour, "runs": {}}
    grand = grand_normal = words = 0
    for spec in a.runs.split(";"):
        h, dt, n = spec.split(",")
        h, dt, n = int(h), float(dt), int(n)
        tag = f"h{h}-dt{dt:g}-n{n}"
        wrf = rg.read_wrf(a.set_dir / f"wrf-{a.flavour}-{tag}.bin")
        assert wrf["ncol"] == ncol and wrf["nz"] == nz and wrf["nsteps"] == n
        gpu = run_gpu(a.set_dir / "columns.bin", h, dt, n, a.ny)
        fields = rg.grade(wrf, gpu, regimes)
        subcol = subnormal_columns(level, wrf)
        keep = ~subcol
        fn = {}
        for istep, (ws, gs) in enumerate(zip(wrf["steps"], gpu)):
            if keep.any():
                rg._accumulate(fn, istep, rg._pairs(ws, gs), regimes, keep=keep)
        bad = sum(r["bit_mismatch"] for r in fields.values())
        badn = sum(r["bit_mismatch"] for r in fn.values())
        w = sum(r["words"] for r in fields.values())
        grand += bad
        grand_normal += badn
        words += w
        report["runs"][tag] = {"words": w, "bit_mismatch": bad,
                               "subnormal_columns": [regimes[i]["name"] for i in np.nonzero(subcol)[0]],
                               "bit_mismatch_outside_subnormal_columns": badn,
                               "fields": fields, "fields_outside_subnormal_columns": fn}
        print(f"{tag}: words {w} differ {bad} (outside {int(subcol.sum())} subnormal cols: {badn}); "
              + ", ".join(f"{k} {r['bit_mismatch']}/{r['max_ulp']}u" for k, r in fields.items()
                          if r["bit_mismatch"]), flush=True)
    report.update(total_words=words, total_bit_mismatch=grand,
                  total_outside_subnormal_columns=grand_normal)
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("set", "strict_build", "nvrtc", "device", "tier", "ny",
                                             "total_words", "total_bit_mismatch",
                                             "total_outside_subnormal_columns")}))


if __name__ == "__main__":
    main()
