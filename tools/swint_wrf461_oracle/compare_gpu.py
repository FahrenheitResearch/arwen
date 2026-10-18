"""Run WOOF's swint_opt = 1 kernels over the oracle's columns and compare
every output word with WRF v4.6.1's Fortran.

Usage: python compare_gpu.py BUILD_DIRECTORY OUT.json [--wrf O0|O2vec]

Reads BUILD_DIRECTORY/columns.npz and BUILD_DIRECTORY/<scenario>.<build>.out.bin
(run_swint.F90).  WOOF is driven through its production entry points,
gpuwm.core.swint.ShortwaveInterpolation.radiation_step on a radiation step
and .between_calls on every other step, with the scenario's start time,
latitude and longitude, the live ALBEDO of each step in fields['albedo'],
and the shortwave scheme's SWDOWN/SWDDIR as inputs.  The radiation step's
own COSZEN (WRF's calc_coszen at xtime + radt*0.5) is given to WOOF from
the WRF output so the fit is graded on identical anchors; the kernel's own
value of that cosine is graded separately (``coszen_rad_kernel``), and so
is the production radiation adapter's host cosine
(gpuwm.core.rrtmg_legacy.calc_coszen, ``coszen_rad_rrtmg_adapter``), which
is the RRTMG adapter's documented seam, not this scheme's code.

Graded per step: coszen_loc, bb, bx, gg, gx, coszen_ref, swdown_ref,
swddir_ref, swdown, swddir, swddni, swddif, gsw.  Bitwise (uint32) equality
and the maximum ULP distance (ordered-integer metric; +0 and -0 equal) and
absolute difference are recorded per field, per scenario and per regime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

F = np.float32
ROWS = ("coszen_loc", "coszen", "bb", "bx", "gg", "gx", "coszen_ref",
        "swdown_ref", "swddir_ref", "swdown", "swddir", "swddni", "swddif",
        "gsw")
GRADED = tuple(r for r in ROWS if r != "coszen")
STATE = {"bb": "swint_bb", "bx": "swint_bx", "gg": "swint_gg",
         "gx": "swint_gx", "coszen_ref": "coszen_ref",
         "swdown_ref": "swdown_ref", "swddir_ref": "swddir_ref"}
FLUX = ("swdown", "swddir", "swddni", "swddif", "gsw")


def ordered(a):
    i = np.asarray(a, F).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def grade(got, want):
    got = np.asarray(got, F)
    want = np.asarray(want, F)
    differ = got.view(np.uint32) != want.view(np.uint32)
    nan = np.isnan(got) | np.isnan(want)
    ulp = np.abs(ordered(got) - ordered(want))
    ulp = np.where(differ, ulp, 0)
    absd = np.abs(got.astype(np.float64) - want.astype(np.float64))
    absd = np.where(differ & ~nan, absd, 0.0)
    return {
        "words": int(got.size),
        "differ": int(differ.sum()),
        "nan_mismatch": int((np.isnan(got) != np.isnan(want)).sum()),
        "max_ulp": int(ulp.max()) if ulp.size else 0,
        "max_abs": float(absd.max()) if absd.size else 0.0,
    }


def merge(a, b):
    if a is None:
        return dict(b)
    return {"words": a["words"] + b["words"],
            "differ": a["differ"] + b["differ"],
            "nan_mismatch": a["nan_mismatch"] + b["nan_mismatch"],
            "max_ulp": max(a["max_ulp"], b["max_ulp"]),
            "max_abs": max(a["max_abs"], b["max_abs"])}


def read_wrf(path, nstep, ncol):
    raw = np.fromfile(path, dtype=F)
    expect = nstep * len(ROWS) * ncol
    if raw.size != expect:
        raise SystemExit(f"{path}: {raw.size} words, expected {expect}")
    return raw.reshape(nstep, len(ROWS), ncol)


def run_woof(cols, name, wrf):
    import cupy as cp
    from gpuwm.core import swint
    from gpuwm.core.rrtmg_legacy import calc_coszen, radconst

    start = datetime.fromisoformat(str(cols[f"{name}/start"]))
    lat = cols[f"{name}/xlat"]
    lon = cols[f"{name}/xlon"]
    elapsed = cols[f"{name}/elapsed"]
    julian = cols[f"{name}/julian"]
    xtime = cols[f"{name}/xtime"]
    gmt = cols[f"{name}/gmt"]
    radflag = cols[f"{name}/radflag"]
    swdown_in = cols[f"{name}/swdown"]
    swddir_in = cols[f"{name}/swddir"]
    albedo = cols[f"{name}/albedo"]
    radt = F(cols["radt_minutes"])
    ncol = lat.size
    nstep = julian.size
    shape = (1, ncol)

    # the calendar the engine forms from start time and elapsed seconds is
    # the one the Fortran was given
    for s in range(nstep):
        j, x, g = swint.calendar_scalars(start, float(elapsed[s]))
        assert (j, x, g) == (julian[s], xtime[s], gmt[s]), (name, s)

    carrier = swint.ShortwaveInterpolation(
        start_time=start, latitude_deg=lat.reshape(shape),
        longitude_deg=lon.reshape(shape), shape=shape)
    fields = {k: cp.zeros(shape, dtype=cp.float32)
              for k in (*swint.STATE_FIELDS, "swdown", "gsw", "albedo")}
    out = np.zeros((nstep, len(ROWS), ncol), F)
    krad, wrad, arad = [], [], []
    icall = 0
    for s in range(nstep):
        fields["albedo"][...] = cp.asarray(albedo[s].reshape(shape))
        if radflag[s]:
            fields["swdown"][...] = cp.asarray(swdown_in[icall].reshape(shape))
            carrier.radiation_step(
                fields, coszen=cp.asarray(wrf[s, 1].reshape(shape)),
                swddir=cp.asarray(swddir_in[icall].reshape(shape)),
                elapsed_seconds=float(elapsed[s]))
            icall += 1
        else:
            carrier.between_calls(fields, elapsed_seconds=float(elapsed[s]))
        out[s, 0] = cp.asnumpy(carrier._coszen_loc).reshape(-1)
        out[s, 1] = wrf[s, 1]
        for r, key in STATE.items():
            out[s, ROWS.index(r)] = cp.asnumpy(fields[key]).reshape(-1)
        for r in FLUX:
            out[s, ROWS.index(r)] = cp.asnumpy(fields[r]).reshape(-1)
        if radflag[s]:
            xt_mid = F(xtime[s] + F(radt * F(0.5)))
            k = cp.asnumpy(carrier.coszen_loc_at(julian[s], xt_mid, gmt[s]))
            krad.append(k.reshape(-1).copy())
            wrad.append(wrf[s, 1].copy())
            declin, _ = radconst(julian[s])
            arad.append(calc_coszen(julian[s], xt_mid, gmt[s], lat, lon,
                                    declin))
    return out, np.stack(krad), np.stack(wrad), np.stack(arad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("out")
    ap.add_argument("--wrf", default="O0")
    ap.add_argument("--save", default=None,
                    help="optional .npz for WOOF's outputs")
    args = ap.parse_args()
    build = Path(args.build)
    cols = np.load(build / "columns.npz")
    report = {"wrf_build": args.wrf,
              "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
              "scenarios": {}, "fields": {}, "regimes": {},
              "coszen_rad_kernel": None, "coszen_rad_rrtmg_adapter": None,
              "columns": 0, "steps_x_columns": 0}
    saved = {}
    for name in [str(s) for s in cols["scenarios"]]:
        ncol = cols[f"{name}/xlat"].size
        nstep = cols[f"{name}/julian"].size
        wrf = read_wrf(build / f"{name}.{args.wrf}.out.bin", nstep, ncol)
        woof, krad, wrad, arad = run_woof(cols, name, wrf)
        saved[f"{name}/woof"] = woof
        labels = cols[f"{name}/label"].astype(str)
        sc = {}
        for r in GRADED:
            i = ROWS.index(r)
            g = grade(woof[:, i], wrf[:, i])
            sc[r] = g
            report["fields"][r] = merge(report["fields"].get(r), g)
            for lab in np.unique(labels):
                m = labels == lab
                key = f"{name}/{lab}"
                report["regimes"][key] = merge(
                    report["regimes"].get(key), grade(woof[:, i][:, m],
                                                      wrf[:, i][:, m]))
        sc["coszen_rad_kernel"] = grade(krad, wrad)
        sc["coszen_rad_rrtmg_adapter"] = grade(arad, wrad)
        report["coszen_rad_kernel"] = merge(report["coszen_rad_kernel"],
                                            sc["coszen_rad_kernel"])
        report["coszen_rad_rrtmg_adapter"] = merge(
            report["coszen_rad_rrtmg_adapter"], sc["coszen_rad_rrtmg_adapter"])
        # where WOOF and WRF part, which columns and why (first few)
        bad = np.zeros(ncol, bool)
        for r in GRADED:
            i = ROWS.index(r)
            bad |= (woof[:, i].view(np.uint32) != wrf[:, i].view(np.uint32)).any(0)
        sc["columns_with_any_difference"] = int(bad.sum())
        sc["regimes_with_difference"] = sorted(set(labels[bad].tolist()))
        sc["columns"] = ncol
        sc["steps"] = nstep
        report["scenarios"][name] = sc
        report["columns"] += ncol
        report["steps_x_columns"] += ncol * nstep
        print(f"{name}: {ncol} columns x {nstep} steps; columns with any "
              f"difference {int(bad.sum())}", flush=True)
    total = None
    for g in report["fields"].values():
        total = merge(total, g)
    report["total"] = total
    report["exact"] = total["differ"] == 0 and total["nan_mismatch"] == 0
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.save:
        np.savez_compressed(args.save, **saved)
    print(json.dumps({"total": total, "exact": report["exact"],
                      "fields": report["fields"],
                      "coszen_rad_kernel": report["coszen_rad_kernel"],
                      "coszen_rad_rrtmg_adapter":
                          report["coszen_rad_rrtmg_adapter"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
