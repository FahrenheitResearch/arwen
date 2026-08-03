#!/usr/bin/env python3
"""Sampling spread of the WP-L7 headline statistics, from within one run.

An IC-perturbed ensemble is the right way to get an ordinary-divergence
envelope, and this is NOT that: it is the cheaper and more limited question of
how much each statistic moves when the averaging window moves, on one
realisation. It answers "is an ArWen-vs-WRF difference larger than this run's
own sampling noise?" and nothing more. Any use of it as an envelope would be
claiming a property it does not have.

Statistics are recomputed over consecutive disjoint windows of the requested
length across the whole run, and the spread over the post-spin-up windows is
reported.

Usage: window_spread.py <run_dir> <out_json> [--win-min 15] [--spinup-min 60]
"""
import sys
import os
import glob
import json

import numpy as np
from netCDF4 import Dataset

G = 9.81


def arr(v):
    return np.asarray(np.ma.getdata(v), dtype=np.float64)


def main():
    run_dir = sys.argv[1]
    out_json = sys.argv[2]
    win = 15.0
    spin = 60.0
    if "--win-min" in sys.argv:
        win = float(sys.argv[sys.argv.index("--win-min") + 1])
    if "--spinup-min" in sys.argv:
        spin = float(sys.argv[sys.argv.index("--spinup-min") + 1])

    files = sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*")))
    nml = open(os.path.join(run_dir, "namelist.input")).read()
    Qs = 0.24
    for line in nml.splitlines():
        if line.strip().startswith("tke_heat_flux"):
            Qs = float(line.split("=")[1].strip().rstrip(","))

    # one pass: per-frame profiles
    frames = []
    for f in files:
        d = Dataset(f)
        xt = arr(d.variables["XTIME"][:])
        for i in range(len(xt)):
            th = arr(d.variables["T"][i]) + 300.0
            w = arr(d.variables["W"][i])
            ph = arr(d.variables["PH"][i])
            phb = arr(d.variables["PHB"][i])
            khv = arr(d.variables["XKHV"][i])
            zw = ((ph + phb) / G).mean(axis=(1, 2))
            zm = 0.5 * (zw[:-1] + zw[1:])
            nz = th.shape[0]
            wm = 0.5 * (w[:-1] + w[1:])
            wp = wm - wm.reshape(nz, -1).mean(axis=1)[:, None, None]
            tb = th.reshape(nz, -1).mean(axis=1)
            tp = th - tb[:, None, None]
            res = (wp * tp).reshape(nz, -1).mean(axis=1)
            sgs = np.zeros(nz + 1)
            for k in range(1, nz):
                Kw = 0.5 * (khv[k] + khv[k - 1])
                sgs[k] = np.mean(-Kw * (th[k] - th[k - 1]) / (zm[k] - zm[k - 1]))
            frames.append(dict(t=float(xt[i]), zm=zm, res=res, sgs=sgs, th=tb))
        d.close()

    t = np.array([fr["t"] for fr in frames])
    tend = t.max()
    results = []
    lo = spin
    while lo + win <= tend + 1e-9:
        idx = [i for i in range(len(t)) if lo < t[i] <= lo + win]
        if len(idx) >= 3:
            zm = np.mean([frames[i]["zm"] for i in idx], axis=0)
            res = np.mean([frames[i]["res"] for i in idx], axis=0)
            sgs = np.mean([frames[i]["sgs"] for i in idx], axis=0)
            th = np.mean([frames[i]["th"] for i in idx], axis=0)
            tot = res + 0.5 * (sgs[:-1] + sgs[1:])
            zi = float(zm[int(np.argmin(tot))])
            band = (zm >= 0.2 * zi) & (zm <= 0.7 * zi)
            thml = float(th[band].mean())
            ws = float((G / thml * Qs * zi) ** (1.0 / 3.0))
            fb = (zm >= 0.1 * zi) & (zm <= 0.7 * zi)
            x = zm[fb] / zi
            y = tot[fb] / Qs
            b, a = np.polyfit(x, y, 1)
            results.append(dict(
                t_lo=lo, t_hi=lo + win, n=len(idx),
                zi_m=zi, wstar=ws, theta_ml=thml,
                flux_fit_intercept=float(a), flux_fit_slope=float(b),
                flux_fit_rms=float(np.sqrt(np.mean((y - (a + b * x)) ** 2))),
                entrainment_min_over_qs=float(tot.min() / Qs),
                wth_res_max_over_qs=float(res.max() / Qs)))
        lo += win

    keys = ["zi_m", "wstar", "flux_fit_intercept", "flux_fit_slope",
            "flux_fit_rms", "entrainment_min_over_qs", "wth_res_max_over_qs"]
    summary = {}
    print("%-28s %10s %10s %10s %10s" % ("statistic", "mean", "sd", "min", "max"))
    for k in keys:
        v = np.array([r[k] for r in results])
        summary[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          min=float(v.min()), max=float(v.max()), n_windows=len(v))
        print("%-28s %10.5g %10.3g %10.5g %10.5g"
              % (k, v.mean(), summary[k]["sd"], v.min(), v.max()))
    out = dict(run_dir=os.path.abspath(run_dir), window_minutes=win,
               spinup_minutes=spin, windows=results, summary=summary,
               disclaimer=("within-run window spread; NOT an IC-perturbed "
                           "ensemble and not an ordinary-divergence envelope"))
    json.dump(out, open(out_json, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
