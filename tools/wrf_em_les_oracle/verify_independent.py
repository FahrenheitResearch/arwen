#!/usr/bin/env python3
"""Second-opinion verifier for the WP-L7 oracle receipts.

Deliberately NOT a refactor of score_wrf_les.py. It recomputes the headline
numbers by different routes so that an implementation mistake in one shows up
as a disagreement rather than being reproduced identically:

  * z_i by three independent definitions, not one
  * SGS heat flux by plain centred differencing with an arithmetic-mean K,
    instead of WRF's fnm/fnp weights
  * the flux-profile linear fit by normal equations rather than np.polyfit
  * Parseval closure checked against a real-space variance computed separately
  * the mass drift recomputed from the raw MU/MUB arrays

Imports nothing from gpuwm and nothing from the primary scorer.

Usage: verify_independent.py <run_dir> <primary_json>
"""
import sys
import glob
import os
import json

import numpy as np
from netCDF4 import Dataset


def arr(v):
    """netCDF4 hands back MaskedArrays; strip the mask, keep float64."""
    return np.asarray(np.ma.getdata(v), dtype=np.float64)


G = 9.81


def main():
    run_dir, primary = sys.argv[1], sys.argv[2]
    ref = json.load(open(primary))
    files = sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*")))
    d = Dataset(files[-1])
    nt = d.dimensions["Time"].size
    xt = arr(d.variables["XTIME"][:])
    Qs = ref["surface_heat_flux"]

    # frames inside the final 30 minutes
    sel = [i for i in range(nt) if xt[i] >= xt[-1] - 30.0 - 1e-9]

    acc_res = None
    acc_sgs = None
    acc_th = None
    acc_z = None
    for i in sel:
        th = arr(d.variables["T"][i]) + 300.0
        w = arr(d.variables["W"][i])
        ph = arr(d.variables["PH"][i])
        phb = arr(d.variables["PHB"][i])
        khv = arr(d.variables["XKHV"][i])
        zw = ((ph + phb) / G).mean(axis=(1, 2))
        zm = 0.5 * (zw[:-1] + zw[1:])
        nz = th.shape[0]

        # resolved flux: build w on mass levels by an explicit loop-free slice,
        # subtract per-level means computed with a separate reduction
        wm = 0.5 * (w[:-1] + w[1:])
        wbar = wm.reshape(nz, -1).mean(axis=1)
        tbar = th.reshape(nz, -1).mean(axis=1)
        res = ((wm.reshape(nz, -1) - wbar[:, None])
               * (th.reshape(nz, -1) - tbar[:, None])).mean(axis=1)

        # SGS flux: centred difference, arithmetic-mean K (NOT fnm/fnp)
        sgs = np.zeros(nz + 1)
        for k in range(1, nz):
            Kw = 0.5 * (khv[k] + khv[k - 1])
            sgs[k] = np.mean(-Kw * (th[k] - th[k - 1]) / (zm[k] - zm[k - 1]))

        acc_res = res if acc_res is None else acc_res + res
        acc_sgs = sgs if acc_sgs is None else acc_sgs + sgs
        acc_th = tbar if acc_th is None else acc_th + tbar
        acc_z = zm if acc_z is None else acc_z + zm

    n = len(sel)
    res = acc_res / n
    sgs = acc_sgs / n
    thb = acc_th / n
    zm = acc_z / n
    total = res + 0.5 * (sgs[:-1] + sgs[1:])

    # --- z_i by three definitions
    zi_min = float(zm[int(np.argmin(total))])
    # zero crossing of the total flux, linear interpolation
    zi_zero = float("nan")
    for k in range(1, len(total)):
        if total[k - 1] > 0 >= total[k]:
            f = total[k - 1] / (total[k - 1] - total[k])
            zi_zero = float(zm[k - 1] + f * (zm[k] - zm[k - 1]))
            break
    # max theta gradient
    dth = np.diff(thb) / np.diff(zm)
    zi_grad = float(0.5 * (zm[int(np.argmax(dth))] + zm[int(np.argmax(dth)) + 1]))

    # --- w* and the fit, by normal equations
    band = (zm >= 0.2 * zi_min) & (zm <= 0.7 * zi_min)
    th_ml = float(thb[band].mean())
    wstar = float((G / th_ml * Qs * zi_min) ** (1.0 / 3.0))

    fb = (zm >= 0.1 * zi_min) & (zm <= 0.7 * zi_min)
    x = zm[fb] / zi_min
    y = total[fb] / Qs
    A = np.vstack([np.ones_like(x), x]).T
    coef = np.linalg.solve(A.T @ A, A.T @ y)
    a, b = float(coef[0]), float(coef[1])
    rms = float(np.sqrt(np.mean((y - (a + b * x)) ** 2)))

    # --- mass drift straight off the raw arrays across all files
    m = []
    for f in files:
        dd = Dataset(f)
        for i in range(dd.dimensions["Time"].size):
            m.append(float((arr(dd.variables["MU"][i])
                            + arr(dd.variables["MUB"][i])).sum()))
        dd.close()
    drift = max(abs(v - m[0]) for v in m) / abs(m[0])

    # --- Parseval, real-space variance vs binned spectral sum
    kmid = int(np.argmin(np.abs(zm - 0.5 * zi_min)))
    w = arr(d.variables["W"][nt - 1])
    f2 = 0.5 * (w[kmid] + w[kmid + 1])
    f2 = f2 - f2.mean()
    N = f2.shape[0]
    C = np.fft.fft2(f2) / (N * N)
    P = np.abs(C) ** 2
    var_real = float((f2 * f2).sum() / (N * N))     # separate variance route
    var_spec = float(P.sum() - P[0, 0])
    d.close()

    out = {
        "zi_flux_min_m": zi_min,
        "zi_flux_zero_crossing_m": zi_zero,
        "zi_max_theta_gradient_m": zi_grad,
        "theta_ml_K": th_ml,
        "wstar": wstar,
        "fit_intercept": a, "fit_slope": b, "fit_rms": rms,
        "entrainment_min_over_qs": float(total.min() / Qs),
        "wth_res_max_over_qs": float(res.max() / Qs),
        "mass_drift_rel": drift,
        "parseval_var_realspace": var_real,
        "parseval_var_spectral": var_spec,
        "parseval_residual_rel": abs(var_real - var_spec) / max(var_real, 1e-300),
        "n_frames_in_window": n,
    }

    print("independent recomputation vs primary receipt")
    print("%-34s %14s %14s %12s" % ("quantity", "primary", "independent", "reldiff"))
    checks = [
        ("zi_m", "zi_flux_min_m"),
        ("theta_ml", "theta_ml_K"),
        ("wstar", "wstar"),
        ("flux_fit_intercept", "fit_intercept"),
        ("flux_fit_slope", "fit_slope"),
        ("flux_fit_rms", "fit_rms"),
        ("entrainment_min_over_qs", "entrainment_min_over_qs"),
        ("wth_res_max_over_qs", "wth_res_max_over_qs"),
        ("mass_drift_rel", "mass_drift_rel"),
    ]
    worst = 0.0
    for pk, ik in checks:
        pv = ref.get(pk)
        iv = out[ik]
        if pv is None:
            continue
        rd = abs(iv - pv) / max(abs(pv), 1e-300)
        worst = max(worst, rd)
        print("%-34s %14.6g %14.6g %12.3g" % (pk, pv, iv, rd))
    print()
    print("z_i by three definitions: flux-min %.1f m, flux zero-crossing %.1f m, "
          "max dtheta/dz %.1f m" % (zi_min, zi_zero, zi_grad))
    print("Parseval: real-space var %.6g vs spectral sum %.6g, residual %.3g"
          % (var_real, var_spec, out["parseval_residual_rel"]))
    print("worst relative disagreement with the primary scorer: %.3g" % worst)
    print("NOTE: the SGS flux here uses centred differencing with an "
          "arithmetic-mean K, not WRF's fnm/fnp weights, so small differences "
          "in flux-derived quantities are expected and are the point.")
    json.dump(out, open(primary.replace(".json", "_independent.json"), "w"),
              indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
