"""Read the probe's JSON and answer the closure question with numbers.

Two things are asked of a spectrum here, and only the second one is a
judgement about the closure:

1. Is there an inertial range at all?  Fit a slope to the middle of the
   retained band and compare with -5/3.
2. Is energy PILING UP at the grid scale?  The signature of an
   under-dissipative closure is a spectrum that flattens or turns up
   toward the Nyquist wavenumber instead of rolling off.  Measured as the
   compensated spectrum k^(5/3) P(k) in the top octave relative to the
   inertial range: a well-behaved LES falls below 1, a piling-up one rises
   above it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def slope(k, p, lo_frac, hi_frac):
    """Least-squares log-log slope over a fractional band of the range."""
    k = np.asarray(k, float)
    p = np.asarray(p, float)
    good = (k > 0) & (p > 0)
    k, p = k[good], p[good]
    if k.size < 6:
        return None, 0
    kmin, kmax = k.min(), k.max()
    band = (k >= kmin * (kmax / kmin) ** lo_frac) & \
           (k <= kmin * (kmax / kmin) ** hi_frac)
    if band.sum() < 4:
        return None, int(band.sum())
    a = np.polyfit(np.log10(k[band]), np.log10(p[band]), 1)
    return float(a[0]), int(band.sum())


def pileup(k, p):
    """Compensated spectrum in the top octave vs the inertial range."""
    k = np.asarray(k, float)
    p = np.asarray(p, float)
    good = (k > 0) & (p > 0)
    k, p = k[good], p[good]
    comp = p * k ** (5.0 / 3.0)
    kmax = k.max()
    top = k >= kmax / 2.0
    mid = (k >= kmax / 16.0) & (k < kmax / 4.0)
    if top.sum() < 2 or mid.sum() < 2:
        return None
    return float(np.median(comp[top]) / np.median(comp[mid]))


def report(path: Path):
    d = json.loads(path.read_text())
    key = "endurance" if "endurance" in d else None
    blocks = []
    if key:
        blocks = [(path.name, d[key])]
    elif "closure" in d:
        blocks = [(f"{path.name} dx={r['dx']} km{r['km_opt']}", r)
                  for r in d["closure"]]
    for name, b in blocks:
        k = b.get("spectrum_k")
        p = b.get("spectrum_power")
        cfg = b.get("config", b)
        dx = cfg.get("dx")
        km = cfg.get("km_opt")
        print(f"\n=== {name}  dx={dx} km_opt={km} ===")
        if "samples" in b and b["samples"]:
            s = b["samples"][-1]
            print(f"  final: step={s['step']} t={s['t_sim_s']:.0f}s "
                  f"w_max={s['w_max']:.3f} e_res={s['e_res_ml']:.4g} "
                  f"e_sgs={s['e_sgs_ml']:.4g} sgs_frac={s['sgs_fraction']:.4f}")
            print(f"  basis: {s['e_sgs_basis']}")
            print(f"  mass_drift_rel={s['mass_drift_rel']:.3e} "
                  f"cfl_sound={s['cfl_sound_horiz']:.4f} nan={s['nan']}")
        else:
            print(f"  e_res={b.get('e_res_ml'):.4g} "
                  f"e_sgs={b.get('e_sgs_ml'):.4g} "
                  f"sgs_frac={b.get('sgs_fraction'):.4f}")
        er = b.get("e_res_profile")
        es = b.get("e_sgs_profile")
        z = b.get("z_mass")
        if er and es and z:
            er = np.asarray(er, float)
            es = np.asarray(es, float)
            z = np.asarray(z, float)
            frac = es / np.maximum(er + es, 1e-30)
            print("  SGS fraction by height (the tornado's corner flow "
                  "lives in the bottom rows):")
            for zz in (2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0):
                i = int(np.argmin(np.abs(z - zz)))
                if abs(z[i] - zz) < max(3.0, 0.25 * zz):
                    print(f"    z={z[i]:6.1f} m  e_res={er[i]:9.4g}  "
                          f"e_sgs={es[i]:9.4g}  sgs={frac[i]:6.3f}")
        if k and p:
            sl, n = slope(k, p, 0.15, 0.55)
            sl_hi, nh = slope(k, p, 0.55, 1.0)
            pu = pileup(k, p)
            print(f"  spectrum at z={b.get('spectrum_height_m'):.1f} m, "
                  f"{len(k)} retained bins, k in "
                  f"[{min(k):.4g}, {max(k):.4g}] cyc/m")
            print(f"  slope (low-mid band, {n} bins) = {sl:.3f}   "
                  f"(-5/3 = -1.667)")
            print(f"  slope (high band,   {nh} bins) = {sl_hi:.3f}")
            print(f"  grid-scale pile-up ratio       = {pu:.4f}  "
                  f"({'PILING UP' if pu and pu > 1.0 else 'rolling off'})")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(Path(arg))
