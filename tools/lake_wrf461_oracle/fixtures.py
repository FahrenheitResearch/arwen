"""Write the two test fixtures of the correctly rounded lake intrinsics.

    python3 fixtures.py OUT BUILD DEST

``OUT`` holds a finished regime campaign (regimes.py forcing, fortran-cr)
and libm-inputs-dt60.npz (libm_trace.py --dump); ``BUILD`` holds
libm_host.so; ``DEST`` is tests/data/oracles/lake.  Needs mpmath.

f64cr_cases.npz: arguments and their correctly rounded binary64 results
(240-bit mpmath) for exp, log, log10, sin, atan and pow: every lake argument
of the dt60 campaign where the host libm is not correctly rounded (up to 64
per function), random arguments across each domain, the structured hard
cases near exp(0) and log(1), x^2 midpoints and IEEE special cases (whose
expected words are the host library's, which follows C99 Annex F).

lake_cr_regimes.npz: 16 regime columns, 120 steps at 60 s, every state word
at steps 30/60/90/120 and every output word at every step, from native WRF
linked to correctly rounded binary64 functions (lake_fortran_cr.so).
"""
from __future__ import annotations

import ctypes
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cr_check  # noqa: E402

FUNCS = cr_check.FUNCS
COLUMNS = ("gale#0", "gale#2", "tropical_deep_equator#0", "tropical_deep_equator#1", "high_plateau_open#0",
           "high_plateau_open#3", "polar_summer_open#0", "monsoon_heavy_rain#0", "desert_shallow_hot#0",
           "convective_warm_day#1", "evening_transition#0", "andes_altiplano#1", "ice_with_deep_snow#0",
           "snowfall_on_ice#0", "freeze_up#0", "stable_night_fog#0")
STEPS = 120
STATE_STEPS = (30, 60, 90, 120)


def cases(out: Path, build: Path):
    rng = np.random.default_rng(7)
    lake = np.load(out / "libm-inputs-dt60.npz")
    glibc = ctypes.CDLL(str(build / "libm_host.so"))
    rand = cr_check.random_arguments(4096, seed=11)
    result = {}
    with mp.Pool(32) as pool:
        for f in FUNCS:
            pairs = lake[f]
            xs, ys = pairs[:, 0].copy(), pairs[:, 1].copy()
            # every lake argument the host library misrounds, found in chunks
            found = []
            for i in range(0, len(xs), 200000):
                cx, cy = xs[i:i + 200000], ys[i:i + 200000]
                cr = np.concatenate(pool.map(cr_check._cr, [(f, cx[k:k + 2000], cy[k:k + 2000])
                                                            for k in range(0, len(cx), 2000)]))
                g = cr_check.host_eval(glibc, f, cx, cy, "libm_host_eval")
                bad = np.flatnonzero(~np.isnan(cr) & (g.view(np.int64) != cr.view(np.int64)))
                found += [(cx[j], cy[j]) for j in bad]
                if len(found) >= 64:
                    break
            found = found[:64]
            r = rand[f]
            rx, ry = (r if isinstance(r, tuple) else (r, np.zeros_like(r)))
            pick = rng.choice(len(rx), 512, replace=False)
            gx = [a for a, _ in found] + list(np.ascontiguousarray(rx[pick]).view(np.uint64))
            gy = [b for _, b in found] + list(np.ascontiguousarray(ry[pick]).view(np.uint64))
            if f == "exp":
                hard = [k * 2.0 ** -s for k in (1, -1, 3, -3, 5, 7, -9) for s in (53, 54, 55, 60)]
                gx += list(np.array(hard).view(np.uint64)); gy += [0] * len(hard)
            if f == "log":
                hard = [1 + k * 2.0 ** -52 for k in (1, 2, 3, 5, 7, 104, -1, -2, -3)]
                gx += list(np.array(hard).view(np.uint64)); gy += [0] * len(hard)
            if f == "pow":
                mids = [float(m) for m in (94906267, 100000001, 123456789, 134217727)]
                gx += list(np.array(mids).view(np.uint64)); gy += list(np.full(len(mids), 2.0).view(np.uint64))
            X, Y = np.array(gx, np.uint64), np.array(gy, np.uint64)
            cr = np.concatenate(pool.map(cr_check._cr, [(f, X[k:k + 500], Y[k:k + 500]) for k in range(0, len(X), 500)]))
            if f == "pow":
                sq = Y.view(np.float64) == 2.0
                cr[sq] = X[sq].view(np.float64) ** 2  # one IEEE product is the correctly rounded square
            keep = ~np.isnan(cr)
            result[f + "_x"], result[f + "_y"], result[f + "_expected"] = X[keep], Y[keep], cr[keep].view(np.uint64)
            result[f + "_host_misrounded"] = np.int64(len(found))
            # IEEE special cases: expected word is the host library's.
            sx = np.array(cr_check.SPECIALS)
            if f == "pow":
                a, b = np.meshgrid(sx, sx)
                sx, sy = a.ravel(), b.ravel()
            else:
                sy = np.zeros_like(sx)
            g = cr_check.host_eval(glibc, f, sx.view(np.uint64), sy.view(np.uint64), "libm_host_eval")
            result[f + "_special_x"], result[f + "_special_y"] = sx.view(np.uint64), sy.view(np.uint64)
            result[f + "_special_expected"] = g.view(np.uint64)
            print(f, int(keep.sum()), "cases,", len(found), "host-misrounded lake arguments", flush=True)
    return result


def regimes(out: Path):
    names = json.loads((out / "columns.json").read_text())["names"]
    sel = np.array([names.index(c) for c in COLUMNS])
    inputs = np.load(out / "inputs-dt60.npz")
    ref = np.load(out / "fortran_cr-dt60.npz")
    return dict(columns=np.array(COLUMNS), dt=inputs["dt"],
                seed=np.ascontiguousarray(inputs["seed"][:, sel]),
                forcing=np.ascontiguousarray(inputs["forcing"][:STEPS][:, :, sel]),
                init_state=np.ascontiguousarray(ref["init_state"][:, sel]),
                init_static=np.ascontiguousarray(ref["init_static"][:, sel]),
                state_steps=np.array(STATE_STEPS),
                state=np.ascontiguousarray(ref["trace_state"][np.array(STATE_STEPS) - 1][:, :, sel]),
                output=np.ascontiguousarray(ref["trace_output"][:STEPS][:, :, sel]))


if __name__ == "__main__":
    out, build, dest = map(Path, sys.argv[1:4])
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "lake_cr_regimes.npz", **regimes(out))
    np.savez_compressed(dest / "f64cr_cases.npz", **cases(out, build))
    for p in sorted(dest.glob("*.npz")):
        print(p, p.stat().st_size)
