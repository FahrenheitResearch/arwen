"""Correct rounding census of every distinct binary64 library argument the
lake used in a regime campaign (libm_trace.py --dump).

For each argument this computes the correctly rounded binary64 result from a
200-bit mpmath evaluation and compares it with (a) the host libm the native
WRF oracle links (libm_host.so) and (b), when present, WOOF's own
correctly rounded functions compiled for the host (lake_crlibm_host.so).

    python3 cr_check.py OUT BUILD --campaign dt60 [--jobs 48]

Needs mpmath (the system python3 on the verification box has it).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

FUNCS = ("exp", "log", "log10", "sin", "atan", "pow")


def _cr(args):
    fname, xs, ys = args
    import mpmath
    from mpmath.libmp import round_nearest
    mpmath.mp.prec = 240
    f = {"exp": mpmath.exp, "log": mpmath.log, "log10": mpmath.log10, "sin": mpmath.sin,
         "atan": mpmath.atan}.get(fname)
    out = np.empty(len(xs), np.float64)
    two1074 = mpmath.mpf(2) ** 1074
    for i, (xb, yb) in enumerate(zip(xs, ys)):
        x = float(np.uint64(xb).view(np.float64))
        y = float(np.uint64(yb).view(np.float64))
        if fname == "pow":
            if x <= 0 or not np.isfinite(x) or not np.isfinite(y):
                out[i] = np.nan
                continue
            v = mpmath.power(mpmath.mpf(x), mpmath.mpf(y))
        else:
            if (fname in ("log", "log10") and x <= 0) or not np.isfinite(x):
                out[i] = np.nan
                continue
            v = f(mpmath.mpf(x))
        if v == 0:
            out[i] = 0.0
        elif abs(v) < mpmath.mpf(2) ** -1022:
            k = mpmath.nint(v * two1074)  # nint is ties-to-even
            out[i] = float(int(k)) * 2.0 ** -1074
        else:
            out[i] = mpmath.libmp.to_float(v._mpf_, rnd=round_nearest)
    return out


def host_eval(lib, fname, xs, ys, symbol):
    D = ctypes.POINTER(ctypes.c_double)
    fn = np.full(len(xs), FUNCS.index(fname), np.int32)
    x = xs.view(np.float64).copy()
    y = ys.view(np.float64).copy()
    out = np.empty(len(xs), np.float64)
    getattr(lib, symbol)(len(xs), fn.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), x.ctypes.data_as(D),
                         y.ctypes.data_as(D), out.ctypes.data_as(D))
    return out


def random_arguments(n, seed=20261007):
    """Arguments well beyond the lake's: each function across its whole
    finite domain (bit patterns, so every binade and subnormals), plus the
    ranges where reduction or cancellation is hardest."""
    rng = np.random.default_rng(seed)

    def bits(lo, hi, k):  # uniform over the binary64 words of [lo, hi], lo, hi > 0
        a, b = np.float64(lo).view(np.int64), np.float64(hi).view(np.int64)
        return rng.integers(a, b, k, endpoint=True).view(np.float64)

    q = n // 4
    out = {}
    out["exp"] = np.concatenate([rng.uniform(-745.2, 709.78, q), bits(1e-30, 745.0, q) * rng.choice([-1, 1], q),
                                 np.where(rng.random(q) < 0.5, rng.uniform(-1, 1, q) * 0.0055,
                                          rng.integers(-4096, 4096, q) * 2.0 ** -rng.integers(40, 56, q)),
                                 rng.uniform(-745.2, -708.0, n - 3 * q)])
    out["log"] = np.concatenate([bits(5e-324, 1.7e308, q), rng.uniform(0.99, 1.01, q),
                                 1.0 + rng.uniform(-1, 1, q) * 2.0 ** rng.integers(-52, -10, q), bits(5e-324, 2.2e-308, n - 3 * q)])
    out["log10"] = np.concatenate([bits(5e-324, 1.7e308, n - q), rng.uniform(0.5, 2.0, q)])
    out["sin"] = np.concatenate([rng.uniform(-np.pi, np.pi, q), bits(1e-300, 1e6, q) * rng.choice([-1, 1], q),
                                 rng.uniform(-1.6, 1.6, q), np.pi / 2 * rng.integers(-600000, 600000, n - 3 * q)
                                 + rng.uniform(-1e-6, 1e-6, n - 3 * q)])
    out["atan"] = np.concatenate([bits(5e-324, 1.7e308, q) * rng.choice([-1, 1], q), rng.uniform(-20, 20, q),
                                  rng.uniform(0.9, 1.1, q), bits(1e-20, 1e20, n - 3 * q)])
    px = np.concatenate([bits(1e-30, 1e30, q), rng.uniform(0.0, 1.0, q), rng.uniform(0.9, 1.1, q), bits(1e-10, 1e10, n - 3 * q)])
    py = np.concatenate([rng.uniform(-20, 20, q), rng.uniform(-3, 3, q), rng.uniform(-500, 500, q),
                         np.round(rng.uniform(-40, 40, n - 3 * q), 3)])
    ok = np.abs(py * np.log(np.maximum(px, 1e-300))) < 700
    out["pow"] = (px[ok], py[ok])
    return out


SPECIALS = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 1e-310, -1e-310, 5e-324, 1e308, -1e308,
            float("inf"), float("-inf"), float("nan"), 709.782712893384, 709.79, -745.13321910194122,
            -745.2, -708.4, 1.5707963267948966, -1.5707963267948966, 1e-17, 2.0 ** -54, 2.0 ** -1022]


def specials(glibc, woof):
    """IEEE special cases: WOOF must return glibc's word (NaN for NaN)."""
    xs = np.array(SPECIALS)
    rows = []
    for fname in FUNCS:
        if fname == "pow":
            xx, yy = np.meshgrid(xs, xs)
            x, y = xx.ravel(), yy.ravel()
        else:
            x, y = xs, np.zeros_like(xs)
        g = host_eval(glibc, fname, x.view(np.uint64), y.view(np.uint64), "libm_host_eval")
        w = host_eval(woof, fname, x.view(np.uint64), y.view(np.uint64), "lake_crlibm_host_eval")
        same = (g.view(np.int64) == w.view(np.int64)) | (np.isnan(g) & np.isnan(w))
        for j in np.flatnonzero(~same):
            rows.append(dict(f=fname, x=float(x[j]).hex(), y=float(y[j]).hex(), host=float(g[j]).hex(), woof=float(w[j]).hex()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("build", type=Path)
    p.add_argument("--campaign", default="dt60")
    p.add_argument("--jobs", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--random", type=int, default=0, help="check N random arguments per function instead")
    a = p.parse_args()
    if a.random:
        data = {}
        for f, v in random_arguments(a.random).items():
            x, y = v if isinstance(v, tuple) else (v, np.zeros_like(v))
            data[f] = np.stack([np.ascontiguousarray(x).view(np.uint64), np.ascontiguousarray(y).view(np.uint64)], 1)
    else:
        data = np.load(a.out / f"libm-inputs-{a.campaign}.npz")
    glibc = ctypes.CDLL(str(a.build / "libm_host.so"))
    woof_path = a.build / "lake_crlibm_host.so"
    woof = ctypes.CDLL(str(woof_path)) if woof_path.exists() else None
    report = {}
    with mp.Pool(a.jobs) as pool:
        for fname in FUNCS:
            pairs = data[fname]
            if a.limit:
                pairs = pairs[: a.limit]
            xs, ys = np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1])
            chunks = [(fname, xs[i:i + 2000], ys[i:i + 2000]) for i in range(0, len(xs), 2000)]
            cr = np.concatenate(pool.map(_cr, chunks)) if chunks else np.zeros(0)
            valid = ~np.isnan(cr)
            g = host_eval(glibc, fname, xs, ys, "libm_host_eval")
            entry = dict(arguments=int(len(xs)), checked=int(valid.sum()),
                         host_libm_not_correctly_rounded=int((g.view(np.int64) != cr.view(np.int64))[valid].sum()))
            bad = np.flatnonzero(valid & (g.view(np.int64) != cr.view(np.int64)))
            entry["host_examples"] = [dict(x=float(xs[j:j+1].view(np.float64)[0]).hex(),
                                           y=float(ys[j:j+1].view(np.float64)[0]).hex(),
                                           host=float(g[j]).hex(), correct=float(cr[j]).hex()) for j in bad[:5]]
            if woof is not None:
                woof.lake_crlibm_ambiguous.restype = ctypes.c_ulonglong
                before = woof.lake_crlibm_ambiguous()
                w = host_eval(woof, fname, xs, ys, "lake_crlibm_host_eval")
                entry["woof_rounding_test_ambiguous"] = int(woof.lake_crlibm_ambiguous() - before)
                entry["woof_not_correctly_rounded"] = int((w.view(np.int64) != cr.view(np.int64))[valid].sum())
                entry["woof_vs_host_libm_differing"] = int((w.view(np.int64) != g.view(np.int64)).sum())
                wb = np.flatnonzero(valid & (w.view(np.int64) != cr.view(np.int64)))
                entry["woof_examples"] = [dict(x=float(xs[j:j+1].view(np.float64)[0]).hex(),
                                               y=float(ys[j:j+1].view(np.float64)[0]).hex(),
                                               woof=float(w[j]).hex(), correct=float(cr[j]).hex()) for j in wb[:5]]
            report[fname] = entry
            print(fname, {k: v for k, v in entry.items() if not k.endswith("examples")}, flush=True)
    if woof is not None:
        report["ieee_special_cases_differing_from_host"] = specials(glibc, woof)
        print("special cases differing:", report["ieee_special_cases_differing_from_host"], flush=True)
    tag = "-limit" if a.limit else ""
    name = f"cr-random-{a.random}.json" if a.random else f"cr-census-{a.campaign}{tag}.json"
    (a.out / name).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
