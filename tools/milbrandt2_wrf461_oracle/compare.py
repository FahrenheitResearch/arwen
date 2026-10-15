"""Bitwise comparison of two Milbrandt-Yau oracle output streams.

    python compare.py A.bin B.bin --input in.bin [--columns in.columns.txt]

Prints, per field, the number of float32 words that differ in their bits,
the maximum ULP distance (total-order, so -0.0 vs +0.0 is 1 ULP and counts),
and the maximum absolute difference, over every step, column and level.
Exit status 0 only when every word of every field is bit-identical.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LEVEL_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                "nc", "nr", "ni", "ns", "ng", "nh", "th", "zet")
SURFACE_FIELDS = ("rainnc", "rainncv", "snownc", "snowncv", "grplnc",
                  "grplncv", "hailnc", "hailncv", "sr")


def read_output(path, ncol, nz, nsteps):
    raw = np.fromfile(path, dtype="<f4")
    per_step = len(LEVEL_FIELDS) * ncol * nz + len(SURFACE_FIELDS) * ncol
    if raw.size != per_step * nsteps:
        raise SystemExit(f"{path}: {raw.size} words, expected "
                         f"{per_step * nsteps}")
    out = {k: np.empty((nsteps, ncol, nz), np.float32) for k in LEVEL_FIELDS}
    out.update({k: np.empty((nsteps, ncol), np.float32)
                for k in SURFACE_FIELDS})
    off = 0
    for s in range(nsteps):
        for k in LEVEL_FIELDS:
            n = ncol * nz
            out[k][s] = raw[off:off + n].reshape((ncol, nz), order="F")
            off += n
        for k in SURFACE_FIELDS:
            out[k][s] = raw[off:off + ncol]
            off += ncol
    return out


def write_output(path, out):
    nsteps, ncol, nz = out["qv"].shape
    with open(path, "wb") as fh:
        for s in range(nsteps):
            for k in LEVEL_FIELDS:
                np.asarray(out[k][s], "<f4").ravel(order="F").tofile(fh)
            for k in SURFACE_FIELDS:
                np.asarray(out[k][s], "<f4").tofile(fh)


def ordered(a):
    """float32 bits mapped onto a monotone signed integer line."""
    i = a.view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp(a, b):
    return np.abs(ordered(a) - ordered(b))


def compare(a, b):
    rows = []
    for k in LEVEL_FIELDS + SURFACE_FIELDS:
        x, y = a[k], b[k]
        bits = x.view(np.uint32) != y.view(np.uint32)
        u = ulp(x, y)
        with np.errstate(invalid="ignore"):
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        rows.append((k, int(bits.sum()), int(x.size),
                     int(u.max()) if u.size else 0,
                     float(np.nanmax(d)) if d.size else 0.0, bits))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--input", required=True)
    ap.add_argument("--columns")
    ap.add_argument("--first", type=int, default=5,
                    help="show the first N differing (step, column, level)")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_columns import read_stream
    ncol, nz, nsteps, dt, _, _ = read_stream(args.input)
    names = None
    if args.columns:
        names = [ln.split()[2] for ln in Path(args.columns).read_text().splitlines()]
    a = read_output(args.a, ncol, nz, nsteps)
    b = read_output(args.b, ncol, nz, nsteps)
    rows = compare(a, b)
    total = 0
    print(f"{ncol} columns x {nz} levels x {nsteps} steps, dt={dt}")
    print(f"{'field':8s} {'differ':>8s} {'of':>8s} {'maxULP':>12s} {'maxabs':>12s}")
    for k, n, size, mu, md, bits in rows:
        total += n
        print(f"{k:8s} {n:8d} {size:8d} {mu:12d} {md:12.4e}")
    print(f"TOTAL differing words: {total}")
    if total and args.first:
        for k, n, size, mu, md, bits in rows:
            if not n:
                continue
            idx = np.argwhere(bits)[: args.first]
            for ix in idx:
                where = tuple(int(v) for v in ix)
                col = names[where[1]] if names else where[1]
                print(f"  {k} step={where[0] + 1} col={col} "
                      f"at={where[1:]} a={a[k][where]!r} b={b[k][where]!r}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
