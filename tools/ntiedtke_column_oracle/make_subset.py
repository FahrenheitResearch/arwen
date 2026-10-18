"""Cut a column subset out of a case, as a new case of shape nx = N, ny = 1.

Selection is by input regime, deterministic (fixed seed), so the same
subset comes back every run:

  mix     a stratified sample: ocean, high terrain (surface below 750 hPa),
          cold surface (t < 273.15 K at the lowest level), ice aloft,
          stable surface layer (HFX < 0), moist warm low levels (the likely
          convective columns), dry low levels, and a uniform remainder.
  cols    the flat column indices given with --cols, in that order.
  quiet   the first N columns of CASE in which the WRF run named by --wrf
          did nothing at all: a group with no convection anywhere, which is
          where cuascn's chunk-wide llo3 stays false.
  stable  land columns with HFX < 0 and qv(1) < 8 g/kg only: a population
          in which no column should trigger, which is the llo3-false case.

    python make_subset.py CASE OUT --kind mix|stable [--n N] [--seed S]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from make_columns import IFACE, LEV, SFC


def read_meta(case):
    meta = {}
    for line in open(pathlib.Path(case) / "meta.txt"):
        k, _, val = line.strip().partition(" ")
        meta[k] = val
    return meta


def load(case):
    case = pathlib.Path(case)
    m = read_meta(case)
    nx, ny, nz = int(m["nx"]), int(m["ny"]), int(m["nz"])
    lev = {n: np.fromfile(case / f"{n}.f32", "<f4").reshape(ny, nz, nx)
           for n in LEV}
    iface = {n: np.fromfile(case / f"{n}.f32", "<f4").reshape(ny, nz + 1, nx)
             for n in IFACE}
    sfc = {n: np.fromfile(case / f"{n}.f32", "<f4").reshape(ny, nx)
           for n in SFC}
    return m, lev, iface, sfc


def classes(lev, iface, sfc):
    """Per-column regime flags, flattened to (ny*nx,)."""
    t1 = lev["t3d"][:, 0, :].ravel()
    q1 = lev["qv3d"][:, 0, :].ravel()
    psfc = iface["p8w"][:, 0, :].ravel()
    qi_max = lev["qi3d"].max(axis=1).ravel()
    hfx = sfc["hfx"].ravel()
    ocean = sfc["xland"].ravel() > 1.5
    return {
        "ocean": ocean,
        "high_terrain": psfc < 75000.0,
        "cold_surface": t1 < 273.15,
        "ice_aloft": qi_max > 1.0e-6,
        "stable_surface": hfx < 0.0,
        "moist_warm": (q1 > 0.012) & (t1 > 290.0),
        "dry": q1 < 0.002,
    }


def pick(flags, n, seed):
    rng = np.random.default_rng(seed)
    ncol = next(iter(flags.values())).size
    chosen = []
    per = n // (len(flags) + 1)
    for name, mask in flags.items():
        idx = np.flatnonzero(mask)
        if idx.size:
            chosen.extend(rng.choice(idx, min(per, idx.size), replace=False))
    rest = np.setdiff1d(np.arange(ncol), np.asarray(chosen, dtype=np.int64))
    chosen.extend(rng.choice(rest, n - len(set(chosen)), replace=False))
    return np.unique(np.asarray(chosen, dtype=np.int64))[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--kind", choices=("mix", "stable", "cols", "quiet"),
                    required=True)
    ap.add_argument("--wrf", default="",
                    help="--kind quiet: a WRF run of CASE to read activity from")
    ap.add_argument("--cols", default="",
                    help="--kind cols: comma-separated flat column indices")
    ap.add_argument("--dt", type=float, default=None,
                    help="override the case's model time step")
    ap.add_argument("--n", type=int, default=4099)
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args()

    m, lev, iface, sfc = load(a.case)
    flags = classes(lev, iface, sfc)
    if a.kind == "mix":
        sel = pick(flags, a.n, a.seed)
    elif a.kind == "cols":
        sel = np.asarray([int(x) for x in a.cols.split(",")], dtype=np.int64)
    elif a.kind == "quiet":
        ny0, nx0 = sfc["xland"].shape
        nz0 = int(m["nz"])
        wrf = pathlib.Path(a.wrf)
        act = np.fromfile(wrf / "raincv.f32", "<f4") > 0
        for f in ("rthcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten"):
            act |= (np.fromfile(wrf / f"{f}.f32", "<f4").reshape(
                ny0, nz0, nx0) != 0).any(axis=1).ravel()
        sel = np.flatnonzero(~act)[:a.n]
    else:
        mask = (~flags["ocean"]) & flags["stable_surface"] & (
            lev["qv3d"][:, 0, :].ravel() < 0.008)
        idx = np.flatnonzero(mask)
        rng = np.random.default_rng(a.seed)
        sel = np.sort(rng.choice(idx, min(a.n, idx.size), replace=False))
    ny, nx = sfc["xland"].shape
    jj, ii = np.unravel_index(sel, (ny, nx))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in LEV:
        np.ascontiguousarray(lev[n][jj, :, ii].T[None]).astype("<f4").tofile(
            out / f"{n}.f32")
    for n in IFACE:
        np.ascontiguousarray(iface[n][jj, :, ii].T[None]).astype(
            "<f4").tofile(out / f"{n}.f32")
    for n in SFC:
        np.ascontiguousarray(sfc[n][jj, ii][None]).astype("<f4").tofile(
            out / f"{n}.f32")
    np.savetxt(out / "selection.txt", np.stack([jj, ii], 1), fmt="%d")
    with open(out / "meta.txt", "w", newline="\n") as f:
        for k, val in m.items():
            if k == "dt" and a.dt is not None:
                val = a.dt
            if k not in ("nx", "ny"):
                f.write(f"{k} {val}\n")
        f.write(f"nx {sel.size}\nny 1\nsubset_of {pathlib.Path(a.case).name}"
                f"\nsubset_kind {a.kind}\n")
    print(f"subset {a.out}: {sel.size} columns ({a.kind}) of {a.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
