"""Time the LETKF neighbour search at the recent case's shape.

A synthetic stand-in with the recent case's dimensions and observation
roster (241 x 241 x 49 at 3 km, 32 members, 14 analysed fields, merged
reflectivity and clear-air batches, eleven windowed radial-velocity
batches, three surface batches), used when the real saved analysis inputs
are not at hand.  It runs :func:`gpuwm.da.letkf.analyze` on the bounded
host-staged route and prints one JSON line of timings and the increments'
digest so two neighbour-search modes can be compared byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np


def build(seed=20261005, nz=49, ny=241, nx=241, members=32, fields=14,
          radars=11):
    from gpuwm.da.letkf import GridGeometry, GriddedObs, Localization
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lat = 27.75 + jj * 0.02697 + ii * 1e-4
    lon = -92.7 + ii * 0.0309 - jj * 2e-4
    terrain = 60.0 + 250.0 * np.exp(-(((jj - 170) / 40.0) ** 2
                                      + ((ii - 60) / 50.0) ** 2))
    eta = np.linspace(0.0, 1.0, nz) ** 1.6
    heights = terrain[None] + 20.0 + eta[:, None, None] * 19500.0
    grid = GridGeometry(dx_m=3000.0, dy_m=3000.0, heights_m=heights,
                        lat_deg=lat, lon_deg=lon)
    prior = {}
    base = rng.standard_normal((members, nz, ny, nx), dtype=np.float64)
    for n in range(fields):
        prior[f"f{n:02d}"] = base * (0.5 + 0.05 * n) + n
    sim0 = prior["f00"]
    # Echo: storms in a band; clear air elsewhere inside radar reach.
    storm = ((np.sin(jj / 17.0) + np.cos(ii / 23.0)) > 1.1)
    echo = np.zeros(shape, bool)
    echo[:30] = storm[None]
    reach = ((jj - 120) ** 2 + (ii - 120) ** 2) < 115 ** 2
    clear = np.zeros(shape, bool)
    clear[:25] = (reach & ~storm)[None]
    batches = []

    def thin(mask, cells):
        out = np.zeros_like(mask)
        out[:, ::cells, ::cells] = mask[:, ::cells, ::cells]
        return out

    loc = Localization(12000.0, 6000.0)
    for name, mask, cells, err in (("z", echo, 2, 5.0), ("z0", clear, 4, 7.5)):
        m = thin(mask, cells)
        values = np.where(m, sim0.mean(0) + rng.normal(size=shape), np.nan)
        batches.append(GriddedObs(name, values, np.full(shape, err), sim0 * 1.0,
                                  m, localization=loc))
    for r in range(radars):
        margin, half = min(30, ny // 4), min(80, ny // 3)
        cj, ci = rng.integers(margin, ny - margin), rng.integers(margin, nx - margin)
        j0, j1 = max(0, cj - half), min(ny - 1, cj + half)
        i0, i1 = max(0, ci - half), min(nx - 1, ci + half)
        wshape = (nz, j1 - j0 + 1, i1 - i0 + 1)
        m = thin(echo[:, j0:j1 + 1, i0:i1 + 1], 4)
        sim = sim0[:, :, j0:j1 + 1, i0:i1 + 1] * (0.3 + 0.05 * r)
        values = np.where(m, sim.mean(0) + rng.normal(size=wshape), np.nan)
        batches.append(GriddedObs(f"vr:K{r:03d}", values, np.full(wshape, 2.0),
                                  sim, m, window=(j0, j1, i0, i1),
                                  localization=loc))
    sloc = Localization(12000.0, 3000.0)
    stations = rng.choice(ny * nx, size=220, replace=False)
    for name in ("t2", "wspd", "q2"):
        m = np.zeros(shape, bool)
        m[0].reshape(-1)[stations] = True
        sim = np.zeros((members,) + shape)
        sim[:, 0] = prior["f01"][:, 0]
        values = np.where(m, sim.mean(0) + rng.normal(size=shape), np.nan)
        batches.append(GriddedObs(name, values, np.full(shape, 1.5), sim, m,
                                  localization=sloc))
    return prior, batches, grid


def digest(increments):
    h = hashlib.sha256()
    for name in sorted(increments):
        h.update(name.encode())
        h.update(np.ascontiguousarray(increments[name]).tobytes())
    return h.hexdigest()


def main(argv=None):
    from dataclasses import replace
    from gpuwm.da.letkf import LetkfConfig, LetkfDiagnostics, Localization, analyze
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("index", "forward"), default="index")
    parser.add_argument("--namespace", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument("--storage", choices=("staged", "resident"),
                        default="staged",
                        help="host-staged prior (the 32 GB card route) or "
                             "the prior and observations resident on the card")
    parser.add_argument("--budget-mib", type=float, default=14336)
    parser.add_argument("--ny", type=int, default=241)
    parser.add_argument("--nz", type=int, default=49)
    parser.add_argument("--save", default=None, help="npz of increments")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    prior, batches, grid = build(ny=args.ny, nx=args.ny, nz=args.nz)
    t_build = time.perf_counter() - t0
    cfg = LetkfConfig(Localization(12000.0, 6000.0), tuple(prior), 0.9,
                      memory_budget_mib=args.budget_mib,
                      neighbor_search=args.mode)
    if args.namespace == "cupy":
        import cupy as xp
    else:
        xp = np
    diag = LetkfDiagnostics()
    t1 = time.perf_counter()
    if args.storage == "resident":
        prior = {k: xp.asarray(v) for k, v in prior.items()}
        batches = [replace(o, values=xp.asarray(o.values),
                           errors=xp.asarray(o.errors),
                           simulated=xp.asarray(o.simulated),
                           mask=xp.asarray(o.mask)) for o in batches]
        t_stage = time.perf_counter() - t1
        t1 = time.perf_counter()
        inc = analyze(prior, batches, grid, cfg, diag)
        inc = {k: xp.asnumpy(v) for k, v in inc.items()}
    else:
        t_stage = 0.0
        inc = analyze(prior, batches, grid, cfg, diag, solve_namespace=xp)
    wall = time.perf_counter() - t1
    if args.save:
        np.savez(args.save, **{k: v.astype(np.float64) for k, v in inc.items()})
    print(json.dumps({
        "mode": args.mode, "namespace": args.namespace,
        "storage": args.storage, "shape": [args.nz, args.ny, args.ny],
        "stage_seconds": round(t_stage, 2),
        "synthetic_build_seconds": round(t_build, 2),
        "analyze_seconds": round(wall, 2),
        "setup_seconds": round(diag.setup_seconds, 2),
        "weights_seconds": round(diag.weights_seconds, 2),
        "transform_seconds": round(diag.transform_seconds, 2),
        "finish_seconds": round(diag.finish_seconds, 2),
        "active_points": diag.active_points,
        "max_local_obs": diag.max_local_obs,
        "device_chunks": diag.device_chunks,
        "chunk_points": diag.chunk_points,
        "neighbor_index": diag.neighbor_index,
        "digest": digest(inc)}), flush=True)


if __name__ == "__main__":
    main()
