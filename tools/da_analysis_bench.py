"""A synthetic LETKF analysis at a real case's shape, for timing only.

Builds a float64 prior ``(R, nz, ny, nx)`` for each analysed field, one
windowed radial-velocity batch per radar, one whole-grid reflectivity and
clear-air batch, and lowest-level surface batches, then runs the SAME
execution path the cycle uses (``radar_assimilation._execute_analysis``
around ``letkf.analyze``) on the requested device and storage route.
Prints the filter diagnostics as JSON.  Measurement tooling only.

    python -m tools.da_analysis_bench --members 32 --shape 49 241 241 \
        --device cuda --route host-staged-cuda --out bench.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

FIELDS = ("thp", "qv", "u", "v", "qs", "qg", "qc", "nc", "qr", "nr", "qi",
          "ni", "nwfa", "nifa")


def build_case(*, members, shape, radars, seed=7, dx=3000.0,
               vr_fraction=0.03, z_fraction=0.03, z0_fraction=0.04,
               surface_points=400, radar_range_m=230e3,
               horizontal_m=12000.0, vertical_m=6000.0,
               sfc_horizontal_m=12000.0, sfc_vertical_m=3000.0,
               fields=FIELDS, geodesic=True, centre=(31.0, -89.0)):
    from gpuwm.da.letkf import GridGeometry, GriddedObs, Localization

    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    # Terrain-following heights: a stretched column plus a gentle ridge.
    eta = np.linspace(0.0, 1.0, nz)
    column = 20.0 + 19000.0 * eta ** 1.5
    jj, ii = np.mgrid[0:ny, 0:nx]
    terrain = 300.0 * np.exp(-(((jj - ny / 3) / (ny / 6)) ** 2
                               + ((ii - nx / 2) / (nx / 5)) ** 2))
    heights = column[:, None, None] + terrain[None] * (1.0 - eta)[:, None, None]
    lat = lon = None
    if geodesic:
        # A plain lat/lon mesh at the case's spacing and centre, so the
        # localisation takes the geodesic branch a projected domain takes.
        lat = centre[0] + (jj - (ny - 1) / 2) * dx / 111195.0
        lon = centre[1] + (ii - (nx - 1) / 2) * dx / (
            111195.0 * np.cos(np.radians(lat)))
    grid = GridGeometry(dx_m=dx, dy_m=dx, heights_m=heights,
                        lat_deg=lat, lon_deg=lon)

    prior = {}
    for index, name in enumerate(fields):
        base = rng.standard_normal((1, nz, ny, nx)) * (1 + index)
        prior[name] = base + 0.3 * rng.standard_normal((members, nz, ny, nx))
    probe = prior["u"]

    loc = Localization(horizontal_m=horizontal_m, vertical_m=vertical_m)
    batches = []
    radius = int(radar_range_m / dx)
    centres = [(int(ny * (0.15 + 0.7 * rng.random())),
                int(nx * (0.15 + 0.7 * rng.random()))) for _ in range(radars)]
    for number, (cj, ci) in enumerate(centres):
        j0, j1 = max(0, cj - radius), min(ny - 1, cj + radius)
        i0, i1 = max(0, ci - radius), min(nx - 1, ci + radius)
        wj, wi = np.mgrid[j0:j1 + 1, i0:i1 + 1]
        disc = (wj - cj) ** 2 + (wi - ci) ** 2 <= radius ** 2
        mask = (rng.random((nz, j1 - j0 + 1, i1 - i0 + 1)) < vr_fraction) \
            & disc[None]
        sim = probe[:, :, j0:j1 + 1, i0:i1 + 1] * 2.0
        values = sim.mean(axis=0) + rng.standard_normal(mask.shape)
        batches.append(GriddedObs(
            name=f"vr_{number}", values=values, errors=np.full(mask.shape, 2.0),
            simulated=sim, mask=mask, localization=loc,
            window=(j0, j1, i0, i1)))
    for name, fraction in (("reflectivity", z_fraction),
                           ("clear_air", z0_fraction)):
        mask = rng.random(shape) < fraction
        sim = prior.get("qr", probe) * 3.0
        values = sim.mean(axis=0) + rng.standard_normal(shape)
        batches.append(GriddedObs(name=name, values=values,
                                  errors=np.full(shape, 5.0),
                                  simulated=sim, mask=mask, localization=loc))
    sloc = Localization(horizontal_m=sfc_horizontal_m, vertical_m=sfc_vertical_m)
    for name, field in (("sfc_t2", "thp"), ("sfc_u10", "u"), ("sfc_v10", "v"),
                        ("sfc_td", "qv")):
        mask = np.zeros(shape, bool)
        flat = rng.choice(ny * nx, size=min(surface_points, ny * nx // 4),
                          replace=False)
        mask[0].reshape(-1)[flat] = True
        sim = prior.get(field, probe).copy()
        values = sim.mean(axis=0) + rng.standard_normal(shape)
        batches.append(GriddedObs(name=name, values=values,
                                  errors=np.full(shape, 2.0), simulated=sim,
                                  mask=mask, localization=sloc))
    return prior, batches, grid, loc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--shape", type=int, nargs=3, default=(49, 241, 241))
    parser.add_argument("--radars", type=int, default=8)
    parser.add_argument("--fields", type=int, default=len(FIELDS))
    parser.add_argument("--device", choices=("host", "cuda"), default="cuda")
    parser.add_argument("--route", choices=("auto", "cuda-resident",
                                            "host-staged-cuda", "host"),
                        default="auto")
    parser.add_argument("--memory-budget-mib", type=float, default=14336)
    parser.add_argument("--cartesian", action="store_true",
                        help="no lat/lon: index-space horizontal distances")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from gpuwm.da import radar_assimilation as ra
    from gpuwm.da.letkf import LetkfConfig, LetkfDiagnostics, analyze

    t0 = time.perf_counter()
    fields = FIELDS[:args.fields]
    prior, batches, grid, loc = build_case(
        members=args.members, shape=tuple(args.shape), radars=args.radars,
        fields=fields, geodesic=not args.cartesian)
    build_seconds = time.perf_counter() - t0
    config = LetkfConfig(localization=loc, analysis_fields=fields,
                         rtps_alpha=0.9, memory_budget_mib=args.memory_budget_mib)
    namespace = np
    if args.device == "cuda":
        import cupy as namespace
    diagnostics = LetkfDiagnostics()
    t1 = time.perf_counter()
    if args.route in ("auto", "host"):
        result = ra._execute_analysis(analyze, prior, batches, grid, config,
                                      namespace=namespace, device=args.device,
                                      diagnostics=diagnostics)
        storage, attempts = result[4], result[5]
        vars(diagnostics).update(vars(result[1]))
    else:
        increments, diag, stage, unstage = ra._analysis_attempt(
            analyze, prior, batches, grid, config, namespace=namespace,
            storage=args.route, supports_staging=True, progress=None,
            diagnostics=diagnostics)
        storage, attempts = args.route, [dict(stage_seconds=stage,
                                              unstage_seconds=unstage)]
    wall = time.perf_counter() - t1
    report = {
        "schema": "gpuwm-da.analysis-bench.v1",
        "members": args.members, "shape": list(args.shape),
        "radars": args.radars, "fields": list(fields),
        "device": args.device, "storage": storage, "attempts": attempts,
        "build_seconds": build_seconds, "analysis_wall_seconds": wall,
        "obs_counts": {b.name: int(np.count_nonzero(b.mask)) for b in batches},
        "diagnostics": {k: v for k, v in vars(diagnostics).items()
                        if isinstance(v, (int, float, str, bool, type(None)))},
    }
    text = json.dumps(report, indent=1, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
