"""A/B the observation-sparse device LETKF against another analysis route.

Builds a real-shape synthetic analysis (the generator of the profiling
lane's ``tools.da_analysis_bench``, carried here so this tool stands alone:
241 x 241 x 49, 32 members, 14 fields, windowed radial-velocity batches,
whole-grid reflectivity and clear air, surface batches) and solves it with
``cuda-obs-sparse`` twice (run-to-run bytes) and with one reference route,
then reports wall seconds per route and, per field, the max and RMS of the
increment difference against the reference beside the increment RMS
itself.  Measurement tooling only; nothing in a cycle calls this.

    python -m tools.da_letkf_device_ab --reference cuda-resident --out ab.json
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


def _solve(route, prior, batches, grid, config):
    import cupy as cp

    from gpuwm.da import radar_assimilation as ra
    from gpuwm.da.letkf import LetkfDiagnostics, analyze

    diagnostics = LetkfDiagnostics()
    cp.get_default_memory_pool().free_all_blocks()
    started = time.perf_counter()
    namespace = np if route == "host" else cp
    increments, diagnostics, stage, unstage = ra._analysis_attempt(
        analyze, prior, batches, grid, config, namespace=namespace,
        storage=route, supports_staging=True, progress=None,
        diagnostics=diagnostics)
    wall = time.perf_counter() - started
    keep = {k: v for k, v in vars(diagnostics).items()
            if isinstance(v, (int, float, str, bool, type(None)))}
    return increments, dict(route=route, wall_seconds=wall,
                            stage_seconds=stage, unstage_seconds=unstage,
                            diagnostics=keep)


def compare(new, ref) -> dict:
    """Per field: max |d|, RMS d, RMS of the reference increment, max |ref|."""
    rows = {}
    for name in ref:
        a = np.asarray(new[name], dtype=np.float64)
        b = np.asarray(ref[name], dtype=np.float64)
        d = a - b
        rows[name] = {
            "max_abs_diff": float(np.abs(d).max()),
            "rms_diff": float(np.sqrt(np.mean(d * d))),
            "rms_increment": float(np.sqrt(np.mean(b * b))),
            "max_abs_increment": float(np.abs(b).max()),
            "identical_bytes": bool(a.tobytes() == b.tobytes()),
            "zero_support_equal": bool(np.array_equal(
                np.all(a == 0.0, axis=0), np.all(b == 0.0, axis=0))),
        }
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--members", type=int, default=32)
    parser.add_argument("--shape", type=int, nargs=3, default=(49, 241, 241))
    parser.add_argument("--radars", type=int, default=8)
    parser.add_argument("--fields", type=int, default=14)
    parser.add_argument("--reference", default="cuda-resident",
                        help="cuda-resident, host-staged-cuda, host, "
                             "cuda-obs-sparse or none")
    parser.add_argument("--memory-budget-mib", type=float, default=14336)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--route", default="cuda-obs-sparse",
                        help="the route under test (default cuda-obs-sparse)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from gpuwm.da.letkf import LetkfConfig

    fields = FIELDS[:args.fields]
    t0 = time.perf_counter()
    prior, batches, grid, loc = build_case(
        members=args.members, shape=tuple(args.shape), radars=args.radars,
        fields=fields)
    build = time.perf_counter() - t0
    config = LetkfConfig(localization=loc, analysis_fields=fields,
                         rtps_alpha=0.9,
                         memory_budget_mib=args.memory_budget_mib)
    report = {"schema": "gpuwm-da.letkf-device-ab.v1",
              "members": args.members, "shape": list(args.shape),
              "radars": args.radars, "fields": list(fields),
              "build_seconds": build, "runs": []}
    first = None
    for n in range(args.repeat):
        inc, row = _solve(args.route, prior, batches, grid, config)
        report["runs"].append(row)
        print(json.dumps(row, default=str), flush=True)
        if first is None:
            first = inc
            import hashlib
            report["sha256"] = {name: hashlib.sha256(
                np.ascontiguousarray(inc[name]).tobytes()).hexdigest()
                for name in fields}
        else:
            report[f"run_to_run_{n}"] = {
                name: bool(first[name].tobytes() == inc[name].tobytes())
                for name in fields}
            del inc
    if args.reference != "none":
        ref, row = _solve(args.reference, prior, batches, grid, config)
        report["runs"].append(row)
        print(json.dumps(row, default=str), flush=True)
        report["against_reference"] = {"reference": args.reference,
                                       "fields": compare(first, ref)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report.get("against_reference"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
