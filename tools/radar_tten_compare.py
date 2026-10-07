#!/usr/bin/env python3
"""Compare one ``rw_nexrad grid-ref`` window against the Python superob.

Validation, not production: the Python side is the existing radar path
(``rw_nexrad decode`` to a pack, ``superob_volume``, ``merge_contributions``)
run on the SAME volumes the Rust window used, and both are read in NOAA's
``ref2tten`` convention (echo dBZ, -99 observed no echo, -99999 no
coverage).  Two Python configurations run:

* ``deck``: exactly ``tools/conus_da_observations.py``'s parameters (REF,
  VEL, RHO decoded; region-global dealias; CC QC; clear air from censor
  codes), ``z_reduce`` max;
* ``aligned``: the same superob with the rules the Rust product does not
  share switched to the Rust ones where Python has a knob (no CC QC, a
  -100 dBZ floor in place of -15, one clear gate in place of four, REF and
  RHO decoded, no dealias), so what is left between the two is the Rust
  declared RhoHV filter and arithmetic.

The report goes to ``--out`` as JSON and a short text summary on stdout.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_GRID = None


def _init(grid_path):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    global _GRID
    from gpuwm.obs.target_grid import TargetGrid
    _GRID = TargetGrid.from_wrfout(Path(grid_path))


def _params(config):
    from gpuwm.obs.cc_qc import CcQcParams
    from gpuwm.obs.dealias import DealiasParams
    from gpuwm.obs.superob import SuperobParams
    if config == "deck":
        return SuperobParams(max_range_km=250.0, max_elevation_deg=20.0,
                             dealias=DealiasParams(engine="region-global"),
                             cc_qc=CcQcParams())
    return SuperobParams(max_range_km=250.0, max_elevation_deg=20.0, dealias=None,
                         cc_qc=None, min_reflectivity_dbz=-100.0,
                         clear_air_min_gates=1.0)


def _one(task):
    from gpuwm.obs.nexrad import find_nexrad_bin, run_decode
    from gpuwm.obs.superob import superob_volume
    from gpuwm.obs.sweeps import read_sweep_pack
    volume, work, config = task
    t0 = time.perf_counter()
    pack = Path(work) / (Path(volume).name + f".{config}.pack")
    moments = ("REF", "VEL", "RHO") if config == "deck" else ("REF", "RHO")
    run_decode(find_nexrad_bin(), volume=Path(volume), out=pack, moments=moments,
               max_range_km=250.0, max_elevation_deg=20.0, censor_flags=True)
    swept = read_sweep_pack(pack)
    t_decode = time.perf_counter() - t0
    t0 = time.perf_counter()
    contribution = superob_volume(swept, _GRID, params=_params(config),
                                  clear_air_from_censor=True)
    t_superob = time.perf_counter() - t0
    pack.unlink()
    return contribution, t_decode, t_superob


def python_product(volumes, grid_path, work, config, workers):
    import numpy as np
    from gpuwm.obs.superob import merge_contributions
    from gpuwm.obs.target_grid import TargetGrid
    grid = TargetGrid.from_wrfout(Path(grid_path))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = [None] * len(volumes)
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                initargs=(str(grid_path),),
                                mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {pool.submit(_one, (str(v), str(work), config)): i
                for i, v in enumerate(volumes)}
        for job in cf.as_completed(jobs):
            results[jobs[job]] = job.result()
    t_volumes = time.perf_counter() - t0
    t0 = time.perf_counter()
    merged = merge_contributions([r[0] for r in results], grid, params=_params(config),
                                 z_reduce="max")
    t_merge = time.perf_counter() - t0
    ref = np.full(merged.z_obs.shape, -99999.0, dtype=np.float32)
    ref[merged.z0_mask != 0] = -99.0
    ref[merged.z_mask != 0] = merged.z_obs[merged.z_mask != 0].astype(np.float32)
    seconds = {"volumes_wall": round(t_volumes, 2), "merge": round(t_merge, 2),
               "wall": round(t_volumes + t_merge, 2),
               "decode_sum": round(sum(r[1] for r in results), 2),
               "superob_sum": round(sum(r[2] for r in results), 2)}
    counts = {}
    for r in results:
        payload = r[0].counts.to_payload()
        for key in ("cc_reflectivity_gates_rejected", "gates_below_floor",
                    "gates_out_of_grid", "gates_out_of_column"):
            counts[key] = counts.get(key, 0) + int(payload.get(key, 0) or 0)
    return ref, seconds, counts


def compare(rust, python):
    import numpy as np
    cats = {}
    for name, field in (("rust", rust), ("python", python)):
        cats[name] = np.where(field > -90.0, 0, np.where(field == -99.0, 1, 2))
    labels = ("echo", "clear", "none")
    table = {}
    total = rust.size
    for a in range(3):
        for b in range(3):
            n = int(((cats["rust"] == a) & (cats["python"] == b)).sum())
            table[f"rust_{labels[a]}__python_{labels[b]}"] = {
                "cells": n, "fraction": round(n / total, 6)}
    agree = int((cats["rust"] == cats["python"]).sum())
    both = (cats["rust"] == 0) & (cats["python"] == 0)
    diff = (rust[both].astype(np.float64) - python[both].astype(np.float64))
    stats = {}
    if diff.size:
        q = np.percentile(diff, [1, 5, 25, 50, 75, 95, 99])
        stats = {"cells": int(diff.size), "mean": round(float(diff.mean()), 4),
                 "std": round(float(diff.std()), 4),
                 "abs_max": round(float(np.abs(diff).max()), 4),
                 "percentiles_1_5_25_50_75_95_99": [round(float(v), 4) for v in q],
                 "fraction_exact": round(float((diff == 0).mean()), 6),
                 "fraction_abs_le_0p5": round(float((np.abs(diff) <= 0.5).mean()), 6),
                 "fraction_abs_le_2": round(float((np.abs(diff) <= 2).mean()), 6)}
    # Cells where one side says echo and the other not: their dBZ on the side
    # that has echo, which is what tells a floor or QC rule from geometry.
    rust_only = (cats["rust"] == 0) & (cats["python"] != 0)
    python_only = (cats["python"] == 0) & (cats["rust"] != 0)
    edges = [-100, -15, 0, 10, 20, 30, 45, 80]
    hist = lambda v: {f"{edges[i]}..{edges[i + 1]}": int(((v >= edges[i]) & (v < edges[i + 1])).sum())
                      for i in range(len(edges) - 1)}
    return {"cells": int(total), "category_agreement": round(agree / total, 6),
            "categories": table, "both_echo_dbz_diff_rust_minus_python": stats,
            "rust_only_echo_dbz_histogram": hist(rust[rust_only]),
            "python_only_echo_dbz_histogram": hist(python[python_only])}


def main(argv=None) -> int:
    import numpy as np
    from gpuwm.obs.radar_tten_grid import grid_reflectivity
    from gpuwm.obs.target_grid import TargetGrid
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--window-dir", required=True,
                        help="OUT/<stamp> of tools/radar_tten_windows.py")
    parser.add_argument("--grid-wrfout", required=True)
    parser.add_argument("--grid-descriptor", required=True)
    parser.add_argument("--out", required=True, help="report JSON")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--configs", nargs="*", default=["deck", "aligned"])
    args = parser.parse_args(argv)

    window = Path(args.window_dir)
    receipt = json.loads((window / "ref.json").read_text())
    volumes = [Path(v["path"]) for v in receipt["volumes"]]
    work = Path(args.out).parent / "compare-work"
    grid = TargetGrid.from_wrfout(Path(args.grid_wrfout))

    t0 = time.perf_counter()
    rust_max, rust_receipt = grid_reflectivity(
        grid, volumes, work / "rust-max", reduce="max", threads=args.threads,
        grid_descriptor=Path(args.grid_descriptor))
    rust_wall = time.perf_counter() - t0
    report = {"window": receipt["window"], "volumes": len(volumes),
              "rust": {"wall_seconds": round(rust_wall, 2),
                       "internal_seconds": rust_receipt["seconds"],
                       "window_product_seconds": receipt["seconds"],
                       "gates": rust_receipt["counts"]["gates"],
                       "threads": args.threads},
              "python": {}}
    for config in args.configs:
        ref, seconds, counts = python_product(volumes, Path(args.grid_wrfout),
                                              work / config, config, args.workers)
        np.save(work / f"python-{config}-max.npy", ref)
        report["python"][config] = {"seconds": seconds, "workers": args.workers,
                                    "counts": counts,
                                    "compare_max": compare(rust_max, ref)}
        print(json.dumps({config: report["python"][config]}, indent=1), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    raise SystemExit(main())
