"""From the measured 1 m numbers to 8 x RTX PRO 6000.  Arithmetic, labelled.

Everything here is EXTRAPOLATION.  The only measured inputs are the ones
named in ``MEASURED``; everything else is an assumption with its ground
stated next to it.  Two of the assumptions are load-bearing and neither was
measurable on the hardware this probe had:

* the per-GPU speed of an RTX PRO 6000 relative to the RTX 5090 the numbers
  were taken on.  Both cards carry ~1.79 TB/s of memory bandwidth and the
  PRO 6000 has 10.6% more cores, so the ratio is 1.00 if the dycore is
  bandwidth-bound and 1.106 if it is core-bound.  Both are carried.
* 8-GPU parallel efficiency.  The project has MEASURED 1.99x on two GPUs at
  1536^2 with a 2-5% seam cost; 8 GPUs are not measured anywhere, so a
  range is carried and the halo arithmetic that motivates it is shown.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GIB = 2.0 ** 30


def halo_overhead(side_cells: float, ngpu: int, halo: int = 16) -> float:
    """Redundant-compute factor for a 2-D decomposition into ``ngpu`` parts.

    ``halo`` is ``harness.halo_radius`` at ``time_step_sound = 4``: 16 mass
    cells per side per step, the project's own measured figure, not tuned.
    A near-square decomposition minimises perimeter, so that is what is
    assumed; a worse split costs more.
    """
    px = int(round(math.sqrt(ngpu)))
    while ngpu % px:
        px -= 1
    py = ngpu // px
    sx, sy = side_cells / px, side_cells / py
    return ((sx + 2 * halo) * (sy + 2 * halo)) / (sx * sy)


def box(wall_seconds, *, ns_cell_step, nsteps, ngpu, scale, eff, nz,
        bytes_per_cell, vram_gib_per_gpu, usable=0.90, halo=16):
    """Largest cubic-footprint box that finishes in ``wall_seconds``.

    Solves for the horizontal side, INCLUDING the halo redundancy that the
    side itself determines, by fixed-point iteration (two passes converge:
    the halo term is a few percent).
    """
    budget_cell_steps = (wall_seconds * ngpu * scale * eff) / (ns_cell_step * 1e-9)
    side = math.sqrt(budget_cell_steps / (nsteps * nz))
    for _ in range(40):
        redundancy = halo_overhead(side, ngpu, halo)
        side = math.sqrt(budget_cell_steps / (nsteps * nz * redundancy))
    cells = side * side * nz
    vram_needed = cells * bytes_per_cell / GIB
    vram_avail = ngpu * vram_gib_per_gpu * usable
    return {
        "wall_hours": wall_seconds / 3600.0,
        "side_cells": side,
        "side_km": side / 1000.0,
        "nz": nz,
        "cells": cells,
        "gcells": cells / 1e9,
        "halo_redundancy": halo_overhead(side, ngpu, halo),
        "vram_needed_gib": vram_needed,
        "vram_available_gib": vram_avail,
        "vram_fits": vram_needed <= vram_avail,
        "vram_limited_side_cells": math.sqrt(
            vram_avail * GIB / bytes_per_cell / nz),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns-cell-step", type=float, required=True)
    ap.add_argument("--bytes-per-cell", type=float, required=True)
    ap.add_argument("--cfl-max", type=float, required=True)
    ap.add_argument("--wind", type=float, default=100.0)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--lifecycle-s", type=float, default=600.0)
    ap.add_argument("--nz", type=int, nargs="+", default=[150, 300, 400])
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--vram-gib", type=float, default=96.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    dt = a.cfl_max * a.dx / a.wind
    nsteps = a.lifecycle_s / dt
    scales = {"bandwidth-bound (1.00x)": 1.00, "core-bound (1.106x)": 1.106}
    effs = {"8 GPU @ 95%": 0.95 * a.ngpu / a.ngpu,
            "8 GPU @ 85%": 0.85,
            "8 GPU @ 70%": 0.70}
    effs = {"95%": 0.95, "85%": 0.85, "70%": 0.70}

    out = {"inputs": vars(a) | {"out": str(a.out)},
           "dt_seconds": dt, "steps_for_lifecycle": nsteps, "rows": []}
    print(f"dt = {dt * 1000:.3f} ms   (CFL {a.cfl_max} at {a.wind} m/s, "
          f"dx = {a.dx} m)")
    print(f"steps for a {a.lifecycle_s:.0f} s lifecycle = {nsteps:,.0f}")
    print()
    for nz in a.nz:
        for sname, scale in scales.items():
            for ename, eff in effs.items():
                for days, label in ((1, "1 day"), (3, "3 days"), (7, "1 week")):
                    r = box(days * 86400.0, ns_cell_step=a.ns_cell_step,
                            nsteps=nsteps, ngpu=a.ngpu, scale=scale, eff=eff,
                            nz=nz, bytes_per_cell=a.bytes_per_cell,
                            vram_gib_per_gpu=a.vram_gib)
                    r |= {"nz": nz, "scale": sname, "efficiency": ename,
                          "budget": label}
                    out["rows"].append(r)
    # compact headline table: the central assumption set
    print(f"{'nz':>5} {'budget':>7} {'side (m)':>9} {'Gcell':>8} "
          f"{'VRAM GiB':>9} {'fits 8x96':>10} {'halo':>6}")
    for r in out["rows"]:
        if r["scale"].startswith("bandwidth") and r["efficiency"] == "85%":
            print(f"{r['nz']:>5} {r['budget']:>7} {r['side_cells']:9.0f} "
                  f"{r['gcells']:8.2f} {r['vram_needed_gib']:9.1f} "
                  f"{str(r['vram_fits']):>10} {r['halo_redundancy']:6.3f}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(f"\nWROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
