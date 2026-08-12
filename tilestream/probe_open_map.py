"""WHERE the open-boundary tiled answer leaves the monolithic one.

A verdict of "seam-local" is not enough to choose a fix: with 14 x-seams and
10 y-seams in a 256x192 plan, "within 16 cells of a seam" covers most of the
domain.  This prints the actual bands -- the x indices and y indices at which
a column differs, compressed into runs, against the WINDOW EDGES of the plan
and the domain's own edges -- so the question "is the defect the open flag
reaching an interior seam, or is it the halo" is answered by looking at the
picture rather than by a threshold.

    python -m tilestream.probe_open_map open_x
    python -m tilestream.probe_open_map open_xy 3
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

from gpuwm.core import streaming
from tilestream import harness
from tilestream import spec as tspec
from tilestream import test_open_bc as gate


def runs(idx: np.ndarray) -> str:
    """``[0 1 2 15 16]`` -> ``'0-2,15-16'``."""
    if idx.size == 0:
        return "(none)"
    out, start, prev = [], int(idx[0]), int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = v
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    arm = argv[0] if argv else "open_x"
    nsteps = int(argv[1]) if len(argv) > 1 else 1
    rung = argv[2] if len(argv) > 2 else "dry"

    cfg = gate.open_cfg(gate.NX, gate.NY, rung, arm)
    halo = harness.halo_radius(cfg)
    px, py = streaming._periodic_axes(cfg)
    specs = tspec.plan_tiles(gate.NX, gate.NY, gate.TX, gate.TY, halo,
                             periodic_x=px, periodic_y=py)
    bnd = gate.domain_boundaries(cfg, arm)

    print(f"{arm} / {rung} / N={nsteps}  {gate.NX}x{gate.NY}x{gate.NZ} "
          f"tile {gate.TX}x{gate.TY} halo {halo} "
          f"periodic-plan x={px} y={py}")
    win_x = sorted({s.ci0 for s in specs} | {s.ci0 + s.cnx - 1 for s in specs})
    win_y = sorted({s.cj0 for s in specs} | {s.cj0 + s.cny - 1 for s in specs})
    print(f"  window x edges {win_x}")
    print(f"  window y edges {win_y}")
    print(f"  interior x edges "
          f"{sorted({s.i0 for s in specs} | {s.i1 for s in specs})}")
    print()

    ref = gate.monolithic(cfg, arm, (nsteps,), boundaries=bnd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got = gate.streamed(cfg, arm, nsteps, gate.TX, gate.TY,
                            boundaries=bnd, halo=halo)
    snap = ref["snapshots"][nsteps]

    da, db = gate.digest_arrays(snap), gate.digest_arrays(got)
    differing = sorted(n for n in da if da[n] != db.get(n))
    print(f"  {len(differing)} of {len(da)} carriers differ: {differing[:12]}")
    print()

    for name in differing[:6]:
        a = np.asarray(snap[name], dtype=np.float64)
        b = np.asarray(got[name], dtype=np.float64)
        if a.shape != b.shape or a.ndim < 2:
            continue
        d = np.abs(a - b)
        mask = d > 0.0
        while mask.ndim > 2:
            mask = mask.any(axis=0)
        xs = np.nonzero(mask.any(axis=0))[0]
        ys = np.nonzero(mask.any(axis=1))[0]
        print(f"  {name}: max|d|={d.max():.6g}  "
              f"{int(mask.sum())} of {mask.size} columns")
        print(f"      x: {runs(xs)}")
        print(f"      y: {runs(ys)}")
        # Per-tile: does this tile's OWN INTERIOR carry the difference?
        hit = [s.index for s in specs
               if mask[s.j0:s.j1, s.i0:s.i1].any()]
        print(f"      tiles whose interior differs: {len(hit)} of "
              f"{len(specs)}  {hit[:10]}")
    cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
