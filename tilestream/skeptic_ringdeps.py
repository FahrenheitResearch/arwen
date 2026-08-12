"""Independent check of the probe's ring/multi-GPU serialisation claim.

The probe reported, "measured structurally at 2048^2, tile 512, 16 tiles":

  * block  -- GPU 1's FIRST tile (8) depends on GPU 0's tiles 4, 5 and 7,
              GPU 0's LAST tile, so GPU 1 cannot start until GPU 0 finishes.
  * stripe -- 15 of 16 tiles carry cross-worker dependencies.

Both are structural properties of ``rings.build_ring_plan().patch_deps`` under
``mgstream.partition_tiles``, so they need no GPU.  This recomputes them from
the shipped code, and adds what the probe did not report: how much of the
critical path is actually serialised, and whether the conclusion survives at
other tile counts (the claim "two GPUs take as long as one" is a statement
about the whole sweep, not about tile 8).
"""

from __future__ import annotations

import argparse
import json

from tilestream import mgstream, rings
from tilestream import spec as tspec


def analyse(nx, ny, tile, nworkers=2, mode="block", halo=16):
    specs = tspec.plan_tiles(nx, ny, tile, tile, halo)
    plan = rings.build_ring_plan(specs)
    ntiles = len(specs)
    order, owner = mgstream.partition_tiles(ntiles, nworkers, mode)

    deps = [tuple(d) for d in plan.patch_deps]
    cross = {k: [d for d in deps[k] if owner[d] != owner[k]]
             for k in range(ntiles)}
    ncross = sum(1 for k in range(ntiles) if cross[k])

    # Critical path: earliest finish for each tile given (a) its ring deps and
    # (b) its own worker's serial order.  Unit cost per tile.
    finish = [0] * ntiles
    last_on_worker = {}
    for k in range(ntiles):
        ready = 0
        for d in deps[k]:
            ready = max(ready, finish[d])
        w = owner[k]
        if w in last_on_worker:
            ready = max(ready, finish[last_on_worker[w]])
        finish[k] = ready + 1
        last_on_worker[w] = k
    makespan = max(finish)

    # The same with one worker: strictly serial.
    serial = ntiles
    # The same ignoring ring deps entirely (shadow mode): perfect split.
    perfect = max(sum(1 for k in range(ntiles) if owner[k] == w)
                  for w in range(nworkers))

    first_of_w1 = next((k for k in range(ntiles) if owner[k] == 1), None)
    last_of_w0 = max((k for k in range(ntiles) if owner[k] == 0), default=None)

    return {
        "nx": nx, "ny": ny, "tile": tile, "ntiles": ntiles,
        "nworkers": nworkers, "mode": mode,
        "tiles_with_cross_worker_deps": ncross,
        "first_tile_of_worker1": first_of_w1,
        "its_cross_deps": cross.get(first_of_w1, []),
        "last_tile_of_worker0": last_of_w0,
        "depends_on_worker0_last": (last_of_w0 in cross.get(first_of_w1, [])
                                    if first_of_w1 is not None else None),
        "makespan_units": makespan,
        "serial_units": serial,
        "shadow_ideal_units": perfect,
        "ring_speedup_vs_serial": serial / makespan,
        "shadow_speedup_vs_serial": serial / perfect,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    out = []
    print("ring-mode cross-GPU dependency structure "
          "(unit cost per tile, 2 workers)\n")
    hdr = (f"{'domain':>10s} {'tile':>5s} {'n':>3s} {'mode':>6s} "
           f"{'xdeps':>6s} {'makespan':>9s} {'serial':>7s} "
           f"{'ring x':>7s} {'shadow x':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for nx, tile in ((2048, 512), (2048, 256), (1536, 384), (1440, 240),
                     (672, 336), (2048, 128)):
        for mode in ("block", "stripe"):
            r = analyse(nx, nx, tile, 2, mode)
            out.append(r)
            print(f"{nx:>6d}^2 {tile:>7d} {r['ntiles']:>3d} {mode:>6s} "
                  f"{r['tiles_with_cross_worker_deps']:>4d}/{r['ntiles']:<2d}"
                  f"{r['makespan_units']:>8d} {r['serial_units']:>7d} "
                  f"{r['ring_speedup_vs_serial']:>7.2f} "
                  f"{r['shadow_speedup_vs_serial']:>9.2f}")

    print("\nthe probe's headline case, 2048^2 tile 512 block:")
    r = analyse(2048, 2048, 512, 2, "block")
    print(f"  worker1's first tile = {r['first_tile_of_worker1']}, "
          f"cross-worker deps = {r['its_cross_deps']}")
    print(f"  worker0's last tile  = {r['last_tile_of_worker0']}, "
          f"is it a dependency? {r['depends_on_worker0_last']}")
    print(f"  ring makespan {r['makespan_units']} vs serial "
          f"{r['serial_units']} -> {r['ring_speedup_vs_serial']:.2f}x")
    print(f"  shadow ideal  {r['shadow_ideal_units']} -> "
          f"{r['shadow_speedup_vs_serial']:.2f}x")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
