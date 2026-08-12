"""The halo MARGIN on an open-boundary domain, printed as data.

The halo is ``harness.halo_radius(cfg)`` and nothing else -- never a sweep.
This probe exists because the open-boundary case is the one where the margin
was predicted to vanish: the specified-BC seam perturbs the RK TENDENCY and
its cone is strictly inside the dycore's, but the open treatment assigns
STATE at the window edge, and a state perturbation present at the start of a
step has exactly the dycore's own radius.  If that prediction were right the
smallest passing halo would be AT the radius rather than below it.

So the number is measured and REPORTED, in the way ``tilestream.test_join``
reports its own: as data with the machine and the step count attached, never
as a value anything takes its halo from.  The margin moves with step count,
domain size and GPU -- the join lane found halo 15 passing on one 4090 and
halo 14 passing on the other 4090 of the same box -- so a single number here
is a fact about this run, not a recommendation.

    python -m tilestream.probe_open_halo open_xy 8
"""

from __future__ import annotations

import sys
import warnings

from gpuwm.core import streaming
from tilestream import harness
from tilestream import spec as tspec
from tilestream import test_open_bc as gate


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    arm = argv[0] if argv else "open_xy"
    nsteps = int(argv[1]) if len(argv) > 1 else 8
    widths = [int(w) for w in argv[2].split(",")] if len(argv) > 2 else \
        list(range(8, 19))

    cfg = gate.open_cfg(gate.NX, gate.NY, "dry", arm)
    prescribed = harness.halo_radius(cfg)
    px, py = streaming._periodic_axes(cfg)
    print(f"{arm} dry N={nsteps}  {gate.NX}x{gate.NY}x{gate.NZ} "
          f"tile {gate.TX}x{gate.TY}  plan periodic x={px} y={py}")
    print(f"  harness.halo_radius(cfg) = {prescribed} "
          f"(= 10 + 3*{cfg.time_step_sound}//2), and that is what a run uses")
    print(f"  {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    ref = gate.retry_on_oom(gate.monolithic, cfg, arm, (nsteps,))
    smallest = None
    for halo in sorted(widths):
        specs = tspec.plan_tiles(gate.NX, gate.NY, gate.TX, gate.TY, halo,
                                 periodic_x=px, periodic_y=py)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            got = gate.retry_on_oom(gate.streamed, cfg, arm, nsteps, gate.TX,
                                    gate.TY, halo=halo)
        res = gate.localise(ref["snapshots"][nsteps], got, specs, halo,
                            gate.NX, gate.NY)
        ok = res["bitexact"]
        if ok and smallest is None:
            smallest = halo
        print(f"    halo {halo:2d}  {'BIT-EXACT' if ok else 'differs':10s}"
              + ("" if ok else f"  ndiff={res['ndiff']}/{res['ntotal']} "
                               f"max|d|={res['max_abs']:.6g} "
                               f"verdict={res.get('verdict')}"))
        del got
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    print(f"  smallest passing halo on THIS card at N={nsteps}: {smallest}; "
          f"the prescribed {prescribed} clears it by "
          f"{prescribed - smallest if smallest else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
