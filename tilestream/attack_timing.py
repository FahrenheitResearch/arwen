"""ATTACK 4: the chain's cost in time, in a window where the physics FIRES.

``tilestream.vram_share_bench`` is this workstream's timing lane and its own
docstring states the rule: "Radiation and cumulus have to actually fire, on
both sides."  THE CODE DOES NOT ENFORCE THAT.  ``main`` compares only
``arms[0]['fired'] != arms[1]['fired']`` -- equality BETWEEN the arms -- and
never checks that either count is non-zero.  The one result the lane has
produced (the stage-3 log on the 2x4090) reports

    19169.0 ms/step  as shipped
    23637.2 ms/step  shared + chain
    cadence fired in the timed window: {'noah': 4, 'sfclay': 4, 'ysu': 4}

-- ``radiation`` and ``cumulus`` are both ABSENT from that dict, which the
lane prints only for non-zero counts.  At ``radt = 12 min`` with ``dt = 3 s``
radiation is due on step 1, the warm-up sweep consumes it, and the four timed
steps reach step 241 never.  So the only number that exists for the cost of
the compute chain was taken in a window where the two most expensive schemes
in the configuration fired ZERO times -- the exact window this project has
already lost hours to.

This file measures the same comparison with the cadence shortened so that
radiation and cumulus fire SEVERAL times inside the timed window, prints the
counts for both arms, and REFUSES to print a ratio if either count is zero on
either side.  The compute window is 288x288x49 = 4.1 Mcell per tile step,
which is a saturated GPU, not a launch-latency measurement.
"""
import argparse
import sys

import cupy as cp

from tilestream import harness, vram, vram_share_bench as bench
from tilestream.vram_probe import RUNGS

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=512)
p.add_argument("--ny", type=int, default=512)
p.add_argument("--tile", type=int, default=256)
p.add_argument("--nsteps", type=int, default=6)
p.add_argument("--nbuffers", type=int, default=2)
p.add_argument("--radt-minutes", type=float, default=0.15)
p.add_argument("--cudt-minutes", type=float, default=0.15)
p.add_argument("--rung", default="full+MYNN+Noah-MP")
args = p.parse_args()

RUNG = args.rung


def make_cfg(rung, nx, ny, nz=49):
    """``vram_probe.make_cfg`` with the physics cadence shortened.

    Nothing else changes: same rung, same dx/dt, same sounding.  Shortening
    ``radt``/``cudt`` moves radiation and cumulus INTO the timed window
    instead of leaving them 240 steps away, which is the only difference
    between a physics timing and a dycore timing at this rung.
    """
    # RUNGS already carries radt_minutes=12.0 and cudt_minutes=5.0 (that is
    # the production cadence, and the reason the shipped lane's timed window
    # sees neither scheme), so these have to REPLACE the rung's values, not
    # be passed alongside them.
    kw = dict(RUNGS[rung])
    kw["radt_minutes"] = args.radt_minutes
    kw["cudt_minutes"] = args.cudt_minutes
    return harness.make_config(nx, ny, nz, **kw)


bench.make_cfg = make_cfg          # one_arm and seed_physics_host_store use it

free, total = cp.cuda.runtime.memGetInfo()
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(f"cupy {cp.__version__}  {name}  {free / 2**30:.2f} GiB free of "
      f"{total / 2**30:.2f}")
if free / total < 0.85:
    print("*** THE CARD IS NOT IDLE.  A timing taken next to another tenant "
          "is not a timing; refusing.")
    raise SystemExit(2)
print(f"{args.nx}x{args.ny}x49 pinned host store, {args.tile}^2 tiles, "
      f"nbuffers={args.nbuffers}, N={args.nsteps}, rung {RUNG}, "
      f"radt={args.radt_minutes} min, cudt={args.cudt_minutes} min")
print("=" * 78)

arms = []
for label, kw in (
        ("as shipped: private workspaces, no compute chain",
         dict(share=False, chain=False)),
        ("shared workspaces + compute chain",
         dict(share=True, chain=True)),
):
    try:
        rec = bench.one_arm(nx=args.nx, ny=args.ny, nz=49, tile=args.tile,
                            nsteps=args.nsteps, nbuffers=args.nbuffers,
                            rung=RUNG, **kw)
    except cp.cuda.memory.OutOfMemoryError as exc:
        print(f"  DOES NOT FIT -- {exc}\n        {label}")
        vram.trim_pool()
        continue
    rec["label"] = label
    arms.append(rec)
    print(f"  {rec['ms_per_step']:9.1f} ms/step  {rec['ns_per_cell']:7.3f} "
          f"ns/cell  compute {rec['tile_compute']}  "
          f"pool_used {rec['pool_used'] / 2**30:.3f} GiB  "
          f"pool_total {rec['pool_total'] / 2**30:.3f}  "
          f"shared {rec['shared_bytes'] / 2**30:.3f}")
    print(f"        {label}")
    print(f"        FIRED in the timed window: {rec['fired']}")
    print(f"        (warm-up window: {rec['warm_fired']})")

if len(arms) < 2:
    print("only one arm ran; nothing to compare")
    raise SystemExit(1)

bad = [a["label"] for a in arms
       if not a["fired"].get("radiation") or not a["fired"].get("cumulus")]
if bad:
    print()
    print("*** RADIATION OR CUMULUS FIRED ZERO TIMES in: "
          + "; ".join(bad))
    print("*** REFUSING to print a ratio -- that is the measurement this "
          "project has already been burned by.")
    raise SystemExit(1)
if arms[0]["fired"] != arms[1]["fired"]:
    print("*** THE TWO ARMS DID NOT FIRE THE SAME PHYSICS; not a comparison.")
    raise SystemExit(1)

ratio = arms[1]["ms_per_step"] / arms[0]["ms_per_step"]
saved = arms[0]["pool_total"] - arms[1]["pool_total"]
print()
print(f"  shared+chained is {ratio:.3f}x the shipped time "
      f"({100 * (ratio - 1):+.1f}%) and holds {saved / 2**30:+.3f} GiB less "
      f"pool_total ({arms[0]['pool_total'] / max(arms[1]['pool_total'], 1):.2f}x)")
print(f"  both arms fired {arms[0]['fired']}")
