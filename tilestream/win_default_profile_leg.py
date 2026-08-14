"""The DEFAULT PROFILE streamed, on Windows, against its own resident run.

The gate that did not exist.  Every streamed bit-exactness gate in the tree
PINS the tiling (``test_join.streamed`` takes ``tile_nx``/``tile_ny``
positionally, ``test_gate`` likewise), and a pinned tiling returns from
``streaming.decide`` before the planner is reached::

    if options.tile_nx is not None:
        # A pinned tiling asks no question, so it consults no planner ...
        return StreamingDecision(True, "[tiles] pins the tiling", ...)

So the planner-driven path -- which is what a bare ``[tiles] mode = "on"``
in a user's TOML actually runs -- had no GPU coverage at all, on any
platform.  On Linux it worked and nobody looked; on Windows
``autoplan.Machine.detect`` raised before ``[tiles]`` was read and streaming
could not start.

This leg configures ``mode = "on"`` and NOTHING ELSE, lets the planner pick
the tiling, and compares the result carrier-by-carrier against the same
configuration integrated as one resident block.

    python -m tilestream.win_default_profile_leg --json receipt.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from gpuwm.core import streaming
from tilestream import test_join as tj


def run_leg(nx: int, ny: int, nsteps: int, rung: str,
            *, bind_clock: bool = False) -> dict:
    cfg = tj.join_cfg(nx, ny, rung=rung)

    # THE CALL THAT USED TO RAISE.  No pinned tiling, so this consults
    # autoplan.Machine.detect -- the whole defect, in one line.
    options = streaming.StreamingOptions(mode="on")
    t0 = time.time()
    decision = streaming.decide(cfg, options)
    plan_seconds = time.time() - t0
    if not decision.stream:
        raise SystemExit(f"the planner declined to stream: {decision.reason}")

    import cupy as cp

    # The gate's own sequence (test_join.main): two unwarmed states give the
    # boundary tables a nonzero time tendency, and the snapshot is taken off
    # the first before both are released.
    bnd_a, _ = tj.build_domain(cfg, seed=tj.SEED, warmup=0)
    bnd_b, _ = tj.build_domain(cfg, seed=tj.SEED + 1, warmup=0)
    boundaries = tj.domain_boundaries(cfg, bnd_a, bnd_b)
    del bnd_a, bnd_b
    cp.get_default_memory_pool().free_all_blocks()

    # ``--bind-clock`` (task #219): the arm every production real-data root
    # actually runs -- a DomainClock bound to the external LBC mirror, WRF's
    # post-increment dtbc.  warmup=0 is load-bearing: a one-step warmup
    # shifts the retired elapsed-based dtbc by exactly +dt and makes the two
    # semantics numerically coincide at dt-multiple times, so a bound leg
    # with warmup=1 would pass on a build that drops the binding.
    warmup = 0 if bind_clock else 1
    t0 = time.time()
    ref = tj.monolithic(
        cfg, nsteps, boundaries=boundaries, warmup=warmup,
        clock=(tj.bound_lbc_clock(cfg, nsteps=nsteps) if bind_clock
               else None))
    resident_seconds = time.time() - t0

    report: dict = {}
    t0 = time.time()
    got = tj.streamed(cfg, nsteps,
                      decision.tile_nx, decision.tile_ny,
                      nbuffers=decision.nbuffers,
                      boundaries=boundaries, store=decision.store,
                      warmup=warmup,
                      clock=(tj.bound_lbc_clock(cfg, nsteps=nsteps)
                             if bind_clock else None),
                      report=report)
    streamed_seconds = time.time() - t0

    verdict = tj.compare(ref, got)
    return {
        "domain": f"{nx}x{ny}x{cfg.nz}",
        "rung": rung,
        "nsteps": nsteps,
        "dt": float(cfg.dt),
        "platform": sys.platform,
        "platform_detail": platform.platform(),
        "configured": "[tiles] mode = 'on'   (nothing else)",
        "lbc_clock": "bound (production dtbc)" if bind_clock else "unbound",
        "planner_tiling": f"{decision.tile_nx}x{decision.tile_ny}",
        "planner_nbuffers": decision.nbuffers,
        "planner_halo": decision.halo,
        "planner_store": decision.store,
        "planner_reason": decision.reason,
        "plan_seconds": round(plan_seconds, 3),
        "resident_seconds": round(resident_seconds, 2),
        "streamed_seconds": round(streamed_seconds, 2),
        "tiling_tax": round(streamed_seconds / max(resident_seconds, 1e-9), 3),
        "bitexact": verdict["bitexact"],
        "ncarriers": verdict["ntotal"],
        "ndiffering": verdict["ndiff"],
        "differing": verdict["differing"],
        "max_abs": verdict["max_abs"],
        "nonfinite": verdict["nonfinite"],
        "carrier_digests": tj.digest_arrays(got),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=438)
    ap.add_argument("--ny", type=int, default=350)
    ap.add_argument("--nsteps", type=int, default=12)
    ap.add_argument("--rung", default="dry")
    ap.add_argument("--bind-clock", action="store_true",
                    help="bind the production DomainClock on both arms "
                         "(WRF post-increment dtbc; task #219)")
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    out = run_leg(args.nx, args.ny, args.nsteps, args.rung,
                  bind_clock=bool(args.bind_clock))
    digests = out.pop("carrier_digests")

    print(f"domain          {out['domain']}  rung={out['rung']}  "
          f"nsteps={out['nsteps']}  dt={out['dt']} s")
    print(f"platform        {out['platform_detail']}")
    print(f"configured      {out['configured']}")
    print(f"lbc clock       {out['lbc_clock']}")
    print(f"planner chose   {out['planner_tiling']} tiles, "
          f"nbuffers={out['planner_nbuffers']}, halo={out['planner_halo']}, "
          f"store={out['planner_store']}  ({out['plan_seconds']} s)")
    print(f"resident        {out['resident_seconds']} s")
    print(f"streamed        {out['streamed_seconds']} s   "
          f"(tiling tax {out['tiling_tax']}x)")
    print(f"carriers        {out['ncarriers']} compared, "
          f"{out['ndiffering']} differing, {out['nonfinite']} non-finite")
    print(f"VERDICT         {'BIT-EXACT' if out['bitexact'] else 'DIFFERS'}")
    if not out["bitexact"]:
        print(f"  differing     {out['differing']}")
        print(f"  max_abs       {out['max_abs']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({**out, "carrier_digests": digests}, handle, indent=2)
        print(f"receipt         {args.json}")
    return 0 if out["bitexact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
