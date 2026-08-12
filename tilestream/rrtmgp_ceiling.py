"""Largest square tile that survives a radiation step, with two buffers.

WHY NOT ``vram_probe ceiling``.  That bisector drives the scratch-arena and
MYNN sharing from branch ``tilestream-vram``, whose ``make_physics_tile_state``
and ``default_builder`` take a ``shared=`` argument this branch's do not.
Mixing the two would measure that branch's reclamation and this one's at the
same time and report the sum as either.  This bisector varies ONE thing --
the RRTMGP chunk workspace, shipped layout against tightened -- and holds
everything else at what this branch actually ships.  The arena/MYNN sharing
is orthogonal and multiplies with this; it is not re-derived here.

WHAT "FITS" MEANS.  Not "allocated": ``RESULTS.md`` records a 2400^2 state
that allocated cleanly and died inside the first ``dycore.step``.  A trial
here builds ``nbuffers`` buffers, steps twice, and the SECOND step is forced
to be a radiation step -- the largest transient in the configuration and one
a short run at the production 12-minute cadence would never reach.  The
radiation fire count is printed for every trial, and a trial that did not
fire is refused rather than counted as a survivor.

THE CAP.  The CuPy pool is capped at the card's total minus the measured
non-pool footprint (CUDA context + NVRTC module images), so every refusal is
a budget refusal citing the cap and not a device-level OOM caused by a
co-tenant.  One fresh subprocess per trial: an in-process retry after an OOM
measures a fragmented pool, not a card.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

GIB = 1 << 30
EXIT_FITS, EXIT_NO_FIT, EXIT_NO_RADIATION = 0, 3, 4


def trial(*, rung: str, n: int, nz: int, nbuffers: int, tight: bool,
          column_chunk: int | None, pool_limit_bytes: int) -> int:
    import cupy as cp

    from tilestream import driver as tdriver, harness
    from tilestream import physics_inventory as physinv
    from tilestream.rrtmgp_bench import RUNGS, _force_radiation_due
    from tilestream.rrtmgp_lazy import attach_lazy

    cp.get_default_memory_pool().set_limit(size=int(pool_limit_bytes))
    cfg = harness.make_config(n, n, nz, **RUNGS[rung])
    try:
        buffers = [tdriver.make_physics_tile_state(cfg, warmup=1)
                   for _ in range(nbuffers)]
        state = buffers[0]
        driver_obj = state.physics
        chunk = min(int(column_chunk or 3125), n * n)
        if tight:
            from tilestream import rrtmgp_tight

            workspace = rrtmgp_tight.build(nz, chunk, float(state.p_top),
                                           tight=True, lazy=False)
        else:
            from gpuwm.core.model import SharedRRTMGPChunkWorkspace

            workspace = SharedRRTMGPChunkWorkspace(
                nz=nz, column_chunk=chunk, p_top=float(state.p_top))
        # Every buffer shares the one workspace -- that is what the object is
        # for, and it is the configuration both arms of this comparison use.
        for buf in buffers:
            attach_lazy(buf, workspace)
        before = dict(getattr(driver_obj, "call_counts", {}) or {})
        harness.run_steps(state, cfg, 1)
        _force_radiation_due(state, cfg, driver_obj)
        harness.run_steps(state, cfg, 1)
        cp.cuda.runtime.deviceSynchronize()
        counts = {k: int(v) - int(before.get(k, 0))
                  for k, v in (getattr(driver_obj, "call_counts", {})
                               or {}).items()
                  if int(v) - int(before.get(k, 0))}
    except cp.cuda.memory.OutOfMemoryError as exc:
        print(f"  {n}^2 x{nz} nbuffers={nbuffers} tight={tight}: NO FIT -- "
              f"{str(exc)[:120]}")
        return EXIT_NO_FIT
    pool = cp.get_default_memory_pool()
    print(f"  {n}^2 x{nz} nbuffers={nbuffers} tight={tight} "
          f"chunk={chunk}: FITS  W={workspace.nbytes / GIB:.3f} GiB  "
          f"pool_total={pool.total_bytes() / GIB:.3f}  "
          f"pool_used={pool.used_bytes() / GIB:.3f}  cadence={counts}")
    if not counts.get("radiation"):
        print("  *** radiation did NOT fire; this is not the real peak")
        return EXIT_NO_RADIATION
    return EXIT_FITS


def bisect(args) -> dict:
    lo, hi, step = int(args.low), int(args.high), int(args.step)
    best, log = None, []
    while lo <= hi:
        mid = max(((lo + hi) // 2 // step) * step, lo)
        argv = [sys.executable, "-u", "-m", "tilestream.rrtmgp_ceiling",
                "trial", "--rung", args.rung, "--n", str(mid),
                "--nz", str(args.nz), "--nbuffers", str(args.nbuffers),
                "--pool-limit-bytes", str(args.pool_limit_bytes)]
        if args.tight:
            argv.append("--tight")
        if args.column_chunk:
            argv += ["--column-chunk", str(args.column_chunk)]
        proc = subprocess.run(argv, capture_output=True, text=True)
        tail = (proc.stdout.strip().splitlines() or [proc.stderr[-200:]])[-1]
        print(f"  trial {mid:>5d}^2 -> exit {proc.returncode}: {tail.strip()}")
        log.append({"n": mid, "exit": proc.returncode, "line": tail.strip()})
        if proc.returncode == EXIT_FITS:
            best, lo = mid, mid + step
        elif proc.returncode == EXIT_NO_FIT:
            hi = mid - step
        else:
            raise RuntimeError(
                f"trial at {mid}^2 exited {proc.returncode}, which is not a "
                f"fit and not an OOM; a bisection would turn that into a "
                f"ceiling:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    print(f"  LARGEST THAT FITS: {best}^2 x {args.nz}"
          + ("" if best is None
             else f" = {best * best * args.nz / 1e6:.1f} Mcell, "
                  f"{best * best:,} columns"))
    return {"best": best, "log": log, "rung": args.rung,
            "tight": bool(args.tight), "nbuffers": args.nbuffers,
            "column_chunk": args.column_chunk,
            "pool_limit_bytes": args.pool_limit_bytes}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rung", default="full")
    common.add_argument("--nz", type=int, default=49)
    common.add_argument("--nbuffers", type=int, default=2)
    common.add_argument("--tight", action="store_true")
    common.add_argument("--column-chunk", type=int, default=None)
    common.add_argument("--pool-limit-bytes", type=int, required=True)
    t = sub.add_parser("trial", parents=[common])
    t.add_argument("--n", type=int, required=True)
    b = sub.add_parser("bisect", parents=[common])
    b.add_argument("--low", type=int, default=64)
    b.add_argument("--high", type=int, default=768)
    b.add_argument("--step", type=int, default=16)
    b.add_argument("--json", default=None)
    args = p.parse_args(argv)

    if args.cmd == "trial":
        return trial(rung=args.rung, n=args.n, nz=args.nz,
                     nbuffers=args.nbuffers, tight=args.tight,
                     column_chunk=args.column_chunk,
                     pool_limit_bytes=args.pool_limit_bytes)
    out = bisect(args)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
