"""Where the time in a radiation call goes, per chunk and per stage.

WHY THIS EXISTS.  The reclamation matrix measured RRTMGP at four column
chunks on one domain and the per-CALL cost tracked the number of chunks, not
the number of columns: 16,384 columns cost 825 ms in 6 chunks of 3125 and
7,360 ms in 64 chunks of 256 -- 137 ms and 115 ms per chunk respectively.  A
cost that is flat in chunk width is not the solver doing arithmetic on
columns; something is being paid once per chunk regardless of how much work
the chunk carries.  That distinction decides whether chunking is a cheap VRAM
knob or an expensive one, so it is measured rather than inferred.

METHOD.  Every stage function the chunk loop calls is wrapped with a
device-synchronised timer.  Synchronising each stage serialises what CUDA
would otherwise overlap, so the STAGE SHARES are the trustworthy output and
the wrapped total is an upper bound on the real call -- reported alongside
the unwrapped call time so the inflation is visible rather than hidden.
"""

from __future__ import annotations

import argparse
import collections
import json
import time


STAGES = (
    "_prepare_above_model_chunk",
    "_gas_optics",
    "_cloud_optics",
    "_mcica_cloud_masks",
    "_finalize_cloud_optics",
    "_planck_sources",
    "_expand_band_to_gpoint",
    "_lw_rte",
    "_sw_rte",
    "cal_cldfra1",
    "hydrometeor_paths",
    "_interpolation_metadata",
    "_validate_device_call",
    "_interface_temperatures",
    "_fluxes_to_radiation",
    "_surface_emissivity_bands",
)


def instrument(module, names=STAGES):
    """Wrap ``names`` on ``module`` with synchronised timers.  Returns totals."""
    import cupy as cp

    totals: dict[str, float] = collections.defaultdict(float)
    calls: dict[str, int] = collections.defaultdict(int)

    def wrap(name, fn):
        def timed(*a, **kw):
            cp.cuda.runtime.deviceSynchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            cp.cuda.runtime.deviceSynchronize()
            totals[name] += (time.perf_counter() - t) * 1e3
            calls[name] += 1
            return out
        timed.__name__ = getattr(fn, "__name__", name)
        return timed

    for name in names:
        fn = getattr(module, name, None)
        if fn is None:
            continue
        setattr(module, name, wrap(name, fn))
    return totals, calls


def profile(*, nx: int, ny: int, nz: int, column_chunk: int,
            rung: str = "full") -> dict:
    import cupy as cp

    from gpuwm.core import rrtmgp as rr
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    from tilestream import harness, physics_inventory as physinv
    from tilestream.rrtmgp_bench import RUNGS, _force_radiation_due
    from tilestream.rrtmgp_lazy import attach_lazy

    cfg = harness.make_config(nx, ny, nz, **RUNGS[rung])
    state, driver = physinv.default_builder(cfg)
    ncol = nx * ny
    chunk = max(1, min(int(column_chunk), ncol))
    workspace = SharedRRTMGPChunkWorkspace(
        nz=nz, column_chunk=chunk, p_top=float(state.p_top))
    attach_lazy(state, workspace)

    # Warm every kernel and cache before anything is timed.
    harness.run_steps(state, cfg, 2)
    for _ in range(2):
        _force_radiation_due(state, cfg, driver)
        harness.run_steps(state, cfg, 1)
    cp.cuda.runtime.deviceSynchronize()

    # Unwrapped reference call.
    _force_radiation_due(state, cfg, driver)
    cp.cuda.runtime.deviceSynchronize()
    t = time.perf_counter()
    harness.run_steps(state, cfg, 1)
    cp.cuda.runtime.deviceSynchronize()
    plain_ms = (time.perf_counter() - t) * 1e3

    totals, calls = instrument(rr)
    _force_radiation_due(state, cfg, driver)
    cp.cuda.runtime.deviceSynchronize()
    t = time.perf_counter()
    harness.run_steps(state, cfg, 1)
    cp.cuda.runtime.deviceSynchronize()
    wrapped_ms = (time.perf_counter() - t) * 1e3

    nchunks = -(-ncol // chunk)
    return {
        "nx": nx, "ny": ny, "nz": nz, "ncol": ncol, "column_chunk": chunk,
        "nchunks": nchunks,
        "plain_step_ms": plain_ms, "wrapped_step_ms": wrapped_ms,
        "stage_ms": dict(totals), "stage_calls": dict(calls),
        "accounted_ms": sum(totals.values()),
        "digest": harness.hash_state(state),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=49)
    p.add_argument("--column-chunk", type=int, default=3125)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    res = profile(nx=args.nx, ny=args.ny, nz=args.nz,
                  column_chunk=args.column_chunk)
    print(f"{res['nx']}x{res['ny']}x{res['nz']}  ncol={res['ncol']:,}  "
          f"chunk={res['column_chunk']}  nchunks={res['nchunks']}")
    print(f"  step (unwrapped)     {res['plain_step_ms']:9.1f} ms")
    print(f"  step (wrapped)       {res['wrapped_step_ms']:9.1f} ms  "
          f"(sync inflation "
          f"{res['wrapped_step_ms'] - res['plain_step_ms']:+.1f})")
    print(f"  accounted by stages  {res['accounted_ms']:9.1f} ms")
    print()
    print(f"  {'stage':<28}{'ms':>10}{'calls':>8}{'ms/call':>10}"
          f"{'ms/chunk':>10}   % of call")
    print("  " + "-" * 76)
    for name, ms in sorted(res["stage_ms"].items(), key=lambda kv: -kv[1]):
        n = res["stage_calls"][name]
        print(f"  {name:<28}{ms:>10.1f}{n:>8}{ms / n:>10.2f}"
              f"{ms / res['nchunks']:>10.2f}"
              f"{100.0 * ms / res['plain_step_ms']:>10.1f}%")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
