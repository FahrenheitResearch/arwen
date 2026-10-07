"""Run the DA cycle controller with every device analysis solved by two routes.

At the analysis seam (``radar_assimilation._execute_analysis``) each
ensemble-transform analysis is solved by the PRODUCTION route the cycle then
continues with, and again by a SHADOW route on the very same inputs.  The
per-field difference between the two increments (max, RMS, beside the
increment RMS), both routes' wall seconds and filter diagnostics are written
as one JSON per analysis call.  The cycle's own outputs are the production
route's, so a run under this wrapper is byte-comparable with a production run
of that route.  Measurement tooling only.

    python -m tools.da_letkf_route_shadow --record DIR \
        --production host-staged-cuda --shadow cuda-obs-sparse \
        -- <tools.da_cycle_prepared arguments>
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path


def install(record: Path, production: str, shadow: str | None) -> None:
    import numpy as np

    from gpuwm.da import letkf
    from gpuwm.da import radar_assimilation as ra
    from tools.da_letkf_device_ab import compare

    record.mkdir(parents=True, exist_ok=True)
    original = ra._execute_analysis
    calls = {"n": 0}
    started = time.time()

    def rusage():
        import resource
        me = resource.getrusage(resource.RUSAGE_SELF)
        kids = resource.getrusage(resource.RUSAGE_CHILDREN)
        (record / "rusage.json").write_text(json.dumps({
            "controller_max_rss_gib": me.ru_maxrss / 2**20,
            "children_max_rss_gib": kids.ru_maxrss / 2**20,
            "controller_cpu_seconds": me.ru_utime + me.ru_stime,
            "wall_seconds": time.time() - started}, indent=1) + "\n")

    import atexit
    atexit.register(rusage)

    def attempt(storage, solver, prior, batches, geometry, config,
                namespace, progress, device_options=None):
        import cupy as cp

        diag = letkf.LetkfDiagnostics()
        cp.get_default_memory_pool().free_all_blocks()
        started = time.perf_counter()
        inc, diag, stage, unstage = ra._analysis_attempt(
            solver, prior, batches, geometry, config, namespace=namespace,
            storage=storage, supports_staging=True, progress=progress,
            diagnostics=diag, **({} if device_options is None
                                 else {"device_options": device_options}))
        return inc, diag, stage, unstage, time.perf_counter() - started

    def wrapped(solver, prior, batches, geometry, config, *, namespace,
                device, progress=None, diagnostics=None, device_options=None):
        # device_options (cards, in-filter withhold, positivity hook) go to
        # the production attempt as the seam passes them.  The shadow gets
        # them too; on a route other than cuda-obs-sparse they are ignored,
        # so its increments are then the bare filter's and its difference
        # row compares a different stage of the analysis.
        if solver is not letkf.analyze or device != "cuda":
            return original(solver, prior, batches, geometry, config,
                            namespace=namespace, device=device,
                            progress=progress, diagnostics=diagnostics,
                            **({} if device_options is None
                               else {"device_options": device_options}))
        calls["n"] += 1
        n = calls["n"]
        inc, diag, stage, unstage, wall = attempt(
            production, solver, prior, batches, geometry, config, namespace,
            progress, device_options)
        row = {"schema": "gpuwm-da.route-shadow.v1", "call": n,
               "fields": list(config.analysis_fields),
               "batches": [b.name for b in batches],
               "production": {"route": production, "wall_seconds": wall,
                              "diagnostics": _plain(diag)}}
        print(f"route-shadow call {n}: {production} {wall:.1f} s", flush=True)
        if shadow is not None:
            sinc, sdiag, _s, _u, swall = attempt(
                shadow, solver, prior, batches, geometry, config, namespace,
                None, device_options)
            row["shadow"] = {"route": shadow, "wall_seconds": swall,
                             "diagnostics": _plain(sdiag)}
            row["difference"] = compare(sinc, inc)
            print(f"route-shadow call {n}: {shadow} {swall:.1f} s", flush=True)
            if n == 1:
                np.savez(record / "call-001-member0.npz",
                         **{f"{shadow}:{k}": np.asarray(v[0]) for k, v in sinc.items()},
                         **{f"{production}:{k}": np.asarray(v[0]) for k, v in inc.items()})
            del sinc
        (record / f"call-{n:03d}.json").write_text(
            json.dumps(row, indent=1, default=str) + "\n")
        if diagnostics is not None:
            vars(diagnostics).update(vars(diag))
        attempts = [dict(attempt=1, storage=production, status="computed",
                         wall_seconds=wall, committed=False)]
        return inc, diag, stage, unstage, production, attempts

    ra._execute_analysis = wrapped


def _plain(diag) -> dict:
    return {k: v for k, v in vars(diag).items()
            if isinstance(v, (int, float, str, bool, type(None), dict))}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--production", default="host-staged-cuda")
    parser.add_argument("--shadow", default="cuda-obs-sparse")
    parser.add_argument("--no-shadow", action="store_true")
    args = parser.parse_args(argv[:split])
    install(args.record, args.production,
            None if args.no_shadow else args.shadow)
    sys.argv = ["tools.da_cycle_prepared", *argv[split + 1:]]
    runpy.run_module("tools.da_cycle_prepared", run_name="__main__",
                     alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
