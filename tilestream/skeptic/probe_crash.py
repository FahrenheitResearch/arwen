"""Isolate the illegal memory access the fast-cadence lane hit under replay.

``bench_graph --rung "full fast cadence" --mode step --windows 128,...``
died with ``cudaErrorIllegalAddress`` at the FIRST capture-and-launch, on the
one rung where radiation is due on every step.  The gate never sees this: it
runs fast cadence only at 48x40 tiles, and the whole-step benchmarks that
were quoted ran the ship config, where radiation fires zero times.

Three things have to be separated before this means anything:

  * is it the GRAPH, or does the same window fault on the ordinary stream
    path too (a pre-existing bug the graph lane merely reached first)?
  * is it the WINDOW, i.e. does it appear only above some size?
  * is it reproducible, or was it a one-off on a shared card?

Each configuration runs in its own process (--one) so a fault cannot poison
the next measurement; the driver process reports the exit status.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))


def one(rung: str, window: int, mode: str, steps: int) -> int:
    import cupy as cp

    from tilestream import graphcap, harness, physics_inventory
    from tilestream.test_gate import PHYSICS_RUNGS
    from gpuwm.core.dycore import step

    cfg = harness.make_config(window, window, 49, **PHYSICS_RUNGS[rung])
    state, _drv = physics_inventory.default_builder(cfg, harness.DEFAULT_SEED)
    harness.run_steps(state, cfg, 1)
    print(f"  built {rung} at {window}^2, mode={mode}", flush=True)

    if mode == "stream":
        for i in range(steps):
            step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        print(f"  stream path OK over {steps} steps", flush=True)
        return 0

    s = cp.cuda.Stream(non_blocking=True)
    stepper = graphcap.GraphStepper(
        cfg, mode="require", reuse=("run" if mode == "graph_run" else "sweep"),
        scalars_fn=physics_inventory.carrier_scalars,
        set_scalars_fn=physics_inventory.set_carrier_scalars)
    with s:
        kind = stepper.run(state, s)
    s.synchronize()
    print(f"  first {kind} OK, {next(iter(stepper.graphs.values())).nodes} "
          f"nodes", flush=True)
    for i in range(steps):
        with s:
            kind = stepper.run(state, s)
        s.synchronize()
    cp.cuda.runtime.deviceSynchronize()
    print(f"  {mode} OK over {steps} more steps", flush=True)
    return 0


def main() -> int:
    if "--one" in sys.argv:
        i = sys.argv.index("--one")
        rung, window, mode, steps = (sys.argv[i + 1], int(sys.argv[i + 2]),
                                     sys.argv[i + 3], int(sys.argv[i + 4]))
        return one(rung, window, mode, steps)

    cases = []
    for rung in ("full fast cadence", "full(real74) +KF"):
        for window in (96, 128, 160, 224):
            for mode in ("stream", "graph_run", "graph_sweep"):
                cases.append((rung, window, mode))
    for rung, window, mode in cases:
        tag = f"{rung:20s} w={window:4d} {mode:12s}"
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--one",
             rung, str(window), mode, "6"],
            cwd=HERE, capture_output=True, text=True, timeout=1800,
            env=dict(os.environ, PYTHONPATH=HERE))
        if p.returncode == 0:
            print(f"{tag} OK", flush=True)
        else:
            err = [ln for ln in p.stderr.splitlines()
                   if "Error" in ln or "error" in ln]
            print(f"{tag} EXIT {p.returncode}  {err[-1][:120] if err else ''}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
