"""Do 11 of the gate's 14 physics rungs really capture?

The report's central enabling claim is a census: after the health ledger and
the memoised constant tables, ELEVEN of the fourteen rungs capture as a CUDA
graph and three (full+MYNN, full+Noah-MP, full+MYNN+Noah-MP) fall back.  That
number is what justifies the whole workstream, so it is worth counting rather
than accepting.  Each rung runs in its own process because a refused capture
can leave the CUDA context unhappy and a shared process would let one rung's
failure contaminate the next one's verdict.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def one(rung: str, window: int) -> int:
    import cupy as cp

    from tilestream import graphcap, harness, physics_inventory
    from tilestream.test_gate import PHYSICS_RUNGS

    cfg = harness.make_config(window, window, 49, **PHYSICS_RUNGS[rung])
    if rung == "dry (control)":
        state = harness.make_state(cfg)
    else:
        state, _drv = physics_inventory.default_builder(
            cfg, harness.DEFAULT_SEED)
    harness.run_steps(state, cfg, 1)
    s = cp.cuda.Stream(non_blocking=True)
    stepper = graphcap.GraphStepper(
        cfg, mode="require", reuse="sweep",
        scalars_fn=physics_inventory.carrier_scalars,
        set_scalars_fn=physics_inventory.set_carrier_scalars)
    try:
        with s:
            stepper.run(state, s)
        s.synchronize()
    except Exception as exc:                                # noqa: BLE001
        print(f"REFUSED {type(exc).__name__}: {str(exc)[:150]}")
        return 3
    nodes = next(iter(stepper.graphs.values())).nodes
    print(f"CAPTURED {nodes} nodes")
    return 0


def main() -> int:
    if "--one" in sys.argv:
        i = sys.argv.index("--one")
        return one(sys.argv[i + 1], int(sys.argv[i + 2]))

    from tilestream.test_gate import PHYSICS_RUNGS

    ok = bad = 0
    for rung in PHYSICS_RUNGS:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--one", rung, "64"],
            cwd=HERE, capture_output=True, text=True, timeout=1800,
            env=dict(os.environ, PYTHONPATH=HERE))
        line = next((l for l in p.stdout.splitlines()
                     if l.startswith(("CAPTURED", "REFUSED"))), None)
        if line is None:
            line = f"CRASHED rc={p.returncode} " + " ".join(
                l for l in p.stderr.splitlines()[-1:])
        ok += line.startswith("CAPTURED")
        bad += not line.startswith("CAPTURED")
        print(f"{rung:22s} {line[:130]}", flush=True)
    print(f"\n{ok} of {ok + bad} rungs capture; {bad} do not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
