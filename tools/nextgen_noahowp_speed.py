"""Profile a repeated native Noah-OWP column; this is a capacity diagnostic.

Rust reads the sealed slab bytes. CUDA expands them across catchments. The
ordinary full update is timed without per-step instrumentation. Five stage
codes are profiled only on the first interval, then checked against the full
kernel on all stored words. This does not validate spatial case inputs.
"""
from __future__ import annotations

import argparse
import ctypes
from functools import partial
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--rust-io", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--catchments", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.catchments < 1:
        parser.error("catchments must be positive")
    from gpuwm import numerics
    numerics.select("fast")
    numerics.arm()
    import cupy as cp
    import numpy as np
    from gpuwm.hydro.noahowp.runtime import PreparedNoahOWP
    from gpuwm.nextgen_execution import ResidentSequence, Stage, measure

    index = json.loads((args.fixture / "index.json").read_text())
    specs = index["calls"][0]["device_slabs"]["arrays"]
    source = json.loads(args.source_identity.read_text())
    io = ctypes.CDLL(str(args.rust_io))
    io.read_into.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    io.read_into.restype = ctypes.c_int
    seed = {}
    for key, spec in specs.items():
        if spec["shape"][1] != 1:
            raise ValueError("diagnostic requires one captured native column")
        dtype = {"f32": np.float32, "i32": np.int32, "f64": np.float64}[key]
        host = np.empty(tuple(spec["shape"]), dtype=dtype)
        status = io.read_into(os.fsencode(args.fixture / spec["path"]), host.ctypes.data, host.nbytes)
        if status:
            raise ValueError(f"Rust slab read failed: {key} status={status}")
        seed[key] = cp.asarray(host)
    dt = float(seed["f32"][0, 0].item())  # one setup-only clock read
    parallelism = f"GPU {(args.catchments+63)//64} blocks x64 threads; one catchment/thread"

    def fresh_slabs():
        return {key: cp.repeat(value, args.catchments, axis=1) for key, value in seed.items()}

    def component(slabs, stage="full", advance=True):
        return PreparedNoahOWP(slabs=slabs, numerics="fast", stage=stage, advance=advance,
            parameter_sha256=specs["f32"]["sha256"], fabric_sha256="repeated-native-column-only",
            time_origin="captured native first update")

    built = []
    def factory():
        c = component(fresh_slabs())
        built.append(c)
        return ResidentSequence([Stage("noahowp-full", partial(c.launch, dt), parallelism)],
            step_seconds=dt, keepalive=c, metadata={"numerics": "fast", "catchments": args.catchments,
                "scope": "capacity only: one native Bondville column repeated; static forcing",
                "source": source, "seed": specs, "native_case": index["case"]})

    setup_start = time.perf_counter_ns()
    result = measure(factory, steps=args.steps, warmup=args.warmup)
    result["measurement_and_setup_wall_ns"] = time.perf_counter_ns() - setup_start
    result["kernel_attributes"] = built[0]._kernel.attributes
    result["batch_health"] = [{"nonzero_status": int(cp.count_nonzero(c.arrays["status"]).item()),
        "nonfinite_f32": int(cp.count_nonzero(~cp.isfinite(c.arrays["slab/f32"])).item())} for c in built]

    # A split sequence lacks the full kernel's datetime prelude. It is valid
    # here only for one fresh interval, with that same prelude value seeded.
    whole_slabs, split_slabs = fresh_slabs(), fresh_slabs()
    whole = component(whole_slabs)
    pieces = [component(split_slabs, name, advance=name == "water")
              for name in ("utilities", "forcing", "interception", "energy", "water")]
    whole.launch(dt)
    cp.copyto(split_slabs["f64"][2], whole_slabs["f64"][2])
    stream = cp.cuda.get_current_stream()
    events = [cp.cuda.Event() for _ in range(len(pieces)+1)]
    events[0].record(stream)
    for c, event in zip(pieces, events[1:]):
        c.launch(dt)
        event.record(stream)
    events[-1].synchronize()
    result["first_interval_split_profile"] = {
        "clock_prelude": "CUDA copy of full first-interval curr_datetime before the first event",
        "fresh_state": True, "warmup_steps": 0, "includes_enqueue_gaps": True,
        "stages": [{"name": c.restart_identity["stage"], "elapsed_ms": cp.cuda.get_elapsed_time(events[i], events[i+1]),
                    "parallelism": parallelism} for i,c in enumerate(pieces)],
        "slab_differing_words": {key: int(cp.count_nonzero(a.view(cp.uint32 if a.dtype.itemsize == 4 else cp.uint64)
            != split_slabs[key].view(cp.uint32 if a.dtype.itemsize == 4 else cp.uint64)).item())
            for key,a in whole_slabs.items()},
        "stage_nonzero_status": [int(cp.count_nonzero(c.arrays["status"]).item()) for c in pieces],
        "whole_nonzero_status": int(cp.count_nonzero(whole.arrays["status"]).item()),
        "scope": "diagnostic split launches; production remains one full launch, not the sum of these intervals",
    }
    result["numerics"] = numerics.receipt()
    result["command"] = sys.argv
    result["shared_pool"] = True
    result["isolated_card_speed"] = False
    result["whole_hydro_acceptance"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
