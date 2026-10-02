"""Time the production CUDA closure dispatch from captured forcing inputs."""
import json
import pickle
import sys
import time
from pathlib import Path

import cupy as cp

from gpuwm.ingest import closure_device, real
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend

rows = []
original = closure_device.thompson_cold_start_moment_closure


def timed(*args, **kwargs):
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    result = original(*args, **kwargs)
    cp.cuda.runtime.deviceSynchronize()
    seconds = time.perf_counter() - start
    print(f"CLOSURE seconds={seconds:.6f}", flush=True)
    rows[-1]["closure_seconds"] = seconds
    rows[-1]["receipt"] = result
    return result


closure_device.thompson_cold_start_moment_closure = timed
for filename in sys.argv[1:]:
    rec = pickle.loads(Path(filename).read_bytes())
    kw = {k: v for k, v in rec["kwargs"].items() if not k.endswith("__name")}
    kw["preprocess_backend"] = resolve_preprocess_backend("cuda")
    kw["state_backend"] = "preprocess"
    rows.append({"capture": filename})
    report = {}
    print(f"CAPTURE {filename}", flush=True)
    start = time.perf_counter()
    result = real.initialize_real(HorizontalSnapshot(**rec["snapshot"]), rec["cfg"],
                                 rec["coord"], rec["terrain"], timing_report=report, **kw)
    cp.cuda.runtime.deviceSynchronize()
    rows[-1]["initialize_seconds"] = time.perf_counter() - start
    rows[-1]["timing_report"] = report
    del result
    cp.get_default_memory_pool().free_all_blocks()
Path("closure-dispatch-time.json").write_text(json.dumps(rows, indent=2))
