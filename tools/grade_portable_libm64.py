"""Chunked host-library grading and CUDA-event throughput measurement."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cupy as cp
from gpuwm.core import portable_math as pm
from gpuwm.core.noahmp_libm import log1pf_array
from gpuwm.core.kernels import load_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--float-start", type=int, default=0)
    parser.add_argument("--float-count", type=int, default=1_000_000)
    parser.add_argument("--chunk", type=int, default=1 << 20)
    parser.add_argument("--distribution", choices=("mixed", "bits", "loguniform"), default="mixed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert pm.implementation() == pm.IMPLEMENTATION, pm.implementation()
    module = load_module("portable_libm64_grade")
    k64, k32 = module.get_function("plm_grade64"), module.get_function("plm_grade32")
    result = {"host": pm.implementation(), "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
              "frames": {k.name: k.attributes for k in (k64, k32)},
              "distribution": args.distribution, "verdicts": []}
    rng = np.random.default_rng(641)
    begin = time.monotonic()
    for op, fn in enumerate((pm.exp, pm.log, pm.log1p, pm.power)):
        count, bad, device_ms = 0, 0, 0.0
        for start in range(0, args.samples, args.chunk):
            n = min(args.chunk, args.samples - start)
            x = rng.bit_generator.random_raw(n).view(np.float64)
            y = rng.bit_generator.random_raw(n).view(np.float64)
            # The original mixed set alternates raw bits and finite bin strata.
            if args.distribution != "bits" and (start // args.chunk) & 1:
                x = np.ldexp(rng.uniform(0.5, 1.0, n), rng.integers(-1073, 1025, n))
                x[rng.integers(0, 2, n).astype(bool)] *= -1
                y = rng.uniform(-2048.0, 2048.0, n)
            if args.distribution == "loguniform":
                # Sample log2 magnitude itself, rather than a linear mantissa.
                # The open upper endpoint avoids 2**1024 overflow.
                x = np.exp2(rng.uniform(-1074.0, 1024.0, n))
                y = np.exp2(rng.uniform(-1074.0, 1024.0, n))
                x[rng.integers(0, 2, n).astype(bool)] *= -1
                y[rng.integers(0, 2, n).astype(bool)] *= -1
            dx, dy, out = cp.asarray(x), cp.asarray(y), cp.empty(n, dtype=cp.float64)
            a, b = cp.cuda.Event(), cp.cuda.Event()
            a.record(); k64(((n + 255) // 256,), (256,), (dx, dy, out, np.uint64(n), np.int32(op))); b.record(); b.synchronize()
            device_ms += cp.cuda.get_elapsed_time(a, b)
            actual = out.get().view(np.uint64)
            expected = (fn(x, y) if op == 3 else fn(x)).view(np.uint64)
            indices = np.flatnonzero(actual != expected)
            bad += indices.size; count += n
            if indices.size:
                i = int(indices[0]); print(json.dumps({"op": op, "index": start + i, "x": hex(int(x.view(np.uint64)[i])), "y": hex(int(y.view(np.uint64)[i])), "host": hex(int(expected[i])), "device": hex(int(actual[i]))}), flush=True)
                break
            if count % (args.chunk * 128) == 0: print(f"PROGRESS op={op} count={count}", flush=True)
        row = {"op": op, "count": count, "differences": int(bad), "device_ms": device_ms}
        print("VERDICT " + json.dumps(row), flush=True); result["verdicts"].append(row)
    count, bad, device_ms = 0, 0, 0.0
    for start in range(args.float_start, args.float_start + args.float_count, args.chunk):
        n = min(args.chunk, args.float_start + args.float_count - start)
        x = np.arange(start, start + n, dtype=np.uint32).view(np.float32)
        dx, out = cp.asarray(x), cp.empty(n, dtype=cp.float32)
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record(); k32(((n + 255) // 256,), (256,), (dx, out, np.uint64(n))); b.record(); b.synchronize()
        device_ms += cp.cuda.get_elapsed_time(a, b)
        actual = out.get().view(np.uint32)
        with np.errstate(all="ignore"): expected = log1pf_array(x).view(np.uint32)
        indices = np.flatnonzero(actual != expected); bad += indices.size; count += n
        if indices.size:
            i = int(indices[0]); print(json.dumps({"op": "log1pf", "x": hex(start + i), "host": hex(int(expected[i])), "device": hex(int(actual[i]))}), flush=True); break
        if count % (args.chunk * 128) == 0: print(f"PROGRESS log1pf count={count}", flush=True)
    row = {"op": "log1pf", "count": count, "differences": int(bad), "device_ms": device_ms}
    print("VERDICT " + json.dumps(row), flush=True); result["verdicts"].append(row)
    result["wall_seconds"] = time.monotonic() - begin
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if any(row["differences"] for row in result["verdicts"]): raise SystemExit(1)


if __name__ == "__main__": main()
