"""Actual case(b) output history/checkpoint transport; no routing physics replay.

The Rust capture-troute-netcdf tool creates the request and raw snapshot.
Native Q/V/D/nudge words pass through CUDA copy stages and the production
history/writer. This referee helper is not a product hydrology implementation.
"""
from pathlib import Path
from types import MappingProxyType
import argparse
import hashlib
import json
import time
import cupy as cp
import numpy as np
from gpuwm.hydro.driver import ResidentHydroDriver, Stage
from gpuwm.hydro.output import History, prepare, _call


class Source:
    def __init__(self, plan, snapshot):
        _, raw = _call({"operation": "raw-read", "path": str(snapshot)})
        n, frames = plan["reaches"], plan["frames"]
        arrays = {"reach_id": cp.asarray(np.ndarray((n,), dtype="<i8", buffer=raw,
                                                   offset=plan["ids_offset"])),
                  "step": cp.zeros((), dtype=cp.uint64)}
        keys = ("flow", "velocity", "depth", "nudge")
        for key in keys:
            arrays[key + "_input"] = cp.asarray(np.ndarray((frames, n), dtype="<f4",
                buffer=raw, offset=plan[key + "_offset"]))
            arrays[key] = cp.zeros(n, dtype=cp.float32)
        self.arrays = MappingProxyType(arrays)
        self.restart_identity = {"source": "actual native LowerColorado output",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "copy transport, no routing physics"}
        module = cp.RawModule(code=r'''
extern "C" __global__ void copy_frame(
 const float* qi,const float* vi,const float* di,const float* ni,
 float* q,float* v,float* d,float* n,const unsigned long long* step,
 unsigned long long count) {
 const unsigned long long r=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(r<count) {const unsigned long long i=*step*count+r;
 q[r]=qi[i];v[r]=vi[i];d[r]=di[i];n[r]=ni[i];}
}
extern "C" __global__ void tick(unsigned long long* step) {++*step;}
''')
        self.copy, self.tick = module.get_function("copy_frame"), module.get_function("tick")
        self.args = tuple(arrays[k + "_input"] for k in keys) + tuple(arrays[k] for k in keys)
        self.args += (arrays["step"], np.uint64(n))
        self.tick_args = (arrays["step"],)
        self.grid = ((n + 255) // 256,)

    def advance(self, dt_seconds, *, stream=None):
        self.copy(self.grid, (256,), self.args, stream=stream)
        self.tick((1,), (1,), self.tick_args, stream=stream)

    def reach_output(self):
        return {"reach_id": self.arrays["reach_id"],
                "discharge_m3_s": self.arrays["flow"],
                "velocity_m_s": self.arrays["velocity"], "depth_m": self.arrays["depth"],
                "nudge_m3_s": self.arrays["nudge"]}


def build(plan, snapshot, outdir):
    source = Source(plan, snapshot)
    interval = plan["interval_seconds"]
    settings = {"interval_seconds": interval}
    history = History(settings=settings, options={"capacity_steps": plan["frames"],
        "cadence_steps": 1, "origin": plan["origin"], "types": plan["types"],
        "formats": ["troute_netcdf"]}, source=source, device_id=0)
    driver = ResidentHydroDriver(device_id=0, interval_seconds=interval,
        identity={"scope": "actual case(b) output transport only"}, after=(
            Stage("source", source, source.advance), Stage("history", history, history.advance)),
        reach_output=source.reach_output())
    return driver, history, prepare(settings=settings, driver=driver, outdir=outdir)


def advance(driver, count):
    stream = cp.cuda.get_current_stream()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    began = time.perf_counter()
    for _ in range(count):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)
    end.record(stream)
    end.synchronize()
    return {"steps": count, "completed_ms": (time.perf_counter()-began)*1000,
            "gpu_sequence_ms": cp.cuda.get_elapsed_time(start, end)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.capture / "request.json").read_text())["plan"]
    snapshot = args.capture / "snapshot.raw"
    count = plan["frames"]
    assert count > 1 and count % 2 == 0
    began = time.perf_counter()
    full, history, writer = build(plan, snapshot, args.out / "continuous")
    setup_ms = (time.perf_counter()-began)*1000
    receipt = {"scope": "actual native case(b) output transport; no routing physics validation",
               "setup_ms": setup_ms, "continuous": advance(full, count)}
    writer.finish(full, receipt)
    split, _, first = build(plan, snapshot, args.out / "half")
    advance(split, count // 2)
    first.finish(split, {})
    resumed, _, second = build(plan, snapshot, args.out / "resumed")
    began = time.perf_counter()
    second.restore(args.out / "half/restart.nc", resumed)
    restore_ms = (time.perf_counter()-began)*1000
    advance(resumed, count // 2)
    second.finish(resumed, {})
    before, raw = _call({"operation": "checkpoint-read", "path": str(args.out / "continuous/restart.nc")})
    after, other = _call({"operation": "checkpoint-read", "path": str(args.out / "resumed/restart.nc")})
    assert before["metadata"] == after["metadata"] and before["arrays"] == after["arrays"]
    assert raw == other
    history.arrays["step"].fill(0)
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(count):
        history.advance(plan["interval_seconds"])
    end.record()
    end.synchronize()
    receipt.update({"restore_ms": restore_ms, "arrays": len(before["arrays"]),
        "payload_bytes": len(raw), "different_payload_bytes": 0,
        "history_capture_gpu_ms_per_step": cp.cuda.get_elapsed_time(start, end)/count,
        "features": plan["reaches"], "frames": count,
        "parallelism": f"RTX4090 pool; {history._grid[0]} blocks x256 capture threads, scalar cursor; two source copy launches",
        "graphs": False})
    (args.out / "transport-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
