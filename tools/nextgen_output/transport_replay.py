"""Exercise output transport/resume with real shipped ngen flow words.

This replays stored output through a CUDA copy stage, not hydrologic numerics.
It never claims a native module or LowerColorado case (b) physics proof.
"""
from pathlib import Path
import argparse
import json
import time
import cupy as cp
import numpy as np
from gpuwm.hydro.driver import ResidentHydroDriver, Stage
from gpuwm.hydro.output import History, prepare, _call


class Source:
    def __init__(self, fixture):
        _, raw = _call({"operation": "raw-read", "path": str(fixture)})
        self.arrays = {"fixture": cp.asarray(np.frombuffer(raw, dtype="<f8")),
                       "step": cp.zeros((), dtype=cp.uint64),
                       "q": cp.zeros(1, dtype=cp.float64),
                       "ids": cp.asarray(np.array([11223], dtype="int64")),
                       "empty": cp.zeros(0, dtype=cp.int32)}
        self.restart_identity = {"fixture": "ngen-be11a180-shipped-nex-11223-rounded",
                                 "scope": "copy transport only; no hydrologic numerics"}
        self._kernel = cp.RawKernel(r'''
extern "C" __global__ void replay(const double* input, double* q,
                                   unsigned long long* step) {
    *q=input[*step]; ++*step;
}''', "replay")
        self._kernel.compile()
        self._args = tuple(self.arrays[key] for key in ("fixture", "q", "step"))

    def advance(self, dt_seconds, *, stream=None):
        self._kernel((1,), (1,), self._args, stream=stream)

    def reach_output(self):
        return {"reach_id": self.arrays["ids"], "discharge_m3_s": self.arrays["q"]}


def build(fixture, outdir):
    source = Source(fixture)
    settings = {"interval_seconds": 3600}
    history = History(settings=settings, options={"capacity_steps": 24,
        "cadence_steps": 1, "origin": "2015-12-01 00:00:00", "types": ["nex"],
        "nexus_ids": ["nex-11223"], "formats": ["nexus_csv", "analysis_csv"]},
        source=source, device_id=0)
    driver = ResidentHydroDriver(device_id=0, interval_seconds=3600,
        identity={"scope": "output transport replay"}, after=(
            Stage("source", source, source.advance), Stage("history", history, history.advance)),
        reach_output=source.reach_output())
    return driver, history, prepare(settings=settings, driver=driver, outdir=outdir)


def advance(driver, count):
    stream = cp.cuda.get_current_stream()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record(stream)
    begin = time.perf_counter()
    for _ in range(count):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)
    end.record(stream)
    end.synchronize()
    return {"completed_ms": (time.perf_counter()-begin)*1000,
            "gpu_sequence_ms": cp.cuda.get_elapsed_time(start, end), "steps": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    setup = time.perf_counter()
    full, history, writer = build(args.fixture, args.out / "continuous")
    setup_ms = (time.perf_counter()-setup)*1000
    full_timing = advance(full, 24)
    receipt = {"scope": "real rounded upstream flow transport; no routing/runoff numerical port",
               "setup_ms": setup_ms, "continuous": full_timing}
    writer.finish(full, receipt)
    split, _, writer_a = build(args.fixture, args.out / "half")
    advance(split, 12)
    writer_a.finish(split, {})
    resumed, _, writer_b = build(args.fixture, args.out / "resumed")
    begin = time.perf_counter()
    writer_b.restore(args.out / "half" / "restart.nc", resumed)
    restore_ms = (time.perf_counter()-begin)*1000
    advance(resumed, 12)
    writer_b.finish(resumed, {})
    before, raw = _call({"operation": "checkpoint-read", "path": str(args.out / "continuous" / "restart.nc")})
    after, other = _call({"operation": "checkpoint-read", "path": str(args.out / "resumed" / "restart.nc")})
    assert before["metadata"] == after["metadata"]
    assert before["arrays"] == after["arrays"]
    assert raw == other
    # Identity mismatch must be rejected before any device mutation.
    saved = resumed._identity
    resumed._identity = {"wrong_hydrofabric": True}
    try:
        writer_b.restore(args.out / "half" / "restart.nc", resumed)
    except ValueError:
        rejected = True
    else:
        raise AssertionError("wrong identity restored")
    finally:
        resumed._identity = saved
    writer_b.outdir = args.out / "after-rejection"
    writer_b.finish(resumed, {})
    _, unchanged = _call({"operation": "checkpoint-read", "path": str(writer_b.outdir / "restart.nc")})
    assert unchanged == other
    # Isolate capture overhead using the already compiled bound launches.
    stream = cp.cuda.get_current_stream()
    history.arrays["step"].fill(0)
    events = [cp.cuda.Event(), cp.cuda.Event()]
    events[0].record(stream)
    for _ in range(24):
        history.advance(3600, stream=stream)
    events[1].record(stream)
    events[1].synchronize()
    receipt.update({"restore_ms": restore_ms, "payload_bytes": len(raw),
                    "different_payload_bytes": 0, "arrays": len(before["arrays"]),
                    "identity_rejection_before_mutation": rejected,
                    "history_capture_gpu_ms_per_step": cp.cuda.get_elapsed_time(*events)/24,
                    "parallelism": "RTX 4090, pool; history1thread/reach +1cursor; copy-only source1thread",
                    "graphs": False})
    (args.out / "transport-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
