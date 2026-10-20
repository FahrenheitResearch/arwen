"""Small two-host CUDA shard check. Synthetic inputs; no hydro physics proof.

Run worker 5 and worker 9 on separate pool cards, copy their member directories
under one root, then run reduce on a pool card. Python handles metadata and
buffer transport; comparisons/reduction use CUDA and file IO uses Rust.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cupy as cp

from gpuwm.ensemble.nextgen_mass import MemberReachCapture, reduce_members
from gpuwm.ensemble.nextgen_points import ReachSample
from gpuwm.ensemble.restart_roster import atomic_json
from gpuwm.io.classic_product import ClassicProduct
from gpuwm.netcdf_bridge import Dataset

TIME = "2024-01-01T00:00:00Z"
OPTIONS = {"thresholds_m3_s": [1., 2.]}


def worker(root, member):
    directory = root / "members" / f"member-{member}"
    directory.mkdir(parents=True, exist_ok=False)
    ids = cp.asarray([2**60+3, 7, 42, 99], dtype=cp.int64)
    q = cp.asarray([1., 2., float("nan"), 0.] if member == 5 else [1., 0., 3., 2.], dtype=cp.float64)
    if member == 5:
        q[0] = cp.nextafter(q[0], cp.float64(0))
    depth = cp.asarray([.1, .2, float("nan"), 0.] if member == 5 else [.3, 0., .5, .6], dtype=cp.float64)
    sample = ReachSample(ids, q, depth)
    driver = SimpleNamespace(device_id=0, arrays={"ids": ids, "q": q, "depth": depth},
                             reach_output=lambda: sample)
    state = SimpleNamespace(physics=SimpleNamespace(hydro=driver))
    sink_calls = []
    def sink(**row):
        # Identity check is metadata, not a numerical Q-to-depth conversion.
        assert row["channel_depth_m"] is depth
        sink_calls.append(row["member_id"])
    capture = MemberReachCapture(lambda **kw: None, member_id=member, directory=directory,
                                 options=OPTIONS, array_module=cp, depth_sink=sink)
    cp.cuda.get_current_stream().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    started = time.perf_counter()
    start.record()
    capture(state=state, member_id=member, valid_time=TIME, grid_id=1)
    end.record()
    document = capture.seal()
    end.synchronize()
    elapsed = time.perf_counter()-started
    with ClassicProduct(directory / "member-sample.nc") as file:
        file.setncattr("evidence_scope", "synthetic transport check, no hydrology advanced")
        file.setncattr("member_id", member)
        file.setncattr("valid_time", TIME)
        file.createDimension("feature", 4)
        file.createDimension("word", 2)
        for name, value, dims, units in (("reach_id_words", ids.view(cp.uint32).reshape(-1, 2), ("feature", "word"), "1"),
                ("discharge_m3_s", q, ("feature",), "m3 s-1"), ("channel_depth_m", depth, ("feature",), "m")):
            variable = file.createVariable(name, value.dtype, dims)
            variable.setncattr("units", units)
            variable[:] = cp.asnumpy(value)
    assert sink_calls == [member]
    row = {"member_id": member, "status": "completed", "nextgen_points": document,
        "nextgen_directory": str(directory.relative_to(root)),
        "evidence_scope": "synthetic two-host transport; no hydro run",
        "card": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "capture_cuda_ms": cp.cuda.get_elapsed_time(start, end),
        "capture_and_rust_checkpoint_seconds": elapsed, "cuda_graphs": False,
        "timing_scope": "first output deadline incl allocator/JIT/context effects; not isolated kernel"}
    atomic_json(directory / "member.json", row)
    print(json.dumps(row, sort_keys=True), flush=True)


def reduce(root):
    rows = [json.loads((root / "members" / f"member-{m}" / "member.json").read_text()) for m in (5, 9)]
    started = time.perf_counter()
    result = reduce_members(root, rows, expected_member_ids=(5, 9), options=OPTIONS, array_module=cp)
    seconds = time.perf_counter()-started
    with Dataset(result["products"][0]) as file:
        file.set_auto_maskandscale(False)
        counts = cp.asarray(file.variables["exceedance_count"][:], dtype=cp.uint32)
        finite = cp.asarray(file.variables["finite_count"][:], dtype=cp.uint32)
        probability = cp.asarray(file.variables["probability"][:], dtype=cp.float32)
        ids = cp.asarray(file.variables["feature_id_words"][:], dtype=cp.uint32).reshape(-1).view(cp.int64)
    assert bool(cp.array_equal(counts, cp.asarray([[1, 1, 1, 1], [0, 1, 1, 1]], dtype=cp.uint32)))
    assert bool(cp.array_equal(finite, cp.asarray([2, 2, 1, 2], dtype=cp.uint32)))
    assert bool(cp.array_equal(probability, cp.asarray([[.5, .5, float("nan"), .5], [0., .5, float("nan"), .5]], dtype=cp.float32), equal_nan=True))
    assert bool(cp.array_equal(ids, cp.asarray([2**60+3, 7, 42, 99], dtype=cp.int64)))
    document = json.loads(Path(result["products"][-1]).read_text())
    assert document["frames"][0]["requested_member_ids"] == [5, 9]
    atomic_json(root / "transport-result.json", {"pass": True, "scope": "synthetic physical two-host integer shards; no hydro physics proof",
        "cards": [row["card"] for row in rows], "worker_receipts": rows,
        "rust_read_gpu_merge_rust_write_seconds": seconds, "reduction": result,
        "exact_id_check": "all int64 bits incl 2^60+3", "data_path": "CUDA + Rust NetCDF",
        "parallelism": "independent host/card workers; parent one CUDA thread/reach, 256/block",
        "cuda_graphs": False})
    print("PASS two-host integer shards, native f64 threshold edge, nonfinite mask, exact ID words", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "reduce"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--member-id", type=int, choices=(5, 9))
    args = parser.parse_args()
    if args.mode == "worker":
        if args.member_id is None:
            parser.error("worker requires --member-id")
        worker(args.root, args.member_id)
    else:
        reduce(args.root)
