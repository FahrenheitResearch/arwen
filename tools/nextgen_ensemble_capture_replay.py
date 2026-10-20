"""Replay retained member publication vectors through the current products ABI.

Original Apache-2.0 orchestration. This is an offline output-boundary tool,
not a forecast, driver factory, numerical hydrology replay or spread claim.
NetCDF decoding/writing uses Rust; vector operations and reduction use CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def replay(args):
    import cupy as cp
    from gpuwm.netcdf_bridge import Dataset
    from gpuwm.io.classic_product import ClassicProduct
    from gpuwm.ensemble.nextgen_points import ReachAccumulator, ReachSample

    members = []
    for spec in args.member:
        member, source = spec.split("=", 1)
        members.append((int(member), Path(source).resolve()))
    accumulator = ReachAccumulator([member for member, _ in members],
        options={"thresholds_m3_s": args.threshold}, array_module=cp)
    args.out.mkdir(parents=True, exist_ok=False)
    samples, timings, vectors = [], [], []
    for member, source in members:
        started = time.perf_counter()
        stream = cp.cuda.Stream(non_blocking=True)
        with Dataset(source) as reader, stream:
            reader.set_auto_maskandscale(False)
            # Rust's decoded-double transport preserves these uint32 words;
            # reconstructing IDs on the GPU avoids int64 precision loss.
            ids = cp.asarray(reader.variables["reach_id_words"][:], dtype=cp.uint32).reshape(-1).view(cp.int64)
            variable = reader.variables["discharge_m3_s"]
            if variable.dtype.kind != "f" or variable.dtype.itemsize not in (4, 8):
                raise ValueError("captured discharge must have native float32/64 storage")
            q = cp.asarray(variable[args.record], dtype=variable.dtype)
            depth = None
            if "channel_depth_m" in reader.variables:
                depth_variable = reader.variables["channel_depth_m"]
                if depth_variable.ndim != 1:
                    raise ValueError("channel depth capture must be an explicit output-deadline vector")
                depth_ids = cp.asarray(reader.variables["channel_reach_id_words"][:], dtype=cp.uint32).reshape(-1).view(cp.int64)
                depth = cp.asarray(depth_variable[:], dtype=depth_variable.dtype)
        stream.synchronize()  # offline input/setup boundary
        loaded = time.perf_counter()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        begin.record(stream)
        accumulator.accept(member_id=member, valid_time=args.valid_time,
            sample=ReachSample(ids, q, stream=stream))
        end.record(stream)
        end.synchronize()  # offline measurement, never a routing-step hook
        captured = time.perf_counter()
        snapshot = args.out / f"member-{member}.nc"
        with ClassicProduct(snapshot) as writer:
            writer.setncattr("member_id", member)
            writer.setncattr("valid_time", args.valid_time)
            writer.setncattr("source_scope", args.source_scope)
            writer.createDimension("feature", ids.size)
            writer.createDimension("word", 2)
            fields = [("reach_id", ids, ("feature",), "1"),
                ("reach_id_words", ids.view(cp.uint32).reshape(-1, 2), ("feature", "word"), "1"),
                ("discharge_m3_s", q, ("feature",), "m3 s-1")]
            if depth is not None:
                writer.createDimension("channel", depth_ids.size)
                fields.extend([("channel_reach_id", depth_ids, ("channel",), "1"),
                    ("channel_reach_id_words", depth_ids.view(cp.uint32).reshape(-1, 2), ("channel", "word"), "1"),
                    ("channel_depth_m", depth, ("channel",), "m")])
            for name, value, dims, units in fields:
                v = writer.createVariable(name, value.dtype, dims)
                v.setncattr("units", units)
                v[:] = cp.asnumpy(value)
        def buffer(variable, value, units):
            return dict(path=snapshot.name, variable=variable, dtype=value.dtype.str,
                shape=list(value.shape), units=units)
        sample = dict(member_id=member, valid_time=args.valid_time,
            reach_id=buffer("reach_id", ids, "1"),
            discharge_m3_s=buffer("discharge_m3_s", q, "m3 s-1"),
            channel_depth_m=None, device=int(ids.device.id),
            stream="borrowed producer stream during replay; file snapshot after completion",
            source=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), record=args.record),
            sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        if depth is not None:
            sample["channel_output"] = dict(reach_id=buffer("channel_reach_id", depth_ids, "1"),
                channel_depth_m=buffer("channel_depth_m", depth, "m"), depth_reference="local_channel_bed",
                record="source stores final output-deadline channel vector; independent of discharge record selector")
        samples.append(sample)
        vectors.append((ids, q))
        timings.append(dict(member_id=member, rust_read_upload_ms=(loaded-started)*1000,
            capture_cuda_ms=cp.cuda.get_elapsed_time(begin, end), capture_wall_ms=(captured-loaded)*1000,
            sample_rust_write_ms=(time.perf_counter()-captured)*1000))
    started = time.perf_counter()
    accumulator.finish(args.out / "nextgen")
    finish_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    published = accumulator.published_buffers()
    published["manifest"] = "nextgen/reach-ensemble.json"
    published["base_directory"] = "nextgen"
    # Independent vector-expression check of this actual captured frame,
    # including zero discharge. No member is selected/dropped by wetness.
    with Dataset(args.out / "nextgen/reach-exceedance-00000.nc") as reader:
        reader.set_auto_maskandscale(False)
        actual_count = cp.asarray(reader.variables["exceedance_count"][:], dtype=cp.uint32)
        actual_finite = cp.asarray(reader.variables["finite_count"][:], dtype=cp.uint32)
        actual_probability = cp.asarray(reader.variables["probability"][:], dtype=cp.float32)
        thresholds = cp.asarray(reader.variables["discharge_threshold"][:], dtype=cp.float32)
    expected_count, expected_finite = cp.zeros_like(actual_count), cp.zeros_like(actual_finite)
    for ids, q in vectors:
        finite = cp.isfinite(q)
        expected_finite += finite.astype(cp.uint32)
        expected_count += (finite[None, :] & (q[None, :].astype(cp.float64) >= thresholds[:, None])).astype(cp.uint32)
    expected_probability = cp.where(expected_finite[None, :] == len(members),
        (expected_count.astype(cp.float64) / len(members)).astype(cp.float32), cp.nan)
    differences = dict(count=int(cp.count_nonzero(actual_count != expected_count).item()),
        finite=int(cp.count_nonzero(actual_finite != expected_finite).item()),
        probability=int(cp.count_nonzero(actual_probability.view(cp.uint32) != expected_probability.view(cp.uint32)).item()))
    if any(differences.values()):
        raise AssertionError(differences)
    result = dict(schema="arwen.nextgen.captured-products-handoff/v1", source_scope=args.source_scope,
        requested_member_ids=list(accumulator.members), valid_time=args.valid_time, samples=samples,
        reduced=published, reduction=accumulator.receipt(), timings=timings, differences=differences,
        dry_reach_counts=[dict(member_id=member, count=int(cp.count_nonzero(q == 0).item()))
            for (member, _), (_, q) in zip(members, vectors)],
        finish_wall_ms=finish_ms, validation_wall_ms=(time.perf_counter()-started)*1000,
        parallelism="CUDA 256 threads/block, one thread/reach, event-ordered members; Rust boundary I/O",
        cuda_graphs=False, routing_step_executed=False, forecast_executed=False,
        native_physics_parity_claim=False, card=cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    (args.out / "handoff.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(handoff=str(args.out / "handoff.json"), samples=len(samples),
        finish_wall_ms=result["finish_wall_ms"], card=result["card"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", required=True, metavar="ID=NETCDF")
    parser.add_argument("--record", type=int, required=True)
    parser.add_argument("--valid-time", required=True)
    parser.add_argument("--threshold", action="append", type=float, required=True)
    parser.add_argument("--source-scope", required=True)
    parser.add_argument("--out", type=Path, required=True)
    replay(parser.parse_args())
