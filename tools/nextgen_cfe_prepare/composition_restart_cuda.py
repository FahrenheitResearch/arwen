"""Private combined CFE/transfer continuation and member isolation consumer.

The second update holds captured forcing fixed: it is a continuation self-test,
not another native comparison or a physical parent/event qualification. Rust
owns checkpoint bytes and comparisons; CUDA owns all model arithmetic.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

from composition_cuda import prepare_composition


def main():
    root = Path(sys.argv[1]).resolve()
    identity = json.loads((root / "identity.json").read_text())
    for relative, digest in identity["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    mode = os.environ["GPUWM_NUMERICS"]
    out = root / mode
    out.mkdir()
    packet = json.loads((root / "bmi/manifest.json").read_text())
    publication = json.loads((root / "publication/fixture.json").read_text())
    assert packet["step"] == identity["native_step"]
    assert publication["catchment_ids"] == packet["bmi"]["framework_witness"]["catchment_ids"]
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    from gpuwm.hydro.output import _snapshot, _call, restore_checkpoint
    from gpuwm.nextgen.raw_cuda import RawCuda
    from gpuwm.certify.kernel_manifest import kernel_manifest
    cp.get_default_memory_pool().set_limit(size=64 * 1024 * 1024)
    raw = RawCuda(root / "libnextgen_replay.so")
    stream = cp.cuda.Stream(non_blocking=True)
    timings = {}
    drivers = []
    t = time.perf_counter_ns()
    for name, member in (("continuous", 0), ("fresh", 0), ("peer", 1)):
        directory = out / name
        directory.mkdir()
        with member_output_scope(MemberOutputCapture(callback=None, member_id=member)):
            drivers.append(prepare_composition(root, packet, publication, directory, raw, stream, identity))
    stream.synchronize()
    timings["three_driver_preparation_ms"] = (time.perf_counter_ns() - t) / 1e6
    (cfe, transfer, continuous), (fresh_cfe, fresh_transfer, fresh), (_, _, peer) = drivers
    pointers = [{k: a.data.ptr for k, a in d.arrays.items()} for _, _, d in drivers]
    # Compare allocation extents, including the strided CFE output view.
    extents = [[(a.data.ptr, a.data.ptr + sum((n - 1) * s for n, s in zip(a.shape, a.strides))
                 + a.dtype.itemsize) for a in d.arrays.values() if a.size] for _, _, d in drivers]
    for i, left in enumerate(extents):
        for right in extents[i + 1:]:
            assert all(a1 <= b0 or b1 <= a0 for a0, a1 in left for b0, b1 in right)
    fields, exports = [], []
    dtype = {"float32": "f32", "float64": "f64", "int32": "i32", "int64": "i64"}

    def compare_arrays(label, expected, actual):
        for i, (name, array) in enumerate(actual.items()):
            before = out / f"{label}-{i}-expected.raw"
            after = out / f"{label}-{i}-actual.raw"
            exports.extend([(expected[name], before), (array, after)])
            fields.append(dict(name=f"{label}/{name}", dtype=dtype[str(array.dtype)],
                shape=list(array.shape), order="c", expected=dict(path=str(before)), actual=str(after)))

    def advance(driver):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)
        stream.synchronize()

    with stream:
        peer_initial = {k: a.copy() for k, a in peer.arrays.items()}
        peer_metadata = peer.checkpoint_metadata()
        advance(continuous)
        saved_metadata = continuous.checkpoint_metadata()
        assert saved_metadata["steps"] == 1 and saved_metadata["elapsed_seconds"] == 3600
        t = time.perf_counter_ns()
        roster, _, payload = _snapshot(continuous.arrays, stream)
        checkpoint, _ = _call(dict(operation="checkpoint-write", path=str(out / "restart.nc"),
            metadata=saved_metadata, arrays=roster), payload)
        timings["checkpoint_snapshot_write_ms"] = (time.perf_counter_ns() - t) / 1e6
        saved_arrays = {k: a.copy() for k, a in continuous.arrays.items()}
        t = time.perf_counter_ns()
        advance(continuous)
        timings["continuous_second_update_host_ms"] = (time.perf_counter_ns() - t) / 1e6
        final_metadata = continuous.checkpoint_metadata()
        assert fresh.steps == 0 and fresh.elapsed_seconds == 0
        t = time.perf_counter_ns()
        restore_checkpoint(out / "restart.nc", fresh)
        timings["fresh_checkpoint_read_restore_ms"] = (time.perf_counter_ns() - t) / 1e6
        assert fresh.checkpoint_metadata() == saved_metadata
        # Save the restored boundary before the second update overwrites it.
        restored_arrays = {k: a.copy() for k, a in fresh.arrays.items()}
        compare_arrays("restored-boundary", saved_arrays, restored_arrays)
        t = time.perf_counter_ns()
        advance(fresh)
        timings["fresh_second_update_host_ms"] = (time.perf_counter_ns() - t) / 1e6
        assert fresh.checkpoint_metadata() == final_metadata
        assert fresh.steps == 2 and fresh.elapsed_seconds == 7200
        compare_arrays("continued-final", continuous.arrays, fresh.arrays)
        t = time.perf_counter_ns()
        try:
            restore_checkpoint(out / "restart.nc", peer)
        except ValueError as error:
            refusal = str(error)
        else:
            raise AssertionError("member 0 checkpoint was accepted by member 1")
        timings["wrong_member_refusal_ms"] = (time.perf_counter_ns() - t) / 1e6
        assert peer.checkpoint_metadata() == peer_metadata
        compare_arrays("untouched-member-1", peer_initial, peer.arrays)
        for i, (component, routing, driver) in enumerate(drivers):
            assert pointers[i] == {k: a.data.ptr for k, a in driver.arrays.items()}
            assert routing.arrays["depth"].data.ptr == component.fields["flux_Qout_m"].data.ptr
            assert driver.check_health(stream=stream) == {"cfe/status": 0, "transfer/runoff": 0}
        raw.store_many(exports, stream=stream)
    for field in fields:
        field["expected"]["sha256"] = raw.sha256(field["expected"]["path"])
    request = dict(schema="nextgen.raw.compare/v1", unit="cfe-transfer-fresh-restart",
        case="same captured forcing held for a second update; independent member refusal",
        numerics=mode, candidate_revision=identity["revision"], command=[sys.executable, *sys.argv],
        provenance=dict(origin="tool-self-test", sources=[dict(name="NOAA-OWP/cfe",
            revision="f9182dfb3d81c407c66c5b782114173893c30cf2", licence="Apache-2.0")],
            evidence=[dict(path=str(root / "identity.json"), sha256=raw.sha256(root / "identity.json"))]),
        fields=fields)
    request_path = out / "compare.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    t = time.perf_counter_ns()
    proc = subprocess.run([str(root / "nextgen-replay"), str(request_path), str(out / "verdict.json")],
                          capture_output=True, text=True)
    timings["rust_comparison_ms"] = (time.perf_counter_ns() - t) / 1e6
    from cupy.cuda import driver as cuda_driver
    resources = {name: cuda_driver.funcGetAttribute(getattr(cuda_driver, constant), cfe.kernel.ptr)
        for name, constant in (("num_regs", "CU_FUNC_ATTRIBUTE_NUM_REGS"),
            ("local_size_bytes", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"),
            ("shared_size_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"),
            ("max_threads_per_block", "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"))}
    record = dict(revision=identity["revision"], mode=mode, scope=request["case"], timings=timings,
        checkpoint=checkpoint, checkpoint_carriers=len(continuous.arrays),
        logical_carrier_bytes_per_driver=sum(a.nbytes for a in continuous.arrays.values()),
        depth_alias_preserved=True, pointers_preserved=True, allocations_disjoint_between_drivers=True,
        member_ids=[d.member_id for _, _, d in drivers], wrong_member_refusal=refusal,
        saved_metadata=saved_metadata, final_metadata=final_metadata,
        peer_metadata=peer_metadata, stream=int(stream.ptr), cuda_graphs=False,
        ordinary_model_launches=6, launches_per_update=2, threads_per_block=128,
        whole_step_speed_claim=False, resources=dict(cfe=resources, transfer=transfer.kernel.attributes),
        cupy_used_bytes=cp.get_default_memory_pool().used_bytes(),
        cupy_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        driver_api=cp.cuda.runtime.driverGetVersion(), nvrtc=nvrtc.getVersion(), cupy=cp.__version__,
        compiler_manifest=kernel_manifest(), comparison_returncode=proc.returncode,
        comparison_stdout=proc.stdout, comparison_stderr=proc.stderr)
    (out / "checks.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(mode=mode, returncode=proc.returncode, timings=timings)))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
