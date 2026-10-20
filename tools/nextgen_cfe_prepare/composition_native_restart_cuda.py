"""Two captured native BMI updates through CFE, transfer and a fresh CDF5 restore.

Rust packs and compares native words. CUDA copies only the next captured external
precipitation/PET inputs; state and clocks continue without replacement. This is
a native component composition check, not a physical routed event or cost pair.
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
    packet = json.loads((root / "bmi/manifest.json").read_text())
    following = json.loads((root / "next/manifest.json").read_text())
    publication = json.loads((root / "publication/fixture.json").read_text())
    assert packet["step"] == identity["native_step"]
    assert following["step"] == packet["step"] + 1
    assert following["native_capture_index_sha256"] == packet["native_capture_index_sha256"]
    assert following["bmi"]["framework_witness"]["catchment_ids"] == publication["catchment_ids"]
    assert following["bmi"]["precipitation_units"] == packet["bmi"]["precipitation_units"]
    mode = os.environ["GPUWM_NUMERICS"]
    out = root / mode
    out.mkdir()
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    from gpuwm.hydro.output import _snapshot, _call, restore_checkpoint
    from gpuwm.nextgen.raw_cuda import RawCuda
    from gpuwm.certify.kernel_manifest import kernel_manifest
    cp.get_default_memory_pool().set_limit(size=64 * 1024 * 1024)
    raw = RawCuda(root / "libnextgen_replay.so")
    stream = cp.cuda.Stream(non_blocking=True)
    drivers = []
    timings = {}
    t = time.perf_counter_ns()
    for name in ("continuous", "fresh", "stale"):
        directory = out / name
        directory.mkdir()
        with member_output_scope(MemberOutputCapture(callback=None, member_id=0)):
            drivers.append(prepare_composition(root, packet, publication, directory, raw, stream, identity))
    stream.synchronize()
    timings["three_driver_preparation_ms"] = (time.perf_counter_ns() - t) / 1e6
    pointers = [{k: a.data.ptr for k, a in d.arrays.items()} for _, _, d in drivers]
    precipitation = packet["bmi"]["precipitation_buffer"]
    forcing_names = (precipitation, "potential_et_m_s")
    specs = {r["name"]: dict(r, path=str(root / "next/input" / Path(r["path"]).name), order="c")
             for r in following["buffers"] if r["name"] in forcing_names}
    assert set(specs) == set(forcing_names)
    next_inputs = raw.load_many(specs, stream=stream)
    fields, exports = [], []
    dtype = {"float32": "f32", "float64": "f64", "int32": "i32", "int64": "i64"}

    def export(label, arrays):
        paths = {name: out / f"{label}-{i}.raw" for i, name in enumerate(arrays)}
        exports.extend((a, paths[name]) for name, a in arrays.items())
        return paths

    def same_arrays(label, expected, actual):
        before = export(label + "-expected", expected)
        after = export(label + "-actual", actual)
        for name, array in actual.items():
            fields.append(dict(name=label + "/" + name, dtype=dtype[str(array.dtype)],
                shape=list(array.shape), order="c", expected=dict(path=str(before[name])), actual=str(after[name])))

    def advance(driver):
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)
        stream.synchronize()

    (cfe, transfer, continuous), (fresh_cfe, fresh_transfer, fresh), (stale_cfe, stale_transfer, stale) = drivers
    with stream:
        advance(continuous)
        advance(stale)
        first_nexus = transfer.arrays["routing_discharge_m3_s"].copy()
        saved_metadata = continuous.checkpoint_metadata()
        assert saved_metadata["steps"] == 1 and saved_metadata["elapsed_seconds"] == 3600
        t = time.perf_counter_ns()
        roster, _, payload = _snapshot(continuous.arrays, stream)
        checkpoint, _ = _call(dict(operation="checkpoint-write", path=str(out / "restart.nc"),
            metadata=saved_metadata, arrays=roster), payload)
        timings["checkpoint_snapshot_write_ms"] = (time.perf_counter_ns() - t) / 1e6
        t = time.perf_counter_ns()
        restore_checkpoint(out / "restart.nc", fresh)
        timings["fresh_checkpoint_read_restore_ms"] = (time.perf_counter_ns() - t) / 1e6
        assert fresh.checkpoint_metadata() == saved_metadata
        same_arrays("restored-boundary", continuous.arrays, fresh.arrays)
        # Copies must snapshot this boundary before either state advances.
        raw.store_many(exports, stream=stream)
        exports.clear()
        for component in (cfe, fresh_cfe):
            for name in forcing_names:
                cp.copyto(component.arrays[name], next_inputs[name])
        t = time.perf_counter_ns()
        advance(continuous)
        advance(fresh)
        advance(stale)  # Deliberately omit next forcing for this negative arm.
        timings["three_second_updates_host_ms"] = (time.perf_counter_ns() - t) / 1e6
        assert continuous.checkpoint_metadata() == fresh.checkpoint_metadata()
        assert continuous.steps == 2 and continuous.elapsed_seconds == 7200
        same_arrays("continued-final", continuous.arrays, fresh.arrays)
        paths = {label: export(label, component.arrays) for label, component in
                 (("native-continuous", cfe), ("native-fresh", fresh_cfe), ("stale", stale_cfe))}
        nexus = export("nexus", dict(first=first_nexus,
            continuous=transfer.arrays["routing_discharge_m3_s"],
            fresh=fresh_transfer.arrays["routing_discharge_m3_s"],
            stale=stale_transfer.arrays["routing_discharge_m3_s"]))
        raw.store_many(exports, stream=stream)
    for field in fields:
        field["expected"]["sha256"] = raw.sha256(field["expected"]["path"])
    native = json.loads((root / "next/compare.json").read_text())
    negative = []
    for label in ("native-continuous", "native-fresh", "stale"):
        target = negative if label == "stale" else fields
        for row in native["fields"]:
            target.append(dict(row, name=label + "/" + row["name"],
                expected=dict(row["expected"], path=str(root / "next/expected" / Path(row["expected"]["path"]).name)),
                actual=str(paths[label][row["name"]])))
        target.append(dict(name=label + "/original-native-nexus", dtype="f64", shape=[len(publication["node_ids"])],
            order="c", expected=dict(path=str(root / "next-nexus.raw"), sha256=raw.sha256(root / "next-nexus.raw")),
            actual=str(nexus["stale" if label == "stale" else label.removeprefix("native-")])))
    fields.append(dict(name="first-hour-original-native-nexus", dtype="f64", shape=[len(publication["node_ids"])],
        order="c", expected=dict(path=str(root / "native-nexus.raw"), sha256=raw.sha256(root / "native-nexus.raw")),
        actual=str(nexus["first"])))
    request = dict(native, unit="cfe-transfer-native-fresh-restart", case="consecutive native BMI updates plus fresh CDF5 continuation",
        numerics=mode, candidate_revision=identity["revision"], command=[sys.executable, *sys.argv], fields=fields)
    request["provenance"]["evidence"] = [dict(path=str(root / "evidence" / Path(row["path"]).name), sha256=row["sha256"])
        for row in native["provenance"]["evidence"]]
    request["provenance"]["evidence"].append(dict(path=str(root / "identity.json"), sha256=raw.sha256(root / "identity.json")))
    results = {}
    for label, rows in (("positive", fields), ("negative", negative)):
        path = out / (label + ".json")
        path.write_text(json.dumps(dict(request, fields=rows), indent=2) + "\n")
        t = time.perf_counter_ns()
        proc = subprocess.run([str(root / "nextgen-replay"), str(path), str(out / (label + "-verdict.json"))], capture_output=True, text=True)
        results[label] = dict(returncode=proc.returncode, wall_ms=(time.perf_counter_ns() - t) / 1e6,
            stdout=proc.stdout, stderr=proc.stderr)
    for i, (component, routing, driver) in enumerate(drivers):
        assert pointers[i] == {k: a.data.ptr for k, a in driver.arrays.items()}
        assert routing.arrays["depth"].data.ptr == component.fields["flux_Qout_m"].data.ptr
        assert driver.check_health(stream=stream) == {"cfe/status": 0, "transfer/runoff": 0}
    from cupy.cuda import driver as cuda_driver
    resources = {name: cuda_driver.funcGetAttribute(getattr(cuda_driver, constant), cfe.kernel.ptr)
        for name, constant in (("num_regs", "CU_FUNC_ATTRIBUTE_NUM_REGS"), ("local_size_bytes", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"),
            ("shared_size_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"), ("max_threads_per_block", "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"))}
    record = dict(revision=identity["revision"], mode=mode, results=results, timings=timings, checkpoint=checkpoint,
        native_steps=[packet["step"], following["step"]], external_inputs_replaced=list(forcing_names),
        saved_metadata=saved_metadata, final_metadata=continuous.checkpoint_metadata(),
        checkpoint_carriers=len(continuous.arrays), logical_carrier_bytes=sum(a.nbytes for a in continuous.arrays.values()),
        pointers_preserved=True, depth_alias_preserved=True, cuda_graphs=False, ordinary_model_launches=10,
        launches_per_update=2, threads_per_block=128, whole_step_speed_claim=False,
        resources=dict(cfe=resources, transfer=transfer.kernel.attributes), compiler_manifest=kernel_manifest(),
        cupy_used_bytes=cp.get_default_memory_pool().used_bytes(), cupy_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        driver_api=cp.cuda.runtime.driverGetVersion(), nvrtc=nvrtc.getVersion(), cupy=cp.__version__)
    (out / "checks.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(mode=mode, results=results, timings=timings)))
    assert results["positive"]["returncode"] == 0, results
    assert results["negative"]["returncode"] == 1, results


if __name__ == "__main__":
    main()
