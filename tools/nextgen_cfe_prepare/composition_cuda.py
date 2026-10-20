"""Private native-entry CFE -> resident runoff transfer replay consumer.

Uses an existing BMI packet, explicit original native areas/nexuses, and the
original nexus words at that same update. No routing sampler or physical event.
All payload I/O/comparison is Rust; model arithmetic and transport are CUDA.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import subprocess
import sys
import time


def prepare_composition(root, packet, publication, out, raw, stream, identity):
    """Bind the existing CFE and transfer factories, retaining their depth alias."""
    import cupy as cp
    from gpuwm.hydro import ResidentHydroDriver, Stage
    from gpuwm.hydro.cfe import prepare as prepare_cfe
    from gpuwm.hydro.runoff import prepare as prepare_transfer
    precipitation = packet["bmi"]["precipitation_buffer"]
    bmi_names = {precipitation, "potential_et_m_s", "current_time_step",
                 "time_step_fraction", "nwm_ponded_depth_m"}
    core = dict(packet, buffers=[r for r in packet["buffers"] if r["name"] not in bmi_names])
    core_path = out / "core.json"
    core_path.write_text(json.dumps(core, indent=2) + "\n")
    specs = {r["name"]: dict(r, path=str(root / "bmi/input" / Path(r["path"]).name), order="c")
             for r in packet["buffers"] if r["name"] in bmi_names}
    for name in ("area", "receiver"):
        row = publication["inputs"][name]
        specs[name] = dict(row, path=str(root / "publication" / row["path"]))
    with stream:
        inputs = raw.load_many(specs, stream=stream)
        binding = dict(catchment_ids=publication["catchment_ids"],
                       time_origin=identity["time_origin"],
                       origin="captured native BMI inputs; no live upstream model")
        cfe = prepare_cfe(settings={}, state=None, physics=None, device_id=0,
            mode="standalone", options=dict(manifest=str(core_path),
                library=str(root / "libnextgen_replay.so"), payload_root=str(root / "bmi/input"),
                precipitation_component="rain", precipitation_field=precipitation,
                precipitation_units=packet["bmi"]["precipitation_units"], pet_component="pet",
                initial_step=packet["bmi"]["initial_step"],
                time_step_fraction=packet["bmi"]["time_step_fraction"], identity=binding),
            components=dict(rain=SimpleNamespace(fields={precipitation: inputs[precipitation]},
                            restart_identity=binding),
                pet=SimpleNamespace(fields={"potential_et_m_s": inputs["potential_et_m_s"]},
                                    restart_identity=binding)))
        for name in ("current_time_step", "time_step_fraction", "nwm_ponded_depth_m"):
            cp.copyto(cfe.arrays[name], inputs[name])
        fabric = SimpleNamespace(catchment_ids=tuple(publication["catchment_ids"]),
            node_ids=tuple(publication["node_ids"]), arrays={"catchment_to_node": inputs["receiver"]},
            restart_identity=dict(native_publication=publication["native_index"],
                                  scope="original distinct nexuses only; no reach routing"))
        areas = SimpleNamespace(catchment_ids=fabric.catchment_ids, arrays={"areasqkm": inputs["area"]},
            restart_identity=dict(native_area=publication["inputs"]["area"], units="km2"))
        transfer = prepare_transfer(settings={}, state=None, physics=None, device_id=0,
            mode="standalone", options=dict(source_component="cfe", network_component="fabric",
                area_component="areas", area_array="areasqkm", area_units="km2", depth_field="flux_Qout_m",
                source_catchments_path=["preparation", "binding", "ordering", "catchment_ids"]),
            components=dict(cfe=cfe, fabric=fabric, areas=areas))
        assert transfer.arrays["depth"].data.ptr == cfe.fields["flux_Qout_m"].data.ptr
        driver = ResidentHydroDriver(device_id=0, interval_seconds=3600,
            identity=dict(scope="one native-entry CFE update then native hourly discharge publication"),
            after=[Stage("cfe", cfe, cfe.advance), Stage("transfer", transfer, transfer.advance)])
    return cfe, transfer, driver


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
    assert len(publication["catchment_ids"]) == len(packet["catchments"])
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    from gpuwm.nextgen.raw_cuda import RawCuda
    from gpuwm.certify.kernel_manifest import kernel_manifest
    cp.get_default_memory_pool().set_limit(size=64 * 1024 * 1024)
    raw = RawCuda(root / "libnextgen_replay.so")
    stream = cp.cuda.Stream(non_blocking=True)
    cfe, transfer, driver = prepare_composition(root, packet, publication, out, raw, stream, identity)
    with stream:
        pointers = {k: a.data.ptr for k, a in driver.arrays.items()}
        stream.synchronize()
        t = time.perf_counter_ns()
        driver.before_lsm(stream=stream)
        driver.advance(stream=stream)
        stream.synchronize()
        wall_ms = (time.perf_counter_ns() - t) / 1e6
        assert driver.steps == 1 and driver.elapsed_seconds == 3600
        assert driver.check_health(stream=stream) == {"cfe/status": 0, "transfer/runoff": 0}
        assert pointers == {k: a.data.ptr for k, a in driver.arrays.items()}
        raw.store_many([(a, out / f"{k}.bin") for k, a in cfe.arrays.items()] +
            [(transfer.arrays["routing_discharge_m3_s"], out / "nexus.raw")], stream=stream)
    request = json.loads((root / "bmi/compare.json").read_text())
    for row in request["fields"]:
        row["expected"]["path"] = str(root / "bmi/expected" / Path(row["expected"]["path"]).name)
        row["actual"] = str(out / Path(row["actual"]).name)
    for evidence in request["provenance"]["evidence"]:
        evidence["path"] = str(root / "evidence" / Path(evidence["path"]).name)
    request["provenance"]["evidence"].append(dict(path=str(root / "identity.json"),
        sha256=raw.sha256(root / "identity.json")))
    request.update(numerics=mode, candidate_revision=identity["revision"], command=[sys.executable, *sys.argv])
    request["fields"].append(dict(name="original-native-nexus-discharge", dtype="f64",
        shape=[len(publication["node_ids"])], order="c", units="m3/s",
        expected=dict(path=str(root / "native-nexus.raw"), sha256=identity["files"]["native-nexus.raw"]),
        actual=str(out / "nexus.raw")))
    request_path = out / "compare.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    proc = subprocess.run([str(root / "nextgen-replay"), str(request_path), str(out / "verdict.json")],
                          text=True, capture_output=True)
    # CFE binds a low-level Function; the transfer binds a RawKernel.
    from cupy.cuda import driver as cuda_driver
    cfe_resources = {name: cuda_driver.funcGetAttribute(getattr(cuda_driver, constant), cfe.kernel.ptr)
        for name, constant in (("num_regs", "CU_FUNC_ATTRIBUTE_NUM_REGS"),
            ("local_size_bytes", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"),
            ("shared_size_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"),
            ("max_threads_per_block", "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"))}
    record = dict(revision=identity["revision"], mode=mode, native_step=packet["step"],
        scope=driver.checkpoint_metadata()["identity"], launch_and_sync_host_ms=wall_ms,
        launches=2, threads_per_block=128, checkpoint_carriers=len(driver.arrays),
        logical_carrier_bytes=sum(a.nbytes for a in driver.arrays.values()),
        depth_alias_preserved=True, pointers_preserved=True, stream=int(stream.ptr),
        cfe_identity=cfe.restart_identity, transfer_identity=transfer.restart_identity,
        resources=dict(cfe=cfe_resources, transfer=transfer.kernel.attributes),
        cupy_used_bytes=cp.get_default_memory_pool().used_bytes(),
        cupy_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
        driver_api=cp.cuda.runtime.driverGetVersion(), nvrtc=nvrtc.getVersion(), cupy=cp.__version__,
        compiler_manifest=kernel_manifest(), comparison_returncode=proc.returncode,
        comparison_stdout=proc.stdout, comparison_stderr=proc.stderr)
    (out / "checks.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(mode=mode, comparison_returncode=proc.returncode,
        launches=2, launch_and_sync_host_ms=wall_ms, checkpoint_carriers=len(driver.arrays))))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
