#!/usr/bin/env python3
"""One real native BMI-update replay; shared Rust I/O, resident CUDA physics."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("raw_library", type=Path)
    parser.add_argument("raw_driver", type=Path, help="shared owner's unchanged raw_cuda.py")
    parser.add_argument("output", type=Path)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--factory", action="store_true", help="test published factory with explicitly captured source components")
    parser.add_argument("--catchment-ids", help="audited comma-separated roster for factory identity")
    parser.add_argument("--time-origin", help="actual native UTC origin for factory identity")
    parser.add_argument("--initial-manifest", type=Path,
                        help="production parameter initializer packet for a native step-zero proof")
    parser.add_argument("--initial-payload-root", type=Path)
    args = parser.parse_args()
    if args.initial_manifest and not args.factory:
        parser.error("--initial-manifest requires --factory")
    if args.initial_payload_root and not args.initial_manifest:
        parser.error("--initial-payload-root requires --initial-manifest")
    args.output.mkdir()
    import cupy as cp
    from gpuwm.hydro.cfe import CFEGPU, FIELD_NAMES, OPTION_NAMES
    from gpuwm.numerics import receipt
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text())
    if (manifest["schema"] != "cfe.capture.pack/v1" or not manifest.get("bmi")
            or tuple(manifest["field_names"]) != FIELD_NAMES
            or tuple(manifest["option_names"]) != OPTION_NAMES):
        raise ValueError("manifest must use the published CFE BMI layout")
    spec = importlib.util.spec_from_file_location("shared_nextgen_raw_cuda", args.raw_driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bridge = module.RawCuda(args.raw_library.resolve(strict=True))
    specs = {row["name"]: dict(row, order="c", path=str(
        args.payload_root / Path(row["path"]).name) if args.payload_root else row["path"])
        for row in manifest["buffers"]}
    precipitation = manifest["bmi"].get("precipitation_buffer", "rainrate_mm_s")
    bmi_names = (precipitation, "potential_et_m_s", "current_time_step",
                 "time_step_fraction", "nwm_ponded_depth_m")
    arrays = bridge.load_many({name: row for name, row in specs.items()
                              if not args.factory or name in bmi_names})
    bmi = {name: arrays.pop(name) for name in (
        precipitation, "potential_et_m_s", "current_time_step",
        "time_step_fraction", "nwm_ponded_depth_m")}
    if args.factory:
        from gpuwm.hydro.cfe import prepare
        if not args.catchment_ids or not args.time_origin:
            raise ValueError("factory proof requires audited catchment IDs and native origin")
        identity = dict(catchment_ids=args.catchment_ids.split(","), time_origin=args.time_origin)
        capture_identity = dict(identity, origin="captured native BMI inputs; not live upstream GPU physics",
                                capture_index_sha256=manifest["native_capture_index_sha256"])
        core_packet = dict(manifest, buffers=[row for row in manifest["buffers"] if row["name"] not in bmi_names])
        core_path = (args.output / "core-manifest.json").resolve()
        core_path.write_text(json.dumps(core_packet,indent=2)+"\n")
        if args.initial_manifest:
            initial = json.loads(args.initial_manifest.read_text())
            if (manifest["step"] != 0 or manifest["bmi"]["initial_step"] != 0
                    or initial["schema"] != "cfe.initial.pack/v1"):
                raise ValueError("parameter initialization proof requires native BMI step zero")
        initial_path = args.initial_manifest or core_path
        payload_root = args.initial_payload_root if args.initial_manifest else args.payload_root
        component = prepare(settings=None, state=None, physics=None,
            device_id=cp.cuda.runtime.getDevice(), mode="standalone",
            options=dict(manifest=str(initial_path), library=str(args.raw_library.resolve()),
                payload_root=str(payload_root.resolve()) if payload_root else None,
                precipitation_component="captured-precipitation", precipitation_field=precipitation,
                precipitation_units=manifest["bmi"].get("precipitation_units", "mm/s"),
                pet_component="captured-pet", initial_step=manifest["bmi"]["initial_step"],
                time_step_fraction=manifest["bmi"]["time_step_fraction"], identity=identity),
            components={"captured-precipitation":SimpleNamespace(
                fields={precipitation:bmi[precipitation]}, restart_identity=capture_identity),
                "captured-pet":SimpleNamespace(fields={"potential_et_m_s":bmi["potential_et_m_s"]},
                                               restart_identity=capture_identity)})
    else:
        units = manifest["bmi"].get("precipitation_units", "mm/s")
        kwargs = {"rainrate_units":units} if units != "mm/s" else {}
        component = CFEGPU(**arrays, period_seconds=manifest["period_seconds"],
            rainrate=bmi[precipitation], potential_et=bmi["potential_et_m_s"],
            initial_step=manifest["bmi"]["initial_step"],
            time_step_fraction=manifest["bmi"]["time_step_fraction"], **kwargs,
            identity={"capture_index_sha256":manifest["native_capture_index_sha256"],
                      "step":manifest["step"], "catchments":manifest["catchments"]})
    # Parameter-initialization proofs exercise its ordinary constructor values.
    # Other module replays restore the exact captured carriers at setup only.
    if not args.initial_manifest:
        for name in ("current_time_step", "time_step_fraction", "nwm_ponded_depth_m"):
            cp.copyto(component.arrays[name], bmi[name])
    cp.cuda.get_current_stream().synchronize()
    prepared = time.perf_counter()
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record()
    component.advance(manifest["period_seconds"])
    end.record()
    end.synchronize()
    executed = time.perf_counter()
    elapsed_ms = cp.cuda.get_elapsed_time(begin, end)
    bridge.store_many([(array, args.output / f"{name}.bin")
                       for name, array in component.arrays.items()])
    exported = time.perf_counter()
    props = cp.cuda.runtime.getDeviceProperties(component.device_id)
    report = {"schema":"cfe.gpu.replay-run/v1", "candidate_revision":manifest["candidate_revision"],
        "manifest":str(args.manifest), "numerics":receipt(),
        "card":props["name"].decode(), "catchments":component.ncatchments,
        "command":[sys.executable,*sys.argv], "cwd":os.getcwd(), "host":socket.gethostname(),
        "environment":{key:os.environ.get(key) for key in (
            "GPUWM_NUMERICS", "CUPY_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR", "RAYON_NUM_THREADS")},
        "cuda_runtime":cp.cuda.runtime.runtimeGetVersion(), "cupy":cp.__version__,
        "raw_driver_sha256":bridge.sha256(args.raw_driver),
        "preparation":"published factory with captured source components" if args.factory else "direct constructor",
        "initial_manifest":str(args.initial_manifest) if args.initial_manifest else None,
        "initial_manifest_sha256":bridge.sha256(args.initial_manifest) if args.initial_manifest else None,
        "parallelism":"one plain CUDA launch; 128 threads/block; one thread/catchment; no CUDA graphs",
        "setup_wall_ms":(prepared-started)*1000, "first_launch_gpu_ms":elapsed_ms,
        "launch_and_output_sync_wall_ms":(executed-prepared)*1000,
        "download_and_rust_write_wall_ms":(exported-executed)*1000,
        "scope":manifest["scope"], "restart_identity":component.restart_identity}
    with (args.output / "run.json").open("x") as stream:
        json.dump(report,stream,indent=2); stream.write("\n")
    print(json.dumps({k:report[k] for k in ("card","catchments","first_launch_gpu_ms","setup_wall_ms")}))


if __name__ == "__main__":
    main()
