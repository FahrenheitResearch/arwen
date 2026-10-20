#!/usr/bin/env python3
"""CFE replay orchestration. Rust owns payload I/O; CUDA owns numerics.

Run once per numerics process. Setup/upload/output boundaries synchronize;
the single CFE step only launches. No Python hydrological array processing.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import socket
import sys
import time


class WriteItem(ct.Structure):
    _fields_ = [("path", ct.c_char_p), ("source", ct.c_void_p),
                ("bytes", ct.c_size_t)]


def prepare(manifest_path, raw_library, *, payload_root=None, strict=None):
    """Delegate to the single runtime loader (Rust owns every payload read)."""
    from gpuwm.hydro.cfe import load_prepared
    return load_prepared(manifest_path, raw_library,
                         payload_root=payload_root, strict=strict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("raw_library", type=Path)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--payload-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    import cupy as cp
    from gpuwm.numerics import receipt
    started = time.perf_counter()
    component, manifest, lib = prepare(args.manifest,args.raw_library,payload_root=args.payload_root)
    prepared = time.perf_counter()
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record()
    component.step()
    end.record()
    end.synchronize()
    executed = time.perf_counter()
    elapsed_ms = cp.cuda.get_elapsed_time(begin, end)
    arrays = component.arrays
    lib.nextgen_raw_write_many.argtypes = [ct.POINTER(WriteItem), ct.c_size_t,
                                         ct.c_void_p, ct.c_size_t]
    lib.nextgen_raw_write_many.restype = ct.c_int
    pinned = [cp.cuda.alloc_pinned_memory(max(1,a.nbytes)) for a in arrays.values()]
    paths = [str(args.output / f"{name}.bin").encode() for name in arrays]
    jobs = (WriteItem * len(arrays))()
    for j, (a, ptr, path) in enumerate(zip(arrays.values(), pinned, paths)):
        if a.nbytes:
            cp.cuda.runtime.memcpy(ptr.ptr, a.data.ptr, a.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
        jobs[j] = WriteItem(path,ptr.ptr,a.nbytes)
    error = ct.create_string_buffer(4096)
    if lib.nextgen_raw_write_many(jobs,len(jobs),error,len(error)):
        raise RuntimeError(error.value.decode())
    properties = cp.cuda.runtime.getDeviceProperties(component.device_id)
    report = {"schema":"cfe.gpu.replay-run/v1", "manifest":str(args.manifest),
              "candidate_revision":manifest["candidate_revision"],"numerics":receipt(),
              "card":properties["name"].decode(),"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),
              "cupy":cp.__version__, "catchments":component.ncatchments,
              "command":[sys.executable,*sys.argv],
              "cwd":os.getcwd(), "host":socket.gethostname(),
              "environment":{key:os.environ.get(key) for key in
                             ("GPUWM_NUMERICS","CUPY_CACHE_DIR","CUDA_CACHE_PATH","TMPDIR","RAYON_NUM_THREADS")},
              "parallelism":"one plain CUDA launch; 128 threads/block; one thread/catchment; no CUDA graphs",
              "setup_wall_ms":(prepared-started)*1000,"first_launch_gpu_ms":elapsed_ms,
              "launch_and_output_sync_wall_ms":(executed-prepared)*1000,
              "download_and_rust_write_wall_ms":(time.perf_counter()-executed)*1000,
              "scope":manifest["scope"],"restart_identity":component.restart_identity}
    with (args.output / "run.json").open("x") as stream:
        json.dump(report,stream,indent=2);stream.write("\n")
    print(json.dumps({k:report[k] for k in ("card","catchments","first_launch_gpu_ms","setup_wall_ms")}))


if __name__ == "__main__":
    main()
