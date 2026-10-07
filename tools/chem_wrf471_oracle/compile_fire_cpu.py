"""Compile assembled CUDA through NVRTC without loading CUDA or using a device.

Export sources on the desktop, compile on the CPU node. NVRTC is a host
compiler; this script never imports CuPy or calls a CUDA runtime/driver API.
"""
import argparse
import ctypes
import importlib.metadata
import json
import hashlib
from pathlib import Path


def compile_sources(root):
    distribution=importlib.metadata.distribution("nvidia-cuda-nvrtc")
    paths=[distribution.locate_file(f) for f in distribution.files
           if "libnvrtc.so" in str(f) and "builtins" not in str(f)]
    lib=ctypes.CDLL(str(paths[0]))
    ptr=ctypes.c_void_p
    lib.nvrtcCreateProgram.argtypes=[ctypes.POINTER(ptr),ctypes.c_char_p,ctypes.c_char_p,
                                    ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    lib.nvrtcCompileProgram.argtypes=[ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]
    lib.nvrtcGetProgramLogSize.argtypes=[ptr,ctypes.POINTER(ctypes.c_size_t)]
    lib.nvrtcGetProgramLog.argtypes=[ptr,ctypes.c_char_p]
    lib.nvrtcDestroyProgram.argtypes=[ctypes.POINTER(ptr)]
    report={}
    major,minor=ctypes.c_int(),ctypes.c_int()
    lib.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor))
    for path in sorted(root.glob("*.cu")):
        for arch in ("compute_89","compute_120"):
            program=ptr()
            status=lib.nvrtcCreateProgram(ctypes.byref(program),path.read_bytes(),path.name.encode(),0,None,None)
            if status:
                raise RuntimeError(f"NVRTC create failed {status}")
            options=(ctypes.c_char_p*2)(b"--std=c++17",f"--gpu-architecture={arch}".encode())
            status=lib.nvrtcCompileProgram(program,2,options)
            size=ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(program,ctypes.byref(size))
            log=ctypes.create_string_buffer(size.value)
            lib.nvrtcGetProgramLog(program,log)
            lib.nvrtcDestroyProgram(ctypes.byref(program))
            report[f"{path.name}/{arch}"]={"status":status,"log":log.value.decode(),
                "nvrtc_version":f"{major.value}.{minor.value}",
                "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "options":["--std=c++17",f"--gpu-architecture={arch}"]}
    (root / "nvrtc-report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    assert all(v["status"]==0 for v in report.values()),report
    print("NVRTC: two CUDA modules compiled for compute_89 and compute_120; no device APIs called")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--export",action="store_true")
    args=parser.parse_args()
    if args.export:
        from gpuwm.core.kernels import module_source
        args.directory.mkdir(parents=True,exist_ok=True)
        for name in ("chem_fire","chem_wetdep_ls"):
            (args.directory / f"{name}.cu").write_bytes(module_source(name).encode("utf-8"))
    else:
        compile_sources(args.directory)
