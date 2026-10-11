"""Compile the shipped cu6 source through NVRTC without creating a GPU context.

The DLL path is explicit. This utility never loads a CUDA driver or launches
a kernel. PTX stays in memory and only a small JSON receipt is written.
"""
from __future__ import annotations
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def compile_source(source, library, options, ptx_path=None):
    library = Path(library).resolve()
    directory = os.add_dll_directory(str(library.parent)) if os.name == "nt" else None
    try:
        lib = c.CDLL(str(library))
        program = c.c_void_p()
        lib.nvrtcCreateProgram.argtypes = [c.POINTER(c.c_void_p), c.c_char_p, c.c_char_p,
                                          c.c_int, c.c_void_p, c.c_void_p]
        lib.nvrtcCompileProgram.argtypes = [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)]
        lib.nvrtcGetProgramLogSize.argtypes = [c.c_void_p, c.POINTER(c.c_size_t)]
        lib.nvrtcGetProgramLog.argtypes = [c.c_void_p, c.c_void_p]
        lib.nvrtcGetPTXSize.argtypes = [c.c_void_p, c.POINTER(c.c_size_t)]
        lib.nvrtcGetPTX.argtypes = [c.c_void_p, c.c_void_p]
        lib.nvrtcDestroyProgram.argtypes = [c.POINTER(c.c_void_p)]
        status = lib.nvrtcCreateProgram(c.byref(program), source.encode(), b"tiedtke.cu", 0, None, None)
        if status:
            raise RuntimeError(f"nvrtcCreateProgram status {status}")
        try:
            opts = (c.c_char_p * len(options))(*(s.encode() for s in options))
            status = lib.nvrtcCompileProgram(program, len(options), opts)
            size = c.c_size_t()
            lib.nvrtcGetProgramLogSize(program, c.byref(size))
            log = c.create_string_buffer(size.value)
            lib.nvrtcGetProgramLog(program, log)
            receipt = {"status": status, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "options": options, "log": log.value.decode(errors="replace"),
                       "gpu_context_created": False, "kernel_launched": False}
            if status == 0:
                lib.nvrtcGetPTXSize(program, c.byref(size))
                ptx = c.create_string_buffer(size.value)
                lib.nvrtcGetPTX(program, ptx)
                receipt["ptx_bytes"] = len(ptx.value)
                receipt["ptx_sha256"] = hashlib.sha256(ptx.value).hexdigest()
                if ptx_path is not None:
                    Path(ptx_path).write_bytes(ptx.value)
            return receipt
        finally:
            lib.nvrtcDestroyProgram(c.byref(program))
    finally:
        if directory is not None:
            directory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--architecture", default="compute_89")
    parser.add_argument("--ptx", type=Path)
    args = parser.parse_args()
    from gpuwm.core.kernels import module_source, module_options
    options = [*module_options("tiedtke"), "--gpu-architecture=" + args.architecture]
    receipt = compile_source(module_source("tiedtke"), args.library, options, args.ptx)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt["log"])
    print(json.dumps({k: v for k, v in receipt.items() if k != "log"}))
    return int(receipt["status"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
