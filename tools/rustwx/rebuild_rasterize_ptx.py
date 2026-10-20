#!/usr/bin/env python3
"""Rebuild the renderer's driver-only PTX with an explicit CUDA 12.0 NVRTC.

No GPU or runtime compiler is needed by the shipped renderer. Use --out in
scratch to reproduce and compare the checked-in file before replacing it.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvrtc-library", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "crates/rustwx-render/src/rasterize_structured.cu"
    compiler = C.CDLL(str(args.nvrtc_library.resolve()))
    compiler.nvrtcVersion.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
    major, minor = C.c_int(), C.c_int()
    assert compiler.nvrtcVersion(C.byref(major), C.byref(minor)) == 0
    if (major.value, minor.value) != (12, 0):
        parser.error("use NVRTC 12.0 to retain PTX 8.0 and the CUDA 12 driver floor")
    compiler.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p]
    compiler.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    compiler.nvrtcGetProgramLogSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    compiler.nvrtcGetProgramLog.argtypes = [C.c_void_p, C.c_void_p]
    compiler.nvrtcGetPTXSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    compiler.nvrtcGetPTX.argtypes = [C.c_void_p, C.c_void_p]
    compiler.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
    program = C.c_void_p()
    code = source.read_bytes()
    assert compiler.nvrtcCreateProgram(C.byref(program), code, source.name.encode(), 0, None, None) == 0
    options = ["--gpu-architecture=compute_60", "--std=c++17", "--fmad=false",
               "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
    try:
        result = compiler.nvrtcCompileProgram(program, len(options), (C.c_char_p * len(options))(*(x.encode() for x in options)))
        size = C.c_size_t()
        assert compiler.nvrtcGetProgramLogSize(program, C.byref(size)) == 0
        log = C.create_string_buffer(size.value)
        assert compiler.nvrtcGetProgramLog(program, log) == 0
        if result:
            raise RuntimeError(log.value.decode(errors="replace"))
        assert compiler.nvrtcGetPTXSize(program, C.byref(size)) == 0
        ptx = C.create_string_buffer(size.value)
        assert compiler.nvrtcGetPTX(program, ptx) == 0
        # NVRTC pads PTX lines with trailing blanks and ends with blank lines; the checked-in file is
        # normalised (no trailing whitespace, one final newline) so the release export ships it unchanged.
        text = b"\n".join(line.rstrip(b" \t") for line in ptx.value.split(b"\n")).rstrip(b"\n") + b"\n"
        args.out.write_bytes(text)
        print(json.dumps({"options": options, "source_sha256": hashlib.sha256(code).hexdigest(),
                          "ptx_sha256": hashlib.sha256(text).hexdigest(), "log": log.value.decode()}))
    finally:
        compiler.nvrtcDestroyProgram(C.byref(program))


if __name__ == "__main__":
    main()
