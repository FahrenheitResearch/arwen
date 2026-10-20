#!/usr/bin/env python3
"""Replay sealed, externally observed t-route calls; Python orchestrates only."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PIN = "12a8eae0cdfed437143c590659fa7077605a5e70"
INPUTS = ["dt", "qup", "quc", "qdp", "ql", "dx", "bw", "tw", "twcc",
          "n", "ncc", "cs", "s0", "velp", "depthp"]
OUTPUTS = ["qdc", "velc", "depthc", "ck", "cn", "X"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(manifest):
    if manifest.get("schema") != "troute-mc-calls-v1" or manifest.get("source_pin") != PIN:
        raise ValueError("fixture must name the fresh pinned t-route native source")
    if manifest.get("inputs") != INPUTS or manifest.get("outputs") != OUTPUTS:
        raise ValueError("all 15 inputs and six outputs, in native order, are required")
    if not isinstance(manifest.get("calls"), int) or manifest["calls"] <= 0:
        raise ValueError("nonempty native calls required")
    if manifest.get("nonperturbation", {}).get("pass") is not True:
        raise ValueError("observer must preserve the whole native case result")
    for name, width in (("input_array", 15), ("expected_array", 6)):
        field = manifest[name]
        if field.get("dtype") != "<f4" or field.get("shape") != [manifest["calls"], width]:
            raise ValueError("fixture arrays must retain native float32 C AoS shapes")
        if len(field.get("sha256", "")) != 64 or any(c not in "0123456789abcdef" for c in field["sha256"]):
            raise ValueError("fixture array SHA256 missing/malformed")


class Raw:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        args = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        for name in ("troute_raw_load", "troute_raw_store"):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, ctypes.c_int

    def transfer(self, name, path, pointer, size):
        error = ctypes.create_string_buffer(4096)
        result = getattr(self.lib, name)(os.fsencode(path), pointer, size, error, len(error))
        if result:
            raise RuntimeError(error.value.decode())

    def load(self, cp, path, shape):
        array = cp.empty(tuple(shape), dtype=cp.float32)
        host = cp.cuda.alloc_pinned_memory(array.nbytes)
        self.transfer("troute_raw_load", path, host.ptr, array.nbytes)
        cp.cuda.runtime.memcpy(array.data.ptr, host.ptr, array.nbytes, cp.cuda.runtime.memcpyHostToDevice)
        return array

    def store(self, cp, path, array):
        host = cp.cuda.alloc_pinned_memory(array.nbytes)
        cp.cuda.runtime.memcpy(host.ptr, array.data.ptr, array.nbytes, cp.cuda.runtime.memcpyDeviceToHost)
        self.transfer("troute_raw_store", path, host.ptr, array.nbytes)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--port-module", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads(args.fixture.read_text())
    validate(manifest)
    base = args.fixture.resolve().parent
    for field in ("input_array", "expected_array"):
        if sha(base / manifest[field]["path"]) != manifest[field]["sha256"]:
            raise ValueError(f"{field} seal mismatch")
    args.output.mkdir()
    import cupy as cp
    from gpuwm.numerics import is_strict, mode
    spec = importlib.util.spec_from_file_location("troute_mc_under_test", args.port_module)
    port = importlib.util.module_from_spec(spec)
    # Dataclass annotation resolution looks up the defining module by name.
    # Preserve normal import semantics for the port loaded from an exact path.
    sys.modules[spec.name] = port
    spec.loader.exec_module(port)
    bridge = Raw(args.library.resolve())
    x = bridge.load(cp, base / manifest["input_array"]["path"], manifest["input_array"]["shape"])
    y = cp.empty((manifest["calls"], 6), dtype=cp.float32)
    compiled = port.module(strict=is_strict())
    kernel = compiled.get_function("troute_mc_replay")
    arguments = (cp.int32(manifest["calls"]), x, y)
    grid, block = ((manifest["calls"] + 127) // 128,), (128,)
    # First launch is the real replay. Warm-up/benchmark repeats the same
    # immutable independent calls; they do not advance a network/model state.
    kernel(grid, block, arguments)
    cp.cuda.get_current_stream().synchronize()
    bridge.store(cp, args.output / "actual.f32", y)
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    begin.record()
    for _ in range(10):
        kernel(grid, block, arguments)
    end.record()
    end.synchronize()
    launch_ms = cp.cuda.get_elapsed_time(begin, end) / 10
    expected = base / manifest["expected_array"]["path"]
    command = [str(args.tool.resolve()), "compare", str(expected), str(args.output / "actual.f32"), "4"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    comparison = json.loads(result.stdout)
    (args.output / "result.json").write_text(json.dumps(comparison, indent=2) + "\n")
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    receipt = dict(schema="troute-mc-gpu-v1", source_pin=PIN, numerics=mode(),
                   fixture_sha256=sha(args.fixture), input_sha256=manifest["input_array"]["sha256"],
                   expected_sha256=manifest["expected_array"]["sha256"], actual_sha256=sha(args.output / "actual.f32"),
                   port_module_sha256=sha(args.port_module), native_provenance=manifest["native"],
                   raw_bridge_sha256=sha(args.library), comparator_sha256=sha(args.tool),
                   device_name=props["name"].decode(), grid=list(grid), block=list(block),
                   calls=manifest["calls"], compared_columns=OUTPUTS, comparisons=comparison,
                   batch_kernel_ms=launch_ms, timing_scope="independent captured-call batch, not network step",
                   options=list(compiled.options), source_sha256=hashlib.sha256(compiled.code.encode()).hexdigest(),
                   attributes=kernel.attributes, cuda_graphs=False,
                   wall_seconds=time.perf_counter() - start)
    (args.output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(comparison))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
