"""CPU-only NVRTC replay in an isolated process, never a CUDA context.

The explicit target selector is the same one the AOT builder uses. All
compiler options, including the engine's final numerical options, come from
the parent request. Only compilation runs here; the parent loads its image.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache


@lru_cache(maxsize=1)
def _identity():
    from .kernel_aot import compiler_identity
    return compiler_identity()


def compile_request(arguments, arch, identity, *, trace=False):
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    from . import numerics

    if _identity() != identity:
        raise ValueError("kernel-plan worker compiler identity changed")
    target = str(arch).removeprefix("sm_")
    if compiler._use_ptx or int(target) not in nvrtc.getSupportedArchs():
        raise ValueError("kernel-plan worker requires the explicit supported cubin target")
    selector = compiler._get_arch_for_options_for_nvrtc

    def explicit(selected=None):
        if str(selected) != target:
            raise ValueError("kernel-plan worker target changed")
        return "-arch=sm_" + target, "cubin"

    begin_ns = time.monotonic_ns() if trace else None
    wall, cpu = time.perf_counter(), time.process_time()
    compiler._get_arch_for_options_for_nvrtc = explicit
    previous_mode = numerics._mode
    # The parent has already applied its numerical boundary. In particular,
    # a builder may compile strict AND fast recipes in one process pool.
    # Never reinterpret those final options using this worker's environment.
    numerics._mode = "strict"
    try:
        result = compiler._compile_using_nvrtc_no_warning(
            arguments["source"], tuple(arguments["options"]), target,
            arguments["filename"], arguments["name_expressions"], None,
            arguments["cache_in_memory"], None, arguments["method"])
    finally:
        numerics._mode = previous_mode
        compiler._get_arch_for_options_for_nvrtc = selector
    timing = {"pid": os.getpid(), "seconds": time.perf_counter() - wall,
              "cpu_seconds": time.process_time() - cpu}
    if trace:
        # Same-host monotonic boundaries around the successful compiler call.
        # These include wrapper overhead and are not low-level NVRTC events.
        timing.update(compile_begin_mono_ns=begin_ns,
                      compile_end_mono_ns=time.monotonic_ns())
    return result, timing
