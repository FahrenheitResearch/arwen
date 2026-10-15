"""Record the actual new-generation CUDA local frames and compile policy."""
import hashlib
import json
from pathlib import Path
import sys

import cupy as cp
from cupy_backends.cuda.api import driver
from gpuwm.core.kernels import get_kernel, get_kernel_int_defines, module_source, module_options

entries = ("ysu_column", "ysu_column_bep", "ysu_column_topo")
frames = {name: int(driver.funcGetAttribute(3, get_kernel("ysu_480", name).ptr))
          for name in entries}  # CUDA CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3.
receipt = {"module": "ysu_480", "options": module_options("ysu_480"),
           "source_sha256": hashlib.sha256(module_source("ysu_480").encode()).hexdigest(),
           "local_frames_bytes": frames, "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
           "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode()}
receipt["legacy_source_ordered_frames_bytes"] = {
    name: int(driver.funcGetAttribute(3, get_kernel_int_defines(
        "ysu_480", name, (("YSU_LEGACY_TOPDOWN", 1),)).ptr)) for name in entries}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
assert all(value == 0 for value in frames.values()), "preflight's zero local-frame bound needs to change"
assert all(value == 0 for value in receipt["legacy_source_ordered_frames_bytes"].values())
