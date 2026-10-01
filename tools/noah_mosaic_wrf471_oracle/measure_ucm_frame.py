"""Read the launched functions from their recorded production modules."""
import hashlib
import json
import cupy as cp
from gpuwm.core.noah_mosaic import _mosaic_module, _mosaic_ucm_module, mosaic_ucm_source
from gpuwm.core.kernels import module_source
rows = {"platform": dict(cupy=cp.__version__,
                         compute_capability=cp.cuda.Device().compute_capability,
                         nvrtc=list(cp.cuda.nvrtc.getVersion()),
                         cuda_runtime=cp.cuda.runtime.runtimeGetVersion())}
for name, module, function, source in (
    ("noah_mosaic_unit", _mosaic_module(), "noah_mosaic_column", module_source("noah_mosaic")),
    ("noah_mosaic_ucm_unit", _mosaic_ucm_module(), "noah_mosaic_ucm_column", mosaic_ucm_source())):
    kernel = module.get_function(function)
    rows[name] = dict(function=function, local_size_bytes=kernel.local_size_bytes,
                     num_regs=kernel.num_regs, const_size_bytes=kernel.const_size_bytes,
                     source_sha256=hashlib.sha256(source.encode()).hexdigest())
print(json.dumps(rows, indent=2))
