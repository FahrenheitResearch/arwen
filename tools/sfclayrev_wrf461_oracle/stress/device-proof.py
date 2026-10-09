"""Device/toolchain receipt. Run only inside the lane's GPU queue hold."""
import json
import cupy as cp
from cupy.cuda import nvrtc

p = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
print(json.dumps(dict(cupy=cp.__version__, nvrtc=list(nvrtc.getVersion()),
    runtime=cp.cuda.runtime.runtimeGetVersion(), driver=cp.cuda.runtime.driverGetVersion(),
    device=p['name'].decode(), compute_capability=[p['major'], p['minor']]), indent=2))
