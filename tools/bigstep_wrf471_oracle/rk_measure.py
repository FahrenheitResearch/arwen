"""Measure every WRF RK output word with the production CUDA launch paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gpuwm.verify.bigstep_rk_oracle import measure_rk_parity


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--full",action="store_true",help="also measure the full rk_tendency orchestrator")
    args=parser.parse_args()
    import cupy as cp
    import gpuwm.core.moist as moist
    import gpuwm.core.dycore as dycore
    from gpuwm.core.kernels import module_source
    result={"measurements":measure_rk_parity(),"source_sha256":{},
            "device":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy_version":cp.__version__,"cuda_runtime":cp.cuda.runtime.runtimeGetVersion()}
    if args.full:
        from gpuwm.verify.bigstep_rk_oracle import measure_rk_tendency_parity
        result["rk_tendency_measurements"]=measure_rk_tendency_parity()
        result["rk_tendency_stored_theta_measurements"]=measure_rk_tendency_parity("rk-tendency-stored-theta.npz")
    for name,module in (("gpuwm/core/moist.py",moist),("gpuwm/core/dycore.py",dycore)):
        result["source_sha256"][name]=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    for name in ("held_heating","rk_bookkeeping","dycore","advection","acoustic",
                 "coriolis_map","openbc"):
        result["source_sha256"][name+".cu"]=hashlib.sha256(module_source(name).encode()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
