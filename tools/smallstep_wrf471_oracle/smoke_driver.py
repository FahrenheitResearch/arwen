"""Exercise the complete native acoustic driver on the six real-state fixtures."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import cupy as cp
import numpy as np
from gpuwm.core import acoustic
from gpuwm.core.dycore import apply_emdiv_filter
from gpuwm.core.kernels import module_source
from gpuwm.verify.smallstep_oracle import load_cases
from gpuwm.verify.smallstep_vertical_oracle import make_vertical_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    results=[]
    for name,raw,metadata in load_cases():
        state,cfg=make_vertical_state(raw,dict(metadata,damp_opt=3))
        cfg=replace(cfg,moist=True,moist_cq=True,mp_physics=8,emdiv=.05)
        for field,key in (("qv","QVAPOR"),("qc","QCLOUD"),("qr","QRAIN"),
                          ("qi","QICE"),("qs","QSNOW"),("qg","QGRAUP")):
            setattr(state,field,cp.asarray(raw[key],dtype=cp.float32))
        mudf=cp.zeros((cfg.ny,cfg.nx),dtype=cp.float32)
        for substep in range(2):
            acoustic.acoustic_substep(state,cfg,.25,first=substep==0,mudf=mudf)
        apply_emdiv_filter(state,cfg,mudf,cp.zeros_like(state.mu_pp))
        cp.cuda.get_current_stream().synchronize()
        fields={key:cp.asnumpy(getattr(state,key)) for key in
                ("u_pp","v_pp","w_pp","ph_pp","mu_pp","th_pp","p_pp","al_pp","ww_pp")}
        if any(not np.all(np.isfinite(value)) for value in fields.values()):
            raise AssertionError("Nonfinite native acoustic output in "+name)
        results.append(dict(case=name,substeps=2,moist_loading=True,damp_opt=3,all_finite=True,
                            words=sum(a.size for a in fields.values()),
                            hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in fields.items()}))
    receipt=dict(cases=results,cupy=cp.__version__,
                 runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                 driver_version=cp.cuda.runtime.driverGetVersion(),
                 device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                 acoustic_source_sha256=hashlib.sha256(module_source("acoustic").encode()).hexdigest(),
                 scope="Complete unmodified native acoustic_substep driver and external divergence filter; finite-output smoke only, not trajectory parity")
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"cases":len(results),"substeps":12,"all_finite":True}))


if __name__=="__main__":
    main()
