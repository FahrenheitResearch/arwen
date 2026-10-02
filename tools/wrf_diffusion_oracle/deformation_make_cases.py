"""Record all compiled WRF tensor and coefficient outputs for real-state cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cases import raw_state, state_cases, core
from deformation_reference import reference_for_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source");p.add_argument("reader");p.add_argument("library")
    p.add_argument("output",type=Path)
    p.add_argument("--evolved",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    raw,metadata=raw_state(args.source,args.reader,args.output/"raw")
    metadata["p_top"]=float(raw["P_TOP"])
    manifest={"wrf_release":"4.7.1","cases":[],"raw_state":metadata}
    sets=[(raw,metadata,"")]
    if args.evolved is not None:
        from vertical_evolved import evolved_flow
        winds,flow_meta=evolved_flow(args.evolved,args.reader,args.output/"raw-flow")
        sets.append(({**raw,**winds},{**metadata,**flow_meta},"evolved_"))
    cases=((prefix+name,arrays,meta) for fields,info,prefix in sets
           for name,arrays,meta in state_cases(fields,info))
    for name,arrays,meta in cases:
        arrays["tke"]=np.full_like(arrays["alt"],.5)
        if name.endswith("zero_flow"):
            arrays["tke"].fill(1.e-20 if name.endswith("near_zero_flow") else 0.)
        meta.update(dt=1.,c_s=.25,c_k=.15,mix_upper_bound=.1,seed_on=1)
        saved={"input__"+k:v for k,v in arrays.items()}
        saved["meta_json"]=np.array(json.dumps(meta,sort_keys=True))
        for km,isotropic in ((4,0),(2,0),(2,1),(3,0),(3,1)):
            ref=reference_for_case(args.library,arrays,meta,km_opt=km,isotropic=isotropic)
            if km==4:
                for key in ("rdz","rdzw","rho","zx","zy"):
                    saved["metric__"+key]=core(ref[key],meta["nx"],meta["ny"],meta["nz"],
                                               "z" if key in ("rdz","zx","zy") else "")
            keys=("div","d11","d22","d33","d12","d13","d23") if km==4 else ()
            for key in (*keys,"kmh","kmv","khh","khv","bn2"):
                nx,ny,nz=[meta[k] for k in ("nx","ny","nz")]
                if key=="d13":
                    value=np.ascontiguousarray(ref[key][3:3+nx+1,:nz+1,3:3+ny].transpose(1,2,0))
                elif key=="d23":
                    value=np.ascontiguousarray(ref[key][3:3+nx,:nz+1,3:3+ny+1].transpose(1,2,0))
                elif key=="d12":
                    value=np.ascontiguousarray(ref[key][3:3+nx+1,:nz,3:3+ny+1].transpose(1,2,0))
                else:
                    value=core(ref[key],nx,ny,nz)
                saved[f"ref__km{km}_iso{isotropic}__{key}"]=value
        fixture=args.output/f"deformation-{name}.npz"
        np.savez_compressed(fixture,**saved)
        manifest["cases"].append({"file":fixture.name,"sha256":hashlib.sha256(fixture.read_bytes()).hexdigest(),
                                  "name":name,"words":sum(v.size for k,v in saved.items() if k.startswith("ref__"))})
        print(name,flush=True)
    (args.output/"deformation-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8", newline="\n")


if __name__=="__main__":main()
