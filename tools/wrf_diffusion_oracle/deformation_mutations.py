"""Capture and compare calculate_km_kh's mutable TKE and moisture words."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cases import core
from deformation_reference import reference_for_case

ARMS=((4,0,1),(2,0,1),(2,1,1),(2,0,0),(2,1,0),(3,0,1),(3,1,1))
FIELDS=("tke_after","qv_after","qc_after","qi_after","moist0_after")

def capture(args):
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={"schema":"wrf471-km-mutable-outputs-v1","cases":[],"files":{}}
    for path in sorted(args.inputs.glob("deformation-*.npz")):
        with np.load(path) as source:
            arrays={k.removeprefix("input__"):source[k] for k in source.files if k.startswith("input__")}
            metadata=json.loads(str(source["meta_json"]))
        payload={}
        nx,ny,nz=(metadata[k] for k in ("nx","ny","nz"))
        for km,iso,seed in ARMS:
            meta={**metadata,"seed_on":seed}
            ref=reference_for_case(args.library,arrays,meta,km_opt=km,isotropic=iso)
            prefix=f"km{km}_iso{iso}_seed{seed}_"
            payload[prefix+"tke_after"]=core(ref["tke"],nx,ny,nz)
            for slot,key in enumerate(("moist0","qv","qc","qi")):
                payload[prefix+key+"_after"]=core(ref["moist"][:,:,:,slot],nx,ny,nz)
        target=args.output/path.name;np.savez_compressed(target,**payload)
        manifest["cases"].append({"file":target.name,"source_fixture_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest["files"][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    receipt=json.loads((args.library.parent/"build-receipt.json").read_text())
    receipt["commands"]=[[Path(arg).name if arg.startswith("/") else arg for arg in command] for command in receipt["commands"]]
    manifest["fortran_build"]=receipt
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8", newline="\n")

def compare_case(source,path):
    from deformation_compare import port_outputs
    from gpuwm.verify.diffusion_oracle import word_comparison
    with np.load(source) as data,np.load(path) as expected:
        arrays={k.removeprefix("input__"):data[k] for k in data.files if k.startswith("input__")}
        metadata=json.loads(str(data["meta_json"]))
        result={}
        for km,iso,seed in ARMS:
            meta={**metadata,"seed_on":seed}
            got=port_outputs(arrays,meta,km_opt=km,isotropic=iso)
            got["moist0_after"]=np.zeros_like(arrays["alt"])
            prefix=f"km{km}_iso{iso}_seed{seed}"
            result[prefix]={key:{**word_comparison(got[key],expected[prefix+"_"+key]),
                "gpu_sha256":hashlib.sha256(got[key].tobytes()).hexdigest()} for key in FIELDS}
        return result

def compare(args):
    import cupy as cp
    from gpuwm.core.kernels import module_source
    manifest=json.loads((args.folder/"manifest.json").read_text())
    result={case["file"]:compare_case(args.inputs/case["file"],args.folder/case["file"]) for case in manifest["cases"]}
    receipt={"cases":result,"cupy":cp.__version__,"device":str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
             "kernel_source_sha256":hashlib.sha256(module_source("smag2d").encode()).hexdigest()}
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({case:{arm:max(v["max_ulp"] for v in fields.values()) for arm,fields in outputs.items()} for case,outputs in result.items()},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    cap=sub.add_parser("capture")
    for key in ("inputs","library","output"):cap.add_argument(key,type=Path)
    comp=sub.add_parser("compare")
    for key in ("inputs","folder","receipt"):comp.add_argument(key,type=Path)
    a=p.parse_args();capture(a) if a.action=="capture" else compare(a)
