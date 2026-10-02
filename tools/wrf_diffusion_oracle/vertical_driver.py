"""Pin all mutable outputs from the actual WRF vertical diffusion driver."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np
from unittest.mock import patch
from cases import core,pad2
from deformation_reference import reference_for_case

FIELDS=("u","v","w","theta","tke","qv","qc","qi","moist0","chem0","scalar0","tracer0",
        "nba","chem_after","thp_after","moist0_after","qv_after","qc_after","qi_after","hfx_after","qfx_after")

def reference(library,prepared,metadata,*,km_opt,isfflx):
    nx,ny,nz=(metadata[k] for k in ("nx","ny","nz"))
    shape=(nx+6,nz+1,ny+6)
    inputs=[np.array(prepared[k],dtype=np.float32,order="F",copy=True) for k in
            ("u","v","thp","theta","tke","moist","d13","d23","d33","div","kmh","kmv",
             "khv","rho","rdz","rdzw","fnm","fnp","dn","dnw","hfx","qfx","ust")]
    outputs=[np.zeros(shape,np.float32,order="F") for _ in range(5)]
    if km_opt==4:outputs[4].fill(.03125)
    outputs.append(np.zeros((*shape,4),np.float32,order="F"));outputs[5][:,:,:,0]=.125
    outputs += [np.full((*shape,1),v,np.float32,order="F") for v in (.25,.375,.5)]
    outputs += [np.full((*shape,9),.625,np.float32,order="F"),np.full((*shape,1),.75,np.float32,order="F")]
    fn=ctypes.CDLL(str(library)).oracle_vertical_driver_complete
    fn.restype=None
    fn.argtypes=[ctypes.c_int]*7+[ctypes.c_float]*2+[ctypes.c_void_p]*(len(inputs)+len(outputs))
    fn(nx,ny,nz,metadata["bx"],metadata["by"],km_opt,isfflx,.0013,.24,
       *[ctypes.c_void_p(a.ctypes.data) for a in inputs+outputs])
    result={name:core(a,nx,ny,nz,stagger) for name,a,stagger in
            zip(FIELDS[:5],outputs[:5],("x","y","z","",""))}
    for slot,name in enumerate(("moist0","qv","qc","qi")):
        result[name]=core(outputs[5][:,:,:,slot],nx,ny,nz)
    for name,a in zip(("chem0","scalar0","tracer0","chem_after"),[outputs[k] for k in (6,7,8,10)]):
        result[name]=core(a[:,:,:,0],nx,ny,nz)
    result["nba"]=np.stack([core(outputs[9][:,:,:,i],nx,ny,nz) for i in range(9)],axis=-1)
    result["thp_after"]=core(inputs[2],nx,ny,nz)
    result["moist0_after"]=core(inputs[5][:,:,:,0],nx,ny,nz)
    for slot,name in enumerate(("qv","qc","qi"),1):result[name+"_after"]=core(inputs[5][:,:,:,slot],nx,ny,nz)
    for key,slot in (("hfx",20),("qfx",21)):
        result[key+"_after"]=np.ascontiguousarray(inputs[slot][3:3+nx,3:3+ny].T)
    return result

def capture(args):
    args.output.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((args.inputs/"vertical-fixtures.json").read_text())
    manifest={"schema":"wrf471-vertical-outer-driver-v1","cases":[],"files":{}}
    for case in source_manifest["cases"]:
        meta=case["metadata"]
        source=args.inputs/case["file"]
        with np.load(source) as data:arrays={k[3:]:data[k] for k in data.files if k.startswith("in_")}
        payload={"input_"+k:v for k,v in arrays.items()}
        for km in (2,4):
            prep=reference_for_case(args.preparation,arrays,meta,km_opt=km)
            for key in ("ust","hfx","qfx"):prep[key]=pad2(arrays[key],meta["nx"],meta["ny"],meta["bx"],meta["by"])
            for key in ("kmh","kmv","khv"):payload[f"km{km}_"+key]=core(prep[key],meta["nx"],meta["ny"],meta["nz"])
            for flux in (0,1,2):
                payload.update({f"wrf_km{km}_flux{flux}_"+k:v for k,v in reference(args.library,prep,meta,km_opt=km,isfflx=flux).items()})
        payload["meta_json"]=np.array(json.dumps(meta,sort_keys=True))
        target=args.output/(case["case"]+".npz");np.savez_compressed(target,**payload)
        manifest["cases"].append({"name":case["case"],"file":target.name,"source_fixture_sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
        manifest["files"][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    for key,path in (("fortran_build",args.library),("preparation_build",args.preparation)):
        receipt=json.loads((path.parent/"build-receipt.json").read_text())
        receipt["commands"]=[[Path(arg).name if arg.startswith("/") else arg for arg in command] for command in receipt["commands"]]
        manifest[key]=receipt
    manifest["staging"]="Raw WRF outer-driver outputs versus production vertical launcher outputs, before the mixing package clears open rows"
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8", newline="\n")

def compare_case(path,*,reference_density_from=None):
    from gpuwm.verify.diffusion_oracle import word_comparison
    from vertical_gpu import vertical_driver_gpu
    with np.load(path) as data,ExitStack() as stack:
        if reference_density_from is not None:
            import cupy as cp
            import gpuwm.core.dycore as dycore
            import gpuwm.core.kernels as kernels
            from deformation_arithmetic import _edit
            with np.load(Path(reference_density_from)/("vertical-"+path.name)) as native:
                density=cp.asarray(native["metric_rho"])
            source="__device__ const float* oracle_density;\n"+kernels.module_source("smag2d")
            body="return oracle_density[I3(k,wrf_iy(q,j),wrf_ix(q,i),q.ny,q.nx)];"
            source=_edit(source,"wrf_rho",lambda original:body)
            source+='\nextern "C" __global__ void oracle_set_density(const real* rho) { oracle_density=rho; }\n'
            module=cp.RawModule(code=source,options=("-std=c++17",))
            module.get_function("oracle_set_density")((1,),(1,),(density,))
            original=kernels.get_kernel
            controlled=lambda name,symbol:module.get_function(symbol) if name=="smag2d" else original(name,symbol)
            stack.enter_context(patch.object(dycore,"get_kernel",controlled))
            stack.enter_context(patch.object(kernels,"get_kernel",controlled))
        arrays={k.removeprefix("input_"):data[k] for k in data.files if k.startswith("input_")}
        meta=json.loads(str(data["meta_json"]))
        result={}
        for km in (2,4):
            coef={k:data[f"km{km}_"+k] for k in ("kmh","kmv","khv")}
            for flux in (0,1,2):
                got=vertical_driver_gpu(arrays,meta,coef,km_opt=km,isfflx=flux,
                                        initial_tke=.03125 if km==4 else 0.)
                for key,value in (("moist0",.125),("chem0",.25),("scalar0",.375),("tracer0",.5),("chem_after",.75)):
                    got[key]=np.full_like(arrays["alt"],value)
                got["nba"]=np.full((*arrays["alt"].shape,9),.625,np.float32)
                got["moist0_after"]=np.zeros_like(arrays["alt"])
                prefix=f"km{km}_flux{flux}"
                result[prefix]={key:{**word_comparison(got[key],data[f"wrf_{prefix}_"+key]),
                    "gpu_sha256":hashlib.sha256(got[key].tobytes()).hexdigest()} for key in FIELDS}
        return result

def compare(args):
    import cupy as cp
    from gpuwm.core.kernels import module_source
    manifest=json.loads((args.folder/"manifest.json").read_text())
    result={case["name"]:compare_case(args.folder/case["file"],reference_density_from=args.reference_density_from) for case in manifest["cases"]}
    receipt={"cases":result,"cupy":cp.__version__,"device":str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
             "diagnostic_reference_density":args.reference_density_from is not None,
             "kernel_source_sha256":hashlib.sha256(module_source("smag2d").encode()).hexdigest()}
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({case:{arm:max(v["max_ulp"] for v in fields.values()) for arm,fields in outputs.items()} for case,outputs in result.items()},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    cap=sub.add_parser("capture")
    for key in ("inputs","preparation","library","output"):cap.add_argument(key,type=Path)
    comp=sub.add_parser("compare");comp.add_argument("folder",type=Path);comp.add_argument("receipt",type=Path)
    comp.add_argument("--reference-density-from",type=Path)
    a=p.parse_args();capture(a) if a.action=="capture" else compare(a)
