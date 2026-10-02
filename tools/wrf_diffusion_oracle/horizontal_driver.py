"""Record the compiled WRF outer horizontal driver and production launch words."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cases import core
from deformation_reference import reference_for_case

FIELDS=("u","v","w","theta","tke","moist0","qv","qc","qi","chem0","scalar0","tracer0","nba")


def reference(library,prepared,meta,km):
    nx,ny,nz=[meta[k] for k in ("nx","ny","nz")]
    shape=(nx+6,nz+1,ny+6)
    theta=prepared["theta"]
    theta_pert=prepared["thp"]
    inputs=[theta_pert,theta,prepared["tke"],prepared["moist"]]
    inputs += [prepared[k] for k in ("kmh","kmv","khh","div","d11","d22","d12","d13","d23",
               "rho","rdz","rdzw","zx","zy","msfu","msfv","msft","dn","dnw","fnm","fnp")]
    outputs=[np.zeros(shape,np.float32,order="F") for _ in range(5)]
    if km==4:outputs[4].fill(.03125)
    outputs += [np.zeros((*shape,4),np.float32,order="F")]
    outputs[5][:,:,:,0]=.125
    outputs += [np.full((*shape,1),value,np.float32,order="F") for value in (.25,.375,.5)]
    outputs += [np.full((*shape,9),.625,np.float32,order="F")]
    fn=ctypes.CDLL(str(library)).oracle_horizontal_driver
    fn.restype=None
    fn.argtypes=[ctypes.c_int]*6+[ctypes.c_void_p]*len(inputs)+[ctypes.c_float]*5+[ctypes.c_void_p]*len(outputs)
    ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
    fn(nx,ny,nz,meta["bx"],meta["by"],km,*map(ptr,inputs),1/meta["dx"],1/meta["dy"],
       meta["cf1"],meta["cf2"],meta["cf3"],*map(ptr,outputs))
    result={name:core(a,nx,ny,nz,stagger) for name,a,stagger in
            zip(FIELDS[:5],outputs[:5],("x","y","z","",""))}
    result.update({name:core(outputs[5][:,:,:,slot],nx,ny,nz) for slot,name in enumerate(FIELDS[5:9])})
    result.update({name:core(a[:,:,:,0],nx,ny,nz) for name,a in zip(FIELDS[9:12],outputs[6:9])})
    result["nba"]=np.stack([core(outputs[9][:,:,:,slot],nx,ny,nz) for slot in range(9)],axis=-1)
    return result


def capture(args):
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={"schema":"wrf471-horizontal-outer-driver-v1","cases":[],"files":{}}
    for path in sorted(args.inputs.glob("deformation-*.npz")):
        with np.load(path) as source:
            arrays={k.removeprefix("input__"):source[k] for k in source.files if k.startswith("input__")}
            meta=json.loads(str(source["meta_json"]))
        du=np.diff(arrays["u"],axis=2);dv=np.diff(arrays["v"],axis=1)
        arrays["tke"]=np.asarray(.5+.01*(du*du+dv*dv),dtype=np.float32)
        meta["tke_input_note"]="Positive TKE from actual neighboring wind variance: 0.5 + 0.01*(du**2+dv**2)"
        meta["operator_inputs_note"]="Compiled WRF K and staged tensor inputs supplied identically; horizontal operator isolation"
        meta["initialization_fixture_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        for km in (2,4):
            prep=reference_for_case(args.preparation,arrays,meta,km_opt=km)
            payload={"input_"+k:v for k,v in arrays.items()}
            for k in ("kmh","kmv","khh","d11","d22","d12"):
                payload["operator_"+k]=core(prep[k],meta["nx"],meta["ny"],meta["nz"])
            for k in ("rdz","rdzw","rho","zx","zy"):
                payload["metric_"+k]=core(prep[k],meta["nx"],meta["ny"],meta["nz"],"z" if k in ("rdz","zx","zy") else "")
            payload.update({"wrf_"+k:v for k,v in reference(args.library,prep,meta,km).items()})
            case_meta={**meta,"km_opt":km}
            payload["meta_json"]=np.array(json.dumps(case_meta,sort_keys=True))
            name=path.stem.removeprefix("deformation-")+f"-km{km}"
            target=args.output/(name+".npz");np.savez_compressed(target,**payload)
            manifest["cases"].append({"name":name,"file":target.name})
            manifest["files"][target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["fortran_build"]=json.loads((args.library.parent/"build-receipt.json").read_text())
    manifest["preparation_build"]=json.loads((args.preparation.parent/"build-receipt.json").read_text())
    for build in ("fortran_build","preparation_build"):
        for command in manifest[build]["commands"]:
            for i,arg in enumerate(command):
                if arg.startswith("/"):command[i]=Path(arg).name
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8", newline="\n")


def compare_case(path,*,diagnostic=False):
    import cupy as cp
    from gpuwm.config import RunConfig
    import gpuwm.core.dycore as dycore
    from gpuwm.core.kernels import module_source
    from gpuwm.verify.diffusion_oracle import device_state,word_comparison
    with np.load(path) as data:
        arrays={k.removeprefix("input_"):data[k] for k in data.files if k.startswith("input_")}
        meta=json.loads(str(data["meta_json"]))
        state=device_state(arrays,cf=tuple(meta[k] for k in ("cf1","cf2","cf3")))
        cfg=RunConfig(nx=meta["nx"],ny=meta["ny"],nz=meta["nz"],dx=meta["dx"],dy=meta["dy"],
                      dt=1.,ztop=20000.,run_seconds=0.,open_x=bool(meta["bx"]),open_y=bool(meta["by"]),km_opt=meta["km_opt"])
        ops={k:cp.asarray(data["operator_"+k]) for k in ("kmh","kmv","khh","d11","d22","d12")}
        deformation=tuple(ops[k] for k in ("d11","d22","d12"))
        with ExitStack() as stack:
            if diagnostic:
                from horizontal_compare import reference_metric_source
                from horizontal_arithmetic import reference_flux_order
                source=reference_flux_order(reference_metric_source(module_source("smag2d")))
                module=cp.RawModule(code=source,options=("-std=c++17","--fmad=false"))
                metrics=[cp.asarray(data["metric_"+k]) for k in ("rdz","rdzw","rho","zx","zy")]
                module.get_function("oracle_set_metrics")((1,),(1,),tuple(metrics))
                stack.enter_context(patch.object(dycore,"get_kernel",lambda name,symbol:module.get_function(symbol)))
            out={}
            for name,field,stag,xk in (("u",state.u,"x",ops["kmh"]),("v",state.v,"y",ops["kmh"]),
                    ("w",state.w,"z",ops["kmv"]),("theta",state.thp,"",ops["khh"]),
                    ("qv",state.qv,"",ops["khh"]),("qc",state.qc,"",ops["khh"]),("qi",state.qi,"",ops["khh"])):
                tend=cp.zeros_like(field)
                dycore.launch_wrf_smag2d_hd(state,cfg,field,xk,tend,stagger=stag,time_t=False,
                                           full_theta=name=="theta" and not diagnostic,deformation=deformation)
                dycore._zero_open_strips(tend,cfg,1)
                out[name]=cp.asnumpy(tend)
            if meta["km_opt"]==2:
                tend=cp.zeros_like(state.tke)
                dycore.launch_wrf_smag2d_hd(state,cfg,state.tke,ops["kmh"],tend,stagger="",time_t=False)
                dycore._zero_open_strips(tend,cfg,1)
                out["tke"]=cp.asnumpy(2.*tend)
            else:out["tke"]=np.full_like(arrays["tke"],.03125)
            for name,value in (("moist0",.125),("chem0",.25),("scalar0",.375),("tracer0",.5)):
                out[name]=np.full_like(arrays["alt"],value)
            out["nba"]=np.full((*arrays["alt"].shape,9),.625,np.float32)
        return {name:{**word_comparison(out[name],data["wrf_"+name]),
                      "gpu_sha256":hashlib.sha256(out[name].tobytes()).hexdigest()} for name in FIELDS}


def compare(args):
    manifest=json.loads((args.folder/"manifest.json").read_text())
    result={case["name"]:compare_case(args.folder/case["file"],diagnostic=args.diagnostic) for case in manifest["cases"]}
    import cupy as cp
    from gpuwm.core.kernels import module_source
    receipt={"cases":result,"diagnostic":args.diagnostic,"cupy":cp.__version__,
             "device":str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
             "kernel_source_sha256":hashlib.sha256(module_source("smag2d").encode()).hexdigest()}
    args.receipt.write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({case:max(field["max_ulp"] for field in fields.values()) for case,fields in result.items()},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    cap=sub.add_parser("capture");cap.add_argument("inputs",type=Path);cap.add_argument("preparation",type=Path)
    cap.add_argument("library",type=Path);cap.add_argument("output",type=Path)
    comp=sub.add_parser("compare");comp.add_argument("folder",type=Path);comp.add_argument("receipt",type=Path)
    comp.add_argument("--diagnostic",action="store_true")
    a=p.parse_args();capture(a) if a.action=="capture" else compare(a)
