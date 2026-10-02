"""Capture and compare direct WRF horizontal diffusion fixtures."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from cases import raw_state,state_cases,pad3,core
from deformation_reference import reference_for_case

def reference_metric_source(source):
    """A tools-only intervention attributing inherited metric rounding.

    This is never used to accept a production result or build a fixture.
    """
    import re
    prefix="\n__device__ const real *oracle_rdz,*oracle_rdzw,*oracle_rho,*oracle_zx,*oracle_zy;\n"
    source=source.replace("struct WrfSmagGrid",prefix+"struct WrfSmagGrid",1)
    bodies={
        "wrf_rdzw":"k = k < 0 ? 0 : (k >= q.nz ? q.nz-1 : k); return oracle_rdzw[I3(k,wrf_iy(q,j),wrf_ix(q,i),q.ny,q.nx)];",
        "wrf_rdz":"kw = kw < 0 ? 0 : (kw > q.nz ? q.nz : kw); return oracle_rdz[I3(kw,wrf_iy(q,j),wrf_ix(q,i),q.ny,q.nx)];",
        "wrf_rho":"return oracle_rho[I3(k,wrf_iy(q,j),wrf_ix(q,i),q.ny,q.nx)];",
        "wrf_zx":"return oracle_zx[I3(kw,wrf_iy(q,j),wrf_ix(q,iface),q.ny,q.nx)];",
        "wrf_zy":"return oracle_zy[I3(kw,wrf_iy(q,jface),wrf_ix(q,i),q.ny,q.nx)];",
    }
    for name,body in bodies.items():
        match=re.search(r"real "+name+r"\([^{}]*\)\s*\{",source)
        if match is None: raise ValueError(name)
        begin=match.end();end=begin;depth=1
        while depth:
            depth += (source[end]=="{")-(source[end]=="}")
            end+=1
        source=source[:begin]+body+source[end-1:]
    return source+'''\nextern "C" __global__ void oracle_set_metrics(const real* rdz,const real* rdzw,const real* rho,const real* zx,const real* zy) {
      oracle_rdz=rdz; oracle_rdzw=rdzw; oracle_rho=rho; oracle_zx=zx; oracle_zy=zy;
    }\n'''

def reference(library, prepared, arrays, meta, field):
    nx,ny,nz=(meta[k] for k in ("nx","ny","nz"))
    inputs=[pad3(arrays[field],nx,ny,nz,meta["bx"],meta["by"])]
    inputs += [prepared[k] for k in ("kmh","kmv","khh","div","d11","d22","d12","d13","d23","tke",
                                       "rho","rdz","rdzw","zx","zy","msfu","msfv","msft","dn","dnw","fnm","fnp")]
    shape=(nx+6,nz+1,ny+6)
    outputs=[np.zeros(shape,dtype=np.float32,order="F") for _ in range(4)]
    fn=ctypes.CDLL(str(library)).oracle_horizontal
    fn.argtypes=[ctypes.c_int]*6+[ctypes.c_void_p]*len(inputs)+[ctypes.c_float]*5+[ctypes.c_void_p]*4
    fn.restype=None
    ptr=lambda a: ctypes.c_void_p(a.ctypes.data)
    fn(nx,ny,nz,meta["bx"],meta["by"],0,*map(ptr,inputs),1/meta["dx"],1/meta["dy"],
       meta["cf1"],meta["cf2"],meta["cf3"],*map(ptr,outputs))
    return {name:core(a,nx,ny,nz,stagger) for name,a,stagger in zip(("u","v","w","s"),outputs,("x","y","z",""))}

def capture(args):
    args.output.mkdir(parents=True,exist_ok=True)
    raw,metadata=raw_state(args.input,args.reader,args.output/"raw")
    manifest={"schema":"wrf471-horizontal-diffusion-v1","cases":[],"files":{}}
    state_sets=[(raw,metadata,"")]
    if args.evolved:
        from vertical_evolved import evolved_flow
        winds,flowmeta=evolved_flow(args.evolved,args.reader,args.output/"raw-evolved-flow")
        state_sets.append(({**raw,**winds},{**metadata,**flowmeta},"evolved_"))
    iterator=((prefix+name,a,m) for data,meta,prefix in state_sets for name,a,m in state_cases(data,meta))
    for name,arrays,meta in iterator:
        prepared=reference_for_case(args.deformation,arrays,meta)
        payload={"input_"+k:v for k,v in arrays.items()}
        for k in ("kmh","kmv","khh","d11","d22","d12"):
            payload["operator_"+k]=core(prepared[k],meta["nx"],meta["ny"],meta["nz"])
        for k in ("rdz","rdzw","rho","zx","zy"):
            payload["metric_"+k]=core(prepared[k],meta["nx"],meta["ny"],meta["nz"],"z" if k in ("zx","zy","rdz") else "")
        for field in ("thp","qv"):
            refs=reference(args.library,prepared,arrays,meta,field)
            for k,v in refs.items():
                payload["wrf_"+field+"_"+k]=v
        path=args.output/(name+".npz")
        np.savez_compressed(path,**payload)
        manifest["cases"].append({"name":name,"file":path.name,**meta})
        manifest["files"][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for key,library in (("fortran_build",args.library),("preparation_build",args.deformation)):
        receipt=json.loads((library.parent/"build-receipt.json").read_text())
        receipt["commands"]=[[Path(token).name if token.startswith("/") else token for token in command]
                             for command in receipt["commands"]]
        manifest[key]=receipt
    manifest["tools_sha256"]={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("build_common.py","cases.py","horizontal_wrapper.F90","horizontal_build.py",
                     "deformation_reference.py","vertical_evolved.py")}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8", newline="\n")

def compare_case(path,meta):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import launch_wrf_smag2d_hd,_zero_open_strips
    from gpuwm.verify.diffusion_oracle import device_state,word_comparison
    with np.load(path) as fixture:
        arrays={k.removeprefix("input_"):fixture[k] for k in fixture.files if k.startswith("input_")}
        state=device_state(arrays,cf=(meta["cf1"],meta["cf2"],meta["cf3"]))
        cfg=RunConfig(nx=meta["nx"],ny=meta["ny"],nz=meta["nz"],dx=meta["dx"],dy=meta["dy"],dt=12.,ztop=20000.,run_seconds=0.,
                      open_x=bool(meta["bx"]),open_y=bool(meta["by"]),km_opt=4,bl_pbl_physics=0,sf_sfclay_physics=0)
        operators={k:cp.asarray(fixture["operator_"+k]) for k in ("kmh","kmv","khh","d11","d22","d12")}
        deformation=tuple(operators[k] for k in ("d11","d22","d12"))
        measured={}
        for field in ("thp","qv"):
            for name,stag in (("u","x"),("v","y"),("w","z"),("s","")):
                f=getattr(state,field if name=="s" else name)
                tendency=cp.zeros_like(f)
                xk=operators[{"u":"kmh","v":"kmh","w":"kmv","s":"khh"}[name]]
                launch_wrf_smag2d_hd(state,cfg,f,xk,tendency,stagger=stag,time_t=False,deformation=deformation)
                _zero_open_strips(tendency,cfg,1)
                actual=cp.asnumpy(tendency)
                measured[field+"_"+name]=word_comparison(actual,fixture["wrf_"+field+"_"+name])
                measured[field+"_"+name]["gpu_sha256"]=hashlib.sha256(actual.tobytes()).hexdigest()
        return measured

def compare(args):
    if args.reference_order and not args.reference_metrics:
        raise ValueError("The interpolation-order control also requires --reference-metrics")
    from gpuwm.core.kernels import module_source
    executed_source=module_source("smag2d")
    options=("-std=c++17",)
    if args.no_fma or args.reference_metrics:
        import cupy as cp
        import gpuwm.core.dycore as dycore
        from gpuwm.core.kernels import module_source
        source=module_source("smag2d")
        if args.reference_metrics:
            source=reference_metric_source(source)
            if args.reference_order:
                from horizontal_arithmetic import reference_flux_order
                source=reference_flux_order(source)
        module=cp.RawModule(code=source,options=("-std=c++17","--fmad=false"))
        executed_source=source
        options=("-std=c++17","--fmad=false")
        original=dycore.get_kernel
        dycore.get_kernel=lambda name,symbol: module.get_function(symbol) if name=="smag2d" else original(name,symbol)
        if args.reference_metrics:
            original_compare=globals()["compare_case"]
            def diagnostic_compare(path,meta):
                with np.load(path) as fixture:
                    values=[cp.asarray(fixture["metric_"+k]) for k in ("rdz","rdzw","rho","zx","zy")]
                module.get_function("oracle_set_metrics")((1,),(1,),tuple(values))
                return original_compare(path,meta)
            globals()["compare_case"]=diagnostic_compare
    manifest=json.loads((args.output/"manifest.json").read_text())
    rows={case["name"]:compare_case(args.output/case["file"],case) for case in manifest["cases"]}
    from gpuwm.core.kernels import module_source
    import cupy as cp
    receipt={"cases":rows,"kernel_source_sha256":hashlib.sha256(module_source("smag2d").encode()).hexdigest(),
             "executed_source_sha256":hashlib.sha256(executed_source.encode()).hexdigest(),"compiler_options":list(options),
             "cupy":cp.__version__,"device":str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
             "manifest_sha256":hashlib.sha256((args.output/"manifest.json").read_bytes()).hexdigest()}
    receipt["diagnostic_no_fma"]=args.no_fma
    receipt["diagnostic_reference_metrics"]=args.reference_metrics
    receipt["diagnostic_reference_order"]=args.reference_order
    filename="reference-order-receipt.json" if args.reference_order else "reference-metrics-receipt.json" if args.reference_metrics else "no-fma-receipt.json" if args.no_fma else "gpu-receipt.json"
    (args.output/filename).write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(rows,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("capture","compare"));p.add_argument("output",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--reader",type=Path)
    p.add_argument("--library",type=Path);p.add_argument("--deformation",type=Path)
    p.add_argument("--evolved",type=Path)
    p.add_argument("--no-fma",action="store_true",help="Diagnostic only: same source and launcher with contraction disabled")
    p.add_argument("--reference-metrics",action="store_true",help="Diagnostic only: read compiled WRF metric words")
    p.add_argument("--reference-order",action="store_true",help="Diagnostic only: restore WRF scalar interpolation and flux rounding")
    args=p.parse_args()
    (capture if args.mode=="capture" else compare)(args)
