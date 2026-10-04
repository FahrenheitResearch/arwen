"""Compare every persisted lake word and output against native WRF Fortran."""
from __future__ import annotations
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np

FP = ctypes.POINTER(ctypes.c_float)
IP = ctypes.POINTER(ctypes.c_int)


def lib(path):
    dll = ctypes.CDLL(str(path))
    dll.lake_init_columns.argtypes = [ctypes.c_int,FP,FP,FP,ctypes.c_int,ctypes.c_int,ctypes.c_float,IP]
    dll.lake_step_columns.argtypes = [ctypes.c_int,FP,FP,FP,FP,ctypes.c_float,IP]
    return dll


def ptr(a):
    return a.ctypes.data_as(IP if a.dtype==np.int32 else FP)


def inputs():
    # Exercise all snow-layer counts, shallow/deep water, frozen and thawing
    # columns, and the WRF soil-14 remapping. Distinct physical inputs are
    # generated independently of the transcription.
    snow = np.array([0,0,0,4,7,16,30,50,85,150,0,0],np.float32)
    n = len(snow)
    seed = np.zeros((5,n),np.float32)
    seed[0] = [1,3,8,14,7,2,11,18,19,8,4,6]
    seed[1] = [1,5,20,50,100,3,10,25,50,100,2,60]
    seed[2] = [295,288,277,270,268,266,271,272,263,269,274,280]
    seed[3] = snow
    seed[4] = [0,0,0.5,0.51,1,0.2,0.7,0.8,1,0.49,0,0]
    forc = np.zeros((13,n),np.float32)
    forc[0] = seed[2]+3
    forc[1] = 98000
    forc[2] = 97400
    forc[3] = 60
    forc[4] = [0.014,0.009,0.005,0.002,0.002,0.001,0.003,0.003,0.001,0.002,0.004,0.008]
    forc[5] = [4,7,1,3,7,2,5,2,8,4,0,0.02]
    forc[6] = 1
    forc[6,10:] = [0,0.02]
    forc[7] = 300
    forc[8] = 0.98
    forc[9] = [0,0,0,0.1,0.2,0.3,0.4,0.5,0.01,0,0.2,0]
    forc[10] = [500,300,0,50,20,0,100,40,0,25,300,700]
    forc[11] = 0.08
    forc[12] = [45,38,55,60,40,50,-30,20,65,48,0,30]
    return seed,forc


def metric(a,b):
    ai,bi = a.view(np.int32).astype(np.int64),b.view(np.int32).astype(np.int64)
    ai = np.where(ai<0,-0x80000000-ai,ai)
    bi = np.where(bi<0,-0x80000000-bi,bi)
    finite=np.isfinite(a)&np.isfinite(b)
    d=np.abs(ai-bi)
    idx=np.unravel_index(np.argmax(d),d.shape)
    return dict(max_ulp=int(d.max()),differing=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),words=int(a.size),
                max_abs=float(np.max(np.abs(a-b))),finite=bool(finite.all()),
                worst=list(map(int,idx)),reference=float(a[idx]),actual=float(b[idx]))


def run(build,steps=20):
    libraries=[lib(build/"lake_fortran.so"),lib(build/"lake_cpp.so")]
    seed,forcing=inputs();n=seed.shape[1]
    states=[np.zeros((131,n),np.float32) for _ in libraries]
    statics=[np.zeros((71,n),np.float32) for _ in libraries]
    outputs=[np.zeros((9,n),np.float32) for _ in libraries]
    errors=[np.zeros(n,np.int32) for _ in libraries]
    for dll,state,static,err in zip(libraries,states,statics,errors):
        dll.lake_init_columns(n,ptr(seed),ptr(state),ptr(static),1,1,50.,ptr(err))
        assert not err.any(),err
    initial_state=states[0].copy()
    trace_state=[];trace_output=[]
    report={"initial_state":metric(*states),"static":metric(*statics),"steps":[],"init_controls":[]}
    for step in range(steps):
        for dll,state,static,out,err in zip(libraries,states,statics,outputs,errors):
            dll.lake_step_columns(n,ptr(forcing),ptr(state),ptr(static),ptr(out),30.,ptr(err))
            assert not err.any(),err
        report["steps"].append({"step":step+1,"state":metric(*states),"outputs":metric(*outputs)})
        trace_state.append(states[0].copy());trace_output.append(outputs[0].copy())
    np.savez_compressed(build/"lake_columns.npz",seed=seed,forcing=forcing,initial_state=initial_state,
             reference_state=states[0],reference_static=statics[0],reference_output=outputs[0],
             trace_state=np.stack(trace_state),trace_output=np.stack(trace_output))
    control_state=[];control_static=[]
    for default in (50.,0.,-1.):
        for dll,state,static,err in zip(libraries,states,statics,errors):
            dll.lake_init_columns(n,ptr(seed),ptr(state),ptr(static),0,0,default,ptr(err))
            assert not err.any(),err
        report["init_controls"].append({"use_lakedepth":0,"lakedepth_default":default,
            "state":metric(*states),"static":metric(*statics),"depth":statics[0][0].tolist()})
        control_state.append(states[0].copy());control_static.append(statics[0].copy())
    np.savez_compressed(build/"lake_initialization.npz",seed=seed,defaults=np.asarray([50.,0.,-1.],np.float32),
                        reference_state=np.stack(control_state),reference_static=np.stack(control_static))
    return report


def run_cuda(build, kernels, steps=300):
    import cupy as cp
    from gpuwm.core.kernels import get_kernel, load_module, module_options, module_source
    dll=lib(build/"lake_fortran.so")
    init=get_kernel("lake","lake_init_columns")
    advance=get_kernel("lake","lake_step_columns")
    seed,forcing=inputs();n=seed.shape[1]
    state=np.zeros((131,n),np.float32);static=np.zeros((71,n),np.float32)
    out=np.zeros((9,n),np.float32);err=np.zeros(n,np.int32)
    ds,df=cp.asarray(seed),cp.asarray(forcing)
    dc=cp.zeros_like(state);dt=cp.zeros_like(static);do=cp.zeros_like(out);de=cp.zeros_like(err)
    dll.lake_init_columns(n,ptr(seed),ptr(state),ptr(static),1,1,50.,ptr(err))
    init((1,),(32,),(n,ds,dc,dt,1,1,np.float32(50),de))
    assert not cp.asnumpy(de).any()
    report={"device":cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "init_attributes":init.attributes,"step_attributes":advance.attributes,
            "initial_state":metric(state,cp.asnumpy(dc)),"static":metric(static,cp.asnumpy(dt)),"steps":[]}
    for step in range(steps):
        dll.lake_step_columns(n,ptr(forcing),ptr(state),ptr(static),ptr(out),30.,ptr(err))
        advance((1,),(32,),(n,df,dc,dt,do,np.float32(30),de))
        assert not cp.asnumpy(de).any()
        report["steps"].append({"step":step+1,"state":metric(state,cp.asnumpy(dc)),"outputs":metric(out,cp.asnumpy(do))})
    return report


def run_multi(build,kernels,reference,device_count=2):
    import cupy as cp
    from gpuwm.core.kernels import get_kernel, module_options, module_source
    oracle=np.load(reference)
    if cp.cuda.runtime.getDeviceCount()<device_count:
        raise RuntimeError("Requested lake decomposition proof exceeds visible GPUs")
    import hashlib
    report={"device_count":device_count,"ranks":[],"steps":len(oracle["trace_state"]),
            "source_sha256":hashlib.sha256(module_source("lake").encode()).hexdigest(),
            "options":module_options("lake")}
    states=[];outputs=[];streams=[];kernels_device=[]
    for device in range(device_count):
        with cp.cuda.Device(device):
            init=get_kernel("lake","lake_init_columns");step=get_kernel("lake","lake_step_columns")
            selected=np.arange(device,oracle["seed"].shape[1],device_count)
            n=len(selected)
            seed=cp.asarray(np.ascontiguousarray(oracle["seed"][:,selected]));forcing=cp.asarray(np.ascontiguousarray(oracle["forcing"][:,selected]))
            state=cp.zeros((131,n),cp.float32);static=cp.zeros((71,n),cp.float32);output=cp.zeros((9,n),cp.float32);err=cp.zeros(n,cp.int32)
            init((1,),(32,),(n,seed,state,static,1,1,np.float32(50),err))
            cp.cuda.get_current_stream().synchronize()
            assert not cp.asnumpy(err).any()
            rank={"device":device,"name":cp.cuda.runtime.getDeviceProperties(device)["name"].decode(),
                  "init_attributes":init.attributes,"step_attributes":step.attributes,
                  "initial_state":metric(oracle["initial_state"][:,selected],cp.asnumpy(state)),
                  "static":metric(oracle["reference_static"][:,selected],cp.asnumpy(static)),"trace":[]}
            report["ranks"].append(rank)
            states.append((n,selected,state,static,forcing,output,err))
            kernels_device.append(step)
    for iteration in range(report["steps"]):
        for device,(n,selected,state,static,forcing,output,err) in enumerate(states):
            with cp.cuda.Device(device):
                kernels_device[device]((1,),(32,),(n,forcing,state,static,output,np.float32(30),err))
        for device,(n,selected,state,static,forcing,output,err) in enumerate(states):
            with cp.cuda.Device(device):
                assert not cp.asnumpy(err).any()
                report["ranks"][device]["trace"].append({"step":iteration+1,
                    "state":metric(oracle["trace_state"][iteration][:,selected],cp.asnumpy(state)),
                    "output":metric(oracle["trace_output"][iteration][:,selected],cp.asnumpy(output))})
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("build",type=Path);p.add_argument("--steps",type=int,default=20);p.add_argument("--cuda",type=Path);p.add_argument("--multi-reference",type=Path);p.add_argument("--devices",type=int,default=2)
    args=p.parse_args()
    if args.multi_reference:
        result=run_multi(args.build,args.cuda,args.multi_reference,args.devices)
        (args.build/"comparison-multi.json").write_text(json.dumps(result,indent=2)+"\n")
        for rank in result["ranks"]:
            trace=rank.pop("trace")
            rank["max_state_ulp"]=max(t["state"]["max_ulp"] for t in trace)
            rank["max_output_ulp"]=max(t["output"]["max_ulp"] for t in trace)
        print(json.dumps(result,indent=2));raise SystemExit(0)
    result=run_cuda(args.build,args.cuda,args.steps) if args.cuda else run(args.build,args.steps)
    (args.build/("comparison-cuda.json" if args.cuda else "comparison.json")).write_text(json.dumps(result,indent=2)+"\n")
    summary={k:v for k,v in result.items() if k!="steps"}
    summary["steps"]=len(result["steps"])
    summary["max_state_ulp"]=max(row["state"]["max_ulp"] for row in result["steps"])
    summary["max_output_ulp"]=max(row["outputs"]["max_ulp"] for row in result["steps"])
    print(json.dumps(summary,indent=2))
