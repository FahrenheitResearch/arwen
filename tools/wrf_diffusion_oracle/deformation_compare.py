"""Measure every production diffusion tensor/coefficient word against WRF."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from gpuwm.config import RunConfig
from gpuwm.verify.diffusion_oracle import device_state,word_comparison


def port_outputs(arrays,meta,*,km_opt=4,isotropic=0,contract=True,metrics=None,reference_order=False):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.kernels import module_source
    arrays=dict(arrays)
    arrays.setdefault("mup",np.subtract(arrays["mut"],arrays["mub2d"],dtype=np.float32))
    arrays.setdefault("mup0",arrays["mup"])
    state=device_state(arrays,cf=tuple(meta[k] for k in ("cf1","cf2","cf3")))
    nx,ny,nz=[int(meta[k]) for k in ("nx","ny","nz")]
    cfg=RunConfig(nx=nx,ny=ny,nz=nz,dx=meta["dx"],dy=meta["dy"],ztop=20000.,
                  dt=meta.get("dt",1.),run_seconds=0.,open_x=bool(meta["bx"]),open_y=bool(meta["by"]),
                  km_opt=km_opt,c_s=meta.get("c_s",.25),c_k=meta.get("c_k",.15),
                  mix_isotropic=isotropic,mix_upper_bound=meta.get("mix_upper_bound",.1),
                  isfflx=0 if meta.get("seed_on",1) else 1)
    km=cp.zeros_like(state.alt);kh=cp.zeros_like(km)
    out={}
    opts=("-std=c++17",) if contract else ("-std=c++17","-fmad=false")
    source=module_source("smag2d")
    metric_values=None
    if metrics is not None:
        from horizontal_compare import reference_metric_source
        source=reference_metric_source(source)
        metric_values=[cp.asarray(metrics[k]) for k in ("rdz","rdzw","rho","zx","zy")]
    if reference_order:
        from deformation_arithmetic import reference_tensor_order,reference_smag2d_coefficient_order
        source=reference_tensor_order(source)
        source=reference_smag2d_coefficient_order(source)
    with ExitStack() as stack:
        if not contract or metrics is not None:
            module=cp.RawModule(code=source,options=opts)
            if metric_values is not None:
                module.get_function("oracle_set_metrics")((1,),(1,),tuple(metric_values))
            stack.enter_context(patch.object(dycore,"get_kernel",lambda name,fun:module.get_function(fun)))
            stack.enter_context(patch.object(dycore,"calc_n2_kernel",lambda:module.get_function("wrf_calc_n2")))
        launch={4:dycore.launch_wrf_smag2d_km,2:dycore.launch_wrf_tke_km,3:dycore.launch_wrf_smag3d_km}[km_opt]
        d11,d22,d12=launch(state,cfg,km,kh,time_t=False)
        out.update(d11=cp.asnumpy(d11),d22=cp.asnumpy(d22))
        # The staged D12 is stored mass-shaped. Its redundant corner faces
        # follow the same periodic/open indexing that production wrf_d uses.
        ii=np.minimum(np.arange(nx+1),nx-1) if meta["bx"] else np.arange(nx+1)%nx
        jj=np.minimum(np.arange(ny+1),ny-1) if meta["by"] else np.arange(ny+1)%ny
        out["d12"]=np.ascontiguousarray(cp.asnumpy(d12)[:,jj][:,:,ii])
        out.update(kmh=cp.asnumpy(km),khh=cp.asnumpy(kh))
        if km_opt==4:
            out["kmv"]=out["kmh"].copy();out["khv"]=np.zeros_like(out["kmh"])
            bn=cp.zeros_like(km);dycore.launch_wrf_calc_n2(state,cfg,bn,time_t=False)
        else:
            out["kmv"]=cp.asnumpy(state.scratch(km.shape,"smag_kmv"))
            out["khv"]=cp.asnumpy(state.scratch(km.shape,"smag_khv"))
            bn=state.scratch((nz,ny,nx+1),"diff6_x").reshape(-1)[:nz*ny*nx].reshape(km.shape)
        out["bn2"]=cp.asnumpy(bn)
        out.update({key+"_after":cp.asnumpy(getattr(state,key)) for key in ("tke","qv","qc","qi")})
        probe_source=source.replace("#undef WRF_SMAG_GRID_ARGS","").replace("#undef WRF_SMAG_MAKE_GRID","")
        probe=cp.RawModule(code=probe_source+Path(__file__).with_name("deformation_probe.cu").read_text(),options=opts)
        if metric_values is not None:
            probe.get_function("oracle_set_metrics")((1,),(1,),tuple(metric_values))
        extra=[cp.zeros_like(km),cp.zeros_like(km),cp.zeros((nz+1,ny,nx+1),cp.float32),cp.zeros((nz+1,ny+1,nx),cp.float32)]
        probe.get_function("oracle_expose_deformation")(((nx+128)//128,ny+1,nz+1),(128,1,1),
              tuple(dycore._wrf_smag_grid_args(state,cfg,time_t=False)+[d11,d22]+extra+
                    [np.int32(nz),np.int32(ny),np.int32(nx),np.int32(1),np.int32(meta["bx"]),np.int32(meta["by"])]))
        out.update(zip(("d33","div","d13","d23"),[cp.asnumpy(a) for a in extra]))
    return out


def measure_fixture(fixture,*,contract=True,reference_metrics=False,reference_order=False):
    with np.load(fixture) as data:
        arrays={k.removeprefix("input__"):data[k] for k in data.files if k.startswith("input__")}
        meta=json.loads(str(data["meta_json"]))
        metrics={k.removeprefix("metric__"):data[k] for k in data.files if k.startswith("metric__")} if reference_metrics else None
        result={}
        for km,iso in ((4,0),(2,0),(2,1),(3,0),(3,1)):
            got=port_outputs(arrays,meta,km_opt=km,isotropic=iso,contract=contract,metrics=metrics,reference_order=reference_order)
            prefix=f"ref__km{km}_iso{iso}__"
            result[f"km{km}_iso{iso}"]={key.removeprefix(prefix):{
                **word_comparison(got[key.removeprefix(prefix)],data[key]),
                "gpu_sha256":hashlib.sha256(got[key.removeprefix(prefix)].tobytes()).hexdigest()}
                for key in data.files if key.startswith(prefix)}
        return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixtures",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--no-fma",action="store_true")
    p.add_argument("--reference-metrics",action="store_true")
    p.add_argument("--reference-order",action="store_true");args=p.parse_args()
    measured={f.name:measure_fixture(f,contract=not args.no_fma,reference_metrics=args.reference_metrics,reference_order=args.reference_order)
              for f in sorted(args.fixtures.glob("deformation-*.npz"))}
    import cupy as cp
    from gpuwm.core.kernels import module_source
    receipt={"cases":measured,"cupy":cp.__version__,
             "device":str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
             "kernel_source_sha256":hashlib.sha256(module_source("smag2d").encode()).hexdigest(),
             "diagnostic_no_fma":args.no_fma,"diagnostic_reference_metrics":args.reference_metrics,
             "diagnostic_reference_order":args.reference_order}
    args.output.write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({name:{k:max(vv["max_ulp"] for vv in v.values()) for k,v in case.items()}
                      for name,case in measured.items()},indent=2))
