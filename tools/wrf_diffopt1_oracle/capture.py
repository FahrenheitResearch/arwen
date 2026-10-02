"""Capture production coefficient words and exact baseline step words."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
if __package__:
    from .model_case import configuration,model_state,km2_case,word_arrays
else:
    from model_case import configuration,model_state,km2_case,word_arrays


def stats(actual,expected):
    a=actual.view("i4").astype("i8");b=expected.view("i4").astype("i8")
    a=np.where(a<0,0x80000000-a,a);b=np.where(b<0,0x80000000-b,b)
    return dict(words=int(a.size),different=int(np.count_nonzero(actual.view("u4")!=expected.view("u4"))),
                max_ulp=int(np.max(np.abs(a-b))),max_abs=float(np.max(np.abs(actual.astype("f8")-expected))))


def capture(data,output,mode,*,revert_deformation_donor=False,deformation_reference_order=False):
    import cupy as cp
    from gpuwm.core.dycore import launch_wrf_tke_km,launch_wrf_smag2d_km,step
    from gpuwm.core import kernels
    control_source=None
    if revert_deformation_donor or deformation_reference_order:
        if mode not in ("km2","all"):
            raise ValueError("The deformation diagnostic requires coordinate receipts")
        if revert_deformation_donor and deformation_reference_order:
            raise ValueError("Use one deformation attribution intervention per capture")
        original_source=kernels.module_source
        source=original_source("smag2d")
        donor="""    // cal_deform_and_div copies outer tensor faces from the adjacent
    // interior tensor. Clamping each wind separately changes the gradient.
    if (q.boundary_x) i = i <= 0 ? (q.nx > 1 ? 1 : 0)
                                    : (i >= q.nx ? q.nx - 1 : i);
    if (q.boundary_y) j = j <= 0 ? (q.ny > 1 ? 1 : 0)
                                    : (j >= q.ny ? q.ny - 1 : j);
"""
        if revert_deformation_donor:
            if source.count(donor)!=1:
                raise ValueError("The diagnostic must remove exactly the pinned D12 donor change")
            control_source=source.replace(donor,"")
        else:
            combined="""    return mm * (q.rdy * (wrf_uhat(q, k, j, i)
                         - wrf_uhat(q, k, j - 1, i)) - uslope
               + q.rdx * (wrf_vhat(q, k, j, i)
                         - wrf_vhat(q, k, j, i - 1)) - vslope);"""
            separate="""    return __fadd_rn(
        __fmul_rn(mm, __fsub_rn(__fmul_rn(q.rdy,
            __fsub_rn(wrf_uhat(q,k,j,i),wrf_uhat(q,k,j-1,i))),uslope)),
        __fmul_rn(mm, __fsub_rn(__fmul_rn(q.rdx,
            __fsub_rn(wrf_vhat(q,k,j,i),wrf_vhat(q,k,j,i-1))),vslope)));"""
            if source.count(combined)!=1:
                raise ValueError("The diagnostic must replace exactly the default D12 term association")
            control_source=source.replace(combined,separate)
        kernels.module_source=lambda name: control_source if name=="smag2d" else original_source(name)
        kernels.load_module.cache_clear()
        kernels.get_kernel.cache_clear()
    arrays,rows={},[]
    if mode=="legacy":
        from gpuwm.io.restart import write_restart
        for km in (2,4):
            cfg=configuration(km=km,diff=2,mix=True)
            state=model_state(cfg)
            for _ in range(2):step(state,cfg)
            state.elapsed_seconds=2*cfg.dt
            write_restart(output.with_name(output.name+f"-k{km}.npz"),state,cfg)
            for _ in range(2):step(state,cfg)
            for field,value in word_arrays(state).items():arrays[f"k{km}_"+field]=value
            rows.append(dict(name=f"k{km}"))
    elif mode in ("km2","all"):
        with np.load(data/"wrf471.npz") as values:
            for case in json.loads((data/"wrf471.json").read_text())["cases"]:
                get=lambda key:values[case["name"]+"_"+key]
                dev=lambda key:cp.asarray(get(key))
                if mode=="all" and case["family"]=="horizontal":
                    from gpuwm.core.dycore import launch_coordinate_horizontal
                    result=dev("seed")
                    launch_coordinate_horizontal(dev("field"),dev("km"),dev("mu"),dev("c1"),dev("c2"),
                        dev("mt"),dev("mfu"),dev("mfv"),case["dx"],case["dy"],result,
                        stagger=("","x","y","z")[case["stag"]],boundary_x=case["bx"],boundary_y=case["by"],
                        theta_initial=dev("base") if case["perturb"] else None)
                    actual=cp.asnumpy(result);arrays[case["name"]+"_tendency"]=actual
                    rows.append(dict(name=case["name"],family="horizontal",
                                     fields=dict(tendency=stats(actual,get("expected")))))
                    continue
                if mode=="all" and case["family"]=="km4":
                    nz,ny,nx=get("d11").shape
                    km=cp.zeros((nz,ny,nx),"f4");kh=cp.zeros_like(km)
                    kernels.get_kernel("diff_opt1","wrf_diff_opt1_km4")(
                        ((nx+127)//128,ny,nz),(128,1,1),
                        (dev("d11"),dev("d22"),dev("d12"),dev("mt"),np.float32(case["dx"]),
                         np.float32(case["dy"]),np.float32(.25),np.float32(1./3.),km,kh,
                         np.int32(nz),np.int32(ny),np.int32(nx),np.int32(case["bx"]),np.int32(case["by"])))
                    row=dict(name=case["name"],family="km4",fields={})
                    for field,value in (("km",km),("kh",kh)):
                        actual=cp.asnumpy(value);arrays[case["name"]+"_"+field]=actual
                        row["fields"][field]=stats(actual,get("expected_"+field))
                    rows.append(row)
                    continue
                if case["family"] not in ("km2","deform"):continue
                cfg,state=km2_case(case,values)
                km=state.scratch(state.p.shape,"smag_km");kh=state.scratch(state.p.shape,"smag_kh")
                if case["family"]=="deform":
                    tensors=launch_wrf_smag2d_km(state,cfg,km,kh,time_t=False)
                    row=dict(name=case["name"],family="deform",fields={})
                    for field,value in zip(("d11","d22","d12"),tensors):
                        actual=cp.asnumpy(value);arrays[case["name"]+"_"+field]=actual
                        row["fields"][field]=stats(actual,values[case["name"]+"_expected_"+field])
                    rows.append(row)
                    continue
                state.scratch(state.p.shape,"smag_rtke")[:]=np.float32(.125)
                launch_wrf_tke_km(state,cfg,km,kh,time_t=False)
                results=dict(km=km,kh=kh,kmv=state._scratch["smag_kmv"],khv=state._scratch["smag_khv"],tke=state.tke)
                results["bn2"]=state._scratch["diff6_x"].reshape(-1)[:state.p.size].reshape(state.p.shape)
                row=dict(name=case["name"],family="km2",fields={})
                for name,value in results.items():
                    actual=cp.asnumpy(value);arrays[case["name"]+"_"+name]=actual
                    row["fields"][name]=stats(actual,values[case["name"]+"_expected_"+name])
                assert bool(cp.all(state._scratch["smag_rtke"]==np.float32(.125))),"diff_opt=1 must not call tke_rhs"
                rows.append(row)
    else:
        for km in (2,4):
            for boundary in (False,True):
                for moist in (False,True):
                    cfg=configuration(km=km,diff=2,mix=True,boundary=boundary,moist=moist)
                    state=model_state(cfg)
                    for _ in range(3):step(state,cfg)
                    name=f"diff2_k{km}_b{int(boundary)}_m{int(moist)}"
                    for field,value in word_arrays(state).items():arrays[name+"_"+field]=value
                    rows.append(dict(name=name))
    np.savez_compressed(output.with_suffix(".npz"),**arrays)
    receipt=dict(mode=mode,device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),cases=rows,
                 archive_sha256=hashlib.sha256(output.with_suffix(".npz").read_bytes()).hexdigest())
    if mode in ("km2","all"):
        receipt["native_archive_sha256"]=hashlib.sha256((data/"wrf471.npz").read_bytes()).hexdigest()
        receipt["module_source_sha256"]={name:hashlib.sha256(kernels.module_source(name).encode()).hexdigest()
                                         for name in ("smag2d","diff_opt1")}
        if control_source is not None:
            receipt["diagnostic"]=("Only the wrf_defor12 evaluation-point donor mapping from 83fde6032 is removed"
                                   if revert_deformation_donor else
                                   "Only the default wrf_defor12 meridional/zonal terms use separately rounded WRF association")
    output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(mode=mode,cases=len(rows),arrays=len(arrays)),sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--mode",choices=("km2","all","diff2","legacy"),required=True)
    p.add_argument("--revert-deformation-donor",action="store_true",
                   help="Tools-only attribution: remove only the D12 donor evaluation-point mapping")
    p.add_argument("--deformation-reference-order",action="store_true",
                   help="Tools-only attribution: round and weight the D12 directional terms separately")
    a=p.parse_args();capture(a.data,a.output,a.mode,revert_deformation_donor=a.revert_deformation_donor,
                            deformation_reference_order=a.deformation_reference_order)
