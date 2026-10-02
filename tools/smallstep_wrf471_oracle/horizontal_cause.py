"""Quantify the dominant momentum difference from actual fixture input words.

This is an intermediate-rounding witness, not an independent WRF reference.
The oracle answers remain those emitted by compiled WRF in horizontal.npz.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def half_geopotential_witness(path):
    f=np.float32
    with np.load(path,allow_pickle=False) as p:
        prefix="real_initial_map__"
        ph=p[prefix+"in_php"];phb=p[prefix+"in_phb"]
        k,j,i=48,4,6
        half=f(.5)*(phb[:-1]+phb[1:]+ph[:-1]+ph[1:])
        wrf=half[k,j,i]-half[k,j-1,i]
        engine=(f(.5)*((ph[k,j,i]+ph[k+1,j,i])-(ph[k,j-1,i]+ph[k+1,j-1,i]))
                +f(.5)*((phb[k,j,i]+phb[k+1,j,i])-(phb[k,j-1,i]+phb[k+1,j-1,i])))
        pp=p[prefix+"in_p_pp"];fnm=p[prefix+"in_fnm"];fnp=p[prefix+"in_fnp"]
        c1h=p[prefix+"in_c1h"];mu=p[prefix+"in_mu_pp"];rdnw=p[prefix+"in_rdnw"]
        dpn=f(.5)*(fnm[k]*(pp[k,j,i]+pp[k,j-1,i])+fnp[k]*(pp[k-1,j,i]+pp[k-1,j-1,i]))
        coefficient=rdnw[k]*(-dpn)-f(.5)*c1h[k]*(mu[j,i]+mu[j-1,i])
        return dict(location=[k,j,i],wrf_half_phi_face_difference=float(wrf),
                    engine_half_phi_face_difference=float(engine),
                    intermediate_difference=float(engine-wrf),
                    native_term4_coefficient=float(coefficient),
                    predicted_coupled_momentum_difference=float(f(.25)*f(1/3000)*(engine-wrf)*coefficient))


def reference_omega_witness(path,library_path,case_name="real_initial_map"):
    """Ask compiled WRF for the two large words its final subtraction uses."""
    from gpuwm.verify.smallstep_horizontal_oracle import load_horizontal_fixture,_pack_args
    from gpuwm.verify.smallstep_oracle import WRFOracle
    cases={name:(data,cfg) for name,data,cfg,_out,_probe in load_horizontal_fixture(path)}
    data,cfg=cases[case_name]
    oracle=WRFOracle(library_path,cfg.nx,cfg.ny,cfg.nz,
                     periodic=not cfg.open_x and not cfg.specified)
    _uv,mass=_pack_args(oracle,data,cfg)
    ref={key:(value.copy(order="F") if isinstance(value,np.ndarray) else value) for key,value in mass.items()}
    for key in ("u","v","mu","ww","t","t_1","ft","mu_tend"):
        ref[key].fill(0)
    oracle.call("advance_mu_t",**ref)
    with np.load(path,allow_pickle=False) as archive:
        mass["u"]=oracle.array(archive[case_name+"__gpu_u_pp"])
        mass["v"]=oracle.array(archive[case_name+"__gpu_v_pp"])
        gpu=archive[case_name+"__gpu_ww_pp"].copy()
        expected=archive[case_name+"__wrf_ww_pp"].copy()
    # WW_1 remains zero for this call, exposing the pre-subtraction total.
    oracle.call("advance_mu_t",**mass)
    reference=oracle.extract(ref["ww"],gpu.shape)
    total=oracle.extract(mass["ww"],gpu.shape)
    loc=np.unravel_index(np.abs(gpu-expected).argmax(),gpu.shape)
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    uloc=np.unravel_index(fp32_ulp_distance(gpu,expected).argmax(),gpu.shape)
    result=dict(case=case_name,location=list(map(int,loc)),
                wrf_total_omega=float(total[loc]),wrf_reference_omega=float(reference[loc]),
                wrf_subtracted_omega=float(np.float32(total[loc]-reference[loc])),
                fixture_wrf_omega=float(expected[loc]),engine_perturbation_only_omega=float(gpu[loc]),
                reference_word_spacing=float(abs(np.spacing(reference[loc]))),
                absolute_difference=float(abs(gpu[loc]-expected[loc])))
    result["max_ulp_point"]=dict(location=list(map(int,uloc)),
        wrf_total_omega=float(total[uloc]),wrf_reference_omega=float(reference[uloc]),
        wrf_subtracted_omega=float(np.float32(total[uloc]-reference[uloc])),
        engine_perturbation_only_omega=float(gpu[uloc]),
        reference_word_spacing=float(abs(np.spacing(reference[uloc]))))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("fixture",type=Path)
    parser.add_argument("--library",type=Path)
    parser.add_argument("--case",default="real_initial_map")
    args=parser.parse_args()
    result={"half_geopotential":half_geopotential_witness(args.fixture)}
    if args.library:
        result["reference_omega"]=reference_omega_witness(args.fixture,args.library,args.case)
    print(json.dumps(result,indent=2))
