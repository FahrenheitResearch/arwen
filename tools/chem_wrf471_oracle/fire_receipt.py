"""Measure every float32 fixture output and optional KIND=8 differences."""
import argparse
import json
from pathlib import Path
import numpy as np
from gpuwm.verify.chem_oracle import load, ulp_table
from gpuwm.verify.chem_fire_ref import add_emiss_burn, smoke_prep, classify_fire, column_mass_change, carry_ebu, frp_diagnostic
from gpuwm.verify.chem_wetdep_ls_ref import wetdep_ls


def measure(root, kind8=None):
    results={}
    for case_name,case in load(root / "wetdep_ls").items():
        var=case["initial"].transpose(3,1,2,0).copy()
        met={n:case[n].transpose(1,2,0).copy() for n in ("qc","rho","dz","w")}
        alpha=np.array([0,1,0,0,.5,.5,0,.8,0,.8,.5,.5,.5,.5,.5,1,1,1,1],np.float32)
        for step in range(1,6):
            var,mass=wetdep_ls(var,case["rain"].T,**met,dt=case["dt"],alpha=alpha)
            results[f"wetdep_ls/{case_name}/step{step}"]=ulp_table(var,case[f"step{step}"].transpose(3,1,2,0))
            results[f"wetdep_ls/{case_name}/removed{step}"]=ulp_table(mass,case[f"removed{step}"].transpose(2,1,0))
    for case_name,case in load(root / "add_emiss_burn_gsl").items():
        chem=case["initial"].transpose(1,2,0).copy()
        vol={n:case[n].transpose(1,2,0).copy() for n in ("ebu","rho","dz")}
        plane={n:case[n].T.copy() for n in ("hwp","prev","sw","ends","type")}
        coef=np.ones((1,20),np.float32);hist=coef.copy()
        for step in range(1,7):
            if step>1:
                vol["ebu"]=carry_ebu(carry,coef,True)
            before=chem.copy()
            chem,coef,hist=add_emiss_burn(chem,**vol,coef=coef,hist=hist,hwp=plane["hwp"],
                prev=plane["prev"],sw=plane["sw"],ends=plane["ends"],fire_type=plane["type"],
                dt=case["dt"],time=(step-1)*13*3600,mode=int(case_name[-1]),minimum=case["minimum"])
            for name,out in (("chem",chem),("coef",coef),("hist",hist)):
                ref=case[f"{name}{step}"]
                ref=ref.transpose(1,2,0) if name=="chem" else ref.T
                results[f"add_emiss_burn_gsl/{case_name}/{name}{step}"]=ulp_table(out,ref)
            mass=column_mass_change(before,chem,vol["rho"],vol["dz"])
            results[f"add_emiss_burn_gsl/{case_name}/emitted{step}"]=ulp_table(mass,case[f"emitted{step}"].T)
            carry=carry_ebu(vol["ebu"],coef,False)
            results[f"add_emiss_burn_gsl/{case_name}/carry{step}"]=ulp_table(carry,case[f"carry{step}"].T[:,None,:])
            results[f"add_emiss_burn_gsl/{case_name}/frp{step}"]=ulp_table(frp_diagnostic(case["frp_mw"].T,coef),case[f"frp{step}"][None,:])
    for case_name,case in load(root / "smoke_prep_gsl").items():
        fields={n:case[n].transpose(1,2,0).copy() for n in ("t","p","qv","z","z_at_w")}
        fields.update({n:case[n].T[:,None,:].copy() for n in ("u","v")})
        fields.update({n:case[n].reshape(1,-1).copy() for n in
            ("pbl","oro","u10","v10","t2m","dpt2m","wetness","totprcp","totprcp_24hrs","swdown","snow")})
        out=smoke_prep(**fields,hour=case["hour"],method=int(case_name[6]))
        for name,value in out.items():
            results[f"smoke_prep_gsl/{case_name}/{name}"]=ulp_table(value,case[name].T)
    if kind8:
        for family,folder,names in (("add_emiss_burn_gsl","add_emiss_burn",
            tuple(f"{name}{s}" for name in ("chem","coef","hist","emitted","carry","frp") for s in range(1,7))),
            ("smoke_prep_gsl","smoke_prep",("kpbl","kpbl_thetav","uspdavg2d","windgustpot","hpbl2d","hwp"))):
            a,b=load(root / family),load(kind8 / folder)
            for case in a:
                for name in names:
                    record=ulp_table(b[case][name],a[case][name])
                    record["max_absolute_difference"]=float(np.max(np.abs(
                        b[case][name].astype(np.float64)-a[case][name].astype(np.float64))))
                    results[f"KIND8-vs4/{family}/{case}/{name}"]=record
    for case_name,case in load(root / "fire_rules_gsl").items():
        values=np.array([classify_fire(case["fractions"][i,:,0],case["latitude"][i,0],
            case["longitude"][i,0],case["emission"][i,0],case["minimum"]) for i in range(case["fire_type"].shape[0])],np.int32)
        results[f"fire_rules_gsl/{case_name}/fire_type"]=ulp_table(values,case["fire_type"][:,0])
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("--kind8",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(measure(args.root,args.kind8),indent=2)+"\n",encoding="utf-8")
