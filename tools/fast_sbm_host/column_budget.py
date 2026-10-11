"""Sustained kinematic FAST_SBM column check, with a physical water ledger.

No dycore or forecast qualification is implied. The initial state is a
pinned WRF oracle tile; subsequent calls evolve the entire state in place.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from . import host as H, replay as RP
from tools.fast_sbm_wrf480_oracle import fixtures as fx


def run(family="kc1", hours=3.0, flags=None):
    case = RP.Case(family,"o64")
    if flags is None: flags = H.flags_corrected(hail=0 if family=="kc2g" else 1)
    host = RP.host_for("o64",flags)
    _, pre, _ = next(fx.iter_replay(f"{family}_replay_o64"))
    state = case.state(pre.fields)
    t = case.tile
    ti = slice(t.i(t.its),t.i(t.ite)+1)
    tj = slice(t.j(t.jts),t.j(t.jte)+1)
    sl=(ti,slice(None,t.kte),tj)
    weight=state["rho"][sl]*state["dz8w"][sl]
    def water():
        q=state["qv"][sl]+np.sum(state["chem"][sl+(slice(1,100),)],axis=3)
        return float(np.sum(weight*q))
    def precip(): return float(np.sum(state["rainnc"][ti,tj]))
    initial=water()+precip()
    steps=int(round(hours*3600/12))
    rows=[]; ledger=np.zeros(10); point_residual=0.; fall_residual=0.
    started=time.perf_counter()
    for step in range(1,steps+1):
        status,tally=host.call(step,12.,1000.,1000.,t.dims(),state,ledger=ledger)
        point_residual+=ledger[2]-ledger[0]
        fall_residual+=ledger[3]+ledger[4]-ledger[2]-ledger[1]
        if status: raise RuntimeError(f"FAST_SBM stopped at source line {status}, step {step}")
        if step==1 or step%100==0 or step==steps:
            atm,rain=water(),precip()
            rows.append(dict(step=step,time_s=step*12,water_kg_m2_sum=atm,
                  precip_kg_m2_sum=rain,residual_kg_m2_sum=atm+rain-initial,
                  clip_mass_unweighted=tally["clip_mass"],point_residual=point_residual,fall_residual=fall_residual))
    duration=time.perf_counter()-started
    finite=all(np.all(np.isfinite(a)) for a in state.values())
    return dict(family=family,precision="float64",preset="corrected",dt_s=12.,
          steps=steps,simulated_hours=steps*12/3600,initial_water_plus_precip=initial,
          elapsed_s=duration,ms_per_call=1000*duration/steps,all_fields_finite=finite,
          point_processes=dict(zip(("nucleation","condensation","collision","phase_change","breakup"),ledger[5:].tolist())),
          relative_residual=rows[-1]["residual_kg_m2_sum"]/initial,checkpoints=rows,
          scope="Kinematic columns with fixed thermodynamic geometry and winds, no external source tendencies; physical rho*dz ledger, no dycore")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--family",default="kc1",choices=("kc1","kc2","kc2g")); p.add_argument("--hours",type=float,default=3); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); result=run(a.family,a.hours); a.out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
