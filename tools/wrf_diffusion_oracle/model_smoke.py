"""One real-input RK3 step through the changed diffusion paths."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def run(source,output):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.ingest.wrfinput import read_wrfinput,restore_domain_state
    from gpuwm.core.dycore import step,close_periodic_alias
    from gpuwm.core.kernels import module_source
    nx=ny=150;nz=49
    input_config=RunConfig(nx=nx,ny=ny,nz=nz,dx=3000.,dy=3000.,ztop=24000.,dt=1.,run_seconds=1.,
                          moist=True,mp_physics=8,sf_surface_physics=2,terrain_opt=1)
    dims={"west_east":nx,"west_east_stag":nx+1,"south_north":ny,"south_north_stag":ny+1,
          "bottom_top":nz,"bottom_top_stag":nz+1,"soil_layers_stag":4}
    restored=read_wrfinput(source,expected_dimensions=dims,cfg=input_config)
    rows=[]
    for km in (2,4):
        cfg=replace(input_config,km_opt=km,mp_physics=0,sf_surface_physics=0,sf_sfclay_physics=0,
                    bl_pbl_physics=0,ra_lw_physics=0,ra_sw_physics=0,cu_physics=0,
                    open_x=False,open_y=False,diff_6th_opt=2,diff_6th_factor=.12,
                    isfflx=0,tke_drag_coefficient=0.,tke_heat_flux=0.)
        state=restore_domain_state(restored,cfg)
        close_periodic_alias(state,cfg)
        if km==2:
            state.tke.fill(cp.float32(.5))
            state.tke0[...] = state.tke
        before=hashlib.sha256(cp.asnumpy(state.u).tobytes()).hexdigest()
        step(state,cfg)
        arrays={k:cp.asnumpy(getattr(state,k)) for k in ("u","v","w","thp","php","mup","p","alt","qv")}
        for k,a in arrays.items():
            assert np.isfinite(a).all(),(km,k)
        assert arrays["p"].min()>0.
        assert arrays["alt"].min()>0.
        after=hashlib.sha256(arrays["u"].tobytes()).hexdigest()
        assert before!=after
        rows.append({"km_opt":km,"steps":1,"dt":1.,"shape":[nz,ny,nx],"diff_6th_opt":2,"boundaries":"periodic operator probe",
                     "finite":True,"minimum_pressure_pa":float(arrays["p"].min()),
                     "maximum_abs_w_ms":float(np.abs(arrays["w"]).max()),
                     "initial_u_sha256":before,"final_u_sha256":after,
                     "final_state_sha256":hashlib.sha256(b"".join(a.tobytes() for a in arrays.values())).hexdigest()})
        del state
        cp.get_default_memory_pool().free_all_blocks()
    receipt={"source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),"rows":rows,
             "kernel_sha256":{k:hashlib.sha256(module_source(k).encode()).hexdigest() for k in ("smag2d","diff6","diff6_seam")},
             "scope":"One real initial state RK3 step with diffusion and sixth-order filtering; other physics disabled. No forecast-skill claim."}
    Path(output).write_text(json.dumps(receipt,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(rows,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();run(a.source,a.output)
