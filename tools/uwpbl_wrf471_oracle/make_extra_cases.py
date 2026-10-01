"""Additional full-driver columns for branches make_cases.py does not reach.

    python make_extra_cases.py OUTDIR      (writes cases-extra35.bin + index)

Four probe families on one 35-level grid, 12 columns each, 2 steps: a
surface-connected stable turbulent layer, a surface-based radiatively
driven stratus layer, weak convective layers above a stable surface, and an
elevated convective layer.  They reach caleddy's surface STL, surface SRCL,
low-energy floor and no-surface-first-CL arms, which the six regime
families leave unvisited.  Branch probes built from make_cases.py's
profiles, not qualified atmospheric analyses.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

spec=importlib.util.spec_from_file_location(
    'make_cases', Path(__file__).resolve().with_name('make_cases.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def main():
    dest=Path(sys.argv[1]); dest.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(9103042); columns=[]; metadata=[]; nk=35
    for mode in ('surface_stl','surface_srcl','weak_cl','elevated_cl'):
        for j in range(12):
            c=m.build_column(rng,'stable' if mode=='surface_stl' else 'coldpool' if mode=='surface_srcl' else 'convective',nk,14000.)
            z=c['z']-c['ht'][0]
            if mode=='surface_stl':
                c['u']=(2.+(0.025+0.01*j)*np.minimum(z,600.)).astype(np.float32)
                c['v']=np.full(nk,.5,dtype=np.float32)
                c['hfx'][:]=-2.; c['ust'][:]=.3
                c['qc'][:]=0.; c['qi'][:]=0.; c['cldfra'][:]=0.
            elif mode=='surface_srcl':
                c['qc'][:]=0.; c['qc'][0]=np.float32(0.0001+0.0002*j)
                c['qnc'][:]=0.; c['qnc'][0]=1.e8
                c['cldfra'][:]=0.; c['cldfra'][0]=1.
                c['rthratenlw'][:]=-1.e-5; c['rthratenlw'][0]=-.002
                c['hfx'][:]=-5.; c['qfx'][:]=0.; c['ust'][:]=.04
            else:
                # Alternating weak/strong inversions above a stable surface.
                c['th']+=(1.5*np.sin(z/(100.+30.*j))).astype(np.float32)
                if mode=='elevated_cl':
                    mask=z<450.
                    c['th'][mask]=(c['th'][0]+0.03*z[mask]).astype(np.float32)
                c['hfx'][:]=-5.; c['qfx'][:]=0.
            c['t']=(c['th']*c['exner']).astype(np.float32)
            c['rho']=(c['p']/(m.R_D*c['t']*(1.+.608*c['qv']))).astype(np.float32)
            columns.append(c); metadata.append(dict(family=mode,member=j))
    path=dest/'cases-extra35.bin'
    m.write_case_file(path,columns,nk,2,20.)
    (dest/'cases-index.json').write_text(json.dumps({'extra35':dict(file=path.name,nk=nk,dt=20.,nsteps=2,columns=metadata)},indent=2))
if __name__=='__main__': main()
