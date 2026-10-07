"""CPU rounding-shim audit of the CUDA source, explicitly not device parity."""
import argparse,ctypes as c,json
from pathlib import Path
import numpy as np
from gpuwm.verify.chem_oracle import load,ulp_table

p=argparse.ArgumentParser();p.add_argument('library');p.add_argument('fixtures');p.add_argument('groups');p.add_argument('output');a=p.parse_args()
lib=c.CDLL(str(Path(a.library).resolve()));fn=lib.freitas_columns
fn.argtypes=[c.c_void_p]*11+[c.c_int]*4+[c.c_void_p]+[c.c_int]*2;fn.restype=None
# freitas_columns keeps each column's plume state in a workspace slice (PL_WS_ROWS x 202).
WORKSPACE=np.zeros(131*202,dtype=np.float32)
redistribute=lib.ebu_distribute;redistribute.argtypes=[c.c_void_p]*7+[c.c_int]*4;redistribute.restype=None
landuse_cache=lib.ebu_distribute_landuse;landuse_cache.argtypes=[c.c_void_p]*5+[c.c_int]*4;landuse_cache.restype=None
groups=json.loads(Path(a.groups).read_text())['groups'];params=np.zeros(20,dtype=np.float32)
params[:7]=1.e7,1.e9,2000.,5.,2.e10,.05,1.
for g,row in enumerate(groups):params[7+2*g:9+2*g]=row['heat_min'],row['heat_max'];params[16+g]=row['single_heat_pass']
report={}
for arm,name in [(1,'plumerise_wrfchem'),(0,'plumerise_frp_gsl')]:
    totals={};fails=[]
    for case,d in load(Path(a.fixtures)/name).items():
        nz=int(d['nz']); nr=len(d['ebu_in']);met=np.zeros((8,nz+1),dtype=np.float32)
        for i,key in enumerate(('t_phy','p_phy','rho','qv_in','u_phy','v_phy','z','z_at_w')): met[i,:len(d[key])]=d[key]
        prop=np.zeros(12,dtype=np.float32)
        if arm: prop[1:5]=d['mean_fct'];prop[5:9]=d['firesize']
        else: prop[0]=d['frp_inst'];prop[9:12]=d['kpbl'],d['uspdavg2d'],d['hpbl2d']
        emitted=np.ascontiguousarray(d['ebu_in']);out=np.zeros((nr,nz),dtype=np.float32)
        bounds=np.zeros(2,dtype=np.int32);diag=np.zeros(10,dtype=np.float32)
        profiles=np.zeros((7,200),dtype=np.float32);cache=np.zeros(16,dtype=np.float32)
        steps=np.zeros(8,dtype=np.int32);status=np.zeros(1,dtype=np.int32)
        arrays=(met,emitted,prop,params,out,bounds,diag,profiles,cache,steps,status)
        fn(*(v.ctypes.data for v in arrays),1,nz,nr,arm,WORKSPACE.ctypes.data,0,1)
        if status[0]: raise RuntimeError('CPU shader audit hit an injection bound refusal: the CUDA entry would write outside model levels')
        actual=dict(ebu=out.T,ztopmax=diag[1:9].reshape(4,2).T,steps=steps.reshape(4,2).T)
        for f,key in enumerate(('w','t','qv','qc','qh','qi','radius')): actual[key]=profiles[f]
        if not arm: actual.update(k_min=bounds[0],k_max=bounds[1],flam_frac=diag[0])
        columns=np.zeros(1,dtype=np.int32);cached=np.zeros((nr,nz),dtype=np.float32)
        if arm:
            landuse_cache(columns.ctypes.data,cache.ctypes.data,emitted.ctypes.data,met[7].ctypes.data,cached.ctypes.data,1,nz,1,nr)
        else:
            lower=np.asarray(bounds[0:1],dtype=np.int32);upper=np.asarray(bounds[1:2],dtype=np.int32);frac=np.asarray(diag[:1],dtype=np.float32)
            redistribute(columns.ctypes.data,lower.ctypes.data,upper.ctypes.data,frac.ctypes.data,emitted.ctypes.data,met[7].ctypes.data,cached.ctypes.data,1,nz,1,nr)
        actual['cached_ebu']=cached.T
        for key,v in actual.items():
            stat=ulp_table(v,d['ebu' if key=='cached_ebu' else key]);t=totals.setdefault(key,dict(max_ulp=0,n_nonzero=0,n=0));t['max_ulp']=max(t['max_ulp'],stat['max_ulp']);t['n_nonzero']+=stat['n_nonzero'];t['n']+=stat['n']
            ref=d['ebu' if key=='cached_ebu' else key]
            if stat['n_nonzero'] or not np.array_equal(np.asarray(v).view(np.uint32),np.asarray(ref).view(np.uint32)):
                fails.append(dict(case=case,output=key,stats=stat))
    report[name]=dict(ulp=totals,failures=fails)
Path(a.output).write_text(json.dumps(report,indent=2));print({k:len(v['failures']) for k,v in report.items()},flush=True)
if any(v['failures'] for v in report.values()):raise SystemExit(1)
