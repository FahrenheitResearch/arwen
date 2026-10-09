"""New referee comparisons, including finite-value checks absent from old table."""
import copy,json,os,pathlib,sys,traceback,hashlib
import numpy as np
import cupy as cp
L=pathlib.Path('/work/pool-sfclay-noftz-mynn-mm5-eta')
sys.path[:0]=[str(L/'src'),str(L/'src/tests')]
import _mynn_columns_gsd41 as C
import _mynn_families_gsd41 as G
from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda
meta=json.loads((L/'receipts/extra-inputs.json').read_text())
R=pathlib.Path(os.environ.get('SOL_CHECK_RECEIPTS',str(L/'receipts')))
R.mkdir(parents=True,exist_ok=True)
rows=[]
def driver(v,**kw):
    return mynn_bl_driver_cuda({k:cp.asarray(np.ascontiguousarray(x)) for k,x in v.items()},**kw)
for case in meta['cases']:
  for asis in (False,True):
    oracle=C.load(path=L/'extra'/case['name']/('asis.bin.gz' if asis else 'sq.bin.gz'))
    if case['cycling']:
        first=oracle['steps'][0][1]
        first['qke'][:]=np.float32(case['qke']);first['qc_bl'][:]=np.float32(2e-5);first['cldfra_bl'][:]=np.float32(.3)
        state=first
        for i,(init,_,out) in enumerate(oracle['steps']):
            oracle['steps'][i]=(init,state,out);state=G.advance(state,out,oracle['delt'])
    for replay in (True,False):
      row=dict(case=case['name'],asis=asis,replay=replay,ncol=oracle['ncol'],nz=oracle['nz'],nstep=len(oracle['steps']),dt=float(oracle['delt']))
      try:
        result=C.integrate(driver,asis=asis,replay=replay,to_host=cp.asnumpy,oracle=oracle,cycling=case['cycling'],bl_mynn_cloud_tendency_form='gsd_41')
        table=C.compare(result,meta['names'])
        actual_hash=hashlib.sha256()
        for _,actual,_ in result:
            for field in (*C.PROFILE_OUTPUTS,*C.COLUMN_OUTPUTS,*C.INDEX_OUTPUTS):
                actual_hash.update(np.ascontiguousarray(actual[field]).tobytes())
        row['actual_sha256']=actual_hash.hexdigest()
        finite={}
        for field in (*C.PROFILE_OUTPUTS,*C.COLUMN_OUTPUTS):
            actual=sum(int((~np.isfinite(a[field])).sum()) for _,a,e in result)
            expected=sum(int((~np.isfinite(e[field])).sum()) for _,a,e in result)
            if actual or expected:finite[field]={'actual_nonfinite':actual,'ref_nonfinite':expected}
        row.update(outputs=table,nonfinite=finite,max_ulp=max(x['ulp'] for x in table.values()),bits_differing=sum(x['bits'] for x in table.values()),values=sum(x['values'] for x in table.values()))
      except Exception as e:
        row.update(error=repr(e),traceback=traceback.format_exc())
      rows.append(row)
      print(json.dumps({k:v for k,v in row.items() if k not in ('outputs','traceback')}),flush=True)
      (R/f'extra-{os.environ.get("GPUWM_WRF_EXACT","default")}.json').write_text(json.dumps(dict(mode=os.environ.get('GPUWM_WRF_EXACT','default'),configs=rows),indent=1))
# Deliberately exercise options the GPU driver refuses. Refusal is not parity.
v=G.driver_values(C.load(False)['steps'][0][1],np.zeros((84,40),np.float32))
options=[dict(restart=True),dict(spp_pbl=1),dict(bl_mynn_cloudpdf=1),dict(bl_mynn_mixqt=1),dict(bl_mynn_edmf=0)]
refusals=[]
for opt in options:
    try:
        driver(v,initflag=1,delt=np.float32(20),bl_mynn_mixlength=2,bl_mynn_version='gsd_41',bl_mynn_cloud_tendency_form='gsd_41',**opt)
        refusals.append(dict(option=opt,result='unexpected acceptance'))
    except Exception as e:refusals.append(dict(option=opt,result=repr(e)))
(R/f'refusals-{os.environ.get("GPUWM_WRF_EXACT","default")}.json').write_text(json.dumps(refusals,indent=1))
