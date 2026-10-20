"""Compare the newly produced whole-forecast history files, then retain receipts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('driver',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists()
def files(root):return {x.relative_to(root).as_posix():x for x in (root/'wrfout').rglob('wrfout_*') if x.is_file() and x.suffix!='.json'}
l,r=files(a.base),files(a.driver);assert l and l.keys()==r.keys(),(list(l),list(r))
result={'base':str(a.base),'driver':str(a.driver),'files':[],'typed_words':0,'different_words':0,'schema_differences':[]}
for name in sorted(l):
 row={'name':name,'file_sha256':{},'variables':[]}
 for arm,path in [('base',l[name]),('driver',r[name])]:row['file_sha256'][arm]=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
 with Dataset(l[name]) as x,Dataset(r[name]) as y:
  x.set_auto_maskandscale(False);y.set_auto_maskandscale(False)
  if x.variables.keys()!=y.variables.keys():result['schema_differences'].append(name+': variables')
  if {k:len(v) for k,v in x.dimensions.items()}!={k:len(v) for k,v in y.dimensions.items()}:result['schema_differences'].append(name+': dimensions')
  for key in sorted(set(x.variables)&set(y.variables)):
   u=np.ascontiguousarray(x[key][:]);v=np.ascontiguousarray(y[key][:])
   item={'name':key,'shape':list(u.shape),'dtype':str(u.dtype),'words':int(u.size),'base_sha256':hashlib.sha256(u.tobytes()).hexdigest(),'driver_sha256':hashlib.sha256(v.tobytes()).hexdigest()}
   if u.shape!=v.shape or u.dtype!=v.dtype:
    result['schema_differences'].append(name+': '+key);item['schema_mismatch']=True
   else:
    mask=np.any(u.view(np.uint8).reshape(-1,u.dtype.itemsize)!=v.view(np.uint8).reshape(-1,v.dtype.itemsize),axis=1)
    item['different_words']=int(np.count_nonzero(mask));result['typed_words']+=u.size;result['different_words']+=item['different_words']
    if u.dtype.kind in 'fiu':
     uf=u.astype(np.float64);vf=v.astype(np.float64);finite=np.isfinite(uf)&np.isfinite(vf)
     item['nonfinite_base']=int(np.count_nonzero(~np.isfinite(uf)));item['nonfinite_driver']=int(np.count_nonzero(~np.isfinite(vf)))
     if np.any(finite):
      d=uf[finite]-vf[finite];item['max_abs']=float(np.max(np.abs(d)));item['rms']=float(np.sqrt(np.mean(d*d)))
   row['variables'].append(item)
 result['files'].append(row)
result['status']='PASS' if not result['schema_differences'] and result['different_words']==0 else 'FAIL'
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},default=int))
