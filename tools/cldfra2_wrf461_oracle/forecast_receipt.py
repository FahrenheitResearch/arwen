"""Record full-field hashes and comparison numbers from the Rust NetCDF reader."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from gpuwm import netcdf_bridge
p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('out',type=Path);a=p.parse_args()

def inventory(directory):
 frames=[]
 for path in sorted(directory.glob('wrfout*')):
  rows={}
  with netcdf_bridge.open_dataset(path) as ds:
   for name,var in ds.variables.items():
    if var.dtype.kind not in 'fiu':continue
    var.set_auto_maskandscale(False)
    values=np.asarray(var[:])
    rows[name]=dict(shape=list(values.shape),dtype=str(values.dtype),sha256=hashlib.sha256(values.tobytes(order='C')).hexdigest())
   cf=np.asarray(ds.variables['CLDFRA'][:],dtype='f4')
   selected={n:np.asarray(ds.variables[n][:],dtype='f4') for n in ('CLDFRA','SWDOWN','OLR','T2')}
  frames.append(dict(file=path.name,fields=rows,cldfra_unique=np.unique(cf).tolist(),selected=selected))
 return frames
left=inventory(a.left);right=inventory(a.right);assert len(left)==len(right)
comparisons=[]
for l,r in zip(left,right):
 assert l['file']==r['file'];assert set(l['fields'])==set(r['fields'])
 unequal=[n for n in l['fields'] if l['fields'][n]!=r['fields'][n]]
 selected={}
 for n,x in l['selected'].items():
  y=r['selected'][n];d=np.abs(np.float64(x)-np.float64(y))
  selected[n]=dict(max_abs=float(np.max(d)),mean_abs=float(np.mean(d)),left_min=float(np.nanmin(x)),left_max=float(np.nanmax(x)),right_min=float(np.nanmin(y)),right_max=float(np.nanmax(y)))
 comparisons.append(dict(frame=l['file'],numeric_fields=len(l['fields']),unequal=unequal,selected=selected))
for arm in (left,right):
 for f in arm:del f['selected']
a.out.write_text(json.dumps(dict(reader='gpuwm.netcdf_bridge -> rw_netcdf',left=left,right=right,comparisons=comparisons),indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
