"""Record every retained WRF history array using the Rust NetCDF reader."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from gpuwm.netcdf_bridge import open_dataset
p=argparse.ArgumentParser();p.add_argument('--wrfout',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
frames=[]
for path in sorted(a.wrfout.rglob('wrfout_d*')):
    if not path.is_file():continue
    fields={}
    with open_dataset(path) as ds:
        for name,var in sorted(ds.variables.items()):
            var.set_auto_maskandscale(False)
            x=np.ascontiguousarray(var[:])
            r={'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
            if x.dtype.kind in 'fiu':
                r.update(min=float(x.min()),max=float(x.max()),finite=bool(np.isfinite(x).all()))
            fields[name]=r
    frames.append({'name':path.name,'fields':fields})
if not frames:raise ValueError('no WRF history frames')
a.out.write_text(json.dumps({'frames':frames},indent=2)+'\n');print('frames',len(frames),'fields',sum(len(x['fields']) for x in frames))
