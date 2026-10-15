"""Small receipts for an actual forecast. NetCDF decoding is the Rust reader."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from gpuwm import netcdf_bridge

def summarize(root: Path):
    files=sorted(p for p in root.rglob('wrfout_d01*') if p.is_file() and p.suffix not in ('.json','.png','.log'))
    if not files:
        raise RuntimeError(f'no d01 history files under {root}')
    report={'files': [], 'nonfinite': 0, 'masked': 0}
    for path in files:
        with netcdf_bridge.Dataset(path) as ds:
            row={'path':str(path), 'attributes':{k:ds.getncattr(k) for k in
                 ('DX','DY','MP_PHYSICS') if k in ds.ncattrs()}, 'fields':{}}
            names=[n for n,v in ds.variables.items() if not v.is_character]
            ds.prefetch(names)
            for name in names:
                values=np.asanyarray(ds.variables[name][:])
                mask=int(np.ma.getmaskarray(values).sum())
                values=np.asarray(np.ma.getdata(values))
                bad=int((~np.isfinite(values)).sum())
                canonical=np.ascontiguousarray(values, dtype=ds.variables[name].dtype)
                row['fields'][name]={'shape':list(values.shape),'words':int(values.size),
                  'nonfinite':bad,'masked':mask,'min':float(values.min()),
                  'max':float(values.max()),'sha256':hashlib.sha256(canonical.tobytes()).hexdigest()}
                report['nonfinite']+=bad; report['masked']+=mask
            if 'Times' in ds.variables:
                row['times']=[''.join(v.decode() if isinstance(v,bytes) else str(v) for v in line)
                              for line in ds.variables['Times'][:]]
            report['files'].append(row)
    times=sorted({time for row in report['files'] for time in row.get('times',[])})
    report['times']=times
    report['elapsed_model_seconds']=(datetime.strptime(times[-1],'%Y-%m-%d_%H:%M:%S')
          - datetime.strptime(times[0],'%Y-%m-%d_%H:%M:%S')).total_seconds() if len(times)>1 else 0
    report['finite']=report['nonfinite']==0 and report['masked']==0
    report['all_12km_wsm3']=all(float(r['attributes'].get('DX',0))==12000
          and float(r['attributes'].get('DY',0))==12000
          and int(r['attributes'].get('MP_PHYSICS',-1))==3 for r in report['files'])
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('out',type=Path)
    args=p.parse_args(); result=summarize(args.root)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    return 0 if result['finite'] and result['all_12km_wsm3'] and result['elapsed_model_seconds']==600 else 1
if __name__=='__main__':
    raise SystemExit(main())
