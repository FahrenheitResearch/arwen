"""Retain per-variable history words and new-state bounds through the Rust reader."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from gpuwm import netcdf_bridge

NEW=('TVG_URB','TT_URB','XXXVG_URB','TVGL_URB','SMG_URB','CMCG_URB','FLXHUMVG_URB','FLXHUMT_URB')


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--require-urban480',action='store_true');a=p.parse_args();rows=[]
    history = a.directory/'wrfout' if (a.directory/'wrfout').is_dir() else a.directory
    for path in sorted(history.glob('wrfout_d*')):
        fields={}
        with netcdf_bridge.open_dataset(path) as ds:
            for name,var in ds.variables.items():
                value=np.asarray(var[...])
                if value.dtype.kind in 'fiu':value=value.astype(value.dtype.newbyteorder('<'),copy=False)
                elif value.dtype.kind not in 'SU':continue
                row={'shape':list(value.shape),'dimensions':list(var.dimensions),
                     'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
                if value.dtype.kind=='f':
                    row.update(nonfinite=int(np.count_nonzero(~np.isfinite(value))),
                               minimum=float(value.min()),maximum=float(value.max()))
                fields[name]=row
            absent=[name for name in NEW if name not in fields]
            if a.require_urban480 and absent:raise ValueError(f'{path}: missing new urban fields {absent}')
            if a.require_urban480 and any(fields[n]['nonfinite'] for n in NEW):raise ValueError(f'{path}: new urban state contains nonfinite words')
        rows.append({'file':path.name,'bytes':path.stat().st_size,'fields':fields})
    if not rows:raise ValueError('no history files found')
    a.receipt.write_text(json.dumps({'frames':rows,'field_hashes':sum(len(r['fields']) for r in rows),
                                   'reader':'gpuwm.netcdf_bridge Rust backend','word_policy':'original numeric kind and precision, little endian'},indent=2))
    print(json.dumps({'frames':len(rows),'field_hashes':sum(len(r['fields']) for r in rows)}))


if __name__=='__main__':main()
