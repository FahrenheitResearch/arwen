"""Grade the entire emitted column against the separately defined reference."""
from __future__ import annotations
import argparse,csv,ctypes,hashlib,json
from pathlib import Path
import numpy as np
from gpuwm.core.urban_tables import load_urban_params
from gpuwm.core.urban_ucm import UCM_GLOBAL_LAYOUT,UCM_SWITCHES


ALIASES={n+'_urb':n for n in ('cmr','chr','cmc','chc','cmgr','chgr')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--listing',type=Path,required=True)
    p.add_argument('--csv',type=Path,required=True);p.add_argument('--tables',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--lcz',type=int,default=0)
    p.add_argument('--tree',type=int,default=1);p.add_argument('--tt',type=int,default=0)
    p.add_argument('--fvg',type=float,default=.2);p.add_argument('--switch',action='append',default=[]);p.add_argument('--fgr',type=float);p.add_argument('--gpu',action='store_true');p.add_argument('--debug-listing',type=Path);a=p.parse_args()
    import gzip
    opener=gzip.open if a.csv.suffix=='.gz' else open
    with opener(a.csv,'rt') as f:rows=list(csv.DictReader(f))
    schema_path=(Path(__file__).resolve().parents[2]/'gpuwm/data/urban/oracle/ucm480/schema.json' if a.gpu else a.listing.with_suffix('.json'))
    schema=json.loads(schema_path.read_text())
    params=load_urban_params(1,a.lcz,tbl_dir=a.tables,generation='wrf_480')
    n=len(rows);values={}
    table=np.stack([np.asarray(params.values[name.upper()+'_TBL'],np.float32) for name in schema['table']],axis=1)
    table[:,schema['table'].index('fvg')]=np.float32(a.fvg)
    glob=np.concatenate([np.atleast_1d(params.values[name.upper()])[:count] for name,count in UCM_GLOBAL_LAYOUT]).astype(np.float32)
    switches=[]
    for name in UCM_SWITCHES:
        key=name.upper()+('_DATA' if name.startswith('bound') or name in ('ch_scheme','ts_scheme') else '')
        switches.append(int(params.scalar(key)))
    for item in a.switch:
        key,value=item.split('=',1);switches[UCM_SWITCHES.index(key)]=int(value)
    if a.fgr is not None:glob[-1]=np.float32(a.fgr)
    switches=np.asarray(switches+[params.icate,a.tree,a.tt,1],np.int32)
    zero={'mh_urb','stdh_urb','lp_urb','hgt_urb','lb_urb','zo_check','lf_urb_s','z0_urb','vegfrac_in'}
    layer={'dzr','dzb','dzg'}
    for v in schema['arguments']:
        name=v['name'];alias=ALIASES.get(name,name);dtype=np.int32 if v['type'] in ('int','bool') else np.float32
        if name=='lsolar':data=np.zeros(n,dtype)
        elif name.startswith('num_'):data=np.full(n,4,dtype)
        elif name in zero:data=np.zeros(n,dtype)
        elif name=='lf_urb':data=np.zeros((4,n),dtype)
        elif name in layer:data=np.repeat(np.asarray(params.values[name.upper()],np.float32)[:,None],n,axis=1)
        elif v['layers']:
            data=np.asarray([[float(r[f'{alias}{k}_in']) for r in rows] for k in range(1,5)],dtype)
        elif alias+'_in' in rows[0]:data=np.asarray([float(r[alias+'_in']) for r in rows],dtype)
        elif alias in rows[0]:data=np.asarray([float(r[alias]) for r in rows],dtype)
        elif name=='ssgd':data=np.asarray([np.float32(.8)*np.float32(r['ssg']) for r in rows],dtype)
        elif name=='ssgq':data=values['ssg']-values['ssgd']
        else:data=np.zeros(n,dtype)
        values[name]=np.ascontiguousarray(data)
    if a.gpu:
        import cupy as cp
        from gpuwm.core.kernels import get_kernel
        device={k:cp.asarray(v) for k,v in values.items()}
        pointers=cp.asarray(np.asarray([device[v['name']].data.ptr for v in schema['arguments']],np.uint64))
        codes=cp.zeros(n,cp.int32)
        if a.debug_listing:
            from gpuwm.core.kernels import module_source,module_options
            from gpuwm.cuda_subnormals import install
            install();src=module_source('urban_ucm480')
            old=a.listing.read_text();assert src.count(old)==1
            src=src.replace(old,a.debug_listing.read_text())
            module=cp.RawModule(code=src,options=module_options('urban_ucm480'))
            kernel=module.get_function('urban480_columns')
        else:kernel=get_kernel('urban_ucm480','urban480_columns')
        kernel(((n+127)//128,),(128,),
            (pointers,cp.asarray(table),cp.asarray(glob),cp.asarray(switches),codes,np.int32(n)))
        codes=codes.get();values={k:v.get() for k,v in device.items()}
    else:
        fn=ctypes.CDLL(str(a.listing.with_suffix('.so').resolve())).urban480_host
        fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int];fn.restype=None
        pointers=np.asarray([values[v['name']].ctypes.data for v in schema['arguments']],np.uint64)
        codes=np.zeros(n,np.int32)
        fn(pointers.ctypes.data,table.ctypes.data,glob.ctypes.data,switches.ctypes.data,codes.ctypes.data,n)
        ctypes.CDLL(None).fflush(None)
    errors=[];fields={};words=0
    for v in schema['arguments']:
        name=v['name'];alias=ALIASES.get(name,name)
        keys=[f'{alias}{k}' for k in range(1,5)] if v['layers'] else [alias]
        for k,key in enumerate(keys):
            if key not in rows[0] or key+'_in' not in rows[0] and key not in ('ts','qs','sh','lh','lh_kinematic','sw','alb','lw','g','rn','psim','psih','gz1oz0','u10','v10','th2','q2','ust','znt'):continue
            want=np.asarray([float(r[key]) for r in rows],np.float32)
            got=values[name][k] if v['layers'] else values[name]
            bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32));words+=len(want)
            fields[key]={'differences':len(bad),'reference_sha256':hashlib.sha256(want.tobytes()).hexdigest(),'port_sha256':hashlib.sha256(got.tobytes()).hexdigest()}
            for row in bad:errors.append({'field':key,'row':int(row),'got':int(got.view(np.uint32)[row]),'want':int(want.view(np.uint32)[row])})
    receipt={'rows':n,'words':words,'codes':np.unique(codes).tolist(),'differing_words':sum(f['differences'] for f in fields.values()),'first_cells':errors[:100],'cells':errors,'fields':fields,'reference':str(a.csv),'listing_sha256':hashlib.sha256(a.listing.read_bytes()).hexdigest()}
    a.out.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:receipt[k] for k in ('rows','words','codes','differing_words')}))
    raise SystemExit(1 if receipt['differing_words'] or np.any(codes) else 0)


if __name__=='__main__':main()
