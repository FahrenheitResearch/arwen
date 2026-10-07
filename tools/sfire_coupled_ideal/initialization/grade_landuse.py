"""Grade the original complete native landuse_init against its GPU consumer."""
from pathlib import Path
import argparse
import json
import numpy as np
from tools.sfire_coupled_ideal.initialization.grade_geometry import load


def replay(root, name):
    import cupy as cp
    from gpuwm.core.sfire_ideal import initialize_native_landuse
    from tools.sfire_wrf471_oracle.fixture import words
    f=load(root,name)
    lu=f['lu'].copy()
    nodata=0
    if '/case_17_' in name:
        lu[0,0]=0.;nodata=28
    result=initialize_native_landuse(lu,f['table'].transpose(1,0,2),julday=int(f['julday']),
        cen_lat=float(f['cen_lat']),iswater=int(f['iswater']),isice=int(f['isice']),
        snowc=f['snow'],xice=f['xice'],snoalb=f['snoalb'],initial_albbck=f['initial_albbck'],
        usemonalb=bool(f['monalb']),fractional_seaice=int(f['fractional']),nodata_category=nodata)
    if nodata:
        assert int(result['ivgtyp'][0,0].item())==28
    assert result['season']==int(f['season'])
    return {key:words(cp.asnumpy(result[key]),f[key]) for key in
        ('albedo','albbck','mavail','emiss','embck','znt','z0','thc','xland','xicem')}


def grade(root,destination):
    import cupy as cp
    receipt=json.loads((Path(root)/'receipt.json').read_text())
    cases={name:replay(root,name) for name in receipt['cases']}
    entries=[g for grades in cases.values() for g in grades.values()]
    summary=dict(cases=len(cases),words=sum(g['words'] for g in entries),
        different_words=sum(g['different_words'] for g in entries),max_ulp=max(g['max_ulp'] for g in entries))
    result=dict(summary=summary,device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),cases=cases)
    Path(destination).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))
    if summary['different_words']:
        print(json.dumps({name:{k:g for k,g in grades.items() if g['different_words']} for name,grades in cases.items()}))
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');args=p.parse_args();grade(args.root,args.destination)
