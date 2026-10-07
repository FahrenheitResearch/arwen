"""Grade compiled native units/geometry against the actual GPU composition."""
from pathlib import Path
import hashlib
import json
import sys

import cupy as cp
import numpy as np
from gpuwm.core import sfire_coupler
from gpuwm.core.sfire_core import nearest,IgnitionLine
from tools.sfire_wrf471_oracle.fixture import words


def main():
    root=Path(__file__).parent
    record=json.loads((root/'fixtures/receipt.json').read_text())
    fpath=root/'fixtures/native.npz'
    assert hashlib.sha256(fpath.read_bytes()).hexdigest()==record['sha256']
    with np.load(fpath,allow_pickle=False) as f:f=dict(f)
    x,y=sfire_coupler.geographic_ignition_units(f['latitude'])
    native=words(cp.asnumpy(cp.stack((x,y))),f['units'])
    geometry=[]
    for i in range(32):
        ax,ay,sx,sy,ex,ey=map(float,f['coords'][:,i])
        for kind in range(2):
            line=IgnitionLine(sx,sy,sx if kind==0 else ex,sy if kind==0 else ey,0,60,100,1)
            d,t=nearest(cp.asarray([ax],cp.float32),cp.asarray([ay],cp.float32),line,
                unit_x=float(x[i].item()),unit_y=float(y[i].item()))
            geometry.append(words(cp.asnumpy(cp.stack((d,t)))[:,0],f['nearest'][kind,:,i]))
    old_y=np.float32(2.*np.pi*6370000./360.)
    old_x=np.asarray(np.cos(f['latitude'].astype(np.float64)*np.pi/180.)*float(old_y),np.float32)
    old=np.stack((old_x,np.full(32,old_y,np.float32)))
    source_names=['gpuwm/core/sfire_coupler.py','gpuwm/core/kernels/sfire_coupling.cu',
                  'gpuwm/core/kernels/__init__.py','gpuwm/core/kernels/glibc_trig_flt32.cuh',
                  'gpuwm/core/sfire_core.py','gpuwm/core/kernels/sfire_core.cu',
                  'tests/test_sfire_geographic_wrf471_parity.py']
    summary=dict(words=native['words']+sum(g['words'] for g in geometry),
        different_words=native['different_words']+sum(g['different_words'] for g in geometry),
        max_ulp=max([native['max_ulp']]+[g['max_ulp'] for g in geometry]))
    receipt=dict(schema='sfire-geographic-gpu-replay-v1',reference='unchanged compiled WRF v4.7.1 driver/constants and nearest',
        corpus_sha256=hashlib.sha256((root/'fixtures/receipt.json').read_bytes()).hexdigest(),
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in source_names},
        actual_composed_source_sha256=hashlib.sha256(sfire_coupler.module_source().encode()).hexdigest(),
        actual_compiler_options=list(sfire_coupler.MODULE_OPTIONS),summary=summary,units=native,
        point_line_geometry=geometry,former_double_constructor_negative_control=words(old,f['units']),
        former_unit_latitude_m_per_degree=float(old_y),native_unit_latitude_m_per_degree=float(f['units'][1,0]),
        constructor_qualification=dict(pytest_cases=10,source='tests/test_sfire_geographic_wrf471_parity.py',
            path='actual FireCoupler -> FireState -> actual ignite_fire GPU kernel',
            latitudes=f['latitude'][[0,7,13,18,25]].tolist(),point_and_line=True))
    if summary['different_words']:raise ValueError(summary)
    Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
