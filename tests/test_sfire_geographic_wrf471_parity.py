"""Native geographic driver units and actual constructor/kernel wiring."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
from conftest import requires_gpu


def load():
    root=Path(__file__).parents[1]/'tools/sfire_coupled_ideal/geographic_units/fixtures'
    receipt=json.loads((root/'receipt.json').read_text())
    raw=root/'native.npz'
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==receipt['sha256']
    with np.load(raw,allow_pickle=False) as f:return dict(f)


@requires_gpu
def test_original_driver_units_and_point_line_words():
    import cupy as cp
    from gpuwm.core.sfire_coupler import geographic_ignition_units
    from gpuwm.core.sfire_core import nearest,IgnitionLine
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f=load();x,y=geographic_ignition_units(f['latitude'])
    assert_bit_exact(cp.asnumpy(cp.stack((x,y))),f['units'],'native geographic units')
    for i in range(32):
        ax,ay,sx,sy,ex,ey=map(float,f['coords'][:,i])
        for kind in range(2):
            line=IgnitionLine(sx,sy,sx if kind==0 else ex,sy if kind==0 else ey,0,60,100,1)
            d,t=nearest(cp.asarray([ax],cp.float32),cp.asarray([ay],cp.float32),line,
                        unit_x=float(x[i].item()),unit_y=float(y[i].item()))
            assert_bit_exact(cp.asnumpy(cp.stack((d,t)))[:,0],f['nearest'][kind,:,i],f'nearest {kind}/{i}')


@requires_gpu
@pytest.mark.parametrize('index',[0,7,13,18,25])
@pytest.mark.parametrize('line',[False,True])
def test_actual_geographic_constructor_passes_native_units_to_ignition_kernel(index,line,monkeypatch):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.core import sfire_core
    from gpuwm.core.fp32_ulp import assert_bit_exact
    f=load();latitude=float(f['latitude'][index]);nx,ny,nz=8,6,4;sr=2
    cfg=RunConfig(nx=nx,ny=ny,nz=nz,dx=90.,dy=90.,ztop=1200.,dt=.5,run_seconds=1.,
        ifire=2,sr_x=sr,sr_y=sr,fire_fuel_read=0,fire_fuel_cat=1,fire_boundary_guard=-1,
        fire_num_ignitions=1,fire_ignition_start_lon1=-121.,fire_ignition_start_lat1=latitude,
        fire_ignition_end_lon1=-120.992 if line else -121.,fire_ignition_end_lat1=latitude+.009 if line else latitude,
        fire_ignition_end_time1=60. if line else 0.,fire_ignition_radius1=100.,fire_ignition_ros1=1000.)
    shape=(ny*sr,nx*sr)
    static=dict(FIRE_COORDINATE_MODE='geographic',CEN_LAT=latitude,ZSF=cp.zeros(shape,cp.float32),
        DZDXF=cp.zeros(shape,cp.float32),DZDYF=cp.zeros(shape,cp.float32),FZ0=cp.full(shape,.1,cp.float32),
        FXLONG=cp.full(shape,-121.,cp.float32),FXLAT=cp.full(shape,latitude,cp.float32))
    state=SimpleNamespace(p=cp.zeros((nz,ny,nx),cp.float32),ht=cp.zeros((ny,nx),cp.float32),elapsed_seconds=0.)
    fire=FireCoupler(state,cfg,dict(znt=cp.full((ny,nx),.1,cp.float32)),static)
    assert_bit_exact(np.asarray([fire.grid.unit_x,fire.grid.unit_y],np.float32),f['units'][:,index],'constructor units')
    original=sfire_core.ignite_fire;calls=[]
    def observe(*args,**kwargs):
        calls.append([kwargs['unit_x'],kwargs['unit_y']])
        return original(*args,**kwargs)
    monkeypatch.setattr(sfire_core,'ignite_fire',observe)
    fire.grid._apply_ignitions(np.float32(0),np.float32(.5))
    assert len(calls)==1
    assert_bit_exact(np.asarray(calls[0],np.float32),f['units'][:,index],'actual ignition kernel inputs')
    assert fire.grid.last_ignited_counts==(nx*ny*sr*sr,)
    assert bool(cp.all(fire.grid.data['lfn'][fire.grid.interior]<=0))


@requires_gpu
def test_nonfinite_coordinate_units_refuse_before_kernel():
    from gpuwm.core.sfire_coupler import geographic_ignition_units
    with pytest.raises(ValueError,match='latitude must be finite'):
        geographic_ignition_units(np.asarray([float('nan'),float('inf')],np.float32))
