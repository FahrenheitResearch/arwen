"""Device checks. Run only under the sprint card queue."""
from pathlib import Path
import numpy as np
import pytest
pytestmark=pytest.mark.gpu

def test_full_wrf_driver_oracle_and_image(tmp_path):
    from tools.lightning_wrf461_oracle.validate_lightning import grade
    root=Path(__file__).resolve().parents[1]
    grade(root/"gpuwm/data/lightning/wrf461.json",tmp_path/"grade.json")

def test_tiled_domain_reduction_matches_serial_patch():
    import cupy as cp
    from tools.lightning_wrf461_oracle.validate_lightning import fixture,OUTPUTS
    from gpuwm.core.lightning import launch_lightning
    fields={k:cp.asarray(v) for k,v in fixture("deep").items()}
    def run(tiles):
        out={name:cp.zeros((2,3),cp.float32) for name in OUTPUTS}
        launch_lightning(fields,out,option=2,method=3,cellcount_method=2,dx=3000.,dy=4000.,interval=15.,tiles=tiles)
        return {name:cp.asnumpy(value).view(np.uint32) for name,value in out.items()}
    serial=run(None);tiled=run([(0,1,0,2),(1,3,0,2)])
    for name in OUTPUTS:np.testing.assert_array_equal(serial[name],tiled[name])

def test_restart_carries_integer_top_calendar_and_all_flash_fields(tmp_path):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    from gpuwm.core.lightning import initialize_lightning
    from gpuwm.io.restart import write_restart,restore_restart
    from gpuwm.io import restart
    def _fill_setup(state):
        for index, name in enumerate(restart.STATE_SETUP_ARRAYS):
            array=getattr(state,name)
            array[...] = cp.arange(array.size,dtype=cp.float32).reshape(array.shape)*cp.float32(.01*(index+1))
    cfg=RunConfig(nx=6,ny=4,nz=5,dx=2000.,dy=2000.,ztop=8000.,dt=10.,run_seconds=0.,
                  lightning_option=3,mp_physics=10,moist=True)
    a=DomainState(cfg);_fill_setup(a)
    initialize_lightning(a,cfg)
    for name,value in a._scratch.items():
        if name.startswith("ltng_") and value.dtype==cp.float32:
            value[...] = cp.arange(value.size,dtype=cp.float32).reshape(value.shape)*cp.float32(.125)
    a.scratch((4,6),"ltng_ktop_deep",dtype=np.int32)[...] = 4
    path=write_restart(tmp_path/"lightning.npz",a,cfg)
    b=DomainState(cfg);_fill_setup(b);initialize_lightning(b,cfg)
    restore_restart(path,b,cfg)
    for name,value in a._scratch.items():
        if name.startswith("ltng_"):
            np.testing.assert_array_equal(cp.asnumpy(value).view(np.uint32),cp.asnumpy(b._scratch[name]).view(np.uint32))
