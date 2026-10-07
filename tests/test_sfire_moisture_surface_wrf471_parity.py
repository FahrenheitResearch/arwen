"""Compiled native ground extrapolation for a corrected no-exchange caller."""
from pathlib import Path
import pytest
from conftest import requires_gpu

ROOT=Path(__file__).parents[1]/"tools/sfire_coupled_ideal/initialization/surface_fixtures"


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("case",range(1,5))
def test_native_ground_diagnostics_leave_exchange_and_atmosphere_unchanged(case):
    import cupy as cp
    from tools.sfire_coupled_ideal.initialization.grade_geometry import load
    from tools.sfire_wrf471_oracle.fixture import words
    from gpuwm.core.sfire_ideal import diagnose_moisture_surface
    f=load(ROOT,f"moisture_surface/case_{case}")
    atmosphere={key:cp.asarray(f[key]) for key in ("temperature","theta","qv")}
    atmosphere['p_interface']=cp.broadcast_to(cp.asarray(f['psfc']), (4,*f['psfc'].shape)).copy()
    before={key:cp.asnumpy(value).tobytes() for key,value in atmosphere.items()}
    surface={key:cp.zeros(f['psfc'].shape,cp.float32) for key in ('t2','th2','q2','psfc')}
    surface.update(hfx=cp.full(f['psfc'].shape,23.75,cp.float32),qfx=cp.full(f['psfc'].shape,.000075,cp.float32))
    held={key:cp.asnumpy(surface[key]).tobytes() for key in ('hfx','qfx')}
    diagnose_moisture_surface(atmosphere,*f['coeff'],surface)
    for key in ('t2','th2','q2','psfc'):
        assert words(cp.asnumpy(surface[key]),f[key])['different_words']==0,key
    assert before=={key:cp.asnumpy(value).tobytes() for key,value in atmosphere.items()}
    assert held=={key:cp.asnumpy(surface[key]).tobytes() for key in held}
