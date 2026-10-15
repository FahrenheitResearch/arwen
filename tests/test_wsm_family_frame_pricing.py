"""Admission must price the production column tier from its GPU reading."""
from datetime import datetime
import pytest
from conftest import requires_gpu
from test_wsm7_admission import cfg
from gpuwm.experiment import experiment_from_run_config
from gpuwm.core import preflight as pf
from gpuwm.core.wsm_family import row

@pytest.mark.parametrize('levels,expected',[(45,7216),(72,9008)])
def test_admission_prices_the_launched_tier_instead_of_an_assumed_bound(levels,expected):
    exp=experiment_from_run_config(cfg(nz=levels),datetime(2000,1,1))
    unit=row(24).unit
    assert unit not in pf.assumed_bound_modules({unit})
    assert pf.kernel_local_frame_bytes(exp)[unit]==expected

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('levels', [45,72])
def test_admission_covers_the_production_column_frame(levels):
    import cupy as cp
    from gpuwm.core.wsm_family import _module,level_tier
    r=row(24)
    tier=level_tier(24,levels)
    module=_module(r.unit,r.kmax_define,tier,cp.cuda.runtime.getDevice())
    actual=max(cp.cuda.driver.funcGetAttribute(3,module.get_function(s).ptr)
               for s in [r.column_kernel,r.refl_kernel])
    exp=experiment_from_run_config(cfg(nz=levels),datetime(2000,1,1))
    # A price is the ceiling over recorded platforms. An unrecorded
    # compiler may emit a smaller frame, but must never exceed the price.
    assert pf.kernel_local_frame_bytes(exp)[r.unit]>=actual
