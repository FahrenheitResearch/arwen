"""Fire controls independent of CUDA; device continuation uses oracle tests."""

import pytest

from gpuwm.core.sfire import FireOptions, FireState, registry_core_options
from gpuwm.core.sfire_core import IgnitionLine
from gpuwm.core.sfire_phys import FuelTable


def test_registry_defaults_reach_fire_state():
    opt = registry_core_options()
    assert (opt.advection,opt.viscosity_ngp,opt.boundary_guard) == (1,2,8)
    assert FireOptions().core == opt


def test_freeze_is_after_elapsed_ignition_time():
    line = IgnitionLine(0,0,0,0,10,10,1,1)
    state = FireState({},1,1,(1,10,1,10),((1,10,1,10),),
                      FireOptions(const_time=20),FuelTable(),(line,))
    assert not state.frozen(0)
    assert not state.frozen(10)
    assert not state.frozen(29)
    assert state.frozen(30)
    assert state.frozen(31)
    state.options = FireOptions(const_time=-1)
    assert not state.frozen(100)


def test_moisture_requires_modeled_fuel_input():
    with pytest.raises(ValueError,match="fire_fmc_read=0"):
        FireOptions(fmoist_run=True,fire_fmc_read=1)
    assert FireOptions(fmoist_run=True,fire_fmc_read=0).fmoist_run


def test_optional_substeps_require_defined_duration():
    with pytest.raises(ValueError,match="finite and positive"):
        FireOptions(max_fire_dt=0)
    with pytest.raises(ValueError,match="finite and positive"):
        FireOptions(max_fire_dt=float("inf"))


@pytest.mark.parametrize("key", ["fire_fuel_left_irl", "fire_fuel_left_jrl"])
@pytest.mark.parametrize("value", [1, 3, 0, True, 2.0])
def test_native_fuel_subcell_geometry_refuses_unsupported_config(key, value):
    from gpuwm.config import RunConfig
    from gpuwm.sfire_config import validate_fire_config
    from gpuwm.core.sfire_coupler import FireCoupler
    cfg = RunConfig(nx=8, ny=8, nz=6, dx=90.,dy=120.,ztop=2000.,dt=.25,
                    run_seconds=1.,ifire=2,sr_x=3,sr_y=2,**{key:value})
    with pytest.raises(ValueError, match=key):
        validate_fire_config(cfg)
    with pytest.raises(ValueError, match=key):
        FireCoupler(None, cfg, None)
