"""Fire clocks follow both exact ticks and WRF's kernel-facing time image."""
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.core.clock import DomainClock, DomainTicks
from gpuwm.core.state import refresh_model_time
from gpuwm.core.sfire_clock import bind_state_clock, bind_streamed_clock


def fractional_clock():
    dt=np.float32(np.float32(20)/np.float32(3))/np.float32(4)
    spec=DomainTicks(grid_id=3,parent_id=2,parent_time_step_ratio=4,
        step_ticks=20,dt_fp32=dt,history_ticks=240,restart_ticks=None,
        radt_ticks=None,stepra=None,cudt_ticks=None,stepcu=None,
        bldt_ticks=None,stepbl=None)
    return DomainClock(spec,12,240)


def test_chained_fractional_clock_reproduces_and_fixes_neighboring_words():
    clock=fractional_clock()
    old_end=float(clock.dt_fp32);clock.advance()
    assert np.float32(old_end).view(np.uint32)==0x3fd55555
    assert clock.elapsed_seconds_fp32.view(np.uint32)==0x3fd55556
    grid=SimpleNamespace(time_seconds=0.0,step_count=0,moisture_lasttime=0.0,moisture_nexttime=600.0)
    state=SimpleNamespace(elapsed_seconds=0.0,physics=SimpleNamespace(fire=SimpleNamespace(grid=grid)),
        chemdiag_sfire_source_lasttime=np.zeros((2,3),np.float64))
    clock=fractional_clock()
    for _ in range(12):
        refresh_model_time(state,clock,kernel_launch=True)
        assert state.elapsed_seconds==grid.time_seconds
        end=state.elapsed_seconds+float(clock.dt_fp32)
        state.elapsed_seconds=end;grid.time_seconds=end;grid.step_count+=1
        state.chemdiag_sfire_source_lasttime.fill(end)
        refresh_model_time(state,clock,after_step=True);clock.advance()
        assert state.elapsed_seconds==grid.time_seconds==clock.elapsed_seconds
        assert np.all(state.chemdiag_sfire_source_lasttime==clock.elapsed_seconds)
    assert grid.moisture_lasttime==0 and grid.moisture_nexttime==600


def test_clock_mismatch_is_refused_before_any_binding_mutation():
    grid=SimpleNamespace(time_seconds=2.0,step_count=1)
    field=np.full((2,3),2.0)
    state=SimpleNamespace(elapsed_seconds=1.0,physics=SimpleNamespace(fire=SimpleNamespace(grid=grid)),
        chemdiag_sfire_source_lasttime=field)
    with pytest.raises(ValueError,match='before authoritative'):
        bind_state_clock(state,1.0)
    assert grid.time_seconds==2 and np.all(field==2)


def test_streamed_binding_preserves_biology_and_current_field_owner():
    clocks={'time_seconds':5/3,'step_count':1,'moisture_lasttime':0.0,'moisture_nexttime':600.0}
    header={'fire':{'grid':dict(clocks)}}
    scalars={'elapsed_seconds':5/3,'fire_clocks':clocks,'fire_header':header}
    field=np.full((2,3),5/3,np.float64)
    point=float(fractional_clock().dt_fp32)
    bind_streamed_clock(scalars,{'state/chemdiag_sfire_source_lasttime':field},point)
    assert clocks['time_seconds']==header['fire']['grid']['time_seconds']==point
    assert np.all(field==point)
    assert clocks['moisture_lasttime']==0 and clocks['moisture_nexttime']==600
