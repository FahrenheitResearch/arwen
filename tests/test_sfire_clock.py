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


def _fire_scalars(seconds, step_count):
    clocks={'time_seconds':seconds,'step_count':step_count}
    return {'elapsed_seconds':seconds,'fire_clocks':clocks,
            'fire_header':{'fire':{'grid':dict(clocks)}}}


def test_streamed_binding_reads_a_lazy_store_only_when_a_source_clock_moves():
    """Guards 67e213588's per-step drain: the store is read only for a move.

    Every case but the last must leave the store unread, because on the
    ranked multi-card road reading it drains every slab to the host.
    """
    reads=[]
    field=np.full((2,3),5.0)
    def store():
        reads.append(1)
        return {'state/chemdiag_sfire_source_lasttime':field}
    bind_streamed_clock({'elapsed_seconds':5.0},store,6.0)        # no fire clocks
    unchanged=_fire_scalars(5.0,3)
    bind_streamed_clock(unchanged,store,5.0)                       # clock already published
    first=_fire_scalars(0.0,0)
    bind_streamed_clock(first,store,5.0)                           # no completed fire step
    assert reads==[] and first['fire_clocks']['time_seconds']==5.0
    moving=_fire_scalars(5.0,3)
    bind_streamed_clock(moving,store,6.0)
    assert reads==[1] and np.all(field==6.0)
    assert moving['fire_clocks']['time_seconds']==moving['fire_header']['fire']['grid']['time_seconds']==6.0


class _RankedDomain:
    """StreamedDomain.impose_clock over a store that counts its drains."""
    from gpuwm.core.streaming import StreamedDomain
    impose_clock=StreamedDomain.impose_clock
    ranked=True

    def __init__(self,scalars,keys):
        self.scalars=scalars
        self.drains=0
        self.domain_clock=[]
        self._home={key:np.full((2,3),float(scalars['elapsed_seconds'])) for key in keys}
        self._run=SimpleNamespace(store_keys=self._home.keys,
                                  impose_domain_clock=self.domain_clock.append)

    @property
    def store(self):
        self.drains+=1
        return self._home


@pytest.mark.parametrize('fire,keys',[
    (False,('state/thp',)),
    (True,('state/thp',)),
    (True,('state/thp','state/chemdiag_sfire_source_lasttime'))])
def test_impose_clock_before_and_after_every_step_drains_nothing(fire,keys):
    """THE BREAKAGE: 67e213588 made StreamedDomain.impose_clock read
    ``self.store``, which on the ranked road (2 or more cards) is a full
    drain of every slab plus a full re-gather before the next sweep.  The
    executor calls it before and after every step (gpuwm/core/model.py), so
    2.8.7 drained 720 times in 720 steps on M1 and a 2-card run was 0.78x of
    one card.  A forecast whose clock and fire clock agree must drain nothing.
    """
    dt=20.0
    scalars=_fire_scalars(0.0,0) if fire else {'elapsed_seconds':0.0}
    domain=_RankedDomain(scalars,keys)
    for step in range(12):
        now=step*dt
        domain.impose_clock(now)
        scalars['elapsed_seconds']+=dt                     # the sweep's own clock
        if fire:
            scalars['fire_clocks']['time_seconds']+=dt     # and the fire step's
            scalars['fire_clocks']['step_count']+=1
        domain.impose_clock(now+dt)
    assert domain.drains==0
    assert domain.domain_clock[-1]==12*dt and scalars['elapsed_seconds']==12*dt


@pytest.mark.parametrize('carried',[True,False])
def test_impose_clock_still_moves_a_carried_source_clock(carried):
    """The lazy store keeps 67e213588's fix: a moved fire clock moves its source.

    A domain that does not carry the source field has nothing to move, and
    its membership is asked without a drain.
    """
    key='state/chemdiag_sfire_source_lasttime'
    scalars=_fire_scalars(5.0,2)
    domain=_RankedDomain(scalars,(key,) if carried else ('state/thp',))
    domain.impose_clock(6.0)
    assert domain.drains==int(carried)
    if carried:
        assert np.all(domain._home[key]==6.0)
    assert scalars['fire_clocks']['time_seconds']==scalars['elapsed_seconds']==6.0
