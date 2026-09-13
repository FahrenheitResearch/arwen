"""Small device controls for WDM6 counter safety and unchanged valid columns."""
import cupy as cp
import numpy as np
import pytest

from test_wdm6_count_safety import fields,launch

pytestmark=pytest.mark.gpu


@pytest.mark.parametrize('which', ['rain','outer','geometry'])
def test_bad_count_keeps_affected_column_and_reports_failure(which):
    values=fields(cp)
    dt=60.
    if which=='rain': values['dz'][0,0,1]=np.float32(9e-7)
    if which=='outer': dt=float(np.float32(120.*2**31))
    if which=='geometry': values['dz'][0,0,1]=np.float32(0.)
    before={name:value.get() for name,value in values.items()}
    with pytest.raises(FloatingPointError): launch(values,dt=dt)
    for name,value in values.items():
        if which=='outer': np.testing.assert_array_equal(value.get(),before[name])
        else: np.testing.assert_array_equal(value.get()[...,1],before[name][...,1])


def test_graph_replay_reports_count_failure_at_existing_ledger_drain():
    from gpuwm.core import health_ledger
    values=fields(cp)
    status=cp.zeros(1,cp.uint32)
    ledger=health_ledger.HealthLedger()
    stream=cp.cuda.Stream(non_blocking=True)
    with stream,health_ledger.deferring(ledger):
        launch(values,count_status=status)
    stream.synchronize();ledger.drain()
    with stream,health_ledger.deferring(ledger):
        stream.begin_capture()
        launch(values,count_status=status)
        graph=stream.end_capture()
    values['dz'][0,0,1]=np.float32(9e-7)
    cp.cuda.get_current_stream().synchronize()
    before={name:value.get() for name,value in values.items()}
    graph.launch(stream);stream.synchronize()
    with pytest.raises(FloatingPointError,match='signed 32-bit counter'): ledger.drain()
    for name,value in values.items(): np.testing.assert_array_equal(value.get()[...,1],before[name][...,1])


def test_count_guard_stays_within_the_registered_column_frames():
    from gpuwm.core import preflight
    from gpuwm.core.kernels import get_kernel,get_kernel_int_defines
    for bound in (64,80):
        kernel=(get_kernel('wdm6','wdm6_column') if bound==64 else
                get_kernel_int_defines('wdm6','wdm6_column',(('WDM6_KMAX',bound),)))
        assert kernel.attributes['local_size_bytes'] <= preflight.WDM6_TIER_FRAME.frame_bytes(bound)
