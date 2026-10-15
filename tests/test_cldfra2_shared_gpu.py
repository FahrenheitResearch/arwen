"""Both legacy preps consume one driver binary cloud-fraction field."""
import numpy as np
import pytest
cp=pytest.importorskip('cupy')
from test_rrtmg_legacy_wiring import profile, env, adapter, _call
from gpuwm.core import rrtmgp

def test_one_binary_fraction_field_per_radiation_event(env, adapter, monkeypatch):
    env.cfg.icloud=2
    real=rrtmgp.cal_cldfra2
    calls=[]
    def measured(qc,qi,**flags):
        out=real(qc,qi,**flags)
        calls.append(out)
        return out
    monkeypatch.setattr(rrtmgp,'cal_cldfra2',measured)
    result=_call(adapter,env)
    assert len(calls)==1, 'LW and SW recomputed the driver cloud fraction'
    expected=calls[0].reshape(result.cldfra.shape)
    np.testing.assert_array_equal(cp.asnumpy(result.cldfra).view('u4'),cp.asnumpy(expected).view('u4'))
