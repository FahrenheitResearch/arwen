"""Actual fused health kernel follows canonical device-store pointers."""
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.core import health, streaming


@pytest.mark.gpu
@pytest.mark.parametrize('store_kind', ['host', 'device'])
def test_attached_store_poison_uses_canonical_values_and_full_base(monkeypatch, store_kind):
    import cupy as cp

    class State:
        def __init__(self):
            self._scratch = {}
            self.physics = None
            self._lateral_boundary_device = None
            self.thp = cp.full((3, 8, 4), 4, dtype=cp.float32)
            self.mup = cp.full((8, 4), 200, dtype=cp.float32)
            self.thb = cp.full((3, 8, 4), 296, dtype=cp.float32)
            self.mub2d = cp.full((8, 4), 90000, dtype=cp.float32)
        def scratch(self, shape, slot):
            if slot not in self._scratch:
                self._scratch[slot] = cp.zeros(shape, dtype=cp.float32)
            return self._scratch[slot]

    state = State()
    carriers = {'state/thp': state.thp, 'state/mup': state.mup}
    store = {k: (v.copy() if store_kind == 'device' else cp.asnumpy(v))
             for k, v in carriers.items()}
    state._streamed_domain = SimpleNamespace(store=store)
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(
        grid_id=1, run=SimpleNamespace(nx=4, ny=8)))
    model = SimpleNamespace(_prepared_by_grid_id={})
    monkeypatch.setattr(streaming, 'streamed_store_inventory', lambda:
                        lambda template, _: carriers)
    gate = health.health_validator_for_domain(model, node)
    gate.require_healthy(phase='healthy')
    assert gate.device is (store_kind == 'device')
    assert gate.coverage['not_in_store'] == ()
    state.thp[-1, -1, -1] = cp.nan
    gate.require_healthy(phase='stale-resident-poison')
    store['state/thp'][2, 7, 3] = np.float32(np.nan)
    report = gate.validate(phase='canonical-poison')
    assert (report.first_bad_field, report.first_bad_index) == ('thp', (2, 7, 3))
    store['state/thp'][2, 7, 3] = np.float32(-400.)
    report = gate.validate(phase='canonical-total-theta')
    assert (report.first_bad_field, report.first_bad_value) == ('thp', -104.)
