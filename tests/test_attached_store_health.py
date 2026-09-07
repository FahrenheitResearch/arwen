"""Canonical host stores remain authoritative after resident initialization."""
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.core import health, streaming
from test_store_health_gate import _state, _store, NZ, NY, NX


@pytest.mark.parametrize('prepared_base', [False, True])
def test_attached_host_gate_sees_canonical_poison_and_ignores_stale_snapshot(
        monkeypatch, prepared_base):
    state, carriers = _state(NY)
    store = _store()
    state._streamed_domain = SimpleNamespace(store=store)
    cfg = SimpleNamespace(ny=NY, nx=NX)
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(grid_id=1, run=cfg))
    prepared = (SimpleNamespace(initial_result=SimpleNamespace(base=SimpleNamespace(
        thb=state.thb, mub=state.mub2d, p_top=None))) if prepared_base else None)
    model = SimpleNamespace(_prepared_by_grid_id={1: prepared})
    monkeypatch.setattr(streaming, 'streamed_store_inventory', lambda:
                        lambda template, _: carriers)
    monkeypatch.setattr(health, 'StateHealthValidator', lambda *a, **k:
                        pytest.fail('stale resident validator selected'))
    gate = health.health_validator_for_domain(model, node)
    gate.require_healthy(phase='canonical-healthy')
    assert gate.coverage['not_in_store'] == ()
    state.thp[-1, -1, -1] = np.nan
    gate.require_healthy(phase='stale-resident-poison-is-not-the-forecast')
    store['state/thp'][NZ-1, NY-1, NX-1] = np.nan
    report = gate.validate(phase='canonical-poison')
    assert not report.ok
    assert report.first_bad_field == 'thp'
    assert report.first_bad_index == (NZ-1, NY-1, NX-1)
    assert report.phase == 'canonical-poison'


def test_rebuilt_store_gate_uses_new_template_store_and_full_base(monkeypatch):
    template, carriers = _state(2)
    store = _store()
    full_base = np.full((NZ, NY, NX), 300., np.float32)
    full_mass = np.full((NY, NX), 1000., np.float32)
    stream = SimpleNamespace(template=template, store=store,
        _geography={'setup/thb': full_base, 'setup/mub2d': full_mass})
    state = SimpleNamespace(_streamed_domain=stream, thb=full_base, mub2d=full_mass, p_top=None)
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(grid_id=2,
        run=SimpleNamespace(ny=NY, nx=NX)))
    model = SimpleNamespace(_prepared_by_grid_id={2: SimpleNamespace(streamed_store=object())})
    def inventory(actual, _):
        assert actual is template
        return carriers
    monkeypatch.setattr(streaming, 'streamed_store_inventory', lambda: inventory)
    gate = health.health_validator_for_domain(model, node)
    assert gate.bundle.store is store
    assert gate.bundle.base.thb is full_base
    gate.require_healthy(phase='rebuilt')
    store['state/thp'][-1, -1, -1] = np.nan
    assert not gate.validate(phase='new-store-poison').ok
