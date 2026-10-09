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


class _MirroredOwner:
    """A streamed owner whose host store is current only after a drain.

    ``StreamedDomain.store`` drains before it hands the mapping out: the
    deferred seam's scatter tail on one card, every slab of every rank on the
    ranked road (``tilestream.ranks.RankedRun.store``).  ``slabs`` stands for
    the device copy a sweep writes; the mirror catches up on a read.
    """

    def __init__(self, store):
        self._home = store
        self.slabs = {key: value.copy() for key, value in store.items()}
        self.drains = 0

    @property
    def store(self):
        self.drains += 1
        for key, value in self.slabs.items():
            self._home[key][...] = value
        return self._home


def test_attached_gate_drains_the_owner_at_every_validation(monkeypatch):
    """THE BREAKAGE: the gate captured ``attached.store`` once and scanned that
    mapping at every later phase with no drain.  2.8.7's
    ``StreamedDomain.impose_clock`` drained after every step (67e213588), which
    kept the mirror current by accident; the mc-clock fix removed that drain,
    and a 2-card run's periodic and --health-debug gates then scanned the host
    mirror of the last drain while a NaN sat on the slabs.
    """
    state, carriers = _state(NY)
    owner = _MirroredOwner(_store())
    state._streamed_domain = owner
    cfg = SimpleNamespace(ny=NY, nx=NX)
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(grid_id=1, run=cfg))
    model = SimpleNamespace(_prepared_by_grid_id={1: None})
    monkeypatch.setattr(streaming, 'streamed_store_inventory', lambda:
                        lambda template, _: carriers)
    gate = health.health_validator_for_domain(model, node)
    gate.require_healthy(phase='attached')
    before = owner.drains
    owner.slabs['state/thp'][NZ-1, NY-1, NX-1] = np.nan     # the sweep's step
    report = gate.validate(phase='post-step')
    assert owner.drains > before, 'the gate read a store it did not drain'
    assert not report.ok
    assert report.first_bad_field == 'thp'
    assert report.first_bad_index == (NZ-1, NY-1, NX-1)
