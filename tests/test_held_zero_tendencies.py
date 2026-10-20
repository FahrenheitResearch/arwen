"""Zero held sources share storage without sharing an active/composed rate."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('shape', [(3, 4, 5), (3, 5, 4)])
def test_held_zero_sources_preserve_coupling_composition_and_logical_inventory(monkeypatch, shape):
    from gpuwm.core import physics as ph
    from gpuwm.config import RunConfig
    from gpuwm.core.preflight import physics_array_shapes
    monkeypatch.setattr(ph, 'cp', np)
    monkeypatch.setattr(ph.tendency_coupling, 'ready', lambda *a, **kw: False)
    nz, ny, nx = shape
    state = SimpleNamespace(p=np.zeros(shape, np.float32),
                            c1h=np.linspace(.1, .8, nz, dtype=np.float32),
                            c2h=np.ones(nz, np.float32), has_msf=False,
                            total_mu=lambda: np.full((ny, nx), 9000., np.float32))
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000., dy=3000., dt=15.,
                    moist=True, cu_physics=1, bl_pbl_physics=1, bldt=1.,
                    ztop=15000., run_seconds=60.)
    rates = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * np.float32(.001)
    radiation_zeros = ('ru', 'rv', 'rqv', 'rqc')
    old_rad = ph.couple_column_tendencies(state, cfg, rtheta=rates)
    new_rad = ph.couple_column_tendencies(state, cfg, rtheta=rates,
                                         shared_zero_components=radiation_zeros)
    old_cu = ph.couple_column_tendencies(state, cfg, rtheta=rates, rqv=-rates, rqc=rates)
    new_cu = ph.couple_column_tendencies(state, cfg, rtheta=rates, rqv=-rates, rqc=rates,
                                        shared_zero_components=('ru', 'rv'))
    for old, new in ((old_rad, new_rad), (old_cu, new_cu)):
        assert vars(old).keys() == vars(new).keys()
        for name, want in vars(old).items():
            got = getattr(new, name)
            if want is None:
                assert got is None
            else:
                assert got.flags.c_contiguous
                np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    for name in radiation_zeros:
        assert np.shares_memory(new_rad.rv, getattr(new_rad, name))
    assert np.shares_memory(new_cu.ru, new_cu.rv)
    assert not np.shares_memory(new_rad.rv, new_cu.rv)
    assert not np.shares_memory(new_rad.rtheta, new_rad.rv)
    assert not np.shares_memory(new_cu.rqv, new_cu.rv)
    targets = []
    for rad, cu in ((old_rad, old_cu), (new_rad, new_cu)):
        pbl = ph.PhysicsTendencies.zeros(state)
        pbl.rtheta[...] = rates
        pbl.rqv[...] = -rates
        driver = SimpleNamespace(radiation_active=True, cu_physics=1,
            fire_tendencies=None, pbl_tendencies=pbl, radiation_tendencies=rad,
            cumulus_tendencies=cu, tendencies=ph.PhysicsTendencies.zeros(state))
        ph.PhysicsDriver._compose_tendencies(driver, cfg)
        targets.append(driver.tendencies)
    for name in ('ru', 'rv', 'rtheta', 'rqv', 'rqc'):
        np.testing.assert_array_equal(getattr(targets[0], name).view(np.uint32),
                                      getattr(targets[1], name).view(np.uint32))
    assert not np.any(new_rad.rv)
    assert not np.any(new_cu.rv)
    eager = ph.PhysicsTendencies.zeros(state, shared_zero_components=radiation_zeros)
    for name in radiation_zeros:
        assert np.shares_memory(eager.rv, getattr(eager, name))
    assert not np.shares_memory(eager.rtheta, eager.rv)
    inventory = physics_array_shapes(cfg)
    for stack in ('radiation_tendencies', 'cumulus_tendencies'):
        # Here ny >= nx only in the second arm: the larger face is selected
        # by element count, not by a hard-coded orientation.
        largest = 'rv' if nx >= ny else 'ru'
        assert f'{stack}/{largest}' in inventory
        assert sum(f'{stack}/{comp}' in inventory for comp in ('ru', 'rv')) == 1
    with pytest.raises(ValueError, match='active'):
        ph.couple_column_tendencies(state, cfg, rqv=rates,
                                    shared_zero_components=('rqv',))
