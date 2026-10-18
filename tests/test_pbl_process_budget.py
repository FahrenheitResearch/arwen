"""Independent two-layer pressure/flux ledger checks. Apache-2.0."""
from types import SimpleNamespace

import cupy as cp
import pytest

from gpuwm.verify.forecast_process_budget import PblMassLedger


def test_pressure_measures_and_surface_supply():
    shape = (2, 1, 1)
    state = SimpleNamespace(qv=cp.zeros(shape, cp.float32),
                            c1h=cp.ones(2, cp.float32),
                            c2h=cp.zeros(2, cp.float32),
                            dnw=cp.full(2, -.5, cp.float32),
                            total_mu=lambda: cp.full((1, 1), 4., cp.float32))
    driver = SimpleNamespace(urban=None, fields={
        'qfx': cp.full((1, 1), .1, cp.float32),
        'hfx': cp.full((1, 1), 7., cp.float32)})
    rates = {'dqv': cp.asarray([.5, -.1], cp.float32).reshape(shape)}
    src = rates['dqv']*4
    coupled = SimpleNamespace(scalar_for=lambda k: src if k == 'qv' else None)
    ledger = PblMassLedger(state, lambda a: .05*a.sum(axis=(1, 2, 3)),
                          lambda a: a.sum(axis=(1, 2)))
    ledger.sample(driver, SimpleNamespace(bl_pbl_physics=1, bldt=0, dt=2.),
                  rates, coupled, {'p_interface': cp.asarray([10., 7., 2.],
                                                    cp.float32).reshape(3, 1, 1)})
    # C=4, moist pressure thickness=3,5, -dnw/g=.05, dt=2.
    # Dry source .16; moist source .20; metric correction .04.
    for key, expected in [('coupled_source', .16), ('raw_dry_source', .16),
                          ('ysu_pressure_source', .20),
                          ('pressure_measure_difference', .04)]:
        cp.testing.assert_allclose(ledger.values[key], [expected, 0, 0, 0, 0, 0],
                                   rtol=1e-7, atol=1e-9)
    cp.testing.assert_allclose(ledger.values['surface_qfx_hfx'], [.2, 14], rtol=1e-7)
    src.fill(100)
    rates['dqv'].fill(100)
    cp.testing.assert_allclose(ledger.values['coupled_source'][0], .16, rtol=1e-7)
    assert ledger.calls == 1


@pytest.mark.parametrize('scheme,cadence,urban', [(2, 0, None), (1, 1, None),
                                                (1, 0, SimpleNamespace(pbl_terms={}))])
def test_unsupported_pbl_contract_refused(scheme, cadence, urban):
    ledger = PblMassLedger(None, None, None)
    with pytest.raises(ValueError, match='every-step YSU'):
        ledger.sample(SimpleNamespace(urban=urban),
                      SimpleNamespace(bl_pbl_physics=scheme, bldt=cadence),
                      {}, None, {})
