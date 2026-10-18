"""Independent source/mass/clipping examples for the research scalar observer.
SPDX-License-Identifier: Apache-2.0
"""
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.verify.scalar_process_budget import ScalarProcessLedger, TERMS, MASS_TERMS


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('transport,expected_clip', [(-4., 0.), (-32., 4.5)])
def test_final_stage_sources_mass_clamp_and_aliased_boundary(monkeypatch, transport, expected_clip):
    import cupy as cp
    from gpuwm.core import dycore, moist
    from gpuwm.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip('research observer unavailable in strict mode')
    q = cp.ones((2, 2, 2), cp.float32)
    state = SimpleNamespace(qv=q)
    cfg = SimpleNamespace(moist_adv_opt=1, open_x=False, open_y=False,
                          km_opt=4, diff_opt=1, diff_6th_opt=2,
                          diff_6th_factor=.12, diff_6th_slopeopt=1)
    qold = q.copy()
    c1 = cp.ones(2, cp.float32)
    c2 = cp.zeros(2, cp.float32)
    mu0 = cp.full(4, 4., cp.float32)
    mu = cp.full(4, 8., cp.float32)
    workspace = cp.empty_like(q)
    folded = cp.empty_like(q)
    state.scratch = lambda shape, slot: folded
    physics = SimpleNamespace(scalar_for=lambda name: cp.full_like(q, 4.))
    state._scratch = {'smag_rqv': cp.empty_like(q)}

    def mixing(s, cfg, km, kh, specs, **kw):
        s._scratch['smag_rqv'].fill(1.)

    def diff6(cfg, slot, f0, target, *args, **kw):
        target += cp.float32(1.)

    def prepare(s, cfg):
        specs = [(qold, None, None, c1, c2, 'smag_rqv', '')]
        dycore._compute_wrf_smag_tendencies(s, cfg, None, None, specs, time_t=True)
        dycore.add_diff6_row(cfg, 'smag_rqv', qold, s._scratch['smag_rqv'])

    monkeypatch.setattr(dycore, 'prepare_fixed_tendencies', prepare)
    monkeypatch.setattr(dycore, '_compute_wrf_smag_tendencies', mixing)
    monkeypatch.setattr(dycore, 'add_diff6_row', diff6)

    def step(s, cfg, *, final):
        if not final:
            return
        dycore.prepare_fixed_tendencies(s, cfg)
        workspace.fill(6.)
        eff = moist._pd_fold_sources(s, cfg, 'qv', qold, cp.full_like(q, 4.), 2.,
                                     physics, fixed_tendency=s._scratch['smag_rqv'],
                                     lbc_held=workspace)
        # The production caller destroys this aliased boundary source before
        # the update; its ledger must have captured it at the producer.
        workspace.fill(transport)
        moist._update_scalar_in_place(q, eff, workspace, c1, c2, mu0, mu, 2., clamp=True)

    monkeypatch.setattr(dycore, 'advance_scalars_stage', step)
    step(state, cfg, final=True)
    stock = q.get().view(np.uint32)
    q.fill(1.)
    # Independent two-layer dry-pressure integral: each layer has delta_eta
    # .5, gravity is 10 for this analytic fixture. Uniform map areas cancel.
    observer = ScalarProcessLedger(state, lambda fields: fields.mean(axis=(1, 2, 3)),
        mass_reduce=lambda fields: .05*fields.mean(axis=(2, 3)).sum(axis=1))
    previous_fold = moist._pd_fold_sources
    with observer.observe():
        for final in (False, False, True):
            dycore.advance_scalars_stage(state, cfg, final=final)
    np.testing.assert_array_equal(q.get().view(np.uint32), stock)
    assert moist._pd_fold_sources is previous_fold
    assert dycore.advance_scalars_stage is step
    assert observer.calls['qv'] == 1
    values = dict(zip(TERMS, observer.values['qv'].get()))
    expected = dict(fixed_mixing=.5, physics=1., lateral_source=1.5,
                    fold_roundoff=0., mass_change=-.5, transport=transport/4.,
                    update_roundoff=0., positive_clamp=expected_clip)
    for name, value in expected.items():
        assert values[name] == value
    assert values['final_scalar_delta'] == sum(expected.values())
    mass = dict(zip(MASS_TERMS, observer.mass_values['qv'].get()))
    expected_mass = dict(fixed_mixing=.4, physics=.8, lateral_source=1.2,
                        fold_roundoff=0., transport=.2*transport,
                        update_roundoff=0., positive_clamp=.8*expected_clip)
    for name, value in expected_mass.items():
        assert mass[name] == pytest.approx(value, abs=1e-14)
    # Original mass .4; stock result is q=2.5 or 0 in layers with new C=8.
    assert mass['coupled_scalar_delta'] == pytest.approx(1.6 if transport == -4. else -.4)
    assert mass['coupled_scalar_delta'] == pytest.approx(sum(expected_mass.values()), abs=1e-14)
    np.testing.assert_array_equal(observer.mixing_values['qv'].get(), [.25, .25, 0.])
    np.testing.assert_allclose(observer.mass_mixing_values['qv'].get(), [.2, .2, 0.], rtol=0, atol=1e-14)
    assert observer.mixing_calls['prepare'] == 1
    assert observer.mixing_calls['mixing_package'] == 1
    assert observer.mixing_calls['sixth_order']['qv'] == 1
    assert dycore.prepare_fixed_tendencies is prepare
    assert dycore._compute_wrf_smag_tendencies is mixing
    assert dycore.add_diff6_row is diff6


@pytest.mark.gpu
@requires_gpu
def test_filter_records_realized_addition_not_zero_buffer_proposal(monkeypatch):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip('research observer unavailable in strict mode')
    state = SimpleNamespace(_scratch={'smag_rqv': cp.zeros((1, 1, 1), cp.float32)})
    cfg = SimpleNamespace(km_opt=4, diff_opt=1, diff_6th_opt=2,
                          diff_6th_factor=.12, diff_6th_slopeopt=1)

    def mixing(s, cfg, km, kh, specs, **kw):
        s._scratch['smag_rqv'].fill(2**24)

    def diff6(cfg, slot, f0, target, *args, **kw):
        target += cp.float32(1.)

    def prepare(s, cfg):
        specs = [(None, None, None, None, None, 'smag_rqv', '')]
        dycore._compute_wrf_smag_tendencies(s, cfg, None, None, specs, time_t=True)
        dycore.add_diff6_row(cfg, 'smag_rqv', None, s._scratch['smag_rqv'])

    monkeypatch.setattr(dycore, 'prepare_fixed_tendencies', prepare)
    monkeypatch.setattr(dycore, '_compute_wrf_smag_tendencies', mixing)
    monkeypatch.setattr(dycore, 'add_diff6_row', diff6)
    observer = ScalarProcessLedger(state, lambda x: x)
    with observer.observe():
        dycore.prepare_fixed_tendencies(state, cfg)
    # A separately evaluated unit filter would report +1; the actual rounded
    # addition to 2**24 is zero, and only that value closes the held tendency.
    assert observer.mixing_raw['qv']['sixth_order_realized'].item() == 0.
    assert observer.mixing_raw['qv']['mixing_package'].item() == 2**24
