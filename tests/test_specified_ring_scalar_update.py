"""The specified ring of a boundary-forced scalar moves by its boundary tendency only.

WRF ``rk_update_scalar`` (dyn_em/module_em.F:1671-1678, used at :1695-1700
and :1746-1751) applies the advective tendency only INSIDE the specified
zone of a specified or nested domain; the ring itself takes ``sc_tend``
alone, which on the final positive-definite stage is already folded into
the time-t scalar.  ``advect_scalar_pd`` writes the ring's vertical
divergence all the same (module_advect_em.F:7787-7791) and WRF never uses
it.  The port's final PD stage applied it, so on the ring a vertical mass
flux converging into a layer had no horizontal divergence to balance it and
the scalar there grew by a fixed fraction every step.  A supplied aerosol
number compounded that way in the model-top layer of a specified parent's
boundary row until the full-state health gate stopped the forecast at
1.03e15 per kg.

These tests drive ``advance_scalars_stage`` directly, so they read the RK
update itself.  On a specified domain WRF's end-of-step ``spec_bdy_final``
does not rewrite a scalar-array species such as the aerosol, so the RK
update is the only thing that holds that ring on its table; the whole-step
test runs the finalizer every step to pin exactly that.
"""
import numpy as np
import pytest

from gpuwm.config import RunConfig


def _cfg(**changes):
    values = dict(nx=13, ny=12, nz=6, dx=12000., dy=12000., ztop=10000.,
                  dt=1., run_seconds=120., moist=True, mp_physics=28,
                  specified=True, aer_init_opt=1, wif_input_opt=1)
    return RunConfig(**(values | changes))


def _specified_aerosol_state(cfg):
    """A resting mp=28 domain whose aerosol tables rise 25 percent a minute."""
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (
        attach_lateral_boundaries, build_lateral_boundaries,
        domain_boundary_snapshot)

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.qv[:] = .01
    state.qv0[:] = state.qv
    for name, value in (('nwfa', 2e8), ('nifa', 2e5), ('nc', 1e5)):
        getattr(state, name)[:] = value
        getattr(state, name + '0')[:] = value
    state.mup0[:] = state.mup
    assert state._external_scalar_boundary_fields == ('qv', 'nwfa', 'nifa')
    first = domain_boundary_snapshot(state)
    for name in ('nwfa', 'nifa'):
        getattr(state, name)[:] *= 1.25
    second = domain_boundary_snapshot(state)
    for name in ('nwfa', 'nifa'):
        getattr(state, name)[:] = getattr(state, name + '0')
    attach_lateral_boundaries(
        state, build_lateral_boundaries([first, second], [0., 60.]))
    return state


def _ring(field, sz=1):
    """Every specified-zone cell of a (nz, ny, nx) field, level by level."""
    return np.concatenate([
        field[:, :sz, :].reshape(field.shape[0], -1),
        field[:, -sz:, :].reshape(field.shape[0], -1),
        field[:, sz:-sz, :sz].reshape(field.shape[0], -1),
        field[:, sz:-sz, -sz:].reshape(field.shape[0], -1)], axis=1)


def _lid_converging_flux(cp, state):
    """No horizontal transport and a uniform eta mass flux on the interior
    faces (zero at the surface and the lid): the bottom layer diverges and
    the lid layer converges in EVERY column, ring included."""
    ru = cp.zeros_like(state.u)
    rv = cp.zeros_like(state.v)
    ww = cp.zeros_like(state.w)
    ww[1:-1] = -400.
    return ru, rv, ww


@pytest.mark.gpu
def test_pd_final_stage_ring_takes_only_the_boundary_tendency_under_vertical_flux():
    cp = pytest.importorskip('cupy')
    from gpuwm.core.moist import advance_scalars_stage

    cfg = _cfg()
    assert cfg.moist_adv_opt == 1, 'the PD final stage is the path under test'
    state = _specified_aerosol_state(cfg)
    ru, rv, ww = _lid_converging_flux(cp, state)
    advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff=1., final=True,
                          apply_relax=True)
    for name, value in (('nwfa', 2e8), ('nifa', 2e5)):
        ring = _ring(cp.asnumpy(getattr(state, name)))
        np.testing.assert_allclose(ring, value * (1. + .25 / 60.), rtol=3e-7,
                                   err_msg=f'{name} ring moved by advection')
    # The interior still advects: the lid layer of an interior column has
    # gained the converging flux and the bottom layer has lost it.
    centre = cp.asnumpy(state.nwfa)[:, cfg.ny // 2, cfg.nx // 2]
    assert centre[-1] > 2e8 * (1. + 1e-3)
    assert centre[0] < 2e8 * (1. - 1e-3)


@pytest.mark.gpu
def test_repeated_pd_steps_do_not_compound_the_ring_lid_layer():
    """Twenty final stages under the same converging flux: the ring's lid
    value follows the boundary table linearly instead of compounding."""
    cp = pytest.importorskip('cupy')
    from gpuwm.core.moist import advance_scalars_stage

    cfg = _cfg()
    state = _specified_aerosol_state(cfg)
    ru, rv, ww = _lid_converging_flux(cp, state)
    steps = 20
    for _ in range(steps):
        for name in ('qv', 'nc', 'nwfa', 'nifa'):
            getattr(state, name + '0')[:] = getattr(state, name)
        advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff=1., final=True,
                              apply_relax=True)
    lid_ring = _ring(cp.asnumpy(state.nwfa))[-1]
    np.testing.assert_allclose(lid_ring, 2e8 * (1. + steps * .25 / 60.),
                               rtol=5e-6)


@pytest.mark.gpu
def test_non_pd_stages_already_leave_the_ring_to_the_boundary_tendency():
    """The first two RK stages take flux_div_scalar, which writes nothing on
    the ring; this pins that the final-stage repair made the two paths
    agree rather than changing the stages that were already right."""
    cp = pytest.importorskip('cupy')
    from gpuwm.core.moist import advance_scalars_stage

    cfg = _cfg()
    state = _specified_aerosol_state(cfg)
    ru, rv, ww = _lid_converging_flux(cp, state)
    advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff=1., final=False,
                          apply_relax=True)
    ring = _ring(cp.asnumpy(state.nwfa))
    np.testing.assert_allclose(ring, 2e8 * (1. + .25 / 60.), rtol=3e-7)


@pytest.mark.gpu
def test_the_rk_update_alone_holds_the_aerosol_ring_through_whole_steps():
    """Twenty whole steps (final stage, then the end-of-step finalizer) under
    the same converging flux.  The ring is read BEFORE each finalizer call,
    so the RK update has to hold it on the table: on a specified domain the
    finalizer does not put the aerosol back, as WRF's does not."""
    cp = pytest.importorskip('cupy')
    from gpuwm.core.moist import advance_scalars_stage
    from gpuwm.ingest.lateral_bc import apply_state_boundary_values

    cfg = _cfg()
    state = _specified_aerosol_state(cfg)
    ru, rv, ww = _lid_converging_flux(cp, state)
    for step in range(20):
        for name in ('qv', 'nc', 'nwfa', 'nifa'):
            getattr(state, name + '0')[:] = getattr(state, name)
        advance_scalars_stage(state, cfg, ru, rv, ww, dt_eff=1., final=True,
                              apply_relax=True)
        for name, value in (('nwfa', 2e8), ('nifa', 2e5)):
            np.testing.assert_allclose(
                _ring(cp.asnumpy(getattr(state, name))),
                value * (1. + (step + 1) * .25 / 60.), rtol=5e-6,
                err_msg=f'{name} ring left its table at step {step + 1}')
        apply_state_boundary_values(state, cfg, elapsed_seconds=step + 1.)
