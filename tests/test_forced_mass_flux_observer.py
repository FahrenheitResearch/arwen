"""The dry-mass observer on specified and nested grids.

On a boundary-forced grid the acoustic mass update divergence-updates only
the cells inside a one-row frame; the frame cells advance by ``rmu_t``
alone.  The observer must therefore integrate the faces between the frame
and the first active row.  Integrating the domain's outer faces instead
reported a residual of about 4 percent over a 720-step specified forecast
whose mass the model had in fact kept.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import requires_gpu


def _fake_state(mapped, nz=2, ny=5, nx=6):
    rng = np.random.default_rng(7)
    state = SimpleNamespace(
        has_msf=mapped,
        u=np.zeros((nz, ny, nx + 1), np.float32),
        v=np.zeros((nz, ny + 1, nx), np.float32),
        u_pp=np.zeros((nz, ny, nx + 1), np.float32),
        v_pp=np.zeros((nz, ny + 1, nx), np.float32),
        dnw=np.array([-0.4, -0.6], np.float32),
        c1h=np.array([1.0, 0.5], np.float32),
        c2h=np.array([0.0, 30000.0], np.float32),
        msfu=(1.0 + 0.3 * rng.random((ny, nx + 1))).astype(np.float32),
        msfv=(1.0 + 0.3 * rng.random((ny + 1, nx))).astype(np.float32),
        mup=np.zeros((ny, nx), np.float32),
        mub2d=(80000.0 + 500.0 * rng.random((ny, nx))).astype(np.float32))
    state.total_mu = lambda: state.mub2d + state.mup
    return state


def _cfg(*, nested=False, specified=True):
    return SimpleNamespace(specified=specified, nested=nested,
                           open_x=False, open_y=False, dx=1000.0, dy=1000.0)


def _active_cell_divergence(state, cfg):
    """Independent cell-by-cell sum of the kernel's active-cell mass
    tendency divided by the cell's own m**2 (its area weight)."""
    mu = state.total_mu()
    nz, ny, nx = state.u.shape[0], mu.shape[0], mu.shape[1]
    total = 0.0
    for k in range(nz):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                flux = []
                for a, b, wind, pert, msf in (
                        (mu[j, i - 1], mu[j, i], state.u[k, j, i],
                         state.u_pp[k, j, i], state.msfu[j, i]),
                        (mu[j, i], mu[j, i + 1], state.u[k, j, i + 1],
                         state.u_pp[k, j, i + 1], state.msfu[j, i + 1]),
                        (mu[j - 1, i], mu[j, i], state.v[k, j, i],
                         state.v_pp[k, j, i], state.msfv[j, i]),
                        (mu[j, i], mu[j + 1, i], state.v[k, j + 1, i],
                         state.v_pp[k, j + 1, i], state.msfv[j + 1, i])):
                    ref = (float(state.c1h[k]) * 0.5 * (float(a) + float(b))
                           + float(state.c2h[k])) * float(wind)
                    if state.has_msf:
                        ref /= float(msf)
                    flux.append(float(pert) + ref)
                total += float(state.dnw[k]) * (
                    (flux[1] - flux[0]) / cfg.dx
                    + (flux[3] - flux[2]) / cfg.dy)
    return total


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_outer_faces_carry_no_flux_into_active_cells(monkeypatch, mapped,
                                                     nested):
    from gpuwm.core import dycore
    monkeypatch.setattr(dycore, "cp", np)
    state = _fake_state(mapped)
    state.u[:, :, 0] = 30.0
    state.u[:, :, -1] = -12.0
    state.v[:, 0, :] = 8.0
    state.v[:, -1, :] = -20.0
    state.u_pp[:, :, 0] = 50.0
    cfg = _cfg(nested=nested, specified=not nested)
    # No active cell touches an outer face: the frame advances by rmu_t.
    assert float(dycore.boundary_mass_tendency_device(state, cfg)) == 0.0


@pytest.mark.parametrize("mapped", [False, True])
def test_forced_flux_equals_active_cell_divergence(monkeypatch, mapped):
    from gpuwm.core import dycore
    monkeypatch.setattr(dycore, "cp", np)
    state = _fake_state(mapped)
    rng = np.random.default_rng(11)
    for name in ("u", "v", "u_pp", "v_pp"):
        field = getattr(state, name)
        field[...] = (10.0 * rng.standard_normal(field.shape)).astype(
            np.float32)
    state.mup[...] = (300.0 * rng.standard_normal(state.mup.shape)).astype(
        np.float32)
    cfg = _cfg()
    expected = _active_cell_divergence(state, cfg)
    got = float(dycore.boundary_mass_tendency_device(state, cfg))
    assert got == pytest.approx(expected, rel=2e-6, abs=1e-3)


def test_open_grids_keep_their_outer_faces(monkeypatch):
    from gpuwm.core import dycore
    monkeypatch.setattr(dycore, "cp", np)
    state = _fake_state(False)
    state.u[:, :, 0] = 30.0
    cfg = SimpleNamespace(specified=False, nested=False, open_x=True,
                          open_y=False, dx=1000.0, dy=1000.0)
    got = float(dycore.boundary_mass_tendency_device(state, cfg))
    mu_west = state.total_mu()[:, 0]
    expected = -float(np.sum(
        state.dnw[:, None] / cfg.dx
        * (state.c1h[:, None] * mu_west[None] + state.c2h[:, None]) * 30.0))
    assert got == pytest.approx(expected, rel=1e-6)


@requires_gpu
@pytest.mark.gpu
def test_specified_step_mass_closes_against_observer():
    """Through the production step(): a specified grid whose only wind is
    on the outer faces keeps its dry mass to rounding, and the observer
    says so instead of reporting the outer-face flux as a loss."""
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import domain_mass_measure, step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    cfg = RunConfig(nx=12, ny=10, nz=6, dx=1000.0, dy=1000.0,
                    ztop=12000.0, dt=5.0, run_seconds=5.0, specified=True)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 290.0 + 0.004 * np.asarray(z),
                           cfg.p_surf, cfg.ztop)
    states = [init_at_rest(cfg, coord, base) for _ in range(3)]
    for s in states:
        s.u[:, :, 0] = cp.float32(10.0)
        s.u[:, :, -1] = cp.float32(-4.0)
        s.v[:, 0, :] = cp.float32(6.0)
    boundaries = build_state_lateral_boundaries(
        states, [0.0, 21600.0, 43200.0])
    attach_lateral_boundaries(states[0], boundaries)
    state = states[0]
    mass0 = domain_mass_measure(state)
    increments = []
    step(state, cfg, mass_flux_observer=increments.append)
    mass1 = domain_mass_measure(state)
    assert len(increments) == cfg.time_step_sound
    observed = float(sum(increments))
    residual = abs((mass1 - mass0) - observed) / mass0
    assert residual < 1.0e-7, (mass1 - mass0, observed)
