# tests/test_anisotropic_mixing_w_stability.py
"""Measured: per-axis mixing lengths can amplify the 2dx mode in w.

WRF hands ``horizontal_diffusion_w_2`` the VERTICAL exchange coefficient
(``dyn_em/module_diffusion_em.F:3003``, dummy spelled ``xkmv`` at
``:3524``), and with ``mix_isotropic = 0`` ``smag_km`` both builds AND
caps that coefficient on the vertical length scale (``:1890-1900``)::

    xkmv = min( c_s^2 * dz^2 * |S| ,  mix_upper_bound * dz^2 / dt )

The operator that consumes it then differences over the HORIZONTAL
spacing.  Nothing in the chain compares the coefficient against that
spacing, so the largest ratio the operator can be handed is
``mix_upper_bound * (dz_max/dx)^2`` -- independent of ``dt``, because the
cap carries ``1/dt`` and the ratio carries ``dt``.  An explicit Laplacian
multiplies a 2-grid-interval mode by ``1 - 4*K*dt/dx^2`` per step, so
past 1/4 the sign flips and past 1/2 the mode grows: the diffusion
amplifies exactly the structure it exists to remove.

This file MEASURES that, rather than modelling it.  Seed a 2dx/2dy
checkerboard in ``w``, run the model's own once-per-step held-tendency
builder, apply the increment the dycore would apply over one ``dt``, and
read the checkerboard's amplification straight off the result.  The pin
is that the isotropic-length configuration never amplifies; the control
is that the per-axis one does, at the same grid, so the instrument is
known to be able to fail.

The closed form of the same statement, and the load-time advisory built
on it, are in ``tests/test_anisotropic_mixing_advisory.py`` -- separately,
because those are CPU and this is not.

Generic throughout: the criterion is grid geometry, not a case.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.config import EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT

# A grid whose layers are ~5.4x deeper than it is wide, which is what a
# shared mesoscale ladder looks like from a hectometre child.  The
# closed-form ratio here is 0.1 * 5.37^2 = 2.9, an order of magnitude
# past the limit.
DX = 100.0
DT = 0.5
CS = 0.25
MIX_UPPER_BOUND = 0.1
NZ, NY, NX = 12, 16, 16
DZ = 537.0
SHEAR = 0.004        # so the deformation invariant is storm-like, not zero
#: Amplitudes in m/s.  The mode's growth is not linear in the amplitude:
#: |S| rises with it, so K rises with it, and the ratio that matters is
#: reached only once the flow is violent.  A single small amplitude would
#: have reported this operator as stable.
AMPLITUDES = (10.0, 20.0, 40.0, 80.0, 160.0)


def _build(mix_isotropic: int):
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    cfg = RunConfig(
        nx=NX, ny=NY, nz=NZ, dx=DX, dy=DX, ztop=NZ * DZ,
        dt=DT, run_seconds=0.0,
        km_opt=3, c_s=CS, mix_isotropic=mix_isotropic,
        mix_upper_bound=MIX_UPPER_BOUND, isfflx=0,
        bl_pbl_physics=0, sf_sfclay_physics=0, sf_surface_physics=0,
        diff_6th_opt=0, moist=True, mp_physics=0)
    coord = make_vertical_coord(NZ)
    base = make_base_state(coord, lambda z: 300.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(
        cfg, coord, base, lambda z: 0.008 + 0.0 * np.asarray(z))
    return cfg, state


def _measure(mix_isotropic: int, amplitude: float) -> dict:
    """One checkerboard amplification, through the dycore's own fold."""

    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.dycore import prepare_fixed_tendencies

    cfg, state = _build(mix_isotropic)
    k = NZ // 2

    jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    chk = ((-1.0) ** (ii + jj)).astype(np.float32)

    w = np.zeros((NZ + 1, NY, NX), dtype=np.float32)
    w[k] = amplitude * chk
    state.w[...] = cp.asarray(w)
    u = np.zeros((NZ, NY, NX + 1), dtype=np.float32)
    for kk in range(NZ):
        u[kk] = SHEAR * (kk - NZ / 2) * DZ
    u[:, :, NX] = u[:, :, 0]
    state.u[...] = cp.asarray(u)

    for name in ("u", "v", "w", "thp", "php", "mup"):
        getattr(state, name + "0")[...] = getattr(state, name)
    for name in ("qv", "qc", "qr"):
        getattr(state, name + "0")[...] = getattr(state, name)

    update_diagnostics(state, cfg.hypsometric_opt)
    prepare_fixed_tendencies(state, cfg)

    tend = cp.asnumpy(state.scratch(state.w.shape, "smag_rw"))[k]
    kmv = cp.asnumpy(state.scratch(state.p.shape, "smag_kmv"))[k]
    kmh = cp.asnumpy(state.scratch(state.p.shape, "smag_km"))[k]
    mu = cp.asnumpy(state.mub2d + state.mup0)
    c1f = float(cp.asnumpy(state.c1f)[k])
    c2f = float(cp.asnumpy(state.c2f)[k])
    coupling = c1f * mu + c2f

    dw = DT * tend / coupling                      # the dycore's own fold
    new = amplitude * chk + dw
    # Projection onto the checkerboard: this IS the mode's amplification.
    amp = float((chk * new).mean() / (chk * chk).mean() / amplitude)
    # Layer depth read from the live geopotential -- the same quantity the
    # kernel's rdzw is built from, not the nominal one.
    phb = cp.asnumpy(state.phb)
    php = cp.asnumpy(state.php)
    z = (phb if phb.ndim == 3 else phb[:, None, None]) + php
    return {
        "amplification": amp,
        "kmh": float(kmh.mean()),
        "kmv": float(kmv.mean()),
        "kmv_dt_over_dx2": float(kmv.mean()) * DT / DX / DX,
        "dz": float((z[k + 1] - z[k]).mean() / 9.81),
    }


@requires_gpu
def test_isotropic_mixing_length_never_amplifies_the_2dx_w_mode():
    """The pin.  A diffusion operator must not amplify: |amp| <= 1.

    ``mix_isotropic = 1`` builds one length from ``(dx*dy*dz)^(1/3)`` and
    caps the vertical coefficient against the horizontal one, so what the
    horizontal operator receives is bounded by the spacing it differences
    over.  That has to hold at every amplitude, including the violent
    ones -- this is the regression.
    """

    rows = [_measure(1, amplitude) for amplitude in AMPLITUDES]
    for amplitude, row in zip(AMPLITUDES, rows):
        assert abs(row["amplification"]) <= 1.0, (
            f"isotropic mixing length amplified the 2dx w mode at "
            f"A = {amplitude} m/s: amplification {row['amplification']:.4f}, "
            f"K*dt/dx^2 = {row['kmv_dt_over_dx2']:.4f}")
    # The mechanism, not just the outcome: isotropic means the two
    # coefficients ARE one coefficient.
    for row in rows:
        assert row["kmv"] == pytest.approx(row["kmh"], rel=1e-5), (
            "mix_isotropic = 1 must hand the horizontal operator the same "
            "coefficient it built on the horizontal spacing")


@requires_gpu
def test_per_axis_mixing_length_amplifies_it_at_the_same_grid():
    """The control that proves the instrument can fail.

    Same grid, same amplitudes, ``mix_isotropic = 0``: the vertical
    coefficient is built on a layer 5.4x deeper than the grid is wide,
    and the mode the operator should be erasing comes back inverted and
    larger.  If this ever stops amplifying, the pin above has stopped
    measuring anything and both tests need re-deriving.
    """

    rows = [_measure(0, amplitude) for amplitude in AMPLITUDES]
    worst = max(rows, key=lambda r: abs(r["amplification"]))
    assert abs(worst["amplification"]) > 1.0, (
        "per-axis mixing lengths no longer amplify the 2dx w mode at "
        f"dz/dx = {worst['dz'] / DX:.2f}; amplifications "
        f"{[round(r['amplification'], 4) for r in rows]}")
    assert worst["kmv"] > 4.0 * worst["kmh"], (
        "the amplification has to come from the VERTICAL coefficient being "
        "the larger one, which is the whole mechanism")
    assert worst["kmv_dt_over_dx2"] > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
