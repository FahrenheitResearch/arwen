"""Launchers for the flux-form advection kernels (kernels/advection.cu).

5th-order upwind horizontal / 3rd-order upwind vertical fluxes on the C-grid,
periodic in x/y, zero flux through the domain top/bottom.  Each launcher ADDS
the flux divergence into its tendency array.  ``add_advection_tendencies``
assembles the coupled mass fluxes ((c1h*mu+c2h)*u etc.) for the Phase-1
advection-only path, which requires w == 0 (checked): its historical
vertical flux ``rw = -(mu*w)`` was only Omega-correct there, and Task 5
retired it -- the full ``dycore.step`` path advects theta/momentum with
the diagnosed Omega from ``dycore.stage_fluxes`` and the moisture scalars
with the acoustic time-averaged fluxes built on it (WRF sumflux; see
gpuwm.core.moist).

The launchers' ``coord`` argument is anything carrying the vertical-spacing
arrays ``rdnw`` (and ``rdn`` for w): a host ``VerticalCoord`` from
``gpuwm.core.grid`` or a ``DomainState`` (device FP32 copies).
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import (DTYPE, DomainState, mu_at_u_faces,
                              mu_at_v_faces)

_TPB = 128  # threads per block along i (i fastest)


def _launch(func: str, q, ru, rv, rw, tend, rdnw, fnm, fnp,
            dx: float, dy: float,
            nz: int, ny: int, nx: int, n_i: int, n_j: int, n_k: int,
            open_x: bool = False, open_y: bool = False,
            msf=None, has_msf=None, msf_shape=None,
            spec: bool = False) -> None:
    """Common launch path: 3D grid over the target points, i fastest.

    ``open_x``/``open_y`` select the kernels' WRF open-lateral-boundary
    path (degraded near-boundary stencils, loop-bound exclusions, non-cb
    open advective terms; see advection.cu header); the defaults are the
    bitwise-unchanged periodic path.  An open axis needs >= 7 cells so the
    WRF degrade bands cannot overlap.

    ``msf`` is the 2-D map factor at the tendency points (Task 3); when
    ``has_msf`` (default: msf given) the kernel weights the horizontal
    divergence by it on BOTH the periodic and the open/specified paths
    (WRF applies mrdx/mrdy unconditionally; the real74 production path is
    specified + Lambert msf).

    ``fnm``/``fnp`` are the stretched-grid half->full interpolation
    weights consumed by the 2nd-order vertical fallback faces and
    advect_w's horizontal advecting velocities (WRF fzm/fzp).  ``spec``
    is WRF's `specified` logical: no non-cb open terms, and the
    upstream-normal-wind substitution in the boundary-adjacent u/v
    fluxes.
    """
    if open_x and nx < 7:
        raise ValueError(f"open_x advection needs nx >= 7, got {nx}")
    if open_y and ny < 7:
        raise ValueError(f"open_y advection needs ny >= 7, got {ny}")
    if has_msf is None:
        has_msf = msf is not None
    if msf is None:
        msf = cp.ones(msf_shape, dtype=DTYPE)
    kern = get_kernel("advection", func)
    grid = ((n_i + _TPB - 1) // _TPB, n_j, n_k)
    kern(grid, (_TPB, 1, 1),
         (q, ru, rv, rw, tend, cp.asarray(rdnw, dtype=DTYPE),
          cp.asarray(fnm, dtype=DTYPE), cp.asarray(fnp, dtype=DTYPE), msf,
          DTYPE(1.0 / dx), DTYPE(1.0 / dy),
          np.int32(nz), np.int32(ny), np.int32(nx),
          np.int32(open_x), np.int32(open_y), np.int32(has_msf),
          np.int32(spec)))


def launch_flux_div_scalar(q, ru, rv, rw, tend, coord, dx, dy,
                           open_x=False, open_y=False,
                           msf=None, has_msf=None, spec=False) -> None:
    """Add the flux divergence of scalar ``q`` (mass points) into ``tend``.

    ``msf`` (optional) is the mass-point map factor ``msft (ny, nx)``."""
    nz, ny, nx = q.shape
    _launch("flux_div_scalar", q, ru, rv, rw, tend, coord.rdnw,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx, ny, nz, open_x, open_y,
            msf, has_msf, (ny, nx), spec)


def launch_flux_div_u(u, ru, rv, rw, tend, coord, dx, dy,
                      open_x=False, open_y=False,
                      msf=None, has_msf=None, spec=False) -> None:
    """Add the flux divergence of ``u`` (u-points) into ``tend``.

    ``msf`` (optional) is the u-point map factor ``msfu (ny, nx+1)``."""
    nz, ny, nxp1 = u.shape
    nx = nxp1 - 1
    _launch("flux_div_u", u, ru, rv, rw, tend, coord.rdnw,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx + 1, ny, nz, open_x, open_y,
            msf, has_msf, (ny, nx + 1), spec)


def launch_flux_div_v(v, ru, rv, rw, tend, coord, dx, dy,
                      open_x=False, open_y=False,
                      msf=None, has_msf=None, spec=False) -> None:
    """Add the flux divergence of ``v`` (v-points) into ``tend``.

    ``msf`` (optional) is the v-point map factor ``msfv (ny+1, nx)``."""
    nz, nyp1, nx = v.shape
    ny = nyp1 - 1
    _launch("flux_div_v", v, ru, rv, rw, tend, coord.rdnw,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx, ny + 1, nz, open_x, open_y,
            msf, has_msf, (ny + 1, nx), spec)


def launch_flux_div_w(w, ru, rv, rw, tend, coord, dx, dy,
                      open_x=False, open_y=False,
                      msf=None, has_msf=None, spec=False) -> None:
    """Add the flux divergence of ``w`` (w-points, interior levels only)
    into ``tend``.  Uses ``coord.rdn`` (w-level spacing), not ``rdnw``.

    ``msf`` (optional) is the mass-point map factor ``msft (ny, nx)``."""
    nzp1, ny, nx = w.shape
    nz = nzp1 - 1
    _launch("flux_div_w", w, ru, rv, rw, tend, coord.rdn,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx, ny, nz + 1, open_x, open_y,
            msf, has_msf, (ny, nx), spec)


def add_advection_tendencies(state: DomainState, cfg: RunConfig) -> None:
    """Accumulate advective tendencies into ``ru_t, rv_t, rw_t, rth_t``.

    Phase-1 advection-only path (``dycore.step(acoustic=False)``): builds
    the coupled mass fluxes ``ru = (c1h*<mu>_x + c2h) * u``,
    ``rv = (c1h*<mu>_y + c2h) * v`` in scratch (mu = mub2d + mu' averaged
    to the staggered faces by the shared ``state`` helpers, periodic), then
    launches the four flux-divergence kernels; theta is advected in coupled
    form (flux divergence of mu*theta with q = total theta).

    The eta-directed vertical flux is zero: the historical placeholder
    ``rw = -(mu*w)`` was dimensionally wrong as Omega = mu*d(eta)/dt except
    in its only valid regime w == 0, so Task 5 retired it -- this path now
    *requires* w == 0 (raises otherwise), and the full ``dycore.step`` path
    advects with the diagnosed Omega from ``dycore.stage_fluxes``.
    """
    nz, ny, nx = state.p.shape
    if bool(cp.any(state.w)):
        raise ValueError(
            "add_advection_tendencies (advection-only path) requires "
            "w == 0: its former vertical flux rw = -(mu*w) was only the "
            "eta mass flux Omega for w == 0.  The full dycore.step path "
            "advects with the diagnosed Omega from dycore.stage_fluxes.")
    mu = state.total_mu()                            # (ny, nx) total dry mass
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]

    ru = state.scratch((nz, ny, nx + 1), "adv_ru")
    rv = state.scratch((nz, ny + 1, nx), "adv_rv")
    rw = state.scratch((nz + 1, ny, nx), "adv_rw")

    ru[...] = (c1h * mu_at_u_faces(mu)[None] + c2h) * state.u
    rv[...] = (c1h * mu_at_v_faces(mu)[None] + c2h) * state.v
    if state.has_msf:                                # WRF couple_momentum
        ru /= state.msfu[None]
        rv /= state.msfv[None]
    rw[...] = 0.0                                    # Omega == 0 when w == 0

    launch_flux_div_scalar(state.total_theta(), ru, rv, rw, state.rth_t,
                           state, cfg.dx, cfg.dy,
                           msf=state.msft, has_msf=state.has_msf)
    launch_flux_div_u(state.u, ru, rv, rw, state.ru_t, state, cfg.dx, cfg.dy,
                      msf=state.msfu, has_msf=state.has_msf)
    launch_flux_div_v(state.v, ru, rv, rw, state.rv_t, state, cfg.dx, cfg.dy,
                      msf=state.msfv, has_msf=state.has_msf)
    launch_flux_div_w(state.w, ru, rv, rw, state.rw_t, state, cfg.dx, cfg.dy,
                      msf=state.msft, has_msf=state.has_msf)


def advect_scalar_rk3_periodic_test(q0, u0: float, dx: float,
                                    t_end: float, cfl: float) -> np.ndarray:
    """1-D uniform-flow RK3 advection driver over the GPU scalar kernel.

    Test harness only: runs ``flux_div_scalar`` with nz = ny = 1 (vertical
    and y fluxes vanish) and constant coupled flux ru = u0, i.e. mu = 1.
    Returns the advected profile after ``t_end`` as a host float32 array.
    """
    q0 = np.asarray(q0)
    nx = q0.size
    q = cp.asarray(q0, dtype=DTYPE).reshape(1, 1, nx)
    ru = cp.full((1, 1, nx + 1), u0, dtype=DTYPE)
    rv = cp.zeros((1, 2, nx), dtype=DTYPE)
    rw = cp.zeros((2, 1, nx), dtype=DTYPE)
    tend = cp.zeros((1, 1, nx), dtype=DTYPE)
    rdnw = np.full(1, -1.0)                          # unused: fz == 0 at nz=1
    kern = get_kernel("advection", "flux_div_scalar")
    grid = ((nx + _TPB - 1) // _TPB, 1, 1)
    args_tail = (cp.asarray(rdnw, dtype=DTYPE),
                 cp.zeros(1, dtype=DTYPE),               # fnm/fnp unused
                 cp.zeros(1, dtype=DTYPE),               # at nz == 1
                 cp.ones((1, nx), dtype=DTYPE),          # msf == 1
                 DTYPE(1.0 / dx),
                 DTYPE(1.0 / dx), np.int32(1), np.int32(1), np.int32(nx),
                 np.int32(0), np.int32(0), np.int32(0),  # periodic path
                 np.int32(0))

    def rhs(qs):
        tend[...] = 0
        kern(grid, (_TPB, 1, 1), (qs, ru, rv, rw, tend) + args_tail)
        return tend

    nsub = max(1, int(np.ceil(t_end / (cfl * dx / abs(u0)))))
    dt = t_end / nsub
    for _ in range(nsub):                            # WRF RK3 (dt/3, dt/2, dt)
        q1 = q + (dt / 3.0) * rhs(q)
        q2 = q + (dt / 2.0) * rhs(q1)
        q = q + dt * rhs(q2)
    return cp.asnumpy(q).ravel()
