"""Constant-K diffusion and Rayleigh damping layer (kernels/diffusion.cu).

``add_diffusion_tendencies`` adds simple constant-eddy-viscosity 2nd-order
diffusion of u, v, w and theta' (WRF ``diff_opt=1`` style) to the coupled
slow tendencies whenever ``cfg.khdif > 0 or cfg.kvdif > 0``.  The Laplacian
is evaluated on coordinate surfaces horizontally and in physical space
vertically, using per-level base-state dz.  Design choice (per plan): the
mixing tendencies are recomputed on *every* RK stage from that stage's
estimate, whereas WRF computes them ONCE per timestep on RK stage 1 (from
the time-t fields, into the fixed ``*_tendf`` accumulators) and applies
that same tendency on every stage — Straka's published setups use
per-stage mixing and the benchmark tolerances absorb the difference.
Theta is diffused in perturbation form (theta' = theta - thb) so the base
stratification is not mixed; u, v, w have a zero (at-rest) base state, so
perturbation and full fields coincide.

Comparison semantics (Straka investigation): gpuwm's constant-K
deliberately follows Straka's definition — the configured K applies to
theta (and momentum) exactly as given — whereas WRF's ``km_opt=1`` gives
scalars ``khdif/prandtl`` with prandtl = 1/3 (i.e. 3x the momentum
diffusivity; v4.6.1 module_diffusion_em.F ``khdq = 3.*khdif`` and
share/module_model_constants.F).  Cross-model comparisons must record the
effective per-field diffusivities, not just the namelist K.

``apply_rayleigh_damping`` is a relaxational sponge utility (WRF
``damp_opt=2`` style, restricted to fields with a zero/at-rest reference):
w and theta' are implicitly relaxed toward the at-rest base state above
``ztop - zdamp``, ``f_new = f / (1 + dt*tau(z))``, ``tau = dampcoef *
sin^2(pi/2 * (z - (ztop - zdamp)) / zdamp)``.  Phase 2 Task 4 demoted it
from the model step: it no longer touches u/v (that conflated WRF's
``damp_opt=2`` relaxation-to-base-state with ``damp_opt=3`` and decelerated
any mean flow through the sponge), and ``dycore.step`` no longer calls it —
``damp_opt=3`` now selects the true Klemp-Dudhia-Hassiotis implicit w-only
damper inside the acoustic w solve (gpuwm.core.acoustic / acoustic.cu
``advance_w_phi``).  Only ``damp_opt == 3`` engages this utility when called
directly; any other value is a no-op.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import (DTYPE, DomainState, mu_at_u_faces,
                              mu_at_v_faces)

_TPB = 128  # threads per block along i (i fastest)


def _dz_spacings(zf, stagger: str):
    """Inverse vertical spacings for :func:`launch_add_diff2`, float64.

    ``zf (nzf,)`` are the full-level (w-level) heights.  Returns
    ``(rdzf, rdzc)``: inverse center-to-center spacings at the ``nlev-1``
    interior faces and inverse cell thicknesses at the ``nlev`` levels of
    the field.  Half-level fields (nlev = nzf-1) live at the layer
    midpoints; w-staggered fields (nlev = nzf) live at ``zf`` itself, with
    unused boundary ``rdzc`` entries set to zero.
    ``gpuwm.verify.npref._diff2_spacings`` re-derives the same spacings
    independently for the comparison tests (final-review carry-over T18: a
    shared conceptual error must not be able to cancel there).
    """
    zf = np.asarray(zf, dtype=np.float64)
    zh = 0.5 * (zf[:-1] + zf[1:])
    if stagger == "z":
        rdzf = 1.0 / np.diff(zf)
        rdzc = np.zeros(zf.size)
        rdzc[1:-1] = 1.0 / (zh[1:] - zh[:-1])
    else:
        rdzf = 1.0 / np.diff(zh)
        rdzc = 1.0 / np.diff(zf)
    return rdzf, rdzc


def launch_add_diff2(f, tend, kh, kv, dx, dy, zf, stagger: str = "") -> None:
    """Add ``kh*(f_xx + f_yy) + kv*(f_z)_z`` into ``tend`` (uncoupled).

    ``stagger`` selects the grid position of ``f``: ``""`` mass points
    (nz, ny, nx), ``"x"`` u-points (nz, ny, nx+1), ``"y"`` v-points
    (nz, ny+1, nx), ``"z"`` w-points (nz+1, ny, nx; boundary levels get no
    tendency).  ``zf (nz+1,)`` are the full-level base-state heights.
    """
    nlev, nys, nxs = f.shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    rdzf, rdzc = _dz_spacings(zf, stagger)
    kern = get_kernel("diffusion", "add_diff2")
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    kern(grid, (_TPB, 1, 1),
         (f, tend, DTYPE(kh), DTYPE(kv),
          DTYPE(1.0 / dx ** 2), DTYPE(1.0 / dy ** 2),
          cp.asarray(rdzf, dtype=DTYPE), cp.asarray(rdzc, dtype=DTYPE),
          np.int32(nlev), np.int32(ny), np.int32(nys),
          np.int32(nx), np.int32(nxs), np.int32(1 if stagger == "z" else 0)))


def add_diffusion_tendencies(state: DomainState, cfg: RunConfig) -> None:
    """Accumulate mu-coupled diffusion of u, v, w, theta' into the slow
    tendencies (hybrid coupling ``c1h*mu + c2h`` on half levels /
    ``c1f*mu + c2f`` on w levels).  No-op unless ``cfg.khdif > 0 or
    cfg.kvdif > 0``."""
    if cfg.khdif <= 0.0 and cfg.kvdif <= 0.0:
        return
    if state.phb.ndim != 1:
        raise NotImplementedError(
            "constant-K diffusion assumes a flat base state (1-D phb for "
            "the vertical spacings); terrain cases must run with "
            "khdif = kvdif = 0")
    zf = cp.asnumpy(state.phb).astype(np.float64) / c.G
    mu = state.total_mu()                            # (ny, nx) total dry mass
    mux = mu_at_u_faces(mu)                          # shared face helpers
    muy = mu_at_v_faces(mu)
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    c1f = state.c1f[:, None, None]
    c2f = state.c2f[:, None, None]

    for f, tend, muf, cc1, cc2, slot, stag in (
            (state.u, state.ru_t, mux, c1h, c2h, "diff_u", "x"),
            (state.v, state.rv_t, muy, c1h, c2h, "diff_v", "y"),
            (state.w, state.rw_t, mu, c1f, c2f, "diff_w", "z"),
            (state.thp, state.rth_t, mu, c1h, c2h, "diff_th", "")):
        tmp = state.scratch(f.shape, slot)
        tmp[...] = 0
        launch_add_diff2(f, tmp, cfg.khdif, cfg.kvdif, cfg.dx, cfg.dy, zf,
                         stagger=stag)
        tend += (cc1 * muf[None] + cc2) * tmp


def _damp_factors(z, cfg: RunConfig) -> np.ndarray:
    """Per-level implicit factors 1/(1 + dt*tau(z)), float32 host array."""
    arg = np.clip((np.asarray(z, dtype=np.float64) - (cfg.ztop - cfg.zdamp))
                  / cfg.zdamp, 0.0, 1.0) * (0.5 * np.pi)
    tau = cfg.dampcoef * np.sin(arg) ** 2
    return (1.0 / (1.0 + cfg.dt * tau)).astype(np.float32)


def apply_rayleigh_damping(state: DomainState, cfg: RunConfig) -> None:
    """Implicitly relax w and theta' toward the at-rest base state in the
    upper damping layer (see module docstring; demoted utility, not called
    by ``dycore.step``).  No-op unless ``cfg.damp_opt == 3``."""
    if cfg.damp_opt != 3:
        return
    if state.phb.ndim != 1:
        raise NotImplementedError(
            "apply_rayleigh_damping assumes a flat base state (1-D phb); "
            "with terrain use the damp_opt=3 implicit w damper inside the "
            "acoustic solve instead")
    z_full = cp.asnumpy(state.phb).astype(np.float64) / c.G
    z_half = 0.5 * (z_full[:-1] + z_full[1:])
    fac_half = cp.asarray(_damp_factors(z_half, cfg))
    fac_full = cp.asarray(_damp_factors(z_full, cfg))
    kern = get_kernel("diffusion", "rayleigh_damp")
    for f, fac in ((state.thp, fac_half), (state.w, fac_full)):
        nlev = f.shape[0]
        plane = f.shape[1] * f.shape[2]
        grid = ((nlev * plane + _TPB - 1) // _TPB, 1, 1)
        kern(grid, (_TPB, 1, 1), (f, fac, np.int32(nlev), np.int32(plane)))


def diffuse_only_test(q0, K: float, dx: float, dt: float,
                      t_end: float) -> np.ndarray:
    """1-D periodic pure-diffusion driver over the GPU kernel (test harness).

    Forward-Euler steps of ``dq/dt = K * d2q/dx2`` on a single-level,
    single-row grid (vertical and y terms vanish identically).  Returns the
    diffused profile after ``t_end`` as a host float32 array.
    """
    q0 = np.asarray(q0)
    nx = q0.size
    q = cp.asarray(q0, dtype=DTYPE).reshape(1, 1, nx)
    tend = cp.zeros_like(q)
    zf = np.array([0.0, dx])                         # single layer, kv unused
    for _ in range(int(round(t_end / dt))):
        tend[...] = 0
        launch_add_diff2(q, tend, kh=K, kv=0.0, dx=dx, dy=dx, zf=zf)
        q += DTYPE(dt) * tend
    return cp.asnumpy(q).ravel()
