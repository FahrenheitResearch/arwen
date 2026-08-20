"""WRF-ARW RK3 split-explicit time-stepping driver.

One ``step`` advances the state by ``cfg.dt`` with the three-stage
Runge-Kutta scheme of Wicker & Skamarock (ARW Tech Note sec. 3.1.2): each
stage recomputes the slow tendencies R^t* from the latest stage estimate
(the acoustic reference state t*), then re-integrates the *time-t* fields
forward with acoustic substeps —

  stage 1:  1 substep,             dtau = dt/3
  stage 2:  ns/2 substeps,         dtau = dt/ns
  stage 3:  ns substeps,           dtau = dt/ns

with ns = ``cfg.time_step_sound``.  The slow forcings are the flux-form
advection (vertical flux = the diagnosed eta mass flux Omega, WRF
``calc_ww_cp``), the horizontal pressure gradient of the t* state (WRF
``horizontal_pressure_gradient``), the buoyancy/vertical-pressure-gradient
term g*(d(p')/d(eta) - mu') (WRF ``pg_buoy_w``) and the advective-form
geopotential RHS with its g*w term (WRF ``rhs_ph``); the perturbation
pressure-gradient and buoyancy terms live inside the acoustic substeps
(gpuwm.core.acoustic, Tech Note eqns 3.4-3.14).

``acoustic=False`` keeps the Phase-1 advection-only transport path (mu' and
phi' frozen) used by the pure-advection verification test.

Time-step rule (WRF guidance): ``dt <= ~6 s per km of dx``; benchmarks run
at CFL ~ 0.3-0.5.
"""

from __future__ import annotations

from functools import lru_cache
import math

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig, validate_km_opt
from gpuwm.core import constants as c
from gpuwm.core.acoustic import (prepare_acoustic_coefficients,
                                 prepare_acoustic_substep_launch,
                                 prepare_moist_cq)
from gpuwm.core.advection import (add_advection_tendencies,
                                  launch_flux_div_scalar, launch_flux_div_u,
                                  launch_flux_div_v, launch_flux_div_w)
from gpuwm.core.diagnostics import update_diagnostics
from gpuwm.core.diffusion import add_diffusion_tendencies
from gpuwm.core.kernels import get_kernel
from gpuwm.core.microphysics import apply as apply_microphysics
from gpuwm.core.moist import (SPECIES, WRF_MOIST_ARRAY_SPECIES,
                              extra_moist_species, advance_scalars_stage)
from gpuwm.core.moist_n2_mutation import calc_n2_kernel
from gpuwm.core.physics import physics_enabled
from gpuwm.core import tke_budget
from gpuwm.core.state import (DTYPE, DomainState, mu_at_u_faces,
                              mu_at_v_faces)
from gpuwm.core.uh_diag import update_up_heli_max
from gpuwm.ingest.lateral_bc import (apply_state_boundary_values,
                                     apply_state_lateral_boundaries)

_TPB = 128  # threads per block along i (i fastest)

#: Prognostic fields saved to their *0 time-t copies at the start of a step.
_PROGNOSTICS = ("u", "v", "w", "thp", "php", "mup")

#: Slow-tendency accumulators zeroed at the start of every RK stage.
_TENDENCIES = ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t")

#: WRF turbulent Prandtl number (share/module_model_constants.F:
#: prandtl = 1./3.0) -- scalars mix with K_h = K_m/prandtl = 3*K_m.
_PRANDTL = 1.0 / 3.0

#: km_opt=4 application kernels by field stagger (smag2d.cu).
_SMAG_HD = {"": "smag_hd_s", "x": "smag_hd_u",
            "y": "smag_hd_v", "z": "smag_hd_w"}

#: WRF v4.6.1 diff_opt=2 stress/scalar kernels used by production.  The
#: legacy ``_SMAG_HD`` entry points remain available only for the frozen flat
#: Phase-2 kernel-oracle tests.
_WRF_SMAG_HD = {"": "wrf_smag_hd_s", "x": "wrf_smag_hd_u",
                "y": "wrf_smag_hd_v", "z": "wrf_smag_hd_w"}


def _b3(arr: cp.ndarray) -> cp.ndarray:
    """Base profile broadcast: 1-D flat column -> (n, 1, 1); 3-D through."""
    return arr if arr.ndim == 3 else arr[:, None, None]


def _boundary_x(cfg: RunConfig) -> bool:
    return cfg.open_x or _boundary_forced(cfg)


def _boundary_y(cfg: RunConfig) -> bool:
    return cfg.open_y or _boundary_forced(cfg)


def _boundary_forced(cfg: RunConfig) -> bool:
    """WRF ``specified_bdy``: external specified OR nested forcing."""
    return bool(getattr(cfg, "specified", False)
                or getattr(cfg, "nested", False))


def _omega_ref(state: DomainState, cfg: RunConfig,
               ru: cp.ndarray, rv: cp.ndarray) -> cp.ndarray:
    """Reference eta mass flux Omega (nz+1, ny, nx) at w levels.

    WRF ``calc_ww_cp``: integrate the continuity equation over the column
    for the (eta-uniform) d(mu)/dt, then diagnose Omega level by level from
    the coupled horizontal mass fluxes ``ru``/``rv``, weighting the column-
    mass tendency by the hybrid c1h (``ww(k) = ww(k-1) -
    dnw(k-1)*c1h(k-1)*dmdt - divv(k-1)``).  Omega = 0 at the surface and
    (by construction, since sum(dnw*c1h) = -1) at the model top.

    Map factors (Task 3): the layer divergence carries WRF's ``msftx``
    weight (``divv = msft*dnw*(d(ru)/dx + d(rv)/dy)`` with ru/rv already
    msf-coupled), making ``ww`` the tech-note Omega = mu*deta/dt / m_y.
    """
    nz, ny, nx = state.p.shape
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    ww = state.scratch((nz + 1, ny, nx), "rk_ww")
    dnw = state.dnw[:, None, None]
    c1h = state.c1h[:, None, None]
    divv = dnw * (rdx * (ru[:, :, 1:] - ru[:, :, :-1])
                  + rdy * (rv[:, 1:, :] - rv[:, :-1, :]))
    if state.has_msf:                                  # WRF calc_ww_cp msftx
        divv *= state.msft[None]
    dmdt = divv.sum(axis=0)                            # (ny, nx)
    ww[0] = 0.0
    ww[1:nz] = -cp.cumsum((c1h * dnw)[:nz - 1] * dmdt[None] + divv[:nz - 1],
                          axis=0)
    ww[nz] = 0.0
    return ww


@lru_cache(maxsize=None)
def _couple_momentum_kernel(has_msf: bool):
    """WRF ``couple_momentum`` for one staggering, in a single pass.

    Replaces five full-size ufunc launches per component -- the c1h*muface
    multiply, the c2h add, the wind multiply, the copy into the flux scratch
    and the map-factor divide -- along with the two staggered temporaries
    they round through.  Same discipline as
    :func:`gpuwm.core.moist._update_scalar_kernel`: explicit round-to-nearest
    intrinsics plus ``-fmad=false``, because the chain this replaces rounds
    to FP32 at every operator boundary, and the map-factor divide stays a
    separate division AFTER the multiply rather than folding into it.
    """
    params = ["T wind", "raw T c1h", "raw T c2h", "raw T muface"]
    body = ["const int lev = static_cast<int>(i) / ncol;",
            "const int col = static_cast<int>(i) % ncol;",
            "T v = __fmul_rn(__fadd_rn(__fmul_rn(c1h[lev], muface[col]), "
            "c2h[lev]), wind);"]
    if has_msf:
        params.append("raw T msf")
        body.append("v = __fdiv_rn(v, msf[col]);")
    params.append("int32 ncol")
    body.append("flux = v;")
    return cp.ElementwiseKernel(", ".join(params), "T flux", "\n".join(body),
                                "gpuwm_couple_momentum",
                                options=("-fmad=false",))


def stage_fluxes(state: DomainState, cfg: RunConfig
                 ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Public RK-stage transport surface (Task 5): ``(ru, rv, ww)``.

    The stage's coupled horizontal mass fluxes ``ru = (c1h*<mu>_x +
    c2h)*u/msfu`` / ``rv = (c1h*<mu>_y + c2h)*v/msfv`` (WRF
    ``couple_momentum``; the msf divisions are identity with the default
    map factors) and the diagnosed eta mass flux Omega ``ww`` (WRF
    ``calc_ww_cp``), all evaluated at the current stage reference t*.
    These are the fluxes that advect theta and momentum in
    ``_add_slow_tendencies`` — exactly WRF ``rk_tendency``'s ru/rv/ww.
    The moisture scalars do NOT use them directly: WRF advects scalars
    with the acoustic-substep time-averaged fluxes ru_m/rv_m/ww_m
    (``sumflux``, "needed for consistent mass-conserving scalar
    advection"), which ``step`` accumulates over each stage's substeps as
    mean(u'') + these reference fluxes (solve_em.F:2210-2212).  This
    remains the single sanctioned source of Omega; nothing downstream may
    re-derive it.  Backed by the persistent scratch slots
    ``rk_ru``/``rk_rv``/``rk_ww``: the views stay valid until the next
    ``stage_fluxes`` call on the same state (the acoustic substeps do not
    touch them).
    """
    nz, ny, nx = state.p.shape
    mu = state.total_mu()                              # (ny, nx) t* mass
    mux = mu_at_u_faces(mu)
    muy = mu_at_v_faces(mu)
    # Open boundaries: the boundary-face column mass is the boundary CELL's
    # (WRF's muu/muv under the zero-gradient mu ghost copy), not the
    # periodic wrap average.
    if _boundary_x(cfg):
        mux[:, 0] = mu[:, 0]
        mux[:, -1] = mu[:, -1]
    if _boundary_y(cfg):
        muy[0, :] = mu[0, :]
        muy[-1, :] = mu[-1, :]
    ru = state.scratch((nz, ny, nx + 1), "rk_ru")
    rv = state.scratch((nz, ny + 1, nx), "rk_rv")
    kernel = _couple_momentum_kernel(state.has_msf)    # WRF couple_momentum:
    for wind, muface, msf, flux in ((state.u, mux, state.msfu, ru),
                                    (state.v, muy, state.msfv, rv)):
        args = [wind, state.c1h, state.c2h, muface.reshape(-1)]
        if state.has_msf:                              # U = C(mu)*u/msfu,
            args.append(msf.reshape(-1))               # V = C(mu)*v/msfv
        kernel(*args, np.int32(muface.size), flux)
    return ru, rv, _omega_ref(state, cfg, ru, rv)


def domain_mass_measure(state: DomainState) -> float:
    """FP64 area-weighted domain dry-mass measure ``sum(mu/msft**2)``.

    The ONE measure both the flat and the mapped boundary-tendency
    branches close against, so a residual cannot be read in one unit and
    scored against the other.  With identity map factors the weight is
    exactly 1.0 and this is the plain column-mass sum.
    """
    return float(cp.sum(state.total_mu().astype(cp.float64)
                        * state.cell_area_weight(), dtype=cp.float64))


def _boundary_mass_tendency_flat(state: DomainState,
                                 cfg: RunConfig) -> cp.ndarray:
    """Unmapped telescoped boundary tendency as a 0-d FP64 device scalar."""
    boundary_x = _boundary_x(cfg)
    boundary_y = _boundary_y(cfg)
    tendency = cp.float64(0.0)
    if not boundary_x and not boundary_y:
        return tendency
    mu = state.total_mu()
    dnw = state.dnw[:, None]
    c1h = state.c1h[:, None]
    c2h = state.c2h[:, None]
    if boundary_x:
        west = (state.u_pp[:, :, 0]
                + (c1h * mu[:, 0][None] + c2h) * state.u[:, :, 0])
        east = (state.u_pp[:, :, -1]
                + (c1h * mu[:, -1][None] + c2h) * state.u[:, :, -1])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dx) * (east - west)).astype(cp.float64),
            dtype=cp.float64)
    if boundary_y:
        south = (state.v_pp[:, 0, :]
                 + (c1h * mu[0, :][None] + c2h) * state.v[:, 0, :])
        north = (state.v_pp[:, -1, :]
                 + (c1h * mu[-1, :][None] + c2h) * state.v[:, -1, :])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dy) * (north - south)).astype(cp.float64),
            dtype=cp.float64)
    return tendency


def _boundary_mass_tendency_mapped(state: DomainState,
                                   cfg: RunConfig) -> cp.ndarray:
    """Mapped telescoped boundary tendency as a 0-d FP64 device scalar.

    Derived from the in-tree mapped acoustic kernel rather than from a
    textbook form.  ``advance_mu_th_msf`` advances the column mass by

        d(mu_c)/dt = m_c**2 * sum_k dnw_k * (rdx*dFx + rdy*dFy) + rmu_c

    with ``m_c**2 = msft*msft`` and the total face flux
    ``F = u'' + (c1h*mu_face + c2h)*u/msfu`` (the reference momentum
    already carries its FACE map factor; ``u''`` carries its own from
    ``small_step_prep``).  Dividing the cell equation by ``m_c**2`` -- the
    weight :meth:`DomainState.cell_area_weight` applies -- removes the map
    factor from the flux term entirely, so the divergence telescopes over
    the domain and only the outermost faces survive.  The ``rmu_t``
    forcing and the specified-zone reset are deliberately NOT folded in
    here: they are separate budget terms, and a flux-only "closure" on a
    forced domain would be a false receipt.

    The mapped face factor is the only arithmetic difference from the flat
    branch, which is why the reduction control on an identity-map state is
    exact rather than approximate.
    """
    boundary_x = _boundary_x(cfg)
    boundary_y = _boundary_y(cfg)
    tendency = cp.float64(0.0)
    if not boundary_x and not boundary_y:
        return tendency
    mu = state.total_mu()
    dnw = state.dnw[:, None]
    c1h = state.c1h[:, None]
    c2h = state.c2h[:, None]
    if boundary_x:
        west = (state.u_pp[:, :, 0]
                + (c1h * mu[:, 0][None] + c2h) * state.u[:, :, 0]
                / state.msfu[:, 0][None])
        east = (state.u_pp[:, :, -1]
                + (c1h * mu[:, -1][None] + c2h) * state.u[:, :, -1]
                / state.msfu[:, -1][None])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dx) * (east - west)).astype(cp.float64),
            dtype=cp.float64)
    if boundary_y:
        south = (state.v_pp[:, 0, :]
                 + (c1h * mu[0, :][None] + c2h) * state.v[:, 0, :]
                 / state.msfv[0, :][None])
        north = (state.v_pp[:, -1, :]
                 + (c1h * mu[-1, :][None] + c2h) * state.v[:, -1, :]
                 / state.msfv[-1, :][None])
        tendency += cp.sum(
            (dnw * DTYPE(1.0 / cfg.dy) * (north - south)).astype(cp.float64),
            dtype=cp.float64)
    return tendency


def boundary_mass_tendency_device(state: DomainState,
                                  cfg: RunConfig) -> cp.ndarray:
    """0-d FP64 device scalar form of :func:`boundary_mass_tendency`.

    Keeping the device scalar unread is what lets the accumulator observer
    run without a per-substep host synchronization.
    """
    if state.has_msf:
        return _boundary_mass_tendency_mapped(state, cfg)
    return _boundary_mass_tendency_flat(state, cfg)


def boundary_mass_tendency(state: DomainState, cfg: RunConfig) -> float:
    """FP64 domain-sum dry-mass tendency through the lateral boundary.

    This is the telescoped boundary form of WRF ``advance_mu_t`` evaluated
    after ``advance_uv`` has updated the acoustic perturbation momenta.  The
    total face flux is ``u_pp + (c1h*mu_face+c2h)*u/msfu`` (and analogously
    for v); multiplying its opposing-face difference by ``dnw/dx`` or
    ``dnw/dy`` gives exactly the domain sum of the column-mass equation for
    the measure :func:`domain_mass_measure`.  Internal ``rmu_t`` sources
    are intentionally excluded so a closure residual detects them.  Mapped
    domains take the ARW cell-area weighting
    (:meth:`DomainState.cell_area_weight`); the flat, map-factor-one
    branch keeps the WK82 arithmetic unchanged.

    Reading the result synchronizes the device; call
    :func:`boundary_mass_tendency_device` from a hot loop instead.
    """
    return float(boundary_mass_tendency_device(state, cfg))


class MassFluxAccumulator:
    """Device-resident FP64 running sum of the substep boundary increments.

    The list-appending ``mass_flux_observer`` reads one device scalar per
    acoustic substep, which is a host synchronization inside the
    innermost loop.  This accumulator adds the same FP64 increments in the
    same order on the device and is read once, at the end of the run, so a
    receipt-enabled real-case integration pays no per-substep sync.  The
    two are mutually exclusive keywords on :func:`step` precisely because
    running both would double-count nothing but would reintroduce the sync
    the accumulator exists to remove.
    """

    __slots__ = ("_total", "count")

    def __init__(self) -> None:
        self._total = cp.zeros((), dtype=cp.float64)
        #: number of accumulated substep increments
        self.count = 0

    def add(self, increment) -> None:
        """Accumulate one FP64 increment without reading it."""
        self._total += increment
        self.count += 1

    def total(self) -> float:
        """Host FP64 total; the one synchronization this observer takes."""
        return float(self._total)

    def reset(self) -> None:
        self._total = cp.zeros((), dtype=cp.float64)
        self.count = 0


def _launch_slow_pgf(state: DomainState, cfg: RunConfig, *, cq=None) -> None:
    """Subtract WRF's moist-cq-scaled large-step horizontal PGF."""
    nz, ny, nx = state.p.shape
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    cqu, cqv, _cqw, use_cq = cq
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    kernel = get_kernel("dycore", "slow_pgf")
    n = nz * (ny + 1) * (nx + 1)
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.ru_t, state.rv_t, state.p, state.pb, state.al, state.alt,
            state.php, state.phb, state.mup, state.mub2d,
            state.c1h, state.c2h, state.rdnw, state.fnm, state.fnp,
            state.cf1, state.cf2, state.cf3,
            np.int32(cfg.top_lid), state.cfn, state.cfn1,
            cqu, cqv, np.int32(use_cq),
            DTYPE(rdx), DTYPE(rdy), DTYPE(0.5 * rdx), DTYPE(0.5 * rdy),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
            np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _launch_slow_buoyancy(state: DomainState, cfg: RunConfig) -> None:
    """Add the fused vertical pressure-gradient and moist buoyancy term."""
    nz, ny, nx = state.p.shape
    dummy = state.p
    if state.qv is None:
        moist_mode = 0
        qv = qc = qr = qi = qs = qg = qh = dummy
    else:
        if getattr(state, "qh", None) is not None:
            moist_mode = 3
        else:
            moist_mode = 2 if getattr(state, "qi", None) is not None else 1
        qv, qc, qr = state.qv, state.qc, state.qr
        if moist_mode >= 2:
            qi = state.qi
            # P3 (mp=50) is the one scheme with qi and NO qs/qg, and
            # q_total's modes are 0/1/2/3 with no "one ice mass" arm.  The
            # absent pair takes the shared zero plane rather than reopening
            # a frozen kernel; moist.absent_mass_plane argues the identity.
            if getattr(state, "qs", None) is None:
                from gpuwm.core.moist import absent_mass_plane
                qs = qg = absent_mass_plane(state)
            else:
                qs, qg = state.qs, state.qg
        else:
            qi = qs = qg = dummy
        qh = state.qh if moist_mode == 3 else dummy
    kernel = get_kernel("dycore", "slow_buoyancy")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.rw_t, state.p, state.pb, state.mup, state.mub2d,
            qv, qc, qr, qi, qs, qg, qh, state.rdn, state.rdnw,
            state.c1f, state.c2f,
            state.msft, np.int32(moist_mode), np.int32(state.has_msf),
            np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _validate_geopotential_config(cfg: RunConfig, nx: int, ny: int) -> None:
    """Validate the horizontal geopotential-advection stencil."""
    if cfg.h_sca_adv_order not in (2, 5):
        raise ValueError(
            f"h_sca_adv_order must be 2 or 5, got {cfg.h_sca_adv_order}")
    if cfg.h_sca_adv_order == 5:
        if cfg.open_x or cfg.open_y:
            raise NotImplementedError(
                "h_sca_adv_order=5 with radiative open boundaries is not "
                "wired (periodic and specified only)")
        if nx < 7 or ny < 7:
            raise ValueError(
                f"h_sca_adv_order=5 needs nx, ny >= 7 (7-point stencil), "
                f"got {nx} x {ny}")


def _launch_slow_geopotential(state: DomainState, cfg: RunConfig,
                              ww: cp.ndarray, *, add_vertical: bool) -> None:
    """Apply fused vertical/g*w and horizontal geopotential RHS terms."""
    nz, ny, nx = state.p.shape
    _validate_geopotential_config(cfg, nx, ny)
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    kernel = get_kernel("dycore", "slow_geopotential")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.rph_t, ww, state.w, state.u, state.v, state.php, state.phb,
            state.mup, state.mub2d, state.rdnw, state.fnm, state.fnp,
            state.c1f, state.c2f, state.cfn, state.cfn1,
            state.msft, state.msfu, state.msfv,
            DTYPE(0.25 * rdx), DTYPE(0.25 * rdy),
            np.int32(state.has_msf), np.int32(_boundary_x(cfg)),
            np.int32(_boundary_y(cfg)), np.int32(_boundary_forced(cfg)),
            np.int32(cfg.h_sca_adv_order), np.int32(add_vertical),
            np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _launch_slow_geopotential_faces(state: DomainState, cfg: RunConfig,
                                    mux: cp.ndarray,
                                    muy: cp.ndarray) -> None:
    """Apply horizontal geopotential advection with supplied face masses."""
    nz, ny, nx = state.p.shape
    _validate_geopotential_config(cfg, nx, ny)
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    kernel = get_kernel("dycore", "slow_geopotential_faces")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
            (state.rph_t, state.u, state.v, state.php, state.phb,
            mux, muy, state.c1f, state.c2f,
            state.cfn, state.cfn1,
            state.msft, state.msfu, state.msfv,
            DTYPE(0.25 * rdx), DTYPE(0.25 * rdy),
            np.int32(state.has_msf), np.int32(_boundary_x(cfg)),
            np.int32(_boundary_y(cfg)), np.int32(_boundary_forced(cfg)),
            np.int32(cfg.h_sca_adv_order), np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _launch_slow_geopotential_vertical(state: DomainState,
                                       ww: cp.ndarray) -> None:
    """Apply only the vertical Omega and g*w terms on a failing config."""
    nz, ny, nx = state.p.shape
    kernel = get_kernel("dycore", "slow_geopotential_vertical")
    n = nz * ny * nx
    blocks = (n + 255) // 256
    kernel((blocks,), (256,),
           (state.rph_t, ww, state.w, state.php, state.phb,
            state.mup, state.mub2d, state.rdnw, state.fnm, state.fnp,
            state.c1f, state.c2f, state.msft,
            np.int32(state.has_msf), np.int32(state.phb.ndim == 3),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def _add_slow_tendencies(state: DomainState, cfg: RunConfig,
                         ru: cp.ndarray, rv: cp.ndarray,
                         ww: cp.ndarray, *, cq=None) -> None:
    """Accumulate the RK stage forcings R^t* into the coupled tendencies.

    General hybrid/terrain reduction of WRF ``rk_tendency``: flux-form
    advection of u/v/w/theta with the stage transport fluxes from
    :func:`stage_fluxes` (diagnosed Omega as vertical flux, c1/c2-weighted
    coupled momenta), plus the t*-state pressure-gradient/buoyancy terms
    that force the acoustic system — including the alpha'*d(pb)/dx term,
    which survives over terrain where the base pressure varies on eta
    surfaces.  Moist states use WRF's full ``pg_buoy_w`` form for the w
    buoyancy (vapor + hydrometeor loading).  rmu_t stays zero — the
    mass-divergence part of R_mu is computed inside the acoustic
    ``advance_mu_th`` kernel.

    Map factors + Coriolis (Task 3): the working tendencies follow WRF's
    conventions exactly — ru_t/rv_t/rw_t force the msf-coupled momenta
    U = C(mu)u/msfu, V = C(mu)v/msfv, W = C_f(mu)w/msft and rth_t/rph_t
    carry an extra 1/msft (their updates multiply it back) — and the
    Coriolis+curvature kernel joins the slot when rotation is enabled.
    With the default map factors every msf branch is skipped and the step
    is bitwise Phase 2 (regression-pinned).
    """
    launch_flux_div_scalar(state.total_theta(), ru, rv, ww, state.rth_t,
                           state, cfg.dx, cfg.dy,
                           open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                           msf=state.msft, has_msf=state.has_msf,
                           spec=_boundary_forced(cfg))
    launch_flux_div_u(state.u, ru, rv, ww, state.ru_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msfu, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg))
    launch_flux_div_v(state.v, ru, rv, ww, state.rv_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msfv, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg))
    launch_flux_div_w(state.w, ru, rv, ww, state.rw_t, state, cfg.dx, cfg.dy,
                      open_x=_boundary_x(cfg), open_y=_boundary_y(cfg),
                      msf=state.msft, has_msf=state.has_msf,
                      spec=_boundary_forced(cfg))

    # Fused WRF horizontal_pressure_gradient.  dycore.cu retains every
    # former eager-CuPy FP32 operator boundary explicitly.
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    _launch_slow_pgf(state, cfg, cq=cq)

    # Fused WRF pg_buoy_w dry/moist vertical forcing.
    _launch_slow_buoyancy(state, cfg)

    # Preserve rhs_ph's historical failure sequencing: it applied vertical
    # Omega and g*w before rejecting an invalid horizontal stencil.  Valid
    # configurations stay on the single combined hot-path launch.
    try:
        _validate_geopotential_config(cfg, state.p.shape[2], state.p.shape[1])
    except (ValueError, NotImplementedError):
        _launch_slow_geopotential_vertical(state, ww)
        raise

    # Fused WRF rhs_ph vertical, g*w, and horizontal-advection terms.
    _launch_slow_geopotential(state, cfg, ww, add_vertical=True)

    # --- Coriolis + curvature (Task 3; WRF rk_tendency's coriolis and
    # curvature calls, kernels/coriolis_map.cu): no-op unless rotation is
    # enabled (set_map_coriolis with nonzero f/e or non-uniform msf).
    if state.rotational:
        add_coriolis_curvature(state, cfg, ru, rv)


def capture_advective_theta_forcing(state: DomainState) -> None:
    """EXPORT the stage's pure advective theta rate as WRF ``RTHFTEN``.

    The exact inverse of :func:`add_h_diabatic_tendency`'s coupling:
    ``rth_t`` forces the WRF-coupled theta and carries an extra ``1/msfty``
    (see :func:`_add_slow_tendencies`), so the uncoupled K s-1 rate a
    cumulus scheme wants is ``rth_t * msfty / (c1h*mut + c2h)`` with the
    stage's own total dry mass ``mut``.  Same units, same one-step lag and
    the same producer/consumer split as ``h_diabatic``.

    CALL SITE IS THE CONTRACT.  This must run in the window after
    :func:`_add_slow_tendencies` returns -- where ``rth_t`` holds the flux
    divergence of theta and nothing else -- and before
    ``physics_tendencies.add_to_slow``, :func:`add_h_diabatic_tendency`,
    :func:`add_diffusion_tendencies` and the lateral-boundary fold touch
    it.  WRF's ``module_cumulus_driver.F:867`` pre-folds
    ``RTHRATEN + RTHBLTEN`` into ``RTHFTEN`` for G3SCHEME and
    NTIEDTKESCHEME and NOT for GFSCHEME, which sums the lanes itself
    (``gpuwm/core/kernels/gf.cu:4146``), so an export taken one line later
    makes the scheme integrate the boundary layer and the radiation twice.
    ``tests/test_dycore_advective_forcing_export.py`` is the gate.

    WHAT THE NUMBER IS, precisely, because "advective tendency" is
    ambiguous and the difference is measurable: this is the flux-form
    TRANSPORT tendency of the coupled scalar divided by the stage dry
    mass, which is the quantity ``rth_t`` carries and the quantity the
    h_diabatic coupling inverts.  It differs from the material derivative
    ``-v.grad(theta)`` by the mass-divergence term
    ``theta*(dmu/dt)/mu`` -- order 1e-3 K s-1 against a measured 3e-3
    K s-1 rms on a 12 km CONUS domain, so it is a stated part of the
    export rather than a rounding detail.  Both halves of the pair use
    the same construction and the same reference mass, so the theta and
    qv rates GF sums are consistent with each other.

    A no-op on a state with no advective-forcing consumer, where the
    buffers are ``None``.
    """
    if getattr(state, "rthften", None) is None:
        return
    rate = state.rth_t / (state.c1h[:, None, None] * state.total_mu()[None]
                          + state.c2h[:, None, None])
    if state.has_msf:                                  # WRF: the /msfty
        rate *= state.msft[None]                       # rth_t carries
    state.rthften[...] = rate


def add_h_diabatic_tendency(state: DomainState) -> None:
    """ADD the retained microphysics heating to the coupled theta tendency.

    WRF ``rk_addtend_dry`` (module_em.F:1076-1080): ``t_tend = t_tend +
    ... + (c1(k)*mut + c2(k))*h_diabatic(i,k,j)/msfty``, executed on EVERY
    RK step with the step's total dry mass ``mut = mub + mu_2``
    (rk_step_prep's ``CALL calculate_full``, module_em.F:143 /
    solve_em.F:652-666 — gpuwm's ``total_mu()`` at the stage start).
    ``h_diabatic`` (K/s, uncoupled)
    is the PREVIOUS step's microphysics theta increment per second
    (microphysics.moist_physics_finish); the acoustic substeps integrate
    this tendency (advance_mu_t, module_small_step_em.F:1142), and
    ``_finish_small_steps`` removes the final stage's accumulated
    contribution so the state is heated exactly once, by the direct
    microphysics update.  Float64 mirror:
    ``gpuwm.verify.npref.np_h_diabatic_tendency``.
    """
    hd = (state.c1h[:, None, None] * state.total_mu()[None]
          + state.c2h[:, None, None]) * state.h_diabatic
    if state.has_msf:                                  # WRF: /msfty
        hd /= state.msft[None]
    state.rth_t += hd


def add_rhs_ph_hadv(state: DomainState, cfg: RunConfig,
                    mux: cp.ndarray, muy: cp.ndarray) -> None:
    """SUBTRACT the rhs_ph horizontal phi advection from ``state.rph_t``.

    WRF ``rhs_ph`` advects the FULL geopotential (ph + phb) in advective
    form with the ``h_sca_adv_order`` stencil
    (module_big_step_utilities_em.F:1435 ``advective_order =
    config_flags%h_sca_adv_order``); the advecting face momenta are the
    (c1f*muuf+c2f)-coupled two-half-level sums ``(u(k)+u(k-1))*msfux``
    and the whole term carries 1/msfty at the mass point.  ``mux``/``muy``
    are the t* face masses (WRF muuf/muvf).  Interior full levels 1..nz-1
    only — the k = kte row belongs to the documented rigid-lid deviation.

    ``cfg.h_sca_adv_order == 2`` keeps the frozen Phase 1/2 two-face form
    (WRF's <=2 branch, :1516-1584) verbatim — bitwise-pinned by the flat
    regressions — with the open/specified zero-gradient boundary-normal
    faces documented below.

    ``cfg.h_sca_adv_order == 5`` is the reference configuration (Registry
    default, unset in the reference namelist): WRF's <=6 branch
    1/60-weighted 7-point centered stencil on ph and phb
    (:1786-1795 y / :1949-1959 x) applied everywhere when periodic, and
    with WRF's specified-BC narrowing otherwise —

      y rows (mass, 0-based):  0, ny-1 nothing; 1, ny-2 2nd order
        (:1882-1906); 2, ny-3 4th order 1/12 (:1819-1850, gated
        ``open_ys .or. specified``); [3, ny-4] 5th/6th interior
        (:1780-1781 narrowing);
      x cols:  0, nx-1 nothing; 1, nx-2 2nd order (:2023-2048);
        2, nx-3 NOTHING — WRF's 4th-order x pickups are gated on
        ``open_xs``/``open_xe`` ONLY (:1973, :1997), so a specified run
        skips those columns entirely (binding v4.6.1 quirk, transcribed
        as-is); [3, nx-4] 5th/6th interior (:1937-1938).

    Radiative open boundaries with order 5 are not wired (they need WRF's
    boundary-row ph_old upwind terms, :2085-2178); config validation and
    this function both refuse the combination.

    Float64 mirror: ``gpuwm.verify.npref.np_rhs_ph_hadv``.
    """
    _launch_slow_geopotential_faces(state, cfg, mux, muy)


def launch_coriolis_curvature(ru, rv, u, v, w, mut, msft, msfu, msfv, f, e,
                              c1f, c2f, fnm, fnp, dx, dy,
                              ru_t, rv_t, rw_t, *, sina=None, cosa=None,
                              boundary_x=False, boundary_y=False) -> None:
    """ADD the WRF Coriolis + curvature tendencies (coriolis_map.cu).

    Transcribed from WRF v4.6.1 ``module_big_step_utilities_em.F``
    ``coriolis``/``curvature`` (WRF open/specified exclusions; isotropic
    map factors; the full sina/cosa rotation terms — see the kernel header
    for the documented reductions and the WRF line cites).  ``ru``/``rv``
    are the stage's msf-coupled momenta from
    :func:`stage_fluxes`; ``u``/``v``/``w`` the uncoupled stage winds;
    ``mut (ny, nx)`` the total dry mass (the kernel forms the coupled
    ``rw = (c1f*mut + c2f)*w/msft`` inline); ``f``/``e`` the Coriolis
    parameters at mass points; ``sina``/``cosa`` the local map-rotation
    angle at mass points (geo_em SINALPHA/COSALPHA; ``None`` selects WRF's
    unrotated identity sina = 0 / cosa = 1); ``fnm``/``fnp`` the
    half->full weights (WRF passes them as fzm/fzp).  Mirror:
    ``gpuwm.verify.npref.np_coriolis_curvature``.
    """
    nz, ny, nxp1 = ru.shape
    nx = nxp1 - 1
    if sina is None:
        sina = cp.zeros((ny, nx), dtype=DTYPE)
    if cosa is None:
        cosa = cp.ones((ny, nx), dtype=DTYPE)
    kern = get_kernel("coriolis_map", "coriolis_curvature")
    n = nz * (ny + 1) * (nx + 1)
    blocks = (n + 255) // 256
    kern((blocks,), (256,),
         (ru, rv, u, v, w, mut, msft, msfu, msfv, f, e, sina, cosa,
          c1f, c2f, fnm, fnp,
          DTYPE(1.0 / dx), DTYPE(1.0 / dy),
          ru_t, rv_t, rw_t, np.int32(boundary_x), np.int32(boundary_y),
          np.int32(nz), np.int32(ny), np.int32(nx)))


def add_coriolis_curvature(state: DomainState, cfg: RunConfig,
                           ru: cp.ndarray, rv: cp.ndarray) -> None:
    """State-level wrapper: add Coriolis + curvature to the slow tendencies
    (the WRF ``rk_tendency`` coriolis/curvature slot)."""
    launch_coriolis_curvature(
        ru, rv, state.u, state.v, state.w, state.total_mu(),
        state.msft, state.msfu, state.msfv, state.f, state.e,
        state.c1f, state.c2f, state.fnm, state.fnp, cfg.dx, cfg.dy,
        state.ru_t, state.rv_t, state.rw_t,
        sina=state.sina, cosa=state.cosa,
        boundary_x=(cfg.open_x or _boundary_forced(cfg)),
        boundary_y=(cfg.open_y or _boundary_forced(cfg)))


def launch_smag2d_km(u, v, xkmh, xkhh, dx, dy, c_s) -> None:
    """Fill the km_opt=4 eddy viscosities at mass points (smag2d.cu).

    ``u (nz,ny,nx+1)`` / ``v (nz,ny+1,nx)`` device winds; ``xkmh``/``xkhh``
    ``(nz,ny,nx)`` outputs -- momentum K and scalar K = K_m/prandtl.
    Mirror: ``gpuwm.verify.npref.np_smag2d_km``.
    """
    nz, ny, nx = xkmh.shape
    kern = get_kernel("smag2d", "smag2d_km")
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    kern(grid, (_TPB, 1, 1),
         (u, v, xkmh, xkhh, DTYPE(1.0 / dx), DTYPE(1.0 / dy),
          DTYPE(math.sqrt(dx * dy)), DTYPE(c_s), DTYPE(_PRANDTL),
          np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_smag2d_hd(f, xk, mut, c1, c2, dx, dy, tend, stagger="",
                     open_x=False, open_y=False) -> None:
    """ADD the variable-K coupled mixing tendency of one field into ``tend``
    (WRF ``horizontal_diffusion``, coordinate surfaces, smag2d.cu).

    ``stagger`` as in ``launch_add_diff2``: ``""`` mass points (WRF 'm'),
    ``"x"`` u, ``"y"`` v, ``"z"`` w (c1/c2 must then be c1f/c2f; boundary
    levels get no tendency).  ``xk (nz,ny,nx)`` is the mass-point eddy
    viscosity, ``mut (ny,nx)`` the total dry mass.  ``open_x``/``open_y``
    (consumed by the ``"x"``/``"y"`` kernels only) switch the
    boundary-normal face nx-1 / ny-1 to WRF's honest boundary-datum read
    (field(ide) / field(jde), the stored last column/row) instead of the
    periodic wrap; the caller still zeroes WRF's excluded width-1 strip
    afterwards (``_zero_open_strips``).  Mirror:
    ``gpuwm.verify.npref.np_smag2d_hd``.
    """
    nz, ny, nx = xk.shape
    kern = get_kernel("smag2d", _SMAG_HD[stagger])
    nlev, nys, nxs = f.shape
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    args = [f, xk, mut, c1, c2, DTYPE(1.0 / dx), DTYPE(1.0 / dy), tend,
            np.int32(nz), np.int32(ny), np.int32(nx)]
    if stagger == "x":
        args.append(np.int32(open_x))
    elif stagger == "y":
        args.append(np.int32(open_y))
    kern(grid, (_TPB, 1, 1), tuple(args))


def _wrf_smag_grid_args(state: DomainState, cfg: RunConfig, *,
                        time_t: bool) -> list:
    """Common device arguments for WRF's metric-aware diff_opt=2 kernels."""
    suffix = "0" if time_t else ""
    moist = state.qv is not None
    qv = getattr(state, "qv" + suffix) if moist else state.alt
    return [getattr(state, "u" + suffix), getattr(state, "v" + suffix),
            getattr(state, "w" + suffix), getattr(state, "php" + suffix),
            state.phb, state.alt, qv, state.msft, state.msfu, state.msfv,
            state.fnm, state.fnp, state.dn, state.dnw,
            DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy),
            DTYPE(cfg.dx), DTYPE(cfg.dy),
            DTYPE(state.cf1), DTYPE(state.cf2), DTYPE(state.cf3),
            np.int32(moist)]


def launch_wrf_smag2d_km(state: DomainState, cfg: RunConfig,
                          xkmh, xkhh, *, time_t: bool):
    """WRF v4.6.1 ``cal_deform_and_div`` + ``smag2d_km`` on device.

    Unlike :func:`launch_smag2d_km`'s retained flat-oracle API, this path
    carries total geopotential, map factors, vertical metrics and w.  The
    kernels derive ``rdz/rdzw/zx/zy`` on demand, stage the required
    deformation tensors in dead carrying-buffer prefixes, apply WRF's local
    mixing length and slope limiter, and then preserve WRF's cold-zeroed
    logical outer coefficient row at physical boundaries.  ``phy_bc`` only
    extends that active row into outside ghost cells; it does not copy an
    interior K value into the active boundary.
    """
    nz, ny, nx = xkmh.shape
    n_mass = nz * ny * nx
    # WRF deformation tensors are live only through the u/v stress launches.
    # Borrow the prefixes of the not-yet-built carrying buffers, then replace
    # them with their final tendencies below.  This avoids three additional
    # d04-sized allocations without aliasing any live result.
    d11 = state.scratch((nz + 1, ny, nx), "smag_rw").reshape(-1)[:n_mass]
    d22 = state.scratch((nz, ny, nx + 1), "smag_ru").reshape(-1)[:n_mass]
    d12 = state.scratch((nz, ny + 1, nx), "smag_rv").reshape(-1)[:n_mass]
    d11 = d11.reshape((nz, ny, nx))
    d22 = d22.reshape((nz, ny, nx))
    d12 = d12.reshape((nz, ny, nx))
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (_TPB, 1, 1), tuple(common + [d11, d22, d12] + dims))
    tail = [DTYPE(cfg.c_s), DTYPE(_PRANDTL), d11, d22, d12, xkmh, xkhh,
            np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    get_kernel("smag2d", "wrf_smag2d_km")(
        grid, (_TPB, 1, 1), tuple(common + tail))
    if _boundary_x(cfg) or _boundary_y(cfg):
        get_kernel("smag2d", "wrf_smag_km_bc")(
            grid, (_TPB, 1, 1),
            (xkmh, xkhh, np.int32(nz), np.int32(ny), np.int32(nx),
             np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))))
    return d11, d22, d12


def launch_wrf_calc_n2(state: DomainState, cfg: RunConfig, bn2, *,
                       time_t: bool) -> None:
    """WRF v4.6.1 ``calculate_N2`` on device (module_diffusion_em.F:
    1485-1713): moist Brunt-Vaisala frequency at mass points into ``bn2``,
    with the saturated moist-adiabatic branch (qv >= qvs or qc >= 1e-5),
    the unsaturated qv/qtot form, the MARTA/WCS one-sided surface level,
    and the ktf copy.  Mirror: ``gpuwm.verify.npref.np_wrf_calc_n2``.

    The kernel is fetched through ``gpuwm.core.moist_n2_mutation`` rather
    than straight from ``get_kernel``, so the LES spec 3.3 mutation control
    -- the saturated branch forced off, for instrument qualification -- can
    select a separately compiled variant under its own cache key.  It is
    off by default and the default path resolves to the same cached
    production kernel as before.
    """
    nz, ny, nx = bn2.shape
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    suffix = "0" if time_t else ""
    thp = getattr(state, "thp" + suffix)
    moist = state.qv is not None
    qc = getattr(state, "qc" + suffix) if moist else None
    qi = (getattr(state, "qi" + suffix)
          if moist and getattr(state, "qi", None) is not None else None)
    dummy = state.alt
    calc_n2_kernel()(
        grid, (_TPB, 1, 1),
        tuple(common + [
            thp, state.thb, np.int32(state.thb.ndim == 3), state.p,
            qc if qc is not None else dummy,
            qi if qi is not None else dummy,
            np.int32(qc is not None), np.int32(qi is not None),
            bn2,
        ] + dims))


def _tke_seed(cfg: RunConfig) -> float:
    """WRF tke_km's seed rule (module_diffusion_em.F:2162-2174): without
    surface drag or a surface heat flux there is no way to generate TKE
    from nothing, so isfflx=0 with both prescribed constants off seeds at
    1e-6; any other isfflx leaves the seed at zero."""
    if cfg.isfflx != 0:
        return 0.0
    if cfg.bl_pbl_physics == 0:          # the diff_opt=2 PBL-off branch
        if (cfg.tke_drag_coefficient < 1.0e-10
                and cfg.tke_heat_flux < 1.0e-10):
            return 1.0e-6
        return 0.0
    return 1.0e-6


def launch_wrf_tke_km(state: DomainState, cfg: RunConfig,
                      xkmh, xkhh, *, time_t: bool):
    """WRF v4.6.1 km_opt=2: ``cal_deform_and_div`` + ``calculate_N2`` +
    ``tke_km`` (module_diffusion_em.F:2049-2260) on device, then the
    ``tke_rhs`` forward source (shear + buoyancy + dissipation + the
    positivity limiter) into the ``smag_rtke`` carrying buffer.

    Fills the four exchange coefficients from the time-t prognostic TKE
    (``smag_km``/``smag_kh`` plus the vertical pair ``smag_kmv``/
    ``smag_khv``).  BN2 borrows the ``diff6_x`` prefix exactly as the
    km_opt=3 launcher and stays live through the tke_rhs launch (both
    precede the u/v horizontal staging).  Returns the (d11, d22, d12)
    deformation triple.
    """
    nz, ny, nx = xkmh.shape
    n_mass = nz * ny * nx
    d11 = state.scratch((nz + 1, ny, nx), "smag_rw").reshape(-1)[:n_mass]
    d22 = state.scratch((nz, ny, nx + 1), "smag_ru").reshape(-1)[:n_mass]
    d12 = state.scratch((nz, ny + 1, nx), "smag_rv").reshape(-1)[:n_mass]
    d11 = d11.reshape((nz, ny, nx))
    d22 = d22.reshape((nz, ny, nx))
    d12 = d12.reshape((nz, ny, nx))
    xkmv = state.scratch((nz, ny, nx), "smag_kmv")
    xkhv = state.scratch((nz, ny, nx), "smag_khv")
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (_TPB, 1, 1), tuple(common + [d11, d22, d12] + dims))

    bn2 = state.scratch((nz, ny, nx + 1), "diff6_x").reshape(-1)[:n_mass]
    bn2 = bn2.reshape((nz, ny, nx))
    launch_wrf_calc_n2(state, cfg, bn2, time_t=time_t)

    suffix = "0" if time_t else ""
    thp = getattr(state, "thp" + suffix)
    tke = getattr(state, "tke" + suffix)
    get_kernel("smag2d", "wrf_tke_km")(
        grid, (_TPB, 1, 1),
        tuple(common + [
            thp, state.thb, np.int32(state.thb.ndim == 3), state.p,
            tke, bn2,
            DTYPE(cfg.c_k), DTYPE(_PRANDTL), DTYPE(cfg.dt),
            DTYPE(cfg.mix_upper_bound), np.int32(cfg.mix_isotropic),
            DTYPE(_tke_seed(cfg)),
            xkmh, xkhh, xkmv, xkhv,
        ] + dims))
    if _boundary_x(cfg) or _boundary_y(cfg):
        for pair in ((xkmh, xkhh), (xkmv, xkhv)):
            get_kernel("smag2d", "wrf_smag_km_bc")(
                grid, (_TPB, 1, 1),
                (*pair, np.int32(nz), np.int32(ny), np.int32(nx),
                 np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))))

    # tke_rhs: the once-per-step forward TKE source, before the borrowed
    # deformation prefixes are replaced by the u/v stress staging.
    fields = (
        state.physics.fields
        if state.physics is not None and hasattr(state.physics, "fields")
        else {}
    )
    dummy = state.mup0
    ustm = fields.get("ustm", dummy)
    hfx = fields.get("hfx", dummy)
    rtke = state.scratch((nz, ny, nx), "smag_rtke")
    mut = state.scratch((ny, nx), "smag_mut")
    mut[...] = state.mub2d + (state.mup0 if time_t else state.mup)
    budget_on = tke_budget.enabled(cfg)
    budget_terms = [tke_budget.term(state, cfg, name)
                    for name in ("shear", "buoyancy", "dissipation",
                                 "limiter")] if budget_on else [rtke] * 4
    get_kernel("smag2d", "wrf_tke_rhs")(
        grid, (_TPB, 1, 1),
        tuple(common + [
            thp, state.thb, np.int32(state.thb.ndim == 3),
            tke, bn2, d11, d22, d12,
            xkmh, xkmv, xkhv,
            mut, state.c1h, state.c2h,
            ustm, hfx,
            np.int32("ustm" in fields), np.int32("hfx" in fields),
            DTYPE(cfg.c_k), DTYPE(cfg.dt),
            DTYPE(cfg.tke_drag_coefficient), DTYPE(cfg.tke_heat_flux),
            np.int32(cfg.isfflx),
            rtke,
        ] + budget_terms + [np.int32(budget_on)] + dims))
    return d11, d22, d12


def launch_wrf_smag3d_km(state: DomainState, cfg: RunConfig,
                         xkmh, xkhh, *, time_t: bool):
    """WRF v4.6.1 km_opt=3: ``cal_deform_and_div`` + ``calculate_N2`` +
    ``smag_km`` on device (module_diffusion_em.F:1485-1713, :1777-1929).

    Fills the four exchange-coefficient fields -- ``xkmh``/``xkhh`` in the
    shared ``smag_km``/``smag_kh`` slots plus the km_opt=3-only vertical
    pair in ``smag_kmv``/``smag_khv`` -- from the FULL deformation
    invariant (D13/D23/D33 included, off-diagonal tensors averaged to mass
    points before squaring), the ``sqrt(max(0, D^2 - N^2/Pr))`` buoyancy
    reduction, and WRF's two ``mix_isotropic`` mixing-length branches with
    their exact floors and ``mix_upper_bound/dt`` caps.  BN2 briefly
    borrows the ``diff6_x`` face workspace prefix (dead until the u/v
    horizontal staging that follows the K computation).  Returns the
    (d11, d22, d12) deformation triple exactly as the km_opt=4 launcher.
    """
    nz, ny, nx = xkmh.shape
    n_mass = nz * ny * nx
    d11 = state.scratch((nz + 1, ny, nx), "smag_rw").reshape(-1)[:n_mass]
    d22 = state.scratch((nz, ny, nx + 1), "smag_ru").reshape(-1)[:n_mass]
    d12 = state.scratch((nz, ny + 1, nx), "smag_rv").reshape(-1)[:n_mass]
    d11 = d11.reshape((nz, ny, nx))
    d22 = d22.reshape((nz, ny, nx))
    d12 = d12.reshape((nz, ny, nx))
    xkmv = state.scratch((nz, ny, nx), "smag_kmv")
    xkhv = state.scratch((nz, ny, nx), "smag_khv")
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    dims = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    get_kernel("smag2d", "wrf_smag_deform")(
        grid, (_TPB, 1, 1), tuple(common + [d11, d22, d12] + dims))

    # calculate_N2 inputs: time-t prognostic theta/moisture, current
    # diagnostic p (refreshed against the time-t state by step() before
    # prepare_fixed_tendencies, exactly as alt in the common args).
    bn2 = state.scratch((nz, ny, nx + 1), "diff6_x").reshape(-1)[:n_mass]
    bn2 = bn2.reshape((nz, ny, nx))
    launch_wrf_calc_n2(state, cfg, bn2, time_t=time_t)

    get_kernel("smag2d", "wrf_smag3d_km")(
        grid, (_TPB, 1, 1),
        tuple(common + [
            DTYPE(cfg.c_s), DTYPE(_PRANDTL), DTYPE(cfg.dt),
            DTYPE(cfg.mix_upper_bound), np.int32(cfg.mix_isotropic),
            d11, d22, d12, bn2, xkmh, xkhh, xkmv, xkhv,
        ] + dims))
    if _boundary_x(cfg) or _boundary_y(cfg):
        for pair in ((xkmh, xkhh), (xkmv, xkhv)):
            get_kernel("smag2d", "wrf_smag_km_bc")(
                grid, (_TPB, 1, 1),
                (*pair, np.int32(nz), np.int32(ny), np.int32(nx),
                 np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))))
    return d11, d22, d12


def launch_wrf_smag2d_hd(state: DomainState, cfg: RunConfig, f, xk, tend,
                          *, stagger: str, time_t: bool,
                          full_theta: bool = False,
                          deformation=None) -> None:
    """Add WRF's metric-aware tensor-stress/scalar-flux horizontal tendency.

    WRF passes ``grid%t_2 = theta - T0`` to the scalar operator.  For the
    theta row, ``full_theta`` reconstructs that field on demand from gpuwm's
    ``thp = theta - thb`` storage; moisture rows pass their mixing ratios
    unchanged.  Reconstruction inside the flux kernel avoids a full-grid
    temporary and handles both one-dimensional and terrain-following 3-D
    ``thb``.
    """
    if stagger not in _WRF_SMAG_HD:
        raise ValueError(f"unknown Smagorinsky stagger {stagger!r}")
    nz, ny, nx = xk.shape
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    nlev, nys, nxs = f.shape
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    if stagger == "":
        flux_x = state.scratch((nz, ny, nx + 1), "diff6_x")
        flux_y = state.scratch((nz, ny + 1, nx), "diff6_y")
        flux_grid = ((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz)
        get_kernel("smag2d", "wrf_smag_flux_s")(
            flux_grid, (_TPB, 1, 1),
            tuple(common + [f, xk, state.thb, np.int32(full_theta),
                            np.int32(state.thb.ndim == 3),
                            flux_x, flux_y] + tail))
        payload = [flux_x, flux_y, tend]
    elif stagger == "x":
        if deformation is None:
            raise ValueError("u Smagorinsky stress requires deformation")
        d11, _d22, d12 = deformation
        payload = [xk, d11, d12, tend]
    elif stagger == "y":
        if deformation is None:
            raise ValueError("v Smagorinsky stress requires deformation")
        _d11, d22, d12 = deformation
        payload = [xk, d22, d12, tend]
    else:
        payload = [xk, tend]
    get_kernel("smag2d", _WRF_SMAG_HD[stagger])(
        grid, (_TPB, 1, 1), tuple(common + payload + tail))


def launch_wrf_smag2d_vertical(
        state: DomainState, cfg: RunConfig, km, *,
        ru, rv, rw, rth, rqv, time_t: bool,
        kmv=None, khv=None, scalar_rows=None) -> None:
    """Add WRF v4.6.1 ``vertical_diffusion_2`` for the PBL-off diff_opt=2
    path (module_first_rk_step_part2.F:1011-1074).

    km_opt=4: ``smag2d_km`` defines ``xkmv=xkmh`` and ``xkhv=0``, so only
    u/v/w have interior vertical stresses (``kmv``/``khv`` omitted).
    km_opt=3 passes ``kmv`` (vertical momentum K, consumed by the tau13/
    tau23 operators -- WRF hands ``xkmv`` to vertical_diffusion_u_2/v_2
    but ``xkmh`` to vertical_diffusion_w_2, transcribed exactly) and
    ``khv`` plus ``scalar_rows`` -- ``(field, tendency, full_theta)``
    triples mixed by ``vertical_diffusion_s`` with the vertical scalar K.

    Surface forcing follows WRF's ``SELECT CASE(isfflx)`` matrix:
    isfflx=0 takes the prescribed ``tke_drag_coefficient`` wall stress and
    ``tke_heat_flux`` heat flux with no moisture flux; isfflx=1 takes
    USTM/HFX/QFX from the surface driver; isfflx=2 takes USTM/QFX from
    the driver but the constant ``tke_heat_flux`` heat.  A surface-layer-
    free run supplies no fields and the ust-based arms stay cold-zero,
    exactly the no-writer WRF state.
    """
    nz, ny, nx = km.shape
    common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
    tail = [np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(state.phb.ndim == 3),
            np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
    if kmv is None:
        kmv = km
    launches = (
        ("wrf_smag_vd_u", kmv, ru, (nx + 1, ny, nz)),
        ("wrf_smag_vd_v", kmv, rv, (nx, ny + 1, nz)),
        # DIVERGENCE, deliberate.  WRF hands vertical_diffusion_w_2 xkmh
        # (module_diffusion_em.F:4145-4155); gpuwm hands it xkmv.  See
        # this function's docstring for the derivation and the evidence.
        ("wrf_smag_vd_w", kmv, rw, (nx, ny, nz + 1)),
    )
    for name, xk, tendency, (nxs, nys, nlev) in launches:
        grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
        get_kernel("smag2d", name)(
            grid, (_TPB, 1, 1), tuple(common + [xk, tendency] + tail))

    mass_grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    if khv is not None and scalar_rows:
        for field, tendency, full_theta in scalar_rows:
            get_kernel("smag2d", "wrf_smag_vd_s")(
                mass_grid, (_TPB, 1, 1),
                tuple(common + [
                    field, state.thb, np.int32(bool(full_theta)),
                    np.int32(state.thb.ndim == 3), khv, tendency,
                ] + tail))

    fields = (
        state.physics.fields
        if state.physics is not None and hasattr(state.physics, "fields")
        else {}
    )
    active = int(
        cfg.sf_sfclay_physics != 0
        and all(name in fields for name in ("ustm", "hfx", "qfx")))
    isfflx = cfg.isfflx
    dummy = state.mup0
    ustm = fields.get("ustm", dummy)
    hfx = fields.get("hfx", dummy)
    qfx = fields.get("qfx", dummy)
    if isfflx == 0:
        # vflux CASE(0): constant drag coefficient, no surface scheme.
        cd0 = DTYPE(cfg.tke_drag_coefficient)
        for name, tendency, nxs, nys in (
                ("wrf_smag_surface_u_cd0", ru, nx + 1, ny),
                ("wrf_smag_surface_v_cd0", rv, nx, ny + 1)):
            grid = ((nxs + _TPB - 1) // _TPB, nys, 1)
            get_kernel("smag2d", name)(
                grid, (_TPB, 1, 1),
                tuple(common + [cd0, tendency] + tail))
    else:
        # vflux CASE(1,2): ustar from the surface routine (USTM).
        for name, tendency, nxs, nys in (
                ("wrf_smag_surface_u", ru, nx + 1, ny),
                ("wrf_smag_surface_v", rv, nx, ny + 1)):
            grid = ((nxs + _TPB - 1) // _TPB, nys, 1)
            get_kernel("smag2d", name)(
                grid, (_TPB, 1, 1),
                tuple(common + [ustm, np.int32(active), tendency] + tail))
    scalar_grid = ((nx + _TPB - 1) // _TPB, ny, 1)
    if isfflx in (0, 2):
        # hflux CASE(0,2): prescribed constant kinematic heat flux.
        get_kernel("smag2d", "wrf_smag_surface_heat_const")(
            scalar_grid, (_TPB, 1, 1),
            tuple(common + [DTYPE(cfg.tke_heat_flux), rth] + tail))
    apply_heat = int(isfflx == 1) and active
    apply_moist = int(isfflx in (1, 2)) and active
    get_kernel("smag2d", "wrf_smag_surface_scalars")(
        scalar_grid, (_TPB, 1, 1),
        tuple(common + [
            hfx, qfx, np.int32(apply_heat), np.int32(apply_moist), rth,
            rqv if rqv is not None else rth,
        ] + tail))


def _horizontal_w_km(state: DomainState, cfg: RunConfig):
    """WRF's ``xkmv`` where ``horizontal_diffusion_w_2`` needs it.

    ``None`` for km_opt=4 (and for the diff6-only path, which consumes no
    K at all): ``smag2d_km`` defines ``xkmv = xkmh``
    (module_diffusion_em.F:2035), so the caller's ``xkmh`` IS ``xkmv``
    there.  km_opt=2/3 fill a separate vertical pair in ``smag_kmv``
    (``tke_km`` :2049-2260 / ``smag_km`` :1890-1908), and on an
    anisotropic grid it is smaller than ``xkmh`` by (dz/dx)^2 -- the whole
    point of ``mix_isotropic = 0``.
    """
    if cfg.km_opt not in (2, 3):
        return None
    return state.scratch(state.p.shape, "smag_kmv")


def _smag2d_specs(state: DomainState, km, kh, *, time_t: bool = False,
                  kmv=None):
    """(field, tend-or-None, K, c1, c2, scratch slot, stagger) rows for the
    km_opt=4 package: momentum takes K_m, scalars (WRF ``theta - T0`` and
    moisture) take K_h = K_m/prandtl; moisture rows carry no state tendency array (their
    increment folds into the scalar update).  ``time_t`` binds the saved
    ``*0`` fields used by WRF's once-per-step forward tendencies.

    ``kmv`` is the w row's horizontal K.  WRF's ``horizontal_diffusion_2``
    hands ``xkmh`` to ``horizontal_diffusion_u_2``/``_v_2`` but ``xkmv`` to
    ``horizontal_diffusion_w_2`` (module_diffusion_em.F:2978-3006; the
    dummy argument is spelled ``xkmv`` at :3524), because that operator is
    the divergence of tau13/tau23 and those stresses take the VERTICAL
    momentum coefficient everywhere else too (``vertical_diffusion_u_2``/
    ``_v_2``, :4128-4147).  ``smag2d_km`` sets ``xkmv = xkmh`` for km_opt=4
    (:2035, "v4.2 and later, this is used for hor. diff. of w"), so the
    default ``None`` -- pass ``xkmh`` -- is that identity, not a shortcut.
    km_opt=2/3 compute a genuinely different vertical pair and must pass
    it."""
    def field(name):
        return getattr(state, name + "0" if time_t else name)
    specs = [(field("u"), state.ru_t, km, state.c1h, state.c2h,
              "smag_ru", "x"),
             (field("v"), state.rv_t, km, state.c1h, state.c2h,
              "smag_rv", "y"),
             (field("w"), state.rw_t, km if kmv is None else kmv,
              state.c1f, state.c2f, "smag_rw", "z"),
             (field("thp"), state.rth_t, kh, state.c1h, state.c2h,
              "smag_rth", "")]
    if state.qv is not None:
        specs += [(field(name), None, kh, state.c1h, state.c2h,
                   "smag_r" + name, "") for name in SPECIES]
        specs += [(field(name), None, kh, state.c1h, state.c2h,
                   "smag_r" + name, "")
                  for name in extra_moist_species(state)]
    return specs


def diff6_exempt_slots(cfg: RunConfig) -> frozenset[str]:
    """Carrying-buffer slots the 6th-order filter must skip this run.

    WRF's ``&dynamics`` filter switches are per Registry ARRAY, and
    ``rk_scalar_tend`` calls ``sixth_order_diffusion`` under
    ``(diff_6th_opt .NE. 0) .and. (.not. mix6_off)`` with the array's own
    switch (dyn_em/module_em.F:1421).  ``moist_mix6_off`` therefore removes
    the moist array's rows and NOTHING else: theta keeps its filter, the
    number/volume tracers keep theirs (they are WRF ``scalar``-package
    fields with their own ``scalar_mix6_off``), and TKE keeps
    ``tke_mix6_off``.

    Returned as slot names because the diff6 row set is addressed by
    carrying buffer, and a name filter cannot accidentally exempt a row
    whose field happens to share a shape with a moist one.
    """

    if not cfg.moist_mix6_off:
        return frozenset()
    return frozenset("smag_r" + name for name in WRF_MOIST_ARRAY_SPECIES)


def _compute_wrf_smag_tendencies(state: DomainState, cfg: RunConfig,
                                  km, kh, specs, *, time_t: bool) -> None:
    """Build the once-per-step WRF metric/stress forward tendencies.

    The deformation tensors temporarily occupy mass-sized prefixes of the
    u/v/w carrying buffers.  u and v therefore launch first into the reusable
    diff6 face workspaces; after both have consumed all three tensors, their
    results replace those borrowed buffers.  w and every scalar can then be
    produced normally.  The scalar operator uses the same face workspaces for
    its two explicit metric flux passes.
    """
    if cfg.km_opt == 2:
        deformation = launch_wrf_tke_km(
            state, cfg, km, kh, time_t=time_t)
    elif cfg.km_opt == 3:
        deformation = launch_wrf_smag3d_km(
            state, cfg, km, kh, time_t=time_t)
    else:
        deformation = launch_wrf_smag2d_km(
            state, cfg, km, kh, time_t=time_t)

    # u/v must both see the complete deformation set before either borrowed
    # carrying buffer is replaced by its final stress divergence.
    for row, tmp_slot in zip(specs[:2], ("diff6_x", "diff6_y")):
        f, _tend, xk, _c1, _c2, _slot, stag = row
        tmp = state.scratch(f.shape, tmp_slot)
        tmp[...] = 0
        launch_wrf_smag2d_hd(state, cfg, f, xk, tmp, stagger=stag,
                             time_t=time_t, deformation=deformation)
        _zero_open_strips(tmp, cfg, 1)
    for row, tmp_slot in zip(specs[:2], ("diff6_x", "diff6_y")):
        f, _tend, _xk, _c1, _c2, slot, _stag = row
        state.scratch(f.shape, slot)[...] = state.scratch(f.shape, tmp_slot)

    # D11 occupied smag_rw; it is dead after both horizontal-momentum calls.
    # Scalars use fresh H1/H2 fluxes in diff6_x/y for each field.
    for f, _tend, xk, _c1, _c2, slot, stag in specs[2:]:
        buf = state.scratch(f.shape, slot)
        buf[...] = 0
        launch_wrf_smag2d_hd(state, cfg, f, xk, buf, stagger=stag,
                             time_t=time_t,
                             full_theta=(slot == "smag_rth"))
        _zero_open_strips(buf, cfg, 1)

    if cfg.km_opt == 2:
        # TKE self-diffusion: horizontal with 2*Km_h regardless of the PBL
        # (module_diffusion_em.F:3020-3032, doubling :3988-3996 -- WRF's
        # doing_tke tendency = tmptendf + 2*(tendency - tmptendf)).
        nz_, ny_, nx_ = km.shape
        tke_t = getattr(state, "tke0" if time_t else "tke")
        rtke = state.scratch((nz_, ny_, nx_), "smag_rtke")
        tmp = state.scratch((nz_, ny_, nx_), "smag_tke_tmp")
        if not getattr(cfg, "tke_mix2_off", False):
            tmp[...] = 0
            launch_wrf_smag2d_hd(state, cfg, tke_t, km, tmp, stagger="",
                                 time_t=time_t, full_theta=False)
            _zero_open_strips(tmp, cfg, 1)
            rtke += 2.0 * tmp
            budget_h = tke_budget.term(state, cfg, "diffusion_h")
            if budget_h is not None:
                budget_h[...] = 2.0 * tmp

    if cfg.bl_pbl_physics == 0:
        buffers = {
            slot: state.scratch(f.shape, slot)
            for f, _tend, _xk, _c1, _c2, slot, _stag in specs
        }
        kmv = khv = None
        scalar_rows = None
        if cfg.km_opt in (2, 3):
            # The closure's vertical pair, filled by launch_wrf_tke_km /
            # launch_wrf_smag3d_km above.  Interior vertical scalar mixing
            # covers WRF's rt_tendf row (theta reconstructed from thp+thb)
            # and every moist species (vertical_diffusion_2's moist_loop).
            nz_, ny_, nx_ = km.shape
            kmv = state.scratch((nz_, ny_, nx_), "smag_kmv")
            khv = state.scratch((nz_, ny_, nx_), "smag_khv")
            scalar_rows = [
                (f, buffers[slot], slot == "smag_rth")
                for f, _tend, _xk, _c1, _c2, slot, stag in specs
                if stag == ""
            ]
        launch_wrf_smag2d_vertical(
            state, cfg, km,
            ru=buffers["smag_ru"],
            rv=buffers["smag_rv"],
            rw=buffers["smag_rw"],
            rth=buffers["smag_rth"],
            rqv=buffers.get("smag_rqv"),
            time_t=time_t,
            kmv=kmv, khv=khv, scalar_rows=scalar_rows,
        )
        if cfg.km_opt == 2:
            # Vertical TKE self-diffusion with 2*Km_v (vertical_diffusion_2
            # :4332-4341, doubling :4896-4904), PBL-off only like the rest
            # of vertical_diffusion_2.  Same vertical_diffusion_s operator
            # as the scalars, with Km_v in place of Kh_v and the doubled
            # increment.
            tke_t = getattr(state, "tke0" if time_t else "tke")
            rtke = state.scratch((nz_, ny_, nx_), "smag_rtke")
            tmp = state.scratch((nz_, ny_, nx_), "smag_tke_tmp")
            tmp[...] = 0
            common = _wrf_smag_grid_args(state, cfg, time_t=time_t)
            tail = [np.int32(nz_), np.int32(ny_), np.int32(nx_),
                    np.int32(state.phb.ndim == 3),
                    np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg))]
            mass_grid = ((nx_ + _TPB - 1) // _TPB, ny_, nz_)
            get_kernel("smag2d", "wrf_smag_vd_s")(
                mass_grid, (_TPB, 1, 1),
                tuple(common + [
                    tke_t, state.thb, np.int32(0),
                    np.int32(state.thb.ndim == 3), kmv, tmp,
                ] + tail))
            rtke += 2.0 * tmp
            budget_v = tke_budget.term(state, cfg, "diffusion_v")
            if budget_v is not None:
                budget_v[...] = 2.0 * tmp
        for f, _tend, _xk, _c1, _c2, slot, _stag in specs:
            _zero_open_strips(
                buffers[slot], cfg, 1)


def prepare_fixed_tendencies(state: DomainState, cfg: RunConfig) -> None:
    """Build WRF's time-t ``*_tendf``/``scalar_tends`` once per step.

    ``module_first_rk_step_part2`` computes km_opt=4 mixing and sixth-order
    diffusion from the RK-step-1 (time-t) fields.  ``rk_addtend_dry`` and
    ``rk_update_scalar[_pd]`` then consume the same fixed tendencies on all
    three RK passes.  gpuwm shares the existing ``smag_r*`` carrying
    buffers between both source packages, while one ``diff6_*`` temporary
    per staggering preserves their different open-boundary widths without
    retaining a second full set of per-species buffers.
    """
    nz, ny, nx = state.p.shape
    include_smag = cfg.km_opt in (2, 3, 4)
    include_diff6 = cfg.diff_6th_opt > 0
    if not (include_smag or include_diff6):
        return

    # The K arrays are needed only by Smagorinsky.  Scalar placeholders let
    # _smag2d_specs describe the common carrying buffers without allocating
    # two mass-grid K fields on a diff6-only run.
    if include_smag:
        km = state.scratch((nz, ny, nx), "smag_km")
        kh = state.scratch((nz, ny, nx), "smag_kh")
    else:
        km = kh = None
    specs = _smag2d_specs(state, km, kh, time_t=True,
                          kmv=_horizontal_w_km(state, cfg))
    for f0, _tend, _xk, _c1, _c2, slot, _stag in specs:
        state.scratch(f0.shape, slot)[...] = 0
    if cfg.km_opt == 2:
        # The prognostic-TKE forward tendency (tke_rhs + self-diffusion),
        # consumed by advance_tke_stage on every RK pass.
        state.scratch((nz, ny, nx), "smag_rtke")[...] = 0
        # A term this configuration never produces must read as an honest
        # zero for the step, not as the previous step's value.
        tke_budget.clear_fields(state, cfg)

    mu_t = state.mub2d + state.mup0
    if include_smag:
        _compute_wrf_smag_tendencies(
            state, cfg, km, kh, specs, time_t=True)

    if include_diff6:
        factor = _clock_scaled_diff6_factor(cfg)
        temp_slot = {"x": "diff6_x", "y": "diff6_y",
                     "z": "diff6_z", "": "diff6_m"}
        exempt = diff6_exempt_slots(cfg)
        # row[5] is the row's carrying slot (see _smag2d_specs).
        diff6_rows = [row for row in specs if row[5] not in exempt]
        if cfg.km_opt == 2:
            # WRF applies the 6th-order filter to tke through
            # rk_scalar_tend unless tke_mix6_off (Registry default
            # .false., Registry.EM_COMMON:2893).
            diff6_rows.append((state.tke0, None, None, state.c1h,
                               state.c2h, "smag_rtke", ""))
        for f0, _tend, _xk, c1, c2, slot, stag in diff6_rows:
            tmp = state.scratch(f0.shape, temp_slot[stag])
            tmp[...] = 0
            launch_diff6(f0, tmp, mu_t, c1, c2, factor, cfg.dt,
                         cfg.diff_6th_opt, stagger=stag,
                         phb=state.phb, msfu=state.msfu, msfv=state.msfv,
                         slopeopt=cfg.diff_6th_slopeopt,
                         thresh=cfg.diff_6th_thresh,
                         dx=cfg.dx, dy=cfg.dy,
                         # Boundary-aware reads: the outermost computed
                         # staggered face takes WRF's honest boundary
                         # datum (u ide-3 / v jde-3); the width-3 mask
                         # below is then exactly WRF's loop exclusion.
                         bnd_x=_boundary_x(cfg), bnd_y=_boundary_y(cfg))
            _zero_open_strips(tmp, cfg, 3)
            state.scratch(f0.shape, slot)[:] += tmp
            if slot == "smag_rtke":
                budget_6 = tke_budget.term(state, cfg, "diffusion_6th")
                if budget_6 is not None:
                    budget_6[...] = tmp


def add_fixed_dry_tendencies(state: DomainState, cfg: RunConfig) -> None:
    """Add the held time-t forward tendencies to one RK slow pass."""
    if cfg.km_opt not in (2, 3, 4) and cfg.diff_6th_opt <= 0:
        return
    # K values are not consumed here; the specs provide shapes/targets.
    for f0, tend, _xk, _c1, _c2, slot, _stag in _smag2d_specs(
            state, None, None, time_t=True):
        if tend is not None:
            tend += state.scratch(f0.shape, slot)


def fixed_scalar_tendencies(state: DomainState, cfg: RunConfig):
    """Return held scalar forward tendencies by Registry field name."""
    if state.qv is None or (cfg.km_opt not in (2, 3, 4)
                            and cfg.diff_6th_opt <= 0):
        return None
    names = list(SPECIES)
    names += list(extra_moist_species(state))
    shape = state.p.shape
    return {name: state.scratch(shape, "smag_r" + name) for name in names}


def add_smag2d_tendencies(state: DomainState, cfg: RunConfig,
                          first: bool) -> None:
    """WRF km_opt=4 metric-aware ``diff_opt=2`` Smagorinsky mixing.

    WRF timing semantics (module_first_rk_step_part2 + rk_addtend_dry): the
    deformation, K, and the mixing tendencies are computed ONCE per model
    step on RK stage 1 (``first=True``) from the time-t fields into the
    forward-tendency scratch buffers, then ADDED to every stage's slow
    tendencies.  Momentum mixes with K_m; WRF's ``theta - T0`` field and
    moisture mix with K_h = K_m/prandtl = 3*K_m.  gpuwm reconstructs the
    theta field from ``thp + thb - T0`` inside the scalar-flux kernel.  The
    production kernels carry
    WRF's geopotential metrics, map factors, terrain-coordinate deformation,
    slope-limited K, tensor momentum stress, and metric scalar fluxes.
    """
    nz, ny, nx = state.p.shape
    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    specs = _smag2d_specs(state, km, kh,
                          kmv=_horizontal_w_km(state, cfg))
    if first:
        _compute_wrf_smag_tendencies(
            state, cfg, km, kh, specs, time_t=False)
    for f, tend, _xk, _c1, _c2, slot, _stag in specs:
        if tend is not None:
            tend += state.scratch(f.shape, slot)


def apply_smag2d_moisture(state: DomainState, cfg: RunConfig,
                          dt_eff: float) -> None:
    """Fold the stage's km_opt=4 moisture mixing into qv/qc/qr.

    WRF ``rk_update_scalar`` advances scalars with ``advect_tend +
    sc_tend`` in one coupled update; ``advance_scalars_stage`` applied the
    advective part, so adding ``dt_eff*sc_tend/C(mu_new)`` here completes
    the identical sum.  Deviation from WRF on the PD final stage: WRF's
    ``advect_scalar_pd`` folds the accumulated non-advective tendencies --
    this mixing included -- into its provisional low-order state
    ``ph_low``, so WRF's PD limiter DOES see the mixing when it
    renormalizes the fluxes; gpuwm instead adds the mixing after the
    PD-limited advective update, outside the limiter's positivity guard.
    The mixing term itself is identical either way, but any negative
    excursion it produces here is not renormalized away -- the same order
    of deviation ``apply_diff6`` documents for the moisture increment
    (the K_h fluxes are down-gradient, so the excursions stay at
    rounding level in practice; tests/test_smag2d.py gates q >= -1e-6).
    """
    nz, ny, nx = state.p.shape
    chm = (state.c1h[:, None, None] * state.total_mu()[None]
           + state.c2h[:, None, None])
    for name in SPECIES:
        q = getattr(state, name)
        q += dt_eff * state.scratch((nz, ny, nx), "smag_r" + name) / chm
    for name in extra_moist_species(state):
        q = getattr(state, name)
        q += dt_eff * state.scratch((nz, ny, nx), "smag_r" + name) / chm


def _prepare_small_step_init_launch(state: DomainState, cfg: RunConfig):
    """Bind the two invariant small-step initialization launches.

    ``cfg`` is consumed for one thing only: whether each horizontal axis is
    periodic.  The uv kernel builds WRF's ``muu``/``muus`` inline and
    ``calc_mu_uv`` picks the boundary face's off-domain neighbour from
    ``periodic_x``/``periodic_y`` (module_big_step_utilities_em.F:59-115),
    so the flag has to reach the kernel; ArWen's spelling of "not periodic"
    is ``_boundary_x``/``_boundary_y`` (open, specified, or nested), the same
    predicate ``slow_pgf`` and the advection launchers already pass.
    """
    nz, ny, nx = state.p.shape
    block = (256,)
    uv_n = nz * (ny + 1) * (nx + 1)
    uv_grid = ((uv_n + 255) // 256,)
    uv_kernel = get_kernel("dycore", "small_step_init_uv")
    uv_args = (
        state.u_pp, state.v_pp, state.u0, state.v0, state.u, state.v,
        state.mup0, state.mup, state.mub2d, state.c1h, state.c2h,
        state.msfu, state.msfv, np.int32(state.has_msf),
        np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(nz), np.int32(ny), np.int32(nx),
    )

    column_grid = ((ny * nx + 255) // 256,)
    column_kernel = get_kernel("dycore", "small_step_init_column")
    column_args = (
        state.w_pp, state.th_pp, state.ph_pp, state.mu_pp, state.al_pp,
        state.p_pp, state.p_pp_old, state.w0, state.w, state.thp0,
        state.thp, state.php0, state.php, state.p, state.alt, state.mup0,
        state.mup, state.mub2d, state.thb, state.c1h, state.c2h,
        state.c1f, state.c2f, state.rdnw, state.msft,
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(nz), np.int32(ny), np.int32(nx),
    )

    def launch() -> None:
        uv_kernel(uv_grid, block, uv_args)
        column_kernel(column_grid, block, column_args)

    return launch


def _init_small_steps(state: DomainState, cfg: RunConfig) -> None:
    """WRF ``small_step_prep`` + ``calc_p_rho`` (step 0).

    The acoustic perturbations are the deviations of the *time-t* fields
    (the *0 copies) from the current stage reference t*, coupled by the
    matching hybrid column-mass increments (``c1h*mu + c2h`` on half
    levels, ``c1f*mu + c2f`` on full levels); p''/alpha'' are then
    diagnosed from the linearized EOS so the first substep's pressure
    gradient is consistent.  On stage 1 (t* = t) every perturbation is
    exactly zero.
    """
    _prepare_small_step_init_launch(state, cfg)()


def _prepare_small_step_finish_launch(state: DomainState, cfg: RunConfig,
                                      hdiab_dt: float = 0.0):
    """Bind the two invariant small-step finish launches.

    ``cfg`` carries the same periodic-axis decision the init launcher
    documents: ``small_step_finish`` divides by the very ``muu``/``muus``
    pair ``small_step_prep`` multiplied by (module_small_step_em.F:96-98),
    so the two kernels must agree on the boundary face's neighbour or the
    coupling is not undone.
    """
    nz, ny, nx = state.p.shape
    block = (256,)
    uv_n = nz * (ny + 1) * (nx + 1)
    uv_grid = ((uv_n + 255) // 256,)
    uv_kernel = get_kernel("dycore", "small_step_finish_uv")
    uv_args = (
        state.u, state.v, state.u_pp, state.v_pp, state.mup, state.mu_pp,
        state.mub2d, state.c1h, state.c2h, state.msfu, state.msfv,
        np.int32(state.has_msf),
        np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(nz), np.int32(ny), np.int32(nx),
    )

    h_diabatic = state.h_diabatic if hdiab_dt else state.p
    column_grid = ((ny * nx + 255) // 256,)
    column_kernel = get_kernel("dycore", "small_step_finish_column")
    column_args = (
        state.w, state.thp, state.php, state.mup, state.w_pp, state.th_pp,
        state.ph_pp, state.mu_pp, state.mub2d, state.thb, state.c1h,
        state.c2h, state.c1f, state.c2f, state.msft, h_diabatic,
        DTYPE(hdiab_dt), np.int32(bool(hdiab_dt)),
        np.int32(state.has_msf), np.int32(state.thb.ndim == 3),
        np.int32(nz), np.int32(ny), np.int32(nx),
    )

    def launch() -> None:
        uv_kernel(uv_grid, block, uv_args)
        column_kernel(column_grid, block, column_args)

    return launch


def _finish_small_steps(state: DomainState, cfg: RunConfig,
                        hdiab_dt: float = 0.0) -> None:
    """WRF ``small_step_finish``: fold the acoustic perturbations into the
    uncoupled prognostic fields — the new RK stage estimate.

    ``hdiab_dt`` engages WRF's h_diabatic removal on the FINAL RK step
    (module_small_step_em.F:408-426, the ``rk_step == rk_order`` branch):
    the coupled theta numerator drops
    ``dts*number_of_small_timesteps*(c1h(k)*mut+c2h(k))*h_diabatic``
    (:421) — exactly the amount the substeps integrated from the
    ``add_h_diabatic_tendency`` term with the same stage mass ``mu_s``
    (WRF ``mut``), so theta(t+dt) carries NO net h_diabatic contribution
    and the heating enters the state once, in
    ``microphysics.moist_physics_finish``.  Callers pass the final stage's
    length ``nsub*dtau`` (= cfg.dt) there and 0.0 elsewhere (stages 1-2
    keep the heating in their provisional estimates, :408-415).  Float64
    mirror: ``gpuwm.verify.npref.np_small_step_finish_theta``.
    """
    _prepare_small_step_finish_launch(state, cfg, hdiab_dt)()


def _advance_stage(state: DomainState, dt_eff: float) -> None:
    """Advection-only uncoupled prognostic update from the stage-0 fields.

    The tendencies are coupled (mass-weighted with the hybrid increments),
    so each variable q advances through C(mu_new)*q_new = C(mu_old)*q_old
    + dt_eff * r_q_t and is then uncoupled by the new column mass at its
    staggering.  With mu' frozen (acoustic=False) mu_old == mu_new and
    this reduces to q_new = q_old + dt_eff * r_q_t / C(mu).
    """
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    c1f = state.c1f[:, None, None]
    c2f = state.c2f[:, None, None]
    mu0 = state.mub2d + state.mup0                    # (ny, nx) stage-0 mass
    mu = state.mub2d + state.mup                      # (ny, nx) updated mass

    # Map factors (Task 3): the working tendencies are for the msf-coupled
    # variables (U = C*u/msfu, theta as (1/msft)*d(mu*theta)/dt), so
    # uncoupling multiplies each by its msf (identity by default).
    if state.has_msf:
        rth = state.rth_t * state.msft[None]
        rut = state.ru_t * state.msfu[None]
        rvt = state.rv_t * state.msfv[None]
        rwt = state.rw_t * state.msft[None]
    else:
        rth, rut, rvt, rwt = (state.rth_t, state.ru_t, state.rv_t,
                              state.rw_t)

    thb = _b3(state.thb)
    state.thp[...] = (((c1h * mu0[None] + c2h) * (thb + state.thp0)
                       + dt_eff * rth)
                      / (c1h * mu[None] + c2h)) - thb

    mux0, mux = mu_at_u_faces(mu0), mu_at_u_faces(mu)
    state.u[...] = (((c1h * mux0[None] + c2h) * state.u0
                     + dt_eff * rut) / (c1h * mux[None] + c2h))

    muy0, muy = mu_at_v_faces(mu0), mu_at_v_faces(mu)
    state.v[...] = (((c1h * muy0[None] + c2h) * state.v0
                     + dt_eff * rvt) / (c1h * muy[None] + c2h))

    state.w[...] = (((c1f * mu0[None] + c2f) * state.w0
                     + dt_eff * rwt) / (c1f * mu[None] + c2f))


def launch_diff6(f, tend, mut, c1, c2, factor: float, dt: float, opt: int,
                 stagger: str = "", *, phb=None, msfu=None, msfv=None,
                 slopeopt: int = 0, thresh: float = 0.10,
                 dx: float = 0.0, dy: float = 0.0,
                 bnd_x: bool = False, bnd_y: bool = False) -> None:
    """ADD the WRF 6th-order horizontal diffusion coupled tendency for one
    field into ``tend`` (kernels/diff6.cu; float64 mirror
    ``gpuwm.verify.npref.np_diff6``).

    ``mut (ny, nx)`` is the total dry column mass the fluxes are coupled
    with (WRF ``mut``); ``c1``/``c2`` the hybrid coefficients at the
    field's levels (``c1h/c2h`` for half-level fields, ``c1f/c2f`` for w).
    ``opt`` is ``diff_6th_opt``: 2 zeroes any up-gradient flux (monotonic),
    any other positive value is the plain operator.  ``stagger`` selects
    the grid position as in ``launch_add_diff2``: ``""`` mass points,
    ``"x"`` u, ``"y"`` v, ``"z"`` w points (BC-pinned boundary levels get
    no tendency).  No dx/dy enters the untapered operator: it removes
    ``factor`` of a 2-D 2dx checkerboard's amplitude per full-``dt``
    integration by construction (``coef = factor/2^6/(2*dt)``, the Fortran
    normalization).

    ``slopeopt >= 1`` with a 3-D ``phb`` engages WRF's terrain-slope taper
    (``diff_6th_slopeopt``, sixth_order_diffusion
    module_big_step_utilities_em.F:6487-6501/6569-6583): each face flux is
    scaled by ``max(1 - dzmax/(thresh*9.81*dx), 0)`` with ``dzmax`` the
    msf-scaled ``phb`` face jump at the field's own level; ``dx``/``dy``
    must then be the physical grid spacings.  ``msfu``/``msfv`` default to
    identity when omitted.  A 1-D (flat) ``phb`` or ``slopeopt = 0`` keeps
    the untapered arithmetic bitwise.

    ``bnd_x``/``bnd_y`` (callers pass ``_boundary_x(cfg)``/``_boundary_y``:
    open or specified/nested forcing on that axis) enable the seam
    post-pass for the staggered field on that axis: the outermost
    computed staggered face -- WRF's u(ide-3)/v(jde-3), which the
    specified/nested and open loop bounds INCLUDE -- is recomputed by
    ``kernels/diff6_seam.cu`` with WRF's honest read of the stored true
    boundary datum ``field(ide)``/``field(jde)``
    (module_big_step_utilities_em.F:6354-6358/:6381-6385 bounds,
    :6465-6467/:6547-6549 reads), replacing the periodic-wrap kernel's
    corrupt value there.  ``tend`` must enter zeroed when a flag is set
    (both production callers zero it): the seam face's prior
    accumulation is discarded by the replacement.  The main kernel
    binary is untouched, so every face other than the seam set is
    bit-identical with the flags on or off, and periodic launches are
    bit-identical to the pre-seam tree
    (tests/test_diff6_boundary_face.py pins both).
    """
    nlev, nys, nxs = f.shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    variant = 1 if stagger == "x" else (2 if stagger == "y" else 0)
    coef = factor * 0.015625 / (2.0 * dt)
    slope = int(slopeopt) >= 1 and phb is not None and phb.ndim == 3
    if slope and (dx <= 0.0 or dy <= 0.0):
        raise ValueError("diff_6th_slopeopt >= 1 needs positive dx/dy")
    if slope:
        phb_arg = phb
        msfu_arg = (msfu if msfu is not None
                    else cp.ones((ny, nx + 1), dtype=DTYPE))
        msfv_arg = (msfv if msfv is not None
                    else cp.ones((ny + 1, nx), dtype=DTYPE))
    else:                                  # never dereferenced by the kernel
        phb_arg = msfu_arg = msfv_arg = mut
    # WRF: dzthresh = diff_6th_thresh*9.81*dx (the routine's literal 9.81)
    kern = get_kernel("diff6", "diff6")
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    kern(grid, (_TPB, 1, 1),
         (f, tend, mut, c1, c2, phb_arg, msfu_arg, msfv_arg,
          DTYPE(coef), np.int32(opt), np.int32(1 if slope else 0),
          DTYPE(thresh * 9.81 * dx), DTYPE(thresh * 9.81 * dy),
          np.int32(nlev), np.int32(ny), np.int32(nys),
          np.int32(nx), np.int32(nxs), np.int32(variant),
          np.int32(1 if stagger == "z" else 0)))
    if bnd_x and stagger == "x":
        _launch_diff6_seam("diff6_seam_u", f, tend, mut, c1, c2, phb_arg,
                           msfu_arg, msfv_arg, coef, opt, slope, thresh,
                           dx, dy, nlev, ny, nx, bnd_y)
    if bnd_y and stagger == "y":
        _launch_diff6_seam("diff6_seam_v", f, tend, mut, c1, c2, phb_arg,
                           msfu_arg, msfv_arg, coef, opt, slope, thresh,
                           dx, dy, nlev, ny, nx, bnd_x)


def _launch_diff6_seam(name, f, tend, mut, c1, c2, phb_arg, msfu_arg,
                       msfv_arg, coef, opt, slope, thresh, dx, dy,
                       nlev, ny, nx, bnd_cross) -> None:
    """Recompute the WRF-computed high-side staggered face (kernels/
    diff6_seam.cu): u's east column nx-3 / v's north row ny-3, whose
    dflux_p1 reads the stored true boundary datum field(ide)/field(jde).

    The main periodic-wrap kernel's value on that face is corrupt (it
    wraps to the OPPOSITE boundary), so the face is zeroed here and the
    seam kernel writes WRF's honest arithmetic over WRF's own index range
    -- the cross-axis range [3, n-4] when the cross axis is also forced
    (``bnd_cross``), the full periodic range otherwise, matching the
    caller's subsequent width-3 ``_zero_open_strips`` exactly.  Callers
    zero ``tend`` before ``launch_diff6``, so replacing this face's
    accumulation is exact (documented in ``launch_diff6``).
    """
    seam_u = name == "diff6_seam_u"
    n_along, n_cross = (nx, ny) if seam_u else (ny, nx)
    if n_along < 6:                    # WRF bounds empty: ids+3 > ide-3
        return
    h0, h1 = (3, n_cross - 4) if bnd_cross else (0, n_cross - 1)
    if h1 < h0:
        return
    if seam_u:
        tend[:, :, nx - 3] = 0         # drop the wrapped-read value
    else:
        tend[:, ny - 3, :] = 0
    kern = get_kernel("diff6_seam", name)
    span = h1 - h0 + 1
    kern(((span + _TPB - 1) // _TPB, 1, nlev), (_TPB, 1, 1),
         (f, tend, mut, c1, c2, phb_arg, msfu_arg, msfv_arg,
          DTYPE(coef), np.int32(opt), np.int32(1 if slope else 0),
          DTYPE(thresh * 9.81 * dx), DTYPE(thresh * 9.81 * dy),
          np.int32(nlev), np.int32(ny), np.int32(nx),
          np.int32(h0), np.int32(h1), np.int32(1 if bnd_cross else 0)))


def _clock_scaled_diff6_factor(cfg: RunConfig) -> float:
    """Per-step factor whose clock-interval composition equals WRF's."""
    clock_dt = cfg.clock_dt if cfg.clock_dt > 0.0 else cfg.dt
    factor = float(cfg.diff_6th_factor)
    if clock_dt == cfg.dt:
        return factor
    if not 0.0 <= factor <= 1.0:
        raise ValueError(
            "clock-scaled diff_6th_factor must lie in [0, 1], got "
            f"{factor}")
    if factor == 1.0:
        return 1.0
    # A 2dx mode retains (1-factor) over one WRF model-clock step.  Taking
    # the matching fractional retention prevents eight 7.5 s compatibility
    # steps from applying the 60 s namelist factor eight times.
    return -math.expm1((cfg.dt / clock_dt) * math.log1p(-factor))


def apply_diff6(state: DomainState, cfg: RunConfig) -> None:
    """Apply one complete diff6 increment (verification utility).

    Production :func:`step` does **not** use this post-update helper.  It
    calls :func:`prepare_fixed_tendencies` once on the time-t fields, adds
    the held dry tendencies to all three RK passes, and passes the held
    scalar tendencies through ``rk_update_scalar[_pd]``.  This helper is
    retained for kernel normalization/clock-composition tests: callers must
    seed the ``*0`` fields, and it applies ``dt*tendf`` directly through the
    post-update hybrid mass.

    At non-periodic lateral boundaries the outermost boundary-normal
    STAGGERED face (WRF's ide-3/jde-3) is computed exactly as WRF
    computes it: the boundary-aware kernel reads the stored true
    boundary datum field(ide)/field(jde) (``launch_diff6`` ``bnd_x``/
    ``bnd_y``), and the width-3 host mask is then precisely WRF's loop
    exclusion on every axis and stagger.

    The real74 compatibility driver advances eight internal dynamics steps
    per 60 s WRF model-clock interval.  In that case the per-call factor is
    the eighth-root retention equivalent, so the eight 2dx applications
    compose to exactly the namelist ``diff_6th_factor`` instead of applying
    that 60 s factor eight times.

    Applied to u, v, w, theta' and all allocated transported moisture
    scalars.  WRF diffuses theta up to a constant offset, which is
    identical to theta' on flat coordinate surfaces.  Under
    ``moist_mix6_off`` the WRF ``moist``-array rows drop out here exactly as
    they drop out of the production row set (:func:`diff6_exempt_slots`), so
    the helper keeps measuring what production applies.
    """
    factor, opt = _clock_scaled_diff6_factor(cfg), cfg.diff_6th_opt
    mu_t = state.mub2d + state.mup0                # time-t mass (WRF mut)
    mu = state.total_mu()                          # post-step mass: uncouple
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    c1f = state.c1f[:, None, None]
    c2f = state.c2f[:, None, None]
    chm = c1h * mu[None] + c2h                     # mass-point coupling
    targets = [
        (state.u0, state.u, "x", state.c1h, state.c2h,
         c1h * mu_at_u_faces(mu)[None] + c2h, "diff6_x"),
        (state.v0, state.v, "y", state.c1h, state.c2h,
         c1h * mu_at_v_faces(mu)[None] + c2h, "diff6_y"),
        (state.w0, state.w, "z", state.c1f, state.c2f,
         c1f * mu[None] + c2f, "diff6_z"),
        (state.thp0, state.thp, "", state.c1h, state.c2h, chm, "diff6_m"),
    ]
    if state.qv is not None:
        exempt = diff6_exempt_slots(cfg)
        names = [name for name in SPECIES + tuple(extra_moist_species(state))
                 if "smag_r" + name not in exempt]
        targets += [(getattr(state, name + "0"), getattr(state, name), "",
                     state.c1h, state.c2h, chm, "diff6_m")
                    for name in names]
    for f0, f, stag, c1, c2, chmf, slot in targets:
        tendf = state.scratch(f0.shape, slot)
        tendf[...] = 0
        launch_diff6(f0, tendf, mu_t, c1, c2, factor, cfg.dt, opt,
                     stagger=stag,
                     # WRF diff_6th_slopeopt terrain taper (no-op with the
                     # default 0 or a flat 1-D phb; the base-state slope
                     # and per-face msf enter exactly as the Fortran).
                     phb=state.phb, msfu=state.msfu, msfv=state.msfv,
                     slopeopt=cfg.diff_6th_slopeopt,
                     thresh=cfg.diff_6th_thresh, dx=cfg.dx, dy=cfg.dy,
                     bnd_x=_boundary_x(cfg), bnd_y=_boundary_y(cfg))
        _zero_open_strips(tendf, cfg, 3)        # WRF sixth_order_diffusion
        f += DTYPE(cfg.dt) * tendf / chmf       # non-periodic loop bounds


def _prepare_emdiv_filter_launch(state: DomainState, cfg: RunConfig,
                                  mudf: cp.ndarray,
                                  mu_prev: cp.ndarray | None = None):
    """Bind one stage's invariant external-mode filter launch."""
    nz, ny, nx = state.p.shape
    # The raw union-grid kernel retains the five eager FP32 boundaries in
    # each gx/gy chain.  Saving mu_prev is independent and uses the k=0
    # owner thread for each mass column, avoiding a separate device copy.
    save = mu_prev is not None
    mu_prev_arg = mu_prev if save else mudf
    n = nz * (ny + 1) * (nx + 1)
    grid = ((n + 255) // 256,)
    block = (256,)
    kernel = get_kernel("acoustic", "apply_emdiv")
    args = (
        state.u_pp, state.v_pp, mudf, state.mu_pp, mu_prev_arg, state.c1h,
        state.msfu, state.msfv, DTYPE(-cfg.emdiv * cfg.dx),
        DTYPE(-cfg.emdiv * cfg.dy), np.int32(state.has_msf),
        np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
        np.int32(_boundary_forced(cfg)), np.int32(cfg.spec_zone),
        np.int32(save), np.int32(nz), np.int32(ny), np.int32(nx),
    )

    def launch() -> None:
        kernel(grid, block, args)

    return launch


def apply_emdiv_filter(state: DomainState, cfg: RunConfig,
                       mudf: cp.ndarray,
                       mu_prev: cp.ndarray | None = None) -> None:
    """WRF external-mode divergence damping (module_small_step_em.F).

    Before each acoustic substep the perturbation momenta get WRF
    ``advance_uv``'s ``mudf_xy`` term (lines 809/868, 880/942; map factors
    1): ``u'' += c1h * (-emdiv*dx*(mudf_i - mudf_{i-1}))`` and the y
    analogue, where ``mudf (ny, nx)`` is the PREVIOUS substep's
    column-mass tendency (``advance_mu_t``: dmdt + mu_tend; zeroed by
    ``small_step_prep`` at RK stage 1 ONLY — module_small_step_em.F:128
    guards the reset with ``IF (rk_step == 1)`` — so only the very first
    acoustic iteration of the model step adds nothing; stages 2/3 inherit
    the prior stage's final tendency).  This damps the column-integrated (external) mode --
    WRF's stock stabilizer for open lateral boundaries (the em_quarter_ss
    namelist runs emdiv = 0.01).  Boundary-normal faces at open boundaries
    are excluded exactly like the acoustic pressure gradient (the mudf_xy
    loop shares advance_uv's bounds); periodic faces wrap.  Adding the
    increment before the substep kernel is equivalent to WRF's in-kernel
    ordering: both land on u'' before ``advance_mu_t`` consumes it.
    Reference: ``gpuwm.verify.npref.np_emdiv_uv``.
    """
    _prepare_emdiv_filter_launch(state, cfg, mudf, mu_prev)()


def _prepare_emdiv_mudf_launch(state: DomainState, cfg: RunConfig,
                               mudf: cp.ndarray, mu_prev: cp.ndarray,
                               dtau: float):
    """Bind one stage's invariant mudf recurrence launch."""
    ny, nx = state.mup.shape
    grid = ((ny * nx + 255) // 256,)
    block = (256,)
    kernel = get_kernel("acoustic", "update_mudf")
    boundary_forced = np.int32(_boundary_forced(cfg))
    args = (
        mudf, state.mu_pp, mu_prev, DTYPE(dtau), boundary_forced,
        boundary_forced, np.int32(cfg.spec_zone), np.int32(ny), np.int32(nx),
    )

    def launch() -> None:
        kernel(grid, block, args)

    return launch


def _update_emdiv_mudf(state: DomainState, cfg: RunConfig,
                       mudf: cp.ndarray, mu_prev: cp.ndarray,
                       dtau: float) -> None:
    """Finish the exact FP32 mudf recurrence and boundary-strip zeroing."""
    _prepare_emdiv_mudf_launch(state, cfg, mudf, mu_prev, dtau)()


def _prepare_sumflux_launch(name: str, targets: tuple, sources: tuple = (),
                            nsub: int = 0):
    """Bind one invariant batched WRF sumflux launch."""
    sizes = tuple(np.uint64(array.size) for array in targets)
    nmax = max(int(size) for size in sizes)
    grid = ((nmax + 255) // 256,)
    block = (256,)
    args = (*targets, *sources)
    if name == "finish_sumflux":
        args += (DTYPE(nsub),)
    args += (*sizes, np.uint64(nmax))
    kernel = get_kernel("acoustic", name)

    def launch() -> None:
        kernel(grid, block, args)

    return launch


def _sumflux_launch(name: str, targets: tuple, sources: tuple = (),
                    nsub: int = 0) -> None:
    """Batch three independent staggered WRF sumflux array operations."""
    _prepare_sumflux_launch(name, targets, sources, nsub)()


def _zero_open_strips(buf: cp.ndarray, cfg: RunConfig, width: int,
                      stag_high_extra: int = 0) -> None:
    """Zero a coupled mixing tendency over the strip WRF's loop bounds skip
    at open lateral boundaries (no-op when periodic).

    ``width = 3`` mirrors ``sixth_order_diffusion`` and ``width = 1``
    mirrors ``horizontal_diffusion``.  On a non-staggered axis the outer
    ``width`` entries on each side are exactly the points WRF's bounds
    exclude (e.g. mass fields under open_x: WRF computes ids+3..ide-4 of
    the ids..ide-1 cells, so 3 columns go to zero per side); without this
    the wrapped stencils couple the two open boundaries.  On the
    boundary-normal STAGGERED axis (``nx + 1`` u faces under open_x /
    ``ny + 1`` v faces under open_y) WRF's high-side exclusion is
    ``width`` faces counted from the boundary face ide itself, and the
    high side here takes ``width + stag_high_extra``: a caller whose
    stencil cannot reproduce WRF's read of the true boundary datum at the
    outermost computed face may pass ``stag_high_extra = 1`` to zero that
    face as well.  No production caller does any more: the diff6 kernel's
    ``bnd_x``/``bnd_y`` mode and the smag2d u/v kernels both make the
    honest boundary-datum read themselves (WRF computes u face ide-3
    reading field(i+3) = u(ide); smag2d.cu ``open_x``/``open_y``,
    diff6.cu ``bndx``/``bndy``), so ``width = 3`` (diff6) and ``width =
    1`` (smag2d) are exactly WRF's exclusions for every stagger.  The
    parameter is retained for reconstructing the historical pre-fix mask
    (tests/test_diff6_boundary_face.py's 4d2ce99 capture)."""
    x_hi = width + (stag_high_extra if buf.shape[-1] == cfg.nx + 1 else 0)
    y_hi = width + (stag_high_extra if buf.shape[-2] == cfg.ny + 1 else 0)
    if _boundary_x(cfg):
        buf[..., :width] = 0
        buf[..., -x_hi:] = 0
    if _boundary_y(cfg):
        # Ellipsis keeps this valid for 2-D (ny, nx) buffers too -- the
        # emdiv mudf strip is the live 2-D caller.
        buf[..., :width, :] = 0
        buf[..., -y_hi:, :] = 0


def set_w_surface(state: DomainState, cfg: RunConfig) -> None:
    """Kinematic lower boundary condition on the uncoupled w (WRF
    ``set_w_surface``, module_bc_em.F): ``w(sfc) = u.grad(ht)`` with the
    cf1..cf3-weighted three lowest half levels of u/v, periodic in x/y.
    Exactly zero over flat terrain.  ``dycore.step`` calls this at the end
    of every step ("reset surface w for consistency", WRF solve_em); case
    builders call it once at init (WRF start_em).
    """
    ht = state.ht
    uc = (state.cf1 * state.u[0] + state.cf2 * state.u[1]
          + state.cf3 * state.u[2])                    # (ny, nx+1)
    vc = (state.cf1 * state.v[0] + state.cf2 * state.v[1]
          + state.cf3 * state.v[2])                    # (ny+1, nx)
    dyn = cp.roll(ht, -1, 0) - ht
    dys = ht - cp.roll(ht, 1, 0)
    dxe = cp.roll(ht, -1, 1) - ht
    dxw = ht - cp.roll(ht, 1, 1)
    if _boundary_y(cfg):
        dys[0, :] = dyn[0, :]
        dyn[-1, :] = dys[-1, :]
    if _boundary_x(cfg):
        dxw[:, 0] = dxe[:, 0]
        dxe[:, -1] = dxw[:, -1]
    sfc = ((0.5 / cfg.dy) * (dyn * vc[1:, :] + dys * vc[:-1, :])
           + (0.5 / cfg.dx) * (dxe * uc[:, 1:] + dxw * uc[:, :-1]))
    if state.has_msf:              # WRF set_w_surface: msfty*(v part) +
        sfc *= state.msft          # msftx*(u part); isotropic single msft
    state.w[0] = sfc


#: Radiation phase speed c* (m/s) of the Klemp-Wilhelmson open lateral BC:
#: WRF's cb = 25 (share/module_model_constants.F:47, consumed by the open
#: radiative blocks in dyn_em/module_advect_em.F), adjudicated over the
#: plan's original 30 (the published KW78 value) — the local WRF source is
#: authoritative.  Must match npref.OPEN_CB.
OPEN_CB = 25.0

_BC_THREADS = 256


def apply_open_radiative_bc(state: DomainState, cfg: RunConfig) -> None:
    """Radiative open-BC tendency for the boundary-normal velocities.

    WRF gravity-wave radiative lateral BC (dyn_em/module_advect_em.F
    ``advect_u``/``advect_v`` open blocks, ``tendency = tendency + ...``):
    with ``cfg.open_x``/``open_y`` the one-sided Klemp-Wilhelmson radiative
    term with the outbound-only phase speed ``u_n -/+ c*`` (``OPEN_CB``) is
    ADDED to the coupled slow tendency at the two boundary-normal velocity
    faces (Task 11 prerequisite; Task 9 REPLACED, dropping the terms WRF
    retains there).  The open-aware advection kernels already excluded the
    boundary-normal advection at those faces -- the radiative term stands
    in for it -- and they skip the acoustic pressure gradient
    (gpuwm.core.acoustic) plus the large-step PGF, so over the substeps
    they integrate the radiation equation on top of whatever advection WRF
    keeps (u's vertical advection when only x is open).  No-op with the
    periodic defaults.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    if cfg.open_x:
        kernel = get_kernel("openbc", "open_u_radiative")
        blocks = (nz * ny + _BC_THREADS - 1) // _BC_THREADS
        kernel((blocks,), (_BC_THREADS,),
               (state.ru_t, state.u, state.mup, state.mub2d,
                state.c1h, state.c2h, DTYPE(1.0 / cfg.dx), DTYPE(OPEN_CB),
                np.int32(nz), np.int32(ny), np.int32(nx)))
    if cfg.open_y:
        kernel = get_kernel("openbc", "open_v_radiative")
        blocks = (nz * nx + _BC_THREADS - 1) // _BC_THREADS
        kernel((blocks,), (_BC_THREADS,),
               (state.rv_t, state.v, state.mup, state.mub2d,
                state.c1h, state.c2h, DTYPE(1.0 / cfg.dy), DTYPE(OPEN_CB),
                np.int32(nz), np.int32(ny), np.int32(nx)))


def apply_w_damping(state: DomainState, cfg: RunConfig,
                    ww: cp.ndarray) -> None:
    """WRF ``w_damp`` (module_big_step_utilities_em.F), ``w_damping = 1``.

    Where the vertical Courant number ``|ww/(c1f*mu+c2f)*rdnw*dt|`` of the
    stage's diagnosed eta mass flux ``ww`` exceeds the activation value
    (w_beta = 1), the coupled w tendency is pushed against the vertical
    motion by ``w_alpha*(vert_cfl - w_crit_cfl)`` — a limiter, not physics
    (WRF adds it for robustness at marginal CFL).  Interior w levels only;
    no-op unless ``cfg.w_damping == 1``.
    """
    if cfg.w_damping != 1:
        return
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    kernel = get_kernel("openbc", "w_damp")
    blocks = ((nz - 1) * ny * nx + _BC_THREADS - 1) // _BC_THREADS
    kernel((blocks,), (_BC_THREADS,),
           (state.rw_t, ww, state.w, state.mup, state.mub2d,
            state.c1f, state.c2f, state.rdnw, DTYPE(cfg.dt),
            np.int32(nz), np.int32(ny), np.int32(nx)))


def apply_open_zero_gradient(state: DomainState, cfg: RunConfig) -> None:
    """Zero-gradient outbound boundary values at open lateral boundaries.

    The plan's "scalars/theta zero-gradient outbound" clause (WRF
    ``set_physical_bc3d`` open-BC extension, share/module_bc.F, applied to
    the boundary cells themselves since gpuwm carries no ghost strip): at
    every level where the boundary-normal velocity at the boundary face
    points outward, theta', w (interior levels), the tangential velocity,
    and the moisture scalars in the boundary column are copied from the
    first interior neighbour; inflow levels keep their prognostic values.
    phi'/mu' stay prognostic everywhere (their boundary evolution is
    column-local plus the radiated boundary-face divergence).  Called at
    the end of every RK stage; no-op with the periodic defaults.

    The tangential velocity's mask needs one more row/column than the
    boundary-normal mask has, and where that extra one comes from is decided
    by the TANGENTIAL axis's own boundary condition -- see the comment at the
    two ``concatenate`` calls.  ``open_x`` with a periodic y is the case that
    made it matter.
    """
    if cfg.open_x:
        uw = state.u[:, :, 0]                          # (nz, ny) west face
        ue = state.u[:, :, -1]                         # east face
        outw, oute = uw < 0.0, ue > 0.0
        for q in (state.thp, state.qv, state.qc, state.qr):
            if q is None:
                continue
            q[:, :, 0] = cp.where(outw, q[:, :, 1], q[:, :, 0])
            q[:, :, -1] = cp.where(oute, q[:, :, -2], q[:, :, -1])
        if getattr(state, "qi", None) is not None:
            for name in extra_moist_species(state):
                q = getattr(state, name)
                q[:, :, 0] = cp.where(outw, q[:, :, 1], q[:, :, 0])
                q[:, :, -1] = cp.where(oute, q[:, :, -2], q[:, :, -1])
        oww = (uw[1:] + uw[:-1]) < 0.0                 # w levels 1..nz-1
        owe = (ue[1:] + ue[:-1]) > 0.0
        state.w[1:-1, :, 0] = cp.where(oww, state.w[1:-1, :, 1],
                                       state.w[1:-1, :, 0])
        state.w[1:-1, :, -1] = cp.where(owe, state.w[1:-1, :, -2],
                                        state.w[1:-1, :, -1])
        # The mask is (nz, ny) at mass rows and v needs (nz, ny+1) rows, so
        # the tangential axis has to supply one more.  WHICH one is the y
        # boundary condition's business, not a detail: on a NON-periodic y
        # the extra row is a real north face and takes the zero-gradient
        # repeat; on a PERIODIC y row ny is the ALIAS of row 0
        # (tilestream/spec.py's module docstring states the convention, and
        # `harness.make_state` seeds it) and must take row 0's mask, or the
        # two copies of one physical face are given opposite treatments.
        #
        # MEASURED at 96x64x25, open_x with y periodic, ONE resident step:
        # the repeat leaves max|v[ny] - v[0]| = 2.51 m/s, at exactly the two
        # x boundary columns and nowhere else, from a seeded state where it
        # was zero.  The periodic arm of the same probe stays at 0, which is
        # what shows the probe can see the invariant at all.  It is also
        # what made this configuration untileable: a tiled scatter writes the
        # alias FROM row 0 and so cannot reproduce a domain that has broken
        # its own alias.
        y_alias = slice(-1, None) if _boundary_y(cfg) else slice(0, 1)
        mvw = cp.concatenate([outw, outw[:, y_alias]], axis=1)  # v rows 0..ny
        mve = cp.concatenate([oute, oute[:, y_alias]], axis=1)
        state.v[:, :, 0] = cp.where(mvw, state.v[:, :, 1], state.v[:, :, 0])
        state.v[:, :, -1] = cp.where(mve, state.v[:, :, -2],
                                     state.v[:, :, -1])
    if cfg.open_y:
        vs = state.v[:, 0, :]                          # (nz, nx) south face
        vn = state.v[:, -1, :]                         # north face
        outs, outn = vs < 0.0, vn > 0.0
        for q in (state.thp, state.qv, state.qc, state.qr):
            if q is None:
                continue
            q[:, 0, :] = cp.where(outs, q[:, 1, :], q[:, 0, :])
            q[:, -1, :] = cp.where(outn, q[:, -2, :], q[:, -1, :])
        if getattr(state, "qi", None) is not None:
            for name in extra_moist_species(state):
                q = getattr(state, name)
                q[:, 0, :] = cp.where(outs, q[:, 1, :], q[:, 0, :])
                q[:, -1, :] = cp.where(outn, q[:, -2, :], q[:, -1, :])
        ows = (vs[1:] + vs[:-1]) < 0.0
        own = (vn[1:] + vn[:-1]) > 0.0
        state.w[1:-1, 0, :] = cp.where(ows, state.w[1:-1, 1, :],
                                       state.w[1:-1, 0, :])
        state.w[1:-1, -1, :] = cp.where(own, state.w[1:-1, -2, :],
                                        state.w[1:-1, -1, :])
        # The x mirror of the y rule above: with ``open_y`` and a PERIODIC x
        # (``open_x`` off and no specified/nested forcing) u's slot nx is the
        # alias of slot 0 and takes slot 0's mask.
        x_alias = slice(-1, None) if _boundary_x(cfg) else slice(0, 1)
        mus = cp.concatenate([outs, outs[:, x_alias]], axis=1)  # u faces 0..nx
        mun = cp.concatenate([outn, outn[:, x_alias]], axis=1)
        state.u[:, 0, :] = cp.where(mus, state.u[:, 1, :], state.u[:, 0, :])
        state.u[:, -1, :] = cp.where(mun, state.u[:, -2, :],
                                     state.u[:, -1, :])



def close_periodic_alias(state: DomainState, cfg: RunConfig) -> None:
    """On a periodic axis, the extra staggered face IS face 0.  Say so.

    ``u`` has ``nx+1`` faces for ``nx`` mass cells and ``v`` has ``ny+1`` for
    ``ny``.  When the axis wraps, the last of those is not a face of its own:
    it is the SAME FACE as index 0, reached the other way round.
    ``tilestream.spec`` builds every gather, scatter and halo band on exactly
    that identity (spec.py:34-52: "gathers never read the alias slot"; under
    ``periodic=True`` the alias slot is logical face 0), and
    ``TileSpec.scatter`` writes the domain's alias slot FROM face 0 for that
    reason.

    NOTHING WAS MAINTAINING IT.  No periodic stencil writes the slot -- they
    wrap their indices instead -- so it kept whatever the initialiser left,
    which for a window of a larger analysis is the real column one past the
    window: on this case's 128-wide window, ``u[:, :, nx]`` sat 17.4 m/s away
    from ``u[:, :, 0]`` at t=0 and stayed there.

    Consumers read it.  ``dycore._mass_divergence`` (:122-123) differences
    ``ru[:, :, 1:] - ru[:, :, :-1]``, so the last mass column's mass tendency
    came off that stale face, and ``physics._prepare_atmosphere``
    (physics.py:1379-1381) destaggers ``0.5*(u[:-1] + u[1:])``, so the last
    mass column and the top mass row of every surface-layer and PBL carrier
    were driven by a wind that is not part of the solution.  That is wrong on
    ONE GPU.

    It is also the whole of the single-vs-multi physics divergence.  A
    decomposition re-derives the slot from face 0 -- by the contract above,
    and unavoidably: the rank that holds the domain's alias slot receives it
    through the halo exchange from the rank that owns face 0 -- so the two
    arms fed different winds to the same cells and differed after ONE step,
    diffusely and nowhere near a rank seam.  MEASURED, 128x96x49, one step,
    full physics through ``MultiGPUDomain``: 61 of 158 carriers differ
    without this call, 0 with it, at 1x1, 1x2 and 2x2 alike.

    Closing it here, at the end of the step, is inert for a RANK: the halo
    exchange overwrites the tile's outermost faces before the next step, and
    the outermost mass column those faces feed is a halo column that
    ``TileSpec.scatter`` discards.  A NON-PERIODIC axis carries a real
    closing boundary face -- the specified/open lateral boundary owns it --
    and is left alone.
    """
    if not _boundary_x(cfg):
        state.u[..., -1] = state.u[..., 0]
    if not _boundary_y(cfg):
        state.v[..., -1, :] = state.v[..., 0, :]


def step(state: DomainState, cfg: RunConfig, *, acoustic: bool = True,
         mass_flux_observer=None, mass_flux_accumulator=None,
         refl_10cm_due: bool = False) -> None:
    """Advance ``state`` one full RK3 step of length ``cfg.dt``.

    ``acoustic=True`` (default) runs the full ARW split-explicit loop: per
    stage, zero tendencies -> EOS diagnostics at t* -> slow tendencies ->
    initialize the acoustic perturbations from the time-t fields -> acoustic
    substeps -> fold the perturbations into the new stage estimate.
    ``acoustic=False`` is the Phase-1 advection-only path (w, phi', mu'
    frozen).  ``mass_flux_observer``, when supplied, is called with each
    final-RK-stage acoustic substep's boundary-flux mass increment.  Summing
    those increments independently closes the completed step's domain-mass
    change and is otherwise a zero-cost dormant diagnostic.
    ``mass_flux_accumulator`` is the device-resident alternative (a
    :class:`MassFluxAccumulator`): it takes the same increments in the
    same order without reading any of them, so a long integration pays no
    per-substep host synchronization.  The two keywords are mutually
    exclusive and supplying both raises.
    ``refl_10cm_due`` is the history-step flag threaded to microphysics;
    the active scheme computes and stashes radar reflectivity before its
    finish-stage theta writeback and the post-microphysics EOS refresh.

    Config-gated physics (no-ops with the defaults): with ``km_opt=1``,
    constant-K diffusion joins every stage's slow tendencies when
    ``cfg.khdif/kvdif > 0``;
    ``cfg.km_opt=4`` adds the WRF 2-D Smagorinsky horizontal mixing
    (:func:`add_smag2d_tendencies` -- computed once per step on stage 1,
    applied every stage; moisture via :func:`apply_smag2d_moisture`); and
    ``damp_opt=3`` engages the Klemp-Dudhia-Hassiotis implicit w-only
    Rayleigh damper inside the acoustic w solve (gpuwm.core.acoustic; no-op
    on the ``acoustic=False`` path, which has no acoustic substeps); and
    ``diff_6th_opt > 0`` computes WRF's 6th-order monotonic horizontal
    forward tendency once from the time-t fields and feeds that held
    tendency to every RK stage (the advection-only test path skips it).

    Moist states (Task 5): each stage's transport fluxes come from the
    public :func:`stage_fluxes` surface, and qv/qc/qr advance right after
    the stage's acoustic loop (``gpuwm.core.moist.advance_scalars_stage``;
    PD limiter on the final stage) so the next stage's EOS sees consistent
    (theta, qv).  The advection-only path stays dry.

    Microphysics (Task 6): with ``cfg.mp_physics != 0`` the configured
    scheme (``gpuwm.core.microphysics.apply``) adjusts theta/qv/qc/qr once
    per step after the RK3 loop -- WRF solve_em's non-timesplit
    microphysics slot -- followed by a diagnostics refresh; the default
    ``mp_physics = 0`` leaves the step bitwise unchanged.  The scheme's
    clamped theta increment is retained as ``state.h_diabatic`` (K/s, WRF
    moist_physics_finish_em) and the NEXT step feeds it to every RK
    stage's theta tendency (:func:`add_h_diabatic_tendency`, WRF
    rk_addtend_dry) while the final stage's fold removes its accumulated
    net contribution (``_finish_small_steps``, WRF small_step_finish) --
    the dynamics see the heating continuously, the state is heated once.

    Open lateral boundaries + w-damping (Task 9 + Task 11 rework, acoustic
    path only): with ``cfg.open_x``/``open_y`` the advection kernels take
    WRF's open-aware bounds, each stage ADDS the radiative term at the
    boundary-normal velocity faces (:func:`apply_open_radiative_bc`), the
    acoustic substeps skip the boundary-face pressure gradient and clamp
    their cross-boundary ghost reads, and each stage ends with the
    zero-gradient-outbound boundary values
    (:func:`apply_open_zero_gradient`).  ``cfg.emdiv > 0`` engages WRF's
    external-mode divergence damping across the acoustic substeps
    (:func:`apply_emdiv_filter`).  ``cfg.w_damping = 1`` adds WRF's
    vertical-velocity limiter per stage (:func:`apply_w_damping`).  All
    no-ops with the defaults; the ``acoustic=False`` Phase-1 test path
    stays periodic.

    Unsupported combinations fail loudly here (and, for the config-only
    parts, in ``gpuwm.config.load_config``): terrain (``terrain_opt != 0``
    or any nonzero ``state.ht``) with radiative-open boundaries, and
    constant-K diffusion (``khdif/kvdif > 0``) with radiative-open or
    specified boundaries, raise ``NotImplementedError`` because their
    remaining stencils/bounds are periodic-only.  The non-monotonic
    ``diff_6th_opt = 1`` with moisture raises ``ValueError`` (unlimited
    fluxes bypass the PD limiter).  Coriolis/curvature is boundary-aware.
    """
    if mass_flux_observer is not None and mass_flux_accumulator is not None:
        raise ValueError(
            "mass_flux_observer and mass_flux_accumulator are mutually "
            "exclusive: the accumulator exists to remove the per-substep "
            "host synchronization the list observer takes, and running "
            "both reinstates it")
    if acoustic and cfg.time_step_sound % 2 != 0:
        raise ValueError(
            f"time_step_sound must be even, got {cfg.time_step_sound}: RK3 "
            "stage 2 runs time_step_sound//2 acoustic substeps of "
            "dt/time_step_sound, which mis-times the dt/2 stage for odd "
            "values."
        )
    # THE fail-closed km_opt decision: this is the site that actually
    # decides whether a horizontal mixing operator runs, so it asks the
    # same shared question the loaders ask rather than restating it (the
    # two used to be separate transcriptions of one rule).
    validate_km_opt(cfg)
    if cfg.km_opt in (2, 3, 4) and (cfg.khdif > 0.0 or cfg.kvdif > 0.0):
        raise ValueError(
            f"km_opt={cfg.km_opt} selects WRF Smagorinsky mixing; "
            "khdif/kvdif are constant-K controls for km_opt=1 and cannot "
            "also be active")
    if cfg.mp_physics != 0 and getattr(state, "h_diabatic", None) is None:
        raise ValueError(
            f"mp_physics={cfg.mp_physics} requires cfg.moist=True: the "
            "state carries no h_diabatic array for the retained "
            "microphysics heating")
    if cfg.open_x or cfg.open_y:
        # getattr: the CPU-only guard test drives step() with a stub state
        # that carries just ht/qv (tests/test_config.py).
        if cfg.terrain_opt != 0 or bool((state.ht != 0).any()):
            raise NotImplementedError(
                "terrain + open lateral boundaries is not wired: "
                "set_w_surface and the advance_w_phi kinematic surface BC "
                "difference ht with unconditional periodic wraps, which "
                "would couple the two open boundaries through the terrain "
                "slope")
    if ((cfg.open_x or cfg.open_y or _boundary_forced(cfg))
            and (cfg.khdif > 0.0 or cfg.kvdif > 0.0)):
        raise NotImplementedError(
            "constant-K diffusion (khdif/kvdif > 0) + open or specified "
            "lateral boundaries is not wired: launch_add_diff2 has no "
            "boundary-aware path, so its stencils would wrap across the "
            "domain; use km_opt=4 and/or diff_6th_opt=2 for boundary "
            "dissipation")
    if cfg.diff_6th_opt == 1 and state.qv is not None:
        raise ValueError(
            "diff_6th_opt=1 (non-monotonic) with moisture is not allowed: "
            "the unlimited 6th-order fluxes are applied outside the "
            "PD-limited transport and can drive qv/qc/qr negative; use "
            "the monotonic diff_6th_opt=2")
    if not acoustic and physics_enabled(cfg):
        raise NotImplementedError(
            "non-timesplit physics requires the acoustic RK3 path "
            "(step(acoustic=False) is the Phase-1 dry advection test path)")
    for name in _PROGNOSTICS:                         # device time-t copies
        getattr(state, name + "0")[...] = getattr(state, name)
    if state.qv is not None:
        for name in SPECIES:
            getattr(state, name + "0")[...] = getattr(state, name)
        if getattr(state, "qi", None) is not None:
            for name in extra_moist_species(state):
                getattr(state, name + "0")[...] = getattr(state, name)
    if getattr(state, "tke", None) is not None:       # km_opt=2 carrier
        state.tke0[...] = state.tke

    # WRF solve_em: non-timesplit physics is evaluated once during the
    # first RK pass and held fixed for all three passes.  EOS/phy_prep must
    # see the time-t state.  The default scheme IDs are all zero, so this
    # branch performs no device operation for every frozen Phase-1/2 case.
    physics_tendencies = None
    if physics_enabled(cfg):
        if state.physics is None:
            raise RuntimeError(
                "physics is enabled but the state has no PhysicsDriver; "
                "call gpuwm.core.physics.initialize_physics first")
        update_diagnostics(state, cfg.hypsometric_opt)
        physics_tendencies = state.physics.compute(state, cfg)

    if not acoustic:
        if state.qv is not None:
            raise NotImplementedError(
                "moist transport requires the acoustic dycore path "
                "(step(acoustic=False) is the Phase-1 dry advection test "
                "path)")
        if cfg.km_opt in (2, 3, 4):
            raise NotImplementedError(
                f"km_opt={cfg.km_opt} Smagorinsky mixing requires the "
                "acoustic dycore path (step(acoustic=False) is the "
                "Phase-1 dry advection test path)")
        for istage, dt_eff in enumerate((cfg.dt / 3.0, cfg.dt / 2.0,
                                         cfg.dt)):
            for name in _TENDENCIES:
                getattr(state, name)[...] = 0
            update_diagnostics(state, cfg.hypsometric_opt)
            add_advection_tendencies(state, cfg)
            if cfg.km_opt == 1:
                add_diffusion_tendencies(state, cfg)
            apply_state_lateral_boundaries(state, cfg, rk_stage=istage)
            _advance_stage(state, dt_eff)
        set_w_surface(state, cfg)
        apply_state_boundary_values(state, cfg,
                                    state.elapsed_seconds + cfg.dt)
        state.elapsed_seconds += cfg.dt
        return

    # WRF module_first_rk_step_part2/rk_scalar_tend: compute the forward
    # mixing/diff6 tendencies once from the saved time-t fields.  Every RK
    # pass below consumes these same buffers; the final PD scalar pass folds
    # them before flux renormalization in advance_scalars_stage.
    # Smagorinsky's stresses/fluxes use WRF's dry-air density with vapor
    # loading, rho=(1+qv)/alt (exactly 1/alt for a dry state).  A freshly
    # initialized state has not otherwise run phy_prep/EOS yet, so refresh
    # the time-t diagnostics before evaluating K and its forward tendencies.
    if cfg.km_opt in (2, 3, 4):
        update_diagnostics(state, cfg.hypsometric_opt)
    prepare_fixed_tendencies(state, cfg)
    fixed_scalars = fixed_scalar_tendencies(state, cfg)

    ns = cfg.time_step_sound
    stages = ((1, cfg.dt / 3.0),
              (max(ns // 2, 1), cfg.dt / ns),
              (ns, cfg.dt / ns))
    emdiv = cfg.emdiv > 0.0
    nzs, nys, nxs = state.p.shape
    launch_small_step_init = _prepare_small_step_init_launch(state, cfg)
    launch_small_step_finish = _prepare_small_step_finish_launch(state, cfg)
    if cfg.mp_physics != 0:
        final_hdiab_dt = stages[-1][0] * stages[-1][1]
        launch_small_step_finish_final = _prepare_small_step_finish_launch(
            state, cfg, final_hdiab_dt)
    else:
        launch_small_step_finish_final = launch_small_step_finish
    small_step_finishes = (
        launch_small_step_finish, launch_small_step_finish,
        launch_small_step_finish_final)
    for istage, (nsub, dtau) in enumerate(stages):
        for name in _TENDENCIES:
            getattr(state, name)[...] = 0
        update_diagnostics(state, cfg.hypsometric_opt)  # p, al, alt at t*
        ru, rv, ww = stage_fluxes(state, cfg)
        # WRF calc_cq is fixed at the RK-stage reference state and shared by
        # horizontal_pressure_gradient plus every acoustic substep.
        stage_cq = prepare_moist_cq(state, cfg)
        _add_slow_tendencies(state, cfg, ru, rv, ww, cq=stage_cq)
        if istage == 0:
            # WRF RTHFTEN for the cumulus schemes that take it.  THIS LINE
            # is the whole contract: rth_t holds the stage reference
            # fluxes' theta advection and nothing else until the next
            # statement folds physics into it (see the function's
            # docstring for the GFSCHEME double-count trap).  Stage 1
            # matches the once-per-step capture convention h_diabatic and
            # the qv lateral tendency already use.
            capture_advective_theta_forcing(state)
        if physics_tendencies is not None:
            physics_tendencies.add_to_slow(state)
        if cfg.mp_physics != 0:                       # WRF rk_addtend_dry's
            add_h_diabatic_tendency(state)            # h_diabatic slot, every
        if cfg.km_opt == 1:
            add_diffusion_tendencies(state, cfg)      # RK stage
        add_fixed_dry_tendencies(state, cfg)           # held Smag/diff6 tendf
        apply_w_damping(state, cfg, ww)               # w_damping=1 only
        apply_state_lateral_boundaries(state, cfg, rk_stage=istage)
        apply_open_radiative_bc(state, cfg)           # open_x/open_y only
        launch_small_step_init()                     # additive stage seed
        acoustic_coefficients = prepare_acoustic_coefficients(
            state, cfg, dtau, cq=stage_cq)            # fixed for this stage
        mudf = None
        if emdiv:
            mudf = state.scratch(state.mup.shape, "acoustic_mudf")
            launch_emdiv_filter = _prepare_emdiv_filter_launch(
                state, cfg, mudf)
            if istage == 0:
                # WRF small_step_prep zeros MUDF under IF(rk_step==1) ONLY
                # (module_small_step_em.F:128-136); stages 2/3 must inherit
                # the previous stage's final column-mass tendency so emdiv
                # acts on 6 of 7 acoustic iterations, not 4 of 7.
                mudf[...] = 0
        launch_acoustic_substep = prepare_acoustic_substep_launch(
            state, cfg, dtau, acoustic_coefficients, mudf=mudf)
        # WRF sumflux (module_small_step_em.F:1473, called every acoustic
        # iteration, solve_em.F:1561): accumulate the small-timestep
        # time-averaged mass fluxes ru_m/rv_m/ww_m -- "needed for
        # consistent mass-conserving scalar advection".  Scalars only;
        # theta/momentum keep the stage fluxes exactly as WRF does.
        moist = state.qv is not None
        # Prognostic TKE (km_opt=2) advects with the same acoustic
        # time-averaged mass fluxes as the moist scalars (WRF
        # rk_scalar_tend for tke, solve_em.F:2362-2399), so a dry TKE run
        # accumulates sumflux too.
        scalars = moist or getattr(state, "tke", None) is not None
        if scalars:
            ru_m = state.scratch((nzs, nys, nxs + 1), "rk_ru_m")
            rv_m = state.scratch((nzs, nys + 1, nxs), "rk_rv_m")
            ww_m = state.scratch((nzs + 1, nys, nxs), "rk_ww_m")
            _sumflux_launch(                          # sumflux iteration==1
                "zero_sumflux", (ru_m, rv_m, ww_m))
            launch_sumflux_accumulation = _prepare_sumflux_launch(
                "accumulate_sumflux", (ru_m, rv_m, ww_m),
                (state.u_pp, state.v_pp, state.ww_pp))
        for i in range(nsub):
            if emdiv:
                launch_emdiv_filter()                 # previous substep's
                                                      # mudf (zero only on
                                                      # step's 1st substep)
            launch_acoustic_substep(first=(i == 0))
            if scalars:                               # WRF sumflux: post-
                launch_sumflux_accumulation()
            if mass_flux_observer is not None and istage == 2:
                mass_flux_observer(
                    dtau * boundary_mass_tendency(state, cfg))
            elif mass_flux_accumulator is not None and istage == 2:
                mass_flux_accumulator.add(
                    dtau * boundary_mass_tendency_device(state, cfg))
            # advance_mu_th stores WRF MUDF directly before its rounded mass
            # update; reconstructing it from two FP32 states loses sub-ULP
            # column-mass tendencies.
        # WRF small_step_finish: the h_diabatic removal runs on the final RK
        # step only, over dts*number_of_small_timesteps = dt.
        small_step_finishes[istage]()
        if scalars:                                   # stage length nsub*dtau
            # WRF sumflux (iteration == number_of_small_timesteps): the
            # substep mean plus the stage-reference coupled fluxes --
            # ru_m = mean(u'') + C(muu)*u_t*/msfuy (F:1584-1592), ww_m =
            # mean(ww'') + ww_1.  Scalars advect with these (solve_em.F:
            # 2210-2212), making the q == const tendency telescope exactly
            # against the acoustic mu update (SK2008 D11).
            _sumflux_launch(
                "finish_sumflux", (ru_m, rv_m, ww_m), (ru, rv, ww), nsub)
            if moist:
                advance_scalars_stage(
                    state, cfg, ru_m, rv_m, ww_m, nsub * dtau,
                    final=(istage == len(stages) - 1),
                    apply_relax=(istage == 0),
                    physics_tendencies=physics_tendencies,
                    fixed_tendencies=fixed_scalars,
                    # WRF RQVFTEN, the qv half of the cumulus advective
                    # forcing pair.  Stage 1, where the non-PD branch runs
                    # and the tendency is still pure advection.
                    export_advective_forcing=(istage == 0))
            if getattr(state, "tke", None) is not None:
                from gpuwm.core.moist import advance_tke_stage
                advance_tke_stage(
                    state, cfg, ru_m, rv_m, ww_m, nsub * dtau,
                    final=(istage == len(stages) - 1),
                    fixed_tendency=state.scratch(
                        state.p.shape, "smag_rtke"))
        apply_open_zero_gradient(state, cfg)          # radiative-open BCs
    # km_opt=2 budget: one device reduction over the completed step, taken
    # here because state.mup is still the mass the final RK scalar update
    # divided by.  Report-only (gpuwm/core/tke_budget.py); a no-op unless
    # cfg.tke_budget is on.
    tke_budget.accumulate(state, cfg)
    apply_state_boundary_values(state, cfg,
                                state.elapsed_seconds + cfg.dt)
    set_w_surface(state, cfg)                         # WRF solve_em epilogue
    update_diagnostics(state, cfg.hypsometric_opt)
    if cfg.nwp_diagnostics == 1:
        # WRF's nwp_diagnostics severe-weather lane, UP_HELI_MAX member:
        # fold this completed step's updraft helicity into the running max
        # (u/v/w/ph are final here; the later microphysics adjustment does
        # not touch them).  Reads model state only; writes only the
        # diagnostic's own scratch slots -- inertness is pinned by
        # tests/test_uh_lifecycle.py.
        update_up_heli_max(state, cfg)
    if cfg.mp_physics != 0:                           # post-RK3 adjustment
        # h_diabatic capture cadence: once per INTERNAL step with cfg.dt.
        # Under the real74 compatibility integrator (dt = clock_dt/8) this
        # deviates from WRF's once-per-model-clock-step cadence — ratified
        # and documented in PROVENANCE.md entry D1 (self-consistent: the
        # capture dt and apply window are the same internal step; the
        # native dt=60 path restores WRF cadence with no code change).
        microphysics_result = apply_microphysics(     # (WRF microphysics
            state, cfg, cfg.dt, refl_10cm_due=refl_10cm_due)
        if state.physics is not None:
            state.physics.accept_microphysics(microphysics_result)
        update_diagnostics(state, cfg.hypsometric_opt)  # after the RK loop)
    close_periodic_alias(state, cfg)
    state.elapsed_seconds += cfg.dt


def run_steps(state: DomainState, cfg: RunConfig, n: int, *,
              acoustic: bool = True) -> None:
    """Advance ``state`` by ``n`` full RK3 steps."""
    for _ in range(n):
        step(state, cfg, acoustic=acoustic)


def stability_report(state: DomainState, cfg: RunConfig | None = None,
                     *, boundary_width: int | None = None) -> dict:
    """Runtime health check with one compact device-to-host result readback.

    The two-stage device reduction returns max |u|, |w|, and |theta'| plus
    max ``|w_upper| / dz_cell`` with each upper-face velocity paired with
    its own live geopotential layer thickness.  NaNs propagate through the
    same maxima, so ``"nan"`` checks exactly the same fields as before; bad
    layer geometry makes the CFL non-finite and therefore fails the runner's
    safety gate.  When ``boundary_width`` is supplied, the same traversal
    also returns the first overall |w| argmax and boundary/free-interior |w|
    maxima used by the real74 integration monitor.
    """
    if state.u.size == 0 or state.w.size == 0 or state.thp.size == 0:
        raise ValueError(
            "zero-size array to reduction operation maximum which has no "
            "identity")
    width = 0 if boundary_width is None else int(boundary_width)
    if boundary_width is not None:
        ny, nx = state.w.shape[1:]
        if width <= 0:
            raise ValueError("boundary_width must be positive")
        if ny - 2 * width <= 0 or nx - 2 * width <= 0:
            raise ValueError(
                f"boundary_width={width} leaves an empty w interior for "
                f"{ny} x {nx}")
    largest = max(state.u.size, state.w.size, state.thp.size)
    nblocks = min(256, max(1, (largest + 255) // 256))
    partial = state.scratch((nblocks, 9), "integration_health_partial")
    result = state.scratch((8,), "integration_health_result")
    if cfg is None:
        ph = phb = state.w  # ncells=0 below: valid, never dereferenced
        ncells = 0
        phb_full = 0
    else:
        ph = state.php
        phb = state.phb
        ncells = state.thp.size
        phb_full = int(state.phb.ndim == 3)
    kernel = get_kernel("health", "health_partial")
    kernel((nblocks,), (256,),
           (state.u, state.w, state.thp, ph, phb, partial,
            np.uint64(state.u.size), np.uint64(state.w.size),
            np.uint64(state.thp.size), np.uint64(ncells),
            np.int32(phb_full), np.int32(width),
            np.int32(state.w.shape[1]), np.int32(state.w.shape[2]),
            DTYPE(c.G)))
    kernel = get_kernel("health", "health_final")
    kernel((1,), (256,), (partial, result, np.int32(nblocks)))
    host = cp.asnumpy(result)                           # sole health readback
    return decode_stability_record(host, cfg, boundary_width=boundary_width)


def decode_stability_record(host, cfg: RunConfig | None = None, *,
                            boundary_width: int | None = None) -> dict:
    """``health_final``'s eight-word record, as :func:`stability_report`'s dict.

    Factored out so a STREAMED domain, whose record is folded per tile inside
    the sweep (:mod:`gpuwm.core.streaming`), decodes through this exact
    function rather than through a copy of it.  The two paths differ in which
    memory the reduction READS and in nothing else, which is the whole claim
    that fold rests on -- and a duplicated decoder is how such a claim stops
    being true a year later.
    """
    u_max, w_max, th_max = (float(value) for value in host[:3])
    nan = not (math.isfinite(u_max) and math.isfinite(w_max)
               and math.isfinite(th_max))
    cfl = None
    horizontal_cfl = None
    vertical_cfl = None
    if cfg is not None and not nan:
        horizontal_cfl = cfg.dt * u_max / cfg.dx
        vertical_cfl = cfg.dt * float(host[5])
        cfl = max(horizontal_cfl, vertical_cfl)
    report = {"u_max": u_max, "w_max": w_max, "th_max": th_max,
              "cfl": cfl, "horizontal_cfl": horizontal_cfl,
              "vertical_cfl": vertical_cfl, "nan": nan}
    if boundary_width is not None:
        index_words = host[6:8].view(np.uint32)
        w_argmax = int(index_words[0]) | (int(index_words[1]) << 32)
        report.update(
            boundary_w_max=float(host[3]), interior_w_max=float(host[4]),
            w_argmax=w_argmax)
    return report


def stability_gate_failed(report: dict, *, max_cfl: float,
                          max_w_ms: float) -> bool:
    """True when a single-domain history sample crosses a safety limit.

    Equality remains accepted, preserving the established threshold
    convention; the first representable value above either limit fails.
    """

    cfl = report.get("cfl")
    w_max = report.get("w_max")
    return (
        bool(report.get("nan"))
        or cfl is None
        or not math.isfinite(float(cfl))
        or float(cfl) > max_cfl
        or w_max is None
        or not math.isfinite(float(w_max))
        or float(w_max) > max_w_ms
    )
