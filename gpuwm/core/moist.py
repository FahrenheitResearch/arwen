"""Moisture: positive-definite scalar transport and moist thermodynamics.

Transport (WRF ``module_advect_em.F`` ``advect_scalar_pd``; Skamarock 2006
MWR; ``moist_adv_opt=1``): qv/qc/qr are dry-mass-coupled scalars advected
through all three RK3 stages with WRF's acoustic-substep TIME-AVERAGED
mass fluxes ru_m/rv_m/ww_m (``sumflux``, module_small_step_em.F:1473 --
"needed for consistent mass-conserving scalar advection";
solve_em.F:2210-2212), accumulated by ``dycore.step`` as the substep mean
of the perturbation momenta/Omega'' plus the stage-reference fluxes from
:func:`gpuwm.core.dycore.stage_fluxes`.  Theta and momentum keep the
stage fluxes exactly as WRF's ``rk_tendency`` does; moisture never
re-derives Omega (Task 5 retired the ``rw = -(mu*w)`` placeholder in
advection.py).
Stages 1-2 use the unlimited 5th/3rd-order flux divergence
(advection.py/advection.cu); the RK3 FINAL stage only -- exactly as WRF
applies its PD filter -- decomposes every face flux as ``F = F_upwind1 +
F_corr`` and renormalizes the outgoing corrections of any cell they would
overdraw (kernels/pd_advection.cu: ``pd_fluxes`` + ``pd_renorm_apply``;
float64 mirrors ``np_pd_fluxes``/``np_pd_renorm_apply`` in
gpuwm.verify.npref).

The renormalization is exactly positive in exact arithmetic; the final
stage update clamps FP32 rounding residuals (order 1 ulp of the
renormalized outflow, in cells the limiter already drove to ~0) at zero,
which perturbs total scalar mass far below the 1e-6 relative conservation
gate.

Thermodynamics (ARW Tech Note ch. 2; dry-theta prognostic, WRF
``use_theta_m=0`` style): the EOS diagnostics use ``theta_m = theta *
(1 + Rv/Rd*qv)`` (diagnostics.cu; the ratio comes from the constants), the
w-equation buoyancy takes WRF's full moist ``pg_buoy_w`` form -- the
``1/(1+q_tot)`` factor on the vertical p' gradient plus the hydrometeor/
vapor loading ``-q_tot/(1+q_tot)*(c1f*mub+c2f)`` with ``q_tot = qv+qc+qr``
(dycore._add_slow_tendencies) -- and the acoustic coefficients pick up
theta_m through the moist pressure in ``c2a = gamma*p/alpha``.  The
linearized-EOS ratios ``(mu*theta)''/theta_ref`` remain invariant under the
per-cell ``(1 + Rv/Rd*qv)`` factor while qv is frozen over the substeps.

The acoustic path also matches WRF ``calc_cq`` each RK stage: ``cqu`` and
``cqv`` scale the horizontal acoustic pressure gradients, while ``cqw``
scales the implicit vertical pressure coefficients and forcing.  The sum
contains every active WRF ``moist`` mass species (qv/qc/qr and, for
mixed-phase schemes, qi/qs/qg), never their number-moment ``scalar`` fields.
RunConfig's
default-ON ``moist_cq`` switch exists only for falsification attribution;
it is not a science-tuning option and production configurations leave it
enabled.  A dry state has no qv allocation and stays on the exact legacy
kernel expressions without cq scratch or launch work.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.advection import launch_flux_div_scalar
from gpuwm.core.grid import BaseState, VerticalCoord
from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE, DomainState, init_at_rest

_TPB = 128  # threads per block along i (i fastest)

#: Frozen Phase-2 moisture species (allocation/operation order in state).
SPECIES = ("qv", "qc", "qr")

#: Ice-category mass species shared by WSM6, Thompson, and Morrison.
ICE_MASS_SPECIES = ("qi", "qs", "qg")

#: Morrison-only prognostic number moments.  ``nc`` is deliberately absent:
#: WRF's
#: default INUM=1 path diagnoses a fixed 250 cm-3 every Morrison call rather
#: than carrying a droplet-number tracer (module_mp_morr...F:116-124,
#: 1493-1496).  Precipitating/ice moments are transported as mixing ratios.
MORRISON_NUMBER_SPECIES = ("nr", "ni", "ns", "ng")
MORRISON_SPECIES = ICE_MASS_SPECIES + MORRISON_NUMBER_SPECIES

# Thompson mp=8 transports rain and ice number only.  The ordering follows
# its WRF Registry scalar package (qni, qnr) after gpuwm's historical
# rain-before-ice state allocation; all loops below discover actual fields so
# they never invent Morrison snow/graupel moments on a Thompson state.
THOMPSON_NUMBER_SPECIES = ("nr", "ni")
TRANSPORTED_NUMBER_SPECIES = MORRISON_NUMBER_SPECIES

# Native NSSL option 18 carries hail mass plus cloud/rain/ice/snow/graupel/
# hail number, predicted CCN, and graupel/hail particle-volume scalars.  All
# are Registry-transported dry-mass mixing ratios; only the first four ice/
# hail entries contribute water mass and acoustic/buoyancy loading.
NSSL_MASS_SPECIES = ("qi", "qs", "qg", "qh")
NSSL_SCALAR_SPECIES = (
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh")
NSSL_SPECIES = NSSL_MASS_SPECIES + NSSL_SCALAR_SPECIES


def extra_moist_species(state: DomainState) -> tuple[str, ...]:
    """Active transported species beyond qv/qc/qr.

    WSM6 owns the three ice mass fields.  Thompson owns those plus rain/ice
    number, while Morrison owns all four precipitating/ice number moments.
    Presence is checked per field so a Thompson ``nr`` cannot accidentally
    imply Morrison-only ``ns``/``ng`` storage.
    """
    if getattr(state, "qi", None) is None:
        return ()
    if getattr(state, "qh", None) is not None:
        return NSSL_SPECIES
    numbers = tuple(name for name in TRANSPORTED_NUMBER_SPECIES
                    if getattr(state, name, None) is not None)
    return ICE_MASS_SPECIES + numbers


def moist_species(state: DomainState) -> tuple[str, ...]:
    """Return the active transported scalar registry for ``state``.

    The conditional preserves the exact qv/qc/qr loop for mp=0/1 while
    exposing the exact prognostic mass/number moments carried by the active
    mixed-phase scheme.
    """
    return SPECIES + extra_moist_species(state)


def _mut2d(mu, ny: int, nx: int) -> cp.ndarray:
    """Column mass as a contiguous (ny, nx) float32 device array (accepts a
    scalar or any broadcastable array, mirroring the npref mirrors)."""
    return cp.ascontiguousarray(
        cp.broadcast_to(cp.asarray(mu, dtype=DTYPE), (ny, nx)))


def launch_pd_fluxes(q, q0, ru, rv, rw, mut, coord, dx, dy, dt,
                     fxl, fxc, fyl, fyc, fzl, fzc,
                     msft=None, has_msf=None,
                     open_x=False, open_y=False) -> None:
    """Fill the six PD flux arrays for scalar ``q`` (see pd_advection.cu).

    ``q`` is the RK stage estimate (high-order fluxes), ``q0`` the time-t
    scalar (upwind fluxes), ``mut`` the post-acoustic stage column mass
    (WRF ``muts``; scalar or (ny, nx)).  ``coord`` is anything carrying
    ``rdnw``/``c1h``/``c2h`` (a ``VerticalCoord`` or a ``DomainState``).
    ``msft`` (optional, Task 3) is the mass-point map factor: the upwind
    fluxes/Courant numbers then use the physical face spacings (WRF
    ``advect_scalar_pd``).  ``open_x``/``open_y`` select WRF's
    specified/open boundary treatment along that axis (zero boundary-normal
    faces + degraded near-boundary stencils, no wrap); an open axis needs
    >= 7 cells so the degrade bands cannot overlap.
    """
    nz, ny, nx = q.shape
    if open_x and nx < 7:
        raise ValueError(f"open_x PD advection needs nx >= 7, got {nx}")
    if open_y and ny < 7:
        raise ValueError(f"open_y PD advection needs ny >= 7, got {ny}")
    if has_msf is None:
        has_msf = msft is not None
    if msft is None:
        msft = cp.ones((ny, nx), dtype=DTYPE)
    kern = get_kernel("pd_advection", "pd_fluxes")
    grid = ((nx + 1 + _TPB - 1) // _TPB, ny + 1, nz + 1)
    kern(grid, (_TPB, 1, 1),
         (q, q0, ru, rv, rw, _mut2d(mut, ny, nx),
          cp.asarray(coord.c1h, dtype=DTYPE),
          cp.asarray(coord.c2h, dtype=DTYPE),
          cp.asarray(coord.rdnw, dtype=DTYPE),
          cp.asarray(coord.fnm, dtype=DTYPE),
          cp.asarray(coord.fnp, dtype=DTYPE), msft,
          DTYPE(dx), DTYPE(dy), DTYPE(dt),
          fxl, fxc, fyl, fyc, fzl, fzc,
          np.int32(nz), np.int32(ny), np.int32(nx), np.int32(has_msf),
          np.int32(open_x), np.int32(open_y)))


def launch_pd_renorm_apply(q0, mu_old, fxl, fxc, fyl, fyc, fzl, fzc,
                           tend, coord, dx, dy, dt,
                           msft=None, has_msf=None,
                           open_x=False, open_y=False) -> None:
    """ADD the renormalized PD flux divergence into ``tend``.

    ``mu_old`` is the time-t column mass (WRF ``mu_old``, coupling
    ``ph_low``); flux arrays as filled by :func:`launch_pd_fluxes`.
    ``msft`` (optional, Task 3): ph_low/flux_out weight the horizontal
    divergence by msft^2 (WRF msftx*msfty) and the vertical by msft, and
    the applied tendency weights the horizontal divergence by msft.
    ``open_x``/``open_y`` (must match :func:`launch_pd_fluxes`): the
    limiter skips the outermost cells and the applied horizontal
    divergences skip the boundary cells (WRF specified/open bounds,
    module_advect_em.F:7697-7715, 7817-7821, 7852-7856).
    """
    nz, ny, nx = q0.shape
    if has_msf is None:
        has_msf = msft is not None
    if msft is None:
        msft = cp.ones((ny, nx), dtype=DTYPE)
    kern = get_kernel("pd_advection", "pd_renorm_apply")
    grid = ((nx + _TPB - 1) // _TPB, ny, nz)
    kern(grid, (_TPB, 1, 1),
         (q0, _mut2d(mu_old, ny, nx), fxl, fxc, fyl, fyc, fzl, fzc,
          cp.asarray(coord.c1h, dtype=DTYPE),
          cp.asarray(coord.c2h, dtype=DTYPE),
          cp.asarray(coord.rdnw, dtype=DTYPE), msft,
          DTYPE(1.0 / dx), DTYPE(1.0 / dy), DTYPE(dt),
          tend, np.int32(nz), np.int32(ny), np.int32(nx),
          np.int32(has_msf), np.int32(open_x), np.int32(open_y)))


def _pd_fold_sources(state: DomainState, cfg: RunConfig, name: str,
                     q0, chm0, dt_eff: float, physics_tendencies,
                     fixed_tendency=None, lbc_held=None):
    """WRF ``rk_update_scalar_pd`` (module_em.F:1803-1916): on the final
    RK step with PD advection, fold the accumulated physics tendency into
    the TIME-T scalar with the time-t mass (mu_old = mu_new = mu_1 in the
    solve_em.F:1849-1866 call) and zero it, BEFORE the flux limiter -- so
    ``advect_scalar_pd``'s ph_low budget (module_advect_em.F:7733-7737)
    sees the sources ('add in physics tendency first if positive definite
    advection is used', solve_em.F:1839-1841).

    The fold covers the FULL tile INCLUDING the specified zone: the
    ``_spc`` loop bounds are captured at module_em.F:1863-1868 BEFORE the
    specified narrowing at F:1870-1878 (dead code for this routine -- the
    narrowed bounds are never used afterwards), and the fold loop at
    F:1889-1893 runs over the ``_spc`` bounds.  The ring cell's folded
    value is live interior forcing: it is the upwind donor for the first
    interior face and enters the first interior cell's ph_low
    (pinned by tests/test_pd_advection.py::
    test_pd_fold_covers_specified_ring).

    ``fixed_tendency`` carries WRF's stage-1 forward tendency (horizontal
    mixing and/or sixth-order diffusion).  ``lbc_held`` carries qv's held
    lateral-boundary tendency (spec +
    relax, captured once per model step on RK stage 1 exactly like WRF's
    ``moist_tend`` from relax_bdy_scalar/spec_bdy_scalar at
    solve_em.F:2255-2292); WRF's fold covers the full moist_tend, so the
    held LBC part folds here alongside physics.

    Returns ``q0`` itself when there is nothing to fold, else a scratch
    copy with the source applied (the caller must use it for the upwind
    fluxes, ph_low, AND the coupled update, and must NOT re-add the
    physics tendency afterwards).
    """
    physics = (physics_tendencies.scalar_for(name)
               if physics_tendencies is not None else None)
    if physics is None and fixed_tendency is None and lbc_held is None:
        return q0
    nz, ny, nx = q0.shape
    q0_eff = state.scratch((nz, ny, nx), "moist_pd_q0")
    q0_eff[...] = q0
    if fixed_tendency is not None:
        q0_eff += dt_eff * fixed_tendency / chm0
    if physics is not None:
        q0_eff += dt_eff * physics / chm0
    if lbc_held is not None:
        q0_eff += dt_eff * lbc_held / chm0
    return q0_eff


def advance_scalars_stage(state: DomainState, cfg: RunConfig,
                          ru, rv, ww, dt_eff: float, final: bool,
                          apply_relax: bool = True,
                          physics_tendencies=None,
                          fixed_tendencies=None) -> None:
    """Advance qv/qc/qr one RK stage from their time-t copies (``*0``).

    Called by ``dycore.step`` after each stage's acoustic loop with that
    stage's acoustic time-averaged mass fluxes ru_m/rv_m/ww_m (WRF
    ``sumflux``; see the module docstring) as ``ru``/``rv``/``ww``.
    Each scalar advances in coupled form (WRF
    ``rk_update_scalar``): ``C(mu_t)*q_t + dt_eff*tend``, uncoupled by the
    post-acoustic stage mass ``C(mu_new)``.  On the final stage with
    ``cfg.moist_adv_opt == 1`` the tendency is the PD-limited one and the
    update clamps FP32 rounding residuals at zero (module docstring).

    Boundary routing: stages 1-2 use the open-aware ``flux_div_scalar``
    path (Task 11 prerequisite) like theta.  Under SPECIFIED BCs the
    final PD stage runs with WRF's specified bounds in pd_advection.cu
    (advect_scalar_pd, module_advect_em.F:7697-7715) -- the production
    real74 path.  Radiative-OPEN domains keep the unlimited final stage
    plus clamp: WRF's advect_scalar_pd DOES support open BCs (it carries
    its own non-cb open radiation blocks, module_advect_em.F:7266-7330),
    but gpuwm's PD kernels do not implement those blocks -- an accepted,
    documented gpuwm deviation.  On the PD final stage the
    accumulated stage-1 forward tendencies, physics tendencies, AND qv's
    held lateral spec+relax tendency are folded into the time-t scalar
    BEFORE the limiter (WRF
    rk_update_scalar_pd; ``_pd_fold_sources``) so the renormalization
    budget sees the sources.  The held qv LBC tendency is captured once
    per model step on the first stage from the time-t state (WRF
    moist_tend, solve_em.F:2255-2292) and enters every stage's update --
    relax rows additively, spec rows replacing the advective tendency;
    the spec rows are subsequently overwritten by
    ``apply_state_boundary_values`` exactly as in WRF.
    """
    nz, ny, nx = state.p.shape
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]
    mu0 = state.mub2d + state.mup0                     # time-t column mass
    mu = state.mub2d + state.mup                       # post-acoustic mass
    chm0 = c1h * mu0[None] + c2h
    chm = c1h * mu[None] + c2h
    tend = state.scratch((nz, ny, nx), "moist_rq_t")
    boundary_forced = cfg.specified or cfg.nested
    boundary_x = cfg.open_x or boundary_forced
    boundary_y = cfg.open_y or boundary_forced
    # WRF advect_scalar_pd fully supports specified domains (dedicated
    # limiter bounds, module_advect_em.F:7697-7715), and the ratified
    # real74 namelist runs moist_adv_opt=1 -- so the PD final stage is
    # ENABLED under specified BCs (the pre-fix routing forced the
    # unlimited-plus-clamp path there, manufacturing scalar mass).
    # Radiative-open domains keep the non-PD routing.  WRF's
    # advect_scalar_pd DOES support open BCs -- it carries its own non-cb
    # open radiation blocks (module_advect_em.F:7266-7330) -- but gpuwm's
    # PD kernels do not implement those blocks, so open domains run the
    # unlimited final stage plus clamp instead: an accepted, documented
    # gpuwm deviation (open-BC benchmark pins).
    pd = (final and cfg.moist_adv_opt == 1
          and not (cfg.open_x or cfg.open_y))
    # WRF captures qv's lateral spec+relax tendency ONCE per model step,
    # on RK step 1 from the time-t scalar (relax_bdy_scalar +
    # spec_bdy_scalar into moist_tend, solve_em.F:2255-2292, gated
    # im == P_QV), and moist_tend persists into EVERY stage's
    # rk_update_scalar and the final rk_update_scalar_pd fold
    # (module_em.F:1889-1893).  ``apply_relax`` marks the capture stage
    # (dycore passes istage == 0, where state.qv still holds time t).
    # Final-review MAJOR: the relax part previously reached only the
    # discarded stage-1 provisional estimate.
    held_lbc = {}
    apply_scalar_lbc = None
    if boundary_forced and state.lateral_boundaries is not None:
        from gpuwm.ingest.lateral_bc import (
            apply_state_scalar_lateral_boundary as apply_scalar_lbc,
        )
        if cfg.specified:
            # Preserve d01's frozen qv held-slot operation order exactly.
            held = state.scratch((nz, ny, nx), "lbc_qv_held")
            held_lbc["qv"] = held
            if apply_relax:
                held[...] = 0
                apply_scalar_lbc(state, cfg, "qv", held, apply_relax=True)
    if pd:
        bufs = (state.scratch((nz, ny, nx + 1), "pd_fxl"),
                state.scratch((nz, ny, nx + 1), "pd_fxc"),
                state.scratch((nz, ny + 1, nx), "pd_fyl"),
                state.scratch((nz, ny + 1, nx), "pd_fyc"),
                state.scratch((nz + 1, ny, nx), "pd_fzl"),
                state.scratch((nz + 1, ny, nx), "pd_fzc"))
    for name in SPECIES:
        q = getattr(state, name)
        q0 = getattr(state, name + "0")
        tend[...] = 0
        if pd:
            nested_held = None
            if cfg.nested and apply_scalar_lbc is not None:
                apply_scalar_lbc(state, cfg, name, tend, apply_relax=True,
                                 source_field=q0)
                nested_held = tend
            q0_eff = _pd_fold_sources(
                state, cfg, name, q0, chm0, dt_eff, physics_tendencies,
                fixed_tendency=(fixed_tendencies.get(name)
                                if fixed_tendencies is not None else None),
                lbc_held=(nested_held if nested_held is not None
                          else held_lbc.get(name)))
            # WRF module_em.F:1889-1894 folds sc_tend into the time-t
            # scalar and then clears it before advect_scalar_pd accumulates
            # the advective tendency.  Nested forcing aliases ``tend`` here,
            # so retaining it would consume B once in q0_eff and again as
            # msft*B in the final update.
            tend[...] = 0
            launch_pd_fluxes(q, q0_eff, ru, rv, ww, mu, state,
                             cfg.dx, cfg.dy, dt_eff, *bufs,
                             msft=state.msft, has_msf=state.has_msf,
                             open_x=boundary_x, open_y=boundary_y)
            launch_pd_renorm_apply(q0_eff, mu0, *bufs, tend=tend,
                                   coord=state,
                                   dx=cfg.dx, dy=cfg.dy, dt=dt_eff,
                                   msft=state.msft, has_msf=state.has_msf,
                                   open_x=boundary_x, open_y=boundary_y)
            if state.has_msf:                 # WRF rk_update_scalar:
                tend *= state.msft[None]      # tendency = advect_tend*msfty
            q[...] = cp.maximum((chm0 * q0_eff + dt_eff * tend) / chm, 0.0)
        else:
            launch_flux_div_scalar(q, ru, rv, ww, tend, state,
                                   cfg.dx, cfg.dy,
                                   open_x=boundary_x, open_y=boundary_y,
                                   msf=state.msft, has_msf=state.has_msf,
                                   spec=boundary_forced)
            if state.has_msf:                 # WRF rk_update_scalar:
                tend *= state.msft[None]      # tendency = advect_tend*msfty
            if cfg.nested and apply_scalar_lbc is not None:
                apply_scalar_lbc(state, cfg, name, tend, apply_relax=True,
                                 source_field=q0)
            held = held_lbc.get(name)
            if held is not None:
                # The held spec+relax tendency enters EVERY stage (WRF
                # rk_update_scalar adds moist_tend on every rk step over
                # the _spc bounds).  Relax rows add; spec rows REPLACE
                # the advective tendency (spec_bdy_scalar semantics --
                # the capture wrote the pure boundary tendency there).
                sz = cfg.spec_zone
                tend += held
                tend[:, :sz, :] = held[:, :sz, :]
                tend[:, ny - sz:, :] = held[:, ny - sz:, :]
                tend[:, sz:ny - sz, :sz] = held[:, sz:ny - sz, :sz]
                tend[:, sz:ny - sz, nx - sz:] = (
                    held[:, sz:ny - sz, nx - sz:])
            if physics_tendencies is not None:
                physics = physics_tendencies.scalar_for(name)
                if physics is not None:
                    tend += physics
            if fixed_tendencies is not None:
                fixed = fixed_tendencies.get(name)
                if fixed is not None:
                    tend += fixed
            updated = (chm0 * q0 + dt_eff * tend) / chm
            q[...] = cp.maximum(updated, 0.0) if final else updated

    if getattr(state, "qi", None) is not None:
        for name in extra_moist_species(state):
            q = getattr(state, name)
            q0 = getattr(state, name + "0")
            tend[...] = 0
            if pd:
                nested_held = None
                if cfg.nested and apply_scalar_lbc is not None:
                    apply_scalar_lbc(state, cfg, name, tend,
                                     apply_relax=True, source_field=q0)
                    nested_held = tend
                q0_eff = _pd_fold_sources(
                    state, cfg, name, q0, chm0, dt_eff,
                    physics_tendencies,
                    fixed_tendency=(fixed_tendencies.get(name)
                                    if fixed_tendencies is not None else None),
                    lbc_held=(nested_held if nested_held is not None
                              else held_lbc.get(name)))
                # See the qv/qc/qr loop above: WRF clears sc_tend after the
                # positive-definite source fold and before advection.
                tend[...] = 0
                launch_pd_fluxes(q, q0_eff, ru, rv, ww, mu, state,
                                 cfg.dx, cfg.dy, dt_eff, *bufs,
                                 msft=state.msft, has_msf=state.has_msf,
                                 open_x=boundary_x, open_y=boundary_y)
                launch_pd_renorm_apply(q0_eff, mu0, *bufs, tend=tend,
                                       coord=state,
                                       dx=cfg.dx, dy=cfg.dy, dt=dt_eff,
                                       msft=state.msft,
                                       has_msf=state.has_msf,
                                       open_x=boundary_x, open_y=boundary_y)
                if state.has_msf:
                    tend *= state.msft[None]
                q[...] = cp.maximum((chm0 * q0_eff + dt_eff * tend) / chm,
                                    0.0)
            else:
                launch_flux_div_scalar(q, ru, rv, ww, tend, state,
                                       cfg.dx, cfg.dy,
                                       open_x=boundary_x, open_y=boundary_y,
                                       msf=state.msft,
                                       has_msf=state.has_msf,
                                       spec=boundary_forced)
                if state.has_msf:
                    tend *= state.msft[None]
                if cfg.nested and apply_scalar_lbc is not None:
                    apply_scalar_lbc(state, cfg, name, tend,
                                     apply_relax=True, source_field=q0)
                held = held_lbc.get(name)
                if held is not None:
                    sz = cfg.spec_zone
                    tend += held
                    tend[:, :sz, :] = held[:, :sz, :]
                    tend[:, ny - sz:, :] = held[:, ny - sz:, :]
                    tend[:, sz:ny - sz, :sz] = held[:, sz:ny - sz, :sz]
                    tend[:, sz:ny - sz, nx - sz:] = (
                        held[:, sz:ny - sz, nx - sz:])
                if physics_tendencies is not None:
                    physics = physics_tendencies.scalar_for(name)
                    if physics is not None:
                        tend += physics
                if fixed_tendencies is not None:
                    fixed = fixed_tendencies.get(name)
                    if fixed is not None:
                        tend += fixed
                updated = (chm0 * q0 + dt_eff * tend) / chm
                q[...] = cp.maximum(updated, 0.0) if final else updated
    if cfg.specified:
        from gpuwm.ingest.lateral_bc import apply_flow_dependent_boundaries
        fields = tuple(getattr(state, name) for name in moist_species(state)
                       if name != "qv")
        # The fused CUDA ABI carries nine independent fields per launch.
        # NSSL's full option-18 scalar package is larger, so submit adjacent
        # Registry-order batches; each field operation is independent.
        for start in range(0, len(fields), 9):
            apply_flow_dependent_boundaries(
                fields[start:start + 9], ru, rv, cfg.spec_zone)


def init_moist_balanced(cfg: RunConfig, coord: VerticalCoord,
                        base: BaseState, qv_func,
                        thp_func=None) -> DomainState:
    """Discretely balanced moist at-rest state (+ optional theta bubble).

    Transcribed from WRF v4.6.1 ``module_initialize_ideal.F`` (the
    quarter_ss full-state construction, lines 1026-1063): with the DRY base
    state and mu' = 0 (vapor rides as a scalar on the unchanged dry-mass
    column), integrate the perturbation pressure DOWN the column by
    inverting the discrete moist w-equation buoyancy (``pg_buoy_w`` == 0
    exactly, row by row), invert the moist EOS ``alt = (Rd/p0) * theta *
    (1 + Rv/Rd*qv) * (p/p0)^(-cv/cp)`` for the dry specific volume, then
    integrate the perturbation geopotential UP with the discrete
    hydrostatic recurrence.  The theta perturbation does not enter the
    pressure construction (WRF adds its bubble without recomputing p), so
    one builder serves the at-rest and bubble cases; ``qv_func == 0``
    reduces to the dry rebalance exactly.

    ``qv_func(z) -> (nz,) or (nz, ny, nx)`` maps the base-state half-level
    heights to vapor mixing ratio; ``thp_func(x, z)`` is the
    ``init_theta_perturbation`` contract.  Flat base states only (the
    terrain moist init is a later-task concern); float64 throughout;
    qc = qr = 0.
    """
    from gpuwm.core.diagnostics import update_diagnostics

    if not cfg.moist:
        raise ValueError("init_moist_balanced requires cfg.moist=True "
                         "(the state carries no qv arrays otherwise)")
    if base.terrain_z is not None:
        raise NotImplementedError(
            "init_moist_balanced supports flat base states only")
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    s = init_at_rest(cfg, coord, base)

    z = s.height_half()                                # (nz,)
    x = (np.arange(nx) + 0.5) * cfg.dx - 0.5 * nx * cfg.dx
    qv = np.asarray(qv_func(z), dtype=np.float64)
    qv3 = np.broadcast_to(qv[:, None, None] if qv.ndim == 1 else qv,
                          (nz, ny, nx))
    thp = (np.zeros((nz, ny, nx))
           if thp_func is None else np.asarray(thp_func(x, z), np.float64))
    th = np.asarray(base.thb, dtype=np.float64)[:, None, None] + thp

    mub = float(base.mub)
    mup = 0.0                                          # dry mass unchanged
    c1f, c2f = coord.c1f, coord.c2f
    c1h, c2h = coord.c1h, coord.c2h

    # Perturbation pressure down from the top: each row inverts the
    # discrete pg_buoy_w balance at the w level above (WRF qvf1/qvf2).
    p = np.zeros((nz, ny, nx))
    qvf1 = qv3[nz - 1]
    qvf2 = 1.0 / (1.0 + qvf1)
    qvf1 = qvf1 * qvf2
    p[nz - 1] = (-0.5 * (c1f[nz] * mup + qvf1 * (c1f[nz] * mub + c2f[nz]))
                 / coord.rdnw[nz - 1] / qvf2)
    for kk in range(nz - 2, -1, -1):
        kw = kk + 1                                    # w level above cell kk
        qvf1 = 0.5 * (qv3[kk] + qv3[kk + 1])
        qvf2 = 1.0 / (1.0 + qvf1)
        qvf1 = qvf1 * qvf2
        p[kk] = p[kk + 1] - (c1f[kw] * mup
                             + qvf1 * (c1f[kw] * mub + c2f[kw])) \
            / qvf2 / coord.rdn[kw]

    # Moist EOS inverted for the dry specific volume; then the discrete
    # hydrostatic recurrence for the perturbation geopotential.
    qvf = 1.0 + c.RVOVRD * qv3
    pb3 = np.asarray(base.pb, dtype=np.float64)[:, None, None]
    alt = (c.RD / c.P0) * th * qvf * ((p + pb3) / c.P0) ** (-c.CV / c.CP)
    al = alt - np.asarray(base.alb, dtype=np.float64)[:, None, None]
    php = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        php[k + 1] = php[k] - coord.dnw[k] * (
            ((c1h[k] * mub + c2h[k]) + c1h[k] * mup) * al[k]
            + (c1h[k] * mup) * base.alb[k])

    s.thp[...] = cp.asarray(thp, dtype=DTYPE)
    s.php[...] = cp.asarray(php, dtype=DTYPE)
    s.qv[...] = cp.asarray(qv3, dtype=DTYPE)
    # Idealized cases always use hypsometric_opt=1, regardless of namelist
    # setting (WRF share/input_wrf.F:1038); the php recurrence above is the
    # opt-1 discrete hydrostatic inversion, so the default key matches.
    update_diagnostics(s)
    return s
