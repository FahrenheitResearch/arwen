"""Chem species transport: WRF-Chem's ``chem_scalar_advance`` (DESIGN 2.2-2.3).

WRF advances every chem species exactly like a moist scalar: ``rk_scalar_tend``
with ``chem_adv_opt`` on each RK stage and ``rk_update_scalar`` after it
(``dyn_em/solve_em.F:2466-2616``), the positive-definite source fold
``rk_update_scalar_pd`` before the final stage (:1981-2039), and the acoustic
time-averaged mass fluxes ``ru_m``/``rv_m``/``ww_m`` throughout.  So this module
calls the moist path's launchers UNCHANGED (``launch_flux_div_scalar``,
``launch_pd_fluxes``, ``launch_pd_renorm_apply``, the fold and the fused
coupled update from :mod:`gpuwm.core.moist`), species by species, and edits
nothing in that module: every non-chem trajectory keeps its bytes.

Where chem is NOT moisture, it follows WRF-Chem, not the moist path:

* No physics tendency.  Chem runs as its own operator after the step
  (:mod:`gpuwm.core.chem_driver`); nothing folds into it from the physics
  driver.  It never enters ``calc_cq``, buoyancy or ``q_total``.
* The held forward tendency is the chem array's own: 2nd-order horizontal
  mixing unless ``chem_mix2_off`` and the 6th-order filter unless
  ``chem_mix6_off`` (``module_diffusion_em.F:3054``, ``module_em.F:1421``),
  in the ``smag_rchem_<row>`` carrying buffers ``dycore`` fills.
* Outer (specified) domain: NO relax zone and NO spec tendency for chem
  (solve_em.F:2504-2508 says so); after every stage's update the spec zone is
  set by ``flow_dep_bdy_chem`` (solve_em.F:2599-2616): outflow copies the
  first inner row, inflow takes the row's boundary value.  See
  :func:`apply_chem_flow_boundaries`.
* Nested domain: relax plus spec on RK step 1 into the held tendency, the
  moist nested form (solve_em.F:2509-2545).
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.advection import launch_flux_div_scalar, vertical_orders
from gpuwm.core.chem_context import MONO_IMPLICIT_SLOT, MONO_SCRATCH_SLOTS
from gpuwm.core.moist import (
    _exclude_specified_ring_advection, _ieva_scalar, _pd_fold_sources,
    _update_scalar_in_place, launch_pd_fluxes, launch_pd_renorm_apply,
)

__all__ = ["MONO_IMPLICIT_SLOT", "MONO_SCRATCH_SLOTS", "advance_chem_stage",
           "apply_chem_flow_boundaries", "chem_fixed_slot",
           "chem_fixed_tendencies"]


def chem_fixed_slot(row) -> str:
    """The carrying-buffer slot of a chem row's held forward tendency."""
    return "smag_r" + row.state_attr


def chem_fixed_tendencies(state, cfg):
    """Held forward tendencies of the chem rows, by ``state_attr``, or None.

    The buffers ``dycore.prepare_fixed_tendencies`` fills once per step
    whenever a mixing operator (km_opt 2/3/4) or the 6th-order filter runs;
    ``chem_mix2_off``/``chem_mix6_off`` leave a row's buffer holding only
    the part WRF still applies (``dycore.chem_mix2_exempt_slots``,
    ``dycore.diff6_exempt_slots``).
    """
    if cfg.km_opt not in (2, 3, 4) and cfg.diff_6th_opt <= 0:
        return None
    shape = state.p.shape
    return {row.state_attr: state.scratch(shape, chem_fixed_slot(row))
            for row in state.chem.transported}


def advance_chem_stage(state, cfg, ru, rv, ww, dt_eff: float, final: bool,
                       fixed_tendencies=None, implicit=None) -> None:
    """Advance every transported chem row one RK stage from its time-t copy.

    Called by ``dycore.step`` right after the moist scalars, with the same
    stage fluxes ``ru_m``/``rv_m``/``ww_m`` and stage length.
    ``fixed_tendencies`` maps a row's ``state_attr`` to its held forward
    tendency (mixing and/or 6th-order filter), or is None.
    ``implicit`` is the same IEVA split the moist scalars consume: explicit
    fluxes use wwE, the monotonic budget also sees wwI, and the shared GPU
    column solve replaces each row's tendency before the coupled update.
    """
    chem = state.chem
    vorder = vertical_orders(cfg)[0]
    ww_explicit = ww if implicit is None else implicit[0]
    nz, ny, nx = state.p.shape
    mu0 = state.mub2d + state.mup0                      # time-t column mass
    mu = state.mub2d + state.mup                        # post-acoustic mass
    mu0_row = mu0.reshape(-1)
    mu_row = mu.reshape(-1)
    msft_row = state.msft.reshape(-1)
    tend = state.scratch((nz, ny, nx), "moist_rq_t")
    boundary_forced = cfg.specified or cfg.nested
    boundary_x = cfg.open_x or boundary_forced
    boundary_y = cfg.open_y or boundary_forced
    # chem_adv_opt = 1 (positive definite) on the final stage, exactly the
    # moist routing, including its documented open-boundary deviation (the
    # PD kernels carry no open radiation blocks, so an open domain takes the
    # unlimited final stage plus the clamp).
    pd = (final and cfg.chem_adv_opt == 1
          and not (cfg.open_x or cfg.open_y))
    # chem_adv_opt = 2: WRF's advect_scalar_mono on the final stage
    # (module_em.F:1282-1297), after the same source fold (solve_em.F:1981
    # folds for every chem_adv_opt but 0 and 3).  Its kernel carries WRF's
    # own open radiation blocks, so it runs on open domains too.
    mono = final and cfg.chem_adv_opt == 2
    chm0 = (state.c1h[:, None, None] * mu0[None] + state.c2h[:, None, None]
            if (pd or mono) else None)
    apply_scalar_lbc = None
    if cfg.nested and state.lateral_boundaries is not None:
        from gpuwm.ingest.lateral_bc import (
            apply_state_scalar_lateral_boundary as apply_scalar_lbc,
        )
    if pd or mono:
        bufs = (state.scratch((nz, ny, nx + 1), "pd_fxl"),
                state.scratch((nz, ny, nx + 1), "pd_fxc"),
                state.scratch((nz, ny + 1, nx), "pd_fyl"),
                state.scratch((nz, ny + 1, nx), "pd_fyc"),
                state.scratch((nz + 1, ny, nx), "pd_fzl"),
                state.scratch((nz + 1, ny, nx), "pd_fzc"))
    if mono:
        from gpuwm.core.chem_advect_mono import (
            launch_mono_fluxes, launch_mono_renorm_apply,
        )
        mono_scratch = tuple(state.scratch((nz, ny, nx), slot)
                             for slot in MONO_SCRATCH_SLOTS)
        # The explicit-only route keeps its zero carrier; IEVA supplies the
        # same nonzero implicit share used by the GPU scalar column solve.
        ww_implicit = (state.scratch((nz + 1, ny, nx), MONO_IMPLICIT_SLOT)
                       if implicit is None else implicit[1])
        mono_boundary = "open" if (cfg.open_x or cfg.open_y) else "specified"
    msft = state.msft if state.has_msf else None
    for row in chem.transported:
        name = row.state_attr
        q = getattr(state, name)
        q0 = getattr(state, row.time_attr)
        fixed = (fixed_tendencies.get(name)
                 if fixed_tendencies is not None else None)
        if pd:
            nested_held = None
            if apply_scalar_lbc is not None:
                tend[...] = 0
                apply_scalar_lbc(state, cfg, name, tend, apply_relax=True,
                                 source_field=q0)
                nested_held = tend
            q0_eff = _pd_fold_sources(
                state, cfg, name, q0, chm0, dt_eff, None,
                fixed_tendency=fixed, lbc_held=nested_held)
            tend[...] = 0
            launch_pd_fluxes(q, q0_eff, ru, rv, ww_explicit, mu, state,
                             cfg.dx, cfg.dy, dt_eff, *bufs,
                             msft=state.msft, has_msf=state.has_msf,
                             open_x=boundary_x, open_y=boundary_y,
                             vorder=vorder)
            launch_pd_renorm_apply(q0_eff, mu0, *bufs, tend=tend,
                                   coord=state, dx=cfg.dx, dy=cfg.dy,
                                   dt=dt_eff, msft=state.msft,
                                   has_msf=state.has_msf,
                                   open_x=boundary_x, open_y=boundary_y)
            if implicit is not None:
                _ieva_scalar(state, tend, q0_eff, implicit, mu0, mu, dt_eff)
            if boundary_forced:
                _exclude_specified_ring_advection(tend, cfg.spec_zone)
            _update_scalar_in_place(
                q, q0_eff, tend, state.c1h, state.c2h, mu0_row, mu_row,
                dt_eff, msft=(msft_row if state.has_msf else None),
                clamp=True)
        elif mono:
            nested_held = None
            if apply_scalar_lbc is not None:
                tend[...] = 0
                apply_scalar_lbc(state, cfg, name, tend, apply_relax=True,
                                 source_field=q0)
                nested_held = tend
            q0_eff = _pd_fold_sources(
                state, cfg, name, q0, chm0, dt_eff, None,
                fixed_tendency=fixed, lbc_held=nested_held)
            tend[...] = 0
            # WRF's arguments: mut = muts (post-acoustic total mass), and
            # the old-mass budget ((c1*mub + c2) + c1*mu_1)*q0 from the
            # base and time-t perturbation masses in that order
            # (module_advect_em.F:10378-10408).
            work = launch_mono_fluxes(
                q, q0_eff, ru, rv, ww_explicit, mu, state, cfg.dx, cfg.dy, dt_eff,
                *bufs, msft=state.msft, has_msf=state.has_msf,
                open_x=boundary_x, open_y=boundary_y,
                rw_implicit=ww_implicit, boundary=mono_boundary,
                scratch=mono_scratch, v_order=vorder)
            launch_mono_renorm_apply(
                q0_eff, state.mup0, *bufs, tend, state, cfg.dx, cfg.dy,
                dt_eff, msft=state.msft, has_msf=state.has_msf,
                open_x=boundary_x, open_y=boundary_y, workspace=work,
                mub=state.mub2d)
            if implicit is not None:
                _ieva_scalar(state, tend, q0_eff, implicit, mu0, mu, dt_eff)
            if boundary_forced:
                _exclude_specified_ring_advection(tend, cfg.spec_zone)
            _update_scalar_in_place(
                q, q0_eff, tend, state.c1h, state.c2h, mu0_row, mu_row,
                dt_eff, msft=(msft_row if state.has_msf else None),
                clamp=True)
        else:
            tend[...] = 0
            launch_flux_div_scalar(q, ru, rv, ww_explicit, tend, state,
                                   cfg.dx, cfg.dy,
                                   open_x=boundary_x, open_y=boundary_y,
                                   msf=state.msft, has_msf=state.has_msf,
                                   spec=boundary_forced, vorder=vorder)
            if implicit is not None:
                _ieva_scalar(state, tend, q0, implicit, mu0, mu, dt_eff)
            lbc_after_msf = apply_scalar_lbc is not None
            if msft is not None and lbc_after_msf:
                tend *= msft[None]
            if lbc_after_msf:
                apply_scalar_lbc(state, cfg, name, tend, apply_relax=True,
                                 source_field=q0)
            _update_scalar_in_place(
                q, q0, tend, state.c1h, state.c2h, mu0_row, mu_row, dt_eff,
                msft=(msft_row if msft is not None and not lbc_after_msf
                      else None),
                physics=None, fixed=fixed, clamp=final)
    if cfg.specified:
        apply_chem_flow_boundaries(state, cfg, ru, rv, dt_eff)


def apply_chem_flow_boundaries(state, cfg, ru, rv, dt_rk=0.0) -> None:
    """WRF ``flow_dep_bdy_chem`` on the specified zone of every chem row.

    ``chem/module_input_chem_data.F:1531-2031``, called after every RK
    stage's ``rk_update_scalar`` (solve_em.F:2599-2616) with the acoustic
    time-averaged coupled mass fluxes (``grid%ru_m``/``grid%rv_m``), whose
    SIGN alone is read: outflow copies the first row inside the spec zone,
    inflow takes the row's boundary value.  One launch for every row
    (:func:`gpuwm.core.chem_bdy.apply_chem_flow_boundaries`, bit-identical
    to WRF's own loops on 24 oracle cases; its docstring maps WRF's (i, k, j)
    faces onto gpuwm's arrays).

    Without a boundary source this run (WRF ``have_bcs_chem`` false) the
    inflow value is the row's ``default_inflow`` -- ``bdy_chem_value_gocart``'s
    per-species constant (:1267-1298) or a tracer constant.  A row WITH a
    boundary source (WRF ``have_bcs_chem``, ``i_bdy_method = 6``) takes
    ``max(epsilc, chem_b + chem_bt*dt)`` from the OUTERMOST boundary row of
    this domain's sealed forcing, which carries the row's own mixing ratio
    on every forcing frame (the frames' chem fields, filled by
    :mod:`gpuwm.chem_source_init`), packed by
    :func:`gpuwm.ingest.lateral_bc.chem_boundary_rows`.  ``dt`` is WRF's
    ``dt_rk + grid%dtbc`` (solve_em.F:2604), both REAL, with ``dtbc`` the
    step's post-increment value the met tables use.
    """
    from gpuwm.core.chem_bdy import apply_chem_flow_boundaries as launch
    from gpuwm.chem_source_init import chem_boundary_fields
    from gpuwm.ingest.lateral_bc import chem_boundary_rows

    chem = state.chem
    rows = chem.transported
    if not rows or int(cfg.spec_zone) <= 0:
        return
    names = [row.state_attr for row in rows]
    supplied = chem_boundary_rows(state, cfg, names)
    expected = set(chem_boundary_fields(chem.table, cfg))
    if expected:
        carried = ({name for name, has in zip(names, supplied[0]) if has}
                   if supplied is not None else set())
        if not expected <= carried:
            raise RuntimeError(
                f"chem rows {sorted(expected - carried)} name an enabled "
                "boundary source, and this domain's lateral forcing carries "
                "no table for them, so they would take their default inflow "
                "instead of the source; re-prepare the forcing with this "
                "configuration")
    if supplied is None:
        launch(chem.fields(state), ru, rv, int(cfg.spec_zone),
               [0] * len(rows), [row.default_inflow for row in rows],
               None, None, None, None, None, None, None, None, 0.0)
        return
    has, tables, dtbc = supplied
    dt = np.float32(np.float32(dt_rk) + np.float32(dtbc))
    launch(chem.fields(state), ru, rv, int(cfg.spec_zone),
           has, [row.default_inflow for row in rows], *tables, dt)
