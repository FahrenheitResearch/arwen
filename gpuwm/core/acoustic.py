"""Launchers for the split-explicit acoustic substeps (ARW sec. 3.1.2).

``acoustic_substep`` drives one complete substep: the explicit forward
steps of the perturbation horizontal momenta (``advance_uv``) and of
mu''/Omega''/(mu*theta)'' (``advance_mu_th``), the vertically implicit
Crank-Nicolson w''-phi'' solve (``calc_coefs`` + ``advance_w_phi``, which
also applies the terrain kinematic lower BC and the damp_opt=3 implicit
w damper), with the linearized-EOS p''/alpha'' diagnosis fused into the
column solve and specified-frame handling.
All kernels take the general hybrid/terrain arguments (2-D ``mub2d``,
c1/c2 coefficient arrays, 1-D or 3-D base profiles via ``base3d``).
"""

from __future__ import annotations

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE, DomainState

_THREADS = 256


def _base3d(state: DomainState) -> np.int32:
    """1 when the base profiles are per-column 3-D fields (terrain)."""
    return np.int32(state.thb.ndim == 3)


def _boundary_forced(cfg: RunConfig) -> bool:
    """True for specified or parent-forced nested lateral frames."""
    return bool(getattr(cfg, "specified", False)
                or getattr(cfg, "nested", False))


def _boundary_x(cfg: RunConfig) -> bool:
    return cfg.open_x or _boundary_forced(cfg)


def _boundary_y(cfg: RunConfig) -> bool:
    return cfg.open_y or _boundary_forced(cfg)


def _spec_zone(cfg: RunConfig) -> int:
    return int(cfg.spec_zone) if _boundary_forced(cfg) else 0


def _mass_w_boundary_zone(cfg: RunConfig) -> int:
    """WRF advance_mu_t/advance_w omit only the physical outer row.

    ``advance_uv`` deliberately narrows by the full configured specified
    zone, but WRF's mass/theta and implicit w/phi loops use ids+1/ide-2 and
    jds+1/jde-2 regardless of ``spec_zone``.  The wider frame is overwritten
    by ``spec_bdyupdate`` only after those provisional dynamics (and MUDF)
    have been formed.
    """
    return 1 if _boundary_forced(cfg) else 0


def prepare_moist_cq(state: DomainState, cfg: RunConfig) -> tuple:
    """Build WRF ``cqu/cqv/cqw`` once from the RK-stage moisture state.

    WRF ``calc_cq`` sums the active ``moist`` Registry package selected by
    ``mp_physics``: qv only for 0, qv/qc/qr for Kessler, and additionally
    qi/qs/qg for Morrison.  Number moments live in WRF's separate ``scalar``
    registry and are not mass loading.  The returned face arrays are
    scratch-arena views.  A dry state or the falsification-only
    ``moist_cq=False`` path returns harmless existing pointers plus a false
    flag, allocating and launching nothing.
    """
    use_cq = bool(getattr(cfg, "moist_cq", True)
                  and getattr(state, "qv", None) is not None)
    if not use_cq:
        return state.p, state.p, state.p, False

    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    cqu = state.scratch((nz, ny, nx + 1), "acoustic_cqu")
    cqv = state.scratch((nz, ny + 1, nx), "acoustic_cqv")
    cqw = state.scratch((nz + 1, ny, nx), "acoustic_cqw")
    if cfg.mp_physics == 0:
        qi = qs = qg = state.qv              # placeholders are never read
        n_mass = 1
    elif cfg.mp_physics == 1:
        qi = qs = qg = state.qv              # ice placeholders are never read
        n_mass = 3
    elif cfg.mp_physics in (6, 8, 10, 18, 28):
        # mp=28 is numerically IDENTICAL to mp=8 here.  WRF's calc_cq sums
        # the Registry ``moist`` package only, and Registry.EM_COMMON:3036
        # gives aerosol-aware Thompson
        #     moist:qv,qc,qr,qi,qs,qg
        #     scalar:qni,qnr,qnc,qnwfa,qnifa,qnbca
        # -- six masses, exactly mp=8's, with every number moment in the
        # separate ``scalar`` package.  They must therefore stay out of cq
        # AND out of the w-equation buoyancy loading; a droplet number of
        # order 1e8 entering q_tot would be a catastrophic, not a subtle,
        # error.
        qi = state.qi
        qs, qg = state.qs, state.qg
        n_mass = 7 if cfg.mp_physics == 18 else 6
    else:
        raise ValueError(f"unsupported mp_physics={cfg.mp_physics} for cq")
    qh = state.qh if cfg.mp_physics == 18 else state.qv
    n = (nz + 1) * (ny + 1) * (nx + 1)
    blocks = (n + _THREADS - 1) // _THREADS
    get_kernel("acoustic", "calc_cq")(
        (blocks,), (_THREADS,),
        (state.qv, state.qc, state.qr, qi, qs, qg, qh, cqu, cqv, cqw,
         np.int32(n_mass), np.int32(nz), np.int32(ny), np.int32(nx)))
    return cqu, cqv, cqw, True


def acoustic_substep_explicit(state: DomainState, cfg: RunConfig,
                              dtau: float, first: bool, cq=None,
                              mudf=None) -> None:
    """Explicit part of one acoustic substep of length ``dtau``.

    Advances ``u_pp``/``v_pp`` with the damped pressure gradient (the
    gradient sees ``p_pp + smdiv*(p_pp - p_pp_old)``; on the ``first``
    substep there is no history and the damping is dropped), saves
    the pre-update ``p_pp``/``mu_pp``/``th_pp`` histories while traversing
    those fields, then advances ``mu_pp``, ``ww_pp`` and ``th_pp`` from the
    updated momenta.

    Open lateral boundaries (Task 9): with ``cfg.open_x``/``open_y`` the
    boundary-normal momentum at the two boundary faces skips the pressure
    gradient and advances by the large-step tendency alone — WRF
    ``advance_uv``'s ``i_start_up``/``j_start_vp`` loop-bound exclusions
    (module_small_step_em.F).  The kernel computes the periodic update
    everywhere (the seam gradient is meaningless there) and the boundary
    faces are then rebuilt from their saved pre-substep values plus
    ``dtau * ru_t`` (the radiative tendency installed by
    ``dycore.apply_open_radiative_bc``).
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    smdiv = 0.0 if first else cfg.smdiv
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    cqu, cqv, _cqw, use_cq = cq
    mudf_arg = state.mup if mudf is None else mudf
    write_mudf = np.int32(mudf is not None)
    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")

    radiative_x = cfg.open_x and not _boundary_forced(cfg)
    radiative_y = cfg.open_y and not _boundary_forced(cfg)
    if radiative_x:
        sx = state.scratch((nz, ny, 2), "openbc_upp_faces")
        sx[..., 0] = state.u_pp[:, :, 0]
        sx[..., 1] = state.u_pp[:, :, -1]
    if radiative_y:
        sy = state.scratch((nz, 2, nx), "openbc_vpp_faces")
        sy[:, 0, :] = state.v_pp[:, 0, :]
        sy[:, 1, :] = state.v_pp[:, -1, :]

    kernel = get_kernel("acoustic", "advance_uv")
    n = nz * (ny + 1) * (nx + 1)
    blocks = (n + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (state.u_pp, state.v_pp, state.ru_t, state.rv_t,
            state.p_pp, state.p_pp_old, state.ph_pp, state.php, state.phb,
            state.alt, state.al_pp, state.pb,
             state.mup, state.mu_pp, state.mub2d,
             state.c1h, state.c2h, state.fnm, state.fnp, state.rdnw,
             cqu, cqv, np.int32(use_cq),
             state.cf1, state.cf2, state.cf3,
            np.int32(cfg.top_lid),
            DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy),
            DTYPE(dtau), DTYPE(smdiv),
            np.int32(_spec_zone(cfg)), _base3d(state),
            np.int32(nz), np.int32(ny), np.int32(nx)))

    if radiative_x:
        state.u_pp[:, :, 0] = sx[..., 0] + DTYPE(dtau) * state.ru_t[:, :, 0]
        state.u_pp[:, :, -1] = (sx[..., 1]
                                + DTYPE(dtau) * state.ru_t[:, :, -1])
    if radiative_y:
        state.v_pp[:, 0, :] = sy[:, 0, :] + DTYPE(dtau) * state.rv_t[:, 0, :]
        state.v_pp[:, -1, :] = (sy[:, 1, :]
                                + DTYPE(dtau) * state.rv_t[:, -1, :])
    # Open boundaries: the kernel's cross-boundary neighbour reads clamp
    # (WRF's zero-gradient ghosts) instead of wrapping.  Map factors
    # (Task 3) select the _msf kernel variant — the msf==1 kernel stays
    # byte-identical to Phase 2 so its codegen cannot drift off the pinned
    # bitwise regression.
    blocks = (ny * nx + _THREADS - 1) // _THREADS
    if state.has_msf:
        kernel = get_kernel("acoustic", "advance_mu_th_msf")
        kernel((blocks,), (_THREADS,),
               (state.u_pp, state.v_pp, state.u, state.v,
                 state.mup, state.mu_pp, mu_old, state.rmu_t,
                 mudf_arg, write_mudf,
                 state.thp, state.thb, state.th_pp, th_old, state.rth_t,
                 state.ww_pp, state.p_pp, state.p_pp_old,
                 state.dnw, state.rdnw, state.fnm, state.fnp,
                state.c1h, state.c2h, state.mub2d,
                state.msft, state.msfu, state.msfv,
                DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy), DTYPE(dtau),
                _base3d(state), np.int32(nz), np.int32(ny), np.int32(nx),
                np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
                np.int32(_mass_w_boundary_zone(cfg))))
    else:
        kernel = get_kernel("acoustic", "advance_mu_th")
        kernel((blocks,), (_THREADS,),
               (state.u_pp, state.v_pp, state.u, state.v,
                 state.mup, state.mu_pp, mu_old, state.rmu_t,
                 mudf_arg, write_mudf,
                 state.thp, state.thb, state.th_pp, th_old, state.rth_t,
                 state.ww_pp, state.p_pp, state.p_pp_old,
                 state.dnw, state.rdnw, state.fnm, state.fnp,
                state.c1h, state.c2h, state.mub2d,
                DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy), DTYPE(dtau),
                _base3d(state), np.int32(nz), np.int32(ny), np.int32(nx),
                np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
                np.int32(_mass_w_boundary_zone(cfg))))


#: Max full levels of the in-thread tridiagonal solve (acoustic.cu
#: WPHI_MAX_LEV; plan constraint nz <= 128).
_MAX_LEV = 129


def prepare_acoustic_coefficients(state: DomainState, cfg: RunConfig,
                                  dtau: float, cq=None) -> tuple:
    """Prepare the fixed-reference implicit coefficients for one RK stage.

    ``p``, ``alt`` and ``mup`` are the stage-reference diagnostics/state;
    acoustic substeps update only the perturbation ``*_pp`` fields.  The
    returned persistent buffers therefore remain valid until the stage is
    folded back by ``dycore._finish_small_steps``.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    if nz + 1 > _MAX_LEV:
        raise ValueError(f"nz={nz} exceeds the in-thread solve limit "
                         f"({_MAX_LEV - 1} half levels)")
    c2a = state.scratch((nz, ny, nx), "acoustic_c2a")
    a = state.scratch((nz + 1, ny, nx), "acoustic_a")
    alpha = state.scratch((nz + 1, ny, nx), "acoustic_alpha")
    gam = state.scratch((nz + 1, ny, nx), "acoustic_gamma")
    if cq is None:
        cq = prepare_moist_cq(state, cfg)
    cqu, cqv, cqw, use_cq = cq
    blocks = (ny * nx + _THREADS - 1) // _THREADS
    kernel = get_kernel("acoustic", "calc_coefs")
    kernel((blocks,), (_THREADS,),
           (state.p, state.alt, state.mup, c2a, a, alpha, gam,
            state.rdn, state.rdnw,
            state.c1h, state.c2h, state.c1f, state.c2f, state.mub2d,
            cqw, np.int32(use_cq),
            DTYPE(dtau), DTYPE(cfg.epssm), np.int32(cfg.top_lid),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    base = (c2a, a, alpha, gam)
    return base + cq if use_cq else base


def prepare_acoustic_substep_launch(state: DomainState, cfg: RunConfig,
                                    dtau: float, coefficients: tuple, *,
                                    mudf=None):
    """Bind one stage's invariant acoustic launch metadata once.

    The returned callable submits the same raw kernels, grids, blocks and
    argument values as :func:`acoustic_substep`.  Device arrays are retained
    by reference and continue to expose their in-place updates; only the
    immutable Python launch containers and scalar wrappers are reused.
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    if nz + 1 > _MAX_LEV:
        raise ValueError(f"nz={nz} exceeds the in-thread solve limit "
                         f"({_MAX_LEV - 1} half levels)")

    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")
    c2a, a, alpha, gam = coefficients[:4]
    if len(coefficients) == 4:
        cq = (state.p, state.p, state.p, False)
    else:
        cq = coefficients[4:]
    cqu, cqv, cqw, use_cq = cq
    mudf_arg = state.mup if mudf is None else mudf
    write_mudf = np.int32(mudf is not None)

    block = (_THREADS,)
    uv_n = nz * (ny + 1) * (nx + 1)
    uv_grid = ((uv_n + _THREADS - 1) // _THREADS,)
    column_grid = ((ny * nx + _THREADS - 1) // _THREADS,)
    rdx = DTYPE(1.0 / cfg.dx)
    rdy = DTYPE(1.0 / cfg.dy)
    dtau_arg = DTYPE(dtau)
    base3d = _base3d(state)
    nz_arg, ny_arg, nx_arg = np.int32(nz), np.int32(ny), np.int32(nx)
    spec_zone = np.int32(_spec_zone(cfg))
    mass_w_zone = np.int32(_mass_w_boundary_zone(cfg))
    boundary_x = np.int32(_boundary_x(cfg))
    boundary_y = np.int32(_boundary_y(cfg))

    uv_kernel = get_kernel("acoustic", "advance_uv")
    uv_prefix = (
        state.u_pp, state.v_pp, state.ru_t, state.rv_t,
        state.p_pp, state.p_pp_old, state.ph_pp, state.php, state.phb,
        state.alt, state.al_pp, state.pb,
        state.mup, state.mu_pp, state.mub2d,
        state.c1h, state.c2h, state.fnm, state.fnp, state.rdnw,
        cqu, cqv, np.int32(use_cq),
        state.cf1, state.cf2, state.cf3, np.int32(cfg.top_lid),
        rdx, rdy, dtau_arg,
    )
    uv_suffix = (spec_zone, base3d, nz_arg, ny_arg, nx_arg)
    uv_first_args = uv_prefix + (DTYPE(0.0),) + uv_suffix
    uv_later_args = uv_prefix + (DTYPE(cfg.smdiv),) + uv_suffix

    if state.has_msf:
        mu_name = "advance_mu_th_msf"
        mu_map_args = (state.msft, state.msfu, state.msfv)
    else:
        mu_name = "advance_mu_th"
        mu_map_args = ()
    mu_kernel = get_kernel("acoustic", mu_name)
    mu_args = (
        state.u_pp, state.v_pp, state.u, state.v,
        state.mup, state.mu_pp, mu_old, state.rmu_t,
        mudf_arg, write_mudf,
        state.thp, state.thb, state.th_pp, th_old, state.rth_t,
        state.ww_pp, state.p_pp, state.p_pp_old,
        state.dnw, state.rdnw, state.fnm, state.fnp,
        state.c1h, state.c2h, state.mub2d,
    ) + mu_map_args + (
        rdx, rdy, dtau_arg, base3d, nz_arg, ny_arg, nx_arg,
        boundary_x, boundary_y, mass_w_zone,
    )

    dampmag = dtau * cfg.dampcoef if cfg.damp_opt == 3 else 0.0
    w_name = "advance_w_phi_msf" if state.has_msf else "advance_w_phi"
    w_kernel = get_kernel("acoustic", w_name)
    w_map_args = (state.msft,) if state.has_msf else ()
    w_args = (
        state.w_pp, state.ph_pp, state.rw_t, state.rph_t, state.ww_pp,
        state.mu_pp, mu_old, state.th_pp, th_old,
        state.thp, state.thb, state.php, state.phb, state.alt, c2a,
        a, alpha, gam, state.mup,
        state.u_pp, state.v_pp, state.w, state.ht,
        state.rdn, state.rdnw, state.fnm, state.fnp,
        state.c1h, state.c2h, state.c1f, state.c2f, state.mub2d,
        cqw, np.int32(use_cq),
    ) + w_map_args + (
        state.p_pp, state.al_pp, state.cf1, state.cf2, state.cf3,
        rdx, rdy, dtau_arg, DTYPE(cfg.epssm), DTYPE(dampmag),
        DTYPE(cfg.zdamp), boundary_x, boundary_y, mass_w_zone, base3d,
        np.int32(cfg.top_lid), nz_arg, ny_arg, nx_arg,
    )

    frame_kernel = None
    frame_args = None
    if cfg.specified:
        frame_kernel = get_kernel("acoustic", "advance_specified_phi_w")
        frame_args = (
            state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
            state.th_pp, state.mup, state.mu_pp, state.mub2d, state.rmu_t,
            state.php, state.thp, state.thb, state.alt, c2a, state.rdnw,
            state.c1h, state.c2h, state.c1f, state.c2f, dtau_arg,
            np.int32(cfg.spec_zone), base3d, nz_arg, ny_arg, nx_arg,
        )
    elif cfg.nested:
        frame_kernel = get_kernel("acoustic", "advance_nested_phi_w")
        frame_args = (
            state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
            state.rw_t, state.th_pp, state.mup, state.mu_pp, state.mub2d,
            state.rmu_t, state.php, state.thp, state.thb, state.alt, c2a,
            state.rdnw, state.c1h, state.c2h, state.c1f, state.c2f,
            dtau_arg, np.int32(cfg.spec_zone), base3d,
            nz_arg, ny_arg, nx_arg,
        )

    radiative_x = cfg.open_x and not _boundary_forced(cfg)
    radiative_y = cfg.open_y and not _boundary_forced(cfg)
    sx = (state.scratch((nz, ny, 2), "openbc_upp_faces")
          if radiative_x else None)
    sy = (state.scratch((nz, 2, nx), "openbc_vpp_faces")
          if radiative_y else None)

    def launch(*, first: bool) -> None:
        if radiative_x:
            sx[..., 0] = state.u_pp[:, :, 0]
            sx[..., 1] = state.u_pp[:, :, -1]
        if radiative_y:
            sy[:, 0, :] = state.v_pp[:, 0, :]
            sy[:, 1, :] = state.v_pp[:, -1, :]

        uv_kernel(uv_grid, block,
                  uv_first_args if first else uv_later_args)

        if radiative_x:
            state.u_pp[:, :, 0] = (sx[..., 0]
                                   + dtau_arg * state.ru_t[:, :, 0])
            state.u_pp[:, :, -1] = (sx[..., 1]
                                    + dtau_arg * state.ru_t[:, :, -1])
        if radiative_y:
            state.v_pp[:, 0, :] = (sy[:, 0, :]
                                   + dtau_arg * state.rv_t[:, 0, :])
            state.v_pp[:, -1, :] = (sy[:, 1, :]
                                    + dtau_arg * state.rv_t[:, -1, :])

        mu_kernel(column_grid, block, mu_args)
        w_kernel(column_grid, block, w_args)
        if frame_kernel is not None:
            frame_kernel(column_grid, block, frame_args)

    return launch


def acoustic_substep(state: DomainState, cfg: RunConfig,
                     dtau: float, first: bool, coefficients=None,
                     mudf=None) -> None:
    """One complete acoustic substep of length ``dtau``.

    Explicit part (Task 10), then the vertically implicit w''-phi'' solve
    off-centered by ``cfg.epssm`` (WRF ``calc_coef_w``/``advance_w``,
    kinematic terrain bottom BC, and WRF's configurable open/rigid top), then
    the linearized-EOS p''/alpha'' diagnosis.  The pre-substep
    mu''/(mu*theta)'' are saved first — the implicit solve's buoyancy needs
    their epssm averages (WRF ``muave``/``t_2ave``).
    """
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    if nz + 1 > _MAX_LEV:
        raise ValueError(f"nz={nz} exceeds the in-thread solve limit "
                         f"({_MAX_LEV - 1} half levels)")

    if coefficients is None:
        cq = prepare_moist_cq(state, cfg)
    elif len(coefficients) == 4:
        cq = (state.p, state.p, state.p, False)
    else:
        cq = coefficients[4:]
    acoustic_substep_explicit(state, cfg, dtau, first, cq=cq, mudf=mudf)
    mu_old = state.scratch((ny, nx), "acoustic_mu_pp_old")
    th_old = state.scratch((nz, ny, nx), "acoustic_th_pp_old")

    # Standalone callers may omit a prepared tuple; dycore prepares exactly
    # once per stage and supplies it to every substep.
    if coefficients is None:
        coefficients = prepare_acoustic_coefficients(state, cfg, dtau, cq=cq)
    c2a, a, alpha, gam = coefficients[:4]
    cqu, cqv, cqw, use_cq = cq
    blocks = (ny * nx + _THREADS - 1) // _THREADS

    dampmag = dtau * cfg.dampcoef if cfg.damp_opt == 3 else 0.0
    # Map factors (Task 3): the _msf variant carries WRF advance_w's msfty
    # factors; the msf==1 kernel stays byte-identical to Phase 2.
    name = "advance_w_phi_msf" if state.has_msf else "advance_w_phi"
    kernel = get_kernel("acoustic", name)
    args = [state.w_pp, state.ph_pp, state.rw_t, state.rph_t, state.ww_pp,
            state.mu_pp, mu_old, state.th_pp, th_old,
             state.thp, state.thb, state.php, state.phb, state.alt, c2a,
             a, alpha, gam, state.mup,
            state.u_pp, state.v_pp, state.w, state.ht,
            state.rdn, state.rdnw, state.fnm, state.fnp,
             state.c1h, state.c2h, state.c1f, state.c2f, state.mub2d,
             cqw, np.int32(use_cq)]
    if state.has_msf:
        args.append(state.msft)
    args += [state.p_pp, state.al_pp,
             state.cf1, state.cf2, state.cf3,
             DTYPE(1.0 / cfg.dx), DTYPE(1.0 / cfg.dy),
              DTYPE(dtau), DTYPE(cfg.epssm),
              DTYPE(dampmag), DTYPE(cfg.zdamp),
              np.int32(_boundary_x(cfg)), np.int32(_boundary_y(cfg)),
              np.int32(_mass_w_boundary_zone(cfg)), _base3d(state),
              np.int32(cfg.top_lid),
              np.int32(nz), np.int32(ny), np.int32(nx)]
    kernel((blocks,), (_THREADS,), tuple(args))

    if cfg.specified:
        # The raw frame kernel preserves every eager FP32 round point in
        # spec_bdyupdate_ph and combines its independent zero-gradient w copy.
        n = ny * nx
        blocks = (n + _THREADS - 1) // _THREADS
        get_kernel("acoustic", "advance_specified_phi_w")(
            (blocks,), (_THREADS,),
            (state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
             state.th_pp, state.mup, state.mu_pp, state.mub2d, state.rmu_t,
             state.php, state.thp, state.thb, state.alt, c2a, state.rdnw,
             state.c1h, state.c2h, state.c1f, state.c2f, DTYPE(dtau),
             np.int32(cfg.spec_zone), _base3d(state), np.int32(nz),
             np.int32(ny), np.int32(nx)))
    elif cfg.nested:
        # solve_em.F:1577-1611 ELSE: spec_bdyupdate_ph followed by
        # spec_bdyupdate(w_2, rw_tend, dts_rk).  This child-only raw kernel
        # also repeats dyn's fused pressure diagnosis on the changed frame.
        n = ny * nx
        blocks = (n + _THREADS - 1) // _THREADS
        get_kernel("acoustic", "advance_nested_phi_w")(
            (blocks,), (_THREADS,),
            (state.ph_pp, state.w_pp, state.p_pp, state.al_pp, state.rph_t,
             state.rw_t, state.th_pp, state.mup, state.mu_pp, state.mub2d,
             state.rmu_t, state.php, state.thp, state.thb, state.alt, c2a,
             state.rdnw, state.c1h, state.c2h, state.c1f, state.c2f,
             DTYPE(dtau), np.int32(cfg.spec_zone), _base3d(state),
             np.int32(nz), np.int32(ny), np.int32(nx)))


def run_acoustic_only(state: DomainState, cfg: RunConfig,
                      dtau: float, n: int) -> None:
    """Drive ``n`` acoustic substeps around the frozen reference state.

    With zero large-step tendencies this integrates the pure (linearized)
    acoustic system — used by the sound-wave-speed test.
    """
    if n <= 0:
        return
    coefficients = prepare_acoustic_coefficients(state, cfg, dtau)
    launch_substep = prepare_acoustic_substep_launch(
        state, cfg, dtau, coefficients)
    for i in range(n):
        launch_substep(first=(i == 0))
