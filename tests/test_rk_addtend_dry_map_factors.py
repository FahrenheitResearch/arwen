# tests/test_rk_addtend_dry_map_factors.py
"""rk_addtend_dry's 1/msf belongs to the mixing package, not to diff6.

Oracle: WRF v4.6.1 ``dyn_em/module_em.F`` ``rk_addtend_dry``, which adds
each forward (time-t, held) tendency to that RK stage's slow tendency
after dividing it by the target's own map factor --

  :1043   ru_tend = ru_tend + ru_tendf/msfuy      ! "divide by my to couple u"
  :1054   rv_tend = rv_tend + rv_tendf*msfvx_inv  ! "divide by mx to couple v"
  :1065   rw_tend = rw_tend + rw_tendf/msfty      ! "divide by my to couple w"
  :1078    t_tend =  t_tend +  t_tendf/msfty      ! ... and theta likewise

Two source packages fill those held buffers and they do NOT agree on what
they put in:

* the Smagorinsky/vertical mixing package carries the map factor --
  ``horizontal_diffusion_u_2`` builds ``mrdx=msfux(i,j)*rdx``
  (module_diffusion_em.F:3304-3312), which ``wrf_smag_hd_u``
  (kernels/smag2d.cu:786-792) transcribes, and the vertical rows carry
  none in either code -- so the division above is the whole conversion
  into gpuwm's coupled ``ru_t``/``rv_t``/``rw_t``/``rth_t``;
* ``sixth_order_diffusion`` ALSO multiplies by the map factor
  (module_big_step_utilities_em.F:6509/:6522/:6531 for x and
  :6599/:6605/:6614 for y) and ``rk_addtend_dry`` divides it straight back
  out, so WRF's net diff6 contribution to the dry tendencies carries no
  map factor at all -- and kernels/diff6.cu omits both operations to the
  same end (its header: "map factors 1 on the tendency").

gpuwm shares ONE carrying buffer between the two packages
(``prepare_fixed_tendencies``), so the division is taken on the mixing
half alone.  Dividing the sum instead under-applies the dry 6th-order
filter by exactly the map factor; dividing neither leaves the mixing rows
msf times WRF's.  Both errors are measured here, and neither is visible
on an unmapped grid or with only one of the two packages switched on --
which is why every gate below runs the operators on a mapped grid and two
of them run both packages at once.

The moist rows take ``rk_update_scalar``'s ``msfty`` in gpuwm.core.moist
(module_em.F:1697-1709) and must be left uncoupled here; that asymmetry
is graded against the same authority as the theta row, so a division
sprayed over every scalar cannot pass.

GPU: both packages are kernel-resident, so grading the composition means
running them.  Everything is measured through ``prepare_fixed_tendencies``
+ ``add_fixed_dry_tendencies`` -- the production pair -- against the
float64 WRF authorities in gpuwm.core.smag2d and gpuwm.verify.npref, so
no assertion here restates the implementation's own premise.
``diff_6th_slopeopt`` stays 0 throughout, which keeps diff6's map-factor
arguments off the slope taper and leaves this file measuring only the
tendency-side convention.
"""
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

# (carrying slot, coupled tendency attribute, rk_addtend_dry's map factor)
_ROWS = (("smag_ru", "ru_t", "msfu"),
         ("smag_rv", "rv_t", "msfv"),
         ("smag_rw", "rw_t", "msft"),
         ("smag_rth", "rth_t", "msft"))

_NX, _NY, _NZ = 8, 7, 6
_BASE = dict(nx=_NX, ny=_NY, nz=_NZ, dx=900.0, dy=1100.0, ztop=9000.0,
             dt=2.0, run_seconds=0.0, terrain_opt=1, hybrid_opt=2,
             c_s=0.25, moist=True)


def _state(cfg, mapped=True):
    """A stirred 3-D-terrain state, optionally on a mapped C grid.

    Smooth analytic fields and the C-grid-consistent map factors of
    tests/test_smag2d.py's authority fixture, so the fp32 kernels and the
    fp64 authority agree to the tolerances used there.  The ``*0`` copies
    are set too: ``prepare_fixed_tendencies`` reads the saved time-t
    fields, as WRF's ``module_first_rk_step_part2`` does.
    """
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    coord = make_vertical_coord(nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    xx = 2.0 * np.pi * np.arange(nx) / nx
    yy = 2.0 * np.pi * np.arange(ny) / ny
    terrain = (260.0 + 110.0 * np.sin(xx)[None, :]
               + 75.0 * np.cos(yy)[:, None]
               + 35.0 * np.sin(yy[:, None] + xx[None, :]))
    base = make_base_state(
        coord, lambda z: 300.0 + 0.004 * np.asarray(z, dtype=np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=terrain)
    state = init_at_rest(cfg, coord, base, terrain_z=terrain)

    if mapped:
        # Distinct on both axes and mutually distinct across the three
        # staggerings, so a row that reached for the wrong map factor
        # cannot pass by numerical coincidence.
        msft = (1.08 + 0.055 * np.sin(xx)[None, :]
                + 0.035 * np.cos(yy)[:, None]
                + 0.018 * np.sin(2.0 * yy[:, None] + xx[None, :]))
        msfu_core = 0.5 * (msft + np.roll(msft, 1, axis=1))
        msfv_core = 0.5 * (msft + np.roll(msft, 1, axis=0))
        state.set_map_coriolis(
            msft=msft,
            msfu=np.concatenate([msfu_core, msfu_core[:, :1]], axis=1),
            msfv=np.concatenate([msfv_core, msfv_core[:1, :]], axis=0))
        assert state.has_msf
        assert np.ptp(msft) > 0.05 and not np.allclose(msft, 1.0)
    else:
        assert not state.has_msf

    xf = 2.0 * np.pi * np.arange(nx) / nx
    xm = 2.0 * np.pi * (np.arange(nx) + 0.5) / nx
    yf = 2.0 * np.pi * np.arange(ny) / ny
    ym = 2.0 * np.pi * (np.arange(ny) + 0.5) / ny
    zm = np.arange(nz, dtype=np.float64) / (nz - 1)
    zw = np.arange(nz + 1, dtype=np.float64) / nz
    u_core = (4.5 * np.sin(xf[None, None, :] + 0.35 * ym[None, :, None])
              + 2.2 * np.cos(ym[None, :, None])
              + 3.0 * zm[:, None, None]
                * (1.0 + 0.22 * np.cos(xf[None, None, :])))
    v_core = (-3.8 * np.cos(xm[None, None, :] - 0.30 * yf[None, :, None])
              + 2.7 * np.sin(yf[None, :, None] + 0.25 * xm[None, None, :])
              - 2.1 * zm[:, None, None]
                * (1.0 + 0.17 * np.sin(yf[None, :, None])))
    w = (np.sin(np.pi * zw)[:, None, None]
         * (1.4 * np.sin(xm[None, None, :] + 0.65 * ym[None, :, None])
            + 0.55 * np.cos(2.0 * xm[None, None, :] - ym[None, :, None])))
    qv = (0.018 + 0.004 * np.sin(xm)[None, None, :]
          + 0.003 * np.cos(ym)[None, :, None] + 0.002 * zm[:, None, None])
    thp = ((1.25 + 0.35 * zm[:, None, None])
           * np.sin(xm[None, None, :] + 0.45 * ym[None, :, None])
           + 0.55 * np.cos(2.0 * ym[None, :, None]
                           - 0.30 * xm[None, None, :]))
    fields = {
        "u": np.concatenate([u_core, u_core[:, :, :1]], axis=2),
        "v": np.concatenate([v_core, v_core[:, :1, :]], axis=1),
        "w": w, "qv": qv, "thp": thp,
    }
    for name, value in fields.items():
        live = getattr(state, name)
        live[...] = cp.asarray(value, dtype=cp.float32)
        saved = getattr(state, name + "0", None)
        if saved is not None:
            saved[...] = live
    update_diagnostics(state, cfg.hypsometric_opt)
    return state


def _delivered(state, cfg):
    """Run the production pair; return the four coupled dry tendencies.

    ``prepare_fixed_tendencies`` builds WRF's held ``*_tendf`` once, then
    ``add_fixed_dry_tendencies`` is WRF's ``rk_addtend_dry`` for one RK
    pass.  Reading the tendencies rather than the carrying buffers is what
    makes this file blind to WHERE the coupling is applied and sensitive
    only to whether the delivered numbers are WRF's.
    """
    import cupy as cp

    from gpuwm.core import dycore
    dycore.prepare_fixed_tendencies(state, cfg)
    for _slot, tend, _msf in _ROWS:
        getattr(state, tend)[...] = 0
    dycore.add_fixed_dry_tendencies(state, cfg)
    return {tend: cp.asnumpy(getattr(state, tend)).astype(np.float64)
            for _slot, tend, _msf in _ROWS}


def _authorities(state, cfg):
    """Float64 WRF mixing authorities for this state (momentum + scalar)."""
    import cupy as cp

    from gpuwm.core.smag2d import (wrf_periodic_momentum_authority,
                                   wrf_periodic_scalar_metric_tendency)
    phb = cp.asnumpy(state.phb).astype(np.float64)
    if phb.ndim == 1:
        phb = phb[:, None, None]
    phi = cp.asnumpy(state.php).astype(np.float64) + phb
    alt_h = cp.asnumpy(state.alt).astype(np.float64)
    rho = (1.0 + cp.asnumpy(state.qv).astype(np.float64)) / alt_h
    grid = dict(msft=cp.asnumpy(state.msft), msfu=cp.asnumpy(state.msfu),
                msfv=cp.asnumpy(state.msfv), fnm=cp.asnumpy(state.fnm),
                fnp=cp.asnumpy(state.fnp), dnw=cp.asnumpy(state.dnw),
                cf1=float(state.cf1), cf2=float(state.cf2),
                cf3=float(state.cf3), dx=cfg.dx, dy=cfg.dy)
    momentum = wrf_periodic_momentum_authority(
        u=cp.asnumpy(state.u), v=cp.asnumpy(state.v),
        w=cp.asnumpy(state.w), phi=phi, rho=rho,
        dn=cp.asnumpy(state.dn), c_s=cfg.c_s, **grid)
    thb_h = cp.asnumpy(state.thb).astype(np.float64)
    scalar = dict(kh=momentum["kh"], rho=rho, phi=phi,
                  dn=cp.asnumpy(state.dn), **grid)
    theta = wrf_periodic_scalar_metric_tendency(
        field=cp.asnumpy(state.thp).astype(np.float64) + thb_h - 300.0,
        **scalar)
    qv = wrf_periodic_scalar_metric_tendency(
        field=cp.asnumpy(state.qv).astype(np.float64), **scalar)
    return momentum, theta, qv


@requires_gpu
def test_the_mixing_rows_take_rk_addtend_drys_map_factor():
    """km_opt=4 alone: delivered tendency == mixing authority / msf.

    The authority (gpuwm.core.smag2d.wrf_periodic_momentum_authority, a
    host reduction of ``cal_deform_and_div`` + ``smag2d_km`` +
    ``cal_titau_*`` + ``horizontal_diffusion_[uvw]_2``) already carries
    WRF's ``mrdx=msfux*rdx``, so the only thing left for this gate to
    catch is ``rk_addtend_dry``'s division -- module_em.F:1043-1078.
    Without it every row is msf times WRF's: msft here spans 1.0-1.2 and
    the miss is 8-20%, two orders outside the tolerance below.

    ``bl_pbl_physics=1`` keeps ``vertical_diffusion_2`` out, so the held
    buffer is exactly the authority's horizontal rows.
    """
    from gpuwm.config import RunConfig

    cfg = RunConfig(**_BASE, km_opt=4, bl_pbl_physics=1, diff_6th_opt=0)
    state = _state(cfg)
    import cupy as cp
    msf = {name: cp.asnumpy(getattr(state, name)).astype(np.float64)
           for name in ("msft", "msfu", "msfv")}
    momentum, theta, _qv = _authorities(state, cfg)
    got = _delivered(state, cfg)

    raw = {"ru_t": momentum["ru"], "rv_t": momentum["rv"],
           "rw_t": momentum["rw"], "rth_t": theta}
    for _slot, tend, msf_name in _ROWS:
        ref = raw[tend] / msf[msf_name][None]
        scale = max(float(np.max(np.abs(ref))), 1.0e-10)
        assert scale > 1.0e-6, f"{tend}: the authority produced nothing"
        np.testing.assert_allclose(got[tend], ref, rtol=3.0e-3,
                                   atol=4.0e-5 * scale, err_msg=tend)


@requires_gpu
def test_the_diff6_rows_reach_the_tendency_with_no_map_factor():
    """diff_6th_opt=2 alone on a mapped grid: the delivered dry tendency
    must carry NO map factor, because WRF multiplies by msf inside
    ``sixth_order_diffusion`` and divides it back out in
    ``rk_addtend_dry`` -- and diff6.cu omits both.

    Graded against gpuwm's own float64 diff6 mirror, so the number is
    WRF's operator and not this module's premise.  A ``1/msf`` applied to
    the shared carrying buffer shows up here as an 8-20% deficit.
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core import dycore
    from gpuwm.verify.npref import np_diff6

    cfg = RunConfig(**_BASE, km_opt=0, diff_6th_opt=2, diff_6th_factor=0.12)
    state = _state(cfg)
    got = _delivered(state, cfg)

    mu_t = cp.asnumpy(state.mub2d + state.mup0).astype(np.float64)
    factor = dycore._clock_scaled_diff6_factor(cfg)
    rows = {"ru_t": ("u0", state.c1h, state.c2h, "x"),
            "rv_t": ("v0", state.c1h, state.c2h, "y"),
            "rw_t": ("w0", state.c1f, state.c2f, "z"),
            "rth_t": ("thp0", state.c1h, state.c2h, "")}
    for _slot, tend, _msf in _ROWS:
        name, c1, c2, stag = rows[tend]
        f0 = cp.asnumpy(getattr(state, name)).astype(np.float64)
        ref = np_diff6(f0, mu_t, cp.asnumpy(c1), cp.asnumpy(c2),
                       factor, cfg.dt, cfg.diff_6th_opt, stagger=stag)
        scale = max(float(np.max(np.abs(ref))), 1.0e-10)
        assert scale > 1.0e-6, f"{tend}: the mirror produced nothing"
        np.testing.assert_allclose(got[tend], ref, rtol=1.0e-4,
                                   atol=1.0e-6 * scale, err_msg=tend)


@requires_gpu
def test_both_packages_at_once_deliver_coupled_mixing_plus_raw_diff6():
    """km_opt=4 AND diff_6th_opt=2 on a mapped grid, each half graded
    against its own float64 authority in the one run that shares the
    carrying buffer.

    This is the gate the two single-package gates cannot be: dividing the
    SUM is still additive, so a composition test that only checks
    ``both == mixing + diff6`` passes on every arrangement.  Requiring the
    delivered tendency to be ``authority/msf + np_diff6`` instead fails
    both ways -- as ``(S+D)/msf`` when the division is applied to the sum,
    and as ``S+D`` when it is applied to neither.
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core import dycore
    from gpuwm.verify.npref import np_diff6

    cfg = RunConfig(**_BASE, km_opt=4, bl_pbl_physics=1, diff_6th_opt=2,
                    diff_6th_factor=0.12)
    state = _state(cfg)
    msf = {name: cp.asnumpy(getattr(state, name)).astype(np.float64)
           for name in ("msft", "msfu", "msfv")}
    momentum, theta, _qv = _authorities(state, cfg)
    mixing = {"ru_t": momentum["ru"], "rv_t": momentum["rv"],
              "rw_t": momentum["rw"], "rth_t": theta}
    mu_t = cp.asnumpy(state.mub2d + state.mup0).astype(np.float64)
    factor = dycore._clock_scaled_diff6_factor(cfg)
    rows = {"ru_t": ("u0", state.c1h, state.c2h, "x"),
            "rv_t": ("v0", state.c1h, state.c2h, "y"),
            "rw_t": ("w0", state.c1f, state.c2f, "z"),
            "rth_t": ("thp0", state.c1h, state.c2h, "")}
    got = _delivered(state, cfg)

    for _slot, tend, msf_name in _ROWS:
        name, c1, c2, stag = rows[tend]
        f0 = cp.asnumpy(getattr(state, name)).astype(np.float64)
        filtered = np_diff6(f0, mu_t, cp.asnumpy(c1), cp.asnumpy(c2),
                            factor, cfg.dt, cfg.diff_6th_opt, stagger=stag)
        coupled = mixing[tend] / msf[msf_name][None]
        # Neither half may be negligible against the other, or the sum
        # would grade only one of them.
        big, small = np.max(np.abs(coupled)), np.max(np.abs(filtered))
        assert min(big, small) > 0.02 * max(big, small), tend
        ref = coupled + filtered
        scale = max(float(np.max(np.abs(ref))), 1.0e-10)
        np.testing.assert_allclose(got[tend], ref, rtol=3.0e-3,
                                   atol=4.0e-5 * scale, err_msg=tend)


@requires_gpu
def test_the_moist_rows_are_not_coupled_but_theta_is():
    """One operator, two conventions: ``rk_addtend_dry`` divides theta
    (module_em.F:1078) while ``rk_update_scalar`` adds the moist rows'
    ``sc_tend`` raw (:1697-1709, applied in gpuwm.core.moist).

    Both rows come out of the same scalar mixing kernel, so grading them
    against the same float64 authority in one test is what stops a
    division from being sprayed over every scalar -- or lifted off theta.
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core import dycore

    cfg = RunConfig(**_BASE, km_opt=4, bl_pbl_physics=1, diff_6th_opt=0)
    state = _state(cfg)
    msft = cp.asnumpy(state.msft).astype(np.float64)
    _momentum, theta, qv_ref = _authorities(state, cfg)
    got = _delivered(state, cfg)

    theta_ref = theta / msft[None]
    scale = max(float(np.max(np.abs(theta_ref))), 1.0e-10)
    assert scale > 1.0e-6
    np.testing.assert_allclose(got["rth_t"], theta_ref, rtol=3.0e-3,
                               atol=4.0e-5 * scale, err_msg="rth_t")

    held_qv = cp.asnumpy(
        state.scratch(state.qv.shape, "smag_rqv")).astype(np.float64)
    qv_scale = max(float(np.max(np.abs(qv_ref))), 1.0e-10)
    assert qv_scale > 1.0e-9
    np.testing.assert_allclose(held_qv, qv_ref, rtol=3.0e-3,
                               atol=4.0e-5 * qv_scale, err_msg="smag_rqv")
    # ...and the two conventions really are distinguishable on this grid.
    assert np.max(np.abs(qv_ref / msft[None] - qv_ref)) > 1.0e-2 * qv_scale


@requires_gpu
def test_an_unmapped_grid_delivers_the_held_buffer_bitwise():
    """has_msf off is the unmapped idealized lane: WRF's msf == 1 makes
    ``rk_addtend_dry``'s division an identity, and gpuwm must skip it
    entirely rather than divide by ones, so the Phase-2 bitwise baselines
    do not move by a rounding step."""
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core import dycore

    cfg = RunConfig(**_BASE, km_opt=4, bl_pbl_physics=1, diff_6th_opt=2,
                    diff_6th_factor=0.12)
    state = _state(cfg, mapped=False)
    dycore.prepare_fixed_tendencies(state, cfg)
    held = {slot: cp.asnumpy(state.scratch(getattr(state, tend).shape, slot))
            for slot, tend, _msf in _ROWS}
    for _slot, tend, _msf in _ROWS:
        getattr(state, tend)[...] = 0
    dycore.add_fixed_dry_tendencies(state, cfg)
    for slot, tend, _msf in _ROWS:
        assert np.max(np.abs(held[slot])) > 0.0, slot
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(state, tend)), held[slot], err_msg=tend)
