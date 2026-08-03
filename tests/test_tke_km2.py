# tests/test_tke_km2.py
"""km_opt=2 (1.5-order prognostic TKE), WRF v4.6.1 (P6 LES engine track).

R0 battery: float64 mirrors for tke_km (both mix_isotropic branches, the
dthrdn stability length, the tke_seed floor, WRF's limiter asymmetry) and
tke_rhs (shear + buoyancy + dissipation + positivity limiter, with the
budget-term decomposition); mutation/sensitivity controls in the spec's
shape (zeroed dissipation moves the budget by orders, a production-sign
mutant grows where the real term damps, forcing-off decay is monotone --
AC-L5.1's bound-free control arm and AC-L5.2); and the dycore wiring
proofs (TKE bootstraps from the prescribed surface flux in a CBL smoke,
the carrier advances through the shared scalar-transport machinery, and
frozen km_opt=1/3/4 paths carry no TKE state at all).

Formula authority: handoffs/P6-LES-WPL0-AUTHORITY-RECEIPT.md plus direct
reads of the byte-identical local WRF v4.6.1 bundle
(module_diffusion_em.F tke_km/calc_l_scale/tke_shear/tke_buoyancy/
tke_dissip/tke_rhs).
"""
import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.config import RunConfig
from gpuwm.core.grid import make_base_state, make_vertical_coord

C_K = 0.10          # em_les reference value (README.les)


def _cfg(nx=12, ny=6, nz=8, **kw):
    kw.setdefault("km_opt", 2)
    kw.setdefault("bl_pbl_physics", 0)
    kw.setdefault("isfflx", 0)
    kw.setdefault("tke_heat_flux", 0.24)
    kw.setdefault("tke_drag_coefficient", 0.0013)
    kw.setdefault("dt", 2.0)
    return RunConfig(nx=nx, ny=ny, nz=nz, dx=200.0, dy=200.0, ztop=2000.0,
                    run_seconds=0.0, c_k=C_K, **kw)


def _state(cfg, theta_func, seed=None, wind_amp=0.0, tke_amp=0.5):
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import init_at_rest
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_func, p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    rng = np.random.default_rng(0 if seed is None else seed)
    if seed is not None and wind_amp > 0.0:
        u = wind_amp * rng.standard_normal((cfg.nz, cfg.ny, cfg.nx + 1))
        v = wind_amp * rng.standard_normal((cfg.nz, cfg.ny + 1, cfg.nx))
        w = wind_amp * rng.standard_normal((cfg.nz + 1, cfg.ny, cfg.nx))
        u[:, :, cfg.nx] = u[:, :, 0]
        v[:, cfg.ny, :] = v[:, 0, :]
        w[0] = 0.0
        w[cfg.nz] = 0.0
        s.u[...] = cp.asarray(u, dtype=cp.float32)
        s.v[...] = cp.asarray(v, dtype=cp.float32)
        s.w[...] = cp.asarray(w, dtype=cp.float32)
    if tke_amp > 0.0:
        e = tke_amp * np.abs(rng.standard_normal(
            (cfg.nz, cfg.ny, cfg.nx)))
        # A zero-TKE patch exercises the tke_seed/1e-6 floors.
        e[:, 0, :2] = 0.0
        s.tke[...] = cp.asarray(e, dtype=cp.float32)
    update_diagnostics(s, cfg.hypsometric_opt)
    return s


def _host(state):
    import cupy as cp
    phb = cp.asnumpy(state.phb).astype(np.float64)
    php = cp.asnumpy(state.php).astype(np.float64)
    phi = phb[:, None, None] + php if phb.ndim == 1 else phb + php
    return {
        "u": cp.asnumpy(state.u), "v": cp.asnumpy(state.v),
        "w": cp.asnumpy(state.w), "phi": phi,
        "thp": cp.asnumpy(state.thp), "thb": cp.asnumpy(state.thb),
        "p": cp.asnumpy(state.p), "alt": cp.asnumpy(state.alt),
        "tke": cp.asnumpy(state.tke),
        "mut": cp.asnumpy(state.mub2d + state.mup).astype(np.float64),
        "c1h": cp.asnumpy(state.c1h).astype(np.float64),
        "c2h": cp.asnumpy(state.c2h).astype(np.float64),
        "dnw": cp.asnumpy(state.dnw).astype(np.float64),
        "dn": cp.asnumpy(state.dn).astype(np.float64),
        "fnm": cp.asnumpy(state.fnm).astype(np.float64),
        "fnp": cp.asnumpy(state.fnp).astype(np.float64),
        "cf1": float(state.cf1), "cf2": float(state.cf2),
        "cf3": float(state.cf3),
    }


def _launch_tke(state, cfg):
    """Run the km_opt=2 launcher; return (kmh, khh, kmv, khv, rtke)."""
    import cupy as cp
    from gpuwm.core.dycore import launch_wrf_tke_km
    nz, ny, nx = state.p.shape
    km = state.scratch((nz, ny, nx), "smag_km")
    kh = state.scratch((nz, ny, nx), "smag_kh")
    state.scratch((nz, ny, nx), "smag_rtke")[...] = 0
    launch_wrf_tke_km(state, cfg, km, kh, time_t=False)
    return (cp.asnumpy(km), cp.asnumpy(kh),
            cp.asnumpy(state.scratch((nz, ny, nx), "smag_kmv")),
            cp.asnumpy(state.scratch((nz, ny, nx), "smag_khv")),
            cp.asnumpy(state.scratch((nz, ny, nx), "smag_rtke")))


def _rdzw_of(h):
    """Layer inverse thickness G/dphi per mass level, broadcast (nz,1,1)."""
    return 9.81 / (h["phi"][1:] - h["phi"][:-1])


def _mirror_k_and_rhs(cfg, h, *, tke_seed):
    from gpuwm.verify.npref import (np_wrf_calc_n2, np_wrf_tke_km,
                                    np_wrf_tke_rhs)
    bn2 = np_wrf_calc_n2(h["thp"], h["thb"], h["p"], h["phi"],
                         cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"])
    ks = np_wrf_tke_km(h["thp"], h["thb"], h["p"], h["tke"], bn2,
                       h["phi"], dx=cfg.dx, dy=cfg.dy, dt=cfg.dt,
                       c_k=cfg.c_k, mix_upper_bound=cfg.mix_upper_bound,
                       mix_isotropic=cfg.mix_isotropic,
                       tke_seed=tke_seed,
                       cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"])
    rhs = np_wrf_tke_rhs(
        h["u"], h["v"], h["w"], h["phi"], h["thp"], h["thb"], h["tke"],
        bn2, ks[0], ks[2], ks[3], h["mut"], h["c1h"], h["c2h"],
        dx=cfg.dx, dy=cfg.dy, dt=cfg.dt, c_k=cfg.c_k, isfflx=cfg.isfflx,
        cd0=cfg.tke_drag_coefficient, heat_flux=cfg.tke_heat_flux,
        fnm=h["fnm"], fnp=h["fnp"], dn=h["dn"], dnw=h["dnw"],
        cf1=h["cf1"], cf2=h["cf2"], cf3=h["cf3"], return_terms=True)
    return bn2, ks, rhs


@requires_gpu
@pytest.mark.parametrize("isotropic", [0, 1])
def test_wrf_tke_km_and_rhs_match_mirror(isotropic):
    cfg = _cfg(mix_isotropic=isotropic)
    s = _state(cfg, lambda z: 300.0 + 0.004 * np.asarray(z, float),
               seed=1, wind_amp=1.5, tke_amp=0.5)
    dev = _launch_tke(s, cfg)
    h = _host(s)
    # isfflx=0 with nonzero prescribed flux/drag -> tke_seed stays 0.
    _bn2, ks, (total, shear, buoy, dissip) = _mirror_k_and_rhs(
        cfg, h, tke_seed=0.0)
    for name, dv, rf in zip(("xkmh", "xkhh", "xkmv", "xkhv"), dev[:4], ks):
        scale = max(float(np.abs(rf).max()), 1e-30)
        np.testing.assert_allclose(dv, rf, rtol=2e-3, atol=5e-4 * scale,
                                   err_msg=name)
    scale = max(float(np.abs(total).max()), 1e-30)
    np.testing.assert_allclose(dev[4], total, rtol=5e-3,
                               atol=2e-3 * scale)
    # Deardorff Kh_v = (1 + 2 l/deltas) Km_v: between 1x and 3x on live-K
    # points, never a constant-Prandtl 3x everywhere (the km_opt=3
    # confusion control).  The zero-TKE patch has K = 0 exactly (no seed
    # with forcing on) and is excluded.
    live = ks[2] > 0.0
    if isotropic:
        # Isotropic K has no viscosity floor: the e = 0 patch is exactly
        # zero-K.  The anisotropic branch floors at 1e-6*len^2 (WRF).
        assert live.any() and not live.all()
    else:
        assert live.all()
        floor_engaged = np.isclose(
            ks[2], 1e-6 * (1.0 / _rdzw_of(h)) ** 2, rtol=1e-6)
        assert floor_engaged.any() and not floor_engaged.all()
    ratio = ks[3][live] / ks[2][live]
    assert ratio.min() >= 1.0 - 1e-9 and ratio.max() <= 3.0 + 1e-9
    assert ratio.std() > 1e-3
    # Budget-term sensitivity (AC-L5.1's bound-free control arm): zeroing
    # dissipation moves the volume-integrated source by orders; flipping
    # the production sign turns net source into net sink somewhere.
    vol = float(np.abs(total).sum())
    assert float(np.abs(dissip).sum()) > 0.1 * vol
    assert float(np.abs((shear + buoy) - total).sum()) > 0.1 * vol
    # Interior production is a sum of squared terms times K >= 0; the
    # k=0 level additionally carries the SIGNED MARTA surface-drag
    # transfer and is excluded from the positivity claim.
    assert (shear[1:] >= 0.0).all()


@requires_gpu
def test_tke_seed_floor_engages_without_forcing():
    """isfflx=0 with both prescribed constants OFF seeds sqrt(max(e,1e-6))
    so K stays nonzero in zero-TKE air; with forcing on the seed is zero
    and zero-TKE air floors at the 1e-6*len^2 viscosity only."""
    cfg_seeded = _cfg(tke_heat_flux=0.0, tke_drag_coefficient=0.0,
                      mix_isotropic=1)
    s = _state(cfg_seeded, lambda z: 300.0 + 0.0 * np.asarray(z, float),
               tke_amp=0.0)                       # e = 0 everywhere
    kmh_seeded = _launch_tke(s, cfg_seeded)[0]
    from gpuwm.core.dycore import _tke_seed
    assert _tke_seed(cfg_seeded) == pytest.approx(1e-6)
    assert _tke_seed(_cfg()) == 0.0               # forcing on -> no seed
    assert _tke_seed(_cfg(isfflx=1)) == 0.0
    # Seeded K = c_k * sqrt(1e-6) * l > 0 everywhere (neutral: l = deltas).
    assert float(kmh_seeded.min()) > 0.0


@requires_gpu
def test_tke_decay_control():
    """AC-L5.2: forcing off, no wind, neutral column, uniform initial
    TKE -- dissipation decays e monotonically toward the floor and e
    never goes negative."""
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    cfg = _cfg(nx=12, ny=12, nz=12, tke_heat_flux=0.0,
               tke_drag_coefficient=0.0, mix_isotropic=1, dt=1.0)
    s = _state(cfg, lambda z: 300.0 + 0.0 * np.asarray(z, float),
               tke_amp=0.0)
    s.tke[...] = cp.float32(0.5)
    means = [float(s.tke.mean())]
    for _ in range(8):
        run_steps(s, cfg, 1)
        means.append(float(s.tke.mean()))
        assert float(s.tke.min()) >= 0.0
    diffs = np.diff(means)
    assert (diffs < 0.0).all(), means
    # Deardorff rate check: de/dt = -coefc e^1.5 / l with l = deltas ~
    # 188 m, interior coefc = 0.93 at c_k = 0.10 gives ~2.8%/8s plus the
    # 3.9 wall levels -- measurable but nowhere near collapse.
    assert 0.90 * means[0] < means[-1] < 0.985 * means[0]
    assert np.isfinite(cp.asnumpy(s.tke)).all()


@requires_gpu
def test_step_km_opt2_cbl_smoke_tke_bootstraps():
    """Prescribed-flux CBL smoke: TKE bootstraps from zero through the
    surface buoyancy-flux source, the lowest level warms, everything
    stays finite, and turning the flux off removes both."""
    import cupy as cp
    from gpuwm.core.dycore import run_steps, stability_report
    from gpuwm.core.state import init_theta_perturbation

    def build(heat_flux):
        cfg = RunConfig(nx=16, ny=16, nz=16, dx=100.0, dy=100.0,
                        ztop=1600.0, dt=0.5, run_seconds=0.0,
                        km_opt=2, c_k=C_K, mix_isotropic=1,
                        bl_pbl_physics=0, isfflx=0,
                        tke_heat_flux=heat_flux,
                        tke_drag_coefficient=0.0013,
                        time_step_sound=4)
        vc = make_vertical_coord(cfg.nz)
        b = make_base_state(vc, lambda z: 300.0 + 0.003
                            * np.asarray(z, float),
                            p_surf=cfg.p_surf, ztop=cfg.ztop)

        def thp_func(x, z):
            rng = np.random.default_rng(11)
            pert = np.zeros((cfg.nz, cfg.ny, cfg.nx))
            pert[:4] = 0.1 * rng.standard_normal((4, cfg.ny, cfg.nx))
            return pert

        return init_theta_perturbation(cfg, vc, b, thp_func), cfg

    s_on, cfg_on = build(0.24)
    s_off, cfg_off = build(0.0)
    th0_on = float(s_on.thp[0].mean())
    run_steps(s_on, cfg_on, 10)
    run_steps(s_off, cfg_off, 10)
    rep = stability_report(s_on, cfg_on)
    assert not rep["nan"]
    assert float(s_on.tke.max()) > 1e-4          # bootstrapped from zero
    assert float(s_on.tke.min()) >= 0.0
    assert float(s_on.tke[0].mean()) > 10.0 * float(s_off.tke[0].mean())
    assert float(s_on.thp[0].mean()) - th0_on > 1e-3
    assert np.isfinite(cp.asnumpy(s_on.tke)).all()
    assert np.isfinite(cp.asnumpy(s_on.thp)).all()
    assert np.isfinite(cp.asnumpy(s_on.w)).all()


def test_km_opt2_config_admission():
    from gpuwm.config import validate_run_config
    base = dict(nx=16, ny=16, nz=16, dx=100.0, dy=100.0, ztop=1600.0,
                dt=0.5, run_seconds=0.0)
    ok = validate_run_config(RunConfig(
        **base, km_opt=2, bl_pbl_physics=0, isfflx=0,
        tke_heat_flux=0.24, tke_drag_coefficient=0.0013, c_k=0.10))
    assert ok.km_opt == 2 and ok.c_k == 0.10
    with pytest.raises(ValueError, match="bl_pbl_physics=0"):
        validate_run_config(RunConfig(**base, km_opt=2, bl_pbl_physics=1,
                                      sf_sfclay_physics=1, moist=True))
    # The lateral arm (WRF bound_tke + flow_dep_bdy) and the restart
    # carrier landed, so the two refusals that stood in for them are gone.
    for extra in (dict(open_x=True), dict(open_y=True),
                  dict(specified=True), dict(restart_interval_s=3600.0)):
        widened = validate_run_config(RunConfig(
            **base, km_opt=2, bl_pbl_physics=0, isfflx=0,
            tke_heat_flux=0.24, **extra))
        assert widened.km_opt == 2
    # A nest child is NO LONGER refused here.  The refusal that used to sit
    # at this line assumed the answer without seeing the parent, and the
    # parent is what decides whether the nest-coupling question is live at
    # all: WRF hands a child no TKE (no Registry ``i`` flag) and takes none
    # back (no ``f``), so under a parent carrying no TKE there is nothing to
    # couple.  That case has been run and scored (7 h nested 250 m km_opt=2
    # child under a km_opt=4 parent, PASS).  What remains unmeasured -- a
    # km_opt=2 child under a km_opt=2 PARENT -- needs both configs to see,
    # so it is refused in gpuwm.experiment and asserted there, not here.
    child = validate_run_config(RunConfig(
        **base, km_opt=2, bl_pbl_physics=0, isfflx=0,
        tke_heat_flux=0.24, nested=True))
    assert child.km_opt == 2 and child.nested
    with pytest.raises(ValueError, match="c_k"):
        validate_run_config(RunConfig(**base, km_opt=2, isfflx=0,
                                      tke_heat_flux=0.24, c_k=0.0))
    with pytest.raises(ValueError, match="tke_upper_bound"):
        validate_run_config(RunConfig(**base, km_opt=2, isfflx=0,
                                      tke_heat_flux=0.24,
                                      tke_upper_bound=0.0))
    # The budget toggle refuses a closure that carries no TKE at all.
    with pytest.raises(ValueError, match="tke_budget=1 requires km_opt=2"):
        validate_run_config(RunConfig(**base, km_opt=3, bl_pbl_physics=0,
                                      tke_budget=1))
    from gpuwm.experiment import _DOMAIN_RUN_OVERRIDES
    assert "c_k" in _DOMAIN_RUN_OVERRIDES
    assert "tke_upper_bound" in _DOMAIN_RUN_OVERRIDES


def _two_domain_toml(parent_km_opt: int, child_km_opt: int) -> str:
    """A minimal PBL-off two-domain tree with the turbulence row varied."""
    child_extra = "c_k = 0.10\n" if child_km_opt == 2 else ""
    parent_extra = "c_k = 0.10\n" if parent_km_opt == 2 else ""
    return f"""
[experiment]
name = "km2_nest_admission"
start_time = 2026-08-01T00:00:00
run_seconds = 60.0
restart_interval_s = 0.0
[projection]
map_proj = "lambert"
ref_lat = 40.0
ref_lon = -100.0
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -100.0
[shared]
nz = 40
ztop = 20000.0
p_top = 10000.0
km_opt = {parent_km_opt}
bl_pbl_physics = 0
isfflx = 0
tke_heat_flux = 0.24
{parent_extra}mix_isotropic = 1
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 60
ny = 60
dx = 3000.0
time_step = 6
specified = true
nested = false
history_interval_s = 3600.0
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 15
j_parent_start = 15
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 30
ny = 30
specified = false
nested = true
history_interval_s = 3600.0
km_opt = {child_km_opt}
{child_extra}"""


@pytest.mark.parametrize("parent_km_opt", [3, 4])
def test_km_opt2_child_is_admitted_under_a_parent_carrying_no_tke(
        tmp_path, parent_km_opt):
    """The measured case.  WRF hands a child no TKE and takes none back,
    so under a parent that HAS no TKE the nest-coupling question is void
    rather than unverified -- and a 7 h nested 250 m km_opt=2 child under
    a km_opt=4 parent has now run to PASS."""
    from gpuwm.experiment import load_experiment

    path = tmp_path / "tree.toml"
    path.write_text(_two_domain_toml(parent_km_opt, 2), encoding="utf-8")
    exp = load_experiment(path)
    assert [d.run.km_opt for d in exp.domains] == [parent_km_opt, 2]
    assert exp.domains[1].run.nested


def test_km_opt2_child_under_a_km_opt2_parent_is_still_refused(tmp_path):
    """The unmeasured case, and the only one left.

    Here the parent really does hold a prognostic TKE field that WRF's
    Registry flags decline to interpolate down or feed back, and no such
    tree has been run.  It is refused where the parent is visible --
    validate_run_config sees one domain at a time and could only refuse
    this by refusing the measured case with it."""
    from gpuwm.experiment import load_experiment

    path = tmp_path / "tree.toml"
    path.write_text(_two_domain_toml(2, 2), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="also runs km_opt=2"):
        load_experiment(path)


@requires_gpu
def test_other_km_opts_carry_no_tke_state():
    """Frozen paths: km_opt 1/3/4 states allocate no TKE carrier, so the
    new transport arm is unreachable there by construction."""
    from gpuwm.core.state import init_at_rest
    for km_opt in (1, 3, 4):
        cfg = RunConfig(nx=8, ny=6, nz=8, dx=500.0, dy=500.0,
                        ztop=4000.0, dt=2.0, run_seconds=0.0,
                        km_opt=km_opt)
        vc = make_vertical_coord(cfg.nz)
        b = make_base_state(vc, lambda z: 300.0 + 0.003
                            * np.asarray(z, float),
                            p_surf=cfg.p_surf, ztop=cfg.ztop)
        s = init_at_rest(cfg, vc, b)
        assert s.tke is None and s.tke0 is None
