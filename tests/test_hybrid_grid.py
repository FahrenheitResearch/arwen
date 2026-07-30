# tests/test_hybrid_grid.py  (Phase 2 Task 2: hybrid coordinate + 2-D dry mass)
import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.core import constants as c
from gpuwm.core.grid import (compute_hybrid_coeffs, hybrid_b_poly,
                             make_base_state, make_vertical_coord)


def theta_const(z):
    return np.full_like(np.asarray(z, float), 300.0)


def phase1_base_state(coord, sounding, p_surf, ztop):
    """Verbatim Phase 1 ``make_base_state`` algorithm (pre-hybrid).

    The bit-compatibility anchor: for ``hybrid_opt=0``, flat terrain and
    integer ztop the rewritten ``make_base_state`` must reproduce every
    array bitwise.
    """
    zf = np.arange(0.0, ztop + 1.0, 1.0)
    dz = np.diff(zf)
    z_mid = 0.5 * (zf[:-1] + zf[1:])
    pi_surf = (p_surf / c.P0) ** c.RCP
    pi = np.concatenate(([pi_surf],
                         pi_surf - np.cumsum(c.G * dz / (c.CP * sounding(z_mid)))))
    p_of_z = c.P0 * pi ** (1.0 / c.RCP)
    p_top = float(p_of_z[-1])
    mub = float(p_surf - p_top)
    pb = coord.znu * mub + p_top
    z_of_pb = np.interp(pb, p_of_z[::-1], zf[::-1])
    thb = sounding(z_of_pb)
    alb = c.RD * thb * (pb / c.P0) ** c.RCP / pb
    phb = np.zeros(coord.znw.size)
    for k in range(coord.dnw.size):
        phb[k + 1] = phb[k] - coord.dnw[k] * mub * alb[k]
    return mub, p_top, pb, alb, thb, phb


# ---- (a) hybrid_opt=0: exact Phase 1 identity --------------------------------

def test_hybrid_opt0_coeffs_are_exact_identity():
    znw = np.linspace(1.0, 0.0, 33)
    znu = 0.5 * (znw[:-1] + znw[1:])
    hy = compute_hybrid_coeffs(znw, hybrid_opt=0, etac=0.2, p0=c.P0, pt=5000.0)
    assert np.all(hy["c1f"] == 1.0) and np.all(hy["c1h"] == 1.0)
    assert np.all(hy["c2f"] == 0.0) and np.all(hy["c2h"] == 0.0)
    assert np.array_equal(hy["c3f"], znw)
    assert np.array_equal(hy["c3h"], znu)
    assert np.all(hy["c4f"] == 0.0) and np.all(hy["c4h"] == 0.0)


def test_vertical_coord_carries_identity_coeffs_by_default():
    vc = make_vertical_coord(16)
    assert vc.hybrid_opt == 0
    assert np.all(vc.c1f == 1.0) and np.all(vc.c1h == 1.0)
    assert np.all(vc.c2f == 0.0) and np.all(vc.c2h == 0.0)
    assert np.array_equal(vc.c3f, vc.znw)
    assert np.array_equal(vc.c3h, vc.znu)
    assert np.all(vc.c4f == 0.0) and np.all(vc.c4h == 0.0)


def test_invalid_hybrid_opt_rejected():
    znw = np.linspace(1.0, 0.0, 9)
    with pytest.raises(ValueError):
        compute_hybrid_coeffs(znw, hybrid_opt=5, etac=0.2, p0=c.P0, pt=5000.0)


def test_hybrid_opt0_base_state_bitwise_phase1():
    vc = make_vertical_coord(64)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    mub, p_top, pb, alb, thb, phb = phase1_base_state(
        vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    # scalar-broadcast compatibility: flat terrain keeps scalar mub, 1-D cols
    assert isinstance(b.mub, float) and b.mub == mub
    assert b.p_top == p_top
    assert b.terrain_z is None
    for got, ref, name in ((b.pb, pb, "pb"), (b.alb, alb, "alb"),
                           (b.thb, thb, "thb"), (b.phb, phb, "phb")):
        assert got.shape == ref.shape, name
        assert np.array_equal(got, ref), name
    # Phase 1 test_base_state_discrete_balance tolerances
    resid = (b.phb[1:] - b.phb[:-1]) + vc.dnw * b.mub * b.alb
    assert np.max(np.abs(resid)) < 1e-8
    assert abs(b.phb[-1] / c.G - 6400.0) < 50.0


def test_ptop_exact_for_non_integer_ztop():
    # final-review carry-over (ledger minor T5): ztop is appended to the fine
    # z-grid so p_top = p(ztop) exactly for non-integer ztop.
    vc = make_vertical_coord(32)
    ztop = 6400.5
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=ztop)
    zf = np.append(np.arange(0.0, ztop, 1.0), ztop)
    dz = np.diff(zf)
    z_mid = 0.5 * (zf[:-1] + zf[1:])
    pi_surf = (1.0e5 / c.P0) ** c.RCP
    pi = pi_surf - np.cumsum(c.G * dz / (c.CP * theta_const(z_mid)))[-1]
    assert b.p_top == float(c.P0 * pi ** (1.0 / c.RCP))
    # integrating half a metre higher must strictly lower p_top
    b_int = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    assert b.p_top < b_int.p_top


# ---- (b) hybrid_opt=2 B(eta) curve --------------------------------------------

def test_hybrid_opt2_b_curve_properties():
    etac = 0.2
    znw = np.linspace(1.0, 0.0, 65)
    hy = compute_hybrid_coeffs(znw, hybrid_opt=2, etac=etac, p0=c.P0,
                               pt=5000.0)
    B = hy["c3f"]
    # endpoints exact
    assert B[0] == 1.0 and B[-1] == 0.0
    # B == 0 below etac
    assert np.all(B[znw < etac] == 0.0)
    # monotone: B decreases along k as eta decreases (non-strict below etac)
    assert np.all(np.diff(B) <= 0.0)
    assert np.all(np.diff(B[znw >= etac]) < 0.0)
    # cubic constraints via the solved polynomial: B(1)=1, B'(1)=1,
    # B(etac)=0, and B' continuous (-> 0) at etac to 1e-10
    b1, b2, b3, b4 = hybrid_b_poly(etac)
    Bp = lambda e: b1 + b2 * e + b3 * e * e + b4 * e ** 3
    dBp = lambda e: b2 + 2.0 * b3 * e + 3.0 * b4 * e * e
    assert abs(Bp(1.0) - 1.0) < 1e-12
    assert abs(dBp(1.0) - 1.0) < 1e-12
    assert abs(Bp(etac)) < 1e-10
    assert abs(dBp(etac)) < 1e-10


def test_hybrid_opt2_matches_wrf_closed_form():
    # WRF v4.6.1 dyn_em/module_initialize_ideal.F lines 796-810 (also
    # nest_init_utils.F compute_vcoord_1d_coeffs): closed-form Klemp cubic.
    etac, pt = 0.2, 5000.0
    znw = np.linspace(1.0, 0.0, 41)
    znu = 0.5 * (znw[:-1] + znw[1:])
    hy = compute_hybrid_coeffs(znw, hybrid_opt=2, etac=etac, p0=c.P0, pt=pt)
    B1 = 2.0 * etac ** 2 * (1.0 - etac)
    B2 = -etac * (4.0 - 3.0 * etac - etac ** 3)
    B3 = 2.0 * (1.0 - etac ** 3)
    B4 = -(1.0 - etac ** 2)
    B5 = (1.0 - etac) ** 4
    c3f = (B1 + B2 * znw + B3 * znw ** 2 + B4 * znw ** 3) / B5
    c3f[znw < etac] = 0.0
    c3f[0], c3f[-1] = 1.0, 0.0
    np.testing.assert_allclose(hy["c3f"], c3f, rtol=0.0, atol=1e-12)
    # and the WRF derived-coefficient construction, exactly
    assert np.array_equal(hy["c4f"], (znw - hy["c3f"]) * (c.P0 - pt))
    assert np.array_equal(hy["c3h"], 0.5 * (hy["c3f"][:-1] + hy["c3f"][1:]))
    assert np.array_equal(hy["c4h"], (znu - hy["c3h"]) * (c.P0 - pt))
    assert np.array_equal(hy["c1h"],
                          np.diff(hy["c3f"]) / np.diff(znw))
    assert np.array_equal(hy["c2h"], (1.0 - hy["c1h"]) * (c.P0 - pt))
    c1f_int = np.diff(hy["c3h"]) / np.diff(znu)
    assert np.array_equal(hy["c1f"][1:-1], c1f_int)
    assert hy["c1f"][0] == 1.0 and hy["c1f"][-1] == 0.0
    assert np.array_equal(hy["c2f"], (1.0 - hy["c1f"]) * (c.P0 - pt))


def test_hybrid_opt2_flat_base_state_recurrence():
    vc = make_vertical_coord(48, hybrid_opt=2, etac=0.2)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=16000.0)
    # make_base_state installed the pressure-weighted coeffs (pt now known)
    assert np.any(vc.c2h != 0.0) and np.any(vc.c4h != 0.0)
    # hybrid discrete hydrostatic recurrence holds to round-off
    resid = ((b.phb[1:] - b.phb[:-1])
             + vc.dnw * (vc.c1h * b.mub + vc.c2h) * b.alb)
    assert np.max(np.abs(resid)) < 1e-8
    # surface dry pressure closes: c3f(1)*mub + c4f(1) + pt == p_surf
    assert abs(vc.c3f[0] * b.mub + vc.c4f[0] + b.p_top - 1.0e5) < 1e-6
    # reference dry pressure monotonically decreasing (WRF's validity check)
    pd_f = vc.c3f * b.mub + vc.c4f + b.p_top
    assert np.all(np.diff(pd_f) < 0.0)


# ---- (c) terrain: bell hill ---------------------------------------------------

@pytest.mark.parametrize("hybrid_opt", [0, 2])
def test_terrain_base_state_bell_hill(hybrid_opt):
    nz, ny, nx = 48, 2, 64
    dx, h0, a = 2000.0, 1000.0, 10000.0
    x = (np.arange(nx) + 0.5) * dx - 0.5 * nx * dx
    hill = h0 / (1.0 + (x / a) ** 2)
    terrain = np.broadcast_to(hill, (ny, nx)).copy()

    vc = make_vertical_coord(nz, hybrid_opt=hybrid_opt, etac=0.2)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=16000.0,
                        terrain_z=terrain)

    assert b.mub.shape == (ny, nx)
    assert b.pb.shape == (nz, ny, nx) and b.alb.shape == (nz, ny, nx)
    assert b.thb.shape == (nz, ny, nx) and b.phb.shape == (nz + 1, ny, nx)
    assert np.array_equal(b.terrain_z, terrain)

    # mub varies: less dry air above high terrain
    ic = int(np.argmax(hill))
    assert b.mub[0, ic] == b.mub.min() and b.mub.max() > b.mub.min()
    # the removed column mass is the hydrostatic weight of ~1000 m of air
    deficit = (1.0e5 - b.p_top) - b.mub[0, ic]
    assert 0.9e4 < deficit < 1.3e4

    # every column satisfies the hybrid discrete hydrostatic recurrence
    incr = (vc.c1h[:, None, None] * b.mub[None, :, :]
            + vc.c2h[:, None, None])
    resid = (b.phb[1:] - b.phb[:-1]) + vc.dnw[:, None, None] * incr * b.alb
    assert np.max(np.abs(resid)) < 1e-8

    # surface geopotential records the terrain exactly
    assert np.array_equal(b.phb[0], c.G * terrain)


def test_hybrid_coordinate_folding_rejected():
    # WRF's compute_vcoord_1d_coeffs validity check: high terrain + large
    # etac makes c1h*mub + c2h < 0 somewhere (the cubic's dB/deta exceeds 1
    # near the surface, so c2h < 0), folding the reference dry pressure.
    # 300 K isentrope at 4000 m gives p_s ~ 61 kPa, well inside the etac=0.5
    # folding region.
    vc = make_vertical_coord(48, hybrid_opt=2, etac=0.5)
    terrain = np.full((1, 8), 4000.0)
    with pytest.raises(ValueError, match="monoton"):
        make_base_state(vc, theta_const, p_surf=1.0e5, ztop=16000.0,
                        terrain_z=terrain)


def test_terrain_z_validation():
    vc = make_vertical_coord(8)
    bad_shape = np.zeros(5)
    with pytest.raises(ValueError):
        make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0,
                        terrain_z=bad_shape)
    too_high = np.full((1, 4), 7000.0)
    with pytest.raises(ValueError):
        make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0,
                        terrain_z=too_high)
    negative = np.full((1, 4), -5.0)
    with pytest.raises(ValueError):
        make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0,
                        terrain_z=negative)


# ---- Phase 2 final-review carry-overs (Phase 3 Task 1: T2-1, T2-5) -----------

def test_make_base_state_records_ptop_on_coord():
    # T2-1: the first finalize records p_top on the coord; the placeholder
    # install in make_vertical_coord (pt = P0) records nothing.
    vc = make_vertical_coord(48, hybrid_opt=2, etac=0.2)
    assert vc.p_top is None
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=16000.0)
    assert vc.p_top == b.p_top


def test_hybrid_coord_reuse_with_different_ptop_rejected():
    # T2-1: with hybrid_opt=2 the pressure-weighted c2/c4 carry a factor
    # (p0 - p_top); reusing the coord with a different p_top must raise
    # instead of silently re-finalizing (which would corrupt every state
    # built from the first base state), and must leave the coefficients
    # untouched.
    vc = make_vertical_coord(48, hybrid_opt=2, etac=0.2)
    b1 = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=16000.0)
    c2h_saved, c4h_saved = vc.c2h.copy(), vc.c4h.copy()
    c2f_saved, c4f_saved = vc.c2f.copy(), vc.c4f.copy()
    with pytest.raises(ValueError, match="p_top"):
        make_base_state(vc, theta_const, p_surf=1.0e5, ztop=12000.0)
    assert vc.p_top == b1.p_top
    assert np.array_equal(vc.c2h, c2h_saved)
    assert np.array_equal(vc.c4h, c4h_saved)
    assert np.array_equal(vc.c2f, c2f_saved)
    assert np.array_equal(vc.c4f, c4f_saved)


def test_hybrid_coord_reuse_with_identical_ptop_allowed():
    # Reuse with bitwise-identical inputs is legal and reproduces the base
    # state exactly (the coefficients are already final; nothing reinstalls).
    vc = make_vertical_coord(48, hybrid_opt=2, etac=0.2)
    b1 = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=16000.0)
    b2 = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=16000.0)
    assert vc.p_top == b1.p_top == b2.p_top
    for name in ("pb", "alb", "thb", "phb"):
        assert np.array_equal(getattr(b2, name), getattr(b1, name)), name
    assert b2.mub == b1.mub


@pytest.mark.parametrize("hybrid_opt", [0, 1])
def test_identity_coord_reuse_with_different_ptop_allowed(hybrid_opt):
    # hybrid_opt=0/1: c2/c4 are identically zero for ANY p_top, so coord
    # reuse across model tops is inert and stays legal (the behavior
    # test_ptop_exact_for_non_integer_ztop relies on); the record follows
    # the latest base state.
    vc = make_vertical_coord(32, hybrid_opt=hybrid_opt)
    b1 = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    b2 = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.5)
    assert b1.p_top != b2.p_top
    assert vc.p_top == b2.p_top
    assert np.all(vc.c2h == 0.0) and np.all(vc.c4h == 0.0)
    assert np.all(vc.c2f == 0.0) and np.all(vc.c4f == 0.0)


def test_terrain_z_defensively_copied():
    # T2-5: BaseState.terrain_z must not alias the caller's array.
    vc = make_vertical_coord(16)
    terrain = np.full((2, 8), 150.0)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0,
                        terrain_z=terrain)
    assert b.terrain_z is not terrain
    assert b.terrain_z.dtype == np.float64
    terrain[0, 0] = 5000.0          # caller mutates its array afterwards
    assert b.terrain_z[0, 0] == 150.0


# ---- stretched eta levels (final-review requirement) -------------------------

def test_uniform_coord_unchanged_by_default():
    vc = make_vertical_coord(16)
    assert np.array_equal(vc.znw, np.linspace(1.0, 0.0, 17))


def test_stretched_coord_properties():
    nz = 32
    vc = make_vertical_coord(nz, stretch=2.0)
    assert vc.znw[0] == 1.0 and vc.znw[-1] == 0.0
    assert np.all(np.diff(vc.znw) < 0.0)          # strictly decreasing
    assert np.all(vc.dnw < 0.0)
    np.testing.assert_allclose(vc.rdnw, 1.0 / vc.dnw)
    # interpolation weights: partition of unity on the interior...
    np.testing.assert_allclose(vc.fnm[1:] + vc.fnp[1:], 1.0, atol=1e-12)
    # ...and non-degenerate (fnm != fnp), which the uniform grid cannot test
    assert np.max(np.abs(vc.fnm[1:] - vc.fnp[1:])) > 1e-3
    # clustering toward the surface: thinnest layer at the bottom
    assert np.abs(vc.dnw[0]) == np.min(np.abs(vc.dnw))
    assert np.abs(vc.dnw[-1]) == np.max(np.abs(vc.dnw))


def test_stretch_must_be_positive():
    with pytest.raises(ValueError):
        make_vertical_coord(16, stretch=-1.0)


@pytest.mark.gpu
@requires_gpu
def test_stretched_grid_acoustic_substep_matches_reference():
    # fnm/fnp and rdn/rdnw orientations are degenerate on the uniform grid
    # (fnm == fnp == 0.5); the stretched grid makes them observable.
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import (np_acoustic_substep,
                                    random_acoustic_state, snapshot)
    s, cfg = random_acoustic_state(seed=13, stretch=2.0)
    fnm = cp.asnumpy(s.fnm)
    fnp = cp.asnumpy(s.fnp)
    assert np.max(np.abs(fnm[1:] - fnp[1:])) > 1e-3   # non-degenerate grid
    before = snapshot(s)
    acoustic_substep(s, cfg, dtau=0.5, first=False)
    ref = np_acoustic_substep(before, cfg, dtau=0.5)
    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp", "p_pp"):
        np.testing.assert_allclose(cp.asnumpy(getattr(s, name)), ref[name],
                                   rtol=2e-4, atol=1e-6, err_msg=name)
