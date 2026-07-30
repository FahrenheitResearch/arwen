# tests/test_advection.py
import numpy as np
import pytest
from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
def test_scalar_flux_div_matches_reference():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    from gpuwm.verify.npref import np_flux_div_scalar
    from gpuwm.core.grid import make_vertical_coord
    nz, ny, nx = 12, 4, 24
    rng = np.random.default_rng(1)
    q  = rng.normal(300, 5, (nz, ny, nx)).astype(np.float32)
    ru = rng.normal(0, 10, (nz, ny, nx + 1)).astype(np.float32)
    rv = rng.normal(0, 10, (nz, ny + 1, nx)).astype(np.float32)
    rw = rng.normal(0, 1, (nz + 1, ny, nx)).astype(np.float32)
    rw[0] = rw[-1] = 0.0
    vc = make_vertical_coord(nz)
    ref = np_flux_div_scalar(q.astype(np.float64), ru.astype(np.float64),
                             rv.astype(np.float64), rw.astype(np.float64),
                             vc, dx=100.0, dy=100.0)
    tend = cp.zeros((nz, ny, nx), cp.float32)
    get_kernel("advection", "flux_div_scalar")  # smoke-check: kernel is registered and loads
    from gpuwm.core.advection import launch_flux_div_scalar
    launch_flux_div_scalar(cp.asarray(q), cp.asarray(ru), cp.asarray(rv),
                           cp.asarray(rw), tend, vc, 100.0, 100.0)
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=5e-4, atol=5e-4)


@pytest.mark.gpu
@requires_gpu
def test_uniform_flow_gaussian_transport():
    # advect a Gaussian one full domain length in uniform u; shape preserved
    import cupy as cp
    from gpuwm.core.advection import advect_scalar_rk3_periodic_test
    nx = 200; dx = 100.0; u0 = 10.0
    x = (np.arange(nx) + 0.5) * dx
    q0 = np.exp(-((x - 5000.0) / 800.0) ** 2).astype(np.float32)
    q_final = advect_scalar_rk3_periodic_test(q0, u0=u0, dx=dx,
                                              t_end=nx * dx / u0, cfl=0.5)
    err = np.max(np.abs(q_final - q0))
    assert err < 0.02          # 5th-order upwind, ~2% peak clipping budget


def test_convergence_order():
    # pure-NumPy reference: error ratio between dx and dx/2 for smooth profile
    import numpy as np
    from gpuwm.verify.npref import np_advect_1d_rk3
    errs = []
    for n in (100, 200):
        x = (np.arange(n) + 0.5) / n
        q0 = np.sin(2 * np.pi * x) ** 4
        qf = np_advect_1d_rk3(q0, u0=1.0, dx=1.0 / n, t_end=1.0, cfl=0.3)
        errs.append(np.max(np.abs(qf - q0)))
    order = np.log2(errs[0] / errs[1])
    assert order > 3.5         # formally 5th in space; RK3 time error keeps it >3.5


def _random_momentum_fields(seed, nz, ny, nx):
    rng = np.random.default_rng(seed)
    ru = rng.normal(0, 10, (nz, ny, nx + 1)).astype(np.float32)
    rv = rng.normal(0, 10, (nz, ny + 1, nx)).astype(np.float32)
    rw = rng.normal(0, 1, (nz + 1, ny, nx)).astype(np.float32)
    return rng, ru, rv, rw


@pytest.mark.gpu
@requires_gpu
def test_u_flux_div_matches_reference():
    import cupy as cp
    from gpuwm.core.advection import launch_flux_div_u
    from gpuwm.verify.npref import np_flux_div_u
    from gpuwm.core.grid import make_vertical_coord
    nz, ny, nx = 12, 4, 24
    rng, ru, rv, rw = _random_momentum_fields(2, nz, ny, nx)
    u = rng.normal(0, 10, (nz, ny, nx + 1)).astype(np.float32)
    vc = make_vertical_coord(nz)
    ref = np_flux_div_u(u.astype(np.float64), ru.astype(np.float64),
                        rv.astype(np.float64), rw.astype(np.float64),
                        vc, dx=100.0, dy=100.0)
    tend = cp.zeros((nz, ny, nx + 1), cp.float32)
    launch_flux_div_u(cp.asarray(u), cp.asarray(ru), cp.asarray(rv),
                      cp.asarray(rw), tend, vc, 100.0, 100.0)
    got = cp.asnumpy(tend)
    np.testing.assert_allclose(got, ref, rtol=5e-4, atol=5e-4)
    # periodic redundancy: the wrapped column must equal column 0 exactly
    np.testing.assert_array_equal(got[:, :, nx], got[:, :, 0])


@pytest.mark.gpu
@requires_gpu
def test_v_flux_div_matches_reference():
    import cupy as cp
    from gpuwm.core.advection import launch_flux_div_v
    from gpuwm.verify.npref import np_flux_div_v
    from gpuwm.core.grid import make_vertical_coord
    nz, ny, nx = 12, 4, 24
    rng, ru, rv, rw = _random_momentum_fields(3, nz, ny, nx)
    v = rng.normal(0, 10, (nz, ny + 1, nx)).astype(np.float32)
    vc = make_vertical_coord(nz)
    ref = np_flux_div_v(v.astype(np.float64), ru.astype(np.float64),
                        rv.astype(np.float64), rw.astype(np.float64),
                        vc, dx=100.0, dy=100.0)
    tend = cp.zeros((nz, ny + 1, nx), cp.float32)
    launch_flux_div_v(cp.asarray(v), cp.asarray(ru), cp.asarray(rv),
                      cp.asarray(rw), tend, vc, 100.0, 100.0)
    got = cp.asnumpy(tend)
    np.testing.assert_allclose(got, ref, rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(got[:, ny, :], got[:, 0, :])


@pytest.mark.gpu
@requires_gpu
def test_w_flux_div_matches_reference():
    import cupy as cp
    from gpuwm.core.advection import launch_flux_div_w
    from gpuwm.verify.npref import np_flux_div_w
    from gpuwm.core.grid import make_vertical_coord
    nz, ny, nx = 12, 4, 24
    rng, ru, rv, rw = _random_momentum_fields(4, nz, ny, nx)
    w = rng.normal(0, 5, (nz + 1, ny, nx)).astype(np.float32)
    vc = make_vertical_coord(nz)
    ref = np_flux_div_w(w.astype(np.float64), ru.astype(np.float64),
                        rv.astype(np.float64), rw.astype(np.float64),
                        vc, dx=100.0, dy=100.0)
    tend = cp.zeros((nz + 1, ny, nx), cp.float32)
    launch_flux_div_w(cp.asarray(w), cp.asarray(ru), cp.asarray(rv),
                      cp.asarray(rw), tend, vc, 100.0, 100.0)
    got = cp.asnumpy(tend)
    np.testing.assert_allclose(got, ref, rtol=5e-4, atol=5e-4)
    # boundary w-levels get no advective tendency
    assert np.all(got[0] == 0.0) and np.all(got[nz] == 0.0)


@pytest.mark.gpu
@requires_gpu
def test_vertical_face_weights_stretched_hand_pin():
    """WRF uses the stretched-grid interpolation weights fzm/fzp (= gpuwm
    fnm/fnp) for the 2nd-order vertical fallback faces one face in from the
    domain top/bottom -- vflux = rom*(fzm(k)*f(k) + fzp(k)*f(k-1)) at
    k=kts+1 and k=ktf for every advected field (module_advect_em.F
    vert_order 3: scalars :4322/:4327, u :1486/:1490, v mirrors) -- NOT
    0.5/0.5.  Hand-computed pin on a tanh-stretched grid: with ru = rv = 0
    and a k-only scalar profile, the k=0 and k=nz-1 tendencies are exactly
    -(fz[1]-0)*rdnw[0] and -(0-fz[nz-1])*rdnw[nz-1] with the weighted face
    values.  Pre-fix (0.5/0.5 faces) this fails."""
    import cupy as cp
    from gpuwm.core.advection import launch_flux_div_scalar
    from gpuwm.core.grid import make_vertical_coord
    nz, ny, nx = 6, 4, 8
    vc = make_vertical_coord(nz, stretch=1.5)
    assert abs(vc.fnm[1] - 0.5) > 0.01          # weights are non-degenerate
    qk = np.array([3.0, 1.0, 2.5, 0.5, 2.0, 4.0])
    vk = np.array([0.0, 1.5, -2.0, 1.0, -0.5, 2.5, 0.0])
    q = cp.asarray(np.broadcast_to(qk[:, None, None], (nz, ny, nx)),
                   dtype=cp.float32)
    ru = cp.zeros((nz, ny, nx + 1), cp.float32)
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw = cp.asarray(np.broadcast_to(vk[:, None, None], (nz + 1, ny, nx)),
                    dtype=cp.float32)
    tend = cp.zeros((nz, ny, nx), cp.float32)
    launch_flux_div_scalar(q, ru, rv, rw, tend, vc, 100.0, 100.0)
    got = cp.asnumpy(tend)
    fz1 = vk[1] * (vc.fnm[1] * qk[1] + vc.fnp[1] * qk[0])
    fztop = vk[nz - 1] * (vc.fnm[nz - 1] * qk[nz - 1]
                          + vc.fnp[nz - 1] * qk[nz - 2])
    np.testing.assert_allclose(got[0], -fz1 * vc.rdnw[0], rtol=1e-5)
    np.testing.assert_allclose(got[nz - 1], fztop * vc.rdnw[nz - 1],
                               rtol=1e-5)


@pytest.mark.gpu
@requires_gpu
def test_w_horizontal_velocity_weights_stretched_hand_pin():
    """WRF interpolates the horizontal advecting momenta to w levels with
    fzm/fzp -- vel = fzm(k)*ru(i,k,j) + fzp(k)*ru(i,k-1,j)
    (module_advect_em.F advect_w :5004/:4531) -- not 0.5/0.5.
    Hand-computed pin: w linear in x (flux5 of a linear field is the exact
    midpoint), ru constant per level, so the interior tendency at w level
    k is exactly -(fnm[k]*ru[k] + fnp[k]*ru[k-1]) * b / dx.  Pre-fix
    (0.5/0.5 level averaging) this fails."""
    import cupy as cp
    from gpuwm.core.advection import launch_flux_div_w
    from gpuwm.core.grid import make_vertical_coord
    nz, ny, nx = 6, 4, 12
    dx = 100.0
    vc = make_vertical_coord(nz, stretch=1.5)
    b = 0.25
    w_host = np.broadcast_to(b * np.arange(nx, dtype=np.float64),
                             (nz + 1, ny, nx)).copy()
    rk = np.array([2.0, -1.0, 3.0, 0.5, -2.5, 1.5])
    ru = cp.asarray(np.broadcast_to(rk[:, None, None], (nz, ny, nx + 1)),
                    dtype=cp.float32)
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    rw = cp.zeros((nz + 1, ny, nx), cp.float32)
    w = cp.asarray(w_host, dtype=cp.float32)
    tend = cp.zeros((nz + 1, ny, nx), cp.float32)
    launch_flux_div_w(w, ru, rv, rw, tend, vc, dx, dx)
    got = cp.asnumpy(tend)
    for k in range(1, nz):
        velx = vc.fnm[k] * rk[k] + vc.fnp[k] * rk[k - 1]
        expected = -velx * b / dx
        np.testing.assert_allclose(got[k, :, 4:nx - 4], expected,
                                   rtol=1e-4, atol=1e-7,
                                   err_msg=f"w level {k}")
