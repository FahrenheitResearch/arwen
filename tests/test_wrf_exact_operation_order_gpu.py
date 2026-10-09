"""Strict-mode dynamics kernels in WRF 4.6.1's operation order, word for word.

The combo sweep of 2026-10-07 (round2 LOCALIZE.md) located three places where
the strict kernels rounded a WRF quantity in another order.  Each test below
writes WRF's Fortran expression as a chain of float32 NumPy operations (each
one correctly rounded, no contraction) and requires every word of the kernel's
output to equal it.  The inputs are random, so the grouped forms the kernels
used before would fail: the tests assert that too, so they cannot pass by
accident on inputs where the orders agree.

Run in a process started with ``GPUWM_WRF_EXACT=1`` (the strict controls are
on by default there).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import requires_gpu

f32 = np.float32
G = f32(9.81)


def _strict_process():
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("requires a process started with GPUWM_WRF_EXACT=1")
    from gpuwm import wrf_exact
    assert all(wrf_exact.effective_controls().values()), wrf_exact.effective_controls()


def _rand(rng, shape, lo, hi):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@requires_gpu
def test_small_step_finish_uv_uncouples_with_wrf_muts_order():
    """WRF small_step_finish divides by c1h*muus+c2h, muus = calc_mu_uv_1 of
    muts = (mub+mu)+mu'' (module_small_step_em.F:392, :1106)."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(20261007)
    nz, ny, nx = 6, 9, 11
    u = _rand(rng, (nz, ny, nx + 1), -30, 30)
    v = _rand(rng, (nz, ny + 1, nx), -30, 30)
    u_pp = _rand(rng, (nz, ny, nx + 1), -50, 50)
    v_pp = _rand(rng, (nz, ny + 1, nx), -50, 50)
    mup = _rand(rng, (ny, nx), -900, 900)
    mu_pp = _rand(rng, (ny, nx), -40, 40)
    mub = _rand(rng, (ny, nx), 80000, 95000)
    c1h = _rand(rng, (nz,), 0.2, 1.0)
    c2h = _rand(rng, (nz,), 0.0, 3000.0)
    msfu = _rand(rng, (ny, nx + 1), 0.98, 1.03)
    msfv = _rand(rng, (ny + 1, nx), 0.98, 1.03)

    out_u, out_v = cp.asarray(u), cp.asarray(v)
    kernel = get_kernel("dycore", "small_step_finish_uv")
    n = nz * (ny + 1) * (nx + 1)
    kernel(((n + 255) // 256,), (256,),
           (out_u, out_v, cp.asarray(u_pp), cp.asarray(v_pp), cp.asarray(mup),
            cp.asarray(mu_pp), cp.asarray(mub), cp.asarray(c1h), cp.asarray(c2h),
            cp.asarray(msfu), cp.asarray(msfv), np.int32(1), np.int32(1), np.int32(1),
            np.int32(nz), np.int32(ny), np.int32(nx),
            # round 3: WRF's grid%muts arrives as the dycore's carrier; in the
            # interior it is advance_mu_t's (mub+mu)+mu''.
            cp.asarray((mub + mup) + mu_pp)))

    def wrf(field, pp, msf, a_idx, b_idx, axis):
        take = (lambda arr, idx: arr[:, idx]) if axis == 1 else (lambda arr, idx: arr[idx, :])
        mu_a, mu_b = take(mup, a_idx), take(mup, b_idx)
        mb_a, mb_b = take(mub, a_idx), take(mub, b_idx)
        muu = f32(0.5) * (((mu_a + mu_b) + mb_a) + mb_b)              # calc_mu_uv
        muts_a = (mb_a + mu_a) + take(mu_pp, a_idx)                    # advance_mu_t
        muts_b = (mb_b + mu_b) + take(mu_pp, b_idx)
        muus = f32(0.5) * (muts_a + muts_b)                            # calc_mu_uv_1
        grouped = f32(0.5) * ((((mu_a + take(mu_pp, a_idx)) + (mu_b + take(mu_pp, b_idx)))
                               + mb_a) + mb_b)
        cs = c1h[:, None, None] * muu[None] + c2h[:, None, None]
        cn = c1h[:, None, None] * muus[None] + c2h[:, None, None]
        cn_old = c1h[:, None, None] * grouped[None] + c2h[:, None, None]
        num = cs * field + pp * msf[None]
        return num / cn, num / cn_old

    i = np.arange(nx + 1)
    want_u, old_u = wrf(u, u_pp, msfu, np.minimum(i, nx - 1), np.maximum(i - 1, 0), 1)
    j = np.arange(ny + 1)
    want_v, old_v = wrf(v, v_pp, msfv, np.minimum(j, ny - 1), np.maximum(j - 1, 0), 0)
    np.testing.assert_array_equal(_words(cp.asnumpy(out_u)), _words(want_u))
    np.testing.assert_array_equal(_words(cp.asnumpy(out_v)), _words(want_v))
    assert np.any(_words(old_u) != _words(want_u)), "inputs do not separate the two orders"


@requires_gpu
def test_slow_buoyancy_scales_by_reciprocal_map_factor_first():
    """WRF pg_buoy_w: rw_tend + (1./msfty)*g*(rdn*(p(k)-p(k-1)) - c1f*mu)
    (module_big_step_utilities_em.F:2480, :2490), dry column."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(7)
    nz, ny, nx = 7, 8, 9
    p = _rand(rng, (nz, ny, nx), -600, 600)
    pb = _rand(rng, (nz, ny, nx), 20000, 90000)
    mup = _rand(rng, (ny, nx), -900, 900)
    mub = _rand(rng, (ny, nx), 80000, 95000)
    rdn = _rand(rng, (nz + 1,), -60, -10)
    rdnw = _rand(rng, (nz,), -60, -10)
    c1f = _rand(rng, (nz + 1,), 0.1, 1.0)
    c2f = _rand(rng, (nz + 1,), 0.0, 3000.0)
    msft = _rand(rng, (ny, nx), 0.97, 1.04)
    rw0 = _rand(rng, (nz + 1, ny, nx), -5, 5)
    dummy = cp.zeros((nz, ny, nx), dtype=cp.float32)

    rw = cp.asarray(rw0)
    kernel = get_kernel("dycore", "slow_buoyancy")
    n = nz * ny * nx
    kernel(((n + 255) // 256,), (256,),
           (rw, cp.asarray(p), cp.asarray(pb), cp.asarray(mup), cp.asarray(mub),
            dummy, dummy, dummy, dummy, dummy, dummy, dummy,
            cp.asarray(rdn), cp.asarray(rdnw), cp.asarray(c1f), cp.asarray(c2f),
            cp.asarray(msft), np.int32(0), np.int32(1), np.int32(1),
            np.int32(nz), np.int32(ny), np.int32(nx)))

    scale = (f32(1.0) / msft) * G
    want = rw0.copy()
    old = rw0.copy()
    for k in range(1, nz):
        term = rdn[k] * (p[k] - p[k - 1]) - c1f[k] * mup
        want[k] = rw0[k] + scale * term
        old[k] = rw0[k] + (G * term) / msft
    top = ((f32(1.0) * f32(2.0)) * rdnw[nz - 1]) * (f32(0.0) - p[nz - 1]) - c1f[nz] * mup
    want[nz] = rw0[nz] + scale * top
    old[nz] = rw0[nz] + (G * top) / msft
    np.testing.assert_array_equal(_words(cp.asnumpy(rw)), _words(want))
    assert np.any(_words(old) != _words(want)), "inputs do not separate the two orders"


@requires_gpu
def test_slow_pgf_half_level_geopotential_sums_left_to_right():
    """WRF calc_php: 0.5*(phb(k)+phb(k+1)+ph(k)+ph(k+1)), left to right
    (module_big_step_utilities_em.F:1261), inside horizontal_pressure_gradient.

    p = 0 isolates the geopotential terms: every pressure difference and the
    interpolated vertical pressure gradient are zero, so the u-face tendency
    is WRF's term 1 plus term 4 with dpn = 0, written out below.
    """
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(11)
    nz, ny, nx = 5, 6, 8
    p = np.zeros((nz, ny, nx), np.float32)
    pb = _rand(rng, (nz, ny, nx), 20000, 90000)
    al = _rand(rng, (nz, ny, nx), -0.01, 0.01)
    alt = _rand(rng, (nz, ny, nx), 0.8, 1.6)
    ph = _rand(rng, (nz + 1, ny, nx), -300, 300)
    phb = np.cumsum(_rand(rng, (nz + 1, ny, nx), 900, 1300), axis=0).astype(np.float32)
    mup = _rand(rng, (ny, nx), -900, 900)
    mub = _rand(rng, (ny, nx), 80000, 95000)
    c1h = _rand(rng, (nz,), 0.2, 1.0)
    c2h = _rand(rng, (nz,), 0.0, 3000.0)
    rdnw = _rand(rng, (nz,), -60, -10)
    fnm = _rand(rng, (nz + 1,), 0.4, 0.6)
    fnp = (f32(1.0) - fnm).astype(np.float32)
    rdx, rdy = f32(1.0 / 3000.0), f32(1.0 / 3000.0)
    half_rdx, half_rdy = f32(0.5) * rdx, f32(0.5) * rdy
    ru = cp.zeros((nz, ny, nx + 1), dtype=cp.float32)
    rv = cp.zeros((nz, ny + 1, nx), dtype=cp.float32)
    kernel = get_kernel("dycore", "slow_pgf")
    n = nz * (ny + 1) * (nx + 1)
    kernel(((n + 255) // 256,), (256,),
           (ru, rv, cp.asarray(p), cp.asarray(pb), cp.asarray(al), cp.asarray(alt),
            cp.asarray(ph), cp.asarray(phb), cp.asarray(mup), cp.asarray(mub),
            cp.asarray(c1h), cp.asarray(c2h), cp.asarray(rdnw), cp.asarray(fnm),
            cp.asarray(fnp), f32(1.5), f32(-0.6), f32(0.1), np.int32(0), f32(1.2),
            f32(-0.2), ru, rv, np.int32(0), rdx, rdy, half_rdx, half_rdy,
            np.int32(0), np.int32(0), np.int32(1), np.int32(nz), np.int32(ny), np.int32(nx)))

    i = np.arange(nx + 1)
    a, b = i % nx, (i - 1 + nx) % nx
    want = np.empty((nz, ny, nx + 1), np.float32)
    old = np.empty_like(want)
    old_mass = np.empty_like(want)
    for k in range(nz):
        # round 3: grid%muu from calc_mu_uv, MU(i)+MU(i-1)+MUB(i)+MUB(i-1).
        muf = f32(0.5) * (((mup[:, a] + mup[:, b]) + mub[:, a]) + mub[:, b])
        layer = c1h[k] * muf + c2h[k]
        bracket = ((ph[k + 1][:, a] - ph[k + 1][:, b]) + ph[k][:, a]) - ph[k][:, b]
        bracket = bracket + (alt[k][:, a] + alt[k][:, b]) * (p[k][:, a] - p[k][:, b])
        bracket = bracket + (al[k][:, a] + al[k][:, b]) * (pb[k][:, a] - pb[k][:, b])
        left = (half_rdx * layer) * bracket

        def php(c, grouped=False):
            if grouped:
                return f32(0.5) * ((phb[k][:, c] + phb[k + 1][:, c]) + (ph[k][:, c] + ph[k + 1][:, c]))
            return f32(0.5) * (((phb[k][:, c] + phb[k + 1][:, c]) + ph[k][:, c]) + ph[k + 1][:, c])

        vertical = rdnw[k] * (f32(0.0) - f32(0.0)) - f32(0.5) * (c1h[k] * mup[:, b] + c1h[k] * mup[:, a])
        want[k] = f32(0.0) - (left + (rdx * (php(a) - php(b))) * vertical)
        old[k] = f32(0.0) - (left + (rdx * (php(a, True) - php(b, True))) * vertical)
        muf_old = f32(0.5) * ((mub[:, a] + mup[:, a]) + (mub[:, b] + mup[:, b]))
        left_old = (half_rdx * (c1h[k] * muf_old + c2h[k])) * bracket
        old_mass[k] = f32(0.0) - (left_old + (rdx * (php(a) - php(b))) * vertical)
    np.testing.assert_array_equal(_words(cp.asnumpy(ru)), _words(want))
    assert np.any(_words(old) != _words(want)), "inputs do not separate the two orders"
    assert np.any(_words(old_mass) != _words(want)), "inputs do not separate the face masses"


@requires_gpu
def test_sixth_order_filter_accumulates_into_the_held_tendency(monkeypatch):
    """WRF sixth_order_diffusion: tendency = tendency + tendency_x + tendency_y
    in place (module_big_step_utilities_em.F:6624); the open strip WRF's loop
    bounds skip keeps the held value."""
    _strict_process()
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core import dycore

    rng = np.random.default_rng(3)
    nz, ny, nx = 4, 14, 15
    field = cp.asarray(_rand(rng, (nz, ny, nx), 280, 320))
    held0 = _rand(rng, (nz, ny, nx), -3e-3, 3e-3)
    mut = cp.asarray(_rand(rng, (ny, nx), 80000, 95000))
    c1 = cp.asarray(_rand(rng, (nz,), 0.2, 1.0))
    c2 = cp.asarray(_rand(rng, (nz,), 0.0, 3000.0))
    buffers = {"smag_rt": cp.asarray(held0), "diff6_m": cp.zeros((nz, ny, nx), cp.float32)}
    state = SimpleNamespace(phb=None, msfu=None, msfv=None,
                            msft=cp.asarray(_rand(rng, (ny, nx), 0.98, 1.03)),
                            scratch=lambda shape, name: buffers[name])
    cfg = SimpleNamespace(diff_6th_opt=2, diff_6th_slopeopt=0, diff_6th_thresh=0.1,
                          dx=3000.0, dy=3000.0, diff_6th_factor=0.12, dt=15.0)
    monkeypatch.setattr(dycore, "_boundary_x", lambda c: True)
    monkeypatch.setattr(dycore, "_boundary_y", lambda c: True)
    monkeypatch.setattr(dycore, "_diff6_factor", lambda c, s: 0.12)
    monkeypatch.setattr(dycore, "_diff6_dt", lambda c, s: 15.0)
    dycore._accumulate_diff6_wrf_order(state, cfg, field, buffers["diff6_m"], mut, c1, c2,
                                       "smag_rt", "")
    got = cp.asnumpy(buffers["smag_rt"])

    in_place = cp.asarray(held0)
    dycore.launch_diff6(field, in_place, mut, c1, c2, 0.12, 15.0, 2, msft=state.msft,
                        bnd_x=True, bnd_y=True)
    want = cp.asnumpy(in_place)
    want[:, :, :3] = held0[:, :, :3]
    want[:, :, -3:] = held0[:, :, -3:]
    want[:, :3, :] = held0[:, :3, :]
    want[:, -3:, :] = held0[:, -3:, :]
    np.testing.assert_array_equal(_words(got), _words(want))

    separate = cp.zeros((nz, ny, nx), cp.float32)
    dycore.launch_diff6(field, separate, mut, c1, c2, 0.12, 15.0, 2, msft=state.msft,
                        bnd_x=True, bnd_y=True)
    old = held0 + cp.asnumpy(separate)
    inner = (slice(None), slice(3, -3), slice(3, -3))
    assert np.any(_words(old[inner]) != _words(want[inner])), "inputs do not separate the orders"
