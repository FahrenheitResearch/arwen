"""Strict-mode kernels in WRF 4.6.1's operation order, round 3, word for word.

The localize pair of the 2026-10-07 combo sweep (WRF built with dump hooks
against WOOF's strict engine, round2/LOCALIZE.md) located the operations
below.  Each test writes WRF's Fortran statement as a chain of float32 NumPy
operations (each correctly rounded, no contraction) and requires every word
of the kernel's output to equal it, and asserts that the order the kernel used
before differs on the same inputs, so no test passes on inputs where the two
orders happen to agree.

Run in a process started with ``GPUWM_WRF_EXACT=1``.
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


def _rand(rng, shape, lo, hi):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@requires_gpu
def test_calc_coef_w_top_row_raises_rdnw_to_the_power_first():
    """calc_coef_w: a(kde) = -2.*cof*rdnw(kde-1)**2*c2a*lid_flag/(...) and
    b = 1.+2.*cof*rdnw(kde-1)**2*c2a/(...), the power before the product
    (module_small_step_em.F).  The left-to-right product moved the top
    alpha and, through the back-substitution, w/ph at levels 44-50."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(31)
    nz, ny, nx = 8, 5, 6
    p = _rand(rng, (nz, ny, nx), 20000, 95000)
    alt = _rand(rng, (nz, ny, nx), 0.8, 3.0)
    mup = _rand(rng, (ny, nx), -900, 900)
    mub = _rand(rng, (ny, nx), 80000, 95000)
    rdn = _rand(rng, (nz + 1,), -60, -10)
    rdnw = _rand(rng, (nz,), -61, -9)
    c1h = _rand(rng, (nz,), 0.0, 1.0)
    c2h = _rand(rng, (nz,), 0.0, 3000.0)
    c1f = _rand(rng, (nz + 1,), 0.0, 1.0)
    c2f = _rand(rng, (nz + 1,), 0.0, 3000.0)
    out = [cp.zeros((nz + (s != 0), ny, nx), cp.float32) for s in range(4)]
    c2a, a, alpha, gam = out
    dtau, epssm = f32(5.0), f32(0.1)
    get_kernel("acoustic", "calc_coefs")(
        ((ny * nx + 255) // 256,), (256,),
        (cp.asarray(p), cp.asarray(alt), cp.asarray(mup), c2a, a, alpha, gam,
         cp.asarray(rdn), cp.asarray(rdnw), cp.asarray(c1h), cp.asarray(c2h),
         cp.asarray(c1f), cp.asarray(c2f), cp.asarray(mub), cp.asarray(p),
         np.int32(0), dtau, epssm, np.int32(0), np.int32(nz), np.int32(ny), np.int32(nx)))
    mut = mub + mup
    cof = (f32(0.5) * dtau * G * (f32(1.0) + epssm))
    cof = cof * cof
    c2a_h = cp.asnumpy(c2a)[nz - 1]
    gam_below = cp.asnumpy(gam)[nz - 1]
    chm = c1h[nz - 1] * mut + c2h[nz - 1]
    sq = rdnw[nz - 1] * rdnw[nz - 1]
    a_top = ((f32(-2.0) * cof) * sq) * c2a_h / (chm * (c1f[nz - 1] * mut + c2f[nz - 1]))
    b_top = f32(1.0) + ((f32(2.0) * cof) * sq) * c2a_h / (chm * (c1f[nz] * mut + c2f[nz]))
    alpha_top = f32(1.0) / (b_top - a_top * gam_below)
    np.testing.assert_array_equal(_words(cp.asnumpy(a)[nz]), _words(a_top))
    np.testing.assert_array_equal(_words(cp.asnumpy(alpha)[nz]), _words(alpha_top))
    a_old = (((f32(-2.0) * cof) * rdnw[nz - 1]) * rdnw[nz - 1]) * c2a_h / (
        chm * (c1f[nz - 1] * mut + c2f[nz - 1]))
    assert np.any(_words(a_old) != _words(a_top)), "inputs do not separate the two orders"


@requires_gpu
def test_surface_w_takes_each_map_factor_into_its_own_coefficient():
    """set_w_surface: msfty*.5*rdy*(...) + msftx*.5*rdx*(...) (module_bc_em.F)."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(32)
    nz, ny, nx = 4, 7, 9
    u = _rand(rng, (nz, ny, nx + 1), -20, 20)
    v = _rand(rng, (nz, ny + 1, nx), -20, 20)
    ht = _rand(rng, (ny, nx), 0, 900)
    msft = _rand(rng, (ny, nx), 0.97, 1.04)
    w = cp.zeros((nz + 1, ny, nx), cp.float32)
    cf1, cf2, cf3 = f32(1.6), f32(-0.8), f32(0.2)
    rdx = f32(1.0) / f32(3000.0)
    half = f32(0.5) * rdx
    get_kernel("surface_w", "set_surface_w")(
        ((ny * nx + 127) // 128,), (128,),
        (cp.asarray(u), cp.asarray(v), cp.asarray(ht), cp.asarray(msft), w,
         cf1, cf2, cf3, half, half, np.int32(1), np.int32(1), np.int32(1),
         np.int32(ny), np.int32(nx)))
    uc = cf1 * u[0] + cf2 * u[1] + cf3 * u[2]
    vc = cf1 * v[0] + cf2 * v[1] + cf3 * v[2]
    jp = np.minimum(np.arange(ny) + 1, ny - 1)
    jm = np.maximum(np.arange(ny) - 1, 0)
    ip = np.minimum(np.arange(nx) + 1, nx - 1)
    im = np.maximum(np.arange(nx) - 1, 0)
    ybr = (ht[jp, :] - ht) * vc[1:, :] + (ht - ht[jm, :]) * vc[:-1, :]
    xbr = (ht[:, ip] - ht) * uc[:, 1:] + (ht - ht[:, im]) * uc[:, :-1]
    want = ((msft * f32(0.5)) * rdx) * ybr + ((msft * f32(0.5)) * rdx) * xbr
    old = (half * ybr + half * xbr) * msft
    np.testing.assert_array_equal(_words(cp.asnumpy(w)[0]), _words(want))
    assert np.any(_words(old) != _words(want)), "inputs do not separate the two orders"


@requires_gpu
def test_slow_buoyancy_moist_cqw_accumulates_species_pairs():
    """calc_cq: qtot = qtot + moist(k) + moist(k-1) species by species and
    cqw = 0.5*qtot; pg_buoy_w then uses cq1 = 1./(1.+cqw), cq2 = cqw*cq1."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(33)
    nz, ny, nx = 7, 6, 8
    p = _rand(rng, (nz, ny, nx), -600, 600)
    pb = _rand(rng, (nz, ny, nx), 20000, 90000)
    mup = _rand(rng, (ny, nx), -900, 900)
    mub = _rand(rng, (ny, nx), 80000, 95000)
    q = [_rand(rng, (nz, ny, nx), 0.0, 3e-2 if s == 0 else 3e-3) for s in range(7)]
    rdn = _rand(rng, (nz + 1,), -60, -10)
    rdnw = _rand(rng, (nz,), -60, -10)
    c1f = _rand(rng, (nz + 1,), 0.1, 1.0)
    c2f = _rand(rng, (nz + 1,), 0.0, 3000.0)
    msft = _rand(rng, (ny, nx), 0.97, 1.04)
    rw0 = _rand(rng, (nz + 1, ny, nx), -5, 5)
    rw = cp.asarray(rw0)
    get_kernel("dycore", "slow_buoyancy")(
        ((nz * ny * nx + 255) // 256,), (256,),
        (rw, cp.asarray(p), cp.asarray(pb), cp.asarray(mup), cp.asarray(mub),
         *[cp.asarray(a) for a in q],
         cp.asarray(rdn), cp.asarray(rdnw), cp.asarray(c1f), cp.asarray(c2f),
         cp.asarray(msft), np.int32(2), np.int32(1), np.int32(1),
         np.int32(nz), np.int32(ny), np.int32(nx)))
    scale = (f32(1.0) / msft) * G

    def cqw_wrf(a, b):
        t = np.zeros_like(q[0][0])
        for s in range(6):
            t = (t + q[s][a]) + q[s][b]
        return f32(0.5) * t

    def cqw_old(a, b):
        tot = lambda k: ((((q[0][k] + q[1][k]) + q[2][k]) + q[3][k]) + q[4][k]) + q[5][k]
        return f32(0.5) * (tot(a) + tot(b))

    want, old = rw0.copy(), rw0.copy()
    for k in range(1, nz + 1):
        for out, cq in ((want, cqw_wrf), (old, cqw_old)):
            if k < nz:
                cqw = cq(k, k - 1)
                cq1 = f32(1.0) / (f32(1.0) + cqw)
                term = (cq1 * rdn[k]) * (p[k] - p[k - 1]) - c1f[k] * mup
            else:
                cqw = cq(nz - 1, nz - 2)
                cq1 = f32(1.0) / (f32(1.0) + cqw)
                term = ((cq1 * f32(2.0)) * rdnw[nz - 1]) * (f32(0.0) - p[nz - 1]) - c1f[nz] * mup
            term = term - (cqw * cq1) * (c1f[k] * mub + c2f[k])
            out[k] = rw0[k] + scale * term
    np.testing.assert_array_equal(_words(cp.asnumpy(rw)), _words(want))
    assert np.any(_words(old) != _words(want)), "inputs do not separate the two orders"


@requires_gpu
def test_exact_frame_carries_wrf_muts_on_the_ring_and_mut_plus_mu_inside():
    """advance_mu_t sets MUTS = MUT + MU'' where it integrates; on the
    specified ring solve_em advances it with spec_bdyupdate(muts, mu_tend,
    dts) from small_step_prep's value instead."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(34)
    nz, ny, nx, sz = 3, 9, 10, 1
    mu_pp0 = _rand(rng, (ny, nx), -40, 40)
    rmu = _rand(rng, (ny, nx), -5, 5)
    mub = _rand(rng, (ny, nx), 80000, 95000)
    mup = _rand(rng, (ny, nx), -900, 900)
    muts0 = _rand(rng, (ny, nx), 80000, 95000)
    mu_pp, muts = cp.asarray(mu_pp0), cp.asarray(muts0)
    th = cp.zeros((nz, ny, nx), cp.float32)
    dtau = f32(3.75)
    get_kernel("acoustic", "advance_exact_frame_mu_t")(
        ((ny * nx + 255) // 256,), (256,),
        (mu_pp, th, cp.asarray(rmu), th, dtau, np.int32(sz), np.int32(0),
         np.int32(nz), np.int32(ny), np.int32(nx), muts, cp.asarray(mub), cp.asarray(mup)))
    ring = np.zeros((ny, nx), bool)
    ring[:sz], ring[-sz:], ring[:, :sz], ring[:, -sz:] = True, True, True, True
    want_muts = np.where(ring, muts0 + dtau * rmu, (mub + mup) + mu_pp0)
    want_mu = np.where(ring, mu_pp0 + dtau * rmu, mu_pp0)
    np.testing.assert_array_equal(_words(cp.asnumpy(muts)), _words(want_muts))
    np.testing.assert_array_equal(_words(cp.asnumpy(mu_pp)), _words(want_mu))
    on_the_fly = (mub + mup) + (mu_pp0 + dtau * rmu)
    assert np.any(_words(on_the_fly[ring]) != _words(want_muts[ring]))


@requires_gpu
def test_specified_ring_phi_groups_field_times_numerator_first():
    """spec_bdyupdate_ph: field*(c1*mu_old+c2)/(c1*muts+c2) + dt*tend/(...)
    + ph_save*((...)/(...) - 1.), with muts the ring carrier."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(35)
    nz, ny, nx, sz = 4, 6, 7, 1
    f = lambda shape, lo, hi: _rand(rng, shape, lo, hi)
    ph_pp0 = f((nz + 1, ny, nx), -40, 40)
    rph = f((nz + 1, ny, nx), -5, 5)
    php = f((nz + 1, ny, nx), -300, 300)
    mup, mub = f((ny, nx), -900, 900), f((ny, nx), 80000, 95000)
    mu_pp, rmu = f((ny, nx), -30, 30), f((ny, nx), -4, 4)
    th_pp, thp = f((nz, ny, nx), -1, 1), f((nz, ny, nx), -5, 5)
    alt, c2a = f((nz, ny, nx), 0.8, 2.0), f((nz, ny, nx), 1e5, 2e5)
    rdnw = f((nz,), -60, -10)
    c1h, c2h = f((nz,), 0, 1), f((nz,), 0, 3000)
    c1f, c2f = f((nz + 1,), 0, 1), f((nz + 1,), 0, 3000)
    muts = f((ny, nx), 80000, 95000)
    ph_pp = cp.asarray(ph_pp0)
    w_pp = cp.zeros((nz + 1, ny, nx), cp.float32)
    p_pp = cp.zeros((nz, ny, nx), cp.float32)
    al_pp = cp.zeros((nz, ny, nx), cp.float32)
    dtau = f32(3.75)
    c = lambda a: cp.asarray(a)
    get_kernel("acoustic", "advance_specified_phi_w")(
        ((ny * nx + 255) // 256,), (256,),
        (ph_pp, w_pp, p_pp, al_pp, c(rph), c(th_pp), c(mup), c(mu_pp), c(mub), c(rmu),
         c(php), c(thp), c(thp), c(alt), c(c2a), c(rdnw), c(c1h), c(c2h), c(c1f), c(c2f),
         dtau, np.int32(sz), np.int32(0), np.int32(nz), np.int32(ny), np.int32(nx), c(muts)))
    mu_old = muts - dtau * rmu
    num = c1f[:, None, None] * mu_old[None] + c2f[:, None, None]
    den = c1f[:, None, None] * muts[None] + c2f[:, None, None]
    want = (((ph_pp0 * num) / den) + ((dtau * rph) / den)) + php * ((num / den) - f32(1.0))
    old = (((ph_pp0 * (num / den))) + ((dtau * rph) / den)) + php * ((num / den) - f32(1.0))
    ring = np.zeros((ny, nx), bool)
    ring[:sz], ring[-sz:], ring[:, :sz], ring[:, -sz:] = True, True, True, True
    got = cp.asnumpy(ph_pp)
    np.testing.assert_array_equal(_words(got[:, ring]), _words(want[:, ring]))
    np.testing.assert_array_equal(_words(got[:, ~ring]), _words(ph_pp0[:, ~ring]))
    assert np.any(_words(old[:, ring]) != _words(want[:, ring]))
    # calc_p_rho on the ring divides by the carrier's mass as well.
    chm = c1h[:, None, None] * muts[None] + c2h[:, None, None]
    al = (f32(-1.0) / chm) * (alt * (c1h[:, None, None] * mu_pp[None])
                              + rdnw[:, None, None] * (got[1:] - got[:-1]))
    np.testing.assert_array_equal(_words(cp.asnumpy(al_pp)[:, ring]), _words(al[:, ring]))


@requires_gpu
def test_finalize_writes_only_specified_rows_from_wrf_muts():
    """spec_bdy_final: field = xmsf*bfield/xmu on the specified rows only,
    xmu = c1*mu+c2 with mu = grid%muts (or calc_mu_uv_1 faces for u/v)."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(36)
    nz, ny, nx, width, sz = 3, 8, 9, 5, 1
    target0 = _rand(rng, (nz, ny, nx + 1), -20, 20)          # u
    muts = _rand(rng, (ny, nx), 80000, 95000)
    c1h, c2h = _rand(rng, (nz,), 0, 1), _rand(rng, (nz,), 0, 3000)
    msfu = _rand(rng, (ny, nx + 1), 0.97, 1.04)
    sides = {"w": _rand(rng, (nz, ny, width), -1e6, 1e6), "e": _rand(rng, (nz, ny, width), -1e6, 1e6),
             "s": _rand(rng, (nz, width, nx + 1), -1e6, 1e6), "n": _rand(rng, (nz, width, nx + 1), -1e6, 1e6)}
    tends = {k: _rand(rng, v.shape, -50, 50) for k, v in sides.items()}
    dtbc = f32(15.0)
    target = cp.asarray(target0)
    c = lambda a: cp.asarray(a)
    dummy = cp.zeros((ny, nx), cp.float32)
    args = [target, dummy, c(_rand(rng, (ny, nx), 80000, 95000)), dummy, c(np.zeros(nz, np.float32)), target,
            c(c1h), c(c2h), c(c1h), c(c2h), dummy, c(msfu), dummy]
    for k in ("w", "e", "s", "n"):
        args += [c(sides[k]), c(tends[k])]
    args += [dtbc, np.int32(width), np.int32(sz), np.int32(1), np.int32(0), np.int32(0),
             np.int32(nz), np.int32(ny), np.int32(nx + 1), np.int32(ny), np.int32(nx), c(muts)]
    get_kernel("lbc_state", "finalize_state_field")(
        ((nz * ny * (nx + 1) + 255) // 256,), (256,), tuple(args))
    got = cp.asnumpy(target)
    want = target0.copy()
    i = np.arange(nx + 1)
    muus = f32(0.5) * (muts[:, np.minimum(i, nx - 1)] + muts[:, np.maximum(i - 1, 0)])
    xmu = c1h[:, None, None] * muus[None] + c2h[:, None, None]

    def put(j, ii, value):
        want[:, j, ii] = (msfu[j, ii][None] * value) / xmu[:, j, ii]

    put(0, slice(None), sides["s"][:, 0, :] + dtbc * tends["s"][:, 0, :])
    put(ny - 1, slice(None), sides["n"][:, 0, :] + dtbc * tends["n"][:, 0, :])
    for j in range(1, ny - 1):
        want[:, j, 0] = (msfu[j, 0] * (sides["w"][:, j, 0] + dtbc * tends["w"][:, j, 0])) / xmu[:, j, 0]
        want[:, j, nx] = (msfu[j, nx] * (sides["e"][:, j, 0] + dtbc * tends["e"][:, j, 0])) / xmu[:, j, nx]
    np.testing.assert_array_equal(_words(got), _words(want))


@requires_gpu
def test_advance_uv_adds_external_mode_damping_last():
    """advance_uv: u = u - dts*cqu*dpxy + (c1h*mudf_xy), mudf_xy =
    -emdiv*dx*(MUDF(i)-MUDF(i-1))/msfuy -- inside the statement, last."""
    _strict_process()
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    rng = np.random.default_rng(37)
    nz, ny, nx = 5, 6, 7
    c = lambda a: cp.asarray(a)
    shapes = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx), "m": (nz, ny, nx),
              "f": (nz + 1, ny, nx), "c": (ny, nx)}
    ru, rv = _rand(rng, shapes["u"], -5, 5), _rand(rng, shapes["v"], -5, 5)
    p_pp, ph_pp = _rand(rng, shapes["m"], -50, 50), _rand(rng, shapes["f"], -20, 20)
    php, phb = _rand(rng, shapes["f"], -300, 300), np.cumsum(_rand(rng, shapes["f"], 900, 1300), 0).astype(np.float32)
    alt, al_pp = _rand(rng, shapes["m"], 0.8, 2.0), _rand(rng, shapes["m"], -1e-3, 1e-3)
    pb = _rand(rng, shapes["m"], 20000, 90000)
    mup, mu_pp, mub = _rand(rng, shapes["c"], -900, 900), _rand(rng, shapes["c"], -30, 30), _rand(rng, shapes["c"], 8e4, 9.5e4)
    c1h, c2h = _rand(rng, (nz,), 0, 1), _rand(rng, (nz,), 0, 3000)
    fnm = _rand(rng, (nz + 1,), 0.4, 0.6)
    fnp = (f32(1) - fnm).astype(np.float32)
    rdnw = _rand(rng, (nz,), -60, -10)
    mudf = _rand(rng, shapes["c"], -2, 2)
    msfu, msfv = _rand(rng, (ny, nx + 1), 0.97, 1.04), _rand(rng, (ny + 1, nx), 0.97, 1.04)
    xscale = f32(-f32(0.01)) * f32(3000.0)
    u0, v0 = _rand(rng, shapes["u"], -30, 30), _rand(rng, shapes["v"], -30, 30)

    def run(u_start, v_start, use):
        u, v = c(u_start), c(v_start)
        get_kernel("acoustic", "advance_uv")(
            ((nz * (ny + 1) * (nx + 1) + 255) // 256,), (256,),
            (u, v, c(ru), c(rv), c(p_pp), c(p_pp), c(ph_pp), c(php), c(phb), c(alt), c(al_pp),
             c(pb), c(mup), c(mu_pp), c(mub), c(c1h), c(c2h), c(fnm), c(fnp), c(rdnw),
             u, v, np.int32(0), f32(1.5), f32(-0.6), f32(0.1), np.int32(0),
             f32(1 / 3000.0), f32(1 / 3000.0), f32(3.75), f32(0.0),
             np.int32(1), np.int32(1), np.int32(nz), np.int32(ny), np.int32(nx),
             c(mudf), c(msfu), c(msfv), xscale, xscale, np.int32(use), np.int32(1)))
        return cp.asnumpy(u), cp.asnumpy(v)

    plain_u, _ = run(u0, v0, 0)
    got_u, _ = run(u0, v0, 1)
    i = np.arange(nx + 1)
    gx = (xscale * (mudf[:, i % nx] - mudf[:, (i - 1 + nx) % nx])) / msfu
    inc = c1h[:, None, None] * gx[None]
    want = plain_u.copy()
    inner = (slice(None), slice(1, ny - 1), slice(1, nx))
    want[inner] = plain_u[inner] + inc[inner]
    np.testing.assert_array_equal(_words(got_u), _words(want))
    before_u, _ = run(u0 + inc, v0, 0)                  # the default placement
    assert np.any(_words(before_u[inner]) != _words(want[inner]))


@requires_gpu
def test_analysis_frame_reflectivity_is_wrfs_zero_array():
    """WRF writes refl_10cm's initial array, all zeros, at the analysis
    frame (stock 4.6.1 A088 frame 0 hashes to an all-zero array)."""
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.refl import analysis_refl_10cm

    state = SimpleNamespace(qv=cp.ones((3, 4, 5), cp.float32))
    refl = analysis_refl_10cm(state)
    assert refl.shape == (3, 4, 5) and refl.dtype == cp.float32
    assert not bool(cp.any(refl.view(cp.uint32)))
