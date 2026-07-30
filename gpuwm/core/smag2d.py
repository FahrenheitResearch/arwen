"""Small host-side WRF Smagorinsky algebra helpers.

Production remains CUDA-only.  These routines let source-level CPU tests pin
WRF v4.6.1 terrain-slope deformation/K reduction, tensor momentum stresses,
and metric-aware scalar fluxes that the historical flat kernel could not
represent.  They intentionally do not mirror the CUDA loop implementation.
"""

from __future__ import annotations

import math

import numpy as np

from gpuwm.core.constants import G


def wrf_d11_algebra(*, du_dx_hat: float, du_deta_hat: float,
                    zx: float, rdzw: float,
                    msftx: float = 1.0, msfty: float = 1.0) -> float:
    """One WRF terrain-coordinate D11 value (source lines 176-215).

    The arguments are the already stagger-averaged quantities at a mass
    point.  Keeping this reduction separate makes the sloping-grid regression
    test the coordinate transform itself instead of injecting its answer.
    """
    return (2.0 * float(msftx) * float(msfty)
            * (float(du_dx_hat)
               - float(zx) * float(rdzw) * float(du_deta_hat)))


def wrf_smag2d_km_algebra(*, d11: float, d22: float, d12: float,
                          dx: float, dy: float, msftx: float, msfty: float,
                          c_s: float, slope_alpha: float = 1.0) -> float:
    """One WRF ``smag2d_km`` point (module_diffusion_em.F:2004-2038).

    ``d12`` is the four-corner mean used by the mass-point deformation
    invariant and ``slope_alpha`` is WRF's already-computed ``alpha``.  The
    high-deformation branch applies ``alpha**2`` exactly as the source does.
    """
    mlen = math.sqrt((float(dx) / float(msftx)) *
                     (float(dy) / float(msfty)))
    strain = math.sqrt(0.25 * (float(d11) - float(d22)) ** 2
                       + float(d12) ** 2)
    km = min(float(c_s) ** 2 * mlen ** 2 * strain, 10.0 * mlen)
    alpha = max(float(slope_alpha), 1.0)
    def_limit = max(10.0 / mlen, 1.0e-3)
    return km / (alpha ** 2 if strain > def_limit else alpha)


def wrf_flat_u_cross_stress_tendency(*, v, km, rho, dx: float, dy: float):
    """Flat-grid u forcing from WRF's off-diagonal ``tau12`` stress.

    This is the ``u=0`` reduction of ``cal_titau_12_21`` and
    ``horizontal_diffusion_u_2`` (module_diffusion_em.F:5456-5594 and
    :3233-3313).  It is deliberately narrow: the full production operator is
    in ``kernels/smag2d.cu``.
    """
    v = np.asarray(v, dtype=np.float64)
    km = np.asarray(km, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    ny, nx = km.shape
    if v.shape != (ny + 1, nx) or rho.shape != (ny, nx):
        raise ValueError("v, km, rho shapes do not describe one C grid")

    # D12 at vorticity points: dv/dx for this u=0, flat-grid reduction.
    d12 = (v[:ny] - np.roll(v[:ny], 1, axis=1)) / float(dx)
    rho_corner = 0.25 * (rho + np.roll(rho, 1, axis=0)
                         + np.roll(rho, 1, axis=1)
                         + np.roll(np.roll(rho, 1, axis=0), 1, axis=1))
    km_corner = 0.25 * (km + np.roll(km, 1, axis=0)
                        + np.roll(km, 1, axis=1)
                        + np.roll(np.roll(km, 1, axis=0), 1, axis=1))
    tau12 = -rho_corner * km_corner * d12
    # u lives between mass columns; return the core (non-duplicate) faces.
    return (np.roll(tau12, -1, axis=0) - tau12) / float(dy)


def wrf_periodic_scalar_metric_tendency(
        *, field, kh, rho, phi, dnw, dn, fnm, fnp,
        cf1: float, cf2: float, cf3: float, dx: float, dy: float,
        msft=None, msfu=None, msfv=None,
        return_components: bool = False):
    """Float64 WRF ``horizontal_diffusion_s`` on a periodic grid.

    This is an authority reduction of ``module_diffusion_em.F:3711-3999``
    for the periodic test grids.  It retains arbitrary positive C-grid map
    factors, total-geopotential ``rdzw/zx/zy`` metrics, face density/K
    averages, WRF bottom/top extrapolation, and the vertical metric-flux
    correction.  Omitted map factors default to one for the retained flat
    host gates.  It is intentionally independent of the CUDA traversal.

    ``return_components=True`` returns ``(total, horizontal, metric)`` so
    tests can distinguish the ordinary horizontal divergence from WRF's
    sloping-coordinate correction.  The two components sum to ``total``.
    """
    field = np.asarray(field, dtype=np.float64)
    kh = np.asarray(kh, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    dnw = np.asarray(dnw, dtype=np.float64)
    dn = np.asarray(dn, dtype=np.float64)
    fnm = np.asarray(fnm, dtype=np.float64)
    fnp = np.asarray(fnp, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("field must have shape (nz, ny, nx)")
    nz, ny, nx = field.shape
    if nz < 3:
        raise ValueError("WRF scalar boundary extrapolation needs nz >= 3")
    if kh.shape != field.shape or rho.shape != field.shape:
        raise ValueError("kh and rho must match field")
    if phi.shape != (nz + 1, ny, nx):
        raise ValueError("phi must have shape (nz+1, ny, nx)")
    if any(a.shape != (nz,) for a in (dnw, dn, fnm, fnp)):
        raise ValueError("dnw, dn, fnm, and fnp must have length nz")

    msft = (np.ones((ny, nx), dtype=np.float64) if msft is None
            else np.asarray(msft, dtype=np.float64))
    msfu = (np.ones((ny, nx + 1), dtype=np.float64) if msfu is None
            else np.asarray(msfu, dtype=np.float64))
    msfv = (np.ones((ny + 1, nx), dtype=np.float64) if msfv is None
            else np.asarray(msfv, dtype=np.float64))
    expected_maps = ((msft, (ny, nx), "msft"),
                     (msfu, (ny, nx + 1), "msfu"),
                     (msfv, (ny + 1, nx), "msfv"))
    for array, shape, name in expected_maps:
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(array).all() or array.min() <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    rdx, rdy = 1.0 / float(dx), 1.0 / float(dy)
    rdzw = G / (phi[1:] - phi[:-1])
    zx = rdx * (phi - np.roll(phi, 1, axis=2)) / G
    zy = rdy * (phi - np.roll(phi, 1, axis=1)) / G

    def at_w_faces(face_pairs):
        out = np.zeros((nz + 1, ny, nx), dtype=np.float64)
        out[0] = 0.5 * (float(cf1) * face_pairs[0]
                        + float(cf2) * face_pairs[1]
                        + float(cf3) * face_pairs[2])
        for kw in range(1, nz):
            out[kw] = 0.5 * (fnm[kw] * face_pairs[kw]
                             + fnp[kw] * face_pairs[kw - 1])
        top = 0.5 * dnw[-1] / dn[-1]
        out[nz] = 0.5 * (face_pairs[-1]
                         + top * (face_pairs[-1] - face_pairs[-2]))
        return out

    left = np.roll(field, 1, axis=2)
    below = np.roll(field, 1, axis=1)
    field_xw = at_w_faces(left + field)
    field_yw = at_w_faces(below + field)

    rho_left, kh_left = (np.roll(a, 1, axis=2) for a in (rho, kh))
    rdzw_left = np.roll(rdzw, 1, axis=2)
    rdzu = 2.0 / (1.0 / rdzw + 1.0 / rdzw_left)
    zx_face = 0.5 * (zx[:-1] + zx[1:])
    h1 = (-0.5 * (rho_left + rho) * 0.5 * (kh_left + kh)
          * (rdx * (field - left)
             - zx_face * (field_xw[1:] - field_xw[:-1]) * rdzu))
    h1 *= msfu[None, :, :nx]

    rho_below, kh_below = (np.roll(a, 1, axis=1) for a in (rho, kh))
    rdzw_below = np.roll(rdzw, 1, axis=1)
    rdzv = 2.0 / (1.0 / rdzw + 1.0 / rdzw_below)
    zy_face = 0.5 * (zy[:-1] + zy[1:])
    h2 = (-0.5 * (rho_below + rho) * 0.5 * (kh_below + kh)
          * (rdy * (field - below)
             - zy_face * (field_yw[1:] - field_yw[:-1]) * rdzv))
    h2 *= msfv[None, :ny, :]

    h1w = np.zeros((nz + 1, ny, nx), dtype=np.float64)
    h2w = np.zeros_like(h1w)
    for kw in range(1, nz):
        h1w[kw] = 0.5 * (
            fnm[kw] * (h1[kw] + np.roll(h1[kw], -1, axis=1))
            + fnp[kw] * (h1[kw - 1]
                         + np.roll(h1[kw - 1], -1, axis=1)))
        h2w[kw] = 0.5 * (
            fnm[kw] * (h2[kw] + np.roll(h2[kw], -1, axis=0))
            + fnp[kw] * (h2[kw - 1]
                         + np.roll(h2[kw - 1], -1, axis=0)))

    zx_mass = 0.25 * (zx[:-1] + np.roll(zx[:-1], -1, axis=2)
                      + zx[1:] + np.roll(zx[1:], -1, axis=2))
    zy_mass = 0.25 * (zy[:-1] + np.roll(zy[:-1], -1, axis=1)
                      + zy[1:] + np.roll(zy[1:], -1, axis=1))
    divh = msft[None] * (
        rdx * (np.roll(h1, -1, axis=2) - h1)
        + rdy * (np.roll(h2, -1, axis=1) - h2))
    div_metric = msft[None] * (
        zx_mass * (h1w[1:] - h1w[:-1]) * rdzw
        + zy_mass * (h2w[1:] - h2w[:-1]) * rdzw)
    factor = G / (dnw[:, None, None] * rdzw)
    horizontal = factor * divh
    metric = -factor * div_metric
    total = horizontal + metric
    return (total, horizontal, metric) if return_components else total


def wrf_periodic_momentum_authority(
        *, u, v, w, phi, rho, msft, msfu, msfv, fnm, fnp, dn, dnw,
        cf1: float, cf2: float, cf3: float, dx: float, dy: float,
        c_s: float):
    """Float64 WRF ``diff_opt=2, km_opt=4`` momentum authority.

    This is a direct, host-only reduction of WRF v4.6.1
    ``module_diffusion_em.F``: ``cal_deform_and_div``, ``smag2d_km``,
    ``cal_titau_11_22_33``, ``cal_titau_12_21``, and
    ``horizontal_diffusion_[u,v,w]_2``.  It is deliberately independent of
    the CUDA traversal and covers periodic 3-D terrain coordinates with
    arbitrary positive C-grid map factors.  The compact GPU acceptance
    fixture uses it pointwise; production never calls this routine.

    Returned ``ru_cross``/``rv_cross`` isolate the off-diagonal tau12
    contribution, while ``rw_x``/``rw_y`` isolate the tau13 and tau23
    contributions.  These make a missing cross-component stress observable
    even if another component happens to compensate in the total.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    msft = np.asarray(msft, dtype=np.float64)
    msfu = np.asarray(msfu, dtype=np.float64)
    msfv = np.asarray(msfv, dtype=np.float64)
    fnm = np.asarray(fnm, dtype=np.float64)
    fnp = np.asarray(fnp, dtype=np.float64)
    dn = np.asarray(dn, dtype=np.float64)
    dnw = np.asarray(dnw, dtype=np.float64)
    if rho.ndim != 3:
        raise ValueError("rho must have shape (nz, ny, nx)")
    nz, ny, nx = rho.shape
    expected = {
        "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx), "phi": (nz + 1, ny, nx),
        "msft": (ny, nx), "msfu": (ny, nx + 1),
        "msfv": (ny + 1, nx),
    }
    actual = {"u": u.shape, "v": v.shape, "w": w.shape,
              "phi": phi.shape, "msft": msft.shape,
              "msfu": msfu.shape, "msfv": msfv.shape}
    bad = {name: (actual[name], shape) for name, shape in expected.items()
           if actual[name] != shape}
    if bad:
        raise ValueError(f"arrays do not describe one periodic C grid: {bad}")
    if any(a.shape != (nz,) for a in (fnm, fnp, dn, dnw)):
        raise ValueError("fnm, fnp, dn, and dnw must have length nz")
    if nz < 3 or min(nx, ny) < 3:
        raise ValueError("WRF deformation authority needs nz,nx,ny >= 3")
    if (not np.isfinite(msft).all() or not np.isfinite(msfu).all()
            or not np.isfinite(msfv).all()
            or min(msft.min(), msfu.min(), msfv.min()) <= 0.0):
        raise ValueError("map factors must be finite and positive")

    rdx, rdy = 1.0 / float(dx), 1.0 / float(dy)

    def ix(i):
        return i % nx

    def iy(j):
        return j % ny

    def ph(kw, j, i):
        return phi[min(max(kw, 0), nz), iy(j), ix(i)]

    def rdzw(k, j, i):
        kk = min(max(k, 0), nz - 1)
        return G / (ph(kk + 1, j, i) - ph(kk, j, i))

    def rdz(kw, j, i):
        if kw <= 0:
            return 2.0 * G / (ph(1, j, i) - ph(0, j, i))
        if kw >= nz:
            return 2.0 * G / (ph(nz, j, i) - ph(nz - 1, j, i))
        return 2.0 * G / (ph(kw + 1, j, i) - ph(kw - 1, j, i))

    def zx(kw, j, iface):
        return rdx * (ph(kw, j, iface) - ph(kw, j, iface - 1)) / G

    def zy(kw, jface, i):
        return rdy * (ph(kw, jface, i) - ph(kw, jface - 1, i)) / G

    def rh(k, j, i):
        return rho[k, iy(j), ix(i)]

    def uhat(k, j, iface):
        jj, ii = iy(j), iface % nx
        return u[k, jj, ii] / msfu[jj, ii]

    def vhat(k, jface, i):
        jj, ii = jface % ny, ix(i)
        return v[k, jj, ii] / msfv[jj, ii]

    def what(kw, j, i):
        jj, ii = iy(j), ix(i)
        return w[kw, jj, ii] / msft[jj, ii]

    def full_weights(kw, p0, p1, p2, plast, pprev, pcur, pbelow):
        if kw <= 0:
            return float(cf1) * p0 + float(cf2) * p1 + float(cf3) * p2
        if kw >= nz:
            cft2 = -0.5 * dnw[-1] / dn[-1]
            return (1.0 - cft2) * plast + cft2 * pprev
        return fnm[kw] * pcur + fnp[kw] * pbelow

    def vertical_pair(kw, getter_a, getter_b):
        p0 = getter_a(0) + getter_b(0)
        p1 = getter_a(1) + getter_b(1)
        p2 = getter_a(2) + getter_b(2)
        plast = getter_a(nz - 1) + getter_b(nz - 1)
        pprev = getter_a(nz - 2) + getter_b(nz - 2)
        kc = min(kw, nz - 1)
        kb = max(kc - 1, 0)
        pcur = getter_a(kc) + getter_b(kc)
        pbelow = getter_a(kb) + getter_b(kb)
        return 0.5 * full_weights(
            kw, p0, p1, p2, plast, pprev, pcur, pbelow)

    def u_w_xcenter(kw, j, i):
        return vertical_pair(
            kw, lambda k: uhat(k, j, i),
            lambda k: uhat(k, j, i + 1))

    def v_w_ycenter(kw, j, i):
        return vertical_pair(
            kw, lambda k: vhat(k, j, i),
            lambda k: vhat(k, j + 1, i))

    def u_w_corner(kw, j, i):
        return vertical_pair(
            kw, lambda k: uhat(k, j - 1, i),
            lambda k: uhat(k, j, i))

    def v_w_corner(kw, j, i):
        return vertical_pair(
            kw, lambda k: vhat(k, j, i - 1),
            lambda k: vhat(k, j, i))

    def defor11(k, j, i):
        tmpzx = 0.25 * (zx(k, j, i) + zx(k, j, i + 1)
                        + zx(k + 1, j, i) + zx(k + 1, j, i + 1))
        slope = ((u_w_xcenter(k + 1, j, i) - u_w_xcenter(k, j, i))
                 * tmpzx * rdzw(k, j, i))
        mm = msft[iy(j), ix(i)]
        return (2.0 * mm * mm
                * (rdx * (uhat(k, j, i + 1) - uhat(k, j, i)) - slope))

    def defor22(k, j, i):
        tmpzy = 0.25 * (zy(k, j, i) + zy(k, j + 1, i)
                        + zy(k + 1, j, i) + zy(k + 1, j + 1, i))
        slope = ((v_w_ycenter(k + 1, j, i) - v_w_ycenter(k, j, i))
                 * tmpzy * rdzw(k, j, i))
        mm = msft[iy(j), ix(i)]
        return (2.0 * mm * mm
                * (rdy * (vhat(k, j + 1, i) - vhat(k, j, i)) - slope))

    def defor12(k, j, i):
        iu, jm, jc = i % nx, iy(j - 1), iy(j)
        iv, im, jv = ix(i), ix(i - 1), j % ny
        mm = (0.25 * (msfu[jm, iu] + msfu[jc, iu])
              * (msfv[jv, im] + msfv[jv, iv]))
        rr = (rdzw(k, j, i) + rdzw(k, j, i - 1)
              + rdzw(k, j - 1, i - 1) + rdzw(k, j - 1, i))
        tmpzy = 0.25 * (zy(k, j, i - 1) + zy(k, j, i)
                        + zy(k + 1, j, i - 1) + zy(k + 1, j, i))
        uslope = ((u_w_corner(k + 1, j, i) - u_w_corner(k, j, i))
                  * 0.25 * tmpzy * rr)
        tmpzx = 0.25 * (zx(k, j - 1, i) + zx(k, j, i)
                        + zx(k + 1, j - 1, i) + zx(k + 1, j, i))
        vslope = ((v_w_corner(k + 1, j, i) - v_w_corner(k, j, i))
                  * 0.25 * tmpzx * rr)
        return mm * (
            rdy * (uhat(k, j, i) - uhat(k, j - 1, i)) - uslope
            + rdx * (vhat(k, j, i) - vhat(k, j, i - 1)) - vslope)

    d11 = np.empty((nz, ny, nx), dtype=np.float64)
    d22 = np.empty_like(d11)
    d12 = np.empty_like(d11)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                d11[k, j, i] = defor11(k, j, i)
                d22[k, j, i] = defor22(k, j, i)
                d12[k, j, i] = defor12(k, j, i)

    def at(a, k, j, i):
        return a[k, iy(j), ix(i)]

    km = np.empty_like(d11)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                d12_mean = 0.25 * (
                    at(d12, k, j, i) + at(d12, k, j + 1, i)
                    + at(d12, k, j, i + 1) + at(d12, k, j + 1, i + 1))
                strain = math.sqrt(
                    0.25 * (d11[k, j, i] - d22[k, j, i]) ** 2
                    + d12_mean ** 2)
                map_factor = msft[j, i]
                dxm, dym = float(dx) / map_factor, float(dy) / map_factor
                mlen = math.sqrt(dxm * dym)
                value = min(float(c_s) ** 2 * mlen ** 2 * strain,
                            10.0 * mlen)
                sx = (0.25 * (abs(zx(k, j, i)) + abs(zx(k, j, i + 1))
                              + abs(zx(k + 1, j, i))
                              + abs(zx(k + 1, j, i + 1)))
                      * rdzw(k, j, i) * dxm)
                sy = (0.25 * (abs(zy(k, j, i)) + abs(zy(k, j + 1, i))
                              + abs(zy(k + 1, j, i))
                              + abs(zy(k + 1, j + 1, i)))
                      * rdzw(k, j, i) * dym)
                alpha = max(math.sqrt(sx * sx + sy * sy), 1.0)
                def_limit = max(10.0 / mlen, 1.0e-3)
                km[k, j, i] = value / (
                    alpha * alpha if strain > def_limit else alpha)

    def tau11(k, j, i):
        return -rh(k, j, i) * at(km, k, j, i) * at(d11, k, j, i)

    def tau22(k, j, i):
        return -rh(k, j, i) * at(km, k, j, i) * at(d22, k, j, i)

    def tau12(k, j, i):
        rhoavg = 0.25 * (rh(k, j, i) + rh(k, j, i - 1)
                         + rh(k, j - 1, i - 1) + rh(k, j - 1, i))
        kavg = 0.25 * (at(km, k, j, i) + at(km, k, j, i - 1)
                       + at(km, k, j - 1, i - 1)
                       + at(km, k, j - 1, i))
        return -rhoavg * kavg * at(d12, k, j, i)

    def tau11_uavg(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        return 0.5 * (
            fnm[kw] * (tau11(kw, j, i - 1) + tau11(kw, j, i))
            + fnp[kw] * (tau11(kw - 1, j, i - 1)
                         + tau11(kw - 1, j, i)))

    def tau12_uavg(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        return 0.5 * (
            fnm[kw] * (tau12(kw, j + 1, i) + tau12(kw, j, i))
            + fnp[kw] * (tau12(kw - 1, j + 1, i)
                         + tau12(kw - 1, j, i)))

    def tau12_vavg(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        return 0.5 * (
            fnm[kw] * (tau12(kw, j, i + 1) + tau12(kw, j, i))
            + fnp[kw] * (tau12(kw - 1, j, i + 1)
                         + tau12(kw - 1, j, i)))

    def tau22_vavg(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        return 0.5 * (
            fnm[kw] * (tau22(kw, j - 1, i) + tau22(kw, j, i))
            + fnp[kw] * (tau22(kw - 1, j - 1, i)
                         + tau22(kw - 1, j, i)))

    def w_xavg(k, j, i):
        return 0.25 * (what(k, j, i) + what(k + 1, j, i)
                       + what(k, j, i - 1) + what(k + 1, j, i - 1))

    def w_yavg(k, j, i):
        return 0.25 * (what(k, j, i) + what(k + 1, j, i)
                       + what(k, j - 1, i) + what(k + 1, j - 1, i))

    def defor13(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        rz = 0.5 * (rdz(kw, j, i) + rdz(kw, j, i - 1))
        slope = ((w_xavg(kw, j, i) - w_xavg(kw - 1, j, i))
                 * zx(kw, j, i) * rz)
        jj, iface = iy(j), i % nx
        dwdx = msfu[jj, iface] ** 2 * (
            rdx * (what(kw, j, i) - what(kw, j, i - 1)) - slope)
        dudz = (u[kw, jj, iface] - u[kw - 1, jj, iface]) * rz
        return dwdx + dudz

    def defor23(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        rz = 0.5 * (rdz(kw, j, i) + rdz(kw, j - 1, i))
        slope = ((w_yavg(kw, j, i) - w_yavg(kw - 1, j, i))
                 * zy(kw, j, i) * rz)
        jface, ii = j % ny, ix(i)
        dwdy = msfv[jface, ii] ** 2 * (
            rdy * (what(kw, j, i) - what(kw, j - 1, i)) - slope)
        dvdz = (v[kw, jface, ii] - v[kw - 1, jface, ii]) * rz
        return dwdy + dvdz

    def tau13(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        rhoavg = 0.5 * (
            fnm[kw] * (rh(kw, j, i - 1) + rh(kw, j, i))
            + fnp[kw] * (rh(kw - 1, j, i - 1) + rh(kw - 1, j, i)))
        kavg = 0.5 * (
            fnm[kw] * (at(km, kw, j, i - 1) + at(km, kw, j, i))
            + fnp[kw] * (at(km, kw - 1, j, i - 1)
                         + at(km, kw - 1, j, i)))
        return -rhoavg * kavg * defor13(kw, j, i)

    def tau23(kw, j, i):
        if kw <= 0 or kw >= nz:
            return 0.0
        rhoavg = 0.5 * (
            fnm[kw] * (rh(kw, j - 1, i) + rh(kw, j, i))
            + fnp[kw] * (rh(kw - 1, j - 1, i) + rh(kw - 1, j, i)))
        kavg = 0.5 * (
            fnm[kw] * (at(km, kw, j - 1, i) + at(km, kw, j, i))
            + fnp[kw] * (at(km, kw - 1, j - 1, i)
                         + at(km, kw - 1, j, i)))
        return -rhoavg * kavg * defor23(kw, j, i)

    def tau13_mavg(k, j, i):
        return 0.25 * (tau13(k, j, i) + tau13(k, j, i + 1)
                       + tau13(k + 1, j, i) + tau13(k + 1, j, i + 1))

    def tau23_mavg(k, j, i):
        return 0.25 * (tau23(k, j, i) + tau23(k, j + 1, i)
                       + tau23(k + 1, j, i) + tau23(k + 1, j + 1, i))

    ru = np.zeros((nz, ny, nx + 1), dtype=np.float64)
    rv = np.zeros((nz, ny + 1, nx), dtype=np.float64)
    rw = np.zeros((nz + 1, ny, nx), dtype=np.float64)
    ru_cross = np.zeros_like(ru)
    rv_cross = np.zeros_like(rv)
    rw_x = np.zeros_like(rw)
    rw_y = np.zeros_like(rw)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx + 1):
                ic, im = ix(i), ix(i - 1)
                msf = msfu[iy(j), i % nx]
                tmpdz = 0.5 * (1.0 / rdzw(k, j, ic)
                                + 1.0 / rdzw(k, j, im))
                zx_u = 0.5 * (zx(k, j, i) + zx(k + 1, j, i))
                zy_u = 0.125 * (
                    zy(k, j, im) + zy(k, j, ic)
                    + zy(k, j + 1, im) + zy(k, j + 1, ic)
                    + zy(k + 1, j, im) + zy(k + 1, j, ic)
                    + zy(k + 1, j + 1, im) + zy(k + 1, j + 1, ic))
                normal_h = msf * rdx * (
                    tau11(k, j, ic) - tau11(k, j, im))
                cross_h = msf * rdy * (
                    tau12(k, j + 1, i) - tau12(k, j, i))
                normal_z = msf * zx_u * (
                    tau11_uavg(k + 1, j, i)
                    - tau11_uavg(k, j, i)) / tmpdz
                cross_z = msf * zy_u * (
                    tau12_uavg(k + 1, j, i)
                    - tau12_uavg(k, j, i)) / tmpdz
                factor = G * tmpdz / dnw[k]
                ru[k, j, i] = factor * (
                    normal_h + cross_h - normal_z - cross_z)
                ru_cross[k, j, i] = factor * (cross_h - cross_z)
            for i in range(nx):
                jc, jm = iy(j), iy(j - 1)
                msf = msfv[j % ny, ix(i)]
                tmpdz = 0.5 * (1.0 / rdzw(k, jc, i)
                                + 1.0 / rdzw(k, jm, i))
                zx_v = 0.125 * (
                    zx(k, jm, i) + zx(k, jm, i + 1)
                    + zx(k, jc, i) + zx(k, jc, i + 1)
                    + zx(k + 1, jm, i) + zx(k + 1, jm, i + 1)
                    + zx(k + 1, jc, i) + zx(k + 1, jc, i + 1))
                zy_v = 0.5 * (zy(k, j, i) + zy(k + 1, j, i))
                normal_h = msf * rdy * (
                    tau22(k, jc, i) - tau22(k, jm, i))
                cross_h = msf * rdx * (
                    tau12(k, j, i + 1) - tau12(k, j, i))
                cross_z = msf * zx_v * (
                    tau12_vavg(k + 1, j, i)
                    - tau12_vavg(k, j, i)) / tmpdz
                normal_z = msf * zy_v * (
                    tau22_vavg(k + 1, j, i)
                    - tau22_vavg(k, j, i)) / tmpdz
                factor = G * tmpdz / dnw[k]
                rv[k, j, i] = factor * (
                    normal_h + cross_h - cross_z - normal_z)
                rv_cross[k, j, i] = factor * (cross_h - cross_z)
        rv[k, ny] = rv[k, 0]
        rv_cross[k, ny] = rv_cross[k, 0]

    for kw in range(1, nz):
        for j in range(ny):
            for i in range(nx):
                map_factor = msft[j, i]
                zx_w = 0.5 * (zx(kw, j, i) + zx(kw, j, i + 1))
                zy_w = 0.5 * (zy(kw, j, i) + zy(kw, j + 1, i))
                rz = rdz(kw, j, i)
                x_h = map_factor * rdx * (
                    tau13(kw, j, i + 1) - tau13(kw, j, i))
                y_h = map_factor * rdy * (
                    tau23(kw, j + 1, i) - tau23(kw, j, i))
                x_z = map_factor * rz * zx_w * (
                    tau13_mavg(kw, j, i)
                    - tau13_mavg(kw - 1, j, i))
                y_z = map_factor * rz * zy_w * (
                    tau23_mavg(kw, j, i)
                    - tau23_mavg(kw - 1, j, i))
                factor = G / (dn[kw] * rz)
                rw_x[kw, j, i] = factor * (x_h - x_z)
                rw_y[kw, j, i] = factor * (y_h - y_z)
                rw[kw, j, i] = rw_x[kw, j, i] + rw_y[kw, j, i]

    return {"km": km, "kh": 3.0 * km, "d11": d11, "d22": d22,
            "d12": d12, "ru": ru, "rv": rv, "rw": rw,
            "ru_cross": ru_cross, "rv_cross": rv_cross,
            "rw_x": rw_x, "rw_y": rw_y}


__all__ = ["wrf_d11_algebra", "wrf_flat_u_cross_stress_tendency",
           "wrf_periodic_momentum_authority",
           "wrf_periodic_scalar_metric_tendency",
           "wrf_smag2d_km_algebra"]
