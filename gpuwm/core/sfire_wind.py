"""WRF v4.7.1 atmosphere-to-fire wind interpolation on the device.

The input winds remain staggered. Geopotential supplies ground-relative
mass heights, roughness is averaged onto faces, and the native logarithmic
profile, boundary continuation and refined-grid offsets preserve WRF order.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math

import numpy as np

MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "gpuwm.core.sfire_wind:sfire_wind"
_BLOCK = 128


def module_source(kernel_dir=None):
    from gpuwm.core.kernels import _preamble
    root = Path(__file__).parent / "kernels" if kernel_dir is None else Path(kernel_dir)
    return _preamble(root) + (root / "glibc_flt32.cuh").read_text() + (root / "sfire_wind.cu").read_text()


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "sfire_wind.cu")
        result = cp.cuda.function.Module()
        result.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS, module=None)
    return result


def _launch(name, count, args):
    import cupy as cp
    _module(cp.cuda.Device().id).get_function(name)(
        ((count + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _field(value, shape, name):
    import cupy as cp
    value = cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
    if value.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}; staggered fire exchange grids must align")
    return value


def _bounds(value, shape, name):
    if len(value) != 4 or any(isinstance(v, bool) or int(v) != v for v in value):
        raise ValueError(f"{name} must contain four inclusive integer bounds")
    xl, xh, yl, yh = map(int, value)
    ny, nx = shape
    if not (0 <= xl < xh < nx and 0 <= yl < yh < ny):
        raise ValueError(f"{name} requires at least two nodes per axis inside the allocation")
    return xl, xh, yl, yh


def _continuation_bounds(tile, domain):
    xl, xh, yl, yh = tile
    dl, dh, bl, bh = domain
    return (xl - int(xl == dl), xh + int(xh == dh),
            yl - int(yl == bl), yh + int(yh == bh))


def interpolate_atm2fire(u, v, ph, phb, z0, zs, z0f, sr_x, sr_y, *,
                         fire_wind_height=6.5, domain=None, fine_domain=None,
                         u_frame=0.0, v_frame=0.0, fire_lsm_zcoupling=False,
                         fire_lsm_zcoupling_ref=50.0, tiles=None, fine_tiles=None,
                         domain_includes_terminal_face=True):
    """Return ``uf``, ``vf``, ``uah`` and ``vah`` using the native driver.

    ``u`` and ``v`` share a ``(nz,ny,nx)`` allocation; ``ph`` and ``phb``
    contain ``nz+1`` w levels. Horizontal domain bounds name the WRF driver
    nodes. The default accepts a routine control whose fine domain spans
    ``(xh-xl)*sr_x`` by ``(yh-yl)*sr_y`` fire cells. With
    ``domain_includes_terminal_face=False`` the bounds name the mass cells
    passed by the coupled WRF driver and the fine extent uses their inclusive
    count. Allocations may
    include halos. Atmosphere and fire tiles are paired in the same order.
    Frame corrections are accepted and unused, as in the original routine.
    """
    import cupy as cp
    u = cp.ascontiguousarray(cp.asarray(u, dtype=cp.float32))
    if u.ndim != 3 or u.shape[0] < 1:
        raise ValueError("fire wind exchange requires atmospheric mass-level columns")
    nz, ny, nx = u.shape
    v = _field(v, u.shape, "v")
    ph = _field(ph, (nz + 1, ny, nx), "ph")
    phb = _field(phb, ph.shape, "phb")
    z0 = _field(z0, (ny, nx), "z0")
    _field(zs, (ny, nx), "zs")
    z0f = cp.ascontiguousarray(cp.asarray(z0f, dtype=cp.float32))
    if z0f.ndim != 2:
        raise ValueError("fire roughness must be a horizontal fine-grid allocation")
    if any(isinstance(r, bool) or int(r) != r or r < 1 for r in (sr_x, sr_y)):
        raise ValueError("fire refinement must contain positive integers; fractional grids misalign staggered winds")
    sr_x, sr_y = int(sr_x), int(sr_y)
    for value, name in ((fire_wind_height, "fire_wind_height"),
                        (fire_lsm_zcoupling_ref, "fire_lsm_zcoupling_ref")):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive for logarithmic wind interpolation")
    if not all(math.isfinite(f) for f in (u_frame, v_frame)):
        raise ValueError("fire velocity frame corrections must be finite")
    domain = _bounds(domain or (0, nx - 2, 0, ny - 2), (ny, nx), "atmospheric domain")
    xl, xh, yl, yh = domain
    if xh - xl < 2 or yh - yl < 2:
        raise ValueError("fire wind continuation needs two valid staggered faces per atmospheric axis")
    if xh + 1 >= nx or yh + 1 >= ny:
        raise ValueError("fire wind diagnostics need the native upper continuation halo inside the atmospheric allocation")
    fy, fx = z0f.shape
    if fine_domain is None:
        fine_domain = (0, fx - 1, 0, fy - 1)
    fxl, fxh, fyl, fyh = map(int, fine_domain)
    if not (0 <= fxl <= fxh < fx and 0 <= fyl <= fyh < fy):
        raise ValueError("fine fire domain lies outside its roughness allocation")
    mass_count = int(not domain_includes_terminal_face)
    if (fxh - fxl + 1, fyh - fyl + 1) != ((xh - xl + mass_count) * sr_x, (yh - yl + mass_count) * sr_y):
        raise ValueError("fire cells must exactly refine the atmospheric driver domain")
    atiles = tuple(tiles or (domain,))
    ftiles = tuple(fine_tiles or (fine_domain,))
    if len(atiles) != len(ftiles):
        raise ValueError("atmosphere and fire tiles must be paired to preserve native interpolation ownership")
    # The original routine leaves diagnostics outside the tile unwritten.
    # A stable sentinel makes that ownership explicit without changing words.
    sentinel = np.array(0x7fffffff, dtype=np.uint32).view(np.float32).item()
    uah, vah = cp.full((ny, nx), sentinel, cp.float32), cp.full((ny, nx), sentinel, cp.float32)
    uf, vf = cp.full((fy, fx), sentinel, cp.float32), cp.full((fy, fx), sentinel, cp.float32)
    for tile, ftile in zip(atiles, ftiles):
        tl, th, tb, tt = _bounds(tile, (ny, nx), "atmospheric tile")
        if not (xl <= tl <= th <= xh and yl <= tb <= tt <= yh):
            raise ValueError("atmospheric fire tile must lie within its domain")
        fl, fh, fb, ft = map(int, ftile)
        if not (fxl <= fl <= fh <= fxh and fyl <= fb <= ft <= fyh):
            raise ValueError("fine fire tile must lie within its domain")
        ml, mb, tx, ty = tl - 2, tb - 2, th - tl + 5, tt - tb + 5
        ua, va = cp.full((ty, tx), sentinel, cp.float32), cp.full((ty, tx), sentinel, cp.float32)
        ub = (tl + int(tl == xl), th + int(th != xh), tb - int(tb > yl), tt + int(tt != yh))
        vb = (tl - int(tl > xl), th + int(th != xh), tb + int(tb == yl), tt + int(tt != yh))
        for axis, wind, out, bounds, d in ((0, u, ua, ub, (xl + 1, xh, yl, yh)),
                                         (1, v, va, vb, (xl, xh, yl + 1, yh))):
            _launch("sfire_wind_vertical", tx * ty,
                    (wind, ph, phb, z0, out, *map(np.int32, (nx, ny, nz, ml, mb, tx, ty, axis, *bounds)),
                     np.float32(fire_lsm_zcoupling_ref if fire_lsm_zcoupling else fire_wind_height)))
            for direction in (0, 1, 2):
                _launch("sfire_wind_continue", tx * ty,
                        (out, *map(np.int32, (ml, mb, tx, ty, *bounds, *d, direction))))
        _launch("sfire_wind_diagnostics", tx * ty,
                (ua, va, uah, vah, *map(np.int32, (nx, ny, ml, mb, tx, ty, tl,
                                                 th + int(th == xh), tb, tt + int(tt == yh)))))
        for axis, source, dest, bounds, d in ((0, ua, uf, ub, (xl + 1, xh, yl, yh)),
                                            (1, va, vf, vb, (xl, xh, yl + 1, yh))):
            cb = _continuation_bounds(bounds, d)
            _launch("sfire_wind_refine", (fh - fl + 1) * (ft - fb + 1),
                    (source, dest, *map(np.int32, (ml, mb, tx, ty, fx, sr_x, sr_y, xl, yl,
                                                 fxl, fyl, fl, fh, fb, ft, axis, *cb))))
        if fire_lsm_zcoupling:
            _launch("sfire_wind_zcoupling", (fh - fl + 1) * (ft - fb + 1),
                    (uf, vf, z0f, *map(np.int32, (fx, fl, fh, fb, ft)),
                     np.float32(fire_wind_height), np.float32(fire_lsm_zcoupling_ref)))
    return dict(uf=uf, vf=vf, uah=uah, vah=vah)


def interpolate_native_atm2fire(u, v, ph, phb, z0, zs, z0f, sr_x, sr_y, *,
                                fine_domain=None, return_staggered_diagnostics=False,
                                domain_includes_terminal_face=False, coarse_domain=None, **options):
    """Adapt compact native atmospheric arrays using device copy kernels.

    U is ``(nz,ny,nx+1)``, V is ``(nz,ny+1,nx)``, perturbation
    geopotential is ``(nz+1,ny,nx)``, and base geopotential is either
    the same allocation or a ``(nz+1,)`` profile. Horizontal fields
    are ``(ny,nx)``. Domain bounds name these mass cells, as in the coupled
    WRF driver. Its terminal wind faces are linearly continued from the
    interior. Surrounding scalar allocation halos use nearest boundary
    values. Fine
    roughness has its own explicit domain or symmetric surrounding halo.
    UAH and VAH default to mass-grid shape. With
    ``return_staggered_diagnostics=True`` they retain their native X/Y
    diagnostic faces for the WRF history schema. The explicit
    ``domain_includes_terminal_face=True`` option retains the smaller-domain
    routine-control convention.
    """
    import cupy as cp
    u = cp.ascontiguousarray(cp.asarray(u, dtype=cp.float32))
    if u.ndim != 3:
        raise ValueError("native fire U must have atmospheric staggered shape")
    nz, ny, nxx = u.shape
    nx = nxx - 1
    if min(nx, ny, nz) < 1:
        raise ValueError("native fire atmosphere must have positive grid dimensions")
    v = _field(v, (nz, ny + 1, nx), "native v")
    ph = _field(ph, (nz + 1, ny, nx), "native ph")
    phb = cp.ascontiguousarray(cp.asarray(phb, dtype=cp.float32))
    profile = phb.shape == (nz + 1,)
    if not profile:
        phb = _field(phb, ph.shape, "native phb")
    z0 = _field(z0, (ny, nx), "native roughness")
    zs = _field(zs, (ny, nx), "native terrain")
    z0f = cp.ascontiguousarray(cp.asarray(z0f, dtype=cp.float32))
    if z0f.ndim != 2:
        raise ValueError("native fire roughness requires a horizontal grid")
    if fine_domain is None:
        hx, hy = z0f.shape[1] - nx * sr_x, z0f.shape[0] - ny * sr_y
        if hx < 0 or hy < 0 or hx % 2 or hy % 2:
            raise ValueError("fine roughness allocation needs an explicit fire domain or symmetric halo")
        fx, fy = hx // 2, hy // 2
        fine_domain = (fx, fx + nx * sr_x - 1, fy, fy + ny * sr_y - 1)
    hx, hy = nx + 3, ny + 3
    packed_u, packed_v = cp.empty((nz, hy, hx), cp.float32), cp.empty((nz, hy, hx), cp.float32)
    packed_ph, packed_phb = cp.empty((nz + 1, hy, hx), cp.float32), cp.empty((nz + 1, hy, hx), cp.float32)
    packed_z0, packed_zs = cp.empty((hy, hx), cp.float32), cp.empty((hy, hx), cp.float32)
    _launch("sfire_wind_pack_native", (nz + 1) * hx * hy,
            (u, v, ph, phb, z0, zs, packed_u, packed_v, packed_ph, packed_phb, packed_z0, packed_zs,
             *map(np.int32, (nx, ny, nz, profile))))
    result = interpolate_atm2fire(packed_u, packed_v, packed_ph, packed_phb, packed_z0, packed_zs,
                                  z0f, sr_x, sr_y,
                                  domain=(coarse_domain if coarse_domain is not None else
                                          (1, nx + int(domain_includes_terminal_face),
                                           1, ny + int(domain_includes_terminal_face))),
                                  fine_domain=fine_domain,
                                  domain_includes_terminal_face=domain_includes_terminal_face, **options)
    result["uah"] = cp.ascontiguousarray(result["uah"][1:ny + 1, 1:nx + 1 + int(return_staggered_diagnostics)])
    result["vah"] = cp.ascontiguousarray(result["vah"][1:ny + 1 + int(return_staggered_diagnostics), 1:nx + 1])
    return result
