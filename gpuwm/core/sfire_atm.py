"""Device atmospheric exchange routines from WRF v4.7.1 SFIRE.

The exponential release follows ``module_fr_fire_atm.F:112-304``. The
truncated-Gaussian release integrates its CDF over mass layers and injects
the resulting conserved fractions directly. Separate crown heat and water
profiles use the crown height and extinction depth. Smoke emits
over every interior column rather than the original single-column loop.
These are intentional corrections, recorded by the atmospheric oracle.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math

import numpy as np

MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "gpuwm.core.sfire_atm:sfire_atm"
_BLOCK = 128


def module_source(kernel_dir=None) -> str:
    from gpuwm.core.kernels import _preamble
    root = Path(__file__).parent / "kernels" if kernel_dir is None else Path(kernel_dir)
    return _preamble(root) + "".join((root/name).read_text(encoding="utf-8") for name in
                               ("glibc_flt32.cuh", "sfire_libm.cuh", "sfire_atm.cu"))


@lru_cache(maxsize=None)
def _module(device: int):
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(source, MODULE_OPTIONS, None, "sfire_atm.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS, module=None)
    return module


def _launch(name, count, args):
    import cupy as cp
    _module(cp.cuda.Device().id).get_function(name)(
        ((count + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)


def _array(value, shape, name):
    import cupy as cp
    result = cp.ascontiguousarray(cp.asarray(value, dtype=cp.float32))
    if result.shape != tuple(shape):
        raise ValueError(f"{name} shape {result.shape} differs from {tuple(shape)}; exchange grids must align")
    return result


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive; invalid geometry corrupts fire flux divergence")
    return np.float32(value)


def _domain(shape, domain, *, smoke=False):
    ny, nx = shape
    if domain is None:
        domain = (1, nx - 2 if smoke else nx - 1, 1, ny - 2 if smoke else ny - 1)
    if len(domain) != 4:
        raise ValueError("fire atmospheric domain must give four inclusive bounds")
    lo_x, hi_x, lo_y, hi_y = domain
    if not (0 <= lo_x <= hi_x < nx and 0 <= lo_y <= hi_y < ny):
        raise ValueError("fire atmospheric domain lies outside the exchange arrays")
    return tuple(np.int32(v) for v in domain)


def sum_fire_cells(field, sr_x: int, sr_y: int, *, mean=False):
    """WRF ``sum_2d_cells``, with its serial joff/ioff summation order.

    Fire cells are (ny*sr_y, nx*sr_x). The optional mean multiplies the
    completed sum by WRF's single precision ``1./(sr_x*sr_y)``.
    """
    import cupy as cp
    if isinstance(sr_x, bool) or isinstance(sr_y, bool) or int(sr_x) != sr_x or int(sr_y) != sr_y or min(sr_x, sr_y) < 1:
        raise ValueError("fire refinement must contain positive integers; nonintegral grids lose fire cells")
    field = cp.ascontiguousarray(cp.asarray(field, dtype=cp.float32))
    if field.ndim != 2 or field.shape[0] % sr_y or field.shape[1] % sr_x:
        raise ValueError("fire grid must be an integer refinement of the atmospheric grid")
    ny, nx = field.shape[0] // sr_y, field.shape[1] // sr_x
    result = cp.empty((ny, nx), cp.float32)
    _launch("sfire_sum_cells", nx * ny, (field, result, np.int32(nx), np.int32(ny),
            np.int32(sr_x), np.int32(sr_y), np.int32(mean)))
    return result


def interpolate_2d(field, fine_shape, sr_x: int, sr_y: int, *,
                   coarse_origin=(0.0, 0.0), fine_origin=(0.0, 0.0)):
    """WRF util ``interpolate_2d`` for a complete rectangular fine patch.

    Origins are zero based coordinates that align a coarse node with a
    fine node. The last coarse cell wins on shared boundaries, preserving
    WRF's loop ordering. Nodes outside the supplied coarse extent refuse
    rather than inheriting unwritten memory.
    """
    import cupy as cp
    if int(sr_x) != sr_x or int(sr_y) != sr_y or min(sr_x, sr_y) < 1:
        raise ValueError("fire interpolation refinement must contain positive integers")
    field = cp.ascontiguousarray(cp.asarray(field, dtype=cp.float32))
    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("fire interpolation needs at least two nodes on each coarse axis")
    fy, fx = map(int, fine_shape)
    if min(fx, fy) < 1:
        raise ValueError("fire interpolation patch must have positive dimensions")
    cx, cy = map(float, coarse_origin)
    ox, oy = map(float, fine_origin)
    if not all(math.isfinite(v) for v in (cx, cy, ox, oy)):
        raise ValueError("fire interpolation origins must be finite")
    xmin, ymin = ox - sr_x * cx, oy - sr_y * cy
    xmax, ymax = xmin + sr_x * (field.shape[1] - 1), ymin + sr_y * (field.shape[0] - 1)
    if xmin > 0 or ymin > 0 or xmax < fx - 1 or ymax < fy - 1:
        raise ValueError("coarse fire interpolation extent does not cover every requested fine node")
    result = cp.empty((fy, fx), cp.float32)
    _launch("sfire_interpolate_2d", fx * fy, (field, result, np.int32(field.shape[1]),
            np.int32(field.shape[0]), np.int32(fx), np.int32(fy), np.int32(sr_x),
            np.int32(sr_y), np.float32(cx), np.float32(cy), np.float32(ox), np.float32(oy)))
    return result


def log_wind_at_height(wind, heights, roughness, height: float):
    """WRF driver log interpolation for already staggered wind columns.

    ``heights`` gives the heights above the staggered ground of each mass
    level. Below the first level the wind is zero at roughness height;
    above the last level it holds the last wind, as WRF does.
    """
    import cupy as cp
    wind = cp.ascontiguousarray(cp.asarray(wind, dtype=cp.float32))
    if wind.ndim != 3 or wind.shape[0] < 1:
        raise ValueError("fire wind must contain mass-level columns")
    z = _array(heights, wind.shape, "fire wind heights")
    z0 = _array(roughness, wind.shape[1:], "fire wind roughness")
    target = _positive(height, "fire_wind_height")
    if bool(cp.any(~cp.isfinite(z))) or bool(cp.any(z <= 0)) or bool(cp.any(z[1:] <= z[:-1])) or bool(cp.any(~cp.isfinite(z0))) or bool(cp.any(z0 <= 0)):
        raise ValueError("fire log wind requires positive, increasing heights and positive roughness")
    result = cp.empty(wind.shape[1:], cp.float32)
    n = result.size
    _launch("sfire_log_wind", n, (wind, z, z0, result, np.int32(n), np.int32(wind.shape[0]), target))
    return result


def truncated_gaussian(dz8w, z_at_w, terrain, *, peak: float, upper: float,
                       extinction: float, domain=None, active_levels=None):
    """Corrected WRF ``tg_dist``; return conserved mass-layer fractions.

    The WRF tanh approximation to the Gaussian CDF is integrated between
    successive w levels, clipped to ``[0,upper]`` above the supplied terrain.
    Fractions are normalized over the active column. This avoids negative
    heat release from differentiating WRF's sampled PDF as an interface flux
    and avoids losing smoke when a peak crosses a layer boundary.
    """
    import cupy as cp
    dz = cp.ascontiguousarray(cp.asarray(dz8w, dtype=cp.float32))
    if dz.ndim != 3 or dz.shape[0] < 2:
        raise ValueError("fire release needs at least two mass levels")
    nz, ny, nx = dz.shape
    z = _array(z_at_w, (nz + 1, ny, nx), "z_at_w")
    zs = _array(terrain, (ny, nx), "terrain")
    ext = _positive(extinction, "fire extinction depth")
    ub = _positive(upper, "fire_tg_ub")
    if not math.isfinite(peak):
        raise ValueError("fire release peak must be finite")
    active = nz if active_levels is None else int(active_levels)
    if not 1 <= active <= nz:
        raise ValueError("fire release active levels must be within the atmospheric column")
    if bool(cp.any(~cp.isfinite(z))) or bool(cp.any(z[1:] <= z[:-1])) or bool(cp.any(~cp.isfinite(zs))):
        raise ValueError("fire release requires finite terrain and increasing interface heights")
    result = cp.zeros_like(dz)
    status = cp.zeros((),dtype=cp.int32)
    bounds = _domain((ny, nx), domain)
    _launch("sfire_tg_dist", nx * ny, (dz, z, zs, result, np.int32(nx), np.int32(ny),
            np.int32(nz), np.float32(peak), ub, ext, np.int32(active), status, *bounds))
    if int(status.item()):
        raise ValueError("truncated Gaussian release has no representable mass in the active column")
    return result


def fire_tendency(grnhfx, grnqfx, canhfx, canqfx, *, terrain, z_at_w, dz8w,
                  mu, c1h, c2h, rho, fire_ext_grnd: float, fire_ext_crwn: float,
                  crown_height: float, fire_sfc_flx=0, fire_heat_peak=0.0,
                  fire_tg_ub=1000.0, domain=None):
    """Mass-coupled theta and water tendencies at WRF's A locations.

    WRF computes interface fluxes on the first nz w-levels, differences
    on nz-1 mass levels and zero tendency on the last mass level. The
    output uses those exact bounds. Horizontal ``domain`` bounds include
    only columns the caller owns; the remaining output is zero.
    """
    import cupy as cp
    rho = cp.ascontiguousarray(cp.asarray(rho, dtype=cp.float32))
    if rho.ndim != 3 or rho.shape[0] < 2:
        raise ValueError("fire atmosphere requires at least two mass levels")
    nz, ny, nx = rho.shape
    planes = [_array(v, (ny, nx), name) for name, v in (
        ("GRNHFX", grnhfx), ("GRNQFX", grnqfx), ("CANHFX", canhfx),
        ("CANQFX", canqfx), ("terrain", terrain), ("mu", mu))]
    z = _array(z_at_w, (nz + 1, ny, nx), "z_at_w")
    dz = _array(dz8w, rho.shape, "dz8w")
    c1 = _array(c1h, (nz,), "c1h")
    c2 = _array(c2h, (nz,), "c2h")
    if fire_sfc_flx not in (0, 1):
        raise ValueError("fire_sfc_flx must be 0 (exponential) or 1 (truncated Gaussian)")
    grnd = _positive(fire_ext_grnd, "fire_ext_grnd")
    crwn = _positive(fire_ext_crwn, "fire_ext_crwn")
    if not math.isfinite(crown_height) or crown_height < 0:
        raise ValueError("crown heat release height must be finite and nonnegative")
    if bool(cp.any(~cp.isfinite(rho))) or bool(cp.any(rho <= 0)) or bool(cp.any(~cp.isfinite(dz))) or bool(cp.any(dz <= 0)):
        raise ValueError("fire flux divergence requires finite positive density and layer thickness")
    bounds = _domain((ny, nx), domain)
    prop = cp.zeros_like(rho)
    prop_crown = cp.zeros_like(rho)
    if fire_sfc_flx == 1:
        prop = truncated_gaussian(dz, z, planes[4], peak=fire_heat_peak,
                                  upper=fire_tg_ub, extinction=fire_ext_grnd, domain=domain,
                                  active_levels=nz-1)
        if bool(cp.any(planes[2] != 0)) or bool(cp.any(planes[3] != 0)):
            prop_crown = truncated_gaussian(dz,z,planes[4]+np.float32(crown_height),
                peak=fire_heat_peak,upper=fire_tg_ub,extinction=fire_ext_crwn,
                domain=domain,active_levels=nz-1)
    th, qv = cp.zeros_like(rho), cp.zeros_like(rho)
    _launch("sfire_fire_tendency", nx * ny, (*planes[:4], planes[4], z, dz, planes[5],
            c1, c2, rho, prop, prop_crown, th, qv, np.int32(nx), np.int32(ny), np.int32(nz),
            grnd, crwn, np.float32(crown_height), np.int32(fire_sfc_flx), *bounds))
    return th, qv


def add_fire_tracer_emissions(tracer, burnt_area_dt, fuel_loading, *, rho,
                              dz8w, sr_x: int, sr_y: int, smoke_yield: float,
                              scheme=0, z_at_w=None, terrain=None, peak=0.0,
                              upper=1000.0, extinction=50.0, domain=None):
    """WRF SFIRE smoke increment, in g smoke per kg air, applied in place.

    ``burnt_area_dt`` is the burned fraction during the completed fire
    step, not a rate. There is no additional multiplication by dt. The
    caller converts the tracer unit when using a different species table.
    """
    import cupy as cp
    if not isinstance(tracer, cp.ndarray) or tracer.dtype != cp.float32 or not tracer.flags.c_contiguous:
        raise ValueError("fire smoke tracer must be a contiguous float32 device array for in-place injection")
    if tracer.ndim != 3 or tracer.shape[0] < 2:
        raise ValueError("fire smoke needs atmospheric mass-level columns")
    if scheme not in (0, 1) or int(sr_x) != sr_x or int(sr_y) != sr_y or min(sr_x, sr_y) < 1:
        raise ValueError("fire smoke scheme must be 0 or 1, with positive integer fire refinement")
    if not math.isfinite(smoke_yield) or smoke_yield < 0:
        raise ValueError("fire smoke yield must be finite and nonnegative")
    nz, ny, nx = tracer.shape
    fine = (ny * sr_y, nx * sr_x)
    burnt = _array(burnt_area_dt, fine, "burnt_area_dt")
    fuel = _array(fuel_loading, fine, "fuel_loading")
    density = _array(rho, tracer.shape, "rho")
    dz = _array(dz8w, tracer.shape, "dz8w")
    if bool(cp.any(density <= 0)) or bool(cp.any(dz <= 0)):
        raise ValueError("fire smoke requires positive density and layer thickness")
    bounds = _domain((ny, nx), domain, smoke=True)
    prop = cp.zeros_like(tracer)
    if scheme == 1:
        prop = truncated_gaussian(dz, z_at_w, terrain, peak=peak, upper=upper,
                                  extinction=extinction, domain=tuple(map(int, bounds)))
    _launch("sfire_smoke_emissions", nx * ny, (tracer, burnt, fuel, density, dz,
            prop, np.int32(nx), np.int32(ny), np.int32(nz), np.int32(sr_x),
            np.int32(sr_y), np.float32(smoke_yield), np.int32(scheme), *bounds))
    return tracer
