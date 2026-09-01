"""Radar reflectivity diagnostic: WRF ``do_radar_ref=1`` (REFL_10CM).

WRF computes ``refl_10cm`` inside the microphysics driver call on history
steps (``diagflag .and. do_radar_ref == 1``, phys/module_microphysics_driver.F
refl_10cm argument; Morrison wrapper phys/module_mp_morr_two_moment.F:911-918)
from the scheme's own post-call 1-D columns, so the diagnostic is consistent
with the ACTIVE scheme's intercepts, densities, and prognostic number
concentrations.  gpuwm uses the same arrangement: an output-due flag reaches
the scheme adapter, which calls :func:`compute_refl_10cm` with the prepared
pressure/Exner and the scheme's post-call temperature before
``moist_physics_finish`` or the dycore EOS refresh.  The result is stashed
until the output frame consumes it once (see PROVENANCE.md D2).
Morrison's fixed-Nc cloud path (INUM=1, NC3D = 250 cm-3/rho) never enters:
``refl10cm_hm`` reads no cloud-water moment at all (F:4502-4675).

Transcription authority (local WRF v4.6.1):

- ``phys/module_mp_morr_two_moment.F``: the radar m(D) parameter block
  (:528-542: xam_r = PI*RHOW/6, xbm_r = 3, xmu_r = 0; xam_s = CS; xam_g = CG;
  ``morr_rimed_ice`` selects RHOG/CG, with WRF's Registry default 1 = hail),
  the wrapper call/clamp (:911-918), and ``refl10cm_hm`` (:4502-4675).
- ``phys/module_mp_radar.F``: ``radar_init`` (:74-190) products consumed by
  the column routine -- WRF builds them once at scheme init on the host, and
  :func:`radar_init` mirrors that split: the products are computed here in
  float64 and handed to both the CUDA kernel and the npref mirror (the same
  single-source pattern as ``CUDA_DEFINES``/``noah_frh2o``).  Per-column math
  (the Blahak melting-particle soak) is transcribed separately in
  ``kernels/refl.cu`` and ``gpuwm.verify.npref.np_refl10cm_morrison_column``.

Kessler (mp_physics=1) fallback: WRF's Kessler carries NO reflectivity
diagnostic (module_mp_kessler.F has none), so the fallback is the classic
rain-only Rayleigh form for an exponential Marshall-Palmer PSD with fixed
intercept N0r = 8e6 m-4 and rho_w = 1000 kg m-3 (Smith et al. 1975, J. Appl.
Meteor. 14, 1156-1165, eq. for Ze of liquid rain; the same constants
Kessler's own terminal-velocity closure assumes and the form wrf-python/RIP
``dbzcalc`` uses for one-moment warm rain):

    lambda_r = (pi * rho_w * N0r / (rho * qr))**0.25
    Ze = Gamma(7) * N0r * lambda_r**-7 * 1e18   [mm6 m-3]

with the same air-density diagnosis, q-thresholds, and -35 dBZ floor
conventions as the Morrison routine so the two schemes' outputs are
interchangeable in wrfout/products.

Compatibility limits inherited from WRF are deliberate.  For an active
species (mass mixing ratio ``> 1e-9``), its number moment must be finite and
strictly positive.  ``refl10cm_hm`` gates only on mass and immediately forms
the fractional-power slope (module_mp_morr_two_moment.F:4544-4584); it does
not define a meteorological result for a zero, negative, or non-finite number
moment.  Normal Morrison outputs satisfy this contract because the scheme
clips number concentrations nonnegative and reconstructs them from bounded
slopes (:1528-1635).  gpuwm explicitly leaves the affected output non-finite
instead of allowing CUDA's ``fmaxf`` to turn invalid slope arithmetic into
the meteorological -35 dBZ floor.  Likewise, the melting calculation is the
exact WRF 50-log-bin weighted sum (module_mp_radar.F:83-90/:124-146 and
module_mp_morr_two_moment.F:4626-4667), retained for compatibility; it is not
claimed to be a converged composite-Simpson integral.

WSM6 (mp_physics=6) uses its native fixed-intercept rain/snow/graupel-or-
hail PSDs from ``mp_wsm6.F90:2275-2444``, including the same Blahak melting
particle integration.  Float64 mirrors: ``np_refl10cm_morrison_column``,
``np_refl10cm_wsm6_column``, and ``np_refl10cm_kessler_column``
(``gpuwm/verify/npref.py``).

Classic Thompson (mp_physics=8) uses ``calc_refl10cm`` from
``module_mp_thompson.F:5710-6028``: two-moment rain, the Field et al. snow
moment distribution and wet-snow Blahak integration, and its private
same-call classic-graupel number moment.  That graupel moment is supplied as
an output-due scratch shadow; it is not transported model state.

Aerosol-aware Thompson (mp_physics=28) uses the SAME routine with the same
inputs.  ``calc_refl10cm`` is not duplicated or specialized in WRF: one
subroutine serves both packages, it takes no droplet-number argument, and
its only cloud-water local is dead (see :func:`compute_refl_10cm`).  So
REFL_10CM under mp=28 is a function of qv/qr/nr/qs/qg/T/p exactly as it is
under mp=8, and the prognostic ``nc`` that defines mp=28 has no influence on
it whatsoever.  This is the one user-facing product field the aerosol port
adds no new numerics to, and stating that is more useful than a branch that
implies otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from gpuwm.core.morrison_constants import rimed_ice_constants
from gpuwm.core.wsm6_constants import rimed_ice_constants as wsm6_rimed

_COLUMN_TPB = 32
# The Kessler fallback is one independent cell per thread: no shared state,
# atomics, synchronization, or block-shape-dependent arithmetic.
_CELL_TPB = 256
_KMAX = 256                # unspecialized REFL_KMAX in kernels/refl.cu

#: Per-thread local frame each column kernel compiles to, per level of
#: ``REFL_KMAX``.  The column arrays are all that scale with the bound, so
#: the driver-reported frame is exactly ``bytes_per_level * REFL_KMAX``.
#: Measured on the RTX 5090 (driver 610.74, NVRTC 13.x, ``-std=c++17``) at
#: bounds 256 and 49; ``tests/test_kernel_local_bounds.py`` re-measures it.
KERNEL_LOCAL_FRAME_BYTES_PER_LEVEL = {
    "refl10cm_morrison_column": 72,     # 18,432 B at 256
    "refl10cm_thompson_column": 63,     # 16,128 B at 256
    "refl10cm_wsm6_column": 55,         # 14,080 B at 256
    "refl10cm_kessler_cell": 0,         # per-cell, no column arrays
}


def kernel_local_frame_bytes(nz: int) -> int:
    """Widest per-thread local frame in ``refl.cu`` at ``REFL_KMAX = nz``."""
    return max(KERNEL_LOCAL_FRAME_BYTES_PER_LEVEL.values()) * int(nz)


def kernel_capacity(nz: int) -> int:
    """The ``REFL_KMAX`` this ``nz`` compiles against.

    Exactly ``nz``.  Every loop in the three column kernels runs to the
    runtime ``nz`` and the highest index any of them forms is ``nz - 1``
    (the melting scan reads ``k + 1`` from ``k <= nz - 2``), so the bound is
    a pure allocation size.  At the unspecialized 256 on a 49-level case
    ``refl10cm_morrison_column`` costs 4,335 MiB of driver local-memory
    backing store the moment the first history frame comes due.
    """
    nz = int(nz)
    if nz < 1 or nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    return nz


def _column_kernel(symbol: str, nz: int, module: str = "refl"):
    """One reflectivity column kernel, compiled at ``REFL_KMAX = nz``.

    ``module`` is the translation unit.  WDM6's kernel lives in its own
    (``wdm6_refl.cu``) rather than in ``refl.cu``, because ``refl.cu`` is
    byte-frozen -- see that file's header and
    ``tests/test_mp8_frozen.py::test_every_frozen_kernel_module_is_unchanged``
    -- so a new scheme may not be appended to it.
    """
    from gpuwm.core.kernels import get_kernel, get_kernel_int_defines

    capacity = kernel_capacity(nz)
    if capacity == _KMAX:
        return get_kernel(module, symbol)
    return get_kernel_int_defines(module, symbol,
                                  (("REFL_KMAX", capacity),))

#: module_mp_radar.F:40/43/63-64: 50 size bins, 10 cm wavelength, and the
#: 90% external-meltwater assumption for snow and graupel.
NRBINS = 50
LAMDA_RADAR = 0.10
MELT_OUTSIDE_S = 0.9
MELT_OUTSIDE_G = 0.9

#: Morrison module PI (module_mp_morr_two_moment.F:99) -- also the radar
#: module's PIx (module_mp_radar.F:203/283).
_PI = 3.1415926535897932384626434
#: module_model_constants.F:19 r_d, Morrison's R (F:92).
_R = 287.0


def _m_complex_water_ray(lam: float, t: float) -> complex:
    """Complex refractive index of water after Ray (1972);
    module_mp_radar.F:193-222 verbatim (T in deg C, lam in m)."""
    epsinf = 5.27137 + 0.02164740 * t - 0.00131198 * t * t
    epss = 78.54 * (1.0 - 4.579e-3 * (t - 25.0)
                    + 1.190e-5 * (t - 25.0) * (t - 25.0)
                    - 2.800e-8 * (t - 25.0) * (t - 25.0) * (t - 25.0))
    alpha = -16.8129 / (t + 273.16) + 0.0609265
    lambdas = 0.00033836 * math.exp(2513.98 / (t + 273.16)) * 1e-2
    nenner = (1.0 + 2.0 * (lambdas / lam) ** (1.0 - alpha)
              * math.sin(alpha * _PI * 0.5)
              + (lambdas / lam) ** (2.0 - 2.0 * alpha))
    epsr = epsinf + ((epss - epsinf) * ((lambdas / lam) ** (1.0 - alpha)
                                        * math.sin(alpha * _PI * 0.5)
                                        + 1.0)) / nenner
    epsi = (((epss - epsinf) * ((lambdas / lam) ** (1.0 - alpha)
                                * math.cos(alpha * _PI * 0.5)
                                + 0.0)) / nenner
            + lam * 1.25664 / 1.88496)
    return complex(np.sqrt(complex(epsr, -epsi)))


def _m_complex_ice_maetzler(lam: float, t: float) -> complex:
    """Complex refractive index of ice after Maetzler (1998);
    module_mp_radar.F:226-262 verbatim (T in deg C, lam in m)."""
    c = 2.99e8
    tk = t + 273.16
    f = c / lam * 1e-9
    b1 = 0.0207
    b2 = 1.16e-11
    b = 335.0
    deltabeta = math.exp(-10.02 + 0.0364 * (tk - 273.16))
    betam = (b1 / tk) * (math.exp(b / tk)
                         / ((math.exp(b / tk) - 1.0) ** 2)) + b2 * f * f
    beta = betam + deltabeta
    theta = 300.0 / tk - 1.0
    alfa = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    m = complex(3.1884 + 9.1e-4 * (tk - 273.16), 0.0)
    m = m + complex(0.0, alfa / f + beta * f)
    return complex(np.sqrt(np.conj(m)))


@dataclass(frozen=True)
class RadarInit:
    """``radar_init`` products (module_mp_radar.F:74-190), float64.

    Gamma moments are exact ``math.gamma`` where WRF evaluates its
    REAL WGAMMA/GAMMLN pair (Numerical Recipes 2.02, :561-597) -- the same
    substitution the Morrison port already makes; the relative difference
    is O(1e-7), far below the FP32 state floor.
    """

    xam_r: float
    xam_s: float
    xam_g: float
    xcre: tuple      # (1+bm, 1+mu, 1+bm+mu, 1+2bm+mu) for rain, and
    xcse: tuple      # likewise snow/graupel (all bm=3, mu=0 for Morrison)
    xcge: tuple
    xcrg: tuple      # Gamma(xcre(n))
    xcsg: tuple
    xcgg: tuple
    xorg2: float     # 1/Gamma(1+mu) per species
    xosg2: float
    xogg2: float
    xobmr: float     # 1/bm and the (1/am)**(1/bm) soak geometry factors
    xoams: float
    xobms: float
    xocms: float
    xoamg: float
    xobmg: float
    xocmg: float
    pi5: float       # 3.14159**5 (truncated pi, :77 verbatim)
    lamda4: float
    m_w_0: complex
    m_i_0: complex
    k_w: float
    xxds: np.ndarray     # snow bin centers / widths (100 um .. 2 cm)
    xdts: np.ndarray
    xxdg: np.ndarray     # graupel bin centers / widths (100 um .. 5 cm)
    xdtg: np.ndarray
    simpson: np.ndarray  # composite-Simpson weights, first NRBINS entries


def _frozen(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _log_bins(dmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Log-spaced bin centers/widths from 100 um to ``dmax``
    (module_mp_radar.F:124-146)."""
    edges = np.empty(NRBINS + 1, dtype=np.float64)
    edges[0] = 100.0e-6
    edges[NRBINS] = dmax
    for n in range(1, NRBINS):
        edges[n] = math.exp(n / NRBINS * math.log(edges[NRBINS] / edges[0])
                            + math.log(edges[0]))
    return np.sqrt(edges[:-1] * edges[1:]), np.diff(edges)


@lru_cache(maxsize=2)
def radar_init(morr_rimed_ice: int = 1) -> RadarInit:
    """Mirror of ``radar_init`` for Morrison's m(D) parameters.

    Inputs are the values MORR_TWO_MOMENT_INIT sets before its
    ``call radar_init`` (module_mp_morr_two_moment.F:532-542): rain
    am = PI*RHOW/6 (RHOW = 997), bm = 3, mu = 0; snow am = CS = 100*PI/6;
    dense-ice am = CG = RHOG*PI/6, selected by ``morr_rimed_ice``.  WRF's
    Registry default 1 uses hail RHOG=900; explicit 0 uses graupel RHOG=400
    (Registry.EM_COMMON:2663-2666; F:337-411).
    """
    xam_r = _PI * 997.0 / 6.0
    xam_s = 100.0 * _PI / 6.0
    xam_g = rimed_ice_constants(morr_rimed_ice).rhog * _PI / 6.0
    xbm = 3.0
    xmu = 0.0

    pi5 = 3.14159 ** 5                        # :77, truncated pi verbatim
    lamda4 = LAMDA_RADAR ** 4                 # :78
    m_w_0 = _m_complex_water_ray(LAMDA_RADAR, 0.0)      # :79
    m_i_0 = _m_complex_ice_maetzler(LAMDA_RADAR, 0.0)   # :80
    k_w = abs((m_w_0 * m_w_0 - 1.0) / (m_w_0 * m_w_0 + 2.0)) ** 2  # :81

    simpson = np.zeros(NRBINS + 1, dtype=np.float64)    # :83-90
    basis = (1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0)
    for n in range(0, NRBINS - 1, 2):
        simpson[n] += basis[0]
        simpson[n + 1] += basis[1]
        simpson[n + 2] += basis[2]

    xxds, xdts = _log_bins(0.02)              # :123-134 snow to 2 cm
    xxdg, xdtg = _log_bins(0.05)              # :136-146 graupel to 5 cm

    # :152-181: per-species PSD-moment exponents and gamma values (bm and
    # mu identical across Morrison's three radar species).
    xce = (1.0 + xbm, 1.0 + xmu, 1.0 + xbm + xmu, 1.0 + 2.0 * xbm + xmu)
    xcg = tuple(math.gamma(v) for v in xce)
    o2 = 1.0 / xcg[1]

    return RadarInit(
        xam_r=xam_r, xam_s=xam_s, xam_g=xam_g,
        xcre=xce, xcse=xce, xcge=xce,
        xcrg=xcg, xcsg=xcg, xcgg=xcg,
        xorg2=o2, xosg2=o2, xogg2=o2,
        xobmr=1.0 / xbm,                       # :183-189
        xoams=1.0 / xam_s, xobms=1.0 / xbm,
        xocms=(1.0 / xam_s) ** (1.0 / xbm),
        xoamg=1.0 / xam_g, xobmg=1.0 / xbm,
        xocmg=(1.0 / xam_g) ** (1.0 / xbm),
        pi5=pi5, lamda4=lamda4, m_w_0=m_w_0, m_i_0=m_i_0, k_w=k_w,
        xxds=_frozen(xxds), xdts=_frozen(xdts),
        xxdg=_frozen(xxdg), xdtg=_frozen(xdtg),
        simpson=_frozen(simpson[:NRBINS]))


@lru_cache(maxsize=2)
def radar_init_wsm6(hail_opt: int = 0) -> RadarInit:
    """``radar_init`` products for WSM6's 1000/100/500-or-700 densities."""
    rimed = wsm6_rimed(hail_opt)
    xam_g = rimed.deng * _PI / 6.0
    base = radar_init(0)
    return replace(
        base, xam_r=1000.0 * _PI / 6.0, xam_g=xam_g,
        xoamg=1.0 / xam_g, xocmg=(1.0 / xam_g) ** (1.0 / 3.0))


@lru_cache(maxsize=2)
def radar_init_wdm6(hail_opt: int = 0) -> RadarInit:
    """``radar_init`` products for WDM6's m(D) inputs (wdm6init:2196-2208).

    Densities are WSM6's (rain 1000, snow 100, graupel 500 or 700 by
    ``hail_opt``), but WDM6 sets ``xmu_r = 1`` before ``call radar_init``
    while snow and graupel keep ``xmu = 0``.  So only the RAIN exponent and
    gamma tuples move: ``xcre = (1+bm, 1+mu, 1+bm+mu, 1+2bm+mu)`` becomes
    (4, 2, 5, 8) with ``xcrg = (6, 1, 24, 5040)`` and ``xorg2 = 1/G(2) = 1``.
    Snow and graupel keep the mu = 0 tuples ``radar_init`` already returns.
    """
    rimed = wsm6_rimed(hail_opt)
    xam_g = rimed.deng * _PI / 6.0
    base = radar_init(0)
    xbm, xmu_r = 3.0, 1.0
    xcre = (1.0 + xbm, 1.0 + xmu_r, 1.0 + xbm + xmu_r,
            1.0 + 2.0 * xbm + xmu_r)
    xcrg = tuple(math.gamma(v) for v in xcre)
    return replace(
        base, xam_r=1000.0 * _PI / 6.0, xam_g=xam_g,
        xcre=xcre, xcrg=xcrg, xorg2=1.0 / xcrg[1],
        xoamg=1.0 / xam_g, xocmg=(1.0 / xam_g) ** (1.0 / 3.0))


@lru_cache(maxsize=1)
def _device_tables():
    """Packed float64 device copy of the radar_init bin/weight tables:
    [xxDs | xdts | xxDg | xdtg | simpson], each NRBINS long."""
    import cupy as cp

    rc = radar_init()
    return cp.asarray(np.concatenate(
        [rc.xxds, rc.xdts, rc.xxdg, rc.xdtg, rc.simpson]))


def _check_fields(fields: dict, shape) -> None:
    from gpuwm.core.state import DTYPE

    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, "
                             f"got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def launch_refl10cm_morrison(qv, qr, nr, qs, ns, qg, ng, t, p, refl, *,
                             morr_rimed_ice: int = 1) -> None:
    """One CUDA thread per column: Morrison ``refl10cm_hm`` into ``refl``.

    All arrays are contiguous FP32 ``(nz, ny, nx)``; ``t`` is air
    temperature (K), ``p`` full pressure (Pa), mixing ratios kg/kg and
    number concentrations kg-1 (the scheme's own prognostic moments).
    ``refl`` receives dBZ with the wrapper's -35 floor
    (module_mp_morr_two_moment.F:913-917).

    As in WRF, an active mass moment requires a finite, strictly positive
    matching number moment.  WRF's routine has no number-moment guard
    (:4544-4584); post-Morrison states meet the precondition through the
    scheme's nonnegative clipping and bounded-slope reconstruction
    (:1528-1635).  Inputs outside that production contract have no defined
    meteorological interpretation; gpuwm returns non-finite output at the
    affected level rather than silently mapping it to clear air.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    _check_fields({"qv": qv, "qr": qr, "nr": nr, "qs": qs, "ns": ns,
                   "qg": qg, "ng": ng, "t": t, "p": p, "refl": refl}, shape)
    rc = radar_init(morr_rimed_ice)
    kernel = _column_kernel("refl10cm_morrison_column", nz)
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,),
           (qv, qr, nr, qs, ns, qg, ng, t, p, _device_tables(),
            np.float64(rc.k_w),
            np.float64(rc.m_w_0.real), np.float64(rc.m_w_0.imag),
            np.float64(rc.m_i_0.real), np.float64(rc.m_i_0.imag),
            np.float64(rc.xam_g),
            refl, np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_refl10cm_kessler(qv, qr, t, p, refl) -> None:
    """Per-cell Smith-1975 rain-only reflectivity into ``refl`` (dBZ).

    Same array contract as :func:`launch_refl10cm_morrison` minus the ice
    moments; the module docstring derives the fixed-intercept form.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    _check_fields({"qv": qv, "qr": qr, "t": t, "p": p, "refl": refl}, shape)
    ncell = int(np.prod(shape))
    # Declares no column arrays, so its own frame is 0 B either way; it comes
    # from the specialized module so one compilation of refl.cu serves the
    # process instead of two.
    kernel = _column_kernel("refl10cm_kessler_cell", shape[0])
    blocks = (ncell + _CELL_TPB - 1) // _CELL_TPB
    kernel((blocks,), (_CELL_TPB,), (qv, qr, t, p, refl, np.int32(ncell)))


def launch_refl10cm_wsm6(qv, qr, qs, qg, t, p, refl, *,
                          hail_opt: int = 0) -> None:
    """One CUDA thread per column: WRF ``refl10cm_wsm6`` into dBZ."""
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    _check_fields({"qv": qv, "qr": qr, "qs": qs, "qg": qg,
                   "t": t, "p": p, "refl": refl}, shape)
    rc = radar_init_wsm6(hail_opt)
    rimed = wsm6_rimed(hail_opt)
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    _column_kernel("refl10cm_wsm6_column", nz)(
        (blocks,), (_COLUMN_TPB,),
        (qv, qr, qs, qg, t, p, _device_tables(),
         np.float64(rc.k_w), np.float64(rc.m_w_0.real),
         np.float64(rc.m_w_0.imag), np.float64(rc.m_i_0.real),
         np.float64(rc.m_i_0.imag), np.float64(rc.xam_g),
         np.float64(rimed.n0g), refl,
         np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_refl10cm_wdm6(qv, qr, nr, qs, qg, t, p, refl, *,
                         hail_opt: int = 0) -> None:
    """One CUDA thread per column: WRF ``refl10cm_wdm6`` into dBZ.

    ``nr`` is WDM6's transported rain number moment.  Unlike Thompson's
    diagnostic, it needs no private shadow: WDM6 predicts rain number as
    state, and the driver hands the routine the same ``nr`` array the
    scheme just updated (module_mp_wdm6.F:281-296).
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    _check_fields({"qv": qv, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
                   "t": t, "p": p, "refl": refl}, shape)
    rc = radar_init_wdm6(hail_opt)
    rimed = wsm6_rimed(hail_opt)
    # ONE SOURCE OF TRUTH.  The rain moments below are the ONLY statement of
    # these numbers: kernels/wdm6_refl.cu takes them as arguments rather
    # than re-spelling them as literals.  Two exponents stay compile-time in
    # the kernel -- xcre(2) = 2 as ``lam*lam`` and xcre(4) = 8 as
    # ``pow(ilamr, 8.0)`` -- so a different xmu_r must not reach it silently.
    if rc.xcre != (4.0, 2.0, 5.0, 8.0):
        raise ValueError(
            "kernels/wdm6_refl.cu hard-codes the rain PSD exponents "
            f"xcre(2)=2 and xcre(4)=8; radar_init_wdm6 now derives "
            f"xcre={rc.xcre}, so the kernel would silently use the old "
            "exponents. Update the kernel with the derivation, do not "
            "relax this check.")
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    _column_kernel("refl10cm_wdm6_column", nz, module="wdm6_refl")(
        (blocks,), (_COLUMN_TPB,),
        (qv, qr, nr, qs, qg, t, p, _device_tables(),
         np.float64(rc.k_w), np.float64(rc.m_w_0.real),
         np.float64(rc.m_w_0.imag), np.float64(rc.m_i_0.real),
         np.float64(rc.m_i_0.imag), np.float64(rc.xam_g),
         np.float64(rimed.n0g),
         np.float64(rc.xam_r), np.float64(rc.xcrg[2]),
         np.float64(rc.xcrg[3]), np.float64(rc.xorg2), refl,
         np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_refl10cm_thompson(
        qv, qr, nr, qs, qg, graupel_number_shadow, t, p, refl) -> None:
    """One CUDA thread per column: classic Thompson ``calc_refl10cm``.

    ``graupel_number_shadow`` is WRF classic mp=8's private, per-call ng1d
    moment after sources and number fallout.  It is consumed only here and is
    deliberately absent from the transported/restart state.
    """
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds REFL_KMAX={_KMAX}")
    _check_fields({
        "qv": qv, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
        "graupel_number_shadow": graupel_number_shadow,
        "t": t, "p": p, "refl": refl,
    }, shape)
    rc = radar_init()
    blocks = (ny * nx + _COLUMN_TPB - 1) // _COLUMN_TPB
    _column_kernel("refl10cm_thompson_column", nz)(
        (blocks,), (_COLUMN_TPB,),
        (qv, qr, nr, qs, qg, graupel_number_shadow, t, p,
         _device_tables(), np.float64(rc.k_w),
         np.float64(rc.m_w_0.real), np.float64(rc.m_w_0.imag),
         np.float64(rc.m_i_0.real), np.float64(rc.m_i_0.imag),
         refl, np.int32(nz), np.int32(ny), np.int32(nx)))


class NativeReflectivityScheme(ValueError):
    """The active scheme produces REFL_10CM, but not through this module.

    A ``ValueError`` subclass so every existing caller keeps catching it,
    and a NAMED type so a caller that can use the scheme's own field --
    a radar observation operator, say -- can tell "this scheme has no dBZ"
    apart from "this scheme's dBZ is on the state already".
    """


#: Schemes deliberately OUTSIDE :func:`compute_refl_10cm`'s dispatch even
#: though every one of them produces REFL_10CM, with the reason each is out.
#:
#: The PRODUCER set below (1, 6, 8, 10, 16, 28) and the CONSUMER set
#: ``REFL_10CM_MICROPHYSICS`` (1, 6, 8, 9, 10, 16, 18, 28, 50) differ on
#: purpose, and these keys are exactly the difference: WRF computes their
#: dBZ inside the scheme call and gpuwm stashes the SCHEME's own field, so
#: there is no generic operator call to make.  The asymmetry was already
#: asserted arithmetically in
#: ``tests/test_mp28_runtime_reachability.py::
#: test_every_refl_10cm_admission_constant_admits_28``; it is recorded HERE,
#: at the gate, because a decision that lives only in a test reaches the
#: user as an omission -- which is what the mp=28 reachability audit found
#: and what this table exists to stop repeating.
#:
#: Adding any of these keys to the gate tuple is a DEFECT, not a widening:
#: none of them has a routine here to dispatch to, and the fall-through arm
#: is Kessler's rain-only Marshall-Palmer form.
SCHEME_NATIVE_REFL_10CM: dict[int, str] = {
    9: (
        "Milbrandt-Yau two-moment carries Zet as an INOUT dummy that WRF's "
        "driver binds straight to refl_10cm (module_microphysics_driver.F:"
        "1878), so the scheme writes the slot on every call and "
        "gpuwm/core/milbrandt2.py stashes that array rather than calling a "
        "generic operator."
    ),
    18: (
        "NSSL two-moment has its own S-band diagnostic, radardd02 "
        "(module_mp_nssl_2mom.F), reading five ice categories, five number "
        "moments and two volume moments; gpuwm ports it in "
        "gpuwm.core.nssl2_diagnostics and dispatches to it directly "
        "(gpuwm/da/obsop.py routes mp=18 there before reaching this "
        "module).  It also floors at 0 dBZ, not -35."
    ),
    50: (
        "P3 one-category has no standalone reflectivity routine anywhere "
        "to dispatch to: WRF computes dBZ INSIDE p3_main -- ze_rain from "
        "the scheme's own mu_r/lamr rain PSD (module_mp_p3.F:4755), ze_ice "
        "from the lookup-table normalised reflectivity moment f1pr13 read "
        "at the ice-PSD indices (:4789, accumulated :4868), and diag_ze = "
        "10*log10((ze_rain+ze_ice)*1e18) (:4887) -- and the driver binds "
        "diag_zdbz_3d straight to refl_10cm in CASE(P3_1CATEGORY), passing "
        "no do_radar_ref argument at all "
        "(module_microphysics_driver.F:1557-1602).  P3 predicts ONE ice "
        "category with a rime pair (qir rime mass, qib rime volume) and "
        "allocates NO qs and NO qg, so every branch this dispatcher owns "
        "except Kessler would ask the state for fields P3 never has, and "
        "the Kessler fall-through would report rain-only Rayleigh Z for an "
        "ice-bearing column -- the mp=28-on-Kessler class of defect, in "
        "the reflectivity field this time.  gpuwm mirrors WRF: "
        "gpuwm/core/p3.py stashes the scheme's own zdbz into this module's "
        "refl_10cm slot on every output-due step.  Note the floor: P3's "
        "clear-air value is -99 dBZ (module_mp_p3.F:2273, left standing by "
        "the hydrometeor-free skip at :2439/:3972; mirrored at "
        "gpuwm/core/p3.py:513 and gpuwm/core/kernels/p3.cu:626), NOT the "
        "-35 dBZ this dispatcher's family floors at, so a caller "
        "differencing H(x) against a clear-air observation must take the "
        "floor from the scheme and not from here."
    ),
}


def compute_refl_10cm(
        state, cfg, *, temperature=None, pressure=None,
        thompson_graupel_number=None):
    """Compute REFL_10CM (dBZ, FP32 ``(nz, ny, nx)``).

    The microphysics adapters pass BOTH ``temperature`` and ``pressure``:
    respectively the scheme's post-call temperature and the unchanged
    prepared pressure that WRF passes as ``t1d``/``p1d`` at
    module_mp_morr_two_moment.F:913-914.  Omitting both retains the useful
    standalone diagnostic over the current state, but production history
    output uses the explicit microphysics-time pair (PROVENANCE.md D2).
    Dispatches like ``microphysics.apply``: 1 = Kessler fallback, 6 =
    WSM6 ``refl10cm_wsm6``, 8 and 28 = Thompson ``calc_refl10cm``,
    10 = Morrison ``refl10cm_hm``, and 16 = WDM6 ``refl10cm_wdm6``, whose
    rain intercept is diagnosed from the scheme's own prognostic ``nr``
    rather than fixed.  Thompson requires its output-due private
    graupel-number scratch through ``thompson_graupel_number``.  Results
    land in the persistent
    ``refl_10cm`` scratch slot and are returned.

    The admitted set is NARROWER than the set of schemes that produce
    REFL_10CM, and the difference is recorded rather than implied: 9, 18
    and 50 compute their own dBZ inside the scheme call and refuse here BY
    NAME through :data:`SCHEME_NATIVE_REFL_10CM`, which also says where
    their field actually is.

    mp=28 shares the mp=8 branch VERBATIM, and that is WRF's own structure,
    not a convenience.  ``calc_refl10cm``
    (module_mp_thompson.F:5710-6028) is ONE routine with no
    ``is_aerosol_aware`` branch, reached from the single call site
    ``mp_gt_driver:1458`` -- which is itself gated only on
    ``diagflag .and. do_radar_ref == 1`` (:1449), never on the scheme.
    Its complete argument list
    (:5710-5711) is ``qv1d, qc1d, qr1d, nr1d, qs1d, qg1d, ng1d, qb1d, t1d,
    p1d, dBZ, kts, kte, ii, jj, ke_diag``: no ``nc1d``, no ``nwfa``, no
    ``nifa`` anywhere in the routine's 319 lines.  Cloud water enters only
    as ``rc(k) = MAX(R1, qc1d(k)*rho(k))`` at :5764, and ``rc`` is READ
    NOWHERE ELSE in the routine -- it is a dead local, so no cloud-water and
    no droplet-number information reaches any of the four Rayleigh sums.
    The prognostic droplet number that distinguishes mp=28 from mp=8
    therefore contributes EXACTLY ZERO to REFL_10CM in WRF, and a future
    "improvement" folding cloud water or nc into the sum would be a
    divergence from WRF, not a refinement.  Pinned by
    ``tests/test_mp28_runnable.py::
    test_refl_10cm_is_bit_identical_under_two_very_different_nc_fields``.
    """
    import cupy as cp

    from gpuwm.core import constants as c
    from gpuwm.core.state import DTYPE

    if cfg.mp_physics not in (1, 6, 8, 10, 16, 28):
        native = SCHEME_NATIVE_REFL_10CM.get(int(cfg.mp_physics))
        if native is not None:
            # NOT "no reflectivity": the scheme already produced its own,
            # and saying otherwise is how a reachability gap reads as a
            # missing feature.  Name it, and name where the field is.
            raise NativeReflectivityScheme(
                f"mp_physics={int(cfg.mp_physics)} computes REFL_10CM "
                "itself and is deliberately not dispatched here. "
                + native + "  Read the scheme's own dBZ out of the "
                "DomainState 'refl_10cm' scratch slot, which its adapter "
                "fills on every output-due microphysics step, instead of "
                "recomputing it from another scheme's particle-size "
                "distributions.")
        raise ValueError("do_radar_ref needs an active microphysics scheme "
                         f"(mp_physics 1, 6, 8, 10, 16, or 28), got "
                         f"{cfg.mp_physics}")
    if state.qv is None:
        raise ValueError("reflectivity requires a moist state")
    if (temperature is None) != (pressure is None):
        raise ValueError("temperature and pressure must be supplied together")
    nz, ny, nx = state.p.shape
    if temperature is None:
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        t = state.scratch((nz, ny, nx), "refl_t")
        p = state.p
        # Standalone-current-state path.  Production output passes the
        # microphysics adapter's post-call T and prepared p explicitly.
        t[...] = (thb + state.thp) * cp.power(
            p / DTYPE(c.P0), DTYPE(c.RCP))
    else:
        t = temperature
        p = pressure
    refl = state.scratch((nz, ny, nx), "refl_10cm")
    if cfg.mp_physics == 10:
        required = ("qr", "nr", "qs", "ns", "qg", "ng")
        missing = [name for name in required
                   if getattr(state, name, None) is None]
        if missing:
            raise ValueError("mp_physics=10 reflectivity lacks Morrison "
                             "moments: " + ", ".join(missing))
        launch_refl10cm_morrison(state.qv, state.qr, state.nr,
                                 state.qs, state.ns, state.qg, state.ng,
                                 t, p, refl,
                                 morr_rimed_ice=cfg.morr_rimed_ice)
    elif cfg.mp_physics in (8, 28):
        # ONE branch for both Thompson schemes.  See the docstring: WRF's
        # calc_refl10cm has no aerosol-aware arm and reads no droplet
        # number, so a separate mp=28 branch could only ever differ from
        # WRF.  The required-field list is identical too -- mp=28 allocates
        # a strict superset of mp=8's state.
        missing = [name for name in ("qr", "nr", "qs", "qg")
                   if getattr(state, name, None) is None]
        if missing:
            raise ValueError(
                f"mp_physics={cfg.mp_physics} reflectivity lacks Thompson "
                "fields: " + ", ".join(missing))
        if thompson_graupel_number is None:
            raise ValueError(
                f"mp_physics={cfg.mp_physics} reflectivity requires the "
                "same-call classic graupel number shadow")
        launch_refl10cm_thompson(
            state.qv, state.qr, state.nr, state.qs, state.qg,
            thompson_graupel_number, t, p, refl)
    elif cfg.mp_physics == 16:
        missing = [name for name in ("qr", "nr", "qs", "qg")
                   if getattr(state, name, None) is None]
        if missing:
            raise ValueError("mp_physics=16 reflectivity lacks WDM6 fields: "
                             + ", ".join(missing))
        launch_refl10cm_wdm6(state.qv, state.qr, state.nr, state.qs,
                             state.qg, t, p, refl,
                             hail_opt=cfg.wdm6_hail_opt)
    elif cfg.mp_physics == 6:
        missing = [name for name in ("qr", "qs", "qg")
                   if getattr(state, name, None) is None]
        if missing:
            raise ValueError("mp_physics=6 reflectivity lacks WSM6 fields: "
                             + ", ".join(missing))
        launch_refl10cm_wsm6(state.qv, state.qr, state.qs, state.qg,
                             t, p, refl, hail_opt=cfg.wsm6_hail_opt)
    else:
        launch_refl10cm_kessler(state.qv, state.qr, t, p, refl)
    return refl


def stash_refl_10cm(state, refl) -> None:
    """Stash one output-due reflectivity field on the physics driver.

    A second unconsumed field is a cadence bug, so overwrites fail loudly.
    The array itself is state-owned ``refl_10cm`` scratch; the driver holds
    only the one-frame handoff reference.
    """
    driver = getattr(state, "physics", None)
    if driver is None:
        raise RuntimeError("cannot stash REFL_10CM without a physics driver")
    if driver.refl_10cm is not None:
        raise RuntimeError("REFL_10CM stash was not consumed before reuse")
    driver.refl_10cm = refl


def refl_10cm_is_stashed(state) -> bool:
    """Return the D2 handoff state without exposing its array to callers."""
    driver = getattr(state, "physics", None)
    return driver is not None and driver.refl_10cm is not None


def refl_10cm_stash_is_due(ticks: int, *, domain_start_ticks: int = 0
                           ) -> bool:
    """Does a history frame at ``ticks`` consume a microphysics stash?

    ONE predicate for every consume site, because the answer is a
    property of the handoff and not of the route asking.  A stash exists
    only after a microphysics step OF THAT DOMAIN has run with
    ``refl_10cm_due`` (:func:`stash_refl_10cm` is reachable from nowhere
    else), so the frame due at the domain's own start tick -- its
    analysis frame, before any of its steps -- consumes nothing.

    ``domain_start_ticks`` is that tick, on the domain's own clock
    (``DomainClock.spec.start_ticks``).  It is 0 for a domain that starts
    with the experiment and the ACTIVATION EPOCH for a nest that starts
    later.  Reading the absolute 0 there is defect #205: an activating
    nest's first frame consumed a stash no step had produced, and the
    run died at that frame with "REFL_10CM output is due but no
    microphysics-time field is stashed" -- deterministically, hours of
    integration in.
    """
    return int(ticks) != int(domain_start_ticks)


def domain_start_ticks_of(node) -> int:
    """The start tick of ``node``'s own clock, 0 when it carries none.

    The tree routes hand :class:`gpuwm.core.model.DomainNode` objects to
    the history seam and every one of those carries a
    :class:`gpuwm.core.clock.DomainClock`.  A node without one is a
    single-domain or hand-built caller, whose domain starts with the
    experiment by construction -- the same 0 the field defaults to.
    """
    spec = getattr(getattr(node, "clock", None), "spec", None)
    return int(getattr(spec, "start_ticks", 0) or 0)


def consume_refl_10cm(state):
    """Consume and clear the one-frame REFL_10CM handoff exactly once."""
    driver = getattr(state, "physics", None)
    if driver is None or driver.refl_10cm is None:
        raise RuntimeError(
            "REFL_10CM output is due but no microphysics-time field is stashed")
    refl = driver.refl_10cm
    driver.refl_10cm = None
    return refl


def compute_and_stash_refl_10cm(
        state, cfg, temperature, pressure, *,
        thompson_graupel_number=None) -> None:
    """Compute from WRF's microphysics-time inputs and stage for output."""
    stash_refl_10cm(
        state, compute_refl_10cm(
            state, cfg, temperature=temperature, pressure=pressure,
            thompson_graupel_number=thompson_graupel_number))
