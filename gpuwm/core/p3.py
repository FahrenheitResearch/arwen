"""P3 (Predicted Particle Properties) microphysics, WRF ``mp_physics=50``.

Transcription authority: byte-frozen WRF v4.6.1 ``phys/module_mp_p3.F``
(P3 v4.5.2, Morrison & Milbrandt 2015): ``p3_init`` constants :177-316,
``mp_p3_wrapper_wrf`` :690-932, ``p3_main`` :1905-5201, and the helper
subprograms ``access_lookup_table`` :5205-5243, ``access_lookup_table_coll``
:5246-5313, ``polysvp1`` :6017-6086, ``find_lookupTable_indices_1a/1b/3``
:6316-6410/:6589-6632, ``get_cloud_dsd2`` :6635-6701, ``get_rain_dsd2``
:6705-6780, ``calc_bulkRhoRime`` :6784-6830, ``impose_max_total_Ni``
:6833-6855, ``qv_sat`` :6859-6889.  Every citation ``:NNNN`` below is a
line in that file.

SCOPE — exactly the ``mp_physics = 50`` (``p3_1category``) configuration,
which is how WRF's driver calls the scheme (module_microphysics_driver.F:
1557-1602 passes no ``nc_3d`` and no ``qzi1_3d``; module_physics_init.F:4569
calls ``p3_init('.', 1, .false., 'WRF', ...)``):

* ``nCat = 1``: the multi-category blocks (``iceice_interaction1``
  :2754-2829, ``icecat_destination``, the category-merge section
  :4615-4697, lookup table 2) are structurally unreachable and NOT ported;
  ``mp_physics = 52`` is refused by name in gpuwm/config.py.
* 2-moment ice: the ``log_3momentIce`` branches (``zitot``, the 3-moment
  lookup tables, ``compute_mu_3moment``/``G_of_mu``) are NOT ported;
  ``mp_physics = 53`` is refused by name.
* ``log_predictNc = .false.``: droplet number is the diagnostic
  ``nccnst/rho`` (:2350, :3875); the CCN-activation path and its ``derf``
  are NOT ported; ``mp_physics = 51`` is refused by name.
* ``scpf_on = .false.`` (the wrapper hardcodes it, :815): ``compute_SCPF``'s
  off-branch is transcribed (SCF = iSCF = SPF = iSPF = 1, SPF_clr = 0,
  Qv_cld = qv, :1889-1899); the Sundqvist cloud-fraction branch is NOT
  ported.  ``SCF_out`` is not computed: WRF's wrapper receives it into a
  local ``cldfrac`` it never reads (:811, :868).
* ``typeDiags_ON = .false.``, ``debug_on = .false.`` (wrapper parameters,
  :802-803): the precip-type partition and ``check_values`` are NOT ported.
* The visibility diagnostics (``diag_vis*``) and user diagnostic arrays
  ``diag_2d``/``diag_3d`` are absent/zeroed in the WRF call and are NOT
  ported; ``mflux_r``/``mflux_i`` exist only to feed ``diag_vis2/3``
  (:4989-4997) and are not stored.

TRANSCRIBED QUIRKS (faithful, deliberately not "fixed"):

* The nucleation-possible test keeps Fortran operator precedence
  (``.and.`` binds tighter than ``.or.``), so ``.not. SCPF_on`` guards only
  the warm branch (:2358-2360).
* ``xxlv`` uses the CONSTANT-Lv expression ``3.1484e6-2370.*273.15``
  (:2325, the ``t(i,k)`` factor is commented out in the authority).
* ``t`` is diagnosed once from theta at entry and deliberately NOT
  refreshed after latent heating (:3856-3858 comment); homogeneous
  freezing thresholds use the stale value (:4540, :4578).
* The first-step guard ``max(t_old, 1.)`` (:2329) covers WRF's
  zero-initialized ``th_old`` on the first call.

PRECISION MODEL — float32 throughout (NumPy scalars under NEP 50 weak
promotion), with the authority's explicit double-precision excursions kept
in float64: the ``dexp(dble(...))`` saturation-relaxation factor (:3202,
:3204, :3231) and the rain-freezing log-space products (:3025-3029).
Transcendentals (exp/log/pow/gamma) come from NumPy/libm rather than
gfortran's libm — the same accepted difference class as every other port
here; the WRF-Fortran oracle campaign is the declared next stage and NO
ULP claim is made in this module.

EXECUTION MODEL — this is the CPU float32 authority (scalar per-cell
loops, matching the Fortran's statement order exactly).  There is NO CUDA
mirror yet: the DomainState adapter (:func:`apply`) round-trips device
state through the host, which is correct but slow — suitable for column
smokes and small-domain verification, not production-domain throughput.
The CUDA port is declared future work alongside the oracle campaign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gpuwm.core.p3_tables import (
    DENSIZE,
    ISIZE,
    RCOLLSIZE,
    RIMSIZE,
    generate_rain_tables,
    load_lookup_table_1,
    p3_table_root,
)

f32 = np.float32
f64 = np.float64


def _rescue_overflowed_product(*factors):
    """The same product, evaluated so a PARTIAL product cannot overflow.

    THE DEFECT, measured on a real 6 h forecast, not postulated.  A GFS run
    on the shipped ``p3-mp50-...`` suite went non-finite at step 284 in two
    cells near 198 K, and BOTH arms -- the CUDA kernel and this
    transcription -- produced it identically, out of
    ``module_mp_p3.F:2994-2996``::

        tmp1  = cdist1(i,k)*exp(aimm*(273.15-t(i,k)))
        Q_nuc = cons6*gamma(7.+mu_c(i,k))*tmp1*dum**2

    At the measured cell -- t = 198.99 K, cdist1 = 3.394e5, mu_c = 12.576,
    lamc = 1.6846e4 -- the LEFT-TO-RIGHT single-precision partial product
    ``cons6*gamma*tmp1`` reaches 4.8e45 and overflows to +Inf, while the
    mathematical result, 4.4e19 once the ``dum**2`` = 9.26e-27 factor is
    applied, is an ordinary float32.  The Inf then meets the conservation
    limiter at :3571-3583, where ``ratio = sources/sinks`` is 0 and
    ``qcheti = Inf * 0`` is NaN; that NaN is what reached theta and vapour.

    WRF IS UNDEFINED HERE rather than wrong-but-authoritative.  Fortran does
    not fix the association of ``a*b*c*d``, so whether the expression
    overflows is the compiler's choice, and P3's own authors left the
    double-precision form commented out three lines below the live one
    (:2999-3002) -- they met this.  gpuwm's standing rule for that case is
    to implement defined behaviour and document the divergence, never to be
    bit-exact to a bug.

    THE DIVERGENCE IS EXACTLY THE OVERFLOWING CASE AND NOTHING ELSE.  Every
    caller evaluates WRF's single-precision chain first and reaches for this
    only when the result is not finite, so every value WRF's arithmetic can
    represent is still produced by WRF's arithmetic, bit for bit, and every
    Fortran-oracle receipt already issued still stands.
    """

    product = f64(1.0)
    for factor in factors:
        product = product * f64(factor)
    return f32(product)

# ---------------------------------------------------------------------------
# p3_init physical constants (:177-316), computed in float32 in source order.
# ---------------------------------------------------------------------------

PI = f32(3.14159265)                      # :178
THRD = f32(1.0) / f32(3.0)                # :180
SXTH = f32(1.0) / f32(6.0)                # :181
PIOV3 = PI * THRD                         # :182
PIOV6 = PI * SXTH                         # :183
MAX_TOTAL_NI = f32(2000.0e3)              # :186
IPARAM = 3                                # :193 Khairoutdinov and Kogan 2000
NCCNST = f32(200.0e6)                     # :196
KC = f32(9.44e9)                          # :199 (Seifert-Beheng, iparam=1)
KR = f32(5.78e3)                          # :200
CP = f32(1005.0)                          # :203
INV_CP = f32(1.0) / CP                    # :204
G = f32(9.816)                            # :205
RD = f32(287.15)                          # :206
RV = f32(461.51)                          # :207
EP_2 = f32(0.622)                         # :208
RHOSUR = f32(100000.0) / (RD * f32(273.15))   # :209
RHOSUI = f32(60000.0) / (RD * f32(253.15))    # :210
AR = f32(841.99667)                       # :211
BR = f32(0.8)                             # :212
F1R = f32(0.78)                           # :213
F2R = f32(0.32)                           # :214
ECR = f32(1.0)                            # :215
RHOW = f32(1000.0)                        # :216
CPW = f32(4218.0)                         # :217
INV_RHOW = f32(1.0) / RHOW                # :218
MU_R_CONSTANT = f32(0.0)                  # :219
INV_DRMAX = f32(1.0) / f32(0.002)         # :222
RHO_RIMEMIN = f32(50.0)                   # :225
RHO_RIMEMAX = f32(900.0)                  # :226
INV_RHO_RIMEMAX = f32(1.0) / RHO_RIMEMAX  # :227
QSMALL = f32(1.0e-14)                     # :230
NSMALL = f32(1.0e-16)                     # :231
BSMALL = QSMALL * INV_RHO_RIMEMAX         # :232
BIMM = f32(2.0)                           # :239 Barklie and Gokhale (1959)
AIMM = f32(0.65)                          # :240
RIN = f32(0.1e-6)                         # :241
MI0 = f32(4.0) * PIOV3 * f32(900.0) * f32(1.0e-18)   # :242
ECI = f32(0.5)                            # :244
ERI = f32(1.0)                            # :245
BCN = f32(2.0)                            # :246
DBRK = f32(600.0e-6)                      # :249
NMLTRATIO = f32(1.0)                      # :251
MU_I_INITIAL = f32(10.0)                  # :254 (3-moment only; unused here)
CONS1 = PIOV6 * RHOW                                       # :259
CONS2 = f32(4.0) * PIOV3 * RHOW                            # :260
CONS3 = f32(1.0) / (CONS2 * f32(25.0e-6) ** f32(3.0))      # :261
CONS4 = f32(1.0) / (DBRK ** f32(3.0) * PI * RHOW)          # :262
CONS5 = PIOV6 * BIMM                                       # :263
CONS6 = PIOV6 ** f32(2.0) * RHOW * BIMM                    # :264
CONS7 = f32(4.0) * PIOV3 * RHOW * f32(1.0e-6) ** f32(3.0)  # :265
CONS8 = f32(1.0) / (CONS2 * f32(40.0e-6) ** f32(3.0))      # :266

#: Droplet mass spectral shape values (:301-316), Seifert-Beheng only
#: (iparam = 1); kept for textual parity, unread under iparam = 3.
DNU = np.asarray([0.0, -0.557, -0.430, -0.307, -0.186, -0.067, 0.050,
                  0.167, 0.282, 0.397, 0.512, 0.626, 0.739, 0.853,
                  0.966, 0.966], dtype=np.float32)


def polysvp1(t: np.float32, i_type: int) -> np.float32:
    """Saturation vapor pressure [Pa] (:6017-6086), float32.

    Flatau et al. (1992) polynomials above 195.8 K (ice) / 202.0 K
    (liquid), Goff-Gratch below.
    """
    t = f32(t)
    if i_type == 1 and t < f32(273.15):
        if t >= f32(195.8):
            dt = t - f32(273.15)
            a0i, a1i, a2i = f32(6.11147274), f32(0.503160820), f32(0.188439774e-1)
            a3i, a4i, a5i = f32(0.420895665e-3), f32(0.615021634e-5), f32(0.602588177e-7)
            a6i, a7i, a8i = f32(0.385852041e-9), f32(0.146898966e-11), f32(0.252751365e-14)
            out = a0i + dt * (a1i + dt * (a2i + dt * (a3i + dt * (a4i + dt * (
                a5i + dt * (a6i + dt * (a7i + a8i * dt)))))))
            return out * f32(100.0)
        return f32(10.0) ** (f32(-9.09718) * (f32(273.16) / t - f32(1.0))
                             - f32(3.56654) * np.log10(f32(273.16) / t)
                             + f32(0.876793) * (f32(1.0) - t / f32(273.16))
                             + np.log10(f32(6.1071))) * f32(100.0)
    # liquid branch (i_type == 0 .or. t >= 273.15)
    if t >= f32(202.0):
        dt = t - f32(273.15)
        a0, a1, a2 = f32(6.11239921), f32(0.443987641), f32(0.142986287e-1)
        a3, a4, a5 = f32(0.264847430e-3), f32(0.302950461e-5), f32(0.206739458e-7)
        a6, a7, a8 = f32(0.640689451e-10), f32(-0.952447341e-13), f32(-0.976195544e-15)
        out = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt * (a4 + dt * (
            a5 + dt * (a6 + dt * (a7 + a8 * dt)))))))
        return out * f32(100.0)
    return f32(10.0) ** (f32(-7.90298) * (f32(373.16) / t - f32(1.0))
                         + f32(5.02808) * np.log10(f32(373.16) / t)
                         - f32(1.3816e-7) * (f32(10.0) ** (f32(11.344) * (
                             f32(1.0) - t / f32(373.16))) - f32(1.0))
                         + f32(8.1328e-3) * (f32(10.0) ** (f32(-3.49149) * (
                             f32(373.16) / t - f32(1.0))) - f32(1.0))
                         + np.log10(f32(1013.246))) * f32(100.0)


#: Saturation pressure at T = 0 C (:257) -- depends on polysvp1, so defined
#: after it, exactly as p3_init orders the assignments.
E0 = polysvp1(f32(273.15), 0)


def qv_sat(t_atm: np.float32, p_atm: np.float32, i_wrt: int) -> np.float32:
    """Saturation mixing ratio w.r.t. liquid (0) or ice (1) (:6859-6889)."""
    e_pres = polysvp1(t_atm, i_wrt)
    return EP_2 * e_pres / max(f32(1.0e-3), p_atm - e_pres)


def _gammaf(x) -> np.float32:
    """Fortran REAL ``gamma`` intrinsic: float32 result via libm gamma."""
    return f32(math.gamma(float(x)))


@dataclass(frozen=True)
class P3Runtime:
    """The p3_init products the solver reads (tables; constants are module
    level).  ``itab``/``itabcoll`` come from the SHA-256-validated packaged
    table; ``vn/vm/revap`` are the generated rain tables."""
    itab: np.ndarray        # (densize, rimsize, isize, tabsize)
    itabcoll: np.ndarray    # (densize, rimsize, isize, rcollsize, 2)
    vn_table: np.ndarray    # (300, 10)
    vm_table: np.ndarray    # (300, 10)
    revap_table: np.ndarray  # (300, 10)


_RUNTIME_CACHE: dict[str, P3Runtime] = {}


def p3_init() -> P3Runtime:
    """The WRF ``CASE (P3_1CATEGORY): CALL p3_init('.',1,.false.,'WRF',...)``
    equivalent (module_physics_init.F:4568-4569): read table 1 (2momI) and
    generate the rain tables, once per process per table root."""
    root = p3_table_root()
    runtime = _RUNTIME_CACHE.get(root)
    if runtime is None:
        itab, itabcoll = load_lookup_table_1(root)
        vn, vm, revap = generate_rain_tables()
        runtime = P3Runtime(itab=itab, itabcoll=itabcoll, vn_table=vn,
                            vm_table=vm, revap_table=revap)
        _RUNTIME_CACHE[root] = runtime
    return runtime


# ---------------------------------------------------------------------------
# Lookup-table access and index helpers (1-based Fortran index convention is
# kept in the integer values; the -1 shift happens only at array access).
# ---------------------------------------------------------------------------

def access_lookup_table(itab, dumjj, dumii, dumi, index, dum1, dum4, dum5):
    """Trilinear interpolation in the main ice table (:5205-5243)."""
    t = itab
    iproc1 = t[dumjj-1, dumii-1, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj-1, dumii-1, dumi, index-1] - t[dumjj-1, dumii-1, dumi-1, index-1])
    gproc1 = t[dumjj-1, dumii, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj-1, dumii, dumi, index-1] - t[dumjj-1, dumii, dumi-1, index-1])
    tmp1 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    iproc1 = t[dumjj, dumii-1, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj, dumii-1, dumi, index-1] - t[dumjj, dumii-1, dumi-1, index-1])
    gproc1 = t[dumjj, dumii, dumi-1, index-1] + (dum1 - f32(dumi)) * (
        t[dumjj, dumii, dumi, index-1] - t[dumjj, dumii, dumi-1, index-1])
    tmp2 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    return tmp1 + (dum5 - f32(dumjj)) * (tmp2 - tmp1)


def access_lookup_table_coll(itabcoll, dumjj, dumii, dumj, dumi, index,
                             dum1, dum3, dum4, dum5):
    """Quadrilinear interpolation for ice-rain collection (:5246-5313)."""
    t = itabcoll

    def pair(jj, ii):
        dproc1 = t[jj, ii, dumi-1, dumj-1, index-1] + (dum1 - f32(dumi)) * (
            t[jj, ii, dumi, dumj-1, index-1] - t[jj, ii, dumi-1, dumj-1, index-1])
        dproc2 = t[jj, ii, dumi-1, dumj, index-1] + (dum1 - f32(dumi)) * (
            t[jj, ii, dumi, dumj, index-1] - t[jj, ii, dumi-1, dumj, index-1])
        return dproc1 + (dum3 - f32(dumj)) * (dproc2 - dproc1)

    iproc1 = pair(dumjj-1, dumii-1)
    gproc1 = pair(dumjj-1, dumii)
    tmp1 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    iproc1 = pair(dumjj, dumii-1)
    gproc1 = pair(dumjj, dumii)
    tmp2 = iproc1 + (dum4 - f32(dumii)) * (gproc1 - iproc1)
    return tmp1 + (dum5 - f32(dumjj)) * (tmp2 - tmp1)


def find_lookupTable_indices_1a(qitot, nitot, qirim, rhop):
    """Main ice-table indices (:6316-6370): returns
    (dumi, dumjj, dumii, dum1, dum4, dum5)."""
    dum1 = (np.log10(qitot / nitot) + f32(18.0)) * f32(3.444606) - f32(10.0)
    dumi = int(dum1)
    dum1 = min(dum1, f32(ISIZE))
    dum1 = max(dum1, f32(1.0))
    dumi = max(1, dumi)
    dumi = min(ISIZE - 1, dumi)

    dum4 = (qirim / qitot) * f32(3.0) + f32(1.0)
    dumii = int(dum4)
    dum4 = min(dum4, f32(RIMSIZE))
    dum4 = max(dum4, f32(1.0))
    dumii = max(1, dumii)
    dumii = min(RIMSIZE - 1, dumii)

    if rhop <= f32(650.0):
        dum5 = (rhop - f32(50.0)) * f32(0.005) + f32(1.0)
    else:
        dum5 = (rhop - f32(650.0)) * f32(0.004) + f32(4.0)
    dumjj = int(dum5)
    dum5 = min(dum5, f32(DENSIZE))
    dum5 = max(dum5, f32(1.0))
    dumjj = max(1, dumjj)
    dumjj = min(DENSIZE - 1, dumjj)
    return dumi, dumjj, dumii, dum1, dum4, dum5


def find_lookupTable_indices_1b(qr, nr):
    """Rain index for ice-rain collection (:6374-6410): (dumj, dum3)."""
    if qr >= QSMALL and nr > f32(0.0):
        dumlr = (qr / (PI * RHOW * nr)) ** THRD
        dum3 = (np.log10(f32(1.0) * dumlr) + f32(5.0)) * f32(10.70415)
        dumj = int(dum3)
        dum3 = min(dum3, f32(RCOLLSIZE))
        dum3 = max(dum3, f32(1.0))
        dumj = max(1, dumj)
        dumj = min(RCOLLSIZE - 1, dumj)
    else:
        dumj = 1
        dum3 = f32(1.0)
    return dumj, dum3


def find_lookupTable_indices_3(mu_r, lamr):
    """Rain fallspeed/ventilation table indices (:6589-6632):
    (dumii, dumjj, dum1, rdumii, rdumjj, inv_dum3)."""
    dum1 = (mu_r + f32(1.0)) / lamr
    if dum1 <= f32(195.0e-6):
        inv_dum3 = f32(0.1)
        rdumii = (dum1 * f32(1.0e6) + f32(5.0)) * inv_dum3
        rdumii = max(rdumii, f32(1.0))
        rdumii = min(rdumii, f32(20.0))
        dumii = int(rdumii)
        dumii = max(dumii, 1)
        dumii = min(dumii, 20)
    else:
        inv_dum3 = THRD * f32(0.1)
        rdumii = (dum1 * f32(1.0e6) - f32(195.0)) * inv_dum3 + f32(20.0)
        rdumii = max(rdumii, f32(20.0))
        rdumii = min(rdumii, f32(300.0))
        dumii = int(rdumii)
        dumii = max(dumii, 20)
        dumii = min(dumii, 299)
    rdumjj = mu_r + f32(1.0)
    rdumjj = max(rdumjj, f32(1.0))
    rdumjj = min(rdumjj, f32(10.0))
    dumjj = int(rdumjj)
    dumjj = max(dumjj, 1)
    dumjj = min(dumjj, 9)
    return dumii, dumjj, dum1, rdumii, rdumjj, inv_dum3


def get_cloud_dsd2(qc_grd, nc_grd, rho, iscf):
    """Cloud DSD parameters (:6635-6701).

    Returns ``(nc_grd, mu_c, nu, lamc, cdist, cdist1)`` -- ``nc_grd`` is the
    (possibly lambda-limited) modified grid-mean droplet number the Fortran
    passes back through its INOUT argument.
    """
    qc = qc_grd * iscf
    if qc >= QSMALL:
        nc = nc_grd * iscf
        nc = max(nc, NSMALL)
        mu_c = f32(0.0005714) * (nc * f32(1.0e-6) * rho) + f32(0.2714)
        mu_c = f32(1.0) / (mu_c * mu_c) - f32(1.0)  # 1./(mu_c**2)-1.
        mu_c = max(mu_c, f32(2.0))
        mu_c = min(mu_c, f32(15.0))
        if IPARAM == 1:  # pragma: no cover - iparam is pinned to 3 (:193)
            dumi = int(mu_c)
            nu = DNU[dumi-1] + (DNU[dumi] - DNU[dumi-1]) * (mu_c - f32(dumi))
        else:
            nu = f32(0.0)
        lamc = (CONS1 * nc * (mu_c + f32(3.0)) * (mu_c + f32(2.0))
                * (mu_c + f32(1.0)) / qc) ** THRD
        lammin = (mu_c + f32(1.0)) * f32(2.5e4)
        lammax = (mu_c + f32(1.0)) * f32(1.0e6)
        if lamc < lammin:
            lamc = lammin
            nc = f32(6.0) * lamc ** f32(3.0) * qc / (
                PI * RHOW * (mu_c + f32(3.0)) * (mu_c + f32(2.0))
                * (mu_c + f32(1.0)))
        elif lamc > lammax:
            lamc = lammax
            nc = f32(6.0) * lamc ** f32(3.0) * qc / (
                PI * RHOW * (mu_c + f32(3.0)) * (mu_c + f32(2.0))
                * (mu_c + f32(1.0)))
        cdist = nc * (mu_c + f32(1.0)) / lamc
        cdist1 = nc / _gammaf(mu_c + f32(1.0))
        nc_grd = nc / iscf
        return nc_grd, mu_c, nu, lamc, cdist, cdist1
    return nc_grd, f32(0.0), f32(0.0), f32(0.0), f32(0.0), f32(0.0)


def get_rain_dsd2(qr_grd, nr_grd, ispf):
    """Rain DSD parameters (:6705-6780).

    Returns ``(nr_grd, mu_r, lamr, cdistr, logn0r)`` -- ``nr_grd`` is the
    modified grid-mean rain number.
    """
    qr = qr_grd * ispf
    if qr >= QSMALL:
        nr = nr_grd * ispf
        nr = max(nr, NSMALL)
        # inv_dum feeds only the commented-out variable-mu_r path; computed
        # for parity with the authority's statement order (:6734).
        _ = (qr / (CONS1 * nr * f32(6.0))) ** THRD
        mu_r = MU_R_CONSTANT
        lamr = (CONS1 * nr * (mu_r + f32(3.0)) * (mu_r + f32(2.0))
                * (mu_r + f32(1.0)) / qr) ** THRD
        lammax = (mu_r + f32(1.0)) * f32(1.0e5)
        lammin = (mu_r + f32(1.0)) * INV_DRMAX
        if lamr < lammin:
            lamr = lammin
            nr = np.exp(f32(3.0) * np.log(lamr) + np.log(qr)
                        + np.log(_gammaf(mu_r + f32(1.0)))
                        - np.log(_gammaf(mu_r + f32(4.0)))) / CONS1
        elif lamr > lammax:
            lamr = lammax
            nr = np.exp(f32(3.0) * np.log(lamr) + np.log(qr)
                        + np.log(_gammaf(mu_r + f32(1.0)))
                        - np.log(_gammaf(mu_r + f32(4.0)))) / CONS1
        logn0r = np.log10(nr) + (mu_r + f32(1.0)) * np.log10(lamr) \
            - np.log10(_gammaf(mu_r + f32(1.0)))
        cdistr = nr / _gammaf(mu_r + f32(1.0))
        nr_grd = nr / ispf
        return nr_grd, mu_r, lamr, cdistr, logn0r
    return nr_grd, f32(0.0), f32(0.0), f32(0.0), f32(0.0)


def calc_bulkRhoRime(qi_tot, qi_rim, bi_rim):
    """Bulk rime density with qirim/birim consistency (:6784-6830).

    Returns ``(qi_rim, bi_rim, rho_rime)``.
    """
    if bi_rim >= f32(1.0e-15):
        rho_rime = qi_rim / bi_rim
        if rho_rime < RHO_RIMEMIN:
            rho_rime = RHO_RIMEMIN
            bi_rim = qi_rim / rho_rime
        elif rho_rime > RHO_RIMEMAX:
            rho_rime = RHO_RIMEMAX
            bi_rim = qi_rim / rho_rime
    else:
        qi_rim = f32(0.0)
        bi_rim = f32(0.0)
        rho_rime = f32(0.0)
    if qi_rim > qi_tot and rho_rime > f32(0.0):
        qi_rim = qi_tot
        bi_rim = qi_rim / rho_rime
    if qi_rim < QSMALL:
        qi_rim = f32(0.0)
        bi_rim = f32(0.0)
    return qi_rim, bi_rim, rho_rime


def impose_max_total_Ni(nitot, inv_rho_local):
    """nCat = 1 form of the total ice number cap (:6833-6855)."""
    if nitot >= f32(1.0e-20):
        dum = MAX_TOTAL_NI * inv_rho_local / nitot
        nitot = nitot * min(dum, f32(1.0))
    return nitot


# ---------------------------------------------------------------------------
# p3_main (:1905-5201), nCat = 1 / 2-moment ice / WRF orientation.
# ---------------------------------------------------------------------------

def p3_main(qc, nc, qr, nr, th_old, th, qv_old, qv, dt, qitot, qirim, nitot,
            birim, ssat, pres, dzq, it, prt_liq, prt_sol,
            diag_ze, diag_effc, diag_effi, diag_vmi, diag_di, diag_rhoi,
            *, n_cat: int = 1, log_predictNc: bool = False,
            model: str = "WRF", clbfact_dep: float = 1.0,
            clbfact_sub: float = 1.0, runtime: P3Runtime | None = None
            ) -> None:
    """The P3 main solver on an (ni, nk) slab, k index 0 = surface.

    All (ni, nk) float32 arrays are updated in place exactly as the
    Fortran INOUT arguments are; ``prt_liq``/``prt_sol`` (ni,) receive the
    liquid/solid surface precipitation rates [m s-1].  ``uzpl`` (vertical
    velocity) is omitted: the authority reads one element solely to
    silence an unused-argument warning (:2204).
    """
    if n_cat != 1:
        raise NotImplementedError(
            "gpuwm's P3 port is the 1-category configuration (mp_physics="
            "50); nCat>1 (mp_physics=52) is not ported.")
    if log_predictNc:
        raise NotImplementedError(
            "gpuwm's P3 port specifies droplet number (mp_physics=50); the "
            "prognostic-Nc path (mp_physics=51) is not ported.")
    if model != "WRF":
        raise NotImplementedError("only the WRF orientation is ported")
    rt = runtime if runtime is not None else p3_init()
    itab = rt.itab
    itabcoll = rt.itabcoll
    vn_table = rt.vn_table
    vm_table = rt.vm_table
    revap_table = rt.revap_table

    ni_pts, nk = qc.shape
    kbot, ktop, kdir = 0, nk - 1, 1     # WRF orientation (:2211-2215)

    dt = f32(dt)
    odt = f32(1.0) / dt                 # :2261
    # timeScaleFactor (:2265) feeds only the commented-out soft rain-lambda
    # limiter (:4741-4750); not carried.

    inv_dzq = (f32(1.0) / dzq).astype(np.float32)   # :2260

    prt_liq[:] = f32(0.0)
    prt_sol[:] = f32(0.0)
    diag_ze[:] = f32(-99.0)
    diag_effc[:] = f32(10.0e-6)
    diag_effi[:] = f32(25.0e-6)
    diag_vmi[:] = f32(0.0)
    diag_di[:] = f32(0.0)
    diag_rhoi[:] = f32(0.0)
    ze_ice = np.full((ni_pts, nk), f32(1.0e-22), dtype=np.float32)
    ze_rain = np.full((ni_pts, nk), f32(1.0e-22), dtype=np.float32)
    rhorime_c = f32(400.0)              # :2290 (assigned once, outside loops)

    # :2293-2297 -- slab-wide diagnostics at entry.
    tmparr1 = ((pres * f32(1.0e-5)) ** (RD * INV_CP)).astype(np.float32)
    invexn = (f32(1.0) / tmparr1).astype(np.float32)
    t = (th * tmparr1).astype(np.float32)
    t_old = (th_old * tmparr1).astype(np.float32)
    np.maximum(qv, f32(0.0), out=qv)

    # constant-Lv thermodynamic coefficients (:2325-2327) -- these carry no
    # t dependence in the authority, so they are plain scalars here.
    xxlv = f32(3.1484e6) - f32(2370.0) * f32(273.15)
    xxls = xxlv + f32(0.3337e6)
    xlf = xxls - xxlv

    for i in range(ni_pts):
        log_hydrometeorsPresent = False
        log_nucleationPossible = False

        rho = np.empty(nk, dtype=np.float32)
        inv_rho = np.empty(nk, dtype=np.float32)
        qvs = np.empty(nk, dtype=np.float32)
        qvi = np.empty(nk, dtype=np.float32)
        sup = np.empty(nk, dtype=np.float32)
        supi = np.empty(nk, dtype=np.float32)
        rhofacr = np.empty(nk, dtype=np.float32)
        rhofaci = np.empty(nk, dtype=np.float32)
        acn = np.empty(nk, dtype=np.float32)

        # ------------------------------------------------------------------
        # k_loop_1 (:2320-2411): atmospheric variables + dry-air clipping.
        # ------------------------------------------------------------------
        for k in range(nk):
            rho[k] = pres[i, k] / (RD * t[i, k])
            inv_rho[k] = f32(1.0) / rho[k]
            # first-call guard: t_old is 0 before p3_main ever ran (:2328-2330)
            qvs[k] = qv_sat(max(t_old[i, k], f32(1.0)), pres[i, k], 0)
            qvi[k] = qv_sat(max(t_old[i, k], f32(1.0)), pres[i, k], 1)
            # THE COLD-START qvs/qvi FLOOR, DEFAULT-ON IN ALL THREE ARMS
            # (2026-08-31; ``max(qvs, 1.e-20)`` after both qv_sat calls,
            # the remedy P3 releases newer than this port's v4.5.2
            # transcription target carry themselves).  On
            # the very first p3_main call both th_old and qv_old are the
            # zero the allocation left (WRF never initialises them either
            # -- Registry.EM_COMMON:1598-1599 declare them plain state
            # and no real.exe or start_em path writes them); WRF's
            # `max(t_old,1.)` guard at :2329 keeps polysvp1's argument
            # finite, but polysvp1(1 K) underflows to 0, so stock qvs and
            # qvi are BOTH exactly 0 and the sup/supi diagnoses below
            # evaluate 0.0/0.0 -- a quiet domain-wide NaN this port used
            # to TRANSCRIBE deliberately.  The floor pins step-1 sup/supi
            # at exactly -1 (fully subsaturated, the intended meaning)
            # and is inert from step 2 on, once t_old is real.  Never
            # bit-exact to a bug: where the reference computes an
            # undefined 0/0, this port implements the defined behaviour
            # and documents the divergence, here.
            #
            # This is a REAL behavior change beyond the NaN cosmetics
            # and it is DECLARED, not hidden: with sup = -1 finite, the
            # step-1 dry-air clip below (`qc < 1e-8 .and. sup < -0.1`)
            # can fire in cells whose wrfinput carries trace condensate,
            # where NaN made every comparison false.  Runs are therefore
            # bit-identical to shipped 2.6.0 from step 2 onward; step 1
            # carries exactly this documented floor delta.  The old
            # NaN-containment pin flipped to the floor pin:
            # tests/test_p3_port.py::
            # test_the_first_step_qvs_floor_pins_sup_at_minus_one.
            # The rejected alternative -- an opt-in flag, or a floor on
            # one arm only -- would either ship the defect in a bare
            # default run ("fixed means default") or fork the core into
            # two step-1 behaviors under one module, the exact property
            # the three-arm byte gate exists to protect.
            qvs[k] = max(qvs[k], f32(1.0e-20))
            qvi[k] = max(qvi[k], f32(1.0e-20))
            # log_predictSsat = .false. (:2252) -> always the diagnosed branch
            ssat[i, k] = qv_old[i, k] - qvs[k]
            sup[k] = qv_old[i, k] / qvs[k] - f32(1.0)
            supi[k] = qv_old[i, k] / qvi[k] - f32(1.0)
            rhofacr[k] = (RHOSUR * inv_rho[k]) ** f32(0.54)
            rhofaci[k] = (RHOSUI * inv_rho[k]) ** f32(0.54)
            dum = f32(1.496e-6) * t[i, k] ** f32(1.5) / (t[i, k] + f32(120.0))
            acn[k] = G * RHOW / (f32(18.0) * dum)
            # specified droplet number (1-moment cloud, :2349-2351)
            nc[i, k] = NCCNST * inv_rho[k]

            # Fortran precedence quirk preserved: A .or. (B .and. .not.SCPF)
            # (:2358-2360; scpf_on is hardwired false in the WRF wrapper)
            if ((t[i, k] < f32(273.15) and supi[k] >= f32(-0.05))
                    or (t[i, k] >= f32(273.15) and sup[k] >= f32(-0.05))):
                log_nucleationPossible = True

            # mass clipping in dry air (:2365-2407)
            if qc[i, k] < QSMALL or (qc[i, k] < f32(1.0e-8)
                                     and sup[k] < f32(-0.1)):
                qv[i, k] = qv[i, k] + qc[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qc[i, k] * xxlv * INV_CP
                qc[i, k] = f32(0.0)
                nc[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qr[i, k] < QSMALL or (qr[i, k] < f32(1.0e-8)
                                     and sup[k] < f32(-0.1)):
                qv[i, k] = qv[i, k] + qr[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qr[i, k] * xxlv * INV_CP
                qr[i, k] = f32(0.0)
                nr[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qitot[i, k] < QSMALL or (qitot[i, k] < f32(1.0e-8)
                                        and supi[k] < f32(-0.1)):
                qv[i, k] = qv[i, k] + qitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xxls * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if (qitot[i, k] >= QSMALL and qitot[i, k] < f32(1.0e-8)
                    and t[i, k] >= f32(273.15)):
                qr[i, k] = qr[i, k] + qitot[i, k]
                nr[i, k] = nr[i, k] + nitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xlf * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)

        # first compute_SCPF (:2432-2434), scpf off-branch (:1889-1899):
        # SCF = iSCF = SPF = iSPF = 1, SPF_clr = 0, Qv_cld = qv snapshot.
        iscf = f32(1.0)
        scf = f32(1.0)
        spf = f32(1.0)
        ispf = f32(1.0)
        spf_clr = f32(0.0)
        qv_cld = qv[i, :].copy()

        if not (log_nucleationPossible or log_hydrometeorsPresent):
            continue    # goto 333 (:2439); SCF_out is discarded by WRF
        log_hydrometeorsPresent = False   # :2441

        # ------------------------------------------------------------------
        # k_loop_main (:2449-3927): process rates + prognostic update.
        # ------------------------------------------------------------------
        for k in range(nk):
            # 555-skip: dry + subsaturated level with no hydrometeors
            log_exitlevel = not (qc[i, k] >= QSMALL or qr[i, k] >= QSMALL
                                 or qitot[i, k] >= QSMALL)
            if log_exitlevel and (
                    (t[i, k] < f32(273.15) and supi[k] < f32(-0.05))
                    or (t[i, k] >= f32(273.15) and sup[k] < f32(-0.05))):
                continue    # goto 555 (:2462-2464)

            # zero process rates (:2467-2485); scalars, nCat = 1
            qcacc = qrevp = qccon = f32(0.0)
            qcaut = qcevp = qrcon = f32(0.0)
            ncacc = ncnuc = ncslf = f32(0.0)
            ncautc = qcnuc = nrslf = f32(0.0)
            nrevp = ncautr = f32(0.0)
            qchetc = qisub = nrshdr = f32(0.0)
            qcheti = qrcol = qcshd = f32(0.0)
            qrhetc = qimlt = qccol = f32(0.0)
            qrheti = qinuc = nimlt = f32(0.0)
            nchetc = nccol = ncshdc = f32(0.0)
            ncheti = nrcol = nislf = f32(0.0)
            nrhetc = ninuc = qidep = f32(0.0)
            nrheti = nisub = qwgrth = f32(0.0)
            qrmul = nimul = f32(0.0)
            log_wetgrowth = False
            # predict_supersaturation (:2488-2519): log_predictSsat = false.

            # 444-skip: no hydrometeors -> jump to nucleation (:2523-2528)
            log_exitlevel = not (qc[i, k] >= QSMALL or qr[i, k] >= QSMALL
                                 or qitot[i, k] >= QSMALL)

            # values shared between the process region and the 444 region
            dqsdT = f32(0.0)
            epsi = f32(0.0)
            epsi_tot = f32(0.0)
            f1pr04 = f1pr05 = f1pr09 = f1pr10 = f1pr14 = f32(0.0)
            f1pr02 = f1pr03 = f32(0.0)
            f1pr07 = f1pr08 = f32(-99.0)
            mu_c = lamc = cdist = cdist1 = f32(0.0)
            mu_r = lamr = cdistr = logn0r = f32(0.0)
            abi = f32(1.0)

            if not log_exitlevel:
                # time/space varying physical variables (:2531-2554)
                mu = f32(1.496e-6) * t[i, k] ** f32(1.5) / (t[i, k] + f32(120.0))
                dv = f32(8.794e-5) * t[i, k] ** f32(1.81) / pres[i, k]
                sc = mu / (rho[k] * dv)
                dum = f32(1.0) / (RV * t[i, k] ** f32(2.0))
                dqsdT = xxlv * qvs[k] * dum
                dqsidT = xxls * qvi[k] * dum
                ab = f32(1.0) + dqsdT * xxlv * INV_CP
                abi = f32(1.0) + dqsidT * xxls * INV_CP
                kap = f32(1.414e3) * mu
                if t[i, k] < f32(253.15):
                    eii = f32(0.001)
                elif t[i, k] < f32(273.15):
                    eii = f32(0.001) + (t[i, k] - f32(253.15)) * (
                        f32(0.3) - f32(0.001)) / f32(20.0)
                else:
                    eii = f32(0.3)

                nc[i, k], mu_c, nu, lamc, cdist, cdist1 = get_cloud_dsd2(
                    qc[i, k], nc[i, k], rho[k], iscf)
                nr[i, k], mu_r, lamr, cdistr, logn0r = get_rain_dsd2(
                    qr[i, k], nr[i, k], ispf)

                epsi_tot = f32(0.0)   # :2564
                nitot[i, k] = impose_max_total_Ni(nitot[i, k], inv_rho[k])

                eii_fact = f32(1.0)
                if qitot[i, k] >= QSMALL:   # qitot_notsmall_1 (:2570-2679)
                    nitot[i, k] = max(nitot[i, k], NSMALL)
                    nr[i, k] = max(nr[i, k], NSMALL)
                    # mean-mass ice diameter (:2577-2578) feeds only the
                    # rime-splintering/multicat paths (off at nCat=1).
                    qirim[i, k], birim[i, k], rhop = calc_bulkRhoRime(
                        qitot[i, k], qirim[i, k], birim[i, k])
                    dumi, dumjj, dumii, dum1, dum4, dum5 = \
                        find_lookupTable_indices_1a(
                            qitot[i, k], nitot[i, k], qirim[i, k], rhop)
                    dumj, dum3 = find_lookupTable_indices_1b(qr[i, k], nr[i, k])

                    f1pr02 = access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                                 dum1, dum4, dum5)
                    f1pr03 = access_lookup_table(itab, dumjj, dumii, dumi, 3,
                                                 dum1, dum4, dum5)
                    f1pr04 = access_lookup_table(itab, dumjj, dumii, dumi, 4,
                                                 dum1, dum4, dum5)
                    f1pr05 = access_lookup_table(itab, dumjj, dumii, dumi, 5,
                                                 dum1, dum4, dum5)
                    f1pr09 = access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                                 dum1, dum4, dum5)
                    f1pr10 = access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                                 dum1, dum4, dum5)
                    f1pr14 = access_lookup_table(itab, dumjj, dumii, dumi, 10,
                                                 dum1, dum4, dum5)
                    if qr[i, k] >= QSMALL:
                        f1pr07 = access_lookup_table_coll(
                            itabcoll, dumjj, dumii, dumj, dumi, 1,
                            dum1, dum3, dum4, dum5)
                        f1pr08 = access_lookup_table_coll(
                            itabcoll, dumjj, dumii, dumj, dumi, 2,
                            dum1, dum3, dum4, dum5)
                    else:
                        f1pr07 = f32(-99.0)
                        f1pr08 = f32(-99.0)

                    # lambda limiters on nitot (:2649-2650)
                    nitot[i, k] = min(nitot[i, k], f1pr09 * qitot[i, k])
                    nitot[i, k] = max(nitot[i, k], f1pr10 * qitot[i, k])

                    # ice-ice collection shutoff factor (:2665-2677)
                    if qirim[i, k] > f32(0.0):
                        tmp1 = qirim[i, k] / qitot[i, k]
                        if tmp1 < f32(0.6):
                            eii_fact = f32(1.0)
                        elif tmp1 < f32(0.9):
                            eii_fact = f32(1.0) - (tmp1 - f32(0.6)) / f32(0.3)
                        else:
                            eii_fact = f32(0.0)
                    else:
                        eii_fact = f32(1.0)

                # -- collection of droplets (:2697-2710)
                if (qitot[i, k] >= QSMALL and qc[i, k] >= QSMALL
                        and t[i, k] <= f32(273.15)):
                    qccol = rhofaci[k] * f1pr04 * qc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                    nccol = rhofaci[k] * f1pr04 * nc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                if (qitot[i, k] >= QSMALL and qc[i, k] >= QSMALL
                        and t[i, k] > f32(273.15)):
                    qcshd = rhofaci[k] * f1pr04 * qc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                    nccol = rhofaci[k] * f1pr04 * nc[i, k] * ECI * rho[k] \
                        * nitot[i, k] * iscf
                    ncshdc = qcshd * f32(1.923e6)

                # -- collection of rain (:2725-2748); SPF-SPF_clr = 1 here
                if (qitot[i, k] >= QSMALL and qr[i, k] >= QSMALL
                        and t[i, k] <= f32(273.15)):
                    qrcol = f32(10.0) ** (f1pr08 + logn0r) * rho[k] \
                        * rhofaci[k] * ERI * nitot[i, k] * iscf \
                        * (spf - spf_clr)
                    nrcol = f32(10.0) ** (f1pr07 + logn0r) * rho[k] \
                        * rhofaci[k] * ERI * nitot[i, k] * iscf \
                        * (spf - spf_clr)
                if (qitot[i, k] >= QSMALL and qr[i, k] >= QSMALL
                        and t[i, k] > f32(273.15)):
                    nrcol = f32(10.0) ** (f1pr07 + logn0r) * rho[k] \
                        * rhofaci[k] * ERI * nitot[i, k] * iscf \
                        * (spf - spf_clr)
                    # collected rain mass (:2743) feeds only the
                    # commented-out shedding source (:2747).

                # iceice_interaction1 (:2754-2829): iice >= 2 only -- never
                # reached at nCat = 1.

                # -- self-collection of ice (:2840-2842)
                if qitot[i, k] >= QSMALL:
                    nislf = f1pr03 * rho[k] * eii * eii_fact * rhofaci[k] \
                        * nitot[i, k] * nitot[i, k] * iscf

                # -- melting (:2851-2865)
                if qitot[i, k] >= QSMALL and t[i, k] > f32(273.15):
                    qsat0 = f32(0.622) * E0 / (pres[i, k] - E0)
                    dum = f32(0.0)
                    qimlt = ((f1pr05 + f1pr14 * sc ** THRD
                              * (rhofaci[k] * rho[k] / mu) ** f32(0.5))
                             * ((t[i, k] - f32(273.15)) * kap
                                - rho[k] * xxlv * dv * (qsat0 - qv_cld[k]))
                             * f32(2.0) * PI / xlf + dum) * nitot[i, k]
                    qimlt = max(qimlt, f32(0.0))
                    nimlt = qimlt * (nitot[i, k] / qitot[i, k])

                # -- wet growth (:2873-2894)
                if (qitot[i, k] >= QSMALL
                        and (qc[i, k] + qr[i, k]) >= f32(1.0e-6)
                        and t[i, k] < f32(273.15)):
                    qsat0 = f32(0.622) * E0 / (pres[i, k] - E0)
                    qwgrth = ((f1pr05 + f1pr14 * sc ** THRD
                               * (rhofaci[k] * rho[k] / mu) ** f32(0.5))
                              * f32(2.0) * PI
                              * (rho[k] * xxlv * dv * (qsat0 - qv_cld[k])
                                 - (t[i, k] - f32(273.15)) * kap)
                              / (xlf + CPW * (t[i, k] - f32(273.15)))) \
                        * nitot[i, k]
                    qwgrth = max(qwgrth, f32(0.0))
                    dum = max(f32(0.0), (qccol + qrcol) - qwgrth)
                    if dum >= f32(1.0e-10):
                        nrshdr = nrshdr + dum * f32(1.923e6)
                        if (qccol + qrcol) >= f32(1.0e-10):
                            dum1 = f32(1.0) / (qccol + qrcol)
                            qcshd = qcshd + dum * qccol * dum1
                            qccol = qccol - dum * qccol * dum1
                            qrcol = qrcol - dum * qrcol * dum1
                        log_wetgrowth = True

                # -- inverse supersaturation relaxation timescale (:2900-2906)
                if qitot[i, k] >= QSMALL and t[i, k] < f32(273.15):
                    epsi = ((f1pr05 + f1pr14 * sc ** THRD
                             * (rhofaci[k] * rho[k] / mu) ** f32(0.5))
                            * f32(2.0) * PI * rho[k] * dv) * nitot[i, k]
                    epsi_tot = epsi_tot + epsi
                else:
                    epsi = f32(0.0)

                # -- rime density, Cober and List 1993 (:2925-2969)
                if qccol >= QSMALL and t[i, k] < f32(273.15):
                    vtrmi1 = f1pr02 * rhofaci[k]
                    iTc = f32(1.0) / min(f32(-0.001), t[i, k] - f32(273.15))
                    if qc[i, k] >= QSMALL:
                        Vt_qc = acn[k] * _gammaf(f32(4.0) + BCN + mu_c) / (
                            lamc ** BCN * _gammaf(mu_c + f32(4.0)))
                        D_c = (mu_c + f32(4.0)) / lamc
                        V_impact = abs(vtrmi1 - Vt_qc)
                        Ri = -(f32(0.5e6) * D_c) * V_impact * iTc
                        Ri = max(f32(1.0), min(Ri, f32(12.0)))
                        if Ri <= f32(8.0):
                            rhorime_c = (f32(0.051) + f32(0.114) * Ri
                                         - f32(0.0055) * Ri ** f32(2.0)) \
                                * f32(1000.0)
                        else:
                            rhorime_c = f32(611.0) + f32(72.25) * (Ri - f32(8.0))
                else:
                    rhorime_c = f32(400.0)

                # -- immersion freezing of droplets (:2984-3015)
                if qc[i, k] >= QSMALL and t[i, k] <= f32(269.15):
                    dum = (f32(1.0) / lamc) ** f32(3.0)
                    # WRF's own chain first and unchanged, so a value its
                    # arithmetic can represent is still its value, bit for
                    # bit.  The rescue below is reached only when it cannot
                    # -- see _rescue_overflowed_product.
                    with np.errstate(over="ignore", invalid="ignore"):
                        tmp1 = cdist1 * np.exp(AIMM * (f32(273.15) - t[i, k]))
                        gam_q = _gammaf(f32(7.0) + mu_c)
                        gam_n = _gammaf(mu_c + f32(4.0))
                        Q_nuc = CONS6 * gam_q * tmp1 * dum ** f32(2.0)
                        N_nuc = CONS5 * gam_n * tmp1 * dum
                    if not np.isfinite(Q_nuc) or not np.isfinite(N_nuc):
                        # Argument in float32, exp in double: the authority's
                        # own commented-out form, dexp(dble(aimm*(273.15-t)))
                        # at :2999, and what the rain branch below does.
                        exp_aimm = np.exp(f64(AIMM * (f32(273.15) - t[i, k])))
                        if not np.isfinite(Q_nuc):
                            Q_nuc = _rescue_overflowed_product(
                                CONS6, gam_q, cdist1, exp_aimm, dum, dum)
                        if not np.isfinite(N_nuc):
                            N_nuc = _rescue_overflowed_product(
                                CONS5, gam_n, cdist1, exp_aimm, dum)
                    qcheti = Q_nuc
                    ncheti = N_nuc

                # -- immersion freezing of rain (:3021-3043)
                if qr[i, k] * ispf >= QSMALL and t[i, k] <= f32(269.15):
                    tmpdbl1 = np.exp(f64(np.log(cdistr)
                                         + np.log(_gammaf(f32(7.0) + mu_r))
                                         - f32(6.0) * np.log(lamr)))
                    tmpdbl2 = np.exp(f64(np.log(cdistr)
                                         + np.log(_gammaf(mu_r + f32(4.0)))
                                         - f32(3.0) * np.log(lamr)))
                    tmpdbl3 = np.exp(f64(AIMM * (f32(273.15) - t[i, k])))
                    # Same class, same rescue.  WRF's ``sngl(tmpdbl1*tmpdbl3)``
                    # (:3037-3038) rounds a double to float32 and can land on
                    # +Inf for the same cold-cloud reason, reaching the same
                    # limiter and the same NaN.  The double product is already
                    # in hand, so the rescue is one more multiply.
                    with np.errstate(over="ignore", invalid="ignore"):
                        qrheti = CONS6 * f32(tmpdbl1 * tmpdbl3) * spf
                        nrheti = CONS5 * f32(tmpdbl2 * tmpdbl3) * spf
                    if not np.isfinite(qrheti):
                        qrheti = _rescue_overflowed_product(
                            CONS6, tmpdbl1 * tmpdbl3, spf)
                    if not np.isfinite(nrheti):
                        nrheti = _rescue_overflowed_product(
                            CONS5, tmpdbl2 * tmpdbl3, spf)

                # rime splintering (:3049-3106): log_hmossopOn = (nCat>1)
                # (:2256) is FALSE for the ported 1-category configuration.

                # -- condensation/evaporation/deposition/sublimation
                if qr[i, k] * ispf >= QSMALL:   # :3114-3129
                    dumii_r, dumjj_r, dum1, rdumii, rdumjj, inv_dum3 = \
                        find_lookupTable_indices_3(mu_r, lamr)
                    dum1 = revap_table[dumii_r-1, dumjj_r-1] \
                        + (rdumii - f32(dumii_r)) * (
                            revap_table[dumii_r, dumjj_r-1]
                            - revap_table[dumii_r-1, dumjj_r-1])
                    dum2 = revap_table[dumii_r-1, dumjj_r] \
                        + (rdumii - f32(dumii_r)) * (
                            revap_table[dumii_r, dumjj_r]
                            - revap_table[dumii_r-1, dumjj_r])
                    dum = dum1 + (rdumjj - f32(dumjj_r)) * (dum2 - dum1)
                    epsr = f32(2.0) * PI * cdistr * rho[k] * dv * (
                        F1R * _gammaf(mu_r + f32(2.0)) / lamr
                        + F2R * (rho[k] / mu) ** f32(0.5) * sc ** THRD * dum)
                else:
                    epsr = f32(0.0)

                if qc[i, k] >= QSMALL:
                    epsc = f32(2.0) * PI * rho[k] * dv * cdist
                else:
                    epsc = f32(0.0)

                if t[i, k] < f32(273.15):
                    oabi = f32(1.0) / abi
                    xx = epsc + epsr + epsi_tot * (
                        f32(1.0) + xxls * INV_CP * dqsdT) * oabi
                else:
                    oabi = f32(1.0) / abi   # value irrelevant on this branch
                    xx = epsc + epsr

                dumqvi = qvi[k]    # :3144

                # 'A' term (:3171-3180)
                dum = -CP / G * (t[i, k] - t_old[i, k]) * odt
                if t[i, k] < f32(273.15):
                    aaa = (qv[i, k] - qv_old[i, k]) * odt \
                        - dqsdT * (-dum * G * INV_CP) \
                        - (qvs[k] - dumqvi) * (f32(1.0) + xxls * INV_CP
                                               * dqsdT) * oabi * epsi_tot
                else:
                    aaa = (qv[i, k] - qv_old[i, k]) * odt \
                        - dqsdT * (-dum * G * INV_CP)

                xx = max(f32(1.0e-20), xx)   # :3182
                oxx = f32(1.0) / xx

                # scpf off (:3185-3190)
                ssat_cld = ssat[i, k]
                ssat_r = ssat[i, k]
                sup_cld = sup[k]
                sup_r = sup[k]
                supi_cld = supi[k]

                if qc[i, k] >= QSMALL:   # :3201-3202
                    qccon = (aaa * epsc * oxx
                             + (ssat_cld * scf - aaa * oxx) * odt * epsc
                             * oxx * (f32(1.0) - f32(np.exp(-f64(xx * dt))))
                             ) / ab
                if qr[i, k] >= QSMALL:   # :3203-3204
                    qrcon = (aaa * epsr * oxx
                             + (ssat_r * spf - aaa * oxx) * odt * epsr
                             * oxx * (f32(1.0) - f32(np.exp(-f64(xx * dt))))
                             ) / ab

                if sup_cld < f32(-0.001) and qc[i, k] < f32(1.0e-12):
                    qccon = -qc[i, k] * odt
                if sup_r < f32(-0.001) and qr[i, k] < f32(1.0e-12):
                    qrcon = -qr[i, k] * odt

                if qccon < f32(0.0):
                    qcevp = -qccon
                    qccon = f32(0.0)
                else:
                    qccon = min(qccon, qv[i, k] * odt)

                if qrcon < f32(0.0):
                    qrevp = -qrcon
                    nrevp = qrevp * (nr[i, k] / qr[i, k])
                    qrcon = f32(0.0)
                else:
                    qrcon = min(qrcon, qv[i, k] * odt)

                # iice_loop_depsub (:3226-3255)
                if qitot[i, k] >= QSMALL and t[i, k] < f32(273.15):
                    qidep = (aaa * epsi * oxx
                             + (ssat_cld * scf - aaa * oxx) * odt * epsi
                             * oxx * (f32(1.0) - f32(np.exp(-f64(xx * dt))))
                             ) * oabi + (qvs[k] - dumqvi) * epsi * oabi
                if supi_cld < f32(-0.001) and qitot[i, k] < f32(1.0e-12):
                    qidep = -qitot[i, k] * odt
                if qidep < f32(0.0):
                    qisub = -qidep
                    qisub = qisub * f32(clbfact_sub)
                    qisub = min(qisub, qitot[i, k] * odt)
                    nisub = qisub * (nitot[i, k] / qitot[i, k])
                    qidep = f32(0.0)
                else:
                    qidep = qidep * f32(clbfact_dep)
                    qidep = min(qidep, qv[i, k] * odt)

            # 444 continue (:3257) -------------------------------------------

            # -- deposition/condensation-freezing nucleation (:3264-3293)
            sup_cld = sup[k]
            supi_cld = supi[k]
            if t[i, k] < f32(258.15) and supi_cld >= f32(0.05):
                dum = f32(0.005) * np.exp(f32(0.304) * (f32(273.15) - t[i, k])) \
                    * f32(1000.0) * inv_rho[k]                # Cooper (1986)
                dum = min(dum, f32(100.0e3) * inv_rho[k] * scf)
                N_nuc = max(f32(0.0), (dum - nitot[i, k]) * odt)
                if N_nuc >= f32(1.0e-20):
                    Q_nuc = max(f32(0.0), (dum - nitot[i, k]) * MI0 * odt)
                    qinuc = Q_nuc
                    ninuc = N_nuc

            # -- droplet activation, specified-Nc branch (:3303-3311)
            if sup_cld > f32(1.0e-6) and it > 1:
                dum = NCCNST * inv_rho[k] * CONS7 - qc[i, k]
                dum = max(f32(0.0), dum * iscf)
                dumqvs = qv_sat(t[i, k], pres[i, k], 0)
                dqsdT_l = xxlv * dumqvs / (RV * t[i, k] * t[i, k])
                ab_l = f32(1.0) + dqsdT_l * xxlv * INV_CP
                dum = max(f32(0.0), min(dum, (qv_cld[k] - dumqvs) / ab_l))
                qcnuc = dum * odt * scf
            # (:3313-3339 is the log_predictNc activation path -- not ported)

            # -- first-step saturation adjustment (:3348-3356)
            if it <= 1:
                dumt = th[i, k] * (pres[i, k] * f32(1.0e-5)) ** (RD * INV_CP)
                dumqv = qv_cld[k]
                dumqvs = qv_sat(dumt, pres[i, k], 0)
                dums = dumqv - dumqvs
                qccon = dums / (f32(1.0) + xxlv ** f32(2.0) * dumqvs
                                / (CP * RV * dumt ** f32(2.0))) * odt * scf
                qccon = max(f32(0.0), qccon)
                if qccon <= f32(1.0e-7):
                    qccon = f32(0.0)

            # -- autoconversion, iparam = 3 (KK2000) (:3362-3416)
            if qc[i, k] * iscf >= f32(1.0e-8):
                dum = qc[i, k] * iscf
                qcaut = f32(1350.0) * dum ** f32(2.47) \
                    * (nc[i, k] * iscf * f32(1.0e-6) * rho[k]) ** f32(-1.79) \
                    * scf
                ncautr = qcaut * CONS3
                ncautc = qcaut * nc[i, k] / qc[i, k]
                if qcaut == f32(0.0):
                    ncautc = f32(0.0)
                if ncautc == f32(0.0):
                    qcaut = f32(0.0)

            # -- self-collection of droplets, iparam = 3 (:3421-3435)
            if qc[i, k] >= QSMALL:
                ncslf = f32(0.0)

            # -- accretion, iparam = 3 (:3440-3472)
            if qr[i, k] >= QSMALL and qc[i, k] >= QSMALL:
                dum2 = spf - spf_clr
                qcacc = f32(67.0) * (qc[i, k] * iscf * qr[i, k] * ispf) \
                    ** f32(1.15) * dum2
                ncacc = qcacc * nc[i, k] / qc[i, k]
                if qcacc == f32(0.0):
                    ncacc = f32(0.0)
                if ncacc == f32(0.0):
                    qcacc = f32(0.0)

            # -- rain self-collection/breakup, iparam = 3 (:3478-3504)
            if qr[i, k] >= QSMALL:
                dum1 = f32(280.0e-6)
                nr[i, k] = max(nr[i, k], NSMALL)
                dum2 = (qr[i, k] / (PI * RHOW * nr[i, k])) ** THRD
                if dum2 < dum1:
                    dum = f32(1.0)
                else:
                    dum = f32(2.0) - np.exp(f32(2300.0) * (dum2 - dum1))
                nrslf = dum * f32(5.78) * nr[i, k] * ispf * qr[i, k] * ispf \
                    * rho[k] * spf

            # -- conservation of mass (:3515-3644)
            dumqvs = qv_sat(t[i, k], pres[i, k], 0)
            qcon_satadj = (qv_cld[k] - dumqvs) / (
                f32(1.0) + xxlv ** f32(2.0) * dumqvs
                / (CP * RV * t[i, k] ** f32(2.0))) * odt * scf
            tmp1 = qccon + qrcon + qcnuc
            if tmp1 > f32(0.0) and qcon_satadj < f32(0.0):
                qccon = f32(0.0)
                qrcon = f32(0.0)
                qcnuc = f32(0.0)
                ncnuc = f32(0.0)
            else:
                if tmp1 > f32(0.0) and tmp1 > qcon_satadj:
                    ratio = max(f32(0.0), qcon_satadj) / tmp1
                    ratio = min(f32(1.0), ratio)
                    qccon = qccon * ratio
                    qrcon = qrcon * ratio
                    qcnuc = qcnuc * ratio
                    ncnuc = ncnuc * ratio
                elif qcevp + qrevp > f32(0.0):
                    ratio = max(f32(0.0), -qcon_satadj) / (qcevp + qrevp)
                    ratio = min(f32(1.0), ratio)
                    qcevp = qcevp * ratio
                    qrevp = qrevp * ratio
                    nrevp = nrevp * ratio

            qv_tmp = qv_cld[k] + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt
            t_tmp = t[i, k] + (qcnuc + qccon + qrcon - qcevp - qrevp) \
                * xxlv * INV_CP * dt
            dumqvi = qv_sat(t_tmp, pres[i, k], 1)
            qdep_satadj = (qv_tmp - dumqvi) / (
                f32(1.0) + xxls ** f32(2.0) * dumqvi
                / (CP * RV * t_tmp ** f32(2.0))) * odt * scf
            tmp1 = qidep + qinuc
            if tmp1 > f32(0.0) and qdep_satadj < f32(0.0):
                qidep = f32(0.0)
                qinuc = f32(0.0)
                ninuc = f32(0.0)
            else:
                if tmp1 > f32(0.0) and tmp1 > qdep_satadj:
                    ratio = max(f32(0.0), qdep_satadj) / tmp1
                    ratio = min(f32(1.0), ratio)
                    qidep = qidep * ratio
                    qinuc = qinuc * ratio
                    ninuc = ninuc * ratio
                dum = max(qisub, f32(1.0e-20))
                qisub = qisub * min(f32(1.0), max(f32(0.0), -qdep_satadj)
                                    / max(qisub, f32(1.0e-20)))
                nisub = nisub * min(f32(1.0), qisub / dum)

            # cloud (:3571-3590)
            sinks = (qcaut + qcacc + qccol + qcevp + qchetc + qcheti
                     + qcshd) * dt
            sources = qc[i, k] + (qccon + qcnuc) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qcaut = qcaut * ratio
                qcacc = qcacc * ratio
                qcevp = qcevp * ratio
                qccol = qccol * ratio
                qcheti = qcheti * ratio
                qcshd = qcshd * ratio
                ncautc = ncautc * ratio
                ncacc = ncacc * ratio
                nccol = nccol * ratio
                ncheti = ncheti * ratio

            # rain (:3593-3606)
            sinks = (qrevp + qrcol + qrhetc + qrheti + qrmul) * dt
            sources = qr[i, k] + (qrcon + qcaut + qcacc + qimlt + qcshd) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qrevp = qrevp * ratio
                qrcol = qrcol * ratio
                qrheti = qrheti * ratio
                qrmul = qrmul * ratio
                nrevp = nrevp * ratio
                nrcol = nrcol * ratio
                nrheti = nrheti * ratio

            # ice (:3609-3630)
            sinks = (qisub + qimlt) * dt
            sources = qitot[i, k] + (qidep + qinuc + qrcol + qccol + qrhetc
                                     + qrheti + qchetc + qcheti + qrmul) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qisub = qisub * ratio
                qimlt = qimlt * ratio
                nisub = nisub * ratio
                nimlt = nimlt * ratio

            # vapor (:3633-3644)
            sinks = (qccon + qrcon + qcnuc + qidep + qinuc) * dt
            sources = qv[i, k] + (qcevp + qrevp + qisub) * dt
            if sinks > sources and sinks >= f32(1.0e-20):
                ratio = sources / sinks
                qccon = qccon * ratio
                qrcon = qrcon * ratio
                qcnuc = qcnuc * ratio
                qidep = qidep * ratio
                qinuc = qinuc * ratio
                ninuc = ninuc * ratio
                ncnuc = ncnuc * ratio

            # update_refl_processes (:3652-3743): 3-moment only, not ported.

            # -- update prognostic variables (:3756-3921)
            rimevolume = f32(0.0)
            rimefraction = f32(0.0)
            if qitot[i, k] >= QSMALL:   # iice_loop2 (:3756-3765)
                tmp1 = f32(1.0) / qitot[i, k]
                rimevolume = birim[i, k] * tmp1
                rimefraction = qirim[i, k] * tmp1

            # iice_loop3 (:3767-3865)
            qc[i, k] = qc[i, k] + (-qchetc - qcheti - qccol - qcshd) * dt
            qr[i, k] = qr[i, k] + (-qrcol + qimlt - qrhetc - qrheti + qcshd
                                   - qrmul) * dt
            nr[i, k] = nr[i, k] + (-nrcol - nrhetc - nrheti
                                   + NMLTRATIO * nimlt + nrshdr + ncshdc) * dt

            birim[i, k] = birim[i, k] - (qisub + qimlt) * dt * rimevolume
            qirim[i, k] = qirim[i, k] - (qisub + qimlt) * dt * rimefraction
            qitot[i, k] = qitot[i, k] - (qisub + qimlt) * dt

            dum = (qrcol + qccol + qrhetc + qrheti + qchetc + qcheti
                   + qrmul) * dt
            qitot[i, k] = qitot[i, k] + (qidep + qinuc) * dt + dum
            qirim[i, k] = qirim[i, k] + dum
            birim[i, k] = birim[i, k] + (qrcol * INV_RHO_RIMEMAX
                                         + qccol / rhorime_c
                                         + (qrhetc + qrheti + qchetc + qcheti
                                            + qrmul) * INV_RHO_RIMEMAX) * dt
            nitot[i, k] = nitot[i, k] + (ninuc - nimlt - nisub - nislf
                                         + nrhetc + nrheti + nchetc + ncheti
                                         + nimul) * dt

            if qirim[i, k] < f32(0.0):
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)

            # wet-growth densification (:3835-3838)
            if log_wetgrowth:
                qirim[i, k] = qitot[i, k]
                birim[i, k] = qirim[i, k] * INV_RHO_RIMEMAX

            # melting densification toward solid ice (:3841-3845)
            if (qitot[i, k] >= QSMALL and birim[i, k] >= BSMALL
                    and qimlt > f32(0.0)):
                tmp1 = qirim[i, k] / birim[i, k]
                tmp2 = qitot[i, k] + qimlt * dt
                birim[i, k] = qirim[i, k] / (tmp1 + (f32(917.0) - tmp1)
                                             * qimlt * dt / tmp2)

            qv[i, k] = qv[i, k] + (-qidep + qisub - qinuc) * dt
            th[i, k] = th[i, k] + invexn[i, k] * (
                (qidep - qisub + qinuc) * xxls * INV_CP
                + (qrcol + qccol + qchetc + qcheti + qrhetc + qrheti
                   + qrmul - qimlt) * xlf * INV_CP) * dt

            # warm-phase updates (:3869-3885)
            qc[i, k] = qc[i, k] + (-qcacc - qcaut + qcnuc + qccon - qcevp) * dt
            qr[i, k] = qr[i, k] + (qcacc + qcaut + qrcon - qrevp) * dt
            nc[i, k] = NCCNST * inv_rho[k]    # not predictNc (:3875)
            nr[i, k] = nr[i, k] + (ncautr - nrslf - nrevp) * dt  # iparam=3
            qv[i, k] = qv[i, k] + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt
            th[i, k] = th[i, k] + invexn[i, k] * (
                (qcnuc + qccon + qrcon - qcevp - qrevp) * xxlv * INV_CP) * dt

            # clipping (:3889-3918)
            if qc[i, k] < QSMALL:
                qv[i, k] = qv[i, k] + qc[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qc[i, k] * xxlv * INV_CP
                qc[i, k] = f32(0.0)
                nc[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qr[i, k] < QSMALL:
                qv[i, k] = qv[i, k] + qr[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qr[i, k] * xxlv * INV_CP
                qr[i, k] = f32(0.0)
                nr[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            if qitot[i, k] < QSMALL:
                qv[i, k] = qv[i, k] + qitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xxls * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)
            else:
                log_hydrometeorsPresent = True

            qv[i, k] = max(f32(0.0), qv[i, k])
            nitot[i, k] = impose_max_total_Ni(nitot[i, k], inv_rho[k])
            # 555 continue

        # second compute_SCPF (:3968-3970): refresh the qv snapshot.
        qv_cld = qv[i, :].copy()

        if not log_hydrometeorsPresent:
            continue    # goto 333 (:3972)

        # ==================================================================
        # Sedimentation (:3984-4495), adaptive substepping.
        # ==================================================================

        # ---- cloud, 1-moment (log_predictNc = .false., :4072-4119) ----
        log_qxpresent = False
        k_qxtop = kbot
        for k in range(ktop, kbot - 1, -kdir):
            if qc[i, k] * iscf >= QSMALL:
                log_qxpresent = True
                k_qxtop = k
                break
        if log_qxpresent:
            dt_left = dt
            prt_accum = f32(0.0)
            k_qxbot = kbot
            for k in range(kbot, k_qxtop + 1, kdir):
                if qc[i, k] * iscf >= QSMALL:
                    k_qxbot = k
                    break
            V_qc = np.zeros(nk, dtype=np.float32)
            flux_qx = np.zeros(nk, dtype=np.float32)
            while dt_left > f32(1.0e-4):
                Co_max = f32(0.0)
                V_qc[:] = f32(0.0)
                for k in range(k_qxtop, k_qxbot - 1, -kdir):
                    if qc[i, k] * iscf > QSMALL:
                        nc[i, k], mu_c, nu, lamc, _t1, _t2 = get_cloud_dsd2(
                            qc[i, k], nc[i, k], rho[k], iscf)
                        dum = f32(1.0) / lamc ** BCN
                        V_qc[k] = acn[k] * _gammaf(f32(4.0) + BCN + mu_c) \
                            * dum / _gammaf(mu_c + f32(4.0))
                    Co_max = max(Co_max, V_qc[k] * dt_left * inv_dzq[i, k])
                tmpint1 = int(Co_max + f32(1.0))
                dt_sub = min(dt_left, dt_left / f32(tmpint1))
                if k_qxbot == kbot:
                    k_temp = k_qxbot
                else:
                    k_temp = k_qxbot - kdir
                for k in range(k_temp, k_qxtop + 1, kdir):
                    flux_qx[k] = V_qc[k] * qc[i, k] * rho[k]
                if k_qxbot == kbot:
                    prt_accum = prt_accum + flux_qx[kbot] * dt_sub
                k = k_qxtop
                fluxdiv_qx = -flux_qx[k] * inv_dzq[i, k]
                qc[i, k] = qc[i, k] + fluxdiv_qx * dt_sub * inv_rho[k]
                for k in range(k_qxtop - kdir, k_temp - 1, -kdir):
                    fluxdiv_qx = (flux_qx[k + kdir] - flux_qx[k]) \
                        * inv_dzq[i, k]
                    qc[i, k] = qc[i, k] + fluxdiv_qx * dt_sub * inv_rho[k]
                dt_left = dt_left - dt_sub
                if k_qxbot != kbot:
                    k_qxbot = k_qxbot - kdir
            prt_liq[i] = prt_accum * INV_RHOW * odt

        # ---- rain (:4131-4245) ----
        log_qxpresent = False
        k_qxtop = kbot
        for k in range(ktop, kbot - 1, -kdir):
            if qr[i, k] * ispf >= QSMALL:
                log_qxpresent = True
                k_qxtop = k
                break
        if log_qxpresent:
            dt_left = dt
            prt_accum = f32(0.0)
            k_qxbot = kbot
            for k in range(kbot, k_qxtop + 1, kdir):
                if qr[i, k] * ispf >= QSMALL:
                    k_qxbot = k
                    break
            V_qr = np.zeros(nk, dtype=np.float32)
            V_nr = np.zeros(nk, dtype=np.float32)
            flux_qx = np.zeros(nk, dtype=np.float32)
            flux_nx = np.zeros(nk, dtype=np.float32)
            while dt_left > f32(1.0e-4):
                Co_max = f32(0.0)
                V_qr[:] = f32(0.0)
                V_nr[:] = f32(0.0)
                for k in range(k_qxtop, k_qxbot - 1, -kdir):
                    if qr[i, k] * ispf > QSMALL:
                        nr[i, k] = max(nr[i, k], NSMALL)
                        nr[i, k], mu_r, lamr, cdistr, logn0r = get_rain_dsd2(
                            qr[i, k], nr[i, k], ispf)
                        dumii_r, dumjj_r, dum1, rdumii, rdumjj, inv_dum3 = \
                            find_lookupTable_indices_3(mu_r, lamr)
                        dum1 = vm_table[dumii_r-1, dumjj_r-1] \
                            + (rdumii - f32(dumii_r)) * (
                                vm_table[dumii_r, dumjj_r-1]
                                - vm_table[dumii_r-1, dumjj_r-1])
                        dum2 = vm_table[dumii_r-1, dumjj_r] \
                            + (rdumii - f32(dumii_r)) * (
                                vm_table[dumii_r, dumjj_r]
                                - vm_table[dumii_r-1, dumjj_r])
                        V_qr[k] = dum1 + (rdumjj - f32(dumjj_r)) * (dum2 - dum1)
                        V_qr[k] = V_qr[k] * rhofacr[k]
                        dum1 = vn_table[dumii_r-1, dumjj_r-1] \
                            + (rdumii - f32(dumii_r)) * (
                                vn_table[dumii_r, dumjj_r-1]
                                - vn_table[dumii_r-1, dumjj_r-1])
                        dum2 = vn_table[dumii_r-1, dumjj_r] \
                            + (rdumii - f32(dumii_r)) * (
                                vn_table[dumii_r, dumjj_r]
                                - vn_table[dumii_r-1, dumjj_r])
                        V_nr[k] = dum1 + (rdumjj - f32(dumjj_r)) * (dum2 - dum1)
                        V_nr[k] = V_nr[k] * rhofacr[k]
                    Co_max = max(Co_max, V_qr[k] * dt_left * inv_dzq[i, k])
                tmpint1 = int(Co_max + f32(1.0))
                dt_sub = min(dt_left, dt_left / f32(tmpint1))
                if k_qxbot == kbot:
                    k_temp = k_qxbot
                else:
                    k_temp = k_qxbot - kdir
                for k in range(k_temp, k_qxtop + 1, kdir):
                    flux_qx[k] = V_qr[k] * qr[i, k] * rho[k]
                    flux_nx[k] = V_nr[k] * nr[i, k] * rho[k]
                if k_qxbot == kbot:
                    prt_accum = prt_accum + flux_qx[kbot] * dt_sub
                k = k_qxtop
                fluxdiv_qx = -flux_qx[k] * inv_dzq[i, k]
                fluxdiv_nx = -flux_nx[k] * inv_dzq[i, k]
                qr[i, k] = qr[i, k] + fluxdiv_qx * dt_sub * inv_rho[k]
                nr[i, k] = nr[i, k] + fluxdiv_nx * dt_sub * inv_rho[k]
                for k in range(k_qxtop - kdir, k_temp - 1, -kdir):
                    fluxdiv_qx = (flux_qx[k + kdir] - flux_qx[k]) \
                        * inv_dzq[i, k]
                    fluxdiv_nx = (flux_nx[k + kdir] - flux_nx[k]) \
                        * inv_dzq[i, k]
                    qr[i, k] = qr[i, k] + fluxdiv_qx * dt_sub * inv_rho[k]
                    nr[i, k] = nr[i, k] + fluxdiv_nx * dt_sub * inv_rho[k]
                dt_left = dt_left - dt_sub
                if k_qxbot != kbot:
                    k_qxbot = k_qxbot - kdir
            prt_liq[i] = prt_liq[i] + prt_accum * INV_RHOW * odt

        # ---- ice, 2-moment (:4251-4495) ----
        log_qxpresent = False
        k_qxtop = kbot
        for k in range(ktop, kbot - 1, -kdir):
            if qitot[i, k] >= QSMALL:
                log_qxpresent = True
                k_qxtop = k
                break
        if log_qxpresent:
            dt_left = dt
            prt_accum = f32(0.0)
            k_qxbot = kbot
            for k in range(kbot, k_qxtop + 1, kdir):
                if qitot[i, k] >= QSMALL:
                    k_qxbot = k
                    break
            V_qit = np.zeros(nk, dtype=np.float32)
            V_nit = np.zeros(nk, dtype=np.float32)
            flux_qit = np.zeros(nk, dtype=np.float32)
            flux_nit = np.zeros(nk, dtype=np.float32)
            flux_qir = np.zeros(nk, dtype=np.float32)
            flux_bir = np.zeros(nk, dtype=np.float32)
            while dt_left > f32(1.0e-4):
                Co_max = f32(0.0)
                V_qit[:] = f32(0.0)
                V_nit[:] = f32(0.0)
                for k in range(k_qxtop, k_qxbot - 1, -kdir):
                    if qitot[i, k] >= QSMALL:
                        nitot[i, k] = max(nitot[i, k], NSMALL)
                        qirim[i, k], birim[i, k], rhop = calc_bulkRhoRime(
                            qitot[i, k], qirim[i, k], birim[i, k])
                        dumi, dumjj, dumii, dum1, dum4, dum5 = \
                            find_lookupTable_indices_1a(
                                qitot[i, k], nitot[i, k], qirim[i, k], rhop)
                        f1pr01 = access_lookup_table(
                            itab, dumjj, dumii, dumi, 1, dum1, dum4, dum5)
                        f1pr02 = access_lookup_table(
                            itab, dumjj, dumii, dumi, 2, dum1, dum4, dum5)
                        f1pr09 = access_lookup_table(
                            itab, dumjj, dumii, dumi, 7, dum1, dum4, dum5)
                        f1pr10 = access_lookup_table(
                            itab, dumjj, dumii, dumi, 8, dum1, dum4, dum5)
                        nitot[i, k] = min(nitot[i, k], f1pr09 * qitot[i, k])
                        nitot[i, k] = max(nitot[i, k], f1pr10 * qitot[i, k])
                        V_qit[k] = f1pr02 * rhofaci[k]
                        V_nit[k] = f1pr01 * rhofaci[k]
                    Co_max = max(Co_max, V_qit[k] * dt_left * inv_dzq[i, k])
                tmpint1 = int(Co_max + f32(1.0))
                dt_sub = min(dt_left, dt_left / f32(tmpint1))
                if k_qxbot == kbot:
                    k_temp = k_qxbot
                else:
                    k_temp = k_qxbot - kdir
                for k in range(k_temp, k_qxtop + 1, kdir):
                    flux_qit[k] = V_qit[k] * qitot[i, k] * rho[k]
                    flux_nit[k] = V_nit[k] * nitot[i, k] * rho[k]
                    flux_qir[k] = V_qit[k] * qirim[i, k] * rho[k]
                    flux_bir[k] = V_qit[k] * birim[i, k] * rho[k]
                if k_qxbot == kbot:
                    prt_accum = prt_accum + flux_qit[kbot] * dt_sub
                k = k_qxtop
                fluxdiv_qit = -flux_qit[k] * inv_dzq[i, k]
                fluxdiv_qir = -flux_qir[k] * inv_dzq[i, k]
                fluxdiv_bir = -flux_bir[k] * inv_dzq[i, k]
                fluxdiv_nit = -flux_nit[k] * inv_dzq[i, k]
                qitot[i, k] = qitot[i, k] + fluxdiv_qit * dt_sub * inv_rho[k]
                qirim[i, k] = qirim[i, k] + fluxdiv_qir * dt_sub * inv_rho[k]
                birim[i, k] = birim[i, k] + fluxdiv_bir * dt_sub * inv_rho[k]
                nitot[i, k] = nitot[i, k] + fluxdiv_nit * dt_sub * inv_rho[k]
                for k in range(k_qxtop - kdir, k_temp - 1, -kdir):
                    fluxdiv_qit = (flux_qit[k + kdir] - flux_qit[k]) \
                        * inv_dzq[i, k]
                    fluxdiv_qir = (flux_qir[k + kdir] - flux_qir[k]) \
                        * inv_dzq[i, k]
                    fluxdiv_bir = (flux_bir[k + kdir] - flux_bir[k]) \
                        * inv_dzq[i, k]
                    fluxdiv_nit = (flux_nit[k + kdir] - flux_nit[k]) \
                        * inv_dzq[i, k]
                    qitot[i, k] = qitot[i, k] + fluxdiv_qit * dt_sub \
                        * inv_rho[k]
                    qirim[i, k] = qirim[i, k] + fluxdiv_qir * dt_sub \
                        * inv_rho[k]
                    birim[i, k] = birim[i, k] + fluxdiv_bir * dt_sub \
                        * inv_rho[k]
                    nitot[i, k] = nitot[i, k] + fluxdiv_nit * dt_sub \
                        * inv_rho[k]
                dt_left = dt_left - dt_sub
                if k_qxbot != kbot:
                    k_qxbot = k_qxbot - kdir
            prt_sol[i] = prt_sol[i] + prt_accum * INV_RHOW * odt

        # third compute_SCPF (:4521-4523), quick: off-branch again.
        qv_cld = qv[i, :].copy()

        # ---- homogeneous freezing of cloud and rain (:4528-4610) ----
        for k in range(nk):
            if qc[i, k] >= QSMALL and t[i, k] < f32(233.15):
                Q_nuc = qc[i, k]
                nc[i, k] = max(nc[i, k], NSMALL)
                N_nuc = nc[i, k]
                qirim[i, k] = qirim[i, k] + Q_nuc
                qitot[i, k] = qitot[i, k] + Q_nuc
                birim[i, k] = birim[i, k] + Q_nuc * INV_RHO_RIMEMAX
                nitot[i, k] = nitot[i, k] + N_nuc
                th[i, k] = th[i, k] + invexn[i, k] * Q_nuc * xlf * INV_CP
                qc[i, k] = f32(0.0)
                nc[i, k] = f32(0.0)
            if qr[i, k] >= QSMALL and t[i, k] < f32(233.15):
                Q_nuc = qr[i, k]
                nr[i, k] = max(nr[i, k], NSMALL)
                N_nuc = nr[i, k]
                qirim[i, k] = qirim[i, k] + Q_nuc
                qitot[i, k] = qitot[i, k] + Q_nuc
                birim[i, k] = birim[i, k] + Q_nuc * INV_RHO_RIMEMAX
                nitot[i, k] = nitot[i, k] + N_nuc
                th[i, k] = th[i, k] + invexn[i, k] * Q_nuc * xlf * INV_CP
                qr[i, k] = f32(0.0)
                nr[i, k] = f32(0.0)

        # category merge (:4615-4697): nCat > 1 only, not ported.

        # ---- final checks + diagnostics (:4722-4895) ----
        for k in range(nk):
            if qc[i, k] * iscf >= QSMALL:
                nc[i, k], mu_c, nu, lamc, _t1, _t2 = get_cloud_dsd2(
                    qc[i, k], nc[i, k], rho[k], iscf)
                diag_effc[i, k] = f32(0.5) * (mu_c + f32(3.0)) / lamc
            else:
                qv[i, k] = qv[i, k] + qc[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qc[i, k] * xxlv * INV_CP
                qc[i, k] = f32(0.0)
                nc[i, k] = f32(0.0)

            if qr[i, k] >= QSMALL:
                nr[i, k], mu_r, lamr, _t1, _t2 = get_rain_dsd2(
                    qr[i, k], nr[i, k], f32(1.0))   # iSPF literal 1. (:4739)
                ze_rain[i, k] = rho[k] * nr[i, k] * (mu_r + f32(6.0)) \
                    * (mu_r + f32(5.0)) * (mu_r + f32(4.0)) \
                    * (mu_r + f32(3.0)) * (mu_r + f32(2.0)) \
                    * (mu_r + f32(1.0)) / lamr ** f32(6.0)
                ze_rain[i, k] = max(ze_rain[i, k], f32(1.0e-22))
            else:
                qv[i, k] = qv[i, k] + qr[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qr[i, k] * xxlv * INV_CP
                qr[i, k] = f32(0.0)
                nr[i, k] = f32(0.0)

            nitot[i, k] = impose_max_total_Ni(nitot[i, k], inv_rho[k])

            if qitot[i, k] >= QSMALL:
                nitot[i, k] = max(nitot[i, k], NSMALL)
                nr[i, k] = max(nr[i, k], NSMALL)
                qirim[i, k], birim[i, k], rhop = calc_bulkRhoRime(
                    qitot[i, k], qirim[i, k], birim[i, k])
                dumi, dumjj, dumii, dum1, dum4, dum5 = \
                    find_lookupTable_indices_1a(
                        qitot[i, k], nitot[i, k], qirim[i, k], rhop)
                f1pr02 = access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                             dum1, dum4, dum5)
                f1pr06 = access_lookup_table(itab, dumjj, dumii, dumi, 6,
                                             dum1, dum4, dum5)
                f1pr09 = access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                             dum1, dum4, dum5)
                f1pr10 = access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                             dum1, dum4, dum5)
                f1pr13 = access_lookup_table(itab, dumjj, dumii, dumi, 9,
                                             dum1, dum4, dum5)
                f1pr15 = access_lookup_table(itab, dumjj, dumii, dumi, 11,
                                             dum1, dum4, dum5)
                f1pr16 = access_lookup_table(itab, dumjj, dumii, dumi, 12,
                                             dum1, dum4, dum5)
                # f1pr22/f1pr23 (lambda_i/mu_i, table cols 13/14) feed only
                # the absent diag_lami/diag_mui/diag_dhmax outputs (:4860-4865)
                nitot[i, k] = min(nitot[i, k], f1pr09 * qitot[i, k])
                nitot[i, k] = max(nitot[i, k], f1pr10 * qitot[i, k])
                if qirim[i, k] < QSMALL:
                    qirim[i, k] = f32(0.0)
                    birim[i, k] = f32(0.0)
                diag_vmi[i, k] = f1pr02 * rhofaci[k]
                diag_effi[i, k] = f1pr06
                diag_di[i, k] = f1pr15
                diag_rhoi[i, k] = f1pr16
                ze_ice[i, k] = ze_ice[i, k] + f32(0.1892) * f1pr13 \
                    * nitot[i, k] * rho[k]
                ze_ice[i, k] = max(ze_ice[i, k], f32(1.0e-22))
            else:
                qv[i, k] = qv[i, k] + qitot[i, k]
                th[i, k] = th[i, k] - invexn[i, k] * qitot[i, k] * xxls * INV_CP
                qitot[i, k] = f32(0.0)
                nitot[i, k] = f32(0.0)
                qirim[i, k] = f32(0.0)
                birim[i, k] = f32(0.0)
                diag_di[i, k] = f32(0.0)

            diag_ze[i, k] = f32(10.0) * np.log10(
                (ze_rain[i, k] + ze_ice[i, k]) * f32(1.0e18))

            if qr[i, k] < QSMALL:
                nr[i, k] = f32(0.0)

        # 333 continue: SCF_out (:4941-4952) discarded by the WRF wrapper.

    # save end-of-microphysics theta/qv for the next step (:5018-5021)
    th_old[...] = th
    qv_old[...] = qv


# ---------------------------------------------------------------------------
# WRF wrapper (:690-932) on a flattened column slab.
# ---------------------------------------------------------------------------

def mp_p3_wrapper_wrf(th, qv, qc, qr, nr, qi, qir, ni, qib, th_old, qv_old,
                      pres, dz, dt, itimestep, rainnc, rainncv, sr, snownc,
                      snowncv, refl_10cm, re_cloud, re_ice, vmi3d, di3d,
                      rhopo3d, *, runtime: P3Runtime | None = None) -> None:
    """``mp_p3_wrapper_wrf`` for the mp=50 call shape
    (module_microphysics_driver.F:1569-1602: no ``nc_3d``, no ``qzi1_3d``,
    ``n_iceCat = 1``).

    All arrays are float32; 3-D fields are (ni, nk) column slabs with k=0
    at the surface (the WRF j-loop over independent (i, k) slabs and a
    single flattened slab are the same computation -- columns never
    interact inside p3_main).  Precipitation fields are (ni,) and are
    updated with WRF's mm conversions (:892-898).  ``pii`` (Exner) is a
    wrapper argument WRF marks "currently not used" (:725) and is omitted;
    ``w`` likewise feeds only the unused ``uzpl`` (:2204).
    """
    ni_pts, nk = th.shape
    nc = np.zeros((ni_pts, nk), dtype=np.float32)     # :847 (specified Nc)
    ssat = np.zeros((ni_pts, nk), dtype=np.float32)   # :851
    prt_liq = np.zeros(ni_pts, dtype=np.float32)
    prt_sol = np.zeros(ni_pts, dtype=np.float32)

    p3_main(qc, nc, qr, nr, th_old, th, qv_old, qv, dt, qi, qir, ni, qib,
            ssat, pres, dz, itimestep, prt_liq, prt_sol,
            refl_10cm, re_cloud, re_ice, vmi3d, di3d, rhopo3d,
            n_cat=1, log_predictNc=False, model="WRF",
            clbfact_dep=1.0, clbfact_sub=1.0, runtime=runtime)

    dum1 = f32(1000.0) * f32(dt)
    total = prt_liq + prt_sol
    rainnc += total * dum1
    rainncv[:] = total * dum1
    snownc += prt_sol * dum1
    snowncv[:] = prt_sol * dum1
    sr[:] = prt_sol / (prt_liq + prt_sol + f32(1.0e-12))


# ---------------------------------------------------------------------------
# DomainState adapter (gpuwm dispatch surface).
# ---------------------------------------------------------------------------

def _host(arr) -> np.ndarray:
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _to_slab(field3d) -> np.ndarray:
    """(nz, ny, nx) -> contiguous (ny*nx, nz) float32 host slab."""
    host = _host(field3d)
    nz = host.shape[0]
    return np.ascontiguousarray(host.reshape(nz, -1).T)


def _as_backend(dev, host):
    if hasattr(dev, "get"):
        import cupy as cp
        return cp.asarray(host)
    return host


def _from_slab(dev, slab) -> None:
    """Write an (ncol, nz) slab back into the (nz, ny, nx) device field."""
    host = np.ascontiguousarray(slab.T).reshape(dev.shape)
    dev[...] = _as_backend(dev, host)


def apply_reference(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """THE REFERENCE/DEBUG PATH: prepare WRF's inputs and run the CPU
    float32 transcription on ``state`` in place.

    This is not the production path -- :func:`apply` dispatches to the
    device kernels (``gpuwm/core/p3_device.py``).  It is kept, and kept
    reachable through ``run.p3_backend = "reference"``, because it is
    what every later device optimisation is checkable against and it is
    what runs where there is no CuPy at all.

    The CPU float32 authority runs on host copies of the device state and
    the results are written back; see the module docstring for the
    execution-model caveat.  The WRF prep contract matches the other
    adapters (full theta, layer depths from the full geopotential) and the
    theta update goes through the ``moist_physics_prep/finish`` bracket.
    """
    from gpuwm.core import constants as c
    from gpuwm.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    from gpuwm.core.state import DTYPE

    required = ("qi", "nr", "ni", "qir", "qib", "th_old", "qv_old")
    missing = [name for name in required
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=50 state lacks P3 fields: "
                         + ", ".join(missing))
    runtime = p3_init()   # loud table refusal happens here, before any prep

    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]

    th_dev = state.scratch((nz, ny, nx), "mp_th")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    dz_dev = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th_dev[...] = thb + state.thp
    pii[...] = (state.p / DTYPE(c.P0)) ** DTYPE(c.RCP)
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz_dev[...] = z8w[1:] - z8w[:-1]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    sr = state.scratch(surface, "mp_sr")
    vmi3d = state.scratch((nz, ny, nx), "p3_vmi")
    di3d = state.scratch((nz, ny, nx), "p3_di")
    rhopo3d = state.scratch((nz, ny, nx), "p3_rhopo")

    # WRF's itimestep starts at 1 (:1980 note); the counter gates only the
    # first-step saturation adjustment (:3348) and the it>1 activation
    # guard (:3303) -- nothing reads its magnitude, only ``it <= 1``.
    #
    # WRF restart-carries itimestep, and gpuwm honours that WITHOUT putting
    # a scalar in the restart stream: the only thing a resume needs to know
    # is whether this is the model's first step, and the model clock
    # already says so and is already serialized.  Seeding from
    # ``elapsed_seconds`` is therefore equivalent to carrying the counter,
    # and it is what stops a resumed trajectory from replaying the it=1
    # saturation adjustment in the middle of itself.  The attribute is
    # classified ``infra`` in gpuwm/io/restart.py for exactly this reason:
    # it is reconstructed from the clock, not dropped.
    prior = getattr(state, "p3_itimestep", None)
    if prior is None:
        prior = 1 if float(getattr(state, "elapsed_seconds", 0.0)) > 0.0 else 0
    it = int(prior) + 1
    state.p3_itimestep = it

    slabs = {}
    fields = {"th": th_dev, "qv": state.qv, "qc": state.qc, "qr": state.qr,
              "nr": state.nr, "qi": state.qi, "qir": state.qir,
              "ni": state.ni, "qib": state.qib, "th_old": state.th_old,
              "qv_old": state.qv_old, "pres": state.p, "dz": dz_dev}
    for name, dev in fields.items():
        slabs[name] = _to_slab(dev)

    ncol = ny * nx
    host_surface = {name: _host(arr).reshape(ncol).astype(np.float32,
                                                          copy=True)
                    for name, arr in (("rainnc", rainnc),
                                      ("rainncv", rainncv),
                                      ("sr", sr), ("snownc", snownc),
                                      ("snowncv", snowncv))}
    refl = np.empty((ncol, nz), dtype=np.float32)
    re_cloud = np.empty((ncol, nz), dtype=np.float32)
    re_ice = np.empty((ncol, nz), dtype=np.float32)
    vmi_slab = np.empty((ncol, nz), dtype=np.float32)
    di_slab = np.empty((ncol, nz), dtype=np.float32)
    rhopo_slab = np.empty((ncol, nz), dtype=np.float32)

    save_pre_mp_theta(state)   # WRF moist_physics_prep_em
    mp_p3_wrapper_wrf(
        slabs["th"], slabs["qv"], slabs["qc"], slabs["qr"], slabs["nr"],
        slabs["qi"], slabs["qir"], slabs["ni"], slabs["qib"],
        slabs["th_old"], slabs["qv_old"], slabs["pres"], slabs["dz"],
        float(dt), it,
        host_surface["rainnc"], host_surface["rainncv"], host_surface["sr"],
        host_surface["snownc"], host_surface["snowncv"],
        refl, re_cloud, re_ice, vmi_slab, di_slab, rhopo_slab,
        runtime=runtime)

    for name in ("qv", "qc", "qr", "nr", "qi", "qir", "ni", "qib",
                 "th_old", "qv_old"):
        _from_slab(fields[name], slabs[name])
    _from_slab(th_dev, slabs["th"])
    for name, dev in (("rainnc", rainnc), ("rainncv", rainncv), ("sr", sr),
                      ("snownc", snownc), ("snowncv", snowncv)):
        dev[...] = _as_backend(dev, host_surface[name].reshape(ny, nx))
    # effective radii: P3 returns metres; the state carries the
    # radiation-facing MICRON convention (state.py mp arms).
    state.effc[...] = _as_backend(
        state.effc, np.ascontiguousarray(
            (re_cloud * np.float32(1.0e6)).T).reshape(nz, ny, nx))
    state.effi[...] = _as_backend(
        state.effi, np.ascontiguousarray(
            (re_ice * np.float32(1.0e6)).T).reshape(nz, ny, nx))
    _from_slab(vmi3d, vmi_slab)
    _from_slab(di3d, di_slab)
    _from_slab(rhopo3d, rhopo_slab)

    if refl_10cm_due:
        from gpuwm.core.refl import stash_refl_10cm
        refl_dev = state.scratch((nz, ny, nx), "refl_10cm")
        _from_slab(refl_dev, refl)
        stash_refl_10cm(state, refl_dev)

    moist_physics_finish(state, cfg, th_dev, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv)


# ---------------------------------------------------------------------------
# The production adapter: device-resident P3.
# ---------------------------------------------------------------------------

def _p3_backend(state, cfg) -> str:
    """Which arm this call takes, and why.

    ``run.p3_backend`` selects; ``cuda`` is the default and is what "fixed"
    means here.  The reference path is taken WITHOUT being asked when the
    state is not on a CuPy backend at all -- a host-array state has no
    device to launch on, and refusing there would turn every CPU-side unit
    test of the scheme into a GPU test.
    """
    want = str(getattr(cfg, "p3_backend", "cuda"))
    if want == "reference":
        return "reference"
    if not hasattr(state.qv, "get"):        # host arrays: no device to use
        return "reference"
    return want


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Run P3 on ``state`` in place, on the card.

    Prognostic state never leaves the device.  The lookup tables are
    uploaded once per process and stay resident.  The per-step host work is
    the kernel launches and nothing else: no slab transpose, no host copy
    of any prognostic field, no Python loop over columns.

    ``run.p3_backend = "reference"`` selects the CPU float32 transcription
    (:func:`apply_reference`) instead, and a state whose arrays are not on
    a CuPy backend takes that path automatically.
    """
    backend = _p3_backend(state, cfg)
    if backend == "reference":
        return apply_reference(state, cfg, dt, refl_10cm_due=refl_10cm_due)

    import cupy as cp

    from gpuwm.core import constants as c
    from gpuwm.core import p3_device as PD
    from gpuwm.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    from gpuwm.core.state import DTYPE

    required = ("qi", "nr", "ni", "qir", "qib", "th_old", "qv_old")
    missing = [name for name in required
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=50 state lacks P3 fields: "
                         + ", ".join(missing))
    # p3_init stays on the host and runs once: it carries the loud SHA-256
    # table refusal, which must fire before any live state is touched.
    runtime = p3_init()
    tables = PD.device_tables(runtime)

    nz, ny, nx = state.p.shape
    ncol = ny * nx
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]

    th_dev = state.scratch((nz, ny, nx), "mp_th")
    dz_dev = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th_dev[...] = thb + state.thp
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz_dev[...] = z8w[1:] - z8w[:-1]
    # ``mp_pii`` is NOT allocated on this path.  The device kernels build
    # the Exner factor themselves from pres, exactly where the authority
    # builds it (:2293), so a separate full-domain array would be a second
    # copy of a value the kernel already holds in a register.

    surface = (ny, nx)
    diag_slots = {"zdbz": "refl_10cm", "effc": "p3_effc", "effi": "p3_effi",
                  "vmi": "p3_vmi", "di": "p3_di", "rhopo": "p3_rhopo"}
    diag = {name: state.scratch((nz, ny, nx), slot)
            for name, slot in diag_slots.items()}
    surf_slots = {"prt_liq": "p3_prt_liq", "prt_sol": "p3_prt_sol",
                  "rainnc": "mp_rainnc", "rainncv": "mp_rainncv",
                  "sr": "mp_sr", "snownc": "mp_snownc",
                  "snowncv": "mp_snowncv"}
    surf = {name: state.scratch(surface, slot)
            for name, slot in surf_slots.items()}

    # SEAM 1: ssat is a real (nz, ny, nx) device array threaded through
    # every kernel that the authority threads it through.  It is zero on
    # entry (the wrapper's :851) and p3_main DIAGNOSES it; the seam is that
    # it is storage and an argument, not a constant folded away because
    # log_predictSsat is false today.
    ssat = state.scratch((nz, ny, nx), "p3_ssat")
    ssat[...] = 0.0
    # SEAM 3: log_predictNc is driven by whether nc is a prognostic field
    # on the state, exactly as the authority drives it from `present(nc)`
    # (:819-820).  mp=50 has no such field and gets the specified-Nc
    # scratch; mp=51 would pass its own.
    prognostic_nc = getattr(state, "nc", None)
    log_predict_nc = bool(cfg.mp_physics == 51 and prognostic_nc is not None)
    nc_dev = prognostic_nc if log_predict_nc else state.scratch(
        (nz, ny, nx), "p3_nc")

    fields = {"qc": state.qc, "nc": nc_dev, "qr": state.qr, "nr": state.nr,
              "qi": state.qi, "qir": state.qir, "ni": state.ni,
              "qib": state.qib, "th": th_dev, "qv": state.qv,
              "th_old": state.th_old, "qv_old": state.qv_old,
              "ssat": ssat, "pres": state.p, "dz": dz_dev}

    # See gpuwm/core/p3.py:apply_reference for why the clock, not a
    # serialized counter, is what makes ``it`` survive a restart.
    prior = getattr(state, "p3_itimestep", None)
    if prior is None:
        prior = 1 if float(getattr(state, "elapsed_seconds", 0.0)) > 0.0 else 0
    it = int(prior) + 1
    state.p3_itimestep = it

    ws = getattr(state, "_p3_workspace", None)
    if ws is None or ws.ncol != ncol or ws.nk != nz:
        # Every P3 companion comes from DomainState.scratch, so the
        # allocation gate sees all of it and a normal timestep allocates
        # nothing.  gpuwm/core/preflight.py prices these slots for mp=50.
        ws = PD.make_workspace(
            ncol, nz,
            allocate=lambda slot, nlev: state.scratch((nlev, ny, nx), slot))
        state._p3_workspace = ws

    save_pre_mp_theta(state)          # WRF moist_physics_prep_em
    PD.run_p3_device(
        {k: v.reshape(nz, ncol) for k, v in fields.items()},
        {k: v.reshape(nz, ncol) for k, v in diag.items()},
        {k: v.reshape(ncol) for k, v in surf.items()},
        workspace=ws, tables=tables, dt=float(dt), it=it,
        log_predictNc=log_predict_nc,
        arm=PD.CONFIG_ARM[str(getattr(cfg, "p3_backend", "cuda"))])

    # P3 returns effective radii in METRES; the state carries the
    # radiation-facing MICRON convention (gpuwm/core/state.py mp arms).
    state.effc[...] = diag["effc"] * cp.float32(1.0e6)
    state.effi[...] = diag["effi"] * cp.float32(1.0e6)

    if refl_10cm_due:
        from gpuwm.core.refl import stash_refl_10cm
        stash_refl_10cm(state, diag["zdbz"])

    moist_physics_finish(state, cfg, th_dev, dt)
    return MicrophysicsDiagnostics(
        rainnc=surf["rainnc"], rainncv=surf["rainncv"], sr=surf["sr"],
        snownc=surf["snownc"], snowncv=surf["snowncv"])
