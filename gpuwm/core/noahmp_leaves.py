"""CPU references for pinned WRF v4.6.1 Noah-MP leaf routines.

Each function is a direct FP32 transcription of one ``private`` procedure of
``phys/module_sf_noahmplsm.F`` at commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``, validated bitwise against
``gpuwm/data/noahmp/oracle/noahmp-leaves.csv``.  That fixture is produced by
``tools/noahmp_wrf461_oracle/build_leaves.sh`` from a scratch copy of the
pinned source carrying only the audited visibility patch
``patches/noahmp-lsm-leaf-visibility.patch``.

Nothing here admits ``sf_surface_physics=4``.  These are the leaves of the
Noah-MP call tree; ENERGY, WATER and NOAHMP_SFLX are still open.

Conventions that the oracle and these references share:

* ``kind_phys`` is ``kind(1.0)``, i.e. FP32.  Every literal is materialised as
  ``np.float32`` so no expression silently widens.
* Layered arrays span WRF's ``-NSNOW+1 : NSOIL``.  They are carried as 0-based
  arrays of length ``NSNOW + NSOIL``; layer ``k`` lives at position
  ``k + NSNOW - 1``.
* Array entries the Fortran leaves undefined (INTENT(OUT) slots at or above
  ``ISNOW``, ROSR12 entries above ``NTOP``) are returned as 0.0, matching the
  harness pre-fill.  Callers must not consume them.
* Transcendentals go through ``gpuwm.core.noahmp_libm``, which reproduces
  glibc 2.39's ``expf``/``logf``/``log10f``/``powf`` bit for bit.  Rounding the
  FP64 result once -- the usual shim -- is **not** sufficient here: glibc's
  ``log10f`` is the 1993 SunPro FP32 reduction and disagrees with the
  correctly-rounded result on 18.47% of the FP32 domain (0.1, 1.0], which is
  exactly TDFCND's ``LOG10(SATRATIO)`` domain.  gfortran also routes a real
  constant exponent -- ``x**2.0``, ``x**3.0`` -- through ``powf`` rather than
  repeated multiplication, which for the cube differs from ``x*x*x``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


F = np.float32

# ---------------------------------------------------------------------------
# module_sf_noahmplsm.F:204-220 module parameters.
# ---------------------------------------------------------------------------
GRAV = F(9.80616)
SB = F(5.67e-08)
VKC = F(0.40)
TFRZ = F(273.16)
HSUB = F(2.8440e06)
HVAP = F(2.5104e06)
HFUS = F(0.3336e06)
CWAT = F(4.188e06)
CICE = F(2.094e06)
CPAIR = F(1004.64)
TKWAT = F(0.6)
TKICE = F(2.2)
TKAIR = F(0.023)
RAIR = F(287.04)
RW = F(461.269)
DENH2O = F(1000.0)
DENICE = F(917.0)

# WRF's own topology limits for the pinned four-layer / three-snow-layer stack.
NSNOW = 3
NSOIL = 4
NLAY = NSNOW + NSOIL


def _slot(k: int) -> int:
    """Position of WRF layer index ``k`` in a 0-based ``-NSNOW+1:NSOIL`` array."""
    return k + NSNOW - 1


# ---------------------------------------------------------------------------
# glibc-faithful FP32 transcendentals.  See gpuwm/core/noahmp_libm.py for the
# provenance and the measured verification against the live glibc 2.39.
# ---------------------------------------------------------------------------
from gpuwm.core.noahmp_libm import expf as _expf        # noqa: E402
from gpuwm.core.noahmp_libm import log10f as _log10f    # noqa: E402
from gpuwm.core.noahmp_libm import powf as _powf        # noqa: E402


# ---------------------------------------------------------------------------
# ATM -- module_sf_noahmplsm.F:1083-1251
# ---------------------------------------------------------------------------
RHO_GRPL = F(500.0)   # :1130
RHO_HAIL = F(917.0)   # :1131

ATM_INPUTS = (
    "sfcprs", "sfctmp", "q2", "prcpconv", "prcpnonc", "prcpshcv",
    "prcpsnow", "prcpgrpl", "prcphail", "soldn", "cosz",
)
ATM_OUTPUTS = (
    "thair", "qair", "eair", "rhoair", "qprecc", "qprecl", "solad1", "solad2",
    "solai1", "solai2", "swdown", "bdfall", "rain", "snow", "fp", "fpice",
    "prcp",
)


def atm(sfcprs, sfctmp, q2, prcpconv, prcpnonc, prcpshcv,
        prcpsnow, prcpgrpl, prcphail, soldn, cosz):
    """Re-process atmospheric forcing under the pinned ``OPT_SNF = 1``.

    ``PRCPSNOW``/``PRCPGRPL``/``PRCPHAIL`` are accepted because WRF's argument
    list carries them, and are unread because the only consumer is the
    ``IF(OPT_SNF == 4)`` block at :1217-1228.  The oracle's discrimination
    sweep proves they move nothing.
    """
    sfcprs = F(sfcprs)
    sfctmp = F(sfctmp)
    q2 = F(q2)
    prcpconv = F(prcpconv)
    prcpnonc = F(prcpnonc)
    prcpshcv = F(prcpshcv)
    soldn = F(soldn)
    cosz = F(cosz)

    pair = sfcprs                                              # :1144
    thair = F(sfctmp * _powf(F(sfcprs / pair), F(RAIR / CPAIR)))  # :1145
    qair = q2                                                  # :1147
    eair = F(F(qair * sfcprs) / F(F(0.622) + F(F(0.378) * qair)))  # :1149
    rhoair = F(F(sfcprs - F(F(0.378) * eair)) / F(RAIR * sfctmp))  # :1150

    if cosz <= F(0.0):                                         # :1152-1156
        swdown = F(0.0)
    else:
        swdown = soldn

    solad1 = F(F(swdown * F(0.7)) * F(0.5))                    # :1158
    solad2 = solad1                                            # :1159
    solai1 = F(F(swdown * F(0.3)) * F(0.5))                    # :1160
    solai2 = solai1                                            # :1161

    prcp = F(F(prcpconv + prcpnonc) + prcpshcv)                # :1163
    qprecc = F(F(0.10) * prcp)                                 # :1169
    qprecl = F(F(0.90) * prcp)                                 # :1170

    fp = F(0.0)                                                # :1175
    if F(qprecc + qprecl) > F(0.0):                            # :1176-1177
        fp = F(F(qprecc + qprecl)
               / F(F(F(10.0) * qprecc) + qprecl))

    # Jordan (1991), OPT_SNF == 1, :1183-1195.
    if sfctmp > F(TFRZ + F(2.5)):
        fpice = F(0.0)
    elif sfctmp <= F(TFRZ + F(0.5)):
        fpice = F(1.0)
    elif sfctmp <= F(TFRZ + F(2.0)):
        fpice = F(F(1.0) - F(F(-54.632) + F(F(0.2) * sfctmp)))
    else:
        fpice = F(0.6)

    # Hedstrom & Pomeroy (1998) fresh-snow density, :1216.
    bdfall = min(F(120.0),
                 F(F(67.92) + F(F(51.25) * _expf(F(F(sfctmp - TFRZ)
                                                   / F(2.59))))))
    bdfall = F(bdfall)

    rain = F(prcp * F(F(1.0) - fpice))                         # :1247
    snow = F(prcp * fpice)                                     # :1248

    return (thair, qair, eair, rhoair, qprecc, qprecl, solad1, solad2,
            solai1, solai2, swdown, bdfall, rain, snow, fp, fpice, prcp)


# ---------------------------------------------------------------------------
# ESAT -- module_sf_noahmplsm.F:4952-5001
# ---------------------------------------------------------------------------
_ESAT_A = (F(6.107799961), F(4.436518521e-01), F(1.428945805e-02),
           F(2.650648471e-04), F(3.031240396e-06), F(2.034080948e-08),
           F(6.136820929e-11))
_ESAT_B = (F(6.109177956), F(5.034698970e-01), F(1.886013408e-02),
           F(4.176223716e-04), F(5.824720280e-06), F(4.838803174e-08),
           F(1.838826904e-10))
_ESAT_C = (F(4.438099984e-01), F(2.857002636e-02), F(7.938054040e-04),
           F(1.215215065e-05), F(1.036561403e-07), F(3.532421810e-10),
           F(-7.090244804e-13))
_ESAT_D = (F(5.030305237e-01), F(3.773255020e-02), F(1.267995369e-03),
           F(2.477563108e-05), F(3.005693132e-07), F(2.158542548e-09),
           F(7.131097725e-12))

ESAT_INPUTS = ("tc",)
ESAT_OUTPUTS = ("esw", "esi", "desw", "desi")


def _esat_poly(t: np.float32, k: Sequence[np.float32]) -> np.float32:
    """``100.0*(k0+T*(k1+T*(k2+T*(k3+T*(k4+T*(k5+T*k6))))))``, inner to outer."""
    acc = F(k[5] + F(t * k[6]))
    acc = F(k[4] + F(t * acc))
    acc = F(k[3] + F(t * acc))
    acc = F(k[2] + F(t * acc))
    acc = F(k[1] + F(t * acc))
    acc = F(k[0] + F(t * acc))
    return F(F(100.0) * acc)


def esat(tc):
    """Saturation vapour pressure and its derivative, argument in **Celsius**.

    Every WRF call site passes ``TDC(T) = MIN(50, MAX(-50, T-TFRZ))`` from the
    statement functions at :3813 and :4332, so the domain is [-50, 50] C.
    """
    t = F(tc)
    return (_esat_poly(t, _ESAT_A), _esat_poly(t, _ESAT_B),
            _esat_poly(t, _ESAT_C), _esat_poly(t, _ESAT_D))


# ---------------------------------------------------------------------------
# ROSR12 -- module_sf_noahmplsm.F:5534-5591
# ---------------------------------------------------------------------------

def rosr12(a, b, c, d, ntop, nsoil=NSOIL, nsnow=NSNOW):
    """Thomas-algorithm tridiagonal solve over WRF layers ``NTOP .. NSOIL``.

    ``a``, ``b``, ``c``, ``d`` are 0-based arrays of length ``nsnow+nsoil``.
    Returns ``(p, delta, c_out)``, all the same shape.  ``c`` is INTENT(INOUT)
    in WRF: only ``C(NSOIL)`` is written, so ``c_out`` echoes the input
    elsewhere.  ``p``/``delta`` entries above ``NTOP`` are returned as 0.0,
    matching WRF leaving them untouched.
    """
    nlay = nsnow + nsoil
    a = np.asarray(a, dtype=F)
    b = np.asarray(b, dtype=F)
    c = np.asarray(c, dtype=F).copy()
    d = np.asarray(d, dtype=F)
    p = np.zeros(nlay, dtype=F)
    delta = np.zeros(nlay, dtype=F)

    def s(k: int) -> int:
        return k + nsnow - 1

    c[s(nsoil)] = F(0.0)                                       # :5565
    p[s(ntop)] = F(-c[s(ntop)] / b[s(ntop)])                    # :5566
    delta[s(ntop)] = F(d[s(ntop)] / b[s(ntop)])                 # :5570
    for k in range(ntop + 1, nsoil + 1):                        # :5574-5578
        recip = F(F(1.0) / F(b[s(k)] + F(a[s(k)] * p[s(k - 1)])))
        p[s(k)] = F(F(-c[s(k)]) * recip)
        delta[s(k)] = F(F(d[s(k)] - F(a[s(k)] * delta[s(k - 1)])) * recip)
    p[s(nsoil)] = delta[s(nsoil)]                               # :5582
    for k in range(ntop + 1, nsoil + 1):                        # :5586-5589
        kk = nsoil - k + (ntop - 1) + 1
        p[s(kk)] = F(F(p[s(kk)] * p[s(kk + 1)]) + delta[s(kk)])
    return p, delta, c


# ---------------------------------------------------------------------------
# CSNOW -- module_sf_noahmplsm.F:2514-2569
# ---------------------------------------------------------------------------

def csnow(isnow, snice, snliq, dzsnso, nsnow=NSNOW, nsoil=NSOIL):
    """Snow bulk density, volumetric heat capacity, and thermal conductivity.

    ``snice``/``snliq`` are 0-based arrays of length ``nsnow`` covering layers
    ``-nsnow+1 .. 0``.  ``dzsnso`` spans ``-nsnow+1 .. nsoil``; only its snow
    half is read.  Returns ``(tksno, cvsno, snicev, snliqv, epore)``, each of
    length ``nsnow``, zero above ``ISNOW``.
    """
    snice = np.asarray(snice, dtype=F)
    snliq = np.asarray(snliq, dtype=F)
    dzsnso = np.asarray(dzsnso, dtype=F)
    tksno = np.zeros(nsnow, dtype=F)
    cvsno = np.zeros(nsnow, dtype=F)
    snicev = np.zeros(nsnow, dtype=F)
    snliqv = np.zeros(nsnow, dtype=F)
    epore = np.zeros(nsnow, dtype=F)
    bdsnoi = np.zeros(nsnow, dtype=F)

    for iz in range(isnow + 1, 1):                              # :2547-2551
        j = iz + nsnow - 1
        snicev[j] = min(F(1.0), F(snice[j] / F(dzsnso[j] * DENICE)))
        epore[j] = F(F(1.0) - snicev[j])
        snliqv[j] = min(epore[j], F(snliq[j] / F(dzsnso[j] * DENH2O)))
    for iz in range(isnow + 1, 1):                              # :2553-2557
        j = iz + nsnow - 1
        bdsnoi[j] = F(F(snice[j] + snliq[j]) / dzsnso[j])
        cvsno[j] = F(F(CICE * snicev[j]) + F(CWAT * snliqv[j]))
    for iz in range(isnow + 1, 1):                              # :2561-2567
        j = iz + nsnow - 1
        tksno[j] = F(F(3.2217e-6) * _powf(bdsnoi[j], F(2.0)))
    return tksno, cvsno, snicev, snliqv, epore


# ---------------------------------------------------------------------------
# TDFCND -- module_sf_noahmplsm.F:2573-2680
# ---------------------------------------------------------------------------
TDFCND_INPUTS = ("smc", "sh2o", "smcmax", "quartz")
TDFCND_OUTPUTS = ("df",)


def tdfcnd(smc, sh2o, smcmax, quartz):
    """Peters-Lidard soil thermal conductivity for one layer."""
    smc = F(smc)
    sh2o = F(sh2o)
    smcmax = F(smcmax)
    quartz = F(quartz)

    satratio = F(smc / smcmax)                                  # :2628
    thkw = F(0.57)                                              # :2629
    thko = F(2.0)                                               # :2631
    thkqtz = F(7.7)                                             # :2634
    thks = F(_powf(thkqtz, quartz)
             * _powf(thko, F(F(1.0) - quartz)))                 # :2637

    xunfroz = F(1.0)                                            # :2640
    if smc > F(0.0):                                            # :2641
        xunfroz = F(sh2o / smc)
    xu = F(xunfroz * smcmax)                                    # :2643
    thksat = F(F(_powf(thks, F(F(1.0) - smcmax))
                 * _powf(TKICE, F(smcmax - xu)))
               * _powf(thkw, xu))                               # :2646-2647
    gammd = F(F(F(1.0) - smcmax) * F(2700.0))                   # :2650
    thkdry = F(F(F(F(0.135) * gammd) + F(64.7))
               / F(F(2700.0) - F(F(0.947) * gammd)))            # :2652

    if F(sh2o + F(0.0005)) < smc:                               # :2654
        ake = satratio
    elif satratio > F(0.1):                                     # :2664
        ake = F(_log10f(satratio) + F(1.0))                     # :2666
    else:
        ake = F(0.0)                                            # :2671
    return F(F(ake * F(thksat - thkdry)) + thkdry)              # :2677


# ---------------------------------------------------------------------------
# THERMOPROP -- module_sf_noahmplsm.F:2400-2510
# ---------------------------------------------------------------------------

def thermoprop(isnow, ist, dzsnso, dt, snowh, snice, snliq, smc, sh2o, stc,
               smcmax, csoil, quartz, urban_flag,
               nsnow=NSNOW, nsoil=NSOIL):
    """Snow/soil thermal conductivity, heat capacity, and phase-change factor.

    ``TG``, ``UR``, ``LAT``, ``Z0M`` and ``VEGTYP`` are part of WRF's argument
    list but the body never references them, so they are not parameters here;
    the oracle's discrimination sweep is the proof.  ``stc`` is read only for
    soil layers under ``IST == 2``.

    Returns ``(df, hcpct, snicev, snliqv, epore, fact)``.  ``fact`` entries at
    or above ``ISNOW`` are 0.0 -- WRF leaves them undefined.
    """
    nlay = nsnow + nsoil
    dzsnso = np.asarray(dzsnso, dtype=F)
    smc = np.asarray(smc, dtype=F)
    sh2o = np.asarray(sh2o, dtype=F)
    stc = np.asarray(stc, dtype=F)
    smcmax = np.asarray(smcmax, dtype=F)
    quartz = np.asarray(quartz, dtype=F)
    dt = F(dt)
    snowh = F(snowh)
    csoil = F(csoil)
    if dzsnso.shape[0] != nlay or stc.shape[0] != nlay:
        raise ValueError("dzsnso and stc must span -nsnow+1:nsoil")

    hcpct = np.zeros(nlay, dtype=F)                             # :2446
    df = np.zeros(nlay, dtype=F)                                # :2447
    fact = np.zeros(nlay, dtype=F)

    tksno, cvsno, snicev, snliqv, epore = csnow(
        isnow, snice, snliq, dzsnso, nsnow=nsnow, nsoil=nsoil)  # :2451

    for iz in range(isnow + 1, 1):                              # :2454-2457
        j = iz + nsnow - 1
        df[j] = tksno[j]
        hcpct[j] = cvsno[j]

    for iz in range(1, nsoil + 1):                              # :2461-2466
        j = iz + nsnow - 1
        sice = F(smc[iz - 1] - sh2o[iz - 1])
        acc = F(sh2o[iz - 1] * CWAT)
        acc = F(acc + F(F(F(1.0) - smcmax[iz - 1]) * csoil))
        acc = F(acc + F(F(smcmax[iz - 1] - smc[iz - 1]) * CPAIR))
        acc = F(acc + F(sice * CICE))
        hcpct[j] = acc
        df[j] = tdfcnd(smc[iz - 1], sh2o[iz - 1],
                       smcmax[iz - 1], quartz[iz - 1])

    if urban_flag:                                              # :2468-2472
        for iz in range(1, nsoil + 1):
            df[iz + nsnow - 1] = F(3.24)

    if ist == 2:                                                # :2483-2493
        for iz in range(1, nsoil + 1):
            j = iz + nsnow - 1
            if stc[j] > TFRZ:
                hcpct[j] = CWAT
                df[j] = TKWAT
            else:
                hcpct[j] = CICE
                df[j] = TKICE

    for iz in range(isnow + 1, nsoil + 1):                      # :2497-2499
        j = iz + nsnow - 1
        fact[j] = F(dt / F(hcpct[j] * dzsnso[j]))

    one = _slot(1)
    zero = _slot(0)
    if isnow == 0:                                              # :2503-2504
        df[one] = F(F(F(df[one] * dzsnso[one]) + F(F(0.35) * snowh))
                    / F(snowh + dzsnso[one]))
    else:                                                       # :2506
        df[one] = F(F(F(df[one] * dzsnso[one]) + F(df[zero] * dzsnso[zero]))
                    / F(dzsnso[zero] + dzsnso[one]))
    return df, hcpct, snicev, snliqv, epore, fact


# ---------------------------------------------------------------------------
# WDFCND1 -- module_sf_noahmplsm.F:9153-9188
# ---------------------------------------------------------------------------
WDFCND1_INPUTS = ("smc", "fcr", "smcmax", "bexp", "dwsat", "dksat")
WDFCND2_INPUTS = ("smc", "sice", "smcmax", "bexp", "dwsat", "dksat")
WDFCND_OUTPUTS = ("wdf", "wcnd")


def wdfcnd1(smc, fcr, smcmax, bexp, dwsat, dksat):
    """Soil water diffusivity and hydraulic conductivity, ``OPT_INF = 1`` form.

    WRF's local ``VKWGT`` (:9172) is declared and never used.
    """
    smc = F(smc)
    fcr = F(fcr)
    factr = max(F(0.01), F(smc / F(smcmax)))                    # :9177
    factr = F(factr)
    expon = F(F(bexp) + F(2.0))                                 # :9178
    wdf = F(F(dwsat) * _powf(factr, expon))                     # :9179
    wdf = F(wdf * F(F(1.0) - fcr))                              # :9180
    expon = F(F(F(2.0) * F(bexp)) + F(3.0))                     # :9184
    wcnd = F(F(dksat) * _powf(factr, expon))                    # :9185
    wcnd = F(wcnd * F(F(1.0) - fcr))                            # :9186
    return wdf, wcnd


# ---------------------------------------------------------------------------
# WDFCND2 -- module_sf_noahmplsm.F:9192-9232
# ---------------------------------------------------------------------------

def wdfcnd2(smc, sice, smcmax, bexp, dwsat, dksat):
    """Soil water diffusivity and hydraulic conductivity, ice-weighted form.

    Reached under the pinned ``OPT_INF = 1`` through ``INFIL`` at :7703, whose
    only gate is ``IF (QINSUR > 0.0)`` at :7655.
    """
    smc = F(smc)
    sice = F(sice)
    smcmax = F(smcmax)
    dwsat = F(dwsat)
    factr1 = F(F(0.05) / smcmax)                                # :9216
    factr2 = max(F(0.01), F(smc / smcmax))                      # :9217
    factr2 = F(factr2)
    factr1 = F(min(factr1, factr2))                             # :9218
    expon = F(F(bexp) + F(2.0))                                 # :9219
    wdf = F(dwsat * _powf(factr2, expon))                       # :9220
    if sice > F(0.0):                                           # :9222
        # (500*SICE)**3.0 is a real constant exponent: gfortran routes it
        # through powf, so this rounds once, not three times.
        vkwgt = F(F(1.0) / F(F(1.0) + _powf(F(F(500.0) * sice), F(3.0))))
        wdf = F(F(vkwgt * wdf)
                + F(F(F(F(1.0) - vkwgt) * dwsat) * _powf(factr1, expon)))
    expon = F(F(F(2.0) * F(bexp)) + F(3.0))                     # :9229
    wcnd = F(F(dksat) * _powf(factr2, expon))                   # :9230
    return wdf, wcnd


# ---------------------------------------------------------------------------
# Flat-slot adapters.  These mirror ``tools/noahmp_wrf461_oracle/run_leaves.F90``
# exactly, so the oracle CSV can be replayed without re-deriving any packing.
# ---------------------------------------------------------------------------

def _eval_atm(x, ix):
    return np.asarray(atm(*[x[i] for i in range(11)]), dtype=F)


def _eval_esat(x, ix):
    return np.asarray(esat(x[0]), dtype=F)


def _eval_rosr12(x, ix):
    a = x[0:NLAY]
    b = x[NLAY:2 * NLAY]
    c = x[2 * NLAY:3 * NLAY]
    d = x[3 * NLAY:4 * NLAY]
    p, delta, c_out = rosr12(a, b, c, d, int(ix[0]) + 1)
    return np.concatenate((p, delta, c_out)).astype(F)


def _eval_csnow(x, ix):
    snice = x[0:NSNOW]
    snliq = x[NSNOW:2 * NSNOW]
    dzsnso = np.full(NLAY, F(-9999.0), dtype=F)
    dzsnso[:NSNOW] = x[2 * NSNOW:3 * NSNOW]
    tksno, cvsno, snicev, snliqv, epore = csnow(int(ix[0]), snice, snliq,
                                               dzsnso)
    return np.concatenate((tksno, cvsno, snicev, snliqv, epore)).astype(F)


def _eval_tdfcnd(x, ix):
    return np.asarray([tdfcnd(x[0], x[1], x[2], x[3])], dtype=F)


def _eval_thermoprop(x, ix):
    dzsnso = x[0:NLAY]
    snice = x[NLAY:NLAY + NSNOW]
    snliq = x[NLAY + NSNOW:NLAY + 2 * NSNOW]
    base = NLAY + 2 * NSNOW
    smc = x[base:base + NSOIL]
    sh2o = x[base + NSOIL:base + 2 * NSOIL]
    base += 2 * NSOIL
    stc = x[base:base + NLAY]
    base += NLAY
    snowh, dt = x[base], x[base + 1]
    base += 7                        # snowh, dt, tg, ur, lat, z0m, zlvl
    smcmax = x[base:base + NSOIL]
    csoil = x[base + NSOIL]
    quartz = x[base + NSOIL + 1:base + 2 * NSOIL + 1]
    df, hcpct, snicev, snliqv, epore, fact = thermoprop(
        int(ix[0]), int(ix[1]), dzsnso, dt, snowh, snice, snliq, smc, sh2o,
        stc, smcmax, csoil, quartz, bool(ix[3]))
    return np.concatenate((df, hcpct, snicev, snliqv, epore, fact)).astype(F)


def _eval_wdfcnd1(x, ix):
    return np.asarray(wdfcnd1(x[0], x[1], x[2], x[3], x[4], x[5]), dtype=F)


def _eval_wdfcnd2(x, ix):
    return np.asarray(wdfcnd2(x[0], x[1], x[2], x[3], x[4], x[5]), dtype=F)


LEAF_EVALUATORS: Mapping[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "atm": _eval_atm,
    "esat": _eval_esat,
    "rosr12": _eval_rosr12,
    "csnow": _eval_csnow,
    "tdfcnd": _eval_tdfcnd,
    "thermoprop": _eval_thermoprop,
    "wdfcnd1": _eval_wdfcnd1,
    "wdfcnd2": _eval_wdfcnd2,
}
