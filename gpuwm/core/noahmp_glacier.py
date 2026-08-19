"""Bit-faithful FP32 transcription of WRF v4.6.1 ``NOAHMP_GLACIER``.

Source of truth
---------------
``phys/module_sf_noahmp_glacier.F`` at commit
``d66e442fccc04111067e29274c9f9eaccc3cef28`` of the pinned WRF v4.6.1
checkout, sha256
``bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579``
(3,080 lines) -- the same byte-frozen tree every other Noah-MP fixture in
this repository cites (``gpuwm/data/noahmp/oracle/PROVENANCE-driver.md``).
Every routine below carries its line anchors from that file.

This is the solver ``module_sf_noahmpdrv.F:1045-1150`` dispatches when a
land column's ``VEGTYP == ISICE_TABLE``: no vegetation, glacier energy
and water over land ice, snow over ice.  It replaces the whole
``NOAHMP_SFLX`` column for those points; the driver-side marshalling and
write-back live in :mod:`gpuwm.core.noahmp_runtime` next to their own
anchors.

Pinned option identity
----------------------
The glacier options come off the same namelist as the main column's and
are pinned to the same published identity
(:data:`gpuwm.config.NOAHMP_OPTION_IDENTITY`):

* ``opt_alb = 2`` -- CLASS snow albedo.  ``SNOWALB_BATS_GLACIER``
  (:812-858) is dead and is deliberately not transcribed.
* ``opt_snf = 1`` -- Jordan (1991) rain/snow partition (:2079-2091).
  The BATS and Noah arms (:2093-2107) are dead.
* ``opt_tbot = 2`` -- TBOT at ZBOT (:1461-1464).  The zero-flux arm is
  dead.
* ``opt_stc = 1`` -- semi-implicit layer-1 scheme.  The ``OPT_STC == 2``
  ``BI`` form (:1476-1478) and the post-TSNOSOI ``TG = TFRZ`` clamp
  (:519-521) are dead.
* ``opt_gla = 1`` -- glacier with ice phase change.  Every
  ``OPT_GLA == 2`` arm (:269-272, :1709-1733, :2155-2168, :2871-2892)
  is dead and deliberately not transcribed; :func:`noahmp_glacier`
  refuses any other identity rather than guessing.

What "bit-faithful" means here
------------------------------
``kind_phys == kind(1.0)`` is FP32.  Arithmetic is carried on
``numpy.float32`` operands so every binary operation rounds exactly
where gfortran rounds, expressions are grouped by Fortran's
left-to-right associativity, and the libm calls -- ``EXP`` in
SNOW_AGE/SNOWALB_CLASS/COMPACT/BDFALL, ``LOG``/``ATAN``/``**0.25`` in
SFCDIF1, ``**2.``/``**0.25`` real-exponent powers, ``**3``/``**4``
integer powers -- go through :mod:`gpuwm.core.noahmp_libm`'s glibc 2.39
transcriptions and the ``powi3``/``powi4`` multiply expansions, exactly
as the sibling leaf ports do.  ``SQRT`` is correctly rounded on both
sides and stays ``math.sqrt`` on a rounded operand.

Shared routines
---------------
``ESAT`` (:1123-1172) and ``SFCDIF1_GLACIER`` (:1175-1331) are
statement-for-statement identical to the main module's ``ESAT`` and
``SFCDIF1``, which :mod:`gpuwm.core.noahmp_bareflux` already transcribes
and holds at max_ulp 0 against the unmodified WRF leaf oracle; they are
imported from there rather than duplicated.  ``COMBO_GLACIER``
(:2638-2687) is identical to the main ``COMBO`` and is imported from
:mod:`gpuwm.core.noahmp_snow`.  The remaining snowpack routines are NOT
shared: the glacier module retains the pre-He-et-al.-2021 constants
(``ETA0 = 0.8e6``, ``DZMIN = 0.045/0.05/0.2``, the 0.05 m first-layer
threshold, the ``SSI`` percolation capacity) that the main module has
since moved off, so each is transcribed fresh below with its own
anchors.

Two documented departures from undefined WRF behaviour
------------------------------------------------------
Both follow the house rule: where WRF reads an undefined value, gpuwm
implements the defined behaviour and documents the divergence
(``never bit-exact to a bug``).

* ``HCPCT``: the driver's SNOWENERGY/SOILENERGY integrals
  (``module_sf_noahmpdrv.F:1381-1394``) read ``HCPCT`` for glacier
  columns too, but the glacier arm never writes it -- WRF reads
  whatever the previous NOAHMP_SFLX column left in the loop-carried
  local.  gpuwm returns the glacier column's own heat capacity from
  THERMOPROP_GLACIER (:534-604), which is the quantity the integral
  means.
* ``EFLXB``: same story (``EFLXBXY`` is written from a local the
  glacier arm never sets).  gpuwm returns TSNOSOI_GLACIER's ``BOTFLX``
  (:1459-1464), the glacier column's actual bottom energy influx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gpuwm.core.noahmp_bareflux import (_Sfcdif1State, _fmax, _fmin, _powi3,
                                        _powi4, _tdc, esat, sfcdif1)
from gpuwm.core.noahmp_libm import expf, logf, powf
from gpuwm.core.noahmp_snow import SnowColumn, _Col, combo

__all__ = [
    "GLACIER_OPTION_IDENTITY",
    "GlacierBalanceError",
    "GlacierResult",
    "noahmp_glacier",
]


def _f(x) -> np.float32:
    return np.float32(x)


# --- NOAHMP_GLACIER_GLOBALS, module_sf_noahmp_glacier.F:10-26 --------------
GRAV = _f(9.80616)
SB = _f(5.67e-08)
VKC = _f(0.40)
TFRZ = _f(273.16)
HSUB = _f(2.8440e06)
HVAP = _f(2.5104e06)
HFUS = _f(0.3336e06)
CWAT = _f(4.188e06)
CICE = _f(2.094e06)
CPAIR = _f(1004.64)
RAIR = _f(287.04)
RW = _f(461.269)
DENH2O = _f(1000.0)
DENICE = _f(917.0)

# --- snow-process parameters, :58-61 ---------------------------------------
Z0SNO = _f(0.002)   # snow surface roughness length (m)
SSI = _f(0.03)      # liquid water holding capacity for snowpack (m3/m3)
SWEMX = _f(1.00)    # new snow mass to fully cover old snow (mm)

# --- COMPACT_GLACIER's own PARAMETERs, :2389-2395 --------------------------
# These are the ORIGINAL Noah-MP values; the main module's COMPACT moved to
# He et al. 2021 (ETA0 = 1.33e6) and the glacier module did not follow.
_C2 = _f(21.0e-3)
_C3 = _f(2.5e-6)
_C4 = _f(0.04)
_C5 = _f(2.0)
_DM = _f(100.0)
_ETA0 = _f(0.8e+6)

# --- COMBINE_GLACIER's DZMIN DATA statement, :2500-2501 --------------------
# The glacier module keeps the pre-"MB: change limit" values.
_DZMIN = (_f(0.045), _f(0.05), _f(0.2))

# --- NOAHMP_GLACIER locals -------------------------------------------------
_ZBOT = _f(-8.0)          # :215
_EMG = _f(0.98)           # :474
_ALBICE = (_f(0.80), _f(0.55))   # :708-709
_NBAND = 2                # :705
_NITERB = 5               # :1009
_MPE = _f(1e-6)           # :1010

_ZERO = _f(0.0)
_ONE = _f(1.0)
_HALF = _f(0.5)
_TWO = _f(2.0)

#: The one glacier option identity this transcription implements; anything
#: else refuses.  Mirrors gpuwm.config.NOAHMP_OPTION_IDENTITY.
GLACIER_OPTION_IDENTITY = {
    "opt_alb": 2, "opt_snf": 1, "opt_tbot": 2, "opt_stc": 1, "opt_gla": 1,
}


class GlacierBalanceError(FloatingPointError):
    """ERROR_GLACIER's ``wrf_error_fatal`` (or FIRE <= 0), as an exception."""


# ===========================================================================
# ATM_GLACIER -- :299-349
# ===========================================================================

def _atm_glacier(sfcprs, sfctmp, q2, soldn, cosz):
    """Re-process atmospheric forcing; returns (qair, eair, rhoair,
    solad, solai, swdown).

    ``THAIR`` (:331) is computed by WRF and passed to nothing -- no
    glacier routine takes it -- so it is not evaluated here.
    """
    sfcprs, sfctmp = _f(sfcprs), _f(sfctmp)
    q2, soldn, cosz = _f(q2), _f(soldn), _f(cosz)

    qair = q2                                                     # :333
    eair = qair * sfcprs / (_f(0.622) + _f(0.378) * qair)         # :335
    rhoair = (sfcprs - _f(0.378) * eair) / (RAIR * sfctmp)        # :336

    if cosz <= _ZERO:                                             # :338-342
        swdown = _ZERO
    else:
        swdown = soldn

    solad = (swdown * _f(0.7) * _HALF, swdown * _f(0.7) * _HALF)  # :344-345
    solai = (swdown * _f(0.3) * _HALF, swdown * _f(0.3) * _HALF)  # :346-347
    return qair, eair, rhoair, solad, solai, swdown


# ===========================================================================
# CSNOW_GLACIER (:607-661) + THERMOPROP_GLACIER (:534-604)
# ===========================================================================

def _thermoprop_glacier(col: SnowColumn, dt):
    """Thermal conductivity/heat capacity; returns ``(df, hcpct, fact)``,
    each a full ``(-NSNOW+1 : NSOIL)`` float32 array (0-based storage)."""
    nsnow, nsoil = col.nsnow, col.nsoil
    dt = _f(dt)
    n = nsnow + nsoil
    df = _Col(np.zeros(n, dtype=np.float32), -nsnow + 1)
    hcpct = _Col(np.zeros(n, dtype=np.float32), -nsnow + 1)
    fact = _Col(np.zeros(n, dtype=np.float32), -nsnow + 1)
    SNICE, SNLIQ, DZSNSO = col.SNICE, col.SNLIQ, col.DZSNSO

    # CSNOW_GLACIER :639-659
    snicev = _Col(np.zeros(nsnow, dtype=np.float32), -nsnow + 1)
    epore = _Col(np.zeros(nsnow, dtype=np.float32), -nsnow + 1)
    snliqv = _Col(np.zeros(nsnow, dtype=np.float32), -nsnow + 1)
    bdsnoi = _Col(np.zeros(nsnow, dtype=np.float32), -nsnow + 1)
    for iz in range(col.isnow + 1, 1):                            # :639-643
        snicev[iz] = _fmin(_ONE, SNICE[iz] / (DZSNSO[iz] * DENICE))
        epore[iz] = _ONE - snicev[iz]
        snliqv[iz] = _fmin(epore[iz], SNLIQ[iz] / (DZSNSO[iz] * DENH2O))
    for iz in range(col.isnow + 1, 1):                            # :645-649
        bdsnoi[iz] = (SNICE[iz] + SNLIQ[iz]) / DZSNSO[iz]
        # CVSNO -> HCPCT snow slot (:573-576)
        hcpct[iz] = CICE * snicev[iz] + CWAT * snliqv[iz]
    for iz in range(col.isnow + 1, 1):                            # :653-659
        # TKSNO = 3.2217E-6*BDSNOI**2. -- REAL exponent, so powf.
        df[iz] = _f(3.2217e-6) * powf(bdsnoi[iz], _TWO)

    # soil thermal properties, Noah glacial-ice approximations :580-587
    for iz in range(1, nsoil + 1):
        zmid = _HALF * DZSNSO[iz]
        for iz2 in range(1, iz):
            zmid = zmid + DZSNSO[iz2]
        hcpct[iz] = _f(1.0e6) * (_f(0.8194) + _f(0.1309) * zmid)
        df[iz] = _f(0.32333) + (_f(0.10073) * zmid)

    for iz in range(col.isnow + 1, nsoil + 1):                    # :591-593
        fact[iz] = dt / (hcpct[iz] * DZSNSO[iz])

    # snow/soil interface :597-601
    if col.isnow == 0:
        df[1] = ((df[1] * DZSNSO[1] + _f(0.35) * col.snowh)
                 / (col.snowh + DZSNSO[1]))
    else:
        df[1] = ((df[1] * DZSNSO[1] + df[0] * DZSNSO[0])
                 / (DZSNSO[0] + DZSNSO[1]))
    return df, hcpct, fact


# ===========================================================================
# SNOW_AGE_GLACIER -- :758-809
# ===========================================================================

def _snow_age_glacier(dt, tg, sneqvo, sneqv, tauss):
    """BATS snow age; returns ``(tauss, fage)``.  Constants are the
    glacier module's own literals, not MPTABLE parameters."""
    dt, tg = _f(dt), _f(tg)
    sneqvo, sneqv, tauss = _f(sneqvo), _f(sneqv), _f(tauss)

    if sneqv <= _ZERO:                                            # :789-790
        tauss = _ZERO
    elif sneqv > _f(800.0):                                       # :791-792
        tauss = _ZERO
    else:                                                         # :793-805
        dela0 = _f(1.0e-6) * dt
        arg = _f(5.0e3) * (_ONE / TFRZ - _ONE / tg)
        age1 = expf(arg)
        age2 = expf(_fmin(_ZERO, _f(10.0) * arg))
        age3 = _f(0.3)
        tage = age1 + age2 + age3
        dela = dela0 * tage
        dels = _fmax(_ZERO, sneqv - sneqvo) / SWEMX
        sge = (tauss + dela) * (_ONE - dels)
        tauss = _fmax(_ZERO, sge)

    fage = tauss / (tauss + _ONE)                                 # :807
    return tauss, fage


# ===========================================================================
# SNOWALB_CLASS_GLACIER -- :861-904
# ===========================================================================

def _snowalb_class_glacier(qsnow, dt, albold):
    """CLASS albedo; returns the single-band ``ALB`` (all four albedo
    slots take it, :899-902)."""
    qsnow, dt, albold = _f(qsnow), _f(dt), _f(albold)
    alb = _f(0.55) + (albold - _f(0.55)) * expf(_f(-0.01) * dt / _f(3600.0))
    if qsnow > _ZERO:                                             # :895-897
        alb = alb + _fmin(qsnow * dt, SWEMX) * (_f(0.84) - alb) / SWEMX
    return alb


# ===========================================================================
# RADIATION_GLACIER -- :663-756
# ===========================================================================

def _radiation_glacier(dt, tg, sneqvo, sneqv, cosz, qsnow, solad, solai,
                       albold, tauss):
    """Returns ``(albold, tauss, sag, fsr, fsa)`` under opt_alb = 2."""
    albsnd = [_ZERO, _ZERO]                                       # :706-707
    albsni = [_ZERO, _ZERO]

    if cosz > _ZERO:                                              # :714-727
        tauss, _fage = _snow_age_glacier(dt, tg, sneqvo, sneqv, tauss)
        alb = _snowalb_class_glacier(qsnow, dt, albold)           # opt_alb=2
        albold = alb
        albsni[0] = alb                                           # :899-902
        albsni[1] = alb
        albsnd[0] = alb
        albsnd[1] = alb

    sag = _ZERO                                                   # :731-733
    fsa = _ZERO
    fsr = _ZERO

    fsno = _ZERO                                                  # :735-736
    if sneqv > _ZERO:
        fsno = _ONE

    for ib in range(_NBAND):                                      # :740-754
        albsnd[ib] = _ALBICE[ib] * (_ONE - fsno) + albsnd[ib] * fsno
        albsni[ib] = _ALBICE[ib] * (_ONE - fsno) + albsni[ib] * fsno
        absorbed = (solad[ib] * (_ONE - albsnd[ib])
                    + solai[ib] * (_ONE - albsni[ib]))
        sag = sag + absorbed
        fsa = fsa + absorbed
        ref = solad[ib] * albsnd[ib] + solai[ib] * albsni[ib]
        fsr = fsr + ref

    return albold, tauss, sag, fsr, fsa


# ===========================================================================
# GLACIER_FLUX -- :906-1121  (SFCDIF1_GLACIER and ESAT are the imported
# bareflux transcriptions; the glacier copies are statement-identical.)
# ===========================================================================

def _glacier_flux(*, isnow, df, dzsnso, zlvl, zpd, qair, sfctmp, rhoair,
                  sfcprs, ur, gamma, rsurf, lwdn, rhsur, smc, eair, stc,
                  sag, snowh, lathea, sh2o, cm, ch, tgb, qsfc, nsnow, nsoil):
    """Newton-Raphson glacier ground temperature and fluxes.

    Returns ``(cm, ch, tgb, qsfc, irb, shb, evb, ghb, t2mb, q2b, ehb2)``.
    ``df``/``dzsnso``/``stc`` are ``_Col`` views; ``smc``/``sh2o`` are
    ``_Col`` views over ``1..NSOIL``.
    """
    tgb = _f(tgb)
    mpe = _MPE                                                    # :1010
    h = _ZERO                                                     # :1015
    st = _Sfcdif1State(moz=_f(0.0), mozsgn=0, fh2=_f(0.0), fv=_f(0.1))

    cir = _EMG * SB                                               # :1018
    cgh = _TWO * df[isnow + 1] / dzsnso[isnow + 1]                # :1019

    csh = cev = estg = _ZERO
    irb = shb = evb = ghb = _ZERO
    qsfc = _f(qsfc)

    for it in range(1, _NITERB + 1):                              # :1022-1087
        z0h = Z0SNO                                               # :1024

        # SFCDIF1_GLACIER :1028-1031 -- the imported SFCDIF1 transcription.
        sfcdif1(st, it, _f(sfctmp), _f(rhoair), h, _f(qair),
                _f(zlvl), _f(zpd), Z0SNO, z0h, _f(ur), mpe)

        ramb = _fmax(_ONE, _ONE / (st.cm * _f(ur)))               # :1033
        rahb = _fmax(_ONE, _ONE / (st.ch * _f(ur)))               # :1034
        rawb = rahb                                               # :1035

        t = _tdc(tgb)                                             # :1039
        esatw, esati, dsatw, dsati = esat(t)                      # :1040
        if t > _ZERO:                                             # :1041-1047
            estg = esatw
            destg = dsatw
        else:
            estg = esati
            destg = dsati

        csh = _f(rhoair) * CPAIR / rahb                           # :1049
        # OPT_GLA == 1: the CEV arm is unconditional (:1050-1054).
        cev = _f(rhoair) * CPAIR / _f(gamma) / (_f(rsurf) + rawb)

        irb = cir * _powi4(tgb) - _EMG * _f(lwdn)                 # :1058
        shb = csh * (tgb - _f(sfctmp))                            # :1059
        evb = cev * (estg * _f(rhsur) - _f(eair))                 # :1060
        ghb = cgh * (tgb - stc[isnow + 1])                        # :1061

        b = _f(sag) - irb - shb - evb - ghb                       # :1063
        cir4t3 = _f(4.0) * cir * _powi3(tgb)
        a = cir4t3 + csh + cev * destg + cgh                      # :1064
        dtg = b / a                                               # :1065

        irb = irb + cir4t3 * dtg                                  # :1067
        shb = shb + csh * dtg                                     # :1068
        evb = evb + cev * destg * dtg                             # :1069
        ghb = ghb + cgh * dtg                                     # :1070

        tgb = tgb + dtg                                           # :1073

        h = csh * (tgb - _f(sfctmp))                              # :1076

        t = _tdc(tgb)                                             # :1078-1084
        esatw, esati, _dsw, _dsi = esat(t)
        estg = esatw if t > _ZERO else esati
        er = estg * _f(rhsur)
        qsfc = _f(0.622) * er / (_f(sfcprs) - _f(0.378) * er)     # :1085

    # :1092-1105 -- OPT_STC = 1, OPT_GLA = 1: reset TG over ice/snow.
    max_sice = _ZERO
    for j in range(1, nsoil + 1):
        sice_j = smc[j] - sh2o[j]                                 # :1092
        if j == 1 or sice_j > max_sice:
            max_sice = sice_j
    if (max_sice > _ZERO or _f(snowh) > _ZERO) and tgb > TFRZ:
        tgb = TFRZ
        t = _tdc(tgb)
        esatw, esati, _dsw, _dsi = esat(t)
        estg = esati                                              # :1098
        er = estg * _f(rhsur)
        qsfc = _f(0.622) * er / (_f(sfcprs) - _f(0.378) * er)     # :1099
        irb = cir * _powi4(tgb) - _EMG * _f(lwdn)                 # :1100
        shb = csh * (tgb - _f(sfctmp))                            # :1101
        evb = cev * (estg * _f(rhsur) - _f(eair))                 # :1102
        ghb = _f(sag) - (irb + shb + evb)                         # :1103

    # 2 m diagnostics :1108-1116
    z0h = Z0SNO
    ehb2 = st.fv * VKC / (logf((_TWO + z0h) / z0h) - st.fh2)      # :1108
    cq2b = ehb2                                                   # :1109
    if ehb2 < _f(1.0e-5):                                         # :1110-1116
        t2mb = tgb
        q2b = qsfc
    else:
        t2mb = tgb - shb / (_f(rhoair) * CPAIR) * _ONE / ehb2
        q2b = qsfc - evb / (_f(lathea) * _f(rhoair)) * (_ONE / cq2b
                                                        + _f(rsurf))

    ch = _ONE / rahb                                              # :1119
    return st.cm, ch, tgb, qsfc, irb, shb, evb, ghb, t2mb, q2b, ehb2


# ===========================================================================
# HRT_GLACIER (:1396-1491), HSTEP_GLACIER (:1494-1546),
# ROSR12_GLACIER (:1548-1605), TSNOSOI_GLACIER (:1333-1393)
# ===========================================================================

def _hrt_glacier(col: SnowColumn, stc, tbot, zbot, df, hcpct, ssoil):
    """Tri-diagonal coefficients; returns ``(ai, bi, ci, rhsts, botflx)``.

    ``PHI`` is identically zero (:1375) and is folded into the
    subtractions as the exact ``- 0.0`` it is.
    """
    nsnow, nsoil = col.nsnow, col.nsoil
    isnow = col.isnow
    n = nsnow + nsoil
    lo = -nsnow + 1
    ai = _Col(np.zeros(n, dtype=np.float32), lo)
    bi = _Col(np.zeros(n, dtype=np.float32), lo)
    ci = _Col(np.zeros(n, dtype=np.float32), lo)
    rhsts = _Col(np.zeros(n, dtype=np.float32), lo)
    ddz = _Col(np.zeros(n, dtype=np.float32), lo)
    denom = _Col(np.zeros(n, dtype=np.float32), lo)
    dtsdz = _Col(np.zeros(n, dtype=np.float32), lo)
    eflux = _Col(np.zeros(n, dtype=np.float32), lo)
    ZSNSO = col.ZSNSO
    botflx = _ZERO
    phi = _ZERO                                                   # :1375

    for k in range(isnow + 1, nsoil + 1):                         # :1442-1467
        if k == isnow + 1:
            denom[k] = -ZSNSO[k] * hcpct[k]
            temp1 = -ZSNSO[k + 1]
            ddz[k] = _TWO / temp1
            dtsdz[k] = _TWO * (stc[k] - stc[k + 1]) / temp1
            eflux[k] = df[k] * dtsdz[k] - _f(ssoil) - phi
        elif k < nsoil:
            denom[k] = (ZSNSO[k - 1] - ZSNSO[k]) * hcpct[k]
            temp1 = ZSNSO[k - 1] - ZSNSO[k + 1]
            ddz[k] = _TWO / temp1
            dtsdz[k] = _TWO * (stc[k] - stc[k + 1]) / temp1
            eflux[k] = (df[k] * dtsdz[k] - df[k - 1] * dtsdz[k - 1]) - phi
        else:
            denom[k] = (ZSNSO[k - 1] - ZSNSO[k]) * hcpct[k]
            # OPT_TBOT == 2 (:1461-1464): TBOT at ZBOT.
            dtsdz[k] = ((stc[k] - _f(tbot))
                        / (_HALF * (ZSNSO[k - 1] + ZSNSO[k]) - _f(zbot)))
            botflx = -df[k] * dtsdz[k]
            eflux[k] = (-botflx - df[k - 1] * dtsdz[k - 1]) - phi

    for k in range(isnow + 1, nsoil + 1):                         # :1469-1489
        if k == isnow + 1:
            ai[k] = _ZERO
            ci[k] = -df[k] * ddz[k] / denom[k]
            bi[k] = -ci[k]                                        # OPT_STC=1
        elif k < nsoil:
            ai[k] = -df[k - 1] * ddz[k - 1] / denom[k]
            ci[k] = -df[k] * ddz[k] / denom[k]
            bi[k] = -(ai[k] + ci[k])
        else:
            ai[k] = -df[k - 1] * ddz[k - 1] / denom[k]
            ci[k] = _ZERO
            bi[k] = -(ai[k] + ci[k])
        rhsts[k] = eflux[k] / (-denom[k])

    return ai, bi, ci, rhsts, botflx


def _rosr12_glacier(p, a, b, c, d, delta, ntop, nsoil):
    """Tri-diagonal solve (:1548-1605); mutates ``p``/``c``/``delta``."""
    c[nsoil] = _ZERO                                              # :1579
    p[ntop] = -c[ntop] / b[ntop]                                  # :1580
    delta[ntop] = d[ntop] / b[ntop]                               # :1584
    for k in range(ntop + 1, nsoil + 1):                          # :1588-1592
        p[k] = -c[k] * (_ONE / (b[k] + a[k] * p[k - 1]))
        delta[k] = ((d[k] - a[k] * delta[k - 1])
                    * (_ONE / (b[k] + a[k] * p[k - 1])))
    p[nsoil] = delta[nsoil]                                       # :1596
    for k in range(ntop + 1, nsoil + 1):                          # :1600-1603
        kk = nsoil - k + (ntop - 1) + 1
        p[kk] = p[kk] * p[kk + 1] + delta[kk]


def _tsnosoi_glacier(col: SnowColumn, stc, tbot, ssoil, dt, snowh,
                     df, hcpct):
    """Snow/soil temperatures (:1333-1393); mutates ``stc``.
    Returns ``botflx`` (WRF's discarded EFLXB, published as the defined
    replacement for the driver's undefined ``EFLXBXY`` read)."""
    nsnow, nsoil = col.nsnow, col.nsoil
    isnow = col.isnow
    zbotsno = _ZBOT - _f(snowh)                                   # :1379

    ai, bi, ci, rhsts, botflx = _hrt_glacier(
        col, stc, tbot, zbotsno, df, hcpct, ssoil)                # :1383-1387

    # HSTEP_GLACIER :1522-1544
    dt = _f(dt)
    for k in range(isnow + 1, nsoil + 1):                         # :1522-1527
        rhsts[k] = rhsts[k] * dt
        ai[k] = ai[k] * dt
        bi[k] = _ONE + bi[k] * dt
        ci[k] = ci[k] * dt
    lo = -nsnow + 1
    rhstsin = _Col(rhsts.data.copy(), lo)                         # :1531-1534
    ciin = _Col(ci.data.copy(), lo)
    _rosr12_glacier(ci, ai, bi, ciin, rhstsin, rhsts,
                    isnow + 1, nsoil)                             # :1538
    for k in range(isnow + 1, nsoil + 1):                         # :1542-1544
        stc[k] = stc[k] + ci[k]
    return botflx


# ===========================================================================
# PHASECHANGE_GLACIER -- :1608-1995 (OPT_GLA == 1)
# ===========================================================================

def _phasechange_glacier(col: SnowColumn, stc, dt, fact, smc, sh2o):
    """Melting/freezing of snow and glacier ice; mutates ``col`` (snice/
    snliq/sneqv/snowh), ``stc``, ``smc``, ``sh2o``.

    Returns ``(qmelt, imelt, ponding)`` with ``imelt`` a full-span
    ``_Col`` of int32.
    """
    nsnow, nsoil = col.nsnow, col.nsoil
    isnow = col.isnow
    dt = _f(dt)
    lo = -nsnow + 1
    n = nsnow + nsoil

    imelt = _Col(np.zeros(n, dtype=np.int32), lo)
    hm = _Col(np.zeros(n, dtype=np.float32), lo)
    xm = _Col(np.zeros(n, dtype=np.float32), lo)
    wmass0 = _Col(np.zeros(n, dtype=np.float32), lo)
    wice0 = _Col(np.zeros(n, dtype=np.float32), lo)
    wliq0 = _Col(np.zeros(n, dtype=np.float32), lo)
    mice = _Col(np.zeros(n, dtype=np.float32), lo)
    mliq = _Col(np.zeros(n, dtype=np.float32), lo)
    heatr = _Col(np.zeros(n, dtype=np.float32), lo)
    SNICE, SNLIQ, DZSNSO = col.SNICE, col.SNLIQ, col.DZSNSO

    qmelt = _ZERO                                                 # :1660-1662
    ponding = _ZERO
    xmf = _ZERO

    for j in range(isnow + 1, 1):                                 # :1664-1667
        mice[j] = SNICE[j]
        mliq[j] = SNLIQ[j]

    for j in range(isnow + 1, 1):                                 # :1669-1676
        imelt[j] = 0
        hm[j] = _ZERO
        xm[j] = _ZERO
        wice0[j] = mice[j]
        wliq0[j] = mliq[j]
        wmass0[j] = mice[j] + mliq[j]

    for j in range(isnow + 1, 1):                                 # :1678-1686
        if mice[j] > _ZERO and stc[j] >= TFRZ:
            imelt[j] = 1
        if mliq[j] > _ZERO and stc[j] < TFRZ:
            imelt[j] = 2

    for j in range(isnow + 1, 1):                                 # :1690-1705
        if imelt[j] > 0:
            hm[j] = (stc[j] - TFRZ) / fact[j]
            stc[j] = TFRZ
        if imelt[j] == 1 and hm[j] < _ZERO:
            hm[j] = _ZERO
            imelt[j] = 0
        if imelt[j] == 2 and hm[j] > _ZERO:
            hm[j] = _ZERO
            imelt[j] = 0
        xm[j] = hm[j] * dt / HFUS

    # (:1709-1733 is OPT_GLA == 2 and is dead under the pinned identity.)

    for j in range(isnow + 1, 1):                                 # :1737-1759
        if imelt[j] > 0 and abs(hm[j]) > _ZERO:
            heatr[j] = _ZERO
            if xm[j] > _ZERO:
                mice[j] = _fmax(_ZERO, wice0[j] - xm[j])
                heatr[j] = hm[j] - HFUS * (wice0[j] - mice[j]) / dt
            elif xm[j] < _ZERO:
                mice[j] = _fmin(wmass0[j], wice0[j] - xm[j])
                heatr[j] = hm[j] - HFUS * (wice0[j] - mice[j]) / dt
            mliq[j] = _fmax(_ZERO, wmass0[j] - mice[j])
            if abs(heatr[j]) > _ZERO:
                stc[j] = stc[j] + fact[j] * heatr[j]
                if mliq[j] * mice[j] > _ZERO:
                    stc[j] = TFRZ
            qmelt = qmelt + _fmax(_ZERO, wice0[j] - mice[j]) / dt

    # ---- OPT_GLA == 1: operate on the ice (soil) layers :1761-1977 --------
    for j in range(1, nsoil + 1):                                 # :1763-1766
        mliq[j] = sh2o[j] * DZSNSO[j] * _f(1000.0)
        mice[j] = (smc[j] - sh2o[j]) * DZSNSO[j] * _f(1000.0)

    for j in range(1, nsoil + 1):                                 # :1768-1775
        imelt[j] = 0
        hm[j] = _ZERO
        xm[j] = _ZERO
        wice0[j] = mice[j]
        wliq0[j] = mliq[j]
        wmass0[j] = mice[j] + mliq[j]

    for j in range(1, nsoil + 1):                                 # :1777-1791
        if mice[j] > _ZERO and stc[j] >= TFRZ:
            imelt[j] = 1
        if mliq[j] > _ZERO and stc[j] < TFRZ:
            imelt[j] = 2
        # snow exists but no layer :1786-1790
        if isnow == 0 and col.sneqv > _ZERO and j == 1:
            if stc[j] >= TFRZ:
                imelt[j] = 1

    for j in range(1, nsoil + 1):                                 # :1795-1810
        if imelt[j] > 0:
            hm[j] = (stc[j] - TFRZ) / fact[j]
            stc[j] = TFRZ
        if imelt[j] == 1 and hm[j] < _ZERO:
            hm[j] = _ZERO
            imelt[j] = 0
        if imelt[j] == 2 and hm[j] > _ZERO:
            hm[j] = _ZERO
            imelt[j] = 0
        xm[j] = hm[j] * dt / HFUS

    # snow without a layer :1814-1832
    if isnow == 0 and col.sneqv > _ZERO and xm[1] > _ZERO:
        temp1 = col.sneqv
        col.sneqv = _fmax(_ZERO, temp1 - xm[1])
        propor = col.sneqv / temp1
        col.snowh = _fmax(_ZERO, propor * col.snowh)
        heatr[1] = hm[1] - HFUS * (temp1 - col.sneqv) / dt
        if heatr[1] > _ZERO:
            xm[1] = heatr[1] * dt / HFUS
            hm[1] = heatr[1]
            imelt[1] = 1
        else:
            xm[1] = _ZERO
            hm[1] = _ZERO
            imelt[1] = 0
        qmelt = _fmax(_ZERO, temp1 - col.sneqv) / dt
        xmf = HFUS * qmelt
        ponding = temp1 - col.sneqv

    # rate of melting and freezing for soil :1836-1863
    for j in range(1, nsoil + 1):
        if imelt[j] > 0 and abs(hm[j]) > _ZERO:
            heatr[j] = _ZERO
            if xm[j] > _ZERO:
                mice[j] = _fmax(_ZERO, wice0[j] - xm[j])
                heatr[j] = hm[j] - HFUS * (wice0[j] - mice[j]) / dt
            elif xm[j] < _ZERO:
                mice[j] = _fmin(wmass0[j], wice0[j] - xm[j])
                heatr[j] = hm[j] - HFUS * (wice0[j] - mice[j]) / dt
            mliq[j] = _fmax(_ZERO, wmass0[j] - mice[j])
            if abs(heatr[j]) > _ZERO:
                stc[j] = stc[j] + fact[j] * heatr[j]
                # J <= 0 arm (:1852-1854) is unreachable in this 1..NSOIL
                # loop and the J < 1 QMELT arm (:1859-1861) likewise; both
                # are transcribed as the dead code they are.
            if j > 0:
                xmf = xmf + HFUS * (wice0[j] - mice[j]) / dt

    heatr.data[...] = _ZERO                                       # :1864-1865
    xm.data[...] = _ZERO

    # ---- residuals in ice/soil :1867-1975 ---------------------------------
    def _any_gt(lo_, hi_, pred):
        return any(pred(stc[j]) for j in range(lo_, hi_ + 1))

    # FIRST REMOVE EXCESS HEAT BY REDUCING TEMPERATURE OF LAYERS :1871-1892
    if (_any_gt(1, 4, lambda v: v > TFRZ)
            and _any_gt(1, 4, lambda v: v < TFRZ)):
        for j in range(1, nsoil + 1):
            if stc[j] > TFRZ:
                heatr[j] = (stc[j] - TFRZ) / fact[j]
                for k in range(1, nsoil + 1):
                    if j != k and stc[k] < TFRZ and heatr[j] > _f(0.1):
                        heatr[k] = (stc[k] - TFRZ) / fact[k]
                        if abs(heatr[k]) > heatr[j]:
                            heatr[k] = heatr[k] + heatr[j]
                            stc[k] = TFRZ + heatr[k] * fact[k]
                            heatr[j] = _ZERO
                        else:
                            heatr[j] = heatr[j] + heatr[k]
                            heatr[k] = _ZERO
                            stc[k] = TFRZ
                stc[j] = TFRZ + heatr[j] * fact[j]

    # NOW REMOVE EXCESS COLD BY INCREASING TEMPERATURE :1896-1917
    if (_any_gt(1, 4, lambda v: v > TFRZ)
            and _any_gt(1, 4, lambda v: v < TFRZ)):
        for j in range(1, nsoil + 1):
            if stc[j] < TFRZ:
                heatr[j] = (stc[j] - TFRZ) / fact[j]
                for k in range(1, nsoil + 1):
                    if j != k and stc[k] > TFRZ and heatr[j] < _f(-0.1):
                        heatr[k] = (stc[k] - TFRZ) / fact[k]
                        if heatr[k] > abs(heatr[j]):
                            heatr[k] = heatr[k] + heatr[j]
                            stc[k] = TFRZ + heatr[k] * fact[k]
                            heatr[j] = _ZERO
                        else:
                            heatr[j] = heatr[j] + heatr[k]
                            heatr[k] = _ZERO
                            stc[k] = TFRZ
                stc[j] = TFRZ + heatr[j] * fact[j]

    # NOW REMOVE EXCESS HEAT BY MELTING ICE :1921-1946
    if (_any_gt(1, 4, lambda v: v > TFRZ)
            and any(mice[j] > _ZERO for j in range(1, 5))):
        for j in range(1, nsoil + 1):
            if stc[j] > TFRZ:
                heatr[j] = (stc[j] - TFRZ) / fact[j]
                xm[j] = heatr[j] * dt / HFUS
                for k in range(1, nsoil + 1):
                    if j != k and mice[k] > _ZERO and xm[j] > _f(0.1):
                        if mice[k] > xm[j]:
                            mice[k] = mice[k] - xm[j]
                            xmf = xmf + HFUS * xm[j] / dt
                            stc[k] = TFRZ
                            xm[j] = _ZERO
                        else:
                            xm[j] = xm[j] - mice[k]
                            xmf = xmf + HFUS * mice[k] / dt
                            mice[k] = _ZERO
                            stc[k] = TFRZ
                        mliq[k] = _fmax(_ZERO, wmass0[k] - mice[k])
                heatr[j] = xm[j] * HFUS / dt
                stc[j] = TFRZ + heatr[j] * fact[j]

    # NOW REMOVE EXCESS COLD BY FREEZING LIQUID :1950-1975
    if (_any_gt(1, 4, lambda v: v < TFRZ)
            and any(mliq[j] > _ZERO for j in range(1, 5))):
        for j in range(1, nsoil + 1):
            if stc[j] < TFRZ:
                heatr[j] = (stc[j] - TFRZ) / fact[j]
                xm[j] = heatr[j] * dt / HFUS
                for k in range(1, nsoil + 1):
                    if j != k and mliq[k] > _ZERO and xm[j] < _f(-0.1):
                        if mliq[k] > abs(xm[j]):
                            mice[k] = mice[k] - xm[j]
                            xmf = xmf + HFUS * xm[j] / dt
                            stc[k] = TFRZ
                            xm[j] = _ZERO
                        else:
                            xm[j] = xm[j] + mliq[k]
                            xmf = xmf - HFUS * mliq[k] / dt
                            mice[k] = wmass0[k]
                            stc[k] = TFRZ
                        mliq[k] = _fmax(_ZERO, wmass0[k] - mice[k])
                heatr[j] = xm[j] * HFUS / dt
                stc[j] = TFRZ + heatr[j] * fact[j]

    for j in range(isnow + 1, 1):                                 # :1979-1982
        SNLIQ[j] = mliq[j]
        SNICE[j] = mice[j]

    for j in range(1, nsoil + 1):                                 # :1984-1993
        # OPT_GLA == 1
        sh2o[j] = mliq[j] / (_f(1000.0) * DZSNSO[j])
        sh2o[j] = _fmax(_ZERO, _fmin(_ONE, sh2o[j]))
        smc[j] = _ONE

    del xmf, wliq0
    return qmelt, imelt, ponding


# ===========================================================================
# ENERGY_GLACIER -- :352-532
# ===========================================================================

def _energy_glacier(col: SnowColumn, *, dt, qsnow, rhoair, eair, sfcprs,
                    qair, sfctmp, lwdn, uu, vv, solad, solai, cosz, zref,
                    tbot, stc, smc, sh2o, tg, sneqvo, albold, cm, ch,
                    tauss, qsfc):
    """Energy budget; mutates ``col`` (via phasechange), ``stc``, ``smc``,
    ``sh2o``.  Returns a dict of the driver-visible outputs plus the
    ``hcpct`` array and the bottom flux."""
    # wind speed :449 -- REAL exponents, so powf.
    ur = _fmax(_f(math.sqrt(_f(powf(_f(uu), _TWO) + powf(_f(vv), _TWO)))),
               _ONE)

    z0mg = Z0SNO                                                  # :453
    zpd = col.snowh                                               # :454
    zlvl = zpd + _f(zref)                                         # :456

    df, hcpct, fact = _thermoprop_glacier(col, dt)                # :460-463

    albold, tauss, sag, fsr, fsa = _radiation_glacier(
        dt, tg, sneqvo, col.sneqv, cosz, qsnow, solad, solai,
        albold, tauss)                                            # :467-470

    emg = _EMG                                                    # :474
    rhsur = _ONE                                                  # :478
    rsurf = _ONE                                                  # :479
    lathea = HSUB                                                 # :483
    gamma = CPAIR * _f(sfcprs) / (_f(0.622) * lathea)             # :484

    smc_v = _Col(np.asarray(smc, dtype=np.float32), 1)
    sh2o_v = _Col(np.asarray(sh2o, dtype=np.float32), 1)
    cm, ch, tg, qsfc, irb, shb, evb, ghb, t2mb, q2b, ehb2 = _glacier_flux(
        isnow=col.isnow, df=df, dzsnso=col.DZSNSO, zlvl=zlvl, zpd=zpd,
        qair=qair, sfctmp=sfctmp, rhoair=rhoair, sfcprs=sfcprs, ur=ur,
        gamma=gamma, rsurf=rsurf, lwdn=lwdn, rhsur=rhsur, smc=smc_v,
        eair=eair, stc=stc, sag=sag, snowh=col.snowh, lathea=lathea,
        sh2o=sh2o_v, cm=cm, ch=ch, tgb=tg, qsfc=qsfc,
        nsnow=col.nsnow, nsoil=col.nsoil)                         # :488-494

    fira = irb
    fsh = shb
    fgev = evb
    ssoil = ghb

    fire = _f(lwdn) + fira                                        # :498
    if fire <= _ZERO:                                             # :500
        raise GlacierBalanceError(
            "NOAHMP_GLACIER: emitted longwave <= 0 "
            "(module_sf_noahmp_glacier.F:500 wrf_error_fatal)")
    emissi = emg                                                  # :503
    trad = powf((fire - (_ONE - emissi) * _f(lwdn)) / (emissi * SB),
                _f(0.25))                                         # :509

    eflxb = _tsnosoi_glacier(col, stc, tbot, ssoil, dt, col.snowh,
                             df, hcpct)                           # :513-516

    # :519-521 (OPT_STC == 2 TG clamp) is dead under opt_stc = 1.

    qmelt, imelt, ponding = _phasechange_glacier(
        col, stc, dt, fact, smc_v, sh2o_v)                        # :525-529

    smc[...] = smc_v.data
    sh2o[...] = sh2o_v.data
    return {
        "tg": tg, "cm": cm, "ch": ch, "qsfc": qsfc, "albold": albold,
        "tauss": tauss, "imelt": imelt, "qmelt": qmelt,
        "ponding": ponding, "sag": sag, "fsa": fsa, "fsr": fsr,
        "fira": fira, "fsh": fsh, "fgev": fgev, "trad": trad,
        "t2m": t2mb, "ssoil": ssoil, "lathea": lathea, "q2e": q2b,
        "emissi": emissi, "ch2b": ehb2, "hcpct": hcpct.data.copy(),
        "eflxb": eflxb,
    }


# ===========================================================================
# SNOWFALL_GLACIER -- :2302-2364
# ===========================================================================

def _snowfall_glacier(col: SnowColumn, dt, qsnow, snowhin, sfctmp) -> None:
    """New snowfall; the glacier variant keeps the 0.05 m first-layer
    threshold and the ``QSNOW > 0`` requirement the main module dropped."""
    dt, qsnow = _f(dt), _f(qsnow)
    snowhin, sfctmp = _f(snowhin), _f(sfctmp)
    DZSNSO, STC, SNICE, SNLIQ = col.DZSNSO, col.STC, col.SNICE, col.SNLIQ

    newnode = 0                                                   # :2335

    if col.isnow == 0 and qsnow > _ZERO:                          # :2339-2342
        col.snowh = col.snowh + snowhin * dt
        col.sneqv = col.sneqv + qsnow * dt

    if col.isnow == 0 and qsnow > _ZERO and col.snowh >= _f(0.05):
        col.isnow = -1                                            # :2346-2354
        newnode = 1
        DZSNSO[0] = col.snowh
        col.snowh = _ZERO
        STC[0] = _fmin(_f(273.16), sfctmp)
        SNICE[0] = col.sneqv
        SNLIQ[0] = _ZERO

    if col.isnow < 0 and newnode == 0 and qsnow > _ZERO:          # :2358-2361
        SNICE[col.isnow + 1] = SNICE[col.isnow + 1] + qsnow * dt
        DZSNSO[col.isnow + 1] = DZSNSO[col.isnow + 1] + snowhin * dt


# ===========================================================================
# COMPACT_GLACIER -- :2367-2464
# ===========================================================================

def _compact_glacier(col: SnowColumn, dt, stc, imelt, ficeold) -> None:
    """Snow compaction with the ORIGINAL constants (ETA0 = 0.8e6)."""
    dt = _f(dt)
    SNICE, SNLIQ, DZSNSO = col.SNICE, col.SNLIQ, col.DZSNSO
    fice = _Col(np.zeros(col.nsnow, dtype=np.float32), -col.nsnow + 1)

    burden = _ZERO                                                # :2411
    for j in range(col.isnow + 1, 1):                             # :2413-2462
        wx = SNICE[j] + SNLIQ[j]
        fice[j] = SNICE[j] / wx
        void = _ONE - (SNICE[j] / DENICE + SNLIQ[j] / DENH2O) / DZSNSO[j]

        if void > _f(0.001) and SNICE[j] > _f(0.1):
            bi = SNICE[j] / DZSNSO[j]
            td = _fmax(_ZERO, TFRZ - stc[j])
            dexpf = expf(-_C4 * td)

            ddz1 = -_C3 * dexpf                                   # :2427
            if bi > _DM:                                          # :2429
                ddz1 = ddz1 * expf(_f(-46.0e-3) * (bi - _DM))
            if SNLIQ[j] > _f(0.01) * DZSNSO[j]:                   # :2433
                ddz1 = ddz1 * _C5

            ddz2 = (-(burden + _HALF * wx)
                    * expf(_f(-0.08) * td - _C2 * bi) / _ETA0)    # :2437

            if imelt[j] == 1:                                     # :2441-2446
                ddz3 = _fmax(_ZERO, (ficeold[j] - fice[j])
                             / _fmax(_f(1.0e-6), ficeold[j]))
                ddz3 = -ddz3 / dt
            else:
                ddz3 = _ZERO

            pdzdtc = (ddz1 + ddz2 + ddz3) * dt                    # :2450-2451
            pdzdtc = _fmax(_f(-0.5), pdzdtc)

            DZSNSO[j] = DZSNSO[j] * (_ONE + pdzdtc)               # :2455

        burden = burden + wx                                      # :2460


# ===========================================================================
# COMBINE_GLACIER -- :2466-2634
# ===========================================================================

def _combine_glacier(col: SnowColumn, stc, sh2o, sice, ponding1, ponding2):
    """Combine thin snow layers; returns ``(ponding1, ponding2)``."""
    SNICE, SNLIQ, DZSNSO = col.SNICE, col.SNLIQ, col.DZSNSO

    isnow_old = col.isnow                                         # :2505

    for j in range(isnow_old + 1, 1):                             # :2507-2539
        if SNICE[j] <= _f(0.1):
            if j != 0:
                SNLIQ[j + 1] = SNLIQ[j + 1] + SNLIQ[j]
                SNICE[j + 1] = SNICE[j + 1] + SNICE[j]
            else:
                if isnow_old < -1:                                # :2513-2515
                    SNLIQ[j - 1] = SNLIQ[j - 1] + SNLIQ[j]
                    SNICE[j - 1] = SNICE[j - 1] + SNICE[j]
                else:                                             # :2517-2522
                    ponding1 = ponding1 + SNLIQ[j]
                    col.sneqv = SNICE[j]
                    col.snowh = DZSNSO[j]
                    SNLIQ[j] = _ZERO
                    SNICE[j] = _ZERO
                    DZSNSO[j] = _ZERO
            # shift all elements above this down by one :2529-2536
            if j > col.isnow + 1 and col.isnow < -1:
                for i in range(j, col.isnow + 1, -1):
                    stc[i] = stc[i - 1]
                    SNLIQ[i] = SNLIQ[i - 1]
                    SNICE[i] = SNICE[i - 1]
                    DZSNSO[i] = DZSNSO[i - 1]
            col.isnow = col.isnow + 1                             # :2537

    if sice[1] < _ZERO:                                           # :2543-2546
        sh2o[1] = sh2o[1] + sice[1]
        sice[1] = _ZERO

    if col.isnow == 0:                                            # :2548
        return ponding1, ponding2

    col.sneqv = _ZERO                                             # :2550-2560
    col.snowh = _ZERO
    zwice = _ZERO
    zwliq = _ZERO
    for j in range(col.isnow + 1, 1):
        col.sneqv = col.sneqv + SNICE[j] + SNLIQ[j]
        col.snowh = col.snowh + DZSNSO[j]
        zwice = zwice + SNICE[j]
        zwliq = zwliq + SNLIQ[j]

    # all snow gone -- the glacier variant keeps the 0.05 threshold
    if col.snowh < _f(0.05) and col.isnow < 0:                    # :2566-2571
        col.isnow = 0
        col.sneqv = zwice
        ponding2 = ponding2 + zwliq
        if col.sneqv <= _ZERO:
            col.snowh = _ZERO

    if col.isnow < -1:                                            # :2582-2632
        isnow_old = col.isnow
        mssi = 1
        for i in range(isnow_old + 1, 1):
            if DZSNSO[i] < _DZMIN[mssi - 1]:
                if i == col.isnow + 1:
                    neibor = i + 1
                elif i == 0:
                    neibor = i - 1
                else:
                    neibor = i + 1
                    if (DZSNSO[i - 1] + DZSNSO[i]) < (DZSNSO[i + 1]
                                                      + DZSNSO[i]):
                        neibor = i - 1

                if neibor > i:                                    # :2600-2606
                    j, l = neibor, i
                else:
                    j, l = i, neibor

                dzc, wliqc, wicec, tc = combo(
                    DZSNSO[j], SNLIQ[j], SNICE[j], stc[j],
                    DZSNSO[l], SNLIQ[l], SNICE[l], stc[l])        # :2608-2609
                DZSNSO[j] = dzc
                SNLIQ[j] = wliqc
                SNICE[j] = wicec
                stc[j] = tc

                if j - 1 > col.isnow + 1:                         # :2612-2619
                    for k in range(j - 1, col.isnow + 1, -1):
                        stc[k] = stc[k - 1]
                        SNICE[k] = SNICE[k - 1]
                        SNLIQ[k] = SNLIQ[k - 1]
                        DZSNSO[k] = DZSNSO[k - 1]

                col.isnow = col.isnow + 1                         # :2622
                if col.isnow >= -1:
                    break
            else:
                mssi = mssi + 1                                   # :2627

    return ponding1, ponding2


# ===========================================================================
# DIVIDE_GLACIER -- :2689-2812
# ===========================================================================

def _divide_glacier(col: SnowColumn, stc) -> None:
    """Subdivide thick snow layers; glacier keeps the 0.10 m layer-2
    subdivision threshold (:2763)."""
    nsnow = col.nsnow
    SNICE, SNLIQ, DZSNSO = col.SNICE, col.SNLIQ, col.DZSNSO

    dz = np.zeros(nsnow, dtype=np.float32)                        # 1-based
    swice = np.zeros(nsnow, dtype=np.float32)
    swliq = np.zeros(nsnow, dtype=np.float32)
    tsno = np.zeros(nsnow, dtype=np.float32)

    for j in range(1, nsnow + 1):                                 # :2722-2729
        if j <= abs(col.isnow):
            dz[j - 1] = DZSNSO[j + col.isnow]
            swice[j - 1] = SNICE[j + col.isnow]
            swliq[j - 1] = SNLIQ[j + col.isnow]
            tsno[j - 1] = stc[j + col.isnow]

    msno = abs(col.isnow)                                         # :2731

    if msno == 1:                                                 # :2733-2745
        if dz[0] > _f(0.05):
            msno = 2
            dz[0] = dz[0] / _TWO
            swice[0] = swice[0] / _TWO
            swliq[0] = swliq[0] / _TWO
            dz[1] = dz[0]
            swice[1] = swice[0]
            swliq[1] = swliq[0]
            tsno[1] = tsno[0]

    if msno > 1:                                                  # :2747-2781
        if dz[0] > _f(0.05):
            drr = dz[0] - _f(0.05)
            propor = drr / dz[0]
            zwice = propor * swice[0]
            zwliq = propor * swliq[0]
            propor = _f(0.05) / dz[0]
            swice[0] = propor * swice[0]
            swliq[0] = propor * swliq[0]
            dz[0] = _f(0.05)

            dzc, wliqc, wicec, tc = combo(dz[1], swliq[1], swice[1],
                                          tsno[1], drr, zwliq, zwice,
                                          tsno[0])                # :2758-2759
            dz[1], swliq[1], swice[1], tsno[1] = dzc, wliqc, wicec, tc

            # glacier keeps DZ(2) > 0.10 (:2763)
            if msno <= 2 and dz[1] > _f(0.10):
                msno = 3
                dtdz = (tsno[0] - tsno[1]) / ((dz[0] + dz[1]) / _TWO)
                dz[1] = dz[1] / _TWO
                swice[1] = swice[1] / _TWO
                swliq[1] = swliq[1] / _TWO
                dz[2] = dz[1]
                swice[2] = swice[1]
                swliq[2] = swliq[1]
                tsno[2] = tsno[1] - dtdz * dz[1] / _TWO
                if tsno[2] >= TFRZ:
                    tsno[2] = tsno[1]
                else:
                    tsno[1] = tsno[1] + dtdz * dz[1] / _TWO

    if msno > 2:                                                  # :2783-2796
        if dz[1] > _f(0.2):
            drr = dz[1] - _f(0.2)
            propor = drr / dz[1]
            zwice = propor * swice[1]
            zwliq = propor * swliq[1]
            propor = _f(0.2) / dz[1]
            swice[1] = propor * swice[1]
            swliq[1] = propor * swliq[1]
            dz[1] = _f(0.2)
            dzc, wliqc, wicec, tc = combo(dz[2], swliq[2], swice[2],
                                          tsno[2], drr, zwliq, zwice,
                                          tsno[1])                # :2793-2794
            dz[2], swliq[2], swice[2], tsno[2] = dzc, wliqc, wicec, tc

    col.isnow = -msno                                             # :2798

    for j in range(col.isnow + 1, 1):                             # :2800-2805
        DZSNSO[j] = dz[j - col.isnow - 1]
        SNICE[j] = swice[j - col.isnow - 1]
        SNLIQ[j] = swliq[j - col.isnow - 1]
        stc[j] = tsno[j - col.isnow - 1]


# ===========================================================================
# SNOWH2O_GLACIER -- :2814-2971
# ===========================================================================

def _snowh2o_glacier(col: SnowColumn, stc, dt, qsnfro, qsnsub, qrain,
                     sh2o, sice, ponding1, ponding2):
    """Snowpack hydrology; returns ``(qsnbot, ponding1, ponding2)``.
    The OPT_GLA == 2 arms (:2871-2874, :2889-2892) are dead."""
    dt = _f(dt)
    qsnfro, qsnsub, qrain = _f(qsnfro), _f(qsnsub), _f(qrain)
    SNICE, SNLIQ, DZSNSO = col.SNICE, col.SNLIQ, col.DZSNSO
    nsnow = col.nsnow

    if col.sneqv == _ZERO:                                        # :2868-2876
        # OPT_GLA == 1
        sice[1] = sice[1] + (qsnfro - qsnsub) * dt / (DZSNSO[1] * _f(1000.0))

    if col.isnow == 0 and col.sneqv > _ZERO:                      # :2883-2904
        # OPT_GLA == 1
        temp = col.sneqv
        col.sneqv = col.sneqv - qsnsub * dt + qsnfro * dt
        propor = col.sneqv / temp
        col.snowh = _fmax(_ZERO, propor * col.snowh)

        if col.sneqv < _ZERO:                                     # :2895-2899
            sice[1] = sice[1] + col.sneqv / (DZSNSO[1] * _f(1000.0))
            col.sneqv = _ZERO
            col.snowh = _ZERO
        if sice[1] < _ZERO:                                       # :2900-2903
            sh2o[1] = sh2o[1] + sice[1]
            sice[1] = _ZERO

    if col.snowh <= _f(1.0e-8) or col.sneqv <= _f(1.0e-6):        # :2906-2909
        col.snowh = _ZERO
        col.sneqv = _ZERO

    if col.isnow < 0:                                             # :2913-2929
        wgdif = SNICE[col.isnow + 1] - qsnsub * dt + qsnfro * dt
        SNICE[col.isnow + 1] = wgdif
        if wgdif < _f(1.0e-6) and col.isnow < 0:
            ponding1, ponding2 = _combine_glacier(
                col, stc, sh2o, sice, ponding1, ponding2)         # :2918-2921
        if col.isnow < 0:
            SNLIQ[col.isnow + 1] = SNLIQ[col.isnow + 1] + qrain * dt
            SNLIQ[col.isnow + 1] = _fmax(_ZERO, SNLIQ[col.isnow + 1])

    # porosity and partial volume :2935-2941
    vol_ice = _Col(np.zeros(nsnow, dtype=np.float32), -nsnow + 1)
    epore = _Col(np.zeros(nsnow, dtype=np.float32), -nsnow + 1)
    vol_liq = _Col(np.zeros(nsnow, dtype=np.float32), -nsnow + 1)
    for j in range(-nsnow + 1, 1):
        if j >= col.isnow + 1:
            vol_ice[j] = _fmin(_ONE, SNICE[j] / (DZSNSO[j] * DENICE))
            epore[j] = _ONE - vol_ice[j]
            vol_liq[j] = _fmin(epore[j], SNLIQ[j] / (DZSNSO[j] * DENH2O))

    qin = _ZERO                                                   # :2943-2944
    qout = _ZERO

    for j in range(-nsnow + 1, 1):                                # :2948-2965
        if j >= col.isnow + 1:
            SNLIQ[j] = SNLIQ[j] + qin
            if j <= -1:
                if epore[j] < _f(0.05) or epore[j + 1] < _f(0.05):
                    qout = _ZERO
                else:
                    qout = _fmax(_ZERO, (vol_liq[j] - SSI * epore[j])
                                 * DZSNSO[j])
                    qout = _fmin(qout, (_ONE - vol_ice[j + 1]
                                        - vol_liq[j + 1]) * DZSNSO[j + 1])
            else:
                qout = _fmax(_ZERO, (vol_liq[j] - SSI * epore[j])
                             * DZSNSO[j])
            qout = qout * _f(1000.0)
            SNLIQ[j] = SNLIQ[j] - qout
            qin = qout

    qsnbot = qout / dt                                            # :2969
    return qsnbot, ponding1, ponding2


# ===========================================================================
# SNOWWATER_GLACIER -- :2174-2300
# ===========================================================================

def _snowwater_glacier(col: SnowColumn, stc, dt, sfctmp, snowhin, qsnow,
                       qsnfro, qsnsub, qrain, ficeold, imelt, zsoil,
                       sh2o, sice):
    """Snowpack driver; returns ``(qsnbot, snoflow, ponding1, ponding2)``."""
    nsnow, nsoil = col.nsnow, col.nsoil
    SNICE, SNLIQ, DZSNSO, ZSNSO = (col.SNICE, col.SNLIQ, col.DZSNSO,
                                   col.ZSNSO)
    snoflow = _ZERO                                               # :2221-2223
    ponding1 = _ZERO
    ponding2 = _ZERO

    _snowfall_glacier(col, dt, qsnow, snowhin, sfctmp)            # :2225-2228

    if col.isnow < 0:                                             # :2230-2242
        _compact_glacier(col, dt, stc, imelt, ficeold)
        ponding1, ponding2 = _combine_glacier(
            col, stc, sh2o, sice, ponding1, ponding2)
        _divide_glacier(col, stc)

    for iz in range(-nsnow + 1, col.isnow + 1):                   # :2246-2252
        SNICE[iz] = _ZERO
        SNLIQ[iz] = _ZERO
        stc[iz] = _ZERO
        DZSNSO[iz] = _ZERO
        ZSNSO[iz] = _ZERO

    qsnbot, ponding1, ponding2 = _snowh2o_glacier(
        col, stc, dt, qsnfro, qsnsub, qrain, sh2o, sice,
        ponding1, ponding2)                                       # :2254-2259

    if col.sneqv > _f(5000.0):                                    # :2263-2269
        bdsnow = SNICE[0] / DZSNSO[0]
        snoflow = col.sneqv - _f(5000.0)
        SNICE[0] = SNICE[0] - snoflow
        DZSNSO[0] = DZSNSO[0] - snoflow / bdsnow
        snoflow = snoflow / _f(dt)

    if col.isnow != 0:                                            # :2273-2278
        col.sneqv = _ZERO
        for iz in range(col.isnow + 1, 1):
            col.sneqv = col.sneqv + SNICE[iz] + SNLIQ[iz]

    # reset ZSNSO and DZSNSO :2282-2298
    for iz in range(col.isnow + 1, 1):
        DZSNSO[iz] = -DZSNSO[iz]
    DZSNSO[1] = _f(zsoil[0])
    for iz in range(2, nsoil + 1):
        DZSNSO[iz] = _f(zsoil[iz - 1]) - _f(zsoil[iz - 2])
    ZSNSO[col.isnow + 1] = DZSNSO[col.isnow + 1]
    for iz in range(col.isnow + 2, nsoil + 1):
        ZSNSO[iz] = ZSNSO[iz - 1] + DZSNSO[iz]
    for iz in range(col.isnow + 1, nsoil + 1):
        DZSNSO[iz] = -DZSNSO[iz]

    return qsnbot, snoflow, ponding1, ponding2


# ===========================================================================
# WATER_GLACIER -- :1997-2171
# ===========================================================================

def _water_glacier(col: SnowColumn, stc, *, dt, prcp, sfctmp, qvap, qdew,
                   ficeold, zsoil, imelt, ponding, sh2o, sice):
    """Water budget; mutates ``col``/``stc``/``sh2o``/``sice``.
    Returns ``(runsrf, runsub, qsnow, ponding1, ponding2, qsnbot,
    fpice)``."""
    dt, prcp, sfctmp = _f(dt), _f(prcp), _f(sfctmp)
    nsoil = col.nsoil
    DZSNSO = col.DZSNSO

    runsub = _ZERO                                                # :2068-2070
    runsrf = _ZERO
    sice_save = sice.data.copy()                                  # :2071-2072
    sh2o_save = sh2o.data.copy()

    # OPT_SNF == 1: Jordan (1991) :2079-2091
    if sfctmp > TFRZ + _f(2.5):
        fpice = _ZERO
    else:
        if sfctmp <= TFRZ + _f(0.5):
            fpice = _ONE
        elif sfctmp <= TFRZ + _TWO:
            fpice = _ONE - (_f(-54.632) + _f(0.2) * sfctmp)
        else:
            fpice = _f(0.6)

    bdfall = _fmin(_f(120.0), _f(67.92) + _f(51.25)
                   * expf((sfctmp - TFRZ) / _f(2.59)))            # :2113

    qrain = prcp * (_ONE - fpice)                                 # :2115-2117
    qsnow = prcp * fpice
    snowhin = qsnow / bdfall

    qsnsub = qvap                                                 # :2122-2123
    qsnfro = qdew

    qsnbot, snoflow, ponding1, ponding2 = _snowwater_glacier(
        col, stc, dt, sfctmp, snowhin, qsnow, qsnfro, qsnsub, qrain,
        ficeold, imelt, zsoil, sh2o, sice)                        # :2125-2131

    runsrf = (ponding + ponding1 + ponding2) / dt                 # :2135

    if col.isnow == 0:                                            # :2137-2141
        runsrf = runsrf + qsnbot + qrain
    else:
        runsrf = runsrf + qsnbot

    # OPT_GLA == 1 :2147-2154
    replace = _ZERO
    for ilev in range(1, nsoil + 1):
        replace = replace + DZSNSO[ilev] * (sice[ilev] - sice_save[ilev - 1]
                                            + sh2o[ilev]
                                            - sh2o_save[ilev - 1])
    replace = replace * _f(1000.0) / dt

    for ilev in range(1, nsoil + 1):
        sice[ilev] = _fmin(_ONE, sice_save[ilev - 1])
    for ilev in range(1, nsoil + 1):                              # :2158
        sh2o[ilev] = _ONE - sice[ilev]

    runsub = snoflow + replace                                    # :2163-2164

    return runsrf, runsub, qsnow, ponding1, ponding2, qsnbot, fpice


# ===========================================================================
# ERROR_GLACIER -- :2974-3048
# ===========================================================================

def _error_glacier(swdown, fsa, fsr, fira, fsh, fgev, ssoil, sag, prcp,
                   edir, runsrf, runsub, sneqv, dt, beg_wb, iloc, jloc):
    """Balance checks; raises :class:`GlacierBalanceError` where WRF
    calls ``wrf_error_fatal``."""
    errsw = _f(swdown) - (_f(fsa) + _f(fsr))                      # :3008
    if errsw > _f(0.01):
        raise GlacierBalanceError(
            f"NOAHMP_GLACIER radiation budget problem at (j={jloc}, "
            f"i={iloc}): ERRSW={float(errsw)} W m-2 "
            "(module_sf_noahmp_glacier.F:3009-3016)")

    erreng = _f(sag) - (_f(fira) + _f(fsh) + _f(fgev) + _f(ssoil))
    if erreng > _f(0.01):                                         # :3018-3025
        raise GlacierBalanceError(
            f"NOAHMP_GLACIER energy budget problem at (j={jloc}, "
            f"i={iloc}): ERRENG={float(erreng)} W m-2")

    end_wb = _f(sneqv)                                            # :3027-3028
    errwat = (end_wb - _f(beg_wb)
              - (_f(prcp) - _f(edir) - _f(runsrf) - _f(runsub)) * _f(dt))
    if abs(errwat) > _f(0.1):                                     # :3031-3045
        raise GlacierBalanceError(
            f"NOAHMP_GLACIER water budget problem at (j={jloc}, "
            f"i={iloc}): ERRWAT={float(errwat)} mm per timestep")


# ===========================================================================
# NOAHMP_GLACIER -- :105-297
# ===========================================================================

@dataclass
class GlacierResult:
    """Everything the driver's glacier arm reads back: the INOUT state
    plus the OUT diagnostics, in the driver's own spellings."""

    # prognostic / INOUT
    isnow: int
    sneqv: np.float32
    snowh: np.float32
    smc: np.ndarray
    sh2o: np.ndarray
    stc: np.ndarray
    zsnso: np.ndarray
    snice: np.ndarray
    snliq: np.ndarray
    tg: np.float32
    tauss: np.float32
    qsfc: np.float32
    qsnow: np.float32
    sneqvo: np.float32
    albold: np.float32
    cm: np.float32
    ch: np.float32
    # OUT
    fsa: np.float32
    fsr: np.float32
    fira: np.float32
    fsh: np.float32
    fgev: np.float32
    ssoil: np.float32
    trad: np.float32
    edir: np.float32
    runsrf: np.float32
    runsub: np.float32
    sag: np.float32
    albedo: np.float32
    qsnbot: np.float32
    ponding: np.float32
    ponding1: np.float32
    ponding2: np.float32
    t2m: np.float32
    q2e: np.float32
    emissi: np.float32
    fpice: np.float32
    ch2b: np.float32
    qmelt: np.float32
    # defined replacements for WRF's undefined reads (module docstring)
    hcpct: np.ndarray
    eflxb: np.float32


def noahmp_glacier(*, iloc=0, jloc=0, cosz, nsnow, nsoil, dt, sfctmp,
                   sfcprs, uu, vv, q2, soldn, prcp, lwdn, tbot, zlvl,
                   ficeold, zsoil, qsnow, sneqvo, albold, cm, ch, isnow,
                   sneqv, smc, zsnso, snowh, snice, snliq, tg, stc, sh2o,
                   tauss, qsfc, opt_alb=2, opt_snf=1, opt_tbot=2,
                   opt_stc=1, opt_gla=1) -> GlacierResult:
    """One ``NOAHMP_GLACIER`` column call (:105-297).

    Array arguments are plain 0-based sequences: ``ficeold``/``snice``/
    ``snliq`` of length ``nsnow`` (layers ``-NSNOW+1..0``), ``zsoil``/
    ``smc``/``sh2o`` of length ``nsoil``, ``zsnso``/``stc`` of length
    ``nsnow + nsoil``.  Inputs are not mutated; the returned
    :class:`GlacierResult` carries the advanced state.
    """
    got = {"opt_alb": int(opt_alb), "opt_snf": int(opt_snf),
           "opt_tbot": int(opt_tbot), "opt_stc": int(opt_stc),
           "opt_gla": int(opt_gla)}
    if got != GLACIER_OPTION_IDENTITY:
        raise NotImplementedError(
            f"NOAHMP_GLACIER is transcribed at {GLACIER_OPTION_IDENTITY} "
            f"only, got {got}; the other arms are dead under the pinned "
            "identity and deliberately not ported")
    if int(nsnow) != 3 or int(nsoil) != 4:
        raise ValueError(
            f"NOAHMP_GLACIER is pinned at nsnow=3, nsoil=4, got "
            f"{nsnow}/{nsoil}")

    nsnow, nsoil = int(nsnow), int(nsoil)
    ficeold = np.asarray(ficeold, dtype=np.float32).copy()
    zsoil = np.asarray(zsoil, dtype=np.float32).copy()
    smc = np.asarray(smc, dtype=np.float32).copy()

    # ZSNSO carries negative interface depths; the working column carries
    # positive thicknesses.  Build DZSNSO exactly as :229-235 does.
    col = SnowColumn(
        nsnow=nsnow, nsoil=nsoil, isnow=int(isnow), snowh=_f(snowh),
        sneqv=_f(sneqv), snice=np.asarray(snice, dtype=np.float32),
        snliq=np.asarray(snliq, dtype=np.float32),
        stc=np.asarray(stc, dtype=np.float32),
        zsnso=np.asarray(zsnso, dtype=np.float32), sh2o=sh2o,
        sice=np.zeros(nsoil, dtype=np.float32))
    # SnowColumn copies its arrays; from here on the column's buffers ARE
    # the working state, so rebind the locals to them.
    sh2o = col.sh2o
    STC = col.STC
    ZSNSO = col.ZSNSO
    DZSNSO = col.DZSNSO
    FICEOLD = _Col(ficeold, -nsnow + 1)
    smc_v = _Col(smc, 1)
    sh2o_v = col.SH2O
    sice_v = col.SICE

    _qair, eair, rhoair, solad, solai, swdown = _atm_glacier(
        sfcprs, sfctmp, q2, soldn, cosz)                          # :222-223

    beg_wb = _f(sneqv)                                            # :225

    for iz in range(col.isnow + 1, nsoil + 1):                    # :229-235
        if iz == col.isnow + 1:
            DZSNSO[iz] = -ZSNSO[iz]
        else:
            DZSNSO[iz] = ZSNSO[iz - 1] - ZSNSO[iz]

    energy = _energy_glacier(
        col, dt=dt, qsnow=qsnow, rhoair=rhoair, eair=eair, sfcprs=sfcprs,
        qair=_qair, sfctmp=sfctmp, lwdn=lwdn, uu=uu, vv=vv, solad=solad,
        solai=solai, cosz=cosz, zref=zlvl, tbot=tbot, stc=STC, smc=smc,
        sh2o=sh2o, tg=tg, sneqvo=sneqvo, albold=albold, cm=cm, ch=ch,
        tauss=tauss, qsfc=qsfc)                                   # :239-248

    for j in range(1, nsoil + 1):                                 # :250
        sice_v[j] = _fmax(_ZERO, smc_v[j] - sh2o_v[j])
    sneqvo_out = col.sneqv                                        # :251

    lathea = energy["lathea"]
    fgev = energy["fgev"]
    qvap = _fmax(fgev / lathea, _ZERO)                            # :253
    qdew = abs(_fmin(fgev / lathea, _ZERO))                       # :254
    edir = qvap - qdew                                            # :255

    runsrf, runsub, qsnow_out, ponding1, ponding2, qsnbot, fpice = \
        _water_glacier(col, STC, dt=dt, prcp=prcp, sfctmp=sfctmp,
                       qvap=qvap, qdew=qdew, ficeold=FICEOLD, zsoil=zsoil,
                       imelt=energy["imelt"], ponding=energy["ponding"],
                       sh2o=sh2o_v, sice=sice_v)                  # :259-267

    # :269-277 (OPT_GLA == 2 EDIR recompute; melted-glacier wrf_debug) --
    # dead / no observable effect under the pinned identity.

    _error_glacier(swdown, energy["fsa"], energy["fsr"], energy["fira"],
                   energy["fsh"], fgev, energy["ssoil"], energy["sag"],
                   prcp, edir, runsrf, runsub, col.sneqv, dt, beg_wb,
                   iloc, jloc)                                    # :281-283

    if col.snowh <= _f(1.0e-6) or col.sneqv <= _f(1.0e-3):        # :285-288
        col.snowh = _ZERO
        col.sneqv = _ZERO

    if swdown != _ZERO:                                           # :290-294
        albedo = energy["fsr"] / swdown
    else:
        albedo = _f(-999.9)

    return GlacierResult(
        isnow=col.isnow, sneqv=col.sneqv, snowh=col.snowh,
        smc=smc_v.data.copy(), sh2o=col.sh2o.copy(),
        stc=col.stc.copy(), zsnso=col.zsnso.copy(),
        snice=col.snice.copy(), snliq=col.snliq.copy(),
        tg=energy["tg"], tauss=energy["tauss"], qsfc=energy["qsfc"],
        qsnow=qsnow_out, sneqvo=sneqvo_out, albold=energy["albold"],
        cm=energy["cm"], ch=energy["ch"],
        fsa=energy["fsa"], fsr=energy["fsr"], fira=energy["fira"],
        fsh=energy["fsh"], fgev=fgev, ssoil=energy["ssoil"],
        trad=energy["trad"], edir=edir, runsrf=runsrf, runsub=runsub,
        sag=energy["sag"], albedo=albedo, qsnbot=qsnbot,
        ponding=energy["ponding"], ponding1=ponding1, ponding2=ponding2,
        t2m=energy["t2m"], q2e=energy["q2e"], emissi=energy["emissi"],
        fpice=fpice, ch2b=energy["ch2b"], qmelt=energy["qmelt"],
        hcpct=energy["hcpct"], eflxb=energy["eflxb"])
