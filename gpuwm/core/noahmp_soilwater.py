"""Noah-MP soil-water routines, transcribed from WRF v4.6.1 in FP32.

Ports ``MODULE_SF_NOAHMPLSM``'s CANWATER (6265-6394), SOILWATER (7234-7556),
INFIL (7616-7712), SRT (7716-7846) and SSTEP (7850-7973) from
``phys/module_sf_noahmplsm.F`` at WRF commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``
(``sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282``).

``kind_phys == kind(1.0)`` in that build, so every quantity here is IEEE
binary32 and every arithmetic boundary is pinned to ``numpy.float32``.

``WDFCND1``/``WDFCND2`` and ``ROSR12`` are already pinned in
:mod:`gpuwm.core.noahmp_leaves` and are imported, not re-transcribed.
Transcendentals come from :mod:`gpuwm.core.noahmp_libm`, the verified glibc
transcriptions; neither ``numpy.exp`` on float32 nor "compute in float64 and
round once" reproduces glibc's ``expf``/``powf`` bit for bit.

Pinned option identity
----------------------
``opt_run=3`` (Schaake96), ``opt_inf=1`` (NY06), ``opt_tdrn=0``, ``opt_irr=0``.
Everything those kill is absent here, not stubbed:

* ``opt_run=3`` kills GROUNDWATER, SHALLOWWATERTABLE, ZWTEQ,
  COMPUTE_VIC_SURFRUNOFF, COMPUTE_XAJ_SURFRUNOFF and DYNAMIC_VIC, plus
  SOILWATER's OPT_RUN 1/2/4/5/6/7/8 blocks, SRT's OPT_RUN 1/2/4/5 drainage
  forms and SSTEP's water-table block;
* ``opt_inf=1`` kills SRT's WDFCND2 loop -- WDFCND2 itself stays live through
  INFIL;
* ``opt_tdrn=0`` kills TILE_DRAIN and TILE_HOOGHOUDT;
* ``opt_irr=0`` kills the irrigation routines, none of which this group calls.

Two INTENT(OUT) aliasing hazards, reproduced rather than tidied
---------------------------------------------------------------
* SOILWATER declares ``RUNSUB`` INTENT(OUT) at 7280 but under ``OPT_RUN==3``
  the only statement that touches it is ``RUNSUB = RUNSUB - XS/DT`` at 7549,
  which reads it before it is ever assigned.  gfortran passes scalar dummies by
  reference, so it behaves as INOUT; :func:`soilwater` therefore takes
  ``runsub`` as an argument and returns it.  WATER always enters with 0.0
  (6109), so the forecast path is well defined.
* INFIL declares ``PDDUM`` and ``RUNSRF`` INTENT(OUT) (7638-7639) but assigns
  them only inside ``IF (QINSUR > 0.0)``.  :func:`infil` therefore takes both
  and returns them unchanged on that path.

Layer conventions
-----------------
Soil arrays are 0-based length ``NSOIL`` covering WRF's ``1..NSOIL``.
``dzsnso`` spans WRF's ``-NSNOW+1..NSOIL``; only its soil slots are read by
anything here, and :func:`_soil` is the only place the offset is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.core.noahmp_leaves import rosr12, wdfcnd1, wdfcnd2
from gpuwm.core.noahmp_libm import expf, powf

__all__ = [
    "NSNOW_DEFAULT",
    "NSOIL_DEFAULT",
    "SoilParameters",
    "canwater",
    "infil",
    "srt",
    "sstep",
    "soilwater",
]


def _f(x) -> np.float32:
    return np.float32(x)


NSNOW_DEFAULT = 3
NSOIL_DEFAULT = 4

# --- module_sf_noahmplsm.F lines 207-220 -----------------------------------
TFRZ = _f(273.16)      # freezing/melting point (K)
HSUB = _f(2.8440e06)   # latent heat of sublimation (J/kg)
HVAP = _f(2.5104e06)   # latent heat of vaporization (J/kg)
HFUS = _f(0.3336e06)   # latent heat of fusion (J/kg)
CWAT = _f(4.188e06)    # volumetric heat capacity of water (J/m3/K)
CICE = _f(2.094e06)    # volumetric heat capacity of ice (J/m3/K)
DENH2O = _f(1000.0)    # density of water (kg/m3)
DENICE = _f(917.0)     # density of ice (kg/m3)

_ZERO = _f(0.0)
_ONE = _f(1.0)
_TWO = _f(2.0)


@dataclass
class SoilParameters:
    """The ``noahmp_parameters`` components this group consumes.

    Per-layer entries are 0-based arrays of length ``NSOIL`` covering WRF's
    ``1..NSOIL``.  ``timean`` and ``fsatmx`` are carried because the fixture
    varies them; nothing under ``opt_run=3`` reads either.
    """

    smcmax: np.ndarray
    smcwlt: np.ndarray
    bexp: np.ndarray
    dksat: np.ndarray
    dwsat: np.ndarray
    kdt: np.float32
    frzx: np.float32
    slope: np.float32
    ch2op: np.float32
    urban_flag: bool = False
    timean: np.float32 = _f(0.0)
    fsatmx: np.float32 = _f(0.0)

    def __post_init__(self) -> None:
        for name in ("smcmax", "smcwlt", "bexp", "dksat", "dwsat"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float32))
        for name in ("kdt", "frzx", "slope", "ch2op", "timean", "fsatmx"):
            setattr(self, name, _f(getattr(self, name)))


def _soil(dzsnso, nsnow: int = NSNOW_DEFAULT) -> np.ndarray:
    """WRF's ``DZSNSO(1:NSOIL)`` out of a ``-NSNOW+1:NSOIL`` array.

    Accepts either the full snow+soil stack or a bare soil array, so callers
    that never had snow slots do not have to fabricate them.
    """
    a = np.asarray(dzsnso, dtype=np.float32)
    return a[nsnow:] if a.size > NSOIL_DEFAULT else a


# ---------------------------------------------------------------------------
# CANWATER -- module_sf_noahmplsm.F:6265-6394
# ---------------------------------------------------------------------------

def canwater(parameters, dt, fcev, fctr, elai, esai, fveg, bdfall,
             frozen_canopy, canliq, canice, tv):
    """Canopy hydrology.

    ``VEGTYP`` (6268), ``ILOC`` (6280), ``JLOC`` (6279) and ``TG`` (6289) are
    declared INTENT(IN) and never referenced in the body, so they are not
    arguments here; ``tests/test_noahmp_soilwater.py`` holds that claim against
    ``canw_inert_probe``.

    ``canliq``, ``canice`` and ``tv`` are INTENT(INOUT); the updated values come
    back in the result.  Returns
    ``(canliq, canice, tv, cmc, ecan, etran, fwet,
       qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc)``.
    """
    dt = _f(dt)
    fcev = _f(fcev)
    fctr = _f(fctr)
    elai = _f(elai)
    esai = _f(esai)
    fveg = _f(fveg)
    bdfall = _f(bdfall)
    canliq = _f(canliq)
    canice = _f(canice)
    tv = _f(tv)

    ecan = _ZERO                                                    # :6318

    lsai = _f(elai + esai)
    maxliq = _f(_f(fveg * parameters.ch2op) * lsai)                 # :6323

    if not frozen_canopy:                                           # :6327
        etran = max(_f(fctr / HVAP), _ZERO)                         # :6328
        qevac = max(_f(fcev / HVAP), _ZERO)                         # :6329
        qdewc = abs(min(_f(fcev / HVAP), _ZERO))                    # :6330
        qsubc = _ZERO
        qfroc = _ZERO
    else:                                                           # :6333
        etran = max(_f(fctr / HSUB), _ZERO)                         # :6334
        qevac = _ZERO
        qdewc = _ZERO
        qsubc = max(_f(fcev / HSUB), _ZERO)                         # :6337
        qfroc = abs(min(_f(fcev / HSUB), _ZERO))                    # :6338

    qevac = min(_f(canliq / dt), qevac)                             # :6344
    canliq = max(_ZERO, _f(canliq + _f(_f(qdewc - qevac) * dt)))    # :6345
    if canliq <= _f(1.0e-06):                                       # :6346
        canliq = _ZERO

    # :6351.  `FVEG * 6.6*(0.27+46.0/BDFALL) * (ELAI+ESAI)` is a chain of
    # left-associative multiplications, so FVEG*6.6 rounds before the bracket
    # multiplies in.
    maxsno = _f(_f(_f(fveg * _f(6.6))
                   * _f(_f(0.27) + _f(_f(46.0) / bdfall))) * lsai)

    qsubc = min(_f(canice / dt), qsubc)                             # :6353
    canice = max(_ZERO, _f(canice + _f(_f(qfroc - qsubc) * dt)))    # :6354
    if canice <= _f(1.0e-6):                                        # :6355
        canice = _ZERO

    if canice > _ZERO and canice >= canliq:                         # :6359
        fwet = _f(max(_ZERO, canice) / max(maxsno, _f(1.0e-06)))    # :6360
    else:                                                           # :6361
        fwet = _f(max(_ZERO, canliq) / max(maxliq, _f(1.0e-06)))    # :6362
    fwet = powf(min(fwet, _ONE), _f(0.667))                         # :6364

    qmeltc = _ZERO                                                  # :6368
    qfrzc = _ZERO                                                   # :6369
    cmc = _f(canliq + canice)                                       # :6370

    if canice > _f(1.0e-6) and tv > TFRZ:                           # :6372
        qmeltc = min(_f(canice / dt),
                     _f(_f(_f(_f(_f(tv - TFRZ) * CICE) * canice) / DENICE)
                        / _f(dt * HFUS)))                           # :6373
        canice = max(_ZERO, _f(canice - _f(qmeltc * dt)))           # :6374
        canliq = max(_ZERO, _f(cmc - canice))                       # :6375
        tv = _f(_f(fwet * TFRZ) + _f(_f(_ONE - fwet) * tv))         # :6376

    if canliq > _f(1.0e-6) and tv < TFRZ:                           # :6379
        qfrzc = min(_f(canliq / dt),
                    _f(_f(_f(_f(_f(TFRZ - tv) * CWAT) * canliq) / DENH2O)
                       / _f(dt * HFUS)))                            # :6380
        canliq = max(_ZERO, _f(canliq - _f(qfrzc * dt)))            # :6381
        canice = max(_ZERO, _f(cmc - canliq))                       # :6382
        tv = _f(_f(fwet * TFRZ) + _f(_f(_ONE - fwet) * tv))         # :6383

    cmc = _f(canliq + canice)                                       # :6388
    ecan = _f(_f(_f(qevac + qsubc) - qdewc) - qfroc)                # :6392

    return (canliq, canice, tv, cmc, ecan, etran, fwet,
            qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc)


# ---------------------------------------------------------------------------
# INFIL -- module_sf_noahmplsm.F:7616-7712
# ---------------------------------------------------------------------------

_CVFRZ = 3   # :7652


def infil(parameters, dt, zsoil, sh2o, sice, sicemax, qinsur, pddum, runsrf,
          nsoil: int = NSOIL_DEFAULT):
    """Surface infiltration rate and surface runoff.

    ``pddum`` and ``runsrf`` are INTENT(OUT) in WRF but are assigned only
    inside ``IF (QINSUR > 0.0)`` at 7655, so they are passed in and returned
    unchanged on the other path.  Returns ``(pddum, runsrf)``.
    """
    dt = _f(dt)
    qinsur = _f(qinsur)
    sicemax = _f(sicemax)
    pddum = _f(pddum)
    runsrf = _f(runsrf)
    zsoil = np.asarray(zsoil, dtype=np.float32)
    sh2o = np.asarray(sh2o, dtype=np.float32)
    sice = np.asarray(sice, dtype=np.float32)

    if qinsur > _ZERO:                                              # :7655
        dt1 = _f(dt / _f(86400.0))                                  # :7656
        smcav = _f(parameters.smcmax[0] - parameters.smcwlt[0])     # :7657

        dmax = np.zeros(nsoil, dtype=np.float32)
        dmax[0] = _f(_f(-zsoil[0]) * smcav)                         # :7661
        dice = _f(_f(-zsoil[0]) * sice[0])                          # :7662
        dmax[0] = _f(dmax[0] * _f(_ONE - _f(
            _f(_f(sh2o[0] + sice[0]) - parameters.smcwlt[0]) / smcav)))  # :7663
        dd = dmax[0]                                                # :7665

        for k in range(1, nsoil):                                   # :7667-7672
            dice = _f(dice + _f(_f(zsoil[k - 1] - zsoil[k]) * sice[k]))
            dmax[k] = _f(_f(zsoil[k - 1] - zsoil[k]) * smcav)
            dmax[k] = _f(dmax[k] * _f(_ONE - _f(
                _f(_f(sh2o[k] + sice[k]) - parameters.smcwlt[k]) / smcav)))
            dd = _f(dd + dmax[k])

        val = _f(_ONE - expf(_f(-_f(parameters.kdt * dt1))))        # :7674
        ddt = _f(dd * val)                                          # :7675
        px = max(_ZERO, _f(qinsur * dt))                            # :7676
        infmax = _f(_f(px * _f(ddt / _f(px + ddt))) / dt)           # :7677

        fcr = _ONE                                                  # :7681
        if dice > _f(1.0e-2):                                       # :7682
            acrt = _f(_f(_f(_CVFRZ) * parameters.frzx) / dice)      # :7683
            ssum = _ONE                                             # :7684
            ialp1 = _CVFRZ - 1                                      # :7685
            for j in range(1, ialp1 + 1):                           # :7686-7692
                kk = 1
                for jj in range(j + 1, ialp1 + 1):
                    kk = kk * jj
                # ACRT**(CVFRZ-J) has an INTEGER exponent, so gfortran expands
                # it to repeated multiplication rather than calling powf.
                ssum = _f(ssum + _f(_ipow(acrt, _CVFRZ - j) / _f(float(kk))))
            fcr = _f(_ONE - _f(expf(_f(-acrt)) * ssum))             # :7693

        infmax = _f(infmax * fcr)                                   # :7698

        _wdf, wcnd = wdfcnd2(sh2o[0], sicemax, parameters.smcmax[0],
                             parameters.bexp[0], parameters.dwsat[0],
                             parameters.dksat[0])                   # :7703
        infmax = max(infmax, wcnd)                                  # :7704
        infmax = min(infmax, _f(px / dt))                           # :7705

        runsrf = max(_ZERO, _f(qinsur - infmax))                    # :7707
        pddum = _f(qinsur - runsrf)                                 # :7708

    return pddum, runsrf


def _ipow(x: np.float32, n: int) -> np.float32:
    """``x**n`` for a literal INTEGER ``n``, the way gfortran expands it.

    ``CVFRZ`` is 3 and ``J`` runs 1..2, so ``n`` is only ever 2 or 1 here; both
    are expanded to multiplications by ``__builtin_powi``, not routed through
    ``powf``.  Written generally so the shape is auditable, and exercised at
    both values by the fixture.
    """
    r = _ONE
    b = _f(x)
    while n > 0:
        if n & 1:
            r = _f(r * b)
        n >>= 1
        if n:
            b = _f(b * b)
    return r


# ---------------------------------------------------------------------------
# SRT -- module_sf_noahmplsm.F:7716-7846
# ---------------------------------------------------------------------------

def srt(parameters, zsoil, pddum, etrani, qseva, smc, fcr,
        nsoil: int = NSOIL_DEFAULT):
    """Right-hand side and tridiagonal coefficients of the Richards equation.

    ``DT`` (7739), ``ILOC`` (7731) and ``JLOC`` (7732) are declared and never
    referenced.  ``SH2O`` and ``SICEMAX`` are read only in the ``OPT_INF==2``
    loop, ``ZWT`` and ``SMCWTD`` only under ``OPT_RUN==5``, ``FCRMAX`` only
    under ``OPT_RUN==4``; none of them is an argument here.
    ``srt_inert_probe`` holds all eight of those claims.

    Returns ``(rhstt, ai, bi, ci, qdrain, wcnd)``.
    """
    zsoil = np.asarray(zsoil, dtype=np.float32)
    etrani = np.asarray(etrani, dtype=np.float32)
    smc = np.asarray(smc, dtype=np.float32)
    fcr = np.asarray(fcr, dtype=np.float32)
    pddum = _f(pddum)
    qseva = _f(qseva)

    wdf = np.zeros(nsoil, dtype=np.float32)
    wcnd = np.zeros(nsoil, dtype=np.float32)
    smx = np.zeros(nsoil, dtype=np.float32)
    for k in range(nsoil):                                          # :7775-7778
        wdf[k], wcnd[k] = wdfcnd1(smc[k], fcr[k], parameters.smcmax[k],
                                  parameters.bexp[k], parameters.dwsat[k],
                                  parameters.dksat[k])
        smx[k] = smc[k]

    denom = np.zeros(nsoil, dtype=np.float32)
    ddz = np.zeros(nsoil, dtype=np.float32)
    dsmdz = np.zeros(nsoil, dtype=np.float32)
    wflux = np.zeros(nsoil, dtype=np.float32)
    qdrain = _ZERO

    for k in range(nsoil):                                          # :7792-7825
        if k == 0:                                                  # :7793
            denom[k] = _f(-zsoil[k])
            temp1 = _f(-zsoil[k + 1])
            ddz[k] = _f(_TWO / temp1)
            dsmdz[k] = _f(_f(_TWO * _f(smx[k] - smx[k + 1])) / temp1)
            wflux[k] = _f(_f(_f(_f(_f(wdf[k] * dsmdz[k]) + wcnd[k]) - pddum)
                             + etrani[k]) + qseva)
        elif k < nsoil - 1:                                         # :7799
            denom[k] = _f(zsoil[k - 1] - zsoil[k])
            temp1 = _f(zsoil[k - 1] - zsoil[k + 1])
            ddz[k] = _f(_TWO / temp1)
            dsmdz[k] = _f(_f(_TWO * _f(smx[k] - smx[k + 1])) / temp1)
            wflux[k] = _f(_f(_f(_f(_f(wdf[k] * dsmdz[k]) + wcnd[k])
                                - _f(wdf[k - 1] * dsmdz[k - 1])) - wcnd[k - 1])
                          + etrani[k])
        else:                                                       # :7805
            denom[k] = _f(zsoil[k - 1] - zsoil[k])
            # OPT_RUN==3: QDRAIN = SLOPE*WCND(NSOIL) at 7807.  The OPT_RUN
            # 1/2/4/5 forms at 7803-7818 are dead.
            qdrain = _f(parameters.slope * wcnd[k])
            wflux[k] = _f(_f(_f(_f(-_f(wdf[k - 1] * dsmdz[k - 1]))
                                - wcnd[k - 1]) + etrani[k]) + qdrain)

    ai = np.zeros(nsoil, dtype=np.float32)
    bi = np.zeros(nsoil, dtype=np.float32)
    ci = np.zeros(nsoil, dtype=np.float32)
    rhstt = np.zeros(nsoil, dtype=np.float32)
    for k in range(nsoil):                                          # :7828-7844
        if k == 0:                                                  # :7829
            ai[k] = _ZERO
            bi[k] = _f(_f(wdf[k] * ddz[k]) / denom[k])
            ci[k] = _f(-bi[k])
        elif k < nsoil - 1:                                         # :7833
            ai[k] = _f(-_f(_f(wdf[k - 1] * ddz[k - 1]) / denom[k]))
            ci[k] = _f(-_f(_f(wdf[k] * ddz[k]) / denom[k]))
            bi[k] = _f(-_f(ai[k] + ci[k]))
        else:                                                       # :7837
            ai[k] = _f(-_f(_f(wdf[k - 1] * ddz[k - 1]) / denom[k]))
            ci[k] = _ZERO
            bi[k] = _f(-_f(ai[k] + ci[k]))
        rhstt[k] = _f(wflux[k] / _f(-denom[k]))                     # :7843

    return rhstt, ai, bi, ci, qdrain, wcnd


# ---------------------------------------------------------------------------
# SSTEP -- module_sf_noahmplsm.F:7850-7973
# ---------------------------------------------------------------------------

def sstep(parameters, dt, dzsnso, sice, sh2o, ai, bi, ci, rhstt,
          nsoil: int = NSOIL_DEFAULT, nsnow: int = NSNOW_DEFAULT):
    """Advance soil moisture one fine step and redistribute saturation excess.

    ``ILOC``/``JLOC`` (7859-7860), ``ZSOIL`` and ``ZWT`` (both OPT_RUN==5 only
    at 7927) are not arguments; nor are ``SMCWTD``/``QDRAIN``/``DEEPRECH``,
    which SSTEP writes only inside the OPT_RUN==5 block and which the caller
    therefore keeps unchanged.  ``SMC`` on entry is unconditionally overwritten
    by ``SMC = SH2O + SICE`` at 7971, so it is not an argument either.
    ``sstep_inert_probe`` holds all of that.

    ``sh2o``, ``ai``, ``bi``, ``ci`` and ``rhstt`` are INTENT(INOUT); the
    updated values come back.  Returns ``(sh2o, smc, ai, bi, ci, rhstt, wplus)``.
    """
    dt = _f(dt)
    dz = _soil(dzsnso, nsnow)
    sice = np.asarray(sice, dtype=np.float32)
    sh2o = np.asarray(sh2o, dtype=np.float32).copy()
    ai = np.asarray(ai, dtype=np.float32).copy()
    bi = np.asarray(bi, dtype=np.float32).copy()
    ci = np.asarray(ci, dtype=np.float32).copy()
    rhstt = np.asarray(rhstt, dtype=np.float32).copy()

    wplus = _ZERO                                                   # :7894

    for k in range(nsoil):                                          # :7896-7901
        rhstt[k] = _f(rhstt[k] * dt)
        ai[k] = _f(ai[k] * dt)
        bi[k] = _f(_ONE + _f(bi[k] * dt))
        ci[k] = _f(ci[k] * dt)

    rhsttin = rhstt.copy()                                          # :7906-7909
    ciin = ci.copy()

    # CALL ROSR12 (CI,AI,BI,CIIN,RHSTTIN,RHSTT,1,NSOIL,0) at 7913.  NSNOW is 0
    # here, so the shared transcription's slot map is the identity.
    ci, rhstt, _ciin = rosr12(ai, bi, ciin, rhsttin, 1, nsoil=nsoil, nsnow=0)

    for k in range(nsoil):                                          # :7915-7917
        sh2o[k] = _f(sh2o[k] + ci[k])

    # The OPT_RUN==5 block at 7923-7947 is dead under opt_run=3.

    for k in range(nsoil - 1, 0, -1):                               # :7951-7956
        epore = max(_f(1.0e-4), _f(parameters.smcmax[k] - sice[k]))
        wplus = _f(max(_f(sh2o[k] - epore), _ZERO) * dz[k])
        sh2o[k] = min(epore, sh2o[k])
        sh2o[k - 1] = _f(sh2o[k - 1] + _f(wplus / dz[k - 1]))

    epore = max(_f(1.0e-4), _f(parameters.smcmax[0] - sice[0]))     # :7958
    wplus = _f(max(_f(sh2o[0] - epore), _ZERO) * dz[0])             # :7959
    sh2o[0] = min(epore, sh2o[0])                                   # :7960

    if wplus > _ZERO:                                               # :7962
        sh2o[1] = _f(sh2o[1] + _f(wplus / dz[1]))                   # :7963
        for k in range(1, nsoil - 1):                               # :7964-7969
            epore = max(_f(1.0e-4), _f(parameters.smcmax[k] - sice[k]))
            wplus = _f(max(_f(sh2o[k] - epore), _ZERO) * dz[k])
            sh2o[k] = min(epore, sh2o[k])
            sh2o[k + 1] = _f(sh2o[k + 1] + _f(wplus / dz[k + 1]))

        epore = max(_f(1.0e-4),
                    _f(parameters.smcmax[nsoil - 1] - sice[nsoil - 1]))
        wplus = _f(max(_f(sh2o[nsoil - 1] - epore), _ZERO) * dz[nsoil - 1])
        sh2o[nsoil - 1] = min(epore, sh2o[nsoil - 1])               # :7971

    smc = np.asarray([_f(sh2o[k] + sice[k]) for k in range(nsoil)],
                     dtype=np.float32)                              # :7974

    return sh2o, smc, ai, bi, ci, rhstt, wplus


# ---------------------------------------------------------------------------
# SOILWATER -- module_sf_noahmplsm.F:7234-7556
# ---------------------------------------------------------------------------

_A = _f(4.0)   # :7317, the frozen-fraction decay constant


def soilwater(parameters, dt, zsoil, dzsnso, qinsur, qseva, etrani, sice,
              sh2o, smc, runsub, nsoil: int = NSOIL_DEFAULT,
              nsnow: int = NSNOW_DEFAULT):
    """Surface runoff and the soil-moisture update, ``opt_run=3``.

    ``ILOC``/``JLOC`` (7253-7254), ``VEGTYP`` (7268), ``DX`` (7267) and
    ``TDFRACMP`` (7266) are not arguments: the first two are only forwarded to
    SRT/SSTEP where they are inert too, ``DX`` is read only by TILE_HOOGHOUDT
    and ``TDFRACMP`` only in the OPT_TDRN gates.  ``ZWT``, ``SMCWTD``,
    ``DEEPRECH`` and ``QTLDRN`` are INOUT arguments that no ``opt_run=3``
    statement writes, so the caller keeps them; ``slw_inert_probe`` holds all
    nine claims.

    ``SMC`` on entry is never read -- every path reaches ``SMC = SH2O + SICE``
    inside SSTEP -- so it is not an argument either.

    ``smc`` **is** read: the first SRT call of the iteration loop sees the
    caller's entry array, and only later passes see SSTEP's ``SH2O + SICE``.

    ``runsub`` is passed in because SOILWATER reads it before assigning it at
    7549; see the module docstring.  Returns
    ``(sh2o, smc, runsrf, qdrain, runsub, wcnd, fcrmax)``.

    The returned ``smc`` is SSTEP's, **not** ``sh2o + sice``: the WATMIN fixup
    at 7546-7574 rewrites ``SH2O`` and never touches ``SMC``, so on exit the
    two are inconsistent by exactly that correction.  The fixture pins the
    difference -- ``slw_moderate_rain`` returns ``sh2o[1]=0x3E65569F`` against
    ``smc[1]=0x3E65569E``.
    """
    dt = _f(dt)
    qinsur = _f(qinsur)
    qseva = _f(qseva)
    runsub = _f(runsub)
    zsoil = np.asarray(zsoil, dtype=np.float32)
    dz = _soil(dzsnso, nsnow)
    etrani = np.asarray(etrani, dtype=np.float32)
    sice = np.asarray(sice, dtype=np.float32)
    sh2o = np.asarray(sh2o, dtype=np.float32).copy()
    smc = np.asarray(smc, dtype=np.float32).copy()

    runsrf = _ZERO                                                  # :7318
    pddum = _ZERO                                                   # :7319
    rsat = _ZERO                                                    # :7320

    for k in range(nsoil):                                          # :7324-7328
        epore = max(_f(1.0e-4), _f(parameters.smcmax[k] - sice[k]))
        rsat = _f(rsat + _f(max(_ZERO, _f(sh2o[k] - epore)) * dz[k]))
        sh2o[k] = min(epore, sh2o[k])

    fcr = np.zeros(nsoil, dtype=np.float32)
    for k in range(nsoil):                                          # :7333-7336
        fice = min(_ONE, _f(sice[k] / parameters.smcmax[k]))
        fcr[k] = _f(max(_ZERO, _f(expf(_f(-_f(_A * _f(_ONE - fice))))
                                  - expf(_f(-_A))))
                    / _f(_ONE - expf(_f(-_A))))

    sicemax = _ZERO                                                 # :7341
    fcrmax = _ZERO                                                  # :7342
    for k in range(nsoil):                                          # :7344-7348
        if sice[k] > sicemax:
            sicemax = sice[k]
        if fcr[k] > fcrmax:
            fcrmax = fcr[k]
        # SH2OMIN at 7343/7347 is a local that nothing downstream reads.

    # The OPT_RUN==2 baseflow block at 7352-7357 is dead.

    if parameters.urban_flag:                                       # :7361
        fcr[0] = _f(0.95)

    # OPT_RUN 1/5/2/4/6/7/8 surface-runoff blocks (7363-7434) are dead.
    pddum, runsrf = infil(parameters, dt, zsoil, sh2o, sice, sicemax,
                          qinsur, pddum, runsrf, nsoil=nsoil)       # :7411

    niter = 3                                                       # :7440
    if _f(pddum * dt) > _f(dz[0] * parameters.smcmax[0]):           # :7443
        niter = niter * 2
    dtfine = _f(dt / _f(float(niter)))                              # :7449

    qdrain_save = _ZERO                                             # :7453
    runsrf_save = _ZERO                                             # :7454
    qdrain = _ZERO
    wcnd = np.zeros(nsoil, dtype=np.float32)

    for _iter in range(niter):                                      # :7456-7492
        if qinsur > _ZERO:                                          # :7457
            pddum, runsrf = infil(parameters, dtfine, zsoil, sh2o, sice,
                                  sicemax, qinsur, pddum, runsrf, nsoil=nsoil)

        # SRT reads SMC, not SH2O, under OPT_INF==1 (7775-7778).  On the first
        # pass that is the caller's entry array; SSTEP rewrites it below.
        rhstt, ai, bi, ci, qdrain, wcnd = srt(
            parameters, zsoil, pddum, etrani, qseva, smc, fcr, nsoil=nsoil)

        sh2o, smc, ai, bi, ci, rhstt, wplus = sstep(
            parameters, dtfine, dzsnso, sice, sh2o, ai, bi, ci, rhstt,
            nsoil=nsoil, nsnow=nsnow)

        rsat = _f(rsat + wplus)                                     # :7489
        qdrain_save = _f(qdrain_save + qdrain)                      # :7490
        runsrf_save = _f(runsrf_save + runsrf)                      # :7491

    qdrain = _f(qdrain_save / _f(float(niter)))                     # :7494
    runsrf = _f(runsrf_save / _f(float(niter)))                     # :7495

    runsrf = _f(_f(runsrf * _f(1000.0))
                + _f(_f(rsat * _f(1000.0)) / dt))                   # :7497
    qdrain = _f(qdrain * _f(1000.0))                                # :7498

    # OPT_TDRN 1/2 (7521-7529) and the OPT_RUN==2 groundwater removal
    # (7533-7541) are dead.

    # IF(OPT_RUN /= 1) at 7546 is true under opt_run=3.
    mliq = np.asarray([_f(_f(sh2o[k] * dz[k]) * _f(1000.0))
                       for k in range(nsoil)], dtype=np.float32)    # :7548
    watmin = _f(0.01)                                               # :7551
    for iz in range(nsoil - 1):                                     # :7552-7560
        xs = _f(watmin - mliq[iz]) if mliq[iz] < _ZERO else _ZERO
        mliq[iz] = _f(mliq[iz] + xs)
        mliq[iz + 1] = _f(mliq[iz + 1] - xs)

    iz = nsoil - 1                                                  # :7562
    xs = _f(watmin - mliq[iz]) if mliq[iz] < watmin else _ZERO
    mliq[iz] = _f(mliq[iz] + xs)
    runsub = _f(runsub - _f(xs / dt))                               # :7569
    # IF(OPT_RUN == 5) DEEPRECH = ... at 7570 is dead.

    for iz in range(nsoil):                                         # :7572-7574
        sh2o[iz] = _f(mliq[iz] / _f(dz[iz] * _f(1000.0)))

    # SMC is deliberately not recomputed here; see the docstring.
    return sh2o, smc, runsrf, qdrain, runsub, wcnd, fcrmax
