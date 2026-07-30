"""Noah-MP driver-side cold start, in FP32.

Ports ``module_sf_noahmpdrv``'s ``SNOW_INIT`` (2340-2440) and ``NOAHMP_INIT``
(1828-2335) from ``phys/module_sf_noahmpdrv.F`` at WRF commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``
(``sha256 9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890``).

``kind_phys == kind(1.0)`` in that build, so every quantity here is IEEE
binary32 and every arithmetic boundary is pinned through
:func:`gpuwm.core.noahmp_libm.f32`.

This is marshalling, not physics: unit conversions, a snow-layer topology
ladder, soil-layer bookkeeping and the option-identity plumbing that decides
which of them run.  The one transcendental is the supercooled-liquid initial
guess at 2095-2096, whose ``x**y`` gfortran lowers to a scalar ``powf`` call;
it goes through :func:`gpuwm.core.noahmp_libm.powf`, the verified glibc 2.39
transcription, never through ``numpy.float32`` or ``**``.

Pinned option identity
----------------------
``dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0``, with
``sf_urban_physics=0``.  ``NOAHMP_INIT`` reads five of those, and every one is
an argument here, checked rather than assumed:

* ``iopt_run != 5`` selects the constant aquifer cold start at 2145-2148 and
  kills the groundwater block at 2299-2331 together with every OPTIONAL
  groundwater argument, ``STEPWTD`` and ``areaxy``.  ``iopt_run == 5`` is
  refused, not silently approximated.
* ``iopt_crop == 0`` kills the Liu crop block (2201-2232) and the gecros block
  (2236-2260).  ``cropcat`` is then written **only** on the
  barren/ice/urban/water branch (2178); on a vegetated column no statement
  writes it, and since it is an ``INTENT(OUT)`` dummy that gfortran passes by
  reference, the caller's value stands.  :func:`noahmp_init_column` reproduces
  that by returning ``cropcat`` unchanged, which is the defined behaviour of
  the pinned build and is pinned by the fixture, not an assumption.
* ``iopt_irr == 0`` kills the irrigation cold start (2263-2278), so none of
  ``irnumsi``/``irnummi``/``irnumfi``/``irwatsi``/``irwatmi``/``irwatfi``/
  ``ireloss``/``irsivol``/``irmivol``/``irfivol``/``irrsplh`` is an argument
  here at all.
* ``sf_urban_physics == 0`` routes ``ISURBAN`` and ``LCZ_1..LCZ_11`` into the
  zeroing branch at 2164-2178, which makes the ``NATURAL_TABLE`` form of
  ``masslai`` at 2185 unreachable.  It is implemented anyway, because the
  branch is decided by a runtime argument rather than by the identity, and it
  is reached only when a caller passes ``sf_urban_physics > 0``.

Arguments the pinned identity does not consume
----------------------------------------------
``XLAT`` (read only by ``gecros_init`` at 2258), ``TMN`` (its only write, 2080,
is commented out in the pinned source), ``croptype`` (2203, 2238-2240),
``FNDSOILW`` (declared at 1891 and never referenced in the body) and the eleven
irrigation carriers.  None of them is an argument here;
:mod:`tests.test_noahmp_driver` holds the claim against the fixture, which
drives all fourteen non-zero and requires entry to equal exit.

Where WRF leaves a slot undefined
---------------------------------
``SNOW_INIT`` declares ``ZSNSOXY`` ``INTENT(OUT)`` over ``-NSNOW+1:NSOIL`` but
assigns only ``ISNOWXY+1..NSOIL`` (2432-2435).  With ``ISNOWXY == 0`` the three
snow slots are never written.  The defined behaviour of the pinned build is
that the caller's values stand, so :func:`snow_init_column` copies the caller's
array and mutates it rather than starting from zeros.  Its local ``DZSNO`` has
the same shape of hazard -- zeroed only on the ``SNODEP < 0.025`` branch
(2383) -- but every slot it leaves unassigned is above ``ISNOW`` and therefore
never read, which the fixture's ``-finit-real=snan`` control proves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# _Col is the project's one Fortran-lower-bound view; noahmp_snow owns it and
# this module must not fork a second one.  Both routines here address the
# snow/soil stack over -NSNOW+1:NSOIL exactly as the Fortran does.
from gpuwm.core.noahmp_snow import _Col
from gpuwm.core.noahmp_libm import f32, powf

__all__ = [
    "NSNOW_DEFAULT",
    "NSOIL_DEFAULT",
    "LandUseIdentity",
    "SnowInitColumn",
    "ColdStartColumn",
    "snow_init_column",
    "noahmp_init_column",
]

NSNOW_DEFAULT = 3
NSOIL_DEFAULT = 4

# --- module_sf_noahmpdrv.F 1988-1991 ---------------------------------------
# BLIM (1988) is declared in the same PARAMETER block and never referenced
# anywhere in the body, so it is deliberately absent here.
HLICE = f32(3.335e5)   # latent heat of fusion (J/kg)
GRAV = f32(9.81)       # gravitational acceleration (m/s2)
T0 = f32(273.15)       # freezing point (K)

# 2094.  Deliberately not T0: WRF tests against 273.149, so a layer at exactly
# 273.15 K takes the *unfrozen* branch and one at 273.148 K does not.
_FREEZE_TEST = f32(273.149)

_SWE_CAP = f32(2000.0)        # 2037-2039
_GLACIER_TSLB_CAP = f32(263.15)  # 2078
_GLACIER_SWE_FLOOR = f32(10.0)   # 2081
_SNOWH_PER_SWE = f32(0.005)   # 2022
_GLACIER_SNOWH_PER_SWE = f32(0.01)  # 2082
_FK_FLOOR = f32(0.02)         # 2097
_LAI_FLOOR = f32(0.05)        # 2182
_SAI_PER_LAI = f32(0.1)       # 2183
_MASSSAI = f32(f32(1000.0) / f32(3.0))  # 2190, a constant expression
_WA_COLD = f32(4900.0)        # 2146
_EAH_COLD = f32(2000.0)       # 2124
_ALBOLD_COLD = f32(0.65)      # 2140
_CHSTAR_COLD = f32(0.1)       # 2133
_GRAIN_COLD = f32(1e-10)      # 2176, 2196
_ROOT_COLD = f32(500.0)       # 2192-2193
_CARBON_COLD = f32(1000.0)    # 2194-2195


@dataclass(frozen=True)
class LandUseIdentity:
    """The ``NOAHMP_TABLES`` category indices the cold start branches on.

    These come from the ``MPTABLE.TBL`` land-use block that
    :func:`gpuwm.core.noahmp.load_noahmp_parameters` already reads, so nothing
    here re-parses a table.  ``lcz`` is the eleven LCZ classes in the order
    ``module_sf_noahmpdrv.F`` tests them at 2157-2160.
    """

    isice: int
    isurban: int
    iswater: int
    isbarren: int
    natural: int
    lcz: tuple[int, ...]

    def is_urban_point(self, vegtyp: int) -> bool:
        """2156-2162: ``urbanpt_flag``."""
        return vegtyp == self.isurban or vegtyp in self.lcz


@dataclass
class SnowInitColumn:
    """What ``SNOW_INIT`` leaves behind for one column."""

    isnow: int
    zsnso: np.ndarray   # (-nsnow+1 : nsoil)
    tsno: np.ndarray    # (-nsnow+1 : 0)
    snice: np.ndarray   # (-nsnow+1 : 0)
    snliq: np.ndarray   # (-nsnow+1 : 0)


@dataclass
class ColdStartColumn:
    """What ``NOAHMP_INIT`` leaves behind for one column."""

    snow: float
    snowh: float
    canwat: float
    tslb: np.ndarray
    smois: np.ndarray
    sh2o: np.ndarray
    zsoil: np.ndarray
    isnow: int
    zsnso: np.ndarray
    tsno: np.ndarray
    snice: np.ndarray
    snliq: np.ndarray
    tv: float
    tg: float
    canice: float
    canliq: float
    eah: float
    tah: float
    cm: float
    ch: float
    fwet: float
    sneqvo: float
    albold: float
    qsnow: float
    qrain: float
    wslake: float
    zwt: float
    wa: float
    wt: float
    lai: float
    xsai: float
    lfmass: float
    rtmass: float
    stmass: float
    wood: float
    stblcp: float
    fastcp: float
    grain: float
    gdd: float
    t2mv: float
    t2mb: float
    chstar: float
    qtdrain: float
    cropcat: int


# ---------------------------------------------------------------------------
# SNOW_INIT -- module_sf_noahmpdrv.F 2340-2440
# ---------------------------------------------------------------------------

def snow_init_column(nsnow, nsoil, zsoil, swe, tgxy, snodep,
                     zsnsoxy, tsnoxy, snicexy, snliqxy) -> SnowInitColumn:
    """One column of ``SNOW_INIT``.

    ``zsoil`` is ``ZSOIL(1:NSOIL)`` (0-based here).  ``zsnsoxy``, ``tsnoxy``,
    ``snicexy`` and ``snliqxy`` are the caller's ``INTENT(OUT)`` arrays over
    ``-nsnow+1:nsoil`` and ``-nsnow+1:0``; they are copied, not aliased, and
    the slots WRF never assigns come back carrying their entry values.

    ``ims``/``ime``/``jms``/``jme``/``its``/``itf``/``jts``/``jtf`` are pure
    loop bounds and are not arguments here; a column has no use for them.
    """
    nsnow = int(nsnow)
    nsoil = int(nsoil)
    zs = np.asarray(zsoil, dtype=np.float32)
    if zs.size != nsoil:
        raise ValueError(f"zsoil has {zs.size} layers, expected {nsoil}")
    swe = f32(swe)
    tgxy = f32(tgxy)
    snodep = f32(snodep)

    zsnso = np.asarray(zsnsoxy, dtype=np.float32).copy()
    tsno = np.asarray(tsnoxy, dtype=np.float32).copy()
    snice = np.asarray(snicexy, dtype=np.float32).copy()
    snliq = np.asarray(snliqxy, dtype=np.float32).copy()
    for name, arr, want in (("zsnsoxy", zsnso, nsnow + nsoil),
                            ("tsnoxy", tsno, nsnow),
                            ("snicexy", snice, nsnow),
                            ("snliqxy", snliq, nsnow)):
        if arr.size != want:
            raise ValueError(f"{name} has {arr.size} slots, expected {want}")

    ZSOIL = _Col(zs, 1)
    ZSNSO = _Col(zsnso, -nsnow + 1)
    TSNO = _Col(tsno, -nsnow + 1)
    SNICE = _Col(snice, -nsnow + 1)
    SNLIQ = _Col(snliq, -nsnow + 1)

    # DZSNO is a local that only the SNODEP < 0.025 branch zeroes (2383).  NaN
    # marks the slots WRF leaves undefined so a port that reads one is a hard
    # failure here instead of silently agreeing with whatever was on the stack.
    dzsno_data = np.full(nsnow, np.float32("nan"), dtype=np.float32)
    DZSNO = _Col(dzsno_data, -nsnow + 1)
    dzsnso_data = np.full(nsnow + nsoil, np.float32("nan"), dtype=np.float32)
    DZSNSO = _Col(dzsnso_data, -nsnow + 1)

    # 2381-2409: the depth ladder.  The five tests are exhaustive over the
    # reals once SNODEP >= 0.025, so the `else` at 2406 -- wrf_error_fatal --
    # is reachable only for a NaN.
    if snodep < f32(0.025):
        isnow = 0
        dzsno_data[:] = np.float32(0.0)
    elif f32(0.025) <= snodep <= f32(0.05):
        isnow = -1
        DZSNO[0] = snodep
    elif f32(0.05) < snodep <= f32(0.10):
        isnow = -2
        DZSNO[-1] = f32(snodep / f32(2.0))
        DZSNO[0] = f32(snodep / f32(2.0))
    elif f32(0.10) < snodep <= f32(0.25):
        isnow = -2
        DZSNO[-1] = f32(0.05)
        DZSNO[0] = f32(snodep - DZSNO[-1])
    elif f32(0.25) < snodep <= f32(0.45):
        isnow = -3
        DZSNO[-2] = f32(0.05)
        DZSNO[-1] = f32(f32(0.5) * f32(snodep - DZSNO[-2]))
        DZSNO[0] = f32(f32(0.5) * f32(snodep - DZSNO[-2]))
    elif snodep > f32(0.45):
        isnow = -3
        DZSNO[-2] = f32(0.05)
        DZSNO[-1] = f32(0.20)
        DZSNO[0] = f32(f32(snodep - DZSNO[-1]) - DZSNO[-2])
    else:
        # 2407.  WRF calls wrf_error_fatal here; nothing but a NaN gets in.
        raise ValueError("Problem with the logic assigning snow layers.")

    # 2411-2418
    tsno[:] = np.float32(0.0)
    snice[:] = np.float32(0.0)
    snliq[:] = np.float32(0.0)
    for iz in range(isnow + 1, 1):
        TSNO[iz] = tgxy
        SNLIQ[iz] = f32(0.0)
        # `1.00 * DZSNO(IZ) * (SWE/SNODEP)` associates left to right, and
        # 1.0*x is exact for every finite binary32 x.
        SNICE[iz] = f32(float(DZSNO[iz]) * f32(swe / snodep))

    # 2421-2429
    for iz in range(isnow + 1, 1):
        DZSNSO[iz] = f32(-float(DZSNO[iz]))
    DZSNSO[1] = ZSOIL[1]
    for iz in range(2, nsoil + 1):
        DZSNSO[iz] = f32(float(ZSOIL[iz]) - float(ZSOIL[iz - 1]))

    # 2432-2435.  Slots above ISNOW+1 are never written; they keep the values
    # the caller handed in.
    ZSNSO[isnow + 1] = DZSNSO[isnow + 1]
    for iz in range(isnow + 2, nsoil + 1):
        ZSNSO[iz] = f32(float(ZSNSO[iz - 1]) + float(DZSNSO[iz]))

    return SnowInitColumn(isnow=isnow, zsnso=zsnso, tsno=tsno, snice=snice,
                          snliq=snliq)


# ---------------------------------------------------------------------------
# NOAHMP_INIT -- module_sf_noahmpdrv.F 1828-2335
# ---------------------------------------------------------------------------

def noahmp_init_column(*, nsoil, dzs, fndsnowh, identity, vegtyp, soiltyp,
                       xice, tsk, lai, bexp, smcmax, psisat, sla,
                       snow, snowh, tslb, smois, zsnsoxy, tsnoxy, snicexy,
                       snliqxy, cropcat, sla_natural=None,
                       nsnow=NSNOW_DEFAULT, iopt_run=3, iopt_crop=0,
                       iopt_irr=0, iopt_irrm=0, sf_urban_physics=0,
                       restart=False) -> ColdStartColumn:
    """One column of ``NOAHMP_INIT``.

    ``bexp``/``smcmax``/``psisat`` are ``BEXP_TABLE``/``SMCMAX_TABLE``/
    ``PSISAT_TABLE`` already indexed by ``soiltyp`` (2085-2087), and ``sla`` is
    ``SLA_TABLE(vegtyp)`` (2187).  Indexing the tables is
    :mod:`gpuwm.core.noahmp`'s job, not this one's; passing the scalars keeps
    this function exactly the marshalling WRF performs.  ``sla_natural`` is
    ``SLA_TABLE(NATURAL_TABLE)`` and is required only on the
    ``sf_urban_physics > 0`` path (2185).

    ``sh2o`` is not an argument: every live path at 2076/2098/2100/2105
    overwrites it before anything reads it, so its entry value is dead.
    """
    if restart:
        # 2009: the whole body is skipped.  Nothing is defined to happen, and
        # a caller asking for it is asking for a no-op it can do itself.
        raise ValueError("restart=True: NOAHMP_INIT does nothing; do not "
                         "call it")
    if iopt_run == 5:
        raise ValueError("iopt_run=5 needs the groundwater cold start at "
                         "2299-2331, which this port does not implement")
    if iopt_crop != 0:
        raise ValueError(f"iopt_crop={iopt_crop} needs the crop cold start at "
                         "2201-2260, which this port does not implement")
    if iopt_irr != 0:
        raise ValueError(f"iopt_irr={iopt_irr} needs the irrigation cold "
                         "start at 2263-2278, which this port does not "
                         "implement")
    if soiltyp < 1:
        # 2047-2057: WRF calls wrf_error_fatal.
        raise ValueError(f"out of range ISLTYP {soiltyp}")

    nsoil = int(nsoil)
    nsnow = int(nsnow)
    dz = np.asarray(dzs, dtype=np.float32)
    if dz.size != nsoil:
        raise ValueError(f"dzs has {dz.size} layers, expected {nsoil}")
    tslb = np.asarray(tslb, dtype=np.float32).copy()
    smois = np.asarray(smois, dtype=np.float32).copy()
    sh2o = np.zeros(nsoil, dtype=np.float32)
    vegtyp = int(vegtyp)
    xice = f32(xice)
    tsk = f32(tsk)
    lai = f32(lai)
    bexp = f32(bexp)
    smcmax = f32(smcmax)
    psisat = f32(psisat)
    snow = f32(snow)
    snowh = f32(snowh)

    DZS = _Col(dz, 1)
    TSLB = _Col(tslb, 1)
    SMOIS = _Col(smois, 1)
    SH2O = _Col(sh2o, 1)

    # --- 2017-2025: physical snow height, when the input file had none.
    if not fndsnowh:
        snowh = f32(snow * _SNOWH_PER_SWE)

    # --- 2028-2042: cap SWE at 2000 mm, preserving the pack density.
    # The `snow>0 .and. snowh==0 .or. snowh>0 .and. snow==0` test at 2032 only
    # emits a wrf_message; it changes no state and is not reproduced here.
    if snow > _SWE_CAP:
        snowh = f32(f32(snowh * _SWE_CAP) / snow)
        snow = _SWE_CAP

    # --- 2072-2110: soil ice and the glacier fill.
    if vegtyp == identity.isice and xice <= f32(0.0):
        for ns in range(1, nsoil + 1):
            SMOIS[ns] = f32(1.0)
            SH2O[ns] = f32(0.0)
            TSLB[ns] = min(TSLB[ns], _GLACIER_TSLB_CAP)
        snow = max(snow, _GLACIER_SWE_FLOOR)
        snowh = f32(snow * _GLACIER_SNOWH_PER_SWE)
    else:
        for ns in range(1, nsoil + 1):
            if SMOIS[ns] > smcmax:
                SMOIS[ns] = smcmax
        if bexp > f32(0.0) and smcmax > f32(0.0) and psisat > f32(0.0):
            for ns in range(1, nsoil + 1):
                if TSLB[ns] < _FREEZE_TEST:
                    SH2O[ns] = _supercooled_guess(TSLB[ns], bexp, smcmax,
                                                  psisat, SMOIS[ns])
                else:
                    SH2O[ns] = SMOIS[ns]
        else:
            for ns in range(1, nsoil + 1):
                SH2O[ns] = SMOIS[ns]

    # --- 2114-2153: the unconditional cold-start constants.
    # 2118/2120/2126/2130/2132 all clamp to 273.15 under the same predicate.
    cold_skin = f32(tsk)
    if snow > f32(0.0) and tsk > T0:
        cold_skin = T0
    qtdrain = f32(0.0)
    tv = cold_skin
    tg = cold_skin
    canwat = f32(0.0)
    canliq = canwat        # 2122 reads CANWAT *after* 2121 zeroed it
    canice = f32(0.0)
    eah = _EAH_COLD
    tah = cold_skin
    t2mv = cold_skin
    t2mb = cold_skin
    chstar = _CHSTAR_COLD
    cm = f32(0.0)
    ch = f32(0.0)
    fwet = f32(0.0)
    sneqvo = f32(0.0)
    albold = _ALBOLD_COLD
    qsnow = f32(0.0)
    qrain = f32(0.0)
    wslake = f32(0.0)

    # 2145-2148.  iopt_run == 5 was refused above.
    wa = _WA_COLD
    wt = wa
    zwt = f32(f32(f32(25.0) + f32(2.0))
              - f32(f32(wa / f32(1000.0)) / f32(0.2)))

    # --- 2156-2280: the vegetation cold start.
    urbanpt = identity.is_urban_point(vegtyp)
    bare = (vegtyp == identity.isbarren or vegtyp == identity.isice
            or (sf_urban_physics == 0 and urbanpt)
            or vegtyp == identity.iswater)
    if bare:
        lai_out = f32(0.0)
        xsai = f32(0.0)
        lfmass = f32(0.0)
        stmass = f32(0.0)
        rtmass = f32(0.0)
        wood = f32(0.0)
        stblcp = f32(0.0)
        fastcp = f32(0.0)
        grain = _GRAIN_COLD
        gdd = f32(0.0)
        cropcat_out = 0
    else:
        lai_out = max(lai, _LAI_FLOOR)
        xsai = max(f32(_SAI_PER_LAI * lai_out), _LAI_FLOOR)
        if urbanpt:
            if sla_natural is None:
                raise ValueError("sf_urban_physics>0 on an urban column needs "
                                 "sla_natural = SLA_TABLE(NATURAL_TABLE) "
                                 "(2185)")
            sla_used = f32(sla_natural)
        else:
            sla_used = f32(sla)
        masslai = f32(f32(1000.0) / max(sla_used, f32(1.0)))
        lfmass = f32(lai_out * masslai)
        stmass = f32(xsai * _MASSSAI)
        rtmass = _ROOT_COLD
        wood = _ROOT_COLD
        stblcp = _CARBON_COLD
        fastcp = _CARBON_COLD
        grain = _GRAIN_COLD
        gdd = f32(0.0)
        # iopt_crop == 0: no statement writes cropcat on this branch, and the
        # INTENT(OUT) dummy is passed by reference, so the caller's value
        # stands.
        cropcat_out = int(cropcat)

    # --- 2288-2291: soil interface depths, negative downward.
    zsoil = np.zeros(nsoil, dtype=np.float32)
    ZSOIL = _Col(zsoil, 1)
    ZSOIL[1] = f32(-float(DZS[1]))
    for ns in range(2, nsoil + 1):
        ZSOIL[ns] = f32(float(ZSOIL[ns - 1]) - float(DZS[ns]))

    # --- 2295-2297.  NSNOW is the literal 3 at the call site; it is an
    # argument here only so a caller with a different snow stack is not
    # silently given WRF's.
    si = snow_init_column(nsnow, nsoil, zsoil, snow, tg, snowh,
                          zsnsoxy, tsnoxy, snicexy, snliqxy)

    return ColdStartColumn(
        snow=snow, snowh=snowh, canwat=canwat, tslb=tslb, smois=smois,
        sh2o=sh2o, zsoil=zsoil, isnow=si.isnow, zsnso=si.zsnso, tsno=si.tsno,
        snice=si.snice, snliq=si.snliq, tv=tv, tg=tg, canice=canice,
        canliq=canliq, eah=eah, tah=tah, cm=cm, ch=ch, fwet=fwet,
        sneqvo=sneqvo, albold=albold, qsnow=qsnow, qrain=qrain, wslake=wslake,
        zwt=zwt, wa=wa, wt=wt, lai=lai_out, xsai=xsai, lfmass=lfmass,
        rtmass=rtmass, stmass=stmass, wood=wood, stblcp=stblcp, fastcp=fastcp,
        grain=grain, gdd=gdd, t2mv=t2mv, t2mb=t2mb, chstar=chstar,
        qtdrain=qtdrain, cropcat=cropcat_out)


def _supercooled_guess(tslb, bexp, smcmax, psisat, smois) -> np.float32:
    """2095-2098: the frozen-soil liquid-water initial guess.

    ``(-1/BEXP)`` is REAL division -- ``-1`` is a default INTEGER that Fortran
    converts to REAL before dividing -- so the exponent is ``-1.0/BEXP`` and
    the whole power is a scalar ``powf`` call, which is what ``nm -u`` shows on
    the pinned object.  glibc's ``powf`` is not correctly rounded, so
    evaluating this in binary64 and rounding once would be a *different*
    function, not a more accurate one; :mod:`gpuwm.core.noahmp_libm` is the
    only admissible source.
    """
    base = f32(f32(HLICE / f32(GRAV * f32(-float(psisat))))
               * f32(f32(float(tslb) - T0) / float(tslb)))
    fk = f32(float(powf(base, f32(f32(-1.0) / float(bexp)))) * float(smcmax))
    fk = max(fk, _FK_FLOOR)
    return np.float32(min(fk, float(smois)))
