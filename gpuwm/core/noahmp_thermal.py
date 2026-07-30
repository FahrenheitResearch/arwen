"""CPU references for the pinned WRF v4.6.1 Noah-MP *thermal* leaf group.

TSNOSOI, HRT, HSTEP, PHASECHANGE and FRH2O -- the snow/soil temperature
solver, its tridiagonal chain, and the phase-change bookkeeping that follows
it.  Each function is a direct FP32 transcription of one ``private`` procedure
of ``phys/module_sf_noahmplsm.F`` at commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``, validated bitwise against
``gpuwm/data/noahmp/oracle/noahmp-thermal.csv``.  That fixture is produced by
``tools/noahmp_wrf461_oracle/build_thermal.sh`` from a scratch copy of the
pinned source carrying only the audited visibility patch
``patches/noahmp-lsm-leaf-visibility.patch``.

Nothing here admits ``sf_surface_physics=4``.  These are leaves of the Noah-MP
call tree; ENERGY, WATER and NOAHMP_SFLX are still open.

Conventions shared with the oracle and with ``gpuwm.core.noahmp_leaves``:

* ``kind_phys`` is ``kind(1.0)``, i.e. FP32.  Every literal is materialised as
  ``np.float32`` so no expression silently widens.
* Layered arrays span WRF's ``-NSNOW+1 : NSOIL``.  They are carried as 0-based
  arrays of length ``NSNOW + NSOIL``; layer ``k`` lives at position
  ``k + NSNOW - 1``.
* ``INTENT(OUT)`` entries the Fortran leaves undefined (slots above ``ISNOW``)
  are returned as 0.0, matching the harness pre-fill.  ``INTENT(INOUT)``
  entries above ``ISNOW`` are returned **unchanged** -- WRF never touches them
  and the fixture asserts the echo bit for bit.
* ROSR12 is not re-transcribed here; ``gpuwm.core.noahmp_leaves.rosr12`` is
  already pinned at ``max_ulp 0`` and HSTEP imports it, so a failure in the
  tridiagonal sweep localises to one module.
* Transcendentals go through ``gpuwm.core.noahmp_libm`` (glibc 2.39's
  ``logf``/``powf``), never numpy float32 and never "FP64 then round once".

Option identity: the pinned WRF Registry default.  What that kills inside this
group, asserted rather than ported:

* ``opt_tbot=2`` -- HRT's ``BOTFLX = 0`` branch at :5440-5442 is dead.
* ``opt_stc=1``  -- HRT's ``OPT_STC == 2`` diagonal at :5458-5460 is dead.
* ``opt_frz=1``  -- PHASECHANGE's ``CALL FRH2O`` at :5690-5693 is dead, so
  ``phasechange`` below implements the :5683-5689 closed form only.  ``frh2o``
  is transcribed anyway because it is a self-contained leaf with its own
  fixture, but **nothing calls it**: it is unreachable under this identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from gpuwm.core.noahmp_leaves import NLAY, NSNOW, NSOIL, rosr12
from gpuwm.core.noahmp_libm import logf as _logf
from gpuwm.core.noahmp_libm import powf as _powf

F = np.float32

# module_sf_noahmplsm.F:204-220
TFRZ = F(273.16)
HFUS = F(0.3336e06)
GRAV = F(9.80616)


def _slot(k: int) -> int:
    """Position of WRF layer index ``k`` in a 0-based ``-NSNOW+1:NSOIL`` array."""
    return k + NSNOW - 1


# ---------------------------------------------------------------------------
# HRT -- module_sf_noahmplsm.F:5375-5473
# ---------------------------------------------------------------------------
HRT_OUTPUTS = ("ai", "bi", "ci", "rhsts", "botflx")


def hrt(isnow, zsnso, stc, tbot, zbot, df, hcpct, ssoil, phi,
        nsnow=NSNOW, nsoil=NSOIL):
    """Tridiagonal coefficients and right-hand side of the heat equation.

    ``DT`` is part of WRF's argument list (:5397) and the body never
    references it, so it is not a parameter here; the oracle's zero-probe
    sweep is the proof.  Under the pinned ``OPT_TBOT = 2`` the bottom flux
    comes from :5443-5446, and under ``OPT_STC = 1`` the top diagonal is
    ``-CI`` (:5455-5457); the ``OPT_TBOT = 1`` and ``OPT_STC = 2`` forms are
    dead and are not transcribed.

    Returns ``(ai, bi, ci, rhsts, botflx)``.  Array entries above ``ISNOW``
    are 0.0 -- WRF leaves them undefined.
    """
    nlay = nsnow + nsoil
    zsnso = np.asarray(zsnso, dtype=F)
    stc = np.asarray(stc, dtype=F)
    df = np.asarray(df, dtype=F)
    hcpct = np.asarray(hcpct, dtype=F)
    phi = np.asarray(phi, dtype=F)
    tbot = F(tbot)
    zbot = F(zbot)
    ssoil = F(ssoil)
    for name, array in (("zsnso", zsnso), ("stc", stc), ("df", df),
                        ("hcpct", hcpct), ("phi", phi)):
        if array.shape[0] != nlay:
            raise ValueError(f"{name} must span -nsnow+1:nsoil")

    ai = np.zeros(nlay, dtype=F)
    bi = np.zeros(nlay, dtype=F)
    ci = np.zeros(nlay, dtype=F)
    rhsts = np.zeros(nlay, dtype=F)
    denom = np.zeros(nlay, dtype=F)
    ddz = np.zeros(nlay, dtype=F)
    dtsdz = np.zeros(nlay, dtype=F)
    eflux = np.zeros(nlay, dtype=F)
    botflx = F(0.0)

    for k in range(isnow + 1, nsoil + 1):                       # :5424-5449
        j = k + nsnow - 1
        if k == isnow + 1:                                      # :5425-5430
            denom[j] = F(-F(zsnso[j] * hcpct[j]))
            temp1 = F(-zsnso[j + 1])
            ddz[j] = F(F(2.0) / temp1)
            dtsdz[j] = F(F(F(2.0) * F(stc[j] - stc[j + 1])) / temp1)
            eflux[j] = F(F(F(df[j] * dtsdz[j]) - ssoil) - phi[j])
        elif k < nsoil:                                         # :5431-5436
            denom[j] = F(F(zsnso[j - 1] - zsnso[j]) * hcpct[j])
            temp1 = F(zsnso[j - 1] - zsnso[j + 1])
            ddz[j] = F(F(2.0) / temp1)
            dtsdz[j] = F(F(F(2.0) * F(stc[j] - stc[j + 1])) / temp1)
            eflux[j] = F(F(F(df[j] * dtsdz[j])
                           - F(df[j - 1] * dtsdz[j - 1])) - phi[j])
        else:                                                   # :5437-5447
            denom[j] = F(F(zsnso[j - 1] - zsnso[j]) * hcpct[j])
            # :5439 assigns TEMP1 here and never reads it again.
            dtsdz[j] = F(F(stc[j] - tbot)
                         / F(F(F(0.5) * F(zsnso[j - 1] + zsnso[j])) - zbot))
            botflx = F(-F(df[j] * dtsdz[j]))
            eflux[j] = F(F(F(-botflx) - F(df[j - 1] * dtsdz[j - 1])) - phi[j])

    for k in range(isnow + 1, nsoil + 1):                       # :5451-5471
        j = k + nsnow - 1
        if k == isnow + 1:                                      # :5452-5460
            ai[j] = F(0.0)
            ci[j] = F(-F(F(df[j] * ddz[j]) / denom[j]))
            bi[j] = F(-ci[j])
        elif k < nsoil:                                         # :5461-5464
            ai[j] = F(-F(F(df[j - 1] * ddz[j - 1]) / denom[j]))
            ci[j] = F(-F(F(df[j] * ddz[j]) / denom[j]))
            bi[j] = F(-F(ai[j] + ci[j]))
        else:                                                   # :5465-5468
            ai[j] = F(-F(F(df[j - 1] * ddz[j - 1]) / denom[j]))
            ci[j] = F(0.0)
            bi[j] = F(-F(ai[j] + ci[j]))
        rhsts[j] = F(eflux[j] / F(-denom[j]))                   # :5470
    return ai, bi, ci, rhsts, botflx


# ---------------------------------------------------------------------------
# HSTEP -- module_sf_noahmplsm.F:5477-5530
# ---------------------------------------------------------------------------

def hstep(isnow, dt, ai, bi, ci, rhsts, stc, nsnow=NSNOW, nsoil=NSOIL):
    """Scale the tridiagonal system by ``DT``, solve it, and update ``STC``.

    All five arrays are ``INTENT(INOUT)`` and WRF touches only
    ``ISNOW+1 .. NSOIL``, so entries above ``ISNOW`` are echoed unchanged.
    ``CI`` comes back as ROSR12's ``P`` (the solution) and ``RHSTS`` as its
    ``DELTA``; ``CI(NSOIL)`` on input is inert because ROSR12 zeroes its own
    ``C(NSOIL)`` at :5565 before reading it.

    Returns ``(ai, bi, ci, rhsts, stc)``.
    """
    nlay = nsnow + nsoil
    dt = F(dt)
    ai = np.asarray(ai, dtype=F).copy()
    bi = np.asarray(bi, dtype=F).copy()
    ci = np.asarray(ci, dtype=F).copy()
    rhsts = np.asarray(rhsts, dtype=F).copy()
    stc = np.asarray(stc, dtype=F).copy()
    for name, array in (("ai", ai), ("bi", bi), ("ci", ci),
                        ("rhsts", rhsts), ("stc", stc)):
        if array.shape[0] != nlay:
            raise ValueError(f"{name} must span -nsnow+1:nsoil")

    ntop = isnow + 1
    for k in range(ntop, nsoil + 1):                            # :5506-5511
        j = k + nsnow - 1
        rhsts[j] = F(rhsts[j] * dt)
        ai[j] = F(ai[j] * dt)
        bi[j] = F(F(1.0) + F(bi[j] * dt))
        ci[j] = F(ci[j] * dt)

    rhstsin = rhsts.copy()                                      # :5515-5518
    ciin = ci.copy()

    # :5522  CALL ROSR12 (CI, AI, BI, CIIN, RHSTSIN, RHSTS, ISNOW+1, ...)
    # P = CI, A = AI, B = BI, C = CIIN, D = RHSTSIN, DELTA = RHSTS.
    p, delta, _ = rosr12(ai, bi, ciin, rhstsin, ntop,
                         nsoil=nsoil, nsnow=nsnow)
    for k in range(ntop, nsoil + 1):
        j = k + nsnow - 1
        ci[j] = p[j]
        rhsts[j] = delta[j]

    for k in range(ntop, nsoil + 1):                            # :5526-5528
        j = k + nsnow - 1
        stc[j] = F(stc[j] + ci[j])
    return ai, bi, ci, rhsts, stc


# ---------------------------------------------------------------------------
# TSNOSOI -- module_sf_noahmplsm.F:5258-5371
# ---------------------------------------------------------------------------

def tsnosoi(isnow, zsnso, stc, tbot, ssoil, dt, snowh, df, hcpct, zbot,
            nsnow=NSNOW, nsoil=NSOIL):
    """Advance the snow/soil column temperature one step.

    The observable body ends at the unconditional ``RETURN`` on :5346, so what
    remains is: form ``ZBOTSNO`` (:5314), call HRT (:5324), call HSTEP
    (:5330).  ``ICE``, ``IST``, ``ILOC``, ``JLOC``, ``SAG``, ``TG`` and
    ``DZSNSO`` are part of WRF's argument list and reach nothing that survives
    that ``RETURN``, so they are not parameters here; the oracle's zero-probe
    sweep and the mutation study are the proof.

    ``zbot`` is ``parameters%ZBOT`` -- the only component of ``parameters``
    the routine references.

    ``PHI`` (:5310) is a local that TSNOSOI zeroes over ``ISNOW+1..NSOIL``
    before handing it to HRT, and HRT reads no other entry, so the solar
    penetration term is identically zero on this path.

    Returns ``(stc, eflxb)``.  ``stc`` above ``ISNOW`` is echoed unchanged.
    """
    nlay = nsnow + nsoil
    zbotsno = F(F(zbot) - F(snowh))                             # :5314
    phi = np.zeros(nlay, dtype=F)                               # :5310
    ai, bi, ci, rhsts, eflxb = hrt(                             # :5324
        isnow, zsnso, stc, tbot, zbotsno, df, hcpct, ssoil, phi,
        nsnow=nsnow, nsoil=nsoil)
    _, _, _, _, stc_out = hstep(                                # :5330
        isnow, dt, ai, bi, ci, rhsts, stc, nsnow=nsnow, nsoil=nsoil)
    # :5337-5342 fills the local EFLXB2, which the RETURN at 5346 discards.
    return stc_out, eflxb


# ---------------------------------------------------------------------------
# PHASECHANGE -- module_sf_noahmplsm.F:5595-5810
# ---------------------------------------------------------------------------

def phasechange(isnow, ist, dt, fact, dzsnso, stc, snice, snliq, sneqv,
                snowh, smc, sh2o, smcmax, psisat, bexp,
                nsnow=NSNOW, nsoil=NSOIL):
    """Melting and freezing of snow water and soil water.

    Under the pinned ``OPT_FRZ = 1`` the supercooled-water content comes from
    the closed form at :5683-5689; the ``CALL FRH2O`` at :5690-5693 belongs to
    ``OPT_FRZ == 2`` and is dead, so it is not reached from here.

    ``HCPCT`` (:5617), ``ILOC`` (:5608) and ``JLOC`` (:5609) are in WRF's
    argument list and the body never references them, so they are not
    parameters here.  ``DZSNSO`` is read only for soil layers (:5665-5668,
    :5687, :5804-5806), so its snow half is unread too.

    ``XMF`` (:5646, :5762, :5789) accumulates the latent heat of the phase
    change into a local that nothing ever reads -- WRF neither returns it nor
    uses it -- so it is not transcribed.

    Returns ``(stc, snice, snliq, sneqv, snowh, smc, sh2o, qmelt, ponding,
    imelt)``.  ``imelt`` entries above ``ISNOW`` are 0 -- WRF leaves them
    undefined.
    """
    nlay = nsnow + nsoil
    dt = F(dt)
    sneqv = F(sneqv)
    snowh = F(snowh)
    fact = np.asarray(fact, dtype=F)
    dzsnso = np.asarray(dzsnso, dtype=F)
    stc = np.asarray(stc, dtype=F).copy()
    snice = np.asarray(snice, dtype=F).copy()
    snliq = np.asarray(snliq, dtype=F).copy()
    smc = np.asarray(smc, dtype=F).copy()
    sh2o = np.asarray(sh2o, dtype=F).copy()
    smcmax = np.asarray(smcmax, dtype=F)
    psisat = np.asarray(psisat, dtype=F)
    bexp = np.asarray(bexp, dtype=F)

    qmelt = F(0.0)                                              # :5644
    ponding = F(0.0)                                            # :5645
    imelt = np.zeros(nlay, dtype=np.int32)
    supercool = np.zeros(nlay, dtype=F)                         # :5648-5650
    mice = np.zeros(nlay, dtype=F)
    mliq = np.zeros(nlay, dtype=F)
    wice0 = np.zeros(nlay, dtype=F)
    wliq0 = np.zeros(nlay, dtype=F)
    wmass0 = np.zeros(nlay, dtype=F)
    hm = np.zeros(nlay, dtype=F)
    xm = np.zeros(nlay, dtype=F)

    for j in range(isnow + 1, 1):                               # :5652-5655
        mice[_slot(j)] = snice[j + nsnow - 1]
        mliq[_slot(j)] = snliq[j + nsnow - 1]
    for j in range(1, nsoil + 1):                               # :5657-5660
        s = _slot(j)
        mliq[s] = F(F(sh2o[j - 1] * dzsnso[s]) * F(1000.0))
        mice[s] = F(F(F(smc[j - 1] - sh2o[j - 1]) * dzsnso[s]) * F(1000.0))
    for j in range(isnow + 1, nsoil + 1):                       # :5662-5669
        s = _slot(j)
        imelt[s] = 0
        hm[s] = F(0.0)
        xm[s] = F(0.0)
        wice0[s] = mice[s]
        wliq0[s] = mliq[s]
        wmass0[s] = F(mice[s] + mliq[s])

    if ist == 1:                                                # :5671-5697
        for j in range(1, nsoil + 1):
            s = _slot(j)
            if stc[s] < TFRZ:                                   # :5684, OPT_FRZ==1
                smp = F(F(HFUS * F(TFRZ - stc[s])) / F(GRAV * stc[s]))
                supercool[s] = F(smcmax[j - 1]
                                 * _powf(F(smp / psisat[j - 1]),
                                         F(-F(F(1.0) / bexp[j - 1]))))
                supercool[s] = F(F(supercool[s] * dzsnso[s]) * F(1000.0))

    for j in range(isnow + 1, nsoil + 1):                       # :5699-5713
        s = _slot(j)
        if mice[s] > F(0.0) and stc[s] >= TFRZ:
            imelt[s] = 1
        if mliq[s] > supercool[s] and stc[s] < TFRZ:
            imelt[s] = 2
        if isnow == 0 and sneqv > F(0.0) and j == 1:
            if stc[s] >= TFRZ:
                imelt[s] = 1

    for j in range(isnow + 1, nsoil + 1):                       # :5717-5731
        s = _slot(j)
        if imelt[s] > 0:
            hm[s] = F(F(stc[s] - TFRZ) / fact[s])
            stc[s] = TFRZ
        # :5723-5730.  FACT is DT/(HCPCT*DZSNSO) at :2497-2499 and therefore
        # strictly positive at every call site, which makes both of these
        # resets unreachable in a forecast: IMELT==1 implies STC >= TFRZ
        # implies HM >= 0, and IMELT==2 implies STC < TFRZ implies HM <= 0.
        # They are transcribed because they are part of the routine, and the
        # fixture binds them with a deliberately negative FACT.
        if imelt[s] == 1 and hm[s] < F(0.0):
            hm[s] = F(0.0)
            imelt[s] = 0
        if imelt[s] == 2 and hm[s] > F(0.0):
            hm[s] = F(0.0)
            imelt[s] = 0
        xm[s] = F(F(hm[s] * dt) / HFUS)

    one = _slot(1)
    if isnow == 0 and sneqv > F(0.0) and xm[one] > F(0.0):      # :5735-5752
        temp1 = sneqv
        sneqv = max(F(0.0), F(temp1 - xm[one]))
        propor = F(sneqv / temp1)
        snowh = max(F(0.0), F(propor * snowh))
        snowh = min(max(snowh, F(sneqv / F(500.0))),
                    F(sneqv / F(50.0)))
        heatr = F(hm[one] - F(F(HFUS * F(temp1 - sneqv)) / dt))
        if heatr > F(0.0):
            xm[one] = F(F(heatr * dt) / HFUS)
            hm[one] = heatr
        else:
            xm[one] = F(0.0)
            hm[one] = F(0.0)
        qmelt = F(max(F(0.0), F(temp1 - sneqv)) / dt)
        # :5751  XMF = HFUS*QMELT -- a local nothing reads.
        ponding = F(temp1 - sneqv)

    for j in range(isnow + 1, nsoil + 1):                       # :5756-5795
        s = _slot(j)
        if imelt[s] > 0 and abs(hm[s]) > F(0.0):
            heatr = F(0.0)
            if xm[s] > F(0.0):                                  # :5760-5762
                mice[s] = max(F(0.0), F(wice0[s] - xm[s]))
                heatr = F(hm[s] - F(F(HFUS * F(wice0[s] - mice[s])) / dt))
            elif xm[s] < F(0.0):                                # :5763-5776
                if j <= 0:
                    mice[s] = min(wmass0[s], F(wice0[s] - xm[s]))
                else:
                    if wmass0[s] < supercool[s]:
                        mice[s] = F(0.0)
                    else:
                        mice[s] = min(F(wmass0[s] - supercool[s]),
                                      F(wice0[s] - xm[s]))
                        mice[s] = max(mice[s], F(0.0))
                heatr = F(hm[s] - F(F(HFUS * F(wice0[s] - mice[s])) / dt))
            mliq[s] = max(F(0.0), F(wmass0[s] - mice[s]))       # :5779
            if abs(heatr) > F(0.0):                             # :5781-5790
                stc[s] = F(stc[s] + F(fact[s] * heatr))
                if j <= 0:
                    if F(mliq[s] * mice[s]) > F(0.0):
                        stc[s] = TFRZ
                    if mice[s] == F(0.0):                       # BARLAGE
                        stc[s] = TFRZ
                        hm[s + 1] = F(hm[s + 1] + heatr)
                        xm[s + 1] = F(F(hm[s + 1] * dt) / HFUS)
            # :5789  XMF = XMF + ... -- a local nothing reads.
            if j < 1:                                           # :5791-5793
                qmelt = F(qmelt
                          + F(max(F(0.0), F(wice0[s] - mice[s])) / dt))

    for j in range(isnow + 1, 1):                               # :5797-5800
        snliq[j + nsnow - 1] = mliq[_slot(j)]
        snice[j + nsnow - 1] = mice[_slot(j)]
    for j in range(1, nsoil + 1):                               # :5802-5805
        s = _slot(j)
        sh2o[j - 1] = F(mliq[s] / F(F(1000.0) * dzsnso[s]))
        smc[j - 1] = F(F(mliq[s] + mice[s]) / F(F(1000.0) * dzsnso[s]))

    return (stc, snice, snliq, sneqv, snowh, smc, sh2o, qmelt, ponding,
            imelt)


# ---------------------------------------------------------------------------
# FRH2O -- module_sf_noahmplsm.F:5814-5946
#
# DEAD under the pinned option identity.  Its only call site in the module is
# :5692, inside ``IF (OPT_FRZ == 2)``, and the pinned identity is
# ``opt_frz = 1``.  Nothing in gpuwm calls this: ``phasechange`` above
# implements the :5683-5689 closed form and never reaches here.  It is
# transcribed and pinned because it is a self-contained leaf whose fixture
# costs nothing, so that if the option identity ever moves the port is already
# measured rather than written blind.
# ---------------------------------------------------------------------------
FRH2O_CK = F(8.0)          # :5851
FRH2O_BLIM = F(5.5)        # :5851
FRH2O_ERROR = F(0.005)     # :5851


def frh2o(tkelv, smc, sh2o, bexp, psisat, smcmax):
    """Koren (1999) eqn 17 supercooled liquid water, by Newton iteration.

    ``ISOIL`` is a pure index in WRF and is fixed at 1 here: the three
    ``parameters`` components the routine reads (``BEXP``, ``PSISAT``,
    ``SMCMAX``) are carried as ordinary scalars.

    ``DICE`` (:5852) is a named constant the body never uses.

    Returns ``free``.
    """
    tkelv = F(tkelv)
    smc = F(smc)
    sh2o = F(sh2o)
    bexp = F(bexp)
    psisat = F(psisat)
    smcmax = F(smcmax)

    bx = bexp                                                   # :5860
    if bexp > FRH2O_BLIM:                                       # :5866
        bx = FRH2O_BLIM
    nlog = 0
    kcount = 0
    swl = F(0.0)

    if tkelv > F(TFRZ - F(1.0e-3)):                             # :5872
        return smc                                              # :5873

    # :5878  IF (CK /= 0.0) -- CK is the parameter 8.0, so always taken.
    swl = F(smc - sh2o)                                         # :5879
    if swl > F(smc - F(0.02)):                                  # :5883
        swl = F(smc - F(0.02))
    if swl < F(0.0):                                            # :5887
        swl = F(0.0)
    while nlog < 10 and kcount == 0:                            # :5888-5889
        nlog += 1
        arg = F(F(F(F(psisat * GRAV) / HFUS)
                  * _powf(F(F(1.0) + F(FRH2O_CK * swl)), F(2.0)))
                * _powf(F(smcmax / F(smc - swl)), bx))          # :5891-5893
        df = F(_logf(arg) - _logf(F(-F(F(tkelv - TFRZ) / tkelv))))
        denom = F(F(F(F(2.0) * FRH2O_CK) / F(F(1.0) + F(FRH2O_CK * swl)))
                  + F(bx / F(smc - swl)))                       # :5894
        swlk = F(swl - F(df / denom))                           # :5895
        if swlk > F(smc - F(0.02)):                             # :5899
            swlk = F(smc - F(0.02))
        if swlk < F(0.0):                                       # :5900
            swlk = F(0.0)
        dswl = abs(F(swlk - swl))                               # :5905
        swl = swlk                                              # :5909
        if dswl <= FRH2O_ERROR:                                 # :5910
            kcount += 1
    free = F(smc - swl)                                         # :5919

    if kcount == 0:                                             # :5928
        # :5929-5930 writes a diagnostic through wrf_message.
        fk = F(_powf(F(F(HFUS / F(GRAV * F(-psisat)))
                       * F(F(tkelv - TFRZ) / tkelv)),
                     F(-F(F(1.0) / bx)))
               * smcmax)                                        # :5931-5932
        if fk < F(0.02):                                        # :5933
            fk = F(0.02)
        free = min(fk, smc)                                     # :5934
    return F(free)


# ---------------------------------------------------------------------------
# Flat-slot adapters.  These mirror
# ``tools/noahmp_wrf461_oracle/run_thermal.F90`` exactly, so the oracle CSV can
# be replayed without re-deriving any packing.
# ---------------------------------------------------------------------------

def _eval_hrt(x, ix):
    zsnso = x[0:NLAY]
    stc = x[NLAY:2 * NLAY]
    df = x[2 * NLAY:3 * NLAY]
    hcpct = x[3 * NLAY:4 * NLAY]
    phi = x[4 * NLAY:5 * NLAY]
    tbot, zbot, ssoil = x[5 * NLAY], x[5 * NLAY + 1], x[5 * NLAY + 3]
    # x[5 * NLAY + 2] is DT: declared INTENT(IN) at :5397, never referenced.
    ai, bi, ci, rhsts, botflx = hrt(int(ix[0]), zsnso, stc, tbot, zbot,
                                    df, hcpct, ssoil, phi)
    return np.concatenate((ai, bi, ci, rhsts,
                           np.asarray([botflx], dtype=F))).astype(F)


def _eval_hstep(x, ix):
    ai = x[0:NLAY]
    bi = x[NLAY:2 * NLAY]
    ci = x[2 * NLAY:3 * NLAY]
    rhsts = x[3 * NLAY:4 * NLAY]
    stc = x[4 * NLAY:5 * NLAY]
    dt = x[5 * NLAY]
    out = hstep(int(ix[0]), dt, ai, bi, ci, rhsts, stc)
    return np.concatenate(out).astype(F)


def _eval_tsnosoi(x, ix):
    zsnso = x[0:NLAY]
    stc = x[NLAY:2 * NLAY]
    df = x[2 * NLAY:3 * NLAY]
    hcpct = x[3 * NLAY:4 * NLAY]
    # x[4*NLAY : 5*NLAY] is DZSNSO, read only after the RETURN at :5346.
    base = 5 * NLAY
    tbot, ssoil = x[base], x[base + 1]
    # x[base + 2] is SAG, x[base + 5] is TG: neither survives the RETURN.
    dt, snowh, zbot = x[base + 3], x[base + 4], x[base + 6]
    # ix[1] ICE, ix[2] IST, ix[3] ILOC, ix[4] JLOC: same.
    stc_out, eflxb = tsnosoi(int(ix[0]), zsnso, stc, tbot, ssoil, dt, snowh,
                             df, hcpct, zbot)
    return np.concatenate(
        (stc_out, np.asarray([eflxb], dtype=F))).astype(F)


def _eval_phasechange(x, ix):
    fact = x[0:NLAY]
    dzsnso = x[NLAY:2 * NLAY]
    # x[2*NLAY : 3*NLAY] is HCPCT, declared INTENT(IN) at :5617 and never
    # referenced.
    stc = x[3 * NLAY:4 * NLAY]
    base = 4 * NLAY
    snice = x[base:base + NSNOW]
    snliq = x[base + NSNOW:base + 2 * NSNOW]
    base += 2 * NSNOW
    smc = x[base:base + NSOIL]
    sh2o = x[base + NSOIL:base + 2 * NSOIL]
    base += 2 * NSOIL
    sneqv, snowh, dt = x[base], x[base + 1], x[base + 2]
    base += 3
    smcmax = x[base:base + NSOIL]
    psisat = x[base + NSOIL:base + 2 * NSOIL]
    bexp = x[base + 2 * NSOIL:base + 3 * NSOIL]
    # ix[2] ILOC and ix[3] JLOC are declared at :5608-5609 and never read.
    out = phasechange(int(ix[0]), int(ix[1]), dt, fact, dzsnso, stc,
                      snice, snliq, sneqv, snowh, smc, sh2o,
                      smcmax, psisat, bexp)
    (stc_o, snice_o, snliq_o, sneqv_o, snowh_o, smc_o, sh2o_o,
     qmelt, ponding, imelt) = out
    return np.concatenate((
        stc_o, snice_o, snliq_o, smc_o, sh2o_o,
        np.asarray([sneqv_o, snowh_o, qmelt, ponding], dtype=F),
        imelt.astype(F))).astype(F)


def _eval_frh2o(x, ix):
    return np.asarray([frh2o(x[0], x[1], x[2], x[3], x[4], x[5])], dtype=F)


THERMAL_EVALUATORS: Mapping[
    str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "hrt": _eval_hrt,
    "hstep": _eval_hstep,
    "tsnosoi": _eval_tsnosoi,
    "phasechange": _eval_phasechange,
    "frh2o": _eval_frh2o,
}
