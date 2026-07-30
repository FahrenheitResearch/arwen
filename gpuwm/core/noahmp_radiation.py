"""Bitwise CPU transcription of the Noah-MP shortwave-radiation leaves.

Source of truth
---------------
WRF 4.6.1 ``phys/module_sf_noahmplsm.F`` at tree commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``,
``sha256(module_sf_noahmplsm.F) = bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282``.

  SNOW_AGE        lines 3119-3167
  SNOWALB_CLASS   lines 3226-3275
  GROUNDALB       lines 3279-3332
  SURRAD          lines 2994-3115
  TWOSTREAM       lines 3336-3574
  ALBEDO          lines 2810-2990

Pinned option identity (the exact WRF Registry defaults)
--------------------------------------------------------
``opt_rad = 3``  -- TWOSTREAM's gap model is ``GAP = KOPEN = 1 - FVEG``.  The
``opt_rad = 1`` branch (crown geometry: ``ACOS``/``ATAN``/``TAN``/``COS``,
``parameters%RC``/``HVT``/``HVB``/``DEN``) and the ``opt_rad = 2`` branch are
dead and are deliberately NOT transcribed; :func:`twostream` asserts them off.

``opt_alb = 2`` -- ALBEDO calls SNOWALB_CLASS.  SNOWALB_BATS is dead and is
deliberately NOT transcribed; :func:`albedo` asserts it off.

Precision contract
------------------
``kind_phys == kind(1.0)`` is FP32.  Every arithmetic boundary below is
``np.float32``; every expression is grouped exactly as Fortran's
left-to-right, precedence-respecting evaluation orders it, because at
``-O0 -ffp-contract=off`` gfortran neither reassociates nor contracts.

``EXP``/``LOG``/``**`` are glibc calls in the oracle, so they route through
:mod:`gpuwm.core.noahmp_libm` (glibc 2.39 ``expf``/``logf``/``powf``), never
through numpy or ``math``.  ``SQRT`` is IEEE-correctly-rounded and maps to
``np.sqrt`` on float32, matching the ``sqrtss`` gfortran emits inline.

Acceptance: ``tests/test_noahmp_radiation.py`` gates every output of every
leaf at max_ulp 0 and bitwise identity against
``gpuwm/data/noahmp/oracle/noahmp-radiation-*.csv``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from gpuwm.core.noahmp_libm import expf, logf, powf

__all__ = [
    "TFRZ",
    "snow_age",
    "snowalb_class",
    "groundalb",
    "surrad",
    "twostream",
    "albedo",
    "RadiationOptions",
    "PINNED_OPTIONS",
]

F = np.float32

#: module_sf_noahmplsm.F:207 -- REAL, PARAMETER :: TFRZ = 273.16
TFRZ = F(273.16)

_MPE_ALBEDO = F(1.0e-06)      # ALBEDO's local MPE
_SIGMA_FLOOR = F(1.0e-6)      # TWOSTREAM's |SIGMA| clamp


class RadiationOptions:
    """The two Noah-MP options this subsystem branches on."""

    __slots__ = ("opt_rad", "opt_alb")

    def __init__(self, opt_rad: int, opt_alb: int) -> None:
        self.opt_rad = int(opt_rad)
        self.opt_alb = int(opt_alb)


#: The WRF Registry defaults (Registry/Registry.EM_COMMON lines 2686-2687).
PINNED_OPTIONS = RadiationOptions(opt_rad=3, opt_alb=2)


# --------------------------------------------------------------------------
# scalar helpers -- named for what the Fortran writes, so the transcription
# below can be read against the source line by line
# --------------------------------------------------------------------------
def _exp(x) -> np.float32:
    return F(expf(float(F(x))))


def _log(x) -> np.float32:
    return F(logf(float(F(x))))


def _pow(x, y) -> np.float32:
    return F(powf(float(F(x)), float(F(y))))


def _sqrt(x) -> np.float32:
    return F(np.sqrt(F(x)))


def _fsign(a, b) -> np.float32:
    """Fortran ``SIGN(A, B)`` as gfortran implements it: IEEE ``copysign``.

    gfortran lowers SIGN on REAL to a sign-bit splice, so the sign of a
    negative zero is honoured.  Written explicitly rather than via numpy so
    the -0.0 case is unambiguous.
    """
    return F(math.copysign(float(F(a)), float(F(b))))


# --------------------------------------------------------------------------
# SNOW_AGE -- module_sf_noahmplsm.F:3119-3167
# --------------------------------------------------------------------------
def snow_age(tau0, grain_growth, extra_growth, dirt_soot, swemx,
             dt, tg, sneqvo, sneqv, tauss):
    """Return ``(tauss_out, fage)``.

    ``tauss`` is INTENT(INOUT) in the Fortran; it is returned rather than
    mutated.  Yang et al. (1997) eqns 10a-10d.
    """
    dt = F(dt); tg = F(tg); sneqvo = F(sneqvo); sneqv = F(sneqv)
    tauss = F(tauss)
    tau0 = F(tau0); grain_growth = F(grain_growth)
    extra_growth = F(extra_growth); dirt_soot = F(dirt_soot); swemx = F(swemx)

    if sneqv <= F(0.0):
        tauss = F(0.0)
    else:
        dela0 = F(dt / tau0)
        arg = F(grain_growth * F(F(F(1.0) / TFRZ) - F(F(1.0) / tg)))
        age1 = _exp(arg)
        age2 = _exp(min(F(0.0), F(extra_growth * arg)))
        age3 = dirt_soot
        tage = F(F(age1 + age2) + age3)
        dela = F(dela0 * tage)
        dels = F(max(F(0.0), F(sneqv - sneqvo)) / swemx)
        sge = F(F(tauss + dela) * F(F(1.0) - dels))
        tauss = max(F(0.0), sge)

    fage = F(tauss / F(tauss + F(1.0)))
    return tauss, fage


# --------------------------------------------------------------------------
# SNOWALB_CLASS -- module_sf_noahmplsm.F:3226-3275
# --------------------------------------------------------------------------
def snowalb_class(swemx, nband, qsnow, dt, albold):
    """Return ``(alb, albsnd, albsni)`` with the two albedo arrays as tuples.

    ``ILOC``/``JLOC`` carry no arithmetic in this routine and are omitted.
    ``NBAND`` only sizes the zeroing of ALBSND/ALBSNI, both elements of which
    are then assigned unconditionally, so it cannot affect the result; it is
    accepted and validated rather than silently dropped.
    """
    if int(nband) != 2:
        raise ValueError(
            "NBAND != 2 is unreachable: ALBEDO sets NBAND = 2 unconditionally "
            "(module_sf_noahmplsm.F:2822)"
        )
    swemx = F(swemx); qsnow = F(qsnow); dt = F(dt); albold = F(albold)

    alb = F(F(0.55) + F(F(albold - F(0.55)) * _exp(F(F(F(-0.01) * dt) / F(3600.0)))))
    if qsnow > F(0.0):
        cap = F(swemx / dt)
        alb = F(alb + F(F(min(qsnow, cap) * F(F(0.84) - alb)) / cap))

    return alb, (alb, alb), (alb, alb)


# --------------------------------------------------------------------------
# GROUNDALB -- module_sf_noahmplsm.F:3279-3332
# --------------------------------------------------------------------------
def groundalb(albsat, albdry, alblak, nsoil, nband, ice, ist,
              fsno, smc, albsnd, albsni, cosz, tg):
    """Return ``(albgrd, albgri)`` as 2-tuples.

    ``ICE`` is declared INTENT(IN) but never referenced in the body of the
    pinned source; it is accepted so the signature matches, and the mutation
    study records it as a provably dead argument.  ``NSOIL`` only sizes the
    SMC dummy -- only ``SMC(1)`` is read.
    """
    if int(nband) != 2:
        raise ValueError("NBAND != 2 is unreachable (ALBEDO sets NBAND = 2)")
    del ice, nsoil  # dead in the pinned source; see the docstring
    fsno = F(fsno); cosz = F(cosz); tg = F(tg)
    smc1 = F(smc[0])

    albgrd = [F(0.0), F(0.0)]
    albgri = [F(0.0), F(0.0)]
    for ib in range(2):
        inc = max(F(F(0.11) - F(F(0.40) * smc1)), F(0.0))
        if int(ist) == 1:                       # soil
            albsod = min(F(F(albsat[ib]) + inc), F(albdry[ib]))
            albsoi = albsod
        elif tg > TFRZ:                         # unfrozen lake, wetland
            albsod = F(F(0.06) / F(_pow(max(F(0.01), cosz), F(1.7)) + F(0.15)))
            albsoi = F(0.06)
        else:                                   # frozen lake, wetland
            albsod = F(alblak[ib])
            albsoi = albsod

        albgrd[ib] = F(F(albsod * F(F(1.0) - fsno)) + F(F(albsnd[ib]) * fsno))
        albgri[ib] = F(F(albsoi * F(F(1.0) - fsno)) + F(F(albsni[ib]) * fsno))

    return tuple(albgrd), tuple(albgri)


# --------------------------------------------------------------------------
# SURRAD -- module_sf_noahmplsm.F:2994-3115
# --------------------------------------------------------------------------
def surrad(mpe, fsun, fsha, elai, vai, laisun, laisha,
           solad, solai, fabd, fabi, ftdd, ftid, ftii,
           albgrd, albgri, albd, albi, frevd, frevi, fregd, fregi):
    """Return ``(parsun, parsha, sav, sag, fsa, fsr, fsrv, fsrg)``.

    No transcendental appears in this routine; it is pure FP32 arithmetic.
    """
    mpe = F(mpe); fsun = F(fsun); fsha = F(fsha)
    elai = F(elai); vai = F(vai); laisun = F(laisun); laisha = F(laisha)

    sag = F(0.0)
    sav = F(0.0)
    fsa = F(0.0)
    cad = [F(0.0), F(0.0)]
    cai = [F(0.0), F(0.0)]

    for ib in range(2):
        cad[ib] = F(F(solad[ib]) * F(fabd[ib]))
        cai[ib] = F(F(solai[ib]) * F(fabi[ib]))
        sav = F(F(sav + cad[ib]) + cai[ib])
        fsa = F(F(fsa + cad[ib]) + cai[ib])

        trd = F(F(solad[ib]) * F(ftdd[ib]))
        tri = F(F(F(solad[ib]) * F(ftid[ib])) + F(F(solai[ib]) * F(ftii[ib])))

        abso = F(F(trd * F(F(1.0) - F(albgrd[ib])))
                 + F(tri * F(F(1.0) - F(albgri[ib]))))
        sag = F(sag + abso)
        fsa = F(fsa + abso)

    laifra = F(elai / max(vai, mpe))
    if fsun > F(0.0):
        parsun = F(F(F(cad[0] + F(fsun * cai[0])) * laifra) / max(laisun, mpe))
        parsha = F(F(F(fsha * cai[0]) * laifra) / max(laisha, mpe))
    else:
        parsun = F(0.0)
        parsha = F(F(F(cad[0] + cai[0]) * laifra) / max(laisha, mpe))

    rvis = F(F(F(albd[0]) * F(solad[0])) + F(F(albi[0]) * F(solai[0])))
    rnir = F(F(F(albd[1]) * F(solad[1])) + F(F(albi[1]) * F(solai[1])))
    fsr = F(rvis + rnir)

    fsrv = F(F(F(F(F(frevd[0]) * F(solad[0])) + F(F(frevi[0]) * F(solai[0])))
               + F(F(frevd[1]) * F(solad[1]))) + F(F(frevi[1]) * F(solai[1])))
    fsrg = F(F(F(F(F(fregd[0]) * F(solad[0])) + F(F(fregi[0]) * F(solai[0])))
               + F(F(fregd[1]) * F(solad[1]))) + F(F(fregi[1]) * F(solai[1])))

    return parsun, parsha, sav, sag, fsa, fsr, fsrv, fsrg


# --------------------------------------------------------------------------
# TWOSTREAM -- module_sf_noahmplsm.F:3336-3574, OPT_RAD = 3 only
# --------------------------------------------------------------------------
def twostream(xl, omegas, betads, betais,
              ib, ic, cosz, vai, fwet, t, albgrd, albgri, rho, tau, fveg,
              fab, fre, ftd, fti, gdir, frev, freg, bgap, wgap,
              options: RadiationOptions = PINNED_OPTIONS):
    """Dickinson (1983) / Sellers (1985) two-stream, ``OPT_RAD = 3`` only.

    ``fab``/``fre``/``ftd``/``fti``/``frev``/``freg`` are INTENT(INOUT)
    2-vectors: only element ``ib`` is written, the other passes through.
    ``bgap``/``wgap`` are INTENT(INOUT) and are written *only* under
    ``OPT_RAD = 1``, so under the pinned identity they pass straight through.

    Returns ``(fab, fre, ftd, fti, gdir, frev, freg, bgap, wgap)`` with the
    six arrays as 2-tuples.  ``VEGTYP``, ``IST``, ``ILOC`` and ``JLOC`` are
    dummy arguments the pinned body never references and are not accepted.
    """
    if options.opt_rad != 3:
        raise NotImplementedError(
            "only OPT_RAD = 3 is transcribed; OPT_RAD = 1 (crown geometry) "
            "and OPT_RAD = 2 are dead under the pinned Registry defaults"
        )
    ib0 = int(ib) - 1
    cosz = F(cosz); vai = F(vai); fwet = F(fwet); t = F(t); fveg = F(fveg)
    fab = [F(fab[0]), F(fab[1])]
    fre = [F(fre[0]), F(fre[1])]
    ftd = [F(ftd[0]), F(ftd[1])]
    fti = [F(fti[0]), F(fti[1])]
    frev = [F(frev[0]), F(frev[1])]
    freg = [F(freg[0]), F(freg[1])]
    bgap = F(bgap); wgap = F(wgap)

    # --- within/between canopy gaps -------------------------------------
    if vai == F(0.0):
        gap = F(1.0)
        kopen = F(1.0)
    else:
        gap = F(F(1.0) - fveg)
        kopen = F(F(1.0) - fveg)

    # --- two-stream parameters ------------------------------------------
    coszi = max(F(0.001), cosz)
    chil = min(max(F(xl), F(-0.4)), F(0.6))
    if abs(chil) <= F(0.01):
        chil = F(0.01)
    phi1 = F(F(F(0.5) - F(F(0.633) * chil)) - F(F(F(0.330) * chil) * chil))
    phi2 = F(F(0.877) * F(F(1.0) - F(F(2.0) * phi1)))
    gdir = F(phi1 + F(phi2 * coszi))
    ext = F(gdir / coszi)
    avmu = F(F(F(1.0) - F(F(phi1 / phi2) * _log(F(F(phi1 + phi2) / phi1))))
             / phi2)
    omegal = F(F(rho[ib0]) + F(tau[ib0]))
    tmp0 = F(gdir + F(phi2 * coszi))
    tmp1 = F(phi1 * coszi)
    asu = F(F(F(F(F(0.5) * omegal) * gdir) / tmp0)
            * F(F(1.0) - F(F(tmp1 / tmp0) * _log(F(F(tmp1 + tmp0) / tmp1)))))
    betadl = F(F(F(F(1.0) + F(avmu * ext)) / F(F(omegal * avmu) * ext)) * asu)
    q = F(F(F(F(1.0) + chil) / F(2.0)) * F(F(F(1.0) + chil) / F(2.0)))
    betail = F(F(F(0.5) * F(F(F(rho[ib0]) + F(tau[ib0]))
                            + F(F(F(rho[ib0]) - F(tau[ib0])) * q))) / omegal)

    # --- adjust for intercepted snow ------------------------------------
    if t > TFRZ:
        tmp0 = omegal
        tmp1 = betadl
        tmp2 = betail
    else:
        tmp0 = F(F(F(F(1.0) - fwet) * omegal) + F(fwet * F(omegas[ib0])))
        tmp1 = F(F(F(F(F(F(1.0) - fwet) * omegal) * betadl)
                   + F(F(fwet * F(omegas[ib0])) * F(betads))) / tmp0)
        tmp2 = F(F(F(F(F(F(1.0) - fwet) * omegal) * betail)
                   + F(F(fwet * F(omegas[ib0])) * F(betais))) / tmp0)

    omega = tmp0
    betad = tmp1
    betai = tmp2

    # --- absorbed / reflected / transmitted per unit incoming -----------
    b = F(F(F(1.0) - omega) + F(omega * betai))
    c = F(omega * betai)
    tmp0 = F(avmu * ext)
    d = F(F(tmp0 * omega) * betad)
    f = F(F(tmp0 * omega) * F(F(1.0) - betad))
    tmp1 = F(F(b * b) - F(c * c))
    h = F(_sqrt(tmp1) / avmu)
    sigma = F(F(tmp0 * tmp0) - tmp1)
    if abs(sigma) < _SIGMA_FLOOR:
        sigma = _fsign(_SIGMA_FLOOR, sigma)
    p1 = F(b + F(avmu * h))
    p2 = F(b - F(avmu * h))
    p3 = F(b + tmp0)
    p4 = F(b - tmp0)
    s1 = _exp(F(F(-h) * vai))
    s2 = _exp(F(F(-ext) * vai))
    if int(ic) == 0:
        u1 = F(b - F(c / F(albgrd[ib0])))
        u2 = F(b - F(c * F(albgrd[ib0])))
        u3 = F(f + F(c * F(albgrd[ib0])))
    else:
        u1 = F(b - F(c / F(albgri[ib0])))
        u2 = F(b - F(c * F(albgri[ib0])))
        u3 = F(f + F(c * F(albgri[ib0])))
    tmp2 = F(u1 - F(avmu * h))
    tmp3 = F(u1 + F(avmu * h))
    d1 = F(F(F(p1 * tmp2) / s1) - F(F(p2 * tmp3) * s1))
    tmp4 = F(u2 + F(avmu * h))
    tmp5 = F(u2 - F(avmu * h))
    d2 = F(F(tmp4 / s1) - F(tmp5 * s1))
    h1 = F(F(F(-d) * p4) - F(c * f))
    tmp6 = F(d - F(F(h1 * p3) / sigma))
    tmp7 = F(F(F(d - c) - F(F(h1 / sigma) * F(u1 + tmp0))) * s2)
    h2 = F(F(F(F(tmp6 * tmp2) / s1) - F(p2 * tmp7)) / d1)
    h3 = F(-F(F(F(F(tmp6 * tmp3) * s1) - F(p1 * tmp7)) / d1))
    h4 = F(F(F(-f) * p3) - F(c * d))
    tmp8 = F(h4 / sigma)
    tmp9 = F(F(u3 - F(tmp8 * F(u2 - tmp0))) * s2)
    h5 = F(-F(F(F(F(tmp8 * tmp4) / s1) + tmp9) / d2))
    h6 = F(F(F(F(tmp8 * tmp5) * s1) + tmp9) / d2)
    h7 = F(F(c * tmp2) / F(d1 * s1))
    h8 = F(F(F(F(-c) * tmp3) * s1) / d1)
    h9 = F(tmp4 / F(d2 * s1))
    h10 = F(F(F(-tmp5) * s1) / d2)

    # --- downward direct and diffuse below vegetation -------------------
    if int(ic) == 0:
        ftds = F(F(s2 * F(F(1.0) - gap)) + gap)
        ftis = F(F(F(F(F(h4 * s2) / sigma) + F(h5 * s1)) + F(h6 / s1))
                 * F(F(1.0) - gap))
    else:
        ftds = F(0.0)
        ftis = F(F(F(F(h9 * s1) + F(h10 / s1)) * F(F(1.0) - kopen)) + kopen)
    ftd[ib0] = ftds
    fti[ib0] = ftis

    # --- flux reflected by the surface ----------------------------------
    if int(ic) == 0:
        fres = F(F(F(F(F(h1 / sigma) + h2) + h3) * F(F(1.0) - gap))
                 + F(F(albgrd[ib0]) * gap))
        freveg = F(F(F(F(h1 / sigma) + h2) + h3) * F(F(1.0) - gap))
        frebar = F(F(albgrd[ib0]) * gap)
    else:
        fres = F(F(F(h7 + h8) * F(F(1.0) - kopen)) + F(F(albgri[ib0]) * kopen))
        freveg = F(F(F(h7 + h8) * F(F(1.0) - kopen)) + F(F(albgri[ib0]) * kopen))
        frebar = F(0.0)
    fre[ib0] = fres
    frev[ib0] = freveg
    freg[ib0] = frebar

    # --- flux absorbed by vegetation ------------------------------------
    fab[ib0] = F(F(F(F(1.0) - fre[ib0])
                   - F(F(F(1.0) - F(albgrd[ib0])) * ftd[ib0]))
                 - F(F(F(1.0) - F(albgri[ib0])) * fti[ib0]))

    return (tuple(fab), tuple(fre), tuple(ftd), tuple(fti), gdir,
            tuple(frev), tuple(freg), bgap, wgap)


# --------------------------------------------------------------------------
# ALBEDO -- module_sf_noahmplsm.F:2810-2990
# --------------------------------------------------------------------------
def albedo(tau0, grain_growth, extra_growth, dirt_soot, swemx,
           albsat, albdry, alblak, rhol, rhos, taul, taus, xl,
           omegas, betads, betais,
           vegtyp, ist, ice, nsoil,
           dt, cosz, fage, elai, esai, tg, tv, snowh, fsno, fwet,
           smc, sneqvo, sneqv, qsnow, fveg, albold, tauss,
           frevd_in=(0.0, 0.0), frevi_in=(0.0, 0.0),
           fregd_in=(0.0, 0.0), fregi_in=(0.0, 0.0),
           options: RadiationOptions = PINNED_OPTIONS):
    """Surface albedos and per-unit-flux canopy fluxes.

    Returns a dict keyed exactly like the oracle CSV's output columns.

    ``VEGTYP``, ``SNOWH``, ``ICE``, ``ILOC`` and ``JLOC`` reach only dead
    arguments of the callees under the pinned identity; they are accepted so
    the signature matches the Fortran and the mutation study can record them.
    ``FAGE`` is a dummy argument with no INTENT: it is written by SNOW_AGE on
    the daytime path and left untouched when ``COSZ <= 0``, so its entry
    value is part of the contract and is threaded through.

    FREVD/FREVI/FREGD/FREGI are the one real defect in the pinned source that
    this transcription has to model rather than fix.  ALBEDO's initialisation
    loop (lines 2829-2842) zeroes every other INTENT(OUT) array but not those
    four, so when ``COSZ <= 0`` and the routine jumps to label 100 they are
    returned *undefined* -- in the gfortran build, exactly the bytes the
    caller passed in, because an INTENT(OUT) array dummy is passed by
    reference with no copy.  ``frevd_in``/``frevi_in``/``fregd_in``/
    ``fregi_in`` carry those entry values so the night-time contract can be
    pinned bit-for-bit instead of being papered over with zeros.  In WRF the
    caller (RADIATION) passes uninitialised locals, so at night SURRAD reads
    genuinely undefined memory; it is benign only because SOLAD and SOLAI are
    zero there, which makes FSRV and FSRG zero regardless.
    """
    if options.opt_alb != 2:
        raise NotImplementedError(
            "only OPT_ALB = 2 (CLASS) is transcribed; SNOWALB_BATS is dead "
            "under the pinned Registry defaults"
        )
    del vegtyp, snowh  # dead under the pinned identity; see the docstring

    nband = 2
    mpe = _MPE_ALBEDO
    cosz = F(cosz)
    elai = F(elai); esai = F(esai)
    fage = F(fage)
    albold = F(albold); tauss = F(tauss)

    bgap = F(0.0)
    wgap = F(0.0)
    albd = [F(0.0), F(0.0)]
    albi = [F(0.0), F(0.0)]
    albgrd = [F(0.0), F(0.0)]
    albgri = [F(0.0), F(0.0)]
    albsnd = [F(0.0), F(0.0)]
    albsni = [F(0.0), F(0.0)]
    fabd = [F(0.0), F(0.0)]
    fabi = [F(0.0), F(0.0)]
    ftdd = [F(0.0), F(0.0)]
    ftid = [F(0.0), F(0.0)]
    ftii = [F(0.0), F(0.0)]
    fsun = F(0.0)
    # Not zeroed by ALBEDO's init loop: undefined on the COSZ <= 0 exit, and
    # in the gfortran build that means "whatever the caller passed".
    frevd = [F(frevd_in[0]), F(frevd_in[1])]
    frevi = [F(frevi_in[0]), F(frevi_in[1])]
    fregd = [F(fregd_in[0]), F(fregd_in[1])]
    fregi = [F(fregi_in[0]), F(fregi_in[1])]
    # FTDI and GDIR are ALBEDO locals: never read before the TWOSTREAM loop
    # assigns them, and they do not escape.
    ftdi = [F(0.0), F(0.0)]
    gdir = F(0.0)

    if cosz > F(0.0):
        rho = [F(0.0), F(0.0)]
        tau = [F(0.0), F(0.0)]
        vai = F(0.0)
        for ib in range(nband):
            vai = F(elai + esai)
            wl = F(elai / max(vai, mpe))
            ws = F(esai / max(vai, mpe))
            rho[ib] = max(F(F(F(rhol[ib]) * wl) + F(F(rhos[ib]) * ws)), mpe)
            tau[ib] = max(F(F(F(taul[ib]) * wl) + F(F(taus[ib]) * ws)), mpe)

        tauss, fage = snow_age(tau0, grain_growth, extra_growth, dirt_soot,
                               swemx, dt, tg, sneqvo, sneqv, tauss)

        alb, albsnd_t, albsni_t = snowalb_class(swemx, nband, qsnow, dt, albold)
        albsnd = list(albsnd_t)
        albsni = list(albsni_t)
        albold = alb

        albgrd_t, albgri_t = groundalb(albsat, albdry, alblak, nsoil, nband,
                                       ice=ice, ist=ist, fsno=fsno, smc=smc,
                                       albsnd=albsnd, albsni=albsni,
                                       cosz=cosz, tg=tg)
        albgrd = list(albgrd_t)
        albgri = list(albgri_t)

        for ib in range(1, nband + 1):
            fabd_t, albd_t, ftdd_t, ftid_t, gdir, frevd_t, fregd_t, bgap, wgap = \
                twostream(xl, omegas, betads, betais,
                          ib=ib, ic=0, cosz=cosz, vai=vai, fwet=fwet, t=tv,
                          albgrd=albgrd, albgri=albgri, rho=rho, tau=tau,
                          fveg=fveg,
                          fab=fabd, fre=albd, ftd=ftdd, fti=ftid, gdir=gdir,
                          frev=frevd, freg=fregd, bgap=bgap, wgap=wgap,
                          options=options)
            fabd, albd, ftdd, ftid = (list(fabd_t), list(albd_t),
                                      list(ftdd_t), list(ftid_t))
            frevd, fregd = list(frevd_t), list(fregd_t)

            fabi_t, albi_t, ftdi_t, ftii_t, gdir, frevi_t, fregi_t, bgap, wgap = \
                twostream(xl, omegas, betads, betais,
                          ib=ib, ic=1, cosz=cosz, vai=vai, fwet=fwet, t=tv,
                          albgrd=albgrd, albgri=albgri, rho=rho, tau=tau,
                          fveg=fveg,
                          fab=fabi, fre=albi, ftd=ftdi, fti=ftii, gdir=gdir,
                          frev=frevi, freg=fregi, bgap=bgap, wgap=wgap,
                          options=options)
            fabi, albi, ftdi, ftii = (list(fabi_t), list(albi_t),
                                      list(ftdi_t), list(ftii_t))
            frevi, fregi = list(frevi_t), list(fregi_t)

        ext = F(F(gdir / cosz) * _sqrt(F(F(F(1.0) - rho[0]) - tau[0])))
        fsun = F(F(F(1.0) - _exp(F(F(-ext) * vai))) / max(F(ext * vai), mpe))
        ext = fsun
        wl = F(0.0) if ext < F(0.01) else ext
        fsun = wl

    return {
        "fage": fage, "albold": albold, "tauss": tauss, "fsun": fsun,
        "bgap": bgap, "wgap": wgap,
        "albgrd": tuple(albgrd), "albgri": tuple(albgri),
        "albd": tuple(albd), "albi": tuple(albi),
        "fabd": tuple(fabd), "fabi": tuple(fabi),
        "ftdd": tuple(ftdd), "ftid": tuple(ftid), "ftii": tuple(ftii),
        "frevd": tuple(frevd), "frevi": tuple(frevi),
        "fregd": tuple(fregd), "fregi": tuple(fregi),
        "albsnd": tuple(albsnd), "albsni": tuple(albsni),
    }
