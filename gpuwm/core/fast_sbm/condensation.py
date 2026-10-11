"""Warm diffusional growth and evaporation of drops (WRF FSBM-2, ``ONECOND1``).

Host float64 transcription of the routines WRF's fast spectral bin
microphysics uses to grow and evaporate drops by vapour diffusion, gated
against WRF's own Fortran promoted to double precision (the O64 oracle
build, fast-sbm design 5.1 and 5.5).  It is a verification reference, not a
runtime data path: nothing here imports cupy or is selected by a model
configuration.

Source and attribution
----------------------
WRF v4.8.0 ``phys/module_mp_fast_sbm.F``, 9,053 lines, SHA-256 of the LF
content ``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``
(identical in v4.6.1 and v4.7.1).  WRF is developed at NCAR, operated by UCAR,
and placed in the public domain; UCAR's notice travels in the repository
``NOTICE`` and in ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The scheme is the
Hebrew University Cloud Model spectral bin microphysics formulated and coded
by Alexander Khain; FSBM-2 in WRF was coded and is maintained by Jacob Shpund
with the contributors the package docstring names, and WRF asks users to cite
Shpund et al. (2019a) (fast-sbm design, section 8).

Routines transcribed, with their lines:

* ``module_mp_SBM_Auxiliary`` parameters, :1704-1712 (``ISIGN_KO_1``,
  ``ISIGN_KO_2``, ``ISIGN_3POINT``, ``COEFF_REMAPING``, ``VENTPL_MAX``,
  ``RW_PW_MIN``, ``RI_PI_MIN``, ``RW_PW_RI_PI_MIN``, ``RATIO_ICEW_MIN``).
* ``POLYSVP``, :1716-1742 -> :func:`polysvp`.
* ``JERRATE_KS``, :1744-1921, for drops (``ID = 1``, ``IN = 1``, so every bin
  takes the water branch of ``nskin`` at :1788-1794) -> :func:`jerrate_drops`.
* ``JERTIMESC_KS``, :1925-1964 -> :func:`jertimesc`.
* ``JERSUPSAT_KS``, :1966-2330, every branch -> :func:`jersupsat`.
* ``JERDFUN_KS``, :2334-2442 -> :func:`jerdfun`.
* ``JERNEWF_KS``, :2444-2757 -> :func:`jernewf`.
* ``JERDFUN_NEW_KS``, :2759-2834 -> :func:`jerdfun_new`.
* ``ONECOND1``, :6748-7143 -> :func:`onecond1`; its ``DATA`` at :6830-6845.

Published sources of the method: diffusional growth with ventilation and the
supersaturation relaxation solved analytically over each substep, Khain and
Sednev (1996) and Khain et al. (2004); the growth-rate coefficients
(diffusivity, thermal conductivity, ventilation), Pruppacher and Klett (1997,
chapter 13); the remap of grown masses onto the fixed mass grid conserving
number and mass, Kovetz and Olund (1969).  Full references are in section 9
of the fast-sbm design.  Nothing here comes from any other scheme's source.

Transcription rules (fast-sbm design 4.2)
-----------------------------------------
* Float64 throughout, which is what the O64 build (``-freal-4-real-8``)
  holds: every ``REAL`` and ``REAL(KIND=4)`` of the source is a double there,
  and every literal enters as the double nearest its decimal.  The literals
  are written here exactly as WRF writes them (``3.141593``, ``12.566372``,
  ``0.0066667``, ``1.0001e-11`` and so on).
* Expressions keep Fortran's left-to-right order within one precedence level
  and the source's parentheses, so each rounding happens where the O64 build
  rounds.  Transcendentals use the C library through :mod:`math`, as the
  Fortran does through libm; the gate's 1e-10 relative tolerance covers the
  last-place differences between two C libraries.
* Sums keep Fortran's sequential order (explicit loops, never ``np.sum``).
* Arrays are zero-based and keep Fortran index order: ``a[k - 1]`` is
  ``A(K)``.  Inputs are never modified; every routine returns new arrays.
* Each ``wrf_error_fatal`` on these paths becomes :class:`FsbmRefusal`
  whose ``condition`` starts with WRF's own message text.  Where WRF's
  behaviour is undefined (a ``SELECT CASE`` with no match, a division by zero
  or an overflow that the Fortran would carry on as Inf or NaN) the port
  refuses by name instead.

What the WRF code does that a reader might not expect, all reproduced:

* ``ONECOND1``'s substep loop (``GOTO 56``) runs exactly once:
  ``TIMENEW`` becomes ``DT`` on the first pass.
* The spectrum ``ONECOND1`` returns comes from ``JERDFUN_NEW_KS`` applied to
  the entry spectrum with the accumulated ``B11_MY * D1N``; the
  ``JERDFUN_KS`` call inside the loop only feeds the in-loop temperature-jump
  check.
* If ``JERSUPSAT_KS`` clears ``ISYM1`` (all rates at most 1e-10), neither
  remap runs and ``ONECOND1`` returns the caller's ``PSI1`` array as the new
  spectrum.  ``FAST_SBM`` passes ``FF1IN``, a copy of the entry spectrum
  (:4683, :4718), which is this port's default; the sweep harness passes
  zeros.
* ``ONECOND1`` hands ``DYN1 = DYN2 = 0`` to ``JERSUPSAT_KS`` and zero ice
  rates, so only the water branches run on the warm path; with no forcing
  the relaxation cannot change the sign of the supersaturation, so the two
  sign-flip checks (:6982-6985, :7026-7029) cannot fire on well-formed input.
  They are kept, as is the ``ONECOND1-out`` check (:7119-7134), which cannot
  fire unless the in-loop check (limit 3 K) has already.
* ``JERNEWF_KS``'s 3-point step writes ``PSI(K) = FI(K)*RR(K)`` over the
  Kovetz-Olund result for a bin whose mass did not change, and its
  smoothing test is ``(a .AND. b .AND. c) .OR. d`` by Fortran precedence.
* The tail remap over bins 6-12 merges a bin into the one below when it
  holds less than ``COEFF_REMAPING`` of that bin's number; that conserves
  number, not mass.
* In the Kovetz-Olund step a bin whose mass moved by less than 1e-16 g is
  assigned, ``PSINEW(K) = FI(K)*RR(K)`` (:2541-2543), not accumulated, so it
  drops what lower bins already moved into it.  Under uniform
  supersaturation the mass change grows with bin size, so no lower bin
  moves when bin K does not, and the rule changes nothing.

Findings in the mixed-phase branches of ``JERSUPSAT_KS``, reproduced here and
left for the ice milestone (M3) to decide, because only ``ONECOND2`` and
``ONECOND3`` reach them (``ONECOND1`` passes zero ice rates), and all five
callers in WRF pass ``DYN1 = DYN2 = 0``.  Fast-sbm design 3.7 lists them by
these ids with the other ice-path findings:

* S1.  Branches 8 and 9 (:2252-2256, :2286-2290) expand
  ``ALFA = sqrt(PI**2 + X)`` as ``PI*(1 + (X/PI)/2 - (X/PI)**2/8)``, where the
  expansion variable is ``X/PI**2`` (branch 9 likewise with ``RW``): WRF has
  ``PI + X/2`` where ``PI + X/(2 PI)`` is right, and the terms do not share
  units.  The first-order terms of ``BETA`` and ``GAMA`` are right; their
  second-order terms carry the same slip.  On the oracle's own branch 8
  case every output is 2.5e-5 from the exact solution, all of it through
  ``ALFA``.
* S2.  Branch 7's clamp (:2202) replaces any ``GAMA`` below 1e-10 with
  5e-14.  The physical rates give ``G2 > 0``, so ``GAMA`` is negative
  whenever both phases are present and the clamp fires every time branch 7
  (``|GAMA DT| <= 1e-6``) runs: the slow mode stops decaying over the step
  (error of order ``|GAMA| DT``, at most 1e-6).
* S3.  Branches 7-9 carry ``DEL1INT`` to first order in ``GAMA`` only, where
  ``DEL2INT`` has the second-order term ``A5DEL2INT``; the difference is
  proportional to ``G31``, which is zero without forcing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import (AA1_MY, AA2_MY, AL1, AL2, BB1_MY, BB2_MY, COL, NKR,
                        FsbmRefusal)

# ---------------------------------------------------------------------------
# module_mp_SBM_Auxiliary parameters (:1704-1712)
# ---------------------------------------------------------------------------

#: Kovetz-Olund fallback switches after a negative spectrum; both 0, and the
#: fallback call is commented out in the source (:2413-2417, :2819-2823).
ISIGN_KO_1 = 0
ISIGN_KO_2 = 0
#: The 3-point remap is on (:1704); ``JERDFUN_KS`` and ``JERDFUN_NEW_KS``
#: pass it to ``JERNEWF_KS`` as ``I3POINT``.
ISIGN_3POINT = 1
#: Tail remap merge criterion over bins 6-12 (:1706).
COEFF_REMAPING = 0.0066667
#: Ventilation factor cap (:1707).
VENTPL_MAX = 5.0
#: Rate thresholds of ``JERSUPSAT_KS`` (:1709-1712).
RW_PW_MIN = 1.0e-10
RI_PI_MIN = 1.0e-10
RW_PW_RI_PI_MIN = 1.0e-10
RATIO_ICEW_MIN = 1.0e-4

#: Bins of the drop tail remap in ``JERNEWF_KS`` (:2465).
KRDROP_REMAPING_MIN = 6
KRDROP_REMAPING_MAX = 12

# ---------------------------------------------------------------------------
# ONECOND1 DATA (:6830-6845) and fixed locals (:6850-6877)
# ---------------------------------------------------------------------------

#: ``EPSDEL``: the sign-flip checks fire only when the new supersaturation
#: exceeds this in magnitude (:6843).
EPSDEL = 0.1e-03
#: ``GAM`` and ``CWHUCM``, ``XRAD``: ``B6 = CWHUCM*GAM - XRAD`` is zero (:6831,
#: :6872-6874), so ``QW`` and ``QI`` are zero and unused.
GAM = 1.0e-4
CWHUCM = 0.0
XRAD = 0.0
#: ``ONECOND1-in`` temperature-jump limit, K (:7061).
ONECOND1_IN_LIMIT = 3.0
#: ``ONECOND1-out`` temperature-jump limit, K (:7119).
ONECOND1_OUT_LIMIT = 5.0

# ---------------------------------------------------------------------------
# JERSUPSAT_KS branch codes.  The numbering is the oracle harness's
# classification (tools/fast_sbm_wrf480_oracle/run_sweeps.F90,
# supsat_branch), so a host result compares with a sweep record directly.
# ---------------------------------------------------------------------------

#: Every rate at most ``RW_PW_RI_PI_MIN``: all rates zeroed, no sink
#: (:1995-2007, then :2061-2068).
SUPSAT_ALL_BELOW_MIN = 0
#: Neither phase left a sink after the per-phase tests (:2061-2068).
SUPSAT_NO_SINK = 1
#: Water only, ``|RW DT| > 1e-6``: exponential solution (:2078-2090).
SUPSAT_WATER = 2
#: Water only, ``|RW DT| <= 1e-6``: fifth-order series ``EXPM1`` (:2094-2104).
SUPSAT_WATER_SERIES = 3
#: Ice only, ``|PI DT| > 1e-6`` (:2115-2129).
SUPSAT_ICE = 4
#: Ice only, ``|PI DT| <= 1e-6`` (:2133-2143).
SUPSAT_ICE_SERIES = 5
#: Both phases, ``|GAMA DT| > 1e-6``: the two-exponential solution
#: (:2151-2193).
SUPSAT_MIXED = 6
#: Both phases, ``|GAMA DT| <= 1e-6``, both rate ratios above 1e-12, with
#: the ``GAMA`` clamp (:2197-2246).
SUPSAT_MIXED_SERIES = 7
#: Both phases, ``|GAMA DT| <= 1e-6``, ``|RW/RI| <= 1e-12``: ``ALFA``,
#: ``BETA``, ``GAMA`` expanded in ``X/PI`` (:2248-2278).
SUPSAT_MIXED_SMALL_RW = 8
#: Both phases, ``|GAMA DT| <= 1e-6``, ``|RI/RW| <= 1e-12``: expanded in
#: ``X/RW`` (:2282-2312).
SUPSAT_MIXED_SMALL_RI = 9

SUPSAT_BRANCH_NAMES = {
    SUPSAT_ALL_BELOW_MIN: "all rates below RW_PW_RI_PI_MIN",
    SUPSAT_NO_SINK: "no sink in either phase",
    SUPSAT_WATER: "water only",
    SUPSAT_WATER_SERIES: "water only, small-argument series",
    SUPSAT_ICE: "ice only",
    SUPSAT_ICE_SERIES: "ice only, small-argument series",
    SUPSAT_MIXED: "water and ice",
    SUPSAT_MIXED_SERIES: "water and ice, small GAMA*DT",
    SUPSAT_MIXED_SMALL_RW: "water and ice, small GAMA*DT, |RW/RI| <= 1e-12",
    SUPSAT_MIXED_SMALL_RI: "water and ice, small GAMA*DT, |RI/RW| <= 1e-12",
}

#: ``KCOND`` values of ``ONECOND1`` (:6944-6945).
KCOND_EVAPORATION = 10
KCOND_CONDENSATION = 11


# ---------------------------------------------------------------------------
# IEEE helpers.  Python raises where the Fortran would carry Inf or NaN; the
# port refuses by name there (the module docstring says why).
# ---------------------------------------------------------------------------

def _exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def _pow(x: float, y: float, where: str) -> float:
    try:
        return math.pow(x, y)
    except (ValueError, OverflowError) as exc:
        raise FsbmRefusal(
            where, f"{x!r}**{y!r} has no real double value ({exc}); WRF would "
            "carry a NaN or Inf here") from exc


def _vector(a, n: int, name: str, where: str) -> list[float]:
    arr = np.asarray(a, dtype=np.float64)
    if arr.shape != (n,):
        raise FsbmRefusal(where, f"{name} has shape {arr.shape}, expected ({n},)")
    return [float(v) for v in arr]


# ---------------------------------------------------------------------------
# POLYSVP (:1716-1742)
# ---------------------------------------------------------------------------

def polysvp(t: float, itype: int) -> float:
    """Saturation vapour pressure over water (``itype`` 0) or ice (1), dyn/cm2.

    ``POLYSVP`` (:1716-1742): ``AA1_MY*EXP(-BB1_MY/TT)`` or
    ``AA2_MY*EXP(-BB2_MY/TT)``, the same Magnus-type fits ``FAST_SBM`` uses
    (constants :1723-1724, equal to the ``FAST_SBM`` DATA at :4088).  WRF's
    ``SELECT CASE`` leaves the result undefined for any other ``itype``; the
    port refuses it.
    """
    tt = float(t)
    if itype == 0:
        return AA1_MY * _exp(-(BB1_MY / tt))
    if itype == 1:
        return AA2_MY * _exp(-(BB2_MY / tt))
    raise FsbmRefusal("POLYSVP (:1727-1739)",
                      f"ITYPE {itype!r} is neither 0 (liquid) nor 1 (ice); "
                      "WRF leaves POLYSVP undefined")


# ---------------------------------------------------------------------------
# JERRATE_KS for drops (:1744-1921)
# ---------------------------------------------------------------------------

#: ``JERRATE_KS`` constants, CGS (:1768, :1782, :1786, :1816, :1829, :1839).
RV_MY = 461.5e4
CF_MY = 2.4e3
D_MYIN = 0.211
PZERO = 1.013e6
TZERO = 273.15
AL1_MY_WATER = 2.5e10
CONST_4PI = 12.566372
COEFF_VISCOUS = 0.13


def jerrate_drops(xl, t: float, p_cgs: float, v, rlec, ro1bl) -> np.ndarray:
    """Diffusional growth-rate coefficient ``B11_MY`` of each drop bin.

    ``JERRATE_KS`` (:1744-1921) called as ``ONECOND1`` calls it (:6915-6916),
    with ``ID = 1`` and ``IN = 1``, so every bin uses the water saturation
    pressure and latent heat of vaporization (``nskin = 1``, :1788-1794).

    Per bin: vapour diffusivity ``D = 0.211 (PZERO/P) (T/TZERO)**1.94``;
    Schmidt number ``0.13 / D``; Reynolds number
    ``B V (x/rho)**(1/3)`` with ``B = 2 (3/(4 pi))**(1/3) / 0.13``;
    ventilation ``1 + 0.108 (Re**(1/2) Sc**(1/3))**2`` below Re 2.5, else
    ``0.78 + 0.308 Re**(1/2) Sc**(1/3)``, capped at ``VENTPL_MAX = 5``; then
    ``B11 = f 4 pi C / (FK + FD)`` with the thermal term
    ``FK = (L/(Rv T) - 1) L / (K T)`` and the diffusion term
    ``FD = Rv T / (D e_s)`` (Pruppacher and Klett 1997, eq. 13-28 form).

    ``xl`` drop masses (g), ``t`` K, ``p_cgs`` dyn/cm2, ``v`` fall speeds
    (cm/s), ``rlec`` capacities (cm), ``ro1bl`` bulk densities (g/cm3), each
    (33,).  Returns ``B11_MY`` (33,).
    """
    where = "JERRATE_KS (:1744-1921)"
    x = _vector(xl, NKR, "xl", where)
    vv = _vector(v, NKR, "v", where)
    cap = _vector(rlec, NKR, "rlec", where)
    rho = _vector(ro1bl, NKR, "ro1bl", where)
    tp = float(t)
    pp = float(p_cgs)

    deg01 = 1.0 / 3.0
    deg03 = 1.0 / 3.0
    d_my = D_MYIN * (PZERO / pp) * _pow(tp / TZERO, 1.94, where)
    shmidt_number = COEFF_VISCOUS / d_my
    a = 2.0 * _pow(3.0 / 4.0 / 3.141593, deg01, where)
    b = a / COEFF_VISCOUS
    rvt = RV_MY * tp
    esat1 = polysvp(tp, 0)

    ventplm = [0.0] * NKR
    shmidt_number03 = _pow(shmidt_number, deg03, where)
    for kr in range(NKR):
        reinolds_number = b * vv[kr] * _pow(x[kr] / rho[kr], deg03, where)
        reshm = math.sqrt(reinolds_number) * shmidt_number03
        if reinolds_number < 2.5:
            ventplm[kr] = 1.0 + 0.108 * reshm * reshm
        else:
            ventplm[kr] = 0.78 + 0.308 * reshm

    out = np.empty(NKR, dtype=np.float64)
    for kr in range(NKR):
        ventpl = ventplm[kr]
        if ventpl > VENTPL_MAX:
            ventpl = VENTPL_MAX
        constl = CONST_4PI * cap[kr]
        fd1 = rvt / d_my / esat1
        fk1 = (AL1_MY_WATER / rvt - 1.0) * AL1_MY_WATER / CF_MY / tp
        xl_my1 = ventpl * constl
        detl = fk1 + fd1
        out[kr] = xl_my1 / detl
    return out


# ---------------------------------------------------------------------------
# JERTIMESC_KS (:1925-1964)
# ---------------------------------------------------------------------------

def jertimesc(f, x, b, cf: float, col: float = COL) -> float:
    """Sum of growth rates over the spectrum, ``SFN11 = CF sum f_k 3 COL x_k B_k``.

    ``JERTIMESC_KS`` (:1925-1964) for one species (``ID = 1``): the
    sequential sum over bins of ``FK*DELM*B11`` with ``DELM = X*3*COL``,
    times ``CF`` (``ONECOND1`` passes ``B8L = 1/ROR``).  This is the phase
    relaxation rate of the supersaturation equation before its
    thermodynamic factor.
    """
    where = "JERTIMESC_KS (:1925-1964)"
    ff = _vector(f, NKR, "f", where)
    xx = _vector(x, NKR, "x", where)
    bb = _vector(b, NKR, "b", where)
    col = float(col)
    s = 0.0
    for kr in range(NKR):
        fk = ff[kr]
        delm = xx[kr] * 3.0 * col
        fun = fk * delm
        s = s + fun * bb[kr]
    return float(cf) * s


# ---------------------------------------------------------------------------
# JERSUPSAT_KS (:1966-2330)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JersupsatResult:
    """What ``JERSUPSAT_KS`` returns through its arguments, and which branch ran.

    ``del1n``, ``del2n``: water and ice supersaturation at the end of the
    step; ``del1int``, ``del2int``: their time integrals over the step;
    ``rw``, ``pw``, ``ri``, ``pi``: the rates as the routine left them
    (it zeroes or floors them, :1995-2048); ``isym``: the seven flags
    ``(ISYM1, ISYM2(1:3), ISYM3, ISYM4, ISYM5)`` as it left them.
    ``branch`` is one of the ``SUPSAT_*`` codes; ``g2_clamp`` says the
    determinant ``G2`` was below 1e-20 and replaced (:2178); ``gama_clamp``
    says branch 7 raised ``GAMA`` to its floor (:2202).
    """

    del1n: float
    del2n: float
    del1int: float
    del2int: float
    rw: float
    pw: float
    ri: float
    pi: float
    isym: tuple[int, int, int, int, int, int, int]
    branch: int
    g2_clamp: bool
    gama_clamp: bool

    @property
    def branch_name(self) -> str:
        return SUPSAT_BRANCH_NAMES[self.branch]


def _expm1_series(x: float) -> float:
    # EXPM1 statement function (:1986-1987): a fifth-order Taylor series.
    return (x + x * x / 2.0 + x * x * x / 6.0 + x * x * x * x / 24.0
            + x * x * x * x * x / 120.0)


def jersupsat(del1: float, del2: float, rw: float, pw: float, ri: float,
              pi: float, dt: float, dyn1: float, dyn2: float,
              isym: Sequence[int]) -> JersupsatResult:
    """Supersaturation over one step from the linear relaxation system.

    ``JERSUPSAT_KS`` (:1966-2330) solves, over ``dt``,

        dD1/dt = DYN1 - RW D1 - RI D2,    dD2/dt = DYN2 - PW D1 - PI D2,

    for the water (``D1``) and ice (``D2``) supersaturations, in closed form
    (Khain and Sednev 1996; Khain et al. 2004), and returns the end values
    and their time integrals.  Every branch of the source is transcribed:
    the rate screening (:1995-2056), the no-sink case (:2061-2070), water
    only and ice only with their small-argument series (:2075-2149), and
    both phases (:2151-2320) with the ``G2`` clamp (:2178), the ``A < 0``
    fatal check (:2153-2167) and the three small-``GAMA DT`` forms, the
    ``GAMA`` clamp (:2202) among them.

    ``isym`` holds ``(ISYM1, ISYM2(1), ISYM2(2), ISYM2(3), ISYM3, ISYM4,
    ISYM5)``; the routine may clear them and the result carries them back.
    """
    where = "JERSUPSAT_KS (:1966-2330)"
    try:
        return _jersupsat(float(del1), float(del2), float(rw), float(pw),
                          float(ri), float(pi), float(dt), float(dyn1),
                          float(dyn2), isym)
    except ZeroDivisionError as exc:
        raise FsbmRefusal(where, "division by zero; WRF would carry an Inf or "
                          f"NaN (RW={rw!r}, PW={pw!r}, RI={ri!r}, PI={pi!r}, "
                          f"DT={dt!r})") from exc


def _jersupsat(DEL1, DEL2, RW, PW, RI, PI, DT, DYN1, DYN2, isym):
    flags = [int(v) for v in isym]
    if len(flags) != 7:
        raise FsbmRefusal("JERSUPSAT_KS (:1966-2330)",
                          f"isym has {len(flags)} flags, expected 7 "
                          "(ISYM1, ISYM2(1:3), ISYM3, ISYM4, ISYM5)")
    ISYM1 = flags[0]
    ISYM2 = flags[1:4]
    ISYM3, ISYM4, ISYM5 = flags[4], flags[5], flags[6]
    EXPM1 = _expm1_series

    ISYMICE = sum(ISYM2) + ISYM3 + ISYM4 + ISYM5
    IRW = 1
    IRI = 1
    g2_clamp = False
    gama_clamp = False
    screened_all = False

    if max(RW, PW, RI, PI) <= RW_PW_RI_PI_MIN:
        RW = 0.0
        PW = 0.0
        RI = 0.0
        IRI = 0
        PI = 0.0
        IRW = 0
        ISYM1 = 0
        ISYMICE = 0
        screened_all = True
    else:
        if max(RW, PW) > RW_PW_MIN:
            # (KS) - a zero can pass through, assign a minimum value (:2012)
            if RW < RW_PW_MIN * RW_PW_MIN:
                RW = 1.0e-20
                IRW = 0
            if PW < RW_PW_MIN * RW_PW_MIN:
                PW = 1.0e-20
            if max(PI / PW, RI / RW) <= RATIO_ICEW_MIN:
                # only water (:2022-2029)
                RI = 0.0
                IRI = 0
                PI = 0.0
                ISYMICE = 0
            if min(PI / PW, RI / RW) > 1.0 / RATIO_ICEW_MIN:
                # only ice (:2031-2038)
                RW = 0.0
                IRW = 0
                PW = 0.0
                ISYM1 = 0
        else:
            # only ice (:2040-2047)
            RW = 0.0
            IRW = 0
            PW = 0.0
            ISYM1 = 0

    if ISYMICE == 0:
        ISYM2 = [0, 0, 0]
        ISYM3 = 0
        ISYM4 = 0
        ISYM5 = 0

    def result(d1n, d2n, d1i, d2i, branch):
        return JersupsatResult(
            del1n=d1n, del2n=d2n, del1int=d1i, del2int=d2i,
            rw=RW, pw=PW, ri=RI, pi=PI,
            isym=(ISYM1, ISYM2[0], ISYM2[1], ISYM2[2], ISYM3, ISYM4, ISYM5),
            branch=branch, g2_clamp=g2_clamp, gama_clamp=gama_clamp)

    if IRW == 0 and IRI == 0:
        # (:2061-2070)
        DEL1N = DEL1 + DYN1 * DT
        DEL2N = DEL2 + DYN2 * DT
        DEL1INT = DEL1 * DT + DYN1 * DT * DT / 2.0
        DEL2INT = DEL2 * DT + DYN2 * DT * DT / 2.0
        branch = SUPSAT_ALL_BELOW_MIN if screened_all else SUPSAT_NO_SINK
        return result(DEL1N, DEL2N, DEL1INT, DEL2INT, branch)

    if IRI == 0:
        # only water (:2075-2110)
        EXPR = _exp(-RW * DT)
        if abs(RW * DT) > 1.0e-6:
            DEL1N = DEL1 * EXPR + (DYN1 / RW) * (1.0 - EXPR)
            DEL2N = (PW * DEL1 * EXPR / RW - PW * DYN1 * DT / RW
                     - PW * DYN1 * EXPR / (RW * RW) + DYN2 * DT
                     + DEL2 - PW * DEL1 / RW + PW * DYN1 / (RW * RW))
            DEL1INT = (-DEL1 * EXPR / RW + DYN1 * DT / RW
                       + DYN1 * EXPR / (RW * RW) + DEL1 / RW - DYN1 / (RW * RW))
            DEL2INT = (PW * DEL1 * EXPR / (-RW * RW) - PW * DYN1 * DT * DT / (2.0 * RW)
                       + PW * DYN1 * EXPR / (RW * RW * RW) + DYN2 * DT * DT / 2.0
                       + DEL2 * DT - PW * DEL1 * DT / RW + PW * DYN1 * DT / (RW * RW)
                       + PW * DEL1 / (RW * RW) - PW * DYN1 / (RW * RW * RW))
            return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_WATER)
        EXPR = EXPM1(-RW * DT)
        DEL1N = DEL1 + DEL1 * EXPR + (DYN1 / RW) * (0.0 - EXPR)
        DEL2N = (PW * DEL1 * EXPR / RW - PW * DYN1 * DT / RW
                 - PW * DYN1 * EXPR / (RW * RW) + DYN2 * DT + DEL2)
        DEL1INT = -DEL1 * EXPR / RW + DYN1 * DT / RW + DYN1 * EXPR / (RW * RW)
        DEL2INT = (PW * DEL1 * EXPR / (-RW * RW) - PW * DYN1 * DT * DT / (2.0 * RW)
                   + PW * DYN1 * EXPR / (RW * RW * RW) + DYN2 * DT * DT / 2.0
                   + DEL2 * DT - PW * DEL1 * DT / RW + PW * DYN1 * DT / (RW * RW))
        return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_WATER_SERIES)

    if IRW == 0:
        # only ice (:2112-2149)
        EXPP = _exp(-PI * DT)
        if abs(PI * DT) > 1.0e-6:
            DEL2N = DEL2 * EXPP + (DYN2 / PI) * (1.0 - EXPP)
            DEL2INT = (-DEL2 * EXPP / PI + DYN2 * DT / PI
                       + DYN2 * EXPP / (PI * PI) + DEL2 / PI - DYN2 / (PI * PI))
            DEL1N = (+RI * DEL2 * EXPP / PI - RI * DYN2 * DT / PI
                     - RI * DYN2 * EXPP / (PI * PI) + DYN1 * DT
                     + DEL1 - RI * DEL2 / PI + RI * DYN2 / (PI * PI))
            DEL1INT = (-RI * DEL2 * EXPP / (PI * PI) - RI * DYN2 * DT * DT / (2.0 * PI)
                       + RI * DYN2 * EXPP / (PI * PI * PI) + DYN1 * DT * DT / 2.0
                       + DEL1 * DT - RI * DEL2 * DT / PI + RI * DYN2 * DT / (PI * PI)
                       + RI * DEL2 / (PI * PI) - RI * DYN2 / (PI * PI * PI))
            return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_ICE)
        EXPP = EXPM1(-PI * DT)
        DEL2N = DEL2 + DEL2 * EXPP - EXPP * DYN2 / PI
        DEL2INT = -DEL2 * EXPP / PI + DYN2 * DT / PI + DYN2 * EXPP / (PI * PI)
        DEL1N = (+RI * DEL2 * EXPP / PI - RI * DYN2 * DT / PI
                 - RI * DYN2 * EXPP / (PI * PI) + DYN1 * DT + DEL1)
        DEL1INT = (-RI * DEL2 * EXPP / (PI * PI) - RI * DYN2 * DT * DT / (2.0 * PI)
                   + RI * DYN2 * EXPP / (PI * PI * PI) + DYN1 * DT * DT / 2.0
                   + DEL1 * DT - RI * DEL2 * DT / PI + RI * DYN2 * DT / (PI * PI))
        return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_ICE_SERIES)

    # water and ice (:2151-2320); here IRW == 1 and IRI == 1
    A = (RW - PI) * (RW - PI) + 4.0 * PW * RI
    if A < 0.0:
        raise FsbmRefusal(
            "JERSUPSAT_KS (:2155-2167)",
            "fatal error: STOP 1905:A < 0, model stop "
            f"(A={A!r}, RW={RW!r}, PW={PW!r}, RI={RI!r}, PI={PI!r})")
    ALFA = math.sqrt((RW - PI) * (RW - PI) + 4.0 * PW * RI)
    BETA = 0.5 * (ALFA + RW + PI)
    GAMA = 0.5 * (ALFA - RW - PI)
    G31 = PI * DYN1 - RI * DYN2
    G32 = -PW * DYN1 + RW * DYN2
    G2 = RW * PI - RI * PW
    if G2 < 1.0e-20:
        # (KS) - 24th,May,2016 (:2178); the last factor is a default-real
        # literal, a double in the O64 build
        G2 = 1.0004e-11 * 1.0003e-11 - 1.0002e-11 * 1.0001e-11
        g2_clamp = True
    EXPB = _exp(-BETA * DT)
    EXPG = _exp(GAMA * DT)

    if abs(GAMA * DT) > 1.0e-6:
        C11 = (BETA * DEL1 - RW * DEL1 - RI * DEL2 - BETA * G31 / G2 + DYN1) / ALFA
        C21 = (GAMA * DEL1 + RW * DEL1 + RI * DEL2 - GAMA * G31 / G2 - DYN1) / ALFA
        C12 = (BETA * DEL2 - PW * DEL1 - PI * DEL2 - BETA * G32 / G2 + DYN2) / ALFA
        C22 = (GAMA * DEL2 + PW * DEL1 + PI * DEL2 - GAMA * G32 / G2 - DYN2) / ALFA
        DEL1N = C11 * EXPG + C21 * EXPB + G31 / G2
        DEL1INT = (C11 * EXPG / GAMA - C21 * EXPB / BETA + (C21 / BETA - C11 / GAMA)
                   + G31 * DT / G2)
        DEL2N = C12 * EXPG + C22 * EXPB + G32 / G2
        DEL2INT = (C12 * EXPG / GAMA - C22 * EXPB / BETA + (C22 / BETA - C12 / GAMA)
                   + G32 * DT / G2)
        return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_MIXED)

    if abs(RI / RW) > 1.0e-12:
        if abs(RW / RI) > 1.0e-12:
            # (:2199-2246)
            ALFA = math.sqrt((RW - PI) * (RW - PI) + 4.0 * PW * RI)
            BETA = 0.5 * (ALFA + RW + PI)
            GAMA = 0.5 * (ALFA - RW - PI)
            if GAMA < 0.5 * 2.0e-10:
                # (KS) - 24th,May,2016 (:2202)
                GAMA = 0.5 * (2.002e-10 - 2.001e-10)
                gama_clamp = True
            EXPG = EXPM1(GAMA * DT)
            EXPB = _exp(-BETA * DT)

            C11 = (BETA * DEL1 - RW * DEL1 - RI * DEL2 + DYN1) / ALFA
            C21 = (GAMA * DEL1 + RW * DEL1 + RI * DEL2 - GAMA * G31 / G2 - DYN1) / ALFA
            C12 = (BETA * DEL2 - PW * DEL1 - PI * DEL2 + DYN2) / ALFA
            C22 = (GAMA * DEL2 + PW * DEL1 + PI * DEL2 - GAMA * G32 / G2 - DYN2) / ALFA

            A1DEL1N = C11
            A2DEL1N = C11 * EXPG
            A3DEL1N = C21 * EXPB
            A4DEL1N = G31 / G2 * (GAMA / ALFA + (GAMA / ALFA - 1.0) * EXPG)
            DEL1N = A1DEL1N + A2DEL1N + A3DEL1N + A4DEL1N

            A1DEL1INT = C11 * EXPG / GAMA
            A2DEL1INT = -C21 * EXPB / BETA
            A3DEL1INT = C21 / BETA
            A4DEL1INT = G31 / G2 * DT * (GAMA / ALFA)
            DEL1INT = A1DEL1INT + A2DEL1INT + A3DEL1INT + A4DEL1INT

            A1DEL2N = C12
            A2DEL2N = C12 * EXPG
            A3DEL2N = C22 * EXPB
            A4DEL2N = G32 / G2 * (GAMA / ALFA
                                  + (GAMA / ALFA - 1.0)
                                  * (GAMA * DT + GAMA * GAMA * DT * DT / 2.0))
            DEL2N = A1DEL2N + A2DEL2N + A3DEL2N + A4DEL2N

            A1DEL2INT = C12 * EXPG / GAMA
            A2DEL2INT = -C22 * EXPB / BETA
            A3DEL2INT = C22 / BETA
            A4DEL2INT = G32 / G2 * DT * (GAMA / ALFA)
            A5DEL2INT = G32 / G2 * (GAMA / ALFA - 1.0) * (GAMA * DT * DT / 2.0)
            DEL2INT = A1DEL2INT + A2DEL2INT + A3DEL2INT + A4DEL2INT + A5DEL2INT
            return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_MIXED_SERIES)

        # |RW/RI| <= 1e-12 (:2248-2278)
        X = -2.0 * RW * PI + RW * RW + 4.0 * PW * RI
        ALFA = PI * (1 + (X / PI) / 2.0 - (X / PI) * (X / PI) / 8.0)
        BETA = PI + (X / PI) / 4.0 - (X / PI) * (X / PI) / 16.0 + RW / 2.0
        GAMA = (X / PI) / 4.0 - (X / PI) * (X / PI) / 16.0 - RW / 2.0
        EXPG = EXPM1(GAMA * DT)
        EXPB = _exp(-BETA * DT)
        C11 = (BETA * DEL1 - RW * DEL1 - RI * DEL2 + DYN1) / ALFA
        C21 = (GAMA * DEL1 + RW * DEL1 + RI * DEL2 - GAMA * G31 / G2 - DYN1) / ALFA
        C12 = (BETA * DEL2 - PW * DEL1 - PI * DEL2 + DYN2) / ALFA
        C22 = (GAMA * DEL2 + PW * DEL1 + PI * DEL2 - GAMA * G32 / G2 - DYN2) / ALFA
        DEL1N = (C11 + C11 * EXPG + C21 * EXPB
                 + G31 / G2 * (GAMA / ALFA + (GAMA / ALFA - 1) * EXPG))
        DEL1INT = (C11 * EXPG / GAMA - C21 * EXPB / BETA + (C21 / BETA)
                   + G31 / G2 * DT * (GAMA / ALFA))
        DEL2N = (C12 + C12 * EXPG + C22 * EXPB + G32 / G2 * (GAMA / ALFA
                 + (GAMA / ALFA - 1.0)
                 * (GAMA * DT + GAMA * GAMA * DT * DT / 2.0)))
        DEL2INT = (C12 * EXPG / GAMA - C22 * EXPB / BETA
                   + (C22 / BETA) + G32 / G2 * DT * (GAMA / ALFA)
                   + G32 / G2 * (GAMA / ALFA - 1.0) * (GAMA * DT * DT / 2.0))
        return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_MIXED_SMALL_RW)

    # |RI/RW| <= 1e-12 (:2282-2312)
    X = -2.0 * RW * PI + PI * PI + 4.0 * PW * RI
    ALFA = RW * (1.0 + (X / RW) / 2.0 - (X / RW) * (X / RW) / 8.0)
    BETA = RW + (X / RW) / 4.0 - (X / RW) * (X / RW) / 16.0 + PI / 2.0
    GAMA = (X / RW) / 4.0 - (X / RW) * (X / RW) / 16.0 - PI / 2.0
    EXPG = EXPM1(GAMA * DT)
    EXPB = _exp(-BETA * DT)
    C11 = (BETA * DEL1 - RW * DEL1 - RI * DEL2 + DYN1) / ALFA
    C21 = (GAMA * DEL1 + RW * DEL1 + RI * DEL2 - GAMA * G31 / G2 - DYN1) / ALFA
    C12 = (BETA * DEL2 - PW * DEL1 - PI * DEL2 + DYN2) / ALFA
    C22 = (GAMA * DEL2 + PW * DEL1 + PI * DEL2 - GAMA * G32 / G2 - DYN2) / ALFA
    DEL1N = (C11 + C11 * EXPG + C21 * EXPB
             + G31 / G2 * (GAMA / ALFA + (GAMA / ALFA - 1.0) * EXPG))
    DEL1INT = (C11 * EXPG / GAMA - C21 * EXPB / BETA + (C21 / BETA)
               + G31 / G2 * DT * (GAMA / ALFA))
    DEL2N = (C12 + C12 * EXPG + C22 * EXPB + G32 / G2
             * (GAMA / ALFA
                + (GAMA / ALFA - 1.0) * (GAMA * DT + GAMA * GAMA * DT * DT / 2.0)))
    DEL2INT = (C12 * EXPG / GAMA - C22 * EXPB / BETA + C22 / BETA
               + G32 / G2 * DT * (GAMA / ALFA)
               + G32 / G2 * (GAMA / ALFA - 1.0) * (GAMA * DT * DT / 2.0))
    return result(DEL1N, DEL2N, DEL1INT, DEL2INT, SUPSAT_MIXED_SMALL_RI)


# ---------------------------------------------------------------------------
# JERNEWF_KS (:2444-2757)
# ---------------------------------------------------------------------------

def jernewf(rr, fi, rn, *, i3point: int = ISIGN_3POINT, tpn: float = 0.0,
            idrop: int = 1, col: float = COL) -> tuple[np.ndarray, int]:
    """Remap a spectrum whose bin masses moved onto the fixed mass grid.

    ``JERNEWF_KS`` (:2444-2757).  ``rr`` are the grid masses (g), ``fi`` the
    size distribution before growth (f-form), ``rn`` the masses the bins
    grew or evaporated to.  Returns the new distribution ``psi`` on ``rr``
    and ``IDROP`` as the routine leaves it.

    Steps, all as WRF writes them:

    * a negative new mass becomes 1e-50 and its bin's content is dropped
      (:2477-2483);
    * evaporation (``RN(1) < RR(1)``) turns the 3-point step off and sets
      ``IDROP = 0`` (:2493-2504);
    * with any mass changed (:2521-2530), the Kovetz-Olund step splits each
      bin's number ``FI*RR`` between the two grid masses around its new mass
      so that number and mass are kept (Kovetz and Olund 1969), a new mass
      below bin 1 going to bin 1 with fraction ``RN/RR(1)`` (:2534-2583);
    * when ``I3POINT`` is 1 and ``IDROP`` is 1, the 3-point step spreads
      each bin over three grid masses with quadratic weights, subject to the
      positivity and smoothing tests, and leaves the loop at the first bin
      that would go negative (:2588-2680);
    * the tail remap over bins 6-12 merges any bin holding less than
      ``COEFF_REMAPING`` of the number of the bin below into it
      (:2693-2722);
    * a negative result is WRF's fatal ``JERNEWF PSI(KR)<0`` (:2734-2752).

    ``tpn`` is accepted for the contract; WRF v4.8.0 no longer reads it (the
    line that did is commented out at :2468).
    """
    where = "JERNEWF_KS (:2444-2757)"
    try:
        return _jernewf(rr, fi, rn, int(i3point), int(idrop), float(col))
    except ZeroDivisionError as exc:
        raise FsbmRefusal(where, "division by zero; WRF would carry an Inf or "
                          "NaN into the spectrum") from exc


def _jernewf(rr_in, fi_in, rn_in, I3POINT, IDROP, COL_):
    where = "JERNEWF_KS (:2444-2757)"
    NRX = NKR
    NRXP = NRX + 1
    NRX1 = NKR
    # One-based working arrays: index 0 is unused, so RR[K] is RR(K).
    RR = [0.0] + _vector(rr_in, NRX, "rr", where)
    FI = [0.0] + _vector(fi_in, NRX, "fi", where)
    RN = [0.0] + _vector(rn_in, NRX, "rn", where)
    PSI = [0.0] * (NRX + 1)
    CDROP = [0.0] * (NRX + 1)
    DELTA_CDROP = [0.0] * (NRX + 1)
    RRS = [0.0] * (NRXP + 1)
    PSINEW = [0.0] * (NRXP + 1)

    for I in range(1, NRX + 1):
        if RN[I] < 0.0:
            RN[I] = 1.0e-50
            FI[I] = 0.0

    for K in range(1, NRX + 1):
        RRS[K] = RR[K]

    I3POINT_CONDEVAP = I3POINT
    if RN[1] < RRS[1]:
        # evaporation (:2494-2502)
        I3POINT_CONDEVAP = 0
        IDROP = 0
        NRX1 = NRX
    if IDROP == 0:
        I3POINT_CONDEVAP = 0

    RRS[NRXP] = RRS[NRX] * 1024.0
    PSINEW[NRXP] = 0.0

    ISIGN_DIFFUSIONAL_GROWTH = 0
    for K in range(1, NRX + 1):
        if RN[K] != RR[K]:
            ISIGN_DIFFUSIONAL_GROWTH = 1
            break

    if ISIGN_DIFFUSIONAL_GROWTH == 1:
        # Kovetz-Olund method (:2534-2583)
        for K in range(1, NRX1 + 1):
            if FI[K] > 0.0:
                if abs(RN[K] - RR[K]) < 1.0e-16:
                    PSINEW[K] = FI[K] * RR[K]
                    continue
                I = 1
                while (not (RRS[I] <= RN[K] and RRS[I + 1] >= RN[K])
                       and I < NRX1):
                    I = I + 1
                if RN[K] < RRS[1]:
                    RNTMP = RN[K]
                    RRTMP = 0.0
                    RRP = RRS[1]
                    GMAT2 = (RNTMP - RRTMP) / (RRP - RRTMP)
                    PSINEW[1] = PSINEW[1] + FI[K] * RR[K] * GMAT2
                else:
                    RNTMP = RN[K]
                    RRTMP = RRS[I]
                    RRP = RRS[I + 1]
                    GMAT2 = (RNTMP - RRTMP) / (RRP - RRTMP)
                    GMAT = (RRP - RNTMP) / (RRP - RRTMP)
                    PSINEW[I] = PSINEW[I] + FI[K] * RR[K] * GMAT
                    PSINEW[I + 1] = PSINEW[I + 1] + FI[K] * RR[K] * GMAT2
        for KR in range(1, NRX1 + 1):
            PSI[KR] = PSINEW[KR]
        for KR in range(NRX1 + 1, NRX + 1):
            PSI[KR] = FI[KR]

        # 3-point method (:2588-2680)
        if I3POINT_CONDEVAP == 1:
            for K in range(1, NRX1):
                if FI[K] > 0.0:
                    if abs(RN[K] - RR[K]) < 1.0e-16:
                        PSI[K] = FI[K] * RR[K]
                        continue
                    if RRS[2] < RN[K]:
                        I = 2
                        while (not (RRS[I] <= RN[K] and RRS[I + 1] >= RN[K])
                               and I < NRX1 - 1):
                            I = I + 1
                        RNTMP = RN[K]
                        RRTMP = RRS[I]
                        RRP = RRS[I + 1]
                        RRM = RRS[I - 1]
                        RNTMP2 = RN[K + 1]
                        RRTMP2 = RRS[I + 1]
                        RRP2 = RRS[I + 2]
                        RRM2 = RRS[I]
                        GN1 = ((RRP - RNTMP) * (RRTMP - RNTMP) / (RRP - RRM)
                               / (RRTMP - RRM))
                        GN1P = ((RRP2 - RNTMP2) * (RRTMP2 - RNTMP2)
                                / (RRP2 - RRM2) / (RRTMP2 - RRM2))
                        GN2 = ((RRP - RNTMP) * (RNTMP - RRM) / (RRP - RRTMP)
                               / (RRTMP - RRM))
                        GMAT = (RRP - RNTMP) / (RRP - RRTMP)
                        GN3 = ((RRTMP - RNTMP) * (RRM - RNTMP) / (RRP - RRM)
                               / (RRP - RRTMP))
                        GMAT2 = (RNTMP - RRTMP) / (RRP - RRTMP)
                        PSI_IM = PSI[I - 1] + GN1 * FI[K] * RR[K]
                        PSI_I = (PSI[I] + GN1P * FI[K + 1] * RR[K + 1]
                                 + (GN2 - GMAT) * FI[K] * RR[K])
                        PSI_IP = PSI[I + 1] + (GN3 - GMAT2) * FI[K] * RR[K]
                        if PSI_IM > 0.0:
                            if PSI_IP > 0.0:
                                if I > 2:
                                    # smoothing criteria, (a .AND. b .AND. c)
                                    # .OR. d by Fortran precedence (:2644-2645)
                                    if ((PSI_IM > PSI[I - 2] and PSI_IM < PSI_I
                                         and PSI[I - 2] < PSI[I])
                                            or PSI[I - 2] >= PSI[I]):
                                        PSI[I - 1] = PSI_IM
                                        PSI[I] = PSI[I] + FI[K] * RR[K] * (GN2 - GMAT)
                                        PSI[I + 1] = PSI_IP
                            else:
                                break
                        else:
                            break

        for K in range(1, NRX1 + 1):
            PSI[K] = PSI[K] / RR[K]
        for K in range(NRX1 + 1, NRX + 1):
            PSI[K] = FI[K]

        if IDROP == 1:
            # drop tail remap over bins 6-12 (:2693-2722)
            for K in range(KRDROP_REMAPING_MIN, KRDROP_REMAPING_MAX + 1):
                CDROP[K] = 3.0 * COL_ * PSI[K] * RR[K]
            KMAX = KRDROP_REMAPING_MIN
            for K in range(KRDROP_REMAPING_MAX, KRDROP_REMAPING_MIN - 1, -1):
                KMAX = K
                if PSI[K] > 0.0:
                    break
            for K in range(KMAX - 1, KRDROP_REMAPING_MIN - 1, -1):
                if CDROP[K] > 0.0:
                    DELTA_CDROP[K] = CDROP[K + 1] / CDROP[K]
                    if DELTA_CDROP[K] < COEFF_REMAPING:
                        CDROP[K] = CDROP[K] + CDROP[K + 1]
                        CDROP[K + 1] = 0.0
            for K in range(KRDROP_REMAPING_MIN, KMAX + 1):
                PSI[K] = CDROP[K] / (3.0 * COL_ * RR[K])
    else:
        for K in range(1, NRX + 1):
            PSI[K] = FI[K]

    for KR in range(1, NRX + 1):
        if PSI[KR] < 0.0:
            raise FsbmRefusal(
                "JERNEWF_KS (:2734-2752)",
                "fatal error in SUBROUTINE JERNEWF PSI(KR)<0, < min, model stop "
                f"(KR={KR}, PSI={PSI[KR]!r}, ISIGN_DIFFUSIONAL_GROWTH="
                f"{ISIGN_DIFFUSIONAL_GROWTH}, I3POINT_CONDEVAP={I3POINT_CONDEVAP})")

    return np.array(PSI[1:], dtype=np.float64), IDROP


# ---------------------------------------------------------------------------
# JERDFUN_KS (:2334-2442) and JERDFUN_NEW_KS (:2759-2834)
# ---------------------------------------------------------------------------

def jerdfun(xi, b, fi, d_int: float, *, ind: int, itype: int, tpn: float,
            idrop: int, col: float = COL, isym2: int = 1, psi_in=None,
            xin_in=None) -> tuple[np.ndarray, np.ndarray, int]:
    """New masses from the integrated supersaturation, then the remap.

    ``JERDFUN_KS`` (:2334-2442).  With ``IND = 1`` and ``ITYPE = 1`` (drops)
    each bin mass follows the diffusional growth law in ``m**(2/3)``
    (:2379-2391),

        xiN = (xi**(2/3) + (2/3) DEL2N B21 / xi**(1/3))**(3/2),

    with ``xiN = 1e-50`` where the bracket goes negative (evaporated away);
    otherwise mass grows linearly, ``xiN = xi + DEL2N B21`` (:2393-2399).
    ``d_int`` is ``DEL2N``, the time-integrated supersaturation of the step.
    Then :func:`jernewf` remaps with ``I3POINT = ISIGN_3POINT``, and a
    negative result is the fatal ``PSI2R(KR)<0.0D0`` check (:2420-2431).

    Returns ``(psi, xin, idrop)``.  When ``isym2`` is not 1 WRF assigns
    neither ``PSI2`` nor ``xiN``, so they come back as the caller had them:
    ``psi_in`` and ``xin_in`` (zeros when not given).
    """
    where = "JERDFUN_KS (:2334-2442)"
    x = _vector(xi, NKR, "xi", where)
    bb = _vector(b, NKR, "b", where)
    f = _vector(fi, NKR, "fi", where)
    if isym2 != 1:
        psi = (np.zeros(NKR) if psi_in is None
               else np.array(_vector(psi_in, NKR, "psi_in", where)))
        xin = (np.zeros(NKR) if xin_in is None
               else np.array(_vector(xin_in, NKR, "xin_in", where)))
        return psi, xin, int(idrop)

    C = 2.0 / 3.0
    DEGREE1 = 1.0 / 3.0
    DEGREE2 = C
    DEGREE3 = 3.0 / 2.0
    del2n = float(d_int)
    xin = np.empty(NKR, dtype=np.float64)
    if ind == 1 and itype == 1:
        for kr in range(NKR):
            D = _pow(x[kr], DEGREE1, where)
            RATEXI = C * del2n * bb[kr] / D
            B = _pow(x[kr], DEGREE2, where)
            A = B + RATEXI
            if A < 0.0:
                xin[kr] = 1.0e-50
            else:
                xin[kr] = _pow(A, DEGREE3, where)
    else:
        for kr in range(NKR):
            RATEXI = del2n * bb[kr]
            xin[kr] = x[kr] + RATEXI

    psi, idrop_out = jernewf(x, f, xin, i3point=ISIGN_3POINT, tpn=tpn,
                             idrop=idrop, col=col)
    for kr in range(NKR):
        if psi[kr] < 0.0:
            raise FsbmRefusal(
                "JERDFUN_KS (:2426-2429)",
                "fatal error in PSI2R(KR)<0.0D0, in JERDFUN_KS, model stop "
                f"(KR={kr + 1}, PSI2R={psi[kr]!r})")
    return psi, xin, idrop_out


def jerdfun_new(xi, b, fi, *, tpn: float, idrop: int,
                col: float = COL) -> tuple[np.ndarray, int]:
    """Final drop spectrum from the accumulated growth ``SUPINTW``.

    ``JERDFUN_NEW_KS`` (:2759-2834): the ``m**(2/3)`` law of
    :func:`jerdfun` with the integrated supersaturation already folded into
    ``b`` (``ONECOND1`` passes ``SUPINTW = sum B11_MY * D1N``, :7075, and
    ``RATExi = C*B21_MY/D``, :2802), then :func:`jernewf` with
    ``I3POINT = ISIGN_3POINT``.  ``DEGREE1`` is written ``C/2`` here, which
    is the same double as ``1/3``.  Unlike ``JERDFUN_KS`` it does not return
    the new masses (``xiNR`` is local, :2792).  Returns ``(psi, idrop)``.
    """
    where = "JERDFUN_NEW_KS (:2759-2834)"
    x = _vector(xi, NKR, "xi", where)
    bb = _vector(b, NKR, "b", where)
    f = _vector(fi, NKR, "fi", where)
    C = 2.0 / 3.0
    DEGREE1 = C / 2.0
    DEGREE2 = C
    DEGREE3 = 3.0 / 2.0
    xinr = np.empty(NKR, dtype=np.float64)
    for kr in range(NKR):
        D = _pow(x[kr], DEGREE1, where)
        RATEXI = C * bb[kr] / D
        B = _pow(x[kr], DEGREE2, where)
        A = B + RATEXI
        if A < 0.0:
            xinr[kr] = 1.0e-50
        else:
            xinr[kr] = _pow(A, DEGREE3, where)
    return jernewf(x, f, xinr, i3point=ISIGN_3POINT, tpn=tpn, idrop=idrop,
                   col=col)


# ---------------------------------------------------------------------------
# ONECOND1 (:6748-7143)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Onecond1Result:
    """What ``ONECOND1`` returns through its arguments, and how it got there.

    ``tt`` (K), ``qq`` (kg/kg), ``del1n``, ``del2n`` (water and ice
    supersaturation after the step) and ``ff1`` (the new drop distribution,
    f-form) are the arguments ``ONECOND1`` changes.  ``isym`` holds the
    seven flags ``(ISYM1, ISYM2(1:3), ISYM3, ISYM4, ISYM5)`` as
    ``JERSUPSAT_KS`` left them.  ``kcond`` is 11 (condensation branch,
    ``DEL1N >= 0`` at entry) or 10 (evaporation); ``idrop`` the ``IDROP``
    the final remap left; ``supsat`` the last ``JERSUPSAT_KS`` result;
    ``b11`` the growth-rate coefficients; ``delmass`` the change of liquid
    mixing ratio ``RMASSLAA - RMASSLBB`` that set ``tt`` and ``qq``.
    """

    tt: float
    qq: float
    del1n: float
    del2n: float
    ff1: np.ndarray
    isym: tuple[int, int, int, int, int, int, int]
    kcond: int
    idrop: int | None
    supsat: JersupsatResult
    b11: np.ndarray
    delmass: float


def _oper2(ar1: float) -> float:
    # OPER2 statement function (:6827)
    return 0.622 / (0.622 + 0.378 * ar1) / ar1


def _liquid(f: list[float], r1: list[float], col3: float, rori: float) -> float:
    # RMASSLBB / RMASSLAA sums (:7038-7054, :7096-7112), sequential
    s = 0.0
    for k in range(NKR):
        fi1_k = f[k]
        r1_k = r1[k]
        fi1r1 = fi1_k * r1_k * r1_k
        s = s + fi1r1
    return s * col3 * rori


def onecond1(tt: float, qq: float, pp: float, ror: float, vr1, del1n: float,
             del2n: float, ff1, tables, dtcond: float, *, psi1=None,
             isym: Sequence[int] = (1, 0, 0, 0, 0, 0, 0),
             al1: float = AL1) -> Onecond1Result:
    """One warm diffusional growth or evaporation step of the drop spectrum.

    ``ONECOND1`` (:6748-7143), the routine ``FAST_SBM`` calls when drops are
    present and ice is absent or the air is warmer than -0.187 C
    (:4711-4722).  Inputs as ``FAST_SBM`` passes them: ``tt`` K, ``qq``
    kg/kg, ``pp`` dyn/cm2, ``ror`` air density g/cm3, ``vr1`` the pressure
    corrected drop fall speeds cm/s (33,), ``del1n``, ``del2n`` the water and
    ice supersaturations, ``ff1`` the drop distribution (33,), ``dtcond``
    the substep s.  ``tables`` supplies ``xl`` (``R1``), ``rlec`` and
    ``ro1bl`` (float64 view for the O64 gates).

    The step: growth coefficients from :func:`jerrate_drops` at the entry
    temperature and the phase relaxation rate from :func:`jertimesc`
    (:6912-6920); the relaxation rates ``RW``, ``PW`` with the
    thermodynamic factor ``OPER2(q) + L/c_p B/T**2`` (:6926-6937, ice rates
    zero); :func:`jersupsat` with no dynamic forcing (:6963-6974 or
    :7006-7017); the in-loop remap :func:`jerdfun` and its temperature-jump
    check (:6976-7071); then the final spectrum from :func:`jerdfun_new` on
    the entry spectrum with the accumulated ``B11_MY * D1N`` (:7087-7091),
    and ``qq``, ``tt`` from the change of liquid mass with ``AL1 = 2500``
    (:7093-7117).

    Fatal checks, each an :class:`FsbmRefusal` with WRF's text: ``RW`` or
    ``PW`` NaN (:6939-6942), a negative substep (:6954, :6998), the sign
    flips of the supersaturation (:6982-6985, :7026-7029), and the
    temperature jumps over 3 K in the loop (:7061-7071) and 5 K at the end
    (:7119-7134).

    ``psi1`` is the array WRF passes as ``PSI1``; it only matters when
    ``JERSUPSAT_KS`` clears ``ISYM1``, and then it is returned as the new
    spectrum.  ``FAST_SBM`` passes a copy of ``ff1`` (``FF1IN``), the
    default here.  ``isym`` is the seven ``ISYM`` flags ``FAST_SBM`` passes
    (``ISYM1`` is 1 whenever it calls ``ONECOND1``).  ``al1`` is WRF's
    ``AL1`` DATA value, exposed so a gate can prove it can fail.
    """
    where = "ONECOND1 (:6748-7143)"
    try:
        return _onecond1(tt, qq, pp, ror, vr1, del1n, del2n, ff1, tables,
                         dtcond, psi1, isym, float(al1))
    except ZeroDivisionError as exc:
        raise FsbmRefusal(where, "division by zero; WRF would carry an Inf or "
                          "NaN into T, q or the spectrum") from exc


def _onecond1(TT, QQ, PP, ROR, vr1, DEL1N, DEL2N, ff1, tables, DTCOND,
              psi1, isym, AL1_):
    where = "ONECOND1 (:6748-7143)"
    TT = float(TT)
    QQ = float(QQ)
    PP = float(PP)
    ROR = float(ROR)
    DEL1N = float(DEL1N)
    DEL2N = float(DEL2N)
    DT = float(DTCOND)
    VR1 = _vector(vr1, NKR, "vr1", where)
    FF1 = _vector(ff1, NKR, "ff1", where)
    R1 = _vector(tables.xl, NKR, "tables.xl", where)
    RLEC = _vector(tables.rlec, NKR, "tables.rlec", where)
    RO1BL = _vector(tables.ro1bl, NKR, "tables.ro1bl", where)
    PSI1 = list(FF1) if psi1 is None else _vector(psi1, NKR, "psi1", where)
    flags = [int(v) for v in isym]
    if len(flags) != 7:
        raise FsbmRefusal(where, f"isym has {len(flags)} flags, expected 7")
    ISYM1 = flags[0]
    ISYM_REST = flags[1:]

    B11_MY = [0.0] * NKR
    COL3 = 3.0 * COL
    B8L = 1.0 / ROR
    RORI = 1.0 / ROR

    FF1_OLD = list(FF1)
    SUPINTW = [0.0] * NKR

    TPN = TT
    QPN = QQ
    FI1 = list(FF1)

    TIMENEW = 0.0
    TOLD = TPN
    QOLD = QPN
    R1D = list(R1)
    SFN11 = 0.0
    IDROP = None
    supsat = None
    KCOND = None

    while True:  # label 56 (:6902); runs once, TIMENEW reaches DT
        TIMEREV = DT - TIMENEW
        DEL1 = DEL1N
        DEL2 = DEL2N
        DEL1S = DEL1N
        DEL2S = DEL2N
        TPS = TPN
        QPS = QPN

        if ISYM1 == 1:
            B11_MY = [float(v) for v in
                      jerrate_drops(R1D, TPS, PP, VR1, RLEC, RO1BL)]
            SFN11 = jertimesc(FI1, R1D, B11_MY, B8L, COL)

        SFN12 = 0.0
        SFNL = SFN11 + SFN12
        SFNI = 0.0

        B5L = BB1_MY / TPS / TPS
        B5I = BB2_MY / TPS / TPS
        DOPL = 1.0 + DEL1S
        DOPI = 1.0 + DEL2S
        RW = (_oper2(QPS) + B5L * AL1_) * DOPL * SFNL
        RI = (_oper2(QPS) + B5L * AL2) * DOPL * SFNI
        PW = (_oper2(QPS) + B5I * AL1_) * DOPI * SFNL
        PI = (_oper2(QPS) + B5I * AL2) * DOPI * SFNI
        # QW = B7L*DOPL and QI = B7I*DOPI (:6934, :6937) are zero (B6 = 0)
        # and never read.

        if RW != RW or PW != PW:
            raise FsbmRefusal(
                "ONECOND1 (:6939-6942)",
                "fatal error in ONECOND1 (RW or PW are NaN), model stop "
                f"(RW={RW!r}, PW={PW!r})")

        KCOND = KCOND_EVAPORATION
        if DEL1N >= 0.0:
            KCOND = KCOND_CONDENSATION

        DTNEWL = min(DT, TIMEREV)
        TIMENEW = TIMENEW + DTNEWL
        DTT = DTNEWL
        if DTT < 0.0:
            text = ("fatal error in ONECOND1-DEL1N>0:(DTT<0), model stop"
                    if KCOND == KCOND_CONDENSATION else
                    "fatal error in ONECOND1-DEL1N<0:(DTT<0), model stop")
            line = ":6954" if KCOND == KCOND_CONDENSATION else ":6998"
            raise FsbmRefusal(f"ONECOND1 ({line})", f"{text} (DTT={DTT!r})")

        supsat = jersupsat(DEL1, DEL2, RW, PW, RI, PI, DTT, 0.0, 0.0,
                           [ISYM1] + ISYM_REST)
        DEL1N = supsat.del1n
        DEL2N = supsat.del2n
        RW, PW, RI, PI = supsat.rw, supsat.pw, supsat.ri, supsat.pi
        D1N = supsat.del1int
        ISYM1 = supsat.isym[0]
        ISYM_REST = list(supsat.isym[1:])

        if ISYM1 == 1:
            IDROP = ISYM1
            psi, _xin, IDROP = jerdfun(R1D, B11_MY, FI1, D1N, ind=1, itype=1,
                                       tpn=TPN, idrop=IDROP, col=COL,
                                       isym2=ISYM1)
            PSI1 = [float(v) for v in psi]

        if KCOND == KCOND_CONDENSATION:
            if (DEL1 > 0 and DEL1N < 0) and abs(DEL1N) > EPSDEL:
                raise FsbmRefusal(
                    "ONECOND1 (:6982-6985)",
                    "fatal error in ONECOND1-1 (DEL1.GT.0.AND.DEL1N.LT.0), "
                    f"model stop (DEL1={DEL1!r}, DEL1N={DEL1N!r})")
        else:
            if (DEL1 < 0 and DEL1N > 0) and abs(DEL1N) > EPSDEL:
                raise FsbmRefusal(
                    "ONECOND1 (:7026-7029)",
                    "fatal error in ONECOND1-2 (DEL1.LT.0.AND.DEL1N.GT.0), "
                    f"model stop (DEL1={DEL1!r}, DEL1N={DEL1N!r})")

        RMASSLBB = _liquid(FI1, R1, COL3, RORI)
        if RMASSLBB <= 0.0:
            RMASSLBB = 0.0
        RMASSLAA = _liquid(PSI1, R1, COL3, RORI)
        if RMASSLAA <= 0.0:
            RMASSLAA = 0.0
        DELMASSL1 = RMASSLAA - RMASSLBB
        QPN = QPS - DELMASSL1
        DAL1 = AL1_
        TPN = TPS + DAL1 * DELMASSL1
        if abs(DAL1 * DELMASSL1) > ONECOND1_IN_LIMIT:
            raise FsbmRefusal(
                "ONECOND1 (:7061-7071)",
                "fatal error in ONECOND1-in (ABS(DAL1*DELMASSL1) > 3.0), model "
                f"stop (DAL1*DELMASSL1={DAL1 * DELMASSL1!r}, TPS={TPS!r}, "
                f"QPS={QPS!r})")

        if ISYM1 == 1:
            for kr in range(NKR):
                SUPINTW[kr] = SUPINTW[kr] + B11_MY[kr] * D1N

        if not TIMENEW < DT:
            break

    # label 57 (:7085)
    if ISYM1 == 1:
        psi, IDROP = jerdfun_new(R1D, SUPINTW, FF1_OLD, tpn=TPN, idrop=IDROP,
                                 col=COL)
        PSI1 = [float(v) for v in psi]

    RMASSLBB = _liquid(FF1_OLD, R1, COL3, RORI)
    if RMASSLBB < 0.0:
        RMASSLBB = 0.0
    RMASSLAA = _liquid(PSI1, R1, COL3, RORI)
    if RMASSLAA < 0.0:
        RMASSLAA = 0.0
    DELMASSL1 = RMASSLAA - RMASSLBB
    QPN = QOLD - DELMASSL1
    DAL1 = AL1_
    TPN = TOLD + DAL1 * DELMASSL1
    if abs(DAL1 * DELMASSL1) > ONECOND1_OUT_LIMIT:
        raise FsbmRefusal(
            "ONECOND1 (:7119-7134)",
            "fatal error in ONECOND1-out (ABS(DAL1*DELMASSL1) > 5.0), model "
            f"stop (DAL1*DELMASSL1={DAL1 * DELMASSL1!r})")

    return Onecond1Result(
        tt=TPN, qq=QPN, del1n=DEL1N, del2n=DEL2N,
        ff1=np.array(PSI1, dtype=np.float64),
        isym=(ISYM1, *ISYM_REST), kcond=KCOND, idrop=IDROP, supsat=supsat,
        b11=np.array(B11_MY, dtype=np.float64), delmass=DELMASSL1)



__all__ = [
    "COEFF_REMAPING", "EPSDEL", "ISIGN_3POINT", "ISIGN_KO_1", "ISIGN_KO_2",
    "JersupsatResult", "KCOND_CONDENSATION", "KCOND_EVAPORATION",
    "KRDROP_REMAPING_MAX", "KRDROP_REMAPING_MIN", "ONECOND1_IN_LIMIT",
    "ONECOND1_OUT_LIMIT", "Onecond1Result", "RATIO_ICEW_MIN", "RI_PI_MIN",
    "RW_PW_MIN", "RW_PW_RI_PI_MIN", "SUPSAT_ALL_BELOW_MIN",
    "SUPSAT_BRANCH_NAMES", "SUPSAT_ICE", "SUPSAT_ICE_SERIES", "SUPSAT_MIXED",
    "SUPSAT_MIXED_SERIES", "SUPSAT_MIXED_SMALL_RI", "SUPSAT_MIXED_SMALL_RW",
    "SUPSAT_NO_SINK", "SUPSAT_WATER", "SUPSAT_WATER_SERIES", "VENTPL_MAX",
    "jerdfun", "jerdfun_new", "jernewf", "jerrate_drops", "jersupsat",
    "jertimesc", "onecond1", "polysvp",
]
