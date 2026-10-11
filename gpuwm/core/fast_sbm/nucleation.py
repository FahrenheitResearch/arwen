"""Droplet activation and the initial aerosol column, float64 on the host.

Transcribed from WRF v4.8.0 ``phys/module_mp_fast_sbm.F`` (LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1).  WRF is NCAR/UCAR public domain; the notice is in the
repository ``NOTICE`` and in ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The
scheme is FSBM-2 by Jacob Shpund and co-authors (see the package docstring);
the drop and ice nucleation routines carry the header "Authors: Khain A.P. &
Pokrovsky A.G. July 2002 at Huji, Israel" (:3198-3204, :3332-3338).

Routines and the lines each function transcribes
------------------------------------------------
* ``supmax_coeff``: ``supmax_COEFF`` (:3572-3733), the Koehler constants and
  the coefficient ``C3`` of the cloud-base maximum supersaturation.
* ``cloud_base_super`` and ``cloud_base_rcriti``: ``Cloud_Base_Super``
  (:3479-3570), the cloud-base critical radius from the crossing of the
  left-hand side ``PL`` and the right-hand side ``PR = C3 w^(3/4)`` of
  Pinsky et al. (2012), eqs. 19 and 21 as the WRF comments cite them
  (:3504, :3519), with the log interpolation of :3551-3565.  The fatal stop
  when no bin satisfies ``PL <= PR`` (``NN = -1``, :3545-3549) becomes
  ``FsbmRefusal``.
* ``koehler_rcriti_in_cloud``: the in-cloud critical dry radius of
  ``WATER_NUCLEATION`` (:3248-3252, :3258-3263).
* ``water_nucleation``: ``WATER_NUCLEATION`` (:3194-3330): the right edge of
  the aerosol spectrum (:3233-3240), the loop that activates one aerosol bin
  per iteration from the largest down (:3242-3319), the partial-bin split in
  ln r (:3286-3302), the placement of activated number into drop bins 1-8
  (:3304-3312), the ``RMASSL_NUCL`` value WRF computes and discards (:3305,
  :3308, :3311, :3313) and the conversion of activated number into the drop
  size distribution (:3321-3327).
* ``jernucl01_warm``: the warm part of ``JERNUCL01_KS`` (:3037-3192): the
  drop-nucleation gate (:3088-3099), the refusal of the homogeneous freezing
  transfer (:3100-3110) and of the crystal nucleation branch (:3113-3182),
  both milestone 3 processes, and the optional water sink (finding D3).
* ``cloud_base_level``: the per-column cloud-base detection of ``FAST_SBM``
  (:4372-4390, the ``DZZ`` lines :4392-4398 that follow feed only the ice
  supersaturation tendency and are not transcribed), under two rules (D9).
* ``is_this_cloud_base``: the per-point cloud-base flag (:4649-4650).
* ``initial_aerosol_column``: the first-step aerosol fill of ``FAST_SBM``
  (:4280-4311): the continental spectrum on land (``XLAND == 1``), the
  maritime spectrum elsewhere, each scaled by the level's air density over
  the column's lowest-level density (:4288, :4299-4305).  ``FACTZ`` (:4290-
  4294) is computed there and never used; it is not transcribed.
* ``_polysvp``: ``POLYSVP`` (:1716-1742), used only by the corrected water
  sink, which recomputes the supersaturations the way the ice branch does
  (:3169-3176).  The same routine is the condensation item's ``polysvp``;
  this private copy lets the two wave-2 items build in parallel and is
  gated here against the ``sweep_polysvp`` oracle records.

Module parameters of ``module_mp_SBM_Nucleation`` used here (:3024-3031):
``Use_cloud_base_nuc = 1``, ``T_NUCL_DROP_MIN = -80.0D0``,
``T_NUCL_ICE_MIN = -37.0D0``, ``ice_nucl_method = 0``,
``ISIGN_TQ_ICENUCL = 1``.  The aerosol composition (``IONS = 3``,
``MWAERO = 115``, ``RO_SOLUTE = 1.79``, :3991-3995) comes from
``constants``; every routine that uses it takes it as a keyword so a
perturbation control can reach it.

Published sources: the Koehler equation (Pruppacher and Klett 1997, chapter
6), whose critical supersaturation ``S* = (4 A^3 / (27 B r^3))^(1/2)``
inverts to the critical dry radius ``r* = (A/3)(4 / (B S^2))^(1/3)``; the
cloud-base droplet concentration of Pinsky, Khain, Mazin and Korolev (2012),
"Analytical estimation of droplet concentration at cloud base", J. Geophys.
Res., 117, D18211.

Conventions
-----------
Float64, pure (array inputs are copied, never modified), CGS inside the
scheme (dyn/cm2, g/cm3, cm/s, K), zero-based arrays that keep Fortran index
order.  Literals are written as WRF writes them (``3.141593``, ``3.3D-05``,
``4.3``).  Sums keep Fortran's sequential order.  Level indices returned or
taken by ``cloud_base_level`` and ``is_this_cloud_base`` are zero based:
WRF's ``KZ_Cloud_Base`` minus one.

Findings this module carries
----------------------------
D3 (design 3.7).  ``WATER_NUCLEATION`` adds activated drops to ``PSI1`` but
takes no vapour and releases no latent heat.  ``FsbmBehavior.wrf_v480()``
reproduces that; with ``nucleation_water_sink`` (the corrected preset) the
activated drop mass is removed from ``qv`` and ``AL1`` times it is added to
``T``, then both supersaturations are recomputed, exactly as the crystal
branch does for ice (:3129-3179).  The mass is the change in the liquid water
mixing ratio of ``PSI1``, formed as the ice branch forms it
(``COL3 RORI sum(f x x)`` before and after, each clipped at zero).

D3, detail found while transcribing.  The ``RMASSL_NUCL`` WRF computes is
not the activated mass and cannot serve as the sink: it is reset at the top
of every iteration (:3305), so on return it holds only the last bin's value,
and it multiplies the activated number by ``x_k^2 COL 3 / rho`` (:3308,
:3311, :3313) where the drop mass it describes is number times ``x_k / rho``.
It is therefore the last bin's activated mass times ``3 COL x_k``, about
``2.3e-11`` for drop bin 1.  ``water_nucleation`` returns it as
``rmassl_nucl`` (and every iteration's value in ``rmassl_nucl_iterations``)
for the record, and returns the true mass beside it.

D9 (oracle manifest).  The cloud-base test at :4387 joins its terms with
``.and.`` around ``K_found .eqv. .FALSE.``; ``.EQV.`` binds more loosely than
``.AND.``, so WRF evaluates ``(SUP_WATER > 0 .and. w > 10 cm/s .and.
K_found) .eqv. (.FALSE. .and. ...)``.  ``KZ_Cloud_Base`` therefore ends as
the highest level that is not (supersaturated with w above 10 cm/s), or
level 1, never the lowest cloudy updraft level above the second that the
code's comment and the design describe.  ``cloud_base_level`` reproduces
WRF under ``rule="wrf_v480"`` and applies the intended rule under
``rule="first_level"`` (``CLOUD_BASE_RULES``, defined in ``constants``).
``FsbmBehavior.cloud_base_rule`` selects the rule: the ``wrf_v480()``
preset takes WRF's evaluation and the ``corrected()`` preset (and ``None``,
the corrected reference default) the intended rule.  The field has no
default, so an A/B arm made from a preset with one unrelated field changed
(D1, D3) keeps its preset's rule and never switches the cloud-base
detection with it.  An object that carries no ``cloud_base_rule`` is
refused unless the caller passes ``rule=``, and an explicit ``rule=`` is
always taken as given.

Unreachable quirk, recorded.  When the critical radius falls in aerosol bin
2 (``NCRITI == 1``, :3293-3297) WRF splits bin 2 with the geometry of bin 1
(from ``RCCN(1)/10000`` to ``RCCN(1)``), so the split does not conserve
aerosol number there, and ``DLN2`` is negative whenever the critical radius
lies above ``RCCN(1)``.  Reaching bin 2 (a critical dry radius near 0.16 nm)
needs a supersaturation of several hundred percent, so no physical state
gets there; the port transcribes it as written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gpuwm.core.fast_sbm.constants import (
    AA1_MY,
    AA2_MY,
    AL1,
    BB1_MY,
    BB2_MY,
    CLOUD_BASE_RULES,
    COL,
    IONS,
    MWAERO,
    NKR,
    NKR_AEROSOL,
    RO_SOLUTE,
    FsbmBehavior,
    FsbmRefusal,
)

# ---------------------------------------------------------------------------
# module_mp_SBM_Nucleation parameters (:3024-3031)
# ---------------------------------------------------------------------------

#: 1: the cloud-base branch is used where ``Is_This_CloudBase == 1`` (:3024).
USE_CLOUD_BASE_NUC = 1
#: Drops nucleate only above this temperature, deg C (:3025).
T_NUCL_DROP_MIN = -80.0
#: Crystals nucleate only at or above this temperature, deg C (:3026).
T_NUCL_ICE_MIN = -37.0
#: Homogeneous freezing transfer threshold, deg C (:3101).
T_HOMOGENEOUS_FREEZING = -38.0
#: Number of the drop bins activated number can be placed in (:3306-3311,
#: :3323).
N_DROP_BINS_NUCLEATED = 8

# Literals of the cloud-base detection (:4383-4387).
_W_THRESHOLD_CGS = 0.1 * 1.0e2        # 0.1*1.0D2, cm/s
_Z_CLOUD_BASE_MAX_CGS = 3.0 * 1.0e5   # 3.0*1.0D5, cm
#: Drop concentration bound of the cloud-base flag (:4650): ``col*sum(f x)``
#: is one third of the drop number per cm3.
_CLOUD_BASE_DROP_BOUND = 5.0

_M3_ICE = ("crystal nucleation (ICE_NUCL, Meyers et al. 1992, :3113-3182) "
           "with its vapour sink and sublimation heating")
_M3_HOMOGENEOUS = ("homogeneous freezing of every drop bin into plates with "
                   "the latent heat of fusion (:3100-3110)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fpow(x: float, y: float) -> float:
    """``x**y`` with a real exponent, as Fortran evaluates it through pow.

    A negative base with a non-integer exponent gives NaN (Python's ``**``
    would return a complex number there).
    """

    try:
        return math.pow(x, y)
    except ValueError:
        return math.nan
    except OverflowError:
        return math.inf


def _seq_sum(values: np.ndarray) -> float:
    """Sum in increasing index order, as Fortran's SUM accumulates."""

    total = 0.0
    for v in values:
        total = total + float(v)
    return total


def _vector(name: str, value, n: int) -> np.ndarray:
    arr = np.array(value, dtype=np.float64)
    if arr.shape != (n,):
        raise FsbmRefusal("fast_sbm.nucleation",
                          f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _oper3(ar1: float, ar2: float) -> float:
    """``OPER3(AR1,AR2) = AR1*AR2/(0.622D0+0.378D0*AR1)`` (:3068, :3224)."""

    return ar1 * ar2 / (0.622 + 0.378 * ar1)


def _polysvp(tt: float, itype: int) -> float:
    """``POLYSVP`` (:1716-1742): water (0) or ice (1) saturation, dyn/cm2."""

    if itype == 0:
        return AA1_MY * math.exp(-BB1_MY / tt)
    if itype == 1:
        return AA2_MY * math.exp(-BB2_MY / tt)
    raise FsbmRefusal("POLYSVP :1727", f"ITYPE {itype!r} is not 0 or 1")


def _koehler_b(ro_solute: float, ions: int, mwaero: float) -> float:
    """``BKOE`` as :3251-3252 and :3683-3684 form it."""

    bkoe = ions * 4.3 / mwaero
    return bkoe * (4.0 / 3.0) * 3.141593 * ro_solute


# ---------------------------------------------------------------------------
# supmax_COEFF (:3572-3733)
# ---------------------------------------------------------------------------

def supmax_coeff(pp: float, tt: float, *, ro_solute: float = RO_SOLUTE,
                 ions: int = IONS, mwaero: float = MWAERO
                 ) -> tuple[float, float, float]:
    """``supmax_COEFF`` (:3572-3733): ``(AKOE, BKOE, C3)``.

    ``pp`` is pressure in dyn/cm2 and ``tt`` temperature in K.  ``AKOE`` (cm)
    and ``BKOE`` are the Koehler constants with a temperature-dependent
    surface tension, water density and latent heat (:3627-3684); ``C3`` is the
    coefficient of ``PR = C3 w^(3/4)``, Pinsky et al. (2012) eq. 21 as the
    comment at :3504 cites it (:3688-3728).  WRF's literals and evaluation
    order are kept, including the rounded ``PI = 3.141593D0`` (:3590) and the
    ``tpc + TZERO`` round trip.
    """

    rv_my = 461.5e4
    cp = 1005.0e4
    g = 9.8e2
    rd_my = 287.0e4
    pi = 3.141593

    tt = float(tt)
    pp = float(pp)
    tpc = tt - 273.15
    pzero = 1.01325e6
    tzero = 273.15

    alw1 = (-6.143419998e-2 * _fpow(tpc, 3.0) + 1.58927 * _fpow(tpc, 2.0)
            - 2.36418e3 * tpc + 2.50079e6)
    alw1 = alw1 * 10.0e3

    if tpc < -5.5:
        sw = (5.285e-11 * _fpow(tpc, 6.0) + 6.283e-9 * _fpow(tpc, 5.0)
              + 2.933e-7 * _fpow(tpc, 4.0) + 6.511e-6 * _fpow(tpc, 3.0)
              + 6.818e-5 * _fpow(tpc, 2.0) + 1.15e-4 * tpc + 7.593e-2)
    else:
        sw = -1.55e-4 * tpc + 7.566165e-2
    sw = sw * 10.0e2

    if tpc < 0.0:
        ro_w = (-7.497e-9 * _fpow(tpc, 6.0) - 3.6449e-7 * _fpow(tpc, 5.0)
                - 6.9987e-6 * _fpow(tpc, 4.0) + 1.518e-4 * _fpow(tpc, 3.0)
                - 8.486e-3 * _fpow(tpc, 2.0) + 6.69e-2 * tpc + 9.9986e2)
    else:
        ro_w = ((-3.932952e-10 * _fpow(tpc, 5.0) + 1.497562e-7 * _fpow(tpc, 4.0)
                 - 5.544846e-5 * _fpow(tpc, 3.0) - 7.92221e-3 * _fpow(tpc, 2.0)
                 + 1.8224944e1 * tpc + 9.998396e2)
                / (1.0 + 1.8159725e-2 * tpc))
    ro_w = ro_w * 1.0e-3

    hc = 7.1128e-5 * tpc + 2.380696e-2
    hc = hc * 10.0e4

    ew = (6.38780966e-9 * _fpow(tpc, 6.0) + 2.03886313e-6 * _fpow(tpc, 5.0)
          + 3.02246994e-4 * _fpow(tpc, 4.0) + 2.65027242e-2 * _fpow(tpc, 3.0)
          + 1.43053301 * _fpow(tpc, 2.0) + 4.43986062e1 * tpc + 6.1117675e2)
    ew = ew * 10.0

    akoe = 2.0 * sw / (rv_my * ro_w * (tpc + tzero))
    bkoe = ions * 4.3 / mwaero
    bkoe = bkoe * (4.0 / 3.0) * pi * ro_solute

    ro_v = ew / (rv_my * (tpc + tzero))
    dv = 0.211 * (pzero / pp) * _fpow((tpc + tzero) / tzero, 1.94)
    ro_a = pzero / ((tpc + tzero) * rd_my)
    fl = (ro_w * _fpow(alw1, 2.0)) / (hc * rv_my * _fpow(tpc + tzero, 2.0))
    fr = ro_w * rv_my * (tpc + tzero) / (ew * dv)
    f = fl + fr
    qv = ro_v / ro_a
    a1 = ((g * alw1 / (cp * rv_my * _fpow(tpc + tzero, 2.0)))
          - (g / (rd_my * (tpc + tzero))))
    a2 = (1.0 / qv) + (_fpow(alw1, 2.0)) / (cp * rv_my * _fpow(tpc + tzero, 2.0))
    c1 = 1.058
    c3 = (c1 * _fpow(f * a1 / 3.0, 0.75)
          * math.sqrt(3.0 * ro_a / (4.0 * pi * ro_w * a2)))
    return akoe, bkoe, c3


# ---------------------------------------------------------------------------
# Cloud_Base_Super (:3479-3570)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CloudBaseSuperResult:
    """Everything ``Cloud_Base_Super`` forms on its way to ``RCRITI``.

    ``nn`` is WRF's one-based ``NN``, the first aerosol bin whose ``PL`` is at
    or below ``PR``.  ``pl`` and ``supmax`` are zero-based copies of the
    routine's arrays (``pl[0]`` is ``PL(1) = 0``, never assigned by the loop;
    ``supmax`` keeps its fill value 999 where the loop never reached).
    ``dl1`` and ``dl2`` are ``|PL(NN-1) - PR|`` and ``|PL(NN) - PR|``.
    """

    rcriti: float
    nn: int
    pr: float
    pl: np.ndarray
    supmax: np.ndarray
    dl1: float
    dl2: float
    akoe: float
    bkoe: float
    c3: float


def cloud_base_super(fccnr, rccn, tt: float, pp: float, wbase: float,
                     col: float = COL, *, ro_solute: float = RO_SOLUTE,
                     ions: int = IONS, mwaero: float = MWAERO
                     ) -> CloudBaseSuperResult:
    """``Cloud_Base_Super`` (:3479-3570) with its intermediate values.

    ``fccnr`` (43,) is the aerosol distribution in cm-3 per unit ln r,
    ``rccn`` (43,) the dry radii in cm, ``wbase`` the updraft in cm/s.  For
    each bin ``NR`` from 2 upward, ``PL(NR) = supmax(NR) sqrt(N_act(NR))``
    with ``supmax(NR)`` the Koehler critical supersaturation of that dry
    radius and ``N_act(NR) = sum_{KR >= NR} COL FCCNR(KR)``; the first bin
    with ``PL <= PR`` fixes ``NN`` and the critical radius is interpolated in
    ln r between ``RCCN(NN-1)`` and ``RCCN(NN)`` by the distances of ``PL``
    from ``PR`` (:3556-3565).  No such bin is WRF's fatal stop (:3545-3549),
    raised here as ``FsbmRefusal``; a negative ``wbase`` makes ``PR`` NaN and
    reaches that stop, as it does in WRF.
    """

    fccnr = np.array(fccnr, dtype=np.float64)
    rccn = np.array(rccn, dtype=np.float64)
    nkr = fccnr.shape[0]
    if fccnr.ndim != 1 or rccn.shape != fccnr.shape or nkr < 2:
        raise FsbmRefusal("Cloud_Base_Super",
                          f"fccnr {fccnr.shape} and rccn {rccn.shape} must be "
                          "equal one-dimensional aerosol grids")
    akoe, bkoe, c3 = supmax_coeff(pp, tt, ro_solute=ro_solute, ions=ions,
                                  mwaero=mwaero)
    col = float(col)
    pr = c3 * _fpow(float(wbase), 0.75)
    supmax = np.full(nkr, 999.0)
    pl = np.zeros(nkr)
    nn = -1
    for nr in range(2, nkr + 1):
        supmax[nr - 1] = math.sqrt((4.0 * _fpow(akoe, 3.0))
                                   / (27.0 * _fpow(float(rccn[nr - 1]), 3.0) * bkoe))
        ccnconact = 0.0
        for kr in range(nr, nkr + 1):
            ccnconact = ccnconact + col * float(fccnr[kr - 1])
        pl[nr - 1] = supmax[nr - 1] * math.sqrt(ccnconact)
        if pl[nr - 1] <= pr:
            nn = nr
            break
    if nn == -1:
        raise FsbmRefusal(
            "Cloud_Base_Super :3545-3549",
            "NN is not defined in cloud base routine, model stop: no aerosol "
            f"bin has PL <= PR (PR = {pr!r} from wbase = {float(wbase)!r} cm/s "
            f"and C3 = {c3!r}; smallest PL = {float(np.min(pl[1:]))!r})")
    dl1 = abs(float(pl[nn - 2]) - pr)
    dl2 = abs(float(pl[nn - 1]) - pr)
    rcriti = float(rccn[nn - 2]) * math.exp(col * dl1 / (dl1 + dl2))
    return CloudBaseSuperResult(rcriti=rcriti, nn=nn, pr=pr, pl=pl,
                                supmax=supmax, dl1=dl1, dl2=dl2, akoe=akoe,
                                bkoe=bkoe, c3=c3)


def cloud_base_rcriti(fccnr, rccn, tt: float, pp: float, wbase: float,
                      col: float = COL, *, ro_solute: float = RO_SOLUTE,
                      ions: int = IONS, mwaero: float = MWAERO) -> float:
    """The critical dry radius (cm) ``Cloud_Base_Super`` returns (:3565)."""

    return cloud_base_super(fccnr, rccn, tt, pp, wbase, col,
                            ro_solute=ro_solute, ions=ions,
                            mwaero=mwaero).rcriti


def koehler_rcriti_in_cloud(tt: float, sup1: float, *,
                            ro_solute: float = RO_SOLUTE, ions: int = IONS,
                            mwaero: float = MWAERO) -> float:
    """In-cloud critical dry radius, cm (:3248-3252, :3259 and :3263).

    ``RCRITI = (AKOE/3) (4/BKOE/SUP1/SUP1)^(1/3)`` with ``AKOE = 3.3D-05/TT``
    and ``BKOE = IONS 4.3/MWAERO (4/3) 3.141593 RO_SOLUTE``: the Koehler
    critical dry radius at supersaturation ``SUP1`` (a fraction).
    """

    akoe = 3.3e-05 / float(tt)
    bkoe = _koehler_b(ro_solute, ions, mwaero)
    deg01 = 1.0 / 3.0
    s = float(sup1)
    return (akoe / 3.0) * _fpow(4.0 / bkoe / s / s, deg01)


# ---------------------------------------------------------------------------
# WATER_NUCLEATION (:3194-3330)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NucleationStep:
    """One pass of the loop at :3245-3319.

    ``imax`` and ``ncriti`` are WRF's one-based bin numbers; ``split`` is
    ``"whole"`` (:3299-3301), ``"partial"`` (:3288-3292) or ``"partial_bin1"``
    (:3293-3297); ``drop_bin`` is the one-based drop bin the number went to;
    ``ccnconc`` the activated number, cm-3.  ``cloud_base`` holds the
    ``Cloud_Base_Super`` result when that branch set ``rcriti``.
    """

    imax: int
    ncriti: int
    rcriti: float
    split: str
    ccnconc: float
    drop_bin: int
    rmassl_nucl: float
    cloud_base: CloudBaseSuperResult | None


@dataclass(frozen=True)
class WaterNucleationResult:
    """What ``WATER_NUCLEATION`` returns, and what it computes and drops.

    ``psi1`` (33,) and ``fccnr`` (43,) are the updated distributions;
    ``dropconcn`` (33,) the activated number per drop bin, cm-3 (only bins 1-8
    can be non-zero).  ``tt``, ``qq`` and ``sup1`` are returned unchanged, as
    WRF leaves them.  ``branch`` is ``"cloud_base"`` or ``"in_cloud"``;
    ``exit_rcriti`` is the critical radius of the pass that ended the loop by
    ``RCRITI >= RCCN(IMAX)`` (:3266), or ``None`` when the loop ran out of
    bins.  ``rmassl_nucl`` is WRF's ``RMASSL_NUCL`` on return (the last
    pass's value; see finding D3 in the module docstring) and
    ``rmassl_nucl_iterations`` every pass's value.  ``activated_number`` is
    ``sum(dropconcn)`` in cm-3 and ``activated_mass`` the increase of the
    liquid water mixing ratio (g/g), ``COL3 RORI sum(f x x)`` after minus
    before, the form the ice branch uses (:3134-3161).  As the difference of
    two whole-spectrum sums it carries a rounding error of order machine
    epsilon times the liquid already present, a relative error of about
    ``(liquid before / activated_mass)`` epsilon against the direct
    ``sum(dropconcn xl) / ror``; the budget test checks it against that
    direct sum so the budget closure does not hold only by construction.
    """

    psi1: np.ndarray
    fccnr: np.ndarray
    dropconcn: np.ndarray
    tt: float
    qq: float
    sup1: float
    branch: str
    steps: tuple[NucleationStep, ...]
    exit_rcriti: float | None
    rmassl_nucl: float
    rmassl_nucl_iterations: tuple[float, ...]
    activated_number: float
    activated_mass: float


def liquid_mass_mixing_ratio(psi1, xl, ror: float, col: float = COL) -> float:
    """``COL3 RORI sum(PSI1 XL XL)``, g/g, as :3134-3143 forms the ice mass.

    The sum runs bin by bin in increasing order and is clipped at zero
    afterwards, as :3145 and :3159 clip the ice sums.
    """

    psi1 = np.asarray(psi1, dtype=np.float64)
    xl = np.asarray(xl, dtype=np.float64)
    total = 0.0
    for k in range(psi1.shape[0]):
        f = float(psi1[k])
        x = float(xl[k])
        total = total + f * x * x
    col3 = 3.0 * float(col)
    rori = 1.0 / float(ror)
    total = total * col3 * rori
    return 0.0 if total < 0.0 else total


def water_nucleation(fccnr, psi1, xl, tt: float, qq: float, ror: float,
                     sup1: float, pp: float, is_cloud_base: int, w: float,
                     rccn, col: float = COL, *, ro_solute: float = RO_SOLUTE,
                     ions: int = IONS, mwaero: float = MWAERO
                     ) -> WaterNucleationResult:
    """``WATER_NUCLEATION`` (:3194-3330).

    ``fccnr`` (43,) aerosol, cm-3 per unit ln r; ``psi1`` (33,) drops, the
    f-form in 1/(g cm3); ``xl`` (33,) drop bin masses, g; ``ror`` air density,
    g/cm3; ``sup1`` water supersaturation (fraction); ``pp`` pressure,
    dyn/cm2; ``w`` the updraft at the point, cm/s (only the cloud-base branch
    reads it); ``rccn`` (43,) aerosol dry radii, cm.

    Each pass takes the largest non-empty aerosol bin ``IMAX``, forms the
    critical dry radius (``Cloud_Base_Super`` where ``is_cloud_base == 1``,
    the Koehler radius at ``sup1`` elsewhere), stops if it is at or above
    ``RCCN(IMAX)``, and otherwise activates bin ``IMAX``: wholly when the
    critical radius lies below ``RCCN(IMAX-1)``, or the part of the bin above
    it in ln r (the partial split).  Aerosol bins up to ``NKR-7`` (36) feed
    drop bin 1, bins 37-43 drop bins 2-8.  The activated number becomes
    ``PSI1(KR) += DROPCONCN(KR)/(3 COL XL(KR))`` for bins 1-8.  WRF passes
    ``NKR_aerosol`` as the routine's ``NKR`` (:3097), so ``NKR`` here is 43.
    """

    fccnr = _vector("fccnr", fccnr, NKR_AEROSOL)
    rccn = _vector("rccn", rccn, NKR_AEROSOL)
    psi1 = _vector("psi1", psi1, NKR)
    psi1_in = psi1.copy()
    xl = _vector("xl", xl, NKR)
    tt = float(tt)
    qq = float(qq)
    sup1 = float(sup1)
    ror = float(ror)
    col = float(col)
    nkr = NKR_AEROSOL
    cloud_base = int(is_cloud_base) == 1
    branch = ("cloud_base" if (USE_CLOUD_BASE_NUC == 1 and cloud_base)
              else "in_cloud")

    dropconcn = np.zeros(NKR)
    deg01 = 1.0 / 3.0
    rori = 1.0 / ror

    imax = nkr
    for i in range(imax, 0, -1):
        if fccnr[i - 1] > 0.0:
            imax = i
            break

    ncriti = 0
    steps: list[NucleationStep] = []
    rmassl_nucl = 0.0
    rmassl_iters: list[float] = []
    exit_rcriti: float | None = None
    while imax >= ncriti:
        akoe = 3.3e-05 / tt
        bkoe = ions * 4.3 / mwaero
        bkoe = bkoe * (4.0 / 3.0) * 3.141593 * ro_solute
        cb_result = None
        if branch == "cloud_base":
            cb_result = cloud_base_super(fccnr, rccn, tt, float(pp), float(w),
                                         col, ro_solute=ro_solute, ions=ions,
                                         mwaero=mwaero)
            rcriti = cb_result.rcriti
        else:
            rcriti = (akoe / 3.0) * _fpow(4.0 / bkoe / sup1 / sup1, deg01)

        if rcriti >= rccn[imax - 1]:
            exit_rcriti = rcriti
            break

        ncriti = imax
        while rcriti <= rccn[ncriti - 1] and ncriti > 1:
            ncriti = ncriti - 1

        rccn_minimum = float(rccn[0]) / 10000.0
        ccnconc_imax = 0.0
        if ncriti == imax - 1:
            if ncriti > 1:
                dln1 = math.log(rcriti) - math.log(float(rccn[imax - 2]))
                dln2 = col - dln1
                ccnconc_imax = dln2 * float(fccnr[imax - 1])
                fccnr[imax - 1] = float(fccnr[imax - 1]) * dln1 / col
                split = "partial"
            else:
                dln1 = math.log(rcriti) - math.log(rccn_minimum)
                dln2 = math.log(float(rccn[0])) - math.log(rcriti)
                ccnconc_imax = dln2 * float(fccnr[imax - 1])
                fccnr[imax - 1] = float(fccnr[imax - 1]) * dln1 / (dln1 + dln2)
                split = "partial_bin1"
        else:
            ccnconc_imax = col * float(fccnr[imax - 1])
            fccnr[imax - 1] = 0.0
            split = "whole"

        rmassl_nucl = 0.0
        if imax <= nkr - 7:
            drop_bin = 1
        else:
            drop_bin = 8 - (nkr - imax)
        dropconcn[drop_bin - 1] = dropconcn[drop_bin - 1] + ccnconc_imax
        x = float(xl[drop_bin - 1])
        rmassl_nucl = rmassl_nucl + ccnconc_imax * x * x
        rmassl_nucl = rmassl_nucl * col * 3.0 * rori
        rmassl_iters.append(rmassl_nucl)
        steps.append(NucleationStep(imax=imax, ncriti=ncriti, rcriti=rcriti,
                                    split=split, ccnconc=ccnconc_imax,
                                    drop_bin=drop_bin, rmassl_nucl=rmassl_nucl,
                                    cloud_base=cb_result))
        imax = imax - 1

    if _seq_sum(dropconcn) > 0.0:
        for kr in range(1, N_DROP_BINS_NUCLEATED + 1):
            dx = 3.0 * col * float(xl[kr - 1])
            psi1[kr - 1] = float(psi1[kr - 1]) + float(dropconcn[kr - 1]) / dx

    activated_mass = (liquid_mass_mixing_ratio(psi1, xl, ror, col)
                      - liquid_mass_mixing_ratio(psi1_in, xl, ror, col))
    return WaterNucleationResult(
        psi1=psi1, fccnr=fccnr, dropconcn=dropconcn, tt=tt, qq=qq, sup1=sup1,
        branch=branch, steps=tuple(steps), exit_rcriti=exit_rcriti,
        rmassl_nucl=rmassl_nucl, rmassl_nucl_iterations=tuple(rmassl_iters),
        activated_number=_seq_sum(dropconcn), activated_mass=activated_mass)


# ---------------------------------------------------------------------------
# JERNUCL01_KS, warm part (:3037-3192)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NucleationResult:
    """What the warm ``JERNUCL01_KS`` hands back to ``FAST_SBM``.

    ``psi1`` (33,), ``fccnr`` (43,), ``tt`` (K), ``qq`` (g/g), ``sup1`` and
    ``sup2`` are the routine's inout arguments after the call.  ``branch`` is
    ``"cloud_base"``, ``"in_cloud"`` or ``None`` when ``WATER_NUCLEATION`` was
    not called (no water supersaturation, or no aerosol); ``activated`` says
    whether any drop was added.  ``delmassl`` is the activated drop mass
    (g/g); ``sink_applied`` says whether it was taken from ``qq`` with its
    latent heat (``FsbmBehavior.nucleation_water_sink``).  ``water`` is the
    full ``WATER_NUCLEATION`` record.
    """

    psi1: np.ndarray
    fccnr: np.ndarray
    tt: float
    qq: float
    sup1: float
    sup2: float
    branch: str | None
    activated: bool
    delmassl: float
    sink_applied: bool
    water: WaterNucleationResult | None


def jernucl01_warm(psi1, fccnr, xl, tt: float, qq: float, ror: float,
                   pp: float, sup1: float, sup2: float, w: float,
                   is_cloud_base: int, rccn, col: float = COL,
                   behavior: FsbmBehavior | None = None, *,
                   ro_solute: float = RO_SOLUTE, ions: int = IONS,
                   mwaero: float = MWAERO) -> NucleationResult:
    """The warm branch of ``JERNUCL01_KS`` (:3037-3192).

    Drops nucleate when ``SUP1 > 0`` and ``T - 273.15 > T_NUCL_DROP_MIN``
    (:3094) and the aerosol holds any number (:3095).  Two parts of the
    routine are milestone 3 processes and refuse here by name
    (``FsbmRefusal``), at the point WRF would enter them: the transfer of
    every drop into plates when the drop branch runs at or below -38 C
    (:3100-3110), and the crystal nucleation branch, entered when
    ``T < 0 C``, ``T >= T_NUCL_ICE_MIN`` and ``SUP2 > 0`` (:3113-3182).
    A warm state never reaches either.

    ``behavior`` (default ``FsbmBehavior.corrected()``): with
    ``nucleation_water_sink`` the activated drop mass ``DELMASSL`` is removed
    from ``qq``, ``AL1 DELMASSL`` is added to ``tt``, and ``sup1`` and
    ``sup2`` are recomputed from ``POLYSVP`` and ``OPER3`` at the new state,
    the sequence the ice branch applies to crystals (:3129-3179) with ``AL1``
    (2500, :3220) in place of ``AL2``.  The activation itself is WRF's in
    both presets; the ``wrf_v480`` preset leaves ``tt``, ``qq``, ``sup1`` and
    ``sup2`` untouched (finding D3).
    """

    if behavior is None:
        behavior = FsbmBehavior.corrected()
    psi1 = _vector("psi1", psi1, NKR)
    fccnr = _vector("fccnr", fccnr, NKR_AEROSOL)
    xl = _vector("xl", xl, NKR)
    rccn = _vector("rccn", rccn, NKR_AEROSOL)
    tt = float(tt)
    qq = float(qq)
    sup1 = float(sup1)
    sup2 = float(sup2)
    ror = float(ror)
    pp = float(pp)
    col = float(col)

    tpc = tt - 273.15
    branch = None
    water = None
    delmassl = 0.0
    sink_applied = False
    activated = False
    if sup1 > 0.0 and tpc > T_NUCL_DROP_MIN:
        if tpc <= T_HOMOGENEOUS_FREEZING:
            raise FsbmRefusal(
                "JERNUCL01_KS :3101",
                f"T = {tt!r} K is at or below -38 C with water "
                f"supersaturation {sup1!r}: WRF would run the {_M3_HOMOGENEOUS}, "
                "a milestone 3 ice process the warm reference does not carry")
        if _seq_sum(fccnr) > 0.0:
            water = water_nucleation(fccnr, psi1, xl, tt, qq, ror, sup1, pp,
                                     is_cloud_base, w, rccn, col,
                                     ro_solute=ro_solute, ions=ions,
                                     mwaero=mwaero)
            branch = water.branch
            activated = bool(np.any(water.psi1 != psi1))
            psi1 = water.psi1
            fccnr = water.fccnr
            delmassl = water.activated_mass
            if behavior.nucleation_water_sink and activated:
                qpn = qq - delmassl
                qq = qpn
                tpn = tt + AL1 * delmassl
                tt = tpn
                es1n = _polysvp(tpn, 0)
                es2n = _polysvp(tpn, 1)
                ew1n = _oper3(qpn, pp)
                sup1 = ew1n / es1n - 1.0
                sup2 = ew1n / es2n - 1.0
                sink_applied = True

    del2n = 100.0 * sup2
    tpc = tt - 273.15
    if tpc < 0.0 and tpc >= T_NUCL_ICE_MIN and del2n > 0.0:
        raise FsbmRefusal(
            "JERNUCL01_KS :3117",
            f"T = {tt!r} K is below 0 C with ice supersaturation {sup2!r}: "
            f"WRF would run the {_M3_ICE}, a milestone 3 ice process the warm "
            "reference does not carry")

    return NucleationResult(psi1=psi1, fccnr=fccnr, tt=tt, qq=qq, sup1=sup1,
                            sup2=sup2, branch=branch, activated=activated,
                            delmassl=delmassl, sink_applied=sink_applied,
                            water=water)


# ---------------------------------------------------------------------------
# FAST_SBM: cloud-base detection (:4372-4390) and flag (:4649-4650)
# ---------------------------------------------------------------------------

def cloud_base_rule(behavior: FsbmBehavior | None = None) -> str:
    """The cloud-base detection rule a behaviour selects (finding D9).

    ``FsbmBehavior.cloud_base_rule`` decides: ``"wrf_v480"`` in the
    ``wrf_v480()`` preset, ``"first_level"`` (the intended rule) in the
    ``corrected()`` preset, and ``None`` is the corrected reference default.
    An A/B arm built from a preset with one field changed keeps that
    preset's rule, so an arm meant to isolate D1 or D3 against ``wrf_v480``
    does not also move KC-1's cloud-base level from 40 to levels 6-13 or
    none.

    An object that carries no ``cloud_base_rule`` is refused: mapping it to
    either rule would pick the ``Cloud_Base_Super`` points for it silently.
    Such a caller passes ``rule=`` to ``cloud_base_level`` explicitly.
    """

    if behavior is None:
        return FsbmBehavior.corrected().cloud_base_rule
    own = getattr(behavior, "cloud_base_rule", None)
    if own is None:
        raise FsbmRefusal(
            "cloud_base_rule",
            f"{behavior!r} carries no cloud_base_rule field, so it does not "
            "say which cloud-base rule (finding D9) it wants; pass rule= "
            f"explicitly, one of {CLOUD_BASE_RULES}")
    if own not in CLOUD_BASE_RULES:
        raise FsbmRefusal("cloud_base_rule",
                          f"cloud_base_rule {own!r} is not one of "
                          f"{CLOUD_BASE_RULES}")
    return own


def cloud_base_level(t_new, qv, p_cgs, w_full, zcgs, *,
                     behavior: FsbmBehavior | None = None,
                     rule: str | None = None) -> int | None:
    """The cloud-base level of one column, zero based, or ``None``.

    ``t_new`` (nz,) K, ``qv`` (nz,) g/g (the current, advected vapour),
    ``p_cgs`` (nz,) dyn/cm2, ``w_full`` (nz+1,) vertical velocity on full
    levels in m/s, ``zcgs`` (nz,) height of each level above the surface, cm.
    Per level (:4379-4386): ``SUP_WATER = EW1N/ES1N - 1`` with
    ``ES1N = AA1_MY exp(-BB1_MY/T_NEW)`` and ``EW1N = qv p/(0.622+0.378 qv)``,
    and ``w`` averaged to the mass level, ``50 (w(k) + w(k+1))`` cm/s, or
    ``100 w(k)`` at the top level.

    ``rule="first_level"``: the first level above the second (WRF ``K > 2``)
    with ``SUP_WATER > 0``, ``w > 10 cm/s`` and ``zcgs < 3 km``, as the code's
    comment and the design describe; ``None`` when no level qualifies.
    ``rule="wrf_v480"``: :4387 as gfortran evaluates it (finding D9), the
    highest level that is not (supersaturated with w above 10 cm/s), level 1
    when every level above it is; never ``None``.  Without ``rule`` the rule
    follows ``behavior`` through ``cloud_base_rule``, which refuses an
    object that carries no ``cloud_base_rule`` field; an explicit ``rule`` is
    always taken as given.  The bare call of the design contract (4.2), with
    neither, gives the corrected rule, so a caller replaying ``wrf_v480``
    against the oracle (the warm column) passes ``behavior=`` or ``rule=``:
    on the KC-1 replay WRF's rule gives level 40 (one based) at all 64
    column steps and the intended rule levels 6-13 at 10 of them and no
    level at the other 54.
    WRF runs this detection only under ``ICEPROCS == 1`` (:4372), its
    compile-time default.
    """

    if rule is None:
        rule = cloud_base_rule(behavior)
    if rule not in CLOUD_BASE_RULES:
        raise FsbmRefusal("cloud_base_level",
                          f"rule {rule!r} is not one of {CLOUD_BASE_RULES}")
    t_new = np.asarray(t_new, dtype=np.float64)
    nz = t_new.shape[0]
    qv = _vector("qv", qv, nz)
    p_cgs = _vector("p_cgs", p_cgs, nz)
    zcgs = _vector("zcgs", zcgs, nz)
    w_full = _vector("w_full", w_full, nz + 1)
    if t_new.ndim != 1 or nz < 1:
        raise FsbmRefusal("cloud_base_level",
                          f"t_new must be a non-empty column, got {t_new.shape}")

    kz: int | None = None
    k_found = False
    for k in range(nz):
        es1n = AA1_MY * math.exp(-BB1_MY / float(t_new[k]))
        q = float(qv[k])
        ew1n = q * float(p_cgs[k]) / (0.622 + 0.378 * q)
        sup_water = ew1n / es1n - 1.0
        if k < nz - 1:
            w_stag = 50.0 * (float(w_full[k]) + float(w_full[k + 1]))
        else:
            w_stag = 100.0 * float(w_full[k])
        cloudy_updraft = sup_water > 0.0 and w_stag > _W_THRESHOLD_CGS
        if rule == "wrf_v480":
            # (A .and. B .and. K_found) .eqv. (.FALSE. .and. ...), whose right
            # side is always false, so the test is .NOT.(A .and. B .and. K_found)
            take = not (cloudy_updraft and k_found)
        else:
            take = (cloudy_updraft and not k_found and k + 1 > 2
                    and float(zcgs[k]) < _Z_CLOUD_BASE_MAX_CGS)
        if take:
            kz = k
            k_found = True
    return kz


def is_this_cloud_base(k: int, kz_cloud_base: int | None, ff1in, xl,
                       col: float = COL) -> bool:
    """``Is_This_CloudBase`` of one point (:4649-4650).

    True when level ``k`` is the column's cloud-base level (both zero based)
    and ``COL sum(FF1IN XL) < 5``, that is fewer than 15 drops per cm3.
    """

    if kz_cloud_base is None or int(k) != int(kz_cloud_base):
        return False
    ff1in = _vector("ff1in", ff1in, NKR)
    xl = _vector("xl", xl, NKR)
    return float(col) * _seq_sum(ff1in * xl) < _CLOUD_BASE_DROP_BOUND


# ---------------------------------------------------------------------------
# FAST_SBM: initial aerosol (:4280-4311)
# ---------------------------------------------------------------------------

def initial_aerosol_column(fccnr_con, fccnr_mar, rho_cgs,
                           is_land: bool) -> np.ndarray:
    """The first-step aerosol of one column, ``(nz, 43)``, cm-3 per ln r.

    ``chem_new = (FCCNR_CON/rhoair_max) rhocgs(k)`` on land (``XLAND == 1``,
    ``is_land``) and ``FCCNR_MAR`` in its place elsewhere, ``rhoair_max``
    being the column's lowest-level density (:4288, :4299-4305).  This is the
    scheme-internal form (``FsbmDistributions.fccn``); the exit conversion
    (:5297) turns it into the advected number per kg.
    """

    fccnr_con = _vector("fccnr_con", fccnr_con, NKR_AEROSOL)
    fccnr_mar = _vector("fccnr_mar", fccnr_mar, NKR_AEROSOL)
    rho_cgs = np.asarray(rho_cgs, dtype=np.float64)
    if rho_cgs.ndim != 1 or rho_cgs.shape[0] < 1:
        raise FsbmRefusal("initial_aerosol_column",
                          f"rho_cgs must be a non-empty column, got {rho_cgs.shape}")
    source = fccnr_con if bool(is_land) else fccnr_mar
    rhoair_max = float(rho_cgs[0])
    nz = rho_cgs.shape[0]
    out = np.empty((nz, NKR_AEROSOL))
    for k in range(nz):
        rk = float(rho_cgs[k])
        for kr in range(NKR_AEROSOL):
            out[k, kr] = (float(source[kr]) / rhoair_max) * rk
    return out


__all__ = [
    "CLOUD_BASE_RULES",
    "CloudBaseSuperResult",
    "NucleationResult",
    "NucleationStep",
    "T_NUCL_DROP_MIN",
    "T_NUCL_ICE_MIN",
    "USE_CLOUD_BASE_NUC",
    "WaterNucleationResult",
    "cloud_base_level",
    "cloud_base_rcriti",
    "cloud_base_rule",
    "cloud_base_super",
    "initial_aerosol_column",
    "is_this_cloud_base",
    "jernucl01_warm",
    "koehler_rcriti_in_cloud",
    "liquid_mass_mixing_ratio",
    "supmax_coeff",
    "water_nucleation",
]
