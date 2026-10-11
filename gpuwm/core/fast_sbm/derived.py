"""Derived tables of the fast SBM and the collision kernels at a pressure.

Float64 host reference.  Transcribed from WRF v4.8.0
``phys/module_mp_fast_sbm.F`` (LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1).  WRF is NCAR/UCAR public domain; the notice is in
``NOTICE`` and ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.  The scheme is FSBM-2
by Jacob Shpund and co-authors, from Alexander Khain's Hebrew University
Cloud Model (see the package docstring for the full credit and the citation
the WRF header requests).

Routines and lines transcribed
------------------------------
``build_derived`` is the derived part of ``FAST_HUCMINIT`` (:5384-6236), the
statements after the table READs, in WRF's order:

* :6074-6085, the module mass grids in mg, ``xl_mg(i) = xl(i)*1.e3`` and its
  snow, graupel, hail and crystal siblings (``mass_grids_mg``).
* :6087-6091, ``CHUCM`` and ``IMA`` zeroed, then ``courant_bott_KS``
  (:1636-1689), which builds its own mass grid ``xl_mg(1:nkr) =
  xl(1:nkr)*1.0D3`` (:1656) and from it the Bott category ``ima`` and the
  Courant number ``chucm`` of every drop pair (``courant_bott``).
* :6095-6101, ``DROPRADII = (3.0*XL/4.0/3.141593/1.0)**DEG01``
  (``gpuwm.core.fast_sbm.state.drop_radii``, reused here and gated here).
* :6106-6124, the aerosol arrays zeroed and ``Scale_CCN_Factor = 1.0``, then
  ``LogNormal_modes_Aerosol`` (:3735-3838) twice, maritime (``IType = 1``)
  and continental (``IType = 2``) (``lognormal_modes_aerosol``).
* :6127-6135, ``ECOALMASSM`` and ``BRKWEIGHT`` zeroed, then ``BREAKINIT_KS``
  (:8644-8780): after reading ``PKIJ`` and ``QKJ`` (which the table loader
  owns), it fills ``ECOALMASSM(I,J) = ECOALMASS(xl(I), xl(J))`` (:8761-8767)
  through ``ecoalmass`` (:8784-8804), ``ecoaldiam`` (:8807-8860),
  ``ecoalLowList`` (:8863-8908), ``ecoalOchs`` (:8912-8951), ``COLLENERGY``
  (:8955-9021) and ``vTBeard`` (:9026-9048), then sets
  ``ECOALMASSM(NKR, J) = 0.1D-29`` for ``J = 25..31`` (:8768-8772)
  (``coalescence_efficiency_matrix``).
* :6156-6215, every kernel array zeroed, then ``Kernals_KS(dt, nkr,
  7.6E6)``.  See "The initialisation kernels" below.
* :6218-6224, ``Prob``, ``Gain_Var_New`` and ``NND`` zeroed, then
  ``Spontanous_Init`` (:120-189) with ``I_Break_Method = 1`` (:106-107)
  (``spontaneous_init``).

``kernels_at_pressure`` is ``Kernals_KS`` (:6238-6541) with ``ckern_z``
(:6544-6558), as ``COAL_BOTT_NEW`` calls it at the start of every collision
step (:8317-8318).

The initialisation kernels
--------------------------
``FAST_HUCMINIT`` calls ``Kernals_KS(dt, nkr, 7.6E6)`` (:6215), which fills
every kernel array at 7.6e6 dyn/cm2 (7600 hPa, above the 1000 hPa slice, so
the 1000 hPa and 750 hPa slices are used unchanged).  Those values are never
read: the only statements that read the kernel arrays are inside
``COAL_BOTT_NEW``, and every ``COAL_BOTT_NEW`` call first calls
``Kernals_KS(dt_coll, nkr, PP_r)`` at the point's own pressure (:8317-8318),
which overwrites every family the init call set.  The init call is therefore
superseded at every collision call, and ``FsbmDerived`` does not carry
kernels.  ``kernels_at_pressure(tables, derived, 7.6e6, dt)`` reproduces the
init values if anyone needs them.

What WRF leaves at zero, reproduced
-----------------------------------
* ``ima(i, j)`` and ``chucm(i, j)`` stay at the zero :6089-6090 sets for
  every pair whose summed mass exceeds the largest bin, because
  ``courant_bott_KS`` finds no bracketing category and assigns nothing: every
  pair that includes bin 33 (both orders).  ``ima`` holds WRF's one-based bin
  number, so 0 there is not a bin; ``coll_xxx_lwf`` limits both of its
  indices to ``nkr - 1`` (:680-689), so it never reaches those pairs.
* ``BRKWEIGHT`` is zeroed at :6134, passed to ``BREAKINIT_KS`` and never
  assigned: it stays zero.
* ``Gain_Var_New`` and ``NND`` are set only for rows ``j >= ikr_spon_break``
  (:166-172, :182-186).  On the 33-bin grid ``ikr_spon_break`` is 33, so only
  row 33, columns 1-32, are nonzero.
* ``cwig_1..3`` and ``cwih_1..3`` are allocated and zeroed (:6190-6196,
  :6211-6212) and never assigned or read; ``FsbmKernels`` carries them as the
  zeros WRF holds.
* The ``ECOALMASSM`` correction at :8770-8772 sets row 33, columns 25-31, and
  not the transposed entries, so ``ECOALMASSM`` and ``cwll`` are not
  symmetric there.

Finding D10, the fall speed ``vTBeard`` hands a bin-centre drop
-----------------------------------------------------------------
``ecoalmass`` turns a bin mass into a diameter with pi written ``3.1415927``
(:8795), while ``DROPRADII``, the radii ``vTBeard`` compares against, uses
``3.141593`` (:6100).  The drop's radius is therefore 3.18e-8 relative above
its own ``DROPRADII``, and ``vTBeard``'s search (``aa > DROPRADII(KR) .and.
aa <= DROPRADII(KR+1)``, :9041-9045) returns ``VR1(KR+1)``, the fall speed
of the next bin up, for every bin from 1 to 32; bin 33 gets ``VR1(33)``.  So
every Beard-Ochs Weber number and every Low-List collision energy in
``ECOALMASSM`` is formed from the speeds of the two bins above the pair.
The margin is far above float64 rounding, so the promoted build and this
port agree exactly.  WRF's native build takes the next bin's speed for every
bin as well: measured on the O32 dump, each drop's radius (formed in double
from the float32 ``XL``, as ``BREAKINIT_KS`` does) lies 2.6e-8 to 3.3e-7
relative above its float32 ``DROPRADII``, mostly because ``DEG01 = 1./3.``
rounds up to 0.33333334 in float32 and so lowers most radii.  Measured on the float64 tables, giving each drop its own
bin's speed changes 295 of the 1089 entries, by up to 0.32 (pair 30, 33:
0.3228 with WRF's speeds, the 1e-30 floor with its own, because the larger
speed difference puts the Low-List collision energy above 5 uJ).

The ``"own_bin"`` lookup gives each bin-centre drop its own bin's speed,
by forming the radii ``vTBeard`` searches with the same pi and operations as
the diameter, so a drop lands exactly on its own bin's upper bound.
``"wrf_v480"`` reproduces WRF, and is what the oracle gates run.  The
lookup is ``FsbmBehavior.fall_speed_lookup``: the ``corrected()`` preset,
the reference default, takes ``"own_bin"`` and the ``wrf_v480()`` preset
takes ``"wrf_v480"``.  ``build_derived`` takes it from the behaviour it is
given, and with none from ``corrected()``, so the bare call builds the
corrected tables; ``FsbmDerived.fall_speed_lookup`` records which one a
table holds, and the warm call refuses a table whose lookup is not its
behaviour's.  On KC-1 the corrected lookup adds 0.20 and 0.10 percent to
the hour's rain in the two updraft columns.

Precision and order
-------------------
Everything is float64, as WRF's source promoted with ``-freal-4-real-8``
(the O64 oracle build) computes it.  Tables of either precision are accepted
and widened exactly; a float32 view gives the float64 computation on the
values WRF's native build reads, never an emulation of its float32
arithmetic.  Every expression keeps Fortran's evaluation order (left to right
within a precedence level, ``a*b/c`` as ``(a*b)/c``); integer powers are
expanded as gfortran expands them (``x**2 = x*x``, ``x**3 = (x*x)*x``,
``x**4 = (x*x)*(x*x)``); ``x**2.0`` with a real exponent is written ``x*x``,
the value gfortran's folding of ``pow(x, 2.0)`` gives; the one reduction
(``q_m`` in ``Spontanous_Init``) is a sequential loop.  ``2**(3.0d0/2.0d0)``
is a compile-time constant in gfortran (MPFR, correctly rounded), written
here as ``math.sqrt(8.0)``, the same correctly rounded value.  Transcendental
functions come from the platform's libm through ``math`` and numpy, so a
result can differ from the oracle's glibc by a unit in the last place; the
oracle gate is relative 1e-12 for that reason.

Arrays are zero-based and keep Fortran index order: ``chucm[i-1, j-1]`` is
``CHUCM(I,J)``.  Every array ``build_derived`` and ``kernels_at_pressure``
return is read-only.

This module is the verification authority for the derived tables, not a
production data path: milestone 2 of the fast-sbm design moves their
generation into CUDA init kernels gated against these values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gpuwm.core.fast_sbm.constants import (
    COL,
    FALL_SPEED_LOOKUPS,
    FsbmBehavior,
    FsbmRefusal,
    ICEMAX,
    JBREAK,
    NKR,
    NKR_AEROSOL,
    RO_SOLUTE,
)
from gpuwm.core.fast_sbm.state import drop_radii

__all__ = [
    "FALL_SPEED_LOOKUPS",
    "FsbmDerived",
    "FsbmKernels",
    "KERNEL_FAMILIES",
    "build_derived",
    "coalescence_efficiency",
    "coalescence_efficiency_matrix",
    "courant_bott",
    "drop_drop_kernel",
    "kernels_at_pressure",
    "lognormal_modes_aerosol",
    "mass_grids_mg",
    "spontaneous_init",
]

# ---------------------------------------------------------------------------
# Literals, exactly as WRF writes them
# ---------------------------------------------------------------------------

#: ``SCAL = 1.d0`` (:3890), passed to ``courant_bott_KS``.
_SCAL = 1.0
#: ``1.0d0-1.d-08``: a Courant number above this is folded into the next
#: category (:1669).
_CHUCM_FOLD = 1.0 - 1.0e-08

#: ``Spontanous_Init`` (:120-189).  ``gamma = 0.453d0`` (:133); ``Prob`` for
#: ``I_Break_Method = 1`` (:106-107, :151-154); the fragment weights (:168,
#: :170); the radius at which spontaneous breakup starts (:145).
_SPON_GAMMA = 0.453
_SPON_PROB_A = 2.94e-7
_SPON_PROB_B = 34.0
_SPON_GAIN_A = 145.37
_SPON_GAIN_B = 7.0
_SPON_RADIUS_CM = 0.3

#: ``LogNormal_modes_Aerosol`` (:3735-3838).
_AERO_PI = 3.14159265          # PARAMETER PI (:3760)
_AERO_PI_R0 = 3.141593         # the literal in R0 (:3772)
_RCCN_MAX = 0.4e-4             # cm (:3756)
_RCCN_MIN_3LN = 0.00048e-4     # cm (:3759)
#: (concentration cm-3, mean radius cm, geometric standard deviation) per
#: mode, IType 1 maritime (:3777-3789) and IType 2 continental (:3791-3803).
_AEROSOL_MODES = {
    1: ((340.000, 0.00500e-04, 1.60000),
        (60.0000, 0.03500e-04, 2.00000),
        (3.10000, 0.31000e-04, 2.70000)),
    2: ((1000.000, 0.00800e-04, 1.60000),
        (800.0000, 0.03400e-04, 2.10000),
        (0.72000, 0.46000e-04, 2.20000)),
}
#: ``Scale_CCN_Factor = 1.0`` (:6117).
_SCALE_CCN_FACTOR = 1.0

#: ``BREAKINIT_KS`` correction (:8770-8772): ``ECOALMASSM(NKR, J) = 0.1D-29``
#: for ``J = 25..31``.
_ECOAL_CORRECTION = 0.1e-29
_ECOAL_CORRECTION_COLUMNS = range(25, 32)

#: ``ecoalmass``, ``ecoaldiam``, ``ecoalOchs``, ``COLLENERGY`` (:8784-9021).
_ECOAL_PI = 3.1415927
_ECOAL_RHO = 1.0               # g/cm3 (:8793, :8969)
_ECOAL_EPS = 1.0e-30           # ecoaldiam floor (:8814)
_ECOAL_SMALL_DIAMETER = 100.0e-4   # cm, E = 1 below it (:8827)
_ECOAL_QMIN = 250.0e-4         # cm (:8824)
_ECOAL_QMAX = 500.0e-4         # cm (:8825)
_ECOAL_FALLBACK = 0.999        # the unreachable else branch (:8853)
_SIGMA = 72.8                  # g/s2 (:8876, :8923, :8970)
_LL_AKA = 0.778                # (:8877)
_LL_AKB = 2.61e-4              # (:8878)
_LL_EPSI = 1.0e-20             # (:8872)
_LL_ET_MAX = 50.0              # g cm2/s2, 5 uJ (:8885)
_FPMIN = 1.0e-30               # (:8920, :8964)
_OCHS_A = 0.767                # (:8946)
_OCHS_B = 10.14                # (:8946)
#: ``2**(3.0d0/2.0d0)`` (:8943), folded at compile time to the correctly
#: rounded 2**1.5, which is the correctly rounded sqrt(8).
_TWO_POW_THREE_HALVES = math.sqrt(8.0)

#: ``Kernals_KS`` pressures, dyn/cm2 (:6247).
_P1 = 1.0e6
_P2 = 0.75e6
_P3 = 0.50e6
_P4 = 0.3e6

#: Every kernel array ``Kernals_KS`` leaves, in the order WRF declares them
#: (:3967-3979).
KERNEL_FAMILIES = (
    "cwll",
    "cwli_1", "cwli_2", "cwli_3",
    "cwil_1", "cwil_2", "cwil_3",
    "cwlg", "cwlh", "cwls",
    "cwgl", "cwhl", "cwsl",
    "cwii_1_1", "cwii_1_2", "cwii_1_3",
    "cwii_2_1", "cwii_2_2", "cwii_2_3",
    "cwii_3_1", "cwii_3_2", "cwii_3_3",
    "cwis_1", "cwis_2", "cwis_3",
    "cwsi_1", "cwsi_2", "cwsi_3",
    "cwig_1", "cwig_2", "cwig_3",
    "cwih_1", "cwih_2", "cwih_3",
    "cwsg", "cwss",
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FsbmDerived:
    """What ``FAST_HUCMINIT`` derives from the tables, float64, read-only.

    ``xl_mg`` is the module grid ``xl(i)*1.e3`` (:6078) that
    ``COAL_BOTT_NEW`` hands to the collision routines (:8375); ``xl_mg_bott``
    is the grid ``xl(i)*1.0D3`` that ``courant_bott_KS`` builds for itself
    (:1656).  In float64 they are the same product; in WRF's native build
    the first is a float32 product.  ``xs_mg``, ``xg_mg``, ``xh_mg`` and
    ``xi_mg`` (33, 3) are the ice grids of the same loop (:6079-6084), for
    the ice milestone.

    ``ima`` is int32 and holds WRF's one-based category; ``ikr_spon_break``
    is WRF's one-based bin number.  Masses in mg, radii in cm, ``prob`` per
    second, ``fccnr_*`` in cm-3 per unit ln radius, ``xccn`` in g.
    ``fall_speed_lookup`` is the D10 lookup ``ecoalmassm`` was built with
    (``FsbmBehavior.fall_speed_lookup``); it is the only field the lookup
    changes.
    """

    table_precision: str
    fall_speed_lookup: str
    dt: float
    xl_mg: np.ndarray            # (33,)
    xl_mg_bott: np.ndarray       # (33,)
    xs_mg: np.ndarray            # (33,)
    xg_mg: np.ndarray            # (33,)
    xh_mg: np.ndarray            # (33,)
    xi_mg: np.ndarray            # (33, 3)
    ima: np.ndarray              # (33, 33) int32
    chucm: np.ndarray            # (33, 33)
    dropradii: np.ndarray        # (33,)
    fccnr_con: np.ndarray        # (43,)
    fccnr_mar: np.ndarray        # (43,)
    xccn: np.ndarray             # (43,)
    rccn: np.ndarray             # (43,)
    scale_ccn_factor: float
    ecoalmassm: np.ndarray       # (33, 33)
    brkweight: np.ndarray        # (18,)
    prob: np.ndarray             # (33,)
    gain_var_new: np.ndarray     # (33, 33)
    nnd: np.ndarray              # (33, 33)
    ikr_spon_break: int


@dataclass(frozen=True, eq=False)
class FsbmKernels:
    """Every kernel array after ``Kernals_KS(dt_coll, nkr, p_cgs)``.

    Each family is (33, 33), float64, read-only, in Fortran index order and
    named as WRF names it (``KERNEL_FAMILIES``).  ``cwll`` is the drop-drop
    kernel times ``ECOALMASSM``; the ice-related families are the tabulated
    kernels times ``dt_coll ln2/3``; ``cwil_k``, ``cwgl``, ``cwhl``,
    ``cwsl`` and ``cwsi_k`` are the transposes of ``cwli_k``, ``cwlg``,
    ``cwlh``, ``cwls`` and ``cwis_k`` (:6515-6537); ``cwig_k`` and
    ``cwih_k`` are the zeros WRF never assigns.
    """

    p_cgs: float
    dt_coll: float
    cwll: np.ndarray
    cwli_1: np.ndarray
    cwli_2: np.ndarray
    cwli_3: np.ndarray
    cwil_1: np.ndarray
    cwil_2: np.ndarray
    cwil_3: np.ndarray
    cwlg: np.ndarray
    cwlh: np.ndarray
    cwls: np.ndarray
    cwgl: np.ndarray
    cwhl: np.ndarray
    cwsl: np.ndarray
    cwii_1_1: np.ndarray
    cwii_1_2: np.ndarray
    cwii_1_3: np.ndarray
    cwii_2_1: np.ndarray
    cwii_2_2: np.ndarray
    cwii_2_3: np.ndarray
    cwii_3_1: np.ndarray
    cwii_3_2: np.ndarray
    cwii_3_3: np.ndarray
    cwis_1: np.ndarray
    cwis_2: np.ndarray
    cwis_3: np.ndarray
    cwsi_1: np.ndarray
    cwsi_2: np.ndarray
    cwsi_3: np.ndarray
    cwig_1: np.ndarray
    cwig_2: np.ndarray
    cwig_3: np.ndarray
    cwih_1: np.ndarray
    cwih_2: np.ndarray
    cwih_3: np.ndarray
    cwsg: np.ndarray
    cwss: np.ndarray


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _f64(tables, name: str) -> np.ndarray:
    """A table widened exactly to float64 (a copy, never the shared view)."""
    return np.array(getattr(tables, name), dtype=np.float64)


def _positive_finite(value, name: str, where: str, why: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise FsbmRefusal(where, f"{name} {value!r} is not a number") from None
    if not math.isfinite(number) or number <= 0.0:
        raise FsbmRefusal(where, f"{name} is {number!r}; it must be positive "
                          f"and finite, because {why}")
    return number


# ---------------------------------------------------------------------------
# Mass grids and the Bott categories
# ---------------------------------------------------------------------------

def mass_grids_mg(tables) -> dict[str, np.ndarray]:
    """The module mass grids in mg, ``x(i)*1.e3`` (:6077-6085).

    Keys ``xl_mg``, ``xs_mg``, ``xg_mg``, ``xh_mg`` (33,) and ``xi_mg``
    (33, 3), the three crystal habits of ``xi1_mg``, ``xi2_mg``, ``xi3_mg``.
    """
    return {
        "xl_mg": _f64(tables, "xl") * 1.0e3,
        "xs_mg": _f64(tables, "xs") * 1.0e3,
        "xg_mg": _f64(tables, "xg") * 1.0e3,
        "xh_mg": _f64(tables, "xh") * 1.0e3,
        "xi_mg": _f64(tables, "xi") * 1.0e3,
    }


def courant_bott(xl) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``courant_bott_KS`` (:1636-1689): ``(chucm, ima, xl_mg)``.

    For each pair ``i <= j`` the summed mass ``x0 = xl_mg(i) + xl_mg(j)`` is
    placed between the bins that bracket it, ``xl_mg(k-1) < x0 <=
    xl_mg(k)`` searched from ``k = j`` upward, and the Courant number is
    ``chucm = ln(x0 / xl_mg(k-1)) / (3 dlnr)`` with ``dlnr = ln 2 / (3
    SCAL)``.  A Courant number above ``1 - 1e-8`` is folded to 0 in the next
    category.  ``ima = min(nkr - 1, kk - 1)`` is the one-based lower bin of
    the bracket.  Both are mirrored to ``(j, i)``.  A pair with no bracket
    (summed mass above the largest bin) keeps the zero FAST_HUCMINIT set
    (:6089-6090).  ``xl_mg`` is the routine's own grid ``xl*1.0D3``
    (:1656).  The search is the scalar loop WRF runs, so every comparison
    is WRF's.
    """
    xl_mg = np.asarray(xl, dtype=np.float64) * 1.0e3
    nkr = xl_mg.shape[0]
    dlnr = math.log(2.0) / (3.0 * _SCAL)
    chucm = np.zeros((nkr, nkr), dtype=np.float64)
    ima = np.zeros((nkr, nkr), dtype=np.int32)
    grid = [float(v) for v in xl_mg]
    for i in range(1, nkr + 1):
        for j in range(i, nkr + 1):
            x0 = grid[i - 1] + grid[j - 1]
            for k in range(j, nkr + 1):
                kk = k
                if k == 1:
                    continue
                if grid[k - 1] >= x0 and grid[k - 2] < x0:
                    value = math.log(x0 / grid[k - 2]) / (3.0 * dlnr)
                    if value > _CHUCM_FOLD:
                        value = 0.0
                        kk = kk + 1
                    chucm[i - 1, j - 1] = value
                    ima[i - 1, j - 1] = min(nkr - 1, kk - 1)
                    break
            chucm[j - 1, i - 1] = chucm[i - 1, j - 1]
            ima[j - 1, i - 1] = ima[i - 1, j - 1]
    return chucm, ima, xl_mg


# ---------------------------------------------------------------------------
# Aerosol spectra (LogNormal_modes_Aerosol, :3735-3838)
# ---------------------------------------------------------------------------

def lognormal_modes_aerosol(xl, itype: int, *, col: float = COL,
                            ro_solute: float = RO_SOLUTE,
                            scale_fa: float = _SCALE_CCN_FACTOR,
                            nkr_local: int = NKR_AEROSOL
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``LogNormal_modes_Aerosol`` (:3735-3838): ``(fccnr, xccn, rccn)``.

    The dry-aerosol grid starts at ``XL(2) / 2**NKR_local`` and doubles in
    mass 43 times at the solute density; ``RCCN = (3 X0 / 4 / 3.141593 /
    RO)**(1/3)`` (:3766-3775).  Three lognormal modes of ``dN/dln r``
    (:3777-3805, Shpund et al. 2019b) are summed in mode order inside the window
    ``0.48 nm < RCCN < 0.4 um``, each times ``ln(2)/3``, and divided by
    ``col`` (:3817-3830); bins outside the window stay zero.  ``itype`` 1 is
    maritime (WRF stores it in ``FCCNR_MAR``), 2 continental
    (``FCCNR_CON``).  The printed ``CONCCCNIN`` (:3832-3833) is not a
    result and is not formed.
    """
    if itype not in _AEROSOL_MODES:
        raise FsbmRefusal("lognormal_modes_aerosol",
                          f"IType {itype!r} is neither 1 (maritime) nor 2 "
                          "(continental); WRF assigns no mode for any other "
                          "value and would read undefined mode parameters")
    xl = np.asarray(xl, dtype=np.float64)
    deg01 = 1.0 / 3.0
    x0drop = float(xl[1])
    x0ccn = x0drop / (2.0 ** nkr_local)
    xccn = np.zeros(nkr_local, dtype=np.float64)
    rccn = np.zeros(nkr_local, dtype=np.float64)
    for kr in range(nkr_local, 0, -1):
        roccn = ro_solute
        x0 = x0ccn * 2.0 ** kr
        r0 = (3.0 * x0 / 4.0 / _AERO_PI_R0 / roccn) ** deg01
        xccn[kr - 1] = x0
        rccn[kr - 1] = r0

    (c1, rm1, s1), (c2, rm2, s2), (c3, rm3, s3) = _AEROSOL_MODES[itype]
    root_two_pi = math.sqrt(2.0 * _AERO_PI)
    arg11 = c1 / (root_two_pi * math.log(s1))
    arg21 = c2 / (root_two_pi * math.log(s2))
    arg31 = c3 / (root_two_pi * math.log(s3))
    ln2_3 = math.log(2.0) / 3.0
    fccnr_tmp = np.zeros(nkr_local, dtype=np.float64)
    for kr in range(nkr_local, 0, -1):
        r = float(rccn[kr - 1])
        if r > _RCCN_MIN_3LN and r < _RCCN_MAX:
            t = math.log(r / rm1)
            arg12 = t * t
            ls = math.log(s1)
            arg13 = 2.0 * (ls * ls)
            norm1 = arg11 * math.exp(-arg12 / arg13) * ln2_3
            t = math.log(r / rm2)
            arg22 = t * t
            ls = math.log(s2)
            arg23 = 2.0 * (ls * ls)
            norm2 = norm1 + arg21 * math.exp(-arg22 / arg23) * ln2_3
            t = math.log(r / rm3)
            arg32 = t * t
            ls = math.log(s3)
            arg33 = 2.0 * (ls * ls)
            norm3 = norm2 + arg31 * math.exp(-arg32 / arg33) * ln2_3
            fccnr_tmp[kr - 1] = norm3 / col
    return scale_fa * fccnr_tmp, xccn, rccn


# ---------------------------------------------------------------------------
# Coalescence efficiency (BREAKINIT_KS and its functions, :8644-9048)
# ---------------------------------------------------------------------------

def _vt_beard(diam: float, dropradii, vr1) -> float:
    """``vTBeard`` (:9026-9048): the table speed of the bin whose radius
    interval ``(DROPRADII(KR), DROPRADII(KR+1)]`` holds ``diam/2``."""
    aa = diam / 2.0
    speed = None
    if aa <= dropradii[0]:
        speed = vr1[0]
    if aa > dropradii[-1]:
        speed = vr1[-1]
    for kr in range(len(dropradii) - 1):
        if aa > dropradii[kr] and aa <= dropradii[kr + 1]:
            speed = vr1[kr + 1]
    if speed is None:   # only a NaN diameter reaches this
        raise FsbmRefusal("vTBeard", f"diameter {diam!r} matches no bin; "
                          "WRF would return an undefined fall speed")
    return speed


def _collenergy(dgr: float, dkl: float, dropradii, vr1, speed):
    """``COLLENERGY`` (:8955-9021): ``(CKE, ST, Sc)``, g cm2/s2."""
    ak10 = _ECOAL_RHO * _ECOAL_PI / 12.0
    dgr = max(dgr, _FPMIN)
    dkl = max(dkl, _FPMIN)
    dgka2 = (dgr * dgr) + (dkl * dkl)
    dgka3 = ((dgr * dgr) * dgr) + ((dkl * dkl) * dkl)
    if dgr != dkl:
        v1 = speed(dgr)
        v2 = speed(dkl)
        dv = v1 - v2
        if dv < _FPMIN:
            dv = _FPMIN
        dv = dv * dv
        if dv < _FPMIN:
            dv = _FPMIN
        dgkb3 = ((dgr * dgr) * dgr) * ((dkl * dkl) * dkl)
        cke = ak10 * dv * dgkb3 / dgka3
    else:
        cke = 0.0
    st = _ECOAL_PI * _SIGMA * dgka2
    sc = _ECOAL_PI * _SIGMA * dgka3 ** (2.0 / 3.0)
    return cke, st, sc


def _ecoal_low_list(dgr: float, dkl: float, dropradii, vr1, speed) -> float:
    """``ecoalLowList`` (:8863-8908), Low and List (1982a)."""
    cke, st, sc = _collenergy(dgr, dkl, dropradii, vr1, speed)
    dstsc = st - sc
    et = cke + dstsc
    if et < _LL_ET_MAX:
        qq0 = 1.0 + (dkl / dgr)
        qq1 = _LL_AKA / (qq0 * qq0)
        qq2 = _LL_AKB * _SIGMA * (et * et) / (sc + _LL_EPSI)
        return qq1 * math.exp(-qq2)
    return 0.0


def _ecoal_ochs(d_l: float, d_s: float, dropradii, vr1, speed) -> float:
    """``ecoalOchs`` (:8912-8951), Beard and Ochs (1995)."""
    r_s = 0.5 * d_s
    r_l = 0.5 * d_l
    p = r_s / r_l
    vtl = speed(d_l)
    vts = speed(d_s)
    dv = abs(vtl - vts)
    if dv < _FPMIN:
        dv = _FPMIN
    weber = r_s * (dv * dv) / _SIGMA
    pa1 = 1.0 + p
    pa2 = 1.0 + (p * p)
    pa3 = 1.0 + ((p * p) * p)
    p2 = p * p
    g = _TWO_POW_THREE_HALVES / (6.0 * _ECOAL_PI) * (p2 * p2) * pa1 / (pa2 * pa3)
    x = weber ** 0.5 * g
    return _OCHS_A - _OCHS_B * x


def _ecoal_diam(deta: float, dksi: float, dropradii, vr1, speed) -> float:
    """``ecoaldiam`` (:8807-8860): E from the two diameters, cm."""
    dgr = max(deta, dksi)
    dkl = min(deta, dksi)
    rgr = 0.5 * dgr
    rkl = 0.5 * dkl
    q = 0.5 * (rkl + rgr)
    if dkl < _ECOAL_SMALL_DIAMETER:
        e = 1.0
    elif q < _ECOAL_QMIN:
        e = _ecoal_ochs(dgr, dkl, dropradii, vr1, speed)
    elif q >= _ECOAL_QMIN and q < _ECOAL_QMAX:
        x = (q - _ECOAL_QMIN) / (_ECOAL_QMAX - _ECOAL_QMIN)
        sin1 = math.sin(_ECOAL_PI / 2.0 * x)
        cos1 = math.cos(_ECOAL_PI / 2.0 * x)
        e1 = _ecoal_ochs(dgr, dkl, dropradii, vr1, speed)
        e2 = _ecoal_low_list(dgr, dkl, dropradii, vr1, speed)
        e = (cos1 * cos1) * e1 + (sin1 * sin1) * e2
    elif q >= _ECOAL_QMAX:
        e = _ecoal_low_list(dgr, dkl, dropradii, vr1, speed)
    else:   # only a NaN mean radius reaches this
        e = _ECOAL_FALLBACK
    return max(min(1.0, e), _ECOAL_EPS)


def _speed_function(xl, dropradii, vr1, fall_speed_lookup: str):
    if fall_speed_lookup not in FALL_SPEED_LOOKUPS:
        raise FsbmRefusal("coalescence_efficiency",
                          f"fall_speed_lookup {fall_speed_lookup!r} is not one "
                          f"of {FALL_SPEED_LOOKUPS}")
    speeds = [float(v) for v in vr1]
    if fall_speed_lookup == "wrf_v480":
        radii = [float(v) for v in dropradii]
    else:
        # "own_bin": the radii vTBeard searches are formed exactly as
        # ecoalmass forms a drop's diameter (same pi, same operations), so a
        # bin-centre drop lands on its own bin's upper bound, which the
        # search's "<=" assigns to that bin.
        akpi = 6.0 / _ECOAL_PI
        radii = [((akpi * float(x) / _ECOAL_RHO) ** (1.0 / 3.0)) / 2.0
                 for x in xl]
    return lambda diam: _vt_beard(diam, radii, speeds)


def coalescence_efficiency(x1: float, x2: float, xl, dropradii, vr1, *,
                           fall_speed_lookup: str) -> float:
    """``ecoalmass`` (:8784-8804): E for drops of masses ``x1``, ``x2`` (g).

    Diameters ``(6/pi x / rho)**(1/3)`` with ``pi = 3.1415927``, then
    ``ecoaldiam``: E = 1 when the small drop is below 100 um diameter;
    Beard and Ochs (1995) when the mean radius is below 250 um; Low and List
    (1982a) above 500 um; between them a blend ``cos2 E_Ochs + sin2 E_LL``
    in the scaled mean radius (Seifert et al. 2005); the result is clamped
    to ``[1e-30, 1]``.  Fall speeds come from ``vTBeard`` over ``dropradii``
    and ``vr1`` (``fall_speed_lookup``: see finding D10 in the module
    docstring; ``xl``, the drop mass grid, is what the ``"own_bin"`` lookup
    forms its radii from).  The lookup has no default here: a routine-level
    caller names WRF's or the corrected speeds, and ``build_derived`` passes
    the behaviour's.
    """
    xl = np.asarray(xl, dtype=np.float64)
    speed = _speed_function(xl, dropradii, vr1, fall_speed_lookup)
    return _ecoalmass(float(x1), float(x2), dropradii, vr1, speed)


def _ecoalmass(x1: float, x2: float, dropradii, vr1, speed) -> float:
    akpi = 6.0 / _ECOAL_PI
    deta = (akpi * x1 / _ECOAL_RHO) ** (1.0 / 3.0)
    dksi = (akpi * x2 / _ECOAL_RHO) ** (1.0 / 3.0)
    return _ecoal_diam(deta, dksi, dropradii, vr1, speed)


def coalescence_efficiency_matrix(xl, dropradii, vr1, *,
                                  fall_speed_lookup: str) -> np.ndarray:
    """``ECOALMASSM`` as ``BREAKINIT_KS`` fills it (:8761-8772).

    ``ECOALMASSM(I,J) = ECOALMASS(xl(I), xl(J))`` in WRF's loop order (J
    outer, I inner), then ``ECOALMASSM(NKR, J) = 0.1D-29`` for ``J = 25..31``,
    which WRF labels a correction of the drop collision kernels.
    ``fall_speed_lookup`` (D10) has no default, as in
    ``coalescence_efficiency``.
    """
    xl = [float(v) for v in np.asarray(xl, dtype=np.float64)]
    nkr = len(xl)
    speed = _speed_function(xl, dropradii, vr1, fall_speed_lookup)
    ecoal = np.zeros((nkr, nkr), dtype=np.float64)
    for j in range(nkr):
        for i in range(nkr):
            ecoal[i, j] = _ecoalmass(xl[i], xl[j], dropradii, vr1, speed)
    for j in _ECOAL_CORRECTION_COLUMNS:
        ecoal[nkr - 1, j - 1] = _ECOAL_CORRECTION
    return ecoal


# ---------------------------------------------------------------------------
# Spontaneous rain breakup (Spontanous_Init, :120-189)
# ---------------------------------------------------------------------------

def spontaneous_init(xl, dropradii
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``Spontanous_Init`` (:120-189): ``(prob, gain_var_new, nnd,
    ikr_spon_break)``.

    ``ikr_spon_break`` is the first bin (one-based) whose radius is at least
    0.3 cm (:143-146).  ``Prob = 2.94e-7 exp(34 r)`` per second for every
    bin (Srivastava 1971, ``I_Break_Method = 1``, :151-154).  For each
    breaking bin ``j >= ikr_spon_break`` and fragment bin ``i < j`` the
    fragment weight is ``145.37 / x_i (r_i / r_j) exp(-7 r_i / r_j)``
    (:168) and ``NND = gamma exp(-gamma d_i) / (1 - exp(-gamma d_j))`` with
    ``d = 20 r`` and ``gamma = 0.453`` (:170).  The weights are divided by
    ``ratio_new = sum_i gain_var(ikr, i) x_i**2 / x_ikr`` (sequential sum,
    :174-179), so the fragments of one drop of bin ``ikr_spon_break`` carry
    its mass (:182-186).  Entries WRF never assigns stay at the zero
    :6221-6223 sets.
    """
    xl = [float(v) for v in np.asarray(xl, dtype=np.float64)]
    dr = [float(v) for v in np.asarray(dropradii, dtype=np.float64)]
    nkr = len(xl)
    diameter = [r * 2.0 * 10.0 for r in dr]
    ikr = 0
    for kr in range(1, nkr + 1):
        ikr = kr
        if dr[kr - 1] >= _SPON_RADIUS_CM:
            break
    prob = np.array([_SPON_PROB_A * math.exp(_SPON_PROB_B * dr[kr])
                     for kr in range(nkr)], dtype=np.float64)
    gain_var = np.zeros((nkr, nkr), dtype=np.float64)
    nnd = np.zeros((nkr, nkr), dtype=np.float64)
    for j in range(ikr, nkr + 1):
        for i in range(1, j):
            ri, rj = dr[i - 1], dr[j - 1]
            gain_var[j - 1, i - 1] = ((_SPON_GAIN_A / xl[i - 1]) * (ri / rj)
                                      * math.exp(-(_SPON_GAIN_B * ri / rj)))
            nnd[j - 1, i - 1] = (
                _SPON_GAMMA * math.exp(-(_SPON_GAMMA * diameter[i - 1]))
                / (1.0 - math.exp(-(_SPON_GAMMA * diameter[j - 1]))))
    q_m = 0.0
    for i in range(1, ikr):
        q_m = q_m + gain_var[ikr - 1, i - 1] * (xl[i - 1] * xl[i - 1])
    ratio_new = q_m / xl[ikr - 1]
    gain_var_new = np.zeros((nkr, nkr), dtype=np.float64)
    for j in range(ikr, nkr + 1):
        for i in range(1, j):
            gain_var_new[j - 1, i - 1] = gain_var[j - 1, i - 1] / ratio_new
    return prob, gain_var_new, nnd, ikr


# ---------------------------------------------------------------------------
# build_derived: the derived part of FAST_HUCMINIT
# ---------------------------------------------------------------------------

def build_derived(tables, dt, *, behavior: FsbmBehavior | None = None,
                  fall_speed_lookup: str | None = None) -> FsbmDerived:
    """Everything ``FAST_HUCMINIT(DT)`` derives from the tables (:6074-6224).

    ``tables`` is an ``FsbmTables`` view of either precision (widened
    exactly; the computation is float64).  ``dt`` (s) is recorded: in
    FAST_HUCMINIT it reaches only the superseded init kernels (:6215) and
    ``Spontanous_Init``, which does not use it (:137).

    ``ECOALMASSM`` is built with the fall-speed lookup of ``behavior``
    (``FsbmBehavior.fall_speed_lookup``, finding D10 in the module
    docstring); with no behaviour it is ``FsbmBehavior.corrected()``, the
    reference default, so the bare call gives each drop its own bin's
    speed.  The oracle gates pass ``behavior=FsbmBehavior.wrf_v480()``.
    ``fall_speed_lookup`` names the lookup directly, for a measurement; a
    lookup that disagrees with a behaviour also given is refused, because
    the table would then be labelled with one behaviour and hold the
    other's efficiencies.
    """
    if behavior is not None and not isinstance(behavior, FsbmBehavior):
        raise FsbmRefusal("build_derived",
                          f"behavior must be an FsbmBehavior, got "
                          f"{type(behavior).__name__}")
    if fall_speed_lookup is None:
        fall_speed_lookup = (FsbmBehavior.corrected() if behavior is None
                             else behavior).fall_speed_lookup
    if fall_speed_lookup not in FALL_SPEED_LOOKUPS:
        raise FsbmRefusal("build_derived",
                          f"fall_speed_lookup {fall_speed_lookup!r} is not one "
                          f"of {FALL_SPEED_LOOKUPS}")
    if behavior is not None and fall_speed_lookup != behavior.fall_speed_lookup:
        raise FsbmRefusal("build_derived",
                          f"fall_speed_lookup {fall_speed_lookup!r} disagrees "
                          f"with the behaviour's "
                          f"{behavior.fall_speed_lookup!r} (D10): the table "
                          f"would carry one behaviour's coalescence "
                          f"efficiencies under the other's label")
    dt = _positive_finite(dt, "dt", "build_derived",
                          "it is the model step the scheme is initialised "
                          "with, and WRF's own init call Kernals_KS(dt, nkr, "
                          "7.6E6) (:6215) would form zero, negative or NaN "
                          "kernels from such a step")
    grids = mass_grids_mg(tables)
    xl = _f64(tables, "xl")
    vr1 = _f64(tables, "vr1")
    chucm, ima, xl_mg_bott = courant_bott(xl)
    dropradii = np.asarray(drop_radii(tables), dtype=np.float64)
    fccnr_mar, xccn, rccn = lognormal_modes_aerosol(xl, 1)
    fccnr_con, xccn_con, rccn_con = lognormal_modes_aerosol(xl, 2)
    # The continental call rewrites XCCN and RCCN with the same values.
    if not (np.array_equal(xccn, xccn_con) and np.array_equal(rccn, rccn_con)):
        raise FsbmRefusal("build_derived", "the two LogNormal_modes_Aerosol "
                          "calls built different aerosol grids")
    ecoalmassm = coalescence_efficiency_matrix(
        xl, dropradii, vr1, fall_speed_lookup=fall_speed_lookup)
    brkweight = np.zeros(JBREAK, dtype=np.float64)
    prob, gain_var_new, nnd, ikr = spontaneous_init(xl, dropradii)
    arrays = dict(
        xl_mg=grids["xl_mg"], xl_mg_bott=xl_mg_bott, xs_mg=grids["xs_mg"],
        xg_mg=grids["xg_mg"], xh_mg=grids["xh_mg"], xi_mg=grids["xi_mg"],
        ima=ima, chucm=chucm, dropradii=dropradii, fccnr_con=fccnr_con,
        fccnr_mar=fccnr_mar, xccn=xccn, rccn=rccn, ecoalmassm=ecoalmassm,
        brkweight=brkweight, prob=prob, gain_var_new=gain_var_new, nnd=nnd)
    for name, array in arrays.items():
        arrays[name] = _frozen(np.array(array, copy=True))
    return FsbmDerived(
        table_precision=str(tables.precision),
        fall_speed_lookup=fall_speed_lookup, dt=dt,
        scale_ccn_factor=_SCALE_CCN_FACTOR, ikr_spon_break=int(ikr),
        **arrays)


# ---------------------------------------------------------------------------
# Kernals_KS and ckern_z (:6238-6558)
# ---------------------------------------------------------------------------

def _ckern_z(p_z: float, p_1: float, p_2: float, p_3: float,
             ckern_1: np.ndarray, ckern_2: np.ndarray, ckern_3: np.ndarray
             ) -> np.ndarray:
    """``ckern_z`` (:6544-6558), elementwise over a kernel slice."""
    result = None
    if p_z >= p_1:
        result = ckern_1
    if p_z <= p_3:
        result = ckern_3
    if p_z < p_1 and p_z >= p_2:
        result = ckern_2 + (ckern_1 - ckern_2) * (p_z - p_2) / (p_1 - p_2)
    if p_z < p_2 and p_z > p_3:
        result = ckern_3 + (ckern_2 - ckern_3) * (p_z - p_3) / (p_2 - p_3)
    return result


def _checked_kernel_inputs(p_cgs, dt_coll, where: str) -> tuple[float, float]:
    p = _positive_finite(p_cgs, "p_cgs", where,
                         "the kernels are interpolated in pressure and a "
                         "NaN pressure selects no slice, leaving WRF's "
                         "kernel undefined")
    dt = _positive_finite(dt_coll, "dt_coll", where,
                          "the kernels are the collision step times the "
                          "tabulated rate, so a zero, negative or NaN step "
                          "gives zero, negative or NaN kernels")
    return p, dt


def drop_drop_kernel(tables, derived: FsbmDerived, p_cgs, dt_coll
                     ) -> np.ndarray:
    """``cwll`` alone (:6256-6278), for callers on the warm path.

    ``cwll(i,j) = ECOALMASSM(i,j) (ckern_z(p) dt_coll dlnr)``, the drop-drop
    kernel ``YWLL`` interpolated linearly in pressure between 1000, 750 and
    500 hPa (held constant outside), times the collision step, times
    ``dlnr = ln 2 / 3``, times the coalescence efficiency.
    """
    p, dt = _checked_kernel_inputs(p_cgs, dt_coll, "drop_drop_kernel")
    return _frozen(_cwll(tables, derived, p, dt))


def _cwll(tables, derived: FsbmDerived, p: float, dt: float) -> np.ndarray:
    scal = 1.0
    dlnr = math.log(2.0) / (3.0 * scal)
    ywll = np.asarray(tables.ywll, dtype=np.float64)
    ckern = _ckern_z(p, _P1, _P2, _P3, ywll[0], ywll[1], ywll[2])
    cwll = ckern * dt * dlnr
    return np.asarray(derived.ecoalmassm, dtype=np.float64) * cwll


def kernels_at_pressure(tables, derived: FsbmDerived, p_cgs, dt_coll
                        ) -> FsbmKernels:
    """``Kernals_KS(dt_coll, nkr, p_cgs)`` (:6238-6541): every kernel family.

    ``p_cgs`` in dyn/cm2 (``PP_r`` at :8317), ``dt_coll`` in s (``DTCOLL =
    DT/NCOLL``, :4207).  The drop-drop kernel is interpolated over 1000, 750
    and 500 hPa by ``ckern_z`` and multiplied by ``dt_coll ln2/3`` and by
    ``ECOALMASSM`` (:6263-6278).  Every ice-related family is taken at 750
    hPa at or above 750 hPa, at 300 hPa at or below 300 hPa, and otherwise
    interpolated linearly between the two neighbouring slices of 750, 500
    and 300 hPa as ``(k_low + (k_high - k_low) pdm) dt_coll ln2/3``
    (:6280-6513), a different arithmetic form from ``ckern_z``.  The
    transposed partners follow (:6515-6537).  The init call at 7.6e6 dyn/cm2
    (:6215) is superseded by this call at every collision step (:8318).
    """
    p, dt = _checked_kernel_inputs(p_cgs, dt_coll, "kernels_at_pressure")
    scal = 1.0
    dlnr = math.log(2.0) / (3.0 * scal)
    dtimelnr = dt * dlnr

    out: dict[str, np.ndarray] = {"cwll": _cwll(tables, derived, p, dt)}

    ywli = np.asarray(tables.ywli, dtype=np.float64)   # (300, 500, 750)
    ywii = np.asarray(tables.ywii, dtype=np.float64)
    ywis = np.asarray(tables.ywis, dtype=np.float64)
    plain = {name: np.asarray(getattr(tables, "yw" + name[2:]),
                              dtype=np.float64)
             for name in ("cwlg", "cwlh", "cwls", "cwsg", "cwss")}
    # (family, its 300, 500 and 750 hPa slices), in WRF's statement order.
    ice: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for k in range(ICEMAX):
        ice.append((f"cwli_{k + 1}", tuple(ywli[s][:, :, k] for s in range(3))))
    for name in ("cwlg", "cwlh", "cwls"):
        ice.append((name, tuple(plain[name][s] for s in range(3))))
    for a in range(ICEMAX):
        for b in range(ICEMAX):
            ice.append((f"cwii_{a + 1}_{b + 1}",
                        tuple(ywii[s][:, :, a, b] for s in range(3))))
    for k in range(ICEMAX):
        ice.append((f"cwis_{k + 1}", tuple(ywis[s][:, :, k] for s in range(3))))
    for name in ("cwsg", "cwss"):
        ice.append((name, tuple(plain[name][s] for s in range(3))))

    p_1, p_2, p_3 = _P2, _P3, _P4
    for name, (k300, k500, k750) in ice:
        value = None
        if p >= p_1:
            value = k750 * dtimelnr
        if p <= p_3:
            value = k300 * dtimelnr
        if p < p_1 and p >= p_2:
            pdm = (p - p_2) / (p_1 - p_2)
            value = (k500 + (k750 - k500) * pdm) * dtimelnr
        if p < p_2 and p > p_3:
            pdm = (p - p_3) / (p_2 - p_3)
            value = (k300 + (k500 - k300) * pdm) * dtimelnr
        out[name] = value

    for k in range(1, ICEMAX + 1):
        out[f"cwil_{k}"] = out[f"cwli_{k}"].T
        out[f"cwsi_{k}"] = out[f"cwis_{k}"].T
    out["cwgl"] = out["cwlg"].T
    out["cwhl"] = out["cwlh"].T
    out["cwsl"] = out["cwls"].T
    for k in range(1, ICEMAX + 1):
        out[f"cwig_{k}"] = np.zeros((NKR, NKR), dtype=np.float64)
        out[f"cwih_{k}"] = np.zeros((NKR, NKR), dtype=np.float64)

    frozen = {name: _frozen(np.array(out[name], dtype=np.float64, order="C",
                                     copy=True))
              for name in KERNEL_FAMILIES}
    return FsbmKernels(p_cgs=p, dt_coll=dt, **frozen)
