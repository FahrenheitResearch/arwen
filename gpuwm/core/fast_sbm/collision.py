"""Drop collision-coalescence, collisional breakup and spontaneous breakup.

Host float64 transcription of the warm-rain collision routines of WRF v4.8.0
``phys/module_mp_fast_sbm.F`` (LF SHA-256
``fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e``,
identical to v4.6.1 and v4.7.1).  WRF is NCAR/UCAR public domain; the notice
is in the repository ``NOTICE`` and in ``gpuwm/data/fast_sbm/LICENSE-WRF.txt``.
The scheme is FSBM-2 by Jacob Shpund and co-authors, from Alexander Khain's
Hebrew University Cloud Model (see the package docstring for the full credit
and the citation WRF asks for).  Nothing here imports cupy; this is the
verification reference the later device kernels are gated against.

Routines and the lines they transcribe
--------------------------------------
``coll_xxx_lwf``
    ``coll_xxx_lwf`` (:652-827) in ``module_mp_SBM_Collision``, with that
    module's ``kp_flux_max = 44`` and ``G_LIM = 1.0D-16`` (:457-458).  Bott's
    (1998) flux method for self-collection of one species on the
    mass-doubling grid, with the liquid-fraction bookkeeping (``fl``) and the
    fatal check on it (:754-813).
``coll_breakup``
    ``coll_breakup_KS`` (:1505-1634): Bleck's (1970) first-order explicit
    step for collisional breakup over the top ``JBREAK = 18`` drop bins, with
    the gain coefficients ``PKIJ`` and loss coefficients ``QKJ`` that
    ``BREAKINIT_KS`` reads (:8725, :8755), after Low and List (1982b) and
    Seifert et al. (2005).  ``NKRinput`` equals ``NKR`` at its only call
    (:8394), so ``nkrdiff`` is zero and is dropped.
``coal_bott_warm``
    The warm path of ``COAL_BOTT_NEW`` (:8263-8642): the Bott units
    (:8324-8325), the breakup trigger (:8335-8337), drop-drop coalescence
    with ``prdkrn = 1`` (:8372-8375), the breakup retry logic (:8376-8434)
    and the final reconstruction of ``FF1R`` with its fatal check
    (:8604-8611).  ``IBREAKUP = 1`` (:3880).
``spont_rain_breakup``
    ``Spont_Rain_BreakUp`` (:205-300) in ``module_mp_SBM_BreakUp``, with that
    module's ``I_Break_Method = 1`` and ``COL`` (:106-108): Srivastava's
    (1971) exponential breakup probability, fragments spread over the
    smaller bins with the mass-normalised weights of ``Spontanous_Init``
    (:120-189).

What is not here, and why
-------------------------
* ``Kernals_KS`` (:6238-6541) and ``ckern_z`` (:6544-6558) build the kernels
  from the tables at the point's pressure at the top of every
  ``COAL_BOTT_NEW`` call (:8317-8318).  They belong to the derived-table
  item, so the kernels come in as an argument (an object with a ``cwll``
  attribute, or the ``cwll`` array itself) and this module does not import
  ``derived.py``.
* ``modkrn_KS`` (:1395-1503) is called at :8320 with ``ICase = 11``.  Above
  0 C it returns at :1417 before it sets its output ``factor_t`` (design
  finding D6), so ``PRDKRN`` and the ``CollEff`` it is copied into (:8322)
  are left unset on the warm path.  Nothing on the warm path reads them:
  drop-drop coalescence passes the literal ``1.d0`` (:8375), and ``CollEff``
  only feeds a diagnostic.  ``CoalResult.colleff`` is therefore ``None``.
* The cold collision block (:8437-8602: drop-snow, drop-graupel, drop-hail,
  snow-snow, Hallett-Mossop, the riming temperature correction and its
  ``deldrop`` fatal check) runs only when ``TT <= 273.15``.  It is milestone
  M3 scope, so ``coal_bott_warm`` refuses a temperature at or below 273.15 K
  and refuses nonzero snow, crystal or dense-ice bins.  With no ice and
  ``TT > 273.15`` the rest of ``COAL_BOTT_NEW`` leaves ``TT``, ``DEL1in``
  and ``DEL2in`` unchanged, reconstructs zero ice bins as zero, and cannot
  reach its ``Del_T 5 K`` check (:8635).
* The spontaneous breakup gate ``SUM(FF1R) > 43.0*1.0D-30`` (:4826) is the
  caller's; ``spont_rain_breakup`` starts where the routine does, with its
  own ``nkr*1.D-30`` early return (:228).

Conventions
-----------
Arrays are zero-based and keep Fortran index order (``cwll[i-1, j-1]`` is
``CWLL(I,J)``, ``pkij[k-1, i-1, j-1]`` is ``PKIJ(K,I,J)``,
``gain_var_new[j-1, i-1]`` is ``Gain_Var_New(J,I)``).  ``ima`` holds WRF's
bin numbers as the oracle dump holds them (one-based values, ``0`` where
``courant_bott_KS`` found no target), and ``ikr_spon_break`` is a WRF bin
number (33 on this grid).  Every loop runs in WRF's order and every product
keeps Fortran's left-to-right association: the Bott update is sequential and
in place, so the order is part of the result.  Sums are sequential loops,
never ``np.sum``.  Every function returns new arrays and leaves its inputs
alone, except the ``tally`` accumulator, which is updated in place.

The literals are the promoted build's (``-freal-4-real-8``): ``1.001`` at
:8423 and ``COL = 0.23105`` at :108 are the doubles nearest those decimals,
where WRF's native build holds the float32 value widened to double
(``1.00100004673``, ``0.231049999594688``).  That difference belongs to the
native-envelope gate.

WRF's fatal checks become ``FsbmRefusal``:
``coll_xxx_lwf`` liquid fraction outside [0, 1.0001] (:754-813);
``coll_breakup_KS`` NaN (:1578-1593); ``COAL_BOTT_NEW`` NaN after breakup
(:8409-8415) and NaN or negative ``FF1R`` at the end (:8606-8611).  The
breakup retry that WRF leaves unbounded (design finding D7, :8423-8425)
refuses once it would take ``ndiv`` above ``behavior.breakup_retry_cap``.

Sources: Bott, A., 1998, J. Atmos. Sci., 55, 2284-2293 (flux method);
Bleck, R., 1970, J. Geophys. Res., 75, 5165-5171 (breakup integration);
Low, T. B., and R. List, 1982b, J. Atmos. Sci., 39, 1607-1618 (fragment
distributions); Seifert, A., A. Khain, U. Blahak, and K. D. Beheng, 2005,
J. Atmos. Sci., 62, 1917-1931 (breakup in a bin model); Srivastava, R. C.,
1971, J. Atmos. Sci., 28, 410-415 (spontaneous breakup); Pinsky, M.,
A. Khain, and M. Shapiro, 2001, J. Atmos. Sci., 58, 742-764 (the drop-drop
kernels the tables carry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    JMAX,
    KRMIN_BREAKUP,
    NKR,
    FsbmBehavior,
    FsbmRefusal,
)

# ---------------------------------------------------------------------------
# Module parameters
# ---------------------------------------------------------------------------

#: ``module_mp_SBM_Collision``: flux halving above this target bin (:457).
#: Unreachable with 33 bins; kept so the transcription reads as WRF does.
KP_FLUX_MAX = 44
#: ``module_mp_SBM_Collision``: ``G_LIM``, g/cm3 (:458).  ``coll_xxx_lwf``
#: works with ``gmin = g_lim*1.0d3`` (:668).
G_LIM_COLLISION = 1.0e-16
#: ``COAL_BOTT_NEW``: ``g_lim = 1.0D-19*1.0D3`` (:8304), the threshold for
#: ``icol_drop`` and ``icol_drop_brk``.
G_LIM_COAL = 1.0e-19 * 1.0e3
#: ``IBREAKUP`` (:3880): collisional breakup is on.
IBREAKUP = 1
#: ``COAL_BOTT_NEW``: the drop-drop ``prdkrn`` literal (:8375).
PRDKRN_DROP_DROP = 1.0
#: ``COAL_BOTT_NEW``: the negativity retry doubles ``ndiv`` while it is
#: below this bound (:8400), then gives up on breakup for the call.
NDIV_NEGATIVE_LIMIT = 8
#: ``COAL_BOTT_NEW``: the mass-gain criterion (:8423), as the promoted
#: build holds the literal.
BREAKUP_MASS_GAIN_LIMIT = 1.001
#: ``module_mp_SBM_BreakUp``: ``COL`` (:108), as the promoted build holds it.
#: Equal to ``constants.COL`` in float64; the native build widens the float32
#: value of the literal instead.
COL_BREAKUP = 0.23105
#: ``module_mp_SBM_BreakUp``: ``I_Break_Method`` (:107), Srivastava (1971).
I_BREAK_METHOD = 1
#: The cold collision block runs when ``tt <= 273.15`` (:8440).
T_COLD_BLOCK = 273.15


# ---------------------------------------------------------------------------
# Helpers with Fortran's semantics for the cases Python treats differently
# ---------------------------------------------------------------------------

def _dlog(value: float) -> float:
    """``DLOG``: NaN for a negative argument and -Inf for zero, as gfortran
    returns them, where ``math.log`` raises."""
    if value > 0.0:
        return math.log(value)
    if value == 0.0:
        return -math.inf
    return math.nan


def _dexp(value: float) -> float:
    """``DEXP``: +Inf on overflow, as gfortran returns it."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _vector(values, where: str, name: str, size: int | None = None) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or (size is not None and arr.shape[0] != size):
        want = f"({size},)" if size is not None else "one-dimensional"
        raise FsbmRefusal(where, f"{name} has shape {arr.shape}, expected {want}")
    return [float(v) for v in arr]


def _matrix(values, where: str, name: str, shape: tuple[int, ...]) -> list:
    arr = np.asarray(values)
    if arr.shape != shape:
        raise FsbmRefusal(where, f"{name} has shape {arr.shape}, expected {shape}")
    return arr.tolist()


def _kernel_array(kernels) -> np.ndarray:
    cwll = getattr(kernels, "cwll", kernels)
    return np.asarray(cwll, dtype=np.float64)


# ---------------------------------------------------------------------------
# coll_xxx_lwf (:652-827)
# ---------------------------------------------------------------------------

def coll_xxx_lwf(g, fl, ckxx, x, c, ima, prdkrn, *, tally=None):
    """Self-collection of one species by Bott's flux method, ``coll_xxx_lwf``.

    Transcribes ``coll_xxx_lwf`` (:652-827).  ``g`` is Bott's mass
    distribution on the bins (``3 x^2 f`` in mg/cm3 per unit ln r, the
    ``G1 = FF1R*3*XL^2*1e3`` of :8325), ``fl`` its liquid fraction, ``ckxx``
    the collection kernel already multiplied by the time step and ``dlnr``
    (``CWLL`` for drops), ``x`` the bin masses in mg (``XL_MG``, :6078), ``c``
    the Courant numbers ``CHUCM`` and ``ima`` the target bins ``IMA`` of
    ``courant_bott_KS`` (WRF bin numbers), ``prdkrn`` the kernel factor.

    The update is sequential and in place, pair by pair in WRF's order: bins
    ``ix0..ix1`` (both at most ``nkr-1``, :670-684), ``i`` outer, ``j >= i``
    inner.  A pair takes ``gsi`` from bin ``i`` and ``gsj`` from bin ``j``
    (:693-710), adds their sum to the target bin ``k = ima(i, j)`` (:711) and
    moves Bott's upwind flux on to ``k + 1`` (:719-746).  The floors at ``gmin``
    on ``g(k)`` and ``g(k+1)`` (:736, :743) and the clip at zero on ``g(i)``
    and ``g(j)`` (:705-709) are WRF's.  A liquid fraction outside
    [0, 1.0001] after the two 1.0001 snaps (:748-752) is WRF's fatal stop
    (:754-813) and refuses.

    ``tally``, when given, is a float64 array of shape (nkr, nkr) that is
    added to in place: ``tally[a-1, b-1]`` receives the mass (in the units
    of ``g``) taken from bin ``a`` by its collisions with bin ``b``, as the
    decrement WRF applies at :705-709 (after the clip at zero), or the reset
    to zero of :713-717 where that branch runs.  A self-collision adds both
    decrements to the diagonal.  Row ``a`` therefore sums to the mass
    removed from bin ``a``, and ``tally + tally.T`` above the diagonal is
    what each pair took from its two bins.  When ``tally`` is ``None`` the
    loop does no extra work.

    Reading the tally: for ``i < j`` the target is the larger bin itself
    (``k = j``), so ``tally[i-1, j-1]`` is the mass of small drops collected
    by bin ``j``, and ``tally[j-1, i-1]`` is Bott's collector term ``gsj``,
    which WRF takes from bin ``j`` without the clip (:707-709) and returns
    with the collected mass in the same step; it counts collector mass once
    per collision and can exceed what the bin holds.  Two WRF limits make
    the decrements differ from ``gsi`` and ``gsj``: the clip at zero, and
    for a self-collection the limiter at :694-696, which allows
    ``gsi = gsj = g(i)``, twice what the bin holds; the clip then removes
    only what is there while the full ``gsk`` reaches the target.

    Returns new arrays ``(g, fl)``; the inputs are not modified.
    """
    where = "coll_xxx_lwf (:652-827)"
    g = _vector(g, where, "g")
    nkr = len(g)
    fl = _vector(fl, where, "fl", nkr)
    x = _vector(x, where, "x", nkr)
    ck = _matrix(np.asarray(ckxx, dtype=np.float64), where, "ckxx", (nkr, nkr))
    cc = _matrix(np.asarray(c, dtype=np.float64), where, "c", (nkr, nkr))
    im = _matrix(np.asarray(ima), where, "ima", (nkr, nkr))
    prdkrn = float(prdkrn)
    if tally is not None:
        if not isinstance(tally, np.ndarray) or tally.shape != (nkr, nkr) \
                or tally.dtype != np.float64:
            raise FsbmRefusal(where, f"tally must be a float64 array of shape "
                                     f"({nkr}, {nkr})")

    gmin = G_LIM_COLLISION * 1.0e3

    # ix0 - lower limit of integration by i (:672-677)
    ix0 = 1
    for i in range(1, nkr):
        ix0 = i
        if g[i - 1] > gmin:
            break
    if ix0 == nkr - 1:
        return np.array(g), np.array(fl)

    # ix1 - upper limit of integration by i (:680-684)
    ix1 = nkr - 1
    for i in range(nkr - 1, 0, -1):
        ix1 = i
        if g[i - 1] > gmin:
            break

    # collisions (:687-816)
    for i in range(ix0, ix1 + 1):
        if g[i - 1] <= gmin:
            continue
        for j in range(i, ix1 + 1):
            if g[j - 1] <= gmin:
                continue
            k = int(im[i - 1][j - 1])
            kp = k + 1
            x01 = ck[i - 1][j - 1] * g[i - 1] * g[j - 1] * prdkrn
            x02 = min(x01, g[i - 1] * x[j - 1])
            if j != k:
                x03 = min(x02, g[j - 1] * x[i - 1])
            if j == k:
                x03 = x02
            gsi = x03 / x[j - 1]
            gsj = x03 / x[i - 1]
            gsk = gsi + gsj
            if gsk <= gmin:
                continue
            gsi_w = gsi * fl[i - 1]
            gsj_w = gsj * fl[j - 1]
            gsk_w = gsi_w + gsj_w
            gsk_w = min(gsk_w, gsk)
            gi_before = g[i - 1]
            g[i - 1] = g[i - 1] - gsi
            g[i - 1] = max(g[i - 1], 0.0)
            gi_after = g[i - 1]
            gj_before = g[j - 1]
            g[j - 1] = g[j - 1] - gsj
            # new change of 23.01.11 (:708-710)
            if j != k:
                g[j - 1] = max(g[j - 1], 0.0)
            gk = g[k - 1] + gsk

            if g[j - 1] < 0.0 and gk <= gmin:
                g[j - 1] = 0.0
                if tally is not None:
                    _tally_pair(tally, i, j, gi_before - gi_after, gj_before)
                g[k - 1] = g[k - 1] + gsi
                continue
            if tally is not None:
                _tally_pair(tally, i, j, gi_before - gi_after, gj_before - g[j - 1])

            if gk <= gmin:
                continue

            gk_w = g[k - 1] * fl[k - 1] + gsk_w
            gk_w = min(gk_w, gk)

            fl_gk = gk_w / gk  # noqa: F841  (computed, unused, as in WRF)
            fl_gsk = gsk_w / gsk
            flux = 0.0
            x1 = _dlog(g[kp - 1] / gk + 1.0e-15)
            flux = gsk / x1 * (_dexp(0.5 * x1) - _dexp(x1 * (0.5 - cc[i - 1][j - 1])))
            flux = min(flux, gsk)
            flux = min(flux, gk)
            if kp > KP_FLUX_MAX:
                flux = 0.5 * flux
            flux_w = flux * fl_gsk
            flux_w = min(flux_w, gsk_w)
            flux_w = min(flux_w, gk_w)
            g[k - 1] = gk - flux
            g[k - 1] = max(g[k - 1], gmin)
            g_k_w = gk_w - flux_w
            g_k_w = min(g_k_w, g[k - 1])
            g_k_w = max(g_k_w, 0.0)
            fl[k - 1] = g_k_w / g[k - 1]
            g_kp_old = g[kp - 1]
            g[kp - 1] = g[kp - 1] + flux
            g[kp - 1] = max(g[kp - 1], gmin)
            g_kp_w = g_kp_old * fl[kp - 1] + flux_w
            g_kp_w = min(g_kp_w, g[kp - 1])
            fl[kp - 1] = g_kp_w / g[kp - 1]

            if 1.0 < fl[k - 1] <= 1.0001:
                fl[k - 1] = 1.0
            if 1.0 < fl[kp - 1] <= 1.0001:
                fl[kp - 1] = 1.0

            if (fl[k - 1] > 1.0001 or fl[kp - 1] > 1.0001
                    or fl[k - 1] < 0.0 or fl[kp - 1] < 0.0):
                raise FsbmRefusal(
                    "coll_xxx_lwf (:754-813)",
                    f"liquid fraction out of range after pair i={i}, j={j} "
                    f"(k={k}, kp={kp}): fl(k)={fl[k - 1]!r}, "
                    f"fl(kp)={fl[kp - 1]!r}; WRF stops the model here")

    return np.array(g), np.array(fl)


def _tally_pair(tally: np.ndarray, i: int, j: int, taken_i: float,
                taken_j: float) -> None:
    if i == j:
        tally[i - 1, i - 1] += taken_i + taken_j
    else:
        tally[i - 1, j - 1] += taken_i
        tally[j - 1, i - 1] += taken_j


# ---------------------------------------------------------------------------
# coll_breakup_KS (:1505-1634)
# ---------------------------------------------------------------------------

def coll_breakup(gt_mg, xt_mg, dt, pkij, qkj):
    """One explicit step of collisional breakup, ``coll_breakup_KS``.

    Transcribes ``coll_breakup_KS`` (:1505-1634) with ``nkrdiff = 0`` (its
    only call passes ``NKR`` twice, :8394).  ``gt_mg`` is the mass
    distribution in COAL_BOTT_NEW's breakup units (``G1*1.D-3``, :8384),
    ``xt_mg`` the bin masses those units go with (``XL_MG*1.D-3``, :8386,
    grams despite the name), ``dt`` the substep, ``pkij`` (jbreak, jbreak,
    jbreak) and ``qkj`` (jbreak, jbreak) the gain and loss coefficients.

    The number density ``ft = gt/xt/xt`` of the top ``jbreak`` bins
    (:1539-1553) gains ``sum_{i, j<=i} fa_i fa_j PKIJ(k,i,j)`` and loses
    ``fa_k sum_j fa_j QKJ(k,j)``, weighted by ``2/(x_{j+1}^2 - x_j^2)`` with
    ``x_{jmax+1} = 2 x_jmax`` (:1559-1576), and ``gt`` is advanced by one
    first-order step (:1596-1630).  A NaN rate is WRF's fatal stop
    (:1578-1593) and refuses.  Negative results are returned as they are:
    ``COAL_BOTT_NEW`` decides what to do with them (:8397-8408).

    Returns a new array; the inputs are not modified.
    """
    where = "coll_breakup_KS (:1505-1634)"
    gt = _vector(gt_mg, where, "gt_mg")
    jmax = len(gt)
    xt = _vector(xt_mg, where, "xt_mg", jmax)
    pk = np.asarray(pkij, dtype=np.float64)
    if pk.ndim != 3 or len(set(pk.shape)) != 1 or pk.shape[0] > jmax:
        raise FsbmRefusal(where, f"pkij has shape {pk.shape}, expected "
                                 f"(jbreak, jbreak, jbreak) with jbreak <= {jmax}")
    jbreak = pk.shape[0]
    pk = pk.tolist()
    qk = _matrix(np.asarray(qkj, dtype=np.float64), where, "qkj", (jbreak, jbreak))
    dt = float(dt)
    ie = je = ke = jbreak

    # :1538-1543
    ft = [0.0] * jmax
    for j in range(1, jmax + 1):
        ft[j - 1] = gt[j - 1] / xt[j - 1] / xt[j - 1]

    # shift between coagulation and breakup grid (:1546-1553)
    jdiff = jmax - jbreak
    fa = [0.0] * jmax
    for k in range(1, ke + 1):
        fa[k - 1] = ft[k + jdiff - 1]

    # :1559
    xt = xt + [xt[jmax - 1] * 2.0]

    dbreak = [0.0] * jbreak
    for k in range(1, ke + 1):
        gain = 0.0
        for i in range(1, ie + 1):
            for j in range(1, i + 1):
                gain = gain + fa[i - 1] * fa[j - 1] * pk[k - 1][i - 1][j - 1]
        aloss = 0.0
        for j in range(1, je + 1):
            aloss = aloss + fa[j - 1] * qk[k - 1][j - 1]
        j = jmax - jbreak + k
        amweight = 2.0 / (xt[j] * xt[j] - xt[j - 1] * xt[j - 1])
        dbreak[k - 1] = amweight * (gain - fa[k - 1] * aloss)
        if dbreak[k - 1] != dbreak[k - 1]:
            raise FsbmRefusal(
                "coll_breakup_KS (:1578-1593)",
                f"breakup rate is NaN at breakup bin {k} (drop bin {j}); "
                f"WRF stops the model here")

    # shift rate to coagulation grid (:1596-1604)
    df = [0.0] * jmax
    for j in range(1, ke + 1):
        df[j + jdiff - 1] = dbreak[j - 1]

    # transformation to mass distribution function g(ln x) (:1606-1609)
    dg = [0.0] * jmax
    for j in range(1, jmax + 1):
        dg[j - 1] = df[j - 1] * xt[j - 1] * xt[j - 1]

    # time integration (:1613-1628)
    for j in range(1, jmax + 1):
        gt[j - 1] = gt[j - 1] + dg[j - 1] * dt

    return np.array(gt)


# ---------------------------------------------------------------------------
# COAL_BOTT_NEW, warm path (:8263-8642)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoalResult:
    """What the warm path of ``COAL_BOTT_NEW`` returns.

    ``ff1r``
        The drop distribution after the call, ``FF1R`` (:8607).
    ``tt``, ``del1in``, ``del2in``
        Unchanged on the warm path (the riming correction at :8588-8602
        needs ``TT <= 273.15``); returned so a caller can treat the warm and
        cold paths alike.
    ``icol_drop``, ``icol_drop_brk``
        WRF's flags (:8335-8337): any drop bin above ``g_lim``; any bin
        above ``KRMIN_BREAKUP`` above ``g_lim``.
    ``breakup_path``
        0: breakup did not run; 1: it ran and was applied; 2: it gave up
        because ``FF1R`` stayed negative at ``ndiv >= 8`` (:8403-8405), so
        ``G1`` keeps the coalesced spectrum.  The oracle probe uses the same
        numbering.
    ``ndiv``
        The final substep count, ``None`` when breakup did not run.
    ``ndiv_trials``
        Every ``ndiv`` tried, in order.
    ``break_drop_per``
        Mass after breakup over mass before (:8418-8422) for the accepted
        attempt, ``None`` unless ``breakup_path == 1``.
    ``g1_before_xxx``, ``g1_after_xxx``, ``g1_after_breakup``
        ``G1`` entering and leaving ``coll_xxx_lwf`` and at label 11
        (:8436), in Bott units (``None`` for the first two when
        ``icol_drop`` is 0).
    ``colleff``
        ``None``: ``modkrn_KS`` does not set ``PRDKRN`` above 0 C (D6).
    """

    ff1r: np.ndarray
    tt: float
    del1in: float
    del2in: float
    icol_drop: int
    icol_drop_brk: int
    breakup_path: int
    ndiv: int | None
    ndiv_trials: tuple[int, ...]
    break_drop_per: float | None
    g1_before_xxx: np.ndarray | None
    g1_after_xxx: np.ndarray | None
    g1_after_breakup: np.ndarray
    colleff: None = None


def coal_bott_warm(ff1r, tt, qq, pp, rho, dt_coll, tables, derived, kernels,
                   behavior: FsbmBehavior | None = None, *, del1in=0.0,
                   del2in=0.0, ff2r=None, ff3r=None, ff4r=None, ff5r=None,
                   tally=None) -> CoalResult:
    """The warm path of ``COAL_BOTT_NEW`` (:8263-8642) for one point.

    ``ff1r`` is the drop distribution ``FF1R`` (33,), 1/(g cm3); ``tt`` K;
    ``qq`` kg/kg; ``pp`` dyn/cm2; ``rho`` g/cm3; ``dt_coll`` s.  ``tables``
    supplies ``xl``, ``pkij`` and ``qkj`` (an ``FsbmTables``); ``derived``
    supplies ``xl_mg``, ``ima`` and ``chucm`` (an ``FsbmDerived`` or any
    object with those attributes); ``kernels`` is the drop-drop kernel at
    this pressure and ``dt_coll``, as an object with a ``cwll`` attribute or
    the (33, 33) ``cwll`` array itself, which ``Kernals_KS`` builds at
    :8318.  ``qq`` and ``rho`` are read only by the cold path; they are
    accepted so the signature is the routine's.

    Steps, in WRF's order:

    1. Bott units ``G1 = FF1R*3*XL*XL*1e3`` (:8325); ``icol_drop`` when any
       bin exceeds ``g_lim`` and ``icol_drop_brk`` when any bin above
       ``KRMIN_BREAKUP`` does (:8335-8337).
    2. Drop-drop coalescence, ``coll_xxx_lwf`` with ``prdkrn = 1``
       (:8374-8375).
    3. Collisional breakup when both flags are set (:8377-8432): ``ndiv``
       calls of ``coll_breakup_KS`` with ``dt_coll/ndiv``, restarting from
       the coalesced ``G1``.  If a reconstructed ``FF1R`` bin is negative,
       ``ndiv`` doubles while below 8; at 8 or more breakup is abandoned for
       the call (label 11).  If the drop mass grows by more than 1.001,
       ``ndiv`` doubles again; WRF sets no bound on this (design finding
       D7), so here it refuses once ``ndiv`` would exceed
       ``behavior.breakup_retry_cap``.  An accepted attempt replaces ``G1``.
    4. ``FF1R = G1/(3*XL*XL*1e3)``; a NaN or negative bin is WRF's fatal
       stop (:8606-8611) and refuses.

    Refuses ``tt <= 273.15`` and any nonzero ``ff2r``, ``ff3r``, ``ff4r``
    or ``ff5r``: the cold block and the ice collisions are milestone M3.
    ``tally`` is passed to ``coll_xxx_lwf`` (drop-drop pairs only).
    """
    where = "coal_bott_warm"
    behavior = FsbmBehavior.corrected() if behavior is None else behavior
    tt = float(tt)
    if not tt > T_COLD_BLOCK:
        raise FsbmRefusal(
            where, f"TT = {tt!r} K is at or below 273.15 K; COAL_BOTT_NEW's "
                   f"cold collision block (:8440-8602) is milestone M3 scope")
    for name, bins in (("ff2r", ff2r), ("ff3r", ff3r), ("ff4r", ff4r),
                       ("ff5r", ff5r)):
        if bins is not None and np.any(np.asarray(bins, dtype=np.float64) != 0.0):
            raise FsbmRefusal(
                where, f"{name} holds nonzero ice bins; collisions involving "
                       f"snow, crystals or dense ice are milestone M3 scope")

    ff = _vector(ff1r, where, "ff1r", NKR)
    xl = _vector(tables.xl, where, "tables.xl", NKR)
    xl_mg = _vector(derived.xl_mg, where, "derived.xl_mg", NKR)
    cwll = _kernel_array(kernels)
    if cwll.shape != (NKR, NKR):
        raise FsbmRefusal(where, f"cwll has shape {cwll.shape}, expected "
                                 f"({NKR}, {NKR})")
    dt_coll = float(dt_coll)

    # :8324-8344
    icol_drop_brk = 0
    icol_drop = 0
    g1 = [0.0] * NKR
    for kr in range(1, NKR + 1):
        g1[kr - 1] = ff[kr - 1] * 3.0 * xl[kr - 1] * xl[kr - 1] * 1.0e3
        if kr > KRMIN_BREAKUP and g1[kr - 1] > G_LIM_COAL:
            icol_drop_brk = 1
        if IBREAKUP != 1:
            icol_drop_brk = 0
        if g1[kr - 1] > G_LIM_COAL:
            icol_drop = 1

    g1_before_xxx = None
    g1_after_xxx = None
    breakup_path = 0
    ndiv_final = None
    trials: list[int] = []
    break_drop_per = None

    if icol_drop == 1:
        # Drop-drop collisions (:8374-8375)
        g1_before_xxx = np.array(g1)
        fl1 = np.ones(NKR)
        g1_arr, _ = coll_xxx_lwf(g1, fl1, cwll, xl_mg, derived.chucm,
                                 derived.ima, PRDKRN_DROP_DROP, tally=tally)
        g1 = [float(v) for v in g1_arr]
        g1_after_xxx = np.array(g1)

        # Breakup (:8377-8432)
        if icol_drop_brk == 1:
            ndiv = 1
            cap = behavior.breakup_retry_cap
            while True:  # label 10
                trials.append(ndiv)
                for it in range(1, ndiv + 1):
                    dtbreakup = dt_coll / ndiv
                    if it == 1:
                        gdumb = [v * 1.0e-3 for v in g1]
                        xl_dumb = [v * 1.0e-3 for v in xl_mg]
                        break_drop_bef = 0.0
                        for kr in range(JMAX):
                            break_drop_bef = break_drop_bef + g1[kr] * 1.0e-3
                    gdumb = [float(v) for v in coll_breakup(
                        gdumb, xl_dumb, dtbreakup, tables.pkij, tables.qkj)]

                # :8397-8416
                restart = False
                give_up = False
                for kr in range(1, NKR + 1):
                    f = (1.0e3 * gdumb[kr - 1]) / (3.0 * xl[kr - 1] * xl[kr - 1] * 1.0e3)
                    if f < 0.0:
                        if ndiv < NDIV_NEGATIVE_LIMIT:
                            ndiv = 2 * ndiv
                            restart = True
                        else:
                            give_up = True
                        break
                    if f != f:
                        raise FsbmRefusal(
                            "COAL_BOTT_NEW (:8409-8415)",
                            f"FF1R is NaN in bin {kr} after collisional "
                            f"breakup with ndiv={ndiv}; WRF stops the model here")
                if restart:
                    continue
                if give_up:
                    breakup_path = 2
                    ndiv_final = ndiv
                    break

                # :8418-8430
                break_drop_aft = 0.0
                for kr in range(JMAX):
                    break_drop_aft = break_drop_aft + gdumb[kr]
                per = break_drop_aft / break_drop_bef
                if per > BREAKUP_MASS_GAIN_LIMIT:
                    if 2 * ndiv > cap:
                        raise FsbmRefusal(
                            "COAL_BOTT_NEW (:8423-8425), design finding D7",
                            f"collisional breakup still multiplies the drop mass "
                            f"by {per!r} at ndiv={ndiv}; doubling again would "
                            f"exceed breakup_retry_cap={cap}, where WRF doubles "
                            f"without a bound")
                    ndiv = ndiv * 2
                    continue
                g1 = [v * 1.0e3 for v in gdumb]
                breakup_path = 1
                ndiv_final = ndiv
                break_drop_per = per
                break

    # label 11 (:8436)
    g1_after_breakup = np.array(g1)

    # :8604-8611
    out = [0.0] * NKR
    for kr in range(1, NKR + 1):
        out[kr - 1] = g1[kr - 1] / (3.0 * xl[kr - 1] * xl[kr - 1] * 1.0e3)
        if out[kr - 1] != out[kr - 1] or out[kr - 1] < 0.0:
            raise FsbmRefusal(
                "COAL_BOTT_NEW (:8606-8611)",
                f"stop at end coal_bott - FF1R NaN or FF1R < 0.0: bin {kr} is "
                f"{out[kr - 1]!r}")

    return CoalResult(
        ff1r=np.array(out), tt=tt, del1in=float(del1in), del2in=float(del2in),
        icol_drop=icol_drop, icol_drop_brk=icol_drop_brk,
        breakup_path=breakup_path, ndiv=ndiv_final, ndiv_trials=tuple(trials),
        break_drop_per=break_drop_per, g1_before_xxx=g1_before_xxx,
        g1_after_xxx=g1_after_xxx, g1_after_breakup=g1_after_breakup)


# ---------------------------------------------------------------------------
# Spont_Rain_BreakUp (:205-300)
# ---------------------------------------------------------------------------

def spont_rain_breakup(dt, ff1r, xl, prob, gain_var_new, ikr_spon_break):
    """Spontaneous raindrop breakup, ``Spont_Rain_BreakUp``.

    Transcribes ``Spont_Rain_BreakUp`` (:205-300) with ``I_Break_Method = 1``
    (Srivastava 1971).  ``dt`` s; ``ff1r`` (33,) 1/(g cm3); ``xl`` the drop
    bin masses (g); ``prob`` (33,) the breakup rate ``2.94e-7 exp(34 r)``
    per second and ``gain_var_new`` (33, 33) the fragment weights, both from
    ``Spontanous_Init`` (:120-189); ``ikr_spon_break`` the first breaking
    bin as a WRF bin number.

    Returns ``ff1r`` unchanged when ``SUM(FF1R) <= nkr*1e-30`` (:228) or
    when the largest positive bin is below ``ikr_spon_break`` (:230-236).
    Otherwise each bin from the largest positive one down to
    ``ikr_spon_break`` keeps ``exp(-prob dt)`` of its drops (:252-261), and
    the broken drops are spread over the smaller bins with
    ``Gain_Var_New(J,I)*XL(I)`` per broken drop (:271-277), which
    ``Spontanous_Init`` normalised so that a drop of bin ``ikr_spon_break``
    conserves its mass.  The local ``DROPRADII`` of :223-226 is computed and
    never used by WRF, and is not computed here.  ``NND`` is used only by
    method 2 and is not an argument.  Returns a new array.
    """
    where = "Spont_Rain_BreakUp (:205-300)"
    ff = _vector(ff1r, where, "ff1r")
    nkr = len(ff)
    xl = _vector(xl, where, "xl", nkr)
    prob = _vector(prob, where, "prob", nkr)
    gvn = _matrix(np.asarray(gain_var_new, dtype=np.float64), where,
                  "gain_var_new", (nkr, nkr))
    ikr = int(ikr_spon_break)
    if not 1 <= ikr <= nkr:
        raise FsbmRefusal(where, f"ikr_spon_break {ikr} is not a bin number "
                                 f"1..{nkr}")
    dtime_spon_break = float(dt)

    # :228 SUM(FF1R), sequential
    total = 0.0
    for v in ff:
        total = total + v
    if total <= nkr * 1.0e-30:
        return np.array(ff)

    # :230-236
    imax = nkr
    for i in range(nkr, 0, -1):
        imax = i
        if ff[i - 1] > 0.0:
            break
    if imax < ikr:
        return np.array(ff)

    # :239-248
    psi1 = ff[:]
    drops_break = [0.0] * nkr
    dropconc_bf = [0.0] * nkr
    for kr in range(1, imax + 1):
        dm = 3.0 * COL_BREAKUP * xl[kr - 1]
        dropconc_bf[kr - 1] = dropconc_bf[kr - 1] + dm * psi1[kr - 1]
    dropconc_af = dropconc_bf[:]

    # :252-261
    for kr in range(imax, ikr - 1, -1):
        tmp_1 = prob[kr - 1] * dtime_spon_break
        tmp_2 = _dexp(-tmp_1)
        tmp_3 = dropconc_bf[kr - 1]
        dropconc_af[kr - 1] = tmp_2 * tmp_3
        drops_break[kr - 1] = dropconc_bf[kr - 1] - dropconc_af[kr - 1]

    # :271-286, I_Break_Method = 1 (method 2 runs the same loop in WRF v4.8.0)
    for j in range(ikr, imax + 1):
        for i in range(1, j):
            dropconc_af[i - 1] = (dropconc_af[i - 1]
                                  + drops_break[j - 1] * gvn[j - 1][i - 1] * xl[i - 1])

    # :290-295
    for kr in range(1, imax + 1):
        dm = 3.0 * COL_BREAKUP * xl[kr - 1]
        psi1[kr - 1] = dropconc_af[kr - 1] / dm

    return np.array(psi1)


__all__ = [
    "BREAKUP_MASS_GAIN_LIMIT", "COL_BREAKUP", "CoalResult", "G_LIM_COAL",
    "G_LIM_COLLISION", "IBREAKUP", "I_BREAK_METHOD", "KP_FLUX_MAX",
    "NDIV_NEGATIVE_LIMIT", "PRDKRN_DROP_DROP", "T_COLD_BLOCK", "coal_bott_warm",
    "coll_breakup", "coll_xxx_lwf", "spont_rain_breakup",
]
