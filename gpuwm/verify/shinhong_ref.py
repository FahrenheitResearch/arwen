"""Float32 NumPy authority for WRF v4.6.1 ``module_bl_shinhong`` (one column).

``np_shinhong_column`` is a statement-order transcription of the 3-D
``shinhong`` wrapper plus ``shinhong2d``, ``tridi1n``/``tridin_ysu``, and --
when ``tke_diag`` is 1 -- ``mixlen``/``prodq2``/``vdifq``, from the
byte-frozen WRF v4.6.1 ``phys/module_bl_shinhong.F`` (sha256
``99f44dbe...``, pinned in ``gpuwm/data/shinhong/oracle/oracle-sha256sums.txt``).
The five partition functions are exposed as ``np_pu``/``np_pq``/``np_pthnl``/
``np_pthl``/``np_ptke`` because the oracle probes them directly.

Transcription rules, established adversarially on the oracle's own toolchain
(gfortran 15.2.0 -O0, glibc 2.43) and recorded bit-for-bit in
``gpuwm/data/shinhong/oracle/pow-probe.txt``:

* everything is float32; every Fortran literal is an ``np.float32``; every
  expression mirrors WRF's association order, one rounding per operation.
* ``x**2.`` and ``x**pfac`` (pfac = 2.0) fold to ``x*x`` -- bitwise per the
  probe, negative bases included -- and are spelled as multiplies here.
* every other real-exponent power stays a correctly-rounded ``powf``,
  implemented as ``np.float32(np.float64(x) ** np.float64(np.float32(y)))``:
  the double-rounding surrogate for glibc's correctly rounded ``powf``,
  verified bitwise against all 26 probed forms of pow-probe.txt (including
  ``x**3.`` and ``x**4.``, which are 1 ULP away from the multiply chains at
  some inputs, and ``x**0.`` == 1.0 for every x including 0).  ``exp`` and
  ``tanh`` use the same float64-round-once rule; ``sqrt`` is hardware-exact
  on the float32 value directly.
* Fortran ``max``/``min``/``amax1`` are ``np.maximum``/``np.minimum``,
  including NaN propagation: case 13 of the fixture is WRF's own
  ``prfac2 = 15.9*wstar3/ust3/(...)`` 0/0 (module_bl_shinhong.F:1010, both
  cubes exactly zero) and the resulting NaN heat column must reproduce.

Two deliberate divergences from the pinned source, both defined-behaviour
guards against reads WRF itself never defines:

* ``efxpbl`` (module_bl_shinhong.F:1557) reads ``q2xk(kpbl+1)``, which is one
  past the end of a ``kts:kte`` array when ``kpbl == kte``.  The read is
  guarded here (``kpbl < kte``); the fixture never exercises the undefined
  arm (case 22 runs ``tke_diag = 0`` for exactly this reason -- see
  tools/shinhong_wrf461_oracle/README.md "Not covered, deliberately").
* ``zfacent`` is only ever assigned below ``kpbl`` (:1006) but the TKE
  staging loop (:1550-1552) reads it at every level; WRF reads stack garbage
  there and never uses the result (every consumer is guarded
  ``k .le. kpbl``).  This port keeps those lanes at 0.0 -- unobservable by
  construction, and no garbage.

Nothing in this module knows the fixture or a tolerance; it computes what the
Fortran computes and hands the words back.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "np_shinhong_column",
    "np_pu",
    "np_pq",
    "np_pthnl",
    "np_pthl",
    "np_ptke",
]

F = np.float32

# --------------------------------------------------------------------------
# libm surrogates (see module docstring; verified against pow-probe.txt).
# --------------------------------------------------------------------------


def _powf(x, y):
    """glibc's correctly rounded powf, as float64 pow rounded once."""
    return F(np.float64(F(x)) ** np.float64(F(y)))


def _expf(x):
    return F(np.exp(np.float64(F(x))))


def _tanhf(x):
    return F(np.tanh(np.float64(F(x))))


def _sqrtf(x):
    return np.sqrt(F(x))


# --------------------------------------------------------------------------
# WRF module_model_constants values exactly as the oracle driver computes
# them (tools/shinhong_wrf461_oracle/run_bl_shinhong.F90:46-55): default
# REAL parameter expressions, folded per operation in float32.
# --------------------------------------------------------------------------
G = F(9.81)
RD = F(287.0)
RV = F(461.6)
CP = F(7.0) * RD / F(2.0)
XLV = F(2.5e6)
EP1 = RV / RD - F(1.0)
KARMAN = F(0.4)

# --------------------------------------------------------------------------
# shinhong2d parameters (module_bl_shinhong.F:321-346).
# --------------------------------------------------------------------------
XKZMINM = F(0.1)
XKZMINH = F(0.01)
XKZMAX = F(1000.0)
RIMIN = F(-100.0)
RLAM = F(30.0)
PRMIN = F(0.25)
PRMAX = F(4.0)
BRCR_UB = F(0.0)
BRCR_SB = F(0.25)
CORI = F(1.0e-4)
AFAC = F(6.8)
BFAC = F(6.8)
PFAC = F(2.0)
PFAC_Q = F(2.0)
PHIFAC = F(8.0)
SFCFRAC = F(0.1)
D1 = F(0.02)
D2 = F(0.05)
D3 = F(0.001)
H1 = F(0.33333333)  # rounds to float32 1/3 (pow-probe.txt: x**h1 == x**(1./3.))
H2 = F(0.6666667)
ZFMIN = F(1.0e-8)
APHI5 = F(5.0)
APHI16 = F(16.0)
TMIN = F(1.0e-2)
GAMCRT = F(3.0)
GAMCRQ = F(2.0e-3)
EPSQ2L = F(0.01)
C_1 = F(1.0)
GAMCRE = F(0.224)
MLTOP = F(1.0)
SFCFRACN1 = F(0.075)
NLFRAC = F(0.7)
ENLFRAC = F(-0.4)
A11 = F(1.0)
A12 = F(-1.15)
EZFAC = F(1.5)
CPENT = F(-0.4)
RIGSMAX = F(100.0)
ENTFMIN = F(1.0)
ENTFMAX = F(5.0)

# --------------------------------------------------------------------------
# mixlen parameters (module_bl_shinhong.F:1789-1826).  The a1/a2x/b1/b2/c1
# decimals are double-precision literals assigned to default REAL: np.float32
# of the decimal (gfortran rounds the literal once to REAL(4); parsing via
# float64 then float32 lands on the same value -- these decimals sit nowhere
# near a float32 tie).  Every derived parameter is a compile-time folding
# gfortran performs per operation in the target kind, in the association
# order of the expression tree -- reproduced below in the same order.
# --------------------------------------------------------------------------
_BLCKDR = F(0.0063)
_CN = F(0.75)
_EPS1 = F(1.0e-12)
_EPSL = F(0.32)
_EPSRU = F(1.0e-7)
_EPSRS = F(1.0e-7)
_EL0MAX = F(1000.0)
_EL0MIN = F(1.0)
_ELFC = F(0.23) * F(0.5)                     # elfc = 0.23*0.5 (:1791)
_ALPH = F(0.30)
_BETA = F(1.0) / F(273.0)                    # beta = 1./273. (:1792)
_G_QNSE = F(9.81)                            # mixlen/prodq2 local g, not the dummy
_BTG = _BETA * _G_QNSE                       # btg = beta*g (:1792)
_A1 = F(0.659888514560862645)
_A2X = F(0.6574209922667784586)
_B1 = F(11.87799326209552761)
_B2 = F(7.226971804046074028)
_C1 = F(0.000830955950095854396)
# adnh = 9.*a1*a2x*a2x*(12.*a1+3.*b2)*btg*btg (:1796)
_ADNH = F(9.0) * _A1 * _A2X * _A2X * (F(12.0) * _A1 + F(3.0) * _B2) * _BTG * _BTG
# adnm = 18.*a1*a1*a2x*(b2-3.*a2x)*btg (:1797)
_ADNM = F(18.0) * _A1 * _A1 * _A2X * (_B2 - F(3.0) * _A2X) * _BTG
# bdnh = 3.*a2x*(7.*a1+b2)*btg, bdnm = 6.*a1*a1 (:1798)
_BDNH = F(3.0) * _A2X * (F(7.0) * _A1 + _B2) * _BTG
_BDNM = F(6.0) * _A1 * _A1
# aeqh, aeqm (:1802-1805)
_AEQH = (F(9.0) * _A1 * _A2X * _A2X * _B1 * _BTG * _BTG
         + F(9.0) * _A1 * _A2X * _A2X * (F(12.0) * _A1 + F(3.0) * _B2) * _BTG * _BTG)
_AEQM = (F(3.0) * _A1 * _A2X * _B1
         * (F(3.0) * _A2X + F(3.0) * _B2 * _C1 + F(18.0) * _A1 * _C1 - _B2) * _BTG
         + F(18.0) * _A1 * _A1 * _A2X * (_B2 - F(3.0) * _A2X) * _BTG)
# requ = -aeqh/aeqm; epsgm = requ*epsgh (:1809-1810)
_REQU = -(_AEQH / _AEQM)
_EPSGH = F(1.0e-9)
_EPSGM = _REQU * _EPSGH
# ubryl/ubry/ubry3 (:1814-1817)
_UBRYL = ((F(18.0) * _REQU * _A1 * _A1 * _A2X * _B2 * _C1 * _BTG
           + F(9.0) * _A1 * _A2X * _A2X * _B2 * _BTG * _BTG)
          / (_REQU * _ADNM + _ADNH))
_UBRY = (F(1.0) + _EPSRS) * _UBRYL
_UBRY3 = F(3.0) * _UBRY
# aubh/aubm/bubh/bubm/cubr/rcubr (:1818-1822)
_AUBH = F(27.0) * _A1 * _A2X * _A2X * _B2 * _BTG * _BTG - _ADNH * _UBRY3
_AUBM = F(54.0) * _A1 * _A1 * _A2X * _B2 * _C1 * _BTG - _ADNM * _UBRY3
_BUBH = (F(9.0) * _A1 * _A2X + F(3.0) * _A2X * _B2) * _BTG - _BDNH * _UBRY3
_BUBM = F(18.0) * _A1 * _A1 * _C1 - _BDNM * _UBRY3
_CUBR = F(1.0) - _UBRY3
_RCUBR = F(1.0) / _CUBR
_ELCBL = F(0.77)

# prodq2 parameters (:2029) and vdifq parameters (:2140).
_C0 = F(0.55)
_CEPS = F(16.6)
_C_K = F(1.0)


# --------------------------------------------------------------------------
# The five partition functions (module_bl_shinhong.F:2297-2427).  In every
# one, ``(doh)**b1`` with b1 = 2.0 folds to a multiply (pow-probe.txt
# ``x**pfac(2.0)``) and the fractional ``b2`` powers stay powf.  The
# ``h .ne. 0`` early-out is exact -- both signed zeros of h return 1.0 --
# and IS reachable from the scheme (cslen of a non-convective column).
# --------------------------------------------------------------------------


def np_pu(d, h):
    """pu (:2297-2319): nonlocal/local momentum-transport partition."""
    a1, a2, a3, a4, a5 = F(1.0), F(0.070), F(1.0), F(0.142), F(0.071)
    b2 = F(0.6666667)
    d, h = F(d), F(h)
    if h != 0:
        doh = d / h
        t2 = doh * doh
        tb = _powf(doh, b2)
        num = a1 * t2 + a2 * tb
        den = a3 * t2 + a4 * tb + a5
        pu = num / den
    else:
        pu = F(1.0)
    pu = np.maximum(pu, F(0.0))
    pu = np.minimum(pu, F(1.0))
    return F(pu)


def np_pq(d, h):
    """pq (:2323-2345): moisture-transport partition (b1 = 2.0 only)."""
    a1, a2, a3, a4, a5 = F(1.0), F(-0.098), F(1.0), F(0.106), F(0.5)
    d, h = F(d), F(h)
    if h != 0:
        doh = d / h
        t2 = doh * doh
        num = a1 * t2 + a2
        den = a3 * t2 + a4
        pq = a5 * num / den + (F(1.0) - a5)
    else:
        pq = F(1.0)
    pq = np.maximum(pq, F(0.0))
    pq = np.minimum(pq, F(1.0))
    return F(pq)


def np_pthnl(d, h):
    """pthnl (:2349-2372): nonlocal heat-transport partition (b2 = 0.875)."""
    a1, a2, a3 = F(1.000), F(0.936), F(-1.110)
    a4, a5, a6, a7 = F(1.000), F(0.312), F(0.329), F(0.243)
    b2 = F(0.875)
    d, h = F(d), F(h)
    if h != 0:
        doh = d / h
        t2 = doh * doh
        tb = _powf(doh, b2)
        num = a1 * t2 + a2 * tb + a3
        den = a4 * t2 + a5 * tb + a6
        pthnl = a7 * num / den + (F(1.0) - a7)
    else:
        pthnl = F(1.0)
    pthnl = np.maximum(pthnl, F(0.0))
    pthnl = np.minimum(pthnl, F(1.0))
    return F(pthnl)


def np_pthl(d, h):
    """pthl (:2376-2399): local heat-transport partition (b2 = 0.5).

    ``(doh)**b2`` stays a powf here too: glibc's correctly rounded powf at
    exponent 0.5 equals sqrtf everywhere, and pow-probe.txt shows the two
    agree bitwise at every probed input, but the port spells what WRF calls.
    """
    a1, a2, a3 = F(1.000), F(0.870), F(-0.913)
    a4, a5, a6, a7 = F(1.000), F(0.153), F(0.278), F(0.280)
    b2 = F(0.5)
    d, h = F(d), F(h)
    if h != 0:
        doh = d / h
        t2 = doh * doh
        tb = _powf(doh, b2)
        num = a1 * t2 + a2 * tb + a3
        den = a4 * t2 + a5 * tb + a6
        pthl = a7 * num / den + (F(1.0) - a7)
    else:
        pthl = F(1.0)
    pthl = np.maximum(pthl, F(0.0))
    pthl = np.minimum(pthl, F(1.0))
    return F(pthl)


def np_ptke(d, h):
    """ptke (:2403-2426): TKE-transport partition (same curve as pu)."""
    a1, a2, a3, a4, a5 = F(1.000), F(0.070), F(1.000), F(0.142), F(0.071)
    b2 = F(0.6666667)
    d, h = F(d), F(h)
    if h != 0:
        doh = d / h
        t2 = doh * doh
        tb = _powf(doh, b2)
        num = a1 * t2 + a2 * tb
        den = a3 * t2 + a4 * tb + a5
        ptke = num / den
    else:
        ptke = F(1.0)
    ptke = np.maximum(ptke, F(0.0))
    ptke = np.minimum(ptke, F(1.0))
    return F(ptke)


def _z1(n):
    """A zeroed float32 scratch array addressed 1-based, like the Fortran."""
    return np.zeros(n, dtype=np.float32)


# --------------------------------------------------------------------------
# The tridiagonal solvers, with the sequence-association shift applied.
#
# SEQUENCE ASSOCIATION (module_bl_shinhong.F:1198/:1324/:1450 against
# :1626/:1717): both solvers receive the actual argument ``al``, dimensioned
# (its:ite, kts:kte), into a dummy ``cl`` dimensioned (its:ite, kts+1:kte+1).
# With one column the storage shift is exactly one k-slot, so ``cl(i,k)``
# inside the solver is the caller's ``al(i,k-1)`` -- transcribed below as
# ``al[k-1]`` wherever the Fortran says ``cl(i,k)``.
#
# One routine serves both tridi1n (:1619) and tridin_ysu (:1710): per
# right-hand side the two emit the identical operation sequence, and the
# ``au(i,k) = fk*cu(i,k)`` recomputation tridin_ysu performs on every ``it``
# sweep is idempotent (fk depends only on cm, cl and au(k-1), all already at
# their final values after the first sweep), so one LU factorisation serves
# every right-hand side.
# --------------------------------------------------------------------------


def _tridi_shared(al, ad, cu, r_list, kte):
    """Solve the WRF tridiagonal system for each right-hand side in r_list."""
    n = kte
    fkk = _z1(n + 1)
    au = _z1(n + 1)
    fkk[1] = F(1.0) / ad[1]
    au[1] = fkk[1] * cu[1]
    for k in range(2, n):                       # kts+1 .. n-1
        fkk[k] = F(1.0) / (ad[k] - al[k - 1] * au[k - 1])   # cl(i,k) == al(k-1)
        au[k] = fkk[k] * cu[k]
    fkk[n] = F(1.0) / (ad[n] - al[n - 1] * au[n - 1])
    outs = []
    for r in r_list:
        f = _z1(n + 1)
        f[1] = fkk[1] * r[1]
        for k in range(2, n):
            f[k] = fkk[k] * (r[k] - al[k - 1] * f[k - 1])
        f[n] = fkk[n] * (r[n] - al[n - 1] * f[n - 1])
        for k in range(n - 1, 0, -1):
            f[k] = f[k] - au[k] * f[k + 1]
        outs.append(f)
    return outs


# --------------------------------------------------------------------------
# mixlen (module_bl_shinhong.F:1776-2012), one column, lmh = lmxl = 1.
# --------------------------------------------------------------------------


def _mixlen(u, v, t, the, q, cwm, q2, z, ustar, corf, epshol,
            hpbl, lpbl, pblflg, hgamu, hgamv, hgamq,
            mf, ufxpbl, vfxpbl, qfxpbl, kte):
    """Returns (s2, gh, ri, el), each 1-based with live entries 2..kte."""
    elocp = F(2.72e6) / CP                       # :1880
    ct = F(0.0)                                  # :1881 (inout, zeroed here)
    q1 = _z1(kte + 1)
    dth = _z1(kte + 1)
    for k in range(2, kte + 1):                  # :1887-1889
        dth[k] = the[k] - the[k - 1]
    for k in range(3, kte + 1):                  # :1891-1896
        if dth[k] > 0 and dth[k - 1] <= 0:
            dth[k] = dth[k] + ct                 # ct == 0.; dth[k] > 0 so exact
            break
    s2 = _z1(kte + 1)
    gh = _z1(kte + 1)
    ri = _z1(kte + 1)
    en2 = _z1(kte + 1)
    el = _z1(kte + 1)
    for k in range(kte, 1, -1):                  # :1900-1926
        rdz = F(2.0) / (z[k + 1] - z[k - 1])
        du = u[k] - u[k - 1]
        dv = v[k] - v[k - 1]
        s2l = (du * du + dv * dv) * rdz * rdz    # integer **2 -> multiply
        if pblflg and k <= lpbl:
            suk = (u[k] - u[k - 1]) * rdz
            svk = (v[k] - v[k - 1]) * rdz
            s2l = ((suk - hgamu / hpbl - ufxpbl[k]) * suk
                   + (svk - hgamv / hpbl - vfxpbl[k]) * svk)
        s2l = np.maximum(s2l, _EPSGM)
        s2[k] = s2l
        tem = (t[k] + t[k - 1]) * F(0.5)
        thm = (the[k] + the[k - 1]) * F(0.5)
        a = thm * EP1                            # p608 == ep1
        b = (elocp / tem - F(1.0) - EP1) * thm
        ghl = ((dth[k] * ((q[k] + q[k - 1] + cwm[k] + cwm[k - 1]) * (F(0.5) * EP1) + F(1.0))
                + (q[k] - q[k - 1] + cwm[k] - cwm[k - 1]) * a
                + (cwm[k] - cwm[k - 1]) * b) * rdz)
        if pblflg and k <= lpbl:                 # :1918-1920
            ghl = ghl - mf[k] - (hgamq / hpbl + qfxpbl[k]) * a
        if np.abs(ghl) <= _EPSGH:
            ghl = _EPSGH
        en2[k] = ghl * _G_QNSE / thm
        gh[k] = ghl
        ri[k] = en2[k] / s2l
    elm = _z1(kte + 1)
    for k in range(kte, 1, -1):                  # :1930-1950
        s2l = s2[k]
        ghl = gh[k]
        if ghl >= _EPSGH:
            if s2l / ghl <= _REQU:
                elm[k] = _EPSL
            else:
                aubr = (_AUBM * s2l + _AUBH * ghl) * ghl
                bubr = _BUBM * s2l + _BUBH * ghl
                qol2st = (F(-0.5) * bubr
                          + _sqrtf(bubr * bubr * F(0.25) - aubr * _CUBR)) * _RCUBR
                eloq2x = F(1.0) / qol2st
                elm[k] = np.maximum(_sqrtf(eloq2x * q2[k]), _EPSL)
        else:
            aden = (_ADNM * s2l + _ADNH * ghl) * ghl
            bden = _BDNM * s2l + _BDNH * ghl
            qol2un = F(-0.5) * bden + _sqrtf(bden * bden * F(0.25) - aden)
            eloq2x = F(1.0) / (qol2un + _EPSRU)
            elm[k] = np.maximum(_sqrtf(eloq2x * q2[k]), _EPSL)
    for k in range(lpbl, 0, -1):                 # :1952-1954  (lmh == 1)
        q1[k] = _sqrtf(q2[k])
    szq = F(0.0)
    sq = F(0.0)
    for k in range(kte, 1, -1):                  # :1956-1962
        qdzl = (q1[k] + q1[k - 1]) * (z[k] - z[k - 1])
        szq = (z[k] + z[k - 1] - z[1] - z[1]) * qdzl + szq
        sq = qdzl + sq
    el0 = np.minimum(_ALPH * szq * F(0.5) / sq, _EL0MAX)   # :1966
    el0 = np.maximum(el0, _EL0MIN)
    rel = _z1(kte + 1)
    lpblm = min(lpbl + 1, kte)                   # :1971
    for k in range(kte, lpblm - 1, -1):          # :1972-1975
        el[k] = (z[k + 1] - z[k - 1]) * _ELFC
        rel[k] = el[k] / elm[k]
    epshol = np.minimum(F(epshol), F(0.0))       # :1979
    ckp = _ELCBL * _powf(F(1.0) - F(8.0) * epshol, F(1.0) / F(3.0))   # :1980
    if lpbl > 1:                                 # :1981-1992
        for k in range(lpbl, 1, -1):
            vkrmz = (z[k] - z[1]) * KARMAN
            if pblflg:
                vkrmz = ckp * (z[k] - z[1]) * KARMAN
                el[k] = vkrmz / (vkrmz / el0 + F(1.0))
            else:
                el[k] = vkrmz / (vkrmz / el0 + F(1.0))
            rel[k] = el[k] / elm[k]
    for k in range(lpbl - 1, 2, -1):             # :1994-1997  (lmh+2 == 3)
        srel = np.minimum(((rel[k - 1] + rel[k + 1]) * F(0.5) + rel[k]) * F(0.5),
                          rel[k])
        el[k] = np.maximum(srel * elm[k], _EPSL)
    fcor = np.maximum(corf, _EPS1)               # :2001 f = max(corf,eps1)
    rlambda = fcor / (_BLCKDR * ustar)           # ustar == 0 -> inf -> el == 0 (IEEE)
    for k in range(kte, 1, -1):                  # :2003-2010
        if en2[k] >= F(0.0):
            vkrmz = (z[k] - z[1]) * KARMAN
            rlb = rlambda + F(1.0) / vkrmz
            rln = _sqrtf(F(2.0) * en2[k] / q2[k]) / _CN
            el[k] = F(1.0) / (rlb + rln)
    return s2, gh, ri, el


# --------------------------------------------------------------------------
# prodq2 (module_bl_shinhong.F:2016-2125), one column.  Mutates q2 in place.
# Note dtturbl is dt, NOT dt2 (:1578 passes the wrapper's dt).
# --------------------------------------------------------------------------


def _prodq2(dtturbl, ustar, s2, ri, q2, el, z, akm, akh,
            uxk, vxk, thxk, thvxk, hgamu, hgamv, hgamq, delxy,
            hpbl, pblflg, kpbl, mf, ufxpbl, vfxpbl, qfxpbl, kte):
    rc02 = F(2.0) / (_C0 * _C0)                  # :2073
    for k in range(2, kte + 1):                  # :2077-2119
        deltaz = F(0.5) * (z[k + 1] - z[k - 1])
        s2l = s2[k]
        q2l = q2[k]
        suk = (uxk[k] - uxk[k - 1]) / deltaz
        svk = (vxk[k] - vxk[k - 1]) / deltaz
        gthvk = (thvxk[k] - thvxk[k - 1]) / deltaz
        govrthvk = _G_QNSE / (F(0.5) * (thvxk[k] + thvxk[k - 1]))
        akml = akm[k]
        akhl = akh[k]
        thm = (thxk[k] + thxk[k - 1]) * F(0.5)
        if pblflg and k <= kpbl:                 # :2092-2098
            pru = (akml * (suk - hgamu / hpbl - ufxpbl[k])) * suk
            prv = (akml * (svk - hgamv / hpbl - vfxpbl[k])) * svk
        else:
            pru = akml * suk * suk
            prv = akml * svk * svk
        pr = pru + prv
        if pblflg and k <= kpbl:                 # :2103-2107
            bpr = (akhl * (gthvk - mf[k]
                           - (hgamq / hpbl + qfxpbl[k]) * EP1 * thm)) * govrthvk
        else:
            bpr = akhl * gthvk * govrthvk
        disel = np.minimum(delxy, _CEPS * el[k])  # :2111
        dis = _powf(q2l, F(1.5)) / disel          # (q2l)**1.5 stays powf
        q2l = q2l + F(2.0) * (pr - bpr - dis) * dtturbl
        q2[k] = np.maximum(q2l, EPSQ2L)           # amax1: NaN propagates
    q2[1] = np.maximum(rc02 * ustar * ustar, EPSQ2L)   # :2123


# --------------------------------------------------------------------------
# vdifq (module_bl_shinhong.F:2129-2231), one column, lmh = 1.  Mutates q2.
# The akq/cm/cr/dtoz/rsq2 arrays run kts+2..kte in the Fortran (:2172-2176);
# the loops below are transcribed index-exactly against that (trap: the
# lower boundary handling at :2212-2225 reaches akq(3)/cm(3)/cr(3) only).
# --------------------------------------------------------------------------


def _vdifq(dtdif, q2, el, z, akhk, ptke1, hgame, hpbl, pblflg, kpbl,
           efxpbl, kte):
    lmh = 1
    zfacentk = _z1(kte + 1)
    for k in range(2, kte + 1):                  # :2183-2186
        zak = F(0.5) * (z[k] + z[k - 1])         # za(k-1) of shinhong2d
        zfacentk[k] = _powf(zak / hpbl, F(3.0))  # ((zak/hpbl))**3.0 -- a powf
    dtoz = _z1(kte + 1)
    akq = _z1(kte + 1)
    cr = _z1(kte + 1)
    cm = _z1(kte + 1)
    rsq2 = _z1(kte + 1)
    for k in range(kte, 2, -1):                  # :2188-2193 (kte .. kts+2)
        dtoz[k] = (dtdif + dtdif) / (z[k + 1] - z[k - 1])
        akq[k] = _C_K * (akhk[k] / (z[k + 1] - z[k - 1])
                         + akhk[k - 1] / (z[k] - z[k - 2]))
        akq[k] = akq[k] * ptke1[k]
        cr[k] = -(dtoz[k] * akq[k])
    akqs = _C_K * akhk[2] / (z[3] - z[1])        # :2195-2196
    akqs = akqs * ptke1[2]
    cm[kte] = dtoz[kte] * akq[kte] + F(1.0)      # :2197-2198
    rsq2[kte] = q2[kte]
    for k in range(kte - 1, 2, -1):              # :2200-2210
        cf = -(dtoz[k] * akq[k + 1] / cm[k + 1])
        cm[k] = -(cr[k + 1] * cf) + (akq[k + 1] + akq[k]) * dtoz[k] + F(1.0)
        rsq2[k] = -(rsq2[k + 1] * cf) + q2[k]
        if pblflg and k < kpbl:
            rsq2[k] = (rsq2[k]
                       - dtoz[k] * (F(2.0) * hgame[k] / hpbl) * akq[k + 1] * (z[k + 1] - z[k])
                       + dtoz[k] * (F(2.0) * hgame[k - 1] / hpbl) * akq[k] * (z[k] - z[k - 1]))
            rsq2[k] = (rsq2[k]
                       - dtoz[k] * F(2.0) * efxpbl * zfacentk[k + 1]
                       + dtoz[k] * F(2.0) * efxpbl * zfacentk[k])
    dtozs = (dtdif + dtdif) / (z[3] - z[1])      # :2212
    cf = -(dtozs * akq[lmh + 2] / cm[lmh + 2])   # :2213
    if pblflg and (lmh + 1) < kpbl:              # :2215-2225
        q2[2] = (dtozs * akqs * q2[1] - rsq2[3] * cf + q2[2]
                 - dtozs * (F(2.0) * hgame[2] / hpbl) * akq[3] * (z[3] - z[2])
                 + dtozs * (F(2.0) * hgame[1] / hpbl) * akqs * (z[2] - z[1]))
        q2[2] = (q2[2] - dtozs * F(2.0) * efxpbl * zfacentk[3]
                 + dtozs * F(2.0) * efxpbl * zfacentk[2])
        q2[2] = q2[2] / ((akq[3] + akqs) * dtozs - cr[3] * cf + F(1.0))
    else:
        q2[2] = ((dtozs * akqs * q2[1] - rsq2[3] * cf + q2[2])
                 / ((akq[3] + akqs) * dtozs - cr[3] * cf + F(1.0)))
    for k in range(3, kte + 1):                  # :2227-2229 (lmh+2 .. kte)
        q2[k] = (-(cr[k] * q2[k - 1]) + rsq2[k]) / cm[k]


# --------------------------------------------------------------------------
# The column: shinhong (:9-262) around shinhong2d (:266-1615).
# --------------------------------------------------------------------------


def np_shinhong_column(u, v, t, qv, qc, qi, p2d, pi2d, p2di, dz8w, tke_in, *,
                       psfc, znt, ust, hfx, qfx, wspd, br, psim, psih,
                       xland, corf, u10, v10, dt, dx, dy, tke_diag,
                       ctopo=1.0, ctopo2=1.0):
    """One column of WRF v4.6.1 ``shinhong``, float32, statement for statement.

    Level inputs are 1-D length-nz float32 (``p2di`` is nz+1); the rest are
    scalars.  ``ctopo``/``ctopo2`` select the arm: (1, 1) is arm A, WRF's
    Registry topo_wind=0 default; (0.85, 0.7) is the oracle's arm B.  Returns
    the wrapper's own outputs: tendencies (``rthblten`` already divided by
    pi3d at :253), ``exch_h``, ``tke``, ``el``, ``hpbl``, 1-based ``kpbl``,
    ``wstar``, ``delta``, and the blended ``u10``/``v10``.

    The whole body runs under ``np.errstate(all="ignore")`` because WRF's own
    arithmetic divides by zero on purpose-built fixture columns (the prfac2
    0/0 at :1010 on case 13, mixlen's rlambda = f/0 at :2002 when ust == 0)
    and the resulting inf/NaN propagation IS the behaviour being transcribed;
    a RuntimeWarning here is expected output, not a defect.
    """
    with np.errstate(divide="ignore", invalid="ignore",
                     over="ignore", under="ignore"):
        return _column(
            np.asarray(u, np.float32), np.asarray(v, np.float32),
            np.asarray(t, np.float32), np.asarray(qv, np.float32),
            np.asarray(qc, np.float32), np.asarray(qi, np.float32),
            np.asarray(p2d, np.float32), np.asarray(pi2d, np.float32),
            np.asarray(p2di, np.float32), np.asarray(dz8w, np.float32),
            np.asarray(tke_in, np.float32),
            F(psfc), F(znt), F(ust), F(hfx), F(qfx), F(wspd), F(br),
            F(psim), F(psih), F(xland), F(corf), F(u10), F(v10),
            F(dt), F(dx), F(dy), int(tke_diag), F(ctopo), F(ctopo2))


def _pad1(a, extra=0):
    out = np.zeros(a.shape[0] + 1 + extra, dtype=np.float32)
    out[1:1 + a.shape[0]] = a
    return out


def _column(u, v, t, qv, qc, qi, p2d, pi2d, p2di, dz8w, tke_in,
            psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, corf,
            u10, v10, dt, dx, dy, tke_diag, ctopo, ctopo2):
    kte = int(u.shape[0])
    klpbl = kte
    # 1-based views, as the Fortran indexes them.
    ux, vx, tx = _pad1(u), _pad1(v), _pad1(t)
    qvx, qcx, qix = _pad1(qv), _pad1(qc), _pad1(qi)
    p2dx, pi2dx = _pad1(p2d), _pad1(pi2d)
    p2dix = _pad1(p2di)                          # 1..kte+1
    dz8wx = _pad1(dz8w)
    tkex = _pad1(tke_in)

    cont = CP / G                                # :544
    conpr = BFAC * KARMAN * SFCFRAC              # :548
    # conq/conw/conwrc (:545-547) feed only the dusfc/dvsfc/dtsfc/dqsfc
    # accumulators, which the wrapper never reads; not transcribed.

    thx = _z1(kte + 1)
    thvx = _z1(kte + 1)
    for k in range(1, kte + 1):                  # :557-561
        thx[k] = tx[k] / pi2dx[k]
    for k in range(1, kte + 1):                  # :563-568
        tvcon = F(1.0) + EP1 * qvx[k]
        thvx[k] = thx[k] * tvcon
    tvcon = F(1.0) + EP1 * qvx[1]                # :570-574
    rhox = psfc / (RD * tx[1] * tvcon)
    govrth = G / thx[1]

    zq = _z1(kte + 2)                            # :579-587
    for k in range(1, kte + 1):
        zq[k + 1] = dz8wx[k] + zq[k]
    za = _z1(kte + 1)
    delp = _z1(kte + 1)
    for k in range(1, kte + 1):                  # :589-595 (dzq computed there
        za[k] = F(0.5) * (zq[k] + zq[k + 1])     # and never read; omitted)
        delp[k] = p2dix[k] - p2dix[k + 1]
    dza = _z1(kte + 1)
    dza[1] = za[1]                               # :597-605
    for k in range(2, kte + 1):
        dza[k] = za[k] - za[k - 1]

    utnp = _z1(kte + 1)                          # :610-613
    vtnp = _z1(kte + 1)
    ttnp = _z1(kte + 1)
    qtnp = np.zeros((4, kte + 1), dtype=np.float32)   # components 1..3

    wspd1 = _sqrtf(ux[1] * ux[1] + vx[1] * vx[1]) + F(1.0e-9)   # :616

    dt2 = F(2.0) * dt                            # :624-626
    rdt = F(1.0) / dt2

    hgamu = F(0.0)
    hgamv = F(0.0)
    delta = F(0.0)
    efxpbl = F(0.0)
    epshol = F(0.0)
    deltaoh = F(0.0)
    rigs = F(0.0)
    enlfrac2 = F(0.0)
    cslen = F(0.0)
    q2x = _z1(kte + 1)
    for k in range(1, kte + 1):                  # :665-669
        q2x[k] = F(2.0) * tkex[k]
    # :671-679 -- el_pbl is zeroed UNCONDITIONALLY, tke_diag on or off.
    el_pbl = _z1(kte + 1)
    tflux_e = _z1(kte + 1)
    qflux_e = _z1(kte + 1)
    tvflux_e = _z1(kte + 1)
    hgame2d = _z1(kte + 1)
    mf = _z1(kte + 1)                            # :681-686
    zfacmf = _z1(kte + 1)
    xkzom = _z1(kte + 1)                         # :688-693 (kts..klpbl-1)
    xkzoh = _z1(kte + 1)
    for k in range(1, klpbl):
        xkzom[k] = XKZMINM
        xkzoh[k] = XKZMINH

    hgamt = F(0.0)                               # :702-715
    hgamq = F(0.0)
    kpbl = 1
    hpbl = zq[1]
    zl1 = za[1]
    thermal = thvx[1]
    pblflg = True
    sfcflg = True
    sflux = hfx / rhox / CP + qfx / rhox * EP1 * thx[1]
    if br > F(0.0):
        sfcflg = False

    # ---- first guess of the PBL height (:719-749) ----
    stable = False
    brup = br
    brcr = BRCR_UB
    brdn = F(0.0)
    for k in range(2, klpbl + 1):
        if not stable:
            brdn = brup
            spdk2 = np.maximum(ux[k] * ux[k] + vx[k] * vx[k], F(1.0))
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2
            kpbl = k
            stable = bool(brup > brcr)
    k = kpbl
    if brdn >= brcr:
        brint = F(0.0)
    elif brup <= brcr:
        brint = F(1.0)
    else:
        brint = (brcr - brdn) / (brup - brdn)
    hpbl = za[k - 1] + brint * (za[k] - za[k - 1])
    if hpbl < zq[2]:
        kpbl = 1
    if kpbl <= 1:
        pblflg = False

    # ---- surface-layer scales (:751-780) ----
    fm = psim
    fh = psih
    zol1 = np.maximum(br * fm * fm / fh, RIMIN)
    if sfcflg:
        zol1 = np.minimum(zol1, -ZFMIN)
    else:
        zol1 = np.maximum(zol1, ZFMIN)
    hol1 = zol1 * hpbl / zl1 * SFCFRAC
    epshol = hol1
    if sfcflg:
        phim = _powf(F(1.0) - APHI16 * hol1, -(F(1.0) / F(4.0)))
        phih = _powf(F(1.0) - APHI16 * hol1, -(F(1.0) / F(2.0)))
        bfx0 = np.maximum(sflux, F(0.0))
        # hfx0/qfx0 (:766-767) are computed by WRF and never read; omitted.
        wstar3 = govrth * bfx0 * hpbl
        wstar = _powf(wstar3, H1)
    else:
        phim = F(1.0) + APHI5 * hol1
        phih = phim
        wstar = F(0.0)
        wstar3 = F(0.0)
    ust3 = _powf(ust, F(3.0))                    # ust**3. stays powf
    wscale = _powf(ust3 + PHIFAC * KARMAN * wstar3 * F(0.5), H1)
    wscale = np.minimum(wscale, ust * APHI16)
    wscale = np.maximum(wscale, ust / APHI5)

    # ---- countergradient terms (:785-800) ----
    if sfcflg and sflux > F(0.0):
        gamfac = BFAC / rhox / wscale
        hgamt = np.minimum(gamfac * hfx / CP, GAMCRT)
        hgamq = np.minimum(gamfac * qfx, GAMCRQ)
        vpert = (hgamt + EP1 * thx[1] * hgamq) / BFAC * AFAC
        thermal = thermal + (np.maximum(vpert, F(0.0))
                             * np.minimum(za[1] / (SFCFRAC * hpbl), F(1.0)))
        hgamt = np.maximum(hgamt, F(0.0))
        hgamq = np.maximum(hgamq, F(0.0))
        brint = F(-15.9) * ust * ust / wspd * wstar3 / _powf(wscale, F(4.0))
        hgamu = brint * ux[1]
        hgamv = brint * vx[1]
    else:
        pblflg = False

    # ---- enhance the PBL height by the thermal excess (:804-853) ----
    if pblflg:
        kpbl = 1
        hpbl = zq[1]
        stable = False
        brup = br
        brcr = BRCR_UB
    for k in range(2, klpbl + 1):
        if (not stable) and pblflg:
            brdn = brup
            spdk2 = np.maximum(ux[k] * ux[k] + vx[k] * vx[k], F(1.0))
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2
            kpbl = k
            stable = bool(brup > brcr)
    if pblflg:
        k = kpbl
        if brdn >= brcr:
            brint = F(0.0)
        elif brup <= brcr:
            brint = F(1.0)
        else:
            brint = (brcr - brdn) / (brup - brdn)
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1])
        if hpbl < zq[2]:
            kpbl = 1
        if kpbl <= 1:
            pblflg = False
        # :844-851 -- still inside the outer if(pblflg) block entered above,
        # so csfac/cslen are computed even when pblflg was just falsified.
        if wstar != 0:
            uwst = np.abs(ust / wstar - F(0.5))
            uwstx = F(-80.0) * uwst + F(14.0)
            csfac = F(0.5) * (_tanhf(uwstx) + F(3.0))
        else:
            csfac = F(1)     # dead in WRF: pblflg requires sflux>0 => wstar>0
        cslen = csfac * hpbl

    # ---- stable boundary layer (:857-912) ----
    if (not sfcflg) and hpbl < zq[2]:
        brup = br
        stable = False
    else:
        stable = True
    brcr_sbro = F(0.0)
    if (not stable) and (xland - F(1.5)) >= 0:   # :867-874
        wspd10 = u10 * u10 + v10 * v10
        wspd10 = _sqrtf(wspd10)
        ross = wspd10 / (CORI * znt)
        brcr_sbro = np.minimum(F(0.16) * _powf(F(1.0e-7) * ross, F(-0.18)),
                               F(0.3))
    if not stable:                               # :876-884
        if (xland - F(1.5)) >= 0:
            brcr = brcr_sbro
        else:
            brcr = BRCR_SB
    for k in range(2, klpbl + 1):                # :886-896
        if not stable:
            brdn = brup
            spdk2 = np.maximum(ux[k] * ux[k] + vx[k] * vx[k], F(1.0))
            brup = (thvx[k] - thermal) * (G * za[k] / thvx[1]) / spdk2
            kpbl = k
            stable = bool(brup > brcr)
    if (not sfcflg) and hpbl < zq[2]:            # :898-912
        k = kpbl
        if brdn >= brcr:
            brint = F(0.0)
        elif brup <= brcr:
            brint = F(1.0)
        else:
            brint = (brcr - brdn) / (brup - brdn)
        hpbl = za[k - 1] + brint * (za[k] - za[k - 1])
        if hpbl < zq[2]:
            kpbl = 1
        if kpbl <= 1:
            pblflg = False

    # ---- scale dependency, nonlocal momentum and moisture (:916-926) ----
    delxy = _sqrtf(dx * dy)
    pu1 = np_pu(delxy, cslen)                    # cslen == 0 columns take the
    pq1 = np_pq(delxy, cslen)                    # h == 0 early-out and get 1.
    if pblflg:
        hgamu = hgamu * pu1
        hgamv = hgamv * pu1
        hgamq = hgamq * pq1

    # ---- entrainment parameters (:930-985) ----
    prpbl = F(0.0)
    wm2 = F(0.0)
    we = F(0.0)
    bfxpbl = F(0.0)                              # :629-637
    hfxpbl = F(0.0)
    qfxpbl = F(0.0)
    ufxpbl = F(0.0)
    vfxpbl = F(0.0)
    dthvx = F(0.0)
    if pblflg:
        k = kpbl - 1
        prpbl = F(1.0)
        wm3 = wstar3 + F(5.0) * ust3
        wm2 = _powf(wm3, H2)
        bfxpbl = F(-0.15) * thvx[1] / G * wm3 / hpbl
        dthvx = np.maximum(thvx[k + 1] - thvx[k], TMIN)
        dthx = np.maximum(thx[k + 1] - thx[k], TMIN)
        dqx = np.minimum(qvx[k + 1] - qvx[k], F(0.0))
        we = np.maximum(bfxpbl / dthvx, -_sqrtf(wm2))
        hfxpbl = we * dthx                       # :943 -- overwritten at :1114;
        pq1 = np_pq(delxy, cslen)                # ported as written (see there)
        qfxpbl = we * dqx * pq1
        pu1 = np_pu(delxy, cslen)
        dux = ux[k + 1] - ux[k]
        dvx = vx[k + 1] - vx[k]
        if dux > TMIN:
            ufxpbl = np.maximum(prpbl * we * dux * pu1, -(ust * ust))
        elif dux < -TMIN:
            ufxpbl = np.minimum(prpbl * we * dux * pu1, ust * ust)
        else:
            ufxpbl = F(0.0)
        if dvx > TMIN:
            vfxpbl = np.maximum(prpbl * we * dvx * pu1, -(ust * ust))
        elif dvx < -TMIN:
            vfxpbl = np.minimum(prpbl * we * dvx * pu1, ust * ust)
        else:
            vfxpbl = F(0.0)
        delb = govrth * D3 * hpbl
        delta = np.minimum(D1 * hpbl + D2 * wm2 / delb, F(100.0))
        delb = govrth * dthvx
        deltaoh = D1 * hpbl + D2 * wm2 / delb
        deltaoh = np.maximum(EZFAC * deltaoh, hpbl - za[kpbl - 1] - F(1.0))
        deltaoh = np.minimum(deltaoh, hpbl)
        if (dux != 0) or (dvx != 0):
            # dux**2. and dvx**2. fold to multiplies (pow-probe, negative
            # bases included).
            rigs = govrth * dthvx * deltaoh / (dux * dux + dvx * dvx)
        else:
            rigs = RIGSMAX
        rigs = np.maximum(np.minimum(rigs, RIGSMAX), RIMIN)
        if (rigs > 0) and (np.abs(rigs + CPENT) <= F(1.0e-6)):
            cenlfrac = ENTFMAX
        else:
            cenlfrac = rigs / (rigs + CPENT)
        cenlfrac = np.minimum(cenlfrac, ENTFMAX)
        enlfrac2 = np.maximum(wm3 / wstar3 * cenlfrac, ENTFMIN)
        enlfrac2 = enlfrac2 * ENLFRAC

    entfacmf = _z1(kte + 1)                      # :987-998
    entfac = _z1(kte + 1)
    for k in range(1, klpbl + 1):
        if pblflg:
            tmf = (zq[k + 1] - hpbl) / deltaoh
            entfacmf[k] = _sqrtf(tmf * tmf)      # sqrt((...)**2.), **2. folds
        if pblflg and k >= kpbl:
            te = (zq[k + 1] - hpbl) / deltaoh
            entfac[k] = te * te
        else:
            entfac[k] = F(1.0e30)

    # ---- diffusion coefficients below the PBL top (:1002-1036) ----
    xkzm = _z1(kte + 1)
    xkzh = _z1(kte + 1)
    xkzq = _z1(kte + 1)
    xkzml = _z1(kte + 1)
    xkzhl = _z1(kte + 1)
    zfacent = _z1(kte + 1)
    for k in range(1, klpbl + 1):
        if k < kpbl:
            zfac = np.minimum(np.maximum(
                F(1.0) - (zq[k + 1] - zl1) / (hpbl - zl1), ZFMIN), F(1.0))
            zfacent[k] = _powf(F(1.0) - zfac, F(3.0))   # (1.-zfac)**3. -- powf
            wscalek = _powf(
                ust3 + PHIFAC * KARMAN * wstar3 * (F(1.0) - zfac), H1)
            if sfcflg:
                prfac = conpr
                # :1010 -- WRF's own 0/0 when both cubes are exactly zero
                # (fixture case 13); the NaN column downstream is the point.
                prfac2 = (F(15.9) * wstar3 / ust3
                          / (F(1.0) + F(4.0) * KARMAN * wstar3 / ust3))
                m = np.maximum(zq[k + 1] - SFCFRAC * hpbl, F(0.0))
                prnumfac = F(-3.0) * (m * m) / (hpbl * hpbl)   # **2. folds
            else:
                prfac = F(0.0)
                prfac2 = F(0.0)
                prnumfac = F(0.0)
                phim8z = F(1.0) + APHI5 * zol1 * zq[k + 1] / zl1
                wscalek = ust / phim8z
                wscalek = np.maximum(wscalek, F(0.001))
            prnum0 = phih / phim + prfac
            prnum0 = np.maximum(np.minimum(prnum0, PRMAX), PRMIN)
            xkzm[k] = wscalek * KARMAN * zq[k + 1] * (zfac * zfac)  # **pfac folds
            prnum = F(1.0) + (prnum0 - F(1.0)) * _expf(prnumfac)
            # zfac**(pfac_q-pfac) == zfac**0. == 1.0 for every zfac
            # (pow-probe.txt, including 0); the multiply is kept.
            xkzq[k] = xkzm[k] / prnum * _powf(zfac, PFAC_Q - PFAC)
            prnum0 = prnum0 / (F(1.0) + prfac2 * KARMAN * SFCFRAC)
            prnum = F(1.0) + (prnum0 - F(1.0)) * _expf(prnumfac)
            xkzh[k] = xkzm[k] / prnum
            xkzm[k] = xkzm[k] + xkzom[k]
            xkzh[k] = xkzh[k] + xkzoh[k]
            xkzq[k] = xkzq[k] + xkzoh[k]
            xkzm[k] = np.minimum(xkzm[k], XKZMAX)
            xkzh[k] = np.minimum(xkzh[k], XKZMAX)
            xkzq[k] = np.minimum(xkzq[k], XKZMAX)

    # ---- free atmosphere (:1040-1086) ----
    for k in range(1, kte):
        if k >= kpbl:
            du = ux[k + 1] - ux[k]
            dv = vx[k + 1] - vx[k]
            ss = (du * du + dv * dv) / (dza[k + 1] * dza[k + 1]) + F(1.0e-9)
            govrthv = G / (F(0.5) * (thvx[k + 1] + thvx[k]))
            ri = govrthv * (thvx[k + 1] - thvx[k]) / (ss * dza[k + 1])
            # :1049-1058 in cloud (imvdif == 1, nwmass == 3 always):
            # qx(i,kqc+k-1) is qc(k) and qx(i,kqi+k-1) is qi(k) in the
            # ndiff-packed array (kqc = 1+kte, kqi = 1+2*kte).
            if (qcx[k] + qix[k]) > F(0.01e-3) and \
                    (qcx[k + 1] + qix[k + 1]) > F(0.01e-3):
                qmean = F(0.5) * (qvx[k] + qvx[k + 1])
                tmean = F(0.5) * (tx[k] + tx[k + 1])
                alpha = XLV * qmean / RD / tmean
                chi = XLV * XLV * qmean / CP / RV / tmean / tmean
                ri = (F(1.0) + alpha) * (
                    ri - G * G / ss / tmean / CP
                    * ((chi - alpha) / (F(1.0) + chi)))
            zk = KARMAN * zq[k + 1]
            rlamdz = np.minimum(np.maximum(F(0.1) * dza[k + 1], RLAM), F(300.0))
            rlamdz = np.minimum(dza[k + 1], rlamdz)
            tt = zk * rlamdz / (rlamdz + zk)
            rl2 = tt * tt                        # integer **2 -> multiply
            dk = rl2 * _sqrtf(ss)
            if ri < F(0.0):
                ri = np.maximum(ri, RIMIN)
                sri = _sqrtf(-ri)
                xkzm[k] = dk * (F(1.0) + F(8.0) * (-ri) / (F(1.0) + F(1.746) * sri))
                xkzh[k] = dk * (F(1.0) + F(8.0) * (-ri) / (F(1.0) + F(1.286) * sri))
            else:
                tt = F(1.0) + F(5.0) * ri
                xkzh[k] = dk / (tt * tt)         # integer **2 -> multiply
                prnum = F(1.0) + F(2.1) * ri
                prnum = np.minimum(prnum, PRMAX)
                xkzm[k] = xkzh[k] * prnum
            xkzm[k] = xkzm[k] + xkzom[k]
            xkzh[k] = xkzh[k] + xkzoh[k]
            xkzm[k] = np.minimum(xkzm[k], XKZMAX)
            xkzh[k] = np.minimum(xkzh[k], XKZMAX)
            xkzml[k] = xkzm[k]
            xkzhl[k] = xkzh[k]

    # ---- prescribed nonlocal heat transport (:1090-1130) ----
    deltaoh = deltaoh / hpbl                     # :1091, every column
    delxy = _sqrtf(dx * dy)
    mlfrac = MLTOP - deltaoh
    # ezfrac (:1097) is computed by WRF and never read; omitted.
    # :1098 -- zfacmf(i,1) is first computed CLAMPED and feeds sfcfracn; the
    # k-loop below then RE-computes it UNCLAMPED (:1119), overwriting index 1,
    # and the mf profile uses the unclamped value.
    zfacmf1 = np.minimum(np.maximum(zq[2] / hpbl, ZFMIN), F(1.0))
    sfcfracn = np.maximum(SFCFRACN1, zfacmf1)
    sflux0 = (A11 + A12 * sfcfracn) * sflux
    snlflux0 = NLFRAC * sflux0
    amf1 = snlflux0 / sfcfracn
    amf2 = F(0.0)
    bmf2 = F(0.0)
    if pblflg:
        amf2 = -(snlflux0 / (mlfrac - sfcfracn))
        bmf2 = -(mlfrac * amf2)
    if (deltaoh == 0) and (enlfrac2 == 0):       # :1108, exact compares
        amf3 = F(0.0)
    else:
        amf3 = snlflux0 * enlfrac2 / deltaoh
    bmf3 = -(amf3 * mlfrac)
    # :1114-1116 -- hfxpbl is OVERWRITTEN here (the :943 we*dthx value is
    # discarded) and scaled by pthnl once ...
    hfxpbl = amf3 + bmf3
    pth1 = np_pthnl(delxy, cslen)
    hfxpbl = hfxpbl * pth1
    for k in range(1, klpbl + 1):
        zfacmf[k] = np.maximum(zq[k + 1] / hpbl, ZFMIN)   # :1119, unclamped
        if pblflg and k < kpbl:
            if zfacmf[k] <= sfcfracn:
                mf[k] = amf1 * zfacmf[k]
            elif zfacmf[k] <= mlfrac:
                mf[k] = amf2 * zfacmf[k] + bmf2
            # ... and the WHOLE mf -- entrainment term included -- is scaled
            # by pth1 AGAIN at :1127, so the entrainment piece is
            # double-scaled.  WRF's arithmetic, ported as written.
            mf[k] = mf[k] + hfxpbl * _expf(-entfacmf[k])
            mf[k] = mf[k] * pth1

    # ---- heat: matrix assembly (:1134-1187), solve, recover ----
    au = _z1(kte + 1)
    al = _z1(kte + 1)
    ad = _z1(kte + 1)
    f1 = _z1(kte + 1)
    ad[1] = F(1.0)
    f1[1] = thx[1] - F(300.0) + hfx / cont / delp[1] * dt2
    exch_hx = _z1(kte + 1)   # exch_h(:,1) is never written by the scheme
    delxy = _sqrtf(dx * dy)
    for k in range(1, kte):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p2dx[k] - p2dx[k + 1]
        rdz = F(1.0) / dza[k + 1]
        tem1 = dsig * xkzh[k] * rdz
        if pblflg and k < kpbl:
            dsdzt = tem1 * (-(mf[k] / xkzh[k]))
            f1[k] = f1[k] + dtodsd * dsdzt
            f1[k + 1] = thx[k + 1] - F(300.0) - dtodsu * dsdzt
        elif pblflg and k >= kpbl and entfac[k] < F(4.6):
            # :1161-1164 -- the entrainment-zone overwrite of xkzh ...
            xkzh[k] = -(we * dza[kpbl] * _expf(-entfac[k]))
            xkzh[k] = _sqrtf(xkzh[k] * xkzhl[k])
            xkzh[k] = np.maximum(xkzh[k], xkzoh[k])
            xkzh[k] = np.minimum(xkzh[k], XKZMAX)
            f1[k + 1] = thx[k + 1] - F(300.0)
        else:
            f1[k + 1] = thx[k + 1] - F(300.0)
        # :1169 -- ... after which tem1 is RECOMPUTED from the updated
        # coefficient before au/al.  Sequence kept exactly.
        tem1 = dsig * xkzh[k] * rdz
        dsdz2 = tem1 * rdz
        au[k] = -(dtodsd * dsdz2)
        al[k] = -(dtodsu * dsdz2)
        # :1176-1178 -- LOCAL heat partition: h = hpbl (not cslen) and delxy
        # scaled by zfacdx; the NONLOCAL calls above use h = cslen, plain delxy.
        zfacdx = F(0.2) * hpbl / zq[k + 1]
        delxy = _sqrtf(dx * dy) * np.maximum(zfacdx, F(1.0))
        pth1 = np_pthl(delxy, hpbl)
        if pblflg and k < kpbl:
            au[k] = au[k] * pth1
            al[k] = al[k] * pth1
        ad[k] = ad[k] - au[k]
        ad[k + 1] = F(1.0) - al[k]
        exch_hx[k + 1] = xkzh[k]                 # :1185, after the overwrite
    f1 = _tridi_shared(al, ad, au.copy(), [f1.copy()], kte)[0]
    for k in range(kte, 0, -1):                  # :1202-1213
        ttend = (f1[k] - thx[k] + F(300.0)) * rdt * pi2dx[k]
        ttnp[k] = ttnp[k] + ttend                # 0. + ttend (:612)
        if k == kte:
            tflux_e[k] = ttend * dz8wx[k]
        else:
            tflux_e[k] = tflux_e[k + 1] + ttend * dz8wx[k]

    # ---- moisture, clouds, ice: assembly (:1217-1320), solve, recover ----
    au = _z1(kte + 1)
    al = _z1(kte + 1)
    ad = _z1(kte + 1)
    f3 = np.zeros((4, kte + 1), dtype=np.float32)
    ad[1] = F(1.0)
    f3[1][1] = qvx[1] + qfx * G / delp[1] * dt2
    f3[2][1] = qcx[1]                            # :1238-1245, ic = 2, 3
    f3[3][1] = qix[1]
    for k in range(1, kte):                      # :1247-1253
        if k >= kpbl:
            xkzq[k] = xkzh[k]
    for k in range(1, kte):                      # :1255-1293
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p2dx[k] - p2dx[k + 1]
        rdz = F(1.0) / dza[k + 1]
        tem1 = dsig * xkzq[k] * rdz
        if pblflg and k < kpbl:
            dsdzq = tem1 * (-(qfxpbl * zfacent[k] / xkzq[k]))
            f3[1][k] = f3[1][k] + dtodsd * dsdzq
            f3[1][k + 1] = qvx[k + 1] - dtodsu * dsdzq
        elif pblflg and k >= kpbl and entfac[k] < F(4.6):
            xkzq[k] = -(we * dza[kpbl] * _expf(-entfac[k]))
            xkzq[k] = _sqrtf(xkzq[k] * xkzhl[k])
            xkzq[k] = np.maximum(xkzq[k], xkzoh[k])
            xkzq[k] = np.minimum(xkzq[k], XKZMAX)
            f3[1][k + 1] = qvx[k + 1]
        else:
            f3[1][k + 1] = qvx[k + 1]
        tem1 = dsig * xkzq[k] * rdz              # recompute, as at :1275
        dsdz2 = tem1 * rdz
        au[k] = -(dtodsd * dsdz2)
        al[k] = -(dtodsu * dsdz2)
        zfacdx = F(0.2) * hpbl / zq[k + 1]       # :1282-1284, h = hpbl
        delxy = _sqrtf(dx * dy) * np.maximum(zfacdx, F(1.0))
        pq1 = np_pq(delxy, hpbl)
        if pblflg and k < kpbl:
            au[k] = au[k] * pq1
            al[k] = al[k] * pq1
        ad[k] = ad[k] - au[k]
        ad[k + 1] = F(1.0) - al[k]
    for k in range(1, kte):                      # :1295-1304
        f3[2][k + 1] = qcx[k + 1]
        f3[3][k + 1] = qix[k + 1]
    # :1324 -- one tridin_ysu call, ndiff = 3: all three components share the
    # same au/al/ad built from xkzq.
    solved = _tridi_shared(al, ad, au.copy(),
                           [f3[1].copy(), f3[2].copy(), f3[3].copy()], kte)
    for k in range(kte, 0, -1):                  # :1328-1340
        qtend = (solved[0][k] - qvx[k]) * rdt
        qtnp[1][k] = qtnp[1][k] + qtend
        if k == kte:
            qflux_e[k] = qtend * dz8wx[k]
        else:
            qflux_e[k] = qflux_e[k + 1] + qtend * dz8wx[k]
        tvflux_e[k] = tflux_e[k] + qflux_e[k] * EP1 * thx[k]
    for k in range(1, kte + 1):                  # :1342-1356
        if pblflg and k < kpbl:
            hgame_c = (C_1 * F(0.2) * F(2.5) * (G / thvx[k]) * wstar
                       / (F(0.25) * (q2x[k + 1] + q2x[k])))
            hgame_c = np.minimum(hgame_c, GAMCRE)
            # the k == kte arm (:1347) is unreachable: k < kpbl <= kte
            hgame2d[k] = hgame_c * F(0.5) * (tvflux_e[k] + tvflux_e[k + 1]) * hpbl
            hgame2d[k] = np.maximum(hgame2d[k], F(0.0))
    for ic in (2, 3):                            # :1358-1370
        for k in range(kte, 0, -1):
            qsrc = qcx if ic == 2 else qix
            qtend = (solved[ic - 1][k] - qsrc[k]) * rdt
            qtnp[ic][k] = qtnp[ic][k] + qtend

    # ---- momentum: assembly (:1374-1436), solve, recover ----
    au = _z1(kte + 1)
    al = _z1(kte + 1)
    ad = _z1(kte + 1)
    f1m = _z1(kte + 1)
    f2m = _z1(kte + 1)
    s = wspd1 / wspd
    ad[1] = (F(1.0) + ctopo * (ust * ust) / wspd1 * rhox * G / delp[1] * dt2
             * (s * s))                          # :1387, both arms run this
    f1m[1] = ux[1]
    f2m[1] = vx[1]
    delxy = _sqrtf(dx * dy)
    for k in range(1, kte):
        dtodsd = dt2 / delp[k]
        dtodsu = dt2 / delp[k + 1]
        dsig = p2dx[k] - p2dx[k + 1]
        rdz = F(1.0) / dza[k + 1]
        tem1 = dsig * xkzm[k] * rdz
        if pblflg and k < kpbl:
            dsdzu = tem1 * (-(hgamu / hpbl) - ufxpbl * zfacent[k] / xkzm[k])
            dsdzv = tem1 * (-(hgamv / hpbl) - vfxpbl * zfacent[k] / xkzm[k])
            f1m[k] = f1m[k] + dtodsd * dsdzu
            f1m[k + 1] = ux[k + 1] - dtodsu * dsdzu
            f2m[k] = f2m[k] + dtodsd * dsdzv
            f2m[k + 1] = vx[k + 1] - dtodsu * dsdzv
        elif pblflg and k >= kpbl and entfac[k] < F(4.6):
            xkzm[k] = prpbl * xkzh[k]
            xkzm[k] = _sqrtf(xkzm[k] * xkzml[k])
            xkzm[k] = np.maximum(xkzm[k], xkzom[k])
            xkzm[k] = np.minimum(xkzm[k], XKZMAX)
            f1m[k + 1] = ux[k + 1]
            f2m[k + 1] = vx[k + 1]
        else:
            f1m[k + 1] = ux[k + 1]
            f2m[k + 1] = vx[k + 1]
        tem1 = dsig * xkzm[k] * rdz              # recompute, as at :1419
        dsdz2 = tem1 * rdz
        au[k] = -(dtodsd * dsdz2)
        al[k] = -(dtodsu * dsdz2)
        zfacdx = F(0.2) * hpbl / zq[k + 1]       # :1426-1428, h = hpbl
        delxy = _sqrtf(dx * dy) * np.maximum(zfacdx, F(1.0))
        pu1 = np_pu(delxy, hpbl)
        if pblflg and k < kpbl:
            au[k] = au[k] * pu1
            al[k] = al[k] * pu1
        ad[k] = ad[k] - au[k]
        ad[k + 1] = F(1.0) - al[k]
    f1m, f2m = _tridi_shared(al, ad, au.copy(),
                             [f1m.copy(), f2m.copy()], kte)   # tridi1n :1450
    for k in range(kte, 0, -1):                  # :1454-1463
        utend = (f1m[k] - ux[k]) * rdt
        vtend = (f2m[k] - vx[k]) * rdt
        utnp[k] = utnp[k] + utend
        vtnp[k] = vtnp[k] + vtend

    # ---- 10 m wind blend (:1471-1474): runs for BOTH arms; with ctopo2 = 1
    # the arithmetic is still performed (u10*1 + (1-1)*ux).
    u10_out = ctopo2 * u10 + (F(1.0) - ctopo2) * ux[1]
    v10_out = ctopo2 * v10 + (F(1.0) - ctopo2) * vx[1]

    # ---- SGS TKE diagnostics (:1478-1608) ----
    tke_out = tkex[1:kte + 1].copy()             # tke untouched when diag off
    if tke_diag == 1:
        ptke1 = np.ones(kte + 1, dtype=np.float32)   # :1509
        for k in range(1, kte):                  # :1530-1536
            if pblflg and k <= kpbl:             # k .le. kpbl, and za(k), not
                zfacdx = F(0.2) * hpbl / za[k]   # zq -- unlike the tri loops
                delxy = _sqrtf(dx * dy) * np.maximum(zfacdx, F(1.0))
                ptke1[k + 1] = np_ptke(delxy, hpbl)
        akmk = _z1(kte + 1)
        akhk = _z1(kte + 1)
        mfk = _z1(kte + 1)
        ufxpblk = _z1(kte + 1)
        vfxpblk = _z1(kte + 1)
        qfxpblk = _z1(kte + 1)
        for k in range(2, kte + 1):              # :1546-1553
            akmk[k] = xkzm[k - 1]
            akhk[k] = xkzh[k - 1]
            mfk[k] = mf[k - 1] / xkzh[k - 1]
            # WRF reads zfacent at k-1 >= kpbl here, where it never assigned
            # it (stack garbage); every consumer is guarded k <= kpbl so the
            # garbage is dead.  This port's zfacent is 0 there instead.
            ufxpblk[k] = ufxpbl * zfacent[k - 1] / xkzm[k - 1]
            vfxpblk[k] = vfxpbl * zfacent[k - 1] / xkzm[k - 1]
            qfxpblk[k] = qfxpbl * zfacent[k - 1] / xkzq[k - 1]
        q2xk = q2x.copy()
        if pblflg and kpbl < kte:
            # :1555-1559.  DOCUMENTED DIVERGENCE: WRF reads q2xk(kpbl+1) with
            # no guard, which is one past the end when kpbl == kte (:1557).
            # The port computes efxpbl only for kpbl < kte -- the defined
            # behaviour -- and the registry warning covers it (Phase D).  The
            # fixture never reaches the undefined arm (case 22 runs diag off).
            k = kpbl - 1
            dex = F(0.25) * (q2xk[k + 2] - q2xk[k])
            efxpbl = we * dex
        delxy = _sqrtf(dx * dy)                  # :1561
        s2k, ghk, rigk, elk = _mixlen(
            ux, vx, tx, thx, qvx, qcx, q2xk, zq, ust, corf, epshol,
            hpbl, kpbl, pblflg, hgamu, hgamv, hgamq,
            mfk, ufxpblk, vfxpblk, qfxpblk, kte)
        _prodq2(dt, ust, s2k, rigk, q2xk, elk, zq, akmk, akhk,
                ux, vx, thx, thvx, hgamu, hgamv, hgamq, delxy,
                hpbl, pblflg, kpbl, mfk, ufxpblk, vfxpblk, qfxpblk, kte)
        _vdifq(dt, q2xk, elk, zq, akhk, ptke1, hgame2d, hpbl, pblflg, kpbl,
               efxpbl, kte)
        for k in range(1, kte + 1):              # :1601-1605
            q2x[k] = np.maximum(q2xk[k], EPSQ2L)   # amax1: NaN propagates
            tke_out[k - 1] = F(0.5) * q2x[k]
            if k != 1:
                el_pbl[k] = elk[k]               # el(kts) stays the :673 zero

    # ---- the wrapper's own epilogue (:251-258) ----
    rthblten = np.empty(kte, dtype=np.float32)
    for k in range(1, kte + 1):
        # :253 -- ttnp is (f1-thx+300)*rdt*pi2d (:1204); dividing by pi3d
        # here does NOT round-trip: (x*pi)/pi is not the float32 identity.
        rthblten[k - 1] = ttnp[k] / pi2dx[k]

    return {
        "rublten": utnp[1:kte + 1].copy(),
        "rvblten": vtnp[1:kte + 1].copy(),
        "rthblten": rthblten,
        "rqvblten": qtnp[1][1:kte + 1].copy(),
        "rqcblten": qtnp[2][1:kte + 1].copy(),
        "rqiblten": qtnp[3][1:kte + 1].copy(),
        "exch_h": exch_hx[1:kte + 1].copy(),
        "tke": tke_out,
        "el": el_pbl[1:kte + 1].copy(),
        "hpbl": F(hpbl),
        "kpbl": int(kpbl),
        "wstar": F(wstar),
        "delta": F(delta),
        "u10": F(u10_out),
        "v10": F(v10_out),
        # Diagnostic, not a WRF output: the entrainment TKE flux the
        # kpbl < kte guard protects (:1555-1559).  Zero when the guard held
        # the arm off (or the flux is genuinely zero); the parity gate uses
        # it to prove the guard discriminates rather than being dead code.
        "efxpbl": F(efxpbl),
    }

