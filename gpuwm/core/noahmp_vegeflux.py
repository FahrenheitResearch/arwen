"""FP32 transcription of the Noah-MP VEGE_FLUX subtree from WRF v4.6.1.

Transcribes, operation for operation, from the pinned
``phys/module_sf_noahmplsm.F`` (WRF commit d66e442fccc04111067e29274c9f9eaccc3cef28,
sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282):

  ``ESAT``      saturation vapour pressure polynomials
  ``RAGRB``     under-canopy aerodynamic and leaf boundary-layer resistances
  ``SFCDIF1``   Monin-Obukhov surface exchange coefficients
  ``STOMATA``   Ball-Berry stomatal resistance / photosynthesis
  ``VEGE_FLUX`` the vegetated surface energy balance (NITERC=20, NITERG=5)

Numerical contract
------------------
``kind_phys == kind(1.0)`` is FP32.  Every arithmetic boundary here is FP32:
:class:`R4` rounds to binary32 after every ``+ - * /`` and converts every
Python literal to binary32 *before* the operation, because that is what
gfortran does with a default-kind real literal.  Double rounding through
binary64 is innocuous for ``+ - * /`` and ``sqrt`` on binary32 operands
(53 >= 2*24 + 2), so the intermediate Python float carries no extra precision.

Transcendentals go through :mod:`gpuwm.core.noahmp_libm`, which reproduces
glibc 2.39's ``logf``/``expf``/``powf``/``atanf`` bit for bit.  ``numpy``'s
float32 transcendentals and "FP64 then round once" are both *different
functions* and cannot hold a max_ulp 0 gate.  This lane originally carried its
own ``logf``/``atanf``/``sqrtf`` in ``noahmp_vegeflux_libm``; that module was
folded into ``noahmp_libm`` after the two transcriptions were shown to agree
bit for bit on 3,875,000 arguments and on every pinned glibc word in
``gpuwm/data/noahmp/oracle/``.

Integer powers follow gfortran's expansion, which is not associativity-free:
``x**3`` becomes ``x*(x*x)`` and ``x**4`` becomes ``(x*x)*(x*x)``.

Pinned option identity (WRF Registry defaults) and what it kills
----------------------------------------------------------------
``opt_sfc=1``   -> SFCDIF2 is dead; only SFCDIF1 is reachable.
``opt_crs=1``   -> CANRES is dead; only STOMATA is reachable.
``opt_crop=0``  -> the whole gecros branch is dead, including the alternate
                   ``CTW`` form and the effective-LAI rescaling.
``opt_stc=1``   -> of the snow/TG reset block only the ``TG = TFRZ`` leg is
                   reachable; the ``opt_stc=3`` FSNO-weighted leg is dead.
``dveg=4``      -> CARBON/CO2FLUX never runs (not called from here anyway).
Those branches are asserted off rather than transcribed; calling with a
different identity raises.

Validated against ``gpuwm/data/noahmp/oracle/noahmp-vegeflux.csv`` at max_ulp 0
by ``tests/test_noahmp_vegeflux.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from gpuwm.core.noahmp_libm import (F32_PACK, F32_UNPACK, atanf, expf, f32,
                                    logf, powf, sqrtf)

__all__ = [
    "R4", "VegeFluxParameters", "esat", "ragrb", "sfcdif1", "stomata",
    "vege_flux", "GRAV", "SB", "VKC", "TFRZ", "CPAIR", "CWAT", "CICE",
    "DENH2O", "DENICE",
]


# --------------------------------------------------------------------------
# FP32 arithmetic carrier
# --------------------------------------------------------------------------
class R4(float):
    """A binary32 value.  Every arithmetic operation rounds to binary32.

    Subclasses ``float`` so comparisons, ``min``/``max`` and formatting work,
    but every operator is overridden so no binary64 result can leak into the
    next operation.  The right-hand operand is converted to binary32 *first*,
    which is what gfortran does with a default-kind real literal: writing
    ``x * 0.622`` in Fortran multiplies by ``f32(0.622)``, not by the binary64
    nearest to 0.622.

    Why the operators are written this flatly
    -----------------------------------------
    This is the hottest type in the Noah-MP host column.  Measured on the
    reference box with ``cProfile`` over one 48-land-column LSM step:
    ``__new__`` is 337,728 calls (7,036 per land column) and ``f32`` is
    282,634 (5,888 per column), and the two of them plus ``_struct.pack`` /
    ``_struct.unpack`` are 56% of the column's own time.  Nothing else in the
    file is within an order of magnitude, so the constant factor of an
    operator IS the scheme's host cost.

    Three spellings, all bit-identical, measured per operation at 500,000
    repetitions:

    ==========================================  ======  =======  ==========
    spelling                                     ``a*b``  ``a*b+a``  ``a*L``
    ==========================================  ======  =======  ==========
    ``R4(float.__mul__(self, float(R4(o))))``    ~830 ns  1666 ns    1224 ns
    ... rounding ``o`` in place, one ``f32``      480 ns   997 ns     674 ns
    ... ``f32`` inlined as pack/unpack (this)     411 ns   852 ns     516 ns
    ==========================================  ======  =======  ==========

    The first spelling costs three Python calls and two ``__new__`` bodies per
    operation because it re-enters ``__new__`` to round the right-hand side
    and again to wrap the result.  The second removes both re-entries.  The
    third removes the ``f32`` call frame as well: ``_unpack(_pack(v))[0]`` is
    character for character the body of
    :func:`gpuwm.core.noahmp_libm.f32`, which publishes its two halves as
    :data:`~gpuwm.core.noahmp_libm.F32_PACK` /
    :data:`~gpuwm.core.noahmp_libm.F32_UNPACK` for exactly this caller.  It is
    not a cheaper rounding -- it is the same rounding without the frame, so
    the overflow guard (an argument outside binary32 raises ``OverflowError``
    rather than saturating) is inherited unchanged.

    The rounding ORDER is unchanged, and that is the part that must not move:
    the right-hand operand is still rounded to binary32 *before* the
    operation.  ``tests/test_noahmp_vegeflux.py``'s
    ``test_the_fast_r4_operators_are_the_three_call_spelling`` sweeps all
    427,700 ``(op, lhs, rhs)`` triples built from a 329-value corpus -- both
    zeros, the smallest and largest subnormals, the subnormal/normal boundary
    from both sides, values one ULP either side of a power of two, exact
    halfway cases between adjacent binary32 values, the binary32 extremes,
    four binary64 values that overflow binary32 (which is why 325 of the 329
    can be a left operand and all 329 can be a right one), both infinities, a
    NaN, and a seeded spread across every binary32 exponent including the
    subnormal band -- against the three-call spelling, and requires 0
    mismatches.  It is not a gate that has never failed: replacing ``__mul__``
    with one that leaves the right operand unrounded is caught on 4,502 of the
    427,700, and a companion test proves the same sweep catches a carrier that
    flushes subnormals -- the ``-ftz`` failure this port has already hit twice
    on the device.  ``_pack``, ``_unpack`` and ``_new`` are default arguments
    rather than globals so the body does no name lookup outside its own frame.
    """

    __slots__ = ()

    def __new__(cls, value, _new=float.__new__, _pack=F32_PACK,
                _unpack=F32_UNPACK):
        if type(value) is cls:
            return value
        return _new(cls, _unpack(_pack(value))[0])

    # -- binary operators ---------------------------------------------------
    # Each rounds `other` to binary32 first (the gfortran literal rule), then
    # rounds the binary64 result once.  `type(other) is R4` means it is
    # already binary32, so the rounding is skipped rather than repeated.
    def __add__(self, other, _new=float.__new__, _pack=F32_PACK,
                _unpack=F32_UNPACK, _op=float.__add__):
        if type(other) is not R4:
            other = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(_op(self, other)))[0])

    def __sub__(self, other, _new=float.__new__, _pack=F32_PACK,
                _unpack=F32_UNPACK, _op=float.__sub__):
        if type(other) is not R4:
            other = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(_op(self, other)))[0])

    def __mul__(self, other, _new=float.__new__, _pack=F32_PACK,
                _unpack=F32_UNPACK, _op=float.__mul__):
        if type(other) is not R4:
            other = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(_op(self, other)))[0])

    def __truediv__(self, other, _new=float.__new__, _pack=F32_PACK,
                    _unpack=F32_UNPACK, _op=float.__truediv__):
        if type(other) is not R4:
            other = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(_op(self, other)))[0])

    # The reflected forms are rare -- none of them appears in the 48-column
    # profile at all -- so they keep the readable spelling and just drop the
    # two `f32` frames.
    def __radd__(self, other, _new=float.__new__, _pack=F32_PACK,
                 _unpack=F32_UNPACK):
        left = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(left + float(self)))[0])

    def __rsub__(self, other, _new=float.__new__, _pack=F32_PACK,
                 _unpack=F32_UNPACK):
        left = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(left - float(self)))[0])

    def __rmul__(self, other, _new=float.__new__, _pack=F32_PACK,
                 _unpack=F32_UNPACK):
        left = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(left * float(self)))[0])

    def __rtruediv__(self, other, _new=float.__new__, _pack=F32_PACK,
                     _unpack=F32_UNPACK):
        left = _unpack(_pack(other))[0]
        return _new(R4, _unpack(_pack(left / float(self)))[0])

    # -0.0 and abs() are exact in binary32, so neither needs a rounding.
    def __neg__(self, _new=float.__new__):
        return _new(R4, float.__neg__(self))

    def __pos__(self):
        return self

    def __abs__(self, _new=float.__new__):
        return _new(R4, float.__abs__(self))

    def __repr__(self):
        return f"R4({float(self)!r})"


def _mn(*args) -> R4:
    """Fortran ``MIN`` over binary32 values."""
    return R4(min(float(R4(a)) for a in args))


def _mx(*args) -> R4:
    """Fortran ``MAX`` over binary32 values."""
    return R4(max(float(R4(a)) for a in args))


def _p3(x: R4) -> R4:
    """gfortran's expansion of ``x**3``: ``x*(x*x)``."""
    return x * (x * x)


def _p4(x: R4) -> R4:
    """gfortran's expansion of ``x**4``: ``(x*x)*(x*x)``."""
    xx = x * x
    return xx * xx


def _log(x: R4) -> R4:
    return R4(logf(float(x)))


def _exp(x: R4) -> R4:
    return R4(expf(float(x)))


def _pow(x: R4, y) -> R4:
    return R4(powf(float(x), float(R4(y))))


def _sqrt(x: R4) -> R4:
    return R4(sqrtf(float(x)))


def _atan(x: R4) -> R4:
    return R4(atanf(float(x)))


# --------------------------------------------------------------------------
# module constants (module_sf_noahmplsm.F lines 204-220)
# --------------------------------------------------------------------------
GRAV = R4(9.80616)
SB = R4(5.67E-08)
VKC = R4(0.40)
TFRZ = R4(273.16)
CPAIR = R4(1004.64)
CWAT = R4(4.188E06)
CICE = R4(2.094E06)
DENH2O = R4(1000.0)
DENICE = R4(917.0)


@dataclass
class VegeFluxParameters:
    """The ``noahmp_parameters`` components the VEGE_FLUX subtree reads."""

    DLEAF: R4 = R4(0.04)
    HVT: R4 = R4(20.0)
    CBIOM: R4 = R4(0.02)
    C3PSN: R4 = R4(1.0)
    KC25: R4 = R4(30.0)
    AKC: R4 = R4(2.1)
    KO25: R4 = R4(3.0E4)
    AKO: R4 = R4(1.2)
    AVCMX: R4 = R4(2.4)
    VCMX25: R4 = R4(60.0)
    BP: R4 = R4(2.0E3)
    MP: R4 = R4(9.0)
    QE25: R4 = R4(0.06)
    FOLNMX: R4 = R4(1.5)

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            setattr(self, name, R4(getattr(self, name)))


# --------------------------------------------------------------------------
# ESAT
# --------------------------------------------------------------------------
_A = tuple(R4(v) for v in (6.107799961, 4.436518521E-01, 1.428945805E-02,
                           2.650648471E-04, 3.031240396E-06, 2.034080948E-08,
                           6.136820929E-11))
_B = tuple(R4(v) for v in (6.109177956, 5.034698970E-01, 1.886013408E-02,
                           4.176223716E-04, 5.824720280E-06, 4.838803174E-08,
                           1.838826904E-10))
_C = tuple(R4(v) for v in (4.438099984E-01, 2.857002636E-02, 7.938054040E-04,
                           1.215215065E-05, 1.036561403E-07, 3.532421810E-10,
                           -7.090244804E-13))
_D = tuple(R4(v) for v in (5.030305237E-01, 3.773255020E-02, 1.267995369E-03,
                           2.477563108E-05, 3.005693132E-07, 2.158542548E-09,
                           7.131097725E-12))


def _horner(t: R4, c) -> R4:
    """``c0 + T*(c1 + T*(c2 + T*(c3 + T*(c4 + T*(c5 + T*c6)))))``."""
    acc = c[5] + t * c[6]
    acc = c[4] + t * acc
    acc = c[3] + t * acc
    acc = c[2] + t * acc
    acc = c[1] + t * acc
    return c[0] + t * acc


def esat(t) -> Tuple[R4, R4, R4, R4]:
    """Saturation vapour pressure over water and ice, and both derivatives.

    Straight-line: no branch, so a single case would bind the whole routine.
    """
    t = R4(t)
    esw = R4(100.0) * _horner(t, _A)
    esi = R4(100.0) * _horner(t, _B)
    desw = R4(100.0) * _horner(t, _C)
    desi = R4(100.0) * _horner(t, _D)
    return esw, esi, desw, desi


# --------------------------------------------------------------------------
# RAGRB
# --------------------------------------------------------------------------
def ragrb(iter_: int, vai, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc,
          z0h, fv, cwp, mpe, tv, mozg, fhg, dleaf):
    """Under-canopy aerodynamic resistances and leaf boundary-layer resistance.

    ``tv`` is declared INTENT(INOUT) by WRF but is never read or written by the
    routine body; it is accepted here for signature fidelity and deliberately
    unused (see the mutation study).
    """
    del tv
    vai = R4(vai); rhoair = R4(rhoair); hg = R4(hg); tah = R4(tah)
    zpd = R4(zpd); z0mg = R4(z0mg); z0hg = R4(z0hg); hcan = R4(hcan)
    uc = R4(uc); z0h = R4(z0h); fv = R4(fv); cwp = R4(cwp); mpe = R4(mpe)
    fhg = R4(fhg); dleaf = R4(dleaf)

    mozg = R4(0.0)
    molg = R4(0.0)

    if iter_ > 1:
        tmp1 = VKC * (GRAV / tah) * hg / (rhoair * CPAIR)
        if abs(tmp1) <= mpe:
            tmp1 = mpe
        molg = R4(-1.0) * _p3(fv) / tmp1
        mozg = _mn((zpd - z0mg) / molg, 1.0)

    if mozg < 0.0:
        fhgnew = _pow(R4(1.0) - R4(15.0) * mozg, -0.25)
    else:
        fhgnew = R4(1.0) + R4(4.7) * mozg

    if iter_ == 1:
        fhg = fhgnew
    else:
        fhg = R4(0.5) * (fhg + fhgnew)

    cwpc = _pow(cwp * vai * hcan * fhg, 0.5)

    tmp1 = _exp(-(cwpc * z0hg / hcan))
    tmp2 = _exp(-(cwpc * (z0h + zpd) / hcan))
    tmprah2 = hcan * _exp(cwpc) / cwpc * (tmp1 - tmp2)

    kh = _mx(VKC * fv * (hcan - zpd), mpe)
    ramg = R4(0.0)
    rahg = tmprah2 / kh
    rawg = rahg

    tmprb = cwpc * R4(50.0) / (R4(1.0) - _exp(-(cwpc / R4(2.0))))
    rb = tmprb * _sqrt(dleaf / uc)
    rb = _mn(_mx(rb, 5.0), 50.0)

    return mozg, fhg, ramg, rahg, rawg, rb


# --------------------------------------------------------------------------
# SFCDIF1
# --------------------------------------------------------------------------
def sfcdif1(iter_: int, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h, ur,
            mpe, moz, mozsgn: int, fm, fh, fm2, fh2, fv):
    """Monin-Obukhov drag coefficients (opt_sfc = 1)."""
    sfctmp = R4(sfctmp); rhoair = R4(rhoair); h = R4(h); qair = R4(qair)
    zlvl = R4(zlvl); zpd = R4(zpd); z0m = R4(z0m); z0h = R4(z0h); ur = R4(ur)
    mpe = R4(mpe); moz = R4(moz); fm = R4(fm); fh = R4(fh)
    fm2 = R4(fm2); fh2 = R4(fh2); fv = R4(fv)

    mozold = moz

    if zlvl <= zpd:
        raise ValueError("SFCDIF1: ZLVL <= ZPD; WRF calls wrf_error_fatal here")

    tmpcm = _log((zlvl - zpd) / z0m)
    tmpch = _log((zlvl - zpd) / z0h)
    tmpcm2 = _log((R4(2.0) + z0m) / z0m)
    tmpch2 = _log((R4(2.0) + z0h) / z0h)

    if iter_ == 1:
        fv = R4(0.0)
        moz = R4(0.0)
        moz2 = R4(0.0)
    else:
        tvir = (R4(1.0) + R4(0.61) * qair) * sfctmp
        tmp1 = VKC * (GRAV / tvir) * h / (rhoair * CPAIR)
        if abs(tmp1) <= mpe:
            tmp1 = mpe
        mol = R4(-1.0) * _p3(fv) / tmp1
        moz = _mn((zlvl - zpd) / mol, 1.0)
        moz2 = _mn((R4(2.0) + z0h) / mol, 1.0)

    if mozold * moz < 0.0:
        mozsgn += 1
    if mozsgn >= 2:
        moz = R4(0.0)
        fm = R4(0.0)
        fh = R4(0.0)
        moz2 = R4(0.0)
        fm2 = R4(0.0)
        fh2 = R4(0.0)

    if moz < 0.0:
        tmp1 = _pow(R4(1.0) - R4(16.0) * moz, 0.25)
        tmp2 = _log((R4(1.0) + tmp1 * tmp1) / R4(2.0))
        tmp3 = _log((R4(1.0) + tmp1) / R4(2.0))
        fmnew = R4(2.0) * tmp3 + tmp2 - R4(2.0) * _atan(tmp1) + R4(1.5707963)
        fhnew = R4(2.0) * tmp2

        tmp12 = _pow(R4(1.0) - R4(16.0) * moz2, 0.25)
        tmp22 = _log((R4(1.0) + tmp12 * tmp12) / R4(2.0))
        tmp32 = _log((R4(1.0) + tmp12) / R4(2.0))
        fm2new = R4(2.0) * tmp32 + tmp22 - R4(2.0) * _atan(tmp12) + R4(1.5707963)
        fh2new = R4(2.0) * tmp22
    else:
        fmnew = R4(-5.0) * moz
        fhnew = fmnew
        fm2new = R4(-5.0) * moz2
        fh2new = fm2new

    if iter_ == 1:
        fm = fmnew
        fh = fhnew
        fm2 = fm2new
        fh2 = fh2new
    else:
        fm = R4(0.5) * (fm + fmnew)
        fh = R4(0.5) * (fh + fhnew)
        fm2 = R4(0.5) * (fm2 + fm2new)
        fh2 = R4(0.5) * (fh2 + fh2new)

    fh = _mn(fh, R4(0.9) * tmpch)
    fm = _mn(fm, R4(0.9) * tmpcm)
    fh2 = _mn(fh2, R4(0.9) * tmpch2)
    fm2 = _mn(fm2, R4(0.9) * tmpcm2)

    cmfm = tmpcm - fm
    chfh = tmpch - fh
    cm2fm2 = tmpcm2 - fm2
    ch2fh2 = tmpch2 - fh2
    if abs(cmfm) <= mpe:
        cmfm = mpe
    if abs(chfh) <= mpe:
        chfh = mpe
    if abs(cm2fm2) <= mpe:
        cm2fm2 = mpe
    if abs(ch2fh2) <= mpe:
        ch2fh2 = mpe
    cm = VKC * VKC / (cmfm * cmfm)
    ch = VKC * VKC / (cmfm * chfh)

    fv = ur * _sqrt(cm)
    ch2 = VKC * fv / ch2fh2

    return moz, mozsgn, fm, fh, fm2, fh2, cm, ch, fv, ch2


# --------------------------------------------------------------------------
# STOMATA
# --------------------------------------------------------------------------
def _f1(ab: R4, bc: R4) -> R4:
    """``AB**((BC-25.0)/10.0)`` -- the Q10 statement function."""
    return _pow(ab, (bc - R4(25.0)) / R4(10.0))


def _f2(ab: R4) -> R4:
    """``1.0 + EXP((-2.2E05+710.0*(AB+273.16))/(8.314*(AB+273.16)))``."""
    t = ab + R4(273.16)
    return R4(1.0) + _exp((R4(-2.2E05) + R4(710.0) * t) / (R4(8.314) * t))


def stomata(p: VegeFluxParameters, mpe, apar, foln, tv, ei, ea, sfctmp,
            sfcprs, fveg, o2, co2, igs, btran, rb, niter: int = 3):
    """Ball-Berry stomatal resistance and leaf photosynthesis (opt_crs = 1)."""
    mpe = R4(mpe); apar = R4(apar); foln = R4(foln); tv = R4(tv); ei = R4(ei)
    ea = R4(ea); sfctmp = R4(sfctmp); sfcprs = R4(sfcprs); fveg = R4(fveg)
    o2 = R4(o2); co2 = R4(co2); igs = R4(igs); btran = R4(btran); rb = R4(rb)

    apar_scale = apar / _mx(fveg, 1.0e-6)
    cf = sfcprs / (R4(8.314) * sfctmp) * R4(1.0e06)
    rs = R4(1.0) / p.BP * cf
    psn = R4(0.0)

    if apar_scale <= 0.0:
        return rs, psn

    fnf = _mn(foln / _mx(mpe, p.FOLNMX), 1.0)
    tc = tv - TFRZ
    ppf = R4(4.6) * apar_scale
    j = ppf * p.QE25
    kc = p.KC25 * _f1(p.AKC, tc)
    ko = p.KO25 * _f1(p.AKO, tc)
    awc = kc * (R4(1.0) + o2 / ko)
    cp = R4(0.5) * kc / ko * o2 * R4(0.21)
    vcmx = p.VCMX25 / _f2(tc) * fnf * btran * _f1(p.AVCMX, tc)

    ci = R4(0.7) * co2 * p.C3PSN + R4(0.4) * co2 * (R4(1.0) - p.C3PSN)
    rlb = rb / cf
    cea = _mx(R4(0.25) * ei * p.C3PSN + R4(0.40) * ei * (R4(1.0) - p.C3PSN),
              _mn(ea, ei))

    for _ in range(niter):
        wj = (_mx(ci - cp, 0.0) * j / (ci + R4(2.0) * cp) * p.C3PSN
              + j * (R4(1.0) - p.C3PSN))
        wc = (_mx(ci - cp, 0.0) * vcmx / (ci + awc) * p.C3PSN
              + vcmx * (R4(1.0) - p.C3PSN))
        we = (R4(0.5) * vcmx * p.C3PSN
              + R4(4000.0) * vcmx * ci / sfcprs * (R4(1.0) - p.C3PSN))
        psn = _mn(wj, wc, we) * igs

        cs = _mx(co2 - R4(1.37) * rlb * sfcprs * psn, mpe)
        a = p.MP * psn * sfcprs * cea / (cs * ei) + p.BP
        b = (p.MP * psn * sfcprs / cs + p.BP) * rlb - R4(1.0)
        c = -rlb
        if b >= 0.0:
            q = R4(-0.5) * (b + _sqrt(b * b - R4(4.0) * a * c))
        else:
            q = R4(-0.5) * (b - _sqrt(b * b - R4(4.0) * a * c))
        r1 = q / a
        r2 = c / q
        rs = _mx(r1, r2)
        ci = _mx(cs - psn * sfcprs * R4(1.65) * rs, 0.0)

    rs = rs * cf
    return rs, psn


# --------------------------------------------------------------------------
# VEGE_FLUX
# --------------------------------------------------------------------------
NITERC = 20
NITERG = 5


@dataclass
class VegeFluxState:
    """Everything VEGE_FLUX returns (INOUT arguments included)."""

    EAH: R4 = R4(0.0)
    TAH: R4 = R4(0.0)
    TV: R4 = R4(0.0)
    TG: R4 = R4(0.0)
    CM: R4 = R4(0.0)
    CH: R4 = R4(0.0)
    TAUXV: R4 = R4(0.0)
    TAUYV: R4 = R4(0.0)
    IRG: R4 = R4(0.0)
    IRC: R4 = R4(0.0)
    SHG: R4 = R4(0.0)
    SHC: R4 = R4(0.0)
    EVG: R4 = R4(0.0)
    EVC: R4 = R4(0.0)
    TR: R4 = R4(0.0)
    GH: R4 = R4(0.0)
    T2MV: R4 = R4(0.0)
    PSNSUN: R4 = R4(0.0)
    PSNSHA: R4 = R4(0.0)
    CANHS: R4 = R4(0.0)
    QSFC: R4 = R4(0.0)
    Q2V: R4 = R4(0.0)
    CAH2: R4 = R4(0.0)
    CHLEAF: R4 = R4(0.0)
    CHUC: R4 = R4(0.0)
    RSSUN: R4 = R4(0.0)
    RSSHA: R4 = R4(0.0)
    SAV: R4 = R4(0.0)
    SAG: R4 = R4(0.0)
    FSR: R4 = R4(0.0)
    iters: int = 0
    branches: Dict[str, object] = field(default_factory=dict)


def vege_flux(p: VegeFluxParameters, nsnow: int, nsoil: int, isnow: int,
              dt, sav, sag, lwdn, ur, uu, vv, sfctmp, thair, qair, eair,
              rhoair, snowh, vai, gammav, gammag, fwet, laisun, laisha, cwp,
              dzsnso: Dict[int, float], zlvl, zpd, z0m, fveg, z0mg, emv, emg,
              canliq, fsno, canice, stc: Dict[int, float], df: Dict[int, float],
              rsurf, latheav, latheag, parsun, parsha, igs, foln, co2air,
              o2air, btran, sfcprs, rhsur, q2, pahv, pahg, eah, tah, tv, tg,
              cm, ch, qc, qsfc, psfc, fsr,
              opt_sfc: int = 1, opt_crs: int = 1, opt_crop: int = 0,
              opt_stc: int = 1) -> VegeFluxState:
    """Vegetated surface energy balance, WRF v4.6.1 Registry option identity.

    ``dzsnso``/``stc``/``df`` are dicts keyed by the Fortran index, which runs
    ``-nsnow+1 .. nsoil``.

    ``rssun``/``rssha`` are INTENT(OUT) in WRF and are written by STOMATA on
    iteration 1 before anything reads them, so no incoming value exists; they
    are not arguments here.
    """
    if opt_sfc != 1:
        raise NotImplementedError(
            "opt_sfc != 1 selects SFCDIF2, which the WRF Registry default "
            "(opt_sfc=1) makes dead; it is not transcribed")
    if opt_crs != 1:
        raise NotImplementedError(
            "opt_crs != 1 selects CANRES, dead under the Registry default "
            "opt_crs=1; it is not transcribed")
    if opt_crop != 0:
        raise NotImplementedError(
            "opt_crop != 0 activates the gecros chain, dead under the Registry "
            "default opt_crop=0; it is not transcribed")
    if opt_stc not in (1, 2):
        raise NotImplementedError(
            "only opt_stc=1 (Registry default) and the opt_stc=2 no-reset case "
            "are transcribed; opt_stc=3 is dead under the default")

    dt = R4(dt); sav = R4(sav); sag = R4(sag); lwdn = R4(lwdn); ur = R4(ur)
    uu = R4(uu); vv = R4(vv); sfctmp = R4(sfctmp); thair = R4(thair)
    qair = R4(qair); eair = R4(eair); rhoair = R4(rhoair); snowh = R4(snowh)
    vai = R4(vai); gammav = R4(gammav); gammag = R4(gammag); fwet = R4(fwet)
    laisun = R4(laisun); laisha = R4(laisha); cwp = R4(cwp)
    zlvl = R4(zlvl); zpd = R4(zpd); z0m = R4(z0m); fveg = R4(fveg)
    z0mg = R4(z0mg); emv = R4(emv); emg = R4(emg); canliq = R4(canliq)
    fsno = R4(fsno); canice = R4(canice); rsurf = R4(rsurf)
    latheav = R4(latheav); latheag = R4(latheag); parsun = R4(parsun)
    parsha = R4(parsha); igs = R4(igs); foln = R4(foln); co2air = R4(co2air)
    o2air = R4(o2air); btran = R4(btran); sfcprs = R4(sfcprs); rhsur = R4(rhsur)
    pahv = R4(pahv); pahg = R4(pahg); eah = R4(eah); tah = R4(tah)
    tv = R4(tv); tg = R4(tg); cm = R4(cm); ch = R4(ch); qsfc = R4(qsfc)
    psfc = R4(psfc); fsr = R4(fsr)
    del q2, qc, thair          # read by WRF's signature, never by the body
    stc = {k: R4(v) for k, v in stc.items()}
    df = {k: R4(v) for k, v in df.items()}
    dzsnso = {k: R4(v) for k, v in dzsnso.items()}

    def tdc(t: R4) -> R4:
        return _mn(50.0, _mx(-50.0, t - TFRZ))

    out = VegeFluxState()
    mpe = R4(1E-6)
    liter = 0
    fv = R4(0.1)

    dtv = R4(0.0)
    dtg = R4(0.0)
    moz = R4(0.0)
    mozsgn = 0
    fh2 = R4(0.0)
    hg = R4(0.0)
    h = R4(0.0)
    fm = R4(0.0)
    fh = R4(0.0)
    fm2 = R4(0.0)
    mozg = R4(0.0)
    fhg = R4(0.0)
    rssun = R4(0.0)
    rssha = R4(0.0)
    psnsun = R4(0.0)
    psnsha = R4(0.0)

    vaie = _mn(6.0, vai)
    laisune = _mn(6.0, laisun)
    laishae = _mn(6.0, laisha)

    t = tdc(tg)
    esatw, esati, dsatw, dsati = esat(t)
    estg = esatw if t > 0.0 else esati

    qsfc = R4(0.622) * eair / (psfc - R4(0.378) * eair)

    hcan = p.HVT
    # The first `UC = UR*LOG(HCAN/Z0M)/LOG(ZLVL/Z0M)` in WRF is dead: the very
    # next statement overwrites UC unconditionally and the expression has no
    # side effect.
    uc = ur * _log((hcan - zpd + z0m) / z0m) / _log(zlvl / z0m)
    if (hcan - zpd) <= 0.0:
        raise ValueError("VEGE_FLUX: HCAN <= ZPD; WRF calls wrf_error_fatal here")

    air = (-(emv * (R4(1.0) + (R4(1.0) - emv) * (R4(1.0) - emg)) * lwdn)
           - emv * emg * SB * _p4(tg))
    cir = (R4(2.0) - emv * (R4(1.0) - emg)) * emv * SB

    cah = R4(0.0)
    cvh = R4(0.0)
    rahg = R4(1.0)
    z0h = z0m
    iters = 0

    for it in range(1, NITERC + 1):
        iters = it
        z0h = z0m
        z0hg = z0mg

        moz, mozsgn, fm, fh, fm2, fh2, cm, ch, fv, ch2 = sfcdif1(
            it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h, ur,
            mpe, moz, mozsgn, fm, fh, fm2, fh2, fv)

        ramc = _mx(1.0, R4(1.0) / (cm * ur))
        rahc = _mx(1.0, R4(1.0) / (ch * ur))
        rawc = rahc
        del ramc

        mozg, fhg, ramg, rahg, rawg, rb = ragrb(
            it, vaie, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc,
            z0h, fv, cwp, mpe, tv, mozg, fhg, p.DLEAF)
        del ramg

        t = tdc(tv)
        esatw, esati, dsatw, dsati = esat(t)
        if t > 0.0:
            estv, destv = esatw, dsatw
        else:
            estv, destv = esati, dsati

        if it == 1:
            rssun, psnsun = stomata(p, mpe, parsun, foln, tv, estv, eah,
                                    sfctmp, sfcprs, fveg, o2air, co2air,
                                    igs, btran, rb)
            rssha, psnsha = stomata(p, mpe, parsha, foln, tv, estv, eah,
                                    sfctmp, sfcprs, fveg, o2air, co2air,
                                    igs, btran, rb)

        cah = R4(1.0) / rahc
        cvh = R4(2.0) * vaie / rb
        cgh = R4(1.0) / rahg
        cond = cah + cvh + cgh
        ata = (sfctmp * cah + tg * cgh) / cond
        bta = cvh / cond
        csh = (R4(1.0) - bta) * rhoair * CPAIR * cvh

        caw = R4(1.0) / rawc
        cew = fwet * vaie / rb
        ctw = (R4(1.0) - fwet) * (laisune / (rb + rssun) + laishae / (rb + rssha))
        cgw = R4(1.0) / (rawg + rsurf)
        cond = caw + cew + ctw + cgw
        aea = (eair * caw + estg * cgw) / cond
        bea = (cew + ctw) / cond
        cev = (R4(1.0) - bea) * cew * rhoair * CPAIR / gammav
        ctr = (R4(1.0) - bea) * ctw * rhoair * CPAIR / gammav

        tah = ata + bta * tv
        eah = aea + bea * estv

        irc = fveg * (air + cir * _p4(tv))
        shc = fveg * rhoair * CPAIR * cvh * (tv - tah)
        evc = fveg * rhoair * CPAIR * cew * (estv - eah) / gammav
        tr = fveg * rhoair * CPAIR * ctw * (estv - eah) / gammav
        if tv > TFRZ:
            evc = _mn(canliq * latheav / dt, evc)
        else:
            evc = _mn(canice * latheav / dt, evc)

        hcv = (p.CBIOM * vaie * CWAT + canliq * CWAT / DENH2O
               + canice * CICE / DENICE)

        b = sav - irc - shc - evc - tr + pahv
        a = fveg * (R4(4.0) * cir * _p3(tv) + csh + (cev + ctr) * destv + hcv / dt)
        dtv = b / a

        irc = irc + fveg * R4(4.0) * cir * _p3(tv) * dtv
        shc = shc + fveg * csh * dtv
        evc = evc + fveg * cev * destv * dtv
        tr = tr + fveg * ctr * destv * dtv
        canhs = dtv * fveg * hcv / dt

        tv = tv + dtv

        h = rhoair * CPAIR * (tah - sfctmp) / rahc
        hg = rhoair * CPAIR * (tg - tah) / rahg

        qsfc = (R4(0.622) * eah) / (sfcprs - R4(0.378) * eah)

        if liter == 1:
            break
        if it >= 5 and abs(dtv) <= 0.01 and liter == 0:
            liter = 1

    # ---- under-canopy fluxes and TG ---------------------------------------
    air = -(emg * (R4(1.0) - emv) * lwdn) - emg * emv * SB * _p4(tv)
    cir = emg * SB
    csh = rhoair * CPAIR / rahg
    cev = rhoair * CPAIR / (gammag * (rawg + rsurf))
    cgh = R4(2.0) * df[isnow + 1] / dzsnso[isnow + 1]

    estg = R4(0.0)
    irg = shg = evg = gh = R4(0.0)
    for _ in range(NITERG):
        t = tdc(tg)
        esatw, esati, dsatw, dsati = esat(t)
        if t > 0.0:
            estg, destg = esatw, dsatw
        else:
            estg, destg = esati, dsati

        irg = cir * _p4(tg) + air
        shg = csh * (tg - tah)
        evg = cev * (estg * rhsur - eah)
        gh = cgh * (tg - stc[isnow + 1])

        b = sag - irg - shg - evg - gh + pahg
        a = R4(4.0) * cir * _p3(tg) + csh + cev * destg + cgh
        dtg = b / a

        irg = irg + R4(4.0) * cir * _p3(tg) * dtg
        shg = shg + csh * dtg
        evg = evg + cev * destg * dtg
        gh = gh + cgh * dtg
        tg = tg + dtg

    reset = False
    if opt_stc == 1:
        if snowh > 0.05 and tg > TFRZ:
            reset = True
            tg = TFRZ
            irg = (cir * _p4(tg) - emg * (R4(1.0) - emv) * lwdn
                   - emg * emv * SB * _p4(tv))
            shg = csh * (tg - tah)
            evg = cev * (estg * rhsur - eah)
            gh = sag + pahg - (irg + shg + evg)

    tauxv = -(rhoair * cm * ur * uu)
    tauyv = -(rhoair * cm * ur * vv)

    # 2 m diagnostics (opt_sfc = 1)
    # The first CAH2 assignment in WRF is dead; the next line overwrites it.
    cah2 = fv * VKC / (_log((R4(2.0) + z0h) / z0h) - fh2)
    cq2v = cah2
    if cah2 < 1.0E-5:
        t2mv = tah
        q2v = qsfc
    else:
        t2mv = tah - (shg + shc / fveg) / (rhoair * CPAIR) * R4(1.0) / cah2
        q2v = qsfc - ((evc + tr) / fveg + evg) / (latheav * rhoair) * R4(1.0) / cq2v

    ch = cah
    chleaf = cvh
    chuc = R4(1.0) / rahg

    out.EAH, out.TAH, out.TV, out.TG = eah, tah, tv, tg
    out.CM, out.CH = cm, ch
    out.TAUXV, out.TAUYV = tauxv, tauyv
    out.IRG, out.IRC, out.SHG, out.SHC = irg, irc, shg, shc
    out.EVG, out.EVC, out.TR, out.GH = evg, evc, tr, gh
    out.T2MV, out.PSNSUN, out.PSNSHA, out.CANHS = t2mv, psnsun, psnsha, canhs
    out.QSFC, out.Q2V, out.CAH2 = qsfc, q2v, cah2
    out.CHLEAF, out.CHUC = chleaf, chuc
    out.RSSUN, out.RSSHA = rssun, rssha
    out.SAV, out.SAG, out.FSR = sav, sag, fsr
    out.iters = iters
    out.branches = {
        "loop1_iters": iters,
        "tg_reset": reset,
        "cah2_small": bool(cah2 < 1.0E-5),
    }
    return out
