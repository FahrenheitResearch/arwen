"""CPU references for the pinned WRF v4.6.1 Noah-MP flux-preparation leaves.

``SFCDIF1``, ``RAGRB`` and ``STOMATA``: the three routines VEGE_FLUX and
BARE_FLUX call before they solve a surface energy balance.  Each function here
is a direct FP32 transcription of one ``private`` procedure of
``phys/module_sf_noahmplsm.F`` at commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``, validated bitwise against
``gpuwm/data/noahmp/oracle/noahmp-fluxprep.csv``.  That fixture is produced by
``tools/noahmp_wrf461_oracle/build_fluxprep.sh`` from a scratch copy of the
pinned source carrying only the audited visibility patch
``patches/noahmp-lsm-leaf-visibility.patch``.

This module is a sibling of ``noahmp_leaves.py`` rather than an extension of
it, so that the parallel porting lanes never edit one file.  It shares that
module's conventions:

* ``kind_phys`` is ``kind(1.0)``, i.e. FP32.  Every literal is materialised as
  ``np.float32`` so no expression silently widens.
* Transcendentals go through ``gpuwm.core.noahmp_libm``.  gfortran on x86-64
  lowers ``EXP``/``LOG``/``ATAN`` to glibc's ``expf``/``logf``/``atanf`` and a
  **real** constant exponent -- ``x**0.5``, ``x**0.25``, ``x**(-0.25)`` -- to
  ``powf``, none of which is correctly rounded.  An **integer** constant
  exponent is different again: ``FV**3`` becomes libgcc's ``__powisf2``, which
  is binary exponentiation, ``fl(x * fl(x*x))`` -- not ``powf(x, 3.0)`` and not
  a single rounding.  ``_powi3`` below is that expansion.

Nothing here admits ``sf_surface_physics=4``; these are validation surfaces for
the leaf ports, not a runtime path.

Dead under the pinned option identity, asserted off rather than ported:

* ``SFCDIF2`` (:4747-4948) -- only reachable from ``IF(OPT_SFC == 2)``;
  ``opt_sfc`` is 1.
* ``CANRES`` (:5141-5220) and ``CALHUM`` (:5224-5262) -- only reachable from
  ``IF (OPT_CRS == 2)``; ``opt_crs`` is 1, which is why ``STOMATA`` is the live
  canopy-conductance leaf.
* the Gecros crop chain at VEGE_FLUX:3958 -- ``IF (opt_crop == 2)``;
  ``opt_crop`` is 0.

Iteration counts the pinned identity produces: ``SFCDIF1`` and ``RAGRB`` are
called with ``ITER`` running 1..NITERC = 20 in VEGE_FLUX (:3792, :3877) and
1..NITERB = 5 in BARE_FLUX (:4329, :4351), and both bodies test only
``ITER == 1`` versus ``ITER > 1``.  ``STOMATA`` is called only at ``ITER == 1``
(:3934) and runs exactly ``NITER = 3`` Ball-Berry sweeps (:5044-5046), a
compile-time constant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from gpuwm.core.noahmp_libm import atanf as _atanf
from gpuwm.core.noahmp_libm import expf as _expf
from gpuwm.core.noahmp_libm import logf as _logf
from gpuwm.core.noahmp_libm import powf as _powf


F = np.float32

# module_sf_noahmplsm.F:204-220.
GRAV = F(9.80616)
VKC = F(0.40)
TFRZ = F(273.16)
CPAIR = F(1004.64)


def _sqrtf(x) -> np.float32:
    """``SQRT`` on REAL(4).  glibc's ``sqrtf`` is the SSE instruction, which is
    correctly rounded, so numpy agrees with it bit for bit."""
    return np.sqrt(F(x), dtype=F)


def _powi3(x) -> np.float32:
    """``x**3`` as gfortran emits it: libgcc ``__powisf2``, binary exponentiation.

    ``n = 3`` is odd, so ``y = x``; the single loop pass squares ``x`` and
    multiplies, giving ``fl(x * fl(x*x))``.  That is two roundings, whereas
    ``powf(x, 3.0)`` is one, and they differ.
    """
    x = F(x)
    return F(x * F(x * x))


# ---------------------------------------------------------------------------
# RAGRB -- module_sf_noahmplsm.F:4483-4579
# ---------------------------------------------------------------------------
RAGRB_INTS = ("iter", "vegtyp", "iloc", "jloc")
RAGRB_INPUTS = (
    "vai", "rhoair", "hg", "tv", "tah", "zpd", "z0mg", "z0hg", "hcan", "uc",
    "z0h", "fv", "cwp", "mpe", "mozg", "fhg", "dleaf",
)
RAGRB_OUTPUTS = ("mozg", "fhg", "ramg", "rahg", "rawg", "rb")


def ragrb(iteration, vai, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc, z0h,
          fv, cwp, mpe, fhg, dleaf):
    """Under-canopy aerodynamic resistances and leaf boundary-layer resistance.

    ``TV``, ``VEGTYP``, ``ILOC`` and ``JLOC`` are in WRF's argument list and the
    body never references them, so they are not parameters here.  ``MOZG`` is
    ``INTENT(INOUT)`` but :4536 assigns it 0.0 before any read, so its incoming
    value cannot be consumed either; it is an output only.  The oracle's
    substitution sweep is the proof for all five.

    Returns ``(mozg, fhg, ramg, rahg, rawg, rb)``.
    """
    vai, rhoair, hg, tah = F(vai), F(rhoair), F(hg), F(tah)
    zpd, z0mg, z0hg, hcan = F(zpd), F(z0mg), F(z0hg), F(hcan)
    uc, z0h, fv, cwp = F(uc), F(z0h), F(fv), F(cwp)
    mpe, fhg, dleaf = F(mpe), F(fhg), F(dleaf)

    mozg = F(0.0)                                              # :4536
    if iteration > 1:                                          # :4540
        tmp1 = F(F(F(VKC * F(GRAV / tah)) * hg) / F(rhoair * CPAIR))
        if abs(tmp1) <= mpe:                                   # :4542
            tmp1 = mpe
        molg = F(-F(F(F(1.0) * _powi3(fv)) / tmp1))            # :4543
        mozg = min(F(F(zpd - z0mg) / molg), F(1.0))            # :4544
        mozg = F(mozg)

    if mozg < F(0.0):                                          # :4547
        fhgnew = _powf(F(F(1.0) - F(F(15.0) * mozg)), F(-0.25))
    else:                                                      # :4549
        fhgnew = F(F(1.0) + F(F(4.7) * mozg))

    if iteration == 1:                                         # :4552
        fhg = fhgnew
    else:                                                      # :4554
        fhg = F(F(0.5) * F(fhg + fhgnew))

    cwpc = _powf(F(F(F(cwp * vai) * hcan) * fhg), F(0.5))      # :4557

    tmp1 = _expf(F(-F(F(cwpc * z0hg) / hcan)))                 # :4560
    tmp2 = _expf(F(-F(F(cwpc * F(z0h + zpd)) / hcan)))         # :4561
    tmprah2 = F(F(F(hcan * _expf(cwpc)) / cwpc) * F(tmp1 - tmp2))  # :4562

    kh = max(F(F(VKC * fv) * F(hcan - zpd)), mpe)              # :4566
    kh = F(kh)
    ramg = F(0.0)                                              # :4567
    rahg = F(tmprah2 / kh)                                     # :4568
    rawg = rahg                                                # :4569

    tmprb = F(F(cwpc * F(50.0))
              / F(F(1.0) - _expf(F(-F(cwpc / F(2.0))))))       # :4573
    rb = F(tmprb * _sqrtf(F(dleaf / uc)))                      # :4574
    rb = F(min(max(rb, F(5.0)), F(50.0)))                      # :4575
    return mozg, fhg, ramg, rahg, rawg, rb


# ---------------------------------------------------------------------------
# SFCDIF1 -- module_sf_noahmplsm.F:4583-4743
# ---------------------------------------------------------------------------
SFCDIF1_INTS = ("iter", "mozsgn", "iloc", "jloc")
SFCDIF1_INPUTS = (
    "sfctmp", "rhoair", "h", "qair", "zlvl", "zpd", "z0m", "z0h", "ur", "mpe",
    "moz", "fm", "fh", "fm2", "fh2", "fv",
)
SFCDIF1_OUTPUTS = (
    "moz", "mozsgn", "fm", "fh", "fm2", "fh2", "fv", "cm", "ch", "ch2",
)


class SfcdifDomainError(ValueError):
    """``ZLVL <= ZPD``: WRF calls ``wrf_error_fatal`` and stops the model.

    :4649-4652 is a live branch under the pinned identity, but its result is
    termination rather than a value, so no fixture row can pin it.  The port
    raises instead of inventing an answer.
    """


def sfcdif1(iteration, mozsgn, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h,
            ur, mpe, moz, fm, fh, fm2, fh2, fv):
    """Monin-Obukhov drag coefficients for momentum and heat.

    ``ILOC`` and ``JLOC`` are in WRF's argument list and never referenced.
    Returns ``(moz, mozsgn, fm, fh, fm2, fh2, fv, cm, ch, ch2)``.
    """
    sfctmp, rhoair, h, qair = F(sfctmp), F(rhoair), F(h), F(qair)
    zlvl, zpd, z0m, z0h = F(zlvl), F(zpd), F(z0m), F(z0h)
    ur, mpe, moz = F(ur), F(mpe), F(moz)
    fm, fh, fm2, fh2, fv = F(fm), F(fh), F(fm2), F(fh2), F(fv)

    mozold = moz                                               # :4647
    if zlvl <= zpd:                                            # :4649
        raise SfcdifDomainError(
            f"ZLVL ({zlvl}) <= ZPD ({zpd}); WRF calls wrf_error_fatal at "
            f"module_sf_noahmplsm.F:4651")

    tmpcm = _logf(F(F(zlvl - zpd) / z0m))                      # :4654
    tmpch = _logf(F(F(zlvl - zpd) / z0h))                      # :4655
    tmpcm2 = _logf(F(F(F(2.0) + z0m) / z0m))                   # :4656
    tmpch2 = _logf(F(F(F(2.0) + z0h) / z0h))                   # :4657

    if iteration == 1:                                         # :4659
        fv = F(0.0)
        moz = F(0.0)
        moz2 = F(0.0)
    else:                                                      # :4664
        tvir = F(F(F(1.0) + F(F(0.61) * qair)) * sfctmp)
        tmp1 = F(F(F(VKC * F(GRAV / tvir)) * h) / F(rhoair * CPAIR))
        if abs(tmp1) <= mpe:                                   # :4667
            tmp1 = mpe
        mol = F(-F(F(F(1.0) * _powi3(fv)) / tmp1))             # :4668
        moz = F(min(F(F(zlvl - zpd) / mol), F(1.0)))           # :4669
        moz2 = F(min(F(F(F(2.0) + z0h) / mol), F(1.0)))        # :4670

    if F(mozold * moz) < F(0.0):                               # :4675
        mozsgn = mozsgn + 1
    if mozsgn >= 2:                                            # :4676
        moz = F(0.0)
        fm = F(0.0)
        fh = F(0.0)
        moz2 = F(0.0)
        fm2 = F(0.0)
        fh2 = F(0.0)

    if moz < F(0.0):                                           # :4686
        tmp1 = _powf(F(F(1.0) - F(F(16.0) * moz)), F(0.25))
        tmp2 = _logf(F(F(F(1.0) + F(tmp1 * tmp1)) / F(2.0)))
        tmp3 = _logf(F(F(F(1.0) + tmp1) / F(2.0)))
        fmnew = F(F(F(F(F(2.0) * tmp3) + tmp2)
                    - F(F(2.0) * _atanf(tmp1))) + F(1.5707963))
        fhnew = F(F(2.0) * tmp2)

        tmp12 = _powf(F(F(1.0) - F(F(16.0) * moz2)), F(0.25))  # :4694
        tmp22 = _logf(F(F(F(1.0) + F(tmp12 * tmp12)) / F(2.0)))
        tmp32 = _logf(F(F(F(1.0) + tmp12) / F(2.0)))
        fm2new = F(F(F(F(F(2.0) * tmp32) + tmp22)
                     - F(F(2.0) * _atanf(tmp12))) + F(1.5707963))
        fh2new = F(F(2.0) * tmp22)
    else:                                                      # :4700
        fmnew = F(F(-5.0) * moz)
        fhnew = fmnew
        fm2new = F(F(-5.0) * moz2)
        fh2new = fm2new

    if iteration == 1:                                         # :4709
        fm, fh, fm2, fh2 = fmnew, fhnew, fm2new, fh2new
    else:                                                      # :4714
        fm = F(F(0.5) * F(fm + fmnew))
        fh = F(F(0.5) * F(fh + fhnew))
        fm2 = F(F(0.5) * F(fm2 + fm2new))
        fh2 = F(F(0.5) * F(fh2 + fh2new))

    fh = F(min(fh, F(F(0.9) * tmpch)))                         # :4722
    fm = F(min(fm, F(F(0.9) * tmpcm)))                         # :4723
    fh2 = F(min(fh2, F(F(0.9) * tmpch2)))                      # :4724
    fm2 = F(min(fm2, F(F(0.9) * tmpcm2)))                      # :4725

    cmfm = F(tmpcm - fm)                                       # :4727
    chfh = F(tmpch - fh)
    cm2fm2 = F(tmpcm2 - fm2)
    ch2fh2 = F(tmpch2 - fh2)
    if abs(cmfm) <= mpe:                                       # :4731
        cmfm = mpe
    if abs(chfh) <= mpe:
        chfh = mpe
    if abs(cm2fm2) <= mpe:
        cm2fm2 = mpe
    if abs(ch2fh2) <= mpe:
        ch2fh2 = mpe
    cm = F(F(VKC * VKC) / F(cmfm * cmfm))                      # :4735
    ch = F(F(VKC * VKC) / F(cmfm * chfh))                      # :4736

    fv = F(ur * _sqrtf(cm))                                    # :4741
    ch2 = F(F(VKC * fv) / ch2fh2)                              # :4742
    return moz, mozsgn, fm, fh, fm2, fh2, fv, cm, ch, ch2


# ---------------------------------------------------------------------------
# STOMATA -- module_sf_noahmplsm.F:5005-5137
# ---------------------------------------------------------------------------
STOMATA_INTS = ("vegtyp", "iloc", "jloc")
STOMATA_INPUTS = (
    "mpe", "apar", "foln", "tv", "ei", "ea", "sfctmp", "sfcprs", "fveg", "o2",
    "co2", "igs", "btran", "rb", "bp", "folnmx", "qe25", "kc25", "akc",
    "ko25", "ako", "vcmx25", "avcmx", "c3psn", "mp",
)
STOMATA_OUTPUTS = ("rs", "psn")

#: ``DATA NITER /3/`` at :5045.  A compile-time constant, not a convergence
#: test: STOMATA always runs exactly three Ball-Berry sweeps.
STOMATA_NITER = 3


def _f1(ab, tc):
    """``F1(AB,BC) = AB**((BC-25.0)/10.0)`` -- the statement function at :5074."""
    return _powf(F(ab), F(F(F(tc) - F(25.0)) / F(10.0)))


def _f2(tc):
    """``F2(AB) = 1.0+EXP((-2.2E05+710.0*(AB+273.16))/(8.314*(AB+273.16)))``, :5075."""
    tc = F(tc)
    shifted = F(tc + F(273.16))
    return F(F(1.0) + _expf(F(F(F(-2.2e05) + F(F(710.0) * shifted))
                              / F(F(8.314) * shifted))))


def stomata(mpe, apar, foln, tv, ei, ea, sfctmp, sfcprs, fveg, o2, co2, igs,
            btran, rb, bp, folnmx, qe25, kc25, akc, ko25, ako, vcmx25, avcmx,
            c3psn, mp):
    """Ball-Berry stomatal resistance and leaf photosynthesis, ``OPT_CRS = 1``.

    ``VEGTYP``, ``ILOC`` and ``JLOC`` are in WRF's argument list and never
    referenced.  The eleven ``parameters`` components the body reads are
    ordinary arguments here.  Returns ``(rs, psn)``.

    The ``C3PSN`` terms are transcribed in full even though WRF's pinned
    MPTABLE.TBL ships ``C3PSN = 1.0`` for all 20 MODIS classes.  They are
    arithmetic in one expression, not a branch behind an option switch, and
    adding ``J*(1.0-C3PSN)`` is not a no-op in FP32 when the other addend is
    ``-0.0``.
    """
    mpe, apar, foln, tv = F(mpe), F(apar), F(foln), F(tv)
    ei, ea, sfctmp, sfcprs = F(ei), F(ea), F(sfctmp), F(sfcprs)
    fveg, o2, co2, igs = F(fveg), F(o2), F(co2), F(igs)
    btran, rb, bp, folnmx = F(btran), F(rb), F(bp), F(folnmx)
    qe25, kc25, akc, ko25 = F(qe25), F(kc25), F(akc), F(ko25)
    ako, vcmx25, avcmx = F(ako), F(vcmx25), F(avcmx)
    c3psn, mp = F(c3psn), F(mp)

    apar_scale = F(apar / max(fveg, F(1.0e-6)))                # :5083
    cf = F(F(sfcprs / F(F(8.314) * sfctmp)) * F(1.0e06))       # :5084
    rs = F(F(F(1.0) / bp) * cf)                                # :5085
    psn = F(0.0)                                               # :5086

    if apar_scale <= F(0.0):                                   # :5088
        return rs, psn

    fnf = F(min(F(foln / max(mpe, folnmx)), F(1.0)))           # :5090
    tc = F(tv - TFRZ)                                          # :5091
    ppf = F(F(4.6) * apar_scale)                               # :5092
    j = F(ppf * qe25)                                          # :5093
    kc = F(kc25 * _f1(akc, tc))                                # :5094
    ko = F(ko25 * _f1(ako, tc))                                # :5095
    awc = F(kc * F(F(1.0) + F(o2 / ko)))                       # :5096
    cp = F(F(F(F(F(0.5) * kc) / ko) * o2) * F(0.21))           # :5097
    vcmx = F(F(F(F(vcmx25 / _f2(tc)) * fnf) * btran)
             * _f1(avcmx, tc))                                 # :5098

    ci = F(F(F(F(0.7) * co2) * c3psn)
           + F(F(F(0.4) * co2) * F(F(1.0) - c3psn)))           # :5102
    rlb = F(rb / cf)                                           # :5106
    cea = F(max(F(F(F(F(0.25) * ei) * c3psn)
                  + F(F(F(0.40) * ei) * F(F(1.0) - c3psn))),
                F(min(ea, ei))))                               # :5110

    for _ in range(STOMATA_NITER):                             # :5114
        clipped = F(max(F(ci - cp), F(0.0)))
        wj = F(F(F(F(clipped * j) / F(ci + F(F(2.0) * cp))) * c3psn)
               + F(j * F(F(1.0) - c3psn)))                     # :5115
        wc = F(F(F(F(clipped * vcmx) / F(ci + awc)) * c3psn)
               + F(vcmx * F(F(1.0) - c3psn)))                  # :5116
        we = F(F(F(F(0.5) * vcmx) * c3psn)
               + F(F(F(F(F(4000.0) * vcmx) * ci) / sfcprs)
                   * F(F(1.0) - c3psn)))                       # :5117
        psn = F(F(min(min(wj, wc), we)) * igs)                 # :5118

        cs = F(max(F(co2 - F(F(F(F(1.37) * rlb) * sfcprs) * psn)), mpe))
        a = F(F(F(F(F(F(mp * psn) * sfcprs) * cea) / F(cs * ei))) + bp)
        b = F(F(F(F(F(F(mp * psn) * sfcprs) / cs) + bp) * rlb) - F(1.0))
        c = F(-rlb)
        disc = _sqrtf(F(F(b * b) - F(F(F(4.0) * a) * c)))
        if b >= F(0.0):                                        # :5124
            q = F(F(-0.5) * F(b + disc))
        else:                                                  # :5126
            q = F(F(-0.5) * F(b - disc))
        r1 = F(q / a)                                          # :5129
        r2 = F(c / q)                                          # :5130
        rs = F(max(r1, r2))                                    # :5131
        ci = F(max(F(cs - F(F(F(psn * sfcprs) * F(1.65)) * rs)),
                   F(0.0)))                                    # :5132

    rs = F(rs * cf)                                            # :5136
    return rs, psn


# ---------------------------------------------------------------------------
# Flat-slot adapters.  These mirror
# ``tools/noahmp_wrf461_oracle/run_fluxprep.F90`` exactly, so the oracle CSV is
# replayed slot for slot with no repacking on either side.
# ---------------------------------------------------------------------------

def _eval_ragrb(x, ix):
    out = ragrb(int(ix[0]), x[0], x[1], x[2], x[4], x[5], x[6], x[7], x[8],
                x[9], x[10], x[11], x[12], x[13], x[15], x[16])
    return np.asarray(out, dtype=F)


def _eval_sfcdif1(x, ix):
    out = sfcdif1(int(ix[0]), int(ix[1]), x[0], x[1], x[2], x[3], x[4], x[5],
                  x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14],
                  x[15])
    return np.asarray((out[0], F(out[1]), *out[2:]), dtype=F)


def _eval_stomata(x, ix):
    out = stomata(*[x[i] for i in range(25)])
    return np.asarray(out, dtype=F)


FLUXPREP_EVALUATORS: Mapping[
    str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "ragrb": _eval_ragrb,
    "sfcdif1": _eval_sfcdif1,
    "stomata": _eval_stomata,
}
