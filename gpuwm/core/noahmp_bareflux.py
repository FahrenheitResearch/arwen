"""Bit-exact float32 transcription of WRF v4.6.1 Noah-MP ``BARE_FLUX``.

Source of truth
---------------
``phys/module_sf_noahmplsm.F`` at commit d66e442fccc04111067e29274c9f9eaccc3cef28
of the pinned WRF v4.6.1 checkout, sha256
``bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282``.
``BARE_FLUX`` occupies lines 4174-4479; the two module procedures it calls on
the live path, ``SFCDIF1`` (4583-4743) and ``ESAT`` (4952-5001), are
transcribed here as well because they are inlined into the same acceptance
gate.

What "bit-exact" means here
---------------------------
``kind_phys == kind(1.0)`` is FP32.  Every arithmetic result below is rounded
to float32 through :func:`~gpuwm.core.noahmp_libm.f32` at exactly the points
gfortran rounds, and every expression is grouped the way Fortran's
left-to-right associativity for equal-precedence operators groups it.  Where
the Fortran writes ``LOG``, ``ATAN`` or ``**0.25`` on a REAL(4), gfortran emits
a call to glibc ``logf`` / ``atanf`` / ``powf``; those are *not* correctly
rounded and are taken from :mod:`gpuwm.core.noahmp_libm`, which reproduces
glibc 2.39's own algorithms.  ``SQRT`` is the hardware ``sqrtss``, which is
correctly rounded, so :func:`math.sqrt` on a float32-rounded operand matches.

Pinned option identity
----------------------
The WRF Registry defaults select ``opt_sfc = 1`` and ``opt_stc = 1``.  Under
that identity:

* ``SFCDIF2`` is dead -- ``OPT_SFC == 2`` never runs, so neither does the
  ``CH = CH/UR`` rescale nor the ``SNOWH > 0`` clamp of CM/CH that follows it.
* the ``OPT_STC == 3`` snow-melt blend ``TGB = (1-FSNO)*TGB + FSNO*TFRZ`` is
  dead, which is why ``FSNO`` has no effect on any output.
* ``NITERB`` is a ``DATA``-initialised local fixed at 5 (line 4329; the
  ``DATA NITERB /3/`` above it is commented out), so the stability loop count
  is a compile-time constant, not an option.

Both are asserted rather than branched on: :func:`bare_flux` raises if it is
handed any other option pair.

Arguments that do not reach any output
--------------------------------------
``DT``, ``THAIR``, ``Q2``, ``DX``, ``DZ8W``, ``QC``, ``SFCPRS``, ``IVGTYP``,
``ILOC``, ``JLOC`` and ``FSNO`` are accepted and ignored: BARE_FLUX either
never reads them, or reads them only inside a branch that the pinned option
identity kills.  ``CM``, ``CH`` and ``QSFC`` are declared INTENT(INOUT) but
their incoming values are never read -- SFCDIF1 writes CM/CH as INTENT(OUT) on
every iteration and QSFC is assigned before any use.  They are kept in the
signature so the call site matches WRF's, and the mutation study in
``tests/test_noahmp_bareflux.py`` proves each of these claims against the
fixture rather than asserting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gpuwm.core.noahmp_libm import atanf, f32, logf, powf

__all__ = [
    "BareFluxOut",
    "bare_flux",
    "esat",
    "sfcdif1",
    "GRAV", "SB", "VKC", "TFRZ", "CPAIR",
]

# --------------------------------------------------------------------------
# module_sf_noahmplsm.F lines 204-220.  Written as the decimal literals the
# Fortran uses; f32() applies the same rounding the compiler does.
# --------------------------------------------------------------------------
GRAV = f32(9.80616)
SB = f32(5.67e-08)
VKC = f32(0.40)
TFRZ = f32(273.16)
CPAIR = f32(1004.64)

_MPE = f32(1e-6)


def _fmin(a: float, b: float) -> float:
    """Fortran ``MIN`` on two REAL(4)s, as gfortran's ``minss`` computes it.

    ``minss`` returns its second operand whenever the first is not strictly
    less, which is what decides the answer when the operands are ``-0.0`` and
    ``+0.0``.  Spelling it this way rather than as ``min()`` keeps that case
    pinned instead of left to Python's tie-breaking.
    """
    return a if a < b else b


def _fmax(a: float, b: float) -> float:
    """Fortran ``MAX`` on two REAL(4)s, as gfortran's ``maxss`` computes it."""
    return a if a > b else b


def _powi3(x: float) -> float:
    """``x**3`` as gfortran expands ``__builtin_powi(x, 3)``: ``(x*x)*x``."""
    x2 = f32(x * x)
    return f32(x2 * x)


def _powi4(x: float) -> float:
    """``x**4`` as gfortran expands ``__builtin_powi(x, 4)``: ``(x*x)*(x*x)``."""
    x2 = f32(x * x)
    return f32(x2 * x2)


# --------------------------------------------------------------------------
# ESAT -- module_sf_noahmplsm.F 4952-5001
# --------------------------------------------------------------------------
_ESAT_A = (f32(6.107799961), f32(4.436518521e-01), f32(1.428945805e-02),
           f32(2.650648471e-04), f32(3.031240396e-06), f32(2.034080948e-08),
           f32(6.136820929e-11))
_ESAT_B = (f32(6.109177956), f32(5.034698970e-01), f32(1.886013408e-02),
           f32(4.176223716e-04), f32(5.824720280e-06), f32(4.838803174e-08),
           f32(1.838826904e-10))
_ESAT_C = (f32(4.438099984e-01), f32(2.857002636e-02), f32(7.938054040e-04),
           f32(1.215215065e-05), f32(1.036561403e-07), f32(3.532421810e-10),
           f32(-7.090244804e-13))
_ESAT_D = (f32(5.030305237e-01), f32(3.773255020e-02), f32(1.267995369e-03),
           f32(2.477563108e-05), f32(3.005693132e-07), f32(2.158542548e-09),
           f32(7.131097725e-12))


def _horner7(c, t: float) -> float:
    """``100.0*(c0+T*(c1+T*(c2+T*(c3+T*(c4+T*(c5+T*c6))))))`` in float32."""
    y = f32(c[5] + f32(t * c[6]))
    y = f32(c[4] + f32(t * y))
    y = f32(c[3] + f32(t * y))
    y = f32(c[2] + f32(t * y))
    y = f32(c[1] + f32(t * y))
    y = f32(c[0] + f32(t * y))
    return f32(100.0 * y)


def esat(t: float) -> tuple[float, float, float, float]:
    """Noah-MP ``ESAT``: returns ``(ESW, ESI, DESW, DESI)`` for T in degrees C."""
    t = f32(t)
    return (_horner7(_ESAT_A, t), _horner7(_ESAT_B, t),
            _horner7(_ESAT_C, t), _horner7(_ESAT_D, t))


# --------------------------------------------------------------------------
# SFCDIF1 -- module_sf_noahmplsm.F 4583-4743
# --------------------------------------------------------------------------
@dataclass
class _Sfcdif1State:
    """The INOUT block SFCDIF1 threads through the BARE_FLUX stability loop."""

    moz: float = 0.0
    mozsgn: int = 0
    fm: float = 0.0
    fh: float = 0.0
    fm2: float = 0.0
    fh2: float = 0.0
    fv: float = 0.0
    cm: float = 0.0
    ch: float = 0.0
    ch2: float = 0.0


def sfcdif1(state: _Sfcdif1State, it: int, sfctmp: float, rhoair: float,
            h: float, qair: float, zlvl: float, zpd: float, z0m: float,
            z0h: float, ur: float, mpe: float) -> None:
    """Monin-Obukhov drag coefficients (OPT_SFC == 1).  Mutates ``state``.

    ``FM``, ``FH`` and ``FM2`` are read on iterations 2..5 only; on iteration 1
    they are written before any read, which is why WRF leaves them
    uninitialised in BARE_FLUX and why the dataclass defaults are never
    observable.
    """
    mozold = state.moz

    if zlvl <= zpd:
        raise ValueError("SFCDIF1: ZLVL <= ZPD; WRF calls wrf_error_fatal here")

    tmpcm = logf(f32(f32(zlvl - zpd) / z0m))
    tmpch = logf(f32(f32(zlvl - zpd) / z0h))
    tmpcm2 = logf(f32(f32(2.0 + z0m) / z0m))
    tmpch2 = logf(f32(f32(2.0 + z0h) / z0h))

    if it == 1:
        state.fv = f32(0.0)
        state.moz = f32(0.0)
        mol = f32(0.0)
        moz2 = f32(0.0)
    else:
        tvir = f32(f32(1.0 + f32(f32(0.61) * qair)) * sfctmp)
        tmp1 = f32(f32(f32(VKC * f32(GRAV / tvir)) * h) / f32(rhoair * CPAIR))
        if abs(tmp1) <= mpe:
            tmp1 = mpe
        mol = f32(f32(f32(-1.0) * _powi3(state.fv)) / tmp1)
        state.moz = _fmin(f32(f32(zlvl - zpd) / mol), f32(1.0))
        moz2 = _fmin(f32(f32(2.0 + z0h) / mol), f32(1.0))

    if f32(mozold * state.moz) < 0.0:
        state.mozsgn += 1
    if state.mozsgn >= 2:
        state.moz = f32(0.0)
        state.fm = f32(0.0)
        state.fh = f32(0.0)
        moz2 = f32(0.0)
        state.fm2 = f32(0.0)
        state.fh2 = f32(0.0)

    if state.moz < 0.0:
        tmp1 = powf(f32(1.0 - f32(16.0 * state.moz)), f32(0.25))
        tmp2 = logf(f32(f32(1.0 + f32(tmp1 * tmp1)) / 2.0))
        tmp3 = logf(f32(f32(1.0 + tmp1) / 2.0))
        fmnew = f32(f32(f32(f32(2.0 * tmp3) + tmp2)
                        - f32(2.0 * atanf(tmp1))) + f32(1.5707963))
        fhnew = f32(2.0 * tmp2)

        tmp12 = powf(f32(1.0 - f32(16.0 * moz2)), f32(0.25))
        tmp22 = logf(f32(f32(1.0 + f32(tmp12 * tmp12)) / 2.0))
        tmp32 = logf(f32(f32(1.0 + tmp12) / 2.0))
        fm2new = f32(f32(f32(f32(2.0 * tmp32) + tmp22)
                         - f32(2.0 * atanf(tmp12))) + f32(1.5707963))
        fh2new = f32(2.0 * tmp22)
    else:
        fmnew = f32(f32(-5.0) * state.moz)
        fhnew = fmnew
        fm2new = f32(f32(-5.0) * moz2)
        fh2new = fm2new

    if it == 1:
        state.fm = fmnew
        state.fh = fhnew
        state.fm2 = fm2new
        state.fh2 = fh2new
    else:
        state.fm = f32(f32(0.5) * f32(state.fm + fmnew))
        state.fh = f32(f32(0.5) * f32(state.fh + fhnew))
        state.fm2 = f32(f32(0.5) * f32(state.fm2 + fm2new))
        state.fh2 = f32(f32(0.5) * f32(state.fh2 + fh2new))

    state.fh = _fmin(state.fh, f32(f32(0.9) * tmpch))
    state.fm = _fmin(state.fm, f32(f32(0.9) * tmpcm))
    state.fh2 = _fmin(state.fh2, f32(f32(0.9) * tmpch2))
    state.fm2 = _fmin(state.fm2, f32(f32(0.9) * tmpcm2))

    cmfm = f32(tmpcm - state.fm)
    chfh = f32(tmpch - state.fh)
    cm2fm2 = f32(tmpcm2 - state.fm2)
    ch2fh2 = f32(tmpch2 - state.fh2)
    if abs(cmfm) <= mpe:
        cmfm = mpe
    if abs(chfh) <= mpe:
        chfh = mpe
    if abs(cm2fm2) <= mpe:
        cm2fm2 = mpe
    if abs(ch2fh2) <= mpe:
        ch2fh2 = mpe

    state.cm = f32(f32(VKC * VKC) / f32(cmfm * cmfm))
    # WRF really does divide by CMFM*CHFH, not CHFH*CHFH, for CH.
    state.ch = f32(f32(VKC * VKC) / f32(cmfm * chfh))

    state.fv = f32(ur * f32(math.sqrt(state.cm)))
    state.ch2 = f32(f32(VKC * state.fv) / ch2fh2)


# --------------------------------------------------------------------------
# BARE_FLUX -- module_sf_noahmplsm.F 4174-4479
# --------------------------------------------------------------------------
NITERB = 5  # DATA NITERB /5/, line 4329


@dataclass
class BareFluxOut:
    """Everything BARE_FLUX writes: the three INOUTs plus the nine OUTs."""

    tgb: float
    cm: float
    ch: float
    qsfc: float
    tauxb: float
    tauyb: float
    irb: float
    shb: float
    evb: float
    ghb: float
    t2mb: float
    q2b: float
    ehb2: float


def _tdc(t: float) -> float:
    """``TDC(T) = MIN(50.0, MAX(-50.0, T-TFRZ))`` -- the statement function."""
    return _fmin(f32(50.0), _fmax(f32(-50.0), f32(t - TFRZ)))


def bare_flux(*, nsnow: int, nsoil: int, isnow: int, dt: float, sag: float,
              lwdn: float, ur: float, uu: float, vv: float, sfctmp: float,
              thair: float, qair: float, eair: float, rhoair: float,
              snowh: float, dzsnso, zlvl: float, zpd: float, z0m: float,
              fsno: float, emg: float, stc, df, rsurf: float, lathea: float,
              gamma: float, rhsur: float, iloc: int, jloc: int, q2: float,
              pahb: float, tgb: float, cm: float, ch: float, dx: float,
              dz8w: float, ivgtyp: int, qc: float, qsfc: float, psfc: float,
              sfcprs: float, urban_flag: bool,
              opt_sfc: int = 1, opt_stc: int = 1) -> BareFluxOut:
    """Newton-Raphson ground temperature and surface fluxes for bare soil.

    ``dzsnso``, ``stc`` and ``df`` are sequences indexed from ``-nsnow+1`` to
    ``nsoil``; pass them as plain 0-based sequences of length
    ``nsnow + nsoil`` in that order, i.e. element 0 is the ``-nsnow+1`` layer.
    Only the ``isnow+1`` element of each is read.
    """
    if opt_sfc != 1:
        raise NotImplementedError(
            "opt_sfc != 1 selects SFCDIF2, which the WRF Registry default "
            "(opt_sfc = 1) makes dead code; it is deliberately not ported"
        )
    if opt_stc != 1:
        raise NotImplementedError(
            "opt_stc != 1 selects the opt_stc == 3 snow-melt blend (or "
            "disables the snow reset entirely); the WRF Registry default is "
            "opt_stc = 1 and the other arms are deliberately not ported"
        )

    def layer(seq, k: int) -> float:
        """Fortran ``seq(k)`` for an array declared ``(-NSNOW+1:NSOIL)``."""
        return f32(seq[k + nsnow - 1])

    tgb = f32(tgb)
    mpe = _MPE
    h = f32(0.0)
    st = _Sfcdif1State(moz=f32(0.0), mozsgn=0, fh2=f32(0.0), fv=f32(0.1))

    cir = f32(emg * SB)
    cgh = f32(f32(2.0 * layer(df, isnow + 1)) / layer(dzsnso, isnow + 1))

    # These four survive the loop and are read after it.
    csh = cev = estg = ehb = f32(0.0)
    z0h = f32(z0m)
    irb = shb = evb = ghb = f32(0.0)
    qsfc_out = f32(qsfc)

    for it in range(1, NITERB + 1):
        z0h = f32(z0m)  # both arms of the ITER==1 test assign Z0M

        sfcdif1(st, it, f32(sfctmp), f32(rhoair), h, f32(qair),
                f32(zlvl), f32(zpd), f32(z0m), z0h, f32(ur), mpe)

        ramb = _fmax(f32(1.0), f32(1.0 / f32(st.cm * ur)))
        rahb = _fmax(f32(1.0), f32(1.0 / f32(st.ch * ur)))
        rawb = rahb
        ehb = f32(1.0 / rahb)

        t = _tdc(tgb)
        esatw, esati, dsatw, dsati = esat(t)
        if t > 0.0:
            estg = esatw
            destg = dsatw
        else:
            estg = esati
            destg = dsati

        csh = f32(f32(rhoair * CPAIR) / rahb)
        cev = f32(f32(f32(rhoair * CPAIR) / gamma) / f32(rsurf + rawb))

        irb = f32(f32(cir * _powi4(tgb)) - f32(emg * lwdn))
        shb = f32(csh * f32(tgb - sfctmp))
        evb = f32(cev * f32(f32(estg * rhsur) - eair))
        ghb = f32(cgh * f32(tgb - layer(stc, isnow + 1)))

        b = f32(f32(f32(f32(f32(sag - irb) - shb) - evb) - ghb) + pahb)
        cir4t3 = f32(f32(4.0 * cir) * _powi3(tgb))
        a = f32(f32(f32(cir4t3 + csh) + f32(cev * destg)) + cgh)
        dtg = f32(b / a)

        irb = f32(irb + f32(cir4t3 * dtg))
        shb = f32(shb + f32(csh * dtg))
        evb = f32(evb + f32(f32(cev * destg) * dtg))
        ghb = f32(ghb + f32(cgh * dtg))

        tgb = f32(tgb + dtg)

        h = f32(csh * f32(tgb - sfctmp))

        t = _tdc(tgb)
        esatw, esati, _, _ = esat(t)
        estg = esatw if t > 0.0 else esati
        er = f32(estg * rhsur)
        qsfc_out = f32(f32(f32(0.622) * er) / f32(psfc - f32(f32(0.378) * er)))
        # QFX is computed here in WRF and never used again; omitted.

    # opt_stc == 1: reset TG to freezing over deep snow and rebalance.
    if snowh > f32(0.05) and tgb > TFRZ:
        tgb = TFRZ
        irb = f32(f32(cir * _powi4(tgb)) - f32(emg * lwdn))
        shb = f32(csh * f32(tgb - sfctmp))
        evb = f32(cev * f32(f32(estg * rhsur) - eair))
        ghb = f32(f32(sag + pahb) - f32(f32(irb + shb) + evb))

    tauxb = f32(-f32(f32(f32(rhoair * st.cm) * ur) * uu))
    tauyb = f32(-f32(f32(f32(rhoair * st.cm) * ur) * vv))

    # opt_sfc == 1: 2 m diagnostics.  WRF assigns EHB2 twice; the first
    # assignment is dead and is not reproduced.
    ehb2 = f32(f32(st.fv * VKC) / f32(logf(f32(f32(2.0 + z0h) / z0h)) - st.fh2))
    cq2b = ehb2
    if ehb2 < f32(1.0e-5):
        t2mb = tgb
        q2b = qsfc_out
    else:
        t2mb = f32(tgb - f32(f32(f32(shb / f32(rhoair * CPAIR)) * 1.0) / ehb2))
        q2b = f32(qsfc_out - f32(f32(evb / f32(lathea * rhoair))
                                 * f32(f32(1.0 / cq2b) + rsurf)))
    if urban_flag:
        q2b = qsfc_out

    # "update CH" -- the last statement of BARE_FLUX overwrites the INOUT CH
    # with the conductance EHB from the final stability iteration.
    return BareFluxOut(
        tgb=tgb, cm=st.cm, ch=ehb,
        qsfc=qsfc_out, tauxb=tauxb, tauyb=tauyb, irb=irb, shb=shb, evb=evb,
        ghb=ghb, t2mb=t2mb, q2b=q2b, ehb2=ehb2,
    )
