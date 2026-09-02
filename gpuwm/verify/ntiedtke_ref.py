"""NumPy float32 mirror of New Tiedtke's column preparation (cu_physics=16).

Mirror authority for ``gpuwm/core/kernels/ntiedtke.cu``.  Transcribes
``phys/module_cu_ntiedtke.F:391-455`` (``cu_ntiedtke_pre_run``) and
``phys/physics_mmm/cu_ntiedtke.F90:228-277`` (the ``scale_fac`` block and
the mass-flux variable conversion), WRF v4.6.1, digests pinned in
``tools/ntiedtke_wrf461_oracle/README.md``.

THE VERTICAL IS INVERTED HERE AND NOWHERE ELSE IN gpuwm's CUMULUS CODE.
WRF hands the driver a bottom-up column (k = kts is the surface).
``cu_ntiedtke_pre_run`` reverses it, so everything from ``cu_ntiedtke_run``
downward runs TOP-DOWN: k = 0 is the model top and k = nz-1 is the surface.
That is the ECMWF convention.  ``module_cu_gf_deep.F`` and
``module_cu_kfeta.F`` are both bottom-up, so this is the one structural
inversion in the port, and its failure mode is SILENT -- an upside-down
column produces finite, plausible numbers rather than a crash.  It is
graded first for exactly that reason.

EVERY ARITHMETIC OPERATION IS PINNED.  ``float32(a) * float32(b)`` in NumPy
is a single correctly-rounded multiply, and the kernel spells the same
expression with ``__fmul_rn`` / ``__fadd_rn`` / ``__fmaf_rn``, which NVIDIA
guarantees are never merged.  The association is NOT free: ptxas contracts
by local register pressure, so a runtime branch can leave two clones of the
same arithmetic rounding differently.  ``dot`` below is the live example --
``-0.5*grav*rho*(w_k + w_k1)`` is a left-to-right chain of three multiplies
over one add, and evaluating it as an FMA moves bits.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["np_ntiedtke_prep", "np_ntiedtke_scale_fac", "NT_DXREF"]

#: ``cu_ntiedtke.F90:230``.  The branch is ``<``, not ``<=``, so dx == DXREF
#: takes the ELSE arm; the factor is DISCONTINUOUS here.
NT_DXREF = np.float32(15000.0)

_F = np.float32


def np_ntiedtke_scale_fac(dx):
    """``scale_fac`` and ``scale_fac2`` (``cu_ntiedtke.F90:230-238``).

    ``(1.06133 + log(dxref/dx))**3`` below the reference spacing, and
    ``1 + 1.33e-5*dx`` at or above it.  Two properties a port gets wrong:
    the comparison is strict, and the coarse branch is INCREASING in dx, so
    the factor is not monotonic across the join (27 km damps more than
    15 km).
    """
    dx = _F(dx)
    if dx < NT_DXREF:
        # LOG MUST NOT BE np.log ON A float32.  The reference calls glibc's
        # logf, which (sysdeps/ieee754/flt-32/e_logf.c, the ARM
        # optimized-routines import) evaluates its polynomial in DOUBLE and
        # rounds once to float.  NumPy's float32 log is a separate
        # single-precision SIMD path and is several ULP away: MEASURED on
        # this fixture, np.log(float32) disagrees with the oracle at
        # dx = 9000 and dx = 13500 (5 and 4 ULP) while agreeing at the other
        # four, which is exactly the shape of a bug that a coarser sweep
        # would have missed.
        #
        # math.log is the C library's double log; rounding that once to
        # float32 reproduces glibc's logf on every argument in this fixture.
        # That is a MODEL of logf, not logf -- glibc's is ~0.9 ULP and not
        # correctly rounded, so the two are not identical everywhere -- and
        # it is verified here rather than assumed.  The KERNEL has no such
        # caveat: it calls gfk_log, which IS glibc's own algorithm.
        # gf_ref.py models tgammaf the same way for the same reason.
        base = _F(_F(1.06133) + _F(math.log(float(_F(NT_DXREF / dx)))))
        # **3 on an INTEGER exponent: gfortran emits a multiply chain, not
        # powf.  Left to right, so (base*base)*base.
        sf = _F(_F(base * base) * base)
        # **0.5 on a real exponent is powf in principle, but gfortran folds
        # the 0.5 case to sqrt; both are correctly rounded, so they agree.
        sf2 = _F(np.sqrt(sf))
    else:
        sf = _F(_F(1.0) + _F(_F(1.33e-5) * dx))
        sf2 = _F(1.0)
    return sf, sf2


def np_ntiedtke_prep(
    *,
    t3d, qv3d, qc3d, qi3d, u3d, v3d,
    pcps, p8w, dz8w, rho3d, w,
    qvften, thften,
    xland, hfx, qfx, dx,
    dt, stepcu=1, itimestep=2, grav=_F(9.81),
):
    """One column of ``cu_ntiedtke_pre_run``, WRF order in, scheme order out.

    Half-level inputs are ``(nz,)``; ``p8w`` and ``w`` are ``(nz+1,)``.
    Returns a dict of the arrays ``cu_ntiedtke_run`` is called with, all
    float32 and all TOP-DOWN.
    """
    t3d, qv3d, qc3d, qi3d = map(_np32, (t3d, qv3d, qc3d, qi3d))
    u3d, v3d = _np32(u3d), _np32(v3d)
    pcps, dz8w, rho3d = _np32(pcps), _np32(dz8w), _np32(rho3d)
    p8w, w = _np32(p8w), _np32(w)
    qvften, thften = _np32(qvften), _np32(thften)
    grav = _F(grav)
    nz = t3d.shape[0]

    # module_cu_ntiedtke.F:396-398.  delt = dt*stepcu.
    delt = _F(_F(dt) * _F(stepcu))

    # :400-402.  slimsk = (abs(xland-2.)) -- a REAL expression assigned to an
    # INTEGER, so it truncates toward zero.  xland = 2 (water) gives 0 and
    # xland = 1 (land) gives 1; anything between truncates, which is why this
    # is a truncation and not a round.
    slimsk = int(abs(_F(xland) - _F(2.0)))

    # :404-411.  Interface heights by upward accumulation from zero.  The
    # accumulation order is load-bearing: zi[k+1] = zi[k] + dz[k] carries the
    # rounding of every layer below it.
    zi = np.zeros(nz + 1, dtype=np.float32)
    for k in range(nz):
        zi[k + 1] = _F(zi[k] + dz8w[k])

    # :412-417.  Layer-mean height, and the vertical pressure velocity the
    # scheme sees.  omg is the ONLY path by which grid-scale ascent reaches
    # the convection, so its rounding matters more than its size suggests.
    zl = np.empty(nz, dtype=np.float32)
    dot = np.empty(nz, dtype=np.float32)
    for k in range(nz):
        zl[k] = _F(_F(0.5) * _F(zi[k] + zi[k + 1]))
        # -0.5*grav*rho*(w_k + w_k1), left to right over one add.
        t1 = _F(_F(0.5) * grav)
        t2 = _F(t1 * rho3d[k])
        t3 = _F(w[k] + w[k + 1])
        dot[k] = _F(-_F(t2 * t3))

    # :419-445.  THE FLIP.  Fortran writes zz = kte+1-pp for interfaces and
    # zz = kte-pp for half levels, with pp running 0..; both are a full
    # reversal, which in 0-based NumPy is a [::-1] view.  Spelled as an
    # explicit reversal rather than a loop because the loop would invite an
    # off-by-one that the oracle would catch but a reader would not.
    ghti = zi[::-1].copy()
    prsi = p8w[::-1].copy()
    ghtl = zl[::-1].copy()
    omg = dot[::-1].copy()
    prsl = pcps[::-1].copy()
    tf = t3d[::-1].copy()
    qvf = qv3d[::-1].copy()
    qcf = qc3d[::-1].copy()
    qif = qi3d[::-1].copy()
    uf = u3d[::-1].copy()
    vf = v3d[::-1].copy()

    # :449-462.  The forcing tendencies are ZEROED on the first timestep
    # rather than flipped.  That is not a nicety: the nonequil closure's
    # zcape2 term is built entirely from ptte/pqte, so at itimestep == 1 the
    # deep closure loses it and answers differently.
    if itimestep == 1:
        qvftenz = np.zeros(nz, dtype=np.float32)
        thftenz = np.zeros(nz, dtype=np.float32)
    else:
        qvftenz = qvften[::-1].copy()
        thftenz = thften[::-1].copy()

    sf, sf2 = np_ntiedtke_scale_fac(dx)

    return {
        "prsl": prsl, "ghtl": ghtl, "omg": omg,
        "tf": tf, "qvf": qvf, "qcf": qcf, "qif": qif,
        "uf": uf, "vf": vf,
        "qvftenz": qvftenz, "thftenz": thftenz,
        "prsi": prsi, "ghti": ghti,
        "slimsk": slimsk, "delt": delt,
        "dx_hv": _F(dx), "hfx_hv": _F(hfx), "qfx_hv": _F(qfx),
        "scale_fac": sf, "scale_fac2": sf2,
    }


def _np32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


# ===========================================================================
# Slice 2: the conversion block, cuadjtqn (kcall = 0) and cuinin
# ===========================================================================
# cu_ntiedtke.F90:3542-3589 (the foe* functions), :3381-3398 (cuadjtqn's
# kcall = 0 arm) and :1141-1215 (cuinin).
#
# cuinin is COLUMN-UNIVERSAL: it takes no ldcum, no ktype and no ierr, its
# only flag is loflag = .true. set unconditionally, and it runs at
# cumastrn:474 -- BEFORE cutypen:490 decides the convection type.  So it is
# gradeable against the plain 108-column fixture with no trigger visibility.
# cuadjtqn's kcall = 0 arm does not even read ldflag.

_TMELT = _F(273.16)
_RTWAT = _TMELT
_RTICE = _F(_TMELT - _F(23.0))
_C1ES = _F(610.78)
_C3LES = _F(17.2693882)
_C3IES = _F(21.875)
_C4LES = _F(35.86)
_C4IES = _F(7.66)


def _exp32(x):
    """glibc's expf, modelled the way np_ntiedtke_scale_fac models logf --
    double evaluation rounded once.  np.exp on a float32 is NumPy's own
    single-precision path and is several ULP away; the KERNEL calls
    gfk_exp, which is glibc's algorithm and needs no model."""
    return _F(math.exp(float(x)))


#: cu_ntiedtke_common:17,32 -- parameters, so literals here.
_T13 = _F(_F(1.0) / _F(3.0))
_ZDNOPRC = _F(2.0e4)


def _pow32(x, y):
    """glibc's powf, modelled the same way expf and logf are: evaluate in
    double, round once.  The kernel calls gfk_pow, which IS glibc's."""
    return _F(math.pow(float(x), float(y)))

class NtConstants:
    """What cu_ntiedtke_init (:100-118) derives from the caller's constants.

    New Tiedtke runs on WRF's own constants -- unlike Grell-Freitas, which
    takes its own from module_gfs_physcons and disagrees with WRF -- so
    there is no second set to reconcile.
    """

    def __init__(self, cp=_F(1004.5), rd=_F(287.0), rv=_F(461.6),
                 xlv=_F(2.5e6), xlf=_F(3.50e5), grav=_F(9.81)):
        self.cpd = _F(cp)
        self.rd, self.rv, self.g = _F(rd), _F(rv), _F(grav)
        self.alv, self.alf = _F(xlv), _F(xlf)
        self.als = _F(_F(xlv) + _F(xlf))
        self.rcpd = _F(_F(1.0) / _F(cp))
        self.c2es = _F(_F(_C1ES * _F(rd)) / _F(rv))
        self.c5les = _F(_C3LES * _F(_TMELT - _C4LES))
        self.c5ies = _F(_C3IES * _F(_TMELT - _C4IES))
        self.r5alvcp = _F(_F(self.c5les * self.alv) * self.rcpd)
        self.r5alscp = _F(_F(self.c5ies * self.als) * self.rcpd)
        self.ralvdcp = _F(self.alv * self.rcpd)
        self.ralsdcp = _F(self.als * self.rcpd)
        self.ralfdcp = _F(self.alf * self.rcpd)
        self.vtmpc1 = _F(_F(_F(rv) / _F(rd)) - _F(1.0))
        self.zrg = _F(_F(1.0) / _F(grav))


def nt_foealfa(tt):
    """:3542-3556.  1 over water, 0 over ice, quadratic in between."""
    tt = _F(tt)
    num = _F(_F(max(_RTICE, min(_RTWAT, tt))) - _RTICE)
    r = _F(num / _F(_RTWAT - _RTICE))
    return _F(min(_F(1.0), _F(r * r)))


def nt_foeewm(tt, c):
    """:3566-3573.  Saturation vapour pressure over the mixed-phase ramp."""
    tt = _F(tt)
    a = nt_foealfa(tt)
    el = _exp32(_F(_F(_C3LES * _F(tt - _TMELT)) / _F(tt - _C4LES)))
    ei = _exp32(_F(_F(_C3IES * _F(tt - _TMELT)) / _F(tt - _C4IES)))
    return _F(c.c2es * _F(_F(a * el) + _F(_F(_F(1.0) - a) * ei)))


def nt_foedem(tt, c):
    """:3576-3580."""
    tt = _F(tt)
    a = nt_foealfa(tt)
    dl = _F(tt - _C4LES)
    di = _F(tt - _C4IES)
    return _F(_F(_F(a * c.r5alvcp) * _F(_F(1.0) / _F(dl * dl)))
              + _F(_F(_F(_F(1.0) - a) * c.r5alscp)
                   * _F(_F(1.0) / _F(di * di))))


def nt_foeldcpm(tt, c):
    """:3583-3588."""
    a = nt_foealfa(tt)
    return _F(_F(a * c.ralvdcp) + _F(_F(_F(1.0) - a) * c.ralsdcp))


def np_ntiedtke_convert(*, tf, qvf, uf, vf, omg, ghtl, ghti, prsl,
                        qvftenz, thftenz, c=None):
    """cu_ntiedtke_run's variable conversion (:240-277).

    Everything is already TOP-DOWN here -- this runs after the flip.
    pgeoh gets its km1 entry in the scalar loop BEFORE the level loop fills
    1..km, which matters because nothing else writes it.
    """
    c = NtConstants() if c is None else c
    nz = tf.shape[0]
    ztp1 = _np32(tf).copy()
    zqp1 = np.empty(nz, dtype=np.float32)
    zqsat = np.empty(nz, dtype=np.float32)
    pgeo = np.empty(nz, dtype=np.float32)
    pgeoh = np.empty(nz + 1, dtype=np.float32)
    pgeoh[nz] = _F(c.g * _F(ghti[nz]))
    for k in range(nz):
        zqp1[k] = _F(_F(qvf[k]) / _F(_F(1.0) + _F(qvf[k])))
        pgeo[k] = _F(c.g * _F(ghtl[k]))
        pgeoh[k] = _F(c.g * _F(ghti[k]))
        zqs = _F(nt_foeewm(ztp1[k], c) / _F(prsl[k]))
        zqs = _F(min(_F(0.5), zqs))
        zcor = _F(_F(1.0) / _F(_F(1.0) - _F(c.vtmpc1 * zqs)))
        zqsat[k] = _F(zqs * zcor)
    return {"ztp1": ztp1, "zqp1": zqp1, "zqsat": zqsat, "pgeo": pgeo,
            "pgeoh": pgeoh, "pum1": _np32(uf).copy(),
            "pvm1": _np32(vf).copy(), "pverv": _np32(omg).copy(),
            "ptte": _np32(thftenz).copy(), "pqte": _np32(qvftenz).copy()}


def nt_cuadjtqn0(pt, pq, k, psp, c):
    """cuadjtqn's kcall == 0 arm (:3381-3398), in place at level k.

    Two identical Newton passes.  This arm never reads ldflag -- it adjusts
    every column unconditionally, which is part of why cuinin needs no
    trigger visibility.
    """
    zqp = _F(_F(1.0) / _F(psp))
    for _ in range(2):
        zqsat = _F(nt_foeewm(pt[k], c) * zqp)
        zqsat = _F(min(_F(0.5), zqsat))
        zcor = _F(_F(1.0) / _F(_F(1.0) - _F(c.vtmpc1 * zqsat)))
        zqsat = _F(zqsat * zcor)
        den = _F(_F(1.0) + _F(_F(zqsat * zcor) * nt_foedem(pt[k], c)))
        zcond1 = _F(_F(pq[k] - zqsat) / den)
        pt[k] = _F(pt[k] + _F(nt_foeldcpm(pt[k], c) * zcond1))
        pq[k] = _F(pq[k] - zcond1)


def np_ntiedtke_cuinin(*, pten, pqen, pqsen, puen, pven, pverv, pgeo,
                       paph, pgeoh, c=None):
    """cuinin (:1141-1215), one column, 0-based.

    Fortran is 1-based here and the index arithmetic is load-bearing, so
    the loop bounds carry their Fortran form in comments.  klev is nz;
    klevm1 is nz-1.

    pqsenh[0] IS NEVER WRITTEN.  The jk loop starts at 2 (1-based) and the
    tail block sets only ptenh(1) and pqenh(1).  It is left undefined in
    WRF too -- a cumastrn local that nothing downstream reads -- so it is
    excluded from grading rather than invented.
    """
    c = NtConstants() if c is None else c
    nz = pten.shape[0]
    ptenh = np.zeros(nz, dtype=np.float32)
    pqenh = np.zeros(nz, dtype=np.float32)
    pqsenh = np.zeros(nz, dtype=np.float32)

    # do jk = 2, klev   ->   k = 1 .. nz-1
    for k in range(1, nz):
        a = _F(_F(c.cpd * _F(pten[k - 1])) + _F(pgeo[k - 1]))
        b = _F(_F(c.cpd * _F(pten[k])) + _F(pgeo[k]))
        ptenh[k] = _F(_F(_F(max(a, b)) - _F(pgeoh[k])) * c.rcpd)
        pqenh[k] = _F(pqen[k - 1])
        pqsenh[k] = _F(pqsen[k - 1])
        # if ( jk >= klev-1 .or. jk < 2 ) cycle  ->  skip k >= nz-2
        if k >= nz - 2:
            continue
        nt_cuadjtqn0(ptenh, pqsenh, k, _F(paph[k]), c)
        v = _F(_F(min(_F(pqen[k - 1]), _F(pqsen[k - 1])))
               + _F(pqsenh[k] - _F(pqsen[k - 1])))
        pqenh[k] = _F(max(v, _F(0.0)))

    ptenh[nz - 1] = _F(_F(_F(_F(c.cpd * _F(pten[nz - 1])) + _F(pgeo[nz - 1]))
                          - _F(pgeoh[nz - 1])) * c.rcpd)
    pqenh[nz - 1] = _F(pqen[nz - 1])
    ptenh[0] = _F(pten[0])
    pqenh[0] = _F(pqen[0])
    klwmin = nz                      # 1-based klev
    zwmax = _F(0.0)

    # do jk = klevm1, 2, -1   ->   k = nz-2 .. 1
    for k in range(nz - 2, 0, -1):
        z1 = _F(_F(c.cpd * ptenh[k]) + _F(pgeoh[k]))
        z2 = _F(_F(c.cpd * ptenh[k + 1]) + _F(pgeoh[k + 1]))
        ptenh[k] = _F(_F(_F(max(z1, z2)) - _F(pgeoh[k])) * c.rcpd)

    # do jk = klev, 3, -1   ->   k = nz-1 .. 2; klwmin stays 1-BASED
    for k in range(nz - 1, 1, -1):
        if _F(pverv[k]) < zwmax:
            zwmax = _F(pverv[k])
            klwmin = k + 1

    ptu, ptd = ptenh.copy(), ptenh.copy()
    pqu, pqd = pqenh.copy(), pqenh.copy()
    plu = np.zeros(nz, dtype=np.float32)
    puu = np.zeros(nz, dtype=np.float32)
    pud = np.zeros(nz, dtype=np.float32)
    pvu = np.zeros(nz, dtype=np.float32)
    pvd = np.zeros(nz, dtype=np.float32)
    for k in range(nz):
        ik = 0 if k == 0 else k - 1        # ik = jk-1, ik = 1 when jk = 1
        puu[k] = pud[k] = _F(puen[ik])
        pvu[k] = pvd[k] = _F(pven[ik])
    klab = np.zeros(nz, dtype=np.int32)

    return {"ptenh": ptenh, "pqenh": pqenh, "pqsenh": pqsenh,
            "ptu": ptu, "pqu": pqu, "ptd": ptd, "pqd": pqd,
            "puu": puu, "pvu": pvu, "pud": pud, "pvd": pvd,
            "plu": plu, "klab": klab, "klwmin": klwmin}


# ===========================================================================
# Slice 3: cuadjtqn's kcall == 1 arm, and cutypen
# ===========================================================================
# cu_ntiedtke.F90:3324-3357 (cuadjtqn kcall == 1) and :1330-1748 (cutypen).
#
# cutypen is the TRIGGER: it decides ktype, and ktype is what selects which
# scale factor applies downstream (:676 scale_fac for deep, :716 scale_fac2
# for shallow).  It is the branchiest routine in the scheme and it assigns
# ktype 0, 1 and 2 ONLY -- ktype 3 (mid-level) is assigned later, in
# cubasmcn from cuascn:1968, and is not this routine's business.
#
# INDEXING.  Everything inside is kept 1-BASED, matching the Fortran, with
# arrays allocated at nz+2 and index 0 unused.  This routine is dense with
# jk+1 / jk+2 / klev-1 / levels+1 arithmetic and three different loop
# directions; translating each of those to 0-based by hand is where a
# transcription silently goes wrong.  The conversion happens once, at the
# boundary of this function, and nowhere else.

def _f1(nz):
    """A 1-based float32 column: index 1..nz usable, 0 unused."""
    return np.zeros(nz + 2, dtype=np.float32)


def _i1(nz):
    return np.zeros(nz + 2, dtype=np.int32)


def nt_cuadjtqn1(pt, pq, k, psp, c):
    """cuadjtqn's kcall == 1 arm (:3324-3357), in place at level k.

    NOT the kcall == 0 arm with a different guard: it computes saturation
    INLINE off reciprocals (`exp(c3les*(pt-tmelt)*zl)` with
    `zl = 1/(pt-c4les)`) where kcall == 0 goes through foeewm and its
    DIVISION.  A multiply by a reciprocal is not a division in float32, so
    the two arms round differently on the same argument and neither can
    stand in for the other.

    The second Newton pass is guarded twice: it only runs when the first
    condensate is positive, and its result is discarded when the first
    condensate is denormal-small.
    """
    zqp = _F(_F(1.0) / _F(psp))
    zl = _F(_F(1.0) / _F(pt[k] - _C4LES))
    zi = _F(_F(1.0) / _F(pt[k] - _C4IES))
    a = nt_foealfa(pt[k])
    el = _exp32(_F(_F(_C3LES * _F(pt[k] - _TMELT)) * zl))
    ei = _exp32(_F(_F(_C3IES * _F(pt[k] - _TMELT)) * zi))
    zqsat = _F(c.c2es * _F(_F(a * el) + _F(_F(_F(1.0) - a) * ei)))
    zqsat = _F(zqsat * zqp)
    zqsat = _F(min(_F(0.5), zqsat))
    zcor = _F(_F(1.0) - _F(c.vtmpc1 * zqsat))
    zf = _F(_F(_F(a * c.r5alvcp) * _F(zl * zl))
            + _F(_F(_F(_F(1.0) - a) * c.r5alscp) * _F(zi * zi)))
    zcond = _F(_F(_F(pq[k] * _F(zcor * zcor)) - _F(zqsat * zcor))
               / _F(_F(zcor * zcor) + _F(zqsat * zf)))
    if zcond > _F(0.0):
        pt[k] = _F(pt[k] + _F(nt_foeldcpm(pt[k], c) * zcond))
        pq[k] = _F(pq[k] - zcond)
        zl = _F(_F(1.0) / _F(pt[k] - _C4LES))
        zi = _F(_F(1.0) / _F(pt[k] - _C4IES))
        a = nt_foealfa(pt[k])
        el = _exp32(_F(_F(_C3LES * _F(pt[k] - _TMELT)) * zl))
        ei = _exp32(_F(_F(_C3IES * _F(pt[k] - _TMELT)) * zi))
        zqsat = _F(c.c2es * _F(_F(a * el) + _F(_F(_F(1.0) - a) * ei)))
        zqsat = _F(zqsat * zqp)
        zqsat = _F(min(_F(0.5), zqsat))
        zcor = _F(_F(1.0) - _F(c.vtmpc1 * zqsat))
        zf = _F(_F(_F(a * c.r5alvcp) * _F(zl * zl))
                + _F(_F(_F(_F(1.0) - a) * c.r5alscp) * _F(zi * zi)))
        zcond1 = _F(_F(_F(pq[k] * _F(zcor * zcor)) - _F(zqsat * zcor))
                    / _F(_F(zcor * zcor) + _F(zqsat * zf)))
        if abs(zcond) < _F(1.0e-20):
            zcond1 = _F(0.0)
        pt[k] = _F(pt[k] + _F(nt_foeldcpm(pt[k], c) * zcond1))
        pq[k] = _F(pq[k] - zcond1)


def _cloud_base(jk, ptu, pqu, plu, kup, klab, paph, kcbot, klev, c):
    """The exact-cloud-base block, identical in both passes (:1453-1463 and
    :1654-1685).  Picks whichever half level the LCL is nearer."""
    ik = jk + 1
    zqsu = _F(nt_foeewm(ptu[ik], c) / _F(paph[ik]))
    zqsu = _F(min(_F(0.5), zqsu))
    zcor = _F(_F(1.0) / _F(_F(1.0) - _F(c.vtmpc1 * zqsu)))
    zqsu = _F(zqsu * zcor)
    zdq = _F(min(_F(0.0), _F(pqu[ik] - zqsu)))
    zalfaw = nt_foealfa(ptu[ik])
    dl = _F(ptu[ik] - _C4LES)
    di = _F(ptu[ik] - _C4IES)
    zfacw = _F(c.c5les / _F(dl * dl))
    zfaci = _F(c.c5ies / _F(di * di))
    zfac = _F(_F(zalfaw * zfacw) + _F(_F(_F(1.0) - zalfaw) * zfaci))
    zesdp = _F(nt_foeewm(ptu[ik], c) / _F(paph[ik]))
    zcor = _F(_F(1.0) / _F(_F(1.0) - _F(c.vtmpc1 * zesdp)))
    zdqsdt = _F(_F(zfac * zcor) * zqsu)
    zdtdp = _F(_F(c.rd * ptu[ik]) / _F(c.cpd * _F(paph[ik])))
    zdp = _F(zdq / _F(zdqsdt * zdtdp))
    zcbase = _F(paph[ik] + zdp)
    zpdifftop = _F(zcbase - _F(paph[jk]))
    zpdiffbot = _F(_F(paph[jk + 1]) - zcbase)
    if zpdifftop > zpdiffbot and kup[jk + 1] > _F(0.0):
        ikb = min(klev - 1, jk + 1)
        klab[ikb] = 2
        klab[jk] = 2
        kcbot[0] = ikb
        plu[jk + 1] = _F(1.0e-8)
    elif zpdifftop <= zpdiffbot and kup[jk] > _F(0.0):
        klab[jk] = 2
        kcbot[0] = jk


def np_ntiedtke_cutypen(*, pqen, ptenh, pqenh, pqsenh, pgeoh, paph,
                        hfx, qfx, pgeo, pqsen, pap, pten, lndj,
                        cutu, cuqu, culab, culu, c=None):
    """cutypen (:1330-1748), one column.  Arrays in/out are 0-based; the
    body is 1-based.

    Returns the routine's outputs, including ldcum/ktype/cubot/cutop/kdpl/
    wbase and the REBUILT cutu/cuqu/culu/culab -- those four are intent(out)
    here, so cuinin's values are overwritten rather than consumed.
    """
    c = NtConstants() if c is None else c
    nz = pten.shape[0]
    klev, klevm1 = nz, nz - 1

    def up(a):
        """0-based float32 in -> 1-based working copy."""
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_pqen, P_ptenh, P_pqenh = up(pqen), up(ptenh), up(pqenh)
    P_pgeo, P_pqsen, P_pap, P_pten = up(pgeo), up(pqsen), up(pap), up(pten)
    P_paph, P_pgeoh = _f1(nz), _f1(nz)
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]      # paph is (nz+1)
    P_pgeoh[1:nz + 2] = _np32(pgeoh)[:nz + 1]

    o_cutu, o_cuqu, o_culu = up(cutu), up(cuqu), up(culu)
    o_culab = _i1(nz)
    o_culab[1:nz + 1] = np.asarray(culab, dtype=np.int32)

    ptu, pqu, plu = _f1(nz), _f1(nz), _f1(nz)
    dh, dhen, kup = _f1(nz), _f1(nz), _f1(nz)
    vptu, vten, zbuo, abuoy = _f1(nz), _f1(nz), _f1(nz), _f1(nz)
    klab = _i1(nz)

    kcbot, kctop = [klev], [klev]
    kdpl, ktype = klev, 0
    wbase = _F(0.0)
    ldcum = False

    # ---- shallow pass (:1332-1345) -------------------------------------
    for jk in range(1, klev + 1):
        plu[jk] = o_culu[jk]
        ptu[jk] = o_cutu[jk]
        pqu[jk] = o_cuqu[jk]
        klab[jk] = o_culab[jk]
        dh[jk] = dhen[jk] = kup[jk] = _F(0.0)
        vptu[jk] = vten[jk] = zbuo[jk] = abuoy[jk] = _F(0.0)

    lldcum, loflag = False, True
    eta = dz = coef = _F(0.0)

    for jk in range(klevm1, 1, -1):                     # klevm1 .. 2
        if jk == klevm1:
            rho = _F(_F(P_pap[klev])
                     / _F(c.rd * _F(_F(P_pten[klev])
                          * _F(_F(1.0) + _F(c.vtmpc1 * P_pqen[klev])))))
            part1 = _F(_F(_F(_F(1.5) * _F(0.4)) * P_pgeo[klev])
                       / _F(rho * P_pten[klev]))
            part2 = _F(_F(-_F(_F(hfx) * c.rcpd))
                       - _F(_F(c.vtmpc1 * P_pten[klev]) * _F(qfx)))
            root = _F(_F(0.001) - _F(part1 * part2))
            if part2 < _F(0.0):
                # conw = 1.2*root**t13 -- a genuine cube root through powf.
                conw = _F(_F(1.2) * _pow32(root, _T13))
                deltt = _F(max(_F(_F(_F(1.5) * _F(hfx))
                                  / _F(_F(rho * c.cpd) * conw)), _F(0.0)))
                deltq = _F(max(_F(_F(_F(1.5) * _F(qfx)) / _F(rho * conw)),
                               _F(0.0)))
                kup[klev] = _F(_F(0.5) * _F(conw * conw))
                pqu[klev] = _F(P_pqenh[klev] + deltq)
                dhen[klev] = _F(P_pgeoh[klev] + _F(P_ptenh[klev] * c.cpd))
                dh[klev] = _F(dhen[klev] + _F(deltt * c.cpd))
                ptu[klev] = _F(_F(dh[klev] - P_pgeoh[klev]) * c.rcpd)
                vptu[klev] = _F(ptu[klev]
                                * _F(_F(1.0) + _F(c.vtmpc1 * pqu[klev])))
                vten[klev] = _F(P_ptenh[klev]
                                * _F(_F(1.0) + _F(c.vtmpc1 * P_pqenh[klev])))
                zbuo[klev] = _F(_F(vptu[klev] - vten[klev]) / vten[klev])
                klab[klev] = 1
            else:
                loflag = False
        if not loflag:
            break                                        # is == 0 -> exit

        eta = _F(_F(_F(0.8) / _F(P_pgeo[jk] * c.zrg)) + _F(2.0e-4))
        dz = _F(_F(P_pgeoh[jk] - P_pgeoh[jk + 1]) * c.zrg)
        coef = _F(_F(_F(0.5) * eta) * dz)
        dhen[jk] = _F(P_pgeoh[jk] + _F(c.cpd * P_ptenh[jk]))
        dh[jk] = _F(_F(_F(coef * _F(dhen[jk + 1] + dhen[jk]))
                       + _F(_F(_F(1.0) - coef) * dh[jk + 1]))
                    / _F(_F(1.0) + coef))
        pqu[jk] = _F(_F(_F(coef * _F(P_pqenh[jk + 1] + P_pqenh[jk]))
                        + _F(_F(_F(1.0) - coef) * pqu[jk + 1]))
                     / _F(_F(1.0) + coef))
        ptu[jk] = _F(_F(dh[jk] - P_pgeoh[jk]) * c.rcpd)
        zqold = pqu[jk]
        nt_cuadjtqn1(ptu, pqu, jk, P_paph[jk], c)

        zdq = _F(max(_F(zqold - pqu[jk]), _F(0.0)))
        plu[jk] = _F(plu[jk + 1] + zdq)
        zlglac = _F(zdq * _F(_F(_F(1.0) - nt_foealfa(ptu[jk]))
                             - _F(_F(1.0) - nt_foealfa(ptu[jk + 1]))))
        plu[jk] = _F(min(plu[jk], _F(5.0e-3)))
        dh[jk] = _F(P_pgeoh[jk]
                    + _F(c.cpd * _F(ptu[jk] + _F(c.ralfdcp * zlglac))))
        vptu[jk] = _F(_F(ptu[jk] * _F(_F(_F(1.0) + _F(c.vtmpc1 * pqu[jk]))
                                      - plu[jk]))
                      + _F(c.ralfdcp * zlglac))
        vten[jk] = _F(P_ptenh[jk] * _F(_F(1.0)
                                       + _F(c.vtmpc1 * P_pqenh[jk])))
        zbuo[jk] = _F(_F(vptu[jk] - vten[jk]) / vten[jk])
        abuoy[jk] = _F(_F(_F(zbuo[jk] + zbuo[jk + 1]) * _F(0.5)) * c.g)
        atop1 = _F(_F(1.0) - _F(_F(2.0) * coef))
        atop2 = _F(_F(_F(2.0) * dz) * abuoy[jk])
        abot = _F(_F(1.0) + _F(_F(2.0) * coef))
        kup[jk] = _F(_F(_F(atop1 * kup[jk + 1]) + atop2) / abot)

        if plu[jk] > _F(0.0) and klab[jk + 1] == 1:
            _cloud_base(jk, ptu, pqu, plu, kup, klab, P_paph, kcbot, klev, c)

        if kup[jk] < _F(0.0):
            loflag = False
            if plu[jk + 1] > _F(0.0):
                kctop[0] = jk
                lldcum = True
            else:
                lldcum = False
        else:
            klab[jk] = 2 if plu[jk] > _F(0.0) else 1

    ikb, ikt = kcbot[0], kctop[0]
    if _F(P_paph[ikb] - P_paph[ikt]) > _ZDNOPRC:
        lldcum = False
    if lldcum:
        ktype, ldcum = 2, True
        wbase = _F(np.sqrt(_F(max(_F(_F(2.0) * kup[ikb]), _F(0.0)))))
        cubot, cutop, kdpl = ikb, ikt, klev
    else:
        cutop, cubot, kdpl = -1, -1, klev - 1
        ldcum, wbase = False, _F(0.0)

    for jk in range(klev, 0, -1):
        if jk >= kctop[0]:
            o_culab[jk] = klab[jk]
            o_cutu[jk] = ptu[jk]
            o_cuqu[jk] = pqu[jk]
            o_culu[jk] = plu[jk]

    # ---- deep pass (:1517-1746) ----------------------------------------
    deltt, deltq = _F(0.2), _F(1.0e-4)
    deepflag = False
    itoppacel = klev
    for jk in range(klev, 0, -1):
        if _F(P_paph[klev + 1] - P_paph[jk]) < _F(350.0e2):
            itoppacel = jk

    for levels in range(klevm1 - 1, klev // 2, -1):     # klevm1-1 .. klev/2+1
        for jk in range(1, klev + 1):
            plu[jk] = ptu[jk] = pqu[jk] = _F(0.0)
            dh[jk] = dhen[jk] = kup[jk] = _F(0.0)
            vptu[jk] = vten[jk] = abuoy[jk] = zbuo[jk] = _F(0.0)
            klab[jk] = 0
        kcbot[0] = kctop[0] = levels
        lldcum, resetflag = False, False
        loflag = (not deepflag) and (levels >= itoppacel)

        for jk in range(levels, 1, -1):
            if not loflag:
                break
            if jk == levels:
                if _F(P_paph[klev + 1] - P_paph[jk]) < _F(60.0e2):
                    tmix = qmix = zmix = pmix = _F(0.0)
                    for nk in range(jk + 2, jk - 1, -1):
                        if pmix < _F(50.0e2):
                            dp = _F(P_paph[nk] - P_paph[nk - 1])
                            tmix = _F(tmix + _F(dp * P_ptenh[nk]))
                            qmix = _F(qmix + _F(dp * P_pqenh[nk]))
                            zmix = _F(zmix + _F(dp * P_pgeoh[nk]))
                            pmix = _F(pmix + dp)
                    tmix = _F(tmix / pmix)
                    qmix = _F(qmix / pmix)
                    zmix = _F(zmix / pmix)
                else:
                    tmix, qmix = P_ptenh[jk + 1], P_pqenh[jk + 1]
                    zmix = P_pgeoh[jk + 1]
                pqu[jk + 1] = _F(qmix + deltq)
                dhen[jk + 1] = _F(zmix + _F(tmix * c.cpd))
                dh[jk + 1] = _F(dhen[jk + 1] + _F(deltt * c.cpd))
                ptu[jk + 1] = _F(_F(dh[jk + 1] - P_pgeoh[jk + 1]) * c.rcpd)
                kup[jk + 1] = _F(0.5)
                klab[jk + 1] = 1
                vptu[jk + 1] = _F(ptu[jk + 1]
                                  * _F(_F(1.0) + _F(c.vtmpc1 * pqu[jk + 1])))
                vten[jk + 1] = _F(P_ptenh[jk + 1]
                                  * _F(_F(1.0)
                                       + _F(c.vtmpc1 * P_pqenh[jk + 1])))
                zbuo[jk + 1] = _F(_F(vptu[jk + 1] - vten[jk + 1])
                                  / vten[jk + 1])

            r = _F(P_pqsen[jk] / P_pqsen[levels])
            fscale = _F(min(_F(1.0), _F(_F(r * r) * r)))
            eta = _F(_F(1.75e-3) * fscale)
            dz = _F(_F(P_pgeoh[jk] - P_pgeoh[jk + 1]) * c.zrg)
            coef = _F(_F(_F(0.5) * eta) * dz)
            dhen[jk] = _F(P_pgeoh[jk] + _F(c.cpd * P_ptenh[jk]))
            dh[jk] = _F(_F(_F(coef * _F(dhen[jk + 1] + dhen[jk]))
                           + _F(_F(_F(1.0) - coef) * dh[jk + 1]))
                        / _F(_F(1.0) + coef))
            pqu[jk] = _F(_F(_F(coef * _F(P_pqenh[jk + 1] + P_pqenh[jk]))
                            + _F(_F(_F(1.0) - coef) * pqu[jk + 1]))
                         / _F(_F(1.0) + coef))
            ptu[jk] = _F(_F(dh[jk] - P_pgeoh[jk]) * c.rcpd)
            zqold = pqu[jk]
            nt_cuadjtqn1(ptu, pqu, jk, P_paph[jk], c)

            zdq = _F(max(_F(zqold - pqu[jk]), _F(0.0)))
            plu[jk] = _F(plu[jk + 1] + zdq)
            zlglac = _F(zdq * _F(_F(_F(1.0) - nt_foealfa(ptu[jk]))
                                 - _F(_F(1.0) - nt_foealfa(ptu[jk + 1]))))
            plu[jk] = _F(_F(0.5) * plu[jk])          # NOT the shallow clamp
            dh[jk] = _F(P_pgeoh[jk]
                        + _F(c.cpd * _F(ptu[jk] + _F(c.ralfdcp * zlglac))))
            vptu[jk] = _F(_F(ptu[jk]
                             * _F(_F(_F(1.0) + _F(c.vtmpc1 * pqu[jk]))
                                  - plu[jk]))
                          + _F(c.ralfdcp * zlglac))
            vten[jk] = _F(P_ptenh[jk] * _F(_F(1.0)
                                           + _F(c.vtmpc1 * P_pqenh[jk])))
            zbuo[jk] = _F(_F(vptu[jk] - vten[jk]) / vten[jk])
            abuoy[jk] = _F(_F(_F(zbuo[jk] + zbuo[jk + 1]) * _F(0.5)) * c.g)
            atop1 = _F(_F(1.0) - _F(_F(2.0) * coef))
            atop2 = _F(_F(_F(2.0) * dz) * abuoy[jk])
            abot = _F(_F(1.0) + _F(_F(2.0) * coef))
            kup[jk] = _F(_F(_F(atop1 * kup[jk + 1]) + atop2) / abot)

            if plu[jk] > _F(0.0) and klab[jk + 1] == 1:
                _cloud_base(jk, ptu, pqu, plu, kup, klab, P_paph, kcbot,
                            klev, c)

            if kup[jk] < _F(0.0):
                loflag = False
                if plu[jk + 1] > _F(0.0):
                    kctop[0] = jk
                    lldcum = True
                else:
                    lldcum = False
            else:
                klab[jk] = 2 if plu[jk] > _F(0.0) else 1

        ikb, ikt = kcbot[0], kctop[0]
        if _F(P_paph[ikb] - P_paph[ikt]) < _ZDNOPRC:
            lldcum = False
        if lldcum:
            ktype, ldcum, deepflag = 1, True, True
            wbase = _F(np.sqrt(_F(max(_F(_F(2.0) * kup[ikb]), _F(0.0)))))
            cubot, cutop = ikb, ikt
            kdpl = levels + 1
            resetflag = True

        if resetflag:
            ikt, ikb = kctop[0], kdpl
            for jk in range(klev, 0, -1):
                if ikt <= jk <= ikb:
                    o_culab[jk] = klab[jk]
                    o_cutu[jk] = ptu[jk]
                    o_cuqu[jk] = pqu[jk]
                    o_culu[jk] = plu[jk]
                else:
                    o_culab[jk] = 1
                    o_cutu[jk] = P_ptenh[jk]
                    o_cuqu[jk] = P_pqenh[jk]
                    o_culu[jk] = _F(0.0)
                if jk < ikt:
                    o_culab[jk] = 0

    return {"ldcum": bool(ldcum), "ktype": int(ktype),
            "cubot": int(cubot), "cutop": int(cutop), "kdpl": int(kdpl),
            "wbase": _F(wbase),
            "cutu": o_cutu[1:nz + 1].copy(), "cuqu": o_cuqu[1:nz + 1].copy(),
            "culu": o_culu[1:nz + 1].copy(),
            "culab": o_culab[1:nz + 1].copy()}


# ===========================================================================
# Slice 4a: cubasmcn and cuentrn
# ===========================================================================
# cu_ntiedtke.F90:3457-3482 (cubasmcn) and :3516-3536 (cuentrn).
#
# cubasmcn is where ktype = 3 is assigned (:3480) and therefore where the
# mid-level arm of the scheme begins.  cuascn calls it once per level,
# jk = klev-1 .. 3.
#
# THE ALIASING RULE APPLIES HERE HARDER THAN ANYWHERE.  All THIRTEEN of
# cubasmcn's outputs are written only inside its guard, so a column that
# does not trigger keeps the caller's value in every one of them.  These
# functions therefore MUTATE the caller's arrays in place and return
# nothing -- they must never allocate fresh outputs, and the CUDA kernel
# must never zero those slots at entry.  See
# gpuwm/data/ntiedtke/oracle/nt-aliasing-audit.txt.

_CMFCMIN = _F(1.0e-10)
_CMFCMAX = _F(1.0)


def np_ntiedtke_cubasmcn(kk, *, pten, pqen, pqsen, pverv, pgeo, pgeoh,
                         ldcum, ktype, klab, plrain, pmfu, pmfub, kcbot,
                         ptu, pqu, plu, pmfus, pmfuq, pmful, pdmfup,
                         c=None):
    """cubasmcn at one level, one column, IN PLACE (:3457-3482).

    ``kk`` is 1-based, matching the Fortran; the arrays are 0-based, so
    every subscript below carries its Fortran form.  Mutates its arguments
    and returns nothing -- the untouched case is the whole point.

    ``ktype``, ``kcbot``, ``pmfub``, ``ldcum`` are single-element lists so
    the caller sees the scalar updates.
    """
    c = NtConstants() if c is None else c
    k0 = kk - 1                                  # kk, 0-based
    k1 = kk                                      # kk+1, 0-based

    # if(.not.ldcum .and. klab(kk+1) == 0)
    if ldcum[0] or klab[k1] != 0:
        return
    # lmfmid is a parameter, .true.  The three remaining gates are the
    # humidity one and the two height ones; zrg = 1/g, so pgeo*zrg is
    # height in metres and the window is 500 m to 10 km.
    if not (_F(pqen[k0]) > _F(_F(0.80) * _F(pqsen[k0]))
            and _F(pgeo[k0] * c.zrg) > _F(5.0e2)
            and _F(pgeo[k0] * c.zrg) < _F(1.0e4)):
        return

    ptu[k1] = _F(_F(_F(_F(c.cpd * _F(pten[k0])) + _F(pgeo[k0]))
                    - _F(pgeoh[k1])) * c.rcpd)
    pqu[k1] = _F(pqen[k0])
    plu[k1] = _F(0.0)
    zzzmb = _F(max(_CMFCMIN, _F(-_F(pverv[k0]) * c.zrg)))
    zzzmb = _F(min(zzzmb, _CMFCMAX))
    pmfub[0] = zzzmb
    pmfu[k1] = pmfub[0]
    pmfus[k1] = _F(pmfub[0] * _F(_F(c.cpd * ptu[k1]) + _F(pgeoh[k1])))
    pmfuq[k1] = _F(pmfub[0] * pqu[k1])
    pmful[k1] = _F(0.0)
    pdmfup[k1] = _F(0.0)
    kcbot[0] = kk
    klab[k1] = 1
    plrain[k1] = _F(0.0)
    ktype[0] = 3


def np_ntiedtke_cuentrn(kk, *, kcbot, ldcum, ldwork, pgeoh, pmfu,
                        pdmfen, pdmfde, c=None):
    """cuentrn at one level, one column, IN PLACE (:3516-3536).

    ``zentr`` is set to zero and never assigned again, so ``pdmfen`` is
    IDENTICALLY ZERO in v4.6.1 -- the entrainment term here is dead code.
    It is transcribed anyway, as the multiply the reference performs,
    because a later WRF that revives ``zentr`` must break the parity gate
    rather than silently change the answer.

    The whole body sits inside ``if (ldwork)``, so both outputs keep the
    caller's values when ldwork is false.
    """
    c = NtConstants() if c is None else c
    if not ldwork:
        return
    pdmfen[0] = _F(0.0)
    pdmfde[0] = _F(0.0)
    zentr = _F(0.0)
    if not ldcum[0]:
        return
    k0 = kk - 1
    zdz = _F(_F(_F(pgeoh[k0]) - _F(pgeoh[k0 + 1])) * c.zrg)
    zmf = _F(_F(pmfu[k0 + 1]) * zdz)
    if kk < kcbot[0]:
        pdmfen[0] = _F(zentr * zmf)
        pdmfde[0] = _F(_F(0.75e-4) * zmf)


# ===========================================================================
# Slice 4b: cumastrn's first-guess cloud-base mass flux
# ===========================================================================
# cu_ntiedtke.F90:500-541.  Runs BETWEEN cutypen and cuascn, and it is a
# prerequisite rather than a stage: it produces pmfub, which THREE routines
# consume --
#
#   cuascn   (:553 -> :1949-1952, :1992)  the whole updraft mass flux
#   cudlfsn  (:602 -> :2469)              zmftop = -cmfdeps*pmfub
#   the closure (:684, :698, :713, :722, :732, :745) via zmfub1
#
# -- and it can flip ldcum to .false. (:536) for a ktype = 2 column whose
# PBL moist static energy budget is non-positive.
#
# A fixture that skips it hands cuascn pmfub = 0, and every mass-flux
# quantity downstream is structurally zero: green, and meaningless.  That is
# the same defect that made the cuentrn capture degenerate, showing up
# twice, which is the tell that it is a prerequisite and not a coverage gap.
#
# upbl is computed here too and is consumed by the closure at :636-637
# (ztaubl, the nonequil timescale), not by anything in this slice.  It is
# captured and graded now so the closure slice inherits it proven.


def np_ntiedtke_mfub(*, ldcum, ktype, kcbot, ptte, pqte, paph, puen, pven,
                     ptu, pqu, plu, ztenh, zqenh, lndj, dt, c=None):
    """cumastrn:500-541, one column.

    ``ldcum`` is a single-element list because the block can clear it.
    Returns ``(zdhpbl, upbl, zmfub)``; ``ldcum[0]`` is updated in place.

    Index convention: the arrays are 0-based, the Fortran is 1-based, and
    ``kcbot`` arrives 1-based from cutypen, so every subscript below carries
    its Fortran form.
    """
    c = NtConstants() if c is None else c
    nz = ptte.shape[0]
    klev = nz

    # cumastrn:468-469.  zcons2 = 3/(g*dt); the 3 is not a typo -- zcons at
    # :468 is the same quantity with 1.
    zcons2 = _F(_F(3.0) / _F(c.g * _F(dt)))

    zdhpbl = _F(0.0)
    upbl = _F(0.0)

    # do jk = 2, klev   ->   k = 1 .. nz-1 (0-based)
    for k in range(1, klev):
        jk = k + 1                                   # Fortran index
        if not (jk >= kcbot and ldcum[0]):
            continue
        dp = _F(_F(paph[jk]) - _F(paph[jk - 1]))     # paph(jk+1)-paph(jk)
        zdhpbl = _F(zdhpbl + _F(_F(_F(c.alv * _F(pqte[k]))
                                   + _F(c.cpd * _F(ptte[k]))) * dp))
        if lndj == 0:
            wspeed = _F(np.sqrt(_F(_F(_F(puen[k]) * _F(puen[k]))
                                   + _F(_F(pven[k]) * _F(pven[k])))))
            upbl = _F(upbl + _F(wspeed * dp))

    if not ldcum[0]:
        return zdhpbl, upbl, _F(0.0)

    ikb = kcbot                                       # 1-based
    zmfmax = _F(_F(_F(paph[ikb - 1]) - _F(paph[ikb - 2])) * zcons2)

    if ktype == 1:
        zmfub = _F(_F(0.1) * zmfmax)
    elif ktype == 2:
        zqumqe = _F(_F(_F(ptu_q(pqu, ikb)) + _F(ptu_q(plu, ikb)))
                    - _F(ptu_q(zqenh, ikb)))
        zdqmin = _F(max(_F(_F(0.01) * _F(ptu_q(zqenh, ikb))), _F(1.0e-10)))
        zdh = _F(_F(c.cpd * _F(_F(ptu_q(ptu, ikb)) - _F(ptu_q(ztenh, ikb))))
                 + _F(c.alv * zqumqe))
        zdh = _F(c.g * _F(max(zdh, _F(_F(1.0e5) * zdqmin))))
        if zdhpbl > _F(0.0):
            zmfub = _F(zdhpbl / zdh)
            zmfub = _F(min(zmfub, zmfmax))
        else:
            zmfub = _F(_F(0.1) * zmfmax)
            ldcum[0] = False
    else:
        # ktype 0 or 3 with ldcum true.  cutypen only produces 0/1/2 and
        # ldcum is false whenever ktype is 0, so this arm is unreachable
        # from cumastrn -- but the reference leaves zmfub at its
        # initialised zero here rather than assigning, so the port does too.
        zmfub = _F(0.0)

    return zdhpbl, upbl, zmfub


def ptu_q(arr, ikb):
    """``arr(ikb)`` with ikb 1-based -- spelled out so the off-by-one has
    one place to be wrong instead of six."""
    return arr[ikb - 1]


# ===========================================================================
# Slice 5: the CAPE closure (cumastrn:620-745)
# ===========================================================================
# THE ARITHMETIC THE WHOLE PORT TURNS ON.  This is where scale_fac and
# scale_fac2 are actually applied, and they go to DIFFERENT ktypes:
#
#   :676  ztau = ztauc * scale_fac        ktype == 1 (deep) only
#   :716  zmfub1 = zmfub1 / scale_fac2    ktype == 2 (shallow) only
#   :722  zmfub1 = zmfub                  ktype == 3, neither
#
# The deep arm is NOT a division by scale_fac.  zmfub1 =
# zcape*zmfub/(zheat*ztau) is a full CAPE closure with a max(zmfub1, 0.001)
# floor and a zmfmax cap, and it can EXCEED its own first guess -- measured
# at 141.2% for dx = 15000.  What scales with resolution is zmfub1 through
# 1/ztau, which is why the cross-resolution ratio is
# scale_fac(15000)/scale_fac(4500) = 10.3%.
#
# Inputs come from captures at the closure's own entry, never reconstructed.
# ztenh/zqenh in particular are the POST-cuascn values: cuascn rewrites them
# at :2119-2120 on 108 of 5,292 rows, and the pre-cuascn ones would be wrong
# there.

_CMFCMIN_C = _F(1.0e-10)


def np_ntiedtke_closure(*, ldcum, ktype, kcbot, kctop, kdpl, loddraf,
                        wup, upbl, zmfub, scale_fac, scale_fac2, lndj,
                        pgeoh, paph, pap, pgeo, pten, pqen, ptenh, pqenh,
                        ptu, pqu, plu, pmfu, pmfd, ptd, pqd, ptte, pqte,
                        pmfds, pmfdq, pdmfdp, pmfdde_rate, zdhpbl,
                        dt, c=None):
    """cumastrn:620-745, one column.

    Arrays are 0-based; ``kcbot``/``kctop``/``kdpl`` arrive 1-based from
    cuascn, so every subscript carries its Fortran form.  ``pmfd``,
    ``pmfds``, ``pmfdq``, ``pdmfdp`` and ``pmfdde_rate`` are MUTATED by
    section 6.4 and are returned.
    """
    c = NtConstants() if c is None else c
    nz = pten.shape[0]
    klev = nz
    zcons2 = _F(_F(3.0) / _F(c.g * _F(dt)))

    zheat = _F(0.0); zcape = _F(0.0)
    zcape1 = _F(0.0); zcape2 = _F(0.0)
    ztauc = _F(0.0); ztaubl = _F(0.0); ztau = _F(0.0)
    zmfub1 = _F(0.0)
    upbl_out = _F(upbl)

    deep = bool(ldcum) and ktype == 1

    # :622-641  timescales
    if deep:
        ikb, ikt = kcbot, kctop
        zmfub1 = _F(zmfub)
        ztauc = _F(_F(_F(pgeoh[ikt - 1]) - _F(pgeoh[ikb - 1]))
                   / _F(_F(_F(2.0) + _F(min(_F(15.0), _F(wup)))) * c.g))
        if lndj == 0:
            upbl_out = _F(_F(2.0) + _F(_F(upbl)
                          / _F(_F(paph[klev]) - _F(paph[ikb - 1]))))
            ztaubl = _F(_F(_F(pgeoh[ikb - 1]) - _F(pgeoh[klev]))
                        / _F(c.g * upbl_out))
            ztaubl = _F(min(_F(300.0), ztaubl))
        else:
            ztaubl = ztauc

    # :644-668  the CAPE and heating integrals
    for jk in range(1, klev + 1):
        k0 = jk - 1
        if deep and jk <= kcbot and jk > kctop:
            zdz = _F(_F(pgeo[k0 - 1]) - _F(pgeo[k0]))
            zdp = _F(_F(pap[k0]) - _F(pap[k0 - 1]))
            zheat = _F(zheat + _F(
                _F(_F(_F(_F(_F(pten[k0 - 1]) - _F(pten[k0]))
                         + _F(zdz * c.rcpd)) / _F(ptenh[k0]))
                   + _F(c.vtmpc1 * _F(_F(pqen[k0 - 1]) - _F(pqen[k0]))))
                * _F(c.g * _F(_F(pmfu[k0]) + _F(pmfd[k0])))))
            zcape1 = _F(zcape1 + _F(
                _F(_F(_F(_F(ptu[k0]) - _F(ptenh[k0])) / _F(ptenh[k0]))
                   + _F(_F(c.vtmpc1 * _F(_F(pqu[k0]) - _F(pqenh[k0])))
                        - _F(plu[k0]))) * zdp))
        if deep and jk >= kcbot:
            if _F(_F(paph[klev]) - _F(paph[kdpl - 1])) < _F(50.0e2):
                zdp = _F(_F(paph[k0 + 1]) - _F(paph[k0]))
                zcape2 = _F(zcape2 + _F(_F(ztaubl * _F(
                    _F(_F(_F(1.0) + _F(c.vtmpc1 * _F(pqen[k0])))
                       * _F(ptte[k0]))
                    + _F(_F(c.vtmpc1 * _F(pten[k0])) * _F(pqte[k0]))))
                    * zdp))

    # :670-694  the deep closure
    if deep:
        ikb = kcbot
        ztauc = _F(max(_F(dt), ztauc))
        ztauc = _F(max(_F(360.0), ztauc))
        ztauc = _F(min(_F(10800.0), ztauc))
        ztau = _F(ztauc * _F(scale_fac))
        # nonequil is .true. (cu_ntiedtke_common:49)
        zcape2 = _F(max(_F(0.0), zcape2))
        zcape = _F(max(_F(0.0), _F(min(_F(zcape1 - zcape2), _F(5000.0)))))
        zheat = _F(max(_F(1.0e-4), zheat))
        zmfub1 = _F(_F(zcape * _F(zmfub)) / _F(zheat * ztau))
        zmfub1 = _F(max(zmfub1, _F(0.001)))
        zmfmax = _F(_F(_F(paph[ikb - 1]) - _F(paph[ikb - 2])) * zcons2)
        zmfub1 = _F(min(zmfub1, zmfmax))

    # :696-720  the shallow closure -- the ONLY place scale_fac2 is used
    if bool(ldcum) and ktype == 2:
        ikb = kcbot
        if _F(pmfd[ikb - 1]) < _F(0.0) and loddraf:
            zeps = _F(-_F(pmfd[ikb - 1]) / _F(max(_F(zmfub), _CMFCMIN_C)))
        else:
            zeps = _F(0.0)
        zqumqe = _F(_F(_F(_F(pqu[ikb - 1]) + _F(plu[ikb - 1]))
                       - _F(zeps * _F(pqd[ikb - 1])))
                    - _F(_F(_F(1.0) - zeps) * _F(pqenh[ikb - 1])))
        zdqmin = _F(max(_F(_F(0.01) * _F(pqenh[ikb - 1])), _CMFCMIN_C))
        zmfmax = _F(_F(_F(paph[ikb - 1]) - _F(paph[ikb - 2])) * zcons2)
        zdh = _F(_F(c.cpd * _F(_F(_F(ptu[ikb - 1])
                                  - _F(zeps * _F(ptd[ikb - 1])))
                               - _F(_F(_F(1.0) - zeps) * _F(ptenh[ikb - 1]))))
                 + _F(c.alv * zqumqe))
        zdh = _F(c.g * _F(max(zdh, _F(_F(1.0e5) * zdqmin))))
        # zdhpbl is not recomputed here: cumastrn keeps the value from
        # :506-517, so it is an INPUT to this slice, not a local.
        # zdhpbl is an INPUT, captured at the closure's entry.  cumastrn
        # built it at :506-517 from CUTYPEN's kcbot and ldcum, and cuascn
        # has since changed both -- recomputing it from closure-time state
        # is wrong on every shallow column, measured.
        if _F(zdhpbl) > _F(0.0):
            zmfub1 = _F(_F(zdhpbl) / zdh)
        else:
            zmfub1 = _F(zmfub)
        zmfub1 = _F(zmfub1 / _F(scale_fac2))
        zmfub1 = _F(min(zmfub1, zmfmax))

    # :722-724  mid-level takes NEITHER factor
    if bool(ldcum) and ktype == 3:
        zmfub1 = _F(zmfub)

    # :726-740  scale the downdraft by zmfub1/zmfub
    o_pmfd = _np32(pmfd).copy()
    o_pmfds = _np32(pmfds).copy()
    o_pmfdq = _np32(pmfdq).copy()
    o_pdmfdp = _np32(pdmfdp).copy()
    o_rate = _np32(pmfdde_rate).copy()
    if ldcum:
        zfac = _F(zmfub1 / _F(max(_F(zmfub), _CMFCMIN_C)))
        for k in range(klev):
            o_pmfd[k] = _F(o_pmfd[k] * zfac)
            o_pmfds[k] = _F(o_pmfds[k] * zfac)
            o_pmfdq[k] = _F(o_pmfdq[k] * zfac)
            o_pdmfdp[k] = _F(o_pdmfdp[k] * zfac)
            o_rate[k] = _F(o_rate[k] * zfac)

    return {"zheat": zheat, "zcape": zcape, "zcape1": zcape1,
            "zcape2": zcape2, "ztauc": ztauc, "ztaubl": ztaubl,
            "ztau": ztau, "zmfub1": zmfub1, "upbl": upbl_out,
            "pmfd": o_pmfd, "pmfds": o_pmfds, "pmfdq": o_pmfdq,
            "pdmfdp": o_pdmfdp, "pmfdde_rate": o_rate}


# ===========================================================================
# Slice 6: cuascn -- the entraining/detraining updraft (cu_ntiedtke.F90:1755-2258)
# ===========================================================================
# The largest routine in the scheme, and the one that owns the plume:
# entrainment, detrainment, moist ascent through cuadjtqn, the buoyancy/KE
# integration, cloud-base refinement, precipitation conversion and rain
# fallout.  cubasmcn and cuentrn are called from inside it; both are already
# ported, and are called here rather than re-transcribed.
#
# EIGHT CLASS-1 DUMMIES.  ldcum, kctop0, klab, ptenh, pqenh, ptu, plu are all
# read before they are written -- see nt-aliasing-audit.txt.  So each is an
# INPUT as well as an output, and the fixture captures them at cuascn's own
# call site rather than reconstructing them.  ptenh/pqenh are also OUTPUTS:
# :2118-2119 rewrites them on the negative-buoyancy branch.
#
# llo3 IS A TILE-WIDE HORIZONTAL DEPENDENCY, and the only one in the routine.
# :1994 forms  is = sum over ALL columns of klab(jl,jk+1)  and :2009 sets
# llo3 .true. if is > 0.  llo3 is initialised .false. ONCE at :1903 and never
# cleared, so it is monotone: true at every level from the first level where
# any column in the tile carries a label.  The entire body of the level loop
# hangs off it (:2012), including the departure-level reset at :2069-2075,
# which is NOT guarded by loflag and therefore runs for every column.  So a
# column's ptu/pqu can in principle depend on what its tile-mates are doing,
# which also means WRF's own answer here is tile-decomposition dependent.
#
# It is passed in rather than assumed, and the gate that makes passing True
# exact lives in test_ntiedtke_cuascn_parity.py: every column entering cuascn
# with ldcum true has klab(klev) > 0 (108 of 108, all six dx), which makes
# is > 0 at jk = klevm1 -- the first iteration -- and monotonicity carries it
# the rest of the way.  A tile violating that would need a block-wide OR
# reduction in the kernel; the gate fails rather than silently diverging.

_CPRCON = _F(1.4e-3)
_RTBER = _F(_TMELT - _F(5.0))


def np_ntiedtke_cuascn(*, ptenh, pqenh, pten, pqen, pqsen, pgeo, pgeoh,
                       pap, paph, ldcum, ktype, klab, ptu, pqu, plu,
                       pmfu, pmfub, pmfus, pmfuq, pmful, plude, pdmfup,
                       kcbot, kctop, kctop0, ztmst, plglac, lndj, wbase,
                       kdpl, pverv, llo3=True, c=None):
    """cuascn, one column.  Arrays 0-based in and out; body 1-based.

    1-BASED INTERNALLY, like cutypen, because the loop runs klevm1 down to 3
    over dense jk+/-1 subscripts in three directions, and an off-by-one in
    the translation is the failure mode that costs a day.

    ``ldcum``, ``ktype``, ``kcbot``, ``kctop``, ``kctop0`` and ``pmfub`` are
    scalars in and out.  ``llo3`` is the tile-wide flag described above.
    """
    c = NtConstants() if c is None else c
    nz = pten.shape[0]
    klev, klevm1 = nz, nz - 1

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    def upi(a):
        o = _i1(nz)
        o[1:nz + 1] = np.asarray(a, dtype=np.int32)
        return o

    P_ptenh, P_pqenh = up(ptenh), up(pqenh)
    P_pten, P_pqen, P_pqsen = up(pten), up(pqen), up(pqsen)
    P_pgeo, P_pap, P_pverv = up(pgeo), up(pap), up(pverv)
    P_pgeoh, P_paph = _f1(nz), _f1(nz)
    P_pgeoh[1:nz + 2] = _np32(pgeoh)[:nz + 1]
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]

    P_ptu, P_pqu, P_plu = up(ptu), up(pqu), up(plu)
    P_pmfu, P_pmfus, P_pmfuq = up(pmfu), up(pmfus), up(pmfuq)
    P_pmful, P_plude, P_pdmfup = up(pmful), up(plude), up(pdmfup)
    P_plglac = up(plglac)
    P_klab = upi(klab)

    zlrain, zbuo, kup = _f1(nz), _f1(nz), _f1(nz)
    pdmfen, pmfude_rate = _f1(nz), _f1(nz)

    ld = bool(ldcum)
    kt = int(ktype)
    kb, ktp, kt0 = int(kcbot), int(kctop), int(kctop0)
    mfub = _F(pmfub)

    # ---- 1. parameters (:1893-1899) ------------------------------------
    # Each is transcribed in the reference's own association.
    #
    # CORRECTED.  This comment used to say cuascn's zcons2 is "three times
    # looser than the closure's, which uses the same name for a different
    # number".  THAT IS WRONG: zcons2 is 3/(g*dt) in ALL THREE scopes that
    # declare it -- cumastrn:469, cuascn:1893, cuflxn:2859 -- and they are
    # identical.  What differs by a factor of three is zcons2 against
    # zcons (cumastrn:468, 1/(g*dt)), which are one character apart and
    # coexist in cumastrn's scope.  zcons has exactly ONE consumer, the
    # momentum rescale at :1000.  See test_ntiedtke_constant_family.py.
    zcons2 = _F(_F(3.0) / _F(c.g * _F(ztmst)))
    zfacbuo = _F(_F(0.5) / _F(_F(1.0) + _F(0.5)))
    zprcdgw = _F(_CPRCON * c.zrg)
    z_cldmax = _F(5.0e-3)
    z_cwifrac = _F(0.5)
    z_cprc2 = _F(0.5)
    z_cwdrag = _F(_F(_F(_F(3.0) / _F(8.0)) * _F(0.506)) / _F(0.2))

    # ---- 2. defaults (:1903-1937) --------------------------------------
    zluold = _F(0.0)
    wup = _F(0.0)
    zdpmean = _F(0.0)
    zoentr = _F(0.0)
    if not ld:
        kt = 0
        kb = -1
        mfub = _F(0.0)
        P_pqu[klev] = _F(0.0)

    for jk in range(1, klev + 1):
        if jk != kb:
            P_plu[jk] = _F(0.0)
        P_pmfu[jk] = _F(0.0); P_pmfus[jk] = _F(0.0)
        P_pmfuq[jk] = _F(0.0); P_pmful[jk] = _F(0.0)
        P_plude[jk] = _F(0.0); P_plglac[jk] = _F(0.0)
        P_pdmfup[jk] = _F(0.0); zlrain[jk] = _F(0.0)
        zbuo[jk] = _F(0.0); kup[jk] = _F(0.0)
        pdmfen[jk] = _F(0.0); pmfude_rate[jk] = _F(0.0)
        if (not ld) or kt == 3:
            P_klab[jk] = 0
        if (not ld) and P_paph[jk] < _F(4.0e4):
            kt0 = jk
    if kt == 3:
        ld = False

    # ---- 3. cloud base (:1943-1953) ------------------------------------
    ktp = kb
    if ld:
        ikb = kb
        kup[ikb] = _F(_F(0.5) * _F(wbase * wbase))
        P_pmfu[ikb] = mfub
        P_pmfus[ikb] = _F(mfub * _F(_F(c.cpd * P_ptu[ikb]) + P_pgeoh[ikb]))
        P_pmfuq[ikb] = _F(mfub * P_pqu[ikb])
        P_pmful[ikb] = _F(mfub * P_plu[ikb])

    zdmfen = _F(0.0)
    zdmfde = _F(0.0)

    # ---- 4. the ascent (:1959-2245) ------------------------------------
    for jk in range(klevm1, 2, -1):
        # cubasmcn (:1968), already ported and graded.
        ldc = [ld]; ktv = [kt]; kbv = [kb]; mfv = [mfub]
        np_ntiedtke_cubasmcn(
            jk, pten=P_pten[1:klev + 1], pqen=P_pqen[1:klev + 1],
            pqsen=P_pqsen[1:klev + 1], pverv=P_pverv[1:klev + 1],
            pgeo=P_pgeo[1:klev + 1], pgeoh=P_pgeoh[1:klev + 2],
            ldcum=ldc, ktype=ktv, klab=P_klab[1:klev + 1],
            plrain=zlrain[1:klev + 1], pmfu=P_pmfu[1:klev + 1],
            pmfub=mfv, kcbot=kbv, ptu=P_ptu[1:klev + 1],
            pqu=P_pqu[1:klev + 1], plu=P_plu[1:klev + 1],
            pmfus=P_pmfus[1:klev + 1], pmfuq=P_pmfuq[1:klev + 1],
            pmful=P_pmful[1:klev + 1], pdmfup=P_pdmfup[1:klev + 1], c=c)
        ld, kt, kb, mfub = ldc[0], ktv[0], kbv[0], mfv[0]

        # :1980-2001
        llo1 = False
        zprecip = _F(0.0)
        if P_klab[jk + 1] == 0:
            P_klab[jk] = 0
        loflag = ((ld and P_klab[jk + 1] == 2)
                  or (kt == 3 and P_klab[jk + 1] == 1))
        zph = P_paph[jk]
        if kt == 3 and jk == kb:
            zmfmax = _F(_F(P_paph[jk] - P_paph[jk - 1]) * zcons2)
            if mfub > zmfmax:
                zfac = _F(zmfmax / mfub)
                P_pmfu[jk + 1] = _F(P_pmfu[jk + 1] * zfac)
                P_pmfus[jk + 1] = _F(P_pmfus[jk + 1] * zfac)
                P_pmfuq[jk + 1] = _F(P_pmfuq[jk + 1] * zfac)
                mfub = zmfmax
            mfub = _F(min(mfub, zmfmax))

        # cuentrn (:2006), already ported.  ldwork is llo3, the tile flag.
        en, de = [zdmfen], [zdmfde]
        np_ntiedtke_cuentrn(jk, kcbot=[kb], ldcum=[ld], ldwork=llo3,
                            pgeoh=P_pgeoh[1:klev + 2],
                            pmfu=P_pmfu[1:klev + 1],
                            pdmfen=en, pdmfde=de, c=c)
        zdmfen, zdmfde = en[0], de[0]

        if not llo3:
            continue

        # :2015-2065  entrainment, detrainment and the plume update
        zqold = _F(0.0)
        if loflag:
            zdmfde = _F(min(zdmfde, _F(_F(0.75) * P_pmfu[jk + 1])))
            if jk == kb:
                r = _F(min(_F(1.0), _F(P_pqen[jk] / P_pqsen[jk])))
                zoentr = _F(_F(_F(_F(-_F(1.75e-3)) * _F(r - _F(1.0)))
                               * _F(P_pgeoh[jk] - P_pgeoh[jk + 1])) * c.zrg)
                zoentr = _F(_F(min(_F(0.4), zoentr)) * P_pmfu[jk + 1])
            if jk < kb:
                zmfmax = _F(_F(P_paph[jk] - P_paph[jk - 1]) * zcons2)
                zxs = _F(max(_F(P_pmfu[jk + 1] - zmfmax), _F(0.0)))
                wup = _F(wup + _F(kup[jk + 1]
                                  * _F(P_pap[jk + 1] - P_pap[jk])))
                # :2036 is  zdpmean + pap(jk+1) - pap(jk)  with NO
                # parentheses, so it associates left-to-right.  The wup
                # accumulator one line up IS parenthesised.  Bracketing the
                # difference here costs 1 ULP in wup on one column.
                zdpmean = _F(_F(zdpmean + P_pap[jk + 1]) - P_pap[jk])
                zdmfen = zoentr
                if kt >= 2:
                    zdmfen = _F(_F(2.0) * zdmfen)
                    zdmfde = zdmfen
                zdmfde = _F(zdmfde * _F(_F(1.6) - _F(min(
                    _F(1.0), _F(P_pqen[jk] / P_pqsen[jk])))))
                zmftest = _F(_F(P_pmfu[jk + 1] + zdmfen) - zdmfde)
                zchange = _F(max(_F(zmftest - zmfmax), _F(0.0)))
                zxe = _F(max(_F(zchange - zxs), _F(0.0)))
                zdmfen = _F(zdmfen - zxe)
                zchange = _F(zchange - zxe)
                zdmfde = _F(zdmfde + zchange)
            pdmfen[jk] = _F(zdmfen - zdmfde)
            P_pmfu[jk] = _F(_F(P_pmfu[jk + 1] + zdmfen) - zdmfde)
            zqeen = _F(P_pqenh[jk + 1] * zdmfen)
            zseen = _F(_F(_F(c.cpd * P_ptenh[jk + 1]) + P_pgeoh[jk + 1])
                       * zdmfen)
            zscde = _F(_F(_F(c.cpd * P_ptu[jk + 1]) + P_pgeoh[jk + 1])
                       * zdmfde)
            zqude = _F(P_pqu[jk + 1] * zdmfde)
            P_plude[jk] = _F(P_plu[jk + 1] * zdmfde)
            zmfusk = _F(_F(P_pmfus[jk + 1] + zseen) - zscde)
            zmfuqk = _F(_F(P_pmfuq[jk + 1] + zqeen) - zqude)
            zmfulk = _F(P_pmful[jk + 1] - P_plude[jk])
            inv = _F(_F(1.0) / _F(max(_CMFCMIN, P_pmfu[jk])))
            P_plu[jk] = _F(zmfulk * inv)
            P_pqu[jk] = _F(zmfuqk * inv)
            P_ptu[jk] = _F(_F(_F(zmfusk * inv) - P_pgeoh[jk]) * c.rcpd)
            P_ptu[jk] = _F(max(_F(100.0), P_ptu[jk]))
            P_ptu[jk] = _F(min(_F(400.0), P_ptu[jk]))
            zqold = P_pqu[jk]
            zlrain[jk] = _F(_F(zlrain[jk + 1]
                               * _F(P_pmfu[jk + 1] - zdmfde)) * inv)
            zluold = P_plu[jk]

        # :2069-2075  reset below the departure level.  NOT under loflag.
        if jk > kdpl:
            P_ptu[jk] = P_ptenh[jk]
            P_pqu[jk] = P_pqenh[jk]
            P_plu[jk] = _F(0.0)
            zluold = P_plu[jk]

        # :2081-2083  cuadjtqn, icall = 1
        if loflag:
            nt_cuadjtqn1(P_ptu, P_pqu, jk, zph, c)

        # :2086-2093  glaciation
        if loflag and P_pqu[jk] != zqold:
            P_plglac[jk] = _F(P_plu[jk] * _F(
                _F(_F(1.0) - nt_foealfa(P_ptu[jk]))
                - _F(_F(1.0) - nt_foealfa(P_ptu[jk + 1]))))
            P_ptu[jk] = _F(P_ptu[jk] + _F(c.ralfdcp * P_plglac[jk]))

        # :2096-2179  buoyancy, kinetic energy, cloud top
        if loflag and P_pqu[jk] != zqold:
            P_klab[jk] = 2
            P_plu[jk] = _F(_F(P_plu[jk] + zqold) - P_pqu[jk])
            zbc = _F(P_ptu[jk] * _F(_F(_F(_F(1.0)
                     + _F(c.vtmpc1 * P_pqu[jk])) - P_plu[jk + 1])
                     - zlrain[jk + 1]))
            zbe = _F(P_ptenh[jk] * _F(_F(1.0)
                                      + _F(c.vtmpc1 * P_pqenh[jk])))
            zbuo[jk] = _F(zbc - zbe)
            if kt == 3 and P_klab[jk + 1] == 1:
                if zbuo[jk] > _F(-0.5):
                    ld = True
                    ktp = jk
                    kup[jk] = _F(0.5)
                else:
                    P_klab[jk] = 0
                    P_pmfu[jk] = _F(0.0)
                    P_plude[jk] = _F(0.0)
                    P_plu[jk] = _F(0.0)
            if P_klab[jk + 1] == 2:
                if zbuo[jk] < _F(0.0):
                    P_ptenh[jk] = _F(_F(0.5)
                                     * _F(P_pten[jk] + P_pten[jk - 1]))
                    P_pqenh[jk] = _F(_F(0.5)
                                     * _F(P_pqen[jk] + P_pqen[jk - 1]))
                    zbuo[jk] = _F(zbc - _F(P_ptenh[jk] * _F(
                        _F(1.0) + _F(c.vtmpc1 * P_pqenh[jk]))))
                zbuoc = _F(_F(_F(zbuo[jk] / _F(P_ptenh[jk] * _F(
                    _F(1.0) + _F(c.vtmpc1 * P_pqenh[jk]))))
                    + _F(zbuo[jk + 1] / _F(P_ptenh[jk + 1] * _F(
                        _F(1.0) + _F(c.vtmpc1 * P_pqenh[jk + 1])))))
                    * _F(0.5))
                zdkbuo = _F(_F(_F(P_pgeoh[jk] - P_pgeoh[jk + 1]) * zfacbuo)
                            * zbuoc)
                if zdmfen > _F(0.0):
                    zdken = _F(min(_F(1.0), _F(
                        _F(_F(_F(1.0) + z_cwdrag) * zdmfen)
                        / _F(max(_CMFCMIN, P_pmfu[jk + 1])))))
                else:
                    zdken = _F(min(_F(1.0), _F(
                        _F(_F(_F(1.0) + z_cwdrag) * zdmfde)
                        / _F(max(_CMFCMIN, P_pmfu[jk + 1])))))
                kup[jk] = _F(_F(_F(kup[jk + 1] * _F(_F(1.0) - zdken))
                                + zdkbuo) / _F(_F(1.0) + zdken))
                if zbuo[jk] < _F(0.0):
                    zkedke = _F(kup[jk] / _F(max(_F(1.0e-10), kup[jk + 1])))
                    zkedke = _F(max(_F(0.0), _F(min(_F(1.0), zkedke))))
                    zmfun = _F(_F(np.sqrt(zkedke)) * P_pmfu[jk + 1])
                    zdmfde = _F(max(zdmfde, _F(P_pmfu[jk + 1] - zmfun)))
                    P_plude[jk] = _F(P_plu[jk + 1] * zdmfde)
                    P_pmfu[jk] = _F(_F(P_pmfu[jk + 1] + zdmfen) - zdmfde)
                if zbuo[jk] > _F(-0.2):
                    ikb = kb
                    rr = _F(min(_F(1.0), _F(P_pqen[jk - 1]
                                            / P_pqsen[jk - 1])))
                    q3 = _F(min(_F(1.0), _F(P_pqsen[jk] / P_pqsen[ikb])))
                    zoentr = _F(_F(_F(_F(_F(1.75e-3)
                        * _F(_F(0.3) - _F(rr - _F(1.0))))
                        * _F(P_pgeoh[jk - 1] - P_pgeoh[jk])) * c.zrg)
                        * _F(_F(q3 * q3) * q3))
                    zoentr = _F(_F(min(_F(0.4), zoentr)) * P_pmfu[jk])
                else:
                    zoentr = _F(0.0)
                if jk > kdpl:
                    P_pmfu[jk] = P_pmfu[jk + 1]
                    kup[jk] = _F(0.5)
                if kup[jk] > _F(0.0) and P_pmfu[jk] > _F(0.0):
                    ktp = jk
                    llo1 = True
                else:
                    P_klab[jk] = 0
                    P_pmfu[jk] = _F(0.0)
                    kup[jk] = _F(0.0)
                    zdmfde = P_pmfu[jk + 1]
                    P_plude[jk] = _F(P_plu[jk + 1] * zdmfde)
                if P_pmfu[jk + 1] > _F(0.0):
                    pmfude_rate[jk] = zdmfde
        elif loflag and kt == 2 and P_pqu[jk] == zqold:
            P_klab[jk] = 0
            P_pmfu[jk] = _F(0.0)
            kup[jk] = _F(0.0)
            zdmfde = P_pmfu[jk + 1]
            P_plude[jk] = _F(P_plu[jk + 1] * zdmfde)
            pmfude_rate[jk] = zdmfde

        # :2182-2216  precipitation conversion
        if llo1:
            zdshrd = _F(5.0e-4) if lndj == 1 else _F(3.0e-4)
            if P_plu[jk] > zdshrd:
                zwu = _F(min(_F(15.0), _F(np.sqrt(_F(_F(2.0) * _F(
                    max(_F(0.1), kup[jk + 1])))))))
                zprcon = _F(zprcdgw / _F(_F(0.75) * zwu))
                zdt = _F(min(_F(_RTBER - _RTICE),
                             _F(max(_F(_RTBER - P_ptu[jk]), _F(0.0)))))
                zcbf = _F(_F(1.0) + _F(z_cprc2 * _F(np.sqrt(zdt))))
                zzco = _F(zprcon * zcbf)
                zlcrit = _F(zdshrd / zcbf)
                zdfi = _F(P_pgeoh[jk] - P_pgeoh[jk + 1])
                zc = _F(P_plu[jk] - zluold)
                rq = _F(P_plu[jk] / zlcrit)
                zarg = _F(rq * rq)
                if zarg < _F(25.0):
                    zd = _F(_F(zzco * _F(_F(1.0) - _exp32(_F(-zarg))))
                            * zdfi)
                else:
                    zd = _F(zzco * zdfi)
                zint = _exp32(_F(-zd))
                zlnew = _F(_F(zluold * zint)
                           + _F(_F(zc / zd) * _F(_F(1.0) - zint)))
                zlnew = _F(max(_F(0.0), _F(min(P_plu[jk], zlnew))))
                zlnew = _F(min(z_cldmax, zlnew))
                zprecip = _F(max(_F(0.0), _F(_F(zluold + zc) - zlnew)))
                P_pdmfup[jk] = _F(zprecip * P_pmfu[jk])
                zlrain[jk] = _F(zlrain[jk] + zprecip)
                P_plu[jk] = zlnew

        # :2219-2236  rain fallout
        if llo1 and zlrain[jk] > _F(0.0):
            zvw = _F(_F(21.18) * _pow32(zlrain[jk], _F(0.2)))
            zvi = _F(z_cwifrac * zvw)
            zalfaw = nt_foealfa(P_ptu[jk])
            zvv = _F(_F(zalfaw * zvw) + _F(_F(_F(1.0) - zalfaw) * zvi))
            zrold = _F(zlrain[jk] - zprecip)
            zc = zprecip
            zwu = _F(min(_F(15.0), _F(np.sqrt(_F(_F(2.0) * _F(
                max(_F(0.1), kup[jk])))))))
            zd = _F(zvv / zwu)
            zint = _exp32(_F(-zd))
            zrnew = _F(_F(zrold * zint)
                       + _F(_F(zc / zd) * _F(_F(1.0) - zint)))
            zrnew = _F(max(_F(0.0), _F(min(zlrain[jk], zrnew))))
            zlrain[jk] = zrnew

        # :2239-2243
        if loflag:
            P_pmful[jk] = _F(P_plu[jk] * P_pmfu[jk])
            P_pmfus[jk] = _F(_F(_F(c.cpd * P_ptu[jk]) + P_pgeoh[jk])
                             * P_pmfu[jk])
            P_pmfuq[jk] = _F(P_pqu[jk] * P_pmfu[jk])

    # ---- 5. final (:2248-2256) -----------------------------------------
    if ktp == -1:
        ld = False
    kb = max(kb, ktp)
    if ld:
        wup = _F(max(_F(1.0e-2), _F(wup / _F(max(_F(1.0), zdpmean)))))
        wup = _F(np.sqrt(_F(_F(2.0) * wup)))

    return {"ldcum": ld, "ktype": kt, "kcbot": kb, "kctop": ktp,
            "kctop0": kt0, "pmfub": mfub, "wup": wup,
            "ptenh": P_ptenh[1:nz + 1].copy(),
            "pqenh": P_pqenh[1:nz + 1].copy(),
            "ptu": P_ptu[1:nz + 1].copy(), "pqu": P_pqu[1:nz + 1].copy(),
            "plu": P_plu[1:nz + 1].copy(),
            "pmfu": P_pmfu[1:nz + 1].copy(),
            "pmfus": P_pmfus[1:nz + 1].copy(),
            "pmfuq": P_pmfuq[1:nz + 1].copy(),
            "pmful": P_pmful[1:nz + 1].copy(),
            "plude": P_plude[1:nz + 1].copy(),
            "pdmfup": P_pdmfup[1:nz + 1].copy(),
            "plglac": P_plglac[1:nz + 1].copy(),
            "klab": P_klab[1:nz + 1].copy(),
            "pmfude_rate": pmfude_rate[1:nz + 1].copy()}


# ===========================================================================
# Slice 7: cudlfsn -- the level of free sinking (cu_ntiedtke.F90:2262-2487)
# ===========================================================================
# Where downdrafts start.  It finds the level of minimum saturation moist
# static energy, wet-bulbs the environment there with cuadjtqn's kcall == 2
# arm, mixes 50/50 with cloud air and takes the first level whose mixture is
# negatively buoyant AND has enough rain to evaporate.
#
# THE APPARENT HORIZONTAL DEPENDENCY HERE IS INERT, unlike cuascn's llo3.
# :2437-2447 sums `is` over every column and :2448 does `if (is == 0) cycle`.
# But ztenwb/zqenwb/zph are set for ALL jl BEFORE the cycle; cuadjtqn is
# masked by llo2 and does nothing to a column with llo2 false; and section
# 2.2 is itself `if (llo2(jl))`.  So the reduction only skips work that
# would have been a no-op.  Stated because the same shape in cuascn was NOT
# inert, and "it looks like llo3" is not an argument either way -- each one
# has to be read.
#
# pud AND pvd ARE NEVER WRITTEN.  Both are dummies of cudlfsn (:2268) and
# neither appears anywhere in its body: the downdraft momentum is cududvn's
# job.  They are class-2 outputs in the strongest sense -- unconditionally
# untouched -- so the mirror does not return them and the kernel must not
# write them.  Graded by the fixture carrying them unchanged.


def nt_foelhm(tt, c):
    """:3562.  The latent heat blend, alv/als by foealfa."""
    a = nt_foealfa(tt)
    return _F(_F(a * c.alv) + _F(_F(_F(1.0) - a) * c.als))


def nt_cuadjtqn2(pt, pq, k, psp, c):
    """cuadjtqn's kcall == 2 arm (:3359-3379), in place at level k.

    The DOWNDRAFT arm: evaporation, so both condensate steps are clamped
    with min(.,0) rather than max.  It computes saturation through foeewm,
    NOT inline off reciprocals the way the kcall == 1 arm does -- the two
    are different expressions of the same quantity and are not
    interchangeable at max_ulp == 0.

    The `abs(zcond) < 1e-20` guard on the second clamp is transcribed as
    written: it is not `zcond == 0`, and on the columns where zcond is a
    denormal the two differ.
    """
    zqp = _F(_F(1.0) / psp)
    zqsat = _F(nt_foeewm(pt[k], c) * zqp)
    zqsat = _F(min(_F(0.5), zqsat))
    zcor = _F(_F(1.0) / _F(_F(1.0) - _F(c.vtmpc1 * zqsat)))
    zqsat = _F(zqsat * zcor)
    zcond = _F(_F(pq[k] - zqsat) / _F(_F(1.0) + _F(
        _F(zqsat * zcor) * nt_foedem(pt[k], c))))
    zcond = _F(min(zcond, _F(0.0)))
    pt[k] = _F(pt[k] + _F(nt_foeldcpm(pt[k], c) * zcond))
    pq[k] = _F(pq[k] - zcond)
    zqsat = _F(nt_foeewm(pt[k], c) * zqp)
    zqsat = _F(min(_F(0.5), zqsat))
    zcor = _F(_F(1.0) / _F(_F(1.0) - _F(c.vtmpc1 * zqsat)))
    zqsat = _F(zqsat * zcor)
    zcond1 = _F(_F(pq[k] - zqsat) / _F(_F(1.0) + _F(
        _F(zqsat * zcor) * nt_foedem(pt[k], c))))
    if abs(float(zcond)) < 1.0e-20:
        zcond1 = _F(min(zcond1, _F(0.0)))
    pt[k] = _F(pt[k] + _F(nt_foeldcpm(pt[k], c) * zcond1))
    pq[k] = _F(pq[k] - zcond1)


_CMFDEPS = _F(0.30)


def np_ntiedtke_cudlfsn(*, kcbot, kctop, lndj, ldcum, ptenh, pqenh,
                        pten, pqsen, pgeo, pgeoh, paph, ptu, pqu,
                        pmfub, prfl, ptd, pqd, pmfd, pmfds, pmfdq,
                        pdmfdp, c=None):
    """cudlfsn, one column.  Arrays 0-based in and out; body 1-based.

    ``prfl`` is an UPDATED parameter, not an input: :2478 adds the
    evaporated precipitation back into it.  Returned.

    ``puu``/``pvu``/``plu`` are dummies the routine never reads, and
    ``pud``/``pvd`` are dummies it never writes; none of the four is in
    this signature.  Checked by grepping the executable body, not assumed.
    """
    c = NtConstants() if c is None else c
    nz = pten.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_ptenh, P_pqenh = up(ptenh), up(pqenh)
    P_pten, P_pqsen, P_pgeo = up(pten), up(pqsen), up(pgeo)
    P_ptu, P_pqu = up(ptu), up(pqu)
    P_pgeoh, P_paph = _f1(nz), _f1(nz)
    P_pgeoh[1:nz + 2] = _np32(pgeoh)[:nz + 1]
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]

    # THE SIX CLASS-2 SLOTS.  nt-aliasing-audit.txt lists each as "writes
    # only at" a single line, so every level the routine does not reach
    # keeps the CALLER's value.  They arrive from the capture at cudlfsn's
    # own entry and are NOT zeroed here -- zeroing them was wrong on levels
    # 1-4 of every column, which is what the audit had already said and the
    # fixture had not yet been made to honour.
    P_ptd, P_pqd = up(ptd), up(pqd)
    P_pmfd, P_pmfds = up(pmfd), up(pmfds)
    P_pmfdq, P_pdmfdp = up(pmfdq), up(pdmfdp)
    ztenwb, zqenwb = _f1(nz), _f1(nz)

    rfl = _F(prfl)
    ld = bool(ldcum)
    kb, ktop = int(kcbot), int(kctop)

    # ---- 1. defaults (:2393-2398) --------------------------------------
    lddraf = False
    kdtop = klev + 1
    ikhsmin = klev + 1
    zhsmin = _F(1.0e8)

    # ---- 2. the level of minimum hs (:2422-2431) -----------------------
    for jk in range(3, klev - 1):
        zhsk = _F(_F(_F(c.cpd * P_pten[jk]) + P_pgeo[jk])
                  + _F(nt_foelhm(P_pten[jk], c) * P_pqsen[jk]))
        if zhsk < zhsmin:
            zhsmin = zhsk
            ikhsmin = jk

    # ---- 2.1-2.2 the descent (:2435-2484) ------------------------------
    ike = klev - 3
    for jk in range(3, ike + 1):
        ztenwb[jk] = P_ptenh[jk]
        zqenwb[jk] = P_pqenh[jk]
        zph = P_paph[jk]
        llo2 = (ld and rfl > _F(0.0) and not lddraf
                and (jk < kb and jk > ktop) and jk >= ikhsmin)
        if not llo2:
            continue

        nt_cuadjtqn2(ztenwb, zqenwb, jk, zph, c)

        zttest = _F(_F(0.5) * _F(P_ptu[jk] + ztenwb[jk]))
        zqtest = _F(_F(0.5) * _F(P_pqu[jk] + zqenwb[jk]))
        zbuo = _F(_F(zttest * _F(_F(1.0) + _F(c.vtmpc1 * zqtest)))
                  - _F(P_ptenh[jk] * _F(_F(1.0)
                                        + _F(c.vtmpc1 * P_pqenh[jk]))))
        zcond = _F(P_pqenh[jk] - zqenwb[jk])
        zmftop = _F(-_F(_CMFDEPS * _F(pmfub)))
        if zbuo < _F(0.0) and rfl > _F(_F(_F(10.0) * zmftop) * zcond):
            kdtop = jk
            lddraf = True
            P_ptd[jk] = zttest
            P_pqd[jk] = zqtest
            P_pmfd[jk] = zmftop
            P_pmfds[jk] = _F(P_pmfd[jk] * _F(_F(c.cpd * P_ptd[jk])
                                         + P_pgeoh[jk]))
            P_pmfdq[jk] = _F(P_pmfd[jk] * P_pqd[jk])
            P_pdmfdp[jk - 1] = _F(_F(-_F(0.5)) * _F(P_pmfd[jk] * zcond))
            rfl = _F(rfl + P_pdmfdp[jk - 1])

    return {"kdtop": kdtop, "lddraf": lddraf, "prfl": rfl,
            "ptd": P_ptd[1:nz + 1].copy(), "pqd": P_pqd[1:nz + 1].copy(),
            "pmfd": P_pmfd[1:nz + 1].copy(),
            "pmfds": P_pmfds[1:nz + 1].copy(),
            "pmfdq": P_pmfdq[1:nz + 1].copy(),
            "pdmfdp": P_pdmfdp[1:nz + 1].copy()}


# ===========================================================================
# Slice 8: cuddrafn -- the moist downdraft descent (:2495-2721)
# ===========================================================================
# The descent itself: organised entrainment driven by accumulated negative
# buoyancy, evaporative cooling through cuadjtqn's kcall == 2 arm, and a
# buoyancy check that can shut the downdraft off level by level.
#
# SIX CLASS-1 DUMMIES -- prfl, ptd, pqd, pmfd, pmfds, pmfdq -- every one of
# them read at jk-1 before jk is written.  They are cudlfsn's outputs and
# nothing runs between the two calls, so cudlfsn's exit capture would serve;
# the fixture captures them again at cuddrafn's own call site anyway,
# because stitching one routine's exit into another's entry is the
# reconstruction this port keeps being burned by.
#
# paph[klev+1] IS READ, three times (:2618, :2648, :2649).  That is the
# surface interface the cuascn fixture deliberately poisons with NaN because
# cuascn never touches it.  Here it is load-bearing and captured.
#
# pud AND pvd ARE NEVER WRITTEN, exactly as in cudlfsn: both are
# intent(inout) dummies (:2580) that appear nowhere else in the routine.
# Momentum is cududvn's.  Not in this signature; gated on the mirror's
# shape, because the oracle cannot tell "left alone" from "not implemented".
#
# THE `is == 0 cycle` AT :2632 IS INERT, on the same three limbs as
# cudlfsn's: zph is set for every column before it, cuadjtqn is masked by
# the same per-column llo2 the reduction sums, and every other block is
# inside `if (llo2(jl))`.  Read, not inferred.

_ENTRDD = _F(2.0e-4)


def np_ntiedtke_cuddrafn(*, lddraf, ptenh, pqenh, pgeo, pgeoh, paph,
                         pmfu, ptd, pqd, pmfd, pmfds, pmfdq, pdmfdp,
                         prfl, c=None):
    """cuddrafn, one column.  Arrays 0-based in and out; body 1-based.

    ``paph`` must carry ``klev+1`` entries -- the surface interface is read.
    ``prfl`` is updated and returned.  ``pmfdde_rate`` is an output.
    """
    c = NtConstants() if c is None else c
    nz = ptenh.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_ptenh, P_pqenh = up(ptenh), up(pqenh)
    P_pgeo, P_pmfu = up(pgeo), up(pmfu)
    P_ptd, P_pqd = up(ptd), up(pqd)
    P_pmfd, P_pmfds, P_pmfdq = up(pmfd), up(pmfds), up(pmfdq)
    P_pdmfdp = up(pdmfdp)
    # paph needs klev+1 -- :2618, :2648 and :2649 read the surface
    # interface.  pgeoh does NOT: every subscript is jk or jk-1 with
    # jk <= klev.  So pgeoh is taken at klev entries and its klev+1 slot
    # is left NaN, which turns a mistake in that reading into a hard
    # failure rather than a silent zero.
    P_pgeoh, P_paph = _f1(nz), _f1(nz)
    P_pgeoh[:] = np.nan
    P_pgeoh[1:nz + 1] = _np32(pgeoh)[:nz]
    if _np32(paph).shape[0] < nz + 1:
        raise ValueError(
            "cuddrafn reads paph[klev+1]; pass klev+1 entries")
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]
    pmfdde_rate = _f1(nz)

    ld = bool(lddraf)
    rfl = _F(prfl)

    # ---- defaults (:2609-2614) -----------------------------------------
    zoentr = _F(0.0)
    zbuoy = _F(0.0)
    zdmfen = _F(0.0)
    zdmfde = _F(0.0)
    zcond = _F(0.0)

    # ---- itopde (:2616-2621) -------------------------------------------
    # The loop DESCENDS klev..1 and the last assignment wins, so itopde ends
    # at the TOPMOST level within 60 hPa of the surface.  It is never
    # initialised in the reference; jk = klev always satisfies the test in
    # practice (the bottom layer is thinner than 60 hPa), and the mirror
    # keeps None so that a column which did not set it fails loudly instead
    # of reading a stale value.
    itopde = None
    for jk in range(klev, 0, -1):
        pmfdde_rate[jk] = _F(0.0)
        if _F(P_paph[klev + 1] - P_paph[jk]) < _F(60.0e2):
            itopde = jk

    # ---- the descent (:2623-2719) --------------------------------------
    for jk in range(3, klev + 1):
        zph = P_paph[jk]
        llo2 = ld and P_pmfd[jk - 1] < _F(0.0)
        if not llo2:
            continue
        assert itopde is not None, "itopde was never set for this column"

        # :2634-2640
        zentr = _F(_F(_F(_ENTRDD * P_pmfd[jk - 1])
                      * _F(P_pgeoh[jk - 1] - P_pgeoh[jk])) * c.zrg)
        zdmfen = zentr
        zdmfde = zentr

        # :2642-2652
        if jk > itopde:
            zdmfen = _F(0.0)
            zdmfde = _F(_F(P_pmfd[itopde]
                           * _F(P_paph[jk] - P_paph[jk - 1]))
                        / _F(P_paph[klev + 1] - P_paph[itopde]))

        # :2654-2665
        if jk <= itopde:
            zdz = _F(-_F(_F(P_pgeoh[jk - 1] - P_pgeoh[jk]) * c.zrg))
            zzentr = _F(_F(zoentr * zdz) * P_pmfd[jk - 1])
            zdmfen = _F(zdmfen + zzentr)
            zdmfen = _F(max(zdmfen, _F(_F(0.3) * P_pmfd[jk - 1])))
            zdmfen = _F(max(zdmfen,
                            _F(_F(-_F(_F(0.75) * P_pmfu[jk]))
                               - _F(P_pmfd[jk - 1] - zdmfde))))
            zdmfen = _F(min(zdmfen, _F(0.0)))

        # :2667-2682
        P_pmfd[jk] = _F(_F(P_pmfd[jk - 1] + zdmfen) - zdmfde)
        zseen = _F(_F(_F(c.cpd * P_ptenh[jk - 1]) + P_pgeoh[jk - 1])
                   * zdmfen)
        zqeen = _F(P_pqenh[jk - 1] * zdmfen)
        zsdde = _F(_F(_F(c.cpd * P_ptd[jk - 1]) + P_pgeoh[jk - 1])
                   * zdmfde)
        zqdde = _F(P_pqd[jk - 1] * zdmfde)
        zmfdsk = _F(_F(P_pmfds[jk - 1] + zseen) - zsdde)
        zmfdqk = _F(_F(P_pmfdq[jk - 1] + zqeen) - zqdde)
        inv = _F(_F(1.0) / _F(min(-_CMFCMIN, P_pmfd[jk])))
        P_pqd[jk] = _F(zmfdqk * inv)
        P_ptd[jk] = _F(_F(_F(zmfdsk * inv) - P_pgeoh[jk]) * c.rcpd)
        P_ptd[jk] = _F(min(_F(400.0), P_ptd[jk]))
        P_ptd[jk] = _F(max(_F(100.0), P_ptd[jk]))
        zcond = P_pqd[jk]

        # :2686  cuadjtqn, icall = 2
        nt_cuadjtqn2(P_ptd, P_pqd, jk, zph, c)

        # :2689-2714
        zcond = _F(zcond - P_pqd[jk])
        zbuo = _F(_F(P_ptd[jk] * _F(_F(1.0) + _F(c.vtmpc1 * P_pqd[jk])))
                  - _F(P_ptenh[jk] * _F(_F(1.0)
                                        + _F(c.vtmpc1 * P_pqenh[jk]))))
        if rfl > _F(0.0) and P_pmfu[jk] > _F(0.0):
            zrain = _F(rfl / P_pmfu[jk])
            zbuo = _F(zbuo - _F(P_ptd[jk] * zrain))
        if zbuo >= _F(0.0) or rfl <= _F(P_pmfd[jk] * zcond):
            P_pmfd[jk] = _F(0.0)
            zbuo = _F(0.0)
        P_pmfds[jk] = _F(_F(_F(c.cpd * P_ptd[jk]) + P_pgeoh[jk])
                         * P_pmfd[jk])
        P_pmfdq[jk] = _F(P_pqd[jk] * P_pmfd[jk])
        zdmfdp = _F(-_F(P_pmfd[jk] * zcond))
        P_pdmfdp[jk - 1] = zdmfdp
        rfl = _F(rfl + zdmfdp)

        # organised entrainment for the next level down
        zbuoyz = _F(zbuo / P_ptenh[jk])
        zbuoyz = _F(min(zbuoyz, _F(0.0)))
        zdz = _F(-_F(P_pgeo[jk - 1] - P_pgeo[jk]))
        zbuoy = _F(zbuoy + _F(zbuoyz * zdz))
        zoentr = _F(_F(_F(c.g * zbuoyz) * _F(0.5))
                    / _F(_F(1.0) + zbuoy))
        pmfdde_rate[jk] = _F(-zdmfde)

    return {"prfl": rfl,
            "ptd": P_ptd[1:nz + 1].copy(), "pqd": P_pqd[1:nz + 1].copy(),
            "pmfd": P_pmfd[1:nz + 1].copy(),
            "pmfds": P_pmfds[1:nz + 1].copy(),
            "pmfdq": P_pmfdq[1:nz + 1].copy(),
            "pdmfdp": P_pdmfdp[1:nz + 1].copy(),
            "pmfdde_rate": pmfdde_rate[1:nz + 1].copy()}


# ===========================================================================
# Slice 9: cuflxn -- the final convective fluxes (:2725-3060)
# ===========================================================================
# The largest routine left, and the one that turns mass fluxes into rain and
# snow: it converts the updraft/downdraft fluxes to their flux-form
# anomalies, tapers them below cloud base, melts snow, and evaporates
# falling precipitation into the sub-cloud layer.
#
# ktopm2 IS NOT A HORIZONTAL DEPENDENCY, and this is the one place in the
# scheme where that had to be derived rather than assumed.  cumastrn:565
# sets `itopm2 = kctop(jl)` INSIDE a `do jl` loop, so the value that
# survives the loop is the LAST column's cloud top -- a genuine horizontal
# leak.  It is passed here as ktopm2, intent(inout).  But :2877 does
#
#     ktopm2 = 2
#
# unconditionally, at routine top level (the preceding `enddo` closes the
# `do jl` loop above it), and NOTHING reads ktopm2 between cuflxn's entry
# and that line.  Every later use -- :2878, :2941, :2974, :3012 here, and
# :3107/:3137 in cudtdqn and :3191-3242 in cududvn -- is after it, and
# cumastrn calls cuflxn (:826) before cudtdqn (:922) and cududvn (:1026).
# So the leaked value is DEAD.  Re-derived from the source, second limb
# first, because the sibling claim about column independence in cuascn did
# not survive the same scrutiny.
#
# FOUR CLASS-1 DUMMIES -- lddraf, ktype, pmfu, pmfd -- and beyond those,
# pmfus/pmfuq/pmfds/pmfdq/plglac/pqsen/pdmfup/pdmfdp are all rewritten IN
# PLACE off their incoming values.  Nearly the whole argument list is
# load-bearing on entry, so the fixture captures it whole.
#
# pmflxr AND pmflxs ARE klev+1 ARRAYS.  :2920-2923 writes their klev+1 slot
# and the loop writes jk+1 up to klev+1; the surface slot is the scheme's
# actual surface rain and snow flux.  paph is read at klev+1 throughout.
#
# A THIRD zcons2.  :2857 declares `zcons2 = 3./(g*ztmst)` -- numerically the
# same as cuascn's and different from the closure's, under the same name for
# the third time in one file.

_ZTAUMEL = _F(18000.0)
_ZCUCOV = _F(0.05)


def np_ntiedtke_cuflxn(*, ldcum, lddraf, ktype, kcbot, kctop, kdtop, lndj,
                       pten, pqen, pqsen, ptenh, pqenh, paph, pap, pgeoh,
                       pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful,
                       plude, plglac, pdmfup, pdmfdp, pmfdde_rate,
                       ztmst, c=None):
    """cuflxn, one column.  Arrays 0-based in and out; body 1-based.

    ``paph`` must carry ``klev+1`` entries.  Returns ``pmflxr``/``pmflxs``
    with ``klev+1`` entries each, plus every array it rewrote in place.

    ``ldcum``, ``lddraf`` and ``ktype`` are scalars in and out: :2867-2868
    can clear the first two.
    """
    c = NtConstants() if c is None else c
    nz = pten.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_pten, P_pqen, P_pqsen = up(pten), up(pqen), up(pqsen)
    P_ptenh, P_pqenh = up(ptenh), up(pqenh)
    P_pap, P_pgeoh = up(pap), up(pgeoh)
    P_pmfu, P_pmfd = up(pmfu), up(pmfd)
    P_pmfus, P_pmfds = up(pmfus), up(pmfds)
    P_pmfuq, P_pmfdq = up(pmfuq), up(pmfdq)
    P_pmful, P_plude = up(pmful), up(plude)
    P_plglac = up(plglac)
    P_pdmfup, P_pdmfdp = up(pdmfup), up(pdmfdp)
    P_paph = _f1(nz)
    if _np32(paph).shape[0] < nz + 1:
        raise ValueError("cuflxn reads paph[klev+1]; pass klev+1 entries")
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]
    # klev+1 arrays in their own right.
    pmflxr, pmflxs = _f1(nz), _f1(nz)
    pdpmel = _f1(nz)
    # CONDITIONALLY WRITTEN, and declared WITHOUT an intent attribute, so
    # it was invisible to all three of the audit's original reports.  The
    # incoming value is cuddrafn's and is kept wherever the taper loop does
    # not reach.  The audit's fourth report exists because of this one.
    P_rate = up(pmfdde_rate)

    ld = bool(ldcum)
    ldd = bool(lddraf)
    kt = int(ktype)
    kb, ktop, kdt = int(kcbot), int(kctop), int(kdtop)

    # ---- constants (:2856-2860) ----------------------------------------
    zcons1a = _F(c.cpd / _F(_F(c.alf * c.g) * _ZTAUMEL))
    zcons2 = _F(_F(3.0) / _F(c.g * _F(ztmst)))
    zcpecons = _F(_F(5.44e-4) / c.g)

    # ---- 1.0 (:2865-2875) ----------------------------------------------
    prain = _F(0.0)
    if (not ld) or kdt < ktop:
        ldd = False
    if not ld:
        kt = 0
    idbas = klev
    rhevap = _F(0.7) if lndj == 1 else _F(0.9)

    # ktopm2 = 2, unconditionally.  See the header.
    ktopm2 = 2

    for jk in range(ktopm2, klev + 1):
        ikb = min(jk + 1, klev)
        pmflxr[jk] = _F(0.0)
        pmflxs[jk] = _F(0.0)
        pdpmel[jk] = _F(0.0)
        if ld and jk >= ktop:
            P_pmfus[jk] = _F(P_pmfus[jk] - _F(P_pmfu[jk] * _F(
                _F(c.cpd * P_ptenh[jk]) + P_pgeoh[jk])))
            P_pmfuq[jk] = _F(P_pmfuq[jk] - _F(P_pmfu[jk] * P_pqenh[jk]))
            P_plglac[jk] = _F(P_pmfu[jk] * P_plglac[jk])
            llddraf = ldd and jk >= kdt
            if llddraf and jk >= kdt:
                P_pmfds[jk] = _F(P_pmfds[jk] - _F(P_pmfd[jk] * _F(
                    _F(c.cpd * P_ptenh[jk]) + P_pgeoh[jk])))
                P_pmfdq[jk] = _F(P_pmfdq[jk] - _F(P_pmfd[jk] * P_pqenh[jk]))
            else:
                P_pmfd[jk] = _F(0.0)
                P_pmfds[jk] = _F(0.0)
                P_pmfdq[jk] = _F(0.0)
                P_pdmfdp[jk - 1] = _F(0.0)
            if llddraf and P_pmfd[jk] < _F(0.0) \
                    and abs(float(P_pmfd[ikb])) < 1.0e-20:
                idbas = jk
        else:
            P_pmfu[jk] = _F(0.0); P_pmfd[jk] = _F(0.0)
            P_pmfus[jk] = _F(0.0); P_pmfds[jk] = _F(0.0)
            P_pmfuq[jk] = _F(0.0); P_pmfdq[jk] = _F(0.0)
            P_pmful[jk] = _F(0.0); P_plglac[jk] = _F(0.0)
            P_pdmfup[jk - 1] = _F(0.0); P_pdmfdp[jk - 1] = _F(0.0)
            P_plude[jk - 1] = _F(0.0)

    pmflxr[klev + 1] = _F(0.0)
    pmflxs[klev + 1] = _F(0.0)

    # ---- the cloud-base taper (:2926-2938) -----------------------------
    if ld:
        ikb = kb
        ik = ikb + 1
        zzp = _F(_F(P_paph[klev + 1] - P_paph[ik])
                 / _F(P_paph[klev + 1] - P_paph[ikb]))
        if kt == 3:
            zzp = _F(zzp * zzp)
        P_pmfu[ik] = _F(P_pmfu[ikb] * zzp)
        P_pmfus[ik] = _F(_F(P_pmfus[ikb]
                            - _F(nt_foelhm(P_ptenh[ikb], c)
                                 * P_pmful[ikb])) * zzp)
        P_pmfuq[ik] = _F(_F(P_pmfuq[ikb] + P_pmful[ikb]) * zzp)
        P_pmful[ik] = _F(0.0)

    for jk in range(ktopm2, klev + 1):
        if ld and jk > kb + 1:
            ikb = kb + 1
            zzp = _F(_F(P_paph[klev + 1] - P_paph[jk])
                     / _F(P_paph[klev + 1] - P_paph[ikb]))
            if kt == 3:
                zzp = _F(zzp * zzp)
            P_pmfu[jk] = _F(P_pmfu[ikb] * zzp)
            P_pmfus[jk] = _F(P_pmfus[ikb] * zzp)
            P_pmfuq[jk] = _F(P_pmfuq[ikb] * zzp)
            P_pmful[jk] = _F(0.0)
        ik = idbas
        llddraf = ldd and jk > ik and ik < klev
        if llddraf and ik == kb + 1:
            zzp = _F(_F(P_paph[klev + 1] - P_paph[jk])
                     / _F(P_paph[klev + 1] - P_paph[ik]))
            if kt == 3:
                zzp = _F(zzp * zzp)
            P_pmfd[jk] = _F(P_pmfd[ik] * zzp)
            P_pmfds[jk] = _F(P_pmfds[ik] * zzp)
            P_pmfdq[jk] = _F(P_pmfdq[ik] * zzp)
            P_rate[jk] = _F(-_F(P_pmfd[jk - 1] - P_pmfd[jk]))
        elif llddraf and ik != kb + 1 and jk == ik + 1:
            P_rate[jk] = _F(-_F(P_pmfd[jk - 1] - P_pmfd[jk]))

    # ---- 2. melting and the rain/snow split (:2975-3011) ---------------
    for jk in range(ktopm2, klev + 1):
        if ld and jk >= ktop - 1:
            prain = _F(prain + P_pdmfup[jk])
            if pmflxs[jk] > _F(0.0) and P_pten[jk] > _TMELT:
                zcons1 = _F(zcons1a * _F(_F(1.0) + _F(
                    _F(0.5) * _F(P_pten[jk] - _TMELT))))
                zfac = _F(zcons1 * _F(P_paph[jk + 1] - P_paph[jk]))
                zsnmlt = _F(min(pmflxs[jk],
                                _F(zfac * _F(P_pten[jk] - _TMELT))))
                pdpmel[jk] = zsnmlt
                P_pqsen[jk] = _F(nt_foeewm(
                    _F(P_pten[jk] - _F(zsnmlt / zfac)), c) / P_pap[jk])
            zalfaw = nt_foealfa(P_pten[jk])
            # No liquid precipitation above the melting level.
            if P_pten[jk] < _TMELT and zalfaw > _F(0.0):
                P_plglac[jk] = _F(P_plglac[jk] + _F(zalfaw * _F(
                    P_pdmfup[jk] + P_pdmfdp[jk])))
                zalfaw = _F(0.0)
            pmflxr[jk + 1] = _F(_F(pmflxr[jk] + _F(zalfaw * _F(
                P_pdmfup[jk] + P_pdmfdp[jk]))) + pdpmel[jk])
            pmflxs[jk + 1] = _F(_F(pmflxs[jk] + _F(
                _F(_F(1.0) - zalfaw) * _F(P_pdmfup[jk] + P_pdmfdp[jk])))
                - pdpmel[jk])
            if _F(pmflxr[jk + 1] + pmflxs[jk + 1]) < _F(0.0):
                P_pdmfdp[jk] = _F(-_F(_F(pmflxr[jk] + pmflxs[jk])
                                      + P_pdmfup[jk]))
                pmflxr[jk + 1] = _F(0.0)
                pmflxs[jk + 1] = _F(0.0)
                pdpmel[jk] = _F(0.0)
            elif pmflxr[jk + 1] < _F(0.0):
                pmflxs[jk + 1] = _F(pmflxs[jk + 1] + pmflxr[jk + 1])
                pmflxr[jk + 1] = _F(0.0)
            elif pmflxs[jk + 1] < _F(0.0):
                pmflxr[jk + 1] = _F(pmflxr[jk + 1] + pmflxs[jk + 1])
                pmflxs[jk + 1] = _F(0.0)

    # ---- the sub-cloud evaporation (:3012-3057) ------------------------
    for jk in range(ktopm2, klev + 1):
        if ld and jk >= kb:
            zrfl = _F(pmflxr[jk] + pmflxs[jk])
            if zrfl > _F(1.0e-20):
                zdrfl1 = _F(_F(_F(_F(zcpecons * _F(max(
                    _F(0.0), _F(P_pqsen[jk] - P_pqen[jk])))) * _ZCUCOV)
                    * _pow32(_F(_F(_F(np.sqrt(_F(
                        P_paph[jk] / P_paph[klev + 1])))
                        / _F(5.09e-3)) * _F(zrfl / _ZCUCOV)), _F(0.5777)))
                    * _F(P_paph[jk + 1] - P_paph[jk]))
                zrnew = _F(zrfl - zdrfl1)
                zrmin = _F(zrfl - _F(_F(_ZCUCOV * _F(max(_F(0.0), _F(
                    _F(rhevap * P_pqsen[jk]) - P_pqen[jk]))))
                    * _F(zcons2 * _F(P_paph[jk + 1] - P_paph[jk]))))
                zrnew = _F(max(zrnew, zrmin))
                zrfln = _F(max(zrnew, _F(0.0)))
                zdrfl = _F(min(_F(0.0), _F(zrfln - zrfl)))
                zdenom = _F(_F(1.0) / _F(max(_F(1.0e-20),
                                             _F(pmflxr[jk] + pmflxs[jk]))))
                zalfaw = nt_foealfa(P_pten[jk])
                if P_pten[jk] < _TMELT:
                    zalfaw = _F(0.0)
                zpdr = _F(zalfaw * P_pdmfdp[jk])
                zpds = _F(_F(_F(1.0) - zalfaw) * P_pdmfdp[jk])
                pmflxr[jk + 1] = _F(_F(_F(pmflxr[jk] + zpdr) + pdpmel[jk])
                                    + _F(_F(zdrfl * pmflxr[jk]) * zdenom))
                pmflxs[jk + 1] = _F(_F(_F(pmflxs[jk] + zpds) - pdpmel[jk])
                                    + _F(_F(zdrfl * pmflxs[jk]) * zdenom))
                P_pdmfup[jk] = _F(P_pdmfup[jk] + zdrfl)
                if _F(pmflxr[jk + 1] + pmflxs[jk + 1]) < _F(0.0):
                    P_pdmfup[jk] = _F(P_pdmfup[jk]
                                      - _F(pmflxr[jk + 1] + pmflxs[jk + 1]))
                    pmflxr[jk + 1] = _F(0.0)
                    pmflxs[jk + 1] = _F(0.0)
                    pdpmel[jk] = _F(0.0)
                elif pmflxr[jk + 1] < _F(0.0):
                    pmflxs[jk + 1] = _F(pmflxs[jk + 1] + pmflxr[jk + 1])
                    pmflxr[jk + 1] = _F(0.0)
                elif pmflxs[jk + 1] < _F(0.0):
                    pmflxr[jk + 1] = _F(pmflxr[jk + 1] + pmflxs[jk + 1])
                    pmflxs[jk + 1] = _F(0.0)
            else:
                pmflxr[jk + 1] = _F(0.0)
                pmflxs[jk + 1] = _F(0.0)
                P_pdmfdp[jk] = _F(0.0)
                pdpmel[jk] = _F(0.0)

    return {"ldcum": ld, "lddraf": ldd, "ktype": kt, "kdtop": kdt,
            "idbas": idbas, "prain": prain, "ktopm2": ktopm2,
            "pmfu": P_pmfu[1:nz + 1].copy(),
            "pmfd": P_pmfd[1:nz + 1].copy(),
            "pmfus": P_pmfus[1:nz + 1].copy(),
            "pmfds": P_pmfds[1:nz + 1].copy(),
            "pmfuq": P_pmfuq[1:nz + 1].copy(),
            "pmfdq": P_pmfdq[1:nz + 1].copy(),
            "pmful": P_pmful[1:nz + 1].copy(),
            "plude": P_plude[1:nz + 1].copy(),
            "plglac": P_plglac[1:nz + 1].copy(),
            "pdmfup": P_pdmfup[1:nz + 1].copy(),
            "pdmfdp": P_pdmfdp[1:nz + 1].copy(),
            "pdpmel": pdpmel[1:nz + 1].copy(),
            "pqsen": P_pqsen[1:nz + 1].copy(),
            "pmfdde_rate": P_rate[1:nz + 1].copy(),
            "pmflxr": pmflxr[1:nz + 2].copy(),
            "pmflxs": pmflxs[1:nz + 2].copy()}


# ===========================================================================
# Slice 10: cudtdqn -- the heat and moisture tendencies (:3064-3148)
# ===========================================================================
# Where the mass fluxes become RTHCUTEN and RQVCUTEN.  Compact, and the
# first routine in this port whose hazards were known BEFORE it was written:
# the aliasing audit's third report named ptent and ptenq as self-referential
# accumulators (:3140-3141, `ptent = ptent + zdtdt`) while cuflxn was still
# being graded.
#
# THE ACCUMULATION IS INTO A NON-ZERO ARRAY.  Measured at this routine's own
# entry capture: ptent and ptenq are non-zero on 4,428 of 5,292 rows,
# because cu_ntiedtke_run:273-276 seeds them with the FORCING (ptf/pqvf) and
# saves copies, then :309-310 differences against those copies so only the
# convective increment escapes.  See §17.  So the mirror MUST add, not
# assign, and the caller must difference.
#
# pcte is conditionally written and its entry value is captured too, though
# it is measured zero on all 5,292 rows -- recorded rather than assumed.
#
# ktopm2 is 2 here for the same reason as in cuflxn: cuflxn overwrote it
# unconditionally before cudtdqn runs.  It is a parameter of this mirror
# because cudtdqn declares it intent(in) and genuinely reads it, unlike
# cuflxn which overwrites it.


def np_ntiedtke_cudtdqn(*, ktopm2, ldcum, paph, pten, plglac, plude,
                        pmfus, pmfds, pmfuq, pmfdq, pmful, pdmfup,
                        pdmfdp, pdpmel, ptent, ptenq, pcte, c=None):
    """cudtdqn, one column.  Arrays 0-based in and out; body 1-based.

    ``paph`` must carry ``klev+1`` entries: ``zdp`` reads ``paph(jk+1)``.
    ``ptent`` and ``ptenq`` are ACCUMULATED into and returned.
    """
    c = NtConstants() if c is None else c
    nz = pten.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_pten, P_plglac, P_plude = up(pten), up(plglac), up(plude)
    P_pmfus, P_pmfds = up(pmfus), up(pmfds)
    P_pmfuq, P_pmfdq = up(pmfuq), up(pmfdq)
    P_pmful, P_pdmfup = up(pmful), up(pdmfup)
    P_pdmfdp, P_pdpmel = up(pdmfdp), up(pdpmel)
    P_ptent, P_ptenq, P_pcte = up(ptent), up(ptenq), up(pcte)
    P_paph = _f1(nz)
    if _np32(paph).shape[0] < nz + 1:
        raise ValueError("cudtdqn reads paph[klev+1]; pass klev+1 entries")
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]

    if not ldcum:
        return {"ptent": P_ptent[1:nz + 1].copy(),
                "ptenq": P_ptenq[1:nz + 1].copy(),
                "pcte": P_pcte[1:nz + 1].copy()}

    zdp, zdtdt, zdqdt = _f1(nz), _f1(nz), _f1(nz)

    # ---- 1.0 (:3096-3103) ----------------------------------------------
    for jk in range(1, klev + 1):
        zdp[jk] = _F(c.g / _F(P_paph[jk + 1] - P_paph[jk]))

    # ---- 2.0 (:3106-3132) ----------------------------------------------
    for jk in range(ktopm2, klev + 1):
        zalv = nt_foelhm(P_pten[jk], c)
        if jk < klev:
            inner = _F(_F(_F(_F(P_pmful[jk + 1] - P_pmful[jk])
                              - P_plude[jk]) - P_pdmfup[jk]) - P_pdmfdp[jk])
            big = _F(_F(_F(_F(_F(_F(P_pmfus[jk + 1] - P_pmfus[jk])
                                  + P_pmfds[jk + 1]) - P_pmfds[jk])
                            + _F(c.alf * P_plglac[jk]))
                        - _F(c.alf * P_pdpmel[jk])) - _F(zalv * inner))
            zdtdt[jk] = _F(_F(zdp[jk] * c.rcpd) * big)
            zdqdt[jk] = _F(zdp[jk] * _F(_F(_F(_F(_F(_F(_F(_F(
                P_pmfuq[jk + 1] - P_pmfuq[jk]) + P_pmfdq[jk + 1])
                - P_pmfdq[jk]) + P_pmful[jk + 1]) - P_pmful[jk])
                - P_plude[jk]) - P_pdmfup[jk]) - P_pdmfdp[jk]))
        else:
            big = _F(_F(_F(P_pmfus[jk] + P_pmfds[jk])
                          + _F(c.alf * P_pdpmel[jk]))
                     - _F(zalv * _F(_F(_F(P_pmful[jk] + P_pdmfup[jk])
                                        + P_pdmfdp[jk]) + P_plude[jk])))
            zdtdt[jk] = _F(-_F(_F(zdp[jk] * c.rcpd) * big))
            zdqdt[jk] = _F(-_F(zdp[jk] * _F(
                _F(_F(P_pmfuq[jk] + P_plude[jk]) + P_pmfdq[jk])
                + _F(_F(P_pmful[jk] + P_pdmfup[jk]) + P_pdmfdp[jk]))))

    # ---- 3.0 (:3136-3145).  ADD, do not assign. ------------------------
    for jk in range(ktopm2, klev + 1):
        P_ptent[jk] = _F(P_ptent[jk] + zdtdt[jk])
        P_ptenq[jk] = _F(P_ptenq[jk] + zdqdt[jk])
        P_pcte[jk] = _F(zdp[jk] * P_plude[jk])

    return {"ptent": P_ptent[1:nz + 1].copy(),
            "ptenq": P_ptenq[1:nz + 1].copy(),
            "pcte": P_pcte[1:nz + 1].copy()}


# ===========================================================================
# Slice 11: cududvn -- the momentum tendencies (:3152-3252)
# ===========================================================================
# The last routine, and the one that closes the momentum story this port
# kept running into.  cuascn, cudlfsn and cuddrafn were each found to take
# puu/pvu/pud/pvd and NEVER WRITE THEM -- three separate findings, each
# gated on the mirror's shape because the oracle cannot tell "left alone"
# from "not implemented".
#
# CORRECTED.  This used to conclude "cuinin sets them, nothing between
# touches them, cududvn reads them."  THE CONCLUSION WAS WRONG: cumastrn
# :927-995 rewrites zuu/zvu/zud/zvd as the updraft and downdraft momentum
# profiles, and puu differs on 1,926 of 5,292 slots between cuinin's exit
# and cududvn's entry.  The three local findings were each right; chaining
# them into a claim about the whole path was not, because the glue between
# the routines was never checked.
#
# THE VALUES HERE ARE STILL RIGHT, and only because the fixture captures
# them at cududvn's OWN call site rather than at cuinin's exit.  That is
# the capture-first rule producing a correct answer despite incorrect
# reasoning -- which is the strongest argument for it in this port, and
# better evidence than the :996 instance where the reasoning happened to
# hold.
#
# ptenu/ptenv are self-referential accumulators (:3245-3246), named by the
# audit's third report before this routine was written.  UNLIKE ptent/ptenq
# they are seeded ZERO -- cu_ntiedtke_run:258-259 sets pvom = pvol = 0.
# before the cumastrn call -- so accumulate and replace coincide here.  The
# mirror still adds, and the fixture still captures the seed, because
# "measured zero" and "assumed zero" are the distinction §17 is about.
#
# THE MASS FLUXES ARE THE SCALED PAIR.  cumastrn:833-915 rescales the
# updraft and downdraft fluxes into zmfuus/zmfdus, and it is those -- not
# zmfu/zmfd -- that reach here.  Feeding the unscaled pair would be wrong
# on every column the rescaling touched, and silently so.


def np_ntiedtke_cududvn(*, ktopm2, ktype, kcbot, ldcum, paph, puen, pven,
                        pmfu, pmfd, puu, pud, pvu, pvd, ptenu, ptenv,
                        c=None):
    """cududvn, one column.  Arrays 0-based in and out; body 1-based.

    ``pmfu``/``pmfd`` are the SCALED fluxes (cumastrn's zmfuus/zmfdus).
    ``ptenu``/``ptenv`` are accumulated into and returned.
    """
    c = NtConstants() if c is None else c
    nz = puen.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_puen, P_pven = up(puen), up(pven)
    P_pmfu, P_pmfd = up(pmfu), up(pmfd)
    P_puu, P_pud = up(puu), up(pud)
    P_pvu, P_pvd = up(pvu), up(pvd)
    P_ptenu, P_ptenv = up(ptenu), up(ptenv)
    P_paph = _f1(nz)
    if _np32(paph).shape[0] < nz + 1:
        raise ValueError("cududvn reads paph[klev+1]; pass klev+1 entries")
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]

    if not ldcum:
        return {"ptenu": P_ptenu[1:nz + 1].copy(),
                "ptenv": P_ptenv[1:nz + 1].copy()}

    zdp = _f1(nz)
    zmfuu, zmfuv = _f1(nz), _f1(nz)
    zmfdu, zmfdv = _f1(nz), _f1(nz)
    zdudt, zdvdt = _f1(nz), _f1(nz)

    # ---- setup (:3179-3187).  zuen/zven are copies of puen/pven. -------
    for jk in range(1, klev + 1):
        zdp[jk] = _F(c.g / _F(P_paph[jk + 1] - P_paph[jk]))

    # ---- 1.0 the fluxes (:3191-3201) -----------------------------------
    for jk in range(ktopm2, klev + 1):
        ik = jk - 1
        zmfuu[jk] = _F(P_pmfu[jk] * _F(P_puu[jk] - P_puen[ik]))
        zmfuv[jk] = _F(P_pmfu[jk] * _F(P_pvu[jk] - P_pven[ik]))
        zmfdu[jk] = _F(P_pmfd[jk] * _F(P_pud[jk] - P_puen[ik]))
        zmfdv[jk] = _F(P_pmfd[jk] * _F(P_pvd[jk] - P_pven[ik]))

    # ---- linear fluxes below cloud (:3203-3215) ------------------------
    for jk in range(ktopm2, klev + 1):
        if jk > kcbot:
            ikb = kcbot
            zzp = _F(_F(P_paph[klev + 1] - P_paph[jk])
                     / _F(P_paph[klev + 1] - P_paph[ikb]))
            if ktype == 3:
                zzp = _F(zzp * zzp)
            zmfuu[jk] = _F(zmfuu[ikb] * zzp)
            zmfuv[jk] = _F(zmfuv[ikb] * zzp)
            zmfdu[jk] = _F(zmfdu[ikb] * zzp)
            zmfdv[jk] = _F(zmfdv[ikb] * zzp)

    # ---- 2.0 the tendencies (:3219-3238) -------------------------------
    for jk in range(ktopm2, klev + 1):
        if jk < klev:
            ik = jk + 1
            zdudt[jk] = _F(zdp[jk] * _F(_F(_F(zmfuu[ik] - zmfuu[jk])
                                            + zmfdu[ik]) - zmfdu[jk]))
            zdvdt[jk] = _F(zdp[jk] * _F(_F(_F(zmfuv[ik] - zmfuv[jk])
                                            + zmfdv[ik]) - zmfdv[jk]))
        else:
            zdudt[jk] = _F(-_F(zdp[jk] * _F(zmfuu[jk] + zmfdu[jk])))
            zdvdt[jk] = _F(-_F(zdp[jk] * _F(zmfuv[jk] + zmfdv[jk])))

    # ---- 3.0 (:3242-3249).  ADD, do not assign. ------------------------
    for jk in range(ktopm2, klev + 1):
        P_ptenu[jk] = _F(P_ptenu[jk] + zdudt[jk])
        P_ptenv[jk] = _F(P_ptenv[jk] + zdvdt[jk])

    return {"ptenu": P_ptenu[1:nz + 1].copy(),
            "ptenv": P_ptenv[1:nz + 1].copy()}


# ===========================================================================
# Slice 12: the cloud-depth check -- cumastrn:562-590
# ===========================================================================
# THE FIRST PIECE OF ORCHESTRATION, and the one the whole port exists for.
# Thirty lines between cuascn and cudlfsn that nothing owned, containing:
#
#   :566-568  THE KTYPE FLIP.  A deep column whose cloud is shallower than
#             200 hPa becomes ktype 2; a shallow one that is deeper becomes
#             ktype 1.  ktype selects scale_fac (deep) or scale_fac2
#             (shallow) in the closure, which is the entire reason for the
#             port.  This is the fifth failure's line: it sits between two
#             stages that look adjacent, and feeding cuascn's ktype to the
#             closure runs the wrong arm.
#   :580-588  the downdraft-array zeroing that four class-2 excuses in
#             test_ntiedtke_aliasing_audit.py rest on.  Owning it clears
#             those debts.
#
# No new capture was needed: this block's inputs are cuascn's outputs and
# its outputs are cudlfsn's and cuddrafn's captured entry state, so it is
# graded against rows that already exist.

_ZDNOPRC = _F(2.0e4)


def np_ntiedtke_cloud_depth(*, ldcum, ktype, kcbot, kctop, kctop0,
                            paph, pdmfup, nz=None):
    """cumastrn:562-590, one column.  Arrays 0-based in and out.

    ``paph`` is indexed 1-based internally because kcbot/kctop are.
    Returns the flipped ``ktype``, ``ictop0``, the summed ``prfl`` and the
    five zeroed downdraft arrays.
    """
    n = pdmfup.shape[0] if nz is None else nz
    P_paph = _f1(n)
    P_paph[1:n + 2] = _np32(paph)[:n + 1]

    kt = int(ktype)
    kt0 = int(kctop0)
    if ldcum:
        ikb = int(kcbot)
        itopm2 = int(kctop)
        zpbmpt = _F(P_paph[ikb] - P_paph[itopm2])
        if kt == 1 and zpbmpt < _ZDNOPRC:
            kt = 2
        if kt == 2 and zpbmpt >= _ZDNOPRC:
            kt = 1
        kt0 = itopm2

    # :571-577.  zrfl starts at level 1 and sums upward through klev.
    up = _np32(pdmfup)
    zrfl = _F(up[0])
    for jk in range(1, n):
        zrfl = _F(zrfl + up[jk])

    z = np.zeros(n, dtype=np.float32)
    return {"ktype": kt, "ictop0": kt0, "prfl": zrfl,
            "pmfd": z.copy(), "pmfds": z.copy(), "pmfdq": z.copy(),
            "pdmfdp": z.copy(), "pdpmel": z.copy()}


# ===========================================================================
# Slice 13: the adjustments block -- cumastrn:833-919
# ===========================================================================
# Between cuflxn and cudtdqn.  Five things, and the first is a stability
# rescale that has nothing to do with the MOMENTUM rescale at :996-1016 --
# the two use the same local name `zmfs` for different quantities, computed
# against different limits, and I attributed this range's job to the wrong
# one once already.
#
#   :838-847   the DOWNDRAFT stability cap.  zmfs is the largest factor
#              that keeps |pmfd| under 0.98*pmfu at every level.
#   :849-861   apply it, and carry the precipitation the downdraft no
#              longer transports into pmflxr through zmfuub -- an
#              accumulator that runs DOWNWARD through the column.
#   :863-880   entrainment-rate floors, and zdmfup recomputed from the
#              precipitation-flux divergence.
#   :883-892   the downdraft-top humidity guard.
#   :896-913   the near-cloud-top humidity guard, which can REDUCE plude.
#
# Its inputs are cuflxn's outputs and its outputs are cudtdqn's captured
# entry state plus one new capture, so it needed no new replication.


def np_ntiedtke_adjust(*, ldcum, loddraf, idtop, kctop, kcbot, ztmst,
                       paph, pqen, pmfu, pmfd, pmfds, pmfdq, pmfuq,
                       pmful, plude, pdmfup, pdmfdp, pmfdde_rate,
                       pmfude_rate, pmflxr, pmflxs, c=None):
    """cumastrn:833-919, one column.  Arrays 0-based in and out.

    ``paph``, ``pmflxr`` and ``pmflxs`` must carry ``klev+1`` entries.
    """
    c = NtConstants() if c is None else c
    nz = pqen.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    def upe(a):
        o = _f1(nz)
        o[1:nz + 2] = _np32(a)[:nz + 1]
        return o

    P_pqen, P_pmfu = up(pqen), up(pmfu)
    P_pmfd, P_pmfds, P_pmfdq = up(pmfd), up(pmfds), up(pmfdq)
    P_pmfuq, P_pmful, P_plude = up(pmfuq), up(pmful), up(plude)
    P_pdmfup, P_pdmfdp = up(pdmfup), up(pdmfdp)
    P_ddr, P_udr = up(pmfdde_rate), up(pmfude_rate)
    P_paph, P_pmflxr, P_pmflxs = upe(paph), upe(pmflxr), upe(pmflxs)

    ld, ldd = bool(ldcum), bool(loddraf)
    idt, ktop, kb = int(idtop), int(kctop), int(kcbot)

    # ---- :834-847  the downdraft stability cap -------------------------
    zmfs = _F(1.0)
    zmfuub = _F(0.0)
    for jk in range(2, klev + 1):
        if ldd and jk >= idt - 1:
            zmfmax = _F(P_pmfu[jk] * _F(0.98))
            if _F(_F(P_pmfd[jk] + zmfmax) + _F(1.0e-15)) < _F(0.0):
                zmfs = _F(min(zmfs, _F(_F(-zmfmax) / P_pmfd[jk])))

    # ---- :849-861  apply it ---------------------------------------------
    # zmfuub runs DOWNWARD through the column: the precipitation the
    # downdraft no longer carries is handed to pmflxr at the level below.
    for jk in range(2, klev + 1):
        if zmfs < _F(1.0) and jk >= idt - 1:
            P_pmfd[jk] = _F(P_pmfd[jk] * zmfs)
            P_pmfds[jk] = _F(P_pmfds[jk] * zmfs)
            P_pmfdq[jk] = _F(P_pmfdq[jk] * zmfs)
            P_ddr[jk] = _F(P_ddr[jk] * zmfs)
            zmfuub = _F(zmfuub - _F(_F(_F(1.0) - zmfs) * P_pdmfdp[jk]))
            P_pmflxr[jk + 1] = _F(P_pmflxr[jk + 1] + zmfuub)
            P_pdmfdp[jk] = _F(P_pdmfdp[jk] * zmfs)

    # ---- :863-880  entrainment floors, and zdmfup from the flux ---------
    for jk in range(2, klev):
        if ldd and jk >= idt - 1:
            zerate = _F(_F(_F(-P_pmfd[jk]) + P_pmfd[jk - 1]) + P_ddr[jk])
            if zerate < _F(0.0):
                P_ddr[jk] = _F(P_ddr[jk] - zerate)
        if ld and jk >= ktop - 1:
            zerate = _F(_F(_F(P_pmfu[jk] - P_pmfu[jk + 1])) + P_udr[jk])
            if zerate < _F(0.0):
                P_udr[jk] = _F(P_udr[jk] - zerate)
            P_pdmfup[jk] = _F(_F(_F(P_pmflxr[jk + 1] + P_pmflxs[jk + 1])
                                 - P_pmflxr[jk]) - P_pmflxs[jk])
            P_pdmfdp[jk] = _F(0.0)

    # ---- :883-892  the downdraft-top humidity guard ---------------------
    if ldd:
        jk = idt
        ik = min(jk + 1, klev)
        if P_pmfdq[jk] < _F(_F(0.3) * P_pmfdq[ik]):
            P_pmfdq[jk] = _F(_F(0.3) * P_pmfdq[ik])

    # ---- :896-913  the near-cloud-top humidity guard --------------------
    for jk in range(2, klev + 1):
        if ld and jk >= ktop - 1 and jk < kb:
            zdz = _F(_F(_F(ztmst) * c.g)
                     / _F(P_paph[jk + 1] - P_paph[jk]))
            zmfa = _F(_F(_F(_F(_F(_F(P_pmfuq[jk + 1] + P_pmfdq[jk + 1])
                                  - P_pmfuq[jk]) - P_pmfdq[jk])
                            + P_pmful[jk + 1]) - P_pmful[jk])
                      + P_pdmfup[jk])
            zmfa = _F(_F(zmfa - P_plude[jk]) * zdz)
            if _F(P_pqen[jk] + zmfa) < _F(0.0):
                P_plude[jk] = _F(P_plude[jk] + _F(
                    _F(_F(2.0) * _F(P_pqen[jk] + zmfa)) / zdz))
            if P_plude[jk] < _F(0.0):
                P_plude[jk] = _F(0.0)
        if not ld:
            P_udr[jk] = _F(0.0)
        if abs(float(P_pmfd[jk - 1])) < 1.0e-20:
            P_ddr[jk] = _F(0.0)

    return {"prsfc": P_pmflxr[klev + 1], "pssfc": P_pmflxs[klev + 1],
            "zmfs": zmfs,
            "pmfd": P_pmfd[1:nz + 1].copy(),
            "pmfds": P_pmfds[1:nz + 1].copy(),
            "pmfdq": P_pmfdq[1:nz + 1].copy(),
            "pdmfdp": P_pdmfdp[1:nz + 1].copy(),
            "pdmfup": P_pdmfup[1:nz + 1].copy(),
            "pmfdde_rate": P_ddr[1:nz + 1].copy(),
            "pmfude_rate": P_udr[1:nz + 1].copy(),
            "plude": P_plude[1:nz + 1].copy(),
            "pmflxr": P_pmflxr[1:nz + 2].copy(),
            "pmflxs": P_pmflxs[1:nz + 2].copy()}


# ===========================================================================
# Slice 14: the momentum mass-flux rescale -- cumastrn:996-1016
# ===========================================================================
# THE ONLY CONSUMER OF `zcons` IN THE ENTIRE SCHEME.
#
# Every other mass-flux cap in New Tiedtke uses `zcons2` = 3/(g*dt).  This
# one uses `zcons` = 1/(g*dt) -- one character away, both declared in
# cumastrn, three times tighter.  Getting it wrong makes the momentum
# rescale three times too permissive, and the result is finite, plausible
# and off by a fixed ratio: the least visible arithmetic error available.
# Gated by tests/test_ntiedtke_constant_family.py, built before this slice
# rather than after the second misreading.
#
# It produces zmfuus/zmfdus, and it is THOSE, not pmfu/pmfd, that cududvn
# consumes.  A cududvn fed the unscaled pair is wrong on exactly the
# columns the rescale touched.


def np_ntiedtke_momentum_rescale(*, ldcum, kctop, paph, pmfu, pmfd,
                                 ztmst, c=None):
    """cumastrn:996-1016, one column.  Arrays 0-based in and out.

    Returns ``zmfuus``/``zmfdus`` and the scalar ``zmfs`` that produced
    them.  Note the second loop runs 1..klev and assigns the UNSCALED
    value first, so a level outside the cloud carries pmfu/pmfd unchanged
    rather than zero.
    """
    c = NtConstants() if c is None else c
    nz = pmfu.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_pmfu, P_pmfd = up(pmfu), up(pmfd)
    P_paph = _f1(nz)
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]

    # zcons, NOT zcons2.  See the header.
    zcons = _F(_F(1.0) / _F(c.g * _F(ztmst)))
    ktop = int(kctop)

    zmfs = _F(1.0)
    if ldcum:
        for jk in range(2, klev + 1):
            if jk >= ktop - 1:
                zmfmax = _F(_F(P_paph[jk] - P_paph[jk - 1]) * zcons)
                if P_pmfu[jk] > zmfmax and jk >= ktop:
                    zmfs = _F(min(zmfs, _F(zmfmax / P_pmfu[jk])))

    zmfuus, zmfdus = _f1(nz), _f1(nz)
    for jk in range(1, klev + 1):
        zmfuus[jk] = P_pmfu[jk]
        zmfdus[jk] = P_pmfd[jk]
        if ldcum and jk >= ktop - 1:
            zmfuus[jk] = _F(P_pmfu[jk] * zmfs)
            zmfdus[jk] = _F(P_pmfd[jk] * zmfs)

    return {"zmfs": zmfs,
            "zmfuus": zmfuus[1:nz + 1].copy(),
            "zmfdus": zmfdus[1:nz + 1].copy()}


# ===========================================================================
# Slice 15: the updraft rescale and two cleanups -- cumastrn:743-819
# ===========================================================================
# WHERE THE CLOSURE'S ANSWER ACTUALLY LANDS.  :745 forms
#
#     zmfs = zmfub1 / max(cmfcmin, zmfub)
#
# and the block applies it to the whole updraft.  zmfub1 is what the CAPE
# closure produced -- the quantity scale_fac and scale_fac2 act on, and the
# reason this port exists.  Section 9 measured its retention; this is the
# code that spends it.
#
# THREE SUB-BLOCKS, and the middle one is the one to read twice:
#
#   :747-761  taper pmfu below cloud base, then CAP zmfs so no level
#             exceeds zcons2.  Note the cap tests `pmfu*zmfs > zmfmax` but
#             divides by the UNSCALED pmfu -- so zmfs shrinks toward the
#             tightest level, and re-reads its own running value.
#   :762-774  apply zmfs to the seven updraft arrays.
#   :777-783  6.6: a ktype 2 column with kcbot == kctop at the bottom of
#             the domain is switched off entirely.
#   :798-818  6.7: downdraft fluxes zeroed above cloud top, and idtop
#             pushed below it.
#
# The dead block at :786-802 sits between 6.6 and 6.7 and is skipped: both
# its guards are `.true.` parameters.  See
# test_ntiedtke_cumastrn_ownership.py.


def np_ntiedtke_updraft_scale(*, ldcum, ktype, kcbot, kctop, idtop,
                              loddraf, zmfub1, zmfub, paph, pmfu, pmfus,
                              pmfuq, pmful, pdmfup, plude, pmfude_rate,
                              pmfd, pmfds, pmfdq, pdmfdp, pmfdde_rate,
                              ztmst, c=None):
    """cumastrn:743-819, one column.  Arrays 0-based in and out."""
    c = NtConstants() if c is None else c
    nz = pmfu.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_pmfu, P_pmfus = up(pmfu), up(pmfus)
    P_pmfuq, P_pmful = up(pmfuq), up(pmful)
    P_pdmfup, P_plude = up(pdmfup), up(plude)
    P_udr = up(pmfude_rate)
    P_pmfd, P_pmfds, P_pmfdq = up(pmfd), up(pmfds), up(pmfdq)
    P_pdmfdp, P_ddr = up(pdmfdp), up(pmfdde_rate)
    P_paph = _f1(nz)
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]

    ld = bool(ldcum)
    kt = int(ktype)
    kb, ktop, idt = int(kcbot), int(kctop), int(idtop)
    ldd = bool(loddraf)
    zcons2 = _F(_F(3.0) / _F(c.g * _F(ztmst)))

    # ---- :743-746  the closure's factor --------------------------------
    zmfs = _F(1.0)
    if ld:
        zmfs = _F(_F(zmfub1) / _F(max(_CMFCMIN, _F(zmfub))))

    # ---- :747-761  taper, then cap -------------------------------------
    for jk in range(2, klev + 1):
        if ld and jk >= ktop - 1:
            ikb = kb
            if jk > ikb:
                zdz = _F(_F(P_paph[klev + 1] - P_paph[jk])
                         / _F(P_paph[klev + 1] - P_paph[ikb]))
                P_pmfu[jk] = _F(P_pmfu[ikb] * zdz)
            zmfmax = _F(_F(P_paph[jk] - P_paph[jk - 1]) * zcons2)
            # The test scales pmfu; the division does NOT.  Transcribed as
            # written -- the two are different once zmfs < 1.
            if _F(P_pmfu[jk] * zmfs) > zmfmax:
                zmfs = _F(min(zmfs, _F(zmfmax / P_pmfu[jk])))

    # ---- :762-774  apply ------------------------------------------------
    for jk in range(2, klev + 1):
        if ld and jk <= kb and jk >= ktop - 1:
            P_pmfu[jk] = _F(P_pmfu[jk] * zmfs)
            P_pmfus[jk] = _F(P_pmfus[jk] * zmfs)
            P_pmfuq[jk] = _F(P_pmfuq[jk] * zmfs)
            P_pmful[jk] = _F(P_pmful[jk] * zmfs)
            P_pdmfup[jk] = _F(P_pdmfup[jk] * zmfs)
            P_plude[jk] = _F(P_plude[jk] * zmfs)
            P_udr[jk] = _F(P_udr[jk] * zmfs)

    # ---- 6.6 (:777-783) -------------------------------------------------
    if kt == 2 and kb == ktop and kb >= klev - 1:
        ld = False
        kt = 0

    # ---- 6.7 (:798-818) -------------------------------------------------
    if ldd and idt <= ktop:
        idt = ktop + 1
    for jk in range(2, klev + 1):
        if ldd:
            if jk < idt:
                P_pmfd[jk] = _F(0.0)
                P_pmfds[jk] = _F(0.0)
                P_pmfdq[jk] = _F(0.0)
                P_ddr[jk] = _F(0.0)
                P_pdmfdp[jk] = _F(0.0)
            elif jk == idt:
                P_ddr[jk] = _F(0.0)

    return {"ldcum": ld, "ktype": kt, "idtop": idt, "zmfs": zmfs,
            "pmfu": P_pmfu[1:nz + 1].copy(),
            "pmfus": P_pmfus[1:nz + 1].copy(),
            "pmfuq": P_pmfuq[1:nz + 1].copy(),
            "pmful": P_pmful[1:nz + 1].copy(),
            "pdmfup": P_pdmfup[1:nz + 1].copy(),
            "plude": P_plude[1:nz + 1].copy(),
            "pmfude_rate": P_udr[1:nz + 1].copy(),
            "pmfd": P_pmfd[1:nz + 1].copy(),
            "pmfds": P_pmfds[1:nz + 1].copy(),
            "pmfdq": P_pmfdq[1:nz + 1].copy(),
            "pdmfdp": P_pdmfdp[1:nz + 1].copy(),
            "pmfdde_rate": P_ddr[1:nz + 1].copy()}


# ===========================================================================
# Slice 16: the updraft/downdraft momentum profiles -- cumastrn:927-995
# ===========================================================================
# What produces the puu/pvu/pud/pvd that cududvn consumes -- and the block
# that falsified this port's eighth wrong claim.  cuascn, cudlfsn and
# cuddrafn genuinely never write them; the chained conclusion "so cuinin
# sets them and nothing between touches them" was wrong, because THIS does.
# Measured: puu differs on 1,926 of 5,292 slots between cuinin's exit and
# cududvn's entry.
#
# momtrans = 2 (a parameter), so :943-955 -- the `if (momtrans == 1)` arm --
# is a THIRD dead block and the pressure-gradient `else` is the live one.
# Transcribed here as the live arm only, recorded in the docstring so the
# omission cannot read as a transcription slip.
#
# pgcoef = 0.7.  The pressure-gradient term is a centred difference in the
# environmental wind weighted by the mass flux at both interfaces.

_PGCOEF = _F(0.7)


def np_ntiedtke_momentum_profile(*, ldcum, ktype, kcbot, kctop, kdpl,
                                 idtop, puen, pven, pmfu, pmfd, puu, pvu,
                                 pud, pvd, pmfude_rate, pmfdde_rate,
                                 c=None):
    """cumastrn:927-995, one column.  Arrays 0-based in and out.

    ``puu``/``pvu``/``pud``/``pvd`` are read AND written: the updraft loop
    seeds from the environment at cloud base and integrates upward, and the
    downdraft loop seeds at ``idtop`` from the updraft's own value.
    """
    c = NtConstants() if c is None else c
    nz = puen.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_puen, P_pven = up(puen), up(pven)
    P_pmfu, P_pmfd = up(pmfu), up(pmfd)
    P_uu, P_vu = up(puu), up(pvu)
    P_ud, P_vd = up(pud), up(pvd)
    P_udr, P_ddr = up(pmfude_rate), up(pmfdde_rate)

    ld = bool(ldcum)
    kt, kb, ktop = int(ktype), int(kcbot), int(kctop)
    kdp, idt = int(kdpl), int(idtop)

    if not ld:
        return {"puu": P_uu[1:nz + 1].copy(), "pvu": P_vu[1:nz + 1].copy(),
                "pud": P_ud[1:nz + 1].copy(), "pvd": P_vd[1:nz + 1].copy()}

    # ---- the updraft profile (:930-971), DOWNWARD in index -------------
    for jk in range(klev - 1, 1, -1):
        ik = jk + 1
        if jk == kb and kt < 3:
            ikb = kdp
            P_uu[jk] = P_puen[ikb - 1]
            P_vu[jk] = P_pven[ikb - 1]
        elif jk == kb and kt == 3:
            P_uu[jk] = P_puen[jk - 1]
            P_vu[jk] = P_pven[jk - 1]
        if jk < kb and jk >= ktop:
            # momtrans == 2: the pressure-gradient arm.
            pgf_u = _F(-_F(_F(_PGCOEF * _F(0.5)) * _F(
                _F(P_pmfu[ik] * _F(P_puen[ik] - P_puen[jk]))
                + _F(P_pmfu[jk] * _F(P_puen[jk] - P_puen[jk - 1])))))
            pgf_v = _F(-_F(_F(_PGCOEF * _F(0.5)) * _F(
                _F(P_pmfu[ik] * _F(P_pven[ik] - P_pven[jk]))
                + _F(P_pmfu[jk] * _F(P_pven[jk] - P_pven[jk - 1])))))
            zerate = _F(_F(P_pmfu[jk] - P_pmfu[ik]) + P_udr[jk])
            zderate = P_udr[jk]
            zmfa = _F(_F(1.0) / _F(max(_CMFCMIN, P_pmfu[jk])))
            P_uu[jk] = _F(_F(_F(_F(_F(P_uu[ik] * P_pmfu[ik])
                                   + _F(zerate * P_puen[jk]))
                                - _F(zderate * P_uu[ik])) + pgf_u) * zmfa)
            P_vu[jk] = _F(_F(_F(_F(_F(P_vu[ik] * P_pmfu[ik])
                                   + _F(zerate * P_pven[jk]))
                                - _F(zderate * P_vu[ik])) + pgf_v) * zmfa)

    # ---- the downdraft profile (:972-991), UPWARD in index -------------
    for jk in range(3, klev + 1):
        ik = jk - 1
        if jk == idt:
            P_ud[jk] = _F(_F(0.5) * _F(P_uu[jk] + P_puen[ik]))
            P_vd[jk] = _F(_F(0.5) * _F(P_vu[jk] + P_pven[ik]))
        elif jk > idt:
            zerate = _F(_F(_F(-P_pmfd[jk]) + P_pmfd[ik]) + P_ddr[jk])
            zmfa = _F(_F(1.0) / _F(min(_F(-_CMFCMIN), P_pmfd[jk])))
            P_ud[jk] = _F(_F(_F(_F(P_ud[ik] * P_pmfd[ik])
                                - _F(zerate * P_puen[ik]))
                             + _F(P_ddr[jk] * P_ud[ik])) * zmfa)
            P_vd[jk] = _F(_F(_F(_F(P_vd[ik] * P_pmfd[ik])
                                - _F(zerate * P_pven[ik]))
                             + _F(P_ddr[jk] * P_vd[ik])) * zmfa)

    return {"puu": P_uu[1:nz + 1].copy(), "pvu": P_vu[1:nz + 1].copy(),
            "pud": P_ud[1:nz + 1].copy(), "pvd": P_vd[1:nz + 1].copy()}


# ===========================================================================
# Slice 17: the kinetic-energy dissipation -- cumastrn:1030-1056
# ===========================================================================
# THE LAST ARITHMETIC IN cumastrn, and the only place the momentum tendency
# feeds BACK into the heat tendency.  cududvn has just changed pvom/pvol;
# this measures how much kinetic energy that change removed from the
# resolved flow and returns it as sensible heat.
#
# ztenu/ztenv are copies of pvom/pvol taken BEFORE cududvn (:1021-1022), so
# `pvom - ztenu` is exactly cududvn's increment.  They are cududvn's own
# captured input and output -- which is precisely the "a neighbour's capture
# will do" shape, so the fixture records them at this block's own boundary
# instead.
#
# zsum22 and zsum12 are COLUMN INTEGRALS: the loop that fills them must
# finish before the loop that divides by them starts, so unlike most of this
# port the two passes cannot be fused.


def np_ntiedtke_ke_dissipation(*, ldcum, kctop, paph, puen, pven,
                               ztenu, ztenv, pvom, pvol, ptte, c=None):
    """cumastrn:1030-1056, one column.  Arrays 0-based in and out.

    ``ptte`` is ACCUMULATED into and returned -- the same add-not-assign
    contract as cudtdqn, and for the same reason (section 17).
    """
    c = NtConstants() if c is None else c
    nz = ptte.shape[0]
    klev = nz

    def up(a):
        o = _f1(nz)
        o[1:nz + 1] = _np32(a)
        return o

    P_puen, P_pven = up(puen), up(pven)
    P_tu, P_tv = up(ztenu), up(ztenv)
    P_vom, P_vol = up(pvom), up(pvol)
    P_ptte = up(ptte)
    P_paph = _f1(nz)
    P_paph[1:nz + 2] = _np32(paph)[:nz + 1]

    ld = bool(ldcum)
    ktop = int(kctop)
    zuv2 = _f1(nz)
    zsum12 = _F(0.0)
    zsum22 = _F(0.0)

    # ---- the integrals (:1034-1048) ------------------------------------
    for jk in range(1, klev + 1):
        zuv2[jk] = _F(0.0)
        if ld and jk >= ktop - 1:
            zdz = _F(P_paph[jk + 1] - P_paph[jk])
            zduten = _F(P_vom[jk] - P_tu[jk])
            zdvten = _F(P_vol[jk] - P_tv[jk])
            zuv2[jk] = _F(np.sqrt(_F(_F(zduten * zduten)
                                     + _F(zdvten * zdvten))))
            zsum22 = _F(zsum22 + _F(zuv2[jk] * zdz))
            zsum12 = _F(zsum12 - _F(_F(_F(P_puen[jk] * zduten)
                                       + _F(P_pven[jk] * zdvten)) * zdz))

    # ---- the heating (:1049-1056).  ADD, do not assign. ----------------
    for jk in range(1, klev + 1):
        if ld and jk >= ktop - 1:
            ztdis = _F(_F(_F(c.rcpd * zsum12) * zuv2[jk])
                       / _F(max(_F(1.0e-15), zsum22)))
            P_ptte[jk] = _F(P_ptte[jk] + ztdis)

    return {"ptte": P_ptte[1:nz + 1].copy(),
            "zsum12": zsum12, "zsum22": zsum22}


# ===========================================================================
# cu_ntiedtke_post_run -- module_cu_ntiedtke.F:502-527
# ===========================================================================
# THE EIGHT FIELDS nt-levels.csv IS GRADED ON, and until this existed they
# traced to NOTHING: cu_ntiedtke_run produces pt/pqv/pqc/pqi/pu/pv and
# zprecc, and nothing turned those into rthcuten/rucuten/raincv/pratec.  The
# gap was visible in the oracle's own header the whole time and was found by
# a gate driven off that header (test_ntiedtke_output_provenance.py).
#
# TWO VERTICAL CONVENTIONS IN ONE STATEMENT, and this is the only routine in
# the port where that is true.  `exner/qv/qc/qi/t/u/v` are the driver's
# untouched WRF-order inputs -- k = 0 is the SURFACE -- and they are the
# reference state the tendency is measured against.  `tf/qvf/qcf/qif/uf/vf`
# carry cu_ntiedtke_run's answer in SCHEME order, k = 0 the model TOP.  The
# routine pairs them by flipping, `tf(i,zz)` with `zz = kte - pp`, and the
# fixture records each array at its OWN index so the flip stays the port's
# job rather than being pre-applied where an error could hide.
#
# ASSIGNED, NOT ACCUMULATED, and unconditionally: unlike cudtdqn and the KE
# dissipation there is no `if` in the loop and no add.  Every level of every
# column is written, so nothing here carries a caller value -- which is why
# post_run has no class-2 rows in the aliasing audit despite six intent(inout)
# arrays.  Measured by reading :514-524, not assumed from the intent.
#
# THE ASSOCIATION IS LOAD-BEARING.  `(tf - t)/exner*rdelt` is left to right:
# subtract, DIVIDE, then multiply.  Folding it as `(tf - t) * (rdelt/exner)`
# is algebraically identical and bitwise different.


def np_ntiedtke_post_run(*, stepcu, dt, exner, qv, qc, qi, t, u, v,
                         qvf, qcf, qif, tf, uf, vf, rn):
    """``cu_ntiedtke_post_run``, one column.  Arrays 0-based in and out.

    ``exner/qv/qc/qi/t/u/v`` are WRF order, ``qvf/qcf/qif/tf/uf/vf`` scheme
    order, and the returned tendencies are WRF order -- the same convention
    the driver hands back to the model.
    """
    nz = int(np.asarray(t).shape[0])

    # delt = dt*stepcu with stepcu an INTEGER, so Fortran promotes it; the
    # reciprocal is a separate rounding and both are kept.
    delt = _F(_F(dt) * _F(stepcu))
    rdelt = _F(_F(1.0) / delt)

    T = _np32(t)
    QV, QC, QI = _np32(qv), _np32(qc), _np32(qi)
    U, V, EX = _np32(u), _np32(v), _np32(exner)
    TF, UF, VF = _np32(tf), _np32(uf), _np32(vf)
    QVF, QCF, QIF = _np32(qvf), _np32(qcf), _np32(qif)

    out = {name: np.zeros(nz, dtype=np.float32) for name in
           ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rucuten", "rvcuten")}

    # :505-508.  stepcu is an integer in both, and `stepcu*dt` in pratec is
    # ONE product that is then divided into -- not rn/stepcu/dt.
    raincv = _F(_F(rn) / _F(stepcu))
    pratec = _F(_F(rn) / _F(_F(stepcu) * _F(dt)))

    # :510-524.  k is the WRF index; zz walks the scheme column downward.
    for k in range(nz):
        zz = nz - 1 - k
        out["rthcuten"][k] = _F(_F(_F(TF[zz] - T[k]) / EX[k]) * rdelt)
        out["rqvcuten"][k] = _F(_F(QVF[zz] - QV[k]) * rdelt)
        out["rqccuten"][k] = _F(_F(QCF[zz] - QC[k]) * rdelt)
        out["rqicuten"][k] = _F(_F(QIF[zz] - QI[k]) * rdelt)
        out["rucuten"][k] = _F(_F(UF[zz] - U[k]) * rdelt)
        out["rvcuten"][k] = _F(_F(VF[zz] - V[k]) * rdelt)

    out["raincv"] = raincv
    out["pratec"] = pratec
    return out


# ===========================================================================
# cu_ntiedtke_run's post-conversion -- cu_ntiedtke.F90:278-320
# ===========================================================================
# THE MISSING LINK.  cumastrn leaves tendencies (ptte, pqte, pvom, pvol) and
# a detrained condensate rate (pcte); cu_ntiedtke_post_run differences the
# updated STATE against the reference state.  This block is what turns one
# into the other, and without it the chain from the last cumastrn stage to
# the eight graded fields of nt-levels.csv has a hole in it.
#
# It is inside cu_ntiedtke_run, so -- unlike pre_run and post_run -- it
# cannot be reached by objcopy and must be transcribed.  That transcription
# is convergence-proved inside run_nt_cumastrn.F90 (docs/ntiedtke/PORT-RECORD.md §29's
# one remaining convergence argument); this mirror is graded against the
# capture taken at the block's own boundary, which is strictly stronger.
#
# EVERYTHING HERE IS SCHEME ORDER, k = 0 the model top.  The flip back to
# WRF order happens in post_run, not here.
#
# THE CONDENSATE ARM IS CONDITIONAL.  `if (pcte > 0.)` -- and on the false
# arm pqc/pqi keep the values they arrived with, so this is a class-2 shape:
# a port that zeroed them at entry would diverge on every level that does
# not detrain, which is most of them.
#
# zqp1 IS UPDATED IN PLACE and then read: pqv = zqp1/(1 - zqp1) uses the
# NEW value.  A capture taken after the block would record the answer, so
# the fixture records zqp1 before it.


def np_ntiedtke_post_conversion(*, delt, pcte, ztp1, ptte, ztt, pqte, zqq,
                                zqp1, qcf, qif, uf, vf, pvom, pvol,
                                prsfc, pssfc, c=None):
    """``cu_ntiedtke.F90:278-320``, one column.  Arrays 0-based in and out."""
    c = NtConstants() if c is None else c
    nz = int(np.asarray(ztp1).shape[0])
    dt = _F(delt)

    PCTE, T1 = _np32(pcte), _np32(ztp1)
    PTTE, ZTT = _np32(ptte), _np32(ztt)
    PQTE, ZQQ = _np32(pqte), _np32(zqq)
    Q1 = _np32(zqp1).copy()
    QCF, QIF = _np32(qcf), _np32(qif)
    UF, VF = _np32(uf), _np32(vf)
    VOM, VOL = _np32(pvom), _np32(pvol)

    out = {n: np.zeros(nz, dtype=np.float32) for n in
           ("pqc", "pqi", "pt", "pqv", "pu", "pv")}

    # :296-305.  fliq/fice split the detrained condensate by temperature.
    for k in range(nz):
        if PCTE[k] > _F(0.0):
            fliq = nt_foealfa(T1[k])
            fice = _F(_F(1.0) - fliq)
            out["pqc"][k] = _F(QCF[k] + _F(_F(fliq * PCTE[k]) * dt))
            out["pqi"][k] = _F(QIF[k] + _F(_F(fice * PCTE[k]) * dt))
        else:
            out["pqc"][k] = QCF[k]
            out["pqi"][k] = QIF[k]

    # :308-314.  zqp1 is updated IN PLACE and the new value feeds pqv.
    for k in range(nz):
        out["pt"][k] = _F(T1[k] + _F(_F(PTTE[k] - ZTT[k]) * dt))
        Q1[k] = _F(Q1[k] + _F(_F(PQTE[k] - ZQQ[k]) * dt))
        out["pqv"][k] = _F(Q1[k] / _F(_F(1.0) - Q1[k]))

    # :316-318.  amax1 clamps a negative surface flux product to zero.
    zprecc = _F(max(_F(0.0), _F(_F(_F(prsfc) + _F(pssfc)) * dt)))

    # :319-325, guarded by lmfdudv -- a PARAMETER, .true. at :55, so the
    # branch is not a runtime choice and the port does not carry one.
    for k in range(nz):
        out["pu"][k] = _F(UF[k] + _F(VOM[k] * dt))
        out["pv"][k] = _F(VF[k] + _F(VOL[k] * dt))

    out["zprecc"] = zprecc
    out["zqp1"] = Q1
    return out
