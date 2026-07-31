"""CPU reference for the WRF v4.6.1 MYNN surface layer.

The implementation is a direct, single-precision translation of
``SFCLAY1D_mynn`` for the admitted option identities: every defined
``isftcflx`` over water, with ``iz0tlnd=0``, ``spp_pbl=0`` and ``psi_opt=0``.
It is a numerical reference for oracle/CUDA work; importing it does not admit
the MYNN surface-layer runtime selector.

Over-water branch selection, read from ``module_sf_mynn.F:631-710`` rather
than from the header comment at :35-41, which does not mention that
``ISFTCFLX=4`` exists::

    z0     (:631-662)   0 -> charnock_1955          COARE 3.0
                        1 -> davis_etal_2008
                        2 -> davis_etal_2008
                        3 -> Taylor_Yelland_2001
                        4 -> charnock_1955          COARE 3.0
    zt,zq  (:680-710)   0 -> fairall_etal_2003      COARE 3.0
                        1 -> fairall_etal_2003      COARE 3.0
                        2 -> garratt_1992, water arm
                        3 -> fairall_etal_2003      COARE 3.0

``ISFTCFLX=4`` sets a roughness and then falls off the end of the zt/zq chain
without assigning ``z_t``/``z_q``, which are undecorated local automatics
(:474).  WRF does not define that identity, so this reference rejects it
instead of inventing a value; :func:`mynn_surface_layer_default` admits
``isftcflx`` 0, 1, 2 and 3 only.

``COARE_OPT`` is a ``REAL, PARAMETER`` fixed at 3.0 (:85), so the COARE 3.5
half of the 0/1/3 arms -- ``edson_etal_2013`` for z0 and ``fairall_etal_2014``
for zt/zq -- is unreachable through ``SFCLAY1D_mynn``.  Both are transcribed
here as :func:`_edson_etal_2013` and :func:`_fairall_etal_2014` and are pinned
against the unmodified module at their own entry points by the leaf oracle
(``gpuwm/data/mynn/oracle/surface-layer-water-leaf.csv``), but nothing in the
column solver may call them: WRF has no runtime switch that reaches them, and
adding one here would be a gpuwm invention rather than a port.

Every water roughness leaf routes its transcendentals through
``gpuwm.core.noahmp_libm``, so its answer is the glibc 2.39 answer gfortran
linked against and does not depend on the host NumPy.  **So do ``psi_init``'s
four lookup tables** (:func:`psi_tables`), which used to be built at import
time with scalar ``np.arctan``/``np.log``/``**`` on ``np.float32`` and were
therefore the MSVC ``powf`` answer on Windows and NumPy's own ``log`` answer
everywhere: 364 of their 4004 words were not the words gfortran's oracle
carries, worst 32 ULP.  Routing them cut the reference's measured distance
from the WRF v4.6.1 oracle over 9678 compared elements from 3343 to 1469 ULP
on Windows and from 2207 to 1463 on Ubuntu, and collapsed the three-build
disagreement from 243 elements to 6.

The land and snow leaves still call NumPy and still carry the measured
three-platform ULP union in ``tests/test_mynn_surface.py``; they are a
separate lane's identity (``iz0tlnd``) and are deliberately left alone here,
as are ``_li_etal_2010`` and the ``**``/``np.exp`` sites in the column solver.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from functools import lru_cache

import numpy as np

from gpuwm.core.noahmp_libm import atanf as _atanf
from gpuwm.core.noahmp_libm import expf as _expf
from gpuwm.core.noahmp_libm import logf as _logf
from gpuwm.core.noahmp_libm import powf as _powf
from gpuwm.core.noahmp_libm import sqrtf as _sqrtf


F = np.float32
CP = F(1004.5)
G = F(9.81)
R = F(287.0)
RV = F(461.6)
ROVCP = F(R / CP)
XLV = F(2.5e6)
SVP1 = F(0.6112)
SVP2 = F(17.67)
SVP3 = F(29.65)
SVPT0 = F(273.15)
EP1 = F(RV / R - F(1.0))
EP2 = F(R / RV)
EP3 = F(1.0) - EP2
KARMAN = F(0.4)
PRT = F(1.0)
WMIN = F(0.1)
VCONVC = F(1.25)

_INPUT_NAMES = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)


#: ``SQRT(3.)`` and ``ATAN(1.)``, the two loop-invariant constants WRF
#: recomputes inside ``psi_init``'s table loops (module_sf_mynn.F:213-262).
#: ``sqrtf`` is correctly rounded, so hoisting them changes nothing.
_RT3 = _sqrtf(F(3.0))
_ATAN1 = _atanf(F(1.0))


def _psim_stable_full(zolf):
    zolf = F(zolf)
    return F(-F(6.1) * _logf(
        zolf + _powf(F(1.0) + _powf(zolf, F(2.5)), F(1.0 / 2.5))
    ))


def _psih_stable_full(zolf):
    zolf = F(zolf)
    return F(-F(5.3) * _logf(
        zolf + _powf(F(1.0) + _powf(zolf, F(1.1)), F(1.0 / 1.1))
    ))


def _psim_unstable_full(zolf):
    zolf = F(zolf)
    x = _powf(F(1.0) - F(16.0) * zolf, F(0.25))
    psimk = F(
        F(2.0) * _logf(F(0.5) * (F(1.0) + x))
        + _logf(F(0.5) * (F(1.0) + x * x))
        - F(2.0) * _atanf(x)
        + F(2.0) * _ATAN1
    )
    ym = _powf(F(1.0) - F(10.0) * zolf, F(0.33))
    psimc = F(
        F(1.5) * _logf((ym * ym + ym + F(1.0)) / F(3.0))
        - _RT3 * _atanf((F(2.0) * ym + F(1.0)) / _RT3)
        + F(4.0) * _ATAN1 / _RT3
    )
    return F((psimk + zolf * zolf * psimc) / (F(1.0) + zolf * zolf))


def _psih_unstable_full(zolf):
    zolf = F(zolf)
    # WRF writes ``**.5`` here, not ``SQRT``, and gfortran at the oracle's
    # flags emits a ``powf`` call for a REAL(4) constant exponent.  Measured:
    # glibc ``powf(x, 0.5f)`` and ``sqrtf(x)`` agree on all 1001 table
    # arguments, so this choice is invisible in the table and only reaches
    # the |zolf| >= 10 fallback below.
    y = _powf(F(1.0) - F(16.0) * zolf, F(0.5))
    psihk = F(F(2.0) * _logf((F(1.0) + y) / F(2.0)))
    yh = _powf(F(1.0) - F(34.0) * zolf, F(0.33))
    psihc = F(
        F(1.5) * _logf((yh * yh + yh + F(1.0)) / F(3.0))
        - _RT3 * _atanf((F(2.0) * yh + F(1.0)) / _RT3)
        + F(4.0) * _ATAN1 / _RT3
    )
    return F((psihk + zolf * zolf * psihc) / (F(1.0) + zolf * zolf))


#: SHA-256 of the four 1001-entry FP32 lookup tables, little-endian, in
#: ``_PSIM_STAB``, ``_PSIH_STAB``, ``_PSIM_UNSTAB``, ``_PSIH_UNSTAB`` order.
#: Every word is a pure function of this repository's own glibc 2.39
#: transcriptions (:mod:`gpuwm.core.noahmp_libm`), which are integer bit
#: arithmetic, so the digest is host- and NumPy-independent by construction.
#: ``tests/test_mynn_surface.py`` pins it and shows the NumPy-built words
#: it replaced failing against it.
PSI_TABLE_SHA256 = (
    "35f08242c537456f09fae5950e9cd1518469c0fa55e41c9fa4b79f99368b8898"
)


@lru_cache(maxsize=1)
def psi_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """WRF ``psi_init``'s four tables (module_sf_mynn.F:213-262).

    Built on first use rather than at import: the coupled runtime imports
    this module only for :data:`ISFTCFLX_DEFINED`, and an import-time table
    is a side effect that no test can regenerate or diff.  The words are
    pinned by :data:`PSI_TABLE_SHA256`.
    """
    build = (
        (_psim_stable_full, F(1.0)),
        (_psih_stable_full, F(1.0)),
        (_psim_unstable_full, F(-1.0)),
        (_psih_unstable_full, F(-1.0)),
    )
    return tuple(
        np.asarray([full(sign * F(n) * F(0.01)) for n in range(1001)],
                   dtype=np.float32)
        for full, sign in build
    )


def psi_table_digest() -> str:
    """SHA-256 of the four tables as shipped, for the pinning gate."""
    return hashlib.sha256(b"".join(
        np.ascontiguousarray(table).tobytes() for table in psi_tables()
    )).hexdigest()


def _table_value(zolf, table, full, *, unstable=False):
    zolf = F(zolf)
    scaled = F((-zolf if unstable else zolf) * F(100.0))
    index = int(scaled)
    fraction = F(scaled - F(index))
    if index + 1 <= 1000:
        return F(table[index] + fraction * (table[index + 1] - table[index]))
    return F(full(zolf))


def _psim_stable(zolf):
    return _table_value(zolf, psi_tables()[0], _psim_stable_full)


def _psih_stable(zolf):
    return _table_value(zolf, psi_tables()[1], _psih_stable_full)


def _psim_unstable(zolf):
    return _table_value(
        zolf, psi_tables()[2], _psim_unstable_full, unstable=True
    )


def _psih_unstable(zolf):
    return _table_value(
        zolf, psi_tables()[3], _psih_unstable_full, unstable=True
    )


def _li_etal_2010(rib, zaz0, z0zt):
    rib, zaz0, z0zt = F(rib), F(zaz0), F(z0zt)
    zaz02 = min(max(zaz0, F(100.0)), F(100000.0))
    z0zt2 = min(max(z0zt, F(0.5)), F(100.0))
    alfa = F(np.log(zaz02))
    beta = F(np.log(z0zt2))
    if rib <= F(0.0):
        zl = F(
            F(0.045) * alfa * rib ** F(2.0)
            + (
                (F(0.003) * beta + F(0.0059)) * alfa ** F(2.0)
                + (F(-0.0828) * beta + F(0.8845)) * alfa
                + (F(0.1739) * beta ** F(2.0)
                   + F(-0.9213) * beta + F(-0.1057))
            ) * rib
        )
        return F(min(max(zl, F(-15.0)), F(0.0)))
    if rib <= F(0.2):
        zl = F(
            ((F(0.5738) * beta + F(-0.4399)) * alfa
             + (F(-4.901) * beta + F(52.50))) * rib ** F(2.0)
            + ((F(-0.0539) * beta + F(1.540)) * alfa
               + (F(-0.669) * beta + F(-3.282))) * rib
        )
        return F(min(max(zl, F(0.0)), F(4.0)))
    zl = F(
        (F(0.7529) * alfa + F(14.94)) * rib
        + F(0.1569) * alfa + F(-0.3091) * beta + F(-1.303)
    )
    return F(min(max(zl, F(1.0)), F(20.0)))


def _zolrib(ri, za, z0, zt, logz0, logzt, zol1):
    ri, za, z0, zt = F(ri), F(za), F(z0), F(zt)
    logz0, logzt, zol1 = F(logz0), F(logzt), F(zol1)
    if zol1 * ri < F(0.0):
        zol1 = F(0.0)
    if ri < F(0.0):
        zolold, result = F(-99999.0), F(-66666.0)
    else:
        zolold, result = F(99999.0), F(66666.0)
    iteration = 1
    while abs(F(zolold - result)) > F(0.01) and iteration < 20:
        zolold = zol1 if iteration == 1 else result
        zol20 = F(zolold * z0 / za)
        zol3 = F(zolold + zol20)
        zolt = F(zolold * zt / za)
        if ri < F(0.0):
            psit2 = max(F(logzt - (
                _psih_unstable(zol3) - _psih_unstable(zolt)
            )), F(1.0))
            psix2 = max(F(logz0 - (
                _psim_unstable(zol3) - _psim_unstable(zol20)
            )), F(1.0))
        else:
            psit2 = max(F(logzt - (
                _psih_stable(zol3) - _psih_stable(zolt)
            )), F(1.0))
            psix2 = max(F(logz0 - (
                _psim_stable(zol3) - _psim_stable(zol20)
            )), F(1.0))
        result = F(ri * psix2 ** F(2.0) / psit2)
        iteration += 1
    if iteration == 20 and abs(F(zolold - result)) > F(0.01):
        result = _li_etal_2010(ri, za / z0, z0 / zt)
    return F(result)


def _zilitinkevich_land(z0, restar):
    zt = F(z0 * np.exp(-KARMAN * F(0.085) * np.sqrt(restar)))
    zt = min(zt, F(0.75) * z0)
    return F(zt), F(zt)


#: ``LOG(10./1e-4)`` in ``charnock_1955`` (:1351) and ``LOG(10/1e-4)`` in
#: ``edson_etal_2013`` (:1379).  Fortran converts the integer 10 before
#: dividing, so both are ``LOG`` of the same REAL(4) 1e5, and gfortran folds
#: them at compile time.  The folded word, glibc's ``logf`` and NumPy's all
#: agree here -- measured, not assumed, by the leaf oracle at ``max_ulp 0``.
_LOG_10_OVER_Z0REF = _logf(F(10.0) / F(1.0e-4))

#: The over-water roughness identities ``SFCLAY1D_mynn`` defines.  4 sets a z0
#: and then leaves ``z_t``/``z_q`` unassigned (:680-702 has no arm for it), and
#: anything above 4 leaves ZNT unassigned as well; neither is a behaviour to
#: port.
ISFTCFLX_DEFINED = (0, 1, 2, 3)


def _charnock_1955(ustar, wsp, visc, zu):
    """``module_sf_mynn.F:1337`` -- COARE 3.0 varying-Charnock z0."""

    wsp10 = F(wsp * _LOG_10_OVER_Z0REF / _logf(zu / F(1.0e-4)))
    czc = F(F(0.011) + F(0.007) * min(
        max(F((wsp10 - F(10.0)) / F(8.0)), F(0.0)), F(1.0)
    ))
    z0 = F(czc * ustar * ustar / G
           + F(0.11) * visc / max(ustar, F(0.05)))
    return F(min(max(z0, F(1.27e-7)), F(2.85e-3)))


def _edson_etal_2013(ustar, wsp, visc, zu):
    """``module_sf_mynn.F:1362`` -- COARE 3.5 varying-Charnock z0.

    Dead under the compiled-in ``COARE_OPT=3.0``; pinned by the leaf oracle
    and never called from the column solver.  Note the ``MAX(ustar,0.07)``
    floor, which is not the 0.05 of :func:`_charnock_1955`, and the ``m``
    corrected by the Edson et al. (2014) corrigendum.
    """

    wsp10 = F(wsp * _LOG_10_OVER_Z0REF / _logf(zu / F(1.0e-4)))
    wsp10 = min(F(19.0), wsp10)
    czc = max(F(F(0.0017) * wsp10 + F(-0.005)), F(0.0))
    z0 = F(czc * ustar * ustar / G
           + F(0.11) * visc / max(ustar, F(0.07)))
    return F(min(max(z0, F(1.27e-7)), F(2.85e-3)))


def _davis_etal_2008(ustar):
    """``module_sf_mynn.F:1281`` -- Donelan/Davis z0, a function of u* alone.

    ``ZW`` blends the low-wind Charnock limb ``ZN1`` into the Donelan limb
    ``ZN2`` and saturates at u* = 1.06.  Note ``OZO = 1.59e-5`` keeps ``ZN1``
    -- and so the blend whenever ``ZW < 1`` -- above the 1.27e-7 floor, and
    ``ZN2`` at ``ZW == 1`` is already 5.3e-3 at u* = 2, so the lower clamp is
    unreachable from this leaf and is not claimed as covered.
    """

    zw = min(_powf(ustar / F(1.06), F(0.3)), F(1.0))
    zn1 = F(F(0.011) * ustar * ustar / G + F(1.59e-5))
    zn2 = F(F(10.0) * _expf(F(-9.5) * _powf(ustar, F(-0.3333)))
            + F(0.11) * F(1.5e-5) / max(ustar, F(0.01)))
    z0 = F((F(1.0) - zw) * zn1 + zw * zn2)
    return F(min(max(z0, F(1.27e-7)), F(2.85e-3)))


def _taylor_yelland_2001(wsp):
    """``module_sf_mynn.F:1311`` -- wave-steepness z0.

    ``ustar`` is an argument of the Fortran subroutine and is never read in
    its body, so this transcription does not take it.  ``hs`` and ``Lp`` are
    both proportional to ``wsp10**2`` above the 0.1 floor, so the steepness
    ``hs/Lp`` is constant there and z0 grows as ``wsp10**2``: it reaches the
    2.85e-3 ceiling near 26 m/s and sits under the 1.27e-7 floor below about
    0.18 m/s.  Both clamps are live.
    """

    hs = F(F(0.0248) * _powf(wsp, F(2.0)))
    tp = F(F(0.729) * max(wsp, F(0.1)))
    lp = F(G * F(tp * tp) / F(F(2.0) * F(3.14159265)))
    z0 = F(F(1200.0) * hs * _powf(hs / lp, F(4.5)))
    return F(min(max(z0, F(1.27e-7)), F(2.85e-3)))


def _fairall_etal_2003(restar):
    """``module_sf_mynn.F:1425`` -- COARE 3.0 zt/zq.

    The ``Ren <= 2`` test at :1442 selects between two arms that compute the
    identical expression, so it cannot change an answer; it is transcribed as
    the single expression it evaluates to.  ``Zq`` is assigned from the
    already-clamped ``Zt`` at :1466-1467, not from itself, so zq == zt always.
    """

    zt = F(F(5.5e-5) * _powf(restar, F(-0.60)))
    zt = F(min(max(zt, F(2.0e-9)), F(1.0e-4)))
    return zt, zt


def _fairall_etal_2014(restar):
    """``module_sf_mynn.F:1473`` -- COARE 3.5/4.0 zt/zq.

    Dead under the compiled-in ``COARE_OPT=3.0``; pinned by the leaf oracle
    and never called from the column solver.  ``Zq`` is again assigned from
    ``Zt`` (:1495), so zq == zt.
    """

    zt = min(F(1.6e-4), F(F(5.8e-5) / _powf(restar, F(0.72))))
    zt = max(zt, F(2.0e-9))
    return zt, zt


#: ``e**2.`` from ``garratt_1992``'s land arm (:1417).  Both operands are
#: constants, so gfortran folds it; the leaf oracle pins that the folded word
#: is the one this reproduces.
_GARRATT_E_SQUARED = _powf(F(2.71828183), F(2.0))


def _garratt_1992(z0, restar, landsea):
    """``module_sf_mynn.F:1392`` -- garratt_1992, both arms.

    The ISFTCFLX=2 caller (:695) passes ``XLAND`` straight through and this
    leaf makes its own ``landsea-1.5 .GT. 0`` test, which is NOT the same
    predicate as the ``.GE. 0`` at :625 that sent the column here.  A column
    at exactly XLAND=1.5 therefore takes the water roughness selection and
    then the LAND arm of this leaf -- so the land arm is reachable from this
    lane and is transcribed.  It is also what ``iz0tlnd=3`` would use, but
    that call site (:743) is a different identity and stays unported.

    Unlike :func:`_fairall_etal_2003`, zt and zq are independent in the water
    arm: the exponents are 2.48 and 2.28.
    """

    if F(landsea - F(1.5)) > F(0.0):
        quarter = _powf(restar, F(0.25))
        zt = F(z0 * _expf(F(F(2.0) - F(2.48) * quarter)))
        zq = F(z0 * _expf(F(F(2.0) - F(2.28) * quarter)))
        zq = F(max(min(zq, F(5.5e-5)), F(2.0e-9)))
        zt = F(max(min(zt, F(5.5e-5)), F(2.0e-9)))
        return zt, zq
    zq = F(z0 / _GARRATT_E_SQUARED)
    return zq, zq


def _water_roughness(isftcflx, ustar, wsp, visc, za, xland):
    """``module_sf_mynn.F:631-710`` -- the over-water ISFTCFLX selection.

    Returns ``(z0, restar, zt, zq)``.  ``restar`` is recomputed from the NEW
    z0 (:675), which is why the two selections cannot be collapsed into one.
    ``xland`` is only read by the ISFTCFLX=2 arm, which forwards it into
    :func:`_garratt_1992`'s own land/water test.
    """

    if isftcflx == 0:
        z0 = _charnock_1955(ustar, wsp, visc, za)
    elif isftcflx in (1, 2):
        z0 = _davis_etal_2008(ustar)
    elif isftcflx == 3:
        z0 = _taylor_yelland_2001(wsp)
    else:  # pragma: no cover - rejected by mynn_surface_layer_default
        raise ValueError(f"undefined isftcflx: {isftcflx}")
    restar = max(F(ustar * z0 / visc), F(0.1))
    if isftcflx == 2:
        zt, zq = _garratt_1992(z0, restar, xland)
    else:
        zt, zq = _fairall_etal_2003(restar)
    return z0, restar, zt, zq


def _andreas_snow(z0, visc, ustar):
    zntsno = F(
        F(0.135) * visc / ustar
        + F(0.035) * ustar * ustar / F(9.8)
        * (F(5.0) * np.exp(-((ustar - F(0.18)) / F(0.1)) ** F(2.0))
           + F(1.0))
    )
    ren = min(F(ustar * zntsno / visc), F(1000.0))
    log_ren = F(np.log(ren))
    if ren <= F(0.135):
        bt0, bt1, bt2 = F(1.25), F(0.0), F(0.0)
        bq0, bq1, bq2 = F(1.61), F(0.0), F(0.0)
    elif ren < F(2.5):
        bt0, bt1, bt2 = F(0.149), F(-0.55), F(0.0)
        bq0, bq1, bq2 = F(0.351), F(-0.628), F(0.0)
    else:
        bt0, bt1, bt2 = F(0.317), F(-0.565), F(-0.183)
        bq0, bq1, bq2 = F(0.396), F(-0.512), F(-0.180)
    zt = F(zntsno * np.exp(bt0 + bt1 * log_ren + bt2 * log_ren ** F(2.0)))
    zq = F(zntsno * np.exp(bq0 + bq1 * log_ren + bq2 * log_ren ** F(2.0)))
    return zt, zq


def _as_columns(values: Mapping[str, object]) -> tuple[dict[str, np.ndarray], int]:
    columns = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in _INPUT_NAMES
    }
    shapes = {value.shape for value in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 1:
        raise ValueError("MYNN surface inputs must be equal-length 1-D arrays")
    if any(not np.isfinite(value).all() for value in columns.values()):
        raise ValueError("MYNN surface inputs must be finite")
    if np.any(columns["znt"] <= 0.0) or np.any(columns["rho1"] <= 0.0):
        raise ValueError("MYNN znt and rho1 must be positive")
    return columns, next(iter(shapes))[0]


def mynn_surface_layer_default(
    values: Mapping[str, object],
    *,
    dx: float = 3000.0,
    itimestep: int = 1,
    isfflx: int = 1,
    isftcflx: int = 0,
    mol: object | None = None,
    ustm: object | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate the WRF MYNN surface layer for independent columns.

    ``isftcflx`` selects the over-water roughness identity and is ignored on
    land, exactly as ``module_sf_mynn.F:625`` is.  Only the four identities
    WRF defines are accepted; see the module docstring for why 4 is not one of
    them.
    """

    if not isinstance(itimestep, int) or itimestep < 1:
        raise ValueError("itimestep must be a positive integer")
    if isfflx not in (0, 1):
        raise ValueError("isfflx must be 0 or 1")
    if (not isinstance(isftcflx, (int, np.integer))
            or isinstance(isftcflx, bool)
            or isftcflx not in ISFTCFLX_DEFINED):
        raise ValueError(
            "isftcflx must be 0 (COARE 3.0 z0 and zt/zq), 1 (Davis z0, "
            "COARE 3.0 zt/zq), 2 (Davis z0, Garratt zt/zq) or 3 "
            "(Taylor-Yelland z0, COARE 3.0 zt/zq); module_sf_mynn.F:680-702 "
            "assigns no z_t/z_q for any other value, so WRF does not define "
            f"one, and gpuwm will not invent one.  Got {isftcflx}."
        )
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive and finite")
    source, count = _as_columns(values)
    initial_mol = np.zeros(count, dtype=np.float32) if mol is None else np.asarray(
        mol, dtype=np.float32
    )
    initial_ustm = source["ust"].copy() if ustm is None else np.asarray(
        ustm, dtype=np.float32
    )
    if initial_mol.shape != (count,) or initial_ustm.shape != (count,):
        raise ValueError("mol and ustm must match the input column count")

    output_names = (
        "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
        "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
        "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2",
        "gz1oz0", "wspd", "br", "ck", "cka", "cd", "cda", "wstar",
        "qstar", "cpm", "znt",
    )
    result = {name: np.zeros(count, dtype=np.float32) for name in output_names}
    dx32 = F(dx)

    for i in range(count):
        u1, v1 = F(source["u1"][i]), F(source["v1"][i])
        u2, v2 = F(source["u2"][i]), F(source["v2"][i])
        t1, qv1, p1 = F(source["t1"][i]), F(source["qv1"][i]), F(source["p1"][i])
        rho1, dz1, dz2 = F(source["rho1"][i]), F(source["dz1"][i]), F(source["dz2"][i])
        psfcpa, tsk = F(source["psfc"][i]), F(source["tsk"][i])
        pblh, mavail = F(source["pblh"][i]), F(source["mavail"][i])
        xland, snowh = F(source["xland"][i]), F(source["snowh"][i])
        hfx, qfx = F(source["hfx"][i]), F(source["qfx"][i])
        z0, qsfc, ust = F(source["znt"][i]), F(source["qsfc"][i]), F(source["ust"][i])
        ustm_i, mol_i = F(initial_ustm[i]), F(initial_mol[i])

        psfc = F(psfcpa / F(1000.0))
        thgb = F(tsk * (F(100.0) / psfc) ** ROVCP)
        pl = F(p1 / F(1000.0))
        th1 = F(t1 * (F(100.0) / pl) ** ROVCP)
        tc1 = F(t1 - F(273.15))
        qvsh = F(qv1 / (F(1.0) + qv1))
        tvcon = F(F(1.0) + EP1 * qvsh)
        thv1 = F(th1 * tvcon)
        za = F(F(0.5) * dz1)
        za2 = F(dz1 + F(0.5) * dz2)
        govrth = F(G / th1)

        if tsk < F(273.15):
            e1 = F(SVP1 * np.exp(
                F(4648.0) * (F(1.0) / F(273.15) - F(1.0) / tsk)
                - F(11.64) * np.log(F(273.15) / tsk)
                + F(0.02265) * (F(273.15) - tsk)
            ))
        else:
            e1 = F(SVP1 * np.exp(SVP2 * (tsk - SVPT0) / (tsk - SVP3)))
        # module_sf_mynn.F:532 tests the INCOMING QSFC: a land column whose
        # LSM never set QSFC is recomputed from saturation here.
        if xland > F(1.5) or qsfc <= F(0.0):
            qsfc = F(EP2 * e1 / (psfc - EP3 * e1))
            qsfcmr = F(EP2 * e1 / (psfc - e1))
        else:
            qsfcmr = F(qsfc / (F(1.0) - qsfc))

        if tsk < F(273.15):
            e1 = F(SVP1 * np.exp(
                F(4648.0) * (F(1.0) / F(273.15) - F(1.0) / t1)
                - F(11.64) * np.log(F(273.15) / t1)
                + F(0.02265) * (F(273.15) - t1)
            ))
        else:
            e1 = F(SVP1 * np.exp(SVP2 * (t1 - SVPT0) / (t1 - SVP3)))
        qgh = F(EP2 * e1 / (pl - e1))
        cpm = F(CP * (F(1.0) + F(0.84) * qv1))

        wsp = F(np.sqrt(u1 * u1 + v1 * v1))
        thvgb = F(thgb * (F(1.0) + EP1 * qsfc))
        dthvdz = F(thv1 - thvgb)
        fluxc = max(F(hfx / rho1 / CP + EP1 * thvgb * qfx / rho1), F(0.0))
        # module_sf_mynn.F:573 spells the same predicate a second time, but it
        # runs after the :533 saturation update, so a land column that entered
        # with QSFC<=0 now takes the LAND height scale.
        wstar_water = xland > F(1.5) or qsfc <= F(0.0)
        height = pblh if wstar_water else min(F(1.5) * pblh, F(4000.0))
        wstar = F(VCONVC * (G / tsk * height * fluxc) ** F(0.33))
        vsgd = F(F(0.32) * max(dx32 / F(5000.0) - F(1.0), F(0.0)) ** F(0.33))
        wsp = F(max(np.sqrt(wsp * wsp + wstar * wstar + vsgd * vsgd), WMIN))
        br = F(govrth * za * dthvdz / (wsp * wsp))
        limit = F(2.0 if itimestep == 1 else 4.0)
        br = F(min(max(br, -limit), limit))

        visc = F(F(1.326e-5) * (
            F(1.0) + F(6.542e-3) * tc1 + F(8.301e-6) * tc1 * tc1
            - F(4.84e-9) * tc1 * tc1 * tc1
        ))
        if xland >= F(1.5):
            z0, restar, zt, zq = _water_roughness(
                isftcflx, ust, wsp, visc, za, xland
            )
        else:
            restar = max(F(ust * z0 / visc), F(0.1))
            if snowh >= F(0.1):
                zt, zq = _andreas_snow(z0, visc, ust)
            else:
                zt, zq = _zilitinkevich_land(z0, restar)

        zratio = F(z0 / zt)
        gz1oz0 = F(np.log((za + z0) / z0))
        gz1ozt = F(np.log((za + z0) / zt))
        gz2ozt = F(np.log((F(2.0) + z0) / zt))
        gz10oz0 = F(np.log((F(10.0) + z0) / z0))
        gz10ozt = F(np.log((F(10.0) + z0) / zt))

        if br > F(0.0):
            regime = F(1.0 if br > F(0.2) else 2.0)
            if itimestep <= 1:
                zol = _li_etal_2010(br, za / z0, zratio)
            else:
                zol = F(za * KARMAN * G * mol_i
                        / (th1 * max(ust * ust, F(0.0001))))
                zol = F(min(max(zol, F(0.0)), F(20.0)))
            zol = F(min(max(_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                            F(0.0)), F(20.0)))
            zolzt, zolz0 = F(zol * zt / za), F(zol * z0 / za)
            zolza = F(zol * (za + z0) / za)
            zol10 = F(zol * (F(10.0) + z0) / za)
            zol2 = F(zol * (F(2.0) + z0) / za)
            psim = F(_psim_stable(zolza) - _psim_stable(zolz0))
            psih = F(_psih_stable(zolza) - _psih_stable(zolzt))
            psim10 = F(_psim_stable(zol10) - _psim_stable(zolz0))
            psih10 = F(_psih_stable(zol10) - _psih_stable(zolz0))
            psih2 = F(_psih_stable(zol2) - _psih_stable(zolz0))
        elif br == F(0.0):
            regime, zol = F(3.0), F(0.0)
            psim = psih = psim10 = psih10 = psih2 = F(0.0)
        else:
            regime = F(4.0)
            if itimestep <= 1:
                zol = _li_etal_2010(br, za / z0, zratio)
            else:
                zol = F(za * KARMAN * G * mol_i
                        / (th1 * max(ust * ust, F(0.001))))
                zol = F(min(max(zol, F(-20.0)), F(0.0)))
            zol = F(min(max(_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                            F(-20.0)), F(0.0)))
            zolzt, zolz0 = F(zol * zt / za), F(zol * z0 / za)
            zolza = F(zol * (za + z0) / za)
            zol10 = F(zol * (F(10.0) + z0) / za)
            zol2 = F(zol * (F(2.0) + z0) / za)
            psim = F(_psim_unstable(zolza) - _psim_unstable(zolz0))
            psih = F(_psih_unstable(zolza) - _psih_unstable(zolzt))
            psim10 = F(_psim_unstable(zol10) - _psim_unstable(zolz0))
            psih10 = F(_psih_unstable(zol10) - _psih_unstable(zolz0))
            psih2 = F(_psih_unstable(zol2) - _psih_unstable(zolz0))
            psih = min(psih, F(0.9) * gz1ozt)
            psim = min(psim, F(0.9) * gz1oz0)
            psih2 = min(psih2, F(0.9) * gz2ozt)
            psim10 = min(psim10, F(0.9) * gz10oz0)
            psih10 = min(psih10, F(0.9) * gz10ozt)
        rmol = F(zol / za)

        psix = F(gz1oz0 - psim)
        psix10 = F(gz10oz0 - psim10)
        ust = F(F(0.5) * ust + F(0.5) * KARMAN * wsp / psix)
        wspdi = max(F(np.sqrt(u1 * u1 + v1 * v1)), WMIN)
        ustm_i = F(F(0.5) * ustm_i + F(0.5) * KARMAN * wspdi / psix)
        if xland < F(1.5):
            ust = max(ust, F(0.005))
            ustm_i = ust

        gz1ozt = F(np.log((za + z0) / zt))
        gz2ozt = F(np.log((F(2.0) + z0) / zt))
        psit = max(F(gz1ozt - psih), F(1.0))
        psit2 = max(F(gz2ozt - psih2), F(1.0))
        psiq = max(F(np.log((za + z0) / zq) - psih), F(1.0))
        psiq2 = max(F(np.log((F(2.0) + z0) / zq) - psih2), F(1.0))
        psiq10 = max(F(np.log((F(10.0) + z0) / zq) - psih10), F(1.0))
        mol_i = F(KARMAN * (thv1 - thvgb) / psit / PRT)
        qstar = F(KARMAN * (qvsh - qsfc) * F(1000.0) / psiq / PRT)

        # WRF recomputes moisture resistance with zq in the numerator here.
        psiq = max(F(np.log((za + zq) / zq) - psih), F(1.0))
        psiq2 = max(F(np.log((F(2.0) + zq) / zq) - psih2), F(1.0))
        psiq10 = max(F(np.log((F(10.0) + zq) / zq) - psih10), F(1.0))
        if isfflx < 1:
            qfx = hfx = flhc = flqc = lh = chs = ch = F(0.0)
            chs2 = cqs2 = ck = cka = cd = cda = F(0.0)
        else:
            flqc = F(rho1 * mavail * ust * KARMAN / psiq)
            flhc = F(rho1 * cpm * ust * KARMAN / psit)
            qfx = max(F(flqc * (qsfcmr - qv1)), F(-0.02))
            lh = F(XLV * qfx)
            # module_sf_mynn.F:1065-1076.  The two arms test XLAND-1.5 for
            # .GT. and .LT. and nothing else: a column at exactly XLAND=1.5
            # takes neither, and HFX keeps the value it entered with.  That is
            # defined behaviour, not a hole, so it is transcribed as written
            # rather than folded into an if/else.
            if xland > F(1.5):
                hfx = F(flhc * (thgb - th1))
                # :1067-1072, the AHW dissipative-heating term.  It is the
                # third and last thing ISFTCFLX selects, it is water-only, and
                # it is keyed on ISFTCFLX.NE.0 -- so every non-default
                # roughness identity, including the ones that share COARE 3.0
                # zt/zq with the default, also adds this to HFX.  USTM is the
                # u* computed without the vconv/vsgd gust (:953-956), which is
                # what the :952 comment means.
                if isftcflx != 0:
                    hfx = F(hfx + F(rho1 * ustm_i * ustm_i * wspdi))
            elif xland < F(1.5):
                hfx = F(flhc * (thgb - th1))
                hfx = max(hfx, F(-250.0))
            chs = F(ust * KARMAN / psit)
            ch = F(flhc / (cpm * rho1))
            cqs2 = F(ust * KARMAN / psiq2)
            chs2 = F(ust * KARMAN / psit2)
            ck = F((KARMAN / psix10) * (KARMAN / psiq10))
            cd = F((KARMAN / psix10) ** F(2.0))
            cka = F((KARMAN / psix) * (KARMAN / psiq))
            cda = F((KARMAN / psix) ** F(2.0))

        if za <= F(7.0):
            if F(7.0) < za2 < F(13.0):
                u10, v10 = u2, v2
            else:
                ratio = F(np.log(F(10.0) / z0) / np.log(za / z0))
                u10, v10 = F(u1 * ratio), F(v1 * ratio)
        elif za < F(13.0):
            ratio = F(np.log(F(10.0) / z0) / np.log(za / z0))
            u10, v10 = F(u1 * ratio), F(v1 * ratio)
        else:
            u10, v10 = F(u1 * psix10 / psix), F(v1 * psix10 / psix)

        th2 = F(thgb + (th1 - thgb) * psit2 / psit)
        if ((th1 > thgb and not thgb <= th2 <= th1)
                or (th1 < thgb and not th1 <= th2 <= thgb)):
            th2 = F(thgb + F(2.0) * (th1 - thgb) / za)
        t2 = F(th2 * (psfc / F(100.0)) ** ROVCP)
        q2 = F(qsfcmr + (qv1 - qsfcmr) * psiq2 / psiq)
        q2 = max(q2, min(qsfcmr, qv1))
        q2 = min(q2, F(1.05) * qv1)

        values_out = {
            "regime": regime, "zol": zol, "rmol": rmol, "ust": ust,
            "ustm": ustm_i, "mol": mol_i, "psim": psim, "psih": psih,
            "chs": chs, "chs2": chs2, "cqs2": cqs2, "ch": ch,
            "flhc": flhc, "flqc": flqc, "qgh": qgh, "qsfc": qsfc,
            "hfx": hfx, "qfx": qfx, "lh": lh, "u10": u10, "v10": v10,
            "th2": th2, "t2": t2, "q2": q2, "gz1oz0": gz1oz0,
            "wspd": wsp, "br": br, "ck": ck, "cka": cka, "cd": cd,
            "cda": cda, "wstar": wstar, "qstar": qstar, "cpm": cpm,
            # module_sf_mynn.F:436 declares ZNT INTENT(INOUT) and every water
            # arm -- :635/:647 charnock_1955, :641 davis_etal_2008, :643
            # Taylor_Yelland_2001 -- mutates it in place; the updated value
            # persists into the next step's ZNTstoch/restar/z_t/z_q.
            "znt": z0,
        }
        for name, value in values_out.items():
            result[name][i] = value

    return result


def mynn_sfclay_first_step_state(
    u1: object, v1: object, qv1: object
) -> dict[str, np.ndarray]:
    """Return the WRF ``SFCLAY_mynn`` ``itimestep==1`` seeding state.

    ``module_sf_mynn.F:329-337`` runs this block in the wrapper, before
    ``SFCLAY1D_mynn`` is entered, from the lowest model level only.  It is a
    wrapper-level prologue, not part of the column solver, so the reference
    exposes it separately instead of folding it into
    :func:`mynn_surface_layer_default`.
    """

    u = np.asarray(u1, dtype=np.float32)
    v = np.asarray(v1, dtype=np.float32)
    q = np.asarray(qv1, dtype=np.float32)
    if u.shape != v.shape or u.shape != q.shape:
        raise ValueError("u1, v1 and qv1 must share one shape")
    ust = np.maximum(
        np.float32(0.04) * np.sqrt(u * u + v * v), np.float32(0.001)
    ).astype(np.float32)
    return {
        "ust": ust,
        "mol": np.zeros(u.shape, dtype=np.float32),
        "qsfc": (q / (np.float32(1.0) + q)).astype(np.float32),
        "qstar": np.zeros(u.shape, dtype=np.float32),
    }


__all__ = [
    "ISFTCFLX_DEFINED",
    "mynn_sfclay_first_step_state",
    "mynn_surface_layer_default",
]
