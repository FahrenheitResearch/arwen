"""Stock-WRF mixscalars arms of MYNN — the W4 full-admission lane (stock half).

ADDITIVE translation unit (the W4 wave of the MYNN port): the frozen
``kernels/mynn_pbl.cu`` (byte pin ``b53ab90e...`` in
``tests/test_mp8_frozen.py``) is never edited.  The five stock qn-family
tridiagonal solves (``module_bl_mynn.F:4654/:4695/:4736/:4778/:4820``,
``nonloc=1.0`` parameter at ``:4123``) and the ``scalar_opt>0`` DMP_mf
updraft-flux accumulation (``:6447-6456``, plus the plume init/entrain/store
lines ``:6140-6144``/``:6213-6217``/``:6351-6355``) live HERE, consuming the
plume-edge terms the admitted DMP_mf transcription already computes
(``up_w``/``up_a``/``ent``/``rhoz``/``psig_w``), never recomputing them.

Source of record for every line: WRF v4.6.1 ``phys/module_bl_mynn.F``,
SHA-256 ``b36c8b935cd9c8265359e78dbe8db285d99e992c7cab29917b5fca9d57d49452``
(the sha the anchored ``w4-oracle-fixtures`` family
pins; verified on node-1 this lane).  FP32 op order is WRF's, in the same
``F(...)``-per-operation discipline as :mod:`gpuwm.core.mynn_pbl`.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.mynn_pbl import _tridiag2_fp32, _up

F = np.float32

#: The five stock qn-family species, in WRF's solve order
#: (qni :4654, qnc :4695, qnwfa :4736, qnifa :4778, qnbca :4820).
QN_SOLVE_ORDER: tuple[str, ...] = ("qni", "qnc", "qnwfa", "qnifa", "qnbca")

#: module_bl_mynn.F:4121-4123 — "Activate nonlocal mixing from the
#: mass-flux scheme for number concentrations and aerosols".  Parameter,
#: not namelist; 1.0 in stock WRF 4.6.1.
NONLOC = F(1.0)


def dmp_qn_flux_column(
    qn: np.ndarray,
    dz: np.ndarray,
    zw: np.ndarray,
    up_w: np.ndarray,
    up_a: np.ndarray,
    ent: np.ndarray,
    rhoz: np.ndarray,
    psig_w: np.float32,
    plume_active: bool,
    limiter_adjustment: np.float32,
) -> np.ndarray:
    """One species' ``s_awqn`` for one column (``module_bl_mynn.F``).

    Replays ONLY the qn plume lines against the plume-edge terms the
    admitted DMP_mf already produced (nothing about plume dynamics is
    recomputed):

    * init ``:6140-6144``: ``UPQN(1,I)`` is the surface interface
      interpolation of the environment profile;
    * entrainment update ``:6213-6217``: ``QNn = UPQN(k-1,I)*(1-EntExp)
      + QN(k)*EntExp`` with ``EntExp = ENT(K,I)*(ZW(k+1)-ZW(k))``, the
      linearized form the admitted qt/thl updates share;
    * store ``:6351-6355``: kept only where the plume survived
      (``Wn > 0``), which is exactly ``up_w[k, i] > 0`` — above plume
      death ``up_w`` is structurally zero, so the stale ``UPQN`` levels
      can never reach an accumulated flux;
    * accumulation ``:6447-6456`` (gate ``if (scalar_opt > 0)``):
      ``s_awqn(k+1) += rhoz(k)*UPA(K,i)*UPW(K,i)*UPQN(K,i)*Psig_w`` with
      the same product grouping as the admitted ``s_awthl`` line.

    ``plume_active`` is WRF's ``NUP2 > 0``: when the plume model bailed
    out (``:6170`` k=1 dead plume, or the outer trigger at ``:6035``),
    the whole ``:6404-6461`` accumulation block is skipped and every
    ``s_awqn`` stays exactly zero even though the surface plume init is
    nonzero — the gate cannot be inferred from ``up_w`` alone.
    """

    nz = qn.size
    nup = up_w.shape[1]
    s_awqn = np.zeros(nz + 1, dtype=np.float32)
    if not plume_active:
        return s_awqn

    up_qn = np.zeros((nz + 1, nup), dtype=np.float32)
    for i in range(nup):
        # :6140-6144 surface plume value.
        up_qn[0, i] = _up(qn[0], qn[1], dz[0], dz[1])
        for k in range(1, nz - 1):
            if up_w[k, i] > F(0.0):
                # :6213 EntExp, :6217 QNn, :6351-6355 store.
                ent_exp = F(ent[k, i] * F(zw[k + 1] - zw[k]))
                up_qn[k, i] = F(
                    F(up_qn[k - 1, i] * F(F(1.0) - ent_exp))
                    + F(qn[k] * ent_exp)
                )
            # else: the Fortran loop broke before storing; up_qn stays 0
            # exactly like UPQNC there, and up_w stays 0 above too.

    # :6447-6456 — k outer, plume inner, so the FP32 addition order into
    # each interface accumulator is WRF's.  The Fortran loop runs
    # ``do k=kts,kte`` (one level past the admitted s_aw loop); the extra
    # top term is an exact +0.0 because ``up_w(kte+1,:)`` is never
    # written, and it is kept so the loop shape stays the source's.
    for k in range(nz):
        for i in range(nup):
            s_awqn[k + 1] = F(s_awqn[k + 1] + F(F(F(
                F(rhoz[k] * up_a[k, i]) * up_w[k, i]) * up_qn[k, i])
                * psig_w))

    # :6485-6489 — the heat-flux limiter scales every s_awqn* alongside
    # the admitted s_aw* lines, BEFORE UPA is scaled at :6497, which is
    # why ``up_a`` above must be the pre-limiter plume area.  When the
    # limiter did not fire the factor is exactly 1.0 and this multiply is
    # an FP32 identity.
    for k in range(nz + 1):
        s_awqn[k] = F(s_awqn[k] * limiter_adjustment)
    return s_awqn


def mix_scalar_column(
    qn: np.ndarray,
    dtz: np.ndarray,
    rhoinv: np.ndarray,
    khdz: np.ndarray,
    hdz: np.ndarray,
    dzinv: np.ndarray,
    s_aw: np.ndarray,
    s_awqn: np.ndarray,
    delt: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """One stock qn tridiagonal solve + its tendency, for one column.

    ``module_bl_mynn.F:4654-4689`` (qni; the qnc/qnwfa/qnifa/qnbca blocks
    at ``:4695/:4736/:4778/:4820`` are the same arithmetic — the only
    textual differences are commutative ``khdz`` operand swaps).  Inputs
    ``dtz``/``rhoinv``/``khdz``/``hdz``/``dzinv`` are the SAME arrays the
    admitted ``mynn_tendencies`` solve built — including the ``:4163-4169``
    stability floors on ``khdz`` — consumed, not recomputed.  There is NO
    surface-flux term in any qn RHS (fixture manifest, verified against
    the source: ``d(kts)`` carries only ``qn`` and the ``s_awqn`` flux),
    no ``sd_aw`` term (``bl_mynn_edmf_dd=0``), and the top boundary is the
    prescribed value ``d(kte)=qn(kte)``.  Tendency: ``(qn2-qn)/delt``
    (``:4957-4966``/``:4998-5007``/``:5060-5077``/``:5082-5091``), with no
    positivity clamp — the clamp lines are commented out in the source.
    """

    nz = qn.size
    a = np.empty(nz, dtype=np.float32)
    b = np.empty(nz, dtype=np.float32)
    c = np.empty(nz, dtype=np.float32)
    d = np.empty(nz, dtype=np.float32)

    a[0] = -F(F(dtz[0] * khdz[0]) * rhoinv[0])
    b[0] = F(F(F(1.0) + F(F(dtz[0] * F(khdz[1] + khdz[0])) * rhoinv[0]))
             - F(F(hdz[0] * s_aw[1]) * NONLOC))
    c[0] = F(-F(F(dtz[0] * khdz[1]) * rhoinv[0])
             - F(F(hdz[0] * s_aw[1]) * NONLOC))
    d[0] = F(qn[0] - F(F(dzinv[0] * s_awqn[1]) * NONLOC))
    for k in range(1, nz - 1):
        a[k] = F(-F(F(dtz[k] * khdz[k]) * rhoinv[k])
                 + F(F(hdz[k] * s_aw[k]) * NONLOC))
        b[k] = F(F(F(1.0)
                   + F(F(dtz[k] * F(khdz[k] + khdz[k + 1])) * rhoinv[k]))
                 + F(F(hdz[k] * F(s_aw[k] - s_aw[k + 1])) * NONLOC))
        c[k] = F(-F(F(dtz[k] * khdz[k + 1]) * rhoinv[k])
                 - F(F(hdz[k] * s_aw[k + 1]) * NONLOC))
        d[k] = F(qn[k] + F(F(dzinv[k] * F(s_awqn[k] - s_awqn[k + 1]))
                           * NONLOC))
    a[nz - 1] = F(0.0)
    b[nz - 1] = F(1.0)
    c[nz - 1] = F(0.0)
    d[nz - 1] = qn[nz - 1]

    qn2 = _tridiag2_fp32(a, b, c, d)
    dqn = np.asarray(
        [F(F(qn2[k] - qn[k]) / delt) for k in range(nz)], dtype=np.float32
    )
    return qn2, dqn


__all__ = ["QN_SOLVE_ORDER", "NONLOC", "dmp_qn_flux_column",
           "mix_scalar_column"]
