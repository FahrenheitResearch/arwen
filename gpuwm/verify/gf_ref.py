"""Float32 CPU reference for WRF v4.6.1 Grell-Freitas (cu_physics = 3).

Built stage by stage against ``tools/gf_wrf461_oracle``.  Each function here
mirrors one span of the pinned source and is measured bitwise against the
oracle before the next one is written; nothing is added on the strength of
reading the Fortran alone.

Stage 1, ``gf_driver_prep``, is the column preparation GFDRV does before it
calls anything (module_cu_gf_wrfdrv.F:383-492).

Precision, which is not uniform and is the first thing this port had to get
right
----------------------------------------------------------------------------
GFDRV's ``g``, ``cp``, ``xlv`` and ``r_v`` come from ``module_gfs_physcons``,
where they are declared ``real(kind_phys)`` with ``kind_phys =
selected_real_kind(13,60)`` -- i.e. **real(8)** -- and then initialised from
default-real literals (``con_g = 9.80665e+0``, not ``9.80665d+0``).  Three
driver expressions therefore evaluate in **double** and round exactly once,
on the store into a real(4) local:

    omeg(I,K) = -g*rho*w                                      (:471)
    mconv(i)  = mconv(i) + omeg(i,k)*dq/g                     (:487)
    DHDT(I,K) = cp*RTHBLTEN*pi + XLV*RQVBLTEN                 (:416)

Everything else in the preparation is float32 end to end.

Which double, though, is a question the source does not settle and the
fixture does.  Assigned to a real(8) and written out, ``con_g`` reads
``0x40239D0140000000`` = ``float64(float32(9.80665))``, not the honest
``0x40239D013A92A305``.  But the arithmetic does not behave that way.
Measured over the whole fixture -- 8640 lanes each of ``omeg`` and ``dhdt``
from ``gf-stage-levels.csv``:

    ================================  ==========  ==========
    constants used                    omeg wrong  dhdt wrong
    ================================  ==========  ==========
    float64(9.80665) etc                       0           0
    float64(float32(9.80665)) etc           1488         204
    float32 throughout                      2700         360
    ================================  ==========  ==========

so the constants below are the honest doubles, and the stored-word reading is
recorded in ``gf-pow-probe.txt`` as the trap it is.  Every miss in the middle
row is exactly 1 ULP, which is precisely the size of error that survives a
careless port and then gets amplified by a branch.

The deep and shallow modules do NOT share these constants.  They declare
their own (``g = 9.81``, ``cp = 1004.``, ``r_v = 461.``, ``xlv = 2.5e6``) as
plain real(4) parameters, and three of the four disagree with the GFS set.
Both sets are live in one call.  Nothing here harmonises them.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GFS_G",
    "GFS_CP",
    "GFS_XLV",
    "GFS_RV",
    "DEEP_G",
    "DEEP_CP",
    "DEEP_RV",
    "DEEP_XLV",
    "gf_driver_prep",
]

# --- the driver's constants, at the width its expressions evaluate in ------
# Honest float64, per the fixture measurement in this module's docstring --
# NOT float64(float32(v)), which is what the stored parameter word reads as.
GFS_G = np.float64(9.80665e0)
GFS_CP = np.float64(1.0046e3)
GFS_XLV = np.float64(2.5000e6)
GFS_RV = np.float64(4.6150e2)

# --- the cloud model's own constants: plain real(4) -------------------------
DEEP_G = np.float32(9.81)
DEEP_CP = np.float32(1004.0)
DEEP_RV = np.float32(461.0)
DEEP_XLV = np.float32(2.5e6)

_QMIN = np.float32(1.0e-08)
_TMIN = np.float32(200.0)
_MB = np.float32(0.01)
_HALF = np.float32(0.5)
_CCN = np.float32(150.0)


def gf_driver_prep(
    *,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    t: np.ndarray,
    qv: np.ndarray,
    p: np.ndarray,
    pi: np.ndarray,
    rho: np.ndarray,
    dz8w: np.ndarray,
    p8w: np.ndarray,
    rthften: np.ndarray,
    rqvften: np.ndarray,
    rthraten: np.ndarray,
    rthblten: np.ndarray,
    rqvblten: np.ndarray,
    ht: np.ndarray,
    hfx: np.ndarray,
    qfx: np.ndarray,
    xland: np.ndarray,
    kpbl: np.ndarray,
    dt: float,
    dx: float,
) -> dict[str, np.ndarray]:
    """GFDRV's column preparation, module_cu_gf_wrfdrv.F:383-492.

    Per-level inputs are ``(ncol, nz)`` float32; per-column inputs ``(ncol,)``.
    Returns the locals GFDRV hands to ``CUP_gf_sh`` and ``cup_gf``, by their
    WRF names.
    """
    ncol, nz = t.shape
    f32 = np.float32
    dtf = f32(dt)

    def a32(x):
        return np.asarray(x, dtype=np.float32)

    u, v, w, t, qv = (a32(x) for x in (u, v, w, t, qv))
    p, pi, rho, dz8w, p8w = (a32(x) for x in (p, pi, rho, dz8w, p8w))
    rthften, rqvften, rthraten = (a32(x) for x in (rthften, rqvften, rthraten))
    rthblten, rqvblten = (a32(x) for x in (rthblten, rqvblten))
    ht, hfx, qfx, xland = (a32(x) for x in (ht, hfx, qfx, xland))

    # --- per-column scalars (:353-357, :369, :384-395) ---------------------
    psur = (p8w[:, 0] * _MB).astype(np.float32)
    ter11 = np.maximum(f32(0.0), ht).astype(np.float32)
    hfxi = hfx.copy()
    qfxi = qfx.copy()
    xlandi = xland.copy()
    kpbli = np.asarray(kpbl, dtype=np.int32).copy()
    dxi = np.full(ncol, f32(dx), dtype=np.float32)
    ccn = np.full(ncol, _CCN, dtype=np.float32)

    # --- the zo half-layer stack (:396-399) --------------------------------
    # Sequential by construction: zo(k) depends on zo(k-1), so this cannot be
    # a cumulative sum in a wider type without changing the rounding.
    zo = np.empty((ncol, nz), dtype=np.float32)
    zo[:, 0] = (ter11 + _HALF * dz8w[:, 0]).astype(np.float32)
    for k in range(1, nz):
        zo[:, k] = (
            zo[:, k - 1] + _HALF * (dz8w[:, k - 1] + dz8w[:, k])
        ).astype(np.float32)

    # --- the mass-level column (:402-438) ----------------------------------
    po = (p * _MB).astype(np.float32)
    p2d = po
    rhoi = rho.copy()
    us = u.copy()
    vs = v.copy()
    t2d = t.copy()

    # IF(Q2d.LT.1.E-08) Q2d = 1.E-08 -- a floor, not a max(): NaN would pass
    # through the .LT. as false and survive, which is what WRF does.
    q2d = np.where(qv < _QMIN, _QMIN, qv).astype(np.float32)

    tn = (t2d + (rthften + rthraten + rthblten) * pi * dtf).astype(np.float32)
    qo = (q2d + (rqvften + rqvblten) * dtf).astype(np.float32)
    tshall = (t2d + rthblten * pi * dtf).astype(np.float32)
    qshall = (q2d + rqvblten * dtf).astype(np.float32)

    # dhdt: real(8) constants, so the whole right-hand side is evaluated in
    # double and rounded once on the store (:416).
    dhdt = (
        GFS_CP * rthblten.astype(np.float64) * pi.astype(np.float64)
        + GFS_XLV * rqvblten.astype(np.float64)
    ).astype(np.float32)

    tn = np.where(tn < _TMIN, t2d, tn).astype(np.float32)
    qo = np.where(qo < _QMIN, _QMIN, qo).astype(np.float32)

    # omeg: same story (:471).
    omeg = (
        -GFS_G * rho.astype(np.float64) * w.astype(np.float64)
    ).astype(np.float32)

    # --- mconv (:484-492) ---------------------------------------------------
    # dq is a real(4) local, so omeg*dq is float32; the division by the real(8)
    # g widens, and the accumulator rounds back to float32 every iteration.
    mconv = np.zeros(ncol, dtype=np.float32)
    for k in range(nz - 1):
        dq = (q2d[:, k + 1] - q2d[:, k]).astype(np.float32)
        term = (omeg[:, k] * dq).astype(np.float32)
        mconv = (mconv.astype(np.float64) + term.astype(np.float64) / GFS_G).astype(
            np.float32
        )
    mconv = np.where(mconv < f32(0.0), f32(0.0), mconv).astype(np.float32)

    return {
        "zo": zo,
        "po": po,
        "p2d": p2d,
        "t2d": t2d,
        "q2d": q2d,
        "tn": tn,
        "qo": qo,
        "tshall": tshall,
        "qshall": qshall,
        "dhdt": dhdt,
        "us": us,
        "vs": vs,
        "rhoi": rhoi,
        "omeg": omeg,
        "psur": psur,
        "ter11": ter11,
        "hfxi": hfxi,
        "qfxi": qfxi,
        "xlandi": xlandi,
        "kpbli": kpbli,
        "dxi": dxi,
        "ccn": ccn,
        "mconv": mconv,
    }
