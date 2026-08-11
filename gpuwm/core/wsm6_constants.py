"""WRF v4.6.1 WSM6 coefficient initialization.

The values and operation order below follow ``mp_wsm6_init`` in
``phys/physics_mmm/mp_wsm6.F90``.  Keeping the derived constants on the
host makes the hail/graupel selection explicit and lets restart identity
bind the actual coefficient set instead of only the namelist integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math


@dataclass(frozen=True)
class WSM6RimedIceConstants:
    n0g: float
    deng: float
    avtg: float
    bvtg: float
    lamdagmax: float


@dataclass(frozen=True)
class WSM6Coefficients:
    rimed: WSM6RimedIceConstants
    xlv1: float
    qc0: float
    qck1: float
    pidnc: float
    pidn0r: float
    pidn0s: float
    pidn0g: float
    pvtr: float
    pvts: float
    pvtg: float
    pacrr: float
    pacrs: float
    pacrg: float
    pacrc: float
    precr1: float
    precr2: float
    precs1: float
    precs2: float
    precg1: float
    precg2: float
    roqimax: float
    rslopermax: float
    rslopesmax: float
    rslopegmax: float
    rsloperbmax: float
    rslopesbmax: float
    rslopegbmax: float
    rsloper2max: float
    rslopes2max: float
    rslopeg2max: float
    rsloper3max: float
    rslopes3max: float
    rslopeg3max: float


def _rgmma(x: float) -> float:
    """WSM6's Weierstrass-product Gamma implementation, verbatim."""
    if x == 1.0:
        return 0.0
    value = x * math.exp(0.577215664901532 * x)
    for index in range(1, 10001):
        y = float(index)
        value *= (1.0 + x / y) * math.exp(-x / y)
    return 1.0 / value


def rimed_ice_constants(hail_opt: int) -> WSM6RimedIceConstants:
    """Return WRF's five ``hail_opt``-selected WSM6 constants."""
    if int(hail_opt) == 0:
        return WSM6RimedIceConstants(4.0e6, 500.0, 330.0, 0.8, 6.0e4)
    if int(hail_opt) == 1:
        return WSM6RimedIceConstants(4.0e4, 700.0, 285.0, 0.8, 2.0e4)
    raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")


@lru_cache(maxsize=2)
def coefficients(hail_opt: int = 0) -> WSM6Coefficients:
    """Reproduce ``mp_wsm6_init`` for liquid density 1000 kg m-3."""
    r = rimed_ice_constants(hail_opt)
    pi = 4.0 * math.atan(1.0)
    den0, denr, dens = 1.28, 1000.0, 100.0
    cliq, cpv = 4190.0, 1846.4
    n0r, n0s = 8.0e6, 2.0e6
    avtr, bvtr, avts, bvts = 841.9, 0.8, 11.72, 0.41

    g3r, g4r = _rgmma(3.0 + bvtr), _rgmma(4.0 + bvtr)
    g5r2 = _rgmma(2.5 + 0.5 * bvtr)
    g3s, g4s = _rgmma(3.0 + bvts), _rgmma(4.0 + bvts)
    g5s2 = _rgmma(2.5 + 0.5 * bvts)
    g3g, g4g = _rgmma(3.0 + r.bvtg), _rgmma(4.0 + r.bvtg)
    g5g2 = _rgmma(2.5 + 0.5 * r.bvtg)

    rr, rs, rg = 1.0 / 8.0e4, 1.0 / 1.0e5, 1.0 / r.lamdagmax
    return WSM6Coefficients(
        rimed=r,
        xlv1=cliq - cpv,
        qc0=(4.0 / 3.0) * pi * denr * (0.8e-5 ** 3) * 3.0e8 / den0,
        qck1=(0.104 * 9.8 * 0.55 / ((3.0e8 * denr) ** (1.0 / 3.0))
              / 1.718e-5 * den0 ** (4.0 / 3.0)),
        pidnc=pi * denr / 6.0,
        pidn0r=pi * denr * n0r,
        pidn0s=pi * dens * n0s,
        pidn0g=pi * r.deng * r.n0g,
        pvtr=avtr * g4r / 6.0,
        pvts=avts * g4s / 6.0,
        pvtg=r.avtg * g4g / 6.0,
        pacrr=pi * n0r * avtr * g3r * 0.25,
        pacrs=pi * n0s * avts * g3s * 0.25,
        pacrg=pi * r.n0g * r.avtg * g3g * 0.25,
        pacrc=pi * n0s * avts * g3s * 0.25,
        precr1=2.0 * pi * n0r * 0.78,
        precr2=2.0 * pi * n0r * 0.31 * math.sqrt(avtr) * g5r2,
        precs1=4.0 * n0s * 0.65,
        precs2=4.0 * n0s * 0.44 * math.sqrt(avts) * g5s2,
        precg1=2.0 * pi * r.n0g * 0.78,
        precg2=2.0 * pi * r.n0g * 0.31 * math.sqrt(r.avtg) * g5g2,
        roqimax=2.08e22 * 500.0e-6 ** 8,
        rslopermax=rr, rslopesmax=rs, rslopegmax=rg,
        rsloperbmax=rr ** bvtr, rslopesbmax=rs ** bvts,
        rslopegbmax=rg ** r.bvtg,
        rsloper2max=rr * rr, rslopes2max=rs * rs,
        rslopeg2max=rg * rg,
        rsloper3max=rr ** 3, rslopes3max=rs ** 3,
        rslopeg3max=rg ** 3,
    )


#: ``kernels/wsm6.cu``'s ``WSM6_KMAX`` ladder.  One CUDA thread owns a whole
#: column, so the bound sizes the per-thread work arrays and the driver
#: reserves a local-memory backing store for the device's entire resident
#: -thread capacity at first launch -- the law at the head of
#: ``gpuwm/core/preflight.py``.  ``gpuwm/core/wsm6.py`` therefore compiles the
#: SHALLOWEST tier that holds the configuration's ``nz``, and compiles a TIER
#: rather than ``nz`` itself so a nested tree with several level counts loads
#: at most two WSM6 modules.
#:
#: 64 is the ``#ifndef`` default baked into the source, which is why it is
#: also the value the driver-regeneration gate reads back for the
#: unspecialized module and therefore the value recorded in
#: ``KERNEL_MAX_LOCAL_SIZE_BYTES``.  The 80 tier prices ABOVE that row;
#: ``preflight.WSM6_TIER_FRAME`` is what carries it.
#:
#: This ladder lives in the constants module, not in ``gpuwm/core/wsm6.py``,
#: for one reason: the adapter imports CuPy at module scope and the memory
#: estimator must price a configuration on a host with no device at all.
WSM6_SHALLOW_KMAX = 64
WSM6_DEEP_KMAX = 80
WSM6_KERNEL_LEVEL_TIERS = (WSM6_SHALLOW_KMAX, WSM6_DEEP_KMAX)
#: ``(minimum, maximum)`` levels WSM6 admits: 2 (the scheme needs a layer
#: above and below to sediment between) up to the deepest compiled tier.
WSM6_VERTICAL_LEVEL_BOUNDS = (2, WSM6_DEEP_KMAX)


def wsm6_level_tier(nz: int) -> int:
    """Return the ``WSM6_KMAX`` tier ``wsm6.cu`` compiles for ``nz``."""
    nz = int(nz)
    minimum, maximum = WSM6_VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WSM6 requires {minimum} <= nz <= {maximum}, got {nz}")
    return next(tier for tier in WSM6_KERNEL_LEVEL_TIERS if nz <= tier)


__all__ = ["WSM6Coefficients", "WSM6RimedIceConstants",
           "WSM6_DEEP_KMAX", "WSM6_KERNEL_LEVEL_TIERS", "WSM6_SHALLOW_KMAX",
           "WSM6_VERTICAL_LEVEL_BOUNDS", "coefficients",
           "rimed_ice_constants", "wsm6_level_tier"]
