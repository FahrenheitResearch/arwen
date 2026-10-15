"""WRF v4.6.1 WSM3 coefficient words and the kernel's level ladder.

A CuPy-free leaf: the memory preflight prices a WSM3 configuration on a
host with no device, and the tests read the frozen words without a card.
"""

from __future__ import annotations

#: The float32 words WRF v4.6.1's ``wsm3init`` leaves in module_mp_wsm3
#: (phys/module_mp_wsm3.F:926-981), called as module_physics_init.F:4483
#: calls it (rhoair0, rhowater, rhosnow, cliq, cpv), in the order
#: ``kernels/wsm3.cu``'s ``Wsm3Coef`` enum reads them.
#:
#: They are WRF's own float32 arithmetic -- the Weierstrass-product
#: ``rgmma`` run in float32 through the libm ``expf`` included -- which is
#: why they are frozen bit patterns and not recomputed here.  Dumped by
#: tools/wsm3_wrf461_column_oracle/dump_wsm3_init.F90 (gfortran 13.3.0 -O0,
#: glibc 2.39) into ``wsm3-init-f32.txt``, and PROVEN by the column oracle:
#: a wrong word cannot reproduce WRF's outputs bit for bit
#: (tests/test_wsm3_wrf461_column_exact.py).
WRF_INIT_F32 = (
    ("pi", 0x40490FDB),  # 3.141592741E+00
    ("xlv1", 0x4512799A),  # 2.343600098E+03
    ("qc0", 0x3A03C498),  # 5.026548170E-04
    ("qck1", 0x40D8C3BE),  # 6.773894310E+00
    ("pidnc", 0x4402E653),  # 5.235988159E+02
    ("pidn0r", 0x50BB40E7),  # 2.513274266E+10
    ("pidn0s", 0x4E15CD85),  # 6.283185280E+08
    ("pvtr", 0x451C4105),  # 2.500063721E+03
    ("pvts", 0x41A069C1),  # 2.005163765E+01
    ("pacrr", 0x50B8DFD0),  # 2.481340416E+10
    ("pacrs", 0x4C537EB0),  # 5.544211200E+07
    ("precr1", 0x4C159029),  # 3.920707600E+07
    ("precr2", 0x4E44E61A),  # 8.258535680E+08
    ("precs1", 0x4A9EB100),  # 5.200000000E+06
    ("precs2", 0x4B8E882C),  # 1.868194400E+07
    ("roqimax", 0x38AA64C8),  # 8.125003660E-05
    ("xmmax", 0x30F2A2ED),  # 1.765412372E-09
    ("rslopermax", 0x3751B717),  # 1.249999968E-05
    ("rslopesmax", 0x3727C5AC),  # 9.999999747E-06
    ("rsloperbmax", 0x38FAB3B8),  # 1.195440418E-04
    ("rslopesbmax", 0x3C1205C6),  # 8.912509307E-03
    ("rsloper2max", 0x2F2BCC76),  # 1.562499852E-10
    ("rslopes2max", 0x2EDBE6FE),  # 9.999999440E-11
    ("rsloper3max", 0x270CBCCB),  # 1.953124780E-15
    ("rslopes3max", 0x26901D7C),  # 9.999998977E-16
)

#: Names in kernel order, for the device upload and the table tests.
WRF_INIT_NAMES = tuple(name for name, _ in WRF_INIT_F32)


def wrf_init_words():
    """``wsm3init``'s words as a float32 numpy array, in kernel order."""
    import numpy as np

    return np.asarray([bits for _, bits in WRF_INIT_F32],
                      dtype=np.uint32).view(np.float32)


#: ``kernels/wsm3.cu``'s ``WSM3_KMAX`` ladder.  One CUDA thread owns a whole
#: column, so the bound sizes the per-thread work arrays; ``gpuwm/core/
#: wsm3.py`` compiles the SHALLOWEST tier that holds the configuration's
#: ``nz``.  64 is the ``#ifndef`` default baked into the source.  The ladder
#: matches WSM6's (gpuwm/core/wsm6_constants.py) so the two single-moment
#: schemes admit the same level counts.
WSM3_SHALLOW_KMAX = 64
WSM3_DEEP_KMAX = 80
WSM3_KERNEL_LEVEL_TIERS = (WSM3_SHALLOW_KMAX, WSM3_DEEP_KMAX)
#: ``(minimum, maximum)`` levels WSM3 admits: 2 (its interface interpolation
#: reads a layer above and below) up to the deepest compiled tier.
WSM3_VERTICAL_LEVEL_BOUNDS = (2, WSM3_DEEP_KMAX)


def wsm3_level_tier(nz: int) -> int:
    """Return the ``WSM3_KMAX`` tier ``wsm3.cu`` compiles for ``nz``."""
    nz = int(nz)
    minimum, maximum = WSM3_VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WSM3 requires {minimum} <= nz <= {maximum}, got {nz}")
    return next(tier for tier in WSM3_KERNEL_LEVEL_TIERS if nz <= tier)


__all__ = ["WRF_INIT_F32", "WRF_INIT_NAMES", "WSM3_DEEP_KMAX",
           "WSM3_KERNEL_LEVEL_TIERS", "WSM3_SHALLOW_KMAX",
           "WSM3_VERTICAL_LEVEL_BOUNDS", "wrf_init_words", "wsm3_level_tier"]
