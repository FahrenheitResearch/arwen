"""MYJSFCINIT similarity-function tables (WRF v4.6.1 module_sf_myjsfc.F).

NumPy-only on purpose: this is the one piece of the MYJ surface layer both
the CPU float32 authority (:mod:`gpuwm.verify.myj_ref`) and the CUDA
launcher (:mod:`gpuwm.core.myjsfc`) consume, and core modules must not
import the verification tree (the ``sase_limits`` precedent in
gpuwm/core/state.py) while verify importing core is the allowed direction.

``build_psi_tables`` transcribes the function-definition loop of
``MYJSFCINIT`` (module_sf_myjsfc.F:1174-1299): KZTM=10001 entries per
table, Paulson 1970 in the unstable range, Holtslag and de Bruin 1988 in
the stable range, with the float32 running ``ZETA`` accumulation that makes
``ZTMAX`` the accumulated endpoint minus ``EPS`` (:1288-1299) rather than
the literal upper bound.  Sea (``*1``) and land (``*2``) tables are built
independently exactly as the Fortran builds them even though v4.6.1's
ranges make them identical.  ``FH01 = FH02 = 1`` (:1177-1178).
"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType

import numpy as np

F = np.float32

#: module_sf_myjsfc.F:55.
KZTM = 10001
KZTM2 = KZTM - 2

_PI = F(3.1415926)
_PIHF = F(F(0.5) * _PI)


def _build_pair(ztmin: np.float32, dzeta: np.float32):
    psim = np.empty(KZTM, dtype=F)
    psih = np.empty(KZTM, dtype=F)
    zeta = ztmin
    ztmax = ztmin
    for k in range(KZTM):
        if zeta < F(0.0):
            x = F(np.sqrt(np.sqrt(F(F(1.0) - F(F(16.0) * zeta)))))
            # Left-to-right exactly as the Fortran writes it (:1216):
            # (((-2*log((x+1)/2)) - log((x*x+1)/2)) + 2*atan(x)) - PIHF.
            psim[k] = F(F(F(F(F(-2.0)
                              * F(np.log(F(F(x + F(1.0)) / F(2.0)))))
                            - F(np.log(F(F(F(x * x) + F(1.0)) / F(2.0)))))
                          + F(F(2.0) * F(np.arctan(x))))
                        - _PIHF)
            psih[k] = F(F(-2.0)
                        * F(np.log(F(F(F(x * x) + F(1.0)) / F(2.0)))))
        else:
            psim[k] = F(F(F(0.7) * zeta)
                        + F(F(F(F(0.75) * zeta)
                              * F(F(6.0) - F(F(0.35) * zeta)))
                            * F(np.exp(F(F(-0.35) * zeta)))))
            psih[k] = psim[k]
        if k == KZTM - 1:
            ztmax = zeta
        zeta = F(zeta + dzeta)
    ztmax = F(ztmax - F(1.0e-6))
    return psim, psih, ztmax


@lru_cache(maxsize=1)
def build_psi_tables():
    """The complete MYJSFCINIT table set, cached once per process."""
    ztmin1 = F(-5.0)
    ztmin2 = F(-5.0)
    dzeta1 = F(F(F(1.0) - ztmin1) / F(KZTM - 1))
    dzeta2 = F(F(F(1.0) - ztmin2) / F(KZTM - 1))
    psim1, psih1, ztmax1 = _build_pair(ztmin1, dzeta1)
    psim2, psih2, ztmax2 = _build_pair(ztmin2, dzeta2)
    psim1.setflags(write=False)
    psih1.setflags(write=False)
    psim2.setflags(write=False)
    psih2.setflags(write=False)
    return MappingProxyType({
        "psim1": psim1, "psih1": psih1, "psim2": psim2, "psih2": psih2,
        "dzeta1": dzeta1, "dzeta2": dzeta2,
        "ztmin1": ztmin1, "ztmax1": ztmax1,
        "ztmin2": ztmin2, "ztmax2": ztmax2,
        "fh01": F(1.0), "fh02": F(1.0),
    })


__all__ = ["KZTM", "KZTM2", "build_psi_tables"]
