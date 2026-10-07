"""Independent float32 WRF-Chem flow-dependent boundary reference."""
from __future__ import annotations
import numpy as np
from gpuwm.core.chem_bdy import EPSILC, _validate

def apply_chem_flow_boundaries_cpu(fields, ru_m, rv_m, spec_zone, has_bc,
                                  default_inflow, bxs, btxs, bxe, btxe,
                                  bys, btys, bye, btye, dt):
    """In-place NumPy float32 reference of the pinned side loops."""
    fields = tuple(fields)
    bounds = (bxs, btxs, bxe, btxe, bys, btys, bye, btye)
    checked = _validate(fields, ru_m, rv_m, spec_zone, has_bc, default_inflow, bounds, np)
    if checked is None:
        return
    nz, ny, nx, w, has, defaults = checked
    dt = np.float32(dt)
    for n, f in enumerate(fields):
        for side in range(4):
            for d in range(w):
                distance = d if side in (0, 2) else w-1-d
                transverse = (range(distance, nx-distance) if side < 2
                              else range(distance+1, ny-distance-1))
                for k in range(nz):
                    for a in transverse:
                        if side < 2:
                            j, i = (d if side == 0 else ny-w+d), a
                            inner = min(max(i, w), nx-1-w)
                            out = rv_m[k, j if side == 0 else j+1, i]
                            out = out < 0 if side == 0 else out > 0
                            source = (k, w if side == 0 else ny-1-w, inner)
                            b, bt = (bys, btys) if side == 0 else (bye, btye)
                        else:
                            i, j = (d if side == 2 else nx-w+d), a
                            inner = min(max(j, w), ny-1-w)
                            out = ru_m[k, j, i if side == 2 else i+1]
                            out = out < 0 if side == 2 else out > 0
                            source = (k, inner, w if side == 2 else nx-1-w)
                            b, bt = (bxs, btxs) if side == 2 else (bxe, btxe)
                        if out:
                            f[k, j, i] = f[source]
                        elif has[n]:
                            f[k, j, i] = np.maximum(EPSILC, np.float32(
                                b[n, k, a] + np.float32(bt[n, k, a] * dt)))
                        else:
                            f[k, j, i] = defaults[n]
