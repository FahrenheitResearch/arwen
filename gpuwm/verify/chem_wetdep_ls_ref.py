"""Float32 transcription of chem/module_wetdep_ls.F:49-91 (WRF v4.7.1).

Inputs use (row, k, j, i), with nz the chem_driver mass-level count.
Every column starts from zeroed sums; the source writes every scratch read.
The added removed-mass diagnostic uses dry density, not the moist density
used by the transcribed scavenging arithmetic.
"""
import numpy as np


def wetdep_ls(var, rain, qc, rho, dz, w, dt, alpha, dryrho=None):
    f = np.float32
    out = np.array(var, dtype=f, copy=True)
    dryrho = rho if dryrho is None else dryrho
    removed = np.zeros((out.shape[0], *out.shape[2:]), dtype=f)
    nz = out.shape[1]
    for r, a in enumerate(alpha):
        a = f(a)
        if a == 0:
            continue
        for j, i in np.ndindex(out.shape[2:]):
            clw, total = f(0), f(0)
            if not rain[j, i] > f(1e-10):
                continue
            for k in range(nz - 1):
                ix = (k, j, i)
                dvar = max(f(0), f(f(f(qc[ix] * rho[ix]) * w[ix]) * dz[ix]))
                clw = f(clw + dvar)
                total = f(total + f(out[r, k, j, i] * rho[ix]))
            if not (total > f(1e-10) and clw > f(1e-10)):
                continue
            frc = max(f(1e-6), min(f(f(rain[j, i] / f(dt)) / clw), f(.005)))
            for k in range(nz - 2):
                ix = (k, j, i)
                old = out[r, k, j, i]
                if old > f(1e-16) and qc[ix] > 0:
                    factor = max(f(0), f(f(f(frc * rho[ix]) * dz[ix]) * w[ix]))
                    dvar = f(f(f(a * factor) / f(f(1) + factor)) * old)
                    new = max(f(1e-16), f(old - dvar))
                    out[r, k, j, i] = new
                    mass = f(f(f(old - new) * dryrho[ix]) * dz[ix])
                    removed[r, j, i] = f(removed[r, j, i] + mass)
    return out, removed
