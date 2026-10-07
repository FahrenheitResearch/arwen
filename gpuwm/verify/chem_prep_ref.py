"""Independent float32 WRF-Chem column-preparation reference."""
from __future__ import annotations
import numpy as np
from gpuwm.core.noahmp_libm import expf, logf, powf
from gpuwm.core.chem_context import CHEM_PREP_FIELDS as NAMES
F = np.float32
RCP = F(F(287) / F(F(7) * F(287) / F(2)))

def chem_prep_reference(p, theta, alt, ph, phb, u, v, qv, fnm, fnp):
    """Float32 CPU reference; inputs follow DomainState, theta is total.

    Transcendentals use the existing glibc scalar reproductions, avoiding
    the platform-dependent NumPy vector libm. All other operations round
    after each binary operation as module_chem_utilities.F:79-177 does.
    """
    p, theta, alt, ph, phb, u, v, qv, fnm, fnp = (
        np.asarray(x, dtype=np.float32) for x in
        (p, theta, alt, ph, phb, u, v, qv, fnm, fnp))
    nz, ny, nx = p.shape
    if nz < 2:
        raise ValueError('chem_prep needs two mass levels for boundary extrapolation')
    out = {name: np.empty((nz + (name in ('z_at_w','p8w','t8w')), ny, nx),
                         dtype=np.float32) for name in NAMES}
    out['p_phy'][...] = p
    out['alt'][...] = alt
    out['dryrho'][...] = F(1) / alt
    out['rho'][...] = out['dryrho'] * (F(1) + qv)
    out['u_phy'][...] = F(.5) * (u[:, :, :-1] + u[:, :, 1:])
    out['v_phy'][...] = F(.5) * (v[:, :-1, :] + v[:, 1:, :])
    out['z_at_w'][...] = (phb + ph) / F(9.81)
    zw = out['z_at_w']
    out['dz8w'][...] = zw[1:] - zw[:-1]
    out['z'][...] = F(.5) * (zw[:-1] + zw[1:])
    for j in range(ny):
        for i in range(nx):
            for k in range(nz):
                t = F(theta[k,j,i] * powf(F(p[k,j,i]/F(100000)), RCP))
                out['t_phy'][k,j,i] = t
                exponent = F(F(F(17.27)*F(t-F(273))) / F(t-F(36)))
                sat = F(F(F(3.80)*expf(exponent)) / F(F(.01)*p[k,j,i]))
                out['rh'][k,j,i] = max(F(.1), min(F(.95), F(qv[k,j,i]/sat)))
            for k in range(1,nz):
                for dest,source in (('p8w','p_phy'),('t8w','t_phy')):
                    a = out[source]
                    out[dest][k,j,i] = F(F(fnm[k]*a[k,j,i])+F(fnp[k]*a[k-1,j,i]))
            for kw,k1,k2 in ((0,0,1),(nz,nz-1,nz-2)):
                z0,z1,z2 = zw[kw,j,i],out['z'][k1,j,i],out['z'][k2,j,i]
                w1 = F(F(z0-z2)/F(z1-z2)); w2 = F(F(1)-w1)
                p1,p2 = p[k1,j,i],p[k2,j,i]
                out['p8w'][kw,j,i] = (expf(F(F(w1*logf(p1))+F(w2*logf(p2))))
                                      if kw==nz else F(F(w1*p1)+F(w2*p2)))
                t1,t2 = out['t_phy'][k1,j,i],out['t_phy'][k2,j,i]
                out['t8w'][kw,j,i] = F(F(w1*t1)+F(w2*t2))
    return out
