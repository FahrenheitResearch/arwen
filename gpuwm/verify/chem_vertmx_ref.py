"""Independent float32 WRF-Chem vertical-mixing and deposition reference."""
from __future__ import annotations
import numpy as np
F = np.float32
MWDRY = F(28.966)

def _solve(phi, ekm, dryrho, zw, z, dt, vd):
    """module_vertmx_wrf.F:6-198, operation order of REAL(4) at -O0."""
    n=len(phi)
    a=np.zeros(n,dtype=np.float32)
    b=np.empty(n,dtype=np.float32)
    for k in range(n):
        b[k]=F(F(1)/F(dryrho[k]*F(zw[k+1]-zw[k])))
    for k in range(1,n):
        a[k]=F(F(F(.5)*F(dryrho[k]+dryrho[k-1]))/F(z[k]-z[k-1]))
    l1=np.zeros(n,dtype=np.float32); l2=l1.copy(); l3=l1.copy(); rhs=l1.copy()
    for k in range(n):
        a1=F(a[k]*ekm[k]) if k else F(0)
        a2=F(a[k+1]*ekm[k+1]) if k<n-1 else F(0)
        inv=F(F(1)/F(dt*b[k]))
        loss=F(F(vd*dryrho[k])+a2) if k==0 else (a1 if k==n-1 else F(a1+a2))
        center=F(F(inv-F(F(.25)*loss))*phi[k])
        if k==0:
            gain=F(F(.25)*F(a2*phi[k+1]))
        elif k==n-1:
            gain=F(F(.25)*F(a1*phi[k-1]))
        else:
            gain=F(F(.25)*F(F(a1*phi[k-1])+F(a2*phi[k+1])))
        rhs[k]=F(center+gain)
        l1[k]=F(-F(.75)*a1) if k else F(0)
        l2[k]=F(inv+F(F(.75)*loss))
        l3[k]=F(-F(.75)*a2) if k<n-1 else F(0)
    q=np.empty(n,dtype=np.float32)
    q[0]=F(-l3[0]/l2[0]);rhs[0]=F(rhs[0]/l2[0])
    for k in range(1,n):
        p=F(F(1)/F(l2[k]+F(l1[k]*q[k-1])))
        q[k]=F(-l3[k]*p)
        rhs[k]=F(F(rhs[k]-F(l1[k]*rhs[k-1]))*p)
    for k in range(n-2,-1,-1):
        rhs[k]=F(rhs[k]+F(q[k]*rhs[k+1]))
    return rhs


def vertmx_reference(chem, alt, z_at_w, z, dz8w, exch_h, vd, dt, *,
                     phase='aerosol', floor=1e-16, anth_co_kts=None,
                     fire_co_k1=None, anth_pm25_pair=None, anth_pm25=None,
                     sf_urban_physics=0):
    """Return output, mixed, ekmfull, old, new, ddmassn on CPU.

    Optional source fields represent Registry presence, not species names.
    The PM pair and single-bin PM gates are separate WRF branches
    (dry_dep_driver.F:709-717). Both nest inside the CO-presence gate.
    """
    chem,alt,z_at_w,z,dz8w,exch_h=(np.asarray(x,dtype=np.float32) for x in
                                  (chem,alt,z_at_w,z,dz8w,exch_h))
    if chem.ndim!=3 or chem.shape[0]<11:
        raise ValueError('vertmx needs at least 11 mass levels: WRF mixing floor writes kts+10')
    nz,ny,nx=chem.shape
    if any(x.shape!=chem.shape for x in (alt,z,dz8w,exch_h)) or z_at_w.shape!=(nz+1,ny,nx):
        raise ValueError('vertmx mass/w shapes disagree: diffusion would read the wrong interface')
    if phase not in ('gas','aerosol'):
        raise ValueError('vertmx phase must give gas molar or aerosol mass bookkeeping')
    dt=F(dt);floor=F(floor)
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('vertmx dt must be positive and finite: the solver divides by dt')
    def plane(x): return None if x is None else np.broadcast_to(np.asarray(x,dtype=np.float32),(ny,nx))
    anth=plane(anth_co_kts);fire=plane(fire_co_k1);pm=plane(anth_pm25)
    pair=None if anth_pm25_pair is None else tuple(plane(x) for x in anth_pm25_pair)
    vd=plane(vd)
    result={'output':chem.copy(),'mixed':np.empty_like(chem),
            'ekmfull':np.empty_like(z_at_w),
            'old':np.zeros((ny,nx),dtype=np.float32),
            'new':np.zeros((ny,nx),dtype=np.float32),
            'ddmassn':np.zeros((ny,nx),dtype=np.float32)}
    threshold=F(F(8.19e-4)*F(200))
    for j in range(ny):
        for i in range(nx):
            ek=np.empty(nz+1,dtype=np.float32)
            ek[:nz]=np.maximum(F(1e-6),exch_h[:,j,i]);ek[0]=0;ek[nz]=0
            if anth is not None and sf_urban_physics==0:
                if anth[j,i]>F(0): ek[1:11]=np.maximum(ek[1:11],F(1))
                raised=(anth[j,i]>F(200) or
                        (pair is not None and F(pair[0][j,i]+pair[1][j,i])>threshold) or
                        (pm is not None and pm[j,i]>threshold))
                if raised: ek[1:nz//2]=np.maximum(ek[1:nz//2],F(2))
            if fire is not None and fire[j,i]>F(0):
                ek[1:nz//2]=np.maximum(ek[1:nz//2],F(2))
            density=F(1)/alt[:,j,i]
            zw=z_at_w[:,j,i]-z_at_w[0,j,i]
            zz=z[:,j,i]-z_at_w[0,j,i]
            mixed=_solve(np.maximum(floor,chem[:,j,i]),ek,density,zw,zz,dt,vd[j,i])
            old=new=F(0)
            for k in range(nz-1):
                fac=(F(F(F(F(F(1e-6)*density[k])*F(1))/F(MWDRY*F(1e-3)))*dz8w[k,j,i])
                     if phase=='gas' else F(density[k]*dz8w[k,j,i]))
                old=F(old+F(max(floor,chem[k,j,i])*fac))
                new=F(new+F(max(floor,mixed[k])*fac))
            result['old'][j,i]=old;result['new'][j,i]=new
            result['ddmassn'][j,i]=max(F(0),F(old-new))
            result['output'][:-1,j,i]=np.maximum(floor,mixed[:-1])
            result['mixed'][:,j,i]=mixed;result['ekmfull'][:,j,i]=ek
    return result
