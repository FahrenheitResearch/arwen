"""WRF v4.7.1 monotonic final-stage scalar transport.

``dyn_em/module_advect_em.F:9495-10560`` supplies the flux expressions,
donor extrema, two-sided limiter and additive tendency order. Arrays use
``(k,j,i)`` and staggered face shapes, as moist.py's PD launchers do.
Full-domain periodic or degraded axes are supported. ``open_x/open_y``
select degradation; ``boundary='open'`` also adds WRF radiation tendencies.
Specified rings are computed here and excluded by the caller's RK update.

The first launcher returns workspace for the second: mono needs ``mut``
and extrema, which PD's six flux arrays do not carry. ``mu_old`` is total
time-t column mass unless ``mub`` is supplied, in which case it is WRF's
perturbation mass. Optional ``rw_implicit`` is WRF ``wwI``. The isotropic
``msft`` convention can be extended with ``msfty`` for anisotropic maps.
Source folding belongs to the caller, before both launchers, exactly as
``module_em.F:1803-1916``. Extrema constrain the source-folded time level,
not the pre-source field. No species identifiers enter this module.

The NumPy reference uses float32 operations in WRF expression order and
imports no device code. It is an oracle reference, not an ingest path.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

F = np.float32


def _check(shape, h_order, v_order, open_x, open_y, boundary):
    nz, ny, nx = shape
    if h_order != 5:
        raise ValueError("monotonic h_order must be 5: WRF's mono routine "
                         "has no other horizontal flux arm")
    if v_order not in (3, 5):
        raise ValueError("monotonic v_order must be 3 or 5: other vertical "
                         "stencils are absent from WRF's mono routine")
    if nz < (4 if v_order == 5 else 2):
        raise ValueError("monotonic vertical stencil would read outside the column")
    if (open_x and nx < 7) or (open_y and ny < 7):
        raise ValueError("degraded monotonic horizontal axes need at least 7 cells "
                         "to keep boundary degradation bands from overlapping")
    if boundary not in ("specified", "open"):
        raise ValueError("boundary must be specified or open: an unknown mode "
                         "would omit radiation or limit the wrong ring")


def _flux3(a, b, c, d, vel):
    # module_advect_em.F:9594-9599, constants rounded before multiplication.
    centered = F(7/12)*(c+b) - F(1/12)*(d+a)
    return centered + np.copysign(F(1), vel)*F(1/12)*((d-a)-F(3)*(c-b))


def _flux5(a, b, c, d, e, f, vel):
    # module_advect_em.F:9601-9608; retain each statement-function association.
    centered = F(37/60)*(d+c) - F(2/15)*(e+b) + F(1/60)*(f+a)
    return centered - np.copysign(F(1), vel)*F(1/60)*((f-a)-F(5)*(e-b)+F(10)*(d-c))


def _upwind(a, b, vel):
    sign = np.copysign(F(1), vel)
    return F(0.5)*(F(1)+sign)*a + F(0.5)*(F(1)-sign)*b


def np_mono_fluxes(q, q0, ru, rv, rw, mut, coord, dx, dy, dt,
                   msft=None, has_msf=None, open_x=False, open_y=False,
                   *, h_order=5, v_order=3, boundary="specified"):
    """Six float32 low/correction face arrays, in PD positional order."""
    q, q0 = np.asarray(q, dtype=F), np.asarray(q0, dtype=F)
    _check(q.shape, h_order, v_order, open_x, open_y, boundary)
    nz, ny, nx = q.shape
    fluxes = []
    for axis, velocity, degraded in ((2, ru, open_x), (1, rv, open_y)):
        n = q.shape[axis]
        face = np.arange(n+1)
        sample = lambda a, offset: np.take(a, (face+offset) % n, axis=axis)
        vel = np.asarray(velocity, dtype=F)
        low = vel*_upwind(sample(q0, -1), sample(q0, 0), vel)
        high = vel*_flux5(*(sample(q, d) for d in (-3,-2,-1,0,1,2)), vel)
        if degraded:
            for f in (1, n-1):
                s = [slice(None)]*3; s[axis] = f; s = tuple(s)
                high[s] = F(0.5)*vel[s]*(sample(q, 0)[s]+sample(q, -1)[s])
            third = vel*_flux3(*(sample(q, d) for d in (-2,-1,0,1)), vel)
            for f in (2, n-2):
                s = [slice(None)]*3; s[axis] = f; s = tuple(s)
                high[s] = third[s]
            for f in (0, n):
                s = [slice(None)]*3; s[axis] = f; s = tuple(s)
                high[s] = F(0); low[s] = F(0)
        fluxes.extend((low, high-low))
    low = np.zeros((nz+1,ny,nx), dtype=F)
    high = np.zeros_like(low)
    rw = np.asarray(rw, dtype=F)
    for k in range(1,nz):
        vel = rw[k]
        low[k] = vel*_upwind(q0[k-1], q0[k], -vel)
        if k in (1,nz-1):
            high[k] = vel*(F(coord.fnm[k])*q[k]+F(coord.fnp[k])*q[k-1])
        elif v_order == 5 and 3 <= k <= nz-3:
            high[k] = vel*_flux5(*(q[k+d] for d in (-3,-2,-1,0,1,2)), -vel)
        else:
            high[k] = vel*_flux3(*(q[k+d] for d in (-2,-1,0,1)), -vel)
    return (*fluxes, low, high-low)


def _extrema(q0, ru, rv, rw, open_x, open_y):
    # WRF updates only the receiving cell's extrema at each computed face.
    qmin, qmax = q0.copy(), q0.copy()
    for axis, vel, degraded in ((2,ru,open_x), (1,rv,open_y)):
        n = q0.shape[axis]
        for offset, positive in ((-1,True),(1,False)):
            s = [slice(None)]*3; s[axis] = slice(0,n) if positive else slice(1,n+1)
            flow = vel[tuple(s)] > F(0) if positive else vel[tuple(s)] <= F(0)
            if degraded:
                s = [slice(None)]*3; s[axis] = 0 if positive else n-1
                flow[tuple(s)] = False
            donor = np.roll(q0, -offset, axis=axis)
            qmax = np.where(flow, np.maximum(qmax,donor),qmax)
            qmin = np.where(flow, np.minimum(qmin,donor),qmin)
    flow = -rw[1:-1] > F(0)
    qmax[1:] = np.where(flow,np.maximum(qmax[1:],q0[:-1]),qmax[1:])
    qmin[1:] = np.where(flow,np.minimum(qmin[1:],q0[:-1]),qmin[1:])
    flow = -rw[1:-1] <= F(0)
    qmax[:-1] = np.where(flow,np.maximum(qmax[:-1],q0[1:]),qmax[:-1])
    qmin[:-1] = np.where(flow,np.minimum(qmin[:-1],q0[1:]),qmin[:-1])
    return qmin,qmax


def np_mono_renorm_apply(q0, mu_old, fxl, fxc, fyl, fyc, fzl, fzc,
                         tend, coord, dx, dy, dt, msft=None, has_msf=None,
                         open_x=False, open_y=False, *, q, ru, rv, rw, mut,
                         mub=None, rw_implicit=None, msfty=None,
                         boundary="specified", h_order=5, v_order=3):
    """Return additive tendency and diagnostics without modifying inputs.

    WRF's horizontal diagnostic is defined only on the rectangle excluding
    degraded rings. This reference uses zero elsewhere for inspectability.
    """
    _check(q0.shape,h_order,v_order,open_x,open_y,boundary)
    q,q0,ru,rv,rw = (np.asarray(a,dtype=F) for a in (q,q0,ru,rv,rw))
    fxl,fxc,fyl,fyc,fzl,fzc = (np.asarray(a,dtype=F) for a in (fxl,fxc,fyl,fyc,fzl,fzc))
    nz,ny,nx = q0.shape
    mapped = msft is not None if has_msf is None else bool(has_msf)
    if mapped and msft is None:
        raise ValueError("has_msf requires msft: unit maps would change the limiter budget")
    mx = np.asarray(msft,dtype=F) if mapped else np.ones((ny,nx),dtype=F)
    my = np.asarray(msfty,dtype=F) if mapped and msfty is not None else mx
    c1,c2,rd = (np.asarray(getattr(coord,n),dtype=F)[:,None,None] for n in ("c1h","c2h","rdnw"))
    base = np.zeros((ny,nx),dtype=F) if mub is None else np.asarray(mub,dtype=F)
    implicit = np.zeros((nz+1,ny,nx),dtype=F) if rw_implicit is None else np.asarray(rw_implicit,dtype=F)
    rdx,rdy,dt = F(1/dx),F(1/dy),F(dt)
    qmin,qmax = _extrema(q0,ru,rv,rw,open_x,open_y)
    ieva = (c1*np.asarray(mut,dtype=F)+c2)+dt*my*rd*(implicit[1:]-implicit[:-1])
    ph = ((c1*base+c2)+(c1*np.asarray(mu_old,dtype=F)))*q0 - dt*(mx*my*(rdx*(fxl[:,:,1:]-fxl[:,:,:-1])+rdy*(fyl[:,1:]-fyl[:,:-1]))+my*rd*(fzl[1:]-fzl[:-1]))
    fi = -dt*((mx*my)*(rdx*(np.minimum(F(0),fxc[:,:,1:])-np.maximum(F(0),fxc[:,:,:-1]))+rdy*(np.minimum(F(0),fyc[:,1:])-np.maximum(F(0),fyc[:,:-1])))+my*rd*(np.maximum(F(0),fzc[1:])-np.minimum(F(0),fzc[:-1])))
    fo = dt*((mx*my)*(rdx*(np.maximum(F(0),fxc[:,:,1:])-np.minimum(F(0),fxc[:,:,:-1]))+rdy*(np.maximum(F(0),fyc[:,1:])-np.minimum(F(0),fyc[:,:-1])))+my*rd*(np.minimum(F(0),fzc[1:])-np.maximum(F(0),fzc[:-1])))
    hi,lo = ieva*qmax-ph, ph-ieva*qmin
    # Only compute the quotient selected by WRF, avoiding invalid divisions.
    si,so = np.ones_like(q0),np.ones_like(q0)
    mask = fi>hi
    si[mask] = np.maximum(F(0),hi[mask]/(fi[mask]+F(1e-20)))
    mask = fo>lo
    so[mask] = np.maximum(F(0),lo[mask]/(fo[mask]+F(1e-20)))
    if open_x: si[:,:,(0,-1)]=F(1); so[:,:,(0,-1)]=F(1)
    if open_y: si[:,(0,-1),:]=F(1); so[:,(0,-1),:]=F(1)
    limited = []
    for axis,fc in ((2,fxc),(1,fyc)):
        n = q0.shape[axis]; faces = np.arange(n+1)
        take = lambda a,d: np.take(a,(faces+d)%n,axis=axis)
        scale = np.where(fc>F(0),np.minimum(take(si,0),take(so,-1)),np.minimum(take(so,0),take(si,-1)))
        corr = scale*fc
        # WRF only limits transverse faces inside the limiter rectangle.
        if axis==2 and open_y: corr[:,(0,-1),:]=fc[:,(0,-1),:]
        if axis==1 and open_x: corr[:,:,(0,-1)]=fc[:,:,(0,-1)]
        limited.append(corr)
    zc = fzc.copy()
    zc[1:-1] = np.where(fzc[1:-1]<F(0),np.minimum(si[1:],so[:-1]),np.minimum(so[1:],si[:-1]))*fzc[1:-1]
    if open_x: zc[:,:,(0,-1)]=fzc[:,:,(0,-1)]
    if open_y: zc[:,(0,-1),:]=fzc[:,(0,-1),:]
    xc,yc = limited
    result = np.asarray(tend,dtype=F).copy()
    if boundary=="open":
        if open_x:
            for i,f,sgn in ((0,0,-1),(nx-1,nx-1,1)):
                ub = np.minimum(F(0.5)*(ru[:,:,f]+ru[:,:,f+1]),F(0)) if sgn<0 else np.maximum(F(0.5)*(ru[:,:,f]+ru[:,:,f+1]),F(0))
                diff = q0[:,:,1]-q0[:,:,0] if sgn<0 else q0[:,:,-1]-q0[:,:,-2]
                result[:,:,i] = result[:,:,i]-rdx*(ub*diff+q[:,:,i]*(ru[:,:,f+1]-ru[:,:,f]))
        if open_y:
            for j,f,sgn in ((0,0,-1),(ny-1,ny-1,1)):
                vb = np.minimum(F(0.5)*(rv[:,f]+rv[:,f+1]),F(0)) if sgn<0 else np.maximum(F(0.5)*(rv[:,f]+rv[:,f+1]),F(0))
                diff = q0[:,1]-q0[:,0] if sgn<0 else q0[:,-1]-q0[:,-2]
                result[:,j] = result[:,j]-rdy*(vb*diff+q[:,j]*(rv[:,f+1]-rv[:,f]))
    dz = rd*((zc[1:]-zc[:-1])+fzl[1:]-fzl[:-1])
    result = result-dz
    zt = F(0)-dz
    dxterm = mx*(rdx*((xc[:,:,1:]-xc[:,:,:-1])+fxl[:,:,1:]-fxl[:,:,:-1]))
    dyterm = mx*(rdy*((yc[:,1:]-yc[:,:-1])+fyl[:,1:]-fyl[:,:-1]))
    xs = slice(1,-1) if open_x else slice(None)
    ys = slice(1,-1) if open_y else slice(None)
    result[:,:,xs] = result[:,:,xs]-dxterm[:,:,xs]
    result[:,ys] = result[:,ys]-dyterm[:,ys]
    ht = np.zeros_like(q0)
    ht[:,ys,xs] = (F(0)-dxterm[:,ys,xs])-dyterm[:,ys,xs]
    return {"tendency":result,"h_tendency":ht,"z_tendency":zt,
            "scale_in":si,"scale_out":so,"qmin":qmin,"qmax":qmax}


def np_advect_mono(q, q0, ru, rv, rw, mut, mu_old, coord, dx, dy, dt,
                    tend=None, msft=None, has_msf=None, open_x=False,
                    open_y=False, *, mub=None, rw_implicit=None, msfty=None,
                    boundary="specified", h_order=5, v_order=3):
    """Float32 CPU oracle reference for one full-domain final stage."""
    q,q0,ru,rv,rw = (np.asarray(a,dtype=F) for a in (q,q0,ru,rv,rw))
    fl = np_mono_fluxes(q,q0,ru,rv,rw,mut,coord,dx,dy,dt,msft,has_msf,
                       open_x,open_y,h_order=h_order,v_order=v_order,boundary=boundary)
    if tend is None: tend=np.zeros_like(q0)
    return np_mono_renorm_apply(q0,mu_old,*fl,tend,coord,dx,dy,dt,msft,has_msf,
        open_x,open_y,q=q,ru=ru,rv=rv,rw=rw,mut=mut,mub=mub,
        rw_implicit=rw_implicit,msfty=msfty,boundary=boundary,h_order=h_order,v_order=v_order)


def _cupy():
    if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
        raise RuntimeError("GPUWM_NO_LOCAL_GPU forbids a local CUDA launch")
    import cupy as cp
    return cp


def _kernel(name):
    # The engine's one kernel loader (gpuwm.core.kernels), so this module is
    # compiled, cached, digest-pinned and frame-recorded like every other
    # kernel the forecast launches.  Every FP operation in the source is an
    # explicit round-to-nearest intrinsic, so no contraction can move a word.
    from gpuwm.core.kernels import get_kernel
    return get_kernel("mono_advection", name)


def _array(cp, a, shape, name):
    if not isinstance(a, cp.ndarray) or a.dtype != F or a.shape != shape or not a.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous device float32 {shape}: "
                         "another layout would misaddress the monotonic stencil")
    if a.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f"{name} belongs to another device: its stencil pointer "
                         "is not valid in the active context")
    return a


def _plane(cp, a, shape, name):
    if np.isscalar(a):
        return cp.full(shape, F(a), dtype=F)
    return _array(cp,a,shape,name)


@dataclass
class MonoWorkspace:
    """One decomposition's retained inputs and limiter scratch.

    Use once; keep inputs and flux buffers unchanged between launchers.
    This storage is per scalar launch, never keyed by a species identifier.
    """
    inputs: tuple
    fluxes: tuple
    scratch: tuple
    shape: tuple
    scalars: tuple
    flags: tuple
    device: int
    stream: int
    mapped: bool
    used: bool = False


def launch_mono_fluxes(q, q0, ru, rv, rw, mut, coord, dx, dy, dt,
                       fxl, fxc, fyl, fyc, fzl, fzc,
                       msft=None, has_msf=None, open_x=False, open_y=False,
                       *, msfty=None, rw_implicit=None, h_order=5, v_order=3,
                       boundary="specified", scratch=None):
    """Fill PD-shaped low/correction buffers and return required workspace.

    Additional scratch comprises extrema, two scales and two diagnostics.
    Unlike PD, low-order fluxes do not use mut, map factors or a CFL clamp.
    mut and maps are retained for the second launcher's receiving budget.
    ``scratch`` is six caller-owned mass-shaped float32 arrays (qmin, qmax,
    scale_in, scale_out, h_tendency, z_tendency) that every launch
    overwrites before reading; without it they are allocated here.
    """
    _check(q.shape,h_order,v_order,open_x,open_y,boundary)
    if not all(np.isfinite(x) and x>0 for x in (dx,dy,dt)):
        raise ValueError("positive finite dx, dy and dt are required to "
                         "prevent undefined Courant and limiter budgets")
    cp = _cupy()
    nz,ny,nx = q.shape
    mass_shape = (ny,nx)
    inputs = tuple(_array(cp,a,s,n) for a,s,n in (
        (q,q.shape,"q"),(q0,q.shape,"q0"),(ru,(nz,ny,nx+1),"ru"),
        (rv,(nz,ny+1,nx),"rv"),(rw,(nz+1,ny,nx),"rw")))
    inputs += (_plane(cp,mut,mass_shape,"mut"),)
    inputs += tuple(_array(cp,cp.asarray(getattr(coord,n),dtype=F),(nz,),n)
                    for n in ("c1h","c2h","rdnw","fnm","fnp"))
    if has_msf is None: has_msf=msft is not None
    if has_msf and msft is None:
        raise ValueError("has_msf requires msft: unit maps would change the limiter budget")
    mx = _array(cp,msft,mass_shape,"msft") if has_msf else cp.ones(mass_shape,dtype=F)
    my = _array(cp,msfty,mass_shape,"msfty") if has_msf and msfty is not None else mx
    wi = cp.zeros((nz+1,ny,nx),dtype=F) if rw_implicit is None else _array(cp,rw_implicit,(nz+1,ny,nx),"rw_implicit")
    inputs += (mx,my,wi)
    fluxes = tuple(_array(cp,a,s,n) for a,s,n in (
        (fxl,(nz,ny,nx+1),"fxl"),(fxc,(nz,ny,nx+1),"fxc"),
        (fyl,(nz,ny+1,nx),"fyl"),(fyc,(nz,ny+1,nx),"fyc"),
        (fzl,(nz+1,ny,nx),"fzl"),(fzc,(nz+1,ny,nx),"fzc")))
    if scratch is None:
        scratch = tuple(cp.empty(q.shape,dtype=F) for _ in range(6))
    else:
        if len(scratch) != 6:
            raise ValueError("mono scratch is six arrays: extrema, two scales "
                             "and two diagnostics; fewer would alias outputs")
        scratch = tuple(_array(cp,a,q.shape,"scratch") for a in scratch)
    scalars = (F(1/dx),F(1/dy),F(dt))
    flags = (np.int32(open_x),np.int32(open_y),np.int32(boundary=="open"),np.int32(v_order))
    work = MonoWorkspace(inputs,fluxes,scratch,q.shape,scalars,flags,
                         cp.cuda.runtime.getDevice(),cp.cuda.get_current_stream().ptr,bool(has_msf))
    # mu0, mub and t are not read by the decomposition kernel; mut stands
    # in for the two mass planes so the launch allocates nothing.
    dummy = inputs[5]
    args = (*inputs,dummy,dummy,*fluxes,*scratch[:4],scratch[4],*scratch[4:],
            *scalars,np.int32(nz),np.int32(ny),np.int32(nx),*flags)
    # Outputs must not overlap inputs or one another; CUDA restrict-style
    # assumptions and neighboring stencil reads require separate storage.
    _disjoint(inputs,(*fluxes,*scratch))
    _kernel("mono_fluxes")(((nx+1+127)//128,ny+1,nz+1),(128,1,1),args)
    return work


def _disjoint(reads, writes):
    def overlap(a,b):
        return a.data.ptr < b.data.ptr+b.nbytes and b.data.ptr < a.data.ptr+a.nbytes
    for i,a in enumerate(writes):
        if any(overlap(a,b) for b in (*reads,*writes[:i])):
            raise ValueError("monotonic outputs overlap inputs or each other: "
                             "a stencil would read partially overwritten values")


def launch_mono_renorm_apply(q0, mu_old, fxl, fxc, fyl, fyc, fzl, fzc,
                             tend, coord, dx, dy, dt,
                             msft=None, has_msf=None, open_x=False, open_y=False,
                             *, workspace=None, mub=None, msfty=None):
    """ADD mono-limited divergence into tend; return device diagnostics.

    The PD positional contract is retained. ``workspace`` from the matching
    flux launch is required because mono's receiving bound uses mut and
    upstream extrema. Coordinate/map arrays are retained from that launch.
    Diagnostics outside the defined WRF horizontal rectangle are zero.
    """
    if not isinstance(workspace,MonoWorkspace) or workspace.used:
        raise ValueError("a fresh matching mono workspace is required: "
                         "missing or reused extrema would limit the wrong stage")
    work=workspace
    if q0 is not work.inputs[1] or any(a is not b for a,b in
        zip((fxl,fxc,fyl,fyc,fzl,fzc),work.fluxes)):
        raise ValueError("workspace does not match q0 and flux buffers: "
                         "the receiving bounds would belong to another scalar")
    if (F(1/dx),F(1/dy),F(dt)) != work.scalars or (int(open_x),int(open_y)) != tuple(work.flags[:2]):
        raise ValueError("workspace spacing, timestep or bounds changed: "
                         "the decomposition and limiter would use different budgets")
    cp=_cupy()
    if cp.cuda.runtime.getDevice()!=work.device:
        raise ValueError("workspace device changed: retained pointers belong to another device")
    if cp.cuda.get_current_stream().ptr!=work.stream:
        raise ValueError("workspace stream changed: the limiter could read fluxes "
                         "before their decomposition completes")
    if has_msf is not None and bool(has_msf)!=work.mapped:
        raise ValueError("workspace map mode changed: the limiter would use a different mass metric")
    if msft is not None and has_msf is not False and msft is not work.inputs[11]:
        raise ValueError("workspace map changed: the limiter would use a different mass metric")
    if msfty is not None and msfty is not work.inputs[12]:
        raise ValueError("workspace msfty changed: the vertical budget would use a different metric")
    for n,a in zip(("c1h","c2h","rdnw","fnm","fnp"),work.inputs[6:11]):
        supplied=getattr(coord,n)
        if isinstance(supplied,cp.ndarray) and supplied is not a:
            raise ValueError("workspace coordinate changed: fluxes and budget would use different layers")
    nz,ny,nx=work.shape
    mu0=_plane(cp,mu_old,(ny,nx),"mu_old")
    base=cp.zeros((ny,nx),dtype=F) if mub is None else _plane(cp,mub,(ny,nx),"mub")
    tend=_array(cp,tend,work.shape,"tend")
    _disjoint((*work.inputs,mu0,base,*work.fluxes,*work.scratch[:2]),
              (*work.scratch[2:],tend))
    args=(*work.inputs,mu0,base,*work.fluxes,*work.scratch[:4],tend,*work.scratch[4:],
          *work.scalars,np.int32(nz),np.int32(ny),np.int32(nx),*work.flags)
    grid=((nx+127)//128,ny,nz)
    _kernel("mono_scales")(grid,(128,1,1),args)
    _kernel("mono_apply")(grid,(128,1,1),args)
    work.used=True
    return {"tendency":tend,"h_tendency":work.scratch[4],"z_tendency":work.scratch[5],
            "qmin":work.scratch[0],"qmax":work.scratch[1],"scale_in":work.scratch[2],"scale_out":work.scratch[3]}
