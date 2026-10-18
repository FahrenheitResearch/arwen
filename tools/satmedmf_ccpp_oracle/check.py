"""Independent input columns and word comparison for the CCPP port.
This module is a verification harness, never a forecast data path.
"""
from __future__ import annotations
import ctypes,hashlib,json
from pathlib import Path
import numpy as np

FIELDS=("du","dv","tdt","rtg","kpbl","hpbl","dkt","dku","sflux")
REGIMES=("convective_land","stable_night","stratocumulus","neutral_windy",
         "very_shallow","coarse_dx","fine_dx","weakly_stable")

def columns(ncolumns=64,nlevels=48,ntrac=5):
    if ncolumns<64 or nlevels<8:
        raise ValueError("qualification requires at least 64 columns and 8 levels")
    n,k=ncolumns,nlevels
    f=lambda a: np.array(a,dtype=np.float64,order="F",copy=True)
    regime=np.arange(n)%len(REGIMES)
    dz=np.where(regime==4,12.0,80.0)+(np.arange(n)%5)*0.75
    zi=f(dz[:,None]*np.arange(k+1)[None,:])
    z=f((zi[:,:-1]+zi[:,1:])*0.5)
    prsi=f(100000.0*np.exp(-zi/8200.0))
    prsl=f((prsi[:,:-1]+prsi[:,1:])*0.5)
    # The pinned CCPP physical constants carry the kind_phys suffix.
    g=9.80665; rd=287.05; cp=1004.6
    theta=f(297.0+0.003*z)
    theta[regime==1]+=np.minimum(z[regime==1],600.0)*0.022
    theta[regime==2]+=np.maximum(z[regime==2]-900.0,0.0)*0.012
    theta[regime==4]+=np.minimum(z[regime==4],60.0)*0.09
    prslk=f((prsl/100000.0)**(rd/cp))
    t=f(theta*prslk)
    u=f(np.broadcast_to(4.0+0.001*z,(n,k)))
    v=f(np.broadcast_to(1.0+0.0002*z,(n,k)))
    u[regime==3]+=12.0
    q=np.zeros((n,k,ntrac),dtype=np.float64,order="F")
    q[:,:,0]=0.011*np.exp(-z/2300.0)
    cloud=(regime[:,None]==2)&(z>650.0)&(z<1000.0)
    q[:,:,1]=np.where(cloud,0.00035,0.0)
    q[:,:,-1]=0.12+0.03*np.exp(-z/300.0)
    swh=np.zeros((n,k),dtype=np.float64,order="F")
    hlw=np.zeros_like(swh,order="F")
    hlw[cloud]=-0.0012
    # psk, rbsoil, zorl[cm], tsea, heat[K m/s], evap[kg/kg m/s],
    # stress[m2/s2], spd1, u10, v10, fm, fh, xmu, garea[m2], zvfun, sigmaf.
    sfc=np.zeros((n,16),dtype=np.float64,order="F")
    sfc[:,0]=1; sfc[:,1]=-0.06
    sfc[regime==1,1]=0.15; sfc[regime==7,1]=0.015
    sfc[:,2]=10; sfc[:,3]=299
    sfc[:,4]=0.04; sfc[regime==1,4]=-0.012
    sfc[regime==3,4]=0; sfc[regime==4,4]=0.004
    sfc[:,5]=2e-5; sfc[regime==1,5]=0
    sfc[:,6]=0.08; sfc[:,7]=np.sqrt(u[:,0]**2+v[:,0]**2)
    sfc[:,8]=u[:,0]; sfc[:,9]=v[:,0]; sfc[:,10:13]=1
    dx=np.where(regime==5,25000.0,np.where(regime==6,250.0,3000.0))
    sfc[:,13]=dx**2; sfc[:,14]=1; sfc[:,15]=0.3
    return dict(u=u,v=v,t=t,q=q,swh=swh,hlw=hlw,prsi=prsi,prsl=prsl,
                prslk=prslk,phii=f(zi*g),phil=f(z*g),sfc=sfc)

class Oracle:
    def __init__(self,library):
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        self.run=self.library.satmedmf_column_oracle
        self.run.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double]+[ctypes.c_void_p]*21
        self.run.restype=None
        self.solver=self.library.satmedmf_solver_oracle
        self.solver.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*8
        self.solver.restype=None

    def __call__(self,inputs,dt=60.0):
        n,k=inputs["u"].shape
        nt=inputs["q"].shape[2]
        shapes={"du":(n,k),"dv":(n,k),"tdt":(n,k),"rtg":(n,k,nt),"kpbl":(n,),
                "hpbl":(n,),"dkt":(n,k),"dku":(n,k),"sflux":(n,4)}
        out={key:np.zeros(shape,dtype=np.int32 if key=="kpbl" else np.float64,order="F")
             for key,shape in shapes.items()}
        arrays=[*inputs.values(),*out.values()]
        for a in arrays:
            if not a.flags.f_contiguous:
                raise ValueError("oracle arrays must be Fortran contiguous")
        self.run(n,k,nt,dt,*[ctypes.c_void_p(a.ctypes.data) for a in arrays])
        return out

    def solve(self,which,cl,cm,cu,r1,r2,nt):
        n,k=cm.shape
        out=[np.zeros((n,k-1),order="F"),np.zeros((n,k),order="F"),
             np.zeros((n,k*nt),order="F")]
        self.solver(n,k,nt,which,*[ctypes.c_void_p(a.ctypes.data) for a in [cl,cm,cu,r1,r2,*out]])
        return dict(zip(("au","a1","a2"),out))

def ordered_words(a):
    bits=np.ascontiguousarray(a).view(np.uint64)
    sign=np.uint64(0x8000000000000000)
    return np.where((bits&sign)!=0,~bits,bits|sign)

def compare(expected,actual,*,strict=True,tolerances=None):
    if set(expected)!=set(actual):
        raise ValueError(f"output fields differ: {set(expected)^set(actual)}")
    result={}
    for name,e in expected.items():
        a=np.asarray(actual[name])
        if a.shape!=e.shape or a.dtype!=e.dtype:
            raise ValueError(f"{name}: shape or dtype mismatch")
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: non-finite output")
        if e.dtype.kind=="f":
            eb=ordered_words(e); ab=ordered_words(a)
            ulp=np.maximum(eb,ab)-np.minimum(eb,ab)
            mismatch=np.ascontiguousarray(e).view(np.uint64)!=np.ascontiguousarray(a).view(np.uint64)
            max_ulp=int(ulp.max(initial=0))
            abs_error=float(np.max(np.abs(e-a),initial=0))
            if strict:
                passed=not bool(mismatch.any())
            else:
                if tolerances is None or name not in tolerances:
                    raise ValueError(f"no declared default tolerance for {name}")
                atol,rtol=tolerances[name]
                passed=bool(np.all(np.abs(e-a)<=atol+rtol*np.abs(e)))
        else:
            mismatch=e!=a; max_ulp=int(np.max(np.abs(e.astype(np.int64)-a),initial=0))
            abs_error=float(max_ulp); passed=not bool(mismatch.any())
        first=np.argwhere(mismatch)
        result[name]={"pass":passed,"mismatches":int(mismatch.sum()),"max_ulp":max_ulp,
                      "max_absolute_error":abs_error,"first":first[0].tolist() if len(first) else None,
                      "expected_sha256":hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest(),
                      "actual_sha256":hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
    return result

def advance(inputs,out,dt):
    # Oracle-only integration. Both candidates receive these same independent inputs.
    updated={key:value.copy(order="F") for key,value in inputs.items()}
    for field,tendency in (("u","du"),("v","dv"),("t","tdt"),("q","rtg")):
        updated[field]+=dt*out[tendency]
    return updated
