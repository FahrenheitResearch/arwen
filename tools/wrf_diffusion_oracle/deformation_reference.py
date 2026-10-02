"""Call the compiled WRF metrics, physics prep, tensor and coefficient bodies."""
from __future__ import annotations

import ctypes
import numpy as np
from cases import pad2, pad3


def _invoke(lib, name, args):
    getattr(lib, name)(*[ctypes.c_void_p(x.ctypes.data) if isinstance(x, np.ndarray)
                        else ctypes.c_int(x) if isinstance(x, int)
                        else ctypes.c_float(x) for x in args])


def reference_for_case(library, arrays, meta, *, km_opt=4, isotropic=0):
    """Return WRF-padded inputs and all outputs; core slices use cases.core.

    Shapes and slots follow the real WRF C grid, rather than flattening a
    scalar point into a different routine's argument layout.
    """
    lib = library if isinstance(library, ctypes.CDLL) else ctypes.CDLL(str(library))
    nx,ny,nz,bx,by = [int(meta[k]) for k in ("nx","ny","nz","bx","by")]
    head = [nx,ny,nz,bx,by]
    shape = (nx+6,nz+1,ny+6)
    out = {k:pad3(arrays[k],nx,ny,nz,bx,by) for k in
           ("u","v","w","php","phb","p","alt","thp")}
    out.update({k:pad2(arrays[k],nx,ny,bx,by) for k in ("msfu","msfv","msft","mut")})
    for k in ("dn","dnw","fnm","fnp","fzm","fzp","znw","c1h","c2h","c1f","c2f"):
        # wrfinput omits the physics-specific interior interpolation arrays.
        # Only phy_prep's surface/top extrapolation is consumed by diffusion;
        # those branches do not read FZM/FZP.
        f = np.asarray(arrays.get(k,arrays[{"fzm":"fnm","fzp":"fnp"}.get(k,k)]),dtype=np.float32)
        out[k] = np.asfortranarray(f[np.minimum(np.arange(nz+1),f.size-1)])
    moist = np.zeros((*shape,4),dtype=np.float32,order="F")
    for slot,key in enumerate(("qv","qc","qi"),start=1):
        moist[:,:,:,slot] = pad3(arrays.get(key,np.zeros_like(arrays["alt"])),nx,ny,nz,bx,by)
    out["moist"] = moist
    out["tke"] = pad3(arrays.get("tke",np.full_like(arrays["alt"],.5)),nx,ny,nz,bx,by)
    for key in ("z","rdz","rdzw","zx","zy","rho","theta","temp","p8w","t8w","zw",
                "div","d11","d22","d33","d12","d13","d23","kmh","kmv","khh","khv","bn2"):
        out[key] = np.zeros(shape,dtype=np.float32,order="F")
    def pointers(*keys): return [out[k] for k in keys]
    rdx,rdy = [float(np.float32(1.0/meta[k])) for k in ("dx","dy")]
    cf = [float(meta[k]) for k in ("cf1","cf2","cf3")]
    _invoke(lib,"oracle_metrics",head+pointers("php","phb")+[rdx,rdy]+pointers("z","rdz","rdzw","zx","zy"))
    _invoke(lib,"oracle_phy",head+pointers("u","v","p","alt","php","phb","thp","moist","mut",
              "c1h","c2h","c1f","c2f","dnw","fzm","fzp","znw")+[float(meta.get("p_top",5000.))]+
              pointers("rho","theta","temp","p8w","t8w","z","zw"))
    _invoke(lib,"oracle_deform",head+pointers("u","v","w","msfu","msfv","msft","rdz","rdzw","zx","zy",
              "dn","dnw","fnm","fnp")+[rdx,rdy,*cf]+pointers("div","d11","d22","d33","d12","d13","d23"))
    # WRF phy_bc extends tensors using their own staggering before K is built.
    for key,stag in (("div",0),("d11",0),("d22",0),("d33",0),("d12",4),("d13",5),("d23",6)):
        _invoke(lib,"oracle_bc",head+[stag,out[key]])
    _invoke(lib,"oracle_km",head+[int(km_opt),int(isotropic)]+pointers("theta","temp","p","p8w","t8w","moist","tke",
              "msft","rdz","rdzw","zx","zy","dn","dnw","div","d11","d22","d33","d12","d13","d23")+
              [float(meta["dx"]),float(meta["dy"]),float(meta.get("dt",1.)),*cf,float(meta.get("c_s",.25)),
               float(meta.get("c_k",.15)),float(meta.get("mix_upper_bound",.1)),int(meta.get("seed_on",1))]+
              pointers("kmh","kmv","khh","khv","bn2"))
    for key in ("kmh","kmv","khh","khv","bn2"):
        _invoke(lib,"oracle_bc",head+[0,out[key]])
    return out
