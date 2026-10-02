"""ctypes calls into the WRF vertical and TKE ABI adapters."""
from __future__ import annotations

import ctypes
import numpy as np
from cases import core, pad3


def _array(value):
    return np.asfortranarray(value, dtype=np.float32)


def _pointer(value):
    return ctypes.c_void_p(value.ctypes.data)


def vertical_reference(library, prepared, deformation, metadata, *,
                       kmv, khv, var, tke=None, doing_tke=False):
    """Invoke all four WRF leaves without changing a routine body."""
    nx, ny, nz = (metadata[k] for k in ("nx", "ny", "nz"))
    shape = (nx+6, nz+1, ny+6)
    tke = np.zeros(shape, dtype=np.float32, order="F") if tke is None else _array(tke)
    inputs = [_array(deformation[k]) for k in ("d13", "d23", "d33", "div")]
    inputs += [tke, _array(kmv), _array(khv), _array(var)]
    inputs += [_array(prepared[k]) for k in ("rho", "rdz", "rdzw", "fnm", "fnp", "dn", "dnw")]
    outputs = [np.zeros(shape, dtype=np.float32, order="F") for _ in range(4)]
    fn = ctypes.CDLL(str(library)).oracle_vertical
    fn.argtypes = [ctypes.c_int]*6 + [ctypes.c_void_p]*(len(inputs)+len(outputs))
    fn.restype = None
    fn(nx, ny, nz, metadata["bx"], metadata["by"], int(doing_tke),
       *map(_pointer, inputs+outputs))
    return {name: core(a, nx, ny, nz, stagger) for name, a, stagger in
            zip(("vertical_u", "vertical_v", "vertical_w", "vertical_s"),
                outputs, ("x", "y", "z", ""))}


def tke_reference(library, prepared, deformation, coefficients, metadata, *,
                  isfflx=0, isotropic=0, c_k=.15, dt=12., cd0=.0013, heat=.24):
    """Export every cumulative term and the independently called WRF RHS."""
    nx, ny, nz = (metadata[k] for k in ("nx", "ny", "nz"))
    shape = (nx+6, nz+1, ny+6)
    inputs = [_array(prepared[k]) for k in ("u", "v", "w")]
    inputs += [_array(deformation[k]) for k in ("d11", "d22", "d33", "d12", "d13", "d23", "div")]
    inputs += [_array(prepared[k]) for k in ("tke", "bn2", "theta", "p", "p8w", "t8w", "z", "rdz", "rdzw", "zx", "zy")]
    inputs += [_array(coefficients[k]) for k in ("kmh", "kmv", "khv")]
    inputs += [_array(prepared[k]) for k in ("qv", "rho", "msft", "mut", "ust", "hfx", "qfx", "dn", "dnw", "fnm", "fnp", "c1h", "c2h")]
    outputs = [np.zeros(shape, dtype=np.float32, order="F") for _ in range(4)]
    floats = [c_k, metadata["dx"], metadata["dy"], dt, cd0, heat,
              metadata["cf1"], metadata["cf2"], metadata["cf3"]]
    fn = ctypes.CDLL(str(library)).oracle_tke_rhs
    fn.argtypes = [ctypes.c_int]*7 + [ctypes.c_float]*9 + [ctypes.c_void_p]*(len(inputs)+len(outputs))
    fn.restype = None
    fn(nx, ny, nz, metadata["bx"], metadata["by"], isfflx, isotropic,
       *floats, *map(_pointer, inputs+outputs))
    return {name: core(a,nx,ny,nz) for name,a in
            zip(("tke_shear", "tke_buoyancy", "tke_dissip", "tke_rhs"), outputs)}


def vertical_driver_reference(library,prepared,metadata,*,km_opt=2,isfflx=0,cd0=.0013,heat=.24):
    """Call vertical_diffusion_2 including all three surface switch arms."""
    nx,ny,nz = (metadata[k] for k in ("nx","ny","nz"))
    shape = (nx+6,nz+1,ny+6)
    inputs = [_array(prepared[k]).copy(order="F") for k in
              ("u","v","thp","theta","tke","moist","d13","d23","d33","div",
               "kmh","kmv","khv","rho","rdz","rdzw","fnm","fnp","dn","dnw","hfx","qfx","ust")]
    outputs = [np.zeros(shape,dtype=np.float32,order="F") for _ in range(5)]
    outputs.append(np.zeros((*shape,4),dtype=np.float32,order="F"))
    fn = ctypes.CDLL(str(library)).oracle_vertical_driver
    fn.argtypes = [ctypes.c_int]*7 + [ctypes.c_float]*2 + [ctypes.c_void_p]*(len(inputs)+len(outputs))
    fn.restype = None
    fn(nx,ny,nz,metadata["bx"],metadata["by"],km_opt,isfflx,cd0,heat,*map(_pointer,inputs+outputs))
    result = {name:core(a,nx,ny,nz,stagger) for name,a,stagger in
              zip(("u","v","w","theta","tke"),outputs[:5],("x","y","z","",""))}
    for slot,name in enumerate(("qv","qc","qi"),1):
        result[name] = core(outputs[5][:,:,:,slot],nx,ny,nz)
        result[name+"_after"] = core(inputs[5][:,:,:,slot],nx,ny,nz)
    result["thp_after"] = core(inputs[2],nx,ny,nz)
    result["hfx_after"] = np.ascontiguousarray(inputs[20][3:3+nx,3:3+ny].T)
    result["qfx_after"] = np.ascontiguousarray(inputs[21][3:3+nx,3:3+ny].T)
    return result
