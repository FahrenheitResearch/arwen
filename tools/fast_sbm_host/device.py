"""CUDA verification driver for the shared FAST_SBM source.

The tile launch preserves WRF cross-column carries for oracle replay. The
column launch uses the corrected preset and frozen neighbor inputs.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import cupy as cp
from . import host as H

EXTRA_TABLES = ("xl_mg", "xs_mg", "xg_mg", "xh_mg", "ima", "chucm",
                "ecoalmassm", "prob", "gain_var_new", "dropradii",
                "fccnr_con", "fccnr_mar", "rccn")


@lru_cache(maxsize=None)
def _module(device):
    # RawModule appends -ftz=true after caller options in CuPy 14.2.
    # Direct NVRTC keeps the scheme's subnormal intermediates in a bare run.
    from gpuwm import nvrtc_ptx_cache as compiler
    kernel_dir = H.KERNEL_DIR
    options=("--std=c++17", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true", f"-I{kernel_dir}")
    with cp.cuda.Device(device):
        ptx,_=compiler.compile_using_nvrtc((kernel_dir/"fast_sbm.cu").read_text(),
                                         options,None,"fast_sbm.cu")
        module=cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx,str) else ptx)
    return module

class Device:
    def __init__(self, host):
        self.precision = host.precision
        self.dtype = host.dtype
        self.flags = {k: int(v) for k, v in zip(H.FLAG_FIELDS, host._flags)}
        self._tables = [cp.asarray(x) for x in host._tables]
        derived = host.derived()
        for name in EXTRA_TABLES:
            key = ("ecoal_own" if self.flags["fall_speed_own_bin"] else "ecoal_wrf") if name == "ecoalmassm" else name
            dtype = np.int32 if name == "ima" else (self.dtype if name in ("dropradii", "rccn") else np.float64)
            self._tables.append(cp.asarray(derived[key], dtype=dtype))
        self.tp = cp.asarray([x.data.ptr for x in self._tables], dtype=np.uint64)
        self.ikr = np.int32(derived["ikr_spon_break"][0])
        self.ff = cp.asarray(host._flags)
        self.module = _module(cp.cuda.runtime.getDevice())

    def call(self, step, dt, dx, dy, dims, state, *, parallel=False):
        if parallel and (self.flags["dtcond_carry"] or not all(self.flags[k] for k in
                  ("rainnc_own_column", "dsupice_w_own_point", "dsupice_wind_own_point"))):
            raise ValueError("parallel FAST_SBM requires column-local precipitation, condensation clock and ice forcing; WRF cross-column carries require the tile launch")
        d = np.asarray(dims, dtype=np.int32)
        n = int((d[7]-d[6]+1)*(d[11]-d[10]+1)*(d[9]-d[8]+1))
        nz = int(d[17]-d[16]+1)
        nc = int((d[13]-d[12]+1)*(d[15]-d[14]+1))
        arrays = []
        for name in H.STATE_FIELDS:
            a = state[name]
            if a.dtype != self.dtype or not a.flags.f_contiguous:
                raise TypeError(f"{name} must be Fortran-contiguous {self.dtype}")
            arrays.append(cp.array(a, order="F"))
        ap = cp.asarray([a.data.ptr for a in arrays], dtype=np.uint64)
        work = cp.zeros(7*n+2*nc*(nz+2), dtype=self.dtype)
        kz = cp.zeros(nc, dtype=np.int32)
        tally = cp.zeros((nc,17), dtype=np.float64)
        status = cp.zeros(nc, dtype=np.int32)
        args = (np.int32(step), self.dtype(dt), self.dtype(dx), self.dtype(dy),
                cp.asarray(d), self.tp, self.ikr, self.ff, ap, work, kz, tally, status)
        if parallel:
            halo_columns=(min(int(d[1])-1,int(d[13])+1)-max(int(d[12])-1,1)+1)*(min(int(d[3])-1,int(d[15])+1)-max(int(d[14])-1,1)+1)
            self.module.get_function(f"fsbm_grid_fields_{self.precision}")((halo_columns,),(1,),args)
            self.module.get_function(f"fsbm_prepare_columns_{self.precision}")((nc,),(1,),args)
            self.module.get_function(f"fsbm_columns_{self.precision}")((nc,),(1,),args)
        else:
            self.module.get_function(f"fsbm_tile_{self.precision}")((1,),(1,),args)
        for name, a in zip(H.STATE_FIELDS, arrays):
            state[name][...] = cp.asnumpy(a)
        return cp.asnumpy(status), cp.asnumpy(tally)
