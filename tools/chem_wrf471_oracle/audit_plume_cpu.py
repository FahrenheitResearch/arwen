"""Audit every observed full-solver case with the float32 CPU transcription."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from gpuwm.verify import chem_plumerise_ref
from gpuwm.verify.chem_oracle import load,ulp_table
from gpuwm.verify.chem_plumerise_ref import reference_column


def accelerated_namespace():
    """The reference's arithmetic, JIT-compiled for the full-case audit.

    Numba runs the same scalar statements without fastmath. Linux glibc
    expf/powf are bound directly, matching gfortran's scalar calls. A
    maintainer instrument: no forecast, test or CUDA launcher reaches it, and
    the shipped reference needs only NumPy.
    """
    import ctypes
    import types
    import numba
    libm = ctypes.CDLL('libm.so.6')
    exp = libm.expf; exp.argtypes=[ctypes.c_float]; exp.restype=ctypes.c_float
    power = libm.powf; power.argtypes=[ctypes.c_float,ctypes.c_float]; power.restype=ctypes.c_float
    ns=dict(vars(chem_plumerise_ref)); ns['fexp']=exp; ns['fpow']=power
    ns['fsqrt']=numba.njit(lambda x: np.float32(np.sqrt(np.float32(x))),fastmath=False,error_model='numpy')
    # Clone functions so their global calls resolve to the accelerated copies.
    names=['word','fadd','fsub','fmul','fdiv','fabs','fmin','fmax','fsum','fpowi']
    names += [name for name in ns if name.startswith(('wrf_','gsl_')) and callable(ns[name])]
    for name in names:
        f=ns[name]
        ns[name]=numba.njit(types.FunctionType(f.__code__,ns,name,f.__defaults__),fastmath=False,error_model='numpy')
    ns['_libm_keepalive']=libm
    return ns


p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('output'); p.add_argument('--interpreted',action='store_true'); p.add_argument('--case')
a=p.parse_args(); root=Path(a.root); report={}
namespace=None if a.interpreted else accelerated_namespace()
for arm,name in [('landuse','plumerise_wrfchem'),('frp','plumerise_frp_gsl')]:
    cases=load(root/name); totals={}; failures=[]; start=time.monotonic()
    for case,data in cases.items():
        if a.case and case!=a.case: continue
        actual=reference_column(data,arm,namespace=namespace)
        for key,value in actual.items():
            d=ulp_table(value,data[key]); t=totals.setdefault(key,dict(max_ulp=0,n_nonzero=0,n=0))
            t['max_ulp']=max(t['max_ulp'],d['max_ulp']); t['n_nonzero']+=d['n_nonzero']; t['n']+=d['n']
            bitwise=np.array_equal(np.asarray(value).view(np.uint32),np.asarray(data[key]).view(np.uint32))
            if d['n_nonzero'] or not bitwise:
                failures.append(dict(case=case,output=key,stats=d))
    report[arm]=dict(seconds=time.monotonic()-start,ulp=totals,failures=failures)
    print(arm,json.dumps(report[arm]),flush=True)
Path(a.output).write_text(json.dumps(report,indent=2))
if any(v['failures'] for v in report.values()): raise SystemExit(1)
