#!/usr/bin/env python3
"""The CPU reference arm (``p3_backend = "reference"``) against a Fortran dump.

    python run_reference_arm.py <fixdir> <fortran_out_dir> <case>[,<case>...]

Drives ``gpuwm.core.p3.mp_p3_wrapper_wrf`` (what ``apply_reference`` calls)
over the same fixture bytes the device runner reads, steps counted from 1,
and prints per step and field how many float32 words differ from the dump
and the largest ULP distance.  CPU only.  The 2026-10-07 measurement
(lane report, PHYSICS-VERIFY-2026-10-07/p3-mp50) found this arm is NOT
bit-identical to WRF: its NumPy float32 libm calls are different functions
from the C library's, where the device arms are 0 ULP.
"""
import json, sys, time, numpy as np
from gpuwm.core.p3 import mp_p3_wrapper_wrf, p3_init
IN3D=["th","qv","qc","qr","nr","qi","ni","qir","qib","th_old","qv_old","pii","p","dz","w"]
OUT3D=["th","qv","qc","qr","nr","qi","ni","qir","qib","th_old","qv_old","zdbz","effc","effi","vmi","di","rhopo"]
OUT2D=["rainnc","rainncv","sr","snownc","snowncv"]
fixdir, refdir, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
rt=p3_init()
idx={c["name"]:c for c in json.load(open(fixdir+"/index.json"))["cases"]}
for name in names:
    c=idx[name]; ncol,nk,ns,dt=c["ncol"],c["nk"],c["nsteps"],c["dt"]
    buf=np.fromfile(f"{fixdir}/{name}.bin","<f4"); h={}; o=0
    for n in IN3D: h[n]=buf[o:o+ncol*nk].reshape((nk,ncol)).T.copy(); o+=ncol*nk
    s2=[buf[o+i*ncol:o+(i+1)*ncol].copy() for i in range(5)]
    d={n:np.zeros((ncol,nk),np.float32) for n in ["zdbz","effc","effi","vmi","di","rhopo"]}
    frames=[]
    def snap():
        parts=[(h[n] if n in h else d[n]).T.ravel() for n in OUT3D]+[s for s in s2]
        frames.append(np.concatenate(parts).astype("<f4"))
    want=np.fromfile(f"{refdir}/{name}.ref","<f4")
    per=want.size//(ns+1)
    t0=time.time()
    for it in range(1,ns+1):
        mp_p3_wrapper_wrf(h["th"],h["qv"],h["qc"],h["qr"],h["nr"],h["qi"],h["qir"],h["ni"],h["qib"],h["th_old"],h["qv_old"],h["p"],h["dz"],float(dt),it,*s2,d["zdbz"],d["effc"],d["effi"],d["vmi"],d["di"],d["rhopo"],runtime=rt)
        snap()
    got=np.concatenate(frames); w=want[per:]
    dif=got.view(np.uint32)!=w.view(np.uint32)
    both=np.isfinite(got)&np.isfinite(w)
    ulp=np.abs(got.view(np.int32).astype(np.int64)-w.view(np.int32).astype(np.int64))
    nf=len(OUT3D)*ncol*nk
    for st in range(ns):
      base=st*per
      for i,n in enumerate(OUT3D):
        sl=slice(base+i*ncol*nk, base+(i+1)*ncol*nk); dd=int(dif[sl].sum())
        if dd: print("  step",st+1,n,dd,"max ulp",int(ulp[sl][both[sl]].max()) if both[sl].any() else -1)
      for i,n in enumerate(OUT2D):
        sl=slice(base+nf+i*ncol, base+nf+(i+1)*ncol); dd=int(dif[sl].sum())
        if dd: print("  step",st+1,n,dd)
    print(f"{name}: differing {int(dif.sum())} of {got.size}, max ulp {int(ulp[both].max())}, {time.time()-t0:.0f}s", flush=True)
