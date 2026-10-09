"""emu_al.py WRF_DUMP WOOF_DUMP: WRF calc_p_rho_phi (hypsometric 2) at the end of RK stage 1, float32 with
glibc logf (WRF's library), from WRF's own dumped words; which muts reproduces WRF's stage-2 al?"""
import sys, glob, ctypes
import numpy as np
f32 = np.float32
libm = ctypes.CDLL("libm.so.6"); libm.logf.restype = ctypes.c_float; libm.logf.argtypes = [ctypes.c_float]
vlog = np.vectorize(lambda x: f32(libm.logf(float(x))), otypes=[np.float32])
wd, od = sys.argv[1], sys.argv[2]
ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme = map(int, open(f"{wd}/dims.txt").read().split())
def W(rk, it, stage, name, nk, nj, ni):
    raw = open(f"{wd}/s001_rk{rk}_it{it:02d}_{stage}__{name}.bin", "rb").read()
    i0, i1, k0, k1, j0, j1 = np.frombuffer(raw[:24], "<i4")
    a = np.frombuffer(raw[24:], "<f4").reshape((i1 - i0 + 1, k1 - k0 + 1, j1 - j0 + 1), order="F")
    return a[ids - i0: ids - i0 + ni, kds - k0: kds - k0 + nk, jds - j0: jds - j0 + nj].transpose(1, 2, 0)
z = np.load(sorted(glob.glob(f"{od}/*_s001_rk1_it00_b_prep.npz"))[0])
nz, ny, nx = z["p"].shape
mub = z["mub2d"].astype(f32); phb = z["phb"].astype(f32); alb = z["alb"].astype(f32)
c3f, c4f, c3h, c4h = (z[n].astype(f32) for n in ("c3f", "c4f", "c3h", "c4h")); import netCDF4; ptop = f32(netCDF4.Dataset("/work/pverify/combo/localize/cases/p1/wrfinput_d01").variables["P_TOP"][0])
ph = W(1, 0, "h_finish", "ph_2", nz + 1, ny, nx); mu2 = W(1, 0, "h_finish", "mu_2", 1, ny, nx)[0]
muts_ss = W(1, 1, "g_ss", "muts", 1, ny, nx)[0]
al_w = W(2, 0, "b_prep", "al", nz, ny, nx)
if phb.ndim == 1: phb = phb[:, None, None] * np.ones((1, ny, nx), f32)
if alb.ndim == 1: alb = alb[:, None, None] * np.ones((1, ny, nx), f32)
for label, muts in (("WRF grid%muts (acoustic)", muts_ss), ("mub + mu_2 (WOOF)", (mub + mu2).astype(f32))):
    al = np.empty((nz, ny, nx), f32)
    for k in range(nz):
        pfu = ((c3f[k + 1] * muts).astype(f32) + c4f[k + 1]).astype(f32) + ptop
        pfd = ((c3f[k] * muts).astype(f32) + c4f[k]).astype(f32) + ptop
        phm = ((c3h[k] * muts).astype(f32) + c4h[k]).astype(f32) + ptop
        dph = (((ph[k + 1] - ph[k]).astype(f32) + phb[k + 1]).astype(f32) - phb[k]).astype(f32)
        al[k] = ((dph / phm).astype(f32) / vlog((pfd / pfu).astype(f32))).astype(f32) - alb[k]
    print(f"{label:28s}: differs from WRF stage-2 al in {(al.view(np.int32) != al_w.view(np.int32)).sum()} of {al.size}")
print("muts words that differ (acoustic vs mub+mu_2):", int((muts_ss.view(np.int32) != (mub + mu2).astype(f32).view(np.int32)).sum()))
