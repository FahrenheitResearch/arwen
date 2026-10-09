"""cmpphys.py WRF_DUMP WOOF_DUMP: WRF tendf after update_phy_ten (physics only) vs WOOF's coupled physics."""
import sys, glob, numpy as np
wd, od = sys.argv[1], sys.argv[2]
ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme = map(int, open(f"{wd}/dims.txt").read().split())
def W(name, shape):
    raw = open(f"{wd}/s001_rk1_it00_c0_phy__{name}.bin", "rb").read()
    i0, i1, k0, k1, j0, j1 = np.frombuffer(raw[:24], "<i4")
    a = np.frombuffer(raw[24:], "<f4").reshape((i1 - i0 + 1, k1 - k0 + 1, j1 - j0 + 1), order="F")
    nk, nj, ni = shape
    return a[ids - i0: ids - i0 + ni, kds - k0: kds - k0 + nk, jds - j0: jds - j0 + nj].transpose(1, 2, 0)
z = np.load(sorted(glob.glob(f"{od}/*_s001_*c_phys.npz"))[0])
for wn, on in (("ru_tendf", "x_ru"), ("rv_tendf", "x_rv"), ("t_tendf", "x_rtheta")):
    b = z[on]; a = W(wn, b.shape)
    bad = a.view(np.int32) != b.view(np.int32); bad &= ~((a == 0) & (b == 0))
    ks = sorted(set(np.argwhere(bad)[:, 0].tolist()))
    print(f"{wn:9s} differ {int(bad.sum())}/{a.size}", (f"k {ks[0]}..{ks[-1]}, max rel {np.max(np.abs(a[bad]-b[bad])/np.maximum(np.abs(a[bad]),1e-30)):.2e}" if ks else ""))
