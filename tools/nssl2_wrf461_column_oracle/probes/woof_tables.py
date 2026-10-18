# Read WOOF device init tables back and compare with WRF tables.f90 readback
import sys, re, numpy as np, cupy as cp
from gpuwm.core.kernels import load_module, get_kernel
mod = load_module("nssl2_fused_gs")
get_kernel("nssl2_fused_gs", "nssl2_gs_init_tables")((4,), (256,), ())
cp.cuda.Device().synchronize()
g = mod.get_global("g_nssl2_gs_tables")
nf = 9 + 9 + 81 + 81 + 4 + 802 + 802
nb = nf * 4 + 2 * 802 * 8 + 8
raw = np.empty(nb, np.uint8)
cp.cuda.runtime.memcpy(raw.ctypes.data, g.ptr, nb, cp.cuda.runtime.memcpyDeviceToHost)
f = raw[:nf * 4].view(np.float32)
d = raw[nf * 4: nf * 4 + 2 * 802 * 8].view(np.float64)
da0 = f[0:9]; da1 = f[9:18]; dab0 = f[18:99].reshape(9, 9); dab1 = f[99:180].reshape(9, 9)
vent = f[180:184]; cr = f[184:986].reshape(401, 2); qr = f[986:1788].reshape(401, 2)
g1 = d[:802].reshape(401, 2); g4 = d[802:].reshape(401, 2)
wrf = {}
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) >= 2: wrf[p[0]] = p[1]
def chk(name, woofv, key, dbl=False):
    want = wrf[key]
    got = format(np.float64(woofv).view(np.uint64), "016X") if dbl else format(np.float32(woofv).view(np.uint32), "08X")
    return got == want, got, want
bad = 0
for i in range(401):
    for nm, arr, key, dbl in (("g4", g4[i,0], f"g4_{i}", True), ("g1", g1[i,0], f"g1_{i}", True),
                              ("cr", cr[i,0], f"cr_{i}", False), ("qr", qr[i,0], f"qr_{i}", False),
                              ("g4b", g4[i,1], f"g4b_{i}", True), ("g1b", g1[i,1], f"g1b_{i}", True),
                              ("crb", cr[i,1], f"crb_{i}", False), ("qrb", qr[i,1], f"qrb_{i}", False)):
        ok, got, want = chk(nm, arr, key, dbl)
        if not ok:
            bad += 1
            if bad < 25: print(nm, i, "woof", got, "wrf", want, float(arr))
for il in range(3, 9):
    ok, got, want = chk("da0", da0[il], f"da0_{il}")
    if not ok: print("da0", il, got, want); bad += 1
for nm, v in (("c1sw", vent[3]), ("ventrn", vent[1]), ("ventc", vent[2]), ("ventr", vent[0])):
    ok, got, want = chk(nm, v, nm)
    if not ok: print(nm, got, want); bad += 1
print("table mismatches:", bad)
