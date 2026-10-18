# Does WOOF's log10 word (k*log10(2) hi/lo split over WOOF's natural log)
# reproduce the box libm's log10f bit for bit?  logf here is the box libm
# logf, which gfk_log already matches.
import ctypes, numpy as np
m = ctypes.CDLL("libm.so.6")
m.log10f.restype = ctypes.c_float; m.log10f.argtypes = [ctypes.c_float]
m.logf.restype = ctypes.c_float; m.logf.argtypes = [ctypes.c_float]
f = np.float32
HI = np.array([0x3e9a2080], np.uint32).view(np.float32)[0]
LO = np.array([0x355427db], np.uint32).view(np.float32)[0]
IV = np.array([0x3ede5bd9], np.uint32).view(np.float32)[0]
def mine(x):
    hx = int(np.array([x], np.float32).view(np.uint32)[0]); k = 0
    if hx < 0x00800000:
        k = -25; x = f(x * f(33554432.0)); hx = int(np.array([x], np.float32).view(np.uint32)[0])
    k += (hx >> 23) - 127
    i = 1 if k < 0 else 0
    hx = (hx & 0x7fffff) | ((0x7f - i) << 23)
    y = f(k + i)
    xm = np.array([hx], np.uint32).view(np.float32)[0]
    z = f(f(y * LO) + f(IV * f(m.logf(float(xm)))))
    return f(z + f(y * HI))
rng = np.random.default_rng(1)
xs = np.concatenate([f(10.0) ** rng.uniform(-30, 30, 200000).astype(np.float32),
                     rng.uniform(0.5, 2.0, 100000).astype(np.float32),
                     np.array([1e-40, 1e-3, 0.9999999, 1.0000001, 10, 100, 1e6], np.float32)])
bad = 0
for x in xs:
    a = f(m.log10f(float(x))); b = mine(f(x))
    if np.array([a], np.float32).view(np.uint32)[0] != np.array([b], np.float32).view(np.uint32)[0]:
        bad += 1
        if bad < 5: print("diff", x, a, b)
print("checked", len(xs), "mismatches", bad)
