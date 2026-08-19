"""The one diverging exp input: is the exp quadrant also contracted?"""
import ctypes
import sys
from pathlib import Path

import numpy as np

u = ctypes.CDLL("ucrtbase")
u.fmaf.restype = ctypes.c_float
u.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]


def b(bits):
    return np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0]


F = np.float32
FMA = lambda a, x, c: F(u.fmaf(float(a), float(x), float(c)))

MAGIC = b(0x4B400000)
LOG2E = b(0x3FB8AA3B)
C1, C2 = b(0xBF317200), b(0xB5BFBE8E)
P = [b(0x3F800000), b(0x3F39CBD5), b(0x3E7D4C58), b(0x3D517D8C),
     b(0x3BDD7159), b(0x3A053DD8)]
Q = [b(0x3F800000), b(0xBE8C6857), b(0x3CB0E832)]

d = Path(sys.argv[1])
exp_in = np.frombuffer((d / "exp_in.f32").read_bytes(), dtype=np.float32)
exp_np = np.frombuffer((d / "exp_out.f32").read_bytes(), dtype=np.float32)


def np_exp_variant(x, fused_quadrant):
    if fused_quadrant:
        q = FMA(x, LOG2E, MAGIC) - MAGIC
    else:
        q = F(F(x * LOG2E) + MAGIC) - MAGIC
    xr = FMA(q, C1, x)
    xr = FMA(q, C2, xr)
    num = FMA(P[5], xr, P[4])
    num = FMA(num, xr, P[3])
    num = FMA(num, xr, P[2])
    num = FMA(num, xr, P[1])
    num = FMA(num, xr, P[0])
    den = FMA(Q[2], xr, Q[1])
    den = FMA(den, xr, Q[0])
    poly = F(num / den)
    # scalef (normal path)
    bits = np.uint32(poly.view(np.uint32) + np.uint32(np.int32(int(q)) << 23))
    return np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0]


i = 104439
x = exp_in[i]
print("x =", repr(x), format(int(x.view(np.uint32)), "08x"))
print("numpy  ", format(int(exp_np[i].view(np.uint32)), "08x"))
print("unfused", format(int(np_exp_variant(x, False).view(np.uint32)), "08x"))
print("fused  ", format(int(np_exp_variant(x, True).view(np.uint32)), "08x"))

for fused in (False, True):
    bad = 0
    for k, v in enumerate(exp_in):
        if not (-103.97208404541015625 < float(v) < 88.72283935546875):
            continue
        if np_exp_variant(v, fused).view(np.uint32) != \
                exp_np[k].view(np.uint32):
            bad += 1
    print(f"fused={fused}: exp mismatches {bad}")
