"""Does quadrant = fmaf(x, 2/pi, magic) - magic reproduce numpy on the
whole sweep (all four kernels' rint sites)?"""
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

TWO_OVER_PI = b(0x3F22F983)
MAGIC = b(0x4B400000)
C1, C2, C3 = b(0xBFC90FD8), b(0xB4A8885A), b(0xA7C234C4)
S9, S7, S5, S3 = b(0x363E9DDE), b(0xB95035DD), b(0x3C0888CD), b(0xBE2AAAAB)
C8, C6, C4 = b(0x37CC730B), b(0xBAB6036E), b(0x3D2AAA9E)


def sincos(x, is_cos, fused_quadrant):
    if fused_quadrant:
        q = FMA(x, TWO_OVER_PI, MAGIC) - MAGIC
    else:
        q = F(F(x * TWO_OVER_PI) + MAGIC) - MAGIC
    r = FMA(q, C1, x)
    r = FMA(q, C2, r)
    r = FMA(q, C3, r)
    r2 = F(r * r)
    s = FMA(S9, r2, S7)
    s = FMA(s, r2, S5)
    s = FMA(s, r2, S3)
    s = FMA(s, r2, F(0.0))
    s = FMA(s, r, r)
    c = FMA(C8, r2, C6)
    c = FMA(c, r2, C4)
    c = FMA(c, r2, F(-0.5))
    c = FMA(c, r2, F(1.0))
    iq = int(q) + (1 if is_cos else 0)
    out = s if iq & 1 == 0 else c
    if iq & 2 == 2:
        out = F(0.0) - out
    return out


d = Path(sys.argv[1])
trig = np.frombuffer((d / "trig_in.f32").read_bytes(), dtype=np.float32)
sin_np = np.frombuffer((d / "sin_out.f32").read_bytes(), dtype=np.float32)
cos_np = np.frombuffer((d / "cos_out.f32").read_bytes(), dtype=np.float32)

for fused in (False, True):
    bad_sin = bad_cos = 0
    for i, x in enumerate(trig):
        ax = abs(float(x))
        if ax <= 117435.9921875:
            if sincos(x, False, fused).view(np.uint32) != \
                    sin_np[i].view(np.uint32):
                bad_sin += 1
        if ax <= 71476.0625:
            if sincos(x, True, fused).view(np.uint32) != \
                    cos_np[i].view(np.uint32):
                bad_cos += 1
    print(f"fused_quadrant={fused}: sin mismatches {bad_sin}, "
          f"cos mismatches {bad_cos}")
