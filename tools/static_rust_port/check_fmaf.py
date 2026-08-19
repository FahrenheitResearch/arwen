"""Is UCRT fmaf correctly rounded on the diverging reduction operands?"""
import ctypes
from fractions import Fraction

import numpy as np

u = ctypes.CDLL("ucrtbase")
u.fmaf.restype = ctypes.c_float
u.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]

x = np.float32(66365.36)
q = np.float32(42250.0)
c1 = np.frombuffer(np.uint32(0xBFC90FD8).tobytes(), dtype=np.float32)[0]
c2 = np.frombuffer(np.uint32(0xB4A8885A).tobytes(), dtype=np.float32)[0]
c3 = np.frombuffer(np.uint32(0xA7C234C4).tobytes(), dtype=np.float32)[0]


def exact_fmaf(a, b, c):
    exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    # round-to-nearest-even to f32 via successive approximation
    lo = np.float32(float(exact))  # double rounding risk, refine below
    candidates = {lo, np.nextafter(lo, np.float32(np.inf)),
                  np.nextafter(lo, np.float32(-np.inf))}
    best = min(candidates, key=lambda v: abs(Fraction(float(v)) - exact))
    return np.float32(best)


steps = [(q, c1, x)]
r = None
for a, b, c in steps:
    pass

r1_u = u.fmaf(float(q), float(c1), float(x))
r1_e = exact_fmaf(q, c1, x)
print("step1 ucrt", np.float32(r1_u).view(np.uint32), "exact",
      r1_e.view(np.uint32), "equal", np.float32(r1_u) == r1_e)
r2_u = u.fmaf(float(q), float(c2), float(r1_u))
r2_e = exact_fmaf(q, c2, r1_e)
print("step2 ucrt", np.float32(r2_u).view(np.uint32), "exact",
      r2_e.view(np.uint32), "equal", np.float32(r2_u) == r2_e)
r3_u = u.fmaf(float(q), float(c3), float(r2_u))
r3_e = exact_fmaf(q, c3, r2_e)
print("step3 ucrt", np.float32(r3_u).view(np.uint32), "exact",
      r3_e.view(np.uint32), "equal", np.float32(r3_u) == r3_e)
