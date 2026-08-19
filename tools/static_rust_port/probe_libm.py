"""Measure, on THIS box, whether numpy's transcendental bit results match
the UCRT libm that Rust std links (the "libm risk" named in
docs/dev/static-rust-port.md).  Lane 1 chooses its Rust evaluation
strategy per function from this table, then the goldens hold it.

f64: numpy ufunc vs CPython math.* (math calls UCRT directly).
f32: numpy ufunc vs UCRT's float functions via ctypes vs
     double-rounding ((f32)f64func((f64)x)).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np

rng = np.random.default_rng(20260817)

# Value pools shaped like the projection math actually sees.
DEG = np.concatenate([
    rng.uniform(-179.9, 179.9, 4000),
    rng.uniform(-90.0, 90.0, 4000),
    rng.uniform(30.0, 45.0, 4000),
    np.array([0.0, -0.0, 45.0, -45.0, 90.0, -90.0, 38.0, 41.0, 71.0]),
])
RAD = DEG * (np.pi / 180.0)
POS = np.abs(rng.uniform(0.01, 3.0, 8000)) + 1e-6
UNIT = rng.uniform(-1.0, 1.0, 8000)

ucrt = ctypes.CDLL("ucrtbase")


def bind(name, restype, argtypes):
    fn = getattr(ucrt, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def compare64(label, np_fn, math_fn, values):
    a = np_fn(np.asarray(values, dtype=np.float64))
    b = np.array([math_fn(float(v)) for v in np.asarray(values, np.float64)])
    n = int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
    print(f"f64 {label:10s} mismatches {n}/{a.size}")


def compare32(label, np_fn, crt_name, math_fn, values):
    x = np.asarray(values, dtype=np.float32)
    a = np_fn(x)
    crt = bind(crt_name, ctypes.c_float, [ctypes.c_float])
    b = np.array([crt(ctypes.c_float(float(v)).value) for v in x],
                 dtype=np.float32)
    c = np.array([np.float32(math_fn(float(v))) for v in x],
                 dtype=np.float32)
    na = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
    nc = int(np.count_nonzero(a.view(np.uint32) != c.view(np.uint32)))
    print(f"f32 {label:10s} vs-ucrt {na}/{a.size}   vs-dblround {nc}/{a.size}")


compare64("sin", np.sin, math.sin, RAD)
compare64("cos", np.cos, math.cos, RAD)
compare64("tan", np.tan, math.tan, RAD)
compare64("atan", np.arctan, math.atan, np.concatenate([UNIT, POS * 10]))
compare64("asin", np.arcsin, math.asin, UNIT)
compare64("acos", np.arccos, math.acos, UNIT)
compare64("log", np.log, math.log, POS)
compare64("log10", np.log10, math.log10, POS)
compare64("exp", np.exp, math.exp, np.concatenate([UNIT * 5, RAD]))
compare64("sqrt", np.sqrt, math.sqrt, POS)

# atan2 / pow need two args
a2 = np.arctan2(UNIT, np.roll(UNIT, 1))
b2 = np.array([math.atan2(float(y), float(x))
               for y, x in zip(UNIT, np.roll(UNIT, 1))])
print("f64 atan2      mismatches",
      int(np.count_nonzero(a2.view(np.uint64) != b2.view(np.uint64))),
      "/", a2.size)
pw = np.power(POS, 0.715)
pw2 = np.array([math.pow(float(v), 0.715) for v in POS])
print("f64 pow        mismatches",
      int(np.count_nonzero(pw.view(np.uint64) != pw2.view(np.uint64))),
      "/", pw.size)
sq = np.power(POS, 2.0)
sq2 = np.array([float(v) * float(v) for v in POS])
sq3 = np.array([math.pow(float(v), 2.0) for v in POS])
print("f64 pow(x,2.0) vs x*x",
      int(np.count_nonzero(sq.view(np.uint64) != sq2.view(np.uint64))),
      " vs pow", int(np.count_nonzero(sq.view(np.uint64) != sq3.view(np.uint64))))
# scalar ** 2.0 on np.float64 (the polar gi2 spelling)
g = np.float64(POS[0])
print("f64 scalar **2.0 == x*x:", (g ** 2.0) == g * g,
      " == pow:", (g ** 2.0) == math.pow(float(g), 2.0))

compare32("sin", np.sin, "sinf", math.sin, RAD)
compare32("cos", np.cos, "cosf", math.cos, RAD)
compare32("tan", np.tan, "tanf", math.tan, RAD)
compare32("atan", np.arctan, "atanf", math.atan, UNIT)
compare32("asin", np.arcsin, "asinf", math.asin, UNIT)
compare32("acos", np.arccos, "acosf", math.acos, UNIT)
compare32("log", np.log, "logf", math.log, POS)
compare32("log10", np.log10, "log10f", math.log10, POS)
compare32("exp", np.exp, "expf", math.exp, UNIT * np.float32(5.0))
compare32("sqrt", np.sqrt, "sqrtf", math.sqrt, POS)

x32 = np.asarray(UNIT, np.float32)
y32 = np.roll(x32, 1)
a32 = np.arctan2(x32, y32)
crt_atan2f = bind("atan2f", ctypes.c_float, [ctypes.c_float, ctypes.c_float])
b32 = np.array([crt_atan2f(float(y), float(x)) for y, x in zip(x32, y32)],
               dtype=np.float32)
c32 = np.array([np.float32(math.atan2(float(y), float(x)))
                for y, x in zip(x32, y32)], dtype=np.float32)
print("f32 atan2      vs-ucrt",
      int(np.count_nonzero(a32.view(np.uint32) != b32.view(np.uint32))),
      "  vs-dblround",
      int(np.count_nonzero(a32.view(np.uint32) != c32.view(np.uint32))),
      "/", a32.size)

p32 = np.asarray(POS, np.float32)
e32 = np.float32(0.715)
pow32 = np.power(p32, e32)
crt_powf = bind("powf", ctypes.c_float, [ctypes.c_float, ctypes.c_float])
powb = np.array([crt_powf(float(v), float(e32)) for v in p32],
                dtype=np.float32)
powc = np.array([np.float32(math.pow(float(v), float(e32))) for v in p32],
                dtype=np.float32)
print("f32 pow        vs-ucrt",
      int(np.count_nonzero(pow32.view(np.uint32) != powb.view(np.uint32))),
      "  vs-dblround",
      int(np.count_nonzero(pow32.view(np.uint32) != powc.view(np.uint32))),
      "/", pow32.size)

m32 = np.mod(np.asarray(DEG, np.float32) + np.float32(360.0),
             np.float32(360.0))
mm = []
for v in np.asarray(DEG, np.float32):
    va = np.float32(v) + np.float32(360.0)
    r = math.fmod(float(va), 360.0)
    r32 = np.float32(r)
    if r32 != np.float32(0.0) and (r32 < 0) != (np.float32(360.0) < 0):
        r32 = r32 + np.float32(360.0)
    mm.append(r32)
mm = np.array(mm, dtype=np.float32)
print("f32 mod(+360,360) vs fmodf-adjust",
      int(np.count_nonzero(m32.view(np.uint32) != mm.view(np.uint32))),
      "/", m32.size)

m64 = np.mod(DEG + 360.0, 360.0)
mm64 = []
for v in DEG:
    va = v + 360.0
    r = math.fmod(va, 360.0)
    if r != 0.0 and (r < 0) != (360.0 < 0):
        r += 360.0
    mm64.append(r)
mm64 = np.array(mm64)
print("f64 mod(+360,360) vs fmod-adjust",
      int(np.count_nonzero(m64.view(np.uint64) != mm64.view(np.uint64))),
      "/", m64.size)

# scalar-vs-array consistency inside numpy itself (twin setup is scalars)
s = np.float32(38.0) * np.float32(np.pi / 180.0)
print("f32 scalar-vs-array cos consistent:",
      np.cos(s) == np.cos(np.array([s]))[0])
print("f32 scalar-vs-array sin consistent:",
      np.sin(s) == np.sin(np.array([s]))[0])
