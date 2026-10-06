"""Array transport to the regional rain kernels in the Rust obs-score library."""
from __future__ import annotations

import ctypes as ct
import numpy as np
from gpuwm import obs_score_bridge as bridge

_LIBRARY = None


def load():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    lib = bridge.load()
    f64, u8, i64 = ct.POINTER(ct.c_double), ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int64)
    size, real, code = ct.c_size_t, ct.c_double, ct.c_uint32
    signatures = {
        "laea": [f64, f64, size, real, real, f64, f64],
        "corners": [f64, size, size, f64],
        "remap": [f64, u8, f64, f64, size, size, f64, f64, size, size, f64, u8, f64, f64],
        "hour": [f64, f64, u8, i64, f64, size, size, real, real, real, code, f64, u8, u8, f64],
        "fss": [f64, f64, u8, size, size, real, real, real, f64, u8],
        "sum": [f64, u8, size, f64],
        "combine": [f64, size, size, code, f64],
        "mask": [f64, u8, size, real, u8],
    }
    try:
        probe = lib.gpuwm_rain_gate_abi_version
        probe.argtypes = []
        probe.restype = ct.c_uint32
        if probe() != 1:
            raise ValueError("regional rain native ABI is incompatible")
        for name, args in signatures.items():
            fn = getattr(lib, "gpuwm_rain_gate_" + name)
            fn.argtypes = args
            fn.restype = ct.c_int32
    except AttributeError as error:
        raise bridge.ObsScoreBridgeError(
            "obs-score has no regional rain kernels; build this checkout with "
            "cargo build --release --offline --locked -p obs-score -j 2 "
            "from tools/rustwx, then set GPUWM_OBSSCORE_BRIDGE to that library"
        ) from error
    _LIBRARY = lib
    return lib


def array(value, dtype=np.float64):
    return np.ascontiguousarray(value, dtype=dtype)


def call(name, *args):
    lib = load()
    converted = [bridge._ptr(a) if isinstance(a, np.ndarray) else a for a in args]
    bridge._check(lib, getattr(lib, "gpuwm_rain_gate_" + name)(*converted))


def sum_masked(field, mask):
    field, mask = array(field), array(mask, np.uint8)
    if field.shape != mask.shape:
        raise ValueError("scored mask and field shape differ")
    out = np.empty(1, np.float64)
    call("sum", field, mask, field.size, out)
    return float(out[0])


def combine(fields, mode="sum"):
    fields = array(fields)
    if fields.ndim < 2:
        raise ValueError("native combination needs a leading field axis")
    out = np.empty(fields.shape[1:], np.float64)
    call("combine", fields, len(fields), out.size, {"sum": 0, "max": 1, "and": 2, "product": 3}[mode], out)
    return out


def quality_mask(field, valid, minimum):
    field, valid = array(field), array(valid, np.uint8)
    if field.shape != valid.shape:
        raise ValueError("quality field and native bitmap shapes differ")
    out = np.empty(field.shape, np.uint8)
    call("mask", field, valid, field.size, float(minimum), out)
    return out.view(np.bool_)
