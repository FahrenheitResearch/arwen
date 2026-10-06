"""ABI buffers for native f64 Noah soil liquid-water initialization."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import sys

import numpy as np

NOAH_SH2O_ENTRY = "gpuwm_noah_initialize_sh2o_f64"
NOAH_FRH2O_ENTRY = "gpuwm_noah_frh2o_f64"
NOAH_CATEGORY_SCAN_ENTRY = "gpuwm_noah_category_scan_extended"


class NoahInitUnavailable(RuntimeError):
    """The selected preparation library cannot initialize frozen Noah soil."""


@lru_cache(maxsize=8)
def _load(path):
    from gpuwm.ingest.cpu_backend import CPU_BACKEND_ABI
    library = ctypes.CDLL(str(path))
    try:
        probe = library.gpuwm_preprocess_cpu_abi_version
    except AttributeError as error:
        raise NoahInitUnavailable(
            "the selected Rust preparation library lacks gpuwm_preprocess_cpu_abi_version, "
            "so its Noah cold-init ABI cannot be checked; stage current bridges with "
            "gpuwm fetch-bridges or rebuild tools/grib1_bridge with "
            "cargo build --release --lib --offline") from error
    probe.argtypes = []
    probe.restype = ctypes.c_uint32
    version = int(probe())
    if version != CPU_BACKEND_ABI:
        raise NoahInitUnavailable(
            f"Noah soil initialization needs preparation ABI {CPU_BACKEND_ABI}, "
            f"the selected library supplies {version}")
    try:
        initialize = getattr(library, NOAH_SH2O_ENTRY)
        scalar = getattr(library, NOAH_FRH2O_ENTRY)
        extended_scan = getattr(library, NOAH_CATEGORY_SCAN_ENTRY)
    except AttributeError as error:
        raise NoahInitUnavailable(
            f"the selected Rust preparation library lacks {NOAH_SH2O_ENTRY} "
            f"or {NOAH_FRH2O_ENTRY} or {NOAH_CATEGORY_SCAN_ENTRY}, so Noah frozen-soil liquid water cannot "
            "be initialized; stage current bridges with gpuwm fetch-bridges "
            "or rebuild tools/grib1_bridge with cargo build --release --lib --offline") from error
    f64 = ctypes.POINTER(ctypes.c_double)
    size = ctypes.c_size_t
    scalar.argtypes = [ctypes.c_double] * 6 + [ctypes.c_uint32, f64,
        ctypes.POINTER(size), ctypes.POINTER(ctypes.c_uint32)]
    scalar.restype = ctypes.c_int32
    initialize.argtypes = [f64] * 4 + [size] * 4 + [f64, ctypes.POINTER(size)]
    initialize.restype = ctypes.c_int32
    extended_scan.argtypes = [ctypes.POINTER(ctypes.c_uint8), size,
        ctypes.c_uint32, ctypes.c_uint32, f64, ctypes.POINTER(size)]
    extended_scan.restype = ctypes.c_int32
    return library


def load():
    """Load and prove the selected native cold-init entries."""
    from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
    return _load(resolve_cpu_bridge())


def unavailable_reason():
    """The concrete native capability failure for doctor and setup checks."""
    try:
        load()
    except (OSError, RuntimeError, AttributeError) as error:
        return str(error)
    return None


def _check(code):
    if code == 42:
        raise ValueError("math domain error")
    if code == 43:
        raise ZeroDivisionError("float division by zero")
    if code == 44:
        raise OverflowError("Numerical result out of range")
    if code == 45:
        raise TypeError("must be real number, not complex")
    if code:
        raise RuntimeError(f"native Noah soil initialization failed with code {code}")


def frh2o(temperature, moisture, liquid, maximum, exponent, suction):
    """One scalar solve through the same native implementation as grid setup."""
    args = (temperature, moisture, liquid, maximum, exponent, suction)
    kinds = 0
    for index, value in enumerate(args):
        if isinstance(value, np.generic):
            if isinstance(value, np.float32):
                kind = 1
            elif isinstance(value, np.float64):
                kind = 2
            else:
                raise TypeError(
                    "native scalar Noah FRH2O accepts NumPy float32/float64; "
                    "narrowing another scalar dtype could change its arithmetic")
            kinds |= kind << (2 * index)
    output = ctypes.c_double()
    iterations = ctypes.c_size_t()
    fallback = ctypes.c_uint32()
    code = getattr(load(), NOAH_FRH2O_ENTRY)(
        *(float(value) for value in args), kinds,
        ctypes.byref(output), ctypes.byref(iterations), ctypes.byref(fallback))
    _check(code)
    return output.value


def initialize(moisture, temperature, soil_type, soil_table, *, workers=None):
    """Prepare contiguous buffers; Rust owns validation and cell arithmetic."""
    from gpuwm.ingest.cpu_backend import _workers
    moisture = np.ascontiguousarray(moisture, dtype=np.float64)
    temperature = np.ascontiguousarray(temperature, dtype=np.float64)
    original_categories = np.asarray(soil_type)
    if original_categories.dtype.kind not in "biuf":
        raise TypeError("isltyp must contain real numeric categories")
    if original_categories.dtype.kind == "f" and original_categories.dtype.itemsize > 8:
        info = np.finfo(original_categories.dtype)
        formats = {(63, 15, 16): 0, (112, 15, 16): 1}
        format_code = formats.get((info.nmant, info.nexp, original_categories.dtype.itemsize))
        if format_code is None:
            raise NoahInitUnavailable(
                "native Noah category validation cannot read this extended-float format, "
                "and narrowing it could admit fractional soil categories")
        raw = np.ascontiguousarray(original_categories)
        categories = np.empty(raw.shape, dtype=np.float64)
        byte_order = raw.dtype.byteorder
        big_endian = byte_order == ">" or (byte_order in "=|" and sys.byteorder == "big")
        error_index = ctypes.c_size_t()
        code = getattr(load(), NOAH_CATEGORY_SCAN_ENTRY)(
            raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), raw.size,
            format_code, int(big_endian), categories.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(error_index))
        if code == 40:
            raise ValueError("isltyp must contain finite integer categories")
        _check(code)
    else:
        categories = np.ascontiguousarray(original_categories, dtype=np.float64)
    table = np.ascontiguousarray(soil_table, dtype=np.float64)
    if moisture.shape != temperature.shape or moisture.ndim == 0:
        raise ValueError("smois and tslb must be same-shape soil profiles")
    columns = int(np.prod(moisture.shape[1:], dtype=np.int64))
    if categories.size != columns or table.ndim != 2 or table.shape[1] != 3:
        raise ValueError("Noah soil categories or parameter-table shape do not match the profiles")
    workers = _workers(workers, max(columns, 1))
    output = np.empty_like(moisture)
    error_column = ctypes.c_size_t()
    pointer = ctypes.POINTER(ctypes.c_double)
    code = getattr(load(), NOAH_SH2O_ENTRY)(
        *(value.ctypes.data_as(pointer) for value in (moisture, temperature, categories, table)),
        table.shape[0], moisture.shape[0], columns, int(workers),
        output.ctypes.data_as(pointer), ctypes.byref(error_column))
    if code == 40:
        raise ValueError("isltyp must contain finite integer categories")
    if code == 41:
        category = int(original_categories.reshape(-1)[error_column.value])
        raise ValueError(f"isltyp category {category} is outside table")
    _check(code)
    return output
