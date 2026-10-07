"""Table-named source conversions through the Rust preprocessing bridge.

UPP MDLFLD.f:2442 at 1296eebb295251d0fe1cc697f47058c62d887977
uses (1/RD)*(P/T), RD=287.04 (params.F:57). This effective density
has no water-vapor term. Inverting it recovers the source dry-air tracer
convention, subject to independent GRIB packing of mass, pressure and T.
"""
from __future__ import annotations

import ctypes
import numpy as np


def weighted_source_fields(fields, weights, *, cpu_bridge=None):
    """Combine boundary terms in row order through Rust, on source levels."""
    from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
    fields = tuple(np.ascontiguousarray(value, dtype=np.float32) for value in fields)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    if not fields or weights.shape != (len(fields),) or any(value.shape != fields[0].shape for value in fields):
        raise ValueError("boundary terms or weights have inconsistent shapes; prevents combining different source levels")
    if len(fields) == 1 and weights[0] == 1:
        return fields[0]
    packed = np.ascontiguousarray(np.stack(fields))
    output = np.empty_like(fields[0])
    library = ctypes.CDLL(str(resolve_cpu_bridge(cpu_bridge)))
    try:
        function = library.gpuwm_boundary_weighted_sum_f32
    except AttributeError as exc:
        raise RuntimeError("CPU bridge lacks weighted boundary sum; rebuild to prevent incorrect source composition") from exc
    function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t] * 2
    function.restype = ctypes.c_int32
    code = function(*(ctypes.c_void_p(value.ctypes.data) for value in (packed, weights, output)), len(fields), output.size)
    if code:
        raise ValueError(f"weighted boundary sum rejected invalid terms (code {code}); prevents negative or non-finite tracer forcing")
    return output


def convert_source(conversion, values, met, *, parameters=None, cpu_bridge=None):
    """Convert equal-shaped source-level FP32 arrays before any remapping.

    ``met`` provides PRES (Pa), TT (K) from the same record level. Source
    variable metadata supplies ``parameters``: density convention and gas
    constant. No device is used. Result: contiguous host FP32 in row units.
    """
    if conversion == "identity":
        return np.ascontiguousarray(values, dtype=np.float32)
    functions = {"kg_m3_to_ug_kg_dry": "gpuwm_kg_m3_to_ug_kg_dry_f32"}
    if conversion not in functions:
        raise ValueError(f"conversion {conversion!r} has no implementation; refusing incorrect tracer units")
    if parameters is None or parameters.get("density") != "pressure_over_temperature":
        raise ValueError("source density convention is absent or unsupported; prevents using the wrong density in tracer conversion")
    if "gas_constant_j_kg_k" not in parameters:
        raise ValueError("source gas constant is absent; prevents silently changing the source density formula")
    from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
    arrays = [np.ascontiguousarray(a, dtype=np.float32)
              for a in (values, met["PRES"], met["TT"])]
    if not arrays[0].size or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("source conversion shapes differ or are empty; refusing density from a different level")
    library = ctypes.CDLL(str(resolve_cpu_bridge(cpu_bridge)))
    try:
        function = getattr(library, functions[conversion])
    except AttributeError as exc:
        raise RuntimeError("CPU bridge lacks boundary conversion; rebuild to prevent unconverted tracer units") from exc
    function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t, ctypes.c_float]
    function.restype = ctypes.c_int32
    output = np.empty_like(arrays[0])
    code = function(*(ctypes.c_void_p(a.ctypes.data) for a in (*arrays, output)),
                    output.size, float(parameters["gas_constant_j_kg_k"]))
    if code:
        raise ValueError(f"boundary conversion rejected invalid mass or density (code {code}); prevents non-finite or negative chemistry")
    return output
