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


class UnitsEpochError(ValueError):
    """A source field's units depend on its cycle and the cycle is unknown or
    the field's ``units_epochs`` rows are malformed."""


def _cycle_utc(moment):
    from datetime import datetime, timezone
    if isinstance(moment, str):
        moment = datetime.fromisoformat(moment.replace("Z", "+00:00"))
    if not isinstance(moment, datetime):
        raise UnitsEpochError(f"cycle {moment!r} is not a date and time")
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def units_epoch_scale(spec, cycle, *, source="source", variable="field"):
    """The factor that brings one source variable's bytes to its declared units.

    ``spec`` is the source row's variable entry.  Its optional
    ``units_epochs`` rows name cycle ranges (``from_cycle`` inclusive,
    ``until_cycle`` exclusive, UTC) in which the producer wrote other units
    under the same GRIB2 parameter, with the ``scale`` that converts them to
    ``spec["units"]``.  Outside every epoch the factor is 1.0, so a field
    whose cycle sits in the declared-units era is read unchanged.

    ``cycle`` is the cycle (reference time) of the files the bytes came from.
    A variable with epochs and no cycle is refused: its units cannot be
    known.  Breakage this prevents: NOAA HRRR wrfnat and RAP awp130bgrb wrote
    MASSDEN in ug m-3 through the 2021-12-21 13Z cycle and kg m-3 from 14Z,
    both labelled kg m-3, so an earlier smoke start read as kg m-3 began
    1e9 too large (8e8 ug/kg smoke, nwfa ~6e17, an initial health failure).
    """
    epochs = spec.get("units_epochs") or ()
    if not epochs:
        return 1.0
    if cycle is None:
        bounds = ", ".join(str(e.get("until_cycle") or e.get("from_cycle")) for e in epochs)
        raise UnitsEpochError(
            f"{source} {variable}: its units changed at cycle {bounds} (the source "
            f"row's units_epochs), and this read carries no file cycle, so it cannot "
            f"tell {spec.get('units')!r} bytes from {[e.get('units') for e in epochs]}; "
            "refusing rather than risk a 1e9 unit error. Pass the cycle of the files "
            "(the fetch receipt's, or the snapshot's valid time minus its lead).")
    when = _cycle_utc(cycle)
    matches = []
    for epoch in epochs:
        scale = epoch.get("scale")
        if (not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0
                or not epoch.get("units") or not epoch.get("provenance")
                or not (epoch.get("from_cycle") or epoch.get("until_cycle"))):
            raise UnitsEpochError(
                f"{source} {variable}: a units_epochs row needs units, a finite positive "
                "scale, provenance and from_cycle and/or until_cycle; prevents applying "
                "an undefined unit change")
        lower = epoch.get("from_cycle")
        upper = epoch.get("until_cycle")
        if ((lower is None or when >= _cycle_utc(lower))
                and (upper is None or when < _cycle_utc(upper))):
            matches.append(float(scale))
    if len(matches) > 1:
        raise UnitsEpochError(
            f"{source} {variable}: cycle {when.isoformat()} falls in more than one "
            "units_epochs row; prevents an ambiguous unit change")
    return matches[0] if matches else 1.0


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
