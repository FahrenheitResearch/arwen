"""Parallel Rust CPU fallback for native preprocessing transforms.

The shared library is built by ``tools/grib1_bridge`` alongside the native
GRIB decoders.  Both public operations consume the same FP32 array contract
as the CUDA implementations.  Work is split only across independent target
points/columns, so worker count does not affect an element's arithmetic.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Final

import numpy as np


CPU_BACKEND_ABI: Final[int] = 1
CPU_BRIDGE_ENV: Final[str] = "GPUWM_CPU_PREPROCESS_BRIDGE"

_ERRORS = {
    1: "null buffer",
    2: "invalid dimensions or option",
    3: "non-finite value or invalid pressure",
    4: "source pressure is not strictly descending",
    5: "no source level is above the surface",
    6: "target pressure lies above the source top",
    7: "no interpolation window fits the assembled column",
    127: "native CPU backend panicked",
}


def _library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("gpuwm_preprocess_cpu.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("libgpuwm_preprocess_cpu.dylib",)
    return ("libgpuwm_preprocess_cpu.so",)


def cpu_bridge_candidates() -> tuple[Path, ...]:
    """Deterministic library candidates without probing/loading them.

    One resolver, not two: this delegates to
    :func:`gpuwm.bridges.artifact_candidates`, so the CPU library is
    searched in exactly the order the bridge executables are --
    environment override (:data:`CPU_BRIDGE_ENV`), checkout
    release/debug, ``libexec/bridges``, the user-level default
    directory.
    """

    from gpuwm.bridges import artifact_candidates

    return artifact_candidates(CPU_BRIDGE_ENV, _library_names()[0])


def resolve_cpu_bridge(path: Path | str | None = None) -> Path:
    """Resolve the native CPU backend, failing with all searched locations.

    Shares :func:`gpuwm.bridges.find_artifact` semantics: a
    :data:`CPU_BRIDGE_ENV` override that names a missing file raises
    immediately, naming the variable and the path -- explicit
    configuration never silently falls through to a different library.
    """

    from gpuwm.bridges import find_artifact

    if path is not None:
        explicit = Path(path)
        if explicit.is_file():
            return explicit.resolve()
        raise FileNotFoundError(
            "GPUWM parallel CPU preprocessing bridge was not found; "
            f"searched:\n  {explicit}")
    found = find_artifact(CPU_BRIDGE_ENV, _library_names()[0])
    if found is not None:
        return found
    rendered = "\n  ".join(
        str(candidate) for candidate in cpu_bridge_candidates())
    raise FileNotFoundError(
        "GPUWM parallel CPU preprocessing bridge was not found; searched:\n  "
        + rendered)


def _workers(value: int | None, independent_count: int) -> int:
    if value is None:
        value = os.cpu_count() or 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError("workers must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("workers must be positive")
    return min(value, independent_count)


def _host_f32(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.ascontiguousarray(value, dtype=np.float32)


class _CpuRegularPlan:
    """Reusable host index plan for one source/target regular-grid map."""

    def __init__(self, backend: "CpuPreprocessBackend", latitude, longitude,
                 target_lat, target_lon):
        from gpuwm.ingest.horiz import _regular_coordinates

        y, x = _regular_coordinates(
            latitude, longitude, target_lat, target_lon)
        self._bind_coordinates(backend, (len(latitude), len(longitude)), y, x)

    @classmethod
    def from_index_coordinates(cls, backend: "CpuPreprocessBackend",
                               source_shape, y, x) -> "_CpuRegularPlan":
        """Build a plan from zero-based fractional source indices.

        Projected sources such as HRRR already own an exact target-to-source
        transform.  The Rust interpolation ABI consumes those same index
        arrays, so routing them through synthetic latitude/longitude axes
        would only add an avoidable second coordinate transform.
        """

        plan = cls.__new__(cls)
        plan._bind_coordinates(backend, source_shape, y, x)
        return plan

    def _bind_coordinates(self, backend, source_shape, y, x) -> None:
        try:
            source_shape = tuple(int(value) for value in source_shape)
        except (TypeError, ValueError) as exc:
            raise TypeError("source_shape must contain two integers") from exc
        if len(source_shape) != 2 or min(source_shape) < 1:
            raise ValueError("source_shape must contain two positive dimensions")
        y = np.asarray(y)
        x = np.asarray(x)
        if y.shape != x.shape or y.size == 0:
            raise ValueError("indexed target coordinates must be non-empty and equal-shaped")
        if not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError("indexed target coordinates must be finite")
        if y.size == 0:
            raise ValueError("target grid is empty")
        self.backend = backend
        self.source_shape = source_shape
        self.target_shape = tuple(map(int, y.shape))
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.x = np.ascontiguousarray(x, dtype=np.float32)

    def apply(self, field, method: str = "parabolic", *,
              workers: int | None = None) -> np.ndarray:
        """Apply this geometry while preserving leading field dimensions."""

        methods = {"nearest": 0, "bilinear": 1, "parabolic": 2}
        if method not in methods:
            raise ValueError(
                "method must be 'nearest', 'bilinear', or 'parabolic'")
        source = _host_f32(field)
        if source.ndim < 2 or source.shape[-2:] != self.source_shape:
            raise ValueError(
                "field trailing dimensions do not match source axes")
        leading_shape = source.shape[:-2]
        nlead = int(np.prod(leading_shape, dtype=np.int64)) or 1
        source = source.reshape((nlead, *self.source_shape))
        output = np.empty((nlead, self.y.size), dtype=np.float32)
        count = _workers(workers, self.y.size)
        code = int(self.backend._library.gpuwm_regular_interp_f32(
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(self.y.ctypes.data),
            ctypes.c_void_p(self.x.ctypes.data),
            ctypes.c_void_p(output.ctypes.data),
            nlead, self.source_shape[0], self.source_shape[1], self.y.size,
            methods[method], count,
        ))
        self.backend._raise(code, "horizontal interpolation")
        return output.reshape((*leading_shape, *self.target_shape))


class CpuPreprocessBackend:
    """Loaded ABI-v1 Rust preprocessing backend."""

    name = "cpu-rust-threads"
    arithmetic = "fp32-elementwise-v1"

    def __init__(self, bridge: Path | str | None = None):
        self.path = resolve_cpu_bridge(bridge)
        self._library = ctypes.CDLL(str(self.path))
        self._configure_abi()

    def _configure_abi(self) -> None:
        library = self._library
        library.gpuwm_preprocess_cpu_abi_version.argtypes = []
        library.gpuwm_preprocess_cpu_abi_version.restype = ctypes.c_uint32
        observed = int(library.gpuwm_preprocess_cpu_abi_version())
        if observed != CPU_BACKEND_ABI:
            raise RuntimeError(
                f"CPU preprocessing bridge ABI {observed} != required "
                f"{CPU_BACKEND_ABI}")
        self.abi_version = observed

        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        library.gpuwm_regular_interp_f32.argtypes = [
            pointer, pointer, pointer, pointer,
            size, size, size, size, ctypes.c_int32, size,
        ]
        library.gpuwm_regular_interp_f32.restype = ctypes.c_int32
        library.gpuwm_wrf_vert_interp_f32.argtypes = [
            pointer, pointer, pointer, pointer, pointer, pointer,
            size, size, size, ctypes.c_int32, ctypes.c_int32,
            size, ctypes.c_float, size, size,
        ]
        library.gpuwm_wrf_vert_interp_f32.restype = ctypes.c_int32

    def close(self) -> None:
        """Release the Windows DLL handle held by a short-lived verifier.

        Normal preprocessing keeps the backend alive for the process. Release
        assembly instead loads a DLL from a temporary Cargo target, and
        Windows will not remove that target while the loader handle is open.
        """

        library = getattr(self, "_library", None)
        if library is None:
            return
        self._library = None
        if os.name == "nt":
            import _ctypes

            _ctypes.FreeLibrary(library._handle)

    @staticmethod
    def _raise(code: int, operation: str) -> None:
        if code:
            detail = _ERRORS.get(code, f"unknown native error {code}")
            raise ValueError(f"parallel CPU {operation} failed: {detail}")

    def interpolate_regular(
            self, field, latitude, longitude, target_lat, target_lon, *,
            method: str = "parabolic", workers: int | None = None,
    ) -> np.ndarray:
        """Apply the WPS regular-grid operator on deterministic CPU chunks."""

        return self.regular_plan(
            latitude, longitude, target_lat, target_lon).apply(
                field, method=method, workers=workers)

    def regular_plan(self, latitude, longitude, target_lat, target_lon
                     ) -> _CpuRegularPlan:
        """Prepare one target staggering for repeated field interpolation."""

        return _CpuRegularPlan(
            self, latitude, longitude, target_lat, target_lon)

    def indexed_plan(self, source_shape, y, x) -> _CpuRegularPlan:
        """Prepare repeated interpolation from explicit fractional indices."""

        return _CpuRegularPlan.from_index_coordinates(
            self, source_shape, y, x)

    def wrf_vertical_interpolate(
            self, field, surface_value, source_pressure, surface_pressure,
            target_pressure, *, interp_in_logp: bool = True,
            extrap: str = "constant", force_sfc_in_vinterp: int = 1,
            zap_close_levels: float = 500.0, vboundb: int = 4,
            workers: int | None = None,
    ) -> np.ndarray:
        """Apply WRF-real vertical interpolation with dynamic level counts."""

        if extrap not in ("constant", "temperature"):
            raise ValueError("extrap must be 'constant' or 'temperature'")
        if not isinstance(interp_in_logp, (bool, np.bool_)):
            raise TypeError("interp_in_logp must be boolean")
        values = _host_f32(field)
        source = _host_f32(source_pressure)
        surface_values = _host_f32(surface_value)
        surface_pressures = _host_f32(surface_pressure)
        target = _host_f32(target_pressure)
        if values.ndim != 3 or target.ndim != 3:
            raise ValueError(
                "field and target_pressure must be (level, y, x)")
        if source.shape != values.shape:
            raise ValueError("source_pressure shape does not match field")
        if surface_values.shape != values.shape[1:] \
                or surface_pressures.shape != values.shape[1:]:
            raise ValueError("surface fields must be (y, x)")
        if target.shape[1:] != values.shape[1:]:
            raise ValueError("source and target horizontal shapes differ")
        if not 0 <= int(force_sfc_in_vinterp) <= target.shape[0]:
            raise ValueError(
                "force_sfc_in_vinterp must be within target levels")
        descending = bool(np.all(source[:-1] > source[1:]))
        ascending = bool(np.all(source[:-1] < source[1:]))
        if not descending and not ascending:
            raise ValueError(
                "source pressure must be strictly monotonic in every column")
        if ascending:
            source = np.ascontiguousarray(source[::-1])
            values = np.ascontiguousarray(values[::-1])
        nsource, ny, nx = values.shape
        ntarget = target.shape[0]
        ncolumn = ny * nx
        output = np.empty(target.shape, dtype=np.float32)
        count = _workers(workers, ncolumn)
        code = int(self._library.gpuwm_wrf_vert_interp_f32(
            ctypes.c_void_p(values.ctypes.data),
            ctypes.c_void_p(surface_values.ctypes.data),
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(surface_pressures.ctypes.data),
            ctypes.c_void_p(target.ctypes.data),
            ctypes.c_void_p(output.ctypes.data),
            nsource, ntarget, ncolumn,
            int(bool(interp_in_logp)), int(extrap == "temperature"),
            int(force_sfc_in_vinterp), float(zap_close_levels),
            int(vboundb), count,
        ))
        self._raise(code, "vertical interpolation")
        return output


__all__ = [
    "CPU_BACKEND_ABI",
    "CPU_BRIDGE_ENV",
    "CpuPreprocessBackend",
    "cpu_bridge_candidates",
    "resolve_cpu_bridge",
]
