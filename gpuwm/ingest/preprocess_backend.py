"""Common CUDA/parallel-CPU contract for native preprocessing.

The forecast model itself remains CUDA-only.  This module selects the
backend used by the source-grid and WRF-real setup transforms that precede
the model allocation.  Both implementations consume and emit FP32 arrays;
the CPU implementation delegates interpolation arithmetic to the packaged
Rust bridge and uses deterministic NumPy elementwise helpers for the small
surface/vector transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from gpuwm.ingest.cpu_backend import CPU_BACKEND_ABI, CpuPreprocessBackend


PREPROCESS_IMPLEMENTATION_SCHEMA = "gpuwm-preprocess-implementation-v2"
PSFC_MAPPING_POLICY = "canonical-f32-coordinate-f64-bilinear-single-round-v1"
VERTICAL_STENCIL_POLICY = "wrf-v4.6.1-strict-fp32-zap-close-levels-v1"

_COMMON_IMPLEMENTATION_SOURCES = (
    "gpuwm/ingest/horiz.py",
    "gpuwm/ingest/preprocess_backend.py",
    "gpuwm/ingest/real.py",
    "gpuwm/ingest/vert.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_tree(backend: str) -> dict[str, object]:
    """Bind the trajectory-changing preprocessing source implementation."""

    root = Path(__file__).resolve().parents[2]
    names = list(_COMMON_IMPLEMENTATION_SOURCES)
    if backend == "cpu":
        names.append("gpuwm/ingest/cpu_backend.py")
    elif backend == "cuda":
        names.append("gpuwm/core/kernels/vert_interp.cu")
    else:  # pragma: no cover - internal call sites own the finite inventory
        raise ValueError(f"unsupported provenance backend {backend!r}")
    files = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"preprocessing implementation source is missing: {path}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    encoded = json.dumps(
        files, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return {
        "schema": "gpuwm-preprocess-source-tree-v1",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _shared_contracts() -> dict[str, object]:
    return {
        "surface_pressure_mapping": {
            "policy": PSFC_MAPPING_POLICY,
            "output_dtype": "float32",
        },
        "vertical_stencil": {
            "policy": VERTICAL_STENCIL_POLICY,
            "zap_close_levels_pa": 500.0,
            "predicate": "separation < zap_close_levels",
        },
    }


def _host(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=dtype)


def _masked_nearest_cpu(field, latitude, longitude, target_lat, target_lon,
                        source_landmask, target_landmask, *, surface="match",
                        fill_value=0.0, search_radius=8, strict=True):
    """NumPy FP32 mirror of the bounded WPS surface-nearest operator."""

    from gpuwm.ingest.horiz import _regular_coordinates

    if isinstance(search_radius, (bool, np.bool_)) \
            or not isinstance(search_radius, (int, np.integer)) \
            or int(search_radius) < 0:
        raise ValueError("search_radius must be a non-negative integer")
    y_raw, x_raw = _regular_coordinates(
        latitude, longitude, target_lat, target_lon)
    field = np.asarray(_host(field), dtype=np.float32)
    source_landmask = np.asarray(_host(source_landmask), dtype=np.bool_)
    target_landmask = np.asarray(_host(target_landmask), dtype=np.bool_)
    source_shape = (len(latitude), len(longitude))
    if field.ndim != 2 or field.shape != source_shape \
            or source_landmask.shape != source_shape:
        raise ValueError("field/source_landmask shape does not match source axes")
    if target_landmask.shape != y_raw.shape:
        raise ValueError(
            "target_landmask shape does not match target coordinates")
    if surface == "match":
        active = np.ones_like(target_landmask)
        desired_land = target_landmask
    elif surface == "land":
        active = target_landmask
        desired_land = np.ones_like(target_landmask)
    elif surface == "water":
        active = ~target_landmask
        desired_land = np.zeros_like(target_landmask)
    else:
        raise ValueError("surface must be 'match', 'land', or 'water'")

    y = np.asarray(y_raw, dtype=np.float32)
    x = np.asarray(x_raw, dtype=np.float32)
    center_y = np.rint(y).astype(np.int32)
    center_x = np.rint(x).astype(np.int32)
    best_distance = np.full(y.shape, np.inf, dtype=np.float32)
    best_value = np.full(y.shape, np.float32(fill_value), dtype=np.float32)
    ny, nx = source_shape
    radius = int(search_radius)
    for dj in range(-radius, radius + 1):
        jy = center_y + dj
        in_y = (jy >= 0) & (jy < ny)
        jy_safe = np.clip(jy, 0, ny - 1)
        for di in range(-radius, radius + 1):
            ix = center_x + di
            inside = in_y & (ix >= 0) & (ix < nx)
            ix_safe = np.clip(ix, 0, nx - 1)
            value = field[jy_safe, ix_safe]
            valid = (active & inside & np.isfinite(value)
                     & (source_landmask[jy_safe, ix_safe] == desired_land))
            distance = (y - jy_safe) ** np.float32(2.0) \
                + (x - ix_safe) ** np.float32(2.0)
            take = valid & (distance < best_distance)
            best_distance = np.where(take, distance, best_distance)
            best_value = np.where(take, value, best_value)
    if strict and np.any(active & ~np.isfinite(best_distance)):
        raise ValueError("no matching source surface within search_radius")
    return np.ascontiguousarray(best_value, dtype=np.float32)


def _rotate_earth_to_grid_cpu(u_earth, v_earth, sinalpha, cosalpha):
    u = np.asarray(_host(u_earth), dtype=np.float32)
    v = np.asarray(_host(v_earth), dtype=np.float32)
    sina = np.asarray(_host(sinalpha), dtype=np.float32)
    cosa = np.asarray(_host(cosalpha), dtype=np.float32)
    if u.shape != v.shape:
        raise ValueError("u_earth and v_earth shapes differ")
    return (np.asarray(u * cosa + v * sina, dtype=np.float32),
            np.asarray(v * cosa - u * sina, dtype=np.float32))


def _era5_rh_to_water_cpu(relative_humidity, temperature):
    """WPS v4.6 ``rrpr.F:fix_gfs_rh`` mixed-phase RH to liquid RH (CPU).

    Same transcription as the CUDA path: Murphy and Koop 2005 ice saturation
    vapor pressure, Bolton 1980 liquid, linear 253.15-273.15 K blend, only
    applied at and below freezing.  Float64 setup math, FP32 result.
    """
    rh = np.asarray(_host(relative_humidity), dtype=np.float64)
    t = np.asarray(_host(temperature), dtype=np.float64)
    if rh.shape != t.shape:
        raise ValueError("relative_humidity and temperature shapes differ")
    eis = 0.01 * np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t)
                        - 0.00728332 * t)
    ews = 6.112 * np.exp(17.67 * (t - 273.15) / ((t - 273.15) + 243.5))
    frac = (273.15 - t) / 20.0
    blended = frac * eis + (1.0 - frac) * ews
    r = np.where(t > 253.15, blended, eis)
    converted = np.where(t <= 273.15, rh * (r / ews), rh)
    return np.asarray(converted, dtype=np.float32)


@dataclass(frozen=True)
class _CpuVerticalPlan:
    backend: "ParallelCpuPreprocessBackend"
    source_pressure: np.ndarray
    surface_pressure: np.ndarray
    target_pressure: np.ndarray

    def apply(self, field, surface_value, **options):
        # CUDA can skip a redundant finite-value scan after a caller-side
        # validation.  The native CPU ABI always validates at its boundary.
        options.pop("values_are_finite", None)
        return self.backend._native.wrf_vertical_interpolate(
            field, surface_value, self.source_pressure,
            self.surface_pressure, self.target_pressure,
            workers=self.backend.workers, **options)


@dataclass(frozen=True)
class _CudaVerticalPlan:
    raw_plan: object

    def apply(self, field, surface_value, **options):
        from gpuwm.ingest.vert import _wrf_vert_interp_gpu_prepared

        return _wrf_vert_interp_gpu_prepared(
            field, surface_value, self.raw_plan, **options)


class ParallelCpuPreprocessBackend:
    """Production adapter for the deterministic packaged Rust CPU bridge."""

    name = "cpu"
    implementation = "rust-scoped-threads-fp32-v1"
    array_module = np

    def __init__(self, *, workers: int | None = None,
                 bridge: Path | str | None = None):
        if isinstance(workers, (bool, np.bool_)) or (
                workers is not None
                and not isinstance(workers, (int, np.integer))):
            raise TypeError("workers must be an integer")
        if workers is not None and int(workers) < 1:
            raise ValueError("workers must be positive")
        self.workers = None if workers is None else int(workers)
        self._native = CpuPreprocessBackend(bridge)

    @staticmethod
    def float32(value):
        return np.asarray(_host(value), dtype=np.float32)

    @staticmethod
    def bool_array(value):
        return np.asarray(_host(value), dtype=np.bool_)

    def regular_plan(self, latitude, longitude, target_lat, target_lon):
        native = self._native.regular_plan(
            latitude, longitude, target_lat, target_lon)

        return self._bind_native_plan(native)

    def indexed_plan(self, source_shape, y, x):
        """Bind an exact projected-source fractional-index plan."""

        native = self._native.indexed_plan(source_shape, y, x)
        return self._bind_native_plan(native)

    @property
    def indexed_donor_interp(self) -> bool:
        """Whether this library carries the exact-donor horizontal entry."""

        return bool(getattr(self._native, "indexed_donor_interp", False))

    def indexed_donor_plan(self, source_shape, donor_y, donor_x,
                           fraction_y, fraction_x):
        """Bind an exact projected-source integer-donor plan."""

        native = self._native.indexed_donor_plan(
            source_shape, donor_y, donor_x, fraction_y, fraction_x)
        return self._bind_native_plan(native)

    def _bind_native_plan(self, native):
        workers = self.workers

        class BoundPlan:
            source_shape = native.source_shape
            target_shape = native.target_shape

            def apply(bound_self, field, method="parabolic"):
                return native.apply(
                    field, method=method, workers=workers)

        return BoundPlan()

    @staticmethod
    def masked_nearest(*args, **kwargs):
        return _masked_nearest_cpu(*args, **kwargs)

    @staticmethod
    def rotate_earth_to_grid(*args):
        return _rotate_earth_to_grid_cpu(*args)

    @staticmethod
    def era5_rh_to_water(*args):
        return _era5_rh_to_water_cpu(*args)

    def prepare_wrf_vertical(self, source_pressure, surface_pressure,
                             target_pressure):
        source = np.ascontiguousarray(_host(source_pressure), dtype=np.float32)
        surface = np.ascontiguousarray(
            _host(surface_pressure), dtype=np.float32)
        target = np.ascontiguousarray(_host(target_pressure), dtype=np.float32)
        return _CpuVerticalPlan(self, source, surface, target)

    def receipt(self) -> dict[str, object]:
        """Return proof metadata that binds the native CPU implementation."""

        return {
            "schema": PREPROCESS_IMPLEMENTATION_SCHEMA,
            "backend": self.name,
            "implementation": self.implementation,
            "workers": self.workers if self.workers is not None else "auto",
            "host_cpu_count": os.cpu_count(),
            "bridge": {
                "name": self._native.path.name,
                "sha256": _sha256(self._native.path),
                "abi_version": self._native.abi_version,
                "required_abi_version": CPU_BACKEND_ABI,
            },
            "contracts": _shared_contracts(),
            "implementation_tree": _implementation_tree("cpu"),
        }


class CudaPreprocessBackend:
    """Adapter around the existing CuPy/JIT CUDA preprocessing kernels."""

    name = "cuda"
    implementation = "cupy-fp32"
    workers = 1

    @property
    def array_module(self):
        from gpuwm.ingest.horiz import _cupy

        return _cupy()

    def float32(self, value):
        cp = self.array_module
        return cp.asarray(value, dtype=cp.float32)

    def bool_array(self, value):
        cp = self.array_module
        return cp.asarray(value, dtype=cp.bool_)

    @staticmethod
    def regular_plan(latitude, longitude, target_lat, target_lon):
        from gpuwm.ingest.horiz import _RegularGpuPlan

        return _RegularGpuPlan(
            latitude, longitude, target_lat, target_lon)

    @staticmethod
    def masked_nearest(*args, **kwargs):
        from gpuwm.ingest.horiz import masked_nearest_gpu

        return masked_nearest_gpu(*args, **kwargs)

    @staticmethod
    def rotate_earth_to_grid(*args):
        from gpuwm.ingest.horiz import rotate_earth_to_grid_gpu

        return rotate_earth_to_grid_gpu(*args)

    @staticmethod
    def era5_rh_to_water(*args):
        from gpuwm.ingest.horiz import _era5_rh_to_water_gpu

        return _era5_rh_to_water_gpu(*args)

    @staticmethod
    def prepare_wrf_vertical(source_pressure, surface_pressure,
                             target_pressure):
        from gpuwm.ingest.vert import _prepare_wrf_vert_interp_geometry

        return _CudaVerticalPlan(_prepare_wrf_vert_interp_geometry(
            source_pressure, surface_pressure, target_pressure))

    def receipt(self) -> dict[str, object]:
        """Return proof metadata for the selected CuPy CUDA implementation."""

        cp = self.array_module
        return {
            "schema": PREPROCESS_IMPLEMENTATION_SCHEMA,
            "backend": self.name,
            "implementation": self.implementation,
            "workers": self.workers,
            "cupy_version": cp.__version__,
            "contracts": _shared_contracts(),
            "implementation_tree": _implementation_tree("cuda"),
        }


def resolve_preprocess_backend(backend="cuda", *, workers: int | None = None,
                               cpu_bridge: Path | str | None = None):
    """Resolve a public backend selector without silently changing policy."""

    if backend is None:
        backend = "cuda"
    if not isinstance(backend, str):
        required = (
            "regular_plan", "masked_nearest", "rotate_earth_to_grid",
            "era5_rh_to_water", "prepare_wrf_vertical", "receipt",
        )
        if not all(callable(getattr(backend, name, None)) for name in required):
            raise TypeError("custom preprocessing backend is incomplete")
        if workers is not None or cpu_bridge is not None:
            raise ValueError(
                "workers/cpu_bridge cannot accompany a backend object")
        return backend
    normalized = backend.strip().lower()
    sealed_backends = _sealed_distribution_preprocess_backends()
    if normalized == "cuda":
        if sealed_backends is not None and "cuda" not in sealed_backends:
            raise ValueError(
                "CUDA preprocessing is absent from the sealed native "
                "distribution")
        if workers is not None or cpu_bridge is not None:
            raise ValueError(
                "workers/cpu_bridge apply only to the CPU backend")
        return CudaPreprocessBackend()
    if normalized == "cpu":
        if sealed_backends is not None and "cpu" not in sealed_backends:
            raise ValueError(
                "CPU preprocessing is absent from the sealed native "
                "distribution")
        return ParallelCpuPreprocessBackend(
            workers=workers, bridge=cpu_bridge)
    if normalized == "auto":
        if cpu_bridge is not None:
            raise ValueError("cpu_bridge cannot accompany backend='auto'")
        if sealed_backends is not None and "cuda" not in sealed_backends:
            return ParallelCpuPreprocessBackend(workers=workers)
        try:
            candidate = CudaPreprocessBackend()
            cp = candidate.array_module
            runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
            cupy_major = int(str(cp.__version__).split(".", 1)[0])
            if (
                int(cp.cuda.runtime.getDeviceCount()) > 0
                and cupy_major >= 13
                and 12_000 <= runtime_version < 13_000
            ):
                return candidate
        except (AttributeError, ImportError, RuntimeError, ValueError):
            pass
        return ParallelCpuPreprocessBackend(workers=workers)
    raise ValueError("backend must be 'cuda', 'cpu', or 'auto'")


def _sealed_distribution_preprocess_backends() -> tuple[str, ...] | None:
    """Return the exact backend inventory of an installed native package.

    The inventory is read from a document that has already been checked
    against its WHOLE schema
    (:func:`gpuwm.runtime_manifest.validate_manifest`), not against the
    one key this function needs.  Reading a key at a time is how a
    manifest missing ``contract`` entirely passed the identity gate,
    ran a preparation for minutes, and then died here -- correctly, and
    far too late.  Same refusal, hoisted to the first read.
    """

    from gpuwm.runtime_manifest import manifest_from_environment

    bound = manifest_from_environment()
    if bound is None:
        return None
    _path, payload = bound
    backends = payload["contract"]["platform"]["backends"]
    return tuple(backends)


def release_backend_memory(backend) -> None:
    """Hand this backend's unreferenced blocks back to the device.

    Streaming ingest drops one forcing time's arrays before it builds the
    next, but CuPy's pool keeps every freed block, so the machine-visible
    footprint would stay pinned at the high-water mark of the very first
    time even though nothing is using it.  Asking the pool to release
    what nobody holds is what turns the Python-level release into a
    device-level one.

    ``gc.collect()`` runs first because a released :class:`DomainState`
    can be reachable only through a reference cycle, and a cycle-collected
    array is not freed until the collector runs.  Live arrays are never
    touched: :meth:`free_all_blocks` releases unreferenced blocks only.
    This is a memory operation with no numerical effect whatsoever.
    """

    # Best effort, by design.  This function exists to hand back memory
    # nobody is using; it decides nothing and computes nothing, so it must
    # never be the reason a preparation fails -- nor a reason anything
    # else behaves differently.  Both guards below are about that.
    #
    # A backend with no device pool has nothing to hand back: the CPU
    # backend's arrays are NumPy, freed by refcount, and a CPU stand-in
    # swapped in for cupy is the same story.  Returning before the
    # collector matters because gc.collect() is a WHOLE-PROCESS event --
    # charging every caller of a no-op release for a full collection is
    # a side effect on code that never asked for one.
    pools = []
    if getattr(backend, "name", None) == "cuda":
        module = backend.array_module
        pools = [getattr(module, accessor, None)
                 for accessor in ("get_default_memory_pool",
                                  "get_default_pinned_memory_pool")]
        pools = [accessor for accessor in pools if accessor is not None]
    if not pools:
        return
    # Only now, and only because it pays for itself: a released
    # DomainState can be reachable through a reference cycle, and a
    # cycle-collected array is not returned to the pool until the
    # collector runs.  This is what turns the Python-level release into
    # a device-level one.  No numerical effect whatsoever.
    gc.collect()
    for accessor in pools:
        accessor().free_all_blocks()


__all__ = [
    "CudaPreprocessBackend",
    "ParallelCpuPreprocessBackend",
    "release_backend_memory",
    "resolve_preprocess_backend",
]
