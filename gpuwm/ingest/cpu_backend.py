"""Parallel Rust CPU fallback for native preprocessing transforms.

The shared library is built by ``tools/grib1_bridge`` alongside the native
GRIB decoders.  Every public operation consumes the same FP32 array
contract as the CUDA implementations.  Work is split only across
independent target points/columns, so worker count does not affect an
element's arithmetic.

There are two horizontal entry points, and the difference between them is
WHO OWNS THE DONOR.  ``gpuwm_regular_interp_f32`` takes a fractional
source coordinate and derives the donor itself, which is right for a
regular lat/lon source.  ``gpuwm_indexed_interp_f32`` takes the donor as
an exact integer pair plus its FP32 fraction, which is the only correct
shape for a projected source: that route selects its donor in FP64, and a
local coordinate just below an integer can advance its donor once it is
rounded to FP32.  The second entry may be absent from an older staged
library; :attr:`CpuPreprocessBackend.indexed_donor_interp` reports that,
and the projected caller keeps its NumPy path for exactly that case.

The masked surface fields (soil moisture and temperature, snow, skin
temperature, sea ice) take WPS metgrid's masked chain through
``gpuwm_wps_masked_chain_f64``, in float64 and byte-identical to the NumPy
transcription kept as its test oracle
(``gpuwm/verify/wps_masked_oracle.py``).  Both preprocessing backends call
it: there is no NumPy route for these fields at run time, so a library
that predates the entry is refused by name with the remedy
(:meth:`CpuPreprocessBackend.require_wps_masked_chain`).
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
from typing import Final

import numpy as np


CPU_BACKEND_ABI: Final[int] = 1
CPU_BRIDGE_ENV: Final[str] = "GPUWM_CPU_PREPROCESS_BRIDGE"
_selected_cpu_bridge = ContextVar("selected_preparation_cpu_bridge", default=None)
_selected_cpu_worker_cap = ContextVar("selected_preparation_cpu_worker_cap", default=None)


@contextmanager
def cpu_bridge_scope(path, *, worker_cap=None):
    """Keep one preparation's selected bridge for its default native helpers."""
    if path is None:
        yield
        return
    token = _selected_cpu_bridge.set(resolve_cpu_bridge(path))
    worker_token = _selected_cpu_worker_cap.set(worker_cap)
    try:
        yield
    finally:
        _selected_cpu_bridge.reset(token)
        _selected_cpu_worker_cap.reset(worker_token)

_ERRORS = {
    1: "null buffer",
    2: "invalid dimensions or option",
    3: "non-finite value or invalid pressure",
    4: "source pressure is not strictly descending",
    5: "no source level is above the surface",
    6: "target pressure lies above the source top",
    7: "no interpolation window fits the assembled column",
    8: "input file could not be opened or read",
    9: "input file is not the declared intermediate format",
    10: "target point escapes the source grid",
    11: "unknown WPS interpolation operator",
    30: "a lake target's search found no source water anywhere",
    31: "a bilinear corner lies outside the source grid",
    32: "a water-body label is negative or above the bodies declared",
    127: "native CPU backend panicked",
}

#: The masked-chain entry, and the code of each WPS operator it runs.  A
#: name this table does not hold travels as 255 and is refused by name
#: when a waiting target reaches it, never before (NumPy's order).
WPS_MASKED_CHAIN_ENTRY: Final[str] = "gpuwm_wps_masked_chain_f64"
WPS_LAND_UNIT_SCAN_ENTRY: Final[str] = "gpuwm_wps_land_unit_scan_f64"
WPS_MASKED_CHAIN_IMPLEMENTATION: Final[str] = "rust-wps-masked-chain-f64-v2"
#: The native HRRR route's soil stencil (build, then apply).
MASKED_STENCIL_ENTRY: Final[str] = "gpuwm_masked_bilinear_stencil_f64"
MASKED_STENCIL_APPLY_ENTRY: Final[str] = "gpuwm_masked_stencil_apply_f32"
MASKED_STENCIL_IMPLEMENTATION: Final[str] = "rust-masked-bilinear-stencil-f64-v1"
#: Count slots of the stencil entry's report.
MASKED_STENCIL_COUNT_SLOTS: Final[int] = 10
#: The lake skin search and the water-temperature blends, required by
#: every preparation that assembles a water temperature or searches a
#: lake's skin temperature.
LAKE_WATER_NEAREST_ENTRY: Final[str] = "gpuwm_lake_water_nearest_f64"
WATER_BLEND_ENTRY: Final[str] = "gpuwm_masked_bilinear_blend_f64"
COMPONENT_FILL_ENTRY: Final[str] = "gpuwm_component_fill_f64"
OVERLAY_SAMPLE_ENTRY: Final[str] = "gpuwm_overlay_bilinear_sample_f64"
LABEL_COMPONENTS_ENTRY: Final[str] = "gpuwm_label_components_8"
#: The box repairs of inadmissible water temperature and the per-body
#: assembly, required with the blends.
WATER_REPAIR_ENTRY: Final[str] = "gpuwm_water_repair_f64"
WATER_BODIES_ENTRY: Final[str] = "gpuwm_water_bodies_f64"
#: Which target water body owns each source cell (the donor sets of the
#: per-body assembly).
COMPONENT_OWNER_ENTRY: Final[str] = "gpuwm_component_owner_f64"
#: The bounded surface-nearest search of the CPU preprocessing backend:
#: the newest entry of this library.
MASKED_NEAREST_ENTRY: Final[str] = "gpuwm_masked_nearest_f32"
#: Every entry the water surfaces call, in the order they were built; a
#: library missing any of them is refused naming exactly those.
WATER_ENTRIES: Final[tuple[str, ...]] = (
    LAKE_WATER_NEAREST_ENTRY, WATER_BLEND_ENTRY, COMPONENT_FILL_ENTRY,
    OVERLAY_SAMPLE_ENTRY, LABEL_COMPONENTS_ENTRY, WATER_REPAIR_ENTRY,
    WATER_BODIES_ENTRY, COMPONENT_OWNER_ENTRY)
WATER_BLEND_IMPLEMENTATION: Final[str] = "rust-water-blend-f64-v3"
#: ``surface`` codes of the surface-nearest entry.
_NEAREST_SURFACES = {"match": 0, "land": 1, "water": 2}
#: Count slots per body of the per-body assembly entry.
WATER_BODY_STAT_SLOTS: Final[int] = 7
_WPS_OPERATOR_CODES = {
    "sixteen_pt": 0, "four_pt": 1, "average_4pt": 2,
    "wt_average_4pt": 3, "wt_average_16pt": 4, "search": 5,
    "nearest_neighbor": 6,
}
_WPS_UNKNOWN_OPERATOR = 255
_WPS_CHAIN_MODES = {"plain": 0, "land": 1, "skin": 2}
#: Count slots per layer the entry writes.
WPS_CHAIN_COUNT_SLOTS: Final[int] = 8


class MaskedChainUnavailable(RuntimeError):
    """The CPU preprocessing library cannot run the masked surface chain."""


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

    from gpuwm.bridges import cpu_bridge_remedy, find_artifact

    if path is None:
        path = _selected_cpu_bridge.get()
    filename = _library_names()[0]
    if path is not None:
        explicit = Path(path)
        if explicit.is_file():
            return explicit.resolve()
        raise FileNotFoundError(
            "GPUWM parallel CPU preprocessing bridge was not found; "
            f"searched:\n  {explicit}\n"
            # An explicit path was passed, so the remedy is that path,
            # not a download: staging a correct copy somewhere else
            # would not make THIS argument true.
            "  # that path was given explicitly, so nothing else was "
            "searched;\n"
            "  # drop the explicit path to use the resolution ladder, or "
            "point it\n"
            "  # at a real build.  `gpuwm doctor` reports the estate.")
    found = find_artifact(CPU_BRIDGE_ENV, filename)
    if found is not None:
        return found
    rendered = "\n  ".join(
        str(candidate) for candidate in cpu_bridge_candidates())
    # THE remedy this refusal used to lack.  It listed four paths and
    # stopped -- the only resolver in the estate whose message never said
    # how to fix it -- while the library it wants ships in the bundle
    # `gpuwm fetch-bridges` stages into the last path on that list.
    raise FileNotFoundError(
        "GPUWM parallel CPU preprocessing bridge was not found; searched:\n  "
        + rendered + "\n" + cpu_bridge_remedy(filename))


#: The reference width of the saved host-RAM calibration. Runtime uses
#: the CPU and memory budget, with extra worker scratch priced in preflight.
AUTOMATIC_PREPARATION_WORKERS: Final[int] = 8


def available_cpu_count() -> int:
    """The CPU capacity allowed by affinity and every cgroup quota."""
    from gpuwm.ingest.preparation_workers import cpu_budget
    return cpu_budget()["available_cpus"]


def automatic_workers() -> int:
    """Threads a CPU preparation uses when no count was given."""
    from gpuwm.ingest.preparation_workers import PREPARATION_THREADS_ENV, effective_workers
    configured = os.environ.get(PREPARATION_THREADS_ENV, "")
    return effective_workers(int(configured)) if configured.isdecimal() and int(configured) > 0 else effective_workers()


def host_step_workers(backend=None) -> int:
    """The threads a preparation's host steps run on.

    ``backend`` is the resolved preprocessing backend of the preparation:
    its ``host_step_workers`` (``--preprocess-workers``, or the backend's
    own automatic count) when it has one, else :func:`automatic_workers`.
    The water surfaces and the masked chain take the same count, so an
    explicit ``--preprocess-workers`` reaches every Rust host step.
    """
    bound = getattr(backend, "host_step_workers", None)
    if bound is None:
        return automatic_workers()
    return int(bound)


def _workers(value: int | None, independent_count: int) -> int:
    if value is None:
        value = automatic_workers()
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError("workers must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError("workers must be positive")
    from gpuwm.ingest.preparation_workers import effective_workers
    value = min(effective_workers(value), independent_count)
    cap = _selected_cpu_worker_cap.get()
    return value if cap is None else min(value, cap)


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
        from gpuwm.ingest.interpolation_support import regular_source_support
        self._source_support = regular_source_support(
            self.source_shape, self.y, self.x)

    def apply(self, field, method: str = "parabolic", *,
              workers: int | None = None,
              source_support: bool = False) -> np.ndarray:
        """Apply this geometry while preserving leading field dimensions."""

        methods = {"nearest": 0, "bilinear": 1, "parabolic": 2}
        if method not in methods:
            raise ValueError(
                "method must be 'nearest', 'bilinear', or 'parabolic'")
        support = self._source_support if source_support and method != "nearest" else None
        shape, y, x = self.source_shape, self.y, self.x
        if support is not None:
            field = support.crop(field, self.source_shape)
            shape, y, x = support.shape, support.y, support.x
        source = _host_f32(field)
        if source.ndim < 2 or source.shape[-2:] != shape:
            raise ValueError(
                "field trailing dimensions do not match source axes")
        leading_shape = source.shape[:-2]
        nlead = int(np.prod(leading_shape, dtype=np.int64)) or 1
        source = source.reshape((nlead, *shape))
        output = np.empty((nlead, y.size), dtype=np.float32)
        count = _workers(workers, y.size)
        code = int(self.backend._library.gpuwm_regular_interp_f32(
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(y.ctypes.data),
            ctypes.c_void_p(x.ctypes.data),
            ctypes.c_void_p(output.ctypes.data),
            nlead, shape[0], shape[1], y.size,
            methods[method], count,
        ))
        self.backend._raise(code, "horizontal interpolation")
        return output.reshape((*leading_shape, *self.target_shape))


class _CpuIndexedDonorPlan:
    """Exact-integer-donor plan for a projected (non-lat/lon) source.

    :class:`_CpuRegularPlan` hands the library a fractional source
    coordinate and lets it derive the donor.  A projected source cannot
    do that: it selects the donor in FP64, and a local coordinate just
    below an integer can advance its donor once it is rounded to FP32.
    So this plan carries the donor as an integer pair and the FP32
    fraction separately, all the way to the kernel, and neither is ever
    re-derived from the other.
    """

    #: Nearest is absent on purpose: it reads a DIFFERENT donor pair
    #: (round-to-nearest, not floor), so it is a plain gather the caller
    #: already does exactly in NumPy for well under a percent of this
    #: operator's wall time.
    _METHODS = {"bilinear": 1, "parabolic": 2}

    def __init__(self, backend: "CpuPreprocessBackend", source_shape,
                 donor_y, donor_x, fraction_y, fraction_x):
        if not backend.indexed_donor_interp:
            raise RuntimeError(
                "this CPU preprocessing bridge predates "
                "gpuwm_indexed_interp_f32; rebuild tools/grib1_bridge")
        try:
            source_shape = tuple(int(value) for value in source_shape)
        except (TypeError, ValueError) as exc:
            raise TypeError("source_shape must contain two integers") from exc
        if len(source_shape) != 2 or min(source_shape) < 2:
            raise ValueError(
                "source_shape must contain two dimensions of at least two")
        donor_y = np.asarray(donor_y)
        donor_x = np.asarray(donor_x)
        fraction_y = np.asarray(fraction_y)
        fraction_x = np.asarray(fraction_x)
        shapes = {donor_y.shape, donor_x.shape,
                  fraction_y.shape, fraction_x.shape}
        if len(shapes) != 1 or donor_y.size == 0:
            raise ValueError(
                "donor indices and fractions must be non-empty and "
                "equal-shaped")
        if not np.isfinite(fraction_y).all() or not np.isfinite(fraction_x).all():
            raise ValueError("target fractions must be finite")
        self.backend = backend
        self.source_shape = source_shape
        self.target_shape = tuple(map(int, donor_y.shape))
        self.donor_y = np.ascontiguousarray(donor_y, dtype=np.int32)
        self.donor_x = np.ascontiguousarray(donor_x, dtype=np.int32)
        self.fraction_y = np.ascontiguousarray(fraction_y, dtype=np.float32)
        self.fraction_x = np.ascontiguousarray(fraction_x, dtype=np.float32)

    def apply(self, field, method: str = "parabolic", *,
              workers: int | None = None,
              source_support: bool = False) -> np.ndarray:
        """Apply the distinct indexed-donor arithmetic on the full source."""

        if method not in self._METHODS:
            raise ValueError(
                "method must be 'bilinear' or 'parabolic'; 'nearest' is an "
                "exact gather on the caller's own nearest-donor indices")
        source = _host_f32(field)
        if source.ndim < 2 or source.shape[-2:] != self.source_shape:
            raise ValueError(
                "field trailing dimensions do not match source axes")
        leading_shape = source.shape[:-2]
        nlead = int(np.prod(leading_shape, dtype=np.int64)) or 1
        source = np.ascontiguousarray(
            source.reshape((nlead, *self.source_shape)))
        ntarget = int(self.donor_y.size)
        output = np.empty((nlead, ntarget), dtype=np.float32)
        count = _workers(workers, ntarget)
        code = int(self.backend._library.gpuwm_indexed_interp_f32(
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(self.donor_y.ctypes.data),
            ctypes.c_void_p(self.donor_x.ctypes.data),
            ctypes.c_void_p(self.fraction_y.ctypes.data),
            ctypes.c_void_p(self.fraction_x.ctypes.data),
            ctypes.c_void_p(output.ctypes.data),
            nlead, self.source_shape[0], self.source_shape[1], ntarget,
            self._METHODS[method], count,
        ))
        self.backend._raise(code, "indexed-donor horizontal interpolation")
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
        # The static-dataset entries are looked up the same way the
        # indexed-donor entry below is, and for the same reason.
        self.wps_intermediate_reader = False
        self.cyclic_bilinear = False
        try:
            inventory = library.gpuwm_wps_intermediate_inventory
            read = library.gpuwm_wps_intermediate_read
            message = library.gpuwm_bridge_last_error
        except AttributeError:
            pass
        else:
            inventory.argtypes = [pointer, size, pointer, pointer]
            inventory.restype = ctypes.c_int32
            read.argtypes = [
                pointer, size, pointer, pointer, pointer,
                ctypes.c_uint64, ctypes.c_uint64,
            ]
            read.restype = ctypes.c_int32
            message.argtypes = [pointer, size]
            message.restype = size
            self.wps_intermediate_reader = True
        try:
            cyclic = library.gpuwm_regular_cyclic_bilinear_f32
        except AttributeError:
            pass
        else:
            cyclic.argtypes = [
                pointer, pointer, pointer, pointer,
                size, size, size, size,
                ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, size,
            ]
            cyclic.restype = ctypes.c_int32
            self.cyclic_bilinear = True

        # The masked surface chain: looked up like the entries above, and
        # REQUIRED by every preparation that maps a masked field, so its
        # absence is refused by name at the first such field
        # (require_wps_masked_chain) instead of being a capability answer.
        self.wps_masked_chain_entry = False
        try:
            chain = library.gpuwm_wps_masked_chain_f64
            scan = library.gpuwm_wps_land_unit_scan_f64
        except AttributeError:
            pass
        else:
            chain.argtypes = [
                pointer, pointer, pointer, pointer, pointer, pointer,
                pointer, size, ctypes.c_int32, ctypes.c_double,
                ctypes.c_int32, ctypes.c_double, ctypes.c_double,
                pointer, pointer, pointer, size, size, size, size, size,
            ]
            chain.restype = ctypes.c_int32
            scan.argtypes = [
                pointer, pointer, pointer, size, size, size,
                ctypes.c_double, ctypes.c_double, pointer, pointer, size,
            ]
            scan.restype = ctypes.c_int32
            self.wps_masked_chain_entry = True

        # The native HRRR route's soil stencil: looked up the same way and
        # required the same way (require_masked_stencil).
        self.masked_stencil_entry = False
        try:
            stencil = library.gpuwm_masked_bilinear_stencil_f64
            stencil_apply = library.gpuwm_masked_stencil_apply_f32
        except AttributeError:
            pass
        else:
            stencil.argtypes = [
                pointer, pointer, pointer, size, pointer, size, size,
                ctypes.c_int64, ctypes.c_uint32, ctypes.c_int32,
                ctypes.c_double, size,
                pointer, pointer, pointer, pointer, pointer, pointer,
                pointer, size, pointer, pointer, pointer, pointer, pointer,
                size,
            ]
            stencil.restype = ctypes.c_int32
            stencil_apply.argtypes = [
                pointer, size, size, size, pointer, pointer, pointer, size,
                pointer, ctypes.c_int32, ctypes.c_float, pointer, pointer,
                size,
            ]
            stencil_apply.restype = ctypes.c_int32
            self.masked_stencil_entry = True

        # The lake skin search and the water-temperature blends: looked up
        # the same way and required the same way (require_water_blends),
        # each by name, so a refusal names exactly the entries missing.
        self.water_blend_entry = False
        self.water_blend_missing = tuple(
            name for name in WATER_ENTRIES if not hasattr(library, name))
        if not self.water_blend_missing:
            nearest = library.gpuwm_lake_water_nearest_f64
            blend = library.gpuwm_masked_bilinear_blend_f64
            fill = library.gpuwm_component_fill_f64
            sample = library.gpuwm_overlay_bilinear_sample_f64
            label = library.gpuwm_label_components_8
            repair = library.gpuwm_water_repair_f64
            bodies = library.gpuwm_water_bodies_f64
            owner = library.gpuwm_component_owner_f64
            nearest.argtypes = [
                pointer, pointer, size, size, pointer, pointer, size,
                pointer, size,
            ]
            nearest.restype = ctypes.c_int32
            blend.argtypes = [
                pointer, pointer, size, size, pointer, pointer, pointer,
                size, size, ctypes.c_double, pointer, size,
            ]
            blend.restype = ctypes.c_int32
            fill.argtypes = [pointer, pointer, size, size, size, pointer, size]
            fill.restype = ctypes.c_int32
            sample.argtypes = [
                pointer, pointer, size, size, pointer, pointer, pointer,
                pointer, pointer, size, pointer, pointer, size,
            ]
            sample.restype = ctypes.c_int32
            label.argtypes = [pointer, size, size, pointer, pointer]
            label.restype = ctypes.c_int32
            repair.argtypes = [
                pointer, pointer, pointer, pointer, size, size,
                ctypes.c_double, ctypes.c_double, ctypes.c_int8,
                ctypes.c_int8, pointer, pointer, size,
            ]
            repair.restype = ctypes.c_int32
            bodies.argtypes = [
                pointer, size, size, size, pointer, pointer, pointer,
                pointer, pointer, size, size, pointer, pointer, pointer,
                size, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, size, pointer, pointer, pointer, pointer,
                pointer, pointer, size, pointer, size,
            ]
            bodies.restype = ctypes.c_int32
            owner.argtypes = [
                pointer, pointer, pointer, size, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, size,
                size, pointer, size,
            ]
            owner.restype = ctypes.c_int32
            self.water_blend_entry = True
        self.masked_nearest_entry = False
        try:
            masked_nearest = library.gpuwm_masked_nearest_f32
        except AttributeError:
            pass
        else:
            masked_nearest.argtypes = [
                pointer, pointer, size, size, pointer, pointer, pointer,
                size, ctypes.c_int32, ctypes.c_double, size, pointer,
                pointer, size,
            ]
            masked_nearest.restype = ctypes.c_int32
            self.masked_nearest_entry = True

        # The indexed-donor entry is looked UP, not versioned in.  The ABI
        # integer above describes the shape of the calls that already
        # existed, and bumping it to advertise an addition would refuse
        # every correctly-built older library over a call it was never
        # asked to make -- including a staged bridges bundle older than
        # the checkout driving it.  Absence is a capability answer here,
        # and the caller keeps the NumPy path for exactly that case.
        try:
            indexed = library.gpuwm_indexed_interp_f32
        except AttributeError:
            self.indexed_donor_interp = False
            return
        indexed.argtypes = [
            pointer, pointer, pointer, pointer, pointer, pointer,
            size, size, size, size, ctypes.c_int32, size,
        ]
        indexed.restype = ctypes.c_int32
        self.indexed_donor_interp = True

    def generate_wrf_eta(self, e_vert: int, *, auto_levels_opt=2,
                         p_top=5000.0, max_dz=1000.0, dzbot=50.0,
                         dzstretch_s=1.3, dzstretch_u=1.1,
                         base_temp=290.0) -> np.ndarray:
        """Materialize WRF's automatic full eta levels with its REAL arithmetic.

        Explicit eta arrays bypass this function. This additive symbol leaves
        all existing ABI-v1 interpolation callers valid with older libraries.
        """
        for name, value in (("e_vert", e_vert), ("auto_levels_opt", auto_levels_opt)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if e_vert < 3:
            raise ValueError("automatic eta generation requires e_vert >= 3")
        if auto_levels_opt not in (1, 2):
            raise ValueError("auto_levels_opt must be 1 or 2")
        try:
            call = self._library.gpuwm_wrf_eta_f32
        except AttributeError as exc:
            raise ValueError("CPU preprocessing bridge lacks WRF eta generation; rebuild tools/grib1_bridge or install the matching native bridges") from exc
        call.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32,
                         *([ctypes.c_float] * 6), ctypes.c_void_p, ctypes.c_size_t]
        call.restype = ctypes.c_int32
        out = np.empty(e_vert, dtype=np.float32)
        message = ctypes.create_string_buffer(512)
        code = int(call(out.ctypes.data, e_vert, auto_levels_opt, p_top,
                        max_dz, dzbot, dzstretch_s, dzstretch_u, base_temp,
                        ctypes.addressof(message), len(message)))
        if code:
            reason = message.value.decode("utf-8", "replace")
            raise ValueError(reason or f"WRF eta generation failed with native code {code}")
        return out

    def surface_pressure_from_sea_level(self, pressure, height, terrain, slp, *, workers=1):
        """WRF sfcprs3, parallel by column with strict REAL arithmetic.

        Accept setup's existing f64 storage. The native routine casts each
        scalar in place and follows an index permutation, avoiding two full
        f32 copies and a reordered copy of both atmospheric profiles.
        """
        pressure, height, terrain, slp = (
            np.require(value, dtype=np.float64, requirements=("C", "A"))
            for value in (pressure, height, terrain, slp))
        if pressure.ndim != 3 or height.shape != pressure.shape:
            raise ValueError("sfcprs3 requires matching level/row/column pressure and GHT arrays")
        if pressure.shape[0] < 2 or not all(pressure.shape):
            raise ValueError("sfcprs3 requires at least two profile levels and a nonempty grid")
        if terrain.shape != pressure.shape[1:] or slp.shape != terrain.shape:
            raise ValueError("sfcprs3 terrain and PMSL must match the profile's horizontal grid")
        try:
            call = self._library.gpuwm_wrf_sfcprs3_from_f64
        except AttributeError as exc:
            raise ValueError("CPU preprocessing bridge lacks WRF sea-level pressure reconstruction; rebuild tools/grib1_bridge or install the matching native bridges") from exc
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        call.argtypes = [pointer] * 6 + [size] * 3 + [pointer]
        call.restype = ctypes.c_int32
        order = np.argsort(-pressure[:, 0, 0]).astype(np.uint64)
        out = np.empty(terrain.shape, dtype=np.float32)
        failed_column = size()
        code = int(call(height.ctypes.data, pressure.ctypes.data, terrain.ctypes.data,
                        slp.ctypes.data, order.ctypes.data, out.ctypes.data,
                        pressure.shape[0], terrain.size, _workers(workers, terrain.size),
                        ctypes.addressof(failed_column)))
        if code:
            if failed_column.value < terrain.size:
                row, column = divmod(failed_column.value, terrain.shape[1])
                position = f" at row {row}, column {column}"
            else:
                position = ""
            reason = _ERRORS.get(code, f"native error {code}")
            raise ValueError(f"WRF sfcprs3 pressure reconstruction{position}: {reason}")
        return out

    def _native_message(self) -> str:
        """The sentence behind the last nonzero static-dataset return."""

        if not self.wps_intermediate_reader:
            return ""
        buffer = ctypes.create_string_buffer(1024)
        written = int(self._library.gpuwm_bridge_last_error(
            ctypes.c_void_p(ctypes.addressof(buffer)), len(buffer)))
        return buffer.raw[:written].decode("utf-8", "replace")

    def _raise_native(self, code: int, operation: str) -> None:
        """Raise with the library's own sentence when it has one.

        Codes below 8 describe a shape the CALLER got wrong; 8, 9 and 10
        describe the INPUT, and an integer cannot say which version a
        rejected file declared or which record ran short.  A refusal that
        cannot name its breakage is not a refusal, so those three carry
        the native message.
        """

        if not code:
            return
        detail = _ERRORS.get(code, f"unknown native error {code}")
        message = self._native_message() if code in (8, 9, 10) else ""
        if message:
            raise ValueError(
                f"parallel CPU {operation} failed: {detail}: {message}")
        raise ValueError(f"parallel CPU {operation} failed: {detail}")

    def aerosol_surface_mass(self, number, phb, inverse_density, dx, dy):
        """Operational WRF's REAL surface emission from monthly number."""
        try:
            entry = self._library.gpuwm_aerosol_surface_mass_f32
        except AttributeError:
            raise RuntimeError("CPU bridge lacks aerosol surface emission; rebuild the preprocessing bridge") from None
        arrays = [_host_f32(value) for value in
                  (number, phb[0], phb[1], inverse_density)]
        if not arrays[0].ndim == 2 or any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("surface aerosol operands must share the mass grid")
        output = np.empty_like(arrays[0])
        entry.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_size_t] + [ctypes.c_float] * 3
        entry.restype = ctypes.c_int32
        code = entry(*(a.ctypes.data for a in arrays), output.ctypes.data,
                     output.size, 9.81, float(dx), float(dy))
        self._raise_native(code, "aerosol surface emission")
        return output

    def read_wps_intermediate(self, path):
        """Decode every field record of a WPS intermediate (IFV=5) file.

        Returns ``(records, data)``: ``records`` is a list of per-record
        metadata dicts in file order -- ``field``, ``xlvl``, ``nx``,
        ``ny``, ``iproj``, ``startlat``, ``startlon``, ``deltalat``,
        ``deltalon``, ``offset`` -- and ``data`` is the concatenated FP32
        payload, x fastest.  Nothing here knows what a field MEANS:
        selection and stacking are the caller's table work, so the same
        reader serves ungrib output, ``met_intermediate`` output and a
        ``constants_name`` static dataset alike.
        """

        if not self.wps_intermediate_reader:
            raise RuntimeError(
                "this CPU preprocessing bridge predates "
                "gpuwm_wps_intermediate_read; rebuild tools/grib1_bridge")
        encoded = str(path).encode("utf-8")
        buffer = ctypes.create_string_buffer(encoded)
        n_records = ctypes.c_uint64(0)
        n_points = ctypes.c_uint64(0)
        code = int(self._library.gpuwm_wps_intermediate_inventory(
            ctypes.c_void_p(ctypes.addressof(buffer)), len(encoded),
            ctypes.byref(n_records), ctypes.byref(n_points)))
        self._raise_native(code, "WPS intermediate inventory")
        records = int(n_records.value)
        points = int(n_points.value)
        if records == 0:
            raise ValueError(
                f"{path}: the WPS intermediate file holds no field "
                "records; reading zero records as success would hand the "
                "caller a silently unpopulated grid")
        names = np.empty(records * 9, dtype=np.uint8)
        meta = np.empty(records * 8, dtype=np.float64)
        data = np.empty(points, dtype=np.float32)
        code = int(self._library.gpuwm_wps_intermediate_read(
            ctypes.c_void_p(ctypes.addressof(buffer)), len(encoded),
            ctypes.c_void_p(names.ctypes.data),
            ctypes.c_void_p(meta.ctypes.data),
            ctypes.c_void_p(data.ctypes.data),
            ctypes.c_uint64(records), ctypes.c_uint64(points)))
        self._raise_native(code, "WPS intermediate read")
        out = []
        offset = 0
        raw_names = names.reshape(records, 9).tobytes()
        meta = meta.reshape(records, 8)
        for index in range(records):
            field = raw_names[index * 9:(index + 1) * 9]
            field = field.decode("ascii", "replace").strip()
            nx = int(meta[index, 1])
            ny = int(meta[index, 2])
            out.append({
                "field": field,
                "xlvl": float(meta[index, 0]),
                "nx": nx, "ny": ny,
                "iproj": int(meta[index, 3]),
                "startlat": float(meta[index, 4]),
                "startlon": float(meta[index, 5]),
                "deltalat": float(meta[index, 6]),
                "deltalon": float(meta[index, 7]),
                "offset": offset,
            })
            offset += ny * nx
        return out, data

    def interpolate_regular_cyclic(
            self, field, target_lat, target_lon, *, startlat, deltalat,
            startlon, deltalon, workers: int | None = None) -> np.ndarray:
        """Bilinear from a regular lat/lon source that may be global in x.

        :meth:`interpolate_regular` derives its donor with an FP32 floor
        and clamps the last column, which is right for a bounded source
        and wrong at a GLOBAL source's seam -- a target between the last
        and first columns must take its second donor from column 0.  This
        entry is the global-capable operator, and it decides cyclicity
        from the source's own declared axis span, not from a flag.

        Arithmetic is the tree's
        ``canonical-f32-coordinate-f64-bilinear-single-round-v1`` policy:
        coordinates and weights in FP64, one round to FP32 at the end.
        """

        if not self.cyclic_bilinear:
            raise RuntimeError(
                "this CPU preprocessing bridge predates "
                "gpuwm_regular_cyclic_bilinear_f32; rebuild "
                "tools/grib1_bridge")
        source = _host_f32(field)
        if source.ndim < 2 or min(source.shape[-2:]) < 2:
            raise ValueError(
                "field must carry a source grid of at least 2x2")
        latitudes = np.ascontiguousarray(
            np.asarray(target_lat, dtype=np.float64).ravel())
        longitudes = np.ascontiguousarray(
            np.asarray(target_lon, dtype=np.float64).ravel())
        if latitudes.shape != longitudes.shape or latitudes.size == 0:
            raise ValueError(
                "target latitude/longitude must be non-empty and "
                "equal-shaped")
        target_shape = np.asarray(target_lat).shape
        source_ny, source_nx = int(source.shape[-2]), int(source.shape[-1])
        leading_shape = source.shape[:-2]
        nlead = int(np.prod(leading_shape, dtype=np.int64)) or 1
        source = np.ascontiguousarray(
            source.reshape((nlead, source_ny, source_nx)))
        output = np.empty((nlead, latitudes.size), dtype=np.float32)
        count = _workers(workers, latitudes.size)
        code = int(self._library.gpuwm_regular_cyclic_bilinear_f32(
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(latitudes.ctypes.data),
            ctypes.c_void_p(longitudes.ctypes.data),
            ctypes.c_void_p(output.ctypes.data),
            nlead, source_ny, source_nx, latitudes.size,
            ctypes.c_double(float(startlat)),
            ctypes.c_double(float(deltalat)),
            ctypes.c_double(float(startlon)),
            ctypes.c_double(float(deltalon)),
            count,
        ))
        self._raise_native(code, "cyclic horizontal interpolation")
        return output.reshape((*leading_shape, *target_shape))

    def require_wps_masked_chain(self) -> None:
        """Refuse, by name and with the remedy, a library without the chain.

        The masked surface fields have no other route: the NumPy chain
        this entry replaced is a test oracle now, so a library built
        before the entry would leave soil, snow, skin temperature and sea
        ice unmapped.  The remedy is the one every missing CPU library
        gets (:func:`gpuwm.bridges.cpu_bridge_remedy`).
        """

        if self.wps_masked_chain_entry:
            return
        from gpuwm.bridges import cpu_bridge_remedy

        raise MaskedChainUnavailable(
            f"the CPU preprocessing library at {self.path} predates "
            f"{WPS_MASKED_CHAIN_ENTRY}, which maps every masked surface "
            "field (soil moisture and temperature, snow, skin temperature "
            "and sea ice), so no source with a land-sea mask can be "
            "prepared with it; rebuild or re-fetch it\n"
            + cpu_bridge_remedy(self.path.name))

    def wps_masked_chain(self, layers, donors, partial, target_y, target_x,
                         target_mask, chain, *, mode: str, fill_value,
                         physical_range=None, workers: int | None = None):
        """WPS metgrid's masked chain over every layer of one field.

        ``layers`` is ``(nlayer, ny, nx)`` float64; ``donors`` and
        ``partial`` are ``(ny, nx)`` masks (``partial`` is ignored in the
        ``plain`` mode); ``target_y``/``target_x`` are the zero-based
        fractional source coordinates of the targets and ``target_mask``
        their mask, any matching shapes.  ``mode`` is ``plain`` (one
        call), ``land`` (the land pass with its fractional second chance)
        or ``skin`` (skin temperature on both surfaces).  Returns
        ``(values, counts)``: ``values`` is ``(nlayer, ntarget)`` float64,
        ``counts`` ``(nlayer, 8)`` in the slot order of
        ``tools/grib1_bridge/src/wps_masked.rs``.
        """

        self.require_wps_masked_chain()
        if mode not in _WPS_CHAIN_MODES:
            raise ValueError(f"unknown masked chain mode {mode!r}")
        layers = np.ascontiguousarray(layers, dtype=np.float64)
        if layers.ndim != 3 or layers.shape[0] < 1:
            raise ValueError("masked chain layers must be (layer, y, x)")
        nlayer, ny, nx = (int(value) for value in layers.shape)
        donor_mask = np.ascontiguousarray(donors, dtype=np.bool_)
        if donor_mask.shape != (ny, nx):
            raise ValueError("field and source_valid shapes differ")
        partial_mask = None
        if mode != "plain":
            partial_mask = np.ascontiguousarray(partial, dtype=np.bool_)
            if partial_mask.shape != (ny, nx):
                raise ValueError("field and source_valid shapes differ")
        ty = np.ascontiguousarray(
            np.asarray(target_y, dtype=np.float64).ravel())
        tx = np.ascontiguousarray(
            np.asarray(target_x, dtype=np.float64).ravel())
        mask = np.ascontiguousarray(
            np.asarray(target_mask, dtype=np.bool_).ravel())
        if ty.shape != tx.shape or mask.shape != ty.shape:
            raise ValueError("target_active shape does not match target grid")
        chain = tuple(chain)
        codes = []
        for op in chain:
            try:
                codes.append(_WPS_OPERATOR_CODES.get(op, _WPS_UNKNOWN_OPERATOR))
            except TypeError:
                codes.append(_WPS_UNKNOWN_OPERATOR)
        codes = np.ascontiguousarray(np.array(codes, dtype=np.uint8))
        if physical_range is None:
            has_range, low, high = 0, 0.0, 1.0
        else:
            low, high = (float(bound) for bound in physical_range)
            if not low < high:
                raise ValueError(
                    "physical_range must be (low, high) with low < high")
            has_range = 1
        ntarget = int(ty.size)
        output = np.empty((nlayer, ntarget), dtype=np.float64)
        counts = np.zeros((nlayer, WPS_CHAIN_COUNT_SLOTS), dtype=np.uint64)
        position = ctypes.c_uint64(0)
        count = available_cpu_count() if workers is None else int(workers)
        if count < 1:
            raise ValueError("workers must be positive")

        def address(array):
            return ctypes.c_void_p(
                None if array is None else array.ctypes.data)

        code = int(self._library.gpuwm_wps_masked_chain_f64(
            address(layers), address(donor_mask.view(np.uint8)),
            address(None if partial_mask is None
                    else partial_mask.view(np.uint8)),
            address(ty), address(tx), address(mask.view(np.uint8)),
            address(codes if codes.size else None), int(codes.size),
            _WPS_CHAIN_MODES[mode], float(fill_value), has_range,
            float(low), float(high), address(output), address(counts),
            ctypes.byref(position), nlayer, ny, nx, ntarget, count,
        ))
        if code == 11:
            raise ValueError(
                "unknown WPS interpolation operator "
                f"{chain[int(position.value)]!r}")
        self._raise_native(code, "masked surface interpolation")
        return output, counts

    def land_unit_scan(self, layers, land, partial, physical_range, *,
                       workers: int | None = None):
        """The unit check's numbers for every layer of a bounded land field.

        Returns ``(counts, spans)``: per layer, the donor cells (the
        binarized land, or the fractional land where the source has none),
        how many carry a finite value and how many of those lie inside
        ``physical_range`` widened by packing roundoff; and the least and
        greatest finite donor value.
        """

        self.require_wps_masked_chain()
        layers = np.ascontiguousarray(layers, dtype=np.float64)
        if layers.ndim != 3 or layers.shape[0] < 1:
            raise ValueError("masked chain layers must be (layer, y, x)")
        nlayer, ny, nx = (int(value) for value in layers.shape)
        land = np.ascontiguousarray(land, dtype=np.bool_)
        partial = np.ascontiguousarray(partial, dtype=np.bool_)
        if land.shape != (ny, nx) or partial.shape != (ny, nx):
            raise ValueError("field and source_valid shapes differ")
        low, high = (float(bound) for bound in physical_range)
        counts = np.zeros((nlayer, 3), dtype=np.uint64)
        spans = np.zeros((nlayer, 2), dtype=np.float64)
        count = available_cpu_count() if workers is None else int(workers)
        code = int(self._library.gpuwm_wps_land_unit_scan_f64(
            ctypes.c_void_p(layers.ctypes.data),
            ctypes.c_void_p(land.view(np.uint8).ctypes.data),
            ctypes.c_void_p(partial.view(np.uint8).ctypes.data),
            nlayer, ny, nx, low, high,
            ctypes.c_void_p(counts.ctypes.data),
            ctypes.c_void_p(spans.ctypes.data), max(1, count),
        ))
        self._raise_native(code, "masked surface unit check")
        return counts, spans

    def require_masked_stencil(self) -> None:
        """Refuse, by name and with the remedy, a library without the stencil.

        The native HRRR route maps soil temperature and soil moisture
        through this entry and nothing else: its NumPy builder is a test
        oracle now, so a library built before the entry would leave that
        route's soil unmapped.
        """

        if self.masked_stencil_entry:
            return
        from gpuwm.bridges import cpu_bridge_remedy

        raise MaskedChainUnavailable(
            f"the CPU preprocessing library at {self.path} predates "
            f"{MASKED_STENCIL_ENTRY}, which maps the soil temperature and "
            "soil moisture of the native HRRR route, so that route cannot "
            "be prepared with it; rebuild or re-fetch it\n"
            + cpu_bridge_remedy(self.path.name))

    def masked_bilinear_stencil(self, x, y, source_valid, target_apply, *,
                                fallback_radius: int, closed_edges: int,
                                edges_unknown: bool, distant_cells: float,
                                listed: int, workers: int | None = None):
        """Build the native HRRR route's soil stencil in the Rust library.

        ``x``/``y`` are the targets' zero-based fractional source
        coordinates (any matching shapes), ``source_valid`` the source
        donor mask and ``target_apply`` the targets that take a donor.
        ``closed_edges`` is a bit mask (west 1, east 2, south 4, north 8).
        Returns ``(code, arrays)``: ``code`` is the entry's status (0, or
        a refusal code of ``tools/grib1_bridge/src/masked_stencil.rs``
        for the caller to word) and ``arrays`` its outputs by name, in
        its slot orders.
        """

        self.require_masked_stencil()
        valid = np.ascontiguousarray(source_valid, dtype=np.bool_)
        if valid.ndim != 2:
            raise ValueError("masked-bilinear source_valid must be 2-D")
        ny, nx = (int(value) for value in valid.shape)
        xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
        ys = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        apply = np.ascontiguousarray(
            np.asarray(target_apply, dtype=np.bool_).ravel())
        ntarget = int(xs.size)
        count = available_cpu_count() if workers is None else int(workers)
        if count < 1:
            raise ValueError("workers must be positive")
        listed = int(listed)
        out = {
            "indices_y": np.empty((4, ntarget), dtype=np.int32),
            "indices_x": np.empty((4, ntarget), dtype=np.int32),
            "weights": np.empty((4, ntarget), dtype=np.float32),
            "counts": np.zeros(MASKED_STENCIL_COUNT_SLOTS, dtype=np.uint64),
            "reals": np.zeros(5, dtype=np.float64),
            "flags": np.zeros(2, dtype=np.int32),
            "histogram": np.zeros(ny + nx + 2, dtype=np.uint64),
            "distant_target": np.zeros(max(1, listed), dtype=np.uint64),
            "distant_source": np.zeros((max(1, listed), 2), dtype=np.int64),
            "distant_distance": np.zeros(max(1, listed), dtype=np.float64),
            "distant_reach": np.zeros(max(1, listed), dtype=np.float64),
            "unresolved": np.zeros(max(1, ntarget), dtype=np.uint64),
        }

        def address(array):
            return ctypes.c_void_p(array.ctypes.data)

        code = int(self._library.gpuwm_masked_bilinear_stencil_f64(
            address(xs), address(ys), address(apply.view(np.uint8)),
            ntarget, address(valid.view(np.uint8)), ny, nx,
            int(fallback_radius), int(closed_edges), int(bool(edges_unknown)),
            float(distant_cells), listed,
            address(out["indices_y"]), address(out["indices_x"]),
            address(out["weights"]), address(out["counts"]),
            address(out["reals"]), address(out["flags"]),
            address(out["histogram"]), int(out["histogram"].size),
            address(out["distant_target"]), address(out["distant_source"]),
            address(out["distant_distance"]), address(out["distant_reach"]),
            address(out["unresolved"]), count,
        ))
        if code in (1, 2, 127):
            self._raise_native(code, "masked bilinear stencil")
        return code, out

    def masked_stencil_apply(self, field, indices_y, indices_x, weights, *,
                             select=None, fill=None,
                             workers: int | None = None):
        """Apply a built stencil to every layer of a float32 field.

        ``field`` is ``(..., ny, nx)``; the stencil arrays are ``(4,
        *target_shape)``.  With ``select`` (a target-shape mask), every
        unselected target takes ``fill``: a scalar, or a target-shape
        array used for every layer.  Returns ``(..., *target_shape)``
        float32.
        """

        self.require_masked_stencil()
        field = np.ascontiguousarray(field, dtype=np.float32)
        if field.ndim < 2:
            raise ValueError(
                "HRRR field trailing dimensions do not match window")
        lead_shape = field.shape[:-2]
        ny, nx = (int(value) for value in field.shape[-2:])
        nlayer = int(np.prod(lead_shape, dtype=np.int64)) if lead_shape else 1
        iy = np.ascontiguousarray(indices_y, dtype=np.int32)
        ix = np.ascontiguousarray(indices_x, dtype=np.int32)
        w = np.ascontiguousarray(weights, dtype=np.float32)
        target_shape = iy.shape[1:]
        ntarget = int(np.prod(target_shape, dtype=np.int64))
        mode, scalar, values, chosen = 0, 0.0, None, None
        if select is not None:
            chosen = np.ascontiguousarray(
                np.asarray(select, dtype=np.bool_).ravel())
            if chosen.size != ntarget:
                raise ValueError("select does not match the target grid")
            fill_array = np.asarray(fill)
            if fill_array.ndim == 0:
                mode, scalar = 1, float(np.float32(fill_array))
            else:
                values = np.ascontiguousarray(
                    np.broadcast_to(fill_array, target_shape),
                    dtype=np.float32).ravel()
                mode = 2
        output = np.empty((*lead_shape, *target_shape), dtype=np.float32)
        count = available_cpu_count() if workers is None else int(workers)

        def address(array):
            return ctypes.c_void_p(
                None if array is None else array.ctypes.data)

        code = int(self._library.gpuwm_masked_stencil_apply_f32(
            address(field), nlayer, ny, nx, address(iy), address(ix),
            address(w), ntarget,
            address(None if chosen is None else chosen.view(np.uint8)),
            mode, scalar, address(values), address(output), max(1, count),
        ))
        self._raise_native(code, "masked bilinear stencil apply")
        return output

    def require_water_blends(self) -> None:
        """Refuse, by name and with the remedy, a library without the blends.

        The lake skin search and the water-temperature blends have no
        other route: their NumPy code is a test oracle now
        (``gpuwm/verify/water_blend_oracle.py``), so a library built before
        these entries would leave lake skin temperatures and the water
        temperature of every lake and sea unassembled.
        """

        if self.water_blend_entry:
            return
        from gpuwm.bridges import cpu_bridge_remedy

        missing = getattr(self, "water_blend_missing", None) or WATER_ENTRIES
        raise MaskedChainUnavailable(
            f"the CPU preprocessing library at {self.path} predates "
            f"{', '.join(missing)}, which "
            "assemble and repair the water temperature of lakes and seas "
            "and search a lake's skin temperature, so no source with water "
            "can be prepared with it; rebuild or re-fetch it\n"
            + cpu_bridge_remedy(self.path.name))

    def require_masked_nearest(self) -> None:
        """Refuse, by name and with the remedy, a library without the
        surface-nearest search: its NumPy code is a test oracle now, so the
        CPU backend has no other route for it."""

        if getattr(self, "masked_nearest_entry", False):
            return
        from gpuwm.bridges import cpu_bridge_remedy

        raise MaskedChainUnavailable(
            f"the CPU preprocessing library at {self.path} predates "
            f"{MASKED_NEAREST_ENTRY}, the bounded surface-nearest search of "
            "the CPU preprocessing backend, so that backend cannot search "
            "a masked surface with it; rebuild or re-fetch it\n"
            + cpu_bridge_remedy(self.path.name))

    @staticmethod
    def _water_workers(workers: int | None) -> int:
        count = automatic_workers() if workers is None else int(workers)
        if count < 1:
            raise ValueError("workers must be positive")
        return count

    def lake_water_nearest(self, skin, water, target_y, target_x, *,
                           workers: int | None = None) -> np.ndarray:
        """The nearest source water cell's ``skin`` for every target.

        ``skin`` and ``water`` are ``(ny, nx)``; ``target_y``/``target_x``
        the targets' zero-based fractional source coordinates.  Euclidean
        distance in source cells, ties to the first cell in row-major
        order.  A target with no water anywhere in the source is refused.
        """

        self.require_water_blends()
        skin = np.ascontiguousarray(skin, dtype=np.float64)
        water = np.ascontiguousarray(water, dtype=np.bool_)
        if skin.ndim != 2 or water.shape != skin.shape:
            raise ValueError("skin and water must be matching 2-D fields")
        ty = np.ascontiguousarray(
            np.asarray(target_y, dtype=np.float64).ravel())
        tx = np.ascontiguousarray(
            np.asarray(target_x, dtype=np.float64).ravel())
        if ty.shape != tx.shape:
            raise ValueError("target_y and target_x shapes differ")
        output = np.empty(ty.size, dtype=np.float64)
        if ty.size == 0:
            return output
        ny, nx = (int(value) for value in skin.shape)
        code = int(self._library.gpuwm_lake_water_nearest_f64(
            ctypes.c_void_p(skin.ctypes.data),
            ctypes.c_void_p(water.view(np.uint8).ctypes.data), ny, nx,
            ctypes.c_void_p(ty.ctypes.data), ctypes.c_void_p(tx.ctypes.data),
            int(ty.size), ctypes.c_void_p(output.ctypes.data),
            self._water_workers(workers)))
        if code == 30:
            raise RuntimeError(
                "global source-water search lost validated support")
        self._raise_native(code, "lake skin search")
        return output

    def masked_bilinear_blend(self, field, donors, corners, shape, *,
                              denominator_floor=1e-6,
                              workers: int | None = None) -> np.ndarray:
        """Bilinear interpolation renormalised over the donors that exist.

        ``corners`` is a sequence of ``(rows, columns, weights)`` arrays of
        ``shape``, summed in order; a target whose donor weight does not
        exceed ``denominator_floor`` is NaN.
        """

        self.require_water_blends()
        field = np.ascontiguousarray(field, dtype=np.float64)
        donors = np.ascontiguousarray(donors, dtype=np.bool_)
        if field.ndim != 2 or donors.shape != field.shape:
            raise ValueError("field and donors must be matching 2-D fields")
        shape = tuple(int(value) for value in shape)
        ntarget = int(np.prod(shape, dtype=np.int64))
        # Stacked per call and released with it: the per-body assembly
        # (water_bodies) stacks its corner set once per assembly itself,
        # so nothing here outlives the call on the process's shared
        # backend.
        corners = tuple(corners)

        def stacked(slot, dtype):
            if not corners:
                return np.empty((0, ntarget), dtype=dtype)
            return np.ascontiguousarray(np.stack([
                np.broadcast_to(np.asarray(corner[slot], dtype=dtype),
                                shape).ravel()
                for corner in corners]))

        rows = stacked(0, np.int64)
        cols = stacked(1, np.int64)
        weights = stacked(2, np.float64)
        output = np.empty(ntarget, dtype=np.float64)
        ny, nx = (int(value) for value in field.shape)
        code = int(self._library.gpuwm_masked_bilinear_blend_f64(
            ctypes.c_void_p(field.ctypes.data),
            ctypes.c_void_p(donors.view(np.uint8).ctypes.data), ny, nx,
            ctypes.c_void_p(rows.ctypes.data),
            ctypes.c_void_p(cols.ctypes.data),
            ctypes.c_void_p(weights.ctypes.data), int(rows.shape[0]), ntarget,
            float(denominator_floor), ctypes.c_void_p(output.ctypes.data),
            self._water_workers(workers)))
        self._raise_native(code, "masked bilinear water blend")
        return output.reshape(shape)

    def component_fill(self, values, component, *, max_sweeps=1000,
                       workers: int | None = None) -> np.ndarray:
        """Close a component's NaN holes from its own four-neighbours.

        Returns a new float64 field, NaN outside ``component``.
        """

        self.require_water_blends()
        out = np.array(values, dtype=np.float64, copy=True, order="C")
        component = np.ascontiguousarray(component, dtype=np.bool_)
        if out.ndim != 2 or component.shape != out.shape:
            raise ValueError(
                "values and component must be matching 2-D fields")
        out[~component] = np.nan
        ny, nx = (int(value) for value in out.shape)
        sweeps = ctypes.c_uint64(0)
        code = int(self._library.gpuwm_component_fill_f64(
            ctypes.c_void_p(out.ctypes.data),
            ctypes.c_void_p(component.view(np.uint8).ctypes.data), ny, nx,
            max(0, int(max_sweeps)), ctypes.byref(sweeps),
            self._water_workers(workers)))
        self._raise_native(code, "water component fill")
        return out

    def label_components(self, mask):
        """The 8-connected components of ``mask``, labelled 1, 2, ... in
        the row-major order of each one's first cell (0 off the mask).

        Returns ``(labels, count)``: ``labels`` int32 shaped like ``mask``.
        """

        self.require_water_blends()
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        if mask.ndim != 2:
            raise ValueError("mask must be a 2-D field")
        ny, nx = (int(value) for value in mask.shape)
        labels = np.zeros((ny, nx), dtype=np.int32)
        count = ctypes.c_uint64(0)
        code = int(self._library.gpuwm_label_components_8(
            ctypes.c_void_p(mask.view(np.uint8).ctypes.data), ny, nx,
            ctypes.c_void_p(labels.ctypes.data), ctypes.byref(count)))
        self._raise_native(code, "water component labelling")
        return labels, int(count.value)

    def water_repair(self, values, source, water, labels, *, minimum,
                     maximum, nearest_water_code, surrounding_skin_code,
                     workers: int | None = None):
        """Give every water cell its provider left inadmissible a value.

        The box repairs of ``_fill_missing_water_temperature``: from the
        body's own admissible water ring by ring, else the nearest
        admissible water on the domain, else the surrounding skin.
        Returns ``(values, source, counts, filled)``: new float64 and int8
        fields, the ``own_body``/``nearest_water``/``surrounding_skin``
        tallies and the bool mask of repaired cells.
        """

        self.require_water_blends()
        values = np.array(values, dtype=np.float64, copy=True, order="C")
        source = np.array(source, dtype=np.int8, copy=True, order="C")
        water = np.ascontiguousarray(water, dtype=np.bool_)
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        if values.ndim != 2 or not (
                source.shape == water.shape == labels.shape == values.shape):
            raise ValueError(
                "values, source, water and labels must be matching 2-D "
                "fields")
        ny, nx = (int(value) for value in values.shape)
        counts = np.zeros(3, dtype=np.uint64)
        filled = np.zeros((ny, nx), dtype=np.bool_)
        code = int(self._library.gpuwm_water_repair_f64(
            ctypes.c_void_p(values.ctypes.data),
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(water.view(np.uint8).ctypes.data),
            ctypes.c_void_p(labels.ctypes.data), ny, nx, float(minimum),
            float(maximum), int(nearest_water_code),
            int(surrounding_skin_code), ctypes.c_void_p(counts.ctypes.data),
            ctypes.c_void_p(filled.view(np.uint8).ctypes.data),
            self._water_workers(workers)))
        self._raise_native(code, "water-temperature repair")
        tallies = {"own_body": int(counts[0]),
                   "nearest_water": int(counts[1]),
                   "surrounding_skin": int(counts[2])}
        return values, source, tallies, filled

    def water_bodies(self, *, labels, lake_class, skin, values, source,
                     codes, lake_water=None, sst=None, owner=None,
                     corners=None, denominator_floor=1e-6,
                     min_coverage, minimum, maximum, max_sweeps=1000,
                     max_listed, workers: int | None = None):
        """The per-body loop of the water-temperature assembly.

        Bodies ``1..len(lake_class) - 1`` of ``labels``; ``lake_class`` is
        a bool per label (index 0 unused).  ``values`` and ``source`` are
        written in place.  Returns ``(stats, coverage, listed)``: a
        ``(bodies + 1, 7)`` uint64 table of ``cells, donors, provider,
        analysis, component skin, lake water, lake fallback`` (provider 1
        analysis, 2 skin, 3 lake water), the analysis coverage per body
        and the first ``max_listed`` lake fallback cells.
        """

        self.require_water_blends()
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        if labels.ndim != 2:
            raise ValueError("labels must be a 2-D field")
        shape = labels.shape
        ny, nx = (int(value) for value in shape)
        ntarget = ny * nx
        lake_class = np.ascontiguousarray(lake_class, dtype=np.bool_)
        nlabels = int(lake_class.size) - 1
        if nlabels < 0:
            raise ValueError("lake_class must name label 0")
        skin = np.ascontiguousarray(skin, dtype=np.float64)
        for name, array, dtype in (("values", values, np.float64),
                                   ("source", source, np.int8)):
            if (not isinstance(array, np.ndarray) or array.dtype != dtype
                    or array.shape != shape
                    or not array.flags.c_contiguous
                    or not array.flags.writeable):
                raise ValueError(
                    f"{name} must be a writable C-contiguous "
                    f"{np.dtype(dtype).name} field shaped like labels")
        if skin.shape != shape:
            raise ValueError("skin must be shaped like labels")
        keep = [None, None, None]
        if lake_water is not None:
            lake_water = np.ascontiguousarray(lake_water, dtype=np.float64)
            if lake_water.shape != shape:
                raise ValueError("lake_water must be shaped like labels")
        sny = snx = 0
        ncorner = 0
        if sst is not None:
            sst = np.ascontiguousarray(sst, dtype=np.float64)
            owner = np.ascontiguousarray(owner, dtype=np.int32)
            if sst.ndim != 2 or owner.shape != sst.shape:
                raise ValueError("sst and owner must be matching 2-D fields")
            sny, snx = (int(value) for value in sst.shape)
            corners = tuple(corners)
            ncorner = len(corners)

            def stacked(slot, dtype):
                if not corners:
                    return np.empty((0, ntarget), dtype=dtype)
                return np.ascontiguousarray(np.stack([
                    np.broadcast_to(np.asarray(corner[slot], dtype=dtype),
                                    shape).ravel()
                    for corner in corners]))

            keep = [stacked(0, np.int64), stacked(1, np.int64),
                    stacked(2, np.float64)]
        codes = np.ascontiguousarray(codes, dtype=np.int8)
        if codes.shape != (3,):
            raise ValueError("codes must hold analysis, skin and lake codes")
        stats = np.zeros((nlabels + 1, WATER_BODY_STAT_SLOTS),
                         dtype=np.uint64)
        coverage = np.zeros(nlabels + 1, dtype=np.float64)
        max_listed = max(0, int(max_listed))
        listed = np.zeros((max(1, max_listed), 2), dtype=np.int64)
        listed_count = ctypes.c_uint64(0)

        def at(array):
            return (ctypes.c_void_p(None) if array is None
                    else ctypes.c_void_p(array.ctypes.data))

        code = int(self._library.gpuwm_water_bodies_f64(
            ctypes.c_void_p(labels.ctypes.data), ny, nx, nlabels,
            ctypes.c_void_p(lake_class.view(np.uint8).ctypes.data),
            ctypes.c_void_p(skin.ctypes.data), at(lake_water), at(sst),
            at(owner if sst is not None else None), sny, snx,
            at(keep[0]), at(keep[1]), at(keep[2]), ncorner,
            float(denominator_floor), float(min_coverage), float(minimum),
            float(maximum), max(0, int(max_sweeps)),
            ctypes.c_void_p(codes.ctypes.data),
            ctypes.c_void_p(values.ctypes.data),
            ctypes.c_void_p(source.ctypes.data),
            ctypes.c_void_p(stats.ctypes.data),
            ctypes.c_void_p(coverage.ctypes.data),
            ctypes.c_void_p(listed.ctypes.data), max_listed,
            ctypes.byref(listed_count), self._water_workers(workers)))
        self._raise_native(code, "water-body assembly")
        return stats, coverage, listed[:int(listed_count.value)]

    def component_owner(self, labels, source_lat, source_lon, target_lat,
                        target_lon, source_shape, *,
                        workers: int | None = None) -> np.ndarray:
        """Which target water body owns each source cell (0 = none).

        A source cell goes to the body holding most of the targets whose
        nearest source cell it is; among bodies holding equally many, to
        the highest label.  Returns int32 of ``source_shape``.
        """

        self.require_water_blends()
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        tlat = np.ascontiguousarray(target_lat, dtype=np.float64)
        tlon = np.ascontiguousarray(target_lon, dtype=np.float64)
        if not labels.shape == tlat.shape == tlon.shape:
            raise ValueError(
                "labels, target_lat and target_lon must share one shape")
        lat = np.asarray(source_lat, dtype=np.float64).ravel()
        lon = np.asarray(source_lon, dtype=np.float64).ravel()
        if lat.size < 2 or lon.size < 2:
            raise ValueError("source axes must each hold two values")
        sny, snx = (int(value) for value in source_shape)
        owner = np.zeros((sny, snx), dtype=np.int32)
        code = int(self._library.gpuwm_component_owner_f64(
            ctypes.c_void_p(labels.ctypes.data),
            ctypes.c_void_p(tlat.ctypes.data),
            ctypes.c_void_p(tlon.ctypes.data), int(labels.size),
            float(lat[0]), float(lat[1]), float(lon[0]), float(lon[1]),
            sny, snx, ctypes.c_void_p(owner.ctypes.data),
            self._water_workers(workers)))
        self._raise_native(code, "water-body source owner")
        return owner

    def masked_nearest(self, field, source_landmask, target_y, target_x,
                       target_landmask, *, surface, fill_value, radius,
                       workers: int | None = None):
        """The bounded surface-nearest search in float32.

        ``target_y``/``target_x`` are the targets' zero-based source
        coordinates.  Returns ``(values, unmatched)``: float32 values
        shaped like the targets and the number of active targets with no
        source cell of the wanted surface within ``radius``.
        """

        self.require_masked_nearest()
        if surface not in _NEAREST_SURFACES:
            raise ValueError("surface must be 'match', 'land', or 'water'")
        field = np.ascontiguousarray(field, dtype=np.float32)
        source_landmask = np.ascontiguousarray(source_landmask,
                                               dtype=np.bool_)
        ty = np.ascontiguousarray(target_y, dtype=np.float64)
        tx = np.ascontiguousarray(target_x, dtype=np.float64)
        target_landmask = np.ascontiguousarray(target_landmask,
                                               dtype=np.bool_)
        if field.ndim != 2 or source_landmask.shape != field.shape:
            raise ValueError(
                "field and source_landmask must be matching 2-D fields")
        if not ty.shape == tx.shape == target_landmask.shape:
            raise ValueError(
                "target coordinates and target_landmask shapes differ")
        output = np.empty(ty.shape, dtype=np.float32)
        unmatched = ctypes.c_uint64(0)
        ny, nx = (int(value) for value in field.shape)
        code = int(self._library.gpuwm_masked_nearest_f32(
            ctypes.c_void_p(field.ctypes.data),
            ctypes.c_void_p(source_landmask.view(np.uint8).ctypes.data),
            ny, nx, ctypes.c_void_p(ty.ctypes.data),
            ctypes.c_void_p(tx.ctypes.data),
            ctypes.c_void_p(target_landmask.view(np.uint8).ctypes.data),
            int(ty.size), _NEAREST_SURFACES[surface], float(fill_value),
            int(radius), ctypes.c_void_p(output.ctypes.data),
            ctypes.byref(unmatched), self._water_workers(workers)))
        self._raise_native(code, "masked surface-nearest search")
        return output, int(unmatched.value)

    def overlay_bilinear_sample(self, temperature, valid, y0, x0, fy, fx,
                                inside, *, workers: int | None = None):
        """The corner blend of a water-temperature overlay sample.

        Returns ``(values, covered)`` shaped like ``y0``.
        """

        self.require_water_blends()
        temperature = np.ascontiguousarray(temperature, dtype=np.float64)
        valid = np.ascontiguousarray(valid, dtype=np.bool_)
        if temperature.ndim != 2 or valid.shape != temperature.shape:
            raise ValueError(
                "temperature and valid must be matching 2-D fields")
        shape = np.shape(y0)
        arrays = [np.ascontiguousarray(
            np.broadcast_to(np.asarray(array, dtype=dtype), shape).ravel())
            for array, dtype in ((y0, np.int64), (x0, np.int64),
                                 (fy, np.float64), (fx, np.float64),
                                 (inside, np.bool_))]
        ntarget = int(arrays[0].size)
        values = np.empty(ntarget, dtype=np.float64)
        covered = np.zeros(ntarget, dtype=np.bool_)
        ny, nx = (int(value) for value in temperature.shape)
        pointers = [ctypes.c_void_p(
            (array.view(np.uint8) if array.dtype == np.bool_
             else array).ctypes.data) for array in arrays]
        code = int(self._library.gpuwm_overlay_bilinear_sample_f64(
            ctypes.c_void_p(temperature.ctypes.data),
            ctypes.c_void_p(valid.view(np.uint8).ctypes.data), ny, nx,
            *pointers, ntarget, ctypes.c_void_p(values.ctypes.data),
            ctypes.c_void_p(covered.view(np.uint8).ctypes.data),
            self._water_workers(workers)))
        self._raise_native(code, "water-temperature overlay sample")
        return values.reshape(shape), covered.reshape(shape)

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

    def indexed_donor_plan(self, source_shape, donor_y, donor_x,
                           fraction_y, fraction_x) -> _CpuIndexedDonorPlan:
        """Prepare repeated interpolation from exact donors and fractions."""

        return _CpuIndexedDonorPlan(
            self, source_shape, donor_y, donor_x, fraction_y, fraction_x)

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


#: One loaded library per resolved path, for the callers that do not
#: hold a CPU backend of their own (the CUDA backend's masked chain).
#: Each entry keeps its library loaded for the life of the process, as
#: a ``CpuPreprocessBackend`` does for as long as it is held: on Windows
#: the DLL file stays open, so a rebuild or re-fetch that replaces it in
#: place needs a new process (a later ``gpuwm prep`` is one).  Nothing
#: closes it earlier because every preparation in a process maps masked
#: fields through the same library, and a reload per preparation would
#: only repeat the ABI checks.
_SHARED_BACKENDS: dict[Path, CpuPreprocessBackend] = {}


def shared_cpu_backend() -> CpuPreprocessBackend:
    """The library the resolution ladder picks, loaded once per path.

    The same ladder, the same refusals and the same remedy as
    :class:`CpuPreprocessBackend` itself: a missing library raises
    :func:`resolve_cpu_bridge`'s refusal, a stale checkout build the
    checkout-freshness refusal.
    """

    path = resolve_cpu_bridge()
    backend = _SHARED_BACKENDS.get(path)
    if backend is None:
        backend = CpuPreprocessBackend(path)
        _SHARED_BACKENDS[path] = backend
    return backend


def masked_fields_cpu_backend() -> CpuPreprocessBackend:
    """:func:`shared_cpu_backend` for the masked surface fields.

    Under the CUDA backend nothing else in a preparation needs the CPU
    library, so a refusal that only says the library was not found
    leaves the reader asking why a GPU preparation wants it.  A missing
    or unloadable library is refused as :class:`MaskedChainUnavailable`
    whose first line names the fields that map through it under every
    backend, followed by the resolver's own message and remedy.  A stale
    build keeps its own refusal, which already names the rebuild.
    """

    try:
        return shared_cpu_backend()
    except OSError as error:
        raise MaskedChainUnavailable(
            "the masked surface fields (soil moisture and temperature, "
            "snow, skin temperature and sea ice) map through the CPU "
            "preprocessing library under every preprocessing backend, the "
            "CUDA backend included, and that library could not be loaded:\n"
            + str(error)) from error


def water_blend_backend() -> CpuPreprocessBackend:
    """The library the lake skin search and the water blends run in.

    :func:`masked_fields_cpu_backend` (the same library, under every
    preprocessing backend), refused by name with the remedy when it
    predates the water entries: there is no NumPy route.
    """

    native = masked_fields_cpu_backend()
    native.require_water_blends()
    return native


__all__ = [
    "CPU_BACKEND_ABI",
    "CPU_BRIDGE_ENV",
    "COMPONENT_OWNER_ENTRY",
    "CpuPreprocessBackend",
    "LAKE_WATER_NEAREST_ENTRY",
    "MASKED_NEAREST_ENTRY",
    "MASKED_STENCIL_ENTRY",
    "MaskedChainUnavailable",
    "WATER_BLEND_ENTRY",
    "WATER_BODIES_ENTRY",
    "WATER_ENTRIES",
    "WATER_REPAIR_ENTRY",
    "WPS_MASKED_CHAIN_ENTRY",
    "automatic_workers",
    "available_cpu_count",
    "host_step_workers",
    "cpu_bridge_candidates",
    "masked_fields_cpu_backend",
    "resolve_cpu_bridge",
    "shared_cpu_backend",
    "water_blend_backend",
]
