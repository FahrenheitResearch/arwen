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
    8: "input file could not be opened or read",
    9: "input file is not the declared intermediate format",
    10: "target point escapes the source grid",
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

    from gpuwm.bridges import cpu_bridge_remedy, find_artifact

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
              workers: int | None = None) -> np.ndarray:
        """Apply this geometry while preserving leading field dimensions."""

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


__all__ = [
    "CPU_BACKEND_ABI",
    "CPU_BRIDGE_ENV",
    "CpuPreprocessBackend",
    "cpu_bridge_candidates",
    "resolve_cpu_bridge",
]
