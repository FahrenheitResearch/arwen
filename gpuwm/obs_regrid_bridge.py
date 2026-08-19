"""ctypes seam onto the Rust observation remap (``obs-regrid``).

The library is ``tools/rustwx/crates/obs-regrid``, the port target for
:mod:`gpuwm.verify.obs.regrid` -- both registered remap operators, the
bounded neighbour search that used to be ``scipy.spatial.cKDTree``, and
the validity remap that used to be ``numpy.add.at``.  Drew's Python
boundary names "regrid/transform" as data-path processing, and this is
where that half of the observation battery runs now.

Why ctypes rather than pyo3: the same ruling as
:mod:`gpuwm.io.nc_writer_bridge` and :mod:`gpuwm.static.rust_bridge` --
every Rust library gpuwm drives is a cdylib behind ctypes; one loading
discipline, one staging path, one ABI-marker rule, no per-interpreter
builds.

Why it lives HERE and not under ``gpuwm/verify/obs`` beside the module
it serves.  ``gpuwm doctor`` reports on this library by name, so
``gpuwm/doctor.py`` imports this seam -- and doctor ships in the
standalone RW-WPS preprocessing wheel, which stages no verification
package at all.  Measured 2026-08-18: with this module inside
``gpuwm/verify/obs``, ``tools/build_rw_wps_release.py`` refused to stage
the wheel outright, because its verification-boundary scan saw a staged
runtime module importing ``gpuwm.verify.*``.  That boundary is absolute
rather than exemptible (see ``_OBS_EXCLUDES`` in the builder), and every
other cdylib seam in the tree already sits outside it --
:mod:`gpuwm.netcdf_bridge`, :mod:`gpuwm.mapped_engine_bridge`,
:mod:`gpuwm.io.nc_writer_bridge`, :mod:`gpuwm.static.rust_bridge`,
:mod:`gpuwm.obs.dealias_region`.  This one was the lone exception, and
the exception is what broke the build.  The Python fallback and the
parity reference stay in :mod:`gpuwm.verify.obs.regrid`; only the seam
moved.

Default-on contract (Drew's fixed-means-default law): a bare
``python -m tools.obs_battery_score`` builds and applies its remap plans
here.  The pure-Python implementation stays importable as the parity
reference and as an explicit fallback (``GPUWM_OBSREGRID_PYTHON=1``),
and every use of the fallback is a reported workaround, never silent.

**What the fallback announcement names**, because a refusal or a
degradation that does not name its breakage does not exist: on the
scipy path, two source cells exactly equidistant from one destination
cell are resolved by whichever the k-d tree traversal reached first.
Measured over 400 trials scipy answered the lowest flat index 232 times
and some other tied index 168 times (``golden/gen_regrid_goldens.py``,
probe ``ties``).  The battery's premise is that every arm of a case is
remapped by the identical integer array, so no score can differ because
a neighbour search broke a tie differently -- and a plan whose ties turn
on traversal order is exactly that defect, one scipy upgrade away.  The
Rust engine defines the answer instead: the lowest flat source index
wins.  A run that falls back is a run whose remap plan is no longer a
function of its two grids alone, and it says so.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np

#: The C ABI version this module speaks; a library reporting anything
#: else is refused rather than called.
OBSREGRID_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
OBSREGRID_BRIDGE_ENV: Final[str] = "GPUWM_OBSREGRID_BRIDGE"

#: Opt-out to the pure-Python implementation (a reported workaround).
OBSREGRID_PYTHON_ENV: Final[str] = "GPUWM_OBSREGRID_PYTHON"

#: The exported symbol that identifies THIS contract, for
#: :data:`gpuwm.bridges.BRIDGE_ABI_MARKERS`: a build that loads and
#: answers the version probe but predates the plan builder cannot
#: produce a single remap.
ABI_MARKER: Final[bytes] = b"gpuwm_obsregrid_build_plan"

#: Method codes, matching ``plan::Method::from_code`` in the crate.
METHOD_CODES: Final[dict[str, int]] = {"nearest": 0, "cell_average": 1}


class ObsRegridBridgeError(RuntimeError):
    """The Rust remap engine refused, with its own message."""


#: Operations whose pure-Python fallback has already been reported this
#: process (fixed-means-default: a fallback run is a workaround and must
#: say so, once per operation, never silently and never on every call).
_REPORTED_FALLBACKS: set[str] = set()


def route(operation: str):
    """This module when the Rust seam is the active default, else None.

    ``None`` means the caller must run its pure-Python body -- either
    because :data:`OBSREGRID_PYTHON_ENV` explicitly opted out or because
    the library is not loadable -- and the reason has been printed once
    per operation as a WORKAROUND line naming what the degradation
    costs.
    """
    if python_fallback_requested():
        reason = f"{OBSREGRID_PYTHON_ENV}=1"
    else:
        reason = unavailable_reason()
        if reason is None:
            return sys.modules[__name__]
    report_workaround(operation, reason)
    return None


def report_workaround(operation: str, reason: str) -> None:
    """Print one WORKAROUND line per operation per process.

    The line names the concrete breakage, not just the substitution: a
    scipy plan's tie-breaking is traversal order, so the integer mapping
    a case is scored through stops being a function of its two grids.
    """
    if operation not in _REPORTED_FALLBACKS:
        _REPORTED_FALLBACKS.add(operation)
        print(f"[obs] WORKAROUND: scipy/numpy {operation} "
              f"(Rust obs-regrid bridge not used: {reason}); on this path "
              f"exactly-tied nearest neighbours are resolved by cKDTree "
              f"traversal order rather than by lowest source index, so a "
              f"remap plan is no longer a function of its two grids alone "
              f"and arms remapped on different boxes may not be comparable")


def python_fallback_requested() -> bool:
    """Has the caller explicitly opted out of the Rust path?

    The opt-out exists for parity debugging; any production route that
    honors it must say so in its receipt (fixed-means-default).
    """
    return os.environ.get(OBSREGRID_PYTHON_ENV, "").strip() not in ("", "0")


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("obs_regrid.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("libobs_regrid.dylib",)
    return ("libobs_regrid.so",)


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first (the nc-writer ladder)."""
    from gpuwm.bridges import default_bridge_dir, packaged_bridge_dir
    from gpuwm.rustwx import crate_dir

    filename = library_names()[0]
    candidates: list[Path] = []
    override = os.environ.get(OBSREGRID_BRIDGE_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def resolve_obsregrid_bridge() -> Path:
    """First existing candidate, or a refusal listing every path."""
    override = os.environ.get(OBSREGRID_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            return candidate.resolve()
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{OBSREGRID_BRIDGE_ENV} names a missing file: {candidate}")
    rendered = "\n  ".join(str(c) for c in library_candidates())
    separator = ";" if os.name == "nt" else " &&"
    raise FileNotFoundError(
        "the Rust observation-remap library was not found; searched:\n  "
        + rendered
        + "\n  # stage it with the rest of the bundle:\n"
        "  gpuwm fetch-bridges\n"
        "  # or build it from a checkout:\n"
        f"  cd tools/rustwx{separator} cargo build --release "
        f"-p obs-regrid --offline{separator} cd ../..")


_LIBRARY: ctypes.CDLL | None = None


def unavailable_reason() -> str | None:
    """Why the Rust engine is not loadable here, or None when it is."""
    try:
        load()
    except (FileNotFoundError, OSError, ObsRegridBridgeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def load() -> ctypes.CDLL:
    """Load the library once and bind every signature."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_obsregrid_bridge()
    library = ctypes.CDLL(str(path))

    library.gpuwm_obsregrid_abi_version.argtypes = []
    library.gpuwm_obsregrid_abi_version.restype = ctypes.c_uint32
    observed = int(library.gpuwm_obsregrid_abi_version())
    if observed != OBSREGRID_ABI:
        raise ObsRegridBridgeError(
            f"{path} speaks obs-regrid ABI {observed}, this gpuwm needs "
            f"{OBSREGRID_ABI}; rebuild tools/rustwx")

    u8p = ctypes.POINTER(ctypes.c_uint8)
    i64p = ctypes.POINTER(ctypes.c_int64)
    f64p = ctypes.POINTER(ctypes.c_double)
    size = ctypes.c_size_t

    library.gpuwm_obsregrid_last_error.argtypes = [u8p, size]
    library.gpuwm_obsregrid_last_error.restype = size

    library.gpuwm_obsregrid_build_plan.argtypes = [
        ctypes.c_uint32, f64p, f64p, size, size, f64p, f64p, size, size,
        ctypes.c_double, i64p, u8p, f64p]
    library.gpuwm_obsregrid_build_plan.restype = ctypes.c_int32

    library.gpuwm_obsregrid_apply_plan.argtypes = [
        ctypes.c_uint32, i64p, u8p, size, size, size, size, f64p, u8p,
        f64p, u8p]
    library.gpuwm_obsregrid_apply_plan.restype = ctypes.c_int32

    _LIBRARY = library
    return library


def last_error(library: ctypes.CDLL) -> str:
    length = int(library.gpuwm_obsregrid_last_error(None, 0))
    if length == 0:
        return "the library reported failure without a message"
    buffer = (ctypes.c_uint8 * length)()
    library.gpuwm_obsregrid_last_error(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), length)
    return bytes(buffer).decode("utf-8", "replace")


def _contiguous(array: np.ndarray, dtype) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=dtype))


def _f64p(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _i64p(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _u8p(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def build_plan(*, method: str, source_latitude: np.ndarray,
               source_longitude: np.ndarray,
               destination_latitude: np.ndarray,
               destination_longitude: np.ndarray, max_distance_m: float
               ) -> tuple[np.ndarray, np.ndarray, float]:
    """``(source_index, reachable, max_used_distance_m)`` from the crate.

    Shapes follow the Python module's contract exactly: ``source_index``
    is per destination cell for ``nearest`` and per source cell for
    ``cell_average``; ``reachable`` is the destination mask in both.
    """
    library = load()
    code = METHOD_CODES[method]
    source_shape = np.asarray(source_latitude).shape
    destination_shape = np.asarray(destination_latitude).shape

    source_lat = _contiguous(source_latitude, np.float64)
    source_lon = _contiguous(source_longitude, np.float64)
    destination_lat = _contiguous(destination_latitude, np.float64)
    destination_lon = _contiguous(destination_longitude, np.float64)

    index_shape = destination_shape if code == 0 else source_shape
    index = np.zeros(index_shape, dtype=np.int64)
    reachable = np.zeros(destination_shape, dtype=np.uint8)
    used = ctypes.c_double(0.0)

    status = library.gpuwm_obsregrid_build_plan(
        ctypes.c_uint32(code),
        _f64p(source_lat), _f64p(source_lon),
        ctypes.c_size_t(int(source_shape[0])),
        ctypes.c_size_t(int(source_shape[1])),
        _f64p(destination_lat), _f64p(destination_lon),
        ctypes.c_size_t(int(destination_shape[0])),
        ctypes.c_size_t(int(destination_shape[1])),
        ctypes.c_double(float(max_distance_m)),
        _i64p(index), _u8p(reachable), ctypes.byref(used))
    if status != 0:
        raise ObsRegridBridgeError(last_error(library))
    return index, reachable.astype(bool), float(used.value)


def apply_plan(*, method: str, source_index: np.ndarray,
               reachable: np.ndarray, source_shape: tuple[int, int],
               destination_shape: tuple[int, int], values: np.ndarray,
               valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(values, valid)`` on the destination grid, from the crate."""
    library = load()
    code = METHOD_CODES[method]

    index = _contiguous(source_index, np.int64)
    reachable_bytes = _contiguous(reachable, np.uint8)
    value_bytes = _contiguous(values, np.float64)
    valid_bytes = _contiguous(valid, np.uint8)
    out_values = np.zeros(destination_shape, dtype=np.float64)
    out_valid = np.zeros(destination_shape, dtype=np.uint8)

    status = library.gpuwm_obsregrid_apply_plan(
        ctypes.c_uint32(code), _i64p(index), _u8p(reachable_bytes),
        ctypes.c_size_t(int(source_shape[0])),
        ctypes.c_size_t(int(source_shape[1])),
        ctypes.c_size_t(int(destination_shape[0])),
        ctypes.c_size_t(int(destination_shape[1])),
        _f64p(value_bytes), _u8p(valid_bytes),
        _f64p(out_values), _u8p(out_valid))
    if status != 0:
        raise ObsRegridBridgeError(last_error(library))
    return out_values, out_valid.astype(bool)


__all__ = [
    "ABI_MARKER", "METHOD_CODES", "OBSREGRID_ABI", "OBSREGRID_BRIDGE_ENV",
    "OBSREGRID_PYTHON_ENV", "ObsRegridBridgeError", "apply_plan",
    "build_plan", "last_error", "library_candidates", "library_names",
    "load", "python_fallback_requested", "report_workaround",
    "resolve_obsregrid_bridge", "route", "unavailable_reason",
]
