"""ctypes seam onto the Rust static-field builder (``static-fields``).

The library is ``tools/rustwx/crates/static-fields``, the port target
for everything in :mod:`gpuwm.static` that processes bytes: projection
transforms, WPS_GEOG tile ingest, the geogrid-equivalent field build,
corridor geometry/crop, and the high-resolution overlay compute.  This
module owns ONLY the loading discipline and the low-level call
bindings; the public Python entry points (`build_static`,
`ProjectedGrid` methods, the highres appliers) keep their signatures
and route here by default once their lane lands.

Why ctypes rather than pyo3: same ruling as
:mod:`gpuwm.io.nc_writer_bridge` -- every Rust library gpuwm drives is
a cdylib behind ctypes; one loading discipline, one staging path, one
ABI-marker rule, no per-interpreter builds.

Default-on contract (Drew's fixed-means-default law): once a lane
lands, the bare default path for its entry points is this bridge.  The
pure-Python implementation stays importable as the parity reference
and as an explicit fallback (``GPUWM_STATIC_PYTHON=1``), and every use
of the fallback is a reported workaround, never silent.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np

#: The C ABI version this module speaks; a library reporting anything
#: else is refused rather than called.
STATIC_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
STATIC_BRIDGE_ENV: Final[str] = "GPUWM_STATIC_BRIDGE"

#: Opt-out to the pure-Python implementation (a reported workaround).
STATIC_PYTHON_ENV: Final[str] = "GPUWM_STATIC_PYTHON"

#: The exported symbol that identifies THIS contract, for
#: :data:`gpuwm.bridges.BRIDGE_ABI_MARKERS`: a build that loads and
#: answers the version probe but predates the field build cannot
#: produce a single static field.
ABI_MARKER: Final[bytes] = b"gpuwm_static_build_fields"

#: Stagger codes (`types::Stagger` in the crate).
STAGGER_MASS: Final[int] = 0
STAGGER_U: Final[int] = 1
STAGGER_V: Final[int] = 2
STAGGER_CORNER: Final[int] = 3

#: Array kinds for ``gpuwm_static_grid_array``.
ARRAY_LAT: Final[int] = 0
ARRAY_LON: Final[int] = 1
ARRAY_MAPFAC: Final[int] = 2
ARRAY_CORIOLIS_F: Final[int] = 3
ARRAY_CORIOLIS_E: Final[int] = 4
ARRAY_SINALPHA: Final[int] = 5
ARRAY_COSALPHA: Final[int] = 6


class StaticBridgeError(RuntimeError):
    """The Rust static builder refused, with its own message."""


#: Operations whose pure-Python fallback has already been reported this
#: process (fixed-means-default: a fallback run is a workaround and must
#: say so, once per operation, never silently and never on every call).
_REPORTED_FALLBACKS: set[str] = set()


def route(operation: str):
    """This module when the Rust seam is the active default, else None.

    ``None`` means the caller must run its pure-Python body -- either
    because :data:`STATIC_PYTHON_ENV` explicitly opted out or because
    the library is not loadable -- and the reason has been printed once
    per operation as a WORKAROUND line.  Every default-on entry point in
    :mod:`gpuwm.static` routes through this single decision so the env
    flag is read at call time (parity harnesses toggle it per call).
    """
    if python_fallback_requested():
        reason = f"{STATIC_PYTHON_ENV}=1"
    else:
        reason = unavailable_reason()
        if reason is None:
            return sys.modules[__name__]
    report_workaround(operation, reason)
    return None


def report_workaround(operation: str, reason: str) -> None:
    """Print one WORKAROUND line per operation per process."""
    if operation not in _REPORTED_FALLBACKS:
        _REPORTED_FALLBACKS.add(operation)
        print(f"[static] WORKAROUND: pure-Python {operation} "
              f"(Rust static-fields bridge not used: {reason})")


def python_fallback_requested() -> bool:
    """Has the caller explicitly opted out of the Rust path?

    The opt-out exists for parity debugging; any production route that
    honors it must say so in its receipt (fixed-means-default).
    """
    return os.environ.get(STATIC_PYTHON_ENV, "").strip() not in ("", "0")


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("static_fields.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("libstatic_fields.dylib",)
    return ("libstatic_fields.so",)


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first (the nc-writer ladder)."""
    from gpuwm.bridges import default_bridge_dir, packaged_bridge_dir
    from gpuwm.rustwx import crate_dir

    filename = library_names()[0]
    candidates: list[Path] = []
    override = os.environ.get(STATIC_BRIDGE_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def resolve_static_bridge() -> Path:
    """First existing candidate, or a refusal listing every path."""
    override = os.environ.get(STATIC_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            return candidate.resolve()
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{STATIC_BRIDGE_ENV} names a missing file: {candidate}")
    rendered = "\n  ".join(str(c) for c in library_candidates())
    separator = ";" if os.name == "nt" else " &&"
    raise FileNotFoundError(
        "the Rust static-field library was not found; searched:\n  "
        + rendered
        + "\n  # build it from a checkout:\n"
        f"  cd tools/rustwx{separator} cargo build --release "
        f"-p static-fields --offline{separator} cd ../..")


_LIBRARY: ctypes.CDLL | None = None


def unavailable_reason() -> str | None:
    """Why the Rust builder is not loadable here, or None when it is."""
    try:
        load()
    except (FileNotFoundError, OSError, StaticBridgeError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def load() -> ctypes.CDLL:
    """Load the library once and bind every signature."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_static_bridge()
    library = ctypes.CDLL(str(path))

    library.gpuwm_static_abi_version.argtypes = []
    library.gpuwm_static_abi_version.restype = ctypes.c_uint32
    observed = int(library.gpuwm_static_abi_version())
    if observed != STATIC_ABI:
        raise StaticBridgeError(
            f"{path} speaks static-fields ABI {observed}, this gpuwm "
            f"needs {STATIC_ABI}; rebuild tools/rustwx")

    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    f64p = ctypes.POINTER(ctypes.c_double)
    size = ctypes.c_size_t

    # EVERY binding below is inside one net, and the net names what it
    # caught.  The version handshake above cannot answer this question:
    # a library built before an entry point was ADDED still answers ABI
    # 1, so it passes, and then `getattr` on the entry point it does not
    # export raises a bare `AttributeError` -- which
    # `unavailable_reason` does not catch, because ctypes' lookup
    # failure is not one of the three error classes a missing or
    # unloadable library raises.
    #
    # The concrete breakage that prevented: a box carrying a bundle
    # staged before the high-resolution warp entry points existed
    # (`gpuwm_static_highres_resample`, `..._transform_points`) ran
    # `gpuwm doctor` -- the tool whose entire purpose is diagnosing a
    # broken or partial install -- and got a ctypes traceback out of
    # `_static_builder_check` instead of a report.  Measured on this
    # tree, 2026-08-18, against a real staged
    # `~/.gpuwm/bridges/static_fields.dll`.
    #
    # A missing entry point is the same FACT as a version mismatch --
    # this library predates this gpuwm -- so it is reported the same
    # way: a StaticBridgeError, which makes the builder row say
    # `unusable` with a remedy, and makes the static path take the
    # announced `GPUWM_STATIC_PYTHON` degradation rather than crash.
    try:
        _bind_entry_points(library, u8p=u8p, u64p=u64p, f64p=f64p, size=size)
    except AttributeError as error:
        raise StaticBridgeError(
            f"{path} answers static-fields ABI {observed} but does not "
            f"export an entry point this gpuwm binds ({error}); it is a "
            f"build that predates this release.  Restage it with `gpuwm "
            f"fetch-bridges`, or rebuild from a checkout: cd tools/rustwx"
            f"{';' if os.name == 'nt' else ' &&'} cargo build --release "
            f"-p static-fields --offline") from None

    _LIBRARY = library
    return library


def _bind_entry_points(library: ctypes.CDLL, *, u8p, u64p, f64p, size) -> None:
    """Bind every signature this gpuwm calls.

    Split out of :func:`load` so one ``try`` covers the whole set,
    including entry points added later: a binding that escapes the net
    is a binding that can still crash a report.
    """

    library.gpuwm_static_last_error.argtypes = [u8p, size]
    library.gpuwm_static_last_error.restype = size

    library.gpuwm_static_grid_new.argtypes = [u8p, size, u64p]
    library.gpuwm_static_grid_new.restype = ctypes.c_int32
    library.gpuwm_static_grid_nest.argtypes = [
        ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_double,
        u64p]
    library.gpuwm_static_grid_nest.restype = ctypes.c_int32
    library.gpuwm_static_grid_translated.argtypes = [
        ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, u64p]
    library.gpuwm_static_grid_translated.restype = ctypes.c_int32
    library.gpuwm_static_grid_free.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_grid_free.restype = None
    library.gpuwm_static_grid_array.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, f64p, size]
    library.gpuwm_static_grid_array.restype = ctypes.c_int32
    library.gpuwm_static_grid_transform.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, f64p, f64p, size]
    library.gpuwm_static_grid_transform.restype = ctypes.c_int32
    library.gpuwm_static_grid_identity_probes.argtypes = [
        ctypes.c_uint64, u8p, size]
    library.gpuwm_static_grid_identity_probes.restype = ctypes.c_int64

    library.gpuwm_static_build_fields.argtypes = [
        ctypes.c_uint64, u8p, size, ctypes.c_uint32, u64p]
    library.gpuwm_static_build_fields.restype = ctypes.c_int32
    library.gpuwm_static_fieldset_len.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_fieldset_len.restype = ctypes.c_int64
    library.gpuwm_static_fieldset_name.argtypes = [
        ctypes.c_uint64, size, u8p, size]
    library.gpuwm_static_fieldset_name.restype = ctypes.c_int64
    library.gpuwm_static_field_dims.argtypes = [
        ctypes.c_uint64, u8p, size, u64p, u64p, u64p]
    library.gpuwm_static_field_dims.restype = ctypes.c_int32
    library.gpuwm_static_field_read.argtypes = [
        ctypes.c_uint64, u8p, size, f64p, size]
    library.gpuwm_static_field_read.restype = ctypes.c_int32
    library.gpuwm_static_field_coverage_json.argtypes = [
        ctypes.c_uint64, u8p, size, u8p, size]
    library.gpuwm_static_field_coverage_json.restype = ctypes.c_int64
    library.gpuwm_static_fieldset_free.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_fieldset_free.restype = None

    library.gpuwm_static_highres_terrain.argtypes = [
        ctypes.c_uint64, u8p, size, u64p]
    library.gpuwm_static_highres_terrain.restype = ctypes.c_int32
    library.gpuwm_static_highres_overrides.argtypes = [
        ctypes.c_uint64, u8p, size, u64p]
    library.gpuwm_static_highres_overrides.restype = ctypes.c_int32
    library.gpuwm_static_highres_merge.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, u8p, size, u64p]
    library.gpuwm_static_highres_merge.restype = ctypes.c_int32
    library.gpuwm_static_highres_derive_window.argtypes = [
        u8p, size, u8p, size]
    library.gpuwm_static_highres_derive_window.restype = ctypes.c_int64
    library.gpuwm_static_highres_resample.argtypes = [u8p, size, u64p]
    library.gpuwm_static_highres_resample.restype = ctypes.c_int32
    library.gpuwm_static_highres_transform_points.argtypes = [
        u8p, size, f64p, f64p, size]
    library.gpuwm_static_highres_transform_points.restype = ctypes.c_int32
    library.gpuwm_static_highres_audit_json.argtypes = [
        ctypes.c_uint64, u8p, size]
    library.gpuwm_static_highres_audit_json.restype = ctypes.c_int64
    library.gpuwm_static_highres_audit_drop.argtypes = [ctypes.c_uint64]
    library.gpuwm_static_highres_audit_drop.restype = None


def last_error(library: ctypes.CDLL) -> str:
    length = int(library.gpuwm_static_last_error(None, 0))
    if length == 0:
        return "(no error recorded)"
    buffer = (ctypes.c_uint8 * length)()
    library.gpuwm_static_last_error(buffer, length)
    return bytes(buffer).decode("utf-8", "replace")


def _check(library: ctypes.CDLL, code: int, doing: str) -> None:
    if int(code) != 0:
        raise StaticBridgeError(f"{doing}: {last_error(library)}")


def _utf8(payload: str) -> tuple[ctypes.Array, int]:
    raw = payload.encode("utf-8")
    return (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw), len(raw)


def grid_new(spec: dict) -> int:
    """Create a Rust grid handle from a GridSpec-shaped dict."""
    library = load()
    buffer, length = _utf8(json.dumps(spec))
    handle = ctypes.c_uint64(0)
    _check(library,
           library.gpuwm_static_grid_new(buffer, length,
                                         ctypes.byref(handle)),
           "grid_new")
    return int(handle.value)


def grid_free(handle: int) -> None:
    load().gpuwm_static_grid_free(ctypes.c_uint64(handle))


def grid_translated(reference: int, di_cells: int, dj_cells: int,
                    e_we: int, e_sn: int) -> int:
    """Placement-translated grid handle from a reference handle."""
    library = load()
    handle = ctypes.c_uint64(0)
    _check(library,
           library.gpuwm_static_grid_translated(
               ctypes.c_uint64(reference), int(di_cells), int(dj_cells),
               int(e_we), int(e_sn), ctypes.byref(handle)),
           "grid_translated")
    return int(handle.value)


def grid_array(handle: int, stagger: int, kind: int,
               ny: int, nx: int) -> np.ndarray:
    """One derived array (row-major float64) from a grid handle."""
    library = load()
    out = np.empty((int(ny), int(nx)), dtype=np.float64)
    _check(library,
           library.gpuwm_static_grid_array(
               ctypes.c_uint64(handle), ctypes.c_uint32(stagger),
               ctypes.c_uint32(kind),
               out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               out.size),
           f"grid_array[{stagger},{kind}]")
    return out


def grid_identity_probes_json(handle: int) -> str:
    """The corridor identity-probe JSON document for a grid handle."""
    library = load()
    length = int(library.gpuwm_static_grid_identity_probes(
        ctypes.c_uint64(handle), None, 0))
    if length < 0:
        raise StaticBridgeError(
            f"grid_identity_probes: {last_error(library)}")
    buffer = (ctypes.c_uint8 * length)()
    written = int(library.gpuwm_static_grid_identity_probes(
        ctypes.c_uint64(handle), buffer, length))
    if written != length:
        raise StaticBridgeError(
            f"grid_identity_probes rendered {written} bytes on a "
            f"{length}-byte probe")
    return bytes(buffer).decode("utf-8")


def field_coverage_json(handle: int, name: str) -> str:
    """One field's source-coverage receipt JSON from a fieldset handle."""
    library = load()
    name_buf, name_len = _utf8(name)
    length = int(library.gpuwm_static_field_coverage_json(
        ctypes.c_uint64(handle), name_buf, name_len, None, 0))
    if length < 0:
        raise StaticBridgeError(
            f"field_coverage_json[{name}]: {last_error(library)}")
    buffer = (ctypes.c_uint8 * length)()
    written = int(library.gpuwm_static_field_coverage_json(
        ctypes.c_uint64(handle), name_buf, name_len, buffer, length))
    if written != length:
        raise StaticBridgeError(
            f"field_coverage_json[{name}] rendered {written} bytes on "
            f"a {length}-byte probe")
    return bytes(buffer).decode("utf-8")


def build_fields(grid_handle: int, geog_paths: dict,
                 halo: int | None = None) -> int:
    """Run the full static build; returns a fieldset handle."""
    library = load()
    buffer, length = _utf8(json.dumps(
        {key: str(value) for key, value in geog_paths.items()}))
    handle = ctypes.c_uint64(0)
    halo_code = 0xFFFFFFFF if halo is None else int(halo)
    _check(library,
           library.gpuwm_static_build_fields(
               ctypes.c_uint64(grid_handle), buffer, length,
               ctypes.c_uint32(halo_code), ctypes.byref(handle)),
           "build_fields")
    return int(handle.value)


def fieldset_to_dict(handle: int) -> dict[str, np.ndarray]:
    """Copy every field of a fieldset handle into float64 numpy arrays
    (2-D fields lose their leading singleton plane axis)."""
    library = load()
    count = int(library.gpuwm_static_fieldset_len(ctypes.c_uint64(handle)))
    if count < 0:
        raise StaticBridgeError(f"fieldset_len: {last_error(library)}")
    out: dict[str, np.ndarray] = {}
    for index in range(count):
        cap = 128
        buffer = (ctypes.c_uint8 * cap)()
        length = int(library.gpuwm_static_fieldset_name(
            ctypes.c_uint64(handle), ctypes.c_size_t(index), buffer, cap))
        if length < 0 or length > cap:
            raise StaticBridgeError(
                f"fieldset_name[{index}]: {last_error(library)}")
        name = bytes(buffer[:length]).decode("utf-8")
        name_buf, name_len = _utf8(name)
        planes = ctypes.c_uint64(0)
        ny = ctypes.c_uint64(0)
        nx = ctypes.c_uint64(0)
        _check(library,
               library.gpuwm_static_field_dims(
                   ctypes.c_uint64(handle), name_buf, name_len,
                   ctypes.byref(planes), ctypes.byref(ny),
                   ctypes.byref(nx)),
               f"field_dims[{name}]")
        shape = (int(planes.value), int(ny.value), int(nx.value))
        array = np.empty(shape, dtype=np.float64)
        _check(library,
               library.gpuwm_static_field_read(
                   ctypes.c_uint64(handle), name_buf, name_len,
                   array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   array.size),
               f"field_read[{name}]")
        out[name] = array[0] if shape[0] == 1 else array
    return out


def fieldset_free(handle: int) -> None:
    load().gpuwm_static_fieldset_free(ctypes.c_uint64(handle))


def highres_audit_json(handle: int) -> dict:
    """The audit document a highres call remembered for its fieldset."""
    library = load()
    length = int(library.gpuwm_static_highres_audit_json(
        ctypes.c_uint64(handle), None, 0))
    if length < 0:
        raise StaticBridgeError(f"highres audit: {last_error(library)}")
    buffer = (ctypes.c_uint8 * length)()
    written = int(library.gpuwm_static_highres_audit_json(
        ctypes.c_uint64(handle), buffer, length))
    if written != length:
        raise StaticBridgeError(
            f"highres audit rendered {written} bytes on a {length}-byte "
            "probe")
    return json.loads(bytes(buffer).decode("utf-8"))


def highres_audit_drop(handle: int) -> None:
    load().gpuwm_static_highres_audit_drop(ctypes.c_uint64(handle))


def highres_resample(request: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Run one warp-substrate request; returns (fields, audit).

    The crate's refusals are RAISED as :class:`ValueError` carrying the
    crate's own message, which the committed goldens pin byte-equal to
    the pure-Python messages -- so a caller cannot tell which engine
    refused from the text, only from the WORKAROUND line.
    """
    library = load()
    buffer, length = _utf8(json.dumps(request))
    handle = ctypes.c_uint64(0)
    code = library.gpuwm_static_highres_resample(
        buffer, length, ctypes.byref(handle))
    if code != 0:
        raise ValueError(last_error(library))
    try:
        fields = fieldset_to_dict(int(handle.value))
        audit = highres_audit_json(int(handle.value))
    finally:
        highres_audit_drop(int(handle.value))
        fieldset_free(int(handle.value))
    return fields, audit


def highres_transform_points(to_crs: str, x: np.ndarray, y: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Transform lon/lat degree arrays into ``to_crs`` (a recorded
    ``crs_override`` string), returning new float64 arrays."""
    library = load()
    xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64)).reshape(-1)
    ys = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).reshape(-1)
    if xs.size != ys.size:
        raise ValueError(
            f"transform_points x/y length mismatch: {xs.size} vs {ys.size}")
    xs, ys = xs.copy(), ys.copy()
    buffer, length = _utf8(json.dumps({"to": to_crs}))
    code = library.gpuwm_static_highres_transform_points(
        buffer, length,
        xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ys.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xs.size)
    if code != 0:
        raise ValueError(last_error(library))
    return xs, ys


class StaticCoverageRefusal(ValueError):
    """The crate refused because the footprint reaches past the source.

    Distinct from every other refusal because the caller's ``on_refuse``
    policy is allowed to answer it with the unchanged 30-arc-second
    baseline, and is NOT allowed to answer a decode fault that way.  The
    seam reports the distinction as its own return code (-2), so this is
    a fact off the wire rather than a sentence somebody pattern-matched.
    """


def highres_derive_window(request: dict) -> dict:
    """Derive one cached raster window (mosaic/clip/fill); returns the
    crate's audit document.  The GeoTIFF lands at ``request['out_path']``.

    Raises :class:`StaticCoverageRefusal` for a source-coverage refusal
    and :class:`ValueError` for anything else.
    """
    library = load()
    buffer, length = _utf8(json.dumps(request))
    # ONE call, not a length probe followed by a real one: this entry
    # point WRITES the derived GeoTIFF as a side effect, so probing
    # would derive the window twice.  The audit is a flat document of a
    # dozen scalars; 64 KiB is three orders of magnitude of headroom and
    # an overflow is refused rather than truncated.
    cap = 65536
    out = (ctypes.c_uint8 * cap)()
    written = int(library.gpuwm_static_highres_derive_window(
        buffer, length, out, cap))
    if written == -2:
        raise StaticCoverageRefusal(last_error(library))
    if written < 0:
        raise ValueError(last_error(library))
    if written > cap:
        raise StaticBridgeError(
            f"derive_window audit is {written} bytes, beyond the {cap}-byte "
            "buffer; the window was written but its audit was truncated")
    return json.loads(bytes(out[:written]).decode("utf-8"))
