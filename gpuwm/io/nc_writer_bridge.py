"""ctypes seam onto the Rust classic-NetCDF writer.

This is the write-side counterpart to :mod:`gpuwm.netcdf_bridge`, which is
the read-side facade over ``rw_netcdf``.  The library it loads is
``tools/rustwx/crates/netcdf-writer``, a dependency-free emitter for the
NetCDF classic containers (CDF-1, CDF-2, CDF-5) with the full classic type
set, a record dimension and attributes -- everything a wrfout needs and
none of which the previous Rust writer (``rw-store``'s ``netcdf3.rs``,
CDF-2, ``NC_FLOAT`` only, no record dimension) could express.

Why ctypes rather than pyo3
---------------------------
Every Rust library gpuwm already drives from Python is a cdylib behind
ctypes: ``gpuwm_preprocess_cpu`` (:mod:`gpuwm.ingest.cpu_backend`) and
``region_global_dealias`` (:mod:`gpuwm.obs.dealias_region`,
:mod:`gpuwm.obs.coarse_cost`).  One loading discipline, one staging path
in ``tools/stage_wheel_bridges.py``, one ABI-marker rule in
:mod:`gpuwm.bridges`, and no build-time bind to a CPython version.  A
pyo3 module would need a maturin build per interpreter and would be the
only artifact of its kind in the estate.

What this module does NOT do
----------------------------
It does not change what :class:`gpuwm.io.wrfout.WrfoutWriter` writes with
today.  Flipping the product tape onto this writer is an integration step
with its own dual-write verification, and doing it here would mean
changing the product tape in the same commit that first introduces the
library that writes it.

Shape
-----
The calls mirror the netCDF C API on purpose, because the caller being
ported is written against netCDF4-python::

    from gpuwm.io.nc_writer_bridge import ClassicSchema

    schema = ClassicSchema()                       # CDF-2 by default
    time = schema.def_dim("Time", 0, unlimited=True)
    strlen = schema.def_dim("DateStrLen", 19)
    south_north = schema.def_dim("south_north", ny)
    west_east = schema.def_dim("west_east", nx)
    schema.put_global_attr("TITLE", " OUTPUT FROM GPUWM")
    times = schema.def_var("Times", "S1", (time, strlen))
    t2 = schema.def_var("T2", "f4", (time, south_north, west_east))
    schema.put_var_attr(t2, "units", "K")

    with schema.create(path) as writer:
        writer.write_record(0, times, b"2026-08-16_00:00:00")
        writer.write_record(0, t2, field)          # C-contiguous f4

Adding a variable is a row in the schema table driving ``def_var``.  There
is no per-variable code path on either side of the seam.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Final

import numpy as np

#: The C ABI version this module speaks.  A library reporting anything
#: else is refused rather than called: the signatures are positional and
#: a mismatched build would be interpreted, not detected.
NCWRITE_ABI: Final[int] = 1

#: Environment override for the library path, first rung of the ladder.
NCWRITE_BRIDGE_ENV: Final[str] = "GPUWM_NCWRITE_BRIDGE"

#: The exported symbol that identifies THIS contract, for
#: :data:`gpuwm.bridges.BRIDGE_ABI_MARKERS`.  It names the newest
#: capability a DEFAULT path depends on: ``gpuwm_ncwrite_write_record``
#: held it while the record dimension was that capability (a build
#: predating it exports every other entry point, loads cleanly, answers
#: the version probe -- and then cannot write a ``Times`` variable), and
#: the read-back sweep is now it, because the default wrfinput/wrfbdy
#: export verifies every float variable through
#: :func:`scan_nonfinite` before publishing.
ABI_MARKER: Final[bytes] = b"gpuwm_ncwrite_scan_nonfinite"

# Classic external type codes, as the format writes them.
NC_BYTE: Final[int] = 1
NC_CHAR: Final[int] = 2
NC_SHORT: Final[int] = 3
NC_INT: Final[int] = 4
NC_FLOAT: Final[int] = 5
NC_DOUBLE: Final[int] = 6
NC_UBYTE: Final[int] = 7
NC_USHORT: Final[int] = 8
NC_UINT: Final[int] = 9
NC_INT64: Final[int] = 10
NC_UINT64: Final[int] = 11

#: Container version codes accepted by ``gpuwm_ncwrite_schema_new``.
FORMATS: Final[dict[str, int]] = {"cdf1": 1, "cdf2": 2, "cdf5": 5}

#: ``(numpy kind, itemsize) -> classic type code``.  A table, so adding a
#: type is a row.  ``S`` is bytes, which is how numpy spells ``NC_CHAR``.
_NUMPY_TO_NC: Final[dict[tuple[str, int], int]] = {
    ("i", 1): NC_BYTE,
    ("S", 1): NC_CHAR,
    ("i", 2): NC_SHORT,
    ("i", 4): NC_INT,
    ("f", 4): NC_FLOAT,
    ("f", 8): NC_DOUBLE,
    ("u", 1): NC_UBYTE,
    ("u", 2): NC_USHORT,
    ("u", 4): NC_UINT,
    ("i", 8): NC_INT64,
    ("u", 8): NC_UINT64,
}

_NC_TO_NUMPY: Final[dict[int, str]] = {
    NC_BYTE: "i1",
    NC_CHAR: "S1",
    NC_SHORT: "i2",
    NC_INT: "i4",
    NC_FLOAT: "f4",
    NC_DOUBLE: "f8",
    NC_UBYTE: "u1",
    NC_USHORT: "u2",
    NC_UINT: "u4",
    NC_INT64: "i8",
    NC_UINT64: "u8",
}


class NcWriteError(RuntimeError):
    """The Rust writer refused, with its own message."""


def library_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("netcdf_writer.dll",)
    if os.uname().sysname == "Darwin":  # pragma: no cover - platform route
        return ("libnetcdf_writer.dylib",)
    return ("libnetcdf_writer.so",)


def library_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths, best first.

    The same ladder :mod:`gpuwm.rustwx` walks for ``rw_wrfbatch``, because
    the library is built by the same workspace: environment override,
    checkout release, checkout debug, ``libexec/bridges`` beside the
    package, the wheel-bundled directory, the user-level default.
    """

    from gpuwm.bridges import default_bridge_dir, packaged_bridge_dir
    from gpuwm.rustwx import crate_dir

    filename = library_names()[0]
    candidates: list[Path] = []
    override = os.environ.get(NCWRITE_BRIDGE_ENV)
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


def resolve_ncwrite_bridge() -> Path:
    """The first existing candidate, or a refusal listing every path.

    An override naming a missing file is a hard error: explicit
    configuration must fail loudly rather than fall through to a
    different library.
    """

    override = os.environ.get(NCWRITE_BRIDGE_ENV)
    for candidate in library_candidates():
        if candidate.is_file():
            from gpuwm.bridges import accept_resolved

            return accept_resolved(candidate.resolve(),
                                   executable=False)
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{NCWRITE_BRIDGE_ENV} names a missing file: {candidate}")
    rendered = "\n  ".join(str(candidate) for candidate in library_candidates())
    separator = ";" if os.name == "nt" else " &&"
    raise FileNotFoundError(
        "the Rust NetCDF writer library was not found; searched:\n  "
        + rendered
        + "\n  # build it from a checkout:\n"
        f"  cd tools/rustwx{separator} cargo build --release --offline"
        f"{separator} cd ../..")


_LIBRARY: ctypes.CDLL | None = None
_UNAVAILABLE: str | None = None


def unavailable_reason() -> str | None:
    """Why the Rust writer is not loadable here, or None when it is."""

    try:
        load()
    except (FileNotFoundError, OSError, NcWriteError) as error:
        return f"{type(error).__name__}: {error}"
    return None


def load() -> ctypes.CDLL:
    """Load the library once and bind every signature."""

    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = resolve_ncwrite_bridge()
    library = ctypes.CDLL(str(path))

    library.gpuwm_ncwrite_abi_version.argtypes = []
    library.gpuwm_ncwrite_abi_version.restype = ctypes.c_uint32
    observed = int(library.gpuwm_ncwrite_abi_version())
    if observed != NCWRITE_ABI:
        raise NcWriteError(
            f"{path} speaks NetCDF-writer ABI {observed}, this gpuwm needs "
            f"{NCWRITE_ABI}; rebuild tools/rustwx")

    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    void = ctypes.c_void_p
    size = ctypes.c_size_t

    library.gpuwm_ncwrite_last_error.argtypes = [u8p, size]
    library.gpuwm_ncwrite_last_error.restype = size
    library.gpuwm_ncwrite_schema_new.argtypes = [ctypes.c_uint32]
    library.gpuwm_ncwrite_schema_new.restype = void
    library.gpuwm_ncwrite_schema_free.argtypes = [void]
    library.gpuwm_ncwrite_schema_free.restype = None
    library.gpuwm_ncwrite_def_dim.argtypes = [
        void, ctypes.c_char_p, size, ctypes.c_uint64, ctypes.c_int32]
    library.gpuwm_ncwrite_def_dim.restype = ctypes.c_int64
    library.gpuwm_ncwrite_def_var.argtypes = [
        void, ctypes.c_char_p, size, ctypes.c_uint32, size, u64p]
    library.gpuwm_ncwrite_def_var.restype = ctypes.c_int64
    library.gpuwm_ncwrite_put_att.argtypes = [
        void, ctypes.c_int64, ctypes.c_char_p, size, ctypes.c_uint32, size,
        ctypes.c_char_p]
    library.gpuwm_ncwrite_put_att.restype = ctypes.c_int32
    library.gpuwm_ncwrite_create.argtypes = [void, ctypes.c_char_p, size]
    library.gpuwm_ncwrite_create.restype = void
    library.gpuwm_ncwrite_write_var.argtypes = [
        void, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_char_p, size]
    library.gpuwm_ncwrite_write_var.restype = ctypes.c_int32
    library.gpuwm_ncwrite_write_record.argtypes = [
        void, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32,
        ctypes.c_char_p, size]
    library.gpuwm_ncwrite_write_record.restype = ctypes.c_int32
    library.gpuwm_ncwrite_finish.argtypes = [void]
    library.gpuwm_ncwrite_finish.restype = ctypes.c_int32
    library.gpuwm_ncwrite_abort.argtypes = [void]
    library.gpuwm_ncwrite_abort.restype = None
    library.gpuwm_ncwrite_num_records.argtypes = [void]
    library.gpuwm_ncwrite_num_records.restype = ctypes.c_uint64
    # The read-back sweep. Bound here rather than probed at the call site
    # so a library predating it is refused at LOAD, with the rebuild
    # remedy, instead of half-way through an export.
    try:
        library.gpuwm_ncwrite_scan_nonfinite.argtypes = [ctypes.c_char_p, size]
        library.gpuwm_ncwrite_scan_nonfinite.restype = ctypes.c_int32
        library.gpuwm_ncwrite_last_scan.argtypes = [u8p, size]
        library.gpuwm_ncwrite_last_scan.restype = size
    except AttributeError:
        raise NcWriteError(
            f"{path} exports no gpuwm_ncwrite_scan_nonfinite, so it predates "
            f"this release's netcdf-writer contract (see ABI_MARKER): it can "
            f"write a file but cannot read one back, and the default "
            f"wrfinput/wrfbdy export verifies every float variable before it "
            f"publishes.  Rebuild tools/rustwx (cargo build --release "
            f"--locked --offline -p netcdf-writer), or run "
            f"`gpuwm fetch-bridges` on an installed wheel") from None

    _LIBRARY = library
    return library


def _last_error(library: ctypes.CDLL) -> str:
    needed = int(library.gpuwm_ncwrite_last_error(None, 0))
    if needed == 0:
        return "the Rust writer reported a failure with no message"
    buffer = (ctypes.c_uint8 * needed)()
    library.gpuwm_ncwrite_last_error(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), needed)
    return bytes(buffer).decode("utf-8", "replace")


def scan_nonfinite(path) -> tuple[str, ...]:
    """Read a finished classic file back; name its non-finite variables.

    Returns the float/double variables holding a value that is not
    finite, in definition order -- empty for a clean file.  Anything that
    stops the sweep (a truncated data section, an HDF5 container, an
    unreadable path) is an :class:`NcWriteError`, never an empty tuple:
    "found nothing" and "could not look" must not read the same.

    This is the write side's own verification, kept where the geometry
    is.  ``rw_netcdf`` is the front door for READING a NetCDF file; it
    decodes one variable per process launch through an f64 temp file,
    which is the right shape for resolving a selector and the wrong one
    for sweeping all 183 variables of a ``wrfinput`` on the preparation
    path.
    """

    library = load()
    raw = os.fspath(path).encode("utf-8")
    if library.gpuwm_ncwrite_scan_nonfinite(raw, len(raw)) != 0:
        raise NcWriteError(_last_error(library))
    needed = int(library.gpuwm_ncwrite_last_scan(None, 0))
    if needed == 0:
        return ()
    buffer = (ctypes.c_uint8 * needed)()
    library.gpuwm_ncwrite_last_scan(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)), needed)
    report = bytes(buffer).decode("utf-8", "replace")
    return tuple(name for name in report.split("\n") if name)


def nc_type_of(dtype) -> int:
    """The classic type code for a numpy dtype, or a named refusal."""

    dtype = np.dtype(dtype)
    key = (dtype.kind, dtype.itemsize)
    code = _NUMPY_TO_NC.get(key)
    if code is None:
        raise NcWriteError(
            f"numpy dtype {dtype!r} has no classic NetCDF external type; "
            f"classic supports {sorted(set(_NC_TO_NUMPY.values()))}")
    return code


def _payload(values, nc_type: int, what: str) -> bytes:
    """C-contiguous, native-endian bytes of the declared external type."""

    want = np.dtype(_NC_TO_NUMPY[nc_type])
    if nc_type == NC_CHAR and isinstance(values, (bytes, bytearray)):
        return bytes(values)
    array = np.asarray(values)
    if array.dtype.kind != want.kind or array.dtype.itemsize != want.itemsize:
        raise NcWriteError(
            f"{what} is declared {want.str} but the payload is "
            f"{array.dtype.str}; the writer will not reinterpret it")
    # A non-native byte order would be written back to front: normalise
    # rather than refuse, because a big-endian view of the caller's own
    # array is a legitimate thing to hold.
    array = np.ascontiguousarray(array.astype(want.newbyteorder("="),
                                              copy=False))
    return array.tobytes()


class ClassicWriter:
    """An open file. Use as a context manager, or call :meth:`finish`."""

    def __init__(self, library: ctypes.CDLL, handle: int,
                 var_types: dict[int, int], path: Path) -> None:
        self._library = library
        self._handle = handle
        self._var_types = var_types
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def _check(self, status: int) -> None:
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def _live(self) -> int:
        if self._handle is None:
            raise NcWriteError(f"{self._path} is already closed")
        return self._handle

    def write_var(self, varid: int, values) -> None:
        """Write a whole fixed (non-record) variable."""

        nc_type = self._var_types[varid]
        payload = _payload(values, nc_type, f"variable {varid}")
        self._check(self._library.gpuwm_ncwrite_write_var(
            self._live(), varid, nc_type, payload, len(payload)))

    def write_record(self, recno: int, varid: int, values) -> None:
        """Write one record's slab of a record variable."""

        nc_type = self._var_types[varid]
        payload = _payload(values, nc_type,
                           f"record {recno} of variable {varid}")
        self._check(self._library.gpuwm_ncwrite_write_record(
            self._live(), recno, varid, nc_type, payload, len(payload)))

    def num_records(self) -> int:
        return int(self._library.gpuwm_ncwrite_num_records(self._live()))

    def finish(self) -> None:
        """Patch ``numrecs``, flush, fsync, and release the handle."""

        handle = self._live()
        self._handle = None
        status = int(self._library.gpuwm_ncwrite_finish(handle))
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def abort(self) -> None:
        """Release the handle without finishing. The file stays partial."""

        if self._handle is not None:
            handle, self._handle = self._handle, None
            self._library.gpuwm_ncwrite_abort(handle)

    def __enter__(self) -> "ClassicWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # An exception inside the block leaves a partial file that must
        # not be mistaken for a finished tape, so abort rather than
        # finish -- and let the original exception through.
        if exc_type is not None:
            self.abort()
            return False
        self.finish()
        return False


class ClassicSchema:
    """Dimensions, attributes and variables, before the file exists."""

    def __init__(self, container: str = "cdf2") -> None:
        if container not in FORMATS:
            raise NcWriteError(
                f"unknown container {container!r}; expected one of "
                f"{sorted(FORMATS)}")
        self._library = load()
        handle = self._library.gpuwm_ncwrite_schema_new(FORMATS[container])
        if not handle:
            raise NcWriteError(_last_error(self._library))
        self._handle = handle
        self._var_types: dict[int, int] = {}
        self._container = container

    def _check(self, status: int) -> None:
        if status != 0:
            raise NcWriteError(_last_error(self._library))

    def _live(self):
        if self._handle is None:
            raise NcWriteError("this schema was already handed to create()")
        return self._handle

    def def_dim(self, name: str, length: int = 0, *,
                unlimited: bool = False) -> int:
        """Define a dimension; returns its dimid."""

        raw = name.encode("utf-8")
        dimid = int(self._library.gpuwm_ncwrite_def_dim(
            self._live(), raw, len(raw), int(length), 1 if unlimited else 0))
        if dimid < 0:
            raise NcWriteError(_last_error(self._library))
        return dimid

    def def_var(self, name: str, dtype, dimids=()) -> int:
        """Define a variable; returns its varid.

        ``dtype`` is anything ``numpy.dtype`` accepts, so a schema table
        can carry ``"f4"`` / ``"i4"`` / ``"S1"`` exactly as it does for
        netCDF4-python today.
        """

        nc_type = nc_type_of(dtype)
        raw = name.encode("utf-8")
        ids = list(int(value) for value in dimids)
        buffer = (ctypes.c_uint64 * max(len(ids), 1))(*ids) if ids else None
        varid = int(self._library.gpuwm_ncwrite_def_var(
            self._live(), raw, len(raw), nc_type, len(ids),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint64))
            if buffer is not None else None))
        if varid < 0:
            raise NcWriteError(_last_error(self._library))
        self._var_types[varid] = nc_type
        return varid

    def _put_att(self, varid: int, name: str, value, dtype=None) -> None:
        raw = name.encode("utf-8")
        if isinstance(value, str):
            payload = value.encode("utf-8")
            nc_type, nelems = NC_CHAR, len(payload)
        elif isinstance(value, (bytes, bytearray)):
            payload = bytes(value)
            nc_type, nelems = NC_CHAR, len(payload)
        else:
            array = np.asarray(value if dtype is None else
                               np.asarray(value, dtype=dtype))
            array = np.atleast_1d(array)
            nc_type = nc_type_of(array.dtype)
            array = np.ascontiguousarray(
                array.astype(array.dtype.newbyteorder("="), copy=False))
            payload, nelems = array.tobytes(), int(array.size)
        self._check(self._library.gpuwm_ncwrite_put_att(
            self._live(), varid, raw, len(raw), nc_type, nelems, payload))

    def put_global_attr(self, name: str, value, dtype=None) -> None:
        self._put_att(-1, name, value, dtype)

    def put_var_attr(self, varid: int, name: str, value, dtype=None) -> None:
        self._put_att(int(varid), name, value, dtype)

    def create(self, path) -> ClassicWriter:
        """Create the file and hand back the writer. Consumes the schema."""

        raw = str(path).encode("utf-8")
        handle = self._live()
        self._handle = None  # the Rust side owns it from here
        writer = self._library.gpuwm_ncwrite_create(handle, raw, len(raw))
        if not writer:
            raise NcWriteError(_last_error(self._library))
        return ClassicWriter(self._library, writer, dict(self._var_types),
                             Path(str(path)))

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        handle, self._handle = getattr(self, "_handle", None), None
        if handle is not None:
            try:
                self._library.gpuwm_ncwrite_schema_free(handle)
            except Exception:
                pass


__all__ = [
    "ABI_MARKER",
    "ClassicSchema",
    "ClassicWriter",
    "FORMATS",
    "NCWRITE_ABI",
    "NCWRITE_BRIDGE_ENV",
    "NcWriteError",
    "library_candidates",
    "load",
    "nc_type_of",
    "resolve_ncwrite_bridge",
    "scan_nonfinite",
    "unavailable_reason",
]
