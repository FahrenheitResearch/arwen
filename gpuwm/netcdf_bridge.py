"""The NetCDF read path: Rust decode, Python interpretation.

Every NetCDF meteorological source gpuwm reads is decoded by the
``rw_netcdf`` bridge (``tools/rustwx/crates/rw-netcdf``, built on the
vendored pure-Rust ``netcrust`` facade).  Python resolves selectors,
checks contracts and assembles frames; it does not decode the file.

This module used to have no reason to exist: :mod:`gpuwm.mapped_source`
opened arbitrary third-party NetCDF with ``netCDF4.Dataset`` -- the C
library -- while ``netcrust`` sat vendored in the same tree, already
driven by ``rw_wrfbatch``, ``rw_sat`` and ``rw_fieldcmp`` for exactly
this job.  The GRIB half of the very same module had always shelled out
to ``grib2_inventory`` / ``grib2_dump``.  The two formats now go through
one idiom.

The surface below is deliberately shaped like the small part of the
``netCDF4`` API that the mapping decoder actually used -- ``.variables``,
``.dimensions``, attribute access, ``variable[:]`` -- so the selector and
contract logic above it is untouched.  What changed underneath is who
reads the bytes.

There is NO fallback to a Python NetCDF library.  A missing bridge is a
named refusal (:class:`NetcdfBridgeMissing`) carrying the exact command
to fix it, because a silent degradation into a second decoder is how two
readers drift apart without anyone noticing.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

import numpy as np

from gpuwm.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           default_bridge_dir, ensure_executable,
                           executable_name, packaged_bridge_dir)

#: Environment variable naming a prebuilt ``rw_netcdf``.
NETCDF_ENV = "GPUWM_RW_NETCDF"

#: Logical name of the bridge, and so its filename stem.
NETCDF_NAME = "rw_netcdf"

#: Schemas this module speaks.  A bridge answering a different schema is
#: refused rather than parsed hopefully.
INVENTORY_SCHEMA = "gpuwm-rw-netcdf-inventory-v1"
DUMP_SCHEMA = "gpuwm-rw-netcdf-dump-v1"

_TIMEOUT_S = 900


class NetcdfBridgeMissing(RuntimeError):
    """The Rust NetCDF decoder is not installed, and nothing replaces it."""


class NetcdfDecodeError(RuntimeError):
    """The Rust NetCDF decoder refused a file or a variable."""


def _crate_dir() -> Path:
    return Path(__file__).resolve().parent.parent / RUSTWX_CRATE_RELATIVE


def netcdf_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for ``rw_netcdf``, best first.

    The same ladder every other bridge uses -- see :mod:`gpuwm.bridges`.
    """

    filename = executable_name(NETCDF_NAME)
    candidates: list[Path] = []
    override = os.environ.get(NETCDF_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent
    candidates.extend((
        _crate_dir() / "target" / "release" / filename,
        _crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_netcdf_bin() -> Path | None:
    """First existing candidate, or None.

    An environment override naming a missing file is a hard error:
    explicit configuration must fail loudly, not fall through.
    """

    override = os.environ.get(NETCDF_ENV)
    for candidate in netcdf_candidates():
        if candidate.is_file():
            return ensure_executable(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{NETCDF_ENV} names a missing file: {candidate}.  Point it "
                f"at a built {NETCDF_NAME} binary, or unset {NETCDF_ENV} to "
                f"use the resolution ladder.")
    return None


def netcdf_remedy() -> str:
    """The remedy for a missing decoder, true for THIS install."""

    return artifact_remedy(
        env_var=NETCDF_ENV, filename=executable_name(NETCDF_NAME),
        subject="the NetCDF decoder", crate_relative=RUSTWX_CRATE_RELATIVE)


def resolve_netcdf_bin() -> Path:
    """THE NetCDF decoder, or a refusal naming what to run.

    Never returns a Python decoder.  Reading a NetCDF meteorological
    source is decode work, and this project does decode in Rust; an
    install that cannot do it must say so at the front door rather than
    quietly producing numbers from a second implementation.
    """

    found = find_netcdf_bin()
    if found is not None:
        return found
    # `artifact_remedy` is already install-aware -- it prints the clone
    # for a wheel and the in-tree build for a checkout -- so nothing is
    # appended to it here.  A second, unconditional "or, from a
    # checkout" line was exactly the kind of remedy that names a
    # directory the reader does not have.
    raise NetcdfBridgeMissing(
        f"{NETCDF_NAME} is not installed, so this NetCDF source cannot be "
        f"decoded.  gpuwm decodes NetCDF in Rust and has no Python "
        f"fallback.\n\n{netcdf_remedy()}")


def _run(arguments: list[str], *, what: str) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        arguments, capture_output=True, text=True, timeout=_TIMEOUT_S)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise NetcdfDecodeError(f"{what}: {detail}")
    return completed


class Dimension:
    """One NetCDF dimension.

    ``len(dimension)`` rather than only ``.size`` because that is how
    callers ask -- ``len(dataset.dimensions[name])`` is the netCDF4
    spelling this replaces, and a bare integer here would make that a
    ``TypeError`` at the one place a soil-layer count is checked.
    """

    __slots__ = ("name", "size", "_unlimited")

    def __init__(self, name: str, size: int, *, unlimited: bool = False):
        self.name = name
        self.size = int(size)
        self._unlimited = bool(unlimited)

    def __len__(self) -> int:
        return self.size

    def __int__(self) -> int:
        return self.size

    def isunlimited(self) -> bool:
        return self._unlimited

    def __repr__(self) -> str:
        return f"<Dimension {self.name!r}: size={self.size}>"


class Attributes(dict):
    """Variable attributes, with ``netCDF4``'s accessor spellings."""

    def ncattrs(self) -> list[str]:
        return list(self)

    def getncattr(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


#: netcrust's stored-type name -> the numpy dtype ``netCDF4`` would
#: report for the same variable.
#:
#: Callers ask ``np.dtype(variable.dtype).kind`` to tell a character
#: coordinate from a numeric one, so ``.dtype`` has to be a real numpy
#: dtype rather than the reader's debug spelling.  ``Promoted`` is what
#: a recovered HDF5 dimension scale reports: the raw index does not
#: expose a stored type, and the values arrive as f64.
_NUMPY_DTYPE = {
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "F32": np.float32, "F64": np.float64,
    "Char": np.dtype("S1"), "String": np.dtype(object),
    "Promoted": np.float64,
}


class Variable:
    """One NetCDF variable: metadata now, values on demand.

    Values are pulled through the bridge the first time they are asked
    for and then cached, so resolving a selector -- which only reads
    metadata -- never decodes a byte of payload.
    """

    def __init__(self, dataset: "Dataset", record: Mapping[str, object]):
        self._dataset = dataset
        self.name = str(record["name"])
        self.dimensions = tuple(str(d) for d in record["dimensions"])
        self.shape = tuple(int(s) for s in record["shape"])
        #: The reader's own spelling, kept for refusals that quote it.
        self.stored_dtype = str(record["dtype"])
        self._attributes = Attributes(record.get("attributes") or {})
        self._values: np.ndarray | None = None
        self._times: tuple[datetime, ...] | None = None
        #: Whether the next read skips CF decoding; see set_auto_mask.
        self._raw = False
        self._scale = True

    @property
    def dtype(self):
        """The numpy dtype ``netCDF4`` would report for this variable."""

        mapped = _NUMPY_DTYPE.get(self.stored_dtype)
        if mapped is None:
            raise NetcdfDecodeError(
                f"{self.name}: {NETCDF_NAME} reported stored type "
                f"{self.stored_dtype!r}, which gpuwm.netcdf_bridge has no "
                f"numpy equivalent for; add it to _NUMPY_DTYPE rather than "
                f"guessing one")
        return np.dtype(mapped)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def is_character(self) -> bool:
        """True for a char/string variable, which cannot be decoded here.

        The vendored reader promotes numerics to f64 and exposes no
        character read at all, so a caller that needs the text has to be
        told by name rather than handed an empty array.
        """

        return self.stored_dtype in ("Char", "String")

    # Attribute access: `getattr(variable, "standard_name", None)` and
    # `variable.units` are how the mapping decoder asks, matching netCDF4.
    def __getattr__(self, name: str):
        try:
            return self._attributes[name]
        except KeyError:
            raise AttributeError(
                f"{self.name!r} has no NetCDF attribute {name!r}") from None

    def ncattrs(self) -> list[str]:
        return self._attributes.ncattrs()

    def getncattr(self, name: str):
        return self._attributes.getncattr(name)

    #: CF attributes whose presence makes "masked" and "raw" different
    #: answers for the same variable.
    _CF_MASKING_ATTRS = ("_FillValue", "missing_value",
                         "scale_factor", "add_offset")

    def set_auto_mask(self, on: bool) -> None:
        """``netCDF4``'s masking switch, honoured by the Rust decoder.

        ``True`` (the default) is CF decoding: ``_FillValue`` and
        ``missing_value`` become NaN and ``scale_factor``/``add_offset``
        are applied, all in Rust.

        ``False`` asks for the unmasked stored values instead, and it is
        a real request rather than a formality: the gridded radar and
        satellite products write ``_FillValue = -9.99e30`` and read that
        sentinel back deliberately, so handing them NaN would change
        what their consistency checks see.  The bridge therefore
        re-decodes with ``--no-mask`` rather than refusing -- the switch
        is answered where the decoding happens, not worked around here.

        It turns off masking ONLY.  ``scale_factor``/``add_offset`` are
        still applied, which is what netCDF4-python does and what this
        class emulates; :meth:`set_auto_maskandscale` is the switch that
        turns both off.  Aliasing the two -- which this bridge did until
        the values were compared against netCDF4 on a packed ``i2``
        variable -- silently returned stored counts where the caller
        expected physical units, on a route (GOES ABI) whose products are
        packed in the wild.
        """

        self._set_policy(raw=not on, scale=self._scale if on else True)

    def set_auto_maskandscale(self, on: bool) -> None:
        """``netCDF4``'s combined switch: masking AND scaling together."""

        self._set_policy(raw=not on, scale=bool(on))

    def _set_policy(self, *, raw: bool, scale: bool) -> None:
        if raw != self._raw or scale != self._scale:
            self._raw = raw
            self._scale = scale
            # The cached values were decoded under the other policy.
            self._values = None
            self._times = None

    @property
    def attributes(self) -> Attributes:
        return self._attributes

    def _load(self) -> None:
        if self._values is not None:
            return
        if self.is_character:
            # Named here rather than left to surface as a type error from
            # inside the reader.  This is a KNOWN gap in the vendored
            # netcrust facade, not a property of the file, and a caller
            # that needs the text should be told which is which.
            raise NetcdfDecodeError(
                f"{self.name} is a {self.stored_dtype} variable and cannot be "
                f"decoded: the vendored netcrust reader promotes numeric "
                f"variables to f64 and exposes no character read. Numeric "
                f"variables in the same file read normally.")
        self._values, self._times = self._dataset._decode(
            self.name, raw=self._raw, scale=self._scale)

    def __getitem__(self, item):
        self._load()
        assert self._values is not None
        values = self._values
        # A scalar variable is 0-dimensional, and `values[:]` on a 0-d
        # array is an IndexError -- but `variable[:]` is exactly how
        # netCDF4 spells "give me all of it", and callers use that
        # spelling uniformly.  ERA5 as delivered by the CDS carries a
        # scalar `number`, so this is a real shape, not a corner case.
        if values.ndim == 0 and isinstance(item, slice) and item == slice(None):
            return values[()]
        return values[item]

    def times(self) -> tuple[datetime, ...]:
        """Decoded CF instants, or a refusal.

        The calendar arithmetic happens in the bridge: turning
        ``hours since 1970-01-01`` and a number into an instant is
        decoding, not plumbing, and doing it here would be a second
        implementation of it.
        """

        self._load()
        if not self._times:
            raise NetcdfDecodeError(
                f"{self.name} has no CF reference-time units, so it cannot be "
                f"read as a time coordinate (units="
                f"{self._attributes.get('units')!r})")
        return self._times


class Dataset:
    """A NetCDF file, read through ``rw_netcdf``.

    Presents the read surface :mod:`gpuwm.mapped_source` uses. Opening
    runs the metadata pass only.
    """

    def __init__(self, path: Path | str, *, executable: Path | None = None):
        self.path = Path(path)
        # A file that is not there is FileNotFoundError, exactly as
        # netCDF4 raises it.  Without this the bridge's own "cannot
        # open" refusal arrives as a decode error, and callers that
        # distinguish "no product yet" from "bad product" -- the DA
        # cycle does -- would stop being able to.
        if not self.path.is_file():
            raise FileNotFoundError(
                f"no such NetCDF file: {self.path}")
        self._executable = Path(executable) if executable else resolve_netcdf_bin()
        completed = _run(
            [os.fspath(self._executable), "inventory", os.fspath(self.path)],
            what=f"NetCDF inventory failed for {self.path}")
        try:
            document = json.loads(completed.stdout)
        except ValueError as error:
            raise NetcdfDecodeError(
                f"{NETCDF_NAME} inventory for {self.path} is not JSON: {error}"
            ) from None
        if document.get("schema") != INVENTORY_SCHEMA:
            raise NetcdfDecodeError(
                f"{NETCDF_NAME} answered inventory schema "
                f"{document.get('schema')!r}, expected {INVENTORY_SCHEMA!r}; "
                f"the installed bridge does not speak this release's "
                f"contract.\n\n{netcdf_remedy()}")
        self.format = str(document.get("format", "unknown"))
        self.metadata = dict(document.get("metadata") or {})
        self.dimensions = {
            str(record["name"]): Dimension(
                str(record["name"]), int(record["len"]),
                unlimited=bool(record.get("unlimited")))
            for record in document.get("dimensions", [])
        }
        self.global_attributes = Attributes(
            document.get("global_attributes") or {})
        self.variables: dict[str, Variable] = {}
        for record in document.get("variables", []):
            variable = Variable(self, record)
            self.variables[variable.name] = variable

    # -- context manager, so `with Dataset(path) as dataset:` reads the
    # -- same as the netCDF4 call it replaces.
    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *_exception) -> bool:
        return False

    def close(self) -> None:
        return None

    def ncattrs(self) -> list[str]:
        return self.global_attributes.ncattrs()

    def getncattr(self, name: str):
        return self.global_attributes.getncattr(name)

    def set_auto_mask(self, on: bool) -> None:
        """Dataset-wide masking switch, applied to every variable.

        Same contract as :meth:`Variable.set_auto_mask`: masking on is
        what already happened, masking off is a no-op for variables that
        declare no CF masking attributes and a named refusal for any
        that do.
        """

        for variable in self.variables.values():
            variable.set_auto_mask(on)

    def set_auto_maskandscale(self, on: bool) -> None:
        self.set_auto_mask(on)

    def filepath(self) -> str:
        """The file this dataset reads, as ``netCDF4`` spells it.

        Callers put it in refusals -- "<path> is missing required field
        X" -- so a reader diagnosing a bad tape is told which tape.
        """

        return os.fspath(self.path)

    def __getattr__(self, name: str):
        """Global attributes as attributes: ``dataset.HYBRID_OPT``.

        The convention ``netCDF4`` established, and the one callers use
        through ``getattr(dataset, "ETAC", 0.2)``.  Only reached when
        normal lookup fails, so it can never shadow a real member.
        """

        try:
            return self.__dict__["global_attributes"][name]
        except KeyError:
            raise AttributeError(
                f"{self.path} has no global NetCDF attribute {name!r}"
            ) from None

    def _decode(self, name: str, *, raw: bool = False, scale: bool = True
                ) -> tuple[np.ndarray, tuple[datetime, ...]]:
        """Decode one variable through the bridge.

        ``raw`` turns masking off; ``scale`` says whether
        ``scale_factor``/``add_offset`` still apply while it is off.  The
        two are separate for the same reason netCDF4-python keeps them
        separate -- see :meth:`Variable.set_auto_mask`.
        """

        with tempfile.TemporaryDirectory(prefix="gpuwm-netcdf-") as temporary:
            out = Path(temporary)
            command = [os.fspath(self._executable), "dump"]
            if raw:
                command.append("--raw" if not scale else "--no-mask")
            command += [os.fspath(self.path), os.fspath(out), name]
            _run(command,
                 what=f"NetCDF decode failed for {name} in {self.path}")
            document = json.loads((out / "metadata.json").read_text("utf-8"))
            if document.get("schema") != DUMP_SCHEMA:
                raise NetcdfDecodeError(
                    f"{NETCDF_NAME} answered dump schema "
                    f"{document.get('schema')!r}, expected {DUMP_SCHEMA!r}")
            records = document.get("variables") or []
            if len(records) != 1:
                raise NetcdfDecodeError(
                    f"{NETCDF_NAME} dumped {len(records)} variables for "
                    f"{name}; expected exactly one")
            record = records[0]
            values = np.fromfile(out / str(record["filename"]), dtype="<f8")
            shape = tuple(int(s) for s in record["shape"])
            expected = int(np.prod(shape)) if shape else 1
            if values.size != expected:
                raise NetcdfDecodeError(
                    f"{name}: decoded {values.size} values but shape {shape} "
                    f"needs {expected}")
            values = values.reshape(shape)
            times = tuple(
                _parse_instant(text, name) for text in (record.get("times") or ())
            )
        return values, times


def _parse_instant(text: str, label: str) -> datetime:
    """One RFC-3339 instant from the bridge, as a naive UTC datetime.

    The bridge emits UTC and only UTC (a reference time spelled with a
    UTC offset is converted to UTC inside the decoder before any value
    is decoded), so the trailing ``Z`` is dropped rather than
    interpreted -- and the rest of the module keeps working in the
    naive-UTC datetimes it always used.
    """

    body = str(text)
    if not body.endswith("Z"):
        raise NetcdfDecodeError(
            f"{label}: decoded time {body!r} is not UTC-stamped")
    try:
        return datetime.fromisoformat(body[:-1])
    except ValueError as error:
        raise NetcdfDecodeError(
            f"{label}: cannot read decoded time {body!r}: {error}") from None


def open_dataset(path: Path | str, *, executable: Path | None = None) -> Dataset:
    """Open ``path`` for reading through the Rust decoder."""

    return Dataset(path, executable=executable)


__all__ = [
    "Attributes", "Dataset", "Dimension", "NetcdfBridgeMissing",
    "NetcdfDecodeError",
    "NETCDF_ENV", "NETCDF_NAME", "Variable", "find_netcdf_bin",
    "netcdf_candidates", "netcdf_remedy", "open_dataset", "resolve_netcdf_bin",
]
