"""ERA5 GRIB1 decoding through the group's all-Rust weather stack.

The Rust bridge deliberately has a narrow contract: it unpacks every GRIB1
message into one little-endian float64 stream and writes structural metadata
as JSON.  This module owns the case-specific Vtable mapping and assembles
immutable, validated :class:`Era5Snapshot` objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)


@dataclass(frozen=True)
class VtableEntry:
    """One GRIB1 row from a WPS Vtable."""

    parameter: int | None
    level_type: int
    level1: str
    level2: str
    name: str
    units: str
    description: str


def parse_vtable(path: str | Path) -> tuple[VtableEntry, ...]:
    """Parse parameter rows from a pipe-delimited WPS Vtable.

    Blank-parameter derived rows are retained with ``parameter=None``.  Header,
    separator, comment, and empty lines are ignored.
    """

    entries: list[VtableEntry] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if "|" not in line or line.lstrip().startswith("#"):
                continue
            columns = [column.strip() for column in line.split("|")]
            if len(columns) < 7:
                continue
            try:
                level_type = int(columns[1])
            except ValueError:
                continue
            parameter = None
            if columns[0]:
                try:
                    parameter = int(columns[0])
                except ValueError:
                    continue
            name = columns[4]
            if not name:
                continue
            entries.append(
                VtableEntry(
                    parameter=parameter,
                    level_type=level_type,
                    level1=columns[2],
                    level2=columns[3],
                    name=name,
                    units=columns[5],
                    description=columns[6],
                )
            )
    if not entries:
        raise ValueError(f"no GRIB1 parameter rows found in Vtable {path}")
    return tuple(entries)


# The output names make repeated WPS names (TT/UU/VV at pressure and near the
# surface) unambiguous.  Each tuple also pins the expected Vtable spelling so a
# changed or wrong Vtable fails instead of silently remapping a parameter.
_CANONICAL_SPECS: dict[tuple[int, int], tuple[str, str]] = {
    (129, 100): ("GEOPT", "Z"),
    (156, 100): ("HGT", "HGT"),
    (130, 100): ("TT", "T"),
    (131, 100): ("UU", "U"),
    (132, 100): ("VV", "V"),
    (157, 100): ("RH", "RH"),
    (165, 1): ("UU", "U10"),
    (166, 1): ("VV", "V10"),
    (167, 1): ("TT", "T2"),
    (168, 1): ("DEWPT", "D2"),
    (172, 1): ("LANDSEA", "LANDSEA"),
    (129, 1): ("SOILGEO", "SOILGEO"),
    (156, 1): ("SOILHGT", "SOILHGT"),
    (134, 1): ("PSFC", "PSFC"),
    (151, 1): ("PMSL", "PMSL"),
    (235, 1): ("SKINTEMP", "SKINTEMP"),
    (31, 1): ("SEAICE", "SEAICE"),
    (34, 1): ("SST", "SST"),
    (141, 1): ("SNOW_EC", "SNOW_EC"),
    (139, 1): ("ST000007", "ST000007"),
    (170, 1): ("ST007028", "ST007028"),
    (183, 1): ("ST028100", "ST028100"),
    (236, 1): ("ST100289", "ST100289"),
    (39, 1): ("SM000007", "SM000007"),
    (40, 1): ("SM007028", "SM007028"),
    (41, 1): ("SM028100", "SM028100"),
    (42, 1): ("SM100289", "SM100289"),
}

# Native CDS GRIB1 encodes the four soil temperature/moisture layers as
# level type 112 (depth below land, centimetres).  The long-standing
# Vtable.ERA5_CDO describes CDO-normalized files where those same parameters
# have been flattened to level type 1.  Accept both encodings under one field
# contract; no external normalization step is required for native CDS files.
_NATIVE_LEVEL_ALIASES: dict[tuple[int, int], tuple[int, int]] = {
    (139, 112): (139, 1),
    (170, 112): (170, 1),
    (183, 112): (183, 1),
    (236, 112): (236, 1),
    (39, 112): (39, 1),
    (40, 112): (40, 1),
    (41, 112): (41, 1),
    (42, 112): (42, 1),
}


def _canonical_mapping(entries: tuple[VtableEntry, ...]) -> dict[tuple[int, int], str]:
    by_key = {
        (entry.parameter, entry.level_type): entry.name
        for entry in entries
        if entry.parameter is not None
    }
    mapping: dict[tuple[int, int], str] = {}
    for key, (vtable_name, canonical_name) in _CANONICAL_SPECS.items():
        found = by_key.get(key)
        if found is None:
            continue
        if found != vtable_name:
            raise ValueError(
                f"Vtable key {key} names {found!r}; expected {vtable_name!r}"
            )
        mapping[key] = canonical_name
    for native_key, normalized_key in _NATIVE_LEVEL_ALIASES.items():
        if normalized_key in mapping:
            mapping[native_key] = mapping[normalized_key]
    return mapping


def canonical_units(entries: tuple[VtableEntry, ...]) -> Mapping[str, str]:
    """Return canonical gpuwm field names mapped to declared Vtable units."""

    mapping = _canonical_mapping(entries)
    by_key = {
        (entry.parameter, entry.level_type): entry.units
        for entry in entries
        if entry.parameter is not None
    }
    return MappingProxyType({
        name: by_key[key if key in by_key else _NATIVE_LEVEL_ALIASES[key]]
        for key, name in mapping.items()
    })


@dataclass(frozen=True)
class Era5Snapshot:
    """All decoded ERA5 fields at one UTC valid time.

    Arrays are CPU-side float64 setup data.  Pressure-level fields have shape
    ``(nlevel, nlat, nlon)`` and surface fields have shape ``(nlat, nlon)``.
    Inputs are copied and made read-only so the frozen object is meaningful.
    """

    valid_time: datetime
    levels_hpa: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    fields: Mapping[str, np.ndarray]
    #: ``None`` for a geographic regular grid (the historical meaning of
    #: ``latitude``/``longitude``).  For a source regular in its own
    #: PROJECTION plane, ``{"family": ..., "parameters": {...}}`` from the
    #: mapped grid declaration -- and the two axis arrays are then the
    #: projected y/x coordinates in ``parameters["axis_unit_m"]`` units.
    #: Every downstream consumer that pairs these axes with target
    #: coordinates must transform the target into the same plane
    #: (:func:`gpuwm.ingest.horiz.source_coordinate_transform`).
    projection: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid_time, datetime):
            raise TypeError("valid_time must be a datetime")
        if self.valid_time.tzinfo is not None:
            raise ValueError("valid_time must be naive UTC")
        if self.projection is not None:
            projection = dict(self.projection)
            if not isinstance(projection.get("family"), str) \
                    or not isinstance(projection.get("parameters"), Mapping):
                raise ValueError(
                    "snapshot projection must carry family and parameters")
            object.__setattr__(
                self, "projection",
                MappingProxyType({
                    "family": str(projection["family"]),
                    "parameters": MappingProxyType(
                        dict(projection["parameters"])),
                }),
            )

        axes: dict[str, np.ndarray] = {}
        for name in ("levels_hpa", "latitude", "longitude"):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.float64:
                raise TypeError(f"{name} must have dtype float64")
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f"{name} must be a non-empty 1-D array")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            copied = value.copy()
            copied.setflags(write=False)
            axes[name] = copied

        nlev = axes["levels_hpa"].size
        horizontal = (axes["latitude"].size, axes["longitude"].size)
        copied_fields: dict[str, np.ndarray] = {}
        for name, raw in self.fields.items():
            if not isinstance(name, str) or not name:
                raise TypeError("field names must be non-empty strings")
            value = np.asarray(raw)
            if value.dtype != np.float64:
                raise TypeError(f"field {name} must have dtype float64")
            field_horizontal = self._field_horizontal_shape(name, horizontal)
            allowed_shapes = (field_horizontal, (nlev, *field_horizontal))
            mapped_soil_shape = (
                name in {MAPPED_SOIL_TEMPERATURE, MAPPED_SOIL_MOISTURE}
                and value.ndim == 3
                and value.shape[0] > 0
                and value.shape[1:] == horizontal
            )
            if value.shape not in allowed_shapes and not mapped_soil_shape:
                raise ValueError(
                    f"{name} has shape {value.shape}; expected {horizontal} "
                    f"or {(nlev, *horizontal)}"
                )
            copied = value.copy()
            copied.setflags(write=False)
            copied_fields[name] = copied
            # The item producer may read the next full field on resumption.
            del raw, value, copied

        object.__setattr__(self, "levels_hpa", axes["levels_hpa"])
        object.__setattr__(self, "latitude", axes["latitude"])
        object.__setattr__(self, "longitude", axes["longitude"])
        object.__setattr__(self, "fields", MappingProxyType(copied_fields))

    def _field_horizontal_shape(self, name, horizontal):
        return horizontal

    @classmethod
    def from_field_items(
        cls, *, valid_time: datetime, levels_hpa: np.ndarray,
        latitude: np.ndarray, longitude: np.ndarray,
        field_items: Iterable[tuple[str, np.ndarray]],
        projection: Mapping[str, object] | None = None,
        **snapshot_options,
    ) -> Era5Snapshot:
        """Consume field items once, copying each before requesting the next.

        This uses the same owning constructor and validation as a mapping.
        A producer can release each decoded field after the constructor has
        copied it, instead of keeping a second full atmosphere alive.
        """
        class FieldItems:
            def items(self):
                seen = set()
                for name, values in field_items:
                    if name in seen:
                        raise ValueError(f"repeated snapshot field {name!r}")
                    seen.add(name)
                    yield name, values
                    del values

        return cls(valid_time=valid_time, levels_hpa=levels_hpa,
                   latitude=latitude, longitude=longitude,
                   fields=FieldItems(), projection=projection, **snapshot_options)

    def with_fields(self, replacements: Mapping[str, np.ndarray]) -> Era5Snapshot:
        """Copy validated replacements, sharing this snapshot's read-only fields.

        A surface overlay changes a few 2-D arrays. Copying the independent
        3-D atmosphere again needlessly doubles its residency. Construction
        still owns every supplied replacement and enforces its dtype/shape;
        unchanged fields already belong to this immutable snapshot.
        """
        from dataclasses import replace

        if not replacements:
            return self
        updated = replace(self, fields=replacements)
        object.__setattr__(updated, "fields", MappingProxyType({
            **self.fields, **updated.fields}))
        return updated

    def save_npz(self, path: str | Path) -> None:
        """Write a pickle-free ``np.savez`` snapshot archive."""

        if self.projection is not None:
            # The archive carries no projection record, so a reload would
            # silently reinterpret projected axes as geographic degrees --
            # mis-georeferencing with nothing that looks like a failure.
            raise ValueError(
                "save_npz does not persist a source projection; a "
                "projected snapshot cannot round-trip through this archive")
        names = tuple(self.fields)
        payload: dict[str, np.ndarray] = {
            "valid_time": np.asarray(self.valid_time.isoformat(timespec="seconds")),
            "levels_hpa": self.levels_hpa,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "field_names": np.asarray(names, dtype=np.str_),
        }
        payload.update({f"field__{name}": self.fields[name] for name in names})
        np.savez(Path(path), **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> Era5Snapshot:
        """Load an archive written by :meth:`save_npz` without pickle."""

        with np.load(Path(path), allow_pickle=False) as archive:
            names = tuple(str(name) for name in archive["field_names"].tolist())
            fields = {name: np.asarray(archive[f"field__{name}"]).copy() for name in names}
            return cls(
                valid_time=datetime.fromisoformat(str(archive["valid_time"].item())),
                levels_hpa=np.asarray(archive["levels_hpa"]).copy(),
                latitude=np.asarray(archive["latitude"]).copy(),
                longitude=np.asarray(archive["longitude"]).copy(),
                fields=fields,
            )


@dataclass(frozen=True)
class Grib1Envelope:
    """One strictly validated message envelope in a concatenated GRIB1 file."""

    index: int
    offset: int
    length: int


def inspect_grib1_envelopes(path: str | Path) -> tuple[Grib1Envelope, ...]:
    """Validate every GRIB1 envelope and exact EOF coverage.

    This setup-time scan is deliberately independent of the decoder.  It
    prevents a malformed later message from being mistaken for a successful
    partial file and gives truncated-input errors a file/message/byte address.
    The vendored decoder performs the deeper section and packed-bit checks.
    """

    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"GRIB1 file {path} is empty")
    envelopes: list[Grib1Envelope] = []
    with path.open("rb") as stream:
        offset = 0
        while offset < size:
            stream.seek(offset)
            indicator = stream.read(8)
            index = len(envelopes)
            if len(indicator) != 8:
                raise ValueError(
                    f"truncated GRIB1 file {path}: message {index} at byte "
                    f"{offset} has only {len(indicator)} of 8 indicator bytes"
                )
            if indicator[:4] != b"GRIB":
                raise ValueError(
                    f"invalid GRIB1 file {path}: message {index} at byte "
                    f"{offset} has marker {indicator[:4]!r}, expected b'GRIB'"
                )
            if indicator[7] != 1:
                raise ValueError(
                    f"unsupported GRIB edition {indicator[7]} in {path}, "
                    f"message {index} at byte {offset}; native input must be "
                    "GRIB1"
                )
            length = int.from_bytes(indicator[4:7], "big")
            if length < 12:
                raise ValueError(
                    f"invalid GRIB1 file {path}: message {index} at byte "
                    f"{offset} declares length {length}, minimum is 12"
                )
            end = offset + length
            if end > size:
                raise ValueError(
                    f"truncated GRIB1 file {path}: message {index} at byte "
                    f"{offset} declares end byte {end}, file has {size} bytes"
                )
            stream.seek(end - 4)
            if stream.read(4) != b"7777":
                raise ValueError(
                    f"invalid GRIB1 file {path}: message {index} at byte "
                    f"{offset} lacks the 7777 terminator at byte {end - 4}"
                )
            envelopes.append(Grib1Envelope(index, offset, length))
            offset = end
    return tuple(envelopes)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_rust_bridge(*, release: bool = True) -> Path:
    """Resolve -- building only in a checkout -- the vendored bridge.

    Resolution order (the shared mechanism in :mod:`gpuwm.bridges`):

    1. ``GPUWM_GRIB1_BRIDGE`` names a prebuilt executable (a missing
       file it names is a hard error, never a silent fall-through);
    2. a source checkout's ``tools/grib1_bridge`` crate exists: cargo
       builds it locked/offline exactly as before (the developer path);
    3. an already-built executable in ``<root>/libexec/bridges`` or
       ``~/.gpuwm/bridges`` -- installed wheels ship no compiled Rust,
       so this is the wheel user's route after one cargo build in a
       clone.

    Failure names every searched location and the exact remedy;
    ``gpuwm doctor`` reports the same estate.
    """

    from gpuwm import bridges

    override = os.environ.get(bridges.BRIDGE_ENV["grib1_bridge"])
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(
                f"{bridges.BRIDGE_ENV['grib1_bridge']} names a missing "
                f"file: {path}")
        return path.resolve()
    crate = _repo_root() / "tools" / "grib1_bridge"
    manifest = crate / "Cargo.toml"
    if manifest.is_file():
        command = ["cargo", "build", "--locked", "--offline"]
        if release:
            command.append("--release")
        try:
            result = subprocess.run(
                command, cwd=crate, text=True, capture_output=True,
                check=False
            )
        except OSError:
            # `cargo` itself could not be started.  A refusal naming the
            # missing toolchain, never a bare WinError 2 traceback out
            # of subprocess.
            raise bridges.BridgeBuildError(
                bridges.cargo_missing_refusal(
                    "grib1_bridge", bridges.CRATE_RELATIVE),
                failure_class="cargo-not-installed") from None
        if result.returncode:
            # A BUILD failure, named by class -- so it can never reach a
            # caller wearing a data failure's face.  The reproduction:
            # `gpuwm check --alloc` in a fresh worktree, with this
            # crate's cdylib held open by another process, relayed
            # cargo's whole warning wall under "could not decode/merge
            # forcing inputs" and then reported five more failures about
            # forcing fields nothing had read.
            detail = "\n".join(part for part in (result.stdout, result.stderr)
                               if part)
            raise bridges.BridgeBuildError(
                bridges.cargo_build_refusal(
                    "grib1_bridge", bridges.CRATE_RELATIVE,
                    returncode=result.returncode, output=detail),
                failure_class=bridges.classify_cargo_failure(detail)[0])
        profile = "release" if release else "debug"
        suffix = ".exe" if os.name == "nt" else ""
        executable = crate / "target" / profile / f"grib1_bridge{suffix}"
        if not executable.is_file():
            raise bridges.BridgeBuildError(
                f"cargo reported success in {bridges.CRATE_RELATIVE} but "
                f"the bridge it should have produced is not there: "
                f"{executable}.\n"
                "  why: the build wrote somewhere else (a CARGO_TARGET_DIR "
                "in this environment) or was interrupted between linking "
                "and rename, so nothing here can decode GRIB1.\n"
                "  remedy: " + bridges.install_aware_one_line_hint(
                    bridges.CARGO_BUILD_HINT, bridges.CRATE_RELATIVE,
                    "grib1_bridge"),
                failure_class="artifact-absent-after-build")
        return executable
    prebuilt = bridges.find_bridge("grib1_bridge")
    if prebuilt is not None:
        return prebuilt
    raise FileNotFoundError(
        "the Rust GRIB1 bridge is not available: this installation has "
        f"no {manifest} crate (pip wheels do not ship compiled Rust) and "
        "no prebuilt executable was found; "
        + bridges.bridge_remedy("grib1_bridge")
        + "\n  `gpuwm doctor` checks this estate.")


def _valid_time(message: Mapping[str, object]) -> datetime:
    reference = datetime(
        int(message["year"]), int(message["month"]), int(message["day"]),
        int(message["hour"]), int(message["minute"]),
    )
    p1 = int(message["p1"])
    p2 = int(message["p2"])
    time_range = int(message["time_range_indicator"])
    unit = int(message["time_unit"])
    multipliers = {
        0: timedelta(minutes=1),
        1: timedelta(hours=1),
        2: timedelta(days=1),
        10: timedelta(hours=3),
        11: timedelta(hours=6),
        12: timedelta(hours=12),
        254: timedelta(seconds=1),
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported GRIB1 forecast time unit {unit}")
    # WMO Table 5 (PDS octet 21) decides what octets 19-20 mean.  Octet 19
    # alone is the lead only for indicator 0; indicator 10 spans octets 19-20
    # as one 16-bit P1, so reading P1 alone silently drops its low byte.  The
    # interval indicators (2/3/4/5) are valid at reference + P2 and carry
    # interval quantities, not the instantaneous values this decoder stores.
    # Same closed vocabulary as the fetch-side census in :mod:`gpuwm.fetch`.
    if time_range in (0, 1):
        lead = p1 if time_range == 0 else 0
    elif time_range == 10:
        lead = (p1 << 8) | p2
    else:
        raise ValueError(
            f"unsupported GRIB1 time range indicator {time_range}; forcing "
            "records must be instantaneous"
        )
    return reference + lead * multipliers[unit]


@dataclass(frozen=True)
class _PartialSnapshot:
    source: Path
    valid_time: datetime
    levels_hpa: tuple[int, ...]
    latitude: np.ndarray
    longitude: np.ndarray
    fields: Mapping[str, np.ndarray]
    bitmap_missing: Mapping[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class Era5DecodeResult:
    """Merged forcing snapshots plus per-field source-file provenance."""

    snapshots: tuple[Era5Snapshot, ...]
    field_sources: Mapping[tuple[datetime, str], tuple[Path, ...]]
    bitmap_missing: Mapping[tuple[datetime, str], np.ndarray] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_sources", MappingProxyType({
            key: tuple(paths) for key, paths in self.field_sources.items()
        }))
        snapshots = {snapshot.valid_time: snapshot for snapshot in self.snapshots}
        copied: dict[tuple[datetime, str], np.ndarray] = {}
        for key, raw in self.bitmap_missing.items():
            valid_time, name = key
            snapshot = snapshots.get(valid_time)
            if snapshot is None or name not in snapshot.fields:
                raise ValueError(
                    f"bitmap provenance names absent field {name} at {valid_time}"
                )
            mask = np.asarray(raw, dtype=bool)
            value = snapshot.fields[name]
            if mask.shape != value.shape:
                raise ValueError(
                    f"bitmap provenance for {name} at {valid_time} has shape "
                    f"{mask.shape}; expected {value.shape}"
                )
            if not np.array_equal(mask, ~np.isfinite(value)):
                raise ValueError(
                    f"bitmap provenance for {name} at {valid_time} does not "
                    "match its decoded missing values"
                )
            mask = mask.copy()
            mask.setflags(write=False)
            copied[key] = mask
        object.__setattr__(self, "bitmap_missing", MappingProxyType(copied))


def _load_bridge_partials(
    directory: Path, entries: tuple[VtableEntry, ...], source: Path,
    *, envelope_count: int | None = None,
) -> tuple[_PartialSnapshot, ...]:
    metadata_path = directory / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format_version") != 1 or metadata.get("edition") != 1:
        raise ValueError("bridge output is not gpuwm GRIB1 dump format version 1")
    if metadata.get("dtype") != "<f8":
        raise ValueError(f"unsupported bridge dtype {metadata.get('dtype')!r}")
    if envelope_count is not None and len(metadata["messages"]) != envelope_count:
        # The decoder skips a message whose sections do not parse and still
        # exits 0, so a dump with fewer messages than the file has envelopes
        # is a silently dropped field, not a shorter file.  Envelope counting
        # is duplicated on both sides of the boundary for the same reason the
        # envelope walk itself is.
        raise ValueError(
            f"bridge decoded {len(metadata['messages'])} messages from "
            f"{source}, which carries {envelope_count} GRIB1 message "
            "envelopes; a message was dropped by the decoder"
        )

    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"invalid bridge grid shape {shape}")
    latitude = np.asarray(metadata["latitude"], dtype=np.float64)
    longitude = np.asarray(metadata["longitude"], dtype=np.float64)
    if shape != (latitude.size, longitude.size):
        raise ValueError("bridge coordinate axes do not match its grid shape")

    values = np.fromfile(directory / "values.f64", dtype="<f8")
    canonical = _canonical_mapping(entries)
    layers: dict[datetime, dict[str, dict[int, np.ndarray]]] = {}
    surfaces: dict[datetime, dict[str, np.ndarray]] = {}
    surface_bitmaps: dict[datetime, dict[str, np.ndarray]] = {}
    level_bitmapped: dict[datetime, set[str]] = {}
    scan_modes: set[int] = set()
    for message in metadata["messages"]:
        key = (int(message["parameter"]), int(message["level_type"]))
        name = canonical.get(key)
        if name is None:
            continue
        count = int(message["count"])
        offset = int(message["offset_values"])
        if count != shape[0] * shape[1] or offset < 0 or offset + count > values.size:
            raise ValueError("mapped bridge message points outside the primary grid")
        # The bridge derives one latitude/longitude axis pair from the message
        # with the most points and writes every other message into the same
        # flat stream in that message's OWN storage order.  A point count is
        # neither a shape nor a scan order, so both have to be compared before
        # the reshape -- as the mapped-source reader of this same
        # ``metadata.json`` already does.
        message_shape = (int(message["ny"]), int(message["nx"]))
        if message_shape != shape:
            raise ValueError(
                f"mapped bridge message {name} has grid {message_shape}, not "
                f"the primary grid {shape}"
            )
        scan_modes.add(int(message["scan_mode"]))
        if len(scan_modes) != 1:
            raise ValueError(
                "mapped bridge messages disagree on GRIB1 scanning mode "
                f"{sorted(scan_modes)}; they cannot share one axis pair"
            )
        valid_time = _valid_time(message)
        field = np.asarray(values[offset:offset + count], dtype=np.float64).reshape(shape)
        bitmapped = message.get("has_bitmap") is True
        if not bitmapped and not np.isfinite(field).all():
            # Without a bitmap every grid point was coded, so a non-finite
            # cell is a decode fault carrying no provenance anyone can record
            # it under.  This is the ingest boundary: refuse it here rather
            # than let it reach the initial condition unannounced.
            raise ValueError(
                f"mapped bridge message {name} carries non-finite values with "
                "no bitmap to account for them"
            )
        if key[1] == 100:
            level = int(message["level"])
            level_fields = layers.setdefault(valid_time, {}).setdefault(name, {})
            if level in level_fields:
                raise ValueError(f"duplicate {name} at {level} hPa for {valid_time}")
            level_fields[level] = field
            if bitmapped:
                level_bitmapped.setdefault(valid_time, set()).add(name)
        else:
            surface_fields = surfaces.setdefault(valid_time, {})
            if name in surface_fields:
                raise ValueError(f"duplicate {name} for {valid_time}")
            surface_fields[name] = field
            if bitmapped:
                surface_bitmaps.setdefault(valid_time, {})[name] = ~np.isfinite(field)

    times = sorted(set(layers) | set(surfaces))
    if not times:
        raise ValueError("no Vtable-mapped messages found in bridge output")
    snapshots: list[_PartialSnapshot] = []
    inventories: set[frozenset[str]] = set()
    expected_levels: tuple[int, ...] | None = None
    for valid_time in times:
        pressure = layers.get(valid_time, {})
        if pressure:
            level_sets = {
                tuple(sorted(by_level)) for by_level in pressure.values()
            }
            if len(level_sets) != 1:
                raise ValueError(
                    f"pressure fields have inconsistent levels for {valid_time}"
                )
            levels = next(iter(level_sets))
            if expected_levels is None:
                expected_levels = levels
            elif levels != expected_levels:
                raise ValueError("pressure levels differ between valid times")
            fields = {
                name: np.stack([by_level[level] for level in levels])
                for name, by_level in pressure.items()
            }
        else:
            levels = ()
            fields = {}
        # Pressure-level provenance is recorded per message but validated
        # against the stacked field, and every unbitmapped level above is
        # already known finite.
        bitmaps = {
            name: ~np.isfinite(fields[name])
            for name in level_bitmapped.get(valid_time, ())
        }
        fields.update(surfaces.get(valid_time, {}))
        bitmaps.update(surface_bitmaps.get(valid_time, {}))
        inventories.add(frozenset(fields))
        snapshots.append(
            _PartialSnapshot(
                source=source,
                valid_time=valid_time,
                levels_hpa=levels,
                latitude=latitude,
                longitude=longitude,
                fields=MappingProxyType(fields),
                bitmap_missing=MappingProxyType(bitmaps),
            )
        )
    if len(inventories) != 1:
        raise ValueError("Vtable-mapped field inventory differs between valid times")
    return tuple(snapshots)


def _catalog_selection_context(
    valid_times: Sequence[datetime], excluded_valid_times: Sequence[datetime]
) -> str:
    """Format catalog time authority for decode completeness diagnostics."""

    selected = ", ".join(value.isoformat() for value in valid_times) or "none"
    excluded = (", ".join(value.isoformat() for value in excluded_valid_times)
                or "none")
    return (f"catalog-selected valid times: [{selected}]; "
            f"catalog exclusions: [{excluded}]")


def _merge_partials(
    partials: Sequence[_PartialSnapshot], *,
    valid_times: Sequence[datetime] | None = None,
    excluded_valid_times: Sequence[datetime] = (),
) -> Era5DecodeResult:
    """Merge decoded products, optionally under catalog time authority.

    When ``valid_times`` is supplied, records outside that exact ordered
    selection are discarded before pressure/surface completeness checks.  CDS
    applies requested times per date, so multi-day products can legitimately
    contain cross-product records that the input catalog excluded.
    """

    selection = None if valid_times is None else tuple(valid_times)
    exclusions = tuple(excluded_valid_times)
    if selection is None and exclusions:
        raise ValueError(
            "excluded_valid_times requires a catalog valid_times selection")
    if selection is not None:
        if len(set(selection)) != len(selection):
            raise ValueError("catalog valid_times selection contains duplicates")
        selected = set(selection)
        partials = tuple(
            partial for partial in partials
            if partial.valid_time in selected
        )
        decoded_times = {partial.valid_time for partial in partials}
        missing = tuple(value for value in selection
                        if value not in decoded_times)
        if missing:
            missing_text = ", ".join(value.isoformat() for value in missing)
            raise ValueError(
                "forcing decode is incomplete for catalog selection; "
                f"no records for [{missing_text}]; "
                + _catalog_selection_context(selection, exclusions)
            )

    by_time: dict[datetime, list[_PartialSnapshot]] = {}
    for partial in partials:
        by_time.setdefault(partial.valid_time, []).append(partial)
    if not by_time:
        detail = ("" if selection is None else
                  "; " + _catalog_selection_context(selection, exclusions))
        raise ValueError("no Vtable-mapped snapshots were decoded" + detail)

    snapshots: list[Era5Snapshot] = []
    source_map: dict[tuple[datetime, str], list[Path]] = {}
    bitmap_map: dict[tuple[datetime, str], np.ndarray] = {}
    inventories: set[frozenset[str]] = set()
    expected_levels: tuple[int, ...] | None = None
    reference_grid: tuple[np.ndarray, np.ndarray] | None = None
    reference_time: datetime | None = None
    reference_sources: tuple[Path, ...] = ()
    for valid_time in sorted(by_time):
        group = by_time[valid_time]
        latitude = group[0].latitude
        longitude = group[0].longitude
        levels: tuple[int, ...] | None = None
        fields: dict[str, np.ndarray] = {}
        for partial in group:
            if (not np.array_equal(partial.latitude, latitude)
                    or not np.array_equal(partial.longitude, longitude)):
                raise ValueError(
                    f"forcing grids differ at {valid_time}: {group[0].source} "
                    f"versus {partial.source}"
                )
            if partial.levels_hpa:
                if levels is None:
                    levels = partial.levels_hpa
                elif levels != partial.levels_hpa:
                    raise ValueError(
                        f"pressure levels differ between input files at "
                        f"{valid_time}: {group[0].source} versus {partial.source}"
                    )
            for name, value in partial.fields.items():
                if name in fields:
                    prior = source_map[(valid_time, name)][0]
                    raise ValueError(
                        f"duplicate forcing variable {name} at {valid_time} "
                        f"in {prior} and {partial.source}"
                    )
                fields[name] = value
                source_map.setdefault((valid_time, name), []).append(partial.source)
                bitmap = partial.bitmap_missing.get(name)
                if bitmap is not None:
                    bitmap_map[(valid_time, name)] = bitmap
        if levels is None:
            names = ", ".join(str(partial.source) for partial in group)
            detail = ("" if selection is None else
                      "; " + _catalog_selection_context(
                          selection, exclusions))
            raise ValueError(
                f"no pressure-level fields for {valid_time} in {names}"
                + detail
            )
        if expected_levels is None:
            expected_levels = levels
        elif levels != expected_levels:
            raise ValueError("pressure levels differ between valid times")
        group_sources = tuple(sorted(
            {partial.source for partial in group}, key=lambda path: str(path)
        ))
        if reference_grid is None:
            reference_grid = (latitude, longitude)
            reference_time = valid_time
            reference_sources = group_sources
        elif (not np.array_equal(latitude, reference_grid[0])
              or not np.array_equal(longitude, reference_grid[1])):
            expected = ", ".join(str(path) for path in reference_sources)
            actual = ", ".join(str(path) for path in group_sources)
            raise ValueError(
                "forcing latitude/longitude grids differ across valid times: "
                f"{reference_time.isoformat()} in [{expected}] versus "
                f"{valid_time.isoformat()} in [{actual}]"
            )
        inventories.add(frozenset(fields))
        snapshots.append(Era5Snapshot(
            valid_time=valid_time,
            levels_hpa=np.asarray(levels, dtype=np.float64),
            latitude=latitude,
            longitude=longitude,
            fields=fields,
        ))
    if len(inventories) != 1:
        raise ValueError("Vtable-mapped field inventory differs between valid times")
    result = Era5DecodeResult(
        tuple(snapshots), {key: tuple(paths) for key, paths in source_map.items()},
        bitmap_map,
    )
    if selection is not None:
        actual = tuple(snapshot.valid_time for snapshot in result.snapshots)
        if actual != selection:
            actual_text = ", ".join(value.isoformat() for value in actual)
            raise ValueError(
                "forcing decode did not reproduce the catalog's ordered "
                f"valid-time selection; decoded: [{actual_text}]; "
                + _catalog_selection_context(selection, exclusions)
            )
    return result


def _merge_catalog_partials(
    partials: Sequence[_PartialSnapshot], *,
    valid_times: Sequence[datetime] | None = None,
    excluded_valid_times: Sequence[datetime] = (),
) -> Era5DecodeResult:
    """Merge under input-catalog time discovery or explicit authority.

    A single-level supplement may contain an invariant record at a valid time
    absent from the pressure-level product.  Such a record is an auxiliary
    product, not a forcing snapshot and must not create a catalog time.  During
    discovery only, retain times backed by at least one pressure-level partial
    and merge every complementary partial at those times.  Once the catalog
    supplies ``valid_times``, preserve the stricter exact-selection behavior.
    """

    if valid_times is not None:
        return _merge_partials(
            partials, valid_times=valid_times,
            excluded_valid_times=excluded_valid_times)
    pressure_times = tuple(sorted({
        partial.valid_time for partial in partials
        if partial.levels_hpa
    }))
    if not pressure_times:
        # Preserve _merge_partials' detailed source/time diagnostic when the
        # declared products contain no pressure-backed forcing snapshot.
        return _merge_partials(partials)
    pressure_set = set(pressure_times)
    auxiliary_only = tuple(sorted({
        partial.valid_time for partial in partials
        if partial.valid_time not in pressure_set
    }))
    exclusions = tuple(dict.fromkeys((
        *excluded_valid_times, *auxiliary_only)))
    return _merge_partials(
        partials, valid_times=pressure_times,
        excluded_valid_times=exclusions)


def inspect_era5_forcing_times(
    grib_paths: Sequence[str | Path], vtable_path: str | Path,
    *, bridge: str | Path | None = None,
) -> tuple[datetime, ...]:
    """Read pressure-backed catalog times through native GRIB1 headers only.

    This uses the decoder's Vtable mapping and valid-time interpretation.
    Surface-only auxiliary records do not introduce forcing snapshots, just
    as in :func:`_merge_catalog_partials`. Values are not decoded or staged.
    Full field and spatial validation remains the input catalog's job.
    """
    mapping = _canonical_mapping(parse_vtable(vtable_path))
    executable = Path(bridge) if bridge is not None else build_rust_bridge(release=True)
    times: set[datetime] = set()
    for raw_path in grib_paths:
        path = Path(raw_path)
        envelopes = inspect_grib1_envelopes(path)
        result = subprocess.run(
            [os.fspath(executable), "--inventory", os.fspath(path)],
            text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"Rust GRIB1 metadata inventory failed for {path}: "
                f"{(result.stderr or result.stdout).strip()}")
        metadata = json.loads(result.stdout)
        if (metadata.get("format_version") != 1 or metadata.get("edition") != 1
                or metadata.get("metadata_only") is not True):
            raise ValueError("bridge output is not native GRIB1 metadata inventory version 1")
        messages = metadata["messages"]
        if len(messages) != len(envelopes):
            raise ValueError(
                f"bridge inventoried {len(messages)} messages from {path}, "
                f"which carries {len(envelopes)} GRIB1 envelopes; a message was dropped")
        for message in messages:
            key = (int(message["parameter"]), int(message["level_type"]))
            if key in mapping and key[1] == 100:
                times.add(_valid_time(message))
    if not times:
        raise ValueError("supplied forcing has no Vtable-mapped pressure-level valid times")
    return tuple(sorted(times))


def _decode_bridge_partials(
    grib_path: Path,
    entries: tuple[VtableEntry, ...],
    executable: Path,
) -> tuple[_PartialSnapshot, ...]:
    envelopes = inspect_grib1_envelopes(grib_path)
    with tempfile.TemporaryDirectory(prefix="gpuwm-grib1-") as temporary:
        dump = Path(temporary) / "dump"
        result = subprocess.run(
            [os.fspath(executable), os.fspath(grib_path), os.fspath(dump)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Rust GRIB1 bridge failed for {grib_path}: {detail}"
            )
        return _load_bridge_partials(
            dump, entries, grib_path, envelope_count=len(envelopes)
        )


def decode_era5_grib(
    grib_path: str | Path,
    vtable_path: str | Path,
    *,
    bridge: str | Path | None = None,
) -> tuple[Era5Snapshot, ...]:
    """Decode one combined or per-time ERA5 GRIB1 file.

    ``bridge`` may name a prebuilt executable.  When omitted, the checked-in,
    fully vendored Rust bridge is built in locked/offline release mode.
    """

    grib_path = Path(grib_path)
    vtable_path = Path(vtable_path)
    if not grib_path.is_file():
        raise FileNotFoundError(grib_path)
    entries = parse_vtable(vtable_path)
    executable = Path(bridge) if bridge is not None else build_rust_bridge(release=True)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return _merge_partials(
        _decode_bridge_partials(grib_path, entries, executable)
    ).snapshots


def decode_era5_gribs(
    grib_paths: Sequence[str | Path],
    vtable_path: str | Path,
    *,
    bridge: str | Path | None = None,
    valid_times: Sequence[datetime] | None = None,
    excluded_valid_times: Sequence[datetime] = (),
) -> Era5DecodeResult:
    """Decode and merge complementary ERA5 GRIB1 files by valid time.

    Pressure-level and single-level products may be separate files.  Their
    mapped fields are merged only when coordinates and times agree exactly;
    duplicate variables remain a hard error.  The existing single-file
    :func:`decode_era5_grib` protocol and return type stay unchanged.
    """

    paths = tuple(Path(path) for path in grib_paths)
    if not paths:
        raise ValueError("at least one ERA5 GRIB1 input is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    entries = parse_vtable(vtable_path)
    executable = Path(bridge) if bridge is not None else build_rust_bridge(release=True)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    partials = [
        partial
        for path in paths
        for partial in _decode_bridge_partials(path, entries, executable)
    ]
    return _merge_catalog_partials(
        partials, valid_times=valid_times,
        excluded_valid_times=excluded_valid_times,
    )


@lru_cache(maxsize=8)
def _decode_era5_grib_resolved(
    grib_path: str, vtable_path: str, bridge: str | None
) -> tuple[Era5Snapshot, ...]:
    return decode_era5_grib(grib_path, vtable_path, bridge=bridge)


def cached_era5_snapshots(
    grib_path: str | Path,
    vtable_path: str | Path,
    *,
    bridge: str | Path | None = None,
) -> tuple[Era5Snapshot, ...]:
    """Decode with a cache KEYED BY RESOLVED INPUTS (Phase 5, Task 2).

    Retires the argument-less ``lru_cache(1)`` snapshot pattern: two
    different GRIB/Vtable pairs decoded in one process each get their own
    cache entry instead of silently sharing one, and the same resolved
    pair spelled through different relative paths shares one decode.  The
    Task-3 preflight later keys its :class:`InputCatalog` by SHA-256; this
    process-lifetime cache keys by resolved absolute path, which is
    sufficient for setup-time reuse within a single run.
    """
    grib_key = os.fspath(Path(grib_path).resolve())
    vtable_key = os.fspath(Path(vtable_path).resolve())
    bridge_key = (None if bridge is None
                  else os.fspath(Path(bridge).resolve()))
    return _decode_era5_grib_resolved(grib_key, vtable_key, bridge_key)


@lru_cache(maxsize=8)
def _decode_era5_forcing_partials_resolved(
    input_keys: tuple[tuple[str, str], ...],
    vtable_path: str,
    vtable_key: str,
    bridge: str | None,
) -> tuple[_PartialSnapshot, ...]:
    """Decode immutable input bytes once; selection-specific merges reuse it."""

    # Content keys participate in cache identity; bridge decode consumes paths.
    del vtable_key
    paths = tuple(Path(path) for path, _content_key in input_keys)
    entries = parse_vtable(vtable_path)
    executable = (Path(bridge) if bridge is not None
                  else build_rust_bridge(release=True))
    if not executable.is_file():
        raise FileNotFoundError(executable)
    return tuple(
        partial
        for path in paths
        for partial in _decode_bridge_partials(path, entries, executable)
    )


@lru_cache(maxsize=8)
def _decode_era5_gribs_resolved(
    input_keys: tuple[tuple[str, str], ...],
    vtable_path: str,
    vtable_key: str,
    bridge: str | None,
    valid_times: tuple[datetime, ...] | None,
    excluded_valid_times: tuple[datetime, ...],
) -> Era5DecodeResult:
    partials = _decode_era5_forcing_partials_resolved(
        input_keys, vtable_path, vtable_key, bridge)
    return _merge_catalog_partials(
        partials, valid_times=valid_times,
        excluded_valid_times=excluded_valid_times,
    )


def cached_era5_forcing(
    grib_paths: Sequence[str | Path],
    vtable_path: str | Path,
    *,
    bridge: str | Path | None = None,
    content_sha256: Sequence[str] | None = None,
    valid_times: Sequence[datetime] | None = None,
    excluded_valid_times: Sequence[datetime] = (),
) -> Era5DecodeResult:
    """Keyed multi-file snapshot service used by the input catalog.

    Resolved paths and content identities form the cache key.  Catalog callers
    pass their already-computed SHA-256 values; other callers get a stable
    size/mtime identity without hashing files a second time.  A runtime caller
    passes the catalog's selected and excluded valid times so completeness is
    evaluated only over the catalog-authorized records.
    """

    paths = tuple(Path(path).resolve() for path in grib_paths)
    if content_sha256 is not None:
        identities = tuple(str(value).lower() for value in content_sha256)
        if len(identities) != len(paths):
            raise ValueError("content_sha256 length must match grib_paths")
    else:
        identities = tuple(
            f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}"
            for stat in (path.stat() for path in paths)
        )
    input_keys = tuple(
        (os.fspath(path), identity)
        for path, identity in zip(paths, identities)
    )
    vtable = Path(vtable_path).resolve()
    stat = vtable.stat()
    vtable_key = f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}"
    bridge_key = (None if bridge is None
                  else os.fspath(Path(bridge).resolve()))
    return _decode_era5_gribs_resolved(
        input_keys, os.fspath(vtable), vtable_key, bridge_key,
        None if valid_times is None else tuple(valid_times),
        tuple(excluded_valid_times),
    )


def clear_forcing_caches() -> None:
    """Drop every decode this module memoized, releasing its host arrays.

    The three caches above are process-lifetime memoizations of pure
    functions of immutable input bytes, so clearing one cannot change a
    decoded value: the next call for the same key decodes the same file
    and rebuilds the same arrays.  It costs time and returns host memory.

    What it returns is not small.  A run reaches this module TWICE for
    one forcing product -- the input catalog decodes under its own
    discovery (``valid_times=None``) and the runtime decodes under the
    catalog's selection -- and the two keys are different, so the merged
    cache holds two SEPARATE frozen copies of every valid time on top of
    the partials both were merged from.  Nothing evicts them, because
    ``maxsize=8`` counts entries and one entry is a whole forcing window.

    Safe to call while a caller still holds snapshots.  Every
    :class:`Era5Snapshot` is frozen and COPIED out of its partials
    (:meth:`Era5Snapshot.__post_init__`), so a holder -- the input
    catalog holds its own tuple -- keeps exactly the snapshots it named
    and nothing else; this releases the partials they were copied from
    and every valid time nobody kept.
    """

    _decode_era5_grib_resolved.cache_clear()
    _decode_era5_forcing_partials_resolved.cache_clear()
    _decode_era5_gribs_resolved.cache_clear()


__all__ = [
    "Era5DecodeResult", "Era5Snapshot", "Grib1Envelope", "VtableEntry",
    "build_rust_bridge", "cached_era5_forcing", "cached_era5_snapshots",
    "canonical_units", "clear_forcing_caches", "decode_era5_grib",
    "decode_era5_gribs", "inspect_grib1_envelopes", "inspect_era5_forcing_times", "parse_vtable",
]
