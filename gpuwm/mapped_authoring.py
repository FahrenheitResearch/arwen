"""Fail-closed authoring for declarative mapped-source contracts.

The executable :mod:`gpuwm.mapped_source` contract is deliberately richer
than a classic WPS Vtable.  A Vtable carries GRIB identifiers, a WPS output
name, and a source-unit label; it does *not* define canonical field meaning,
axis order, missing-data policy, derivations, cadence, soil semantics, or the
target WRF contract.  This module therefore imports Vtable rows only when an
explicit ``rw-wps.descriptor.v1`` document supplies all of those semantics.

It also authors the exact ``gpuwm-mapped-composition-inputs-v1`` manifest used
by the runtime.  Outputs are create-only, canonical JSON and round-trip through
the same validators used by execution.  No source-family name is consulted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence
import uuid

from gpuwm.mapped_composition import (
    INPUT_MANIFEST_SCHEMA,
    _decoder_inventory,
    _verify_manifest,
    load_composition,
)
from gpuwm.mapped_source import (
    MAPPING_SCHEMA,
    _grib_selectors_overlap,
    _require_authority_snapshot,
    _sha256,
    _snapshot_authority,
    load_mapping,
)
from gpuwm.native_wrf_distribution import bridge_identity


DESCRIPTOR_SCHEMA = "rw-wps.descriptor.v1"
MAPPING_AUTHORING_SCHEMA = "rw-wps.mapping-authoring-receipt.v1"
MANIFEST_AUTHORING_SCHEMA = "rw-wps.input-manifest-authoring-receipt.v1"
_FORMATS = {"grib1", "grib2", "netcdf"}
_MAPPING_TOP_LEVEL = {
    "schema",
    "name",
    "format",
    "coordinates",
    "fields",
    "derivations",
    "target",
}
_FIELD_KEYS = {
    "selectors",
    "derivation",
    "units",
    "source_axes",
    "target_axes",
    "location",
    "staggering",
    "missing",
    "selector_stack_axis",
}
_VTABLE_REFERENCE_KEYS = {
    "metgrid_name",
    "grib1_level_type",
    "grib2_level_type",
    "level1",
    "level2",
    "selector",
}
_GRIB1_SELECTOR_OVERRIDES = {"table_version", "center", "level_value"}
_GRIB2_SELECTOR_OVERRIDES = {
    "center",
    "subcenter",
    "master_table_version",
    "local_table_version",
    "level_value",
    "second_level_type",
    "second_level_value",
    "member",
}
_MAX_STABLE_SNAPSHOT_BYTES = 128 * 1024 * 1024


def _object(
    value: object,
    label: str,
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing required key(s): {missing}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _u8(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError(f"{label} must be an integer in 0..255")
    return value


def _u16(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535:
        raise ValueError(f"{label} must be an integer in 0..65535")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one JSON object while rejecting every duplicate key.

    Python's default JSON decoder silently keeps the last occurrence.  That is
    unsafe for a science/selector authority because a visually earlier value
    can be replaced without any schema validator seeing the ambiguity.
    """

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json(data: bytes, label: str) -> object:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8 JSON: {error}") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_object_pairs)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error


def _write_new(path: Path, contents: bytes) -> None:
    """Publish bytes create-only through a same-directory hard link.

    The temporary file is fully written and fsynced before publication.
    ``os.link`` is an atomic no-clobber operation on the supported Windows and
    Linux filesystems, unlike a check followed by ``os.replace``.
    """

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _FileSnapshot:
    """Exact bytes and filesystem observation used as one authority."""

    path: Path
    data: bytes
    sha256: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    mode: int

    @property
    def fingerprint(self) -> tuple[str, int, int]:
        return self.sha256, self.size, self.mtime_ns


def _stable_file_snapshot(path: Path) -> _FileSnapshot:
    """Read one small authority once and bind validation to those exact bytes."""

    path = Path(path).resolve()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) \
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input authority is not a regular file: {path}")
        if before.st_size > _MAX_STABLE_SNAPSHOT_BYTES:
            raise ValueError(
                f"retained authority exceeds {_MAX_STABLE_SNAPSHOT_BYTES} "
                f"bytes: {path}"
            )
        data = stream.read()
        after = os.fstat(stream.fileno())
    # Windows' ``Path.stat`` may synthesize executable permission bits from
    # the filename extension while ``fstat`` on the already-open handle does
    # not (notably for ``*.exe`` decoder authorities).  File type remains an
    # identity property on Windows; the synthetic permission bits do not.
    # Keep the stricter complete-mode comparison on POSIX.
    def stable_mode(value: int) -> int:
        return stat.S_IFMT(value) if os.name == "nt" else value

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        stable_mode(before.st_mode),
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        stable_mode(after.st_mode),
    )
    if identity_after != identity_before or len(data) != after.st_size:
        raise ValueError(f"input authority changed while reading: {path}")
    current = path.stat()
    if (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        stable_mode(current.st_mode),
    ) != identity_after:
        raise ValueError(f"input authority path changed while reading: {path}")
    return _FileSnapshot(
        path=path,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mode=int(stable_mode(after.st_mode)),
    )


def _require_snapshot(snapshot: _FileSnapshot) -> None:
    """Reject any byte or identity drift from a validated snapshot."""

    current = _stable_file_snapshot(snapshot.path)
    if current != snapshot:
        raise ValueError(
            f"input authority changed after validation: {snapshot.path}"
        )


def _file_identity(path: Path) -> tuple[int, int]:
    status = path.stat()
    return int(status.st_dev), int(status.st_ino)


def _require_unique_identities(paths: Sequence[Path], label: str) -> None:
    identities = [_file_identity(path) for path in paths]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{label} contains filesystem aliases of one file")


@dataclass(frozen=True)
class VtableRow:
    """One losslessly parsed 11-column WPS Vtable data row."""

    line_number: int
    grib1_parameter: str
    grib1_level_type: str
    level1: str
    level2: str
    metgrid_name: str
    units: str
    description: str
    grib2_discipline: str
    grib2_category: str
    grib2_parameter: str
    grib2_level_type: str

    def identity(self) -> dict[str, object]:
        return {
            "line": self.line_number,
            "metgrid_name": self.metgrid_name,
            "grib1_level_type": self.grib1_level_type,
            "grib2_level_type": self.grib2_level_type,
            "level1": self.level1,
            "level2": self.level2,
            "units": self.units,
            "description": self.description,
        }


def _parse_wps_vtable_bytes(data: bytes, label: str) -> tuple[VtableRow, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8: {error}") from error
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        if stripped.startswith(("GRIB1", "Param", "-----")):
            continue
        columns = [column.strip() for column in line.split("|")]
        if columns and columns[-1] == "":
            columns.pop()
        if len(columns) != 11:
            raise ValueError(
                f"Vtable line {line_number} has {len(columns)} columns; "
                "expected exactly 11"
            )
        if not columns[4]:
            raise ValueError(f"Vtable line {line_number} has an empty metgrid name")
        rows.append(VtableRow(line_number, *columns))
    if not rows:
        raise ValueError("Vtable contains no 11-column data rows")
    return tuple(rows)


def parse_wps_vtable(path: str | Path) -> tuple[VtableRow, ...]:
    """Parse Vtable data rows without guessing any canonical semantics."""

    snapshot = _stable_file_snapshot(Path(path))
    rows = _parse_wps_vtable_bytes(snapshot.data, str(snapshot.path))
    _require_snapshot(snapshot)
    return rows


def _match_level(raw: str, expected: object, label: str) -> bool:
    if expected is None:
        return raw == ""
    if expected == "*":
        return raw == "*"
    numeric = _finite(expected, label)
    try:
        observed = float(raw)
    except ValueError:
        return False
    return observed == numeric


def _match_vtable_row(
    rows: Sequence[VtableRow],
    reference: Mapping[str, object],
    label: str,
) -> VtableRow:
    reference = _object(
        dict(reference),
        label,
        allowed=_VTABLE_REFERENCE_KEYS,
        required={"metgrid_name"},
    )
    name = _string(reference["metgrid_name"], f"{label}.metgrid_name")
    candidates = [row for row in rows if row.metgrid_name == name]
    if "grib1_level_type" in reference:
        expected = _u8(reference["grib1_level_type"], f"{label}.grib1_level_type")
        candidates = [
            row for row in candidates if row.grib1_level_type == str(expected)
        ]
    if "grib2_level_type" in reference:
        expected = _u8(reference["grib2_level_type"], f"{label}.grib2_level_type")
        candidates = [
            row for row in candidates if row.grib2_level_type == str(expected)
        ]
    for key in ("level1", "level2"):
        if key in reference:
            candidates = [
                row
                for row in candidates
                if _match_level(getattr(row, key), reference[key], f"{label}.{key}")
            ]
    if len(candidates) != 1:
        lines = [row.line_number for row in candidates]
        raise ValueError(
            f"{label} resolves {len(candidates)} Vtable rows at lines {lines}; "
            "add exact level metadata until it resolves one row"
        )
    return candidates[0]


def _parse_vtable_integer(raw: str, label: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{label} is not an integer: {raw!r}") from error
    return _u8(value, label)


def _row_level(raw: str, label: str) -> float | None:
    if raw in {"", "*"}:
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{label} is not numeric or '*': {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _selector_from_row(
    row: VtableRow,
    source_format: str,
    reference: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    raw_overrides = reference.get("selector", {})
    allowed = (
        _GRIB1_SELECTOR_OVERRIDES
        if source_format == "grib1"
        else _GRIB2_SELECTOR_OVERRIDES
    )
    overrides = _object(
        raw_overrides,
        f"{label}.selector",
        allowed=allowed,
    )
    if source_format == "grib1":
        missing_authorities = {"center", "table_version"} - set(overrides)
        if missing_authorities:
            raise ValueError(
                f"{label}.selector must explicitly bind GRIB1 center and "
                "table_version; a WPS Vtable does not carry both authorities"
            )
        if row.level2:
            raise ValueError(
                f"{label} selects Vtable line {row.line_number} with Level2; "
                "GRIB1 layer selectors are not represented by mapping.v1"
            )
        selector: dict[str, object] = {
            "format": "grib1",
            "parameter": _parse_vtable_integer(
                row.grib1_parameter, f"Vtable line {row.line_number} GRIB1 parameter"
            ),
            "level_type": _parse_vtable_integer(
                row.grib1_level_type,
                f"Vtable line {row.line_number} GRIB1 level type",
            ),
        }
        level = _row_level(row.level1, f"Vtable line {row.line_number} Level1")
        if level is not None:
            selector["level_value"] = level
            if "level_value" in overrides and _finite(
                overrides["level_value"],
                f"{label}.selector.level_value",
            ) != level:
                raise ValueError(
                    f"{label}.selector.level_value differs from numeric "
                    f"Vtable Level1 {level!r}"
                )
    else:
        selector = {
            "format": "grib2",
            "discipline": _parse_vtable_integer(
                row.grib2_discipline,
                f"Vtable line {row.line_number} GRIB2 discipline",
            ),
            "category": _parse_vtable_integer(
                row.grib2_category,
                f"Vtable line {row.line_number} GRIB2 category",
            ),
            "parameter": _parse_vtable_integer(
                row.grib2_parameter,
                f"Vtable line {row.line_number} GRIB2 parameter",
            ),
            "level_type": _parse_vtable_integer(
                row.grib2_level_type,
                f"Vtable line {row.line_number} GRIB2 level type",
            ),
        }
        vtable_level = _row_level(row.level1, f"Vtable line {row.line_number} Level1")
        if vtable_level is not None and "level_value" not in overrides:
            raise ValueError(
                f"{label} must explicitly bind selector.level_value; WPS Level1 "
                "units are not a general GRIB2 fixed-surface authority"
            )
        if row.level2 and not {
            "second_level_type",
            "second_level_value",
        }.issubset(overrides):
            raise ValueError(
                f"{label} must explicitly bind the GRIB2 second fixed surface"
            )
    for key, value in overrides.items():
        if source_format == "grib2" and key in {"center", "subcenter"}:
            selector[key] = _u16(value, f"{label}.selector.{key}")
        elif key in {
            "table_version",
            "center",
            "master_table_version",
            "local_table_version",
            "second_level_type",
            "member",
        }:
            selector[key] = _u8(value, f"{label}.selector.{key}")
        else:
            selector[key] = _finite(value, f"{label}.selector.{key}")
    missing_identifiers = {
        key
        for key in (
            ("parameter", "level_type", "center", "table_version")
            if source_format == "grib1"
            else ("discipline", "category", "parameter", "second_level_type")
        )
        if selector.get(key) == 255
    }
    if missing_identifiers:
        raise ValueError(
            f"{label} uses missing/undefined identifier code 255 for "
            f"{sorted(missing_identifiers)}"
        )
    if source_format == "grib2":
        if selector.get("level_type") == 255 and any(
            key in selector
            for key in ("level_value", "second_level_type", "second_level_value")
        ):
            raise ValueError(
                f"{label} uses GRIB2 level_type=255 (no fixed surface) with "
                "fixed-surface metadata"
            )
        local_identifiers = {
            key: value
            for key, value in selector.items()
            if key
            in {
                "discipline",
                "category",
                "parameter",
                "level_type",
                "second_level_type",
            }
            and isinstance(value, int)
            and 192 <= value <= 254
        }
        if local_identifiers:
            rendered = ", ".join(
                f"{key}={value}" for key, value in sorted(local_identifiers.items())
            )
            authority_keys = {
                "center",
                "subcenter",
                "master_table_version",
                "local_table_version",
            }
            missing_authority = sorted(authority_keys - set(selector))
            if missing_authority:
                raise ValueError(
                    f"{label} uses GRIB2 local-use identifier(s) {rendered}; "
                    "selector must explicitly bind center, subcenter, "
                    "master_table_version, and local_table_version (missing "
                    f"{missing_authority})"
                )
            if selector["local_table_version"] == 255:
                raise ValueError(
                    f"{label} uses GRIB2 local-use identifier(s) {rendered} "
                    "with local_table_version=255 (no local table)"
                )
    second = {
        key for key in ("second_level_type", "second_level_value") if key in selector
    }
    if second and second != {"second_level_type", "second_level_value"}:
        raise ValueError(
            f"{label}.selector second_level_type and second_level_value "
            "are an atomic pair"
        )
    return selector


def compile_mapping_descriptor(
    descriptor_path: str | Path,
    *,
    vtable_path: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compile one explicit descriptor into a validated mapping document."""

    descriptor_path = Path(descriptor_path).resolve()
    descriptor_snapshot = _stable_file_snapshot(descriptor_path)
    raw = _strict_json(descriptor_snapshot.data, f"descriptor {descriptor_path}")
    descriptor = _object(
        raw,
        "descriptor",
        allowed=_MAPPING_TOP_LEVEL,
        required={"schema", "name", "format", "coordinates", "fields", "target"},
    )
    if descriptor["schema"] != DESCRIPTOR_SCHEMA:
        raise ValueError(
            f"unsupported descriptor schema {descriptor['schema']!r}; "
            f"expected {DESCRIPTOR_SCHEMA!r}"
        )
    source_format = descriptor["format"]
    if source_format not in _FORMATS:
        raise ValueError(f"unsupported descriptor format {source_format!r}")
    if source_format == "netcdf":
        if vtable_path is not None:
            raise ValueError("NetCDF descriptors cannot use a WPS Vtable")
        rows: tuple[VtableRow, ...] = ()
        resolved_vtable = None
    else:
        if vtable_path is None:
            raise ValueError(f"{source_format} descriptors require a WPS Vtable")
        resolved_vtable = Path(vtable_path).resolve()
        vtable_snapshot = _stable_file_snapshot(resolved_vtable)
        rows = _parse_wps_vtable_bytes(
            vtable_snapshot.data,
            str(vtable_snapshot.path),
        )

    candidate = copy.deepcopy(descriptor)
    candidate["schema"] = MAPPING_SCHEMA
    fields = candidate["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError("descriptor.fields must be a non-empty object")
    selected_rows: list[dict[str, object]] = []
    consumed_lines: set[int] = set()
    consumed_selectors: list[tuple[str, int, dict[str, object]]] = []
    for field_name, raw_field in fields.items():
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("descriptor field names must be non-empty strings")
        field = _object(
            raw_field,
            f"descriptor.fields.{field_name}",
            allowed=_FIELD_KEYS | {"vtable_selectors"},
            required={"units", "source_axes", "target_axes", "location", "missing"},
        )
        imported = field.pop("vtable_selectors", None)
        derived = isinstance(field.get("derivation"), str) and bool(field["derivation"])
        if source_format == "netcdf":
            if imported is not None:
                raise ValueError(
                    f"NetCDF field {field_name!r} cannot import a WPS Vtable row"
                )
            continue
        if derived:
            if imported is not None:
                raise ValueError(
                    f"derived field {field_name!r} cannot import Vtable selectors"
                )
            continue
        if "selectors" in field:
            raise ValueError(
                f"GRIB descriptor field {field_name!r} must use "
                "vtable_selectors, not hand-authored selectors"
            )
        if not isinstance(imported, list) or not imported:
            raise ValueError(
                f"GRIB descriptor field {field_name!r} requires a non-empty "
                "vtable_selectors list"
            )
        units = field.get("units")
        if not isinstance(units, dict) or not isinstance(units.get("source"), str):
            raise ValueError(f"field {field_name!r} must declare source units")
        selectors = []
        for index, raw_reference in enumerate(imported):
            label = f"descriptor.fields.{field_name}.vtable_selectors[{index}]"
            reference = _object(
                raw_reference,
                label,
                allowed=_VTABLE_REFERENCE_KEYS,
                required={"metgrid_name"},
            )
            row = _match_vtable_row(rows, reference, label)
            if row.line_number in consumed_lines:
                raise ValueError(
                    f"Vtable line {row.line_number} is assigned more than once; "
                    "derive aliases explicitly instead"
                )
            consumed_lines.add(row.line_number)
            selector = _selector_from_row(
                row,
                str(source_format),
                reference,
                label,
            )
            for previous_field, previous_line, previous in consumed_selectors:
                if _grib_selectors_overlap(
                    previous,
                    selector,
                    str(source_format),
                ):
                    raise ValueError(
                        f"Vtable line {row.line_number} selector overlaps line "
                        f"{previous_line} assigned to field {previous_field!r}; "
                        "one GRIB record cannot directly provide two mapped "
                        "selector slots"
                    )
            consumed_selectors.append((field_name, row.line_number, selector))
            selectors.append(selector)
            selected_rows.append(
                {
                    "field": field_name,
                    "vtable": row.identity(),
                    "selector": selector,
                }
            )
        field["selectors"] = selectors

    # Reuse the executable engine's complete recursive validator before any
    # bytes become publishable.
    with tempfile.TemporaryDirectory(prefix="rw-wps-descriptor-") as directory:
        candidate_path = Path(directory) / "mapping.json"
        candidate_path.write_bytes(_canonical_json(candidate))
        validated = load_mapping(candidate_path)
    if validated != candidate:
        raise RuntimeError("descriptor compilation drifted through mapping validation")
    _require_snapshot(descriptor_snapshot)
    if resolved_vtable is not None:
        _require_snapshot(vtable_snapshot)
    evidence = {
        "schema": MAPPING_AUTHORING_SCHEMA,
        "status": "VALIDATED_NOT_STOCK_WRF_CERTIFIED",
        "descriptor": {
            "path": str(descriptor_path),
            "bytes": descriptor_snapshot.size,
            "sha256": descriptor_snapshot.sha256,
        },
        "vtable": None
        if resolved_vtable is None
        else {
            "path": str(resolved_vtable),
            "bytes": vtable_snapshot.size,
            "sha256": vtable_snapshot.sha256,
        },
        "format": source_format,
        "selected_rows": selected_rows,
    }
    return candidate, evidence


def _require_receipt_authorities(receipt: Mapping[str, object]) -> None:
    """Recheck descriptor/Vtable bytes bound by a compilation receipt."""

    for role in ("descriptor", "vtable"):
        raw = receipt.get(role)
        if raw is None:
            continue
        row = _object(
            raw,
            f"authoring receipt {role}",
            allowed={"path", "bytes", "sha256"},
            required={"path", "bytes", "sha256"},
        )
        snapshot = _stable_file_snapshot(Path(_string(row["path"], f"{role}.path")))
        if snapshot.size != row["bytes"] or snapshot.sha256 != row["sha256"]:
            raise ValueError(
                f"{role} authority changed after mapping compilation: "
                f"{snapshot.path}"
            )


def author_mapping(
    descriptor_path: str | Path,
    output_path: str | Path,
    *,
    vtable_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
    expected_format: str | None = None,
) -> dict[str, object]:
    """Create one validated mapping and its provenance receipt."""

    output_path = Path(output_path).resolve()
    receipt_path = (
        Path(receipt_path).resolve()
        if receipt_path is not None
        else output_path.with_name(f"{output_path.stem}.authoring.json")
    )
    if output_path == receipt_path:
        raise ValueError("mapping and authoring receipt paths must differ")
    for path in (output_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    mapping, receipt = compile_mapping_descriptor(
        descriptor_path,
        vtable_path=vtable_path,
    )
    if expected_format is not None and receipt["format"] != expected_format:
        raise ValueError(
            f"descriptor format {receipt['format']!r} differs from expected "
            f"format {expected_format!r}"
        )
    mapping_bytes = _canonical_json(mapping)
    mapping_digest = hashlib.sha256(mapping_bytes).hexdigest()
    receipt["mapping"] = {
        "path": str(output_path),
        "bytes": len(mapping_bytes),
        "sha256": mapping_digest,
    }
    _require_receipt_authorities(receipt)
    _write_new(output_path, mapping_bytes)
    try:
        loaded = load_mapping(output_path)
        if loaded != mapping or _sha256(output_path) != mapping_digest:
            raise RuntimeError("published mapping failed its exact round trip")
        _require_receipt_authorities(receipt)
        _write_new(receipt_path, _canonical_json(receipt))
    except BaseException:
        if output_path.is_file() and _sha256(output_path) == mapping_digest:
            output_path.unlink()
        raise
    return receipt


def _path_row(
    path: Path,
    manifest_parent: Path,
    fingerprint: tuple[str, int, int],
) -> dict[str, object]:
    try:
        relative = Path(os.path.relpath(path, manifest_parent)).as_posix()
        serialized = relative
    except ValueError:
        serialized = path.as_posix()
    return {
        "path": serialized,
        "bytes": fingerprint[1],
        "sha256": fingerprint[0],
    }


def _inventory_row(
    paths: Sequence[Path],
    manifest_parent: Path,
    fingerprints: Mapping[Path, tuple[str, int, int]],
) -> list[dict[str, object]]:
    return [
        _path_row(path, manifest_parent, fingerprints[path]) for path in paths
    ]


def _load_contract_snapshots(
    mapping_path: Path,
    composition_path: Path,
) -> tuple[
    _FileSnapshot,
    _FileSnapshot,
    dict[str, object],
    dict[str, object],
]:
    """Validate mapping/composition using exactly the bytes later sealed."""

    mapping_snapshot = _stable_file_snapshot(mapping_path)
    composition_snapshot = _stable_file_snapshot(composition_path)
    # Reject ambiguous duplicate keys before the executable validators see the
    # last-wins result produced by Python's default JSON decoder.
    _strict_json(mapping_snapshot.data, f"mapping {mapping_path}")
    _strict_json(composition_snapshot.data, f"composition {composition_path}")
    with tempfile.TemporaryDirectory(prefix="rw-wps-contract-snapshot-") as directory:
        root = Path(directory)
        frozen_mapping = root / "mapping.json"
        frozen_composition = root / "composition.json"
        frozen_mapping.write_bytes(mapping_snapshot.data)
        frozen_composition.write_bytes(composition_snapshot.data)
        mapping = load_mapping(frozen_mapping)
        composition = load_composition(frozen_composition, frozen_mapping)
    _require_snapshot(mapping_snapshot)
    _require_snapshot(composition_snapshot)
    return mapping_snapshot, composition_snapshot, mapping, composition


def _verified_decoder_snapshot(path: Path, role: str) -> _FileSnapshot:
    """Run decoder identity checks on an immutable copy of exact sealed bytes."""

    snapshot = _stable_file_snapshot(path)
    # bridge_identity itself executes a private copy of one stable snapshot.
    identity = bridge_identity(snapshot.path, role)
    if not isinstance(identity, dict):
        raise RuntimeError(f"decoder identity probe returned no receipt for {role}")
    if identity.get("bytes") != snapshot.size or identity.get("sha256") != snapshot.sha256:
        raise RuntimeError(
            f"decoder identity receipt differs from exact {role} bytes"
        )
    _require_snapshot(snapshot)
    return snapshot


def author_input_manifest(
    output_path: str | Path,
    *,
    mapping_path: str | Path,
    composition_path: str | Path,
    primary_files: Sequence[str | Path],
    supplement_files: Mapping[str, str | Path | Sequence[str | Path]],
    provenance_files: Mapping[str, str | Path],
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
    expected_format: str | None = None,
) -> dict[str, object]:
    """Create and round-trip an exact composition input manifest."""

    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    mapping_path = Path(mapping_path).resolve()
    composition_path = Path(composition_path).resolve()
    (
        mapping_snapshot,
        composition_snapshot,
        mapping,
        composition,
    ) = _load_contract_snapshots(mapping_path, composition_path)
    if expected_format is not None and mapping["format"] != expected_format:
        raise ValueError(
            f"mapping format {mapping['format']!r} differs from expected "
            f"format {expected_format!r}"
        )
    primary = tuple(Path(path).resolve() for path in primary_files)
    if not primary or len(set(primary)) != len(primary):
        raise ValueError("primary file inventory must be non-empty and unique")
    supplements: dict[str, tuple[Path, ...]] = {}
    for role, raw in supplement_files.items():
        values = (raw,) if isinstance(raw, (str, Path)) else tuple(raw)
        paths = tuple(Path(path).resolve() for path in values)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError(
                f"supplement role {role!r} must have a non-empty unique inventory"
            )
        supplements[str(role)] = paths
    provenance = {
        str(role): Path(path).resolve() for role, path in provenance_files.items()
    }
    decoders = _decoder_inventory(
        str(mapping["format"]),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )
    terrain = composition["supplements"]["terrain_height"]
    expected_supplements = {str(terrain["data_role"])}
    expected_provenance = {str(terrain["provenance_role"])}
    if set(supplements) != expected_supplements:
        raise ValueError(
            "supplement role inventory differs from the composition contract"
        )
    if set(provenance) != expected_provenance:
        raise ValueError(
            "provenance role inventory differs from the composition contract"
        )
    authorities = (
        mapping_path,
        composition_path,
        *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
        *decoders.values(),
    )
    for path in authorities:
        if not path.is_file():
            raise FileNotFoundError(path)
    _require_unique_identities(primary, "primary file inventory")
    for role, paths in supplements.items():
        _require_unique_identities(paths, f"supplement role {role!r}")
    non_data = (
        mapping_path,
        composition_path,
        *provenance.values(),
        *decoders.values(),
    )
    _require_unique_identities(
        non_data, "mapping/composition/provenance/decoder inventory"
    )
    data_identities = {
        _file_identity(path)
        for path in (
            *primary,
            *(path for paths in supplements.values() for path in paths),
        )
    }
    if any(_file_identity(path) in data_identities for path in non_data):
        raise ValueError(
            "mapping/composition/provenance/decoder authority aliases source data"
        )
    decoder_snapshots = {
        role: _verified_decoder_snapshot(path, role)
        for role, path in decoders.items()
    }
    authority_snapshots = {
        mapping_path: mapping_snapshot,
        composition_path: composition_snapshot,
        **{
            decoders[role]: snapshot
            for role, snapshot in decoder_snapshots.items()
        },
    }
    for path in set(authorities) - set(authority_snapshots):
        authority_snapshots[path] = _snapshot_authority(path)
    fingerprints = {
        path: (snapshot.sha256, snapshot.size, snapshot.mtime_ns)
        for path, snapshot in authority_snapshots.items()
    }
    parent = output_path.parent
    payload = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "mapping_sha256": fingerprints[mapping_path][0],
        "composition_sha256": fingerprints[composition_path][0],
        "primary_files": [
            _path_row(path, parent, fingerprints[path]) for path in primary
        ],
        "supplements": {
            role: _inventory_row(paths, parent, fingerprints)
            for role, paths in supplements.items()
        },
        "provenance": {
            role: _path_row(path, parent, fingerprints[path])
            for role, path in provenance.items()
        },
        "decoders": {
            role: _path_row(path, parent, fingerprints[path])
            for role, path in decoders.items()
        },
    }
    manifest_bytes = _canonical_json(payload)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = output_path.with_name(
        f".{output_path.name}.candidate-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    )
    published = False
    try:
        with candidate.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_manifest(
            candidate,
            manifest_digest,
            mapping_path=mapping_path,
            composition_path=composition_path,
            primary_files=primary,
            supplement_files=supplements,
            provenance_files=provenance,
            decoder_files=decoders,
            _snapshots=authority_snapshots,
            _recheck_snapshots=False,
        )
        for snapshot in authority_snapshots.values():
            if isinstance(snapshot, _FileSnapshot):
                _require_snapshot(snapshot)
            else:
                _require_authority_snapshot(snapshot)
        try:
            os.link(candidate, output_path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {output_path}") from error
        published = True
        if _sha256(output_path) != manifest_digest:
            raise RuntimeError("published manifest bytes differ from candidate")
    except BaseException:
        if (
            published
            and output_path.is_file()
            and _sha256(output_path) == manifest_digest
        ):
            output_path.unlink()
        raise
    finally:
        candidate.unlink(missing_ok=True)
    return {
        "schema": MANIFEST_AUTHORING_SCHEMA,
        "status": "PASS_IDENTITY_BOUND_NOT_STOCK_WRF_CERTIFIED",
        "manifest": {
            "path": str(output_path),
            "bytes": len(manifest_bytes),
            "sha256": manifest_digest,
        },
        "source_format": mapping["format"],
        "primary_file_count": len(primary),
        "supplement_file_count": sum(len(paths) for paths in supplements.values()),
        "provenance_file_count": len(provenance),
        "decoder_file_count": len(decoders),
    }


__all__ = [
    "DESCRIPTOR_SCHEMA",
    "MANIFEST_AUTHORING_SCHEMA",
    "MAPPING_AUTHORING_SCHEMA",
    "VtableRow",
    "author_input_manifest",
    "author_mapping",
    "compile_mapping_descriptor",
    "parse_wps_vtable",
]
