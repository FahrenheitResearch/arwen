"""Native NOAA 20CRv3 every-member GRIB2 adapter.

The supplied 20CRv3 archive does not encode ensemble membership in GRIB2 PDT
metadata.  Membership is therefore an explicit filename/manifest authority:
all files must use ``memNNN_YYYYMMDDHH_{pl,sfc}.grb2``, every valid time must
have exactly one pressure and one surface file, and mixed member IDs fail
before any field decode.

Decoded fields use the existing strict ``rw-wps.mapping.v1`` GRIB2 engine and
canonical source-frame materializer.  This module only owns archive discovery,
member identity, input hashing, and the pressure/surface pairing contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from gpuwm.ingest.source_coverage import ForcingSeriesRefusal
from gpuwm.mapped_source import (
    MappedSourceFrame,
    _DecodedCollection,
    _decode_grib,
    _grib2_inventory,
    _load_json_bytes,
    _materialize_frames,
    _require_authority_snapshot,
    _snapshot_authority,
    load_mapping,
    mapped_frames_to_regular_snapshots,
)


MANIFEST_SCHEMA = "gpuwm-20crv3-grib2-inputs-v1"
RECEIPT_SCHEMA = "gpuwm-20crv3-canonical-frames-v1"
_FILENAME = re.compile(
    r"^mem(?P<member>[0-9]{3})_(?P<valid>[0-9]{10})_"
    r"(?P<role>pl|sfc)\.grb2$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _parse_filename(path: Path) -> tuple[str, datetime, str]:
    match = _FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "20CRv3 files must match memNNN_YYYYMMDDHH_{pl,sfc}.grb2: "
            f"{path.name!r}"
        )
    valid_time = datetime.strptime(match.group("valid"), "%Y%m%d%H")
    return match.group("member"), valid_time, match.group("role")


def discover_20crv3_grib2(source_root: str | Path) -> dict[str, object]:
    """Hash and validate one paired every-member GRIB2 time series."""

    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    files = tuple(sorted(
        path.resolve() for path in source_root.rglob("*.grb2") if path.is_file()
    ))
    if not files:
        raise ValueError("20CRv3 source directory contains no GRIB2 files")
    members: set[str] = set()
    by_time: dict[datetime, dict[str, Path]] = {}
    for path in files:
        member, valid_time, role = _parse_filename(path)
        members.add(member)
        roles = by_time.setdefault(valid_time, {})
        if role in roles:
            raise ValueError(
                f"duplicate 20CRv3 {role} file at {valid_time}: "
                f"{roles[role]}, {path}"
            )
        roles[role] = path
    if len(members) != 1:
        raise ValueError(f"20CRv3 input mixes member IDs {sorted(members)}")
    incomplete = {
        value.isoformat(): sorted({"pl", "sfc"} - set(roles))
        for value, roles in by_time.items()
        if set(roles) != {"pl", "sfc"}
    }
    if incomplete:
        raise ValueError(f"20CRv3 pressure/surface pairing gaps: {incomplete}")
    times = tuple(sorted(by_time))
    if len(times) < 2:
        raise ForcingSeriesRefusal(
            "20CRv3 lateral boundaries require at least two times")
    deltas = tuple(
        int((later - earlier).total_seconds())
        for earlier, later in zip(times, times[1:])
    )
    if any(value <= 0 for value in deltas) or len(set(deltas)) != 1:
        raise ValueError(f"20CRv3 valid-time cadence is irregular: {deltas}")
    rows = []
    for valid_time in times:
        for role in ("pl", "sfc"):
            path = by_time[valid_time][role]
            rows.append({
                "path": str(path),
                "filename": path.name,
                "member": next(iter(members)),
                "valid_time": valid_time.isoformat(),
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    payload = {
        "schema": MANIFEST_SCHEMA,
        "source": "NOAA-CIRES-DOE 20CRv3 every-member GRIB2",
        "member_identity": "filename_memNNN_not_grib2_pdt",
        "member": next(iter(members)),
        "valid_times": [value.isoformat() for value in times],
        "cadence_seconds": deltas[0],
        "file_count": len(rows),
        "files": rows,
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def write_20crv3_manifest(
    source_root: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Write a new manifest atomically, refusing to replace an authority."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = discover_20crv3_grib2(source_root)
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return payload


def _manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    snapshot = _snapshot_authority(path, retain_bytes=True)
    if snapshot.sha256 != expected_sha256.lower():
        raise ValueError(
            f"20CRv3 manifest SHA mismatch: expected {expected_sha256}, "
            f"got {snapshot.sha256}"
        )
    raw = _load_json_bytes(snapshot.data, "20CRv3 manifest", path)
    if not isinstance(raw, dict) or raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported 20CRv3 manifest schema")
    allowed = {
        "schema", "source", "member_identity", "member", "valid_times",
        "cadence_seconds", "file_count", "files", "content_sha256",
    }
    unknown = sorted(set(raw) - allowed)
    missing = sorted(allowed - set(raw))
    if unknown or missing:
        raise ValueError(
            f"20CRv3 manifest shape differs: missing={missing}, unknown={unknown}"
        )
    content = dict(raw)
    declared_content_sha = content.pop("content_sha256")
    if declared_content_sha != _canonical_sha256(content):
        raise ValueError("20CRv3 manifest content hash is stale")
    if raw["member_identity"] != "filename_memNNN_not_grib2_pdt":
        raise ValueError("20CRv3 manifest member identity policy differs")
    member = raw["member"]
    if not isinstance(member, str) or not re.fullmatch(r"[0-9]{3}", member):
        raise ValueError("20CRv3 manifest member must be a three-digit label")
    rows = raw["files"]
    if not isinstance(rows, list) or raw["file_count"] != len(rows):
        raise ValueError("20CRv3 manifest file count differs")
    paths = []
    observed_times: dict[datetime, set[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "path", "filename", "member", "valid_time", "role", "bytes",
            "sha256",
        }:
            raise ValueError(f"20CRv3 manifest file row {index} is malformed")
        path_value = Path(row["path"]).resolve()
        if not path_value.is_file():
            raise FileNotFoundError(path_value)
        filename_member, filename_time, filename_role = _parse_filename(path_value)
        if (
            path_value.name != row["filename"]
            or filename_member != member
            or row["member"] != member
            or filename_time.isoformat() != row["valid_time"]
            or filename_role != row["role"]
        ):
            raise ValueError(
                f"20CRv3 manifest row {index} disagrees with its filename"
            )
        if (
            isinstance(row["bytes"], bool)
            or row["bytes"] != path_value.stat().st_size
            or row["sha256"] != _sha256(path_value)
        ):
            raise ValueError(
                f"20CRv3 manifest row {index} input identity differs"
            )
        observed_times.setdefault(filename_time, set()).add(filename_role)
        paths.append(path_value)
    if len(set(paths)) != len(paths):
        raise ValueError("20CRv3 manifest contains duplicate input paths")
    if any(roles != {"pl", "sfc"} for roles in observed_times.values()):
        raise ValueError("20CRv3 manifest has an incomplete pressure/surface pair")
    times = tuple(sorted(observed_times))
    if raw["valid_times"] != [value.isoformat() for value in times]:
        raise ValueError("20CRv3 manifest valid-time inventory differs")
    deltas = {
        int((later - earlier).total_seconds())
        for earlier, later in zip(times, times[1:])
    }
    if len(deltas) != 1 or raw["cadence_seconds"] != next(iter(deltas)):
        raise ValueError("20CRv3 manifest cadence differs from file inventory")
    _require_authority_snapshot(snapshot)
    return raw, tuple(paths)


def _bind_implicit_member(
    collection: _DecodedCollection,
    member: str,
) -> _DecodedCollection:
    """Replace absent PDT membership with the verified filename authority."""

    encoded = {value.member for value in collection.direct.values()}
    if encoded != {None}:
        raise ValueError(
            "20CRv3 sample unexpectedly encodes a GRIB2 member; a separate "
            f"identity policy is required: {sorted(str(value) for value in encoded)}"
        )
    direct = {}
    for (valid_time, _old_member, field_name), value in collection.direct.items():
        key = (valid_time, member, field_name)
        if key in direct:
            raise ValueError("20CRv3 implicit member binding creates a duplicate field")
        direct[key] = replace(value, member=member)
    cycles = {
        (valid_time, member): cycle
        for (valid_time, _old_member), cycle in collection.source_cycles.items()
    }
    return _DecodedCollection(
        latitude=collection.latitude,
        longitude=collection.longitude,
        vertical_values=collection.vertical_values,
        direct=MappingProxyType(direct),
        source_cycles=MappingProxyType(cycles),
        grid_fingerprint=collection.grid_fingerprint,
    )


def _record_key(row: Mapping[str, str]) -> tuple[object, ...]:
    return (
        int(row["discipline"]),
        int(row["category"]),
        int(row["parameter"]),
        int(row["level_type"]),
        float(row["level_value"]),
        int(row["second_level_type"]),
        float(row["second_level_value"]),
        row["bitmap"] == "true",
    )


def _expected_inventory(
    role: str,
    pressure_levels: Sequence[float],
) -> Counter[tuple[object, ...]]:
    if role == "pl":
        parameters = (
            (0, 3, 5),
            (0, 0, 0),
            (0, 1, 0),
            (0, 2, 2),
            (0, 2, 3),
        )
        return Counter(
            (discipline, category, parameter, 100, float(level), 255, 0.0, False)
            for discipline, category, parameter in parameters
            for level in pressure_levels
        )
    if role != "sfc":
        raise ValueError(f"unsupported 20CRv3 file role {role!r}")
    rows: list[tuple[object, ...]] = [
        (0, 3, 0, 1, 0.0, 255, 0.0, False),
        (0, 3, 5, 1, 0.0, 255, 0.0, False),
        (0, 3, 1, 101, 0.0, 255, 0.0, False),
        (0, 0, 0, 1, 0.0, 255, 0.0, False),
    ]
    for top, bottom in ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)):
        rows.append((2, 0, 192, 106, top, 106, bottom, True))
    for top, bottom in ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)):
        rows.append((0, 0, 0, 106, top, 106, bottom, True))
    rows.extend((
        (0, 1, 11, 1, 0.0, 255, 0.0, True),
        (2, 0, 0, 1, 0.0, 255, 0.0, False),
        (10, 2, 0, 1, 0.0, 255, 0.0, False),
        (0, 0, 0, 103, 2.0, 255, 0.0, False),
        (0, 1, 1, 103, 2.0, 255, 0.0, False),
        (0, 2, 2, 103, 10.0, 255, 0.0, False),
        (0, 2, 3, 103, 10.0, 255, 0.0, False),
    ))
    return Counter(rows)


def _verify_archive_inventory(
    document: Mapping[str, object],
    sources: Sequence[Path],
    inventory_rows: "Callable[[Path], Sequence[Mapping[str, str]]]",
    pressure_levels: Sequence[float],
) -> dict[str, object]:
    """Bind the exact supplied every-member GRIB2 product identity.

    ``inventory_rows`` is the measurement instrument: the engine's raw
    record-inventory surface on the bare default
    (:func:`gpuwm.mapped_engine_bridge.engine_record_inventory`), the
    subprocess ``grib2_inventory`` on the documented Python-engine
    workaround.  Both render the same row vocabulary in the same
    spellings, so this gate's contract and refusal wording are identical
    whichever instrument measured the bytes.
    """

    manifest_rows = {
        Path(row["path"]).resolve(): row for row in document["files"]
    }
    grid: tuple[str, ...] | None = None
    summaries = []
    fixed_grid_keys = (
        "gdt", "nx", "ny", "lat1", "lon1", "dx", "dy", "latin1",
        "latin2", "lov", "scan_mode", "shape_of_earth",
        "resolution_flags",
    )
    for source in sources:
        row_authority = manifest_rows.get(source)
        if row_authority is None:
            raise ValueError(f"20CRv3 inventory source is absent from manifest: {source}")
        role = str(row_authority["role"])
        valid_time = datetime.fromisoformat(str(row_authority["valid_time"]))
        rows = list(inventory_rows(source))
        indices = [int(row["index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"20CRv3 {source.name} field indices are not contiguous")
        observed = Counter(_record_key(row) for row in rows)
        expected = _expected_inventory(role, pressure_levels)
        if observed != expected:
            missing = list((expected - observed).elements())[:8]
            extra = list((observed - expected).elements())[:8]
            raise ValueError(
                f"20CRv3 {source.name} field inventory differs; "
                f"missing={missing}, extra={extra}"
            )
        for row in rows:
            authority = (
                int(row["center"]),
                int(row["subcenter"]),
                int(row["master_table_version"]),
                int(row["local_table_version"]),
            )
            process = (
                int(row["generating_process"]),
                int(row["forecast_generating_process_id"]),
            )
            if authority != (7, 2, 2, 1) or process != (4, 195):
                raise ValueError(
                    f"20CRv3 {source.name} authority/process differs: "
                    f"{authority}, {process}"
                )
            reference_time = datetime.fromisoformat(row["reference_time"])
            forecast_hour = int(row["forecast_time"])
            if (
                int(row["forecast_unit"]) != 1
                or forecast_hour not in {0, 3}
                or reference_time + timedelta(hours=forecast_hour)
                != valid_time
                or int(row["pdt"]) != 0
                or row["member"] != "-"
            ):
                raise ValueError(
                    f"20CRv3 {source.name} time/member metadata differs"
                )
            if (
                int(row["gdt"]) != 0
                or row["scan_mode"] != "0x40"
                or int(row["shape_of_earth"]) != 6
                or row["resolution_flags"] != "0x30"
                or float(row["dx"]) != 0.5
                or float(row["dy"]) != 0.5
                or int(row["drt"]) != 0
            ):
                raise ValueError(
                    f"20CRv3 {source.name} grid/packing contract differs"
                )
            row_grid = tuple(row[key] for key in fixed_grid_keys)
            if grid is None:
                grid = row_grid
            elif row_grid != grid:
                raise ValueError(
                    f"20CRv3 {source.name} grid differs from the member series"
                )
        summaries.append({
            "filename": source.name,
            "role": role,
            "valid_time": valid_time.isoformat(),
            "reference_time": rows[0]["reference_time"],
            "forecast_hour": int(rows[0]["forecast_time"]),
            "field_count": len(rows),
            "process": [4, 195],
        })
    assert grid is not None
    return {
        "status": "PASS_EXACT_20CRV3_EVERY_MEMBER_GRIB2_CONTRACT",
        "authority": {
            "center": 7,
            "subcenter": 2,
            "master_table_version": 2,
            "local_table_version": 1,
            "generating_process": 4,
            "forecast_generating_process_id": 195,
            "pdt": 0,
            "pdt_member": "absent_filename_manifest_authoritative",
        },
        "grid": dict(zip(fixed_grid_keys, grid)),
        "files": summaries,
    }


def decode_20crv3_grib2(
    *,
    mapping: str | Path,
    manifest: str | Path,
    manifest_sha256: str,
    grib2_inventory: str | Path,
    grib2_dump: str | Path,
) -> tuple[tuple[MappedSourceFrame, ...], dict[str, object]]:
    """Decode one hash-bound member series into complete canonical frames."""

    mapping = Path(mapping).resolve()
    manifest = Path(manifest).resolve()
    inventory = Path(grib2_inventory).resolve()
    dump = Path(grib2_dump).resolve()
    for path in (mapping, manifest, inventory, dump):
        if not path.is_file():
            raise FileNotFoundError(path)
    mapping_snapshot = _snapshot_authority(mapping, retain_bytes=True)
    manifest_snapshot = _snapshot_authority(manifest, retain_bytes=True)
    inventory_snapshot = _snapshot_authority(inventory)
    dump_snapshot = _snapshot_authority(dump)
    document, sources = _manifest(manifest, manifest_sha256)
    source_snapshots = {path: _snapshot_authority(path) for path in sources}
    mapping_document = load_mapping(mapping)
    if mapping_document["format"] != "grib2":
        raise ValueError("20CRv3 named adapter requires a GRIB2 mapping")
    if mapping_document["target"]["boundary_interval_seconds"] \
            != document["cadence_seconds"]:
        raise ValueError(
            "20CRv3 manifest cadence differs from the mapping target contract"
        )
    archive_inventory = _verify_archive_inventory(
        document,
        sources,
        lambda source: _grib2_inventory(source, inventory),
        mapping_document["coordinates"]["vertical"]["levels"],
    )
    decoded = _decode_grib(
        mapping_document,
        sources,
        grib1_bridge=None,
        grib2_inventory=inventory,
        grib2_dump=dump,
    )
    member = str(document["member"])
    bound = _bind_implicit_member(decoded, member)
    frames = _materialize_frames(
        mapping_document,
        bound,
        mapping_sha256=mapping_snapshot.sha256,
        input_sha256={
            str(path): snapshot.sha256
            for path, snapshot in source_snapshots.items()
        },
    )
    expected_times = tuple(
        datetime.fromisoformat(value) for value in document["valid_times"]
    )
    if tuple(frame.valid_time for frame in frames) != expected_times:
        raise ValueError("20CRv3 decoded canonical times differ from the manifest")
    if any(frame.member != member for frame in frames):
        raise ValueError("20CRv3 canonical frame lost its member identity")
    regular = mapped_frames_to_regular_snapshots(frames)
    if len(regular) != len(frames):
        raise ValueError("20CRv3 regular-source packing lost a frame")
    for snapshot in (
        mapping_snapshot,
        manifest_snapshot,
        inventory_snapshot,
        dump_snapshot,
        *source_snapshots.values(),
    ):
        _require_authority_snapshot(snapshot)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_CANONICAL_13_TIME_MEMBER_NOT_STOCK_WRF_CERTIFIED",
        "stock_wrf_certified": False,
        "member": member,
        "member_identity": document["member_identity"],
        "archive_inventory": archive_inventory,
        "frame_count": len(frames),
        "valid_times": [frame.valid_time.isoformat() for frame in frames],
        "cadence_seconds": document["cadence_seconds"],
        "grid": {
            "ny": int(frames[0].latitude.size),
            "nx": int(frames[0].longitude.size),
            "vertical_count": int(frames[0].vertical_values.size),
            "vertical_values_pa": [
                float(value) for value in frames[0].vertical_values
            ],
            "fingerprint": frames[0].grid_fingerprint,
        },
        "field_inventory": sorted(frames[0].fields),
        "mapping": {
            "path": str(mapping),
            "sha256": mapping_snapshot.sha256,
        },
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_snapshot.sha256,
            "content_sha256": document["content_sha256"],
        },
        "decoders": {
            "grib2_inventory": {
                "path": str(inventory),
                "sha256": inventory_snapshot.sha256,
            },
            "grib2_dump": {
                "path": str(dump),
                "sha256": dump_snapshot.sha256,
            },
        },
        "input_count": len(sources),
        "input_sha256": {
            str(path): source_snapshots[path].sha256 for path in sources
        },
        "frame_header_sha256": [
            _canonical_sha256(frame.header.to_dict()) for frame in frames
        ],
        "remaining_gates": [
            "native WRFINPUT/WRFBDY export",
            "unchanged stock-WRF startup",
            "private hourly/80-member archive sample compatibility",
        ],
    }
    receipt["receipt_content_sha256"] = _canonical_sha256(receipt)
    return frames, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpuwm.twentycrv3_direct")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("source_root", type=Path)
    manifest.add_argument("output", type=Path)
    decode = commands.add_parser("decode")
    decode.add_argument("--mapping", type=Path, required=True)
    decode.add_argument("--manifest", type=Path, required=True)
    decode.add_argument("--manifest-sha256", required=True)
    decode.add_argument("--grib2-inventory", type=Path, required=True)
    decode.add_argument("--grib2-dump", type=Path, required=True)
    decode.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        payload = write_20crv3_manifest(args.source_root, args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    _frames, receipt = decode_20crv3_grib2(
        mapping=args.mapping,
        manifest=args.manifest,
        manifest_sha256=args.manifest_sha256,
        grib2_inventory=args.grib2_inventory,
        grib2_dump=args.grib2_dump,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MANIFEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "decode_20crv3_grib2",
    "discover_20crv3_grib2",
    "write_20crv3_manifest",
]
