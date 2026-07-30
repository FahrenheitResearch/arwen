"""Hash-bound composition of mapped products on compatible donor grids.

``rw-wps.mapping.v1`` describes one source format and the canonical field
semantics, but a meteorological archive may split those fields across product
files with different geographic extents.  This module implements the first
strict composition slice: an invariant GRIB terrain product whose regular
latitude/longitude grid contains the primary forcing grid as an exact
coordinate subset.

The join is intentionally narrow.  It does not fabricate terrain, treat WRF
target ``HGT_M`` as source terrain, or silently interpolate between donor
grids.  Mapping, composition, data, provenance, and native decoder bytes are
verified before and after decoding.  The terrain selector and unit conversion
remain owned by the sealed mapping document.  The same contract is used for
GRIB1, GRIB2, and NetCDF. Soil source geometry, target geometry, remapping,
and ocean repair are declarative and contain no source-family dispatch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from gpuwm.mapped_source import (
    MappedSourceFrame,
    _DecodedCollection,
    _DirectValue,
    _AuthoritySnapshot,
    _array_sha256,
    _decode_netcdf,
    _decode_grib,
    _load_json_document,
    _load_json_bytes,
    _materialize_frames,
    _require_authority_snapshot,
    _snapshot_authority,
    load_mapping,
    mapped_frames_to_regular_snapshots,
)
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
    validate_soil_layer_contract,
)


COMPOSITION_SCHEMA = "gpuwm-mapped-composition-v2"
INPUT_MANIFEST_SCHEMA = "gpuwm-mapped-composition-inputs-v1"
RECEIPT_SCHEMA = "gpuwm-mapped-composition-receipt-v1"
_EXTERNAL_FIELD = "terrain_height"
_SOURCE_FORMATS = {"grib1", "grib2", "netcdf"}


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


def _digest(value: object, label: str) -> str:
    text = _string(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def load_composition(
    path: str | Path,
    mapping_path: str | Path,
    *,
    _raw: object | None = None,
    _mapping_raw: object | None = None,
) -> dict[str, object]:
    """Validate one composition contract against exact mapping bytes."""

    path = Path(path).resolve()
    mapping_path = Path(mapping_path).resolve()
    raw = _load_json_document(path, "composition") if _raw is None else _raw
    contract = _object(
        raw, "composition",
        allowed={
            "schema", "name", "mapping_binding", "soil_layers", "supplements",
        },
        required={
            "schema", "name", "mapping_binding", "soil_layers", "supplements",
        },
    )
    if contract["schema"] != COMPOSITION_SCHEMA:
        raise ValueError(f"unsupported composition schema {contract['schema']!r}")
    _string(contract["name"], "composition.name")
    if contract["mapping_binding"] != "input_manifest_sha256":
        raise ValueError("composition mapping_binding must be input_manifest_sha256")
    mapping = load_mapping(mapping_path, _raw=_mapping_raw)
    validate_soil_layer_contract(contract["soil_layers"], mapping=mapping)
    supplements = _object(
        contract["supplements"], "composition.supplements",
        allowed={_EXTERNAL_FIELD}, required={_EXTERNAL_FIELD},
    )
    spec = _object(
        supplements[_EXTERNAL_FIELD],
        f"composition.supplements.{_EXTERNAL_FIELD}",
        allowed={
            "data_role", "provenance_role", "format", "field",
            "selector_authority", "grid_alignment", "time_alignment",
            "require_invariant_across_time",
        },
        required={
            "data_role", "provenance_role", "format", "field",
            "selector_authority", "grid_alignment", "time_alignment",
            "require_invariant_across_time",
        },
    )
    for key in (
        "data_role", "provenance_role", "format", "field",
        "selector_authority", "grid_alignment", "time_alignment",
    ):
        _string(spec[key], f"terrain supplement.{key}")
    if spec["field"] != _EXTERNAL_FIELD:
        raise ValueError("terrain supplement must provide terrain_height")
    if spec["format"] != mapping["format"] \
            or spec["format"] not in _SOURCE_FORMATS:
        raise ValueError("terrain supplement format must equal the mapping format")
    if spec["selector_authority"] != "mapping_field_exact":
        raise ValueError("terrain selector authority must be mapping_field_exact")
    if spec["grid_alignment"] != "exact_coordinate_subset":
        raise ValueError("terrain grid alignment must be exact_coordinate_subset")
    if spec["time_alignment"] != "valid_time_exact":
        raise ValueError("terrain time alignment must be valid_time_exact")
    if spec["require_invariant_across_time"] is not True:
        raise ValueError("terrain supplement must be invariant across all supplied times")
    field = mapping["fields"].get(_EXTERNAL_FIELD)
    if field is None or field.get("derivation") is not None \
            or not field.get("selectors"):
        raise ValueError("mapping terrain_height must have a direct selector")
    materialized_source_axes = tuple(
        axis for axis in field["source_axes"] if axis not in {"time", "member"}
    )
    if materialized_source_axes != ("y", "x") \
            or tuple(field["target_axes"]) != ("y", "x") \
            or field["location"] != "surface" \
            or field.get("staggering", "none") != "none" \
            or field["units"]["target"] != "m" \
            or field["missing"]["kind"] != "reject":
        raise ValueError(
            "mapping terrain_height must be finite unstaggered surface metres"
        )
    return contract


def _manifest_file(
    raw: object,
    label: str,
    manifest_path: Path,
    actual_path: Path,
    snapshot: _AuthoritySnapshot | None = None,
) -> None:
    actual_path = actual_path.resolve()
    snapshot = snapshot or _snapshot_authority(actual_path)
    if snapshot.path != actual_path:
        raise ValueError(f"{label} snapshot path differs from requested file")
    row = _object(
        raw, label,
        allowed={"path", "bytes", "sha256"},
        required={"path", "bytes", "sha256"},
    )
    declared = Path(_string(row["path"], f"{label}.path"))
    if not declared.is_absolute():
        declared = manifest_path.parent / declared
    if declared.resolve() != actual_path.resolve():
        raise ValueError(f"{label} path differs from the requested file")
    if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) \
            or row["bytes"] < 0:
        raise ValueError(f"{label}.bytes must be a nonnegative integer")
    if row["bytes"] != snapshot.size:
        raise ValueError(f"{label} byte count differs from the requested file")
    if _digest(row["sha256"], f"{label}.sha256") != snapshot.sha256:
        raise ValueError(f"{label} SHA differs from the requested file")


def _manifest_file_inventory(
    raw: object,
    label: str,
    manifest_path: Path,
    actual_paths: Path | Sequence[Path],
    snapshots: Mapping[Path, _AuthoritySnapshot] | None = None,
) -> None:
    paths = (actual_paths,) if isinstance(actual_paths, Path) \
        else tuple(actual_paths)
    if not paths:
        raise ValueError(f"{label} requested inventory is empty")
    if len(paths) == 1 and isinstance(raw, dict):
        _manifest_file(
            raw,
            label,
            manifest_path,
            paths[0],
            None if snapshots is None else snapshots[paths[0].resolve()],
        )
        return
    if not isinstance(raw, list) or len(raw) != len(paths):
        raise ValueError(f"{label} file inventory differs from the request")
    for index, (row, path) in enumerate(zip(raw, paths)):
        _manifest_file(
            row,
            f"{label}[{index}]",
            manifest_path,
            path,
            None if snapshots is None else snapshots[path.resolve()],
        )


def _verify_manifest(
    path: Path,
    expected_sha256: str,
    *,
    mapping_path: Path,
    composition_path: Path,
    primary_files: Sequence[Path],
    supplement_files: Mapping[str, Path | Sequence[Path]],
    provenance_files: Mapping[str, Path],
    decoder_files: Mapping[str, Path],
    _snapshots: Mapping[Path, _AuthoritySnapshot] | None = None,
    _recheck_snapshots: bool = True,
) -> dict[str, object]:
    path = path.resolve()
    supplied = {
        key.resolve(): value for key, value in (_snapshots or {}).items()
    }
    used: dict[Path, _AuthoritySnapshot] = {}

    def snapshot(actual: Path, *, retain_bytes: bool = False) -> _AuthoritySnapshot:
        actual = actual.resolve()
        value = supplied.get(actual)
        if value is None:
            value = _snapshot_authority(actual, retain_bytes=retain_bytes)
        if value.path != actual or (retain_bytes and value.data is None):
            raise ValueError(f"authority snapshot is incomplete for {actual}")
        used[actual] = value
        return value

    manifest_snapshot = snapshot(path, retain_bytes=True)
    observed = manifest_snapshot.sha256
    if observed != _digest(expected_sha256, "input manifest SHA"):
        raise ValueError("composition input manifest SHA mismatch")
    manifest = _object(
        _load_json_bytes(
            manifest_snapshot.data,
            "composition input manifest",
            path,
        ),
        "composition input manifest",
        allowed={
            "schema", "mapping_sha256", "composition_sha256",
            "primary_files", "supplements", "provenance", "decoders",
        },
        required={
            "schema", "mapping_sha256", "composition_sha256",
            "primary_files", "supplements", "provenance", "decoders",
        },
    )
    if manifest["schema"] != INPUT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported composition input schema {manifest['schema']!r}")
    mapping_snapshot = snapshot(mapping_path)
    composition_snapshot = snapshot(composition_path)
    if _digest(manifest["mapping_sha256"], "manifest.mapping_sha256") \
            != mapping_snapshot.sha256:
        raise ValueError("manifest mapping SHA differs from mapping bytes")
    if _digest(manifest["composition_sha256"], "manifest.composition_sha256") \
            != composition_snapshot.sha256:
        raise ValueError("manifest composition SHA differs from composition bytes")
    primary = manifest["primary_files"]
    if not isinstance(primary, list) or len(primary) != len(primary_files):
        raise ValueError("manifest primary file inventory differs from the request")
    for index, (row, source) in enumerate(zip(primary, primary_files)):
        _manifest_file(
            row,
            f"manifest.primary_files[{index}]",
            path,
            source,
            snapshot(source),
        )
    for section, actual in (
        ("supplements", supplement_files), ("provenance", provenance_files),
        ("decoders", decoder_files),
    ):
        rows = _object(
            manifest[section], f"manifest.{section}",
            allowed=set(actual), required=set(actual),
        )
        for role, file_path in actual.items():
            if section == "supplements":
                _manifest_file_inventory(
                    rows[role],
                    f"manifest.{section}.{role}",
                    path,
                    file_path,
                    snapshots={
                        item.resolve(): snapshot(item)
                        for item in (
                            (file_path,)
                            if isinstance(file_path, Path)
                            else tuple(file_path)
                        )
                    },
                )
            else:
                _manifest_file(
                    rows[role],
                    f"manifest.{section}.{role}",
                    path,
                    file_path,
                    snapshot(file_path),
                )
    if _recheck_snapshots:
        for value in used.values():
            _require_authority_snapshot(value)
    return manifest


def _exact_subset_indices(
    larger: np.ndarray,
    smaller: np.ndarray,
    label: str,
    *,
    cyclic_degrees: bool = False,
) -> np.ndarray:
    larger = np.asarray(larger, dtype=np.float64)
    smaller = np.asarray(smaller, dtype=np.float64)
    if larger.ndim != 1 or smaller.ndim != 1:
        raise ValueError(f"{label} coordinates must be 1-D")
    larger_comparison = np.mod(larger, 360.0) if cyclic_degrees else larger
    smaller_comparison = np.mod(smaller, 360.0) if cyclic_degrees else smaller
    indices = []
    for value in smaller_comparison:
        matches = np.flatnonzero(larger_comparison == value)
        if matches.size != 1:
            raise ValueError(
                f"primary {label} coordinate {value!r} has {matches.size} exact "
                "matches in the terrain grid"
            )
        indices.append(int(matches[0]))
    result = np.asarray(indices, dtype=np.int64)
    if result.size > 1:
        differences = np.diff(result)
        if not (np.all(differences == 1) or np.all(differences == -1)):
            raise ValueError(
                f"primary {label} is not a contiguous terrain-grid subset"
            )
    return result


def _compose_terrain(
    primary: _DecodedCollection,
    terrain: _DecodedCollection,
) -> tuple[_DecodedCollection, dict[str, object]]:
    primary_terrain = [key for key in primary.direct if key[2] == _EXTERNAL_FIELD]
    if primary_terrain:
        raise ValueError(
            "terrain has two providers: the primary source and the declared supplement"
        )
    terrain_inventory = {key[2] for key in terrain.direct}
    if terrain_inventory != {_EXTERNAL_FIELD}:
        raise ValueError(
            f"terrain supplement decoded unexpected fields {sorted(terrain_inventory)}"
        )
    latitude_indices = _exact_subset_indices(
        terrain.latitude, primary.latitude, "latitude"
    )
    longitude_indices = _exact_subset_indices(
        terrain.longitude, primary.longitude, "longitude", cyclic_degrees=True
    )
    terrain_items = sorted(
        terrain.direct.items(), key=lambda item: (item[0][0], str(item[0][1]))
    )
    if not terrain_items:
        raise ValueError("terrain supplement decoded no terrain messages")
    full_reference = terrain_items[0][1].values
    if any(not np.array_equal(full_reference, value.values)
           for _key, value in terrain_items[1:]):
        raise ValueError("terrain supplement changes across supplied valid times")
    primary_keys = tuple(sorted(
        primary.source_cycles, key=lambda item: (item[0], str(item[1]))
    ))
    terrain_by_time = {
        (time, member): value
        for (time, member, _field), value in terrain_items
    }
    missing_times = [key for key in primary_keys if key not in terrain_by_time]
    if missing_times:
        raise ValueError(
            f"terrain supplement lacks exact primary valid time(s) {missing_times}"
        )
    direct = dict(primary.direct)
    subset_reference = None
    for valid_time, member in primary_keys:
        supplied = terrain_by_time[(valid_time, member)]
        values = supplied.values[np.ix_(latitude_indices, longitude_indices)]
        if subset_reference is None:
            subset_reference = values
        elif not np.array_equal(subset_reference, values):
            raise ValueError("terrain subset changes across primary valid times")
        direct[(valid_time, member, _EXTERNAL_FIELD)] = _DirectValue(
            name=_EXTERNAL_FIELD,
            valid_time=valid_time,
            member=member,
            source_cycle=supplied.source_cycle,
            axes=supplied.axes,
            values=values,
            missing_count=supplied.missing_count,
            references=supplied.references,
        )
    assert subset_reference is not None
    receipt = {
        "schema": "gpuwm-mapped-exact-subset-binding-v1",
        "status": "PASS",
        "field": _EXTERNAL_FIELD,
        "primary_shape": [int(primary.latitude.size), int(primary.longitude.size)],
        "supplement_shape": [int(terrain.latitude.size), int(terrain.longitude.size)],
        "latitude_index_range": [int(latitude_indices[0]), int(latitude_indices[-1])],
        "longitude_index_range": [int(longitude_indices[0]), int(longitude_indices[-1])],
        "latitude_sha256": _array_sha256(primary.latitude),
        "longitude_sha256": _array_sha256(primary.longitude),
        "terrain_full_sha256": _array_sha256(full_reference),
        "terrain_subset_sha256": _array_sha256(subset_reference),
        "supplement_valid_times": [key[0].isoformat() for key, _value in terrain_items],
        "matched_primary_valid_times": [key[0].isoformat() for key in primary_keys],
        "invariant_across_all_supplement_times": True,
        "latitude_index_direction": int(np.sign(latitude_indices[-1] - latitude_indices[0])),
        "longitude_index_direction": int(np.sign(longitude_indices[-1] - longitude_indices[0])),
        "coordinate_match": "exact_equivalent_contiguous_subset",
        "longitude_equivalence": "modulo_360_exact",
    }
    return _DecodedCollection(
        latitude=primary.latitude,
        longitude=primary.longitude,
        vertical_values=primary.vertical_values,
        direct=MappingProxyType(direct),
        source_cycles=primary.source_cycles,
        grid_fingerprint=primary.grid_fingerprint,
    ), receipt


@dataclass(frozen=True)
class MappedSourceBundle:
    frames: tuple[MappedSourceFrame, ...]
    mapping_path: Path
    mapping_sha256: str
    composition_path: Path
    composition_sha256: str
    input_manifest_path: Path
    input_manifest_sha256: str
    decoder_paths: Mapping[str, Path]
    decoder_sha256: Mapping[str, str]
    terrain_data_paths: tuple[Path, ...]
    terrain_data_sha256: tuple[str, ...]
    terrain_provenance_path: Path
    terrain_provenance_sha256: str
    soil_layer_contract: Mapping[str, object]
    alignment_receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        if not frames or any(_EXTERNAL_FIELD not in frame.fields for frame in frames):
            raise ValueError("composed source bundle lacks terrain in a canonical frame")
        terrain_hashes = {
            _array_sha256(frame.fields[_EXTERNAL_FIELD].values) for frame in frames
        }
        if len(terrain_hashes) != 1:
            raise ValueError("composed canonical terrain changes across forcing times")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(
            self, "alignment_receipt", MappingProxyType(dict(self.alignment_receipt))
        )
        object.__setattr__(
            self, "soil_layer_contract", MappingProxyType(dict(self.soil_layer_contract))
        )
        decoder_paths = MappingProxyType({
            str(role): Path(path).resolve()
            for role, path in self.decoder_paths.items()
        })
        decoder_sha256 = MappingProxyType({
            str(role): str(digest) for role, digest in self.decoder_sha256.items()
        })
        if set(decoder_paths) != set(decoder_sha256):
            raise ValueError("decoder path and SHA inventories differ")
        object.__setattr__(self, "decoder_paths", decoder_paths)
        object.__setattr__(self, "decoder_sha256", decoder_sha256)
        terrain_paths = tuple(Path(path).resolve() for path in self.terrain_data_paths)
        terrain_sha = tuple(str(digest) for digest in self.terrain_data_sha256)
        if not terrain_paths or len(terrain_paths) != len(terrain_sha):
            raise ValueError("terrain data path and SHA inventories differ")
        object.__setattr__(self, "terrain_data_paths", terrain_paths)
        object.__setattr__(self, "terrain_data_sha256", terrain_sha)

    def regular_snapshots(self):
        """Pack the WRF-real ABI with explicit soil and zero policies."""

        packed = mapped_frames_to_regular_snapshots(self.frames)
        result = []
        zero_fields = {
            "cloud_water_mixing_ratio": "QC",
            "rain_water_mixing_ratio": "QR",
            "cloud_ice_mixing_ratio": "QI",
            "snow_mixing_ratio": "QS",
            "graupel_or_hail_mixing_ratio": "QG",
        }
        for frame, snapshot in zip(self.frames, packed):
            fields = dict(snapshot.fields)
            if "SOURCE_OROGRAPHY" not in fields:
                raise ValueError("canonical terrain did not reach the regular-source ABI")
            if MAPPED_SOIL_TEMPERATURE not in fields \
                    or MAPPED_SOIL_MOISTURE not in fields:
                raise ValueError("canonical mapped soil arrays are absent")
            pressure = np.asarray(fields["PRES"])
            policies = frame.header.initialization_policies
            for canonical, output in zero_fields.items():
                if policies.get(canonical) != "explicit_zero_with_adapter_validation":
                    raise ValueError(
                        f"absent {canonical} lacks the supported explicit-zero policy"
                    )
                fields[output] = np.zeros_like(pressure)
            result.append(Era5Snapshot(
                valid_time=snapshot.valid_time,
                levels_hpa=snapshot.levels_hpa,
                latitude=snapshot.latitude,
                longitude=snapshot.longitude,
                fields=fields,
            ))
        return tuple(result)


def _decoder_inventory(
    source_format: str,
    *,
    grib1_bridge: str | Path | None,
    grib2_inventory: str | Path | None,
    grib2_dump: str | Path | None,
) -> dict[str, Path]:
    provided = {
        role: Path(path).resolve()
        for role, path in {
            "grib1_bridge": grib1_bridge,
            "grib2_inventory": grib2_inventory,
            "grib2_dump": grib2_dump,
        }.items()
        if path is not None
    }
    expected = {
        "grib1": {"grib1_bridge"},
        "grib2": {"grib2_inventory", "grib2_dump"},
        "netcdf": set(),
    }[source_format]
    if set(provided) != expected:
        raise ValueError(
            f"{source_format} decoder inventory differs from the contract; "
            f"missing={sorted(expected - set(provided))}, "
            f"extra={sorted(set(provided) - expected)}"
        )
    return provided


def _partition_mapping(
    mapping: Mapping[str, object], *, terrain_only: bool,
) -> dict[str, object]:
    """Return a decoder-only mapping for one side of the composition.

    The sealed, fully validated mapping remains the sole semantic authority.
    Partitioning only prevents a decoder from demanding a field that the
    composition contract explicitly assigns to the other product.
    """

    partition = copy.deepcopy(dict(mapping))
    fields = partition["fields"]
    assert isinstance(fields, dict)
    partition["fields"] = {
        name: field for name, field in fields.items()
        if (name == _EXTERNAL_FIELD) == terrain_only
    }
    if terrain_only and set(partition["fields"]) != {_EXTERNAL_FIELD}:
        raise ValueError("mapping lacks the direct terrain field")
    if not terrain_only and not partition["fields"]:
        raise ValueError("mapping has no primary fields after terrain partitioning")
    return partition


def _decode_partition(
    mapping: Mapping[str, object],
    files: Sequence[Path],
    decoders: Mapping[str, Path],
) -> _DecodedCollection:
    if mapping["format"] == "netcdf":
        return _decode_netcdf(mapping, files)
    return _decode_grib(
        mapping,
        files,
        grib1_bridge=decoders.get("grib1_bridge"),
        grib2_inventory=decoders.get("grib2_inventory"),
        grib2_dump=decoders.get("grib2_dump"),
    )


def _path_inventory(
    value: str | Path | Sequence[str | Path], label: str,
) -> tuple[Path, ...]:
    raw = (value,) if isinstance(value, (str, Path)) else tuple(value)
    paths = tuple(Path(path).resolve() for path in raw)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"{label} file inventory must be non-empty and unique")
    return paths


def decode_composed_source(
    composition_path: str | Path,
    mapping_path: str | Path,
    primary_files: Sequence[str | Path],
    supplement_files: Mapping[
        str, str | Path | Sequence[str | Path]
    ],
    provenance_files: Mapping[str, str | Path],
    *,
    input_manifest: str | Path,
    input_manifest_sha256: str,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> MappedSourceBundle:
    """Decode a complete mapped source with scientifically sourced terrain."""

    composition_path = Path(composition_path).resolve()
    mapping_path = Path(mapping_path).resolve()
    manifest_path = Path(input_manifest).resolve()
    primary = tuple(Path(path).resolve() for path in primary_files)
    supplements = {
        role: _path_inventory(path, f"supplement {role!r}")
        for role, path in supplement_files.items()
    }
    provenance = {role: Path(path).resolve() for role, path in provenance_files.items()}
    if not primary or len(set(primary)) != len(primary):
        raise ValueError("primary source inventory must be non-empty and unique")
    for path in (
        composition_path, mapping_path, manifest_path,
        *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    composition_snapshot = _snapshot_authority(
        composition_path,
        retain_bytes=True,
    )
    manifest_snapshot = _snapshot_authority(manifest_path, retain_bytes=True)
    mapping_raw = _load_json_bytes(
        mapping_snapshot.data,
        "mapping",
        mapping_path,
    )
    composition_raw = _load_json_bytes(
        composition_snapshot.data,
        "composition",
        composition_path,
    )
    mapping = load_mapping(mapping_path, _raw=mapping_raw)
    contract = load_composition(
        composition_path,
        mapping_path,
        _raw=composition_raw,
        _mapping_raw=mapping_raw,
    )
    decoders = _decoder_inventory(
        str(mapping["format"]),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )
    for path in decoders.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    terrain_spec = contract["supplements"][_EXTERNAL_FIELD]
    data_role = str(terrain_spec["data_role"])
    provenance_role = str(terrain_spec["provenance_role"])
    if set(supplements) != {data_role} or set(provenance) != {provenance_role}:
        raise ValueError("composition role inventory differs from the contract")
    authorities = (
        composition_path, mapping_path, manifest_path,
        *decoders.values(), *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
    )
    snapshots = {
        composition_path: composition_snapshot,
        mapping_path: mapping_snapshot,
        manifest_path: manifest_snapshot,
    }
    for path in authorities:
        if path not in snapshots:
            snapshots[path] = _snapshot_authority(path)
    _verify_manifest(
        manifest_path, input_manifest_sha256,
        mapping_path=mapping_path, composition_path=composition_path,
        primary_files=primary, supplement_files=supplements,
        provenance_files=provenance, decoder_files=decoders,
        _snapshots=snapshots,
        _recheck_snapshots=False,
    )
    for snapshot in (
        mapping_snapshot,
        composition_snapshot,
        manifest_snapshot,
    ):
        _require_authority_snapshot(snapshot)
    before = {str(path): snapshots[path].sha256 for path in authorities}
    primary_collection = _decode_partition(
        _partition_mapping(mapping, terrain_only=False), primary, decoders,
    )
    terrain_collection = _decode_partition(
        _partition_mapping(mapping, terrain_only=True),
        supplements[data_role], decoders,
    )
    combined, alignment = _compose_terrain(
        primary_collection, terrain_collection
    )
    input_hashes = {
        str(path): before[str(path)]
        for path in (
            *primary, *(path for paths in supplements.values() for path in paths)
        )
    }
    frames = _materialize_frames(
        mapping, combined, mapping_sha256=before[str(mapping_path)],
        input_sha256=input_hashes,
    )
    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    return MappedSourceBundle(
        frames=frames,
        mapping_path=mapping_path,
        mapping_sha256=before[str(mapping_path)],
        composition_path=composition_path,
        composition_sha256=before[str(composition_path)],
        input_manifest_path=manifest_path,
        input_manifest_sha256=before[str(manifest_path)],
        decoder_paths=decoders,
        decoder_sha256={
            role: before[str(path)] for role, path in decoders.items()
        },
        terrain_data_paths=supplements[data_role],
        terrain_data_sha256=tuple(
            before[str(path)] for path in supplements[data_role]
        ),
        terrain_provenance_path=provenance[provenance_role],
        terrain_provenance_sha256=before[str(provenance[provenance_role])],
        soil_layer_contract=contract["soil_layers"],
        alignment_receipt=alignment,
    )


def mapped_composition_receipt(bundle: MappedSourceBundle) -> dict[str, object]:
    """Return immutable evidence for the complete canonical materialization."""

    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "CANONICAL_FRAMES_COMPLETE_NOT_STOCK_WRF_CERTIFIED",
        "mapping": {
            "path": str(bundle.mapping_path), "sha256": bundle.mapping_sha256,
        },
        "composition": {
            "path": str(bundle.composition_path),
            "sha256": bundle.composition_sha256,
        },
        "input_manifest": {
            "path": str(bundle.input_manifest_path),
            "sha256": bundle.input_manifest_sha256,
        },
        "decoders": {
            role: {
                "path": str(path),
                "sha256": bundle.decoder_sha256[role],
            }
            for role, path in bundle.decoder_paths.items()
        },
        "terrain_products": [{
            "path": str(path),
            "sha256": digest,
        } for path, digest in zip(
            bundle.terrain_data_paths, bundle.terrain_data_sha256,
        )],
        "terrain_provenance": {
            "provenance_path": str(bundle.terrain_provenance_path),
            "provenance_sha256": bundle.terrain_provenance_sha256,
        },
        "alignment": dict(bundle.alignment_receipt),
        "soil_layers": dict(bundle.soil_layer_contract),
        "frame_count": len(bundle.frames),
        "valid_times": [frame.valid_time.isoformat() for frame in bundle.frames],
        "frames": [{
            "header_sha256": _canonical_sha256(frame.header.to_dict()),
            "terrain_sha256": _array_sha256(frame.fields[_EXTERNAL_FIELD].values),
            "field_count": len(frame.fields),
        } for frame in bundle.frames],
    }
    payload["receipt_content_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "COMPOSITION_SCHEMA", "INPUT_MANIFEST_SCHEMA", "RECEIPT_SCHEMA",
    "MappedSourceBundle", "decode_composed_source", "load_composition",
    "mapped_composition_receipt",
]
