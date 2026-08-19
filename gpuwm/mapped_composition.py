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

Cross-source composition extends the same contract across PACKAGED SOURCES:
a composition may declare that a canonical field's values come from a
DIFFERENT source's decode (soil from a physical analysis under an
AI-forecast atmosphere, and in general any declared field borrowed across
products).  The grammar is per-field source bindings in
``field_sources``: each binding names the contributing source id, pins that
source's OWN sealed mapping document by SHA-256, and declares a
grid-alignment and a cycle/time-alignment rule.  The primary mapping
declares each borrowed field as ``provider: "composition_bound"`` so the
gap is a declaration, never an absence.  Same-grid contributions (an exact
coordinate subset) land; a cross-grid contribution refuses by NAMING the
horizontal regrid capability this composition does not have; a
member-bearing donor refuses by naming member alignment; a vertical-bearing
borrow on a different ladder refuses by naming vertical alignment.  The
contributing mapping document is deliberately NOT a row of the input
manifest: the composition itself pins it by hash, and the manifest pins the
composition, so the donor's table version is bound transitively while the
manifest schema stays v1.  Provenance receipts name every contributing
source with its hashes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field as dataclass_field, replace as _dataclass_replace
from datetime import datetime
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
    _mapped_engine_choice,
    _materialize_frames,
    _require_authority_snapshot,
    _snapshot_authority,
    load_mapping,
    mapped_frames_to_regular_snapshots,
)
from gpuwm.mapped_engine_bridge import ENGINE_RUST as _ENGINE_RUST
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
    validate_soil_layer_contract,
)


COMPOSITION_SCHEMA = "gpuwm-mapped-composition-v2"
#: An explicit "no runnable composition exists yet" declaration.  An
#: atmosphere-only source (complete 3-D state, zero land surface) ships
#: this document in its composition role: loading it always refuses, and
#: the refusal names the state the source does not publish, so no
#: initialization can quietly be built on invented surfaces.  The
#: cross-source composition that will supply that state consumes the
#: same declaration when it arrives.
PENDING_COMPOSITION_SCHEMA = "gpuwm-cross-source-composition-pending-v1"
INPUT_MANIFEST_SCHEMA = "gpuwm-mapped-composition-inputs-v1"
RECEIPT_SCHEMA = "gpuwm-mapped-composition-receipt-v1"
BINDING_RECEIPT_SCHEMA = "gpuwm-cross-source-binding-v1"
_EXTERNAL_FIELD = "terrain_height"
_SOURCE_FORMATS = {"grib1", "grib2", "netcdf"}
_SOIL_PAIR = ("soil_temperature", "volumetric_soil_moisture")

#: The DECLARED clocks a cross-source binding can run on.
#:
#: ``valid_time_exact`` requires the contributing source at every primary
#: valid time.  ``cycle_invariant_broadcast`` is for borrowed statics the
#: donor publishes once per cycle: proven byte-invariant across every
#: supplied donor time, then carried to every primary time of the one
#: primary cycle.  ``source_cycle_analysis_broadcast`` is the hybrid
#: alignment: the donor supplies exactly ONE analysis record whose valid
#: time must BE the primary's source cycle, and that initialization state
#: is carried to every primary lead -- soil from a physical analysis under
#: an AI forecast is deliberately frozen at the shared analysis, and the
#: receipt names every carried time so nothing is silently frozen.
_BOUND_TIME_ALIGNMENTS = frozenset({
    "valid_time_exact", "cycle_invariant_broadcast",
    "source_cycle_analysis_broadcast",
})

#: The only landed grid relationship for a borrowed field: the donor grid
#: contains the primary grid as an exact coordinate subset (identical grids
#: are the degenerate full subset).  Anything else is cross-grid and
#: refuses by naming the regrid capability.
_BOUND_GRID_ALIGNMENTS = frozenset({"exact_coordinate_subset"})

_BINDING_KEYS = {
    "source_id", "mapping_role", "mapping_sha256", "data_role",
    "provenance_role", "fields", "grid_alignment", "time_alignment",
}


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
    if isinstance(raw, dict) and raw.get("schema") == PENDING_COMPOSITION_SCHEMA:
        raise ValueError(_pending_composition_refusal(raw, path))
    contract = _object(
        raw, "composition",
        allowed={
            "schema", "name", "mapping_binding", "soil_layers", "supplements",
            "field_sources",
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

    declared_bound = {
        name for name, field in mapping["fields"].items()
        if field.get("provider") == "composition_bound"
    }
    bindings = _validate_field_sources(
        contract.get("field_sources", {}), mapping, declared_bound,
    )
    covered = {
        name: binding_name
        for binding_name, binding in bindings.items()
        for name in binding["fields"]
    }
    unbound = sorted(declared_bound - set(covered))
    if unbound:
        raise ValueError(
            f"composition_bound field(s) {unbound} have no contributing "
            "source; every declared gap must be bound in field_sources"
        )
    soil_bindings = {
        covered[name] for name in _SOIL_PAIR if name in covered
    }
    if len(soil_bindings) > 1 or (
        soil_bindings and any(name not in covered for name in _SOIL_PAIR)
    ):
        raise ValueError(
            "the canonical soil pair must ride one contributing source: "
            "a soil-depth contract validates against a single donor "
            "mapping's layer geometry"
        )
    validate_soil_layer_contract(
        contract["soil_layers"],
        # When soil is borrowed, the layer geometry belongs to the donor
        # mapping, which decode re-validates once its bytes are pinned.
        mapping=None if soil_bindings else mapping,
    )

    supplements = _object(
        contract["supplements"], "composition.supplements",
        allowed={_EXTERNAL_FIELD}, required=set(),
    )
    terrain_supplement = supplements.get(_EXTERNAL_FIELD)
    if terrain_supplement is not None and _EXTERNAL_FIELD in covered:
        raise ValueError(
            "terrain has two providers: the declared supplement and a "
            "contributing source binding"
        )
    if terrain_supplement is None and _EXTERNAL_FIELD not in covered:
        raise ValueError(
            "terrain_height has no contributing source and no supplement; "
            "the composition must provide terrain through exactly one"
        )
    if terrain_supplement is not None:
        _validate_terrain_supplement(terrain_supplement, mapping)
    return contract


def _validate_terrain_supplement(
    value: object, mapping: Mapping[str, object],
) -> None:
    spec = _object(
        value,
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
    if spec["time_alignment"] not in _TERRAIN_TIME_ALIGNMENTS:
        raise ValueError(
            "terrain time alignment must be one of "
            f"{sorted(_TERRAIN_TIME_ALIGNMENTS)}"
        )
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


def _validate_field_sources(
    value: object,
    mapping: Mapping[str, object],
    declared_bound: set[str],
) -> dict[str, dict[str, object]]:
    """Validate the per-field source bindings of a cross-source composition."""

    if not isinstance(value, dict):
        raise TypeError("composition.field_sources must be an object")
    bindings: dict[str, dict[str, object]] = {}
    covered: dict[str, str] = {}
    roles: dict[str, str] = {}
    for binding_name, raw_binding in value.items():
        _string(binding_name, "field_sources binding name")
        label = f"composition.field_sources.{binding_name}"
        binding = _object(
            raw_binding, label, allowed=_BINDING_KEYS, required=_BINDING_KEYS,
        )
        _string(binding["source_id"], f"{label}.source_id")
        _digest(binding["mapping_sha256"], f"{label}.mapping_sha256")
        for key in ("mapping_role", "data_role", "provenance_role"):
            role = _string(binding[key], f"{label}.{key}")
            previous = roles.get(role)
            if previous is not None:
                raise ValueError(
                    f"{label}.{key} reuses role {role!r} already claimed by "
                    f"{previous}; every binding role must be unique"
                )
            roles[role] = f"{label}.{key}"
        if binding["grid_alignment"] not in _BOUND_GRID_ALIGNMENTS:
            raise ValueError(
                f"{label}.grid_alignment must be one of "
                f"{sorted(_BOUND_GRID_ALIGNMENTS)}: a cross-grid "
                "contribution has no landed regrid capability"
            )
        if binding["time_alignment"] not in _BOUND_TIME_ALIGNMENTS:
            raise ValueError(
                f"{label}.time_alignment must be one of "
                f"{sorted(_BOUND_TIME_ALIGNMENTS)}"
            )
        fields = binding["fields"]
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"{label}.fields must be a non-empty list")
        names = [_string(name, f"{label}.fields[]") for name in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"{label}.fields repeats a field")
        for name in names:
            declared = mapping["fields"].get(name)
            if declared is None:
                raise ValueError(f"{label} binds unknown field {name!r}")
            if name not in declared_bound:
                raise ValueError(
                    f"field {name!r} has two providers: the primary mapping "
                    f"and contributing source binding {binding_name!r}"
                )
            if name in covered:
                raise ValueError(
                    f"field {name!r} is bound by more than one contributing "
                    f"source ({covered[name]!r} and {binding_name!r})"
                )
            covered[name] = binding_name
        bindings[binding_name] = dict(binding, fields=list(names))
    return bindings


def _pending_composition_refusal(raw: Mapping[str, object], path: Path) -> str:
    """The named refusal a pending composition declaration always earns.

    The message is composed from the declaration's own table data: which
    canonical state the source does not publish, why, and where it must
    come from.  A malformed declaration refuses on its own account.
    """

    declaration = _object(
        raw, f"pending composition {path.name}",
        allowed={"schema", "name", "pending"},
        required={"schema", "name", "pending"},
    )
    name = _string(declaration["name"], "pending composition.name")
    pending = _object(
        declaration["pending"], "pending composition.pending",
        allowed={"missing_canonical_state", "reason", "supply_route"},
        required={"missing_canonical_state", "reason", "supply_route"},
    )
    missing = pending["missing_canonical_state"]
    if not isinstance(missing, list) or not missing:
        raise ValueError(
            f"{path.name} pending.missing_canonical_state must be a "
            "non-empty list of canonical field names"
        )
    names = ", ".join(
        _string(entry, "pending.missing_canonical_state entry")
        for entry in missing
    )
    reason = _string(pending["reason"], "pending composition.reason")
    route = _string(pending["supply_route"], "pending composition.supply_route")
    return (
        f"composition {name!r} is an explicit PENDING declaration, not a "
        f"runnable contract: the source publishes no {names}. {reason}. "
        f"Initializing from this source alone would invent those fields, "
        f"so it is refused; {route}."
    )


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
            "member", "member_identity",
        },
        required={
            "schema", "mapping_sha256", "composition_sha256",
            "primary_files", "supplements", "provenance", "decoders",
        },
    )
    if manifest["schema"] != INPUT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported composition input schema {manifest['schema']!r}")
    # An EXPLICIT ensemble-member binding, for archives whose product
    # octets carry none: the caller's verified authority named the
    # member, the manifest seals it, and the composition stamps it onto
    # every canonical frame.  Atomic pair -- a member with no identity
    # policy is a number with no provenance.
    if ("member" in manifest) != ("member_identity" in manifest):
        raise ValueError(
            "manifest member and member_identity are an atomic pair"
        )
    if "member" in manifest:
        _string(manifest["member"], "manifest.member")
        _string(manifest["member_identity"], "manifest.member_identity")
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


#: The two DECLARED shapes a terrain supplement's clock can take.
#:
#: ``valid_time_exact`` requires a terrain frame at every primary valid
#: time -- the default, and the only honest choice for a producer that
#: writes terrain into every step (HRRR's in-band terrain, 20CRv3's
#: per-time supplement).  ``cycle_invariant_broadcast`` is for the
#: producers that write their static fields only into the analysis frame
#: of a cycle (ECMWF open data, DWD's time-invariant objects, GDPS's
#: analysis statics): the terrain decodes wherever the supplement
#: supplies it, must STILL be invariant across every supplied frame, and
#: the one invariant field is carried to every primary valid time of the
#: ONE source cycle, with the carried times named in the receipt.  It is
#: the composition-side twin of the mapping grammar's
#: ``time_binding: cycle_invariant``.
_TERRAIN_TIME_ALIGNMENTS = frozenset({
    "valid_time_exact", "cycle_invariant_broadcast",
})


def _compose_terrain(
    primary: _DecodedCollection,
    terrain: _DecodedCollection,
    *,
    time_alignment: str = "valid_time_exact",
) -> tuple[_DecodedCollection, dict[str, object]]:
    if time_alignment not in _TERRAIN_TIME_ALIGNMENTS:
        raise ValueError(
            "terrain supplement time_alignment must be one of "
            f"{sorted(_TERRAIN_TIME_ALIGNMENTS)}, got {time_alignment!r}"
        )
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
    if missing_times and time_alignment == "valid_time_exact":
        raise ValueError(
            f"terrain supplement lacks exact primary valid time(s) {missing_times}"
        )
    if time_alignment != "valid_time_exact":
        # The producer publishes terrain at the analysis step only; the
        # supplement's one proven-invariant record answers every primary
        # valid time of the one source cycle.  One broadcast belongs to
        # one cycle: mixed primary cycles refuse rather than share an
        # invariant record no cycle proved for itself.
        primary_cycles = {
            str(cycle) for cycle in primary.source_cycles.values()
        }
        if len(primary_cycles) != 1:
            raise ValueError(
                "cycle-invariant terrain broadcast cannot span mixed "
                f"primary source cycles {sorted(primary_cycles)}"
            )
    # The invariance check above proved every supplied terrain frame is
    # byte-equal, so under the broadcast alignment the ONE invariant field
    # -- already the only field there is -- stands in at the primary valid
    # times the producer did not re-publish it for.  The carrier frame's
    # own metadata travels with it, and the receipt names each carried
    # time so a reader can see what was broadcast rather than matched.
    carrier = terrain_items[0][1]
    direct = dict(primary.direct)
    subset_reference = None
    for valid_time, member in primary_keys:
        supplied = terrain_by_time.get((valid_time, member), carrier)
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
        # Recorded only when the composition declared the broadcast
        # alignment, so every receipt written before these keys existed is
        # byte-identical to what the same preparation writes today; under
        # the broadcast the receipt names the alignment AND each carried
        # time, so a reader can see what was broadcast rather than matched.
        **(
            {
                "time_alignment": time_alignment,
                "broadcast_primary_valid_times": [
                    key[0].isoformat() for key in missing_times],
            }
            if time_alignment != "valid_time_exact" else {}
        ),
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


def _bind_manifest_member(
    collection: _DecodedCollection,
    member: str,
) -> _DecodedCollection:
    """Stamp the manifest's EXPLICIT member onto every composed record.

    For archives whose product octets carry no ensemble identity: the
    caller's verified authority named the member, the sealed input
    manifest declares it, and this replaces ABSENT product-defined
    membership only.  Bytes that already encode a member contradict the
    declaration and refuse rather than being overwritten -- overwriting
    would let one archive's frames claim another member's identity.
    """

    encoded = {value.member for value in collection.direct.values()}
    if encoded != {None}:
        raise ValueError(
            f"the composition input manifest binds explicit member "
            f"{member!r}, but the decoded records already encode "
            f"product-defined member(s) "
            f"{sorted(str(value) for value in encoded - {None})}; an "
            "explicit member binding replaces ABSENT product membership "
            "only, so stamping it here would overwrite a producer-declared "
            "identity"
        )
    direct = {}
    for (valid_time, _old_member, field_name), value in collection.direct.items():
        direct[(valid_time, member, field_name)] = _dataclass_replace(
            value, member=member)
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


def _binding_subset_indices(
    primary: _DecodedCollection,
    donor: _DecodedCollection,
    binding_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        latitude = _exact_subset_indices(
            donor.latitude, primary.latitude, "latitude",
        )
        longitude = _exact_subset_indices(
            donor.longitude, primary.longitude, "longitude",
            cyclic_degrees=True,
        )
    except ValueError as error:
        raise ValueError(
            f"contributing source binding {binding_name!r} is cross-grid "
            f"({error}); borrowing a field across grids requires a "
            "horizontal regrid capability this composition does not "
            "declare, so it refuses rather than interpolate"
        ) from error
    return latitude, longitude


def _take_subset(
    values: np.ndarray,
    axes: Sequence[str],
    latitude_indices: np.ndarray,
    longitude_indices: np.ndarray,
) -> np.ndarray:
    result = np.asarray(values)
    result = result.take(latitude_indices, axis=tuple(axes).index("y"))
    result = result.take(longitude_indices, axis=tuple(axes).index("x"))
    return result


def _compose_bound_fields(
    primary: _DecodedCollection,
    donor: _DecodedCollection,
    *,
    binding_name: str,
    binding: Mapping[str, object],
) -> tuple[_DecodedCollection, dict[str, object]]:
    """Inject one contributing source's bound fields into the primary decode.

    The donor collection is a partitioned decode of the contributing
    source's OWN mapping, so units, missing policy, and layer stacking are
    already the donor table's.  This join only aligns grids (exact
    coordinate subset or refuse naming regrid), aligns clocks by the
    binding's declared rule, refuses double provision, and keeps the
    donor's source cycle inside every injected value's provenance.
    """

    fields = tuple(str(name) for name in binding["fields"])
    alignment = str(binding["time_alignment"])
    if alignment not in _BOUND_TIME_ALIGNMENTS:
        raise ValueError(
            f"binding time_alignment must be one of "
            f"{sorted(_BOUND_TIME_ALIGNMENTS)}, got {alignment!r}"
        )
    donor_members = {member for _time, member in donor.source_cycles}
    if len(donor_members) != 1:
        raise ValueError(
            f"contributing source binding {binding_name!r} decoded "
            f"{len(donor_members)} members; cross-source borrowing has no "
            "member-alignment capability and requires a single-member donor"
        )
    donor_member = next(iter(donor_members))
    donor_inventory = {key[2] for key in donor.direct}
    if donor_inventory != set(fields):
        raise ValueError(
            f"contributing source binding {binding_name!r} decoded fields "
            f"{sorted(donor_inventory)} instead of the bound {sorted(fields)}"
        )
    provided_twice = sorted(
        {key[2] for key in primary.direct if key[2] in set(fields)}
    )
    if provided_twice:
        raise ValueError(
            f"field {provided_twice[0]!r} has two providers: the primary "
            f"decode and contributing source binding {binding_name!r}"
        )
    if any("vertical" in value.axes for value in donor.direct.values()) \
            and not np.array_equal(donor.vertical_values, primary.vertical_values):
        raise ValueError(
            f"contributing source binding {binding_name!r} borrows a "
            "vertical-bearing field on a different vertical ladder; "
            "cross-ladder borrowing requires a vertical interpolation "
            "capability this composition does not declare"
        )
    latitude_indices, longitude_indices = _binding_subset_indices(
        primary, donor, binding_name,
    )
    primary_keys = tuple(sorted(
        primary.source_cycles, key=lambda item: (item[0], str(item[1]))
    ))
    primary_cycles = sorted({
        cycle for cycle in primary.source_cycles.values()
    })
    by_field: dict[str, dict[datetime, _DirectValue]] = {}
    for (time_value, _member, name), value in donor.direct.items():
        by_field.setdefault(name, {})[time_value] = value

    direct = dict(primary.direct)
    matched_times: set[datetime] = set()
    broadcast_times: set[datetime] = set()
    subset_hashes: dict[str, str] = {}
    for name in fields:
        supplied = by_field[name]
        if alignment == "valid_time_exact":
            missing = [
                time for time, _member in primary_keys if time not in supplied
            ]
            if missing:
                raise ValueError(
                    f"contributing source binding {binding_name!r} lacks "
                    f"{name!r} at primary valid time(s) {missing}"
                )
            carrier = None
        elif alignment == "source_cycle_analysis_broadcast":
            if len(primary_cycles) != 1:
                raise ValueError(
                    "source-cycle analysis broadcast cannot span mixed "
                    f"primary source cycles {primary_cycles}"
                )
            if len(supplied) != 1:
                raise ValueError(
                    f"contributing source binding {binding_name!r} supplies "
                    f"{name!r} at {len(supplied)} valid times; the analysis "
                    "broadcast requires exactly one analysis record"
                )
            analysis_time = next(iter(supplied))
            if analysis_time != primary_cycles[0]:
                raise ValueError(
                    f"contributing source binding {binding_name!r} supplies "
                    f"{name!r} at {analysis_time.isoformat()}, which is not "
                    f"the primary source cycle "
                    f"{primary_cycles[0].isoformat()}; a hybrid borrows its "
                    "initialization state from the SAME cycle's analysis"
                )
            carrier = supplied[analysis_time]
        else:  # cycle_invariant_broadcast
            if len(primary_cycles) != 1:
                raise ValueError(
                    "cycle-invariant broadcast cannot span mixed primary "
                    f"source cycles {primary_cycles}"
                )
            ordered = [supplied[time] for time in sorted(supplied)]
            reference = ordered[0]
            if any(not np.array_equal(reference.values, value.values)
                   for value in ordered[1:]):
                raise ValueError(
                    f"contributing source binding {binding_name!r} field "
                    f"{name!r} changes across supplied valid times; the "
                    "cycle-invariant broadcast is for proven statics only"
                )
            carrier = reference
        subset_reference = None
        for valid_time, member in primary_keys:
            value = supplied.get(valid_time)
            if value is None:
                value = carrier
                broadcast_times.add(valid_time)
            else:
                matched_times.add(valid_time)
            values = _take_subset(
                value.values, value.axes, latitude_indices, longitude_indices,
            )
            if subset_reference is None:
                subset_reference = values
                subset_hashes[name] = _array_sha256(values)
            direct[(valid_time, member, name)] = _DirectValue(
                name=name,
                valid_time=valid_time,
                member=member,
                source_cycle=value.source_cycle,
                axes=value.axes,
                values=values,
                missing_count=int(np.isnan(values).sum()),
                references=value.references,
            )
    donor_cycles = sorted({
        value.source_cycle for value in donor.direct.values()
    })
    receipt = {
        "schema": BINDING_RECEIPT_SCHEMA,
        "status": "PASS",
        "binding": binding_name,
        "source_id": str(binding["source_id"]),
        "fields": sorted(fields),
        "grid_alignment": str(binding["grid_alignment"]),
        "coordinate_match": "exact_equivalent_contiguous_subset",
        "longitude_equivalence": "modulo_360_exact",
        "primary_shape": [int(primary.latitude.size), int(primary.longitude.size)],
        "donor_shape": [int(donor.latitude.size), int(donor.longitude.size)],
        "latitude_index_range": [
            int(latitude_indices[0]), int(latitude_indices[-1])],
        "longitude_index_range": [
            int(longitude_indices[0]), int(longitude_indices[-1])],
        "time_alignment": alignment,
        "donor_member": donor_member,
        "donor_source_cycles": [cycle.isoformat() for cycle in donor_cycles],
        "donor_valid_times": sorted(
            {key[0].isoformat() for key in donor.direct}
        ),
        "matched_primary_valid_times": sorted(
            time.isoformat() for time in matched_times
        ),
        "broadcast_primary_valid_times": sorted(
            time.isoformat() for time in broadcast_times
        ),
        "field_subset_sha256": subset_hashes,
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
    #: One record per cross-source binding: contributing source id, its own
    #: mapping authority hash, its data and provenance hashes, and the
    #: grid/time alignment receipt.  Empty for single-source compositions.
    contributing_sources: tuple[Mapping[str, object], ...] = dataclass_field(
        default=(),
    )

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
            self, "contributing_sources", tuple(
                MappingProxyType(dict(record))
                for record in self.contributing_sources
            )
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

        land_policy = self.soil_layer_contract["missing"]["land"]
        packed = mapped_frames_to_regular_snapshots(
            self.frames,
            soil_land_repair=(
                land_policy if isinstance(land_policy, Mapping) else None
            ),
        )
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
                projection=snapshot.projection,
            ))
        return tuple(result)


def _decoder_inventory(
    source_format: str | Sequence[str],
    *,
    grib1_bridge: str | Path | None,
    grib2_inventory: str | Path | None,
    grib2_dump: str | Path | None,
    engine: str | Path | None = None,
) -> dict[str, Path]:
    """Exact decoder roles for one source format, or a composed set of them.

    A cross-source composition may decode two formats in one preparation
    (a GRIB2 primary borrowing from a GRIB1 archive); the expected role set
    is then the union of each format's roles.

    On the Rust engine the inventory is ONE role -- the engine itself --
    whatever formats the composition spans, because the engine decodes
    GRIB1, GRIB2 and NetCDF in process and no per-format subprocess tool
    runs.  The manifest that seals a preparation binds the binary that
    actually read the bytes; a manifest sealed against the subprocess
    tools therefore refuses on this route rather than being replayed
    under a decoder that never ran (:func:`_verify_manifest` compares
    the role sets, and :func:`_engine_manifest_refusal` explains it).
    """

    if engine is not None:
        from gpuwm.mapped_engine_bridge import ENGINE_NAME

        provided = {
            role: Path(path).resolve()
            for role, path in {
                "grib1_bridge": grib1_bridge,
                "grib2_inventory": grib2_inventory,
                "grib2_dump": grib2_dump,
            }.items()
            if path is not None
        }
        if provided:
            raise ValueError(
                f"the {ENGINE_NAME} route decodes in process; it cannot "
                f"also run the subprocess tools {sorted(provided)}")
        return {ENGINE_NAME: Path(engine).resolve()}

    formats = (
        (source_format,) if isinstance(source_format, str)
        else tuple(source_format)
    )
    provided = {
        role: Path(path).resolve()
        for role, path in {
            "grib1_bridge": grib1_bridge,
            "grib2_inventory": grib2_inventory,
            "grib2_dump": grib2_dump,
        }.items()
        if path is not None
    }
    per_format = {
        "grib1": {"grib1_bridge"},
        "grib2": {"grib2_inventory", "grib2_dump"},
        "netcdf": set(),
    }
    expected: set[str] = set()
    for name in formats:
        expected |= per_format[name]
    label = "+".join(sorted(set(formats)))
    if set(provided) != expected:
        raise ValueError(
            f"{label} decoder inventory differs from the contract; "
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
        and field.get("provider") != "composition_bound"
    }
    if terrain_only and set(partition["fields"]) != {_EXTERNAL_FIELD}:
        raise ValueError("mapping lacks the direct terrain field")
    if not terrain_only and not partition["fields"]:
        raise ValueError("mapping has no primary fields after terrain partitioning")
    return partition


def _partition_contributing(
    donor_mapping: Mapping[str, object],
    field_names: Sequence[str],
    *,
    binding_name: str,
) -> dict[str, object]:
    """A decoder-only view of one contributing source's bound fields.

    The donor's sealed, fully validated mapping stays the sole semantic
    authority for the borrowed octets; partitioning only keeps the donor
    from demanding fields the composition never asked it for.  A borrowed
    field must be DIRECTLY selected in the donor: a derived donor field
    would need its dependency fields decoded too, and this partitioned
    decode deliberately does not carry them.
    """

    partition = copy.deepcopy(dict(donor_mapping))
    fields = partition["fields"]
    assert isinstance(fields, dict)
    kept: dict[str, object] = {}
    for name in field_names:
        field = fields.get(name)
        if field is None:
            raise ValueError(
                f"contributing source binding {binding_name!r} borrows "
                f"{name!r} but the contributing mapping does not map it"
            )
        if not field.get("selectors"):
            raise ValueError(
                f"contributing source binding {binding_name!r} borrows "
                f"{name!r}, which the contributing mapping does not provide "
                "with direct selectors; a derived or externally bound donor "
                "field is not decodable in a partitioned donor decode"
            )
        kept[name] = field
    partition["fields"] = kept
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


_COMPOSE_SCRATCH_ENV = "GPUWM_COMPOSE_SCRATCH"


def _compose_scratch_base(destination: Path | None) -> Path | None:
    """The directory the engine's compose scratch is created IN.

    The engine stages the WHOLE composed frame stream in the scratch --
    f64 frames, tens of GB for the largest registered sources -- so the
    placement is a correctness matter: the system temp is RAM-backed
    tmpfs on common Linux boxes (Ubuntu with ``/tmp`` on tmpfs and a
    user quota; Fedora mounts ``/tmp`` at half of RAM by default), and a
    bare default prep of the biggest registered source died there,
    deterministically, with "cannot write the frame stream: Disk quota
    exceeded (os error 122)".

    Resolution order:

    1. ``GPUWM_COMPOSE_SCRATCH`` -- an explicit scratch directory.
       Refused by name when it does not exist, because silently falling
       back would put the multi-GB stream on exactly the filesystem the
       caller set the variable to steer away from.
    2. ``destination`` -- the output directory a caller that owns a real
       preparation threads down (:func:`gpuwm.mapped_direct.prepare_mapped_wrf`).
       The scratch is created in the destination's PARENT -- the same
       disk-backed filesystem the output lands on anyway, and the same
       placement the route's own atomic staging uses -- while the
       destination itself stays untouched so its create-only refusal
       keeps meaning something.
    3. ``None`` -- the system temp, for callers with no real output
       directory (in-memory recompose paths), whose streams are sized
       to what they already hold in memory.
    """

    import os

    override = os.environ.get(_COMPOSE_SCRATCH_ENV)
    if override:
        base = Path(override)
        if not base.is_dir():
            raise NotADirectoryError(
                f"{_COMPOSE_SCRATCH_ENV}={override} does not name an "
                "existing directory.  The engine stages the whole composed "
                "frame stream there -- tens of GB for the largest sources "
                "-- and silently falling back to the system temp would put "
                "that stream on exactly the filesystem this variable was "
                "set to avoid")
        return base
    if destination is None:
        return None
    destination = Path(destination).resolve()
    base = destination.parent
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise NotADirectoryError(
            f"cannot create the parent directory of the preparation "
            f"output {destination}: {error}.  The compose scratch stages "
            f"the multi-GB frame stream in {base} so it shares the "
            "output's disk-backed filesystem; choose an output root whose "
            "parent is writable") from error
    return base


def _compose_through_engine(
    *,
    engine: Path,
    mapping_path: Path,
    composition_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    primary: Sequence[Path],
    supplements: Mapping[str, Sequence[Path]],
    provenance: Mapping[str, Path],
    contributing: Mapping[str, Path],
    contract: Mapping[str, object],
    bindings: Mapping[str, Mapping[str, object]],
    terrain_spec: Mapping[str, object] | None,
    decoders: Mapping[str, Path],
    snapshots: Mapping[Path, _AuthoritySnapshot],
    before: Mapping[str, str],
    member: str | None = None,
    member_identity: str | None = None,
    scratch_destination: Path | None = None,
) -> MappedSourceBundle:
    """Compose on the Rust engine, keeping every policy check on this side.

    Everything above this call has already run: manifest verification,
    the role inventory, the primary-versus-donor field agreement, the
    authority hash window.  What crosses the seam is the byte work the
    design's §1.1 names -- partition decode, terrain composition, bound
    fields, frame materialization -- and what comes back is a frameset
    plus the two evidence products only that work can produce.

    The soil-layer contract is validated against the DONOR table here
    exactly as the Python engine does it, because it is a check on
    sealed documents rather than on bytes and must not become the
    engine's to remember.

    ``scratch_destination`` places the engine's scratch -- which holds
    the whole composed frame stream until read-back -- on the same
    filesystem as the preparation output; see
    :func:`_compose_scratch_base` for the resolution order and the
    tmpfs quota failure it prevents.  The scratch is deleted after
    read-back, success or failure, either way.
    """

    import os
    import tempfile

    from gpuwm import mapped_engine_bridge

    scratch_base = _compose_scratch_base(scratch_destination)
    with tempfile.TemporaryDirectory(
        prefix="gpuwm-mapped-compose-",
        dir=None if scratch_base is None else os.fspath(scratch_base),
    ) as work:
        directory = Path(work) / "composed"
        mapped_engine_bridge.run_engine(
            "compose",
            mapping=mapping_path,
            files=primary,
            output=directory,
            composition=composition_path,
            supplements=supplements,
            provenance=provenance,
            contributing_mappings=contributing,
            input_manifest=manifest_path,
            input_manifest_sha256=manifest_sha256,
        )
        evidence = mapped_engine_bridge.read_composition_evidence(directory)
        frames = mapped_engine_bridge.read_frameset(directory)

    for frame in frames:
        if frame.mapping_sha256 != before[str(mapping_path)]:
            raise RuntimeError(
                f"{mapped_engine_bridge.ENGINE_NAME} composed against "
                f"mapping bytes hashing to {frame.mapping_sha256}; this "
                f"call verified {mapping_path} at "
                f"{before[str(mapping_path)]}, so the bundle would be "
                "bound to a document nobody checked")
    if member is not None:
        # The manifest bound an explicit ensemble member; an engine that
        # returned frames without it decoded under a contract this call
        # did not seal, for an ENSEMBLE source.
        stamped = {frame.member for frame in frames}
        if stamped != {member}:
            raise RuntimeError(
                f"{mapped_engine_bridge.ENGINE_NAME} returned frames with "
                f"member(s) {sorted(str(value) for value in stamped)}; the "
                f"sealed input manifest binds explicit member {member!r}, "
                "so the ensemble identity did not survive the engine")
        recorded_alignment = dict(evidence["alignment_receipt"])
        if (recorded_alignment.get("member") != member
                or recorded_alignment.get("member_identity")
                != member_identity):
            raise RuntimeError(
                f"{mapped_engine_bridge.ENGINE_NAME} composed an alignment "
                "receipt without the manifest's explicit member binding, "
                "so the preparation's sealed receipt would lose the "
                "ensemble identity")

    contributing_records = tuple(
        dict(record) for record in evidence["contributing_sources"]
    )
    recorded = {str(record["binding"]) for record in contributing_records}
    if recorded != set(bindings):
        raise RuntimeError(
            f"{mapped_engine_bridge.ENGINE_NAME} composed "
            f"{sorted(recorded)} cross-source bindings; the composition "
            f"declares {sorted(bindings)}, so a declared borrow either "
            "did not happen or was invented")

    soil_donor_mapping: Mapping[str, object] | None = None
    terrain_binding: Mapping[str, object] | None = None
    for binding_name in sorted(bindings):
        binding = bindings[binding_name]
        bound_names = [str(name) for name in binding["fields"]]
        if _EXTERNAL_FIELD in bound_names:
            terrain_binding = binding
        if any(name in bound_names for name in _SOIL_PAIR):
            donor_path = contributing[str(binding["mapping_role"])]
            soil_donor_mapping = load_mapping(donor_path)
    if soil_donor_mapping is not None:
        validate_soil_layer_contract(
            contract["soil_layers"], mapping=soil_donor_mapping,
        )

    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)

    if terrain_spec is not None:
        terrain_paths = tuple(supplements[str(terrain_spec["data_role"])])
        terrain_provenance_path = provenance[
            str(terrain_spec["provenance_role"])]
    else:
        if terrain_binding is None:
            raise ValueError(
                "composed source has neither a terrain supplement nor a "
                "terrain field binding")
        terrain_paths = tuple(supplements[str(terrain_binding["data_role"])])
        terrain_provenance_path = provenance[
            str(terrain_binding["provenance_role"])]
    return MappedSourceBundle(
        frames=frames,
        mapping_path=mapping_path,
        mapping_sha256=before[str(mapping_path)],
        composition_path=composition_path,
        composition_sha256=before[str(composition_path)],
        input_manifest_path=manifest_path,
        input_manifest_sha256=before[str(manifest_path)],
        decoder_paths=dict(decoders),
        decoder_sha256={
            role: before[str(path)] for role, path in decoders.items()
        },
        terrain_data_paths=terrain_paths,
        terrain_data_sha256=tuple(
            before[str(path)] for path in terrain_paths
        ),
        terrain_provenance_path=terrain_provenance_path,
        terrain_provenance_sha256=before[str(terrain_provenance_path)],
        soil_layer_contract=contract["soil_layers"],
        alignment_receipt=dict(evidence["alignment_receipt"]),
        contributing_sources=contributing_records,
    )


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
    contributing_mappings: Mapping[str, str | Path] | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
    scratch_destination: str | Path | None = None,
) -> MappedSourceBundle:
    """Decode a complete mapped source with scientifically sourced terrain.

    A cross-source composition additionally supplies each contributing
    source's own sealed mapping document under its declared
    ``mapping_role`` via ``contributing_mappings``; the observed bytes must
    hash to the SHA-256 the composition pins for that binding.  The
    contributing data files ride ``supplement_files`` and the contributing
    provenance documents ride ``provenance_files``, each under the
    binding's declared role, so the input manifest binds every byte it
    always bound; the contributing mapping itself is pinned by the
    composition (which the manifest pins), not by a manifest row.

    ``scratch_destination`` is the output directory of the preparation
    this decode feeds, when the caller has one: the engine's compose
    scratch -- the whole composed frame stream, tens of GB for the
    largest sources -- is created beside it instead of in the system
    temp, which is RAM-backed tmpfs on common Linux boxes and killed the
    bare default prep of the biggest registered source with a disk
    quota error.  ``GPUWM_COMPOSE_SCRATCH`` overrides it; omitted
    entirely, the system temp remains the documented fallback for
    callers with no real output directory.
    """

    composition_path = Path(composition_path).resolve()
    mapping_path = Path(mapping_path).resolve()
    manifest_path = Path(input_manifest).resolve()
    primary = tuple(Path(path).resolve() for path in primary_files)
    supplements = {
        role: _path_inventory(path, f"supplement {role!r}")
        for role, path in supplement_files.items()
    }
    provenance = {role: Path(path).resolve() for role, path in provenance_files.items()}
    contributing = {
        str(role): Path(path).resolve()
        for role, path in (contributing_mappings or {}).items()
    }
    if not primary or len(set(primary)) != len(primary):
        raise ValueError("primary source inventory must be non-empty and unique")
    for path in (
        composition_path, mapping_path, manifest_path,
        *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
        *contributing.values(),
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
    bindings = {
        str(name): binding
        for name, binding in (contract.get("field_sources") or {}).items()
    }
    terrain_spec = contract["supplements"].get(_EXTERNAL_FIELD)

    expected_mapping_roles = {
        str(binding["mapping_role"]) for binding in bindings.values()
    }
    if set(contributing) != expected_mapping_roles:
        raise ValueError(
            "contributing mapping role inventory differs from the "
            f"composition: expected {sorted(expected_mapping_roles)}, "
            f"got {sorted(contributing)}"
        )
    donor_mappings: dict[str, Mapping[str, object]] = {}
    donor_paths: dict[str, Path] = {}
    contributing_snapshots: dict[Path, _AuthoritySnapshot] = {}
    for binding_name in sorted(bindings):
        binding = bindings[binding_name]
        donor_path = contributing[str(binding["mapping_role"])]
        snapshot = _snapshot_authority(donor_path, retain_bytes=True)
        pinned = _digest(
            binding["mapping_sha256"],
            f"field_sources.{binding_name}.mapping_sha256",
        )
        if snapshot.sha256 != pinned:
            raise ValueError(
                f"contributing source binding {binding_name!r} authority "
                f"hash mismatch: the composition pins {pinned}, the supplied "
                f"mapping bytes hash to {snapshot.sha256}"
            )
        donor_raw = _load_json_bytes(
            snapshot.data, "contributing mapping", donor_path,
        )
        donor_mapping = load_mapping(donor_path, _raw=donor_raw)
        if str(donor_mapping["name"]) != str(binding["source_id"]):
            raise ValueError(
                f"contributing source binding {binding_name!r} names "
                f"source_id {binding['source_id']!r} but the pinned mapping "
                f"is {donor_mapping['name']!r}"
            )
        for name in binding["fields"]:
            name = str(name)
            declared = mapping["fields"][name]
            donor_field = donor_mapping["fields"].get(name)
            if donor_field is None:
                continue  # _partition_contributing refuses with the full story
            for aspect, ours, theirs in (
                ("units.target", declared["units"]["target"],
                 donor_field["units"]["target"]),
                ("target_axes", tuple(declared["target_axes"]),
                 tuple(donor_field["target_axes"])),
                ("location", declared["location"], donor_field["location"]),
                ("staggering", declared.get("staggering", "none"),
                 donor_field.get("staggering", "none")),
                ("missing.kind", declared["missing"]["kind"],
                 donor_field["missing"]["kind"]),
            ):
                if ours != theirs:
                    raise ValueError(
                        f"field {name!r} {aspect} disagrees between the "
                        f"primary mapping ({ours!r}) and contributing source "
                        f"{binding['source_id']!r} ({theirs!r})"
                    )
        donor_mappings[binding_name] = donor_mapping
        donor_paths[binding_name] = donor_path
        contributing_snapshots[donor_path] = snapshot

    formats = {str(mapping["format"])} | {
        str(donor_mapping["format"])
        for donor_mapping in donor_mappings.values()
    }
    engine_binary = None
    if _mapped_engine_choice(
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
        subcommand="compose",
        source_format=str(mapping["format"]),
    ) == _ENGINE_RUST:
        from gpuwm.mapped_engine_bridge import require_engine

        engine_binary = require_engine()
    decoders = _decoder_inventory(
        sorted(formats),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
        engine=engine_binary,
    )
    for path in decoders.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_data_roles: set[str] = set()
    expected_provenance_roles: set[str] = set()
    if terrain_spec is not None:
        terrain_data_role = str(terrain_spec["data_role"])
        terrain_provenance_role = str(terrain_spec["provenance_role"])
        expected_data_roles.add(terrain_data_role)
        expected_provenance_roles.add(terrain_provenance_role)
    for binding in bindings.values():
        expected_data_roles.add(str(binding["data_role"]))
        expected_provenance_roles.add(str(binding["provenance_role"]))
    if set(supplements) != expected_data_roles \
            or set(provenance) != expected_provenance_roles:
        raise ValueError("composition role inventory differs from the contract")

    authorities = (
        composition_path, mapping_path, manifest_path,
        *decoders.values(), *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
        *contributing.values(),
    )
    snapshots = {
        composition_path: composition_snapshot,
        mapping_path: mapping_snapshot,
        manifest_path: manifest_snapshot,
        **contributing_snapshots,
    }
    for path in authorities:
        if path not in snapshots:
            snapshots[path] = _snapshot_authority(path)
    manifest_document = _verify_manifest(
        manifest_path, input_manifest_sha256,
        mapping_path=mapping_path, composition_path=composition_path,
        primary_files=primary, supplement_files=supplements,
        provenance_files=provenance, decoder_files=decoders,
        _snapshots=snapshots,
        _recheck_snapshots=False,
    )
    declared_member = manifest_document.get("member")
    declared_member_identity = manifest_document.get("member_identity")
    for snapshot in (
        mapping_snapshot,
        composition_snapshot,
        manifest_snapshot,
        *contributing_snapshots.values(),
    ):
        _require_authority_snapshot(snapshot)
    before = {str(path): snapshots[path].sha256 for path in authorities}
    if engine_binary is not None:
        return _compose_through_engine(
            engine=engine_binary,
            mapping_path=mapping_path,
            composition_path=composition_path,
            manifest_path=manifest_path,
            manifest_sha256=input_manifest_sha256,
            primary=primary,
            supplements=supplements,
            provenance=provenance,
            contributing=contributing,
            contract=contract,
            bindings=bindings,
            terrain_spec=terrain_spec,
            decoders=decoders,
            snapshots=snapshots,
            before=before,
            member=declared_member,
            member_identity=declared_member_identity,
            scratch_destination=(
                None if scratch_destination is None
                else Path(scratch_destination)),
        )
    combined = _decode_partition(
        _partition_mapping(mapping, terrain_only=False), primary, decoders,
    )
    terrain_supplement_receipt = None
    if terrain_spec is not None:
        terrain_collection = _decode_partition(
            _partition_mapping(mapping, terrain_only=True),
            supplements[terrain_data_role], decoders,
        )
        combined, terrain_supplement_receipt = _compose_terrain(
            combined, terrain_collection,
            time_alignment=str(terrain_spec["time_alignment"]),
        )

    contributing_records: list[dict[str, object]] = []
    terrain_binding: Mapping[str, object] | None = None
    terrain_binding_receipt: dict[str, object] | None = None
    soil_donor_mapping: Mapping[str, object] | None = None
    for binding_name in sorted(bindings):
        binding = bindings[binding_name]
        bound_names = [str(name) for name in binding["fields"]]
        donor_files = supplements[str(binding["data_role"])]
        donor_collection = _decode_partition(
            _partition_contributing(
                donor_mappings[binding_name], bound_names,
                binding_name=binding_name,
            ),
            donor_files, decoders,
        )
        combined, receipt = _compose_bound_fields(
            combined, donor_collection,
            binding_name=binding_name, binding=binding,
        )
        provenance_path = provenance[str(binding["provenance_role"])]
        contributing_records.append({
            "binding": binding_name,
            "source_id": str(binding["source_id"]),
            "mapping": {
                "path": str(donor_paths[binding_name]),
                "sha256": before[str(donor_paths[binding_name])],
            },
            "data": [
                {"path": str(path), "sha256": before[str(path)]}
                for path in donor_files
            ],
            "provenance": {
                "path": str(provenance_path),
                "sha256": before[str(provenance_path)],
            },
            "fields": sorted(bound_names),
            "alignment": receipt,
        })
        if _EXTERNAL_FIELD in bound_names:
            terrain_binding = binding
            terrain_binding_receipt = receipt
        if any(name in bound_names for name in _SOIL_PAIR):
            soil_donor_mapping = donor_mappings[binding_name]
    if soil_donor_mapping is not None:
        # The soil geometry the composition declares must be the DONOR
        # table's, now that the donor's sealed bytes are pinned and loaded.
        validate_soil_layer_contract(
            contract["soil_layers"], mapping=soil_donor_mapping,
        )

    # The manifest's explicit member binding lands LAST, after every
    # join: the composition operates on the identity the bytes carry
    # (donor alignment refuses member-bearing donors on its own terms),
    # and the stamp then reaches every canonical frame at once.
    if declared_member is not None:
        combined = _bind_manifest_member(combined, declared_member)

    # Materialize against the union view: the primary mapping's own fields
    # plus, for every borrowed field, the contributing mapping's sealed
    # field spec -- the decode authority for those octets.
    union = copy.deepcopy(dict(mapping))
    union_fields = dict(union["fields"])
    for binding_name in sorted(bindings):
        donor_mapping = donor_mappings[binding_name]
        for name in bindings[binding_name]["fields"]:
            union_fields[str(name)] = copy.deepcopy(
                dict(donor_mapping["fields"][str(name)])
            )
    union["fields"] = union_fields

    input_hashes = {
        str(path): before[str(path)]
        for path in (
            *primary, *(path for paths in supplements.values() for path in paths)
        )
    }
    frames = _materialize_frames(
        union, combined, mapping_sha256=before[str(mapping_path)],
        input_sha256=input_hashes,
    )
    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    if terrain_spec is not None:
        terrain_paths = supplements[terrain_data_role]
        terrain_provenance_path = provenance[terrain_provenance_role]
        terrain_receipt = terrain_supplement_receipt
    else:
        assert terrain_binding is not None and terrain_binding_receipt is not None
        terrain_paths = supplements[str(terrain_binding["data_role"])]
        terrain_provenance_path = provenance[
            str(terrain_binding["provenance_role"])
        ]
        terrain_receipt = terrain_binding_receipt
    if declared_member is not None:
        # The sealed alignment receipt keeps the ensemble identity the
        # manifest bound, beside the alignment it describes.  Recorded
        # only under an explicit binding, so every receipt written
        # before the binding existed is byte-identical today.
        terrain_receipt = {
            **terrain_receipt,
            "member": declared_member,
            "member_identity": declared_member_identity,
        }
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
        terrain_data_paths=terrain_paths,
        terrain_data_sha256=tuple(
            before[str(path)] for path in terrain_paths
        ),
        terrain_provenance_path=terrain_provenance_path,
        terrain_provenance_sha256=before[str(terrain_provenance_path)],
        soil_layer_contract=contract["soil_layers"],
        alignment_receipt=terrain_receipt,
        contributing_sources=tuple(contributing_records),
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
        # Present only for cross-source compositions, so every receipt a
        # single-source preparation wrote before this key existed stays
        # byte-identical to what the same preparation writes today.  Each
        # entry names one contributing source: its id, its OWN sealed
        # mapping authority hash, its data and provenance hashes, and the
        # grid/time alignment receipt for the borrowed fields.
        **(
            {
                "contributing_sources": [
                    {
                        "binding": str(record["binding"]),
                        "source_id": str(record["source_id"]),
                        "mapping": dict(record["mapping"]),
                        "data": [dict(row) for row in record["data"]],
                        "provenance": dict(record["provenance"]),
                        "fields": list(record["fields"]),
                        "alignment": dict(record["alignment"]),
                    }
                    for record in bundle.contributing_sources
                ],
            }
            if bundle.contributing_sources else {}
        ),
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
