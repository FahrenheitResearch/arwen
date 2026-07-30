"""Verified arbitrary-GRIB front door for the generic mapped source.

``gpuwm adapt`` compiles a WPS Vtable plus an explicit descriptor, inventories
the caller's real GRIB2 files, and publishes only after every capability gate
passes.  The result is runnable through the existing mapped-composition
engine.  It is never promoted to stock-WRF certification by this module.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

from gpuwm.mapped_authoring import (
    DESCRIPTOR_SCHEMA,
    _canonical_json,
    _require_snapshot,
    _stable_file_snapshot,
    _strict_json,
    _write_new,
    author_input_manifest,
    compile_mapping_descriptor,
    parse_wps_vtable,
)
from gpuwm.mapped_composition import COMPOSITION_SCHEMA, load_composition
from gpuwm.mapped_source import (
    _build_grib2_tools,
    _decode_grib,
    _grib2_inventory,
    _require_authority_snapshot,
    _sha256,
    _snapshot_authority,
    load_mapping,
)


ADAPT_PROVENANCE_SCHEMA = "gpuwm-adapt-provenance-v1"
ADAPTER_STATUS = "runnable_mapping_not_stock_wrf_certified"
_SUPPORTED_FORMAT = "grib2"
_GRID_KEYS = (
    "gdt",
    "nx",
    "ny",
    "lat1",
    "lon1",
    "dx",
    "dy",
    "scan_mode",
    "shape_of_earth",
    "resolution_flags",
)
_SELECTOR_INTEGER_KEYS = (
    "discipline",
    "category",
    "parameter",
    "center",
    "subcenter",
    "master_table_version",
    "local_table_version",
    "level_type",
)


def _object(
    value: object,
    label: str,
    *,
    allowed: set[str],
    required: set[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing required key(s): {missing}")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _load_adapt_policy(descriptor_path: Path) -> tuple[dict[str, object], object]:
    snapshot = _stable_file_snapshot(descriptor_path)
    raw = _strict_json(snapshot.data, f"descriptor {snapshot.path}")
    if not isinstance(raw, dict) or raw.get("schema") != DESCRIPTOR_SCHEMA:
        raise ValueError(
            f"adapt descriptor must use schema {DESCRIPTOR_SCHEMA!r}"
        )
    policy = _object(
        raw.get("adapt"),
        "descriptor.adapt",
        allowed={"model_top_pa", "soil_policy"},
        required={"model_top_pa", "soil_policy"},
    )
    _positive_number(policy["model_top_pa"], "descriptor.adapt.model_top_pa")
    soil = _object(
        policy["soil_policy"],
        "descriptor.adapt.soil_policy",
        allowed={"kind", "target_layers_m"},
        required={"kind"},
    )
    if soil["kind"] not in {
        "identity_complete_layers",
        "conservative_layer_means",
    }:
        raise ValueError(
            "descriptor.adapt.soil_policy.kind must be "
            "'identity_complete_layers' or 'conservative_layer_means'"
        )
    if soil["kind"] == "identity_complete_layers":
        if "target_layers_m" in soil:
            raise ValueError(
                "identity_complete_layers derives target layers from the "
                "complete source inventory and cannot set target_layers_m"
            )
    elif "target_layers_m" not in soil:
        raise ValueError(
            "conservative_layer_means requires target_layers_m"
        )
    _require_snapshot(snapshot)
    return policy, snapshot


def _layers(raw: object, label: str) -> list[dict[str, float]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"{label} must be a non-empty list of [top, bottom] metres"
        )
    result = []
    for index, pair in enumerate(raw):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{label}[{index}] must be [top, bottom] metres")
        top = (
            _positive_number(pair[0], f"{label}[{index}][0]")
            if pair[0] != 0
            else 0.0
        )
        bottom = _positive_number(pair[1], f"{label}[{index}][1]")
        if bottom <= top:
            raise ValueError(
                f"{label}[{index}] bottom {bottom:g} m must exceed top "
                f"{top:g} m"
            )
        result.append({"top": top, "bottom": bottom})
    _require_contiguous_layers(result, label)
    return result


def _require_contiguous_layers(
    layers: Sequence[Mapping[str, float]],
    label: str,
) -> None:
    if not layers:
        raise ValueError(f"{label} is empty")
    if not math.isclose(float(layers[0]["top"]), 0.0, abs_tol=1e-12):
        raise ValueError(
            f"soil-layer check failed: {label} starts at "
            f"{float(layers[0]['top']):g} m, not the surface (0 m)"
        )
    for index, (left, right) in enumerate(zip(layers, layers[1:])):
        left_bottom = float(left["bottom"])
        right_top = float(right["top"])
        if not math.isclose(left_bottom, right_top, abs_tol=1e-12):
            relation = "gap" if right_top > left_bottom else "overlap"
            layer_label = (
                "source layers"
                if label == "source soil inventory"
                else f"{label} layers"
            )
            raise ValueError(
                f"soil-layer check failed: {relation} between {layer_label} "
                f"{index} and {index + 1}: {left_bottom:g} m to "
                f"{right_top:g} m"
            )


def _soil_source_layers(
    mapping: Mapping[str, object],
) -> list[dict[str, object]]:
    names = ("soil_temperature", "volumetric_soil_moisture")
    fields = mapping["fields"]
    missing = [name for name in names if name not in fields]
    if missing:
        raise ValueError(
            f"soil-layer check failed: mapping is missing canonical soil "
            f"field(s) {missing}"
        )
    selector_lists = [fields[name].get("selectors", []) for name in names]
    if not all(isinstance(value, list) and value for value in selector_lists):
        raise ValueError(
            "soil-layer check failed: both canonical soil fields require "
            "ordered direct selectors"
        )
    if len(selector_lists[0]) != len(selector_lists[1]):
        raise ValueError(
            "soil-layer check failed: temperature/moisture selector counts "
            f"differ ({len(selector_lists[0])} != {len(selector_lists[1])})"
        )
    source_layers: list[dict[str, object]] = []
    for index, pair in enumerate(zip(*selector_lists)):
        bounds = []
        for name, selector in zip(names, pair):
            if (
                selector.get("level_value") is None
                or selector.get("second_level_type") is None
                or selector.get("second_level_value") is None
            ):
                raise ValueError(
                    f"soil-layer check failed: {name} selector {index} does "
                    "not bind both layer surfaces"
                )
            if int(selector["level_type"]) != int(
                selector["second_level_type"]
            ):
                raise ValueError(
                    f"soil-layer check failed: {name} selector {index} uses "
                    "different first/second surface types"
                )
            bounds.append(
                (
                    float(selector["level_value"]),
                    float(selector["second_level_value"]),
                )
            )
        if bounds[0] != bounds[1]:
            raise ValueError(
                f"soil-layer check failed: temperature/moisture layer "
                f"{index} bounds differ ({bounds[0]} != {bounds[1]})"
            )
        top, bottom = bounds[0]
        if bottom <= top:
            raise ValueError(
                f"soil-layer check failed: source layer {index} bottom "
                f"{bottom:g} m must exceed top {top:g} m"
            )
        source_layers.append(
            {
                "top": top,
                "bottom": bottom,
                "selectors": {
                    name: copy.deepcopy(selector)
                    for name, selector in zip(names, pair)
                },
            }
        )
    _require_contiguous_layers(source_layers, "source soil inventory")
    expected = mapping["target"].get("soil_layer_count")
    if expected != len(source_layers):
        raise ValueError(
            "soil-layer check failed: source layer count "
            f"{len(source_layers)} differs from target.soil_layer_count "
            f"{expected!r}"
        )
    return source_layers


def build_composition(
    mapping: Mapping[str, object],
    adapt_policy: Mapping[str, object],
) -> dict[str, object]:
    """Build the executable v2 composition from explicit descriptor policy."""

    source_layers = _soil_source_layers(mapping)
    soil_policy = adapt_policy["soil_policy"]
    if soil_policy["kind"] == "identity_complete_layers":
        target_layers = [
            {"top": row["top"], "bottom": row["bottom"]}
            for row in source_layers
        ]
    else:
        target_layers = _layers(
            soil_policy["target_layers_m"],
            "descriptor.adapt.soil_policy.target_layers_m",
        )
        source_bottom = float(source_layers[-1]["bottom"])
        target_bottom = float(target_layers[-1]["bottom"])
        if target_bottom > source_bottom + 1e-12:
            raise ValueError(
                "soil-layer check failed: conservative target coverage ends "
                f"at {target_bottom:g} m but source coverage ends at "
                f"{source_bottom:g} m"
            )
    if len(target_layers) != mapping["target"].get("soil_layer_count"):
        raise ValueError(
            "soil-layer check failed: declared target layer count "
            f"{len(target_layers)} differs from target.soil_layer_count "
            f"{mapping['target'].get('soil_layer_count')!r}"
        )
    return {
        "schema": COMPOSITION_SCHEMA,
        "name": f"{mapping['name']}-adapt-composition",
        "mapping_binding": "input_manifest_sha256",
        "soil_layers": {
            "temperature_field": "soil_temperature",
            "moisture_field": "volumetric_soil_moisture",
            "depth_units": "m",
            "source_layers": source_layers,
            "target_layers": target_layers,
            "remap": {
                "kind": "conservative_layer_means",
                "source_value_location": "layer_mean",
                "target_value_location": "layer_mean",
                "coverage": "require_complete",
            },
            "missing": {
                "land": "reject",
                "ocean": {
                    "stage": "after_horizontal_interpolation",
                    "temperature": "skin_temperature",
                    "moisture": 1.0,
                },
            },
        },
        "supplements": {
            "terrain_height": {
                "data_role": "adapt_in_band_terrain",
                "provenance_role": "adapt_authority_provenance",
                "format": _SUPPORTED_FORMAT,
                "field": "terrain_height",
                "selector_authority": "mapping_field_exact",
                "grid_alignment": "exact_coordinate_subset",
                "time_alignment": "valid_time_exact",
                "require_invariant_across_time": True,
            },
        },
    }


def _selector_matches(
    selector: Mapping[str, object],
    row: Mapping[str, str],
) -> bool:
    for key in _SELECTOR_INTEGER_KEYS:
        if key in selector and int(row[key]) != int(selector[key]):
            return False
    if "level_value" in selector and not math.isclose(
        float(row["level_value"]),
        float(selector["level_value"]),
        abs_tol=1e-9,
    ):
        return False
    if "member" in selector:
        observed = None if row["member"] == "-" else int(row["member"])
        if observed != int(selector["member"]):
            return False
    has_second = "second_level_type" in selector
    observed_second = int(row["second_level_type"])
    if not has_second:
        return observed_second == 255
    return (
        observed_second == int(selector["second_level_type"])
        and math.isclose(
            float(row["second_level_value"]),
            float(selector["second_level_value"]),
            abs_tol=1e-9,
        )
    )


def _valid_time(row: Mapping[str, str]) -> datetime:
    reference = datetime.fromisoformat(row["reference_time"])
    unit = int(row["forecast_unit"])
    amount = int(row["forecast_time"])
    factors = {
        0: timedelta(minutes=1),
        1: timedelta(hours=1),
        2: timedelta(days=1),
        10: timedelta(hours=3),
        11: timedelta(hours=6),
        12: timedelta(hours=12),
        13: timedelta(seconds=1),
    }
    if unit not in factors:
        raise ValueError(
            f"record-inventory check failed: unsupported GRIB2 forecast "
            f"time unit {unit}"
        )
    return reference + amount * factors[unit]


def _selector_label(selector: Mapping[str, object]) -> str:
    keys = (
        "discipline",
        "category",
        "parameter",
        "level_type",
        "level_value",
        "second_level_type",
        "second_level_value",
    )
    return ", ".join(
        f"{key}={selector[key]}" for key in keys if key in selector
    )


def _validate_static_bindings(
    mapping: Mapping[str, object],
    adapt_policy: Mapping[str, object],
) -> dict[str, object]:
    if mapping["format"] != _SUPPORTED_FORMAT:
        raise ValueError(
            f"gpuwm adapt v1.1 supports GRIB2 descriptors only; got "
            f"{mapping['format']!r}. Use the named-adapter path for other "
            "formats."
        )
    levels = mapping["coordinates"]["vertical"].get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError(
            "vertical-coverage check failed: descriptor must declare the "
            "complete pressure coordinate in coordinates.vertical.levels"
        )
    source_top = min(float(level) for level in levels)
    model_top = _positive_number(
        adapt_policy["model_top_pa"],
        "descriptor.adapt.model_top_pa",
    )
    if source_top > model_top:
        raise ValueError(
            "vertical-coverage check failed: source top "
            f"{source_top:g} Pa does not cover model top {model_top:g} Pa; "
            f"provide source records at or above {model_top:g} Pa"
        )
    fields = mapping["fields"]
    for required in mapping["target"]["required_fields"]:
        name = required["name"]
        field = fields.get(name)
        if field is None:
            raise ValueError(
                f"units/axes/staggering check failed: required field "
                f"{name!r} is not mapped"
            )
        expected = (
            tuple(required["axes"]),
            required["location"],
            required["target_units"],
        )
        observed = (
            tuple(field["target_axes"]),
            field["location"],
            field["units"]["target"],
        )
        if observed != expected or field.get("staggering", "none") != "none":
            raise ValueError(
                f"units/axes/staggering check failed for {name!r}: "
                f"observed axes/location/units/staggering="
                f"{observed + (field.get('staggering', 'none'),)!r}, "
                f"required={expected + ('none',)!r}"
            )
    return {
        "status": "PASS",
        "source_top_pa": source_top,
        "model_top_pa": model_top,
        "required_field_count": len(mapping["target"]["required_fields"]),
    }


def _check_group_inventory(
    mapping: Mapping[str, object],
    rows: Sequence[Mapping[str, str]],
    label: str,
) -> None:
    vertical_levels = tuple(
        float(value)
        for value in mapping["coordinates"]["vertical"]["levels"]
    )
    for field_name, field in mapping["fields"].items():
        if field.get("derivation") is not None:
            continue
        selectors = field.get("selectors", [])
        matched = [
            row
            for row in rows
            if any(_selector_matches(selector, row) for selector in selectors)
        ]
        axes = tuple(field["source_axes"])
        if "vertical" in axes:
            observed = [float(row["level_value"]) for row in matched]
            missing = [
                level
                for level in vertical_levels
                if not any(
                    math.isclose(level, value, abs_tol=1e-9)
                    for value in observed
                )
            ]
            extra = [
                value
                for value in observed
                if not any(
                    math.isclose(value, level, abs_tol=1e-9)
                    for level in vertical_levels
                )
            ]
            duplicates = sorted(
                {value for value in observed if observed.count(value) > 1}
            )
            if missing or extra or duplicates:
                raise ValueError(
                    f"record-inventory check failed at {label}: field "
                    f"{field_name!r} vertical coverage mismatch; "
                    f"missing={missing}, extra={extra}, "
                    f"duplicates={duplicates}"
                )
        elif "soil" in axes:
            missing = [
                _selector_label(selector)
                for selector in selectors
                if not any(
                    _selector_matches(selector, row) for row in rows
                )
            ]
            duplicates = [
                _selector_label(selector)
                for selector in selectors
                if sum(
                    _selector_matches(selector, row) for row in rows
                )
                > 1
            ]
            if missing or duplicates:
                raise ValueError(
                    f"soil-layer check failed at {label}: field "
                    f"{field_name!r} missing selectors={missing}, "
                    f"duplicate selectors={duplicates}"
                )
        elif len(matched) != 1:
            if not matched:
                detail = "missing selector " + " OR ".join(
                    _selector_label(selector) for selector in selectors
                )
            else:
                detail = (
                    f"matched {len(matched)} records; expected exactly one"
                )
            raise ValueError(
                f"record-inventory check failed at {label}: field "
                f"{field_name!r} {detail}"
            )


def verify_grib2_inputs(
    mapping: Mapping[str, object],
    input_files: Sequence[str | Path],
    *,
    inventory_executable: str | Path,
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Run the fail-closed battery against actual GRIB2 input paths."""

    paths = tuple(Path(path).resolve() for path in input_files)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("adapt input inventory must be non-empty and unique")
    snapshots = tuple(_snapshot_authority(path) for path in paths)
    all_rows: list[dict[str, str]] = []
    for path in paths:
        rows = _grib2_inventory(path, Path(inventory_executable))
        for row in rows:
            copied = dict(row)
            copied["_source"] = str(path)
            all_rows.append(copied)

    selected = [
        row
        for row in all_rows
        if any(
            _selector_matches(selector, row)
            for field in mapping["fields"].values()
            for selector in field.get("selectors", [])
        )
    ]
    if not selected:
        raise ValueError(
            "record-inventory check failed: no GRIB2 records match the "
            "compiled Vtable selector set"
        )
    for row in selected:
        gdt = int(row["gdt"])
        if gdt != 0:
            raise ValueError(
                "grid-family check failed: "
                f"{row['_source']} field {row['index']} uses GDT {gdt}; "
                "generic adapt supports regular latitude/longitude GDT 0 "
                "only. Use the named-adapter path for other grid templates."
            )
        if int(row["scan_mode"], 16) != 0x40:
            raise ValueError(
                "grid-family check failed: "
                f"{row['_source']} field {row['index']} uses scan mode "
                f"{row['scan_mode']}; generic adapt requires GDT 0 scan mode "
                "0x40. Use the named-adapter path for normalization."
            )
    fingerprints = {
        tuple(row[key] for key in _GRID_KEYS) for row in selected
    }
    if len(fingerprints) != 1:
        raise ValueError(
            "grid-family check failed: selected records do not share one "
            "regular latitude/longitude grid"
        )

    by_time: dict[datetime, list[dict[str, str]]] = {}
    for row in selected:
        by_time.setdefault(_valid_time(row), []).append(row)
    for valid_time, rows in by_time.items():
        members = {row["member"] for row in rows}
        if len(members) != 1:
            raise ValueError(
                "record-inventory check failed at "
                f"{valid_time.isoformat()}: selected records mix members "
                f"{sorted(members)}"
            )
        member = next(iter(members))
        _check_group_inventory(
            mapping,
            rows,
            f"{valid_time.isoformat()} member={member}",
        )

    times = sorted(by_time)
    cadence_seconds = None
    if len(times) > 1:
        intervals = {
            int((right - left).total_seconds())
            for left, right in zip(times, times[1:])
        }
        if len(intervals) != 1 or next(iter(intervals)) <= 0:
            raise ValueError(
                f"record-inventory check failed: valid-time cadence is not "
                f"uniform; intervals={sorted(intervals)}"
            )
        cadence_seconds = next(iter(intervals))
        expected = mapping["target"].get("boundary_interval_seconds")
        if cadence_seconds != expected:
            raise ValueError(
                "record-inventory check failed: actual cadence "
                f"{cadence_seconds} s differs from descriptor target "
                f"boundary_interval_seconds={expected}"
            )
    for snapshot in snapshots:
        _require_authority_snapshot(snapshot)
    return {
        "status": "PASS",
        "input_file_count": len(paths),
        "selected_record_count": len(selected),
        "valid_time_count": len(times),
        "first_valid_time": times[0].isoformat(),
        "last_valid_time": times[-1].isoformat(),
        "cadence_seconds": cadence_seconds,
        "grid_template": 0,
        "scan_mode": "0x40",
    }, snapshots


def _validate_candidate_contracts(
    mapping: Mapping[str, object],
    composition: Mapping[str, object],
) -> None:
    with tempfile.TemporaryDirectory(prefix="gpuwm-adapt-contract-") as raw:
        root = Path(raw)
        mapping_path = root / "adapter.mapping.json"
        composition_path = root / "adapter.composition.json"
        mapping_path.write_bytes(_canonical_json(mapping))
        composition_path.write_bytes(_canonical_json(composition))
        if load_mapping(mapping_path) != mapping:
            raise RuntimeError(
                "adapt mapping changed through executable validation"
            )
        if load_composition(composition_path, mapping_path) != composition:
            raise RuntimeError(
                "adapt composition changed through executable validation"
            )


def _input_rows(
    paths: Sequence[Path],
    snapshots: Sequence[object],
) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "bytes": snapshot.size,
            "sha256": snapshot.sha256,
        }
        for path, snapshot in zip(paths, snapshots)
    ]


def _require_compilation_snapshot(
    row: Mapping[str, object],
    snapshot: object,
    label: str,
) -> None:
    if (
        Path(str(row["path"])).resolve() != snapshot.path
        or row["bytes"] != snapshot.size
        or row["sha256"] != snapshot.sha256
    ):
        raise ValueError(
            f"{label} changed between adapt policy/Vtable inspection and "
            "mapping compilation"
        )


def author_adapter(
    *,
    vtable_path: str | Path,
    descriptor_path: str | Path,
    input_files: Sequence[str | Path],
    output_dir: str | Path,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> dict[str, object]:
    """Verify inputs and create the runnable, uncertified authority bundle."""

    descriptor_path = Path(descriptor_path).resolve()
    vtable_path = Path(vtable_path).resolve()
    paths = tuple(Path(path).resolve() for path in input_files)
    policy, descriptor_snapshot = _load_adapt_policy(descriptor_path)
    vtable_snapshot = _stable_file_snapshot(vtable_path)
    mapping, compilation = compile_mapping_descriptor(
        descriptor_path,
        vtable_path=vtable_path,
    )
    _require_compilation_snapshot(
        compilation["descriptor"],
        descriptor_snapshot,
        "descriptor",
    )
    _require_compilation_snapshot(
        compilation["vtable"],
        vtable_snapshot,
        "Vtable",
    )
    static_check = _validate_static_bindings(mapping, policy)
    composition = build_composition(mapping, policy)
    _validate_candidate_contracts(mapping, composition)

    if (grib2_inventory is None) != (grib2_dump is None):
        raise ValueError(
            "--grib2-inventory and --grib2-dump must be supplied together"
        )
    if grib2_inventory is None:
        inventory_path, dump_path = _build_grib2_tools()
    else:
        inventory_path = Path(grib2_inventory).resolve()
        dump_path = Path(grib2_dump).resolve()
    for decoder in (inventory_path, dump_path):
        if not decoder.is_file():
            raise FileNotFoundError(decoder)
    decoder_snapshots = tuple(
        _snapshot_authority(decoder)
        for decoder in (inventory_path, dump_path)
    )

    inventory_check, input_snapshots = verify_grib2_inputs(
        mapping,
        paths,
        inventory_executable=inventory_path,
    )
    # Inventory proves selector completeness cheaply; executing the existing
    # decoder proves those exact selected messages can actually be unpacked,
    # normalized, unit-transformed, axis-bound, and assembled.  A merely
    # parseable Section 5/7 representation must never be called runnable.
    _decode_grib(
        mapping,
        paths,
        grib1_bridge=None,
        grib2_inventory=inventory_path,
        grib2_dump=dump_path,
    )
    for snapshot in input_snapshots:
        _require_authority_snapshot(snapshot)
    for snapshot in decoder_snapshots:
        _require_authority_snapshot(snapshot)
    _require_snapshot(descriptor_snapshot)
    _require_snapshot(vtable_snapshot)

    output_dir = Path(output_dir).resolve()
    outputs = {
        "mapping": output_dir / "adapter.mapping.json",
        "composition": output_dir / "adapter.composition.json",
        "provenance": output_dir / "adapter.provenance.json",
        "input_manifest": output_dir / "adapter.inputs.json",
    }
    collisions = [str(path) for path in outputs.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite adapt output(s) {collisions}"
        )
    authorities = {
        descriptor_path,
        vtable_path,
        *paths,
        inventory_path.resolve(),
        dump_path.resolve(),
    }
    aliases = [
        str(path)
        for path in outputs.values()
        if path.resolve() in authorities
    ]
    if aliases:
        raise ValueError(
            f"adapt output aliases an input/authority path: {aliases}"
        )

    mapping_bytes = _canonical_json(mapping)
    composition_bytes = _canonical_json(composition)
    mapping_sha = hashlib.sha256(mapping_bytes).hexdigest()
    composition_sha = hashlib.sha256(composition_bytes).hexdigest()
    provenance = {
        "schema": ADAPT_PROVENANCE_SCHEMA,
        "status": ADAPTER_STATUS,
        "runnable": True,
        "stock_wrf_certified": False,
        "certification_rule": (
            "unchanged-stock-WRF evidence is a separate exact-authority "
            "gate; this ingest battery does not satisfy it"
        ),
        "mapping": {
            "path": outputs["mapping"].name,
            "bytes": len(mapping_bytes),
            "sha256": mapping_sha,
        },
        "composition": {
            "path": outputs["composition"].name,
            "bytes": len(composition_bytes),
            "sha256": composition_sha,
        },
        "descriptor": compilation["descriptor"],
        "vtable": compilation["vtable"],
        "inputs": _input_rows(paths, input_snapshots),
        "battery": {
            "status": "PASS",
            "record_inventory": inventory_check,
            "units_axes_staggering": {
                "status": "PASS",
                "binding": "descriptor-to-canonical-target-exact",
                "decoder_execution": "PASS",
                "required_field_count": static_check["required_field_count"],
            },
            "soil_layers": {
                "status": "PASS",
                "policy": policy["soil_policy"],
                "source_layer_count": len(
                    composition["soil_layers"]["source_layers"]
                ),
            },
            "grid_family": {
                "status": "PASS",
                "required": "regular_latitude_longitude_gdt_0_scan_0x40",
            },
            "vertical_coverage": static_check,
        },
    }
    provenance_bytes = _canonical_json(provenance)

    published: list[Path] = []
    try:
        for role, contents in (
            ("mapping", mapping_bytes),
            ("composition", composition_bytes),
            ("provenance", provenance_bytes),
        ):
            _write_new(outputs[role], contents)
            published.append(outputs[role])
        manifest_receipt = author_input_manifest(
            outputs["input_manifest"],
            mapping_path=outputs["mapping"],
            composition_path=outputs["composition"],
            primary_files=paths,
            supplement_files={"adapt_in_band_terrain": paths},
            provenance_files={
                "adapt_authority_provenance": outputs["provenance"],
            },
            grib2_inventory=inventory_path,
            grib2_dump=dump_path,
            expected_format=_SUPPORTED_FORMAT,
        )
        published.append(outputs["input_manifest"])
        if load_mapping(outputs["mapping"]) != mapping:
            raise RuntimeError("published adapt mapping failed round trip")
        if load_composition(
            outputs["composition"], outputs["mapping"]
        ) != composition:
            raise RuntimeError(
                "published adapt composition failed round trip"
            )
        for snapshot in input_snapshots:
            _require_authority_snapshot(snapshot)
        for snapshot in decoder_snapshots:
            _require_authority_snapshot(snapshot)
        _require_snapshot(descriptor_snapshot)
        _require_snapshot(vtable_snapshot)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise

    return {
        "status": ADAPTER_STATUS,
        "runnable": True,
        "stock_wrf_certified": False,
        "mapping": {
            "path": str(outputs["mapping"]),
            "sha256": mapping_sha,
        },
        "composition": {
            "path": str(outputs["composition"]),
            "sha256": composition_sha,
        },
        "provenance": {
            "path": str(outputs["provenance"]),
            "sha256": _sha256(outputs["provenance"]),
        },
        "input_manifest": manifest_receipt["manifest"],
        "runtime_bindings": {
            "source": "mapped",
            "source_format": _SUPPORTED_FORMAT,
            "inputs": [str(path) for path in paths],
            "supplements": {
                "adapt_in_band_terrain": [str(path) for path in paths],
            },
            "provenance": {
                "adapt_authority_provenance": str(outputs["provenance"]),
            },
            "decoders": {
                "grib2_inventory": str(inventory_path),
                "grib2_dump": str(dump_path),
            },
        },
        "battery": provenance["battery"],
    }


_CANONICAL_FIELDS = (
    ("air_temperature", "K", ("vertical", "y", "x"), "mass"),
    ("relative_humidity", "%", ("vertical", "y", "x"), "mass"),
    ("eastward_wind", "m s-1", ("vertical", "y", "x"), "mass"),
    ("northward_wind", "m s-1", ("vertical", "y", "x"), "mass"),
    ("geopotential_height", "m", ("vertical", "y", "x"), "mass"),
    ("surface_pressure", "Pa", ("y", "x"), "surface"),
    ("terrain_height", "m", ("y", "x"), "surface"),
    ("skin_temperature", "K", ("y", "x"), "surface"),
    ("air_temperature_2m", "K", ("y", "x"), "surface"),
    ("relative_humidity_2m", "%", ("y", "x"), "surface"),
    ("eastward_wind_10m", "m s-1", ("y", "x"), "surface"),
    ("northward_wind_10m", "m s-1", ("y", "x"), "surface"),
    ("land_fraction", "1", ("y", "x"), "surface"),
    ("soil_temperature", "K", ("soil", "y", "x"), "soil"),
    (
        "volumetric_soil_moisture",
        "m3 m-3",
        ("soil", "y", "x"),
        "soil",
    ),
)


def _canonical_field_names() -> tuple[str, ...]:
    return tuple(name for name, _units, _axes, _location in _CANONICAL_FIELDS)


def _packaged_vtable_hint() -> str:
    """Where the shipped GFS Vtable actually is on THIS install.

    The example lived in `configs/`, which is not a package, so the
    wheel did not carry it and the documented flow named a path only a
    checkout has.  It ships now; printing the resolved path means a pip
    user does not have to know where a wheel puts package data.
    """

    from gpuwm.source_authorities import packaged_gfs_vtable

    try:
        return str(packaged_gfs_vtable())
    except (OSError, RuntimeError) as error:
        return f"(this install cannot resolve it: {error})"


def _skeleton_owner(description: str, names) -> str | None:
    """Which canonical field a Vtable row belongs to: the LONGEST match.

    The prefix rule exists for multi-layer rows -- ``soil_temperature``
    has to collect ``soil_temperature_0_0.1m`` through
    ``soil_temperature_1.0_2.0m``.  Applied without a tie-break it also
    collects ``air_temperature_2m`` under ``air_temperature``, which is
    how the generator came to hand four 3-D fields their surface
    counterpart's selector as well as their own and emit a descriptor
    its own battery refused.

    Longest-match settles it, and settles it by the same rule for every
    field rather than by a list of the four exceptions anyone happened
    to notice: a row that exactly names a canonical field belongs to
    that field, and a row that prefixes two of them belongs to the more
    specific one.
    """

    owner = None
    for name in names:
        if description == name or description.startswith(name + "_"):
            if owner is None or len(name) > len(owner):
                owner = name
    return owner


def _skeleton_reference(rows, field_name: str) -> list[dict[str, object]]:
    names = _canonical_field_names()
    candidates = [
        row
        for row in rows
        if _skeleton_owner(row.description, names) == field_name
    ]
    if not candidates:
        return [{"metgrid_name": "REPLACE_WITH_EXACT_VTABLE_NAME"}]
    result = []
    for row in candidates:
        reference: dict[str, object] = {"metgrid_name": row.metgrid_name}
        if row.grib2_level_type:
            reference["grib2_level_type"] = int(row.grib2_level_type)
        if row.level1 and row.level1 != "*":
            reference["level1"] = float(row.level1)
        if row.level2 and row.level2 != "*":
            reference["level2"] = float(row.level2)
        selector: dict[str, object] = {}
        if row.level1 not in {"", "*"}:
            selector["level_value"] = float(row.level1)
        if selector:
            reference["selector"] = selector
        result.append(reference)
    return result


def descriptor_skeleton(vtable_path: str | Path) -> dict[str, object]:
    """Return a review-required descriptor scaffold populated from a Vtable."""

    rows = parse_wps_vtable(vtable_path)
    fields = {}
    for name, units, axes, location in _CANONICAL_FIELDS:
        fields[name] = {
            "vtable_selectors": _skeleton_reference(rows, name),
            "units": {"source": units, "target": units},
            "source_axes": list(axes),
            "target_axes": list(axes),
            "location": location,
            "staggering": "none",
            "missing": {
                "kind": "preserve_mask"
                if location == "soil"
                else "reject"
            },
        }
    fields["air_pressure"] = {
        "selectors": [],
        "derivation": "pressure-from-coordinate",
        "units": {"source": "Pa", "target": "Pa"},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass",
        "staggering": "none",
        "missing": {"kind": "reject"},
    }
    fields["specific_humidity"] = {
        "selectors": [],
        "derivation": "humidity-from-rh",
        "units": {"source": "kg kg-1", "target": "kg kg-1"},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass",
        "staggering": "none",
        "missing": {"kind": "reject"},
    }
    fields["specific_humidity_2m"] = {
        "selectors": [],
        "derivation": "humidity-2m-from-rh",
        "units": {"source": "kg kg-1", "target": "kg kg-1"},
        "source_axes": ["y", "x"],
        "target_axes": ["y", "x"],
        "location": "surface",
        "staggering": "none",
        "missing": {"kind": "reject"},
    }
    required = [
        {
            "name": name,
            "axes": field["target_axes"],
            "location": field["location"],
            "target_units": field["units"]["target"],
        }
        for name, field in fields.items()
        if name not in {"relative_humidity", "relative_humidity_2m"}
    ]
    zero_fields = (
        "cloud_water_mixing_ratio",
        "rain_water_mixing_ratio",
        "cloud_ice_mixing_ratio",
        "snow_mixing_ratio",
        "graupel_or_hail_mixing_ratio",
        "vertical_velocity",
    )
    return {
        "schema": DESCRIPTOR_SCHEMA,
        "name": "REPLACE_WITH_ADAPTER_NAME",
        "format": _SUPPORTED_FORMAT,
        "adapt": {
            "model_top_pa": "REPLACE_WITH_MODEL_TOP_PA",
            "soil_policy": {"kind": "identity_complete_layers"},
        },
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {
                "kind": "pressure",
                "units": "Pa",
                "positive": "down",
                "levels": [],
            },
            "time": {"kind": "embedded_metadata"},
        },
        "fields": fields,
        "derivations": [
            {
                "name": "pressure-from-coordinate",
                "operation": "pressure_from_vertical_coordinate",
            },
            {
                "name": "humidity-from-rh",
                "operation": "specific_humidity_from_rh",
                "relative_humidity": "relative_humidity",
                "temperature": "air_temperature",
                "pressure": "air_pressure",
            },
            {
                "name": "humidity-2m-from-rh",
                "operation": "specific_humidity_from_rh",
                "relative_humidity": "relative_humidity_2m",
                "temperature": "air_temperature_2m",
                "pressure": "surface_pressure",
            },
        ],
        "target": {
            "name": "REPLACE_WITH_TARGET_NAME",
            "physics_suite": "source-independent",
            "max_dom": 1,
            "require_lateral_boundaries": True,
            "target_vertical_levels": 49,
            "soil_layer_count": 4,
            "boundary_interval_seconds": 10800,
            "required_fields": required,
            "pressure_requirement": "air_pressure",
            "policy_controlled_fields": list(zero_fields),
            "initialization_policies": {
                name: "explicit_zero_with_adapter_validation"
                for name in zero_fields
            },
        },
    }


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "adapt",
        help="verify actual GRIB2 files and author a runnable mapped adapter",
        description=(
            "Verify actual GRIB2 files against a descriptor you write, and "
            "author a runnable mapped adapter. A successful adaptation "
            "establishes that the emitted files implement your descriptor "
            "exactly and that your GRIB files satisfy it. It does not "
            "establish that your descriptor is a correct physical "
            "interpretation of those files: units, absolute geolocation, "
            "cell registration, vertical sufficiency, intended time "
            "semantics, land-mask polarity, and soil depth labels are "
            "trusted from your declaration."
        ),
        epilog=(
            "Before you run the adapter: read "
            "docs/adapt-validation-contract.md, which lists every input "
            "dimension in two columns -- validated for you, and trusted "
            "from your declaration -- with a self-check you can run for "
            "each trusted row. docs/arbitrary-verified-adapters.md covers "
            "the descriptor schema, the battery, and refusal examples. "
            "Both are in the repository, under "
            "https://github.com/FahrenheitResearch/arwen/tree/main/docs, "
            "if this is a wheel install without a checkout."
        ),
    )
    parser.add_argument(
        "--vtable",
        required=True,
        type=Path,
        metavar="VTABLE",
        help="11-column WPS Vtable selector authority. Required, and "
             "never defaulted: this command adapts arbitrary sources, and "
             "quietly reaching for a GFS Vtable would mis-map every other "
             "product. A worked GFS example installs with the package -- "
             f"{_packaged_vtable_hint()}",
    )
    parser.add_argument(
        "--skeleton",
        type=Path,
        metavar="JSON",
        help="create a review-required descriptor scaffold and stop",
    )
    parser.add_argument(
        "--descriptor",
        type=Path,
        metavar="JSON",
        help="completed rw-wps.descriptor.v1 document",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        metavar="GRIB2",
        help="actual GRIB2 input (repeat for every file in the series)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="directory for create-only adapter authorities and manifest",
    )
    parser.add_argument(
        "--grib2-inventory",
        type=Path,
        metavar="EXE",
        help="expert override paired with --grib2-dump",
    )
    parser.add_argument(
        "--grib2-dump",
        type=Path,
        metavar="EXE",
        help="expert override paired with --grib2-inventory",
    )
    parser.set_defaults(func=_from_cli)


def _from_cli(args) -> int:
    if args.skeleton is not None:
        incompatible = {
            "--descriptor": args.descriptor,
            "--input": args.input,
            "--output-dir": args.output_dir,
            "--grib2-inventory": args.grib2_inventory,
            "--grib2-dump": args.grib2_dump,
        }
        used = [name for name, value in incompatible.items() if value]
        if used:
            raise ValueError(
                f"--skeleton cannot be combined with {', '.join(used)}"
            )
        payload = descriptor_skeleton(args.vtable)
        _write_new(args.skeleton, _canonical_json(payload))
        print(
            f"adapt skeleton: wrote {args.skeleton}; replace every "
            "REPLACE_WITH_* value, declare the full pressure levels, and "
            "review every selector before authoring"
        )
        print(
            "What the battery will check for you, and what it will trust "
            "from your declaration -- units, geolocation, cell "
            "registration, level sufficiency, land-mask polarity -- is "
            "docs/adapt-validation-contract.md. Read it while you fill "
            "this in, not after.",
            file=sys.stderr,
        )
        return 0
    missing = [
        name
        for name, value in (
            ("--descriptor", args.descriptor),
            ("--input", args.input),
            ("--output-dir", args.output_dir),
        )
        if not value
    ]
    if missing:
        raise ValueError("adapter authoring requires " + ", ".join(missing))
    result = author_adapter(
        vtable_path=args.vtable,
        descriptor_path=args.descriptor,
        input_files=args.input,
        output_dir=args.output_dir,
        grib2_inventory=args.grib2_inventory,
        grib2_dump=args.grib2_dump,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "RUNNABLE, NOT stock-WRF certified: unchanged-stock-WRF evidence "
        "is a separate exact-authority gate.",
        file=sys.stderr,
    )
    print(
        "The battery proved these files implement your descriptor and that "
        "your GRIB files satisfy it. It did not check that the descriptor "
        "is a correct physical reading of them: units, absolute "
        "geolocation, cell registration, level sufficiency, intended time "
        "semantics, land-mask polarity, and soil depth labels are trusted "
        "from your declaration. Run the self-checks in "
        "docs/adapt-validation-contract.md before you run the adapter.",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "ADAPTER_STATUS",
    "ADAPT_PROVENANCE_SCHEMA",
    "author_adapter",
    "build_composition",
    "descriptor_skeleton",
    "register_cli",
    "verify_grib2_inputs",
]
