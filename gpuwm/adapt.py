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

from gpuwm.experiment import did_you_mean
from gpuwm.explain import layered, warn
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
    _decode_netcdf,
    _grib2_inventory,
    _match_nc_variables,
    _matching_nc_variables,
    _nc_selector_text,
    _nc_vocabulary,
    _selector_names,
    NC_EVIDENCE_NAME,
    NC_EVIDENCE_STANDARD_NAME,
    _require_authority_snapshot,
    _sha256,
    _snapshot_authority,
    load_mapping,
)


ADAPT_PROVENANCE_SCHEMA = "gpuwm-adapt-provenance-v1"
ADAPTER_STATUS = "runnable_mapping_not_stock_wrf_certified"
_SUPPORTED_FORMAT = "grib2"
_SUPPORTED_FORMATS = ("grib2", "netcdf")
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
        # This carried the same sentence the [shared]/[[domain]] loader
        # carried before a4417efb -- "the required-key check below still
        # catches every typo of a key that matters" -- and it is false
        # here for the same structural reason: `allowed - required` is
        # optional by construction, so no check below ever looks.
        # `descriptor.adapt.soil_policy.target_layers_m` is that gap: a
        # misspelling of it under `kind = "identity_complete_layers"`
        # warned once and adapted the source inventory unchanged, which
        # is not the layer set the descriptor asked for.
        named = ", ".join(
            f"{key!r}{did_you_mean(key, allowed)}" for key in unknown)
        raise ValueError(layered(
            f"{label} does not have a key {named}; no key is ignored, "
            "because a dropped key runs a default under the name of "
            "your value.",
            f"Known {label} keys: {sorted(allowed)}."))
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


#: Every value :func:`descriptor_skeleton` leaves for the author to
#: replace carries this prefix.  One string, used by the generator that
#: writes them and by the gate that refuses them, so the two can never
#: drift apart.
SCAFFOLD_MARKER_PREFIX = "REPLACE_WITH_"


def scaffold_gaps(descriptor: object) -> list[str]:
    """Dotted paths in ``descriptor`` still holding ``--skeleton`` scaffold.

    The scaffold's placeholders are deliberate -- the whole point of
    ``--skeleton`` is that its output "cannot accidentally pass as a
    completed adapter" (docs/arbitrary-verified-adapters.md) -- but a
    reader who ran the two documented commands in order met that design
    as ``descriptor.adapt.model_top_pa must be a positive finite
    number``: a type error about one field, from a validator that had
    no idea it was looking at its own generator's output, with three
    more unreplaced values waiting behind it one run at a time.

    This function is what makes the scaffold round-trip *checkable*:
    the generator's contract is that everything its own battery objects
    to appears in this list, so the refusal can name the whole job at
    once and a test can hold the two surfaces to the same set.
    Two markers -- ``name`` and ``target.name`` -- pass every other
    validator unchanged, which is worse than failing: an adapter
    published as ``REPLACE_WITH_ADAPTER_NAME`` is a real bundle with a
    placeholder identity.
    """

    gaps: list[str] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, str):
            if value.startswith(SCAFFOLD_MARKER_PREFIX):
                gaps.append(path)
        elif isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(descriptor, "")
    # The empty pressure inventory is scaffolding too: the skeleton
    # writes `levels: []` on purpose, load_mapping accepts it, and the
    # refusal arrives much later out of the vertical-coverage check.
    if isinstance(descriptor, dict):
        vertical = descriptor.get("coordinates")
        vertical = (vertical.get("vertical")
                    if isinstance(vertical, dict) else None)
        if isinstance(vertical, dict) and vertical.get("levels") == []:
            gaps.append("coordinates.vertical.levels")
    return sorted(gaps)


def _named_gaps(gaps: list[str], limit: int = 4) -> str:
    """``a, b, c and 16 more`` -- a sentence, not a wall of dotted paths.

    A Vtable that matches few canonical fields leaves a marker per
    unmatched field, so the full list runs to twenty entries.  The
    skeleton's own success message prints all of them, one per line,
    because there it IS the to-do list; a refusal states the size of
    the job and enough of it to recognise.
    """

    if len(gaps) <= limit + 1:
        # "and 1 more" is a worse sentence than the item it replaced.
        return ", ".join(gaps)
    return f"{', '.join(gaps[:limit])} and {len(gaps) - limit} more"


def _refuse_scaffold(raw: object, label: str) -> None:
    """Refuse an unfilled scaffold in one sentence that sizes the job."""

    gaps = scaffold_gaps(raw)
    if not gaps:
        return
    raise ValueError(
        f"{label} is still the `--skeleton` scaffold: {len(gaps)} value(s) "
        f"are unfilled ({_named_gaps(gaps)}) -- replace every "
        "REPLACE_WITH_* value and declare the complete pressure inventory "
        "in coordinates.vertical.levels, then author again "
        "(docs/arbitrary-verified-adapters.md)"
    )


def _descriptor_format(descriptor_path: Path) -> str:
    """The descriptor's declared source format, read before compilation.

    The front door has to know whether a Vtable and the GRIB2 decoders are
    required *before* it asks for them, and the descriptor is the only place
    that says so.  A format the adapter cannot author is refused by name.
    """

    raw = _strict_json(
        _stable_file_snapshot(descriptor_path).data,
        f"descriptor {descriptor_path}",
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("format"), str):
        raise ValueError(
            f"descriptor {descriptor_path} must declare a string 'format'; "
            f"supported: {list(_SUPPORTED_FORMATS)}"
        )
    source_format = raw["format"]
    if source_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"gpuwm adapt cannot author {source_format!r} input; supported "
            f"formats are {list(_SUPPORTED_FORMATS)}"
        )
    return source_format


def _load_adapt_policy(descriptor_path: Path) -> tuple[dict[str, object], object]:
    snapshot = _stable_file_snapshot(descriptor_path)
    raw = _strict_json(snapshot.data, f"descriptor {snapshot.path}")
    if not isinstance(raw, dict) or raw.get("schema") != DESCRIPTOR_SCHEMA:
        raise ValueError(
            f"adapt descriptor must use schema {DESCRIPTOR_SCHEMA!r}"
        )
    # Before the field-by-field type checks, because "you have not
    # filled the template in" is the true answer and a type error about
    # one of its markers is a misleading one.
    _refuse_scaffold(raw, f"descriptor {snapshot.path}")
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
        allowed={"kind", "target_layers_m", "source_layers_m"},
        required={"kind"},
    )
    # GRIB selectors carry their own layer bounds, so a second declaration
    # would be a second authority.  NetCDF selectors are variable names and
    # carry no depth at all, so the depths must be declared or they would be
    # guessed from the order -- which is exactly the kind of silent
    # assumption this contract exists to refuse.
    source_format = raw.get("format")
    if source_format == "netcdf":
        if "source_layers_m" not in soil:
            raise ValueError(
                "descriptor.adapt.soil_policy.source_layers_m is required for "
                "NetCDF: variable-name selectors carry no layer depth, and "
                "soil geometry is never inferred from selector order"
            )
    elif "source_layers_m" in soil:
        raise ValueError(
            "descriptor.adapt.soil_policy.source_layers_m is refused for "
            f"{source_format!r}: the GRIB selectors already bind both layer "
            "surfaces, and two authorities for one geometry is one too many"
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
    declared: object = None,
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
    if declared is not None:
        layers = _layers(declared, "descriptor.adapt.soil_policy.source_layers_m")
        if len(layers) != len(selector_lists[0]):
            raise ValueError(
                "soil-layer check failed: source_layers_m declares "
                f"{len(layers)} layer(s) but the soil fields carry "
                f"{len(selector_lists[0])} selector(s); one depth per "
                "selector, shallow to deep"
            )
        rows = [
            {
                "top": layer["top"],
                "bottom": layer["bottom"],
                # The selector bound to this depth, position by position.
                # The composition re-checks this pairing, which is what
                # stops a reordered selector list from silently relabelling
                # the soil column.
                "selectors": {
                    name: copy.deepcopy(selectors[index])
                    for name, selectors in zip(names, selector_lists)
                },
            }
            for index, layer in enumerate(layers)
        ]
        _require_contiguous_layers(rows, "source soil inventory")
        return rows
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
    # NOTE: the source count is deliberately NOT compared against
    # target.soil_layer_count here.  conservative_layer_means exists
    # exactly for a source column with a different layer count than the
    # target; build_composition checks the FINAL target count for both
    # policies, which covers the identity case this check used to.
    return source_layers


def build_composition(
    mapping: Mapping[str, object],
    adapt_policy: Mapping[str, object],
) -> dict[str, object]:
    """Build the executable v2 composition from explicit descriptor policy."""

    soil_policy = adapt_policy["soil_policy"]
    source_layers = _soil_source_layers(
        mapping, soil_policy.get("source_layers_m")
    )
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
                "format": str(mapping["format"]),
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
    if mapping["format"] not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"gpuwm adapt supports {list(_SUPPORTED_FORMATS)} descriptors; "
            f"got {mapping['format']!r}."
        )
    vertical = mapping["coordinates"]["vertical"]
    levels = vertical.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError(
            "vertical-coverage check failed: descriptor must declare the "
            "complete pressure coordinate in coordinates.vertical.levels"
        )
    # GRIB level values are Pa by definition; a NetCDF vertical coordinate
    # carries whatever unit the file declares, and comparing hPa against a
    # model top in Pa silently passes a source that does not reach it.
    units = str(vertical["units"])
    scale = {"Pa": 1.0, "hPa": 100.0, "millibar": 100.0, "mb": 100.0}.get(units)
    if scale is None:
        raise ValueError(
            "vertical-coverage check failed: cannot compare a model top in Pa "
            f"against a vertical coordinate in {units!r}; declare the "
            "vertical coordinate in 'Pa' or 'hPa'"
        )
    source_top = min(float(level) for level in levels) * scale
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
            # The observed uniform cadence is the authority; the
            # descriptor's declared number is stale metadata.  The
            # returned receipt carries the observed value either way.
            warn(f"GRIB2 inventory cadence is {cadence_seconds} s where "
                 f"the descriptor declares boundary_interval_seconds="
                 f"{expected}; the observed cadence is what runs")
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


def verify_netcdf_inputs(
    mapping: Mapping[str, object],
    input_files: Sequence[str | Path],
) -> tuple[dict[str, object], tuple[object, ...]]:
    """Run the fail-closed battery against actual NetCDF input paths.

    The failure this is built against is the silent zero: a descriptor whose
    selectors match nothing, decoded without complaint into an empty result
    that every downstream count reports as success.  Every declared selector
    must resolve, and a selector that resolves nothing is refused with both
    vocabularies printed -- what the descriptor asked for, and what the file
    actually contains.
    """

    from gpuwm import netcdf_bridge

    paths = tuple(Path(path).resolve() for path in input_files)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("adapt input inventory must be non-empty and unique")
    snapshots = tuple(_snapshot_authority(path) for path in paths)

    fields = mapping["fields"]
    wanted: list[tuple[str, int, Mapping[str, object]]] = []
    for field_name, field in fields.items():
        for index, selector in enumerate(field.get("selectors", [])):
            wanted.append((field_name, index, selector))
    if not wanted:
        raise ValueError(
            "record-inventory check failed: the compiled mapping declares no "
            "NetCDF variable selectors at all"
        )

    resolved: dict[tuple[str, int], str] = {}
    evidence: dict[tuple[str, int], str] = {}
    vocabulary = ""
    variable_names: set[str] = set()
    for path in paths:
        with netcdf_bridge.open_dataset(path) as dataset:
            vocabulary = _nc_vocabulary(dataset)
            variable_names.update(dataset.variables)
            for field_name, index, selector in wanted:
                matches = _match_nc_variables(dataset, selector)
                if len(matches) > 1:
                    # Never a silent pick between two candidates: both are
                    # named so the descriptor can be disambiguated.
                    raise ValueError(
                        f"record-inventory check failed: {field_name} "
                        f"selector[{index}] ({_nc_selector_text(selector)}) "
                        f"resolves {len(matches)} variables in {path} "
                        f"({', '.join(sorted(str(v.name) for v, _ in matches))}); "
                        "rw-wps.mapping.v1 requires exactly one"
                    )
                if matches:
                    variable, how = matches[0]
                    resolved[(field_name, index)] = str(variable.name)
                    evidence[(field_name, index)] = how

    unresolved = sorted(
        f"{field_name}.selectors[{index}] ({_nc_selector_text(selector)})"
        for field_name, index, selector in wanted
        if (field_name, index) not in resolved
    )
    if unresolved:
        # Naming only one vocabulary is what makes a mapping miss read as a
        # data problem.  Print both so the mismatch is obvious on sight.
        raise ValueError(
            "record-inventory check failed: "
            f"{len(unresolved)} of {len(wanted)} declared NetCDF selector(s) "
            f"matched no variable in any input file.\n"
            "  descriptor asked for: " + "; ".join(unresolved) + "\n"
            "  files actually contain: " + vocabulary
        )
    if not resolved:
        raise ValueError(
            "record-inventory check failed: no NetCDF variables match the "
            f"compiled selector set. files contain: {vocabulary}"
        )

    for snapshot in snapshots:
        _require_authority_snapshot(snapshot)
    # Counts and evidence, never a boolean: a mapping that matched nothing
    # while reporting success is the failure mode this program keeps hitting.
    by_standard_name = sorted(
        f"{field_name}.selectors[{index}] -> {resolved[(field_name, index)]}"
        for (field_name, index), how in evidence.items()
        if how == NC_EVIDENCE_STANDARD_NAME
        and _selector_names(
            next(
                selector for name, position, selector in wanted
                if name == field_name and position == index
            )
        )
    )
    receipt = {
        "status": "PASS",
        "input_file_count": len(paths),
        "declared_selector_count": len(wanted),
        "resolved_selector_count": len(resolved),
        "resolved_by_name_count": sum(
            1 for how in evidence.values() if how == NC_EVIDENCE_NAME
        ),
        "resolved_by_standard_name_count": sum(
            1 for how in evidence.values()
            if how == NC_EVIDENCE_STANDARD_NAME
        ),
        "resolved_variables": sorted(set(resolved.values())),
        "source_variable_count": len(variable_names),
    }
    if by_standard_name:
        receipt["descriptor_name_drift"] = by_standard_name
        warn(
            f"{len(by_standard_name)} selector(s) resolved by CF "
            "standard_name because the name the descriptor gives no longer "
            f"exists in these files: {'; '.join(by_standard_name)}",
            why="The files are self-describing and were read correctly. "
                "Add the current spelling to those selectors' name lists so "
                "the match stops depending on the standard name alone.",
        )
    return receipt, snapshots


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
    vtable_path: str | Path | None = None,
    descriptor_path: str | Path,
    input_files: Sequence[str | Path],
    output_dir: str | Path,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> dict[str, object]:
    """Verify inputs and create the runnable, uncertified authority bundle.

    ``vtable_path`` is the GRIB selector authority and must be omitted for
    NetCDF descriptors, whose selectors name variables directly and have no
    Vtable to import from.
    """

    descriptor_path = Path(descriptor_path).resolve()
    paths = tuple(Path(path).resolve() for path in input_files)
    policy, descriptor_snapshot = _load_adapt_policy(descriptor_path)
    source_format = _descriptor_format(descriptor_path)
    if source_format == "netcdf":
        if vtable_path is not None:
            raise ValueError(
                "NetCDF descriptors have no WPS Vtable: their selectors name "
                "variables directly. Drop --vtable."
            )
        vtable_snapshot = None
        resolved_vtable = None
    else:
        if vtable_path is None:
            raise ValueError(
                f"{source_format} descriptors require --vtable, the selector "
                "authority"
            )
        resolved_vtable = Path(vtable_path).resolve()
        vtable_snapshot = _stable_file_snapshot(resolved_vtable)
    mapping, compilation = compile_mapping_descriptor(
        descriptor_path,
        vtable_path=resolved_vtable,
    )
    _require_compilation_snapshot(
        compilation["descriptor"],
        descriptor_snapshot,
        "descriptor",
    )
    if vtable_snapshot is not None:
        _require_compilation_snapshot(
            compilation["vtable"],
            vtable_snapshot,
            "Vtable",
        )
    static_check = _validate_static_bindings(mapping, policy)
    composition = build_composition(mapping, policy)
    _validate_candidate_contracts(mapping, composition)

    if source_format == "netcdf":
        if grib2_inventory is not None or grib2_dump is not None:
            raise ValueError(
                "NetCDF input does not use the GRIB2 decoders; drop "
                "--grib2-inventory/--grib2-dump"
            )
        inventory_path = dump_path = None
        decoder_snapshots: tuple[object, ...] = ()
        inventory_check, input_snapshots = verify_netcdf_inputs(mapping, paths)
        # Inventory proves selector completeness cheaply; executing the
        # existing decoder proves those exact variables can actually be read,
        # georeferenced, unit-transformed, axis-bound, and assembled.
        _decode_netcdf(mapping, paths)
    else:
        if (grib2_inventory is None) != (grib2_dump is None):
            raise ValueError(
                "--grib2-inventory and --grib2-dump must be supplied together"
            )
        if grib2_inventory is None:
            try:
                inventory_path, dump_path = _build_grib2_tools()
            except FileNotFoundError as error:
                # ValueError so the CLI boundary prints a refusal, not a
                # traceback -- the same conversion the explicit-path branch
                # below already makes, applied to the branch a wheel install
                # actually takes.  Omitting the expert overrides is the
                # DOCUMENTED default, and until the resolver ladder landed
                # this branch shelled cargo into a directory a wheel does
                # not have and ended in a raw NotADirectoryError.
                raise ValueError(str(error)) from error
        else:
            inventory_path = Path(grib2_inventory).resolve()
            dump_path = Path(grib2_dump).resolve()
        for decoder in (inventory_path, dump_path):
            if not decoder.is_file():
                # ValueError so the CLI boundary prints a refusal, not a
                # traceback with a bare path.
                raise ValueError(
                    f"GRIB2 decoder {decoder} does not exist; build the "
                    "bridges (gpuwm setup, or gpuwm doctor --explain for "
                    "the checkout route)")
        decoder_snapshots = tuple(
            _snapshot_authority(decoder)
            for decoder in (inventory_path, dump_path)
        )

        inventory_check, input_snapshots = verify_grib2_inputs(
            mapping,
            paths,
            inventory_executable=inventory_path,
        )
        # Inventory proves selector completeness cheaply; executing the
        # existing decoder proves those exact selected messages can actually
        # be unpacked, normalized, unit-transformed, axis-bound, and
        # assembled.  A merely parseable Section 5/7 representation must
        # never be called runnable.
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
    if vtable_snapshot is not None:
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
        # ValueError so the CLI boundary prints a refusal, not a
        # traceback.  Create-only stays create-only.
        raise ValueError(
            f"refusing to overwrite adapt output(s) {collisions}; "
            "author into a new --output-dir"
        )
    authorities = {
        descriptor_path,
        *paths,
        *(() if resolved_vtable is None else (resolved_vtable,)),
        *(() if inventory_path is None else (inventory_path.resolve(),)),
        *(() if dump_path is None else (dump_path.resolve(),)),
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
        "source_format": source_format,
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
                "required": (
                    "regular_latitude_longitude_cf_geographic_coordinates"
                    if source_format == "netcdf"
                    else "regular_latitude_longitude_gdt_0_scan_0x40"
                ),
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
            expected_format=source_format,
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
        if vtable_snapshot is not None:
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
            "source_format": source_format,
            "inputs": [str(path) for path in paths],
            "supplements": {
                "adapt_in_band_terrain": [str(path) for path in paths],
            },
            "provenance": {
                "adapt_authority_provenance": str(outputs["provenance"]),
            },
            "decoders": {} if source_format == "netcdf" else {
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
        help="verify your own GRIB2 or NetCDF files and author a runnable "
             "mapped adapter",
        description=(
            "Verify your own GRIB2 or NetCDF files against a descriptor you "
            "write, and author a runnable mapped adapter. This is the "
            "arbitrary-input front door: no source name is blessed in "
            "advance, and no code change is needed for a new dataset. GRIB "
            "descriptors import their selectors from an 11-column WPS "
            "Vtable (--vtable); NetCDF descriptors name CF variables "
            "directly and take no Vtable. A successful adaptation "
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
        type=Path,
        metavar="VTABLE",
        help="11-column WPS Vtable selector authority. Required for GRIB "
             "descriptors, and never defaulted: this command adapts "
             "arbitrary sources, and quietly reaching for a GFS Vtable "
             "would mis-map every other product. Must be omitted for "
             "NetCDF descriptors, whose selectors name CF variables "
             "directly. A worked GFS example installs with the package -- "
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
        metavar="FILE",
        help="your own GRIB2 or NetCDF input file, matching the "
             "descriptor's declared format (repeat for every file in the "
             "series)",
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
        if args.vtable is None:
            raise ValueError(
                "--skeleton scaffolds a GRIB descriptor from a WPS Vtable "
                "and requires --vtable. NetCDF descriptors have no Vtable: "
                "start from the worked example named in "
                "docs/intermediate-format.md."
            )
        incompatible = {
            "--descriptor": args.descriptor,
            "--input": args.input,
            "--output-dir": args.output_dir,
            "--grib2-inventory": args.grib2_inventory,
            "--grib2-dump": args.grib2_dump,
        }
        used = [name for name, value in incompatible.items() if value]
        if used:
            warn(f"--skeleton writes a template and ignores "
                 f"{', '.join(used)}; authoring is the next command, "
                 "after the skeleton is filled in")
        payload = descriptor_skeleton(args.vtable)
        _write_new(args.skeleton, _canonical_json(payload))
        # The scaffold's gaps, named here by the same function that
        # refuses them at authoring time -- so the list a reader is
        # handed is exactly the list the next command will hold them
        # to, rather than one type error per run.
        gaps = scaffold_gaps(payload)
        print(
            f"adapt skeleton: wrote {args.skeleton}; {len(gaps)} value(s) "
            "need you before --descriptor will author it:"
        )
        for gap in gaps:
            print(f"  {gap}")
        warn("the battery trusts your declaration for units, "
             "geolocation, cell registration, level sufficiency and "
             "land-mask polarity",
             why="What the battery checks for you, and what it trusts "
                 "from your declaration, is "
                 "docs/adapt-validation-contract.md.  Read it while you "
                 "fill this in, not after.")
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
    warn("adapter is RUNNABLE, not stock-WRF certified; the descriptor's "
         "physical reading (units, geolocation, levels, land-mask "
         "polarity) is trusted from your declaration",
         why="The battery proved these files implement your descriptor "
             "and that your GRIB files satisfy it.  It did not check "
             "that the descriptor is a correct physical reading of "
             "them: units, absolute geolocation, cell registration, "
             "level sufficiency, intended time semantics, land-mask "
             "polarity, and soil depth labels are trusted from your "
             "declaration.  Run the self-checks in "
             "docs/adapt-validation-contract.md before you run the "
             "adapter.  Unchanged-stock-WRF evidence is a separate "
             "exact-authority gate.")
    return 0


__all__ = [
    "ADAPTER_STATUS",
    "ADAPT_PROVENANCE_SCHEMA",
    "SCAFFOLD_MARKER_PREFIX",
    "author_adapter",
    "build_composition",
    "descriptor_skeleton",
    "register_cli",
    "scaffold_gaps",
    "verify_grib2_inputs",
]
