"""Declarative soil-depth and remapping contract for mapped RW-WPS sources.

The mapped-source path must not infer soil meaning from a product name or an
array alias.  This module validates the complete source geometry, target Noah
geometry, numerical remap, and land/ocean missing-data policy used by both
composition validation and the WRF-real initializer.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from gpuwm.core.noah import NUM_SOIL_LAYERS as _NOAH_NUM_SOIL_LAYERS


MAPPED_SOIL_TEMPERATURE = "RW_SOIL_TEMPERATURE"
MAPPED_SOIL_MOISTURE = "RW_SOIL_MOISTURE"

#: The four-layer Noah/Noah-MP soil target this contract remaps INTO.  It is
#: deliberately a single hardcoded geometry rather than a per-scheme table:
#: RUC's nine levels are a different discretization with a different
#: value-location convention, and mapping a source profile onto them requires
#: RUC's own level table and remap policy, not a widened bound check here.
#: A RUC ingest lane must add its target explicitly and prove its remap
#: against the WRF-real initialization -- fabricating one by relaxing this
#: comparison would silently produce a plausible, unvalidated soil column.
NOAH_LAYER_BOUNDS_M = (
    (0.0, 0.1),
    (0.1, 0.4),
    (0.4, 1.0),
    (1.0, 2.0),
)

#: The count this contract targets, taken from the scheme that owns it rather
#: than counted into a message as the word "four".  The ingest layer holds no
#: soil-layer constant of its own: it holds one geometry, Noah's, and says
#: whose it is.  A drift between this table and Noah's own
#: ``NUM_SOIL_LAYERS`` -- the reviewable failure mode, since these bounds are
#: the accumulation of ``init_soil_depth_2``'s dzs -- is refused at import.
NOAH_TARGET_SOIL_LAYERS = len(NOAH_LAYER_BOUNDS_M)
if NOAH_TARGET_SOIL_LAYERS != _NOAH_NUM_SOIL_LAYERS:
    raise AssertionError(
        f"NOAH_LAYER_BOUNDS_M has {NOAH_TARGET_SOIL_LAYERS} layers but "
        f"gpuwm.core.noah.NUM_SOIL_LAYERS is {_NOAH_NUM_SOIL_LAYERS}; the "
        "mapped soil target and the scheme's own geometry have drifted")

_LINEAR_REMAP = "linear_point_samples"
_CONSERVATIVE_REMAP = "conservative_layer_means"
_CANONICAL_SOIL_FIELDS = (
    "soil_temperature",
    "volumetric_soil_moisture",
)


def _object(
    value: object,
    label: str,
    *,
    allowed: set[str],
    required: set[str],
) -> dict[str, object]:
    # Runtime composition bundles deliberately freeze their validated JSON
    # objects with ``MappingProxyType``.  Accept any Mapping here and make a
    # plain local copy before applying the same closed-key validation.  The
    # previous concrete-dict check meant a contract could pass composition
    # loading and then fail solely because the immutable runtime copy reached
    # Noah preprocessing.
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result = dict(value)
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"{label} is missing required key(s): {missing}")
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _selector(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be one non-empty direct selector")
    source_format = value.get("format")
    allowed = {
        "grib1": {
            "format", "parameter", "table_version", "center", "level_type",
            "level_value",
        },
        "grib2": {
            "format", "discipline", "category", "parameter", "level_type",
            "level_value", "second_level_type", "second_level_value", "member",
            "center", "subcenter", "master_table_version",
            "local_table_version",
        },
        "netcdf": {"format", "name", "standard_name"},
    }
    if source_format not in allowed:
        raise ValueError(f"{label} has an unsupported selector format")
    unknown = sorted(set(value) - allowed[source_format])
    if unknown:
        raise ValueError(
            f"{label} has keys incompatible with {source_format}: {unknown}"
        )
    required = {
        "grib1": {"parameter"},
        "grib2": {"discipline", "category", "parameter"},
        "netcdf": set(),
    }[source_format]
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing selector key(s): {missing}")
    if any(value.get(key) is None for key in required):
        raise ValueError(f"{label} required selector identifiers must not be null")
    if source_format == "netcdf":
        names = tuple(value.get(key) for key in ("name", "standard_name"))
        if not any(isinstance(name, str) and name for name in names):
            raise ValueError(f"{label} needs name and/or standard_name")
        if any(name is not None and (not isinstance(name, str) or not name)
               for name in names):
            raise ValueError(f"{label} NetCDF names must be non-empty strings")
        return value
    integer_keys = (
        "parameter", "table_version", "center", "level_type",
        "second_level_type", "discipline", "category", "member",
        "subcenter", "master_table_version", "local_table_version",
    )
    for key in integer_keys:
        item = value.get(key)
        if item is None:
            continue
        maximum = (
            65535
            if source_format == "grib2" and key in {"center", "subcenter"}
            else 255
        )
        if isinstance(item, bool) or not isinstance(item, int) \
                or not 0 <= item <= maximum:
            raise ValueError(
                f"{label}.{key} must be an integer in [0, {maximum}]"
            )
    for key in ("level_value", "second_level_value"):
        if value.get(key) is not None:
            _number(value[key], f"{label}.{key}")
    second_keys = {
        key for key in ("second_level_type", "second_level_value")
        if value.get(key) is not None
    }
    if source_format != "grib2" and second_keys:
        raise ValueError(f"{label} second fixed surfaces require GRIB2")
    if second_keys and second_keys != {"second_level_type", "second_level_value"}:
        raise ValueError(
            f"{label} second_level_type and second_level_value are an atomic pair"
        )
    identifier_keys = (
        ("parameter", "level_type", "table_version", "center")
        if source_format == "grib1"
        else ("discipline", "category", "parameter", "second_level_type")
    )
    undefined = sorted(key for key in identifier_keys if value.get(key) == 255)
    if undefined:
        raise ValueError(
            f"{label} uses missing/undefined identifier code 255 for {undefined}"
        )
    if source_format == "grib2" and value.get("level_type") == 255 \
            and any(value.get(key) is not None for key in (
                "level_value", "second_level_type", "second_level_value",
            )):
        raise ValueError(
            f"{label} uses level_type=255 with fixed-surface metadata"
        )
    if source_format == "grib2" and any(
        isinstance(value.get(key), int) and 192 <= value[key] <= 254
        for key in (
            "discipline", "category", "parameter", "level_type",
            "second_level_type",
        )
    ):
        authority = {
            "center", "subcenter", "master_table_version", "local_table_version"
        }
        missing_authority = sorted(authority - set(value))
        if missing_authority:
            raise ValueError(
                f"{label} local-use selector is missing Section 1 authority "
                f"{missing_authority}"
            )
        if value["local_table_version"] == 255:
            raise ValueError(
                f"{label} local-use selector cannot use local_table_version=255"
            )
    return value


def _selector_semantics(value: Mapping[str, object]) -> dict[str, object]:
    """Match the typed Rust selector's absent/explicit-null semantics."""

    return {key: item for key, item in value.items() if item is not None}


def _layer_bounds(
    value: object,
    label: str,
    *,
    bind_selectors: bool = False,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty ordered layer list")
    result: list[tuple[float, float]] = []
    selector_formats: set[str] = set()
    previous_top = -math.inf
    previous_bottom: float | None = None
    for index, raw in enumerate(value):
        layer_keys = {"top", "bottom"}
        if bind_selectors:
            layer_keys.add("selectors")
        layer = _object(
            raw,
            f"{label}[{index}]",
            allowed=layer_keys,
            required=layer_keys,
        )
        if bind_selectors:
            bindings = _object(
                layer["selectors"],
                f"{label}[{index}].selectors",
                allowed=set(_CANONICAL_SOIL_FIELDS),
                required=set(_CANONICAL_SOIL_FIELDS),
            )
            for field_name in _CANONICAL_SOIL_FIELDS:
                selector = _selector(
                    bindings[field_name],
                    f"{label}[{index}].selectors.{field_name}",
                )
                selector_formats.add(str(selector["format"]))
        top = _number(layer["top"], f"{label}[{index}].top")
        bottom = _number(layer["bottom"], f"{label}[{index}].bottom")
        if top < 0.0 or bottom <= top:
            raise ValueError(
                f"{label}[{index}] must have non-negative, positive-thickness "
                "depths ordered top-to-bottom"
            )
        if top < previous_top:
            raise ValueError(f"{label} is not ordered shallow-to-deep")
        if previous_bottom is not None:
            if top < previous_bottom:
                raise ValueError(
                    f"{label} layers {index - 1} and {index} overlap"
                )
            if top > previous_bottom:
                raise ValueError(
                    f"{label} has a gap between layers {index - 1} and {index}"
                )
        result.append((top, bottom))
        previous_top = top
        previous_bottom = bottom
    if bind_selectors and len(selector_formats) != 1:
        raise ValueError(
            f"{label} selectors must all use one common source format"
        )
    return tuple(result)


def _validate_mapping_fields(
    contract: Mapping[str, object],
    source_bounds: tuple[tuple[float, float], ...],
    mapping: Mapping[str, object],
) -> None:
    fields = mapping.get("fields")
    target = mapping.get("target")
    if not isinstance(fields, dict) or not isinstance(target, dict):
        raise ValueError("soil contract requires a validated mapped-source mapping")
    expected = (
        (str(contract["temperature_field"]), "K"),
        (str(contract["moisture_field"]), "m3 m-3"),
    )
    for name, units in expected:
        field = fields.get(name)
        if not isinstance(field, dict):
            raise ValueError(f"soil contract field {name!r} is not mapped")
        if tuple(field.get("target_axes", ())) != ("soil", "y", "x") \
                or field.get("location") != "soil" \
                or field.get("staggering", "none") != "none" \
                or not isinstance(field.get("units"), dict) \
                or field["units"].get("target") != units:
            raise ValueError(
                f"soil contract field {name!r} must be unstaggered "
                f"soil/y/x in {units}"
            )
        missing = field.get("missing")
        if not isinstance(missing, dict) \
                or missing.get("kind") not in {"reject", "preserve_mask"}:
            raise ValueError(
                f"soil contract field {name!r} must reject missing values or "
                "preserve an ocean mask for declared repair"
            )
        selectors = field.get("selectors", ())
        if not isinstance(selectors, list) \
                or len(selectors) != len(source_bounds):
            raise ValueError(
                f"soil contract requires exactly {len(source_bounds)} "
                f"ordered direct selectors for {name!r}"
            )
        stack_axis = field.get("selector_stack_axis")
        expected_stack_axis = "soil" if mapping.get("format") == "netcdf" else None
        if stack_axis != expected_stack_axis:
            raise ValueError(
                f"soil contract field {name!r} must use selector_stack_axis "
                f"{expected_stack_axis!r} for {mapping.get('format')!r}"
            )
        declared_selectors = tuple(
            layer["selectors"][name] for layer in contract["source_layers"]
        )
        for index, (selector, declared) in enumerate(
            zip(selectors, declared_selectors)
        ):
            if _selector_semantics(selector) != _selector_semantics(declared):
                raise ValueError(
                    f"{name} selector {index} differs from the selector "
                    "bound to its declared soil depth"
                )

        # WMO GRIB2 type 106 is a layer below land in metres.  When the
        # mapping exposes those bounds, bind every positional selector to the
        # declarative source layer instead of trusting list order alone.
        if mapping.get("format") == "grib2":
            for index, (selector, bounds) in enumerate(zip(selectors, source_bounds)):
                if not isinstance(selector, dict) \
                        or selector.get("level_type") != 106 \
                        or selector.get("second_level_type") != 106 \
                        or "level_value" not in selector \
                        or "second_level_value" not in selector:
                    raise ValueError(
                        f"{name} selector {index} must declare a bounded GRIB2 "
                        "depth-below-land layer (type 106)"
                    )
                observed = (
                    _number(selector["level_value"], f"{name} selector {index} top"),
                    _number(
                        selector["second_level_value"],
                        f"{name} selector {index} bottom",
                    ),
                )
                if observed != bounds:
                    raise ValueError(
                        f"{name} selector {index} depth {observed!r} differs "
                        f"from soil contract layer {bounds!r}"
                    )

    soil_count = target.get("soil_layer_count")
    if isinstance(soil_count, bool) or not isinstance(soil_count, int) \
            or soil_count != len(source_bounds):
        raise ValueError(
            "mapping target.soil_layer_count differs from the declarative "
            "source soil-layer count"
        )


def validate_soil_layer_contract(
    value: object,
    *,
    mapping: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and return one fail-closed mapped soil contract.

    Only metres are currently accepted because neither classic Vtables nor
    the native decoder ABI supply a general soil-depth unit conversion.  The
    target is the exact Noah state consumed by gpuwm/WRF-real, whose layer
    count is :data:`NOAH_TARGET_SOIL_LAYERS`.  This function takes no
    ``RunConfig`` and does not need one: it does not decide how many layers a
    run has -- :func:`gpuwm.config.soil_layer_count` does -- it only refuses
    a target that is not the one geometry this contract has a proven remap
    for.  A nine-layer run reaching a Noah-shaped ingest is caught by shape
    at ``gpuwm/core/physics.py:_as_soil``, which cannot broadcast four layers
    onto nine.
    """

    contract = _object(
        value,
        "composition.soil_layers",
        allowed={
            "temperature_field", "moisture_field", "depth_units",
            "source_layers", "target_layers", "remap", "missing",
        },
        required={
            "temperature_field", "moisture_field", "depth_units",
            "source_layers", "target_layers", "remap", "missing",
        },
    )
    if contract["temperature_field"] != "soil_temperature" \
            or contract["moisture_field"] != "volumetric_soil_moisture":
        raise ValueError("composition must bind the canonical soil fields")
    if contract["depth_units"] != "m":
        raise ValueError(
            "composition soil depth_units must be 'm'; implicit depth-unit "
            "conversion is forbidden"
        )

    source = _layer_bounds(
        contract["source_layers"],
        "soil source_layers",
        bind_selectors=True,
    )
    target = _layer_bounds(contract["target_layers"], "soil target_layers")
    if source[0][0] != 0.0:
        raise ValueError("soil source_layers must begin at the 0 m surface")
    if target != NOAH_LAYER_BOUNDS_M:
        raise ValueError(
            f"soil target_layers differ from the selected "
            f"{NOAH_TARGET_SOIL_LAYERS}-layer Noah contract "
            f"{NOAH_LAYER_BOUNDS_M!r}; this contract has exactly one target, "
            "and RUC's levels are a different discretization with a "
            "different value-location convention rather than a longer list "
            "of the same thing.  Admitting RUC here means adding its target "
            "and proving its remap against the WRF-real initialization; "
            "widening this comparison would fabricate a soil column"
        )

    remap = _object(
        contract["remap"],
        "composition.soil_layers.remap",
        allowed={
            "kind", "source_value_location", "target_value_location",
            "top_anchor", "bottom_anchor", "coverage",
        },
        required={"kind", "source_value_location", "target_value_location"},
    )
    kind = remap["kind"]
    if kind == _LINEAR_REMAP:
        if set(remap) != {
            "kind", "source_value_location", "target_value_location",
            "top_anchor", "bottom_anchor",
        }:
            raise ValueError(
                "linear soil remap requires exactly top_anchor and bottom_anchor"
            )
        if remap["source_value_location"] == "layer_bottom":
            raise ValueError(
                "source_value_location='layer_bottom' is not supported: "
                "WRF places layer-form soil values at the INTEGER-"
                "centimetre layer midpoints (module_optional_input.F:"
                "char2int2, (top+bottom)/2 in whole cm), which is what "
                "linear_point_samples executes; declare "
                "'wrf_integer_cm_layer_midpoint', or add genuine "
                "layer-bottom support before declaring it"
            )
        if remap["source_value_location"] != "wrf_integer_cm_layer_midpoint" \
                or remap["target_value_location"] != "layer_midpoint":
            raise ValueError(
                "linear soil remap requires wrf-integer-cm-layer-midpoint "
                "source values and layer-midpoint target values"
            )
        anchors = []
        expected_anchors = (
            ("top_anchor", 0.0, "skin_temperature", "repeat_shallowest"),
            ("bottom_anchor", 3.0, "deep_soil_temperature", "repeat_deepest"),
        )
        for name, expected_depth, expected_temperature, expected_moisture \
                in expected_anchors:
            anchor = _object(
                remap[name],
                f"composition.soil_layers.remap.{name}",
                allowed={"depth", "temperature", "moisture"},
                required={"depth", "temperature", "moisture"},
            )
            depth = _number(anchor["depth"], f"soil remap {name}.depth")
            if (depth, anchor["temperature"], anchor["moisture"]) != (
                expected_depth, expected_temperature, expected_moisture,
            ):
                raise ValueError(
                    f"soil remap {name} does not match the WRF-real/Noah "
                    "surface/deep boundary contract"
                )
            anchors.append(depth)
        source_points = (anchors[0], *(bottom for _top, bottom in source), anchors[1])
        if any(later <= earlier for earlier, later in zip(source_points, source_points[1:])):
            raise ValueError("linear soil remap sample depths are not strictly ordered")
        target_points = tuple((top + bottom) / 2.0 for top, bottom in target)
        if target_points[0] < source_points[0] or target_points[-1] > source_points[-1]:
            raise ValueError("linear soil remap does not cover every target midpoint")
    elif kind == _CONSERVATIVE_REMAP:
        if set(remap) != {
            "kind", "source_value_location", "target_value_location", "coverage",
        }:
            raise ValueError(
                "conservative soil remap requires exactly one coverage policy"
            )
        if remap["source_value_location"] != "layer_mean" \
                or remap["target_value_location"] != "layer_mean" \
                or remap["coverage"] != "require_complete":
            raise ValueError(
                "conservative soil remap requires layer means and complete coverage"
            )
        if source[0][0] > target[0][0] or source[-1][1] < target[-1][1]:
            raise ValueError(
                "conservative soil source layers do not completely cover target layers"
            )
    else:
        raise ValueError(f"unsupported declarative soil remap kind {kind!r}")

    missing = _object(
        contract["missing"],
        "composition.soil_layers.missing",
        allowed={"land", "ocean"},
        required={"land", "ocean"},
    )
    ocean = _object(
        missing["ocean"],
        "composition.soil_layers.missing.ocean",
        allowed={"stage", "temperature", "moisture"},
        required={"stage", "temperature", "moisture"},
    )
    if missing["land"] != "reject" \
            or ocean["stage"] != "after_horizontal_interpolation" \
            or ocean["temperature"] != "skin_temperature" \
            or _number(ocean["moisture"], "soil ocean moisture repair") != 1.0:
        raise ValueError(
            "soil missing policy must reject land gaps and repair ocean "
            "temperature from skin temperature with moisture 1.0 after "
            "horizontal interpolation"
        )

    if mapping is not None:
        _validate_mapping_fields(contract, source, mapping)
    return contract


def soil_layer_bounds(
    contract: Mapping[str, object], key: str,
) -> tuple[tuple[float, float], ...]:
    """Return already-validated source or target bounds as float tuples."""

    if key not in {"source_layers", "target_layers"}:
        raise ValueError("soil layer key must be source_layers or target_layers")
    return _layer_bounds(
        contract[key],
        f"soil {key}",
        bind_selectors=key == "source_layers",
    )


def conservative_overlap_weights(
    source: tuple[tuple[float, float], ...],
    target: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Return target-by-source overlap weights for layer-mean remapping."""

    weights = np.zeros((len(target), len(source)), dtype=np.float64)
    for target_index, (target_top, target_bottom) in enumerate(target):
        thickness = target_bottom - target_top
        for source_index, (source_top, source_bottom) in enumerate(source):
            overlap = max(
                0.0,
                min(target_bottom, source_bottom) - max(target_top, source_top),
            )
            weights[target_index, source_index] = overlap / thickness
        if not math.isclose(
            float(weights[target_index].sum()), 1.0, rel_tol=0.0, abs_tol=2e-15,
        ):
            raise ValueError(
                f"conservative soil remap target layer {target_index} lacks "
                "complete source coverage"
            )
    return weights


__all__ = [
    "MAPPED_SOIL_MOISTURE", "MAPPED_SOIL_TEMPERATURE",
    "NOAH_LAYER_BOUNDS_M", "NOAH_TARGET_SOIL_LAYERS",
    "conservative_overlap_weights",
    "soil_layer_bounds", "validate_soil_layer_contract",
]
