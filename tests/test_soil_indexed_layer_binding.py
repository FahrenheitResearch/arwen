"""Index-addressed GRIB2 soil layers through the declarative contract.

Some producers publish their soil column as layer MEANS whose GRIB2 fixed
surfaces carry layer ORDINALS on a producer-declared vertical coordinate
(code table 4.5 type 151), not metre depths on type 106; the physical
bounds live in the producer's documented discretization.  ECMWF's open
data encodes every soil record this way.  The contract admits that
geometry only when the composition DECLARES it -- an explicit
``selector_depth_binding`` binding each positional selector to its
ordinal pair -- so a producer that reorders or renumbers its layers is a
refusal, never a silent level swap.  Absent the declaration, behaviour is
unchanged: metre-valued type-106 layers remain required.
"""

from __future__ import annotations

import pytest

from gpuwm.ingest.soil_contract import validate_soil_layer_contract


_BOUNDS = ((0.0, 0.07), (0.07, 0.28), (0.28, 1.0), (1.0, 2.89))


def _indexed_selector(name: str, index: int) -> dict[str, object]:
    category = 3 if name == "soil_temperature" else 0
    parameter = 18 if name == "soil_temperature" else 25
    return {
        "format": "grib2", "discipline": 2, "category": category,
        "parameter": parameter,
        "level_type": 151, "level_value": index,
        "second_level_type": 151, "second_level_value": index + 1,
    }


def _indexed_contract():
    return {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "selector_depth_binding": {
            "kind": "indexed_fixed_surfaces",
            "level_type": 151,
            "first_index": 0,
        },
        "source_layers": [
            {
                "top": top,
                "bottom": bottom,
                "selectors": {
                    "soil_temperature": _indexed_selector(
                        "soil_temperature", index),
                    "volumetric_soil_moisture": _indexed_selector(
                        "volumetric_soil_moisture", index),
                },
            }
            for index, (top, bottom) in enumerate(_BOUNDS)
        ],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1},
            {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0},
            {"top": 1.0, "bottom": 2.0},
        ],
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
    }


def _soil_field(name: str) -> dict[str, object]:
    units = "K" if name == "soil_temperature" else "m3 m-3"
    return {
        "selectors": [
            _indexed_selector(name, index) for index in range(len(_BOUNDS))
        ],
        "units": {"source": units, "target": units},
        "source_axes": ["soil", "y", "x"],
        "target_axes": ["soil", "y", "x"],
        "location": "soil",
        "staggering": "none",
        "missing": {"kind": "preserve_mask"},
    }


def _mapping() -> dict[str, object]:
    return {
        "format": "grib2",
        "fields": {
            "soil_temperature": _soil_field("soil_temperature"),
            "volumetric_soil_moisture": _soil_field(
                "volumetric_soil_moisture"),
        },
        "target": {"soil_layer_count": len(_BOUNDS)},
    }


def test_declared_indexed_layers_validate_with_their_mapping():
    contract = validate_soil_layer_contract(
        _indexed_contract(), mapping=_mapping())
    binding = contract["selector_depth_binding"]
    assert binding["level_type"] == 151
    assert binding["first_index"] == 0


def test_a_renumbered_producer_is_a_refusal_not_a_level_swap():
    mapping = _mapping()
    selectors = mapping["fields"]["soil_temperature"]["selectors"]
    selectors[1], selectors[2] = selectors[2], selectors[1]
    contract = _indexed_contract()
    layers = contract["source_layers"]
    swap = layers[1]["selectors"]
    layers[1]["selectors"] = layers[2]["selectors"]
    layers[2]["selectors"] = swap
    with pytest.raises(ValueError, match="index"):
        validate_soil_layer_contract(contract, mapping=mapping)


def test_the_wrong_fixed_surface_type_is_named_in_the_refusal():
    mapping = _mapping()
    for field in mapping["fields"].values():
        for selector in field["selectors"]:
            selector["level_type"] = 1
    contract = _indexed_contract()
    for layer in contract["source_layers"]:
        for selector in layer["selectors"].values():
            selector["level_type"] = 1
    with pytest.raises(ValueError, match="151"):
        validate_soil_layer_contract(contract, mapping=mapping)


def test_the_binding_may_not_launder_metre_depth_semantics():
    contract = _indexed_contract()
    contract["selector_depth_binding"]["level_type"] = 106
    with pytest.raises(ValueError, match="106"):
        validate_soil_layer_contract(contract, mapping=_mapping())


def test_the_binding_is_a_layer_declaration_not_a_node_one():
    contract = _indexed_contract()
    layers = contract.pop("source_layers")
    contract["source_nodes"] = [
        {"depth": float(index), "selectors": layer["selectors"]}
        for index, layer in enumerate(layers)
    ]
    contract["remap"] = {
        "kind": "linear_node_samples",
        "source_value_location": "level_node",
        "target_value_location": "layer_midpoint",
    }
    with pytest.raises(ValueError, match="source_layers"):
        validate_soil_layer_contract(contract)


def test_absent_declaration_still_requires_metre_valued_type_106():
    contract = _indexed_contract()
    del contract["selector_depth_binding"]
    with pytest.raises(ValueError, match="type 106"):
        validate_soil_layer_contract(contract, mapping=_mapping())
