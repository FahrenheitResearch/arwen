"""Depth-node soil sources through the declarative mapped contract.

RUC-family models (HRRR, RAP) publish soil as point samples at depths --
including both the 0 m surface and the 3 m column bottom -- not as layer
means.  The contract declares that geometry as ``source_nodes`` with the
``linear_node_samples`` remap, and the executor reuses WRF's sorted linear
node interpolation, which is exactly what the certified native HRRR route
runs on its fixed node table.  These tests pin the two against each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.ingest import soil as soil_module
from gpuwm.ingest.soil import (
    HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M, _interp_nodes,
    _remap_declared_soil,
)
from gpuwm.ingest.soil_contract import (
    soil_node_depths, soil_source_sample_count, validate_soil_layer_contract,
)


def _node_selector(name: str, depth: float) -> dict[str, object]:
    parameter = 18 if name == "soil_temperature" else 192
    return {
        "format": "grib2", "discipline": 2, "category": 3 if name == "soil_temperature" else 0,
        "parameter": parameter, "center": 7, "subcenter": 0,
        "master_table_version": 2, "local_table_version": 1,
        "level_type": 106, "level_value": depth,
        "second_level_type": 106, "second_level_value": depth,
    }


def _node_contract(depths=tuple(float(d) for d in HRRR_SOIL_NODE_DEPTHS_M)):
    return {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_nodes": [
            {
                "depth": depth,
                "selectors": {
                    "soil_temperature": _node_selector(
                        "soil_temperature", depth),
                    "volumetric_soil_moisture": _node_selector(
                        "volumetric_soil_moisture", depth),
                },
            }
            for depth in depths
        ],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1},
            {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0},
            {"top": 1.0, "bottom": 2.0},
        ],
        "remap": {
            "kind": "linear_node_samples",
            "source_value_location": "level_node",
            "target_value_location": "layer_midpoint",
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


def test_node_contract_validates_and_reports_its_geometry():
    contract = validate_soil_layer_contract(_node_contract())
    assert soil_source_sample_count(contract) == 9
    assert soil_node_depths(contract) == tuple(
        float(d) for d in HRRR_SOIL_NODE_DEPTHS_M)


def test_node_contract_requires_both_endpoints():
    with pytest.raises(ValueError, match="0.0 m and the deepest at 3.0"):
        validate_soil_layer_contract(
            _node_contract(depths=(0.0, 0.1, 0.4, 1.0, 2.0)))


def test_node_contract_refuses_a_layer_remap_kind():
    contract = _node_contract()
    contract["remap"] = {
        "kind": "conservative_layer_means",
        "source_value_location": "layer_mean",
        "target_value_location": "layer_mean",
        "coverage": "require_complete",
    }
    with pytest.raises(ValueError, match="source_nodes require"):
        validate_soil_layer_contract(contract)


def test_node_contract_refuses_both_geometries_at_once():
    contract = _node_contract()
    contract["source_layers"] = [
        {"top": 0.0, "bottom": 0.1, "selectors": {
            "soil_temperature": _node_selector("soil_temperature", 0.0),
            "volumetric_soil_moisture": _node_selector(
                "volumetric_soil_moisture", 0.0),
        }},
    ]
    with pytest.raises(ValueError, match="exactly one source"):
        validate_soil_layer_contract(contract)


def test_node_remap_matches_the_native_hrrr_node_interpolation():
    """The declared executor IS the native route's arithmetic."""

    contract = validate_soil_layer_contract(_node_contract())
    rng = np.random.default_rng(7)
    temperature = 280.0 + rng.random((9, 4, 5))
    moisture = 0.2 + 0.5 * rng.random((9, 4, 5))
    tsk = np.full((4, 5), 300.0)
    deep = np.full((4, 5), 285.0)
    soil_t, soil_m = _remap_declared_soil(
        temperature, moisture, contract, tsk=tsk, deep=deep)
    native_t = _interp_nodes(
        temperature, HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M)
    native_m = _interp_nodes(
        moisture, HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M)
    np.testing.assert_array_equal(soil_t, native_t)
    np.testing.assert_array_equal(soil_m, native_m)
    # The anchors were never consulted: the endpoints are source samples.
    hot_tsk = np.full((4, 5), 400.0)
    soil_t_again, _ = _remap_declared_soil(
        temperature, moisture, contract, tsk=hot_tsk, deep=deep)
    np.testing.assert_array_equal(soil_t, soil_t_again)


def test_mapped_route_repairs_bounded_sixteen_pt_overshoot_and_refuses_beyond(capsys):
    """WPS sixteen_pt genuinely overshoots on sharp gradients.

    MEASURED on the first real HRRR wrfprs preparation: one land cell of
    22,482 at -0.016 volumetric moisture.  Within the margin the mapped
    route clamps to the physical bound, loudly; beyond it the refusal is
    unchanged.  Both behaviours are default-on -- no flag.
    """

    from gpuwm.ingest.soil import preprocess_noah_soil
    from gpuwm.ingest.soil_contract import (
        MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE)

    contract = _node_contract()
    surface = {
        "LANDSEA": np.asarray([[1.0, 0.0], [1.0, 1.0]]),
        "SKINTEMP": np.asarray([[291.0, 285.0], [289.0, 288.0]]),
        "TMN": np.asarray([[286.0, 280.0], [285.0, 284.0]]),
    }
    temperature = np.broadcast_to(
        np.linspace(290.0, 284.0, 9)[:, None, None], (9, 2, 2)).copy()
    moisture = np.full((9, 2, 2), 0.25)
    moisture[0, 0, 0] = -0.016          # the measured overshoot shape
    fields = {
        **surface,
        MAPPED_SOIL_TEMPERATURE: temperature,
        MAPPED_SOIL_MOISTURE: moisture,
    }
    state = preprocess_noah_soil(
        fields, soil_type=np.full((2, 2), 6),
        soil_layer_contract=contract)
    assert float(np.min(state.soil_moisture)) >= 0.0
    captured = capsys.readouterr()
    assert "overshoot clamped" in captured.err

    beyond = moisture.copy()
    beyond[0, 0, 0] = -0.10             # beyond the admission margin
    fields[MAPPED_SOIL_MOISTURE] = beyond
    with pytest.raises(ValueError, match="outside 0..1 on land"):
        preprocess_noah_soil(
            fields, soil_type=np.full((2, 2), 6),
            soil_layer_contract=contract)


def test_layer_contracts_are_unchanged():
    """The historical layer path validates and executes as before."""

    contract = {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_layers": [
            {"top": top, "bottom": bottom, "selectors": {
                "soil_temperature": {
                    "format": "grib2", "discipline": 0, "category": 0,
                    "parameter": 0, "level_type": 106, "level_value": top,
                    "second_level_type": 106, "second_level_value": bottom,
                },
                "volumetric_soil_moisture": {
                    "format": "grib2", "discipline": 2, "category": 0,
                    "parameter": 192, "center": 7, "subcenter": 2,
                    "master_table_version": 2, "local_table_version": 1,
                    "level_type": 106, "level_value": top,
                    "second_level_type": 106, "second_level_value": bottom,
                },
            }}
            for top, bottom in ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))
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
    validated = validate_soil_layer_contract(contract)
    assert soil_source_sample_count(validated) == 4
    temperature = np.full((4, 2, 2), 290.0)
    moisture = np.full((4, 2, 2), 0.3)
    soil_t, soil_m = _remap_declared_soil(
        temperature, moisture, validated,
        tsk=np.full((2, 2), 300.0), deep=np.full((2, 2), 285.0))
    np.testing.assert_array_equal(soil_t, temperature)
    np.testing.assert_array_equal(soil_m, moisture)
