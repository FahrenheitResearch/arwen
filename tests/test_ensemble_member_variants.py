"""Named existing-land and initial-surface roster planning contracts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.member_variants import (
    member_surface_options, normalize_member_variants, preparation_binding_key,
    require_distinct_variants, variant_configuration)


def _raw():
    return {"experiment": {"run_seconds": 3600, "feedback": 1},
            "shared": {"sf_surface_physics": 3, "num_soil_layers": 6},
            "domain": [{"grid_id": 1}, {"grid_id": 2, "sf_surface_physics": 3, "num_soil_layers": 6}],
            "fetch": {"source": "source", "cycle": "2026-10-04T12", "hours": 1}}


def _variants():
    return [{"name": "control"},
            {"name": "soil-dry20", "surface": {"soil_moisture_scale": 0.8}},
            {"name": "noah-control", "physics": {"sf_surface_physics": 2, "num_soil_layers": 4}}]


def test_fixed_arms_share_only_matching_source_and_soil_authorities():
    raw = _raw()
    variants = normalize_member_variants(_variants(), 3)
    configurations = [variant_configuration(raw, variant) for variant in variants]
    keys = [preparation_binding_key("source-trajectory", config) for config in configurations]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert keys[0] != preparation_binding_key("other-source", configurations[0])
    assert raw == _raw()
    assert configurations[2]["shared"] == {"sf_surface_physics": 2, "num_soil_layers": 4}
    assert configurations[2]["domain"][1] == {"grid_id": 2, "sf_surface_physics": 2, "num_soil_layers": 4}


def test_existing_nine_layer_ruc_selector_retains_its_own_bank():
    variants = normalize_member_variants([
        {"name": "control"}, {"name": "ruc-nine", "physics": {
            "sf_surface_physics": 3, "num_soil_layers": 9}}], 2)
    assert preparation_binding_key("source", _raw()) != preparation_binding_key(
        "source", variant_configuration(_raw(), variants[1]))


def test_preparation_key_binds_grid_clock_static_and_output_authorities():
    raw = _raw()
    original = preparation_binding_key("source", raw)
    same = deepcopy(raw)
    same["ensemble"] = {"members": 10}
    assert preparation_binding_key("source", same) == original
    for key in ("feedback", "run_seconds"):
        changed = deepcopy(raw)
        changed["experiment"][key] += 1
        assert preparation_binding_key("source", changed) != original
    changed = deepcopy(raw)
    changed["domain"][1]["nx"] = 216
    assert preparation_binding_key("source", changed) != original
    changed = deepcopy(raw)
    changed["static"] = {"highres": {"fields": "terrain"}}
    assert preparation_binding_key("source", changed) != original


def test_member_surface_descriptor_detaches_values_and_keeps_identity_control():
    variants = normalize_member_variants(_variants(), 3)
    request = SimpleNamespace(member_variants=variants, perturbation=None)
    assert member_surface_options(request, 0) is None
    options = member_surface_options(request, 1)
    options["soil_moisture_scale"] = 0.5
    assert variants[1]["surface"]["soil_moisture_scale"] != 0.5
    assert member_surface_options(SimpleNamespace(perturbation=None), 0) is None


@pytest.mark.parametrize("variants,words", [
    ([{"name": "duplicate"}, {"name": "duplicate"}], "unique"),
    ([{"name": "control"}, {"name": "bad", "physics": {"num_soil_layers": 4}}], "both integer"),
    ([{"name": "control"}, {"name": "bad", "physics": {"sf_surface_physics": 3, "num_soil_layers": 4}}], "wrong land column"),
    ([{"name": "control"}, {"name": "bad", "physics": {"soil_diffusivity": 2}}], "existing land selectors"),
    ([{"name": "control"}, {"name": "bad", "surface": {"sst_offset_k": [-1, 1]}}], "fixed scalar"),
    ([{"name": "control"}, {"name": "bad", "surface": {"soil_moisture_scale": 0}}], "must be positive"),
])
def test_unbound_or_unrepresentable_member_controls_refuse_before_preparation(variants, words):
    with pytest.raises(ValueError, match=words):
        normalize_member_variants(variants, 2)


def test_different_labels_cannot_fabricate_initial_state_diversity():
    with pytest.raises(ValueError, match="fabricate ensemble size"):
        normalize_member_variants([{"name": "control"}, {"name": "copy"}], 2)


@pytest.mark.parametrize("surface", [
    {"soil_moisture_scale": 1.00000000001},
    {"sst_offset_k": 1e-50},
    {"sst_offset_k": -0.0},
])
def test_distinct_python_descriptors_cannot_repeat_identical_fp32_controls(surface):
    with pytest.raises(ValueError, match="identical FP32.*fabricate ensemble size"):
        normalize_member_variants([{"name": "control"}, {"name": "rounded-copy", "surface": surface}], 2)
