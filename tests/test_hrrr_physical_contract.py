"""Native HRRR units and grid basis must survive physical-store capture/replay."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
from gpuwm.ensemble.hrrr_physical_contract import (
    hrrr_physical_field_contract, validate_hrrr_physical_field_contract,
    validate_hrrr_physical_snapshot,
)


def _contract():
    grid = {"mass_shape": [3, 4]}
    evidence = {name: hashlib.sha256(name.encode()).hexdigest() for name in (
        "native_mapper", "raw_source_manifest", "water_temperature_assembly", "sealed_native_bridge_manifest")}
    return grid, hrrr_physical_field_contract(grid, evidence=evidence)


def test_native_hrrr_declares_hybrid_indices_and_distinct_moisture_quantities():
    grid, contract = _contract()
    assert contract["vertical"] == {"kind": "hybrid_model_levels", "units": "1", "values": "levels_hpa", "pressure_field": "field__PRES"}
    assert contract["arrays"]["field__PRES"]["units"] == "Pa"
    assert contract["arrays"]["field__SNOW"]["units"] == "kg m-2"
    assert contract["arrays"]["field__SNOWH"]["units"] == "m"
    assert contract["arrays"]["field__SOILW"]["units"] == "1"
    assert contract["arrays"]["field__Q2"]["units"] == "kg kg-1"
    assert "Specific humidity" in contract["arrays"]["field__Q2"]["operation"]
    assert validate_hrrr_physical_field_contract(contract, grid) is contract


@pytest.mark.parametrize(("name", "dimensions", "basis"), [
    ("UU", ["level", "y", "x_stag"], "grid_x"),
    ("VV", ["level", "y_stag", "x"], "grid_y"),
    ("U10", ["y", "x_stag"], "grid_x"),
    ("V10", ["y_stag", "x"], "grid_y"),
])
def test_native_hrrr_winds_are_rotated_target_face_vectors(name, dimensions, basis):
    _, contract = _contract()
    observed = contract["arrays"]["field__"+name]
    assert observed["dimensions"] == dimensions and observed["basis"] == basis
    assert "source HRRR grid-relative" in observed["operation"]
    assert "target Lambert grid basis" in observed["operation"]


@pytest.mark.parametrize(("name", "attribute", "wrong"), [
    ("field__TT", "units", "C"), ("field__Q2", "units", "%"),
    ("field__SNOW", "units", "m"), ("field__UU", "basis", "scalar"),
    ("field__SOILT", "dimensions", ["level", "y", "x"]),
])
def test_declared_but_incompatible_fields_are_refused_before_native_real(name, attribute, wrong):
    grid, contract = _contract()
    changed = deepcopy(contract)
    changed["arrays"][name][attribute] = wrong
    with pytest.raises(ValueError, match="incompatible"):
        validate_hrrr_physical_field_contract(changed, grid)


def test_pressure_labels_cannot_replace_native_hybrid_indices():
    snapshot = SimpleNamespace(levels_hpa=np.arange(1., 51.), fields={"SOILT": np.ones((9, 3, 4))})
    assert validate_hrrr_physical_snapshot(snapshot) is snapshot
    snapshot.levels_hpa = np.linspace(1000., 10., 50)
    with pytest.raises(ValueError, match="hybrid indices"):
        validate_hrrr_physical_snapshot(snapshot)
    snapshot.levels_hpa = np.arange(1., 51.)
    snapshot.fields["SOILT"] = np.ones((4, 3, 4))
    with pytest.raises(ValueError, match="nine native soil depths"):
        validate_hrrr_physical_snapshot(snapshot)


def test_native_contract_cannot_be_made_without_decoder_authority():
    with pytest.raises(ValueError, match="authorities"):
        hrrr_physical_field_contract({"mass_shape": [3, 4]}, evidence={})
