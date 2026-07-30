"""Stock-WRF package inventories and arbitrary vertical-grid gates."""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.vertical_contract import (
    expected_coordinate_shapes,
    validate_explicit_eta_grid,
)
from gpuwm.wrf_physics_inventory import (
    WRFINPUT_3D_DIMS,
    stock_wrf_physics_inventory,
    supported_stock_wrf_mp_physics,
)


@pytest.mark.parametrize(
    ("mp_physics", "expected"),
    [
        (6, ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")),
        (
            8,
            (
                "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                "QNICE", "QNRAIN",
            ),
        ),
        (
            10,
            (
                "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                "QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL",
            ),
        ),
        (
            18,
            (
                "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
                "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
            ),
        ),
    ],
)
def test_wrf_v461_registry_package_inventory(mp_physics, expected):
    inventory = stock_wrf_physics_inventory(mp_physics)
    assert tuple(field.netcdf_name for field in inventory.wrfinput_fields) == expected
    assert all(field.dtype == "float32" for field in inventory.wrfinput_fields)
    assert all(field.dimensions == WRFINPUT_3D_DIMS for field in inventory.wrfinput_fields)
    assert inventory.wrfinput_fields[0].initialization == "source_specific_humidity"
    assert all(
        field.initialization == "zero_if_source_absent"
        for field in inventory.wrfinput_fields[1:]
    )


def test_runtime_only_registry_state_is_not_claimed_as_wrfinput():
    for value in supported_stock_wrf_mp_physics():
        inventory = stock_wrf_physics_inventory(value)
        input_names = {field.netcdf_name for field in inventory.wrfinput_fields}
        runtime_names = {
            field.netcdf_name for field in inventory.runtime_state_not_wrfinput
        }
        assert input_names.isdisjoint(runtime_names)
    assert {
        field.netcdf_name
        for field in stock_wrf_physics_inventory(10).runtime_state_not_wrfinput
    } == {"RQRCUTEN", "RQSCUTEN", "RQICUTEN"}
    assert {
        field.netcdf_name
        for field in stock_wrf_physics_inventory(18).runtime_state_not_wrfinput
    } == {"RE_CLOUD", "RE_ICE", "RE_SNOW"}


def test_nssl2_default_inventory_pins_exact_registry_packages_and_units():
    inventory = stock_wrf_physics_inventory(18)

    assert inventory.scheme == "NSSL-2"
    assert inventory.registry_package == (
        "nssl_2mom+nssl2mconc+nssl_hail+nssl_ccn_opt+nssl_hailvol"
    )
    assert {
        field.netcdf_name: field.units for field in inventory.wrfinput_fields
    } == {
        "QVAPOR": "kg kg-1",
        "QCLOUD": "kg kg-1",
        "QRAIN": "kg kg-1",
        "QICE": "kg kg-1",
        "QSNOW": "kg kg-1",
        "QGRAUP": "kg kg-1",
        "QHAIL": "kg kg-1",
        "QNDROP": "# kg-1",
        "QNRAIN": "# kg(-1)",
        "QNICE": "# kg-1",
        "QNSNOW": "# kg(-1)",
        "QNGRAUPEL": "# kg(-1)",
        "QNHAIL": "# kg(-1)",
        "QNCCN": "# kg(-1)",
        "QVGRAUPEL": "m(3) kg(-1)",
        "QVHAIL": "m(3) kg(-1)",
    }


def test_unknown_package_fails_closed_with_actionable_message():
    with pytest.raises(ValueError, match=r"mp_physics=28.*Registry package"):
        stock_wrf_physics_inventory(28)
    with pytest.raises(TypeError, match="WRF integer"):
        stock_wrf_physics_inventory(True)


@pytest.mark.parametrize("mass_levels", [35, 49, 80])
def test_explicit_vertical_grid_is_structural_not_49_level_allowlist(mass_levels):
    eta = np.linspace(1.0, 0.0, mass_levels + 1, dtype=np.float64)
    checked = validate_explicit_eta_grid(
        eta,
        nz=mass_levels,
        p_top=5000.0,
        source_top_pressure_pa=5000.0,
        context=f"{mass_levels}-mass-level RW-WPS fixture",
    )
    assert np.array_equal(checked, eta)
    shapes = expected_coordinate_shapes(mass_levels)
    assert shapes["znu"] == (mass_levels,)
    assert shapes["znw"] == (mass_levels + 1,)


def test_vertical_source_coverage_fails_closed():
    eta = np.linspace(1.0, 0.0, 81, dtype=np.float64)
    with pytest.raises(ValueError, match=r"source atmosphere stops at 10000 Pa"):
        validate_explicit_eta_grid(
            eta,
            nz=80,
            p_top=5000.0,
            source_top_pressure_pa=10000.0,
            context="80-level unsupported source top",
        )
