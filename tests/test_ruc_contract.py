"""Source, geometry, state, and option gates for the RUC LSM port."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gpuwm.core.noahmp_mynn_contract import ReferenceAsset
from gpuwm.core.ruc_contract import (
    NUM_SOIL_LAYERS,
    RUC_INIT_ORACLE_ASSET,
    RUC_INIT_ORACLE_CASES,
    RUC_NAMELIST_DEFAULTS,
    RUC_PACKAGE_STATE_FIELDS,
    RUC_PACKAGE_STATE_SHA256,
    RUC_SOIL_LEVELS_M,
    RUC_SOILPROP_ORACLE_ASSET,
    RUC_SOILPROP_ORACLE_CASES,
    RUC_SOILMOIST_ORACLE_ASSET,
    RUC_SOILMOIST_ORACLE_CASES,
    RUC_SOIL_ORACLE_ASSET,
    RUC_SOIL_ORACLE_CASES,
    RUC_SOILTEMP_ORACLE_ASSET,
    RUC_SOILTEMP_ORACLE_CASES,
    RUC_SOURCE_ASSETS,
    RUC_STEP_ORACLE_ASSET,
    RUC_STEP_ORACLE_CASES,
    RUC_SURFACE_ORACLE_ASSET,
    RUC_SURFACE_ORACLE_CASES,
    RUC_TABLE_ASSETS,
    RUC_TBQ_ORACLE_ASSET,
    RUC_TRANSF_ORACLE_ASSET,
    RUC_TRANSF_ORACLE_CASES,
    SF_SURFACE_PHYSICS,
    WRF_SUPPORTED_NUM_SOIL_LAYERS,
    resolve_default_ruc_options,
    validate_ruc_reference_assets,
)


def test_ruc_selector_geometry_and_supported_layer_identity_are_exact():
    assert SF_SURFACE_PHYSICS == 3
    assert NUM_SOIL_LAYERS == 9
    assert WRF_SUPPORTED_NUM_SOIL_LAYERS == (6, 9)
    assert RUC_SOIL_LEVELS_M == (
        0.00, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00,
    )


def test_ruc_registry_package_inventory_is_complete_and_hash_bound():
    assert RUC_PACKAGE_STATE_FIELDS == (
        "smfr3d", "keepfr3dflag", "soilt1", "rhosnf",
        "snowfallac", "precipfr", "acrunoff",
    )
    encoded = ",".join(RUC_PACKAGE_STATE_FIELDS).encode("ascii")
    assert hashlib.sha256(encoded).hexdigest() == RUC_PACKAGE_STATE_SHA256


def test_ruc_authority_assets_pin_source_geometry_registry_and_tables():
    assert tuple(asset.relative_path for asset in RUC_SOURCE_ASSETS) == (
        "phys/module_sf_ruclsm.F",
        "share/module_soil_pre.F",
        "Registry/Registry.EM_COMMON",
    )
    assert RUC_SOURCE_ASSETS[0].bytes == 288_509
    assert RUC_SOURCE_ASSETS[0].sha256 == (
        "3265f810d08dcbddfaf198371dc7f652e78e8d3a788f703a515c555a3bbb2a12"
    )
    assert tuple(asset.relative_path for asset in RUC_TABLE_ASSETS) == (
        "run/VEGPARM.TBL", "run/SOILPARM.TBL", "run/GENPARM.TBL",
    )
    assert RUC_TABLE_ASSETS[0].bytes == 23_037


def test_ruc_initialization_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_INIT_ORACLE_CASES == (
        "warm_land", "frozen_land", "water", "sea_ice",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "init.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_INIT_ORACLE_ASSET.bytes == 3_614
    assert hashlib.sha256(payload).hexdigest() == RUC_INIT_ORACLE_ASSET.sha256


def test_ruc_full_step_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_STEP_ORACLE_CASES == (
        "warm_rain", "cold_snow", "water", "sea_ice",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "step.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_STEP_ORACLE_ASSET.bytes == 14_042
    assert hashlib.sha256(payload).hexdigest() == RUC_STEP_ORACLE_ASSET.sha256


def test_ruc_surface_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_SURFACE_ORACLE_CASES == (
        "evergreen_cold", "evergreen_warm", "crop_midseason",
        "water_preserve_znt", "lai2d_preserve", "grass_short_season",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "soilvegin.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_SURFACE_ORACLE_ASSET.bytes == 1_600
    assert hashlib.sha256(payload).hexdigest() == RUC_SURFACE_ORACLE_ASSET.sha256


def test_ruc_soilprop_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_SOILPROP_ORACLE_CASES == (
        "warm_wet", "warm_dry", "frozen", "deep_frozen_keep",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "soilprop.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_SOILPROP_ORACLE_ASSET.bytes == 10_137
    assert hashlib.sha256(payload).hexdigest() == RUC_SOILPROP_ORACLE_ASSET.sha256


def test_ruc_transf_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_TRANSF_ORACLE_CASES == (
        "wet_forest", "wilt_dark_grass", "mixed_hot_crop", "bare_high_sun",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "transf.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_TRANSF_ORACLE_ASSET.bytes == 7_499
    assert hashlib.sha256(payload).hexdigest() == RUC_TRANSF_ORACLE_ASSET.sha256


def test_ruc_soilmoist_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_SOILMOIST_ORACLE_CASES == (
        "rain_wet", "dry_evap", "dew", "frozen_melt",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "soilmoist.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_SOILMOIST_ORACLE_ASSET.bytes == 16_834
    assert hashlib.sha256(payload).hexdigest() == RUC_SOILMOIST_ORACLE_ASSET.sha256


def test_ruc_soiltemp_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_SOILTEMP_ORACLE_CASES == (
        "moist_saturated", "dry_second_solve", "warm_rain",
        "humid_condensing",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "soiltemp.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_SOILTEMP_ORACLE_ASSET.bytes == 17_365
    assert hashlib.sha256(payload).hexdigest() == RUC_SOILTEMP_ORACLE_ASSET.sha256


def test_ruc_tbq_oracle_asset_is_pinned():
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "tbq.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_TBQ_ORACLE_ASSET.bytes == 84_334
    assert hashlib.sha256(payload).hexdigest() == RUC_TBQ_ORACLE_ASSET.sha256


def test_ruc_soil_oracle_asset_and_case_inventory_are_pinned():
    assert RUC_SOIL_ORACLE_CASES == (
        "warm_forest_rain", "dry_grass", "frozen_crop_rain",
        "humid_bare_dew",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "soil.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == RUC_SOIL_ORACLE_ASSET.bytes == 32_332
    assert hashlib.sha256(payload).hexdigest() == RUC_SOIL_ORACLE_ASSET.sha256


def test_ruc_default_knobs_resolve_omissions_and_reject_unvalidated_modes():
    resolved = resolve_default_ruc_options({"flag_sm_adj": 0})
    assert dict(resolved) == dict(RUC_NAMELIST_DEFAULTS)
    with pytest.raises(TypeError):
        resolved["flag_sm_adj"] = 1
    for override in (
        {"num_soil_layers": 6},
        {"mosaic_lu": 1},
        {"mosaic_soil": 1},
        {"flag_sm_adj": 1},
        {"spp_lsm": 1},
        {"spp_lsm": False},
    ):
        with pytest.raises(ValueError, match="unsupported overrides"):
            resolve_default_ruc_options(override)
    with pytest.raises(ValueError, match="unknown RUC options"):
        resolve_default_ruc_options({"invented": 0})


def test_reference_validator_fails_closed_on_size_and_byte_drift(tmp_path):
    payload = b"ruc"
    path = tmp_path / "asset"
    path.write_bytes(payload)
    asset = ReferenceAsset(
        "asset", len(payload), hashlib.sha256(payload).hexdigest()
    )
    assert validate_ruc_reference_assets(tmp_path, (asset,)) == (asset,)
    path.write_bytes(b"RUC")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_ruc_reference_assets(tmp_path, (asset,))
    path.write_bytes(b"longer")
    with pytest.raises(ValueError, match="bytes"):
        validate_ruc_reference_assets(tmp_path, (asset,))
