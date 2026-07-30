"""Source, option, and state gates for the Noah-MP + MYNN port."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from gpuwm.core.noahmp_mynn_contract import (
    BL_PBL_PHYSICS,
    MYNN_ADVECTED_FIELDS,
    MYNN_MIXLENGTH_ORACLE_ASSET,
    MYNN_MIXLENGTH_ORACLE_CASES,
    MYNN_NAMELIST_DEFAULTS,
    MYNN_PACKAGE_STATE_FIELDS,
    MYNN_PBL_LEVEL2_ORACLE_ASSET,
    MYNN_PBL_LEVEL2_ORACLE_CASES,
    MYNN_PBLH_SCALE_ORACLE_ASSET,
    MYNN_PBLH_SCALE_ORACLE_CASES,
    MYNN_PREDICT_ORACLE_ASSET,
    MYNN_PREDICT_ORACLE_CASES,
    MYNN_SURFACE_ORACLE_ASSET,
    MYNN_SURFACE_ORACLE_CASES,
    MYNN_TURBULENCE_ORACLE_ASSET,
    MYNN_TURBULENCE_ORACLE_CASES,
    NOAHMP_NAMELIST_DEFAULTS,
    NOAHMP_PACKAGE_STATE_FIELDS,
    NOAHMP_PACKAGE_STATE_SHA256,
    NOAHMP_REFERENCE_COMMIT,
    ReferenceAsset,
    SF_SFCLAY_PHYSICS,
    SF_SURFACE_PHYSICS,
    WRF_REFERENCE_COMMIT,
    resolve_default_mynn_options,
    resolve_default_noahmp_options,
    validate_reference_assets,
)


def test_selectors_and_pinned_revisions_are_exact():
    assert SF_SURFACE_PHYSICS == 4
    assert SF_SFCLAY_PHYSICS == BL_PBL_PHYSICS == 5
    assert WRF_REFERENCE_COMMIT == (
        "d66e442fccc04111067e29274c9f9eaccc3cef28"
    )
    assert NOAHMP_REFERENCE_COMMIT == (
        "848f54ad3d28c4303151fe5ad83724e232694422"
    )


def test_noahmp_registry_package_inventory_is_complete_and_hash_bound():
    encoded = ",".join(NOAHMP_PACKAGE_STATE_FIELDS).encode("ascii")
    assert len(NOAHMP_PACKAGE_STATE_FIELDS) == 224
    assert hashlib.sha256(encoded).hexdigest() == NOAHMP_PACKAGE_STATE_SHA256
    assert NOAHMP_PACKAGE_STATE_FIELDS[:5] == (
        "isnowxy", "tvxy", "tgxy", "canliqxy", "canicexy"
    )
    assert NOAHMP_PACKAGE_STATE_FIELDS[-5:] == (
        "acc_ecanxy", "acc_etranxy", "acc_edirxy", "qmeltxy", "acsnmelt"
    )


def test_mynn_registry_state_includes_advected_tke_and_coupled_state():
    assert MYNN_ADVECTED_FIELDS == ("qke_adv",)
    assert MYNN_PACKAGE_STATE_FIELDS == (
        "qke", "tke_pbl", "sh3d", "sm3d",
        "tsq", "qsq", "cov", "el_pbl",
    )


def test_mynn_surface_oracle_asset_and_regime_inventory_are_pinned():
    assert MYNN_SURFACE_ORACLE_ASSET.bytes == 8_600
    assert MYNN_SURFACE_ORACLE_ASSET.sha256 == (
        "049caf3b3add0a68730fc7e8953041425797715b3136169a7e290bef72ff9b04"
    )
    assert MYNN_SURFACE_ORACLE_CASES == (
        "stable_land", "unstable_land", "neutral_land", "snow_land",
        "stable_water", "unstable_water",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "mynn" / "oracle" / "surface-layer.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == MYNN_SURFACE_ORACLE_ASSET.bytes
    assert hashlib.sha256(payload).hexdigest() == MYNN_SURFACE_ORACLE_ASSET.sha256
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(row["case"] for row in rows) == MYNN_SURFACE_ORACLE_CASES


def test_mynn_pbl_level2_oracle_asset_and_profile_inventory_are_pinned():
    assert MYNN_PBL_LEVEL2_ORACLE_ASSET.bytes == 18_056
    assert MYNN_PBL_LEVEL2_ORACLE_ASSET.sha256 == (
        "a953224da302091dc7a3ec805ea44393f452c481368d35d3ce4b8e150b5b8ea9"
    )
    assert MYNN_PBL_LEVEL2_ORACLE_CASES == (
        "stable_dry", "convective_dry", "neutral_shear", "moist_cloud",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "mynn" / "oracle" / "pbl-level2.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == MYNN_PBL_LEVEL2_ORACLE_ASSET.bytes
    assert hashlib.sha256(payload).hexdigest() == (
        MYNN_PBL_LEVEL2_ORACLE_ASSET.sha256
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        MYNN_PBL_LEVEL2_ORACLE_CASES
    )


def test_mynn_pblh_scale_oracle_asset_and_column_inventory_are_pinned():
    assert MYNN_PBLH_SCALE_ORACLE_ASSET.bytes == 10_682
    assert MYNN_PBLH_SCALE_ORACLE_ASSET.sha256 == (
        "f6f8ef33b36048257786fab3096f77bc0ea320b7f6a377b023518b78218407b1"
    )
    assert MYNN_PBLH_SCALE_ORACLE_CASES == (
        "convective_land", "stable_land", "marine", "cold_pool",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "mynn" / "oracle" / "pblh-scale.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == MYNN_PBLH_SCALE_ORACLE_ASSET.bytes
    assert hashlib.sha256(payload).hexdigest() == (
        MYNN_PBLH_SCALE_ORACLE_ASSET.sha256
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        MYNN_PBLH_SCALE_ORACLE_CASES
    )


def test_mynn_mixlength_oracle_asset_and_column_inventory_are_pinned():
    assert MYNN_MIXLENGTH_ORACLE_ASSET.bytes == 28_310
    assert MYNN_MIXLENGTH_ORACLE_ASSET.sha256 == (
        "733b689bfaf5ecfe522e7949003229fb0cdf8bea16e8759132b2f296ab7a0e77"
    )
    assert MYNN_MIXLENGTH_ORACLE_CASES == (
        "stable", "convective", "high_shear", "edmf_active",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "mynn" / "oracle" / "mixlength.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == MYNN_MIXLENGTH_ORACLE_ASSET.bytes
    assert hashlib.sha256(payload).hexdigest() == MYNN_MIXLENGTH_ORACLE_ASSET.sha256
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        MYNN_MIXLENGTH_ORACLE_CASES
    )


def test_mynn_turbulence_oracle_asset_and_column_inventory_are_pinned():
    assert MYNN_TURBULENCE_ORACLE_ASSET.bytes == 51_156
    assert MYNN_TURBULENCE_ORACLE_ASSET.sha256 == (
        "68671e652ea0ad871ae02d5f2b84e0906888aa8dce9f2b2935d98639d7852144"
    )
    assert MYNN_TURBULENCE_ORACLE_CASES == (
        "stable", "convective", "cloudy", "edmf_active",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "mynn" / "oracle" / "turbulence.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == MYNN_TURBULENCE_ORACLE_ASSET.bytes
    assert hashlib.sha256(payload).hexdigest() == (
        MYNN_TURBULENCE_ORACLE_ASSET.sha256
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        MYNN_TURBULENCE_ORACLE_CASES
    )


def test_mynn_predict_oracle_asset_and_column_inventory_are_pinned():
    assert MYNN_PREDICT_ORACLE_ASSET.bytes == 17_300
    assert MYNN_PREDICT_ORACLE_ASSET.sha256 == (
        "685d4780e32010f0a3e015108b4e5867bb82fe2d99b8e489b0f14bbba881f738"
    )
    assert MYNN_PREDICT_ORACLE_CASES == (
        "stable", "convective", "cloudy", "edmf_active",
    )
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "mynn" / "oracle" / "predict.csv"
    )
    payload = path.read_bytes()
    assert len(payload) == MYNN_PREDICT_ORACLE_ASSET.bytes
    assert hashlib.sha256(payload).hexdigest() == MYNN_PREDICT_ORACLE_ASSET.sha256
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48
    assert tuple(dict.fromkeys(row["case"] for row in rows)) == (
        MYNN_PREDICT_ORACLE_CASES
    )


def test_default_option_identity_resolves_omissions_without_mutation():
    noahmp = resolve_default_noahmp_options({"opt_run": 3})
    mynn = resolve_default_mynn_options({"bl_mynn_closure": 2.6})
    assert dict(noahmp) == dict(NOAHMP_NAMELIST_DEFAULTS)
    assert dict(mynn) == dict(MYNN_NAMELIST_DEFAULTS)
    with pytest.raises(TypeError):
        noahmp["opt_run"] = 1


@pytest.mark.parametrize(
    ("resolver", "override", "match"),
    (
        (resolve_default_noahmp_options, {"opt_run": 1}, "opt_run=1"),
        (resolve_default_noahmp_options, {"opt_run": True}, "opt_run=True"),
        (
            resolve_default_mynn_options,
            {"bl_mynn_edmf": 0},
            "bl_mynn_edmf=0",
        ),
        (
            resolve_default_mynn_options,
            {"bl_mynn_tkeadvect": 0},
            "bl_mynn_tkeadvect=0",
        ),
    ),
)
def test_nondefault_or_type_aliased_options_fail_closed(
    resolver, override, match,
):
    with pytest.raises(ValueError, match=match):
        resolver(override)


def test_unknown_option_fails_closed():
    with pytest.raises(ValueError, match="unknown Noah-MP options: opt_magic"):
        resolve_default_noahmp_options({"opt_magic": 1})


def test_reference_asset_validation_checks_size_then_hash(tmp_path):
    payload = tmp_path / "oracle.F"
    payload.write_bytes(b"pinned")
    asset = ReferenceAsset(
        "oracle.F", payload.stat().st_size,
        hashlib.sha256(payload.read_bytes()).hexdigest(),
    )
    assert validate_reference_assets(tmp_path, (asset,)) == (asset,)

    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="has 7 bytes; expected 6"):
        validate_reference_assets(tmp_path, (asset,))

    payload.write_bytes(b"badbad")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_reference_assets(tmp_path, (asset,))
