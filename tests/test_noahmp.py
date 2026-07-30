"""Pinned parameter-table gates for the Noah-MP runtime port."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import pytest

from gpuwm.core.noahmp import (
    MPTABLE_GROUPS,
    SOIL_PARAMETER_NAMES,
    load_noahmp_parameters,
    parse_mptable,
    parse_soilparm,
)
from gpuwm.core.noahmp_mynn_contract import NOAHMP_PARAMETER_ASSETS


DATA_ROOT = Path(__file__).parents[1] / "gpuwm" / "data" / "noahmp"


def test_packaged_noahmp_parameter_bytes_are_exact_git_blobs():
    for asset in NOAHMP_PARAMETER_ASSETS:
        path = DATA_ROOT / Path(asset.relative_path).name
        payload = path.read_bytes()
        assert len(payload) == asset.bytes
        assert hashlib.sha256(payload).hexdigest() == asset.sha256
        assert b"\r" not in payload


def test_noahmp_parameter_bundle_parses_complete_wrf_inventory():
    bundle = load_noahmp_parameters()
    assert tuple(bundle.mptable) == MPTABLE_GROUPS
    assert sum(len(group.values) for group in bundle.mptable.values()) == 433
    assert {name: len(table.rows) for name, table in bundle.soil.items()} == {
        "STAS": 19, "STAS-RUC": 19,
    }
    assert len(bundle.general.slopes) == 9
    assert bundle.general.values["CSOIL_DATA"] == 2.0e6
    assert bundle.general.values["ZBOT_DATA"] == -8.0
    assert bundle.receipt["status"] == "PASS"
    assert len(bundle.receipt["assets"]) == 3


def test_noahmp_vegetation_datasets_match_wrf_dimensions_and_categories():
    usgs_categories, usgs = load_noahmp_parameters().vegetation_groups("USGS")
    modis_categories, modis = load_noahmp_parameters().vegetation_groups(
        "MODIFIED_IGBP_MODIS_NOAH"
    )
    assert usgs_categories.scalar("VEG_DATASET_DESCRIPTION") == "USGS"
    assert usgs_categories.scalar("NVEG") == 27
    assert modis_categories.scalar("NVEG") == 20
    assert usgs.scalar("ISURBAN") == 1
    assert modis.scalar("ISURBAN") == 13
    assert len(usgs.values["LAI_JUL"]) == 27
    assert len(modis.values["LAI_JUL"]) == 20
    with pytest.raises(ValueError, match="must be USGS"):
        load_noahmp_parameters().vegetation_groups("MODIS")


def test_soil_loader_retains_distinct_noahmp_and_ruc_parameter_sets():
    bundle = load_noahmp_parameters()
    assert len(SOIL_PARAMETER_NAMES) == 17
    assert bundle.soil["STAS"].column("BB")[:3] == (2.79, 4.26, 4.74)
    assert bundle.soil["STAS-RUC"].column("BB")[:3] == (4.05, 4.38, 4.9)
    assert bundle.soil["STAS"].rows[13].description == "WATER"
    assert bundle.soil["STAS-RUC"].rows[18].description == "WHITE SAND"


def test_custom_table_root_rejects_same_length_byte_substitution(tmp_path):
    for path in DATA_ROOT.glob("*.TBL"):
        shutil.copyfile(path, tmp_path / path.name)
    target = tmp_path / "MPTABLE.TBL"
    payload = bytearray(target.read_bytes())
    payload[payload.index(b"USGS")] = ord("X")
    target.write_bytes(payload)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_noahmp_parameters(tmp_path)


def test_restricted_table_parsers_fail_closed_on_structure_drift():
    with pytest.raises(ValueError, match="unterminated"):
        parse_mptable("&noahmp_usgs_veg_categories\nNVEG=27\n")
    with pytest.raises(ValueError, match="SOILPARM"):
        parse_soilparm("not the WRF table\n")
