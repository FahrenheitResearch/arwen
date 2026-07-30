from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.soil import NoahSoilState, preprocess_noah_soil
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
    validate_soil_layer_contract,
)
from gpuwm.mapped_composition import load_composition
from gpuwm.static.lambert import LambertGrid


ROOT = Path(__file__).parents[1]
GFS_MAPPING = ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json"
GFS_COMPOSITION = ROOT / "configs" / "rw-wps-gfs-terrain.composition.json"
ERA5_MAPPING = ROOT / "configs" / "rw-wps-era5-netcdf.mapping.json"
ERA5_COMPOSITION = ROOT / "configs" / "rw-wps-era5-netcdf-terrain.composition.json"
ERA5_GRIB1_MAPPING = ROOT / "configs" / "rw-wps-era5-1974-probe.mapping.json"
ERA5_GRIB1_COMPOSITION = (
    ROOT / "configs" / "rw-wps-era5-1974-terrain.composition.json"
)


def _soil_contract(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["soil_layers"]


def test_runtime_frozen_soil_contract_remains_valid():
    """The immutable contract emitted by MappedCompositionBundle is runnable."""

    contract = MappingProxyType(_soil_contract(ERA5_COMPOSITION))
    validated = validate_soil_layer_contract(contract)
    assert validated["depth_units"] == "m"


def _write_contract_case(
    tmp_path: Path,
    composition: dict[str, object],
    mapping: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    composition_path = tmp_path / "composition.json"
    mapping_path = tmp_path / "mapping.json"
    composition_path.write_text(json.dumps(composition), encoding="utf-8")
    mapping_path.write_text(
        json.dumps(
            mapping
            if mapping is not None
            else json.loads(GFS_MAPPING.read_text(encoding="utf-8"))
        ),
        encoding="utf-8",
    )
    return composition_path, mapping_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda soil: soil["source_layers"][1].update(top=0.11),
            "gap",
        ),
        (
            lambda soil: soil["source_layers"][1].update(top=0.09),
            "overlap",
        ),
        (
            lambda soil: soil["source_layers"].__setitem__(
                slice(0, 2), list(reversed(soil["source_layers"][:2]))
            ),
            "ordered",
        ),
        (
            lambda soil: soil.update(depth_units="cm"),
            "depth_units",
        ),
        (
            lambda soil: soil["missing"]["ocean"].update(moisture=0.0),
            "missing policy",
        ),
        (
            lambda soil: soil["source_layers"][0]["selectors"][
                "soil_temperature"
            ].update(surprise=True),
            "keys incompatible",
        ),
        (
            lambda soil: soil["source_layers"][0]["selectors"].pop(
                "soil_temperature"
            ),
            "missing required key",
        ),
    ],
)
def test_soil_contract_rejects_gap_overlap_order_units_and_ocean_policy(
    tmp_path,
    mutation,
    message,
):
    composition = json.loads(GFS_COMPOSITION.read_text(encoding="utf-8"))
    mutation(composition["soil_layers"])
    composition_path, mapping_path = _write_contract_case(tmp_path, composition)
    with pytest.raises(ValueError, match=message):
        load_composition(composition_path, mapping_path)


def test_grib2_selector_order_is_bound_to_declared_depth_order(tmp_path):
    composition = json.loads(GFS_COMPOSITION.read_text(encoding="utf-8"))
    mapping = json.loads(GFS_MAPPING.read_text(encoding="utf-8"))
    for field_name in ("soil_temperature", "volumetric_soil_moisture"):
        selectors = mapping["fields"][field_name]["selectors"]
        selectors[0], selectors[1] = selectors[1], selectors[0]
    composition_path, mapping_path = _write_contract_case(
        tmp_path, composition, mapping,
    )
    with pytest.raises(ValueError, match="selector 0 differs.*declared soil depth"):
        load_composition(composition_path, mapping_path)


def test_soil_contract_rejects_one_selector_for_multiple_grib_layers(tmp_path):
    composition = json.loads(GFS_COMPOSITION.read_text(encoding="utf-8"))
    mapping = json.loads(GFS_MAPPING.read_text(encoding="utf-8"))
    for field_name in ("soil_temperature", "volumetric_soil_moisture"):
        mapping["fields"][field_name]["selectors"] = mapping["fields"][
            field_name
        ]["selectors"][:1]
    composition_path, mapping_path = _write_contract_case(
        tmp_path, composition, mapping,
    )
    with pytest.raises(ValueError, match="requires exactly 4 ordered direct selectors"):
        load_composition(composition_path, mapping_path)


@pytest.mark.parametrize(
    ("mapping_path", "composition_path"),
    [
        (ERA5_GRIB1_MAPPING, ERA5_GRIB1_COMPOSITION),
        (ERA5_MAPPING, ERA5_COMPOSITION),
    ],
)
def test_soil_contract_binds_grib1_and_netcdf_selector_identity_to_depth(
    tmp_path,
    mapping_path,
    composition_path,
):
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    selectors = mapping["fields"]["soil_temperature"]["selectors"]
    selectors[0], selectors[1] = selectors[1], selectors[0]
    written_composition, written_mapping = _write_contract_case(
        tmp_path, composition, mapping,
    )
    with pytest.raises(ValueError, match="selector 0 differs.*declared soil depth"):
        load_composition(written_composition, written_mapping)


def test_selector_binding_treats_optional_null_as_semantically_absent(tmp_path):
    composition = json.loads(ERA5_COMPOSITION.read_text(encoding="utf-8"))
    mapping = json.loads(ERA5_MAPPING.read_text(encoding="utf-8"))
    mapping["fields"]["soil_temperature"]["selectors"][0][
        "standard_name"
    ] = None
    composition_path, mapping_path = _write_contract_case(
        tmp_path, composition, mapping,
    )
    load_composition(composition_path, mapping_path)


def _surface_fields() -> dict[str, np.ndarray]:
    return {
        "LANDSEA": np.asarray([[1.0, 0.0], [1.0, 1.0]]),
        "SKINTEMP": np.asarray([[291.0, 285.0], [289.0, 288.0]]),
        "SST": np.asarray([[0.0, 286.0], [0.0, 0.0]]),
        "TMN": np.asarray([[286.0, 280.0], [285.0, 284.0]]),
    }


def _assert_soil_state_equal(left: NoahSoilState, right: NoahSoilState) -> None:
    for name in NoahSoilState.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_declarative_era5_remap_is_bit_identical_to_retired_named_packing():
    surface = _surface_fields()
    temperatures = [
        np.full((2, 2), value) for value in (290.0, 288.0, 285.0, 282.0)
    ]
    moistures = [
        np.full((2, 2), value) for value in (0.11, 0.18, 0.24, 0.30)
    ]
    legacy = dict(surface)
    for name, value in zip(
        ("ST000007", "ST007028", "ST028100", "ST100289"), temperatures,
    ):
        legacy[name] = value
    for name, value in zip(
        ("SM000007", "SM007028", "SM028100", "SM100289"), moistures,
    ):
        legacy[name] = value
    declared = {
        **surface,
        MAPPED_SOIL_TEMPERATURE: np.stack(temperatures),
        MAPPED_SOIL_MOISTURE: np.stack(moistures),
    }
    soil_type = np.full((2, 2), 6)
    expected = preprocess_noah_soil(legacy, soil_type=soil_type)
    actual = preprocess_noah_soil(
        declared,
        soil_type=soil_type,
        soil_layer_contract=_soil_contract(ERA5_COMPOSITION),
    )
    _assert_soil_state_equal(actual, expected)


def test_declarative_era5_full_horizontal_path_matches_retired_named_packing():
    latitude = np.linspace(34.0, 46.0, 9, dtype=np.float64)
    longitude = np.linspace(267.0, 283.0, 10, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    base = lat2 + 0.1 * lon2
    temperatures = tuple(base + offset for offset in (220.0, 218.0, 215.0, 212.0))
    moistures = tuple(
        np.full(base.shape, value, dtype=np.float64)
        for value in (0.11, 0.18, 0.24, 0.30)
    )
    common = {
        "LANDSEA": np.ones(base.shape, dtype=np.float64),
        "SKINTEMP": base + 221.0,
    }
    legacy_fields = dict(common)
    for name, value in zip(
        ("ST000007", "ST007028", "ST028100", "ST100289"), temperatures,
    ):
        legacy_fields[name] = value
    for name, value in zip(
        ("SM000007", "SM007028", "SM028100", "SM100289"), moistures,
    ):
        legacy_fields[name] = value
    declared_fields = {
        **common,
        MAPPED_SOIL_TEMPERATURE: np.stack(temperatures),
        MAPPED_SOIL_MOISTURE: np.stack(moistures),
    }
    valid_time = datetime(1974, 4, 3, 12)
    snapshot_options = {
        "valid_time": valid_time,
        "levels_hpa": np.array([1000.0], dtype=np.float64),
        "latitude": latitude,
        "longitude": longitude,
    }
    grid = LambertGrid(
        ref_lat=40.0,
        ref_lon=-85.0,
        truelat1=30.0,
        truelat2=60.0,
        stand_lon=-85.0,
        dx=100_000.0,
        dy=100_000.0,
        e_we=6,
        e_sn=5,
    )
    legacy_horizontal = interpolate_era5_to_lambert(
        Era5Snapshot(fields=legacy_fields, **snapshot_options),
        grid,
        backend="cpu",
        workers=2,
    )
    declared_horizontal = interpolate_era5_to_lambert(
        Era5Snapshot(fields=declared_fields, **snapshot_options),
        grid,
        backend="cpu",
        workers=2,
    )
    target_shape = legacy_horizontal.fields["LANDSEA"].shape
    soil_type = np.full(target_shape, 6)
    deep = np.full(target_shape, 284.0)
    expected = preprocess_noah_soil(
        legacy_horizontal.fields,
        soil_type=soil_type,
        deep_soil_temperature=deep,
    )
    actual = preprocess_noah_soil(
        declared_horizontal.fields,
        soil_type=soil_type,
        deep_soil_temperature=deep,
        soil_layer_contract=_soil_contract(ERA5_COMPOSITION),
    )
    _assert_soil_state_equal(actual, expected)


def test_declarative_gfs_remap_is_bit_identical_to_retired_named_packing():
    surface = _surface_fields()
    temperatures = [
        np.full((2, 2), value) for value in (289.0, 288.0, 287.0, 286.0)
    ]
    moistures = [
        np.full((2, 2), value) for value in (0.10, 0.15, 0.20, 0.25)
    ]
    legacy = dict(surface)
    for name, value in zip(
        ("GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200"),
        temperatures,
    ):
        legacy[name] = value
    for name, value in zip(
        ("GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200"),
        moistures,
    ):
        legacy[name] = value
    declared = {
        **surface,
        MAPPED_SOIL_TEMPERATURE: np.stack(temperatures),
        MAPPED_SOIL_MOISTURE: np.stack(moistures),
    }
    soil_type = np.full((2, 2), 6)
    expected = preprocess_noah_soil(legacy, soil_type=soil_type)
    actual = preprocess_noah_soil(
        declared,
        soil_type=soil_type,
        soil_layer_contract=_soil_contract(GFS_COMPOSITION),
    )
    _assert_soil_state_equal(actual, expected)


def test_declared_ocean_repair_accepts_ocean_mask_but_rejects_land_gap():
    fields = _surface_fields()
    temperature = np.full((4, 2, 2), 285.0)
    moisture = np.full((4, 2, 2), 0.2)
    temperature[:, 0, 1] = np.nan
    moisture[:, 0, 1] = np.nan
    fields[MAPPED_SOIL_TEMPERATURE] = temperature
    fields[MAPPED_SOIL_MOISTURE] = moisture
    state = preprocess_noah_soil(
        fields,
        soil_type=np.full((2, 2), 6),
        soil_layer_contract=_soil_contract(GFS_COMPOSITION),
    )
    np.testing.assert_array_equal(state.soil_temperature[:, 0, 1], 286.0)
    np.testing.assert_array_equal(state.soil_moisture[:, 0, 1], 1.0)

    invalid = copy.deepcopy(fields)
    invalid[MAPPED_SOIL_TEMPERATURE][:, 0, 0] = np.nan
    with pytest.raises(ValueError, match="on land"):
        preprocess_noah_soil(
            invalid,
            soil_type=np.full((2, 2), 6),
            soil_layer_contract=_soil_contract(GFS_COMPOSITION),
        )


def test_conservative_contract_supports_non_named_source_layer_geometry():
    contract = copy.deepcopy(_soil_contract(GFS_COMPOSITION))
    contract["source_layers"] = [
        {
            "top": top,
            "bottom": bottom,
            "selectors": {
                "soil_temperature": {
                    "format": "netcdf",
                    "name": f"soil_temperature_{index}",
                },
                "volumetric_soil_moisture": {
                    "format": "netcdf",
                    "name": f"soil_moisture_{index}",
                },
            },
        }
        for index, (top, bottom) in enumerate(
            (
                (0.0, 0.05),
                (0.05, 0.2),
                (0.2, 0.6),
                (0.6, 1.2),
                (1.2, 2.5),
            ),
            start=1,
        )
    ]
    validate_soil_layer_contract(contract)
    fields = _surface_fields()
    # A constant profile must be exactly invariant under conservative remap.
    fields[MAPPED_SOIL_TEMPERATURE] = np.full((5, 2, 2), 285.0)
    fields[MAPPED_SOIL_MOISTURE] = np.full((5, 2, 2), 0.2)
    state = preprocess_noah_soil(
        fields,
        soil_type=np.full((2, 2), 6),
        soil_layer_contract=contract,
    )
    np.testing.assert_allclose(state.soil_temperature[:, 0, 0], 285.0)
    np.testing.assert_allclose(state.soil_moisture[:, 0, 0], 0.2)


def test_generic_mapped_soil_arrays_cannot_run_without_their_contract():
    fields = _surface_fields()
    fields[MAPPED_SOIL_TEMPERATURE] = np.full((4, 2, 2), 285.0)
    fields[MAPPED_SOIL_MOISTURE] = np.full((4, 2, 2), 0.2)
    with pytest.raises(ValueError, match="explicit soil_layer_contract"):
        preprocess_noah_soil(fields, soil_type=np.full((2, 2), 6))
