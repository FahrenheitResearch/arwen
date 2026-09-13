"""Declared layer water reaches the WRF door as volume fraction in Rust."""

import hashlib

import netCDF4
import numpy as np
import pytest

from gpuwm.ingest import wrfinput as wi
from test_analyzed_scalar_boundaries import _cfg, _input, _read


def _soil(path, cfg, units, thickness_units="m", thickness=None):
    _input(path, cfg)
    with netCDF4.Dataset(path, "a") as dataset:
        layers = len(dataset.dimensions["soil_layers_stag"])
        dz = dataset.createVariable("DZS", "f8", ("Time", "soil_layers_stag"))
        dz.units = thickness_units
        dz[:] = thickness if thickness is not None else np.arange(1, layers + 1) / 10
        for name in ("SMOIS", "SH2O"):
            field = dataset.createVariable(name, "f8", ("Time", "soil_layers_stag", "south_north", "west_east"))
            field.units = units
            field[:] = np.arange(1, layers + 1)[None, :, None, None] * (20 if name == "SMOIS" else 10)
    return path


@pytest.mark.parametrize("units", ["kg m-2", "kg/m^2", "kg m**-2", "mm"])
def test_declared_layer_mass_is_normalized_before_the_physics_door(tmp_path, units):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, units)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = _read(path, cfg)
    np.testing.assert_allclose(restored.raw["SMOIS"], .2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(restored.raw["SH2O"], .1, rtol=0, atol=1e-15)
    assert restored.soil_unit_conversions["SMOIS"]["target_units"] == "m3 m-3"
    assert restored.soil_unit_conversions["SMOIS"]["thickness_variable"] == "DZS"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_volumetric_water_is_unchanged_without_layer_conversion(tmp_path):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, "m3 m-3")
    with netCDF4.Dataset(path, "a") as dataset:
        dataset["SMOIS"][:] = .1234567
        dataset["SH2O"][:] = .1000234
        original = {name: np.asarray(dataset[name][0]) for name in ("SMOIS", "SH2O")}
    restored = _read(path, cfg)
    for name, value in original.items():
        np.testing.assert_array_equal(restored.raw[name], value)
    assert not restored.soil_unit_conversions


def test_conversion_does_not_hide_unphysical_volume_fraction_by_clipping(tmp_path):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, "mm")
    with netCDF4.Dataset(path, "a") as dataset:
        dataset["SMOIS"][:] *= 10
    restored = _read(path, cfg)
    np.testing.assert_allclose(restored.raw["SMOIS"], 2, rtol=0, atol=1e-14)


@pytest.mark.parametrize("thickness", [0.0, -0.1, np.nan])
def test_invalid_layer_geometry_fails_at_read_before_gpu(tmp_path, thickness):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, "mm", thickness=thickness)
    with pytest.raises(wi.netcdf_bridge.NetcdfDecodeError, match="thickness.*finite and positive"):
        _read(path, cfg)


def test_centimetre_thickness_is_not_mistaken_for_metres(tmp_path):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, "mm", thickness_units="cm")
    with netCDF4.Dataset(path, "a") as dataset:
        dataset["DZS"][:] *= 100
    restored = _read(path, cfg)
    np.testing.assert_allclose(restored.raw["SMOIS"], .2, rtol=0, atol=1e-15)


def test_metre_equivalent_water_depth_is_not_treated_as_millimetres(tmp_path):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, "m")
    with netCDF4.Dataset(path, "a") as dataset:
        for name in ("SMOIS", "SH2O"):
            dataset[name][:] /= 1000
    restored = _read(path, cfg)
    np.testing.assert_allclose(restored.raw["SMOIS"], .2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(restored.raw["SH2O"], .1, rtol=0, atol=1e-15)
    assert restored.soil_unit_conversions["SMOIS"]["source_units"] == "m"


def test_packed_layer_geometry_is_unpacked_before_water_conversion(tmp_path):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, "kg m-2")
    with netCDF4.Dataset(path, "a") as dataset:
        dz = dataset["DZS"]
        dz.setncatts({"scale_factor": .01, "add_offset": .1})
        dz.set_auto_maskandscale(False)
        dz[:] = [0, 10, 20, 30]
    restored = _read(path, cfg)
    np.testing.assert_allclose(restored.raw["SMOIS"], .2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(restored.raw["SH2O"], .1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(restored.soil_unit_conversions["SMOIS"]["thickness_m"],
                               [.1, .2, .3, .4], rtol=0, atol=1e-15)


def test_dimension_mismatch_cannot_assign_water_to_another_layers_thickness(tmp_path):
    cfg = _cfg()
    path = _soil(tmp_path / "input", cfg, "mm")
    with netCDF4.Dataset(path, "a") as dataset:
        dataset.renameDimension("soil_layers_stag", "unrelated")
        dataset.createDimension("soil_layers_stag", len(dataset.dimensions["unrelated"]))
    with pytest.raises(ValueError, match="dimensions|shape|geometry"):
        _read(path, cfg)
