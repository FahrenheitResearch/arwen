"""CF georeference refusals for arbitrary mapped NetCDF input.

The descriptor is the authority for which variable is latitude; the file is
the evidence for what that variable holds.  A projected CF dataset's only 1-D
horizontal coordinates are ``x``/``y`` in metres.  Read as degrees they
mis-georeference an entire forecast without raising anything, which is the
silent-zero failure mode in georeference form.  These tests pin the refusals.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm.mapped_source import decode_mapped_source

from test_mapped_source import _mapping, _write_mapping, _write_source


def _projected(path: Path, *, grid_mapping_name: str = "lambert_conformal_conic") -> None:
    """Add a CF grid-mapping container to a source and point data at it."""

    with netCDF4.Dataset(path, "a") as dataset:
        container = dataset.createVariable("crs", "i4")
        container.grid_mapping_name = grid_mapping_name
        container.longitude_of_central_meridian = -97.0
        container.latitude_of_projection_origin = 38.5
        for name, variable in dataset.variables.items():
            if name in {"crs", "time", "level", "latitude", "longitude", "member"}:
                continue
            variable.grid_mapping = "crs"


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    mapping_path = tmp_path / "mapping.json"
    _write_mapping(mapping_path, _mapping())
    source = tmp_path / "source.nc"
    _write_source(source)
    return mapping_path, source


def test_netcdf_declared_projection_is_refused_by_name(tmp_path):
    mapping_path, source = _prepare(tmp_path)
    _projected(source)
    with pytest.raises(ValueError) as error:
        decode_mapped_source(mapping_path, [source])
    message = str(error.value)
    assert "lambert_conformal_conic" in message
    assert "latitude_longitude" in message


def test_netcdf_projection_refusal_names_the_claiming_variables(tmp_path):
    mapping_path, source = _prepare(tmp_path)
    _projected(source, grid_mapping_name="polar_stereographic")
    with pytest.raises(ValueError) as error:
        decode_mapped_source(mapping_path, [source])
    message = str(error.value)
    assert "polar_stereographic" in message
    assert "used by" in message


def test_netcdf_projection_axis_standard_name_is_refused_by_name(tmp_path):
    """The x/y-in-metres trap: 1-D projection axes read as degrees."""

    mapping_path = tmp_path / "mapping.json"
    mapping = _mapping()
    mapping["coordinates"]["horizontal"]["latitude"] = {
        "format": "netcdf", "name": "y_axis",
    }
    _write_mapping(mapping_path, mapping)
    source = tmp_path / "source.nc"
    _write_source(source)
    with netCDF4.Dataset(source, "a") as dataset:
        axis = dataset.createVariable("y_axis", "f8", ("y",))
        axis.units = "m"
        axis.standard_name = "projection_y_coordinate"
        axis[:] = np.linspace(-1.2e6, 1.2e6, dataset.dimensions["y"].size)
    with pytest.raises(ValueError) as error:
        decode_mapped_source(mapping_path, [source])
    message = str(error.value)
    assert "projection_y_coordinate" in message
    assert "y_axis" in message
    assert "degrees_north" in message


def test_netcdf_unlabelled_horizontal_coordinate_is_refused(tmp_path):
    """No CF units and no standard_name means identity cannot be confirmed."""

    mapping_path = tmp_path / "mapping.json"
    mapping = _mapping()
    mapping["coordinates"]["horizontal"]["longitude"] = {
        "format": "netcdf", "name": "x_axis",
    }
    _write_mapping(mapping_path, mapping)
    source = tmp_path / "source.nc"
    _write_source(source)
    with netCDF4.Dataset(source, "a") as dataset:
        axis = dataset.createVariable("x_axis", "f8", ("x",))
        axis.units = "m"
        axis[:] = np.linspace(-2.4e6, 2.4e6, dataset.dimensions["x"].size)
    with pytest.raises(ValueError) as error:
        decode_mapped_source(mapping_path, [source])
    message = str(error.value)
    assert "x_axis" in message
    assert "degrees_east" in message


def test_netcdf_out_of_range_latitude_is_refused(tmp_path):
    """CF-labelled but numerically impossible latitude still fails closed."""

    mapping_path, source = _prepare(tmp_path)
    with netCDF4.Dataset(source, "a") as dataset:
        dataset.variables["latitude"][:] = np.linspace(
            80.0, 1.0e5, dataset.dimensions["y"].size
        )
    with pytest.raises(ValueError) as error:
        decode_mapped_source(mapping_path, [source])
    assert "latitude" in str(error.value)
    assert "90" in str(error.value)


def test_regular_latitude_longitude_grid_still_decodes(tmp_path):
    """The guard must not refuse a legitimate geographic source."""

    mapping_path, source = _prepare(tmp_path)
    with netCDF4.Dataset(source, "a") as dataset:
        container = dataset.createVariable("crs", "i4")
        container.grid_mapping_name = "latitude_longitude"
    frames = decode_mapped_source(mapping_path, [source])
    assert frames
