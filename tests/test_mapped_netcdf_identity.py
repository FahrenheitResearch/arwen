"""CF standard_name is an identity, not a second hurdle.

Requiring the configured NAME and the ``standard_name`` to BOTH match defeats
the purpose of a CF standard name.  The standard name is the stable identity;
the variable name is a label the producer is free to change.  Under the old
conjunction a correctly self-describing file broke on a pure rename, which is
what a current Copernicus delivery does: ``pressure_level`` still declares
``standard_name = air_pressure``, and that could not rescue it.

The spellings and attribute values here are taken from a real CDS delivery
fetched 2026-08-14, not invented to match our own reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm.mapped_source import (
    NC_EVIDENCE_NAME,
    NC_EVIDENCE_STANDARD_NAME,
    _resolve_nc_dimension,
    _resolve_nc_variable,
    load_mapping,
)

REPO = Path(__file__).resolve().parent.parent
ERA5_CONFIG = REPO / "configs" / "rw-wps-era5-netcdf.mapping.json"


def _write(path: Path, *, vertical: str, vertical_standard: str | None,
           time_name: str, time_standard: str | None) -> Path:
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension(time_name, 1)
        dataset.createDimension(vertical, 4)
        dataset.createDimension("latitude", 3)
        dataset.createDimension("longitude", 3)
        time = dataset.createVariable(time_name, "i8", (time_name,))
        time.units = "seconds since 1970-01-01"
        if time_standard:
            time.standard_name = time_standard
        time[:] = [1590969600]
        level = dataset.createVariable(vertical, "f8", (vertical,))
        level.units = "hPa"
        if vertical_standard:
            level.standard_name = vertical_standard
        level[:] = [1000.0, 850.0, 500.0, 250.0]
        for name, unit, standard in (
            ("latitude", "degrees_north", "latitude"),
            ("longitude", "degrees_east", "longitude"),
        ):
            var = dataset.createVariable(name, "f8", (name,))
            var.units = unit
            var.standard_name = standard
            var[:] = np.linspace(1.0, 3.0, 3)
    return path


@pytest.fixture
def era5_mapping():
    return load_mapping(ERA5_CONFIG)


def _vertical(mapping):
    return mapping["coordinates"]["vertical"]["selector"]


def _time(mapping):
    return mapping["coordinates"]["time"]["selector"]


def test_current_producer_spelling_resolves(tmp_path, era5_mapping):
    """Direction A: today's CDS delivery."""

    source = _write(
        tmp_path / "current.nc",
        vertical="pressure_level", vertical_standard="air_pressure",
        time_name="valid_time", time_standard="time",
    )
    report: list[dict[str, object]] = []
    with netCDF4.Dataset(source) as dataset:
        name = _resolve_nc_variable(
            dataset, _vertical(era5_mapping), "vertical", report).name
        dimension, _ = _resolve_nc_dimension(
            dataset, _time(era5_mapping), "time", report)
    assert name == "pressure_level"
    assert dimension == "valid_time"
    assert [row["evidence"] for row in report] == [NC_EVIDENCE_NAME] * 2
    assert not any(row["drifted"] for row in report)


def test_legacy_producer_spelling_still_resolves(tmp_path, era5_mapping):
    """Direction B: files predating the rename are still in the wild."""

    source = _write(
        tmp_path / "legacy.nc",
        vertical="level", vertical_standard="air_pressure",
        time_name="time", time_standard="time",
    )
    with netCDF4.Dataset(source) as dataset:
        name = _resolve_nc_variable(
            dataset, _vertical(era5_mapping), "vertical").name
        dimension, _ = _resolve_nc_dimension(dataset, _time(era5_mapping), "time")
    assert name == "level"
    assert dimension == "time"


def test_standard_name_rescues_an_unlisted_spelling_and_reports_it(
    tmp_path, era5_mapping
):
    """The fix itself: identity from EITHER, and the rescue is reported."""

    source = _write(
        tmp_path / "third.nc",
        vertical="plev", vertical_standard="air_pressure",
        time_name="reference_time", time_standard="time",
    )
    report: list[dict[str, object]] = []
    with netCDF4.Dataset(source) as dataset:
        name = _resolve_nc_variable(
            dataset, _vertical(era5_mapping), "vertical", report).name
        dimension, _ = _resolve_nc_dimension(
            dataset, _time(era5_mapping), "time", report)
    assert name == "plev"
    assert dimension == "reference_time"
    assert [row["evidence"] for row in report] == [NC_EVIDENCE_STANDARD_NAME] * 2
    # Reported, not silent: the descriptor has drifted and a user must see it.
    assert all(row["drifted"] for row in report)
    assert report[0]["configured_names"] == ["level", "pressure_level"]


def test_absent_standard_name_refuses_rather_than_guessing(tmp_path, era5_mapping):
    """Direction C: nothing legitimate identifies it, so nothing may guess."""

    source = _write(
        tmp_path / "bare.nc",
        vertical="plev", vertical_standard=None,
        time_name="reference_time", time_standard=None,
    )
    with netCDF4.Dataset(source) as dataset:
        with pytest.raises(ValueError) as error:
            _resolve_nc_variable(dataset, _vertical(era5_mapping), "vertical")
        message = str(error.value)
        assert "resolved 0 NetCDF variables" in message
        assert "level" in message and "pressure_level" in message  # asked for
        assert "plev" in message                                   # file offers
        with pytest.raises(ValueError):
            _resolve_nc_dimension(dataset, _time(era5_mapping), "time")


def test_two_variables_with_one_standard_name_refuse_and_name_both(
    tmp_path, era5_mapping
):
    """Ambiguity is never resolved by picking one."""

    source = _write(
        tmp_path / "ambiguous.nc",
        vertical="plev", vertical_standard="air_pressure",
        time_name="valid_time", time_standard="time",
    )
    with netCDF4.Dataset(source, "a") as dataset:
        rival = dataset.createVariable("isobaric", "f8", ("plev",))
        rival.units = "hPa"
        rival.standard_name = "air_pressure"
        rival[:] = [1000.0, 850.0, 500.0, 250.0]
    with netCDF4.Dataset(source) as dataset:
        with pytest.raises(ValueError) as error:
            _resolve_nc_variable(dataset, _vertical(era5_mapping), "vertical")
    message = str(error.value)
    assert "resolved 2 NetCDF variables" in message
    assert "ambiguous, all of" in message
    assert "plev" in message and "isobaric" in message


def test_an_explicitly_named_variable_wins_over_a_standard_name_rival(
    tmp_path, era5_mapping
):
    """Precedence: a configured name that matches is authoritative."""

    source = _write(
        tmp_path / "precedence.nc",
        vertical="pressure_level", vertical_standard="air_pressure",
        time_name="valid_time", time_standard="time",
    )
    with netCDF4.Dataset(source, "a") as dataset:
        rival = dataset.createVariable("isobaric", "f8", ("pressure_level",))
        rival.units = "hPa"
        rival.standard_name = "air_pressure"
        rival[:] = [1000.0, 850.0, 500.0, 250.0]
    report: list[dict[str, object]] = []
    with netCDF4.Dataset(source) as dataset:
        name = _resolve_nc_variable(
            dataset, _vertical(era5_mapping), "vertical", report).name
    # The name match is unambiguous, so the standard_name rival never runs.
    assert name == "pressure_level"
    assert report[0]["evidence"] == NC_EVIDENCE_NAME


def test_the_shipped_era5_mapping_accepts_both_spellings(era5_mapping):
    """Accepting both beats trading one break for the other."""

    assert _vertical(era5_mapping)["name"] == ["level", "pressure_level"]
    assert _time(era5_mapping)["name"] == ["time", "valid_time"]
    moisture = era5_mapping["fields"]["volumetric_soil_moisture"]["selectors"]
    assert [selector["name"] for selector in moisture] == [
        ["SWVL1", "swvl1"], ["SWVL2", "swvl2"],
        ["SWVL3", "swvl3"], ["SWVL4", "swvl4"],
    ]


def test_a_selector_name_list_refuses_duplicates_and_emptiness(tmp_path):
    document = json.loads(ERA5_CONFIG.read_text(encoding="utf-8"))
    document["coordinates"]["vertical"]["selector"]["name"] = ["level", "level"]
    path = tmp_path / "duplicate.mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="repeats an accepted spelling"):
        load_mapping(path)

    document["coordinates"]["vertical"]["selector"]["name"] = []
    path = tmp_path / "empty.mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list of accepted spellings"):
        load_mapping(path)
