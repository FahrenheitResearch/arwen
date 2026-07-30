from __future__ import annotations

from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from tools.repair_era5_appended_invariant_time import repair


def _source(path: Path, *, complete_history: bool = True) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("x", 3)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 1900-01-01 00:00:00"
        time.calendar = "gregorian"
        time[:] = netCDF4.date2num(
            datetime(1979, 1, 1), time.units, calendar=time.calendar,
        )
        utc = dataset.createVariable("utc_date", "i8", ("time",))
        utc[:] = [1979010100]
        data = dataset.createVariable("T", "f4", ("time", "x"))
        data[:] = [[271.0, 272.0, 273.0]]
        evidence = (
            "ncks -O -d time,12,12 source.1974040300_1974040323.nc\n"
            "ncks -O -d time,60,60 source.1974040100_1974043023.nc\n"
            "ncks -A invariant.1979010100_1979010100.nc"
        )
        dataset.history = evidence if complete_history else "ncks -A unrelated"


def test_repair_requires_history_and_changes_only_time_arrays(tmp_path):
    source = tmp_path / "source.nc"
    output = tmp_path / "corrected.nc"
    receipt_path = tmp_path / "receipt.json"
    _source(source)
    source_before = source.read_bytes()

    receipt = repair(
        source, output, receipt_path, datetime(1974, 4, 3, 12),
    )

    assert source.read_bytes() == source_before
    assert receipt["status"] == "PASS"
    assert receipt["protected_variable_hashes_unchanged"] is True
    assert receipt["protected_variable_count"] == 1
    with netCDF4.Dataset(output) as dataset:
        actual_time = netCDF4.num2date(
            dataset.variables["time"][:], dataset.variables["time"].units,
            calendar=dataset.variables["time"].calendar,
            only_use_cftime_datetimes=False,
        )[0]
        assert actual_time == datetime(1974, 4, 3, 12)
        assert int(dataset.variables["utc_date"][:][0]) == 1974040312
        np.testing.assert_array_equal(
            dataset.variables["T"][:], [[271.0, 272.0, 273.0]],
        )


def test_repair_rejects_unproven_requested_time(tmp_path):
    source = tmp_path / "source.nc"
    output = tmp_path / "corrected.nc"
    receipt_path = tmp_path / "receipt.json"
    _source(source, complete_history=False)

    with pytest.raises(ValueError, match="does not prove requested time"):
        repair(source, output, receipt_path, datetime(1974, 4, 3, 12))
    assert not output.exists()
    assert not receipt_path.exists()
