from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest

from tools.download_gfs_native_subset import (
    NOMADS_VARIABLES,
    PRESSURE_LEVELS_HPA,
    _forecast_hours,
    nomads_query,
)


def test_gfs_nomads_contract_selects_both_native_soil_products():
    assert "var_TSOIL" in NOMADS_VARIABLES
    assert "var_SOILW" in NOMADS_VARIABLES

    query = parse_qs(urlparse(nomads_query(
        datetime(2026, 7, 21, 12), 3,
        left_lon=250.0, right_lon=278.0,
        bottom_lat=22.0, top_lat=50.0,
    )).query, keep_blank_values=True)
    assert query["file"] == ["gfs.t12z.pgrb2.0p25.f003"]
    assert query["dir"] == ["/gfs.20260721/12/atmos"]
    assert query["var_TSOIL"] == ["on"]
    assert query["var_SOILW"] == ["on"]
    assert tuple(
        level for level in PRESSURE_LEVELS_HPA
        if query[f"lev_{level}_mb"] == ["on"]
    ) == PRESSURE_LEVELS_HPA
    assert query["lev_0-0.1_m_below_ground"] == ["on"]
    assert query["lev_1-2_m_below_ground"] == ["on"]


@pytest.mark.parametrize("raw", ("3,6", "0", "0,2", "0,3,7", "0,387"))
def test_gfs_subset_rejects_non_contract_forecast_series(raw):
    with pytest.raises(ValueError):
        _forecast_hours(raw)


def test_gfs_subset_accepts_hourly_and_three_hourly_series():
    assert _forecast_hours("0,1,2") == (0, 1, 2)
    assert _forecast_hours("0,3,6") == (0, 3, 6)
