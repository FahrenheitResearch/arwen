from copy import deepcopy

import pytest

from tools.da_case_manifest import build_case, select_objects, utc


def _case():
    return {"id": "e4", "event": "event-id", "latitude": 31.0,
            "longitude": -101.0, "radar_site": "KSJT",
            "hrrr_cycle": "2026-05-23T17:00:00Z",
            "da_start": "2026-05-23T22:29:00Z",
            "cutoff": "2026-05-24T00:29:04Z",
            "forecast_fork": "2026-05-24T00:29:00Z",
            "analysis_count": 25, "hrrr_forecast_hours": list(range(15))}


def _object(bucket, key):
    return {"bucket": bucket, "key": key, "bytes": 10,
            "etag": "listing-etag", "last_modified": "listing-time"}


def _inventory():
    prefix = "hrrr.20260523/conus/hrrr.t17z."
    objects = [_object("noaa-hrrr-bdp-pds", prefix + f"wrfprsf{h:02d}.grib2")
               for h in range(15)]
    objects += [_object("noaa-hrrr-bdp-pds", prefix + f"wrf{kind}f00.grib2")
                for kind in ("nat", "sfc")]
    objects += [
        _object("unidata-nexrad-level2", "2026/05/23/KSJT/KSJT20260523_223000_V06"),
        _object("unidata-nexrad-level2", "2026/05/24/KSJT/KSJT20260524_003000_V06"),
        _object("unidata-nexrad-level2", "2026/05/23/KMAF/KMAF20260523_223000_V06"),
        _object("noaa-mrms-pds", "CONUS/PrecipRate_00.00/20260524/MRMS_20260524-062900.grib2.gz"),
        _object("noaa-mrms-pds", "CONUS/PrecipRate_00.00/20260524/MRMS_20260524-063500.grib2.gz"),
    ]
    return {"objects": objects, "coverage": [{"case": "e4", "source": "hrrr", "missing": []}]}


def test_inventory_uses_exact_forcing_radar_site_and_midnight_windows():
    result = select_objects(_case(), _inventory())
    assert len(result) == 19
    keys = {item["key"] for item in result}
    assert any("20260524-062900" in key for key in keys)
    assert not any("KMAF" in key or "003000_V06" in key
                   or "20260524-063500" in key for key in keys)


def test_missing_forcing_object_refuses_instead_of_inventing_identity():
    inventory = _inventory()
    inventory["objects"].pop(2)
    with pytest.raises(ValueError, match="required HRRR objects missing.*wrfprsf02"):
        select_objects(_case(), inventory)


def test_case_manifest_leaves_prepared_receipts_and_slots_missing():
    case, inventory = build_case(_case(), {"nz": 49}, _inventory(), "actual-inventory-digest")
    assert case["status"] == "ACQUISITION_READY_RUN_MANIFEST_BLOCKED"
    assert case["verification"]["hourly_rain_windows"][-1] == {
        "lead_hours": 6, "start": "2026-05-24T05:29:00Z", "end": "2026-05-24T06:29:00Z"}
    assert "proof_sha256" not in case
    assert "slots" not in case
    assert inventory["bytes"] == 190
    assert {item["id"] for item in case["missing"]} == {
        "radar_slots", "prepared_authority", "surface_record", "footprints",
        "truth_support", "historical_research_arm"}


def test_naive_clocks_and_short_forcing_are_refused():
    with pytest.raises(ValueError, match="timezone"):
        utc("2026-05-23T17:00:00")
    short = deepcopy(_case())
    short["hrrr_forecast_hours"] = list(range(9))
    with pytest.raises(ValueError, match="does not cover"):
        select_objects(short, _inventory())
