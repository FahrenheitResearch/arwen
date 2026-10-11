"""Native contract controls for optional observation networks and exact windows.

These small controls are not model runs or forecast-skill evidence. They run
rw_verify and rw_scorecard, rather than mocking either scientific executable.
"""
import json
from pathlib import Path

from test_obs_scorecard_cli import invoke


def test_new_asos_fields_remain_compatible_with_existing_battery_seam():
    from gpuwm.verify.obs.contracts import StationReport, SCORED_SURFACE_VARIABLES
    report = StationReport(station_id="sample", valid_time="2030-01-01T00:00:00",
                           values={"temperature_2m":300.,"wind_gust_10m":10.,"precipitation_1h":2.54})
    assert report.values["precipitation_1h"] == 2.54
    assert SCORED_SURFACE_VARIABLES == ("temperature_2m","dewpoint_2m","wind_speed_10m")


def test_optional_network_seam_and_exact_accumulation_end(tmp_path):
    time = "2030-01-01T01:00:00Z"
    earlier = "2030-01-01T00:55:00Z"
    metrics = [
        {"id": "profile", "source": "rawinsonde", "quantity": "temperature", "units": "K", "required": True},
        {"id": "snow", "source": "snotel", "quantity": "snow_water_equivalent", "units": "mm", "required": True},
        {"id": "gust", "source": "asos", "quantity": "wind_gust_10m", "units": "m s-1", "required": True},
        {"id": "temp", "source": "asos", "quantity": "temperature_2m", "units": "K", "required": True},
        {"id": "rain", "source": "asos", "quantity": "precipitation_1h", "units": "mm"},
    ]
    requests = []
    for source, quantities, observation_time in [
        ("rawinsonde", {"temperature": (300., 301., 302., "K", 150., 350.)}, time),
        ("snotel", {"snow_water_equivalent": (10., 12., 14., "mm", 0., 5000.)}, time),
        ("asos", {"wind_gust_10m": (10., 11., 12., "m s-1", 0., 100.),
                  "temperature_2m": (300., 301., 302., "K", 233.15, 328.15),
                  "precipitation_1h": (1., 2., 3., "mm", 0., 750.)}, earlier),
    ]:
        observed = {q: data[0] for q, data in quantities.items()}
        obs = {"schema": "gpuwm-obs.station-observations.v1", "source": source,
               "units": {q: data[3] for q, data in quantities.items()},
               "provenance": {"sha256": "1" * 64, "is_stub": False,
                              "test_control": True},
               "stations": [{"station_id": "sample-850hpa" if source == "rawinsonde" else "sample",
                             "latitude": 40., "longitude": -100., "elevation_m": 100.}]}
        station = obs["stations"][0]["station_id"]
        obs["reports"] = [{"station_id": station, "valid_time": time,
                           "observation_time": observation_time, "values": observed}]
        obs_path = tmp_path / f"{source}-obs.json"
        obs_path.write_text(json.dumps(obs))
        arms = []
        for label, index in [("A", 1), ("B", 2)]:
            path = tmp_path / f"{source}-{label}.json"
            path.write_text(json.dumps({"valid_time": time, "init_time": "2030-01-01T00:00:00Z",
                                       "points": [{"station_id": station,
                                                   "values": {q: data[index] for q, data in quantities.items()}}]}))
            arms.append({"label": label, "kind": "points", "path": str(path)})
        request = {"schema": "gpuwm.verify-visuals.request.v1", "domain": "d01", "valid_time": time,
                   "arms": arms, "stations_path": str(obs_path), "station_source": source,
                   "station_quantities": [{"quantity": q, "label": q, "units": data[3],
                                           "minimum": data[4], "maximum": data[5],
                                           "exact_time": q == "precipitation_1h"}
                                          for q, data in quantities.items()]}
        request_path = tmp_path / f"{source}-request.json"
        request_path.write_text(json.dumps(request))
        requests.append(str(request_path))
    manifest = {"schema": "gpuwm.obs-scorecard.manifest.v1", "candidate": "A", "reference": "B",
                "cases": [{"id": "contract", "block": "contract", "requests": requests}], "metrics": metrics,
                "notes": ["Synthetic native contract controls, not forecast-skill evidence."]}
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(manifest))
    result = invoke(path, "--out", tmp_path / "out")
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "out/scorecard.json").read_text())
    rows = {r["metric"]: r for r in receipt["case_scores"]}
    assert (rows["profile"]["candidate"], rows["profile"]["reference"]) == (1., 2.)
    assert (rows["snow"]["candidate"], rows["snow"]["reference"]) == (2., 4.)
    assert (rows["gust"]["candidate"], rows["gust"]["reference"]) == (1., 2.)
    assert (rows["temp"]["candidate"], rows["temp"]["reference"]) == (1., 2.)
    assert rows["rain"]["status"] == "missing"
    assert rows["rain"]["count"] == 0
