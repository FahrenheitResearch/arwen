"""AirNow station seam and Rust front-door checks."""
import json
from pathlib import Path
import subprocess
import pytest
from gpuwm.obs.aq_airnow import AIRNOW, AirNowStationSource
from gpuwm.obs.aq_table import VARIABLE_TABLE

DATA = Path(__file__).parent / "data" / "airnow"


def test_station_source():
    source = AirNowStationSource(DATA / "airnow.csv")
    obs = source.observations(["2025-07-30T21:00:00"])
    assert obs.reports
    assert obs.reports[0].values["ozone_mole_fraction"] == 30.0
    assert len({s.station_id for s in obs.stations}) == len(obs.stations)
    assert source.verify(obs.provenance)
    assert source.observations(["2025-07-30T22:00:00"]).reports == ()
    assert source.valid_times() == ("2025-07-30T21:00:00",)


def test_table_tampering_fails(tmp_path):
    table = tmp_path / "airnow.csv"
    table.write_bytes((DATA / "airnow.csv").read_bytes() + b"corruption")
    table.with_suffix(".json").write_bytes((DATA / "airnow.json").read_bytes())
    with pytest.raises(ValueError, match="digest mismatch"):
        AirNowStationSource(table)


def test_vocabulary_mirror():
    rust = (Path(__file__).parents[1] / "tools/rustwx/crates/rw-obs/src/table.rs").read_text()
    for name, (_, lo, hi, error) in VARIABLE_TABLE.items():
        assert f'"{name}"' in rust
        assert f'{lo:.1f}, {hi:.1f}, {error:.1f}' in rust


def test_rust_table_when_binary_present(tmp_path):
    binary = AIRNOW.find()
    if binary is None:
        pytest.skip("rw_airnow is absent; build on the authorized CPU host")
    assert AIRNOW.probe(binary)[0]
    day = tmp_path / "2025/20250730"
    day.mkdir(parents=True)
    fixture = Path(__file__).parents[1] / "tools/rustwx/crates/rw-obs/tests/data/airnow"
    (day / "HourlyData_2025073021.dat").write_bytes((fixture / "hourly.dat").read_bytes())
    (day / "Monitoring_Site_Locations_V2.dat").write_bytes((fixture / "sites.dat").read_bytes())
    out = tmp_path / "airnow.csv"
    record = AIRNOW.run("table", ["--dir", str(tmp_path), "--start", "2025-07-30T21:00:00Z",
                                  "--end", "2025-07-30T21:00:00Z", "--out", str(out)],
                        schema="gpuwm-obs.airnow-table.v1", binary=binary)
    assert record["rows"] > 0
    assert AirNowStationSource(out).observations(["2025-07-30T21:00:00"]).reports
