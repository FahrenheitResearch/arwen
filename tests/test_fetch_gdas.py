"""GDAS: the GFS assimilation cycle through the certified GFS container.

GDAS publishes ``pgrb2.0p25`` -- the same 0.25-degree regular lat/lon
grid, the same variable and level codes, the same originating centre and
table versions as GFS.  Under gpuwm's certified selector it yields the
same 124-record census, in the same south-to-north simple-packed form.
So it rides the certified mapping, bridge and front door with a source
tag and nothing else; what differs is the grib-filter script and the
directory.

**Analysis only.**  Everything verified above was verified at f000.
NCEP tags the later GDAS forecast records with a different forecast
generating process, and the fail-closed ``gfs_grib2_bridge`` is
certified against the analysis tag -- so this ArWen serves the analysis
and refuses the rest up front rather than advertising past what the
bridge certifies.  f000 is an analysis rather than a forecast field,
which is the analysis-quality initial state hindcast work wants; no
comparative forecast-impact claim is made here.

The live smoke (``GPUWM_NETWORK_TESTS=1``) is what makes the container
claim honest rather than assumed -- it fetches one real f000 subset and
checks the census, the grid, the packing, and the front-door contract.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpuwm import cli
from gpuwm import fetch
from tools import download_gfs_native_subset as gfs_transport


def test_gdas_rides_the_gfs_container_declaration():
    assert fetch.GFS_CONTAINER_SOURCES == ("gfs", "gdas")
    assert set(gfs_transport.NOMADS_MODELS) == {"gfs", "gdas"}
    # One selector, two publishers: the variable/level declaration is
    # shared, so the two cannot drift apart on what a subset contains.
    assert (gfs_transport.NOMADS_MODELS["gdas"]["script"]
            == "filter_gdas_0p25.pl")


def test_the_query_differs_only_in_naming():
    box = dict(left_lon=260.0, right_lon=270.0, bottom_lat=30.0,
               top_lat=40.0)
    cycle = datetime(2026, 7, 29, 12)
    gfs = gfs_transport.nomads_query(cycle, 0, model="gfs", **box)
    gdas = gfs_transport.nomads_query(cycle, 0, model="gdas", **box)
    # Everything after the script name and the file/dir stems is the
    # same declaration, character for character.
    def selector(url: str) -> list[str]:
        return [part for part in url.split("&")
                if not part.startswith(("http", "file=", "dir="))]
    assert selector(gfs) == selector(gdas)
    assert "filter_gdas_0p25.pl" in gdas
    assert "file=gdas.t12z.pgrb2.0p25.f000" in gdas
    assert "gdas.20260729%2F12%2Fatmos" in gdas


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown NOMADS model"):
        gfs_transport.nomads_query(
            datetime(2026, 7, 29, 12), 0, model="gefs",
            left_lon=260.0, right_lon=270.0, bottom_lat=30.0, top_lat=40.0)


def test_the_object_urls_and_cycle_cadence_follow_the_source():
    cycle = datetime(2026, 7, 29, 12)
    assert fetch.gfs_object_url(cycle, 0, "gdas").endswith(
        "gdas.20260729/12/atmos/gdas.t12z.pgrb2.0p25.f000")
    assert "/gdas." in fetch.gfs_object_url(cycle, 3, "gdas")
    # GDAS runs on the GFS synoptic grid, so the cadence rule is shared.
    assert fetch.parse_cycle("2026-07-29T12", "gdas") == cycle
    with pytest.raises(ValueError, match="GDAS cycles run at 00/06/12/18"):
        fetch.parse_cycle("2026-07-29T13", "gdas")


def test_the_certified_gdas_ladder_is_the_analysis_alone():
    """The scope constant and the ladder agree, and nothing widens it."""

    assert fetch.GDAS_MAX_FORECAST_HOUR == 0
    assert fetch.GDAS_CERTIFIED_HOURS == (0,)
    assert fetch.gdas_forecast_hours(0) == (0,)
    for beyond in (1, 3, 6, 9, 12):
        with pytest.raises(ValueError, match="ANALYSIS-ONLY"):
            fetch.gdas_forecast_hours(beyond)


def test_a_gdas_request_past_the_analysis_refuses_up_front(tmp_path,
                                                           capsys):
    """Refused at the CLI, before any download, in capability wording."""

    out = tmp_path / "gdas"
    rc = cli.main(["fetch", "--source", "gdas", "--cycle", "2026-07-29T12",
                   "--hours", "12", "--area", "30,-100,40,-90",
                   "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    # Says what IS certified, and that f012 is not it.
    assert "ANALYSIS-ONLY" in err
    assert "f000" in err and "f012" in err
    # ...says why the boundary is where it is...
    assert "gfs_grib2_bridge" in err
    # ...and says what to reach for instead.
    assert "--hours 0" in err
    assert "--source gfs" in err
    # Nothing was created, because nothing was attempted.
    assert not out.exists()


def test_the_library_boundary_refuses_the_same_request(tmp_path):
    """A caller reaching fetch_gfs directly hits the same gate."""

    with pytest.raises(ValueError, match="ANALYSIS-ONLY"):
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 29, 12), hours=(0, 3),
            area=fetch.parse_area("30,-100,40,-90"),
            out=tmp_path / "gdas", source="gdas", progress=lambda _: None)


def test_the_fetch_hint_table_accepts_gdas():
    fetch.validate_fetch_hints({"source": "gdas"}, source="gdas")
    fetch.validate_fetch_hints({"source": "gdas", "hours": 0},
                               source="gdas")
    with pytest.raises(ValueError):
        fetch.validate_fetch_hints({"source": "gefs"}, source="gefs")


def test_a_config_hint_advertising_a_gdas_forecast_is_refused():
    """The config surface is gated too, in the same words."""

    with pytest.raises(ValueError, match="ANALYSIS-ONLY"):
        fetch.validate_fetch_hints({"source": "gdas", "hours": 6},
                                   source="experiment.toml")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")
def test_live_gdas_f000_matches_the_certified_gfs_container(tmp_path):
    """One real GDAS analysis, checked against the GFS container claim.

    Fetches the f000 analysis -- the whole certified ladder -- over a
    10x10 degree box, a few hundred kilobytes, then asserts the census,
    the record bar, the manifest, and that the series/front-door seam
    the certified GFS bridge consumes is authored under the gdas names.
    """

    # Eight hours back: the assimilation cycle has certainly published.
    moment = datetime.now(timezone.utc) - timedelta(hours=8)
    cycle = datetime(moment.year, moment.month, moment.day,
                     (moment.hour // 6) * 6)
    out = tmp_path / "gdas"
    said: list[str] = []
    manifest = fetch.fetch_gfs(
        cycle=cycle, hours=fetch.GDAS_CERTIFIED_HOURS,
        area=fetch.parse_area("30,-100,40,-90"),
        out=out, source="gdas", progress=said.append)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source"] == "gdas"
    subsets = [item for item in payload["files"]
               if item["role"] == "gdas-subset"]
    assert [item["forecast_hour"] for item in subsets] == [0]
    for item in subsets:
        assert item["name"].startswith("gdas.t")
        assert (out / item["name"]).is_file()
        # The certified census, verified against the file itself.
        assert fetch.count_grib2_messages(out / item["name"]) == 124

    # The record bar was derived from the live GDAS index and agreed
    # with the certified GFS constant -- that agreement IS the container
    # claim, so it is the assertion worth making.
    bar = payload["record_bars"][0]
    assert bar["certified"] == 124
    assert bar["derived"] == 124, (
        "the live GDAS inventory no longer yields the certified GFS "
        "census; the container has diverged and the mapping must be "
        "re-certified rather than reused")
    assert bar["inventory_change_accepted"] is False

    # The ingest seam: gdas-series.tsv under the gdas names, which the
    # unchanged gfs_grib2_bridge consumes.
    series = out / "gdas-series.tsv"
    assert series.is_file()
    rows = [line.split("\t") for line in
            series.read_text(encoding="utf-8").splitlines()]
    assert [row[0] for row in rows] == ["0"]
    assert all(row[1].startswith("gdas.t") for row in rows)

    # Front-door dry contract check: author the manifest against a
    # stand-in bridge and confirm every role binds by digest.
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stand-in for the digest binding")
    namelist = tmp_path / "namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    front, digest = fetch.author_gfs_front_door_manifest(
        out=out, bridge=bridge, wps_namelist=namelist,
        experiment_config=config, source="gdas", progress=said.append)
    document = json.loads(front.read_text(encoding="utf-8"))
    assert document["schema"] == fetch.GFS_FRONT_DOOR_MANIFEST_SCHEMA
    assert document["source"]["model"] == "GDAS"
    assert document["source"]["product"] == "pgrb2.0p25"
    assert set(document["files"]) == {
        "series", "bridge", "wps_namelist", "experiment_config",
        "grib-f000"}
    assert len(digest) == 64
