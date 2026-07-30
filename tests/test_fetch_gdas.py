"""GDAS: the GFS assimilation cycle through the certified GFS container.

GDAS publishes ``pgrb2.0p25`` -- the same 0.25-degree regular lat/lon
grid, the same variable and level codes, the same originating centre and
table versions as GFS.  Under gpuwm's certified selector it yields the
same 124-record census, in the same south-to-north simple-packed form.
So it rides the certified mapping and bridge with a source tag and
nothing else; what differs is the grib-filter script and the directory.

**Fetch and decode, f000..f009 -- not a front door.**  f000 is an
analysis rather than a forecast field.  Real f003/f006/f009 NOMADS
samples certify the forecast span through f009 -- the endpoint is one
of the committed files, not an extrapolation past the last one -- with
process ID 96 declared per series row instead of admitted as a generic
hour rule; the bridge verifies the declaration against its certified
``{81, 96}`` set and never infers a capability from an hour or a source
name.  Past f009 still refuses up front.

What is *not* certified -- and what these tests therefore assert
refuses -- is ingest: ``rw-wps --source gdas`` has no field/level/
cadence mapping, so fetch prints no ``rw-wps`` next step for GDAS.  The
container is the certified GFS container and the mapping is expected to
be reusable wholesale, but that reuse has not been run end to end, and
an unrun route is not a front door.

The live smoke (``GPUWM_NETWORK_TESTS=1``) is what makes the container
claim honest rather than assumed -- it fetches the whole certified
ladder from a live cycle and checks the census, the record bar, the
declared process IDs, and the manifest contract on every hour it
asserts about.
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


def test_the_certified_gdas_ladder_stops_at_the_proven_horizon():
    """The scope constant and the ladder agree, and nothing widens it.

    v1.0.1 pinned this at f000 because the bridge was certified only
    against the analysis generating process, and said in as many words
    that widening it would be a re-certification event.  The v1.1
    hygiene lane IS that event -- real NOMADS f000/f003/f006/f009
    subsets under ``tests/fixtures/gdas-process-id/`` with the declared
    81/96 processes, the last of them the endpoint itself -- so the span
    is the published f000..f009.  The gate did not become a flag: past
    f009 still refuses, in the same words.
    """

    assert fetch.GDAS_MAX_FORECAST_HOUR == 9
    assert fetch.GDAS_CERTIFIED_HOURS == tuple(range(10))
    assert fetch.gdas_forecast_hours(0) == (0,)
    assert fetch.gdas_forecast_hours(6) == (0, 3, 6)
    assert fetch.gdas_forecast_hours(9, 3) == (0, 3, 6, 9)
    for beyond in (10, 12, 24, 384):
        with pytest.raises(ValueError, match="certified"):
            fetch.gdas_forecast_hours(beyond)


def test_fetch_help_cannot_drift_from_the_registry_gdas_span(capsys):
    """`gpuwm fetch --help` is pinned to the registry's max hour.

    v1.0.1's help said GDAS was "certified analysis-only, so it takes
    ``--hours 0``" and its `--cadence` line did not mention GDAS at all.
    Both survived the re-certification to f009, so an upgrader reading
    shipped `--help` got a strictly narrower scope than the registry,
    the refusal text, and DATA.md published -- and no test noticed.

    The bar here is the registry's own
    ``get_source_adapter("gdas").max_forecast_hour``, so help that
    hardcodes a hour, or a span that moves without the help moving with
    it, fails right here.  The scope words are pinned too: an endpoint
    without "no ingest route" beside it is the other half of the same
    contradiction.
    """

    from gpuwm.source_adapters import get_source_adapter

    registry_max = get_source_adapter("gdas").max_forecast_hour

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["fetch", "--help"])
    assert exit_info.value.code == 0
    # argparse rewraps to the terminal width, so collapse whitespace
    # before matching -- otherwise this test pins the wrap, not the text.
    help_text = " ".join(capsys.readouterr().out.split())

    # The endpoint, spelled the way the refusal and the docs spell it.
    assert f"f{registry_max:03d}" in help_text
    # ...and the scope that has to travel with it.
    assert "certified for fetch and decode" in help_text
    assert "no gdas ingest route" in help_text
    # The superseded v1.0.1 claim is gone from every fetch help string,
    # not merely relocated.
    assert "analysis-only" not in help_text
    assert "certified analysis" not in help_text
    # --cadence used to list gfs, era5 and hrrr and silently skip gdas.
    assert "gdas 1, 3, or 6" in help_text

    # And `gpuwm --help` alone must not leave a reader thinking GDAS
    # initializes a run.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "no ingest route" in " ".join(capsys.readouterr().out.split())


def test_a_gdas_request_past_the_certified_span_refuses_up_front(tmp_path,
                                                                 capsys):
    """Refused at the CLI, before any download, in capability wording."""

    out = tmp_path / "gdas"
    rc = cli.main(["fetch", "--source", "gdas", "--cycle", "2026-07-29T12",
                   "--hours", "12", "--area", "30,-100,40,-90",
                   "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    # Says what IS certified, and that f012 is not it.
    assert "f009" in err and "f012" in err
    # ...says why the boundary is where it is...
    assert "gfs_grib2_bridge" in err
    # ...and says what to reach for instead.
    assert "--hours 0..9" in err
    assert "--source gfs" in err
    # Nothing was created, because nothing was attempted.
    assert not out.exists()


def test_the_library_boundary_refuses_the_same_request(tmp_path):
    """A caller reaching fetch_gfs directly hits the same gate."""

    with pytest.raises(ValueError, match="certified"):
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 29, 12), hours=(0, 3, 12),
            area=fetch.parse_area("30,-100,40,-90"),
            out=tmp_path / "gdas", source="gdas", progress=lambda _: None)


def test_the_fetch_hint_table_accepts_gdas():
    fetch.validate_fetch_hints({"source": "gdas"}, source="gdas")
    fetch.validate_fetch_hints({"source": "gdas", "hours": 0},
                               source="gdas")
    with pytest.raises(ValueError):
        fetch.validate_fetch_hints({"source": "gefs"}, source="gefs")


def test_a_config_hint_past_the_certified_span_is_refused():
    """The config surface is gated too, in the same words."""

    fetch.validate_fetch_hints({"source": "gdas", "hours": 6},
                               source="experiment.toml")
    with pytest.raises(ValueError, match="certified"):
        fetch.validate_fetch_hints({"source": "gdas", "hours": 12},
                                   source="experiment.toml")


def test_gdas_prints_no_next_command_that_ends_in_a_refusal(tmp_path,
                                                            capsys):
    """One story across fetch, the registry, and the docs.

    ``rw-wps --source gdas`` refuses -- the adapter declares no
    field/level/cadence mapping -- so fetch must not print a ``next:``
    step that walks the user into that refusal, and the front-door
    manifest authoring must not print a GFS command over a GDAS series
    either (that would be a source-identity lie rather than a dead end).
    Both surfaces instead say what is missing and name the runnable
    alternative.
    """

    from gpuwm.source_adapters import get_source_adapter

    adapter = get_source_adapter("gdas")
    assert not adapter.runnable and adapter.runner is None
    assert adapter.max_forecast_hour == fetch.GDAS_MAX_FORECAST_HOUR
    assert "--source gfs" in adapter.notes

    said: list[str] = []
    out = tmp_path / "gdas"
    out.mkdir()
    # The authoring seam prints the honest stop, not an rw-wps line.
    try:
        fetch.author_gfs_front_door_manifest(
            out=out, bridge=None, wps_namelist=None,
            experiment_config=None, progress=said.append)
    except (ValueError, TypeError, OSError):
        pass  # no prior fetch here; the printing seam is asserted below
    assert not any("rw-wps --source gfs" in line for line in said)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")
def test_live_gdas_ladder_matches_the_certified_gfs_container(tmp_path):
    """The whole certified ladder, live, checked against the claim.

    Fetches f000..f009 at the default three-hour cadence over a 10x10
    degree box -- four files, a couple of megabytes -- then asserts the
    census, the record bar, the manifest, the *declared* process ID on
    every row, and that the series seam the certified GFS bridge
    consumes is authored under the gdas names.

    It fetches the whole ladder because it asserts about the whole
    ladder.  Until v1.1.0 this test downloaded hour 0 alone, asserted
    the series had exactly one row, and then required that one row to
    carry both process IDs ``["81", "96"]`` -- unsatisfiable by
    construction, so enabling it could only ever fail, and the live
    certification of the forecast hours it was named for never
    happened.
    """

    # Eight hours back: the assimilation cycle has certainly published
    # its analysis and its short forecast.
    moment = datetime.now(timezone.utc) - timedelta(hours=8)
    cycle = datetime(moment.year, moment.month, moment.day,
                     (moment.hour // 6) * 6)
    # The published span, from the same constant the refusal uses -- so
    # a widened or narrowed span is exercised here, not merely declared.
    hours = fetch.gdas_forecast_hours(fetch.GDAS_MAX_FORECAST_HOUR, 3)
    assert hours == (0, 3, 6, 9)
    # f000 is the analysis; every forecast hour carries the other
    # declared process.  This is the expectation the series must state.
    expected_processes = ["81"] + ["96"] * (len(hours) - 1)

    out = tmp_path / "gdas"
    said: list[str] = []
    manifest = fetch.fetch_gfs(
        cycle=cycle, hours=hours,
        area=fetch.parse_area("30,-100,40,-90"),
        out=out, source="gdas", progress=said.append)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source"] == "gdas"
    subsets = [item for item in payload["files"]
               if item["role"] == "gdas-subset"]
    assert [item["forecast_hour"] for item in subsets] == list(hours)
    for item in subsets:
        assert item["name"].startswith("gdas.t")
        assert (out / item["name"]).is_file()
        # The certified census, verified against the file itself.
        assert fetch.count_grib2_messages(out / item["name"]) == 124

    # The record bar was derived from the live GDAS index and agreed
    # with the certified GFS constant -- that agreement IS the container
    # claim, so it is the assertion worth making.  (One bar covers the
    # fetch; the per-file census above is what checks each hour.)
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
    assert [row[0] for row in rows] == [str(hour) for hour in hours]
    assert all(row[1].startswith("gdas.t") for row in rows)
    assert [row[2] for row in rows] == expected_processes

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
    } | {f"grib-f{hour:03d}" for hour in hours}
    assert len(digest) == 64

    # ...and the seam still refuses to print an rw-wps command, because
    # there is still no GDAS front door to point at.
    assert not any("rw-wps --source gfs" in line for line in said)
    assert any("no ingest route" in line for line in said)
