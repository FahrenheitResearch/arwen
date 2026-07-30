"""``gpuwm fetch`` gates: URL/cycle/area resolution, manifests, ERA5.

Everything here is CPU + mocked transport.  The two transport seams the
fetch front door reuses (``tools.download_gfs_native_subset._download``
and ``tools.download_hrrr_native_subset._download_product``) are
monkeypatched to publish synthetic-but-envelope-valid GRIB payloads, so
these tests pin URL construction, cycle resolution, record-count
verification, resumability, and manifest correctness without a network.
One optional live smoke (``GPUWM_NETWORK_TESTS=1``) touches a single
small ``.idx`` object.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

import gpuwm.cli as cli
import gpuwm.fetch as fetch
from tools import download_gfs_native_subset as gfs_transport
from tools import download_hrrr_native_subset as hrrr_transport


# ---------------------------------------------------------------------------
# Synthetic-but-envelope-valid GRIB payloads
# ---------------------------------------------------------------------------

def _grib2_stream(messages: int) -> bytes:
    """A stream of minimal, envelope-valid GRIB2 messages."""
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _grib1_message(parameter: int, level_type: int, level: int,
                   when: datetime, *, edition: int = 1) -> bytes:
    """One envelope-valid GRIB1 analysis message with a real PDS."""
    pds = bytearray(28)
    pds[0:3] = (28).to_bytes(3, "big")
    pds[3] = 1      # parameter table version
    pds[4] = 98     # ECMWF
    pds[7] = 0      # no GDS/BMS
    pds[8] = parameter
    pds[9] = level_type
    pds[10:12] = level.to_bytes(2, "big")
    year_of_century = when.year % 100 or 100
    century = (when.year - 1) // 100 + 1
    pds[12:17] = bytes((year_of_century, when.month, when.day,
                        when.hour, when.minute))
    pds[17] = 1     # hours
    pds[18] = 0     # P1 = 0: analysis
    pds[20] = 0     # time range indicator 0
    pds[24] = century
    body = bytes(pds)
    total = 8 + len(body) + 4
    return (b"GRIB" + total.to_bytes(3, "big") + bytes((edition,))
            + body + b"7777")


_TIMES = (datetime(1974, 4, 3, 0), datetime(1974, 4, 3, 6))
_LADDER = (100, 200, 300, 500, 700, 850, 925, 1000)


def _era5_set(path: Path, *, soil_level_type: int = 1,
              drop: tuple[int, ...] = (), times=_TIMES,
              orography: bool = True) -> Path:
    payload = bytearray()
    for when in times:
        for parameter in fetch.ERA5_REQUIRED_PRESSURE:
            for level in _LADDER:
                payload += _grib1_message(parameter, 100, level, when)
        for parameter in fetch.ERA5_REQUIRED_SURFACE:
            if parameter in drop:
                continue
            level_type = (soil_level_type
                          if parameter in fetch.ERA5_SOIL_PARAMETERS else 1)
            payload += _grib1_message(parameter, level_type, 0, when)
    if orography:
        payload += _grib1_message(
            fetch.ERA5_OROGRAPHY_PARAMETER, 1, 0, times[0])
    path.write_bytes(bytes(payload))
    return path


# ---------------------------------------------------------------------------
# Area / cycle / hours parsing
# ---------------------------------------------------------------------------

def test_area_parses_corner_order_free_and_rejects_junk():
    area = fetch.parse_area("40,-90,30,-100")
    assert (area.lat_south, area.lon_west, area.lat_north,
            area.lon_east) == (30.0, -100.0, 40.0, -90.0)
    for bad in ("30,-100,40", "a,b,c,d", "30,-100,30,-90", "95,0,96,10"):
        with pytest.raises(ValueError):
            fetch.parse_area(bad)


def test_area_nomads_box_uses_0_360_and_widens_over_greenwich():
    box = fetch.parse_area("30,-100,40,-90").as_nomads()
    assert box == {"left_lon": 260.0, "right_lon": 270.0,
                   "bottom_lat": 30.0, "top_lat": 40.0}
    wrapped = fetch.parse_area("40,-10,50,10").as_nomads()
    assert (wrapped["left_lon"], wrapped["right_lon"]) == (0.0, 360.0)


def test_area_longitude_pair_wider_than_180_crosses_the_antimeridian():
    """Worldwide contract: 170,-170 is the 20-degree Pacific box over
    180E, never the silent 340-degree complement the old sorted()
    corners produced (that bug fetched the wrong data with no error)."""
    area = fetch.parse_area("30,170,50,-170")
    assert (area.lon_west, area.lon_east) == (170.0, -170.0)
    assert area.crosses_antimeridian
    # Order-free like every other corner pair.
    assert fetch.parse_area("50,-170,30,170").lon_west == 170.0
    # NOMADS serves the crossing box as a contiguous 0-360 subregion.
    box = area.as_nomads()
    assert (box["left_lon"], box["right_lon"]) == (170.0, 190.0)
    # CDS reads west > east as a crossing box, in the signed convention.
    assert area.as_cds() == [50.0, 170.0, 30.0, -170.0]
    # Non-crossing boxes are untouched.
    assert not fetch.parse_area("30,-100,40,-90").crosses_antimeridian
    # A full band (span 360) stays a full band.
    band = fetch.parse_area("-10,-180,10,180")
    assert not band.crosses_antimeridian
    assert band.as_nomads()["left_lon"] == 0.0
    assert band.as_nomads()["right_lon"] == 360.0


def test_point_box_wraps_across_the_antimeridian():
    area = fetch.area_from_point("-17.8,178.5", 500.0)
    assert area.crosses_antimeridian
    assert area.lon_west > 0.0 > area.lon_east
    box = area.as_nomads()
    assert box["left_lon"] < 180.0 < box["right_lon"]


def test_point_radius_becomes_a_bounding_box():
    area = fetch.area_from_point("35,-97.5", 300.0)
    assert area.lat_south < 35.0 < area.lat_north
    assert area.lon_west < -97.5 < area.lon_east
    # longitude half-width exceeds latitude half-width off the equator
    assert (area.lon_east - area.lon_west) > (
        area.lat_north - area.lat_south)
    with pytest.raises(ValueError, match="radius-km"):
        fetch.area_from_point("35,-97.5", -1.0)
    with pytest.raises(ValueError, match="pole"):
        fetch.area_from_point("89.9,0", 300.0)


def test_cycle_parsing_enforces_source_cadence():
    assert fetch.parse_cycle("2026-07-28T06", "gfs") == datetime(
        2026, 7, 28, 6)
    assert fetch.parse_cycle("2026-07-28T05", "hrrr") == datetime(
        2026, 7, 28, 5)
    with pytest.raises(ValueError, match="00/06/12/18"):
        fetch.parse_cycle("2026-07-28T05", "gfs")
    with pytest.raises(ValueError, match="YYYY-MM-DDTHH"):
        fetch.parse_cycle("2026-07-28 05:00", "hrrr")


def test_forecast_hour_ladders():
    assert fetch.gfs_forecast_hours(6, 3) == (0, 3, 6)
    assert fetch.gfs_forecast_hours(2, 1) == (0, 1, 2)
    with pytest.raises(ValueError, match="multiple"):
        fetch.gfs_forecast_hours(4, 3)
    with pytest.raises(ValueError, match="horizon"):
        fetch.gfs_forecast_hours(387, 3)
    assert fetch.hrrr_forecast_hours(2, datetime(2026, 7, 28, 5)) == (
        0, 1, 2)
    with pytest.raises(ValueError, match="horizon f18"):
        fetch.hrrr_forecast_hours(24, datetime(2026, 7, 28, 5))
    assert fetch.hrrr_forecast_hours(
        24, datetime(2026, 7, 28, 6))[-1] == 24


# ---------------------------------------------------------------------------
# Latest-cycle resolution off the AWS S3 listing (HEAD probes)
# ---------------------------------------------------------------------------

def test_latest_gfs_probes_synoptic_cycles_newest_first():
    probed = []

    def probe(url: str) -> bool:
        probed.append(url)
        return url == fetch.gfs_object_url(datetime(2026, 7, 27, 18), 6)

    cycle = fetch.resolve_latest_cycle(
        "gfs", 6, now=datetime(2026, 7, 28, 5, 30), probe=probe)
    assert cycle == datetime(2026, 7, 27, 18)
    # Newest candidate first (2026-07-28 00Z), then strictly older.
    assert probed[0] == fetch.gfs_object_url(datetime(2026, 7, 28, 0), 6)
    assert probed[-1].endswith("f006")


def test_latest_hrrr_skips_cycles_whose_horizon_is_too_short():
    def probe(url: str) -> bool:
        return True

    cycle = fetch.resolve_latest_cycle(
        "hrrr", 24, now=datetime(2026, 7, 28, 5, 30), probe=probe)
    # 05Z..01Z only reach f18; the newest extended cycle wins.
    assert cycle == datetime(2026, 7, 28, 0)


def test_latest_hrrr_requires_both_wrfnat_and_wrfprs():
    """A cycle whose final wrfprs (soil source) 404s must not win.

    During a live publication wrfnat can appear minutes before its
    wrfprs sibling; fetching needs both, so completeness probes both.
    """
    newest = datetime(2026, 7, 28, 5)
    probed = []

    def probe(url: str) -> bool:
        probed.append(url)
        # Newest cycle: wrfnat published, wrfprs not yet.
        if url == fetch.hrrr_object_url(newest, 2, "wrfprs"):
            return False
        return True

    cycle = fetch.resolve_latest_cycle(
        "hrrr", 2, now=datetime(2026, 7, 28, 5, 30), probe=probe)
    assert cycle == datetime(2026, 7, 28, 4)
    assert fetch.hrrr_object_url(newest, 2, "wrfnat") in probed
    assert fetch.hrrr_object_url(newest, 2, "wrfprs") in probed
    # The winning cycle's soil object was probed too, not assumed.
    assert fetch.hrrr_object_url(cycle, 2, "wrfprs") in probed


def test_latest_fails_closed_when_no_cycle_is_complete():
    with pytest.raises(RuntimeError, match="explicit --cycle"):
        fetch.resolve_latest_cycle(
            "gfs", 6, now=datetime(2026, 7, 28, 5, 30),
            probe=lambda url: False)
    with pytest.raises(ValueError, match="reanalysis"):
        fetch.resolve_latest_cycle("era5", 6)


# ---------------------------------------------------------------------------
# GFS fetch: URL construction, record-count gate, series, manifest, resume
# ---------------------------------------------------------------------------

def _fake_gfs_download(url: str, destination: Path, **kwargs) -> None:
    destination.write_bytes(_grib2_stream(fetch.GFS_SUBSET_RECORD_COUNT))


def test_fetch_gfs_writes_series_and_manifest(tmp_path, monkeypatch):
    urls = []

    def download(url, destination, **kwargs):
        urls.append(url)
        _fake_gfs_download(url, destination)

    monkeypatch.setattr(gfs_transport, "_download", download)
    out = tmp_path / "gfs"
    manifest_path = fetch.fetch_gfs(
        cycle=datetime(2026, 7, 28, 6), hours=(0, 3, 6),
        area=fetch.parse_area("30,-100,40,-90"), out=out,
        progress=lambda line: None)

    assert len(urls) == 3
    query = dict(parse_qsl(urlsplit(urls[0]).query, keep_blank_values=True))
    assert urlsplit(urls[0]).path.endswith("filter_gfs_0p25.pl")
    assert query["file"] == "gfs.t06z.pgrb2.0p25.f000"
    assert query["dir"] == "/gfs.20260728/06/atmos"
    assert (query["leftlon"], query["rightlon"]) == ("260", "270")
    assert (query["bottomlat"], query["toplat"]) == ("30", "40")
    assert query["var_TSOIL"] == "on" and query["lev_500_mb"] == "on"

    series = (out / "gfs-series.tsv").read_text().splitlines()
    assert series == [
        "0\tgfs.t06z.pgrb2.0p25.f000.subset.grib2",
        "3\tgfs.t06z.pgrb2.0p25.f003.subset.grib2",
        "6\tgfs.t06z.pgrb2.0p25.f006.subset.grib2",
    ]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == fetch.FETCH_MANIFEST_SCHEMA
    assert manifest["source"] == "gfs"
    assert manifest["cycle"] == "2026-07-28T06:00:00Z"
    assert manifest["forecast_hours"] == [0, 3, 6]
    assert manifest["area"]["lon_west"] == -100.0
    roles = [item["role"] for item in manifest["files"]]
    assert roles == ["gfs-subset"] * 3 + ["series"]
    for item in manifest["files"]:
        path = out / item["name"]
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(
            path.read_bytes()).hexdigest()
    assert manifest["payload_bytes"] == sum(
        item["bytes"] for item in manifest["files"])


def test_fetch_gfs_resumes_by_skipping_verified_files(tmp_path, monkeypatch):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    kwargs = dict(cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
                  area=fetch.parse_area("30,-100,40,-90"), out=out)
    lines: list[str] = []
    fetch.fetch_gfs(**kwargs, progress=lines.append)

    def refuse(url, destination, **kw):
        raise AssertionError("re-downloaded a complete file")

    monkeypatch.setattr(gfs_transport, "_download", refuse)
    lines.clear()
    manifest_path = fetch.fetch_gfs(**kwargs, progress=lines.append)
    assert all("skipped" in line for line in lines)
    assert manifest_path.is_file()


def test_fetch_gfs_fails_closed_on_record_count_drift(tmp_path, monkeypatch):
    def short(url, destination, **kwargs):
        destination.write_bytes(_grib2_stream(7))

    monkeypatch.setattr(gfs_transport, "_download", short)
    with pytest.raises(ValueError, match="expected 124"):
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
            area=fetch.parse_area("30,-100,40,-90"), out=tmp_path / "gfs",
            progress=lambda line: None)


def test_fetch_gfs_rejects_a_corrupt_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    out.mkdir()
    stale = out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2"
    stale.write_bytes(_grib2_stream(124) + b"junk")
    with pytest.raises(ValueError, match="GRIB indicator"):
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
            area=fetch.parse_area("30,-100,40,-90"), out=out,
            progress=lambda line: None)


def test_grib2_message_count_walks_envelopes_exactly(tmp_path):
    path = tmp_path / "stream.grib2"
    path.write_bytes(_grib2_stream(5))
    assert fetch.count_grib2_messages(path) == 5
    path.write_bytes(_grib2_stream(5)[:-4] + b"xxxx")
    with pytest.raises(ValueError, match="7777"):
        fetch.count_grib2_messages(path)
    with pytest.raises(ValueError, match="truncated"):
        truncated = tmp_path / "short.grib2"
        truncated.write_bytes(_grib2_stream(5)[:-2])
        fetch.count_grib2_messages(truncated)
    path.write_bytes(b"GRIB" + b"\x00\x00\x00\x01"
                     + (20).to_bytes(8, "big") + b"7777")
    with pytest.raises(ValueError, match="edition"):
        fetch.count_grib2_messages(path)


# ---------------------------------------------------------------------------
# HRRR fetch: product set, manifest, checksums, resume, area gate
# ---------------------------------------------------------------------------

def _fake_hrrr_product(request, *, workers, retries,
                       expected_count=-1):
    # The real record cardinalities: an undersized fake would trip the
    # completeness bar fetch_hrrr now applies to fresh downloads too.
    request.destination.write_bytes(_grib2_stream(
        hrrr_transport.SOIL_RECORD_COUNT if request.kind == "soil"
        else hrrr_transport.ATMOSPHERE_RECORD_COUNT))
    request.index_path.write_text("1:0:fixture\n", encoding="ascii")
    return {"kind": request.kind}


def test_fetch_hrrr_downloads_wrfnat_and_soil_products(tmp_path,
                                                       monkeypatch):
    seen = []

    def product(request, *, workers, retries, expected_count=-1):
        seen.append(request)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        progress=lambda line: None)

    assert [request.kind for request in seen] == [
        "atmosphere", "soil", "atmosphere", "soil"]
    assert seen[0].url.startswith(
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260728/conus/")
    assert seen[0].url.endswith("hrrr.t05z.wrfnatf00.grib2")
    assert seen[0].index_url.endswith(".idx")
    # Soil records come from the pressure file but publish as soilfNN.
    assert seen[1].url.endswith("hrrr.t05z.wrfprsf00.grib2")
    assert seen[1].destination.name == "hrrr.t05z.soilf00.grib2"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["source"] == "hrrr"
    assert [item["role"] for item in manifest["files"]] == [
        "atmosphere", "soil", "atmosphere", "soil", "checksums"]
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 4
    for line in sums:
        digest, name = line.split("  ")
        assert digest == hashlib.sha256(
            (out / name).read_bytes()).hexdigest()


def test_fetch_hrrr_skips_existing_verified_products(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None,
                  out=out)
    fetch.fetch_hrrr(**kwargs, progress=lambda line: None)

    def refuse(request, *, workers, retries):
        raise AssertionError("re-downloaded a complete product")

    monkeypatch.setattr(hrrr_transport, "_download_product", refuse)
    lines: list[str] = []
    fetch.fetch_hrrr(**kwargs, progress=lines.append)
    assert len(lines) == 4 and all("skipped" in line for line in lines)


def test_fetch_hrrr_redownloads_boundary_truncated_existing_file(
        tmp_path, monkeypatch):
    """A prefix truncated at a GRIB message boundary walks clean but
    fails the 561-record count: it must be re-downloaded, never blessed
    with a fresh digest."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                  out=out)
    fetch.fetch_hrrr(**kwargs, progress=lambda line: None)

    victim = out / "hrrr.t05z.wrfnatf00.grib2"
    complete = victim.read_bytes()
    message = len(complete) // hrrr_transport.ATMOSPHERE_RECORD_COUNT
    truncated = complete[:100 * message]  # clean EOF, wrong count
    victim.write_bytes(truncated)
    truncated_sha = hashlib.sha256(truncated).hexdigest()

    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    lines: list[str] = []
    manifest_path = fetch.fetch_hrrr(**kwargs, progress=lines.append)

    assert downloaded == [victim.name]  # the soil file stays skipped
    assert any("expected "
               f"{hrrr_transport.ATMOSPHERE_RECORD_COUNT}" in line
               for line in lines)
    # The truncated bytes were quarantined, not deleted and not blessed.
    rejected = list(out.glob(f"{victim.name}.rejected-*"))
    assert len(rejected) == 1
    assert rejected[0].read_bytes() == truncated
    assert victim.read_bytes() == complete
    manifest = json.loads(manifest_path.read_text())
    shas = {item["name"]: item["sha256"] for item in manifest["files"]}
    assert shas[victim.name] == hashlib.sha256(complete).hexdigest()
    assert truncated_sha not in shas.values()
    assert truncated_sha not in (out / "SHA256SUMS").read_text()


def test_fetch_hrrr_redownloads_on_prior_manifest_digest_mismatch(
        tmp_path, monkeypatch):
    """Right count, wrong bytes: the prior manifest digest is the
    authority, so the file is re-downloaded rather than re-blessed."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                  out=out)
    fetch.fetch_hrrr(**kwargs, progress=lambda line: None)

    victim = out / "hrrr.t05z.soilf00.grib2"
    complete = victim.read_bytes()
    tampered = bytearray(complete)
    tampered[6] ^= 0xFF  # discipline byte: envelope-valid, count intact
    victim.write_bytes(bytes(tampered))

    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    lines: list[str] = []
    fetch.fetch_hrrr(**kwargs, progress=lines.append)

    assert downloaded == [victim.name]
    assert any("prior fetch manifest" in line for line in lines)
    assert victim.read_bytes() == complete


def test_fetch_hrrr_rejects_areas_outside_conus():
    with pytest.raises(ValueError, match="CONUS"):
        fetch.fetch_hrrr(
            cycle=datetime(2026, 7, 28, 5), hours=(0, 1),
            area=fetch.parse_area("45,-10,55,10"), out=Path("unused"),
            progress=lambda line: None)


# ---------------------------------------------------------------------------
# ERA5: template emission + user-file validation
# ---------------------------------------------------------------------------

def test_era5_template_binds_area_times_levels_and_variables(tmp_path):
    area = fetch.parse_area("30,-100,40,-90")
    template = fetch.era5_request_template(
        cycle=datetime(1974, 4, 3, 0), hours=12, area=area)
    pressure, single = (item["request"] for item in template["requests"])
    assert template["requests"][0]["dataset"] == (
        "reanalysis-era5-pressure-levels")
    assert pressure["area"] == [40.0, -100.0, 30.0, -90.0]
    assert pressure["date"] == ["1974-04-03"]
    assert pressure["time"] == ["00:00", "06:00", "12:00"]
    assert len(pressure["pressure_level"]) == 37
    assert "relative_humidity" in pressure["variable"]
    assert "geopotential" in single["variable"]  # invariant orography
    assert "snow_depth" in single["variable"]
    assert {"soil_temperature_level_4",
            "volumetric_soil_water_layer_4"} <= set(single["variable"])
    with pytest.raises(ValueError, match="multiple"):
        fetch.era5_request_template(
            cycle=datetime(1974, 4, 3, 0), hours=5, area=area)


def test_era5_validation_passes_a_complete_set(tmp_path):
    path = _era5_set(tmp_path / "era5.grib")
    report = fetch.validate_era5_files((path,), expected_times=_TIMES)
    assert report.ok, report.format()
    text = report.format()
    assert "8 levels" in text and "fully covered" in text


def test_era5_validation_accepts_native_cds_soil_level_type(tmp_path):
    path = _era5_set(tmp_path / "era5.grib", soil_level_type=112)
    assert fetch.validate_era5_files((path,)).ok


def test_era5_validation_names_missing_fields_and_times(tmp_path):
    dropped = _era5_set(tmp_path / "missing-soil.grib", drop=(41,))
    report = fetch.validate_era5_files((dropped,))
    assert not report.ok
    assert any("swvl3" in failure for failure in report.failures)

    bare = _era5_set(tmp_path / "no-orog.grib", orography=False)
    report = fetch.validate_era5_files((bare,))
    assert any("source-orography" in failure for failure in report.failures)

    short = _era5_set(tmp_path / "short.grib", times=_TIMES[:1])
    report = fetch.validate_era5_files((short,), expected_times=_TIMES)
    assert any("missing valid times" in failure
               for failure in report.failures)


def test_era5_validation_fails_closed_on_transport_defects(tmp_path):
    wrong_edition = tmp_path / "era5.grib"
    wrong_edition.write_bytes(
        _grib1_message(129, 100, 500, _TIMES[0], edition=2))
    report = fetch.validate_era5_files((wrong_edition,))
    assert any("edition" in failure for failure in report.failures)
    report = fetch.validate_era5_files((tmp_path / "absent.grib",))
    assert any("missing input file" in failure
               for failure in report.failures)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_fetch_era5_template_flow(tmp_path, capsys):
    rc = cli.main([
        "fetch", "--source", "era5", "--cycle", "1974-04-03T00",
        "--hours", "12", "--area", "30,-100,40,-90",
        "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cdsapi" in out and "~/.cdsapirc" in out
    spec = json.loads((tmp_path / fetch.ERA5_REQUEST_NAME).read_text())
    assert spec["schema"] == "gpuwm-era5-cds-request-v1"
    assert len(spec["requests"]) == 2


def test_cli_fetch_era5_validate_flow(tmp_path, capsys):
    path = _era5_set(tmp_path / "era5.grib")
    rc = cli.main(["fetch", "--source", "era5", "--validate", str(path)])
    assert rc == 0
    assert "era5 validation: PASS" in capsys.readouterr().out
    bad = _era5_set(tmp_path / "bad.grib", orography=False)
    rc = cli.main(["fetch", "--source", "era5", "--validate", str(bad)])
    assert rc == 1


def test_cli_fetch_argument_contracts(tmp_path, capsys):
    with pytest.raises(SystemExit):  # --source is required with choices
        cli.main(["fetch", "--cycle", "latest"])

    def refused(needle: str, argv: list[str]) -> None:
        # Documented refusals surface as exit 2 + a stderr message
        # through the CLI dispatch boundary, never as a traceback.
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err

    refused("requires --area",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--out", str(tmp_path)])
    refused("mutually exclusive",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--out", str(tmp_path),
             "--area", "30,-100,40,-90", "--point", "35,-97"])
    refused("requires --radius-km",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path),
             "--point", "35,-97"])
    refused("era5 only",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--out", str(tmp_path),
             "--validate", "x.grib"])
    refused("hourly",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path), "--cadence", "3"])
    refused("requires --cycle",
            ["fetch", "--source", "hrrr", "--hours", "2",
             "--out", str(tmp_path)])
    refused("not meaningful for ERA5",
            ["fetch", "--source", "era5", "--cycle", "latest",
             "--hours", "6", "--area", "30,-100,40,-90",
             "--out", str(tmp_path)])


def test_cli_fetch_gfs_end_to_end_with_mocked_transport(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    rc = cli.main(["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
                   "--hours", "6", "--area", "30,-100,40,-90",
                   "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    assert printed.count("fetch gfs f0") == 3
    assert fetch.FETCH_MANIFEST_NAME in printed
    assert (out / "gfs-series.tsv").is_file()


# ---------------------------------------------------------------------------
# Area-blind resume trap: request-identity comparison + --force-refetch
# ---------------------------------------------------------------------------

def test_cli_refetch_with_different_area_refuses(tmp_path, monkeypatch,
                                                 capsys):
    """The acceptance repro: a larger --area into the same --out must
    refuse (per-file record counts are area-blind), naming the exact
    difference and the remedy."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--out", str(out)]
    assert cli.main(base + ["--area", "30,-100,40,-90"]) == 0
    capsys.readouterr()
    before = (out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2").read_bytes()

    rc = cli.main(base + ["--area", "20,-110,50,-80"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "different request" in err
    assert "area" in err and "--force-refetch" in err
    assert "Traceback" not in err
    # The refusal changed nothing on disk.
    assert (out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2"
            ).read_bytes() == before

    # --force-refetch moves the old files aside (never deletes) and
    # re-downloads; the manifest then records the new request.
    rc = cli.main(base + ["--area", "20,-110,50,-80", "--force-refetch"])
    assert rc == 0
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["area"]["lon_west"] == -110.0
    rejected = list(out.glob("*.rejected-*"))
    assert rejected, "old-area files must be quarantined, not deleted"


def test_cli_refetch_with_different_cycle_refuses(tmp_path, monkeypatch,
                                                  capsys):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    tail = ["--hours", "3", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(["fetch", "--source", "gfs",
                     "--cycle", "2026-07-28T06"] + tail) == 0
    capsys.readouterr()
    rc = cli.main(["fetch", "--source", "gfs",
                   "--cycle", "2026-07-28T12"] + tail)
    assert rc == 2
    err = capsys.readouterr().err
    assert "cycle" in err and "different request" in err


def test_refetch_hours_extension_still_resumes(tmp_path, monkeypatch,
                                               capsys):
    """Hours-only growth is byte-safe (same source/cycle/area produce
    identical per-hour files), so the window stays extendable."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base + ["--hours", "3"]) == 0
    capsys.readouterr()
    assert cli.main(base + ["--hours", "6"]) == 0
    printed = capsys.readouterr().out
    assert "verified -- skipped" in printed          # f000/f003 resumed
    assert "f006" in printed
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == [0, 3, 6]


def test_resume_refuses_a_pre_manifest_interrupt(tmp_path, monkeypatch,
                                                 capsys):
    """The manifest is written after the downloads, so a fetch killed
    mid-window leaves envelope-valid files with no recorded request.
    Those files are area-blind (any area passes the 124-record bar) and
    must NOT be resumed; the refusal names both remedies."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base) == 0
    capsys.readouterr()
    # Simulate the interrupt: files present, manifest never written.
    (out / fetch.FETCH_MANIFEST_NAME).unlink()

    rc = cli.main(base)
    assert rc == 2
    err = capsys.readouterr().err
    assert fetch.FETCH_MANIFEST_NAME in err
    assert "UNVERIFIED" in err
    assert "different --out" in err and "--force-refetch" in err
    assert "Traceback" not in err

    # --force-refetch quarantines the unverifiable files and re-fetches.
    assert cli.main(base + ["--force-refetch"]) == 0
    assert (out / fetch.FETCH_MANIFEST_NAME).is_file()
    assert list(out.glob("*.rejected-*")), \
        "unverifiable files must be quarantined, not deleted"


@pytest.mark.parametrize("damage", ["truncated-json", "foreign-schema"])
def test_resume_refuses_a_corrupt_or_foreign_manifest(tmp_path, monkeypatch,
                                                      capsys, damage):
    """A manifest that cannot be read as this schema proves nothing;
    the files beside it are unverified for the request and refuse."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base) == 0
    capsys.readouterr()
    manifest = out / fetch.FETCH_MANIFEST_NAME
    if damage == "truncated-json":
        manifest.write_bytes(manifest.read_bytes()[:40])
    else:
        manifest.write_text(json.dumps({"schema": "someone-elses-v9"}),
                            encoding="utf-8")

    rc = cli.main(base)
    assert rc == 2
    err = capsys.readouterr().err
    assert "UNVERIFIED" in err and "--force-refetch" in err


def test_fetch_into_missing_or_empty_out_still_proceeds(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    """The no-manifest refusal must not over-fire: an absent or empty
    --out has nothing to mis-bless and fetches normally."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    empty = tmp_path / "pre-made"
    empty.mkdir()
    for out in (tmp_path / "absent", empty):
        assert cli.main(
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "3", "--area", "30,-100,40,-90",
             "--out", str(out)]) == 0
    capsys.readouterr()


def test_gfs_resume_rejects_a_swapped_file_by_manifest_digest(
        tmp_path, monkeypatch, capsys):
    """A count-valid file with different bytes under a recorded name
    must not be re-blessed: the prior manifest's sha256 binds it."""
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    base = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
            "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out)]
    assert cli.main(base) == 0
    capsys.readouterr()
    target = out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2"
    swapped = bytearray(target.read_bytes())
    swapped[4] ^= 0x01  # reserved byte: still 124 valid envelopes
    target.write_bytes(bytes(swapped))
    assert fetch.count_grib2_messages(target) == 124

    rc = cli.main(base)
    assert rc == 2
    err = capsys.readouterr().err
    assert "sha256" in err and "--force-refetch" in err


# ---------------------------------------------------------------------------
# fetch -> front door manifest bridge (GFS) + HRRR handoff line
# ---------------------------------------------------------------------------

def _front_door_inputs(tmp_path):
    bridge = tmp_path / "gfs_grib2_bridge.exe"
    bridge.write_bytes(b"MZ fake bridge payload")
    wps = tmp_path / "myarea.namelist.wps"
    wps.write_text("&share\n max_dom = 1,\n/\n", encoding="utf-8")
    config = tmp_path / "myarea.toml"
    config.write_text("[experiment]\nname = 'myarea'\n", encoding="utf-8")
    return bridge, wps, config


def test_front_door_manifest_feeds_the_gfs_verifier_unedited(
        tmp_path, monkeypatch, capsys):
    """The authored manifest must satisfy the front door's own verifier
    (gpuwm.gfs_direct._verify_input_manifest) with zero hand edits --
    schema, source identity block, role inventory, and the bridge
    executable's own sha256."""
    from gpuwm import gfs_direct

    assert (fetch.GFS_FRONT_DOOR_MANIFEST_SCHEMA
            == gfs_direct.INPUT_MANIFEST_SCHEMA)

    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    bridge, wps, config = _front_door_inputs(tmp_path)
    rc = cli.main([
        "fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
        "--hours", "3", "--area", "30,-100,40,-90", "--out", str(out),
        "--author-front-door-manifest", "--bridge", str(bridge),
        "--wps-namelist", str(wps), "--experiment-config", str(config)])
    assert rc == 0
    manifest_path = out / fetch.GFS_INPUT_MANIFEST_NAME
    digest = fetch.sha256_file(manifest_path)
    printed = capsys.readouterr().out
    assert f"--source-manifest-sha256 {digest}" in printed
    assert "rw-wps --source gfs" in printed
    assert "--cycle 2026-07-28_06:00:00" in printed

    payload = json.loads(manifest_path.read_text())
    assert payload["source"] == {"model": "GFS", "product": "pgrb2.0p25",
                                 "cycle": "2026-07-28T06:00:00Z"}
    assert payload["files"]["bridge"]["sha256"] == fetch.sha256_file(bridge)

    roles = {
        "series": out / "gfs-series.tsv",
        "bridge": bridge,
        "wps_namelist": wps,
        "experiment_config": config,
        "grib-f000": out / "gfs.t06z.pgrb2.0p25.f000.subset.grib2",
        "grib-f003": out / "gfs.t06z.pgrb2.0p25.f003.subset.grib2",
    }
    verified = gfs_direct._verify_input_manifest(
        manifest_path, digest, roles)
    assert verified["schema"] == gfs_direct.INPUT_MANIFEST_SCHEMA


def test_front_door_author_only_mode_converts_an_existing_fetch(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gfs_transport, "_download", _fake_gfs_download)
    out = tmp_path / "gfs"
    assert cli.main(["fetch", "--source", "gfs", "--cycle",
                     "2026-07-28T06", "--hours", "3", "--area",
                     "30,-100,40,-90", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    # A plain fetch names the authoring step, so nobody has to read
    # source to learn the front door needs a manifest.
    assert "--author-front-door-manifest" in printed

    bridge, wps, config = _front_door_inputs(tmp_path)
    rc = cli.main([
        "fetch", "--source", "gfs", "--out", str(out),
        "--author-front-door-manifest", "--bridge", str(bridge),
        "--wps-namelist", str(wps), "--experiment-config", str(config)])
    assert rc == 0
    assert (out / fetch.GFS_INPUT_MANIFEST_NAME).is_file()


def test_front_door_author_flag_contracts(tmp_path, capsys):
    def refused(needle: str, argv: list[str]) -> None:
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err

    refused("--source gfs/gdas only",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path),
             "--author-front-door-manifest"])
    refused("requires --out plus --bridge",
            ["fetch", "--source", "gfs", "--out", str(tmp_path),
             "--author-front-door-manifest"])
    refused("belong to --author-front-door-manifest",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "3", "--area", "30,-100,40,-90",
             "--out", str(tmp_path), "--bridge", "x.exe"])
    refused("not a completed",
            ["fetch", "--source", "gfs", "--out", str(tmp_path / "empty"),
             "--author-front-door-manifest", "--bridge", "b.exe",
             "--wps-namelist", "w", "--experiment-config", "c"])


def test_cli_fetch_hrrr_prints_the_front_door_handoff(tmp_path,
                                                      monkeypatch,
                                                      capsys):
    """HRRR's front door consumes SHA256SUMS directly; the missing link
    was its own sha256, so fetch prints the complete pair."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python", "--cycle",
                   "2026-07-28T05", "--hours", "1", "--transport", "s3",
                   "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr().out
    digest = fetch.sha256_file(out / "SHA256SUMS")
    assert "rw-wps --source hrrr" in printed
    assert f"--source-manifest-sha256 {digest}" in printed


# ---------------------------------------------------------------------------
# HRRR NOMADS transport: URLs, auto resolution, manifest, resume, wait mode
# ---------------------------------------------------------------------------

def test_hrrr_object_url_speaks_both_transports():
    cycle = datetime(2026, 7, 28, 5)
    assert fetch.hrrr_object_url(cycle, 2, "wrfnat") == (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260728/conus/"
        "hrrr.t05z.wrfnatf02.grib2")
    assert fetch.hrrr_object_url(cycle, 2, "wrfprs",
                                 transport="nomads") == (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/"
        "hrrr.20260728/conus/hrrr.t05z.wrfprsf02.grib2")
    with pytest.raises(ValueError, match="transport"):
        fetch.hrrr_object_url(cycle, 2, "wrfnat", transport="ftp")


def test_resolve_hrrr_transport_auto_prefers_nomads_when_it_serves():
    cycle = datetime(2026, 7, 28, 5)
    now = datetime(2026, 7, 28, 8)
    probed = []

    def probe(url):
        probed.append(url)
        return True

    got = fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=6, now=now, probe=probe,
        progress=lambda line: None)
    assert got == "nomads"
    # Completeness mirrors resolve_latest_cycle: BOTH final-hour objects.
    assert probed == [
        fetch.hrrr_object_url(cycle, 6, "wrfnat", transport="nomads"),
        fetch.hrrr_object_url(cycle, 6, "wrfprs", transport="nomads")]


def test_resolve_hrrr_transport_auto_falls_back_to_s3():
    cycle = datetime(2026, 7, 28, 5)
    now = datetime(2026, 7, 28, 8)
    got = fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=6, now=now, probe=lambda url: False,
        progress=lambda line: None)
    assert got == "s3"


def test_resolve_hrrr_transport_skips_the_probe_for_archived_cycles():
    """Beyond NOMADS retention there is nothing to probe: auto goes
    straight to the S3 archive."""
    cycle = datetime(2026, 7, 20, 0)
    now = datetime(2026, 7, 28, 8)

    def explode(url):
        raise AssertionError("an archived cycle must not probe NOMADS")

    lines = []
    got = fetch.resolve_hrrr_transport(
        cycle, "auto", last_hour=6, now=now, probe=explode,
        progress=lines.append)
    assert got == "s3"
    assert any("retention" in line for line in lines)


def test_resolve_hrrr_transport_explicit_contracts():
    cycle = datetime(2026, 7, 20, 0)
    now = datetime(2026, 7, 28, 8)

    def explode(url):
        raise AssertionError("--transport s3 must not probe anything")

    assert fetch.resolve_hrrr_transport(
        cycle, "s3", last_hour=6, now=now, probe=explode) == "s3"
    with pytest.raises(ValueError, match="48 h"):
        fetch.resolve_hrrr_transport(
            cycle, "nomads", last_hour=6, now=now,
            probe=lambda url: False)
    with pytest.raises(ValueError, match="transport"):
        fetch.resolve_hrrr_transport(cycle, "ftp", last_hour=6, now=now)


def test_fetch_hrrr_nomads_transport_keeps_the_contracts(tmp_path,
                                                         monkeypatch):
    """Same 561/18 record bars, same manifest discipline -- only the
    host differs, and the manifest records it per file."""
    seen = []

    def product(request, *, workers, retries, expected_count=-1):
        seen.append(request)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        transport="nomads", progress=lambda line: None)

    prefix = fetch.HRRR_NOMADS_BASE + "/hrrr.20260728/conus/"
    assert [request.kind for request in seen] == [
        "atmosphere", "soil", "atmosphere", "soil"]
    assert all(request.url.startswith(prefix) for request in seen)
    assert all(request.index_url == request.url + ".idx"
               for request in seen)
    manifest = json.loads(manifest_path.read_text())
    assert [item["role"] for item in manifest["files"]] == [
        "atmosphere", "soil", "atmosphere", "soil", "checksums"]
    assert [item.get("transport") for item in manifest["files"]] == [
        "nomads"] * 4 + [None]
    assert "byte-identical" in manifest["notes"]
    for item in manifest["files"][:-1]:
        assert item["sha256"] == hashlib.sha256(
            (out / item["name"]).read_bytes()).hexdigest()


def test_fetch_hrrr_resume_carries_across_transports(tmp_path,
                                                     monkeypatch):
    """A directory fetched over S3 resumes over NOMADS: the digest and
    record-count bars are host-independent because the files are
    byte-identical on both hosts."""
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
                  out=out)
    fetch.fetch_hrrr(**kwargs, transport="s3", progress=lambda line: None)

    def refuse(request, *, workers, retries):
        raise AssertionError("re-downloaded a file the digests verify")

    monkeypatch.setattr(hrrr_transport, "_download_product", refuse)
    lines: list[str] = []
    fetch.fetch_hrrr(**kwargs, transport="nomads", progress=lines.append)
    assert len(lines) == 2 and all("skipped" in line for line in lines)


def test_fetch_hrrr_refuses_unresolved_auto_outside_wait_mode(tmp_path):
    with pytest.raises(ValueError, match="resolve_hrrr_transport"):
        fetch.fetch_hrrr(
            cycle=datetime(2026, 7, 28, 5), hours=(0,), area=None,
            out=tmp_path / "hrrr", transport="auto",
            progress=lambda line: None)


class _LivePublication:
    """Fake clock/sleep/probe harness: a URL answers once the fake time
    reaches its scheduled publication instant."""

    def __init__(self, schedule):
        #: {(product, hour, transport): publish_time}; absent = never.
        self.schedule = schedule
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        # The polling contract: never sleep past the 30 s cap.
        assert 0 < seconds <= fetch.HRRR_WAIT_POLL_SECONDS
        self.sleeps.append(seconds)
        self.now += seconds

    def probe(self, url: str) -> bool:
        import re

        name = url[:-4] if url.endswith(".idx") else url
        transport = ("nomads" if name.startswith(fetch.HRRR_NOMADS_BASE)
                     else "s3")
        product = "wrfnat" if "wrfnat" in name else "wrfprs"
        hour = int(re.search(r"f(\d\d)\.grib2$", name).group(1))
        when = self.schedule.get((product, hour, transport))
        return when is not None and self.now >= when


def test_fetch_hrrr_wait_downloads_hours_as_they_publish(tmp_path,
                                                         monkeypatch):
    """The live-cycle mode: f00 is up immediately, f01 publishes 45
    fake-seconds in; both arrive from NOMADS without a single real
    sleep, and the finished manifest is the normal complete one."""
    live = _LivePublication({
        ("wrfnat", 0, "nomads"): 0.0, ("wrfprs", 0, "nomads"): 0.0,
        ("wrfnat", 1, "nomads"): 45.0, ("wrfprs", 1, "nomads"): 45.0,
    })
    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.url)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        transport="auto", wait=True, wait_timeout_s=3600.0,
        probe=live.probe, sleeper=live.sleep, clock=live.clock,
        progress=lambda line: None)

    assert len(downloaded) == 4
    assert all(url.startswith(fetch.HRRR_NOMADS_BASE)
               for url in downloaded)
    assert live.sleeps, "f01 was not up at t=0, so the fetch had to poll"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["forecast_hours"] == [0, 1]
    assert [item.get("transport") for item in manifest["files"]] == [
        "nomads"] * 4 + [None]


def test_fetch_hrrr_wait_auto_falls_back_per_product(tmp_path,
                                                     monkeypatch):
    """Under auto, every poll round tries NOMADS first and S3 second,
    so a product NOMADS never serves still arrives -- from S3."""
    live = _LivePublication({
        ("wrfnat", 0, "nomads"): 0.0, ("wrfprs", 0, "nomads"): 0.0,
        ("wrfnat", 1, "nomads"): 30.0,
        ("wrfprs", 1, "s3"): 60.0,  # never on NOMADS
    })
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    manifest_path = fetch.fetch_hrrr(
        cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None, out=out,
        transport="auto", wait=True, wait_timeout_s=3600.0,
        probe=live.probe, sleeper=live.sleep, clock=live.clock,
        progress=lambda line: None)
    manifest = json.loads(manifest_path.read_text())
    by_name = {item["name"]: item.get("transport")
               for item in manifest["files"]}
    assert by_name["hrrr.t05z.wrfnatf01.grib2"] == "nomads"
    assert by_name["hrrr.t05z.soilf01.grib2"] == "s3"


def test_fetch_hrrr_wait_times_out_honestly_and_resumes(tmp_path,
                                                        monkeypatch):
    """The honest timeout: the complete f00 prefix is manifested (so a
    re-run RESUMES rather than refusing at the pre-manifest-interrupt
    gate), and the error says exactly what was and was not fetched."""
    schedule = {
        ("wrfnat", 0, "nomads"): 0.0, ("wrfprs", 0, "nomads"): 0.0,
        # f01 never publishes anywhere within the window.
    }
    live = _LivePublication(schedule)
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    kwargs = dict(cycle=datetime(2026, 7, 28, 5), hours=(0, 1), area=None,
                  out=out, transport="auto", wait=True,
                  probe=live.probe, sleeper=live.sleep, clock=live.clock)
    with pytest.raises(RuntimeError) as caught:
        fetch.fetch_hrrr(**kwargs, wait_timeout_s=120.0,
                         progress=lambda line: None)
    message = str(caught.value)
    assert "timed out" in message
    assert "f00..f00" in message
    assert "resumes" in message

    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert manifest["forecast_hours"] == [0]
    names = [item["name"] for item in manifest["files"]]
    assert names == ["hrrr.t05z.wrfnatf00.grib2",
                     "hrrr.t05z.soilf00.grib2", "SHA256SUMS"]
    assert len((out / "SHA256SUMS").read_text().splitlines()) == 2

    # The cycle finishes publishing; the same command completes,
    # resuming the verified f00 files instead of re-downloading them.
    schedule[("wrfnat", 1, "nomads")] = live.now
    schedule[("wrfprs", 1, "nomads")] = live.now
    downloaded = []

    def product(request, *, workers, retries, expected_count=-1):
        downloaded.append(request.destination.name)
        return _fake_hrrr_product(request, workers=workers,
                                  retries=retries,
                                  expected_count=expected_count)

    monkeypatch.setattr(hrrr_transport, "_download_product", product)
    lines: list[str] = []
    manifest_path = fetch.fetch_hrrr(**kwargs, wait_timeout_s=120.0,
                                     progress=lines.append)
    assert downloaded == ["hrrr.t05z.wrfnatf01.grib2",
                          "hrrr.t05z.soilf01.grib2"]
    assert sum("skipped" in line for line in lines) == 2
    manifest = json.loads(manifest_path.read_text())
    assert manifest["forecast_hours"] == [0, 1]


def test_cli_fetch_hrrr_transport_and_wait_contracts(tmp_path, capsys):
    def refused(needle: str, argv: list[str]) -> None:
        assert cli.main(argv) == 2
        err = capsys.readouterr().err
        assert needle in err and "Traceback" not in err

    refused("--source hrrr only",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--area", "30,-100,40,-90",
             "--out", str(tmp_path), "--transport", "s3"])
    refused("--source hrrr only",
            ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
             "--hours", "6", "--area", "30,-100,40,-90",
             "--out", str(tmp_path), "--wait-for"])
    refused("belongs to --wait-for",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path),
             "--wait-timeout-minutes", "10"])
    refused("must be positive",
            ["fetch", "--source", "hrrr", "--cycle", "2026-07-28T05",
             "--hours", "2", "--out", str(tmp_path), "--wait-for",
             "--wait-timeout-minutes", "0"])


def test_cli_fetch_hrrr_auto_resolves_the_transport_once(tmp_path,
                                                         monkeypatch,
                                                         capsys):
    """The CLI's plain (non-wait) path resolves 'auto' to one concrete
    transport up front and hands fetch_hrrr the result."""
    calls = {}

    def resolve(cycle, requested, *, last_hour, **kwargs):
        calls["args"] = (cycle, requested, last_hour)
        return "s3"

    monkeypatch.setattr(fetch, "resolve_hrrr_transport", resolve)
    monkeypatch.setattr(
        hrrr_transport, "_download_product", _fake_hrrr_product)
    out = tmp_path / "hrrr"
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python", "--cycle",
                   "2026-07-28T05", "--hours", "1", "--out", str(out)])
    assert rc == 0
    assert calls["args"] == (datetime(2026, 7, 28, 5), "auto", 1)
    manifest = json.loads((out / fetch.FETCH_MANIFEST_NAME).read_text())
    assert {item.get("transport")
            for item in manifest["files"]} == {"s3", None}


def test_cli_fetch_hrrr_wait_timeout_is_an_orderly_exit(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    """A --wait-for window that never fills is an operational outcome
    with a resume story -- exit 2 and the message, never a traceback."""
    monkeypatch.setattr(fetch, "_head_ok", lambda url: False)
    out = tmp_path / "hrrr"
    rc = cli.main(["fetch", "--source", "hrrr", "--engine", "python", "--cycle",
                   "2026-07-28T05", "--hours", "1", "--out", str(out),
                   "--wait-for", "--wait-timeout-minutes", "0.005"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "timed out" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------------------
# Optional live smoke: one small .idx object (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("GPUWM_NETWORK_TESTS"),
                    reason="live network smoke; set GPUWM_NETWORK_TESTS=1")
def test_live_latest_hrrr_index_is_a_canonical_noaa_index():
    from urllib.request import Request, urlopen

    cycle = fetch.resolve_latest_cycle("hrrr", 1)
    url = fetch.hrrr_object_url(cycle, 0, "wrfnat") + ".idx"
    request = Request(url, headers={"User-Agent": "gpuwm-fetch-test/1"})
    with urlopen(request, timeout=120) as response:
        first = response.read(4096).decode("ascii").splitlines()[0]
    sequence, offset, _rest = first.split(":", 2)
    assert (sequence, offset) == ("1", "0")


def test_a_named_incomplete_cycle_refuses_with_the_complete_one(monkeypatch):
    """Node-3 #7: 20 lines of raw urllib.HTTPError for an early cycle.

    `--cycle 2026-07-30T00` an hour before that cycle exists dumped a
    traceback from inside the downloader, in a product that names the
    remedy for almost everything else.  The completeness probe already
    existed for `--cycle latest`; it just never ran for a named one.
    """
    from datetime import datetime
    import gpuwm.fetch as fetch_module

    now = datetime(2026, 7, 30, 1, 30)
    published = datetime(2026, 7, 29, 18)
    complete = {
        url for url in fetch_module.cycle_probe_urls("gfs", published, 6)}

    def probe(url):
        # Only cycles at or before 18Z on the 29th exist.
        return any(url.startswith(u[:u.rindex("/")]) and u == url
                   for u in complete) or _older(url)

    def _older(url):
        for back in range(1, 9):
            older = published - timedelta(hours=6 * back)
            if url in fetch_module.cycle_probe_urls("gfs", older, 6):
                return True
        return False

    from datetime import timedelta
    # A published cycle passes silently.
    fetch_module.require_published_cycle(
        "gfs", published, 6, now=now, probe=probe)

    with pytest.raises(RuntimeError) as excinfo:
        fetch_module.require_published_cycle(
            "gfs", datetime(2026, 7, 30, 0), 6, now=now, probe=probe)
    message = str(excinfo.value)
    assert "2026-07-30T00Z is not published through f006 yet" in message
    assert "newest complete GFS cycle" in message
    assert "2026-07-29T18Z" in message
    assert "--cycle latest" in message
