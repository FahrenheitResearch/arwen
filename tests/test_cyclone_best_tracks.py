"""Exercise the shipped Rust parser and automatic setup binding, offline."""
from datetime import datetime
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess

import pytest
from gpuwm import cyclone_setup, cyclone_tracks, tc_tracks


@pytest.fixture
def native(tmp_path):
    binary = os.environ.get("GPUWM_RW_FETCH")
    if not binary:
        pytest.skip("built rw_fetch required")
    def parse(data, fmt, storm="auto"):
        raw = tmp_path/("source.gz" if data.startswith(b"\x1f\x8b") else "source.txt")
        raw.write_bytes(data)
        out = tmp_path/"out.json"
        request = tmp_path/"request.json"
        request.write_text(json.dumps(dict(input=str(raw), format=fmt,
            start="2020-01-01T00:00:00Z", storm=storm,
            cache=str(tmp_path/"cache"), output=str(out))))
        result = subprocess.run([binary, "cyclone-tracks", "--request", str(request)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise ValueError(result.stderr)
        return json.loads(out.read_text())
    return parse


def test_gzipped_atcf_identity_units_dedup(native):
    line = b"SH, 01, 2020010100, 30, BEST, 0, 123S, 1799E, 50, 980, TS\n"
    doc = native(gzip.compress(line+line), "atcf", "SH012020")
    t = doc["tracks"][0]
    assert t["track_id"] == "SH012020"
    assert len(t["fixes"]) == 1
    f = t["fixes"][0]
    assert f["valid"] == "2020-01-01T00:30:00Z"
    assert f["lat"] == pytest.approx(-12.3)
    assert f["lon"] == pytest.approx(179.9)
    assert f["mslp_pa"] == 98000
    assert f["max_wind10_ms"] == pytest.approx(50*1852/3600)
    assert doc["tracker"]["acquisition"][0]["sha256"]


def test_hurdat_single_identity_and_count(native):
    data = b"AL012020, TEST, 1,\n20200101, 0000, L, HU, 12.3N, 179.9W, -999, -999,\n"
    doc = native(data, "hurdat2", "test")
    assert doc["tracks"][0]["fixes"][0]["flags"] == ["landfall"]
    assert doc["tracks"][0]["fixes"][0]["max_wind10_ms"] is None
    with pytest.raises(ValueError, match="truncated"):
        native(data.replace(b"TEST, 1", b"TEST, 2"), "hurdat2")


def test_ibtracs_jtwc_columns_aliases_and_longitude(native):
    data = (b"SID,NAME,ISO_TIME,LAT,LON,USA_ATCF_ID,USA_WIND,USA_PRES,USA_AGENCY,USA_LAT,USA_LON\n"
            b"2020001S,\"TEST, STORM\",2020-01-01 00:00:00,-12,181,SH012020,50,980,jtwc_sh,-12,181\n"
            b"2020002S,OTHER,2020-01-01 00:00:00,-14,170,SH022020, , ,bom, , \n")
    doc = native(data, "jtwc")
    assert len(doc["tracks"]) == 1
    assert doc["tracks"][0]["name"] == "TEST, STORM"
    assert "2020001S" in doc["tracks"][0]["aliases"]
    assert doc["tracks"][0]["fixes"][0]["lon"] == -179
    doc = native(data, "ibtracs", "SH022020")
    assert doc["tracks"][0]["fixes"][0]["max_wind10_ms"] is None


def test_conflicting_radius_duplicates_refuse(native):
    line = b"AL, 01, 2020010100, , BEST, 0, 123N, 200W, 50, 990\n"
    with pytest.raises(ValueError, match="conflicting"):
        native(line+line.replace(b"123N", b"124N"), "atcf")


def test_best_track_refuses_multiple_events(tmp_path):
    path = tmp_path/"b.dat"
    path.write_text("AL, 01, 2020010100, , BEST, 0, 123N, 200W, 50, 990\n"
                    "EP, 02, 2020010100, , BEST, 0, 150N, 1200W, 40, 995\n")
    with pytest.raises(ValueError, match="select one event"):
        tc_tracks.best_track(path)
    assert tc_tracks.best_track(path, "EP022020")["storm_id"] == "EP022020"


def test_name_only_setup_acquires_before_selection(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    cyclone_setup.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--storm", "TEST",
        "--cycle", "2020-01-01T00", "--hours", "24", "--card", "32gb",
        "--out", str(tmp_path/"setup.toml")])
    calls = []
    def acquired(**kw):
        calls.append(kw)
        raise ValueError("acquisition reached")
    monkeypatch.setattr(cyclone_tracks, "acquire_tracks", acquired)
    assert cyclone_setup.main(args) == 1
    assert calls[0]["storm"] == "TEST"
    assert calls[0]["start"] == datetime(2020, 1, 1)


def test_area_only_setup_defaults_to_auto(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    cyclone_setup.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--search-area=-30,170,-10,-170",
        "--cycle", "2020-01-01T00", "--hours", "24", "--card", "32gb",
        "--out", str(tmp_path/"setup.toml")])
    calls = []
    def acquired(**kw):
        calls.append(kw)
        raise ValueError("acquisition reached")
    monkeypatch.setattr(cyclone_tracks, "acquire_tracks", acquired)
    assert cyclone_setup.main(args) == 1
    assert calls[0]["storm"] == "auto"


def test_ibtracs_sid_does_not_split_when_atcf_id_appears(native):
    header = b"SID,NAME,ISO_TIME,LAT,LON,USA_ATCF_ID,USA_WIND,USA_PRES,USA_AGENCY\n"
    doc = native(header +
        b"2020001S,TEST,2020-01-01 00:00:00,-12,179,,30,995,jtwc_sh\n" +
        b"2020001S,TEST,2020-01-01 06:00:00,-13,-179,SH012020,50,980,jtwc_sh\n",
        "ibtracs", "SH012020")
    assert len(doc["tracks"]) == 1
    assert doc["tracks"][0]["track_id"] == "SH012020"
    assert len(doc["tracks"][0]["fixes"]) == 2


def test_scores_follow_one_trajectory_and_count_missing_model_frames():
    times = ["2020-01-01T00:00:00Z", "2020-01-01T03:00:00Z",
             "2020-01-01T06:00:00Z", "2020-01-01T09:00:00Z"]
    def fix(i, lon):
        return dict(valid=times[i], lat=0, lon=lon, mslp_pa=99000, max_wind10_ms=30)
    m = dict(member_id="m", init=times[0], weight=1,
             frames=[dict(valid=t) for t in times])
    tracks = [dict(member_id="m", track_id="original", fixes=[fix(0, 179), fix(1, 175)]),
              dict(member_id="m", track_id="other", fixes=[fix(0, 178), fix(1, -180), fix(2, -179)])]
    result = tc_tracks.scores([m], tracks, dict(fixes=[fix(0, 179), fix(2, -179)]))["m"]
    assert result["selected_track_id"] == "original"
    assert result["detections"] == 2
    assert result["misses"] == 1
    assert result["unobserved_valid_times"] == 1
    rows = result["common_valid_times"]
    assert rows[1]["centre_distance_km"] > 500
    assert abs(abs(rows[1]["best_fix"]["lon"]) - 180) < 1e-6
    assert rows[2]["track_id"] is None


def test_scores_all_missing_and_long_observation_gaps():
    times = ["2020-01-01T00:00:00Z", "2020-01-01T06:00:00Z", "2020-01-01T12:00:00Z"]
    m = dict(member_id="m", init=times[0], weight=1, frames=[dict(valid=t) for t in times])
    result = tc_tracks.scores([m], [], dict(fixes=[
        dict(valid=times[0], lat=0, lon=0), dict(valid=times[2], lat=0, lon=2)]))["m"]
    assert result["misses"] == 2
    assert result["unobserved_valid_times"] == 1
    assert result["mean_track_error_km"] is None


def test_split_stream_hex_bridge_preserves_native_winds_and_lineage(tmp_path):
    import struct
    fixture = os.environ.get("GPUWM_TC_HEX_FIXTURE")
    mesh_binary = os.environ.get("GPUWM_RW_MPAS_MESH")
    if not fixture or not mesh_binary:
        pytest.skip("built Rust fixture and mesh generator required")
    mesh = tmp_path/"mesh.nc"
    subprocess.run([mesh_binary, "--cells", "64", "--background-km", "3000",
                    "--fit-spacing", "yes", "--out", str(mesh),
                    "--receipt", str(tmp_path/"mesh.json")], check=True,
                   capture_output=True, timeout=60)
    cells = json.loads((tmp_path/"mesh.json").read_text())["emitted"]["n_cells"]
    subprocess.run([fixture, str(tmp_path), str(cells)], check=True, timeout=30)
    work = tmp_path/"work"
    work.mkdir()
    manifest = tc_tracks.hex_input(tmp_path, mesh, work, initial=tmp_path/"init.nc",
                                  window="latlon:0,2,0,2,0.25", workers=8)
    member = manifest["members"][0]
    assert member["family"] == "hex"
    lineage = member["ancestry"]["native_hex"]
    assert lineage["mesh"]["sha256"] == tc_tracks.sha(mesh)
    assert lineage["conversion"]["frames"][0]["split_streams"]["cell_centres_from"] == str(mesh)
    frame = member["frames"][0]
    table = json.loads(tc_tracks.TABLE.read_text())
    request = dict(path=frame["path"], select=frame["select"], out=str(tmp_path/"fields"),
                   vocabulary=table["vocabularies"]["era5-zarr"],
                   fields=["u850", "v850", "msl"])
    req = tmp_path/"fields.json"
    req.write_text(json.dumps(request))
    subprocess.run([tc_tracks.binary("rw_tc_fields"), "extract", str(req)], check=True, timeout=30)
    for key, expected in [("u850", 5), ("v850", 2), ("msl", None)]:
        data = (tmp_path/"fields"/(key+".f32")).read_bytes()
        values = struct.unpack("<"+"f"*(len(data)//4), data)
        if expected is None:
            # Native sea-level reduction differs from the fixture surface pressure.
            assert all(95000 < v < 110000 for v in values)
            assert max(values)-min(values) < .1
        else:
            assert values == pytest.approx([expected]*len(values), abs=.1)


def test_jtwc_uses_agency_positions(native):
    doc = native(b"SID,NAME,ISO_TIME,LAT,LON,USA_ATCF_ID,USA_WIND,USA_PRES,USA_AGENCY,USA_LAT,USA_LON\n"
                 b"2020001S,TEST,2020-01-01 00:00:00,-11,178,SH012020,50,980,jtwc_sh,-12,179\n",
                 "jtwc", "SH012020")
    fix = doc["tracks"][0]["fixes"][0]
    assert (fix["lat"], fix["lon"]) == (-12, 179)


def test_stale_fetch_names_missing_mode(monkeypatch, tmp_path):
    stale = tmp_path/"rw_fetch"
    stale.write_bytes(b"rw_fetch fetch --model MODEL")
    monkeypatch.setattr(tc_tracks, "binary", lambda name: str(stale))
    with pytest.raises(ValueError, match="predates cyclone-tracks"):
        cyclone_tracks.native_track_binary("cyclone-tracks")


def test_observed_track_does_not_replace_forcing_member(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    cyclone_setup.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--source", "gefs", "--member", "p03",
        "--storm", "system-a", "--cycle", "2026090900", "--hours", "24",
        "--card", "32gb", "--out", str(tmp_path/"setup.toml")])
    fixture = Path(__file__).parent/"fixtures/cyclone-tracks.json"
    monkeypatch.setattr(cyclone_tracks, "acquire_tracks", lambda **kw: fixture)
    seen = []
    def plan(**kw):
        seen.append(kw["member"])
        raise ValueError("member captured")
    monkeypatch.setattr(cyclone_setup, "plan_cyclone", plan)
    assert cyclone_setup.main(args) == 1
    assert seen == ["p03"]


def test_atcf_retains_name_that_appears_after_genesis(native):
    def row(hour, name):
        columns = ["SH", "01", "20200101"+hour, "", "BEST", "0", "123S", "1799E", "50", "980"]
        columns += [""]*(27-len(columns))+[name]
        return (",".join(columns)+"\n").encode()
    doc = native(row("00", "")+row("06", "TEST"), "atcf", "TEST")
    assert doc["tracks"][0]["name"] == "TEST"
    assert len(doc["tracks"][0]["fixes"]) == 2
