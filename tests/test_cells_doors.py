"""``gpuwm cells``: the titan seam, held against a synthetic wrfout series.

The fixture is a small wrfout series written by netCDF4 (the writer this
tree already uses for its own history) with one Gaussian storm that
moves north-east and grows, so every frame has a cell titan can find
and track.  The reader under test is the Rust ``rw_netcdf`` bridge, so
the whole module is skipped, by name, on an install without it; the
doors that need the ``titan`` binary skip likewise.

What is pinned:

* the volume codec against the engine's own layout (a decode/encode
  round trip of a stream this module writes, and the frame count the
  engine reports on it);
* the export's byte stability: the same frames on the same ladder give
  the same stream digest;
* the catalog's peak updraft against a direct numpy maximum over the
  same footprint columns -- the join, not the number, is what could go
  wrong;
* the analyze door's refusal when no titan resolves.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.cells import catalog as cells_catalog
from gpuwm.cells import columns as cells_columns
from gpuwm.cells import export as cells_export
from gpuwm.cells import titan as cells_titan
from gpuwm.cells import titan_volume as tv

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

NX, NY, NZ = 40, 36, 24
DX = 3000.0
FRAMES = 4
INTERVAL_MIN = 10

#: The exported stream's digest for the fixture below on the default
#: ladder.  Re-record only when the exporter's rule changes on purpose.
FIXTURE_STREAM_SHA256 = "9bfe11fda29cc98a0d0209b97f63b496a97d1812a741ed63c6aaa19e729a4666"


def _bridge_or_skip():
    from gpuwm import netcdf_bridge
    try:
        found = netcdf_bridge.find_netcdf_bin()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    if found is None:
        pytest.skip("rw_netcdf is not built; gpuwm cells reads wrfout through it")
    return found


def _titan_or_skip() -> Path:
    try:
        found = cells_titan.find_titan()
    except FileNotFoundError as error:
        pytest.skip(str(error))
    if found is None:
        pytest.skip("no titan binary resolves; set GPUWM_TITAN to run the engine legs")
    return found


def write_fixture_series(root: Path) -> list[Path]:
    """A wrfout series with one moving, growing Gaussian storm."""

    netCDF4 = pytest.importorskip("netCDF4")
    root.mkdir(parents=True, exist_ok=True)
    start = dt.datetime(2026, 5, 15, 21, 0, 0)
    ys, xs = np.mgrid[0:NY, 0:NX]
    lat = 38.0 + (ys - NY / 2) * DX / 111000.0
    lon = -98.0 + (xs - NX / 2) * DX / (111000.0 * np.cos(np.radians(38.0)))
    terrain = np.full((NY, NX), 400.0, dtype=np.float32)
    # w-level heights: 100 m near the ground stretching to ~600 m aloft.
    dz = 100.0 + 500.0 * (np.arange(NZ) / (NZ - 1)) ** 0.8
    z_w = np.concatenate([[0.0], np.cumsum(dz)])
    paths: list[Path] = []
    for index in range(FRAMES):
        valid = start + dt.timedelta(minutes=INTERVAL_MIN * index)
        stamp = valid.strftime("%Y-%m-%d_%H_%M_%S")
        path = root / f"wrfout_d02_{stamp}"
        cx, cy = 12.0 + 3.0 * index, 14.0 + 2.0 * index
        radius = 3.0 + 0.5 * index
        r2 = ((xs - cx) ** 2 + (ys - cy) ** 2) / radius ** 2
        column = np.exp(-r2)
        zc = 0.5 * (z_w[:-1] + z_w[1:])
        vertical = np.exp(-((zc - 5000.0) / 3500.0) ** 2)
        refl = (-30.0 + 90.0 * column[None] * vertical[:, None, None]).astype(np.float32)
        w_profile = np.sin(np.pi * np.clip(z_w / 12000.0, 0, 1))
        w = (25.0 * column[None] * w_profile[:, None, None]).astype(np.float32)
        theta_profile = 300.0 + 0.004 * zc
        pressure = (100000.0 * np.exp(-zc / 8400.0))
        temperature = theta_profile * (pressure / 100000.0) ** (287.0 / 1004.5)
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("Time", None)
            ds.createDimension("bottom_top", NZ)
            ds.createDimension("bottom_top_stag", NZ + 1)
            ds.createDimension("south_north", NY)
            ds.createDimension("west_east", NX)
            ds.createDimension("DateStrLen", 19)
            ds.DX = DX
            ds.DY = DX
            ds.GRID_ID = 2
            ds.MAP_PROJ = 1
            ds.TRUELAT1 = 38.0
            ds.TRUELAT2 = 38.0
            ds.STAND_LON = -98.0
            ds.CEN_LAT = 38.0
            ds.CEN_LON = -98.0
            ds.SIMULATION_START_DATE = start.strftime("%Y-%m-%d_%H:%M:%S")

            def var(name, dims, values, **attrs):
                v = ds.createVariable(name, "f4", dims)
                for key, value in attrs.items():
                    setattr(v, key, value)
                v[0] = values
                return v

            xtime = ds.createVariable("XTIME", "f4", ("Time",))
            xtime.units = f"minutes since {start.strftime('%Y-%m-%d %H:%M:%S')}"
            xtime[0] = INTERVAL_MIN * index
            times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
            times[0] = np.frombuffer(
                valid.strftime("%Y-%m-%d_%H:%M:%S").encode("ascii"), dtype="S1")
            m3 = ("Time", "bottom_top", "south_north", "west_east")
            w3 = ("Time", "bottom_top_stag", "south_north", "west_east")
            m2 = ("Time", "south_north", "west_east")
            var("XLAT", m2, lat)
            var("XLONG", m2, lon)
            var("HGT", m2, terrain)
            ph = np.broadcast_to((z_w * 9.81)[:, None, None], (NZ + 1, NY, NX))
            var("PH", w3, np.zeros_like(ph, dtype=np.float32))
            var("PHB", w3, (ph + terrain[None] * 9.81).astype(np.float32))
            var("T", m3, np.broadcast_to((theta_profile - 300.0)[:, None, None],
                                        (NZ, NY, NX)).astype(np.float32))
            var("P", m3, np.zeros((NZ, NY, NX), np.float32))
            var("PB", m3, np.broadcast_to(pressure[:, None, None],
                                         (NZ, NY, NX)).astype(np.float32))
            var("W", w3, w)
            if index > 0:
                # The initial frame of a real run carries no REFL_10CM:
                # the microphysics has not produced one yet.  Both doors
                # must skip it by name rather than refuse the series.
                var("REFL_10CM", m3, refl)
            var("QVAPOR", m3, np.full((NZ, NY, NX), 0.005, np.float32))
            cloud = (0.002 * column[None] * vertical[:, None, None]).astype(np.float32)
            var("QCLOUD", m3, np.where(temperature[:, None, None] > 250.0, cloud, 0.0).astype(np.float32))
            var("QICE", m3, np.where(temperature[:, None, None] <= 250.0, cloud, 0.0).astype(np.float32))
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def series(tmp_path_factory) -> list[Path]:
    _bridge_or_skip()
    return write_fixture_series(tmp_path_factory.mktemp("cells-series"))


def test_codec_round_trips_a_stream_it_wrote(tmp_path):
    rng = np.random.default_rng(7)
    ladder = np.array([500.0, 1500.0, 3000.0])
    volumes = []
    for index in range(2):
        refl = rng.uniform(-30, 60, size=(3, 5, 4)).astype(np.float32)
        refl[0, 0, 0] = np.nan
        volumes.append(tv.TitanVolume(
            timestamp_ms=1_700_000_000_000 + 300_000 * index, nx=4, ny=5,
            z_levels_m=ladder, origin_x_m=-6000.0, origin_y_m=-7500.0,
            dx_m=3000.0, dy_m=3000.0, projection="TEST", source=f"t{index}",
            reflectivity=refl,
            optional={"temperature": rng.uniform(-60, 30, size=(3, 5, 4)).astype(np.float32)}))
    path = tmp_path / "s.tfs"
    tv.write_stream(path, volumes)
    back = tv.read_stream(path)
    assert len(back) == 2
    for a, b in zip(volumes, back):
        assert a.timestamp_ms == b.timestamp_ms
        assert a.projection == b.projection and a.source == b.source
        assert np.array_equal(a.z_levels_m, b.z_levels_m)
        assert np.array_equal(a.reflectivity, b.reflectivity, equal_nan=True)
        assert np.array_equal(a.optional["temperature"], b.optional["temperature"])
    assert tv.encode_stream([tv.encode_frame(v) for v in volumes]) == path.read_bytes()
    x, y, z = tv.cell_xyz(4, 5, tv.cell_index(4, 5, 3, 4, 2))
    assert (int(x), int(y), int(z)) == (3, 4, 2)


def test_ladder_interpolation_states_its_rule():
    nz, ny, nx = 4, 1, 1
    z = np.array([200.0, 1200.0, 2200.0, 3200.0], np.float32).reshape(nz, ny, nx)
    v = np.array([0.0, 10.0, 20.0, 30.0], np.float32).reshape(nz, ny, nx)
    terrain = np.array([[150.0]], np.float32)
    out = cells_export.interpolate_columns(
        v, z, terrain, np.array([100.0, 175.0, 700.0, 3200.0, 4000.0]))
    assert np.isnan(out[0, 0, 0])            # below the terrain
    assert out[1, 0, 0] == 0.0               # above terrain, below level 0
    assert out[2, 0, 0] == pytest.approx(5.0)
    assert out[3, 0, 0] == pytest.approx(30.0)
    assert np.isnan(out[4, 0, 0])            # above the top mass level
    assert len(cells_export.parse_ladder("250:18000:250")) == 72
    with pytest.raises(cells_export.ExportError):
        cells_export.parse_ladder("1000:500:100")


def test_export_is_byte_stable_and_describes_itself(series, tmp_path):
    first = cells_export.export_series(series, tmp_path / "a")
    second = cells_export.export_series(series, tmp_path / "b")
    assert first["stream_sha256"] == second["stream_sha256"]
    assert first["frame_count"] == FRAMES - 1
    assert [Path(s["path"]).name for s in first["skipped"]] == [series[0].name]
    assert first["ladder"]["count"] == 72
    assert first["ladder"]["levels_m_msl"][0] == 250.0
    assert first["grid"]["domain"] == "d02"
    assert first["grid"]["dx_m"] == DX
    assert "linear" in first["interpolation"]
    assert first["reader"].startswith("rw_netcdf")
    volumes = tv.read_stream(tmp_path / "a" / cells_export.STREAM_NAME)
    assert [v.timestamp_ms for v in volumes] == sorted(v.timestamp_ms for v in volumes)
    assert volumes[0].shape == (72, NY, NX)
    assert "temperature" in volumes[0].optional
    assert volumes[0].projection.startswith("ARWEN_GRID MAP_PROJ=1")
    assert np.nanmax(volumes[-1].reflectivity) > 50.0
    digest = hashlib.sha256(
        (tmp_path / "a" / cells_export.STREAM_NAME).read_bytes()).hexdigest()
    assert digest == first["stream_sha256"]
    if os.environ.get("GPUWM_CELLS_RECORD_PIN"):
        print("FIXTURE_STREAM_SHA256 =", digest)
    assert digest == FIXTURE_STREAM_SHA256


def test_cadence_overrides_size_the_trend_window_to_the_series():
    """Hourly history under a radar profile has no trend at all otherwise."""

    profile = {"forecast_history_s": "1800", "max_gap_seconds": "900"}
    hourly = [1_700_000_000_000 + 3_600_000 * k for k in range(13)]
    overrides, interval = cells_titan.cadence_overrides(hourly, profile)
    assert interval == 3600.0
    assert overrides == {"forecast_history_s": "10800", "max_gap_seconds": "7200"}
    five_min = [1_700_000_000_000 + 300_000 * k for k in range(10)]
    assert cells_titan.cadence_overrides(five_min, profile) == ({}, 300.0)
    # A user's own value is never lowered and never overridden.
    wide = {"forecast_history_s": "86400", "max_gap_seconds": "86400"}
    assert cells_titan.cadence_overrides(hourly, wide) == ({}, 3600.0)
    assert cells_titan.cadence_overrides([1], profile) == ({}, None)


def test_analyze_refuses_by_name_without_titan(series, tmp_path, monkeypatch, capsys):
    from gpuwm.cli import main
    monkeypatch.setenv(cells_titan.TITAN_ENV, "")
    monkeypatch.setattr(cells_titan, "titan_candidates", lambda: ())
    code = main(["cells", "analyze", *map(str, series), "--out", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == 2
    assert "titan storm-cell engine" in err and cells_titan.TITAN_ENV in err
    assert "second segmentation" in err


def test_analyze_door_tracks_the_storm_and_catalogs_it(series, tmp_path, capsys):
    titan = _titan_or_skip()
    from gpuwm.cli import main
    out = tmp_path / "case"
    code = main(["cells", "analyze", *map(str, series), "--out", str(out),
                 "--titan", str(titan), "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    receipt = json.loads(captured.out[captured.out.index("{"):])
    root = Path(receipt["root"])
    assert root == out / "d02-3km" / "cells" / "2026-05-15"
    for name in cells_titan.BUNDLE_FILES:
        assert (root / "titan" / name).is_file()
    assert receipt["titan"]["profile"] == "severe"
    # 10 min frames under severe's 900 s gap: the gap is raised, the
    # 1,800 s trend window already spans three intervals and is kept.
    assert receipt["titan"]["cadence"] == {
        "interval_s": 600.0, "overrides": {"max_gap_seconds": "1200"}}
    assert (root / "titan.cfg").read_text("utf-8") == "max_gap_seconds=1200\n"
    assert receipt["stages_seconds"]["titan_analyze_s"] >= 0.0
    catalog = json.loads((root / cells_catalog.JSON_NAME).read_text("utf-8"))
    rows = catalog["rows"]
    assert rows, "titan found no cell in a 60 dBZ Gaussian storm"
    assert len({row["timestamp_ms"] for row in rows}) == FRAMES - 1
    assert len(receipt["catalog"]["skipped"]) == 1
    tracks = {row["track_id"] for row in rows}
    assert len(tracks) == 1, f"one storm should be one track, got {tracks}"
    last = max(rows, key=lambda r: r["timestamp_ms"])
    # The track starts at the first frame that carries reflectivity.
    assert last["lifetime_so_far_s"] == pytest.approx(60.0 * INTERVAL_MIN * (FRAMES - 2))
    assert last["peak_w_ft_min"] == pytest.approx(last["peak_w_mps"] * 196.85)
    assert last["cloud_top_m_msl"] > last["cloud_base_m_msl"]
    assert last["freezing_level_m_msl"] > 0
    assert last["level_minus20c_m_msl"] > last["freezing_level_m_msl"]
    assert last["slwp_max_kg_m2"] >= last["slwp_mean_kg_m2"] > 0
    assert last["footprint_source"] == "voxels"
    # The fixture storm walks 3 cells east and 2 north per 10 min frame:
    # 10.8 km per 600 s, 18.0 m/s toward 056 deg.  titan's fitted trend
    # is that motion; the tracker's Kalman state at a scan is passed
    # through beside it as motion_*, unaltered.
    assert last["trend_speed_mps"] == pytest.approx(18.03, abs=0.5)
    assert last["trend_direction_to_deg"] == pytest.approx(56.3, abs=2.0)
    assert last["motion_speed_mps"] is not None
    assert 37.0 < last["centroid_lat"] < 39.0 and -99.5 < last["centroid_lon"] < -96.5
    for name, meta in catalog["columns"].items():
        assert meta["unit"] and meta["provenance"], name
    header = (root / cells_catalog.CSV_NAME).read_text("utf-8").splitlines()[0]
    assert header.split(",") == list(cells_catalog.COLUMNS)
    geojson = json.loads((root / cells_catalog.GEOJSON_NAME).read_text("utf-8"))
    assert len(geojson["features"]) == len(rows)
    overlays = sorted((root / cells_catalog.OVERLAY_DIR).glob("cells_*.json"))
    assert len(overlays) == FRAMES - 1
    doc = json.loads(overlays[-1].read_text("utf-8"))
    assert doc["lines"] and doc["lines"][0]["closed"] is True
    assert all(len(p) == 2 for p in doc["lines"][0]["points"])


def test_catalog_peak_w_equals_a_direct_numpy_maximum(series, tmp_path):
    """The join: the catalog's peak W is the max over the SAME columns."""

    titan = _titan_or_skip()
    export = cells_export.export_series(series, tmp_path / "x")
    bundle_dir = tmp_path / "bundle"
    cells_titan.analyze(titan, tmp_path / "x" / cells_export.STREAM_NAME, bundle_dir)
    bundle = cells_titan.Bundle.load(bundle_dir)
    by_time = {int(f["timestamp_ms"]): f for f in bundle.frames}
    checked = 0
    for path in series[1:]:
        for frame in cells_columns.open_frames(path):
            frame_json = by_time[frame.timestamp_ms]
            grid = export["grid"]
            projector = cells_catalog.Projector(
                frame, grid["origin_x_m"], grid["origin_y_m"], frame.dx_m, frame.dy_m)
            for obj in frame_json["objects"]:
                cols, source = cells_catalog.footprint_columns(
                    obj, frame.nx, frame.ny, projector)
                assert source == "voxels"
                row, _geometry = cells_catalog.cell_row(
                    bundle, frame_json, obj, frame, 0, projector)
                direct = cells_catalog.peak_w_direct(frame, cols)
                assert row["peak_w_mps"] == direct
                # And the polygon raster agrees with the voxel columns.
                raster = cells_catalog.rasterise_footprint(obj["footprint"], projector)
                assert set(raster.tolist()) == set(cols.tolist())
                checked += 1
    assert checked >= FRAMES - 1
