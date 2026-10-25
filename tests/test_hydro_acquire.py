"""Hydro source acquisition: land-only tiles, one fetch per file, retries, a down host (CPU, no network).

Breakage these prevent: the Milton moving-nest prepare (2026-10-21, a 16 GB card) asked 1,001
inputs for a mostly-ocean 23 x 17 degree box, gave up the whole preparation on one SoilGrids
HTTP 504, and its forecast then asked 739 of them again.
"""
import json
from pathlib import Path

import pytest

from gpuwm.hydro import acquire as a

GLO90 = "https://copernicus-dem-90m.s3.amazonaws.com/"
COVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"


def test_tile_names_give_their_cell():
    assert a._cell("Copernicus_DSM_COG_30_N29_00_W090_00_DEM") == (29, -90)
    assert a._cell("ESA_WorldCover_10m_2021_v200_S03W072_Map.tif") == (-3, -72)


def test_only_land_tiles_are_asked():
    sources = json.loads(a.rosters_table_path().read_text())["sources"]
    cells = {"copernicus-glo90": {(29, -90)}}
    land = GLO90 + "Copernicus_DSM_COG_30_N29_00_W090_00_DEM/Copernicus_DSM_COG_30_N29_00_W090_00_DEM.tif"
    sea = GLO90 + "Copernicus_DSM_COG_30_N25_00_W088_00_DEM/Copernicus_DSM_COG_30_N25_00_W088_00_DEM.tif"
    assert a._over_land(land, sources, cells) and not a._over_land(sea, sources, cells)
    # A 3-degree WorldCover tile is asked when any of its nine cells is land.
    assert a._over_land(COVER + "t", sources, cells, 27, -90)
    assert not a._over_land(COVER + "t", sources, cells, 24, -90)
    # A source without a fetched roster is asked tile by tile, as before.
    assert a._over_land(sea, sources, {})


def test_a_transient_answer_retries_with_backoff(tmp_path, monkeypatch):
    calls = []

    class Answer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size=-1):
            data, self.data = getattr(self, "data", b"payload"), b""
            return data

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise a.urllib.error.HTTPError(request.full_url, 504, "Gateway Time-out", {}, None)
        return Answer()

    monkeypatch.setattr(a.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(a.time, "sleep", lambda seconds: None)
    status, size = a._download("https://example.invalid/f", tmp_path / "f")
    assert (status, size, len(calls)) == ("fetched", 7, 3)
    # A permanent answer is not retried.
    calls.clear()

    def gone(request, timeout):
        calls.append(1)
        raise a.urllib.error.HTTPError(request.full_url, 410, "Gone", {}, None)

    monkeypatch.setattr(a.urllib.request, "urlopen", gone)
    with pytest.raises(RuntimeError, match="410"):
        a._download("https://example.invalid/g", tmp_path / "g")
    assert len(calls) == 1


def test_native_vpu_archives_are_versioned_and_shared_components_fetch_once():
    table = json.loads(a.source_table_path().read_text())
    jobs = a._archive_jobs(table, {'conus-nhdplus-v21'}, ['13'])
    native = [j for j in jobs if j.get('region')]
    assert len(native) == 3  # flowlines, catchments, VAA + elevations together
    assert all(j['region'] == '13' and j['url'].endswith('.7z') for j in native)
    assert any('NHDPlusAttributes_07.7z' in j['url'] for j in native)
    assert not any('10U' in j['url'] for j in jobs)
    assert any(j['input'] == 'processing_units' and j['archive'].endswith('.7z') for j in jobs)


def test_sevenzip_unpack_is_reusable_and_preserves_archive_layout(tmp_path):
    py7zr = pytest.importorskip("py7zr")
    source = tmp_path / 'Catchment.shp'
    source.write_bytes(b'layout-proof')
    archive = tmp_path / 'vpu.7z'
    with py7zr.SevenZipFile(archive, 'w') as bundle:
        bundle.write(source, 'NHDPlusRG/NHDPlus13/NHDPlusCatchment/Catchment.shp')
    destination = tmp_path / 'extracted'
    a._extract(archive, destination)
    out = destination / 'NHDPlusRG/NHDPlus13/NHDPlusCatchment/Catchment.shp'
    before = out.stat().st_mtime_ns
    a._extract(archive, destination)
    assert out.read_bytes() == b'layout-proof' and out.stat().st_mtime_ns == before
