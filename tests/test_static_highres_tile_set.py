"""Whole-tile terrain sets are fetched several tiles at a time, in tile order.

Origin: the 1 km CONUS preparation fetched about 1,000 Copernicus GLO-30
tiles (40 GB) one stream at a time, at about 8-10 MB/s, so the terrain
fetch alone ran over an hour and the preparation kit carried its own
parallel prefill script.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from gpuwm.static import highres_fetch
from gpuwm.static.highres_fetch import FetchedFile, SourceAbsent


def _fetched(path):
    return FetchedFile(path=Path(path), url=f"u:{Path(path).name}", sha256="0" * 64,
                       bytes=1, fetched_utc="", cache_hit=False)


def test_tiles_are_fetched_concurrently_and_returned_in_tile_order(monkeypatch):
    monkeypatch.setattr(highres_fetch, "TILE_FETCH_WORKERS", 4)
    tiles = [f"T{index:02d}" for index in range(12)]
    absent = {"T03", "T07"}
    gate = threading.Barrier(4, timeout=10)
    lock = threading.Lock()
    live = {"now": 0, "most": 0}

    def fake_fetch(url, path, *, urlopen=None):
        with lock:
            live["now"] += 1
            live["most"] = max(live["most"], live["now"])
        try:
            if url in ("u:T00", "u:T01", "u:T02", "u:T03"):
                gate.wait()  # the first four run at the same time
            time.sleep(0.01 * (12 - int(url[-2:])))  # later tiles finish first
            if url[2:] in absent:
                raise SourceAbsent(url)
            return _fetched(path)
        finally:
            with lock:
                live["now"] -= 1

    monkeypatch.setattr(highres_fetch, "fetch_file", fake_fetch)
    fetched, missing = highres_fetch._fetch_tile_set(
        tiles, lambda tile: f"u:{tile}", lambda tile: Path("/c") / tile)
    assert [item.path.name for item in fetched] == [t for t in tiles if t not in absent]
    assert missing == ("T03", "T07")
    assert live["most"] == 4


def test_a_failed_tile_stops_the_queue_and_is_raised(monkeypatch):
    monkeypatch.setattr(highres_fetch, "TILE_FETCH_WORKERS", 2)
    started = []

    def fake_fetch(url, path, *, urlopen=None):
        started.append(url)
        if url == "u:T01":
            raise OSError("disk full")
        time.sleep(0.05)
        return _fetched(path)

    monkeypatch.setattr(highres_fetch, "fetch_file", fake_fetch)
    with pytest.raises(OSError, match="disk full"):
        highres_fetch._fetch_tile_set(
            [f"T{index:02d}" for index in range(40)], lambda tile: f"u:{tile}",
            lambda tile: Path("/c") / tile)
    assert len(started) < 40


def test_copernicus_tiles_go_through_the_tile_set(monkeypatch, tmp_path):
    seen = []

    def fake_set(tiles, url_for, path_for, *, urlopen=None):
        tiles = list(tiles)
        seen.append((url_for(tiles[0]), path_for(tiles[0])))
        return (), tuple(tiles)

    monkeypatch.setattr(highres_fetch, "_fetch_tile_set", fake_set)
    bbox = highres_fetch.FootprintBBox(39.2, 39.8, -104.8, -104.2)
    fetched, absent = highres_fetch.fetch_copernicus_dem_tiles(bbox, tmp_path)
    assert fetched == () and absent == ("N39_00_W105_00",)
    url, path = seen[0]
    assert url == highres_fetch.COPERNICUS_DEM_TILE_URL.format(tile="N39_00_W105_00")
    assert path == tmp_path / "copernicus_dem_glo30" / "Copernicus_DSM_COG_10_N39_00_W105_00_DEM.tif"
