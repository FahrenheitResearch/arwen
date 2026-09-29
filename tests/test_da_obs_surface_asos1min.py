"""The one-minute ASOS stream through the surface seam, on a real page.

The record under ``tests/fixtures/asos1min_real`` is the real ``rw_asos``
writer's output (``fetch`` and ``decode --product asos1min``) over a real
page of the archive's one-minute ASOS route; its README carries the
commands and digests.  What is pinned here is what the product exists for:
a DA cycle of a few minutes gets a report from each one-minute station at
EVERY analysis, each report enters one analysis only, and the record says
which product it holds.  When ``GPUWM_RW_ASOS`` names a binary the decode is
re-run against the committed CSV and compared.

Everything here is CPU/numpy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gpuwm.da.obs_surface import SurfaceObsConfig, surface_to_gridded_obs
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

FIXTURES = Path(__file__).parent / "fixtures" / "asos1min_real"
RECORD = FIXTURES / "surface_1min.v2.json"
T0 = datetime(2024, 5, 21, 12, 0, tzinfo=timezone.utc)


def _record() -> dict:
    with open(RECORD, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _grid(nx: int = 41, ny: int = 41, dx: float = 4000.0,
          nz: int = 8) -> TargetGrid:
    """A real Lambert domain over the fixture's stations."""

    projection = LambertGrid(
        ref_lat=41.7, ref_lon=-93.6, truelat1=40.0, truelat2=43.0,
        stand_lon=-93.6, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    z_w = np.linspace(280.0, 15280.0, nz + 1)
    return TargetGrid.from_projection(projection, z_w=z_w, terrain_m=None,
                                      name="sfc-1min-test")


def _members(grid: TargetGrid, r: int = 4):
    shape = (r, grid.ny, grid.nx)
    spread = np.linspace(-1.0, 1.0, r)[:, None, None]
    return (np.full(shape, 291.0) + spread,
            np.full(shape, 1.0) + 0.5 * spread,
            np.full(shape, 1.0) - 0.25 * spread)


def test_the_record_is_the_one_minute_product_at_a_one_minute_stride():
    record = _record()
    assert record["schema"] == "gpuwm-obs.asos-surface.v2"
    assert record["provenance"]["product"] == "iem-asos-1min"
    assert record["match_seconds"] == 30
    stamps = [datetime.fromisoformat(v) for v in record["valid_times"]]
    assert {b - a for a, b in zip(stamps, stamps[1:])} == {
        timedelta(minutes=1)}
    # Each one-minute report serves the minute it was taken, no other.
    assert record["reports"]
    for report in record["reports"]:
        assert report["observation_time"] == report["valid_time"]
    served = [(r["station_id"], r["valid_time"]) for r in record["reports"]]
    assert len(served) == len(set(served))


def test_every_analysis_of_a_four_minute_cycle_gets_every_station_once():
    record = _record()
    grid = _grid()
    t2, u10, v10 = _members(grid)
    schedule = [T0 + timedelta(minutes=4 * k) for k in range(8)]
    stations = {r["station_id"] for r in record["reports"]}
    assert len(stations) >= 2
    config = SurfaceObsConfig(temperature_error_k=1.5,
                              wind_speed_error_ms=2.0,
                              max_age_seconds=600.0)
    used = []
    for analysis_time in schedule:
        batches, provenance = surface_to_gridded_obs(
            record, target_grid=grid, analysis_time=analysis_time,
            analysis_times=schedule, config=config,
            simulated_t2=t2, simulated_u10=u10, simulated_v10=v10)
        temperature = batches[0]
        assert temperature.name == "temperature_2m:asos"
        # Every station inside the domain reports at every analysis: the
        # METAR stream reaches only the analyses after the top of the hour.
        assert int(np.asarray(temperature.mask).sum()) == len(stations)
        ages = provenance["cadence"]["report_ages_at_assimilation_s"]
        assert ages["max"] is not None and abs(ages["max"]) <= 60.0
        assert provenance["counts"]["reports_superseded_same_station"] > 0
        used.append(provenance)
    assert "asos1min" in used[0]["cadence"]["note"]


def test_decode_through_the_real_writer_roundtrips():
    """Re-run the real ``rw_asos decode --product asos1min`` over the
    committed CSV; skips unless ``GPUWM_RW_ASOS`` names the binary."""

    exe = os.environ.get("GPUWM_RW_ASOS")
    if not exe or not Path(exe).is_file():
        pytest.skip("GPUWM_RW_ASOS does not name an rw_asos binary")
    out = FIXTURES / "roundtrip.tmp.json"
    try:
        subprocess.run(
            [exe, "decode", "--product", "asos1min",
             "--stations", str(FIXTURES / "stations.json"),
             "--obs", str(FIXTURES / "observations_1min.csv"),
             "--start", "2024-05-21T12:00:00Z",
             "--end", "2024-05-21T12:30:00Z", "--out", str(out)],
            check=True, capture_output=True, timeout=120)
        with open(out, "r", encoding="utf-8") as handle:
            fresh = json.load(handle)
        committed = _record()
        for key in ("schema", "status", "station_table_sha256",
                    "valid_times", "match_seconds", "stations", "reports",
                    "screen"):
            assert fresh[key] == committed[key], key
        assert fresh["provenance"]["product"] == "iem-asos-1min"
        # The METAR decode of the same CSV is refused by name.
        refused = subprocess.run(
            [exe, "decode", "--stations", str(FIXTURES / "stations.json"),
             "--obs", str(FIXTURES / "observations_1min.csv"),
             "--start", "2024-05-21T12:00:00Z",
             "--end", "2024-05-21T12:30:00Z", "--out", str(out)],
            capture_output=True, text=True, timeout=120)
        assert refused.returncode != 0
        assert "--product asos1min" in refused.stderr
    finally:
        if out.exists():
            out.unlink()
