"""Domain-wide radar ingest: discovery by coverage, and fail-soft under load.

The single-site builder can afford to die on a bad volume -- there is
nothing else in the cycle to protect.  A domain-wide ingest cannot: run
twenty radars a night and some night a mirror 403s, an antenna is down for
maintenance, or a volume decodes and verifies and still puts nothing on
this grid.  None of those may end the cycle, and none of them may vanish
either.  A cycle that quietly ran on three radars instead of twenty is a
different analysis, and the receipt has to say so.

What this file pins:

* coverage discovery -- a site is in because a measured fraction of the
  domain lies in its range, ranked, thresholded and capped, never named;
* the union path -- N sites merge to one file with N radars on its radar
  axis, and reflectivity from all of them;
* fail-soft, at both levels -- ``ingest_site`` converts any stage's
  exception into an outcome instead of raising, and ``ingest_domain``
  finishes the other sites and records the count that contributed;
* determinism -- the merged radar axis follows the roster, not whichever
  mirror answered first.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.obs import allradar as allradar_mod
from gpuwm.obs.allradar import (
    OUTCOME_EMPTY,
    OUTCOME_FETCH_FAILED,
    OUTCOME_OK,
    SiteCoverage,
    SiteOutcome,
    discover_sites,
    grid_sample_points,
    great_circle_km,
    ingest_domain,
    ingest_site,
    point_in_ring,
    polygon_sample_points,
    site_coverage,
)
from gpuwm.obs.sweeps import Moment, RadarSite, RadarVolume, Sweep
from gpuwm.obs.superob import SuperobParams, superob_volume
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

REF_LAT, REF_LON = 35.3331, -97.2778


def _grid(nx: int = 41, ny: int = 41, dx: float = 2000.0, nz: int = 10,
          top_m: float = 10000.0) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=REF_LAT, ref_lon=REF_LON, truelat1=33.0, truelat2=37.0,
        stand_lon=REF_LON, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, top_m, nz + 1), name="analytic")


def _volume_at(grid: TargetGrid, *, site_id: str, j: int, i: int,
               gates: int = 40) -> RadarVolume:
    """A synthetic volume whose antenna sits at grid index ``(j, i)``."""
    azimuth = np.arange(0.0, 360.0, 10.0, dtype=np.float32)
    ref = np.tile(np.linspace(20.0, 45.0, gates)[None, :],
                  (azimuth.size, 1)).astype(np.float32)
    vel = np.tile(np.linspace(-15.0, 15.0, gates)[None, :],
                  (azimuth.size, 1)).astype(np.float32)
    sweep = Sweep(
        sweep_index=0, elevation_number=1, elevation_angle_deg=0.5,
        nyquist_velocity_ms=32.0, start_status=3, end_status=2,
        cut_sector=0, complete=True,
        azimuth_deg=azimuth,
        elevation_deg=np.full(azimuth.size, 0.5, dtype=np.float32),
        moments={
            "REF": Moment("REF", "dBZ", gates, 2125.0, 250.0, ref),
            "VEL": Moment("VEL", "m/s", gates, 2125.0, 250.0, vel),
        })
    from pathlib import Path as _P
    return RadarVolume(
        site=RadarSite(id=site_id, name="synthetic",
                       lat_deg=float(grid.lat[j, i]),
                       lon_deg=float(grid.lon[j, i]),
                       alt_m=0.0, source="test"),
        valid_time="2026-08-01T11:30:00Z", station_id=site_id,
        volume_file=f"{site_id}20260801_113000_V06",
        volume_sha256="0" * 64, volume_bytes=8102058,
        pack_path=_P("synthetic.rdrpack"), pack_sha256="1" * 64,
        params={"moments": ["REF", "VEL"], "max_range_km": 250.0},
        framing={"magic": "AR2V0006", "block_count": 1},
        sweeps=(sweep,))


def _contribution(grid, site_id, j, i, params):
    return superob_volume(_volume_at(grid, site_id=site_id, j=j, i=i),
                          grid, params=params)


def _coverage(site_id, fraction=1.0):
    return SiteCoverage(site=site_id, lat_deg=REF_LAT, lon_deg=REF_LON,
                        alt_m=0.0, coverage_fraction=fraction,
                        nearest_km=0.0, farthest_km=10.0)


# ---------------------------------------------------------------------------
# Geometry and discovery.
# ---------------------------------------------------------------------------


def test_great_circle_matches_a_known_separation():
    # One degree of latitude on a 6371 km sphere.
    assert great_circle_km(35.0, -97.0, 36.0, -97.0) == pytest.approx(
        111.19, abs=0.05)


def test_point_in_ring_is_inside_outside_and_broadcastable():
    ring = [[-100.0, 30.0], [-90.0, 30.0], [-90.0, 40.0], [-100.0, 40.0]]
    assert point_in_ring(-95.0, 35.0, ring)
    assert not point_in_ring(-105.0, 35.0, ring)
    assert not point_in_ring(-95.0, 45.0, ring)


def test_polygon_sample_points_land_inside_the_ring():
    ring = [[-100.0, 30.0], [-90.0, 30.0], [-90.0, 40.0], [-100.0, 40.0]]
    lat, lon = polygon_sample_points(ring, samples=256)
    assert lat.size > 100
    assert np.all((lat >= 30.0) & (lat <= 40.0))
    assert np.all((lon >= -100.0) & (lon <= -90.0))


def test_coverage_fraction_is_a_measured_fraction_not_a_flag():
    grid = _grid()
    lat, lon = grid_sample_points(grid)
    centred = site_coverage(
        {"id": "aaaa", "lat_deg": float(grid.lat[20, 20]),
         "lon_deg": float(grid.lon[20, 20])},
        lat, lon, range_km=250.0)
    # A 41x41 grid at 2 km is ~80 km across: a centred 250 km radar sees
    # all of it.
    assert centred.coverage_fraction == 1.0
    assert centred.site == "AAAA"

    far = site_coverage(
        {"id": "bbbb", "lat_deg": float(grid.lat[20, 20]) + 8.0,
         "lon_deg": float(grid.lon[20, 20])},
        lat, lon, range_km=250.0)
    assert far.coverage_fraction == 0.0


def test_discovery_ranks_by_coverage_and_honours_threshold_and_cap():
    grid = _grid()
    lat, lon = grid_sample_points(grid)
    catalog = [
        {"id": "near", "lat_deg": float(grid.lat[20, 20]),
         "lon_deg": float(grid.lon[20, 20])},
        {"id": "edge", "lat_deg": float(grid.lat[20, 20]) + 0.55,
         "lon_deg": float(grid.lon[20, 20])},
        {"id": "gone", "lat_deg": float(grid.lat[20, 20]) + 40.0,
         "lon_deg": float(grid.lon[20, 20])},
    ]
    found = discover_sites(catalog, lat, lon, range_km=60.0)
    ids = [c.site for c in found]
    # "gone" has no overlap and is not a candidate at all; "near" outranks
    # "edge" because it sees more of the domain.
    assert "GONE" not in ids
    assert ids[0] == "NEAR"
    assert found[0].coverage_fraction >= found[-1].coverage_fraction

    assert [c.site for c in discover_sites(
        catalog, lat, lon, range_km=60.0, limit=1)] == ["NEAR"]
    assert discover_sites(catalog, lat, lon, range_km=60.0,
                          min_coverage_fraction=1.0) == [
        c for c in found if c.coverage_fraction >= 1.0]


# ---------------------------------------------------------------------------
# The union path.
# ---------------------------------------------------------------------------


def test_n_sites_merge_into_one_file_with_n_radars(monkeypatch):
    """Three antennas, one observation set, three entries on the radar axis."""
    grid = _grid()
    params = SuperobParams()
    placed = {"AAAA": (20, 12), "BBBB": (20, 20), "CCCC": (20, 28)}

    def fake_site(binary, cov, **kw):
        j, i = placed[cov.site]
        contribution = _contribution(grid, cov.site, j, i, params)
        return contribution, SiteOutcome(site=cov.site, status=OUTCOME_OK,
                                         coverage_fraction=1.0)

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    result = ingest_domain(
        None, [_coverage(s) for s in placed], grid=grid, work_dir="unused",
        valid_time="2026-08-01T11:30:00Z", params=params, workers=3)

    assert len(result.contributing) == 3
    obs = result.observations
    assert [r["id"] for r in obs.radars] == ["AAAA", "BBBB", "CCCC"]
    # Radial velocity keeps one plane per radar; reflectivity is merged.
    assert obs.vr_obs.shape[0] == 3
    assert obs.z_mask.sum() > 0
    payload = result.to_payload()
    assert payload["sites_contributing"] == 3
    assert payload["sites_discovered"] == 3


def test_merged_reflectivity_covers_more_than_any_single_radar(monkeypatch):
    """The union is a union, not the first radar that answered."""
    grid = _grid()
    params = SuperobParams()
    placed = {"AAAA": (20, 8), "BBBB": (20, 32)}

    def fake_site(binary, cov, **kw):
        j, i = placed[cov.site]
        return (_contribution(grid, cov.site, j, i, params),
                SiteOutcome(site=cov.site, status=OUTCOME_OK))

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    both = ingest_domain(None, [_coverage(s) for s in placed], grid=grid,
                         work_dir="unused", valid_time="t", params=params,
                         workers=2).observations

    one = ingest_domain(None, [_coverage("AAAA")], grid=grid,
                        work_dir="unused", valid_time="t", params=params,
                        workers=1).observations

    assert both.z_mask.sum() > one.z_mask.sum()


def test_radar_axis_follows_the_roster_not_the_completion_order(monkeypatch):
    """Determinism: a slow mirror must not reorder the file."""
    import time as _time

    grid = _grid()
    params = SuperobParams()
    placed = {"AAAA": (20, 12), "BBBB": (20, 20), "CCCC": (20, 28)}
    # The first site in the roster finishes last.
    delays = {"AAAA": 0.15, "BBBB": 0.05, "CCCC": 0.0}

    def fake_site(binary, cov, **kw):
        _time.sleep(delays[cov.site])
        j, i = placed[cov.site]
        return (_contribution(grid, cov.site, j, i, params),
                SiteOutcome(site=cov.site, status=OUTCOME_OK))

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    result = ingest_domain(None, [_coverage(s) for s in placed], grid=grid,
                           work_dir="unused", valid_time="t", params=params,
                           workers=3)
    assert [r["id"] for r in result.observations.radars] == [
        "AAAA", "BBBB", "CCCC"]
    assert [o.site for o in result.outcomes] == ["AAAA", "BBBB", "CCCC"]


# ---------------------------------------------------------------------------
# Fail-soft.
# ---------------------------------------------------------------------------


def test_ingest_site_converts_a_fetch_failure_into_an_outcome(monkeypatch,
                                                              tmp_path):
    """The no-raise contract, at the level that has to honour it."""
    import gpuwm.obs.radar_source as source_mod

    def boom(*a, **kw):
        raise source_mod.RadarSourceError("the mirror answered 403")

    monkeypatch.setattr(source_mod, "acquire_volume", boom)
    contribution, outcome = ingest_site(
        None, _coverage("AAAA"), grid=_grid(), work_dir=tmp_path,
        valid_time="2026-08-01T11:30:00Z", params=SuperobParams())

    assert contribution is None
    assert outcome.status == OUTCOME_FETCH_FAILED
    assert "403" in outcome.reason
    assert not outcome.contributed
    # The stage that failed is still timed: a mirror that hangs for ninety
    # seconds before failing is a different problem to one that refuses at
    # once, and the receipt should be able to tell them apart.
    assert "fetch" in outcome.seconds


def test_one_site_failing_does_not_stop_the_cycle(monkeypatch):
    """The headline behaviour: record and continue, never silently drop."""
    grid = _grid()
    params = SuperobParams()
    placed = {"AAAA": (20, 12), "BBBB": (20, 20), "CCCC": (20, 28)}

    def fake_site(binary, cov, **kw):
        if cov.site == "BBBB":
            return None, SiteOutcome(site=cov.site,
                                     status=OUTCOME_FETCH_FAILED,
                                     reason="the mirror answered 403")
        j, i = placed[cov.site]
        return (_contribution(grid, cov.site, j, i, params),
                SiteOutcome(site=cov.site, status=OUTCOME_OK))

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    result = ingest_domain(None, [_coverage(s) for s in placed], grid=grid,
                           work_dir="unused", valid_time="t", params=params,
                           workers=3)

    # The cycle produced an analysis-ready set from the survivors ...
    assert result.observations is not None
    assert [r["id"] for r in result.observations.radars] == ["AAAA", "CCCC"]

    payload = result.to_payload()
    assert payload["sites_contributing"] == 2
    assert payload["sites_discovered"] == 3
    assert payload["sites_by_status"][OUTCOME_FETCH_FAILED] == 1

    # ... and the site that failed is IN the receipt, with a reason.  This
    # is the assertion that separates fail-soft from silently dropping.
    failed = [s for s in payload["sites"] if s["site"] == "BBBB"]
    assert len(failed) == 1
    assert failed[0]["status"] == OUTCOME_FETCH_FAILED
    assert "403" in failed[0]["reason"]


def test_an_exception_escaping_a_site_still_does_not_stop_the_cycle(
        monkeypatch):
    """Belt and braces: even a broken ingest_site cannot kill the cycle."""
    grid = _grid()
    params = SuperobParams()

    def fake_site(binary, cov, **kw):
        if cov.site == "BBBB":
            raise RuntimeError("a bug, not a bad night")
        return (_contribution(grid, cov.site, 20, 20, params),
                SiteOutcome(site=cov.site, status=OUTCOME_OK))

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    result = ingest_domain(None, [_coverage("AAAA"), _coverage("BBBB")],
                           grid=grid, work_dir="unused", valid_time="t",
                           params=params, workers=2)

    assert len(result.contributing) == 1
    bad = [o for o in result.outcomes if o.site == "BBBB"][0]
    assert not bad.contributed
    assert "a bug, not a bad night" in bad.reason


def test_every_site_failing_yields_no_observations_but_a_full_receipt(
        monkeypatch):
    def fake_site(binary, cov, **kw):
        return None, SiteOutcome(site=cov.site, status=OUTCOME_FETCH_FAILED,
                                 reason="down")

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    result = ingest_domain(None, [_coverage("AAAA"), _coverage("BBBB")],
                           grid=_grid(), work_dir="unused", valid_time="t",
                           params=SuperobParams(), workers=2)
    assert result.observations is None
    payload = result.to_payload()
    assert payload["sites_contributing"] == 0
    assert payload["sites_discovered"] == 2
    assert len(payload["sites"]) == 2


def test_a_site_that_observed_nothing_is_recorded_and_not_counted(monkeypatch):
    """Decoded, verified, and still contributed nothing: a real outcome."""
    grid = _grid()
    params = SuperobParams()

    def fake_site(binary, cov, **kw):
        if cov.site == "BBBB":
            return None, SiteOutcome(site=cov.site, status=OUTCOME_EMPTY,
                                     cells={"z": 0, "vr": 0})
        return (_contribution(grid, cov.site, 20, 20, params),
                SiteOutcome(site=cov.site, status=OUTCOME_OK))

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    payload = ingest_domain(None, [_coverage("AAAA"), _coverage("BBBB")],
                            grid=grid, work_dir="unused", valid_time="t",
                            params=params, workers=2).to_payload()
    assert payload["sites_contributing"] == 1
    assert payload["sites_by_status"][OUTCOME_EMPTY] == 1


def test_superob_runs_on_the_compute_pool_not_the_fetch_pool(monkeypatch,
                                                             tmp_path):
    """The CPU-bound stage must be bounded separately from the I/O one.

    Measured on Seoul: running superob at fetch concurrency tripled its
    cost per site (3.5 s at 8 workers, 10.5 s at 32) for no wall-clock
    gain, because the fetch pool is sized for a transpacific round trip
    and the box has sixteen cores.  This pins that superob is submitted to
    the pool that is sized for arithmetic.
    """
    import gpuwm.obs.nexrad as nexrad_mod
    import gpuwm.obs.radar_source as source_mod
    import gpuwm.obs.sweeps as sweeps_mod

    grid = _grid()
    seen = {}

    class _Selected:
        filename = "AAAA20260801_113000_V06"
        path = tmp_path / "vol"
        sha256 = "0" * 64
        feed = "archive"
        valid_time = "2026-08-01T11:30:00Z"
        offset_seconds = 0.0

    _Selected.path.write_bytes(b"x")
    monkeypatch.setattr(source_mod, "acquire_volume",
                        lambda *a, **k: _Selected())
    monkeypatch.setattr(nexrad_mod, "run_decode", lambda *a, **k: {})
    monkeypatch.setattr(nexrad_mod, "run_verify",
                        lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(sweeps_mod, "read_sweep_pack",
                        lambda p: _volume_at(grid, site_id="AAAA",
                                             j=20, i=20))

    class _Pool:
        def submit(self, fn):
            seen["submitted"] = True
            return futures_done(fn())

    contribution, outcome = ingest_site(
        None, _coverage("AAAA"), grid=grid, work_dir=tmp_path,
        valid_time="2026-08-01T11:30:00Z", params=SuperobParams(),
        compute_pool=_Pool())

    assert seen.get("submitted"), "superob did not reach the compute pool"
    assert contribution is not None
    assert outcome.status == OUTCOME_OK
    # Queue wait is recorded apart from the arithmetic, so a receipt can
    # tell a slow superob from a queued one.
    assert "superob_queued" in outcome.seconds
    assert "superob" in outcome.seconds


def futures_done(value):
    import concurrent.futures as _f
    fut = _f.Future()
    fut.set_result(value)
    return fut


def test_the_receipt_records_each_site_reach_window(monkeypatch, tmp_path):
    """A receipt saying only "contributed" cannot show it did so on 1% of
    the grid.  The window is what makes the memory claim auditable."""
    import gpuwm.obs.nexrad as nexrad_mod
    import gpuwm.obs.radar_source as source_mod
    import gpuwm.obs.sweeps as sweeps_mod

    grid = _grid()

    class _Selected:
        filename = "AAAA20260801_113000_V06"
        path = tmp_path / "vol"
        sha256 = "0" * 64
        feed = "archive"
        valid_time = "2026-08-01T11:30:00Z"
        offset_seconds = 0.0

    _Selected.path.write_bytes(b"x")
    monkeypatch.setattr(source_mod, "acquire_volume",
                        lambda *a, **k: _Selected())
    monkeypatch.setattr(nexrad_mod, "run_decode", lambda *a, **k: {})
    monkeypatch.setattr(nexrad_mod, "run_verify",
                        lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(sweeps_mod, "read_sweep_pack",
                        lambda p: _volume_at(grid, site_id="AAAA",
                                             j=20, i=20))

    _, outcome = ingest_site(
        None, _coverage("AAAA"), grid=grid, work_dir=tmp_path,
        valid_time="2026-08-01T11:30:00Z", params=SuperobParams())

    assert outcome.status == OUTCOME_OK
    window = outcome.to_payload()["window"]
    for key in ("j0", "j1", "i0", "i1", "nj", "ni", "nz", "cells"):
        assert key in window, f"receipt window missing {key}"
    assert window["cells"] == window["nz"] * window["nj"] * window["ni"]
    assert window["nz"] == grid.nz

    # And where the antenna HEIGHT came from.  The height is the ray
    # origin: the vendored site table carries an unset elevation
    # placeholder for 130 of its 141 entries, the decoder refuses rather
    # than accept one, and this is what lets a cycle show that every
    # contributing radar knew its own height instead of assuming it.
    antenna = outcome.to_payload()["antenna"]
    assert antenna["source"] == "test"
    assert antenna["alt_m"] == 0.0


def test_the_receipt_times_every_stage_separately(monkeypatch):
    """Fetch, decode and superob are different bottlenecks on different
    nights, and a receipt that sums them cannot say which."""
    grid = _grid()
    params = SuperobParams()

    def fake_site(binary, cov, **kw):
        outcome = SiteOutcome(site=cov.site, status=OUTCOME_OK)
        outcome.seconds = {"fetch": 1.0, "decode": 2.0, "verify": 0.5,
                           "superob": 4.0}
        return (_contribution(grid, cov.site, 20, 20, params), outcome)

    monkeypatch.setattr(allradar_mod, "ingest_site", fake_site)
    result = ingest_domain(None, [_coverage("AAAA"), _coverage("BBBB")],
                           grid=grid, work_dir="unused", valid_time="t",
                           params=params, workers=2)
    assert result.seconds["fetch_total"] == pytest.approx(2.0)
    assert result.seconds["decode_total"] == pytest.approx(4.0)
    assert result.seconds["superob_total"] == pytest.approx(8.0)
