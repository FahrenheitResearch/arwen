"""Every radar that can see the domain, ingested as one observation set.

The single-site path (``tools/obs_radar_grid_build.py``) answers "what did
this antenna see?".  This module answers the question an analysis actually
asks: "what did every antenna that can see this domain see?" -- discover
the sites whose coverage intersects a domain, fetch and decode their
volumes concurrently, superob each onto the analysis grid, and merge.

Three properties this module exists to guarantee, none of which the
single-site path needed:

**Discovery is by coverage, not by name.**  A site belongs to a cycle when
some fraction of the analysis domain lies inside its usable range, and
that fraction is computed against the georeference the observations will
land on -- not against a bounding box, and never from a hardcoded roster.
No case names live here; sites are a *result*, and the catalogue is the
Rust site table read through ``rw_nexrad sites``.

**One site's bad night is not the cycle's bad night.**  A radar down for
maintenance, a mirror that 403s, a volume that fails its pack verify: each
is recorded against that site and the cycle continues with the rest.  The
one thing never permitted is a silent drop -- every discovered site
appears in the receipt with an outcome, and the number that actually
contributed is a field, so a cycle that quietly ran on two radars instead
of twenty cannot look like a cycle that ran on twenty.

**Concurrency is in the fetch, because that is where the time goes.**
Measured from Seoul against the Unidata mirror, a single stream pulls
about 2 MB/s while sixty-four concurrent streams pull about 37 MB/s: the
per-request latency, not the bandwidth, is what one-site-at-a-time spends
its night paying.  Decode and superob release the GIL (subprocess and
numpy respectively), so the same pool covers all three stages.
"""

from __future__ import annotations

import concurrent.futures as futures
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

__all__ = [
    "SITE_SCHEMA",
    "INGEST_SCHEMA",
    "EARTH_RADIUS_KM",
    "SiteCoverage",
    "SiteOutcome",
    "AllRadarIngest",
    "great_circle_km",
    "point_in_ring",
    "polygon_sample_points",
    "grid_sample_points",
    "site_coverage",
    "discover_sites",
    "ingest_site",
    "ingest_domain",
]

SITE_SCHEMA = "gpuwm-obs.radar-site-coverage.v1"
INGEST_SCHEMA = "gpuwm-obs.all-radar-ingest.v1"

#: Matches ``tools/da_nowcast_launcher.py``'s constant so two coverage
#: statements about the same site agree to the metre.
EARTH_RADIUS_KM = 6371.0

#: Outcome vocabulary.  ``ok`` contributed; every other value is a site
#: that was discovered, attempted, and did not contribute -- which is a
#: fact the receipt carries rather than a row it omits.
OUTCOME_OK = "ok"
OUTCOME_FETCH_FAILED = "fetch-failed"
OUTCOME_DECODE_FAILED = "decode-failed"
OUTCOME_VERIFY_FAILED = "verify-failed"
OUTCOME_SUPEROB_FAILED = "superob-failed"
OUTCOME_EMPTY = "no-observations"


# ---------------------------------------------------------------------------
# Geometry.  Kept dependency-free on purpose: this runs on every node that
# ingests, and a coverage test is not worth a shapely build.
# ---------------------------------------------------------------------------


def great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km; scalars or numpy arrays, broadcast."""
    import numpy as np

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = (np.sin(dp / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_in_ring(lon: float, lat: float, ring: Sequence[Sequence[float]]):
    """Ray-casting point-in-polygon on a GeoJSON ``[lon, lat]`` ring.

    Planar in lon/lat.  That is the right approximation here and not a
    shortcut: the ring is a model domain a few thousand kilometres across,
    the test decides whether to fetch a radar volume, and a site sitting
    exactly on the boundary is by construction a site whose coverage
    fraction is near zero and which the fraction threshold judges anyway.
    """
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > lat) != (y2 > lat):
            xin = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if xin > lon:
                inside = not inside
    return inside


def polygon_sample_points(ring: Sequence[Sequence[float]], *,
                          samples: int = 4096):
    """A lattice of ``(lat, lon)`` points inside a polygon ring.

    Coverage is a fraction of a domain, so the domain needs a set of points
    to take the fraction over.  With a grid in hand use
    :func:`grid_sample_points`; this is for discovery before a
    georeference exists, which is the order ``da_nowcast`` works in.
    """
    import numpy as np

    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    lo_lon, hi_lon = min(lons), max(lons)
    lo_lat, hi_lat = min(lats), max(lats)
    side = max(2, int(math.sqrt(max(1, samples))))
    gx, gy = np.meshgrid(
        np.linspace(lo_lon, hi_lon, side),
        np.linspace(lo_lat, hi_lat, side),
    )
    keep = [(float(la), float(lo))
            for lo, la in zip(gx.reshape(-1), gy.reshape(-1))
            if point_in_ring(lo, la, ring)]
    if not keep:
        # A ring too thin for the lattice to land in still has a centroid,
        # and one point is a better coverage statement than a refusal.
        return (np.array([sum(lats) / len(lats)]),
                np.array([sum(lons) / len(lons)]))
    return (np.array([p[0] for p in keep]), np.array([p[1] for p in keep]))


def grid_sample_points(grid, *, stride: int = 1):
    """``(lat, lon)`` of the analysis grid's mass points.

    The honest denominator for "what fraction of the domain can this radar
    see": the points the observations will actually be superobbed onto.
    """
    import numpy as np

    lat = np.asarray(grid.lat)[::stride, ::stride].reshape(-1)
    lon = np.asarray(grid.lon)[::stride, ::stride].reshape(-1)
    return lat, lon


@dataclass(frozen=True)
class SiteCoverage:
    """One site's geometric relationship to the domain."""

    site: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    #: Fraction of sampled domain points within ``range_km`` of the antenna.
    coverage_fraction: float
    #: Distance to the nearest and farthest sampled domain point.
    nearest_km: float
    farthest_km: float
    elevation_known: bool = True

    def to_payload(self) -> dict:
        return {
            "schema": SITE_SCHEMA,
            "site": self.site,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "alt_m": self.alt_m,
            "coverage_fraction": self.coverage_fraction,
            "nearest_km": self.nearest_km,
            "farthest_km": self.farthest_km,
            "elevation_known": self.elevation_known,
        }


def site_coverage(site: dict, sample_lat, sample_lon, *,
                  range_km: float) -> SiteCoverage:
    """Score one catalogue row against the domain's sample points."""
    import numpy as np

    lat = float(site["lat_deg"])
    lon = float(site["lon_deg"])
    d = great_circle_km(lat, lon, sample_lat, sample_lon)
    return SiteCoverage(
        site=str(site["id"]).upper(),
        lat_deg=lat, lon_deg=lon,
        alt_m=float(site.get("alt_m") or 0.0),
        coverage_fraction=float(np.mean(d <= range_km)),
        nearest_km=float(np.min(d)),
        farthest_km=float(np.max(d)),
        elevation_known=bool(site.get("elevation_known", True)),
    )


def discover_sites(catalog: Iterable[dict], sample_lat, sample_lon, *,
                   range_km: float,
                   min_coverage_fraction: float = 0.0,
                   limit: int | None = None) -> list[SiteCoverage]:
    """Every catalogue site that sees enough of the domain, best first.

    ``min_coverage_fraction`` of 0.0 keeps every site with any overlap at
    all, which is the right default for assimilation: a radar clipping the
    corner of the domain still contributes real observations there, and
    the reach reject in the LETKF means an unhelpful one costs nothing.
    Raising it is how a caller says "only radars that see a real share of
    this domain", which is a different and equally legitimate question.
    """
    scored = [site_coverage(s, sample_lat, sample_lon, range_km=range_km)
              for s in catalog]
    keep = [c for c in scored
            if c.coverage_fraction > 0.0
            and c.coverage_fraction >= min_coverage_fraction]
    keep.sort(key=lambda c: (-c.coverage_fraction, c.nearest_km, c.site))
    return keep[:limit] if limit is not None else keep


# ---------------------------------------------------------------------------
# Per-site ingest.
# ---------------------------------------------------------------------------


@dataclass
class SiteOutcome:
    """What happened to one site this cycle -- success or otherwise.

    Every discovered site gets one of these.  A site that failed is not
    dropped from the receipt: ``status`` names the stage that failed and
    ``reason`` carries the message, so "nineteen of twenty radars" is
    always distinguishable from "twenty radars" in the record.
    """

    site: str
    status: str
    reason: str | None = None
    coverage_fraction: float = 0.0
    valid_time: str | None = None
    volume: str | None = None
    volume_sha256: str | None = None
    volume_bytes: int = 0
    feed: str | None = None
    offset_seconds: float | None = None
    seconds: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)
    #: The sub-box of the analysis grid this radar's accumulators covered.
    #: Recorded because it is the difference between an ingest that fits a
    #: continental domain and one that does not: a receipt that says only
    #: "this site contributed" cannot show that it did so on 1% of the
    #: grid's cells rather than all of them.
    window: dict = field(default_factory=dict)
    #: Antenna height and where it came from; see `ingest_site`.
    antenna: dict = field(default_factory=dict)

    @property
    def contributed(self) -> bool:
        return self.status == OUTCOME_OK

    def to_payload(self) -> dict:
        return {
            "site": self.site,
            "status": self.status,
            "reason": self.reason,
            "coverage_fraction": self.coverage_fraction,
            "valid_time": self.valid_time,
            "volume": self.volume,
            "volume_sha256": self.volume_sha256,
            "volume_bytes": self.volume_bytes,
            "feed": self.feed,
            "offset_seconds": self.offset_seconds,
            "seconds": dict(self.seconds),
            "cells": dict(self.cells),
            "window": dict(self.window),
            "antenna": dict(self.antenna),
        }


def ingest_site(binary, coverage: SiteCoverage, *, grid, work_dir: Path,
                valid_time: datetime, params, source: str = "auto",
                bucket: str | None = None, live_bucket: str | None = None,
                cache_dir: Path | None = None,
                max_offset_seconds: float = 480.0,
                allow_partial: bool = False,
                min_chunks: int | None = None,
                max_elevation_deg: float = 20.0,
                compute_pool=None):
    """Fetch, decode, verify and superob one site.  Never raises.

    Returns ``(contribution_or_None, SiteOutcome)``.  The no-raise contract
    is the whole point: this runs inside a thread pool over every radar in
    a domain, and an exception escaping here would take out the cycle over
    one antenna's bad night.  The failure is turned into an outcome and the
    caller decides -- which, for assimilation, means continuing.
    """
    from gpuwm.obs.nexrad import run_decode, run_verify
    from gpuwm.obs.radar_source import acquire_volume
    from gpuwm.obs.superob import superob_volume
    from gpuwm.obs.sweeps import read_sweep_pack

    site = coverage.site
    out = SiteOutcome(site=site, status=OUTCOME_OK,
                      coverage_fraction=coverage.coverage_fraction)
    site_dir = Path(work_dir) / site
    site_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        selected = acquire_volume(
            binary, site=site, out_dir=site_dir, valid_time=valid_time,
            source=source, bucket=bucket, live_bucket=live_bucket,
            cache_dir=cache_dir, max_offset_seconds=max_offset_seconds,
            allow_partial=allow_partial, min_chunks=min_chunks)
    except Exception as exc:  # noqa: BLE001 - fail-soft is the contract
        out.status = OUTCOME_FETCH_FAILED
        out.reason = f"{type(exc).__name__}: {exc}"
        out.seconds["fetch"] = time.perf_counter() - t0
        return None, out
    out.seconds["fetch"] = time.perf_counter() - t0
    out.volume = selected.filename
    out.volume_sha256 = selected.sha256
    out.feed = selected.feed
    out.valid_time = str(selected.valid_time)
    out.offset_seconds = selected.offset_seconds
    try:
        out.volume_bytes = int(Path(selected.path).stat().st_size)
    except OSError:
        out.volume_bytes = 0

    t1 = time.perf_counter()
    pack_path = site_dir / (selected.filename + ".pack")
    try:
        run_decode(binary, volume=selected.path, out=pack_path,
                   moments=("REF", "VEL"),
                   max_range_km=params.max_range_km,
                   max_elevation_deg=max_elevation_deg)
    except Exception as exc:  # noqa: BLE001
        out.status = OUTCOME_DECODE_FAILED
        out.reason = f"{type(exc).__name__}: {exc}"
        out.seconds["decode"] = time.perf_counter() - t1
        return None, out
    out.seconds["decode"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    try:
        verify = run_verify(binary, pack=pack_path)
    except Exception as exc:  # noqa: BLE001
        out.status = OUTCOME_VERIFY_FAILED
        out.reason = f"{type(exc).__name__}: {exc}"
        out.seconds["verify"] = time.perf_counter() - t2
        return None, out
    out.seconds["verify"] = time.perf_counter() - t2
    if verify.get("status") != "PASS":
        out.status = OUTCOME_VERIFY_FAILED
        out.reason = f"pack verify did not PASS: {verify.get('status')!r}"
        return None, out

    # Superob is the one CPU-bound stage, and it must not be run at fetch
    # concurrency.  Measured on Seoul over 24 sites: at 8 workers superob
    # cost 3.5 s per site, at 32 it cost 10.5 s for the same work, while
    # end-to-end wall barely moved (27.7 s to 20.3 s).  That is
    # oversubscription -- the fetch pool is sized for a transpacific
    # round trip and there are only so many cores.  Handing this stage to
    # a separate, core-sized pool lets the fetch pool stay wide without
    # the arithmetic paying for it.  The waiting is recorded separately so
    # a receipt can tell a slow superob from a queued one.
    def _superob():
        vol = read_sweep_pack(pack_path)
        return superob_volume(vol, grid, params=params), vol.site

    t3 = time.perf_counter()
    try:
        if compute_pool is not None:
            handle = compute_pool.submit(_superob)
            out.seconds["superob_queued"] = time.perf_counter() - t3
            t3 = time.perf_counter()
            contribution, antenna = handle.result()
        else:
            contribution, antenna = _superob()
    except Exception as exc:  # noqa: BLE001
        out.status = OUTCOME_SUPEROB_FAILED
        out.reason = f"{type(exc).__name__}: {exc}"
        out.seconds["superob"] = time.perf_counter() - t3
        return None, out
    out.seconds["superob"] = time.perf_counter() - t3

    # A contribution carries per-cell gate COUNTS, not the reduced masks --
    # the masks only exist after the merge, which is exactly the step this
    # site might not reach.  A cell with at least one accepted gate is a
    # cell this site put an observation in.
    z_cells = int((contribution.z_count > 0).sum())
    vr_cells = int((contribution.vr_count > 0).sum())
    out.cells = {"z": z_cells, "vr": vr_cells,
                 "vr_rejected": int(contribution.vr_rejected.sum())}
    # The reach box these counts were taken over.  The counts above are
    # whole-array reductions and stay correct under windowing because the
    # window is a superset of the nonzero support -- superob_volume raises
    # rather than let that stop being true.
    out.window = contribution.window_payload()
    # Where this antenna's HEIGHT came from.  The height is the ray
    # origin, so a wrong one puts every gate in the volume at the wrong
    # model level.  The decoder refuses rather than guess -- the vendored
    # site table carries an unset elevation placeholder for 130 of its 141
    # entries -- so a contributing site always has a real height; this
    # records which source supplied it, so "every radar in this cycle knew
    # its own antenna height" is a checkable claim rather than an
    # assumption.
    out.antenna = {"alt_m": float(antenna.alt_m),
                   "source": str(antenna.source)}
    if z_cells == 0 and vr_cells == 0:
        # A volume that decoded and verified but put nothing on this grid:
        # a real outcome (clear air, or an antenna whose disc misses after
        # the range and elevation cuts), and not a failure.  It is still
        # not a contributor, and the receipt says which.
        out.status = OUTCOME_EMPTY
        return None, out
    return contribution, out


@dataclass
class AllRadarIngest:
    """The result of one domain-wide ingest: what merged, and what didn't."""

    observations: object | None
    outcomes: list[SiteOutcome]
    seconds: dict = field(default_factory=dict)

    @property
    def contributing(self) -> list[SiteOutcome]:
        return [o for o in self.outcomes if o.contributed]

    def to_payload(self) -> dict:
        contributed = self.contributing
        by_status: dict[str, int] = {}
        for o in self.outcomes:
            by_status[o.status] = by_status.get(o.status, 0) + 1
        return {
            "schema": INGEST_SCHEMA,
            # The number the whole receipt exists for.  An analysis that
            # ran on three radars because seventeen mirrors flaked is a
            # different analysis, and this is where that shows.
            "sites_contributing": len(contributed),
            "sites_discovered": len(self.outcomes),
            "sites_by_status": by_status,
            "contributing_sites": [o.site for o in contributed],
            "bytes_fetched": sum(o.volume_bytes for o in self.outcomes),
            "seconds": dict(self.seconds),
            "sites": [o.to_payload() for o in self.outcomes],
        }


def ingest_domain(binary, coverages: Sequence[SiteCoverage], *, grid,
                  work_dir: Path, valid_time: datetime, params,
                  workers: int = 16, compute_workers: int | None = None,
                  z_reduce: str = "max",
                  progress: Callable[[SiteOutcome], None] | None = None,
                  **site_kw) -> AllRadarIngest:
    """Ingest every site concurrently and merge what came back.

    Two pools, because the stages are two different problems.  Fetch is
    latency-bound -- a single stream to the mirror moves ~2 MB/s from
    Seoul no matter how idle the box is, so throughput comes from having
    many requests outstanding, and ``workers`` should be far larger than
    the core count.  Superob is CPU-bound numpy, and running it at fetch
    concurrency just oversubscribes the cores and inflates every site's
    cost.  ``compute_workers`` (default: the core count) bounds that stage
    alone.

    Sites are not staged behind a barrier: each runs straight through and
    hops pools at the superob step, so a fast site's arithmetic overlaps a
    slow site's fetch rather than waiting for the whole roster to land.
    """
    import os

    from gpuwm.obs.superob import merge_contributions

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[object | None, SiteOutcome]] = {}
    t0 = time.perf_counter()
    n = max(1, min(int(workers), max(1, len(coverages))))
    if compute_workers is None:
        compute_workers = os.cpu_count() or 4
    n_compute = max(1, min(int(compute_workers), max(1, len(coverages))))
    with futures.ThreadPoolExecutor(max_workers=n_compute) as cpool, \
            futures.ThreadPoolExecutor(max_workers=n) as pool:
        pending = {
            pool.submit(ingest_site, binary, cov, grid=grid,
                        work_dir=work_dir, valid_time=valid_time,
                        params=params, compute_pool=cpool, **site_kw): cov
            for cov in coverages
        }
        for fut in futures.as_completed(pending):
            cov = pending[fut]
            try:
                contribution, outcome = fut.result()
            except Exception as exc:  # noqa: BLE001 - belt and braces
                # ingest_site is contracted not to raise; if it ever does,
                # the cycle still must not die for it.
                contribution = None
                outcome = SiteOutcome(
                    site=cov.site, status=OUTCOME_FETCH_FAILED,
                    reason=f"unhandled {type(exc).__name__}: {exc}",
                    coverage_fraction=cov.coverage_fraction)
            results[cov.site] = (contribution, outcome)
            if progress is not None:
                progress(outcome)
    ingest_seconds = time.perf_counter() - t0

    # Deterministic order regardless of completion order: the merged file's
    # radar axis must not depend on which mirror answered first.
    ordered = [results[c.site] for c in coverages if c.site in results]
    contributions = [c for c, _ in ordered if c is not None]
    outcomes = [o for _, o in ordered]

    t1 = time.perf_counter()
    observations = (merge_contributions(contributions, grid, params=params,
                                        z_reduce=z_reduce)
                    if contributions else None)
    merge_seconds = time.perf_counter() - t1

    return AllRadarIngest(
        observations=observations,
        outcomes=outcomes,
        seconds={
            "ingest": ingest_seconds,
            "merge": merge_seconds,
            "fetch_total": sum(o.seconds.get("fetch", 0.0) for o in outcomes),
            "decode_total": sum(
                o.seconds.get("decode", 0.0) for o in outcomes),
            "verify_total": sum(
                o.seconds.get("verify", 0.0) for o in outcomes),
            "superob_total": sum(
                o.seconds.get("superob", 0.0) for o in outcomes),
            "superob_queued_total": sum(
                o.seconds.get("superob_queued", 0.0) for o in outcomes),
        },
    )
