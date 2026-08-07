"""Which radars can see this domain, computed rather than named.

One radar measures the wind component along its own beam and nothing else.
Two radars whose coverage overlaps measure two projections of the same wind,
and the pair constrains what neither can: that overlap is the single largest
information increment available from Level-II, and it is only reachable if
the pipeline can be told "cover this domain" instead of "use this site".

So this module answers one question -- *which sites cover this grid* -- from
the front door's own vendored NEXRAD site table (``rw_nexrad sites``,
schema ``gpuwm-obs.nexrad-sites.v1``) and the caller's own
:class:`~gpuwm.obs.target_grid.TargetGrid`.  **No site id appears anywhere
in this file.**  Ids are results here, never inputs and never defaults.

**Coverage is a measured fraction, not a yes/no.**  A site whose disc clips
one corner of the domain and a site sitting over its centre are both "in
range", and treating them the same is how an analysis ends up dominated by
a radar that saw almost none of it.  :func:`sites_covering` returns, for
every candidate, the fraction of the domain's own mass points that fall
inside the site's usable range -- the same ``max_range_km`` the superobber
will apply -- so a caller can set a floor and the receipt can say what each
radar was actually asked to contribute.

**The table's altitudes are not used here and must not be.**  130 of the
table's 141 entries carry an unset elevation placeholder, which the decoder
refuses for exactly the right reason: the antenna height is the ray origin.
Discovery needs horizontal position only, and every volume that reaches the
superobber places its own antenna from its Message-31 VOL block.  Nothing
in this module feeds a gate height.

**Range is a planning radius, not a claim about what the beam did.**  The
disc this computes ignores terrain blockage, beam broadening and the cone
of silence.  It is the right tool for deciding which volumes to fetch and
the wrong one for deciding what an observation is worth; the superobber and
the per-radar counts in the receipt are where the second question gets
answered, on the data that actually arrived.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.obs.nexrad import SITES_SCHEMA

#: Mean Earth radius for the great-circle range test, metres.  Discovery
#: decides which volumes to download; a sphere is right to well inside the
#: margin any sane ``min_coverage_fraction`` leaves.
EARTH_RADIUS_M = 6371008.8


class SiteCoverageError(RuntimeError):
    """Coverage could not be established, and the reason is the message."""


@dataclass(frozen=True)
class SiteFix:
    """One row of the vendored table, horizontal position only."""

    id: str
    name: str
    lat_deg: float
    lon_deg: float

    def to_payload(self) -> dict:
        return {"id": self.id, "name": self.name,
                "lat_deg": float(self.lat_deg),
                "lon_deg": float(self.lon_deg)}


@dataclass(frozen=True)
class SiteCoverage:
    """A candidate radar and how much of the domain it can actually reach."""

    site: SiteFix
    #: Fraction of the domain's mass points within ``max_range_km`` of the
    #: antenna, in [0, 1].  This is the number a floor is set against.
    coverage_fraction: float
    #: Domain mass points inside the disc, and the domain's total.
    points_covered: int
    points_total: int
    #: Great-circle range from the antenna to the nearest and farthest
    #: domain mass point, km.  The pair separates "sits inside the domain"
    #: from "clips one corner" without needing the fraction.
    nearest_km: float
    farthest_km: float

    def to_payload(self) -> dict:
        return {
            **self.site.to_payload(),
            "coverage_fraction": round(float(self.coverage_fraction), 6),
            "points_covered": int(self.points_covered),
            "points_total": int(self.points_total),
            "nearest_km": round(float(self.nearest_km), 3),
            "farthest_km": round(float(self.farthest_km), 3),
        }


def read_site_table(binary, *, site: str | None = None) -> dict[str, SiteFix]:
    """The front door's own vendored table, keyed by site id.

    Shelling out to the same binary that decodes the volumes is deliberate:
    a second copy of the table in Python is a second thing to be wrong, and
    the KPBZ longitude transcription error is a live reminder that the
    table is data with a history rather than a constant.  A site whose
    coordinates this returns is the same site the decoder will place.
    """

    from gpuwm.obs.nexrad import run_sites                # noqa: PLC0415

    try:
        record = run_sites(binary, site=site)
    except (OSError, RuntimeError, ValueError) as error:
        raise SiteCoverageError(
            f"the NEXRAD front door at {binary} could not produce its site "
            f"table: {error}") from error
    if record.get("schema") != SITES_SCHEMA:
        raise SiteCoverageError(
            f"the site table declares schema {record.get('schema')!r}, this "
            f"reader understands {SITES_SCHEMA!r}; the binary and this "
            "checkout disagree about the table format")
    table: dict[str, SiteFix] = {}
    for row in record.get("sites", ()):
        try:
            fix = SiteFix(id=str(row["id"]), name=str(row["name"]),
                          lat_deg=float(row["lat_deg"]),
                          lon_deg=float(row["lon_deg"]))
        except (KeyError, TypeError, ValueError) as error:
            raise SiteCoverageError(
                f"site table row {row!r} is not usable: {error}") from error
        if not (-90.0 <= fix.lat_deg <= 90.0):
            raise SiteCoverageError(
                f"site {fix.id} has latitude {fix.lat_deg}, which is not a "
                "latitude; the table this pipeline plans against is wrong")
        table[fix.id] = fix
    if not table:
        raise SiteCoverageError(
            "the vendored NEXRAD site table is empty; nothing can be "
            "discovered from it and a hardcoded site is not the answer")
    return table


def great_circle_km(lat0_deg: float, lon0_deg: float,
                    lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Haversine range from one point to an array of them, km."""

    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    dlat = lat - lat0
    dlon = lon - lon0
    h = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * (EARTH_RADIUS_M / 1000.0) * np.arcsin(np.sqrt(np.clip(
        h, 0.0, 1.0)))


def sites_covering(grid, table: dict[str, SiteFix], *, max_range_km: float,
                   min_coverage_fraction: float = 0.05,
                   limit: int | None = None) -> list[SiteCoverage]:
    """Every table site whose usable range reaches this domain, ranked.

    ``max_range_km`` must be the SAME range the superobber will apply --
    passing the decoder's range and screening on a longer one selects a
    radar whose contribution is then thrown away gate by gate, and the
    receipt would show a radar that contributed nothing without saying why.

    Ranked by coverage fraction descending, then by id, so the order is
    total and reproducible: two runs of the same domain against the same
    table select the same radars in the same sequence, which is what makes
    a multi-radar analysis comparable to the one before it.

    Returns an empty list rather than raising -- "no radar covers this
    domain" is a fact about the domain, and the caller decides whether it
    is fatal.
    """

    range_km = float(max_range_km)
    if not np.isfinite(range_km) or range_km <= 0.0:
        raise SiteCoverageError(
            f"max_range_km is {max_range_km!r}; discovery needs the finite, "
            "positive range the superobber will actually apply")
    floor = float(min_coverage_fraction)
    if not np.isfinite(floor) or not (0.0 <= floor <= 1.0):
        raise SiteCoverageError(
            f"min_coverage_fraction is {min_coverage_fraction!r}; it is a "
            "fraction of the domain and must lie in [0, 1]")
    if limit is not None and int(limit) < 1:
        raise SiteCoverageError(
            f"limit is {limit!r}; asking for fewer than one radar is not a "
            "smaller request, it is an empty one")

    lat = np.asarray(grid.lat, dtype=np.float64)
    lon = np.asarray(grid.lon, dtype=np.float64)
    total = int(lat.size)

    found: list[SiteCoverage] = []
    for site_id in sorted(table):
        fix = table[site_id]
        distance = great_circle_km(fix.lat_deg, fix.lon_deg, lat, lon)
        inside = distance <= range_km
        covered = int(inside.sum())
        if covered == 0:
            continue
        fraction = covered / total
        if fraction < floor:
            continue
        found.append(SiteCoverage(
            site=fix, coverage_fraction=fraction, points_covered=covered,
            points_total=total, nearest_km=float(distance.min()),
            farthest_km=float(distance.max())))

    found.sort(key=lambda entry: (-entry.coverage_fraction, entry.site.id))
    if limit is not None:
        found = found[:int(limit)]
    return found


def discovery_receipt(found, *, max_range_km: float,
                      min_coverage_fraction: float,
                      limit: int | None, table_size: int) -> dict:
    """What discovery decided, in the form the obs receipt carries.

    The parameters are recorded beside the result because the result is
    meaningless without them: "three radars" is a different statement at a
    5% floor than at a 40% one, and a reader six weeks later has only this.
    """

    return {
        "schema": "gpuwm-obs.site-discovery.v1",
        "method": ("great-circle range from the vendored NEXRAD site table "
                   "to every domain mass point; no site id is an input"),
        "table_schema": SITES_SCHEMA,
        "table_sites": int(table_size),
        "max_range_km": float(max_range_km),
        "min_coverage_fraction": float(min_coverage_fraction),
        "limit": None if limit is None else int(limit),
        "selected": [entry.to_payload() for entry in found],
        "caveat": ("a disc, not a beam: terrain blockage, beam broadening "
                   "and the cone of silence are not modelled here, so "
                   "coverage_fraction bounds what a radar could contribute "
                   "and the per-radar cell counts record what it did"),
    }
