"""Freeze the European radar site inventory the DA lanes plan against.

The collection publishes, per radar, its WIGOS identifier, its position, and
the list of moments and scan methods it serves. That is enough to answer the
two questions a campaign asks before it starts:

* which radars cover this domain, and
* do they serve radial velocity, or only reflectivity.

It is *not* enough to assimilate a polar volume from any of them. Two things
are missing and both are recorded rather than guessed:

**Antenna elevation.** The locations endpoint publishes a 2D point -- no
height above mean sea level. Beam height above ground is a function of
antenna height, so a site assimilated with a wrong antenna elevation puts
every gate at the wrong altitude, and the error is a smooth offset rather
than anything that looks like a failure. Probed 2026-08-12, the ``detail``
link each feature carries points at the WMO OSCAR/Surface station search,
which answered ``totalCount: 0`` for these identifiers. So the height is not
in this feed and not one hop from it, and this tool alone freezes every row
with ``elevation_m: null``.

**It is in the volumes.** ODIM carries the antenna height at
``/where/height`` in every polar volume, so the gap closes by reading the
product rather than the metadata: ``tools/harvest_radar_heights.py`` harvests
it per site and ``--heights`` merges the result here, position-cross-checked
per site so a height cannot be read out of another radar's file. A site the
harvest could not reach keeps ``null``, and
:mod:`gpuwm.obs.radar_sites` refuses to hand it out for polar assimilation. A
placeholder would be worse than the absence: it would make the site *look*
usable.

**Elevation angles.** The scan strategy is per-volume metadata, not per-site,
so the list of cuts a radar runs is only knowable from a decoded volume.

Run it when the inventory needs refreshing::

    python tools/harvest_radar_heights.py         --table gpuwm/obs/data/radar_sites_odim.json --out heights.json
    python tools/freeze_radar_sites.py --heights heights.json         --out gpuwm/obs/data/radar_sites_odim.json

``--from-table`` merges heights into an existing frozen table without
refetching the locations feed, for the days the feed answers 429.

Nothing here names a country in code. WMO block numbers and station ids are
data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

#: The collection's location inventory.
LOCATIONS_URL = (
    "https://api.meteogate.eu/eu-eumetnet-weather-radar/collections/"
    "observations/locations?f=GeoJSON")

#: The schema the frozen inventory declares.
TABLE_SCHEMA = "gpuwm-obs.radar-sites.v1"

#: The schema of the antenna-height document ``--heights`` merges in.
#: Restated rather than imported: this tool must run from a checkout with
#: nothing on the path but the standard library.
HEIGHTS_SCHEMA = "gpuwm-obs.radar-heights.v1"

#: The composite pseudo-site. It is a product, not an antenna, and it is
#: filed in the same inventory as the radars; keeping it in the site table
#: would put a station with no position and no scan strategy in front of a
#: caller asking which radars cover a domain.
COMPOSITE_SUFFIX = "-OPERA"

#: Moment names that carry radial velocity. A site with none of them cannot
#: contribute wind to an analysis whatever else it serves.
VELOCITY_MOMENTS = ("VRAD", "VRADH", "VRADV")

#: The scan method that yields a polar volume. The collection also serves
#: ``comp`` (composited), ``point`` and ``sum``, which are products rather
#: than volumes.
VOLUME_METHOD = "scan"

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def fetch(url: str, *, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "gpuwm"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def wmo_block(identifier: str) -> str | None:
    """The WMO block out of a WIGOS id like ``0-191-0-hrdeb``.

    The block is what groups sites by issuing service, and it is the closest
    thing the feed carries to a country. It is kept as the feed's own number
    rather than mapped to a name: a mapping would be this file inventing a
    political fact the archive never stated.
    """

    parts = identifier.split("-")
    return parts[1] if len(parts) >= 2 and parts[1].isdigit() else None


def parse_parameters(values) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``("DBZH:scan", ...)`` split into moment names and scan methods."""

    moments: list[str] = []
    methods: list[str] = []
    for entry in values or ():
        text = str(entry)
        moment, _, method = text.partition(":")
        if moment:
            moments.append(moment)
        if method:
            methods.append(method)
    return tuple(sorted(set(moments))), tuple(sorted(set(methods)))


def merge_heights(table: dict, heights: dict) -> dict:
    """Fill ``elevation_m`` from a ``gpuwm-obs.radar-heights.v1`` document.

    The heights come from the volumes rather than from this feed -- see
    ``tools/harvest_radar_heights.py`` -- so the merge has to be checked
    rather than trusted. Three refusals, all of them because the failure they
    guard against is silent:

    * a document of the wrong schema, which would otherwise be read
      key-by-key and quietly contribute nothing;
    * a position cross-check that failed, meaning at least one height was
      read out of a file belonging to a different radar. Every gate of that
      site would then be placed at another antenna's altitude, and there is
      nothing in the output that would look wrong;
    * a document whose sites are disjoint from this table's, which is what a
      stale harvest against an older site list looks like.

    A site the harvest could not reach keeps ``None``. That is the whole
    point of the null: ``gpuwm.obs.radar_sites.require_assimilable`` refuses
    it by name, and a placeholder would make an unusable site look usable.
    """

    schema = heights.get("schema")
    if schema != HEIGHTS_SCHEMA:
        raise ValueError(
            f"heights document declares schema {schema!r}, expected "
            f"{HEIGHTS_SCHEMA!r}")
    failures = heights.get("position_check_failures") or []
    if failures:
        raise ValueError(
            f"the heights harvest's position cross-check failed for "
            f"{len(failures)} site(s) ({', '.join(failures[:5])}): at least "
            "one antenna height was read out of a file belonging to a "
            "different radar. Merging would place every gate of that site at "
            "another antenna's altitude, smoothly, with nothing in the "
            "output that looks wrong. Re-harvest before merging")

    found = heights.get("sites") or {}
    matched = 0
    for row in table["sites"]:
        record = found.get(row["id"])
        if record is None:
            continue
        row["elevation_m"] = float(record["height_m"])
        matched += 1
    if found and not matched:
        raise ValueError(
            f"none of the heights document's {len(found)} sites is in this "
            "table; it was harvested against a different site list")

    table["sites_with_elevation"] = sum(
        1 for row in table["sites"] if row["elevation_m"] is not None)
    table["elevation_basis"] = (
        "the antenna elevation is absent from the locations feed and from "
        "the WMO registry its detail "
        "link names (probed 2026-08-12: totalCount 0), so it is read from the "
        "volumes instead: ODIM carries the antenna height above mean sea "
        "level at /where/height in every polar volume. Harvested by "
        "tools/harvest_radar_heights.py, one whole file per site, each one "
        "cross-checked against this table's own position for that site so a "
        "height cannot be read out of another radar's file. A site the "
        "harvest could not reach keeps null and stays refused, because a "
        "placeholder does not make a site usable, it makes an unusable site "
        "look usable")
    table["elevation_source"] = {
        "tool": "tools/harvest_radar_heights.py",
        "schema": schema,
        "harvested_at": heights.get("harvested_at"),
        "store_base": heights.get("store_base"),
        "sites_attempted": heights.get("sites_attempted"),
        "sites_resolved": heights.get("sites_resolved"),
        "position_check_max_deg": heights.get("position_check_max_deg"),
        "position_tolerance_deg": heights.get("position_tolerance_deg"),
    }
    return table


def build_table(raw: bytes) -> dict:
    document = json.loads(raw)
    rows = []
    composites = []
    for feature in document.get("features", ()):
        identifier = str(feature.get("id", ""))
        if not identifier:
            continue
        properties = feature.get("properties") or {}
        coordinates = (feature.get("geometry") or {}).get("coordinates")
        if identifier.endswith(COMPOSITE_SUFFIX):
            composites.append(identifier)
            continue
        if not (isinstance(coordinates, (list, tuple))
                and len(coordinates) >= 2):
            # A radar with no position cannot be placed on a domain. It is
            # dropped loudly, into a list, rather than silently.
            composites.append(f"{identifier}: no position")
            continue
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
        moments, methods = parse_parameters(properties.get("parameter-name"))
        rows.append({
            "id": identifier,
            "name": str(properties.get("name", "")),
            "wmo_block": wmo_block(identifier),
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            # Not published by this feed, and not one hop from it. See the
            # module docstring; a number here would make an unusable site
            # look usable.
            "elevation_m": None,
            "moments": list(moments),
            "methods": list(methods),
            "has_velocity": any(m in VELOCITY_MOMENTS for m in moments),
            "has_volume_scan": VOLUME_METHOD in methods,
            "detail": str(properties.get("detail", "")),
        })
    rows.sort(key=lambda row: row["id"])
    with_velocity = sum(1 for row in rows if row["has_velocity"])
    with_volume = sum(1 for row in rows if row["has_volume_scan"])
    return {
        "schema": TABLE_SCHEMA,
        "source_url": LOCATIONS_URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "frozen_at": datetime.now(timezone.utc).strftime(_TIME_FORMAT),
        "elevation_basis": (
            "absent from this feed and from the WMO registry its detail link "
            "names (probed 2026-08-12: totalCount 0). Every row is null and "
            "gpuwm.obs.radar_sites refuses a site for polar assimilation "
            "until an elevation is supplied, because a placeholder would "
            "make an unusable site look usable"),
        "site_count": len(rows),
        "sites_with_velocity": with_velocity,
        "sites_with_volume_scan": with_volume,
        "sites_with_elevation": 0,
        "excluded": sorted(composites),
        "sites": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default=LOCATIONS_URL)
    parser.add_argument("--heights", type=Path, default=None,
                        help="a gpuwm-obs.radar-heights.v1 document from "
                             "tools/harvest_radar_heights.py, whose antenna "
                             "heights fill elevation_m. Omitting it re-freezes "
                             "with every row null, which is the honest state "
                             "of the locations feed alone and refuses every "
                             "site for polar assimilation")
    parser.add_argument("--from-table", type=Path, default=None,
                        help="do not refetch the locations feed: take the "
                             "site list, positions and their provenance from "
                             "this already-frozen table and merge --heights "
                             "into it. For adding antenna heights to a table "
                             "whose site list is current, and for the days "
                             "the feed answers 429 to a full refreeze. The "
                             "written table keeps the original source_sha256 "
                             "and frozen_at, because that is what its site "
                             "list actually came from, and records this pass "
                             "separately")
    arguments = parser.parse_args(argv)

    if arguments.from_table is not None:
        if arguments.heights is None:
            raise SystemExit(
                "--from-table only merges heights into an existing table; "
                "without --heights it would rewrite it unchanged")
        table = json.loads(arguments.from_table.read_text(encoding="utf-8"))
        if table.get("schema") != TABLE_SCHEMA:
            raise SystemExit(
                f"{arguments.from_table} declares schema "
                f"{table.get('schema')!r}, expected {TABLE_SCHEMA!r}")
    else:
        table = build_table(fetch(arguments.url))
    if arguments.heights is not None:
        table = merge_heights(
            table,
            json.loads(arguments.heights.read_text(encoding="utf-8")))
    if not table["sites"]:
        raise SystemExit(
            f"{arguments.url} yielded no radar site; refusing to write an "
            f"empty inventory, which would report every domain as having no "
            f"radar coverage")
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(table, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"{arguments.out}: {table['site_count']} radars, "
          f"{table['sites_with_velocity']} with velocity, "
          f"{table['sites_with_volume_scan']} with a volume scan, "
          f"{table['sites_with_elevation']} with an antenna elevation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
