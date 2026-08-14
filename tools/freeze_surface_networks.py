"""Freeze the surface-observation network table gpuwm resolves domains against.

The archive publishes every network it holds, each with the bounding polygon
of the stations actually in it. That listing is the authority; this tool
copies it into the repository so a domain can be resolved to a network set
without a network round trip, and so the set a case used is a reviewable
artifact rather than whatever the archive answered that afternoon.

Only the ASOS/METAR families are kept. They are the ones
``gpuwm-obs.asos-surface.v1`` is defined over: hourly synoptic-style surface
reports with temperature, dewpoint, wind and pressure. The archive's other
families (COOP, DCP, RWIS, COCORAHS) report different variables on different
cadences and would need their own decode contract before they could be
resolved to.

Run it when the table needs refreshing::

    python tools/freeze_surface_networks.py --out gpuwm/obs/data/surface_networks.json

The output carries the source URL, the instant it was taken and the digest of
the bytes it was taken from, so a later reader can tell whether the table
still matches the archive without diffing 266 rows by eye.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

#: The archive's own network listing.
NETWORKS_URL = "https://mesonet.agron.iastate.edu/geojson/networks.geojson"

#: The schema the frozen table declares.
TABLE_SCHEMA = "gpuwm-obs.surface-networks.v1"

#: Which network families this table admits. The suffix is the archive's own
#: naming, and it is checked rather than assumed: a network whose id does not
#: end this way reports a different set of variables.
ASOS_SUFFIX = "_ASOS"

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def fetch(url: str, *, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


#: How long to wait between station-list requests. The archive is free and
#: asks to be paced; seven case boxes pulled back to back earned an HTTP 429
#: on the twenty-first request when the surface fetch route was measured.
REQUEST_PAUSE_S = 0.5

#: Padding added to every edge of a network's station extent, in degrees.
#:
#: The extent is a *candidate* screen: a network offered needlessly costs one
#: metadata request, a network dropped costs observations silently, so the
#: box errs outward. A tenth of a degree is roughly 11 km, which covers a
#: station commissioned near the edge since the table was frozen without
#: making the box meaningfully coarser. It is the same padding the archive's
#: own network listing applies.
EDGE_PAD_DEG = 0.1


def longitude_span(longitudes: list[float]) -> tuple[float, float]:
    """The shortest longitude interval containing every station.

    Naive ``min``/``max`` is wrong for any network whose stations straddle
    the antimeridian, and the error is not small: New Zealand reports
    stations near +178 and near -176, so ``min``/``max`` yields ``[-176,
    178]`` -- a box spanning 354 of the 360 degrees, which then matches
    domains in the South Atlantic and the Indian Ocean.

    The shortest containing interval is the complement of the *largest gap*
    between adjacent stations around the circle. The returned interval may
    run past +180 (``west=166, east=189``); a consumer that understands the
    wrap reads that as two pieces, and one that does not would at worst be
    over-inclusive, which is the safe direction.
    """

    values = sorted(set(longitudes))
    if not values:
        raise ValueError("a longitude span needs at least one longitude")
    if len(values) == 1:
        return values[0], values[0]
    widest_gap = -1.0
    gap_start_index = 0
    for index in range(len(values)):
        following = values[(index + 1) % len(values)]
        gap = following - values[index]
        if index == len(values) - 1:
            # The gap that wraps through the antimeridian.
            gap = following + 360.0 - values[index]
        if gap > widest_gap:
            widest_gap = gap
            gap_start_index = index
    west = values[(gap_start_index + 1) % len(values)]
    east = values[gap_start_index]
    if east < west:
        east += 360.0
    return west, east


def station_extent(features) -> tuple[float, float, float, float] | None:
    """``(west, south, east, north)`` over a network's actual stations.

    Taken from the station positions themselves rather than from the
    listing's declared polygon. The declared polygon is a padded ``min``/
    ``max`` box and carries the antimeridian defect above; the stations are
    what the fetch route will actually be filtered against.
    """

    longitudes: list[float] = []
    latitudes: list[float] = []
    for feature in features or ():
        coordinates = (feature.get("geometry") or {}).get("coordinates")
        if not (isinstance(coordinates, (list, tuple)) and len(coordinates) == 2):
            continue
        longitude, latitude = coordinates
        if not (isinstance(longitude, (int, float))
                and isinstance(latitude, (int, float))):
            continue
        longitudes.append(float(longitude))
        latitudes.append(float(latitude))
    if not longitudes:
        return None
    west, east = longitude_span(longitudes)
    south = min(latitudes) - EDGE_PAD_DEG
    north = max(latitudes) + EDGE_PAD_DEG
    # No station lies past a pole, so clamping loses no coverage; leaving the
    # padding un-clamped would put a latitude off the globe in the table and
    # make every consumer's range check a lie.
    south = max(south, -90.0)
    north = min(north, 90.0)
    west -= EDGE_PAD_DEG
    east += EDGE_PAD_DEG
    if east - west >= 360.0:
        west, east = -180.0, 180.0
    return (west, south, east, north)


def station_list_url(archive: str, network: str) -> str:
    """The same template ``rw_asos stations`` reads network metadata from."""

    return f"{archive}/geojson/network/{network}.geojson"


def build_table(raw: bytes, *, archive: str, verbose: bool = True) -> dict:
    """The frozen table, one station-list request per network."""

    document = json.loads(raw)
    names = {
        str(feature.get("id", "")): str(
            (feature.get("properties") or {}).get("name", ""))
        for feature in document.get("features", ())
    }
    candidates = sorted(
        network for network in names if network.endswith(ASOS_SUFFIX))

    rows = []
    empty: list[str] = []
    unreachable: list[str] = []
    total_stations = 0
    for index, network in enumerate(candidates, start=1):
        try:
            listing = json.loads(fetch(station_list_url(archive, network)))
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            # A network whose stations could not be listed is recorded as
            # unreachable, never silently given the declared polygon: an
            # extent from a different source would be indistinguishable in
            # the table from one this tool actually measured.
            unreachable.append(f"{network}: {error}")
            continue
        features = listing.get("features") or ()
        extent = station_extent(features)
        if extent is None:
            # "The archive lists this network but placed no stations in it"
            # and "this tool lost it" are different facts, so the first is
            # written down.
            empty.append(network)
            continue
        west, south, east, north = extent
        total_stations += len(features)
        rows.append({
            "id": network,
            "name": names.get(network, ""),
            "stations": len(features),
            "west": round(west, 5),
            "south": round(south, 5),
            "east": round(east, 5),
            "north": round(north, 5),
            "crosses_antimeridian": east > 180.0,
        })
        if verbose and index % 25 == 0:
            print(f"  {index}/{len(candidates)} networks", flush=True)
        time.sleep(REQUEST_PAUSE_S)

    rows.sort(key=lambda row: row["id"])
    return {
        "schema": TABLE_SCHEMA,
        "source_url": NETWORKS_URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "station_list_template": station_list_url(archive, "{network}"),
        "frozen_at": datetime.now(timezone.utc).strftime(_TIME_FORMAT),
        "network_suffix": ASOS_SUFFIX,
        "edge_pad_deg": EDGE_PAD_DEG,
        "extent_basis": (
            "measured over each network's own station positions, with the "
            "shortest containing longitude interval so an antimeridian "
            "network does not span the globe, padded by edge_pad_deg on "
            "every side because the extent is a candidate screen that must "
            "never under-include"),
        "network_count": len(rows),
        "station_count": total_stations,
        "networks_without_stations": sorted(empty),
        "networks_unreachable": sorted(unreachable),
        "networks": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the frozen table")
    parser.add_argument("--url", default=NETWORKS_URL,
                        help="the archive listing to freeze from")
    parser.add_argument("--archive",
                        default="https://mesonet.agron.iastate.edu",
                        help="archive root the station lists are read from")
    arguments = parser.parse_args(argv)

    raw = fetch(arguments.url)
    table = build_table(raw, archive=arguments.archive)
    if not table["networks"]:
        raise SystemExit(
            f"{arguments.url} yielded no {ASOS_SUFFIX} network with an extent; "
            "refusing to write an empty table, which would silently resolve "
            "every domain to no observations")
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(table, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"{arguments.out}: {table['network_count']} networks, "
          f"{table['station_count']} stations, "
          f"source sha256 {table['source_sha256'][:12]}")
    if table["networks_unreachable"]:
        print(f"  unreachable: {len(table['networks_unreachable'])} "
              f"(re-run to pick them up)")
    if table["networks_without_stations"]:
        print(f"  listed but empty: "
              f"{len(table['networks_without_stations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
