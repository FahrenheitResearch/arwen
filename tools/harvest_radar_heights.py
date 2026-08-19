"""Harvest every European radar's antenna height from the volumes themselves.

The second half of ``tools/freeze_radar_sites.py``. That tool freezes the
MeteoGate locations endpoint, which publishes a **2-D point**: latitude and
longitude and no height. The WMO OSCAR/Surface registry its ``detail`` link
names answers ``totalCount: 0``, so the height is neither in the feed nor one
hop from it, and every row of the frozen table said ``elevation_m: null``.

**It is in the data.** ODIM puts the antenna height above mean sea level at
``/where/height`` in every polar volume, and the archive publishes a volume
for every listed radar. So the gap was a consequence of freezing the table
from the metadata endpoint rather than from the product, and it closes by
reading the product.

Why it matters at all: beam height above ground is a function of antenna
height above mean sea level, so a site assimilated with the wrong antenna
height places every gate at the wrong altitude -- smoothly, with nothing that
looks like a failure. That is why the table refuses a null rather than
substituting a zero, and it is why this tool cross-checks (below) rather than
trusting a filename to have told it which radar it is reading.

Discovery
---------
``ListObjectsV2`` on the object store, not the OGC-EDR gateway. The gateway
serves the same objects and its CoverageJSON is the documented route, but a
136-site sweep trips its rate limiter into a sustained ``429`` that does not
clear; measured 2026-08-14, 80 of 136 sites unreachable that way over an hour.
The store's listing has no such limit and the bytes downloaded are identical
objects from the identical allowlisted prefix.

Bytes
-----
One file per site, the smallest the site publishes, deleted after reading.
``/where/height`` is a root attribute identical across every file of a volume,
so a 46 KB single-sweep scan answers the same question a 31 MB whole-volume
``PVOL`` does. Whole files, never a byte-range: this is a small read by
choosing a small file, not by subsetting a large one.

The cross-check, which is the point
-----------------------------------
Every file also carries ``/where/lat`` and ``/where/lon``, and this tool
compares them against the position the frozen table already holds for that
site. Matching the wrong file to the wrong site is the failure mode that
would silently poison the whole column -- every gate of a radar placed at
another radar's altitude -- and it is invisible in the output, because a
height is just a plausible number. The position agreement is what makes the
height trustworthy, so it is recorded per site and summarised, and
``freeze_radar_sites.py`` refuses to merge a document whose worst
disagreement exceeds :data:`POSITION_TOLERANCE_DEG`.

EXPERIMENTAL, like the rest of this lane. No site names belong in this file's
defaults or identifiers: the site list comes from the table.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree

#: Schema of the document this tool writes.
HEIGHTS_SCHEMA = "gpuwm-obs.radar-heights.v1"

#: The object store the archive publishes through. Downloads are refused off
#: this prefix, so a redirected or rewritten listing cannot make this tool
#: fetch from somewhere else.
STORE_BASE = "https://s3.waw3-1.cloudferro.com/openradar-24h"

_S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
_UA = {"User-Agent": "gpuwm-odim-fetch"}

#: How far a file's declared position may sit from the table's before the
#: pairing is not believable. 0.01 degrees is about 1 km; two radars in one
#: national network are tens of kilometres apart, so this separates "the same
#: antenna, rounded differently" from "a different radar" by more than two
#: orders of magnitude.
POSITION_TOLERANCE_DEG = 0.01

#: Heights outside this band are reported by name rather than accepted
#: quietly. Europe genuinely has radars above 2900 m (the Swiss Alps) and at
#: 15 m (the Danish islands), so the band is wide and its purpose is to make a
#: unit error -- feet, or centimetres -- visible.
PLAUSIBLE_HEIGHT_M = (-50.0, 4000.0)


def _get(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _listing(prefix: str, *, delimiter: str = "/"):
    """One ``ListObjectsV2`` walk: (common prefixes, [(key, bytes)])."""

    prefixes: list[str] = []
    contents: list[tuple[str, int]] = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            query["delimiter"] = delimiter
        if token:
            query["continuation-token"] = token
        tree = ElementTree.fromstring(
            _get(f"{STORE_BASE}/?" + urllib.parse.urlencode(query)).decode())
        prefixes += [node.text for node
                     in tree.findall(".//s:CommonPrefixes/s:Prefix", _S3_NS)]
        contents += [(node.find("s:Key", _S3_NS).text,
                      int(node.find("s:Size", _S3_NS).text))
                     for node in tree.findall(".//s:Contents", _S3_NS)]
        following = tree.find("s:NextContinuationToken", _S3_NS)
        if following is None:
            return prefixes, contents
        token = following.text


def default_days(count: int = 2) -> list[str]:
    """The last ``count`` UTC days as ``YYYY/mm/dd`` prefixes, newest first.

    Two by default rather than one: a site listed shortly after midnight UTC
    may have published nothing yet today, and falling back to yesterday costs
    one listing and rescues it. The archive retains 24 hours, so going further
    back buys nothing.
    """

    today = datetime.now(timezone.utc)
    return [(today - timedelta(days=offset)).strftime("%Y/%m/%d")
            for offset in range(count)]


def site_prefixes(days: list[str]) -> dict[str, list[str]]:
    """``{nod: [store prefix, ...]}`` over the days given, newest day first."""

    found: dict[str, list[str]] = {}
    for day in days:
        for country in _listing(day.rstrip("/") + "/")[0]:
            for site in _listing(country)[0]:
                found.setdefault(site.rstrip("/").rsplit("/", 1)[-1],
                                 []).append(site)
    return found


def _smallest_object(prefixes: list[str]) -> tuple[str, int] | None:
    """The smallest ``.h5`` under the first prefix that has any."""

    for prefix in prefixes:
        best = None
        for key, size in _listing(prefix, delimiter="")[1]:
            if key.endswith(".h5") and (best is None or size < best[1]):
                best = (key, size)
        if best is not None:
            return best
    return None


def _read_where(blob: bytes) -> dict:
    """``/where`` attributes of an ODIM file held in memory.

    h5py needs a filename, so the bytes are spilled to a temporary file and
    removed again; nothing downloaded here is a result and none of it is kept.
    """

    try:
        import h5py
    except ImportError:
        raise SystemExit(
            "harvest_radar_heights: pip install 'gpuwm[publish]'  (h5py "
            "reads /where/height out of a polar volume; it is a "
            "maintainer-only dependency because only rebuilding the frozen "
            "site table needs HDF5 from Python -- the product decodes ODIM "
            "through rw_odim, which carries its own reader)")

    handle, path = tempfile.mkstemp(suffix=".h5")
    os.close(handle)
    try:
        Path(path).write_bytes(blob)
        with h5py.File(path, "r") as opened:
            where = opened["/where"].attrs
            missing = [name for name in ("height", "lat", "lon")
                       if name not in where]
            if missing:
                raise ValueError(
                    f"/where carries no {', '.join(missing)}; ODIM stores "
                    f"these as attributes on the group and this file has "
                    f"{sorted(where.keys())!r}")
            return {"height_m": float(where["height"]),
                    "lat_file": float(where["lat"]),
                    "lon_file": float(where["lon"])}
    finally:
        os.unlink(path)


def harvest_site(nod: str, prefixes: list[str], row: dict):
    """One site: smallest object, ``/where``, and the position cross-check."""

    try:
        found = _smallest_object(prefixes)
        if found is None:
            return nod, None, "no .h5 object under this site's prefixes"
        key, size = found
        href = f"{STORE_BASE}/" + urllib.parse.quote(key)
        if not href.startswith(f"{STORE_BASE}/"):
            return nod, None, f"refusing off-allowlist href {href}"
        blob = _get(href, timeout=300)
        if len(blob) != size:
            return nod, None, (f"read {len(blob)} bytes of a {size}-byte "
                               "object; refusing a partial file")
        record = _read_where(blob)
        record.update({
            "file": href,
            "file_bytes": len(blob),
            "file_sha256": hashlib.sha256(blob).hexdigest(),
            "lat_table": row["latitude"],
            "lon_table": row["longitude"],
        })
        record["position_delta_deg"] = max(
            abs(record["lat_file"] - row["latitude"]),
            abs(record["lon_file"] - row["longitude"]))
        return nod, record, None
    except Exception as error:                              # noqa: BLE001
        return nod, None, f"{type(error).__name__}: {error}"


def harvest(table: dict, days: list[str], *, workers: int = 8) -> dict:
    rows = {row["id"].rsplit("-", 1)[-1]: row for row in table["sites"]}
    prefixes = site_prefixes(days)

    sites: dict[str, dict] = {}
    unresolved: dict[str, str] = {
        row["id"]: "this site publishes nothing in the archive"
        for nod, row in rows.items() if nod not in prefixes}

    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(harvest_site, nod, prefixes[nod], row)
                   for nod, row in rows.items() if nod in prefixes]
        for future in futures.as_completed(pending):
            nod, record, error = future.result()
            site_id = rows[nod]["id"]
            if record is None:
                unresolved[site_id] = error
            else:
                sites[site_id] = record

    deltas = [record["position_delta_deg"] for record in sites.values()]
    low, high = PLAUSIBLE_HEIGHT_M
    return {
        "schema": HEIGHTS_SCHEMA,
        "harvested_at": datetime.now(timezone.utc).isoformat(),
        "days_listed": days,
        "store_base": STORE_BASE,
        "basis": ("ODIM /where/height of one whole volume file per site; the "
                  "smallest object the site publishes, read and discarded"),
        "sites_attempted": len(rows),
        "sites_resolved": len(sites),
        "position_check_max_deg": max(deltas) if deltas else None,
        "position_tolerance_deg": POSITION_TOLERANCE_DEG,
        "position_check_failures": sorted(
            site for site, record in sites.items()
            if record["position_delta_deg"] > POSITION_TOLERANCE_DEG),
        "outside_plausible_band": sorted(
            site for site, record in sites.items()
            if not low <= record["height_m"] <= high),
        "sites": dict(sorted(sites.items())),
        "unresolved": dict(sorted(unresolved.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, required=True,
                        help="the frozen site table to harvest heights for; "
                             "it supplies the site list and the positions the "
                             "cross-check is made against")
    parser.add_argument("--out", type=Path, required=True,
                        help=f"{HEIGHTS_SCHEMA} document to write")
    parser.add_argument("--day", action="append", default=None,
                        metavar="YYYY/mm/dd",
                        help="archive day to list, repeatable, newest first "
                             "(default: today and yesterday, UTC)")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent site downloads (default 8)")
    arguments = parser.parse_args(argv)

    table = json.loads(arguments.table.read_text(encoding="utf-8"))
    document = harvest(table, arguments.day or default_days(),
                       workers=arguments.workers)

    if not document["sites"]:
        raise SystemExit(
            "no site yielded a /where/height; refusing to write an empty "
            "heights document, which a merge would read as 'measured, and "
            "there are none'")
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(document, indent=1, sort_keys=False) + "\n",
        encoding="utf-8")

    print(f"{arguments.out}: {document['sites_resolved']} of "
          f"{document['sites_attempted']} sites, worst position disagreement "
          f"{document['position_check_max_deg']:.2e} deg")
    if document["position_check_failures"]:
        print("POSITION CHECK FAILED for: "
              + ", ".join(document["position_check_failures"]))
    if document["outside_plausible_band"]:
        print("outside the plausible height band (reported, not refused): "
              + ", ".join(document["outside_plausible_band"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
