#!/usr/bin/env python3
"""Assemble the observation archive-of-record manifests from fetch records.

The battery keeps its observation archive OUTSIDE the repository -- 9 GB of
archive objects is data, not source -- so what makes that archive an asset
rather than a directory is the manifest: every object named by its archive
URL, its SHA-256 at fetch, its byte count and the instant it was retrieved.
The manifest is what the mirror to the archive-of-record host is verified
against, and it is what a re-fetch three months from now is compared with
when an archive quietly re-issues a file.

Nothing here fetches, decodes or scores.  It reads the records the front
doors already wrote, re-reads nothing, and invents no digest of its own: a
manifest whose numbers came from anywhere but the fetch that took them would
be a second opinion about bytes nobody looked at twice.

A case day's radar objects can arrive in more than one pull -- the whole UTC
day, and then the hourly frames past midnight that the scored window reaches
-- so a ``mrms-fetch-<day>-dayplus1.json`` record beside the whole-day one is
folded into the same case, and every pull is named in ``mrms.windows``.

    python tools/obs_battery_manifest.py --receipts CACHE/receipts \\
        --asos-root CACHE/asos --outdir docs/public/receipts/obsbattery/manifests \\
        --case 2024-05-21 ... --battery-manifest .../OBS-ARCHIVE-MANIFEST.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SCHEMA = "gpuwm.obs-battery-archive-manifest/v1"
BATTERY_SCHEMA = "gpuwm.obs-battery-archive-manifest-index/v1"

#: The accumulation windows the battery pulls for every case day.
STAGE4_ACCUMULATIONS = ("01h", "06h", "24h")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_digest(entries: list[dict]) -> str:
    """One digest over an object list: the sha256 of its sorted digests.

    Cheap to recompute on a mirror without re-reading the manifest's prose,
    and it changes if any object is added, dropped or altered.
    """
    joined = "\n".join(sorted(str(entry["sha256"]) for entry in entries))
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def mrms_entries(record: dict) -> list[dict]:
    bucket = str(record["window"]["bucket"])
    return [{
        "url": f"https://{bucket}.s3.amazonaws.com/{entry['key']}",
        "key": entry["key"],
        "valid_time": entry["valid_time"],
        "bytes": int(entry["bytes"]),
        "sha256": entry["sha256"],
        "fetched_at": entry["fetched_at"],
    } for entry in record["files"]]


def stage4_entries(record: dict) -> list[dict]:
    return [{
        "url": entry["url"],
        "valid_time": entry["valid_time"],
        "accumulation_hours": int(entry["accumulation_hours"]),
        "bytes": int(entry["bytes"]),
        "sha256": entry["sha256"],
        "fetched_at": entry["fetched_at"],
    } for entry in record["files"]]


def mrms_window(record: dict) -> dict:
    return {"start": record["window"]["start"], "end": record["window"]["end"],
            "objects": len(record["files"])}


def case_manifest(day: str, *, receipts: Path, asos_root: Path) -> dict:
    mrms_record = _read(receipts / f"mrms-fetch-{day}.json")
    mrms = mrms_entries(mrms_record)
    windows = [mrms_window(mrms_record)]
    # A case day's radar arrives in more than one pull.  The whole-day pull
    # stops at 23:59:59 and the scored window does not, so the hourly frames
    # past midnight are a second record; and a lead whose nearest frame fails
    # the registered coverage floor needs every frame inside the tolerance
    # decoded (registration v2.1 section 6), which is a third.  Every
    # `mrms-fetch-<day>-*.json` beside the whole-day one is folded in here,
    # because what this file describes is the archive, not any one pull.
    for supplement in sorted(receipts.glob(f"mrms-fetch-{day}-*.json")):
        extra = _read(supplement)
        mrms += mrms_entries(extra)
        windows.append(mrms_window(extra))
    # An object named by two records is one object in the archive.
    mrms = list({entry["key"]: entry for entry in mrms}.values())
    mrms.sort(key=lambda entry: entry["valid_time"])

    stage4: dict[str, list[dict]] = {}
    for token in STAGE4_ACCUMULATIONS:
        path = receipts / f"stage4-fetch-{day}-{token}.json"
        record = _read(path)
        stage4[token] = stage4_entries(record)

    asos_dir = asos_root / day
    asos_manifest = _read(asos_dir / "manifest_asos.json")
    asos_files = []
    for name in ("stations.json", "observations.csv", "surface.json"):
        path = asos_dir / name
        asos_files.append({"name": name, "bytes": path.stat().st_size,
                           "sha256": _sha256_file(path)})

    stage4_flat = [entry for rows in stage4.values() for entry in rows]
    return {
        "schema": MANIFEST_SCHEMA,
        "case_day": day,
        "mrms": {
            "bucket": mrms_record["window"]["bucket"],
            "region": mrms_record["window"]["region"],
            "product": mrms_record["window"]["product"],
            "window": {"start": mrms_record["window"]["start"],
                       "end": mrms_record["window"]["end"]},
            "windows": windows,
            "objects": len(mrms),
            "bytes": sum(entry["bytes"] for entry in mrms),
            "object_list_sha256": _object_digest(mrms),
            "files": mrms,
        },
        "stage4": {
            "archive": _read(receipts / f"stage4-fetch-{day}-01h.json")["archive"],
            "accumulations": {token: {"objects": len(rows),
                                      "bytes": sum(e["bytes"] for e in rows),
                                      "files": rows}
                              for token, rows in stage4.items()},
            "objects": len(stage4_flat),
            "bytes": sum(entry["bytes"] for entry in stage4_flat),
            "object_list_sha256": _object_digest(stage4_flat),
        },
        "asos": {
            "archive": asos_manifest["record_provenance"]["product"],
            "stations_frozen": int(asos_manifest["stations_frozen"]),
            "station_table_sha256": asos_manifest["station_table_sha256"],
            "observations_sha256": asos_manifest["observations_sha256"],
            "screen": asos_manifest["screen"],
            "files": asos_files,
            "bytes": sum(entry["bytes"] for entry in asos_files),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--asos-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--battery-manifest", required=True, type=Path)
    parser.add_argument("--case", action="append", required=True,
                        help="a case day, YYYY-MM-DD; repeat for each")
    parser.add_argument("--cache-root", required=True,
                        help="where the objects live on the box that pulled them")
    parser.add_argument("--evaluator-commit", required=True)
    arguments = parser.parse_args()

    arguments.outdir.mkdir(parents=True, exist_ok=True)
    index = []
    total_objects = total_bytes = 0
    for day in arguments.case:
        manifest = case_manifest(day, receipts=arguments.receipts,
                                 asos_root=arguments.asos_root)
        path = arguments.outdir / f"obs-{day.replace('-', '')}.json"
        text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        objects = (manifest["mrms"]["objects"] + manifest["stage4"]["objects"]
                   + len(manifest["asos"]["files"]))
        size = (manifest["mrms"]["bytes"] + manifest["stage4"]["bytes"]
                + manifest["asos"]["bytes"])
        total_objects += objects
        total_bytes += size
        index.append({
            "case_day": day,
            "manifest": str(path.as_posix()),
            "manifest_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "objects": objects,
            "bytes": size,
            "mrms_objects": manifest["mrms"]["objects"],
            "mrms_object_list_sha256": manifest["mrms"]["object_list_sha256"],
            "stage4_objects": manifest["stage4"]["objects"],
            "stage4_object_list_sha256": manifest["stage4"]["object_list_sha256"],
            "asos_stations_frozen": manifest["asos"]["stations_frozen"],
            "asos_station_table_sha256": manifest["asos"]["station_table_sha256"],
        })
        print(f"{day}: {objects} objects, {size / 1e9:.3f} GB -> {path}")

    battery = {
        "schema": BATTERY_SCHEMA,
        "evaluator_commit": str(arguments.evaluator_commit),
        "cache_root": str(arguments.cache_root),
        "case_days": len(index),
        "objects": total_objects,
        "bytes": total_bytes,
        "routes": {
            "mrms": "AWS Open Data noaa-mrms-pds, anonymous list+get",
            "stage4": ("Iowa Environmental Mesonet archive, anonymous HTTPS "
                       "(WAVE-ERRATA-20260804 section 1)"),
            "asos": "Iowa Environmental Mesonet asos.py window, station-bounded",
        },
        "retention": ("the objects are NOT in the repository; this index and "
                      "the per-case manifests beside it are what a mirror is "
                      "verified against"),
        "cases": index,
    }
    arguments.battery_manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.battery_manifest.write_text(
        json.dumps(battery, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{total_objects} objects, {total_bytes / 1e9:.3f} GB total -> "
          f"{arguments.battery_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
