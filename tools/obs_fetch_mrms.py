#!/usr/bin/env python3
"""Pull one case window of MRMS composite reflectivity and decode it.

The campaign driver: it lists the window, fetches the objects the scoring
valid times actually need, decodes each into a `gpuwm-obs.obs-grid.v1` pack,
writes the geometry once, and leaves a manifest naming every SHA-256 it took.

Case identity lives in the arguments, never here: this script knows about
windows and boxes, and would run the same way for any of them.

    python tools/obs_fetch_mrms.py --start 2024-05-21T02:00:00Z \\
        --end 2024-05-21T18:00:00Z --bbox=-100,37,-88,45 --out CACHE/mrms
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The repository's own package, ahead of any installed wheel: a driver that
# imported a different gpuwm than the checkout it lives in would drive a
# different front door than the one just built.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.obs import frontdoor

TIME_IN = "%Y-%m-%dT%H:%M:%SZ"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="first scored valid time")
    parser.add_argument("--end", required=True, help="last scored valid time")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bbox", default=None,
                        help="W,S,E,N, spelled --bbox=-100,37,-88,45 (an "
                             "attached value; a western longitude opens "
                             "with a minus). Strongly advised: full CONUS "
                             "is 196 MB of float64 per frame")
    parser.add_argument("--step-hours", type=int, default=1)
    parser.add_argument("--window-seconds", type=int, default=240,
                        help="refuse a frame further than this from a valid time")
    parser.add_argument("--product", default=None)
    parser.add_argument("--cache", default=None, type=Path)
    arguments = parser.parse_args()

    door = frontdoor.MRMS
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    packs = out / "packs"
    packs.mkdir(exist_ok=True)

    start = datetime.strptime(arguments.start, TIME_IN).replace(tzinfo=timezone.utc)
    end = datetime.strptime(arguments.end, TIME_IN).replace(tzinfo=timezone.utc)
    common = []
    if arguments.product:
        common += ["--product", arguments.product]
    if arguments.cache:
        common += ["--cache", str(arguments.cache)]

    manifest = {"instrument": "mrms", "frames": [], "geometry": None}
    when = start
    geometry_written = False
    while when <= end:
        stamp = when.strftime(TIME_IN)
        # The archive stamps frames with off-cadence seconds, so the object
        # for a valid time is found by listing, never by constructing a URL.
        nearest = door.run("nearest",
                           ["--valid-time", stamp,
                            "--window-seconds", str(arguments.window_seconds),
                            *common],
                           schema="gpuwm-obs.mrms-nearest.v1")
        frame = nearest["frame"]
        fetched = door.run("fetch",
                           ["--start", frame["valid_time"] + "Z",
                            "--end", frame["valid_time"] + "Z",
                            "--out", str(out), *common],
                           schema="gpuwm-obs.mrms-fetch.v1")
        source = Path(fetched["files"][0]["path"])
        pack = packs / (source.stem + ".obspack")
        decode = ["--file", str(source), "--out", str(pack), *common]
        if arguments.bbox:
            decode += ["--bbox", arguments.bbox]
        record = door.run("decode", decode, schema="gpuwm-obs.mrms-decode.v1")
        manifest["frames"].append({
            "requested_valid_time": stamp,
            "frame_valid_time": frame["valid_time"],
            "offset_seconds": nearest["offset_seconds"],
            "source": str(source),
            "source_sha256": fetched["files"][0]["sha256"],
            "pack": str(pack),
            "pack_sha256": record["content_sha256"],
            "observed_fraction": record["sentinels"]["observed_fraction"],
        })
        if not geometry_written:
            geometry = packs / "geometry.obspack"
            grid = ["--file", str(source), "--out", str(geometry), *common]
            if arguments.bbox:
                grid += ["--bbox", arguments.bbox]
            written = door.run("grid", grid, schema="gpuwm-obs.mrms-grid.v1")
            manifest["geometry"] = {"pack": str(geometry),
                                    "pack_sha256": written["content_sha256"],
                                    "grid": written["grid"]}
            geometry_written = True
        print(f"{stamp} -> {frame['valid_time']} "
              f"({nearest['offset_seconds']:+d} s), observed "
              f"{record['sentinels']['observed_fraction']:.4f}")
        when += timedelta(hours=arguments.step_hours)

    path = out / "manifest_mrms.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest['frames'])} frames, manifest at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
