#!/usr/bin/env python3
"""Pull one case window of MRMS composite reflectivity and decode it.

The campaign driver: it lists the window, fetches the objects the scoring
valid times actually need, decodes each into a `gpuwm-obs.obs-grid.v1` pack,
writes the geometry once, and leaves a manifest naming every SHA-256 it took.

Case identity lives in the arguments, never here: this script knows about
windows and boxes, and would run the same way for any of them.

The archive work itself is `gpuwm.obs.mrms_fetch`, which is also what a local
DA run's nowcast score calls. This file is the campaign's entry point onto
that library and not a second driver of the same front door: two drivers of
one archive are two answers to "which object was taken".

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

from gpuwm.obs import mrms_fetch

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

    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    cache = mrms_fetch.MrmsCompositeCache(
        arguments.cache or out, bbox=arguments.bbox,
        window_seconds=arguments.window_seconds, product=arguments.product)

    start = datetime.strptime(arguments.start, TIME_IN).replace(tzinfo=timezone.utc)
    end = datetime.strptime(arguments.end, TIME_IN).replace(tzinfo=timezone.utc)
    requested = []
    when = start
    while when <= end:
        requested.append(when.strftime("%Y-%m-%dT%H:%M:%S"))
        when += timedelta(hours=arguments.step_hours)

    manifest = {"instrument": "mrms", "frames": [], "geometry": None}
    for stamp in requested:
        frame = cache.ensure(stamp)
        manifest["frames"].append({
            "requested_valid_time": stamp,
            "frame_valid_time": frame.valid_time,
            "offset_seconds": frame.offset_seconds,
            "archive_key": frame.key,
            "archive_object_uri": frame.object_uri,
            "source_sha256": frame.object_sha256,
            "pack": frame.pack_path,
            "pack_sha256": frame.pack_sha256,
            "observed_fraction": frame.observed_fraction,
        })
        print(f"{stamp} -> {frame.valid_time} "
              f"({frame.offset_seconds:+.0f} s), observed "
              f"{frame.observed_fraction:.4f}")
    manifest["geometry"] = cache.record().get("geometry")

    path = out / "manifest_mrms.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{len(manifest['frames'])} frames, manifest at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
