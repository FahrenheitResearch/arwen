#!/usr/bin/env python3
"""Pull one case window of Stage-IV precipitation and decode it.

Fetches both accumulation windows the battery scores (hourly and six-hourly),
decodes each into a `gpuwm-obs.obs-grid.v1` pack, writes the HRAP geometry
once, and leaves a manifest naming every SHA-256 it took.

    python tools/obs_fetch_stage4.py --start 2024-05-21T00:00:00Z \\
        --end 2024-05-22T00:00:00Z --out CACHE/stage4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The repository's own package, ahead of any installed wheel: a driver that
# imported a different gpuwm than the checkout it lives in would drive a
# different front door than the one just built.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.obs import frontdoor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--accumulations", default="01h,06h",
                        help="comma-separated; the battery scores 01h and 06h")
    parser.add_argument("--cache", default=None, type=Path)
    arguments = parser.parse_args()

    door = frontdoor.STAGE4
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    packs = out / "packs"
    packs.mkdir(exist_ok=True)
    common = ["--cache", str(arguments.cache)] if arguments.cache else []

    manifest = {"instrument": "stage4", "accumulations": {}, "geometry": None}
    geometry_written = False
    for token in [t.strip() for t in arguments.accumulations.split(",") if t.strip()]:
        fetched = door.run("fetch",
                           ["--start", arguments.start, "--end", arguments.end,
                            "--accumulation", token, "--out", str(out), *common],
                           schema="gpuwm-obs.stage4-fetch.v1")
        rows = []
        for entry in fetched["files"]:
            source = Path(entry["path"])
            pack = packs / (source.name.replace(".", "_") + ".obspack")
            record = door.run("decode",
                              ["--file", str(source), "--out", str(pack)],
                              schema="gpuwm-obs.stage4-decode.v1")
            rows.append({
                "valid_time": record["valid_time"],
                "accumulation_hours": record["accumulation_hours"],
                "source": str(source),
                "source_sha256": entry["sha256"],
                "pack": str(pack),
                "pack_sha256": record["content_sha256"],
                "observed_fraction": record["mask"]["observed_fraction"],
                # Recorded per object because the vendored decoder's
                # soundness on template 5.3 turns on the missing cells being
                # in a bitmap rather than in missing-value management.
                "data_representation_template":
                    record["packing"]["data_representation_template"],
                "bitmap_present": record["packing"]["bitmap_present"],
            })
            if not geometry_written:
                geometry = packs / "geometry.obspack"
                written = door.run("grid",
                                   ["--file", str(source), "--out", str(geometry)],
                                   schema="gpuwm-obs.stage4-grid.v1")
                manifest["geometry"] = {"pack": str(geometry),
                                        "pack_sha256": written["content_sha256"],
                                        "grid": written["grid"]}
                geometry_written = True
        manifest["accumulations"][token] = rows
        print(f"{token}: {len(rows)} object(s)")

    path = out / "manifest_stage4.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
