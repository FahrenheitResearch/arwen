#!/usr/bin/env python3
"""Freeze a station table, pull one case window of surface obs, decode it.

Three steps in the order the battery needs them: the station table is frozen
and hashed FIRST, because it is registration state and the case's station set
is pinned by its digest; the observation window is then fetched bounded by
that table; and the reports are screened and converted into seam units.

The fetch window is deliberately wider than the scored window: a report is
matched to a valid time within ten minutes either side, so the ends need
slack, and a decode whose window reaches past the CSV drops every station for
incompleteness.

    python tools/obs_fetch_asos.py --networks IA_ASOS,IL_ASOS \\
        --bbox=-100,37,-88,45 --start 2024-05-21T02:00:00Z \\
        --end 2024-05-21T18:00:00Z --out CACHE/asos

``--product asos1min`` takes the same stations from the archive's one-minute
ASOS pages instead of the METARs: a report every minute, decoded at a
one-minute valid-time stride, so a DA cycle of a few minutes gets a surface
report at every analysis rather than at the few after each hour.  The
decoded record is the same seam (``gpuwm-obs.asos-surface.v2``) and its
provenance names the product (``iem-asos-1min``).
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
    parser.add_argument("--networks", required=True,
                        help="comma-separated IEM networks, e.g. IA_ASOS,IL_ASOS")
    parser.add_argument("--start", required=True, help="first scored valid time")
    parser.add_argument("--end", required=True, help="last scored valid time")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bbox", default=None,
                        help="W,S,E,N, spelled --bbox=-100,37,-88,45 (an "
                             "attached value; a western longitude opens with "
                             "a minus, which argparse otherwise reads as "
                             "another flag)")
    parser.add_argument("--slack-minutes", type=int, default=60,
                        help="fetch this much either side of the scored window")
    parser.add_argument("--product", choices=("metar", "asos1min"),
                        default="metar",
                        help="metar (default): routine and special METARs. "
                             "asos1min: the one-minute ASOS pages for the "
                             "same stations, a report every minute")
    stride = parser.add_mutually_exclusive_group()
    stride.add_argument("--step-hours", type=int, default=None,
                        help="valid-time stride in hours (default 1 for "
                             "metar)")
    stride.add_argument("--step-minutes", type=int, default=None,
                        help="valid-time stride in minutes (default 1 for "
                             "asos1min)")
    parser.add_argument("--min-report-rate", type=float, default=None)
    arguments = parser.parse_args()

    door = frontdoor.ASOS
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)

    stations = out / "stations.json"
    freeze = ["--networks", arguments.networks, "--out", str(stations)]
    if arguments.bbox:
        freeze += ["--bbox", arguments.bbox]
    frozen = door.run("stations", freeze, schema="gpuwm-obs.asos-stations.v1")
    print(f"froze {frozen['stations']} stations, "
          f"table sha256 {frozen['content_sha256']}")

    start = datetime.strptime(arguments.start, TIME_IN).replace(tzinfo=timezone.utc)
    end = datetime.strptime(arguments.end, TIME_IN).replace(tzinfo=timezone.utc)
    slack = timedelta(minutes=arguments.slack_minutes)
    csv = out / "observations.csv"
    product = ["--product", arguments.product]
    fetched = door.run("fetch",
                       ["--stations", str(stations),
                        "--start", (start - slack).strftime(TIME_IN),
                        "--end", (end + slack).strftime(TIME_IN),
                        "--out", str(csv), *product],
                       schema="gpuwm-obs.asos-fetch.v1")
    print(f"fetched {fetched['rows']} rows, sha256 {fetched['sha256']}")

    record = out / "surface.json"
    decode = ["--stations", str(stations), "--obs", str(csv),
              "--start", arguments.start, "--end", arguments.end,
              "--out", str(record), *product]
    # The stride the caller gave, else the product's own (one hour for the
    # METARs, one minute for the one-minute pages).
    if arguments.step_hours is not None:
        decode += ["--step-hours", str(arguments.step_hours)]
    if arguments.step_minutes is not None:
        decode += ["--step-minutes", str(arguments.step_minutes)]
    if arguments.min_report_rate is not None:
        decode += ["--min-report-rate", f"{arguments.min_report_rate:g}"]
    decoded = door.run("decode", decode, schema="gpuwm-obs.asos-surface.v2")
    print(f"decoded {decoded['reports']} reports from {decoded['stations']} "
          f"stations over {decoded['valid_times']} valid times")

    manifest = {
        "instrument": "asos",
        "product": arguments.product,
        "station_table": str(stations),
        "station_table_sha256": frozen["content_sha256"],
        "stations_frozen": frozen["stations"],
        "observations_csv": str(csv),
        "observations_sha256": fetched["sha256"],
        "record": str(record),
        "record_provenance": decoded["provenance"],
        "screen": decoded["screen"],
    }
    path = out / "manifest_asos.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmanifest at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
