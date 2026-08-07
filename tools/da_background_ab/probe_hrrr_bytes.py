"""Measure, without downloading it, what this A/B's HRRR fetch will cost.

The disk budget for a staged experiment has to be a number before the
experiment runs, not after.  For the two HRRR arms that number is
dominated by GRIB2, and the honest way to get it is the one thing NOAA
publishes for free: the ``.idx`` byte-range index beside every object.

So this asks the SAME selection the real downloader asks
(``tools.download_hrrr_native_subset._atmosphere_selection`` /
``_soil_selection`` / ``_coalesce``), over the SAME index, and sums the
coalesced ranges.  What it reports is therefore the exact number of
bytes the subset download will move and store -- not an estimate of it,
and not the whole object either, which is the number a reader would
otherwise assume.

It downloads one HEAD and one index per object (tens of kilobytes) and
writes nothing but its own receipt.  It touches no GPU and no GRIB2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.download_hrrr_native_subset import (            # noqa: E402
    ATMOSPHERE_RECORD_COUNT, BASE_URL, SOIL_RECORD_COUNT, _atmosphere_selection,
    _coalesce, _cycle, _head, _parse_index, _request_bytes, _soil_selection)

PROBE_SCHEMA = "gpuwm-da.background-ab-fetch-size.v1"


def probe_object(url: str, kind: str) -> dict:
    headers = _head(url)
    object_bytes = int(headers["content-length"])
    index_payload, _ = _request_bytes(url + ".idx")
    rows = _parse_index(index_payload, object_bytes)
    if kind == "atmosphere":
        selected = _atmosphere_selection(
            rows, expected_count=ATMOSPHERE_RECORD_COUNT)
    else:
        selected = _soil_selection(rows, expected_count=SOIL_RECORD_COUNT)
    ranges = _coalesce(rows, selected, object_bytes)
    subset_bytes = sum(item.size for item in ranges)
    return {
        "url": url,
        "kind": kind,
        "object_bytes": object_bytes,
        "records_published": len(rows),
        "records_selected": len(selected),
        "coalesced_ranges": len(ranges),
        "subset_bytes": subset_bytes,
        "subset_fraction_of_object": round(subset_bytes / object_bytes, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_background_ab.probe_hrrr_bytes",
        description=__doc__.splitlines()[0])
    parser.add_argument("--cycle", action="append", required=True,
                        help="YYYY-MM-DD_HH:MM:SS; repeatable")
    parser.add_argument("--forecast-hours", action="append", required=True,
                        help="comma-separated leads, one per --cycle")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.cycle) != len(args.forecast_hours):
        parser.error("one --forecast-hours per --cycle")

    entries = []
    for raw_cycle, raw_hours in zip(args.cycle, args.forecast_hours):
        cycle = _cycle(raw_cycle)
        prefix = f"{args.base_url.rstrip('/')}/hrrr.{cycle:%Y%m%d}/conus"
        for hour in (int(value) for value in raw_hours.split(",")):
            stem = f"hrrr.t{cycle:%H}z"
            for name, kind in ((f"{stem}.wrfnatf{hour:02d}.grib2",
                                "atmosphere"),
                               (f"{stem}.wrfprsf{hour:02d}.grib2", "soil")):
                record = probe_object(f"{prefix}/{name}", kind)
                record["cycle"] = cycle.isoformat() + "Z"
                record["forecast_hour"] = hour
                entries.append(record)
                print(f"{cycle:%Y-%m-%dT%H}Z f{hour:03d} {kind:<10} "
                      f"{record['subset_bytes'] / 1e6:8.1f} MB of "
                      f"{record['object_bytes'] / 1e6:8.1f} MB "
                      f"({record['records_selected']} records)", flush=True)

    total = sum(entry["subset_bytes"] for entry in entries)
    payload = {
        "schema": PROBE_SCHEMA,
        "objects": entries,
        "total_subset_bytes": total,
        "total_subset_gib": round(total / 2 ** 30, 3),
        "method": ("HEAD + .idx per object, then the downloader's own "
                   "selection and range coalescing; no GRIB2 was moved"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\nTOTAL {total / 2 ** 30:.3f} GiB across {len(entries)} objects "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
