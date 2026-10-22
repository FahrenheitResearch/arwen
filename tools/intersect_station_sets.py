"""Intersect declared station admissions before comparing forecast arms.

This operates on station identities only. Every input must bind the same
station table and valid times. Each scorer checks that it admits every id
of the result, then projects those stations on its own grid.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(p.read_bytes()) for p in args.inputs]
    reference = rows[0]
    ids = set(reference["station_ids"])
    for row in rows:
        for key in ("schema", "station_table_sha256", "valid_times"):
            if row[key] != reference[key]:
                parser.error(f"station admissions disagree on {key}")
        if row["schema"] != "obs-shared-station-set-v1":
            parser.error("unsupported station set schema")
        ids.intersection_update(row["station_ids"])
    if not ids:
        parser.error("no stations are admitted by every arm")
    result = {key:reference[key] for key in ("schema", "station_table_sha256", "valid_times")}
    result["station_ids"] = sorted(ids)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
