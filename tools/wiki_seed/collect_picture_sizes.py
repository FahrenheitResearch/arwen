"""Collect picture-size receipts and rebuild the packaged disk-price table.

An audit directory is needed only to collect new receipts. Rebuilding from
the committed record needs no run folders, renderer, network or device.
Examples, from the repository root::

    python tools/wiki_seed/collect_picture_sizes.py --audits PATH
    python tools/wiki_seed/collect_picture_sizes.py --rebuild
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "tools/wiki_seed/runs/bytes-per-product/sizes.json"
TABLE = ROOT / "gpuwm/data/picture-bytes.v1.json"
PICTURE_FIELDS = ["source_run", "product", "columns", "frame", "bytes", "date", "spacing_m"]
BASIS = (
    "Per-picture measurements, including sampled frames of longer and wet runs, in "
    "tools/wiki_seed/runs/bytes-per-product/sizes.json; reproducible with "
    "tools/wiki_seed/collect_picture_sizes.py --rebuild. Each product uses its maximum "
    "measured bytes in each horizontal-grid bracket. The headroom factor is 1.0: the "
    "maxima over all audits already cover complete runs, and another 10 percent would "
    "exceed the retained run-price ceilings. Unsampled catalog rows use 650/750 kB, covering the aggregate "
    "per-picture sizes in tools/wiki_seed/runs/bytes-per-cell/sizes.json. fallback_bytes "
    "is the largest single picture in the per-product record, before that same headroom. "
    "tools/wiki_seed/runs/bytes-per-picture/sizes.json supplies the independent run "
    "coverage and price-ceiling checks. Groups are a measured wrfout catalog, not a "
    "render availability contract. Interpolate between the horizontal-grid brackets "
    "and hold their endpoints outside them. Grid spacing is recorded but is not a "
    "price dimension: fine-spacing grids can be over-estimated by about 1.7 to 2.5 "
    "times, since field detail and map content affect compression."
)


def collect(audit_root: Path, catalog: dict) -> dict:
    """Keep every audited picture, including samples that are not full inventories."""
    runs, pictures = {}, []
    for path in sorted(audit_root.rglob("artifact-audit.json")):
        payload = path.read_bytes()
        audit = json.loads(payload)
        if not audit.get("pictures"):
            continue
        digest = hashlib.sha256(payload).hexdigest()
        source_run = digest[:16]
        runs[source_run] = {
            "audit_sha256": digest,
            "audit_date": audit["checked_utc"],
            "history_frames": len(audit["frames"]),
            "picture_files": audit["png_count"],
            "audited_pictures": len(audit["pictures"]),
            "complete_inventory": len(audit["pictures"]) == audit["png_count"],
        }
        grids = {}
        for frame in audit["frames"]:
            attrs = frame["attributes"]
            gid = int(attrs["GRID_ID"])
            dims = frame["dimensions"]
            columns = [int(dims["west_east"]), int(dims["south_north"])]
            spacing = [float(attrs["DX"]), float(attrs["DY"])]
            grid = (columns, spacing)
            if gid in grids and grids[gid] != grid:
                raise ValueError("An audit grid changes size or spacing; record frames separately.")
            grids[gid] = grid
        for picture in audit["pictures"]:
            name = picture["path"].replace("\\", "/")
            match = re.search(r"(?:^|/)d(\d+)(?:-|/)", name)
            # Overview images have no single domain. Retain their bytes,
            # but do not invent a grid for the product-size brackets.
            columns, spacing = (None, None) if match is None else grids[int(match[1])]
            parts = name.split("/")
            date = parts[-2] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[-2]) else audit["checked_utc"][:10]
            pictures.append([source_run, picture["product"], columns, parts[-1],
                             int(picture["bytes"]), date, spacing])
    return {
        "schema": "gpuwm-picture-size-record-v1",
        "description": "Per-picture byte counts from default-size rw_wrfbatch artifact audits. "
                       "Source-run ids are audit SHA256 prefixes; dates and grid metadata are retained. "
                       "Sampled longer runs contribute sizes, never inferred full-run totals. "
                       "Overview images without a domain retain null grid metadata.",
        "columns": [1296, 19344], "headroom": 1.0,
        "unmeasured_bytes": [650000, 750000],
        "catalog": {name: row[:2] for name, row in sorted(catalog.items())},
        "runs": runs, "picture_fields": PICTURE_FIELDS,
        "pictures": sorted(pictures),
    }


def build_table(record: dict) -> dict:
    """Maxima within grid brackets; the runtime applies the recorded headroom once."""
    low, _high = record["columns"]
    measured = defaultdict(lambda: [[], []])
    for _run, product, columns, _frame, size, _date, _spacing in record["pictures"]:
        if columns is not None:
            measured[product][int(math.prod(columns) > low)].append(size)
    rows = {}
    for name, (kind, first) in record["catalog"].items():
        small, large = [max(values) if values else fallback
                        for values, fallback in zip(measured[name], record["unmeasured_bytes"])]
        rows[name] = [kind, first, small, max(small, large)]
    return {
        "schema": "gpuwm-picture-bytes-v1", "basis": BASIS,
        "columns": record["columns"], "headroom": record["headroom"],
        "fallback_bytes": max(picture[4] for picture in record["pictures"]),
        "generic_bytes": record["unmeasured_bytes"],
        "row_fields": ["kind", "minimum_hour", "small_grid_bytes", "large_grid_bytes"],
        "products": rows,
    }


def write_document(path: Path, document: dict, rows_key: str) -> None:
    """Compact individual receipt rows, with stable UTF-8/LF bytes."""
    header = dict(document)
    rows = header.pop(rows_key)
    if isinstance(rows, dict):
        body = "{\n" + ",\n".join("    " + json.dumps(k) + ": " + json.dumps(v)
                                    for k, v in rows.items()) + "\n  }"
    else:
        body = "[\n" + ",\n".join("    " + json.dumps(row) for row in rows) + "\n  ]"
    text = json.dumps(header, indent=2)[:-2] + ',\n  "' + rows_key + '": ' + body + "\n}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audits", type=Path)
    action.add_argument("--rebuild", action="store_true")
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--table", type=Path, default=TABLE)
    args = parser.parse_args()
    if args.audits is not None:
        catalog = json.loads(args.table.read_text(encoding="utf-8"))["products"]
        record = collect(args.audits, catalog)
        write_document(args.record, record, "pictures")
    else:
        record = json.loads(args.record.read_text(encoding="utf-8"))
    write_document(args.table, build_table(record), "products")
    print(f"Recorded {len(record['pictures'])} pictures from {len(record['runs'])} audits.")


if __name__ == "__main__":
    main()
