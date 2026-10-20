#!/usr/bin/env python3
"""Refresh the documented inventory from rw_grib2export's own catalog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    catalog = json.loads(subprocess.check_output([str(args.binary), "--list"], text=True))
    doc = Path(__file__).resolve().parents[1] / "docs/grib2-export.md"
    text = doc.read_text()
    for kind in ("surface", "pressure"):
        columns = ["id"] + (["group"] if kind == "surface" else []) + ["grib", "level", "units", "method"]
        title = ["GRIB2" if c == "grib" else c for c in columns]
        header = "| " + " | ".join(title) + " |"
        rows = [header, "| " + " | ".join("---" for _ in columns) + " |"]
        for row in catalog[f"{kind}_" + ("fields" if kind == "surface" else "rows")]:
            values = [str(row[c]).replace("|", "\\|") for c in columns]
            values[0] = f"`{values[0]}`"
            rows.append("| " + " | ".join(values) + " |")
        start = f"<!-- catalog:{kind}:start -->"
        end = f"<!-- catalog:{kind}:end -->"
        generated = start + "\n" + "\n".join(rows) + "\n" + end
        if start in text:
            a = text.index(start)
            b = text.index(end, a) + len(end)
        else:
            a = text.index(header)
            b = text.index("\n\n", a)
        text = text[:a] + generated + text[b:]
    doc.write_text(text)


if __name__ == "__main__":
    main()
