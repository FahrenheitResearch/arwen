#!/usr/bin/env python3
"""Refresh EXPORT-EARTH2.md tables from the packaged export metadata.

Only the marked tables are generated. Examples are copied from whole-run
receipts, and are deliberately outside the generated regions.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/public/EXPORT-EARTH2.md"
AUTHORITY = ROOT / "gpuwm/authorities/earth2-models.v1.json"
PROFILES = ROOT / "gpuwm/data/ml_export/profiles.json"
CATALOG = ROOT / "tools/rustwx/crates/rw-grib2export/src/catalog.rs"
GRIB_IDS = ("specific_humidity", "omega", "vertical_velocity", "u100", "v100")


def cell(value: object) -> str:
    """Keep metadata in one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def code(value: object) -> str:
    return f"`{cell(value)}`"


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def consumer_table(consumers: dict, profiles: dict) -> str:
    rows = []
    for key, row in consumers.items():
        profile = row["profile"]
        if profile not in profiles:
            raise ValueError(f"consumer {key}: unknown export profile {profile}")
        channels = row["input_channels"]
        if not channels or len(channels) != len(set(channels)):
            raise ValueError(f"consumer {key}: empty or repeated input channels")
        rows.append([
            code(key), code(row["earth2studio_class"]), code(profile),
            " x ".join(map(str, row.get("input_grid", {}).get("shape", []))),
            str(len(channels)), ", ".join(map(code, channels)),
        ])
    return table(["Consumer", "Earth2Studio class", "Export profile", "Declared input shape", "Count", "Input channels (in order)"], rows)


def profile_table(profiles: dict) -> str:
    return table(
        ["Profile", "Kind", "Spatial dimensions", "Shape", "Grid parameters"],
        [[
            code(key), code(row["kind"]), ", ".join(map(code, row["dims"])),
            " x ".join(map(str, row.get("shape", []))),
            "; ".join(f"{code(k)}={code(v)}" for k, v in row.items()
                      if k not in {"kind", "dims", "shape", "source"}
                      and not isinstance(v, (dict, list))),
        ] for key, row in profiles.items()],
    )


def grib_table(catalog: str) -> str:
    """Read the five new export rows, including their method strings, verbatim."""
    found = {}
    for line in catalog.splitlines():
        pressure = re.search(
            r'Row \{ id: "([^"]+)".*?discipline: (\d+), category: (\d+), '
            r'number: (\d+), level: \((\d+), ([\d.]+)\).*?'
            r'units: "([^"]+)".*?method: "([^"]+)"', line,
        )
        surface = re.search(
            r'row\("([^"]+)", Source::WindHeight.*?, \((\d+), (\d+), (\d+)\), '
            r'\((\d+), ([\d.]+)\), None, "([^"]+)", "([^"]+)"', line,
        )
        match = pressure or surface
        if match and match[1] in GRIB_IDS:
            key, discipline, category, number, level_type, level_value, units, method = match.groups()
            level = f"{level_type}/pressure in Pa" if pressure else f"{level_type}/{float(level_value):g} m AGL"
            found[key] = [
                code(key), code(f"{discipline}.{category}.{number}"), cell(level),
                cell(units), cell(method),
            ]
    missing = set(GRIB_IDS) - found.keys()
    if missing:
        raise ValueError(f"GRIB2 catalog rows missing or unparsed: {', '.join(sorted(missing))}")
    return table(["Row", "Discipline.category.parameter", "Fixed surface", "Units", "Formula / method"],
                 [found[key] for key in GRIB_IDS])


def replace_region(document: str, name: str, body: str) -> str:
    start = f"<!-- BEGIN GENERATED {name} -->"
    end = f"<!-- END GENERATED {name} -->"
    if document.count(start) != 1 or document.count(end) != 1:
        raise ValueError(f"expected one marker pair for {name}")
    before, rest = document.split(start)
    _, after = rest.split(end)
    return before + start + "\n\n" + body + "\n\n" + end + after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=AUTHORITY)
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--document", type=Path, default=DOCUMENT)
    parser.add_argument("--check", action="store_true", help="report stale tables without writing")
    args = parser.parse_args()
    try:
        consumers = json.loads(args.authority.read_text())["consumers"]
        profiles = json.loads(args.profiles.read_text())["profiles"]
        if not consumers or not profiles:
            raise ValueError("consumer and export-profile tables must both contain rows")
        original = args.document.read_text()
        updated = replace_region(original, "earth2-profiles", profile_table(profiles))
        updated = replace_region(updated, "earth2-channels", consumer_table(consumers, profiles))
        updated = replace_region(updated, "earth2-grib2", grib_table(args.catalog.read_text()))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    if args.check:
        if original != updated:
            print("".join(difflib.unified_diff(
                original.splitlines(keepends=True), updated.splitlines(keepends=True),
                fromfile=str(args.document), tofile="regenerated",
            )), end="")
            return 1
        return 0
    args.document.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
