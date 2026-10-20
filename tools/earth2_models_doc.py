#!/usr/bin/env python3
"""Render the Earth-2 authority rows in the public quickstart."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- earth2-models:begin -->"
END = "<!-- earth2-models:end -->"


def cell(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "&#124;")


def render_table(table_path: Path) -> str:
    data = table_path.read_bytes()
    table = json.loads(data)
    gated = sorted(k for k, v in table["models"].items() if (v.get("weights") or {}).get("gated"))
    accounts = sorted(k for k, v in table["models"].items() if (v.get("weights") or {}).get("account_needed"))
    rows = [
        "Generated from `gpuwm/authorities/earth2-models.v1.json`.",
        f"Authority SHA-256: `{hashlib.sha256(data).hexdigest()}`.",
        "",
        "These are authority declarations, not a list of measured forecasts.",
        "Gated rows declared here: " + (", ".join(f"`{k}`" for k in gated) or "none") + ".",
        "Account-required rows declared here: " + (", ".join(f"`{k}`" for k in accounts) or "none") + ".",
        "",
        "| ID | Model | Step (h) | Licence declared for weights | Use | Weight access declared | Profile | Inference memory |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_id, model in sorted(table["models"].items()):
        weights = model.get("weights") or {}
        licence = weights.get("licence") or "Not declared"
        normalized = str(licence).upper().replace("_", "-")
        restricted = "-NC" in normalized or "NON-COMMERCIAL" in normalized or "NONCOMMERCIAL" in normalized
        use = "Research-only" if restricted else "See declared licence"
        if model.get("licence_class") == "restricted":
            use = "Explicit --accept-licence required"
        if model.get("runtime_refusal"):
            use = ("Research-only; " if restricted else "") + "Refused by runtime"
        access = "Not declared"
        if weights.get("gated"):
            access = "Gated"
        elif weights.get("account_needed"):
            access = "Account required"
        elif weights.get("gated") is False and weights.get("account_needed") is False:
            access = "No account declared"
        rows.append("| " + " | ".join(cell(v) for v in (
            model_id, model["name"], model.get("step_hours", "Not declared"),
            licence, use, access, model.get("profile") or "Not declared",
            (str(model["min_vram_gib"]) + " GiB free; " + model.get("vram_note", "")) if model.get("min_vram_gib") else "Capacity floor not declared",
        )) + " |")
    return "\n".join(rows)


def updated_document(document: str, table: str) -> str:
    if document.count(BEGIN) != 1 or document.count(END) != 1:
        raise ValueError("quickstart must contain exactly one pair of model-table markers")
    start = document.index(BEGIN) + len(BEGIN)
    stop = document.index(END)
    if start > stop:
        raise ValueError("model-table markers are reversed")
    return document[:start] + "\n" + table + "\n" + document[stop:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=ROOT / "gpuwm/authorities/earth2-models.v1.json")
    parser.add_argument("--doc", type=Path, default=ROOT / "docs/public/EARTH2.md")
    parser.add_argument("--check", action="store_true", help="exit 1 if the generated table is stale")
    args = parser.parse_args()
    try:
        before = args.doc.read_text(encoding="utf-8")
        after = updated_document(before, render_table(args.table))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"earth2_models_doc: {exc}\n")
    if args.check:
        if before != after:
            parser.exit(1, "Earth-2 table is stale; run python tools/earth2_models_doc.py\n")
        print("Earth-2 model table matches the authority.")
    elif before != after:
        args.doc.write_text(after, encoding="utf-8")
        print(f"Updated {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
