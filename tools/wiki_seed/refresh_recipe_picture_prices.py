"""Refresh saved recipe prices from their existing fitted grids and measured pictures.

No fitting, downloading or forecasting. Keep the saved download and preparation
receipts and recompute only the projection and its picture sentence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tools.wiki_seed import design_recipes as design

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=ROOT / "gpuwm/gui/seed/recipes")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    changes = []
    for path in sorted(args.recipes.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for recipe in document["recipes"]:
            for row in recipe["cards"]:
                if not row.get("fits"):
                    continue
                fit = {"domains": [{key: d[key] for key in
                                     ("nx", "ny", "nz", "dx_km", "cu_physics")}
                                    for d in row["domains"]],
                       "physics": row["physics"]["profile"],
                       "restart_interval_s": 3600, "fetch": None, "chain": None}
                output = row["output"]
                projected = design.projection(fit, row["length_h"] * 3600,
                    output["history_interval_s"], output["nest_history_interval_s"])
                total = projected["total_bytes"] + row["download"]["bytes"] + row["download"]["preparation_bytes"]
                price = round(total / GIB, 1)
                if total > row["disk_budget_gib"] * GIB:
                    raise ValueError(f"{path.name}/{row['card_gb']}: revised projection exceeds its budget")
                sentence, replacements = re.subn(r"the rendered pictures, [0-9.]+ GiB",
                    f"the rendered pictures, {projected['picture_bytes'] / GIB:.1f} GiB", row["disk_basis"])
                if replacements != 1:
                    raise ValueError(f"{path.name}/{row['card_gb']}: no unique picture-price sentence")
                changes.append({"file": path.name, "card_gb": row["card_gb"],
                    "before_gib": row["disk_gib"], "after_gib": price,
                    "picture_bytes": projected["picture_bytes"], "total_bytes": total})
                row["disk_gib"], row["disk_basis"] = price, sentence
        path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    args.receipt.write_text(json.dumps(changes, indent=2) + "\n")
    print(f"Refreshed {len(changes)} fitted recipe rows with their current measured picture prices.")


if __name__ == "__main__":
    main()
