"""Refresh saved recipe prices from their existing fitted grids and measured pictures.

No downloading or forecasting. Recompute the projection and its sentences,
keeping the saved grids and preparation receipts wherever they still fit.

A row whose saved intervals no longer fit its card takes the densest intervals
of the designer's own ladder that do. If none fit, the original designer chooses
a smaller rung for that event and card using the current engine projection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tools.wiki_seed import design_recipes as design
from tools.wiki_seed.build_recipes import minutes

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def _ladder(kind: str, domains: int):
    ladder = design.TC_INTERVALS if kind == "tc" else design.TORNADO_INTERVALS
    if domains == 1:
        ladder = tuple(dict.fromkeys((root, root) for root, _ in ladder))
    return ladder


def _interval_words(root_s, nest_s, domains: int) -> str:
    words = f"written every {minutes(root_s)} on the outer grid"
    if domains > 1:
        words += f" and every {minutes(nest_s)} on the nests"
    return words


def _subn_once(pattern, replacement, text, what, where):
    text, replacements = re.subn(pattern, replacement, text)
    if replacements != 1:
        raise ValueError(f"{where}: no unique {what} sentence")
    return text


def _redesign(document, event_id, card):
    from tools.wiki_seed import build_recipes as builder

    seed = json.loads((ROOT / "gpuwm/gui/seed/wiki-seed.json").read_text(encoding="utf-8"))
    event = next(event for event in seed["events"] if event["id"] == event_id)
    plan = design.tc_plan(event) if event["type"] == "tropical-cyclone" else design.tornado_plan(event)
    tried = []
    entry = {"rung_index": None, "tried": tried}
    for index, rung in enumerate(plan["rungs"]):
        fit = design.check(rung, plan, card)
        tried.append({"rung": rung["rung"], "ok": fit["ok"], "memory": fit.get("memory"),
                      "disk_gib": fit.get("disk_gib"),
                      "error": None if fit["ok"] else design.pick_error(fit.get("error") or [""])})
        if fit["ok"]:
            entry = dict(rung_index=index, rung=rung, fit=fit, tried=tried)
            break
    # Use the same serialized inputs as design_recipes.py -> build_recipes.py.
    record = json.loads(json.dumps({"plan": {k: v for k, v in plan.items() if k != "track"},
                                    "cards": {str(card): entry}}, default=str))
    geometry = event["geometry"]
    rebuilt = builder.build(event_id, event, record,
                            geometry.get("track_cite") or geometry.get("path_cite"), selected_cards=[card])
    document["sources"].update(rebuilt["sources"])
    return rebuilt["recipes"][0]["cards"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=ROOT / "gpuwm/gui/seed/recipes")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    changes = []
    over_budget = []
    for path in sorted(args.recipes.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        kind = "tc" if path.name.startswith("tc-") else "tornado"
        for recipe in document["recipes"]:
            for index, row in enumerate(recipe["cards"]):
                if not row.get("fits"):
                    continue
                where = f"{path.name}/{row['card_gb']}"
                fit = {"domains": [{key: d[key] for key in
                                     ("nx", "ny", "nz", "dx_km", "cu_physics")}
                                    for d in row["domains"]],
                       "physics": row["physics"]["profile"],
                       "restart_interval_s": 3600, "fetch": None, "chain": None}
                output = row["output"]
                saved = (output["history_interval_s"], output["nest_history_interval_s"])
                extra = row["download"]["bytes"] + row["download"]["preparation_bytes"]
                budget = row["disk_budget_gib"] * GIB
                run_seconds = row["length_h"] * 3600

                def priced(root_s, nest_s):
                    projected = design.projection(fit, run_seconds, root_s, nest_s)
                    return projected, projected["total_bytes"] + extra

                intervals = saved
                projected, total = priced(*saved)
                if total > budget:
                    for root_s, nest_s in _ladder(kind, len(fit["domains"])):
                        candidate, candidate_total = priced(root_s, nest_s)
                        if candidate_total <= budget:
                            intervals, projected, total = (root_s, nest_s), candidate, candidate_total
                            break
                    else:
                        replacement = _redesign(document, path.stem, row["card_gb"])
                        recipe["cards"][index] = replacement
                        over_budget.append({"row": where, "projected_gib": round(total / GIB, 1),
                                            "budget_gib": row["disk_budget_gib"],
                                            "replacement_fits": replacement["fits"],
                                            "replacement_rung": replacement.get("rung")})
                        print(f"Re-designed {where}: saved grid exceeded its disk budget at every interval.",
                              flush=True)
                        continue
                price = round(total / GIB, 1)
                sentence = row["disk_basis"]
                sentence = _subn_once(r"the rendered pictures, [0-9.]+ GiB",
                                      f"the rendered pictures, {projected['picture_bytes'] / GIB:.1f} GiB",
                                      sentence, "picture-price", where)
                # The history figure is the same projection's other half: the
                # writer's inventory moves with the engine (2.9.2 carries WRF's
                # radiation accumulations), and the page reads both sentences.
                sentence = _subn_once(r"the history files, [0-9.]+ GiB",
                                      f"the history files, {projected['history_bytes'] / GIB:.1f} GiB",
                                      sentence, "history-price", where)
                if intervals != saved:
                    sentence = _subn_once(
                        r"written every [^;]+? on the outer grid(?: and every [^;]+? on the nests)?",
                        _interval_words(*intervals, len(fit["domains"])),
                        sentence, "output-interval", where)
                    output["history_interval_s"], output["nest_history_interval_s"] = intervals
                changes.append({"file": path.name, "card_gb": row["card_gb"],
                    "before_gib": row["disk_gib"], "after_gib": price,
                    "intervals_before": list(saved), "intervals_after": list(intervals),
                    "picture_bytes": projected["picture_bytes"],
                    "history_bytes": projected["history_bytes"], "total_bytes": total,
                    "over_budget": total > budget})
                row["disk_gib"], row["disk_basis"] = price, sentence
        path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    args.receipt.write_text(json.dumps({"rows": changes, "over_budget": over_budget}, indent=2) + "\n")
    moved = sum(1 for c in changes if c["intervals_before"] != c["intervals_after"])
    print(f"Refreshed {len(changes)} fitted recipe rows with their current projection; "
          f"{moved} took coarser output intervals.")
    for row in over_budget:
        print(f"Re-designed {row['row']}: old grid projected {row['projected_gib']} GiB "
              f"against {row['budget_gib']} GiB; replacement fits: {row['replacement_fits']}.")


if __name__ == "__main__":
    main()
