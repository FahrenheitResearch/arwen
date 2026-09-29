"""Turn design-result.json into one gpuwm.wiki.v1 recipes document per seed event."""
from __future__ import annotations
import json, math, re, sys
from datetime import datetime, timedelta
from pathlib import Path

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
import design_recipes as design  # noqa: E402

BUILT = "2026-09-25"
OUT = Path(__import__("os").environ.get("RECIPE_OUT", design.WORK / "recipes"))
#: Measured runs, keyed "<event id>/<card gb>". An entry applies only while its "rung" and "start" still
#: name the row, so a redesign never carries a stale time.
_M = S / "measured.json"
MEASURED = json.loads(_M.read_text(encoding="utf-8")) if _M.is_file() else {}

RULE_TC = {
    "kind": "rule", "id": "rule-recipe-best-tc", "title": "Best simulation of a tropical cyclone for each card size",
    "method": (
        "The key time is the storm's first landfall after its peak (the first best-track row, from six hours "
        "before the highest wind on, whose centre is within 25 km of land), or the peak itself when the storm "
        "is already ashore by then or never reaches land. Start 36 hours before the highest wind, so the model "
        "builds its own inner core from the analysis, and stop 18 hours after it or 12 hours after the key "
        "landfall, whichever is later, at most 120 hours. A 1 km grid is the best-track path from 12 hours "
        "before to 12 hours after the key time (IBTrACS positions, linear between rows, every hour) with 240, "
        "180, 130, 90 or 60 km of grid around it, and every rung checks that the centre stays at least 60 km "
        "inside its edge over all of those hours, so the eyewall and not only the centre is on it. The rungs, "
        "best first: 12, 3 and 1 km with the 3 km grid holding the whole track; the same with the 3 km grid "
        "holding the track from 24 hours before the key time to the end and the 12 km grid carrying the "
        "earlier hours; the same pair with a narrower 1 km grid; 12 and 3 km with the 3 km grid holding the "
        "whole track; a 3 km grid that follows the storm inside a fixed 12 km parent, used only while the "
        "storm stays inside that parent; 12 and 3 km with the 3 km grid holding the key hours only; a later "
        "start, 30 then 18 hours before the key time, with 3 km over the rest of the track; a run from 18 "
        "hours before to 6 hours after the key time with a smaller 3 km grid over just those hours; 12 km "
        "alone. Every 12 km grid reaches at least 700 km beyond the 3 km grid (500 for a later start), "
        "measured on the fitted grids. Every cyclone row sets the surface layer's tropical cyclone option "
        "over the sea (isftcflx 1). Each card takes the first rung the engine fits on it with at least 5% of "
        "its memory to spare and whose download, preparation, history, kept checkpoint and pictures fit the "
        "card's disk budget (40, 50, 60, "
        "80 and 100 GiB for 8 to 32 GB) with the nests writing hourly, two-hourly or three-hourly."),
    "code": "tools/wiki_seed/design_recipes.py: tc_plan",
    "inputs": ["rule-recipe-landfall"],
    "retrieved": BUILT,
}
RULE_LANDFALL = {
    "kind": "rule", "id": "rule-recipe-landfall", "title": "The landfall a cyclone recipe is timed around",
    "method": ("The first IBTrACS main-track row at or after six hours before the highest wind whose DIST2LAND "
               "is 25 km or less, or whose USA_RECORD is L. 25 km puts the eyewall ashore when the centre "
               "passes just off a coast."),
    "code": "tools/wiki_seed/landfall_table.py",
    "retrieved": BUILT,
}
RULE_TORNADO = {
    "kind": "rule", "id": "rule-recipe-best-tornado", "title": "Best simulation of a tornado's parent storm for each card size",
    "method": (
        "Start at the 00, 06, 12 or 18 UTC analysis at least 6 hours before the tornado, so storms form in "
        "the model from the morning environment, and run until 3 hours after it ends, at least 12 hours. Put "
        "a 1 km grid over the tornado path and 120 km back along it, where the parent storm forms, with a "
        "margin of 250, 200, 150, 110, 80 or 60 km, inside a 3 km grid that reaches 400 to 180 km beyond it. "
        "From HRRR, itself a 3 km analysis, the 3 km grid is the outer one. From ERA5 (about 31 km), a 12 km "
        "grid reaching 500 km beyond the 3 km one carries the large-scale flow, so the 3 km grid is not driven "
        "straight from 31 km boundaries; every row checks at least 400 km on the fitted grids. When no 1 km "
        "grid fits, the 3 km grid runs alone (inside the 12 km one for ERA5), 1100, 800 or 650 km wide, then "
        "a single 4 km grid. Each card takes the first rung the engine fits on it with at least 5% of its "
        "memory to spare and whose download, preparation, history, kept checkpoint and pictures fit the card's "
        "disk budget (40, 50, 60, 80 "
        "and 100 GiB for 8 to 32 GB) with the nests writing every 15, 20 or 30 minutes."),
    "code": "tools/wiki_seed/design_recipes.py: tornado_plan",
    "retrieved": BUILT,
}
SOURCE_WORDS = {
    "era5": ("ERA5 reanalysis", "one consistent analysis for any date from 1940, read from Google's public "
                                "copy with no key"),
    "hrrr": ("HRRR analysis", "the 3 km analysis with radar data assimilated, archived for dates from late 2014"),
}


def hours_words(h: float) -> str:
    return f"{h:g} hour" + ("" if h == 1 else "s")


def iso(d) -> str:
    return (d if isinstance(d, str) else d.isoformat()).replace(" ", "T")[:16] + "Z"


def utc(d) -> str:
    return (d if isinstance(d, str) else d.isoformat()).replace("T", " ")[:16] + " UTC"


def row_start(rung, plan):
    return datetime.fromisoformat(str(rung.get("start") or plan["start"]).replace(" ", "T"))


def row_hours(rung, plan):
    return rung.get("hours") or plan["hours"]


def grid_words(domains) -> str:
    parts = []
    for d in domains:
        kind = "following" if d.get("following") else ""
        parts.append(f"{d['dx_km']:g} km {kind + ' ' if kind else ''}{d['nx']} by {d['ny']} points "
                     f"({d['width_km']:g} by {d['height_km']:g} km)")
    return "; ".join(parts) + f"; {domains[0]['nz']} levels"


def fitted_domains(fit, rung):
    out = []
    for d in fit["domains"]:
        row = {k: d.get(k) for k in ("grid_id", "parent_id", "dx_km", "nx", "ny", "nz", "width_km", "height_km",
                                     "parent_grid_ratio", "following", "cu_physics")}
        if rung["door"] == "domain":
            row["centre"] = {"lat": rung["box"]["lat"], "lon": rung["box"]["lon"]}
        elif d.get("following"):
            lat, lon = rung["args"]["advisory_position"].split(",")
            row["centre"] = {"lat": float(lat), "lon": float(lon), "moves": "follows the 850 hPa circulation"}
        else:
            lat, lon = rung["args"]["advisory_position"].split(",")
            row["centre"] = {"lat": float(lat), "lon": float(lon)}
        out.append(row)
    return out


# ------------------------------------------------------------------ words per rung

def key_words(plan):
    when = utc(plan["key_time"])
    return f"the landfall at {when}" if plan["key_is"] == "landfall" else f"the peak at {when}"


def cover_words(fit, plan):
    held = fit.get("key_cover_h") if fit else None
    if not held:
        return f"over the hours around {key_words(plan)}"
    return (f"from {held[0]} hours before to {held[1]} hours after {key_words(plan)}, with the best-track "
            f"centre at least {design.EYE_MARGIN_KM:g} km inside its edge the whole time")


def one_km_size(rung):
    return f"{rung['box']['width_km']} by {rung['box']['height_km']} km"


def tc_why(rung, fit, plan):
    name = rung["rung"]
    if name.startswith("static-12-3-1"):
        return (f"The 3 km grid holds the storm for all {plan['hours']} hours, so its eyewall and rain bands are "
                f"resolved from the start, and a {one_km_size(rung)} grid at 1 km holds the eyewall "
                f"{cover_words(fit, plan)}, where the inner core's strength and the damage are decided.")
    if name.startswith("static-12-3key-1"):
        return (f"A {one_km_size(rung)} grid at 1 km holds the eyewall {cover_words(fit, plan)}, inside a 3 km "
                "grid that holds the storm from 24 hours before that to the end; before then the 12 km grid "
                "carries it, which frees the memory the 1 km grid needs.")
    if name == "static-12-3-track":
        return (f"The 3 km grid holds the whole track for all {plan['hours']} hours, so the eyewall is resolved "
                f"from the start through {key_words(plan)} and the storm never crosses a nest edge.")
    if name.startswith("following"):
        return ("A 3 km grid rides on the storm's 850 hPa circulation for the whole run inside a fixed 12 km "
                "parent, which keeps the eyewall resolved at a fraction of the memory of a fixed 3 km grid.")
    if name.startswith("static-12-3-key"):
        return (f"The 3 km grid covers the track from 24 hours before {key_words(plan)} to the end, so that "
                "moment is resolved; before that the storm is carried on the 12 km grid.")
    if name.startswith("key-12-3"):
        return (f"A 3 km grid over just the storm's path from {rung['lead_h']} hours before to {rung['after_h']} "
                f"hours after {key_words(plan)}: on this card the eyewall can be resolved only over those "
                "hours, and a smaller grid over them is worth more than 12 km over the whole storm.")
    if name.startswith("late-12-3"):
        return (f"The run starts {rung['lead_h']} hours before {key_words(plan)} with the storm on a 3 km grid "
                "for every hour, which a card this size can hold only over the shorter track.")
    if name.startswith("late-12"):
        return (f"A single 12 km grid from {rung['lead_h']} hours before {key_words(plan)}; it carries the "
                "storm's track and size over the shorter window, not its eyewall.")
    return "A single 12 km grid over the whole track; it carries the storm's track and size, not its eyewall."


def smallest_three_km(tried):
    """The cheapest 3 km rung this card refused for its memory, as (name, GiB)."""
    priced = []
    for item in tried or []:
        mem = item.get("memory") or {}
        need, budget = mem.get("need_gib"), mem.get("budget_gib")
        if item.get("ok") or not need or not budget or item["rung"].startswith(("single", "late-12-small")):
            continue
        if item["rung"].startswith("late-12") and not item["rung"].startswith("late-12-3"):
            continue
        if need > design.HEADROOM_FRACTION * budget:
            priced.append((item["rung"], need))
    return min(priced, key=lambda pair: pair[1]) if priced else None


def three_km_words(rung):
    if rung.startswith("key-12-3"):
        return "3 km over the storm's path from 18 hours before to 6 hours after the key time"
    if rung.startswith("late-12-3"):
        return "3 km from a later start"
    if rung.startswith("following"):
        return "a 3 km grid that follows the storm"
    if rung.startswith("static-12-3-key"):
        return "3 km over the track from 24 hours before the key time"
    return "3 km over the whole track"


def tc_given_up(rung, fit, plan, ideal, tried=None, card=None):
    name = rung["rung"]
    if name == ideal["rung"]:
        return "Nothing: this is the ideal layout for this storm."
    early = int((datetime.fromisoformat(plan["key_time"]) - timedelta(hours=24)
                 - datetime.fromisoformat(plan["start"])).total_seconds() // 3600)
    early_words = (f"the first {early} hours run at 12 km with a cumulus scheme, so the storm reaches the 3 km "
                   "grid weaker and less organised than it should be" if early > 0 else "")
    lost = []
    if name.startswith("static-12-3-1") or name.startswith("static-12-3key-1"):
        if rung.get("margin_km") != ideal.get("margin_km"):
            lost.append(f"the 1 km grid has {rung['margin_km']:g} km around the key hours' track instead of "
                        f"{ideal['margin_km']:g} km, so the outer eyewall and the inner rain bands sit nearer "
                        "its edge")
        if name.startswith("static-12-3key-1") and early_words:
            lost.append(early_words)
    elif name == "static-12-3-track":
        lost.append(f"no 1 km grid at {key_words(plan)}, so the eyewall is at 3 km and the peak wind is likely "
                    "to be too low")
    elif name.startswith("following"):
        lost.append(f"no 1 km grid at {key_words(plan)}; the 3 km grid is a following nest "
                    f"{fit['domains'][1]['width_km']:g} km wide instead of a fixed grid over the whole track, so "
                    "the outer rain bands leave it")
    elif name.startswith("static-12-3-key"):
        lost.append("no 1 km grid")
        if early_words:
            lost.append(early_words)
    elif name.startswith("key-12-3"):
        peak_before = (plan["key_is"] == "landfall"
                       and datetime.fromisoformat(plan["peak_time"]) < row_start(rung, plan))
        lost.append(f"the run covers only {rung['lead_h']} hours before to {rung['after_h']} hours after "
                    f"{key_words(plan)}, so the storm's earlier life"
                    + (" and its peak are" if peak_before else " is")
                    + f" not simulated and the model has only {rung['lead_h']} hours to build the inner "
                    "core from the 31 km analysis; no 1 km grid")
    elif name.startswith("late"):
        lost.append(f"the run starts {rung['lead_h']} hours before {key_words(plan)} instead of 36 hours "
                    "before the peak, so the model has less time to build the storm's inner core"
                    + (" and the peak itself is not simulated" if plan["key_is"] == "landfall"
                       and datetime.fromisoformat(plan["peak_time"]) < row_start(rung, plan) else ""))
        if not name.startswith("late-12-3"):
            lost.append("no grid finer than 12 km, so the eyewall is not resolved")
    else:
        smallest = smallest_three_km(tried)
        lost.append("no grid finer than 12 km. At 12 km the eyewall, a few tens of km across, is two or three "
                    "cells wide, so the peak wind comes out far too low and a cumulus scheme makes the rain; "
                    "the track and the storm's size are what this layout shows"
                    + (f". The smallest layout with a 3 km grid ({three_km_words(smallest[0])}) needs "
                       f"{smallest[1]:g} GiB of card memory, more than this card holds with 5% to spare"
                       if smallest else ""))
    text = "; ".join(lost) or "a narrower 1 km grid"
    return text[0].upper() + text[1:] + "."


def tornado_why(rung, fit, plan):
    name = rung["rung"]
    ts = utc(plan["tornado_start"])
    ring = outer_ring(fit) or rung.get("synoptic_pad")
    outer = (f" A 12 km grid reaching {ring} km beyond the 3 km grid carries the large-scale flow from "
             "the 31 km reanalysis." if rung.get("synoptic_pad") else "")
    if name.startswith("nest") and rung["finest_km"] == 1.0:
        where = ("the tornado path and the ground 120 km back along it where the parent storm forms"
                 if plan["path_km"] > 5 else "the tornado's point and the ground around it where the parent storm forms")
        return (f"A {rung['inner_w']} by {rung['inner_h']} km grid at 1 km holds {where}, so the supercell and "
                f"its mesocyclone are resolved at {ts}; the 3 km grid around it reaches {rung['outer_pad']} km "
                "further out for the inflow and the boundaries the storm feeds on." + outer)
    return (f"A {rung['finest_km']:g} km grid, {rung['inner_w']} by {rung['inner_h']} km, lets storms form on "
            "their own; it shows where and when supercells form but not their inner structure." + outer)


def outer_ring(fit):
    """km from the 3 km grid's edge to the 12 km grid's, thinnest side, rounded to 10 km, from the fitted grids."""
    rings = (fit or {}).get("rings") or []
    outer = [r["km"] for r in rings if r["parent_id"] == 1]
    return int(round(outer[0] / 10.0) * 10) if outer else None


def tornado_given_up(rung, fit, plan, ideal, tried=None, card=None):
    name = rung["rung"]
    if name == ideal["rung"]:
        return ("Nothing within this layout. A grid finer than 1 km is not used on any card: a fixed box cannot be "
                "sure to hold the simulated storm, which usually forms 20 to 50 km from the real one, and it "
                "multiplies the run time.")
    if name.startswith("nest") and rung["finest_km"] == 1.0:
        return (f"The 1 km grid is {rung['inner_w']} by {rung['inner_h']} km instead of {ideal['inner_w']} by "
                f"{ideal['inner_h']} km and the 3 km grid reaches {rung['outer_pad']} km beyond it instead of "
                f"{ideal['outer_pad']} km, so a simulated storm that forms further from the real one can leave the "
                "1 km grid.")
    extra = "" if rung.get("synoptic_pad") or plan["source"] == "hrrr" else (
        " The grid is driven straight from the 31 km reanalysis, with no 12 km grid between.")
    return (f"No 1 km grid: the storm runs at {rung['finest_km']:g} km, which shows the supercell but not its "
            f"rotation; the grid is {rung['inner_w']} by {rung['inner_h']} km." + extra)


def output_words(intervals, kind, card, nests=True):
    """What the row's output intervals give up against the densest ones, or an empty string."""
    root_s, nest_s = intervals
    best_root, best_nest = (design.TC_INTERVALS if kind == "tc" else design.TORNADO_INTERVALS)[0]
    words = []
    if nests and nest_s > best_nest:
        words.append(f"the nests write every {minutes(nest_s)} instead of every {minutes(best_nest)}")
    if root_s > best_root:
        words.append(f"the {'outer' if nests else 'only'} grid writes every {minutes(root_s)} instead of every "
                     f"{minutes(best_root)}")
    if not words:
        return ""
    return (" and ".join(words) + f", to keep the run within this card's {design.DISK_BUDGET_GIB[card]} GiB "
            "disk budget")


def disk_words(fit, doms, card) -> str:
    """What the row's disk figure counts, part by part, as the engine projects it."""
    root_s, nest_s = fit["intervals"]
    p = design.projection(fit, fit["run_seconds"], root_s, nest_s)
    gib = lambda value: f"{value / 1024 ** 3:.1f} GiB"
    download = p["download"]
    source = str(download.get("source") or "").upper()
    parts = [f"the download, {gib(p['download_bytes'])} of {source} files for {download.get('leads')} "
             "forcing times" if p["download_bytes"] else "no download",
             f"the preparation, {gib(p['preparation_bytes'])}",
             f"the history files, {gib(p['history_bytes'])}, written every {minutes(root_s)} on the outer grid"
             + (f" and every {minutes(nest_s)} on the nests" if len(doms) > 1 else ""),
             f"the one checkpoint the run keeps (two while a new one is written), {gib(p['checkpoint_bytes'])}",
             f"the rendered pictures, {gib(p['picture_bytes'])}"]
    words = "; ".join(parts)
    if p["unpriced"]:
        words += "; not counted, since no size is measured for it: " + ", ".join(p["unpriced"])
    return (f"{words}; as the engine projects them (gpuwm/disk_budget.py, gpuwm/download_budget.py) from "
            f"sizes measured on real runs; within this card's {design.DISK_BUDGET_GIB[int(card)]} GiB disk budget")


def download_record(fit) -> dict:
    """The row's download as the engine prices it, for the time estimate."""
    root_s, nest_s = fit["intervals"]
    p = design.projection(fit, fit["run_seconds"], root_s, nest_s)
    d = p["download"]
    return {"bytes": d.get("bytes"), "transfer_bytes": d.get("transfer_bytes"), "objects": d.get("objects"),
            "leads": d.get("leads"), "source": d.get("source"), "mode": d.get("mode"),
            "preparation_bytes": p["preparation_bytes"], "chain": fit.get("chain")}


def minutes(seconds):
    if seconds == 3600:
        return "hour"
    return f"{seconds // 60:g} minutes" if seconds < 3600 else hours_words(seconds / 3600)


def refusal_words(tried):
    mem = tried.get("memory") or {}
    err = tried.get("error") or ""
    if ("storm moves" in err or "headroom" in err or "cumulus is on" in err or "disk budget" in err
            or "reaches only" in err or "holds the centre" in err):
        return err
    if mem and mem.get("need_gib") and mem.get("budget_gib") and mem["need_gib"] > mem["budget_gib"]:
        return f"needs {mem['need_gib']:g} GiB; this card's budget is {mem['budget_gib']:g} GiB"
    return re.sub(r"[A-Za-z]:[\\/][^ ,;]+|/home/[^ ,;]+", "<path>", err)[:300]


def key_hours(ev, plan):
    if ev["type"] == "tropical-cyclone":
        key = datetime.fromisoformat(plan["key_time"])
        what = "first landfall after the highest wind" if plan["key_is"] == "landfall" else "highest wind"
        return {"start": iso(max(key - timedelta(hours=design.KEY_BEFORE_H), datetime.fromisoformat(plan["start"]))),
                "end": iso(min(key + timedelta(hours=design.KEY_AFTER_H), datetime.fromisoformat(plan["end"]))),
                "key_time": iso(key), "key_is": plan["key_is"],
                "text": f"From 12 hours before to 12 hours after the {what} in the best track."}
    return {"start": iso(plan["tornado_start"]), "end": iso(plan["tornado_end"]),
            "text": "The tornado's own hours in the SPC table."}


def given_up_with_output(text, output):
    if not output:
        return text
    if text.startswith("Nothing"):
        return output[0].upper() + output[1:] + "; the layout itself is the ideal one."
    return text.rstrip(".") + "; " + output + "."


def legacy_recipe(src, plan, rung, doms, card):
    """The New forecast fields, copied faithfully: the box, the outer spacing, and the nests it needs."""
    row = {"source": src, "cycle": design.cyc(row_start(rung, plan)), "hours": row_hours(rung, plan),
           "card": f"{card}gb"}
    if rung["door"] == "domain":
        intent = rung["intent"]
        row.update(lat=rung["box"]["lat"], lon=rung["box"]["lon"], width_km=rung["box"]["width_km"],
                   height_km=rung["box"]["height_km"], dx_km=float(intent["root_dx_km"]))
        if intent.get("chain"):
            row.update(chain=intent["chain"], buffer_km=intent["buffer_km"])
        if src == "era5":
            row["era5_provider"] = "arco"
    else:
        lat, lon = rung["args"]["advisory_position"].split(",")
        row.update(lat=float(lat), lon=float(lon), width_km=doms[0]["width_km"], height_km=doms[0]["height_km"],
                   dx_km=doms[0]["dx_km"], following=True, cyclone_setup=dict(rung["args"]))
    return row


def build(event_id, ev, rec, cite_track):
    plan = rec["plan"]
    tc = ev["type"] == "tropical-cyclone"
    rule = RULE_TC if tc else RULE_TORNADO
    ideal = plan["rungs"][0]
    sources = {rule["id"]: {k: v for k, v in rule.items() if k != "id"}}
    top_cite = [rule["id"], cite_track]
    if tc:
        sources[RULE_LANDFALL["id"]] = {k: v for k, v in RULE_LANDFALL.items() if k != "id"}
        top_cite.append(RULE_LANDFALL["id"])
        if plan.get("landfall"):
            top_cite.append(plan["landfall"]["cite"])
            sources[plan["landfall"]["cite"]] = design.LANDFALL[event_id]["landfall"]["source"]
    cards = []
    for card in map(str, design.CARDS):
        entry = rec["cards"][card]
        if entry.get("rung_index") is None:
            cards.append({"card_gb": int(card), "fits": False,
                          "why": "No rung of this recipe fits this card.",
                          "tried": entry["tried"]})
            continue
        rung, fit = entry["rung"], entry["fit"]
        src = plan["source"]
        profile, physics_why = design.PHYSICS[src]
        fit_id = f"computed-fit-{event_id}-{card}gb"
        mem = fit["memory"]
        door = {"door": rung["door"]}
        if rung["door"] == "domain":
            intent = design.intent_for(rung, plan, fit["intervals"])
            door["intent"] = dict(intent, card=f"{card}gb")
            door["box"] = rung["box"]
            method = ("gpuwm run-plan --resolve on a plan whose config.intent is this recipe's intent with the box "
                      f"as its polygon, sized for a declared {card} GB card, no GPU used")
            code = "gpuwm/runplan.py: resolve"
        else:
            args = {key: value for key, value in fit["args"].items() if value is not None}
            args["card"] = f"{card}gb"
            if rung.get("grow"):
                args["nest_budget_gib"] = int(card)
            door["args"] = args
            method = (f"gpuwm cyclone-setup with these arguments, sized for a declared {card} GB card, no GPU used; "
                      f"the storm moves {fit['moves_km'][0]} km east-west and {fit['moves_km'][1]} km north-south "
                      f"and the nest has room for {fit['room_km'][0]} and {fit['room_km'][1]} km")
            code = "gpuwm/cyclone_setup.py: plan_cyclone"
        doms = fitted_domains(fit, rung)
        cumulus = design.cumulus_words(doms)
        sources[fit_id] = {"kind": "computed", "title": f"Engine fit of the {card} GB recipe", "method": method,
                           "code": code, "inputs": top_cite,
                           "result": f"{grid_words(fit['domains'])}; priced at {mem['need_gib']:g} GiB of a "
                                     f"{mem['budget_gib']:g} GiB budget; cumulus per grid, outer to inner: "
                                     + ", ".join(str(d.get('cu_physics')) for d in doms)}
        measured = MEASURED.get(f"{event_id}/{card}")
        if measured and (measured.get("rung") != rung["rung"] or measured.get("start") != iso(row_start(rung, plan))):
            measured = None
        if measured:
            est, basis = measured["est_minutes"], measured["basis"]
            sources[f"computed-run-{event_id}-{card}gb"] = measured["source"]
        else:
            est, basis = None, "unmeasured: the engine gives no time estimate for a plan"
        finest = min(d["dx_km"] for d in doms)
        row = {
            "card_gb": int(card),
            "rung": rung["rung"],
            "fits": True,
            "domains": doms,
            "start": iso(row_start(rung, plan)),
            "length_h": row_hours(rung, plan),
            "source": src,
            "source_why": f"{SOURCE_WORDS[src][0]}: {SOURCE_WORDS[src][1]}.",
            "physics": {"profile": fit.get("physics", "").split(":")[0].strip() or profile
                        if rung["door"] == "domain" else (fit.get("physics") or profile),
                        "why": f"{physics_why} {cumulus}" + (f" {design.TC_ISFTCFLX_WHY}" if tc else "")},
            "fitted_grid": grid_words(fit["domains"]),
            "memory": {"need_gib": mem["need_gib"], "budget_gib": mem["budget_gib"]},
            "disk_gib": fit["disk_gib"],
            "disk_budget_gib": design.DISK_BUDGET_GIB[int(card)],
            "output": {"history_interval_s": fit["intervals"][0], "nest_history_interval_s": fit["intervals"][1],
                       "keep_checkpoints": design.KEEP_CHECKPOINTS},
            "disk_basis": disk_words(fit, doms, card),
            "download": download_record(fit),
            "est_minutes": est,
            "est_basis": basis,
            "why": (tc_why if tc else tornado_why)(rung, fit, plan),
            "what_is_given_up": given_up_with_output(
                (tc_given_up if tc else tornado_given_up)(rung, fit, plan, ideal, entry["tried"], int(card)),
                output_words(fit["intervals"], "tc" if tc else "tornado", int(card), len(doms) > 1)),
            "finest_km": finest,
            **({"key_cover_h": fit["key_cover_h"]} if fit.get("key_cover_h") else {}),
            **({"rings_km": fit["rings"]} if fit.get("rings") else {}),
            **door,
            # the New forecast fields, a faithful copy of this row (see RECIPES-FORMAT.md)
            "recipe": legacy_recipe(src, plan, rung, doms, card),
            "cite": [fit_id] + top_cite + ([f"computed-run-{event_id}-{card}gb"] if measured else []),
        }
        rejected = [t for t in entry["tried"] if not t["ok"]]
        if rejected:
            row["better_rungs_refused"] = [
                {"rung": t["rung"], "need_gib": (t.get("memory") or {}).get("need_gib"),
                 "reason": refusal_words(t)} for t in rejected]
        cards.append(row)
    recipe = {
        "event": event_id,
        "title": ev["title"],
        "type": ev["type"],
        "key_hours": key_hours(ev, plan),
        "ideal": {"rung": ideal["rung"], "finest_km": ideal["finest_km"],
                  "text": (tc_why if tc else tornado_why)(ideal, None, plan)},
        "disk_budget_gib": dict((str(k), v) for k, v in design.DISK_BUDGET_GIB.items()),
        "start": iso(plan["start"]), "length_h": plan["hours"], "source": plan["source"],
        "cards": cards,
        "cite": top_cite,
    }
    return {"schema": "gpuwm.wiki.v1", "origin": "recipes", "built": BUILT, "sources": sources,
            "recipes": [recipe]}


def main():
    seed = json.load(open(Path(design.SEED) / "gpuwm/gui/seed/wiki-seed.json", encoding="utf-8"))
    result = json.loads((design.WORK / "design-result.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    for ev in seed["events"]:
        if ev["id"] not in result:
            continue
        g = ev["geometry"]
        cite_track = g.get("track_cite") or g.get("path_cite")
        doc = build(ev["id"], ev, result[ev["id"]], cite_track)
        (OUT / f"{ev['id']}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", sum(ev["id"] in result for ev in seed["events"]))


if __name__ == "__main__":
    main()
