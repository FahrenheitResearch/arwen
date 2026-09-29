"""Give every recipe row a run time: the measured one when the row ran, otherwise one scaled from a measured run.

Usage: python estimate_times.py [recipes dir]   (default: gpuwm/gui/seed/recipes of this tree)

The forecast's cost is taken as proportional to grid points times time steps: each grid's nx * ny * nz
divided by its spacing in km (the time step shrinks with the spacing), summed over the grids, times the
hours.  The measured runs in measured.json give the seconds per unit of that work for each kind of event
(a cyclone's strong winds and deep convection cost about twice a tornado day's per point); a row without a
measurement gets that rate times its own work.  A kind with no measured run keeps no estimate.  Rows whose
time was measured keep it.  That is ``est_minutes``, the forecast.

``est_total_minutes`` is the wait from the press to the last picture: the download, the preparation, the
forecast and the pictures.  A run that drew its pictures gives the seconds per picture frame per grid point
(each grid's history files times its east by north points); a row gets that rate times its own frames and
points.  The download and preparation are the row's own measured ones when it ran.  Otherwise they are the
row's own download, as the engine prices it (gpuwm/download_budget.py, the ``download`` record the builder
writes on every row), over the throughput measured on real downloads of the same source and transport
(runs/lead-in/runs.json), and its preparation at the rate measured on real preparations of the same chain,
per downloaded byte or per grid cell and forcing time as that file declares for the chain.  A mean over
every measured run gave an 18.9 GB HRRR download and its preparation the two minutes an ERA5 run takes, and
the row promised 15 minutes where the lead-in alone was 707 seconds.  The pictures were drawn after the
forecast on the measured run, so the total is the wait when they are; the forecast part is the floor when
they are drawn during it.  The rates are the machine the runs were measured on; the basis says so.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

S = Path(__file__).resolve().parent
RECIPES = Path(sys.argv[1]) if len(sys.argv) > 1 else S.parents[1] / "gpuwm" / "gui" / "seed" / "recipes"
MEASURED = json.loads((S / "measured.json").read_text(encoding="utf-8"))
RUNS = S / "runs"
RULE_ID = "rule-recipe-time-estimate"
#: Measured download and preparation wall times, with the request each run downloaded.
LEAD_IN = json.loads((RUNS / "lead-in" / "runs.json").read_text(encoding="utf-8"))


def _download_budget():
    sys.path.insert(0, str(S.parents[1]))
    from gpuwm import download_budget
    return download_budget


def _preparation_work(chain: str, *, download_bytes: float, cells: float, leads: float) -> float:
    """What one chain's preparation time scales with, as runs/lead-in/runs.json declares from its runs."""
    driver = LEAD_IN["preparation_scales_with"][chain]["driver"]
    if driver == "download_bytes":
        return float(download_bytes)
    if driver == "cell_times":
        return float(cells) * float(leads)
    raise SystemExit(f"runs/lead-in: unknown preparation driver {driver!r} for {chain}")


def lead_in_rates() -> tuple[dict, dict]:
    """Measured download throughput per (source, transport) and preparation seconds per unit of work per chain.

    Each is a pooled ratio over the measured runs: their summed wall seconds over their summed bytes (or cells
    times forcing times), the bytes priced by the engine from each run's own request, so a row priced the same
    way scales exactly.
    """
    budget = _download_budget()
    fetch: dict[tuple, list] = {}
    prep: dict[str, list] = {}
    for run in LEAD_IN["runs"]:
        priced = budget.download_estimate(run["request"])
        if priced["bytes"] is None:
            raise SystemExit(f"runs/lead-in: {run['run']}'s download is unpriced: {priced['basis']}")
        a = fetch.setdefault((priced["source"], priced["mode"]), [0.0, 0.0, []])
        a[0] += float(run["fetch_s"])
        a[1] += float(priced["transfer_bytes"])
        a[2].append(run["run"])
        b = prep.setdefault(run["chain"], [0.0, 0.0, []])
        b[0] += float(run["prepare_s"])
        b[1] += _preparation_work(run["chain"], download_bytes=priced["bytes"], cells=run["cells"],
                                  leads=priced["leads"])
        b[2].append(run["run"])
    return fetch, prep


def lead_in_seconds(row: dict, rates: tuple[dict, dict]) -> tuple[float, str]:
    """The seconds from the press to the first forecast step, and how they were reached, for one row."""
    fetch, prep = rates
    download = row.get("download") or {}
    key = (download.get("source"), download.get("mode"))
    if download.get("bytes") is None or key not in fetch or download.get("chain") not in prep:
        raise SystemExit(f"{row.get('rung')}: no measured download or preparation rate for {key} on "
                         f"{download.get('chain')}; add a measured run to runs/lead-in/runs.json")
    seconds, moved, runs = fetch[key]
    fetch_s = float(download["transfer_bytes"]) * seconds / moved
    pseconds, pwork, pruns = prep[download["chain"]]
    cells = sum(d["nx"] * d["ny"] * d["nz"] for d in row.get("domains") or [])
    prepare_s = _preparation_work(download["chain"], download_bytes=download["bytes"], cells=cells,
                                  leads=download.get("leads") or 0) * pseconds / pwork
    words = (f"the download ({download['bytes'] / 1e9:.1f} GB of {str(key[0]).upper()} files, about "
             f"{fetch_s / 60.0:.0f} minutes at the {moved / seconds / 1e6:.0f} MB/s measured on {len(runs)} "
             f"{'download' if len(runs) == 1 else 'downloads'} of that source on the same machine) and the "
             f"preparation (about {prepare_s / 60.0:.0f} minutes, at the rate measured on {len(pruns)} "
             f"{'preparation' if len(pruns) == 1 else 'preparations'} of the same kind)")
    return fetch_s + prepare_s, words


def work(row: dict) -> float:
    domains = row.get("domains")
    if domains is None:
        # A kept run record carries its grids as words: "12 km 120 by 118 points (...); ...; 49 levels".
        import re
        levels = int(re.search(r"(\d+) levels", row["fitted_grid"]).group(1))
        domains = [{"dx_km": float(dx), "nx": int(nx), "ny": int(ny), "nz": levels} for dx, nx, ny in
                   re.findall(r"([\d.]+) km (?:following )?(\d+) by (\d+) points", row["fitted_grid"])]
    return sum(d["nx"] * d["ny"] * d["nz"] / float(d["dx_km"]) for d in domains)


def forecast_hours(summary: dict, row: dict) -> float:
    """Simulated hours the measured forecast covered: its last history time on the outer grid, from the start."""
    from datetime import datetime

    start = datetime.strptime(row["start"].rstrip("Z"), "%Y-%m-%dT%H:%M")
    last = summary["domains"]["d01"]["last_time"]
    return (datetime.strptime(last, "%Y-%m-%d_%H:%M:%S") - start).total_seconds() / 3600.0


def frame_points(row: dict) -> float:
    """Picture frames times grid points: each grid's history files over the row's hours times its nx * ny."""
    out = row.get("output") or {}
    hours = float(row["length_h"])
    total = 0.0
    for i, d in enumerate(row["domains"]):
        step = out.get("history_interval_s" if i == 0 else "nest_history_interval_s") or 3600
        total += (int(hours * 3600 // step) + 1) * d["nx"] * d["ny"]
    return total


def _summary_of(entry: dict) -> tuple[str, dict]:
    run = entry["source"]["method"].split("runs/")[-1].strip("/ ")
    return run, json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))


def main() -> int:
    docs = {p: json.loads(p.read_text(encoding="utf-8")) for p in sorted(RECIPES.glob("*.json"))}
    rows = {}
    for doc in docs.values():
        for recipe in doc.get("recipes") or []:
            for row in recipe.get("cards") or []:
                rows[f"{recipe['event']}/{row['card_gb']}"] = (recipe, row)
    # Seconds of forecast per unit of work, per kind of event, from every kept run.
    rates: dict[str, list[tuple[float, str, str]]] = {}
    # Seconds of pictures per frame per grid point from every kept run; download and preparation rates from
    # the measured lead-in runs.
    picture_rates: list[tuple[float, str]] = []
    lead_in = lead_in_rates()
    for key, entry in MEASURED.items():
        recipe, _ = rows[key]
        run, summary = _summary_of(entry)
        stages = summary["stages_s"]
        if stages.get("render"):
            drawn = sum(d["files"] * d["nx"] * d["ny"] for d in summary["domains"].values())
            picture_rates.append((stages["render"] / drawn, run))
        # The grids that RAN, kept beside the run: the row published now may be a later design.
        row = json.loads((RUNS / run / "row.json").read_text(encoding="utf-8"))
        hours = forecast_hours(summary, row)
        rate = summary["stages_s"]["forecast"] / (work(row) * hours)
        rates.setdefault(recipe["type"], []).append((rate, f"computed-run-{recipe['event']}-{row['card_gb']}gb", run))
    rule = {
        "kind": "rule", "id": RULE_ID, "title": "Run time of a recipe that has not been run",
        "method": (
            "A forecast's time grows with its grid points and its time steps: each grid's points (east by north "
            "by levels) divided by its spacing in km, summed over the grids, times the hours. The measured runs "
            "give the seconds per unit of that for each kind of storm, on the card they ran on; a recipe that has "
            "not run gets that rate times its own grids and hours. The time to all pictures adds the download and "
            "preparation and the pictures. The download is this layout's own, as the engine prices it, over the "
            "throughput measured on real downloads of the same source; the preparation is at the rate measured "
            "on real preparations of the same kind, per byte downloaded for HRRR, whose preparation decodes every "
            "file it downloads, and per grid cell and forcing time for ERA5; the pictures are the measured "
            "seconds per picture frame per grid point times this layout's frames (each grid's history files) "
            "and points."),
        "code": "tools/wiki_seed/estimate_times.py",
        "inputs": sorted({cite for items in rates.values() for _, cite, _ in items}),
        "retrieved": "2026-09-25",
    }
    for path, doc in docs.items():
        changed = False
        for recipe in doc.get("recipes") or []:
            items = rates.get(recipe["type"])
            for row in recipe.get("cards") or []:
                # A row the builder gave its measured run's time (est_minutes set, no est_kind yet) keeps it:
                # on a fresh build this loop used to overwrite every measured row with an estimate.
                measured = row.get("est_kind") == "measured" or (
                    row.get("est_minutes") is not None and not row.get("est_kind"))
                if not row.get("fits") or measured or not items:
                    continue
                rate = sum(r for r, _, _ in items) / len(items)
                minutes = rate * work(row) * float(row["length_h"]) / 60.0
                row["est_minutes"] = int(max(5, round(minutes / 5.0) * 5))
                row["est_kind"] = "estimated"
                row["est_forecast_s"] = minutes * 60.0
                row["est_basis"] = (
                    f"estimated from the speed of {len(items)} measured "
                    f"{'run' if len(items) == 1 else 'runs'} of this kind of storm on an RTX 5070 Ti (16 GB), "
                    "scaled by this layout's grid points, time steps and hours; a faster or slower card changes it")
                row["cite"] = [c for c in row.get("cite") or [] if c != RULE_ID] + [RULE_ID]
                changed = True
            for row in recipe.get("cards") or []:
                if row.get("est_minutes") is not None and not row.get("est_kind"):
                    row["est_kind"] = "measured"
                    changed = True
            for row in recipe.get("cards") or []:
                if row.get("fits") and row.get("est_minutes") is not None and picture_rates:
                    _total(recipe, row, picture_rates, lead_in)
                    changed = True
        if changed:
            sources = doc.setdefault("sources", {})
            sources[RULE_ID] = rule
            path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("pictures", [f"{r * 1e6:.3f}e-6 s ({run})" for r, run in picture_rates])
    for key, (seconds, moved, runs) in sorted(lead_in[0].items()):
        print("download", key, f"{moved / seconds / 1e6:.1f} MB/s over {len(runs)} runs")
    for chain, (seconds, units, runs) in sorted(lead_in[1].items()):
        driver = LEAD_IN["preparation_scales_with"][chain]["driver"]
        print("preparation", chain, f"{seconds / units:.3e} s per unit of {driver} over {len(runs)} runs")
    for kind, items in sorted(rates.items()):
        print(kind, [f"{r * 1e6:.2f}e-6 s ({run})" for r, _, run in items])
    return 0


def _total(recipe: dict, row: dict, picture_rates: list, lead_in: tuple[dict, dict]) -> None:
    """Set the row's minutes from the press to the last picture, and say how in its basis."""
    entry = MEASURED.get(f"{recipe['event']}/{row['card_gb']}")
    if row["est_kind"] == "measured" and entry is not None:
        basis = entry["basis"]
        run, summary = _summary_of(entry)
        stages = summary["stages_s"]
        ran = json.loads((RUNS / run / "row.json").read_text(encoding="utf-8"))
        if stages.get("render") and forecast_hours(summary, ran) >= float(row["length_h"]):
            # The whole row ran and drew its pictures: the wait is what it took.
            row["est_total_minutes"] = int(round(sum(stages.values()) / 60.0))
            row["est_basis"] = basis
            return
        forecast_s = row["est_minutes"] * 60.0
        before_s = stages.get("fetch", 0.0)
        before_words = f"the download as measured ({before_s:.0f} s)"
        rounding = 1
    else:
        basis = row["est_basis"].split("; the time to all pictures adds")[0]
        forecast_s = row.pop("est_forecast_s", None) or row["est_minutes"] * 60.0
        before_s, before_words = lead_in_seconds(row, lead_in)
        rounding = 5
    rate = sum(r for r, _ in picture_rates) / len(picture_rates)
    pictures_s = rate * frame_points(row)
    total = (forecast_s + before_s + pictures_s) / 60.0
    row["est_total_minutes"] = int(max(5, round(total / rounding) * rounding))
    row["est_basis"] = (
        f"{basis}; the time to all pictures adds {before_words} and the pictures, about "
        f"{max(1, round(pictures_s / 60.0))} minutes, scaled from the measured run that drew its pictures by "
        "this layout's picture frames and grid points")


if __name__ == "__main__":
    raise SystemExit(main())
