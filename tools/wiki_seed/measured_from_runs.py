"""Write measured.json from the kept run records under runs/<name>/ (row.json, plan.json, summary.json).

Usage: python measured_from_runs.py <runs dir> <event id for each run as name=event ...> > measured.json
A run that exited 0 gives est_minutes: the forecast stage's wall time scaled linearly from the hours it ran
to the row's full length, plus its preparation, not counting the download.
"""
import json, sys
from pathlib import Path

runs = Path(sys.argv[1])
events = dict(a.split("=", 1) for a in sys.argv[2:])
out = {}
for name, event in events.items():
    d = runs / name
    row = json.loads((d / "row.json").read_text(encoding="utf-8"))
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    ran_h = plan["config"]["intent"]["hours"]
    full_h = row["length_h"]
    card = row["card_gb"]
    st = summ.get("stages_s") or {}
    fid = f"computed-fit-{event}-{card}gb"
    extremes = "; ".join(
        f"{dom}: vertical wind to {v['max_w_ms']:g} m/s, 10 m wind to {v['max_wind10_ms']:g} m/s, "
        f"rain to {v['max_rain_mm']:g} mm, " + ("NaN present" if v.get("nan") else "no NaN")
        for dom, v in sorted((summ.get("domains") or {}).items()) if "max_w_ms" in v)
    reached = None
    lasts = [v.get("last_time") for v in (summ.get("domains") or {}).values() if v.get("last_time")]
    if lasts:
        from datetime import datetime
        t0 = datetime.strptime(plan["config"]["intent"]["cycle"], "%Y-%m-%dT%H")
        reached = (datetime.strptime(min(lasts), "%Y-%m-%d_%H:%M:%S") - t0).total_seconds() / 3600
    if summ.get("exit") != 0 and st.get("forecast") and reached:
        f = summ.get("failed") or {}
        fc = st["forecast"] / 60.0
        prep = (st.get("prepare") or 0) / 60.0
        est = round(fc * full_h / reached + prep)
        basis = (f"measured: this row ran on an RTX 5070 Ti (16 GB) on 2026-09-25 to hour {reached:g} of {full_h:g}, "
                 f"where it stopped because the node's disk filled ({f.get('message')}); the forecast took "
                 f"{fc:.1f} minutes to that hour and preparation {prep:.1f}; scaled to {full_h:g} hours, not "
                 "counting the download")
        result = (f"stopped at hour {reached:g} of {ran_h:g}, disk full; forecast {st['forecast']:.0f} s to that hour, "
                  f"preparation {st.get('prepare', 0):.0f} s, download {st.get('fetch', 0):.0f} s by the run itself; "
                  f"{extremes}; estimated {est} minutes for {full_h:g} hours")
    elif summ.get("exit") == 0 and st.get("forecast"):
        fc = st["forecast"] / 60.0
        prep = (st.get("prepare") or 0) / 60.0
        est = round(fc * full_h / ran_h + prep)
        basis = (f"measured: the first {ran_h:g} of {full_h:g} hours of this row ran on an RTX 5070 Ti (16 GB) on "
                 f"2026-09-25; the forecast took {fc:.1f} minutes and preparation {prep:.1f}; scaled to {full_h:g} "
                 "hours, not counting the download")
        result = (f"exit 0; forecast {st['forecast']:.0f} s for {ran_h:g} simulated hours, preparation "
                  f"{st.get('prepare', 0):.0f} s, download {st.get('fetch', 0):.0f} s; {extremes}; estimated "
                  f"{est} minutes for {full_h:g} hours")
    else:
        est = None
        f = summ.get("failed") or {}
        basis = f"unmeasured: a {ran_h:g} hour run of this row on 2026-09-25 stopped in its {f.get('stage')} stage"
        result = f"exit {summ.get('exit')} in the {f.get('stage')} stage: {str(f.get('message'))[:200]}"
    if plan["config"]["intent"].get("nest_history_interval_s") == 3600 and not event.startswith("tc-"):
        note = "; the nests wrote hourly instead of every 15 minutes, to fit the node's disk"
        basis += note
        result += note
    out[f"{event}/{card}"] = {
        "rung": row["rung"], "start": row["start"], "est_minutes": est, "basis": basis,
        "source": {"kind": "computed", "title": f"Measured run of the {card} GB recipe",
                   "method": (f"gpuwm run-plan on this row's plan with hours set to {ran_h:g}, on an RTX 5070 Ti "
                              "(16 GB); forecast stage wall time scaled linearly to the full length; the run's log "
                              f"and summary are kept as runs/{name}/"),
                   "code": "gpuwm/runplan.py: run", "inputs": [fid], "result": result}}
print(json.dumps(out, indent=1))
