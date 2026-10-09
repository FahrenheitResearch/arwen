"""adapt6.py JOBS OUT DEADLINE : step 6. Adaptive rows re-walked under the 100 m/s peak rule (lead decision 3).

Each job is one row (settings, dx, crest, ridge_slope) of the candidate's adaptive rows whose walk counted a run held
at peak |w| of 100 m/s or more.  Every run already made is reused, re-read under the rule
(terrain_clock_probe.held_under_peak_rule); the walk is then re-taken exactly as step 4 took it (adaptive_sweep_row,
the two chains 20-30 and 40-60 m/s, both CFL target pairs, 12 h, criterion blowup), so only the rungs the old count
never reached are run, now with the probe's own 100 m/s stop.  Claims are directories; no new run after DEADLINE."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.environ["T4"], "tools"))
import terrain_clock_probe as p

jobs_path, out, deadline = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
out.mkdir(parents=True, exist_ok=True); claims = out / "claims"; claims.mkdir(exist_ok=True)
tree = open(os.path.join(os.environ["T4"], "TREE-COMMIT")).read().strip()
for line in jobs_path.read_text().splitlines():
    if not line.strip() or time.time() > deadline:
        continue
    j = json.loads(line)
    key = j["key"]; final = out / f"{key}.json"
    if final.exists() or not p._claim(claims, key):
        continue
    ev = json.loads(Path(j["evidence"]).read_text())
    row = [r for r in ev["rows"] if r["settings"] == j["settings"] and r["dx_m"] == j["dx"]
           and r["crest_m"] == j["crest"] and r["ridge_slope"] == j["ridge_slope"]]
    assert len(row) == 1, key
    row = row[0]
    done = {}
    for r in row["runs"]:
        done[r["key"]] = {**r, "held_as_recorded": r["held"], "held": p.held_under_peak_rule(r), "reused": True}
    ladder = row["ladder_s_per_km"]
    ridge = p.Ridge(float(j["dx"]), float(j["crest"]), float(j["ridge_slope"]))
    began = time.time(); new = []
    def on_run(res):
        res["reused"] = False; res["card"] = os.environ.get("GPU_MUTEX_CARDS"); new.append(res)
        print(f"{key} new run {res['key']}: {'held' if res['held'] else 'stopped at %s s' % res['stopped_at_s']} "
              f"peak w {res['peak_w']:.1f} {res['wall_s']} s", flush=True)
    chains = []
    for winds in ([20.0, 30.0], [40.0, 50.0, 60.0]):
        r = p.adaptive_sweep_row(ridge, winds, ladder, seconds=43200.0, criterion="blowup", settings=j["settings"],
                                 done=done, on_run=on_run, deadline=deadline, log=lambda *a: None)
        chains.append(r)
    complete = all(c["complete"] for c in chains)
    rec = {**j, "complete": complete, "slope": chains[0]["slope"], "ladder_s_per_km": ladder,
           "winds_m_s": chains[0]["winds_m_s"] + chains[1]["winds_m_s"],
           "adaptive_s_per_km": chains[0]["adaptive_s_per_km"] + chains[1]["adaptive_s_per_km"],
           "adaptive_top_s_per_km": chains[0]["adaptive_top_s_per_km"] + chains[1]["adaptive_top_s_per_km"],
           "was_adaptive_s_per_km": row["adaptive_s_per_km"], "was_adaptive_top_s_per_km": row["adaptive_top_s_per_km"],
           "runs": chains[0]["runs"] + chains[1]["runs"], "new_runs": len(new),
           "host": os.uname().nodename, "card": os.environ.get("GPU_MUTEX_CARDS"), "tree": tree,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(began)), "wall_s": round(time.time() - began, 1)}
    if not complete:
        print(f"{key} incomplete at deadline", flush=True)
        continue
    tmp = final.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=1)); tmp.replace(final)
    print(f"{key} done: {rec['was_adaptive_s_per_km']} -> {rec['adaptive_s_per_km']} ({len(new)} new runs)", flush=True)
