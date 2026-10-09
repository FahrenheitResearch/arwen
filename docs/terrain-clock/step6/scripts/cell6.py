"""cell6.py JOBS OUT DEADLINE : step 6 probe cells (the step-5 worker, run on the step-6 probe: blow-up test with the 100 m/s peak rule) (terrain-clock local-face lane), 12 h each under the blow-up held test.

kind "fixed": one map cell (settings, ridge, substeps, wind), tools/terrain_clock_probe.py run_cell, criterion
"blowup", Courant recorded, from the job's first rung down its rungs until one holds (long12.py's walk); the entry is
the first rung that holds, None where none does.

kind "adaptive": one run_adaptive_cell at the job's max_time_step (s/km x dx) and CFL target pair, criterion
"blowup", first step min(5 s/km, max) as the adaptive sweep; the record keeps steps_at_max, so a held run says how
long the controller actually ran at that maximum.

Claims are directories (dead pid taken over); no new job starts after DEADLINE."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.environ["T4"], "tools"))
from terrain_clock_probe import (Ridge, geometry, run_cell, run_adaptive_cell, _claim, ADAPTIVE_INCREASE_PCT)

jobs_path, out, deadline = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
SECONDS = 43200.0
out.mkdir(parents=True, exist_ok=True); claims = out / "claims"; claims.mkdir(exist_ok=True)
tree = open(os.path.join(os.environ["T4"], "TREE-COMMIT")).read().strip()
for line in jobs_path.read_text().splitlines():
    if not line.strip() or time.time() > deadline:
        continue
    j = json.loads(line)
    key = j["key"]
    final = out / f"{key}.json"
    if final.exists() or not _claim(claims, key):
        continue
    ridge = Ridge(float(j["dx"]), float(j["crest"]), float(j["ridge_slope"]))
    shape = geometry(ridge)
    began = time.time()
    if j["kind"] == "fixed":
        runs = []; entry = None
        for per_km in j["rungs"]:
            r = run_cell(ridge, wind=float(j["wind"]), per_km=float(per_km), sound_steps=int(j["sound_steps"]),
                         seconds=SECONDS, etac=shape["etac_exact"], settings=j["settings"], courant=True,
                         criterion="blowup")
            r = {"per_km": float(per_km), **r}
            runs.append(r)
            print(f"{key} {per_km} s/km: {'held' if r['held'] else 'stopped at %s s' % r['stopped_at_s']} peak w "
                  f"{r['peak_w']:.2f} bound {r['bound']:.1f} vc {r['peak_wrf_vertical_courant']:.2f} {r['wall_s']} s",
                  flush=True)
            if r["held"]:
                entry = float(per_km)
                break
        rec = {**j, "slope": shape["slope"], "seconds": SECONDS, "entry_12h": entry,
               "top_12h": float(j["rungs"][0]), "runs": runs}
    else:
        km = ridge.dx / 1000.0
        upper = round(float(j["per_km"]) * km, 2)
        r = run_adaptive_cell(ridge, wind=float(j["wind"]), max_step=upper,
                              start_step=min(round(5.0 * km, 2), upper), seconds=SECONDS,
                              target_cfl=float(j["pair"][0]), target_hcfl=float(j["pair"][1]),
                              increase_pct=ADAPTIVE_INCREASE_PCT, etac=shape["etac_exact"], criterion="blowup",
                              settings=j["settings"])
        r["at_max_fraction"] = (r["steps_at_max"] / r["steps"]) if r["steps"] else None
        print(f"{key}: {'held' if r['held'] else 'stopped at %s s' % r['stopped_at_s']} peak w {r['peak_w']:.2f} "
              f"bound {r['bound']:.1f} at max {r['steps_at_max']}/{r['steps']} mean {r['mean_step_s']} "
              f"{r['wall_s']} s", flush=True)
        rec = {**j, "slope": shape["slope"], "seconds": SECONDS, "run": r}
    rec.update({"host": os.uname().nodename, "card": os.environ.get("GPU_MUTEX_CARDS"), "tree": tree,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(began)),
                "wall_s": round(time.time() - began, 1)})
    tmp = final.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(final)
