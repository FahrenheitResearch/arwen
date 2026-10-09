"""recheck4.py CELLS.json OUT.json : inside the mutex, one card. The deciding step-4 adaptive runs (the LA and SF cells)
run again with the same probe and the same arguments adaptive_sweep_row used; every recorded number must match."""
import importlib.util, json, sys
# the step-4 probe file itself (the tree's tools package would otherwise win the import)
spec = importlib.util.spec_from_file_location("probe4", "/work/tclock/s4/tools/terrain_clock_probe.py")
p = importlib.util.module_from_spec(spec); sys.modules["probe4"] = p; spec.loader.exec_module(p)
cells = json.load(open(sys.argv[1])); res = []
for c in cells:
    ridge = p.Ridge(c["dx"], c["crest"], c["ridge_slope"]); km = c["dx"] / 1000.0
    upper = round(c["per_km"] * km, 2)
    r = p.run_adaptive_cell(ridge, wind=c["wind"], max_step=upper, start_step=min(round(5.0 * km, 2), upper),
                            seconds=43200.0, target_cfl=c["pair"][0], target_hcfl=c["pair"][1],
                            increase_pct=p.ADAPTIVE_INCREASE_PCT, etac=p.geometry(ridge)["etac_exact"],
                            criterion="blowup", settings=c["settings"])
    same = {k: r[k] == v for k, v in c["expect"].items()}
    res.append({**c, "now": {k: r[k] for k in c["expect"]}, "same": same, "all_same": all(same.values())})
    print(c["settings"], c["dx"], c["crest"], c["ridge_slope"], c["wind"], c["per_km"], c["pair"], "all_same", all(same.values()), flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
