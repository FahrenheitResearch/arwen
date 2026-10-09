"""crosscheck.py ROWS6_DIR OLD_ENGINE_MAP ROWS4_DIR : the W1 direct blow-up sweep (rows6) against (a) the step-1/step-2
evidence-assembled candidate (blow-up verdicts inferred over the step-1 map-criterion runs, the blow-up re-checks and the
'steady' sweep) and (b) box S's direct blow-up rows (rows4, cut short when S went down). Per row and wind: equal, or the
two entries side by side. Only cells both measured are compared (top not None on both)."""
import json, sys, glob, os
rows6, old_map, rows4 = sys.argv[1:4]
def key(r): return (r["settings"], float(r["dx_m"]), float(r["crest_m"]), round(float(r["ridge_slope"]), 6), int(r["sound_steps"]))
new = {}
for f in glob.glob(os.path.join(rows6, "*-blowup.json")):
    r = json.load(open(f)); new[key(r)] = r
def compare(name, others):
    same = diff = 0; out = []
    for k, o in others.items():
        n = new.get(k)
        if n is None: continue
        for i, w in enumerate(n["winds_m_s"]):
            a, at = n["stable_s_per_km"][i], n["top_s_per_km"][i]
            b, bt = o["stable_s_per_km"][i], o["top_s_per_km"][i]
            if at is None or bt is None: continue
            # Compare where both walks started at the same rung, else compare the verdict on the shorter top.
            if a == b and at == bt: same += 1; continue
            diff += 1; out.append({"row": "-".join(map(str, k)), "wind": w, "w1": [a, at], name: [b, bt]})
    return {"cells_same": same, "cells_differ": diff, "differ": out}
old = {key(r): r for r in json.load(open(old_map))["rows"]}
r4 = {}
for f in glob.glob(os.path.join(rows4, "*-blowup.json")):
    r = json.load(open(f)); r4[key(r)] = r
res = {"rows6": len(new), "vs_evidence_candidate": compare("evidence", old), "vs_box_S_rows4": compare("S_rows4", r4)}
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "differ"}) for k, v in res.items()}, indent=1))
json.dump(res, open(sys.argv[4], "w"), indent=1)
