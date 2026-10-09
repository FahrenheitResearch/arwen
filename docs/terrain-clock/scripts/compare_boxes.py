"""compare_boxes.py S_DIR G_DIR : rows measured on both boxes (same key), entries and every run's held / peak w."""
import json, sys
from pathlib import Path
def load(d):
    return {json.loads(p.read_text())["provenance"]["key"]: json.loads(p.read_text()) for p in Path(d).glob("*.json") if not p.name.startswith("provenance")}
S, G = load(sys.argv[1]), load(sys.argv[2])
both = sorted(set(S) & set(G))
same_entries = sum(S[k]["stable_s_per_km"] == G[k]["stable_s_per_km"] for k in both)
runs = diff_held = 0; maxrel = 0.0; worst = None
for k in both:
    a = {(r["wind"], r["per_km"]): r for r in S[k]["runs"]}
    for key, rg in ((( r["wind"], r["per_km"]), r) for r in G[k]["runs"]):
        if key not in a: continue
        rs = a[key]; runs += 1
        if rs["held"] != rg["held"]: diff_held += 1
        if rs["held"] and rg["held"]:
            rel = abs(rs["peak_w"] - rg["peak_w"]) / max(abs(rs["peak_w"]), 1e-9)
            if rel > maxrel: maxrel, worst = rel, (k, key, rs["peak_w"], rg["peak_w"])
print(f"rows on both boxes: {len(both)}; identical entries: {same_entries}")
for k in both:
    if S[k]["stable_s_per_km"] != G[k]["stable_s_per_km"]:
        print(f"  differs {k}: S {S[k]['stable_s_per_km']}  G {G[k]['stable_s_per_km']}")
print(f"runs on both: {runs}; held on one box and stopped on the other: {diff_held}; largest relative peak-w difference over runs held on both: {maxrel:.2e} {worst}")
