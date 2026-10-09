"""chart.py CANDIDATE RECHECK_DIR OUT.png : analysis chart (not a weather field): per spacing and settings arm, the
longest step that held three hours (4 substeps) by grid slope and wind, 4.5 km crest.  Cells whose stop the blow-up
re-check found to be a finite wave are hatched; NCAR's 6.0 s/km is the colour break."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
cand = json.load(open(sys.argv[1], encoding="utf-8"))
finite = set()
for p in Path(sys.argv[2]).glob("*.jsonl"):
    for l in p.read_text().splitlines():
        r = json.loads(l)
        if r["held"] and abs(r["per_km"] - 6.0) < 1e-9 and r["sound_steps"] == 4:
            finite.add((r["dx_m"], r["crest_m"], r["ridge_slope"], r["wind"], r["settings"]))
winds = [20, 30, 40, 50, 60]
levels = [0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7]
cmap = ListedColormap(["#9e9e9e", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#e0e0e0", "#d1e5f0", "#4393c3", "#2166ac"])
norm = BoundaryNorm(levels, cmap.N)
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
for j, dx in enumerate((2000.0, 3000.0, 12000.0)):
    for i, st in enumerate(("generated", "ncar")):
        ax = axes[i][j]
        rows = sorted([r for r in cand["rows"] if r["dx_m"] == dx and r["crest_m"] == 4500.0 and r["sound_steps"] == 4 and r["settings"] == st], key=lambda r: r["slope"])
        grid = [[(e if e is not None else 0.5) if t is not None else float("nan") for e, t in zip(r["stable_s_per_km"], r["top_s_per_km"])] for r in rows]
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        for y, r in enumerate(rows):
            for x, (e, t) in enumerate(zip(r["stable_s_per_km"], r["top_s_per_km"])):
                txt = "-" if t is None else ("none" if e is None else f"{e:g}")
                ax.text(x, y, txt, ha="center", va="center", fontsize=8)
                if (dx, 4500.0, r["ridge_slope"], float(winds[x]), st) in finite and (e is None or e < 6.0):
                    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, hatch="///", lw=0))
        ax.set_xticks(range(5), [f"{w}" for w in winds]); ax.set_yticks(range(len(rows)), [f"{r['slope']:.3f}" for r in rows])
        ax.set_title(f"{dx/1000:g} km, 4.5 km crest, {st} settings", fontsize=10)
        ax.set_xlabel("crest-level wind (m/s)"); ax.set_ylabel("grid-read slope")
fig.suptitle("Longest step that held three hours (s/km, 4 substeps); blue = NCAR's 6.0 s/km or longer held; "
             "hatched = stopped by the map's w bound but ran 3 h finite at 6.0 s/km under the blow-up re-check", fontsize=10)
fig.savefig(sys.argv[3], dpi=110)
