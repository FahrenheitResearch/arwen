"""Full HRRR [devices] on 4x RTX 5090: admission terms before/after per card against the budget and the measured peak."""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path("/work/meeting-perf-a13")
price = json.load(open(W / "price22.json"))
BUDGET = 30.86
GIB = 2**30


def nvml_peaks(run):
    path = W / "runs" / run / "nvml.csv"
    if not path.exists():
        return None
    peak = defaultdict(float)
    for row in csv.reader(open(path)):
        try:
            peak[int(row[1])] = max(peak[int(row[1])], float(row[2]) / 1024.0)
        except (ValueError, IndexError):
            continue
    return [peak[k] for k in sorted(peak)] or None


fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
summary = {}
for ax, (grid, run) in zip(axes, (("2x2", "a-2x2"), ("1x4", "b-1x4"))):
    row = price["layouts"][grid]
    peaks = nvml_peaks(run)
    summary[grid] = {"peaks_gib": peaks}
    w = 0.38
    for side, offset, alpha in (("before", -w / 2, 0.45), ("after", w / 2, 1.0)):
        cards = row[side]["cards"]
        bottom = [0.0] * len(cards)
        for term, color in (("resident_bytes", "#2e86ab"), ("seam_bytes", "#f18f01"),
                            ("template_bytes", "#6a994e")):
            vals = [c[term] for c in cards]
            ax.bar([i + offset for i in range(len(cards))], vals, w, bottom=bottom, color=color, alpha=alpha,
                   label=(f"{term.split('_')[0]} ({side})" if grid == "2x2" else None))
            bottom = [b + v for b, v in zip(bottom, vals)]
        summary[grid][side] = [round(b, 2) for b in bottom]
    if peaks:
        ax.scatter(range(len(peaks)), peaks, marker="D", s=60, color="#d1495b", zorder=5,
                   label=("measured peak (nvidia-smi)" if grid == "2x2" else None))
    ax.axhline(BUDGET, color="k", lw=0.9, ls="--")
    ax.text(len(row["after"]["cards"]) - 0.5, BUDGET + 0.25, f"budget {BUDGET} GiB", ha="right", fontsize=8)
    ax.set_xticks(range(len(row["after"]["cards"])))
    ax.set_xticklabels([f"card {i}" for i in range(len(row["after"]["cards"]))])
    ax.set_title(f"Full HRRR 1797x1057x50, [devices] {grid}")
    ax.set_ylim(0, 36)
axes[0].set_ylabel("GiB per card")
axes[0].legend(fontsize=7, loc="lower left", ncol=2)
fig.text(0.01, 0.01, "Faded bars: price at the scale-lane tip (lake arrays at every column). Solid bars: price with lake columns "
         "from the land cover. Diamonds: measured per-card peak, box E, 3 h forecast.", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(sys.argv[1], dpi=130)
print(json.dumps(summary, indent=1))
