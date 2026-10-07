import csv, json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
price = json.load(open("keep/price22.json"))["layouts"]
pk = collections.defaultdict(float)
for row in csv.reader(open("keep/b-1x4/nvml.csv")):
    try: pk[int(row[1])] = max(pk[int(row[1])], float(row[2]) / 1024)
    except (ValueError, IndexError): pass
peaks14 = [pk[k] for k in sorted(pk)]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
w = 0.27
for ax, grid in zip(axes, ("2x2", "1x4")):
    before = [c["total_bytes"] for c in price[grid]["before"]["cards"]]
    after = [c["total_bytes"] for c in price[grid]["after"]["cards"]]
    x = range(len(after))
    ax.bar([i - w for i in x], before, w, color="#9aa5b1", label="price, scale-lane tip")
    ax.bar(list(x), after, w, color="#2e86ab", label="price, this lane")
    if grid == "1x4":
        ax.bar([i + w for i in x], peaks14, w, color="#d1495b", label="measured peak, 3 h run")
        for i, p in enumerate(peaks14):
            ax.text(i + w, p + 0.3, f"{p:.1f}", ha="center", fontsize=8)
    else:
        ax.text(1.5, 20, "3 h run completed on 4 x RTX 5090 (rc 0, 1,682 s)\nper-card peak samples were lost", ha="center", fontsize=9)
    ax.axhline(30.86, color="k", lw=0.9, ls="--"); ax.text(-0.45, 31.2, "budget 30.86 GiB", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"card {i}" for i in x])
    ax.set_title(f"Full HRRR 1797x1057x50, [devices] {grid}, 4 x RTX 5090")
    ax.set_ylim(0, 36)
axes[0].set_ylabel("GiB per card"); axes[1].legend(fontsize=8, loc="lower right")
fig.text(0.01, 0.01, "Box E, 2026-10-06, case 2026-10-03 21Z, HRRR physics, 3 h. Peaks: whole-device nvidia-smi every 0.2 s.", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("keep/hrrr-4card-price-vs-peak.png", dpi=130)
print(json.dumps({"peaks_1x4": [round(p, 2) for p in peaks14],
                  "price_1x4_after": [round(c["total_bytes"], 2) for c in price["1x4"]["after"]["cards"]],
                  "price_2x2_after": [round(c["total_bytes"], 2) for c in price["2x2"]["after"]["cards"]]}))
