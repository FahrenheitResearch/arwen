"""da_cards_chart.py OUT.png LABEL=samples.csv[@card,card..] ...: per-card GPU utilisation timelines.

An analysis chart (not a weather field), from the gpu-samples.csv a DA run script writes every 5 s
(`epoch,util, mem;util, mem;...` per card).  Prints and titles each run's share of card-time busy (>5 %)
and mean utilisation; Lane 9 uses it for the before/after leg-boundary idle picture.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path, cards=None):
    t, u = [], []
    for line in open(path):
        stamp, rest = line.strip().split(",", 1)
        vals = [int(x.split(",")[0]) for x in rest.strip(";").split(";")]
        t.append(int(stamp)); u.append(vals)
    t = np.array(t) - t[0]; u = np.array(u, float)
    # trim leading/trailing all-idle samples (queue wait / teardown)
    active = np.where(u.max(axis=1) > 0)[0]
    lo, hi = active[0], active[-1] + 1
    t, u = t[lo:hi] - t[lo], u[lo:hi]
    if cards is not None:
        u = u[:, cards]
    return t, u


out = sys.argv[1]
runs = []
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1)
    cards = None
    if "@" in path:
        path, c = path.split("@"); cards = [int(x) for x in c.split(",")]
    runs.append((label, *load(path, cards)))
fig, axes = plt.subplots(len(runs), 1, figsize=(12, 2.2 + 0.45 * max(r[2].shape[1] for r in runs) * len(runs)), squeeze=False)
for ax, (label, t, u) in zip(axes[:, 0], runs):
    busy = (u > 5).mean() * 100
    mean = u.mean()
    ax.imshow(u.T, aspect="auto", cmap="viridis", vmin=0, vmax=100, interpolation="nearest",
              extent=(0, t[-1] / 60, u.shape[1] - 0.5, -0.5))
    ax.set_yticks(range(u.shape[1])); ax.set_ylabel("card")
    ax.set_title(f"{label}: cards busy {busy:.0f}% of card-time, mean utilisation {mean:.1f}%, wall {t[-1]/60:.1f} min")
    print(f"{label}: busy {busy:.1f}% mean {mean:.1f}% wall {t[-1]:.0f} s cards {u.shape[1]}")
axes[-1, 0].set_xlabel("minutes from first busy sample (dark = idle card, yellow = 100% busy)")
fig.tight_layout(); fig.savefig(out, dpi=110)
