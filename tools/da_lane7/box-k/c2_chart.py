"""c2_chart.py: C2 (MASPT over the first hour after each analysis) and the S3 insertion-jump chart for
DA lane 7 arms, from each arm's cycle-report.json. Analysis charts (not weather fields): matplotlib.

  python c2_chart.py --arm "one-shot=RUN/da/cycle-report.json" ... --out DIR [--name PREFIX]

Each leg record's MASPT series is placed on minutes relative to the analysis time that leg starts from
(a 4D leg is rewound to t - 30 min, so its series starts at -30). The member mean is drawn per arm; the
unanalysed control of the first arm is the no-DA reference (dashed). Writes PREFIX-c2.png,
PREFIX-s3-jump.png and PREFIX-c2.json (the table behind both)."""
import argparse
import json
from pathlib import Path

import numpy as np

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE, NODA = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#52514e"


def records(leg):
    """(name, record) for every trajectory record of a leg, 4D part A records included."""
    for name, rec in leg["trajectories"].items():
        iau4d = rec.get("iau4d") or {}
        # a split 4D leg's own record is part B: the background over [t_next - 30, t_next], which
        # the rewound leg replaces; only part A (and unsplit records) carry the forecast
        if iau4d.get("split_at_seconds") is None:
            yield name, rec
        part = iau4d.get("part_a")
        if part:
            yield name, part


def series_by_analysis(report):
    """{analysis_s: {"members": [(minutes, maspt, tlow)], "control": [...], "jump": [...]} }"""
    analyses = sorted({float(leg["start_s"]) for leg in report["legs"] if float(leg["start_s"]) > 0})
    out = {t: {"members": [], "control": [], "jump": []} for t in analyses}
    for leg in report["legs"]:
        for name, rec in records(leg):
            m = rec.get("maspt")
            if not m or not m.get("series", {}).get("minutes"):
                continue
            start = float(m["start_seconds"])
            # the analysis this record's start belongs to: its own start, or 30 min later (4D rewind)
            t_a = min((t for t in analyses if 0 <= t - start <= 1800.0 + 1e-6), default=None)
            if t_a is None:
                continue
            minutes = np.asarray(m["series"]["minutes"]) + (start - t_a) / 60.0
            entry = (minutes, np.asarray(m["series"]["maspt"]),
                     np.asarray(m["series"].get("lowest_level_t_tendency_k_per_h", [])))
            out[t_a]["control" if name == "control" else "members"].append(entry)
            if name != "control" and m.get("insertion_jump"):
                out[t_a]["jump"].append(m["insertion_jump"])
    return out


def mean_curve(entries, grid):
    if not entries:
        return None
    stack = [np.interp(grid, mn, v, left=np.nan, right=np.nan) for mn, v, _ in entries]
    return np.nanmean(np.stack(stack), axis=0)


def window_mean(entries, lo, hi):
    vals = []
    for mn, v, _ in entries:
        sel = (mn > lo) & (mn <= hi)
        if sel.any():
            vals.append(float(v[sel].mean()))
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="lane7")
    p.add_argument("--title", default="Surface pressure noise after each analysis")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = []
    for spec in a.arm:
        label, path = spec.split("=", 1)
        arms.append((label, series_by_analysis(json.loads(Path(path).read_text()))))
    analyses = sorted(set().union(*[set(s) for _, s in arms]))
    grid = np.linspace(-30, 60, 181)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = {"analyses_s": analyses, "arms": {}}

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, len(analyses), figsize=(5.2 * len(analyses), 4.6), sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, t_a in zip(axes[0], analyses):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        control = arms[0][1].get(t_a, {}).get("control", [])
        c = mean_curve(control, grid)
        if c is not None:
            ax.plot(grid, c, color=NODA, linewidth=2, linestyle=(0, (4, 3)), label="no DA (control)")
        for i, (label, s) in enumerate(arms):
            cur = mean_curve(s.get(t_a, {}).get("members", []), grid)
            if cur is None:
                continue
            ax.plot(grid, cur, color=SLOTS[i], linewidth=2, label=label)
        ax.axvline(0, color=INK2, linewidth=1)
        hour = int(round(t_a / 3600.0))
        ax.set_title(f"analysis {18 + hour:02d}Z", color=INK, fontsize=12)
        ax.set_xlabel("minutes from the analysis time")
        ax.set_xlim(-30, 60)
    axes[0][0].set_ylabel("MASPT, hPa/h (member mean)")
    axes[0][-1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(a.title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"{a.name}-c2.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    for i, (label, s) in enumerate(arms):
        row = {}
        for t_a in analyses:
            e = s.get(t_a, {})
            jumps = e.get("jump", [])
            row[str(int(t_a))] = {
                "maspt_0_30": window_mean(e.get("members", []), 0.0, 30.0),
                "maspt_minus30_0": window_mean(e.get("members", []), -30.0, 0.0),
                "control_0_30": window_mean(arms[0][1].get(t_a, {}).get("control", []), 0.0, 30.0),
                "t_jump_mean_k": (float(np.mean([j["lowest_level_t_mean_abs_k"] for j in jumps]))
                                  if jumps else 0.0),
                "t_jump_max_k": (float(np.max([j["lowest_level_t_max_abs_k"] for j in jumps]))
                                 if jumps else 0.0),
                "p_jump_mean_hpa": (float(np.mean([j["lowest_level_p_mean_abs_hpa"] for j in jumps]))
                                    if jumps else 0.0),
            }
        table["arms"][label] = row

    # S3: the insertion's own jump in the lowest-level temperature, per arm and analysis
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(arms) * len(analyses) / 2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    width = 0.8 / max(1, len(arms))
    for i, (label, _s) in enumerate(arms):
        xs = np.arange(len(analyses)) + (i - (len(arms) - 1) / 2) * width
        ys = [table["arms"][label][str(int(t))]["t_jump_mean_k"] for t in analyses]
        ax.bar(xs, ys, width=width - 0.02, color=SLOTS[i], label=label)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(analyses)))
    ax.set_xticklabels([f"{18 + int(round(t / 3600)):02d}Z" for t in analyses])
    ax.set_ylabel("lowest-level T jump at insertion, K (mean |dT|)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Temperature jump when the analysis goes in", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / f"{a.name}-s3-jump.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    (out / f"{a.name}-c2.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
