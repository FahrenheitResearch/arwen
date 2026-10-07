"""Report for tools/da_leg0_balance.py runs: MASPT and spread retention.

Reads ``<out>/<arm>/<trajectory>/{maspt.json,snapshots.npz}`` for every
arm and writes ``<out>/report.json`` plus an analysis chart
``<out>/leg0-balance.png`` (matplotlib: an analysis chart, not a weather
field).  CPU only.

Spread retention: at each snapshot time the ensemble standard deviation
over the members at every interior point and level, RMS over the domain
(a 15-cell rim excluded), divided by the same at t = 0.  A balanced start
keeps it near one over the first hour; an unbalanced one loses the
divergent and the ageostrophic part as waves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RIM = 15
FIELDS = ("u", "v", "thp", "qv")


def load_arm(root: Path):
    members = {}
    control = None
    for traj in sorted(root.iterdir()):
        if not (traj / "done.json").exists():
            continue
        entry = {
            "maspt": json.loads((traj / "maspt.json").read_text()),
            "snapshots": np.load(traj / "snapshots.npz"),
            "done": json.loads((traj / "done.json").read_text()),
        }
        if traj.name == "control":
            control = entry
        else:
            members[traj.name] = entry
    return members, control


def spread_curve(members, field):
    """(times_min, rms ensemble std per time) over interior levels."""
    names = sorted(members)
    times = members[names[0]]["snapshots"]["times"]
    out = []
    for t in times:
        tag = f"{int(round(t / 60.0)):03d}"
        stack = np.stack([members[n]["snapshots"][f"{field}_{tag}"]
                          .astype(np.float64) for n in names])
        std = stack.std(axis=0, ddof=1)[:, RIM:-RIM, RIM:-RIM]
        out.append(float(np.sqrt(np.mean(std ** 2))))
    return times / 60.0, np.asarray(out)


def maspt_mean(entries, window):
    vals = [e["maspt"][f"maspt_first_{window}min"] for e in entries
            if e["maspt"].get(f"maspt_first_{window}min") is not None]
    return float(np.mean(vals)) if vals else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="WOOF member start: balance test")
    args = p.parse_args(argv)
    report = {"arms": {}, "rim_cells": RIM}
    curves = {}
    series = {}
    control_entry = None
    for arm_dir in sorted(args.out.iterdir()):
        if not arm_dir.is_dir() or not (arm_dir / "arm.json").exists():
            continue
        members, control = load_arm(arm_dir)
        arm = arm_dir.name
        entry = {"members": len(members)}
        if control is not None:
            control_entry = control
            entry["maspt_first_30min"] = control["maspt"]["maspt_first_30min"]
            entry["maspt_first_60min"] = control["maspt"]["maspt_first_60min"]
            series[arm] = control["maspt"]["series"]
        if members:
            entry["maspt_first_30min"] = maspt_mean(members.values(), 30)
            entry["maspt_first_60min"] = maspt_mean(members.values(), 60)
            jumps = [m["maspt"]["insertion_jump"] for m in members.values()
                     if m["maspt"].get("insertion_jump")]
            if jumps:
                entry["insertion_jump_mean"] = {
                    k: float(np.mean([j[k] for j in jumps])) for k in jumps[0]}
            first = next(iter(members.values()))["maspt"]["series"]
            minutes = np.asarray(first["minutes"])
            stack = np.stack([np.interp(
                minutes, m["maspt"]["series"]["minutes"],
                m["maspt"]["series"]["maspt"]) for m in members.values()])
            series[arm] = {"minutes": minutes.tolist(),
                           "maspt": stack.mean(axis=0).tolist()}
            retention = {}
            for field in FIELDS:
                t, s = spread_curve(members, field)
                retention[field] = {
                    "minutes": t.tolist(), "spread": s.tolist(),
                    "retention": (s / s[0]).tolist() if s[0] > 0 else None}
                curves[(arm, field)] = (t, s / s[0] if s[0] > 0 else s)
            entry["spread"] = retention
        report["arms"][arm] = entry
    if control_entry is not None:
        c30 = control_entry["maspt"]["maspt_first_30min"]
        for arm, entry in report["arms"].items():
            if entry.get("maspt_first_30min") is not None and c30:
                entry["maspt_30min_over_control"] = (
                    entry["maspt_first_30min"] / c30)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; report.json written", flush=True)
        return 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    order = ["control", "independent-nomass", "independent",
             "rotational-nomass", "rotational"]
    colours = {"control": "black", "independent-nomass": "#b2182b",
               "independent": "#ef8a62", "rotational-nomass": "#67a9cf",
               "rotational": "#2166ac"}
    ax = axes[0]
    for arm in order:
        if arm in series:
            s = series[arm]
            ax.plot(s["minutes"], s["maspt"], label=arm, color=colours[arm],
                    lw=1.4)
    ax.set_xlabel("minutes from start")
    ax.set_ylabel("MASPT (hPa/h), interior mean per step")
    ax.set_yscale("log")
    ax.set_title("surface pressure tendency")
    ax.legend(fontsize=8)
    for ax, field, label in ((axes[1], "u", "u wind"),
                             (axes[2], "thp", "theta")):
        for arm in order:
            if (arm, field) in curves:
                t, r = curves[(arm, field)]
                ax.plot(t, r, marker="o", ms=3, label=arm, color=colours[arm])
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("minutes from start")
        ax.set_ylabel("ensemble spread / spread at start")
        ax.set_title(f"spread retention, {label}")
        ax.set_ylim(0.0, 1.3)
        ax.legend(fontsize=8)
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out / "leg0-balance.png", dpi=130)
    print(json.dumps({a: {k: v for k, v in e.items() if k != "spread"}
                      for a, e in report["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
