"""Plot resolved output calendars (analysis chart, no weather-field rendering)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--hours", type=int, default=48)
    args = parser.parse_args()
    from gpuwm.experiment import load_experiment
    from gpuwm.core.clock import resolve_clock
    from gpuwm.io.history_layout import produced_history_shapes, history_frame_bytes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exp = replace(load_experiment(args.recipe), run_seconds=args.hours * 3600)
    clock = resolve_clock(exp)
    spec, cfg = clock.root, exp.root.run
    offsets = [0]
    while True:
        tick = spec.next_history_offset(offsets[-1])
        if tick > clock.run_ticks:
            break
        offsets.append(tick)
    seconds = [tick / clock.tick_den for tick in offsets]
    frame_bytes = history_frame_bytes(cfg)
    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "grid": [cfg.nx, cfg.ny, cfg.nz],
        "history_seconds": seconds,
        "history_frame_count": len(seconds),
        "history_field_count": len(produced_history_shapes(cfg)),
        "history_frame_bytes_uncompressed": frame_bytes,
        "history_bytes_uncompressed": frame_bytes * len(seconds),
        "prec_acc_dt_minutes": cfg.prec_acc_dt,
        "prec_acc_dt1_minutes": cfg.prec_acc_dt1,
        "gsd_reset_minutes": cfg.reset_interval1,
        "qualification": "Calendar and inventory only; no full-grid throughput measurement.",
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "output-controls.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4),
                             gridspec_kw={"height_ratios": [1, 1.35]})
    fig.suptitle("Resolved output calendars", fontsize=19, x=.11, ha="left")
    hours = [value / 3600 for value in seconds]
    rows = [("History", hours, "#1167a6")]
    for label, minutes, color in (
        (f"{cfg.prec_acc_dt1:g}-minute precipitation reset", cfg.prec_acc_dt1, "#aa5b00"),
        (f"{cfg.prec_acc_dt:g}-minute precipitation reset", cfg.prec_acc_dt, "#906c24"),
        ("GSD maxima reset", cfg.reset_interval1, "#6b4198"),
    ):
        period = int(minutes * 60)
        if period > 0:
            rows.append((label, [s / 3600 for s in range(0, args.hours*3600+1, period)], color))
    for row, (_, events, color) in enumerate(rows):
        axes[0].vlines(events, row-.26, row+.26, color=color, linewidth=1)
    axes[0].set_yticks(range(len(rows)), [row[0] for row in rows])
    axes[0].invert_yaxis()
    axes[0].set_title("Clock boundaries; interval resets precede the current step", loc="left")
    axes[1].step(hours, range(1, len(hours)+1), where="post", color="#1167a6",
                 linewidth=2, label=f"Resolved cadence: {len(hours)} frames")
    hourly = list(range(args.hours+1))
    axes[1].step(hourly, range(1, len(hourly)+1), where="post", color="#8b9299",
                 linewidth=2, label=f"Hourly reference: {len(hourly)} frames")
    axes[1].set_ylabel("Cumulative history frames, including t=0")
    axes[1].set_xlabel("Forecast hour")
    axes[1].legend(loc="upper left", frameon=False)
    change = cfg.history_interval_change_s / 3600
    for ax in axes:
        ax.set_xlim(0, args.hours)
        ax.set_xticks(range(0, args.hours+1, 6))
        ax.grid(axis="x", alpha=.16)
        if change:
            ax.axvline(change, color="#253849", linestyle="--", linewidth=1)
    if change:
        at_change = sum(value <= change for value in hours)
        label = (f"{exp.root.history_interval_s/60:g}-minute → "
                 f"{cfg.history_interval2_s/60:g}-minute history at f{change:g}")
        axes[1].annotate(label, xy=(change, at_change),
                         xytext=(change+2, at_change-14), fontsize=10,
                         arrowprops={"arrowstyle": "->", "color": "#253849"})
    caption = (
        f"{cfg.nx} × {cfg.ny} × {cfg.nz}: {receipt['history_field_count']} planned fields; "
        f"{frame_bytes/1024**3:.3f} GiB per uncompressed frame. "
        f"{frame_bytes*len(hours)/1024**4:.3f} TiB over {args.hours} h.\n"
        "Schedule/inventory evidence only. Snapshot capture and queue backpressure can delay stepping.\n"
        "diag_int=15 / wind_int=5 control the independent GPU climate-statistics windows."
    )
    fig.text(.11, .025, caption, fontsize=10, color="#374754", va="bottom")
    fig.tight_layout(rect=(0, .14, 1, .95))
    path = args.destination / "output-controls.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    (args.destination / "output-controls-caption.txt").write_text(caption+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
