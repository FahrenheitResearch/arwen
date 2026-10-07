"""Timing charts for the LETKF neighbour-roster change (matplotlib).

Reads the bench JSON lines (tools/da_obs_index_bench.py) and, when given,
the before/after cycle reports, and writes plain PNG charts of where the
analysis spends its wall clock.  Numbers only; no weather fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PHASES = (("setup_seconds", "setup (prior, QC)", "#8da0cb"),
          ("weights_seconds", "neighbour search", "#e5734a"),
          ("transform_seconds", "transform", "#66a182"),
          ("finish_seconds", "finish", "#b3b3b3"))


def _load(path):
    return json.loads(Path(path).read_text().strip().splitlines()[-1])


def phase_chart(rows, title, out):
    """Stacked horizontal bars, one per labelled run."""
    fig, ax = plt.subplots(figsize=(9.5, 0.9 + 0.75 * len(rows)))
    labels = [label for label, _ in rows]
    left = [0.0] * len(rows)
    for key, name, colour in PHASES:
        widths = [float(row.get(key, 0.0)) for _, row in rows]
        ax.barh(labels, widths, left=left, color=colour, label=name,
                edgecolor="white", height=0.6)
        left = [a + b for a, b in zip(left, widths)]
    for y, total in enumerate(left):
        ax.text(total, y, f"  {total:,.0f} s", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("wall seconds")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, max(left) * 1.18)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", action="append", nargs="+", required=True,
                        metavar=("OUT TITLE", "LABEL=JSON"),
                        help="OUT TITLE LABEL=path [LABEL=path ...]")
    args = parser.parse_args(argv)
    for spec in args.chart:
        out, title, *pairs = spec
        rows = []
        for pair in pairs:
            label, path = pair.split("=", 1)
            rows.append((label, _load(path)))
        phase_chart(rows, title, out)
        print(out)


if __name__ == "__main__":
    main()
