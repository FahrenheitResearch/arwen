"""How much of its own colour bar a rendered vertical cut actually uses.

A cut drawn on a bar whose bottom is far below anything in the air uses
a sliver of that bar and comes out as one colour, which is legible only
as "something is there".  This reads that fraction off the PNG itself
rather than off the values that went into it: it finds the vertical
colour bar, walks it into a ladder of colours, and maps a profile
through the air back onto that ladder.

    python tools/measure_section_bar_use.py [--column X] PICTURE [PICTURE ...]

Pointed at the bar's own column with ``--column``, it reads back the
band it samples and nothing else, 90.2 percent of the bar, which is how
the instrument is checked: the mapping from colour to rung is one to one
and in order, or the bar would not measure as itself.

The profile is taken down one column, at evenly spaced heights between
the fifth and ninety-fifth percentile of the frame, which keeps it clear
of the title at the top and of the terrain mask and its soft edge at the
bottom.  Prints the bar's pixel extent, the rungs the profile lands on,
the fraction of the bar between the lowest and highest of them, and
three of the samples in full.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Where down the frame the profile is taken, as a fraction of the plot's
# width between its left edge and the colour bar.  A quarter of the way
# across this tree's own section line is open water, so the column is air
# from the surface to the top of the frame.
PROFILE_AT = 0.25

# How many heights the profile samples, and the band of the frame it
# keeps to.
SAMPLES = 40
FROM_FRACTION, TO_FRACTION = 0.05, 0.95


def bar_column(rgb: np.ndarray) -> tuple[int, int, int]:
    """The x of the colour bar, and the first and last y it spans.

    A colour bar is the one tall thing in a plot that is identical to the
    column beside it and still carries hundreds of colours: the plot
    itself is noisy across as well as down, and a label or an axis is
    flat.  Its rows are the longest run over which one row differs only
    slightly from the next, and they are the plot box's rows too, because
    the bar is drawn the height of the box.
    """

    _, width, _ = rgb.shape
    best: tuple[int, int] | None = None
    for x in range(width // 2, width - 1):
        if not np.array_equal(rgb[:, x, :], rgb[:, x + 1, :]):
            continue
        colours = len(np.unique(rgb[:, x, :], axis=0))
        if best is None or colours > best[1]:
            best = (x, colours)
    if best is None:
        raise SystemExit("no colour bar found")
    x = best[0]
    step = np.abs(np.diff(rgb[:, x, :].astype(np.int32), axis=0)).max(axis=1)
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for y, smooth in enumerate([*(step <= 24), False]):
        if smooth and run_start is None:
            run_start = y
        elif not smooth and run_start is not None:
            runs.append((run_start, y))
            run_start = None
    top, bottom = max(runs, key=lambda run: run[1] - run[0])
    return x, top, bottom


def measure(path: Path, at_column: int | None = None) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    _, width, _ = rgb.shape
    x, top, bottom = bar_column(rgb)
    ladder = rgb[top:bottom, x, :].astype(np.float32)
    rungs = len(ladder)

    left = width // 12
    column = at_column if at_column is not None else int(left + (x - left) * PROFILE_AT)
    rows = [
        int(bottom - (bottom - top) * fraction)
        for fraction in np.linspace(FROM_FRACTION, TO_FRACTION, SAMPLES)
    ]
    samples = rgb[rows, column, :].astype(np.float32)
    # The bar is drawn top to bottom in descending value, so a rung index
    # counted from the bar's own first row is a position on the bar.
    landed = np.linalg.norm(samples[:, None, :] - ladder[None, :, :], axis=2).argmin(axis=1)
    span = (landed.max() - landed.min() + 1) / rungs

    print(f"{path.name}")
    print(f"  bar at x={x}, rows {top}..{bottom} ({rungs} rungs)")
    print(f"  profile down x={column}, {SAMPLES} heights over the middle 90 percent")
    print(f"  rungs landed on    : {landed.min()}..{landed.max()}")
    print(f"  bar used by the air: {span * 100:.1f} percent")
    for index in (0, SAMPLES // 2, SAMPLES - 1):
        colour = tuple(int(v) for v in rgb[rows[index], column])
        print(f"    row {rows[index]}: rgb{colour} -> rung {landed[index]}")
    return {
        "picture": path.name,
        "bar_x": x,
        "bar_rows": (top, bottom),
        "rungs": rungs,
        "column": column,
        "rung_lo": int(landed.min()),
        "rung_hi": int(landed.max()),
        "bar_used_percent": round(span * 100, 1),
    }


if __name__ == "__main__":
    arguments = sys.argv[1:]
    at_column = None
    if len(arguments) >= 2 and arguments[0] == "--column":
        at_column = int(arguments[1])
        arguments = arguments[2:]
    if not arguments:
        raise SystemExit(__doc__)
    for argument in arguments:
        measure(Path(argument), at_column)
