"""What ink a plate's hot band is drawn in, read off the plate itself.

The fixed temperature ramp's top fifth once carried no chroma, and the
claim that it now does is a claim about pixels, not about a table.  This
reads it off two renders of ONE frame -- the same field drawn by two
builds -- so the comparison is between two inks and nothing else.

    python tools/measure_plate_hot_band.py --was HEX,HEX,HEX BEFORE.png AFTER.png

``--was`` is the three anchors the earlier build carried at the top of
the table; everything below them is unchanged, so the earlier ramp is
this tree's own table with those three put back.  ``--at`` is the
Fahrenheit value the band starts at (default 100).

The cells the band covers are found on the LATER plate, by nearest
colour on the ramp that plate was drawn with, and then read on both.
Prints, for each plate, the HSV saturation of those cells and their
CIELAB L*: the first says whether the band carries colour at all, the
second how much tone it spends, which are two different questions and
were answered as one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

#: The crate the ramp lives in, relative to this file's parent.
TABLE_SOURCE = Path("rustwx/crates/rustwx-render/src/colormaps.rs")

#: The fixed range the table spans, and the product's own step: the 2 m
#: plate asks for one colour per degree over -60..120 F
#: (``rustwx-products/src/plot_design.rs``), so a colour's index IS its
#: value.
RANGE_F = (-60.0, 120.0)
STEP_F = 1.0


def table_anchors(source: Path) -> list[str]:
    """The `TEMPERATURE` anchor list, read out of the crate's own source."""

    text = source.read_text(encoding="utf-8", errors="strict")
    match = re.search(r"const TEMPERATURE: &\[&str\] = &\[(.*?)\];", text, re.S)
    if match is None:
        raise SystemExit(f"{source}: no TEMPERATURE anchor table")
    return re.findall(r"#[0-9a-fA-F]{6}", match.group(1))


def rgb_of(hexes) -> np.ndarray:
    return np.array(
        [[int(value[i:i + 2], 16) for i in (1, 3, 5)] for value in hexes],
        dtype=float)


def ramp(anchors: list[str], n: int) -> np.ndarray:
    """`lerp_hex`: n colours evenly spaced across the anchors, rounded."""

    points = rgb_of(anchors)
    out = np.empty((n, 3))
    for i in range(n):
        t = i / (n - 1) * (len(points) - 1)
        lo = int(t)
        hi = min(lo + 1, len(points) - 1)
        out[i] = np.round(points[lo] + (t - lo) * (points[hi] - points[lo]))
    return out


def lightness(rgb: np.ndarray) -> np.ndarray:
    """CIELAB L* of sRGB, D65."""

    linear = rgb / 255.0
    linear = np.where(linear <= 0.04045, linear / 12.92,
                      ((linear + 0.055) / 1.055) ** 2.4)
    y = (linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522
         + linear[..., 2] * 0.0721750)
    root = np.where(y > 0.008856, np.cbrt(np.maximum(y, 0.0)),
                    (903.3 * y + 16.0) / 116.0)
    return 116.0 * root - 16.0


def saturation(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation, the same measure the palette guard is written on."""

    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    return np.where(high == 0, 0.0, (high - low) / np.maximum(high, 1e-9))


def widest_column_run(mask: np.ndarray) -> tuple[int, int]:
    """The map panel: the widest run of columns holding changed pixels.

    A plate's colour bar changed too, and it is a ramp rather than a
    field, so it would drag every statistic toward the ramp's own shape.
    It is also narrow, and separated from the panel by a margin.
    """

    present = mask.any(axis=0)
    runs, start = [], None
    for x, hit in enumerate(present):
        if hit and start is None:
            start = x
        elif not hit and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, len(present) - 1))
    if not runs:
        raise SystemExit("the two plates are identical")
    return max(runs, key=lambda run: run[1] - run[0])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--was", required=True,
                        help="the three anchors the earlier build had on top")
    parser.add_argument("--at", type=float, default=100.0,
                        help="the Fahrenheit value the band starts at")
    args = parser.parse_args(argv)

    anchors = table_anchors(Path(__file__).resolve().parent / TABLE_SOURCE)
    was = [value if value.startswith("#") else "#" + value
           for value in args.was.split(",")]
    if len(was) != 3:
        raise SystemExit("--was takes the three anchors above 90 F")
    bands = int(round((RANGE_F[1] - RANGE_F[0]) / STEP_F))
    later = ramp(anchors, bands)
    earlier = ramp(anchors[:-3] + was, bands)
    first = int(round((args.at - RANGE_F[0]) / STEP_F))

    after = np.array(Image.open(args.after).convert("RGB")).astype(float)
    before = np.array(Image.open(args.before).convert("RGB")).astype(float)
    if after.shape != before.shape:
        raise SystemExit("the two plates are not the same frame")
    changed = np.any(after != before, axis=2)
    left, right = widest_column_run(changed)
    panel = np.zeros_like(changed)
    panel[:, left:right + 1] = True
    cells = changed & panel

    drawn = after[cells]
    index = np.linalg.norm(drawn[:, None, :] - later[None, :, :],
                           axis=2).argmin(axis=1)
    hot = index >= first
    print(f"panel      columns {left}..{right}, {int(cells.sum())} changed cells")
    print(f"band       {int(hot.sum())} cells at or above {args.at:.0f} F "
          f"on the ramp the later plate was drawn with")
    for name, plate, top in (("earlier", before, was),
                             ("later", after, anchors[-3:])):
        values = plate[cells][hot]
        light = lightness(values)
        sat = saturation(values)
        low, mid, high = np.percentile(light, (5, 50, 95))
        print(f"{name:<10} saturation p50 {np.median(sat):.3f}   "
              f"L* p5 {low:.1f} p50 {mid:.1f} p95 {high:.1f} "
              f"spread {high - low:.1f}")
        print(f"{'':<10} anchors " + "  ".join(
            f"{value} sat {saturation(rgb_of([value])[0]):.3f} "
            f"L* {lightness(rgb_of([value])[0]):.1f}"
            for value in top))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
