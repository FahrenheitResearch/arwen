#!/usr/bin/env python3
"""Side-by-side A/B panels from already-rendered product PNGs.

Concatenates the Rust renderer's own pixels -- control frame beside
treatment frame with a labeled header strip -- so the comparison panel
adds no rendering of its own.  Weather-field pixels stay exactly what
rw_wrfbatch produced.

Usage::

    python tools/ptop_ab_montage.py --control-root DIR --treatment-root DIR \
        --product composite_reflectivity --lead 6 --lead 7 \
        --out DIR [--control-label ...] [--treatment-label ...]

Each root is searched recursively for ``<product>/*/*_fNNN.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

HEADER_PX = 34
GAP_PX = 8
BACKGROUND = (255, 255, 255)
INK = (30, 30, 30)


def find_frame(root: Path, product: str, lead: int) -> Path:
    pattern = f"*/{product}/*/*_f{lead:03d}.png"
    matches = sorted(root.glob(pattern)) or sorted(
        root.glob(f"**/{product}/**/*_f{lead:03d}.png"))
    if not matches:
        raise SystemExit(
            f"no {product} frame for lead {lead} under {root}")
    return matches[0]


def montage(control: Path, treatment: Path, labels: tuple[str, str],
            out: Path) -> None:
    left = Image.open(control).convert("RGB")
    right = Image.open(treatment).convert("RGB")
    height = max(left.height, right.height)
    width = left.width + GAP_PX + right.width
    panel = Image.new("RGB", (width, height + HEADER_PX), BACKGROUND)
    panel.paste(left, (0, HEADER_PX))
    panel.paste(right, (left.width + GAP_PX, HEADER_PX))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 9), labels[0], fill=INK)
    draw.text((left.width + GAP_PX + 8, 9), labels[1], fill=INK)
    panel.save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--treatment-root", type=Path, required=True)
    parser.add_argument("--product", action="append", required=True)
    parser.add_argument("--lead", action="append", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-label", default="control p_top=10000 Pa")
    parser.add_argument("--treatment-label", default="treatment p_top=5000 Pa")
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    for product in arguments.product:
        for lead in arguments.lead:
            control = find_frame(arguments.control_root, product, lead)
            treatment = find_frame(arguments.treatment_root, product, lead)
            target = arguments.out / f"ab_{product}_f{lead:03d}.png"
            montage(control, treatment,
                    (arguments.control_label, arguments.treatment_label),
                    target)
            print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
