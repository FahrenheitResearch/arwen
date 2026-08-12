#!/usr/bin/env python3
"""Drive ``gpuwm render --engine rust`` over a streamed case's wrfout frames.

This is a THIN driver, on purpose.  ``RENDERING.md`` is explicit that the
production renderer is the marketing tier and that hand-rolled matplotlib
styling has no place in forecast imagery, so nothing here draws anything:
it selects files, calls ``gpuwm render``, and reports what came back --
including, loudly, the products the engine refused and why.

    python -m tilestream.render_case_rust --run WORK/run_ks \
        --products composite_reflectivity_uh,composite_reflectivity \
        --out WORK/png --size 2400x1800

``--every N`` renders every Nth frame (the whole sequence is the default),
``--at HH:MM`` renders one valid time, and ``--crop W E S N`` first cuts the
named lon/lat box out of each selected frame with
:mod:`tilestream.case_crop` and renders the crop instead -- same cells, same
values, a smaller file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def select(run_dir: Path, every: int, at: tuple[str, ...]) -> list[Path]:
    files = sorted((run_dir / "wrfout").glob("wrfout_d01_*"))
    if not files:
        raise SystemExit(f"no wrfout frames under {run_dir / 'wrfout'}")
    if at:
        keep = [f for f in files
                if any(stamp in f.name for stamp in at)]
        if not keep:
            raise SystemExit(
                f"no frame matches {at}; have "
                + ", ".join(f.name[-19:] for f in files))
        return keep
    return files[::max(1, every)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--products", default="composite_reflectivity")
    ap.add_argument("--size", default="1600x1200")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--at", nargs="*", default=(),
                    help="valid-time stamps to render, e.g. 2019-05-28_23_00")
    ap.add_argument("--crop", type=float, nargs=4, default=None,
                    metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    ap.add_argument("--crop-dir", type=Path, default=None)
    ap.add_argument("--source-label", default="ArWen")
    ap.add_argument("--heavy", action="store_true")
    args = ap.parse_args(argv)

    files = select(args.run, args.every, tuple(args.at))
    print(f"render: {len(files)} frame(s) selected from {args.run}")

    if args.crop is not None:
        from tilestream import case_crop
        crop_dir = args.crop_dir or (args.run / "wrfout_crop")
        crop_dir.mkdir(parents=True, exist_ok=True)
        cropped = []
        window = None
        import netCDF4
        import numpy as np
        for src in files:
            if window is None:
                with netCDF4.Dataset(src) as ds:
                    lat = np.asarray(ds.variables["XLAT"][0])
                    lon = np.asarray(ds.variables["XLONG"][0])
                window = case_crop.window_from_lonlat(lat, lon, *args.crop)
                print(f"render: crop window i={window[0]}..{window[1]} "
                      f"j={window[2]}..{window[3]} "
                      f"({window[1] - window[0]}x{window[3] - window[2]} "
                      "cells of the 1200x900 domain)")
            dst = crop_dir / src.name
            case_crop.crop(src, dst, *window, grid_id=2)
            cropped.append(dst)
        files = cropped

    args.out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gpuwm.cli", "render",
               "--engine", "rust", "--products", args.products,
               "--out", str(args.out), "--size", args.size,
               "--source-label", args.source_label]
    if args.heavy:
        command.append("--heavy")
    command.extend(str(f) for f in files)
    result = subprocess.run(command, cwd=str(REPO))
    written = sorted(args.out.glob("*.png"))
    print(f"render: {len(written)} PNG(s) in {args.out}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
