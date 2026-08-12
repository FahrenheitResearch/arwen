#!/usr/bin/env python3
"""Find the run's strongest simulated storm, so a crop is chosen by data.

Ranks every history frame by the peak instantaneous 2-5 km updraft helicity
inside the forecast interior (the prescribed boundary frame is excluded --
nothing in it is a forecast), falls back to peak column-max reflectivity
where UH is absent, and prints the valid time, the cell, its lon/lat and a
lon/lat box centred on it that :mod:`tilestream.case_crop` can cut.

Choosing the crop by eye is how a figure ends up centred on a boundary
artefact; choosing it by the field the storm is defined by cannot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--field", choices=("uh", "refl"), default="uh")
    ap.add_argument("--half-width-km", type=float, default=330.0)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--box", type=float, nargs=4, default=None,
                    metavar=("LON0", "LON1", "LAT0", "LAT1"),
                    help="restrict the search to this lon/lat rectangle")
    args = ap.parse_args(argv)

    from tilestream.render_case import _grid_latlon, _read

    meta = json.loads((args.run / "run.json").read_text())
    lat, lon = _grid_latlon(meta)
    edge = int(meta["hard"]) + int(meta["taper"])
    interior = np.zeros(lat.shape, bool)
    interior[edge:-edge, edge:-edge] = True
    if args.box:
        lo0, lo1, la0, la1 = args.box
        interior &= ((lon >= lo0) & (lon <= lo1)
                     & (lat >= la0) & (lat <= la1))

    rows = []
    for path in sorted((args.run / "history").glob("h_*.npz")):
        d = _read(path)
        field = d.get("uh") if args.field == "uh" else d.get("refl_10cm")
        if field is None:
            continue
        masked = np.where(interior, field, -np.inf)
        j, i = np.unravel_index(int(np.argmax(masked)), masked.shape)
        rows.append({
            "file": path.name,
            "valid": str(d["valid"])[:19],
            "peak": float(masked[j, i]),
            "i": int(i), "j": int(j),
            "lon": float(lon[j, i]), "lat": float(lat[j, i]),
            "refl_max": float(np.where(interior, d["refl_10cm"],
                                       -99).max()),
            "wmax": float(np.where(interior, d["wmax"], -99).max()),
        })
    if not rows:
        raise SystemExit(f"no usable history frames in {args.run}")

    rows.sort(key=lambda r: -r["peak"])
    dx_km = meta["dx"] / 1000.0
    half = args.half_width_km
    print(f"top {args.top} frames by peak {args.field} "
          f"(interior only, boundary frame of {edge} cells excluded)")
    for r in rows[:args.top]:
        dlat = half / 111.0
        dlon = half / (111.0 * max(np.cos(np.deg2rad(r["lat"])), 0.2))
        print(f"  {r['valid']}  peak {r['peak']:8.1f}  at "
              f"i={r['i']:4d} j={r['j']:4d}  "
              f"({r['lat']:.2f}N {r['lon']:.2f}E)  "
              f"reflmax {r['refl_max']:5.1f} dBZ  wmax {r['wmax']:5.1f} m/s"
              f"\n      --crop {r['lon'] - dlon:.2f} {r['lon'] + dlon:.2f} "
              f"{r['lat'] - dlat:.2f} {r['lat'] + dlat:.2f}"
              f"   (~{2 * half:.0f} km box, "
              f"~{2 * half / dx_km:.0f} cells across)")
    print("\nall frames, chronological:")
    for r in sorted(rows, key=lambda r: r["valid"]):
        print(f"  {r['valid']}  peak_{args.field} {r['peak']:8.1f}  "
              f"reflmax {r['refl_max']:5.1f}  wmax {r['wmax']:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
