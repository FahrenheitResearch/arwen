#!/usr/bin/env python3
"""Score one real-case run against the SPC storm reports of that day.

The claim a convection-allowing model started hours out can honestly make is
MESOSCALE: convection in roughly the right region, in roughly the right
window, with roughly the right mode and coverage.  It cannot claim the right
county at the right minute, and a metric that pretends otherwise is worse
than no metric.  So this reports three things and nothing else:

``nearest``
    For every observed report, the great-circle distance from it to the
    NEAREST simulated cell exceeding a threshold, at the history frame
    closest in time.  A distribution of those distances is the honest
    statement of "how far off was it".

``coverage``
    The simulated area over the threshold inside a stated verification box,
    per frame, beside the observed report count in the same box and window.
    Too little is a miss; far too much is over-forecasting, and a figure of
    reflectivity alone hides the second.

``timing``
    The first frame at which the simulated threshold area inside the box
    exceeds a floor, against the time of the first observed report in it.

Distances are computed on the model's own lat/lon, so no projection is
assumed and no regridding happens.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EARTH_KM = 6371.0088


def _haversine_km(lat0, lon0, lat, lon):
    p0 = np.deg2rad(lat0)
    p1 = np.deg2rad(lat)
    dp = p1 - p0
    dl = np.deg2rad(lon - lon0)
    a = (np.sin(dp / 2.0) ** 2
         + np.cos(p0) * np.cos(p1) * np.sin(dl / 2.0) ** 2)
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _cell_area_km2(lat, dx_m):
    """Map-projected cell area.  On a Lambert secant cone at 3 km the map
    factor is within a few percent of 1 over CONUS, so the flat dx*dy is used
    and the residual is smaller than the thresholding noise."""
    return (dx_m / 1000.0) ** 2 * np.ones_like(lat)


def score(run_dir: Path, reports, *, box, refl_threshold=40.0,
          uh_threshold=50.0, area_floor_km2=2000.0):
    from tilestream.render_case import _grid_latlon, _read

    meta = json.loads((run_dir / "run.json").read_text())
    lat, lon = _grid_latlon(meta)
    files = sorted((run_dir / "history").glob("h_*.npz"))
    frames = []
    for path in files:
        d = _read(path)
        frames.append((datetime.strptime(str(d["valid"])[:19],
                                         "%Y-%m-%d %H:%M:%S"), d))
    if not frames:
        raise SystemExit(f"no history frames in {run_dir}")

    lon0, lon1, lat0, lat1 = box
    inbox = ((lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1))
    cell_km2 = (meta["dx"] / 1000.0) ** 2
    # The boundary frame is not forecast and must not be scored.
    edge = int(meta["hard"]) + int(meta["taper"])
    interior = np.zeros_like(inbox)
    interior[edge:-edge, edge:-edge] = True
    inbox = inbox & interior

    out = {"case": meta["case"], "box": list(box),
           "refl_threshold_dbz": refl_threshold,
           "uh_threshold_m2s2": uh_threshold,
           "frames": [], "nearest": [], "timing": {}}

    for when, d in frames:
        refl = d["refl_10cm"]
        uh = d.get("uh")
        row = {
            "valid": when.isoformat(),
            "refl_area_km2": float((refl >= refl_threshold)[inbox].sum()
                                   * cell_km2),
            "refl_max_dbz": float(np.nanmax(np.where(inbox, refl, -99))),
            "wmax_ms": float(np.nanmax(np.where(inbox, d["wmax"], -99))),
        }
        if uh is not None:
            row["uh_area_km2"] = float((uh >= uh_threshold)[inbox].sum()
                                       * cell_km2)
            row["uh_max_m2s2"] = float(np.nanmax(np.where(inbox, uh, -99)))
        out["frames"].append(row)

    for kind, rlon, rlat, when in reports:
        if not (lon0 <= rlon <= lon1 and lat0 <= rlat <= lat1):
            continue
        if not (frames[0][0] <= when <= frames[-1][0]):
            continue
        idx = min(range(len(frames)),
                  key=lambda k: abs((frames[k][0] - when).total_seconds()))
        wt, d = frames[idx]
        entry = {"kind": kind, "lon": rlon, "lat": rlat,
                 "observed": when.isoformat(), "frame": wt.isoformat(),
                 "dt_minutes": (wt - when).total_seconds() / 60.0}
        dist = _haversine_km(rlat, rlon, lat, lon)
        for name, field, thr in (("refl", d["refl_10cm"], refl_threshold),
                                 ("uh", d.get("uh"), uh_threshold)):
            if field is None:
                continue
            mask = (field >= thr) & interior
            entry[f"nearest_{name}_km"] = (float(dist[mask].min())
                                           if mask.any() else None)
        out["nearest"].append(entry)

    first_sim = next((r["valid"] for r in out["frames"]
                      if r["refl_area_km2"] >= area_floor_km2), None)
    inbox_reports = [w for _k, rl, ra, w in reports
                     if lon0 <= rl <= lon1 and lat0 <= ra <= lat1]
    out["timing"] = {
        "area_floor_km2": area_floor_km2,
        "first_frame_over_floor": first_sim,
        "first_observed_report": (min(inbox_reports).isoformat()
                                  if inbox_reports else None),
        "observed_reports_in_box": len(inbox_reports),
    }
    for name in ("refl", "uh"):
        vals = [e[f"nearest_{name}_km"] for e in out["nearest"]
                if e.get(f"nearest_{name}_km") is not None]
        if vals:
            v = np.array(vals)
            out[f"nearest_{name}_summary"] = {
                "n": int(v.size), "median_km": float(np.median(v)),
                "p25_km": float(np.percentile(v, 25)),
                "p75_km": float(np.percentile(v, 75)),
                "within_50km_frac": float((v <= 50).mean()),
                "within_100km_frac": float((v <= 100).mean()),
            }
    return out


def main(argv=None) -> int:
    from tilestream.render_case import load_reports

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--reports-stem", required=True)
    ap.add_argument("--box", type=float, nargs=4, required=True,
                    metavar=("LON0", "LON1", "LAT0", "LAT1"))
    ap.add_argument("--kinds", nargs="*", default=["torn", "hail", "wind"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    meta = json.loads((args.run / "run.json").read_text())
    day = datetime.strptime(meta["start"][:10], "%Y-%m-%d")
    reports = load_reports(
        [(k, args.reports_dir / f"{args.reports_stem}_{k}.csv")
         for k in args.kinds], day)
    result = score(args.run, reports, box=tuple(args.box))
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
