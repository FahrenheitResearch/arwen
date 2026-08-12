#!/usr/bin/env python3
"""Render a real HRRR case run: NWS reflectivity, UH, 10 m wind, precip.

Reuses ``tools/da_nowcast_render.py``'s vendored basemap reader and its
rustwx Filled-style palette rather than reinventing either, and the NWS dBZ
levels/colours it already carries.  Everything a reader needs to judge the
figure is stamped ON the figure: the case, the analysis it came from, the
domain and dx, the valid time, the machine, and -- because it matters here --
the fact that the outer frame is boundary zone and not forecast.
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
sys.path.insert(0, str(REPO / "tools"))

from da_nowcast_render import (  # noqa: E402
    Basemap, LAND_FILL, REFL_COLORS, REFL_LEVELS, WATER_FILL)


UH_LEVELS = [25, 50, 75, 100, 150, 200, 300, 400]
UH_COLORS = ["#ffe9b0", "#ffd166", "#fdae4b", "#f4813a", "#e8542c",
             "#c62828", "#8e1b1b"]
PRECIP_LEVELS = [0.5, 1, 2.5, 5, 10, 15, 25, 40, 60, 90, 130]
PRECIP_COLORS = ["#d7ecf7", "#a7d5ee", "#6fb6e2", "#3f8fd0", "#3ec46b",
                 "#2a9e46", "#f5d64b", "#eea22c", "#e05a24", "#b52020"]
WIND_LEVELS = [2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30]
WIND_COLORS = ["#eaf3fa", "#cfe4f4", "#a9d0ec", "#7fb6e1", "#61c07e",
               "#43a95c", "#f2d24c", "#e79b2e", "#dd5f24", "#b81f1f"]


_LABELS = {
    "refl": "simulated composite reflectivity, REFL_10CM column max (dBZ)",
    "uh": ("instantaneous 2-5 km updraft helicity at the plotted time "
           "(m2 s-2)"),
    "wind10": "10 m wind speed (m s-1)",
    "precip": "accumulated total precipitation since model start (mm)",
    "wmax": "column maximum vertical velocity (m s-1)",
}


def _caption(meta, product, labels, reports, report_window_min):
    dxkm = meta["dx"] / 1000.0
    title = (f"ArWen {meta['nx']}x{meta['ny']}x{meta['nz']} @ dx "
             f"{dxkm:.2f} km   |   {labels[product].split(',')[0]}   |   "
             f"{meta['headline']}")
    lines = [
        ("REAL DATA. Initialised from the operational HRRR "
         f"{meta['cycle'][:16].replace('T', ' ')}Z ANALYSIS (f00, 50 native "
         "hybrid levels, NOAA archive noaa-hrrr-bdp-pds); real WPS_GEOG "
         "terrain, land use and soil; lateral boundaries from the SAME HRRR "
         "cycle's f01..f10 on their real hourly cadence."),
        ("Full physics: Morrison mp10 + MYNN PBL/surface layer + Noah-MP + "
         "RTE-RRTMGP, no cumulus parameterisation (convection-allowing). "
         f"dt {meta['dt']:g} s, {meta['nsteps']} steps. "
         "STREAMED OUT OF CORE through "
         f"{meta['tile'][0]}x{meta['tile'][1]} tiles (halo {meta['halo']}) "
         f"on one {meta['device']}; the whole domain lives in pinned host "
         f"RAM. Peak VRAM {meta['peak_vram_gib']:.1f} GiB, wall "
         f"{meta['wall_seconds'] / 60.0:.0f} min."),
    ]
    foot = ("Dashed box = the prescribed lateral-boundary frame "
            f"({meta['hard']} cells hard + {meta['taper']} cells Davies "
            "taper); nothing outside it is a forecast. "
            "This is a SIMULATED field, not an observation.")
    if reports:
        foot += ("   Open symbols are OBSERVED SPC storm reports within "
                 f"+/-{report_window_min / 2:.0f} min of the panel time: "
                 "red triangle tornado, green circle hail, blue square wind.")
    return title, lines, foot


def _read(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _grid_latlon(meta):
    from gpuwm.static.lambert import LambertGrid
    g = LambertGrid(meta["ref_lat"], meta["ref_lon"], meta["truelat1"],
                    meta["truelat2"], meta["stand_lon"],
                    meta["dx"], meta["dx"], meta["nx"] + 1, meta["ny"] + 1)
    return g.latlon_mass()


def _panel(ax, lon, lat, field, levels, colors, *, under=None, over=None):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(colors)
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)
    norm = BoundaryNorm(levels, cmap.N, clip=False)
    return ax.pcolormesh(lon, lat, field, cmap=cmap, norm=norm,
                         shading="nearest", zorder=1.5)


def _frame_box(ax, lon, lat, width, color="#111111"):
    """Outline the boundary zone, so no reader mistakes it for forecast."""
    j0 = j1 = i0 = i1 = width
    box_lon = np.concatenate([
        lon[j0, i0:lon.shape[1] - i1], lon[j0:lat.shape[0] - j1,
                                           lon.shape[1] - i1 - 1],
        lon[lat.shape[0] - j1 - 1, i0:lon.shape[1] - i1][::-1],
        lon[j0:lat.shape[0] - j1, i0][::-1]])
    box_lat = np.concatenate([
        lat[j0, i0:lat.shape[1] - i1], lat[j0:lat.shape[0] - j1,
                                           lat.shape[1] - i1 - 1],
        lat[lat.shape[0] - j1 - 1, i0:lat.shape[1] - i1][::-1],
        lat[j0:lat.shape[0] - j1, i0][::-1]])
    ax.plot(box_lon, box_lat, color=color, lw=1.1, ls=(0, (5, 3)),
            zorder=6, alpha=0.85)


def load_reports(paths, day_start):
    """SPC storm reports as ``(kind, lon, lat, datetime)``.

    The SPC daily files run 12Z to 12Z and spell the time as HHMM with no
    date, so an entry before 1200 belongs to the NEXT calendar day.  Getting
    that wrong puts the 0110Z Clay County tornado twenty-three hours away
    from the storm that produced it.
    """
    import csv
    from datetime import datetime, timedelta

    out = []
    for kind, path in paths:
        if path is None or not Path(path).is_file():
            continue
        for row in csv.DictReader(Path(path).open()):
            t = (row.get("Time") or "").strip()
            if not t.isdigit() or len(t) != 4:
                continue
            hh, mm = int(t[:2]), int(t[2:])
            when = day_start.replace(hour=0, minute=0) + timedelta(
                hours=hh, minutes=mm)
            if hh < 12:
                when += timedelta(days=1)
            try:
                out.append((kind, float(row["Lon"]), float(row["Lat"]), when))
            except (KeyError, TypeError, ValueError):
                continue
    return out


_REPORT_STYLE = {
    "torn": dict(marker="v", color="#d40000", ms=7.5, mew=1.5, z=9),
    "hail": dict(marker="o", color="#00a000", ms=4.0, mew=1.1, z=8),
    "wind": dict(marker="s", color="#1a4fd6", ms=3.6, mew=1.0, z=8),
}


def render(run_dir: Path, out_dir: Path, *, product: str, stride: int,
           zoom=None, tag: str = "", markers=(), columns: int = 4,
           frames=None, reports=(), report_window_min: float = 60.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((run_dir / "run.json").read_text())
    lat, lon = _grid_latlon(meta)
    files = sorted((run_dir / "history").glob("h_*.npz"))
    if frames is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in frames]
    files = files[::stride]
    if not files:
        raise SystemExit(f"no history frames in {run_dir}")

    if zoom is None:
        extent = (float(lon.min()), float(lon.max()),
                  float(lat.min()), float(lat.max()))
    else:
        extent = tuple(float(v) for v in zoom)
    base = Basemap(extent)

    n = len(files)
    cols = min(columns, n)
    rows = int(np.ceil(n / cols))
    aspect = ((extent[1] - extent[0])
              / max(extent[3] - extent[2], 1e-6)
              * np.cos(np.deg2rad(0.5 * (extent[2] + extent[3]))))
    panel_w = 4.9
    panel_h = panel_w / max(aspect, 0.25)
    w = panel_w * cols
    h = panel_h * rows
    labels = _LABELS
    title, lines, foot = _caption(meta, product, labels, reports,
                                  report_window_min)
    import textwrap
    wrap_at = int(max(95, 30 * cols))
    wrapped = []
    for text in lines:
        wrapped.extend(textwrap.wrap(text, wrap_at))
        wrapped.append("")
    wrapped_foot = textwrap.wrap(foot, wrap_at)
    header_in = 0.46 + 0.155 * len(wrapped) + 0.28
    footer_in = 0.92 + 0.145 * len(wrapped_foot)
    H = h + header_in + footer_in
    fig, axes = plt.subplots(rows, cols, figsize=(w, H), squeeze=False)
    mappable = None
    for k, path in enumerate(files):
        ax = axes[k // cols][k % cols]
        d = _read(path)
        valid = str(d["valid"])[:16].replace("T", " ")
        if product == "refl":
            field = np.where(d["refl_10cm"] < REFL_LEVELS[0], np.nan,
                             d["refl_10cm"])
            levels, colors = REFL_LEVELS, REFL_COLORS
        elif product == "uh":
            field = np.where(d["uh"] < UH_LEVELS[0], np.nan, d["uh"])
            levels, colors = UH_LEVELS, UH_COLORS
        elif product == "wind10":
            spd = np.hypot(d["u10"], d["v10"])
            field = np.where(spd < WIND_LEVELS[0], np.nan, spd)
            levels, colors = WIND_LEVELS, WIND_COLORS
        elif product == "precip":
            tot = d["rainnc"] + d.get("snownc", 0.0) + d.get("graupelnc", 0.0)
            field = np.where(tot < PRECIP_LEVELS[0], np.nan, tot)
            levels, colors = PRECIP_LEVELS, PRECIP_COLORS
        elif product == "wmax":
            field = np.where(d["wmax"] < 2.0, np.nan, d["wmax"])
            levels = [2, 4, 6, 8, 10, 15, 20, 25, 30, 40]
            colors = WIND_COLORS[:len(levels) - 1]
        else:
            raise SystemExit(f"unknown product {product!r}")
        base.draw_under(ax)
        mappable = _panel(ax, lon, lat, field, levels, colors,
                          over=colors[-1])
        base.draw_over(ax) if hasattr(base, "draw_over") else None
        _draw_lines(base, ax)
        _frame_box(ax, lon, lat, int(meta["hard"]) + int(meta["taper"]))
        if reports:
            from datetime import datetime, timedelta
            vt = datetime.strptime(str(d["valid"])[:19],
                                   "%Y-%m-%d %H:%M:%S")
            half = timedelta(minutes=report_window_min / 2.0)
            for kind, rlon, rlat, when in reports:
                if not (vt - half <= when <= vt + half):
                    continue
                st = _REPORT_STYLE[kind]
                ax.plot([rlon], [rlat], marker=st["marker"], ms=st["ms"],
                        mfc="none", mec=st["color"], mew=st["mew"],
                        zorder=st["z"], ls="none")
        for mlon, mlat, label in markers:
            ax.plot([mlon], [mlat], marker="v", ms=7, mfc="none",
                    mec="#000000", mew=1.6, zorder=8)
            ax.annotate(label, (mlon, mlat), textcoords="offset points",
                        xytext=(6, 4), fontsize=7.5, zorder=8,
                        color="#000000",
                        bbox=dict(fc="white", ec="none", alpha=0.65,
                                  pad=0.6))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"valid {valid}Z   F+{d['elapsed'] / 3600.0:04.1f} h",
                     fontsize=10.5)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")

    top = 1.0 - header_in / H
    bot = footer_in / H
    fig.subplots_adjust(left=0.008, right=0.992, top=top, bottom=bot,
                        wspace=0.02, hspace=0.10)
    cax = fig.add_axes([0.30, (footer_in - 0.24) / H, 0.40, 0.15 / H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                      extend="max")
    cb.set_label(labels[product], fontsize=9.5)
    cb.ax.tick_params(labelsize=8)

    def _y(inches_from_top):
        return 1.0 - inches_from_top / H

    tsize = min(13.5, max(7.5, 11.0 * w / (0.085 * max(len(title), 1) * 11.0)))
    fig.text(0.5, _y(0.24), title, ha="center", va="top", fontsize=tsize,
             weight="bold")
    y = 0.54
    for chunk in wrapped:
        if chunk:
            fig.text(0.5, _y(y), chunk, ha="center", va="top", fontsize=8.6,
                     color="#333333")
        y += 0.155
    fy = footer_in - 0.62
    for chunk in wrapped_foot:
        fig.text(0.5, fy / H, chunk, ha="center", va="top", fontsize=8.2,
                 color="#555555")
        fy -= 0.145

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"hrrrcase_{meta['case']}_{product}{tag}.png"
    out = out_dir / name
    fig.savefig(out, dpi=125, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB, "
          f"{len(files)} panels)")
    return out


def _draw_lines(base, ax):
    from matplotlib.collections import LineCollection
    from da_nowcast_render import COAST_C, COUNTY_C, NAT_C, STATE_C
    for segs, color, width, z in (
            (base.county, COUNTY_C, 0.22, 4.0),
            (base.state, STATE_C, 0.75, 5.0),
            (base.nat, NAT_C, 0.75, 5.0),
            (base.coast, COAST_C, 0.8, 5.0)):
        if segs:
            ax.add_collection(LineCollection(
                segs, colors=[color], linewidths=width, zorder=z))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--product", default="refl")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--columns", type=int, default=4)
    ap.add_argument("--zoom", type=float, nargs=4, default=None,
                    metavar=("LON0", "LON1", "LAT0", "LAT1"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--markers", default=None,
                    help="JSON list of [lon, lat, label]")
    ap.add_argument("--reports-dir", type=Path, default=None,
                    help="directory holding SPC <stem>_rpts_{torn,hail,wind}"
                         ".csv files")
    ap.add_argument("--reports-stem", default=None)
    ap.add_argument("--report-window-min", type=float, default=60.0)
    args = ap.parse_args(argv)
    markers = json.loads(args.markers) if args.markers else []
    reports = ()
    if args.reports_dir and args.reports_stem:
        from datetime import datetime
        meta = json.loads((args.run / "run.json").read_text())
        day = datetime.strptime(meta["start"][:10], "%Y-%m-%d")
        reports = load_reports(
            [(k, args.reports_dir / f"{args.reports_stem}_{k}.csv")
             for k in ("torn", "hail", "wind")], day)
        print(f"{len(reports)} SPC reports loaded")
    render(args.run, args.out, product=args.product, stride=args.stride,
           zoom=args.zoom, tag=args.tag, markers=markers,
           columns=args.columns, frames=args.frames, reports=reports,
           report_window_min=args.report_window_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
