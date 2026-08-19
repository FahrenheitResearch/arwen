"""Render a streamed real-case forecast, on ArWen's own product-map frame.

DEPRECATED FALLBACK, under the render law (CLAUDE.md, Drew 2026-08-06).
The streamed frames this reads now become real wrfouts by default
(``tilestream/run_bigdomain.py`` -> :mod:`gpuwm.io.surface_wrfout`), so
the field below is drawable by ``rw_wrfbatch``:
``tilestream/bigdomain_render.py --engine rust`` renders the whole
catalog of a streamed run, and ``python -m tilestream.render_case_rust``
renders a case run.  MEASURED 2026-08-17: 22 products per frame.

The ONLY reason this file still draws is its shape -- ``figure_sequence``
is an N-column CONTACT SHEET of many frames, and the renderer composes
one panel.  Nothing here should acquire a new single-panel product.

Reuses the repo's rendering machinery rather than inventing a look:

* the basemap geometry is the vendored Natural Earth 10 m + US Census county
  set the Rust renderer draws from (:func:`gpuwm.rustwx.basemap_dir`), and the
  clip/stack/colour rules are ``tools/da_nowcast_render.Basemap``'s;
* the reflectivity ramp is the NWS dBZ table ``tools/da_nowcast_render``
  already uses for radar-grid panels -- the same 5 dBZ steps to 70;
* every panel carries the rusty-weather title block, ``Init MM/DD HHZ |
  +HH:MM | Valid MM/DD HH:MMZ``, plus a provenance footer naming the
  analysis, the domain, the grid spacing, the machine and what is model and
  what is observation.

    python -m tilestream.realcase_render DUMPDIR OUTDIR --tag superoutbreak

HONESTY, on the figure and not only here: the plotted field is
``REFL_10CM``'s column maximum -- a MODEL diagnostic computed by
``gpuwm.core.refl.compute_refl_10cm`` from the model's own hydrometeors.  It
is not radar.  Where an observed counterpart is drawn it is drawn from the
SPC storm-report record and labelled as such.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# NWS dBZ ramp -- tools/da_nowcast_render.REFL_LEVELS/REFL_COLORS verbatim.
REFL_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
REFL_COLORS = ["#a6d8f0", "#5db5e8", "#2a83c8", "#46c846", "#2ea02e",
               "#1c781c", "#f8f83c", "#e6c22e", "#f09a22", "#ea5f1c",
               "#d42020", "#b01616", "#e83cc8"]

LAND_FILL = "#EEEDE6"
WATER_FILL = "#E0EAF2"
COUNTY_C = (108 / 255, 116 / 255, 128 / 255)
LAKE_EDGE = (118 / 255, 136 / 255, 154 / 255, 220 / 255)
STATE_C = (20 / 255, 24 / 255, 30 / 255)
NAT_C = (74 / 255, 82 / 255, 96 / 255)
COAST_C = (32 / 255, 40 / 255, 50 / 255)
INK = "#16202c"
MUTED = "#5b6672"


def _parts(shape):
    pts = np.asarray(shape.points, float)
    if pts.size == 0:
        return []
    idx = list(shape.parts) + [len(pts)]
    return [pts[idx[k]:idx[k + 1]] for k in range(len(idx) - 1)
            if idx[k + 1] - idx[k] >= 2]


class Basemap:
    """Clipped basemap geometry for one lat/lon extent, vendored assets only."""

    def __init__(self, extent):
        import shapefile
        from gpuwm.rustwx import basemap_dir

        self.extent = extent
        assets = basemap_dir()
        if not assets.is_dir():
            raise SystemExit(f"vendored basemap assets missing at {assets}")

        def collect(rel, *, vertex_clip=False):
            base = assets / rel
            reader = shapefile.Reader(shp=open(f"{base}.shp", "rb"),
                                      shx=open(f"{base}.shx", "rb"))
            x0, x1, y0, y1 = extent
            segs = []
            for shape in reader.shapes():
                if getattr(shape, "shapeTypeName", "") == "NULL":
                    continue
                bx0, by0, bx1, by1 = shape.bbox
                if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                    continue
                for part in _parts(shape):
                    if vertex_clip and not (
                            (part[:, 0] >= x0) & (part[:, 0] <= x1)
                            & (part[:, 1] >= y0) & (part[:, 1] <= y1)).any():
                        continue
                    segs.append(part)
            return segs

        self.county = collect("us_counties_5m/cb_2023_us_county_5m")
        self.state = collect(
            "natural_earth_10m/ne_10m_admin_1_states_provinces_lines")
        self.nat = collect(
            "natural_earth_10m/ne_10m_admin_0_boundary_lines_land")
        self.coast = collect("natural_earth_10m/ne_10m_coastline",
                             vertex_clip=True)
        self.lakes = collect("natural_earth_10m/ne_10m_lakes")
        self.all_land = not self.coast
        self.land = [] if self.all_land else collect(
            "natural_earth_10m/ne_10m_land")

    def draw_under(self, ax) -> None:
        from matplotlib.collections import PolyCollection
        if self.all_land:
            ax.set_facecolor(LAND_FILL)
        else:
            ax.set_facecolor(WATER_FILL)
            if self.land:
                ax.add_collection(PolyCollection(
                    self.land, facecolors=LAND_FILL, edgecolors="none",
                    zorder=0.5))
        if self.lakes:
            ax.add_collection(PolyCollection(
                self.lakes, facecolors=WATER_FILL, edgecolors="none",
                zorder=0.8))

    def draw_over(self, ax, *, counties=True) -> None:
        from matplotlib.collections import LineCollection
        layers = [(self.lakes, LAKE_EDGE, 0.5, 3.1),
                  (self.state, STATE_C, 0.9, 3.2),
                  (self.nat, NAT_C, 0.8, 3.3),
                  (self.coast, COAST_C, 0.8, 3.4)]
        if counties:
            layers.insert(0, (self.county, COUNTY_C, 0.25, 3.0))
        for segs, color, width, z in layers:
            if segs:
                kw = {"alpha": 0.55} if segs is self.county else {}
                ax.add_collection(LineCollection(
                    segs, colors=[color], linewidths=width, zorder=z, **kw))


# --------------------------------------------------------------------------

def load(dumpdir: Path):
    frames = []
    for path in sorted(Path(dumpdir).glob("real_*_f*.npz")):
        with np.load(path, allow_pickle=False) as z:
            frames.append({k: z[k] for k in z.files})
        frames[-1]["_path"] = str(path)
    if not frames:
        raise SystemExit(f"no real_*_f*.npz dumps in {dumpdir}")
    frames.sort(key=lambda f: float(f["elapsed_s"]))
    return frames


def _stamp(frame):
    init = datetime.fromisoformat(str(frame["init_time"]))
    valid = init + timedelta(seconds=float(frame["elapsed_s"]))
    lead = float(frame["elapsed_s"]) / 3600.0
    return (f"Init {init:%m/%d %HZ}  |  +{int(lead):02d}:"
            f"{int(round((lead % 1) * 60)):02d}  |  "
            f"Valid {valid:%m/%d %H:%M}Z"), valid


def _refl_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(REFL_COLORS)
    cmap.set_under((0, 0, 0, 0))
    cmap.set_over(REFL_COLORS[-1])
    return cmap, BoundaryNorm(REFL_LEVELS, cmap.N)


def _footer(fig, meta, extra=""):
    fig.text(0.5, 0.012,
             f"{meta['footer']}{extra}", ha="center", va="bottom",
             fontsize=7.4, color=MUTED)


def panel_refl(ax, frame, base, meta, *, counties=True, title=None,
               reports=None):
    cmap, norm = _refl_cmap()
    lon, lat = frame["LON"], frame["LAT"]
    refl = np.asarray(frame["REFL_COMPOSITE"], dtype=np.float32)
    base.draw_under(ax)
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_less(refl, REFL_LEVELS[0]),
                         cmap=cmap, norm=norm, shading="nearest", zorder=2)
    base.draw_over(ax, counties=counties)
    if reports is not None and len(reports):
        ax.plot(reports[:, 1], reports[:, 0], linestyle="none", marker="v",
                markersize=3.4, markerfacecolor="none",
                markeredgecolor="#111111", markeredgewidth=0.7, zorder=5)
    ax.set_xlim(base.extent[0], base.extent[1])
    ax.set_ylim(base.extent[2], base.extent[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(lat)))))
    ax.set_xticks([])
    ax.set_yticks([])
    stamp, _ = _stamp(frame)
    ax.set_title(title if title is not None else stamp, fontsize=9,
                 color=INK, pad=3)
    return mesh


def figure_sequence(frames, meta, out, *, ncols=4, extent=None,
                    counties=False, reports=None, stride=1, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = frames[::stride]
    nrows = int(np.ceil(len(picks) / ncols))
    ext = extent or meta["extent"]
    base = Basemap(ext)
    aspect = ((ext[1] - ext[0]) * np.cos(np.deg2rad(0.5 * (ext[2] + ext[3])))
              / (ext[3] - ext[2]))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.15 * ncols * max(aspect, 0.55),
                                      3.15 * nrows + 1.5))
    axes = np.atleast_1d(axes).ravel()
    mesh = None
    for ax, frame in zip(axes, picks):
        rep = None
        if reports is not None:
            _, valid = _stamp(frame)
            rep = reports_window(reports, valid, minutes=30)
        mesh = panel_refl(ax, frame, base, meta, counties=counties,
                          reports=rep)
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(title or meta["title"], fontsize=13, y=0.985, color=INK)
    fig.text(0.5, 0.955, meta["subtitle"], ha="center", fontsize=9,
             color=MUTED)
    cax = fig.add_axes([0.25, 0.055, 0.5, 0.012])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                      ticks=REFL_LEVELS)
    cb.set_label("simulated composite reflectivity, column-max REFL_10CM "
                 "(dBZ) -- MODEL, not radar", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.10,
                        wspace=0.02, hspace=0.10)
    _footer(fig, meta)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def reports_window(reports, valid, *, minutes=30):
    if reports is None:
        return None
    lo = valid - timedelta(minutes=minutes)
    hi = valid + timedelta(minutes=minutes)
    keep = [(r["lat"], r["lon"]) for r in reports
            if lo <= r["time"] <= hi]
    return np.asarray(keep, dtype=float) if keep else np.zeros((0, 2))


def figure_single(frame, meta, out, *, extent=None, counties=True,
                  reports=None, note=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = extent or meta["extent"]
    base = Basemap(ext)
    aspect = ((ext[1] - ext[0]) * np.cos(np.deg2rad(0.5 * (ext[2] + ext[3])))
              / (ext[3] - ext[2]))
    fig, ax = plt.subplots(figsize=(9.5 * max(aspect, 0.6) + 1.0, 9.5))
    _, valid = _stamp(frame)
    rep = reports_window(reports, valid, minutes=30) if reports else None
    mesh = panel_refl(ax, frame, base, meta, counties=counties, reports=rep)
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.032,
                      pad=0.012, ticks=REFL_LEVELS)
    cb.set_label("composite REFL_10CM (dBZ) -- MODEL", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(meta["title"], fontsize=13, color=INK, y=0.975)
    extra = ""
    if rep is not None and len(rep):
        extra = (f"   |   open triangles: {len(rep)} SPC storm reports "
                 f"within +/-30 min of valid time (OBSERVED)")
    if note:
        extra += "   |   " + note
    _footer(fig, meta, extra)
    fig.subplots_adjust(left=0.02, right=0.94, top=0.93, bottom=0.05)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def figure_fields(frame, meta, out, *, extent=None):
    """UH, 10 m wind and accumulated precipitation beside the reflectivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ext = extent or meta["extent"]
    base = Basemap(ext)
    lon, lat = frame["LON"], frame["LAT"]
    aspect = ((ext[1] - ext[0]) * np.cos(np.deg2rad(0.5 * (ext[2] + ext[3])))
              / (ext[3] - ext[2]))
    fig, axes = plt.subplots(2, 2, figsize=(7.0 * max(aspect, 0.6) + 1.6,
                                            13.0))
    axes = axes.ravel()

    def frame_ax(ax, title):
        base.draw_over(ax, counties=False)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(lat)))))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, color=INK, pad=3)

    cmap, norm = _refl_cmap()
    base.draw_under(axes[0])
    m0 = axes[0].pcolormesh(
        lon, lat, np.ma.masked_less(frame["REFL_COMPOSITE"], 5),
        cmap=cmap, norm=norm, shading="nearest", zorder=2)
    frame_ax(axes[0], "composite REFL_10CM (dBZ)")
    fig.colorbar(m0, ax=axes[0], fraction=0.03, pad=0.01, ticks=REFL_LEVELS
                 ).ax.tick_params(labelsize=6.5)

    uh_levels = [25, 50, 75, 100, 150, 200, 300, 400]
    uh_colors = ["#cfe3f5", "#8fc0e8", "#4f97d6", "#f4d24a", "#f09a22",
                 "#e2521c", "#b01616"]
    ucmap = ListedColormap(uh_colors)
    ucmap.set_under((0, 0, 0, 0))
    unorm = BoundaryNorm(uh_levels, ucmap.N)
    base.draw_under(axes[1])
    m1 = axes[1].pcolormesh(lon, lat,
                            np.ma.masked_less(frame["UH25"], uh_levels[0]),
                            cmap=ucmap, norm=unorm, shading="nearest",
                            zorder=2)
    frame_ax(axes[1], "2-5 km updraft helicity (m$^2$ s$^{-2}$)")
    fig.colorbar(m1, ax=axes[1], fraction=0.03, pad=0.01, ticks=uh_levels
                 ).ax.tick_params(labelsize=6.5)

    spd = np.hypot(np.asarray(frame["U10"], dtype=np.float32),
                   np.asarray(frame["V10"], dtype=np.float32))
    base.draw_under(axes[2])
    m2 = axes[2].pcolormesh(lon, lat, spd, cmap="YlGnBu", vmin=0, vmax=30,
                            shading="nearest", zorder=2)
    frame_ax(axes[2], "10 m wind speed (m s$^{-1}$)")
    fig.colorbar(m2, ax=axes[2], fraction=0.03, pad=0.01
                 ).ax.tick_params(labelsize=6.5)

    rain = np.asarray(frame.get("RAINNC", np.zeros_like(spd)),
                      dtype=np.float32)
    if "RAINC" in frame:
        rain = rain + np.asarray(frame["RAINC"], dtype=np.float32)
    plevels = [0.5, 1, 2.5, 5, 10, 20, 35, 50, 75, 100, 150]
    pcolors = ["#d9f0d3", "#a6dba0", "#5aae61", "#1b7837", "#c7e9f0",
               "#67a9cf", "#2166ac", "#f7cb44", "#f08c22", "#d6301c"]
    pcmap = ListedColormap(pcolors)
    pcmap.set_under((0, 0, 0, 0))
    pnorm = BoundaryNorm(plevels, pcmap.N)
    base.draw_under(axes[3])
    m3 = axes[3].pcolormesh(lon, lat, np.ma.masked_less(rain, plevels[0]),
                            cmap=pcmap, norm=pnorm, shading="nearest",
                            zorder=2)
    frame_ax(axes[3], "accumulated precipitation since init (mm)")
    fig.colorbar(m3, ax=axes[3], fraction=0.03, pad=0.01, ticks=plevels
                 ).ax.tick_params(labelsize=6.5)

    stamp, _ = _stamp(frame)
    fig.suptitle(meta["title"], fontsize=13, color=INK, y=0.985)
    fig.text(0.5, 0.962, stamp + "   |   " + meta["subtitle"], ha="center",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.945, bottom=0.045,
                        wspace=0.06, hspace=0.07)
    _footer(fig, meta)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def figure_uh_swath(frames, meta, out, *, extent=None, reports=None):
    """The run's whole UH track swath against the observed tornado reports.

    The single most useful verification picture for an outbreak: a
    convection-allowing model's UH tracks are what a forecaster reads as
    "where the rotating storms went", and the SPC report points are where
    the tornadoes actually were.  Both are on the same map, at the same
    scale, with no time matching -- the swath is the maximum over the whole
    forecast and the reports are the whole day.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ext = extent or meta["extent"]
    base = Basemap(ext)
    swath = np.zeros_like(np.asarray(frames[0]["UH25"], dtype=np.float32))
    for frame in frames:
        swath = np.maximum(swath, np.asarray(frame["UH25"],
                                             dtype=np.float32))
    lon, lat = frames[0]["LON"], frames[0]["LAT"]
    aspect = ((ext[1] - ext[0]) * np.cos(np.deg2rad(0.5 * (ext[2] + ext[3])))
              / (ext[3] - ext[2]))
    fig, ax = plt.subplots(figsize=(10.0 * max(aspect, 0.6) + 1.2, 10.0))
    levels = [25, 50, 75, 100, 150, 200, 300, 500]
    colors = ["#cfe3f5", "#8fc0e8", "#4f97d6", "#f4d24a", "#f09a22",
              "#e2521c", "#b01616"]
    cmap = ListedColormap(colors)
    cmap.set_under((0, 0, 0, 0))
    norm = BoundaryNorm(levels, cmap.N)
    base.draw_under(ax)
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_less(swath, levels[0]),
                         cmap=cmap, norm=norm, shading="nearest", zorder=2)
    base.draw_over(ax, counties=False)
    n = 0
    if reports:
        pts = np.asarray([(r["lat"], r["lon"]) for r in reports], dtype=float)
        n = len(pts)
        ax.plot(pts[:, 1], pts[:, 0], linestyle="none", marker="v",
                markersize=4.2, markerfacecolor="none",
                markeredgecolor="#111111", markeredgewidth=0.8, zorder=5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(lat)))))
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(mesh, ax=ax, fraction=0.032, pad=0.012, ticks=levels)
    cb.set_label("forecast-maximum 2-5 km updraft helicity (m$^2$ s$^{-2}$)"
                 " -- MODEL", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(meta["title"], fontsize=13, color=INK, y=0.975)
    _footer(fig, meta,
            f"   |   open triangles: {n} OBSERVED SPC tornado reports, "
            f"whole event")
    fig.subplots_adjust(left=0.02, right=0.94, top=0.93, bottom=0.05)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def load_reports(path):
    if path is None:
        return None
    out = []
    with open(path) as fh:
        for row in json.load(fh):
            out.append({"time": datetime.fromisoformat(row["time"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumpdir")
    ap.add_argument("outdir")
    ap.add_argument("--tag", default="superoutbreak20110427")
    ap.add_argument("--machine", default="unknown")
    ap.add_argument("--reports", default=None)
    ap.add_argument("--zoom", default=None,
                    help="lon0,lon1,lat0,lat1 for the close-up figures")
    args = ap.parse_args()

    frames = load(args.dumpdir)
    f0 = frames[0]
    lat, lon = f0["LAT"], f0["LON"]
    nx, ny, dx = int(f0["nx"]), int(f0["ny"]), float(f0["dx"])
    init = datetime.fromisoformat(str(f0["init_time"]))
    meta = {
        "title": "2011-04-27 SUPER OUTBREAK -- ArWen, streamed out-of-core, "
                 f"{nx}x{ny} at {dx/1000:.0f} km",
        "subtitle": (f"initialised from ERA5 reanalysis {init:%Y-%m-%d %HZ}, "
                     f"real 3-hourly ERA5 lateral boundaries"),
        "footer": (f"ArWen (gpuwm) full physics: Morrison mp10 + MYNN + "
                   f"Noah-MP + RTE-RRTMGP, no cumulus parameterisation "
                   f"(convection-allowing)  |  domain {nx}x{ny}x"
                   f"{int(f0['nz'])} at dx = {dx/1000:.0f} km, dt = "
                   f"{float(f0['dt']):.0f} s  |  out-of-core tiled stream, "
                   f"{args.machine}"),
        "extent": (float(lon.min()), float(lon.max()),
                   float(lat.min()), float(lat.max())),
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reports = load_reports(args.reports)

    zoom = None
    if args.zoom:
        zoom = tuple(float(v) for v in args.zoom.split(","))

    stride = max(1, len(frames) // 12)
    figure_sequence(frames, meta, outdir / f"{args.tag}_refl_sequence.png",
                    ncols=4, stride=stride, reports=reports)
    if zoom:
        figure_sequence(frames, meta,
                        outdir / f"{args.tag}_refl_sequence_zoom.png",
                        ncols=4, stride=stride, extent=zoom, counties=True,
                        reports=reports,
                        title=meta["title"] + "  -- outbreak region")
    peak = max(frames, key=lambda f: float(np.nanmax(f["UH25"])))
    figure_single(peak, meta, outdir / f"{args.tag}_refl_peak.png",
                  extent=zoom, counties=True, reports=reports)
    figure_fields(peak, meta, outdir / f"{args.tag}_fields_peak.png",
                  extent=zoom)
    figure_uh_swath(frames, meta, outdir / f"{args.tag}_uh_swath.png",
                    extent=zoom, reports=reports)


if __name__ == "__main__":
    main()
