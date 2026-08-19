"""Map-styled gallery for a ``tools/da_nowcast.py`` case directory.

DEPRECATED FALLBACK for its weather fields, under the render law
(CLAUDE.md, Drew 2026-08-06).  Every WEATHER FIELD this module draws is
now drawable by the production Rust renderer, MEASURED 2026-08-17 on this
lane's own real inputs:

* a leg's column-max reflectivity composite -- ``tools/da_cycle_prepared``
  writes a real wrfout beside every ``.npz`` it saves (default-on), and
  ``gpuwm render --engine rust`` draws it.  Proven on an archived case
  that predates that default by rebuilding the georeference from the
  case's own ``authority/experiment.toml``: 11 leg composites of
  ``nowcast_kdvn_202608072200``, 33 production panels;
* an observation file's own panels -- ``rw_obsgrid`` reads
  ``gpuwm-obs.radar-grid.v1`` natively (column-max Z, coverage depth,
  distinct-radar overlap, lowest-tilt radial velocity), proven on a real
  three-radar file built by the production spine from archive Level-II
  volumes.

WHAT KEEPS THIS MODULE ALIVE, so it is not a silent second tier: every
figure below is a MULTI-PANEL COMPOSITION (a lead ROW, a per-cycle
STRIP, verification PAIRS) or an analysis chart, and the renderer's
``MapRenderRequest`` composes exactly one panel.  Until there is a
composition primitive -- ``gpuwm/pair_compose.py`` pastes finished PNGs
with Pillow and is the shape one would take -- the panels inside these
sheets are drawn here because the SHEET is drawn here.  Nothing in this
file should acquire a new single-panel product; that belongs to the
binaries above.

The analysis charts (:meth:`fig_numbers`, :meth:`fig_scorecard`) are NOT
demoted: the render law permits matplotlib for charts that are not
weather fields, and those two are innovation/spread lines and a
verification table.

Draws every field panel on the ArWen product-map frame, REUSING the
repo's rendering machinery rather than reinventing it:

* basemap geometry: the vendored assets the rust renderer draws from
  (``tools/rustwx/assets/basemap`` -- Natural Earth 10m + US Census
  counties; located via :func:`gpuwm.rustwx.basemap_dir`);
* layer palette and stacking: ``rustwx-render``'s Filled-style
  constants (land #EEEDE6, water #E0EAF2; counties -> lakes -> states
  -> national -> coast, linework drawn ON TOP of the data fill);
* title block: the rusty-weather convention
  ``Init MM/DD HHZ | +HHH:MM | Valid MM/DD HH:MMZ | ArWen | Δx N km``
  with the source label in the footer.

Layout read from the case directory (the front door's own):

    authority/experiment.toml   init time, case name
    cycle/cycle-report.json     members, legs, per-cycle numbers
    cycle/composites/           per-leg per-member column-max npz
    obs/*.nc                    one radar-grid file per applied cycle
    obsverify/*.nc              optional after-the-fact verification obs
    gallery/                    output (index.html + PNGs, replaced
                                in place on re-render)

HONESTY: demo-grade nowcast output, UNSCORED; free-forecast panels are
stamped "PAST LAST OBS" until an observed counterpart exists, and the
verification numbers (>=35 dBZ column counts in the echo mask, and
FSS(30 dBZ, 27 km) via :func:`gpuwm.verify.field_metrics.fss_distance`)
are labeled demo-grade on the figure.

No radar-site names in this file, its defaults, or its identifiers; the
site id is read from the observation files' own metadata.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

REFL_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
REFL_COLORS = ["#a6d8f0", "#5db5e8", "#2a83c8", "#46c846", "#2ea02e",
               "#1c781c", "#f8f83c", "#e6c22e", "#f09a22", "#ea5f1c",
               "#d42020", "#b01616", "#e83cc8"]

# rustwx-render Filled-style palette (features.rs constants).
LAND_FILL = "#EEEDE6"
WATER_FILL = "#E0EAF2"
COUNTY_C = (108 / 255, 116 / 255, 128 / 255)
LAKE_EDGE = (118 / 255, 136 / 255, 154 / 255, 220 / 255)
STATE_C = (20 / 255, 24 / 255, 30 / 255)
NAT_C = (74 / 255, 82 / 255, 96 / 255)
COAST_C = (32 / 255, 40 / 255, 50 / 255)

#: How far an observation file's requested time may sit from a
#: leg's end and still be that leg's observation.  Generous
#: enough for the few minutes a fetch may land off its request,
#: tight enough that a file for a different cycle never matches.
OBS_MATCH_TOLERANCE_S = 300.0

FSS_BOX_KM = 27.0
FSS_THRESHOLD_DBZ = 30.0
MISSING_OBS_FILL_DBZ = -35.0

#: The column census threshold, reported beside every FSS: how many
#: columns reach convective intensity, inside the observed echo mask.
#: A separate question from the FSS threshold and a separate number, so
#: it gets its own name rather than three bare literals per scorer --
#: which is how it lived until a second scorer declared its own copy.
COLUMN_THRESHOLD_DBZ = 35.0

#: Height in inches of the band reserved at the bottom of every figure
#: for the credit line.  Reserved GEOMETRY, not hope: the layout engine
#: is told to keep the axes out of it, so the footer can never overprint
#: a tick label however short the figure is.
FOOTER_BAND_IN = 0.30


def footer_band_fraction(fig_height_in: float) -> float:
    """The reserved footer band as a fraction of one figure's height.

    Capped so a very short figure keeps most of its canvas for data
    (the band matters most on wide, short figures, where the default
    layout otherwise runs the axes' tick labels straight into it).
    """

    if fig_height_in <= 0.0:
        raise ValueError("figure height must be positive")
    return min(0.12, FOOTER_BAND_IN / fig_height_in)


#: Average glyph advance as a fraction of the point size, for the sans
#: faces this gallery draws with.  Used to size text that MUST fit a
#: known width -- a figure whose panel count changes with the run
#: changes width underneath every fixed font size, and a title that
#: runs off both edges is not a title.
#:
#: MEASURED off a rendered figure rather than assumed: at 0.5 the credit
#: lines were computed to fit and overprinted each other on a two-panel
#: figure.  The credit line came out at 0.583 and the source line at
#: 0.517 of their point sizes on the 2026-08-05 gallery, so this is set
#: above both -- the estimate may only ever shrink text that would have
#: fitted, never let text overflow.
GLYPH_ADVANCE = 0.60

#: Clear space kept between the two credit lines when they share a row.
CREDIT_GAP_IN = 0.4


def radar_ids(variable) -> list[str]:
    """Every contributing radar's id, in the file's own order.

    ``gpuwm-obs.radar-grid.v1`` carries ``radar_id`` as (radar, strlen)
    characters, one row per radar that fed the analysis.  The first row
    is the anchor.
    """

    import numpy as np

    return [b"".join(np.ma.getdata(row)).decode().strip()
            for row in variable]


def site_label(ids) -> str:
    """How the figures name the radars behind an observed composite.

    With one radar this is that radar's id, so single-radar galleries
    are unchanged.  With several it names them all: a composite built
    from three radars and captioned with one radar's id would understate
    the analysis it came from, and saying what an observation actually
    is happens to be this gallery's entire job.
    """

    return "+".join(ids)


def text_width_in(text: str, points: float) -> float:
    return len(text) * GLYPH_ADVANCE * points / 72.0


def credit_layout(fig_width_in: float, foot: str, src: str) -> dict:
    """How to lay out the two credit lines so they always fit.

    Left-and-right on one line while there is room for both, shrinking
    a little first; stacked when there is not.  A gallery that redraws
    itself every few minutes cannot be checked by eye each time, so the
    layout has to be right for whatever width the run produces rather
    than for the width it was written against.
    """

    for points in (6.5, 6.0, 5.5, 5.0):
        if (text_width_in(foot, points - 0.5)
                + text_width_in(src, points)
                + CREDIT_GAP_IN) <= fig_width_in:
            return {"mode": "sides", "foot_pt": points - 0.5,
                    "src_pt": points}
    return {"mode": "stacked", "foot_pt": 5.0, "src_pt": 5.0}


def fit_title(text: str, fig_width_in: float, *, points: float = 12.0,
              min_points: float = 8.5) -> tuple[str, float]:
    """A title sized -- and if need be wrapped -- to the figure it is on."""

    import textwrap

    usable = max(1.0, fig_width_in - 0.3)
    if text_width_in(text, points) <= usable:
        return text, points
    shrunk = max(min_points, usable * 72.0 / (len(text) * GLYPH_ADVANCE))
    if text_width_in(text, shrunk) <= usable:
        return text, shrunk
    per_line = max(20, int(usable * 72.0 / (GLYPH_ADVANCE * min_points)))
    return "\n".join(textwrap.wrap(text, per_line)), min_points


def _parts(shape):
    import numpy as np
    pts = np.asarray(shape.points, float)
    if pts.size == 0:
        return []
    idx = list(shape.parts) + [len(pts)]
    return [pts[idx[k]:idx[k + 1]] for k in range(len(idx) - 1)
            if idx[k + 1] - idx[k] >= 2]


class Basemap:
    """Clipped basemap geometry for one extent, vendored assets only."""

    def __init__(self, extent: tuple[float, float, float, float]):
        # The refusal before the import, not the import's own traceback:
        # reaching a bare `import shapefile` here is how a whole DA
        # nowcast ended in `ModuleNotFoundError: No module named
        # 'shapefile'` with no remedy.  tools/da_nowcast.py refuses this
        # at its front door before the forecast; this is the same
        # sentence for anyone who runs the renderer directly.
        from gpuwm.rustwx import basemap_dir, require_pyshp
        require_pyshp()
        import shapefile
        self.extent = extent
        assets = basemap_dir()
        if not assets.is_dir():
            raise SystemExit(
                f"vendored basemap assets missing at {assets}; the "
                "map frame cannot be drawn without them")

        def shapes(rel: str):
            base = assets / rel
            reader = shapefile.Reader(
                shp=open(f"{base}.shp", "rb"),
                shx=open(f"{base}.shx", "rb"))
            return reader.shapes()

        def collect(rel: str, *, vertex_clip: bool = False):
            x0, x1, y0, y1 = extent
            segs = []
            for shape in shapes(rel):
                if getattr(shape, "shapeTypeName", "") == "NULL":
                    continue
                bx0, by0, bx1, by1 = shape.bbox
                if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                    continue
                for part in _parts(shape):
                    if vertex_clip and not (
                            (part[:, 0] >= x0) & (part[:, 0] <= x1)
                            & (part[:, 1] >= y0)
                            & (part[:, 1] <= y1)).any():
                        continue
                    segs.append(part)
            return segs

        self.county = collect("us_counties_5m/cb_2023_us_county_5m")
        self.state = collect(
            "natural_earth_10m/ne_10m_admin_1_states_provinces_lines")
        self.nat = collect(
            "natural_earth_10m/ne_10m_admin_0_boundary_lines_land")
        # vertex_clip: a continent-bbox coastline far outside the frame
        # must not force ocean handling on an inland domain.
        self.coast = collect("natural_earth_10m/ne_10m_coastline",
                             vertex_clip=True)
        self.lakes = collect("natural_earth_10m/ne_10m_lakes")
        self.all_land = not self.coast
        self.land = ([] if self.all_land
                     else collect("natural_earth_10m/ne_10m_land"))

    def draw_under(self, ax) -> None:
        from matplotlib.collections import PolyCollection
        if self.all_land:
            ax.set_facecolor(LAND_FILL)
        else:
            ax.set_facecolor(WATER_FILL)
            if self.land:
                ax.add_collection(PolyCollection(
                    self.land, facecolors=LAND_FILL,
                    edgecolors="none", zorder=0.5))
        if self.lakes:
            ax.add_collection(PolyCollection(
                self.lakes, facecolors=WATER_FILL, edgecolors="none",
                zorder=0.8))

    def draw_over(self, ax) -> None:
        from matplotlib.collections import LineCollection
        for segs, color, width, z in (
                (self.county, COUNTY_C, 0.35, 3.0),
                (self.lakes, LAKE_EDGE, 0.5, 3.1),
                (self.state, STATE_C, 1.0, 3.2),
                (self.nat, NAT_C, 0.9, 3.3),
                (self.coast, COAST_C, 0.9, 3.4)):
            if segs:
                kwargs = {"alpha": 0.75} if segs is self.county else {}
                ax.add_collection(LineCollection(
                    segs, colors=[color], linewidths=width, zorder=z,
                    **kwargs))


class Gallery:
    """One case directory rendered to one gallery, in place."""

    def __init__(self, case_dir: Path, gallery: Path, dpi: int, *,
                 authority_dir: Path | None = None,
                 cycle_dir: Path | None = None):
        import numpy as np
        import netCDF4 as nc
        import matplotlib
        matplotlib.use("Agg")

        self.np = np
        self.case_dir = case_dir
        self.out = gallery
        self.dpi = dpi
        self.manifest: list[dict] = []
        authority = authority_dir or case_dir / "authority"
        cycle = cycle_dir or case_dir / "cycle"

        exp = tomllib.loads(
            (authority / "experiment.toml")
            .read_text(encoding="utf-8"))["experiment"]
        init = exp["start_time"]
        if init.tzinfo is None:
            init = init.replace(tzinfo=timezone.utc)
        self.init = init.astimezone(timezone.utc)
        self.case_name = exp.get("name", case_dir.name)

        report_path = cycle / "cycle-report.json"
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        self.members = int(self.report["args"]["members"])
        self.leg_seconds = float(self.report["args"]["leg_seconds"])
        self.free_legs = int(self.report["args"]["free_legs"])
        self.legs = self.report["legs"]
        self.composites = cycle / "composites"
        # Every leg's own end, read from the report rather than assumed
        # to be a multiple of one cadence.  A nowcast that cycles on the
        # radar's cadence has legs of unequal length by construction --
        # volumes arrive every 4-6 minutes, not on a clock -- and a
        # figure that dated those frames by leg index would be captioned
        # with times the model never ran to.  The fallback keeps a
        # report written before this key existed rendering unchanged.
        self.leg_end_s = [
            float(leg["end_s"]) if "end_s" in leg
            else (index + 1) * self.leg_seconds
            for index, leg in enumerate(self.legs)]

        # The REPORT says which legs exist, not the obs directory.  A
        # continuously cycling caller writes the next observation before
        # the cycle that consumes it has run, so a renderer that counted
        # files would draw a cycle whose composites do not exist yet --
        # and it did, until this counted legs instead.
        analysis_legs = [index for index, leg in enumerate(self.legs)
                         if leg.get("analysis") is not None]
        self.cycles = len(analysis_legs)
        obs_paths = sorted((case_dir / "obs").glob("*.nc"))
        if not obs_paths:
            raise SystemExit(f"{case_dir}: no observation files under "
                             "obs/")
        if not analysis_legs:
            raise SystemExit(f"{case_dir}: the cycle report carries no "
                             "analysed leg; there is no analysis to draw")
        self.obs = self.match_obs(obs_paths, analysis_legs, nc)
        ref = self.obs[-1]
        self.lat = np.asarray(ref["XLAT"][:], float)
        self.lon = np.asarray(ref["XLONG"][:], float)
        self.dx_km = float(ref.getncattr("DX")) / 1000.0
        # An observation file carries one row per contributing radar.
        # The FIRST is the anchor -- it sites the domain and its lat/lon
        # is what the map draws a marker at -- but every row fed the
        # composite these figures are graded against, so the label names
        # them all.  A three-radar composite captioned with one radar's
        # id would understate the analysis it came from, and this
        # gallery's whole job is to not do that.  With one radar the
        # label IS the id, so single-radar output is unchanged.
        ids = radar_ids(ref["radar_id"])
        self.site = {
            "id": ids[0],
            "ids": ids,
            "label": site_label(ids),
            "lat": float(ref["radar_lat"][0]),
            "lon": float(ref["radar_lon"][0])}

        # A continuously cycling caller leaves a note here when it has
        # something the viewer must be told: a late volume, a skipped
        # cycle, a new background.  It goes ON the figure and on the
        # page, because a gallery that refreshes itself has to say when
        # what it is showing stopped being current.
        self.notice = None
        notice_file = case_dir / "auto-notice.json"
        if notice_file.is_file():
            try:
                self.notice = json.loads(
                    notice_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.notice = None

        self.verify: dict[int, "nc.Dataset"] = {}
        obsverify = case_dir / "obsverify"
        if obsverify.is_dir():
            for path in sorted(obsverify.glob("*.nc")):
                ds = nc.Dataset(path)
                # provenance carries the REQUESTED time; recover the leg
                prov = json.loads(ds.getncattr("provenance"))
                requested = prov["context"]["requested_valid_time"]
                valid = datetime.strptime(
                    requested, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc)
                leg = self.leg_at(
                    (valid - self.init).total_seconds())
                if leg is not None:
                    self.verify[leg] = ds

        pad = 0.05
        self.extent = (self.lon.min() - pad, self.lon.max() + pad,
                       self.lat.min() - pad, self.lat.max() + pad)
        self.basemap = Basemap(self.extent)
        self.range_km = self._great_circle_km()
        self.window_end = self.leg_valid(self.cycles - 1)
        self.foot = (
            f"gpuwm radar-DA nowcast demo (N={self.members}) — real "
            f"{self.site['label']} NEXRAD Level-II — UNSCORED demo, not "
            "campaign evidence — basemap: Natural Earth 10m + US Census "
            "counties (vendored ArWen assets)")
        self.src = (f"source: ArWen (model) · {self.site['label']} "
                    "Level-II (obs)")

    # -- pairing observations with the legs that used them ----------------
    @staticmethod
    def obs_requested_time(ds) -> datetime:
        """The time an observation file was BUILT FOR.

        The volume's own stamp can sit a few minutes off the time it was
        fetched for, so the requested time -- which is the model's leg
        end -- is what pairs a file with a leg.  Files written before
        provenance carried it fall back to their volume stamp.
        """

        try:
            prov = json.loads(ds.getncattr("provenance"))
            stamp = prov["context"]["requested_valid_time"]
        except Exception:
            stamp = ds.getncattr("valid_time")
        return datetime.strptime(
            str(stamp)[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc)

    def match_obs(self, paths, analysis_legs, nc) -> list:
        """One observation per analysed leg, matched by valid time.

        Positional pairing works only while the number of files and the
        number of legs move together, and under a caller that writes the
        next observation before running the cycle for it, they do not.
        """

        entries = [(self.obs_requested_time(ds), ds)
                   for ds in (nc.Dataset(p) for p in paths)]
        matched = []
        for leg in analysis_legs:
            want = self.leg_valid(leg)
            stamp, ds = min(entries, key=lambda e: abs(
                (e[0] - want).total_seconds()))
            offset = abs((stamp - want).total_seconds())
            if offset > OBS_MATCH_TOLERANCE_S:
                raise SystemExit(
                    f"leg {leg} ends at {want:%Y-%m-%dT%H:%M:%SZ} and "
                    f"the nearest observation file is {offset / 60:.1f} "
                    "min away; the case directory pairs observations "
                    "with legs by time and this pair is not one")
            matched.append(ds)
        return matched

    # -- small helpers ----------------------------------------------------
    def leg_elapsed(self, leg: int) -> float:
        """Seconds from init to the end of this leg, per the report."""

        if 0 <= leg < len(self.leg_end_s):
            return self.leg_end_s[leg]
        return (leg + 1) * self.leg_seconds

    def leg_at(self, elapsed_seconds: float) -> int | None:
        """The leg that ends at this elapsed time, or None.

        Nearest leg end, accepted only inside half the shortest leg, so
        an observed composite can never be captioned against a leg the
        model ran to a different time.  Matching on the report's own leg
        ends rather than on ``elapsed / cadence`` is what makes this
        correct when the legs are of unequal length.
        """

        if not self.leg_end_s:
            return None
        starts = [0.0, *self.leg_end_s[:-1]]
        spans = [end - start for start, end
                 in zip(starts, self.leg_end_s)]
        tolerance = max(1.0, min(spans) / 2.0)
        best = min(range(len(self.leg_end_s)),
                   key=lambda i: abs(self.leg_end_s[i]
                                     - elapsed_seconds))
        # >=, not >: a time exactly half a leg from two leg ends is
        # equidistant from both, and picking one of those would be a
        # coin flip dressed up as a match.
        if abs(self.leg_end_s[best] - elapsed_seconds) >= tolerance:
            return None
        return best

    def leg_valid(self, leg: int) -> datetime:
        return self.init + timedelta(seconds=self.leg_elapsed(leg))

    def leg_label(self, leg: int) -> str:
        return self.leg_valid(leg).strftime("%H:%MZ")

    def cadence_label(self) -> str:
        """How the applied cycles were spaced, said honestly.

        A fixed-cadence run says its cadence; a run that cycled on the
        radar's own volume times says the range it actually used, which
        is the truthful caption for legs of unequal length.
        """

        ends = self.leg_end_s[:self.cycles]
        gaps = [round((b - a) / 60.0) for a, b
                in zip([0.0, *ends], ends)]
        if not gaps:
            return ""
        low, high = min(gaps), max(gaps)
        if low == high:
            return f"{low}-min"
        return f"{low}-{high}-min"

    def arwen_subtitle(self, leg: int) -> str:
        lead = int(self.leg_elapsed(leg)) // 60
        valid = self.leg_valid(leg)
        return (f"Init {self.init:%m/%d %H}Z | "
                f"+{lead // 60:03d}:{lead % 60:02d} | "
                f"Valid {valid:%m/%d %H:%M}Z | ArWen | "
                f"Δx {self.dx_km:g} km")

    def obs_subtitle(self, ds) -> str:
        stamp = ds.getncattr("valid_time")
        return (f"{self.site['label']} Level-II composite | "
                f"Valid {stamp[5:10]} {stamp[11:19]}Z | "
                f"Δx {self.dx_km:g} km grid")

    def composite_obs(self, ds):
        np = self.np
        z = np.asarray(ds["z_obs"][:], float)
        m = np.asarray(ds["z_mask"][:]).astype(bool)
        comp = np.where(m, z, -np.inf).max(axis=0)
        return np.where(np.isfinite(comp), comp, np.nan)

    def load_comp(self, leg: int, name: str):
        np = self.np
        with np.load(self.composites
                     / f"leg{leg:02d}_{name}.npz") as z:
            return np.asarray(z["refl_colmax"], float)

    def mean_comp(self, leg: int):
        return self.np.mean(
            [self.load_comp(leg, str(m)) for m in range(self.members)],
            axis=0)

    def _great_circle_km(self):
        np = self.np
        r = 6370.0
        la0 = np.radians(self.site["lat"])
        lo0 = np.radians(self.site["lon"])
        la, lo = np.radians(self.lat), np.radians(self.lon)
        d = np.sin((la - la0) / 2) ** 2 + \
            np.cos(la0) * np.cos(la) * np.sin((lo - lo0) / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(np.clip(d, 0, 1)))

    # -- panels -----------------------------------------------------------
    def refl_cmap(self):
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap = ListedColormap(REFL_COLORS)
        return cmap, BoundaryNorm(REFL_LEVELS, cmap.N)

    def map_frame(self, ax, labels=True):
        np = self.np
        ax.set_aspect(1.0 / np.cos(np.radians(self.lat.mean())))
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        self.basemap.draw_under(ax)
        if labels:
            ax.set_xlabel("lon (°E)", fontsize=7)
            ax.set_ylabel("lat (°N)", fontsize=7)
            ax.tick_params(labelsize=6, length=2.5, color="0.55")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.55")
            spine.set_linewidth(0.7)

    def map_finish(self, ax):
        self.basemap.draw_over(ax)
        ax.contour(self.lon, self.lat, self.range_km,
                   levels=[50, 100, 150], colors="0.45",
                   linewidths=0.5, linestyles=":", zorder=4)
        ax.plot(self.site["lon"], self.site["lat"], marker="*", ms=9,
                mfc="#ffd400", mec="k", mew=0.7, ls="none", zorder=6)

    def refl_panel(self, ax, comp, title, subtitle, labels=True,
                   fs=8.2, future=False, verified=False,
                   compact=False):
        np = self.np
        cmap, norm = self.refl_cmap()
        self.map_frame(ax, labels=labels)
        m = ax.pcolormesh(
            self.lon, self.lat, np.ma.masked_invalid(
                np.where(comp >= REFL_LEVELS[0], comp, np.nan)),
            cmap=cmap, norm=norm, shading="nearest", zorder=2)
        self.map_finish(ax)
        ax.set_title(f"{title}\n{subtitle}", fontsize=fs,
                     fontdict={"linespacing": 1.35})
        # Badge wording follows the panel: the long sentence needs a
        # wide panel, and a strip panel is not one -- an overflowing
        # badge runs into its neighbour and off the figure.
        if future and not verified:
            self.badge(ax, "PAST LAST OBS" if compact
                       else "PAST LAST OBS — unverifiable yet",
                       "#8a2b06", "#fff3e0", compact)
        elif verified:
            self.badge(ax, "VERIFIED" if compact
                       else "VERIFIED AFTER THE FACT — see "
                            "verification row",
                       "#0b5c2e", "#e8f5ec", compact)
        return m

    def badge(self, ax, text, color, face, compact):
        ax.text(0.5, 0.02, text, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.8 if compact else 7.5,
                color=color, fontweight="bold",
                bbox=dict(fc=face, ec=color, lw=0.8, pad=2), zorder=8)

    #: Height in inches reserved at the top of a figure that carries a
    #: notice: the banner AND the title line under it.  Reserved
    #: geometry again, for the same reason the footer band is -- a
    #: banner that overprints the title is worse than no banner.
    NOTICE_BAND_IN = 0.62

    def notice_band_fraction(self, fig_height_in: float) -> float:
        return min(0.16, self.NOTICE_BAND_IN / fig_height_in)

    def notice_stamp(self, fig) -> None:
        """Put the caller's notice on the figure itself.

        On the figure and not only on the page: a PNG gets saved,
        pasted and shown on its own, and one that was drawn while the
        feed was late has to say so wherever it ends up.
        """

        if not self.notice:
            return
        warn = self.notice.get("level") == "warn"
        fig.text(0.5, 0.997,
                 f"{self.notice['headline']} · as of "
                 f"{self.notice.get('updated', '')}",
                 ha="center", va="top", fontsize=8.6,
                 color="#8a2b06" if warn else "#0b3b20",
                 fontweight="bold",
                 bbox=dict(fc="#fff3e0" if warn else "#e8f5ec",
                           ec="#8a2b06" if warn else "#0b5c2e",
                           lw=0.8, pad=2.5))

    def suptitle_y(self, fig) -> float | None:
        """Where a title goes on a figure that may carry a notice."""

        if not self.notice:
            return None
        return 1.0 - self.notice_band_fraction(fig.get_figheight()) * 0.62

    def stamp(self, fig, *, notice: bool = False):
        """Credit line in its own reserved band at the figure foot.

        A managed layout engine is told to keep the axes above the
        band; hand-laid figures already carry a bottom margin larger
        than it.  Either way the footer sits on empty canvas.  When the
        figure also carries a notice, a matching band is reserved at the
        top so banner, title and axes each get their own space.
        """

        band = footer_band_fraction(fig.get_figheight())
        top = (self.notice_band_fraction(fig.get_figheight())
               if notice and self.notice else 0.0)
        engine = fig.get_layout_engine()
        if engine is not None:
            engine.set(rect=(0.0, band, 1.0, 1.0 - band - top))
        layout = credit_layout(fig.get_figwidth(), self.foot, self.src)
        if layout["mode"] == "sides":
            y = 0.34 * band
            fig.text(0.005, y, self.foot, fontsize=layout["foot_pt"],
                     color="0.35", ha="left", va="center")
            fig.text(0.995, y, self.src, fontsize=layout["src_pt"],
                     color="0.30", ha="right", va="center")
        else:
            fig.text(0.005, 0.60 * band, self.foot,
                     fontsize=layout["foot_pt"], color="0.35",
                     ha="left", va="center")
            fig.text(0.005, 0.20 * band, self.src,
                     fontsize=layout["src_pt"], color="0.30",
                     ha="left", va="center")

    def note(self, fname, caption, group):
        self.manifest.append(
            {"file": fname, "caption": caption, "group": group})
        print("wrote", fname, flush=True)

    # -- figures ----------------------------------------------------------
    def fig_lead(self):
        import matplotlib.pyplot as plt
        last = self.cycles - 1
        free = [last + 1 + k for k in range(self.free_legs)]
        picks = free[1::2][:3] or free[:3]
        fig, axes = plt.subplots(1, 2 + len(picks),
                                 figsize=(4.8 * (2 + len(picks)), 5.9),
                                 layout="constrained")
        m = self.refl_panel(
            axes[0], self.composite_obs(self.obs[-1]),
            f"OBSERVED — {self.site['label']} composite (the last "
            "observation any leg saw)",
            self.obs_subtitle(self.obs[-1]))
        self.refl_panel(
            axes[1], self.mean_comp(last),
            "ANALYSIS — member-mean H(Z), last applied cycle",
            self.arwen_subtitle(last))
        for ax, leg in zip(axes[2:], picks):
            lead_min = int((self.leg_elapsed(leg)
                            - self.leg_elapsed(last)) // 60)
            self.refl_panel(
                ax, self.mean_comp(leg),
                f"FREE FORECAST — +{lead_min} min past last obs",
                self.arwen_subtitle(leg), future=True,
                verified=leg in self.verify)
        fig.colorbar(m, ax=list(axes), fraction=0.012, pad=0.008,
                     label="composite reflectivity (dBZ)")
        if self.free_legs:
            end = self.leg_valid(self.cycles + self.free_legs - 1)
            tail = (f"then a free forecast to {end:%H:%M}Z with NO "
                    "further data")
        else:
            # No free legs in this view is a statement, not a gap: the
            # analysis is what exists, and saying "free forecast" over
            # an empty row would be the figure claiming one.
            tail = "no free forecast in this view — analysis only"
        title, points = fit_title(
            f"{self.site['label']} nowcast — {self.cycles} applied "
            f"{self.cadence_label()} cycles to "
            f"{self.window_end:%H:%M}Z (N={self.members}, "
            f"{self.report['args'].get('solve_device', '?')} LETKF), "
            f"{tail}", fig.get_figwidth())
        fig.suptitle(title, fontsize=points, y=self.suptitle_y(fig))
        self.notice_stamp(fig)
        self.stamp(fig, notice=True)
        f = "00-lead-nowcast.png"
        fig.savefig(self.out / f, dpi=self.dpi)
        plt.close(fig)
        self.note(f, "The nowcast in one row on the real map: the last "
                     "observation, the final applied analysis, and the "
                     "free forecast.", "lead")

    def fig_strip(self):
        import matplotlib.pyplot as plt
        ncol = max(self.cycles, self.free_legs)
        if ncol == 0:
            return
        fig, axes = plt.subplots(3, ncol,
                                 figsize=(3.1 * ncol, 10.4),
                                 layout="constrained", squeeze=False)
        m = None
        for c in range(ncol):
            if c < self.cycles:
                m = self.refl_panel(
                    axes[0][c], self.composite_obs(self.obs[c]),
                    f"obs {self.leg_label(c)}",
                    self.obs[c].getncattr("valid_time")[11:19] + "Z",
                    labels=False, fs=7.4)
                self.refl_panel(
                    axes[1][c], self.mean_comp(c),
                    f"cycle {c + 1} — {self.leg_label(c)}",
                    f"+{int(self.leg_elapsed(c)) // 60} min",
                    labels=False, fs=7.4)
            else:
                axes[0][c].set_axis_off()
                axes[1][c].set_axis_off()
            leg = self.cycles + c
            if c < self.free_legs:
                self.refl_panel(
                    axes[2][c], self.mean_comp(leg),
                    f"free — {self.leg_label(leg)}",
                    f"+{int(self.leg_elapsed(leg)) // 60} min",
                    labels=False, fs=7.4, future=True,
                    verified=leg in self.verify, compact=True)
            else:
                axes[2][c].set_axis_off()
        axes[0][0].set_ylabel("OBSERVED", fontsize=9)
        axes[1][0].set_ylabel("MODEL (member mean)", fontsize=9)
        axes[2][0].set_ylabel("FREE FORECAST", fontsize=9)
        if m is not None:
            fig.colorbar(m, ax=[ax for row in axes for ax in row],
                         fraction=0.012, pad=0.008,
                         label="composite dBZ")
        title, points = fit_title(
            f"{self.cycles} cycles against live {self.site['label']} "
            + ("data, then the free forecast — " if self.free_legs
               else "data — analyses only in this view — ")
            + f"Init {self.init:%m/%d %H}Z | ArWen | "
            f"Δx {self.dx_km:g} km", fig.get_figwidth(), points=11.5)
        fig.suptitle(title, fontsize=points, y=self.suptitle_y(fig))
        self.notice_stamp(fig)
        self.stamp(fig, notice=True)
        f = "01-percycle-strip.png"
        fig.savefig(self.out / f, dpi=self.dpi)
        plt.close(fig)
        self.note(f, "Observed vs analysis at every applied cycle, "
                     "then the free forecast, all on the county-level "
                     "map.", "strip")

    def fig_numbers(self):
        import matplotlib.pyplot as plt
        np = self.np
        obs_legs = [leg for leg in self.legs
                    if leg.get("analysis") is not None][:self.cycles]
        if not obs_legs:
            return
        fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2),
                                 layout="constrained")
        x = np.arange(len(obs_legs))
        labels = [self.leg_label(i) for i in range(len(obs_legs))]
        a = [leg["analysis"] for leg in obs_legs]
        axes[0].plot(x, [v["innovations"][0]["innovation_rms"]
                         for v in a], "o-", color="#2f6f9f",
                     label="ensemble mean")
        axes[0].plot(x, [v["control_vr"]["innovation_rms_ms"]
                         for v in a], "s--", color="#c98a2b",
                     label="never-analysed control")
        axes[0].set_title("Vr innovation RMS (m s$^{-1}$)", fontsize=10)
        axes[0].legend(fontsize=8)
        axes[1].plot(x, [v["innovations"][0]["ensemble_spread_mean"]
                         for v in a], "o-", color="#8c4bb5")
        axes[1].set_title(
            f"obs-space spread (m s$^{{-1}}$), N={self.members}",
            fontsize=10)
        for traj, color, lab in (("control", "#c98a2b", "control"),
                                 ("0", "#2f6f9f", "member 0")):
            cols = [leg["trajectories"][traj]["z_obs_space"]
                    ["model_cols_gt35_in_echo"] for leg in obs_legs]
            axes[2].plot(x, cols, "o-", color=color, label=lab)
        axes[2].plot(x, [leg["trajectories"]["control"]["z_obs_space"]
                         ["obs_cols_gt35"] for leg in obs_legs], "k:",
                     label="observed")
        axes[2].set_yscale("symlog")
        axes[2].set_title("columns ≥35 dBZ inside the echo mask",
                          fontsize=10)
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.grid(color="0.9", lw=0.7)
            ax.set_axisbelow(True)
        solves = ", ".join(f"{v['solve_seconds']}s" for v in a)
        fig.suptitle(
            f"Across the applied cycles (solves: {solves}) — "
            "honest numbers, unscored", fontsize=11.5)
        self.stamp(fig)
        f = "02-cycle-numbers.png"
        fig.savefig(self.out / f, dpi=self.dpi)
        plt.close(fig)
        self.note(f, "Innovation RMS vs control, spread, and core "
                     "counts across the applied cycles.", "numbers")

    def verify_numbers(self, leg: int) -> dict:
        from gpuwm.verify.field_metrics import fss_distance
        np = self.np
        ds = self.verify[leg]
        z = np.asarray(ds["z_obs"][:], float)
        zm = np.asarray(ds["z_mask"][:]).astype(bool)
        echo2d = zm.any(axis=0)
        obs_comp = np.where(zm, z, -np.inf).max(axis=0)
        obs_comp = np.where(np.isfinite(obs_comp), obs_comp,
                            MISSING_OBS_FILL_DBZ)
        fcst = self.mean_comp(leg)
        ctrl = self.load_comp(leg, "control")
        hw = max(1, round(FSS_BOX_KM / 2.0 / self.dx_km))
        return {
            "leg": leg,
            "valid": self.leg_valid(leg).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "obs_valid_time": ds.getncattr("valid_time"),
            "obs_cols_gt35": int(
                ((z * zm).max(axis=0) >= COLUMN_THRESHOLD_DBZ).sum()),
            "fcst_cols_gt35_in_echo": int(
                (fcst >= COLUMN_THRESHOLD_DBZ)[echo2d].sum()),
            "control_cols_gt35_in_echo": int(
                (ctrl >= COLUMN_THRESHOLD_DBZ)[echo2d].sum()),
            "fss30_fcst": round(1.0 - fss_distance(
                fcst, obs_comp, threshold=FSS_THRESHOLD_DBZ,
                half_width=hw), 4),
            "fss30_control": round(1.0 - fss_distance(
                ctrl, obs_comp, threshold=FSS_THRESHOLD_DBZ,
                half_width=hw), 4),
            "fss_half_width_cells": hw,
        }

    def fig_verify(self):
        import matplotlib.pyplot as plt
        if not self.verify:
            return None
        legs = sorted(self.verify)
        rows = [self.verify_numbers(leg) for leg in legs]
        ncol = len(legs)
        # Manual layout: the header band (two title lines + the green
        # update stamp) is reserved geometry, so nothing can collide.
        fig = plt.figure(figsize=(3.6 * ncol + 1.4, 8.9))
        gs = fig.add_gridspec(2, ncol, left=0.045, right=0.885,
                              top=0.862, bottom=0.055,
                              wspace=0.06, hspace=0.16)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(ncol)]
                for r in range(2)]
        m = None
        for c, leg in enumerate(legs):
            ds = self.verify[leg]
            vt = ds.getncattr("valid_time")
            lead = int(self.leg_elapsed(leg)) // 60
            m = self.refl_panel(
                axes[0][c], self.composite_obs(ds),
                f"OBSERVED — {self.leg_label(leg)}",
                f"{self.site['label']} composite | Valid {vt[11:19]}Z",
                labels=False, fs=7.8)
            self.refl_panel(
                axes[1][c], self.mean_comp(leg),
                "FORECAST — no data past "
                f"{self.window_end:%H:%M}Z",
                f"+{lead // 60:03d}:{lead % 60:02d} | Valid "
                f"{self.leg_label(leg)} | ArWen | "
                f"Δx {self.dx_km:g} km",
                labels=False, fs=7.8)
            r = rows[c]
            axes[1][c].text(
                0.5, 0.02,
                "≥35 dBZ cols in echo mask — obs "
                f"{r['obs_cols_gt35']} | fcst "
                f"{r['fcst_cols_gt35_in_echo']} | ctrl "
                f"{r['control_cols_gt35_in_echo']}\n"
                f"FSS({FSS_THRESHOLD_DBZ:g} dBZ, {FSS_BOX_KM:g} km) "
                f"— fcst {r['fss30_fcst']:.3f} | ctrl "
                f"{r['fss30_control']:.3f}",
                transform=axes[1][c].transAxes, ha="center",
                va="bottom", fontsize=7.4, color="0.15",
                bbox=dict(fc="white", ec="0.45", lw=0.7, alpha=0.88,
                          pad=2.5), zorder=8)
        axes[0][0].set_ylabel("OBSERVED (after the fact)", fontsize=9)
        axes[1][0].set_ylabel("FREE FORECAST (member mean)", fontsize=9)
        if m is not None:
            cax = fig.add_axes([0.907, 0.20, 0.013, 0.52])
            fig.colorbar(m, cax=cax,
                         label="composite reflectivity (dBZ)")
        total = self.free_legs
        fig.text(0.5, 0.978,
                 "VERIFICATION — observed after the fact: the free "
                 f"forecast vs the {self.site['label']} composites that "
                 "did not exist when it ran", ha="center", va="top",
                 fontsize=11.5)
        fig.text(0.5, 0.947,
                 f"{len(legs)}/{total} frames verified — demo-grade "
                 "numbers, not campaign scores; no data after "
                 f"{self.window_end:%H:%M}Z", ha="center", va="top",
                 fontsize=9.2, color="0.30")
        fig.text(0.885, 0.916,
                 f"verified through {self.leg_label(legs[-1])} · "
                 f"updated {datetime.now(timezone.utc):%H:%M:%S}Z",
                 ha="right", va="top", fontsize=9, color="#0b5c2e",
                 fontweight="bold")
        self.stamp(fig)
        f = "03-verification.png"
        fig.savefig(self.out / f, dpi=self.dpi)
        plt.close(fig)
        self.note(f, "Free-forecast frames beside the observed "
                     "composites at the same valid times, same color "
                     "scale, with per-pair counts and FSS.", "verify")
        return rows

    def fig_scorecard(self, rows):
        import matplotlib.pyplot as plt
        if not rows or len(rows) < 2:
            return
        # Header band and table area are explicit geometry: three
        # text lines up top, the table filling its axes exactly.
        height = 1.9 + 0.42 * (len(rows) + 1)
        fig = plt.figure(figsize=(11.2, height))
        table_h = 0.42 * (len(rows) + 1) / height
        ax = fig.add_axes([0.04, 0.10, 0.92, table_h])
        ax.axis("off")
        cols = ["valid", "obs cols ≥35", "fcst cols ≥35",
                "ctrl cols ≥35", "FSS fcst", "FSS ctrl"]
        cells = [[r["valid"][11:16] + "Z", r["obs_cols_gt35"],
                  r["fcst_cols_gt35_in_echo"],
                  r["control_cols_gt35_in_echo"],
                  f"{r['fss30_fcst']:.3f}",
                  f"{r['fss30_control']:.3f}"] for r in rows]
        tbl = ax.table(cellText=cells, colLabels=cols,
                       cellLoc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("0.75")
            if r == 0:
                cell.set_facecolor("#eef1f4")
                cell.set_text_props(fontweight="bold")
        fig.text(0.5, 1.0 - 0.16 / height,
                 "THE GRADE — the model ran with no data past "
                 f"{self.window_end:%H:%M}Z; reality arrived "
                 "afterwards", ha="center", va="top", fontsize=11.5)
        fig.text(0.5, 1.0 - 0.52 / height,
                 "per-frame ≥35 dBZ column counts in the echo mask "
                 f"and FSS({FSS_THRESHOLD_DBZ:g} dBZ, "
                 f"{FSS_BOX_KM:g} km) vs the observed composite — "
                 "demo-grade, unscored", ha="center", va="top",
                 fontsize=9.2, color="0.30")
        fig.text(0.96, 1.0 - 0.88 / height,
                 f"verified through {rows[-1]['valid'][11:16]}Z · "
                 f"updated {datetime.now(timezone.utc):%H:%M:%S}Z",
                 ha="right", va="top", fontsize=9, color="#0b5c2e",
                 fontweight="bold")
        self.stamp(fig)
        f = "04-scorecard.png"
        fig.savefig(self.out / f, dpi=self.dpi)
        plt.close(fig)
        self.note(f, "Per-frame verification table: observed vs "
                     "forecast vs never-analysed control.", "verify")

    # -- page -------------------------------------------------------------
    def write_page(self, rows):
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{html.escape(self.case_name)} - "
            f"{self.site['label']} nowcast</title>",
            "<style>body{font-family:Segoe UI,system-ui,sans-serif;"
            "margin:2rem;background:#14161a;color:#e8e8e4}h1{font-size:"
            "1.4rem}h2{font-size:1.05rem;margin-top:2.2rem;"
            "border-bottom:1px solid #333;padding-bottom:.3rem}figure"
            "{margin:1.2rem 0}img{max-width:100%;border-radius:6px;"
            "background:#fff}figcaption{font-size:.85rem;color:#9aa;"
            "margin-top:.45rem}.banner{background:#5a2b06;color:"
            "#ffd8b0;padding:.6rem 1rem;border-radius:6px;font-size:"
            ".9rem;margin-bottom:.6rem}.vbanner{background:#0b3b20;"
            "color:#c9ecd6;padding:.6rem 1rem;border-radius:6px;"
            "font-size:.9rem;margin-bottom:.6rem}</style></head><body>",
            f"<h1>{self.site['label']} nowcast — "
            f"{html.escape(self.case_name)}</h1>",
            "<div class='banner'>DEMO-GRADE NOWCAST — UNSCORED, "
            "outside any registered campaign, not campaign evidence. "
            "No skill claim is made or implied.</div>",
        ]
        if self.notice:
            warn = self.notice.get("level") == "warn"
            parts.append(
                f"<div class='{'banner' if warn else 'vbanner'}'>"
                f"<b>{html.escape(self.notice['headline'])}</b> — "
                f"{html.escape(self.notice.get('detail', ''))} "
                f"<span style='opacity:.7'>(as of "
                f"{html.escape(self.notice.get('updated', ''))})</span>"
                "</div>")
        if self.verify:
            done = ", ".join(self.leg_label(i)
                             for i in sorted(self.verify))
            left = [self.leg_label(self.cycles + k)
                    for k in range(self.free_legs)
                    if self.cycles + k not in self.verify]
            parts.append(
                "<div class='vbanner'>VERIFICATION UPDATE: observed "
                f"composites now exist for {done}"
                + (" — still unverified: " + ", ".join(left)
                   if left else " — all free-forecast frames "
                   "verified") + ".</div>")
        elif self.free_legs:
            parts.append(
                "<div class='banner'>THE FREE FORECAST RUNS PAST THE "
                "LAST OBSERVATION. Its verification frames do not "
                "exist yet.</div>")
        parts.append(
            "<p style='font-size:.85rem;color:#9aa'>Every field panel "
            "is drawn on the ArWen product-map frame: county lines, "
            "state borders and water bodies from the vendored basemap "
            "assets, one reflectivity scale throughout.</p>")
        groups = [("lead", "The nowcast"),
                  ("verify", "Verification — observed after the "
                             "fact"),
                  ("strip", "Cycles, then the free forecast"),
                  ("numbers", "Across-cycle numbers")]
        for gid, gtitle in groups:
            entries = [m for m in self.manifest if m["group"] == gid]
            if not entries:
                continue
            parts.append(f"<h2>{html.escape(gtitle)}</h2>")
            for m in entries:
                parts.append(
                    f"<figure><a href='{m['file']}'><img "
                    f"src='{m['file']}' loading='lazy'></a>"
                    f"<figcaption>{html.escape(m['caption'])}"
                    "</figcaption></figure>")
        parts.append("</body></html>")
        (self.out / "index.html").write_text("\n".join(parts),
                                             encoding="utf-8")
        (self.out / "_manifest.json").write_text(
            json.dumps(self.manifest, indent=1), encoding="utf-8")
        if rows:
            (self.out / "_verification.json").write_text(json.dumps({
                "schema": "gpuwm-da.nowcast-gallery-verification.v1",
                "definitions": {
                    "cols_gt35": "tools/da_cycle_prepared.py "
                                 "z_obs_space definitions, identical",
                    "fss": "gpuwm.verify.field_metrics.fss_distance; "
                           f"FSS = 1 - distance, threshold "
                           f"{FSS_THRESHOLD_DBZ:g} dBZ, "
                           f"{FSS_BOX_KM:g} km box",
                    "missing_obs_fill_dbz": MISSING_OBS_FILL_DBZ},
                "frames": rows}, indent=1), encoding="utf-8")

    def render(self):
        self.out.mkdir(parents=True, exist_ok=True)
        # Printed where a user actually is, not only in the docstring: a
        # fallback nobody was told about is how a matplotlib panel ends
        # up in a product gallery.
        print("da_nowcast_render: WARNING -- the WEATHER-FIELD panels in "
              "these sheets are the render law's DEPRECATED FALLBACK, kept "
              "only because the sheets are multi-panel COMPOSITIONS and "
              "the renderer composes one panel.  Single-panel product "
              "tier for the same data:\n"
              "  gpuwm render --engine rust <leg composite wrfout>\n"
              "  rw_obsgrid --obs <obs .nc> --out-dir OUT\n"
              "The innovation/spread chart and the verification table are "
              "NOT demoted; the render law allows matplotlib for charts "
              "that are not weather fields.", flush=True)
        self.fig_lead()
        self.fig_strip()
        self.fig_numbers()
        rows = self.fig_verify()
        if rows:
            self.fig_scorecard(rows)
        self.write_page(rows)
        print(f"gallery: {len(self.manifest)} figures + index.html at "
              f"{self.out}", flush=True)


# ---------------------------------------------------------------------------
# the fine nest over the free forecast
#
# ADOPTION LINE.  These are the entry points a gallery panel needs to
# draw the nested field, and they are deliberately module-level functions
# rather than ``Gallery`` methods: the nest is optional, a case rendered
# before it existed has no nest composites, and a renderer that wants the
# panel opts in with
#
#     nest = nest_panel(cycle_dir, leg, name, gallery.lat, gallery.lon)
#     if nest is not None:
#         ...pcolormesh(nest["lon"], nest["lat"], nest["refl_colmax"])...
#
# beside the existing parent panel, never instead of it.  A fine view OF
# the parent's forecast is only readable next to the forecast it refines.
# ---------------------------------------------------------------------------

def nest_composite_path(cycle_dir: Path, leg: int, name: str,
                        grid_id: int = 2) -> Path:
    """Where ``tools/da_cycle_prepared.py`` writes one nest composite."""
    return (Path(cycle_dir) / "composites"
            / f"leg{leg:02d}_{name}_d{grid_id:02d}.npz")


def nest_latlon(parent_lat, parent_lon, *, i_parent_start: int,
                j_parent_start: int, ratio: int, child_shape):
    """Child mass-point latitude/longitude from the parent's own grid.

    The child's geolocation is DERIVED from the parent's rather than
    recomputed from the projection, for the same reason the child's
    terrain and land are inherited: on this route the nest is a
    refinement of the parent and nothing about it is independently
    known.  Deriving it here also means the panel cannot drift from the
    grid the model actually ran on.

    WRF's nest-down pickup puts child 0-based index ``m`` inside parent
    cell ``ipos - 1 + m // ratio``; within that cell the child centres are
    evenly spaced, so in continuous parent-index coordinates::

        parent_index(m) = (ipos - 1) - 0.5 + (m + 0.5) / ratio

    which is inverted by ordinary bilinear sampling of the parent's own
    latitude/longitude fields.
    """
    import numpy as np

    lat = np.asarray(parent_lat, dtype=float)
    lon = np.asarray(parent_lon, dtype=float)
    if lat.shape != lon.shape or lat.ndim != 2:
        raise ValueError("parent lat/lon must be matching 2-D fields")
    ny_child, nx_child = (int(child_shape[0]), int(child_shape[1]))
    ratio = int(ratio)

    def _coords(n_child: int, start: int) -> np.ndarray:
        m = np.arange(n_child, dtype=float)
        return (start - 1) - 0.5 + (m + 0.5) / ratio

    xi = _coords(nx_child, int(i_parent_start))
    yj = _coords(ny_child, int(j_parent_start))
    ny_parent, nx_parent = lat.shape
    xi = np.clip(xi, 0.0, nx_parent - 1.0)
    yj = np.clip(yj, 0.0, ny_parent - 1.0)

    i0 = np.clip(np.floor(xi).astype(int), 0, nx_parent - 2)
    j0 = np.clip(np.floor(yj).astype(int), 0, ny_parent - 2)
    fx = (xi - i0)[None, :]
    fy = (yj - j0)[:, None]

    def _sample(field):
        a = field[np.ix_(j0, i0)]
        b = field[np.ix_(j0, i0 + 1)]
        c = field[np.ix_(j0 + 1, i0)]
        d = field[np.ix_(j0 + 1, i0 + 1)]
        return ((a * (1 - fx) + b * fx) * (1 - fy)
                + (c * (1 - fx) + d * fx) * fy)

    return _sample(lat), _sample(lon)


def nest_panel(cycle_dir: Path, leg: int, name: str,
               parent_lat, parent_lon, *, grid_id: int = 2):
    """One nested composite plus its geolocation, or ``None`` if absent.

    ``None`` is the ordinary answer for a leg the nest did not cover (the
    assimilation legs) and for any case rendered from a run that had no
    nest at all, so a caller branches on it rather than guarding with a
    file test of its own.
    """
    import numpy as np

    path = nest_composite_path(cycle_dir, leg, name, grid_id)
    if not path.is_file():
        return None
    with np.load(path) as payload:
        comp = np.asarray(payload["refl_colmax"], float)
        record = {
            "refl_colmax": comp,
            "elapsed_seconds": float(payload["elapsed_seconds"]),
            "dx_km": float(payload["dx_m"]) / 1000.0,
            "i_parent_start": int(payload["i_parent_start"]),
            "j_parent_start": int(payload["j_parent_start"]),
            "parent_grid_ratio": int(payload["parent_grid_ratio"]),
        }
    record["lat"], record["lon"] = nest_latlon(
        parent_lat, parent_lon,
        i_parent_start=record["i_parent_start"],
        j_parent_start=record["j_parent_start"],
        ratio=record["parent_grid_ratio"],
        child_shape=comp.shape)
    # The subtitle convention this module already uses, with the child's
    # own spacing -- the one number a reader of a nested panel is looking
    # for.
    record["dx_label"] = f"Δx {record['dx_km']:g} km"
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_nowcast_render",
        description=__doc__.splitlines()[0])
    parser.add_argument("--case-dir", type=Path, required=True,
                        help="a tools/da_nowcast.py case directory")
    parser.add_argument("--gallery", type=Path, default=None,
                        help="output directory (default: "
                             "<case-dir>/gallery)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--authority-dir", type=Path, default=None,
                        help="override for <case-dir>/authority, for a "
                             "case laid out by hand")
    parser.add_argument("--cycle-dir", type=Path, default=None,
                        help="override for <case-dir>/cycle (the "
                             "cycle-report.json + composites/ pair)")
    args = parser.parse_args(argv)
    gallery = args.gallery or args.case_dir / "gallery"
    Gallery(args.case_dir, gallery, args.dpi,
            authority_dir=args.authority_dir,
            cycle_dir=args.cycle_dir).render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
