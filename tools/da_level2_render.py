"""Render the Level-II exploitation A/B: what more radars actually buy.

Two waves, one flat gallery, because a picture of an observation set is
available long before any forecast that used it.

**Wave 1 — the observation input.**  Drawn from
``gpuwm-obs.radar-grid.v1`` files alone, so it needs no model run at all:
how much of the domain one radar can see against what every radar
covering it sees together, where their coverage overlaps (which is the
only place a genuine three-dimensional wind constraint exists), and what
each site contributed on its own.

**Wave 2 — the scored arms.**  A watcher redraws the gallery in place as
each arm's score lands and exits when the last one has.  Never a busy
wait inside anything interactive: it is a detached process with a status
file, the same shape the rolling verifier uses.

Style is the ArWen product map and is NOT reinvented here: the basemap
class, the vendored Natural Earth and US Census county assets, the
reflectivity scale, the credit-band geometry and the honesty banners all
come from :mod:`tools.da_nowcast_render`.  The one thing added is a
diverging radial-velocity scale, because that module renders
reflectivity and this one has to render velocity.

No site names and no case names: every id here is read out of the files.
EXPERIMENTAL, like everything it draws.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.da_nowcast_render import (Basemap, REFL_COLORS, REFL_LEVELS,
                                     credit_layout, fit_title,
                                     footer_band_fraction)

#: Diverging radial-velocity scale, m/s, inbound negative.  Symmetric on
#: purpose: a couplet is only legible if equal speeds toward and away
#: from the antenna are equally saturated, and an asymmetric scale would
#: make one half of every rotation look stronger than the other.
VEL_LEVELS = [-32, -24, -18, -12, -8, -4, -1, 1, 4, 8, 12, 18, 24, 32]
VEL_COLORS = ["#12d0b0", "#18b89a", "#1fa085", "#2b8770", "#3a6f5c",
              "#4a5a4e", "#9a9a9a", "#5c4a4a", "#7a3b3b", "#9c3030",
              "#c02424", "#e01818", "#ff2a2a"]

#: How many radars a cell may be seen by before the discrete overlap
#: scale stops distinguishing them.  Beyond this the colour saturates
#: and the label says so rather than implying a count it cannot show.
OVERLAP_MAX = 4
OVERLAP_COLORS = ["#3b4a5a", "#f0b429", "#ef6c1a", "#c62828"]

#: The range authority discovery and the superobber share.  Drawn as a
#: ring so a reader can see which part of a site's disc actually landed
#: on the domain -- the ring is the planning radius, not a claim about
#: what the beam did through terrain.
RANGE_RING_KM = 250.0

STATUS_SCHEMA = "gpuwm-da.level2-gallery.v1"

#: Arms the wave-2 lead row draws, in order, when their scores exist.
#: Read as "shipped, then each axis, then both" -- the comparison the
#: gallery is for.  An arm absent from the run directory is simply not
#: drawn; the caption says which ones were.
LEAD_ARMS = ("A0-fixed900-1radar", "C1-fixed900-multiradar",
             "D1-pervolume-multiradar-scaled")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(when: datetime | None = None) -> str:
    return (when or _utcnow()).strftime("%H:%MZ")


class Level2Gallery:
    """One flat directory of PNGs plus the index that links them."""

    def __init__(self, out: Path, dpi: int = 150):
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        self.np = np
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.manifest: list[dict] = []
        self.arms_complete = 0
        self.arms_total = 6
        self.foot = (
            "gpuwm / ArWen Level-II exploitation A/B — real NEXRAD "
            "Level-II observations — UNSCORED demo, not campaign "
            "evidence — basemap: Natural Earth 10m + US Census counties "
            "(vendored ArWen assets)")
        self.src = "source: NEXRAD Level-II (obs) · ArWen (model)"

    # -- shared frame ----------------------------------------------------
    def open_obs(self, path: Path) -> dict:
        """One radar-grid file, reduced to what a map needs."""

        import netCDF4 as nc
        np = self.np

        with nc.Dataset(str(path)) as ds:
            lat = np.asarray(ds["XLAT"][:], float)
            lon = np.asarray(ds["XLONG"][:], float)
            vr_mask = np.asarray(ds["vr_mask"][:]).astype(bool)
            z_obs = np.asarray(ds["z_obs"][:], float)
            z_mask = np.asarray(ds["z_mask"][:]).astype(bool)
            vr_obs = np.asarray(ds["vr_obs"][:], float)
            ids = ["".join(c.decode() if isinstance(c, bytes) else str(c)
                           for c in row).strip()
                   for row in np.asarray(ds["radar_id"][:])]
            times = ["".join(c.decode() if isinstance(c, bytes) else str(c)
                             for c in row).strip()
                     for row in np.asarray(ds["radar_valid_time"][:])]
            rlat = np.asarray(ds["radar_lat"][:], float)
            rlon = np.asarray(ds["radar_lon"][:], float)
            valid = ds.getncattr("valid_time")
            provenance = json.loads(ds.getncattr("provenance"))
        # The writer nests the builder's own block under `context`; the
        # builder keys are read from there and not from the top level,
        # where they have never been.
        context = provenance.get("context") or {}
        return {
            "path": path, "lat": lat, "lon": lon, "vr_mask": vr_mask,
            "vr_obs": vr_obs, "z_obs": z_obs, "z_mask": z_mask,
            "radars": [{"id": i, "lat": float(a), "lon": float(o),
                        "valid_time": t}
                       for i, a, o, t in zip(ids, rlat, rlon, times)],
            "valid_time": valid, "provenance": provenance,
            "context": context,
        }

    def extent(self, doc) -> tuple[float, float, float, float]:
        pad = 0.05
        return (doc["lon"].min() - pad, doc["lon"].max() + pad,
                doc["lat"].min() - pad, doc["lat"].max() + pad)

    def frame(self, ax, doc, basemap, *, labels=False):
        np = self.np
        ext = self.extent(doc)
        ax.set_aspect(1.0 / np.cos(np.radians(doc["lat"].mean())))
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        basemap.draw_under(ax)
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

    def sites(self, ax, radars, *, rings=True, label=True):
        """Antenna markers, and the planning radius around each."""

        np = self.np
        for radar in radars:
            if rings:
                # A circle of constant great-circle range, drawn in
                # lat/lon: the longitude radius grows with latitude, so
                # it is scaled rather than drawn as a plain circle.
                theta = np.linspace(0, 2 * np.pi, 361)
                dlat = RANGE_RING_KM / 111.19
                dlon = dlat / max(np.cos(np.radians(radar["lat"])), 1e-6)
                ax.plot(radar["lon"] + dlon * np.cos(theta),
                        radar["lat"] + dlat * np.sin(theta),
                        color="0.30", lw=0.6, ls=":", zorder=5,
                        alpha=0.75)
            ax.plot(radar["lon"], radar["lat"], marker="*", ms=9,
                    mfc="#ffd400", mec="k", mew=0.7, ls="none", zorder=6)
            if label:
                ax.annotate(radar["id"], (radar["lon"], radar["lat"]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=6.5, color="#111", zorder=7,
                            bbox={"boxstyle": "round,pad=0.15",
                                  "fc": "#ffffffcc", "ec": "none"})

    def stamp(self, fig):
        band = footer_band_fraction(fig.get_figheight())
        engine = fig.get_layout_engine()
        if engine is not None:
            engine.set(rect=(0.0, band, 1.0, 1.0 - band))
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
        self.manifest = [m for m in self.manifest if m["file"] != fname]
        self.manifest.append({"file": fname, "caption": caption,
                              "group": group})
        print("wrote", fname, flush=True)

    def save(self, fig, fname, caption, group):
        import matplotlib.pyplot as plt
        self.stamp(fig)
        fig.savefig(self.out / fname, dpi=self.dpi)
        plt.close(fig)
        self.note(fname, caption, group)

    # -- wave 1 ----------------------------------------------------------
    def coverage_depth(self, doc, index=None):
        """Observed levels per column: how deep the radar sees, not just
        whether it does.  A column touched at one level and a column
        sampled through its depth are very different observations and a
        binary mask draws them identically."""

        mask = doc["vr_mask"]
        if index is None:
            return mask.any(axis=0).sum(axis=0)
        return mask[index].sum(axis=0)

    def fig_coverage_pair(self, single, multi):
        """One radar against every radar, same domain, same scale."""

        import matplotlib.pyplot as plt
        np = self.np

        basemap = Basemap(self.extent(multi))
        left = self.coverage_depth(single)
        right = self.coverage_depth(multi)
        top = float(max(left.max(), right.max()))

        fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2),
                                 layout="constrained")
        mesh = None
        for ax, field, doc, title in (
                (axes[0], left, single,
                 f"ONE radar — {len(single['radars'])} site"),
                (axes[1], right, multi,
                 f"EVERY radar covering the domain — "
                 f"{len(multi['radars'])} sites")):
            self.frame(ax, multi, basemap)
            mesh = ax.pcolormesh(
                doc["lon"], doc["lat"],
                np.where(field > 0, field, np.nan),
                cmap="viridis", vmin=1, vmax=top, shading="nearest",
                zorder=2)
            basemap.draw_over(ax)
            self.sites(ax, doc["radars"], rings=True,
                       label=len(doc["radars"]) <= 12)
            cells = int(doc["vr_mask"].sum())
            ax.set_title(f"{title}\n{cells:,} velocity superob cells",
                         fontsize=10.5, pad=6)

        bar = fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.012)
        bar.set_label("model levels carrying a usable radial velocity",
                      fontsize=8)
        bar.ax.tick_params(labelsize=7)
        gain = (int(multi["vr_mask"].sum())
                / max(int(single["vr_mask"].sum()), 1))
        title, points = fit_title(
            f"Radial-velocity coverage at {multi['valid_time']} — "
            f"{gain:.2f}x more observations from the same instant",
            13.4, points=13.0)
        fig.suptitle(title, fontsize=points)
        self.save(fig, "w1-coverage-one-vs-many.png",
                  f"Velocity superob coverage on one georeference at "
                  f"{multi['valid_time']}. Left: the single radar the "
                  f"shipped configuration uses "
                  f"({int(single['vr_mask'].sum()):,} cells). Right: "
                  f"every site the domain discovery selected "
                  f"({int(multi['vr_mask'].sum()):,} cells, {gain:.2f}x). "
                  "Colour is how many model levels in a column carry a "
                  "usable velocity, so depth of sampling is visible and "
                  "not just its footprint. Dotted rings are the 250 km "
                  "range authority; stars are antennas.",
                  "wave1")

    def fig_overlap(self, multi):
        """Where two or more antennas see the same column."""

        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        np = self.np

        basemap = Basemap(self.extent(multi))
        per_radar = multi["vr_mask"].any(axis=1)          # (radar, y, x)
        count = per_radar.sum(axis=0)
        shown = np.clip(count, 0, OVERLAP_MAX)

        cmap = ListedColormap(OVERLAP_COLORS)
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, OVERLAP_MAX + 0.5],
                            cmap.N)

        fig, ax = plt.subplots(figsize=(9.6, 8.4), layout="constrained")
        self.frame(ax, multi, basemap, labels=True)
        ax.pcolormesh(multi["lon"], multi["lat"],
                      np.where(shown > 0, shown, np.nan),
                      cmap=cmap, norm=norm, shading="nearest", zorder=2)
        basemap.draw_over(ax)
        self.sites(ax, multi["radars"], rings=True)

        histogram = {n: int((count == n).sum())
                     for n in range(1, int(count.max()) + 1)}
        dual = int((count > 1).sum())
        # TWO different quantities, and they must never be conflated:
        # `dual` counts COLUMNS with two or more antennas anywhere in
        # them, `dual_cells` counts the individual 3-D superob cells
        # that have two or more.  A tall column seen by two radars at
        # different heights is one column and many cells.
        cell_count = multi["vr_mask"].sum(axis=0)
        dual_cells = int((cell_count > 1).sum())
        handles = []
        from matplotlib.patches import Patch
        for n in range(1, OVERLAP_MAX + 1):
            label = (f"{n} radar" if n == 1 else f"{n} radars")
            if n == OVERLAP_MAX and int(count.max()) > OVERLAP_MAX:
                label = f"{n}+ radars"
            total = sum(v for k, v in histogram.items()
                        if k == n or (n == OVERLAP_MAX and k > n))
            handles.append(Patch(facecolor=OVERLAP_COLORS[n - 1],
                                 label=f"{label} — {total:,} cells"))
        ax.legend(handles=handles, loc="lower left", fontsize=8,
                  framealpha=0.92, title="columns seen by", title_fontsize=8)

        spread = float(multi["context"].get(
            "radar_time_spread_seconds") or 0.0)
        title, points = fit_title(
            f"Dual-Doppler coverage at {multi['valid_time']} — "
            f"{dual:,} columns / {dual_cells:,} cells with two or more "
            "look angles", 9.6, points=12.5)
        ax.set_title(title, fontsize=points, pad=8)
        fig.text(0.5, 0.012,
                 "One radar measures only the wind component along its own "
                 "beam. A column seen by two antennas from different "
                 "directions constrains the actual wind vector; a column "
                 "seen by one does not.\nContributing volumes span "
                 f"{spread:.0f} s — they are NOT simultaneous, and each "
                 "radar's own volume time travels with its data.",
                 ha="center", va="bottom", fontsize=8, color="0.25")
        self.save(fig, "w1-dual-doppler-overlap.png",
                  f"Columns coloured by how many antennas carry a usable "
                  f"radial velocity there, at {multi['valid_time']}. "
                  f"{dual:,} COLUMNS -- {dual_cells:,} individual 3-D "
                  "superob cells -- have two or more independent look "
                  "angles and therefore a real vector constraint; the "
                  "rest have one projection and an unobservable "
                  "cross-beam component. Histogram in the legend. "
                  f"Contributing volume times span {spread:.0f} s.",
                  "wave1")

    def fig_per_radar(self, multi):
        """Each site's own contribution, on one shared scale."""

        import matplotlib.pyplot as plt
        np = self.np

        radars = multi["radars"]
        n = len(radars)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        basemap = Basemap(self.extent(multi))
        depths = [self.coverage_depth(multi, index=i) for i in range(n)]
        top = float(max(1.0, max(d.max() for d in depths)))

        fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols,
                                                      3.5 * rows + 0.6),
                                 layout="constrained")
        axes = np.atleast_1d(axes).ravel()
        mesh = None
        for index, radar in enumerate(radars):
            ax = axes[index]
            self.frame(ax, multi, basemap)
            mesh = ax.pcolormesh(
                multi["lon"], multi["lat"],
                np.where(depths[index] > 0, depths[index], np.nan),
                cmap="viridis", vmin=1, vmax=top, shading="nearest",
                zorder=2)
            basemap.draw_over(ax)
            self.sites(ax, [radar], rings=True, label=False)
            cells = int(multi["vr_mask"][index].sum())
            ax.set_title(f"{radar['id']} — {cells:,} cells\n"
                         f"{radar['valid_time']}", fontsize=8.5, pad=4)
        for spare in axes[n:]:
            spare.axis("off")
        if mesh is not None:
            bar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.7,
                               pad=0.012)
            bar.set_label("levels with a usable velocity", fontsize=8)
            bar.ax.tick_params(labelsize=7)
        title, points = fit_title(
            f"Per-radar contribution at {multi['valid_time']} — each "
            "site's own cells, one shared scale", 3.3 * cols, points=12.0)
        fig.suptitle(title, fontsize=points)
        self.save(fig, "w1-per-radar-contribution.png",
                  "What each contributing site brought to the same "
                  "analysis, on one colour scale. Coverage falls off with "
                  "range and with beam height, so a distant site "
                  "contributes a thin shell over the domain edge rather "
                  "than a uniform share. Each panel is titled with that "
                  "radar's own volume time.",
                  "wave1")

    def fig_velocity_field(self, single, multi):
        """The velocity itself, at the lowest level each column has."""

        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        np = self.np

        basemap = Basemap(self.extent(multi))
        cmap = ListedColormap(VEL_COLORS)
        norm = BoundaryNorm(VEL_LEVELS, cmap.N)

        def lowest(doc):
            """Vr at the lowest observed level, merged nearest-antenna."""
            mask = doc["vr_mask"]
            obs = doc["vr_obs"]
            out = np.full(mask.shape[-2:], np.nan)
            # Fill from the top level down so the LOWEST observation
            # wins: the lowest beam is the one closest to the storm-
            # relative flow a nowcast cares about.
            for radar in range(mask.shape[0]):
                for level in range(mask.shape[1] - 1, -1, -1):
                    here = mask[radar, level]
                    out = np.where(here, obs[radar, level], out)
            return out

        fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2),
                                 layout="constrained")
        mesh = None
        for ax, doc, title in (
                (axes[0], single, f"ONE radar ({len(single['radars'])})"),
                (axes[1], multi,
                 f"EVERY radar ({len(multi['radars'])})")):
            self.frame(ax, multi, basemap)
            mesh = ax.pcolormesh(doc["lon"], doc["lat"], lowest(doc),
                                 cmap=cmap, norm=norm, shading="nearest",
                                 zorder=2)
            basemap.draw_over(ax)
            self.sites(ax, doc["radars"], rings=True,
                       label=len(doc["radars"]) <= 12)
            ax.set_title(title, fontsize=10.5, pad=6)
        bar = fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.012,
                           ticks=VEL_LEVELS)
        bar.set_label("superobbed radial velocity (m/s), positive AWAY "
                      "from the antenna", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        title, points = fit_title(
            f"Observed radial velocity at {multi['valid_time']}, lowest "
            "observed level per column", 13.4, points=13.0)
        fig.suptitle(title, fontsize=points)
        self.save(fig, "w1-velocity-lowest-level.png",
                  "The velocities themselves, at the lowest level each "
                  "column has. Each radar's colours are radial to ITS "
                  "OWN antenna, so where two sites overlap the same wind "
                  "shows two different values -- that disagreement is "
                  "the information, not an error. Velocities above 0.8 "
                  "of a sweep's Nyquist are masked for aliasing risk and "
                  "never dealiased, so the fastest inbound and outbound "
                  "gates are ABSENT from both panels.",
                  "wave1")

    def wave1(self, single_path: Path, multi_path: Path):
        # DEPRECATED FALLBACK, under the render law (CLAUDE.md, Drew
        # 2026-08-06).  Every field these four figures show is drawn
        # natively by ``rw_obsgrid`` from the same
        # ``gpuwm-obs.radar-grid.v1`` files -- column-max Z, coverage
        # depth, distinct-radar overlap, lowest-tilt radial velocity and
        # per-radar contribution are its five products, and it was built
        # for exactly this surface (MEASURED 2026-08-17 on a real
        # three-radar file).
        #
        # What keeps these here is the SHAPE, not the science: three of
        # the four are side-by-side or per-radar GRIDS of panels, and
        # ``MapRenderRequest`` composes one panel.  The A/B question this
        # wave asks is inherently a comparison, so the sheet is the
        # product.  Render the panels with the binary and compose them
        # the day there is a compositor; do not add a fifth figure here.
        print("da_level2_render: WARNING -- wave-1 panels are the render "
              "law's DEPRECATED FALLBACK; the product tier for these same "
              "files is  rw_obsgrid --obs FILE.nc --out-dir OUT  and this "
              "module keeps them only because its figures are multi-panel "
              "comparisons.", flush=True)
        single = self.open_obs(single_path)
        multi = self.open_obs(multi_path)
        self.fig_coverage_pair(single, multi)
        self.fig_overlap(multi)
        self.fig_velocity_field(single, multi)
        self.fig_per_radar(multi)
        return single, multi

    # -- wave 2 ----------------------------------------------------------
    def arm_scores(self, run_dir: Path) -> dict:
        """Whatever arm scores exist right now, keyed by arm name."""

        results = run_dir / "results"
        scores = {}
        if results.is_dir():
            for path in sorted(results.glob("*.json")):
                try:
                    scores[path.stem] = json.loads(
                        path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
        return scores

    def fig_fss_curves(self, scores: dict):
        """Every arm's FSS-versus-neighborhood curve on one chart."""

        import matplotlib.pyplot as plt

        if not scores:
            return
        fig, (ax, tab) = plt.subplots(
            1, 2, figsize=(13.0, 5.6), layout="constrained",
            gridspec_kw={"width_ratios": [1.35, 1.0]})
        colors = plt.get_cmap("tab10")
        rows = []
        for index, (name, score) in enumerate(sorted(scores.items())):
            curve = score.get("neighborhood_curve_mean") or []
            if curve:
                ax.plot([row["box_km_across"] for row in curve],
                        [row["fss30_fcst_mean"] for row in curve],
                        marker="o", ms=4, lw=1.6, color=colors(index % 10),
                        label=name)
            rows.append((name, score))
        # One control curve is enough: it is the same never-analysed
        # trajectory in every arm, and six copies of it would read as
        # six different controls.
        first = next(iter(sorted(scores.items())))[1]
        control = first.get("neighborhood_curve_mean") or []
        if control:
            ax.plot([row["box_km_across"] for row in control],
                    [row["fss30_control_mean"] for row in control],
                    color="0.45", lw=1.4, ls="--",
                    label="control (no DA)")
        ax.set_xlabel("neighborhood — square SIDE length (km), not a radius",
                      fontsize=9)
        ax.set_ylabel("FSS at 30 dBZ, ensemble-mean field", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92)
        ax.set_title("FSS versus scale — the shape is the result, one box "
                     "is a point on it", fontsize=10.5)

        tab.axis("off")
        header = ["arm", "FSS27\nmean-field", "FSS27\nper-member", "control"]
        body = []
        for name, score in rows:
            body.append([
                name,
                f"{score.get('fss30_fcst_mean', float('nan')):.4f}",
                f"{score.get('fss30_per_member_mean', float('nan')):.4f}",
                f"{score.get('fss30_control_mean', float('nan')):.4f}"])
        table = tab.table(cellText=body, colLabels=header,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.5)
        tab.set_title("Mean over the six free-forecast frames",
                      fontsize=10.5)
        self.save(fig, "w2-fss-curves.png",
                  "FSS at 30 dBZ against neighborhood size for every arm "
                  "that has finished scoring, with the never-analysed "
                  "control. The 27 km column is the published box; the "
                  "per-member column scores each member's own field "
                  "rather than the ensemble mean. Single draws, no error "
                  "bars, no dual-run screen.",
                  "wave2")

    def fig_lead_row(self, run_dir: Path, scores: dict):
        """Observed composite against each arm, same time, same scale."""

        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        np = self.np

        available = [a for a in LEAD_ARMS if a in scores]
        if not available:
            return
        # Every arm scores the same verification files in the same order,
        # so frame k is the same valid time in each.
        frames = scores[available[0]].get("frames") or []
        if not frames:
            return

        import netCDF4 as nc
        cmap = ListedColormap(REFL_COLORS)
        norm = BoundaryNorm(REFL_LEVELS, cmap.N)
        obs_dir = Path(scores[available[0]]["obs_dir"])
        obs_files = sorted(obs_dir.glob("*verify*.nc"))
        if not obs_files:
            return

        for row, frame in enumerate(frames):
            leg = int(frame["leg"])
            if row >= len(obs_files):
                break
            with nc.Dataset(str(obs_files[row])) as ds:
                lat = np.asarray(ds["XLAT"][:], float)
                lon = np.asarray(ds["XLONG"][:], float)
                z = np.asarray(ds["z_obs"][:], float)
                zmask = np.asarray(ds["z_mask"][:]).astype(bool)
            observed = np.where(zmask, z, -np.inf).max(axis=0)
            observed = np.where(np.isfinite(observed), observed, np.nan)
            doc = {"lat": lat, "lon": lon}
            basemap = Basemap(self.extent(doc))

            panels = [("observed", observed, "observed composite")]
            for arm in available:
                field = self._arm_composite(run_dir, arm, leg)
                if field is not None:
                    panels.append((arm, field, arm))
            if len(panels) < 2:
                continue

            fig, axes = plt.subplots(
                1, len(panels), figsize=(4.4 * len(panels), 4.9),
                layout="constrained")
            axes = np.atleast_1d(axes).ravel()
            mesh = None
            for ax, (key, field, title) in zip(axes, panels):
                self.frame(ax, doc, basemap)
                mesh = ax.pcolormesh(lon, lat,
                                     np.where(field >= REFL_LEVELS[0],
                                              field, np.nan),
                                     cmap=cmap, norm=norm,
                                     shading="nearest", zorder=2)
                basemap.draw_over(ax)
                sub = ""
                if key in scores:
                    sub = (f"\nFSS27 {frame['fss30_fcst']:.4f}"
                           if key == available[0] else "")
                    entry = next((f for f in scores[key]["frames"]
                                  if int(f["leg"]) == leg), None)
                    if entry:
                        sub = f"\nFSS27 {entry['fss30_fcst']:.4f}"
                ax.set_title(f"{title}{sub}", fontsize=9.5, pad=5)
            bar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.85,
                               pad=0.012, ticks=REFL_LEVELS[::2])
            bar.set_label("column-max reflectivity (dBZ)", fontsize=8)
            bar.ax.tick_params(labelsize=7)
            title, points = fit_title(
                f"Free forecast valid {frame['obs_valid_time']} — "
                "observed against each arm, one reflectivity scale",
                4.4 * len(panels), points=12.0)
            fig.suptitle(title, fontsize=points)
            fname = f"w2-lead-{row:02d}.png"
            self.save(fig, fname,
                      f"Valid {frame['obs_valid_time']} (free-forecast "
                      f"leg {leg}). Observed column-max reflectivity "
                      "beside each arm's ensemble-mean composite, on one "
                      "scale. FSS27 is at the published 27 km square "
                      "side. Single draws, unscored demo.",
                      "wave2")

    def _arm_composite(self, run_dir: Path, arm: str, leg: int):
        """That arm's ensemble-mean column-max for one leg, or None."""

        np = self.np
        roots = [run_dir / arm]
        found = []
        for root in roots:
            if not root.is_dir():
                continue
            found.extend(root.glob(f"**/composites/leg{leg:02d}_*.npz"))
        members = [p for p in found if "control" not in p.name]
        if not members:
            return None
        stack = []
        for path in sorted(members):
            with np.load(path) as handle:
                stack.append(np.asarray(handle["refl_colmax"], float))
        return np.mean(stack, axis=0)

    # -- page ------------------------------------------------------------
    def write_index(self, *, run_dir: Path | None = None,
                    scores: dict | None = None,
                    watching: bool = True):
        scores = scores or {}
        self.arms_complete = len(scores)
        groups = [
            ("wave1", "Wave 1 — what the observations look like",
             "Built from radar-grid observation files alone. No model "
             "run is involved in any figure in this section."),
            ("wave2", "Wave 2 — the scored arms",
             "Redrawn in place as each arm finishes scoring."),
        ]
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,"
            "initial-scale=1'>",
            "<title>Level-II exploitation A/B</title>",
            "<style>body{font-family:Segoe UI,system-ui,sans-serif;"
            "margin:2rem;background:#14161a;color:#e8e8e4;max-width:1600px}"
            "h1{font-size:1.4rem}h2{font-size:1.05rem;margin-top:2.2rem;"
            "border-bottom:1px solid #333;padding-bottom:.3rem}"
            "figure{margin:1.2rem 0}img{max-width:100%;border-radius:6px;"
            "background:#fff}figcaption{font-size:.85rem;color:#9aa;"
            "margin-top:.45rem;max-width:110ch;line-height:1.45}"
            ".banner{background:#5a2b06;color:#ffd8b0;padding:.6rem 1rem;"
            "border-radius:6px;font-size:.9rem;margin-bottom:.6rem}"
            ".vbanner{background:#0b3b20;color:#c9ecd6;padding:.6rem 1rem;"
            "border-radius:6px;font-size:.9rem;margin-bottom:.6rem}"
            ".note{font-size:.85rem;color:#9aa;margin:.3rem 0 1rem}"
            "</style></head><body>",
            "<h1>Level-II exploitation A/B — cadence &times; radar count"
            "</h1>",
            "<div class='banner'>DEMO-GRADE — UNSCORED, outside any "
            "registered campaign, not campaign evidence. Every arm is a "
            "single draw: no error bars, and the no-ECC dual-run screen "
            "is not applied. No skill claim is made or implied.</div>",
            f"<div class='vbanner'>updated {_stamp()} &middot; arms "
            f"complete: {self.arms_complete}/{self.arms_total}"
            + (" &middot; watcher running" if watching
               else " &middot; watcher finished") + "</div>",
            "<div class='banner'>Radial velocity is masked for aliasing "
            "risk and never dealiased. On this case the gate-to-gate fold "
            "scan flags 955 boundaries in the first cycle rising to 6,862 "
            "in the sixth, and the 0.8-of-Nyquist magnitude mask drops "
            "10,676 gates rising to 82,649 &mdash; so the fastest "
            "low-level velocities are absent from every figure here, in "
            "every arm equally.</div>",
        ]
        for gid, gtitle, gnote in groups:
            entries = [m for m in self.manifest if m["group"] == gid]
            if not entries:
                continue
            parts.append(f"<h2>{html.escape(gtitle)}</h2>")
            parts.append(f"<div class='note'>{html.escape(gnote)}</div>")
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


def write_status(path: Path, **fields) -> None:
    payload = {"schema": STATUS_SCHEMA,
               "updated": _utcnow().isoformat(), **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_level2_render",
        description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="flat gallery directory (PNGs + index.html)")
    parser.add_argument("--single-obs", type=Path, default=None,
                        help="a one-radar gpuwm-obs.radar-grid.v1 file")
    parser.add_argument("--multi-obs", type=Path, default=None,
                        help="the many-radar file for the same instant "
                             "and the same georeference")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="the A/B queue's run directory, for wave 2")
    parser.add_argument("--watch", action="store_true",
                        help="redraw wave 2 in place as arm scores land, "
                             "then exit. Detached; never run this inside "
                             "anything that is waited on")
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--max-hours", type=float, default=14.0)
    parser.add_argument("--expect-arms", type=int, default=6)
    parser.add_argument("--final-arm", default="D1-pervolume-multiradar-scaled",
                        help="the watcher exits once this arm has scored "
                             "(or the deadline passes)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--status", type=Path, default=None)
    args = parser.parse_args(argv)

    gallery = Level2Gallery(args.out, dpi=args.dpi)
    gallery.arms_total = int(args.expect_arms)
    status = args.status or (args.out / "_watcher-status.json")

    if args.single_obs and args.multi_obs:
        gallery.wave1(args.single_obs, args.multi_obs)

    scores = (gallery.arm_scores(args.run_dir) if args.run_dir else {})
    if scores:
        gallery.fig_fss_curves(scores)
        gallery.fig_lead_row(args.run_dir, scores)
    gallery.write_index(run_dir=args.run_dir, scores=scores,
                        watching=bool(args.watch))
    write_status(status, phase="wave1-complete" if not args.watch
                 else "watching", arms_complete=len(scores),
                 arms_expected=int(args.expect_arms),
                 figures=len(gallery.manifest), gallery=str(args.out),
                 pid=os.getpid())

    if not args.watch:
        print(f"gallery: {len(gallery.manifest)} figures + index.html at "
              f"{args.out}", flush=True)
        return 0

    if args.run_dir is None:
        raise SystemExit("--watch needs --run-dir; there is nothing to "
                         "watch without the queue's run directory")

    deadline = time.monotonic() + float(args.max_hours) * 3600.0
    seen: set[str] = set(scores)
    while True:
        if time.monotonic() > deadline:
            write_status(status, phase="deadline",
                         arms_complete=len(seen),
                         arms_expected=int(args.expect_arms),
                         detail="the watcher reached its own deadline; "
                                "the queue was not touched",
                         gallery=str(args.out), pid=os.getpid())
            gallery.write_index(run_dir=args.run_dir,
                                scores=gallery.arm_scores(args.run_dir),
                                watching=False)
            return 0
        time.sleep(float(args.poll_seconds))
        scores = gallery.arm_scores(args.run_dir)
        if set(scores) != seen:
            seen = set(scores)
            try:
                gallery.fig_fss_curves(scores)
                gallery.fig_lead_row(args.run_dir, scores)
            except Exception as error:            # keep the watcher alive
                # A half-written score file or a missing composite is a
                # transient, not a reason to stop redrawing: the next
                # poll will find it complete.  Recorded, never silent.
                write_status(status, phase="redraw-failed",
                             arms_complete=len(seen),
                             arms_expected=int(args.expect_arms),
                             detail=f"{error.__class__.__name__}: {error}",
                             gallery=str(args.out), pid=os.getpid())
            gallery.write_index(run_dir=args.run_dir, scores=scores,
                                watching=True)
            write_status(status, phase="watching",
                         arms_complete=len(seen),
                         arms_expected=int(args.expect_arms),
                         figures=len(gallery.manifest),
                         gallery=str(args.out), pid=os.getpid())
        if args.final_arm in seen:
            gallery.write_index(run_dir=args.run_dir, scores=scores,
                                watching=False)
            write_status(status, phase="complete",
                         arms_complete=len(seen),
                         arms_expected=int(args.expect_arms),
                         figures=len(gallery.manifest),
                         gallery=str(args.out), pid=os.getpid())
            print("watcher: final arm scored; exiting", flush=True)
            return 0


if __name__ == "__main__":
    sys.exit(main())
