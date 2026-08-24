#!/usr/bin/env python
"""Draw what a generated MPAS mesh actually looks like.

Four analysis charts, from the grid file's own arrays:

1. cell size across the whole sphere, so the coarse background and the
   refined region are one picture;
2. the transition: delivered spacing against great-circle distance from
   the refinement centre, with the requested field over it;
3. the mesh itself over the refined region -- real Voronoi cells drawn
   from ``verticesOnCell``, not a scatter of dots;
4. the cell-size histogram beside the published uniform mesh a 10 GiB
   card is otherwise stuck with.

These are MESH DIAGNOSTICS, not weather fields, so matplotlib is the
right tool; anything carrying a forecast field goes through the Rust
renderer instead.

Every length is converted off the unit sphere before it is printed: MPAS
grid files carry ``sphere_radius = 1.0`` and ``areaCell``/``dcEdge`` are
therefore dimensionless.  A naive read prints kilometre figures near
zero, which has already cost two attempts at this measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import numpy as np                        # noqa: E402
from netCDF4 import Dataset               # noqa: E402

EARTH_RADIUS_M = 6_371_229.0


def read_mesh(path: Path) -> dict:
    """Cell centres, spacing and cell polygons, all in real units."""

    with Dataset(str(path)) as ds:
        radius = float(getattr(ds, "sphere_radius", 0.0) or 0.0)
        if radius <= 0.0:
            raise SystemExit(
                f"{path} carries no positive sphere_radius, so no length "
                "in it can be converted to metres")
        lat = np.asarray(ds.variables["latCell"][:], dtype=float)
        lon = np.asarray(ds.variables["lonCell"][:], dtype=float)
        area = np.asarray(ds.variables["areaCell"][:], dtype=float)
        dc = np.asarray(ds.variables["dcEdge"][:], dtype=float)
        cells_on_edge = np.asarray(
            ds.variables["cellsOnEdge"][:], dtype=np.int64)
        n_edges_on_cell = np.asarray(
            ds.variables["nEdgesOnCell"][:], dtype=np.int64)
        vertices_on_cell = np.asarray(
            ds.variables["verticesOnCell"][:], dtype=np.int64)
        lat_v = np.asarray(ds.variables["latVertex"][:], dtype=float)
        lon_v = np.asarray(ds.variables["lonVertex"][:], dtype=float)
        density = np.asarray(ds.variables["meshDensity"][:], dtype=float)
        boundary = getattr(ds, "rw_mesh_boundary", "")
        request = getattr(ds, "rw_mesh_request", "")
        spec = getattr(ds, "rw_mesh_spec", "")

    # The unit-sphere trap: every length below is scaled off it once,
    # here, and nowhere else.
    scale = EARTH_RADIUS_M / radius
    dc_m = dc * scale
    area_m2 = area * scale * scale

    # Per-cell spacing: the mean of the edge lengths around the cell.
    # Built from cellsOnEdge rather than edgesOnCell so the padding
    # slots (a zero in a 1-based index array) cannot be counted.
    total = np.zeros(lat.size)
    count = np.zeros(lat.size)
    for side in (0, 1):
        owner = cells_on_edge[:, side] - 1
        keep = owner >= 0
        np.add.at(total, owner[keep], dc_m[keep])
        np.add.at(count, owner[keep], 1.0)
    spacing_km = np.where(count > 0, total / np.maximum(count, 1.0), np.nan) \
        / 1000.0

    return {
        "lat_deg": np.degrees(lat),
        "lon_deg": ((np.degrees(lon) + 180.0) % 360.0) - 180.0,
        "spacing_km": spacing_km,
        "area_km2": area_m2 / 1e6,
        "dc_km": dc_m / 1000.0,
        "n_edges_on_cell": n_edges_on_cell,
        "vertices_on_cell": vertices_on_cell,
        "lat_vertex_deg": np.degrees(lat_v),
        "lon_vertex_deg": ((np.degrees(lon_v) + 180.0) % 360.0) - 180.0,
        "mesh_density": density,
        "sphere_radius": radius,
        "boundary": boundary,
        "request": request,
        "spec": spec,
        "n_cells": lat.size,
    }


def great_circle_deg(lat0: float, lon0: float,
                     lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distance in degrees, by the haversine form."""

    p0, p1 = np.radians(lat0), np.radians(lat)
    dphi = p1 - p0
    dlam = np.radians(lon - lon0)
    a = np.sin(dphi / 2.0) ** 2 + \
        np.cos(p0) * np.cos(p1) * np.sin(dlam / 2.0) ** 2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def plot_global(mesh: dict, centre: tuple[float, float],
                out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.6), dpi=140)
    order = np.argsort(-mesh["spacing_km"])       # fine cells drawn last
    sc = ax.scatter(mesh["lon_deg"][order], mesh["lat_deg"][order],
                    c=mesh["spacing_km"][order], s=1.4, cmap="viridis_r",
                    linewidths=0.0)
    ax.plot([centre[1]], [centre[0]], marker="+", color="crimson",
            markersize=14, markeredgewidth=2.0,
            label="centre of the refined region")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude (degrees east)")
    ax.set_ylabel("latitude (degrees north)")
    ax.set_title(title)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    bar = fig.colorbar(sc, ax=ax, pad=0.015)
    bar.set_label("cell spacing (km)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_transition(mesh: dict, centre: tuple[float, float],
                    requested: dict, out: Path, title: str) -> None:
    d = great_circle_deg(centre[0], centre[1],
                         mesh["lat_deg"], mesh["lon_deg"])
    d_km = np.radians(d) * EARTH_RADIUS_M / 1000.0
    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=140)
    ax.scatter(d_km, mesh["spacing_km"], s=1.2, alpha=0.30, color="#2b6cb0",
               linewidths=0.0, label="one dot per cell (delivered)")
    # Binned median, so the eye follows the field rather than the scatter.
    bins = np.linspace(0.0, d_km.max(), 90)
    idx = np.digitize(d_km, bins)
    xs, ys = [], []
    for b in range(1, bins.size):
        hit = idx == b
        if hit.sum() >= 8:
            xs.append(0.5 * (bins[b - 1] + bins[b]))
            ys.append(float(np.median(mesh["spacing_km"][hit])))
    ax.plot(xs, ys, color="#c53030", linewidth=2.2,
            label="median delivered spacing")
    radius = requested.get("radius_km")
    if radius:
        ax.axvline(radius, color="black", linestyle="--", linewidth=1.2,
                   label=f"edge of the refined region ({radius:g} km)")
    for key, colour, label in (
            ("spacing_km", "#2f855a", "requested inside the region"),
            ("background_km", "#975a16", "requested background")):
        value = requested.get(key)
        if value:
            ax.axhline(value, color=colour, linestyle=":", linewidth=1.6,
                       label=f"{label} ({value:g} km)")
    ax.set_xlabel("great-circle distance from the centre of the refined "
                  "region (km)")
    ax.set_ylabel("cell spacing (km)")
    ax.set_title(title)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_cells(mesh: dict, centre: tuple[float, float], half_deg: float,
               out: Path, title: str) -> None:
    """The real Voronoi cells over the refined region."""

    lat0, lon0 = centre
    lat = mesh["lat_deg"]
    lon = mesh["lon_deg"]
    near = (np.abs(lat - lat0) <= half_deg) & \
           (np.abs(((lon - lon0 + 180.0) % 360.0) - 180.0)
            <= half_deg / max(np.cos(np.radians(lat0)), 0.2))
    picks = np.flatnonzero(near)
    lat_v = mesh["lat_vertex_deg"]
    lon_v = mesh["lon_vertex_deg"]
    polygons, values = [], []
    for cell in picks:
        n = int(mesh["n_edges_on_cell"][cell])
        ring = mesh["vertices_on_cell"][cell, :n] - 1
        if np.any(ring < 0):
            continue
        px = lon_v[ring]
        py = lat_v[ring]
        # Unwrap a ring that straddles the antimeridian relative to the
        # cell centre, so a polygon cannot be drawn across the whole map.
        px = lon[cell] + (((px - lon[cell] + 180.0) % 360.0) - 180.0)
        polygons.append(np.column_stack([px, py]))
        values.append(mesh["spacing_km"][cell])
    fig, ax = plt.subplots(figsize=(9.5, 8.6), dpi=150)
    collection = PolyCollection(
        polygons, array=np.asarray(values), cmap="viridis_r",
        edgecolors="black", linewidths=0.28)
    ax.add_collection(collection)
    ax.set_xlim(lon0 - half_deg / max(np.cos(np.radians(lat0)), 0.2),
                lon0 + half_deg / max(np.cos(np.radians(lat0)), 0.2))
    ax.set_ylim(lat0 - half_deg, lat0 + half_deg)
    ax.set_xlabel("longitude (degrees east)")
    ax.set_ylabel("latitude (degrees north)")
    ax.set_title(title)
    bar = fig.colorbar(collection, ax=ax, pad=0.02)
    bar.set_label("cell spacing (km)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(mesh: dict, reference: dict | None,
                   out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.0), dpi=140)
    ax.hist(mesh["spacing_km"], bins=90, color="#2b6cb0", alpha=0.85,
            label=f"generated mesh ({mesh['n_cells']:,} cells)")
    if reference is not None:
        ax.hist(reference["spacing_km"], bins=90, color="#c53030",
                alpha=0.55,
                label=f"published uniform mesh "
                      f"({reference['n_cells']:,} cells)")
    ax.set_xlabel("cell spacing (km)")
    ax.set_ylabel("number of cells")
    ax.set_title(title)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="draw the cell-size structure of an MPAS grid file")
    parser.add_argument("grid", help="the generated grid file")
    parser.add_argument("--reference", help="a published grid file to "
                                            "compare the histogram against")
    parser.add_argument("--centre", required=True,
                        metavar="LAT,LON",
                        help="centre of the refined region, for the "
                             "transition and close-up charts")
    parser.add_argument("--radius-km", type=float, default=None)
    parser.add_argument("--resolution-km", type=float, default=None)
    parser.add_argument("--background-km", type=float, default=None)
    parser.add_argument("--half-deg", type=float, default=9.0,
                        help="half-width of the close-up window")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--prefix", default="mesh")
    args = parser.parse_args(argv)

    lat0, lon0 = (float(v) for v in args.centre.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh = read_mesh(Path(args.grid))
    reference = read_mesh(Path(args.reference)) if args.reference else None

    requested = {"radius_km": args.radius_km,
                 "spacing_km": args.resolution_km,
                 "background_km": args.background_km}

    finest = float(np.nanmin(mesh["spacing_km"]))
    coarsest = float(np.nanmax(mesh["spacing_km"]))
    plot_global(
        mesh, (lat0, lon0), out / f"{args.prefix}-01-cell-size-global.png",
        f"Cell size across the whole planet: {finest:.1f} km at the "
        f"finest, {coarsest:.0f} km at the coarsest, "
        f"{mesh['n_cells']:,} cells")
    plot_transition(
        mesh, (lat0, lon0), requested,
        out / f"{args.prefix}-02-refinement-transition.png",
        "How the mesh coarsens away from the refined region")
    plot_cells(
        mesh, (lat0, lon0), args.half_deg,
        out / f"{args.prefix}-03-cells-over-region.png",
        "The mesh itself over the refined region (every polygon is one "
        "model cell)")
    plot_histogram(
        mesh, reference, out / f"{args.prefix}-04-cell-size-histogram.png",
        "Where the cells went: a generated mesh spends them unevenly")

    summary = {
        "grid": str(Path(args.grid).resolve()),
        "n_cells": int(mesh["n_cells"]),
        "sphere_radius_in_file": mesh["sphere_radius"],
        "spacing_km": {
            "min": finest, "max": coarsest,
            "median": float(np.nanmedian(mesh["spacing_km"])),
            "ratio": coarsest / finest,
        },
        "area_km2": {"min": float(np.nanmin(mesh["area_km2"])),
                     "max": float(np.nanmax(mesh["area_km2"])),
                     "sum_over_earth": float(
                         mesh["area_km2"].sum()
                         / (4.0 * np.pi * (EARTH_RADIUS_M / 1000.0) ** 2))},
        "boundary": mesh["boundary"],
    }
    if args.radius_km:
        d = great_circle_deg(lat0, lon0, mesh["lat_deg"], mesh["lon_deg"])
        d_km = np.radians(d) * EARTH_RADIUS_M / 1000.0
        inside = d_km <= args.radius_km
        summary["inside_refined_region"] = {
            "cells": int(inside.sum()),
            "spacing_km_median": float(np.nanmedian(
                mesh["spacing_km"][inside])),
            "spacing_km_min": float(np.nanmin(mesh["spacing_km"][inside])),
            "spacing_km_max": float(np.nanmax(mesh["spacing_km"][inside])),
        }
    if reference is not None:
        summary["reference"] = {
            "n_cells": int(reference["n_cells"]),
            "spacing_km_median": float(
                np.nanmedian(reference["spacing_km"])),
        }
    (out / f"{args.prefix}-measurements.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
