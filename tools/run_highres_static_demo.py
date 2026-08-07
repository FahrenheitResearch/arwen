#!/usr/bin/env python3
"""Build baseline-vs-highres statics for one config-declared domain pair.

Every geographic number comes from the demo TOML's WPS namelist; the
library path exercised here is exactly the production seam
(:func:`gpuwm.static.highres_production.apply_highres_statics` over the
30-arc-second baseline build).  Output layout: case -> domain -> product.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np

from gpuwm.case_data import expand_path_variables
from gpuwm.static.build import GeogSelection, build_static
from gpuwm.static.highres_production import (apply_highres_statics,
                                             parse_static_table)
from gpuwm.static.lambert import grids_from_wps_namelist


def _resolve(base: Path, value: str, key: str, source: str) -> Path:
    path = Path(expand_path_variables(value, key, source))
    return path if path.is_absolute() else base / path


def _terrain_metrics(before, after) -> dict[str, float]:
    delta = np.asarray(after, dtype=np.float64) - np.asarray(
        before, dtype=np.float64)
    return {
        "rmse_m": float(np.sqrt(np.mean(delta ** 2))),
        "mean_bias_m": float(delta.mean()),
        "max_abs_m": float(np.abs(delta).max()),
        "baseline_min_m": float(np.min(before)),
        "baseline_max_m": float(np.max(before)),
        "highres_min_m": float(np.min(after)),
        "highres_max_m": float(np.max(after)),
    }


def _category_changes(before, after) -> dict[str, object]:
    changed = np.asarray(before) != np.asarray(after)
    return {"changed_cells": int(changed.sum()),
            "cell_count": int(changed.size)}


def _plot_terrain(path: Path, before, after, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = np.asarray(after) - np.asarray(before)
    vmin = float(min(np.min(before), np.min(after)))
    vmax = float(max(np.max(before), np.max(after)))
    span = float(np.abs(delta).max()) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             constrained_layout=True)
    panels = (
        (before, "30-arc-second baseline terrain (m)", "terrain",
         dict(vmin=vmin, vmax=vmax)),
        (after, "3DEP 1/3 arc-second terrain (m)", "terrain",
         dict(vmin=vmin, vmax=vmax)),
        (delta, "highres - baseline (m)", "coolwarm",
         dict(vmin=-span, vmax=span)),
    )
    for axis, (values, title, cmap, kwargs) in zip(axes, panels):
        image = axis.imshow(values, origin="lower", cmap=cmap, **kwargs)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("west_east cell")
        axis.set_ylabel("south_north cell")
        fig.colorbar(image, ax=axis, shrink=0.85)
    fig.suptitle(f"Terrain elevation before/after -- {label}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_landuse(path: Path, before, after, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    cmap = plt.get_cmap("tab20b", 21)
    norm = colors.BoundaryNorm(np.arange(0.5, 22.5, 1.0), 21)
    changed = np.asarray(before) != np.asarray(after)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             constrained_layout=True)
    panels = (
        (before, "30s MODIS-21 land use (LU_INDEX)"),
        (after, "Annual NLCD -> MODIS-21 land use"),
    )
    for axis, (values, title) in zip(axes, panels):
        image = axis.imshow(values, origin="lower", cmap=cmap, norm=norm)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("west_east cell")
        axis.set_ylabel("south_north cell")
        fig.colorbar(image, ax=axis, shrink=0.85,
                     ticks=np.arange(1, 22, 2))
    image = axes[2].imshow(changed, origin="lower", cmap="gray_r")
    axes[2].set_title(
        f"changed cells: {int(changed.sum())}/{changed.size}", fontsize=10)
    axes[2].set_xlabel("west_east cell")
    axes[2].set_ylabel("south_north cell")
    fig.colorbar(image, ax=axes[2], shrink=0.85)
    fig.suptitle(f"Land use before/after -- {label}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="demo delivery root (case folder is created "
                             "beneath it)")
    args = parser.parse_args()

    config_path = args.config.resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    source = str(config_path)
    base = config_path.parent
    demo = raw["demo"]
    highres = parse_static_table(raw.get("static"), source=source,
                                 base_dir=base)
    if highres is None or not highres.enabled:
        raise SystemExit("demo config must enable [static.highres]")

    case_label = str(demo["case_label"])
    case_date = demo["case_date"]
    namelist = _resolve(base, demo["wps_namelist"], "wps_namelist", source)
    geog_root = _resolve(base, demo["geog_root"], "geog_root", source)
    selection = GeogSelection.fallback(geog_root)
    landuse_attrs = selection.landuse_global_attrs()
    grids = grids_from_wps_namelist(namelist)

    case_dir = args.output / case_label
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = {"case": case_label, "case_date": case_date.isoformat(),
               "config": str(config_path), "wps_namelist": str(namelist),
               "geog_root": str(geog_root), "domains": []}
    for index, grid in enumerate(grids):
        domain_id = index + 1
        label = f"d{domain_id:02d}-{grid.dx / 1000:g}km"
        domain_dir = case_dir / label
        (domain_dir / "terrain").mkdir(parents=True, exist_ok=True)
        (domain_dir / "landuse").mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        baseline = build_static(grid, geog_root, selection=selection)
        baseline_seconds = time.perf_counter() - started
        started = time.perf_counter()
        candidate, receipt = apply_highres_statics(
            baseline, grid, config=highres, domain_id=domain_id,
            case_date=case_date, landuse_attrs=landuse_attrs)
        highres_seconds = time.perf_counter() - started
        if receipt is None or receipt.get("status") != "APPLIED":
            raise SystemExit(
                f"demo domain {label} did not apply the high-resolution "
                f"path: {receipt}")

        receipt_copy = domain_dir / f"static_highres_receipt_{label}.json"
        shutil.copyfile(receipt["receipt_path"], receipt_copy)
        _plot_terrain(domain_dir / "terrain" / "terrain_before_after.png",
                      baseline["HGT_M"], candidate["HGT_M"], label)
        _plot_landuse(domain_dir / "landuse" / "landuse_before_after.png",
                      baseline["LU_INDEX"], candidate["LU_INDEX"], label)

        summary["domains"].append({
            "domain": label,
            "dx_m": grid.dx,
            "mass_shape": [grid.e_sn - 1, grid.e_we - 1],
            "terrain": _terrain_metrics(baseline["HGT_M"],
                                        candidate["HGT_M"]),
            "landuse": _category_changes(baseline["LU_INDEX"],
                                         candidate["LU_INDEX"]),
            "landmask": _category_changes(baseline["LANDMASK"],
                                          candidate["LANDMASK"]),
            "soil_top": _category_changes(baseline["SCT_DOM"],
                                          candidate["SCT_DOM"]),
            "cells_replaced": receipt["cells_replaced"],
            "three_dep_tiles": len(receipt["fetch"]["three_dep_tiles"]),
            "bytes_fetched": receipt["fetch"]["bytes_fetched"],
            "nlcd_year": receipt["fetch"]["nlcd_year"],
            "timing_seconds": {"baseline_30s": baseline_seconds,
                               "highres": highres_seconds},
            "receipt": str(receipt_copy),
        })

    summary_path = case_dir / "demo_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": "PASS", "summary": str(summary_path),
                      "domains": [d["domain"]
                                  for d in summary["domains"]]}))


if __name__ == "__main__":
    main()
