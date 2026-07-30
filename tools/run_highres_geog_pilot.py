#!/usr/bin/env python3
"""Run the hash-bound real74 Ohio high-resolution geography pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from gpuwm.static.build import GeogSelection, build_static
from gpuwm.static.highres import (
    BoundRaster,
    build_highres_overrides,
    merge_highres_overrides,
    sha256_file,
)
from gpuwm.static.lambert import LambertGrid


def _bound(payload, root: Path, defaults=None) -> BoundRaster:
    values = dict(defaults or {})
    values.update(payload)
    if "bytes" in values:
        values["expected_bytes"] = values.pop("bytes")
    ignored = {"component", "depth", "category_mapping",
               "water_policy"}
    kwargs = {key: value for key, value in values.items()
              if key not in ignored}
    path = Path(kwargs["path"])
    kwargs["path"] = path if path.is_absolute() else root / path
    return BoundRaster(**kwargs)


def _grid(manifest, target) -> LambertGrid:
    projection = manifest["pilot_window"]["projection"]
    ny, nx = (int(value) for value in target["mass_shape"])
    dx = float(target["dx_m"])
    return LambertGrid(
        ref_lat=float(manifest["pilot_window"]["center_lat"]),
        ref_lon=float(manifest["pilot_window"]["center_lon"]),
        truelat1=float(projection["truelat1"]),
        truelat2=float(projection["truelat2"]),
        stand_lon=float(projection["stand_lon"]),
        dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1,
    )


def _continuous_metrics(candidate, baseline) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    delta = candidate - baseline
    absolute = np.abs(delta)
    return {
        "baseline_min": float(baseline.min()),
        "baseline_max": float(baseline.max()),
        "candidate_min": float(candidate.min()),
        "candidate_max": float(candidate.max()),
        "mean_bias": float(delta.mean()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "max_abs": float(absolute.max()),
        "delta_p01": float(np.percentile(delta, 1.0)),
        "delta_p50": float(np.percentile(delta, 50.0)),
        "delta_p99": float(np.percentile(delta, 99.0)),
    }


def _roughness(value, dx: float) -> dict[str, float]:
    grad_y, grad_x = np.gradient(np.asarray(value, dtype=np.float64), dx, dx)
    slope = np.hypot(grad_x, grad_y)
    return {
        "slope_p95_m_per_m": float(np.percentile(slope, 95.0)),
        "slope_max_m_per_m": float(slope.max()),
    }


def _category_metrics(candidate, baseline) -> dict[str, object]:
    candidate = np.asarray(candidate)
    baseline = np.asarray(baseline)
    changed = candidate != baseline
    return {
        "agreement_fraction": float(np.mean(~changed)),
        "changed_cells": int(changed.sum()),
        "cell_count": int(changed.size),
        "baseline_histogram": {
            str(int(value)): int(count) for value, count in zip(
                *np.unique(baseline.astype(np.int64), return_counts=True))
        },
        "candidate_histogram": {
            str(int(value)): int(count) for value, count in zip(
                *np.unique(candidate.astype(np.int64), return_counts=True))
        },
    }


def _plot(path: Path, baseline, candidate, dx: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = candidate["HGT_M"] - baseline["HGT_M"]
    mask_delta = candidate["LANDMASK"] - baseline["LANDMASK"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    panels = (
        (baseline["HGT_M"], "30s GMTED terrain (m)", "terrain"),
        (candidate["HGT_M"], "3DEP terrain (m)", "terrain"),
        (delta, "3DEP - GMTED (m)", "coolwarm"),
        (baseline["LANDMASK"], "30s land mask", "gray"),
        (candidate["LANDMASK"], "NLCD-1985 land mask", "gray"),
        (mask_delta, "mask change (+land / -water)", "bwr"),
    )
    for axis, (values, title, cmap) in zip(axes.ravel(), panels):
        image = axis.imshow(values, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("west_east cell")
        axis.set_ylabel("south_north cell")
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle(f"RW-WPS high-resolution Ohio pilot; dx={dx:g} m")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pilot output {args.output}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rw-wps-highres-static-pack-v1":
        raise ValueError("high-resolution pilot manifest schema mismatch")
    if not manifest["historical_semantics"].get("anachronistic"):
        raise ValueError("1974 pilot must explicitly acknowledge land-cover age")
    root = manifest_path.parent
    terrain = _bound(manifest["terrain"], root)
    landcover = _bound(manifest["landcover"], root)
    landcover_mapping = {
        int(key): int(value)
        for key, value in manifest["landcover"]["category_mapping"].items()
    }
    soil_payload = manifest["soil"]
    soil_defaults = {
        "source_id": soil_payload["source_id"],
        "source_url": "https://maps.isric.org/mapserv",
        "license_id": soil_payload["license_id"],
        "license_url": soil_payload["license_url"],
        "nominal_resolution": soil_payload["nominal_resolution"],
        "crs_override": soil_payload["crs_override"],
        "nodata_override": soil_payload["nodata_override"],
        "scale_factor": soil_payload["scale_factor"],
        "reference_year": None,
    }
    soil_sources = {}
    for payload in soil_payload["sources"]:
        values = dict(payload)
        values["role"] = f"soil_{payload['component']}_{payload['depth']}"
        soil_sources[(payload["component"], payload["depth"])] = _bound(
            values, root, soil_defaults)

    args.output.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    results = []
    total_started = time.perf_counter()
    selection = GeogSelection.fallback(args.geog_root)
    geog_hashes = {
        str(selection.path(name).joinpath("index").resolve()):
            sha256_file(selection.path(name) / "index")
        for name in (
            "terrain", "landuse", "soil_top", "soil_bottom", "greenfrac",
            "lai", "albedo", "snow_albedo", "soil_temperature")
    }

    for target in manifest["pilot_window"]["target_grids"]:
        grid = _grid(manifest, target)
        label = f"{int(round(grid.dx))}m"
        baseline_started = time.perf_counter()
        baseline = build_static(grid, args.geog_root, selection=selection)
        baseline_seconds = time.perf_counter() - baseline_started
        highres_started = time.perf_counter()
        overrides, source_audit = build_highres_overrides(
            grid, terrain=terrain, landcover=landcover,
            soil_sources=soil_sources, soil_fallback=baseline,
            landcover_mapping=landcover_mapping,
        )
        highres_seconds = time.perf_counter() - highres_started
        merge_started = time.perf_counter()
        candidate, fallback_audit = merge_highres_overrides(
            baseline, overrides)
        merge_seconds = time.perf_counter() - merge_started

        terrain_metrics = _continuous_metrics(
            candidate["HGT_M"], baseline["HGT_M"])
        terrain_metrics["baseline_roughness"] = _roughness(
            baseline["HGT_M"], grid.dx)
        terrain_metrics["candidate_roughness"] = _roughness(
            candidate["HGT_M"], grid.dx)
        metrics = {
            "terrain": terrain_metrics,
            "landmask": _category_metrics(
                candidate["LANDMASK"], baseline["LANDMASK"]),
            "landuse": _category_metrics(
                candidate["LU_INDEX"], baseline["LU_INDEX"]),
            "soil_top": _category_metrics(
                candidate["SCT_DOM"], baseline["SCT_DOM"]),
            "soil_bottom": _category_metrics(
                candidate["SCB_DOM"], baseline["SCB_DOM"]),
            "tmn": _continuous_metrics(candidate["TMN"], baseline["TMN"]),
        }
        cache_path = args.output / f"comparison-{label}.npz"
        np.savez_compressed(cache_path, **{
            f"baseline_{name}": baseline[name]
            for name in ("HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM",
                         "SCB_DOM", "TMN")
        }, **{
            f"candidate_{name}": candidate[name]
            for name in ("HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM",
                         "SCB_DOM", "TMN")
        })
        plot_path = args.output / f"comparison-{label}.png"
        _plot(plot_path, baseline, candidate, grid.dx)
        results.append({
            "label": label,
            "dx_m": grid.dx,
            "mass_shape": [grid.e_sn - 1, grid.e_we - 1],
            "lat_range": [float(grid.latlon_mass()[0].min()),
                          float(grid.latlon_mass()[0].max())],
            "lon_range": [float(grid.latlon_mass()[1].min()),
                          float(grid.latlon_mass()[1].max())],
            "timing_seconds": {
                "baseline_30s_build": baseline_seconds,
                "highres_override_build": highres_seconds,
                "merge_required_noah_fields": merge_seconds,
            },
            "source_audit": source_audit,
            "fallback_audit": fallback_audit,
            "metrics": metrics,
            "evidence": {
                "cache": cache_path.name,
                "cache_bytes": cache_path.stat().st_size,
                "cache_sha256": sha256_file(cache_path),
                "plot": plot_path.name,
                "plot_bytes": plot_path.stat().st_size,
                "plot_sha256": sha256_file(plot_path),
            },
        })

    receipt = {
        "schema": "rw-wps-highres-static-pilot-receipt-v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "gpuwm native static build plus hash-bound high-res overlay; no geogrid.exe",
        "code_git_head": _git_head(repository),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "historical_semantics": manifest["historical_semantics"],
        "scope": {
            "certifies": "the two small Caesar Creek pilot grids and exact input hashes only",
            "does_not_certify": [
                "full real74 d04 coverage",
                "historically exact 1974 land cover",
                "coastal ocean/lake classification",
                "global fallback implementation",
                "forecast trajectory improvement",
                "stock-WRF numerical parity"
            ],
        },
        "baseline_wps_geog_index_sha256": geog_hashes,
        "results": results,
        "total_wall_seconds": time.perf_counter() - total_started,
    }
    receipt_path = args.output / "receipt.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, receipt_path)
    print(json.dumps({
        "status": "PASS",
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "total_wall_seconds": receipt["total_wall_seconds"],
        "results": [{
            "label": item["label"],
            "terrain_rmse_m": item["metrics"]["terrain"]["rmse"],
            "landmask_agreement": item["metrics"]["landmask"][
                "agreement_fraction"],
        } for item in results],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
