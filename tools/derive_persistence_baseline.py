"""Compute persistence from any declared case's own analyses.

The persistence forecast is the case's initial analysis held fixed until
the final analysis available to that case.  Both fields are remapped to the
case's configured Lambert mass grid with the CPU NumPy mirror of the same
WPS interpolation used by initialization, then scored by the reusable
verification metric and mask implementation.  The tool never imports a
case module, an external WRF oracle, or CUDA.

Examples::

    python tools/derive_persistence_baseline.py
    python tools/derive_persistence_baseline.py configs/may1999_d01_smoke.toml
    python tools/derive_persistence_baseline.py CASE.toml --field T2 --level none

With no path, the frozen real74 experiment config is used for backward
compatibility.  Printed values are computed actuals; this tool does not edit
or hand-derive any profile threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gpuwm.case_data import load_experiment_case
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.preflight import build_input_catalog
from gpuwm.runtime import experiment_grid, forcing_schedule, forcing_snapshots
from gpuwm.verify.metrics import score_pair
from gpuwm.verify.npref import interpolate_regular_np


DEFAULT_CONFIG = REPO / "configs" / "real74_d01_exp.toml"


@dataclass(frozen=True)
class PersistenceResult:
    """One computed initial-analysis persistence score."""

    case: str
    initial_valid_time: datetime
    final_valid_time: datetime
    field: str
    level_hpa: float | None
    mask: str
    interpolation: str
    rmse: float
    pattern_correlation: float


def analysis_field(snapshot: Era5Snapshot, field: str,
                   level_hpa: float | None) -> np.ndarray:
    """Select one surface or pressure-level analysis field as float64."""
    # Horizontal initialization renames ERA5's native pressure-level T/Z
    # to TT/GHT.  Accept either spelling so an AnalysisRecipe can be passed
    # straight to this source-analysis tool.
    source_name = field
    if source_name not in snapshot.fields:
        source_name = {"TT": "T", "GHT": "Z"}.get(field, field)
    if source_name not in snapshot.fields:
        raise ValueError(
            f"analysis {snapshot.valid_time} has no {field!r} field; "
            f"available: {sorted(snapshot.fields)}")
    value = snapshot.fields[source_name]
    if level_hpa is not None:
        if np.ndim(value) != 3:
            raise ValueError(
                f"field {field!r} is not a pressure-level analysis")
        levels = np.asarray(snapshot.levels_hpa, dtype=np.float64)
        index = int(np.argmin(np.abs(levels - level_hpa)))
        if abs(levels[index] - level_hpa) > 1.0e-9:
            raise ValueError(
                f"analysis {snapshot.valid_time} has no {level_hpa:g} hPa "
                f"level for {field!r}")
        value = value[index]
    elif np.ndim(value) != 2:
        raise ValueError(
            f"pressure-level field {field!r} requires --level HPA")
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def derive_persistence(
        initial: Era5Snapshot, final: Era5Snapshot, *, field: str = "TT",
        level_hpa: float | None = 500.0, mask: str = "interior",
        interpolation: str = "parabolic", target_latitude=None,
        target_longitude=None) -> tuple[float, float]:
    """Score an initial analysis held fixed against a final analysis."""
    if initial.valid_time >= final.valid_time:
        raise ValueError("final analysis must be later than initial analysis")
    if (not np.array_equal(initial.latitude, final.latitude)
            or not np.array_equal(initial.longitude, final.longitude)):
        raise ValueError("initial and final analyses use different source grids")
    forecast = analysis_field(initial, field, level_hpa)
    truth = analysis_field(final, field, level_hpa)
    if target_latitude is not None or target_longitude is not None:
        if target_latitude is None or target_longitude is None:
            raise ValueError(
                "target latitude and longitude must be supplied together")
        forecast = interpolate_regular_np(
            forecast, initial.latitude, initial.longitude,
            target_latitude, target_longitude, method=interpolation)
        truth = interpolate_regular_np(
            truth, final.latitude, final.longitude,
            target_latitude, target_longitude, method=interpolation)
    return score_pair(forecast, truth, mask=mask)


def derive_case_persistence(
        config_path: str | Path, *, field: str = "TT",
        level_hpa: float | None = 500.0, mask: str = "interior",
        interpolation: str | None = None) -> PersistenceResult:
    """Load a case and compute persistence from its first/final analyses."""
    exp, data = load_experiment_case(config_path)
    catalog = build_input_catalog(data)
    by_time = forcing_snapshots(data, catalog)
    times = forcing_schedule(exp, data, by_time)
    initial, final = by_time[times[0]], by_time[times[-1]]
    method = ("parabolic" if level_hpa is not None else "bilinear"
              ) if interpolation is None else interpolation
    grid = experiment_grid(exp, data)
    latitude, longitude = grid.latlon_mass()
    rmse, correlation = derive_persistence(
        initial, final, field=field, level_hpa=level_hpa, mask=mask,
        interpolation=method, target_latitude=latitude,
        target_longitude=longitude)
    return PersistenceResult(
        case=exp.name, initial_valid_time=initial.valid_time,
        final_valid_time=final.valid_time, field=field,
        level_hpa=level_hpa, mask=mask, interpolation=method,
        rmse=rmse, pattern_correlation=correlation)


def _level(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="compute initial-analysis persistence for any case TOML")
    parser.add_argument("config", nargs="?", type=Path,
                        default=DEFAULT_CONFIG, metavar="CASE.toml")
    parser.add_argument("--field", default="TT")
    parser.add_argument("--level", type=_level, default=500.0,
                        metavar="HPA|none")
    parser.add_argument("--mask", choices=("full", "interior"),
                        default="interior")
    parser.add_argument("--interpolation",
                        choices=("parabolic", "bilinear", "nearest"),
                        default=None)
    args = parser.parse_args(argv)
    result = derive_case_persistence(
        args.config, field=args.field, level_hpa=args.level, mask=args.mask,
        interpolation=args.interpolation)
    level = "surface" if result.level_hpa is None else f"{result.level_hpa:g} hPa"
    print(f"case={result.case}")
    print(f"initial_analysis={result.initial_valid_time.isoformat()}")
    print(f"final_analysis={result.final_valid_time.isoformat()}")
    print(f"field={result.field} level={level} mask={result.mask} "
          f"interpolation={result.interpolation}")
    print(f"rmse={result.rmse!r}")
    print(f"pattern_correlation={result.pattern_correlation!r}")
    print(f"rounded_rmse={round(result.rmse, 4)}")
    print(f"rounded_pattern_correlation="
          f"{round(result.pattern_correlation, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
