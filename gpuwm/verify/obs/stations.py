"""Surface verification against point observations.

Scoring a grid against a scatter of stations is where a verification system
most easily fools itself, so every step that could quietly manufacture skill
is made explicit and receipted here:

* **The station set is frozen once per case and shared by every arm.**  A set
  chosen per arm -- even innocently, by dropping stations an arm happens to
  miss -- makes the arms unpaired, and an unpaired difference of medians is
  not a difference.  :func:`freeze_station_set` returns the survivors *and*
  every drop with its reason, and the harness records both.
* **A station is dropped, never corrected.**  When model terrain and station
  elevation disagree by more than the registered tolerance, the honest move
  is to stop scoring that station; a lapse-rate adjustment would invent the
  agreement it is measuring.  The dropped count is published.
* **Time matching is one-sided about failure.**  A (station, hour) with no
  report inside the tolerance is missing for *every* arm, not just the arm
  that noticed.
* **Interpolation is registered, and the alternative is a control.**  The
  battery scores under bilinear interpolation of the model field to the
  station; nearest-neighbour exists here as the qualification sensitivity,
  not as a fallback that fires when bilinear is inconvenient.

Nothing here knows a case, a network, or a station id: the station list, the
masks and the tolerances all arrive as arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from gpuwm.verify.obs.contracts import (
    SCORED_SURFACE_VARIABLES, Station, StationObsSet, StationReport,
    parse_valid_time,
)

#: Registered gross-error screen, in seam SI units.  These are the published
#: bounds of physical plausibility for a surface report, not a tuning knob:
#: -40 C to 55 C for temperature and dewpoint, 0 to 75 m/s for wind speed.
GROSS_RANGE: Mapping[str, tuple[float, float]] = {
    "temperature_2m": (233.15, 328.15),
    "dewpoint_2m": (233.15, 328.15),
    "wind_speed_10m": (0.0, 75.0),
}

#: Drop reasons, evaluated in this order so a station's recorded reason is
#: the first thing that disqualified it rather than whichever check ran last.
DROP_OUTSIDE_DOMAIN = "outside-domain"
DROP_OUTSIDE_INTERIOR = "outside-interior-mask"
DROP_NOT_LAND = "not-land"
DROP_TERRAIN_MISMATCH = "terrain-mismatch"
DROP_SCREEN = "quality-screen-rate"
DROP_REPORTING = "insufficient-reporting"

DROP_REASONS = (DROP_OUTSIDE_DOMAIN, DROP_OUTSIDE_INTERIOR, DROP_NOT_LAND,
                DROP_TERRAIN_MISMATCH, DROP_SCREEN, DROP_REPORTING)

BILINEAR = "bilinear"
NEAREST = "nearest"
INTERPOLATIONS = (BILINEAR, NEAREST)


@dataclass(frozen=True)
class StationPosition:
    """A station's location in fractional grid indices.

    ``x`` runs along west_east and ``y`` along south_north, both zero-based
    and fractional, which is what :func:`bilinear_sample` consumes.  The
    mapping itself is not computed here -- it comes from the mandated science
    core's projection code -- because a second implementation of a map
    projection is a second set of answers.
    """

    station_id: str
    x: float
    y: float


def sample_field(field: np.ndarray, position: StationPosition, *,
                 method: str = BILINEAR) -> float:
    """Model field value at a station, by the registered interpolation."""
    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("a station sample needs a 2-D model field")
    ny, nx = array.shape
    x = float(position.x)
    y = float(position.y)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"station {position.station_id} has a non-finite position")
    if not (0.0 <= x <= nx - 1 and 0.0 <= y <= ny - 1):
        raise ValueError(
            f"station {position.station_id} at ({x:g}, {y:g}) is outside the "
            f"{ny}x{nx} grid; it should have been dropped before scoring")
    if method == NEAREST:
        return float(array[int(round(y)), int(round(x))])
    if method != BILINEAR:
        raise ValueError(f"unknown interpolation {method!r}; expected {INTERPOLATIONS}")
    i0 = min(int(math.floor(x)), nx - 2) if nx > 1 else 0
    j0 = min(int(math.floor(y)), ny - 2) if ny > 1 else 0
    i1 = min(i0 + 1, nx - 1)
    j1 = min(j0 + 1, ny - 1)
    tx = x - i0
    ty = y - j0
    return float(
        array[j0, i0] * (1.0 - tx) * (1.0 - ty)
        + array[j0, i1] * tx * (1.0 - ty)
        + array[j1, i0] * (1.0 - tx) * ty
        + array[j1, i1] * tx * ty)


def screen_report(report: StationReport) -> tuple[str, ...]:
    """Variables in one report that fail the registered gross-error screen.

    The dewpoint-above-temperature check is part of the screen and fires on
    the dewpoint, because a supersaturated surface report is a bad dewpoint
    far more often than it is a bad temperature.
    """
    failed: list[str] = []
    values = dict(report.values)
    for name, (low, high) in GROSS_RANGE.items():
        if name in values and not (low <= float(values[name]) <= high):
            failed.append(name)
    temperature = values.get("temperature_2m")
    dewpoint = values.get("dewpoint_2m")
    if (temperature is not None and dewpoint is not None
            and float(dewpoint) > float(temperature)
            and "dewpoint_2m" not in failed):
        failed.append("dewpoint_2m")
    return tuple(sorted(failed))


def match_reports(observations: StationObsSet, valid_times: Sequence[str], *,
                  tolerance_seconds: int
                  ) -> dict[tuple[str, str], StationReport]:
    """Nearest report to each valid time, per station, inside the tolerance.

    Ties -- two reports equidistant from the hour -- resolve to the earlier
    one, so the mapping is a function of the data and not of iteration order.
    """
    if tolerance_seconds <= 0:
        raise ValueError("the report matching tolerance must be positive")
    targets = [(text, parse_valid_time(text)) for text in valid_times]
    matched: dict[tuple[str, str], StationReport] = {}
    for station_id, reports in observations.by_station().items():
        stamped = [(parse_valid_time(report.valid_time), report)
                   for report in reports]
        for text, target in targets:
            best: tuple[float, str, StationReport] | None = None
            for instant, report in stamped:
                offset = abs((instant - target).total_seconds())
                if offset > tolerance_seconds:
                    continue
                candidate = (offset, report.valid_time, report)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            if best is not None:
                matched[(station_id, text)] = best[2]
    return matched


@dataclass(frozen=True, eq=False)
class FrozenStationSet:
    """The stations a case scores, and every station it does not, with reasons."""

    station_ids: tuple[str, ...]
    positions: Mapping[str, StationPosition]
    drops: tuple[dict[str, object], ...]
    parameters: Mapping[str, object]

    def record(self) -> dict[str, object]:
        counts: dict[str, int] = {reason: 0 for reason in DROP_REASONS}
        for drop in self.drops:
            counts[str(drop["reason"])] += 1
        return {
            "station_count": len(self.station_ids),
            "station_ids": list(self.station_ids),
            "dropped_count": len(self.drops),
            "dropped_by_reason": counts,
            "drops": [dict(drop) for drop in self.drops],
            "parameters": dict(self.parameters),
        }


def freeze_station_set(
        stations: Sequence[Station],
        positions: Mapping[str, StationPosition], *,
        observations: StationObsSet,
        valid_times: Sequence[str],
        interior_mask: np.ndarray,
        land_mask: np.ndarray,
        terrain_m: np.ndarray,
        elevation_tolerance_m: float,
        minimum_reporting_fraction: float,
        match_tolerance_seconds: int,
        maximum_screen_fraction: float) -> FrozenStationSet:
    """Apply the registered admission rules once, for every arm of a case."""
    if not 0.0 <= minimum_reporting_fraction <= 1.0:
        raise ValueError("the reporting fraction must lie in [0, 1]")
    if not 0.0 <= maximum_screen_fraction <= 1.0:
        raise ValueError("the screen fraction must lie in [0, 1]")
    if elevation_tolerance_m <= 0.0:
        raise ValueError("the elevation tolerance must be positive")
    interior = np.asarray(interior_mask, dtype=bool)
    land = np.asarray(land_mask, dtype=bool)
    terrain = np.asarray(terrain_m, dtype=np.float64)
    if interior.shape != land.shape or interior.shape != terrain.shape:
        raise ValueError("interior, land and terrain masks must share one grid")
    ny, nx = interior.shape
    hours = list(valid_times)
    if not hours:
        raise ValueError("a frozen station set needs at least one valid time")
    matched = match_reports(observations, hours,
                            tolerance_seconds=match_tolerance_seconds)

    kept: list[str] = []
    drops: list[dict[str, object]] = []
    for station in stations:
        position = positions.get(station.station_id)
        if position is None:
            drops.append({"station_id": station.station_id,
                          "reason": DROP_OUTSIDE_DOMAIN,
                          "detail": "no grid position"})
            continue
        i = int(round(position.x))
        j = int(round(position.y))
        if not (0 <= i < nx and 0 <= j < ny
                and 0.0 <= position.x <= nx - 1
                and 0.0 <= position.y <= ny - 1):
            drops.append({"station_id": station.station_id,
                          "reason": DROP_OUTSIDE_DOMAIN,
                          "detail": f"({position.x:g}, {position.y:g})"})
            continue
        if not bool(interior[j, i]):
            drops.append({"station_id": station.station_id,
                          "reason": DROP_OUTSIDE_INTERIOR, "detail": ""})
            continue
        if not bool(land[j, i]):
            drops.append({"station_id": station.station_id,
                          "reason": DROP_NOT_LAND, "detail": ""})
            continue
        offset = float(terrain[j, i]) - float(station.elevation_m)
        if abs(offset) > float(elevation_tolerance_m):
            drops.append({"station_id": station.station_id,
                          "reason": DROP_TERRAIN_MISMATCH,
                          "detail": f"{offset:+.1f} m"})
            continue
        reports = [matched[(station.station_id, hour)] for hour in hours
                   if (station.station_id, hour) in matched]
        fired = sum(1 for report in reports if screen_report(report))
        if reports and fired / len(reports) > float(maximum_screen_fraction):
            drops.append({"station_id": station.station_id,
                          "reason": DROP_SCREEN,
                          "detail": f"{fired}/{len(reports)}"})
            continue
        clean = len(reports) - fired
        if clean < math.ceil(float(minimum_reporting_fraction) * len(hours)):
            drops.append({"station_id": station.station_id,
                          "reason": DROP_REPORTING,
                          "detail": f"{clean}/{len(hours)}"})
            continue
        kept.append(station.station_id)

    return FrozenStationSet(
        station_ids=tuple(sorted(kept)),
        positions={station_id: positions[station_id] for station_id in kept},
        drops=tuple(drops),
        parameters={
            "elevation_tolerance_m": float(elevation_tolerance_m),
            "minimum_reporting_fraction": float(minimum_reporting_fraction),
            "match_tolerance_seconds": int(match_tolerance_seconds),
            "maximum_screen_fraction": float(maximum_screen_fraction),
            "gross_range": {name: list(bounds)
                            for name, bounds in sorted(GROSS_RANGE.items())},
            "valid_time_count": len(hours),
        })


def shuffle_positions(positions: Mapping[str, StationPosition], *, seed: int
                      ) -> dict[str, StationPosition]:
    """A derangement of station-to-gridpoint assignment, for the control.

    Every station receives another station's grid position and none keeps its
    own, so a scorer that never actually reads the location cannot come out
    unchanged.  Fixed points are excluded rather than tolerated: with even one
    station left in place, "RMSE went up" is weaker evidence than it looks.
    """
    ids = sorted(positions)
    if len(ids) < 2:
        raise ValueError("a station shuffle needs at least two stations")
    generator = np.random.default_rng(int(seed))
    for _ in range(1000):
        order = generator.permutation(len(ids))
        if all(order[index] != index for index in range(len(ids))):
            break
    else:  # pragma: no cover - astronomically unlikely, still not silent
        raise RuntimeError("no derangement found for the station shuffle")
    return {
        ids[index]: StationPosition(
            station_id=ids[index],
            x=positions[ids[int(order[index])]].x,
            y=positions[ids[int(order[index])]].y)
        for index in range(len(ids))}


@dataclass(frozen=True)
class VariableScore:
    """Bias and RMSE for one variable over one arm of one case."""

    variable: str
    bias: float
    rmse: float
    sample_count: int
    station_count: int
    median_station_rmse: float
    station_rmse: Mapping[str, float]
    hourly_rmse: Mapping[int, float]
    hourly_bias: Mapping[int, float]

    def record(self) -> dict[str, object]:
        return {
            "variable": self.variable, "bias": self.bias, "rmse": self.rmse,
            "sample_count": int(self.sample_count),
            "station_count": int(self.station_count),
            "median_station_rmse": self.median_station_rmse,
            "station_rmse": {key: float(value)
                             for key, value in sorted(self.station_rmse.items())},
            "hourly_rmse": {str(hour): float(value)
                            for hour, value in sorted(self.hourly_rmse.items())},
            "hourly_bias": {str(hour): float(value)
                            for hour, value in sorted(self.hourly_bias.items())},
        }


def _rmse(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("RMSE over an empty sample is undefined")
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def surface_scores(
        frozen: FrozenStationSet, *,
        matched: Mapping[tuple[str, str], StationReport],
        model_value: Callable[[str, str, str], float | None],
        valid_times: Sequence[str],
        variables: Sequence[str] = SCORED_SURFACE_VARIABLES,
        ) -> dict[str, VariableScore]:
    """Bias and RMSE over (station x hour), plus the diurnal decomposition.

    ``model_value(station_id, valid_time, variable)`` returns the model's
    value at that station and hour, or ``None`` when the arm has no frame
    there.  The guardrail scalar the promotion rule consumes is
    ``median_station_rmse``: the median over stations of each station's own
    RMSE.  Median over stations rather than a pooled RMSE, because a pooled
    number is dominated by whichever handful of stations sit in the day's
    convection, and the guardrail is meant to catch a broken surface
    *climate*, not a displaced storm.
    """
    scores: dict[str, VariableScore] = {}
    for variable in variables:
        residual_by_station: dict[str, list[float]] = {}
        residual_by_hour: dict[int, list[float]] = {}
        all_residuals: list[float] = []
        for station_id in frozen.station_ids:
            for text in valid_times:
                report = matched.get((station_id, text))
                if report is None:
                    continue
                if variable in screen_report(report):
                    continue
                if variable not in report.values:
                    continue
                modelled = model_value(station_id, text, variable)
                if modelled is None:
                    continue
                residual = float(modelled) - float(report.values[variable])
                if not math.isfinite(residual):
                    raise ValueError(
                        f"{variable} residual at {station_id}/{text} is "
                        f"non-finite")
                residual_by_station.setdefault(station_id, []).append(residual)
                residual_by_hour.setdefault(
                    parse_valid_time(text).hour, []).append(residual)
                all_residuals.append(residual)
        if not all_residuals:
            raise ValueError(
                f"{variable} has no matched (station, hour) pairs to score")
        per_station = {station_id: _rmse(values)
                       for station_id, values in residual_by_station.items()}
        scores[variable] = VariableScore(
            variable=variable,
            bias=float(np.mean(all_residuals, dtype=np.float64)),
            rmse=_rmse(all_residuals),
            sample_count=len(all_residuals),
            station_count=len(per_station),
            median_station_rmse=float(
                np.median(np.asarray(sorted(per_station.values()),
                                     dtype=np.float64))),
            station_rmse=per_station,
            hourly_rmse={hour: _rmse(values)
                         for hour, values in residual_by_hour.items()},
            hourly_bias={hour: float(np.mean(values, dtype=np.float64))
                         for hour, values in residual_by_hour.items()},
        )
    return scores


__all__ = [
    "BILINEAR", "DROP_NOT_LAND", "DROP_OUTSIDE_DOMAIN", "DROP_OUTSIDE_INTERIOR",
    "DROP_REASONS", "DROP_REPORTING", "DROP_SCREEN", "DROP_TERRAIN_MISMATCH",
    "GROSS_RANGE", "INTERPOLATIONS", "NEAREST", "FrozenStationSet",
    "StationPosition", "VariableScore", "freeze_station_set", "match_reports",
    "sample_field", "screen_report", "shuffle_positions", "surface_scores",
]
