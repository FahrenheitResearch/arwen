"""Data-driven verification profiles for real-weather experiments.

The runtime reports raw state and integration diagnostics.  This module
turns those facts into three independent policy layers:

* :class:`HealthProfile` is oracle-free and suitable for every run;
* :class:`AnalysisProfile` describes comparisons with the case's own
  analyses; and
* :class:`OracleProfile` optionally declares external WRF references.

All profile records are frozen.  Numeric gates retain gpuwm's historical
strict open-interval convention: ``lower < value < upper``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Threshold:
    """One named strict numeric gate."""

    name: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("threshold name must be non-empty")
        if self.lower is None and self.upper is None:
            raise ValueError(f"threshold {self.name!r} has no bound")
        for label, value in (("lower", self.lower), ("upper", self.upper)):
            if value is not None and not np.isfinite(value):
                raise ValueError(
                    f"threshold {self.name!r} {label} must be finite")
        if (self.lower is not None and self.upper is not None
                and self.lower >= self.upper):
            raise ValueError(
                f"threshold {self.name!r} lower must be below upper")

    @property
    def interval(self) -> tuple[float | None, float | None]:
        return self.lower, self.upper

    def accepts(self, value: object) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(numeric)
                    and (self.lower is None or self.lower < numeric)
                    and (self.upper is None or numeric < self.upper))


@dataclass(frozen=True)
class OutputSchema:
    """Generic WRF-history inventory and shape contract."""

    required_variables: tuple[str, ...] = (
        "Times", "U", "V", "W", "T", "P", "PB", "PH", "PHB",
        "QVAPOR", "XLAT", "XLONG", "T2",
    )
    require_finite_numeric: bool = True
    require_complete_publication: bool = True

    def __post_init__(self) -> None:
        if not self.required_variables:
            raise ValueError("output schema must require at least one variable")
        if any(not name for name in self.required_variables):
            raise ValueError("output schema variable names must be non-empty")
        if len(set(self.required_variables)) != len(self.required_variables):
            raise ValueError("output schema variable names must be unique")

    def failures(
            self, paths, *, start_time: datetime, run_seconds: float,
            history_interval_s: float, domain_id: int,
            nx: int, ny: int, nz: int) -> tuple[str, ...]:
        """Validate output calendar, filenames, dimensions, and inventory."""
        from gpuwm import netcdf_bridge
        from gpuwm.io.wrfout import wrfout_filename

        paths = tuple(Path(path) for path in paths)
        offsets = tuple(range(
            0, int(round(run_seconds)) + 1,
            int(round(history_interval_s))))
        expected_names = tuple(
            wrfout_filename(
                start_time + timedelta(seconds=offset), domain_id=domain_id)
            for offset in offsets)
        failures = []
        actual_names = tuple(path.name for path in paths)
        if actual_names != expected_names:
            failures.append(
                f"output calendar mismatch: expected {expected_names}, "
                f"got {actual_names}")
        expected_dims = {
            "west_east": nx, "south_north": ny, "bottom_top": nz,
            "west_east_stag": nx + 1, "south_north_stag": ny + 1,
            "bottom_top_stag": nz + 1, "DateStrLen": 19,
        }
        for path in paths:
            if not path.is_file():
                failures.append(f"output is missing: {path}")
                continue
            try:
                with netcdf_bridge.open_dataset(path) as ds:
                    if (self.require_complete_publication
                            and int(getattr(ds, "GPUWM_WRITE_COMPLETE", 0)) != 1):
                        failures.append(
                            f"output publication is incomplete: {path}")
                    for name, size in expected_dims.items():
                        if name not in ds.dimensions:
                            failures.append(f"{path.name} missing dimension {name}")
                        elif len(ds.dimensions[name]) != size:
                            failures.append(
                                f"{path.name} dimension {name} has "
                                f"{len(ds.dimensions[name])}, expected {size}")
                    missing = tuple(name for name in self.required_variables
                                    if name not in ds.variables)
                    if missing:
                        failures.append(
                            f"{path.name} missing variables {missing}")
                    if self.require_finite_numeric:
                        for name in self.required_variables:
                            if name == "Times" or name not in ds.variables:
                                continue
                            value = np.asarray(ds.variables[name][:])
                            if not np.isfinite(value).all():
                                failures.append(
                                    f"{path.name} variable {name} is non-finite")
            except OSError as exc:
                failures.append(f"cannot open output {path}: {exc}")
        return tuple(failures)


@dataclass(frozen=True)
class HealthReport:
    """Result of applying one :class:`HealthProfile`."""

    metrics: Mapping[str, object]
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def require_ok(self) -> None:
        if self.failures:
            raise AssertionError("; ".join(self.failures))


@dataclass(frozen=True)
class HealthProfile:
    """Always-on, oracle-free runtime-health policy.

    The profile covers finite state, optional CFL and vertical-velocity
    bounds, completed model clock, forcing coverage, separate boundary and
    interior diagnostics, and the on-disk output schema.  ``thresholds``
    hosts additional raw summary metrics (for example guard-fire counts)
    without moving their constants into the runtime.
    """

    name: str
    finite_state: bool = True
    cfl_bound: Threshold | None = None
    vertical_velocity_bound: Threshold | None = None
    completed_clock: bool = True
    forcing_coverage: bool = True
    boundary_diagnostics: bool = True
    interior_diagnostics: bool = True
    output_schema: OutputSchema | None = None
    thresholds: tuple[Threshold, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("health profile name must be non-empty")
        names = [threshold.name for threshold in self.all_thresholds]
        if len(set(names)) != len(names):
            raise ValueError("health profile threshold names must be unique")

    @property
    def all_thresholds(self) -> tuple[Threshold, ...]:
        optional = tuple(
            value for value in (self.cfl_bound, self.vertical_velocity_bound)
            if value is not None)
        return optional + self.thresholds

    @classmethod
    def generic_real_case(cls, *, name: str = "generic-real-case",
                          output_schema: OutputSchema | None = None
                          ) -> "HealthProfile":
        """The generic experiment-run health policy used by ARC-A smoke."""
        return cls(
            name=name, output_schema=output_schema,
            thresholds=(Threshold("ysu_nan_guard_fires", upper=0.5),))

    def gates(self) -> dict[str, tuple[float | None, float | None]]:
        return {threshold.name: threshold.interval
                for threshold in self.all_thresholds}

    def evaluate(
            self, summary, *, start_time: datetime | None = None,
            expected_completed_seconds: float | None = None,
            forcing_times=(), history_interval_s: float | None = None,
            domain_id: int = 1, nx: int | None = None,
            ny: int | None = None, nz: int | None = None,
            metric_values: Mapping[str, object] | None = None
            ) -> HealthReport:
        """Apply this policy to a raw runtime summary and run context."""
        metrics = dict(metric_values or {})
        failures: list[str] = []

        if self.finite_state:
            nan_free = bool(getattr(summary, "nan_free", False))
            metrics["nan"] = not nan_free
            if not nan_free:
                failures.append("runtime state is non-finite")

        diagnostic_names = (
            "w_max_ms", "boundary_w_max_ms", "interior_w_max_ms")
        for name in diagnostic_names:
            if hasattr(summary, name):
                metrics.setdefault(name, getattr(summary, name))

        if self.completed_clock:
            actual = getattr(summary, "completed_seconds", None)
            metrics["completed_seconds"] = actual
            if expected_completed_seconds is None:
                failures.append("expected completed clock was not supplied")
            elif (actual is None or not np.isfinite(actual)
                  or not np.isclose(
                      actual, expected_completed_seconds, rtol=0.0,
                      atol=max(1.0e-9,
                               abs(expected_completed_seconds) * 1.0e-12))):
                failures.append(
                    f"completed clock {actual!r} does not equal "
                    f"{expected_completed_seconds!r} s")

        forcing_times = tuple(forcing_times)
        if self.forcing_coverage:
            if (start_time is None or expected_completed_seconds is None
                    or not forcing_times):
                failures.append("forcing coverage context was not supplied")
            else:
                end_time = start_time + timedelta(
                    seconds=expected_completed_seconds)
                covered = (forcing_times[0] <= start_time
                           and forcing_times[-1] >= end_time)
                metrics["forcing_coverage"] = covered
                if not covered:
                    failures.append(
                        f"forcing {forcing_times[0]} .. {forcing_times[-1]} "
                        f"does not cover {start_time} .. {end_time}")

        if self.boundary_diagnostics or self.interior_diagnostics:
            values = [getattr(summary, name, None)
                      for name in diagnostic_names]
            if any(value is None or not np.isfinite(value) for value in values):
                failures.append("boundary/interior w diagnostics are non-finite")
            elif values[0] < max(values[1], values[2]):
                failures.append(
                    "domain w maximum is below a boundary/interior maximum")
            if (self.boundary_diagnostics
                    and bool(getattr(summary, "boundary_zone_blowup", True))):
                failures.append("boundary-zone blowup diagnostic fired")

        for threshold in self.all_thresholds:
            if threshold.name not in metrics and hasattr(summary, threshold.name):
                metrics[threshold.name] = getattr(summary, threshold.name)
            value = metrics.get(threshold.name)
            if not threshold.accepts(value):
                failures.append(
                    f"metric {threshold.name}={value!r} is outside "
                    f"{threshold.interval!r}")

        if self.output_schema is not None:
            context = (start_time, expected_completed_seconds,
                       history_interval_s, nx, ny, nz)
            if any(value is None for value in context):
                failures.append("output-schema context was not supplied")
            else:
                failures.extend(self.output_schema.failures(
                    getattr(summary, "wrfout_paths", ()),
                    start_time=start_time,
                    run_seconds=expected_completed_seconds,
                    history_interval_s=history_interval_s,
                    domain_id=domain_id, nx=nx, ny=ny, nz=nz))

        return HealthReport(metrics=metrics, failures=tuple(failures))


@dataclass(frozen=True)
class AnalysisRecipe:
    """One case-analysis comparison recipe.

    Its data fields are the plan's tuple, in order: valid time, field,
    level, mask, metric, threshold.
    """

    valid_time: datetime
    field: str
    level: float | None
    mask: str
    metric: str
    threshold: Threshold

    def __post_init__(self) -> None:
        if self.valid_time.tzinfo is not None:
            raise ValueError("analysis valid_time must be naive UTC")
        if not self.field or not self.mask or not self.metric:
            raise ValueError("analysis field, mask, and metric must be non-empty")
        if self.level is not None and not np.isfinite(self.level):
            raise ValueError("analysis level must be finite or None")


@dataclass(frozen=True)
class AnalysisProfile:
    """Comparisons with a case's own initial/final analyses."""

    name: str
    recipes: tuple[AnalysisRecipe, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.recipes:
            raise ValueError("analysis profile needs a name and recipes")
        names = [recipe.threshold.name for recipe in self.recipes]
        if len(set(names)) != len(names):
            raise ValueError("analysis threshold names must be unique")

    def gates(self) -> dict[str, tuple[float | None, float | None]]:
        return {recipe.threshold.name: recipe.threshold.interval
                for recipe in self.recipes}


@dataclass(frozen=True)
class OracleProfile:
    """Optional external-oracle paths, thresholds, and named masks."""

    name: str
    reference_paths: tuple[tuple[datetime, Path], ...] = ()
    thresholds: tuple[Threshold, ...] = ()
    masks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("oracle profile name must be non-empty")
        normalized = tuple((valid_time, Path(path))
                           for valid_time, path in self.reference_paths)
        if any(valid_time.tzinfo is not None
               for valid_time, _path in normalized):
            raise ValueError("oracle valid times must be naive UTC")
        if len({valid_time for valid_time, _path in normalized}) != len(normalized):
            raise ValueError("oracle reference valid times must be unique")
        if len({threshold.name for threshold in self.thresholds}) != len(
                self.thresholds):
            raise ValueError("oracle threshold names must be unique")
        if len(set(self.masks)) != len(self.masks) or any(
                not mask for mask in self.masks):
            raise ValueError("oracle masks must be non-empty and unique")
        object.__setattr__(self, "reference_paths", normalized)

    def gates(self) -> dict[str, tuple[float | None, float | None]]:
        return {threshold.name: threshold.interval
                for threshold in self.thresholds}


@dataclass(frozen=True)
class VerificationProfile:
    """One frozen composition of health, analysis, and optional oracle."""

    name: str
    health: HealthProfile
    analysis: AnalysisProfile | None = None
    oracle: OracleProfile | None = None
    gate_order: tuple[str, ...] = ()

    def gates(self) -> dict[str, tuple[float | None, float | None]]:
        tables = [self.health.gates()]
        if self.analysis is not None:
            tables.append(self.analysis.gates())
        if self.oracle is not None:
            tables.append(self.oracle.gates())
        merged = {name: interval for table in tables
                  for name, interval in table.items()}
        if len(merged) != sum(len(table) for table in tables):
            raise ValueError("profile threshold names overlap")
        order = self.gate_order or tuple(merged)
        if set(order) != set(merged) or len(order) != len(merged):
            raise ValueError("gate_order must name every threshold exactly once")
        return {name: merged[name] for name in order}


__all__ = [
    "AnalysisProfile", "AnalysisRecipe", "HealthProfile", "HealthReport",
    "OracleProfile", "OutputSchema", "Threshold", "VerificationProfile",
]
