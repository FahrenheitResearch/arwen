"""The scoring pass: one case, one arm, one small score file.

A campaign's history files are tens of gigabytes per run and do not survive
the campaign.  What survives is what this module writes: a megabyte-scale
document carrying every score, the registration hash it was computed under,
the evaluating commit, and the provenance of every observation that went into
it.  Score-then-delete only works if the score file is complete enough to
never need the wrfout again, so it is written that way.

The pass is arranged so that nothing about *which* case is being scored can
influence *how* it is scored:

* the registration supplies every parameter, and its hash travels with the
  output;
* the case supplies data -- an init instant, a model run, observation
  sources, a station set -- through the seam types;
* the same function scores the faithful arm, each patched arm, the other
  model, and the perturbed twin, because none of them is special here.

Spin-up leads are computed when asked for and filed under a key that says
they are reported and not scored.  They cannot reach the primary scalar: the
scored leads come from the registration, and the reported leads are a
separate argument that feeds a separate block.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from gpuwm.verify.obs import contingency, fss, regrid, stations as stations_mod
from gpuwm.verify.obs.contracts import (
    ModelGrid, ObsGridField, ObsProvenance, ObservedFractionBelowFloor,
    StationObsSet, format_valid_time,
)
from gpuwm.verify.obs.registration import validate_registration, write_json

SCORE_SCHEMA = "gpuwm.obs-battery-scores/v1"


def interior_mask(shape: tuple[int, int], *, boundary_width_cells: int,
                  rim_m: float, dx_m: float) -> np.ndarray:
    """The cells a case scores: not boundary, not relaxation, not rim.

    The excluded width is the specified plus relaxation rows the run was
    configured with, plus the registered physical rim converted at this
    domain's spacing.  Everything outside is where the lateral boundary is
    still telling the interior what to do, and scoring it would score the
    driving analysis.
    """
    ny, nx = int(shape[0]), int(shape[1])
    if boundary_width_cells < 0 or rim_m < 0.0 or dx_m <= 0.0:
        raise ValueError("interior mask parameters must be non-negative")
    rim_cells = int(math.ceil(float(rim_m) / float(dx_m)))
    width = int(boundary_width_cells) + rim_cells
    if ny <= 2 * width or nx <= 2 * width:
        raise ValueError(
            f"a {ny}x{nx} domain has no interior once {width} cells are "
            f"excluded on every side")
    mask = np.zeros((ny, nx), dtype=bool)
    mask[width:ny - width, width:nx - width] = True
    return mask


def valid_times(init_time: str, lead_hours: Sequence[int]) -> tuple[str, ...]:
    """Seam timestamps for a set of forecast leads."""
    start = datetime.fromisoformat(str(init_time))
    return tuple(format_valid_time(start + timedelta(hours=int(hour)))
                 for hour in lead_hours)


def _same_grid(left: ObsGridField, right: ObsGridField) -> bool:
    return (left.shape == right.shape
            and np.array_equal(left.latitude, right.latitude)
            and np.array_equal(left.longitude, right.longitude))


def _finite_mask(field: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(field, dtype=np.float64))


def score_reflectivity(*, registration: Mapping[str, object],
                       model, obs_source, init_time: str,
                       lead_hours: Sequence[int],
                       grid: ModelGrid,
                       scored_region: np.ndarray,
                       collected_provenance: list[ObsProvenance],
                       ) -> dict[str, object]:
    """Neighborhood FSS on composite reflectivity over a set of leads.

    Under the registered coverage floor (amendment v2.1) a lead whose whole
    matching window is below the floor carries no observation this rule will
    score.  It is recorded as excluded -- lead, valid time, and every
    candidate frame's observed fraction -- and left out of the lead mean.
    That exclusion is never silent and never inferred: it is a row in the
    score record, and a pass in which every lead is excluded still fails.
    """
    parameters = registration["parameters"]["reflectivity"]
    thresholds = [float(value) for value in parameters["thresholds_dbz"]]
    half_widths = [int(value) for value in
                   parameters["neighborhood_half_widths_cells"]]
    primary_threshold = float(parameters["primary_threshold_dbz"])
    primary_half_width = int(parameters["primary_half_width_cells"])

    plan: regrid.RegridPlan | None = None
    reference: ObsGridField | None = None
    by_lead: list[dict[str, object]] = []
    scored_hours: list[int] = []
    excluded: list[dict[str, object]] = []
    primary_series: list[fss.FssResult] = []
    interior_valid_fraction: list[float] = []

    for hour, valid_time in zip(lead_hours,
                                valid_times(init_time, lead_hours)):
        try:
            observed = obs_source.field(valid_time)
        except ObservedFractionBelowFloor as outage:
            excluded.append({
                "lead_hours": int(hour),
                "valid_time": valid_time,
                "reason": "observed_fraction_below_floor",
                "minimum_observed_fraction": (
                    outage.minimum_observed_fraction),
                "candidate_frames": outage.candidates,
                "note": str(outage),
            })
            continue
        if observed.quantity != parameters["observed_quantity"]:
            raise ValueError(
                f"the reflectivity source serves {observed.quantity!r}, the "
                f"registration scores {parameters['observed_quantity']!r}")
        collected_provenance.append(observed.provenance)
        if reference is None:
            reference = observed
            plan = regrid.build_plan(
                source_latitude=observed.latitude,
                source_longitude=observed.longitude,
                destination_latitude=grid.latitude,
                destination_longitude=grid.longitude,
                method=str(parameters["regrid_method"]),
                max_distance_m=float(parameters["regrid_max_distance_m"]))
        elif not _same_grid(reference, observed):
            raise ValueError(
                "the observation grid changed inside one case; one remap plan "
                "must serve every lead or the arms are not comparable")

        obs_values, obs_valid = regrid.apply_plan(
            plan, observed.values, observed.valid)
        forecast = np.asarray(model.composite_reflectivity(valid_time),
                              dtype=np.float64)
        if forecast.shape != grid.shape:
            raise ValueError("the model reflectivity does not match the grid")
        valid = fss.shared_validity(obs_valid, _finite_mask(forecast))

        matrix = fss.fss_matrix(
            forecast, obs_values, valid=valid, thresholds=thresholds,
            half_widths=half_widths, score_mask=scored_region,
            boundary=str(parameters["neighborhood_boundary"]))
        primary = matrix[(primary_threshold, primary_half_width)]
        primary_series.append(primary)
        matched = fss.masked_fss(
            forecast, obs_values, valid=valid, threshold=primary_threshold,
            half_width=primary_half_width, score_mask=scored_region,
            frequency_matched=True,
            boundary=str(parameters["neighborhood_boundary"]))
        tables = [
            contingency.score_field(obs_values, forecast, threshold=threshold,
                                    valid=valid & scored_region)
            for threshold in thresholds]
        # How much of the scored interior this lead could actually see.  A
        # case whose interior is largely unobserved is a case whose entry
        # receipt has to answer for it, so the number is published per lead
        # rather than left to be inferred from a scored-cell count.
        interior_cells = int(np.count_nonzero(scored_region))
        valid_fraction = (
            float(np.count_nonzero(valid & scored_region) / interior_cells)
            if interior_cells else 0.0)
        interior_valid_fraction.append(valid_fraction)
        scored_hours.append(int(hour))
        by_lead.append({
            "lead_hours": int(hour),
            "valid_time": valid_time,
            "observation_sha256": observed.provenance.sha256,
            "observation_is_stub": bool(observed.provenance.is_stub),
            "interior_valid_fraction": valid_fraction,
            "fss": fss.matrix_records(matrix, dx_m=grid.dx_m),
            "frequency_matched_primary": matched.record(),
            "contingency": tables,
            "regrid": plan.record(),
        })

    if not primary_series:
        raise ValueError(
            "reflectivity scoring produced no scored leads"
            + (f"; every lead was excluded for observed coverage below the "
               f"registered floor ({len(excluded)} of {len(lead_hours)}), "
               f"which is an unscoreable case and not a score of zero"
               if excluded else ""))
    return {
        "primary_threshold_dbz": primary_threshold,
        "primary_half_width_cells": primary_half_width,
        "primary_box_length_m": fss.box_length_m(primary_half_width,
                                                 grid.dx_m),
        "primary_scalar": fss.mean_fss(primary_series),
        # The useful-skill line for the primary cell, mean over the scored
        # leads.  Published because a bare FSS is unreadable without it, and
        # consumed by the wrong-day control, which needs a threshold nobody
        # invented.
        "primary_fss_useful": float(np.mean(
            [result.fss_useful for result in primary_series],
            dtype=np.float64)),
        "primary_observed_base_rate": float(np.mean(
            [result.observed_base_rate for result in primary_series],
            dtype=np.float64)),
        "primary_by_lead": {
            str(int(hour)): result.fss
            for hour, result in zip(scored_hours, primary_series)},
        "mean_interior_valid_fraction": float(
            np.mean(interior_valid_fraction, dtype=np.float64)),
        "minimum_interior_valid_fraction": float(
            min(interior_valid_fraction)),
        "leads": by_lead,
        # The lead mean above is over the leads that carried an observation.
        # Which leads those were, and which were excluded and why, are both
        # recorded: a mean over a changed denominator that does not say so is
        # the one number a reader cannot check.
        "lead_hours_scored": list(scored_hours),
        "lead_hours_requested": [int(hour) for hour in lead_hours],
        "excluded_leads": excluded,
    }


def score_surface(*, registration: Mapping[str, object], model,
                  observations: StationObsSet,
                  frozen: stations_mod.FrozenStationSet,
                  init_time: str, lead_hours: Sequence[int],
                  collected_provenance: list[ObsProvenance],
                  interpolation: str | None = None) -> dict[str, object]:
    """Bias and RMSE against the frozen station set, plus the guardrails."""
    parameters = registration["parameters"]["surface"]
    method = str(interpolation or parameters["interpolation"])
    hours = list(valid_times(init_time, lead_hours))
    collected_provenance.append(observations.provenance)
    matched = stations_mod.match_reports(
        observations, hours,
        tolerance_seconds=int(parameters["match_tolerance_seconds"]))

    cache: dict[tuple[str, str], np.ndarray] = {}

    def model_value(station_id: str, valid_time: str, variable: str):
        key = (valid_time, variable)
        if key not in cache:
            cache[key] = np.asarray(
                model.surface_field(valid_time, variable), dtype=np.float64)
        position = frozen.positions.get(station_id)
        if position is None:
            return None
        return stations_mod.sample_field(cache[key], position, method=method)

    scores = stations_mod.surface_scores(
        frozen, matched=matched, model_value=model_value, valid_times=hours,
        variables=[str(name) for name in parameters["variables"]])
    return {
        "interpolation": method,
        "station_set": frozen.record(),
        "matched_pairs": len(matched),
        "variables": {name: score.record()
                      for name, score in sorted(scores.items())},
        "guardrails": {
            f"median_station_rmse:{name}": float(score.median_station_rmse)
            for name, score in sorted(scores.items())},
    }


def score_precipitation(*, registration: Mapping[str, object], model,
                        obs_sources: Mapping[int, object], init_time: str,
                        lead_hours: Sequence[int], grid: ModelGrid,
                        model_scored_region: np.ndarray,
                        collected_provenance: list[ObsProvenance],
                        ) -> dict[str, object]:
    """Accumulation FSS and contingency scores on the analysis grid.

    The comparison happens on the observation analysis grid, not the model's:
    an analysis is the referee, and moving the referee onto the contestant's
    grid would smooth exactly the field being judged.

    **A window is scored only at the leads where it closes.**  A 6 h
    accumulation product holds a frame every 6 h, and asking it at an hourly
    lead asks for a frame that cannot exist: on the first full scoring pass
    the 6 h window was asked at all seventeen scored leads and the reader
    refused ten of them, correctly, on frames a 6-hourly product never had.
    The test is declarative -- ``hour % window`` -- and not "skip whatever is
    missing": a frame absent at an *aligned* lead is a hole in the archive,
    and it still refuses loudly.

    Alignment is measured from the init instant, which is where the model's
    own accumulation starts.  That coincides with the analysis product's
    synoptic cadence because the registered inits are synoptic; a case
    registered off that cadence would have to register the offset with it,
    and nothing here can infer it from the leads alone.
    """
    parameters = registration["parameters"]["precipitation"]
    thresholds = [float(value) for value in parameters["thresholds_mm"]]
    half_widths = [int(value) for value in
                   parameters["neighborhood_half_widths_cells"]]
    windows = [int(value) for value in parameters["window_hours"]]
    guard_threshold = float(parameters["guardrail_threshold_mm"])
    guard_half_width = int(parameters["guardrail_half_width_cells"])
    guard_window = int(parameters["guardrail_window_hours"])

    scored: list[dict[str, object]] = []
    guardrail_series: list[float] = []
    for window in windows:
        if window <= 0:
            raise ValueError(
                f"a registered accumulation window is a positive number of "
                f"hours, not {window}")
        source = obs_sources.get(window)
        if source is None:
            raise ValueError(
                f"the registration scores a {window} h accumulation and no "
                f"source was supplied for it")
        plan: regrid.RegridPlan | None = None
        reference: ObsGridField | None = None
        region: np.ndarray | None = None
        for hour in lead_hours:
            # The window has to fit inside the forecast AND close on this
            # lead; see this function's docstring for why the second test is
            # declarative rather than a skip on absence.
            if hour - window < 0 or hour % window:
                continue
            valid_time = valid_times(init_time, [hour])[0]
            start_time = valid_times(init_time, [hour - window])[0]
            observed = source.field(valid_time)
            collected_provenance.append(observed.provenance)
            if reference is None:
                reference = observed
                plan = regrid.build_plan(
                    source_latitude=grid.latitude,
                    source_longitude=grid.longitude,
                    destination_latitude=observed.latitude,
                    destination_longitude=observed.longitude,
                    method=str(parameters["regrid_method"]),
                    max_distance_m=float(parameters["regrid_max_distance_m"]))
                region_values, region_valid = regrid.apply_plan(
                    plan, model_scored_region.astype(np.float64),
                    np.ones(grid.shape, dtype=bool))
                region = region_valid & (region_values > 0.5)
            elif not _same_grid(reference, observed):
                raise ValueError(
                    "the precipitation analysis grid changed inside one case")

            accumulation = (
                np.asarray(model.precipitation_accumulation(valid_time),
                           dtype=np.float64)
                - np.asarray(model.precipitation_accumulation(start_time),
                             dtype=np.float64))
            if np.any(accumulation < -1.0e-6):
                raise ValueError(
                    f"model accumulation over {start_time}..{valid_time} is "
                    f"negative; the frames are out of order or the field is "
                    f"not a run total")
            accumulation = np.clip(accumulation, 0.0, None)
            forecast, forecast_valid = regrid.apply_plan(
                plan, accumulation, np.ones(grid.shape, dtype=bool))
            valid = fss.shared_validity(observed.valid, forecast_valid)

            matrix = fss.fss_matrix(
                forecast, observed.values, valid=valid, thresholds=thresholds,
                half_widths=half_widths, score_mask=region,
                boundary=str(parameters["neighborhood_boundary"]))
            tables = [
                contingency.score_field(observed.values, forecast,
                                        threshold=threshold,
                                        valid=valid & region)
                for threshold in thresholds]
            row = {
                "window_hours": window,
                "lead_hours": int(hour),
                "valid_time": valid_time,
                "observation_sha256": observed.provenance.sha256,
                "observation_is_stub": bool(observed.provenance.is_stub),
                "fss": fss.matrix_records(matrix, dx_m=grid.dx_m),
                "contingency": tables,
                "regrid": plan.record(),
            }
            scored.append(row)
            if window == guard_window:
                guardrail_series.append(
                    matrix[(guard_threshold, guard_half_width)].fss)

    if not scored:
        raise ValueError("precipitation scoring produced no accumulations")
    if not guardrail_series:
        raise ValueError(
            "no accumulation window produced the registered guardrail score")
    return {
        "windows": scored,
        "guardrails": {
            "mean_fss:precipitation": float(
                np.mean(guardrail_series, dtype=np.float64))},
        "guardrail_definition": str(parameters["guardrail_statistic"]),
    }


def score_case_arm(*, registration: Mapping[str, object], case_id: str,
                   arm_id: str, init_time: str, model,
                   reflectivity_obs, grid: ModelGrid | None = None,
                   boundary_width_cells: int,
                   station_obs: StationObsSet | None = None,
                   frozen_stations: stations_mod.FrozenStationSet | None = None,
                   precipitation_obs: Mapping[int, object] | None = None,
                   reported_lead_hours: Sequence[int] = (),
                   rehash: Callable[[ObsProvenance], bool] | None = None,
                   ) -> dict[str, object]:
    """Score one arm of one case and return the score file payload."""
    reg = validate_registration(registration)
    parameters = reg["parameters"]
    lead_hours = [int(hour) for hour in parameters["scored_lead_hours"]]
    reported = [int(hour) for hour in reported_lead_hours]
    overlap = sorted(set(reported) & set(lead_hours))
    if overlap:
        raise ValueError(
            f"leads {overlap} are both scored and reported-not-scored; a lead "
            f"is one or the other")
    model_grid = grid if grid is not None else model.grid()
    region = interior_mask(
        model_grid.shape, boundary_width_cells=boundary_width_cells,
        rim_m=float(parameters["reflectivity"]["interior_rim_m"]),
        dx_m=model_grid.dx_m)

    provenance: list[ObsProvenance] = []
    reflectivity = score_reflectivity(
        registration=reg, model=model, obs_source=reflectivity_obs,
        init_time=init_time, lead_hours=lead_hours, grid=model_grid,
        scored_region=region, collected_provenance=provenance)

    reported_block: dict[str, object] | None = None
    if reported:
        reported_scores = score_reflectivity(
            registration=reg, model=model, obs_source=reflectivity_obs,
            init_time=init_time, lead_hours=reported, grid=model_grid,
            scored_region=region, collected_provenance=provenance)
        reported_block = {
            "note": ("reported, never scored: these leads do not enter the "
                     "primary scalar or any promotion input"),
            "primary_by_lead": reported_scores["primary_by_lead"],
        }

    surface = None
    if station_obs is not None:
        if frozen_stations is None:
            raise ValueError(
                "surface scoring needs the case's frozen station set; freezing "
                "per arm would unpair the arms")
        surface = score_surface(
            registration=reg, model=model, observations=station_obs,
            frozen=frozen_stations, init_time=init_time,
            lead_hours=lead_hours, collected_provenance=provenance)

    precipitation = None
    if precipitation_obs:
        precipitation = score_precipitation(
            registration=reg, model=model, obs_sources=precipitation_obs,
            init_time=init_time, lead_hours=lead_hours, grid=model_grid,
            model_scored_region=region, collected_provenance=provenance)

    guardrails: dict[str, float] = {}
    if surface is not None:
        guardrails.update(surface["guardrails"])
    if precipitation is not None:
        guardrails.update(precipitation["guardrails"])

    unique: dict[str, ObsProvenance] = {}
    for item in provenance:
        unique.setdefault(f"{item.uri}|{item.sha256}", item)
    rehash_records = []
    for item in unique.values():
        rehash_records.append({
            "uri": item.uri, "sha256": item.sha256,
            "matches": bool(rehash(item)) if rehash is not None else False,
            "rehash_performed": rehash is not None,
        })
    uses_stub = any(item.is_stub for item in unique.values())

    payload = {
        "schema": SCORE_SCHEMA,
        "case_id": str(case_id),
        "arm_id": str(arm_id),
        "init_time": str(init_time),
        "registration_sha256": reg["registration_sha256"],
        "evaluator_commit": reg["evaluator_commit"],
        "rule_status": reg["rule_status"],
        "scored_lead_hours": lead_hours,
        "interior": {
            "boundary_width_cells": int(boundary_width_cells),
            "rim_m": float(parameters["reflectivity"]["interior_rim_m"]),
            "scored_cells": int(np.count_nonzero(region)),
            "grid_cells": int(region.size),
        },
        "grid": {"shape": list(model_grid.shape), "dx_m": model_grid.dx_m},
        "reflectivity": reflectivity,
        "reported_not_scored": reported_block,
        "surface": surface,
        "precipitation": precipitation,
        "guardrails": guardrails,
        "primary_scalar": reflectivity["primary_scalar"],
        "observation_provenance": [item.record() for item in unique.values()],
        "observation_rehash": rehash_records,
        "uses_stub_inputs": bool(uses_stub),
    }
    if hasattr(model, "record"):
        payload["model_reader"] = model.record()
    return payload


def write_score_file(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Persist a score file; this is the artifact that outlives the wrfout."""
    return write_json(path, payload)


def collect_primary(score_files: Sequence[Mapping[str, object]], *,
                    arm_id: str) -> dict[str, float]:
    """``case_id -> primary scalar`` for one arm, across a set of score files."""
    collected: dict[str, float] = {}
    for payload in score_files:
        if str(payload["arm_id"]) != str(arm_id):
            continue
        case = str(payload["case_id"])
        if case in collected:
            raise ValueError(
                f"arm {arm_id} carries two score files for case {case}")
        collected[case] = float(payload["primary_scalar"])
    if not collected:
        raise ValueError(f"no score file carries arm {arm_id!r}")
    return collected


def collect_guardrail(score_files: Sequence[Mapping[str, object]], *,
                      arm_id: str, name: str) -> dict[str, float]:
    """``case_id -> guardrail value`` for one arm and one guardrail."""
    collected: dict[str, float] = {}
    for payload in score_files:
        if str(payload["arm_id"]) != str(arm_id):
            continue
        guardrails = payload.get("guardrails") or {}
        if name not in guardrails:
            raise ValueError(
                f"score file for case {payload['case_id']} arm {arm_id} does "
                f"not carry guardrail {name!r}")
        collected[str(payload["case_id"])] = float(guardrails[name])
    if not collected:
        raise ValueError(f"no score file carries arm {arm_id!r}")
    return collected


__all__ = [
    "SCORE_SCHEMA", "collect_guardrail", "collect_primary", "interior_mask",
    "score_case_arm", "score_precipitation", "score_reflectivity",
    "score_surface", "valid_times", "write_score_file",
]
