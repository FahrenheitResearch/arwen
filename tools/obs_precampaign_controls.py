#!/usr/bin/env python3
"""Run the observation-side qualification controls, on real archive bytes.

Four of the battery's seven registered controls (spec section 9.3) need no
forecast at all -- their whole content is what the observations and the
scoring machinery do to each other -- and running them before the venues
free is the difference between discovering an instrument bug during the
shakedown and discovering it during the campaign:

``persistence reference``
    the zero-skill floor itself: the observed field at init, remapped to the
    forecast grid and held, scored against the observation at every scored
    lead by the registered FSS.  This IS the ``persistence`` row of spec
    section 3.4 and the input to the persistence-floor control; the control's
    verdict clause needs model arms and is therefore B6's, but the floor it
    compares against is measurable now, once, from the archive.

``wrong-day negative``
    the same persistence forecast scored against ANOTHER registered day's
    observations, and -- sharper -- a *perfect* forecast of one day scored
    against the other.  Same box, same masks, same climatology, same code
    path: only the weather changes.  An instrument reading masks rather than
    weather cannot tell those apart, and this is where it says so.

``regrid sensitivity``
    the persistence reference scored again under the alternate registered
    remap operator.  The delta is the number the registered choice is
    qualified against.

``station-shuffle machinery``
    the derangement the mutation control applies, built on the real frozen
    station table, with the displacement it produces and the surface RMSE a
    *perfect* arm would suffer under it -- which is the mutation's strength,
    measured, rather than assumed.

What is deliberately NOT here: any verdict that needs a model arm or a twin
band.  ``persistence-floor``, ``station-shuffle-mutation`` and
``wrong-day-negative`` are all registered with clauses phrased in the twin
band, and the twin band does not exist until the WRF twin pair runs.  This
tool measures and publishes the observation-side quantities those clauses
consume; it does not evaluate the clauses, and it does not stand in for
:mod:`gpuwm.verify.obs.controls`, which does.

The tool knows no case: every case-shaped thing -- the grid, the days, the
init instants, the station table -- is an argument.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gpuwm.obs.sources import AsosSurfaceSource, MrmsCompositeSource
from gpuwm.verify.chaos_envelope import canonical_hash
from gpuwm.verify.obs import battery, controls, registration, regrid, stations
from gpuwm.verify.obs.contracts import (SCORED_SURFACE_VARIABLES, ModelGrid,
                                        format_valid_time, parse_valid_time)

RECEIPT_SCHEMA = "gpuwm.obs-battery-precampaign-controls/v1"


class _ReplayArm:
    """An arm that replays a stored field per valid time.

    Used only by the wrong-day control, to score a *perfect* forecast of one
    day against another day's observations.  It satisfies the same reading
    protocol the scoring pass uses for a real run, so the number it produces
    comes out of the identical code path.
    """

    def __init__(self, grid: ModelGrid, by_valid_time: dict[str, np.ndarray]):
        self._grid = grid
        self._fields = dict(by_valid_time)

    def grid(self) -> ModelGrid:
        return self._grid

    def composite_reflectivity(self, valid_time: str) -> np.ndarray:
        try:
            return self._fields[str(valid_time)].copy()
        except KeyError:
            raise LookupError(
                f"the replay arm holds no field for {valid_time}") from None


def _packs(directory: Path) -> tuple[list[Path], Path]:
    """Every gridded pack in a driver's output directory, and its geometry."""
    packs = sorted((directory / "packs").glob("*.obspack"))
    geometry = directory / "packs" / "geometry.obspack"
    frames = [path for path in packs if path != geometry]
    if not frames:
        raise FileNotFoundError(f"{directory} holds no observation packs")
    if not geometry.is_file():
        raise FileNotFoundError(f"{geometry} is missing")
    return frames, geometry


def _model_grid(path: Path) -> ModelGrid:
    with np.load(path) as handle:
        latitude = np.asarray(handle["latitude"], dtype=np.float64)
        longitude = np.asarray(handle["longitude"], dtype=np.float64)
        dx_m = float(handle["dx_m"])
    return ModelGrid(latitude=latitude, longitude=longitude, dx_m=dx_m)


def _score(parameters, *, model, source, init_time, leads, grid, region):
    """One scoring pass through the shipped scorer, with its provenance."""
    collected: list = []
    scores = battery.score_reflectivity(
        registration={"parameters": {"reflectivity": parameters}},
        model=model, obs_source=source, init_time=init_time,
        lead_hours=leads, grid=grid, scored_region=region,
        collected_provenance=collected)
    return scores, collected


def _persistence(source, *, init_time, grid, parameters):
    field = source.field(init_time)
    arm = controls.PersistenceForecast(
        field, grid=grid, method=str(parameters["regrid_method"]),
        max_distance_m=float(parameters["regrid_max_distance_m"]))
    return arm, field


def _nearest_cell_positions(grid: ModelGrid, station_rows, *,
                            max_distance_m: float):
    """Station -> nearest model cell, as integer grid indices.

    Integer rather than fractional on purpose: this tool never samples a
    model field, so it needs a *location*, not an interpolation weight, and
    a second implementation of a map projection inverse is a second set of
    answers.  The neighbour search is the remap module's own.
    """
    latitudes = np.asarray([row["latitude"] for row in station_rows],
                           dtype=np.float64)
    longitudes = np.asarray([row["longitude"] for row in station_rows],
                            dtype=np.float64)
    plan = regrid.build_plan(
        source_latitude=grid.latitude, source_longitude=grid.longitude,
        destination_latitude=latitudes.reshape(1, -1),
        destination_longitude=longitudes.reshape(1, -1),
        method=regrid.NEAREST, max_distance_m=float(max_distance_m))
    ny, nx = grid.shape
    flat = plan.source_index.ravel()
    reachable = plan.reachable.ravel()
    positions: dict[str, stations.StationPosition] = {}
    outside: list[str] = []
    for index, row in enumerate(station_rows):
        if not bool(reachable[index]):
            outside.append(str(row["station_id"]))
            continue
        cell = int(flat[index])
        positions[str(row["station_id"])] = stations.StationPosition(
            station_id=str(row["station_id"]),
            x=float(cell % nx), y=float(cell // nx))
    return positions, outside


def _great_circle_m(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
    return 2.0 * regrid.EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def station_shuffle_machinery(*, grid: ModelGrid, region: np.ndarray,
                              station_table: Path, surface_record: Path,
                              valid_times: tuple[str, ...], seed: int,
                              match_tolerance_seconds: int
                              ) -> dict[str, object]:
    """Build the derangement on the real table and price what it destroys."""
    table = json.loads(Path(station_table).read_text(encoding="utf-8"))
    rows = list(table["stations"])
    positions, outside = _nearest_cell_positions(
        grid, rows, max_distance_m=float(grid.dx_m))
    ny, nx = grid.shape
    interior = {station_id: position for station_id, position in positions.items()
                if bool(region[int(position.y), int(position.x)])}

    # A derangement is only invertible when no two stations share a cell, and
    # the inverse is what says WHICH station's observation a shuffled station
    # would be read against.  Stations sharing a cell are dropped from the
    # control and counted rather than guessed at.
    by_cell: dict[tuple[int, int], list[str]] = {}
    for station_id, position in interior.items():
        by_cell.setdefault((int(position.y), int(position.x)), []).append(station_id)
    shared = sorted(s for group in by_cell.values() if len(group) > 1
                    for s in group)
    unique = {station_id: position for station_id, position in interior.items()
              if station_id not in set(shared)}
    if len(unique) < 2:
        raise ValueError("the shuffle control needs at least two stations")

    frozen = stations.FrozenStationSet(
        station_ids=tuple(sorted(unique)), positions=unique, drops=(),
        parameters={"note": "positions are nearest model cells; this control "
                            "never samples a model field"})
    shuffled = controls.build_shuffled_stations(frozen, seed=int(seed))

    cell_owner = {(int(p.y), int(p.x)): s for s, p in unique.items()}
    mapping: dict[str, str] = {}
    fixed_points: list[str] = []
    for station_id, position in shuffled.positions.items():
        donor = cell_owner[(int(position.y), int(position.x))]
        mapping[station_id] = donor
        if donor == station_id:
            fixed_points.append(station_id)

    coordinates = {str(row["station_id"]): (float(row["latitude"]),
                                            float(row["longitude"]))
                   for row in rows}
    displacement = []
    for station_id, donor in mapping.items():
        lat1, lon1 = coordinates[station_id]
        lat2, lon2 = coordinates[donor]
        displacement.append(_great_circle_m(lat1, lon1, lat2, lon2))
    displacement_array = np.asarray(displacement, dtype=np.float64)

    observations = AsosSurfaceSource(surface_record).observations(valid_times)
    matched = stations.match_reports(
        observations, valid_times,
        tolerance_seconds=int(match_tolerance_seconds))

    # A PERFECT arm reads its own station's value exactly, so its baseline
    # RMSE is zero by construction and the shuffled RMSE below is the whole
    # increase the mutation produces.  Stated rather than computed, because
    # computing zero from a synthetic field would only measure the synthesis.
    rows_out = []
    for variable in SCORED_SURFACE_VARIABLES:
        residuals: list[float] = []
        per_station: dict[str, list[float]] = {}
        for station_id, donor in mapping.items():
            for text in valid_times:
                here = matched.get((station_id, text))
                there = matched.get((donor, text))
                if here is None or there is None:
                    continue
                if variable in stations.screen_report(here):
                    continue
                if variable in stations.screen_report(there):
                    continue
                if variable not in here.values or variable not in there.values:
                    continue
                residual = float(there.values[variable]) - float(here.values[variable])
                residuals.append(residual)
                per_station.setdefault(station_id, []).append(residual)
        if not residuals:
            raise ValueError(f"the shuffle control matched no {variable} pairs")
        array = np.asarray(residuals, dtype=np.float64)
        station_rmse = sorted(
            float(np.sqrt(np.mean(np.asarray(values) ** 2, dtype=np.float64)))
            for values in per_station.values())
        rows_out.append({
            "variable": variable,
            "perfect_arm_baseline_rmse": 0.0,
            "shuffled_rmse": float(np.sqrt(np.mean(array * array,
                                                   dtype=np.float64))),
            "shuffled_median_station_rmse": float(
                np.median(np.asarray(station_rmse, dtype=np.float64))),
            "shuffled_bias": float(np.mean(array, dtype=np.float64)),
            "sample_count": int(array.size),
            "station_count": len(per_station),
        })

    return {
        "control": "station-shuffle-machinery",
        "status": ("machinery-verified" if not fixed_points
                   else "derangement-has-a-fixed-point"),
        "seed": int(seed),
        "station_table": str(station_table),
        "stations_in_table": len(rows),
        "stations_outside_grid": len(outside),
        "stations_outside_interior": len(positions) - len(interior),
        "stations_sharing_a_cell": len(shared),
        "stations_shuffled": len(mapping),
        "fixed_points": fixed_points,
        "displacement_m": {
            "minimum": float(displacement_array.min()),
            "median": float(np.median(displacement_array)),
            "mean": float(displacement_array.mean()),
            "maximum": float(displacement_array.max()),
        },
        "variables": rows_out,
        "note": ("the registered station-shuffle-mutation control compares a "
                 "real arm's RMSE with and without the derangement and asks "
                 "whether the rise exceeds the twin band; neither exists yet, "
                 "so what is measured here is the derangement itself and the "
                 "rise a perfect arm would suffer under it"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-grid", required=True, type=Path,
                        help="npz carrying latitude, longitude and dx_m")
    parser.add_argument("--boundary-width-cells", type=int, required=True,
                        help="specified + relaxation rows the run is configured with")
    parser.add_argument("--packs", required=True, type=Path,
                        help="the case's obs_fetch_mrms.py output directory")
    parser.add_argument("--init-time", required=True,
                        help="seam instant, e.g. 2024-05-21T12:00:00")
    parser.add_argument("--wrong-day-packs", required=True, type=Path)
    parser.add_argument("--wrong-day-init-time", required=True)
    parser.add_argument("--stations", required=True, type=Path)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--shuffle-seed", type=int, default=20260804)
    parser.add_argument("--case-id", required=True,
                        help="the case this measures, carried as data")
    parser.add_argument("--wrong-day-case-id", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    arguments = parser.parse_args()

    grid = _model_grid(arguments.model_grid)
    region = battery.interior_mask(
        grid.shape, boundary_width_cells=int(arguments.boundary_width_cells),
        rim_m=registration.DEFAULT_INTERIOR_RIM_M, dx_m=grid.dx_m)
    leads = list(registration.scored_lead_hours())

    frames, geometry = _packs(arguments.packs)
    wrong_frames, wrong_geometry = _packs(arguments.wrong_day_packs)
    source = MrmsCompositeSource(frames, geometry)
    wrong_source = MrmsCompositeSource(wrong_frames, wrong_geometry)

    registered = registration.reflectivity_parameters()
    alternate = registration.reflectivity_parameters(
        regrid_method=regrid.CELL_AVERAGE)

    # 1. the persistence reference, under the registered remap
    arm, init_field = _persistence(source, init_time=arguments.init_time,
                                   grid=grid, parameters=registered)
    persistence, provenance = _score(
        registered, model=arm, source=source, init_time=arguments.init_time,
        leads=leads, grid=grid, region=region)

    # 2. the same forecast against another registered day
    wrong_day, wrong_provenance = _score(
        registered, model=arm, source=wrong_source,
        init_time=arguments.wrong_day_init_time, leads=leads, grid=grid,
        region=region)

    # 3. a PERFECT forecast of this case's day, scored against that other day
    plan = regrid.build_plan(
        source_latitude=init_field.latitude,
        source_longitude=init_field.longitude,
        destination_latitude=grid.latitude,
        destination_longitude=grid.longitude,
        method=str(registered["regrid_method"]),
        max_distance_m=float(registered["regrid_max_distance_m"]))
    replay: dict[str, np.ndarray] = {}
    same_day_replay: dict[str, np.ndarray] = {}
    for here, there in zip(battery.valid_times(arguments.init_time, leads),
                           battery.valid_times(arguments.wrong_day_init_time,
                                               leads)):
        truth = source.field(here)
        values, valid = regrid.apply_plan(plan, truth.values, truth.valid)
        field = np.where(valid, values, np.nan)
        replay[there] = field
        same_day_replay[here] = field
    perfect, _ = _score(
        registered, model=_ReplayArm(grid, replay), source=wrong_source,
        init_time=arguments.wrong_day_init_time, leads=leads, grid=grid,
        region=region)
    # The same perfect forecast against its OWN day.  It is 1.0 by
    # construction, and measuring it anyway is what turns the wrong-day
    # collapse into a measured pair rather than a number beside an assertion
    # -- and it is a free check that the scorer returns 1 for two identical
    # fields, which is the one answer a broken mask would not give.
    perfect_same_day, _ = _score(
        registered, model=_ReplayArm(grid, same_day_replay), source=source,
        init_time=arguments.init_time, leads=leads, grid=grid, region=region)

    # 4. the remap sensitivity, same reference under the other operator
    alternate_arm, _ = _persistence(source, init_time=arguments.init_time,
                                    grid=grid, parameters=alternate)
    alternate_scores, _ = _score(
        alternate, model=alternate_arm, source=source,
        init_time=arguments.init_time, leads=leads, grid=grid, region=region)

    sensitivity_delta = abs(float(persistence["primary_scalar"])
                            - float(alternate_scores["primary_scalar"]))

    # 5. the mutation machinery, on the real frozen table
    shuffle = station_shuffle_machinery(
        grid=grid, region=region, station_table=arguments.stations,
        surface_record=arguments.surface,
        valid_times=battery.valid_times(arguments.init_time, leads),
        seed=int(arguments.shuffle_seed),
        match_tolerance_seconds=registration.DEFAULT_MATCH_TOLERANCE_SECONDS)

    # 6. the integrity clause's observation half: re-hash every archive
    #    object the scored fields came from.
    rehash = []
    for entry in list(provenance) + list(wrong_provenance):
        rehash.append({"uri": entry.uri, "sha256": entry.sha256,
                       "matches": bool(source.verify(entry)),
                       "is_stub": bool(entry.is_stub)})
    rehash_clean = all(row["matches"] for row in rehash)
    stubs = sorted({row["uri"] for row in rehash if row["is_stub"]})

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "MEASURED",
        "evaluator_commit": str(arguments.evaluator_commit),
        "case_id": str(arguments.case_id),
        "wrong_day_case_id": str(arguments.wrong_day_case_id),
        "init_time": str(arguments.init_time),
        "wrong_day_init_time": str(arguments.wrong_day_init_time),
        "scored_lead_hours": leads,
        "grid": {"shape": list(grid.shape), "dx_m": grid.dx_m,
                 "interior_cells": int(np.count_nonzero(region)),
                 "boundary_width_cells": int(arguments.boundary_width_cells),
                 "rim_m": registration.DEFAULT_INTERIOR_RIM_M},
        "reflectivity_parameters": registered,
        "reflectivity_parameters_sha256": canonical_hash(registered),
        "persistence_reference": persistence,
        "wrong_day_persistence": wrong_day,
        "wrong_day_perfect_forecast": perfect,
        "same_day_perfect_forecast": perfect_same_day,
        "wrong_day_drops": {
            "persistence": {
                "same_day_primary": float(persistence["primary_scalar"]),
                "wrong_day_primary": float(wrong_day["primary_scalar"]),
                "drop": float(persistence["primary_scalar"]
                              - wrong_day["primary_scalar"]),
            },
            "perfect_forecast": {
                "same_day_primary": float(perfect_same_day["primary_scalar"]),
                "wrong_day_primary": float(perfect["primary_scalar"]),
                "drop": float(perfect_same_day["primary_scalar"]
                              - perfect["primary_scalar"]),
            },
            "fss_useful": float(persistence["primary_fss_useful"]),
            "criterion_note": (
                "the registered wrong-day control asks for three clauses: the "
                "score drops, the drop exceeds the twin band, and the "
                "wrong-day score falls below 0.5 + f_obs/2. Two are decidable "
                "from observations alone and are answered here; the twin-band "
                "clause holds for any band below the drop reported here"),
        },
        "regrid_sensitivity": {
            "registered_method": str(registered["regrid_method"]),
            "registered_primary": float(persistence["primary_scalar"]),
            "alternate_method": str(alternate["regrid_method"]),
            "alternate_primary": float(alternate_scores["primary_scalar"]),
            "delta": sensitivity_delta,
            "note": ("the registered choice stands unless this delta exceeds "
                     "the twin band, which does not exist until the twin pair "
                     "runs; the delta is published now so that comparison is "
                     "one subtraction later"),
        },
        "station_shuffle_machinery": shuffle,
        "observation_rehash": {
            "objects": len(rehash),
            "all_match": bool(rehash_clean),
            "stub_inputs": stubs,
            "rows": rehash,
        },
        "not_evaluated_here": [
            "persistence-floor (needs a model arm)",
            "station-shuffle-mutation (needs a model arm and a twin band)",
            "wrong-day-negative (its clauses are phrased in the twin band)",
            "reflectivity-operator-crosscheck (needs a wrfout)",
            "twin-non-degeneracy (needs the twin pair)",
            "determinism (needs the dual-run pair)",
        ],
    }

    print(f"persistence S_refl                 "
          f"{persistence['primary_scalar']:.4f}")
    print(f"  useful-skill line 0.5+f_obs/2    "
          f"{persistence['primary_fss_useful']:.4f}")
    print(f"  observed base rate               "
          f"{persistence['primary_observed_base_rate']:.4f}")
    print(f"  mean interior valid fraction     "
          f"{persistence['mean_interior_valid_fraction']:.4f}")
    print(f"wrong-day persistence S_refl       "
          f"{wrong_day['primary_scalar']:.4f}  "
          f"(drop {persistence['primary_scalar'] - wrong_day['primary_scalar']:+.4f})")
    print(f"same-day PERFECT forecast S_refl   "
          f"{perfect_same_day['primary_scalar']:.4f}")
    print(f"wrong-day PERFECT forecast S_refl  "
          f"{perfect['primary_scalar']:.4f}  "
          f"(drop {perfect_same_day['primary_scalar'] - perfect['primary_scalar']:+.4f})")
    print(f"regrid sensitivity delta           {sensitivity_delta:.4f} "
          f"({registered['regrid_method']} vs {alternate['regrid_method']})")
    print(f"shuffle: {shuffle['stations_shuffled']} stations, "
          f"{len(shuffle['fixed_points'])} fixed points, median displacement "
          f"{shuffle['displacement_m']['median'] / 1000.0:.1f} km")
    for row in shuffle["variables"]:
        print(f"  {row['variable']:<16} shuffled RMSE {row['shuffled_rmse']:.3f}")
    print(f"observation re-hash: {len(rehash)} objects, "
          f"all_match={rehash_clean}")

    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(json.dumps(receipt, indent=2,
                                                sort_keys=True) + "\n",
                                     encoding="utf-8")
        print(f"\nreceipt at {arguments.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
