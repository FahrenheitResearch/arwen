"""EXPERIMENTAL: ``gpuwm-obs.asos-surface.v1`` -> LETKF observation batches.

The obs battery's ``rw_asos`` (BowEcho rustwx, ``tools/rustwx/crates/
rw-obs/src/bin/asos.rs``) owns the decode: it pulls routine + SPECI METARs
from the IEM archive against a frozen, hash-pinned station table, screens
them, converts to SI and writes the seam record this module reads.  Nothing
here parses a METAR, ever -- the decode authority stays in the owner's Rust
stack, and everything this adapter cannot get from the seam is an upstream
(rw-obs) schema extension, not a local workaround.

What the v1 seam can and cannot express, and what that means here:

**Wind is a speed, not a vector.**  IEM's ``drct`` (direction) is fetched by
``rw_asos`` and dropped at decode, so ``wind_speed_10m`` is the only wind
quantity on the seam and u10/v10 innovations cannot be built.  The wind
observation is therefore the SPEED, and its forward operator is
``hypot(u10, v10)`` of each member's own 10 m diagnostics.  The modulus is
rotation-invariant, so grid-relative model winds need no earth rotation --
the one surface operator that is exactly as defensible without the direction
as with it.  Assimilating components needs an ``asos-surface.v2`` carrying
``drct``: upstream work, the owner's call.

**Temperature is the model's own 2 m diagnostic.**  The surface layer
diagnoses ``t2`` every step (MYNN and legacy sfclay both), so H(x) is that
field at the station's gridpoint -- the identity, not a fabricated
similarity-theory inversion on this side of the seam.

**Pressure is deliberately absent.**  The seam's ``mslp`` is IEM's own
reduction and is missing wherever an AWOS reports altimeter only;
differencing it against any model-side reduction mixes two formulas.  The
clean pair (altimeter or station pressure vs ``psfc`` adjusted to station
elevation) needs the v2 schema too, so this adapter does not offer a
pressure type at all rather than offering a wrong one.

**Reports are hourly-matched.**  The decoder matches reports to whole-hour
valid times by design (``--step-hours`` in [1, 24]); a 5-15 min cycling run
therefore sees fresh surface observations at roughly one cycle per hour.
Each report is assimilated ONCE, at the analysis time nearest its valid
time (ties to the earlier analysis), and never once its age exceeds
``max_age_seconds``.  Sub-hourly cadence is the IEM ``asos1min`` dataset --
a new rw-obs route, upstream.

**Station elevation is checked against model terrain.**  A valley or ridge
station the grid does not resolve produces systematic 2 m innovations that
the filter will happily spread into the storm environment.  Stations whose
elevation differs from the model terrain at their gridpoint by more than
``elevation_max_diff_m`` are refused and counted, never silently kept; the
lapse-rate-adjustment alternative is a policy decision that belongs to the
owner, not to a default.

**Errors are standard deviations**, stated by the caller: the v1 record
carries no per-report error, and instrument precision is the wrong number
anyway at storm-scale grids -- WoFS-like representativeness values
(T2 ~ 1.5-2.5 K, wind ~ 1.5-2.5 m/s) are the defensible range.  There is
no default sigma: a number nobody chose is a claim nobody made.

Nothing in this module is wired into a default route.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gpuwm.da.letkf import GriddedObs, Localization

#: The observation contract this adapter reads.
OBS_SCHEMA = "gpuwm-obs.asos-surface.v1"

#: Provenance stamp for the adaptation itself.
ADAPTER_SCHEMA = "gpuwm-da.surface-obs-adapter.v1"

#: Seam quantity ids (the rw_asos ABI marker's own spellings) this adapter
#: can turn into batches, with the units the seam guarantees.
TEMPERATURE_QUANTITY = "temperature_2m"    # K
WIND_SPEED_QUANTITY = "wind_speed_10m"     # m s-1

#: Seam quantities deliberately NOT offered.  ``dewpoint_2m`` needs a q2
#: inversion choice nobody has reviewed; ``mslp`` mixes reduction formulas
#: (see the module docstring).  Both are upstream/v2 questions.
UNSUPPORTED_QUANTITIES = ("dewpoint_2m", "mslp")


class SurfaceObsError(ValueError):
    """The record and the filter cannot be reconciled.  Never a warning."""


@dataclass(frozen=True)
class SurfaceObsConfig:
    """Everything one surface adaptation needs that is not data.

    A quantity is enabled by stating its observation error standard
    deviation; ``None`` leaves it out.  At least one must be stated --
    an adaptation of nothing is a bug, not a configuration.

    temperature_error_k / wind_speed_error_ms
        Observation error STANDARD DEVIATIONS (not variances) in the
        seam's own units.  These should carry representativeness, not
        instrument precision: WoFS-like practice is T2 ~ 1.5-2.5 K and
        10 m wind ~ 1.5-2.5 m/s at storm-scale grids.
    error_inflation
        Multiplies both sigmas; >= 1, same contract as the radar per-type
        inflation knobs.
    elevation_max_diff_m
        Refuse a station whose table elevation differs from the model
        terrain at its gridpoint by more than this.  Conventional surface
        DA gates sit near 100-200 m.
    max_age_seconds
        Refuse a report whose seam valid time is farther than this from
        the analysis time.  Note the seam valid time is the decoder's
        hourly match (within its own ``match_seconds`` of the true obs
        time), so the true report age can differ by up to that much.
    temperature_localization / wind_localization
        Per-type overrides, same contract as the radar config; ``None``
        falls back to the filter's default radii.
    """

    temperature_error_k: float | None = None
    wind_speed_error_ms: float | None = None
    error_inflation: float = 1.0
    elevation_max_diff_m: float = 200.0
    max_age_seconds: float = 900.0
    temperature_localization: Localization | None = None
    wind_localization: Localization | None = None

    def __post_init__(self) -> None:
        if self.temperature_error_k is None and \
                self.wind_speed_error_ms is None:
            raise SurfaceObsError(
                "neither temperature_error_k nor wind_speed_error_ms is "
                "stated, so this config would assimilate nothing. A "
                "quantity is enabled by stating its error standard "
                "deviation; there is no default sigma on purpose")
        for label, value in (
                ("temperature_error_k", self.temperature_error_k),
                ("wind_speed_error_ms", self.wind_speed_error_ms)):
            if value is None:
                continue
            v = float(value)
            if not math.isfinite(v) or v <= 0.0:
                raise SurfaceObsError(
                    f"{label} must be finite and positive, got {value!r}. "
                    "It is a standard deviation; zero would claim an exact "
                    "instrument and a negative one claims nothing at all")
        inflation = float(self.error_inflation)
        if not math.isfinite(inflation) or inflation < 1.0:
            raise SurfaceObsError(
                f"error_inflation must be finite and >= 1, got "
                f"{self.error_inflation!r}; deflating a stated observation "
                "error is a claim of skill nobody measured")
        gate = float(self.elevation_max_diff_m)
        if not math.isfinite(gate) or gate <= 0.0:
            raise SurfaceObsError(
                f"elevation_max_diff_m must be finite and positive, got "
                f"{self.elevation_max_diff_m!r}")
        age = float(self.max_age_seconds)
        if not math.isfinite(age) or age <= 0.0:
            raise SurfaceObsError(
                f"max_age_seconds must be finite and positive, got "
                f"{self.max_age_seconds!r}")

    @property
    def temperature(self) -> bool:
        return self.temperature_error_k is not None

    @property
    def wind_speed(self) -> bool:
        return self.wind_speed_error_ms is not None


# ---------------------------------------------------------------------------
# reading and binding the record
# ---------------------------------------------------------------------------


def read_record(source) -> Mapping:
    """Accept a path or an already-read record; refuse anything else.

    Refusals here are the seam's own promises being broken: wrong schema,
    a status other than READY, or stub provenance.  A stub that reached an
    assimilation call has already defeated the loud-stub discipline once;
    it does not get a second chance here.
    """

    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    elif isinstance(source, Mapping):
        record = source
    else:
        raise SurfaceObsError(
            f"expected a path or a read asos-surface record, got "
            f"{type(source).__name__}")
    if record.get("schema") != OBS_SCHEMA:
        raise SurfaceObsError(
            f"record declares schema {record.get('schema')!r}, this "
            f"adapter reads {OBS_SCHEMA!r}")
    if record.get("status") != "READY":
        raise SurfaceObsError(
            f"record status is {record.get('status')!r}, not 'READY'; an "
            "unready record is the writer's own statement that it must "
            "not be consumed")
    provenance = record.get("provenance") or {}
    if provenance.get("is_stub"):
        raise SurfaceObsError(
            "record provenance says is_stub=true "
            f"(reason: {provenance.get('stub_reason')!r}); stub data is "
            "for wiring tests and never for an analysis")
    for key in ("stations", "reports", "station_table_sha256"):
        if key not in record:
            raise SurfaceObsError(f"record carries no {key!r}")
    return record


def _parse_seam_time(text: str, *, where: str) -> datetime:
    """Seam timestamps are UTC; naive ones by the writer's contract."""

    try:
        stamp = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError as error:
        raise SurfaceObsError(
            f"{where}: {text!r} is not an ISO-8601 time") from error
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _as_utc(stamp: datetime, *, label: str) -> datetime:
    if not isinstance(stamp, datetime):
        raise SurfaceObsError(
            f"{label} must be a datetime, got {type(stamp).__name__}")
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# report selection
# ---------------------------------------------------------------------------


def _select_reports(record: Mapping, analysis_time: datetime,
                    analysis_times: Sequence[datetime] | None,
                    max_age_s: float, counts: dict) -> dict[str, dict]:
    """One report per station for this analysis time, each used once.

    A report is eligible when its seam valid time is within
    ``max_age_s`` of ``analysis_time`` AND, when the caller supplies the
    full analysis schedule, ``analysis_time`` is the nearest analysis to
    the report (ties to the earlier analysis).  The schedule rule is what
    keeps an hourly report from being assimilated at two adjacent 15 min
    cycles: the same number entering the filter twice is a second,
    perfectly correlated observation nobody took.
    """

    if analysis_times is not None:
        schedule = sorted(_as_utc(t, label="analysis_times[]")
                          for t in analysis_times)
        if analysis_time not in schedule:
            raise SurfaceObsError(
                "analysis_time is not one of analysis_times; the "
                "once-per-report routing needs the schedule to contain "
                "the very analysis being run")
    else:
        schedule = None

    chosen: dict[str, dict] = {}
    for report in record["reports"]:
        flags = report.get("flags") or []
        if flags:
            counts["reports_refused_flagged"] += 1
            continue
        valid = _parse_seam_time(report["valid_time"],
                                 where=f"report {report.get('station_id')}")
        age = (analysis_time - valid).total_seconds()
        if abs(age) > max_age_s:
            counts["reports_outside_window"] += 1
            continue
        if schedule is not None:
            nearest = min(
                schedule,
                key=lambda t: (abs((t - valid).total_seconds()), t))
            if nearest != analysis_time:
                counts["reports_routed_to_other_analysis"] += 1
                continue
        station_id = str(report["station_id"])
        previous = chosen.get(station_id)
        if previous is not None:
            # Nearest report wins; ties to the one at or before the
            # analysis time (a nowcast prefers the past over the future).
            if (abs(age), valid > analysis_time) >= (
                    abs(previous["age_s"]), previous["valid"]
                    > analysis_time):
                counts["reports_superseded_same_station"] += 1
                continue
            counts["reports_superseded_same_station"] += 1
        chosen[station_id] = {"report": report, "valid": valid,
                              "age_s": age}
    return chosen


# ---------------------------------------------------------------------------
# the adaptation
# ---------------------------------------------------------------------------


def _member_stack(name: str, value, members: int | None,
                  plane_shape: tuple[int, int]) -> np.ndarray:
    stack = np.asarray(value, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[1:] != plane_shape:
        raise SurfaceObsError(
            f"{name} must be (R, {plane_shape[0]}, {plane_shape[1]}), "
            f"got {stack.shape}")
    if members is not None and stack.shape[0] != members:
        raise SurfaceObsError(
            f"{name} carries {stack.shape[0]} members where a sibling "
            f"diagnostic carries {members}; one ensemble, one R")
    if not np.all(np.isfinite(stack)):
        raise SurfaceObsError(
            f"{name} carries non-finite values; a surface diagnostic "
            "with NaNs in it is a model problem the filter must not "
            "paper over")
    return stack


def surface_to_gridded_obs(
    source,
    *,
    target_grid,
    analysis_time: datetime,
    config: SurfaceObsConfig,
    simulated_t2=None,
    simulated_u10=None,
    simulated_v10=None,
    analysis_times: Sequence[datetime] | None = None,
) -> tuple[list[GriddedObs], dict]:
    """Adapt one asos-surface record to the filter's observation batches.

    Parameters
    ----------
    source
        A ``gpuwm-obs.asos-surface.v1`` path or an already-read record.
    target_grid
        The caller's own :class:`~gpuwm.obs.target_grid.TargetGrid`.
        Placement is its ``mass_index`` and ``inside`` -- the same
        coverage authority the radar path binds to -- and the elevation
        gate reads its ``terrain_m``.
    analysis_time
        The analysis this adaptation serves, UTC (naive = UTC).
    config
        :class:`SurfaceObsConfig`; which quantities run is stated there.
    simulated_t2
        ``(R, ny, nx)`` member 2 m temperature diagnostics, K.  Required
        when temperature is enabled.  Snapshot these at LEG END from the
        live physics driver -- the leg-start diagnostics are re-initialised
        spin-up state in the prepared-cache cycle and must not be used.
    simulated_u10, simulated_v10
        ``(R, ny, nx)`` member 10 m wind diagnostics, m/s, grid-relative.
        Required together when wind speed is enabled; H is their modulus,
        which no grid rotation can change.
    analysis_times
        Optional full analysis schedule of the run.  When given, a report
        is kept only at the analysis nearest its valid time, so no report
        enters two cycles.  When omitted the caller owns that guarantee.

    Returns
    -------
    ``(batches, provenance)``.  Batch names are ``<seam quantity>:<source>``
    (e.g. ``temperature_2m:asos``) -- the seam's own quantity id, suffixed
    with the record's provenance source, never a hardcoded network name.
    Observations sit at k = 0 of the station's gridpoint; the vertical
    localisation from there is well-defined because the filter's
    GridGeometry carries real column heights.
    """

    record = read_record(source)
    if target_grid is None:
        raise SurfaceObsError(
            "target_grid is required: placement and the elevation gate "
            "are the grid's, and no identity string can stand in for it")
    analysis_time = _as_utc(analysis_time, label="analysis_time")

    if config.temperature and simulated_t2 is None:
        raise SurfaceObsError(
            "temperature_error_k is stated but simulated_t2 was not "
            "given; H(x) for 2 m temperature is the member t2 diagnostic "
            "and nobody else can supply it")
    if config.wind_speed and (simulated_u10 is None
                              or simulated_v10 is None):
        raise SurfaceObsError(
            "wind_speed_error_ms is stated but simulated_u10/simulated_v10 "
            "were not both given; H(x) for wind speed is hypot(u10, v10) "
            "of the member diagnostics")

    ny, nx = int(target_grid.ny), int(target_grid.nx)
    nz = int(target_grid.nz)
    plane = (ny, nx)
    members = None
    t2_stack = u10_stack = v10_stack = None
    if config.temperature:
        t2_stack = _member_stack("simulated_t2", simulated_t2, members,
                                 plane)
        members = t2_stack.shape[0]
    if config.wind_speed:
        u10_stack = _member_stack("simulated_u10", simulated_u10, members,
                                  plane)
        members = u10_stack.shape[0]
        v10_stack = _member_stack("simulated_v10", simulated_v10, members,
                                  plane)

    stations = {str(s["station_id"]): s for s in record["stations"]}
    counts = {
        "stations_in_record": len(stations),
        "reports_in_record": len(record["reports"]),
        "reports_refused_flagged": 0,
        "reports_outside_window": 0,
        "reports_routed_to_other_analysis": 0,
        "reports_superseded_same_station": 0,
        "stations_unknown_to_table": 0,
        "stations_outside_domain": 0,
        "stations_refused_elevation": 0,
        "stations_superseded_colocated": 0,
        "values_missing_by_quantity": {},
        "values_nonfinite_by_quantity": {},
    }
    chosen = _select_reports(record, analysis_time, analysis_times,
                             float(config.max_age_seconds), counts)

    terrain = np.asarray(target_grid.terrain_m, dtype=np.float64)
    gate_m = float(config.elevation_max_diff_m)

    #: gridpoint -> the accepted station's placement record; colocation
    #: keeps the smaller elevation difference, then the smaller |age|,
    #: then the lexicographically first id -- deterministic, and counted.
    placed: dict[tuple[int, int], dict] = {}
    qc: dict[str, dict] = {}

    for station_id in sorted(chosen):
        entry = chosen[station_id]
        station = stations.get(station_id)
        if station is None:
            counts["stations_unknown_to_table"] += 1
            qc[station_id] = {"outcome": "refused",
                              "reason": "not in the record's own station "
                                        "table"}
            continue
        lat = float(station["latitude"])
        lon = float(station["longitude"])
        i_frac, j_frac = target_grid.mass_index(lat, lon)
        i = int(np.rint(float(i_frac)))
        j = int(np.rint(float(j_frac)))
        if not bool(target_grid.inside(i, j)):
            counts["stations_outside_domain"] += 1
            qc[station_id] = {"outcome": "refused",
                              "reason": "outside the target grid"}
            continue
        elevation = float(station["elevation_m"])
        diff = elevation - float(terrain[j, i])
        if abs(diff) > gate_m:
            counts["stations_refused_elevation"] += 1
            qc[station_id] = {
                "outcome": "refused",
                "reason": "elevation gate",
                "station_elevation_m": elevation,
                "model_terrain_m": float(terrain[j, i]),
                "difference_m": diff,
                "gate_m": gate_m,
            }
            continue
        candidate = {"station_id": station_id, "i": i, "j": j,
                     "elevation_diff_m": diff, "age_s": entry["age_s"],
                     "report": entry["report"]}
        incumbent = placed.get((j, i))
        if incumbent is not None:
            keep, drop = sorted(
                (incumbent, candidate),
                key=lambda c: (abs(c["elevation_diff_m"]),
                               abs(c["age_s"]), c["station_id"]))
            placed[(j, i)] = keep
            counts["stations_superseded_colocated"] += 1
            qc[drop["station_id"]] = {
                "outcome": "refused",
                "reason": "colocated with "
                          f"{keep['station_id']} at the same gridpoint",
            }
            if drop is candidate:
                continue
        else:
            placed[(j, i)] = candidate
        qc[station_id] = {"outcome": "accepted", "i": i, "j": j,
                          "elevation_diff_m": diff,
                          "age_s": entry["age_s"]}

    # -- batches -------------------------------------------------------------

    source_label = str((record.get("provenance") or {}).get("source")
                       or "surface")
    shape = (nz, ny, nx)
    inflation = float(config.error_inflation)
    batches: list[GriddedObs] = []
    used: list[dict] = []

    quantity_plan = []
    if config.temperature:
        quantity_plan.append(
            (TEMPERATURE_QUANTITY, "K",
             float(config.temperature_error_k), t2_stack,
             config.temperature_localization))
    if config.wind_speed:
        speed_stack = np.hypot(u10_stack, v10_stack)
        quantity_plan.append(
            (WIND_SPEED_QUANTITY, "m s-1",
             float(config.wind_speed_error_ms), speed_stack,
             config.wind_localization))

    ages_used: list[float] = []
    for quantity, units, sigma, member_plane, localization in quantity_plan:
        counts["values_missing_by_quantity"][quantity] = 0
        counts["values_nonfinite_by_quantity"][quantity] = 0
        values = np.full(shape, np.nan, dtype=np.float64)
        mask = np.zeros(shape, dtype=bool)
        observed = 0
        for (j, i), placement in sorted(placed.items()):
            raw = placement["report"].get("values", {}).get(quantity)
            if raw is None:
                counts["values_missing_by_quantity"][quantity] += 1
                continue
            value = float(raw)
            if not math.isfinite(value):
                counts["values_nonfinite_by_quantity"][quantity] += 1
                qc[placement["station_id"]].setdefault(
                    "nonfinite_quantities", []).append(quantity)
                continue
            values[0, j, i] = value
            mask[0, j, i] = True
            observed += 1
            ages_used.append(float(placement["age_s"]))
        sigma_eff = sigma * inflation
        errors = np.full(shape, sigma_eff, dtype=np.float64)
        simulated = np.zeros((member_plane.shape[0],) + shape,
                             dtype=np.float64)
        simulated[:, 0, :, :] = member_plane
        name = f"{quantity}:{source_label}"
        batches.append(GriddedObs(name=name, values=values, errors=errors,
                                  simulated=simulated, mask=mask,
                                  localization=localization))
        used.append({
            "name": name, "kind": "surface", "quantity": quantity,
            "units": units, "level": "k=0 of the station column",
            "error_stddev": sigma, "error_inflation": inflation,
            "observed_points": observed,
        })

    # -- provenance ----------------------------------------------------------

    stride = None
    valid_times = [
        _parse_seam_time(t, where="valid_times[]")
        for t in record.get("valid_times", [])]
    if len(valid_times) >= 2:
        deltas = {
            (b - a).total_seconds()
            for a, b in zip(valid_times, valid_times[1:])}
        stride = sorted(deltas)
    ages = sorted(set(ages_used))
    provenance = {
        "schema": ADAPTER_SCHEMA,
        "stability": "experimental",
        "obs_schema": record.get("schema"),
        "obs_status": record.get("status"),
        "source": source_label,
        "record_provenance": dict(record.get("provenance") or {}),
        "station_table_sha256": record.get("station_table_sha256"),
        "analysis_time": analysis_time.isoformat(),
        "max_age_seconds": float(config.max_age_seconds),
        "elevation_max_diff_m": gate_m,
        "grid_identity_sha256": target_grid.identity_sha256(),
        "grid_shape": [nz, ny, nx],
        "stations_placed": len(placed),
        "counts": counts,
        "station_qc": qc,
        "batches": used,
        "cadence": {
            "record_valid_time_strides_s": stride,
            "report_ages_at_assimilation_s": {
                "min": (min(ages) if ages else None),
                "max": (max(ages) if ages else None),
            },
            "note": "the seam is hourly-matched by decoder design; "
                    "sub-hourly cadence needs the IEM asos1min route in "
                    "rw-obs (upstream)",
        },
        "notes": [
            "wind is SPEED only: drct is dropped at decode (v1 seam); "
            "H is hypot(u10, v10), rotation-invariant",
            "temperature H is the member t2 diagnostic, snapshotted at "
            "leg end, never leg start",
            "errors are standard deviations, stated by the caller",
            "each report enters at most one analysis when the schedule "
            "is supplied",
        ],
    }
    return batches, provenance
