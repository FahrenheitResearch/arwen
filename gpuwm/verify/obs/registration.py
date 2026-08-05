"""Pins for the observation battery, hashed before any score is looked at.

This is the chaos-envelope registration discipline
(:mod:`gpuwm.verify.chaos_envelope`) transplanted to obs scoring: every
parameter that could be chosen after seeing a result is chosen *here*, the
parameter block is canonically hashed, and the hash plus the evaluating
commit travel in every score file the battery writes.  A score whose
registration hash does not match the registration it is published beside is
not a score of that instrument.

Two things this module refuses to do:

* **It does not ratify its own numbers.**  The thresholds, the neighborhood
  set, the primary scalar, the significance level and the guardrail list are
  *proposals* until the project owner rules on them.  A registration built
  without an explicit ratification reference carries
  ``rule_status = "proposed-unratified"``, and the promotion evaluator
  refuses to call anything promoted under that status.  An instrument that
  ratifies its own gates measures its author.
* **It does not know a case.**  Case identity is data in the ``cases`` block
  -- ids, init instants, station-set digests -- and the scoring parameters
  above it are written in leads, thresholds and metres.  The same parameter
  block scores every case in the set, which is the only way the cross-case
  statistics in the promotion rule mean anything.

The ``DEFAULT_*`` values below transcribe the specification's proposals so a
caller does not retype them; they are defaults, not decisions, and the
``rule_status`` field is what says so.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from gpuwm.verify.chaos_envelope import canonical_hash, sha256_file, write_json
from gpuwm.verify.obs import fss
from gpuwm.verify.obs.contracts import SCORED_SURFACE_VARIABLES

REGISTRATION_SCHEMA = "gpuwm.obs-battery-registration/v1"

RATIFIED = "ratified"
UNRATIFIED = "proposed-unratified"
#: Ratified by an agent acting under a written standing delegation from the
#: owner, rather than by the owner ruling on these numbers one at a time.  It
#: is a distinct status and not a synonym for :data:`RATIFIED`, because the
#: provenance is different and a reader is entitled to see which one a
#: verdict rests on.  It carries the same promotion authority, and it carries
#: an overrule window that plain ratification does not.
RATIFIED_UNDER_DELEGATION = "ratified-under-delegation"

#: The statuses under which the promotion evaluator may return a verdict of
#: "promote" rather than withholding it.  Enumerated once, here, so a third
#: status can never be added on the evaluator's side alone.
RATIFYING_STATUSES = (RATIFIED, RATIFIED_UNDER_DELEGATION)
RULE_STATUSES = (RATIFIED, RATIFIED_UNDER_DELEGATION, UNRATIFIED)

#: Reflectivity thresholds (dBZ) and neighborhood half-widths (cells).  At
#: dx 3 km the half-widths are box lengths of 9, 27, 51 and 99 km.
DEFAULT_REFLECTIVITY_THRESHOLDS_DBZ = (20.0, 30.0, 40.0, 50.0)
DEFAULT_NEIGHBORHOOD_HALF_WIDTHS = (1, 4, 8, 16)

#: The primary scalar: mean over the scored leads of FSS at 30 dBZ and the
#: 27 km box.  One number per case per arm, and the promotion rule's input.
DEFAULT_PRIMARY_THRESHOLD_DBZ = 30.0
DEFAULT_PRIMARY_HALF_WIDTH = 4

#: Scoring window in forecast hours.  The first hours are reported and never
#: scored: a free-forecast verdict must not carry the spin-up hole.
DEFAULT_SPINUP_END_HOUR = 2
DEFAULT_SCORE_END_HOUR = 18
DEFAULT_TAIL_END_HOUR = 24

#: Rim excluded from scoring beyond the specified and relaxation rows.
DEFAULT_INTERIOR_RIM_M = 45000.0

#: Coverage floor for frame SELECTION inside the registered matching window
#: (amendment v2.1, docs/public/receipts/obsbattery/registration/
#: REGISTRATION-v2.1-amendment-frame-coverage.md).  It is the case bar's own
#: coverage requirement -- no more than 10% of the interior masked -- applied
#: to the choice of frame rather than only to the verdict on it: inside the
#: unchanged window the scored frame is the nearest one that meets this, and
#: a lead whose whole window is below it is recorded missing-obs and left out
#: of the lead mean.  A registered upstream ingest outage must not kill a case
#: when a qualifying frame sits inside the tolerance already registered.
DEFAULT_FRAME_COVERAGE_FLOOR = 0.9

#: Neighborhood boundary treatment: outside the domain was not observed, so
#: the boxcar is zero-padded rather than edge-extended.  Registered because
#: at the widest neighborhood the scored region reaches the array edge.
DEFAULT_NEIGHBORHOOD_BOUNDARY = fss.ZERO_BOUNDARY

#: Surface admission and matching.
DEFAULT_ELEVATION_TOLERANCE_M = 100.0
DEFAULT_MINIMUM_REPORTING_FRACTION = 0.8
DEFAULT_MATCH_TOLERANCE_SECONDS = 600
DEFAULT_MAXIMUM_SCREEN_FRACTION = 0.05

#: Precipitation thresholds (mm) for the accumulation scores.
DEFAULT_PRECIPITATION_THRESHOLDS_MM = (2.5, 10.0, 25.0)
DEFAULT_PRECIPITATION_WINDOW_HOURS = (1, 6)
DEFAULT_PRECIPITATION_GUARDRAIL_THRESHOLD_MM = 10.0
DEFAULT_PRECIPITATION_GUARDRAIL_HALF_WIDTH = 8
DEFAULT_PRECIPITATION_GUARDRAIL_WINDOW_HOURS = 6

#: Promotion rule.
DEFAULT_ALPHA = 0.05

#: Guardrail directions.  A guardrail is a metric a patch may not damage; the
#: direction says which way damage lies, because the same "no worse than the
#: twin band" sentence means opposite arithmetic for an RMSE and an FSS.
LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"

DEFAULT_GUARDRAILS: tuple[dict[str, object], ...] = tuple(
    {"name": f"median_station_rmse:{variable}", "direction": LOWER_IS_BETTER,
     "source": "surface"}
    for variable in SCORED_SURFACE_VARIABLES
) + (
    {"name": "mean_fss:precipitation", "direction": HIGHER_IS_BETTER,
     "source": "precipitation"},
)


def scored_lead_hours(*, spinup_end_hour: int = DEFAULT_SPINUP_END_HOUR,
                      score_end_hour: int = DEFAULT_SCORE_END_HOUR
                      ) -> tuple[int, ...]:
    """The hourly leads inside the scoring window, inclusive at both ends."""
    if not 0 <= spinup_end_hour < score_end_hour:
        raise ValueError("the scoring window must start before it ends")
    return tuple(range(int(spinup_end_hour), int(score_end_hour) + 1))


def reflectivity_parameters(
        *, thresholds_dbz: Sequence[float] = DEFAULT_REFLECTIVITY_THRESHOLDS_DBZ,
        half_widths: Sequence[int] = DEFAULT_NEIGHBORHOOD_HALF_WIDTHS,
        primary_threshold_dbz: float = DEFAULT_PRIMARY_THRESHOLD_DBZ,
        primary_half_width: int = DEFAULT_PRIMARY_HALF_WIDTH,
        regrid_method: str = "nearest",
        regrid_max_distance_m: float = 4500.0,
        obs_match_tolerance_seconds: int = 240,
        interior_rim_m: float = DEFAULT_INTERIOR_RIM_M,
        neighborhood_boundary: str = DEFAULT_NEIGHBORHOOD_BOUNDARY,
        frequency_matched: bool = True,
        frame_coverage_floor: float = DEFAULT_FRAME_COVERAGE_FLOOR,
        ) -> dict[str, object]:
    """Pins for the primary instrument: neighborhood FSS on composite dBZ."""
    thresholds = tuple(float(value) for value in thresholds_dbz)
    widths = tuple(int(value) for value in half_widths)
    if not thresholds or not widths:
        raise ValueError("reflectivity scoring needs thresholds and neighborhoods")
    if float(primary_threshold_dbz) not in thresholds:
        raise ValueError("the primary threshold must be one of the scored thresholds")
    if int(primary_half_width) not in widths:
        raise ValueError("the primary neighborhood must be one of the scored ones")
    if any(width < 0 for width in widths):
        raise ValueError("neighborhood half-widths must be non-negative")
    if not 0.0 <= float(frame_coverage_floor) <= 1.0:
        raise ValueError("the frame coverage floor is a fraction in [0, 1]")
    return {
        "model_field": "column maximum over k of REFL_10CM, each model's own "
                       "scheme-consistent reflectivity operator",
        "observed_quantity": "composite_reflectivity",
        "thresholds_dbz": list(thresholds),
        "neighborhood_half_widths_cells": list(widths),
        "primary_threshold_dbz": float(primary_threshold_dbz),
        "primary_half_width_cells": int(primary_half_width),
        "primary_scalar": (
            "mean over the scored leads of FSS at the primary threshold and "
            "primary neighborhood"),
        "regrid_method": str(regrid_method),
        "regrid_max_distance_m": float(regrid_max_distance_m),
        "obs_match_tolerance_seconds": int(obs_match_tolerance_seconds),
        "frame_coverage_floor": float(frame_coverage_floor),
        "frame_selection": (
            "inside the matching tolerance, the scored frame is the nearest "
            "frame whose observed fraction meets the coverage floor (ties to "
            "the earlier frame); a lead whose whole window is below the floor "
            "is recorded missing-obs, named in the score record, and left out "
            "of the lead mean. The tolerance itself does not widen"),
        "interior_rim_m": float(interior_rim_m),
        "neighborhood_boundary": str(neighborhood_boundary),
        "frequency_matched_reported": bool(frequency_matched),
        "fss_definition": (
            "Roberts & Lean 2008 with a shared validity mask: neighborhood "
            "fraction = boxcar(event AND valid)/boxcar(valid); cells with an "
            "empty neighborhood are dropped; FSS = 1 when both fraction "
            "fields are identically zero over the scored region"),
        "frequency_matched_definition": (
            "the model threshold is the linear-interpolated quantile of the "
            "model field over the shared-valid scored cells at 1 - f_obs, so "
            "model and observed exceedance coverage agree"),
        "fss_useful_definition": "0.5 + f_obs/2",
        "missing_policy": (
            "observed missing or flagged cells are excluded from both fields; "
            "an arm with a missing frame fails, it is not skipped"),
    }


def surface_parameters(
        *, variables: Sequence[str] = SCORED_SURFACE_VARIABLES,
        interpolation: str = "bilinear",
        elevation_tolerance_m: float = DEFAULT_ELEVATION_TOLERANCE_M,
        minimum_reporting_fraction: float = DEFAULT_MINIMUM_REPORTING_FRACTION,
        match_tolerance_seconds: int = DEFAULT_MATCH_TOLERANCE_SECONDS,
        maximum_screen_fraction: float = DEFAULT_MAXIMUM_SCREEN_FRACTION,
        reported_unscored: Sequence[str] = ("mslp",)) -> dict[str, object]:
    """Pins for the surface comparison against point observations."""
    scored = tuple(str(name) for name in variables)
    if not scored:
        raise ValueError("surface scoring needs at least one variable")
    overlap = sorted(set(scored) & {str(name) for name in reported_unscored})
    if overlap:
        raise ValueError(
            f"{overlap} cannot be both scored and reported-unscored")
    return {
        "variables": list(scored),
        "reported_unscored": list(reported_unscored),
        "model_diagnostics": (
            "surface diagnostics come from the mandated science core; the "
            "battery reimplements none of them"),
        "interpolation": str(interpolation),
        "station_admission": (
            "inside the interior mask, land, |model terrain - station "
            "elevation| <= tolerance (dropped, never corrected), reporting at "
            "least the minimum fraction of scored hours"),
        "elevation_tolerance_m": float(elevation_tolerance_m),
        "minimum_reporting_fraction": float(minimum_reporting_fraction),
        "match_tolerance_seconds": int(match_tolerance_seconds),
        "maximum_screen_fraction": float(maximum_screen_fraction),
        "guardrail_statistic": (
            "median over stations of the per-station RMSE"),
        "pairing": (
            "one frozen station set per case scores every arm; a (station, "
            "hour) with no report inside the tolerance is missing for all arms"),
    }


def precipitation_parameters(
        *, thresholds_mm: Sequence[float] = DEFAULT_PRECIPITATION_THRESHOLDS_MM,
        window_hours: Sequence[int] = DEFAULT_PRECIPITATION_WINDOW_HOURS,
        half_widths: Sequence[int] = DEFAULT_NEIGHBORHOOD_HALF_WIDTHS,
        guardrail_threshold_mm: float = DEFAULT_PRECIPITATION_GUARDRAIL_THRESHOLD_MM,
        guardrail_half_width: int = DEFAULT_PRECIPITATION_GUARDRAIL_HALF_WIDTH,
        guardrail_window_hours: int = DEFAULT_PRECIPITATION_GUARDRAIL_WINDOW_HOURS,
        regrid_method: str = "nearest",
        regrid_max_distance_m: float = 7000.0,
        neighborhood_boundary: str = DEFAULT_NEIGHBORHOOD_BOUNDARY,
        ) -> dict[str, object]:
    """Pins for the accumulation comparison against a gauge/radar analysis."""
    thresholds = tuple(float(value) for value in thresholds_mm)
    windows = tuple(int(value) for value in window_hours)
    if not thresholds or not windows:
        raise ValueError("precipitation scoring needs thresholds and windows")
    if int(guardrail_window_hours) not in windows:
        raise ValueError("the guardrail window must be one of the scored windows")
    if float(guardrail_threshold_mm) not in thresholds:
        raise ValueError("the guardrail threshold must be one of the scored ones")
    if int(guardrail_half_width) not in tuple(int(w) for w in half_widths):
        raise ValueError("the guardrail neighborhood must be one of the scored ones")
    return {
        "model_field": "run-total grid-scale precipitation differenced over "
                       "the accumulation window",
        "observed_quantity": "precipitation_accumulation",
        "thresholds_mm": list(thresholds),
        "window_hours": list(windows),
        "neighborhood_half_widths_cells": [int(w) for w in half_widths],
        "neighborhood_boundary": str(neighborhood_boundary),
        "scoring_grid": "the observation analysis grid",
        "regrid_method": str(regrid_method),
        "regrid_max_distance_m": float(regrid_max_distance_m),
        "guardrail_threshold_mm": float(guardrail_threshold_mm),
        "guardrail_half_width_cells": int(guardrail_half_width),
        "guardrail_window_hours": int(guardrail_window_hours),
        "guardrail_statistic": (
            "case mean over accumulation windows of FSS at the guardrail "
            "threshold and neighborhood"),
        "contingency_scores": (
            "equitable threat score and frequency bias at the same thresholds"),
    }


def promotion_parameters(
        *, alpha: float = DEFAULT_ALPHA,
        guardrails: Sequence[Mapping[str, object]] = DEFAULT_GUARDRAILS,
        twin_band_statistic: str = (
            "median over cases of |S(twin) - S(control)| for the same score"),
        ratification_reference: str = "",
        delegation_reference: str = "",
        overrule_window: str = "") -> dict[str, object]:
    """Pins for the promotion rule, and its ratification status.

    ``ratification_reference`` is the owner's ruling on THESE numbers -- a
    commit, a document section, a dated decision.  Empty means the numbers are
    still proposals, and the evaluator will say so on every verdict it writes.

    ``delegation_reference`` is the other way a rule becomes binding: a written
    standing delegation under which an agent ratified them on the owner's
    behalf.  It quotes the delegation and dates it, and it produces
    :data:`RATIFIED_UNDER_DELEGATION` rather than :data:`RATIFIED`, because a
    reader is entitled to know which of the two a verdict rests on.  A
    delegated ratification also carries an ``overrule_window``: the period in
    which the owner may still overrule any number, stated explicitly rather
    than left to be assumed either way.

    The two references are mutually exclusive.  A record claiming both would
    be a record whose authority nobody could name.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if str(ratification_reference).strip() and str(delegation_reference).strip():
        raise ValueError(
            "a promotion rule is ratified by the owner or under a delegation, "
            "never both; a record claiming both authorities names neither")
    if str(delegation_reference).strip() and not str(overrule_window).strip():
        raise ValueError(
            "a delegated ratification must state its overrule window; "
            "'the owner may still overrule' is a fact with an end date, and "
            "omitting the date is what turns a delegation into a fait accompli")
    rails = [dict(entry) for entry in guardrails]
    for entry in rails:
        if entry.get("direction") not in (LOWER_IS_BETTER, HIGHER_IS_BETTER):
            raise ValueError(f"guardrail {entry} needs a registered direction")
        if not str(entry.get("name", "")).strip():
            raise ValueError("every guardrail needs a name")
    reference = str(ratification_reference).strip()
    delegated = str(delegation_reference).strip()
    if reference:
        status = RATIFIED
    elif delegated:
        status = RATIFIED_UNDER_DELEGATION
    else:
        status = UNRATIFIED
    return {
        "alpha": float(alpha),
        "test": ("one-sided exact Wilcoxon signed-rank test on the per-case "
                 "differences, H1: median > 0"),
        "zero_difference_policy": (
            "exact-zero differences are dropped and the sample size reduced "
            "(Wilcoxon 1945); tied magnitudes take average ranks and the null "
            "distribution is enumerated over sign assignments of the observed "
            "absolute ranks, which keeps the test exact under ties"),
        "chaos_floor": (
            "the cross-case median difference must exceed the twin band for "
            "the same score"),
        "twin_band_statistic": str(twin_band_statistic),
        "guardrails": rails,
        "guardrail_rule": (
            "no guardrail may degrade the cross-case median by more than that "
            "metric's own twin band"),
        "integrity_rule": (
            "every scored integration passed its dual-run byte screen, every "
            "observation re-hashed clean, every receipt carries its "
            "registration hash and evaluator commit, and no stand-in input "
            "appears anywhere in the evidence"),
        "interaction_rule": (
            "single-patch arms are the promotion evidence; a combined-patch "
            "arm is reported and is never a promotion basis"),
        "ratification_reference": reference,
        "delegation_reference": delegated,
        "overrule_window": str(overrule_window).strip() if delegated else "",
        "rule_status": status,
    }


def make_registration(*, evaluator_commit: str,
                      reflectivity: Mapping[str, object],
                      surface: Mapping[str, object],
                      precipitation: Mapping[str, object],
                      promotion: Mapping[str, object],
                      cases: Sequence[Mapping[str, object]],
                      arms: Sequence[Mapping[str, object]],
                      twin: Mapping[str, object],
                      scored_lead_hours_: Sequence[int] | None = None,
                      reader: Mapping[str, object] | None = None,
                      ) -> dict[str, object]:
    """Freeze every choice, hash the block, and stamp the evaluating commit."""
    commit = str(evaluator_commit).lower()
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("evaluator_commit must be a 40-hex Git SHA")
    if not cases:
        raise ValueError("a battery registration needs at least one case")
    if not arms:
        raise ValueError("a battery registration needs at least one arm")
    case_ids = [str(case["case_id"]) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("the case set carries duplicate case ids")
    for case in cases:
        for key in ("case_id", "init_time"):
            if not str(case.get(key, "")).strip():
                raise ValueError(f"case {case} is missing {key}")
        datetime.fromisoformat(str(case["init_time"]))
    arm_ids = [str(arm["arm_id"]) for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("the arm set carries duplicate arm ids")
    leads = (tuple(scored_lead_hours())
             if scored_lead_hours_ is None
             else tuple(int(hour) for hour in scored_lead_hours_))
    if not leads or sorted(leads) != list(leads) or len(set(leads)) != len(leads):
        raise ValueError("scored lead hours must be ascending and unique")
    parameters = {
        "scored_lead_hours": list(leads),
        "spinup_policy": (
            "leads before the first scored hour are reported separately and "
            "never scored"),
        "reflectivity": dict(reflectivity),
        "surface": dict(surface),
        "precipitation": dict(precipitation),
        "promotion": dict(promotion),
        "cases": [dict(case) for case in cases],
        "arms": [dict(arm) for arm in arms],
        "twin": dict(twin),
    }
    if reader is not None:
        # The reader is pinned by CONTENT, not by packaging.  A distribution
        # version is a label somebody typed; the source tree's commit and the
        # built artifact's digest are what actually answers a getvar call, and
        # they belong inside the hashed block so a swapped build cannot pass
        # under an unchanged version string.
        parameters["reader"] = dict(reader)
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "evaluator_commit": commit,
        "rule_status": str(dict(promotion).get("rule_status", UNRATIFIED)),
        "parameters": parameters,
    }
    registration["registration_sha256"] = canonical_hash(parameters)
    return registration


def validate_registration(registration: Mapping[str, object]
                          ) -> dict[str, object]:
    """Refuse a registration whose pins no longer hash to its own digest."""
    reg = dict(registration)
    required = {"schema", "evaluator_commit", "parameters",
                "registration_sha256", "rule_status"}
    missing = required - set(reg)
    if missing:
        raise ValueError(f"battery registration is missing {sorted(missing)}")
    if reg["schema"] != REGISTRATION_SCHEMA:
        raise ValueError("battery registration schema mismatch")
    parameters = reg["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("battery registration parameters must be an object")
    if canonical_hash(parameters) != reg["registration_sha256"]:
        raise ValueError("battery registration hash does not match its pins")
    commit = str(reg["evaluator_commit"])
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError("evaluator_commit must be a 40-hex Git SHA")
    if reg["rule_status"] not in RULE_STATUSES:
        raise ValueError(f"unknown rule status {reg['rule_status']!r}")
    if reg["rule_status"] != parameters["promotion"].get("rule_status"):
        raise ValueError(
            "the registration's rule status disagrees with its promotion pins")
    if reg["rule_status"] == RATIFIED_UNDER_DELEGATION:
        promotion = parameters["promotion"]
        for key in ("delegation_reference", "overrule_window"):
            if not str(promotion.get(key, "")).strip():
                raise ValueError(
                    f"a delegated ratification must carry {key}; the whole "
                    f"difference between this status and plain ratification "
                    f"is that a reader can see where the authority came from")
    return reg


__all__ = [
    "DEFAULT_ALPHA", "DEFAULT_ELEVATION_TOLERANCE_M",
    "DEFAULT_FRAME_COVERAGE_FLOOR", "DEFAULT_GUARDRAILS",
    "DEFAULT_INTERIOR_RIM_M", "DEFAULT_MATCH_TOLERANCE_SECONDS",
    "DEFAULT_NEIGHBORHOOD_BOUNDARY",
    "DEFAULT_MAXIMUM_SCREEN_FRACTION", "DEFAULT_MINIMUM_REPORTING_FRACTION",
    "DEFAULT_NEIGHBORHOOD_HALF_WIDTHS", "DEFAULT_PRECIPITATION_THRESHOLDS_MM",
    "DEFAULT_PRECIPITATION_WINDOW_HOURS", "DEFAULT_PRIMARY_HALF_WIDTH",
    "DEFAULT_PRIMARY_THRESHOLD_DBZ", "DEFAULT_REFLECTIVITY_THRESHOLDS_DBZ",
    "DEFAULT_SCORE_END_HOUR", "DEFAULT_SPINUP_END_HOUR",
    "DEFAULT_TAIL_END_HOUR", "HIGHER_IS_BETTER", "LOWER_IS_BETTER",
    "RATIFIED", "RATIFIED_UNDER_DELEGATION", "RATIFYING_STATUSES",
    "REGISTRATION_SCHEMA", "RULE_STATUSES", "UNRATIFIED", "canonical_hash",
    "make_registration", "precipitation_parameters", "promotion_parameters",
    "reflectivity_parameters", "scored_lead_hours", "sha256_file",
    "surface_parameters", "validate_registration", "write_json",
]
