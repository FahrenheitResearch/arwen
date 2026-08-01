"""Chaos envelope over an ensemble of like-for-like runs.

An envelope answers one question: how far apart do two runs of the SAME
configuration drift, when the only thing separating them is a perturbation
too small to mean anything physically?  That spread is the yardstick a
candidate run is then measured against -- a candidate inside the spread is
indistinguishable from ordinary divergence, and a candidate outside it is not.

Nothing here knows a case.  The domains, their grid spacings, the scored
leads, the fields and the thresholds all arrive in a registration, whose hash
is committed before any output is inspected.  The metric arithmetic is
:mod:`gpuwm.verify.field_metrics` and :mod:`gpuwm.verify.spectral`; the
spectral pins are the one part of the registration a caller does not choose,
since they are pre-registered in that module and hashed into this one's
registration.

Four properties this module exists to keep honest:

* **E95 is nearest-rank, not interpolated.**  ``sorted[ceil(p*n)-1]`` over the
  unordered member-pair distances, the definition the registered gate record
  states.  An interpolated percentile would invent a value no pair achieved.
* **A zero envelope is flagged, not hidden.**  When every pair agrees exactly
  the envelope is degenerate: it measures nothing, so the row carries
  ``envelope_degenerate`` and its verdict says so instead of quietly passing
  every candidate.
* **An envelope belongs to the run it came from.**  The receipt records the
  member geometry and physics identity, and :func:`check_config_identity`
  refuses a receipt whose members are not the configuration it is being cited
  beside.  An envelope measured on a different ladder is a different yardstick.
* **The classes it does not cover are named.**  A receipt lists the metric
  classes its rows compute and, beside them, the classes this program leaves
  unscheduled.  A coverage map that omitted the gap would read as if the
  scheduled set were the whole space of statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from gpuwm.verify import field_metrics, spectral

ENVELOPE_SCHEMA = "gpuwm.chaos-envelope/v1"
REGISTRATION_SCHEMA = "gpuwm.chaos-envelope-registration/v1"

#: Metric classes an envelope row can carry: state, boundary, object,
#: neighborhood, and the scale-resolved class the other four cannot express.
METRIC_CLASSES = (
    "low_pass_state_rmse",
    "applied_boundary_increment_rmse",
    "first_object_time_difference",
    "fss_distance",
    "spectral_log_distance",
)

#: Metric classes this receipt schema names and does NOT compute.  Carrying
#: the gap beside the covered classes is the difference between a coverage
#: map and a claim: no row here is a distributional statistic, and a reader
#: who wants one is told so by the receipt rather than by its absence.
UNSCHEDULED_METRIC_CLASSES = (
    {
        "metric_class": "distributional",
        "status": "unscheduled",
        "note": ("histogram, quantile and CRPS-type agreement is scheduled "
                 "nowhere in this program; no row in this receipt computes "
                 "one, and nothing here may be read as covering that class"),
    },
)

_FRAME_RE = re.compile(
    r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})"
    r"(?:\..*)?$")


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def require_spectral_agreement(reflectivity_field: str) -> None:
    """The spectral pins name a reflectivity variable; it must be this one.

    The spectral class is pre-registered on named fields, so the composite it
    scores is fixed.  If a registration scores its objects and its FSS on some
    other reflectivity variable, one receipt would carry two reflectivities
    under one set of rows.  Substituting either side would be re-writing a pin
    at scoring time, so this refuses instead.
    """
    pinned = spectral.pinned_variable("composite_reflectivity")
    if str(reflectivity_field) != pinned:
        raise ValueError(
            f"the spectral pins score {pinned}; a registration on "
            f"{reflectivity_field!r} cannot carry the spectral class")


def make_registration(*, start_time: str, domain_dx_m: Mapping[str, float],
                      state_fields: Sequence[str], leads_seconds: Sequence[int],
                      cadence_seconds: int, reflectivity_field: str,
                      reflectivity_threshold: float,
                      low_pass_physical_width_m: float,
                      low_pass_interior_exclusion_cells: int,
                      boundary_width_cells: int,
                      fss_radius_m: float, object_min_area_km2: float,
                      object_connectivity: int, evaluator_commit: str,
                      envelope_percentile: float = 95.0) -> dict[str, object]:
    """Pin every metric choice before any member output is inspected."""
    if not domain_dx_m or not state_fields or not leads_seconds:
        raise ValueError("an envelope registration needs domains, fields and leads")
    if cadence_seconds <= 0 or any(
            lead <= 0 or lead % cadence_seconds for lead in leads_seconds):
        raise ValueError("every scored lead must be a positive multiple of the cadence")
    if not 0.0 < envelope_percentile <= 100.0:
        raise ValueError("envelope percentile must lie in (0, 100]")
    if (len(evaluator_commit) != 40
            or any(ch not in "0123456789abcdef" for ch in evaluator_commit.lower())):
        raise ValueError("evaluator_commit must be a 40-hex Git SHA")
    if object_connectivity not in (4, 8):
        raise ValueError("object connectivity must be 4 or 8")
    require_spectral_agreement(reflectivity_field)
    datetime.fromisoformat(start_time)
    spectral_pins = spectral.registration()
    parameters = {
        "domain_dx_m": {str(k): float(v) for k, v in domain_dx_m.items()},
        "state_fields": list(state_fields),
        "leads_seconds": sorted(int(lead) for lead in leads_seconds),
        "cadence_seconds": int(cadence_seconds),
        "reflectivity_field": str(reflectivity_field),
        "reflectivity_threshold": float(reflectivity_threshold),
        "low_pass_filter": "separable edge-extended square boxcar",
        "low_pass_physical_width_m": float(low_pass_physical_width_m),
        "low_pass_interior_exclusion_cells": int(low_pass_interior_exclusion_cells),
        "boundary_width_cells": int(boundary_width_cells),
        "boundary_increment_definition": (
            "RMSE of (frame[lead]-frame[lead-cadence]) differences between "
            "runs over the outer boundary_width_cells frame"),
        "fss_radius_m": float(fss_radius_m),
        "fss_zero_denominator": "FSS=1 when both fractions are identically zero",
        "object_min_area_km2": float(object_min_area_km2),
        "object_connectivity": int(object_connectivity),
        "object_timing_definition": (
            "valid second of the first qualifying object at or before the "
            "lead; quiet runs take lead+cadence"),
        "envelope_statistic": (
            "E{p} = nearest-rank sorted_pair_distance[ceil({f}*n_pairs)-1]"
        ).format(p=envelope_percentile, f=envelope_percentile / 100.0),
        "envelope_percentile": float(envelope_percentile),
        "missing_or_nonfinite_policy": "fail",
        "spectral": spectral_pins["pins"],
        "spectral_pins_sha256": spectral_pins["pins_sha256"],
        "unscheduled_metric_classes": [
            dict(entry) for entry in UNSCHEDULED_METRIC_CLASSES],
    }
    registration = {
        "schema": REGISTRATION_SCHEMA,
        "start_time": start_time,
        "evaluator_commit": evaluator_commit.lower(),
        "parameters": parameters,
    }
    registration["registration_sha256"] = canonical_hash(parameters)
    return registration


def validate_registration(registration: Mapping[str, object]
                          ) -> dict[str, object]:
    reg = dict(registration)
    required = {"schema", "start_time", "evaluator_commit", "parameters",
                "registration_sha256"}
    missing = required - set(reg)
    if missing:
        raise ValueError(f"envelope registration is missing {sorted(missing)}")
    if reg["schema"] != REGISTRATION_SCHEMA:
        raise ValueError("envelope registration schema mismatch")
    parameters = reg["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("envelope registration parameters must be an object")
    if canonical_hash(parameters) != reg["registration_sha256"]:
        raise ValueError("envelope registration hash does not match its pins")
    commit = str(reg["evaluator_commit"])
    if (len(commit) != 40
            or any(ch not in "0123456789abcdef" for ch in commit.lower())):
        raise ValueError("evaluator_commit must be a 40-hex Git SHA")
    return reg


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def discover_frames(run_directory: str | Path, domain: str, *,
                    start_time: str, seconds: Iterable[int]
                    ) -> dict[int, Path]:
    """Map each requested valid second to that domain's history frame."""
    root = Path(run_directory)
    start = datetime.fromisoformat(start_time)
    found: dict[int, Path] = {}
    for path in root.glob(f"wrfout_{domain}_*"):
        match = _FRAME_RE.match(path.name)
        if match is None or match.group(1) != domain:
            raise ValueError(f"invalid history-frame name: {path.name}")
        valid = datetime.strptime(
            f"{match.group(2)} {match.group(3)}:{match.group(4)}:"
            f"{match.group(5)}", "%Y-%m-%d %H:%M:%S")
        offset = int((valid - start).total_seconds())
        if offset in found:
            raise ValueError(f"duplicate {domain} frame at {offset} seconds")
        found[offset] = path
    wanted = sorted(set(seconds))
    absent = [second for second in wanted if second not in found]
    if absent:
        raise ValueError(
            f"{run_directory} is missing {domain} frames at {absent}")
    return {second: found[second] for second in wanted}


def score_pair(left_directory: str | Path, right_directory: str | Path,
               registration: Mapping[str, object]) -> dict[str, float]:
    """Distances between two runs, keyed ``metric|domain|lead``."""
    reg = validate_registration(registration)
    params = reg["parameters"]
    start_time = str(reg["start_time"])
    cadence = int(params["cadence_seconds"])
    leads = [int(lead) for lead in params["leads_seconds"]]
    refl = str(params["reflectivity_field"])
    require_spectral_agreement(refl)
    threshold = float(params["reflectivity_threshold"])
    scores: dict[str, float] = {}
    for domain, dx in sorted(params["domain_dx_m"].items()):
        dx = float(dx)
        needed = sorted({lead for lead in leads}
                        | {lead - cadence for lead in leads})
        left_frames = discover_frames(
            left_directory, domain, start_time=start_time, seconds=needed)
        right_frames = discover_frames(
            right_directory, domain, start_time=start_time, seconds=needed)
        filter_width = field_metrics.odd_width_cells(
            float(params["low_pass_physical_width_m"]), dx)
        min_cells = field_metrics.minimum_object_cells(
            float(params["object_min_area_km2"]), dx)
        for lead in leads:
            for field in params["state_fields"]:
                low_pass, boundary = field_metrics.state_and_boundary_rmse(
                    left_frames=left_frames, right_frames=right_frames,
                    field=str(field), score_times=(lead,),
                    cadence_seconds=cadence,
                    filter_width_cells=filter_width,
                    interior_exclusion_cells=int(
                        params["low_pass_interior_exclusion_cells"]),
                    boundary_width_cells=int(params["boundary_width_cells"]),
                    label=f"{domain}/{field} at {lead}s")
                scores[f"low_pass_state_rmse:{field}|{domain}|{lead}"] = low_pass
                scores[
                    f"applied_boundary_increment_rmse:{field}|{domain}|{lead}"
                ] = boundary
            window = [second for second in sorted(set(leads)) if second <= lead]
            left_refl = {
                second: field_metrics.read_frame_field(
                    left_frames[second], refl) for second in window}
            right_refl = {
                second: field_metrics.read_frame_field(
                    right_frames[second], refl) for second in window}
            quiet = lead + cadence
            left_time = field_metrics.first_object_time(
                left_refl, threshold=threshold, min_cells=min_cells,
                quiet_time_seconds=quiet,
                connectivity=int(params["object_connectivity"]))
            right_time = field_metrics.first_object_time(
                right_refl, threshold=threshold, min_cells=min_cells,
                quiet_time_seconds=quiet,
                connectivity=int(params["object_connectivity"]))
            scores[
                f"first_object_time_difference:{refl}|{domain}|{lead}"
            ] = abs(left_time - right_time)
            scores[f"fss_distance:{refl}|{domain}|{lead}"] = (
                field_metrics.mean_fss_distance(
                    left_refl, right_refl, threshold=threshold,
                    radius_m=float(params["fss_radius_m"]), dx_m=dx,
                    times=(lead,)))
            # The spectral class asks the one question the four above cannot:
            # whether the two runs hold their variance at the same scales.
            # The reflectivity plane is already in hand at this lead.
            left_planes = {refl: left_refl[lead]}
            right_planes = {refl: right_refl[lead]}
            for pin in spectral.FIELD_PINS:
                variable = str(pin["variable"])
                if variable in left_planes:
                    continue
                left_planes[variable] = field_metrics.read_frame_field(
                    left_frames[lead], variable)
                right_planes[variable] = field_metrics.read_frame_field(
                    right_frames[lead], variable)
            for variable, distance in spectral.pair_distances(
                    left_planes, right_planes, dx_m=dx).items():
                scores[
                    f"spectral_log_distance:{variable}|{domain}|{lead}"
                ] = distance
    if not scores or any(not math.isfinite(v) or v < 0.0
                         for v in scores.values()):
        raise ValueError("envelope scoring produced missing/invalid distances")
    return scores


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """``sorted[ceil(p*n)-1]`` -- a value some pair actually achieved."""
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(v) or v < 0.0 for v in ordered):
        raise ValueError("nearest-rank sample is empty, negative or non-finite")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must lie in (0, 100]")
    index = math.ceil(percentile / 100.0 * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def split_key(key: str) -> tuple[str, str, int]:
    """``metric:field|domain|lead`` -> (metric, domain, lead)."""
    metric, _, rest = key.partition(":")
    field, _, tail = rest.partition("|")
    domain, _, lead = tail.partition("|")
    if metric not in METRIC_CLASSES or not field or not domain or not lead:
        raise ValueError(f"malformed envelope metric key: {key!r}")
    return metric, domain, int(lead)


def envelope_rows(pair_distances: Mapping[str, Sequence[float]],
                  candidate_distances: Mapping[str, float], *,
                  percentile: float) -> list[dict[str, object]]:
    """One row per (metric, domain, lead), with its verdict."""
    if set(pair_distances) != set(candidate_distances):
        raise ValueError("member-pair and candidate metric inventories differ")
    if not pair_distances:
        raise ValueError("an envelope needs at least one scored metric")
    counts = {len(values) for values in pair_distances.values()}
    if len(counts) != 1:
        raise ValueError("every metric must carry the same pair count")
    rows: list[dict[str, object]] = []
    for key in sorted(pair_distances):
        metric, domain, lead = split_key(key)
        values = list(pair_distances[key])
        envelope = nearest_rank(values, percentile)
        distance = float(candidate_distances[key])
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f"candidate distance for {key!r} is invalid")
        degenerate = envelope == 0.0
        rows.append({
            "key": key, "metric": metric, "domain": domain, "lead_seconds": lead,
            "pair_count": len(values),
            "envelope": envelope,
            "envelope_degenerate": degenerate,
            "candidate_distance": distance,
            "verdict": ("degenerate-envelope" if degenerate
                        else ("within-envelope" if distance <= envelope
                              else "outside-envelope")),
        })
    return rows


def build_envelope(*, registration: Mapping[str, object],
                   member_directories: Sequence[str | Path],
                   candidate_directory: str | Path,
                   unperturbed_directory: str | Path,
                   member_identity: Mapping[str, object],
                   output: str | Path | None = None) -> dict[str, object]:
    """Score every unordered member pair and the candidate, and reduce."""
    reg = validate_registration(registration)
    members = [Path(directory).resolve() for directory in member_directories]
    if len(members) < 2:
        raise ValueError("an envelope needs at least two members to form a pair")
    pair_distances: dict[str, list[float]] = {}
    pairs = 0
    for index, left in enumerate(members):
        for right in members[index + 1:]:
            scores = score_pair(left, right, reg)
            if not pair_distances:
                pair_distances = {key: [] for key in scores}
            if set(scores) != set(pair_distances):
                raise ValueError("member-pair metric inventories differ")
            for key, value in scores.items():
                pair_distances[key].append(value)
            pairs += 1
    candidate = score_pair(candidate_directory, unperturbed_directory, reg)
    percentile = float(reg["parameters"]["envelope_percentile"])
    rows = envelope_rows(pair_distances, candidate, percentile=percentile)
    receipt = {
        "schema": ENVELOPE_SCHEMA,
        "registration": reg,
        "registration_sha256": reg["registration_sha256"],
        "evaluator_commit": reg["evaluator_commit"],
        "member_identity": json.loads(json.dumps(member_identity)),
        "member_count": len(members),
        "pair_count": pairs,
        "members": [directory.name for directory in members],
        "candidate": Path(candidate_directory).resolve().name,
        "unperturbed": Path(unperturbed_directory).resolve().name,
        "metric_classes": list(METRIC_CLASSES),
        "metric_classes_unscheduled": [
            dict(entry) for entry in UNSCHEDULED_METRIC_CLASSES],
        "spectral_pins_sha256": spectral.PINS_SHA256,
        "rows": rows,
        "degenerate_rows": sum(1 for row in rows if row["envelope_degenerate"]),
        "outside_rows": sum(
            1 for row in rows if row["verdict"] == "outside-envelope"),
    }
    if output is not None:
        write_json(output, receipt)
    return receipt


# --------------------------------------------------------------------------
# the envelope belongs to the run it is cited beside
# --------------------------------------------------------------------------


def config_identity(config_path: str | Path) -> dict[str, object]:
    """Member geometry and physics identity, through the production loader."""
    from gpuwm.case_data import load_experiment_case

    experiment, _case = load_experiment_case(str(config_path))
    domains = []
    for domain in experiment.domains:
        run = domain.run
        domains.append({
            "grid_id": int(domain.grid_id),
            "parent_id": int(domain.parent_id),
            "i_parent_start": int(domain.i_parent_start),
            "j_parent_start": int(domain.j_parent_start),
            "parent_grid_ratio": int(domain.parent_grid_ratio),
            "parent_time_step_ratio": int(domain.parent_time_step_ratio),
            "nx": int(run.nx), "ny": int(run.ny), "nz": int(run.nz),
            "dx_m": float(run.dx), "dy_m": float(run.dy),
            "mp_physics": int(run.mp_physics),
            "bl_pbl_physics": int(run.bl_pbl_physics),
            "sf_sfclay_physics": int(run.sf_sfclay_physics),
            "sf_surface_physics": int(run.sf_surface_physics),
            "cu_physics": int(run.cu_physics),
            "ra_physics": int(run.ra_physics),
            "ra_rrtmg_variant": str(run.ra_rrtmg_variant),
            "wrf_rrtmg_compatibility": str(run.wrf_rrtmg_compatibility),
        })
    return {"domains": domains}


def _spacing_tokens(identity: Mapping[str, object]) -> list[str]:
    """How each domain's spacing reads in prose, derived from the numbers."""
    tokens = []
    for domain in identity["domains"]:
        dx = float(domain["dx_m"])
        if dx >= 1000.0 and abs(dx / 1000.0 - round(dx / 1000.0)) < 1e-9:
            tokens.append(f"{int(round(dx / 1000.0))} km")
        elif abs(dx - round(dx)) < 1e-9:
            tokens.append(f"{int(round(dx))} m")
        else:
            tokens.append(f"{dx:g} m")
    return tokens


def check_config_identity(receipt: Mapping[str, object],
                          config_path: str | Path,
                          document_path: str | Path | None = None
                          ) -> list[str]:
    """Issues that disqualify this receipt from being cited beside that run."""
    issues: list[str] = []
    recorded = receipt.get("member_identity")
    if not isinstance(recorded, dict) or not recorded.get("domains"):
        return ["receipt records no member geometry/physics identity"]
    expected = config_identity(config_path)
    if len(recorded["domains"]) != len(expected["domains"]):
        issues.append(
            f"receipt covers {len(recorded['domains'])} domains, "
            f"{Path(config_path).name} declares {len(expected['domains'])}")
        return issues
    for got, want in zip(recorded["domains"], expected["domains"]):
        for key in sorted(want):
            if key not in got:
                issues.append(f"domain {want['grid_id']}: {key} absent from receipt")
            elif got[key] != want[key]:
                issues.append(
                    f"domain {want['grid_id']}: {key} {got[key]!r} != "
                    f"{want[key]!r}")
    if document_path is not None:
        text = Path(document_path).read_text(encoding="utf-8")
        collapsed = " ".join(text.split())
        for token in _spacing_tokens(expected):
            if token not in collapsed:
                issues.append(
                    f"{Path(document_path).name} states no {token} domain")
    return issues


# --------------------------------------------------------------------------
# reducer admission
# --------------------------------------------------------------------------


def admit_members(members: Sequence[Mapping[str, object]], *,
                  required_input_sha256: Mapping[str, str]
                  ) -> list[dict[str, object]]:
    """Admit only members built from the declared initialization byte set.

    Provenance is enforced here rather than trusted: a member whose consumed
    input hashes are not the preserved set was built from a different base
    state, and the offset between those base states would be absorbed into --
    and would inflate -- the very spread this envelope measures.
    """
    if not required_input_sha256:
        raise ValueError("the admission gate needs a declared input SHA set")
    admitted: list[dict[str, object]] = []
    refused: list[str] = []
    for member in members:
        name = str(member.get("id", "<unnamed>"))
        declared = member.get("input_sha256")
        if not isinstance(declared, dict) or not declared:
            refused.append(f"{name}: records no consumed-input hashes")
            continue
        if dict(declared) != dict(required_input_sha256):
            differing = sorted(
                set(declared) ^ set(required_input_sha256)
                | {key for key in set(declared) & set(required_input_sha256)
                   if declared[key] != required_input_sha256[key]})
            refused.append(f"{name}: input hashes differ at {differing}")
            continue
        admitted.append(dict(member))
    if refused:
        raise ValueError(
            "ensemble members refused by the input-SHA admission gate: "
            + "; ".join(refused))
    if not admitted:
        raise ValueError("no member survived the admission gate")
    return admitted


def reduce_band(receipt: Mapping[str, object], *,
                members: Sequence[Mapping[str, object]],
                pair_score_sha256: str) -> dict[str, object]:
    """The ensemble block an acceptance band carries when it comes from here."""
    rows = receipt["rows"]
    return {
        "band_schema_version": 1,
        "provenance": "wrf-ensemble-envelope",
        "scope": "internal",
        "scope_note": (
            "member artifacts are not redistributable; this block records "
            "digests, never a retrieval path"),
        "registration_sha256": receipt["registration_sha256"],
        "evaluator_commit": receipt["evaluator_commit"],
        "interval_statistic": receipt["registration"]["parameters"][
            "envelope_statistic"],
        "member_count": len(members),
        "member_config_digest": {
            str(member["id"]): str(member["config_digest"])
            for member in members},
        "pair_score_sha256": pair_score_sha256,
        "rows": [
            {"key": row["key"], "envelope": row["envelope"],
             "envelope_degenerate": row["envelope_degenerate"]}
            for row in rows],
    }


__all__ = [
    "ENVELOPE_SCHEMA", "METRIC_CLASSES", "REGISTRATION_SCHEMA",
    "UNSCHEDULED_METRIC_CLASSES", "admit_members", "build_envelope",
    "canonical_hash", "check_config_identity", "config_identity",
    "discover_frames", "envelope_rows", "make_registration", "nearest_rank",
    "reduce_band", "require_spectral_agreement", "score_pair", "sha256_file",
    "split_key", "validate_registration", "write_json",
]
