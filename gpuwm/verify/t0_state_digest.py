"""Full-state t=0 parity digest over a matched candidate/reference pair.

The published matched-run comparator scores eight surface-and-column
diagnostics per frame.  That is enough to describe a forecast, and it is
not enough to say the initial states agree: a t=0 claim built on it infers
the whole prognostic state from a handful of carriers.  This module scores
the state itself -- every registered carrier group that both frames carry,
on every domain that both directories carry, plus the lateral-boundary
tables when both sides retained them.

Three properties are structural rather than incidental:

* **The ceilings are not this module's to choose.**  ``max``, ``p99`` and
  ``mean signed`` all come from :mod:`gpuwm.verify.nest_gates`, imported by
  name.  Those constants were pinned before any frame scored here existed,
  so a receipt cannot calibrate the gate it is judged by, and this module
  holds no numeric ceiling of its own for anyone to nudge.

* **The ULP is the comparator's.**  Every number in the receipt comes from
  :func:`gpuwm.verify.state_equiv.fp32_signed_ulp`, the one certified
  definition, so the digest and the N2 peer-state gate cannot disagree
  about what a ULP is.

* **Nothing about the case is compiled in.**  Domains, valid times,
  grid spacings and boundary-table names are all discovered from the
  staged directories.  The registration fixes which *variables* belong to
  which carrier group -- physics, not case -- and a two-domain pair scores
  exactly as a four-domain pair does.

The registration is a pre-registration: :func:`make_t0_registration`
returns the whole scoring contract, :func:`registration_sha256` hashes it,
and the receipt embeds both, so a later reader can prove the contract was
not edited to suit the numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gpuwm.verify.nest_gates import (
    FP32_STATE_EQUIV_MAX_ULPS,
    FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS,
    FP32_STATE_EQUIV_P99_MAX_ULPS,
)
from gpuwm.verify.state_equiv import fp32_signed_ulp, fp32_state_equiv_report

SCHEMA_ID = "gpuwm.t0-state-digest/v1"

#: History-tape frame name written by every ARW-convention writer:
#: ``wrfout_d<NN>_<valid time>``.  The domain number is read from the name,
#: never assumed -- a two-domain pair and a four-domain pair take the same
#: path through this module.
FRAME_RE = re.compile(
    r"^wrfout_d(?P<domain>\d{2})_"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}[_:]\d{2}[_:]\d{2}[_:]\d{2})$")

#: Lateral-boundary file for whichever domain carries the outer boundary.
BOUNDARY_FILE_RE = re.compile(r"^wrfbdy_d(?P<domain>\d{2})$")

#: A boundary table: one side (``XS``/``XE``/``YS``/``YE``) of one
#: prognostic, either the value table ``_B*`` or the tendency table
#: ``_BT*``.  Discovered by shape, so a file carrying more prognostics than
#: this module has heard of still scores all of them.
BOUNDARY_VAR_RE = re.compile(r"^(?P<field>[A-Z0-9_]+?)_B(?P<tendency>T?)"
                             r"(?P<side>XS|XE|YS|YE)$")


@dataclass(frozen=True)
class CarrierGroup:
    """One registered group of state carriers.

    ``required`` says the group must score at least one array somewhere, or
    the digest is not a full-state digest and the run exits non-zero.  It
    does NOT say every variable is present: a frame written without a
    scheme's carriers simply scores the ones it has.
    """

    name: str
    required: bool
    description: str
    variables: tuple[str, ...]


#: The registered carriers.  Grouping is by physical role, so a claim about
#: one group is a claim about a nameable part of the state rather than
#: about a list of variable spellings.  Membership is deliberately wider
#: than any one configuration writes; presence in BOTH frames is what makes
#: a variable scored.
CARRIER_GROUPS: tuple[CarrierGroup, ...] = (
    CarrierGroup(
        name="dry_dynamics",
        required=True,
        description=(
            "prognostic wind, geopotential, perturbation potential "
            "temperature and the dry-air mass column, with their base "
            "states"),
        variables=(
            "U", "V", "W", "PH", "PHB", "T", "T_INIT",
            "MU", "MUB", "P", "PB", "ALB", "AL", "P_HYD",
        ),
    ),
    CarrierGroup(
        name="moisture",
        required=True,
        description="water-substance mixing ratios and number concentrations",
        variables=(
            "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL",
            "QNCLOUD", "QNICE", "QNRAIN", "QNSNOW", "QNGRAUPEL", "QNHAIL",
            "QNWFA", "QNIFA", "QNBCA", "QVOLG", "QVOLH",
        ),
    ),
    CarrierGroup(
        name="surface",
        required=True,
        description=(
            "near-surface diagnostics, skin state and the static surface "
            "characterization"),
        variables=(
            "T2", "Q2", "TH2", "PSFC", "U10", "V10", "TSK", "SST",
            "HGT", "XLAND", "LU_INDEX", "IVGTYP", "ISLTYP", "VEGFRA",
            "LAI", "ALBEDO", "ALBBCK", "EMISS", "ZNT", "UST", "MOL",
            "XICE", "SNOWC", "F", "E", "SINALPHA", "COSALPHA",
            "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
            "MAPFAC_M", "MAPFAC_U", "MAPFAC_V",
        ),
    ),
    CarrierGroup(
        name="soil",
        required=True,
        description="soil column, snowpack and canopy storage",
        variables=(
            "TSLB", "SMOIS", "SH2O", "SMCREL", "ZS", "DZS", "TMN",
            "SNOW", "SNOWH", "SNOWALB", "CANWAT", "SEAICE", "LANDMASK",
        ),
    ),
    CarrierGroup(
        name="accumulation",
        required=False,
        description="precipitation and runoff accumulators",
        variables=(
            "RAINC", "RAINNC", "RAINSH", "SNOWNC", "GRAUPELNC", "HAILNC",
            "ACSNOW", "ACSNOM", "SFROFF", "UDROFF", "I_RAINC", "I_RAINNC",
        ),
    ),
    CarrierGroup(
        name="diagnostic",
        required=False,
        description="radiative, flux and boundary-layer diagnostics",
        variables=(
            "REFL_10CM", "SWDOWN", "GLW", "OLR", "SWDNB", "SWUPB",
            "LWDNB", "LWUPB", "HFX", "LH", "QFX", "GRDFLX", "PBLH",
            "CLDFRA", "EXCH_H", "TKE_PBL",
        ),
    ),
)

#: The lateral-boundary group is registered separately because its members
#: are discovered from the boundary file rather than enumerated, and
#: because its presence is a property of what was staged (F1 AC-4): the
#: receipt records ``scored`` or ``unavailable`` and never assumes.
BOUNDARY_GROUP = "boundary"

_STATUS_SCORED = "scored"
_STATUS_UNAVAILABLE = "unavailable"

_EVALUATOR_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def registered_variables() -> frozenset[str]:
    """Every variable spelling any registered group claims."""
    return frozenset(
        name for group in CARRIER_GROUPS for name in group.variables)


def make_t0_registration() -> dict[str, object]:
    """The scoring contract, pinned before any frame is opened.

    Everything that decides a verdict is in here: which variables belong to
    which group, which groups a full-state digest must cover, the
    comparator that measures, and the ceilings it measures against --
    quoted from the module that owns them, so the registration hash moves
    if anybody moves a ceiling.
    """
    return {
        "schema": SCHEMA_ID,
        "comparator": {
            "module": "gpuwm.verify.state_equiv",
            "ulp_function": "fp32_signed_ulp",
            "report_function": "fp32_state_equiv_report",
            "kind": "fp32_state_equiv",
        },
        "ceilings": {
            "source": "gpuwm.verify.nest_gates",
            "max_ulp": FP32_STATE_EQUIV_MAX_ULPS,
            "p99_ulp": FP32_STATE_EQUIV_P99_MAX_ULPS,
            "mean_signed_ulp_max_abs":
                FP32_STATE_EQUIV_MEAN_SIGNED_ULP_MAX_ABS,
        },
        "carrier_groups": {
            group.name: {
                "required": group.required,
                "description": group.description,
                "variables": list(group.variables),
            }
            for group in CARRIER_GROUPS
        },
        "boundary_group": {
            "name": BOUNDARY_GROUP,
            "required": False,
            "description": (
                "lateral-boundary value and tendency tables, discovered "
                "from the staged boundary file"),
            "discovery_pattern": BOUNDARY_VAR_RE.pattern,
            "status_values": [_STATUS_SCORED, _STATUS_UNAVAILABLE],
        },
        "frame_pattern": FRAME_RE.pattern,
        "boundary_file_pattern": BOUNDARY_FILE_RE.pattern,
    }


def canonical_json(document: object) -> str:
    """Stable JSON text: the receipt is a function of its inputs alone."""
    return json.dumps(document, sort_keys=True, indent=2,
                      separators=(",", ": "), ensure_ascii=True)


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def registration_sha256(registration: Mapping[str, object] | None = None
                        ) -> str:
    """SHA-256 over the canonical encoding of the registration."""
    if registration is None:
        registration = make_t0_registration()
    return _stable_hash(registration)


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_evaluator_commit(root: Path | None = None) -> str | None:
    """The 40-hex commit of the tree that produced a receipt, or None.

    None is a refusal, not a default: a receipt whose evaluator cannot be
    named is not evidence, so the caller fails closed rather than writing
    a placeholder.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root is not None else None,
            capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    candidate = result.stdout.strip()
    return candidate if _EVALUATOR_COMMIT_RE.match(candidate) else None


# ---------------------------------------------------------------------------
# staging discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePair:
    domain: str
    valid_time: str
    candidate: Path
    reference: Path


def _index_frames(directory: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    if not directory.is_dir():
        return index
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_RE.match(path.name)
        if match is None:
            continue
        key = (f"d{match.group('domain')}",
               match.group("stamp").replace(":", "_"))
        index.setdefault(key, path)
    return index


def discover_frame_pairs(candidate_dir: Path, reference_dir: Path, *,
                         valid_time: str | None = None) -> list[FramePair]:
    """Earliest common frame per domain, or the named valid time.

    Domains come out of the directory listing.  Nothing here knows how many
    there are, how they nest, or what grid spacing any of them has.
    """
    candidate_index = _index_frames(candidate_dir)
    reference_index = _index_frames(reference_dir)
    shared = sorted(set(candidate_index) & set(reference_index))
    by_domain: dict[str, tuple[str, Path, Path]] = {}
    for domain, stamp in shared:
        if valid_time is not None and stamp != valid_time.replace(":", "_"):
            continue
        current = by_domain.get(domain)
        if current is not None and current[0] <= stamp:
            continue
        by_domain[domain] = (stamp,
                             candidate_index[(domain, stamp)],
                             reference_index[(domain, stamp)])
    return [FramePair(domain, stamp, candidate, reference)
            for domain, (stamp, candidate, reference)
            in sorted(by_domain.items())]


def discover_boundary_files(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = BOUNDARY_FILE_RE.match(path.name)
        if match is not None:
            found[f"d{match.group('domain')}"] = path
    return found


# ---------------------------------------------------------------------------
# reading and scoring
# ---------------------------------------------------------------------------


def _open_dataset(path: Path):
    # Decoded in Rust like every other met NetCDF read.  This module was
    # exempted as "bytes" -- needing the stored representation, which the
    # bridge cannot give because it promotes every numeric type to f64 --
    # but that reason does not survive reading the code: `_first_record`
    # casts to FP32 itself and never inspects a dtype or takes a
    # `.tobytes()`, so the stored width is discarded here anyway.  It also
    # reads only numeric variables (CARRIER_GROUPS enumerates them, and
    # BOUNDARY_VAR_RE is uppercase-only, which excludes wrfbdy's lowercase
    # `md___*` character metadata), and it takes its valid times from the
    # filename rather than from `Times`.
    from gpuwm import netcdf_bridge  # noqa: PLC0415

    return netcdf_bridge.open_dataset(path)


def _first_record(variable) -> np.ndarray:
    """The leading record of a variable, as host FP32.

    Auto-masking is switched off deliberately: a fill value replaced by a
    mask would score as an equal element on both sides no matter what the
    file holds, which is the one thing a parity digest must not do.
    """
    variable.set_auto_mask(False)
    if variable.dimensions and variable.dimensions[0] == "Time":
        array = variable[0]
    else:
        array = variable[:]
    return np.ascontiguousarray(np.asarray(array), dtype=np.float32)


def read_frame(path: Path, wanted: Iterable[str]
               ) -> dict[str, np.ndarray]:
    """Read the named variables, first record only, as host FP32."""
    names = set(wanted)
    out: dict[str, np.ndarray] = {}
    with _open_dataset(path) as dataset:
        for name in sorted(names):
            variable = dataset.variables.get(name)
            if variable is None:
                continue
            out[name] = _first_record(variable)
    return out


def read_boundary_tables(path: Path) -> dict[str, np.ndarray]:
    """Every ``_B*``/``_BT*`` side table the file carries."""
    out: dict[str, np.ndarray] = {}
    with _open_dataset(path) as dataset:
        for name in sorted(dataset.variables):
            if BOUNDARY_VAR_RE.match(name) is None:
                continue
            out[name] = _first_record(dataset.variables[name])
    return out


def _max_abs_diff(candidate: np.ndarray, reference: np.ndarray) -> float:
    delta = np.abs(candidate.astype(np.float64)
                   - reference.astype(np.float64))
    return float(np.max(delta)) if delta.size else 0.0


def score_variables(candidate: Mapping[str, np.ndarray],
                    reference: Mapping[str, np.ndarray],
                    names: Sequence[str]) -> dict[str, object]:
    """Per-variable metrics plus the comparator's all-element verdict.

    The verdict comes from ``fp32_state_equiv_report`` over the scored
    variables concatenated -- the registered reduction -- so this module
    adds no second, differently-shaped gate.
    """
    scored = [name for name in names
              if name in candidate and name in reference]
    unscored = {
        name: ("absent from candidate" if name not in candidate else
               "absent from reference")
        for name in names if name not in scored
    }
    if not scored:
        return {
            "status": _STATUS_UNAVAILABLE,
            "scored_arrays": 0,
            "variables": {},
            "unscored": unscored,
            "verdict": None,
            "aggregate": None,
        }

    report = fp32_state_equiv_report(candidate, reference, scored)
    variables: dict[str, object] = {}
    for name in scored:
        metrics = dict(report["fields"][name])
        candidate_array = np.asarray(candidate[name], dtype=np.float32)
        reference_array = np.asarray(reference[name], dtype=np.float32)
        if candidate_array.shape == reference_array.shape:
            metrics["shape"] = list(candidate_array.shape)
            metrics["max_abs_diff"] = _max_abs_diff(
                candidate_array, reference_array)
            metrics["signed_ulp_extrema"] = _signed_extrema(
                candidate_array, reference_array)
        else:
            metrics["shape"] = None
            metrics["max_abs_diff"] = None
            metrics["signed_ulp_extrema"] = None
        variables[name] = metrics
    return {
        "status": _STATUS_SCORED,
        "scored_arrays": len(scored),
        "variables": variables,
        "unscored": unscored,
        "verdict": "PASS" if report["pass"] else "FAIL",
        "aggregate": report["aggregate"],
    }


def _signed_extrema(candidate: np.ndarray, reference: np.ndarray
                    ) -> list[int] | None:
    try:
        signed = fp32_signed_ulp(candidate, reference)
    except ValueError:
        return None
    if signed.size == 0:
        return None
    return [int(np.min(signed)), int(np.max(signed))]


# ---------------------------------------------------------------------------
# the receipt
# ---------------------------------------------------------------------------


def _input_record(side: str, role: str, path: Path, domain: str | None,
                  valid_time: str | None) -> dict[str, object]:
    return {
        "side": side,
        "role": role,
        "domain": domain,
        "valid_time": valid_time,
        "name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_boundary_entry(candidate_dir: Path, reference_dir: Path
                         ) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Score the lateral-boundary tables, or record why they were not.

    The probe is the point: the public tree does not prove either side
    retained a boundary file, so the receipt says which side was missing
    and the group stays out of ``covered_groups`` rather than being
    quietly claimed.
    """
    candidate_files = discover_boundary_files(candidate_dir)
    reference_files = discover_boundary_files(reference_dir)
    shared = sorted(set(candidate_files) & set(reference_files))
    if not shared:
        absent = []
        if not candidate_files:
            absent.append("candidate")
        if not reference_files:
            absent.append("reference")
        if absent:
            reason = ("no lateral-boundary file staged on the "
                      f"{' and '.join(absent)} side")
        else:
            reason = ("both sides staged a lateral-boundary file but for no "
                      f"common domain: candidate has {sorted(candidate_files)}, "
                      f"reference has {sorted(reference_files)}")
        return (
            {
                "status": _STATUS_UNAVAILABLE,
                "absent_sides": absent,
                "reason": reason,
                "domains": {},
            },
            [],
        )

    inputs: list[dict[str, object]] = []
    domains: dict[str, object] = {}
    for domain in shared:
        candidate_path = candidate_files[domain]
        reference_path = reference_files[domain]
        inputs.append(_input_record("candidate", "wrfbdy", candidate_path,
                                    domain, None))
        inputs.append(_input_record("reference", "wrfbdy", reference_path,
                                    domain, None))
        candidate_tables = read_boundary_tables(candidate_path)
        reference_tables = read_boundary_tables(reference_path)
        names = sorted(set(candidate_tables) | set(reference_tables))
        scored = score_variables(candidate_tables, reference_tables, names)
        scored["value_tables"] = sorted(
            name for name in scored["variables"]
            if not BOUNDARY_VAR_RE.match(name).group("tendency"))
        scored["tendency_tables"] = sorted(
            name for name in scored["variables"]
            if BOUNDARY_VAR_RE.match(name).group("tendency"))
        domains[domain] = scored

    scored_arrays = sum(int(entry["scored_arrays"])
                        for entry in domains.values())
    verdicts = [entry["verdict"] for entry in domains.values()
                if entry["verdict"] is not None]
    if scored_arrays == 0:
        return (
            {
                "status": _STATUS_UNAVAILABLE,
                "absent_sides": [],
                "reason": ("boundary files are staged on both sides but "
                           "share no side table"),
                "domains": domains,
            },
            inputs,
        )
    return (
        {
            "status": _STATUS_SCORED,
            "absent_sides": [],
            "reason": None,
            "scored_arrays": scored_arrays,
            "verdict": "FAIL" if "FAIL" in verdicts else "PASS",
            "domains": domains,
        },
        inputs,
    )


def build_t0_receipt(candidate_dir: Path, reference_dir: Path, *,
                     evaluator_commit: str,
                     valid_time: str | None = None) -> dict[str, object]:
    """Score the staged pair and return the receipt document."""
    if not _EVALUATOR_COMMIT_RE.match(str(evaluator_commit)):
        raise ValueError(
            "evaluator_commit must be a 40-hex commit id; a receipt that "
            f"cannot name its evaluator is not evidence (got "
            f"{evaluator_commit!r})")

    candidate_dir = Path(candidate_dir)
    reference_dir = Path(reference_dir)
    registration = make_t0_registration()
    wanted = registered_variables()

    inputs: list[dict[str, object]] = []
    domains: dict[str, object] = {}
    for pair in discover_frame_pairs(candidate_dir, reference_dir,
                                     valid_time=valid_time):
        inputs.append(_input_record("candidate", "wrfout", pair.candidate,
                                    pair.domain, pair.valid_time))
        inputs.append(_input_record("reference", "wrfout", pair.reference,
                                    pair.domain, pair.valid_time))
        candidate_fields = read_frame(pair.candidate, wanted)
        reference_fields = read_frame(pair.reference, wanted)
        groups: dict[str, object] = {}
        for group in CARRIER_GROUPS:
            entry = score_variables(candidate_fields, reference_fields,
                                    group.variables)
            entry["required"] = group.required
            groups[group.name] = entry
        domains[pair.domain] = {
            "valid_time": pair.valid_time,
            "scored_carriers": sum(int(entry["scored_arrays"])
                                   for entry in groups.values()),
            "groups": groups,
        }

    boundary, boundary_inputs = build_boundary_entry(
        candidate_dir, reference_dir)
    inputs.extend(boundary_inputs)

    covered = sorted(
        group.name for group in CARRIER_GROUPS
        if any(domain["groups"][group.name]["scored_arrays"]
               for domain in domains.values()))
    if boundary["status"] == _STATUS_SCORED:
        covered.append(BOUNDARY_GROUP)
        covered.sort()

    unmet = [group.name for group in CARRIER_GROUPS
             if group.required and group.name not in covered]
    verdicts = [
        domain["groups"][name]["verdict"]
        for domain in domains.values() for name in domain["groups"]
        if domain["groups"][name]["verdict"] is not None
    ]
    if boundary["status"] == _STATUS_SCORED:
        verdicts.append(boundary["verdict"])
    if not domains:
        verdict = "FAIL"
    elif "FAIL" in verdicts:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    return {
        "schema": SCHEMA_ID,
        "evaluator_commit": evaluator_commit,
        "registration": registration,
        "registration_sha256": registration_sha256(registration),
        "inputs": inputs,
        "domains": domains,
        "boundary": boundary,
        "covered_groups": covered,
        "uncovered_required_groups": unmet,
        "records": [[domain, name]
                    for domain in sorted(domains)
                    for name in sorted(
                        _domain_variables(domains[domain]))],
        "verdict": verdict,
    }


def _domain_variables(domain_entry: Mapping[str, object]) -> set[str]:
    names: set[str] = set()
    for group in domain_entry["groups"].values():
        names.update(group["variables"])
    return names


def receipt_records(receipt: Mapping[str, object]) -> set[tuple[str, str]]:
    """The ``(domain, variable)`` record set the receipt scored."""
    return {(str(domain), str(name)) for domain, name in receipt["records"]}


def receipt_variables(receipt: Mapping[str, object]) -> set[str]:
    """Every variable the receipt scored on at least one domain."""
    return {name for _domain, name in receipt_records(receipt)}


def render_markdown(receipt: Mapping[str, object]) -> str:
    """A reviewer-facing table of the same numbers the receipt carries."""
    ceilings = receipt["registration"]["ceilings"]
    lines = [
        "# t=0 full-state parity digest",
        "",
        f"- schema: `{receipt['schema']}`",
        f"- evaluator commit: `{receipt['evaluator_commit']}`",
        f"- registration sha256: `{receipt['registration_sha256']}`",
        f"- ceilings (from `{ceilings['source']}`): max "
        f"{ceilings['max_ulp']} ULP, p99 {ceilings['p99_ulp']} ULP, "
        f"|mean signed| {ceilings['mean_signed_ulp_max_abs']} ULP",
        f"- covered groups: {', '.join(receipt['covered_groups']) or 'none'}",
        f"- verdict: **{receipt['verdict']}**",
        "",
        "| domain | group | scored arrays | max ULP | p99 ULP | "
        "mean signed ULP | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for domain in sorted(receipt["domains"]):
        entry = receipt["domains"][domain]
        for name in sorted(entry["groups"]):
            group = entry["groups"][name]
            aggregate = group["aggregate"] or {}
            mean = aggregate.get("mean_signed_ulp")
            lines.append(
                f"| {domain} | {name} | {group['scored_arrays']} | "
                f"{_cell(aggregate.get('max_ulp'))} | "
                f"{_cell(aggregate.get('p99_ulp'))} | "
                f"{'-' if mean is None else f'{mean:+.4f}'} | "
                f"{group['verdict'] or group['status']} |")
    boundary = receipt["boundary"]
    lines.extend([
        "",
        f"## boundary group: {boundary['status']}",
        "",
    ])
    if boundary["status"] == _STATUS_UNAVAILABLE:
        lines.append(f"Not scored: {boundary['reason']}.")
    else:
        for domain in sorted(boundary["domains"]):
            group = boundary["domains"][domain]
            lines.append(
                f"- {domain}: {group['scored_arrays']} side tables "
                f"({len(group['value_tables'])} value, "
                f"{len(group['tendency_tables'])} tendency), "
                f"verdict {group['verdict']}")
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    return "-" if value is None else str(value)


__all__ = [
    "BOUNDARY_GROUP",
    "CARRIER_GROUPS",
    "SCHEMA_ID",
    "CarrierGroup",
    "build_boundary_entry",
    "build_t0_receipt",
    "canonical_json",
    "discover_boundary_files",
    "discover_frame_pairs",
    "make_t0_registration",
    "read_boundary_tables",
    "read_frame",
    "receipt_records",
    "receipt_variables",
    "registered_variables",
    "registration_sha256",
    "render_markdown",
    "resolve_evaluator_commit",
    "score_variables",
    "sha256_file",
]
