"""Executable real74 N3 rung and shared real-chain verification machinery.

This module is deliberately a controller surface.  Importing it is CPU-only;
device work starts only in :func:`execute_production_run`, which uses the same
``build_experiment``/``execute_experiment`` path as ``gpuwm run``.  Every gate
record is obtained through :func:`gpuwm.verify.nest_gates.gate`; numeric policy
is never duplicated here.

The d01 Phase-4 ratchet follows the controller ledger attribution recorded on
2026-07-16: every output variable other than ``REFL_10CM`` is byte-identical,
while ``GPUWM_WRITE_COMPLETE`` is the ratified publication-attribute delta.
Those two names are the complete exception inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from gpuwm.case_data import load_experiment_case
from gpuwm.ingest.nest_init import blend_zone_mask
from gpuwm.verify import metrics as weather_metrics
from gpuwm.verify import nest_gates


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_CONFIG = REPOSITORY_ROOT / "configs" / "real74_4dom.toml"
RUN_SECONDS = 75 * 60
RESTART_SPLIT_SECONDS = 30 * 60

CANONICAL_STATE_SCHEMA = "gpuwm-canonical-state-v2"
CANONICAL_INVENTORY_SCHEMA = "gpuwm-canonical-state-inventory-v2"

# Ratchet state evidence is a role contract, not a caller-selected subset of
# the run provenance.  In particular, N3 carries the d01 no-child control
# consumed by N5 in addition to its d02 artifact-bound control.  MappingProxy
# plus tuple values keeps both the role inventory and each domain inventory
# immutable in process; the same exact inventory is serialized in manifests.
RATCHET_STATE_EVIDENCE_DOMAINS = MappingProxyType({
    "N3": ("d01", "d02"),
    "N4": ("d03",),
})

# A history-frame inventory may grow only through these audited lazy model
# surfaces.  The patterns are intentionally narrower than the restart
# module's general REBUILT scratch prefixes: adding an unrelated work array
# must not silently widen the ancestor/restart ratchets.
CANONICAL_LAZY_MEMBER_CLASSES = (
    "nest rolling value/tendency, donor, and SINT tables",
    "lateral-boundary relaxation weights",
    "REFL_10CM frame stash",
    "microphysics carrying accumulators",
    "cumulus carrying accumulators",
    "KF W0AVG trigger history",
)

_CANONICAL_NEST_FIELD_KINDS = (
    "u", "v", "w", "t", "ph", "mu",
    "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni", "ns", "ng",
    "qh", "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh",
)
_CANONICAL_NEST_SLOTS = frozenset({
    *(f"nest_{kind}_{prefix}{side}"
      for kind in _CANONICAL_NEST_FIELD_KINDS
      for prefix in ("b", "bt")
      for side in ("xs", "xe", "ys", "ye")),
    "nest_parent_field",
    "nest_child_field",
    *(f"nest_sint_{component}_{stagger}"
      for component in ("ci", "ip", "cj", "jp", "xig", "xjg")
      for stagger in ("m", "x", "y")),
})
_CANONICAL_LBC_SLOTS = frozenset({"lbc_weights_0"})
_CANONICAL_REFL_SLOTS = frozenset({"refl_10cm"})
_CANONICAL_EXTRA_PREFIXES = ("nest_", "lbc_weights_", "refl_")
_CANONICAL_EXCLUDED_FRAME_SCRATCH = frozenset({"refl_t"})

# The controller ledger's ratified exceptions.  GPUWM_WRITE_COMPLETE and
# TITLE are global attributes rather than NetCDF variables; keeping one tuple
# makes the attribution inventory machine-pinnable and reviewable.  TITLE was
# added by the F20 amendment (2026-07-17): the Phase-5 four-domain case labels
# its outputs truthfully while the frozen Phase-3/4 baseline carries the
# original case title -- descriptive metadata, not state; both values are
# recorded in the evidence and the candidate title must still be a non-empty
# string.
D01_PHASE4_RATIFIED_EXCEPTIONS = (
    "REFL_10CM",
    "GPUWM_WRITE_COMPLETE",
    "TITLE",
)

N3_METRICS = (
    "d01_bitwise_vs_phase4_13z",
    "d02_mslp_pattern_correlation",
    "d02_t500_rmse_k",
    "d02_t850_rmse_k",
    "d02_boundary_zone_blowup",
    "d02_refl_10cm_structure",
    "d02_refl_10cm_fss",
    "d02_hgt_blend_recheck_m",
    "d02_mub_blend_recheck_pa",
    "d02_blend_zone_t2_tsk_bias",
    "restart_split_bit_identity",
    "two_domain_alloc_check",
)

def _registered_gate(milestone: str, metric: str):
    """The sole gate-record lookup used by rung evaluators."""
    return nest_gates.gate(milestone, metric)


def gate_records(milestone: str, metrics: Iterable[str]):
    """Resolve records one-by-one through ``gate()`` in caller order."""
    return tuple(_registered_gate(milestone, metric) for metric in metrics)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path, *, block_bytes: int = 8 * 1024 * 1024
                ) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: object) -> object:
    """Return strict-JSON evidence, mapping non-finite values to null."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Atomically publish a deterministic machine-readable report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temp, path)
    return path


def load_verdicts(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("structural-verdict file must contain a JSON object")
    return payload


def structural_verdict(verdicts: Mapping[str, object], metric: str
                       ) -> tuple[bool, Mapping[str, object]]:
    """Return the mandatory adjudicated verdict; missing means FAIL."""
    raw = verdicts.get(metric)
    if isinstance(raw, bool):
        return raw, {"passed": raw}
    if isinstance(raw, dict) and isinstance(raw.get("passed"), bool):
        return bool(raw["passed"]), raw
    return False, {"passed": False, "reason": "missing adjudicated verdict"}


def _bound_passed(record, value: object) -> bool:
    """Execute the numeric comparator semantics registered for a record."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(number):
        return False
    if record.kind == "max":
        return number <= record.threshold
    if record.kind == "min":
        return number >= record.threshold
    if record.kind == "strict_max":
        return number < record.threshold
    raise ValueError(f"{record.metric} is not a numeric gate")


def gate_result(record, *, value: object = None, passed: bool | None = None,
                evidence: Mapping[str, object] | None = None) -> dict[str, object]:
    """Materialize one ledger row and its comparator result."""
    if passed is None:
        if record.kind in nest_gates.NUMERIC_KINDS:
            passed = _bound_passed(record, value)
        elif record.kind == "diagnostic":
            passed = True
        else:
            passed = bool(value)
    return {
        "metric": record.metric,
        "kind": record.kind,
        "threshold": record.threshold,
        "comparator": nest_gates.COMPARATORS[record.kind],
        "convention": record.convention,
        "anchor": record.anchor,
        "blocking": record.kind != "diagnostic",
        "value": value,
        "passed": bool(passed),
        "evidence": dict(evidence or {}),
    }


def report_passed(results: Iterable[Mapping[str, object]]) -> bool:
    return all(bool(item["passed"])
               for item in results if bool(item["blocking"]))


def construct_rung_case(
        domain_count: int, *, config_path: str | Path = PRODUCTION_CONFIG,
        run_seconds: int = RUN_SECONDS,
        restart_interval_s: int = RESTART_SPLIT_SECONDS):
    """Build an in-memory N3/N4/N5 view of the production experiment.

    No verification TOML is written.  Domain records are the parent-first
    prefix of ``configs/real74_4dom.toml`` and every derived RunConfig timing
    copy is updated together with its ExperimentConfig authority.
    """
    if domain_count not in (1, 2, 3, 4):
        raise ValueError("real-chain view domain_count must be 1, 2, 3, or 4")
    exp, data = load_experiment_case(config_path)
    selected = exp.domains[:domain_count]
    if tuple(dc.grid_id for dc in selected) != tuple(range(1, domain_count + 1)):
        raise ValueError("production real74 domains are not the d01..d0N prefix")
    domains = tuple(replace(
        dc,
        run=replace(
            dc.run, run_seconds=float(run_seconds),
            restart_interval_s=float(restart_interval_s)))
        for dc in selected)
    return replace(
        exp, run_seconds=float(run_seconds),
        restart_interval_s=float(restart_interval_s), domains=domains), data


def bundle_root(case_data) -> Path:
    """Resolve the explicitly authorized reference bundle from case data."""
    return Path(case_data.wps_namelist).resolve().parent.parent


def child_reference_path(case_data, domain: str) -> Path:
    name = nest_gates.CHILD_REFERENCE_FRAMES[domain]
    return bundle_root(case_data) / "wrfout_reference" / name


def matched_reference_path(case_data, domain: str) -> Path:
    """F21: the SHA-pinned matched-physics FSS reference frame.

    Resolves the frame inside the registered matched-physics bundle (a
    sibling of the mp55 reference bundle) and verifies the byte count and
    SHA-256 against the MATCHED_REFERENCE_FRAMES pin, failing loudly on
    any mismatch.  KeyError for domains without a registered matched
    reference is intentional (their FSS rows block per F21).
    """
    bundle, name, pinned_sha, pinned_bytes = (
        nest_gates.MATCHED_REFERENCE_FRAMES[domain])
    expected_name = nest_gates.CHILD_REFERENCE_FRAMES[domain]
    if name != expected_name:
        raise ValueError(
            f"F21 matched-physics registry names frame {name!r} under "
            f"{domain!r}, but the domain's frame is {expected_name!r}; a "
            "wrong-domain registration must never resolve")
    path = bundle_root(case_data).parent / bundle / "wrfout" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"F21 matched-physics reference missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != pinned_bytes:
        raise ValueError(
            f"F21 matched-physics reference byte count {actual_bytes} != "
            f"pinned {pinned_bytes} for {path}")
    actual_sha = sha256_file(path)
    if actual_sha != pinned_sha:
        raise ValueError(
            f"F21 matched-physics reference SHA-256 {actual_sha} != "
            f"pinned {pinned_sha} for {path}")
    return path


def default_phase4_root() -> Path:
    """Find the controller's read-only d01 anchor trajectory.

    Davies clock bind (2026-07-28): the anchor is ``real74-t7-final-r3``,
    the d01-only trajectory regenerated at the seam-closure tip after the
    root external-boundary clock bind (WRF post-increment dtbc,
    solve_em.F:371-372, plus old-record seam ownership in the final ring
    overwrite) legitimately changed every trajectory -- batched with the
    ring-MP/SEAM-A and init-surface re-ratifications per the Wave-1
    roadmap.  The F26 regeneration pattern applies unchanged: the
    invariance the N3(i) gate protects is re-proven at the fixed code by
    three independent runs (the nested straight run's d01 13:00 frame,
    the d01-only ancestor control, and the separately generated anchor)
    producing byte-identical d01 bytes, with the anchor itself dual-run
    for the no-ECC corruption check.

    History: ``real74-t7-final-r2`` (F26, 2026-07-18, 13:00 sha
    7f588501...) was the post-mudf-lifecycle-fix anchor and encodes the
    retired pre-bind elapsed-based root dtbc; ``real74-t7-final``
    (13:00 sha 1f24638b...) is the original Phase-4 record.  Both are
    historical evidence only.
    """
    candidates = (
        REPOSITORY_ROOT / "out" / "real74-t7-final-r3",
        REPOSITORY_ROOT.parent.parent / "out" / "real74-t7-final-r3",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def frame_path(directory: str | Path, domain_id: int,
               valid_time: datetime) -> Path:
    from gpuwm.io.wrfout import wrfout_filename
    return Path(directory) / wrfout_filename(valid_time, domain_id)


def _chunk_slices(shape: tuple[int, ...], itemsize: int,
                  target_bytes: int = 32 * 1024 * 1024):
    """Yield C-contiguous slabs without loading a complete large frame."""
    if not shape:
        yield ()
        return
    axis = next((index for index, size in enumerate(shape) if size > 1), 0)
    trailing = math.prod(shape[axis + 1:]) * itemsize
    leading = math.prod(shape[:axis])
    per_index = max(1, leading * trailing)
    block = max(1, target_bytes // per_index)
    for start in range(0, shape[axis], block):
        key = [slice(None)] * len(shape)
        key[axis] = slice(start, min(shape[axis], start + block))
        yield tuple(key)


def _variable_bytes_equal(left, right) -> tuple[bool, str | None]:
    if tuple(left.shape) != tuple(right.shape):
        return False, f"shape {tuple(left.shape)} != {tuple(right.shape)}"
    if np.dtype(left.dtype) != np.dtype(right.dtype):
        return False, f"dtype {left.dtype} != {right.dtype}"
    left.set_auto_maskandscale(False)
    right.set_auto_maskandscale(False)
    for key in _chunk_slices(tuple(left.shape), np.dtype(left.dtype).itemsize):
        a = np.ascontiguousarray(np.asarray(left[key]))
        b = np.ascontiguousarray(np.asarray(right[key]))
        if a.tobytes(order="C") != b.tobytes(order="C"):
            return False, f"payload differs in slab {key!r}"
    return True, None


def _attribute_value_equal(left: object, right: object) -> bool:
    """Exact NetCDF attribute equality, including encoded scalar/array type."""
    if isinstance(left, str) or isinstance(right, str):
        return type(left) is type(right) and left == right
    if isinstance(left, bytes) or isinstance(right, bytes):
        return type(left) is type(right) and left == right
    left_array, right_array = np.asarray(left), np.asarray(right)
    if (left_array.dtype != right_array.dtype
            or left_array.shape != right_array.shape):
        return False
    if left_array.dtype.hasobject:
        return bool(np.array_equal(left_array, right_array))
    return (np.ascontiguousarray(left_array).tobytes(order="C")
            == np.ascontiguousarray(right_array).tobytes(order="C"))


def _compare_attribute_sets(
        left, right, *, label: str, excluded: frozenset[str] = frozenset()
        ) -> list[str]:
    """Return every inventory/value mismatch outside a closed exclusion set."""
    left_names = set(left.ncattrs()) - excluded
    right_names = set(right.ncattrs()) - excluded
    mismatches = []
    if left_names != right_names:
        mismatches.append(
            f"{label} attribute inventory differs: "
            f"candidate_only={sorted(left_names-right_names)}, "
            f"baseline_only={sorted(right_names-left_names)}")
    for name in sorted(left_names & right_names):
        if not _attribute_value_equal(
                left.getncattr(name), right.getncattr(name)):
            mismatches.append(f"{label} attribute {name!r} differs")
    return mismatches


def _dimension_signature(dataset) -> dict[str, tuple[int, bool]]:
    return {
        name: (len(dimension), bool(dimension.isunlimited()))
        for name, dimension in dataset.dimensions.items()
    }


def compare_d01_phase4_frame(candidate: str | Path, baseline: str | Path
                             ) -> dict[str, object]:
    """Closed Phase-4 comparator with exactly three named, asymmetric deltas."""
    import netCDF4

    candidate, baseline = Path(candidate), Path(baseline)
    result: dict[str, object] = {
        "candidate": str(candidate), "baseline": str(baseline),
        "scope": ("all dimensions, global attributes, variables, variable "
                  "attributes, dtype/shape/dimensions, and C-order bytes"),
        "ratified_exceptions": list(D01_PHASE4_RATIFIED_EXCEPTIONS),
        "exception_attribution": {
            "REFL_10CM": "ratified D2 scientific-product seam fix",
            "GPUWM_WRITE_COMPLETE": "ratified T15 durable-publication attribute",
            "TITLE": "ratified F20 case-descriptive metadata seam",
        },
        "mismatches": [], "compared_variables": [],
    }
    if not candidate.is_file() or not baseline.is_file():
        result["mismatches"] = ["candidate or baseline frame is missing"]
        result["title_exception"] = {
            "name": "TITLE", "candidate": None, "baseline": None,
            "value_excluded": True}
        result["passed"] = False
        return result
    with netCDF4.Dataset(candidate) as actual, netCDF4.Dataset(baseline) as ref:
        actual_dimensions = _dimension_signature(actual)
        reference_dimensions = _dimension_signature(ref)
        if actual_dimensions != reference_dimensions:
            result["mismatches"].append(
                "dimension inventory/size/unlimited status differs: "
                f"candidate={actual_dimensions!r}, baseline={reference_dimensions!r}")

        result["mismatches"].extend(_compare_attribute_sets(
            actual, ref, label="global",
            excluded=frozenset({"GPUWM_WRITE_COMPLETE", "TITLE"})))
        candidate_title = (actual.getncattr("TITLE")
                           if "TITLE" in actual.ncattrs() else None)
        result["title_exception"] = {
            "name": "TITLE",
            "candidate": candidate_title,
            "baseline": (ref.getncattr("TITLE")
                         if "TITLE" in ref.ncattrs() else None),
            "value_excluded": True,
        }
        if not (isinstance(candidate_title, str) and candidate_title.strip()):
            result["mismatches"].append(
                "candidate TITLE global attribute is missing or empty (the "
                "F20 exception excludes its value, not its presence)")
        publication_present = "GPUWM_WRITE_COMPLETE" in actual.ncattrs()
        publication_value = (
            actual.getncattr("GPUWM_WRITE_COMPLETE")
            if publication_present else None)
        publication_valid = bool(
            publication_present
            and np.asarray(publication_value).shape == ()
            and not isinstance(publication_value, (str, bytes, bool, np.bool_))
            and publication_value == 1)
        if not publication_valid:
            result["mismatches"].append(
                "candidate GPUWM_WRITE_COMPLETE global attribute is missing "
                "or does not have scalar value 1")

        if ("REFL_10CM" not in actual.variables
                or "REFL_10CM" not in ref.variables):
            result["mismatches"].append(
                "ratified REFL_10CM exception variable is missing")
            result["reflectivity_exception"] = {
                "candidate_present": "REFL_10CM" in actual.variables,
                "baseline_present": "REFL_10CM" in ref.variables,
                "excluded_payload": True,
            }
        else:
            actual_refl = actual.variables["REFL_10CM"]
            reference_refl = ref.variables["REFL_10CM"]
            result["mismatches"].extend(_compare_attribute_sets(
                actual_refl, reference_refl, label="REFL_10CM variable"))
            if actual_refl.dimensions != reference_refl.dimensions:
                result["mismatches"].append(
                    "REFL_10CM schema is not ratified: dimensions "
                    f"{actual_refl.dimensions!r} != {reference_refl.dimensions!r}")
            refl_equal, refl_detail = _variable_bytes_equal(
                actual_refl, reference_refl)
            if (refl_detail is not None
                    and (refl_detail.startswith("shape")
                         or refl_detail.startswith("dtype"))):
                result["mismatches"].append(
                    f"REFL_10CM schema is not ratified: {refl_detail}")
            result["reflectivity_exception"] = {
                "candidate_shape": list(actual.variables["REFL_10CM"].shape),
                "baseline_shape": list(ref.variables["REFL_10CM"].shape),
                "dtype": str(actual.variables["REFL_10CM"].dtype),
                "payload_equal": bool(refl_equal),
                "excluded_payload": True,
            }
        actual_names = set(actual.variables) - {"REFL_10CM"}
        reference_names = set(ref.variables) - {"REFL_10CM"}
        if actual_names != reference_names:
            result["mismatches"].append(
                "variable inventory differs outside REFL_10CM: "
                f"candidate_only={sorted(actual_names-reference_names)}, "
                f"baseline_only={sorted(reference_names-actual_names)}")
        for name in sorted(actual_names & reference_names):
            actual_variable = actual.variables[name]
            reference_variable = ref.variables[name]
            result["compared_variables"].append(name)
            result["mismatches"].extend(_compare_attribute_sets(
                actual_variable, reference_variable, label=f"{name} variable"))
            if actual_variable.dimensions != reference_variable.dimensions:
                result["mismatches"].append(
                    f"{name}: dimensions {actual_variable.dimensions!r} != "
                    f"{reference_variable.dimensions!r}")
            equal, detail = _variable_bytes_equal(
                actual_variable, reference_variable)
            if not equal:
                result["mismatches"].append(f"{name}: {detail}")
        result["publication_attribute"] = {
            "name": "GPUWM_WRITE_COMPLETE",
            "candidate": publication_value,
            "candidate_present": publication_present,
            "candidate_required_value": 1,
            "baseline": (ref.getncattr("GPUWM_WRITE_COMPLETE")
                         if "GPUWM_WRITE_COMPLETE" in ref.ncattrs() else None),
            "baseline_value_excluded": True,
        }
    result["passed"] = not result["mismatches"]
    return result


def compare_files_exact(candidate: str | Path, baseline: str | Path,
                        *, block_bytes: int = 8 * 1024 * 1024
                        ) -> dict[str, object]:
    """Streaming whole-file byte comparator used by rung/restart ratchets."""
    candidate, baseline = Path(candidate), Path(baseline)
    if not candidate.is_file() or not baseline.is_file():
        return {"passed": False, "candidate": str(candidate),
                "baseline": str(baseline), "reason": "missing artifact"}
    if candidate.stat().st_size != baseline.stat().st_size:
        return {"passed": False, "candidate": str(candidate),
                "baseline": str(baseline),
                "candidate_bytes": candidate.stat().st_size,
                "baseline_bytes": baseline.stat().st_size,
                "reason": "size mismatch"}
    offset = 0
    with candidate.open("rb") as left, baseline.open("rb") as right:
        while True:
            a, b = left.read(block_bytes), right.read(block_bytes)
            if a != b:
                return {"passed": False, "candidate": str(candidate),
                        "baseline": str(baseline),
                        "first_differing_block_offset": offset,
                        "reason": "payload mismatch"}
            if not a:
                break
            offset += len(a)
    digest = sha256_file(candidate)
    return {"passed": True, "candidate": str(candidate),
            "baseline": str(baseline), "bytes": offset,
            "sha256": digest}


def _inventory_sha256(members: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256(b"gpuwm-canonical-state-inventory-v2\0")
    for member in members:
        descriptor = json.dumps(
            [str(member["name"]), str(member["dtype"]),
             [int(size) for size in member["shape"]]],
            separators=(",", ":"), ensure_ascii=True).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
    return digest.hexdigest()


def _normalize_inventory(raw: Mapping[str, object]) -> dict[str, object]:
    """Validate and canonicalize one frame's complete array inventory."""
    if not isinstance(raw, Mapping):
        raise ValueError("canonical state sample has no frame inventory")
    if raw.get("schema") != CANONICAL_INVENTORY_SCHEMA:
        raise ValueError("canonical state sample has an unknown inventory schema")
    members_raw = raw.get("members")
    if not isinstance(members_raw, (list, tuple)):
        raise ValueError("canonical state inventory members are malformed")
    members = []
    names = set()
    for item in members_raw:
        if not isinstance(item, Mapping) or set(item) != {"name", "dtype", "shape"}:
            raise ValueError("canonical state inventory member is malformed")
        name = str(item["name"])
        dtype = str(item["dtype"])
        shape_raw = item["shape"]
        if (not name or not dtype
                or not isinstance(shape_raw, (list, tuple))):
            raise ValueError("canonical state inventory member is malformed")
        shape = [int(size) for size in shape_raw]
        if any(size < 0 for size in shape) or name in names:
            raise ValueError("canonical state inventory has invalid members")
        names.add(name)
        members.append({"name": name, "dtype": dtype, "shape": shape})
    if [item["name"] for item in members] != sorted(names):
        raise ValueError("canonical state inventory members are not ordered")
    expected_sha256 = _inventory_sha256(members)
    if (int(raw.get("array_count", -1)) != len(members)
            or str(raw.get("sha256")) != expected_sha256):
        raise ValueError("canonical state inventory digest does not match members")
    return {
        "schema": CANONICAL_INVENTORY_SCHEMA,
        "sha256": expected_sha256,
        "array_count": len(members),
        "members": members,
    }


def _inventory_member_set(inventory: Mapping[str, object]
                          ) -> set[tuple[str, str, tuple[int, ...]]]:
    normalized = _normalize_inventory(inventory)
    return {
        (str(item["name"]), str(item["dtype"]),
         tuple(int(size) for size in item["shape"]))
        for item in normalized["members"]
    }


def lazy_inventory_member_class(name: str) -> str | None:
    """Return the documented lazy class for a canonical member path."""
    from gpuwm.io import restart as restart_io

    scratch_slot = (name.removeprefix("scratch/")
                    if name.startswith("scratch/") else None)
    if scratch_slot in _CANONICAL_NEST_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[0]
    if scratch_slot in _CANONICAL_LBC_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[1]
    if scratch_slot in _CANONICAL_REFL_SLOTS:
        return CANONICAL_LAZY_MEMBER_CLASSES[2]
    if name.startswith("scratch/"):
        slot = name.removeprefix("scratch/")
        if slot in restart_io.SERIALIZED_SCRATCH_SLOTS:
            if slot.startswith("mp_"):
                return CANONICAL_LAZY_MEMBER_CLASSES[3]
            if slot.startswith("cu_"):
                return CANONICAL_LAZY_MEMBER_CLASSES[4]
    if name == "cumulus/w0avg":
        return CANONICAL_LAZY_MEMBER_CLASSES[5]
    return None


def restart_boundary_member_class(name: str) -> str | None:
    """Return the REBUILT-only restart-boundary difference class.

    Serialized scratch, held tendencies, fields, driver arrays, and W0AVG are
    restored exactly by the v3 reader.  Only the three frame-state families
    that the restart contract explicitly rebuilds may differ at resume.
    """
    member_class = lazy_inventory_member_class(name)
    if member_class in CANONICAL_LAZY_MEMBER_CLASSES[:3]:
        return member_class
    return None


def validate_inventory_growth(
        previous: Mapping[str, object] | None,
        current: Mapping[str, object], *, domain: str, ticks: int
        ) -> dict[str, object]:
    """Enforce monotonic, allowlisted inventory growth within one run."""
    current = _normalize_inventory(current)
    if previous is None:
        return current
    previous = _normalize_inventory(previous)
    before = _inventory_member_set(previous)
    after = _inventory_member_set(current)
    removed = sorted(before - after)
    if removed:
        raise RuntimeError(
            f"canonical mutable-state inventory shrank for {domain} at "
            f"ticks={ticks}: {removed}")
    added = sorted(after - before)
    forbidden = sorted(
        member for member in added
        if lazy_inventory_member_class(member[0]) is None)
    if forbidden:
        raise RuntimeError(
            f"canonical mutable-state inventory grew outside the lazy-member "
            f"allowlist for {domain} at ticks={ticks}: {forbidden}")
    return current


def _canonical_extra_manifest(state) -> dict[str, object]:
    """Audited rebuilt members whose frame-time bytes are ratchet state.

    Nest rolling tables converge after the first parent STEP -> FORCE after
    restore.  REFL arrays exist at their producing history instant and are
    consumed only after this digest.  LBC weights are deterministic lazy
    attachments.  Other per-call rebuilt scratch is deliberately excluded:
    it has no frame-time/restart contract and its dead residual bytes are not
    model trajectory state.
    """
    manifest = {}
    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        name = f"scratch/{slot}"
        if slot in _CANONICAL_EXCLUDED_FRAME_SCRATCH:
            continue
        if lazy_inventory_member_class(name) in CANONICAL_LAZY_MEMBER_CLASSES[:3]:
            manifest[name] = value
        elif slot.startswith(_CANONICAL_EXTRA_PREFIXES):
            raise RuntimeError(
                f"canonical frame-state scratch member {slot!r} is inside an "
                "audited lazy prefix but is not a concrete registered member")
    return manifest


def _canonical_scalar_bytes(scalars: Mapping[str, object]) -> bytes:
    return json.dumps(
        scalars, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")


#: F23: scratch slots whose contents are a pure function of PARENT DUTY --
#: written only while forcing a child, physically inert to the owning
#: domain's trajectory (N4 evidence: d02 wrfout frames byte-identical
#: 2-dom vs 3-dom while exactly this 1 of 288 canonical members differed).
#: The TRAJECTORY digest scope excludes them so cross-rung/cross-shape
#: comparisons (ratchet state evidence, ancestor inertness) compare only
#: child-independent state; the FULL scope retains them.
CHILD_DUTY_SCRATCH_MEMBERS = (
    "scratch/nest_parent_field",
    "scratch/nest_child_field",
)


def canonical_state_digest(state, clock, *,
                           scope: str = "trajectory") -> dict[str, object]:
    """Hash complete frame-time trajectory state in canonical restart order.

    The restart manifest is the repository's fail-loud authority for which
    arrays survive a step boundary.  Reusing it here prevents the N5
    ancestor check from silently omitting a newly introduced prognostic,
    accumulator, held tendency, or physics field.  Scalar driver counters
    and model time are included because they are serialized in the restart
    header rather than its array payload.  The audited nest/REFL/LBC lazy
    surfaces are added because they are live at a history instant even though
    the restart contract rebuilds them.

    ``scope`` (F23): ``"trajectory"`` (the rung-evidence scope) excludes
    CHILD_DUTY_SCRATCH_MEMBERS so the digest is invariant to whether the
    domain currently forces a child; ``"full"`` includes every member.
    The scope is baked into the digest seed so hashes from different
    scopes can never be compared silently.
    """
    if scope not in ("trajectory", "full"):
        raise ValueError(f"unknown canonical digest scope {scope!r}")
    from gpuwm.io import restart as restart_io

    manifest: dict[str, object] = {}
    manifest.update(restart_io.state_manifest(state))
    manifest.update(restart_io._scratch_manifest(state))
    driver = getattr(state, "physics", None)
    if driver is not None:
        manifest.update(restart_io._driver_manifest(driver))
    manifest.update(_canonical_extra_manifest(state))
    if scope == "trajectory":
        for member in CHILD_DUTY_SCRATCH_MEMBERS:
            manifest.pop(member, None)
    dtbc_fp32_bits = int(
        np.asarray(np.float32(clock.dtbc_fp32)).view(np.uint32).item())
    scalars = {
        "elapsed_seconds": float(state.elapsed_seconds),
        "dtbc_fp32_bits": dtbc_fp32_bits,
        "driver": (None if driver is None else {
            "call_counts": {
                key: int(value)
                for key, value in sorted(driver.call_counts.items())},
            "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
            "microphysics_updates": int(driver.microphysics_updates),
        }),
    }
    scalar_bytes = _canonical_scalar_bytes(scalars)
    digest = hashlib.sha256(
        b"gpuwm-canonical-state-v2:" + scope.encode("ascii") + b"\0"
        + scalar_bytes)
    members = []
    for key in sorted(manifest):
        host = restart_io._host(manifest[key])
        host = np.ascontiguousarray(host)
        member = {
            "name": key, "dtype": str(host.dtype),
            "shape": list(host.shape),
        }
        descriptor = json.dumps(
            [member["name"], member["dtype"], member["shape"]],
            separators=(",", ":"), ensure_ascii=True).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(host.tobytes(order="C"))
        members.append(member)
    inventory = {
        "schema": CANONICAL_INVENTORY_SCHEMA,
        "sha256": _inventory_sha256(members),
        "array_count": len(members),
        "members": members,
    }
    return {
        "schema": CANONICAL_STATE_SCHEMA,
        "sha256": digest.hexdigest(),
        "inventory_sha256": inventory["sha256"],
        "scalar_sha256": hashlib.sha256(scalar_bytes).hexdigest(),
        "array_count": len(members),
        "field_order": [item["name"] for item in members],
        "inventory": inventory,
        "scalars": scalars,
    }


def _read_field(path: str | Path, name: str) -> np.ndarray:
    from gpuwm import netcdf_bridge
    with netcdf_bridge.open_dataset(path) as dataset:
        if name not in dataset.variables:
            raise KeyError(f"{path} has no {name} variable")
        return np.asarray(
            np.ma.filled(dataset.variables[name][0], np.nan),
            dtype=np.float64)


def boundary_zone_blowup_value(
        run_summary: Mapping[str, object]) -> tuple[float, dict[str, object]]:
    """Consume the production run-cumulative, every-substep w diagnostic."""
    required = {
        "boundary_w_max_ms", "interior_w_max_ms", "boundary_zone_blowup"}
    missing = required - set(run_summary)
    if missing:
        raise ValueError(
            f"production RunSummary lacks {sorted(missing)}")
    boundary_max = float(run_summary["boundary_w_max_ms"])
    interior_max = float(run_summary["interior_w_max_ms"])
    fired = bool(run_summary["boundary_zone_blowup"])
    return float(fired), {
        "boundary_w_max_ms": boundary_max,
        "interior_w_max_ms": interior_max,
        "diagnostic_fired": bool(fired),
        "source": "production RunSummary accumulated over every dynamics substep",
    }


def _composite_reflectivity(path: str | Path) -> np.ndarray:
    refl = _read_field(path, "REFL_10CM")
    if not np.isfinite(refl).all():
        raise ValueError(f"REFL_10CM in {path} is non-finite")
    if refl.ndim == 2:
        return refl
    if refl.ndim != 3:
        raise ValueError(f"REFL_10CM must be 2-D or 3-D, got {refl.shape}")
    return np.max(refl, axis=0)


def _neighborhood_fraction(events: np.ndarray, radius_cells: float
                           ) -> np.ndarray:
    """Circular grid-centre neighborhood fraction with in-domain support."""
    if events.ndim != 2:
        raise ValueError("FSS event field must be two-dimensional")
    radius = int(math.floor(radius_cells))
    offsets = tuple(
        (dj, di) for dj in range(-radius, radius + 1)
        for di in range(-radius, radius + 1)
        if dj * dj + di * di <= radius_cells * radius_cells)
    numerator = np.zeros(events.shape, dtype=np.float64)
    denominator = np.zeros(events.shape, dtype=np.float64)
    ny, nx = events.shape
    for dj, di in offsets:
        src_j = slice(max(0, -dj), min(ny, ny - dj))
        src_i = slice(max(0, -di), min(nx, nx - di))
        dst_j = slice(max(0, dj), min(ny, ny + dj))
        dst_i = slice(max(0, di), min(nx, nx + di))
        numerator[dst_j, dst_i] += events[src_j, src_i]
        denominator[dst_j, dst_i] += 1.0
    return numerator / denominator


def fractions_skill_score(candidate: np.ndarray, reference: np.ndarray, *,
                          event_threshold: float, radius_km: float,
                          dx_m: float) -> float:
    """Registered FSS formula over the reusable five-cell interior mask."""
    if candidate.shape != reference.shape or candidate.ndim != 2:
        return float("nan")
    if (not np.isfinite(candidate).all()
            or not np.isfinite(reference).all()
            or not math.isfinite(dx_m) or dx_m <= 0.0):
        return float("nan")
    candidate_fraction = _neighborhood_fraction(
        candidate >= event_threshold, radius_km * 1000.0 / dx_m)
    reference_fraction = _neighborhood_fraction(
        reference >= event_threshold, radius_km * 1000.0 / dx_m)
    region = weather_metrics.interior_region(candidate.shape)
    left, right = candidate_fraction[region], reference_fraction[region]
    denominator = float(np.sum(left * left + right * right, dtype=np.float64))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else float("nan")
    numerator = float(np.sum((left - right) ** 2, dtype=np.float64))
    return 1.0 - numerator / denominator


def _event_coverage(field: np.ndarray, event_threshold: float) -> float:
    """Event fraction on the exact interior mask registered for FSS."""
    region = weather_metrics.interior_region(field.shape)
    interior = field[region]
    if interior.size == 0:
        raise ValueError("FSS event coverage has an empty registered interior")
    return float(np.mean(interior >= event_threshold, dtype=np.float64))


def ensemble_envelope_adjudication(
        held_rows: Sequence[Mapping[str, object]],
        member_frames: Sequence[str | Path], *, domain: str,
        dx_m: float) -> dict[str, object]:
    """Adjudicate F24-held FSS rows against a same-domain WRF ensemble.

    A majority of members below the registered event-coverage floor confirms
    that a held row is meteorologically degenerate.  Otherwise ``revoked`` is
    true; the consumer is responsible for re-opening its rung.
    """
    frames = tuple(Path(frame).resolve() for frame in member_frames)
    if len(frames) < 3:
        raise ValueError("F24 ensemble adjudication needs at least 3 members")
    if len(set(frames)) != len(frames):
        raise ValueError("F24 ensemble members must use distinct frame paths")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("F24 ensemble adjudication dx_m must be finite and positive")
    expected_name = f"wrfout_{domain}_1974-04-03_13_15_00"
    wrong_frames = [str(frame) for frame in frames if frame.name != expected_name]
    if wrong_frames:
        raise ValueError(
            f"F24 ensemble frames must all be same-domain 13:15 {expected_name}: "
            f"{wrong_frames}")

    registered = {
        float(event_dbz): float(radius_km)
        for event_dbz, radius_km, _minimum
        in nest_gates.REFL_10CM_FSS_FAMILY}
    floor = nest_gates.FSS_DEGENERATE_EVENT_FLOOR
    validated_rows: list[
        tuple[Mapping[str, object], float, float, float, float]
    ] = []
    seen_thresholds: set[float] = set()
    for held in held_rows:
        try:
            json.dumps(dict(held), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "F24 held rows must contain strict-JSON values with finite "
                "numeric evidence") from exc
        try:
            event_dbz = float(held["event_dbz"])
            candidate_coverage = float(held["candidate_coverage"])
            reference_coverage = float(held["reference_coverage"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "F24 held rows need numeric event_dbz and candidate/reference "
                "coverages") from exc
        if event_dbz not in registered:
            raise ValueError(
                f"F24 held threshold {event_dbz} is not registered")
        if event_dbz in seen_thresholds:
            raise ValueError(f"F24 held threshold {event_dbz} is duplicated")
        seen_thresholds.add(event_dbz)
        if held.get("degenerate") is not True:
            raise ValueError("F24 held rows must carry degenerate is True")
        if held.get("adjudication") != "held-for-ensemble-envelope-f24":
            raise ValueError(
                "F24 held rows must carry the registered adjudication marker")
        if (not math.isfinite(candidate_coverage)
                or not 0.0 <= candidate_coverage <= 1.0
                or not math.isfinite(reference_coverage)
                or not 0.0 <= reference_coverage <= 1.0):
            raise ValueError("F24 held-row coverages must be finite fractions")
        if min(candidate_coverage, reference_coverage) >= floor:
            raise ValueError(
                "F24 held rows need candidate or reference coverage below "
                "the registered event floor")
        radius_km = registered[event_dbz]
        if ("radius_km" in held
                and float(held["radius_km"]) != radius_km):
            raise ValueError(
                f"F24 held threshold {event_dbz} does not use its registered radius")
        validated_rows.append((
            held, event_dbz, candidate_coverage, reference_coverage, radius_km))

    composites = tuple(_composite_reflectivity(frame) for frame in frames)
    expected_shape = composites[0].shape
    if (len(expected_shape) != 2
            or any(field.shape != expected_shape for field in composites)):
        raise ValueError("F24 ensemble reflectivity fields must share one 2-D shape")
    if any(not np.isfinite(field).all() for field in composites):
        raise ValueError("F24 ensemble reflectivity fields must be finite")

    rows: list[dict[str, object]] = []
    majority_required = len(frames) // 2 + 1
    for (held, event_dbz, candidate_coverage, reference_coverage,
         radius_km) in validated_rows:
        coverage_evidence = []
        for frame, field in zip(frames, composites):
            coverage = _event_coverage(field, event_dbz)
            coverage_evidence.append({
                "frame": str(frame),
                "coverage": coverage,
                "below_floor": bool(coverage < floor),
            })
        degenerate_members = sum(
            item["below_floor"] for item in coverage_evidence)
        confirmed = degenerate_members >= majority_required

        pair_values: list[float] = []
        pair_evidence = []
        for (left_index, left), (right_index, right) in combinations(
                enumerate(composites), 2):
            value = fractions_skill_score(
                left, right, event_threshold=event_dbz,
                radius_km=radius_km, dx_m=dx_m)
            if not math.isfinite(value):
                raise ValueError(
                    f"F24 ensemble pair has no registered FSS at {event_dbz} dBZ")
            pair_values.append(float(value))
            pair_evidence.append({
                "left_frame": str(frames[left_index]),
                "right_frame": str(frames[right_index]),
                "fss": float(value),
            })

        verdict = dict(held)
        verdict.update({
            "event_dbz": event_dbz,
            "radius_km": radius_km,
            "candidate_coverage": candidate_coverage,
            "reference_coverage": reference_coverage,
            "verdict": ("confirmed-degenerate" if confirmed else "revoked"),
            "confirmed_degenerate": bool(confirmed),
            "revoked": bool(not confirmed),
            "member_coverages": coverage_evidence,
            "degenerate_member_count": int(degenerate_members),
            "majority_required": majority_required,
            "pairwise_fss_envelope": {
                "min": min(pair_values),
                "max": max(pair_values),
                "values": pair_values,
                "pairs": pair_evidence,
            },
        })
        rows.append(verdict)

    return {
        "schema": "gpuwm-f24-ensemble-envelope-v1",
        "domain": domain,
        "valid_time": "1974-04-03T13:15:00Z",
        "dx_m": float(dx_m),
        "event_floor": float(floor),
        "member_count": len(frames),
        "member_frames": [str(frame) for frame in frames],
        "rows": rows,
    }


def score_statistical_frame(
        milestone: str, domain: str, candidate: str | Path,
        reference: str | Path, *, dx_m: float,
        run_summary: Mapping[str, object],
        verdicts: Mapping[str, object] | None = None,
        fss_reference: str | Path | None = None) -> dict[str, dict[str, object]]:
    """Evaluate the complete registered child statistical gate family.

    ``fss_reference`` (F21) redirects ONLY the REFL_10CM FSS row to the
    matched-physics reference; every other row keeps ``reference``.  When
    omitted, the FSS row scores against ``reference`` unchanged — for
    domains without a registered matched reference that is the honest
    blocking state the F21 record describes.  F27 applies its calibrated
    envelope-minimum bar and standing-deficiency evidence only to the
    registered d03/d04 20-dBZ rows.
    """
    verdicts = verdicts or {}
    actual = weather_metrics.wrf_diagnostics(Path(candidate))
    oracle = weather_metrics.wrf_diagnostics(Path(reference))
    region = weather_metrics.interior_region(actual["mslp"].shape)
    values = {
        f"{domain}_mslp_pattern_correlation":
            weather_metrics.pattern_correlation(
                actual["mslp"][region], oracle["mslp"][region]),
        f"{domain}_t500_rmse_k": weather_metrics.rmse(
            actual["levels"][500]["temperature"][region],
            oracle["levels"][500]["temperature"][region]),
        f"{domain}_t850_rmse_k": weather_metrics.rmse(
            actual["levels"][850]["temperature"][region],
            oracle["levels"][850]["temperature"][region]),
    }
    blowup, blowup_evidence = boundary_zone_blowup_value(run_summary)
    values[f"{domain}_boundary_zone_blowup"] = blowup
    results: dict[str, dict[str, object]] = {}
    for metric, value in values.items():
        record = _registered_gate(milestone, metric)
        evidence = blowup_evidence if metric.endswith("boundary_zone_blowup") else {}
        results[metric] = gate_result(record, value=value, evidence=evidence)

    structure_metric = f"{domain}_refl_10cm_structure"
    structure_record = _registered_gate(milestone, structure_metric)
    structure_passed, adjudication = structural_verdict(
        verdicts, structure_metric)
    try:
        composite = _composite_reflectivity(candidate)
        structure_evidence = {
            "finite": bool(np.isfinite(composite).all()),
            "minimum_dbz": float(np.min(composite)),
            "maximum_dbz": float(np.max(composite)),
            "adjudication": dict(adjudication),
        }
    except (KeyError, ValueError, OSError) as exc:
        structure_passed = False
        structure_evidence = {"error": str(exc),
                              "adjudication": dict(adjudication)}
    results[structure_metric] = gate_result(
        structure_record, passed=structure_passed,
        evidence=structure_evidence)

    fss_metric = f"{domain}_refl_10cm_fss"
    fss_record = _registered_gate(milestone, fss_metric)
    fss_pair = reference if fss_reference is None else fss_reference
    fss_evidence: list[dict[str, object]] = []
    fss_passed = True
    try:
        candidate_refl = _composite_reflectivity(candidate)
        reference_refl = _composite_reflectivity(fss_pair)
        if candidate_refl.ndim != 2 or reference_refl.ndim != 2:
            raise ValueError("FSS reflectivity composites must both be 2-D")
        candidate_interior = candidate_refl[
            weather_metrics.interior_region(candidate_refl.shape)]
        reference_interior = reference_refl[
            weather_metrics.interior_region(reference_refl.shape)]
        if candidate_interior.size == 0 or reference_interior.size == 0:
            raise ValueError("FSS reflectivity has an empty registered interior")
        coverage_is_valid = (
            candidate_refl.shape == reference_refl.shape
            and np.isfinite(candidate_refl).all()
            and np.isfinite(reference_refl).all()
            and math.isfinite(dx_m)
            and dx_m > 0.0)
        for event_dbz, radius_km, minimum in nest_gates.REFL_10CM_FSS_FAMILY:
            value = fractions_skill_score(
                candidate_refl, reference_refl,
                event_threshold=event_dbz, radius_km=radius_km, dx_m=dx_m)
            envelope_minimum = nest_gates.F27_DOCUMENTED_DEFICIENCY_ROWS.get(
                (domain, event_dbz))
            effective_minimum = (
                minimum if envelope_minimum is None else envelope_minimum)
            candidate_coverage = reference_coverage = float("nan")
            if coverage_is_valid:
                candidate_coverage = _event_coverage(
                    candidate_refl, event_dbz)
                reference_coverage = _event_coverage(
                    reference_refl, event_dbz)
            degenerate = coverage_is_valid and (
                candidate_coverage < nest_gates.FSS_DEGENERATE_EVENT_FLOOR
                or reference_coverage < nest_gates.FSS_DEGENERATE_EVENT_FLOOR)
            documented_deficiency = (
                not degenerate
                and envelope_minimum is not None
                and math.isfinite(value)
                and value < envelope_minimum)
            accepted = math.isfinite(value) and (
                value >= effective_minimum or documented_deficiency)
            row: dict[str, object] = {
                "event_dbz": event_dbz, "radius_km": radius_km,
                "minimum": effective_minimum, "value": value,
                "passed": bool(accepted or degenerate),
            }
            if degenerate:
                row.update({
                    "degenerate": True,
                    "candidate_coverage": candidate_coverage,
                    "reference_coverage": reference_coverage,
                    "adjudication": "held-for-ensemble-envelope-f24",
                })
                if not math.isfinite(value):
                    # The registered zero-denominator FAIL has no numeric
                    # score.  The held row remains provisional, not failed.
                    row.pop("value")
                    row["fss"] = None
            else:
                if documented_deficiency:
                    row.update({
                        "documented_deficiency": True,
                        "envelope_minimum": envelope_minimum,
                        "adjudication":
                            "f25-envelope-standing-deficiency-f27",
                    })
                fss_passed &= accepted
            fss_evidence.append(row)
    except (KeyError, ValueError, OSError) as exc:
        fss_passed = False
        fss_evidence.append({"error": str(exc)})
    evidence: dict[str, object] = {"scores": fss_evidence}
    if fss_reference is not None:
        # Only the F21 redirected path extends the evidence; the omitted
        # path preserves the pre-F21 evidence shape byte-for-byte.
        evidence.update({
            "fss_reference": str(fss_pair),
            "fss_reference_sha256": sha256_file(fss_pair),
            "matched_physics_reference": True,
        })
    results[fss_metric] = gate_result(
        fss_record, passed=fss_passed, evidence=evidence)
    return results


def output_static_recheck(candidate: str | Path, reference: str | Path, *,
                          spec_bdy_width: int, blend_width: int
                          ) -> dict[str, float]:
    """HGT/MUB N1 comparator repeated in the actual output frame."""
    hgt, ref_hgt = _read_field(candidate, "HGT"), _read_field(reference, "HGT")
    mub, ref_mub = _read_field(candidate, "MUB"), _read_field(reference, "MUB")
    if hgt.shape != ref_hgt.shape or mub.shape != ref_mub.shape:
        raise ValueError("candidate/reference HGT or MUB shapes differ")
    zone = blend_zone_mask(
        hgt.shape, spec_bdy_width=spec_bdy_width,
        blend_width=blend_width)
    return {
        "hgt_m": float(np.max(np.abs(hgt[zone] - ref_hgt[zone]))),
        "mub_pa": float(np.max(np.abs(mub[zone] - ref_mub[zone]))),
    }


def blend_zone_surface_bias(candidate: str | Path, reference: str | Path, *,
                            spec_bdy_width: int, blend_width: int
                            ) -> dict[str, object]:
    """Non-blocking WRF-fidelity soil/terrain blind-spot diagnostic."""
    fields = {}
    for name in ("T2", "TSK"):
        actual, oracle = _read_field(candidate, name), _read_field(reference, name)
        zone = blend_zone_mask(
            actual.shape, spec_bdy_width=spec_bdy_width,
            blend_width=blend_width)
        delta = actual[zone] - oracle[zone]
        fields[name] = {
            "mean_bias_k": float(np.mean(delta, dtype=np.float64)),
            "rmse_k": weather_metrics.rmse(actual[zone], oracle[zone]),
            "max_abs_bias_k": float(np.max(np.abs(delta))),
            "samples": int(delta.size),
        }
    return fields


@dataclass(frozen=True)
class ProductionRun:
    output_dir: Path
    wrfout_paths: tuple[Path, ...]
    checkpoint: Path | None
    completed_seconds: float
    execution: Mapping[str, object]
    timing: Mapping[str, float]
    memory: Mapping[str, int | None]

    def paths_for_domain(self, grid_id: int) -> tuple[Path, ...]:
        token = f"wrfout_d{grid_id:02d}_"
        return tuple(path for path in self.wrfout_paths
                     if path.name.startswith(token))


def _experiment_prefix(exp, domain_count: int):
    if domain_count < 1 or domain_count > len(exp.domains):
        raise ValueError(
            f"domain_count {domain_count} outside 1..{len(exp.domains)}")
    return replace(exp, domains=tuple(exp.domains[:domain_count]))


def experiment_prefix_provenance(exp, catalog, domain_count: int
                                 ) -> dict[str, object]:
    """Fingerprint and exact clock range for one resolved experiment prefix."""
    from gpuwm.core.clock import resolve_clock
    from gpuwm.core.model import experiment_fingerprint

    prefix = _experiment_prefix(exp, domain_count)
    clock = resolve_clock(prefix)
    return {
        "experiment_fingerprint": experiment_fingerprint(prefix, catalog),
        "tick_start": 0,
        "tick_stop": int(clock.run_ticks),
        "tick_den": int(clock.tick_den),
        "domain_ids": [dc.grid_id for dc in prefix.domains],
    }


def resolve_experiment_prefix_provenance(exp, case_data, domain_count: int
                                         ) -> dict[str, object]:
    """CPU-only provenance used to refuse stale ratchets before GPU setup."""
    from gpuwm.ingest.preflight import build_input_catalog

    return experiment_prefix_provenance(
        exp, build_input_catalog(case_data), domain_count)


def run_prefix_provenance(run: ProductionRun, domain_count: int
                          ) -> dict[str, object]:
    raw = run.execution.get("experiment_prefix_provenance")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "production execution has no experiment/config fingerprint ledger")
    provenance = raw.get(str(domain_count))
    if not isinstance(provenance, Mapping):
        raise ValueError(
            f"production execution has no {domain_count}-domain provenance")
    result = dict(provenance)
    result["evaluator_commit"] = run.execution.get("evaluator_commit")
    if not isinstance(result["evaluator_commit"], str):
        raise ValueError("production execution has no evaluator commit")
    expected_seconds = (
        (int(result["tick_stop"]) - int(result["tick_start"]))
        / int(result["tick_den"]))
    if run.completed_seconds != expected_seconds:
        raise ValueError(
            "production execution completed_seconds differs from its tick range")
    return result


def run_summary_for_domain(run: ProductionRun, domain: str
                           ) -> Mapping[str, object]:
    summaries = run.execution.get("run_summaries")
    if not isinstance(summaries, Mapping):
        raise ValueError("production execution has no RunSummary measurements")
    summary = summaries.get(domain)
    if not isinstance(summary, Mapping):
        raise ValueError(f"production execution has no {domain} RunSummary")
    return summary


def production_run_from_report(report: Mapping[str, object]) -> ProductionRun:
    """Rehydrate read-only run evidence for post-review re-evaluation."""
    production = report.get("production_execution")
    if not isinstance(production, dict):
        raise ValueError("existing rung report has no production_execution")
    output_dir = Path(production["output_dir"])
    paths = tuple(sorted(path for path in output_dir.glob("wrfout_d*")
                         if path.is_file()))
    if not paths:
        raise ValueError(f"existing run directory has no wrfout frames: {output_dir}")
    return ProductionRun(
        output_dir=output_dir, wrfout_paths=paths, checkpoint=None,
        completed_seconds=float(production["completed_seconds"]),
        execution=dict(production.get("execution", {})),
        timing=dict(production.get("timing", {})),
        memory=dict(production.get("memory", {})))


def gate_evidence_from_report(report: Mapping[str, object], metric: str
                              ) -> Mapping[str, object]:
    for result in report.get("gates", []):
        if result.get("metric") == metric:
            evidence = result.get("evidence", {})
            if isinstance(evidence, dict):
                return evidence
    raise ValueError(f"existing rung report has no {metric} evidence")


def state_hashes_for_domain(
        execution: Mapping[str, object], domain: str
        ) -> tuple[dict[str, object], ...]:
    """Extract and validate one domain's canonical state-hash ledger."""
    raw = execution.get("canonical_state_hashes", ())
    if not isinstance(raw, list):
        raise ValueError("production execution has no canonical state hashes")
    samples = tuple(
        sample for sample in _normalize_state_hashes(raw)
        if sample["domain"] == domain)
    if not samples:
        raise ValueError(f"production execution has no {domain} state hashes")
    return samples


def state_hash_inventory(exp, execution: Mapping[str, object], domain: str
                         ) -> dict[str, object]:
    """Validate one execution's state hashes against its exact schedule."""
    from gpuwm.core.clock import resolve_clock
    from gpuwm.io.wrfout import wrfout_filename

    grid_id = int(domain.removeprefix("d"))
    dc = exp.domain(grid_id)
    tick_den = resolve_clock(exp).tick_den
    expected = {
        wrfout_filename(
            exp.start_time + timedelta(seconds=offset), grid_id): offset
        for offset in range(
            0, int(exp.run_seconds) + 1, int(dc.history_interval_s))}
    samples = state_hashes_for_domain(execution, domain)
    actual = {str(sample["frame"]): sample for sample in samples}
    inventory_equal = set(actual) == set(expected)
    rows = []
    passed = inventory_equal
    for frame, offset in expected.items():
        sample = actual.get(frame)
        expected_instant = Fraction(offset, 1)
        accepted = bool(
            sample is not None
            and int(sample["tick_den"]) == tick_den
            and int(sample["ticks"]) == offset * tick_den
            and _state_sample_instant(sample) == expected_instant)
        passed &= accepted
        rows.append({"frame": frame, "offset_seconds": offset,
                     "passed": accepted})
    return {"passed": bool(passed), "domain": domain,
            "tick_den": tick_den, "inventory_equal": inventory_equal,
            "expected_samples": len(expected), "samples": rows}


class _StopAtCheckpoint(RuntimeError):
    def __init__(self, checkpoint: Path):
        super().__init__(f"intentional stop at {checkpoint}")
        self.checkpoint = checkpoint


def execute_production_run(
        exp, case_data, output_dir: str | Path, *, restart: str | Path | None = None,
        stop_at_checkpoint_seconds: int | None = None) -> ProductionRun:
    """Run the production tree builder/executor with rung instrumentation."""
    import gpuwm.core.dycore as production_dycore
    import gpuwm.core.model as production_model
    from gpuwm.core.dycore import stability_report
    from gpuwm.io.restart import restore_tree_restart, write_tree_restart
    from gpuwm.io.wrfout import PerDomainWrfoutWriters, quarantine_orphan_wrfouts
    from gpuwm.runtime import _submit_tree_history_frame
    from gpuwm.supervisor import validate_manifest_checkpoint

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_orphan_wrfouts(output_dir)
    import cupy as cp

    pool = cp.get_default_memory_pool()
    free_before, _total = cp.cuda.runtime.memGetInfo()
    model = production_model.build_experiment(exp, case_data)
    used_peak = int(pool.used_bytes())
    held_peak = int(pool.total_bytes())
    prior_state_inventories: dict[str, dict[str, object]] = {}
    resume_inventories: dict[str, dict[str, object]] = {}
    if restart is not None:
        restore_tree_restart(validate_manifest_checkpoint(restart), model)
        for node in model.walk_parent_first():
            domain = f"d{node.cfg.grid_id:02d}"
            state_hash = canonical_state_digest(node.state, node.clock)
            inventory = _normalize_inventory(state_hash["inventory"])
            prior_state_inventories[domain] = inventory
            resume_inventories[domain] = {
                "domain": domain,
                "ticks": int(node.clock.ticks),
                "tick_den": int(node.clock.tick_den),
                "valid_seconds": float(node.clock.elapsed_seconds),
                "inventory": inventory,
            }
    last_checkpoint: Path | None = None
    execution_report = None
    stopped = False
    state_hashes: list[dict[str, object]] = []
    checkpoint_state_hashes: list[dict[str, object]] = []
    state_inventories: dict[str, list[dict[str, object]]] = {}
    state_hash_wall_seconds = 0.0
    state_hash_cpu_seconds = 0.0
    summary_wall_seconds = 0.0
    summary_cpu_seconds = 0.0
    run_summaries = {
        f"d{node.cfg.grid_id:02d}": {
            "boundary_w_max_ms": 0.0,
            "interior_w_max_ms": 0.0,
            "boundary_zone_blowup": False,
            "dynamics_substeps": 0,
        }
        for node in model.walk_parent_first()
    }
    # execute_experiment binds step via a function-local
    # `from gpuwm.core.dycore import step` (model.py) resolved at call
    # time, so gpuwm.core.dycore.step is the interceptable name --
    # gpuwm.core.model has no `step` attribute at all.
    original_step = production_dycore.step

    def summary_step(state, cfg, *args, **kwargs):
        """Production step plus runtime-identical every-substep accumulation."""
        nonlocal summary_wall_seconds, summary_cpu_seconds
        result = original_step(state, cfg, *args, **kwargs)
        summary_wall_start = time.perf_counter()
        summary_cpu_start = time.thread_time()
        report = stability_report(
            state, cfg, boundary_width=cfg.spec_bdy_width)
        domain = f"d{cfg.grid_id:02d}"
        summary = run_summaries[domain]
        summary["boundary_w_max_ms"] = max(
            float(summary["boundary_w_max_ms"]),
            float(report["boundary_w_max"]))
        summary["interior_w_max_ms"] = max(
            float(summary["interior_w_max_ms"]),
            float(report["interior_w_max"]))
        summary["dynamics_substeps"] = int(summary["dynamics_substeps"]) + 1
        summary_cpu_seconds += time.thread_time() - summary_cpu_start
        summary_wall_seconds += time.perf_counter() - summary_wall_start
        return result

    with PerDomainWrfoutWriters(
            model, output_dir, start_time=exp.start_time,
            title=case_data.output_title) as writers:
        model._io_manager = writers
        # Host-overhead profiling excludes input decode/tree construction.
        # It covers the op-table walk, launch trains, alarms, validation, and
        # any output back-pressure experienced by the production executor.
        wall_start, cpu_start = time.perf_counter(), time.thread_time()

        def history_handler(tree, node, ticks):
            nonlocal state_hash_wall_seconds, state_hash_cpu_seconds
            from gpuwm.io.wrfout import wrfout_filename

            domain = f"d{node.cfg.grid_id:02d}"
            hash_wall_start = time.perf_counter()
            hash_cpu_start = time.thread_time()
            state_hash = canonical_state_digest(node.state, node.clock)
            state_hash_cpu_seconds += time.thread_time() - hash_cpu_start
            state_hash_wall_seconds += time.perf_counter() - hash_wall_start
            inventory = validate_inventory_growth(
                prior_state_inventories.get(domain), state_hash["inventory"],
                domain=domain, ticks=int(ticks))
            prior_state_inventories[domain] = inventory
            valid = exp.start_time + timedelta(
                seconds=ticks / tree.schedule.clock.tick_den)
            frame = wrfout_filename(valid, node.cfg.grid_id)
            sample = {
                **state_hash, "domain": domain, "ticks": int(ticks),
                "tick_den": int(tree.schedule.clock.tick_den),
                "valid_seconds": float(
                    ticks / tree.schedule.clock.tick_den),
                "frame": frame,
            }
            state_hashes.append(sample)
            state_inventories.setdefault(domain, []).append({
                "frame": frame, "ticks": int(ticks),
                "tick_den": int(tree.schedule.clock.tick_den),
                "valid_seconds": sample["valid_seconds"],
                "inventory": inventory,
            })
            _submit_tree_history_frame(writers, node, ticks)

        def restart_handler(tree, ticks):
            nonlocal last_checkpoint
            writers.drain()
            for checkpoint_node in tree.walk_parent_first():
                checkpoint_domain = f"d{checkpoint_node.cfg.grid_id:02d}"
                checkpoint_hash = canonical_state_digest(
                    checkpoint_node.state, checkpoint_node.clock)
                checkpoint_inventory = validate_inventory_growth(
                    prior_state_inventories.get(checkpoint_domain),
                    checkpoint_hash["inventory"],
                    domain=checkpoint_domain, ticks=int(ticks))
                prior_state_inventories[checkpoint_domain] = (
                    checkpoint_inventory)
                checkpoint_state_hashes.append({
                    **checkpoint_hash,
                    "domain": checkpoint_domain,
                    "ticks": int(ticks),
                    "tick_den": int(tree.schedule.clock.tick_den),
                    "valid_seconds": float(
                        ticks / tree.schedule.clock.tick_den),
                    "frame": f"checkpoint@{int(ticks)}",
                })
            valid = exp.start_time + timedelta(
                seconds=ticks / tree.schedule.clock.tick_den)
            last_checkpoint = write_tree_restart(output_dir, tree, valid)
            tree._last_checkpoint = last_checkpoint
            if (stop_at_checkpoint_seconds is not None
                    and ticks == stop_at_checkpoint_seconds
                    * tree.schedule.clock.tick_den):
                raise _StopAtCheckpoint(last_checkpoint)

        def progress_callback(**_kwargs):
            nonlocal used_peak, held_peak
            used_peak = max(used_peak, int(pool.used_bytes()))
            held_peak = max(held_peak, int(pool.total_bytes()))

        production_dycore.step = summary_step
        try:
            try:
                execution_report = production_model.execute_experiment(
                    model, history_handler=history_handler,
                    restart_handler=restart_handler,
                    progress_callback=progress_callback)
            except _StopAtCheckpoint as signal:
                last_checkpoint = signal.checkpoint
                stopped = True
        finally:
            production_dycore.step = original_step
        if not stopped:
            writers.drain()
        paths = writers.paths
    cp.cuda.runtime.deviceSynchronize()
    used_peak = max(used_peak, int(pool.used_bytes()))
    held_peak = max(held_peak, int(pool.total_bytes()))
    wall_seconds = time.perf_counter() - wall_start
    # Main-thread CPU excludes the asynchronous netCDF worker threads and is
    # the closest direct measurement of the registered Python orchestration
    # surface (op walk, alarms, launch wrappers).
    host_seconds = time.thread_time() - cpu_start
    profiled_wall_seconds = max(
        0.0, wall_seconds - state_hash_wall_seconds - summary_wall_seconds)
    profiled_host_seconds = max(
        0.0, host_seconds - state_hash_cpu_seconds - summary_cpu_seconds)
    clocks = {
        f"d{node.cfg.grid_id:02d}": {
            "ticks": int(node.clock.ticks),
            "step_count": int(node.clock.step_count),
            "tick_den": int(node.clock.tick_den),
        } for node in model.walk_parent_first()
    }
    for summary in run_summaries.values():
        boundary_max = float(summary["boundary_w_max_ms"])
        interior_max = float(summary["interior_w_max_ms"])
        summary["boundary_zone_blowup"] = bool(
            not math.isfinite(boundary_max)
            or boundary_max > 5.0 * max(interior_max, 1.0))
    prefix_provenance = {
        str(count): experiment_prefix_provenance(
            exp, model._input_catalog, count)
        for count in range(1, len(exp.domains) + 1)
    }
    execution = {
        "stopped_at_checkpoint": stopped,
        "evaluator_commit": _git_commit(),
        "experiment_fingerprint": model.experiment_fingerprint,
        "experiment_prefix_provenance": prefix_provenance,
        "clocks": clocks,
        "steps": None if execution_report is None else execution_report.steps,
        "forces": None if execution_report is None else execution_report.forces,
        "feedback_calls": (None if execution_report is None
                           else execution_report.feedback_calls),
        "histories": (None if execution_report is None
                      else dict(execution_report.histories)),
        "canonical_state_inventories": state_inventories,
        "canonical_resume_inventories": resume_inventories,
        "canonical_state_hashes": state_hashes,
        "canonical_checkpoint_state_hashes": checkpoint_state_hashes,
        "run_summaries": run_summaries,
    }
    return ProductionRun(
        output_dir=output_dir, wrfout_paths=paths,
        checkpoint=last_checkpoint,
        completed_seconds=float(model.root.clock.elapsed_seconds),
        execution=execution,
        timing={
            "wall_seconds": wall_seconds,
            "profiled_wall_seconds": profiled_wall_seconds,
            "python_main_thread_seconds": profiled_host_seconds,
            "canonical_hash_wall_seconds": state_hash_wall_seconds,
            "canonical_hash_cpu_seconds": state_hash_cpu_seconds,
            "run_summary_wall_seconds": summary_wall_seconds,
            "run_summary_cpu_seconds": summary_cpu_seconds,
            "host_overhead_fraction": (
                profiled_host_seconds / profiled_wall_seconds
                if profiled_wall_seconds > 0.0 else float("nan")),
        },
        memory={
            "pool_used_peak_bytes": used_peak,
            "pool_held_peak_bytes": held_peak,
            "alloc_estimate_bytes":
                int(model.memory_ledger.estimate.alloc_estimate_bytes),
            "free_before_bytes": int(free_before),
        })


def _inventory_relation(
        candidate: Mapping[str, object], baseline: Mapping[str, object]
        ) -> tuple[str, bool, list[str]]:
    """Classify a restart-boundary inventory difference and its legality."""
    candidate_set = _inventory_member_set(candidate)
    baseline_set = _inventory_member_set(baseline)
    if candidate_set == baseline_set:
        return "equal", True, []
    candidate_by_name = {item[0]: item[1:] for item in candidate_set}
    baseline_by_name = {item[0]: item[1:] for item in baseline_set}
    changed = sorted(
        name for name in set(candidate_by_name) & set(baseline_by_name)
        if candidate_by_name[name] != baseline_by_name[name])
    differing_names = sorted(
        set(candidate_by_name) ^ set(baseline_by_name))
    compatible = (not changed and all(
        restart_boundary_member_class(name) is not None
        for name in differing_names))
    if candidate_set < baseline_set:
        relation = "subset"
    elif candidate_set > baseline_set:
        relation = "superset"
    else:
        relation = "mixed"
        compatible = False
    return relation, bool(compatible), sorted((*changed, *differing_names))


def restart_inventory_convergence(
        straight_samples: Iterable[Mapping[str, object]],
        prefix_samples: Iterable[Mapping[str, object]],
        resumed_samples: Iterable[Mapping[str, object]], *,
        domain: str, resume_inventory: Mapping[str, object],
        split_ticks: int, next_synchronized_ticks: int,
        straight_split_sample: Mapping[str, object] | None = None,
        prefix_split_sample: Mapping[str, object] | None = None,
        ) -> dict[str, object]:
    """Apply the REBUILT-table restart contract to one domain's hashes.

    Prefix samples remain an exact arm of the straight run.  At restore the
    inventory may be an allowlisted subset or superset only for concrete
    REBUILT frame-state members.  Every serialized scratch, driver, field, and
    W0AVG member must already be exactly present.  A post-resume hash is
    compared only after exact inventory equality, which must occur no later
    than the next all-domain synchronized history frame.
    """
    straight = tuple(
        item for item in _normalize_state_hashes(straight_samples)
        if item["domain"] == domain)
    prefix = tuple(
        item for item in _normalize_state_hashes(prefix_samples)
        if item["domain"] == domain and int(item["ticks"]) <= split_ticks)
    resumed = tuple(
        item for item in _normalize_state_hashes(resumed_samples)
        if item["domain"] == domain and int(item["ticks"]) > split_ticks)
    straight_pre = tuple(
        item for item in straight if int(item["ticks"]) <= split_ticks)
    straight_post = tuple(
        item for item in straight if int(item["ticks"]) > split_ticks)
    pre_split = compare_state_hash_samples(prefix, straight_pre, domain)

    baseline_by_tick = {int(item["ticks"]): item for item in straight_post}
    resumed_by_tick = {int(item["ticks"]): item for item in resumed}
    tick_inventory_equal = (
        len(baseline_by_tick) == len(straight_post)
        and len(resumed_by_tick) == len(resumed)
        and set(baseline_by_tick) == set(resumed_by_tick))

    split_baseline = (straight_split_sample if straight_split_sample is not None
                      else next((item for item in straight
                                 if int(item["ticks"]) == split_ticks), None))
    checkpoint_comparison = None
    if straight_split_sample is not None and prefix_split_sample is not None:
        checkpoint_comparison = _compare_state_hash_pair(
            prefix_split_sample, straight_split_sample)
    elif straight_split_sample is not None or prefix_split_sample is not None:
        checkpoint_comparison = {
            "passed": False, "metadata_equal": False,
            "inventory_equal": False, "hash_compared": False,
            "hash_equal": None,
            "reason": "one restart arm lacks the split checkpoint sample",
        }
    else:
        checkpoint_comparison = {
            "passed": False, "metadata_equal": False,
            "inventory_equal": False, "hash_compared": False,
            "hash_equal": None,
            "reason": "both restart arms lack the split checkpoint sample",
        }
    resume_raw = resume_inventory.get("inventory", resume_inventory)
    resume_normalized = _normalize_inventory(resume_raw)
    resume_metadata_equal = True
    if "inventory" in resume_inventory:
        resume_metadata_equal = bool(
            int(resume_inventory.get("ticks", -1)) == split_ticks
            and split_baseline is not None
            and int(resume_inventory.get("tick_den", -1))
            == int(split_baseline["tick_den"])
            and float(resume_inventory.get("valid_seconds", -1.0))
            == split_ticks / int(split_baseline["tick_den"]))
    if split_baseline is None:
        resume_relation, resume_compatible, resume_difference = (
            "missing-baseline", False, [])
    else:
        resume_relation, resume_compatible, resume_difference = (
            _inventory_relation(
                resume_normalized, split_baseline["inventory"]))

    rows = []
    first_equal_ticks = None
    post_passed = tick_inventory_equal
    for ticks in sorted(set(baseline_by_tick) | set(resumed_by_tick)):
        candidate = resumed_by_tick.get(ticks)
        baseline = baseline_by_tick.get(ticks)
        if candidate is None or baseline is None:
            row = {
                "domain": domain, "ticks": ticks, "passed": False,
                "metadata_equal": False, "inventory_equal": False,
                "inventory_difference_allowlisted": False,
                "hash_compared": False, "hash_equal": None,
                "reason": "post-resume state-hash sample missing",
            }
        else:
            exact = _compare_state_hash_pair(candidate, baseline)
            if exact["inventory_equal"]:
                if first_equal_ticks is None:
                    first_equal_ticks = ticks
                row = dict(exact)
                row["inventory_difference_allowlisted"] = True
            else:
                _relation, compatible, differing = _inventory_relation(
                    candidate["inventory"], baseline["inventory"])
                before_convergence = first_equal_ticks is None
                row = {
                    **exact,
                    "passed": bool(
                        exact["metadata_equal"] and compatible
                        and before_convergence
                        and ticks <= next_synchronized_ticks),
                    "inventory_difference_allowlisted": compatible,
                    "differing_members": differing,
                }
            row.update({
                "domain": domain, "frame": candidate["frame"],
                "ticks": ticks,
            })
        post_passed &= bool(row["passed"])
        rows.append(row)

    converged_by_next_sync = bool(
        first_equal_ticks is not None
        and first_equal_ticks <= next_synchronized_ticks)
    passed = bool(
        pre_split["passed"] and resume_metadata_equal
        and resume_compatible and post_passed
        and converged_by_next_sync
        and checkpoint_comparison["passed"])
    return {
        "passed": passed, "domain": domain,
        "split_ticks": int(split_ticks),
        "next_synchronized_ticks": int(next_synchronized_ticks),
        "pre_split": pre_split,
        "split_checkpoint": checkpoint_comparison,
        "resume_metadata_equal": resume_metadata_equal,
        "resume_inventory_relation": resume_relation,
        "resume_inventory_difference_allowlisted": resume_compatible,
        "resume_inventory_differing_members": resume_difference,
        "post_resume_tick_inventory_equal": tick_inventory_equal,
        "first_inventory_equal_ticks": first_equal_ticks,
        "converged_by_next_synchronized_frame": converged_by_next_sync,
        "post_resume_samples": rows,
    }


def _restart_state_comparisons(exp, straight: ProductionRun,
                               prefix: ProductionRun,
                               resumed: ProductionRun
                               ) -> dict[str, object]:
    from gpuwm.core.clock import resolve_clock

    clock = resolve_clock(exp)
    split_ticks = RESTART_SPLIT_SECONDS * int(clock.tick_den)
    history_ticks = [
        int(dc.history_interval_s) * int(clock.tick_den)
        for dc in exp.domains]
    synchronized_period_ticks = math.lcm(*history_ticks)
    next_synchronized_ticks = (
        (split_ticks // synchronized_period_ticks) + 1
        ) * synchronized_period_ticks
    straight_hashes = straight.execution.get("canonical_state_hashes", ())
    prefix_hashes = prefix.execution.get("canonical_state_hashes", ())
    resumed_hashes = resumed.execution.get("canonical_state_hashes", ())
    straight_checkpoints = straight.execution.get(
        "canonical_checkpoint_state_hashes", ())
    prefix_checkpoints = prefix.execution.get(
        "canonical_checkpoint_state_hashes", ())
    resume_inventories = resumed.execution.get(
        "canonical_resume_inventories", {})
    if (not isinstance(straight_hashes, list)
            or not isinstance(prefix_hashes, list)
            or not isinstance(resumed_hashes, list)
            or not isinstance(straight_checkpoints, list)
            or not isinstance(prefix_checkpoints, list)
            or not isinstance(resume_inventories, Mapping)):
        return {"passed": False,
                "reason": "restart arms lack canonical inventory ledgers"}
    domains = {}
    passed = True
    for dc in exp.domains:
        domain = f"d{dc.grid_id:02d}"
        resume_inventory = resume_inventories.get(domain)
        if not isinstance(resume_inventory, Mapping):
            row = {"passed": False, "domain": domain,
                   "reason": "resumed arm lacks resume-instant inventory"}
        else:
            straight_split_sample = next((
                item for item in straight_checkpoints
                if item.get("domain") == domain
                and int(item.get("ticks", -1)) == split_ticks), None)
            prefix_split_sample = next((
                item for item in prefix_checkpoints
                if item.get("domain") == domain
                and int(item.get("ticks", -1)) == split_ticks), None)
            row = restart_inventory_convergence(
                straight_hashes, prefix_hashes, resumed_hashes,
                domain=domain, resume_inventory=resume_inventory,
                split_ticks=split_ticks,
                next_synchronized_ticks=next_synchronized_ticks,
                straight_split_sample=straight_split_sample,
                prefix_split_sample=prefix_split_sample)
        passed &= bool(row["passed"])
        domains[domain] = row
    return {
        "passed": bool(passed), "split_ticks": split_ticks,
        "next_synchronized_ticks": next_synchronized_ticks,
        "domains": domains,
    }


def restart_split_stage(
        exp, case_data, *, straight: ProductionRun, work_dir: str | Path,
        executor: Callable[..., ProductionRun] = execute_production_run
        ) -> dict[str, object]:
    """30-min checkpoint then 45-min continuation vs the straight run."""
    work_dir = Path(work_dir)
    prefix = executor(
        exp, case_data, work_dir / "prefix",
        stop_at_checkpoint_seconds=RESTART_SPLIT_SECONDS)
    if prefix.checkpoint is None:
        return {"passed": False, "reason": "prefix produced no checkpoint"}
    resumed = executor(
        exp, case_data, work_dir / "resumed", restart=prefix.checkpoint)
    state_comparisons = _restart_state_comparisons(
        exp, straight, prefix, resumed)
    valid = exp.start_time + timedelta(seconds=RUN_SECONDS)
    comparisons = {}
    passed = bool(state_comparisons["passed"])
    for dc in exp.domains:
        domain = f"d{dc.grid_id:02d}"
        baseline_path_list = list(straight.paths_for_domain(dc.grid_id))
        split_path_list = [
            *prefix.paths_for_domain(dc.grid_id),
            *resumed.paths_for_domain(dc.grid_id)]
        baseline_paths = {path.name: path for path in baseline_path_list}
        split_paths = {path.name: path for path in split_path_list}
        inventory_equal = (
            len(baseline_paths) == len(baseline_path_list)
            and len(split_paths) == len(split_path_list)
            and set(baseline_paths) == set(split_paths))
        rows = []
        # Compare every emitted split-arm file, not merely every unique name;
        # a mistakenly duplicated resume-boundary frame is both compared and
        # fails the inventory check above.
        for split_path in sorted(split_path_list, key=lambda path: path.name):
            name = split_path.name
            comparison = compare_files_exact(
                split_path,
                baseline_paths.get(name, Path("__missing_straight__")))
            comparison["frame"] = name
            rows.append(comparison)
            passed &= bool(comparison["passed"])
        for name in sorted(set(baseline_paths) - set(split_paths)):
            comparison = compare_files_exact(
                Path("__missing_split__"), baseline_paths[name])
            comparison["frame"] = name
            rows.append(comparison)
            passed &= bool(comparison["passed"])
        passed &= inventory_equal
        comparisons[domain] = {
            "passed": bool(inventory_equal and all(
                row["passed"] for row in rows)),
            "inventory_equal": inventory_equal, "frames": rows,
        }
    return {
        "passed": bool(passed), "split_seconds": RESTART_SPLIT_SECONDS,
        "resume_seconds": RUN_SECONDS - RESTART_SPLIT_SECONDS,
        "checkpoint": str(prefix.checkpoint), "final_valid_time": str(valid),
        "comparisons": comparisons,
        "state_inventory_comparisons": state_comparisons,
    }


def alloc_check_stage(exp) -> dict[str, object]:
    """Execute the controller-only two-domain N0-contract allocation hook."""
    from gpuwm.core.preflight import run_alloc_preflight
    report = run_alloc_preflight(exp)
    return {
        "passed": bool(report.passed),
        "pool_used_peak_bytes": report.pool_used_peak_bytes,
        "alloc_estimate_bytes": report.estimate.alloc_estimate_bytes,
        "free_before_bytes": report.free_before_bytes,
        "reserve_bytes": report.reserve.reserve_bytes,
        "gates": dict(report.gates),
    }


@dataclass(frozen=True)
class RatchetArtifact:
    domain: str
    frame: str
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class RatchetManifest:
    schema: int
    rung: str
    evaluator_commit: str
    experiment_fingerprint: str
    domain_ids: tuple[int, ...]
    tick_start: int
    tick_stop: int
    tick_den: int
    created_utc: str
    artifacts: tuple[RatchetArtifact, ...]
    expected_state_domains: tuple[str, ...]
    state_hashes: tuple[dict[str, object], ...] = ()
    state_hash_schedules: tuple[dict[str, object], ...] = ()


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an exact integer")
    return int(value)


def _exact_seconds(value: object, label: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an exact finite number")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} must be an exact finite number") from exc


def _state_sample_instant(sample: Mapping[str, object]) -> Fraction:
    ticks = _strict_int(sample["ticks"], "canonical state ticks")
    tick_den = _strict_int(sample["tick_den"], "canonical state tick_den")
    if ticks < 0 or tick_den <= 0:
        raise ValueError("canonical state sample has an invalid exact tick pair")
    instant = Fraction(ticks, tick_den)
    if _exact_seconds(
            sample["valid_seconds"], "canonical state valid_seconds") != instant:
        raise ValueError("canonical state valid_seconds disagrees with exact ticks")
    return instant


def _instant_evidence(instant: Fraction) -> dict[str, int]:
    return {
        "instant_numerator": int(instant.numerator),
        "instant_denominator": int(instant.denominator),
    }


def _normalize_sample_scalars(
        raw: object, scalar_sha256: object, *, identity: tuple[str, str],
        instant: Fraction) -> tuple[dict[str, object], str]:
    if not isinstance(raw, Mapping) or set(raw) != {
            "elapsed_seconds", "dtbc_fp32_bits", "driver"}:
        raise ValueError(f"canonical state scalars are malformed for {identity}")
    elapsed = float(raw["elapsed_seconds"])
    if _exact_seconds(
            raw["elapsed_seconds"], "canonical scalar elapsed_seconds") != instant:
        raise ValueError(
            f"canonical scalar elapsed_seconds differs for {identity}")
    dtbc_fp32_bits = _strict_int(
        raw["dtbc_fp32_bits"], "canonical scalar dtbc_fp32_bits")
    if not 0 <= dtbc_fp32_bits <= 0xFFFFFFFF:
        raise ValueError(f"canonical dtbc_fp32_bits is invalid for {identity}")
    driver_raw = raw["driver"]
    if driver_raw is None:
        driver = None
    else:
        if not isinstance(driver_raw, Mapping) or set(driver_raw) != {
                "call_counts", "ysu_nan_guard_fires", "microphysics_updates"}:
            raise ValueError(f"canonical driver scalars are malformed for {identity}")
        call_counts_raw = driver_raw["call_counts"]
        if not isinstance(call_counts_raw, Mapping):
            raise ValueError(f"canonical driver call counts are malformed for {identity}")
        driver = {
            "call_counts": {
                str(key): _strict_int(
                    value, f"canonical driver call count {key!r}")
                for key, value in sorted(call_counts_raw.items())},
            "ysu_nan_guard_fires": _strict_int(
                driver_raw["ysu_nan_guard_fires"],
                "canonical ysu_nan_guard_fires"),
            "microphysics_updates": _strict_int(
                driver_raw["microphysics_updates"],
                "canonical microphysics_updates"),
        }
    scalars = {
        "elapsed_seconds": elapsed,
        "dtbc_fp32_bits": dtbc_fp32_bits,
        "driver": driver,
    }
    expected_sha256 = hashlib.sha256(
        _canonical_scalar_bytes(scalars)).hexdigest()
    if str(scalar_sha256) != expected_sha256:
        raise ValueError(f"canonical scalar digest differs for {identity}")
    return scalars, expected_sha256


def _normalize_state_hashes(
        samples: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    normalized = tuple(dict(sample) for sample in samples)
    identities = set()
    for sample in normalized:
        missing = {"domain", "frame", "ticks", "tick_den", "valid_seconds",
                   "schema", "sha256", "inventory_sha256", "array_count",
                   "field_order", "inventory", "scalar_sha256",
                   "scalars"} - set(sample)
        if missing:
            raise ValueError(
                f"canonical state-hash sample lacks {sorted(missing)}")
        identity = (str(sample["domain"]), str(sample["frame"]))
        if identity in identities:
            raise ValueError(f"duplicate canonical state sample {identity}")
        identities.add(identity)
        instant = _state_sample_instant(sample)
        for key in ("sha256", "inventory_sha256", "scalar_sha256"):
            value = str(sample[key])
            if len(value) != 64 or any(ch not in "0123456789abcdef"
                                       for ch in value):
                raise ValueError(f"invalid {key} for state sample {identity}")
        inventory = _normalize_inventory(sample["inventory"])
        field_order = [item["name"] for item in inventory["members"]]
        if (sample["schema"] != CANONICAL_STATE_SCHEMA
                or str(sample["inventory_sha256"]) != inventory["sha256"]
                or int(sample["array_count"]) != inventory["array_count"]
                or list(sample["field_order"]) != field_order):
            raise ValueError(
                f"canonical state inventory envelope differs for {identity}")
        sample["inventory"] = inventory
        sample["inventory_sha256"] = inventory["sha256"]
        sample["array_count"] = inventory["array_count"]
        sample["field_order"] = field_order
        scalars, scalar_sha256 = _normalize_sample_scalars(
            sample["scalars"], sample["scalar_sha256"],
            identity=identity, instant=instant)
        sample["scalars"] = scalars
        sample["scalar_sha256"] = scalar_sha256
    return tuple(sorted(
        normalized,
        key=lambda item: (str(item["domain"]), _state_sample_instant(item),
                          str(item["frame"]))))


def _compare_state_hash_pair(
        candidate: Mapping[str, object], baseline: Mapping[str, object]
        ) -> dict[str, object]:
    """Compare inventories first; a hash is never consulted on mismatch."""
    candidate_instant = _state_sample_instant(candidate)
    baseline_instant = _state_sample_instant(baseline)
    metadata_equal = (
        str(candidate["domain"]) == str(baseline["domain"])
        and str(candidate["frame"]) == str(baseline["frame"])
        and candidate_instant == baseline_instant)
    candidate_inventory = _normalize_inventory(candidate["inventory"])
    baseline_inventory = _normalize_inventory(baseline["inventory"])
    inventory_equal = candidate_inventory == baseline_inventory
    candidate_scalars, candidate_scalar_sha256 = _normalize_sample_scalars(
        candidate["scalars"], candidate["scalar_sha256"],
        identity=(str(candidate["domain"]), str(candidate["frame"])),
        instant=candidate_instant)
    baseline_scalars, baseline_scalar_sha256 = _normalize_sample_scalars(
        baseline["scalars"], baseline["scalar_sha256"],
        identity=(str(baseline["domain"]), str(baseline["frame"])),
        instant=baseline_instant)
    scalars_equal = bool(
        candidate_scalars == baseline_scalars
        and candidate_scalar_sha256 == baseline_scalar_sha256)
    hash_compared = bool(metadata_equal and inventory_equal and scalars_equal)
    hash_equal = (str(candidate["sha256"]) == str(baseline["sha256"])
                  if hash_compared else None)
    return {
        "passed": bool(hash_compared and hash_equal),
        "metadata_equal": bool(metadata_equal),
        "inventory_equal": bool(inventory_equal),
        "scalars_equal": scalars_equal,
        "candidate_inventory_sha256": candidate_inventory["sha256"],
        "baseline_inventory_sha256": baseline_inventory["sha256"],
        "candidate_scalar_sha256": candidate_scalar_sha256,
        "baseline_scalar_sha256": baseline_scalar_sha256,
        "hash_compared": hash_compared,
        "hash_equal": hash_equal,
        "candidate_sha256": str(candidate["sha256"]),
        "baseline_sha256": str(baseline["sha256"]),
    }


def compare_state_hash_samples(
        candidates: Iterable[Mapping[str, object]],
        baselines: Iterable[Mapping[str, object]], domain: str
        ) -> dict[str, object]:
    """Exact cross-run ratchet comparison keyed by rational instants."""
    candidates = tuple(
        item for item in _normalize_state_hashes(candidates)
        if item["domain"] == domain)
    baselines = tuple(
        item for item in _normalize_state_hashes(baselines)
        if item["domain"] == domain)
    candidate_by_instant = {
        _state_sample_instant(item): item for item in candidates}
    baseline_by_instant = {
        _state_sample_instant(item): item for item in baselines}
    duplicate_candidates = len(candidate_by_instant) != len(candidates)
    duplicate_baselines = len(baseline_by_instant) != len(baselines)
    inventory_equal = (
        bool(candidates) and bool(baselines)
        and not duplicate_candidates and not duplicate_baselines
        and set(candidate_by_instant) == set(baseline_by_instant))
    passed = inventory_equal
    rows = []
    all_instants = sorted(set(candidate_by_instant) | set(baseline_by_instant))
    for instant in all_instants:
        candidate = candidate_by_instant.get(instant)
        baseline = baseline_by_instant.get(instant)
        if candidate is None or baseline is None:
            row = {
                **_instant_evidence(instant), "passed": False,
                "metadata_equal": False, "inventory_equal": False,
                "scalars_equal": False,
                "hash_compared": False, "hash_equal": None,
                "reason": "state-hash sample missing at exact instant",
            }
        else:
            row = _compare_state_hash_pair(candidate, baseline)
            row.update({
                "domain": domain, "frame": candidate["frame"],
                "ticks": int(candidate["ticks"]),
                "tick_den": int(candidate["tick_den"]),
                "baseline_ticks": int(baseline["ticks"]),
                "baseline_tick_den": int(baseline["tick_den"]),
                **_instant_evidence(instant),
            })
        passed &= bool(row["passed"])
        rows.append(row)
    return {
        "passed": bool(passed), "domain": domain,
        "tick_inventory_equal": bool(inventory_equal),
        "instant_inventory_equal": bool(inventory_equal),
        "candidate_only_ticks": sorted(
            int(candidate_by_instant[instant]["ticks"])
            for instant in set(candidate_by_instant) - set(baseline_by_instant)),
        "baseline_only_ticks": sorted(
            int(baseline_by_instant[instant]["ticks"])
            for instant in set(baseline_by_instant) - set(candidate_by_instant)),
        "candidate_only_instants": [
            _instant_evidence(instant)
            for instant in sorted(
                set(candidate_by_instant) - set(baseline_by_instant))],
        "baseline_only_instants": [
            _instant_evidence(instant)
            for instant in sorted(
                set(baseline_by_instant) - set(candidate_by_instant))],
        "expected_samples": len(baselines), "samples": rows,
    }


def state_hash_schedule(exp, domain: str) -> dict[str, object]:
    """Return the authoritative history-sample schedule for one domain."""
    from gpuwm.core.clock import resolve_clock

    grid_id = int(domain.removeprefix("d"))
    dc = exp.domain(grid_id)
    clock = resolve_clock(exp)
    history_interval = Fraction(str(dc.history_interval_s))
    history_interval_ticks = history_interval * int(clock.tick_den)
    if history_interval_ticks.denominator != 1:
        raise ValueError(f"{domain} history interval is off the exact tick lattice")
    period = int(history_interval_ticks)
    expected_samples = len(range(0, int(clock.run_ticks) + 1, period))
    return {
        "domain": domain,
        "tick_start": 0,
        "tick_stop": int(clock.run_ticks),
        "tick_den": int(clock.tick_den),
        "history_interval_ticks": period,
        "expected_samples": expected_samples,
    }


def _normalize_state_hash_schedules(
        schedules: Iterable[Mapping[str, object]]
        ) -> tuple[dict[str, object], ...]:
    normalized = []
    domains = set()
    for raw in schedules:
        if not isinstance(raw, Mapping) or set(raw) != {
                "domain", "tick_start", "tick_stop", "tick_den",
                "history_interval_ticks", "expected_samples"}:
            raise ValueError("canonical state-hash schedule is malformed")
        domain = str(raw["domain"])
        if domain in domains:
            raise ValueError(f"duplicate canonical state-hash schedule {domain}")
        domains.add(domain)
        tick_start = _strict_int(raw["tick_start"], f"{domain} tick_start")
        tick_stop = _strict_int(raw["tick_stop"], f"{domain} tick_stop")
        tick_den = _strict_int(raw["tick_den"], f"{domain} tick_den")
        period = _strict_int(
            raw["history_interval_ticks"],
            f"{domain} history_interval_ticks")
        expected_samples = _strict_int(
            raw["expected_samples"], f"{domain} expected_samples")
        if (tick_start < 0 or tick_stop < tick_start
                or tick_den <= 0 or period <= 0):
            raise ValueError(f"canonical state-hash schedule is invalid for {domain}")
        computed_samples = len(range(tick_start, tick_stop + 1, period))
        if expected_samples != computed_samples or expected_samples <= 0:
            raise ValueError(
                f"canonical state-hash expected sample count differs for {domain}")
        normalized.append({
            "domain": domain,
            "tick_start": tick_start,
            "tick_stop": tick_stop,
            "tick_den": tick_den,
            "history_interval_ticks": period,
            "expected_samples": expected_samples,
        })
    return tuple(sorted(normalized, key=lambda item: str(item["domain"])))


def _validate_state_hash_evidence(
        state_hashes: Iterable[Mapping[str, object]],
        state_hash_schedules: Iterable[Mapping[str, object]]
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...]]:
    hashes = _normalize_state_hashes(state_hashes)
    schedules = _normalize_state_hash_schedules(state_hash_schedules)
    if not hashes or not schedules:
        raise ValueError("schema-3 ratchet requires canonical state evidence")
    samples_by_domain = {
        domain: tuple(item for item in hashes if item["domain"] == domain)
        for domain in {str(item["domain"]) for item in hashes}}
    schedules_by_domain = {
        str(item["domain"]): item for item in schedules}
    if set(samples_by_domain) != set(schedules_by_domain):
        raise ValueError(
            "canonical state evidence domains differ from their schedules")
    for domain, schedule in schedules_by_domain.items():
        samples = samples_by_domain[domain]
        expected_ticks = tuple(range(
            int(schedule["tick_start"]), int(schedule["tick_stop"]) + 1,
            int(schedule["history_interval_ticks"])))
        actual_ticks = tuple(int(item["ticks"]) for item in samples)
        own_clock = all(
            int(item["tick_den"]) == int(schedule["tick_den"])
            for item in samples)
        if (not own_clock or actual_ticks != expected_ticks
                or len(samples) != int(schedule["expected_samples"])):
            raise ValueError(
                f"canonical state evidence is incomplete for {domain}: "
                f"expected ticks {list(expected_ticks)}, got {list(actual_ticks)}")
    return hashes, schedules


def _normalize_expected_state_domains(
        domains: Iterable[str]) -> tuple[str, ...]:
    if isinstance(domains, (str, bytes)):
        raise ValueError("expected state-evidence domains must be an inventory")
    try:
        normalized = tuple(str(domain) for domain in domains)
    except TypeError as exc:
        raise ValueError(
            "expected state-evidence domains must be an inventory") from exc
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(
            "expected state-evidence domain inventory is empty or duplicated")
    return tuple(sorted(normalized))


def _validate_role_state_domains(
        rung: str, domains: Iterable[str]) -> tuple[str, ...]:
    declared = _normalize_expected_state_domains(domains)
    expected = RATCHET_STATE_EVIDENCE_DOMAINS.get(rung)
    if expected is None:
        raise ValueError(f"unsupported ratchet publication role {rung}")
    if declared != expected:
        raise ValueError(
            f"{rung} ratchet expected state-evidence domain inventory differs "
            f"from its immutable role contract: expected {list(expected)}, "
            f"got {list(declared)}")
    return declared


def _validate_bound_state_hash_evidence(
        state_hashes: Iterable[Mapping[str, object]],
        state_hash_schedules: Iterable[Mapping[str, object]],
        provenance: Mapping[str, object],
        artifact_frames: Mapping[str, Sequence[str]],
        expected_state_domains: Iterable[str],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...]]:
    hashes, schedules = _validate_state_hash_evidence(
        state_hashes, state_hash_schedules)
    if not artifact_frames:
        raise ValueError("schema-3 ratchet requires scheduled frame artifacts")
    schedule_by_domain = {
        str(item["domain"]): item for item in schedules}
    expected_domains = set(_normalize_expected_state_domains(
        expected_state_domains))
    if set(schedule_by_domain) != expected_domains:
        raise ValueError(
            "canonical state-evidence domain inventory differs from the "
            f"declared exact inventory: expected {sorted(expected_domains)}, "
            f"got {sorted(schedule_by_domain)}")
    allowed_domains = {
        f"d{int(grid_id):02d}" for grid_id in provenance["domain_ids"]}
    if not expected_domains <= allowed_domains:
        raise ValueError("canonical state schedule contains an unbound domain")
    binding_start = Fraction(
        int(provenance["tick_start"]), int(provenance["tick_den"]))
    binding_stop = Fraction(
        int(provenance["tick_stop"]), int(provenance["tick_den"]))
    for domain, schedule in schedule_by_domain.items():
        own_start = Fraction(
            int(schedule["tick_start"]), int(schedule["tick_den"]))
        own_stop = Fraction(
            int(schedule["tick_stop"]), int(schedule["tick_den"]))
        if own_start != binding_start or own_stop != binding_stop:
            raise ValueError(
                f"canonical state schedule bounds differ for {domain}")
    samples_by_domain = {
        domain: tuple(item for item in hashes if item["domain"] == domain)
        for domain in schedule_by_domain}
    for domain, raw_frames in artifact_frames.items():
        frames = tuple(str(frame) for frame in raw_frames)
        if domain not in schedule_by_domain:
            raise ValueError(
                f"schema-3 ratchet has no {domain} state-hash schedule")
        sample_frames = tuple(
            str(item["frame"]) for item in samples_by_domain[domain])
        if (len(set(frames)) != len(frames)
                or len(set(sample_frames)) != len(sample_frames)
                or set(frames) != set(sample_frames)):
            raise ValueError(
                f"schema-3 ratchet {domain} artifacts differ from its exact "
                "scheduled state evidence")
    return hashes, schedules


def publish_ratchet(
        root: str | Path, rung: str, domain: str,
        paths: Iterable[str | Path], *,
        provenance: Mapping[str, object],
        expected_state_domains: Iterable[str],
        state_hashes: Iterable[Mapping[str, object]],
        state_hash_schedules: Iterable[Mapping[str, object]]) -> Path:
    """Copy immutable rung bytes under ``out/rungs`` with an explicit manifest."""
    root, paths = Path(root), tuple(Path(path) for path in paths)
    provenance = _normalize_ratchet_provenance(provenance)
    expected_state_domains = _validate_role_state_domains(
        rung, expected_state_domains)
    state_hashes, state_hash_schedules = _validate_bound_state_hash_evidence(
        state_hashes, state_hash_schedules, provenance,
        {domain: tuple(path.name for path in paths)}, expected_state_domains)
    evaluator_commit = _git_commit()
    if provenance["evaluator_commit"] != evaluator_commit:
        raise ValueError(
            "ratchet publication execution commit differs from evaluator")
    rung_dir = root / rung
    manifest_path = rung_dir / "manifest.json"
    if manifest_path.exists():
        existing = load_ratchet(root, rung)
        existing_artifacts = tuple(
            item for item in existing.artifacts if item.domain == domain)
        if not existing_artifacts:
            raise FileExistsError(
                f"immutable {rung} ratchet has no {domain} frames")
        expected_names = tuple(path.name for path in paths)
        actual_names = tuple(Path(item.relative_path).name
                             for item in existing_artifacts)
        if expected_names != actual_names:
            raise FileExistsError(
                f"immutable {rung} ratchet already exists with other frames")
        for source, item in zip(paths, existing_artifacts, strict=True):
            if sha256_file(source) != item.sha256:
                raise FileExistsError(
                    f"immutable {rung} ratchet differs for {source.name}")
        if stable_hash(state_hashes) != stable_hash(existing.state_hashes):
            raise FileExistsError(
                f"immutable {rung} canonical-state controls differ")
        if stable_hash(state_hash_schedules) != stable_hash(
                existing.state_hash_schedules):
            raise FileExistsError(
                f"immutable {rung} canonical-state schedules differ")
        if expected_state_domains != existing.expected_state_domains:
            raise FileExistsError(
                f"immutable {rung} state-evidence domain inventory differs")
        existing_provenance = _manifest_provenance(existing)
        if existing_provenance != provenance:
            raise FileExistsError(
                f"immutable {rung} ratchet provenance differs")
        return manifest_path
    artifacts = []
    target_dir = rung_dir / domain
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = target_dir / source.name
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(source, target)
        artifacts.append(RatchetArtifact(
            domain=domain, frame=source.name,
            relative_path=target.relative_to(rung_dir).as_posix(),
            bytes=target.stat().st_size, sha256=sha256_file(target)))
    manifest = RatchetManifest(
        schema=3, rung=rung, evaluator_commit=evaluator_commit,
        experiment_fingerprint=str(provenance["experiment_fingerprint"]),
        domain_ids=tuple(provenance["domain_ids"]),
        tick_start=int(provenance["tick_start"]),
        tick_stop=int(provenance["tick_stop"]),
        tick_den=int(provenance["tick_den"]),
        created_utc=_utc_now(), artifacts=tuple(artifacts),
        expected_state_domains=expected_state_domains,
        state_hashes=state_hashes,
        state_hash_schedules=state_hash_schedules)
    write_json(manifest_path, {
        **{key: value for key, value in asdict(manifest).items()
           if key not in {
               "artifacts", "state_hashes", "state_hash_schedules"}},
        "artifacts": [asdict(item) for item in manifest.artifacts],
        "state_hashes": list(manifest.state_hashes),
        "state_hash_schedules": list(manifest.state_hash_schedules),
    })
    return manifest_path


def load_ratchet(root: str | Path, rung: str, *, domain: str | None = None
                 ) -> RatchetManifest:
    root = Path(root)
    manifest_path = root / rung / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != 3 or payload.get("rung") != rung:
        raise ValueError(f"invalid {rung} ratchet manifest identity")
    artifacts = tuple(RatchetArtifact(**item) for item in payload["artifacts"])
    provenance = _normalize_ratchet_provenance({
        key: payload[key]
        for key in ("evaluator_commit", "experiment_fingerprint", "domain_ids",
                    "tick_start", "tick_stop", "tick_den")})
    artifact_frames = {
        artifact_domain: tuple(
            item.frame for item in artifacts if item.domain == artifact_domain)
        for artifact_domain in {item.domain for item in artifacts}}
    expected_state_domains = _validate_role_state_domains(
        rung, payload.get("expected_state_domains", ()))
    state_hashes, state_hash_schedules = _validate_bound_state_hash_evidence(
        payload.get("state_hashes", ()),
        payload.get("state_hash_schedules", ()), provenance, artifact_frames,
        expected_state_domains)
    for item in artifacts:
        path = root / rung / item.relative_path
        if (not path.is_file() or path.stat().st_size != item.bytes
                or sha256_file(path) != item.sha256):
            raise ValueError(f"{rung} ratchet artifact failed hash: {path}")
    manifest = RatchetManifest(
        schema=3, rung=rung, evaluator_commit=payload["evaluator_commit"],
        experiment_fingerprint=payload["experiment_fingerprint"],
        domain_ids=tuple(int(item) for item in payload["domain_ids"]),
        tick_start=int(payload["tick_start"]),
        tick_stop=int(payload["tick_stop"]), tick_den=int(payload["tick_den"]),
        created_utc=payload["created_utc"], artifacts=artifacts,
        expected_state_domains=expected_state_domains,
        state_hashes=state_hashes,
        state_hash_schedules=state_hash_schedules)
    _manifest_provenance(manifest)
    if domain is None:
        return manifest
    domain_artifacts = tuple(
        item for item in manifest.artifacts if item.domain == domain)
    domain_hashes = tuple(
        item for item in manifest.state_hashes if item["domain"] == domain)
    domain_schedules = tuple(
        item for item in manifest.state_hash_schedules
        if item["domain"] == domain)
    if not domain_artifacts:
        raise ValueError(f"{rung} ratchet has no {domain} artifacts")
    if not domain_hashes or not domain_schedules:
        raise ValueError(f"{rung} ratchet has no {domain} canonical state evidence")
    return replace(
        manifest, artifacts=domain_artifacts, state_hashes=domain_hashes,
        state_hash_schedules=domain_schedules)


def _normalize_ratchet_provenance(
        provenance: Mapping[str, object]) -> dict[str, object]:
    required = {"evaluator_commit", "experiment_fingerprint", "domain_ids",
                "tick_start", "tick_stop", "tick_den"}
    missing = required - set(provenance)
    if missing:
        raise ValueError(f"ratchet provenance lacks {sorted(missing)}")
    fingerprint = str(provenance["experiment_fingerprint"])
    if (len(fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in fingerprint)):
        raise ValueError("ratchet provenance has invalid experiment fingerprint")
    domain_ids = tuple(int(item) for item in provenance["domain_ids"])
    if not domain_ids or domain_ids != tuple(range(1, len(domain_ids) + 1)):
        raise ValueError("ratchet provenance domain ids are not a root prefix")
    tick_start = int(provenance["tick_start"])
    tick_stop = int(provenance["tick_stop"])
    tick_den = int(provenance["tick_den"])
    if tick_start < 0 or tick_stop <= tick_start or tick_den <= 0:
        raise ValueError("ratchet provenance has invalid tick range")
    evaluator_commit = provenance["evaluator_commit"]
    if not isinstance(evaluator_commit, str) or not evaluator_commit:
        raise ValueError("ratchet provenance has invalid evaluator commit")
    return {
        "evaluator_commit": evaluator_commit,
        "experiment_fingerprint": fingerprint,
        "domain_ids": domain_ids,
        "tick_start": tick_start, "tick_stop": tick_stop,
        "tick_den": tick_den,
    }


def _manifest_provenance(manifest: RatchetManifest) -> dict[str, object]:
    return _normalize_ratchet_provenance({
        "evaluator_commit": manifest.evaluator_commit,
        "experiment_fingerprint": manifest.experiment_fingerprint,
        "domain_ids": manifest.domain_ids,
        "tick_start": manifest.tick_start,
        "tick_stop": manifest.tick_stop,
        "tick_den": manifest.tick_den,
    })


def validate_ratchet_provenance(
        manifest: RatchetManifest, expected: Mapping[str, object]) -> None:
    """Refuse a binding consumer on commit, config, or tick-range drift."""
    normalized = dict(expected)
    normalized.setdefault("evaluator_commit", _git_commit())
    normalized = _normalize_ratchet_provenance(normalized)
    current_commit = _git_commit()
    if normalized["evaluator_commit"] != current_commit:
        raise ValueError(
            "current run evaluator commit differs from the active evaluator")
    if _manifest_provenance(manifest) != normalized:
        raise ValueError(
            f"{manifest.rung} ratchet provenance mismatch with current run")


def compare_ratchet_frames(
        candidates: Iterable[str | Path], root: str | Path,
        manifest: RatchetManifest, domain: str, *,
        candidate_state_hashes: Iterable[Mapping[str, object]]
        ) -> dict[str, object]:
    """Compare all frame bytes and mandatory exact-instant state hashes."""
    candidate_paths = tuple(Path(path) for path in candidates)
    candidate_by_name = {path.name: path for path in candidate_paths}
    artifacts = tuple(item for item in manifest.artifacts if item.domain == domain)
    artifact_by_name = {item.frame: item for item in artifacts}
    duplicate_candidates = len(candidate_by_name) != len(candidate_paths)
    duplicate_artifacts = len(artifact_by_name) != len(artifacts)
    inventory_equal = (
        not duplicate_candidates and not duplicate_artifacts
        and set(candidate_by_name) == set(artifact_by_name))
    rows = []
    passed = inventory_equal
    for name in sorted(set(candidate_by_name) | set(artifact_by_name)):
        item = artifact_by_name.get(name)
        baseline = (Path(root) / manifest.rung / item.relative_path
                    if item is not None else Path("__missing_ratchet__"))
        comparison = compare_files_exact(
            candidate_by_name.get(name, Path("__missing_candidate__")),
            baseline)
        comparison["frame"] = name
        rows.append(comparison)
        passed &= bool(comparison["passed"])
    _validate_role_state_domains(
        manifest.rung, manifest.expected_state_domains)
    evidence_domains = {
        str(item["domain"]) for item in manifest.state_hash_schedules}
    validation_domains = (
        manifest.expected_state_domains
        if evidence_domains == set(manifest.expected_state_domains)
        else (domain,))
    validated_hashes, validated_schedules = _validate_bound_state_hash_evidence(
        manifest.state_hashes, manifest.state_hash_schedules,
        _manifest_provenance(manifest),
        {domain: tuple(item.frame for item in artifacts)}, validation_domains)
    baseline_state_hashes = tuple(
        item for item in validated_hashes if item["domain"] == domain)
    if not any(
            item["domain"] == domain for item in validated_schedules):
        raise ValueError(
            f"{manifest.rung} ratchet has no {domain} canonical state schedule")
    state_comparison = compare_state_hash_samples(
        candidate_state_hashes, baseline_state_hashes, domain)
    passed &= bool(state_comparison["passed"])
    return {
        "passed": bool(passed), "domain": domain,
        "manifest_rung": manifest.rung,
        "manifest_evaluator_commit": manifest.evaluator_commit,
        "manifest_experiment_fingerprint": manifest.experiment_fingerprint,
        "inventory_equal": inventory_equal,
        "candidate_only": sorted(set(candidate_by_name)-set(artifact_by_name)),
        "ratchet_only": sorted(set(artifact_by_name)-set(candidate_by_name)),
        "expected_frames": len(artifacts), "frames": rows,
        "state_hashes": state_comparison,
        "baseline_inventory_sha256": stable_hash(
            [(item.frame, item.sha256) for item in artifacts]),
    }


def ratchet_frame(root: str | Path, rung: str, domain: str,
                  frame: str) -> Path:
    manifest = load_ratchet(root, rung, domain=domain)
    for item in manifest.artifacts:
        if item.frame == frame:
            return Path(root) / rung / item.relative_path
    raise KeyError(f"{rung} ratchet has no {domain} frame {frame}")


def evaluate_n3(
        exp, case_data, run: ProductionRun, *, phase4_root: str | Path,
        verdicts: Mapping[str, object], restart_evidence: Mapping[str, object],
        alloc_evidence: Mapping[str, object],
        ancestor_control_execution: Mapping[str, object] | None = None
        ) -> dict[str, object]:
    """Evaluate all and only the registered N3 records."""
    valid_13z = exp.start_time + timedelta(hours=1)
    valid_1315 = exp.start_time + timedelta(seconds=RUN_SECONDS)
    d01 = frame_path(run.output_dir, 1, valid_13z)
    d02 = frame_path(run.output_dir, 2, valid_1315)
    phase4 = frame_path(phase4_root, 1, valid_13z)
    reference = child_reference_path(case_data, "d02")
    results: dict[str, dict[str, object]] = {}

    record = _registered_gate("N3", "d01_bitwise_vs_phase4_13z")
    comparison = compare_d01_phase4_frame(d01, phase4)
    results[record.metric] = gate_result(
        record, passed=bool(comparison["passed"]), evidence=comparison)
    results.update(score_statistical_frame(
        "N3", "d02", d02, reference, dx_m=exp.domain(2).run.dx,
        run_summary=run_summary_for_domain(run, "d02"),
        verdicts=verdicts,
        fss_reference=matched_reference_path(case_data, "d02")))

    static = output_static_recheck(
        d02, reference, spec_bdy_width=exp.spec_bdy_width,
        blend_width=exp.blend_width)
    for metric, value in (
            ("d02_hgt_blend_recheck_m", static["hgt_m"]),
            ("d02_mub_blend_recheck_pa", static["mub_pa"])):
        record = _registered_gate("N3", metric)
        results[metric] = gate_result(record, value=value)

    metric = "d02_blend_zone_t2_tsk_bias"
    record = _registered_gate("N3", metric)
    results[metric] = gate_result(
        record, evidence=blend_zone_surface_bias(
            d02, reference, spec_bdy_width=exp.spec_bdy_width,
            blend_width=exp.blend_width))

    metric = "restart_split_bit_identity"
    record = _registered_gate("N3", metric)
    results[metric] = gate_result(
        record, passed=bool(restart_evidence.get("passed")),
        evidence=restart_evidence)
    metric = "two_domain_alloc_check"
    record = _registered_gate("N3", metric)
    results[metric] = gate_result(
        record, passed=bool(alloc_evidence.get("passed")),
        evidence=alloc_evidence)

    if tuple(results) != N3_METRICS:
        raise RuntimeError(
            f"N3 evaluator inventory drifted: {tuple(results)} != {N3_METRICS}")
    gate_records("N3", results)  # fail loudly if any local name is unregistered
    return {
        "schema": 1, "rung": "N3", "generated_utc": _utc_now(),
        "evaluator_commit": _git_commit(),
        "config_source": str(PRODUCTION_CONFIG),
        "domain_ids": [dc.grid_id for dc in exp.domains],
        "run_seconds": exp.run_seconds,
        "production_execution": {
            "output_dir": str(run.output_dir),
            "completed_seconds": run.completed_seconds,
            "execution": dict(run.execution),
            "timing": dict(run.timing), "memory": dict(run.memory),
        },
        "ancestor_control_execution": dict(ancestor_control_execution or {}),
        "gates": list(results.values()),
        "passed": report_passed(results.values()),
    }


def run_n3(args) -> dict[str, object]:
    exp, data = construct_rung_case(2, config_path=args.config)
    control_exp, control_data = construct_rung_case(
        1, config_path=args.config)
    if args.existing_report is not None:
        with args.existing_report.open("r", encoding="utf-8") as stream:
            prior = json.load(stream)
        straight = production_run_from_report(prior)
        alloc = gate_evidence_from_report(prior, "two_domain_alloc_check")
        restart = gate_evidence_from_report(prior, "restart_split_bit_identity")
        control_execution = prior.get("ancestor_control_execution")
        if not isinstance(control_execution, dict):
            raise ValueError(
                "existing N3 report lacks the d01 no-younger control")
    else:
        alloc = ({"passed": False, "reason": "allocation hook disabled"}
                 if args.no_alloc else alloc_check_stage(exp))
        straight = execute_production_run(exp, data, args.outdir / "straight")
        control = execute_production_run(
            control_exp, control_data, args.outdir / "ancestor-control-d01")
        control_execution = {
            "output_dir": str(control.output_dir),
            "completed_seconds": control.completed_seconds,
            "execution": dict(control.execution),
            "timing": dict(control.timing), "memory": dict(control.memory),
        }
        restart = ({"passed": False, "reason": "restart split disabled"}
                   if args.no_restart_split else restart_split_stage(
                       exp, data, straight=straight,
                       work_dir=args.outdir / "restart-split"))
    report = evaluate_n3(
        exp, data, straight, phase4_root=args.phase4_root,
        verdicts=load_verdicts(args.verdicts),
        restart_evidence=restart, alloc_evidence=alloc,
        ancestor_control_execution=control_execution)
    report["config_source"] = str(args.config.resolve())
    report_path = write_json(args.outdir / "N3-report.json", report)
    report["report_path"] = str(report_path)
    if report["passed"]:
        control_inventory = state_hash_inventory(
            control_exp, control_execution["execution"], "d01")
        d02_inventory = state_hash_inventory(exp, straight.execution, "d02")
        if not control_inventory["passed"] or not d02_inventory["passed"]:
            raise RuntimeError(
                "N3 canonical-state control inventory is incomplete")
        control_hashes = state_hashes_for_domain(
            control_execution["execution"], "d01")
        d02_hashes = state_hashes_for_domain(straight.execution, "d02")
        ratchet = publish_ratchet(
            args.ratchet_root, "N3", "d02", straight.paths_for_domain(2),
            provenance=run_prefix_provenance(straight, 2),
            expected_state_domains=RATCHET_STATE_EVIDENCE_DOMAINS["N3"],
            state_hashes=(*control_hashes, *d02_hashes),
            state_hash_schedules=(
                state_hash_schedule(control_exp, "d01"),
                state_hash_schedule(exp, "d02")))
        report["ratchet_manifest"] = str(ratchet)
        report["ratchet_state_inventories"] = {
            "d01": control_inventory, "d02": d02_inventory}
        write_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.verify.cases.real74_d02",
        description="Controller-only real74 N3 production rung")
    parser.add_argument("--config", type=Path, default=PRODUCTION_CONFIG)
    parser.add_argument("--outdir", type=Path,
                        default=REPOSITORY_ROOT / "out" / "rungs" / "N3-run")
    parser.add_argument("--ratchet-root", type=Path,
                        default=REPOSITORY_ROOT / "out" / "rungs")
    parser.add_argument("--phase4-root", type=Path,
                        default=default_phase4_root())
    parser.add_argument("--verdicts", type=Path,
                        help="JSON structural verdicts; missing verdicts fail")
    parser.add_argument(
        "--existing-report", type=Path,
        help="re-evaluate existing run/restart/alloc evidence without a GPU run")
    parser.add_argument("--no-alloc", action="store_true",
                        help="development only; makes the alloc gate fail")
    parser.add_argument("--no-restart-split", action="store_true",
                        help="development only; makes restart identity fail")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_n3(args)
    print(json.dumps(json_safe(report), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0 if report["passed"] else 1


__all__ = [
    "CANONICAL_INVENTORY_SCHEMA", "CANONICAL_LAZY_MEMBER_CLASSES",
    "CANONICAL_STATE_SCHEMA", "D01_PHASE4_RATIFIED_EXCEPTIONS",
    "N3_METRICS", "PRODUCTION_CONFIG", "RATCHET_STATE_EVIDENCE_DOMAINS",
    "ProductionRun", "RatchetArtifact",
    "RatchetManifest", "alloc_check_stage", "blend_zone_surface_bias",
    "boundary_zone_blowup_value", "canonical_state_digest",
    "child_reference_path", "compare_d01_phase4_frame",
    "compare_files_exact", "compare_ratchet_frames",
    "compare_state_hash_samples", "construct_rung_case",
    "ensemble_envelope_adjudication", "evaluate_n3",
    "experiment_prefix_provenance",
    "execute_production_run", "fractions_skill_score", "gate_records",
    "gate_evidence_from_report", "gate_result", "load_ratchet",
    "json_safe", "production_run_from_report", "publish_ratchet", "ratchet_frame",
    "lazy_inventory_member_class", "report_passed",
    "restart_boundary_member_class", "restart_inventory_convergence",
    "restart_split_stage",
    "score_statistical_frame",
    "resolve_experiment_prefix_provenance", "run_prefix_provenance",
    "run_summary_for_domain", "sha256_file", "stable_hash",
    "state_hash_inventory", "state_hash_schedule", "state_hashes_for_domain",
    "validate_inventory_growth",
    "validate_ratchet_provenance", "write_json",
]


if __name__ == "__main__":  # pragma: no cover - controller entry point
    raise SystemExit(main())
