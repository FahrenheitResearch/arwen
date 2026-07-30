#!/usr/bin/env python3
"""Run a hash-bound RW-WPS prepared hierarchy through GPUWM.

The public source adapters already publish one verified prepared cache per
domain.  This launcher restores that complete tree and hands it to GPUWM's
existing ``DomainNode``/``NestCoupler``/``execute_experiment`` engine.  It
does not implement another integrator and it does not flatten a nest tree
into the single-domain benchmark runner.

The experiment TOML remains the typed authority for arbitrary supported
static one-way layouts and per-domain physics.  In particular, MP8 outer
domains may feed MP18 inner domains through the existing explicit
``mp8-to-mp18-mass-diagnosed-v1`` policy.  Implemented configurations that do
not yet have retained acceptance evidence are reported as warnings, never as
consent gates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gpuwm import __version__  # noqa: E402
from gpuwm.core.microphysics_transition import (  # noqa: E402
    MP8_TO_MP18_POLICY,
    resolve_microphysics_transition,
)
from gpuwm.experiment import load_experiment  # noqa: E402
from gpuwm.ingest.prepared_cache import PreparedCacheReader  # noqa: E402
from gpuwm.native_wrf_contract import (  # noqa: E402
    NATIVE_LANDUSE_IDENTITY,
    load_native_static_cache,
    verify_native_static_receipt,
)
from gpuwm.static.lambert import grids_from_projection_config  # noqa: E402


REPORT_SCHEMA = "gpuwm-prepared-domain-tree-forecast-v1"
PROGRESS_SCHEMA = "gpuwm-prepared-domain-tree-progress-v1"
CAPABILITIES_SCHEMA = "gpuwm-runner-capabilities-v1"
PLAN_SCHEMA = "gpuwm-prepared-domain-tree-plan-v1"
RUNNER = "tools.prepared_domain_tree_forecast"
ARBITRARY_PLAN_ID = "arbitrary-prepared-one-way-domain-tree-v1"
THOMPSON_NSSL_PLAN_ID = "thompson-outer-nssl2-inner-mp8-mp18-v1"
HIERARCHY_SCHEMA = "gpuwm-native-hrrr-hierarchy-direct-v1"
# Every source builds its whole domain tree through the same artifact writer,
# so `hierarchy-artifacts/` is identical across all of them. Only the top-level
# document differs: HRRR prepares its tree in a separate pass and writes
# receipt.json, while the namelist-driven sources build theirs inside RW-WPS
# preparation and write proof.json. Reading both is what makes nested execution
# a property of the topology rather than of the source.
_HIERARCHY_DOCUMENTS = (
    {
        "filename": "receipt.json",
        "schema": HIERARCHY_SCHEMA,
        "status": "PASS",
        "source": "hrrr",
    },
    {
        "filename": "proof.json",
        "schema": "gpuwm-era5-native-hierarchy-proof-v1",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "era5",
    },
    {
        "filename": "proof.json",
        "schema": "gpuwm-gfs-native-hierarchy-proof-v1",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "gfs",
    },
    {
        "filename": "proof.json",
        "schema": "gpuwm-mapped-native-hierarchy-proof-v1",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "20crv3",
    },
)
SUPPORTED_SOURCES = tuple(
    dict.fromkeys(entry["source"] for entry in _HIERARCHY_DOCUMENTS)
)
ARTIFACT_RECEIPT_SCHEMA = "gpuwm-native-hierarchy-artifact-build-v1"
ARTIFACT_MANIFEST_SCHEMA = "gpuwm-native-domain-artifacts-v1"
_HEX = frozenset("0123456789abcdef")
_FORECAST_EXECUTOR_MODULES = (
    "gpuwm.core.clock",
    "gpuwm.core.dycore",
    "gpuwm.core.health",
    "gpuwm.core.model",
    "gpuwm.core.nest",
    "gpuwm.io.restart",
    "gpuwm.io.wrfout",
    "gpuwm.state_digest",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _strict_json(value):
    if isinstance(value, Mapping):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        # v1.1 nests gave DomainConfig an optional per-domain start_time,
        # so a serialized domain config now reaches here carrying a
        # datetime.  ISO 8601 is what every other identity document in
        # the tree already writes (prepared_cache, source_hierarchy,
        # native_domain_artifacts), and identity digests only agree if
        # this agrees with them.
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def _atomic_json(path: Path, payload) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    encoded = (
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value


def _digest(value: str, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in _HEX for char in normalized):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return normalized


def _require_file(path: Path, label: str) -> Path:
    result = Path(path).resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _require_directory(path: Path, label: str) -> Path:
    result = Path(path).resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _sibling_outdir(protected: Path) -> Path:
    """A concrete --outdir the guard below will accept, beside ``protected``.

    Named in the refusal so it reads as an instruction rather than a
    rule.  It matches what the front door now suggests, so the two
    surfaces send the user to the same directory.
    """
    protected = Path(protected)
    return protected.parent / f"{protected.name}-forecast"


def claim_output_directory(output: Path, *, protected_roots: tuple[Path, ...]) -> Path:
    """Create exactly one output directory without adopting old content."""

    result = Path(output).resolve()
    for protected in protected_roots:
        protected = Path(protected).resolve()
        if _inside(result, protected) or _inside(protected, result):
            raise ValueError(
                f"output directory {result} overlaps protected input "
                f"{protected}; the forecast may not write into its own "
                f"inputs.  Pass an --outdir beside them instead, for "
                f"example {_sibling_outdir(protected)}"
            )
    result.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.mkdir()
    except FileExistsError:
        raise FileExistsError(f"refusing existing output directory: {result}") from None
    return result


def _missing_executor_modules() -> list[str]:
    missing = []
    for module in _FORECAST_EXECUTOR_MODULES:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            present = False
        if not present:
            missing.append(module)
    return missing


def runner_capabilities() -> dict[str, object]:
    """Side-effect-free launcher contract for Studio and headless clients."""

    missing = _missing_executor_modules()
    available = not missing
    return {
        "schema": CAPABILITIES_SCHEMA,
        "runner": RUNNER,
        "supported_sources": list(SUPPORTED_SOURCES) if available else [],
        "readiness": (
            "IMPLEMENTED_RUNTIME_PREFLIGHT_REQUIRED_UNVERIFIED"
            if available
            else "FORECAST_EXECUTOR_OMITTED"
        ),
        "modes": {
            "forecast": {
                "available": available,
                "requires_cupy": True,
                "requires_compatible_cuda_gpu": True,
                "missing_executor_modules": missing,
                "included_in_standalone_rw_wps_wheel": False,
            },
        },
        "simulation_plan_ids": [
            ARBITRARY_PLAN_ID,
            THOMPSON_NSSL_PLAN_ID,
        ]
        if available
        else [],
        "simulation_plans": {
            ARBITRARY_PLAN_ID: {
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "topology": "arbitrary-engine-valid-static-one-way-tree",
                "physics": "per-domain-engine-valid-selectors",
                "validity_authority": "gpuwm.experiment.load_experiment",
                "geometry_whitelist": False,
            },
            THOMPSON_NSSL_PLAN_ID: {
                "readiness": "IMPLEMENTED_UNVERIFIED",
                "explicit_expert_consent_required": False,
                "selectors": {
                    "outer_domains": 8,
                    "inner_domains": 18,
                    "transition": MP8_TO_MP18_POLICY,
                },
                "canonical_four_domain_binding": {
                    "d01": {"mp_physics": 8},
                    "d02": {"mp_physics": 8},
                    "d03": {
                        "mp_physics": 18,
                        "nest_microphysics_transition": MP8_TO_MP18_POLICY,
                    },
                    "d04": {
                        "mp_physics": 18,
                        "nest_microphysics_transition": "same-scheme-only",
                    },
                },
            },
        }
        if available
        else {},
        "warning_policy": {
            "implemented_unverified_is_launchable": True,
            "consent_gate": False,
            "warnings_and_receipts_retained": True,
            "hard_errors": [
                "malformed-or-incompatible-plan",
                "missing-or-mutated-authority",
                "missing-runtime-asset",
                "device-allocation-or-execution-failure",
            ],
        },
        "input": {
            "layout": "rw-wps hierarchy-artifacts/domain-artifacts.json",
            "hash_pins": [
                "preparation-receipt-sha256",
                "experiment-config-sha256",
            ],
            "every_domain_cache_reread_and_sha256_verified": True,
        },
        "output": {
            "directory_policy": "create-only",
            "io_modes": ["history", "none"],
            "history_cadence": "per-domain experiment TOML",
            "restart": "experiment-configured-tree-checkpoints",
        },
        "capability_query": {
            "flag": "--show-capabilities",
            "side_effect_free": True,
            "requires_cupy": False,
            "validates_inputs_or_runtime_assets": False,
        },
    }


@dataclass(frozen=True)
class PreparedDomainBundle:
    grid_id: int
    parent_id: int
    bundle: Path
    cache: Path
    static_path: Path
    geometry_receipt_path: Path
    domain_receipt_path: Path
    cache_reader: PreparedCacheReader
    cache_identity: Mapping[str, object]
    static_fields: Mapping[str, np.ndarray]
    authority_sha256: Mapping[str, str]


@dataclass(frozen=True)
class PreparedTreeInputs:
    prepared_root: Path
    hierarchy_root: Path
    preparation_receipt_path: Path
    artifact_receipt_path: Path
    artifact_manifest_path: Path
    experiment_config: Path
    experiment: object
    grids: tuple[object, ...]
    domains: tuple[PreparedDomainBundle, ...]
    forcing_hours: tuple[int, ...]
    boundary_interval_seconds: int
    source_identity: Mapping[str, object]
    execution_plan: Mapping[str, object]
    authority_sha256: Mapping[str, str]
    source: str


def _domain_rows(exp) -> list[dict[str, object]]:
    return [
        {
            "grid_id": int(domain.grid_id),
            "parent_id": int(domain.parent_id),
            "i_parent_start": int(domain.i_parent_start),
            "j_parent_start": int(domain.j_parent_start),
            "parent_grid_ratio": int(domain.parent_grid_ratio),
            "parent_time_step_ratio": int(domain.parent_time_step_ratio),
            "nx": int(domain.run.nx),
            "ny": int(domain.run.ny),
            "nz": int(domain.run.nz),
            "dx_m": float(domain.run.dx),
            "dy_m": float(domain.run.dy),
            "dt_s": float(domain.run.dt),
            "history_interval_s": float(domain.history_interval_s),
            "mp_physics": int(domain.run.mp_physics),
            "moist": bool(domain.run.moist),
            "moist_cq": bool(domain.run.moist_cq),
            "nest_microphysics_transition": str(
                domain.run.nest_microphysics_transition
            ),
        }
        for domain in exp.domains
    ]


def resolve_execution_plan(exp) -> Mapping[str, object]:
    """Resolve every edge through the engine's actual transition authority."""

    if int(exp.feedback) != 0:
        raise ValueError(
            "the prepared domain-tree forecast product carries a static "
            "one-way execution plan and refuses feedback=1; use the native "
            "experiment runner for experimental two-way feedback")
    by_id = {domain.grid_id: domain for domain in exp.domains}
    transitions = []
    for domain in exp.domains:
        if domain.parent_id == 0:
            continue
        parent = by_id[domain.parent_id]
        contract = resolve_microphysics_transition(parent.run, domain.run)
        transitions.append(
            {
                "source_domain": int(parent.grid_id),
                "target_domain": int(domain.grid_id),
                **dict(contract.receipt()),
            }
        )

    canonical_mixed = (
        len(exp.domains) == 4
        and [int(domain.parent_id) for domain in exp.domains] == [0, 1, 2, 3]
        and [int(domain.run.mp_physics) for domain in exp.domains] == [8, 8, 18, 18]
        and [str(domain.run.nest_microphysics_transition) for domain in exp.domains]
        == [
            "same-scheme-only",
            "same-scheme-only",
            MP8_TO_MP18_POLICY,
            "same-scheme-only",
        ]
    )
    mixed = [edge for edge in transitions if edge["mixed"]]
    warnings = [
        "The prepared domain-tree GPU route is implemented but does not yet "
        "have a retained public end-to-end HRRR acceptance run for this "
        "exact topology, source cycle, and physics selection."
    ]
    if mixed:
        warnings.append(
            "Mixed per-domain microphysics is a GPUWM extension and is not "
            "deterministically equivalent to stock WRF, which normalizes "
            "domains to one microphysics selector."
        )
    return MappingProxyType(
        {
            "schema": PLAN_SCHEMA,
            "plan_id": (
                THOMPSON_NSSL_PLAN_ID if canonical_mixed else ARBITRARY_PLAN_ID
            ),
            "status": "IMPLEMENTED_UNVERIFIED",
            "launch_allowed": True,
            "explicit_expert_consent_required": False,
            "domain_count": len(exp.domains),
            "domains": _domain_rows(exp),
            "transitions": transitions,
            "mixed_transition_count": len(mixed),
            # This receipt describes only the edge translation semantics.  It
            # must never be read as a claim that an otherwise arbitrary GPUWM
            # experiment is a certified stock-WRF trajectory.
            "microphysics_edges_stock_wrf_equivalent": not mixed,
            "whole_simulation_stock_wrf_certified": False,
            "warnings": warnings,
        }
    )


def _load_hierarchy_document(prepared_root: Path, expected_sha256: str):
    """Resolve the prepared root's top-level document, whichever source wrote it.

    Returns ``(path, document, source)``.  The digest is still pinned exactly;
    only which filename carries it varies.
    """

    seen = []
    for entry in _HIERARCHY_DOCUMENTS:
        candidate = prepared_root / entry["filename"]
        if not candidate.is_file():
            continue
        if _sha256(candidate) != expected_sha256:
            seen.append(f"{entry['filename']} digest differs")
            continue
        document = _json_object(candidate, "preparation document")
        if document.get("schema") != entry["schema"]:
            seen.append(f"{entry['filename']} schema {document.get('schema')!r}")
            continue
        if document.get("status") != entry["status"]:
            raise ValueError(
                f"prepared root {entry['filename']} is not a "
                f"{entry['status']} {entry['source']} hierarchy"
            )
        return candidate, document, entry["source"]
    raise ValueError(
        "prepared root carries no recognized hierarchy document matching "
        "--preparation-receipt-sha256; looked for "
        + ", ".join(sorted({e["filename"] for e in _HIERARCHY_DOCUMENTS}))
        + (f" ({'; '.join(seen)})" if seen else "")
    )


def _hierarchy_valid_time(document) -> str:
    """The initialization time, however this source's document records it."""

    value = document.get("valid_time")
    if isinstance(value, str):
        return value
    times = document.get("forcing_times")
    if isinstance(times, list) and times and isinstance(times[0], str):
        return times[0]
    raise ValueError("preparation document declares no initialization time")


def _forcing_hours(receipt, identity) -> tuple[int, ...]:
    raw = receipt.get("forcing_hours")
    if raw is None:
        raw = identity.get("forcing_hours")
    if (
        not isinstance(raw, list)
        or len(raw) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        or raw[0] != 0
        or any(later <= earlier for earlier, later in zip(raw, raw[1:]))
    ):
        raise ValueError(
            "prepared hierarchy forcing_hours must be increasing integers "
            "beginning at zero with at least two frames"
        )
    result = tuple(raw)
    deltas = {later - earlier for earlier, later in zip(result, result[1:])}
    if len(deltas) != 1:
        raise ValueError("prepared hierarchy forcing cadence is not uniform")
    return result


def _validate_domain_receipt(
    receipt,
    *,
    domain,
    bundle: Path,
    reader: PreparedCacheReader,
    static_path: Path,
    geometry_path: Path,
) -> None:
    expected_identity = {
        "schema": "gpuwm-native-domain-artifact-build-v1",
        "status": "READY",
        "grid_id": int(domain.grid_id),
        "parent_id": int(domain.parent_id),
        "boundary_mode": (
            "external-specified" if domain.parent_id == 0 else "nested-parent-forced"
        ),
    }
    if any(receipt.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"d{domain.grid_id:02d} artifact receipt identity differs")
    artifacts = receipt.get("artifacts")
    with np.load(static_path, allow_pickle=False) as archive:
        static_fields = sorted(archive.files)
    expected = {
        "prepared_cache": {
            "path": "prepared-cache",
            "content_sha256": reader.content_sha256,
            "payload_bytes": reader.payload_bytes,
            "array_count": len(reader.arrays),
        },
        "static_cache": {
            "path": "native-static.npz",
            "bytes": static_path.stat().st_size,
            "sha256": _sha256(static_path),
            "fields": static_fields,
        },
        "geometry_receipt": {
            "path": "geometry-receipt.json",
            "sha256": _sha256(geometry_path),
            "geometry": _json_object(geometry_path, "geometry receipt").get("geometry"),
        },
    }
    if artifacts != expected:
        raise ValueError(f"d{domain.grid_id:02d} artifact hashes differ from its files")
    verification = receipt.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("status") != "PASS"
        or verification.get("content_sha256") != reader.content_sha256
        or verification.get("array_count") != len(reader.arrays)
        or verification.get("payload_bytes") != reader.payload_bytes
    ):
        raise ValueError(f"d{domain.grid_id:02d} cache verification receipt differs")
    if Path(verification.get("path", "")).name != "prepared-cache":
        raise ValueError(f"d{domain.grid_id:02d} cache receipt path is not relocatable")
    if bundle.resolve() != geometry_path.parent.resolve():
        raise RuntimeError("domain artifact path escaped its bundle")


def _validate_vertical(reader: PreparedCacheReader, exp, grid_id: int) -> None:
    metadata = reader.header.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"d{grid_id:02d} cache metadata is missing")
    eta = tuple(float(value) for value in exp.vertical.eta_levels)
    observed = reader.read_array("coord/znw")
    if observed.shape != (len(eta),) or not np.array_equal(
        observed.astype(np.float64), np.asarray(eta, dtype=np.float64)
    ):
        raise ValueError(
            f"d{grid_id:02d} prepared eta coordinate differs from the experiment config"
        )
    base = metadata.get("base_scalars")
    if not isinstance(base, dict) or float(base.get("p_top", -1.0)) != float(
        exp.vertical.p_top
    ):
        raise ValueError(f"d{grid_id:02d} prepared p_top differs from the experiment")


def preflight_prepared_tree(
    *,
    prepared_root: Path,
    preparation_receipt_sha256: str,
    experiment_config: Path,
    experiment_config_sha256: str,
) -> PreparedTreeInputs:
    """Verify the complete hierarchy and resolve a runnable CPU-only plan."""

    preparation_receipt_sha256 = _digest(
        preparation_receipt_sha256, "preparation-receipt-sha256"
    )
    experiment_config_sha256 = _digest(
        experiment_config_sha256, "experiment-config-sha256"
    )
    prepared_root = _require_directory(prepared_root, "prepared root")
    experiment_config = _require_file(experiment_config, "experiment config")
    receipt_path, preparation, prepared_source = _load_hierarchy_document(
        prepared_root, preparation_receipt_sha256
    )
    if _sha256(experiment_config) != experiment_config_sha256:
        raise ValueError("experiment config differs from --experiment-config-sha256")

    exp = load_experiment(experiment_config)
    if len(exp.domains) < 2:
        raise ValueError("prepared domain-tree runner requires at least two domains")
    if _hierarchy_valid_time(preparation) != exp.start_time.isoformat():
        raise ValueError("preparation valid_time differs from experiment start_time")
    if preparation.get("domain_count") != len(exp.domains):
        raise ValueError("preparation domain count differs from experiment config")
    hierarchy_root = _require_directory(
        prepared_root / "hierarchy-artifacts", "hierarchy artifact root"
    )
    artifact_receipt_path = _require_file(
        hierarchy_root / "receipt.json", "hierarchy artifact receipt"
    )
    artifact_manifest_path = _require_file(
        hierarchy_root / "domain-artifacts.json", "domain artifact manifest"
    )
    artifact_receipt = _json_object(artifact_receipt_path, "hierarchy artifact receipt")
    artifact_manifest = _json_object(artifact_manifest_path, "domain artifact manifest")
    if artifact_receipt != preparation.get("artifact_receipt"):
        raise ValueError(
            "published hierarchy artifact receipt differs from preparation"
        )
    expected_ids = [int(domain.grid_id) for domain in exp.domains]
    expected_manifest = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "domains": [
            {
                "grid_id": grid_id,
                "prepared_cache": f"domains/d{grid_id:02d}/prepared-cache",
                "static_cache": f"domains/d{grid_id:02d}/native-static.npz",
                "geometry_receipt": (f"domains/d{grid_id:02d}/geometry-receipt.json"),
            }
            for grid_id in expected_ids
        ],
    }
    if artifact_manifest != expected_manifest:
        raise ValueError("domain artifact manifest is not canonical")
    expected_artifact_identity = {
        "schema": ARTIFACT_RECEIPT_SCHEMA,
        "status": "READY",
        "domain_count": len(exp.domains),
        "grid_ids": expected_ids,
        "manifest": {
            "path": "domain-artifacts.json",
            "sha256": _sha256(artifact_manifest_path),
        },
        "boundary_inventory": {
            "external": [expected_ids[0]],
            "nested_parent_forced": expected_ids[1:],
        },
    }
    if any(
        artifact_receipt.get(key) != value
        for key, value in expected_artifact_identity.items()
    ):
        raise ValueError("hierarchy artifact receipt identity differs")
    domain_receipts = artifact_receipt.get("domains")
    if not isinstance(domain_receipts, list) or len(domain_receipts) != len(
        exp.domains
    ):
        raise ValueError("hierarchy domain receipt inventory is incomplete")

    grids = tuple(grids_from_projection_config(exp))
    if len(grids) != len(exp.domains):
        raise RuntimeError("experiment grid count differs from domains")
    # The authority triple binds every domain to one preparation. Sources that
    # record it at the top level are pinned against that; the rest are pinned
    # against the first domain's own cache identity, which the per-domain loop
    # below then requires every other domain to equal. Both enforce the same
    # invariant -- one authority across the whole tree -- so neither is weaker.
    provenance = preparation.get("provenance")
    if isinstance(provenance, dict):
        authority = {
            "bridge_manifest_sha256": provenance.get("bridge_manifest_sha256"),
            "source_manifest_sha256": preparation.get(
                "source_manifest_sha256", provenance.get("source_manifest_sha256")
            ),
            "namelist_sha256": provenance.get("native_namelist_input_sha256"),
        }
    else:
        first = exp.domains[0]
        first_header = _json_object(
            hierarchy_root
            / "domains"
            / f"d{int(first.grid_id):02d}"
            / "prepared-cache"
            / "header.json",
            "d01 cache header",
        )
        first_identity = first_header.get("identity")
        if not isinstance(first_identity, dict):
            raise ValueError("d01 cache identity is missing")
        authority = {
            key: first_identity.get(key)
            for key in (
                "bridge_manifest_sha256",
                "source_manifest_sha256",
                "namelist_sha256",
            )
        }
    for label, value in authority.items():
        authority[label] = _digest(value, label)

    bundles = []
    common_source_identity = None
    forcing_hours = None
    for domain, grid, embedded_receipt in zip(exp.domains, grids, domain_receipts):
        label = f"d{int(domain.grid_id):02d}"
        bundle = _require_directory(
            hierarchy_root / "domains" / label, f"{label} bundle"
        )
        cache = _require_directory(bundle / "prepared-cache", f"{label} cache")
        static_path = _require_file(
            bundle / "native-static.npz", f"{label} static cache"
        )
        geometry_path = _require_file(
            bundle / "geometry-receipt.json", f"{label} geometry receipt"
        )
        domain_receipt_path = _require_file(
            bundle / "receipt.json", f"{label} artifact receipt"
        )
        domain_receipt = _json_object(domain_receipt_path, f"{label} artifact receipt")
        if domain_receipt != embedded_receipt:
            raise ValueError(f"{label} receipt differs from hierarchy receipt")

        header = _json_object(cache / "header.json", f"{label} cache header")
        identity = header.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"{label} cache identity is missing")
        if identity.get("domain_config") != _strict_json(asdict(domain)):
            raise ValueError(f"{label} cache domain config differs from experiment")
        for key, expected in authority.items():
            if identity.get(key) != expected:
                raise ValueError(f"{label} cache {key} differs from preparation")
        current_hours = _forcing_hours(preparation, identity)
        if identity.get("forcing_hours") != list(current_hours):
            raise ValueError(f"{label} cache forcing hours differ")
        if forcing_hours is None:
            forcing_hours = current_hours
        elif forcing_hours != current_hours:
            raise ValueError("prepared domain forcing hours differ")
        source_identity = identity.get("source_identity")
        if not isinstance(source_identity, dict) or source_identity.get(
            "grid_id"
        ) != int(domain.grid_id):
            raise ValueError(f"{label} source identity/grid binding differs")
        normalized_source = dict(source_identity)
        normalized_source.pop("grid_id")
        if common_source_identity is None:
            common_source_identity = normalized_source
        elif common_source_identity != normalized_source:
            raise ValueError("prepared domain source identities differ")

        reader = PreparedCacheReader(cache, expected_identity=identity)
        verified = reader.verify_all()
        if verified.get("content_sha256") != header.get("content_sha256"):
            raise ValueError(f"{label} cache content identity differs")
        _validate_vertical(reader, exp, int(domain.grid_id))
        lbc = reader.header.get("metadata", {}).get("lbc")
        if (domain.parent_id == 0 and not isinstance(lbc, dict)) or (
            domain.parent_id != 0 and lbc is not None
        ):
            raise ValueError(f"{label} external/nested LBC ownership differs")
        verify_native_static_receipt(geometry_path, static_path, grid, domain.run)
        static = load_native_static_cache(
            static_path, grid, domain.run.ny, domain.run.nx
        )
        _validate_domain_receipt(
            domain_receipt,
            domain=domain,
            bundle=bundle,
            reader=reader,
            static_path=static_path,
            geometry_path=geometry_path,
        )
        hashes = MappingProxyType(
            {
                "cache_header": _sha256(cache / "header.json"),
                "cache_content": reader.content_sha256,
                "static": _sha256(static_path),
                "geometry_receipt": _sha256(geometry_path),
                "domain_receipt": _sha256(domain_receipt_path),
            }
        )
        bundles.append(
            PreparedDomainBundle(
                grid_id=int(domain.grid_id),
                parent_id=int(domain.parent_id),
                bundle=bundle,
                cache=cache,
                static_path=static_path,
                geometry_receipt_path=geometry_path,
                domain_receipt_path=domain_receipt_path,
                cache_reader=reader,
                cache_identity=MappingProxyType(identity),
                static_fields=MappingProxyType(static),
                authority_sha256=hashes,
            )
        )

    if forcing_hours is None:
        raise RuntimeError("prepared hierarchy resolved no forcing schedule")
    if common_source_identity is None:
        raise RuntimeError("prepared hierarchy resolved no source identity")
    interval_hours = forcing_hours[1] - forcing_hours[0]
    if exp.run_seconds > forcing_hours[-1] * 3600.0:
        raise ValueError("experiment duration exceeds prepared forcing hours")
    execution_plan = resolve_execution_plan(exp)
    authority_hashes = MappingProxyType(
        {
            "preparation_receipt": _sha256(receipt_path),
            "artifact_receipt": _sha256(artifact_receipt_path),
            "artifact_manifest": _sha256(artifact_manifest_path),
            "experiment_config": _sha256(experiment_config),
        }
    )
    return PreparedTreeInputs(
        prepared_root=prepared_root,
        hierarchy_root=hierarchy_root,
        preparation_receipt_path=receipt_path,
        artifact_receipt_path=artifact_receipt_path,
        artifact_manifest_path=artifact_manifest_path,
        experiment_config=experiment_config,
        experiment=exp,
        grids=grids,
        domains=tuple(bundles),
        forcing_hours=forcing_hours,
        boundary_interval_seconds=interval_hours * 3600,
        source_identity=MappingProxyType(common_source_identity),
        execution_plan=execution_plan,
        authority_sha256=authority_hashes,
        source=prepared_source,
    )


def _verify_inputs_unchanged(inputs: PreparedTreeInputs) -> None:
    current = {
        "preparation_receipt": _sha256(inputs.preparation_receipt_path),
        "artifact_receipt": _sha256(inputs.artifact_receipt_path),
        "artifact_manifest": _sha256(inputs.artifact_manifest_path),
        "experiment_config": _sha256(inputs.experiment_config),
    }
    if current != dict(inputs.authority_sha256):
        raise RuntimeError("prepared tree authorities changed during execution")
    for bundle in inputs.domains:
        observed = {
            "cache_header": _sha256(bundle.cache / "header.json"),
            "cache_content": bundle.cache_reader.verify_all()["content_sha256"],
            "static": _sha256(bundle.static_path),
            "geometry_receipt": _sha256(bundle.geometry_receipt_path),
            "domain_receipt": _sha256(bundle.domain_receipt_path),
        }
        if observed != dict(bundle.authority_sha256):
            raise RuntimeError(
                f"prepared d{bundle.grid_id:02d} inputs changed during run"
            )


def _runtime_source_identity() -> Mapping[str, object]:
    files = (
        REPOSITORY_ROOT / "gpuwm/core/model.py",
        REPOSITORY_ROOT / "gpuwm/core/nest.py",
        REPOSITORY_ROOT / "gpuwm/core/microphysics_transition.py",
        REPOSITORY_ROOT / "gpuwm/core/kernels/nest_microphysics.cu",
        Path(__file__).resolve(),
    )
    source_sha256 = {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256(path) for path in files
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
        tree = None
    return MappingProxyType(
        {
            "gpuwm_version": __version__,
            "git_commit": commit,
            "git_tree": tree,
            "source_sha256": source_sha256,
        }
    )


def _verify_thompson_assets(exp) -> None:
    """Byte-validate the mp8 tables wherever they resolve from.

    The two process-environment gates this used to demand
    (GPUWM_EXPERIMENTAL_THOMPSON_MP8=1 and an explicit
    GPUWM_THOMPSON_TABLE_ROOT) predate the packaging promotion: the
    canonical WRF v4.6.1 classic tables now ship as package data,
    `gpuwm fetch-tables` stages the externalized one, and `gpuwm doctor`
    byte-validates all four and reports no gaps.  Demanding the env vars
    anyway meant mp8 -- the wizard's own default at the time -- failed
    twice at runtime, on a machine doctor had just declared clean, with
    neither variable named anywhere in the docs.  What actually
    protected anything was the validation below, and it still runs on
    every launch.
    """

    if not any(domain.run.mp_physics == 8 for domain in exp.domains):
        return
    from gpuwm.physics_compat import thompson_table_root
    from gpuwm.core.thompson_contract import validate_table_assets

    validate_table_assets(Path(thompson_table_root()))


def _rebind_rebuilt_state(state, workspace) -> None:
    if workspace is None:
        return
    for name in workspace.symbols:
        value = getattr(state, name, None)
        if value is not None:
            setattr(state, name, workspace.view(name, value.shape, value.dtype))


def run_prepared_tree(
    inputs: PreparedTreeInputs,
    *,
    output_directory: Path,
    io_mode: str,
    restart: Path | None = None,
    health_debug: bool = False,
) -> dict[str, object]:
    """Restore the prepared domains and execute the existing tree engine."""

    if io_mode not in {"history", "none"}:
        raise ValueError("io_mode must be 'history' or 'none'")
    _verify_thompson_assets(inputs.experiment)

    import cupy as cp

    from gpuwm import runtime
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.dycore import stability_report
    from gpuwm.core.health import StateHealthValidator
    from gpuwm.core.model import (
        DomainNode,
        ExperimentState,
        ModelMemoryLedger,
        ModelRuntimeStatus,
        SharedRRTMGPChunkWorkspace,
        execute_experiment,
        uses_modern_rrtmgp_workspace,
    )
    from gpuwm.core.nest import NestCoupler
    from gpuwm.core.preflight import estimate_experiment
    from gpuwm.core.state import (
        build_shared_dycore_state_workspace,
        build_shared_scratch_arena,
    )
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import restore_prepared_cache
    from gpuwm.io.restart import restore_tree_restart, write_tree_restart
    from gpuwm.io.wrfout import PerDomainWrfoutWriters
    from gpuwm.state_digest import canonical_state_digest
    from gpuwm.supervisor import validate_manifest_checkpoint

    outdir = Path(output_directory).resolve()
    evidence = outdir / "evidence"
    evidence.mkdir()
    progress_path = evidence / "progress.json"
    exp = inputs.experiment
    started_total = time.perf_counter()
    timing: dict[str, float] = {}
    runtime_identity = _runtime_source_identity()
    _atomic_json(
        progress_path,
        {
            "schema": PROGRESS_SCHEMA,
            "status": "RESTORING_PREPARED_DOMAIN_TREE",
            "model_elapsed_seconds": 0.0,
            "requested_run_seconds": float(exp.run_seconds),
            "execution_plan": inputs.execution_plan,
        },
    )

    estimate = estimate_experiment(
        exp, forcing_interval_seconds=inputs.boundary_interval_seconds
    )
    started = time.perf_counter()
    arena = build_shared_scratch_arena(exp.domains)
    rebuilt = build_shared_dycore_state_workspace(exp.domains)
    # The shared helper, never a local restatement of the predicate: the
    # persistent workspace exists only for the MODERN RTE+RRTMGP
    # adapter, and `any(radiation_scheme_ids == (4, 4))` is also true for
    # the legacy-RRTMG variant, which runs one domain at a time and holds
    # no workspace at all.  estimate_experiment() knows the difference,
    # so a tree run under legacy RRTMG allocated a workspace the
    # preflight had not priced and died on the memory-ledger drift guard
    # ("shared radiation allocation differs from preflight") -- exactly
    # the failure uses_modern_rrtmgp_workspace's docstring predicts.
    radiation_workspace = (
        SharedRRTMGPChunkWorkspace(
            nz=exp.root.run.nz, column_chunk=exp.column_chunk, p_top=exp.vertical.p_top
        )
        if uses_modern_rrtmgp_workspace(exp)
        else None
    )
    if arena.nbytes != estimate.scratch_arena_bytes:
        raise RuntimeError("shared scratch allocation differs from preflight")
    if rebuilt.nbytes != estimate.dycore_state_workspace_bytes:
        raise RuntimeError("shared rebuilt-state allocation differs from preflight")
    if (
        radiation_workspace is not None
        and radiation_workspace.nbytes != estimate.workspace_bytes
    ):
        raise RuntimeError("shared radiation allocation differs from preflight")
    timing["allocate_shared_workspaces"] = time.perf_counter() - started
    ledger = ModelMemoryLedger(
        estimate=estimate,
        shared_scratch_arena_bytes=arena.nbytes,
        shared_dycore_state_workspace_bytes=rebuilt.nbytes,
        radiation_workspace=radiation_workspace,
    )

    clock = resolve_clock(exp, lbc_interval_s=float(inputs.boundary_interval_seconds))
    schedule = build_schedule(exp, clock)
    clocks = clock.clocks()
    nodes = {}
    prepared = {}
    drivers = {}
    started = time.perf_counter()
    for domain, grid, bundle in zip(exp.domains, inputs.grids, inputs.domains):
        restored = restore_prepared_cache(
            bundle.cache,
            expected_identity=dict(bundle.cache_identity),
            cfg=domain.run,
            static=bundle.static_fields,
            allow_nested_without_lbc=domain.parent_id != 0,
        )
        _rebind_rebuilt_state(restored.initial_result.state, rebuilt)
        restored.initial_result.state._scratch_arena = arena
        if domain.parent_id != 0:
            restored.initial_result.state._nest_restart_classification = "REBUILT"
        if restored.surface is None:
            raise ValueError(
                f"d{domain.grid_id:02d} prepared cache lacks canonical surface"
            )
        driver = initialize_prepared_physics(
            restored.initial_result,
            domain.run,
            restored.met,
            restored.surface,
            bundle.static_fields,
            NATIVE_LANDUSE_IDENTITY,
            grid,
            exp.start_time,
        )
        radiation = driver.radiation_callable
        if radiation is not None and radiation_workspace is not None:
            radiation.column_chunk = radiation_workspace.column_chunk
            radiation.chunk_workspace = radiation_workspace
        parent = None if domain.parent_id == 0 else nodes[domain.parent_id]
        node = DomainNode(
            cfg=domain,
            grid=grid,
            state=restored.initial_result.state,
            clock=clocks[domain.grid_id],
            parent=parent,
            children=[],
            coupler=None,
        )
        if parent is not None:
            node.coupler = NestCoupler(node)
            parent.children.append(node)
        nodes[domain.grid_id] = node
        prepared[domain.grid_id] = SimpleNamespace(
            static_fields=bundle.static_fields,
            geog_selection=None,
            initial_result=restored.initial_result,
        )
        drivers[domain.grid_id] = driver
    timing["restore_tree_and_initialize_physics"] = time.perf_counter() - started

    fingerprint_payload = {
        "schema": REPORT_SCHEMA,
        "experiment_config_sha256": inputs.authority_sha256["experiment_config"],
        "preparation_receipt_sha256": inputs.authority_sha256["preparation_receipt"],
        "domain_cache_content_sha256": {
            f"d{bundle.grid_id:02d}": bundle.cache_reader.content_sha256
            for bundle in inputs.domains
        },
        "execution_plan": inputs.execution_plan,
        "runtime_source_identity": runtime_identity,
    }
    fingerprint = hashlib.sha256(
        _canonical(_strict_json(fingerprint_payload)).encode("utf-8")
    ).hexdigest()
    model = ExperimentState(
        root=nodes[exp.root.grid_id],
        nodes_by_grid_id=MappingProxyType(nodes),
        schedule=schedule,
        memory_ledger=ledger,
        experiment_fingerprint=fingerprint,
    )
    model._scratch_arena = arena
    model._dycore_state_workspace = rebuilt
    model._prepared_by_grid_id = MappingProxyType(prepared)
    model._input_catalog = None
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._io_manager = None
    model._last_checkpoint = None

    if restart is not None:
        checkpoint = validate_manifest_checkpoint(Path(restart))
        restore_tree_restart(checkpoint, model)
        model._resumed = True

    initial_health = {}
    for grid_id, node in nodes.items():
        result = vars(
            StateHealthValidator(node.state).validate(
                phase=f"initialized.d{grid_id:02d}"
            )
        )
        initial_health[f"d{grid_id:02d}"] = _strict_json(result)
        if not result["ok"]:
            raise FloatingPointError(f"initial d{grid_id:02d} health failed: {result}")

    gpu_peak_used = 0
    pool_peak_total = 0
    history = []

    def record_memory():
        nonlocal gpu_peak_used, pool_peak_total
        free, total = cp.cuda.runtime.memGetInfo()
        gpu_peak_used = max(gpu_peak_used, int(total - free))
        pool_peak_total = max(
            pool_peak_total, int(cp.get_default_memory_pool().total_bytes())
        )

    writers = (
        PerDomainWrfoutWriters(
            model,
            outdir / "wrfout",
            start_time=exp.start_time,
            title=f"gpuwm prepared HRRR domain tree {exp.name}",
        )
        if io_mode == "history"
        else None
    )
    model._io_manager = writers
    forecast_started = time.perf_counter()

    def history_handler(_tree, node, ticks):
        sample = {
            "grid_id": int(node.cfg.grid_id),
            "ticks": int(ticks),
            "elapsed_seconds": float(node.clock.elapsed_seconds),
            **stability_report(
                node.state, node.cfg.run, boundary_width=node.cfg.run.spec_bdy_width
            ),
        }
        history.append(_strict_json(sample))
        if writers is not None:
            runtime._submit_tree_history_frame(writers, node, ticks)
        record_memory()

    def restart_handler(tree, ticks):
        valid = exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
        tree._last_checkpoint = write_tree_restart(outdir, tree, valid)

    def progress_callback(**event):
        record_memory()
        _atomic_json(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "status": "RUNNING",
                "model_elapsed_seconds": event["model_elapsed_seconds"],
                "outer_step": event["outer_step"],
                "requested_run_seconds": float(exp.run_seconds),
                "forecast_wall_seconds": time.perf_counter() - forecast_started,
                "gpu_peak_used_bytes_observed": gpu_peak_used,
                "last_durable_wrfout": event.get("last_durable_wrfout"),
                "last_checkpoint": event.get("last_checkpoint"),
            },
        )

    if writers is None:
        execution = execute_experiment(
            model,
            history_handler=None,
            restart_handler=restart_handler,
            progress_callback=progress_callback,
            validate_state=True,
            health_debug=health_debug,
            skip_feedback_path=True,
            pool_trim_per_period=True,
        )
        wrfout_paths = ()
    else:
        with writers:
            execution = execute_experiment(
                model,
                history_handler=history_handler,
                restart_handler=restart_handler,
                progress_callback=progress_callback,
                validate_state=True,
                health_debug=health_debug,
                skip_feedback_path=True,
                pool_trim_per_period=True,
            )
            writers.drain()
            wrfout_paths = writers.paths
    cp.cuda.Stream.null.synchronize()
    timing["forecast_execution"] = time.perf_counter() - forecast_started
    model._io_manager = None
    record_memory()

    transition_path, transition_sha, transitions = (
        runtime._write_microphysics_transition_receipt(
            evidence, model, exp, resumed=restart is not None
        )
    )
    final_health = {}
    final_stability = {}
    final_digests = {}
    for grid_id, node in nodes.items():
        result = vars(
            StateHealthValidator(node.state).validate(phase=f"final.d{grid_id:02d}")
        )
        final_health[f"d{grid_id:02d}"] = _strict_json(result)
        if not result["ok"]:
            raise FloatingPointError(f"final d{grid_id:02d} health failed: {result}")
        final_stability[f"d{grid_id:02d}"] = _strict_json(
            stability_report(
                node.state, node.cfg.run, boundary_width=node.cfg.run.spec_bdy_width
            )
        )
        final_digests[f"d{grid_id:02d}"] = canonical_state_digest(
            node.state, node.clock, scope="trajectory"
        )

    _verify_inputs_unchanged(inputs)
    if _runtime_source_identity() != runtime_identity:
        raise RuntimeError("forecast implementation changed during execution")
    outputs = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in wrfout_paths
    ]
    timing["total"] = time.perf_counter() - started_total
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "source": inputs.source,
        "readiness": "IMPLEMENTED_UNVERIFIED",
        "execution_plan": inputs.execution_plan,
        "experiment": {
            "name": exp.name,
            "start_time": exp.start_time.isoformat(),
            "run_seconds": float(exp.run_seconds),
            "fingerprint": fingerprint,
            "domains": _domain_rows(exp),
        },
        "wall_seconds": timing["total"],
        "timing_seconds": timing,
        "executor": {
            "steps": int(execution.steps),
            "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "health": {
            "initial": initial_health,
            "final": final_health,
            "final_stability": final_stability,
            "history": history,
        },
        "final_state_digest": final_digests,
        "microphysics_transitions": {
            "path": str(transition_path.resolve()),
            "sha256": transition_sha,
            "edges": transitions,
        },
        "memory": {
            "gpu_peak_used_bytes_observed": gpu_peak_used,
            "cupy_pool_peak_total_bytes_observed": pool_peak_total,
            "preflight_alloc_estimate_bytes": int(estimate.alloc_estimate_bytes),
        },
        "output": {
            "io_mode": io_mode,
            "frame_count": len(outputs),
            "total_bytes": sum(item["bytes"] for item in outputs),
            "files": outputs,
            "last_checkpoint": (
                None
                if model._last_checkpoint is None
                else str(Path(model._last_checkpoint).resolve())
            ),
        },
        "input": {
            "prepared_root": str(inputs.prepared_root),
            "source_identity": dict(inputs.source_identity),
            "forcing_hours": list(inputs.forcing_hours),
            "boundary_interval_seconds": inputs.boundary_interval_seconds,
            "authority_sha256": dict(inputs.authority_sha256),
            "domains": {
                f"d{bundle.grid_id:02d}": dict(bundle.authority_sha256)
                for bundle in inputs.domains
            },
        },
        "runtime_source_identity": runtime_identity,
    }
    _atomic_json(evidence / "run-receipt.json", report)
    _atomic_json(
        progress_path,
        {
            "schema": PROGRESS_SCHEMA,
            "status": "PASS",
            "model_elapsed_seconds": float(exp.run_seconds),
            "requested_run_seconds": float(exp.run_seconds),
            "run_receipt": str((evidence / "run-receipt.json").resolve()),
            "frame_count": len(outputs),
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--preparation-receipt-sha256", required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--experiment-config-sha256", required=True)
    parser.add_argument("--io-mode", choices=("history", "none"), default="history")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--health-debug", action="store_true")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--show-capabilities"]:
        print(json.dumps(runner_capabilities(), sort_keys=True))
        return 0
    args = _parser().parse_args(argv)
    # A rejected --outdir is a usage mistake, not a crash: it must read as
    # one sentence naming the problem and a directory that works.  A node-8
    # pilot met this guard as a raw traceback, on a command the front door
    # itself had suggested.
    try:
        outdir = claim_output_directory(
            args.outdir,
            protected_roots=(args.prepared_root, args.experiment_config))
    except (ValueError, FileExistsError) as error:
        print(f"prepared_domain_tree_forecast: --outdir refused: {error}",
              file=sys.stderr)
        return 2
    try:
        inputs = preflight_prepared_tree(
            prepared_root=args.prepared_root,
            preparation_receipt_sha256=args.preparation_receipt_sha256,
            experiment_config=args.experiment_config,
            experiment_config_sha256=args.experiment_config_sha256,
        )
        report = run_prepared_tree(
            inputs,
            output_directory=outdir,
            io_mode=args.io_mode,
            restart=args.restart,
            health_debug=args.health_debug,
        )
    except BaseException as error:
        evidence = outdir / "evidence"
        evidence.mkdir(exist_ok=True)
        _atomic_json(
            evidence / "failed-run-receipt.json",
            {
                "schema": REPORT_SCHEMA,
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "readiness": report["readiness"],
                "plan_id": report["execution_plan"]["plan_id"],
                "domain_count": report["execution_plan"]["domain_count"],
                "wall_seconds": report["wall_seconds"],
                "frame_count": report["output"]["frame_count"],
                "run_receipt": str(
                    (outdir / "evidence" / "run-receipt.json").resolve()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
