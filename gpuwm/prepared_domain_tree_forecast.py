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
from gpuwm.certify.capsule import emit_run_capsule  # noqa: E402
from gpuwm.core import streaming  # noqa: E402
from gpuwm.core.microphysics_transition import (  # noqa: E402
    MP8_TO_MP18_POLICY,
    resolve_microphysics_transition,
)
from gpuwm.experiment import load_experiment  # noqa: E402
from gpuwm.kernel_compile_notice import (  # noqa: E402
    COMPILING_STATUS, kernel_compile_notice,
)
from gpuwm.physics_compat import (  # noqa: E402
    experimental_selection_sentence,
)
from gpuwm.io.restart import RestartMismatchError  # noqa: E402
from gpuwm.ingest.prepared_cache import (  # noqa: E402
    PreparedCacheReader,
    compare_prepared_domain_config,
    effective_prepared_domain_config,
    prepared_domain_config_identity,
    prepared_identity_refusal,
    undelayed_identity_defaults,
)
from gpuwm.native_wrf_contract import (  # noqa: E402
    NATIVE_LANDUSE_IDENTITY,
    load_native_static_cache,
    verify_native_static_receipt,
)
from gpuwm.static.lambert import grids_from_projection_config  # noqa: E402
from gpuwm.table_assets import MissingTableAssets  # noqa: E402


REPORT_SCHEMA = "gpuwm-prepared-domain-tree-forecast-v1"
PROGRESS_SCHEMA = "gpuwm-prepared-domain-tree-progress-v1"
CAPABILITIES_SCHEMA = "gpuwm-runner-capabilities-v1"
PLAN_SCHEMA = "gpuwm-prepared-domain-tree-plan-v1"
RUNNER = "tools.prepared_domain_tree_forecast"
ARBITRARY_PLAN_ID = "arbitrary-prepared-one-way-domain-tree-v1"
THOMPSON_NSSL_PLAN_ID = "thompson-outer-nssl2-inner-mp8-mp18-v1"
HIERARCHY_SCHEMA = "gpuwm-native-hrrr-hierarchy-direct-v1"
SEALED_EXTENSION_FINGERPRINT_SCHEMA = \
    "gpuwm-prepared-tree-sealed-extension-fingerprint-v1"
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
        "schema": "gpuwm-gfs-native-hierarchy-proof-v2",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "source": "gfs",
    },
    {
        # v1 predates the front-door physics receipt and therefore cannot
        # be promoted to v2 by inference; it stays independently
        # verifiable on its own terms, exactly as the direct proof's v2
        # does beside v3.
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


def _without_forecast_stop(exp):
    """Normalize only the typed experiment's enumerated forecast stops.

    ``ExperimentConfig.run_seconds`` is the one identity field which changes
    between successively longer legs.  The typed loader also copies that
    authority into each ``DomainConfig.run.run_seconds``; remove those exact
    derived paths only after proving they still equal the authority.  Any
    future nested field with the same spelling remains covered.  Keeping the
    exceptions explicit prevents a newly introduced control from silently
    falling outside the restart-extend identity.
    """
    value = _strict_json(asdict(exp))
    if not isinstance(value, Mapping) or "run_seconds" not in value:
        raise ValueError(
            "sealed extension identity requires ExperimentConfig.run_seconds")
    result = dict(value)
    authoritative_stop = result.pop("run_seconds")
    domains = result.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError(
            "sealed extension identity requires typed experiment domains")
    normalized_domains = []
    for index, domain in enumerate(domains):
        if not isinstance(domain, Mapping) or not isinstance(
                domain.get("run"), Mapping):
            raise ValueError(
                f"sealed extension identity domain {index} lacks RunConfig")
        normalized_domain = dict(domain)
        normalized_run = dict(domain["run"])
        if normalized_run.get("run_seconds") != authoritative_stop:
            raise ValueError(
                f"sealed extension identity domain {index} run_seconds "
                "diverges from ExperimentConfig.run_seconds")
        normalized_run.pop("run_seconds")
        normalized_domain["run"] = normalized_run
        normalized_domains.append(normalized_domain)
    result["domains"] = normalized_domains
    return result


def sealed_extension_identity_components(
    exp, runtime_identity
) -> dict[str, object]:
    """The named components whose digest is the sealed-extension identity.

    The sealed-extension counterpart of
    :func:`tree_restart_identity_components`, and named for the same
    reason: these are published beside the digest in the checkpoint
    header, so a refusal can say WHICH component moved instead of only
    that the hash did.  Both checkpoint routes now carry named
    components; before this the sealed route carried none, and a
    horizon extension that failed to line up could only report a bare
    digest difference.
    """

    return {
        "schema": SEALED_EXTENSION_FINGERPRINT_SCHEMA,
        "experiment": _without_forecast_stop(exp),
        "runtime_source_identity": _strict_json(runtime_identity),
    }


def sealed_extension_fingerprint(exp, runtime_identity) -> str:
    """Stable trajectory identity shared by successively longer legs."""
    payload = sealed_extension_identity_components(exp, runtime_identity)
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


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
        "relocation": {
            "follow_sources": (
                "accepted only over a sealed, digest-verified statics "
                "corridor prepared with --statics-corridor; corridor-less "
                "bundles refuse with that remedy named"),
            "bounds_only": "accepted (manual/API mechanism, no runner)",
            "corridor_memory": "disk/host only; no GPU residency",
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
    #: Per-domain identity fields accepted as schema growth rather
    #: than as a match -- empty on a cache written by this release.
    tolerated_identity_fields: Mapping[str, tuple[str, ...]] = (
        MappingProxyType({}))
    #: The verified statics corridor
    #: (:class:`gpuwm.static.corridor.ChildStaticsCorridor`) when the
    #: experiment configures a [relocation] follow source, else ``None``.
    statics_corridor: object | None = None
    #: The corridor cache file, for the unchanged-during-run re-hash.
    statics_corridor_cache_path: Path | None = None


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


#: The restart identity this runner binds, component by component.
#:
#: Until 1.4.1 the whole experiment TOML's SHA-256 was one of these
#: components, which made the tree route's restart identity strictly
#: narrower than the contract `gpuwm run --restart` publishes: "only the
#: forecast length / output and restart cadence may differ".  All three
#: of those live in the TOML, so all three moved the digest and all three
#: were refused -- including extending `run_seconds` from a checkpoint,
#: the worked example in FIRST-LIGHT section 7.  The single-domain route
#: never had the problem: `gpuwm.io.restart._require_config_match` diffs
#: the RunConfig field by field and skips exactly those keys
#: (`CONFIG_RUN_LENGTH_FIELDS`).
#:
#: The experiment component is now the same timing-independent identity
#: `gpuwm.core.model.experiment_fingerprint` binds on the native route,
#: so the two restart contracts agree.  Everything else the digest bound
#: -- preparation receipt, per-domain prepared-cache content, execution
#: plan, runtime source identity -- is bound exactly as before.
TREE_RESTART_IDENTITY_COMPONENTS = (
    "schema", "experiment_identity", "preparation_receipt_sha256",
    "domain_cache_content_sha256", "execution_plan",
    "runtime_source_identity",
)


def tree_restart_identity_components(
    inputs, runtime_identity
) -> dict[str, object]:
    """The named components whose digest is the tree restart fingerprint.

    Named, and stored beside the fingerprint in the checkpoint header,
    so a mismatch can say WHICH component differs.  A bare hash
    comparison could only ever say that something did -- which is what
    made the refusal a nine-word traceback with nothing actionable in it.
    """

    from gpuwm.core.model import restart_identity_payload

    # Strict-JSON at construction, not at hash time: these components are
    # also written into the checkpoint header, and a MappingProxyType or
    # a Path reaching json.dump there fails the checkpoint write itself.
    return _strict_json({
        "schema": REPORT_SCHEMA,
        "experiment_identity": restart_identity_payload(inputs.experiment),
        "preparation_receipt_sha256":
            inputs.authority_sha256["preparation_receipt"],
        "domain_cache_content_sha256": {
            f"d{bundle.grid_id:02d}": bundle.cache_reader.content_sha256
            for bundle in inputs.domains
        },
        "execution_plan": _plan_restart_identity(inputs.execution_plan),
        "runtime_source_identity": runtime_identity,
    })


def _plan_restart_identity(plan) -> dict[str, object]:
    """The execution plan as RESTART identity: what it integrates.

    ``_domain_rows`` describes each domain for the receipt, and one of
    the things it describes is ``history_interval_s`` -- when the run
    writes.  Hashing the receipt as-is re-bound the output cadence the
    ``--restart`` contract publishes as free to change, which is the
    same defect one layer up from the prepared-cache identity.  The plan
    published in the report is unchanged; only this view drops it.
    """

    identity = _strict_json(plan)
    for row in identity.get("domains", ()):
        if isinstance(row, dict):
            row.pop("history_interval_s", None)
    return identity


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

    # One note per FILE, not per (file, schema) candidate.  Several
    # sources write `proof.json`, and the GFS entry alone now has two
    # accepted schemas, so appending per candidate printed
    # `proof.json digest differs` once for each -- three or four
    # identical lines saying nothing about which file was read or what
    # its digest actually was.
    seen: dict[str, str] = {}
    for entry in _HIERARCHY_DOCUMENTS:
        candidate = prepared_root / entry["filename"]
        if not candidate.is_file():
            continue
        observed = _sha256(candidate)
        if observed != expected_sha256:
            seen[entry["filename"]] = (
                f"{candidate} has sha256 {observed}")
            continue
        document = _json_object(candidate, "preparation document")
        if document.get("schema") != entry["schema"]:
            seen.setdefault(
                entry["filename"],
                f"{candidate} carries schema "
                f"{document.get('schema')!r}")
            continue
        if document.get("status") != entry["status"]:
            raise ValueError(
                f"prepared root {entry['filename']} is not a "
                f"{entry['status']} {entry['source']} hierarchy"
            )
        return candidate, document, entry["source"]
    looked_for = ", ".join(
        sorted({e["filename"] for e in _HIERARCHY_DOCUMENTS}))
    if not seen:
        raise ValueError(
            f"prepared root {prepared_root} carries none of the documents a "
            f"hierarchy preparation writes ({looked_for}), so nothing there "
            f"can match --preparation-receipt-sha256 {expected_sha256}")
    raise ValueError(
        f"prepared root {prepared_root} carries no hierarchy document "
        f"matching --preparation-receipt-sha256 {expected_sha256}; "
        + "; ".join(seen[name] for name in sorted(seen))
        + ".  Accepted schemas: "
        + ", ".join(sorted({e["schema"] for e in _HIERARCHY_DOCUMENTS})))


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
    # Same governance one line down from the relocation refusal, and for
    # the same reason: this route restores children from prepared caches,
    # so it neither reserves a dormant nest's VRAM nor evaluates its
    # trigger.  A declared spawn must not integrate as a silently absent
    # nest.
    from gpuwm.experiment import refuse_unrouted_spawn
    refuse_unrouted_spawn(exp, "prepared domain-tree")
    # And the same governance for [tiles], one line down, because this
    # route wires no streamed-domain builder either.  Without this the
    # refusal still happens -- make_stepper raises it -- but 790 lines and
    # one whole resident tree construction later, which is exactly the wrong
    # side of the allocation a streamed run exists to avoid.
    streaming.refuse_unrouted_streaming(exp, "prepared domain-tree")
    # THE predicate, asked rather than restated.  This runner is the
    # door the whole corridor mechanism exists to satisfy, and it used
    # to hold its own copy of the sentence -- so a reading of
    # "[relocation] with moves but no follow" that the preparation doors
    # agreed on could have differed here, in the one place that decides
    # whether a bundle is accepted.  Supersedes the inline
    # exp.relocation.enabled and (follow is not None or moves) test.
    from gpuwm.static.corridor import config_declares_follow_source
    relocation_follow = config_declares_follow_source(exp)
    if relocation_follow:
        # The [static.highres] idiom: an enabled surface refuses on the
        # lanes that cannot honor it.  A prepared tree deliberately runs
        # WITHOUT its ingest inputs, so a relocated child's statics
        # cannot be rebuilt from a GEOG source at runtime.  What CAN
        # honor a follow source here is the sealed statics corridor:
        # child-resolution statics over the whole parent extent, emitted
        # at preparation time (--statics-corridor) with its digest bound
        # into the preparation document, and cropped per footprint at
        # runtime through the same rebuild machinery the case-data route
        # wires.  A bundle prepared without one still refuses -- a
        # config that says "follow the storm" must not integrate as a
        # silently static nest.
        if int(exp.relocation.grid_id) not in {
                int(domain.grid_id) for domain in exp.domains}:
            raise ValueError(
                f"[relocation] names grid_id = {exp.relocation.grid_id}, "
                "which is not a domain of this experiment")
        corridor_set = preparation.get("statics_corridor")
        corridor_label = f"d{int(exp.relocation.grid_id):02d}"
        if not isinstance(corridor_set, Mapping):
            raise ValueError(
                "[relocation] configures a follow source, but this "
                "prepared bundle carries no statics corridor: the "
                "prepared domain-tree route runs without the case's "
                "static (GEOG) source and "
                "cannot rebuild a relocated child's statics for a new "
                "footprint at runtime.  Re-prepare the tree with "
                "--statics-corridor (the tree preparation front door "
                "seals child-resolution statics over the whole parent "
                "extent, which this runner then crops per move), run "
                "the case-data route (gpuwm run), or remove "
                "[relocation.follow]/[[relocation.move]] from this "
                "config.")
        corridor_domains = corridor_set.get("domains")
        if (not isinstance(corridor_domains, Mapping)
                or corridor_label not in corridor_domains):
            covered = (sorted(corridor_domains)
                       if isinstance(corridor_domains, Mapping) else [])
            raise ValueError(
                f"[relocation] names {corridor_label}, but this bundle's "
                f"statics corridor covers only {covered}; re-prepare "
                f"with --statics-corridor covering {corridor_label} "
                "(bare --statics-corridor covers every child domain).")
    # The physics this config selects has to be RUNNABLE before the
    # hierarchy is verified, not after: this is the preflight, and a
    # missing lookup table is exactly the class of thing a preflight
    # exists to name before the GPU is touched.
    _verify_thompson_assets(exp)
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
    # Which identity fields each domain's cache was accepted WITHOUT,
    # because they postdate the header and hold their not-in-use
    # default.  Normally empty; recorded either way, so a run on an
    # upgraded install can show exactly what it tolerated.
    tolerated_identity: dict[str, list[str]] = {}
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
        # Default-tolerant, and only on the document that grows fields
        # as the configuration schema grows.  v1.1.0 added a per-domain
        # `start_time` for staggered nest starts, which made every
        # v1.0.1-era prepared tree unrunnable under a strict-equality
        # check -- refused with a sentence that named the user's
        # experiment file, when the cause was a package upgrade.
        # Both sides through the SAME normalization the hierarchy's root
        # binding uses.  They used to differ: the hierarchy gate pinned an
        # inactive cudt_minutes to 0 and this one compared it raw, so a
        # cumulus-off tree whose cache inherited RunConfig's live 5.0 was
        # refused against a wizard config that wrote the profile's 0.0 --
        # after preparation, on a switch no step of the run reads.  The
        # same table also drops the write cadences and the inert
        # diagnostic toggle, which say when a forecast writes rather than
        # what it integrates.
        tolerated_fields, differing_fields = compare_prepared_domain_config(
            effective_prepared_domain_config(identity.get("domain_config")),
            effective_prepared_domain_config(
                prepared_domain_config_identity(domain)),
            not_in_use=undelayed_identity_defaults(exp))
        if differing_fields:
            raise ValueError(prepared_identity_refusal(
                subject=f"{label} prepared cache", header=header,
                differing=differing_fields,
                re_prepare=(
                    "the front door that wrote this tree, against this "
                    "experiment config")))
        tolerated_identity[label] = list(tolerated_fields)
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
    statics_corridor = None
    statics_corridor_cache_path = None
    if relocation_follow:
        # Digest-verified against the preparation document (whose own
        # digest the caller pinned), geometry-verified against this
        # experiment, grid-probe-verified against this machine's own
        # arithmetic.  A corridor that fails ANY of these refuses loudly
        # -- it never degrades to a silently static nest.
        from gpuwm.static.corridor import (STATICS_CORRIDOR_DIRNAME,
                                           load_child_statics_corridor)
        grid_id = int(exp.relocation.grid_id)
        by_id = {int(domain.grid_id): index
                 for index, domain in enumerate(exp.domains)}
        child = exp.domains[by_id[grid_id]]
        parent_run = exp.domains[by_id[int(child.parent_id)]].run
        corridor_directory = hierarchy_root / STATICS_CORRIDOR_DIRNAME
        statics_corridor = load_child_statics_corridor(
            corridor_directory,
            expected_set_receipt=preparation["statics_corridor"],
            grid_id=grid_id, child_dc=child, parent_run=parent_run,
            reference_grid=grids[by_id[grid_id]])
        statics_corridor_cache_path = corridor_directory / (
            preparation["statics_corridor"]["domains"]
            [f"d{grid_id:02d}"]["cache"]["path"])
    interval_hours = forcing_hours[1] - forcing_hours[0]
    if exp.run_seconds > forcing_hours[-1] * 3600.0:
        # The gate that keeps a longer run_seconds honest.  A restart may
        # extend the forecast, but only into boundaries this tree was
        # actually prepared with; naming both numbers is what tells the
        # user whether to shorten the run or re-prepare from more forcing.
        raise ValueError(
            f"experiment run_seconds = {float(exp.run_seconds):g} s exceeds "
            f"the prepared forcing, which reaches "
            f"{forcing_hours[-1] * 3600.0:g} s after start_time (f"
            f"{forcing_hours[-1]:03d}); shorten the run or re-prepare the "
            "tree from a longer fetch")
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
        tolerated_identity_fields=MappingProxyType({
            label: tuple(names)
            for label, names in tolerated_identity.items()}),
        statics_corridor=statics_corridor,
        statics_corridor_cache_path=statics_corridor_cache_path,
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
    if inputs.statics_corridor_cache_path is not None:
        observed = _sha256(inputs.statics_corridor_cache_path)
        if observed != inputs.statics_corridor.cache_sha256:
            raise RuntimeError(
                "prepared statics corridor changed during execution")
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


def _provenance_receipt() -> dict:
    """The running tree, for the report.  Never raises.

    Same contract as the single-domain runner's: beside
    ``runtime_source_identity``, never instead of it.  That one binds
    the implementation's bytes; this one names the install they came
    out of.
    """

    try:
        from gpuwm.provenance_gate import receipt_block

        return receipt_block()
    except Exception as error:                          # noqa: BLE001
        return {"unavailable": f"{type(error).__name__}: {error}"}


def _runtime_source_identity() -> Mapping[str, object]:
    files = (
        REPOSITORY_ROOT / "gpuwm/core/model.py",
        REPOSITORY_ROOT / "gpuwm/core/nest.py",
        REPOSITORY_ROOT / "gpuwm/core/microphysics_transition.py",
        REPOSITORY_ROOT / "gpuwm/core/kernels/nest_microphysics.cu",
        Path(__file__).resolve(),
    )
    # ``.as_posix()``, not ``str()``: this identity is hashed into
    # ``sealed_extension_fingerprint`` and published in checkpoint
    # headers, so a backslash key makes the SAME code fingerprint
    # differently on Windows than on Linux and a leg cannot extend
    # across machines.  On Linux the two spellings are byte-identical,
    # so no already-sealed fingerprint moves; only Windows converges.
    source_sha256 = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256(path)
        for path in files
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
    from gpuwm.core.thompson_contract import validate_table_assets
    from gpuwm.table_assets import require_thompson_tables

    # Absence first, and in one sentence.  A wheel user reached this
    # line after paying for a fetch and three minutes of preprocessing
    # and got a five-frame FileNotFoundError naming a path inside
    # site-packages -- true, and useless.  require_thompson_tables says
    # which table and which command stages it; the byte validation
    # below is unchanged and still runs on every launch.
    root = require_thompson_tables()
    validate_table_assets(root)


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
    sealed_forcing_extension: bool = False,
    observer=None,
) -> dict[str, object]:
    """Restore the prepared domains and execute the existing tree engine.

    ``observer`` is an optional second consumer of this run's progress,
    for a caller driving the runner in-process rather than reading its
    ``progress.json`` from outside (:class:`gpuwm.runplan.RunObserver`
    is one).  It receives every per-step progress event this runner
    already builds, and -- if it carries the ``output_committed`` hook
    -- each per-domain wrfout as it becomes durable.  ``None`` leaves
    every byte of this runner's behaviour unchanged.
    """

    if io_mode not in {"history", "none"}:
        raise ValueError("io_mode must be 'history' or 'none'")
    _verify_thompson_assets(inputs.experiment)

    import cupy as cp

    from gpuwm import runtime
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.gpu_mem_watch import (
        GpuPeakMemoryWatcher,
        default_cupy_probes,
    )
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
    from gpuwm.runtime import declared_constant_glw
    from gpuwm.ingest.lateral_bc import bind_lateral_boundary_clock
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
    initial_perturbation_receipts: list[dict[str, object]] = []
    started = time.perf_counter()
    compile_notice = kernel_compile_notice()
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
        if exp.perturbation is not None and restart is None:
            # Configured initial-state theta bubbles (PROVENANCE D12).
            # The sealed caches stay the pure analysis; the bubbles are
            # added to the RESTORED state here, per domain, before
            # initialize_prepared_physics rederives p/al/alt from the
            # perturbed prognostics.  A resumed run applies nothing --
            # the checkpoint trajectory already carries the evolved
            # bubble.  Refusals (coarse-domain containment, zero cells)
            # fire here, before any GPU integration.
            from gpuwm.core.diagnostics import update_diagnostics
            from gpuwm.ingest.init_perturbation import (
                build_initial_state_perturbation)
            # The cache serializes p as the preparation left it, and a
            # prepare-only cache leaves the EOS to its consumer -- so
            # diagnose the RESTORED prognostics first; the bubble's
            # rh_preserve reads state.p.  initialize_prepared_physics
            # runs the same diagnostics again below, from the perturbed
            # prognostics, which is the order the seam wants anyway.
            update_diagnostics(
                restored.initial_result.state, domain.run.hypsometric_opt)
            applier = build_initial_state_perturbation(
                exp.perturbation, grid, grid_id=int(domain.grid_id),
                require_containment=domain.parent_id == 0)
            initial_perturbation_receipts.append(
                applier.apply_to_state(restored.initial_result.state))
        # The first domain's physics initialization is where a first
        # run pays its one-time NVRTC compile (~100 s on a modern
        # card), and it used to pay it under the stale RESTORING
        # status -- the first field run of the published wheel watched
        # that silence and concluded a hang.  One line and one status
        # flip, only when the kernel cache says the compile is coming.
        if compile_notice is not None:
            print(compile_notice, flush=True)
            _atomic_json(progress_path, {
                "schema": PROGRESS_SCHEMA,
                "status": COMPILING_STATUS,
                "model_elapsed_seconds": 0.0,
                "requested_run_seconds": float(exp.run_seconds),
                "execution_plan": inputs.execution_plan,
            })
            compile_notice = None
        driver = initialize_prepared_physics(
            restored.initial_result,
            domain.run,
            restored.met,
            restored.surface,
            bundle.static_fields,
            NATIVE_LANDUSE_IDENTITY,
            grid,
            exp.start_time,
            constant_glw_wm2=declared_constant_glw(exp),
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
    # This runner constructs DomainNodes directly rather than going through
    # core.model.build_experiment.  Bind the prepared root's already-attached
    # external mirror before restart validation or the first solve so Davies
    # consumers use WRF's post-increment dtbc semantic (dt..T), not the
    # retired elapsed-based compatibility path (0..T-dt).
    bind_lateral_boundary_clock(
        nodes[exp.root.grid_id].state, nodes[exp.root.grid_id].clock)
    timing["restore_tree_and_initialize_physics"] = time.perf_counter() - started
    if exp.perturbation is not None:
        # Treatment proof, before any integration: the accepted config
        # echoed value for value plus what each restored domain actually
        # received.  A resumed run records the resume instead -- the
        # bubbles live in the checkpoint trajectory, not in a fresh
        # application.
        runtime._write_initial_perturbation_receipt(
            evidence, exp,
            initial_perturbation_receipts if restart is None else [{
                "resumed": True,
                "note": "resumed from checkpoint; the initial "
                        "perturbation was applied when the trajectory "
                        "began and is not re-applied",
            }])

    if sealed_forcing_extension:
        # The sealed route keeps its OWN identity and its own digest: it is
        # deliberately stable across successively longer legs, which is the
        # whole point of a horizon extension, so it cannot be replaced by the
        # restart identity below.  What it gains here is named components,
        # published beside the digest exactly as the restart route's are.
        sealed_components = sealed_extension_identity_components(
            exp, runtime_identity)
        fingerprint = hashlib.sha256(
            _canonical(sealed_components).encode("utf-8")).hexdigest()
        fingerprint_components = _strict_json(sealed_components)
    else:
        fingerprint_components = tree_restart_identity_components(
            inputs, runtime_identity)
        fingerprint = hashlib.sha256(
            _canonical(_strict_json(fingerprint_components)).encode("utf-8")
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
    # Published beside the digest so a checkpoint written here can be
    # refused BY NAME rather than as an unexplained hash difference.
    model._experiment_fingerprint_components = fingerprint_components
    model._prepared_by_grid_id = MappingProxyType(prepared)
    model._input_catalog = None
    model._runtime_status = ModelRuntimeStatus()
    model._resumed = False
    model._resume_committed_history_grid_ids = frozenset()
    model._io_manager = None
    model._last_checkpoint = None

    # The corridor lift: with a verified statics corridor on hand this
    # route wires the SAME RelocationRunner the case-data route does --
    # corridor crops standing in for per-footprint GEOG rebuilds --
    # which is exactly what makes the preflight's corridor-less refusal
    # honest rather than permanent.  Bounds-only [relocation] builds no
    # runner, exactly as everywhere else.
    relocation_runner = (
        runtime.build_prepared_tree_relocation_runner(
            exp, statics_corridor=inputs.statics_corridor, model=model,
            outdir=outdir, radiation_workspace=radiation_workspace)
        if inputs.statics_corridor is not None else None)

    if restart is not None:
        checkpoint = validate_manifest_checkpoint(Path(restart))
        restore_tree_restart(
            checkpoint, model,
            sealed_forcing_extension=sealed_forcing_extension)
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

    history = []
    # Boundary-only sampling under-reported the peak: the executor trims
    # the CuPy pool per STEP and at period commit BEFORE the progress
    # callback fires, so samples taken only in those callbacks missed
    # the intra-step transient working set (19.41 GiB reported against
    # 22.34 GiB true on the four-domain tree shape).  The watcher polls
    # from a daemon thread as well, and the boundary/end-of-run
    # sample() calls below fold into the same maxima.
    memory_watch = GpuPeakMemoryWatcher(default_cupy_probes())

    writers = (
        PerDomainWrfoutWriters(
            model,
            outdir / "wrfout",
            start_time=exp.start_time,
            # The title used to say HRRR on every tree, including the GFS
            # trees this runner has executed since the GFS front door
            # opened.  A durable artifact does not get to name a source
            # its run never touched; `inputs.source` is the run's own.
            title=f"gpuwm prepared {inputs.source.upper()} domain tree "
                  f"{exp.name}",
            # Same contract as the single-domain runner: the prepared
            # cache's source identity carries the initial-condition
            # provenance for sources whose front door publishes one.
            initial_condition=inputs.source_identity.get(
                "initial_condition"),
            source=inputs.source,
        )
        if io_mode == "history"
        else None
    )
    model._io_manager = writers
    forecast_started = time.perf_counter()

    def history_handler(_tree, node, ticks):
        # ASKED OF THE STEPPER, per grid.  ``steppers`` is bound below and
        # read at CALL time.  Under ``[tiles] store = "host"`` the domain
        # is in a pinned host store and ``node.state`` is the copy taken at
        # t = 0 that the sweep never writes, so every history sample would
        # otherwise record the INITIAL nan/w_max/CFL as though it were the
        # frame's.  ``stability_observer`` returns ``dycore.stability_report``
        # itself for a resident grid, which is every grid of a tree that
        # configures no [tiles].
        sample = {
            "grid_id": int(node.cfg.grid_id),
            "ticks": int(ticks),
            "elapsed_seconds": float(node.clock.elapsed_seconds),
            **streaming.stability_observer(
                steppers.get(int(node.cfg.grid_id)))(
                    node.state, node.cfg.run,
                    boundary_width=node.cfg.run.spec_bdy_width),
        }
        history.append(_strict_json(sample))
        if writers is not None:
            runtime._submit_tree_history_frame(writers, node, ticks)
        memory_watch.sample()

    def restart_handler(tree, ticks):
        valid = exp.start_time + timedelta(seconds=ticks / tree.schedule.clock.tick_den)
        tree._last_checkpoint = write_tree_restart(
            outdir, tree, valid,
            sealed_forcing_extension=sealed_forcing_extension)

    def progress_callback(**event):
        if observer is not None:
            observer(**event)
        memory_watch.sample()
        _atomic_json(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "status": "RUNNING",
                "model_elapsed_seconds": event["model_elapsed_seconds"],
                "outer_step": event["outer_step"],
                "requested_run_seconds": float(exp.run_seconds),
                "forecast_wall_seconds": time.perf_counter() - forecast_started,
                "gpu_peak_used_bytes_observed": memory_watch.peak_bytes(
                    "cuda_device_used"),
                "last_durable_wrfout": event.get("last_durable_wrfout"),
                "last_checkpoint": event.get("last_checkpoint"),
            },
        )

    # After the closures, before any submit (the first happens inside
    # execute_experiment below).  `io_mode="none"` has no writers to
    # attach to and commits no outputs, so there is nothing to announce.
    if observer is not None and writers is not None:
        writers.attach_progress_callback(observer)
    if relocation_runner is not None and writers is not None:
        # Same seam as the case-data route: a moved domain's later
        # frames must describe the footprint that produced them.
        relocation_runner.on_child_built.attach_writers(writers)

    # [tiles].  Absent -- the default -- this is an empty mapping and the
    # executor binds gpuwm.core.dycore.step for every grid, which is the
    # function it always bound: no planner is consulted and no tilestream
    # module is imported, so a resident forecast pays nothing for the mode
    # existing.  Configured, a domain that the planner says will not fit
    # resident gets a streamed stepper here or a loud refusal; what it never
    # gets is a silent resident run that dies at the allocation the mode was
    # turned on to avoid.  The builders are what makes "gets a streamed
    # stepper here" true rather than aspirational: until they were wired,
    # every configuration on this route took the refusal branch, including
    # the ones the mode exists for.  A NEST that fires is still refused, on
    # purpose -- see streaming.prepared_domain_builder.
    #
    # The decisions are recorded per grid because the stepper dict cannot
    # report them: a grid that declined to stream is simply missing from it,
    # which is also what a grid looks like when [tiles] was never
    # configured.  On a TREE that ambiguity is worse than on a single
    # domain -- `auto` can legitimately stream the parent and leave a small
    # nest resident, so "some grids are missing" is the NORMAL case and
    # cannot be read as an anomaly.  Only a per-grid verdict distinguishes
    # that from a run where auto declined on every grid.
    streaming_decisions: dict = {}
    steppers = streaming.steppers_for_tree(
        model, exp.tiles,
        builders=streaming.builders_for_tree(model, exp.tiles),
        decisions=streaming_decisions)
    streaming_report = streaming.streaming_receipt(
        exp.tiles, streaming_decisions)
    if streaming_report:
        print(f"prepared tree: {streaming_report['summary']}", flush=True)

    try:
        memory_watch.start()
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
                relocation_runner=relocation_runner,
                steppers=steppers,
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
                    relocation_runner=relocation_runner,
                    steppers=steppers,
                )
                writers.drain()
                wrfout_paths = writers.paths
        if relocation_runner is not None:
            relocation_runner.close_receipt(model)
    finally:
        memory_watch.stop()
    cp.cuda.Stream.null.synchronize()
    timing["forecast_execution"] = time.perf_counter() - forecast_started
    model._io_manager = None
    memory_watch.sample()

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
        # The health gate above still validates node.state and CANNOT be
        # folded (one block per whole field, no windowing), so under a host
        # store it passes on the t = 0 snapshot; the stability record below
        # is folded per tile and is the domain's.
        final_stability[f"d{grid_id:02d}"] = _strict_json(
            streaming.stability_observer(steppers.get(int(grid_id)))(
                node.state, node.cfg.run,
                boundary_width=node.cfg.run.spec_bdy_width))
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
        # Which install produced this tree of forecasts.  The
        # runtime_source_identity below hashes the implementation's
        # bytes; this names the install those bytes came out of, and
        # whether its two version claims agree with each other.
        "provenance": _provenance_receipt(),
        "execution_plan": inputs.execution_plan,
        "experiment": {
            "name": exp.name,
            "start_time": exp.start_time.isoformat(),
            "run_seconds": float(exp.run_seconds),
            "fingerprint": fingerprint,
            "domains": _domain_rows(exp),
        },
        # The physics-fidelity axis as this run RESOLVED it, not as it was
        # requested: the whole vector, every ledger entry, with the value
        # each one took.  A scoreboard row has to be able to say which arm
        # produced it without reading the configuration back, and an
        # ungoverned run says so in the same shape rather than by omission
        # (the morr_rimed_ice-into-receipt pattern, one axis wider).
        "physics_mode": exp.physics_mode.receipt(),
        # The initial-perturbation axis, same posture as physics_mode:
        # the arm says what it is in the receipt, not by omission.  None
        # when the config carries no block; configured, the full echo
        # plus per-domain application stats (also standalone in
        # evidence/initial-perturbation.json, written before
        # integration so a run that dies mid-flight still proves its
        # arm).
        "initial_perturbation": (
            None if exp.perturbation is None else {
                "config": exp.perturbation.receipt(),
                "domains": initial_perturbation_receipts,
            }),
        "restart_contract": {
            "mode": ("sealed-forcing-extension"
                     if sealed_forcing_extension else "exact-setup"),
            "restart_input": (
                None if restart is None else str(Path(restart).resolve())),
            # Non-None only when the restored checkpoint was written after
            # a nest relocation: the promises-nothing posture, stated in
            # the receipt of the run that crossed it.
            "relocation_crossed": getattr(
                model, "_restart_crossed_relocation", None),
        },
        # The [relocation] echo (None when the config never opted in).
        # A follow source reaches execution only over a verified statics
        # corridor (the preflight refuses corridor-less bundles), so a
        # non-None block is either bounds-only or carries the corridor
        # binding plus every move receipt the runner recorded.
        "relocation": (
            None if not exp.relocation.enabled else {
                "config": exp.relocation.receipt(),
                "statics_corridor": (
                    None if inputs.statics_corridor is None else {
                        "grid_id": int(
                            inputs.statics_corridor.geometry["grid_id"]),
                        "corridor_nx": int(
                            inputs.statics_corridor.geometry["corridor_nx"]),
                        "corridor_ny": int(
                            inputs.statics_corridor.geometry["corridor_ny"]),
                        "cache_sha256":
                            inputs.statics_corridor.cache_sha256,
                        "host_bytes": inputs.statics_corridor.host_bytes,
                    }),
                "receipts": _strict_json(list(
                    getattr(model, "_relocation_receipts", ()) or ())),
            }),
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
            "gpu_peak_used_bytes_observed": memory_watch.peak_bytes(
                "cuda_device_used"),
            "cupy_pool_peak_total_bytes_observed": memory_watch.peak_bytes(
                "cupy_pool_total"),
            "cupy_pool_peak_used_bytes_observed": memory_watch.peak_bytes(
                "cupy_pool_used"),
            "preflight_alloc_estimate_bytes": int(estimate.alloc_estimate_bytes),
            # What each number above actually measured, how often it was
            # sampled, and whether observation stayed complete.
            "gpu_peak_sampling": memory_watch.summary(),
            # Host-side, not VRAM: the corridor is cropped on the CPU
            # and the rebuilt child re-occupies the same device
            # footprint, so the GPU preflight estimate is unchanged.
            "statics_corridor_host_bytes": (
                None if inputs.statics_corridor is None
                else inputs.statics_corridor.host_bytes),
            # WHICH EXECUTION MODE the two numbers above describe.  The
            # estimate prices a whole resident tree; a streamed domain's
            # observed peak is a few tile buffers.  Absent this field the
            # gap between them reads as slack in the estimator.  Empty --
            # and the receipt therefore byte-identical to the one written
            # before streaming existed -- whenever [tiles] is off.
            "tiles": streaming.receipt_entry(
                exp.tiles, streaming_decisions),
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
    # Only when [tiles] was configured, so an unconfigured tree writes
    # the receipt it wrote before the mode existed -- see the same guard in
    # prepared_single_domain_forecast.
    if streaming_report:
        report["tiles"] = streaming_report
    _atomic_json(evidence / "run-receipt.json", report)
    emit_run_capsule(
        outdir, emission_site="prepared_domain_tree_forecast",
        run_context={
            "runner_route_and_io_mode": {
                "route": "prepared_domain_tree_forecast", "io_mode": io_mode},
            "output_and_diagnostic_mode": {"io_mode": io_mode},
            # Both DETERMINISM.md byte pins bind here, git or no git:
            # this route refuses to start unless the experiment TOML
            # matches --experiment-config-sha256, so the pin's value
            # exists on every run (the 4090 stress run's honesty
            # finding was this route's sibling reporting it
            # "unavailable" on the published wheel).
            "config_bytes": {
                "path": str(inputs.experiment_config),
                "sha256": inputs.authority_sha256["experiment_config"]},
            "input_artifact_bytes": dict(inputs.authority_sha256),
        },
        input_bytes={"entries": {
            **{name: {"algorithm": "sha256", "digest": digest}
               for name, digest in inputs.authority_sha256.items()},
            **{f"d{bundle.grid_id:02d}:{name}": {
                   "algorithm": "sha256", "digest": digest}
               for bundle in inputs.domains
               for name, digest in bundle.authority_sha256.items()},
        }},
        run_shape={
            "route": "prepared_domain_tree_forecast",
            "domain_count": len(exp.domains),
            "run_seconds": float(exp.run_seconds),
            "experiment_fingerprint": fingerprint,
            "domains": _domain_rows(exp),
        },
        output={"frames": outputs, "trajectory_digest": final_digests},
        receipts={"run_receipt": {
            "path": str((evidence / "run-receipt.json").resolve())}},
    )
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
    parser.add_argument(
        "--restart", type=Path,
        help="resume from any member of a gpuwmrst checkpoint set written "
             "by an earlier run of this prepared tree; only the forecast "
             "length (run_seconds) and the output/restart cadence "
             "(history_interval_s, restart_interval_s) may differ from "
             "the run that wrote it -- the same contract `gpuwm run "
             "--restart` publishes.  Anything else is refused by name")
    parser.add_argument(
        "--sealed-forcing-extension", action="store_true",
        help=("write/restore checkpoints using the explicit append-only "
              "forcing-prefix contract"))
    parser.add_argument("--health-debug", action="store_true")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv=None, *, observer=None) -> int:
    """The tree runner's command line.

    ``observer`` is forwarded to :func:`run_prepared_tree` and is
    otherwise inert: it lets a caller hosting this runner in its own
    process receive the progress it publishes plus each per-domain
    wrfout as it lands.  ``None`` -- every command-line invocation --
    changes nothing.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--show-capabilities"]:
        print(json.dumps(runner_capabilities(), sort_keys=True))
        return 0
    # Which tree is about to integrate this tree of domains.  Same
    # contract as the single-domain runner, including leaving
    # ``--show-capabilities`` above untouched.
    from gpuwm.provenance_gate import announce_for_main

    refusal = announce_for_main("gpuwm-prepared-tree-forecast")
    if refusal is not None:
        print(f"prepared_domain_tree_forecast: {refusal}", file=sys.stderr)
        return 2
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
        # An experimental component option warns on every front door it
        # can be selected through, and this runner is one of them: a
        # domain tree is not runnable through `gpuwm go`, which refuses
        # multi-domain configs and drives the single-domain runner.
        # Without this the tree path was the one way to select an
        # experimental closure and be told nothing.  One sentence, to
        # stderr, and the run continues (owner posture: warn-not-block).
        sentence = experimental_selection_sentence(
            domain.run for domain in inputs.experiment.domains)
        if sentence is not None:
            print(f"prepared tree: {sentence}", file=sys.stderr)
    except MissingTableAssets as error:
        print(f"prepared_domain_tree_forecast: refused: {error}",
              file=sys.stderr)
        return 2
    except ValueError as error:
        # Preflight is the stage whose whole job is to refuse before the
        # GPU is touched, and every one of its refusals is a ValueError
        # naming exactly what differs.  They used to escape as tracebacks
        # exiting 1 -- the same shape the run failures take -- so a
        # config problem and a crashed forecast were indistinguishable
        # to the caller.  Nothing has run here, so no failed-run receipt.
        print(f"prepared_domain_tree_forecast: refused: {error}",
              file=sys.stderr)
        return 2
    try:
        report = run_prepared_tree(
            inputs,
            output_directory=outdir,
            io_mode=args.io_mode,
            restart=args.restart,
            health_debug=args.health_debug,
            observer=observer,
            sealed_forcing_extension=args.sealed_forcing_extension,
        )
    except MissingTableAssets as error:
        # A refusal, not a failed run: no failed-run-receipt, because
        # nothing ran.  One sentence naming the table and the command
        # that stages it -- the shape every other guard in this main
        # already uses.
        print(f"prepared_domain_tree_forecast: refused: {error}",
              file=sys.stderr)
        return 2
    except RestartMismatchError as error:
        # Also a refusal, and the guard is doing exactly the right
        # thing -- but it used to arrive as a 40-line traceback exiting
        # 1, where every sibling refusal in this main is one sentence
        # exiting 2.  Nothing was integrated, so there is no failed run
        # to write a receipt about.
        print(f"prepared_domain_tree_forecast: --restart refused: {error}",
              file=sys.stderr)
        return 2
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
