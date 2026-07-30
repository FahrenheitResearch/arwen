"""Fail-closed audit for public native-preprocessing proof receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from gpuwm.ingest.cpu_backend import CPU_BACKEND_ABI, resolve_cpu_bridge
from gpuwm.ingest.prepared_cache import PreparedCacheReader
from gpuwm.ingest.preprocess_backend import (
    PREPROCESS_IMPLEMENTATION_SCHEMA,
    PSFC_MAPPING_POLICY,
    VERTICAL_STENCIL_POLICY,
)


AUDIT_SCHEMA = "gpuwm-era5-preprocess-receipt-audit-v1"
ERA5_PROOF_SCHEMA = "gpuwm-era5-direct-wrf-proof-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _safe_child(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("artifact path is missing")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path {raw!r}")
    root = root.resolve()
    candidate = (root / relative).resolve()
    candidate.relative_to(root)
    return candidate


def audit_era5_preprocess_receipt(
        proof_path: Path | str, *, output_root: Path | str | None = None,
        verify_runtime: bool = True,
        runtime_root: Path | str | None = None,
        cpu_bridge: Path | str | None = None,
) -> dict[str, object]:
    """Recompute receipt, source-tree, cache, and final-artifact evidence.

    Version-1 ERA5 CPU receipts fail by design: they do not bind the shared
    PSFC mapping policy that removes the backend-dependent 500-Pa stencil
    discontinuity.  ``verify_runtime=False`` is available only for inspecting
    an archived receipt when its original executable is unavailable; the
    default public audit verifies the current source tree and CPU bridge too.
    """

    proof_path = Path(proof_path).resolve()
    root = (proof_path.parent if output_root is None
            else Path(output_root)).resolve()
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    try:
        proof_bytes = proof_path.read_bytes()
        proof = json.loads(proof_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "schema": AUDIT_SCHEMA,
            "status": "FAIL",
            "proof": str(proof_path),
            "checks": [{
                "name": "proof-readable",
                "status": "FAIL",
                "detail": str(exc),
            }],
        }

    schema_ok = proof.get("schema") == ERA5_PROOF_SCHEMA
    record(
        "proof-schema", schema_ok,
        f"expected {ERA5_PROOF_SCHEMA}, got {proof.get('schema')!r}")

    preprocessing = proof.get("preprocessing")
    preprocessing_ok = isinstance(preprocessing, dict)
    record("preprocessing-object", preprocessing_ok,
           "preprocessing receipt must be an object")
    if not preprocessing_ok:
        preprocessing = {}
    backend = preprocessing.get("backend")
    record("backend-explicit", backend in {"cpu", "cuda"},
           f"backend={backend!r}")
    record(
        "preprocessing-schema",
        preprocessing.get("schema") == PREPROCESS_IMPLEMENTATION_SCHEMA,
        "old receipts lacking the PSFC discontinuity fix are rejected")

    observed_receipt_sha = hashlib.sha256(
        _canonical(preprocessing)).hexdigest()
    record(
        "preprocessing-receipt-sha256",
        proof.get("preprocessing_receipt_sha256") == observed_receipt_sha,
        f"recomputed={observed_receipt_sha}")

    contracts = preprocessing.get("contracts")
    contracts = contracts if isinstance(contracts, dict) else {}
    psfc = contracts.get("surface_pressure_mapping")
    psfc = psfc if isinstance(psfc, dict) else {}
    vertical = contracts.get("vertical_stencil")
    vertical = vertical if isinstance(vertical, dict) else {}
    record(
        "psfc-mapping-policy",
        psfc.get("policy") == PSFC_MAPPING_POLICY,
        f"expected {PSFC_MAPPING_POLICY}, got {psfc.get('policy')!r}")
    record(
        "vertical-stencil-policy",
        (vertical.get("policy") == VERTICAL_STENCIL_POLICY
         and vertical.get("predicate")
         == "separation < zap_close_levels"
         and vertical.get("zap_close_levels_pa") == 500.0),
        "requires the exact WRF strict 500-Pa predicate")

    tree = preprocessing.get("implementation_tree")
    tree = tree if isinstance(tree, dict) else {}
    files = tree.get("files")
    files = files if isinstance(files, dict) else {}
    entries_ok = bool(files)
    for name, spec in files.items():
        entries_ok &= (
            isinstance(name, str) and bool(name)
            and isinstance(spec, dict)
            and isinstance(spec.get("bytes"), int)
            and spec.get("bytes", -1) >= 0
            and _is_digest(spec.get("sha256")))
    tree_sha = hashlib.sha256(_canonical(files)).hexdigest()
    record(
        "implementation-tree-receipt",
        (entries_ok
         and tree.get("schema") == "gpuwm-preprocess-source-tree-v1"
         and tree.get("sha256") == tree_sha),
        f"recomputed={tree_sha}")

    if backend == "cpu":
        bridge = preprocessing.get("bridge")
        bridge = bridge if isinstance(bridge, dict) else {}
        record(
            "cpu-bridge-version",
            (bridge.get("abi_version") == CPU_BACKEND_ABI
             and bridge.get("required_abi_version") == CPU_BACKEND_ABI
             and _is_digest(bridge.get("sha256"))),
            f"required ABI={CPU_BACKEND_ABI}")

    source_inputs = proof.get("source_inputs")
    source_inputs = source_inputs if isinstance(source_inputs, dict) else {}
    manifest_files = source_inputs.get("files")
    manifest_files = manifest_files if isinstance(manifest_files, dict) else {}
    source_entries_ok = bool(manifest_files) and all(
        isinstance(spec, dict)
        and isinstance(spec.get("name"), str)
        and _is_digest(spec.get("sha256"))
        for spec in manifest_files.values())
    record("source-hash-inventory", source_entries_ok,
           f"bound roles={sorted(manifest_files)}")

    artifacts = proof.get("initialization_artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}

    def verify_file_artifact(name: str, spec: Mapping[str, object]) -> Path | None:
        try:
            path = _safe_child(root, spec.get("path"))
            actual_bytes = path.stat().st_size
            actual_sha = _sha256(path)
            passed = (actual_bytes == spec.get("bytes")
                      and actual_sha == spec.get("sha256"))
            record(name, passed,
                   f"path={path}; bytes={actual_bytes}; sha256={actual_sha}")
            return path if passed else None
        except (OSError, ValueError) as exc:
            record(name, False, str(exc))
            return None

    source_manifest_spec = artifacts.get("source_manifest")
    source_manifest_spec = (
        source_manifest_spec if isinstance(source_manifest_spec, dict) else {})
    source_manifest_path = verify_file_artifact(
        "source-manifest-artifact", source_manifest_spec)
    if source_manifest_path is not None:
        try:
            portable_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8"))
            matches = (
                portable_manifest.get("schema")
                == source_inputs.get("manifest_schema")
                and portable_manifest.get("files") == manifest_files
                and _sha256(source_manifest_path)
                == source_inputs.get("manifest_sha256")
                == proof.get("input_manifest_sha256"))
            record("source-manifest-content", matches,
                   "portable manifest must match both embedded inventory and hash")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            record("source-manifest-content", False, str(exc))

    for key in ("static_cache", "geometry_receipt"):
        spec = artifacts.get(key)
        verify_file_artifact(
            f"{key.replace('_', '-')}-artifact",
            spec if isinstance(spec, dict) else {})

    wrf_files = artifacts.get("wrf_files")
    wrf_files = wrf_files if isinstance(wrf_files, dict) else {}
    record("wrf-artifact-inventory", set(wrf_files) == {
        "wrfinput_d01", "wrfbdy_d01"}, f"files={sorted(wrf_files)}")
    for name, raw_spec in sorted(wrf_files.items()):
        verify_file_artifact(
            f"wrf-artifact-{name}",
            raw_spec if isinstance(raw_spec, dict) else {})

    cache_spec = artifacts.get("prepared_cache")
    cache_spec = cache_spec if isinstance(cache_spec, dict) else {}
    try:
        cache_path = _safe_child(root, cache_spec.get("path"))
        header = json.loads(
            (cache_path / "header.json").read_text(encoding="utf-8"))
        reader = PreparedCacheReader(
            cache_path, expected_identity=header.get("identity"))
        verified = reader.verify_all()
        passed = (
            verified["content_sha256"] == cache_spec.get("content_sha256")
            and verified["payload_bytes"] == cache_spec.get("payload_bytes"))
        record("prepared-cache-artifact", passed,
               f"arrays={verified['array_count']}; "
               f"content_sha256={verified['content_sha256']}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        record("prepared-cache-artifact", False, str(exc))

    if verify_runtime and files:
        runtime = (Path(runtime_root).resolve() if runtime_root is not None
                   else Path(__file__).resolve().parents[1])
        for name, spec in sorted(files.items()):
            try:
                path = _safe_child(runtime, name)
                passed = (path.stat().st_size == spec["bytes"]
                          and _sha256(path) == spec["sha256"])
                record(f"runtime-source-{name}", passed, f"path={path}")
            except (OSError, ValueError, KeyError) as exc:
                record(f"runtime-source-{name}", False, str(exc))
        if backend == "cpu":
            bridge_spec = preprocessing.get("bridge")
            bridge_spec = bridge_spec if isinstance(bridge_spec, dict) else {}
            try:
                path = resolve_cpu_bridge(cpu_bridge)
                record(
                    "runtime-cpu-bridge",
                    (_sha256(path) == bridge_spec.get("sha256")
                     and path.stat().st_size > 0),
                    f"path={path}; sha256={_sha256(path)}")
            except (OSError, ValueError) as exc:
                record("runtime-cpu-bridge", False, str(exc))

    overall = bool(checks) and all(
        check["status"] == "PASS" for check in checks)
    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if overall else "FAIL",
        "proof": {
            "path": str(proof_path),
            "bytes": len(proof_bytes),
            "sha256": hashlib.sha256(proof_bytes).hexdigest(),
        },
        "output_root": str(root),
        "backend": backend,
        "runtime_verified": bool(verify_runtime),
        "checks": checks,
    }


__all__ = [
    "AUDIT_SCHEMA",
    "ERA5_PROOF_SCHEMA",
    "audit_era5_preprocess_receipt",
]
