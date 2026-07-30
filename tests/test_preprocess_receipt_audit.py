from __future__ import annotations

import hashlib
import json

from gpuwm.ingest.cpu_backend import CPU_BACKEND_ABI
from gpuwm.ingest.preprocess_backend import (
    PREPROCESS_IMPLEMENTATION_SCHEMA,
    PSFC_MAPPING_POLICY,
    VERTICAL_STENCIL_POLICY,
)
from gpuwm.preprocess_receipt_audit import (
    ERA5_PROOF_SCHEMA,
    audit_era5_preprocess_receipt,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()


def _artifact(path, root):
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _valid_fixture(tmp_path, monkeypatch):
    source_manifest = tmp_path / "source-input-manifest.json"
    input_file_digest = hashlib.sha256(b"source-grib").hexdigest()
    manifest = {
        "schema": "gpuwm-era5-direct-input-manifest-v1",
        "files": {
            "grib": {"name": "source.grb", "sha256": input_file_digest},
        },
    }
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    for name, payload in (
        ("native-static.npz", b"static"),
        ("geometry-receipt.json", b"geometry"),
    ):
        (tmp_path / name).write_bytes(payload)
    wrf = tmp_path / "wrf-native-input"
    wrf.mkdir()
    (wrf / "wrfinput_d01").write_bytes(b"wrfinput")
    (wrf / "wrfbdy_d01").write_bytes(b"wrfbdy")
    cache = tmp_path / "prepared-cache"
    cache.mkdir()
    (cache / "header.json").write_text('{"identity":{}}')

    files = {
        "gpuwm/ingest/horiz.py": {
            "bytes": 7,
            "sha256": hashlib.sha256(b"horiz-v2").hexdigest(),
        },
    }
    preprocessing = {
        "schema": PREPROCESS_IMPLEMENTATION_SCHEMA,
        "backend": "cpu",
        "implementation": "rust-scoped-threads-fp32-v1",
        "workers": 8,
        "host_cpu_count": 16,
        "bridge": {
            "name": "libgpuwm_preprocess_cpu.so",
            "sha256": hashlib.sha256(b"bridge").hexdigest(),
            "abi_version": CPU_BACKEND_ABI,
            "required_abi_version": CPU_BACKEND_ABI,
        },
        "contracts": {
            "surface_pressure_mapping": {
                "policy": PSFC_MAPPING_POLICY,
                "output_dtype": "float32",
            },
            "vertical_stencil": {
                "policy": VERTICAL_STENCIL_POLICY,
                "zap_close_levels_pa": 500.0,
                "predicate": "separation < zap_close_levels",
            },
        },
        "implementation_tree": {
            "schema": "gpuwm-preprocess-source-tree-v1",
            "sha256": hashlib.sha256(_canonical(files)).hexdigest(),
            "files": files,
        },
    }
    proof = {
        "schema": ERA5_PROOF_SCHEMA,
        "input_manifest_sha256": _sha(source_manifest),
        "preprocessing": preprocessing,
        "preprocessing_receipt_sha256": hashlib.sha256(
            _canonical(preprocessing)).hexdigest(),
        "source_inputs": {
            "manifest_schema": manifest["schema"],
            "manifest_sha256": _sha(source_manifest),
            "files": manifest["files"],
        },
        "initialization_artifacts": {
            "source_manifest": _artifact(source_manifest, tmp_path),
            "static_cache": _artifact(tmp_path / "native-static.npz", tmp_path),
            "geometry_receipt": _artifact(
                tmp_path / "geometry-receipt.json", tmp_path),
            "prepared_cache": {
                "path": "prepared-cache",
                "content_sha256": hashlib.sha256(b"cache").hexdigest(),
                "payload_bytes": 123,
            },
            "wrf_files": {
                name: _artifact(wrf / name, tmp_path)
                for name in ("wrfinput_d01", "wrfbdy_d01")
            },
        },
    }
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    class FakeReader:
        def __init__(self, path, *, expected_identity):
            assert path == cache
            assert expected_identity == {}

        def verify_all(self):
            return {
                "content_sha256": hashlib.sha256(b"cache").hexdigest(),
                "payload_bytes": 123,
                "array_count": 7,
            }

    monkeypatch.setattr(
        "gpuwm.preprocess_receipt_audit.PreparedCacheReader", FakeReader)
    return proof_path


def test_v2_cpu_receipt_recomputes_all_bound_artifacts(tmp_path, monkeypatch):
    proof = _valid_fixture(tmp_path, monkeypatch)
    report = audit_era5_preprocess_receipt(
        proof, output_root=tmp_path, verify_runtime=False)
    assert report["status"] == "PASS", report


def test_old_cpu_receipt_without_discontinuity_fix_fails_closed(
        tmp_path, monkeypatch):
    proof_path = _valid_fixture(tmp_path, monkeypatch)
    proof = json.loads(proof_path.read_text())
    proof["schema"] = "gpuwm-era5-direct-wrf-proof-v1"
    proof["preprocessing"].pop("schema")
    proof["preprocessing"].pop("contracts")
    proof["preprocessing_receipt_sha256"] = hashlib.sha256(
        _canonical(proof["preprocessing"])).hexdigest()
    proof_path.write_text(json.dumps(proof))

    report = audit_era5_preprocess_receipt(
        proof_path, output_root=tmp_path, verify_runtime=False)
    assert report["status"] == "FAIL"
    failed = {check["name"] for check in report["checks"]
              if check["status"] == "FAIL"}
    assert {"proof-schema", "preprocessing-schema", "psfc-mapping-policy",
            "vertical-stencil-policy"} <= failed


def test_artifact_summary_is_never_trusted_after_file_mutation(
        tmp_path, monkeypatch):
    proof = _valid_fixture(tmp_path, monkeypatch)
    (tmp_path / "wrf-native-input" / "wrfinput_d01").write_bytes(b"edited")
    report = audit_era5_preprocess_receipt(
        proof, output_root=tmp_path, verify_runtime=False)
    assert report["status"] == "FAIL"
    assert any(check["name"] == "wrf-artifact-wrfinput_d01"
               and check["status"] == "FAIL"
               for check in report["checks"])
