from datetime import datetime, timezone
import json

import pytest

from gpuwm.ensemble.acquisition_binding import (
    AcquiredSourceBinding, manifest_trajectory, mapped_input_closure,
)
from gpuwm.ensemble.physical_store import digest_file
from gpuwm.ensemble.recipes import SourceTrajectory


def _write(tmp_path, name, document):
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return {"path": str(path), "sha256": digest_file(path)}


def _mapped(tmp_path):
    payload = {"name": "pairs/member.grib2", "bytes": 10, "sha256": "a" * 64}
    document = {"schema": "gpuwm-fetch-route-manifest-v1",
                "request": {"source": "gefs", "cycle": "2024-05-21T12Z", "member": "p19"},
                "composed": [payload]}
    fetch = _write(tmp_path, "fetch-manifest.json", document)
    trajectory = SourceTrajectory("gefs", datetime(2024, 5, 21, 12, tzinfo=timezone.utc), "p19")
    receipt = {"status": "PASS", "source": "gefs", "cycle": trajectory.cycle.isoformat(),
               "member": "p19", "manifest_sha256": fetch["sha256"], "acquisition_root": str(tmp_path)}
    native = _write(tmp_path, "native-verification.json", {**receipt, "native_member_check": "PASS", "payload_files": [{"path": payload["name"], "bytes": 10, "sha256": payload["sha256"]}]})
    manifest = _write(tmp_path, "input-manifest.json", {
        "primary_files": [{"path": payload["name"], "bytes": 10, "sha256": payload["sha256"]}],
        "supplements": {}})
    receipt["artifacts"] = {"verification_receipt": native, "mapped_input_manifest": manifest}
    return AcquiredSourceBinding(trajectory, fetch["path"], fetch["sha256"], receipt)


def test_original_generic_fetch_envelope_retains_member_and_cycle(tmp_path):
    binding = _mapped(tmp_path)
    assert binding.verify()["trajectory_sha256"] == binding.trajectory.identity
    binding.verification["member"] = "p18"
    with pytest.raises(ValueError, match="another source manifest"):
        binding.verify()


def test_native_artifacts_are_rehashed_at_launch(tmp_path):
    binding = _mapped(tmp_path)
    path = tmp_path / "native-verification.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        binding.verify()


def test_verification_must_pass_and_pin_native_files(tmp_path):
    binding = _mapped(tmp_path)
    binding.verification["status"] = "running"
    with pytest.raises(ValueError, match="passing native"):
        binding.verify()
    binding.verification["status"] = "PASS"
    binding.verification["artifacts"] = {}
    with pytest.raises(ValueError, match="pinned native artifacts"):
        binding.verify()


def test_physical_donor_cannot_substitute_another_input_manifest(tmp_path):
    binding = _mapped(tmp_path)
    binding.verification["artifacts"]["physical_manifest"] = _write(
        tmp_path, "physical-store.json", {"source": {"input_manifest_sha256": "f" * 64}})
    with pytest.raises(ValueError, match="physical donor belongs"):
        binding.verify()


def test_native_mapped_payload_catalog_checks_supplements_and_complete_paths():
    acquisition = {"files": [{"relpath": "upstream/a/source.grib2", "bytes": 3, "sha256": "a"*64}],
                   "composed": [{"name": "pairs/a.grib2", "bytes": 5, "sha256": "b"*64}]}
    inputs = {"primary_files": [{"path": "pairs/a.grib2", "bytes": 5, "sha256": "b"*64}],
              "supplements": {"surface": [{"path": "upstream/a/source.grib2", "bytes": 3, "sha256": "a"*64}]}}
    assert len(mapped_input_closure(acquisition, inputs, "/source")) == 2
    inputs["supplements"]["surface"][0]["path"] = "upstream/b/source.grib2"
    with pytest.raises(ValueError, match="differs from the original"):
        mapped_input_closure(acquisition, inputs, "/source")


def test_native_checksum_and_original_report_are_bound(tmp_path):
    document = {"source": "hrrr", "cycle": "2024-05-21T12:00:00Z",
                "files": [{"name": "hrrr.grib2", "sha256": "a"*64, "role": "atmosphere"}]}
    fetch = _write(tmp_path, "fetch-manifest.json", document)
    hashes = tmp_path / "SHA256SUMS"
    hashes.write_text("a"*64 + "  ./hrrr.grib2\n")
    raw = {"path": str(hashes), "sha256": digest_file(hashes)}
    report = _write(tmp_path, "report.json", {"source_hash_preflight": {
        "status": "PASS", "manifest_sha256": raw["sha256"]}})
    receipt = {**document, "status": "PASS", "manifest_sha256": fetch["sha256"],
               "artifacts": {"raw_hash_manifest": raw, "native_report": report}}
    binding = AcquiredSourceBinding(manifest_trajectory(document), fetch["path"], fetch["sha256"], receipt)
    binding.verify()
    receipt["artifacts"]["native_report"] = _write(tmp_path, "other-report.json", {
        "source_hash_preflight": {"status": "PASS", "manifest_sha256": "b"*64}})
    with pytest.raises(ValueError, match="does not verify"):
        binding.verify()


def test_deterministic_empty_member_and_equivalent_cycle_normalize():
    document = {"source": "hrrr", "cycle": "2024-05-21T12:00:00Z", "member": ""}
    assert manifest_trajectory(document).member is None
    assert manifest_trajectory({"schema": "gpuwm-fetch-route-manifest-v1",
        "request": document, "cycle": "2024-05-21T12+00:00", "member": None}).member is None
    with pytest.raises(ValueError, match="conflicts"):
        manifest_trajectory({"schema": "gpuwm-fetch-route-manifest-v1",
            "request": {"source": "gefs", "cycle": document["cycle"], "member": "p19"},
            "member": "p18"})
