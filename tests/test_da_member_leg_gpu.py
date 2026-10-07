"""Independent transport checks and the explicit node-only packed GPU proof."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools import da_member_leg_proof as proof


def test_identical_values_with_different_dtypes_are_not_byte_identity(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, soil=np.array([1.0, 2.0], dtype=np.float64))
    np.savez(right, soil=np.array([1.0, 2.0], dtype=np.float32))
    comparison = proof.compare_array_files(left, right)
    assert not comparison["equal"]
    assert comparison["mismatches"] == [{"array": "soil", "reason": "dtype", "expected": "<f8", "actual": "<f4"}]


def test_equal_flat_values_with_different_shapes_are_not_identity(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, atmosphere=np.arange(4).reshape(2, 2))
    np.savez(right, atmosphere=np.arange(4).reshape(1, 4))
    comparison = proof.compare_array_files(left, right)
    assert not comparison["equal"]
    assert comparison["mismatches"][0]["reason"] == "shape"


def test_one_bit_in_a_physics_array_fails_the_proof(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    physical = np.array([287.0, 288.0], dtype=np.float64)
    changed = physical.copy()
    changed.view(np.uint64)[1] ^= np.uint64(1)
    np.savez(left, soil_temperature=physical)
    np.savez(right, soil_temperature=changed)
    comparison = proof.compare_array_files(left, right)
    assert not comparison["equal"]
    assert comparison["mismatches"] == [{"array": "soil_temperature", "reason": "raw bytes"}]


def test_restart_metadata_cannot_hide_missing_physics_payload(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, soil=np.zeros(3), rain=np.array([1.0]))
    np.savez(right, soil=np.zeros(3))
    comparison = proof.compare_array_files(left, right)
    assert not comparison["equal"]
    assert comparison["mismatches"] == [{"array": "rain", "reason": "missing array"}]


def test_byte_comparison_preserves_nan_payload_identity(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    a = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)
    b = np.array([0x7ff8000000000002], dtype=np.uint64).view(np.float64)
    assert np.isnan(a[0]) and np.isnan(b[0])
    np.savez(left, payload=a)
    np.savez(right, payload=b)
    assert not proof.compare_array_files(left, right)["equal"]


def test_cpu_plan_binds_all_four_members_and_worker_clock(tmp_path):
    jobs = proof.make_jobs(tmp_path, 0)
    assert [job["member"] for job in jobs] == [0, 1, 2, 3]
    for job in jobs:
        request_path = Path(job["argv"][job["argv"].index("--request") + 1])
        request = json.loads(request_path.read_text())
        assert request["t_start"] == job["t_start"] == 0.0
        assert request["t_end"] == job["t_end"] == 60.0
        assert not request["nested"]
        assert request["previous_result"] is None
        assert request["pending"] is None
        assert job["request_hash"] == proof.digest_file(request_path)


def test_resume_plan_starts_at_saved_clock_with_unapplied_pending(tmp_path):
    jobs = proof.make_jobs(tmp_path, 1, previous_results=[f"restart-{m}.json" for m in range(4)],
                           pending=[f"pending-{m}.npz" for m in range(4)], resumed=True)
    for job in jobs:
        request = proof.read_json(job["argv"][job["argv"].index("--request") + 1])
        assert request["leg"] == 0
        assert request["absolute_leg_number"] == 1
        assert request["resumed"] is True
        assert request["nested"] is True
        assert request["t_start"] == 60.0 and request["t_end"] == 120.0
        assert request["pending"] == f"pending-{job['member']}.npz"


def test_gpu_guard_refuses_without_opening_a_device(monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setenv(proof.OWNER_ENV, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000000")
    with pytest.raises(RuntimeError, match="GPUWM_NO_LOCAL_GPU"):
        proof._gpu_guard()


def test_gpu_guard_requires_owner_check_and_complete_uuid(monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "0")
    monkeypatch.delenv(proof.OWNER_ENV, raising=False)
    with pytest.raises(RuntimeError, match="OWNER check"):
        proof._gpu_guard()
    monkeypatch.setenv(proof.OWNER_ENV, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="complete CUDA_VISIBLE_DEVICES"):
        proof._gpu_guard()


@pytest.mark.gpu
def test_actual_four_member_packed_gpu_legs_analysis_birth_failure_resume(tmp_path):
    if os.environ.get("GPUWM_RUN_DA_MEMBER_LEG_PROOF") != "1":
        pytest.skip("explicit node OWNER session sets GPUWM_RUN_DA_MEMBER_LEG_PROOF=1")
    uuid = proof._gpu_guard()
    receipt = proof.run_suite(tmp_path, uuid)
    assert receipt["status"] == "pass"
    assert receipt["members"] == 4
    assert receipt["observed_legs"] == 2
    assert len(receipt["comparisons"]) == 12
    assert all(row["equal"] for row in receipt["comparisons"])
    assert all(row["equal"] for row in receipt["resume_comparisons"])
    assert receipt["failed_wave_inputs_unchanged"]
    assert all(analysis["max_abs_increment"] > 0 for arm in receipt["analyses"].values() for analysis in arm)
