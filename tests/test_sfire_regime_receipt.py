"""Fail-closed SFIRE word grading and corrected-reference source controls."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import requires_gpu
from tools.sfire_wrf471_oracle.regime_receipt import FIELDS, REGIMES, grade, passed, read_case, validate_regime_fuel


def test_one_ulp_fails_and_hashes_the_full_field():
    expected = np.ones((3, 4), dtype=np.float32)
    actual = expected.copy()
    actual[1, 2] = np.nextafter(actual[1, 2], np.float32(2))
    result = grade(actual, expected)
    assert result["max_ulp"] == result["different_words"] == 1
    assert result["actual_sha256"] != result["expected_sha256"]
    assert not passed(result)
    assert passed(grade(expected, expected))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_equal_nonfinite_is_not_a_pass(value):
    assert not passed(grade(np.array([value], np.float32), np.array([value], np.float32)))


def test_changed_fixture_is_refused(tmp_path):
    path = tmp_path / "case.npz"
    np.savez(path, x=np.ones(2, np.float32))
    with pytest.raises(ValueError, match="fixture hash differs"):
        read_case(tmp_path, "case", {"cases": {"case": {"sha256": "0"*64}}})


def test_native_constant_fuel_reset_cannot_mislabel_timber():
    with pytest.raises(ValueError, match="requires fuel category 8"):
        validate_regime_fuel("timber_slope", np.ones((3, 4), np.float32))
    validate_regime_fuel("timber_slope", np.full((3, 4), 8, np.float32))


def test_crashed_negative_control_is_not_a_measured_kernel_fault():
    from tools.sfire_wrf471_oracle.validate_planted import validate
    with pytest.raises(ValueError, match="did not finish with measured mismatches"):
        validate({"modes": {"strict": {"status": "FAIL", "exit_code": 1,
                                      "error": "CompileException: planted kernel did not compile"}}})


def test_missing_regimes_exit_one_in_both_modes(tmp_path):
    out = tmp_path / "missing.json"
    run = subprocess.run([sys.executable, "-m", "tools.sfire_wrf471_oracle.regime_receipt",
        "--fixtures", str(tmp_path / "absent"), "--out", str(out)], capture_output=True, text=True)
    assert run.returncode == 1
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "FAIL"
    assert set(receipt["modes"]) == {"strict", "default"}
    assert all(v["exit_code"] == 1 for v in receipt["modes"].values())


@requires_gpu
@pytest.mark.gpu
def test_twenty_step_corrected_regimes():
    from tools.sfire_wrf471_oracle.regime_receipt import replay
    root = os.environ.get("SFIRE_REGIME_FIXTURES")
    if not root:
        pytest.skip("Build the corrected native corpus and set SFIRE_REGIME_FIXTURES")
    receipt = replay(Path(root))
    assert receipt["status"] == "PASS", receipt
    assert set(receipt["regimes"]) == set(REGIMES)
    for regime in receipt["regimes"].values():
        assert len(regime["steps"]) >= 20
        assert all(set(step["fields"]) == set(FIELDS) for step in regime["steps"])
    assert receipt["compiles"]
