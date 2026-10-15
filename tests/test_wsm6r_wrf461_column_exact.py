"""Mechanical WSM6R gate against the committed WRF 4.6.1 column oracle.

Run GPU tests through the card queue. Strict mode requires every graded
field hash to match. RAINNCV has the documented per-step accumulation
divergence for multiple minor loops.
"""
import os
from pathlib import Path
import pytest

FIXTURE = Path(__file__).parent / "data" / "oracles" / "wsm6r"

def _device():
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU disabled for CPU run")
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("GPU required")
    return cp

@pytest.mark.gpu
def test_wsm6r_strict_full_field_hashes():
    _device()
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("run with GPUWM_WRF_EXACT=1")
    from tools.wsm6r_wrf461_column_oracle.run_gpu import check_fixture
    report = check_fixture(FIXTURE)
    assert report["strict_build"]
    assert report["excluded_columns"] == []
    assert report["total_mismatch"] == 0, report

@pytest.mark.gpu
def test_wsm6r_revised_term_one_bit_control():
    _device()
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("run with GPUWM_WRF_EXACT=1")
    from tools.wsm6r_wrf461_column_oracle.run_gpu import cmd_plant
    report = cmd_plant(FIXTURE)
    assert report["baseline_mismatch"] == 0, report
    assert report["total_mismatch"] > 0, report

@pytest.mark.gpu
def test_wsm6r_default_full_field_hashes():
    _device()
    if os.environ.get("GPUWM_WRF_EXACT") == "1":
        pytest.skip("default arithmetic requires a fresh process without strict mode")
    from tools.wsm6r_wrf461_column_oracle.run_gpu import check_fixture
    report = check_fixture(FIXTURE)
    assert not report["strict_build"]
    assert report["total_mismatch"] == 0, report
