# tests/test_case_straka.py
import pytest
from conftest import assert_gates, requires_gpu

pytestmark = pytest.mark.gpu


@requires_gpu
def test_straka_900s_benchmarks(tmp_path):
    """Straka gates (single-sourced in gpuwm.verify.cases.straka.GATES):
    theta'_min / front position / symmetry / bounded w."""
    from gpuwm.verify.cases.straka import run
    m = run(outdir=tmp_path)
    assert_gates("straka", m)
    assert (tmp_path / "straka_t900.png").exists()
    assert (tmp_path / "wrfout_straka.nc").exists()
