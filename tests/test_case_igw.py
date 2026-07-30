# tests/test_case_igw.py
import pytest
from conftest import assert_gates, requires_gpu

pytestmark = pytest.mark.gpu


@requires_gpu
def test_igw_3000s_benchmarks(tmp_path):
    """SK94 inertia-gravity-wave gates (single-sourced in
    gpuwm.verify.cases.igw.GATES): theta' extrema, packet centroid ~ u*t,
    bounded w."""
    from gpuwm.verify.cases.igw import run
    m = run(outdir=tmp_path)
    assert_gates("igw", m)
    assert (tmp_path / "igw_t3000.png").exists()
    assert (tmp_path / "wrfout_igw.nc").exists()
