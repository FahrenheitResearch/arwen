# tests/test_case_hill2d.py
import pytest
from conftest import assert_gates, requires_gpu

pytestmark = pytest.mark.gpu


@requires_gpu
def test_hill2d_mountain_wave_benchmarks(tmp_path):
    """Queney mountain-wave gates (single-sourced in
    gpuwm.verify.cases.hill2d.GATES): steady-state w correlates with the
    analytic solution below the sponge, the wave momentum flux is
    constant with height, downward (drag), and of the linear-theory
    magnitude (mflux_ratio = mflux_mean / mflux_linear in (0.5, 1.5))."""
    from gpuwm.verify.cases.hill2d import run
    m = run(outdir=tmp_path)
    assert_gates("hill2d", m)
    assert (tmp_path / "hill2d_w.png").exists()
    assert (tmp_path / "wrfout_hill2d.nc").exists()
