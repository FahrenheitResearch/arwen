# tests/test_case_straka.py
import pytest
from conftest import assert_gates, requires_gpu

pytestmark = pytest.mark.gpu


@requires_gpu
def test_straka_900s_benchmarks(tmp_path):
    """Straka gates (single-sourced in gpuwm.verify.cases.straka.GATES):
    theta'_min / front position / symmetry / bounded w."""
    from gpuwm.core.dycore import isotropic_km_constants
    from gpuwm.verify.cases.straka import default_config, run
    cfg = default_config()
    assert isotropic_km_constants(
        cfg.khdif, cfg.kvdif, cfg.constant_k_prandtl) == (75.0,) * 4
    m = run(outdir=tmp_path)
    assert_gates("straka", m)
    assert (tmp_path / "straka_t900.png").exists()
    assert (tmp_path / "wrfout_straka.nc").exists()
