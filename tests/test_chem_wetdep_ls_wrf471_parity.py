"""WRF v4.7.1 wetdep_ls: both vertical counts and five carried steps."""
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.verify.chem_oracle import ORACLE_ROOT, load, ulp_table
from gpuwm.verify.chem_wetdep_ls_ref import wetdep_ls

ROOT = ORACLE_ROOT / "smoke" / "wetdep_ls"
# Registry GOCART_SIMPLE, not a species decision in process code.
ALPHA = np.array([0,1,0,0,.5,.5,0,.8,0,.8,.5,.5,.5,.5,.5,1,1,1,1], np.float32)


def inputs(case):
    var = case["initial"].transpose(3, 1, 2, 0).copy()
    met = {n: case[n].transpose(1, 2, 0).copy() for n in ("qc", "rho", "dz", "w")}
    return var, met, case["rain"].T.copy()


@pytest.mark.parametrize("name", ("nz49", "nz59"))
def test_cpu_parity(name):
    case = load(ROOT)[name]
    var, met, rain = inputs(case)
    removed = np.zeros((19, 1, 12), np.float32)
    for step in range(1, 6):
        var, mass = wetdep_ls(var, rain, **met, dt=case["dt"], alpha=ALPHA)
        removed += mass
        expected = case[f"step{step}"].transpose(3, 1, 2, 0)
        measured = ulp_table(var, expected)
        assert measured == {"max_ulp": 0, "n_nonzero": 0, "n": var.size}, measured
        measured = ulp_table(mass,case[f"removed{step}"].transpose(2,1,0))
        assert measured == {"max_ulp": 0, "n_nonzero": 0, "n": mass.size}, measured
    assert np.all(removed >= 0)
    assert np.any(removed > 0)
    np.testing.assert_array_equal(var[:, -2:], case["initial"].transpose(3,1,2,0)[:, -2:])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", ("nz49", "nz59"))
def test_kernel_parity(name):
    import cupy as cp
    from gpuwm.core.kernels import load_module
    case = load(ROOT)[name]
    host, met, rain = inputs(case)
    var = cp.asarray(host)
    dev = {n: cp.asarray(v) for n, v in met.items()}
    a = cp.asarray(ALPHA)
    removed = cp.zeros((19, 1, 12), cp.float32)
    kernel = load_module("chem_wetdep_ls").get_function("chem_wetdep_ls")
    for step in range(1, 6):
        removed.fill(0)
        kernel((1,), (128,), (var, cp.asarray(rain), dev["qc"], dev["rho"],
               dev["rho"], dev["dz"], dev["w"], a, removed, case["dt"],
               np.int32(host.shape[1]), np.int32(12), np.int32(19)))
        measured = ulp_table(cp.asnumpy(var), case[f"step{step}"].transpose(3,1,2,0))
        assert measured == {"max_ulp": 0, "n_nonzero": 0, "n": host.size}, measured
        measured=ulp_table(cp.asnumpy(removed),case[f"removed{step}"].transpose(2,1,0))
        assert measured=={"max_ulp":0,"n_nonzero":0,"n":removed.size},measured
