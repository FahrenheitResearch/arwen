"""The real physics preparation must use WRF's float32 Exner words."""
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from conftest import HAS_GPU


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="requires a queued CUDA card")
def test_physics_exner_and_temperature_match_live_fortran(tmp_path):
    from gpuwm.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("scalar Fortran power is the strict arithmetic contract")
    root = os.environ.get("WRF_SOURCE_ROOT")
    if not root or not shutil.which("gfortran"):
        pytest.skip("requires WRF_SOURCE_ROOT and gfortran for the live WRF expression")
    source = Path(root) / "share" / "module_model_constants.F"
    if not source.is_file():
        pytest.skip(f"missing WRF constants source: {source}")
    import cupy as cp
    from tilestream import harness
    from gpuwm.core import constants as c
    from gpuwm.core.physics import _prepare_atmosphere, _physics_exner

    cfg = harness.make_config(nx=12, ny=12, nz=16, moist=True, mp_physics=10)
    state = harness.make_state(cfg, seed=104)
    p = cp.asnumpy(state.p).ravel()
    theta = cp.asnumpy(state.total_theta()).ravel()
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(np.array([p.size], "<i4").tobytes() + p.tobytes() + theta.tobytes())
    driver = Path(__file__).resolve().parents[1] / "tools" / "wrf_exact_localize" / "phy_thermodynamics.F90"
    subprocess.run(["gfortran", "-O0", "-cpp", "-ffp-contract=off", "-ffree-form",
                    "-ffree-line-length-none", str(source), str(driver), "-o", "oracle"],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([str(tmp_path / "oracle"), "input.bin", "output.bin"],
                   cwd=tmp_path, check=True, capture_output=True)
    expected = np.fromfile(tmp_path / "output.bin", "<f4").reshape(2, -1)
    actual = _prepare_atmosphere(state)
    for n, row in (("exner", 0), ("temperature", 1)):
        got = cp.asnumpy(actual[n]).ravel()
        np.testing.assert_array_equal(got.view(np.uint32), expected[row].view(np.uint32),
                                      err_msg=f"phy_prep {n} input words differ")
    # The old expression is a real negative control on the same state.
    old = cp.asnumpy((state.p / np.float32(c.P0)) ** np.float32(c.RCP)).ravel()
    assert (old.view(np.uint32) != expected[0].view(np.uint32)).any()
    # Ensemble preparation writes into an already priced output buffer.
    out = cp.empty_like(state.p)
    assert _physics_exner(state.p, out=out) is out
    np.testing.assert_array_equal(cp.asnumpy(out).ravel().view(np.uint32),
                                  expected[0].view(np.uint32))


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="requires a queued CUDA card")
def test_default_physics_exner_keeps_the_prior_array_expression():
    import cupy as cp
    from gpuwm.core import constants as c
    from gpuwm.core.physics import _physics_exner
    from gpuwm.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip("default process required")
    rng = np.random.default_rng(104)
    pressure = cp.asarray(rng.uniform(3000, 105000, (16, 12, 12)).astype(np.float32))
    prior = (pressure / np.float32(c.P0)) ** np.float32(c.RCP)
    actual = _physics_exner(pressure)
    np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),
                                  cp.asnumpy(prior).view(np.uint32))
    out = cp.empty_like(pressure)
    assert _physics_exner(pressure, out=out) is out
    assert cp.asnumpy(out).tobytes() == cp.asnumpy(prior).tobytes()
