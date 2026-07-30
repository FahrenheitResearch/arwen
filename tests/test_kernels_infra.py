import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu


@requires_gpu
def test_saxpy_kernel():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    n = 1 << 20
    x = cp.random.rand(n, dtype=cp.float32)
    y = cp.random.rand(n, dtype=cp.float32)
    out = cp.empty_like(x)
    k = get_kernel("saxpy", "saxpy")
    k(((n + 255) // 256,), (256,), (np.float32(2.0), x, y, out, np.int32(n)))
    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(2.0 * x + y), rtol=1e-6)


@requires_gpu
def test_constants_injected():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    out = cp.zeros(1, dtype=cp.float32)
    k = get_kernel("saxpy", "emit_g")
    k((1,), (1,), (out,))
    assert abs(float(out[0]) - 9.81) < 1e-5


@requires_gpu
def test_module_cached():
    from gpuwm.core.kernels import load_module
    assert load_module("saxpy") is load_module("saxpy")
