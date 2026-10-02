"""Real compiler witnesses for strict arithmetic and generated array kernels."""
import numpy as np
import pytest
import cupy as cp

from gpuwm import wrf_exact

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not wrf_exact.ENABLED,
                                 reason="requires the opt-in strict compiler")]


def test_raw_and_generated_kernels_preserve_subnormals_and_disable_contraction():
    # (1+2^-23)*(1-2^-23)-1 is zero with two binary32 roundings, but
    # -2^-46 when contracted. The first input is the smallest subnormal.
    inputs = np.array([1, 0x3f800001, 0x3f7ffffe, 0xbf800000], np.uint32)
    x = cp.asarray(inputs.view(np.float32))
    out = cp.empty(2, np.float32)
    source = r'''
extern "C" __global__ void strict_probe(const float* x, float* y) {
    y[0] = x[0] * 2.0f;
    y[1] = x[1] * x[2] + x[3];
}
'''
    kernel = cp.RawKernel(source, "strict_probe",
        options=("--use_fast_math", "--fmad=true", "--ftz=true"))
    kernel((1,), (1,), (x, out))
    assert np.array_equal(cp.asnumpy(out).view(np.uint32), [2, 0])
    generated = cp.ElementwiseKernel(
        "float32 a, float32 b, float32 c", "float32 y", "y = a*b+c;",
        "strict_generated_probe", options=("--use_fast_math",))
    assert int(cp.asnumpy(generated(x[1:2], x[2:3], x[3:4])).view(np.uint32)[0]) == 0
    records = wrf_exact.compile_receipt()["records"]
    assert any(r["kind"] == "nvrtc_compile" for r in records)
    for row in records:
        options = row["options"]
        assert all(o in options for o in wrf_exact.STRICT_OPTIONS)
        assert "-ftz=true" not in options and "--ftz=true" not in options
        assert "--use_fast_math" not in options and "--fmad=true" not in options
