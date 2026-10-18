"""The CLM lake's binary64 intrinsics are WOOF's correctly rounded ones.

Breakage (clm-lake column oracle, 2026-10-07): the lake called CUDA's
binary64 exp, log, sin, atan and pow, which are accurate to 1-2 ulp but not
correctly rounded.  Against native WRF v4.6.1 that moved the lake's hfx and
grdflx outputs in 805 words of a 112-column simulated day (max 1638 ulp on a
near-zero flux).  The lake now evaluates them with gpuwm/core/kernels/
f64cr.cuh, identically on host and device, and native WRF linked to
correctly rounded functions matches the CUDA lake in every word.

Fixtures: tests/data/oracles/lake, written by
tools/lake_wrf461_oracle/fixtures.py on the verification box.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "gpuwm/core/kernels"
FIXTURES = ROOT / "tests/data/oracles/lake"
FUNCS = ("exp", "log", "log10", "sin", "atan", "pow")


def test_lake_transcription_calls_no_platform_binary64_library():
    # A bare exp( or log( in the transcription would be CUDA's binary64
    # function on the device and the host's in the native build again.
    body = (KERNELS / "lake_wrf.cuh").read_text(encoding="utf-8")
    bare = re.findall(r"(?<![\w.])(exp|log|log10|sin|cos|tan|atan|atan2|pow|sinh|cosh|tanh|expm1|log1p)\(", body)
    assert bare == []
    for f in FUNCS[:-1]:
        assert f"lake_{f}(" in body
    support = (KERNELS / "lake_support.cuh").read_text(encoding="utf-8")
    for f in FUNCS[:-1]:
        assert f"LAKE_HD double lake_{f}(double x) {{return wcr_{f}(x);}}" in support
        assert f"LAKE_HD float lake_{f}(float) = delete;" in support
    assert "LAKE_HD double lake_pow_real(double a,double b) {return wcr_pow(a,b);}" in support
    assert "return lake_pow_real(T(a),T(b));" in support


def test_lake_unit_composes_the_correctly_rounded_header():
    from gpuwm.core.kernels import _EXTRA_HEADERS, module_source
    headers = _EXTRA_HEADERS["lake"]
    assert headers.index("f64cr.cuh") < headers.index("lake_support.cuh")
    assert "#define WOOF_F64CR" in module_source("lake")


def test_f64cr_constants_are_the_generator_output():
    pytest.importorskip("mpmath")
    path = ROOT / "tools/lake_wrf461_oracle/f64cr_constants.py"
    spec = importlib.util.spec_from_file_location("f64cr_constants_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    header = (KERNELS / "f64cr.cuh").read_text(encoding="utf-8")
    start = header.index("// CONSTANTS BEGIN")
    end = header.index("// CONSTANTS END") + len("// CONSTANTS END\n")
    assert header[start:end] == module.block()


def _device_eval(fname, x, y):
    import cupy as cp
    from gpuwm.core.kernels import module_options, module_source
    probe = r'''
extern "C" __global__ void f64cr_probe(int f, int n, const double* x, const double* y, double* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double a = x[i], b = y[i], r;
    switch (f) {
    case 0: r = lake_exp(a); break;
    case 1: r = lake_log(a); break;
    case 2: r = lake_log10(a); break;
    case 3: r = lake_sin(a); break;
    case 4: r = lake_atan(a); break;
    default: r = lake_pow(a, b);
    }
    out[i] = r;
}
'''
    module = cp.RawModule(code=module_source("lake") + probe, options=module_options("lake"))
    n = len(x)
    out = cp.empty(n, cp.float64)
    module.get_function("f64cr_probe")(((n + 127) // 128,), (128,),
        (np.int32(FUNCS.index(fname)), np.int32(n), cp.asarray(x.view(np.float64)),
         cp.asarray(y.view(np.float64)), out))
    return cp.asnumpy(out)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("fname", FUNCS)
def test_device_intrinsics_are_correctly_rounded(fname):
    with np.load(FIXTURES / "f64cr_cases.npz") as cases:
        x, y, expected = cases[f"{fname}_x"], cases[f"{fname}_y"], cases[f"{fname}_expected"]
        sx, sy, sexp = cases[f"{fname}_special_x"], cases[f"{fname}_special_y"], cases[f"{fname}_special_expected"]
        misrounded = int(cases[f"{fname}_host_misrounded"])
    got = _device_eval(fname, x, y).view(np.uint64)
    bad = np.flatnonzero(got != expected)
    assert bad.size == 0, [(x[j:j+1].view(np.float64)[0].hex(), got[j:j+1].view(np.float64)[0].hex(),
                            expected[j:j+1].view(np.float64)[0].hex()) for j in bad[:5]]
    # The fixture holds the lake arguments where the host library is not
    # correctly rounded, so this test separates correct rounding from
    # matching one library.
    if fname in ("exp", "log", "atan", "pow"):
        assert misrounded > 0
    sgot = _device_eval(fname, sx, sy)
    sref = sexp.view(np.float64)
    same = (sgot.view(np.uint64) == sexp) | (np.isnan(sgot) & np.isnan(sref))
    assert same.all(), [(sx[j:j+1].view(np.float64)[0], sy[j:j+1].view(np.float64)[0], sgot[j], sref[j])
                        for j in np.flatnonzero(~same)[:5]]


@pytest.mark.gpu
@requires_gpu
def test_lake_columns_match_wrf_with_correctly_rounded_intrinsics():
    # Breakage: with CUDA's binary64 library these columns' hfx left native
    # WRF from step 5 (gale#0) and grdflx from step 23
    # (tropical_deep_equator#0).
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.lake import lake_arena, launch_lake_columns
    from gpuwm.core.lake_schema import LAKE_OUTPUT_WORDS, LAKE_STATE_WORDS, LAKE_STATIC_WORDS
    with np.load(FIXTURES / "lake_cr_regimes.npz") as ref:
        seed, forcing = cp.asarray(ref["seed"]), cp.asarray(ref["forcing"])
        n = ref["seed"].shape[1]
        dt = np.float32(ref["dt"])
        state = cp.zeros((LAKE_STATE_WORDS, n), cp.float32)
        static = cp.zeros((LAKE_STATIC_WORDS, n), cp.float32)
        output = cp.zeros((LAKE_OUTPUT_WORDS, n), cp.float32)
        errors = cp.zeros(n, cp.int32)
        arena = lake_arena(n)
        init, step = get_kernel("lake", "lake_init_columns"), get_kernel("lake", "lake_step_columns")
        launch_lake_columns(init, n, (n, seed, state, static, 1, 1, np.float32(50), np.float32(0.5), errors), arena)
        np.testing.assert_array_equal(cp.asnumpy(state).view(np.uint32), ref["init_state"].view(np.uint32))
        np.testing.assert_array_equal(cp.asnumpy(static).view(np.uint32), ref["init_static"].view(np.uint32))
        checkpoints = {int(s): k for k, s in enumerate(ref["state_steps"])}
        for s in range(ref["output"].shape[0]):
            launch_lake_columns(step, n, (n, forcing[s], state, static, output, dt, np.float32(0.5), errors), arena)
            assert not cp.asnumpy(errors).any()
            np.testing.assert_array_equal(cp.asnumpy(output).view(np.uint32), ref["output"][s].view(np.uint32),
                                          err_msg=f"step {s + 1} output")
            if s + 1 in checkpoints:
                np.testing.assert_array_equal(cp.asnumpy(state).view(np.uint32),
                                              ref["state"][checkpoints[s + 1]].view(np.uint32),
                                              err_msg=f"step {s + 1} state")
