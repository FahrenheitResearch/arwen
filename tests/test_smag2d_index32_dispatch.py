"""The narrow module cannot address large grids or non-scalar stresses.

CPU-only: both compilers are monkeypatched, so no device is opened.  The
module still needs CuPy importable, because gpuwm.core.dycore imports it at
module scope.  That is not handled here: tests/conftest.py
(pytest_pycollect_makemodule) reports a module whose import closure reaches
an unguarded ``import cupy`` as skipped, with the reason, on an install with
no CuPy, and tells the deselection guard why.  A module-level skip of its own
would retire every test here without a declaration, which
tests/test_must_run_gates.py forbids, and would duplicate that mechanism.
"""

import pytest

from gpuwm.core import dycore


@pytest.mark.parametrize("name", ["wrf_smag_flux_s", "wrf_smag_hd_s"])
@pytest.mark.parametrize("shape,wide", [
    ((50, 600, 600), False), ((50, 400, 400), False),
    ((50, 1057, 1797), False), ((50, 10000, 10000), True),
    ((1, 1, 1073741823), True),
])
def test_scalar_pair_selects_only_capacity_safe_module(monkeypatch, name, shape, wide):
    calls = []
    result = object()

    def ordinary(module, symbol):
        calls.append((module, symbol, ()))
        return result

    def narrow(module, symbol, defines):
        calls.append((module, symbol, defines))
        return result

    monkeypatch.setattr(dycore, "get_kernel", ordinary)
    monkeypatch.setattr(dycore, "get_kernel_int_defines", narrow)
    assert dycore._wrf_smag_scalar_kernel(name, *shape) is result
    expected = () if wide else (("GPUWM_SMAG_INDEX32", 1),)
    assert calls == [("smag2d", name, expected)]


@pytest.mark.parametrize("name", [
    "wrf_smag_hd_u", "wrf_smag_hd_v", "wrf_smag_w_stress", "wrf_smag2d_km",
])
def test_non_scalar_symbols_cannot_select_narrow_module(monkeypatch, name):
    def unexpected(*args):
        pytest.fail("non-scalar request reached a kernel compiler")

    monkeypatch.setattr(dycore, "get_kernel", unexpected)
    monkeypatch.setattr(dycore, "get_kernel_int_defines", unexpected)
    with pytest.raises(ValueError, match="not a Smagorinsky scalar kernel"):
        dycore._wrf_smag_scalar_kernel(name, 50, 600, 600)
