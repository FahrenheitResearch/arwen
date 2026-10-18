"""Shared helpers for the NSSL sedimentation route-equivalence tests."""
import pytest


def device_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    return cp


def kernel(name):
    """A production kernel through the real loader (headers included)."""
    from gpuwm.core.kernels import get_kernel
    return get_kernel("nssl2_driver_support", name)
