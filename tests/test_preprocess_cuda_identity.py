"""Preparation receipts identify the CUDA runtime that produced each case."""

from types import SimpleNamespace

import pytest

from gpuwm.ingest import horiz
from gpuwm.ingest.preprocess_backend import CudaPreprocessBackend


@pytest.mark.parametrize(("runtime_version", "cupy_version"), [
    (12_090, "14.1.1"),
    (13_020, "14.2.0"),
])
def test_cuda_receipt_measures_runtime_and_cupy_identity(
        monkeypatch, runtime_version, cupy_version):
    calls = []

    def runtime_get_version():
        calls.append("runtimeGetVersion")
        return runtime_version

    cp = SimpleNamespace(
        __version__=cupy_version,
        cuda=SimpleNamespace(runtime=SimpleNamespace(
            runtimeGetVersion=runtime_get_version)))
    monkeypatch.setattr(horiz, "_cupy", lambda: cp)

    receipt = CudaPreprocessBackend().receipt()

    assert receipt["cuda_runtime_version"] == runtime_version
    assert receipt["cupy_version"] == cupy_version
    assert calls == ["runtimeGetVersion"]
