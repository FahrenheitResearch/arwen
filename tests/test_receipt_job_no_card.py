"""The HRRR hydrometeor receipt job on a host with no usable card.

Breakage: with cupy installed and no NVIDIA driver at all (a CPU-only box,
the WOOF bake's data stage), ``cupy.cuda.runtime.getDevice()`` raises
cudaErrorInsufficientDriver (35), not cudaErrorNoDevice (100), and every
HRRR prepare died in ``_start_receipt_job`` after the CPU preparation ran.
"""
import sys
import types

import pytest

from gpuwm.ingest import real


class _RuntimeError(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status = status


def _fake_cupy(status):
    def get_device():
        raise _RuntimeError(status)

    runtime = types.SimpleNamespace(CUDARuntimeError=_RuntimeError, getDevice=get_device)
    return types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=runtime))


@pytest.mark.parametrize("status", [100, 35])
def test_no_card_status_runs_the_receipt_on_the_host(monkeypatch, status):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(status))
    job = real._start_receipt_job(lambda a, b=0: a + b, 2, b=3)
    assert job.result(timeout=30) == 5


def test_a_device_fault_still_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(2))  # cudaErrorMemoryAllocation
    with pytest.raises(_RuntimeError):
        real._start_receipt_job(lambda: None)
