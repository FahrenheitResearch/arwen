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


class _DriverError(Exception):
    pass


def _visible_card_cupy(current_context):
    """cupy on a box with a visible card 0; ``current_context`` is what
    cuCtxGetCurrent returns on the caller's thread (0 = none)."""
    used = []

    class Device:
        def __init__(self, index):
            self.index = index

        def use(self):
            used.append(self.index)

    runtime = types.SimpleNamespace(CUDARuntimeError=_RuntimeError, getDevice=lambda: 0)
    driver = types.SimpleNamespace(CUDADriverError=_DriverError,
                                   ctxGetCurrent=lambda: current_context)
    cuda = types.SimpleNamespace(runtime=runtime, driver=driver, Device=Device)
    return types.SimpleNamespace(cuda=cuda), used


def test_a_cpu_caller_with_no_context_never_binds_the_card(monkeypatch):
    # U-E7: cudaGetDevice answers 0 whenever a card is visible, and binding it
    # on the job thread (Device.use = cudaSetDevice) opened a ~386 MiB context
    # on the card in every CPU-backend HRRR start.  No current context on the
    # caller's thread means no card is handed to the job.
    fake, used = _visible_card_cupy(current_context=0)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    job = real._start_receipt_job(lambda a, b=0: a + b, 2, b=3)
    assert job.result(timeout=30) == 5
    assert used == []


def test_a_cuda_caller_hands_its_card_to_the_job(monkeypatch):
    fake, used = _visible_card_cupy(current_context=0x5A5A)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    job = real._start_receipt_job(lambda: 7)
    assert job.result(timeout=30) == 7
    assert used == [0]
