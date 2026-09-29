"""Auto preparation shares the fit probe and records measured contention."""

from types import SimpleNamespace

import pytest

from gpuwm.core import device_probe, preflight
from gpuwm.ingest import preprocess_backend as backend


@pytest.fixture
def candidates(monkeypatch):
    runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                              runtimeGetVersion=lambda: 13020)
    cuda = SimpleNamespace(name="cuda", array_module=SimpleNamespace(
        __version__="14.2.0", cuda=SimpleNamespace(runtime=runtime)))
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(backend, "CudaPreprocessBackend", lambda: cuda)
    monkeypatch.setattr(backend, "ParallelCpuPreprocessBackend", lambda **_: cpu)
    monkeypatch.setattr(backend, "_gpu_runtime_installed", lambda: True)
    monkeypatch.setattr(backend, "_ANNOUNCED_AUTO_REASONS", set())
    monkeypatch.delenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", raising=False)
    return cuda, cpu


@pytest.mark.parametrize("free,utilization,reason_fragment", [
    (30, 95, "utilization 95%"),
    (30, 50, "utilization 50%"),
])
def test_auto_uses_cpu_for_measured_busy_card(
        monkeypatch, capsys, candidates, free, utilization, reason_fragment):
    measured = {"free_bytes": free * 2**30, "total_bytes": 32 * 2**30,
                "utilization_gpu_percent": utilization}
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", lambda **_: measured)
    chosen = backend.resolve_preprocess_backend("auto")
    assert chosen is candidates[1]
    selection = chosen.selection
    assert reason_fragment in selection["reason"]
    assert selection["requested"] == "auto"
    load = selection["device_load"]
    assert load["free_bytes"] == measured["free_bytes"]
    assert load["total_bytes"] == measured["total_bytes"]
    assert load["utilization_gpu_percent"] == utilization
    assert load["busy_utilization_threshold_percent"] == 50
    # The free-memory FRACTION is retired (A65): whether a preparation fits
    # is its price against the free memory, weighed by admit_preparation.
    assert "minimum_free_fraction" not in load
    message = capsys.readouterr().err
    assert selection["reason"] in message
    assert message.count("\n") == 1


@pytest.mark.parametrize("probe", [
    {"free_bytes": 30 * 2**30, "total_bytes": 32 * 2**30,
     "utilization_gpu_percent": 0},
    {"free_bytes": 9 * 2**30, "total_bytes": 32 * 2**30,
     "utilization_gpu_percent": 49},
    {"free_bytes": 30 * 2**30, "total_bytes": 32 * 2**30,
     "utilization_gpu_percent": None},
    # A mostly-full card with no price is no longer a CPU answer by itself:
    # the 25% floor passed 7 GiB free for a 32 GiB preparation, and every
    # preparation door now brings its price (tests/test_preparation_price.py).
    {"free_bytes": 2 * 2**30, "total_bytes": 32 * 2**30,
     "utilization_gpu_percent": 0},
    None,
])
def test_auto_keeps_idle_or_unmeasured_card_choice(
        monkeypatch, capsys, candidates, probe):
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", lambda **_: probe)
    chosen = backend.resolve_preprocess_backend("auto")
    assert chosen is candidates[0]
    assert "certified" in chosen.selection["reason"]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("error,nvml,fragments", [
    # Too full for a context: NVML still says how full and how busy.
    ("CUDARuntimeError: cudaErrorMemoryAllocation: out of memory",
     {"used_bytes": 31 * 2**30, "total_bytes": 32 * 2**30,
      "utilization_gpu_percent": 97},
     ("utilization 97%", "cudaErrorMemoryAllocation")),
    # Exclusive-process mode held by another program: NVML can read idle,
    # and the preparation's own context would fail the same way.
    ("CUDARuntimeError: cudaErrorDevicesUnavailable: CUDA-capable device(s) "
     "is/are busy or unavailable",
     {"used_bytes": 2**30, "total_bytes": 32 * 2**30,
      "utilization_gpu_percent": 0},
     ("cudaErrorDevicesUnavailable",)),
    ("CUDARuntimeError: cudaErrorDevicesUnavailable", None,
     ("cudaErrorDevicesUnavailable",)),
])
def test_auto_uses_cpu_when_cuda_cannot_open_the_card(
        monkeypatch, capsys, candidates, error, nvml, fragments):
    asked = {}

    def probe(**kwargs):
        asked.update(kwargs)
        return {"cuda_error": error, "nvml": nvml}

    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", probe)
    chosen = backend.resolve_preprocess_backend("auto")
    assert chosen is candidates[1], "auto kept CUDA on a card CUDA cannot open"
    assert asked.get("report_failure") is True
    selection = chosen.selection
    for fragment in fragments:
        assert fragment in selection["reason"]
    load = selection["device_load"]
    assert load["cuda_error"] == error
    if nvml is None:
        assert load["free_bytes"] is None and load["total_bytes"] is None
        assert load["utilization_gpu_percent"] is None
    else:
        assert load["free_bytes"] == nvml["total_bytes"] - nvml["used_bytes"]
        assert load["total_bytes"] == nvml["total_bytes"]
        assert load["utilization_gpu_percent"] == nvml["utilization_gpu_percent"]
    message = capsys.readouterr().err
    assert selection["reason"] in message
    assert message.count("\n") == 1


def test_explicit_backend_does_not_apply_auto_load_policy(monkeypatch, candidates):
    def unexpected(**_):
        pytest.fail("an explicit choice queried automatic load policy")
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", unexpected)
    assert backend.resolve_preprocess_backend("cuda") is candidates[0]
    assert backend.resolve_preprocess_backend("cpu") is candidates[1]


def test_auto_rechecks_load_for_each_preparation(monkeypatch, candidates):
    values = iter([0, 99])
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess", lambda **_: {
        "free_bytes": 30 * 2**30, "total_bytes": 32 * 2**30,
        "utilization_gpu_percent": next(values)})
    assert backend.resolve_preprocess_backend("auto") is candidates[0]
    assert backend.resolve_preprocess_backend("auto") is candidates[1]


def test_the_preflight_probe_is_the_one_auto_reads():
    """One probe, two import paths: auto reads the leaf RW-WPS stages, and
    the forecast preflight re-exports that same function to its callers, so
    the fit and the backend choice read one card the same way."""

    for name in ("device_memory_probe_subprocess", "device_memory_probe_reason",
                 "_DEVICE_MEMORY_PROBE_SOURCE", "PROBE_EXIT_NO_RUNTIME",
                 "PROBE_EXIT_CARD_UNREAD", "DEVICE_MEMORY_PROBE_TIMEOUT_SECONDS",
                 "PROBE_REASON_NO_RUNTIME", "_probe_error", "_probe_last_line"):
        assert getattr(preflight, name) is getattr(device_probe, name), name
