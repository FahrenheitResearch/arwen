"""Measured runtime and check share the device-wide additional free ceiling."""
from types import SimpleNamespace
import sys

import pytest

from gpuwm.core import preflight as pf, streaming
from tilestream import autoplan

GIB = 1024 ** 3


@pytest.mark.parametrize("total,used,expected,capped", [
    (10, 7, 3, True), (10, 1, 9, False), (10, 0, 9, False),
    (10, 12, 0, True), (None, 7, 9, False), (10, None, 9, False)])
def test_device_wide_cap_never_widens_cuda_free(monkeypatch, total, used, expected, capped):
    monkeypatch.setattr(pf, "device_physical_total_bytes",
                        lambda: None if total is None else total * GIB)
    monkeypatch.setattr(pf, "device_wide_used_bytes",
                        lambda: None if used is None else used * GIB)
    assert pf.cap_free_to_device_wide(9 * GIB) == (expected * GIB, capped)


def test_unavailable_additional_ceiling_preserves_successful_cuda_observation(monkeypatch):
    monkeypatch.setattr(pf, "device_physical_total_bytes", lambda: 10 * GIB)
    def unavailable():
        raise RuntimeError("nvidia-smi unavailable")
    monkeypatch.setattr(pf, "device_wide_used_bytes", unavailable)
    assert pf.cap_free_to_device_wide(9 * GIB) == (9 * GIB, False)


def _fake_cuda(monkeypatch):
    events = []
    class Device:
        def __init__(self, number):
            self.number = number
            self.pci_bus_id = "00000000:04:00.0"
        def __enter__(self):
            events.append(("enter", self.number))
            return self
        def __exit__(self, *_):
            pass
    cuda = SimpleNamespace(Device=Device, runtime=SimpleNamespace(
        memGetInfo=lambda: (9 * GIB, 10 * GIB),
        getDeviceProperties=lambda number: {"name": b"test device"}))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=cuda))
    return events


def test_detect_caps_selected_device_and_explicit_total_mode_does_not_probe_nvml(monkeypatch):
    events = _fake_cuda(monkeypatch)
    def observed(free, **kwargs):
        events.append(("cap", free, kwargs))
        return 3 * GIB, True
    monkeypatch.setattr(pf, "cap_free_to_device_wide", observed)
    measured = autoplan.Machine.detect(host_bytes=64 * GIB, device=2)
    assert measured.vram_bytes == 3 * GIB
    assert measured.host_bytes == 64 * GIB
    assert events == [("enter", 2), ("cap", 9 * GIB,
                                      {"device_id": "00000000:04:00.0"})]
    events.clear()
    total = autoplan.Machine.detect(host_bytes=64 * GIB, device=2, use_free_vram=False)
    assert total.vram_bytes == 10 * GIB
    assert events == [("enter", 2)]


def test_nvml_helpers_apply_the_actual_cuda_device_selector(monkeypatch):
    from gpuwm import supervisor
    observed = []
    def query(args):
        observed.append(args)
        return "10240" if "--query-gpu=memory.total" in args else "7168"
    monkeypatch.setattr(supervisor, "_run_nvidia_smi", query)
    assert pf.cap_free_to_device_wide(9 * GIB, device_id="00000000:04:00.0") == (3 * GIB, True)
    assert len(observed) == 2
    assert all("--id=00000000:04:00.0" in args for args in observed)


def test_supplied_machine_and_declared_budget_do_not_consult_observers(monkeypatch):
    from tilestream.test_ledger_gate import _exp
    def forbidden(*args, **kwargs):
        pytest.fail("supplied machine caused another observation")
    monkeypatch.setattr(autoplan.Machine, "detect", forbidden)
    monkeypatch.setattr(pf, "cap_free_to_device_wide", forbidden)
    exp = _exp(704, "auto", vram_budget_bytes=24 * GIB)
    answer = streaming.decide(exp.root.run, exp.tiles,
        machine=autoplan.Machine(64 * GIB, 256 * GIB))
    assert answer.detail["resident_admission"]["budget_bytes"] == 24 * GIB


def test_public_check_caps_the_current_cuda_card_after_device_reordering(
        tmp_path, monkeypatch, capsys):
    import json
    from gpuwm import doctor, supervisor
    from test_check_gpu_readiness import _args
    from test_check_host_memory import _fetch_only_config

    current_pci = "00000000:04:00.0"
    # CUDA ordinal zero is the second physical card. The first NVML card
    # has9GiB free, the selected one3GiB; using the first would over-admit.
    cuda = SimpleNamespace(Device=lambda: SimpleNamespace(pci_bus_id=current_pci),
        runtime=SimpleNamespace(memGetInfo=lambda: (9 * GIB, 12 * GIB)))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=cuda))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: doctor.Check(
        "CUDA kernel headers", "verified", "controlled test"))
    monkeypatch.setattr(pf, "live_device_local_memory_profile",
                        lambda: pf.card_local_memory_profile(12.))
    monkeypatch.setattr(pf, "declares_the_local_card", lambda *_: False)
    observed = []
    def query(arguments):
        observed.append(arguments)
        selected = f"--id={current_pci}" in arguments
        if "--query-gpu=memory.total" in arguments:
            return "12288" if selected else "10240"
        return "9216" if selected else "1024"
    monkeypatch.setattr(supervisor, "_run_nvidia_smi", query)
    args = _args(tmp_path, "--json")
    args.config = _fetch_only_config(tmp_path)
    pf.check_main(args)
    report = json.loads(capsys.readouterr().out)
    assert report["measured_free_bytes"] == 3 * GIB
    assert report["free_bytes_source"].startswith("measured machine-wide")
    assert len(observed) == 2
    assert all(f"--id={current_pci}" in call for call in observed)
    assert any("--query-gpu=memory.total" in call for call in observed)
    assert any("--query-gpu=memory.used" in call for call in observed)


def test_public_declared_check_never_resolves_a_cuda_device(tmp_path, monkeypatch, capsys):
    import json
    from gpuwm import doctor, supervisor
    from test_check_gpu_readiness import _args
    from test_check_host_memory import _fetch_only_config

    def forbidden(*args, **kwargs):
        pytest.fail("declared target caused a CUDA or NVML memory observation")
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(Device=forbidden, runtime=SimpleNamespace(memGetInfo=forbidden))))
    monkeypatch.setattr(doctor, "_cuda_headers_check", forbidden)
    monkeypatch.setattr(pf, "declares_the_local_card", lambda *_: False)
    monkeypatch.setattr(supervisor, "_run_nvidia_smi", forbidden)
    args = _args(tmp_path, "--free-gib", "24", "--vram-gib", "24", "--json")
    args.config = _fetch_only_config(tmp_path)
    pf.check_main(args)
    report = json.loads(capsys.readouterr().out)
    assert report["measured_free_bytes"] == 24 * GIB
    assert report["free_bytes_source"] == "declared (--free-gib)"
    assert report["gpu_readiness"]["status"] == "not_checked"
