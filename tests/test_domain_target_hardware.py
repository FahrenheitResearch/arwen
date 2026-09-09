"""Remote authoring uses one measured target through fit and emitted check."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gpuwm import cli, domain_wizard as dw
from gpuwm.core import preflight as pf, streaming
from tilestream.autoplan import Machine


def hardware():
    # Different NVML display values prevent accidentally using the wrong rail.
    return {"devices": [{"name": "Display-only GPU", "memory_total_bytes": 40 * dw.GIB}],
        "sizing": {"schema": "arwen.target-sizing.v1", "measured_unix_ms": 1788928327918,
            "total_bytes": 32 * dw.GIB, "free_bytes": 14 * dw.GIB,
            "profile": {"name": "Selected remote GPU", "multiprocessor_count": 170,
                "max_threads_per_multiprocessor": 1536, "default_stack_limit_bytes": 1024,
                "bare_context_bytes": 512 * 1024 ** 2}},
        "host_memory": {"schema": "arwen.target-host-memory.v1", "measured_unix_ms": 1788928327993,
            "total_bytes": 64 * dw.GIB}}


def no_local_probes(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("selected-target sizing consulted local hardware")
    monkeypatch.setattr(dw, "device_memory_probe_subprocess", refused)
    monkeypatch.setattr(pf, "device_memory_probe_subprocess", refused)
    monkeypatch.setattr(pf, "device_physical_total_bytes", refused)
    monkeypatch.setattr(pf, "live_device_local_memory_profile", refused)
    monkeypatch.setattr(pf, "declares_the_local_card", refused)
    monkeypatch.setattr(pf, "host_available_bytes", refused)
    monkeypatch.setattr(streaming, "_host_total_bytes", refused)
    monkeypatch.setattr(streaming, "planner_machine", refused)
    monkeypatch.setattr(Machine, "detect", refused)


@pytest.mark.parametrize("polygon", [False, True])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_domain_target_hardware_reaches_every_fit_and_emitted_check(
        tmp_path, monkeypatch, capsys, polygon, custom, declared):
    no_local_probes(monkeypatch)
    # Bound the point-search fixture; every candidate still uses the real
    # geometry loader, suite estimator and final composed CLI check.
    monkeypatch.setattr(dw, "_MAX_SCALE", 1.5)
    snapshot = tmp_path / "target.json"
    snapshot.write_text(json.dumps(hardware()), encoding="utf-8")
    expected_free = dw.resolve_sizing_budget(None, 16).free_bytes if declared else 14 * dw.GIB
    expected_capacity = 16 if declared else 32
    seen_fit, seen_check = [], []
    native_fit_phases, native_check_phases = dw._sizing_phases, pf.estimate_phases

    def verify_machine(kwargs):
        machine = kwargs.get("machine")
        assert isinstance(machine, Machine)
        assert machine.host_bytes == 64 * dw.GIB
        assert machine.vram_bytes == expected_free
        assert kwargs["vram_gib"] == expected_capacity
        if declared:
            assert machine.device_profile is None
        else:
            assert machine.device_profile.name == "Selected remote GPU"
            assert machine.device_profile.multiprocessor_count == 170
            assert kwargs["profile"] == machine.device_profile
        return machine

    def fit_phases(exp, **kwargs):
        seen_fit.append(verify_machine(kwargs))
        return native_fit_phases(exp, **kwargs)

    def check_phases(exp, **kwargs):
        seen_check.append(verify_machine(kwargs))
        return native_check_phases(exp, **kwargs)

    monkeypatch.setattr(dw, "_sizing_phases", fit_phases)
    monkeypatch.setattr(pf, "estimate_phases", check_phases)
    out = tmp_path / "forecast.toml"
    argv = ["domain", "--source", "gfs", "--cycle", "2026-09-09T00", "--hours", "6",
        "--tiles", "auto", "--out", str(out)]
    argv += (["--vram-gib", "16", "--target-host-memory-json", str(snapshot)] if declared else
             ["--hardware-json", str(snapshot)])
    argv += (["--root-dx", "12", "--chain", "4"] if custom else ["--ladder", "12-3"])
    if polygon:
        footprint = tmp_path / "footprint.json"
        footprint.write_text(json.dumps({"type": "Polygon", "coordinates": [[
            [-100.2, 39.8], [-99.8, 39.8], [-99.8, 40.2], [-100.2, 40.2], [-100.2, 39.8]]]}))
        argv += ["--polygon", str(footprint)]
    else:
        argv += ["--point=40,-100"]
    assert cli.main(argv) == 0
    assert len(seen_fit) >= 2 and len(seen_check) == 1
    assert all(machine is seen_fit[0] for machine in seen_fit + seen_check)
    output = capsys.readouterr().out
    assert "gpuwm check: PASS" in output
    assert "Host RAM: n/a" in output or "Host RAM: unknown" in output
    exp = dw.experiment_from_text(out.read_text(encoding="utf-8"), source=str(out))
    assert (exp.root.run.mp_physics, exp.root.run.bl_pbl_physics, exp.root.run.cu_physics) == (10, 1, 1)
    assert out.with_suffix(".namelist.wps").is_file()


@pytest.mark.parametrize("case", ["missing_host", "bad_free", "missing_profile", "mixed_capacity", "mixed_host", "host_without_card"])
def test_domain_target_hardware_refuses_invalid_inputs_before_local_probe_or_output(
        tmp_path, monkeypatch, case):
    no_local_probes(monkeypatch)
    value = hardware()
    if case == "missing_host": value.pop("host_memory")
    if case == "bad_free": value["sizing"]["free_bytes"] = 33 * dw.GIB
    if case == "missing_profile": value["sizing"]["profile"].pop("multiprocessor_count")
    snapshot = tmp_path / "bad.json"
    snapshot.write_text(json.dumps(value))
    out = tmp_path / "unpublished.toml"
    flags = ["--hardware-json", str(snapshot)]
    if case == "mixed_capacity": flags += ["--vram-gib", "16"]
    if case == "mixed_host": flags += ["--target-host-memory-json", str(snapshot)]
    if case == "host_without_card": flags = ["--target-host-memory-json", str(snapshot)]
    assert cli.main(["domain", "--point=40,-100", "--source", "gfs", "--cycle", "2026-09-09T00",
        "--tiles", "auto", "--out", str(out), *flags]) == 2
    assert not out.exists() and not out.with_suffix(".namelist.wps").exists()


def test_domain_target_hardware_keeps_gpu_only_snapshot_valid_without_streaming(tmp_path, monkeypatch):
    no_local_probes(monkeypatch)
    value = hardware(); value.pop("host_memory")
    snapshot = tmp_path / "gpu-only.json"; snapshot.write_text(json.dumps(value))
    sizing, machine, selected = dw._domain_target_hardware(SimpleNamespace(
        hardware_json=snapshot, card=None, vram_gib=None, tiles="off"))
    assert selected and machine is None and sizing.measured
    assert sizing.vram_gib == 32 and sizing.free_bytes == 14 * dw.GIB


def test_domain_gpu_only_target_check_does_not_read_desktop_ram(tmp_path, monkeypatch):
    no_local_probes(monkeypatch)
    monkeypatch.setattr(dw, "_MAX_SCALE", 1.0)
    value = hardware(); value.pop("host_memory")
    snapshot = tmp_path / "gpu-only.json"; snapshot.write_text(json.dumps(value))
    out = tmp_path / "resident.toml"
    assert cli.main(["domain", "--point=40,-100", "--source", "gfs", "--cycle", "2026-09-09T00",
        "--ladder", "12", "--tiles", "off", "--hardware-json", str(snapshot), "--out", str(out)]) == 0
    assert out.is_file()


def test_internal_selected_machine_requires_matching_binding_and_budget(tmp_path):
    parser = cli.build_parser()
    for bound, free in [(False, 14), (True, 13)]:
        args = parser.parse_args(["check", str(tmp_path / "not-read.toml"), "--free-gib", "14", "--vram-gib", "32"])
        args._target_hardware_supplied = bound
        args._shared_target_machine = Machine(vram_bytes=free * dw.GIB, host_bytes=64 * dw.GIB)
        with pytest.raises(ValueError, match="internal target machine"):
            pf.check_main(args)
