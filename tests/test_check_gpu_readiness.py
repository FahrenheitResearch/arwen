"""A measured check proves compilation and execution before memory admission."""
import argparse
import json

import pytest

from gpuwm import doctor
from gpuwm.core import preflight as pf


def _args(tmp_path, *flags):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pf.register_cli(sub)
    return parser.parse_args(["check", str(tmp_path / "config.toml"), *flags])


@pytest.mark.parametrize("alloc", [False, True])
@pytest.mark.parametrize("payload,expected", [
    ({"self_contained": "ok", "toolkit_headers": "Failed to find CUDA headers"}, 1),
    ({"self_contained": "NVRTC_ERROR", "toolkit_headers": "NVRTC_ERROR"}, 1),
    ({"slow": "no answer within 180 s"}, 2),
    ({"devices": 0}, 2),
])
def test_unusable_or_unverified_gpu_refuses_before_allocation(
        tmp_path, monkeypatch, capsys, alloc, payload, expected):
    from test_check_host_memory import _fetch_only_config
    from tilestream.autoplan import Machine

    config = _fetch_only_config(tmp_path)
    monkeypatch.setattr(pf, "_warn_unstaged_physics_tables", lambda *_: None)
    monkeypatch.setattr(doctor, "no_local_gpu", lambda: False)
    monkeypatch.setattr(doctor, "_nvrtc_header_probe", lambda: payload)
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)
    def forbidden(*_args, **_kwargs):
        pytest.fail("GPU observation, admission, or allocation ran after failed readiness")
    monkeypatch.setattr(pf, "live_device_local_memory_profile", forbidden)
    monkeypatch.setattr(Machine, "detect", forbidden)
    monkeypatch.setattr(pf, "run_alloc_preflight", forbidden)
    flags = ["--json"] + (["--alloc"] if alloc else [])
    args = _args(tmp_path, *flags)
    args.config = config
    assert pf.check_main(args) == expected
    report = json.loads(capsys.readouterr().out)
    assert report["gpu_readiness"]["status"] != "verified"
    required = report["required_memory"]
    assert required["status"] == "estimated"
    assert required["alloc_estimate_bytes"] > required["resident_bytes"] > 0
    assert required["domains"]["d01"]["by_category"]["state"] > 0
    assert required["budget_bytes"] is None and required["measured_free_bytes"] is None
    assert required["memory_verdict"] == "unavailable"
    assert report["cpu_planning"]["command"] == [
        "gpuwm", "check", str(config), "--free-gib", "FREE_GIB", "--vram-gib", "CAPACITY_GIB"]
    if "toolkit_headers" in payload:
        assert report["gpu_readiness"]["action"]


def test_failure_is_concise_and_uses_existing_remedy(tmp_path, monkeypatch, capsys):
    from test_check_host_memory import _fetch_only_config
    config = _fetch_only_config(tmp_path)
    monkeypatch.setattr(pf, "_warn_unstaged_physics_tables", lambda *_: None)
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: doctor.Check(
        "CUDA kernel headers", "missing", "long compiler detail", "long remedy",
        action="pip install 'cupy-cuda13x[ctk]'", brief="toolkit headers missing"))
    args = _args(tmp_path)
    args.config = config
    assert pf.check_main(args) == 1
    output = capsys.readouterr().out
    assert len(output.splitlines()) <= 7
    assert "toolkit headers missing" in output and "cupy-cuda13x[ctk]" in output
    assert "CPU required-memory estimate" in output
    assert "Budget and fit judgment: unavailable" in output
    assert "--free-gib FREE_GIB --vram-gib CAPACITY_GIB" in output
    assert "long compiler detail" not in output


def test_declared_estimate_does_not_probe_gpu(tmp_path, monkeypatch, capsys):
    from test_check_host_memory import _fetch_only_config
    path = _fetch_only_config(tmp_path)
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: pytest.fail("GPU probed"))
    args = _args(tmp_path, "--budget-gib", "100", "--json")
    args.config = path
    assert pf.check_main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["gpu_readiness"]["status"] == "not_checked"


def test_unverified_report_does_not_turn_declared_capacity_into_free_memory(
        tmp_path, monkeypatch, capsys):
    from test_check_host_memory import _fetch_only_config
    monkeypatch.setattr(doctor, "no_local_gpu", lambda: True)
    args = _args(tmp_path, "--vram-gib", "8", "--json")
    args.config = _fetch_only_config(tmp_path)
    assert pf.check_main(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["required_memory"]["status"] == "estimated"
    assert report["required_memory"]["budget_bytes"] is None
    assert report["required_memory"]["measured_free_bytes"] is None
    assert report["required_memory"]["memory_verdict"] == "unavailable"


def test_unavailable_cpu_estimate_keeps_readiness_and_the_portable_route(
        tmp_path, monkeypatch, capsys):
    from test_check_host_memory import _fetch_only_config
    monkeypatch.setattr(doctor, "no_local_gpu", lambda: True)
    def unavailable(*args, **kwargs):
        raise RuntimeError("required table metadata unavailable")
    monkeypatch.setattr(pf, "estimate_experiment", unavailable)
    args = _args(tmp_path, "--json")
    args.config = _fetch_only_config(tmp_path)
    assert pf.check_main(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["required_memory"]["status"] == "unavailable"
    assert report["required_memory"]["reason"] == "required table metadata unavailable"
    assert report["gpu_readiness"]["status"] != "verified"
    assert "--free-gib" in report["cpu_planning"]["command"]


def test_gpu_compilation_gap_blocks_setup_doctor_verdict(monkeypatch):
    monkeypatch.setattr(doctor, "no_local_gpu", lambda: False)
    monkeypatch.setattr(doctor, "_nvrtc_header_probe", lambda: {
        "self_contained": "ok", "toolkit_headers": "Failed to find CUDA headers"})
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)
    check = doctor._cuda_headers_check()
    assert doctor.blocking_gaps([check])


def test_windows_available_ram_uses_physical_not_pagefile(monkeypatch):
    from tilestream import autoplan
    monkeypatch.setattr(autoplan.sys, "platform", "win32")
    monkeypatch.setattr(autoplan, "_windows_memory_status", lambda: (64 << 30, 7 << 30))
    monkeypatch.setattr(pf, "_host_total_bytes_or_none", lambda: 32 << 30)
    assert pf.host_available_bytes() == 7 << 30
    assert autoplan._host_memtotal() == 64 << 30


@pytest.mark.parametrize("available,ceiling,expected", [
    (20 << 30, 8 << 30, 8 << 30), (0, 8 << 30, 0),
    (None, 8 << 30, None), (3 << 30, None, 3 << 30),
])
def test_available_ram_preserves_ceiling_zero_and_unknown(monkeypatch, available, ceiling, expected):
    from tilestream import autoplan
    monkeypatch.setattr(autoplan, "_host_memavailable", lambda: available)
    monkeypatch.setattr(pf, "_host_total_bytes_or_none", lambda: ceiling)
    assert pf.host_available_bytes() == expected


@pytest.mark.parametrize("free_gib", [11.25, 12.0])
def test_declared_free_prices_the_same_streaming_plan(tmp_path, monkeypatch, capsys, free_gib):
    from test_check_host_memory import _fetch_only_config
    from gpuwm.core.streaming import planner_machine
    path = _fetch_only_config(tmp_path)
    body = path.read_text(encoding="utf-8")
    body = (body.replace("nx = 200", "nx = 394").replace("ny = 200", "ny = 316")
            .replace("nz = 49", "nz = 76").replace("cu_physics = 1", "cu_physics = 0")
            .replace("dx = 12000.0", "dx = 3000.0"))
    path.write_text(body + '\n[tiles]\nmode = "auto"\n', encoding="utf-8")
    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda: pytest.fail("GPU probed"))
    monkeypatch.setattr(pf, "declares_the_local_card", lambda *_: False)
    exp = pf._load_experiment_any(path)
    free_bytes = int(free_gib * pf.GIB)
    direct = pf.estimate_phases(
        exp, source="era5", vram_gib=12.,
        profile=pf.card_local_memory_profile(12.),
        machine=planner_machine(vram_bytes=free_bytes, name="declared card"))
    args = _args(tmp_path, "--free-gib", str(free_gib), "--vram-gib", "12", "--json")
    args.config = path
    pf.check_main(args)
    report = json.loads(capsys.readouterr().out)
    assert report["measured_free_bytes"] == free_bytes
    assert report["envelope_budget_bytes"] == free_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert report["peak_envelope_bytes"] == direct.peak_envelope_bytes
    assert report["streamed_forecast"] == direct.streamed_forecast
    assert report["gpu_readiness"]["status"] == "not_checked"
    assert report["free_bytes_source"] == "declared (--free-gib)"


def test_declared_free_is_capped_at_declared_card(tmp_path, capsys):
    from test_check_host_memory import _fetch_only_config
    args = _args(tmp_path, "--free-gib", "24", "--vram-gib", "12", "--json")
    args.config = _fetch_only_config(tmp_path)
    pf.check_main(args)
    report = json.loads(capsys.readouterr().out)
    assert report["measured_free_bytes"] == 12 * pf.GIB
    assert report["free_bytes_capped_to_physical_bytes"] == 12 * pf.GIB


def test_declared_memory_sources_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        _args(tmp_path, "--free-gib", "12", "--budget-gib", "10")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_declared_free_requires_a_real_positive_quantity(tmp_path, value):
    with pytest.raises(ValueError, match="finite positive"):
        pf.check_main(_args(tmp_path, "--free-gib", value))


@pytest.mark.parametrize("option", ["--free-gib", "--budget-gib"])
def test_allocation_measurement_cannot_use_declared_memory(tmp_path, option):
    with pytest.raises(ValueError, match="--alloc measures"):
        pf.check_main(_args(tmp_path, "--alloc", option, "12"))
