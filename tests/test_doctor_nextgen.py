"""Hydro artifact readiness must be visible without running a forecast or fetching."""
from types import SimpleNamespace
import subprocess

import pytest

from gpuwm import doctor
from gpuwm.hydro import forcing_inputs, verify_usgs
from gpuwm import hydrofabric


@pytest.fixture
def probes(monkeypatch):
    payload = SimpleNamespace(_name="payload.so", arwen_hydrofabric_abi_version=lambda: 1,
                              arwen_hydrofabric_read_into=lambda: None)
    forcing = SimpleNamespace(_name="forcing.so", arwen_forcing_abi_version=lambda: 1,
                              arwen_forcing_prepare=lambda: None,
                              arwen_forcing_read_packet=lambda: None)
    monkeypatch.setattr(hydrofabric, "_library", lambda: payload)
    monkeypatch.setattr(forcing_inputs, "_library", lambda: forcing)
    monkeypatch.setattr(verify_usgs, "_binary", lambda: "/verify")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="rw_nextgen_verify nextgen-gauge-table-v1\n")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    return payload, forcing, calls


def test_hydro_checks_use_consumer_loaders_and_only_execute_version(probes):
    checks = doctor._nextgen_checks()
    assert {c.name for c in checks} == {
        "NextGen nextgen_hydrofabric", "NextGen arwen_nextgen_forcing",
        "NextGen rw_nextgen_verify"}
    assert all(c.status == "verified" for c in checks)
    args, kwargs = probes[2][0]
    assert args == ["/verify", "--version"]
    assert kwargs["check"] and kwargs["timeout"] == 10
    assert len(probes[2]) == 1


@pytest.mark.parametrize("failure", ["absent", "abi", "entry", "marker", "exit", "timeout"])
def test_hydro_missing_stale_or_unlaunchable_is_never_verified(monkeypatch, probes, failure):
    target = "arwen_nextgen_forcing"
    if failure == "absent":
        def missing():
            raise FileNotFoundError("forcing.so is absent")
        monkeypatch.setattr(forcing_inputs, "_library", missing)
    elif failure == "abi":
        probes[1].arwen_forcing_abi_version = lambda: 99
    elif failure == "entry":
        del probes[1].arwen_forcing_read_packet
    else:
        target = "rw_nextgen_verify"
        def run(args, **kwargs):
            if failure == "exit":
                raise subprocess.CalledProcessError(1, args)
            if failure == "timeout":
                raise subprocess.TimeoutExpired(args, 10)
            return SimpleNamespace(stdout="unrelated executable\n")
        monkeypatch.setattr(doctor.subprocess, "run", run)
    check = next(c for c in doctor._nextgen_checks() if c.name == f"NextGen {target}")
    assert check.status == "missing"
    assert "cannot run" in check.detail
    assert "nextgen_" in check.remedy


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("crate", ["tools/nextgen_forcing", "tools/nextgen_verify"])
def test_hydro_source_remedies_fetch_locked_dependencies(monkeypatch, windows, crate):
    from gpuwm import bridges
    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    monkeypatch.setattr(bridges, "cargo_is_installed", lambda: True)
    action = bridges.cargo_build_one_liner(crate)
    clone = "\n".join(bridges.build_from_clone_hint(crate))
    for command in (action, clone):
        assert command.index("cargo fetch --locked") < command.index(
            "cargo build --release --locked --offline")
    if windows:
        assert "&&" not in action
    else:
        assert "cargo fetch --locked && cargo build" in action
