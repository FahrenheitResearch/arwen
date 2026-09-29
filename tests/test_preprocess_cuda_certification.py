"""GPU preprocessing is certified per CUDA major, from evidence, and a CPU choice is never silent.

Three contracts:

* ``auto`` prepares on the card on every CUDA major with a row in
  :data:`CERTIFIED_PREPROCESS_CUDA_MAJORS`, and every row has a passing
  certification record behind it: the CPU/CUDA parity tests plus one
  real-data preparation made on both backends and compared field by field.
* Whenever ``auto`` lands on the CPU, for any reason, it prints one line
  and the preparation receipt's ``selection`` block names the same reason.
* A configuration policy that moves preparation to the CPU (a host-tiled
  domain) carries its reason from ``gpuwm go`` to the receipt.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.ingest import preprocess_backend as backend_module
from gpuwm.ingest.cpu_backend import CPU_BACKEND_ABI
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "tests" / "data" / "preprocess_cuda_certification"

#: The CPU/CUDA parity tests a certification record must show passing.
PARITY_TESTS = (
    "tests/test_preprocess_cpu_backend.py::"
    "test_rust_cpu_and_cuda_transforms_satisfy_declared_numeric_parity",
    "tests/test_preprocess_cpu_backend.py::"
    "test_full_horizontal_cpu_and_cuda_paths_satisfy_numeric_parity",
)


# ---------------------------------------------------------------------------
# The table is evidence
# ---------------------------------------------------------------------------

def _table():
    return backend_module.CERTIFIED_PREPROCESS_CUDA_MAJORS


def test_cuda_12_and_13_are_certified_for_gpu_preprocessing():
    assert sorted(_table()) == [12, 13]


@pytest.mark.parametrize("major", [12, 13])
def test_every_certified_major_has_a_passing_certification_record(major):
    """A row with no record is a certification nobody ran."""

    assert major in _table()
    path = EVIDENCE / f"cuda-{major}.json"
    assert path.is_file(), f"CUDA {major} is certified with no record at {path}"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == "gpuwm-preprocess-cuda-certification-v1"
    assert record["status"] == "PASS"
    stack = record["stack"]
    assert int(stack["cuda_runtime_version"]) // 1000 == major
    minimum = _table()[major]["cupy_major_minimum"]
    assert int(str(stack["cupy_version"]).split(".", 1)[0]) >= minimum
    tests = record["parity_tests"]
    assert tests["failed"] == 0 and tests["errors"] == 0
    assert set(PARITY_TESTS) <= set(tests["passed"])
    real = record["real_data_parity"]
    assert real["status"] == "PASS"
    assert real["reference_backend"] == "cpu"
    assert real["candidate_backend"] == "cuda"
    assert real["files"] and all(
        value["status"] == "PASS" and not value["failed_fields"]
        for value in real["files"].values())


@pytest.mark.parametrize("major", [12, 13])
def test_every_additional_case_names_its_runtime_identity_or_why_not(major):
    """A case is evidence for one CUDA runtime and CuPy.  A case with no
    identity cannot say which certified major it proves, so every case
    records both, or records them as unknown with the reason."""

    record = json.loads((EVIDENCE / f"cuda-{major}.json").read_text(
        encoding="utf-8"))
    cases = record["real_data_parity"].get("additional_cases", [])
    minimum = _table()[major]["cupy_major_minimum"]
    for entry in cases:
        case = entry["case"]
        assert {"cuda_runtime_version", "cupy_version"} <= set(case), case
        runtime, cupy = case["cuda_runtime_version"], case["cupy_version"]
        if runtime is None or cupy is None:
            reason = case.get("identity_unknown_reason")
            assert isinstance(reason, str) and reason.strip(), case
        else:
            assert "identity_unknown_reason" not in case, case
            assert int(runtime) // 1000 == major, case
            assert int(str(cupy).split(".", 1)[0]) >= minimum, case


def test_every_certified_major_names_the_extra_that_installs_its_cupy():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(
        encoding="utf-8"))["project"]["optional-dependencies"]
    for major, row in _table().items():
        extra = row["extra"]
        assert any(f"cupy-cuda{major}x" in requirement
                   for requirement in extras[extra]), extras[extra]


# ---------------------------------------------------------------------------
# auto: the card on a certified major, one line and a receipt otherwise
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh(monkeypatch):
    monkeypatch.setattr(backend_module, "_ANNOUNCED_AUTO_REASONS", set())
    monkeypatch.delenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", raising=False)


class _FakeNative:
    """The CPU bridge's identity surface, without the built library."""

    def __init__(self, bridge=None):
        self.path = Path(__file__)
        self.abi_version = CPU_BACKEND_ABI


def _fake_cuda(monkeypatch, *, cupy="14.2.0", runtime=13_020, devices=1):
    monkeypatch.setattr("gpuwm.core.device_probe.device_memory_probe_subprocess",
                        lambda **_: None)
    class Runtime:
        @staticmethod
        def getDevice():
            return 0

        @staticmethod
        def runtimeGetVersion():
            return runtime

        @staticmethod
        def getDeviceCount():
            return devices

    candidate = SimpleNamespace(
        name="cuda", array_module=SimpleNamespace(
            __version__=cupy, cuda=SimpleNamespace(runtime=Runtime())))
    monkeypatch.setattr(backend_module, "CudaPreprocessBackend",
                        lambda: candidate)
    monkeypatch.setattr(backend_module, "_gpu_runtime_installed", lambda: True)
    return candidate


def _real_cpu(monkeypatch):
    monkeypatch.setattr(backend_module, "CpuPreprocessBackend", _FakeNative)


def test_auto_prepares_on_a_certified_cuda_13_card_and_records_why(
        monkeypatch, capsys, fresh):
    candidate = _fake_cuda(monkeypatch, cupy="14.2.0", runtime=13_020)
    chosen = resolve_preprocess_backend("auto")
    assert chosen is candidate
    assert capsys.readouterr().err == ""
    assert chosen.selection["requested"] == "auto"
    assert chosen.selection["backend"] == "cuda"
    assert "13020" in chosen.selection["reason"]
    assert "certified" in chosen.selection["reason"]


def _no_device(monkeypatch):
    _fake_cuda(monkeypatch, devices=0)


def _uncertified_major(monkeypatch):
    _fake_cuda(monkeypatch, cupy="15.0.0", runtime=14_010)


def _cupy_too_old(monkeypatch):
    _fake_cuda(monkeypatch, cupy="13.6.0", runtime=13_000)


def _cupy_missing(monkeypatch):
    from gpuwm.ingest import horiz

    def _no_cupy():
        raise RuntimeError("CuPy is required for GPU horizontal interpolation")

    monkeypatch.setattr(horiz, "_cupy", _no_cupy)
    monkeypatch.setattr(backend_module, "_gpu_runtime_installed",
                        lambda: False)


def _runtime_error(monkeypatch):
    class Runtime:
        @staticmethod
        def runtimeGetVersion():
            return 13_020

        @staticmethod
        def getDeviceCount():
            raise RuntimeError(
                "cudaErrorInsufficientDriver: CUDA driver version is "
                "insufficient for CUDA runtime version")

    candidate = SimpleNamespace(
        name="cuda", array_module=SimpleNamespace(
            __version__="14.2.0", cuda=SimpleNamespace(runtime=Runtime())))
    monkeypatch.setattr(backend_module, "CudaPreprocessBackend",
                        lambda: candidate)
    monkeypatch.setattr(backend_module, "_gpu_runtime_installed", lambda: True)


def _runtime_sees_no_device(monkeypatch):
    """CUDA_VISIBLE_DEVICES="" makes the runtime raise, not count zero."""

    class Runtime:
        @staticmethod
        def runtimeGetVersion():
            return 13_020

        @staticmethod
        def getDeviceCount():
            raise RuntimeError(
                "cudaErrorNoDevice: no CUDA-capable device is detected")

    candidate = SimpleNamespace(
        name="cuda", array_module=SimpleNamespace(
            __version__="14.2.0", cuda=SimpleNamespace(runtime=Runtime())))
    monkeypatch.setattr(backend_module, "CudaPreprocessBackend",
                        lambda: candidate)
    monkeypatch.setattr(backend_module, "_gpu_runtime_installed", lambda: True)


def _sealed_without_cuda(monkeypatch, tmp_path):
    from conftest import complete_runtime_manifest

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(complete_runtime_manifest(
        platform_name="windows-x86_64")), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    monkeypatch.setattr(
        backend_module, "CudaPreprocessBackend",
        lambda: pytest.fail("a sealed CPU-only distribution probed CUDA"))


@pytest.mark.parametrize(("arrange", "says"), [
    (_no_device, "no CUDA device"),
    (_runtime_sees_no_device, "no CUDA device"),
    (_uncertified_major, "CUDA 14"),
    (_cupy_too_old, "cupy 13.6.0 is older than cupy 14"),
    (_cupy_missing, "cupy is not installed"),
    (_runtime_error, "CUDA could not start"),
    (_sealed_without_cuda, "sealed native distribution"),
])
def test_every_cpu_fallback_prints_one_line_and_lands_in_the_receipt(
        monkeypatch, capsys, fresh, tmp_path, arrange, says):
    """Uncertified runtime, no card, no cupy, a runtime that fails, a CPU-only
    distribution: each is one line on stderr and the same sentence in the
    receipt a reader opens after the log is gone."""

    if arrange is _sealed_without_cuda:
        arrange(monkeypatch, tmp_path)
    else:
        arrange(monkeypatch)
    _real_cpu(monkeypatch)
    chosen = resolve_preprocess_backend("auto")
    assert chosen.name == "cpu"
    err = capsys.readouterr().err
    assert err.count("\n") == 1, err
    assert err.startswith("warning: preprocess backend auto: ")
    assert says in err
    selection = chosen.receipt()["selection"]
    assert selection["requested"] == "auto"
    assert selection["backend"] == "cpu"
    assert says in selection["reason"]
    assert selection["reason"] in " ".join(err.split())


def test_an_explicit_backend_records_that_the_caller_named_it(
        monkeypatch, capsys, fresh):
    _real_cpu(monkeypatch)
    chosen = resolve_preprocess_backend("cpu", workers=2)
    assert chosen.receipt()["selection"] == {
        "requested": "cpu", "backend": "cpu",
        "reason": backend_module.NAMED_BY_CALLER}
    assert capsys.readouterr().err == ""


def test_a_policy_reason_replaces_named_by_the_caller(monkeypatch, fresh):
    from gpuwm.preprocess_policy import HOST_TILED_CPU_REASON

    _real_cpu(monkeypatch)
    chosen = resolve_preprocess_backend("cpu", reason=HOST_TILED_CPU_REASON)
    assert chosen.receipt()["selection"]["reason"] == HOST_TILED_CPU_REASON
    with pytest.raises(ValueError, match="sentence"):
        resolve_preprocess_backend("cpu", reason="  ")


# ---------------------------------------------------------------------------
# A host-tiled configuration: go -> prep -> the GFS preparation's receipt
# ---------------------------------------------------------------------------

def _tiled_config(tmp_path):
    path = tmp_path / "forecast.toml"
    path.write_text('[tiles]\nmode = "on"\nstore = "host"\n'
                    '[[domain]]\ngrid_id = 1\nparent_id = 0\n',
                    encoding="utf-8")
    return path


def test_the_tiled_policy_returns_its_reason():
    from gpuwm.preprocess_policy import (
        HOST_TILED_CPU_REASON, preprocess_backend_choice)

    tables = {"tiles": {"mode": "on", "store": "host"},
              "domain": [{"grid_id": 1}]}
    assert preprocess_backend_choice(source="gfs", tables=tables) == (
        "cpu", HOST_TILED_CPU_REASON)
    assert preprocess_backend_choice(source="hrrr", tables=tables) == (
        "auto", None)
    assert preprocess_backend_choice(
        source="gfs", tables=tables, requested="cuda") == ("cuda", None)


def test_go_carries_the_tiled_reason_and_says_it_in_one_line(tmp_path, capsys):
    from gpuwm import go_cli
    from gpuwm.preprocess_policy import HOST_TILED_CPU_REASON

    plan = {"source": "gfs", "config": _tiled_config(tmp_path),
            "data": tmp_path / "data", "authority": tmp_path / "authority",
            "prepared": tmp_path / "prepared"}
    argv = go_cli.prepare_command(
        plan, tmp_path / "bridge", manifest=tmp_path / "manifest.json",
        manifest_sha256="a" * 64, cycle_stamp="2026-09-08_18:00:00",
        geog_root=tmp_path / "geog")
    assert argv[argv.index("--preprocess-backend") + 1] == "cpu"
    assert argv[argv.index("--preprocess-backend-reason") + 1] \
        == HOST_TILED_CPU_REASON
    go_cli.announce_policy_backend(argv)
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert HOST_TILED_CPU_REASON in out


def test_prep_forwards_the_tiled_reason_to_the_gfs_preparation(tmp_path):
    from gpuwm import gfs_direct, source_cli
    from gpuwm.preprocess_policy import HOST_TILED_CPU_REASON

    config = _tiled_config(tmp_path)
    args = source_cli._parser().parse_args([
        "--source", "gfs", "--gfs-series", str(tmp_path / "series.tsv"),
        "--cycle", "2026-09-08_18:00:00", "--bridge", str(tmp_path / "b"),
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--experiment-config", str(config),
        "--source-manifest", str(tmp_path / "manifest.json"),
        "--source-manifest-sha256", "a" * 64,
        "--output-root", str(tmp_path / "out")])
    source_cli._apply_configuration_preprocess_default(args)
    command = source_cli._gfs_command(args)
    assert command[command.index("--preprocess-backend") + 1] == "cpu"
    assert command[command.index("--preprocess-backend-reason") + 1] \
        == HOST_TILED_CPU_REASON
    # The GFS preparation's own parser takes what prep forwards.
    parsed = gfs_direct._parser().parse_args(command[3:])
    assert parsed.preprocess_backend == "cpu"
    assert parsed.preprocess_backend_reason == HOST_TILED_CPU_REASON


def test_a_reason_without_a_named_backend_is_refused(monkeypatch, fresh):
    """auto finds its own reason; a caller's beside it would be dropped."""

    with pytest.raises(ValueError, match="named backend"):
        resolve_preprocess_backend("auto", reason="a policy sentence")


def test_prep_refuses_a_reason_it_would_not_forward(tmp_path, capsys):
    from gpuwm import source_cli

    code = source_cli.main([
        "--source", "era5", "--preprocess-backend", "cpu",
        "--preprocess-backend-reason", "a policy sentence",
        "--output-root", str(tmp_path / "out")])
    assert code != 0
    assert "--preprocess-backend-reason" in capsys.readouterr().err
