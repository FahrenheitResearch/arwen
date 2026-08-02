from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from gpuwm import __version__
from gpuwm.gfs_direct import _git_source_identity
from gpuwm.native_wrf_distribution import (
    BRIDGE_NAMES,
    CPU_BACKEND_LIBRARY,
    CUDA_KERNEL_SOURCES,
    HRRR_HELPERS,
    PYTHON_DISTRIBUTION,
    RUNTIME_SCHEMA,
    WINDOWS_CPU_BACKEND_LIBRARY,
    _BRIDGE_ABI_MARKERS,
    _cpu_backend_self_test,
    _numeric_version,
    _runtime_dependency_modules,
    bridge_identity,
    cpu_backend_library_name,
    distribution_contract,
)
from tools.build_native_wrf_distribution import (
    FORBIDDEN_WHEEL_PAYLOADS,
    _wheel_public_path_violations,
    _write_deterministic_tar,
)
from tools.build_rw_wps_release import (
    NotAGitCheckout,
    _cargo_release_environment,
    _run as _run_release_subprocess,
    _stage_rw_wps_python_project,
    _staged_internal_imports,
    _staged_verification_imports,
)


from tools.build_native_wrf_windows_distribution import (
    _dynamic_msvc_runtime_imports,
    _require_static_msvc_runtime,
    _write_deterministic_zip,
)
from tools.smoke_rw_wps_cpu_install import _safe_extract, _safe_extract_zip


ROOT = Path(__file__).parents[1]


def _stage_or_skip(destination):
    """Stage the standalone project, or say why the tree cannot.

    Staging copies tracked files only, so a tree with no git index
    cannot be staged at all -- the question has no answer there.  That
    is a property of the directory the suite is running in, not a
    finding about the product, and it must not read as one: the Linux
    pre-cut gate ran against a `git archive` extraction and got a
    subprocess traceback ending in `exit status 128`, which looked like
    a staging bug for as long as it took to read the command.

    CI's own checkout is a clone, so this never skips there.
    """

    try:
        return _stage_rw_wps_python_project(destination)
    except NotAGitCheckout as error:
        pytest.skip(str(error))


def test_distribution_builder_runs_directly_outside_checkout(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_native_wrf_distribution.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Build a hash-bound Linux gpuwm" in completed.stdout
    assert "native-WRF runtime distribution." in completed.stdout


@pytest.mark.parametrize(
    ("script", "marker"),
    (
        ("build_rw_wps_release.py", "complete Linux RW-WPS runtime"),
        ("smoke_rw_wps_cpu_install.py", "CPU backend without CUDA"),
    ),
)
def test_standalone_release_tools_run_directly_outside_checkout(
    tmp_path, script, marker,
):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert marker in completed.stdout


def test_release_builder_routes_child_progress_away_from_receipt_stdout(
        monkeypatch):
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "tools.build_rw_wps_release.subprocess.run", fake_run)
    _run_release_subprocess(["builder", "--progress"])

    assert observed["argv"] == ["builder", "--progress"]
    assert observed["check"] is True
    assert observed["stdout"] is sys.stderr
    assert observed["stderr"] is sys.stderr


def test_release_builder_stdout_is_one_strict_json_document(
        tmp_path, monkeypatch, capsys):
    import tools.build_rw_wps_release as builder

    def fake_build(_args):
        print("pip/cargo progress", file=sys.stderr)
        return {"schema": "rw-wps-build-test-v1", "status": "PASS"}

    monkeypatch.setattr(builder, "build_release", fake_build)
    assert builder.main([
        "--output-dir", str(tmp_path / "runtime"),
        "--archive", str(tmp_path / "runtime.tar.gz"),
    ]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "schema": "rw-wps-build-test-v1", "status": "PASS"}
    assert "pip/cargo progress" in captured.err


def test_windows_release_tool_runs_directly_outside_checkout(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_rw_wps_windows_release.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "complete Windows x86_64 CPU RW-WPS runtime" in completed.stdout


def test_clean_install_smoke_rejects_archive_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    payload = b"owned"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("rw-wps/../../outside")
        info.size = len(payload)
        output.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive path"):
        _safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_clean_install_smoke_rejects_zip_traversal_and_case_alias(tmp_path):
    traversal = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(traversal, "w") as output:
        output.writestr("rw-wps/../../outside", "owned")
    with pytest.raises(ValueError, match="unsafe archive path"):
        _safe_extract_zip(traversal, tmp_path / "extract-traversal")
    assert not (tmp_path / "outside").exists()

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as output:
        output.writestr("rw-wps/File.txt", "first")
        output.writestr("rw-wps/file.txt", "second")
    with pytest.raises(ValueError, match="duplicate archive path"):
        _safe_extract_zip(duplicate, tmp_path / "extract-duplicate")


def test_release_archive_is_byte_identical_across_build_roots(tmp_path):
    roots = []
    archives = []
    for index in range(2):
        source = tmp_path / f"build-{index}" / "rw-wps-runtime"
        (source / "bin").mkdir(parents=True)
        executable = source / "bin" / "rw-wps"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
        (source / "manifest.json").write_text(
            '{"status":"READY"}\n', encoding="utf-8")
        # Filesystem clocks and parent locations are explicitly outside the
        # archive identity.
        os.utime(source / "manifest.json", (1000 + index, 2000 + index))
        archive = tmp_path / f"result-{index}" / "rw-wps-runtime.tar.gz"
        archive.parent.mkdir()
        _write_deterministic_tar(source, archive)
        roots.append(source)
        archives.append(archive)
    assert roots[0].parent != roots[1].parent
    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_windows_release_zip_is_byte_identical_across_build_roots(tmp_path):
    archives = []
    for index in range(2):
        source = tmp_path / f"windows-build-{index}" / "rw-wps-runtime"
        (source / "bin").mkdir(parents=True)
        (source / "bin" / "rw-wps.cmd").write_bytes(
            b"@echo off\r\nexit /b 0\r\n")
        (source / "manifest.json").write_text(
            '{"status":"READY"}\n', encoding="utf-8")
        os.utime(source / "manifest.json", (1000 + index, 2000 + index))
        archive = tmp_path / f"windows-result-{index}" / "rw-wps-runtime.zip"
        archive.parent.mkdir()
        _write_deterministic_zip(source, archive)
        archives.append(archive)
    assert archives[0].read_bytes() == archives[1].read_bytes()


def test_windows_runtime_rejects_dynamic_msvc_crt_markers(tmp_path):
    static = tmp_path / "static.exe"
    static.write_bytes(b"MZ\x00KERNEL32.dll\x00")
    assert _dynamic_msvc_runtime_imports(static) == []
    assert _require_static_msvc_runtime(static) == []

    dynamic = tmp_path / "dynamic.exe"
    dynamic.write_bytes(b"MZ\x00VCRUNTIME140.dll\x00")
    assert _dynamic_msvc_runtime_imports(dynamic) == ["vcruntime140.dll"]
    with pytest.raises(RuntimeError, match="imports dynamic MSVC CRT"):
        _require_static_msvc_runtime(dynamic)


def test_rust_release_build_remaps_checkout_and_ignores_host_flags(
    tmp_path, monkeypatch,
):
    source = tmp_path / "checkout with spaces"
    target = tmp_path / "cargo-target"
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=native")
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "caller-specific")

    environment = _cargo_release_environment(
        source_root=source,
        target_dir=target,
        source_date_epoch="1784709589",
    )

    assert "RUSTFLAGS" not in environment
    flags = environment["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[0] == (
        f"--remap-path-prefix={source.resolve()}=/usr/src/rw-wps"
    )
    if sys.platform == "win32":
        assert flags[1:] == [
            "-C", "link-arg=/Brepro",
            "-C", "target-feature=+crt-static",
        ]
    else:
        assert flags == [flags[0]]
    assert environment["CARGO_TARGET_DIR"] == str(target)
    assert environment["SOURCE_DATE_EPOCH"] == "1784709589"


def test_native_wrf_contract_is_versioned_and_explicit():
    contract = distribution_contract("linux-x86_64")
    windows_contract = distribution_contract("windows-x86_64")
    assert contract["schema"] == RUNTIME_SCHEMA
    assert contract["gpuwm_version"] == __version__
    assert contract["python_distribution"] == PYTHON_DISTRIBUTION == "rw-wps"
    assert contract["runtime_forbidden"] == ["WPS", "real.exe"]
    assert contract["platform"]["python"] == ">=3.11"
    assert contract["platform"]["cuda_runtime_family"] == "12.x"
    assert contract["platform"]["identifier"] == "linux-x86_64"
    assert windows_contract["platform"] == {
        "identifier": "windows-x86_64",
        **windows_contract["supported_platforms"]["windows-x86_64"],
    }
    assert windows_contract["platform"]["operating_system"] == "Windows"
    assert windows_contract["platform"]["shell"] == "PowerShell 5.1+"
    assert windows_contract["platform"]["cuda_runtime_family"] is None
    assert windows_contract["platform"]["msvc_runtime"] == "statically linked"
    assert contract["public_controls"]["hrrr"]["cadence_seconds"] == 3600
    assert "seal_hrrr_native_bridge.py" in HRRR_HELPERS
    assert "prepare_hrrr_wrf.py" in HRRR_HELPERS
    assert "download_hrrr_native_subset.py" in HRRR_HELPERS
    assert "common.cuh" in CUDA_KERNEL_SOURCES
    assert "diagnostics.cu" in CUDA_KERNEL_SOURCES
    assert "vert_interp.cu" in CUDA_KERNEL_SOURCES
    assert "thompson.cu" not in CUDA_KERNEL_SOURCES
    assert "dycore.cu" not in CUDA_KERNEL_SOURCES
    assert contract["bundled_native_libraries"] == [CPU_BACKEND_LIBRARY]
    assert windows_contract["bundled_native_libraries"] == [
        WINDOWS_CPU_BACKEND_LIBRARY]
    assert contract["bundled_native_libraries_by_platform"][
        "windows-x86_64"] == WINDOWS_CPU_BACKEND_LIBRARY
    assert contract["supported_platforms"]["windows-x86_64"][
        "backends"] == ["cpu"]
    assert contract["bundled_native_bridges"] == list(BRIDGE_NAMES)
    assert "grib2_inventory" in BRIDGE_NAMES
    assert "grib2_dump" in BRIDGE_NAMES
    assert contract["preprocess_backends"]["cpu"] \
        == "rust-scoped-threads-fp32-v1"
    assert "parallel CPU" in contract["public_controls"]["gfs"]["preprocessing"]
    assert "parallel CPU" in contract["public_controls"]["era5"]["preprocessing"]
    assert "max_dom=4" in contract["public_controls"]["20crv3"]["domain"]
    assert "packaged" in contract["public_controls"]["20crv3"]["authorities"]
    assert "d01 through d04" in (
        contract["public_controls"]["mapped"]["real_gates"]
    )
    assert "max_dom" in contract["public_controls"]["mapped"]["domain"]
    assert "explicit" in contract["public_controls"]["hrrr"]["domain"]
    assert "explicit" in contract["public_controls"]["gfs"]["domain"]
    assert "explicit" in contract["public_controls"]["era5"]["domain"]
    assert "frozen 49" not in json.dumps(contract["public_controls"])


def test_runtime_selects_platform_cpu_library_names():
    assert cpu_backend_library_name("Linux") == CPU_BACKEND_LIBRARY
    assert cpu_backend_library_name("Windows") == WINDOWS_CPU_BACKEND_LIBRARY
    with pytest.raises(RuntimeError, match="unsupported"):
        cpu_backend_library_name("Plan9")


def test_installed_launcher_skips_device_gate_for_cpu_and_nonexecuting_modes():
    launcher = (
        Path(__file__).parents[1] / "tools" / "gpuwm_native_wrf_launcher.sh"
    ).read_text(encoding="utf-8")
    assert "--author-only|--dry-run" in launcher
    assert "--list-sources" in launcher
    assert "--show-source=*" in launcher
    assert "--show-support-matrix" in launcher
    assert "--namelist-support-report" in launcher
    assert "--preprocess-backend=cpu|--preprocess-backend=auto" in launcher
    assert "cpu|auto) skip_gpu=1" in launcher
    assert "runtime_check+=(--skip-gpu)" in launcher


def test_installed_launcher_keeps_the_entire_python_chain_immutable():
    launcher = (
        Path(__file__).parents[1] / "tools" / "gpuwm_native_wrf_launcher.sh"
    ).read_text(encoding="utf-8")
    no_user_site = "export PYTHONNOUSERSITE=1"
    no_bytecode = "export PYTHONDONTWRITEBYTECODE=1"
    first_python = 'PYTHONPATH="$runtime" "$python_path" -P -m'

    assert no_user_site in launcher
    assert no_bytecode in launcher
    assert launcher.index(no_user_site) < launcher.index(first_python)
    assert launcher.index(no_bytecode) < launcher.index(first_python)


def test_installer_and_runtime_support_explicit_no_gpu_authoring_mode():
    installer = (
        Path(__file__).parents[1] / "tools" / "install_gpuwm_native_wrf.sh"
    ).read_text(encoding="utf-8")
    assert "--skip-gpu" in installer
    assert "-name 'rw_wps-*.whl'" in installer
    assert "gpuwm-*.whl" not in installer
    assert "runtime_check+=(--skip-gpu)" in installer
    assert "cupy" not in _runtime_dependency_modules(False)
    assert "matplotlib" not in _runtime_dependency_modules(False)
    assert _runtime_dependency_modules(True)["cupy-cuda12x"] == "cupy"


def test_windows_installer_and_launcher_are_fail_closed():
    installer = (ROOT / "tools" / "install_gpuwm_native_wrf_windows.ps1").read_text(
        encoding="utf-8")
    launcher = (ROOT / "tools" / "gpuwm_native_wrf_launcher_windows.ps1").read_text(
        encoding="utf-8")
    for marker in (
        "Get-FileHash", "ReparsePoint", "Duplicate SHA256SUMS path",
        "Refusing existing GPUWM_INSTALL_ROOT", "--no-index", "--no-deps",
        "--skip-gpu", '".rw-{0}-{1}"', "Substring(0, 8)",
        "projected path length", "[System.IO.Directory]::Move",
        "importlib.util.cache_from_source",
        "str(sys.flags.optimize)",
        "$cacheProjectionScript | & $python -",
        "[System.Math]::Max($partial.Length, $target.Length)",
    ):
        assert marker in installer
    assert "& $python -c" not in installer
    assert ".rw-wps-runtime.partial-" not in installer
    assert "Move-Item -LiteralPath $partial" not in installer
    for marker in (
        "Get-FileHash", "ReparsePoint", "GPUWM_PYTHON differs",
        "--preprocess-backend=cuda", "gpuwm_preprocess_cpu.dll",
        "--skip-gpu", '$env:PYTHONNOUSERSITE = "1"',
        '$env:PYTHONDONTWRITEBYTECODE = "1"',
    ):
        assert marker in launcher
    first_python = launcher.index("& $python -B -P -m")
    assert launcher.index('$env:PYTHONNOUSERSITE = "1"') < first_python
    assert launcher.index('$env:PYTHONDONTWRITEBYTECODE = "1"') < first_python
    assert launcher.count("& $python -B -P -m") == 2


def test_standalone_python_project_excludes_forecast_executor(tmp_path):
    staged = tmp_path / "rw-wps-python"
    receipt = _stage_or_skip(staged)
    files = set(receipt["files"])

    assert receipt["distribution"] == "rw-wps"
    assert receipt["forecast_executor_files"] == []
    assert receipt["verification_imports"] == []
    assert receipt["unresolved_internal_imports"] == []
    assert receipt["optional_internal_imports"]
    assert all(item["optional_reason"]
               for item in receipt["optional_internal_imports"])
    assert {
        "gpuwm/multi_run.py", "gpuwm/stream.py",
    } <= FORBIDDEN_WHEEL_PAYLOADS
    assert not (FORBIDDEN_WHEEL_PAYLOADS & files)
    assert not any(name.startswith("gpuwm/verify/") for name in files)
    assert "gpuwm/source_cli.py" in files
    assert "gpuwm/physics_registry.py" in files
    assert "gpuwm/physics_registry_v2.json" in files
    assert "gpuwm/mapped_direct.py" in files
    assert "gpuwm/twentycrv3.py" in files
    assert "gpuwm/core/state.py" in files
    assert "gpuwm/core/thompson_contract.py" in files
    assert "gpuwm/core/nssl2_contract.py" in files
    assert "gpuwm/core/kernels/vert_interp.cu" in files
    assert "gpuwm/offline_child.py" not in files
    assert "gpuwm/offline_child_run.py" not in files
    assert "gpuwm/offline_child_smoke.py" not in files
    assert "gpuwm/multi_run.py" not in files
    assert "gpuwm/stream.py" not in files
    # `gpuwm resume` locates a forecast checkpoint for the supervisor's
    # `run --restart` dispatch: nothing here imports it, no entry point
    # exposes it, and its own lookups are gpuwm.supervisor (forbidden
    # above) and gpuwm.io.restart (not staged).  A preprocessing wheel
    # has no checkpoints to resume from.
    assert "gpuwm/resume.py" not in files
    # doctor, on the other hand, belongs: a preprocessing install is
    # exactly the one that needs to be told which bridge is missing.  It
    # is here because its WPS_GEOG check reads the dataset list from
    # gpuwm.geog_assets, which stages the tree, rather than from the
    # domain wizard, which imports the CUDA front door.
    assert "gpuwm/doctor.py" in files
    # The resolved-scheme vertical preflight ships with preprocessing:
    # rw-wps must refuse an nz a selected component cannot run.
    assert "gpuwm/physics_vertical_contract.py" in files
    assert "tools/hrrr_state_proof.py" not in files
    assert "tools/hrrr_two_domain_forecast.py" not in files
    assert "LICENSE" in files
    assert "NOTICE" in files
    metadata = (staged / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "rw-wps"' in metadata
    assert 'license = { file = "LICENSE" }' in metadata
    assert (staged / "LICENSE").read_text(encoding="utf-8").startswith(
        "                              Apache License"
    )
    assert "matplotlib" not in metadata


def test_standalone_python_project_imports_without_source_tree_or_cupy(tmp_path):
    staged = tmp_path / "rw-wps-python"
    _stage_or_skip(staged)
    script = r"""
from importlib.abc import MetaPathFinder
from pathlib import Path
import os
import sys

class RejectExternalModules(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "cupy",
            "gpuwm.cli",
            "gpuwm.core.model",
            "gpuwm.core.physics",
            "gpuwm.multi_run",
            "gpuwm.runtime",
            "gpuwm.stream",
            "gpuwm.supervisor",
            "gpuwm.verify",
        )
        if any(fullname == name or fullname.startswith(name + ".")
               for name in forbidden):
            raise ModuleNotFoundError(f"blocked RW-WPS-external module: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalModules())
root = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
import gpuwm.source_cli
import gpuwm.era5_direct
import gpuwm.gfs_direct
import gpuwm.hrrr_hierarchy_direct
import gpuwm.mapped_direct
import gpuwm.twentycrv3
import gpuwm.twentycrv3_wrf
# The estate report is part of this surface: a preprocessing install is
# the one that most needs to be told which bridge is missing, and it
# must reach that conclusion without the CUDA front door the domain
# wizard drags in.
import gpuwm.doctor
import tools.hrrr_single_domain_benchmark
from gpuwm.core.nest_interp import register_nest, sint
import numpy as np

capabilities = tools.hrrr_single_domain_benchmark.runner_capabilities()
assert capabilities["readiness"] == \
    "PREPARATION_ONLY_FORECAST_EXECUTOR_OMITTED"
assert capabilities["modes"]["prepare-only"]["available"] is True
assert capabilities["modes"]["forecast"]["available"] is False
assert "gpuwm.core.model" in \
    capabilities["modes"]["forecast"]["missing_executor_modules"]
assert capabilities["standalone_rw_wps_wheel"][
    "forecast_executor_included"] is False

registration = register_nest(
    nri=3, nrj=3, i_parent_start=4, j_parent_start=3,
    child_nx=9, child_ny=9, parent_nx=12, parent_ny=12,
    stagger="", wrapper="interp",
)
parent = np.linspace(
    -2.0, 4.0, 2 * registration.nyp * registration.nxp,
    dtype=np.float32,
).reshape(2, registration.nyp, registration.nxp)
child = sint(parent, registration)
assert child.shape == (2, registration.nyc, registration.nxc)
assert np.isfinite(child).all()
for module in (
    gpuwm.source_cli,
    gpuwm.era5_direct,
    gpuwm.gfs_direct,
    gpuwm.hrrr_hierarchy_direct,
    gpuwm.mapped_direct,
    gpuwm.twentycrv3,
    gpuwm.twentycrv3_wrf,
    gpuwm.doctor,
    tools.hrrr_single_domain_benchmark,
):
    assert Path(module.__file__).resolve().is_relative_to(root)
# Importing doctor is not the check; running the geography check is.
# The old spelling reached gpuwm.domain_wizard from inside the function
# body, where an import test that only imports modules never sees it.
gpuwm.doctor._geog_tree_checks(Path(os.environ["RW_WPS_STAGED_ROOT"])
                               / "no-such-geog-root")
# and the call really did take the lazy-import path, so the absences
# below are evidence rather than an artifact of nothing having run
assert "gpuwm.geog_assets" in sys.modules
for name in (
    "cupy", "gpuwm.cli", "gpuwm.core.model", "gpuwm.core.physics",
    "gpuwm.domain_wizard", "gpuwm.multi_run", "gpuwm.resume",
    "gpuwm.runtime", "gpuwm.stream", "gpuwm.supervisor", "gpuwm.verify",
):
    assert name not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    completed = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_standalone_package_boundary_scan_rejects_verify_imports(tmp_path):
    package = tmp_path / "staged" / "gpuwm"
    package.mkdir(parents=True)
    (package / "direct.py").write_text(
        "from gpuwm.verify.npref import np_sint\n", encoding="utf-8")
    (package / "dynamic.py").write_text(
        "import importlib\n"
        "importlib.import_module('gpuwm.verify.metrics')\n",
        encoding="utf-8",
    )

    assert _staged_verification_imports(tmp_path / "staged") == [
        {
            "path": "gpuwm/direct.py",
            "line": 1,
            "module": "gpuwm.verify.npref",
            "kind": "from",
        },
        {
            "path": "gpuwm/dynamic.py",
            "line": 2,
            "module": "gpuwm.verify.metrics",
            "kind": "dynamic",
        },
    ]


def test_standalone_package_boundary_scan_rejects_omitted_internal_imports(
        tmp_path):
    package = tmp_path / "staged" / "gpuwm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "direct.py").write_text(
        "from gpuwm.io.restart import setup_fingerprint\n",
        encoding="utf-8",
    )

    assert _staged_internal_imports(tmp_path / "staged") == [{
        "path": "gpuwm/direct.py",
        "line": 1,
        "module": "gpuwm.io.restart",
        "kind": "from",
    }]


def test_cpu_native_export_modules_import_without_cupy():
    """The Rust/NumPy route must not import GPU or forecast executors."""

    script = r"""
from importlib.abc import MetaPathFinder
import sys

class RejectCupy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "cupy",
            "gpuwm.cli",
            "gpuwm.core.model",
            "gpuwm.runtime",
            "gpuwm.supervisor",
        )
        if any(fullname == name or fullname.startswith(name + ".")
               for name in forbidden):
            raise ModuleNotFoundError(
                f"blocked RW-WPS-external module: {fullname}")
        return None

sys.meta_path.insert(0, RejectCupy())
import numpy as np
from gpuwm.config import RunConfig
from gpuwm.core.state import DomainState
import gpuwm.era5_direct
import gpuwm.gfs_direct
import gpuwm.hrrr_hierarchy_direct
import gpuwm.mapped_direct
import gpuwm.twentycrv3_wrf

cfg = RunConfig(
    nx=3, ny=2, nz=2, dx=1000.0, dy=1000.0, ztop=10000.0,
    dt=1.0, run_seconds=1.0, moist=True, mp_physics=6,
)
state = DomainState(cfg, array_module=np)
assert isinstance(state.u, np.ndarray)
try:
    DomainState(cfg)
except RuntimeError as error:
    assert "CuPy is required for CUDA forecast state" in str(error)
else:
    raise AssertionError("default CUDA state did not reject missing CuPy")
assert "cupy" not in sys.modules
for module in (
    "gpuwm.cli", "gpuwm.core.model", "gpuwm.runtime", "gpuwm.supervisor"
):
    assert module not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_cpu_backend_self_test_executes_parallel_fp32_arithmetic():
    class Plan:
        def apply(self, source, *, method, workers):
            assert source.dtype.name == "float32"
            assert method == "bilinear"
            assert workers in (1, 3)
            return source.reshape(-1)[[0, 1]].mean(dtype=source.dtype)[None, None] \
                + source.dtype.type([-0.5, 1.0, 2.5])[None, :]

    class Backend:
        def indexed_plan(self, source_shape, y, x):
            assert source_shape == (2, 2)
            assert y.dtype.name == x.dtype.name == "float32"
            return Plan()

    receipt = _cpu_backend_self_test(Backend())
    assert receipt["status"] == "PASS"
    assert receipt["worker_counts"] == [1, 3]
    assert receipt["output_values"] == [[1.0, 2.5, 4.0]]
    assert len(receipt["output_sha256"]) == 64


def test_runtime_version_parser_handles_local_and_post_releases():
    assert _numeric_version("13.0.0") >= (13, 0)
    assert _numeric_version("1.7.4.post1") >= (1, 6)


def test_wheel_release_audit_rejects_developer_profile_paths(tmp_path):
    wheel = tmp_path / "rw_wps_fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("gpuwm/public.py", "ROOT = '/case/input'\n")
        # Assembled from fragments on purpose: this repository's own
        # release-snapshot scan reads every shipped file for exactly this
        # marker, and a literal here would trip it on the fixture.
        archive.writestr(
            "gpuwm/private.md",
            "staged under C:/" + "Users/example/Downloads/private\n",
        )
    assert _wheel_public_path_violations(wheel) == [{
        "path": "gpuwm/private.md",
        # Assembled, not written literally: see the fixture above.
        "marker": "c:/" + "users/",
        "kind": "Windows user profile",
    }]


def _fake_executable_bridge(
    path: Path, marker: bytes | None,
) -> None:
    path.write_bytes(b"\x7fELF" + (marker or b"") + b"\0fixture")
    path.chmod(0o755)


def _fake_pe_bridge(path: Path, marker: bytes) -> None:
    payload = bytearray(0x84)
    payload[:2] = b"MZ"
    payload[0x3c:0x40] = (0x80).to_bytes(4, "little")
    payload[0x80:0x84] = b"PE\0\0"
    payload.extend(marker)
    path.write_bytes(payload)


@pytest.mark.parametrize(
    ("name", "usage"),
    (
        ("grib2_inventory", "usage: grib2_inventory INPUT.grib2 [--decode]"),
        ("grib2_dump", "usage: grib2_dump INPUT.grib2 INDEX OUTPUT.f32"),
    ),
)
def test_generic_grib2_bridge_identity_binds_tabular_abi(
    tmp_path, monkeypatch, name, usage,
):
    monkeypatch.setattr(
        "gpuwm.native_wrf_distribution.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr=usage,
        ),
    )
    bridge = tmp_path / name
    _fake_executable_bridge(bridge, _BRIDGE_ABI_MARKERS[name])
    identity = bridge_identity(bridge, name)
    assert identity["tabular_abi_marker_sha256"] == hashlib.sha256(
        _BRIDGE_ABI_MARKERS[name]
    ).hexdigest()

    _fake_executable_bridge(bridge, None)
    with pytest.raises(RuntimeError, match="incompatible tabular ABI"):
        bridge_identity(bridge, name)


def test_bridge_identity_accepts_pe_and_preserves_executable_suffix(
    tmp_path, monkeypatch,
):
    bridge = tmp_path / "grib2_inventory.exe"
    _fake_pe_bridge(bridge, _BRIDGE_ABI_MARKERS["grib2_inventory"])

    def run(argv, **_kwargs):
        assert Path(argv[0]).suffix == ".exe"
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="usage: grib2_inventory INPUT.grib2 [--decode]",
        )

    monkeypatch.setattr(
        "gpuwm.native_wrf_distribution.subprocess.run",
        run,
    )
    identity = bridge_identity(bridge, "grib2_inventory")
    assert identity["binary_format"] == "pe"


def test_bridge_identity_executes_frozen_bytes_and_rejects_source_drift(
    tmp_path, monkeypatch
):
    bridge = tmp_path / "grib2_inventory"
    _fake_executable_bridge(
        bridge,
        _BRIDGE_ABI_MARKERS["grib2_inventory"],
    )
    original = bridge.read_bytes()

    def run(argv, **_kwargs):
        executed = Path(argv[0])
        assert executed != bridge
        assert executed.read_bytes() == original
        bridge.write_bytes(original + b"changed")
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="usage: grib2_inventory INPUT.grib2 [--decode]",
        )

    monkeypatch.setattr(
        "gpuwm.native_wrf_distribution.subprocess.run",
        run,
    )
    with pytest.raises(RuntimeError, match="changed during identity probe"):
        bridge_identity(bridge, "grib2_inventory")


def test_bridge_identity_translates_probe_timeout(tmp_path, monkeypatch):
    bridge = tmp_path / "grib2_inventory"
    _fake_executable_bridge(
        bridge,
        _BRIDGE_ABI_MARKERS["grib2_inventory"],
    )
    monkeypatch.setattr(
        "gpuwm.native_wrf_distribution.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], timeout=10)
        ),
    )
    with pytest.raises(RuntimeError, match="identity probe timed out"):
        bridge_identity(bridge, "grib2_inventory")


def test_bridge_identity_translates_noexec_probe_error(tmp_path, monkeypatch):
    bridge = tmp_path / "grib2_inventory"
    _fake_executable_bridge(
        bridge,
        _BRIDGE_ABI_MARKERS["grib2_inventory"],
    )
    monkeypatch.setattr(
        "gpuwm.native_wrf_distribution.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("noexec")
        ),
    )
    with pytest.raises(RuntimeError, match="temporary filesystem may be mounted noexec"):
        bridge_identity(bridge, "grib2_inventory")


def test_gfs_provenance_prefers_bound_distribution_manifest(tmp_path, monkeypatch):
    from conftest import complete_runtime_manifest

    manifest = tmp_path / "manifest.json"
    document = complete_runtime_manifest()
    document["source"].update({"commit": "a" * 40, "tree": "b" * 40})
    manifest.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    identity = _git_source_identity()
    assert identity["identity_source"] == "gpuwm-native-distribution-manifest"
    assert identity["commit"] == "a" * 40
    assert identity["distribution_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()).hexdigest()


def test_a_freshly_sealed_artifact_carries_the_distribution_version():
    """The stale constant was not merely printed -- it was SEALED IN.

    Native runtime contracts, prepared-cache writer identity and the
    standalone RW-WPS wheel's own pyproject all stamped ``0.1.1`` on a
    1.1.1 release.  Two of those feed gates that compare a wheel's
    metadata version against ``gpuwm.__version__``, so the stale
    constant was one truthful version away from refusing its own seal.
    """

    from importlib import metadata
    import gpuwm
    from tools.build_rw_wps_release import _standalone_pyproject

    distribution = metadata.version("gpuwm")
    for platform in ("linux-x86_64", "windows-x86_64"):
        contract = distribution_contract(platform)
        assert contract["gpuwm_version"] == distribution, platform

    # The standalone wheel's version is the one the sealed-runtime
    # receipt compares against; it must be stamped, never typed.
    stamped = [line for line in _standalone_pyproject().splitlines()
               if line.startswith("version = ")]
    assert stamped == [f'version = "{distribution}"'], stamped
    assert gpuwm.__version__ == distribution


def test_a_freshly_written_prepared_cache_stamps_this_release(tmp_path):
    from importlib import metadata

    from gpuwm.ingest.prepared_cache import (
        CACHE_WRITER_KEY, cache_writer_version)

    header = {CACHE_WRITER_KEY: {"gpuwm_version": __version__}}
    assert cache_writer_version(header) == metadata.version("gpuwm")
