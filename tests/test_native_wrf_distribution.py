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
    BRIDGE_WORKSPACES,
    BUNDLED_BRIDGES,
    CPU_BACKEND_LIBRARY,
    CUDA_KERNEL_SOURCES,
    HRRR_HELPERS,
    PYTHON_DISTRIBUTION,
    RUNTIME_SCHEMA,
    WINDOWS_CPU_BACKEND_LIBRARY,
    _BRIDGE_ABI_MARKERS,
    _BRIDGE_USAGE_MARKERS,
    _cpu_backend_self_test,
    _numeric_version,
    _runtime_dependency_modules,
    add_bridge_options,
    bridge_identity,
    bridge_inputs,
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
    assert "gpuwm/source_drivability.py" in files
    assert "gpuwm/core/track_boundary.py" in files
    assert "gpuwm/obs/goes_window.py" not in files
    assert "gpuwm/source_cli.py" in files
    assert "gpuwm/physics_registry.py" in files
    assert "gpuwm/physics_registry_v2.json" in files
    assert "gpuwm/mapped_direct.py" in files
    assert "gpuwm/twentycrv3_direct.py" in files
    # The packaged profiles a wheel user can name on `--source`, shipped as
    # the table data they are.  `gpuwm/twentycrv3.py` used to be asserted
    # here: a 1,185-line per-model NetCDF-CF decoder with no consumer,
    # replaced by these three documents plus one adapter row.
    for role in ("mapping", "composition", "provenance"):
        assert f"gpuwm/authorities/rw-wps-20crv3-netcdf.{role}.json" in files
        assert (
            f"gpuwm/authorities/rw-wps-20crv3-member-grib2.{role}.json"
            in files)
    assert "gpuwm/core/state.py" in files
    assert "gpuwm/core/nest_fields.py" in files
    assert "gpuwm/core/ozone_contract.py" in files
    assert "gpuwm/core/inflow_perturbation.py" in files
    assert "gpuwm/core/attribute_tracking.py" in files
    assert "gpuwm/toml_document.py" in files
    assert "gpuwm/prepared_source_schemas.py" in files
    for name in ("metem_forecast", "wrfinput_forecast", "launchpad_api", "tui_worker",
                 "remote_cli", "remote_worker", "research_workspaces",
                 "starter_template", "tui_products", "case_catalog",
                 "case_catalog_import", "companion_query", "companion_domains",
                 "companion_setups",
                 "companion_forcing", "configuration_recovery", "remote_artifacts",
                 "remote_input_transfer", "remote_plan", "remote_processed",
                 "render", "render_receipts", "background_contract", "regional_preparation",
                 "local_da", "local_da_fetch", "local_da_observations", "local_da_runtime",
                 "cyclone_seed", "cyclone_sources"):
        assert f"gpuwm/{name}.py" not in files
    assert "gpuwm/ingest/case_store.py" not in files
    assert "gpuwm/ingest/relocation_continuation.py" not in files
    assert "gpuwm/core/thompson_contract.py" in files
    assert "gpuwm/core/nssl2_contract.py" in files
    assert "gpuwm/core/kernels/vert_interp.cu" in files
    assert "gpuwm/offline_child.py" not in files
    assert "gpuwm/offline_child_geography.py" not in files
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
    # The saved history a resumed forecast's first pictures read: its
    # importers are the excluded go and live-render doors, and it reaches
    # gpuwm.io.restart, gpuwm.resume and gpuwm.live_products.  Staged, it
    # made this staging refuse outright.
    assert "gpuwm/restart_render.py" not in files
    # Auto's card-load probe ships as its own leaf and the forecast memory
    # preflight that re-exports it stays out, so the backend selector
    # reaches nothing this package omits: no optional import stands in for
    # the load reading.
    assert "gpuwm/core/device_probe.py" in files
    assert "gpuwm/core/preflight.py" not in files
    assert not [item for item in receipt["optional_internal_imports"]
                if item["path"] == "gpuwm/ingest/preprocess_backend.py"]
    # The run disk projection and its download and preparation pricing are
    # read only by the excluded run-plan door, and the pricing parses a
    # fetch argv with gpuwm.cli's parser: staged, the 2.8.0 cut battery
    # found an unresolved import of gpuwm.cli here.
    assert "gpuwm/disk_budget.py" not in files
    assert "gpuwm/download_budget.py" not in files
    # The source table reads ArchiveWindow from the date guidance at import,
    # so the guidance ships; its availability answer reaches the domain
    # wizard, which does not, and that one import is named optional.
    assert "gpuwm/source_availability.py" in files
    assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
        ("gpuwm/source_availability.py", "gpuwm.domain_wizard")}
    # The initial-state perturbation ships with the ingest package; the
    # radiation ceiling it reads when a forecast runner builds one is in
    # the RTE+RRTMGP module, which does not, and that import is named
    # optional.
    assert "gpuwm/ingest/init_perturbation.py" in files
    assert "gpuwm/core/rrtmgp.py" not in files
    assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
        ("gpuwm/ingest/init_perturbation.py", "gpuwm.core.rrtmgp")}
    # Preparation builds and seals the moving-nest statics corridor, so the
    # corridor module ships.  Its check of a corridor sealed under an older
    # build contract reads the relocation planner and the relocation
    # initializer's overlap rule; only the prepared-tree forecast runner
    # loads a corridor, so those two stay out and the imports are named
    # optional.  Unnamed, they made this staging refuse outright.
    assert "gpuwm/static/corridor.py" in files
    assert "gpuwm/core/nest_relocation.py" not in files
    assert "gpuwm/ingest/relocation_init.py" not in files
    assert "gpuwm/prepared_domain_tree_forecast.py" not in files
    assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
        ("gpuwm/static/corridor.py", "gpuwm.core.nest_relocation"),
        ("gpuwm/static/corridor.py", "gpuwm.ingest.relocation_init")}
    # The steep-terrain clock sets the time step a forecast starts with;
    # its importers are the excluded runners and it reaches
    # gpuwm.core.adaptive_clock, which does not ship.  Staged, it made
    # this staging refuse outright.
    assert "gpuwm/acoustic_adaptation.py" not in files
    assert "gpuwm/terrain_clock.py" not in files
    assert "gpuwm/core/adaptive_clock.py" not in files
    # The chained writer ships with the era5, gfs and mapped routes; the
    # forecast admission it reaches only when it chains does not, and a
    # preparation-only install never chains, so that import is optional.
    assert "gpuwm/ingest/boundary_stream.py" in files
    assert "gpuwm/core/urban_state.py" not in files
    assert {(item["path"], item["module"]) for item in receipt["optional_internal_imports"]} >= {
        ("gpuwm/ingest/boundary_stream.py", "gpuwm.core.preflight"),
        ("gpuwm/ingest/boundary_stream.py", "gpuwm.core.urban_state"),
        ("gpuwm/ingest/boundary_stream.py", "gpuwm.prepared_domain_tree_forecast")}
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
            "gpuwm.companion_query",
            "gpuwm.companion_domains",
            "gpuwm.companion_setups",
            "gpuwm.companion_forcing",
            "gpuwm.configuration_recovery",
            "gpuwm.core.model",
            "gpuwm.core.physics",
            "gpuwm.multi_run",
            "gpuwm.runtime",
            "gpuwm.remote_artifacts",
            "gpuwm.remote_input_transfer",
            "gpuwm.remote_plan",
            "gpuwm.remote_processed",
            "gpuwm.render",
            "gpuwm.render_receipts",
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
from gpuwm.fetch import validate_fetch_hints
from gpuwm.source_drivability import intent_drivability
assert intent_drivability()["gfs"]["routes"]
validate_fetch_hints({"source": "gfs", "hours": 3}, source="standalone control")
assert "gpuwm.runplan" not in sys.modules
import gpuwm.gfs_direct
import gpuwm.hrrr_hierarchy_direct
import gpuwm.mapped_direct
import gpuwm.twentycrv3_direct
import gpuwm.twentycrv3_wrf
# The estate report is part of this surface: a preprocessing install is
# the one that most needs to be told which bridge is missing, and it
# must reach that conclusion without the CUDA front door the domain
# wizard drags in.
import gpuwm.doctor
import tools.hrrr_single_domain_benchmark
from gpuwm.core.nest_interp import register_nest, sint
import numpy as np

# Exercise the shared preparation contracts, not only their imports. They
# previously reached forecast-only owners when a real config/input used them.
from dataclasses import replace
from datetime import datetime
import tomllib
from gpuwm.config import RunConfig
from gpuwm.experiment import experiment_from_run_config
from gpuwm.core.ozone_contract import cam_ozone_domain_ids
from gpuwm.ingest.analyzed_numbers import metgrid_number_targets
from gpuwm.experiment_document import render_experiment_document
from gpuwm.prepared_source_schemas import mapped_sources, source_schemas
from gpuwm.core.storm_tracking import build_follow_config
from gpuwm.core.attribute_tracking import validate_attribute_domains
from types import SimpleNamespace

cfg = RunConfig(nx=28, ny=28, nz=12, dx=3000., dy=3000., ztop=16000.,
                dt=3., run_seconds=30.)
exp = experiment_from_run_config(cfg, datetime(2000, 6, 1, 12))
child = replace(exp.root, grid_id=2, parent_id=1, parent_grid_ratio=3,
                i_parent_start=8, j_parent_start=8,
                run=replace(cfg, nx=16, ny=16, dx=1000., dy=1000., dt=1.,
                            ra_lw_physics=4, ra_sw_physics=4,
                            ra_rrtmg_variant="rrtmg_legacy", o3input=2))
assert cam_ozone_domain_ids(replace(exp, domains=(exp.root, child))) == {1, 2}
assert metgrid_number_targets(replace(cfg, moist=True, mp_physics=28)) == {
    'QNI': 'ni', 'QNC': 'nc', 'QNR': 'nr', 'QNWFA': 'nwfa', 'QNIFA': 'nifa'}
raw = {'experiment': {'name': 'prepared'}, 'domain': [{'grid_id': 1}],
       'static': {'highres': {'path': "Ana's terrain", 'enabled': True}}}
assert tomllib.loads(render_experiment_document(raw)) == raw
assert 'mapped' in mapped_sources()
assert source_schemas()['mapped'] == 'gpuwm-mapped-composition-inputs-v1'
assert 'gpuwm.core.preflight' not in sys.modules
assert 'gpuwm.core.cam_ozone' not in sys.modules
assert 'gpuwm.branch' not in sys.modules
assert 'gpuwm.prepared_single_domain_forecast' not in sys.modules

# A staged config reader must retain attribute-following validation, including
# its refusal, while the runtime UI and executor imports remain blocked.
follow = build_follow_config({
    'field': 'attribute', 'attribute': 'theta', 'extremum': 'max',
    'reduction': 'column_max', 'threshold': 301., 'search_margin_cells': 10,
    'min_shift_cells': 1, 'max_shift_cells': 6, 'cooldown_seconds': 20.,
}, 'standalone-config')
validate_attribute_domains(
    [replace(exp.root, follow=None),
     replace(child, follow=SimpleNamespace(tracker=follow))],
    SimpleNamespace(follow=None))
try:
    validate_attribute_domains(
        [replace(exp.root, follow=None),
         replace(child, follow=SimpleNamespace(tracker=replace(follow,
             reduction='model_level', model_level=cfg.nz)))],
        SimpleNamespace(follow=None))
except ValueError as error:
    assert 'outside source d01 mass levels' in str(error)
else:
    raise AssertionError('staged attribute validation accepted a missing level')

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
    gpuwm.twentycrv3_direct,
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
    "gpuwm.remote_cli", "gpuwm.remote_worker", "gpuwm.research_workspaces",
    "gpuwm.starter_template", "gpuwm.tui_products",
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


def test_standalone_auto_backend_reads_the_card_load(tmp_path):
    """The staged package prices its preparation against the card's load.

    Auto's load probe lived in the forecast memory preflight, which this
    package does not carry, so the package kept a certified card another
    program had nearly filled or held busy, where gpuwm prepared on the
    CPU.  The probe here is the real one, run as its own subprocess from
    the staged tree against a stand-in CuPy that reports the card's
    memory; CUDA_VISIBLE_DEVICES=-1 keeps the machine's own nvidia-smi
    out of it.  Whether the preparation fits is its price against that
    free memory (A65), and the price is computed in the staged tree from
    a mapped preparation's decoded inventory: while the price read its
    inventories from the forecast preflight, this import raised
    ModuleNotFoundError here and stopped every preparation.  The fitting
    half also holds the older fix: a probe that cannot be imported must
    not read as a CuPy that could not load.
    """

    staged = tmp_path / "rw-wps-python"
    _stage_or_skip(staged)
    shadow = tmp_path / "shadow"
    (shadow / "cupy").mkdir(parents=True)
    (shadow / "cupy" / "__init__.py").write_text(
        "import os\n"
        "\n"
        "GIB = 1024 ** 3\n"
        "\n"
        "\n"
        "class _Runtime:\n"
        "    @staticmethod\n"
        "    def memGetInfo():\n"
        "        return int(os.environ['SHADOW_FREE_GIB']) * GIB, 32 * GIB\n"
        "\n"
        "    @staticmethod\n"
        "    def getDeviceProperties(device):\n"
        "        return {'name': 'shadow card', 'multiProcessorCount': 64,\n"
        "                'maxThreadsPerMultiProcessor': 1536}\n"
        "\n"
        "    @staticmethod\n"
        "    def deviceGetLimit(limit):\n"
        "        return 1024\n"
        "\n"
        "\n"
        "class cuda:\n"
        "    runtime = _Runtime\n",
        encoding="utf-8")
    script = r"""
from pathlib import Path
from types import SimpleNamespace
import os

GIB = 1024 ** 3
root = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
assert not (root / "gpuwm" / "core" / "preflight.py").exists()
import gpuwm.ingest.preprocess_backend as backend
assert Path(backend.__file__).resolve().is_relative_to(root)

runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                          runtimeGetVersion=lambda: 13020)
card = SimpleNamespace(name="cuda", array_module=SimpleNamespace(
    __version__="14.2.0", cuda=SimpleNamespace(runtime=runtime)))
cpu = SimpleNamespace(name="cpu")
backend.CudaPreprocessBackend = lambda: card
backend.ParallelCpuPreprocessBackend = lambda **_: cpu
backend._gpu_runtime_installed = lambda: True

from gpuwm.config import RunConfig
from gpuwm.ingest.preparation_price import price_forcing_preparation

run = RunConfig(nx=896, ny=512, nz=59, dx=6000.0, dy=6000.0, ztop=20000.0,
                dt=30.0, run_seconds=21600.0, terrain_opt=1, moist=True,
                mp_physics=8, km_opt=4, bl_pbl_physics=1, specified=True,
                spec_bdy_width=5, sf_sfclay_physics=91, sf_surface_physics=2)
exp = SimpleNamespace(domains=[SimpleNamespace(run=run)])
level = SimpleNamespace(shape=(40, 1059, 1799))
snapshot = SimpleNamespace(fields={
    "TT": level, "UU": level, "VV": level, "RH": level, "GHT": level,
    "PSFC": SimpleNamespace(shape=(1059, 1799))})
price = price_forcing_preparation("mapped", exp, [snapshot] * 7)
# The stand-in card is 32 GiB: 2 GiB free cannot hold this preparation
# and 30 GiB free can.
assert 2 * GIB < price.need_bytes < 30 * GIB, price.need_bytes

os.environ["SHADOW_FREE_GIB"] = "2"
chosen = backend.resolve_preprocess_backend("auto", price=price)
assert chosen is cpu, chosen.selection
reason = chosen.selection["reason"]
assert reason.startswith("the CUDA preparation needs "), reason
assert "the card has 2.0 GiB free of 32.0 GiB" in reason, reason
fit = chosen.selection["device_fit"]
assert fit["fits"] is False, fit
assert fit["need_bytes"] == price.need_bytes, fit
assert fit["free_bytes"] == 2 * GIB, fit
assert fit["route"] == "mapped", fit
load = chosen.selection["device_load"]
assert load["free_bytes"] == 2 * GIB, load
assert load["total_bytes"] == 32 * GIB, load

os.environ["SHADOW_FREE_GIB"] = "30"
chosen = backend.resolve_preprocess_backend("auto", price=price)
assert chosen is card, chosen.selection
assert "certified" in chosen.selection["reason"], chosen.selection
assert chosen.selection["device_load"]["free_bytes"] == 30 * GIB, chosen.selection
fit = chosen.selection["device_fit"]
assert fit["fits"] is True, fit
assert fit["need_bytes"] == price.need_bytes, fit
assert fit["free_bytes"] == 30 * GIB, fit

import gpuwm.core.device_probe as probe
assert Path(probe.__file__).resolve().is_relative_to(root)
"""
    environment = os.environ.copy()
    environment.pop("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", None)
    environment.pop("GPUWM_NO_LOCAL_GPU", None)
    environment.pop("PYTHONSAFEPATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["PYTHONPATH"] = os.pathsep.join((str(shadow), str(staged)))
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
    assert "could not be loaded" not in completed.stderr


def test_standalone_preparation_publishes_at_its_seal(tmp_path):
    """The staged chained writer admits a CUDA producer without a forecast.

    The package stages the chained writer for its era5, gfs and mapped
    routes but not the forecast, so a CUDA preparation died at its
    admission on the missing gpuwm.core.preflight.  Run against the staged
    tree, with chaining asked for, the writer declines it by name and
    prices nothing.
    """

    staged = tmp_path / "rw-wps-python"
    _stage_or_skip(staged)
    script = r"""
from pathlib import Path
import os
import sys

root = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
from gpuwm.ingest import boundary_stream
assert Path(boundary_stream.__file__).resolve().is_relative_to(root)
assert not boundary_stream.forecast_installed()
staging = Path(os.environ["RW_WPS_WORK"]) / ".tmp-tree"
staging.mkdir()
writer = boundary_stream.PreparedTreeWriter(
    staging=staging, output_root=staging.parent / "tree", identity={})
expected = {"chained": False,
            "reason": boundary_stream.PREPARATION_ONLY_REASON}
assert writer.chained is False
for backend in ("cuda", "cpu"):
    decision = writer.admit(experiment=object(), backend=backend,
                            device_bytes=1 << 30, card=(1 << 40, 0))
    assert decision == expected, decision
assert "gpuwm.core.preflight" not in sys.modules
"""
    environment = os.environ.copy()
    environment["GPUWM_CHAINED_PREP"] = "1"
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    environment["RW_WPS_WORK"] = str(tmp_path)
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

    assert _staged_internal_imports(
            tmp_path / "staged", source_root=ROOT) == [{
        "path": "gpuwm/direct.py",
        "line": 1,
        "module": "gpuwm.io.restart",
        "kind": "from",
    }]


def test_standalone_package_boundary_scan_reads_from_package_import_module(
        tmp_path):
    """``from gpuwm.core import pace`` imports gpuwm.core.pace.

    The scan read it as an import of gpuwm.core, which is staged, so an
    unstaged submodule imported that way passed and the staged package
    raised ImportError at the call (A126: host_libm, and the staged map
    projections with it).  A name the source tree has
    no module for is an attribute and is not flagged; a staged submodule
    passes; a missing package is reported once, as itself.
    """

    staged = tmp_path / "staged"
    core = staged / "gpuwm" / "core"
    core.mkdir(parents=True)
    for package in (staged / "gpuwm", core):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (core / "constants.py").write_text("", encoding="utf-8")
    (core / "user.py").write_text(
        "from gpuwm.core import constants, pace\n"
        "from . import pace as relative_pace\n"
        "from gpuwm.core import not_a_module_anywhere\n"
        "from gpuwm.io import restart\n",
        encoding="utf-8",
    )
    assert (ROOT / "gpuwm" / "core" / "pace.py").is_file()
    assert (ROOT / "gpuwm" / "io" / "restart.py").is_file()

    assert _staged_internal_imports(staged, source_root=ROOT) == [
        {"path": "gpuwm/core/user.py", "line": 1,
         "module": "gpuwm.core.pace", "kind": "from"},
        {"path": "gpuwm/core/user.py", "line": 2,
         "module": "gpuwm.core.pace", "kind": "from"},
        {"path": "gpuwm/core/user.py", "line": 4,
         "module": "gpuwm.io", "kind": "from"},
    ]


def test_standalone_staging_refuses_an_unstaged_module_imported_from_its_package(
        tmp_path, monkeypatch):
    """The real staging refuses ``from gpuwm.core import pace``.

    gpuwm/core/streaming.py imports the pace model that way, and pace is
    not staged.  With its written reason removed from
    _OPTIONAL_STAGED_IMPORTS the staging refuses by name, which it did
    not do while the scan read the import as one of gpuwm.core.
    """

    import tools.build_rw_wps_release as release

    key = ("gpuwm/core/streaming.py", "gpuwm.core.pace")
    assert key in release._OPTIONAL_STAGED_IMPORTS
    monkeypatch.setattr(release, "_OPTIONAL_STAGED_IMPORTS", {
        k: v for k, v in release._OPTIONAL_STAGED_IMPORTS.items() if k != key})
    with pytest.raises(RuntimeError) as refused:
        _stage_or_skip(tmp_path / "rw-wps-python")
    message = str(refused.value)
    assert "unresolved internal imports" in message
    assert "'module': 'gpuwm.core.pace'" in message
    assert "'path': 'gpuwm/core/streaming.py'" in message


def test_standalone_scan_reads_from_gpuwm_import_verify(tmp_path):
    package = tmp_path / "staged" / "gpuwm"
    package.mkdir(parents=True)
    (package / "direct.py").write_text(
        "from gpuwm import verify\n", encoding="utf-8")

    assert _staged_verification_imports(tmp_path / "staged") == [{
        "path": "gpuwm/direct.py",
        "line": 1,
        "module": "gpuwm.verify",
        "kind": "from",
    }]


def _finish_a_staged_preparation(tmp_path, preamble=""):
    """Run a mapped preparation to its handoff through the staged door.

    A stand-in adapter child writes a finished bundle; ``preamble`` runs
    in the child interpreter before gpuwm is imported.
    """

    from test_source_adapters import _mapped_args

    staged = tmp_path / "rw-wps-python"
    _stage_or_skip(staged)
    root = tmp_path / "prepared"
    argv = _mapped_args()
    argv[argv.index("--output-root") + 1] = str(root)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json, pathlib, sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "root.mkdir()\n"
        "(root / 'proof.json').write_text(json.dumps("
        "{'schema': 'gpuwm-mapped-composition-proof-v1', 'status': 'READY'}))\n"
        "(root / 'experiment.toml').write_text('[experiment]\\nname = \"x\"\\n')\n"
        "(root / 'namelist.wps').write_text('&share\\n/\\n')\n",
        encoding="utf-8",
    )
    script = preamble + r"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

staged = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
import gpuwm.source_cli as source_cli
assert Path(source_cli.__file__).resolve().is_relative_to(staged)
source_cli._mapped_command = lambda args: [
    sys.executable, os.environ["RW_WPS_ADAPTER"], str(args.output_root)]
code = source_cli.main(json.loads(os.environ["RW_WPS_ARGV"]))
assert code == 0, code
from gpuwm.ingest.boundary_stream import prepared_head_urban_columns
urban = SimpleNamespace(domains=(SimpleNamespace(
    run=SimpleNamespace(sf_urban_physics=3)),))
assert prepared_head_urban_columns(urban, {}) is None
assert "gpuwm.core.urban_state" not in sys.modules
for name in ("gpuwm.prepared_single_domain_forecast",
             "gpuwm.prepared_domain_tree_forecast"):
    assert name not in sys.modules, name
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    environment["RW_WPS_ADAPTER"] = str(adapter)
    environment["RW_WPS_ARGV"] = json.dumps(argv)
    completed = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr, completed.stderr
    assert "prep: complete" in completed.stdout
    assert "this installation prepares inputs only" in completed.stdout
    assert "gpuwm.prepared_domain_tree_forecast" in completed.stdout
    assert f"The prepared tree is {root}" in completed.stdout
    assert "Run the forecast" not in completed.stdout
    assert (root / "proof.json").is_file()
    return completed


def test_standalone_preparation_finishes_without_the_forecast_runners(tmp_path):
    """A finished preparation hands off without a forecast to hand to.

    The package stages prep_output and stage_cli but neither forecast
    runner, and the handoff resolved the bundle through the runners'
    schema tables: every finished `rw-wps` preparation ended in
    "ImportError: cannot import name 'prepared_domain_tree_forecast'"
    right after "prep: complete".  Run through the staged rw-wps door,
    with a stand-in adapter child that writes a finished bundle.
    """

    _finish_a_staged_preparation(tmp_path)


#: The finder an editable gpuwm install (``pip install -e .``, as the
#: publish workflow's test job installs it) appends to ``sys.meta_path``,
#: in setuptools' own shape: the top-level package maps to the checkout,
#: and an immediate child the path finders miss resolves there.
_EDITABLE_GPUWM_FINDER = r"""
import importlib.util as _util
import os as _os
from importlib.machinery import PathFinder as _PathFinder
from pathlib import Path as _Path
import sys as _sys


class _EditableGpuwmFinder:
    MAPPING = {"gpuwm": str(_Path(_os.environ["RW_WPS_CHECKOUT"]) / "gpuwm")}

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        parent = fullname.rpartition(".")[0]
        if parent in cls.MAPPING:
            return _PathFinder.find_spec(fullname, path=[cls.MAPPING[parent]])
        return None


_sys.meta_path.append(_EditableGpuwmFinder)
_spec = _util.find_spec("gpuwm.prepared_domain_tree_forecast")
print("CHECKOUT RUNNER", _spec.origin)
"""


def test_standalone_preparation_ignores_a_runner_another_tree_provides(
        tmp_path, monkeypatch):
    """A runner only another gpuwm tree has is a runner this one lacks.

    With gpuwm installed editable (publish.yml's test job) and the staged
    package first on the path, ``importlib.util.find_spec`` found the
    checkout's tree runner through the editable finder, so the handoff
    resolved the bundle and importing that runner against the staged
    gpuwm.core failed ("No module named 'gpuwm.core.adaptive_clock'").
    The finder here is setuptools' shape; the handoff must end on the
    preparation-only line and import neither runner.
    """

    checkout_runner = ROOT / "gpuwm" / "prepared_domain_tree_forecast.py"
    assert checkout_runner.is_file()
    monkeypatch.setenv("RW_WPS_CHECKOUT", str(ROOT))
    completed = _finish_a_staged_preparation(
        tmp_path, preamble=_EDITABLE_GPUWM_FINDER)
    first = completed.stdout.splitlines()[0]
    # The finder is in effect: it hands the checkout's runner to find_spec.
    assert first.startswith("CHECKOUT RUNNER "), completed.stdout
    assert Path(first.removeprefix("CHECKOUT RUNNER ")).resolve() == (
        checkout_runner.resolve())


def test_the_standalone_package_reports_its_own_distribution_version(tmp_path):
    """A clean rw-wps install knows its version.

    gpuwm/__init__.py read the ``gpuwm`` distribution alone, and a clean
    install of the standalone package carries only ``rw-wps``: it
    reported 0+unknown, and its installed runtime check refused every
    clean install ("rw-wps version mismatch: metadata=2.8.1,
    module=0+unknown").  Run in the staged tree with only an rw-wps
    distribution on the path (``-S``: no site-packages).
    """

    import gpuwm

    assert PYTHON_DISTRIBUTION in gpuwm.DISTRIBUTION_NAMES
    assert gpuwm.DISTRIBUTION_NAMES[0] == gpuwm.DISTRIBUTION_NAME
    staged = tmp_path / "rw-wps-python"
    _stage_or_skip(staged)
    site = tmp_path / "site"
    info = site / "rw_wps-9.8.7.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rw-wps\nVersion: 9.8.7\n",
        encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(staged), str(site)))
    completed = subprocess.run(
        [sys.executable, "-S", "-P", "-c",
         "import gpuwm; print(gpuwm.__version__); print(gpuwm.__file__)"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    version, location = completed.stdout.splitlines()
    assert version == "9.8.7"
    assert Path(location).resolve().is_relative_to(staged.resolve())


def test_a_staged_package_reads_its_own_version_beside_an_installed_gpuwm(
        tmp_path):
    """The staged package's runtime check passes beside another gpuwm.

    install_gpuwm_native_wrf.sh installs the rw-wps wheel with ``pip
    install --target`` and runs its runtime check with only that target
    on PYTHONPATH, so an interpreter whose site-packages also carries
    gpuwm (a venv, or a --user install) put a second gpuwm distribution
    on the path.  gpuwm/__init__.py read the first distribution NAMED
    gpuwm, reported that install's number, and the check refused ("rw-wps
    version mismatch: metadata=9.8.7, module=1.2.3").  The staged tree
    here carries the rw-wps .dist-info pip --target writes beside it and a
    later site directory carries an installed gpuwm, package and all.
    """

    staged = tmp_path / "rw-wps-python"
    _stage_or_skip(staged)
    info = staged / "rw_wps-9.8.7.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rw-wps\nVersion: 9.8.7\n",
        encoding="utf-8")
    (info / "RECORD").write_text(
        _record_row("gpuwm/__init__.py", staged / "gpuwm" / "__init__.py")
        + f"{info.name}/METADATA,,\n{info.name}/RECORD,,\n",
        encoding="utf-8")
    site = tmp_path / "site-packages"
    (site / "gpuwm").mkdir(parents=True)
    other = site / "gpuwm" / "__init__.py"
    other.write_bytes(b"__version__ = '1.2.3'\n")
    other_info = site / "gpuwm-1.2.3.dist-info"
    other_info.mkdir()
    (other_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: gpuwm\nVersion: 1.2.3\n",
        encoding="utf-8")
    (other_info / "RECORD").write_text(
        _record_row("gpuwm/__init__.py", other)
        + f"{other_info.name}/METADATA,,\n{other_info.name}/RECORD,,\n",
        encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(staged), str(site)))
    program = (
        "import json, gpuwm;"
        "from gpuwm.native_wrf_distribution import _installed_record_receipt;"
        "print(json.dumps({'version': gpuwm.__version__,"
        " 'file': gpuwm.__file__,"
        " 'receipt': _installed_record_receipt()}))")
    completed = subprocess.run(
        [sys.executable, "-S", "-P", "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "version mismatch" not in completed.stderr, completed.stderr
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert Path(payload["file"]).resolve().is_relative_to(staged.resolve())
    assert payload["version"] == "9.8.7"
    assert payload["receipt"]["distribution_name"] == PYTHON_DISTRIBUTION
    assert payload["receipt"]["distribution_version"] == "9.8.7"
    assert payload["receipt"]["record_file_count"] == 1


def _record_row(name, path):
    import base64

    digest = base64.urlsafe_b64encode(
        hashlib.sha256(path.read_bytes()).digest()).decode().rstrip("=")
    return f"{name},sha256={digest},{path.stat().st_size}\n"


@pytest.mark.parametrize("layout", ["target", "home"])
def test_the_installed_record_check_finds_the_console_scripts_pip_wrote(
        tmp_path, monkeypatch, layout):
    """The runtime check verifies every hashed RECORD row, scripts too.

    install.sh installs the wheel with ``pip install --target``, whose
    RECORD names the console scripts ``../../bin/<name>`` (relative to
    the temporary home's lib/python) while pip moves them to
    ``<target>/bin``.  Python 3.11 lists those rows and the check looked
    two levels above the target, so every clean install there failed
    ("installed wheel file is missing"); from Python 3.12 dist.files
    drops rows missing at the literal path, so a deleted file passed.
    ``home`` is ``pip install --home``, where the scripts do stay there.
    """

    import importlib.metadata

    import gpuwm.native_wrf_distribution as distribution

    if layout == "target":
        root = tmp_path / "runtime"
        scripts = root / "bin"
    else:
        root = tmp_path / "home" / "lib" / "python"
        scripts = tmp_path / "home" / "bin"
    info = root / f"rw_wps-{distribution.__version__}.dist-info"
    info.mkdir(parents=True)
    scripts.mkdir(parents=True)
    (root / "gpuwm").mkdir()
    module = root / "gpuwm" / "entry.py"
    module.write_bytes(b"def main():\n    return 0\n")
    script = scripts / "rw-wps"
    script.write_bytes(b"#!/usr/bin/python3\nfrom gpuwm.entry import main\n")
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: rw-wps\n"
        f"Version: {distribution.__version__}\n", encoding="utf-8")
    (info / "RECORD").write_text(
        _record_row("gpuwm/entry.py", module)
        + _record_row("../../bin/rw-wps", script)
        + "gpuwm/__pycache__/entry.cpython-311.pyc,,\n"
        + f"{info.name}/RECORD,,\n", encoding="utf-8")
    installed = importlib.metadata.PathDistribution(info)
    monkeypatch.setattr(distribution, "metadata", SimpleNamespace(
        distribution=lambda name: installed))

    receipt = distribution._installed_record_receipt()
    assert receipt["record_file_count"] == 2

    script.write_bytes(b"#!/usr/bin/python3\nraise SystemExit(1)\n")
    with pytest.raises(RuntimeError, match="RECORD mismatch for ../../bin/rw-wps"):
        distribution._installed_record_receipt()
    script.unlink()
    with pytest.raises(FileNotFoundError, match="installed wheel file is missing"):
        distribution._installed_record_receipt()


def test_standalone_radar_grid_writer_is_staged(tmp_path):
    """gpuwm.obs.write_radar_grid writes, or refuses by name.

    The staged gpuwm.obs exports the radar-grid writer, whose default
    engine opens gpuwm.io.classic_product; the package did not stage it,
    so the call raised ImportError.  Where the Rust NetCDF writer loads,
    the staged package writes a product and, where the Rust reader also
    loads, reads it back; where the writer does not load, the refusal is
    the named one, not an ImportError.
    """

    staged = tmp_path / "rw-wps-python"
    _stage_or_skip(staged)
    script = r"""
import os
from pathlib import Path
import sys

staged = Path(os.environ["RW_WPS_STAGED_ROOT"]).resolve()
sys.path.append(os.environ["RW_WPS_TESTS"])
import test_obs_radar_grid as helpers
import gpuwm.io.classic_product
import gpuwm.obs
from gpuwm.io import nc_writer_bridge
from gpuwm.obs.grid_product import ObsGridWriterUnavailable
for module in (gpuwm.io.classic_product, gpuwm.obs):
    assert Path(module.__file__).resolve().is_relative_to(staged), module

grid = helpers._grid(nx=21, ny=21)
volume = helpers._volume(grid, reflectivity=[5.0, 25.0, 45.0],
                         velocity=[-10.0, 0.0, 10.0])
observations, params = helpers._gridded(grid, volume)
path = Path(os.environ["RW_WPS_WORK"]) / "radar-grid.nc"
if nc_writer_bridge.unavailable_reason() is None:
    receipt = gpuwm.obs.write_radar_grid(
        path, observations, grid, valid_time="2026-07-28T20:03:16Z",
        params=params)
    assert receipt["schema"] == gpuwm.obs.RADAR_GRID_SCHEMA
    assert path.stat().st_size == receipt["bytes"]
    # The read-back needs the Rust NetCDF reader, a separate native: a
    # bridge folder can hold the writer without it, so it is checked only
    # where the reader resolves.
    from gpuwm import netcdf_bridge
    if netcdf_bridge.find_netcdf_bin() is not None:
        read = gpuwm.obs.read_radar_grid(
            path, expected_grid_identity=grid.identity_sha256())
        assert read["schema"] == receipt["schema"]
        assert read["dims"] == receipt["dims"]
    print("WROTE", receipt["bytes"])
else:
    try:
        gpuwm.obs.write_radar_grid(
            path, observations, grid, valid_time="2026-07-28T20:03:16Z",
            params=params)
    except ObsGridWriterUnavailable as error:
        print("REFUSED", error)
    else:
        raise AssertionError("wrote without a loadable Rust writer")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(staged)
    environment["RW_WPS_STAGED_ROOT"] = str(staged)
    environment["RW_WPS_TESTS"] = str(ROOT / "tests")
    environment["RW_WPS_WORK"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-P", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    # Without the Rust static-fields library (publish.yml's test job
    # builds no natives) the grid helper's projection prints its
    # "[static] WORKAROUND" line first, so the outcome is found by line.
    outcomes = [line for line in completed.stdout.splitlines()
                if line.startswith(("WROTE ", "REFUSED "))]
    assert len(outcomes) == 1, completed.stdout


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


# ---------------------------------------------------------------------------
# The standalone rw-wps bundle's declaration
# ---------------------------------------------------------------------------


def test_the_standalone_bundle_carries_every_bridge_a_shipped_source_names():
    """A route that names a binary must be a route the bundle can run.

    THE defect this file is here to stop from recurring.  2.7.5 added the
    icosahedral remapper ``gdt101_remap`` to the bridge bundle for ICON
    global, and the standalone rw-wps bundle did not grow it, because its
    contents were a hand-kept tuple with no tie to the routes that need
    them.  It shipped as a known limit: ``gpuwm doctor`` reported
    ``MISSING bridge gdt101_remap: not staged`` on an install carrying
    exactly what the bundle declares, and ``rw-wps --source icon-global``
    refused rather than falling back.

    The binding is to the packaged authorities, not to a literal, so it
    is the arbitrary test and not a patch: a producer on a new native
    grid arrives as a normalization document naming the binary that
    reads it, and this fails the moment that binary is outside the
    bundle -- before a release can ship a source the bundle cannot run.
    """

    from gpuwm import source_authorities, source_normalization

    named = {
        source_normalization.load_normalization(name).bridge
        for name in source_authorities.packaged_normalizer_ids()
    }
    assert named, "no packaged source declares an input-normalization binary"
    assert "gdt101_remap" in named
    assert named <= set(BRIDGE_NAMES), (
        "the standalone rw-wps bundle does not carry "
        f"{sorted(named - set(BRIDGE_NAMES))}, which a packaged source's "
        "input normalization resolves and refuses without")


def test_every_declared_bridge_is_one_row_and_nothing_is_written_per_bridge():
    """One row is the whole declaration: name, workspace, option, env, probe.

    The five-places shape this replaces lost two bridges.  These are the
    surfaces a row must reach, checked together so none of them can go
    stale on its own again.
    """

    from gpuwm.bridges import BRIDGE_ENV
    from gpuwm.rustwx_fetch import FETCH_ENV, FETCH_NAME

    # The row's environment variable must be the one the module that
    # RESOLVES that bridge actually reads.  Both launchers are checked
    # against the row below, so a row holding its own copy of the name
    # would let the variable move in the resolver while the row, the
    # launchers and every test stayed green and agreed with each other
    # about a variable nothing reads.
    resolver_env = {**BRIDGE_ENV, FETCH_NAME: FETCH_ENV}
    assert BRIDGE_NAMES == tuple(b.name for b in BUNDLED_BRIDGES)
    assert len(set(BRIDGE_NAMES)) == len(BRIDGE_NAMES)
    assert BRIDGE_WORKSPACES == ("tools/grib1_bridge", "tools/rustwx")
    for bridge in BUNDLED_BRIDGES:
        assert bridge.name in resolver_env, (
            f"{bridge.name} is declared bundled but no module resolves it "
            "from an environment variable, so the launchers would bind a "
            "name nothing reads")
        assert bridge.env_var == resolver_env[bridge.name], (
            f"{bridge.name}'s row binds {bridge.env_var} and its resolver "
            f"reads {resolver_env[bridge.name]}")
        assert bridge.workspace in BRIDGE_WORKSPACES
        assert bridge.option.startswith("--")
        assert bridge.dest == bridge.option[2:].replace("-", "_")
        assert bridge.env_var.startswith("GPUWM_")
        assert bridge.usage_marker == f"usage: {bridge.name}"
        assert _BRIDGE_USAGE_MARKERS[bridge.name] == bridge.usage_marker
        assert bridge.consumer.strip() == bridge.consumer and bridge.consumer

    remapper, = [b for b in BUNDLED_BRIDGES if b.name == "gdt101_remap"]
    assert remapper.workspace == "tools/grib1_bridge"
    assert remapper.env_var == "GPUWM_GDT101_REMAP"
    assert _BRIDGE_ABI_MARKERS["gdt101_remap"] \
        == b"arwen.gdt101-regional-remap.v1"


def test_the_declared_bridges_are_the_ones_both_builders_take_and_stage():
    """The builders' option surface is generated, so it cannot drift.

    ``tools/build_rw_wps_release.py`` handed the packager a namespace
    that never named ``rw_fetch``; the packager read ``args.rw_fetch``
    and died with an ``AttributeError`` after the wheel and the Rust had
    already been built.  Both builders now take their options from the
    table, and a namespace missing a row is refused by name.
    """

    import argparse

    parser = argparse.ArgumentParser()
    add_bridge_options(parser)
    parsed = parser.parse_args([
        argument
        for bridge in BUNDLED_BRIDGES
        for argument in (bridge.option, f"/built/{bridge.name}")
    ])
    resolved = bridge_inputs(parsed)
    assert list(resolved) == list(BRIDGE_NAMES)
    assert resolved["gdt101_remap"].name == "gdt101_remap"

    short = argparse.Namespace(**{
        bridge.dest: Path(f"/built/{bridge.name}")
        for bridge in BUNDLED_BRIDGES if bridge.name != "gdt101_remap"
    })
    with pytest.raises(ValueError, match="gdt101_remap"):
        bridge_inputs(short)


@pytest.mark.parametrize(
    ("script", "staging", "cpu_backend"),
    (
        ("build_rw_wps_release.py",
         "**{bridge.dest: native / bridge.name",
         "native / CPU_BACKEND_LIBRARY"),
        ("build_rw_wps_windows_release.py",
         '**{bridge.dest: native / f"{bridge.name}.exe"',
         "native / WINDOWS_CPU_BACKEND_LIBRARY"),
    ),
)
def test_the_release_builder_compiles_every_workspace_the_table_names(
        script, staging, cpu_backend):
    """rw_fetch builds in the renderer workspace, not the decoder one.

    The shipped builders compiled ``tools/grib1_bridge`` only, so even a
    namespace that had named ``rw_fetch`` would have pointed at a file
    cargo never produced.  The build plan is the table's workspace
    column, read at build time.

    BOTH release builders, because there are two and only one of them
    was mended first: the Linux command was read off the table while
    ``tools/build_rw_wps_windows_release.py`` kept its single manifest
    and its five hand-written literals, and the new refusal in
    :func:`bridge_inputs` then stopped the Windows archive from being
    cut at all.  A guard that reads one of two files bans the shape in
    one of two files.
    """

    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / script).read_text(encoding="utf-8")
    assert "for workspace in BRIDGE_WORKSPACES:" in source
    assert staging in source
    assert cpu_backend in source
    for literal in ("grib1_bridge=native /", "gfs_bridge=native /",
                    "hrrr_bridge=native /", "grib2_dump=native /"):
        assert literal not in source, (
            f"a per-bridge literal is back in tools/{script}; that is "
            "the shape that dropped rw_fetch and then the remapper")
    for literal in ('"grib1_bridge" / "Cargo.toml"',
                    '"gpuwm_preprocess_cpu.dll"',
                    '"libgpuwm_preprocess_cpu.so"'):
        assert literal not in source, (
            f"tools/{script} hard-codes {literal} instead of reading the "
            "declaration, which is how one workspace and five bridges "
            "became the whole build plan")
    for workspace in BRIDGE_WORKSPACES:
        manifest = root / workspace / "Cargo.toml"
        assert manifest.is_file(), f"declared workspace has no {manifest}"


def test_both_installed_launchers_bind_every_declared_bridge():
    """The bundle's Python lives under ``runtime/``, so the ladder needs these.

    ``<root>/libexec/bridges`` is resolved beside the PACKAGE, which in
    an installed bundle is ``<root>/runtime``; the launcher's explicit
    environment binding is what actually reaches ``<root>/libexec``.
    Both launchers bound five of the six bridges before this, so a
    ``rw-wps`` run resolved the fetch backbone through some other rung
    or not at all.
    """

    tools = Path(__file__).resolve().parents[1] / "tools"
    posix = (tools / "gpuwm_native_wrf_launcher.sh").read_text(encoding="utf-8")
    windows = (tools / "gpuwm_native_wrf_launcher_windows.ps1").read_text(
        encoding="utf-8")
    for bridge in BUNDLED_BRIDGES:
        assert f'export {bridge.env_var}="$root/libexec/bridges/' \
            f'{bridge.name}"' in posix, bridge.name
        separator = chr(92)
        assert (f'$env:{bridge.env_var} = Join-Path $Root '
                f'"libexec{separator}bridges{separator}'
                f'{bridge.name}.exe"') in windows, bridge.name
