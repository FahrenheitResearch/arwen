#!/usr/bin/env python3
"""Build the complete Linux RW-WPS runtime from one clean source checkout.

The lower-level distribution builder deliberately accepts already-built
artifacts so it can verify each one independently.  This command is the
end-user entry point: it builds the Python wheel and every vendored Rust
decoder/CPU backend offline, then delegates to that verifier and packager.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.build_native_wrf_distribution import build_distribution  # noqa: E402
from gpuwm.native_wrf_distribution import (  # noqa: E402
    CUDA_KERNEL_SOURCES,
    HRRR_HELPERS,
    PYTHON_DISTRIBUTION,
)


_TOP_LEVEL_EXCLUDES = {
    "cli.py",
    # gpuwm-product front doors (CUDA forecast side): the domain wizard
    # imports the memory preflight + gpuwm.cli, and the downscale front
    # door drives the offline CUDA child.  Neither belongs in the
    # standalone RW-WPS preprocessing wheel.
    "domain_wizard.py",
    # The wizard's prompt session, for the same reason and one more: it
    # reaches gpuwm.domain_wizard and gpuwm.core.preflight, both absent
    # here, so staging it fails this builder's own unresolved-import
    # scan.  A preprocessing wheel has no domain to size.
    "domain_interactive.py",
    # `gpuwm go` drives tools/prepared_single_domain_forecast.py through
    # the GPU forecast.  Its imports happen to resolve against what RW-WPS
    # stages, so the scan would not have caught it -- but the runner it
    # exists to call is not in this wheel, so shipping it would offer a
    # command that cannot run.
    "go_cli.py",
    # The two prepared-cache GPU forecast runners.  Their substance
    # moved out of tools/ and into the package so a `pip install gpuwm`
    # can finish a forecast; that makes them top-level modules here, and
    # they are exactly what this preprocessing wheel does not do.  They
    # reach gpuwm.core.model and the whole CUDA side, which RW-WPS does
    # not stage.
    "prepared_single_domain_forecast.py",
    "prepared_domain_tree_forecast.py",
    "downscale.py",
    "offline_child.py",
    "offline_child_run.py",
    "offline_child_smoke.py",
    # `gpuwm resume` locates a forecast checkpoint and hands it to the
    # supervisor's `run --restart` dispatch.  It reached this staging
    # only because it is a top-level module: nothing RW-WPS ships
    # imports it (the one importer is the excluded gpuwm.cli), no
    # standalone entry point exposes it, and its own two lookups are the
    # restart validator in gpuwm.supervisor and the header reader in
    # gpuwm.io.restart -- one forbidden here, the other not staged at
    # all.  A preprocessing wheel has no checkpoints to resume from.
    "resume.py",
    "runtime.py",
    "state_digest.py",
    "supervisor.py",
}
_CORE_MODULES = {
    "__init__.py",
    "constants.py",
    "diagnostics.py",
    "grid.py",
    "landuse.py",
    "microphysics_transition.py",
    "nest_interp.py",
    "nssl2_contract.py",
    "noah.py",
    "state.py",
    "thompson_contract.py",
}
_INGEST_EXCLUDES = {"preflight.py"}
_ROOT_DATA = {
    "native_wrf_support_v1.json",
    "physics_registry_v2.json",
    "wrf_direct_v461_contract.json",
}
_TOOL_FILES = {"__init__.py", *HRRR_HELPERS}
_FORBIDDEN_STAGED_FILES = {
    "gpuwm/cli.py",
    "gpuwm/domain_wizard.py",
    "gpuwm/domain_interactive.py",
    "gpuwm/go_cli.py",
    "gpuwm/prepared_single_domain_forecast.py",
    "gpuwm/prepared_domain_tree_forecast.py",
    "gpuwm/downscale.py",
    "gpuwm/offline_child.py",
    "gpuwm/offline_child_run.py",
    "gpuwm/offline_child_smoke.py",
    "gpuwm/runtime.py",
    "gpuwm/supervisor.py",
    "gpuwm/core/model.py",
    "gpuwm/core/dycore.py",
    "gpuwm/core/physics.py",
}

_OPTIONAL_STAGED_IMPORTS = {
    ("gpuwm/core/state.py", "gpuwm.core.preflight"):
        "CUDA forecast scratch-preflight path; RW-WPS constructs host state",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.clock"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.dycore"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.health"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.model"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.core.refl"):
        "forecast-only branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.io.wrfout"):
        "forecast-only output branch after the public --prepare-only return",
    ("tools/hrrr_single_domain_benchmark.py", "gpuwm.state_digest"):
        "forecast-only evidence branch after the public --prepare-only return",
    ("gpuwm/ingest/hrrr_physics.py", "gpuwm.core.physics"):
        "forecast-only physics setup after the public --prepare-only return",
}


def _staged_verification_imports(destination: Path) -> list[dict[str, object]]:
    """Return developer-only ``gpuwm.verify`` imports in staged runtime code.

    RW-WPS intentionally excludes the verification package.  Scanning every
    staged Python module closes the package boundary at build time instead of
    relying on a hand-maintained list of public entry points.  Direct imports
    and literal dynamic imports are both rejected.
    """
    violations: list[dict[str, object]] = []
    for source in sorted(destination.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            candidates: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                candidates.extend((alias.name, "import") for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                candidates.append((node.module, "from"))
            elif isinstance(node, ast.Call) and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    function = node.func
                    dynamic = (
                        isinstance(function, ast.Name)
                        and function.id == "__import__"
                    ) or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                    )
                    if dynamic:
                        candidates.append((argument.value, "dynamic"))
            for module, kind in candidates:
                if module == "gpuwm.verify" or module.startswith("gpuwm.verify."):
                    violations.append({
                        "path": source.relative_to(destination).as_posix(),
                        "line": int(getattr(node, "lineno", 0)),
                        "module": module,
                        "kind": kind,
                    })
    return violations


def _staged_internal_imports(
    destination: Path, *, optional: bool = False,
) -> list[dict[str, object]]:
    """Return imports whose internal module is absent from wheel staging."""
    modules: set[str] = set()
    sources: list[tuple[Path, str, bool]] = []
    for source in sorted(destination.rglob("*.py")):
        relative = source.relative_to(destination)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(parts)
        if module:
            modules.add(module)
            sources.append((source, module, is_package))

    violations: list[dict[str, object]] = []
    for source, current_module, is_package in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        type_checking_nodes = set()
        for conditional in ast.walk(tree):
            if not isinstance(conditional, ast.If):
                continue
            marker = conditional.test
            is_type_checking = (
                isinstance(marker, ast.Name) and marker.id == "TYPE_CHECKING"
            ) or (
                isinstance(marker, ast.Attribute)
                and marker.attr == "TYPE_CHECKING"
            )
            if is_type_checking:
                for statement in conditional.body:
                    type_checking_nodes.update(ast.walk(statement))
        current_package = (
            current_module if is_package else current_module.rpartition(".")[0]
        )
        for node in ast.walk(tree):
            if node in type_checking_nodes:
                continue
            candidates: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                candidates.extend((alias.name, "import") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = current_package.split(".") if current_package else []
                    ascend = node.level - 1
                    if ascend > len(package_parts):
                        resolved = ""
                    else:
                        base = package_parts[:len(package_parts) - ascend]
                        if node.module:
                            base.extend(node.module.split("."))
                        resolved = ".".join(base)
                else:
                    resolved = node.module or ""
                if resolved:
                    candidates.append((resolved, "from"))
            elif isinstance(node, ast.Call) and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(
                    argument.value, str
                ):
                    function = node.func
                    dynamic = (
                        isinstance(function, ast.Name)
                        and function.id == "__import__"
                    ) or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                    )
                    if dynamic:
                        candidates.append((argument.value, "dynamic"))
            for module, kind in candidates:
                root = module.partition(".")[0]
                if root not in {"gpuwm", "tools"}:
                    continue
                if module not in modules:
                    relative_path = source.relative_to(destination).as_posix()
                    reason = _OPTIONAL_STAGED_IMPORTS.get(
                        (relative_path, module))
                    if (reason is not None) != optional:
                        continue
                    record = {
                        "path": relative_path,
                        "line": int(getattr(node, "lineno", 0)),
                        "module": module,
                        "kind": kind,
                    }
                    if reason is not None:
                        record["optional_reason"] = reason
                    violations.append(record)
    return violations

#: The token :func:`_standalone_pyproject` replaces with this release.
_STANDALONE_VERSION_PLACEHOLDER = "0.0.0+placeholder"

#: The standalone RW-WPS wheel's own pyproject.  Its version is a
#: placeholder that :func:`_standalone_pyproject` fills in from the
#: running package, because it is not free to be anything else: the
#: sealed-runtime receipt refuses unless the installed ``rw-wps``
#: distribution version equals ``gpuwm.__version__``
#: (``native_wrf_distribution._installed_record_receipt``).  It read
#: ``0.1.1`` and agreed with the stale package constant by coincidence;
#: the moment that constant started telling the truth, a hardcoded
#: version here would have failed the very seal it feeds.
_STANDALONE_PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rw-wps"
version = "0.0.0+placeholder"
description = "Native parallel preprocessing and stock-WRF initialization"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "netCDF4>=1.6"]
keywords = ["WRF", "WPS", "GRIB", "NetCDF", "weather"]
classifiers = ["License :: OSI Approved :: Apache Software License"]

[tool.setuptools]
license-files = ["LICENSE", "NOTICE"]

[project.optional-dependencies]
gpu = ["cupy-cuda12x>=13.0"]
geog = ["rasterio>=1.3", "pyproj>=3.6"]

[project.scripts]
rw-wps = "gpuwm.source_cli:main"
gpuwm-wrf-init = "gpuwm.source_cli:main"
gpuwm-wrf-runtime-check = "gpuwm.native_wrf_distribution:main"
gpuwm-mapped-inspect = "gpuwm.mapped_source:main"

[tool.setuptools.packages.find]
include = ["gpuwm*", "tools"]

[tool.setuptools.package-data]
gpuwm = [
  "native_wrf_support_v1.json",
  "physics_registry_v2.json",
  "wrf_direct_v461_contract.json",
  "authorities/*.json",
  "core/kernels/*.cu",
  "core/kernels/*.cuh",
  "data/noah_tables/*.TBL",
  "data/noah_tables/*.md",
]
tools = ["*.sh"]
"""


def _standalone_pyproject() -> str:
    """The standalone wheel's pyproject, stamped with THIS release.

    A plain substitution rather than ``str.format``: the template is
    TOML and carries braces of its own (``license = { file = ... }``).
    """

    from gpuwm import __version__

    stamped = _STANDALONE_PYPROJECT.replace(
        _STANDALONE_VERSION_PLACEHOLDER, __version__)
    if _STANDALONE_VERSION_PLACEHOLDER in stamped:
        raise RuntimeError(
            "the standalone pyproject template lost its version placeholder")
    return stamped


class NotAGitCheckout(RuntimeError):
    """The tree cannot answer whether a file is tracked.

    Distinct from "this file is untracked", which is a refusal.  A
    snapshot extracted from a tarball or a `git archive` has no index at
    all, so the question has no answer there -- and the two are not the
    same finding.  Conflating them cost a Linux gate run a 40-line
    `CalledProcessError` traceback ending in `exit status 128`, which
    reads as a staging bug rather than as "this directory is not a
    checkout".
    """


def _tree_is_a_git_checkout() -> bool:
    """Does ``REPO`` have a git index to ask?  Asked once, cached."""

    global _GIT_CHECKOUT
    if _GIT_CHECKOUT is None:
        probe = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--git-dir"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _GIT_CHECKOUT = probe.returncode == 0
    return _GIT_CHECKOUT


#: Memo for :func:`_tree_is_a_git_checkout` (None = not yet asked).
_GIT_CHECKOUT: bool | None = None


def _require_tracked(source: Path) -> None:
    relative = source.resolve().relative_to(REPO).as_posix()
    if not _tree_is_a_git_checkout():
        raise NotAGitCheckout(
            f"{REPO} is not a git checkout, so whether {relative} is "
            "tracked cannot be answered.  RW-WPS wheel staging copies "
            "tracked files only; run it from a clone (CI's "
            "actions/checkout leaves one), not from an extracted "
            "archive")
    subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", relative],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _copy_source(source: Path, destination: Path) -> None:
    _require_tracked(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _stage_rw_wps_python_project(destination: Path) -> dict[str, object]:
    """Create the source-matched, forecast-executor-free wheel project."""

    destination.mkdir()
    package = destination / "gpuwm"
    for source in sorted((REPO / "gpuwm").glob("*.py")):
        if source.name not in _TOP_LEVEL_EXCLUDES:
            _copy_source(source, package / source.name)
    for name in sorted(_ROOT_DATA):
        _copy_source(REPO / "gpuwm" / name, package / name)
    for subpackage, excludes in (
        ("ingest", _INGEST_EXCLUDES),
        ("static", set()),
    ):
        for source in sorted((REPO / "gpuwm" / subpackage).glob("*.py")):
            if source.name not in excludes:
                _copy_source(source, package / subpackage / source.name)
    for name in sorted(_CORE_MODULES):
        _copy_source(
            REPO / "gpuwm" / "core" / name,
            package / "core" / name,
        )
    _copy_source(
        REPO / "gpuwm" / "core" / "kernels" / "__init__.py",
        package / "core" / "kernels" / "__init__.py",
    )
    for name in CUDA_KERNEL_SOURCES:
        _copy_source(
            REPO / "gpuwm" / "core" / "kernels" / name,
            package / "core" / "kernels" / name,
        )
    for source in sorted((REPO / "gpuwm" / "authorities").glob("*.json")):
        _copy_source(source, package / "authorities" / source.name)
    for source in sorted((REPO / "gpuwm" / "data" / "noah_tables").iterdir()):
        if source.is_file():
            _copy_source(
                source,
                package / "data" / "noah_tables" / source.name,
            )
    for name in sorted(_TOOL_FILES):
        _copy_source(REPO / "tools" / name, destination / "tools" / name)
    _copy_source(REPO / "README.md", destination / "README.md")
    _copy_source(REPO / "LICENSE", destination / "LICENSE")
    _copy_source(REPO / "NOTICE", destination / "NOTICE")
    (destination / "pyproject.toml").write_text(
        _standalone_pyproject(),
        encoding="utf-8",
        newline="\n",
    )

    files = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*") if path.is_file()
    )
    forbidden = sorted(_FORBIDDEN_STAGED_FILES & set(files))
    if forbidden:
        raise RuntimeError(
            f"RW-WPS wheel staging contains forecast executors: {forbidden}"
        )
    verification_imports = _staged_verification_imports(destination)
    if verification_imports:
        raise RuntimeError(
            "RW-WPS wheel staging imports omitted developer verification "
            f"modules: {verification_imports}"
        )
    internal_imports = _staged_internal_imports(destination)
    if internal_imports:
        raise RuntimeError(
            "RW-WPS wheel staging has unresolved internal imports: "
            f"{internal_imports}"
        )
    optional_internal_imports = _staged_internal_imports(
        destination, optional=True)
    return {
        "distribution": PYTHON_DISTRIBUTION,
        "file_count": len(files),
        "files": files,
        "forecast_executor_files": forbidden,
        "verification_imports": verification_imports,
        "unresolved_internal_imports": internal_imports,
        "optional_internal_imports": optional_internal_imports,
    }


def _run(
    argv: list[str], *, cwd: Path = REPO,
    env: dict[str, str] | None = None,
) -> None:
    # ``main`` reserves stdout for its one machine-readable JSON receipt.
    # pip and Cargo both write progress to stdout, so inheriting their streams
    # produced a file that looked like a receipt but could not be parsed as
    # JSON.  Preserve all build diagnostics on stderr while keeping stdout a
    # strict, single-document interface suitable for atomic capture.
    subprocess.run(
        argv, cwd=cwd, env=env, check=True,
        stdout=sys.stderr, stderr=sys.stderr,
    )


def _cargo_release_environment(
    *, source_root: Path, target_dir: Path, source_date_epoch: str,
) -> dict[str, str]:
    """Return a path-stable environment for the sealed Rust build.

    Rust dependencies can encode absolute source filenames in panic-location
    tables even when application code never prints them.  A release assembled
    from a different clean checkout would then have different bridge hashes.
    Use Cargo's encoded flag channel so checkout paths containing whitespace
    are handled as one rustc argument, and discard caller-provided rust flags
    that would otherwise make the sealed artifact host-dependent.
    """
    environment = os.environ.copy()
    environment.pop("RUSTFLAGS", None)
    rust_flags = [
        f"--remap-path-prefix={source_root.resolve()}=/usr/src/rw-wps",
    ]
    if platform.system() == "Windows":
        # MSVC's linker otherwise writes the wall-clock link time into the PE
        # header and every debug-directory entry. /Brepro replaces those
        # timestamps and the CodeView identity with content-derived values.
        rust_flags.extend((
            "-C", "link-arg=/Brepro",
            "-C", "target-feature=+crt-static",
        ))
    environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(rust_flags)
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    environment["SOURCE_DATE_EPOCH"] = source_date_epoch
    return environment


def build_release(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() not in {
        "x86_64", "AMD64",
    }:
        raise RuntimeError("RW-WPS release assembly requires Linux x86-64")
    if sys.version_info < (3, 11):
        raise RuntimeError("RW-WPS release assembly requires Python 3.11+")

    output = args.output_dir.resolve()
    archive = args.archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("output directory and archive must not exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("refusing to build RW-WPS from a dirty source tree")

    manifest = REPO / "tools" / "grib1_bridge" / "Cargo.toml"
    source_date_epoch = subprocess.check_output(
        ["git", "-C", str(REPO), "show", "-s", "--format=%ct", "HEAD"],
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix="rw-wps-release-build-") as raw:
        temporary = Path(raw)
        python_project = temporary / "python-project"
        python_inventory = _stage_rw_wps_python_project(python_project)
        wheel_dir = temporary / "wheel"
        wheel_dir.mkdir()
        wheel_environment = os.environ.copy()
        wheel_environment["SOURCE_DATE_EPOCH"] = source_date_epoch
        wheel_environment["PYTHONHASHSEED"] = "0"
        _run([
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(python_project),
        ], env=wheel_environment)
        wheels = sorted(wheel_dir.glob("rw_wps-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"wheel build produced {len(wheels)} RW-WPS wheels: {wheels}")

        cargo_target = temporary / "cargo-target"
        environment = _cargo_release_environment(
            source_root=REPO,
            target_dir=cargo_target,
            source_date_epoch=source_date_epoch,
        )
        _run([
            "cargo",
            "build",
            "--manifest-path",
            str(manifest),
            "--release",
            "--locked",
            "--offline",
        ], cwd=manifest.parent, env=environment)
        native = cargo_target / "release"
        distribution_args = argparse.Namespace(
            wheel=wheels[0],
            grib1_bridge=native / "grib1_bridge",
            grib2_inventory=native / "grib2_inventory",
            grib2_dump=native / "grib2_dump",
            gfs_bridge=native / "gfs_grib2_bridge",
            hrrr_bridge=native / "hrrr_grib2_bridge",
            cpu_backend=native / "libgpuwm_preprocess_cpu.so",
            output_dir=output,
            archive=archive,
        )
        result = build_distribution(distribution_args)

    result["build_interface"] = "tools/build_rw_wps_release.py"
    result["rust_build"] = "cargo-release-locked-offline"
    result["python_build"] = "pip-wheel-no-build-isolation-no-deps"
    result["python_package_inventory"] = python_inventory
    result["source_date_epoch"] = source_date_epoch
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        build_release(args), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
