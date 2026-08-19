"""A SOURCE distribution never carries one platform's native binaries.

Measured on the 2.5.0 Linux shakeout at 0f45dcfec: running ``python -m
build`` after ``tools/stage_wheel_bridges.py --platform linux-x86_64``
swept every staged native ELF artifact into the sdist -- the 18 the
declaration carried at that commit, 20 extra members under
``gpuwm-2.5.0/gpuwm/libexec/bridges/``, the tarball up from
95,236,252 B to 114,906,421 B, and a *source* distribution shipping
Linux executables.  The published shape survived only because
``.github/workflows/publish.yml`` happens to run ``--clean``
immediately before ``python -m build``; nothing in ``setup.py``,
``MANIFEST.in`` or the backend made the wrong build fail, so every
local or non-CI build of a staged tree produced it.

Two mechanisms close it, and they answer different questions:

* ``MANIFEST.in`` prunes the staged directory, so the sdist's declared
  SHAPE has no room for platform binaries however it is built.
* ``setup.py`` refuses ``sdist`` on a staged tree, because the prune
  alone would silently mislead: ``python -m build`` builds the wheel
  FROM the sdist it just made, so a pruned sdist would hand the wheel
  build a tree with no staged artifacts and produce a ``py3-none-any``
  wheel while the operator was staging for a platform one.

Both directions are proven -- the refusal on a staged tree, and an
ordinary sdist still building from a clean one -- against the real
``setup.py`` and the real ``MANIFEST.in``.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

pytest.importorskip("setuptools")

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "setup.py"
MANIFEST = REPO_ROOT / "MANIFEST.in"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Where the staging tool puts the artifacts, as the sdist would name
#: them.  Spelled here rather than imported so the test still means
#: something if the package cannot be imported.
STAGED_PREFIX = "gpuwm/libexec/bridges"


def _require_source_tree() -> None:
    if not (SETUP.is_file() and PYPROJECT.is_file()):
        pytest.skip("the sdist shape gate needs the source tree")


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)


def _declared_packages() -> list[str]:
    from setuptools.config import expand

    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    return expand.find_packages(
        include=find.get("include", ["*"]),
        exclude=find.get("exclude", []),
        root_dir=str(REPO_ROOT),
    )


@pytest.fixture()
def staged_tree(tmp_path: pytest.TempPathFactory) -> Path:
    """The real repository, with artifact-shaped bytes staged in it.

    Real staging needs a release cargo build of three workspaces, which
    a test cannot afford; what the packaging machinery sees is a
    directory of files, and that is what this makes.  Removed
    afterwards, and it refuses to touch a directory it did not create.
    """

    _require_source_tree()
    staged = REPO_ROOT / "gpuwm" / "libexec" / "bridges"
    if staged.exists():
        pytest.skip(f"{staged} already exists; this tree is really staged "
                    "and the fixture must not delete somebody's artifacts")
    staged.mkdir(parents=True)
    try:
        (staged / "BUNDLE.json").write_text(
            '{"schema": "gpuwm-wheel-bridge-bundle-v1", '
            '"platform": "linux-x86_64", "artifacts": []}\n',
            encoding="utf-8")
        (staged / "rw_wrfbatch").write_bytes(b"\x7fELF" + b"\x00" * 512)
        (staged / "libnetcdf_writer.so").write_bytes(b"\x7fELF" + b"\x00" * 512)
        yield staged
    finally:
        shutil.rmtree(staged.parent, ignore_errors=True)


def _sdist_file_list(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Every file setuptools would put in the sdist, MANIFEST.in applied.

    ``_add_defaults_python`` is the step that folds ``package_data``
    into the source distribution -- the step that swept the binaries in
    -- and ``read_template`` is where ``MANIFEST.in`` gets to take them
    back out.  Running both measures the tarball's membership without
    spending the minutes a real 95 MB build costs.
    """

    import setuptools
    from setuptools.command.sdist import sdist

    from distutils.filelist import FileList

    config = _pyproject()["tool"]["setuptools"]
    distribution = setuptools.dist.Distribution({
        "name": "gpuwm",
        "version": _pyproject()["project"]["version"],
        "packages": _declared_packages(),
        "package_data": config["package-data"],
        "exclude_package_data": config.get("exclude-package-data", {}),
    })
    distribution.script_name = "setup.py"
    command = sdist(distribution)
    command.finalize_options()
    command.filelist = FileList()
    monkeypatch.chdir(REPO_ROOT)
    command._add_defaults_python()
    command.read_template()
    return {name.replace(os.sep, "/") for name in command.filelist.files}


def test_the_sdist_carries_no_staged_bridge_artifact(
        staged_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The measurement, on a tree that really has artifacts staged."""

    listed = _sdist_file_list(monkeypatch)
    smuggled = sorted(name for name in listed
                      if name.startswith(STAGED_PREFIX + "/"))
    assert not smuggled, (
        f"{len(smuggled)} staged artifact(s) would ship inside the SOURCE "
        f"distribution:\n  " + "\n  ".join(smuggled)
        + f"\nMANIFEST.in must prune {STAGED_PREFIX}.")


def test_the_prune_does_not_take_the_rest_of_the_package_with_it(
        staged_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A prune written one directory too wide is silent until a cut."""

    listed = _sdist_file_list(monkeypatch)
    # Python-side members only: this file list is the ``package_data``
    # half, not the standards files setuptools adds separately.
    for needed in ("gpuwm/__init__.py",
                   "gpuwm/bridges.py",
                   "gpuwm/data/bridges/bridge-pins.json"):
        assert needed in listed, (
            f"{needed} would be absent from the sdist; the staged-bridge "
            "prune is too wide")


def _synthetic_tree(tmp_path: Path, *, staged: bool) -> Path:
    """A minimal tree driven by the REAL setup.py, byte for byte."""

    tree = tmp_path / ("staged" if staged else "clean")
    (tree / "gpuwm" / "data" / "bridges").mkdir(parents=True)
    (tree / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(SETUP, tree / "setup.py")
    (tree / "pyproject.toml").write_text(
        '[build-system]\n'
        'requires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '\n'
        '[project]\n'
        'name = "gpuwm-sdist-gate-proof"\n'
        'version = "0.0.1"\n'
        '\n'
        '[tool.setuptools]\n'
        'packages = ["gpuwm"]\n',
        encoding="utf-8")
    if staged:
        bridges = tree / "gpuwm" / "libexec" / "bridges"
        bridges.mkdir(parents=True)
        (bridges / "BUNDLE.json").write_text(
            '{"schema": "gpuwm-wheel-bridge-bundle-v1", '
            '"platform": "linux-x86_64", "artifacts": []}\n',
            encoding="utf-8")
        (bridges / "rw_wrfbatch").write_bytes(b"\x7fELF" + b"\x00" * 512)
    return tree


def _sdist(tree: Path) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.pop("GPUWM_ALLOW_UNPINNED_WHEEL", None)
    return subprocess.run(
        [sys.executable, "setup.py", "sdist"],
        cwd=str(tree), env=environment, capture_output=True, text=True,
        errors="replace", timeout=900)


def test_an_sdist_of_a_staged_tree_refuses_and_names_both_remedies(
        tmp_path: Path) -> None:
    _require_source_tree()
    tree = _synthetic_tree(tmp_path, staged=True)
    completed = _sdist(tree)
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, (
        "an sdist built from a staged tree succeeded; it carries one "
        "platform's binaries in a source distribution:\n" + output)
    assert "libexec/bridges" in output, output
    # Both ways out, because which one is right depends on which
    # distribution the operator is actually after.
    assert "stage_wheel_bridges.py --clean" in output, output
    assert "--wheel" in output, output
    assert not list(tree.glob("dist/*.tar.gz")), "a tarball was written anyway"


def test_an_sdist_of_a_clean_tree_still_builds(tmp_path: Path) -> None:
    """The gate must not refuse the shape the release actually ships."""

    _require_source_tree()
    tree = _synthetic_tree(tmp_path, staged=False)
    completed = _sdist(tree)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert list(tree.glob("dist/*.tar.gz")), output


def test_the_manifest_prunes_the_staged_directory() -> None:
    """The declaration itself, readable without setuptools.

    The measurement above proves today's behaviour; this names the line
    that produces it, so deleting the line fails with the reason rather
    than with a file list.
    """

    _require_source_tree()
    directives = [
        line.split("#", 1)[0].strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    assert f"prune {STAGED_PREFIX}" in directives, (
        f"MANIFEST.in no longer prunes {STAGED_PREFIX}, so a source "
        "distribution built from a staged tree carries that platform's "
        "native binaries")
