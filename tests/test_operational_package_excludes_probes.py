"""Development probes and beside-the-code test suites never ship.

THE BREAKAGE THIS PREVENTS, measured on the 2.7.0 candidate's artifacts:
all three gpuwm wheels and the sdist carried 42 ``tilestream/test_*.py``
pytest suites (``test_gate.py`` alone 136,872 B) and the six
``tilestream/skeptic_*.py`` fault-injection probes at import top level,
and the sdist carried ``tests/test_n5s_toolchain.py`` although
RELEASE-EXCLUDE.txt drops it from the public tree with the campaign
harness it imports.  Two declarations LOOKED like they handled it:

* ``[tool.setuptools.packages.find] exclude = ["tilestream.skeptic*"]``
  matches packages, and there is no package by that name; the probes are
  modules directly under ``tilestream/``.
* ``prune tilestream/skeptic`` in MANIFEST.in names a directory that does
  not exist.

Neither mechanism can name a MODULE, so the wheel half is a ``build_py``
filter in setup.py (``DEVELOPMENT_MODULE_GLOBS``) and the sdist half is a
``recursive-exclude`` in MANIFEST.in.  Both halves are measured here
against the real setup.py and the real MANIFEST.in, the way
tests/test_sdist_excludes_staged_bridges.py measures the staged-binary
prune, and RELEASE-EXCLUDE.txt's test entries are checked against the
sdist so the public tree and the sdist cannot disagree about a test.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import tomllib

import pytest

setuptools = pytest.importorskip("setuptools")

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP = REPO_ROOT / "setup.py"
MANIFEST = REPO_ROOT / "MANIFEST.in"
PYPROJECT = REPO_ROOT / "pyproject.toml"
RELEASE_EXCLUDE = REPO_ROOT / "RELEASE-EXCLUDE.txt"
TILESTREAM = REPO_ROOT / "tilestream"

#: What the candidate shipped and must not: one suite, one probe, named so
#: a future rename of either family is noticed here rather than in a wheel.
SHIPPED_BY_MISTAKE = ("tilestream/test_gate.py", "tilestream/skeptic_probe.py")

#: Operational tilestream modules that MUST keep shipping, so the filter
#: cannot widen into the package it protects.
OPERATIONAL = ("autoplan", "restart_stream", "conftest")


def _require_source_tree() -> None:
    if not (SETUP.is_file() and MANIFEST.is_file() and TILESTREAM.is_dir()):
        pytest.skip("the packaging gate needs the source tree")


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)


def _declared_packages() -> list[str]:
    from setuptools.config import expand

    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    return expand.find_packages(include=find.get("include", ["*"]),
                                exclude=find.get("exclude", []),
                                root_dir=str(REPO_ROOT))


def _setup_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run the real setup.py with ``setup()`` captured instead of executed."""

    captured: dict = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.chdir(REPO_ROOT)
    namespace = runpy.run_path(str(SETUP), run_name="setup_under_test")
    captured["_namespace"] = namespace
    return captured


def _wheel_modules(monkeypatch: pytest.MonkeyPatch, package: str) -> set[str]:
    """Module names ``build_py`` would copy for ``package``, our cmdclass applied."""

    captured = _setup_kwargs(monkeypatch)
    build_py = captured["cmdclass"]["build_py"]
    distribution = setuptools.dist.Distribution({
        "name": "gpuwm", "packages": _declared_packages()})
    distribution.script_name = "setup.py"
    command = build_py(distribution)
    command.finalize_options()
    return {module for _pkg, module, _path
            in command.find_package_modules(package, package)}


def _sdist_file_list(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Every file setuptools would put in the sdist, MANIFEST.in applied."""

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
    # The WHOLE default set, not only the package half: `tests/test*.py`
    # reaches the sdist through distutils' optional defaults, and that is
    # the route the RELEASE-EXCLUDE'd test took.
    command.add_defaults()
    command.read_template()
    return {name.replace(os.sep, "/") for name in command.filelist.files}


def _release_excluded_tests() -> list[str]:
    rules = [line.strip() for line in RELEASE_EXCLUDE.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    return sorted(rule for rule in rules
                  if rule.startswith("tests/") and rule.endswith(".py"))


def test_the_probes_exist_so_the_measurement_is_not_vacuous() -> None:
    _require_source_tree()
    for relative in SHIPPED_BY_MISTAKE:
        assert (REPO_ROOT / relative).is_file(), relative
    assert len(list(TILESTREAM.glob("test_*.py"))) >= 10
    assert len(list(TILESTREAM.glob("skeptic_*.py"))) == 6
    assert "tests/test_n5s_toolchain.py" in _release_excluded_tests()


def test_the_wheel_drops_the_probes_and_keeps_the_package(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _require_source_tree()
    modules = _wheel_modules(monkeypatch, "tilestream")
    leaked = sorted(m for m in modules
                    if m.startswith(("test_", "skeptic_")))
    assert not leaked, (
        f"{len(leaked)} development module(s) would ship in the wheel: "
        f"{leaked[:6]}...  setup.py's build_py filter is not applied")
    for module in OPERATIONAL:
        assert module in modules, (module, sorted(modules)[:20])
    assert len(modules) >= 20, sorted(modules)


def test_the_filter_is_scoped_to_tilestream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `test_*` module in some OTHER shipped package is that package's
    business; the filter names tilestream and nothing else."""

    _require_source_tree()
    namespace = _setup_kwargs(monkeypatch)["_namespace"]
    globs = namespace["DEVELOPMENT_MODULE_GLOBS"]
    assert set(globs) == {"tilestream"}
    is_dev = namespace["is_development_module"]
    assert is_dev("tilestream", "test_gate")
    assert is_dev("tilestream", "skeptic_probe2")
    assert not is_dev("tilestream", "autoplan")
    assert not is_dev("gpuwm", "test_anything")


def test_the_sdist_drops_the_probes_and_the_release_excluded_tests(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _require_source_tree()
    listed = _sdist_file_list(monkeypatch)
    leaked = sorted(name for name in listed
                    if name.startswith(("tilestream/test_", "tilestream/skeptic_")))
    assert not leaked, (
        f"{len(leaked)} development file(s) would ship in the sdist: "
        f"{leaked[:6]}...  MANIFEST.in's recursive-exclude is missing")
    assert "tilestream/autoplan.py" in listed
    for relative in _release_excluded_tests():
        assert relative not in listed, (
            f"{relative} is in RELEASE-EXCLUDE.txt (dropped from the public "
            "tree) but would still ship inside the sdist; add it to "
            "MANIFEST.in")
    # Excluding the n5s test did not take tests/ with it.
    assert any(name.startswith("tests/test_") for name in listed)


def test_the_manifest_carries_the_sdist_half() -> None:
    _require_source_tree()
    rules = [line.strip() for line in MANIFEST.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    assert "recursive-exclude tilestream test_*.py skeptic_*.py" in rules
    assert "exclude tests/test_n5s_toolchain.py" in rules
