"""No wheel can be built while the packaged bridge pins pin nothing.

The breakage this gate prevents, measured on the 2.5.0 release
candidate: ``tools/build_bridge_bundle.py pin`` was skipped before the
wheel build, so the wheel shipped ``gpuwm/data/bridges/bridge-pins.json``
with ``release: null`` and ``platforms: {}``.  On a clean home
(``USERPROFILE`` redirected to an empty directory), ``pip install gpuwm
&& gpuwm setup`` then reported FAILED bridges -- no GRIB decoder, no
NetCDF decoder, ``renderer rw_wrfbatch: not built`` -- every Rust door
of a fresh install dead, while every check on the builder's own box
passed because its bridges were staged long ago.  The pin step existed
the whole time; nothing made skipping it fail.

Now something does: ``setup.py`` refuses ``bdist_wheel`` while the pins
document declares no release and no platforms, and it fires in
``finalize_options``, before a byte is packed, on every route a wheel is
built by -- ``python -m build``, ``pip wheel .``, ``pip install <tree>``
and ``python setup.py bdist_wheel`` all pass through that command.
``GPUWM_ALLOW_UNPINNED_WHEEL=1`` is the explicit dev override for a
wheel that will never be published (this suite's own fixtures use it);
an override is a workaround, never a fix, and the built wheel stays
exactly as unshippable as before.

Both directions are proven here (validate the instrument): the refusal
on an unpinned tree -- including THIS repository's real tree, whose
committed pins are empty by owner ruling -- and the pass on a pinned
one, via a synthetic package small enough to build in seconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

# The subprocesses below run `sys.executable setup.py bdist_wheel` with an
# inherited environment, so setuptools must be importable by exactly that
# interpreter (on some boxes it lives in the user site).
pytest.importorskip("setuptools")

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"
REAL_PINS = ROOT / "gpuwm" / "data" / "bridges" / "bridge-pins.json"

#: The committed (honestly unpinned) document, byte-shape the release
#: checklist requires of a tagged source tree.
UNPINNED_PINS = {
    "schema": "gpuwm-bridge-pins-v1",
    "release": None,
    "platforms": {},
    "note": "test fixture: the unpinned shape a source tree carries",
}

#: What the pin step leaves behind: a named release and at least one
#: pinned platform.  The gate reads exactly this much; deep validation
#: is `gpuwm.bridge_assets.parse_pins`, which the pin tool already runs
#: on everything it writes.
PINNED_PINS = {
    "schema": "gpuwm-bridge-pins-v1",
    "release": "v9.9.9",
    "platforms": {
        "win-x86_64": {
            "bundle": {"filename": "gpuwm-bridges-v9.9.9-win-x86_64.zip",
                       "bytes": 1, "sha256": "0" * 64},
            "binaries": [{"artifact": "x", "filename": "x.exe",
                          "bytes": 1, "sha256": "0" * 64}],
            "assets": [],
        },
    },
    "note": "test fixture: a pinned document",
}


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("GPUWM_ALLOW_UNPINNED_WHEEL", None)
    env.update(extra)
    return env


def _bdist_wheel(tree: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=str(tree), env=env, capture_output=True, text=True,
        errors="replace", timeout=900)


def _synthetic_tree(tmp_path: Path, pins: dict | None) -> Path:
    """A minimal package tree driven by the REAL setup.py, byte for byte.

    Small enough that the passing direction builds an actual wheel in
    seconds, which the real tree cannot afford in a battery lane.
    """

    tree = tmp_path / "tree"
    bridges = tree / "gpuwm" / "data" / "bridges"
    bridges.mkdir(parents=True)
    (tree / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(SETUP, tree / "setup.py")
    (tree / "pyproject.toml").write_text(
        '[build-system]\n'
        'requires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '\n'
        '[project]\n'
        'name = "gpuwm-pin-gate-proof"\n'
        'version = "0.0.1"\n'
        '\n'
        '[tool.setuptools]\n'
        'packages = ["gpuwm"]\n',
        encoding="utf-8")
    if pins is not None:
        (bridges / "bridge-pins.json").write_text(
            json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    return tree


def _ambient_setuptools_shortfall() -> str | None:
    """Why this interpreter cannot even LOAD the real tree's config.

    The repository's ``[build-system]`` floor (setuptools>=77 for the
    PEP 639 license expression) applies to the real-tree leg only; the
    synthetic trees below use a minimal pyproject any setuptools builds.
    Same guard as ``tests/test_pip_install_reachability.py``.
    """

    import tomllib
    from importlib import metadata

    from packaging.requirements import Requirement

    with (ROOT / "pyproject.toml").open("rb") as stream:
        requires = tomllib.load(stream)["build-system"]["requires"]
    for spec in requires:
        requirement = Requirement(spec)
        if requirement.name != "setuptools":
            continue
        try:
            version = metadata.version("setuptools")
        except metadata.PackageNotFoundError:
            return "setuptools is not importable by this interpreter"
        if not requirement.specifier.contains(version, prereleases=True):
            return (f"ambient setuptools {version} is below the tree's "
                    f"build floor {requirement.specifier}; the real-tree "
                    "leg cannot reach the gate")
    return None


def test_the_real_tree_refuses_a_wheel_while_its_pins_are_empty(tmp_path):
    """The actual repository, the actual setup.py, the actual refusal.

    Skips only when this working tree's pins are populated -- which is
    the transient state of a cut between the pin step and the wheel
    build, exactly when a wheel SHOULD build -- or when the ambient
    setuptools is too old to load the tree's config at all.
    """

    shortfall = _ambient_setuptools_shortfall()
    if shortfall is not None:
        pytest.skip(shortfall)
    payload = json.loads(REAL_PINS.read_text(encoding="utf-8"))
    if payload.get("release") is not None and payload.get("platforms"):
        pytest.skip("this tree's pins are populated (a cut in flight); "
                    "the refusal precondition is absent")
    completed = _bdist_wheel(ROOT, _env())
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, (
        "bdist_wheel succeeded from an unpinned tree; that wheel installs "
        "on a clean home as FAILED bridges (the 2.5.0 blocker):\n" + output)
    assert "bridge-pins.json" in output
    assert "FAILED bridges" in output
    assert "tools/build_bridge_bundle.py pin" in output


def test_an_unpinned_synthetic_tree_refuses_and_names_the_remedy(tmp_path):
    tree = _synthetic_tree(tmp_path, UNPINNED_PINS)
    completed = _bdist_wheel(tree, _env())
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "FAILED bridges" in output
    assert "tools/build_bridge_bundle.py pin" in output
    assert "GPUWM_ALLOW_UNPINNED_WHEEL" in output
    assert not list(tree.glob("dist/*.whl"))


def test_a_tree_with_no_pins_document_refuses(tmp_path):
    """A missing document is the same skip wearing a worse disguise."""

    tree = _synthetic_tree(tmp_path, None)
    completed = _bdist_wheel(tree, _env())
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert "bridge-pins.json" in output
    assert not list(tree.glob("dist/*.whl"))


def test_a_pinned_tree_builds_a_wheel(tmp_path):
    """The gate must not refuse the state the pin step leaves behind."""

    tree = _synthetic_tree(tmp_path, PINNED_PINS)
    completed = _bdist_wheel(tree, _env())
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert list(tree.glob("dist/*.whl")), output


def test_the_dev_override_builds_and_says_unpublishable(tmp_path):
    tree = _synthetic_tree(tmp_path, UNPINNED_PINS)
    completed = _bdist_wheel(tree, _env(GPUWM_ALLOW_UNPINNED_WHEEL="1"))
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert list(tree.glob("dist/*.whl")), output
    assert "never be published" in output
