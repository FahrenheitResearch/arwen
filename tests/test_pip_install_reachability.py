"""A wheel install can reach the forecast runners.  Nothing else needed.

This is the fast half of the four-command acceptance bar

    pip install gpuwm[all] / gpuwm setup / gpuwm domain / gpuwm go

and it exists because the product's own documented chain used to be
unfinishable from an install: its last two stages were script paths
under ``tools/``, which only a git checkout has, so a pip user could
preprocess a GFS cycle and had nowhere to run it.

What is checked here is exactly the thing that changed: a venv with the
built wheel and NOTHING else -- no checkout on ``sys.path``, no
``PYTHONPATH``, a working directory outside the repository -- can
import and invoke both runners and the ``go`` front door.  Against
shipped 1.3.0 this fails: ``gpuwm.prepared_single_domain_forecast`` does
not exist there, and neither console script is declared.

The whole four-command transcript, including live GFS and a real
forecast, is ``tools/acceptance_pip_install_chain.py``; it needs
minutes, a card and a network, so it is a script rather than a test.
This module is marked ``slow`` because it builds a wheel (tens of
seconds, tens of megabytes) -- it is not marked ``network``: it installs
with ``--no-deps`` from the local wheel and never reaches an index.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.slow


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _offline_build_shortfall() -> str | None:
    """Why this interpreter cannot build the wheel offline, or ``None``.

    The fixture builds with ``--no-build-isolation`` so the suite never
    reaches an index, which makes the AMBIENT setuptools the build
    backend.  ``[build-system] requires`` names the floor the backend
    must meet (setuptools >=77, the PEP 639 license-expression reading);
    an interpreter below it cannot build the wheel at all, which is an
    environment prerequisite exactly like having a source tree -- not a
    reachability verdict.  The skip names the one-line remedy.
    """

    import tomllib
    with (ROOT / "pyproject.toml").open("rb") as stream:
        requires = tomllib.load(stream)["build-system"]["requires"]
    floor = None
    for entry in requires:
        spec = entry.replace(" ", "")
        if spec.startswith("setuptools>="):
            floor = spec.split(">=", 1)[1]
    if floor is None:
        return None
    import setuptools
    have = setuptools.__version__

    def release(version: str) -> tuple[int, ...]:
        parts = []
        for piece in version.split("."):
            if not piece.isdigit():
                break
            parts.append(int(piece))
        return tuple(parts)

    if release(have) >= release(floor):
        return None
    return (f"building the wheel offline (--no-build-isolation) needs "
            f"setuptools>={floor}; this interpreter has {have} "
            f"(remedy: python -m pip install -U setuptools)")


@pytest.fixture(scope="module")
def wheel_venv(tmp_path_factory):
    """A venv holding the built wheel and its dependencies, and no checkout.

    Dependencies come from the interpreter running the tests, copied in
    as a plain ``--no-deps`` install cannot import numpy: the venv is
    created with ``--system-site-packages`` and the wheel is installed
    into it, which shadows any outer ``gpuwm`` because a venv's own
    ``site-packages`` precedes the system one.  The assertion that the
    imported ``gpuwm`` really is the venv's is part of the test rather
    than an assumption.
    """

    if not (ROOT / "pyproject.toml").is_file():
        pytest.skip("needs the source tree to build a wheel")
    shortfall = _offline_build_shortfall()
    if shortfall is not None:
        pytest.skip(shortfall)

    work = tmp_path_factory.mktemp("pip-reachability")
    wheelhouse = work / "wheelhouse"
    # This wheel never leaves the temp directory, so the unpinned-tree
    # refusal (setup.py, tests/test_wheel_pin_gate.py) is explicitly
    # waived: runner reachability is under test here, not bridge
    # staging, and a dev checkout's pins are empty by design.
    build_env = dict(os.environ)
    build_env["GPUWM_ALLOW_UNPINNED_WHEEL"] = "1"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
         "--no-build-isolation", "-w", str(wheelhouse)],
        check=True, capture_output=True, text=True, timeout=1800,
        env=build_env)
    wheels = sorted(wheelhouse.glob("gpuwm-*.whl"))
    assert wheels, "no wheel was built"

    venv = work / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                    str(venv)], check=True, timeout=600)
    python = _venv_python(venv)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-index",
         str(wheels[-1])],
        check=True, capture_output=True, text=True, timeout=1800)
    return venv, work


def _run(venv: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the venv's python with no route back to the checkout."""

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["VIRTUAL_ENV"] = str(venv)
    return subprocess.run(
        [str(_venv_python(venv)), *args], cwd=str(cwd), env=env,
        capture_output=True, text=True, errors="replace", timeout=900)


def test_the_installed_gpuwm_is_the_wheels_not_the_checkouts(wheel_venv):
    """Everything below is worthless if the checkout is still winning."""

    venv, work = wheel_venv
    done = _run(venv, work, "-c",
                "import gpuwm, pathlib; "
                "print(pathlib.Path(gpuwm.__file__).parent)")
    assert done.returncode == 0, done.stderr
    resolved = Path(done.stdout.strip())
    assert venv in resolved.parents, resolved
    assert ROOT not in resolved.parents, resolved


@pytest.mark.parametrize("module", [
    "gpuwm.prepared_single_domain_forecast",
    "gpuwm.prepared_domain_tree_forecast",
])
def test_both_forecast_runners_are_importable_from_the_install(
        wheel_venv, module):
    """The regression this wave exists to prevent."""

    venv, work = wheel_venv
    done = _run(venv, work, "-c",
                f"import {module} as m; print(m.__file__)")
    assert done.returncode == 0, done.stderr
    assert str(venv) in done.stdout


def test_the_single_domain_runner_answers_as_a_module(wheel_venv):
    """`python -m gpuwm...` is what every printed command now names."""

    venv, work = wheel_venv
    done = _run(venv, work, "-m",
                "gpuwm.prepared_single_domain_forecast",
                "--show-capabilities")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout)
    assert payload["schema"] == "gpuwm-runner-capabilities-v1"
    # The registry ROUTE id is unchanged: it names a route, not a path.
    assert payload["runner"] == "tools.prepared_single_domain_forecast"


def test_the_go_front_door_is_reachable_and_names_no_checkout(wheel_venv):
    """`gpuwm go --help` from an install, with no tools/ anywhere."""

    venv, work = wheel_venv
    done = _run(venv, work, "-m", "gpuwm.cli", "go", "--help")
    assert done.returncode == 0, done.stderr
    assert "gpuwm go" in done.stdout or "usage" in done.stdout.lower()
    assert "clone of the repository" not in done.stdout


def test_the_console_scripts_are_declared(wheel_venv):
    """The two entry points a pip user can type without -m."""

    venv, _ = wheel_venv
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" else ""
    for name in ("gpuwm-prepared-forecast", "gpuwm-prepared-tree-forecast"):
        assert (bindir / f"{name}{suffix}").exists(), name
