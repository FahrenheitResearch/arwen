"""``tools/ci_test_replay.py`` builds a venv the job it replays would have.

THE DEFECT THIS CLOSES, scratch-proven at the 2.5.0 cut and never
committed.  The publish workflow's ``test`` job runs on ``setup-python``
3.11, whose environment carries setuptools -- ensurepip bundled it through
3.11.  ``python -m venv`` on 3.12+ carries pip and nothing else (PEP 632
took setuptools out), and two of the job's own test files measure what the
wheel would contain by driving setuptools directly.  Those files REFUSE by
design when setuptools is absent, because a silent skip of a packaging gate
reports green.

So on any 3.12+ interpreter the replay failed on its own venv while the job
it claims to replay was fine: the verdict described the harness, not the
tree.  A harness that can only be trusted on one interpreter minor is a
harness nobody can point at a public snapshot.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_test_replay import (
    VENV_SEEDS, WORKFLOW, bootstrap_venv, parse_test_job, venv_python,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_job_the_replay_runs_cannot_run_without_setuptools():
    """The premise, read off the workflow rather than restated.

    If the job's file list ever stops needing setuptools this test says so,
    and the seed below can be reconsidered deliberately instead of being
    carried as cargo.
    """

    _marker, files = parse_test_job(WORKFLOW.read_text(encoding="utf-8"))
    needing = [
        name for name in files
        if "import setuptools" in (REPO_ROOT / name).read_text(
            encoding="utf-8")
    ]
    assert needing, files
    # ...and they refuse rather than skip, so an absent setuptools is a
    # red replay and not a quiet one.
    for name in needing:
        assert "setuptools is not installed" in (
            REPO_ROOT / name).read_text(encoding="utf-8"), name


def test_setuptools_is_seeded_into_the_replay_venv():
    """The remedy, where the harness can act on it."""

    assert "setuptools" in VENV_SEEDS
    assert "build" in VENV_SEEDS


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("GPUWM_NETWORK_TESTS") != "1",
                    reason="builds a venv and installs into it from PyPI; "
                           "set GPUWM_NETWORK_TESTS=1 to run")
def test_the_replay_venv_is_constructed_and_carries_its_seeds(tmp_path: Path):
    """Against the artifact: the harness's own venv, on THIS interpreter.

    Run under 3.12+ this is the whole defect -- the venv is built, and
    ``import setuptools`` inside it is what the job's packaging tests need
    and what a bare 3.12+ venv does not have.
    """

    venv_dir = tmp_path / "venv"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    python = bootstrap_venv(venv_dir, python=sys.executable, env=env)

    assert python == venv_python(venv_dir)
    assert python.is_file()
    for module in ("setuptools", "build"):
        probe = subprocess.run(
            [str(python), "-c", f"import {module}; print({module}.__file__)"],
            capture_output=True, text=True)
        assert probe.returncode == 0, (module, probe.stderr)
        # Inside the replay venv, not borrowed from the box.
        assert str(venv_dir) in probe.stdout, (module, probe.stdout)
