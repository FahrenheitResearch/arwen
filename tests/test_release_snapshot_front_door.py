"""The snapshot builder has to be askable what it does.

`work/build_release_snapshot.py` deletes its destination directory and
rebuilds it from a `git archive` of HEAD.  It had no argument parser at
all, so that happened on *every* invocation -- `--help` included, since
there was nothing to interpret the flag.  The one command an agent
reaches for before running an unfamiliar tool was the command that ran
it, against a hardcoded path, destructively.

The build is unchanged and must stay so; only the door in front of it is
new.  These tests hold both halves: `--help` and a bare run write
nothing, and `--build` is what carries a caller through to the build.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "work" / "build_release_snapshot.py"

requires_builder = pytest.mark.skipif(
    not BUILDER.is_file(),
    reason="work/build_release_snapshot.py is not in this tree "
           "(published snapshot: the builder is publisher scaffolding)",
)


def _snap():
    sys.path.insert(0, str(REPO / "work"))
    try:
        import build_release_snapshot as module
    finally:
        sys.path.pop(0)
    return module


def _run(args, destination: Path):
    """Invoke the file the way a cut agent would, aimed at a scratch path.

    ``ARWEN_SNAPSHOT_DIR`` is redirected so that a regression which does
    reach the build destroys a tmp directory rather than the operator's
    real snapshot -- and so that "created nothing" is checkable.
    """
    env = dict(os.environ)
    env["ARWEN_SNAPSHOT_DIR"] = str(destination)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, str(BUILDER), *args],
                          capture_output=True, text=True, env=env)


# ---- the command line ----------------------------------------------------

@requires_builder
def test_help_answers_and_builds_nothing(tmp_path):
    destination = tmp_path / "snapshot"
    result = _run(["--help"], destination)

    assert result.returncode == 0, result.stderr
    assert "--build" in result.stdout
    assert not destination.exists()


@requires_builder
def test_a_bare_invocation_refuses_and_says_why(tmp_path):
    destination = tmp_path / "snapshot"
    result = _run([], destination)

    assert result.returncode != 0
    message = result.stderr.strip()
    assert message.count("\n") == 0, f"the refusal is one line: {message!r}"
    # It has to name the directory at risk and the flag that accepts it.
    assert str(destination) in message
    assert "--build" in message
    assert not destination.exists()


# ---- the gate, without ever running a build ------------------------------

@requires_builder
def test_the_build_is_unreachable_without_the_flag(monkeypatch):
    snap = _snap()
    def _never():
        raise AssertionError("the build ran on a bare invocation")

    monkeypatch.setattr(snap, "build_snapshot", _never)
    assert snap.main([]) == 2


@requires_builder
def test_the_flag_is_what_carries_a_caller_into_the_build(monkeypatch):
    """--build reaches the build, and its exit status is the build's."""

    snap = _snap()
    calls = []
    monkeypatch.setattr(snap, "build_snapshot",
                        lambda: calls.append(1) or 0)
    assert snap.main(["--build"]) == 0
    assert calls == [1]


@requires_builder
def test_an_unknown_flag_is_refused_rather_than_ignored():
    snap = _snap()
    with pytest.raises(SystemExit) as excinfo:
        snap.build_parser().parse_args(["--rebuild"])
    assert excinfo.value.code != 0


@requires_builder
def test_importing_the_builder_is_side_effect_free(tmp_path):
    """Two test modules import it for its helpers; that must stay cheap."""

    destination = tmp_path / "snapshot"
    env = dict(os.environ)
    env["ARWEN_SNAPSHOT_DIR"] = str(destination)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);"
         " import build_release_snapshot",
         str(REPO / "work")],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert not destination.exists()
