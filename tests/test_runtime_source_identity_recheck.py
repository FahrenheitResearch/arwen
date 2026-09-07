"""A git hiccup must not destroy the receipt of a run that was fine.

``prepared_domain_tree_forecast`` snapshots ``_runtime_source_identity()``
when a forecast starts and compares it again at the very END -- after the
physics, after every frame is written -- so that source edited mid-run
cannot ship under the identity the run started with.  That check is worth
keeping.  What it could not do was tell two different events apart.

Measured 2026-09-02: a four-hour two-domain forecast (``adt4h_dt60``)
completed all 240 outer steps, wrote 62 ``wrfout`` frames, and then died
at the receipt gate on

    RuntimeError: forecast implementation changed during execution

writing ``evidence/failed-run-receipt.json`` instead of a run receipt.
Nothing had changed.  None of the five hashed files was modified, HEAD
had not moved, no commit was made, and creating untracked files was
tested and provably does not move the identity.  By elimination the
``git rev-parse`` at the end of the run failed to run -- and the
``except (FileNotFoundError, CalledProcessError)`` branch answers that
by setting ``commit = None; tree = None``, which a bare ``!=`` cannot
tell from HEAD having genuinely moved.

The cheap event destroyed the expensive one.  On a 72-hour forecast that
is a very expensive way to lose nothing but a git hiccup.

Both halves of the bar are here: a git outage at either end must NOT be
reported as a change, and a HEAD that actually moves must still be.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from gpuwm.provenance import git_executable
from tools import prepared_domain_tree_forecast as runner


_COMMIT = "eada530de44aa0b358c99cfd2e7c7d0eac106479"
_TREE = "8e7ed05eed258abb1018ad44bda60f9519f1af7f"
_OTHER_COMMIT = "1111111111111111111111111111111111111111"
_OTHER_TREE = "2222222222222222222222222222222222222222"

_SOURCES = {
    "gpuwm/core/model.py": "aaaa",
    "gpuwm/core/nest.py": "bbbb",
    "gpuwm/core/microphysics_transition.py": "cccc",
    "gpuwm/core/kernels/nest_microphysics.cu": "dddd",
    "gpuwm/prepared_domain_tree_forecast.py": "eeee",
}


def _identity(*, commit=_COMMIT, tree=_TREE, sources=None, version="2.5.8"):
    """One identity in the shape ``_runtime_source_identity`` publishes."""

    return {
        "gpuwm_version": version,
        "git_commit": commit,
        "git_tree": tree,
        "source_sha256": dict(_SOURCES if sources is None else sources),
    }


# ---------------------------------------------------------------------------
# The bug: an unanswered question is not a changed answer
# ---------------------------------------------------------------------------

def test_git_failing_at_the_end_of_the_run_is_not_a_changed_implementation():
    """The measured failure, reduced to its two identities."""

    launch = _identity()
    end = _identity(commit=None, tree=None)             # git could not run
    assert runner._runtime_source_identity_change(launch, end) is None, (
        "a transient failure to spawn git destroyed the receipt of a "
        "four-hour run that had already written all 62 of its frames")


def test_git_failing_at_launch_is_not_a_changed_implementation_either():
    """Symmetric: which END the outage lands on cannot matter."""

    launch = _identity(commit=None, tree=None)
    assert runner._runtime_source_identity_change(launch, _identity()) is None


def test_git_unavailable_at_both_ends_is_not_a_changed_implementation():
    """The wheel-install case: no repository to ask, at either end."""

    unresolved = _identity(commit=None, tree=None)
    assert runner._runtime_source_identity_change(
        unresolved, _identity(commit=None, tree=None)) is None


def test_losing_only_the_tree_still_compares_the_commit():
    """``.git`` yields the commit without git; the tree needs the object.

    So the realistic outage is HALF resolved, and the half that survives
    must still be doing its job.
    """

    launch = _identity()
    assert runner._runtime_source_identity_change(
        launch, _identity(tree=None)) is None
    moved = runner._runtime_source_identity_change(
        launch, _identity(commit=_OTHER_COMMIT, tree=None))
    assert moved is not None and "git_commit" in moved


# ---------------------------------------------------------------------------
# The check must keep catching what it exists to catch
# ---------------------------------------------------------------------------

def test_a_moved_head_is_still_a_changed_implementation():
    """The other half of the bar: the reason not to delete the gate."""

    moved = runner._runtime_source_identity_change(
        _identity(), _identity(commit=_OTHER_COMMIT, tree=_OTHER_TREE))
    assert moved is not None
    assert "git_commit" in moved and _OTHER_COMMIT in moved


def test_an_edited_hashed_source_is_still_a_changed_implementation():
    """And it is caught even with BOTH git halves unresolved.

    ``source_sha256`` is read from bytes on disk and can never fail to
    resolve, so it is compared always and strictly.  It is what makes
    skipping the git comparison survivable.
    """

    edited = dict(_SOURCES, **{"gpuwm/core/model.py": "ffff"})
    moved = runner._runtime_source_identity_change(
        _identity(commit=None, tree=None),
        _identity(commit=None, tree=None, sources=edited))
    assert moved is not None
    assert "gpuwm/core/model.py" in moved


def test_a_swapped_version_is_still_a_changed_implementation():
    moved = runner._runtime_source_identity_change(
        _identity(), _identity(version="2.5.9"))
    assert moved is not None and "gpuwm_version" in moved


def test_the_refusal_names_the_component_that_moved():
    """A nine-word traceback with nothing actionable in it was the old
    receipt.  Every refusal here says WHICH component differs."""

    for before, after in (
        (_identity(), _identity(version="9.9.9")),
        (_identity(), _identity(commit=_OTHER_COMMIT)),
        (_identity(), _identity(tree=_OTHER_TREE)),
        (_identity(),
         _identity(sources=dict(_SOURCES, **{"gpuwm/core/nest.py": "zzzz"}))),
    ):
        moved = runner._runtime_source_identity_change(before, after)
        assert moved is not None and "->" in moved


# ---------------------------------------------------------------------------
# The same bar, against a real repository whose HEAD really moves
# ---------------------------------------------------------------------------

def _git(repo: Path, *arguments: str) -> str:
    """git against ``repo``, isolated from the user's own configuration."""

    environment = dict(
        os.environ,
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid",
    )
    return subprocess.check_output(
        [str(git_executable()), "-C", str(repo), *arguments],
        text=True, env=environment, stderr=subprocess.DEVNULL).strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """A real one-commit checkout, standing in for ``REPOSITORY_ROOT``."""

    root = tmp_path / "checkout"
    root.mkdir()
    _git(root.parent, "init", "-q", "-b", "main", str(root))
    (root / "model.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "model.py")
    _git(root, "commit", "-q", "-m", "first")
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", root)
    return root


@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_a_real_moved_head_is_reported_as_changed(repository):
    """End to end: two resolutions of a repository that really committed."""

    before = runner._head_commit_and_tree()
    assert before[0] is not None and before[1] is not None
    (repository / "model.py").write_text("x = 2\n", encoding="utf-8")
    _git(repository, "commit", "-q", "-am", "second")
    after = runner._head_commit_and_tree()
    assert after[0] is not None and after != before

    launch = _identity(commit=before[0], tree=before[1])
    end = _identity(commit=after[0], tree=after[1])
    moved = runner._runtime_source_identity_change(launch, end)
    assert moved is not None and "git_commit" in moved


@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_an_untracked_file_does_not_move_the_identity(repository):
    """Tested by hand during the incident; kept, so it stays true."""

    before = runner._head_commit_and_tree()
    (repository / "scratch.log").write_text("noise\n", encoding="utf-8")
    assert runner._head_commit_and_tree() == before


@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_the_commit_survives_a_git_that_cannot_be_spawned(repository,
                                                          monkeypatch):
    """``.git`` says which commit is executing without a subprocess.

    A spawn failure therefore no longer erases the commit at all, which
    is why the degraded comparison above is rarely reached.
    """

    resolved = runner._head_commit_and_tree()
    monkeypatch.setattr("gpuwm.provenance.git_executable", lambda **k: None)
    commit, tree = runner._head_commit_and_tree()
    assert commit == resolved[0], "the commit is readable from .git bytes"
    assert tree is None, "the tree lives inside the commit object; say so"


# ---------------------------------------------------------------------------
# The resolution's hardening, and the shape it must not change
# ---------------------------------------------------------------------------

def test_a_transient_spawn_failure_is_retried(monkeypatch, repository):
    """One failed spawn is the whole incident; it must not be the answer."""

    calls: list[int] = []
    real = subprocess.check_output

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError(2, "The system cannot find the file")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "check_output", flaky)
    assert runner._git_head_query("rev-parse", "HEAD") is not None
    assert len(calls) == 2


@pytest.mark.parametrize("error", [
    PermissionError(13, "Permission denied"),
    OSError(1455, "The paging file is too small"),
    subprocess.TimeoutExpired(cmd="git", timeout=30.0),
    subprocess.CalledProcessError(128, "git"),
])
def test_no_spawn_failure_escapes_as_a_traceback(monkeypatch, repository,
                                                 error):
    """The old clause caught two exception types out of the family.

    A ``PermissionError`` or a ``TimeoutExpired`` went uncaught, so the
    same environmental hiccup could end a finished run with a traceback
    rather than with the refusal above.  Both are now an unanswered
    question, which is what they are.
    """

    def failing(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner.subprocess, "check_output", failing)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    assert runner._git_head_query("rev-parse", "HEAD") is None


def test_a_tree_with_no_repository_asks_git_nothing(tmp_path, monkeypatch):
    """A wheel install answers 128 deterministically; do not pay to hear it."""

    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("git was spawned for a tree with no repository")

    monkeypatch.setattr(runner.subprocess, "check_output", unexpected)
    assert runner._git_head_query("rev-parse", "HEAD") is None


def test_the_published_identity_shape_did_not_move():
    """These keys are hashed into checkpoint headers.

    ``sealed_extension_fingerprint`` and the tree restart identity both
    digest this mapping and publish it in checkpoint headers, so a field
    added here would move every already-sealed fingerprint and strand the
    legs carrying them.  The fix improved the RELIABILITY of two values;
    it must not have added a fifth key.
    """

    assert set(runner._runtime_source_identity()) == {
        "gpuwm_version", "git_commit", "git_tree", "source_sha256"}


def test_the_real_identity_of_this_checkout_compares_equal_to_itself():
    """The gate, resolved twice for real, must pass."""

    first = runner._runtime_source_identity()
    second = runner._runtime_source_identity()
    assert runner._runtime_source_identity_change(first, second) is None
