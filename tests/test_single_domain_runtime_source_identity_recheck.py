"""A git hiccup, or the run's own output, must not destroy its receipt.

``prepared_single_domain_forecast`` snapshots ``_runtime_source_identity()``
when a forecast starts and compares it again at the very END -- after the
physics, after every frame is written -- so that source edited mid-run
cannot ship under the identity the run started with.  That check is worth
keeping.  What ``!=`` could not do was tell three different events apart.

Two of them are not implementation changes at all, and this runner is
exposed to both more badly than the tree runner is:

1. **A transient git failure.**  ``_runtime_source_identity`` builds on
   ``gpuwm.runtime_manifest.provenance(REPO)``, whose checkout branch
   catches ``(OSError, subprocess.SubprocessError)`` and FALLS THROUGH
   to the editable/wheel ladder.  So one flaky ``git rev-parse`` at the
   end of a run re-answers ``identity_source``, ``git_commit``,
   ``git_tree``, ``git_status_short`` and ``installed_wheel`` /
   ``installed_editable`` from a lower rung, all at once, with nothing
   on disk changed -- and the run dies on ``forecast runtime
   implementation changed during run``.

2. **The run's own scratch files.**  ``git_status_short`` is part of
   this identity, and ``git status --short`` lists untracked files.
   ``runs/`` is not ignored (measured in this worktree: 0 status lines
   before ``touch runs/probe``, 1 after), so a run that writes inside
   the checkout fails its own receipt for having produced output.  On an
   editable install it moves twice over, because
   ``installed_editable.git.untracked_files`` counts the same rows.

Fixing (1) surfaced a third thing, measured against the real resolver
rather than reasoned about: the fall-through does not merely LOSE the
git answer here, it substitutes a DIFFERENT TREE's.  ``pip install -e``
points at the main checkout, so in a linked worktree -- which is where
this project's work happens -- the editable rung answered ``c1b32d0``
while the code being hashed was ``eada530d``.  The two ends of a healthy
run were comparing two unrelated commits.  So the git half is re-read
from REPO's own ``.git`` whenever a rung that speaks for another tree
answers, and the rung-specific artifacts are compared only within one
rung.

Both halves of the bar are here: none of these events may be reported as
a change, and a HEAD that really moves -- or a tracked source that is
really edited -- must still be.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from gpuwm import prepared_single_domain_forecast as runner
from gpuwm.provenance import git_executable


_COMMIT = "eada530de44aa0b358c99cfd2e7c7d0eac106479"
_TREE = "8e7ed05eed258abb1018ad44bda60f9519f1af7f"
_OTHER_COMMIT = "1111111111111111111111111111111111111111"
_OTHER_TREE = "2222222222222222222222222222222222222222"

#: The ten modules this runner hashes, abbreviated.  Only the keys and
#: the fact that they are digests of bytes on disk matter here.
_SOURCES = {
    "gpuwm/core/model.py": "aaaa",
    "gpuwm/ingest/hrrr_physics.py": "bbbb",
    "gpuwm/ingest/hrrr_surface.py": "cccc",
    "gpuwm/ingest/lateral_bc.py": "dddd",
    "gpuwm/ingest/prepared_cache.py": "eeee",
    "gpuwm/ingest/real.py": "ffff",
    "gpuwm/io/wrfout.py": "aaab",
    "gpuwm/native_wrf_contract.py": "aaac",
    "gpuwm/source_authorities.py": "aaad",
    "gpuwm/prepared_single_domain_forecast.py": "aaae",
}


def _checkout(*, commit=_COMMIT, tree=_TREE, status=(), sources=None):
    """The identity ``provenance``'s git rung publishes, plus digests."""

    return {
        "identity_source": "git",
        "git_commit": commit,
        "git_tree": tree,
        "git_status_short": None if status is None else list(status),
        "distribution_manifest_sha256": None,
        "installed_wheel": None,
        "installed_editable": None,
        "source_sha256": dict(_SOURCES if sources is None else sources),
    }


def _wheel(sources=None):
    """What the SAME checkout answers after falling through the git rung.

    This is not a hypothetical shape: it is exactly what
    ``provenance()`` returns once its ``git`` branch raises, because the
    ladder below it does not know a repository was ever there.
    """

    return {
        "identity_source": "installed-wheel-record",
        "git_commit": None,
        "git_tree": None,
        "git_status_short": None,
        "distribution_manifest_sha256": None,
        "installed_wheel": {"record_file_count": 812,
                            "record_sha256": "9c9c"},
        "installed_editable": None,
        "source_sha256": dict(_SOURCES if sources is None else sources),
    }


def _editable(*, commit=_COMMIT, dirty_files=0, untracked_files=0,
              status=(), sources=None):
    """The other rung a fallen-through checkout can land on."""

    return {
        "identity_source": "installed-editable-source",
        "git_commit": commit,
        "git_tree": _TREE,
        "git_status_short": None if status is None else list(status),
        "distribution_manifest_sha256": None,
        "installed_wheel": None,
        "installed_editable": {
            "source_root": "/checkout",
            "git": {"commit": commit[:12], "commit_full": commit,
                    "branch": "main", "dirty": dirty_files > 0,
                    "dirty_files": dirty_files,
                    "untracked_files": untracked_files},
        },
        "source_sha256": dict(_SOURCES if sources is None else sources),
    }


# ---------------------------------------------------------------------------
# Mode 1: an unanswered question is not a changed answer
# ---------------------------------------------------------------------------

def test_a_git_fallthrough_at_the_end_is_not_a_changed_implementation():
    """The whole of mode 1, reduced to its two identities.

    Every one of the five moved fields moves at once, which is precisely
    why ``!=`` could not read it: it looks louder than a real change.
    """

    launch, end = _checkout(), _wheel()
    assert runner._runtime_source_identity_change(launch, end) is None, (
        "one git subprocess that failed to spawn destroyed the receipt "
        "of a run that had already written every frame")


def test_a_git_fallthrough_at_launch_is_not_a_changed_implementation():
    """Symmetric: which END the outage lands on cannot matter."""

    assert runner._runtime_source_identity_change(
        _wheel(), _checkout()) is None


def test_a_fallthrough_to_the_editable_rung_is_not_a_change_either():
    """The likelier landing: an editable install in a real checkout.

    ``_editable_provenance`` reads the commit out of ``.git`` bytes, so
    this rung keeps the strongest git binding -- and the comparison must
    still use it rather than trip over the rung's name.
    """

    assert runner._runtime_source_identity_change(
        _checkout(), _editable()) is None


def test_the_commit_is_still_compared_across_a_fallthrough():
    """Not-comparing ``identity_source`` must not stop comparing HEAD.

    Both rungs resolved ``git_commit`` here, so a HEAD that moved during
    the fall-through is still caught.  This is the case that proves the
    relaxation is about the RESOLVER's name and not about the commit.
    """

    moved = runner._runtime_source_identity_change(
        _checkout(), _editable(commit=_OTHER_COMMIT))
    assert moved is not None and "git_commit" in moved


def test_a_wheel_install_compares_equal_to_itself():
    """Both ends on the same rung, nothing resolved: still not a change."""

    assert runner._runtime_source_identity_change(_wheel(), _wheel()) is None


# ---------------------------------------------------------------------------
# Mode 2: the run's own output is not the implementation
# ---------------------------------------------------------------------------

def test_an_untracked_file_written_during_the_run_is_not_a_change():
    """A forecast writing into ``runs/`` failed its own receipt.

    ``runs/`` is not in ``.gitignore``, so the first ``wrfout`` frame a
    run writes appends a ``??`` row to ``git_status_short``.
    """

    launch = _checkout(status=[])
    end = _checkout(status=["?? runs/case/wrfout/wrfout_d01_2024-05-21"])
    assert runner._runtime_source_identity_change(launch, end) is None


def test_many_untracked_files_are_still_not_a_change():
    """A 72-hour forecast writes a directory full of them."""

    end = _checkout(status=[f"?? runs/case/wrfout/frame_{n:03d}"
                            for n in range(64)] + ["?? runs/case/report.json"])
    assert runner._runtime_source_identity_change(_checkout(), end) is None


def test_the_editable_untracked_count_is_not_a_change_either():
    """The same scratch file moves an editable identity twice.

    ``git_status_short`` gains a ``??`` row AND
    ``installed_editable.git.untracked_files`` increments.  Filtering
    only the first would leave the bug standing on that install shape.
    """

    launch = _editable(untracked_files=0, status=[])
    end = _editable(untracked_files=17,
                    status=["?? runs/case/wrfout/frame_000"])
    assert runner._runtime_source_identity_change(launch, end) is None


# ---------------------------------------------------------------------------
# The check must keep catching what it exists to catch
# ---------------------------------------------------------------------------

def test_a_moved_head_is_still_a_changed_implementation():
    """The other half of the bar: the reason not to delete the gate."""

    moved = runner._runtime_source_identity_change(
        _checkout(), _checkout(commit=_OTHER_COMMIT, tree=_OTHER_TREE))
    assert moved is not None
    assert "git_commit" in moved and _OTHER_COMMIT in moved


def test_a_moved_tree_is_still_a_changed_implementation():
    moved = runner._runtime_source_identity_change(
        _checkout(), _checkout(tree=_OTHER_TREE))
    assert moved is not None and "git_tree" in moved


def test_an_edited_hashed_source_is_still_a_changed_implementation():
    """And it is caught with EVERY git rung unresolved at both ends.

    ``source_sha256`` is read from bytes on disk and can never fail to
    resolve, so it is compared always and strictly.  It is what makes
    skipping the git comparison survivable.
    """

    edited = dict(_SOURCES,
                  **{"gpuwm/prepared_single_domain_forecast.py": "9999"})
    moved = runner._runtime_source_identity_change(
        _wheel(), _wheel(sources=edited))
    assert moved is not None
    assert "gpuwm/prepared_single_domain_forecast.py" in moved


@pytest.mark.parametrize("edited_module", sorted(_SOURCES))
def test_every_hashed_module_is_still_bound(edited_module):
    """All ten, not just the ones a test author happened to think of."""

    moved = runner._runtime_source_identity_change(
        _checkout(), _checkout(sources=dict(_SOURCES, **{edited_module: "0"})))
    assert moved is not None and edited_module in moved


def test_a_tracked_edit_outside_the_hashed_ten_is_still_a_change():
    """The filter drops ``??`` rows and nothing else.

    ``git_status_short`` is the only component that sees an edit to a
    tracked file this runner does not hash -- ``gpuwm/core/dycore.py``,
    say -- so filtering it must not cost that signal.
    """

    moved = runner._runtime_source_identity_change(
        _checkout(status=[]), _checkout(status=[" M gpuwm/core/dycore.py"]))
    assert moved is not None and "git_status_short" in moved


def test_a_tracked_edit_is_caught_beside_the_untracked_noise():
    """Both kinds of row at once: the noise must not mask the signal."""

    end = _checkout(status=[" M gpuwm/core/dycore.py",
                            "?? runs/case/wrfout/frame_000"])
    moved = runner._runtime_source_identity_change(_checkout(status=[]), end)
    assert moved is not None and "dycore.py" in moved


def test_the_editable_dirty_count_is_still_a_change():
    """``untracked_files`` is dropped; ``dirty_files`` is not."""

    moved = runner._runtime_source_identity_change(
        _editable(dirty_files=0), _editable(dirty_files=3))
    assert moved is not None and "installed_editable" in moved


def test_a_swapped_wheel_is_still_a_changed_implementation():
    """Both ends on the wheel rung, RECORD digest moved: pip rewrote it."""

    swapped = _wheel()
    swapped["installed_wheel"] = {"record_file_count": 812,
                                  "record_sha256": "dead"}
    moved = runner._runtime_source_identity_change(_wheel(), swapped)
    assert moved is not None and "installed_wheel" in moved


def test_the_refusal_names_the_component_that_moved():
    """A nine-word traceback with nothing actionable in it was the old
    receipt.  Every refusal here says WHICH component differs."""

    for before, after in (
        (_checkout(), _checkout(commit=_OTHER_COMMIT)),
        (_checkout(), _checkout(tree=_OTHER_TREE)),
        (_checkout(status=[]), _checkout(status=[" M gpuwm/core/nest.py"])),
        (_checkout(),
         _checkout(sources=dict(_SOURCES,
                                **{"gpuwm/io/wrfout.py": "zzzz"}))),
        (_editable(dirty_files=0), _editable(dirty_files=1)),
    ):
        moved = runner._runtime_source_identity_change(before, after)
        assert moved is not None and "->" in moved


# ---------------------------------------------------------------------------
# The published shape, which is hashed into checkpoint headers
# ---------------------------------------------------------------------------

def test_the_published_identity_shape_did_not_move():
    """These keys are digested into ``experiment_fingerprint``.

    The fingerprint is published in checkpoint headers, so a field added
    here would move every already-sealed fingerprint and strand the legs
    carrying them.  The fix improved the RELIABILITY of the resolution
    and the COMPARISON; it must not have added a ninth key.
    """

    assert set(runner._runtime_source_identity()) == {
        "identity_source", "git_commit", "git_tree", "git_status_short",
        "distribution_manifest_sha256", "installed_wheel",
        "installed_editable", "source_sha256"}


def test_the_real_identity_of_this_checkout_compares_equal_to_itself():
    """The gate, resolved twice for real, must pass."""

    first = runner._runtime_source_identity()
    second = runner._runtime_source_identity()
    assert runner._runtime_source_identity_change(first, second) is None


def test_the_identity_still_binds_all_ten_modules():
    """The digest set is the deciding half of the comparison."""

    identity = runner._runtime_source_identity()
    assert set(identity["source_sha256"]) == set(_SOURCES)


# ---------------------------------------------------------------------------
# The resolution's hardening
# ---------------------------------------------------------------------------

def test_a_transient_fallthrough_is_retried(monkeypatch, tmp_path):
    """One failed spawn is the whole incident; it must not be the answer.

    ``provenance`` cannot be made to raise on a flaky git -- it swallows
    the error and answers from a lower rung -- so the retry is driven by
    the rung it lands on, and a checkout is asked again.
    """

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    answers = [dict(_wheel()), dict(_checkout())]
    for answer in answers:
        answer.pop("source_sha256")
    calls: list[Path] = []

    def flaky(root, **kwargs):
        calls.append(root)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    monkeypatch.setattr("gpuwm.runtime_manifest.provenance", flaky)
    assert runner._runtime_provenance()["identity_source"] == "git"
    assert len(calls) == 2


def test_a_tree_with_no_repository_is_asked_once(monkeypatch, tmp_path):
    """A wheel install answers the same thing every time; do not pay
    three times to hear it."""

    monkeypatch.setattr(runner, "REPO", tmp_path)
    calls: list[Path] = []

    def resolved(root, **kwargs):
        calls.append(root)
        answer = dict(_wheel())
        answer.pop("source_sha256")
        return answer

    monkeypatch.setattr("gpuwm.runtime_manifest.provenance", resolved)
    assert runner._runtime_provenance()["identity_source"] == (
        "installed-wheel-record")
    assert len(calls) == 1


def test_a_checkout_answering_git_is_asked_once(monkeypatch, tmp_path):
    """The healthy path pays for no retry and no sleep."""

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    calls: list[Path] = []

    def resolved(root, **kwargs):
        calls.append(root)
        answer = dict(_checkout())
        answer.pop("source_sha256")
        return answer

    monkeypatch.setattr("gpuwm.runtime_manifest.provenance", resolved)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: 1 / 0)
    assert runner._runtime_provenance()["identity_source"] == "git"
    assert len(calls) == 1


def test_an_unbindable_tree_still_resolves_to_the_documented_shape(
        monkeypatch, tmp_path):
    """``IdentityError`` is not a crash; it is the lowest rung.

    The per-module digests still bind what ran, and the keys must be the
    ones every consumer of this mapping reads.
    """

    from gpuwm.runtime_manifest import IdentityError

    monkeypatch.setattr(runner, "REPO", tmp_path)

    def refusing(root, **kwargs):
        raise IdentityError("no identity")

    monkeypatch.setattr("gpuwm.runtime_manifest.provenance", refusing)
    identity = runner._runtime_provenance()
    assert identity["identity_source"] == "runtime-module-sha256-only"
    assert identity["git_commit"] is None
    assert set(identity) == {
        "identity_source", "git_commit", "git_tree", "git_status_short",
        "distribution_manifest_sha256", "installed_wheel",
        "installed_editable"}


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
    """A real one-commit checkout, standing in for ``REPO``."""

    root = tmp_path / "checkout"
    root.mkdir()
    _git(root.parent, "init", "-q", "-b", "main", str(root))
    (root / "model.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "model.py")
    _git(root, "commit", "-q", "-m", "first")
    monkeypatch.setattr(runner, "REPO", root)
    return root


@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_a_real_moved_head_is_reported_as_changed(repository):
    """End to end: two resolutions of a repository that really committed."""

    launch = dict(runner._runtime_provenance(), source_sha256=dict(_SOURCES))
    assert launch["identity_source"] == "git"
    assert launch["git_commit"] is not None

    (repository / "model.py").write_text("x = 2\n", encoding="utf-8")
    _git(repository, "commit", "-q", "-am", "second")
    end = dict(runner._runtime_provenance(), source_sha256=dict(_SOURCES))
    assert end["git_commit"] != launch["git_commit"]

    moved = runner._runtime_source_identity_change(launch, end)
    assert moved is not None and "git_commit" in moved


@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_a_real_untracked_file_is_not_reported_as_changed(repository):
    """Mode 2 against real ``git status --short`` output.

    The reduced tests above assert the shape; this asserts that git
    really does emit ``??`` rows for a run's scratch output, and that
    the filter really does read them.
    """

    launch = dict(runner._runtime_provenance(), source_sha256=dict(_SOURCES))
    (repository / "wrfout_d01_2024-05-21_00").write_text("f\n",
                                                         encoding="utf-8")
    end = dict(runner._runtime_provenance(), source_sha256=dict(_SOURCES))

    assert end["git_status_short"] != launch["git_status_short"], (
        "the premise: an untracked file DOES move this identity here")
    assert any(str(line).startswith("??") for line in end["git_status_short"])
    assert runner._runtime_source_identity_change(launch, end) is None


@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_a_real_tracked_edit_is_reported_as_changed(repository):
    """The same real repository, edited rather than littered."""

    launch = dict(runner._runtime_provenance(), source_sha256=dict(_SOURCES))
    (repository / "model.py").write_text("x = 3\n", encoding="utf-8")
    end = dict(runner._runtime_provenance(), source_sha256=dict(_SOURCES))

    moved = runner._runtime_source_identity_change(launch, end)
    assert moved is not None and "git_status_short" in moved


# ---------------------------------------------------------------------------
# The rung's own artifact is not comparable across a changed rung
# ---------------------------------------------------------------------------

def test_a_wheel_record_is_not_measured_against_an_editable_tree():
    """Three rungs answer three different questions.

    ``installed_wheel`` describes what pip wrote, ``installed_editable``
    describes a source tree, ``distribution_manifest_sha256`` describes
    a sealed manifest.  A fall-through swaps which one is populated, and
    comparing the populated one against the absent one -- or against a
    different rung's -- reports a change that means nothing.
    """

    assert runner._runtime_source_identity_change(
        _wheel(), _editable()) is None
    assert runner._runtime_source_identity_change(
        _editable(), _wheel()) is None


def test_the_rungs_own_artifact_is_still_compared_within_one_rung():
    """The relaxation is per-rung, not a deletion.

    Two ends on the SAME rung still bind that rung's artifact strictly,
    which is the whole of the check on a wheel install -- where there is
    no commit to compare at all.
    """

    swapped = _wheel()
    swapped["installed_wheel"] = {"record_file_count": 812,
                                  "record_sha256": "dead"}
    moved = runner._runtime_source_identity_change(_wheel(), swapped)
    assert moved is not None and "installed_wheel" in moved


# ---------------------------------------------------------------------------
# The git half must describe REPO and no other tree
# ---------------------------------------------------------------------------

@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_a_linked_worktree_binds_its_own_head_not_the_main_checkouts(
        tmp_path, monkeypatch):
    """The gap this runner sits in: agents work in linked worktrees.

    ``pip install -e`` points at the MAIN checkout, so when
    ``provenance``'s git branch falls through, the editable rung answers
    with the main checkout's HEAD -- a different tree from the one whose
    bytes ``source_sha256`` just hashed.  Measured against the real
    resolver before this was fixed: the worktree executing was
    ``eada530d`` and the fall-through answered ``c1b32d0``, so the two
    ends of a healthy run compared two unrelated commits and the gate
    fired on a run in which nothing changed.

    ``.git`` is a FILE in a linked worktree; the reader must follow it.
    """

    main = tmp_path / "main"
    main.mkdir()
    _git(tmp_path, "init", "-q", "-b", "main", str(main))
    (main / "model.py").write_text("x = 1\n", encoding="utf-8")
    _git(main, "add", "model.py")
    _git(main, "commit", "-q", "-m", "first")

    linked = tmp_path / "linked"
    _git(main, "worktree", "add", "-q", "-b", "side", str(linked))
    (linked / "model.py").write_text("x = 2\n", encoding="utf-8")
    _git(linked, "commit", "-q", "-am", "on the branch")

    main_head = _git(main, "rev-parse", "HEAD")
    linked_head = _git(linked, "rev-parse", "HEAD")
    assert main_head != linked_head, "the premise: two trees, two commits"
    assert (linked / ".git").is_file(), "a linked worktree's .git is a file"

    monkeypatch.setattr(runner, "REPO", linked)
    assert runner._repo_head_identity()["git_commit"] == linked_head


def test_a_fallthrough_commit_is_replaced_by_the_running_trees_own(
        monkeypatch, tmp_path):
    """The repair, driven through ``_runtime_provenance``.

    A rung that answered about somebody else's tree must not leave that
    tree's commit in the identity of THIS run.
    """

    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(runner, "REPO", tmp_path)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    stranger = dict(_editable(commit=_OTHER_COMMIT))
    stranger.pop("source_sha256")
    monkeypatch.setattr("gpuwm.runtime_manifest.provenance",
                        lambda root, **kwargs: stranger)
    monkeypatch.setattr("gpuwm.provenance.git_dir_identity",
                        lambda root, **kwargs: {"commit_full": _COMMIT})

    identity = runner._runtime_provenance()
    assert identity["git_commit"] == _COMMIT, "REPO's own HEAD, not the rung's"
    assert identity["git_tree"] is None, "the tree id needs the commit object"
    assert identity["git_status_short"] is None, "no git, no working tree"
    assert identity["identity_source"] == "installed-editable-source", (
        "the rung that answered is still named exactly")


def test_a_non_checkout_keeps_the_rungs_answer(monkeypatch, tmp_path):
    """No ``.git``, no better answer: do not overwrite a good one.

    A wheel install, or an editable one whose source tree is elsewhere,
    has nothing for the repair to read.  Blanking its commit would lose
    information rather than correct it.
    """

    monkeypatch.setattr(runner, "REPO", tmp_path)
    answer = dict(_editable(commit=_OTHER_COMMIT))
    answer.pop("source_sha256")
    monkeypatch.setattr("gpuwm.runtime_manifest.provenance",
                        lambda root, **kwargs: answer)
    assert runner._runtime_provenance()["git_commit"] == _OTHER_COMMIT


# ---------------------------------------------------------------------------
# End to end, against the real resolver in the real checkout
# ---------------------------------------------------------------------------

@pytest.mark.skipif(git_executable() is None, reason="git cannot be launched")
def test_a_real_git_outage_across_a_real_run_is_not_a_change(monkeypatch):
    """Mode 1 with nothing faked but the outage itself.

    The identity is resolved for real at both ends of this checkout,
    with ``git_executable`` returning None for the second -- which is
    exactly what a git that cannot be spawned looks like to every rung
    at once.  Five fields move; none of them is the implementation.
    """

    launch = runner._runtime_source_identity()
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    monkeypatch.setattr("gpuwm.provenance.git_executable", lambda **k: None)
    end = runner._runtime_source_identity()

    assert launch != end, (
        "the premise: the shipped `!=` gate DOES fire on this, which is "
        "how a four-hour forecast lost its receipt to a git hiccup")
    assert end["identity_source"] != launch["identity_source"]
    assert runner._runtime_source_identity_change(launch, end) is None
    assert end["git_commit"] == launch["git_commit"], (
        "and the commit binding survives the outage rather than lapsing")
