"""The private-material scan, tested in the direction where it goes quiet.

``tests/test_excluded_subsystem_absent.py`` is the gate that keeps internal
material off every public surface.  Its own docstring calls the commit-message
scan a release blocker and says the other three scans structurally cannot
replace it.

The 2026-08-28 fault-injection audit attacked the SCAN rather than the tree,
and the scan did not defend itself:

* ``F15`` -- point ``CLEAN_BASE`` at a SHA that is not in the clone, which is
  what a fresh clone, a shallow clone or a rebuilt branch produces.  The
  branch-diff scan and the commit-message scan each take their own
  ``pytest.skip`` and the file still exits 0::

      clean:    5 passed                rc=0
      injected: 2 passed, 3 skipped     rc=0     <- NOT CAUGHT

  Nothing in ``test_module_skip_placement.py``,
  ``test_gpu_marker_discipline.py`` or ``test_ci_test_replay.py`` noticed.

* ``F22`` -- narrow the Tier-1 literal set to its first entry
  (``for t in _TIER1[:1]``), leaving the regex patterns in place.  Five
  passed, INCLUDING the module's own falsifiability control, because that
  control exercises one token composition which survives the narrowing.

So the scan protecting private material can be silently switched off by an
absent base ref, or silently narrowed to one token, and the release battery
reads green either way.

This file is the tripwire for both.  It asserts nothing about the tree; it
asserts that the instrument is armed.  It is deliberately a separate module
so that a failure here reads as "the scan is not working" rather than "the
tree is dirty" -- two findings with two different responses.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest


def _load_scan_module():
    """Load the scan module BY PATH, from beside this file.

    ``from tests import test_excluded_subsystem_absent`` resolves to an
    unrelated ``tests`` package in site-packages on the cut box, which is the
    same cwd/site-packages shadowing that ARWEN-ORIENTATION section 9
    records for ``gpuwm`` itself.  A tripwire that tests the wrong module is
    worse than no tripwire, so the path is explicit and asserted.
    """

    path = pathlib.Path(__file__).resolve().parent / \
        "test_excluded_subsystem_absent.py"
    assert path.is_file(), f"{path} is missing; the scan it arms is gone"
    name = "gpuwm_excluded_subsystem_scan_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert pathlib.Path(module.__file__).resolve() == path, (
        f"loaded {module.__file__} instead of {path}")
    return module


scan = _load_scan_module()


def _object_present(rev: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", rev + "^{commit}"],
        cwd=scan.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def test_the_clean_base_is_present_in_this_clone() -> None:
    """F15.  Three of the five scans skip themselves without it.

    A skip is the right behaviour for a scan that cannot run.  A file that
    exits 0 while three of its five scans skipped is not: the release
    acceptance check reads the exit code.  This is the assertion that turns
    a silent skip into a named failure, and it belongs outside the scan
    module so that it cannot itself be skipped by the same missing object.
    """

    assert _object_present(scan.CLEAN_BASE), (
        f"tests/test_excluded_subsystem_absent.py pins CLEAN_BASE = "
        f"{scan.CLEAN_BASE} and this clone does not contain that commit.  "
        "Its branch-diff scan, its generated-document control and its "
        "COMMIT MESSAGE scan all take pytest.skip when the object is "
        "absent, so that file exits 0 with the release blocker unevaluated. "
        " Fetch the base, or re-pin CLEAN_BASE to the commit this branch "
        "was actually re-applied onto and say which in the commit message.  "
        "Do not run a release cut from a clone that cannot resolve it.")


def test_the_clean_base_is_an_ancestor_of_head() -> None:
    """A base that resolves but is off this history diffs the wrong thing."""

    if not _object_present(scan.CLEAN_BASE):
        pytest.fail("CLEAN_BASE is absent; see the test above")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", scan.CLEAN_BASE, "HEAD"],
        cwd=scan.ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode == 0, (
        f"CLEAN_BASE {scan.CLEAN_BASE} is not an ancestor of HEAD.  "
        "`git diff CLEAN_BASE..HEAD` then reports the difference between two "
        "unrelated tips rather than what this branch added, and the "
        "branch-diff scan is measuring the wrong thing while passing.")


@pytest.mark.parametrize("index", range(len(scan._TIER1)))
def test_every_tier1_literal_is_individually_detected(index: int) -> None:
    """F22.  One parametrised case per literal, so narrowing the set fails.

    The scan's own control composes ONE token and asserts it is caught.
    Narrowing ``_TIER1`` to its first entry leaves that control passing and
    seven names undetected.  Driving every entry is the difference between
    a control and a demonstration.
    """

    token = scan._TIER1[index]
    line = f"    value = compute_{token}_bank(state)"
    assert scan._tier1_match(line) is not None, (
        f"Tier-1 literal #{index} is no longer detected by the scan.  It is "
        "one of the names that must not reach a public surface, and the "
        "scan now passes a line containing it.  Either _TIER1 was narrowed, "
        "or _TIER1_RE stopped being built from all of it.")


@pytest.mark.parametrize("index", range(len(scan._TIER1_PATTERNS)))
def test_every_tier1_pattern_is_individually_detected(index: int) -> None:
    """The two pattern forms, which the literal set does not cover.

    These exist because a name-only filter was measured to be unsafe.  A
    pattern silently dropped from the tuple is the same hole as a literal
    silently dropped from the list.
    """

    pattern = scan._TIER1_PATTERNS[index]
    # Pattern 1 is word-bounded and case-SENSITIVE, so the sample has to put
    # the symbol between non-word characters: an underscore is a word
    # character and would defeat the boundary.
    # Spelled in HALVES joined at runtime, per the scanned file's own
    # rule: a file that arms a ban on a string must not itself carry the
    # string, or the tree and diff scans catch their own arming fixture.
    sample = {0: "    " + "_w" + "x_" + "ledger = {}",
              1: "    mass = " + "Ag" + "I" + " * rho_air"}[index]
    assert pattern.search(sample) is not None, (
        f"the sample this test pins for Tier-1 pattern #{index} no longer "
        "matches it; the pattern changed and this test's sample did not.  "
        "Update the sample to one the new pattern is meant to catch, in the "
        "same commit, and say what the pattern now covers.")
    assert scan._tier1_match(sample) is not None, (
        f"Tier-1 pattern #{index} matches its own sample but _tier1_match "
        "does not report it, so the pattern is registered somewhere the "
        "scan no longer consults.")


def test_the_tier1_matcher_is_not_matching_everything() -> None:
    """The other direction.  A matcher that fires on anything is not a gate.

    Without this, every assertion above could be satisfied by
    ``_tier1_match = lambda line: "yes"``.
    """

    innocent = [
        "    return self.domain.nest_ratio * dx_km",
        "def test_the_render_layout_is_nested_not_flat():",
        "    # the forecast hour the analysis frame is committed at",
        "    checkpoint = Path(run_directory) / 'restart.nc'",
    ]
    for line in innocent:
        assert scan._tier1_match(line) is None, (
            f"the Tier-1 matcher reports ordinary source as an offender: "
            f"{line!r}.  A scan that fires on everything gets switched off "
            "by the next person who has to ship, which is how gates die.")


def test_the_false_positive_mask_still_excuses_only_the_innocent_word() -> None:
    """The mask exists because one English word contains a Tier-1 token.

    If the mask widened, the scan would go quiet on real names without any
    literal being removed -- a narrowing that no count would show.
    """

    innocent = "the forecast hour is committed early"
    assert scan._tier1_match(innocent) is None
    for token in scan._TIER1:
        assert scan._tier1_match(f"x = {token}_bank") is not None, (
            f"the false-positive mask now excuses {token!r}; it is meant to "
            "excuse one innocent English word and nothing else")
