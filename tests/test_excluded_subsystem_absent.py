"""This branch must not silently regain the excluded subsystem.

WHAT THIS BRANCH IS.  ``integration/engine-258-clean`` was built by
RE-APPLYING engine work onto a clean base, not by scrubbing a branch that
carried an excluded weather-modification subsystem.  That distinction is
the whole point: a branch that never contained the thing cannot leak it,
while a scrubbed branch is only ever as good as the scrub.  This file is
what keeps the first sentence true after today.

THE CONCRETE BREAKAGE IT PREVENTS, and it is not hypothetical -- it is
what the extraction was done to avoid.  Somebody merges, cherry-picks or
hand-copies a change from the branch this work came from, or from one of
the lanes still cut off it.  The change looks like ordinary engine work in
review, because the excluded parts of those files are small, guarded and
interleaved with real physics.  It lands.  Later this branch is cut as a
public 2.5.8 release, and the release ships a subsystem that was never
supposed to leave this machine.  Nothing before this file would have
failed: every other test still passes, because the excluded code is
presence-based and inert until switched on.

WHY IT IS A NAME SCAN AND NOT A CLEVERER TEST.  There is no import edge to
follow.  Every reference from an engine file into the excluded subsystem
was function-local and runtime-guarded, so a dependency graph sees nothing.
What DOES identify it, unambiguously and mechanically, is a small set of
names.  That is what is checked.

TWO TIERS, because the names are not equally safe to ban:

  Tier 1 -- names that identify the subsystem and appear NOWHERE ELSE in
  this repository's own sources.  Measured, not assumed: when this file was
  written the scope below contained exactly zero of them.  Banned in
  content and in paths.  Two Tier-1 entries were NOT on the original
  exclusion list and are the more important half -- the subsystem also
  spells itself with abbreviated local identifiers and with a bare chemical
  symbol, and 111 added lines across 35 files carried it invisibly to a
  scan of the listed names alone.  Three of those lines were live
  executable code in files a name-only filter would have cleared.

  Tier 2 -- ordinary English that happens to be on the list.  A weather
  model seeds an RNG, seeds inflow turbulence and seeds a first-guess
  field; there are several hundred innocent occurrences here, and no
  allowlist of them would survive contact with the next contributor.
  Banned in PATHS only, where the innocent uses are two known files, and
  in the branch diff below, where there is no innocent-vocabulary problem
  to trade off.

  And one pure false positive, handled by masking rather than by
  exception: the ordinary word for what this model produces CONTAINS one of
  the listed tokens.  188 apparent hits in the original survey were that
  one word.  Counting them would have made this gate cry wolf until
  somebody deleted it.

WHY THE NAMES BELOW ARE SPELLED IN HALVES.  A file that bans a string
cannot contain that string, or it fails its own gate -- and the release
acceptance check greps the whole branch diff, which includes this file.
The halves are joined at import time.  Do not "tidy" them back into
literals.
"""

from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The clean base this branch was re-applied onto.  Used only by the
#: diff scan, which skips itself when the object is not present.
CLEAN_BASE = "f27fc897ab88a6f7e5cfc59d3573f3285aeda9fd"

#: Vendored third-party trees, excluded from every scan here.  They carry
#: innocent matches -- an OpenSSL assembly comment in `ring`, and one
#: module of the `rand` crate's RNG adapters whose FILENAME is one of the
#: ordinary-English names in Tier 2 below (naming that file here would
#: trip this module's own diff scan, which is the point of Tier 2).
#: Nobody edits either tree in this repository.
VENDOR = ("tools/rustwx/vendor/", "tools/rw_wps/vendor/")

#: The release exclusion manifest, read so this gate's idea of "what a
#: public release carries" cannot drift from the builder's.
#: work/build_release_snapshot.py cuts the snapshot as `git archive HEAD`
#: minus every path this file names, so the content scan below asks about
#: exactly the set of bytes a stranger would receive -- no more (CLAUDE.md
#: and the campaign records under docs/superpowers/, handoffs/ and
#: evidence/ are dropped by the manifest and legitimately still discuss
#: the excluded subsystem) and no less (README.md, CHANGELOG.md,
#: PROVENANCE.md, docs/ and runs/ all ship, and an earlier revision of
#: this gate read none of them).
RELEASE_EXCLUDE = ROOT / "RELEASE-EXCLUDE.txt"

#: Tier 1, spelled in halves (see the module docstring).
_TIER1 = tuple("".join(parts) for parts in (
    ("wx", "mod"),        # the subsystem's own name
    ("qn", "ag", "i"),    # its state banks
    ("sil", "ver"),       # the agent, first word
    ("iod", "ide"),       # the agent, second word
    ("tape", "tum"),      # the reference tree it was ported from
    ("ag", "i_"),         # identifier forms of the agent abbreviation
    ("_ag", "i"),
    ("re", "cast"),       # the org prefix; see _FALSE_POSITIVE below
))

#: Tier 1 additions the original exclusion list did not name.  These are
#: the false negatives that made a name-only filter unsafe: local
#: variables written with a two-letter prefix form of the subsystem's own
#: name (a ledger, a key, a mode, an inert flag, a reason), and the bare
#: chemical symbol -- neither of which contains any of the listed forms.
_TIER1_PATTERNS = (
    # Only at the START of an identifier: a legitimate pre-existing name in
    # the Rust fetch crate embeds the same three characters mid-word.
    re.compile("(?<![A-Za-z0-9])" + "_wx" + "_"),
    # Case-SENSITIVE, unlike everything else here: the bare chemical
    # symbol, not every occurrence of those three letters.
    re.compile(r"\b" + "Ag" + "I" + r"\b"),
)

#: Tier 2, banned in paths and in the branch diff.
_TIER2 = tuple("".join(parts) for parts in (("seed", "ing"), ("seed", "ed")))

#: The two pre-existing paths that legitimately carry a Tier-2 name.
#: Pinned rather than pattern-matched, so a third one is a decision.
_TIER2_PATH_ALLOWANCE = frozenset({
    "tests/test_benchmark_" + "seed" + "ed_step.py",
    "tools/benchmark_" + "seed" + "ed_step.py",
})

#: Commit messages that name the subsystem and cannot be reworded: this
#: repository forbids rebase, amend and force-push (CLAUDE.md, forward
#: commits only) and never pushes to any remote, so the branch route this
#: module's message test governs is closed by law rather than by wording.
#: Pinned by full SHA with the reason, so a second entry is a decision.
_MESSAGE_ALLOWANCE = {
    "ccd11d67f31a6a26e9b6490182b046db16e991cb":
        "the 2026-09-01 law commit that sanctioned the private package; "
        "rewording is banned here and the branch is never pushed",
}

#: The innocent English word that contains a Tier-1 token.
_FALSE_POSITIVE = re.compile("fore" + "cast", re.I)

_TIER1_RE = re.compile("|".join(re.escape(t) for t in _TIER1), re.I)
_TIER2_RE = re.compile("|".join(re.escape(t) for t in _TIER2), re.I)

#: An added line longer than this is not hand-written source.  It is a
#: GENERATED, MINIFIED document -- gpuwm/physics_registry_v2.json is one
#: JSON line of about 250 kB -- and line granularity asks the wrong
#: question of it.  See ``_fresh_tier2_on_a_generated_line``.
_MINIFIED_LINE = 4096


def _mask(text: str) -> str:
    return _FALSE_POSITIVE.sub("FCST", text)


def _tier1_match(text: str) -> str | None:
    masked = _mask(text)
    found = _TIER1_RE.search(masked)
    if found is not None:
        return found.group(0)
    for pattern in _TIER1_PATTERNS:
        found = pattern.search(masked)
        if found is not None:
            return found.group(0)
    return None


def _tracked() -> list[str]:
    """Every tracked path except the vendored trees.

    The PATH scan uses this whole set deliberately.  A file NAMED for the
    excluded subsystem is unambiguous wherever it sits, and the places it
    would come back to -- the repository root (its port map and phase
    specs lived there), evidence/, runs/, handoffs/, work/ -- are exactly
    the ones the scope tuple that used to live here did not list.  A
    root-level file carrying the subsystem's own name passed this gate
    until this revision; that was proved, not supposed.
    """

    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [name for name in raw.decode("utf-8").split("\0")
            if name and not name.startswith(VENDOR)]


def _release_exclusions() -> list[str]:
    """The manifest's rules, or [] when it is not present."""

    try:
        lines = RELEASE_EXCLUDE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")]


def _excluded_from_release(name: str, rules: list[str]) -> bool:
    """``build_release_snapshot.matches()``, reimplemented in ten lines.

    Reimplemented rather than imported because the builder lives under
    ``work/``, which the manifest itself drops: inside an extracted
    snapshot there is nothing to import, and a gate that silently skips
    when it cannot find its helper is not a gate.
    """

    for rule in rules:
        if rule.endswith("/**"):
            root = rule[:-3]
            if name == root or name.startswith(root + "/"):
                return True
        elif name == rule:
            return True
    return False


def _release_surface() -> list[str]:
    """What a public snapshot would actually carry."""

    rules = _release_exclusions()
    return [name for name in _tracked()
            if not _excluded_from_release(name, rules)]


@functools.lru_cache(maxsize=None)
def _clean_base_text(path: str) -> str:
    """What ``path`` already contained at the clean base ('' when new)."""

    try:
        raw = subprocess.check_output(
            ["git", "show", f"{CLEAN_BASE}:{path}"],
            cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return raw.decode("utf-8", "replace")


def _fresh_tier2_on_a_generated_line(line: str, path: str) -> str | None:
    """Tier-2 vocabulary this branch INTRODUCES into a minified document.

    THE CONCRETE BREAKAGE THIS PREVENTS, and it is not hypothetical -- it
    is what made this carve-out necessary.  Tier 2 is ordinary English.
    Banning it outright in the diff is affordable for hand-written source
    because a human wrote every added line, which is the trade-off the
    module docstring records.  A GENERATED document breaks that reasoning
    structurally: gpuwm/physics_registry_v2.json is emitted minified, so
    its whole body is ONE line, and regenerating it -- which
    tools/build_registry.py requires and tests/test_build_registry.py
    holds byte-identical -- re-emits every innocent word the registry
    already carried at the clean base as a single added line.  The
    registry's own prose uses both Tier-2 words in their ordinary sense --
    snow placed at model levels in a water-budget experiment, and the
    layouts of it -- none of which has anything to do with the excluded
    subsystem, and all of which predates this branch.  They are not
    spelled here for the reason the module docstring gives.

    Without this, the FIRST required regeneration of the registry turns
    this gate red on vocabulary the branch did not write, the failure is
    unfixable without rewording published science prose, and the
    predictable response is to delete the gate -- which is exactly the
    cry-wolf failure the module docstring says the masking rule exists to
    avoid.

    So the question is narrowed on such a line, and ONLY on such a line,
    to the one this scan actually means: does the branch introduce a
    Tier-2 MENTION this document did not already carry?  Counted, not
    merely looked up by word, so a second unrelated mention in a document
    that already had one is still reported.

    WHY THIS LOSES NO COVERAGE.  The registry is not written; it is
    GENERATED, by tools/build_registry.py, from tables in that file.  Any
    new word in the artifact arrives as a hand-written line in the
    builder, and the builder is ordinary source that this same scan reads
    at ordinary line granularity.  The generated document does not need
    its own Tier-2 net; its source has one.  Tier 1 is untouched here
    either way and stays absolute on every line, generated or not --
    those names are never innocent.
    """

    carried: dict[str, int] = {}
    for found in _TIER2_RE.finditer(_mask(_clean_base_text(path))):
        word = found.group(0).lower()
        carried[word] = carried.get(word, 0) + 1

    seen: dict[str, int] = {}
    for found in _TIER2_RE.finditer(_mask(line)):
        word = found.group(0).lower()
        seen[word] = seen.get(word, 0) + 1
        # COUNTS, not vocabulary.  Excusing the WORD rather than the
        # OCCURRENCE would let a document that already carried a Tier-2
        # word once gain a second, unrelated mention in silence -- which
        # is a hole, and was caught by this module's own carve-out test.
        if seen[word] > carried.get(word, 0):
            return found.group(0)
    return None


def _is_checkout() -> bool:
    try:
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                                cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _is_checkout(), reason="the tracked-file scan needs a checkout")


def test_no_tracked_path_names_the_excluded_subsystem():
    """A file NAMED for it is the loudest way to regain it, and the
    cheapest to check: a path scan opens no file at all."""

    offenders = []
    for name in _tracked():
        if _tier1_match(name) is not None:
            offenders.append(name)
        elif (_TIER2_RE.search(_mask(name))
                and name not in _TIER2_PATH_ALLOWANCE):
            offenders.append(name)
    assert not offenders, (
        "these tracked paths name the excluded subsystem; this branch is "
        "the one a public 2.5.8 release is cut from and it must never "
        f"carry them: {offenders}")


def test_no_tracked_source_contains_the_excluded_subsystem():
    """The content scan.  Tier 1 only -- see the module docstring for why
    the ordinary-English half of the list is a path check instead."""

    offenders: list[str] = []
    for name in _release_surface():
        path = ROOT / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw[:8192]:
            continue  # a binary asset; a byte pattern there is not source
        text = raw.decode("utf-8", "replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            token = _tier1_match(line)
            if token is not None:
                offenders.append(f"{name}:{lineno}: {token!r}")
        if len(offenders) > 40:
            break
    assert not offenders, (
        "the excluded weather-modification subsystem is back in tracked "
        "source.  This is how it returns: a merge or a cherry-pick from "
        "the branch this work was extracted from, or from a lane still cut "
        "off it, carrying a few guarded lines that read as ordinary physics "
        "in review.  Nothing else in the suite would fail, because that "
        "code is inert until switched on -- and this branch is what a "
        "public release is cut from.  Re-apply the engine part of the "
        "change and leave the rest, the way this branch was built:\n  "
        + "\n  ".join(offenders))


def test_the_whole_branch_diff_against_its_clean_base_is_clean():
    """The release acceptance check, made permanent.

    The two scans above ask about the tree as it stands.  This asks the
    question the release asks: is everything this branch ADDED to its clean
    base still clean?

    TIER-1 ONLY, and every path -- shipped or not -- because Tier-1 names
    identify the subsystem unambiguously and have no innocent use here.

    Tier-2 was originally applied to this diff too, on the premise that
    "inside a diff of this branch's own additions there is no
    innocent-vocabulary trade-off to make".  MEASURED FALSE, 2026-08-30,
    five independent times in one day: mesh receipts print their point
    placement with the sowing word, the mesh crate's public placement
    type is named with it, the mutation gate's debt prose used it, the
    offline-downscale comments used it, and radiation test fixtures
    describe SUPERCOOLED CLOUD columns with it -- textbook cloud-physics
    vocabulary.  Ordinary physics English lands in this diff every week,
    and the file's own design note says what a crying-wolf gate becomes.
    Tier-2 keeps its PATH ban above; the subsystem's identifying names
    stay banned here in full.
    """

    try:
        diff = subprocess.check_output(
            ["git", "diff", CLEAN_BASE + "..HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the clean base is not present in this clone")

    offenders: list[str] = []
    current = ""
    for line in diff.decode("utf-8", "replace").splitlines():
        if line.startswith("+++ b/"):
            current = line[len("+++ b/"):]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        token = _tier1_match(line)
        if token is not None:
            offenders.append(f"{current}: {token!r} in: {line[:120]}")
            if len(offenders) > 40:
                break
    assert not offenders, (
        "a line this branch ADDS to its clean base names the excluded "
        "subsystem.  The release acceptance check greps exactly this "
        "diff:\n  " + "\n  ".join(offenders))
def test_the_generated_document_carve_out_still_catches_a_NEW_name():
    """The carve-out above, proved to still bite.

    An exemption nobody tests is an exemption that quietly becomes a hole.
    This asserts both halves of ``_fresh_tier2_on_a_generated_line`` on the
    real registry line: vocabulary the document ALREADY carried at the
    clean base is excused, and a Tier-2 word that is genuinely new is
    still reported -- on the same line, in the same call.
    """

    registry = "gpuwm/physics_registry_v2.json"
    carried = _clean_base_text(registry)
    if not carried or _TIER2_RE.search(_mask(carried)) is None:
        pytest.skip("the clean base carries no Tier-2 word in the registry")

    # The half that must stay quiet: the base text, re-offered as an added
    # line, introduces nothing.
    assert _fresh_tier2_on_a_generated_line(carried, registry) is None

    # The half that must still fire: one genuinely new Tier-2 word.
    intruder = "".join(("seed", "ing")) + "_hook_bank"
    assert _fresh_tier2_on_a_generated_line(
        carried + intruder, registry) is not None

    # And Tier 1 is untouched by the carve-out: it is checked before the
    # length test ever runs, so a Tier-1 name on a minified line is still
    # an offender no matter what the base carried.
    assert _tier1_match(carried + "".join(("wx", "mod"))) is not None


def test_no_new_commit_MESSAGE_names_the_excluded_subsystem():
    """The fourth surface, and the one the first three cannot see.

    A commit message is not a file.  It is not in the tree, so the path
    and content scans above never look at it, and it is not a ``+`` line,
    so the diff scan never looks at it either -- yet it travels with the
    commit permanently, and anyone who publishes this branch AS A BRANCH
    rather than as a snapshot publishes every word of it.

    That is not hypothetical here.  The commit that split this file's own
    token table into halves, so the table would stop matching itself,
    quoted the ASSEMBLED table in its message to record what the halves
    spell.  A second commit named an excluded constant while explaining
    which revision of a file it had deliberately stopped at.  Both were
    written by a scan that only ever read files.

    WHAT IS AND IS NOT AT RISK.  ``work/build_release_snapshot.py`` cuts
    the public snapshot as ``git archive HEAD`` minus RELEASE-EXCLUDE.txt
    -- a file tree, carrying no history -- so the snapshot route is
    unaffected and the three scans above are what govern it.  This test
    governs the other route: pushing the branch, publishing the repository,
    or cutting a release from git history.  This repository has a GitHub
    remote configured, so that route is one command away.

    REMEDIATION.  Rewording -- ``git rebase -r --exec`` over the range, or
    filter-repo -- is what git offers, and it is BANNED in this repository:
    CLAUDE.md allows forward commits only, and the same file forbids any push
    to any remote, which is the barrier the branch route needs.  A message
    that cannot be reworded is registered in ``_MESSAGE_ALLOWANCE`` by full
    SHA with its reason, and this test stays red on any message that is
    not.  Write commit messages on this line without the subsystem's name;
    the allowance is for the past, not a style.
    """

    try:
        log = subprocess.check_output(
            ["git", "log", "--format=%H%n%an <%ae>%n%cn <%ce>%n%B%n",
             CLEAN_BASE + "..HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the clean base is not present in this clone")

    offenders: list[str] = []
    current = ""
    current_full = ""
    for line in log.decode("utf-8", "replace").splitlines():
        if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
            current = line[:9]
            current_full = line
            continue
        if current_full in _MESSAGE_ALLOWANCE:
            continue
        token = _tier1_match(line)
        if token is not None:
            offenders.append(f"{current}: {token!r} in: {line.strip()[:110]}")
    assert not offenders, (
        "a commit message on this branch names the excluded subsystem.  "
        "Commit messages are not files: the path, content and diff scans "
        "in this module cannot see them, and they ship with the branch to "
        "anyone it is pushed to.  The snapshot builder is unaffected (it "
        "archives the tree, not the history); publishing the BRANCH is "
        "not.  These must be reworded before this branch is shared, which "
        "rewrites every SHA from the earliest one onward:\n  "
        + "\n  ".join(offenders))


def test_the_diff_scans_tier1_pattern_fires_on_a_code_line():
    """The whole diff rule is now Tier-1; prove the pattern bites raw code."""
    intruder = "let mass = " + "Ag" + "I" + " * rho;"
    assert _tier1_match(intruder) is not None
