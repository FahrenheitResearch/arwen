"""Every unmerged local branch has a row in the dispositions ledger.

THE BREAKAGE THIS GATE PREVENTS: finished work stranded on unmerged
branches (2026-08-31 sweep; lane/engine-per-time-decode).  That sweep
triaged 198 local branches not merged into ``integration/release-2.5.0``
and what it found was not clutter, it was shipped defects whose finished
fixes were sitting complete and invisible.  The continuous-nowcast
daemon called its observation stage without a required keyword and died
with a TypeError at the first observation of every cycle -- while the
fix plus its regression test sat done at 64b728a60 on a branch nothing
pointed at.  The only implementation of the categorical supersample
derivation sat on a branch while the release line's own fine-mesh
receipt admitted "Nothing here demonstrates a delivered sub-kilometre
cell".  lane/engine-per-time-decode was found the same day, mid-strand,
and set the pattern for the class: work finishes on a branch, the branch
never merges, and no mechanism notices.

The mechanism this file adds: a local branch that is not merged into
``integration/release-2.5.0`` must either merge or carry a row in
``docs/branch-dispositions.md`` saying, in a five-word vocabulary, why
it never will.  A branch with neither is a RED, and the failure lists
the branch names so the fix is one ledger row per name.

Deliberate non-checks, so the gate stays cheap to obey:

* STALE ROWS ARE ALLOWED.  A row whose branch has since been deleted is
  not a failure: deleting a folded branch is exactly the cleanup this
  ledger exists to encourage, and failing on the leftover row would
  punish it.  The row stays as the record of where the content went.
* The note column is free text.  The gate audits the accounting, not
  the prose; per-branch prose accuracy is the ledger reader's problem.

The gate skips, naming why, when git is not on PATH, when this tree is
not a checkout (an sdist or wheel install has no branches to strand),
and when ``integration/release-2.5.0`` does not exist locally (a
mergedness question needs the thing to be merged into).  Enumeration
uses ``git branch --format=%(refname:short) --no-merged <line>``, which
neither prefixes the current branch with ``*`` nor cares what HEAD is,
so the gate reads the same from any worktree branch or from a detached
HEAD.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "branch-dispositions.md"

#: The untracked private sidecar.  Some local branches belong to an
#: internal subsystem whose identifying vocabulary the containment
#: scans (tests/test_excluded_subsystem_absent.py, the senior gate) ban
#: from every tracked line of this branch -- so their rows cannot live
#: in the tracked ledger at all.  They are not dropped: the sidecar
#: carries them in full on the development machine, this gate reads it
#: beside the tracked ledger, and .gitignore keeps it out of the tree.
#: On a clone without those local branches the sidecar is unnecessary;
#: where the branches exist and the sidecar is missing, the main gate
#: goes RED by exactly those branches, which is the accounting working.
SIDECAR = ROOT / "docs" / "branch-dispositions-private.md"


def all_rows() -> dict[str, str]:
    """The tracked ledger's rows plus the private sidecar's, disjoint."""

    rows = parse_ledger(LEDGER.read_text(encoding="utf-8"))
    if SIDECAR.is_file():
        for branch, disposition in parse_ledger(
                SIDECAR.read_text(encoding="utf-8")).items():
            if branch in rows:
                raise ValueError(
                    f"branch {branch!r} has a row in BOTH the tracked "
                    "ledger and the private sidecar; one branch, one row")
            rows[branch] = disposition
    return rows
RELEASE_LINE = "integration/release-2.5.0"

#: The concrete breakage, quoted in every RED so the failure is not an
#: abstract policy violation but the named class of loss it prevents.
BREAKAGE = ("finished work stranded on unmerged branches "
            "(2026-08-31 sweep; lane/engine-per-time-decode)")

#: The whole vocabulary.  Three literals plus two parameterised forms:
#: ``superseded-by-<sha>`` must carry at least 7 hex characters so the
#: replacement is a resolvable commit and not hand-waving, and
#: ``parked-<owner>`` must carry a non-empty owner token so parked work
#: has someone who decides its future.
_LITERAL_DISPOSITIONS = frozenset({
    "merged-content-elsewhere",
    "spent-probe",
    "active-lane",
})
_SUPERSEDED = re.compile(r"^superseded-by-[0-9a-f]{7,40}$")
_PARKED = re.compile(r"^parked-[a-z0-9][a-z0-9._-]*$")

_VOCABULARY_HELP = ("merged-content-elsewhere | superseded-by-<sha> | "
                    "spent-probe | parked-<owner> | active-lane")


# ---------------------------------------------------------------------------
# the pure half: parsing and verdicts, no git, unit-testable RED path
# ---------------------------------------------------------------------------

def disposition_is_valid(disposition: str) -> bool:
    return (disposition in _LITERAL_DISPOSITIONS
            or _SUPERSEDED.match(disposition) is not None
            or _PARKED.match(disposition) is not None)


def parse_ledger(text: str) -> dict[str, str]:
    """Strictly parse the ledger's pipe table into ``{branch: disposition}``.

    Strict means malformed rows raise instead of being skipped: a row
    that silently fails to parse is a branch that silently loses its
    accounting, which is the exact defect this file guards.  Rows are
    lines starting with ``|``; the header row and the ``|---|`` rule are
    the only two shapes excused.
    """

    rows: dict[str, str] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(cell and set(cell) <= set("-: ") for cell in cells):
            continue  # the |---|---|---| rule under the header
        if cells == ["branch", "disposition", "note"]:
            continue  # the header row itself
        if len(cells) != 3:
            raise ValueError(
                f"docs/branch-dispositions.md line {lineno}: expected "
                f"3 cells (branch | disposition | note), got {len(cells)}: "
                f"{stripped!r}")
        branch, disposition, _note = cells
        if not branch:
            raise ValueError(
                f"docs/branch-dispositions.md line {lineno}: empty branch "
                f"cell in {stripped!r}")
        if branch in rows:
            raise ValueError(
                f"docs/branch-dispositions.md line {lineno}: duplicate row "
                f"for branch {branch!r}; one branch, one disposition")
        rows[branch] = disposition
    return rows


def unrowed_branches(branches: list[str], rows: dict[str, str]) -> list[str]:
    """The RED set: unmerged branches the ledger does not account for."""

    return sorted(branch for branch in branches if branch not in rows)


def strand_message(missing: list[str]) -> str:
    """The failure text for the RED path, naming breakage and branches."""

    listing = "\n".join(f"  - {branch}" for branch in missing)
    return (
        f"RED: {BREAKAGE}.\n"
        f"{len(missing)} local branch(es) are not merged into "
        f"{RELEASE_LINE} and have no row in docs/branch-dispositions.md, "
        f"so anything finished on them is invisible to the release "
        f"line:\n{listing}\n"
        f"Merge the branch, or add a row to the ledger with one "
        f"disposition from: {_VOCABULARY_HELP}.")


# ---------------------------------------------------------------------------
# the git half: enumeration, skip-aware
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   stderr=subprocess.DEVNULL)


def _unmerged_branches() -> list[str]:
    try:
        _git("rev-parse", "--show-toplevel")
    except FileNotFoundError:
        pytest.skip("git executable not on PATH; branch dispositions "
                    "cannot be enumerated")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (installed copy or exported "
                    "tree); there are no local branches to strand")
    try:
        _git("rev-parse", "--verify", "--quiet", RELEASE_LINE + "^{commit}")
    except subprocess.CalledProcessError:
        pytest.skip(f"{RELEASE_LINE} does not exist locally; mergedness "
                    f"into it cannot be measured")
    out = _git("branch", "--format=%(refname:short)", "--no-merged",
               RELEASE_LINE)
    return [line.strip() for line in out.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def test_ledger_exists() -> None:
    """Deleting the ledger must not delete the accounting.

    Without this, removing docs/branch-dispositions.md would turn the
    main gate into a vacuous pass over an empty row set only when the
    parse happened to survive; it fails here first, by name.
    """

    assert LEDGER.is_file(), (
        f"RED: {BREAKAGE}. docs/branch-dispositions.md is missing, so no "
        f"unmerged branch is accounted for at all.")


def test_every_ledger_disposition_is_from_the_vocabulary() -> None:
    """An unknown disposition is a row that accounts for nothing.

    A typo like ``merged-elsewhere`` or a bare ``superseded-by-`` reads
    fine to a human and means nothing to the gate, so the row would
    green-light a branch while saying nothing checkable about it.  The
    failure names the branch and the bad disposition.
    """

    rows = all_rows()
    assert rows, ("docs/branch-dispositions.md parsed to zero rows; the "
                  "table body is gone or malformed")
    bad = {branch: disposition for branch, disposition in rows.items()
           if not disposition_is_valid(disposition)}
    assert not bad, (
        "docs/branch-dispositions.md carries dispositions outside the "
        f"vocabulary ({_VOCABULARY_HELP}); superseded-by needs >=7 hex "
        f"sha characters and parked- needs an owner token: {bad!r}")


def test_every_unmerged_branch_has_a_disposition_row() -> None:
    """The gate itself: no unmerged branch without an accounting row.

    Stale rows are deliberately NOT failed here (see the module
    docstring): the comparison is one-directional, ledger must cover
    git, never the reverse.
    """

    branches = _unmerged_branches()
    rows = all_rows()
    missing = unrowed_branches(branches, rows)
    if missing:
        pytest.fail(strand_message(missing))


# ---------------------------------------------------------------------------
# the gate's own falsifiability: prove RED fires, without touching git
# or the ledger on disk
# ---------------------------------------------------------------------------

def test_red_path_names_the_breakage_and_the_branches() -> None:
    """A gate that cannot go RED does not exist; this proves it can.

    Pure-function rehearsal of the failure: two unmerged branches, a
    ledger that rows only one, and the resulting message must carry the
    named breakage, the missing branch, the vocabulary for the fix, and
    must not implicate the branch that IS accounted for.
    """

    branches = ["lane/example-accounted", "lane/example-stranded"]
    rows = {"lane/example-accounted": "active-lane"}
    missing = unrowed_branches(branches, rows)
    assert missing == ["lane/example-stranded"]
    message = strand_message(missing)
    assert BREAKAGE in message
    assert "lane/example-stranded" in message
    assert "lane/example-accounted" not in message
    assert _VOCABULARY_HELP in message


def test_stale_rows_do_not_trip_the_gate() -> None:
    """The reverse direction stays green by design.

    A row for a deleted branch is the trace of a completed fold; the
    gate must not turn branch deletion into a suite failure.
    """

    branches = ["lane/example-live"]
    rows = {"lane/example-live": "active-lane",
            "lane/example-deleted-after-fold": "merged-content-elsewhere"}
    assert unrowed_branches(branches, rows) == []


@pytest.mark.parametrize("disposition,valid", [
    ("merged-content-elsewhere", True),
    ("spent-probe", True),
    ("active-lane", True),
    ("parked-drew", True),
    ("parked-salvage", True),
    ("superseded-by-17cf943ef", True),
    ("superseded-by-" + "a" * 40, True),
    ("superseded-by-12345", False),      # 5 hex chars is not a citation
    ("superseded-by-", False),
    ("parked-", False),                  # parked work needs an owner
    ("merged-elsewhere", False),         # the plausible typo
    ("landed", False),
    ("", False),
])
def test_vocabulary_boundary(disposition: str, valid: bool) -> None:
    """The validator rejects near-misses, both parameterised forms."""

    assert disposition_is_valid(disposition) is valid


def test_parser_rejects_a_malformed_row() -> None:
    """A row the parser skipped would be a branch silently unaccounted.

    Both malformations that matter: wrong cell count (a lost pipe) and
    a duplicate branch (two dispositions, no single answer).
    """

    with pytest.raises(ValueError, match="expected 3 cells"):
        parse_ledger("| lane/example-lost-a-pipe | active-lane |\n")
    with pytest.raises(ValueError, match="duplicate row"):
        parse_ledger("| lane/example-twice | active-lane | first |\n"
                     "| lane/example-twice | spent-probe | second |\n")
