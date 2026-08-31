#!/usr/bin/env python3
r"""Refuse a commit that would give an authored file a carriage return.

WHY THIS EXISTS, measured rather than supposed.  ``core.autocrlf`` is
``false`` in this clone and ``.gitattributes`` declares ``* -text``, so
git converts nothing in either direction -- and files kept flipping to
CRLF anyway.  Three commits on the branch this hook was written for
record it:

    fb9612475  "restore the CRLF line endings the previous commit's
                patch script flattened"
    b8c3aa5e9  "restore the LF line endings this lane's patch scripts
                flattened"
    62e65e1a6  wif-default -- flipped TEN files LF -> CRLF in one
                commit, +20,871 CR bytes, mentioned nowhere in it

The cause is not git.  It is the editing step.  A lane's patch script
reads a file in Python TEXT mode -- universal newlines collapse CRLF and
LF alike to "\n" -- edits a line or two, and writes it back with a plain
``open(path, "w")``.  On Windows that writer translates every "\n" to
"\r\n" on the way out, so a two-line edit rewrites the whole file's line
endings.  It is invisible in review, because the damage only reads as
"whole file changed" once it has already landed, and it is inherent to
the default text-mode writer on this platform rather than to any one
script.  The remedy is a gate at the moment of the mistake.

tests/test_line_ending_stability.py holds the same promise at RELEASE
time, against HEAD.  By then the bytes are landed history and the cost
is a whole-file merge conflict -- which is exactly how two merges went
on 2026-08-27.  This hook asks the same question of the INDEX instead,
so the answer arrives while the fix is still one sed.

The rule is a RATCHET, identical to the test's: a file that already
carries CR at HEAD may stay as it is, and a file that does not may not
start.  ``_is_authored`` is imported from the gate rather than restated,
so the hook and the gate cannot drift apart.

ARMED BY DEFAULT.  It was not, and that is why it stopped nothing.  Until
2026-08-29 the only way to install it was an ``--install`` flag named
nowhere but in this docstring, and nothing in the repository ran it:
``.git/hooks/pre-commit`` did not exist in this clone.  Both of the
incidents this hook describes were then committed straight through it --
``tests/test_offline_child.py`` rewritten to CRLF across all 673 lines
with a real 79-line addition buried in the whole-file diff (repaired by
39ef138c5), and ``fp32_floor_probe.rs`` born with 525 carriage returns
(repaired by abb5ff270).  A remedy behind a flag is a workaround.

So ``conftest.py`` at the repository root calls :func:`ensure_installed`
on every pytest session, and
``tests/test_line_ending_stability.py::test_the_pre_commit_hook_is_armed``
fails if that did not take -- a hook that is quietly not there is the
defect, so failing to arm has to be loud rather than absent.  A fresh
clone is armed by its first test run.

Arm by hand:  python tools/git_hooks/pre_commit_line_endings.py --install
Bypass:       git commit --no-verify   (for a deliberate CR -- then
              record it in the gate's ``_CRLF_DEBT``)
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "tests" / "test_line_ending_stability.py"


def _authority():
    """``_is_authored`` from the release gate itself -- one definition.

    A worktree can carry this script and NOT the gate: the hooks
    directory belongs to the clone, so arming it arms every worktree cut
    from that clone, and the shim's ``[ -f ]`` guard only asks whether
    this file is there.  Without the gate there is no definition of
    "authored", so the hook cannot judge -- and it says so and refuses,
    rather than dying in an import traceback that names nothing, and
    rather than exiting 0, which is the quiet non-gate this whole
    arrangement exists to remove.
    """
    if not GATE.is_file():
        raise SystemExit(
            "line endings: refusing this commit.\n\n"
            "This worktree carries tools/git_hooks/pre_commit_line_endings.py"
            " but not %s, which is where `_is_authored` -- the definition of"
            " which files this check covers -- lives.  The hook will not"
            " guess it: one definition or none.\n\n"
            "Restore that file, or commit with --no-verify.\n" % GATE)
    spec = importlib.util.spec_from_file_location("_line_ending_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True).stdout


def _staged_paths() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "-z",
               "--diff-filter=ACMR").decode("utf-8", "surrogateescape")
    return [entry for entry in out.split("\x00") if entry]


def _blob(rev: str, path: str) -> bytes | None:
    try:
        return _git("show", rev + ":" + path)
    except subprocess.CalledProcessError:
        return None


def normalizing(paths) -> set[str]:
    """The staged paths git NORMALIZES on the way into the index.

    ``.gitattributes`` runs two regimes.  ``* -text`` covers most of the
    tree: no conversion in either direction, so the index bytes are the
    disk bytes and asking the index is asking the file.  Twenty explicit
    ``text eol=lf`` rules -- ``gpuwm/authorities/*.json``,
    ``gpuwm/physics_registry_v2.json``, ``*.sh``, the FTZ receipt -- are
    the other regime: git strips the CR on the way IN, so a CRLF working
    tree stages CLEAN and every question put to the index or to a blob
    answers "fine" while the file on disk is not.

    ``text``, not ``eol``, because the file's LAST matching rule wins and
    only ``text`` decides whether git converts at all: 283 tracked paths
    carry ``eol=lf`` and 23 of them sit under a later ``-text``
    (``tools/rustwx/crates/**`` over ``*.sh``), so they never normalize.
    Asking ``eol`` would put those 23 on the wrong side.
    """
    paths = list(paths)
    if not paths:
        return set()
    asked = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "text"], cwd=ROOT,
        input="\x00".join(paths).encode("utf-8", "surrogateescape"),
        capture_output=True, check=True).stdout
    fields = asked.decode("utf-8", "surrogateescape").split("\x00")
    return {fields[i] for i in range(0, len(fields) - 2, 3)
            if fields[i + 2] == "set"}


def _blob_is_binary(blob: bytes) -> bool:
    """Is this staged content binary, by git's own test?

    A NUL byte in the leading window is how ``git grep -I`` and git's
    diff machinery decide, and the release gate's scan
    (``_cr_bearing_authored``) uses exactly that ``-I``.  The hook has
    to agree with the gate here or it refuses commits the gate can
    never adjudicate: rw-netcdf's checked-in NetCDF-4 test fixtures are
    HDF5, whose eight-byte signature CONTAINS ``\\r\\n`` by
    specification, and this hook refused them as fresh carriage returns
    while its message named a remedy -- record the path in
    ``_CRLF_DEBT`` -- that ``test_the_crlf_debt_does_not_rot`` forbids
    for a binary, because the ``git grep -I`` scan never reports one
    and the entry would read as settled on its first run.  A binary
    blob has no line endings to protect; ``* -text`` already keeps its
    bytes byte-identical in both directions.
    """
    return b"\x00" in blob[:8192]


def offenders() -> list[tuple[str, int, str]]:
    """``(path, CR count, where it came from)`` for staged fresh-CR files.

    Two questions, because the two ``.gitattributes`` regimes fail
    differently.  For the ``-text`` majority the INDEX is the file, and
    the rule is the gate's ratchet: CR at HEAD may stay, CR that was not
    there may not start.  For the ``text eol=lf`` set the index has
    already been normalized, so it is asked to say nothing useful --
    those paths are checked ON DISK, with no ratchet, because a CR there
    can never have been recorded in a blob and is always fresh.

    WHERE THIS HOOK STOPS, measured rather than assumed.  A commit hook
    can only ask about paths that are STAGED, and a CRLF-only flip of a
    normalized path stages nothing at all -- git converts it straight
    back, the index blob is identical to HEAD, and the path never appears
    in ``git diff --cached``.  That case belongs to
    ``test_no_normalized_file_carries_a_carriage_return_on_disk``, which
    reads the working tree directly.  What this branch catches is the
    shape that actually happened: a regenerator rewrites the file, so
    there IS a staged content change, and the CRLF rides in beside it on
    disk while the blob stays clean -- ``gpuwm/physics_registry_v2.json``
    exactly.
    """
    staged_now = _staged_paths()
    if not staged_now:
        return []      # nothing to judge, and importing the gate is 0.25 s
    gate = _authority()
    staged_paths = [p for p in staged_now if gate._is_authored(p)]
    normalized = normalizing(staged_paths)
    found = []
    for path in staged_paths:
        if path in normalized:
            disk = Path(ROOT / path)
            if not disk.is_file():
                continue
            raw = disk.read_bytes()
            if b"\r" in raw:
                found.append((path, raw.count(b"\r"),
                              "on disk; the index is normalized clean"))
            continue
        staged = _blob("", path)          # "" -> the index
        if staged is None or b"\r" not in staged:
            continue
        if _blob_is_binary(staged):
            continue                      # not text; the gate skips it too
        head = _blob("HEAD", path)
        if head is not None and b"\r" in head:
            continue                      # recorded, unchanged debt
        found.append((path, staged.count(b"\r"),
                      "new file" if head is None else "was LF at HEAD"))
    return found


def main() -> int:
    if "--print-shim" in sys.argv:
        sys.stdout.write(hook_shim())
        return 0
    if "--install" in sys.argv:
        return _install()
    found = offenders()
    if not found:
        return 0
    say = sys.stderr.write
    say("line endings: refusing this commit.\n\n")
    say("These staged files carry CR bytes and their HEAD versions do not."
        "  Under `* -text` the bytes in the object database are the bytes"
        " on disk, and the prepared runners hash their own sources -- a"
        " CRLF flip makes a clone and a wheel disagree about the same"
        " file.  Under `text eol=lf` the blob is normalized clean and only"
        " the WORKING TREE carries the CR, which splits the same hashes"
        " while every blob-reading gate stays green.\n\n")
    for path, count, origin in found:
        say("    %6d CR   %s   (%s)\n" % (count, path, origin))
    say("\nAlmost always this is a patch script that read the file in "
        "Python text mode and wrote it back with open(path, 'w'), which on "
        "Windows turns every LF into CRLF.  Write with newline='\\n', or "
        "in binary.\n\nFix and re-stage:\n\n")
    say("    python tools/git_hooks/pre_commit_line_endings.py --fix\n")
    say("    git add " + " ".join(path for path, _c, _origin in found) + "\n")
    say("\nIf the CR is deliberate, commit with --no-verify and record it "
        "in tests/test_line_ending_stability.py::_CRLF_DEBT -- but only "
        "for a `* -text` path, where the CR reaches the blob.  A path "
        "flagged \"on disk\" has a clean blob and belongs in no debt "
        "list; normalize the file.\n")
    return 1


def fix(paths) -> list[tuple[str, int]]:
    """CRLF -> LF in place, refusing anything that is not a pure CR strip."""
    done = []
    for path in paths:
        raw = Path(ROOT / path).read_bytes()
        flat = raw.replace(b"\r\n", b"\n")
        if raw.count(b"\r") != raw.count(b"\r\n"):
            raise SystemExit("%s carries a bare CR; not touching it" % path)
        assert raw.replace(b"\r", b"") == flat.replace(b"\r", b"")
        Path(ROOT / path).write_bytes(flat)
        done.append((path, raw.count(b"\r")))
    return done


#: Written into the shim and looked for when deciding whether an existing
#: ``pre-commit`` is ours to replace.  Never change it without also
#: handling the old spelling, or every armed clone grows a second hook.
SHIM_MARKER = "# installed from tools/git_hooks/pre_commit_line_endings.py"


def hook_shim() -> str:
    r"""The shim a ``pre-commit`` hook needs in order to call this.

    TWO GUARDS, each for a breakage measured on this machine.

    ``[ -f "$script" ] || exit 0`` -- a git hooks directory is a property
    of the CLONE, not of a worktree: ``git rev-parse --git-path hooks``
    from a linked worktree answers with the MAIN checkout's hooks, shared
    by every worktree cut from it.  Censused 2026-08-29: 192 worktrees
    share this clone's hooks, 72 carry this script, and the other 120 are
    older trees whose branches predate it.  The counts move every week;
    the shape does not.  Without this line the shim in those 120 runs
    ``python`` against a path that does not exist, which exits 2 with
    "can't open file" and REFUSES THE COMMIT, saying nothing about line
    endings.  With it, a tree that does not carry the hook is simply not
    gated by it.

    The interpreter search -- a tree that carries the script but has no
    Python REFUSES the commit rather than passing it, because a gate that
    exits 0 when it could not run is the "quietly not there" failure this
    whole change exists to remove.  ``git commit --no-verify`` is still
    the way past it.  Each candidate is made to RUN before it is trusted,
    which is not fastidiousness: ``command -v python3`` succeeds on this
    machine and resolves to the Windows App Execution Alias, which prints
    "Python was not found" and exits 49.  The first version of this shim
    used ``command -v`` and refused a real commit on the first try.
    """
    relative = os.path.relpath(
        Path(__file__).resolve(), ROOT).replace(os.sep, "/")
    return (
        '#!/bin/sh\n'
        + SHIM_MARKER + '\n'
        'script="$(git rev-parse --show-toplevel)/' + relative + '"\n'
        '# A worktree whose branch predates this hook is not gated by it.\n'
        '[ -f "$script" ] || exit 0\n'
        '# `command -v` is not enough: Windows ships python3.exe as an App\n'
        '# Execution Alias that EXISTS on PATH and exits 49 with "Python was\n'
        '# not found".  This refused a real commit.  Each candidate is made\n'
        '# to run something before it is trusted.\n'
        'for py in python3 python py; do\n'
        '    if "$py" -c "" >/dev/null 2>&1; then\n'
        '        exec "$py" "$script" "$@"\n'
        '    fi\n'
        'done\n'
        'echo "line endings: no python on PATH, so the pre-commit line-ending'
        ' gate could not run." >&2\n'
        'echo "Install python, or commit with --no-verify if you mean to'
        ' skip it." >&2\n'
        'exit 1\n')


def hooks_dir() -> Path:
    """Where git will look for ``pre-commit`` from this worktree.

    ``core.hooksPath`` wins when it is set, exactly as git resolves it;
    otherwise ``--git-path hooks``, which from a linked worktree answers
    with the COMMON git directory's hooks -- the main checkout's.  That
    is not a bug to route around: the hooks directory genuinely is one
    per clone, so arming it once is what arms every worktree, and the
    shim's ``[ -f ]`` guard is what makes that safe.
    """
    configured = subprocess.run(["git", "config", "--get", "core.hooksPath"],
                                cwd=ROOT, capture_output=True)
    if configured.returncode == 0 and configured.stdout.strip():
        hooks = Path(configured.stdout.decode().strip())
    else:
        hooks = Path(_git("rev-parse", "--git-path", "hooks").decode().strip())
    if not hooks.is_absolute():
        hooks = ROOT / hooks
    return hooks.resolve()


def ensure_installed() -> tuple[bool, str]:
    """Arm the hook if it is not armed.  ``(armed, what happened)``.

    DEFAULT-ON, and this is the whole point of the function.  The hook was
    reachable only behind ``--install``, documented only inside its own
    docstring, and nothing in the repository ran it -- so
    ``.git/hooks/pre-commit`` did not exist in this clone at all, and the
    two incidents this hook was written to stop were both committed
    straight through it: ``tests/test_offline_child.py`` rewritten to CRLF
    across all 673 lines with a real 79-line addition buried inside the
    whole-file diff (repaired by 39ef138c5), and ``fp32_floor_probe.rs``
    born with 525 carriage returns (repaired by abb5ff270).  "Fixed means
    default" is a project law, and an opt-in flag is a workaround.

    Idempotent, and it NEVER clobbers a hook it did not write: a
    ``pre-commit`` without :data:`SHIM_MARKER` belongs to somebody else
    and is left exactly as it is, with the refusal reported so the caller
    can say so out loud.  Silently overwriting another lane's hook would
    be a worse failure than the one being fixed.
    """
    target = hooks_dir() / "pre-commit"
    desired = hook_shim().encode("utf-8")
    if target.exists():
        current = target.read_bytes()
        if current == desired:
            return True, "already armed: %s" % target
        if SHIM_MARKER.encode("utf-8") not in current:
            return False, (
                "%s exists and is not this hook's shim, so it was left "
                "alone.  Chain this hook from it, or move it aside and "
                "re-run." % target)
    # Written to a neighbour and renamed over, NOT written in place.  One
    # hooks directory is shared by every worktree of this clone -- 192 of
    # them on 2026-08-29 -- and `conftest.py` calls this on every pytest
    # session, so two lanes can arm at the same moment.  An in-place write
    # that loses that race leaves a TRUNCATED pre-commit hook, and a torn
    # `sh` script refuses every commit in all of them at once.  os.replace
    # is
    # atomic on both platforms.
    scratch = target.with_name("pre-commit.%d.tmp" % os.getpid())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_bytes(desired)         # bytes: a shim must stay LF
        scratch.chmod(0o755)
        os.replace(scratch, target)
    except OSError as failure:
        try:
            scratch.unlink()
        except OSError:
            pass
        return False, "could not write %s: %s" % (target, failure)
    return True, "armed %s" % target


def _install() -> int:
    """``--install``: the same arming ``conftest.py`` does, by hand.

    Kept for a clone whose owner would rather not wait for a test run,
    and for saying out loud what went wrong when arming is refused.  It
    is no longer the only way in -- that is what made this hook a
    workaround.
    """
    armed, story = ensure_installed()
    if armed:
        print(story)
        return 0
    sys.stderr.write(story + "\n")
    return 2


if __name__ == "__main__":
    if "--fix" in sys.argv:
        for path, count in fix(p for p, _c, _origin in offenders()):
            print("  %6d CR -> 0   %s" % (count, path))
        raise SystemExit(0)
    raise SystemExit(main())
