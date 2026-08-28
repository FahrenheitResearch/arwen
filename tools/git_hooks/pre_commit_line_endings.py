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

Install:  python tools/git_hooks/pre_commit_line_endings.py --install
Bypass:   git commit --no-verify   (for a deliberate CR -- then record
          it in the gate's ``_CRLF_DEBT``)
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
    """``_is_authored`` from the release gate itself -- one definition."""
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


def offenders() -> list[tuple[str, int, bool]]:
    """``(path, CR count, is_new)`` for every staged fresh-CR authored file."""
    gate = _authority()
    found = []
    for path in _staged_paths():
        if not gate._is_authored(path):
            continue
        staged = _blob("", path)          # "" -> the index
        if staged is None or b"\r" not in staged:
            continue
        head = _blob("HEAD", path)
        if head is not None and b"\r" in head:
            continue                      # recorded, unchanged debt
        found.append((path, staged.count(b"\r"), head is None))
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
        "  .gitattributes says `* -text`: the bytes in the object database"
        " are the bytes on disk, and the prepared runners hash their own"
        " sources -- a CRLF flip makes a clone and a wheel disagree about"
        " the same file.\n\n")
    for path, count, is_new in found:
        origin = "new file" if is_new else "was LF at HEAD"
        say("    %6d CR   %s   (%s)\n" % (count, path, origin))
    say("\nAlmost always this is a patch script that read the file in "
        "Python text mode and wrote it back with open(path, 'w'), which on "
        "Windows turns every LF into CRLF.  Write with newline='\\n', or "
        "in binary.\n\nFix and re-stage:\n\n")
    say("    python tools/git_hooks/pre_commit_line_endings.py --fix\n")
    say("    git add " + " ".join(path for path, _c, _n in found) + "\n")
    say("\nIf the CR is deliberate, commit with --no-verify and record it "
        "in tests/test_line_ending_stability.py::_CRLF_DEBT.\n")
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


def hook_shim() -> str:
    """The shim a ``pre-commit`` hook needs in order to call this."""
    relative = os.path.relpath(
        Path(__file__).resolve(), ROOT).replace(os.sep, "/")
    return ('#!/bin/sh\n'
            '# installed from tools/git_hooks/pre_commit_line_endings.py\n'
            'exec python "$(git rev-parse --show-toplevel)/'
            + relative + '" "$@"\n')


def _install() -> int:
    """Install the shim -- but never into a git directory we do not own.

    THIS CHECK IS NOT DEFENSIVE PADDING; it is the first thing this script
    got wrong, on the tree it was written for, which was a LINKED
    WORKTREE: ``git rev-parse --git-path hooks`` there answers with the hooks
    directory of the MAIN checkout -- shared by that checkout and by every
    other worktree cut from it.  Installing there silently arms this hook
    for repositories whose owner never asked for it.  ``core.hooksPath`` is
    no better: ``--local`` config is shared across worktrees too, absent
    ``extensions.worktreeConfig``.

    When the hooks directory is outside this worktree, this refuses and
    prints the command instead.  Whoever owns that checkout runs one line;
    nobody has their tree changed underneath them.
    """
    hooks = Path(_git("rev-parse", "--git-path", "hooks").decode().strip())
    if not hooks.is_absolute():
        hooks = ROOT / hooks
    hooks = hooks.resolve()
    target = hooks / "pre-commit"
    if not (hooks == ROOT or ROOT in hooks.parents):
        sys.stderr.write(
            "NOT installing.\n"
            "  this worktree : %s\n"
            "  hooks live in : %s\n"
            "That directory belongs to the main checkout and is shared with "
            "every other worktree cut from it, so installing there would arm "
            "this hook for trees whose owner never asked.\n\n"
            "To arm it, the owner of that checkout runs, from this "
            "worktree:\n\n"
            "    python tools/git_hooks/pre_commit_line_endings.py "
            "--print-shim > '%s'\n"
            "    chmod +x '%s'\n" % (ROOT, hooks, target, target))
        return 2
    hooks.mkdir(parents=True, exist_ok=True)
    target.write_text(hook_shim(), newline="\n", encoding="utf-8")
    target.chmod(0o755)
    print("installed " + str(target))
    return 0


if __name__ == "__main__":
    if "--fix" in sys.argv:
        for path, count in fix(p for p, _c, _n in offenders()):
            print("  %6d CR -> 0   %s" % (count, path))
        raise SystemExit(0)
    raise SystemExit(main())
