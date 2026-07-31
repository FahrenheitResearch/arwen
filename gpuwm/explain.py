"""One layering convention for everything the CLI prints.

Every message this project prints earned its words.  The refusals name
what they refused and why the rail exists; the remedies are pasteable
whole; the receipts say what was verified against what.  None of that is
being cut.  What changed is that all of it used to print at once, so the
one line a reader needed -- the next command -- arrived in the middle of
a wall and was read as part of the wall.  A field exhibit: a wizard run
whose correct ``gpuwm fetch`` line sat at line 15 of 20, under a
gray-zone advisory and above a nine-name dataset inventory, and whose
user's verdict was "still can't get it working".

So the words are layered rather than shortened:

* **default** -- what happened, and the single next action.  One line
  per item wherever items repeat.
* **``--explain``** -- the same output with the mechanism prose, the
  alternate routes, and the per-item evidence restored *verbatim*.

The convention is one flag with one name on every subcommand
(:func:`add_explain_flag` puts it on all of them, so the pointer this
module appends is never a lie), and one composition helper
(:func:`layered`) for messages that carry both halves.

Why a text sentinel rather than a structured exception.  The refusals
travel as ``ValueError``/``NotImplementedError`` through call chains
this package does not own, and are asserted on by tests that read
``str(error)``.  :func:`layered` keeps both halves inside that one
string, so every existing content assertion still holds and only the
*print boundary* -- :func:`render` -- decides which half reaches the
terminal.  Nothing is deleted; a layer is chosen.
"""

from __future__ import annotations

import argparse

#: Separates a message's ACTION half from its WHY half inside one string.
#:
#: A sentinel rather than a blank line because prose contains blank
#: lines and a heuristic split would eventually cut a paragraph in half.
#: It is written only by :func:`layered` and removed by :func:`render` in
#: BOTH modes, so it cannot reach a terminal even on a path that forgets
#: to ask which layer it wanted.
EXPLAIN_MARK = "\n[[explain]]\n"

#: Appended to a layered message printed without ``--explain``.
_POINTER = "  (run {command} --explain for the reason)"


def layered(action: str, why: str) -> str:
    """Compose a message from its action half and its explanation half.

    ``action`` is what was refused or what happened, plus the exact
    remedy: everything a reader needs in order to act.  ``why`` is the
    mechanism -- the paragraph that says what the rail is protecting,
    which route was withdrawn, what the alternative costs.

    A caller with no explanation half gets its action back unchanged, so
    wrapping a message that has not been split yet is a no-op rather
    than a message with an empty section.

    Surrounding blank lines are removed from both halves; leading
    INDENTATION on the first line is not.  These messages are written
    as indented blocks -- ``  What to do:`` above ``  Why:`` -- and a
    normalizer that reached for ``strip()`` would left-align the
    explanation while the action kept its gutter, so the two halves
    would print as if they came from different messages.
    """

    action = action.strip("\n").rstrip()
    why = why.strip("\n").rstrip()
    if not why.strip():
        return action
    return action + EXPLAIN_MARK + why


def split(message: str) -> tuple[str, str]:
    """``(action, why)`` for a message; ``why`` is ``""`` when unlayered."""

    text = str(message)
    head, mark, tail = text.partition(EXPLAIN_MARK)
    return (head, tail) if mark else (text, "")


def render(message: str, *, explain: bool, command: str | None = None) -> str:
    """The layer of ``message`` that ``explain`` asked for.

    With ``explain`` the two halves are rejoined with a blank line and
    printed in full -- that is the whole promise of the flag, and it is
    why the explanation half is stored verbatim rather than summarized.

    Without it the action half stands alone, followed by a pointer at
    the flag that produces the rest.  ``command`` is what the reader
    would re-run; when a caller cannot name it the pointer is omitted
    rather than guessed, because a pointer at a command that does not
    take ``--explain`` is worse than no pointer.
    """

    action, why = split(message)
    if not why:
        return action
    if explain:
        return f"{action}\n\n{why}"
    if command:
        return action + "\n" + _POINTER.format(command=command)
    return action


def add_explain_flag(parser: argparse.ArgumentParser) -> None:
    """Register ``--explain`` on one parser, idempotently.

    Idempotent because the CLI adds the flag by sweeping every
    registered subparser, and two registrars share a parser (``check``
    is built by the ingest preflight and extended by the memory
    estimator; ``run``/``resume`` are extended by the supervisor).  A
    second registration would be ``argparse.ArgumentError`` at import
    time, which is a startup crash rather than a message-layer bug.
    """

    if any("--explain" in action.option_strings
           for action in parser._actions):  # noqa: SLF001 - argparse's only API
        return
    parser.add_argument(
        "--explain", action="store_true",
        help="print the full reasoning, alternate routes and per-item "
             "evidence behind this command's output, instead of the "
             "default one-line-per-item summary")


def explain_enabled(args) -> bool:
    """Did the caller ask for the full layer?  Absent flag means no."""

    return bool(getattr(args, "explain", False))


__all__ = [
    "EXPLAIN_MARK", "add_explain_flag", "explain_enabled", "layered",
    "render", "split",
]
