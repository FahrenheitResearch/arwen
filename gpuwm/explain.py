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
import sys

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


#: Process-wide record of whether this invocation asked for the full
#: layer.  Library code that emits warnings has no ``args`` in reach;
#: the CLI stamps the flag here once, right after parsing.
_EXPLAIN_ACTIVE = False


def set_explain(enabled: bool) -> None:
    """Record --explain for this process (set once by the CLI)."""

    global _EXPLAIN_ACTIVE
    _EXPLAIN_ACTIVE = bool(enabled)


#: Callables that also receive every warning, as a typed mapping.
#:
#: The stderr sentence is the reader's interface and does not change.
#: This is the interface for a PROGRAM driving gpuwm as a subprocess:
#: it needs the same facts as fields rather than as prose it would have
#: to recognize by shape.  A generic list of callables rather than one
#: named consumer, so the next machine-facing surface reuses it instead
#: of adding a second hook beside it.
#:
#: Observers are called inside the same call that prints, so a warning
#: is never observed later than it is printed.  One that raises would
#: turn an advisory into a failure, so each is called defensively.
_WARNING_OBSERVERS: list = []


def add_warning_observer(observer) -> None:
    """Also deliver every :func:`warn` to ``observer(record)``.

    ``record`` is ``{"action": ..., "why": ...}`` -- the two halves the
    layering convention already splits every message into, unjoined, so
    a consumer chooses its own layer the way :func:`render` does.
    """

    if not callable(observer):
        raise TypeError("warning observer must be callable")
    _WARNING_OBSERVERS.append(observer)


def remove_warning_observer(observer) -> None:
    """Detach an observer; absent is not an error."""

    try:
        _WARNING_OBSERVERS.remove(observer)
    except ValueError:
        pass


def warn(action: str, why: str = "") -> None:
    """Print one warning sentence and keep going.

    This is the voice of every check that found something worth saying
    but nothing worth stopping for: the run continues, and the reader
    gets exactly one ``warning:`` line saying what happened and (when
    there is one) what to do.  The mechanism prose goes in ``why`` and
    prints only when the invocation carried ``--explain`` -- same
    layering contract as the refusals, same flag.

    Warnings go to stderr so a piped stdout (JSON reports, command
    relays) stays clean.

    Every warning is additionally delivered to each observer registered
    with :func:`add_warning_observer`, which is how a machine consumer
    receives it as a field rather than as a line to recognize.
    """

    action = " ".join(str(action).split())
    print(f"warning: {action}", file=sys.stderr)
    if why and _EXPLAIN_ACTIVE:
        for line in str(why).strip("\n").splitlines():
            print(f"  {line}", file=sys.stderr)
    if not _WARNING_OBSERVERS:
        return
    record = {"action": action, "why": str(why)}
    for observer in tuple(_WARNING_OBSERVERS):
        try:
            observer(record)
        except Exception:  # noqa: BLE001 - an advisory never fails a run
            continue


__all__ = [
    "EXPLAIN_MARK", "add_explain_flag", "add_warning_observer",
    "explain_enabled", "layered", "remove_warning_observer", "render",
    "set_explain", "split", "warn",
]
