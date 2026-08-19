"""What a release changed about the things a reader's scripts touch.

The upgrader persona built the 2.4.1 world for real, installed 2.5.0
over it, and could not tell from any door that anything had happened:
``gpuwm doctor``'s report is shape-identical to 2.4.1's (only counters
move) and ``gpuwm run --help`` was byte-identical.  The doors that
changed named 2.4.1 in their help; the doors that did not change said
nothing at all (UX finding N15).

So this is the one place a release records the handful of changes that
alter what an existing user SEES -- an output path that moved, a
command that split in two -- as opposed to everything a changelog
carries.  It is deliberately short and deliberately not the changelog:
a reader who has just upgraded wants the three sentences that explain
why their script's glob found nothing, not a hundred entries.

Every line is COMPOSED from the code that owns the behaviour --
:func:`gpuwm.run_stamp.describe` and
:func:`gpuwm.render_layout.describe` -- rather than transcribed beside
it, so a layout that moves again cannot leave a true-sounding sentence
behind.
"""

from __future__ import annotations

from typing import Callable


def _release_2_5_0() -> tuple[str, ...]:
    from gpuwm import render_layout, run_stamp

    return (
        "`gpuwm go` still runs the whole forecast; the two stages it "
        "drives are doors of their own now -- `gpuwm prep` builds the "
        "prepared tree and `gpuwm sim` runs it, so a script can hold "
        "either half.",
        "Each forecast writes into its own timestamped run folder: "
        + run_stamp.describe("--outdir")
        + ".  `--run-stamp off` restores the flat tree and is labelled a "
        "workaround, not a supported alternative.",
        # The platform's own separator here: this note is printed on the
        # reader's box, about where their next render will actually put
        # files, so it follows the same spelling that render prints.
        "`gpuwm render` writes " + render_layout.describe("--out")
        + ".  `--layout flat` restores the single directory, on the same "
        "terms.",
    )


#: Release -> the lines that release added.  ONE row per release that
#: changed something a user can see from outside the process; a release
#: that changed nothing user-visible has no row and says nothing.
#:
#: The values are callables so that importing this module costs nothing
#: -- the lines are built out of the layout modules, and a version
#: check must not drag the renderer's imports in behind it.
RELEASE_NOTES: dict[str, Callable[[], tuple[str, ...]]] = {
    "2.5.0": _release_2_5_0,
}


def _ordinal(version: str) -> tuple[int, ...] | None:
    """``"2.5.0"`` -> ``(2, 5, 0)``; ``None`` for anything else.

    Deliberately not ``packaging.version``: packaging is not a declared
    dependency of this product (see :mod:`gpuwm.version_cli`), and the
    only comparison needed here is between plain release numbers.  A
    pre-release or local version compares as its release part, which is
    the answer a reader of an upgrade note wants -- ``2.5.0rc1`` and
    ``2.5.0`` changed the same things.
    """

    head = version.strip().split("+", 1)[0]
    fields: list[int] = []
    for part in head.split("."):
        digits = ""
        for character in part:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        fields.append(int(digits))
    return tuple(fields) if fields else None


def since(previous: str | None, installed: str) -> tuple[str, ...]:
    """Every recorded line for a release in ``(previous, installed]``.

    ``previous`` is the version a reader was last on.  ``None``, an
    unreadable version on either side, or an install that is not newer
    returns nothing at all: an upgrade note needs an upgrade to be
    about, and inventing one for a downgrade or a re-run would make the
    note the noise it exists to avoid.
    """

    if not previous or not installed:
        return ()
    was, now = _ordinal(previous), _ordinal(installed)
    if was is None or now is None or not was < now:
        return ()
    lines: list[str] = []
    for release, build in sorted(RELEASE_NOTES.items(),
                                 key=lambda row: _ordinal(row[0]) or ()):
        mark = _ordinal(release)
        if mark is None or not was < mark <= now:
            continue
        lines.extend(build())
    return tuple(lines)


__all__ = ["RELEASE_NOTES", "since"]
