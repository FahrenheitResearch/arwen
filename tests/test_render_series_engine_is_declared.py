"""`--series` needs the native engine, and the door that builds it says so.

`--series` renders a whole timeline into ONE renderer store so multi-hour
windowed products can be differenced across frames (`qpf_6h` is F012 minus
F006).  The matplotlib arm has none of that machinery: it renders one file at
a time out of a catalog of five per-file instantaneous products, so
``gpuwm/render.py`` refuses the pair at argument resolution.  Nothing a reader
meets first said why.

The Plot history guide is this lane's door onto that command.  It builds
``render ... --series --engine rust`` (``tools/arwen-tui/src/guide.rs``, the
``Kind::Render`` arm of :func:`Guide::request`), and the command it builds is
shown for review and copying before anything runs, so the engine requirement
belongs in the help beside it.

A declaration only counts where the reader can actually read it.  The guide's
help pane does not scroll (``main.rs`` renders it as a plain wrapped
``Paragraph`` with no scroll offset, unlike the help dialog), and the
workspace refuses to draw below 65 x 20, so anything that wraps past the
bottom of that pane is unreachable: no key pages to it and no resize is
offered.  :func:`test_the_plot_history_help_reaches_the_reader_at_65_by_20`
measures the help against that pane.  Its Rust twin,
``guide::tests::the_plot_history_help_reaches_the_reader_at_the_smallest_supported_terminal``,
renders the real dialog through ratatui at 65 x 20 and reads the screen back;
this one repeats the measurement in the portable suite so a wording edit is
caught without a Rust toolchain.

The mechanism behind the requirement does not fit that pane in six rows, so
it is recorded in ``CHANGELOG.md`` instead of in the help, and
:func:`test_the_changelog_keeps_the_mechanism_the_pane_cannot_hold` holds it
there: shortening the help is only safe while the tree still says somewhere
why the native engine is the one that can draw a timeline.

Asserted here on the guide and this repository's changelog alone, because
they are the only sites in reach.  The refusal text at gpuwm/render.py:2338
still names the mechanism and neither the breakage nor the way out, and the
`--series` argparse help at gpuwm/render.py:2530-2531 still names no engine;
that file belongs to the lane editing it and is out of bounds here, so both
are handed back untouched.  docs/public/CLI-OPTIONS.md:1114 is generated from
that argparse help by `python -m tools.build_cli_options_doc` and is never
hand-edited, so it follows render.py:2531 and nothing is to be done there by
hand.  docs/public/CLI-USER-MANUAL.md:463, :476, :483 and its render option
table at :488-498, and docs/public/TUI-USER-MANUAL.md:258, still describe
`--series` with no engine named and are outside this lane's paths; they are
handed back with those line numbers.  No case here asserts on any of them: a
test this lane cannot turn green does not belong in this lane.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "tools" / "arwen-tui" / "src" / "guide.rs"
CHANGELOG = REPO / "CHANGELOG.md"

#: Head of the Plot history question list.  Located by this marker rather than
#: by line number so the assertions survive edits above them (the arm opens at
#: guide.rs:157 on f08085092).
RENDER_QUESTIONS = "Kind::Render=>vec!["

#: The line of the Kind::Render arm of `Guide::request` that starts the
#: timeline request (guide.rs:541 on f08085092).
SERIES_PUSH = 'args.push("--series".into());'

#: The help pane of a guide question at the smallest terminal the workspace
#: draws.  main.rs refuses to draw below 65 x 20; at 65 x 20 the dialog
#: backdrop is 65 x 17, its popup is 63 x 15, the panel border leaves 61 x 13
#: inside, the dialog body keeps 2 rows of notice and 2 rows of buttons for a
#: 9-row body, and the question layout spends one row on the label, one on the
#: editable value and one on the key hint.  Six rows of 61 columns are left,
#: and the pane has no scroll offset.
PANE_COLUMNS = 61
PANE_ROWS = 6


def _wrap(text: str, width: int = PANE_COLUMNS) -> list[str]:
    """Greedy word wrap, the way ratatui wraps a `Paragraph` with `trim:false`.

    Verified against the real widget: the six rows this returns for the
    current help are the six rows the Rust test reads out of the rendered
    screen, word for word.
    """

    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_questions() -> str:
    """The Plot history guide's question list, prompts and help together."""

    text = GUIDE.read_text(encoding="utf-8")
    start = text.index(RENDER_QUESTIONS)
    end = text.index("Kind::", start + len(RENDER_QUESTIONS))
    return text[start:end]


def _history_help() -> str:
    """The help string of the Plot history guide's first question."""

    line = next(
        line
        for line in _render_questions().splitlines()
        if '"Forecast history file or folder"' in line
    )
    fields = re.findall(r'"((?:[^"\\]|\\.)*)"', line)
    return fields[2]


def _series_request() -> str:
    """The command-building lines that put `--series` in the request."""

    text = GUIDE.read_text(encoding="utf-8")
    start = text.index(SERIES_PUSH)
    return text[start:start + 400]


def _mechanism_changelog() -> str:
    """The CHANGELOG section that carries the Plot history guide entry.

    Located by the entry rather than by position.  The mechanism was
    written into the section of the release that ships the guide, which
    was the topmost section while that release was open and is a dated
    one once it cuts.  Reading the topmost section instead pinned this
    record to whichever release is unreleased NOW, so the first lane to
    open the next version's heading turned this case red without
    touching the guide, the help or the entry.  What the case is about
    is that the tree still says somewhere why a timeline needs the
    native engine, and a released section says it just as well.
    """

    sections = CHANGELOG.read_text(encoding="utf-8").split("\n## ")[1:]
    holding = [section for section in sections
               if "Plot history guide" in section]
    assert holding, (
        "no CHANGELOG section carries the Plot history guide entry that "
        "holds the mechanism the help pane cannot: the tree no longer says "
        "anywhere why a timeline needs the native engine")
    return holding[0]


def test_the_guide_source_is_read_in_one_piece() -> None:
    """The slices above found real code, not an empty string."""

    questions = _render_questions()
    assert "Forecast history file or folder" in questions
    assert "Requested plots" in questions
    assert SERIES_PUSH in _series_request()
    assert _history_help().startswith("An actual wrfout file")


def test_the_plot_history_guide_declares_the_engine_its_timeline_needs() -> None:
    help_text = _history_help()
    assert "--engine rust" in help_text, (
        "the Plot history guide never says which engine draws its timeline, "
        "though it builds --series --engine rust and shows that command for "
        "review: " + help_text)
    assert "matplotlib draws one file at a time and refuses --series" in help_text, (
        "the guide does not name the concrete breakage the requirement "
        "prevents, that the matplotlib engine draws one file at a time and "
        "refuses --series: " + help_text)
    assert re.search(r"drop --series", help_text), (
        "the guide names the breakage but not the way out, dropping --series "
        "to draw each file on its own: " + help_text)


def test_the_plot_history_help_reaches_the_reader_at_65_by_20() -> None:
    """Every sentence of the help lands inside the pane that draws it.

    The pane does not scroll, so a wrapped row past `PANE_ROWS` is text the
    reader has no way to reach.  A declaration that is only half readable is
    the defect this help was changed to remove, so it must not be reproduced
    by the change itself.
    """

    help_text = _history_help()
    wrapped = _wrap(help_text)
    assert len(wrapped) <= PANE_ROWS, (
        "the Plot history help wraps to {} rows in a {} x {} pane that cannot "
        "scroll, so {} rows are unreachable at the smallest supported "
        "terminal:\n{}".format(
            len(wrapped), PANE_COLUMNS, PANE_ROWS, len(wrapped) - PANE_ROWS,
            "\n".join(wrapped)))

    visible = " ".join(wrapped[:PANE_ROWS])
    for phrase in (
        # The declaration, its breakage and its way out.
        "--engine rust",
        "matplotlib draws one file at a time and refuses --series",
        "drop --series to use it",
        # Guidance that fit on f08085092 and must still fit beside it.
        "Discovery includes child folders, skips symbolic links, and is bounded.",
        "The exact files appear in the command review.",
    ):
        assert phrase in visible, (
            "{!r} is not on screen at the smallest supported terminal:\n{}"
            .format(phrase, "\n".join(wrapped)))


def test_the_plot_history_guide_builds_the_command_its_help_describes() -> None:
    """The help promises `--series --engine rust`; the builder must deliver it.

    Green on f08085092 as well: it guards the claim the help now makes
    against a later edit that drops the engine from the request and leaves
    the sentence standing.
    """

    request = _series_request()
    assert '"--engine".into(), "rust".into()' in request, (
        "the Plot history guide requests --series without pinning the native "
        "engine, so its help describes a command it no longer builds: "
        + request)


def test_the_changelog_keeps_the_mechanism_the_pane_cannot_hold() -> None:
    """What left the help has to survive somewhere in the tree.

    The help was cut to six rows by moving the mechanism out of it.  If the
    entry that received the mechanism is trimmed too, the tree stops saying
    anywhere why a timeline needs the native engine, and the help's bare
    requirement becomes a rule with no reason behind it.
    """

    entry = _mechanism_changelog()
    for phrase in (
        # Why one timeline has to reach one renderer.
        "one renderer store",
        "differenced across frames",
        "F012 minus F006",
        # Why the other engine cannot do it.
        "renders one file at a time",
        "carries no windowed product",
        # Why the pair is refused instead of ignored.
        "index inside each file",
    ):
        assert phrase in entry, (
            "{!r} left the help for CHANGELOG.md and is now in neither:\n{}"
            .format(phrase, entry))
