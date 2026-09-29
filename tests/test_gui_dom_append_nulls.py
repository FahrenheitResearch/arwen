"""No page code hands an absent line to the DOM's own append.

``Element.append(null)`` does not skip the argument: it inserts the text
``null``.  A forecast refused before its fetch had a message and no
remedy, and its failed card on the Map read the engine's sentence and
then the word "null" on a line of its own.  ``core.js``'s ``append`` and
``h`` skip null, undefined and false, so a list of lines where some may
be absent goes through them; this scan refuses the DOM methods called
with an argument that can be null.
"""

from __future__ import annotations

from pathlib import Path
import re

JS = Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static" / "js"
DOM_CALL = re.compile(r"\.(append|prepend|replaceChildren|before|after)\(")
CAN_BE_NULL = re.compile(r":\s*null\b|\bnull\s*[,)]|&&")


def _arguments(text: str, start: int) -> str:
    depth, i = 1, start
    quote = None
    while i < len(text) and depth:
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        i += 1
    return text[start:i - 1]


def _nested_calls_removed(args: str) -> str:
    # A null inside h(...) or any other nested call is that call's business: h skips it.
    out, depth = [], 0
    for c in args:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            continue
        if depth == 0:
            out.append(c)
    return "".join(out)


def test_no_dom_append_receives_a_possible_null():
    offenders = []
    for path in sorted(JS.glob("*.js")):
        text = path.read_text(encoding="utf-8")
        for match in DOM_CALL.finditer(text):
            top = _nested_calls_removed(_arguments(text, match.end()))
            if CAN_BE_NULL.search(top):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.name}:{line}")
    assert not offenders, (
        "the DOM's append prints a null argument as the word 'null'; pass these lines through core.js's "
        f"append(el, [...]) instead: {offenders}")


def test_the_scan_catches_the_failed_card_shape():
    text = 'card.append(h("p", {}, a), end.remedy ? h("p", {}, end.remedy) : null, h("a", {}, b));'
    match = DOM_CALL.search(text)
    assert CAN_BE_NULL.search(_nested_calls_removed(_arguments(text, match.end())))
    fine = 'card.append(h("p", {}, a ? h("b", {}, a) : null));'
    match = DOM_CALL.search(fine)
    assert not CAN_BE_NULL.search(_nested_calls_removed(_arguments(fine, match.end())))
