"""Fortran namelist input, read the way WRF's and WPS's own READ(NML=) does.

ONE reader for every door that takes a user's namelist.input or
namelist.wps.  It follows the Fortran list-directed namelist rules
rather than the one-key-per-line shape WRF's shipped files happen to use:

* a group is ``&name`` (or ``$name``) .. ``/`` (or ``&end`` / ``$end``),
  and may sit on one line with its assignments;
* values are separated by commas, by blanks, or by line ends, and a new
  ``name =`` starts the next assignment anywhere on a line, so
  ``start_year = 2026, start_month = 08, start_day = 25,`` is three keys;
* a value list continues over as many lines as it needs;
* ``name(3) = v`` assigns one element (and, as gfortran reads it, the
  elements after it when more values follow), ``name(2:4) = ...``
  and ``name(1:5:2) = ...`` assign a section, and a repeated name
  overwrites only the elements it gives -- the rest keep what an earlier
  assignment put there;
* ``3*1.0`` repeats a value and ``3*`` is three null values; a null (an
  empty slot between commas) leaves its element unchanged;
* quoted strings may hold commas, ``=``, ``!`` and ``/``, and a doubled
  quote inside one is a literal quote;
* ``!`` starts a comment outside a string; one that starts right after
  a comma or ``=`` on the same line is an empty value when the list goes
  on below it, as gfortran (which builds WRF) reads it;
* logicals are ``.true.``/``.false.``/``T``/``F`` (optional periods,
  any case), reals may carry a ``D`` exponent.

Where the Fortran runtime would keep a value this reader cannot see --
an element left unset in front of one that is set, which WRF fills from
its Registry default -- the reader refuses by name instead of shifting
the later values into the wrong domain's column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


#: A namelist object name.  Derived-type components (``a%b``) are kept in
#: the name; WRF's namelists declare none.
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:%[A-Za-z][A-Za-z0-9_]*)*")
#: The characters an array subscript may carry.
_SUBSCRIPT = re.compile(r"[ \t0-9+\-:,]*")
#: Fortran repetition count ``N*`` in front of a value (or of nothing).
_REPEAT = re.compile(r"([0-9]+)\*")
#: An undelimited value: everything up to a blank, a value separator,
#: the group terminator or a comment.
_UNDELIMITED = re.compile(r"[^ \t\r\n,/!]+")
#: Fortran double-precision exponent literal (``2.90D2``); Python floats
#: only accept E, so D/d maps to e before conversion.
_D_EXPONENT = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)[dD][+-]?\d+")
#: Fortran logical input: optional period, T or F, optional rest of the
#: word and period (``.true.``, ``T``, ``.f.``, ``false``).
_LOGICAL = re.compile(r"\.?(?:t|true|f|false)\.?", re.IGNORECASE)
_BLANKS = " \t"
_SEPARATOR_AFTER_VALUE = " \t\r\n,/!"


@dataclass(frozen=True)
class NamelistAssignment:
    """One ``name[(subscript)] = values`` inside a group, with its span.

    ``values`` holds the parsed scalars in order, ``None`` for a null
    value.  ``start`` is the offset of the name, ``value_start`` the
    first character after ``=`` and the blanks on its line, and ``end``
    one past the last value or value-separating comma that belongs to
    this assignment -- so ``text[start:end]`` is the whole assignment and
    replacing it touches nothing else on a packed line.
    """

    group: str
    name: str
    subscript: tuple | None
    values: tuple
    start: int
    value_start: int
    end: int


@dataclass(frozen=True)
class NamelistGroup:
    """One ``&name .. /`` group.  ``terminator`` is the span of its ``/``
    or ``&end``, ``None`` when the text ended (or the next group began at
    the start of a line) without one."""

    name: str
    start: int
    terminator: tuple[int, int] | None
    assignments: tuple[NamelistAssignment, ...]


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _skip_blanks_and_comments(text: str, pos: int) -> int:
    n = len(text)
    while pos < n:
        c = text[pos]
        if c in " \t\r\n":
            pos += 1
        elif c == "!":
            end = text.find("\n", pos)
            pos = n if end < 0 else end + 1
        else:
            break
    return pos


def _at_line_start(text: str, pos: int) -> bool:
    start = text.rfind("\n", 0, pos) + 1
    return not text[start:pos].strip(_BLANKS + "\r")


def _subscript(inner: str, where: str) -> tuple:
    """``("element", i)`` or ``("section", start, stop, step)``."""

    if "," in inner:
        raise ValueError(
            f"{where}: a multi-dimensional subscript ({inner.strip()}) -- "
            "WRF's namelist arrays are one-dimensional, and this reader "
            "keeps one value list per name, so it would flatten the array "
            "into the wrong elements")
    parts = inner.split(":")
    try:
        numbers = [int(part) if part.strip() else None for part in parts]
    except ValueError:
        numbers = None
    if numbers is None or len(parts) > 3 or (len(parts) == 1 and numbers[0] is None):
        raise ValueError(f"{where}: ({inner.strip()}) is not an array subscript")
    if len(parts) == 1:
        subscript = ("element", numbers[0], None, 1)
    else:
        start = 1 if numbers[0] is None else numbers[0]
        step = 1 if len(parts) == 2 or numbers[2] is None else numbers[2]
        if step == 0:
            raise ValueError(f"{where}: an array section with a zero stride")
        subscript = ("section", start, numbers[1], step)
    lowest = min(v for v in subscript[1:3] if v is not None)
    if lowest < 1:
        raise ValueError(
            f"{where}: subscript {lowest} is below 1; WRF declares its "
            "namelist arrays from element 1")
    return subscript


def _assignment_target(text: str, pos: int, where):
    """``(name, subscript, offset after '=')`` when ``pos`` opens a
    ``name[(subscript)] =``, else ``None`` (the token is a value)."""

    match = _NAME.match(text, pos)
    if match is None:
        return None
    n = len(text)
    cursor = match.end()
    while cursor < n and text[cursor] in _BLANKS:
        cursor += 1
    subscript = None
    if cursor < n and text[cursor] == "(":
        close = text.find(")", cursor)
        if close < 0 or not _SUBSCRIPT.fullmatch(text, cursor + 1, close):
            return None
        inner = text[cursor + 1:close]
        cursor = close + 1
        while cursor < n and text[cursor] in _BLANKS:
            cursor += 1
        if cursor >= n or text[cursor] != "=":
            return None
        subscript = _subscript(inner, where(pos))
    if cursor >= n or text[cursor] != "=":
        return None
    return match.group().lower(), subscript, cursor + 1


def _scalar(token: str):
    if _LOGICAL.fullmatch(token):
        return token.lstrip(".")[:1].lower() == "t"
    if "_" in token:
        # Python's int()/float() read '1_000' as a number; Fortran does not.
        return token
    if _D_EXPONENT.fullmatch(token):
        return float(token.lower().replace("d", "e"))
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _quoted(text: str, pos: int, where) -> tuple[str, int]:
    quote = text[pos]
    pieces = []
    cursor = pos + 1
    while True:
        close = text.find(quote, cursor)
        if close < 0:
            raise ValueError(
                f"{where(pos)}: a quoted value that never closes its {quote}")
        pieces.append(text[cursor:close])
        if close + 1 < len(text) and text[close + 1] == quote:
            pieces.append(quote)
            cursor = close + 2
            continue
        # A character value continued onto the next record does not
        # include the record boundary (list-directed input rules).
        value = "".join(pieces).replace("\r\n", "").replace("\n", "")
        return value, close + 1


def _constant(text: str, pos: int, where) -> tuple[object, int]:
    c = text[pos]
    if c in "'\"":
        value, end = _quoted(text, pos, where)
    elif c == "(":
        close = text.find(")", pos)
        parts = [] if close < 0 else text[pos + 1:close].split(",")
        try:
            if len(parts) != 2:
                raise ValueError
            value = complex(float(_scalar(parts[0].strip())),
                            float(_scalar(parts[1].strip())))
        except (TypeError, ValueError):
            raise ValueError(
                f"{where(pos)}: a parenthesised value that is not a complex "
                "constant (re, im)") from None
        end = close + 1
    else:
        match = _UNDELIMITED.match(text, pos)
        return _scalar(match.group()), match.end()
    if end < len(text) and text[end] not in _SEPARATOR_AFTER_VALUE:
        raise ValueError(
            f"{where(end)}: {text[end]!r} directly after a value; values "
            "are separated by commas or blanks")
    return value, end


def _values(text: str, pos: int, where) -> tuple[list, int]:
    """One value item at ``pos``: a constant, ``r*c`` or ``r*`` (nulls)."""

    repeat = _REPEAT.match(text, pos)
    if repeat is not None:
        count = int(repeat.group(1))
        if count < 1:
            raise ValueError(
                f"{where(pos)}: repeat count {count} -- Fortran requires a "
                "positive repeat count")
        after = repeat.end()
        if after >= len(text) or text[after] in _SEPARATOR_AFTER_VALUE:
            return [None] * count, after
        value, end = _constant(text, after, where)
        return [value] * count, end
    value, end = _constant(text, pos, where)
    return [value], end


def _group(text: str, name: str, start: int, pos: int,
           where) -> tuple[NamelistGroup, int]:
    n = len(text)
    assignments: list[NamelistAssignment] = []
    current: dict | None = None
    expect_value = False

    def close_current():
        if current is not None:
            assignments.append(NamelistAssignment(
                group=name, name=current["name"],
                subscript=current["subscript"],
                values=tuple(current["values"]), start=current["start"],
                value_start=current["value_start"], end=current["end"]))

    while True:
        # gfortran, which builds WRF, reads a '!' comment that starts on
        # the same line right after '=' or a comma as an empty value when
        # a value follows it: 'a = 1, ! note' then '2' on the next line is
        # a = 1, <unset>, 2 in the model.  A comment after a value, or on
        # a line of its own, is not.  (Measured with gfortran 15.2.)
        comment_null = False
        if expect_value:
            cursor = pos
            while cursor < n and text[cursor] in _BLANKS:
                cursor += 1
            comment_null = cursor < n and text[cursor] == "!"
        pos = _skip_blanks_and_comments(text, pos)
        if pos >= n:
            close_current()
            return NamelistGroup(name, start, None, tuple(assignments)), n
        c = text[pos]
        if c == "/":
            close_current()
            return (NamelistGroup(name, start, (pos, pos + 1),
                                  tuple(assignments)), pos + 1)
        if c in "&$":
            match = _NAME.match(text, pos + 1)
            if match is not None and match.group().lower() == "end":
                close_current()
                return (NamelistGroup(name, start, (pos, match.end()),
                                      tuple(assignments)), match.end())
            if match is not None and _at_line_start(text, pos):
                # The next group opens before this one was closed; the
                # group ends here, as it always has for this reader.
                close_current()
                return NamelistGroup(name, start, None, tuple(assignments)), pos
            raise ValueError(
                f"{where(pos)}: {c!r} inside &{name}; a group ends with '/' "
                "or &end before the next one opens")
        if c == ",":
            if current is None:
                raise ValueError(
                    f"{where(pos)}: a value separator before any 'name =' "
                    f"in &{name}")
            if expect_value:
                current["values"].append(None)
            expect_value = True
            current["end"] = pos + 1
            pos += 1
            continue
        target = _assignment_target(text, pos, where)
        if target is not None:
            close_current()
            key, subscript, after = target
            value_start = after
            while value_start < n and text[value_start] in _BLANKS:
                value_start += 1
            current = {"name": key, "subscript": subscript, "values": [],
                       "start": pos, "value_start": value_start,
                       "end": after}
            expect_value = True
            pos = after
            continue
        if current is None:
            token = (_UNDELIMITED.match(text, pos) or _NAME.match(text, pos))
            shown = text[pos:pos + 20] if token is None else token.group()
            raise ValueError(
                f"{where(pos)}: {shown!r} in &{name} is neither a 'name =' "
                "nor a value of one; WRF's namelist read stops here")
        values, pos = _values(text, pos, where)
        if comment_null:
            current["values"].append(None)
        current["values"].extend(values)
        current["end"] = pos
        expect_value = False


def scan_namelist_text(text: str, *, source: str = "") -> tuple[NamelistGroup, ...]:
    """Every group of a namelist TEXT, with each assignment's span.

    The spans are what a rewriter needs: an editor that replaces
    ``text[a.start:a.end]`` changes exactly one assignment, however the
    file packs its keys, instead of the rest of whatever line it sat on.
    ``source`` prefixes refusals.
    """

    def where(pos: int) -> str:
        return (f"{source}: " if source else "") + f"namelist line {_line(text, pos)}"

    groups: list[NamelistGroup] = []
    n = len(text)
    pos = 0
    while pos < n:
        cursor = pos
        while cursor < n and text[cursor] in _BLANKS + "\r":
            cursor += 1
        if cursor < n and text[cursor] in "&$":
            match = _NAME.match(text, cursor + 1)
            if match is None:
                raise ValueError(f"{where(cursor)}: {text[cursor]!r} with no group name")
            name = match.group().lower()
            if name != "end":
                group, pos = _group(text, name, cursor, match.end(), where)
                groups.append(group)
                if group.terminator is not None:
                    _check_after_terminator(text, pos, group, where)
                continue
        # Text outside a group is skipped, as the Fortran runtime skips it
        # while it looks for the next '&name'.
        end = text.find("\n", cursor)
        pos = n if end < 0 else end + 1
    return tuple(groups)


def _check_after_terminator(text: str, pos: int, group: NamelistGroup, where) -> None:
    """Refuse text left on a terminator's line.

    WRF reads '/' as the end of the group wherever it appears, so an
    unquoted path such as ``geog_data_path = /data/geog`` ends the group
    at its first slash: the value and every key after it in that group are
    never read, by WRF or by this reader.  Anything but a comment or the
    next group left beside a terminator is that mistake, named here rather
    than dropped.
    """

    end = text.find("\n", pos)
    rest = text[pos:len(text) if end < 0 else end]
    body = rest.split("!", 1)[0].strip()
    if body and not body.startswith(("&", "$")):
        raise ValueError(
            f"{where(pos)}: {body!r} follows the '/' that ends &{group.name}; "
            "WRF stops reading the group at that '/', so this text and any "
            "key after it in the group would be silently lost.  Quote a "
            "value that contains '/' (for example a path)")


def value_end_through_blanks(text: str, assignment: NamelistAssignment) -> int:
    """``assignment.end``, extended over blanks that run to the line end.

    A rewriter replacing ``text[assignment.value_start:<this>]`` leaves a
    comment after the value (and the blanks before it) in place and drops
    only blanks that trailed the value to the end of its line.
    """

    cursor = assignment.end
    while cursor < len(text) and text[cursor] in _BLANKS:
        cursor += 1
    if cursor >= len(text) or text[cursor] in "\r\n":
        return cursor
    # An assignment with no value ends at its '='; its value_start is past
    # the blanks after it, and a span must not run backwards.
    return max(assignment.end, assignment.value_start)


def _positions(assignment: NamelistAssignment, where: str):
    subscript = assignment.subscript
    count = len(assignment.values)
    if subscript is None:
        return range(1, count + 1)
    kind, start, stop, step = subscript
    if kind == "element":
        # gfortran reads the values after the first into the following
        # elements (its expanded read, a GNU extension that is on unless a
        # strict -std= is given, which WRF's builds do not give).
        return range(start, start + count)
    if stop is None:
        stop = 1 if step < 0 else start + (count - 1) * step
    positions = range(start, stop + (1 if step > 0 else -1), step)
    if count > len(positions):
        raise ValueError(
            f"{where}: {count} values for the {len(positions)}-element "
            f"section {assignment.name}({start}:{subscript[2] or ''}"
            f"{'' if step == 1 else ':' + str(step)})")
    return positions


def parse_namelist_text(text: str, *, allow_unset: bool = False,
                        source: str = "") -> dict[str, dict[str, list]]:
    """Parse Fortran namelist TEXT into ``{group: {name: [values]}}``.

    Group and object names are lower-cased (Fortran names are case
    blind).  Each list is the array as WRF would hold it after the read,
    element 1 first, as far as the namelist sets it; assignments are
    applied in order, each touching only the elements it names.

    An element left unset in front of a set one (``e_we = 100, , 300``,
    ``e_we = 2*, 300`` or ``e_we(3) = 300`` alone) keeps WRF's Registry
    default in WRF, which this reader does not carry: it is refused by
    name unless ``allow_unset`` is true, in which case it is ``None``.
    Unset trailing elements are simply absent, as in a shorter list.

    A group that appears twice is refused: WRF rewinds and reads the
    FIRST ``&name`` for each group, so the second one's settings would be
    silently ignored by the model while a reader that merged them would
    run something else.
    """

    label = (f"{source}: " if source else "")
    sections: dict[str, dict[str, list]] = {}
    for group in scan_namelist_text(text, source=source):
        if group.name in sections:
            raise ValueError(
                f"{label}namelist line {_line(text, group.start)}: a second "
                f"&{group.name} group; WRF reads only the first "
                f"&{group.name}, so the settings in this one would be "
                "silently ignored.  Merge the two groups")
        table: dict[str, list] = {}
        for assignment in group.assignments:
            values = table.setdefault(assignment.name, [])
            where = f"{label}namelist line {_line(text, assignment.start)}"
            for position, value in zip(_positions(assignment, where),
                                       assignment.values):
                if value is None:
                    continue
                if len(values) < position:
                    values.extend([None] * (position - len(values)))
                values[position - 1] = value
        for name, values in table.items():
            while values and values[-1] is None:
                values.pop()
            unset = [index + 1 for index, value in enumerate(values)
                     if value is None]
            if unset and not allow_unset:
                raise ValueError(
                    f"{label}&{group.name}/{name} leaves element(s) "
                    f"{', '.join(map(str, unset))} unset in front of "
                    "element(s) it sets (an empty slot between commas, a "
                    "'!' comment straight after a comma or '=' with the "
                    "list going on below it, an 'N*' null, or an element "
                    "assignment past the end).  "
                    "WRF keeps its Registry default there, which this "
                    "reader does not carry; reading the list without the "
                    "gap would move every later value into the wrong "
                    "domain's column.  Write the value out")
        sections[group.name] = table
    return sections


def parse_namelist(path: str | Path, *, allow_unset: bool = False) -> dict[str, dict[str, list]]:
    """Parse a Fortran namelist FILE into ``{group: {name: [values]}}``."""

    # utf-8-sig: a byte-order mark in front of the first '&' is skipped by
    # the Fortran runtime's search for the group, and must be here too.
    return parse_namelist_text(
        Path(path).read_text(encoding="utf-8-sig"), allow_unset=allow_unset,
        source=str(path))


__all__ = [
    "NamelistAssignment",
    "NamelistGroup",
    "parse_namelist",
    "parse_namelist_text",
    "scan_namelist_text",
    "value_end_through_blanks",
]
