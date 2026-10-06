"""Name the page ``gpuwm gui`` serves the Weather Library, on whatever tree this runs on.

The page was called the storm wiki while it was built.  It is the Weather
Library everywhere a user sees it: its title and brand, its labels and
sentences, the CLI help, the docs, the unreleased CHANGELOG section and the
release notes.  Its main page's address moves from ``#/wiki`` to
``#/library`` and its API from ``/api/wiki`` to ``/api/library``; the old
addresses keep answering.  docs/dev/WEATHER-LIBRARY-RENAME.md lists every
item with its decision: renamed, kept, or redirected.

    python tools/dev/rename_weather_library.py                 rename this checkout
    python tools/dev/rename_weather_library.py --check         print what would change, write nothing
    python tools/dev/rename_weather_library.py --scan          list old names left where a user sees them
    python tools/dev/rename_weather_library.py --notes FILE    rename a release-notes file too
    python tools/dev/rename_weather_library.py --root DIR      another checkout

Three kinds of rule, applied in this order:

- ``EDITS`` are exact: one file, the text it holds now, the text it gets.
  An edit whose new text is already there is skipped, and so is one whose
  result's marks (``made``) are all there, so lines an edit inserted and
  someone changed since are never inserted twice.  One whose old text is
  there exactly ``count`` times is made.  One whose old and new text are
  both gone had its lines reworded: it counts as made when, after the
  other rules, the file holds every mark of its result and no old name
  (the scan below, and the edit's own ``stale`` patterns), and the run
  prints that it did.  Anything else refuses the file, and when any file
  refuses, nothing at all is written (exit 2).
- ``ROUTES`` rewrite the old page and API addresses in the page's scripts,
  the GUI package's strings, the docs and the unreleased CHANGELOG section.
- ``PROSE`` swaps the old names in words a user reads, and only there:
  copy and seed JSON values, string literals that are not docstrings,
  Markdown outside code, the HTML page's text.

The word rules never read docstrings or comments outside the page's
scripts; the route rule renames the old API paths in the GUI package's
docstrings, and one exact edit renames the main page row of the API's
endpoint table.  It never edits tests/, identifiers (modules, classes,
JSON keys, the assistant's tool names), the file formats and file names
saved runs and the page store use, or a released CHANGELOG section.
Every file it changes is run through all three rules again before
anything is written, and must come out the same, so a second run changes
nothing; and it must still parse: JSON, Python, and for a page script no
name declared twice at its top level, and ``node --check`` when Node is
installed.

Exit 0: done, or nothing to do.  1: ``--check`` found changes, or
``--scan`` found old names.  2: a file refused; nothing was written.
Standard library only.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
NEW = "Weather Library"
#: The decision table: it names the old names on purpose, so no rule reads it.
RENAME_DOC = "docs/dev/WEATHER-LIBRARY-RENAME.md"


# ---------------------------------------------------------------- the exact edits

@dataclass(frozen=True)
class Edit:
    """One exact change to one file.

    ``made`` are patterns that hold once the change is in the file, even after its lines were
    reworded: an edit that inserts lines keeps its old text inside its new text, so it names them,
    and a run that finds them all skips the edit instead of inserting a second copy.  ``stale`` are
    patterns that find the old name the edit removes, for the places the scan does not read.

    When neither the edit's old text nor its new text is in the file, the lines were reworded.  The
    edit then counts as made if, after every rule has run, every ``made`` pattern holds, no
    ``stale`` pattern matches, and the scan finds no old name in the file; otherwise the file is
    refused.
    """

    path: str
    old: str
    new: str
    why: str
    count: int = 1
    made: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()


#: The route strings of app.js: none may be the old route once the page is renamed.
_JS_OLD_ROUTE = r"""(["'`])wiki\1"""

EDITS: tuple[Edit, ...] = (
    # The page's own name: brand, tab title, the line a browser without scripts shows.
    Edit("gpuwm/gui/static/index.html", "<title>ArWen</title>", "<title>Weather Library</title>",
         "the tab title before the page's scripts run",
         stale=(r"<title>(?![^<]*Weather Library)[^<]*\bArWen\b",)),
    Edit("gpuwm/gui/static/index.html", "<p class=\"nojs\">ArWen's page needs JavaScript.",
         "<p class=\"nojs\">The Weather Library needs JavaScript.", "the line a browser without scripts shows",
         stale=(r"class=\"nojs\">(?![^<]*Weather Library)[^<]*\bArWen\b",)),
    Edit("gpuwm/gui/copy/screens.json", "\"name\": \"ArWen\",", "\"name\": \"Weather Library\",",
         "the brand in the header and every tab title",
         stale=(r"\"name\":\s*\"(?![^\"]*Weather Library)[^\"]*\bArWen\b",)),
    Edit("gpuwm/gui/copy/screens.json",
         "\"open_assistant_sub\": \"Ask about the wiki, a forecast or what fits this card\"",
         "\"open_assistant_sub\": \"Ask about a storm, a forecast or what fits this card\"",
         "the assistant's line under Ctrl K: the whole page is the Weather Library, so it names a storm"),
    Edit("gpuwm/gui/copy/assistant.json", "\"off_line\": \"It is optional: the wiki, your forecasts",
         "\"off_line\": \"It is optional: the storm pages, your forecasts",
         "the assistant's off line names the storm pages, not the page it sits in"),
    # The main page's route: #/library, and #/wiki forwards to it.
    Edit("gpuwm/gui/static/js/router.js",
         "export const RUN_PAGES = [\"run\", \"results\", \"explore\"];\n",
         "export const RUN_PAGES = [\"run\", \"results\", \"explore\"];\n"
         "\n"
         "// Routes that moved keep answering: an address whose first part is an old name forwards to the new\n"
         "// name, the rest of the address kept. The main page's route took the page's name, the Weather Library.\n"
         "export const MOVED_ROUTES = new Map([[\"wiki\", \"library\"]]);\n"
         "\n"
         "// The route an old address forwards to (wiki/x to library/x), or null when the address has not moved.\n"
         "export function forwardedRoute(raw) {\n"
         "  const route = String(raw || \"\");\n"
         "  const end = route.search(/[/?]/);\n"
         "  const name = end < 0 ? route : route.slice(0, end);\n"
         "  return MOVED_ROUTES.has(name) ? MOVED_ROUTES.get(name) + route.slice(name.length) : null;\n"
         "}\n",
         "the table of moved routes and the forward",
         made=(r"^export const MOVED_ROUTES\b", r"^export function forwardedRoute\b")),
    Edit("gpuwm/gui/static/js/app.js",
         "import { SCREENS, setNoticeBox, notice, errorText } from \"./router.js\";",
         "import { SCREENS, setNoticeBox, notice, errorText, forwardedRoute } from \"./router.js\";",
         "the shell reads the forward",
         made=(r"import\s*\{[^}]*\bforwardedRoute\b[^}]*\}\s*from\s*\"\./router\.js\"",)),
    Edit("gpuwm/gui/static/js/app.js",
         "  async function route() {\n    const ticket = ++routing;\n",
         "  async function route() {\n"
         "    // An old address forwards to its new route, and the address bar shows the new one.\n"
         "    const moved = forwardedRoute((location.hash || \"\").replace(/^#\\/?/, \"\"));\n"
         "    if (moved !== null) { location.replace(`#/${moved}`); return; }\n"
         "    const ticket = ++routing;\n",
         "an old address forwards before any screen draws",
         made=(r"\bforwardedRoute\(",)),
    Edit("gpuwm/gui/static/js/app.js", "return { name: name || \"wiki\", args",
         "return { name: name || \"library\", args", "an empty address opens the main page's route",
         stale=(_JS_OLD_ROUTE,)),
    Edit("gpuwm/gui/static/js/app.js", "const screen = SCREENS.has(name) ? name : \"wiki\";",
         "const screen = SCREENS.has(name) ? name : \"library\";", "an unknown address opens the main page",
         stale=(_JS_OLD_ROUTE,)),
    Edit("gpuwm/gui/static/js/app.js", "const f = frame(words, s, entry, raw || \"wiki\");",
         "const f = frame(words, s, entry, raw || \"library\");", "the sidebar lights the main page's link",
         stale=(_JS_OLD_ROUTE,)),
    Edit("gpuwm/gui/static/js/wikipages.js", "register(\"wiki\", mainPage, { title: wikiTitle });",
         "register(\"library\", mainPage, { title: wikiTitle });", "the main page registers under its route",
         stale=(r"""register\(\s*(["'`])wiki\1""",)),
    # The API: /api/library is answered by the handler named wiki, and /api/wiki still answers.
    Edit("gpuwm/gui/api.py",
         "    def _dispatch(self, method: str, raw_path: str, query: dict[str, list[str]], body: bytes,\n"
         "                  content_type: str = \"\") -> Reply:\n"
         "        segments = [unquote(part) for part in raw_path.split(\"/\") if part]\n",
         "    #: API sections whose path differs from the name their handler goes by: the Weather Library's\n"
         "    #: pages are served under library by the handler named wiki, and wiki, their path before the page\n"
         "    #: took that name, answers the same.\n"
         "    SECTIONS = {\"library\": \"wiki\"}\n"
         "\n"
         "    def _dispatch(self, method: str, raw_path: str, query: dict[str, list[str]], body: bytes,\n"
         "                  content_type: str = \"\") -> Reply:\n"
         "        segments = [unquote(part) for part in raw_path.split(\"/\") if part]\n"
         "        if len(segments) > 1 and segments[0] == \"api\":\n"
         "            segments[1] = self.SECTIONS.get(segments[1], segments[1])\n",
         "the API answers under its new path and its old one",
         made=(r"^\s*SECTIONS\s*=", r"\bself\.SECTIONS\.get\(")),
    Edit("gpuwm/gui/api.py", "the wiki's main page: kinds, places", "the Weather Library's main page: kinds, places",
         "the endpoint table's main page row (a docstring, which the word rules do not read)",
         stale=(r"\bthe wiki's main page\b",)),
    # The assistant opens the main page by its new route.
    Edit("gpuwm/gui/assistant/agent.py", "_tool(\"open_page\", \"Show a page: wiki, runs, create, machines;",
         "_tool(\"open_page\", \"Show a page: library, runs, create, machines;", "the assistant's page list"),
    Edit("gpuwm/gui/assistant/agent.py", "{\"page\": {\"type\": \"string\", \"enum\": [\"wiki\", \"runs\",",
         "{\"page\": {\"type\": \"string\", \"enum\": [\"library\", \"runs\",", "the page the assistant may open",
         stale=(r"\"page\":\s*\{[^}]*\"enum\":\s*\[[^\]]*\"wiki\"",)),
    # The terminal.
    Edit("gpuwm/gui/server.py", "\"ArWen is ready. Open this link in your browser:\",",
         "\"The Weather Library is ready. Open this link in your browser:\",", "the line gpuwm gui prints"),
    Edit("gpuwm/gui/server.py", "or close the other ArWen page server.\"",
         "or close the other Weather Library page server.\"", "the refusal when the port is taken"),
    Edit("gpuwm/gui/server.py", "\"Close a few ArWen tabs and reload.\"", "\"Close a few Weather Library tabs and reload.\"",
         "the fix when too many live views are open"),
    Edit("gpuwm/gui/assistant/cli.py", "(the same assistant as gpuwm gui's panel)",
         "(the same assistant as the Weather Library's panel)", "gpuwm assistant's help"),
    # Sentences a word swap would leave awkward.
    Edit("docs/public/GUI.md", "# ArWen in your browser: `gpuwm gui`", "# The Weather Library: `gpuwm gui`",
         "the guide's title"),
    Edit("docs/public/GUI.md", "The page opens on the storm wiki; **New forecast** is in the",
         "The page opens on its main page; **New forecast** is in the", "the first forecast's first step"),
    Edit("docs/public/GUI.md", "The sidebar has the storm wiki, your forecasts, **Machines**, the",
         "The sidebar has the **Weather Library** group of storm pages, your forecasts, **Machines**, the",
         "the sidebar's groups, the first named by its heading"),
    Edit("docs/public/GUI.md", "no card memory is used; the storm wiki,\n",
         "no card memory is used; the storm pages,\n", "what works without the assistant"),
    Edit("README.md",
         "2.8.0 adds ArWen in your browser, `gpuwm gui`, as a preview beside the desktop\n"
         "app: a storm wiki of 17 cited events with a best simulation for each card size,",
         "2.8.0 adds the Weather Library, `gpuwm gui`, as a preview beside the desktop\n"
         "app: 17 cited storm events with a best simulation for each card size,", "the release paragraph"),
    Edit("docs/dev/WIKI.md", "`/api/wiki/changes`, `/api/wiki/recipe/ID`. All read-only.",
         "`/api/library/changes`, `/api/library/recipe/ID`. All read-only. The same paths\n"
         "with `wiki` in place of `library` answer too: that was their name before the\n"
         "page was named the Weather Library.", "the endpoints and the old path that still answers"),
)

#: Edits a release-notes file gets when it holds their text; its wording is the cut's, so none refuses.
NOTES_EDITS: tuple[Edit, ...] = (
    Edit("", "a local page with the storm wiki, New forecast", "a local page with pages for past storms, New forecast",
         "the notes' line that introduces the page"),
)


# ---------------------------------------------------------------- word and route rules

@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    replace: Callable[[re.Match, str], str]


def _starts_sentence(segment: str, at: int) -> bool:
    before = segment[:at].rstrip()
    if not before or before.endswith((".", "!", "?", "#")):
        return True
    return before.split("\n")[-1].strip() in ("-", "*", "+")


def _the(match: re.Match, segment: str) -> str:
    return "The" if _starts_sentence(segment, match.start()) else "the"


def _article(match: re.Match) -> str:
    article = match.group("art")
    if not article:
        return ""
    return ("The" if article[0].isupper() else "the") + match.group("sp")


#: A word "wiki" that is a word and not part of a path, file name, route or identifier.
_WORD_BEFORE = r"(?<![\w/#.\-])"
_WORD_AFTER = r"(?![\w/\-]|\.\w)"

PROSE: tuple[Rule, ...] = (
    Rule("storm wiki", re.compile(r"(?:\b(?P<art>[Aa]|[Tt]he)(?P<sp>\s+))?" + _WORD_BEFORE
                                  + r"[Ss]torm[ \-][Ww]iki" + _WORD_AFTER),
         lambda m, s: _article(m) + NEW),
    Rule("the wiki", re.compile(r"\b(?P<art>[Tt]he)(?P<sp>\s+)[Ww]iki" + _WORD_AFTER),
         lambda m, s: m.group("art") + m.group("sp") + NEW),
    Rule("wiki page", re.compile(_WORD_BEFORE + r"[Ww]iki(?P<sp>\s+)(?P<noun>pages?|events?|articles?|documents?|layouts?)\b"),
         lambda m, s: NEW + m.group("sp") + m.group("noun")),
    Rule("ArWen in your browser", re.compile(r"\bArWen in your browser\b"),
         lambda m, s: f"{_the(m, s)} {NEW} in your browser"),
    Rule("ArWen's page", re.compile(r"\bArWen's page\b(?!\s+server)"), lambda m, s: f"{_the(m, s)} {NEW}"),
    Rule("web GUI", re.compile(r"(?:\b(?P<art>[Tt]he)(?P<sp>\s+))?\bweb\s+GUI\b"),
         lambda m, s: (m.group("art") + m.group("sp") if m.group("art") else "") + NEW),
)

_ROUTE_END = r"(?=[/\"'`?\s)\]]|$)"
ROUTES: tuple[Rule, ...] = (
    # a path in an aligned table keeps the table's columns
    Rule("/api/wiki in a table", re.compile(r"(/api/)wiki((?:[/?][^\s\"'`)\]]*)?)( {4,})"),
         lambda m, s: m.group(1) + "library" + m.group(2) + m.group(3)[3:]),
    Rule("/api/wiki", re.compile(r"/api/wiki" + _ROUTE_END), lambda m, s: "/api/library"),
    Rule("#/wiki", re.compile(r"#/wiki" + _ROUTE_END), lambda m, s: "#/library"),
)


def swap(segment: str, rules: Iterable[Rule]) -> tuple[str, list[tuple[int, str, str]]]:
    """Apply the rules to one piece of text, leftmost match first; returns the text and (offset, old, new)."""

    rules = tuple(rules)
    out: list[str] = []
    changes: list[tuple[int, str, str]] = []
    pos = 0
    while True:
        best: tuple[Rule, re.Match] | None = None
        for rule in rules:
            found = rule.pattern.search(segment, pos)
            if found and found.end() > found.start() and (best is None or found.start() < best[1].start()):
                best = (rule, found)
        if best is None:
            break
        rule, found = best
        new = rule.replace(found, segment)
        out.append(segment[pos:found.start()])
        out.append(new)
        if new != found.group(0):
            changes.append((found.start(), found.group(0), new))
        pos = found.end()
    out.append(segment[pos:])
    return "".join(out), changes


# ---------------------------------------------------------------- where a user reads words

Span = tuple[int, int]


def _gaps(lo: int, hi: int, taken: list[Span]) -> list[Span]:
    spans, at = [], lo
    for a, b in sorted(taken):
        if a > at:
            spans.append((at, a))
        at = max(at, b)
    if at < hi:
        spans.append((at, hi))
    return spans


def md_spans(text: str, lo: int = 0, hi: int | None = None) -> tuple[list[Span], list[Span]]:
    """(prose, code) of Markdown between lo and hi: code is fenced blocks and `inline` spans."""

    hi = len(text) if hi is None else hi
    code: list[Span] = []
    fence, opened, at = None, 0, lo
    for line in text[lo:hi].splitlines(keepends=True):
        stripped = line.lstrip()
        if fence is None and stripped.startswith(("```", "~~~")):
            fence, opened = stripped[:3], at
        elif fence is not None:
            if stripped.startswith(fence):
                code.append((opened, at + len(line)))
                fence = None
        else:
            code += [(at + m.start(), at + m.end()) for m in re.finditer(r"`[^`\n]*`", line)]
        at += len(line)
    if fence is not None:
        code.append((opened, hi))
    return _gaps(lo, hi, code), code


_JSON_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def json_value_spans(text: str) -> list[Span]:
    """The inside of every JSON string that is a value, never a key."""

    spans = []
    for m in _JSON_STRING.finditer(text):
        k = m.end()
        while k < len(text) and text[k] in " \t\r\n":
            k += 1
        if k < len(text) and text[k] == ":":
            continue
        spans.append((m.start() + 1, m.end() - 1))
    return spans


def py_string_spans(text: str, *, docstrings: bool) -> list[Span]:
    """The inside of every string literal of a Python module, its docstrings only when asked."""

    documented: set[tuple[int, int]] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                documented.add((first.value.lineno, first.value.col_offset))
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)

    def offset(position: tuple[int, int]) -> int:
        return starts[position[0] - 1] + position[1]

    middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    spans = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING:
            if not docstrings and token.start in documented:
                continue
            a, b = offset(token.start), offset(token.end)
            prefix = len(re.match(r"[A-Za-z]*", token.string).group(0))
            quote = 3 if token.string[prefix:prefix + 3] in ('"""', "'''") else 1
            spans.append((a + prefix + quote, b - quote))
        elif middle is not None and token.type == middle:
            spans.append((offset(token.start), offset(token.start) + len(token.string)))
    return spans


def js_string_spans(text: str) -> list[Span]:
    """The inside of every string and template literal of a script, never its comments or regex literals."""

    spans: list[Span] = []
    i, n, previous = 0, len(text), ""
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if c in "\"'`":
            j = start = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 2
                    continue
                if c != "`" and text[j] == "\n":
                    break
                if c == "`" and text.startswith("${", j):
                    # a template's ${...} is code, not words; the text on either side of it is words
                    spans.append((start, j))
                    j, depth = j + 2, 1
                    while j < n and depth:
                        depth += {"{": 1, "}": -1}.get(text[j], 0)
                        j += 1
                    start = j
                    continue
                j += 1
            spans.append((start, min(j, n)))
            i, previous = j + 1, c
            continue
        if c == "/" and (previous == "" or previous in "(,=:[!&|?{};+-*%<>~^"):
            j, in_class = i + 1, False
            while j < n and text[j] != "\n":
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "[":
                    in_class = True
                elif text[j] == "]":
                    in_class = False
                elif text[j] == "/" and not in_class:
                    break
                j += 1
            i, previous = j + 1, "/"
            continue
        if not c.isspace():
            previous = c
        i += 1
    return spans


def html_text_spans(text: str) -> list[Span]:
    return _gaps(0, len(text), [(m.start(), m.end()) for m in re.finditer(r"<[^>]*>", text)])


def changelog_region(text: str, version: str | None) -> tuple[int, int] | None:
    """The unreleased section: the one headed by the version this tree is making, if there is one."""

    if not version:
        return None
    head = re.search(rf"^## {re.escape(version)}\b.*$", text, re.M)
    if head is None or not re.search(r"\bunreleased\b", head.group(0), re.I):
        return None
    following = re.search(r"^## ", text[head.end():], re.M)
    return head.start(), head.end() + following.start() if following else len(text)


def tree_version(root: Path) -> str | None:
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    found = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return found.group(1) if found else None


# ---------------------------------------------------------------- which file gets which rules

@dataclass(frozen=True)
class Scope:
    kind: str      # md | changelog | json | py | py-gui | js | html
    routes: bool
    prose: bool


def scope_of(rel: str) -> Scope | None:
    if rel.startswith("tests/") or rel.startswith("tools/dev/") or rel == RENAME_DOC:
        return None
    if rel == "CHANGELOG.md":
        return Scope("changelog", True, True)
    if rel in ("README.md", "NOTICE") or re.fullmatch(r"docs/(public|dev)/[^/]+\.md", rel):
        return Scope("md", True, True)
    if re.fullmatch(r"gpuwm/gui/(copy|seed|seed/recipes)/[^/]+\.json", rel):
        return Scope("json", True, True)
    if rel.startswith("gpuwm/gui/static/") and rel.endswith(".js"):
        return Scope("js", True, True)
    if rel.startswith("gpuwm/gui/static/") and rel.endswith(".html"):
        return Scope("html", False, False)
    if rel.startswith("gpuwm/gui/") and rel.endswith(".py"):
        return Scope("py-gui", True, True)
    if (rel.startswith("gpuwm/") and rel.endswith(".py")) or re.fullmatch(r"tools/wiki_seed/[^/]+\.py", rel):
        return Scope("py", False, True)
    return None


def candidate_files(root: Path) -> list[str]:
    found = {"CHANGELOG.md", "README.md", "NOTICE"}
    for pattern in ("docs/public/*.md", "docs/dev/*.md", "gpuwm/gui/copy/*.json", "gpuwm/gui/seed/*.json",
                    "gpuwm/gui/seed/recipes/*.json", "gpuwm/gui/static/**/*.js", "gpuwm/gui/static/**/*.html",
                    "gpuwm/**/*.py", "tools/wiki_seed/*.py"):
        found.update(p.relative_to(root).as_posix() for p in root.glob(pattern) if p.is_file())
    found.update(edit.path for edit in EDITS)
    return sorted(rel for rel in found if (root / rel).is_file())


def route_spans(text: str, scope: Scope, version: str | None) -> list[Span]:
    if scope.kind in ("md", "notes"):
        return [(0, len(text))]
    if scope.kind == "changelog":
        region = changelog_region(text, version)
        return [region] if region else []
    if scope.kind == "json":
        return json_value_spans(text)
    if scope.kind == "js":
        return [(0, len(text))]
    if scope.kind == "py-gui":
        return py_string_spans(text, docstrings=True)
    return []


def prose_spans(text: str, scope: Scope, version: str | None) -> list[Span]:
    if scope.kind in ("md", "notes"):
        return md_spans(text)[0]
    if scope.kind == "changelog":
        region = changelog_region(text, version)
        return md_spans(text, *region)[0] if region else []
    if scope.kind == "json":
        return json_value_spans(text)
    if scope.kind == "js":
        return js_string_spans(text)
    if scope.kind in ("py", "py-gui"):
        return py_string_spans(text, docstrings=False)
    if scope.kind == "html":
        return html_text_spans(text)
    return []


# ---------------------------------------------------------------- applying

@dataclass
class Outcome:
    rel: str
    path: Path
    before: str
    after: str
    lines: list[str]
    refusals: list[str]
    held: list[str] = field(default_factory=list)


@dataclass
class Renamed:
    """One file's text after the rules: the changes made, the refusals, and the edits counted as made."""

    text: str
    lines: list[str]
    refusals: list[str]
    held: list[str]


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _short(text: str) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= 70 else text[:67] + "..."


def _holding(patterns: Iterable[str], text: str) -> list[bool]:
    return [re.search(pattern, text, re.M) is not None for pattern in patterns]


def apply_edits(text: str, edits: Iterable[Edit], *,
                strict: bool) -> tuple[str, list[str], list[str], list[Edit]]:
    """The exact edits: returns the text, the changes, the refusals, and the edits whose lines were reworded."""

    lines, refusals, reworded = [], [], []
    for edit in edits:
        if edit.new in text:
            continue
        marks = _holding(edit.made, text)
        if marks and all(marks):
            continue  # made before and changed since; its old text may still sit inside it
        if any(marks):
            if strict:
                missing = [p for p, held in zip(edit.made, marks) if not held]
                refusals.append(f"edit ({edit.why}): part of its result is here and part is not ({', '.join(missing)})")
            continue
        found = text.count(edit.old)
        if found == edit.count:
            at = text.find(edit.old)
            text = text.replace(edit.old, edit.new)
            lines.append(f"  L{_line(text, at)}  edit: {edit.why}")
        elif found == 0:
            reworded.append(edit)
        elif strict:
            refusals.append(f"edit ({edit.why}): its text is here {found} times, the rule expects {edit.count}:"
                            f"\n      {_short(edit.old)}")
    return text, lines, refusals, reworded


def _old_name_left(edit: Edit, rel: str, text: str, scope: Scope, version: str | None) -> list[str]:
    """Why a reworded edit's file still holds the old name, or [] when it counts as made."""

    left = [f"its result's mark is missing ({p})" for p, held in zip(edit.made, _holding(edit.made, text))
            if not held]
    for pattern in edit.stale:
        for hit in re.finditer(pattern, text, re.M):
            left.append(f"line {_line(text, hit.start())} still names the old page ({_short(hit.group(0))!r})")
    left += [f"line {f.line} still holds {f.text!r} ({f.why})" for f in scan_text(rel, text, scope, version)]
    return left


def apply_rules(text: str, spans: list[Span], rules: Iterable[Rule]) -> tuple[str, list[str]]:
    rules = tuple(rules)
    out, lines, at = [], [], 0
    for a, b in spans:
        new, changes = swap(text[a:b], rules)
        out.append(text[at:a])
        out.append(new)
        lines += [f"  L{_line(text, a + offset)}  {_short(old)!r} -> {_short(rep)!r}" for offset, old, rep in changes]
        at = b
    out.append(text[at:])
    return "".join(out), lines


def rename_text(text: str, scope: Scope, edits: Iterable[Edit], version: str | None, *,
                strict: bool = True, rel: str = "") -> Renamed:
    """One file's text through the edits, then its routes, then its words.

    An edit whose old text and new text are both gone had its lines reworded since: it counts as
    made when, after the routes and words, the file holds no old name and every mark of its result;
    otherwise it refuses the file.
    """

    text, lines, refusals, reworded = apply_edits(text, edits, strict=strict)
    if scope.routes:
        text, more = apply_rules(text, route_spans(text, scope, version), ROUTES)
        lines += more
    if scope.prose:
        text, more = apply_rules(text, prose_spans(text, scope, version), PROSE)
        lines += more
    held = []
    for edit in reworded if strict else ():
        left = _old_name_left(edit, rel or edit.path, text, scope, version)
        if left:
            refusals.append(f"edit ({edit.why}): neither its text nor its result is here, and "
                            + "; ".join(left) + f":\n      {_short(edit.old)}")
        else:
            held.append(f"edit ({edit.why}): its lines were reworded and hold no old name, so it counts as made")
    return Renamed(text, lines, refusals, held)


#: A declaration at a module script's top level (no indentation), by name.
_JS_TOP_DECLARATION = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)", re.M)


def _js_problem(text: str) -> str | None:
    """Why a page script would not load: a name declared twice at its top level, or node's own check."""

    seen: set[str] = set()
    for found in _JS_TOP_DECLARATION.finditer(text):
        if found.group(1) in seen:
            return f"the renamed script declares {found.group(1)} twice (line {_line(text, found.start())})"
        seen.add(found.group(1))
    node = shutil.which("node")
    if node is None:
        return None
    with tempfile.TemporaryDirectory() as folder:
        script = Path(folder) / "renamed.mjs"
        script.write_bytes(text.encode("utf-8"))
        try:
            done = subprocess.run([node, "--check", str(script)], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
    if done.returncode != 0:
        first = next((line for line in done.stderr.splitlines() if "Error" in line), done.stderr.strip()[:200])
        return f"the renamed script does not parse ({first})"
    return None


def _valid(rel: str, text: str) -> str | None:
    try:
        if rel.endswith(".json"):
            json.loads(text)
        elif rel.endswith(".py"):
            ast.parse(text)
        elif rel.endswith(".js"):
            return _js_problem(text)
    except (ValueError, SyntaxError) as error:
        return f"the renamed file does not parse ({error})"
    return None


def plan(root: Path, notes: Iterable[Path] = ()) -> list[Outcome]:
    """Every file's outcome, nothing written."""

    version = tree_version(root)
    outcomes = []
    targets: list[tuple[str, Path, Scope, tuple[Edit, ...], bool]] = []
    for rel in candidate_files(root):
        scope = scope_of(rel) or Scope("none", False, False)
        targets.append((rel, root / rel, scope, tuple(e for e in EDITS if e.path == rel), True))
    for path in notes:
        targets.append((str(path), Path(path), Scope("notes", True, True), NOTES_EDITS, False))
    for rel, path, scope, edits, strict in targets:
        try:
            before = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            if edits:
                outcomes.append(Outcome(rel, path, "", "", [], [f"cannot read it ({error})"]))
            continue
        try:
            done = rename_text(before, scope, edits, version, strict=strict, rel=rel)
        except (SyntaxError, tokenize.TokenError) as error:
            if edits:
                outcomes.append(Outcome(rel, path, before, before, [], [f"cannot read its strings ({error})"]))
            continue
        after, refusals = done.text, done.refusals
        if after != before and not refusals:
            problem = _valid(rel, after)
            again = rename_text(after, scope, edits, version, strict=strict, rel=rel)
            if problem:
                refusals.append(problem)
            elif again.text != after or again.lines:
                refusals.append("a second pass would change it again; a rule's result matches another rule")
        if after != before or refusals or done.held:
            outcomes.append(Outcome(rel, path, before, after, done.lines, refusals, done.held))
    return outcomes


# ---------------------------------------------------------------- what may stay

#: Old-name tokens that stay where a user can read them, each with its reason. Anything else that
#: holds "wiki" in a user-facing place, or an old address, is a finding.
KEPT: tuple[tuple[str, str], ...] = (
    (r"wiki-run\.json", "the file a run started from an event page carries; saved runs are read by that name"),
    (r"gpuwm\.wiki\.v1", "the page store's file format, which the atlas and saved documents are written in"),
    (r"(?:^|[/<>])wiki/", "the page store's folder under the forecasts folder, where the atlas writes"),
    (r"wiki-seed\.json", "the package's seed file"),
    (r"wiki_seed\b", "the seed builder's folder, tools/wiki_seed"),
    (r"gpuwm-wiki-seed", "the seed builder's HTTP user agent"),
    (r"search_wiki", "the assistant's tool name, which the model calls"),
    (r"wiki(?:pages|kit|map)\.js", "the page's script modules, loaded by these names"),
    (r"wikipedia\.org", "a link to Wikipedia, not the page"),
)
_TOKEN_CHARS = re.compile(r"[\w./<>*#:@\-]")
_OLD_PHRASES = re.compile(r"(?i)\bstorm[ \-]wiki\b|\bArWen in your browser\b|\bArWen's page\b|\bArWen page server\b"
                          r"|\bArWen is ready\b|\bArWen tabs?\b|\bweb\s+GUI\b")
_OLD_ROUTE = re.compile(r"/api/wiki" + _ROUTE_END + r"|#/wiki" + _ROUTE_END)


@dataclass(frozen=True)
class Finding:
    rel: str
    line: int
    text: str
    why: str

    def __str__(self) -> str:
        return f"{self.rel}:{self.line}: {self.why}: {self.text!r}"


def _token_at(text: str, a: int, b: int) -> str:
    while a > 0 and _TOKEN_CHARS.match(text[a - 1]):
        a -= 1
    while b < len(text) and _TOKEN_CHARS.match(text[b]):
        b += 1
    return text[a:b].rstrip(".:,;")


def _word_hits(text: str, spans: list[Span], *, code: bool) -> Iterable[tuple[int, str, str]]:
    for a, b in spans:
        piece = text[a:b]
        if code and piece.isidentifier():
            continue  # a name the code uses (a handler, a route table's old name, __all__), not words
        for hit in re.finditer(r"(?i)wiki", piece):
            start = a + hit.start()
            before = text[start - 1] if start > a else ""
            after = text[start + 4:start + 6] if start + 4 < b else ""
            token = _token_at(text, start, start + 4)
            is_word = not (re.match(r"[\w/#.\-]", before) or re.match(r"[\w/\-]|\.\w", after))
            if is_word:
                yield start, token, "the old name"
            elif not any(re.search(pattern, token) for pattern, _ in KEPT):
                yield start, token, "an old name in a name or path the allow-list does not keep"


def scan_text(rel: str, text: str, scope: Scope, version: str | None) -> list[Finding]:
    findings = []
    words = prose_spans(text, scope, version)
    if scope.kind == "md" or scope.kind == "notes" or scope.kind == "changelog":
        everything = route_spans(text, scope, version)
    else:
        everything = words
    code = scope.kind in ("py", "py-gui", "js")
    for start, token, why in _word_hits(text, words, code=code):
        findings.append(Finding(rel, _line(text, start), token, why))
    for a, b in words:
        for hit in _OLD_PHRASES.finditer(text, a, b):
            findings.append(Finding(rel, _line(text, hit.start()), hit.group(0), "the old name"))
    for a, b in everything:
        for hit in _OLD_ROUTE.finditer(text, a, b):
            findings.append(Finding(rel, _line(text, hit.start()), hit.group(0), "an old address"))
    if scope.kind == "js":
        for hit in _OLD_ROUTE.finditer(text):
            findings.append(Finding(rel, _line(text, hit.start()), hit.group(0), "an old address"))
    return sorted(set(findings), key=lambda f: (f.rel, f.line, f.text))


def scan(root: Path = ROOT, notes: Iterable[Path] = ()) -> list[Finding]:
    """Every old name or address left where a user reads it, outside the allow-list."""

    version = tree_version(root)
    findings: list[Finding] = []
    targets = [(rel, root / rel, scope_of(rel)) for rel in candidate_files(root)]
    targets += [(str(path), Path(path), Scope("notes", True, True)) for path in notes]
    for rel, path, scope in targets:
        if scope is None:
            continue
        try:
            text = path.read_bytes().decode("utf-8")
            findings += scan_text(rel, text, scope, version)
        except (OSError, UnicodeDecodeError, SyntaxError, tokenize.TokenError):
            continue
    return findings


# ---------------------------------------------------------------- the command

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Name gpuwm gui's page the Weather Library in this tree.")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout to rename (default: this one)")
    parser.add_argument("--check", action="store_true", help="print what would change and write nothing")
    parser.add_argument("--scan", action="store_true", help="list old names left where a user reads them")
    parser.add_argument("--notes", type=Path, action="append", default=[],
                        help="a release-notes file to rename as well (repeatable)")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()

    if arguments.scan:
        findings = scan(root, arguments.notes)
        for finding in findings:
            print(finding)
        print(f"{len(findings)} old names left." if findings else "No old name left where a user reads it.")
        return 1 if findings else 0

    outcomes = plan(root, arguments.notes)
    refused = [o for o in outcomes if o.refusals]
    if refused:
        for outcome in refused:
            for refusal in outcome.refusals:
                print(f"REFUSED {outcome.rel}: {refusal}")
        print("Nothing was written. Rewrite the rule for the file's text as it is now, then run this again.")
        return 2
    for outcome in outcomes:
        for held in outcome.held:
            print(f"{outcome.rel}: {held}")
    changed = [o for o in outcomes if o.after != o.before]
    for outcome in changed:
        print(outcome.rel)
        for line in outcome.lines:
            print(line)
    count = sum(len(o.lines) for o in changed)
    if not changed:
        print("Nothing to change: the tree already names the page the Weather Library.")
        return 0
    if arguments.check:
        print(f"Would change {len(changed)} files, {count} places.")
        return 1
    for outcome in changed:
        outcome.path.write_bytes(outcome.after.encode("utf-8"))
    print(f"Changed {len(changed)} files, {count} places.")
    left = scan(root, arguments.notes)
    for finding in left:
        print(f"LEFT {finding}")
    if left:
        print(f"{len(left)} old names are left that no rule covers: add a rule or a kept entry, then run again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
